//! The direct model's MoE block (all 48 layers): the router, the routed experts and the shared expert through
//! turboexl3. The K path of today's `trunk/moe.zig` (`moeMLP2WithRouter`'s EXL3 branch, the shared expert's chain),
//! `trunk/fusion.zig`'s shared-expert decisions, `side_stream.zig`'s `addGatedShared` and
//! `sdk/blocks/moe/router_kernel.zig`'s softmax router, launching what they launch in their order:
//! - prefill rows: the router kernel, tx3's routed call, the shared expert beside it on the side stream (from
//!   `SideStream.MIN_ROWS`) or after it;
//! - 2..4 rows: the router's top-k inside tx3's routed prepare launch (the shared gate's logit too, where its gemv order
//!   serves), the shared expert inside the routed launches;
//! - one row and 5..16 rows: the router kernel (one row: the shared gate's dot beside it), tx3's routed call, then the
//!   shared expert.
//! Today's other arms (affine, FP4, gather, CUDA and sorted NAX banks, the routing chain and its compiled closure, the
//! joined shared projections) serve other packs, and are refused by name.

const std = @import("std");
const abi = @import("../mlx_serve/abi/quant.zig");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
/// The gemv gate's two texts, turboexl3's (the same bytes its own gate kernels splice).
const gate_texts = @import("turboexl3").sources;
const fusion = @import("../mlx_serve/plugins/qwen4_exp/trunk/fusion.zig");
const fused_activations = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const router_kernel = @import("../mlx_serve/sdk/blocks/moe/router_kernel.zig");
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const Formats = @import("../mlx_serve/plugins/qwen4_exp/trunk/formats.zig").Formats;
const SideStream = @import("../mlx_serve/plugins/qwen4_exp/side_stream.zig").SideStream;
const MoeMlpWeights = @import("../mlx_serve/sdk/blocks/moe/weights.zig").MoeMlpWeights;
const MoeRouting = @import("../mlx_serve/sdk/blocks/moe/routing.zig").MoeRouting;
const RouterGate = router_kernel.RouterGate;
const ShapeKey = router_kernel.ShapeKey;
const ROUTER_MAX_EXPERTS = router_kernel.ROUTER_MAX_EXPERTS;
const linear = @import("linear.zig");

/// The block's output for `x` (the hc read's mixed input): today's `moeMLP`, `moeMLP2WithRouter(x, x, mw, null,
/// false)`, whose K route is its EXL3 branch.
pub fn mlp(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    try kBlock(self, mw);
    const cfg = &self.config;
    // Today's per-bank quant params (`quantParamsHinted` on the switch banks) feed only the other formats' arms; on K
    // they would only fill the bits cache under the banks' keys, which nothing reads.
    const router_logits = try routerLogits(self, x, mw);
    defer _ = mlx.mlx_array_free(router_logits);

    // The routed shared expert: at 2..4 rows the router's top-k rides the fused shared expert's routed prepare launch
    // (`routedShared`); null: the chain below.
    if (try routedShared(self, x, mw, router_logits)) |y| return y;
    // Where the routed shared expert did not run: the router kernel sums the shared gate's logit in MLX's gemv order
    // for the fused shared expert (`gateRouting`), whose gemv launch then goes.
    const gate_routing = try gateRouting(self, x, mw, router_logits);
    // Top-K + softmax/renormalize as one kernel; at one decode row the shared-expert gate's logit rides it
    // (`softmax_gate`).
    const routed = gate_routing orelse try computeRouting(self, router_logits, gateFold(x, mw, router_logits));
    defer if (routed.gate_logit.ctx != null) {
        _ = mlx.mlx_array_free(routed.gate_logit);
    };
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);
    try scoresInDtypeOf(routed, x);

    // The banks the pack's expert format loaded run as one SwiGLU through it (`trunk/formats.zig`).
    const banks = self.formats.banksOf(mw) orelse return error.Exl3BanksNotLoaded;
    const clamp = swigluClampLimit(cfg.swiglu_limit, cfg.swiglu_clamp) orelse return error.Exl3SwigluLimitUnsupported;
    // The fused shared expert: the shared expert inside the routed launches (`sharedFused`); null: the SwiGLU and the
    // shared expert below.
    if (try sharedFused(self, x, mw, banks, routed.inds, routed.norm_scores, if (gate_routing != null) routed.gate_logit else null)) |y| return y;
    const y = try self.formats.swiglu(banks, x, routed.inds, routed.norm_scores, .{ .limit = @intCast(clamp), .verify_rows = false }, self.s);
    defer _ = mlx.mlx_array_free(y);
    return addGatedShared(self, y, x, mw);
}

/// The K packs' MoE block, or a named error: EXL3 banks (today's arms for the others are not transcribed), the plain
/// softmax router (no Hy3 expert bias, no Gemma 4 or NVFP4 scales), the gated shared expert.
fn kBlock(self: *const Trunk, mw: *const MoeMlpWeights) !void {
    if (self.config.exl3 == null) return error.DirectModelNeedsExl3Banks;
    if (mw.expert_bias != null or mw.router_scale != null or mw.per_expert_scale != null or mw.gate_expert_scale.ctx != null) return error.DirectModelRouterHasBiasOrScale;
    if (mw.shared_ungated or mw.shared_expert_gate_w == null) return error.DirectModelNeedsGatedSharedExpert;
}

/// The router's logits [.., E] (today's `moeRouterLogits`): the router is bf16 on every K pack, so one matmul.
fn routerLogits(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    return linear.dense(x, mw.router_w, mw.router_s, self.s);
}

/// The top-k experts and their renormalized softmax weights (today's `computeMoeRouting`). The router kernel serves
/// every K width (bf16 [.., 512] logits, top-10), so the compiled closure and the op chain behind it
/// (`moeRoutingUncompiled`) are not transcribed: a decline is refused.
fn computeRouting(self: *const Trunk, router_logits: mlx.mlx_array, gate: ?RouterGate) !MoeRouting {
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    return (try routerTopK(self.s, router_logits, k, .softmax, mlx.mlx_array_dtype(router_logits), gate)) orelse error.DirectModelRouterDeclined;
}

/// The routing weights are in x's dtype. The router kernel writes them in the logits' dtype, x's on K (bf16
/// activations over a bf16 router table), so today's cast to x's dtype (for the other routers' f32 weights) never runs.
fn scoresInDtypeOf(routed: MoeRouting, x: mlx.mlx_array) !void {
    if (mlx.mlx_array_dtype(routed.norm_scores) != mlx.mlx_array_dtype(x)) return error.DirectModelRouterWeightsNotInXDtype;
}

/// At one row, the shared gate's weight vector for the softmax router kernel's dot (`softmax_gate`); else null.
fn gateFold(x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) ?RouterGate {
    if (mw.shared_gate_vec.ctx == null) return null;
    const lsh = mlx.getShape(router_logits);
    var rows: usize = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= @intCast(d);
    if (rows != 1 or mlx.mlx_array_size(x) != mlx.mlx_array_size(mw.shared_gate_vec)) return null;
    return .{ .x = x, .w = mw.shared_gate_vec };
}

/// The SwiGLU's clamp: 0 unclamped, the whole-number limit otherwise; null for a limit the kernels cannot bake in
/// (today's `swigluClampLimit`).
fn swigluClampLimit(swiglu_limit: f32, swiglu_clamp: bool) ?u32 {
    if (swiglu_limit <= 0.0) return 0;
    if (!swiglu_clamp or @round(swiglu_limit) != swiglu_limit or swiglu_limit > 1 << 16) return null;
    return @intFromFloat(swiglu_limit);
}

// ── the shared expert ──

/// The shared expert of `x`: gate and up, the SwiGLU, down (today's `moeSharedDown(.qmatmul, ...)`).
fn sharedDown(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const gate = try linear.qmatmul(self, x, mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b);
    defer _ = mlx.mlx_array_free(gate);
    const up = try linear.qmatmul(self, x, mw.shared_up_w, mw.shared_up_s, mw.shared_up_b);
    defer _ = mlx.mlx_array_free(up);
    const act = try linear.computeGeglu(self, gate, up);
    defer _ = mlx.mlx_array_free(act);
    return linear.qmatmul(self, act, mw.shared_down_w, mw.shared_down_s, mw.shared_down_b);
}

/// The shared gate's logit (today's `moeAddSharedDown`, `fusion.sharedGateLogit` and `addGatedShared` each compute it
/// so).
fn sharedGateLogit(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const seg_w = mw.shared_expert_gate_w.?;
    return linear.dense(x, seg_w, mw.shared_expert_gate_s.?, self.s);
}

/// `x`'s shared expert times its gate's sigmoid, on the trunk's stream: the shared expert, the gate's logit, MLX's
/// sigmoid and multiply.
fn sharedGated(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const down = try sharedDown(self, x, mw);
    defer _ = mlx.mlx_array_free(down);
    const gate_logit = try sharedGateLogit(self, x, mw);
    defer _ = mlx.mlx_array_free(gate_logit);
    var gate_sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_sig);
    try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate_logit, self.s));
    var shared_gated = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shared_gated);
    try mlx.check(mlx.mlx_multiply(&shared_gated, gate_sig, down, self.s));
    return shared_gated;
}

/// `sum` plus `x`'s gated shared expert (today's `moeAddGatedShared`: `moeSharedDown`, the gate's logit, then
/// `moeSharedGateTail`).
fn addShared(self: *Trunk, sum: mlx.mlx_array, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const shared_gated = try sharedGated(self, x, mw);
    defer _ = mlx.mlx_array_free(shared_gated);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_add(&result, sum, shared_gated, self.s));
    return result;
}

/// `addShared` where the side stream's `shared` part takes the call (today's `side_stream.addGatedShared`): the shared
/// expert, its gate's logit, sigmoid and product on the side stream beside the routed launches, the add that joins the
/// routed sum on the trunk's stream (one fence back). Its ops on its operands, so its bytes.
fn addGatedShared(self: *Trunk, sum: mlx.mlx_array, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    const rows = SideStream.rowsOf(x);
    const side = SideStream.forPart(.shared, self, rows) orelse return addShared(self, sum, x, mw);
    const main = self.s;
    const gated = blk: {
        self.s = side;
        defer self.s = main;
        break :blk try sharedGated(self, x, mw);
    };
    defer _ = mlx.mlx_array_free(gated);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, sum, gated, main));
    SideStream.note("shared", rows);
    return out;
}

// ── the shared expert inside the routed launches (`trunk/fusion.zig`'s decisions; its entries stay there) ──

/// The expert format, where it also loaded the shared expert's linears (the trunk's format) and has the fused entry.
fn sharedTable(f: *const Formats) ?*const abi.QuantTable {
    const t = f.experts orelse return null;
    if (f.trunk != t or t.swiglu_shared == null) return null;
    return t;
}

fn sharedLinears(f: *const Formats, mw: *const MoeMlpWeights) ?[3]*const abi.Weight {
    return .{ f.linearOf(mw.shared_gate_w) orelse return null, f.linearOf(mw.shared_up_w) orelse return null, f.linearOf(mw.shared_down_w) orelse return null };
}

fn rowsOf(x: mlx.mlx_array) c_int {
    const sh = mlx.getShape(x);
    var rows: c_int = 1;
    if (sh.len > 1) for (sh[0 .. sh.len - 1]) |d| {
        rows *= d;
    };
    return rows;
}

/// The fused shared expert's rows. One row keeps the chain (`swiglu`, then the shared expert): its shared gate and up
/// read only x and so run beside the router, where the fused launches' shared heads wait for the router's slots. The
/// same bytes.
fn sharedRows(rows: c_int) bool {
    return rows >= 2 and rows <= 4;
}

/// The shared expert's chain the fused launches reproduce: a gated shared expert (`kBlock`) on the trunk format's
/// linears, the table SwiGLU (silu, unclamped), bf16 activations, 2..4 rows (`sharedRows`).
fn sharedServes(self: *const Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights) bool {
    const rows = rowsOf(x);
    if (!sharedRows(rows) or mlx.mlx_array_dtype(x) != .bfloat16 or !fused_activations.swigluFusedEnabled()) return false;
    const clamp = swigluClampLimit(self.config.swiglu_limit, self.config.swiglu_clamp) orelse return false;
    return clamp == 0 and sharedTable(&self.formats) != null and sharedLinears(&self.formats, mw) != null;
}

/// The fused shared expert: the routed experts and the gated shared expert in one launch chain at 2..4 rows, byte for
/// byte `swiglu` + `addGatedShared` (the shared linears, the table SwiGLU, the gate's logit through `linear.dense`, MLX's
/// sigmoid, multiply and add). `given_logit`: the gate's logit the router kernel computed in MLX's gemv order
/// (`gateRouting`). Null: that chain.
fn sharedFused(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights, banks: *const [3]abi.Weight, inds: mlx.mlx_array, scores: mlx.mlx_array, given_logit: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!sharedServes(self, x, mw)) return null;
    const sh = sharedLinears(&self.formats, mw).?;
    const logit = given_logit orelse try sharedGateLogit(self, x, mw);
    defer if (given_logit == null) {
        _ = mlx.mlx_array_free(logit);
    };
    const sigtab = try fused_activations.swigluSigTable(self.s, .bfloat16, std.heap.c_allocator);
    const opts: abi.SharedSwiglu = .{ .gate = sh[0], .up = sh[1], .down = sh[2], .sigtab = fusion.arr(sigtab), .logit = fusion.arr(logit) };
    const y = fusion.callSharedSwiglu(self, banks, x, inds, scores, &opts, false) orelse return null;
    fusion.served(.swiglu_shared, "the shared expert inside the routed launches (rows={d}{s})", .{ rowsOf(x), if (given_logit != null) ", the gate's logit from the router" else "" });
    return y;
}

/// Where the router is folded into the fused shared expert (2..4 rows on an EXL3 trunk pack): the path to `sharedFused`
/// (the fused softmax router), with the router's logits in x's dtype. `kBlock` refused an expert bias or per-expert
/// scale; the router kernel is on, and no stand-in or dispatch probe runs, on every K call.
fn routedServes(self: *const Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) bool {
    const rows = rowsOf(x);
    if (!self.config.trunk_format or rows < 2 or rows > 4) return false;
    if (!sharedServes(self, x, mw) or mlx.mlx_array_dtype(router_logits) != mlx.mlx_array_dtype(x)) return false;
    const lsh = mlx.getShape(router_logits);
    const k = self.config.num_experts_per_tok;
    return lsh.len >= 1 and k >= 1 and k <= 31 and lsh[lsh.len - 1] >= @as(c_int, @intCast(k)) and lsh[lsh.len - 1] <= ROUTER_MAX_EXPERTS;
}

/// The shared gate's weight vector where the router kernel sums its logit in MLX's gemv order (`gateGemvShape`), on the
/// folded path; else null.
fn gateVec(self: *const Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) ?mlx.mlx_array {
    const v = mw.shared_gate_vec;
    if (v.ctx == null or !routedServes(self, x, mw, router_logits)) return null;
    const rows = rowsOf(x);
    if (mlx.mlx_array_size(v) * @as(usize, @intCast(rows)) != mlx.mlx_array_size(x)) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (!gateGemvShape(rows, @intCast(mlx.mlx_array_size(v)), xd, mlx.mlx_array_dtype(v), xd)) return null;
    return v;
}

/// The routed shared expert: the call the router kernel + `sharedFused` make at 2..4 rows, with the router's top-k and
/// weights inside the fused shared expert's routed prepare launch (`SharedSwiglu.logits`), and the gate's logit in that
/// launch too where its order is MLX's gemv's (`gateRouting`): one dispatch and one dependent level fewer, the same
/// bytes. Null: the chain.
fn routedShared(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) !?mlx.mlx_array {
    if (!routedServes(self, x, mw, router_logits)) return null;
    const banks = self.formats.banksOf(mw) orelse return null;
    const sh = sharedLinears(&self.formats, mw).?;
    const gvec = gateVec(self, x, mw, router_logits);
    const logit: ?mlx.mlx_array = if (gvec == null) try sharedGateLogit(self, x, mw) else null;
    defer if (logit) |l| {
        _ = mlx.mlx_array_free(l);
    };
    const sigtab = try fused_activations.swigluSigTable(self.s, .bfloat16, std.heap.c_allocator);
    const opts: abi.SharedSwiglu = .{
        .gate = sh[0],
        .up = sh[1],
        .down = sh[2],
        .sigtab = fusion.arr(sigtab),
        .logit = if (logit) |l| fusion.arr(l) else .{},
        .gate_vec = if (gvec) |g| fusion.arr(g) else .{},
        .logits = fusion.arr(router_logits),
        .k = self.config.num_experts_per_tok,
    };
    return fusion.callSharedSwiglu(self, banks, x, .{ .ctx = null }, .{ .ctx = null }, &opts, true);
}

/// The gemv gate logit where the routed shared expert did not run: the router kernel in `softmax_gate_gemv` (the
/// softmax mode's selection and weights, the gate's logit summed in MLX's gemv order) for `sharedFused`, whose gemv
/// launch then goes. Null: the router as before.
fn gateRouting(self: *Trunk, x: mlx.mlx_array, mw: *const MoeMlpWeights, router_logits: mlx.mlx_array) !?MoeRouting {
    const v = gateVec(self, x, mw, router_logits) orelse return null;
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    return routerTopK(self.s, router_logits, k, .softmax_gate_gemv, mlx.mlx_array_dtype(router_logits), .{ .x = x, .w = v });
}

// ── the router kernel (`sdk/blocks/moe/router_kernel.zig`'s softmax modes) ──
//
// One threadgroup per row: phase 1 stages every expert's key in threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a lowest-index tie-break, phase 3 normalizes the selected weights, with the arithmetic of
// the op chain it replaces (`fast::exp`, the reciprocal-multiply softmax, the output dtype's rounding at the chain's
// points), so its bytes are the chain's.

/// `RouterMode`'s softmax modes, by its names: the plain top-k; with the shared-expert gate's logit, a dot of each x
/// row with a flat weight vector that the simdgroups sitting out the selection compute while it runs (one row); with
/// that logit summed exactly as MLX's gemv sums it, simdgroups 1..8 of a 288-thread group playing the gemv's 0..7 (2..4
/// rows). The sigmoid modes are other models' routers.
const Mode = enum { softmax, softmax_gate, softmax_gate_gemv };

/// Where `softmax_gate_gemv`'s order is MLX's: bf16 operands and output, 2..4 rows (one row runs MLX's dot kernel), the
/// gate width K > 64 and K >= 16 rows (matmul's non-transposed gemv takes BM 1 BN 8 there).
fn gateGemvShape(rows: c_int, k: c_int, xd: mlx.mlx_dtype, wd: mlx.mlx_dtype, od: mlx.mlx_dtype) bool {
    return xd == .bfloat16 and wd == .bfloat16 and od == .bfloat16 and rows >= 2 and rows <= 4 and k > 64 and k >= 16 * rows;
}

/// The mode's kernel text: `router.fmt.metal` with the softmax arm's parts and the gate's where the mode has one, as
/// today's `moeRouterSource` splices them.
fn routerSource(comptime mode: Mode) [:0]const u8 {
    const gate_gemv = mode == .softmax_gate_gemv;
    const gate = mode == .softmax_gate or gate_gemv;
    return std.fmt.comptimePrint(kernelSource("kernels/moe/router.fmt.metal"), .{
        "",
        if (gate)
            "threadgroup float gp[TG / 32];"
        else
            "",
        "  rk[e] = float(logits[rbase + e]);",
        if (gate_gemv)
            "if (tid >= 32u && tid < 32u * 9u)\n" ++ gate_texts.gate_gemv
        else if (gate)
            // Simdgroups 1.. take the shared-expert gate's dot while simdgroup 0 selects; their partials meet after the
            // selection, so the dot never waits on it nor the selection on the dot.
            kernelSource("kernels/moe/router_gate_dot.metal")
        else
            "",
        "",
        // MLX's `softmax_single_row`, reduction tree included: ceil(NE/4) virtual threads each summing four consecutive
        // elements in order, a `simd_sum` per simdgroup, a final `simd_sum` over the partials; emulated 32 at a time,
        // the identical butterfly on identical values, so the rounded probability is bit-equal.
        kernelSource("kernels/moe/router_softmax_sum.metal"),
        // softmax(...) writes probabilities in the OUTPUT dtype, so the per-expert rounding happens here, before the
        // top-K sum.
        "    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));",
        "  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));",
        // `sum` over the K probabilities lands back in the output dtype before the divide, exactly as mlx_sum_axis +
        // mlx_divide do.
        "  w = w / float(TOUT(tot));",
        if (gate_gemv)
            // the gemv's simdgroup 0 lane 0 folds the 8 partials in order, then the bf16 store
            "threadgroup_barrier(mem_flags::mem_threadgroup);\nif (tid == 0) {\n" ++ gate_texts.gate_sum ++ "}"
        else if (gate)
            kernelSource("kernels/moe/router_gate_store.metal")
        else
            "",
    });
}

const router_sources = [_][:0]const u8{ routerSource(.softmax), routerSource(.softmax_gate), routerSource(.softmax_gate_gemv) };
const router_names = [_][*:0]const u8{ "mlxserve_moe_router_softmax", "mlxserve_moe_router_softmax_gate", "mlxserve_moe_router_softmax_gate_gemv" };
var router_kernels: [router_names.len]?mlx.mlx_fast_metal_kernel = @splat(null);

/// The mode's kernel, built once (today's `getMoeRouterKernel`).
fn routerKernel(mode: Mode) !mlx.mlx_fast_metal_kernel {
    const mi = @intFromEnum(mode);
    if (router_kernels[mi]) |k| return k;
    const softmax_inputs = [_][*:0]const u8{"logits"};
    const gate_inputs = [_][*:0]const u8{ "logits", "gx", "gw" };
    const input_names: []const [*:0]const u8 = if (mode == .softmax) &softmax_inputs else &gate_inputs;
    const plain_outputs = [_][*:0]const u8{ "inds", "scores" };
    const gate_outputs = [_][*:0]const u8{ "inds", "scores", "glogit" };
    const output_names: []const [*:0]const u8 = if (mode == .softmax) &plain_outputs else &gate_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(router_names[mi], in_vec, out_vec, router_sources[mi], "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    router_kernels[mi] = kernel;
    return kernel;
}

/// Threads per row. Selection needs a full simdgroup; phase 1 wants enough threads to cover the expert row without a
/// long strided loop.
fn routerThreadGroup(num_experts: c_int) c_int {
    const rounded = @divTrunc(num_experts + 31, 32) * 32;
    return @max(@as(c_int, 32), @min(@as(c_int, 256), rounded));
}

/// Everything the cached kernel config bakes in (the output shapes and grid among them, so the key carries the
/// logits' full shape). A mismatch re-arms it.
const RouterCfgKey = struct {
    shape: ShapeKey,
    k: c_int,
    out_dtype: mlx.mlx_dtype,
    gate_k: c_int = 0,
};
const RouterCfgSlot = struct {
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
    key: RouterCfgKey = std.mem.zeroes(RouterCfgKey),
};
var router_cfgs: [router_names.len]RouterCfgSlot = @splat(.{});

/// Top-k selection and weight normalization in one kernel (today's `moeRouterTopK` in its softmax modes, the weights
/// renormalized: `route_norm`). Null where the geometry or dtype is outside the kernel's set. The router is on
/// (`moeRouterFusedEnabled` is a test seam).
fn routerTopK(s: mlx.mlx_stream, logits: mlx.mlx_array, k: c_int, mode_in: Mode, out_dtype: mlx.mlx_dtype, gate: ?RouterGate) !?MoeRouting {
    if (!mlx.streamIsGpu(s)) return null;
    const mode: Mode = if (gate != null and mode_in == .softmax) .softmax_gate else mode_in;
    const lsh = mlx.getShape(logits);
    if (lsh.len < 1 or lsh.len > 4) return null;
    const num_experts = lsh[lsh.len - 1];
    if (num_experts < k or num_experts > ROUTER_MAX_EXPERTS) return null;
    if (k < 1 or k > 32) return null;
    const ldt = mlx.mlx_array_dtype(logits);
    if (ldt != .bfloat16 and ldt != .float16 and ldt != .float32) return null;

    var rows: c_int = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= d;
    if (rows < 1) return null;
    // The gate logit: one flat weight vector, x holding one row of it per router row, in a threadgroup that has a
    // simdgroup to spare for every 32 threads.
    var gate_k: c_int = 0;
    if (gate) |gt| {
        const wn = mlx.mlx_array_size(gt.w);
        if (wn == 0 or wn > 65536 or mlx.mlx_array_size(gt.x) != @as(usize, @intCast(rows)) * wn) return null;
        const xd = mlx.mlx_array_dtype(gt.x);
        const wd = mlx.mlx_array_dtype(gt.w);
        if ((xd != .bfloat16 and xd != .float16) or (wd != .bfloat16 and wd != .float16)) return null;
        gate_k = @intCast(wn);
        if (mode == .softmax_gate_gemv and !gateGemvShape(rows, gate_k, xd, wd, out_dtype)) return null;
    }
    if (mode == .softmax_gate_gemv and gate == null) return null;

    // the gemv mode's logit takes simdgroups 1..8 beside the selection's 0, whatever the expert count
    const tg: c_int = if (mode == .softmax_gate_gemv) 32 * 9 else routerThreadGroup(num_experts);
    // The gate's dot runs on the simdgroups that sit out the selection: with one there is none to run it.
    if (mode == .softmax_gate and tg < 64) return routerTopK(s, logits, k, .softmax, out_dtype, null);

    // One cached config per mode: rebuilding it every call would move the dispatches saved to the CPU side. A model's
    // routing geometry is fixed, so it covers decode, and prefill re-arms it once per width.
    const key = RouterCfgKey{ .shape = ShapeKey.from(lsh), .k = k, .out_dtype = out_dtype, .gate_k = gate_k };
    const slot = &router_cfgs[@intFromEnum(mode)];
    if (slot.cfg.ctx == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(slot.cfg);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        // Outputs keep the caller's leading dims with the expert axis replaced by K.
        var out_shape: [4]c_int = undefined;
        for (lsh[0 .. lsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[lsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, out_dtype));
        if (mode != .softmax) {
            // One logit per router row.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{rows}, 1, out_dtype));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GK", gate_k));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TOUT", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NE", num_experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TG", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NORM", 1));
        slot.* = .{ .cfg = config, .key = key };
    }
    const config = slot.cfg;

    const softmax_inputs = [_]mlx.mlx_array{logits};
    const gate_inputs = [_]mlx.mlx_array{ logits, if (gate) |gt| gt.x else logits, if (gate) |gt| gt.w else logits };
    const inputs_arr: []const mlx.mlx_array = if (mode == .softmax) &softmax_inputs else &gate_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try routerKernel(mode);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != (if (mode == .softmax) @as(usize, 2) else 3)) return error.MetalKernelBadOutputCount;

    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    try mlx.check(mlx.mlx_vector_array_get(&inds, outputs_vec, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, outputs_vec, 1));
    var gate_logit = mlx.mlx_array{ .ctx = null };
    if (mode != .softmax) {
        gate_logit = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&gate_logit, outputs_vec, 2));
    }
    return .{ .inds = inds, .norm_scores = scores, .gate_logit = gate_logit };
}
