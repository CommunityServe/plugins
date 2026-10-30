//! copied_from: src/sdk/blocks/moe/router_kernel.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/moe/router_kernel.
//! Owns RouterMode, ROUTER_MAX_EXPERTS, moeRouterSource, ROUTER_SOURCES, ROUTER_NAMES, router_kernels, and 13 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
/// The gemv gate's two texts, turboexl3's (the same bytes its own gate kernels splice).
const gate_texts = @import("turboexl3").sources;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const groupLimitedRouting = @import("routing.zig").groupLimitedRouting;
const MoeRouting = @import("routing.zig").MoeRouting;
const hy3RoutingChain = @import("routing.zig").hy3RoutingChain;
const kolibriRoutingChain = @import("routing.zig").kolibriRoutingChain;
const moeRoutingChain = @import("routing.zig").moeRoutingChain;

// ── Fused MoE router (top-K selection + weight normalization in ONE kernel) ──
//
// The op chains above are ~11 dispatches each — sigmoid/softmax, bias add,
// negate, argpartition (a full 256-wide sort), slice, take_along_axis, sum,
// eps add, divide, scale multiply, cast — over a [rows, E] tensor that is 512
// bytes wide at decode. Every one of them is launch-latency bound, and a MoE
// model pays the whole chain once per LAYER: 40 layers x 11 = 440 dispatches
// per token on Laguna, none of which move meaningful data.
//
// One threadgroup per row does all of it: phase 1 computes the selection key
// for every expert into threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a LOWEST-INDEX tie-break, phase 3 normalizes the
// selected weights. Selection is exact integer/compare logic, so the picked
// expert set is the same set the chain picks whenever the keys are distinct.
//
// The ARITHMETIC deliberately replicates the chain's rather than improving on
// it — same `fast::exp`, same reciprocal-multiply softmax normalization, same
// MLX `Sigmoid` two-sided form, same intermediate rounding to the output dtype
// at the same points. A "more accurate" fused router is still a DIFFERENT
// model: the first live A/B of this kernel computed the softmax weights in one
// f32 pass and diverged from the chain on gemma4-26B-A4B at token ~80, which
// is the known INT4 near-tie argmax class amplified by an avoidable numerical
// change. The tests below therefore assert BIT equality with the chain, not
// just no-worse-than-reference; only the reduction ORDER over the expert axis
// differs (a simd tree vs MLX's N_READS blocking), which cannot move a bf16
// result except at an exact rounding boundary.
//
// Engagement is LOGGED, never
// inferred — a declined kernel is output-identical to the chain.
/// `sigmoid_bias_grouped` is `sigmoid_bias` with a GROUP LIMIT spliced
/// between the bias and the top-k: the experts are split into NG equal groups,
/// each group is scored by the SUM OF ITS TOP TWO, and every expert outside the
/// best TOPKG groups is pushed to the composed chain's own finite -1e4 penalty
/// before selection. The composed chain is ~28 dispatches per MoE layer x 41 layers,
/// and a dispatch prices at 1.76 us on this arch (diag dispatch-probe
/// slope over 1000 injected ops), so that chain alone is ~2 ms of a 19 ms token.
/// `softmax_gate` is `softmax` plus the shared-expert gate's logit, a dot of one x row with a flat
/// weight vector that the simdgroups sitting out the selection compute while it runs (the matmul it
/// replaces is two dependent launches ahead of the router). It needs at least two simdgroups.
/// `logit_bias` (kolibri1) selects on `logits + bias` and weights by `sigmoid(logits)`.
/// `softmax_gate_gemv` is `softmax_gate` at 2..4 rows with the logit summed exactly as MLX's gemv sums it (simdgroups
/// 1..8 of a 288-thread group play the gemv's 0..7): the bytes of the shared gate's matmul, which its consumer then drops.
pub const RouterMode = enum(u8) { softmax = 0, sigmoid_bias = 1, sigmoid_bias_grouped = 2, softmax_gate = 3, logit_bias = 4, softmax_gate_gemv = 5 };
const n_modes = @typeInfo(RouterMode).@"enum".field_names.len;

/// Largest expert count the kernel will stage in threadgroup memory. The
/// sigmoid arm keeps two f32 planes (key + raw weight), so 2048 experts is
/// 16 KiB of the 32 KiB budget.
pub const ROUTER_MAX_EXPERTS: c_int = 2048;

/// Where `softmax_gate_gemv`'s order is MLX's: bf16 operands and output, 2..4 rows (one row runs MLX's dot kernel), the
/// gate width K > 64 and K >= 16 rows (matmul's non-transposed gemv takes BM 1 BN 8 there).
pub fn gateGemvShape(rows: c_int, k: c_int, xd: mlx.mlx_dtype, wd: mlx.mlx_dtype, od: mlx.mlx_dtype) bool {
    return xd == .bfloat16 and wd == .bfloat16 and od == .bfloat16 and rows >= 2 and rows <= 4 and k > 64 and k >= 16 * rows;
}

fn moeRouterSource(comptime mode: RouterMode) [:0]const u8 {
    const logit_sel = mode == .logit_bias;
    const sigmoid = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped or logit_sel;
    const grouped = mode == .sigmoid_bias_grouped;
    const gate_gemv = mode == .softmax_gate_gemv;
    const gate = mode == .softmax_gate or gate_gemv;
    return std.fmt.comptimePrint(kernelSource("kernels/moe/router.fmt.metal"), .{
        if (grouped)
            "threadgroup float rw[NE];\nthreadgroup float gs[NG];"
        else if (sigmoid)
            "threadgroup float rw[NE];"
        else
            "",
        if (gate)
            "threadgroup float gp[TG / 32];"
        else
            "",
        if (sigmoid)
            // MLX's `Sigmoid` unary op, verbatim (unary_ops.h): the two-sided
            // form, not 1/(1+exp(-x)) — they differ in the last ulp. `exp` is
            // spelled PRECISE because copying upstream's TEXT is not copying
            // its ARITHMETIC: MLX's metallib is built `-fno-fast-math`
            // (kernels/CMakeLists.txt), so its bare `metal::exp` is the precise
            // one, while this source is JIT-compiled at MathMode::Safe where
            // the same spelling lowers to the fast variant. Measured 13.8% of
            // 1M random f32 sigmoids 1 ulp apart, and 6.1% of the routing
            // weights that come out of them; precise is 0/1M. The softmax arm
            // below spells `fast::exp` for the mirror-image reason (softmax.h
            // asks for fast explicitly). Costs ~1% of the kernel at an 8192-row
            // prefill width and nothing measurable at decode.
            kernelSource("kernels/moe/router_sigmoid_score.metal") ++ "\n  // kolibri1 selects on the raw logit + bias; the others on sigmoid + bias.\n  rk[e] = " ++ (if (logit_sel) "l" else "sg") ++ " + float(bias[e]);\n  rw[e] = sg;"
        else
            "  rk[e] = float(logits[rbase + e]);",
        if (gate_gemv)
            "if (tid >= 32u && tid < 32u * 9u)\n" ++ gate_texts.gate_gemv
        else if (gate)
            // Simdgroups 1.. take the shared-expert gate's dot while simdgroup 0 selects; their partials
            // meet after the selection, so the dot never waits on it nor the selection on the dot.
            kernelSource("kernels/moe/router_gate_dot.metal")
        else
            "",
        if (grouped)
            // The group limit (`group_limited_topk` shape). One lane
            // per group scores it by the SUM OF ITS TOP TWO biased experts, then
            // TOPKG rounds of the same max-then-mask/lowest-index selection the
            // expert top-k uses, then every expert in a losing group is pushed to
            // -1e4 — the composed chain's own FINITE penalty, not -inf: biased
            // scores are sigmoid+bias so they live well inside +/-2, and a
            // non-finite sentinel in a tensor that feeds selection is how the
            // composed causal-mask arm once shipped an all-NaN path that scored a
            // perfect parity max_err of 0.
            kernelSource("kernels/moe/router_group_limit.metal")
        else
            "",
        if (sigmoid)
            ""
        else
            // MLX's `softmax_single_row`, reduction tree included. It launches
            // ceil(NE/4) threads, each summing FOUR consecutive elements in
            // order, then one `simd_sum` per simdgroup, then a final `simd_sum`
            // over the per-simdgroup partials sitting in lanes 0..SIMDS-1 with
            // zeros above. Emulating those virtual threads 32 at a time
            // reproduces the identical butterfly on identical values, which is
            // what makes the rounded bf16 probability bit-equal; a plain
            // strided sum over the same data was 1 ulp off on real inputs.
            kernelSource("kernels/moe/router_softmax_sum.metal"),
        if (sigmoid)
            "    w = rw[myidx];"
        else
            // softmax(...) writes probabilities in the OUTPUT dtype, so the
            // per-expert rounding happens here, before the top-K sum.
            "    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));",
        if (sigmoid)
            // The hy3 chain sums FLOAT32 sigmoid scores, so the accumulator is
            // float — but still ascending, still sequential.
            "  for (int j = 0; j < NK; ++j) tot += simd_shuffle(w, ushort(j));"
        else
            "  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));",
        if (sigmoid)
            kernelSource("kernels/moe/router_sigmoid_norm.metal")
        else
            // `sum` over the K probabilities lands back in the output dtype
            // before the divide, exactly as mlx_sum_axis + mlx_divide do.
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

pub const ROUTER_SOURCES = [n_modes][:0]const u8{ moeRouterSource(.softmax), moeRouterSource(.sigmoid_bias), moeRouterSource(.sigmoid_bias_grouped), moeRouterSource(.softmax_gate), moeRouterSource(.logit_bias), moeRouterSource(.softmax_gate_gemv) };
const ROUTER_NAMES = [n_modes][*:0]const u8{ "mlxserve_moe_router_softmax", "mlxserve_moe_router_sigmoid", "mlxserve_moe_router_sigmoid_grouped", "mlxserve_moe_router_softmax_gate", "mlxserve_moe_router_logit_bias", "mlxserve_moe_router_softmax_gate_gemv" };
var router_kernels: [n_modes]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getMoeRouterKernel(mode: RouterMode) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @backingInt(mode);
    if (router_kernels[mi]) |k| return k;
    const softmax_inputs = [_][*:0]const u8{"logits"};
    const gate_inputs = [_][*:0]const u8{ "logits", "gx", "gw" };
    const sigmoid_inputs = [_][*:0]const u8{ "logits", "bias", "scale" };
    const input_names: []const [*:0]const u8 = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate, .softmax_gate_gemv => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped, .logit_bias => &sigmoid_inputs,
    };
    const plain_outputs = [_][*:0]const u8{ "inds", "scores" };
    const gate_outputs = [_][*:0]const u8{ "inds", "scores", "glogit" };
    const output_names: []const [*:0]const u8 = if (mode == .softmax_gate or mode == .softmax_gate_gemv) &gate_outputs else &plain_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        ROUTER_NAMES[mi],
        in_vec,
        out_vec,
        ROUTER_SOURCES[mi],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    router_kernels[mi] = kernel;
    return kernel;
}

/// Threads per row. Selection needs a full simdgroup; phase 1 wants enough
/// threads to cover the expert row without a long strided loop.
pub fn moeRouterThreadGroup(num_experts: c_int) c_int {
    const rounded = @divTrunc(num_experts + 31, 32) * 32;
    return @max(@as(c_int, 32), @min(@as(c_int, 256), rounded));
}

/// Everything the cached kernel config bakes in. A mismatch re-arms it.
/// Full-shape component for cached metal-kernel config keys. A cached config
/// bakes the OUTPUT SHAPES (and grid) of the call that built it, so the key
/// must carry the caller's actual dims: keying on a PRODUCT (rows, n) collides
/// across layouts with equal element counts and hands the colliding caller an
/// output labeled with the FIRST caller's shape. Live (Laguna XS):
/// a MoE-layer SwiGLU at a 16-token hot-cache-restore prefill is (1,16,512) =
/// 8192 elements — the same count/rank/dtype as a dense mlp_only layer at
/// decode (1,1,8192) — and the mislabeled activation killed the server in the
/// dense down-proj matmul. Every cached config key whose config bakes an
/// input-derived shape uses this instead of derived products.
pub const ShapeKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    pub fn from(sh: []const c_int) ShapeKey {
        var key = ShapeKey{};
        for (sh, 0..) |d, i| key.dims[i] = d;
        key.len = @intCast(sh.len);
        return key;
    }
};

const RouterCfgKey = struct {
    shape: ShapeKey,
    k: c_int,
    out_dtype: mlx.mlx_dtype,
    norm: bool,
    scale: f32,
    n_group: c_int = 0,
    topk_group: c_int = 0,
    gate_k: c_int = 0,
};
const RouterCfgSlot = struct {
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
    key: RouterCfgKey = std.mem.zeroes(RouterCfgKey),
    scale_arr: mlx.mlx_array = .{ .ctx = null },
};
var router_cfg_cache: [n_modes]RouterCfgSlot = @splat(.{});

pub var moe_router_fused_override: ?bool = null;
var moe_router_engaged: bool = false;

pub fn moeRouterFusedEnabled() bool {
    return moe_router_fused_override orelse true;
}

/// The shared-expert gate's logit, folded into the softmax router: `x` [rows * gate_k] and the flat
/// weight vector `w` [gate_k], one logit per row.
pub const RouterGate = struct { x: mlx.mlx_array, w: mlx.mlx_array };

/// One-kernel replacement for `moeRoutingChain` / `hy3RoutingChain`.
/// Returns null when the geometry or dtype is outside the supported set, so
/// every caller keeps its chain fallback.
pub fn moeRouterTopK(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode_in: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
    gate: ?RouterGate,
) !?MoeRouting {
    if (!moeRouterFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const mode: RouterMode = if (gate != null and mode_in == .softmax) .softmax_gate else mode_in;
    const sigmoid_mode = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped or mode == .logit_bias;
    const lsh = mlx.getShape(logits);
    if (lsh.len < 1 or lsh.len > 4) return null;
    const num_experts = lsh[lsh.len - 1];
    if (num_experts < k or num_experts > ROUTER_MAX_EXPERTS) return null;
    if (k < 1 or k > 32) return null;
    const ldt = mlx.mlx_array_dtype(logits);
    if (ldt != .bfloat16 and ldt != .float16 and ldt != .float32) return null;
    if (sigmoid_mode and bias.ctx == null) return null;
    if (mode == .sigmoid_bias_grouped) {
        // One lane per group, one bit per group in the keep mask, and a group
        // score that is a top-2 sum. Anything outside that declines to the
        // composed chain rather than computing a different model.
        if (n_group <= 1 or n_group > 32) return null;
        if (topk_group <= 0 or topk_group > n_group) return null;
        if (@rem(num_experts, n_group) != 0) return null;
        if (@divExact(num_experts, n_group) < 2) return null;
    }

    var rows: c_int = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= d;
    if (rows < 1) return null;
    // The gate logit: one flat weight vector, x holding one row of it per router row, in a threadgroup
    // that has a simdgroup to spare for every 32 threads.
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
    const tg: c_int = if (mode == .softmax_gate_gemv) 32 * 9 else moeRouterThreadGroup(num_experts);
    // The gate's dot runs on the simdgroups that sit out the selection: with one there is none to run it.
    if (mode == .softmax_gate and tg < 64) return moeRouterTopK(s, logits, bias, k, .softmax, route_norm, route_scale, out_dtype, n_group, topk_group, null);

    // The whole point of the kernel is to spend ONE dispatch where the chain
    // spent a dozen; rebuilding the config every call would just move that cost
    // to the CPU side (measured +0.16 ms of graph-build per token on Laguna).
    // A model's routing geometry is fixed, so one cached config per mode covers
    // decode, and prefill re-arms it once per width.
    const key = RouterCfgKey{
        .shape = ShapeKey.from(lsh),
        .k = k,
        .out_dtype = out_dtype,
        .norm = route_norm,
        .scale = route_scale,
        .n_group = n_group,
        .topk_group = topk_group,
        .gate_k = gate_k,
    };
    const slot = &router_cfg_cache[@backingInt(mode)];
    if (slot.cfg.ctx == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(slot.cfg);
        if (slot.scale_arr.ctx != null) _ = mlx.mlx_array_free(slot.scale_arr);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        // Outputs keep the caller's leading dims with the expert axis replaced by K.
        var out_shape: [4]c_int = undefined;
        for (lsh[0 .. lsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[lsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, out_dtype));
        if (mode == .softmax_gate or mode == .softmax_gate_gemv) {
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
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NORM", @intFromBool(route_norm)));
        if (mode == .sigmoid_bias_grouped) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NG", n_group));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSZ", @divExact(num_experts, n_group)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPKG", topk_group));
        }
        slot.* = .{ .cfg = config, .key = key, .scale_arr = mlx.mlx_array_new_float(route_scale) };
    }
    const config = slot.cfg;

    const softmax_inputs = [_]mlx.mlx_array{logits};
    const gate_inputs = [_]mlx.mlx_array{ logits, if (gate) |gt| gt.x else logits, if (gate) |gt| gt.w else logits };
    const sigmoid_inputs = [_]mlx.mlx_array{ logits, bias, slot.scale_arr };
    const inputs_arr: []const mlx.mlx_array = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate, .softmax_gate_gemv => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped, .logit_bias => &sigmoid_inputs,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getMoeRouterKernel(mode);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != (if (mode == .softmax_gate or mode == .softmax_gate_gemv) @as(usize, 3) else 2)) return error.MetalKernelBadOutputCount;

    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    try mlx.check(mlx.mlx_vector_array_get(&inds, outputs_vec, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, outputs_vec, 1));
    var gate_logit = mlx.mlx_array{ .ctx = null };
    if (mode == .softmax_gate or mode == .softmax_gate_gemv) {
        gate_logit = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&gate_logit, outputs_vec, 2));
    }

    if (!moe_router_engaged) {
        moe_router_engaged = true;
        log.info("[moe] fused router kernel engaged: mode={s} E={d} K={d} tg={d}\n", .{ @tagName(mode), num_experts, k, tg });
    }
    return .{ .inds = inds, .norm_scores = scores, .gate_logit = gate_logit };
}

// ── tests ──
