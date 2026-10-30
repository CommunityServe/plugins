//! The direct model's PLE (layer 1): the n-gram rows gathered and decoded on the host (cold rows preread, rows staged
//! ahead, the early layers kicked), the projections, the decode tail and the verify ids' lend. The K path of today's
//! `ple.zig` and of the n-gram table's runtime in `qwen4_exp.zig` (the hash, the trellis ring decode, the 48-worker pool
//! and the prefetch's readers), over the copy's types. The table's open is the load's, which makes the pool and the
//! readers with `poolCreate` / `aheadCreate`.
//! A K pack's table is trellis-coded, so its rows decode on the host: the copy's GPU gather and its affine, BF16 and
//! NVFP4 rows are named errors here. Its PLE layer is bf16 with a conv state, so the four-launch tail serves every
//! width, and the copy's MLX chain, which served any other layer, is a named error too.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const crt = @import("../mlx_serve/sdk/util/crt.zig");
const io_util = @import("../mlx_serve/sdk/util/io_util.zig");
const turboexl3 = @import("turboexl3");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const scalarOf = @import("../mlx_serve/mlx/array.zig").scalarOf;
const materializedOwnedCopy = @import("../mlx_serve/mlx/array.zig").materializedOwnedCopy;
const cachedScalarInt = @import("../mlx_serve/sdk/blocks/linear/verify_qmm.zig").cachedScalarInt;
const sigmoidTableFor = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const ShapeKey = @import("../mlx_serve/sdk/blocks/moe/router_kernel.zig").ShapeKey;
const MAX_BATCH_ROWS = @import("../mlx_serve/engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const PleWeights = @import("../mlx_serve/plugins/qwen4_exp/weights.zig").PleWeights;
const PlePending = @import("../mlx_serve/plugins/qwen4_exp/ple.zig").PlePending;
const ssmFreeQsaState = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const linear = @import("linear.zig");
const qwen4 = @import("../mlx_serve/plugins/qwen4_exp/qwen4_exp.zig");
const Qwen4State = qwen4.Qwen4State;
const NgramHash = qwen4.NgramHash;
const NgramTable = qwen4.NgramTable;

/// The copy's types the table's runtime works on, private to qwen4_exp.zig: reached through the fields that hold them.
const PrefetchPool = Pointee(@FieldType(NgramTable, "pool"));
const PoolJob = @FieldType(PrefetchPool, "job");
const Ahead = Pointee(@FieldType(NgramTable, "ahead"));
const AheadSession = Pointee(@FieldType(Ahead, "session"));
const AheadView = Pointee(@FieldType(PrefetchPool, "ahead_view"));
const AheadSlot = @typeInfo(@FieldType(AheadSession, "slots")).pointer.child;

/// `?*T`'s `T`.
fn Pointee(comptime O: type) type {
    return @typeInfo(@typeInfo(O).optional.child).pointer.child;
}

/// The pool's workers, the rows of one fan-out round and a row buffer's bytes (the copy's `N`, `MAX_ROWS`, `ROW_BUF`),
/// the prefetch's readers and the segments one gather sees: the lengths of the arrays the copy's types hold them in.
const POOL_N = @typeInfo(@FieldType(PrefetchPool, "threads")).array.len;
const MAX_ROWS = @typeInfo(@FieldType(PrefetchPool, "bufs")).array.len;
const ROW_BUF = @typeInfo(@typeInfo(@FieldType(PrefetchPool, "bufs")).array.child).array.len;
const AHEAD_READERS = @typeInfo(@FieldType(Ahead, "threads")).array.len;
const AHEAD_VIEW_SEGS = @typeInfo(@FieldType(AheadView, "slots")).array.len;

// ── the interface ──

/// `stream + the PLE layer's addend` `[B, S, hc*hidden]`: the call site's residual add inside the tail's last launch.
/// The gather may stay pending until `flushDeferred` (a `ple_defer` forward at a deferrable width).
pub fn forwardResidual(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer_idx: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    return pleForwardWith(self, ctx, stream, token_ids, pw, entry, layer_idx, batch, seq_len);
}

/// Fill the leaf a `ple_defer` forward was built on: the token ids are read NOW (this is the one host sync of a
/// pipelined qwen4 step), the rows gathered straight into the leaf's buffer, and the per-entry n-gram history advanced.
/// No-op without a pending leaf.
pub fn flushDeferred(self: *Trunk, ctx: *ForwardCtx) !void {
    const p = ctx.ple_pending orelse return;
    // the copy's GPU gather leaves a record that owes only the history; a trellis table never makes one
    if (p.gpu) return error.DirectModelPleGatherIsHost;
    const dst = mlx.mlx_array_data_bfloat16(p.emb) orelse return error.PleLeafUnreadable;
    const out: [*]u16 = @constCast(dst);
    const pk = out[0..mlx.mlx_array_size(p.emb)];
    // a solo verify's ids off the arrays its input was built from, when the caller lent them (the copy lends on an EXL3
    // trunk pack, `trunk_format`, which every K pack is)
    const from_src = if (ctx.ple_ids_src) |src| try pleFillFromSources(self, ctx, src, p.token_ids, p.entry, pk, p.capture) else false;
    if (!from_src) try pleGatherBf16(self, ctx, p.token_ids, p.entry, p.layer, p.seq_len, pk, p.capture);
    ctx.ple_pending = null;
    releasePlePending(p);
}

/// Fill every pending deferred-PLE leaf of a group in ONE pass: every row's ids read in one host sync. The caller
/// retains rollback state until the whole round succeeds.
pub fn flushDeferredPleGroup(self: *Trunk, ctxs: []const *ForwardCtx) !void {
    var ids: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (ids[0..n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (ctxs.len > ids.len) return error.PleGroupTooWide;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (ctxs) |ctx| {
        const p = ctx.ple_pending orelse continue;
        ids[n] = try plePrepareIds(self, p.token_ids);
        _ = mlx.mlx_vector_array_append_value(vec, ids[n]);
        n += 1;
    }
    if (n == 0) return;
    try mlx.check(mlx.mlx_eval(vec));
    var k: usize = 0;
    for (ctxs) |ctx| {
        const p = ctx.ple_pending orelse continue;
        if (p.gpu) return error.DirectModelPleGatherIsHost;
        const dst = mlx.mlx_array_data_bfloat16(p.emb) orelse return error.PleLeafUnreadable;
        const out: [*]u16 = @constCast(dst);
        try pleFillFromIds(self, ctx, ids[k], p.entry, p.layer, p.seq_len, out[0..mlx.mlx_array_size(p.emb)], p.capture);
        k += 1;
        ctx.ple_pending = null;
        releasePlePending(p);
    }
}

/// The copy's GPU arm: a deferred forward's history advanced from ids the caller already read. A host leaf owes its
/// rows, which only a flush fills, and a trellis table leaves no other record.
pub fn settleDeferredPle(_: *Trunk, ctx: *ForwardCtx, _: []const u32) !void {
    const p = ctx.ple_pending orelse return;
    if (!p.gpu) return error.PleLeafUnfilled;
    return error.DirectModelPleGatherIsHost;
}

/// Drop a pending leaf without filling it (the forward that built on it is being abandoned). The spec-PLE slot the
/// build claimed is released too — no gather ever filled its tokens.
pub fn discardDeferredPle(_: *Trunk, ctx: *ForwardCtx) void {
    const p = ctx.ple_pending orelse return;
    ctx.ple_pending = null;
    p.entry.spec_ple_len = 0;
    if (ctx.batch_slots) |slots| {
        for (slots) |sc| sc.ssm_entries.?[p.layer].spec_ple_len = 0;
    }
    releasePlePending(p);
}

pub fn releasePlePending(p: PlePending) void {
    if (p.emb.ctx != null) _ = mlx.mlx_array_free(p.emb);
    _ = mlx.mlx_array_free(p.token_ids);
}

/// Ends the prefill's prefetch session (`prefetchBegin`), on every exit of the prefill.
pub fn plePrefetchEnd(self: *Trunk) void {
    const st = self.qwen4 orelse return;
    prefetchEnd(&st.table);
}

/// A partial accept's n-gram history (the SSM rollback's): the `ngram_size - 1` verify tokens ending at `t1` plus
/// `accepted` drafts, read off the capture's `prev ++ ids`.
pub fn plePrevAfterAccept(entry: *SSMCacheEntry, accepted: u32, verify_len: u32) void {
    const ctx_len: usize = @as(usize, entry.spec_ple_len) - @as(usize, verify_len);
    for (0..ctx_len) |i| entry.ple_prev[i] = entry.spec_ple_tokens[1 + accepted + i];
    entry.ple_prev_valid = true;
}

// ── the layer ──

/// Qwen4ExpTextPLELayer.forward: the n-gram rows and the key projection, then the value projection, the norms, the gate,
/// the dilated causal conv and the stream's add in the tail's four launches. Batched decode hands in the
/// MERGED conv window as `entry.aux_state`.
fn pleForwardWith(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const emb = try pleEmbedding(self, ctx, token_ids, entry, layer, seq_len);
    defer _ = mlx.mlx_array_free(emb);
    const key_raw = try linear.qmatmul(self, emb, pw.key_w, pw.key_s, pw.key_b);
    defer _ = mlx.mlx_array_free(key_raw);
    try decodePleArmed(self, stream, key_raw, pw, entry);
    return decodePleForward(self, stream, emb, key_raw, pw, entry, batch, seq_len);
}

// ── the PLE layer's elementwise tail in four launches ──
//
// Between its MLX projections, norms, gate sum and conv, the chain is 18 elementwise launches on the layer's critical
// path: the group norms' three weight products and the keys x queries product, the gate's seven ops and its value
// product, the conv input's concat, silu's sigmoid and product, the conv branch's add and the call site's residual
// add. Four launches carry them with every op's arithmetic and rounding unchanged: mlx Multiply / Add on two T values,
// the concat's copy, and the two unary chains (the gate's seven ops, MLX's sigmoid) read from tables of MLX's own
// results over every 16-bit pattern, so no transcendental is re-spelled.

/// The tail's operands, which every K pack's PLE layer has: bf16 throughout (the stream, the key projection, the three
/// norm weights [hc, hidden], the conv weight, the conv state if any) and a conv state (kernel >= 2). The copy ran its
/// MLX chain for a layer without them; here that is a named error. Its other arms are pinned: a K pack is an EXL3 trunk
/// pack (`trunk_format`) on an M5-class GPU's stream, and runs no trace.
fn decodePleArmed(self: *Trunk, stream: mlx.mlx_array, key_raw: mlx.mlx_array, pw: *const PleWeights, entry: *const SSMCacheEntry) !void {
    const hw: usize = @as(usize, self.config.hc_count) * @as(usize, self.config.hidden_size);
    for ([_]mlx.mlx_array{ stream, key_raw, pw.norm_key, pw.norm_query, pw.norm_conv, pw.conv_w }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return error.DirectModelPleNotBf16;
    }
    for ([_]mlx.mlx_array{ pw.norm_key, pw.norm_query, pw.norm_conv }) |w| {
        if (mlx.mlx_array_size(w) != hw) return error.DirectModelPleShape;
    }
    if (hw == 0 or mlx.mlx_array_size(stream) % hw != 0 or mlx.mlx_array_size(key_raw) != mlx.mlx_array_size(stream)) return error.DirectModelPleShape;
    if (entry.aux_state.ctx != null and mlx.mlx_array_dtype(entry.aux_state) != .bfloat16) return error.DirectModelPleNotBf16;
    if (self.config.ple_conv_kernel < 2) return error.DirectModelPleNoConvState;
}

/// The groups' RMS norm alone ([B, S, hc, hidden]); the launch that reads it applies the norm's weight.
fn hcGroupRms(self: *Trunk, x: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const n_hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const shape4 = [_]c_int{ batch, seq_len, n_hc, hidden };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, self.s));
    var n4 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4, x4, self.ones_hidden, self.config.rms_norm_eps, self.s));
    return n4;
}

/// The layer past its key projection: its elementwise ops in four launches (`pleDecKq`, `pleDecGv`, `pleDecCat`,
/// `pleDecTail`) and every MLX op between them the chain's own (the value projection, the three RMS norms, the gate's
/// sum, the conv, the state's copy and capture): the same bytes, 14 dispatches and 11 dependent levels fewer per PLE
/// layer.
fn decodePleForward(self: *Trunk, stream: mlx.mlx_array, emb: mlx.mlx_array, key_raw: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const n_hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const value = try linear.qmatmul(self, emb, pw.value_w, pw.value_s, pw.value_b);
    defer _ = mlx.mlx_array_free(value);
    if (mlx.mlx_array_dtype(value) != .bfloat16) return error.DirectModelPleNotBf16;
    // the keys' and queries' norms, then their weight products and product in one launch
    const nk = try hcGroupRms(self, key_raw, batch, seq_len);
    defer _ = mlx.mlx_array_free(nk);
    const nq = try hcGroupRms(self, stream, batch, seq_len);
    defer _ = mlx.mlx_array_free(nq);
    const kq = try pleDecKq(self.s, nk, pw.norm_key, nq, pw.norm_query);
    defer _ = mlx.mlx_array_free(kq);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sum_axis(&gate, kq, -1, true, self.s)); // [B,S,hc,1]
    // the gate's seven ops (one table read) and its value product in one launch: gv [B, S, hc * hidden]
    const gv = try pleDecGv(self.s, gate, value, try pleGateTable(self.s, hidden), batch, seq_len, n_hc, hidden);
    defer _ = mlx.mlx_array_free(gv);
    const nc = try hcGroupRms(self, gv, batch, seq_len);
    defer _ = mlx.mlx_array_free(nc);

    // Dilated depthwise causal conv over [state | gvn]: gvn's weight product inside the concat launch.
    const dilation: c_int = @intCast(cfg.ngram_size);
    const state_len: c_int = (@as(c_int, @intCast(cfg.ple_conv_kernel)) - 1) * dilation;
    var prev_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_state);
    if (entry.aux_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&prev_state, entry.aux_state));
    } else {
        const zshape = [_]c_int{ batch, state_len, n_hc * hidden };
        try mlx.check(mlx.mlx_zeros(&prev_state, &zshape, 3, .bfloat16, self.s));
    }
    const cat = try pleDecCat(self.s, prev_state, nc, pw.norm_conv, batch, state_len, seq_len, n_hc * hidden);
    defer _ = mlx.mlx_array_free(cat);
    var conv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv);
    try mlx.check(mlx.mlx_conv1d(&conv, cat, pw.conv_w, 1, 0, dilation, n_hc * hidden, self.s));
    // New state = the last state_len rows of the concat (materialized — a view outliving `cat` pins the whole buffer).
    {
        const total: c_int = state_len + seq_len;
        const start = [_]c_int{ 0, total - state_len, 0 };
        const stop = [_]c_int{ batch, total, n_hc * hidden };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, cat, &start, 3, &stop, 3, &strides, 3, self.s));
        const owned = try materializedOwnedCopy(self.s, view);
        ssmFreeQsaState(entry); // PLE layer: no QSA state to lose
        entry.aux_state = owned;
    }
    if (self.spec_capture_ssm and entry.spec_ple_len > 0) {
        if (entry.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&entry.spec_ple_input, cat));
    }
    // silu, the conv branch's add and the call site's add in one launch
    return pleDecTail(self.s, conv, gv, try sigmoidTableFor(self.s, .bfloat16), stream);
}

const PLE_DEC_KQ_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_kq.metal");
const PLE_DEC_GV_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_gv.metal");
const PLE_DEC_CAT_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_cat.metal");
/// The tail with the call site's mlx Add of the stream too.
const PLE_DEC_TAIL_RES_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail.metal") ++ kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail_res.metal");

const PleDecKind = enum(u8) { kq = 0, gv = 1, cat = 2, tail_res = 3 };
var ple_dec_kernels: [4]?mlx.mlx_fast_metal_kernel = @splat(null);
const PleDecInt = struct { name: [*:0]const u8, value: c_int };
const PleDecCfgKey = struct { shape: ShapeKey, t0: c_int, t1: c_int };
var ple_dec_cfg: [4]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var ple_dec_cfg_key: [4]PleDecCfgKey = undefined;

fn pleDecInputs(kind: PleDecKind) []const [*:0]const u8 {
    return switch (kind) {
        .kq => &.{ "nk", "wk", "nq", "wq", "N_size" },
        .gv => &.{ "gate", "value", "gtab", "N_size" },
        .cat => &.{ "prev", "nc", "wc", "N_size", "S_rows" },
        .tail_res => &.{ "conv", "gv", "sigtab", "N_size", "h" },
    };
}

fn pleDecOutput(kind: PleDecKind) [*:0]const u8 {
    return switch (kind) {
        .kq => "kq",
        .gv => "gv",
        .cat => "cat",
        .tail_res => "y",
    };
}

const ple_dec_names = [4][*:0]const u8{ "mlxserve_ple_dec_kq", "mlxserve_ple_dec_gv", "mlxserve_ple_dec_cat", "mlxserve_ple_dec_tail_res" };
const ple_dec_sources = [4][:0]const u8{ PLE_DEC_KQ_SOURCE, PLE_DEC_GV_SOURCE, PLE_DEC_CAT_SOURCE, PLE_DEC_TAIL_RES_SOURCE };

fn pleDecKernel(kind: PleDecKind) !mlx.mlx_fast_metal_kernel {
    const i: usize = @intFromEnum(kind);
    if (ple_dec_kernels[i]) |kr| return kr;
    const ins = pleDecInputs(kind);
    const outs = [_][*:0]const u8{pleDecOutput(kind)};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(ple_dec_names[i], in_vec, out_vec, ple_dec_sources[i], "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    ple_dec_kernels[i] = kernel;
    return kernel;
}

/// One elementwise launch: bf16 output `out_shape` (one thread an element, 256 a group), template T bf16 and `ints`,
/// `inputs` in the kernel's order. The caller owns the result.
fn pleDecApply(s: mlx.mlx_stream, kind: PleDecKind, inputs: []const mlx.mlx_array, out_shape: []const c_int, ints: []const PleDecInt) !mlx.mlx_array {
    if (out_shape.len == 0 or out_shape.len > 4) return error.DecodePleShape;
    var n: i64 = 1;
    for (out_shape) |d| n *= d;
    if (n <= 0 or n > std.math.maxInt(c_int) - 256) return error.DecodePleShape;
    const ni: c_int = @intCast(n);
    const tg: c_int = 256;
    const i: usize = @intFromEnum(kind);
    // one config a kernel, rebuilt when the shape or a template constant changes: a config per call is graph-build time
    const key = PleDecCfgKey{ .shape = ShapeKey.from(out_shape), .t0 = if (ints.len > 0) ints[0].value else 0, .t1 = if (ints.len > 1) ints[1].value else 0 };
    if (ple_dec_cfg[i] == null or !std.meta.eql(ple_dec_cfg_key[i], key)) {
        if (ple_dec_cfg[i]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        ple_dec_cfg[i] = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, out_shape.ptr, out_shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, t.name, t.value));
        ple_dec_cfg[i] = config;
        ple_dec_cfg_key[i] = key;
    }
    const kernel = try pleDecKernel(kind);
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, ple_dec_cfg[i].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

/// Launch 1: kq = (nk * wk) * (nq * wq) [B, S, hc, hidden] from the keys' and queries' RMS norms and their norm weights
/// [hc, hidden] (the group norms' weight products and the layer's product).
fn pleDecKq(s: mlx.mlx_stream, nk: mlx.mlx_array, wk: mlx.mlx_array, nq: mlx.mlx_array, wq: mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(nk));
    return pleDecApply(s, .kq, &.{ nk, wk, nq, wq, cachedScalarInt(n) }, mlx.getShape(nk), &.{.{ .name = "HW", .value = @intCast(mlx.mlx_array_size(wk)) }});
}

/// Launch 2: gv [B, S, hc * hidden] = table[gate] * value, from the gate's sum [B, S, hc, 1], the value projection
/// [B, S, hidden] and the gate table (`pleGateTable`).
fn pleDecGv(s: mlx.mlx_stream, gate: mlx.mlx_array, value: mlx.mlx_array, gtab: mlx.mlx_array, batch: c_int, seq_len: c_int, n_hc: c_int, hidden: c_int) !mlx.mlx_array {
    const shape = [_]c_int{ batch, seq_len, n_hc * hidden };
    return pleDecApply(s, .gv, &.{ gate, value, gtab, cachedScalarInt(batch * seq_len * n_hc * hidden) }, &shape, &.{ .{ .name = "H", .value = hidden }, .{ .name = "HC", .value = n_hc } });
}

/// Launch 3: cat [B, SL + S, C] = [state | gvn], gvn the RMS-normed gv [B, S, C] times the conv norm's weight [C].
fn pleDecCat(s: mlx.mlx_stream, prev: mlx.mlx_array, nc: mlx.mlx_array, wc: mlx.mlx_array, batch: c_int, state_len: c_int, seq_len: c_int, width: c_int) !mlx.mlx_array {
    const shape = [_]c_int{ batch, state_len + seq_len, width };
    return pleDecApply(s, .cat, &.{ prev, nc, wc, cachedScalarInt(batch * (state_len + seq_len) * width), cachedScalarInt(seq_len) }, &shape, &.{ .{ .name = "C", .value = width }, .{ .name = "SL", .value = state_len } });
}

/// Launch 4: gv + silu(conv) + the stream `h` (`forwardResidual`); [B, S, C].
fn pleDecTail(s: mlx.mlx_stream, conv: mlx.mlx_array, gv: mlx.mlx_array, sigtab: mlx.mlx_array, h: mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(conv));
    return pleDecApply(s, .tail_res, &.{ conv, gv, sigtab, cachedScalarInt(n), h }, mlx.getShape(conv), &.{});
}

/// The gate's table at the hidden width it was built for (`pleGateTable`): a per-process cache.
var ple_gate_tables: struct { table: ?mlx.mlx_array = null, hidden: c_int = 0 } = .{};

/// `pleGateChain` over every bf16 bit pattern (the raw patterns as its input, so the table is indexed by the gate's own
/// bits, NaN payloads included), evaluated once per hidden width; borrowed by the caller.
fn pleGateTable(s: mlx.mlx_stream, hidden: c_int) !mlx.mlx_array {
    if (ple_gate_tables.table) |t| if (ple_gate_tables.hidden == hidden) return t;
    const bits = try std.heap.c_allocator.alloc(u16, 1 << 16);
    defer std.heap.c_allocator.free(bits);
    for (bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{1 << 16}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const table = try pleGateChain(s, x, hidden);
    errdefer _ = mlx.mlx_array_free(table);
    if (mlx.mlx_array_dtype(table) != .bfloat16) return error.DecodePleTableDtype;
    try mlx.check(mlx.mlx_array_eval(table));
    if (ple_gate_tables.table) |old| _ = mlx.mlx_array_free(old);
    ple_gate_tables = .{ .table = table, .hidden = hidden };
    return table;
}

/// The PLE gate: g -> sigmoid(sqrt(max(|g / sqrt(H)|, 1e-6)) * sign(g / sqrt(H))), seven MLX ops in g's dtype. Caller
/// owns the result.
fn pleGateChain(s: mlx.mlx_stream, gate: mlx.mlx_array, hidden: c_int) !mlx.mlx_array {
    // Scalars in the activation dtype: an f32 scalar promotes the gate and, through the value product, the whole
    // residual stream.
    const gate_dt = mlx.mlx_array_dtype(gate);
    const inv_sqrt_h = try scalarOf(1.0 / @sqrt(@as(f32, @floatFromInt(hidden))), gate_dt, s);
    defer _ = mlx.mlx_array_free(inv_sqrt_h);
    var gate_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_sc);
    try mlx.check(mlx.mlx_multiply(&gate_sc, gate, inv_sqrt_h, s));
    // sqrt(max(|g|, 1e-6)) · sign(g)
    var g_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_abs);
    try mlx.check(mlx.mlx_abs(&g_abs, gate_sc, s));
    const floor = try scalarOf(1e-6, gate_dt, s);
    defer _ = mlx.mlx_array_free(floor);
    var g_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_max);
    try mlx.check(mlx.mlx_maximum(&g_max, g_abs, floor, s));
    var g_sqrt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sqrt);
    try mlx.check(mlx.mlx_sqrt(&g_sqrt, g_max, s));
    var g_sign = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sign);
    try mlx.check(mlx.mlx_sign(&g_sign, gate_sc, s));
    var g_signed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_signed);
    try mlx.check(mlx.mlx_multiply(&g_signed, g_sqrt, g_sign, s));
    var g_sig = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g_sig);
    try mlx.check(mlx.mlx_sigmoid(&g_sig, g_signed, s));
    return g_sig;
}

// ── the rows: the leaf, the host gather, the history ──

/// Host-side n-gram gather: `[B, S, ple_embed_dim]` bf16 for this chunk's token ids, advancing the token history.
/// Serial: `entry`'s history over `[1, S]`. Batched (`ctx.batch_slots`): `[N, 1]`, each row hashed against ITS slot's
/// history (`slots[i].ssm_entries[layer]`), ONE gather for all N·heads rows. A `ple_defer` forward at a deferrable
/// width gets an unfilled leaf and a pending record instead.
fn pleEmbedding(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int) !mlx.mlx_array {
    const st = self.qwen4.?;
    // the copy's GPU gather (`pleEmbeddingGpu`) reads affine and BF16 rows: a trellis table keeps the host gather
    if (st.gpu != null) return error.DirectModelPleGatherIsHost;
    const emb_dim: usize = st.table.dim * st.hash.n_heads;
    const n: usize = mlx.mlx_array_size(token_ids);
    const batch: c_int = @intCast(n / @as(usize, @intCast(seq_len)));
    const shape = [_]c_int{ batch, seq_len, @intCast(emb_dim) };
    const capture = blk: {
        if (ctx.batch_slots) |slots| {
            if (!self.spec_capture_ssm) break :blk false;
            const per: usize = @intCast(seq_len);
            var any = false;
            for (slots) |sc| {
                if (pleClaimSpecCapture(self, &sc.ssm_entries.?[layer], per)) any = true;
            }
            if (any) _ = pleClaimSpecCapture(self, entry, per);
            break :blk any;
        }
        break :blk pleClaimSpecCapture(self, entry, n);
    };
    // Packed bf16 on the host (RNE) so the upload is one copy — no mid-graph eval, no GPU sync inside the layer loop.
    const pk = try self.allocator.alloc(u16, n * emb_dim);
    defer self.allocator.free(pk);
    if (ctx.ple_defer and pleDeferrable(n)) {
        if (ctx.ple_pending != null) return error.PlePendingAlreadySet;
        @memset(pk, 0);
        const emb = mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
        errdefer _ = mlx.mlx_array_free(emb);
        var emb_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(emb_ref);
        try mlx.check(mlx.mlx_array_set(&emb_ref, emb));
        var ids_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ids_ref);
        try mlx.check(mlx.mlx_array_set(&ids_ref, token_ids));
        ctx.ple_pending = .{ .emb = emb_ref, .token_ids = ids_ref, .entry = entry, .layer = layer, .seq_len = seq_len, .capture = capture };
        return emb;
    }
    try pleGatherBf16(self, ctx, token_ids, entry, layer, seq_len, pk, capture);
    return mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
}

/// A deferred PLE leaf is filled after the build, so a build that reads its routing on the host (the EXL3 prefill GEMM,
/// past `DECODE_ROWS_MAX` rows) would evaluate it empty: those widths gather eagerly. (The copy's arm for a pack without
/// EXL3 experts, `cfg.exl3 == null`, is no K pack's.)
fn pleDeferrable(rows: usize) bool {
    return rows <= turboexl3.kernels.DECODE_ROWS_MAX;
}

/// Claim (or clear) `entry`'s fixed spec-PLE token slot for a gather of `n` ids, reporting whether the verify history
/// fits it. ONE predicate for the eager and the deferred arm: with `ple_defer` the gather that would set `spec_ple_len`
/// runs AFTER the PLE layer's conv capture, which keys on it — so the length is claimed here, at build time, and the
/// gather only fills the tokens.
fn pleClaimSpecCapture(self: *Trunk, entry: *SSMCacheEntry, n: usize) bool {
    const ctx_len: usize = self.qwen4.?.hash.ngram_size - 1;
    return claimPleSpecCapture(entry, n, ctx_len, self.spec_capture_ssm);
}

fn claimPleSpecCapture(entry: *SSMCacheEntry, n: usize, ctx_len: usize, capturing: bool) bool {
    if (capturing and ctx_len + n <= entry.spec_ple_tokens.len) {
        entry.spec_ple_len = @intCast(ctx_len + n);
        return true;
    }
    entry.spec_ple_len = 0;
    return false;
}

/// The host side of the n-gram PLE embedding: token ids → hashed rows → gathered + bf16-packed into `pk`
/// (`[n][emb_dim]`), advancing the entry's n-gram history. `capture` comes from `pleClaimSpecCapture` at BUILD time
/// (never re-read off `spec_capture_ssm`, which a deferred flush sees already cleared).
fn pleGatherBf16(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int, pk: []u16, capture: bool) !void {
    const ids_c = try pleEvalIds(self, token_ids);
    defer _ = mlx.mlx_array_free(ids_c);
    try pleKickOff(self);
    return pleFillFromIds(self, ctx, ids_c, entry, layer, seq_len, pk, capture);
}

/// Starts the layers a prefill chunk built before its PLE layer (`Trunk.ple_kick`) on the GPU, so they run under the host
/// gather that follows; called once the ids are read, whose eval would queue behind them. The same graph, started
/// earlier: no op is added and no value changes.
fn pleKickOff(self: *Trunk) !void {
    const kick = self.ple_kick;
    if (kick.ctx == null) return;
    self.ple_kick = .{ .ctx = null };
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{kick}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_async_eval(v));
}

/// `plePrepareIds` evaluated now: the host read of the ids.
fn pleEvalIds(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const ids_c = try plePrepareIds(self, token_ids);
    errdefer _ = mlx.mlx_array_free(ids_c);
    try mlx.check(mlx.mlx_array_eval(ids_c));
    return ids_c;
}

/// The gather ids as one contiguous int32 array, still LAZY: a group evaluates every row's ids together so the flush
/// costs ONE host sync, not one per row. On the wavefront's CPU stream while it builds a chunk (`Trunk.ple_ids_stream`).
fn plePrepareIds(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const s = if (self.ple_ids_stream.ctx != null) self.ple_ids_stream else self.s;
    var ids_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_i32);
    try mlx.check(mlx.mlx_astype(&ids_i32, token_ids, .int32, s));
    var ids_c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids_c);
    try mlx.check(mlx.mlx_contiguous(&ids_c, ids_i32, false, s));
    return ids_c;
}

/// An evaluated int32 id array copied out as tokens. Caller frees.
fn pleHostIds(self: *Trunk, ids_c: mlx.mlx_array) ![]u32 {
    const n: usize = mlx.mlx_array_size(ids_c);
    const ids = try self.allocator.alloc(u32, n);
    errdefer self.allocator.free(ids);
    const src = mlx.mlx_array_data_int32(ids_c) orelse return error.TokenIdsUnreadable;
    for (0..n) |i| ids[i] = @intCast(src[i]);
    return ids;
}

/// Host-only half of the gather: `ids_c` must already be evaluated. Everything that MOVES state (the spec token slot,
/// the n-gram history) happens here.
fn pleFillFromIds(self: *Trunk, ctx: *ForwardCtx, ids_c: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int, pk: []u16, capture: bool) !void {
    const st = self.qwen4.?;
    const ctx_len: usize = st.hash.ngram_size - 1;
    const ids = try pleHostIds(self, ids_c);
    defer self.allocator.free(ids);
    const n: usize = ids.len;
    const nh = st.hash.n_heads;
    const rows = try self.allocator.alloc(i64, n * nh);
    defer self.allocator.free(rows);
    if (ctx.batch_slots) |slots| {
        const S: usize = @intCast(seq_len);
        std.debug.assert(n == slots.len * S);
        for (slots, 0..) |sc, i| {
            const e = &sc.ssm_entries.?[layer];
            std.debug.assert(e.ple_prev_valid); // batchedGdnReady
            const prev = e.ple_prev;
            const ids_i = ids[i * S .. (i + 1) * S];
            if (capture) {
                std.debug.assert(ctx_len + S <= e.spec_ple_tokens.len);
                fillPleSpecTokens(e, prev[0..ctx_len], ids_i);
                e.spec_ple_len = @intCast(ctx_len + S);
            } else e.spec_ple_len = 0;
            rowIds(&st.hash, prev[0..ctx_len], ids_i, rows[i * S * nh ..][0 .. S * nh]);
            advancePlePrev(e, prev, ids_i, ctx_len);
        }
    } else {
        const prev = pleAdvanceSerial(st, entry, ids, capture);
        rowIds(&st.hash, prev[0..ctx_len], ids, rows);
    }
    const emb_dim: usize = st.table.dim * nh;
    const host = try self.allocator.alloc(f32, n * emb_dim);
    defer self.allocator.free(host);
    // The prefetch's view is a kv-position question; `ctx.moe_seq_offset` is the pre-chunk position (`cache.step` is 0
    // forever on a GDN trunk).
    try gather(&st.table, rows, host, @intCast(ctx.moe_seq_offset.*));
    std.debug.assert(pk.len == host.len);
    for (host, pk) |v, *o| o.* = bf16Rne(v);
}

/// The history a serial entry's next ids hash against: all eos for a fresh sequence.
fn plePrevOf(entry: *const SSMCacheEntry, eos: u32) [8]u32 {
    return if (entry.ple_prev_valid) entry.ple_prev else @splat(eos);
}

/// A serial gather's state moves: the verify slot's `prev ++ ids`, then the history past `ids`. Returns the history
/// `ids` hash against.
fn pleAdvanceSerial(st: *const Qwen4State, entry: *SSMCacheEntry, ids: []const u32, capture: bool) [8]u32 {
    const ctx_len: usize = st.hash.ngram_size - 1;
    const prev = plePrevOf(entry, st.hash.eos);
    if (capture) {
        std.debug.assert(ctx_len + ids.len <= entry.spec_ple_tokens.len);
        fillPleSpecTokens(entry, prev[0..ctx_len], ids);
        entry.spec_ple_len = @intCast(ctx_len + ids.len);
    } else entry.spec_ple_len = 0;
    advancePlePrev(entry, prev, ids, ctx_len);
    return prev;
}

fn fillPleSpecTokens(entry: *SSMCacheEntry, prev: []const u32, ids: []const u32) void {
    const ctx_len = prev.len;
    for (0..ctx_len) |i| entry.spec_ple_tokens[i] = prev[i];
    for (ids, 0..) |id, j| entry.spec_ple_tokens[ctx_len + j] = id;
}

/// Advance a PLE entry's (ngram_size-1)-token history past `ids`.
fn advancePlePrev(entry: *SSMCacheEntry, prev: [8]u32, ids: []const u32, ctx_len: usize) void {
    const n = ids.len;
    const tail_from: usize = if (n >= ctx_len) n - ctx_len else 0;
    var k: usize = 0;
    if (n < ctx_len) {
        // Shift the old history left and append everything.
        for (0..ctx_len - n) |i| entry.ple_prev[i] = prev[n + i];
        k = ctx_len - n;
    }
    for (tail_from..n) |i| {
        entry.ple_prev[k] = ids[i];
        k += 1;
    }
    entry.ple_prev_valid = true;
}

// ── the verify's id lend ──

/// The deferred leaf filled from `src` (`ForwardCtx.ple_ids_src`), the arrays the verify input concatenates, instead of
/// an op over that input (`pleEvalIds`' `contiguous` queues behind everything already handed to the GPU, under the
/// verify ladder layer 0): each id is read after its OWN producer's event, and while the next one is still running the
/// rows of the ids already held are gathered. The leaf gets byte for byte what `pleGatherBf16` writes (a token's rows
/// depend only on the history and the tokens up to it, and a token-aligned range keeps each row's head), and the
/// history moves are the same call on the same ids. False, nothing read or moved, where `src` cannot stand in for
/// `token_ids`.
fn pleFillFromSources(self: *Trunk, ctx: *ForwardCtx, src: []const mlx.mlx_array, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, pk: []u16, capture: bool) !bool {
    const st = self.qwen4 orelse return false;
    const n: usize = src.len;
    const nh: usize = st.hash.n_heads;
    const emb_dim: usize = st.table.dim * nh;
    if (!pleSourcesFit(n, mlx.mlx_array_size(token_ids), pk.len, emb_dim, ctx.batch_slots == null)) return false;
    // (the copy's `pleRangesKeepHeads` is pinned true: a trellis table's heads are the hash's, `trellisHeadBias`, so a
    // token-aligned range keeps every row's head)
    for (src) |a| if (!pleSourceUsable(mlx.mlx_array_size(a), mlx.mlx_array_dtype(a))) return false;
    // anything not yet handed to the GPU goes now, in one eval: reading unscheduled ids one by one would sync per id
    {
        const v = mlx.mlx_vector_array_new_data(src.ptr, src.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }
    const ids = try self.allocator.alloc(u32, n);
    defer self.allocator.free(ids);
    const rows = try self.allocator.alloc(i64, n * nh);
    defer self.allocator.free(rows);
    const host = try self.allocator.alloc(f32, n * emb_dim);
    defer self.allocator.free(host);
    const ctx_len: usize = st.hash.ngram_size - 1;
    // the history `pleAdvanceSerial` hashes against: nothing touches the entry until that call below
    const prev = plePrevOf(entry, st.hash.eos);
    const kv_len: u64 = @intCast(ctx.moe_seq_offset.*);
    var sched: PleSrcSchedule = .{};
    for (src, 0..) |a, i| {
        var ready = false;
        _ = mlx._mlx_array_is_available(&ready, a);
        if (sched.beforeRead(ready)) |r| try pleGatherRange(st, prev[0..ctx_len], ids, r, rows, host, pk, kv_len);
        ids[i] = try pleSourceId(a);
        sched.afterRead();
    }
    if (sched.finish()) |r| try pleGatherRange(st, prev[0..ctx_len], ids, r, rows, host, pk, kv_len);
    const moved = pleAdvanceSerial(st, entry, ids, capture);
    std.debug.assert(std.mem.eql(u32, moved[0..ctx_len], prev[0..ctx_len]));
    return true;
}

/// Whether `n` source ids can stand in for a deferred leaf: one per token id, the leaf `n` rows of `emb_dim`, and a
/// solo forward (a batched one hashes per slot).
fn pleSourcesFit(n: usize, n_ids: usize, leaf_len: usize, emb_dim: usize, solo: bool) bool {
    return solo and n > 0 and n == n_ids and leaf_len == n * emb_dim;
}

/// One source id array: a single integer element (the head's `[1]`, a sampler's `[1, 1]`, int32 or uint32), what the
/// verify input concatenates.
fn pleSourceUsable(size: usize, dtype: mlx.mlx_dtype) bool {
    return size == 1 and (dtype == .int32 or dtype == .uint32);
}

/// One id off its source after the source's own event (`mlx_array_eval` on an array already handed to the GPU only
/// waits for it). A token id is non-negative and below 2^31, so its int32 and uint32 readings agree with the input's
/// `astype(int32)`.
fn pleSourceId(a: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const v: i64 = switch (mlx.mlx_array_dtype(a)) {
        .int32 => (mlx.mlx_array_data_int32(a) orelse return error.TokenIdsUnreadable)[0],
        .uint32 => (mlx.mlx_array_data_uint32(a) orelse return error.TokenIdsUnreadable)[0],
        else => return error.PleSourceDtype,
    };
    if (v < 0 or v > std.math.maxInt(i32)) return error.PleSourceRange;
    return @intCast(v);
}

/// The rows of the held ids `ids[r[0]..r[1]]`, hashed over the history and every id up to `r[1]`, gathered and
/// bf16-packed into their slice of the leaf: the rows, decode and rounding `pleFillFromIds` gives the whole range.
fn pleGatherRange(st: *const Qwen4State, prev: []const u32, ids: []const u32, r: [2]usize, rows: []i64, host: []f32, pk: []u16, kv_len: u64) !void {
    const lo = r[0];
    const hi = r[1];
    if (lo >= hi) return;
    const nh: usize = st.hash.n_heads;
    const emb_dim: usize = st.table.dim * nh;
    rowIds(&st.hash, prev, ids[0..hi], rows[0 .. hi * nh]);
    try gather(&st.table, rows[lo * nh .. hi * nh], host[lo * emb_dim .. hi * emb_dim], kv_len);
    for (host[lo * emb_dim .. hi * emb_dim], pk[lo * emb_dim .. hi * emb_dim]) |v, *o| o.* = bf16Rne(v);
}

/// When `pleFillFromSources` gathers: the ids arrive in order, and before WAITING for the next one the rows of every id
/// held but not gathered are gathered, so the host work runs while the GPU still produces the later ids; the rest goes
/// after the last id. Every id is gathered once, in order, in contiguous ranges; all readable at once is ONE range.
const PleSrcSchedule = struct {
    /// Ids read so far.
    held: usize = 0,
    /// Ids whose rows are in the leaf.
    gathered: usize = 0,

    /// Before reading the next id (`ready`: readable without waiting): the token range to gather now, if any.
    fn beforeRead(self: *PleSrcSchedule, ready: bool) ?[2]usize {
        if (ready or self.gathered == self.held) return null;
        return self.take();
    }

    fn afterRead(self: *PleSrcSchedule) void {
        self.held += 1;
    }

    /// After the last id: the range still owed, if any.
    fn finish(self: *PleSrcSchedule) ?[2]usize {
        if (self.gathered == self.held) return null;
        return self.take();
    }

    fn take(self: *PleSrcSchedule) [2]usize {
        const r = [2]usize{ self.gathered, self.held };
        self.gathered = self.held;
        return r;
    }
};

// ── the n-gram hash (`NgramHash`) ──

const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;
const PRIME_1: u64 = 10007;

fn splitmix64(v0: u64) u64 {
    var v = v0 +% SPLITMIX_GAMMA;
    v = (v ^ (v >> 30)) *% SPLITMIX_M1;
    v = (v ^ (v >> 27)) *% SPLITMIX_M2;
    return v ^ (v >> 31);
}

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

fn nthPrimeAfter(start: u64, count: u32) u64 {
    var p = start;
    for (0..count) |_| {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

/// Everything `Qwen4ExpTextNGramEmbedding.__init__` derives from the config (`NgramHash.init`, which A's load calls).
/// `ple_layer_index` is the PLE's ordinal among the config's injection points (0 for the one we support). Fallible:
/// every bound writes a fixed array.
pub fn hashInit(unigram_vocab: u32, ngram_size: u32, heads_per_ngram: u32, vocab_base: u64, divisor: u64, seed: u64, ple_layer_index: u32, eos: u32) !NgramHash {
    if (ngram_size < 2 or ngram_size > qwen4.MAX_NGRAM_SIZE) return error.InvalidQwen4NgramSize;
    if (heads_per_ngram == 0 or (ngram_size - 1) * heads_per_ngram > qwen4.MAX_HEADS) {
        return error.InvalidQwen4NgramHeads;
    }
    if (divisor == 0 or vocab_base < 2) return error.InvalidQwen4NgramVocab;
    var h: NgramHash = .{
        .ngram_size = ngram_size,
        .heads_per_ngram = heads_per_ngram,
        .n_heads = (ngram_size - 1) * heads_per_ngram,
        .eos = eos,
        .multipliers = @splat(0),
        .vocab = @splat(0),
        .offsets = @splat(0),
        .total_rows = 0,
    };
    const max_long: u64 = (1 << 63) - 1;
    const half_bound: u64 = @max(1, (max_long / @max(unigram_vocab, 1)) / 2);
    const base_seed: u64 = seed +% PRIME_1 *% ple_layer_index;
    for (0..ngram_size) |i| {
        const v = base_seed +% SPLITMIX_GAMMA *% (@as(u64, i) + 1);
        h.multipliers[i] = @intCast(2 * (splitmix64(v) % half_bound) + 1);
    }
    var total: u64 = 0;
    for (0..h.n_heads) |i| {
        const global = ple_layer_index * h.n_heads + @as(u32, @intCast(i));
        const size = nthPrimeAfter(vocab_base - 1, global + 1);
        h.vocab[i] = @intCast(size);
        h.offsets[i] = @intCast(total);
        total += size;
    }
    h.total_rows = (total + divisor - 1) / divisor * divisor;
    return h;
}

/// Row ids for `ids`, given the (ngram_size-1) tokens that precede them (`eos` for a fresh sequence). `out` is
/// `[ids.len][n_heads]` row-major. Mirrors `_shift_right_ignore_eos` + the mixed-id hash: a shifted token is `eos` when
/// the shift crosses the most recent eos before it.
fn rowIds(self: *const NgramHash, prev: []const u32, ids: []const u32, out: []i64) void {
    const ctx: usize = self.ngram_size - 1;
    std.debug.assert(prev.len == ctx and out.len == ids.len * self.n_heads);
    var last_eos: i64 = -1;
    var t: usize = 0;
    while (t < ctx + ids.len) : (t += 1) {
        const tok = tokAt(prev, ids, t);
        if (t >= ctx) {
            const seg_pos: i64 = @as(i64, @intCast(t)) - (last_eos + 1);
            var mixed: i64 = @as(i64, tok) *% self.multipliers[0];
            const row = out[(t - ctx) * self.n_heads ..][0..self.n_heads];
            var n: usize = 2;
            var pos: usize = 1;
            while (n <= self.ngram_size) : (n += 1) {
                while (pos < n) : (pos += 1) {
                    const shifted: u32 = if (seg_pos >= @as(i64, @intCast(pos)) and t >= pos) tokAt(prev, ids, t - pos) else self.eos;
                    mixed ^= @as(i64, shifted) *% self.multipliers[pos];
                }
                const h0 = (n - 2) * self.heads_per_ngram;
                for (h0..h0 + self.heads_per_ngram) |h| {
                    row[h] = @mod(mixed, self.vocab[h]) + self.offsets[h];
                }
            }
        }
        if (tok == self.eos) last_eos = @intCast(t);
    }
}

fn tokAt(prev: []const u32, ids: []const u32, i: usize) u32 {
    return if (i < prev.len) prev[i] else ids[i - prev.len];
}

// ── the table's gather and its trellis rows (`NgramTable`) ──

/// Gather + concatenate the `n_heads` rows of each token: `out` is `[ids.len / n_heads][n_heads * dim]` row-major.
/// `kv_len` is the context position this gather runs at, which the prefetch's view is keyed on; the output is
/// byte-identical either way. Trellis rows arrive token-major, head-minor (`rowIds`): row i takes head i % n_heads's
/// bias. The ring decode, not the read, is the cost, so a prefill-width gather decodes on the pool's workers.
fn gather(self: *const NgramTable, row_ids: []const i64, out: []f32, kv_len: u64) !void {
    // the copy's affine, BF16 and NVFP4 rows (its serial walk, its pooled part preads): no K pack's table
    if (self.trellis_k == 0) return error.DirectModelNgramTableNotTrellis;
    // (the copy's prefetch-mode gate is its tests' lever: off nowhere else)
    const wide = row_ids.len > MAX_ROWS;
    if (wide) if (self.pool) |p| {
        // the rows the prefetch staged for this gather decode from their staged bytes (`viewTake`)
        var view: ?AheadView = if (self.ahead) |a| beginGather(a, kv_len, row_ids) else null;
        p.ahead_view = if (view) |*v| v else null;
        runTrellis(p, self, row_ids, out);
        p.ahead_view = null;
        if (view) |*v| endGather(self.ahead.?, v);
        return;
    };
    if (!wide and self.fd >= 0) if (self.pool) |p| if (coldTrellisRows(p, self, row_ids, out)) return;
    return gatherTrellis(self, row_ids, 0, out);
}

/// A gather's trellis rows, row i (of the gather) at head `(first + i) % n_heads`.
fn gatherTrellis(self: *const NgramTable, row_ids: []const i64, first: usize, out: []f32) void {
    const heads: usize = @max(self.n_heads, 1);
    for (row_ids, first..) |r, i| trellisRowMapped(self, @intCast(r), @as(u32, @intCast(i % heads)), out[(i - first) * self.dim ..][0..self.dim]);
}

/// Whether row `r`'s trellis bytes are in the page cache (`mincore` over the pages they span).
fn rowResident(self: *const NgramTable, r: u64) bool {
    if (builtin.is_test and cold_for_test) return false;
    const nb: usize = @as(usize, self.wcols) * 2;
    const page = std.heap.pageSize();
    const off = self.w_off + @as(usize, @intCast(r)) * nb;
    const a0 = off & ~(page - 1);
    const a1 = (off + nb - 1) & ~(page - 1);
    var vec: [2]u8 = .{ 0, 0 };
    if (mincore(self.map.ptr + a0, a1 - a0 + page, &vec) != 0) return false;
    return (vec[0] & 1) != 0 and (a1 == a0 or (vec[1] & 1) != 0);
}

/// A decode step's trellis rows the page cache lacks are read with parallel preads (`coldTrellisRows`), and a prefill
/// gather's with one pread each on its worker, not faulted in through the mapping: faults on one mapping serialize on
/// the VM map lock. The bytes, so the output, are the mapping's.
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;

/// Row `r` of a trellis table decoded from the mapping, plus `head`'s bias (null: the stored ring alone).
fn trellisRowMapped(self: *const NgramTable, r: u64, head: ?u32, out: []f32) void {
    const nb: usize = @as(usize, self.wcols) * 2;
    trellisDecode(self, self.map[self.w_off + r * nb ..][0..nb], head, out);
}

/// The ring on a row's bytes (`raw`: the f16 scale word, then the dim*K-bit stream) at the table's K.
fn trellisDecode(self: *const NgramTable, raw: []const u8, head: ?u32, out: []f32) void {
    switch (self.trellis_k) {
        inline 2...8 => |k| trellisDecodeK(self, k, raw, head, out),
        else => unreachable, // `parseTrellis` admits K 2..8
    }
}

/// value_i = mul1(state_i) * scale (+ the head's bias_i): state_i stacks symbols i, i-1, ... (K bits each, circular
/// over dim) from bit 0 into a 16-bit codeword. The symbols are cut from a copy of the stream padded with its first
/// two bytes (a K <= 8 symbol spans two bytes, the ring wraps there), the states built 16 lanes at a time, MUL1
/// through `mul1_by_sum`.
fn trellisDecodeK(self: *const NgramTable, comptime K: u32, raw: []const u8, head: ?u32, out: []f32) void {
    const L = 16;
    const V = @Vector(L, u32);
    const stack: u32 = (15 + K) / K;
    const lead: usize = stack - 1;
    const kmask: u32 = (1 << K) - 1;
    const dim: usize = self.dim;
    const nbytes: usize = dim * K / 8;
    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[0..2], .little))));
    var sb: [max_trellis_dim + 2]u8 = undefined;
    @memcpy(sb[0..nbytes], raw[2..][0..nbytes]);
    sb[nbytes] = sb[0];
    sb[nbytes + 1] = sb[1];
    // syms[lead + i] = symbol i; syms[0 .. lead] = the ring's last symbols, so state i reads syms[lead + i - j]
    var syms: [8 + max_trellis_dim]u32 = undefined;
    for (0..dim) |i| {
        const bit = i * K;
        const w: u32 = std.mem.readInt(u16, sb[bit >> 3 ..][0..2], .little);
        syms[lead + i] = (w >> @intCast(bit & 7)) & kmask;
    }
    for (0..lead) |j| syms[j] = syms[dim + j];
    const bias: ?[]const f32 = if (head) |h| (if (self.head_bias) |hb| hb[@as(usize, h) * dim ..][0..dim] else null) else null;
    const vscale: @Vector(L, f32) = @splat(scale);
    var i: usize = 0;
    while (i + L <= dim) : (i += L) {
        var st: V = syms[lead + i ..][0..L].*;
        inline for (1..stack) |j| st |= @as(V, syms[lead + i - j ..][0..L].*) << @as(@Vector(L, u5), @splat(@intCast(j * K)));
        const code = st & @as(V, @splat(0xFFFF));
        const x = code *% @as(V, @splat(mul1_mult));
        const y = (x & @as(V, @splat(0x00FF00FF))) + ((x >> @as(@Vector(L, u5), @splat(8))) & @as(V, @splat(0x00FF00FF)));
        const t = (y & @as(V, @splat(0xFFFF))) + (y >> @as(@Vector(L, u5), @splat(16)));
        var m: @Vector(L, f32) = undefined;
        inline for (0..L) |k| m[k] = mul1_by_sum[t[k]];
        var v = m * vscale;
        if (bias) |b| v += @as(@Vector(L, f32), b[i..][0..L].*);
        out[i..][0..L].* = v;
    }
    while (i < dim) : (i += 1) {
        var state: u32 = syms[lead + i];
        inline for (1..stack) |j| state |= syms[lead + i - j] << @intCast(j * K);
        var v: f32 = mul1_by_sum[mul1ByteSum(@truncate(state))] * scale;
        if (bias) |b| v += b[i];
        out[i] = v;
    }
}

/// The widest trellis ring a table may have (`parseTrellis` admits no wider): the decode's stack buffers.
const max_trellis_dim = 1024;

/// exllamav3's MUL1 multiplier: a codeword's weight depends on the byte sum of `code * mul1_mult` alone.
const mul1_mult: u32 = turboexl3.format.MUL1_MULT;

/// MUL1's codeword hash reduced to its byte sum t in [0, 1020].
fn mul1ByteSum(code: u16) u32 {
    const x = @as(u32, code) *% mul1_mult;
    const y = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
    return (y & 0xFFFF) + (y >> 16);
}

/// MUL1's weight per byte sum t, widened to f32: f16(fma(1024 + t, half 0x1EEE, half 0xC931)).
const mul1_by_sum: [1021]f32 = blk: {
    @setEvalBranchQuota(100_000);
    var lut: [1021]f32 = undefined;
    const inverse: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
    const bias: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
    for (&lut, 0..) |*v, t| {
        const h: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0x6400 + t))));
        const q: f16 = @floatCast(@mulAdd(f32, h, inverse, bias));
        v.* = @floatCast(q);
    }
    break :blk lut;
};

/// Round-to-nearest-even f32 -> bf16 bits, the PLE rows' upload format.
fn bf16Rne(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

// ── the gather's workers (`PrefetchPool`) ──
//
// Every row's bytes are one SSD read on the cold table (~100 us), 16 per token: serial mmap faults were ~5 ms of every
// decode step, and more fault threads got SLOWER (faults on one mapping serialize on the VM map lock), so the workers
// `pread` instead. Workers wake on a generation bump and count themselves down; the caller spins (the job is ~100 us).

/// The table's 48 workers, for A's `open` (the copy's `PrefetchPool.create`).
pub fn poolCreate() !*PrefetchPool {
    const a = std.heap.page_allocator;
    const p = try a.create(PrefetchPool);
    p.* = .{};
    var started: usize = 0;
    errdefer {
        poolShutdown(p, started);
        a.destroy(p);
    }
    for (0..POOL_N) |i| {
        // Past std's 256 KiB thread-local signal stack: glibc carves static TLS out of the thread's stack.
        p.threads[i] = try std.Thread.spawn(.{ .stack_size = 1024 * 1024 }, poolWorker, .{ p, i });
        started += 1;
    }
    return p;
}

/// The workers joined and freed, for A's `close`.
pub fn poolDestroy(self: *PrefetchPool) void {
    poolShutdown(self, POOL_N);
    std.heap.page_allocator.destroy(self);
}

fn poolShutdown(self: *PrefetchPool, started: usize) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    self.quit = true;
    self.cv.broadcast(io);
    self.mu.unlock(io);
    for (self.threads[0..started]) |t| t.join();
}

/// Decode a trellis table's rows on the workers, row i into `out[i * dim ..]` at head i % n_heads.
fn runTrellis(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, out: []f32) void {
    _ = poolDispatch(self, table, rows, .trellis, out);
}

/// A decode step's trellis rows (at most MAX_ROWS): those the page cache lacks are read by the workers in parallel and
/// decoded from `bufs`, the rest from the mapping. False when none is cold or a read came up short.
fn coldTrellisRows(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, out: []f32) bool {
    const nb: usize = @as(usize, table.wcols) * 2;
    if (rows.len > MAX_ROWS or nb > ROW_BUF or nb == 0) return false;
    var n_cold: usize = 0;
    for (rows, 0..) |r, i| {
        self.cold[i] = !rowResident(table, @intCast(r));
        if (self.cold[i]) n_cold += 1;
    }
    if (n_cold == 0) return false;
    if (!poolDispatch(self, table, rows, .trellis_cold, &.{})) return false;
    const dim: usize = table.dim;
    const heads: usize = @max(table.n_heads, 1);
    for (rows, 0..) |r, i| {
        const head: u32 = @intCast(i % heads);
        const o = out[i * dim ..][0..dim];
        if (self.cold[i]) trellisDecode(table, self.bufs[i][0..nb], head, o) else trellisRowMapped(table, @intCast(r), head, o);
    }
    return true;
}

/// One round on every worker: `trellis` hands each a contiguous share of the rows to decode into `out` (any count of
/// rows); `trellis_cold` preads the rows `cold` marks into `bufs`. False when a read came up short.
fn poolDispatch(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, job: PoolJob, out: []f32) bool {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    self.table = table;
    self.rows = rows;
    self.job = job;
    self.out = out;
    self.failed.store(0, .release);
    self.pending.store(POOL_N, .release);
    self.gen += 1;
    self.cv.broadcast(io);
    self.mu.unlock(io);
    while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
    return self.failed.load(.acquire) == 0;
}

fn poolWorker(self: *PrefetchPool, idx: usize) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var seen: u64 = 0;
    while (true) {
        self.mu.lockUncancelable(io);
        while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
        if (self.quit) {
            self.mu.unlock(io);
            return;
        }
        seen = self.gen;
        const table = self.table.?;
        const rows = self.rows;
        const job = self.job;
        const out = self.out;
        const view = self.ahead_view;
        self.mu.unlock(io);
        switch (job) {
            // the copy's part preads of an affine or BF16 table's rows: the direct model's gather never asks for them,
            // so a round that does fails
            .pread => _ = self.failed.fetchAdd(1, .acq_rel),
            .trellis_cold => {
                const nb: usize = @as(usize, table.wcols) * 2;
                var i = idx;
                while (i < rows.len) : (i += POOL_N) {
                    if (!self.cold[i]) continue;
                    const off = table.w_off + @as(usize, @intCast(rows[i])) * nb;
                    if (crt.pread(table.fd, &self.bufs[i], nb, @intCast(off)) != @as(isize, @intCast(nb))) _ = self.failed.fetchAdd(1, .acq_rel);
                }
            },
            .trellis => {
                const dim: usize = table.dim;
                const heads: usize = @max(table.n_heads, 1);
                const nb: usize = @as(usize, table.wcols) * 2;
                const preads = table.fd >= 0 and nb <= ROW_BUF;
                const share = (rows.len + POOL_N - 1) / POOL_N;
                const end = @min((idx + 1) * share, rows.len);
                var buf: [ROW_BUF]u8 = undefined;
                var i = @min(idx * share, rows.len);
                while (i < end) : (i += 1) {
                    const r: u64 = @intCast(rows[i]);
                    const head: u32 = @intCast(i % heads);
                    const o = out[i * dim ..][0..dim];
                    // a row the prefetch staged, or one the page cache lacks read with a pread: the file's bytes
                    if (view) |v| if (viewTake(v, i, rows[i])) |raw| {
                        trellisDecode(table, raw, head, o);
                        continue;
                    };
                    if (preads and !rowResident(table, r) and crt.pread(table.fd, &buf, nb, @intCast(table.w_off + r * nb)) == @as(isize, @intCast(nb))) {
                        trellisDecode(table, buf[0..nb], head, o);
                        continue;
                    }
                    trellisRowMapped(table, r, head, o);
                }
            },
        }
        _ = self.pending.fetchSub(1, .acq_rel);
    }
}

// ── the PLE prefetch: a prefill's trellis rows, read ahead of its gathers (`Ahead`, `AheadSession`, `AheadView`) ──

/// Positions per plan segment, the unit a ring slot holds.
const AHEAD_SEG_TOKENS: usize = 4096;
/// Rows per reader task; every segment is a whole number of them (`aheadGeometry` sizes segments in 64-token steps).
const AHEAD_BATCH: usize = 64;
/// A shorter prompt is one small gather with nothing to hide behind: no session.
const AHEAD_MIN_TOKENS: usize = 1024;
/// The staging ring's byte budget; at least two segments are kept whatever it holds.
const AHEAD_BUDGET: usize = 64 << 20;
/// A slot holding no segment (`AheadSlot.seg`'s default).
const AH_NO_SEG: usize = std.math.maxInt(usize);

/// A planned row's state: FREE -> READING (a reader claimed it) -> STAGED or FAILED (a short read), or FREE -> TAKEN (the
/// gather got there first and reads it itself).
const AH_FREE: u8 = 0;
const AH_READING: u8 = 1;
const AH_STAGED: u8 = 2;
const AH_TAKEN: u8 = 3;
const AH_FAILED: u8 = 4;

/// How long a gather waits for a row a reader is reading before it reads the row itself.
const AHEAD_WAIT_NS: u64 = 2_000_000;

/// A prefill of `ids`, its first token at kv position `base` after the n-gram history `prev`, is about to run: every
/// chunk's rows are planned and read in the background, in position order, so each chunk's gather finds them staged.
/// Ends any open session first. A trellis table on the pooled gather only, and a prompt long enough to hide behind.
/// (The copy's `NgramTable.prefetchBegin`, which the serving glue calls; its prefetch-mode gate is its tests' lever.)
pub fn prefetchBegin(self: *NgramTable, hash: *const NgramHash, prev: []const u32, ids: []const u32, base: u64) void {
    prefetchEnd(self);
    // without one of these (or without threads or memory) each gather reads its own rows
    if (self.trellis_k == 0 or self.fd < 0 or self.pool == null) return;
    if (ids.len < AHEAD_MIN_TOKENS or hash.n_heads == 0 or prev.len != hash.ngram_size - 1) return;
    const nb: usize = @as(usize, self.wcols) * 2;
    if (nb == 0 or nb > ROW_BUF) return;
    const a = self.ahead orelse blk: {
        const made = aheadCreate() catch return;
        self.ahead = made;
        break :blk made;
    };
    aheadBegin(a, .{ .hash = hash, .prev = prev, .ids = ids, .base = base, .fd = self.fd, .w_off = self.w_off, .nb = nb, .budget = AHEAD_BUDGET }) catch return;
}

/// Ends the open prefetch session, if any; each reader finishes its row in flight. The prefill calls it on every exit.
fn prefetchEnd(self: *NgramTable) void {
    if (self.ahead) |a| aheadEnd(a);
}

/// Whether a prefill chunk starts its layers before the PLE layer ahead of the host gather (`Trunk.ple_kick`): while a
/// prefetch session is open.
pub fn prefetchKick(self: *const NgramTable) bool {
    const a = self.ahead orelse return false;
    return a.open.load(.acquire);
}

/// Row ids of positions [a, b) of `ids`, which follow the (ngram_size-1)-token history `prev0`: `rowIds` over `ids[a..b]`
/// given the history a chunked gather holds at `a` (`advancePlePrev`). A row depends only on its token and the
/// ngram_size-1 before it, so every chunking hashes the same rows. `out` is `[(b - a) * n_heads]`.
fn planRows(hash: *const NgramHash, prev0: []const u32, ids: []const u32, a: usize, b: usize, out: []i64) void {
    const ctx: usize = hash.ngram_size - 1;
    std.debug.assert(prev0.len == ctx and a <= b and b <= ids.len);
    var prev: [qwen4.MAX_NGRAM_SIZE]u32 = undefined;
    for (0..ctx) |j| {
        const g = a + j; // an index into prev0 ++ ids
        prev[j] = if (g < ctx) prev0[g] else ids[g - ctx];
    }
    rowIds(hash, prev[0..ctx], ids[a..b], out);
}

/// Segments and ring slots for an `n_tokens` prompt of `nh` rows a token, `nb` bytes a row: `AHEAD_SEG_TOKENS`-token
/// segments (one, rounded up to 64 tokens, for a shorter prompt), as many slots as `budget` holds, at least two and at
/// most one per segment. `bytes` is what the session allocates for them.
fn aheadGeometry(n_tokens: usize, nh: usize, nb: usize, budget: usize) qwen4.AheadGeometry {
    const seg_tokens = @min(AHEAD_SEG_TOKENS, std.mem.alignForward(usize, @max(n_tokens, 1), 64));
    const seg_rows = seg_tokens * nh;
    const n_segs = (n_tokens + seg_tokens - 1) / seg_tokens;
    const slot_bytes = seg_rows * (nb + @sizeOf(i64) + 1);
    const slots = @min(n_segs, @max(2, budget / @max(slot_bytes, 1)));
    return .{ .seg_tokens = seg_tokens, .seg_rows = seg_rows, .n_segs = n_segs, .slots = slots, .bytes = slots * slot_bytes };
}

/// A stride coprime with `n`: the help sweep visits batch (h * stride) % n for h in [0, n), every batch once, spread over
/// the gather's range.
fn aheadStride(n: usize) usize {
    if (n <= 2) return 1;
    var s: usize = @max(1, n * 618 / 1000);
    while (std.math.gcd(s, n) != 1) s += 1;
    return s;
}

const AheadPlan = struct { hash: *const NgramHash, prev: []const u32, ids: []const u32, base: u64, fd: c_int, w_off: usize, nb: usize, budget: usize };

const AheadTask = union(enum) {
    plan: struct { seg: usize, slot: *AheadSlot },
    read: struct { slot: *AheadSlot, lo: usize, hi: usize },
};

/// One prefill's plan and staging ring, from `prefetchBegin` to `prefetchEnd`: page-allocator memory bounded by the
/// budget, plus the prompt's ids.
fn sessionCreate(plan: AheadPlan) !*AheadSession {
    const pa = std.heap.page_allocator;
    const nh: usize = plan.hash.n_heads;
    const geo = aheadGeometry(plan.ids.len, nh, plan.nb, plan.budget);
    if (geo.slots == 0 or plan.prev.len > qwen4.MAX_NGRAM_SIZE) return error.EmptyPrefetchPlan;
    const s = try pa.create(AheadSession);
    errdefer pa.destroy(s);
    const tokens = try pa.dupe(u32, plan.ids);
    errdefer pa.free(tokens);
    const cap = geo.slots * geo.seg_rows;
    const rows_mem = try pa.alloc(i64, cap);
    errdefer pa.free(rows_mem);
    const state_mem = try pa.alloc(std.atomic.Value(u8), cap);
    errdefer pa.free(state_mem);
    const bytes_mem = try pa.alloc(u8, cap * plan.nb);
    errdefer pa.free(bytes_mem);
    const slots = try pa.alloc(AheadSlot, geo.slots);
    for (slots, 0..) |*sl, i| sl.* = .{
        .rows = rows_mem[i * geo.seg_rows ..][0..geo.seg_rows],
        .state = state_mem[i * geo.seg_rows ..][0..geo.seg_rows],
        .bytes = bytes_mem[i * geo.seg_rows * plan.nb ..][0 .. geo.seg_rows * plan.nb],
    };
    s.* = .{
        .hash = plan.hash.*,
        .tokens = tokens,
        .base = plan.base,
        .nh = nh,
        .nb = plan.nb,
        .fd = plan.fd,
        .w_off = plan.w_off,
        .seg_tokens = geo.seg_tokens,
        .seg_rows = geo.seg_rows,
        .n_segs = geo.n_segs,
        .slots = slots,
        .rows_mem = rows_mem,
        .state_mem = state_mem,
        .bytes_mem = bytes_mem,
    };
    @memcpy(s.prev0[0..plan.prev.len], plan.prev);
    return s;
}

fn sessionDeinit(s: *AheadSession) void {
    const pa = std.heap.page_allocator;
    pa.free(s.slots);
    pa.free(s.bytes_mem);
    pa.free(s.state_mem);
    pa.free(s.rows_mem);
    pa.free(s.tokens);
    pa.destroy(s);
}

fn totalRows(s: *const AheadSession) usize {
    return s.tokens.len * s.nh;
}

/// One past segment `seg`'s last position.
fn segEnd(s: *const AheadSession, seg: usize) usize {
    return @min((seg + 1) * s.seg_tokens, s.tokens.len);
}

fn segRowCount(s: *const AheadSession, seg: usize) usize {
    return (segEnd(s, seg) - seg * s.seg_tokens) * s.nh;
}

fn slotOf(s: *AheadSession, seg: usize) *AheadSlot {
    return &s.slots[seg % s.slots.len];
}

/// The next task (under `Ahead.mu`), or null: nothing to do now.
fn pick(s: *AheadSession) ?AheadTask {
    if (s.stop.load(.acquire)) return null;
    // while a gather runs, its own rows only: no IO or CPU spent past it while it is the critical path
    if (s.gathering) return pickHelp(s);
    if (pickPlan(s)) |t| return t;
    return pickBackground(s);
}

fn pickPlan(s: *AheadSession) ?AheadTask {
    if (s.planning) return null;
    while (s.planned < s.n_segs and segEnd(s, s.planned) <= s.consumed) s.planned += 1;
    if (s.planned >= s.n_segs) return null;
    const seg = s.planned;
    const slot = slotOf(s, seg);
    if (slot.seg != AH_NO_SEG) {
        // a full ring: its oldest slot is reused once every gather is past that segment and no reader is in it
        if (slot.inflight != 0 or segEnd(s, slot.seg) > s.consumed) return null;
        slot.seg = AH_NO_SEG;
    }
    s.planning = true;
    return .{ .plan = .{ .seg = seg, .slot = slot } };
}

fn pickBackground(s: *AheadSession) ?AheadTask {
    s.bg_next = @max(s.bg_next, s.consumed * s.nh);
    if (s.bg_next >= totalRows(s)) return null;
    const seg = s.bg_next / s.seg_rows;
    const slot = slotOf(s, seg);
    if (slot.seg != seg) return null; // not planned yet: the planner's publish wakes the readers
    const lo = s.bg_next - seg * s.seg_rows;
    const hi = @min(lo + AHEAD_BATCH, segRowCount(s, seg));
    s.bg_next = seg * s.seg_rows + hi;
    slot.inflight += 1;
    return .{ .read = .{ .slot = slot, .lo = lo, .hi = hi } };
}

fn pickHelp(s: *AheadSession) ?AheadTask {
    while (s.coop_next < s.coop_n) {
        const h = s.coop_next;
        s.coop_next += 1;
        const b = s.coop_first + (h * s.coop_stride) % s.coop_n;
        const lo_q = @max(b * AHEAD_BATCH, s.coop_lo);
        const hi_q = @min((b + 1) * AHEAD_BATCH, s.coop_hi);
        if (lo_q >= hi_q) continue;
        const seg = lo_q / s.seg_rows; // a batch never straddles two segments (seg_rows is a multiple of it)
        const slot = slotOf(s, seg);
        if (slot.seg != seg) continue; // not planned: the gather reads those rows itself
        slot.inflight += 1;
        return .{ .read = .{ .slot = slot, .lo = lo_q - seg * s.seg_rows, .hi = hi_q - seg * s.seg_rows } };
    }
    return null;
}

/// Outside the lock: a plan task owns its unpublished slot, a read task claims each row before it touches it.
fn sessionRun(s: *AheadSession, t: AheadTask) void {
    switch (t) {
        .plan => |p| runPlan(s, p.seg, p.slot),
        .read => |r| runRead(s, r.slot, r.lo, r.hi),
    }
}

fn runPlan(s: *AheadSession, seg: usize, slot: *AheadSlot) void {
    const a = seg * s.seg_tokens;
    const b = segEnd(s, seg);
    @memset(slot.state[0 .. (b - a) * s.nh], std.atomic.Value(u8).init(AH_FREE));
    const ctx: usize = s.hash.ngram_size - 1;
    var p = a;
    while (p < b) {
        if (s.stop.load(.monotonic)) return;
        const q = @min(p + 512, b);
        planRows(&s.hash, s.prev0[0..ctx], s.tokens, p, q, slot.rows[(p - a) * s.nh .. (q - a) * s.nh]);
        p = q;
    }
}

fn runRead(s: *AheadSession, slot: *AheadSlot, lo: usize, hi: usize) void {
    var k = lo;
    while (k < hi) : (k += 1) {
        if (s.stop.load(.monotonic)) break;
        const st = &slot.state[k];
        if (st.cmpxchgStrong(AH_FREE, AH_READING, .monotonic, .monotonic) != null) continue;
        const r: usize = @intCast(slot.rows[k]);
        const dst = slot.bytes[k * s.nb ..][0..s.nb];
        const got = crt.pread(s.fd, dst.ptr, s.nb, @intCast(s.w_off + r * s.nb)) == @as(isize, @intCast(s.nb));
        st.store(if (got) AH_STAGED else AH_FAILED, .release);
    }
}

/// A task is done (under `Ahead.mu`). True when a waiter may proceed: a new segment, a slot the planner may reuse, or no
/// reader left for `aheadEnd`.
fn sessionFinish(s: *AheadSession, t: AheadTask) bool {
    s.busy -= 1;
    switch (t) {
        .plan => |p| {
            s.planning = false;
            if (!s.stop.load(.acquire)) {
                p.slot.seg = p.seg;
                s.planned = @max(s.planned, p.seg + 1);
            }
            return true;
        },
        .read => |r| {
            r.slot.inflight -= 1;
            return s.busy == 0 or (r.slot.inflight == 0 and s.planned < s.n_segs);
        },
    }
}

/// The staged bytes of the gather's row `i` (row id `r`), or null: the gather reads that row the plain way. A row no
/// reader has claimed is claimed here, so no reader reads it after; one a reader is reading is waited for, at most
/// `AHEAD_WAIT_NS`. A row whose planned id is not `r` is left alone. The pool's workers call it concurrently.
fn viewTake(v: *AheadView, i: usize, r: i64) ?[]const u8 {
    const q = v.q0 + i;
    if (q >= v.q_end) return null;
    const s = v.sess;
    const seg = q / s.seg_rows;
    const si = seg - v.first_seg;
    if (si >= v.n_slots) return null;
    const slot = v.slots[si] orelse return null;
    const k = q - seg * s.seg_rows;
    if (slot.rows[k] != r) return null;
    const st = &slot.state[k];
    var cur = st.load(.acquire);
    if (cur == AH_FREE) cur = st.cmpxchgStrong(AH_FREE, AH_TAKEN, .acquire, .acquire) orelse return null;
    if (cur == AH_READING) cur = waitRead(st);
    if (cur != AH_STAGED) return null;
    return slot.bytes[k * s.nb ..][0..s.nb];
}

fn waitRead(st: *std.atomic.Value(u8)) u8 {
    const sw = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    var spins: u32 = 0;
    while (true) {
        const cur = st.load(.acquire);
        if (cur != AH_READING) return cur;
        spins +%= 1;
        if (spins % 64 == 0 and sw.read() >= AHEAD_WAIT_NS) return AH_READING;
        std.atomic.spinLoopHint();
    }
}

/// The prefetch's 32 readers, for A's `open` and for `prefetchBegin` when the table has none (the copy's `Ahead.create`):
/// the prompt's rows are planned (`planRows`) and `pread` in position order into a bounded ring of segments while
/// earlier chunks run; a gather takes a staged row only where its planned id is the one it asks for (`viewTake`), so the
/// output is the same bytes. The inference thread opens and ends sessions and runs the gathers; the readers plan and
/// read, and while a gather runs they only help it. All but a row's state and bytes is under `mu`.
pub fn aheadCreate() !*Ahead {
    const pa = std.heap.page_allocator;
    const self = try pa.create(Ahead);
    self.* = .{};
    var started: usize = 0;
    errdefer {
        aheadShutdown(self, started);
        pa.destroy(self);
    }
    for (0..AHEAD_READERS) |i| {
        // the pool's stack size: glibc carves static TLS out of a thread's stack
        self.threads[i] = try std.Thread.spawn(.{ .stack_size = 1024 * 1024 }, aheadWorkerMain, .{self});
        started += 1;
    }
    self.n_threads = started;
    return self;
}

fn aheadShutdown(self: *Ahead, started: usize) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    self.quit = true;
    self.cv.broadcast(io);
    self.mu.unlock(io);
    for (self.threads[0..started]) |t| t.join();
}

fn aheadWorkerMain(self: *Ahead) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    while (!self.quit) {
        const s = self.session orelse {
            self.cv.waitUncancelable(io, &self.mu);
            continue;
        };
        const task = pick(s) orelse {
            self.cv.waitUncancelable(io, &self.mu);
            continue;
        };
        s.busy += 1;
        self.mu.unlock(io);
        sessionRun(s, task);
        self.mu.lockUncancelable(io);
        if (sessionFinish(s, task)) self.cv.broadcast(io);
    }
    self.mu.unlock(io);
}

/// Opens a session for one prefill and wakes the readers. Fails (nothing changes) without memory or with one open.
fn aheadBegin(self: *Ahead, plan: AheadPlan) !void {
    const s = try sessionCreate(plan);
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    const busy = self.session != null;
    if (!busy) {
        self.session = s;
        self.open.store(true, .release);
        self.cv.broadcast(io);
    }
    self.mu.unlock(io);
    if (busy) {
        sessionDeinit(s);
        return error.PrefetchSessionOpen;
    }
}

/// Stops and frees the open session, if any: each reader finishes the row it is reading, a running gather completes.
fn aheadEnd(self: *Ahead) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    const s = self.session orelse {
        self.mu.unlock(io);
        return;
    };
    s.stop.store(true, .release);
    self.cv.broadcast(io);
    while (s.busy != 0 or s.gathers != 0) self.cv.waitUncancelable(io, &self.mu);
    self.session = null;
    self.open.store(false, .release);
    self.mu.unlock(io);
    sessionDeinit(s);
}

/// A gather of `rows` (token-major, `nh` a token) at `kv_len` is starting: its view of the session, or null when it does
/// not engage. One that is not this plan's (another position, a co-tenant's rows, a position already gathered) changes
/// nothing; one that is but finds its first segment not in the ring yet still moves the cursors past its positions.
fn beginGather(self: *Ahead, kv_len: u64, rows: []const i64) ?AheadView {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    defer self.mu.unlock(io);
    const s = self.session orelse return null;
    if (s.stop.load(.acquire) or s.gathering) return null;
    if (rows.len < s.nh or rows.len % s.nh != 0 or kv_len < s.base) return null;
    const pos: usize = @intCast(kv_len - s.base);
    // forward only: the ring may already hold later segments where earlier positions were
    if (pos < s.consumed or pos >= s.tokens.len) return null;
    const end_pos = @min(pos + rows.len / s.nh, s.tokens.len);
    var v: AheadView = .{ .sess = s, .q0 = pos * s.nh, .q_end = end_pos * s.nh, .end_pos = end_pos, .first_seg = pos / s.seg_tokens };
    var seg = v.first_seg;
    while (seg * s.seg_tokens < end_pos and v.n_slots < AHEAD_VIEW_SEGS) : (seg += 1) {
        const slot = slotOf(s, seg);
        v.slots[v.n_slots] = if (slot.seg == seg) slot else null;
        v.n_slots += 1;
    }
    // this plan's own gather: its first token's planned rows are its rows (a co-tenant's at this position are not)
    const slot0 = v.slots[0] orelse {
        // Not in the ring yet: the gather reads its rows itself. When its first token's planned rows are its rows,
        // the cursors still move past it, or a chunk as wide as the ring would keep the planner blocked.
        const hist: usize = s.hash.ngram_size - 1;
        var want: [qwen4.MAX_HEADS]i64 = undefined;
        planRows(&s.hash, s.prev0[0..hist], s.tokens, pos, pos + 1, want[0..s.nh]);
        if (!std.mem.eql(i64, want[0..s.nh], rows[0..s.nh])) return null;
        s.consumed = @max(s.consumed, end_pos);
        s.bg_next = @max(s.bg_next, v.q_end);
        self.cv.broadcast(io);
        return null;
    };
    const k0 = v.q0 - v.first_seg * s.seg_rows;
    if (!std.mem.eql(i64, slot0.rows[k0..][0..s.nh], rows[0..s.nh])) return null;
    s.gathering = true;
    s.gathers += 1;
    s.coop_lo = @max(v.q0, s.bg_next);
    s.coop_hi = v.q_end;
    s.coop_next = 0;
    s.coop_n = 0;
    if (s.coop_lo < s.coop_hi) {
        s.coop_first = s.coop_lo / AHEAD_BATCH;
        s.coop_n = (s.coop_hi - 1) / AHEAD_BATCH - s.coop_first + 1;
        s.coop_stride = aheadStride(s.coop_n);
    }
    self.cv.broadcast(io);
    return v;
}

/// The gather `v` is done (its pool round returned): the gathers are past its positions, so the readers move on.
fn endGather(self: *Ahead, v: *AheadView) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    self.mu.lockUncancelable(io);
    const s = v.sess;
    s.gathering = false;
    s.gathers -= 1;
    s.coop_n = 0;
    s.consumed = @max(s.consumed, v.end_pos);
    s.bg_next = @max(s.bg_next, v.q_end);
    self.cv.broadcast(io);
    self.mu.unlock(io);
}

// ── tests ──

const testing = std.testing;
const load = @import("load.zig");

/// Tests: every row counts as not resident, so the gathers read it with a pread (a test build's only).
var cold_for_test = false;

/// The trellis ring one symbol at a time (the `exl3_ngram_trellis` definition): value_i = MUL1(state_i) * scale
/// + head's bias_i, state_i stacking symbols i, i-1, ... K bits each from bit 0, through turboexl3's `decodeMul1`.
fn trellisRowRef(t: *const NgramTable, r: u64, head: ?u32, out: []f32) void {
    const words: usize = t.wcols;
    const K = t.trellis_k;
    const raw = t.map[t.w_off + r * words * 2 ..][0 .. words * 2];
    const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[0..2], .little))));
    const nstream: u32 = @intCast(words - 1);
    const kmask: u32 = (@as(u32, 1) << @intCast(K)) - 1;
    var syms: [max_trellis_dim]u16 = undefined;
    for (0..t.dim) |i| {
        const bit: u32 = @as(u32, @intCast(i)) * K;
        const w = bit >> 4;
        const lo: u32 = std.mem.readInt(u16, raw[2 + 2 * w ..][0..2], .little);
        const hi: u32 = std.mem.readInt(u16, raw[2 + 2 * ((w + 1) % nstream) ..][0..2], .little);
        syms[i] = @intCast(((lo | (hi << 16)) >> @intCast(bit & 15)) & kmask);
    }
    const stack: u32 = (15 + K) / K;
    for (0..t.dim) |i| {
        var state: u32 = 0;
        for (0..stack) |j| state |= @as(u32, syms[(i + t.dim - j) % t.dim]) << @intCast(j * K);
        const decoded = turboexl3.format.decodeMul1(@truncate(state));
        var v: f32 = @as(f32, @floatCast(@as(f16, @bitCast(decoded)))) * scale;
        if (head) |h| if (t.head_bias) |hb| {
            v += hb[@as(usize, h) * t.dim + i];
        };
        out[i] = v;
    }
}

/// An `exl3_ngram_trellis` file: `rows` rings of `dim` K-bit symbols (random stream bits, a finite f16 scale word per
/// row) behind a 512-byte safetensors header. The caller frees the bytes.
fn trellisFixture(a: std.mem.Allocator, rnd: std.Random, rows: usize, dim: u32, K: u32) ![]u8 {
    const words: usize = 1 + dim * K / 16;
    const HDR: usize = 512;
    var hbuf: [HDR]u8 = undefined;
    const header = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"exl3_ngram_trellis\",\"K\":\"{d}\",\"row_dim\":\"{d}\",\"codebook\":\"mul1\"}}," ++
        "\"ple.ngram_embedding.trellis\":{{\"dtype\":\"I16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ K, dim, rows, words, rows * words * 2 });
    const buf = try a.alloc(u8, 8 + HDR + rows * words * 2);
    std.mem.writeInt(u64, buf[0..8], HDR, .little);
    @memset(buf[8 .. 8 + HDR], ' ');
    @memcpy(buf[8..][0..header.len], header);
    const data = buf[8 + HDR ..];
    for (0..rows) |r| {
        const row_bytes = data[r * words * 2 ..][0 .. words * 2];
        rnd.bytes(row_bytes);
        const scale: f16 = @floatCast(0.25 + 2.0 * rnd.float(f32));
        std.mem.writeInt(u16, row_bytes[0..2], @bitCast(scale), .little);
    }
    return buf;
}

fn trellisBias(a: std.mem.Allocator, rnd: std.Random, heads: u32, dim: u32) ![]u16 {
    const bias = try a.alloc(u16, @as(usize, heads) * dim);
    for (bias) |*b| b.* = @bitCast(@as(f16, @floatCast(rnd.floatNorm(f32) * 0.5)));
    return bias;
}

/// `file` (a `trellisFixture`) in `td`, opened as the load opens a pack's table.
fn openFixture(td: *std.testing.TmpDir, file: []const u8) !NgramTable {
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_embedding.safetensors", .data = file });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var out: [std.fs.max_path_bytes]u8 = undefined;
    return load.openTable(try std.fmt.bufPrint(&out, "{s}/ngram_embedding.safetensors", .{pbuf[0..root_len]}));
}

/// `n` random row ids of a `rows`-row table, the last repeating the first.
fn randomIds(a: std.mem.Allocator, rnd: std.Random, n: usize, rows: usize) ![]i64 {
    const ids = try a.alloc(i64, n);
    for (ids) |*id| id.* = @intCast(rnd.uintLessThan(u64, rows));
    ids[n - 1] = ids[0];
    return ids;
}

/// The definition's rows for `ids` (token-major, head-minor).
fn refRows(a: std.mem.Allocator, t: *const NgramTable, ids: []const i64, heads: u32) ![]f32 {
    const ref = try a.alloc(f32, ids.len * t.dim);
    for (ids, 0..) |r, i| trellisRowRef(t, @intCast(r), @as(u32, @intCast(i % heads)), ref[i * t.dim ..][0..t.dim]);
    return ref;
}

test "direct ple: mul1_by_sum is the EXL3 MUL1 weight at every codeword" {
    var code: u32 = 0;
    while (code <= 0xFFFF) : (code += 1) {
        const want: f32 = @floatCast(@as(f16, @bitCast(turboexl3.format.decodeMul1(@intCast(code)))));
        try testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(mul1_by_sum[mul1ByteSum(@intCast(code))])));
    }
}

test "direct ple: the decoded ring equals the one-symbol definition bit for bit at every K, row and head" {
    const Shape = struct { k: u32, dim: u32 };
    // the served K6 x 160, and every other admitted K at a dim that leaves a scalar tail where it can
    const shapes = [_]Shape{ .{ .k = 6, .dim = 160 }, .{ .k = 2, .dim = 24 }, .{ .k = 3, .dim = 48 }, .{ .k = 4, .dim = 36 }, .{ .k = 5, .dim = 64 }, .{ .k = 7, .dim = 32 }, .{ .k = 8, .dim = 40 } };
    var prng = std.Random.DefaultPrng.init(0x7E11);
    const rnd = prng.random();
    const a = testing.allocator;
    for (shapes) |sh| {
        const rows = 37;
        const heads = 5;
        const file = try trellisFixture(a, rnd, rows, sh.dim, sh.k);
        defer a.free(file);
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        var t = try openFixture(&td, file);
        defer t.close();
        try testing.expectEqual(sh.k, t.trellis_k);
        try testing.expectEqual(sh.dim, t.dim);
        var ref: [max_trellis_dim]f32 = undefined;
        var got: [max_trellis_dim]f32 = undefined;
        // the stored rings, then each head's
        for (0..rows) |r| {
            trellisRowRef(&t, r, null, ref[0..sh.dim]);
            trellisRowMapped(&t, r, null, got[0..sh.dim]);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0..sh.dim]), std.mem.sliceAsBytes(got[0..sh.dim]));
        }
        const bias = try trellisBias(a, rnd, heads, sh.dim);
        defer a.free(bias);
        try load.setHeadBias(&t, bias, heads);
        for (0..rows) |r| for (0..heads) |h| {
            trellisRowRef(&t, r, @as(u32, @intCast(h)), ref[0..sh.dim]);
            trellisRowMapped(&t, r, @as(u32, @intCast(h)), got[0..sh.dim]);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref[0..sh.dim]), std.mem.sliceAsBytes(got[0..sh.dim]));
        };
    }
}

test "direct ple: a prefill gather on the pool and a decode step's serial gather equal the definition bit for bit" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x900D);
    const rnd = prng.random();
    const rows = 4096;
    const heads: u32 = 16;
    const file = try trellisFixture(a, rnd, rows, 160, 6);
    defer a.free(file);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    var t = try openFixture(&td, file);
    defer t.close();
    if (t.pool == null) return error.SkipZigTest;
    const bias = try trellisBias(a, rnd, heads, t.dim);
    defer a.free(bias);
    try load.setHeadBias(&t, bias, heads);
    // a prefill-width gather (token-major, head-minor: 37 tokens x 16 heads), then one decode step's 16 rows
    for ([_]usize{ 37 * heads, heads }) |n| {
        const ids = try randomIds(a, rnd, n, rows);
        defer a.free(ids);
        const ref = try refRows(a, &t, ids, heads);
        defer a.free(ref);
        const got = try a.alloc(f32, n * t.dim);
        defer a.free(got);
        try gather(&t, ids, got, 0);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref), std.mem.sliceAsBytes(got));
    }
}

test "direct ple: an unserved K, a codebook other than MUL1 and a misshapen ring are named errors at the table's open" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBAD7);
    const file = try trellisFixture(a, prng.random(), 4, 160, 6);
    defer a.free(file);
    const Case = struct { from: []const u8, to: []const u8, want: anyerror };
    for ([_]Case{
        .{ .from = "\"K\":\"6\"", .to = "\"K\":\"9\"", .want = error.NgramTableBits },
        .{ .from = "\"codebook\":\"mul1\"", .to = "\"codebook\":\"mcg1\"", .want = error.NgramTableHeader },
        .{ .from = "\"row_dim\":\"160\"", .to = "\"row_dim\":\"128\"", .want = error.NgramTableHeader },
    }) |c| {
        const bad = try std.mem.replaceOwned(u8, a, file, c.from, c.to);
        defer a.free(bad);
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        try testing.expectError(c.want, openFixture(&td, bad));
    }
}

test "direct ple: token-aligned range gathers equal one whole gather bit for bit, resident and cold" {
    // the verify's source-array flush splits its gather at token boundaries (16 rows a token): every range must decode
    // its rows at the same heads, from the same bytes, as one call
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EC5);
    const rnd = prng.random();
    const rows = 4096;
    const heads: u32 = 16;
    const file = try trellisFixture(a, rnd, rows, 160, 6);
    defer a.free(file);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    defer cold_for_test = false;
    var t = try openFixture(&td, file);
    defer t.close();
    const bias = try trellisBias(a, rnd, heads, t.dim);
    defer a.free(bias);
    try load.setHeadBias(&t, bias, heads);
    const n = 4 * heads;
    const ids = try randomIds(a, rnd, n, rows);
    defer a.free(ids);
    const whole = try a.alloc(f32, n * t.dim);
    defer a.free(whole);
    const parts = try a.alloc(f32, n * t.dim);
    defer a.free(parts);
    // the ranges a stepwise chain produces ([t1, d0], [d1], [d2]), every id alone, and one range
    const splits = [_][]const usize{ &.{ 0, 2, 3, 4 }, &.{ 0, 1, 2, 3, 4 }, &.{ 0, 4 } };
    // resident (the serial mapping walk) and every row cold
    for ([_]bool{ false, true }) |cold| {
        cold_for_test = cold;
        try gather(&t, ids, whole, 0);
        for (splits) |cuts| {
            @memset(parts, std.math.nan(f32));
            for (cuts[0 .. cuts.len - 1], cuts[1..]) |lo, hi| try gather(&t, ids[lo * heads .. hi * heads], parts[lo * heads * t.dim .. hi * heads * t.dim], 0);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(whole), std.mem.sliceAsBytes(parts));
        }
    }
}

test "direct ple: a decode step's cold rows, read in parallel, and a prefill gather's, read with a pread each, equal the definition bit for bit" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC01D);
    const rnd = prng.random();
    const rows = 4096;
    const heads: u32 = 16;
    const file = try trellisFixture(a, rnd, rows, 160, 6);
    defer a.free(file);
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    // every row counts as not resident: the decode step's parallel preads, then the prefill workers' own, serve them all
    cold_for_test = true;
    defer cold_for_test = false;
    var t = try openFixture(&td, file);
    defer t.close();
    if (t.pool == null) return error.SkipZigTest;
    const bias = try trellisBias(a, rnd, heads, t.dim);
    defer a.free(bias);
    try load.setHeadBias(&t, bias, heads);
    // decode steps of 1, 3 and 4 verify rows (16 heads each), then a prefill-width gather (37 tokens), a repeated row id
    for ([_]usize{ heads, 3 * heads, 4 * heads, 37 * heads }) |n| {
        const ids = try randomIds(a, rnd, n, rows);
        defer a.free(ids);
        const ref = try refRows(a, &t, ids, heads);
        defer a.free(ref);
        const got = try a.alloc(f32, n * t.dim);
        defer a.free(got);
        try gather(&t, ids, got, 0);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref), std.mem.sliceAsBytes(got));
    }
}
