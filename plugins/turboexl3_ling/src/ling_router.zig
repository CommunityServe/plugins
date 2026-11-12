//! turboexl3_ling: a MoE layer's router and its read-ahead predictor in two launches (ROUTER_FUSED, exact: the block's
//! words), turboexl3's `root.ling.router` in place of nine launches:
//! - launch 1 (`logits`): the router's logits, MLX's one-row gemv words over the loader's f32 table, beside the
//!   predictor's scores of the next routed layer's bf16 router on the call's rows, the words of the streamer's
//!   read-ahead scores at PREDICTOR's rule (src/ling_predictor.zig `lingScores`: MLX's gemv at one row and gemv_wide at
//!   2 to 8, the astype, then Ling's sigmoid(z) + bias).
//! - launch 2 (`select`): the served fused router's text (`sigmoid_bias_grouped`, checked byte for byte below at
//!   compile time) over those logits, its weights stored in bf16 (the MoE block's astype folded into the store).
//! The router weight is the pack's f16 words (ROUTER_F16; the NextN layer's too): the copied trunk.zig `routed` builds
//! no f32 table and serves the weight's transposed view, which the load keeps either way (`transposed` owns it and
//! `evalTrunk` evaluates it), so 8 MiB a layer of f32 table is never made (about 552 MiB resident with the NextN layer;
//! the bank's room is measured after the load, so its slot rows take it), and launch 1 reads 4 MiB a layer in place of
//! 8. Launch 1 reads the f16 words and widens each in registers, then + 0: the f32 table's word (its + 0 turns -0.0
//! into +0.0; the served pack has 38 such words), so the same words. The view is [hidden, 512] column-major, the layout
//! the block's matmul reads (MLX's astype and add keep a view's layout, so the f32 table was column-major too); the
//! launch reads that memory through the view's transpose, [512, hidden] row-major (no copy). A weight in another layout
//! is declined.
//! Where the launches serve: at one row always (serial decode); at 2 to 8 rows only in the exact verify
//! (`ling_exl3.exact_rows`), whose router rows are one-row gemv words already (`denseRows`). Every other call (the
//! prompt chunks, the layer-major spans, more than 8 rows) runs the block's own matmul over the f32 table rebuilt for the
//! call as the loader's `f32Table` built it (`stockTable`: astype, + 0, the same words and layout, so MLX takes the same
//! kernel): 2 launches and about 28 MiB of traffic a layer a call, small beside a prompt chunk. The MoE block (the
//! copied moe.zig `routerLogits` and `route`) asks `logits`, then `select` over the logits `logits` returned; the
//! layer's decode-lane routed call takes the scores (`takeScores`) in place of its own read-ahead scores, and `clear` at
//! the block's end frees them. Each call launch 1 serves marks the run-path line.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ling_exl3 = @import("ling_exl3.zig");
const ran = @import("ling_ran.zig");
const router_kernel = @import("mlx_serve/sdk/blocks/moe/router_kernel.zig");

const R = t3.ling.router;

comptime {
    // the select launch runs the served kernel's text: turboexl3's copy is the host's, byte for byte
    @setEvalBranchQuota(1_000_000);
    if (!std.mem.eql(u8, R.text_select, router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias_grouped)]))
        @compileError("turboexl3's ling.router.text_select is not the served router's sigmoid_bias_grouped text");
}

/// A routed layer's read-ahead gate as the streamer holds it (`LingSource.Gate`): its router_raw (bf16 [experts,
/// hidden]) and expert bias (f32 [experts]); the predictor of routed layer i reads layer i + 1's
const Gate = struct { w: mlx.mlx_array, bias: mlx.mlx_array };
const max_layers = 256;
var gates: [max_layers]Gate = undefined;
var n_gates: u32 = 0;
var first_layer: u32 = 0;

/// The bank's gates in the streamer's order (`openBank`: routed layer i is model layer `first + i`); the arrays stay the
/// model's. More than `max_layers` leaves launch 1 without a predictor.
pub fn setGates(first: u32, gs: anytype) void {
    n_gates = 0;
    if (gs.len > max_layers) return;
    for (gs, 0..) |g, i| gates[i] = .{ .w = g.w, .bias = g.bias };
    n_gates = @intCast(gs.len);
    first_layer = first;
}

/// The scores launch 1 made (this module's until `clear`), the routed call they are for (once), and the logits
/// `select` takes
var held: ?mlx.mlx_array = null;
var pending: ?struct { layer: u32, rows: u32 } = null;
var own_logits: ?*anyopaque = null;

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// Drops what the block's call did not take: the scores and the logits `select` would have taken
pub fn clear() void {
    pending = null;
    own_logits = null;
    if (held) |h| free(h);
    held = null;
}

/// Routed layer `layer`'s scores for its call of `rows` rows, if launch 1 made them for that call (once); null
/// otherwise. This module keeps them until the block's `clear`.
pub fn takeScores(layer: u32, rows: u32) ?mlx.mlx_array {
    const p = pending orelse return null;
    pending = null;
    if (p.layer != layer or p.rows != rows) return null;
    return held;
}

fn rowsOf(sh: []const c_int) c_int {
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| rows *= d;
    return rows;
}

/// Whether `w` is the loader's layout, the one the block's matmul reads: [hidden, 512] column-major (strides 1, hidden)
fn columnMajor(w: mlx.mlx_array, hidden: c_int) bool {
    const ws = mlx.getShape(w);
    if (ws.len != 2 or ws[0] != hidden or ws[1] != R.n_experts) return false;
    const st = mlx.mlx_array_strides(w);
    return st[0] == 1 and st[1] == @as(usize, @intCast(hidden));
}

/// Launch 1 for model layer `layer`'s MoE input `x` [..., hidden] (bf16) over the router's weight `w` as the loader
/// keeps it: the f16 weight's transposed view (an f32 table where a pack stores its router in f32), [hidden, 512]
/// column-major (the array the block's matmul reads); the launch reads its memory as [512, hidden] through a transposed
/// view. The logits in x's shape with 512 last (the caller owns them), the read-ahead scores kept for the layer's call.
/// Null where the block's own router runs: not a GPU stream, x not bf16, 2 to 8 rows outside the exact verify, more than
/// 8 rows, a weight in another dtype or layout, or a shape turboexl3 does not take.
pub fn logits(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, layer: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const wdt = mlx.mlx_array_dtype(w);
    if (mlx.mlx_array_dtype(x) != .bfloat16 or !(wdt == .float32 or wdt == .float16)) return null;
    const sh = mlx.getShape(x);
    if (sh.len < 1 or sh.len > 4) return null;
    const hidden = sh[sh.len - 1];
    const rows = rowsOf(sh);
    if (rows != 1 and !(rows <= R.max_rows and ling_exl3.exact_rows)) return null;
    if (!R.takes(rows, hidden)) return null;
    if (!columnMajor(w, hidden)) return null;
    clear();
    // the next routed layer's gate (none after the last), as the streamer's `predictorGate`, scored by PREDICTOR's rule
    const bl: ?u32 = if (layer >= first_layer and layer - first_layer < n_gates) layer - first_layer else null;
    const pred: ?R.Predictor = if (bl) |b| (if (b + 1 < n_gates) R.Predictor{ .w = gates[b + 1].w, .bias = gates[b + 1].bias, .score = .sigmoid } else null) else null;
    var x2 = mlx.mlx_array_new();
    defer free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, hidden }, 2, s));
    // the table's memory as [512, hidden] row-major: its transposed view (no copy)
    var wt = mlx.mlx_array_new();
    defer free(wt);
    try mlx.check(mlx.mlx_transpose_axes(&wt, w, &[_]c_int{ 1, 0 }, 2, s));
    const l = R.routerLogits(s, x2, wt, pred) catch |e| switch (e) {
        error.BadExl3Shape, error.BadExl3Dtype => return null,
        else => return e,
    };
    defer free(l.logits);
    errdefer if (l.scores) |sc| free(sc);
    var osh: [4]c_int = undefined;
    @memcpy(osh[0 .. sh.len - 1], sh[0 .. sh.len - 1]);
    osh[sh.len - 1] = R.n_experts;
    var y = mlx.mlx_array_new();
    errdefer free(y);
    try mlx.check(mlx.mlx_reshape(&y, l.logits, &osh, sh.len, s));
    if (l.scores) |sc| {
        held = sc;
        pending = .{ .layer = bl.?, .rows = @intCast(rows) };
    }
    own_logits = y.ctx;
    ran.mark(.ROUTER_FUSED);
    // ROUTER_F16: launch 1 read the pack's f16 router words (an f32 table where a pack stores its router in f32)
    if (wdt == .float16) ran.mark(.ROUTER_F16);
    return y;
}

/// The block's own matmul's operand for a call launch 1 does not serve: the f32 table rebuilt from the kept f16 view
/// `w` as the loader's `f32Table` built it (turboexl3's `tableF32`: astype, then + 0, the same words and column-major
/// layout, so MLX's matmul takes the same kernel); the caller owns it. Null for any other weight.
pub fn stockTable(s: mlx.mlx_stream, w: mlx.mlx_array) !?mlx.mlx_array {
    if (mlx.mlx_array_dtype(w) != .float16) return null;
    const t = try R.tableF32(s, w);
    // ROUTER_F16's prompt side: the block's own matmul over the table rebuilt from the kept f16 weight
    ran.mark(.ROUTER_F16);
    return t;
}

/// `route`'s inds and weights in the routing struct's names (`adoptRouting`'s)
pub const Selected = struct { inds: mlx.mlx_array, norm_scores: mlx.mlx_array, gate_logit: mlx.mlx_array = .{ .ctx = null } };

/// Launch 2 over the logits `logits` returned (any other logits: null) at Ling's routing (top-8 of the 4 best of 8
/// groups, normalized; f32 bias): the served kernel's ids and its weights in bf16, in the logits' shape with k last.
/// The caller owns them.
pub fn select(s: mlx.mlx_stream, lg: mlx.mlx_array, bias: mlx.mlx_array, k: c_int, n_group: c_int, topk_group: c_int, norm: bool, scale: f32) !?Selected {
    const own = own_logits orelse return null;
    if (lg.ctx == null or lg.ctx.? != own) return null;
    own_logits = null;
    if (k != R.top_k or n_group != R.n_group or topk_group != R.topk_group or !norm) return null;
    if (mlx.mlx_array_dtype(bias) != .float32) return null;
    const sh = mlx.getShape(lg);
    if (sh.len < 1 or sh.len > 4 or sh[sh.len - 1] != R.n_experts) return null;
    const rows = rowsOf(sh);
    var l2 = mlx.mlx_array_new();
    defer free(l2);
    try mlx.check(mlx.mlx_reshape(&l2, lg, &[_]c_int{ rows, R.n_experts }, 2, s));
    const sel = R.select(s, l2, bias, scale) catch |e| switch (e) {
        error.BadExl3Shape, error.BadExl3Dtype => return null,
        else => return e,
    };
    defer free(sel.inds);
    defer free(sel.weights);
    var osh: [4]c_int = undefined;
    @memcpy(osh[0 .. sh.len - 1], sh[0 .. sh.len - 1]);
    osh[sh.len - 1] = k;
    var inds = mlx.mlx_array_new();
    errdefer free(inds);
    try mlx.check(mlx.mlx_reshape(&inds, sel.inds, &osh, sh.len, s));
    var wts = mlx.mlx_array_new();
    errdefer free(wts);
    try mlx.check(mlx.mlx_reshape(&wts, sel.weights, &osh, sh.len, s));
    return .{ .inds = inds, .norm_scores = wts };
}
