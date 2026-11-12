//! turboexl3_ling's shared expert over its layer's W_hats (SHARED_LAYER_WHAT). Lossless: it removes launches and keeps
//! every word.
//! Under LAYER_MAJOR the prompt pass runs a MoE layer over every span before the next layer (the copied
//! table_forward.zig `moeSpans`: the router span by span, the routed calls, then each span's shared expert and add,
//! `mlpFinish`, all in one graph that `evalLayer` evaluates once). The per-call path decodes each wide span's
//! shared-expert linear (gate, up, down) trellis to an f16 W_hat on every call: 5 times a layer at the standard request's
//! spans (4 of 4,096 rows and 1,965; the 31-row span runs the register path, which decodes into registers and makes no
//! W_hat). Here `moeSpans` opens a scope around the layer (`begin`, `end`), and the copied MoE block's `sharedDown`
//! hands a wide span's three linears here (`run`):
//! - the layer's first wide span decodes each W_hat by the per-call launch (`Dense.dequantize`, lazy) and keeps the array
//!   for the scope;
//! - each later wide span of the layer runs the same matmul and finish over the kept array: gate and down by
//!   `Dense.pieceInner` and `Dense.finishOf` (`linear`'s matmul-path launches less the decode), up by turboexl3's
//!   `lingUpSwigluW` (SWIGLU_EPI's epilogue launch over the given W_hat);
//! - `end` drops the scope's arrays when the layer's spans are done, so the next layer decodes its own.
//! MLX evaluates a kept W_hat once, in the layer's graph: about 30 MiB alive from the layer's first wide span to its
//! evaluation. The words are the per-call path's (the same launches on the same operands; the decode writes each
//! weight from its own trellis tile): src/ling_shared_layer_test.zig. Null keeps the block's chain: outside a scope (the
//! chunk loop, decode and verify rows, the NextN head), rows the matmul path does not take, pg2 on, an un-rotated
//! resident trellis, a dtype or clamp the epilogue does not take.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ran = @import("ling_ran.zig");
const epi = @import("ling_swiglu_epi.zig");

pub const Proj = t3.Proj;
const Dense = t3.dense.Dense;

// ── the scope: one layer's spans (`moeSpans`) ──

/// The most W_hats a scope keeps: a layer's gate, up and down, with room to spare
const max_kept = 8;
const Kept = struct { key: usize, w: mlx.mlx_array };
var kept: [max_kept]Kept = undefined;
var n_kept: usize = 0;
var scope_open: bool = false;
/// The scope's counts: wide spans served here, those whose three W_hats were all kept already, W_hats decoded
var spans_run: u32 = 0;
var spans_reused: u32 = 0;
var decoded: u32 = 0;

/// The scope's state and counts (the byte test)
pub const Stats = struct { open: bool, kept: usize, spans_run: u32, spans_reused: u32, decoded: u32 };

pub fn stats() Stats {
    return .{ .open = scope_open, .kept = n_kept, .spans_run = spans_run, .spans_reused = spans_reused, .decoded = decoded };
}

/// A layer's spans begin (`moeSpans`): an empty scope.
pub fn begin() void {
    drop();
    scope_open = true;
    spans_run = 0;
    spans_reused = 0;
    decoded = 0;
}

/// A layer's spans are done: the scope's W_hats dropped (the layer's graph still holds what it reads until its
/// evaluation). A scope whose later spans read the kept W_hats marks the run-path line.
pub fn end() void {
    if (scope_open and spans_reused > 0) ran.mark(.SHARED_LAYER_WHAT);
    drop();
    scope_open = false;
}

fn drop() void {
    for (kept[0..n_kept]) |k| _ = mlx.mlx_array_free(k.w);
    n_kept = 0;
}

/// `p`'s W_hat in the scope: the kept array (borrowed), else decoded now by `d`'s per-call launch (lazy) and kept;
/// `fresh` set when it was decoded here.
fn what(s: mlx.mlx_stream, d: *Dense, p: Proj, fresh: *bool) !mlx.mlx_array {
    const key = @intFromPtr(p.trellis.ctx orelse return error.BadExl3Shape);
    for (kept[0..n_kept]) |k| if (k.key == key) return k.w;
    if (n_kept == max_kept) return error.LingSharedLayerFull;
    const w = try d.dequantize(s, p.trellis);
    kept[n_kept] = .{ .key = key, .w = w };
    n_kept += 1;
    decoded += 1;
    fresh.* = true;
    return w;
}

// ── the call ──

/// The shared expert's gate, h (up's SwiGLU) and y (the down: the block's output) of one span; the caller owns all three.
pub const Parts = struct {
    gate: mlx.mlx_array,
    h: mlx.mlx_array,
    y: mlx.mlx_array,
    pub fn deinit(p: Parts) void {
        inline for (.{ p.gate, p.h, p.y }) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The shared expert of x [..., hidden] (a layer-major span's rows) over the scope's W_hats at the layer's clamp `limit`
/// (0 none), or null (no launch) where the scope does not serve the call: no scope open, rows the matmul path does not
/// take, pg2 on, an un-rotated resident trellis, a dtype or clamp the epilogue does not take. `host_table` is the MoE
/// block's `fused_act.sigmoidTableFor` (the table its SWIGLU_EPI path reads).
pub fn runParts(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, limit: f32, comptime host_table: anytype) !?Parts {
    if (!scope_open) return null;
    if (!(limit >= 0 and limit <= 128 and limit == @floor(limit))) return null;
    const lim: u32 = @intFromFloat(limit);
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return null;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |r| rows *= @intCast(r);
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const dg = try t3.denseOf(gate, in_dim, xd, .mul1);
    const du = try t3.denseOf(up, in_dim, xd, .mul1);
    const dd = try t3.denseOf(down, du.shape.out, xd, .mul1);
    // where `linear` and `upSwiglu` decode a W_hat: the matmul path, pg2 off, no un-rotated resident
    for ([_]*Dense{ dg, du, dd }) |d| if (d.pathFor(rows) != .matmul) return null;
    if (t3.pgemm.mode.get().* != .off) return null;
    for ([_]Proj{ gate, up, down }) |p| if (t3.dense.residentUnrot(p.trellis)) return null;
    const tab = if (lim == 0) try host_table(s, xd) else try epi.f32Table(s, xd);
    var fresh = false;
    const wg = try what(s, dg, gate, &fresh);
    const wu = try what(s, du, up, &fresh);
    const wd = try what(s, dd, down, &fresh);
    const gv = try linearOver(s, dg, x, gate, wg, rows);
    errdefer _ = mlx.mlx_array_free(gv);
    // up's SwiGLU over its kept W_hat; `upSwigluW` serves whatever passed the checks above (null there would be a
    // turboexl3 that declines more than `upSwiglu`: the per-call epilogue then serves the span)
    const h = (try t3.lingUpSwigluW(s, x, up, wu, gv, lim, tab, .mul1)) orelse
        ((try epi.up(s, x, up.trellis, up.suh, up.svh, gv, limit, host_table, .shared)) orelse return error.LingSharedLayerDeclined);
    errdefer _ = mlx.mlx_array_free(h);
    const y = try linearOver(s, dd, h, down, wd, rows);
    spans_run += 1;
    if (!fresh) spans_reused += 1;
    return .{ .gate = gv, .h = h, .y = y };
}

/// `runParts`'s y alone (the copied MoE block's `sharedDown`); the caller owns it.
pub fn run(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, limit: f32, comptime host_table: anytype) !?mlx.mlx_array {
    const p = (try runParts(s, x, gate, up, down, limit, host_table)) orelse return null;
    _ = mlx.mlx_array_free(p.gate);
    _ = mlx.mlx_array_free(p.h);
    return p.y;
}

/// `linear` of x (rows `rows`, on the matmul path) over a given W_hat `w`: its prepare, MLX's matmul against `w` and its
/// finish (`Dense.pieceInner`, `Dense.finishOf`: `linearOn`'s launches and configs less the decode), y in x's dtype,
/// x's shape with its last axis `p`'s outputs. The caller owns y.
pub fn linearOver(s: mlx.mlx_stream, d: *Dense, x: mlx.mlx_array, p: Proj, w: mlx.mlx_array, rows: usize) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    var osh: [8]c_int = undefined;
    if (xsh.len < 1 or xsh.len > osh.len) return error.BadExl3Shape;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(rows), @intCast(d.shape.in) }, 2, s));
    const z = try d.pieceInner(s, x2, w, p.suh, rows);
    defer _ = mlx.mlx_array_free(z);
    const y2 = try d.finishOf(s, z, p.svh, rows);
    defer _ = mlx.mlx_array_free(y2);
    @memcpy(osh[0..xsh.len], xsh);
    osh[xsh.len - 1] = @intCast(d.shape.out);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &osh, @intCast(xsh.len), s));
    return y;
}

/// The shared expert of x over given W_hats (`wg`, `wu`, `wd`), outside any scope: the byte test's stale-W_hat check
/// (a layer's spans over another layer's W_hats must not give its words). The caller owns the three.
pub fn partsOver(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Proj, up: Proj, down: Proj, wg: mlx.mlx_array, wu: mlx.mlx_array, wd: mlx.mlx_array, limit: u32, tab: mlx.mlx_array) !Parts {
    const xd = mlx.mlx_array_dtype(x);
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return error.BadExl3Shape;
    var rows: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |r| rows *= @intCast(r);
    const in_dim: u32 = @intCast(xsh[xsh.len - 1]);
    const dg = try t3.denseOf(gate, in_dim, xd, .mul1);
    const du = try t3.denseOf(up, in_dim, xd, .mul1);
    const dd = try t3.denseOf(down, du.shape.out, xd, .mul1);
    const gv = try linearOver(s, dg, x, gate, wg, rows);
    errdefer _ = mlx.mlx_array_free(gv);
    const h = (try t3.lingUpSwigluW(s, x, up, wu, gv, limit, tab, .mul1)) orelse return error.LingSharedLayerDeclined;
    errdefer _ = mlx.mlx_array_free(h);
    return .{ .gate = gv, .h = h, .y = try linearOver(s, dd, h, down, wd, rows) };
}
