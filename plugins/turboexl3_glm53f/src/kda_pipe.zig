//! `TURBOEXL3_GLM53F_KDA_PIPE=1` (opt-in, exact, with KDA_CHUNK): a KDA layer's prompt forward of two spans or more
//! as 4,096-row spans (a remainder under 2,048 rows joins the span before it), each span's qkv linear on a side stream
//! beside the recurrence of the span before it. The chunkwise recurrence is latency-bound and runs one span after
//! another on the model's stream; the next span's qkv GEMM, on its own command queue, takes the GPU it leaves.
//!
//! Each value is the unspanned forward's (kda.zig):
//! - the low-rank z, a and b projections over every row, as kda.zig makes them;
//! - each span's qkv rows as turboexl3's `linear` makes them on its matmul path: the prepare, the W_hat (decoded once
//!   for every span), pg2's fused launch or MLX's matmul, and the finish. Each is per row, the row count choosing only
//!   the grid; every span's rows and the whole call's take MLX's regular NAX GEMM (2,048 to 16,384 rows, where pg2's
//!   probe held its words). Under turboexl3's trunk int8 at the qkv's shape (TURBOEXL3_GLM53F_PF_INT8=3,
//!   `t3.trunk_q8.takes`) the int8 weights are decoded once for every span and each span's rows prepared and multiplied
//!   on int8 (`t3.trunk_q8.rowsOn`): a row's int8 prepare and int32 sums are its own, so the spans give the unspanned
//!   int8 call's bytes;
//! - each span's prologue text or fold and chunkwise launch over the conv window and f32 state the span before it
//!   leaves, its 16-row chunks where the whole call's fall (every span but the last a multiple of 16);
//! - kda.zig's prompt tail (the norm-gate text or the chain, then the out projection) over the joined rows.
//! src/kda_gpu_test.zig holds the spanned forward to the unspanned one bit for bit on the device.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const t3 = @import("turboexl3");
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_KDA_PIPE=1`, set by the plugin at load (with KDA_CHUNK on); `install` then makes the side stream.
pub var on: bool = false;
/// The switch's load line, after the install: the span's rows, the fewest rows a span takes alone.
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PIPE=1: a KDA layer's prompt forward of two spans or more as {d}-row spans (a remainder under {d} rows joins the span before it), each span's qkv linear on a side stream beside the chunkwise recurrence of the span before it, the conv window and f32 state carried; the unspanned forward's bits\n";

/// A span's rows (every span but the last; a multiple of the chunk kernel's 16-row chunks).
pub const span_rows: c_int = 4096;
/// The fewest rows a last span takes alone: a shorter remainder joins the span before it, so every span keeps MLX's
/// regular NAX GEMM (pg2's probe: 2,048 to 16,384 rows).
pub const min_last_rows: c_int = 2048;
/// The most spans a forward takes (a 64K-row prompt chunk).
pub const max_spans: u32 = 16;

comptime {
    std.debug.assert(@rem(span_rows, kda_prompt.chunk_rows) == 0);
}

/// A forward's spans: `n` of them over `t` rows.
pub const Plan = struct {
    n: u32,
    t: c_int,

    /// Span `i`'s first row.
    pub fn start(_: Plan, i: u32) c_int {
        return span_rows * @as(c_int, @intCast(i));
    }

    /// Span `i`'s rows: `span_rows`, the last one the rest.
    pub fn rows(p: Plan, i: u32) c_int {
        return if (i + 1 < p.n) span_rows else p.t - p.start(i);
    }
};

/// The spans of a forward of `t` rows, or null where it takes one (fewer than two spans' rows) or past `max_spans`.
pub fn planFor(t: c_int) ?Plan {
    if (t < 2 * span_rows) return null;
    var n: c_int = @divTrunc(t, span_rows);
    if (t - n * span_rows >= min_last_rows) n += 1;
    if (n > max_spans) return null;
    return .{ .n = @intCast(n), .t = t };
}

/// The side stream the spans' qkv linears run on (`install`).
var side: ?mlx.mlx_stream = null;
var engaged = false;
/// Spanned forwards since the load (src/kda_gpu_test.zig reads that the route ran).
pub var runs: u64 = 0;

/// A second stream on the model stream's GPU (its own command queue). Refuses by name: a stream that is not a GPU's.
pub fn install(s: mlx.mlx_stream) !void {
    if (!mlx.streamIsGpu(s)) return error.Glm53fKdaPipeStream;
    if (side != null) return;
    var dev = mlx.mlx_device{ .ctx = null };
    try mlx.check(mlx.mlx_stream_get_device(&dev, s));
    defer _ = mlx.mlx_device_free(dev);
    const rs = mlx.mlx_stream_new_device(dev);
    if (rs.ctx == null) return error.Glm53fKdaPipeStream;
    side = rs;
}

/// The side stream goes with the model.
pub fn deinit() void {
    if (side) |rs| _ = mlx.mlx_stream_free(rs);
    side = null;
    engaged = false;
}

/// Whether kda.zig's prompt forward of `seq_len` rows of one slot asks the spans (it then leaves qkv to them): the
/// switch on and installed, an exllamav3 q|k|v linear, two spans or more.
pub fn wants(la: *const trunk.Kda, seq_len: c_int) bool {
    return on and side != null and la.qkv.x3 != null and planFor(seq_len) != null;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// Rows [r0, r0 + rows) of `x` [1, T, w] on `s`, a new array.
fn rowsOf(s: mlx.mlx_stream, x: Array, r0: c_int, rows: c_int) !Array {
    const w = mlx.getShape(x)[2];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, r0, 0 }, 3, &[_]c_int{ 1, r0 + rows, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// One span's qkv rows [1, rows, out] on `rs`: turboexl3's `linear` on its matmul path (`Dense.linearOn`) over the
/// span's rows of `x`, against `W` (`Dense.dequantize`, made once for every span): the prepare, pg2's fused launch
/// where it takes the shape, else MLX's matmul and the finish. With `q8` (turboexl3's trunk int8 at this shape) `W` is
/// the int8 weights (`t3.trunk_q8.decodeW`, made once for every span) and the span's rows take the int8 prepare and GEMM.
fn spanQkv(rs: mlx.mlx_stream, d: *t3.dense.Dense, p: t3.Proj, W: Array, q8: bool, x: Array, r0: c_int, rows: c_int) !Array {
    const in_dim = mlx.getShape(x)[2];
    const xs = try rowsOf(rs, x, r0, rows);
    defer free(xs);
    var x2 = mlx.mlx_array_new();
    defer free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, xs, &[_]c_int{ rows, in_dim }, 2, rs));
    if (q8) {
        const yq = try t3.trunk_q8.rowsOn(rs, d, x2, p.suh, p.svh, W, @intCast(rows));
        defer free(yq);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_reshape(&y, yq, &[_]c_int{ 1, rows, @intCast(d.shape.out) }, 3, rs));
        return y;
    }
    const xp = try d.prepare(rs, x2, p.suh, @intCast(rows));
    defer free(xp);
    const y2 = (try t3.pgemm.fusedOn(rs, xp, W, p.svh, d.shape.out_dtype)) orelse blk: {
        var inner = mlx.mlx_array_new();
        defer free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, rs));
        break :blk try d.finishOf(rs, inner, p.svh, @intCast(rows));
    };
    defer free(y2);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &[_]c_int{ 1, rows, @intCast(d.shape.out) }, 3, rs));
    return y;
}

/// kda.zig's prompt forward of one slot from its z, a and b projections (`a` and `z` past the low-rank second stages)
/// and the chain's q and k scales, as spans (`wants`): the layer's output, the entry's conv window and state set as the
/// unspanned forward sets them. Null, with nothing launched on the entry, where the forward takes another route (no
/// span plan, a span or the call off turboexl3's matmul path, kda.zig's per-core hooks not taking the recurrence,
/// neither the fold nor the prologue text serving the first span): the caller runs it unspanned.
pub fn forward(m: *const trunk.Model, la: *const trunk.Kda, x: Array, e: anytype, z: Array, a: Array, b: Array, q_scale: Array, k_scale: Array, qk_eps: f32) !?Array {
    const rs = side orelse return null;
    const p = la.qkv.x3 orelse return null;
    const s = m.s;
    const c = m.c;
    const hk: c_int = @intCast(c.linear_num_key_heads);
    const hv: c_int = @intCast(c.linear_num_value_heads);
    const dk: c_int = @intCast(c.linear_key_head_dim);
    const dv: c_int = @intCast(c.linear_value_head_dim);
    const taps: c_int = @intCast(c.linear_conv_kernel_dim);
    const lb = c.kda_gate_lower_bound;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3 or xsh[0] != 1) return null;
    const t = xsh[1];
    const plan = planFor(t) orelse return null;
    // the recurrence kda.zig runs after the fold or the prologue text (qkv in x's dtype: turboexl3's `linear`)
    if (!kda_prompt.percoreServes(hk, hv, dk, dv, mlx.mlx_array_dtype(x), m.ssmStateDtype(), la.a_log_h, la.dt_f32)) return null;
    const d = t3.denseOf(p, @intCast(xsh[2]), mlx.mlx_array_dtype(x), .mul1) catch return null;
    if (d.pathFor(@intCast(t)) != .matmul) return null;
    for (0..plan.n) |i| if (d.pathFor(@intCast(plan.rows(@intCast(i)))) != .matmul) return null;
    // the W_hat, or under turboexl3's trunk int8 at this shape the int8 weights: decoded once for every span
    const q8 = t3.trunk_q8.takes(d);
    const W = if (q8) try t3.trunk_q8.decodeW(rs, d, p.trellis) else try d.dequantize(rs, p.trellis);
    defer free(W);
    // beta over every row, as kda.zig makes it
    var beta = mlx.mlx_array_new();
    defer free(beta);
    try mlx.check(mlx.mlx_sigmoid(&beta, b, s));
    var ys: [max_spans]Array = undefined;
    var made: usize = 0;
    defer for (ys[0..made]) |arr| free(arr);
    for (0..plan.n) |ii| {
        const i: u32 = @intCast(ii);
        const r0 = plan.start(i);
        const rows = plan.rows(i);
        const qkv = try spanQkv(rs, d, p, W, q8, x, r0, rows);
        defer free(qkv);
        const a_i = try rowsOf(s, a, r0, rows);
        defer free(a_i);
        const beta_i = try rowsOf(s, beta, r0, rows);
        defer free(beta_i);
        const prev: ?Array = if (e.initialized) e.conv_state else null;
        var r: kda_prompt.Recur = undefined;
        if (try kda_prompt.foldWindow(s, qkv, prev, la.conv_w, q_scale, k_scale, hk, dk, hv, dv, taps, m.ssmStateDtype(), la.a_log_h, la.dt_f32)) |fw| {
            defer free(fw.before);
            free(e.conv_state);
            e.conv_state = fw.window;
            e.initialized = true;
            if (e.ssm_state.ctx == null) e.ssm_state = try zeroState(s, hv, dv, dk, m.ssmStateDtype());
            r = try kda_prompt.foldRecur(s, qkv, fw.before, la.conv_w, q_scale, k_scale, qk_eps, a_i, beta_i, la.a_log_h, la.dt_f32, lb, e.ssm_state);
        } else if (try kda_prompt.prologue(s, qkv, prev, la.conv_w, q_scale, k_scale, qk_eps, hk, dk, hv, dv, taps)) |pr| {
            defer for ([_]Array{ pr.q, pr.k, pr.v }) |arr| free(arr);
            free(e.conv_state);
            e.conv_state = pr.window;
            e.initialized = true;
            if (e.ssm_state.ctx == null) e.ssm_state = try zeroState(s, hv, dv, dk, m.ssmStateDtype());
            r = try kda_prompt.percoreRecur(s, pr.q, pr.k, pr.v, a_i, beta_i, la.a_log_h, la.dt_f32, lb, e.ssm_state);
        } else {
            if (i == 0) return null;
            return error.Glm53fKdaPipeSpan;
        }
        free(e.ssm_state);
        e.ssm_state = r.state;
        ys[made] = r.y;
        made += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&ys, made);
    defer _ = mlx.mlx_vector_array_free(vec);
    var y = mlx.mlx_array_new();
    defer free(y);
    try mlx.check(mlx.mlx_concatenate_axis(&y, vec, 1, s));
    runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PIPE engaged: {d} rows as {d} spans, the qkv linear on the side stream\n", .{ t, plan.n });
    }
    // kda.zig's prompt tail: the norm-gate text where it serves, else the chain, then the out projection
    if (try kda_prompt.normGate(s, y, z, la.norm_w, c.rms_norm_eps, hv, dv)) |flat| {
        defer free(flat);
        return try m.mm(&la.out, flat);
    }
    const y_shape = [_]c_int{ 1, t, hv, dv };
    var z_heads = mlx.mlx_array_new();
    defer free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z, &y_shape, 4, s));
    const y_normed = try m.rmsNorm(y, la.norm_w);
    defer free(y_normed);
    var gate = mlx.mlx_array_new();
    defer free(gate);
    try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, s));
    var gated = mlx.mlx_array_new();
    defer free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, y_normed, gate, s));
    var out_flat = mlx.mlx_array_new();
    defer free(out_flat);
    try mlx.check(mlx.mlx_reshape(&out_flat, gated, &[_]c_int{ 1, t, hv * dv }, 3, s));
    return try m.mm(&la.out, out_flat);
}

/// The zero state kda.zig makes for a cold layer: [1, hv, dv, dk] in the state's dtype.
fn zeroState(s: mlx.mlx_stream, hv: c_int, dv: c_int, dk: c_int, dt: mlx.mlx_dtype) !Array {
    var st = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_zeros(&st, &[_]c_int{ 1, hv, dv, dk }, 4, dt, s));
    return st;
}

// ── tests (MLX-free) ──

const testing = std.testing;

test "turboexl3_glm53f stream kda pipe: a forward takes 4,096-row spans from 8,192 rows, the last the rest, a remainder under 2,048 rows joined to the span before it, at most 16 spans" {
    // one span's rows or fewer: unspanned
    for ([_]c_int{ 1, 31, 995, 4096, 8191 }) |t| try testing.expect(planFor(t) == null);
    // a 16K prompt's main call: three spans of 4,096 and the 4,068 left
    const p = planFor(16356).?;
    try testing.expectEqual(@as(u32, 4), p.n);
    for (0..3) |i| try testing.expectEqual(span_rows, p.rows(@intCast(i)));
    try testing.expectEqual(@as(c_int, 4068), p.rows(3));
    try testing.expectEqual(@as(c_int, 3 * 4096), p.start(3));
    // a remainder under 2,048 rows joins the span before it
    const q = planFor(8197).?;
    try testing.expectEqual(@as(u32, 2), q.n);
    try testing.expectEqual(@as(c_int, 4101), q.rows(1));
    try testing.expectEqual(@as(u32, 3), planFor(8192 + 2048).?.n);
    try testing.expectEqual(@as(u32, 2), planFor(8192 + 2047).?.n);
    // every span but the last ends on a chunk boundary, and the spans cover the rows
    for ([_]c_int{ 8192, 8197, 12288, 16356, 16384, 65535 }) |t| {
        const pl = planFor(t).?;
        var sum: c_int = 0;
        for (0..pl.n) |i| {
            const r = pl.rows(@intCast(i));
            if (i + 1 < pl.n) try testing.expectEqual(@as(c_int, 0), @rem(r, kda_prompt.chunk_rows));
            try testing.expect(r >= min_last_rows);
            sum += r;
        }
        try testing.expectEqual(t, sum);
    }
    // past 16 spans: unspanned
    try testing.expect(planFor(17 * 4096) == null);
}
