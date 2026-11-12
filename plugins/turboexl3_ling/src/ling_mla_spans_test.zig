//! turboexl3_ling's MLA prompt spans byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips): the copied
//! plugins/bailing_hybrid/attn_spans.zig `attend` (MLA_SPANS) against mla.zig's composed `spanAttn` (one MLX SDPA call,
//! causal, over the call's keys; spans of 4096 rows past that width), word for word, at Ling-3.1-flash's expanded MLA
//! heads (64 heads, qk 192, v 128, bf16), the keys and values views of a wider cache: the chunk loop's chunks (4096 rows
//! after 0 to 16,384 keys), the standard prompt's last chunks, warm prompts whose first rows take MLX's softmax loop,
//! short prompts, layer-major widths, a 16-row tail, and calls too short to split. Each case prints a
//! `TURBOEXL3_LING_MLA_SPANS` line ending in PASS or FAIL. Two `info` lines show the words a split without
//! `attn_spans.spanEnd`'s rules gives (they decide nothing).
//! The flash launch that mla.zig's `spanAttn` tries first (src/ling_flash.zig, MLA_FLASH: a reordering, not the same
//! words) is held to the same composed `spanAttn` within a bound, with a red check (`TURBOEXL3_LING_MLA_FLASH` lines).
//! Nothing is timed.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const spans = @import("mlx_serve/plugins/bailing_hybrid/attn_spans.zig");
const flash = @import("ling_flash.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// mla.zig's `prompt_span_rows`
const ctrl_rows: c_int = 4096;

/// MLX_SERVE_LING_GPU_TESTS=1 (the tests' own variable) and a GPU stream, else the test skips
fn gpu() !Stream {
    const gate = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(gate), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// normal(0, scale) in bf16 from `seed`, evaluated
fn randn(s: Stream, seed: u64, shape: []const c_int, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0, scale, key, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn slice4(a: Array, start: [4]c_int, stop: [4]c_int, s: Stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(mlx.mlx_slice(&r, a, &start, 4, &stop, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return r;
}

fn sdpa(q: Array, k: Array, v: Array, scale: f32, s: Stream) !Array {
    const none = mlx.mlx_array_new();
    defer free(none);
    var o = mlx.mlx_array_new();
    errdefer free(o);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "causal", none, .{ .ctx = null }, false, s));
    return o;
}

/// Query rows [r0, r1) of `q` over keys and values [0, first + r1), causal
fn spanOf(q: Array, k: Array, v: Array, scale: f32, first: c_int, r0: c_int, r1: c_int, s: Stream) !Array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const qr = try slice4(q, .{ 0, 0, r0, 0 }, .{ qs[0], qs[1], r1, qs[3] }, s);
    defer free(qr);
    const kr = try slice4(k, .{ 0, 0, 0, 0 }, .{ ks[0], ks[1], first + r1, ks[3] }, s);
    defer free(kr);
    const vr = try slice4(v, .{ 0, 0, 0, 0 }, .{ vs[0], vs[1], first + r1, vs[3] }, s);
    defer free(vr);
    return sdpa(qr, kr, vr, scale, s);
}

/// The spans' outputs joined on axis 2, evaluated
fn joined(parts: mlx.mlx_vector_array, s: Stream) !Array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 2, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// mla.zig's composed `spanAttn` (no flash launch, no attn_spans spans) with `mode` causal: one MLX SDPA call up to
/// `ctrl_rows` rows, else spans of `ctrl_rows` rows, each over the keys up to its last row
fn spanAttn(q: Array, k: Array, v: Array, scale: f32, seq_len: c_int, s: Stream) !Array {
    if (seq_len <= ctrl_rows) {
        const o = try sdpa(q, k, v, scale, s);
        errdefer free(o);
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }
    const first = mlx.getShape(k)[2] - seq_len;
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var r0: c_int = 0;
    while (r0 < seq_len) : (r0 += ctrl_rows) {
        const o = try spanOf(q, k, v, scale, first, r0, @min(r0 + ctrl_rows, seq_len), s);
        defer free(o);
        _ = mlx.mlx_vector_array_append_value(parts, o);
    }
    return joined(parts, s);
}

/// A split into spans of `rows` rows with none of `spanEnd`'s rules (no softmax-kernel or short-tail extension)
fn naiveSplit(q: Array, k: Array, v: Array, scale: f32, seq_len: c_int, rows: c_int, s: Stream) !Array {
    const first = mlx.getShape(k)[2] - seq_len;
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var r0: c_int = 0;
    while (r0 < seq_len) : (r0 += rows) {
        const o = try spanOf(q, k, v, scale, first, r0, @min(r0 + rows, seq_len), s);
        defer free(o);
        _ = mlx.mlx_vector_array_append_value(parts, o);
    }
    return joined(parts, s);
}

/// Words that differ, of how many
const Diff = struct { n: usize, of: usize };

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word), and the element count
fn differing(a: Array, b: Array, s: Stream) !Diff {
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return .{ .n = n, .of = n };
    var ca = mlx.mlx_array_new();
    defer free(ca);
    try mlx.check(mlx.mlx_contiguous(&ca, a, false, s));
    try mlx.check(mlx.mlx_array_eval(ca));
    var cb = mlx.mlx_array_new();
    defer free(cb);
    try mlx.check(mlx.mlx_contiguous(&cb, b, false, s));
    try mlx.check(mlx.mlx_array_eval(cb));
    const isz = mlx.mlx_array_itemsize(cb);
    const pa = mlx.mlx_array_data_uint8(ca) orelse return error.NoData;
    const pb = mlx.mlx_array_data_uint8(cb) orelse return error.NoData;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(!std.mem.eql(u8, pa[i * isz ..][0..isz], pb[i * isz ..][0..isz]));
    return .{ .n = d, .of = n };
}

/// One prompt call: `rows` query rows after `first` keys (so first + rows keys), the keys and values the first rows of
/// a cache of `cap_extra` more rows (0: the whole array); `qs` scales the queries (larger: peakier softmax rows, more
/// probabilities exactly 0); `split` whether `attend` should split it; `naive` > 0 also prints the words of a split
/// into spans of that many rows with no `spanEnd` rules (info)
const Case = struct { rows: c_int, first: c_int, qs: f32 = 1.0, cap_extra: c_int = 37, split: bool = true, naive: c_int = 0 };

const heads: c_int = 64;
const qk: c_int = 192;
const vd: c_int = 128;

fn runCase(s: Stream, c: Case, seed: u64) !bool {
    defer _ = mlx.mlx_clear_cache();
    const t = c.first + c.rows;
    const cap = t + c.cap_extra;
    const q = try randn(s, seed, &.{ 1, heads, c.rows, qk }, c.qs);
    defer free(q);
    const kc = try randn(s, seed + 1, &.{ 1, heads, cap, qk }, 1.0);
    defer free(kc);
    const vc = try randn(s, seed + 2, &.{ 1, heads, cap, vd }, 1.0);
    defer free(vc);
    const k = try slice4(kc, .{ 0, 0, 0, 0 }, .{ 1, heads, t, qk }, s);
    defer free(k);
    const v = try slice4(vc, .{ 0, 0, 0, 0 }, .{ 1, heads, t, vd }, s);
    defer free(v);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    const n = spans.count(c.first, c.rows, ctrl_rows);
    const want = try spanAttn(q, k, v, scale, c.rows, s);
    defer free(want);
    var got = mlx.mlx_array_new();
    defer free(got);
    const did = try spans.attend(&got, q, k, v, scale, c.rows, ctrl_rows, s);
    var d: Diff = .{ .n = 0, .of = mlx.mlx_array_size(want) };
    if (did) {
        try mlx.check(mlx.mlx_array_eval(got));
        d = try differing(got, want, s);
    }
    const ok = did == c.split and (!did or (d.n == 0 and spans.last_spans == n.spans));
    std.debug.print("TURBOEXL3_LING_MLA_SPANS rows={d} first={d} keys={d} qs={d} cache_rows={d} spans={d} (spanAttn {d}) split={} differing={d}/{d} {s}\n", .{ c.rows, c.first, t, c.qs, cap, n.spans, n.ctrl, did, d.n, d.of, if (ok) "PASS" else "FAIL" });
    if (c.naive > 0) {
        const nv = try naiveSplit(q, k, v, scale, c.rows, c.naive, s);
        defer free(nv);
        const dn = try differing(nv, want, s);
        std.debug.print("TURBOEXL3_LING_MLA_SPANS info: rows={d} first={d} split into {d}-row spans with no spanEnd rules: differing={d}/{d}\n", .{ c.rows, c.first, c.naive, dn.n, dn.of });
    }
    return ok;
}

test "turboexl3_ling MLA prompt spans: attn_spans.attend gives every row spanAttn's words (GPU)" {
    const s = try gpu();
    const cases = [_]Case{
        // the chunk loop's chunks of 4096 rows
        .{ .rows = 4096, .first = 0 },
        .{ .rows = 4096, .first = 4096 },
        .{ .rows = 4096, .first = 8192 },
        .{ .rows = 4096, .first = 12288 },
        .{ .rows = 4096, .first = 16384 },
        // the standard prompt's (18,380 tokens) last chunk, without and with a checkpoint's 30-row backoff, and the
        // backoff span itself (too short to split)
        .{ .rows = 1996, .first = 16384 },
        .{ .rows = 1965, .first = 16384 },
        .{ .rows = 31, .first = 18349, .split = false },
        // warm prompts: rows that take MLX's softmax loop from the first row (3000 keys before: the first span runs to
        // 1097 rows) and from 5000 keys before
        .{ .rows = 4096, .first = 3000, .naive = 512 },
        .{ .rows = 4096, .first = 5000 },
        // short prompts, a 16-row tail, and calls too short to split
        .{ .rows = 600, .first = 0 },
        .{ .rows = 4095, .first = 0 },
        .{ .rows = 528, .first = 0 },
        .{ .rows = 527, .first = 0, .split = false, .naive = 512 },
        .{ .rows = 512, .first = 0, .split = false },
        // layer-major widths: spanAttn's 4096-row spans, each split
        .{ .rows = 8192, .first = 0 },
        .{ .rows = 8192, .first = 3000 },
        // peaky softmax rows (many probabilities exactly 0), and keys and values that are the whole arrays
        .{ .rows = 4096, .first = 4096, .qs = 8.0 },
        .{ .rows = 4096, .first = 8192, .cap_extra = 0 },
    };
    var failed: usize = 0;
    var seed: u64 = 0x5EED_0E07;
    for (cases) |c| {
        failed += @intFromBool(!try runCase(s, c, seed));
        seed += 3;
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// The relative error of `a` against `b`, ||a - b|| / ||b||, in f64 on the host (a shape mismatch is infinite)
fn relErr(a: Array, b: Array, s: Stream) !f64 {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return std.math.inf(f64);
    var fa = mlx.mlx_array_new();
    defer free(fa);
    try mlx.check(mlx.mlx_astype(&fa, a, .float32, s));
    var ca = mlx.mlx_array_new();
    defer free(ca);
    try mlx.check(mlx.mlx_contiguous(&ca, fa, false, s));
    try mlx.check(mlx.mlx_array_eval(ca));
    var fb = mlx.mlx_array_new();
    defer free(fb);
    try mlx.check(mlx.mlx_astype(&fb, b, .float32, s));
    var cb = mlx.mlx_array_new();
    defer free(cb);
    try mlx.check(mlx.mlx_contiguous(&cb, fb, false, s));
    try mlx.check(mlx.mlx_array_eval(cb));
    const n = mlx.mlx_array_size(cb);
    const pa = mlx.mlx_array_data_float32(ca) orelse return error.NoData;
    const pb = mlx.mlx_array_data_float32(cb) orelse return error.NoData;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..n) |i| {
        const x: f64 = pa[i];
        const y: f64 = pb[i];
        num += (x - y) * (x - y);
        den += y * y;
    }
    return @sqrt(num / den);
}

/// The relative error past which the flash launch is wrong, not rounded, against the composed `spanAttn`: the composed
/// path rounds its scores, its probabilities and its output to bf16 and the launch its output, each a relative 2^-9 a
/// word, which leaves well under 1 % at these unit-scale operands. turboexl3's own test holds the launch to an f64
/// reference.
const flash_tolerance: f64 = 0.02;

test "turboexl3_ling MLA prompt attention: the causal flash launch (src/ling_flash.zig, first in mla.zig's spanAttn) is the composed spanAttn's attention within flash_tolerance at the chunk loop's chunks, the standard prompt's last chunk and backoff span and the narrowest call it takes, and the keys one row off land outside it (GPU)" {
    const s = try gpu();
    // (query rows, keys before them): the chunk loop's first and a later chunk, the standard prompt's last chunk
    // after a checkpoint's backoff and the backoff span, and the narrowest call the launch takes (`flash.min_rows`)
    const calls = [_][2]c_int{ .{ 4096, 0 }, .{ 4096, 12288 }, .{ 1965, 16384 }, .{ 31, 18349 }, .{ flash.min_rows, 18363 } };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    var failed: usize = 0;
    var seed: u64 = 0x5EED_F1A5;
    for (calls, 0..) |call, ci| {
        defer _ = mlx.mlx_clear_cache();
        const rows = call[0];
        const before = call[1];
        const t = before + rows;
        const cap = t + 37;
        const q = try randn(s, seed, &.{ 1, heads, rows, qk }, 1.0);
        defer free(q);
        const kc = try randn(s, seed + 1, &.{ 1, heads, cap, qk }, 1.0);
        defer free(kc);
        const vc = try randn(s, seed + 2, &.{ 1, heads, cap, vd }, 1.0);
        defer free(vc);
        seed += 3;
        const k = try slice4(kc, .{ 0, 0, 0, 0 }, .{ 1, heads, t, qk }, s);
        defer free(k);
        const v = try slice4(vc, .{ 0, 0, 0, 0 }, .{ 1, heads, t, vd }, s);
        defer free(v);
        const want = try spanAttn(q, k, v, scale, rows, s);
        defer free(want);
        var got = mlx.mlx_array_new();
        defer free(got);
        const took = flash.takes("causal", rows) and try flash.attend(&got, q, k, v, scale, s);
        const e = if (took) try relErr(got, want, s) else std.math.inf(f64);
        var ok = took and e <= flash_tolerance;
        std.debug.print("TURBOEXL3_LING_MLA_FLASH rows={d} first={d} keys={d} took={} against the composed spanAttn rel_l2={d:.5} (tolerance {d}) {s}\n", .{ rows, before, t, took, e, flash_tolerance, if (ok) "PASS" else "FAIL" });
        // the red check, at the first call: the launch over the cache's keys one row later (every row's scores over
        // other keys) must land outside the bound
        if (ci == 0) {
            const k1 = try slice4(kc, .{ 0, 0, 1, 0 }, .{ 1, heads, t + 1, qk }, s);
            defer free(k1);
            var red = mlx.mlx_array_new();
            defer free(red);
            const red_took = try flash.attend(&red, q, k1, v, scale, s);
            const er = if (red_took) try relErr(red, want, s) else 0;
            const red_ok = red_took and er > flash_tolerance;
            std.debug.print("TURBOEXL3_LING_MLA_FLASH_RED rows={d} first={d}: the keys one row off, against the composed spanAttn rel_l2={d:.4} (must be past {d}) {s}\n", .{ rows, before, er, flash_tolerance, if (red_ok) "PASS" else "FAIL" });
            ok = ok and red_ok;
        }
        failed += @intFromBool(!ok);
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test {
    _ = @import("mlx_serve/plugins/bailing_hybrid/attn_spans.zig");
}
