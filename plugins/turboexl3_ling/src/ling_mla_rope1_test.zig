//! turboexl3_ling's rope-once MLA cache byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips):
//! src/ling_mla_rope1.zig against the expanded (per-head key) cache built here from the same kv_b rows and roped keys,
//! word for word, at Ling-3.1-flash's MLA heads (64; nope 128, rope 64, v 128; bf16), both caches views of the first T
//! rows of a wider buffer, as the cache hands them. The expanded cache is upstream mla.zig's layout, which the copy no
//! longer serves: this test keeps it as the oracle.
//! - the cache's words: the rope-once views' nope keys, values and roped keys against the expanded cache (its keys'
//!   first 128 words of every head, its values, its keys' last 64 words of every head);
//! - decode and verify rows: verify_rows.zig `attendRope1` at 1, 2, 4, 8 and 16 rows whose first row sees N0 keys (100;
//!   1,023 and 1,024 across MLX's one-pass / two-pass threshold; 8,192 across the blocks threshold; 18,380) against the
//!   expanded cache's call: MLX's one-row SDPA call at one row (a decode), `attend` at more (the exact verify's rows,
//!   each its one-row call's words, which src/ling_verify_mla_test.zig holds to one-row SDPA calls);
//! - prompt rows: `flash` (turboexl3's rope-once launch, `Config.mla_rope_once`) against turboexl3's flash launch over
//!   the expanded cache (src/ling_flash.zig's call) at 17, 512 and 4,096 rows after 1,000 cached keys, where the
//!   device passes the NAX gate;
//! - red checks that must differ: token 0's kv_b row negated in the rope-once cache, and token 0's roped key negated.
//! Each case prints a `TURBOEXL3_LING_ROPE1` line ending in PASS or FAIL; nothing is timed.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const rope1 = @import("ling_mla_rope1.zig");
const vr = @import("mlx_serve/plugins/bailing_hybrid/verify_rows.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

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

/// normal(0, scale) in `dt` from `seed`, evaluated
fn randn(s: Stream, seed: u64, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0, scale, key, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word)
fn differing(a: Array, b: Array, s: Stream) !usize {
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return n;
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
    return d;
}

/// Ling-3.1-flash's MLA layer: heads, a key's nope and roped dims, a value's, kv_b's row a token
const n_heads: c_int = 64;
const d_nope: c_int = 128;
const d_rope: c_int = 64;
const d_v: c_int = 128;
const d_row: c_int = n_heads * (d_nope + d_v);

/// `a`'s elements [lo, hi) along axis 2 and axis 3's [c0, c1) (a view)
fn cut(a: Array, lo: c_int, hi: c_int, c0: c_int, c1: c_int, s: Stream) !Array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    const st4 = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&out, a, &[_]c_int{ 0, 0, lo, c0 }, 4, &[_]c_int{ sh[0], sh[1], hi, c1 }, 4, &st4, 4, s));
    return out;
}

/// The first `t` rows of `a` `[B, H, cap, D]` (a view, as the cache hands its rows)
fn first(a: Array, t: c_int, s: Stream) !Array {
    return cut(a, 0, t, 0, mlx.getShape(a)[3], s);
}

/// `a` `[B, H, T, D]` with row 0 negated (a red check's operand), evaluated
fn negRow0(a: Array, s: Stream) !Array {
    const sh = mlx.getShape(a);
    const r0 = try cut(a, 0, 1, 0, sh[3], s);
    defer free(r0);
    const rest = try cut(a, 1, sh[2], 0, sh[3], s);
    defer free(rest);
    var neg = mlx.mlx_array_new();
    defer free(neg);
    try mlx.check(mlx.mlx_negative(&neg, r0, s));
    const parts = [_]Array{ neg, rest };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Both caches of `cap` tokens from one draw of kv_b's rows `[1, cap, H (nope + v)]` and roped keys `[1, 1, cap, rope]`:
/// as the rope-once cache keeps them (`rows` `[1, 1, cap, H (nope + v)]`, `rope`), and as upstream mla.zig writes the
/// expanded cache (`keys` `[1, H, cap, nope + rope]`: each head's nope keys ++ the broadcast roped key; `values`
/// `[1, H, cap, v]`, contiguous), each evaluated
const Caches = struct {
    rows: Array,
    rope: Array,
    keys: Array,
    values: Array,

    fn deinit(self: Caches) void {
        free(self.rows);
        free(self.rope);
        free(self.keys);
        free(self.values);
    }
};

fn caches(s: Stream, seed: u64, cap: c_int) !Caches {
    const kv_flat = try randn(s, seed, &.{ 1, cap, d_row }, 1.0, .bfloat16);
    defer free(kv_flat);
    const rope = try randn(s, seed + 1, &.{ 1, 1, cap, d_rope }, 1.0, .bfloat16);
    errdefer free(rope);
    var rows = mlx.mlx_array_new();
    errdefer free(rows);
    try mlx.check(mlx.mlx_reshape(&rows, kv_flat, &[_]c_int{ 1, 1, cap, d_row }, 4, s));
    // upstream mla.zig's expanded cache: kv_b's rows to heads, the nope and value halves, the roped key for every head
    var kv_heads = mlx.mlx_array_new();
    defer free(kv_heads);
    try mlx.check(mlx.mlx_reshape(&kv_heads, kv_flat, &[_]c_int{ 1, cap, n_heads, d_nope + d_v }, 4, s));
    var kv_t = mlx.mlx_array_new();
    defer free(kv_t);
    try mlx.check(mlx.mlx_transpose_axes(&kv_t, kv_heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const k_nope = try cut(kv_t, 0, cap, 0, d_nope, s);
    defer free(k_nope);
    const v_t = try cut(kv_t, 0, cap, d_nope, d_nope + d_v, s);
    defer free(v_t);
    var k_rot_b = mlx.mlx_array_new();
    defer free(k_rot_b);
    try mlx.check(mlx.mlx_broadcast_to(&k_rot_b, rope, &[_]c_int{ 1, n_heads, cap, d_rope }, 4, s));
    const parts = [_]Array{ k_nope, k_rot_b };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var keys = mlx.mlx_array_new();
    errdefer free(keys);
    try mlx.check(mlx.mlx_concatenate_axis(&keys, vec, -1, s));
    var values = mlx.mlx_array_new();
    errdefer free(values);
    try mlx.check(mlx.mlx_contiguous(&values, v_t, false, s));
    for ([_]Array{ rows, keys, values }) |x| try mlx.check(mlx.mlx_array_eval(x));
    return .{ .rows = rows, .rope = rope, .keys = keys, .values = values };
}

fn passFail(ok: bool) []const u8 {
    return if (ok) "PASS" else "FAIL";
}

test "turboexl3_ling gpu rope-once cache: its views hold the expanded cache's words, and its decode and verify rows (attendRope1) and prompt rows (the rope-once flash launch) give the same calls' words over the expanded cache; a changed kv_b row and a changed roped key differ (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    const cap: c_int = 18380 + 16 + 64;
    const c = try caches(s, 0x201E, cap);
    defer c.deinit();
    const scale: f32 = 1.0 / @sqrt(192.0);
    var lines: usize = 0;
    var failed: usize = 0;
    // 1. the cache's words, every cached row
    {
        var vw = try rope1.views(c.rows, c.rope, n_heads, d_nope, d_v, s);
        defer vw.deinit();
        const kn = try cut(c.keys, 0, cap, 0, d_nope, s);
        defer free(kn);
        const kr = try cut(c.keys, 0, cap, d_nope, d_nope + d_rope, s);
        defer free(kr);
        var rb = mlx.mlx_array_new();
        defer free(rb);
        try mlx.check(mlx.mlx_broadcast_to(&rb, vw.rope, &[_]c_int{ 1, n_heads, cap, d_rope }, 4, s));
        const dn = try differing(vw.nope, kn, s);
        const dv = try differing(vw.v, c.values, s);
        const dr = try differing(rb, kr, s);
        const ok = dn == 0 and dv == 0 and dr == 0;
        std.debug.print("TURBOEXL3_LING_ROPE1 words tokens={d} nope_differing={d} v_differing={d} rope_differing={d} {s}\n", .{ cap, dn, dv, dr, passFail(ok) });
        lines += 1;
        failed += @intFromBool(!ok);
    }
    // 2. decode and verify rows: the expanded cache's call over its first T rows against attendRope1 over the rope-once
    // cache's
    const none = mlx.mlx_array_new();
    defer free(none);
    for ([_]c_int{ 100, 1023, 1024, 8192, 18380 }) |n0| {
        for ([_]c_int{ 1, 2, 4, 8, 16 }) |rows| {
            const t = n0 + rows - 1;
            const kv = try first(c.keys, t, s);
            defer free(kv);
            const vv = try first(c.values, t, s);
            defer free(vv);
            const rv = try first(c.rows, t, s);
            defer free(rv);
            const ropev = try first(c.rope, t, s);
            defer free(ropev);
            var vw = try rope1.views(rv, ropev, n_heads, d_nope, d_v, s);
            defer vw.deinit();
            const q = try randn(s, 0x51 + @as(u64, @intCast(n0)) * 31 + @as(u64, @intCast(rows)), &.{ 1, n_heads, rows, d_nope + d_rope }, 1.0, .bfloat16);
            defer free(q);
            var want = mlx.mlx_array_new();
            defer free(want);
            var served = true;
            if (rows == 1) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&want, q, kv, vv, scale, "", none, .{ .ctx = null }, false, s));
            } else served = try vr.attend(&want, q, kv, vv, scale, rows, s);
            var got = mlx.mlx_array_new();
            defer free(got);
            const took = try vr.attendRope1(&got, q, vw.nope, vw.v, vw.rope, scale, rows, s);
            const n = @as(usize, @intCast(rows)) * @as(usize, @intCast(n_heads * d_v));
            const d = if (served and took) try differing(got, want, s) else n;
            const ok = served and took and d == 0;
            std.debug.print("TURBOEXL3_LING_ROPE1 rows n0={d} rows={d} keys={d} served={s} took={} groups={d} differing={d}/{d} {s}\n", .{ n0, rows, t, if (rows == 1) "sdpa" else "attend", took, vr.last.groups, d, n, passFail(ok) });
            lines += 1;
            failed += @intFromBool(!ok);
            // the red checks, at 4 rows over 100 keys: the rope-once cache with token 0's kv_b row negated, then with
            // its roped key negated, each against the expanded cache's words of the unchanged cache
            if (n0 != 100 or rows != 4) continue;
            const reds = [_][]const u8{ "kv_row0_negated", "rope0_negated" };
            for (reds, 0..) |name, ri| {
                const rows_red = if (ri == 0) try negRow0(rv, s) else try first(c.rows, t, s);
                defer free(rows_red);
                const rope_red = if (ri == 1) try negRow0(ropev, s) else try first(c.rope, t, s);
                defer free(rope_red);
                var vr_red = try rope1.views(rows_red, rope_red, n_heads, d_nope, d_v, s);
                defer vr_red.deinit();
                var red = mlx.mlx_array_new();
                defer free(red);
                const red_took = try vr.attendRope1(&red, q, vr_red.nope, vr_red.v, vr_red.rope, scale, rows, s);
                const dd = if (red_took) try differing(red, want, s) else 0;
                const red_ok = red_took and dd > 0;
                std.debug.print("TURBOEXL3_LING_ROPE1_RED check={s} n0={d} rows={d} differing={d}/{d} (a changed word must differ) {s}\n", .{ name, n0, rows, dd, n, passFail(red_ok) });
                lines += 1;
                failed += @intFromBool(!red_ok);
            }
        }
    }
    // 3. prompt rows: turboexl3's flash launch over the expanded cache (src/ling_flash.zig's call) against the rope-once
    // launch, Config.mla_rope_once set as src/ling_math.zig `configure` sets it
    {
        const saved = t3.config.get().*;
        defer t3.configure(saved) catch {};
        var cf = saved;
        cf.mla_rope_once = true;
        try t3.configure(cf);
        const prompt_rows = [_]c_int{ 17, 512, 4096 };
        if (!cf.nax) std.debug.print("TURBOEXL3_LING_ROPE1 prompt skipped: the device fails turboexl3's NAX gate\n", .{});
        for (prompt_rows) |pr| {
            if (!cf.nax) break;
            const t = 1000 + pr;
            const kv = try first(c.keys, t, s);
            defer free(kv);
            const vv = try first(c.values, t, s);
            defer free(vv);
            const rv = try first(c.rows, t, s);
            defer free(rv);
            const ropev = try first(c.rope, t, s);
            defer free(ropev);
            var vw = try rope1.views(rv, ropev, n_heads, d_nope, d_v, s);
            defer vw.deinit();
            const q = try randn(s, 0x9F + @as(u64, @intCast(pr)), &.{ 1, n_heads, pr, d_nope + d_rope }, 1.0, .bfloat16);
            defer free(q);
            const want = try t3.ling.mlaFlashPrefill(s, q, kv, vv, scale);
            defer free(want);
            var got = mlx.mlx_array_new();
            defer free(got);
            const took = try rope1.flash(&got, q, vw, scale, s);
            const n = @as(usize, @intCast(pr)) * @as(usize, @intCast(n_heads * d_v));
            const d = if (took) try differing(got, want, s) else n;
            const ok = took and d == 0;
            std.debug.print("TURBOEXL3_LING_ROPE1 prompt rows={d} keys={d} took={} differing={d}/{d} {s}\n", .{ pr, t, took, d, n, passFail(ok) });
            lines += 1;
            failed += @intFromBool(!ok);
        }
    }
    std.debug.print("TURBOEXL3_LING_ROPE1_DONE {d} lines, {d} failed\n", .{ lines, failed });
    try std.testing.expectEqual(@as(usize, 0), failed);
}
