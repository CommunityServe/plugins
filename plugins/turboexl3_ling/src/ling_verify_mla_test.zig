//! turboexl3_ling's exact-verify MLA rows byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips): the
//! copied plugins/bailing_hybrid/verify_rows.zig `attend` against mla.zig's `rowAttn` path (each query row its own
//! one-row MLX SDPA call over the keys up to it), word for word, at Ling-3.1-flash's expanded MLA heads (64 heads, qk
//! 192, v 128, bf16; and qk = v = 128) for verifies of 2 to 8 rows whose first row sees N0 keys: N0 short and long
//! (100, 1024, 18,380) and straddling MLX's one-pass / two-pass threshold (1,024 keys) and the two-pass blocks
//! thresholds (8,192), the keys and values views of a wider cache. Each case prints a `TURBOEXL3_LING_VERIFY_MLA` line
//! (the device class, each row's plan, the groups) ending in PASS or FAIL; nothing is timed.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
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

fn slice4(a: Array, stop: [4]c_int, s: Stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(mlx.mlx_slice(&r, a, &[_]c_int{ 0, 0, 0, 0 }, 4, &stop, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return r;
}

/// mla.zig's `rowAttn`: each query row r of q [1, H, S, D] alone, one MLX SDPA call over the first T - S + r + 1 keys,
/// the rows joined on axis 2
fn rowAttn(q: Array, k: Array, v: Array, scale: f32, seq_len: c_int, s: Stream) !Array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const first = ks[2] - seq_len;
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    const none = mlx.mlx_array_new();
    defer free(none);
    var r: c_int = 0;
    while (r < seq_len) : (r += 1) {
        var qr = mlx.mlx_array_new();
        defer free(qr);
        try mlx.check(mlx.mlx_slice(&qr, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ qs[0], qs[1], r + 1, qs[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const kr = try slice4(k, .{ ks[0], ks[1], first + r + 1, ks[3] }, s);
        defer free(kr);
        const vrr = try slice4(v, .{ vs[0], vs[1], first + r + 1, vs[3] }, s);
        defer free(vrr);
        var o = mlx.mlx_array_new();
        defer free(o);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, qr, kr, vrr, scale, "", none, .{ .ctx = null }, false, s));
        _ = mlx.mlx_vector_array_append_value(parts, o);
    }
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 2, s));
    return out;
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

/// One verify: H heads of (qk, v), S rows, the first row over n0 keys (so T = n0 + S - 1 keys), the keys and values
/// the first T rows of a cache of `cap` rows
const Case = struct { h: c_int = 64, qk: c_int = 192, v: c_int = 128, rows: c_int, n0: c_int, cap_extra: c_int = 37, dt: mlx.mlx_dtype = .bfloat16 };

fn runCase(s: Stream, c: Case, seed: u64) !bool {
    const t = c.n0 + c.rows - 1;
    const cap = t + c.cap_extra;
    const q = try randn(s, seed, &.{ 1, c.h, c.rows, c.qk }, 1.0, c.dt);
    defer free(q);
    const kc = try randn(s, seed + 1, &.{ 1, c.h, cap, c.qk }, 1.0, c.dt);
    defer free(kc);
    const vc = try randn(s, seed + 2, &.{ 1, c.h, cap, c.v }, 1.0, c.dt);
    defer free(vc);
    const k = try slice4(kc, .{ 1, c.h, t, c.qk }, s);
    defer free(k);
    const v = try slice4(vc, .{ 1, c.h, t, c.v }, s);
    defer free(v);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.qk)));
    const want = try rowAttn(q, k, v, scale, c.rows, s);
    defer free(want);
    var got = mlx.mlx_array_new();
    defer free(got);
    const served = try vr.attend(&got, q, k, v, scale, c.rows, s);
    const devc = vr.deviceClass();
    var plan_buf: [256]u8 = undefined;
    var at: usize = 0;
    var r: c_int = 0;
    while (r < c.rows) : (r += 1) {
        const p = vr.planRow(devc, 1, @intCast(c.n0 + r), vr.blocksEnv());
        const w = std.fmt.bufPrint(plan_buf[at..], "{s}{s}", .{ if (r == 0) "" else ",", if (p.kind == .one_pass) "1p" else "2p" }) catch break;
        at += w.len;
        if (p.kind == .two_pass) {
            const b = std.fmt.bufPrint(plan_buf[at..], "{d}", .{p.blocks}) catch break;
            at += b.len;
        }
    }
    const d: Diff = if (served) try differing(got, want, s) else .{ .n = mlx.mlx_array_size(want), .of = mlx.mlx_array_size(want) };
    const ok = served and d.n == 0;
    std.debug.print("TURBOEXL3_LING_VERIFY_MLA h={d} qk={d} v={d} x={s} rows={d} n0={d} devc={c} plans={s} groups={d} served={} differing={d}/{d} {s}\n", .{ c.h, c.qk, c.v, @tagName(c.dt), c.rows, c.n0, if (devc == 0) '?' else devc, plan_buf[0..at], if (served) vr.last.groups else 0, served, d.n, d.of, if (ok) "PASS" else "FAIL" });
    return ok;
}

test "turboexl3_ling exact verify MLA rows: verify_rows.attend gives each row its one-row MLX SDPA call's words (rowAttn), 2 to 8 rows across the one-pass / two-pass and blocks thresholds (GPU)" {
    const s = try gpu();
    var failed: usize = 0;
    var seed: u64 = 0x5EED_0A11;
    // 8 rows from short to long contexts, straddling 1,024 (rows 1,017..1,024) and 8,192 (rows 8,190..8,197), and the
    // two-pass tail sizes (4,093..4,100: each row a different N mod 4 blocks)
    for ([_]c_int{ 100, 1017, 1024, 4093, 8190, 18380 }) |n0| {
        failed += @intFromBool(!try runCase(s, .{ .rows = 8, .n0 = n0 }, seed));
        seed += 3;
    }
    // 2 to 7 rows at a short, a straddling and a long first row
    for ([_]c_int{ 2, 3, 4, 5, 6, 7 }) |rows| {
        for ([_]c_int{ 100, 1022, 18380 }) |n0| {
            failed += @intFromBool(!try runCase(s, .{ .rows = rows, .n0 = n0 }, seed));
            seed += 3;
        }
    }
    // f16, and the other served head dims
    failed += @intFromBool(!try runCase(s, .{ .rows = 4, .n0 = 1022, .dt = .float16 }, seed));
    failed += @intFromBool(!try runCase(s, .{ .rows = 5, .n0 = 1021, .qk = 128, .v = 128 }, seed + 3));
    failed += @intFromBool(!try runCase(s, .{ .rows = 8, .n0 = 2000, .qk = 128, .v = 128, .h = 8 }, seed + 6));
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3_ling exact verify MLA rows: shapes it does not serve are declined before any launch (GPU)" {
    const s = try gpu();
    const q = try randn(s, 7, &.{ 1, 4, 3, 192 }, 1.0, .bfloat16);
    defer free(q);
    const k = try randn(s, 8, &.{ 1, 4, 10, 192 }, 1.0, .bfloat16);
    defer free(k);
    const v = try randn(s, 9, &.{ 1, 4, 10, 128 }, 1.0, .bfloat16);
    defer free(v);
    var out = mlx.mlx_array_new();
    defer free(out);
    // f32 operands; GQA (2 key heads for 4 query heads); head dims (192, 192); more rows than the query has
    const qf = try randn(s, 10, &.{ 1, 4, 3, 192 }, 1.0, .float32);
    defer free(qf);
    try std.testing.expect(!try vr.attend(&out, qf, k, v, 0.1, 3, s));
    const k2 = try randn(s, 11, &.{ 1, 2, 10, 192 }, 1.0, .bfloat16);
    defer free(k2);
    const v2 = try randn(s, 12, &.{ 1, 2, 10, 128 }, 1.0, .bfloat16);
    defer free(v2);
    try std.testing.expect(!try vr.attend(&out, q, k2, v2, 0.1, 3, s));
    const v192 = try randn(s, 13, &.{ 1, 4, 10, 192 }, 1.0, .bfloat16);
    defer free(v192);
    try std.testing.expect(!try vr.attend(&out, q, k, v192, 0.1, 3, s));
    try std.testing.expect(!try vr.attend(&out, q, k, v, 0.1, 4, s));
}

test "turboexl3_ling exact verify MLA rows, served layout: keys and values that are the cache's whole buffers (the cache's view takes the whole buffer when its rows fill it) give each row its one-row call's words (GPU)" {
    const s = try gpu();
    var failed: usize = 0;
    var seed: u64 = 0x5EED_0B22;
    for ([_][2]c_int{ .{ 8, 100 }, .{ 8, 1017 }, .{ 8, 8190 }, .{ 3, 18380 }, .{ 2, 1022 } }) |rn| {
        failed += @intFromBool(!try runCase(s, .{ .rows = rn[0], .n0 = rn[1], .cap_extra = 0 }, seed));
        seed += 3;
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}
