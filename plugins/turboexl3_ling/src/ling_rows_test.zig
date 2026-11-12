//! turboexl3_ling's dense-rows byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips): src/ling_rows.zig's
//! `denseRows` against one `matmul` per row (serial decode's one-row gemv), word for word, at Ling-3.1-flash's dense
//! verify shapes (the router 4096 -> 512 in f32 from bf16 and f32 rows; the MLA kv_b 512 -> 16384, its gate 4096 -> 64
//! and KDA's b 4096 -> 64 in bf16) and 2 to 8 rows. Each case prints a `TURBOEXL3_LING_ROWS` line ending in PASS or FAIL.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const rows_mod = @import("ling_rows.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

fn gpu() !Stream {
    // the GPU tests' own variable (test-only): at 1 they run, at any other value or unset they skip
    const run_gpu = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(run_gpu), "1")) return error.SkipZigTest;
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

const Case = struct { name: []const u8, k: c_int, n: c_int, w: mlx.mlx_dtype, x: mlx.mlx_dtype };
const cases = [_]Case{
    .{ .name = "router-bf16x", .k = 4096, .n = 512, .w = .float32, .x = .bfloat16 },
    .{ .name = "router-f32x", .k = 4096, .n = 512, .w = .float32, .x = .float32 },
    .{ .name = "mla-kv_b", .k = 512, .n = 16384, .w = .bfloat16, .x = .bfloat16 },
    .{ .name = "mla-gate", .k = 4096, .n = 64, .w = .bfloat16, .x = .bfloat16 },
    .{ .name = "kda-b", .k = 4096, .n = 64, .w = .bfloat16, .x = .bfloat16 },
};

test "turboexl3_ling gpu dense rows: every row of denseRows is its one-row matmul's, word for word (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    var failed: usize = 0;
    for (cases, 0..) |c, ci| {
        const w = try randn(s, 0x11A6 + @as(u64, ci), &.{ c.k, c.n }, 0.02, c.w);
        defer free(w);
        var rows: c_int = 2;
        while (rows <= 8) : (rows += 1) {
            const x = try randn(s, 0x5EED * (@as(u64, ci) + 1) + @as(u64, @intCast(rows)), &.{ 1, rows, c.k }, 1.0, c.x);
            defer free(x);
            const got = try rows_mod.denseRows(s, x, w);
            defer free(got);
            const parts = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(parts);
            var r: c_int = 0;
            while (r < rows) : (r += 1) {
                var row = mlx.mlx_array_new();
                defer free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, c.k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                var y = mlx.mlx_array_new();
                defer free(y);
                try mlx.check(mlx.mlx_matmul(&y, row, w, s));
                _ = mlx.mlx_vector_array_append_value(parts, y);
            }
            var want = mlx.mlx_array_new();
            defer free(want);
            try mlx.check(mlx.mlx_concatenate_axis(&want, parts, 1, s));
            const d = try differing(got, want, s);
            const n = mlx.mlx_array_size(want);
            std.debug.print("TURBOEXL3_LING_ROWS {s} k={d} n={d} rows={d} differing={d}/{d} {s}\n", .{ c.name, c.k, c.n, rows, d, n, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// What the loader makes of a plain weight (plugins/glm5_next/trunk.zig): the pack's `[out, in]` f16 tensor, its
/// transposed view (`transposed`), then `lin`'s astype to bf16, or for the router `f32Table`'s astype to f32 and
/// `materializedOwnedCopy`'s add of a zero scalar: `[in, out]` with the view's strides, evaluated
const Made = enum { plain, router };
fn servedWeight(s: Stream, seed: u64, k: c_int, n: c_int, made: Made) !Array {
    const raw = try randn(s, seed, &.{ n, k }, 0.02, .float16);
    defer free(raw);
    var t = mlx.mlx_array_new();
    defer free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, raw, &[_]c_int{ 1, 0 }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    switch (made) {
        .plain => try mlx.check(mlx.mlx_astype(&out, t, .bfloat16, s)),
        .router => {
            var f = mlx.mlx_array_new();
            defer free(f);
            try mlx.check(mlx.mlx_astype(&f, t, .float32, s));
            var zero = mlx.mlx_array_new();
            defer free(zero);
            try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 0, .float32, s));
            try mlx.check(mlx.mlx_add(&out, f, zero, s));
        },
    }
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

const ServedCase = struct { name: []const u8, k: c_int, n: c_int, made: Made, x: mlx.mlx_dtype };
const served_cases = [_]ServedCase{
    .{ .name = "router-bf16x", .k = 4096, .n = 512, .made = .router, .x = .bfloat16 },
    .{ .name = "router-f32x", .k = 4096, .n = 512, .made = .router, .x = .float32 },
    .{ .name = "mla-kv_b", .k = 512, .n = 16384, .made = .plain, .x = .bfloat16 },
    .{ .name = "mla-gate", .k = 4096, .n = 64, .made = .plain, .x = .bfloat16 },
    .{ .name = "kda-b", .k = 4096, .n = 64, .made = .plain, .x = .bfloat16 },
};

test "turboexl3_ling gpu dense rows, served layout: every row of denseRows is its one-row matmul's over the weight as the loader makes it, word for word (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    var failed: usize = 0;
    for (served_cases, 0..) |c, ci| {
        const w = try servedWeight(s, 0x5E7D + @as(u64, ci), c.k, c.n, c.made);
        defer free(w);
        const st = mlx.mlx_array_strides(w);
        var rows: c_int = 2;
        while (rows <= 8) : (rows += 1) {
            const x = try randn(s, 0xA0D1 * (@as(u64, ci) + 1) + @as(u64, @intCast(rows)), &.{ 1, rows, c.k }, 1.0, c.x);
            defer free(x);
            const got = try rows_mod.denseRows(s, x, w);
            defer free(got);
            const parts = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(parts);
            var r: c_int = 0;
            while (r < rows) : (r += 1) {
                var row = mlx.mlx_array_new();
                defer free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, c.k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                var y = mlx.mlx_array_new();
                defer free(y);
                try mlx.check(mlx.mlx_matmul(&y, row, w, s));
                _ = mlx.mlx_vector_array_append_value(parts, y);
            }
            var want = mlx.mlx_array_new();
            defer free(want);
            try mlx.check(mlx.mlx_concatenate_axis(&want, parts, 1, s));
            const d = try differing(got, want, s);
            const n = mlx.mlx_array_size(want);
            std.debug.print("TURBOEXL3_LING_ROWS_SERVED {s} k={d} n={d} w={s} w_strides=({d},{d}) rows={d} differing={d}/{d} {s}\n", .{ c.name, c.k, c.n, @tagName(mlx.mlx_array_dtype(w)), st[0], st[1], rows, d, n, if (d == 0) "PASS" else "FAIL" });
            if (d != 0) failed += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}
