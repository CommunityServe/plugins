//! sx4's GPU check (`zig build gpu-check -Dgpu-check-mlx=<host tree>/lib/mlx`, an executable over the kit's `mlx` and
//! that tree's staged MLX; run only inside a GPU window): the two in-place kernels `MLX_SERVE_DSV41_SCRATCH_TAIL` serves,
//! compiled on the GPU at the served template arguments (K 6, D 5120, HC 4, bf16 streams, sources and shared rows, 24
//! source slots; rows 16,383 by default, then 4,097) and their words checked against the stock kernels' over synthetic
//! inputs:
//! - tail: TAIL_FUSE's stock kernel's new streams (a fresh output) against its write-into twin's over a second copy of
//!   the streams, in place: every word equal (and the copy's words changed: the twin wrote it, not a temporary); then
//!   `check`'s path (`expandIntoChecked`: the stock, then the twin over a third copy, compared in the one call) with the
//!   stock's words served.
//! - hc: SCRATCH's hc expand into a region (kernels/hc_expand_into.metal) out of place, against the same kernel over a
//!   second copy of the streams in place (its streams the region itself, as the layer after a twin reads them): every word
//!   equal (and the copy's words changed).
//! Lines `GPUCHECK start`, `GPUCHECK tail`, `GPUCHECK tail-check`, `GPUCHECK hc` (or `... error=` and `GPUCHECK mlx-error`
//! with MLX's message, a kernel that does not compile among them), `GPUCHECK DONE rc=` (0: every check passed).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const tail_fuse = @import("tail_fuse.zig");
const sx1 = @import("scratch_x1.zig");
const scratch = @import("scratch.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const K = 6;
const D = 5120;
const HC = 4;
const n_sources = tail_fuse.sources;

fn onError(msg: [*:0]const u8, _: ?*anyopaque) callconv(.c) void {
    std.debug.print("GPUCHECK mlx-error {s}\n", .{msg});
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn drop(x: *A) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
    x.* = none;
}

fn evalOne(x: A) !void {
    try mlx.check(mlx.mlx_array_eval(x));
}

/// A new evaluated array of normal draws (the same seed: the same words, in a buffer of its own).
fn normal(s: mlx.mlx_stream, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, scale: f32) !A {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dt, 0, scale, key, s));
    try evalOne(out);
    return out;
}

fn words(s: mlx.mlx_stream, a: A, b: A) !scratch.Tally {
    var t: scratch.Tally = .{};
    try scratch.compare(s, a, b, &t);
    return t;
}

fn tailCheck(s: mlx.mlx_stream, n: usize) !bool {
    const rows = ci(n);
    const per: usize = (n * K + n_sources - 1) / n_sources;
    var srcs: [n_sources]A = @splat(none);
    defer for (&srcs) |*x| drop(x);
    for (&srcs, 0..) |*x, i| x.* = try normal(s, &.{ ci(per), D }, .bfloat16, 100 + i, 1.0);
    // Assignment j = row * K + slot reads source j % 24 at its row j / 24.
    const a = std.heap.page_allocator;
    const host = try a.alloc(i32, n * K * 2);
    defer a.free(host);
    for (0..n * K) |j| {
        host[2 * j] = @intCast(j % n_sources);
        host[2 * j + 1] = @intCast(j / n_sources);
    }
    var loc = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ rows, K, 2 }, 3, .int32);
    defer drop(&loc);
    try evalOne(loc);
    var w = try normal(s, &.{ rows, K }, .float32, 200, 0.5);
    defer drop(&w);
    var sh = try normal(s, &.{ rows, D }, .bfloat16, 201, 1.0);
    defer drop(&sh);
    var pc = try normal(s, &.{ rows, HC + HC * HC }, .float32, 202, 0.5);
    defer drop(&pc);
    var res = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res);
    // The stock kernel's new streams, a fresh output.
    var fresh = try tail_fuse.expand(s, &srcs, loc, w, sh, res, pc);
    defer drop(&fresh);
    try evalOne(fresh);
    // The twin over a second copy of the streams, in place.
    var res2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res2);
    const copy0 = try words(s, res2, res);
    var carrier = try tail_fuse.expandInto(s, &srcs, loc, w, sh, res2, pc);
    defer drop(&carrier);
    try evalOne(carrier);
    const t = try words(s, res2, fresh);
    const changed = try words(s, res2, res);
    std.debug.print("GPUCHECK tail rows={d} copy_differ={d} words={d} differ={d} changed={d}\n", .{ n, copy0.differ, t.words, t.differ, changed.differ });
    var ok = copy0.differ == 0 and t.words > 0 and t.differ == 0 and changed.differ > 0;
    drop(&carrier);
    drop(&res2);
    _ = mlx.mlx_clear_cache();
    // `check`'s path over a third copy: the stock kernel's words served, the twin's compared with them in the call.
    var res3 = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res3);
    const r = try tail_fuse.expandIntoChecked(s, &srcs, loc, w, sh, res3, pc);
    var served = r.fresh;
    defer drop(&served);
    const t2 = try words(s, served, fresh);
    std.debug.print("GPUCHECK tail-check rows={d} words={d} differ={d} served_words={d} served_differ={d}\n", .{ n, r.words, r.differ, t2.words, t2.differ });
    ok = ok and r.words > 0 and r.differ == 0 and t2.differ == 0;
    return ok;
}

fn hcCheck(s: mlx.mlx_stream, n: usize) !bool {
    const rows = ci(n);
    var y4 = try normal(s, &.{ 1, rows, D }, .bfloat16, 300, 1.0);
    defer drop(&y4);
    var post = try normal(s, &.{ rows, HC }, .float32, 301, 0.5);
    defer drop(&post);
    var comb = try normal(s, &.{ rows, HC, HC }, .float32, 302, 0.5);
    defer drop(&comb);
    // Out of place: the streams expanded into a region of their own.
    var res = try normal(s, &.{ rows, HC, D }, .bfloat16, 303, 1.0);
    defer drop(&res);
    var dst = try normal(s, &.{ rows, HC, D }, .bfloat16, 304, 1.0);
    defer drop(&dst);
    var x4 = mlx.mlx_array_new();
    defer drop(&x4);
    try mlx.check(mlx.mlx_reshape(&x4, res, &[_]c_int{ 1, rows, HC, D }, 4, s));
    var c1 = try sx1.expandInto(s, y4, x4, post, comb, dst, x4);
    defer drop(&c1);
    try evalOne(c1);
    // In place: a second copy of the streams expanded over itself.
    var res2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 303, 1.0);
    defer drop(&res2);
    var x4b = mlx.mlx_array_new();
    defer drop(&x4b);
    try mlx.check(mlx.mlx_reshape(&x4b, res2, &[_]c_int{ 1, rows, HC, D }, 4, s));
    var c2 = try sx1.expandInto(s, y4, x4b, post, comb, res2, x4b);
    defer drop(&c2);
    try evalOne(c2);
    const t = try words(s, res2, dst);
    const changed = try words(s, res2, res);
    std.debug.print("GPUCHECK hc rows={d} words={d} differ={d} changed={d}\n", .{ n, t.words, t.differ, changed.differ });
    return t.words > 0 and t.differ == 0 and changed.differ > 0;
}

pub fn main() u8 {
    mlx.mlx_set_error_handler(&onError, null, null);
    const s = mlx.mlx_default_gpu_stream_new();
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, 1 << 30);
    const rows_list = [_]usize{ 16383, 4097 };
    std.debug.print("GPUCHECK start sources={d} buffers={d} of {d} K={d} D={d} HC={d} T=bfloat16 S=bfloat16\n", .{ n_sources, tail_fuse.kernel_buffers, tail_fuse.metal_buffers, K, D, HC });
    var rc: u8 = 0;
    for (rows_list) |n| {
        const ok_tail = tailCheck(s, n) catch |e| blk: {
            std.debug.print("GPUCHECK tail rows={d} error={s}\n", .{ n, @errorName(e) });
            break :blk false;
        };
        _ = mlx.mlx_clear_cache();
        const ok_hc = hcCheck(s, n) catch |e| blk: {
            std.debug.print("GPUCHECK hc rows={d} error={s}\n", .{ n, @errorName(e) });
            break :blk false;
        };
        _ = mlx.mlx_clear_cache();
        if (!ok_tail or !ok_hc) rc = 1;
    }
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("GPUCHECK DONE rc={d} peak_mb={d}\n", .{ rc, peak >> 20 });
    return rc;
}
