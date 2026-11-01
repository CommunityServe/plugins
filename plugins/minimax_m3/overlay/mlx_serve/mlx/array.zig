//! This plugin's own, in place of mlx-serve's `src/mlx/array.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! array helpers the copies call, over mlx-c (`ffi.zig`), each the same mlx-c ops in the same order as the helper it
//! stands in for, since the copies' graphs (and so their numerics) are built from them.

const std = @import("std");
const mlx = @import("ffi.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// The rows `idx` of `src` along `axis`, contiguous (dequantize misreads a strided take).
pub fn takeContig(s: Stream, src: Array, idx: Array, axis: c_int) !Array {
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_take_axis(&view, src, idx, axis, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, view, false, s));
    return out;
}

/// A copy of `x` that owns its buffer (`x + 0` in `x`'s dtype): a view's source can be freed once it is evaluated.
pub fn materializedOwnedCopy(s: Stream, x: Array) !Array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 0, mlx.mlx_array_dtype(x), s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, x, zero, s));
    return out;
}

fn concat(s: Stream, parts: []const Array, axis: c_int) !Array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, s));
    return out;
}

pub fn concatAxis0(s: Stream, parts: []const Array) !Array {
    return concat(s, parts, 0);
}

pub fn concatAxis1(s: Stream, parts: []const Array) !Array {
    return concat(s, parts, 1);
}

/// Index `i` of `axis`, the axis kept (a view).
pub fn axisView(s: Stream, arr: Array, axis: usize, i: usize) !Array {
    const shape = mlx.getShape(arr);
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    @memcpy(stop[0..shape.len], shape);
    start[axis] = @intCast(i);
    stop[axis] = @intCast(i + 1);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, shape.len, &stop, shape.len, &strides, shape.len, s));
    return out;
}

/// `val` as a 0-dim array of dtype `dt`.
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: Stream) !Array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

/// `val` as a 0-dim bf16 array (a failed cast leaves the empty handle).
pub fn bf16Scalar(val: f32, s: Stream) Array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, raw, .bfloat16, s);
    return out;
}

/// The last dimension when `x` is set and has a positive one.
pub inline fn lastDim(x: Array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0 or shape[shape.len - 1] <= 0) return null;
    return @intCast(shape[shape.len - 1]);
}

/// `w` with its last two axes swapped (a view).
pub fn transposeBf16Weight(w: Array, s: Stream) !Array {
    const n = mlx.getShape(w).len;
    var perm: [8]c_int = undefined;
    for (perm[0..n], 0..) |*p, i| p.* = @intCast(i);
    std.mem.swap(c_int, &perm[n - 2], &perm[n - 1]);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, w, &perm, n, s));
    return out;
}

/// A `[E, 2M, K]` gate|up pack split on axis 1 into contiguous `[E, M, K]` halves: both views, then both copies.
pub fn splitPackedGateUp(arr: Array, s: Stream) !struct { gate: Array, up: Array } {
    const shape = mlx.getShape(arr);
    if (shape.len != 3) return error.BadPackedGateUpShape;
    const m = @divExact(shape[1], 2);
    const ones = [_]c_int{ 1, 1, 1 };
    var gate_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_view);
    try mlx.check(mlx.mlx_slice(&gate_view, arr, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ shape[0], m, shape[2] }, 3, &ones, 3, s));
    var up_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_view);
    try mlx.check(mlx.mlx_slice(&up_view, arr, &[_]c_int{ 0, m, 0 }, 3, &[_]c_int{ shape[0], shape[1], shape[2] }, 3, &ones, 3, s));
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_contiguous(&gate, gate_view, false, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_contiguous(&up, up_view, false, s));
    return .{ .gate = gate, .up = up };
}
