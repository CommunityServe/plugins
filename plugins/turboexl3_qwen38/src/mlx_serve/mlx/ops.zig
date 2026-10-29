//! copied_from: src/mlx/ops.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MLX for Zig: a Graph owns every array it makes until it ends; `call` runs any mlx-c op.
//! A helper that returns one array opens its own `scope()` and hands the result up with `keep`.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("ffi.zig");

pub const Array = mlx.mlx_array;
pub const Dtype = mlx.mlx_dtype;
pub const Stream = mlx.mlx_stream;

/// Handles a Graph made and has not freed yet; counted in tests only (leak checks).
var live_for_test = std.atomic.Value(i64).init(0);

fn counted(x: Array) Array {
    if (builtin.is_test) _ = live_for_test.fetchAdd(1, .monotonic);
    return x;
}

fn release(x: Array) void {
    if (builtin.is_test) _ = live_for_test.fetchSub(1, .monotonic);
    _ = mlx.mlx_array_free(x);
}

/// The handles one Graph owns: an inline buffer first, a heap list past it.
const Owned = struct {
    const INLINE = 64;
    buf: [INLINE]Array = undefined,
    n: usize = 0,
    heap: std.ArrayList(Array) = .empty,

    fn add(o: *Owned, a: std.mem.Allocator, x: Array) !void {
        if (o.n < INLINE) {
            o.buf[o.n] = x;
            o.n += 1;
        } else try o.heap.append(a, x);
    }

    fn deinit(o: *Owned, a: std.mem.Allocator) void {
        for (o.buf[0..o.n]) |x| release(x);
        for (o.heap.items) |x| release(x);
        o.heap.deinit(a);
        o.* = .{};
    }
};

pub const Mask = union(enum) { none, causal, array: Array };

pub const Graph = struct {
    a: std.mem.Allocator,
    s: Stream,
    owned: Owned = .{},

    /// Never copy a Graph once it owns arrays: both copies would free them.
    pub fn init(a: std.mem.Allocator, s: Stream) Graph {
        return .{ .a = a, .s = s };
    }

    /// Frees every array not kept elsewhere.
    pub fn deinit(g: *Graph) void {
        g.owned.deinit(g.a);
    }

    /// Any mlx-c op of the shape `f(*out, args..., [stream])`; the compiler checks `args`
    /// against mlx-c's signature, so an MLX bump that changes an op is a compile error.
    pub fn call(g: *Graph, comptime f: anytype, args: anytype) !Array {
        const info = @typeInfo(@TypeOf(f)).@"fn";
        comptime std.debug.assert(info.param_types[0].? == *Array and info.return_type.? == c_int);
        const takes_stream = comptime info.param_types[info.param_types.len - 1].? == Stream;
        var out = counted(mlx.mlx_array_new());
        errdefer release(out);
        try mlx.check(if (takes_stream) @call(.auto, f, .{&out} ++ args ++ .{g.s}) else @call(.auto, f, .{&out} ++ args));
        try g.owned.add(g.a, out);
        return out;
    }

    /// Takes ownership of a handle made outside the Graph (freed here on failure).
    pub fn adopt(g: *Graph, x: Array) !Array {
        return g.adoptCounted(counted(x));
    }

    fn adoptCounted(g: *Graph, x: Array) !Array {
        g.owned.add(g.a, x) catch |e| {
            release(x);
            return e;
        };
        return x;
    }

    // ── one line per op ──
    // zig fmt: off
    pub fn add(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_add, .{ x, y }); }
    pub fn mul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_multiply, .{ x, y }); }
    pub fn matmul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_matmul, .{ x, y }); }
    pub fn sigmoid(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sigmoid, .{x}); }
    pub fn astype(g: *Graph, x: Array, dt: Dtype) !Array { return g.call(mlx.mlx_astype, .{ x, dt }); }
    pub fn reshape(g: *Graph, x: Array, sh: []const c_int) !Array { return g.call(mlx.mlx_reshape, .{ x, sh.ptr, sh.len }); }
    pub fn expandDims(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_expand_dims, .{ x, axis }); }
    pub fn take(g: *Graph, x: Array, ids: Array, axis: c_int) !Array { return g.call(mlx.mlx_take_axis, .{ x, ids, axis }); }
    pub fn sum(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_sum_axis, .{ x, axis, keepdims }); }
    pub fn mean(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_mean_axis, .{ x, axis, keepdims }); }
    pub fn argsort(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_argsort_axis, .{ x, axis }); }
    pub fn zeros(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_zeros, .{ sh.ptr, sh.len, dt }); }
    pub fn ones(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_ones, .{ sh.ptr, sh.len, dt }); }
    pub fn rmsNorm(g: *Graph, x: Array, w: Array, eps: f32) !Array { return g.call(mlx.mlx_fast_rms_norm, .{ x, w, eps }); }

    // zig fmt: on

    /// x[start:stop] per axis, unit strides.
    pub fn slice(g: *Graph, x: Array, start: []const c_int, stop: []const c_int) !Array {
        const ones8: [8]c_int = @splat(1);
        return g.call(mlx.mlx_slice, .{ x, start.ptr, start.len, stop.ptr, stop.len, &ones8, start.len });
    }

    /// x[..., start:stop, ...] on one axis.
    pub fn sliceAxis(g: *Graph, x: Array, axis: usize, start: c_int, stop: c_int) !Array {
        const sh = shape(x);
        var lo: [8]c_int = @splat(0);
        var hi: [8]c_int = undefined;
        @memcpy(hi[0..sh.len], sh);
        lo[axis] = start;
        hi[axis] = stop;
        return g.slice(x, lo[0..sh.len], hi[0..sh.len]);
    }

    // ── scalars: in the operand's dtype, never a bare f32 that promotes bf16 ──

    /// A 0-d `v` in `dt` (cast from f32 only when `dt` is not f32).
    pub fn scalar(g: *Graph, v: f32, dt: Dtype) !Array {
        const raw = mlx.mlx_array_new_float(v);
        if (dt == .float32) return g.adopt(raw);
        defer _ = mlx.mlx_array_free(raw);
        return g.astype(raw, dt);
    }

    // ── composites ──

};

pub fn shape(x: Array) []const c_int {
    return mlx.getShape(x);
}

// ── tests ──

const testing = std.testing;
