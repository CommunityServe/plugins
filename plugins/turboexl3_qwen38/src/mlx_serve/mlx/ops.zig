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

pub fn liveForTest() i64 {
    return live_for_test.load(.monotonic);
}

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

    /// Takes `x` out of the list (newest first); false when this Graph does not own it.
    fn remove(o: *Owned, x: Array) bool {
        var i = o.heap.items.len;
        while (i > 0) {
            i -= 1;
            if (o.heap.items[i].ctx == x.ctx) {
                _ = o.heap.swapRemove(i);
                return true;
            }
        }
        i = o.n;
        while (i > 0) {
            i -= 1;
            if (o.buf[i].ctx != x.ctx) continue;
            o.n -= 1;
            o.buf[i] = o.buf[o.n];
            if (o.heap.pop()) |last| {
                o.buf[o.n] = last;
                o.n += 1;
            }
            return true;
        }
        return false;
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

    /// A child on the same stream: a helper's own lifetime.
    pub fn scope(g: *const Graph) Graph {
        return .init(g.a, g.s);
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

    /// A reference on `x` this Graph owns: how a child's result outlives the child.
    pub fn keep(g: *Graph, x: Array) !Array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, x);
        return g.adopt(r);
    }

    /// A reference on `x` that leaves the Graph: the caller frees it.
    pub fn escape(_: *const Graph, x: Array) Array {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, x);
        return r;
    }

    /// Frees `x` now (before an eval that must not pin it); `x` must be this Graph's.
    pub fn drop(g: *Graph, x: Array) void {
        const found = g.owned.remove(x);
        std.debug.assert(found);
        if (found) release(x);
    }

    // ── one line per op ──
    // zig fmt: off
    pub fn add(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_add, .{ x, y }); }
    pub fn sub(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_subtract, .{ x, y }); }
    pub fn mul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_multiply, .{ x, y }); }
    pub fn div(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_divide, .{ x, y }); }
    pub fn matmul(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_matmul, .{ x, y }); }
    pub fn maximum(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_maximum, .{ x, y }); }
    pub fn minimum(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_minimum, .{ x, y }); }
    pub fn less(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_less, .{ x, y }); }
    pub fn greater(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_greater, .{ x, y }); }
    pub fn equal(g: *Graph, x: Array, y: Array) !Array { return g.call(mlx.mlx_equal, .{ x, y }); }
    pub fn where(g: *Graph, c: Array, x: Array, y: Array) !Array { return g.call(mlx.mlx_where, .{ c, x, y }); }
    pub fn exp(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_exp, .{x}); }
    pub fn log(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_log, .{x}); }
    pub fn sin(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sin, .{x}); }
    pub fn cos(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_cos, .{x}); }
    pub fn tanh(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_tanh, .{x}); }
    pub fn sigmoid(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sigmoid, .{x}); }
    pub fn sqrt(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_sqrt, .{x}); }
    pub fn rsqrt(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_rsqrt, .{x}); }
    pub fn square(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_square, .{x}); }
    pub fn abs(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_abs, .{x}); }
    pub fn neg(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_negative, .{x}); }
    pub fn erf(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_erf, .{x}); }
    pub fn astype(g: *Graph, x: Array, dt: Dtype) !Array { return g.call(mlx.mlx_astype, .{ x, dt }); }
    pub fn contiguous(g: *Graph, x: Array) !Array { return g.call(mlx.mlx_contiguous, .{ x, false }); }
    pub fn reshape(g: *Graph, x: Array, sh: []const c_int) !Array { return g.call(mlx.mlx_reshape, .{ x, sh.ptr, sh.len }); }
    pub fn transpose(g: *Graph, x: Array, axes: []const c_int) !Array { return g.call(mlx.mlx_transpose_axes, .{ x, axes.ptr, axes.len }); }
    pub fn broadcast(g: *Graph, x: Array, sh: []const c_int) !Array { return g.call(mlx.mlx_broadcast_to, .{ x, sh.ptr, sh.len }); }
    pub fn expandDims(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_expand_dims, .{ x, axis }); }
    pub fn take(g: *Graph, x: Array, ids: Array, axis: c_int) !Array { return g.call(mlx.mlx_take_axis, .{ x, ids, axis }); }
    pub fn takeAlong(g: *Graph, x: Array, ids: Array, axis: c_int) !Array { return g.call(mlx.mlx_take_along_axis, .{ x, ids, axis }); }
    pub fn repeat(g: *Graph, x: Array, n: c_int, axis: c_int) !Array { return g.call(mlx.mlx_repeat_axis, .{ x, n, axis }); }
    pub fn sum(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_sum_axis, .{ x, axis, keepdims }); }
    pub fn mean(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_mean_axis, .{ x, axis, keepdims }); }
    pub fn max(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_max_axis, .{ x, axis, keepdims }); }
    pub fn argmax(g: *Graph, x: Array, axis: c_int, keepdims: bool) !Array { return g.call(mlx.mlx_argmax_axis, .{ x, axis, keepdims }); }
    pub fn argsort(g: *Graph, x: Array, axis: c_int) !Array { return g.call(mlx.mlx_argsort_axis, .{ x, axis }); }
    pub fn argpartition(g: *Graph, x: Array, kth: c_int, axis: c_int) !Array { return g.call(mlx.mlx_argpartition_axis, .{ x, kth, axis }); }
    pub fn softmax(g: *Graph, x: Array, axis: c_int, precise: bool) !Array { return g.call(mlx.mlx_softmax_axis, .{ x, axis, precise }); }
    pub fn zeros(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_zeros, .{ sh.ptr, sh.len, dt }); }
    pub fn ones(g: *Graph, sh: []const c_int, dt: Dtype) !Array { return g.call(mlx.mlx_ones, .{ sh.ptr, sh.len, dt }); }
    pub fn full(g: *Graph, sh: []const c_int, v: Array, dt: Dtype) !Array { return g.call(mlx.mlx_full, .{ sh.ptr, sh.len, v, dt }); }
    pub fn arange(g: *Graph, start: f64, stop: f64, step: f64, dt: Dtype) !Array { return g.call(mlx.mlx_arange, .{ start, stop, step, dt }); }
    pub fn rmsNorm(g: *Graph, x: Array, w: Array, eps: f32) !Array { return g.call(mlx.mlx_fast_rms_norm, .{ x, w, eps }); }
    pub fn layerNorm(g: *Graph, x: Array, w: Array, b: Array, eps: f32) !Array { return g.call(mlx.mlx_fast_layer_norm, .{ x, w, b, eps }); }
    /// Rotate-half RoPE over the last `dims` at positions offset.. (scale 1, no freqs table).
    pub fn rope(g: *Graph, x: Array, dims: c_int, base: f32, offset: c_int) !Array { return g.call(mlx.mlx_fast_rope, .{ x, dims, false, mlx.mlx_optional_float.some(base), 1.0, offset, Array{} }); }
    pub fn sdpa(g: *Graph, q: Array, k: Array, v: Array, scale: f32, mask: Mask) !Array {
        const mode: [*:0]const u8 = switch (mask) { .none => "", .causal => "causal", .array => "array" };
        return g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, if (mask == .array) mask.array else Array{}, Array{}, false });
    }
    /// x @ dequant(w)^T for an affine-quantized `w`.
    pub fn qmm(g: *Graph, x: Array, w: Array, scales: Array, biases: Array, group_size: c_int, bits: c_int) !Array { return g.call(mlx.mlx_quantized_matmul, .{ x, w, scales, biases, true, mlx.mlx_optional_int.some(group_size), mlx.mlx_optional_int.some(bits), "affine" }); }
    /// Channels-last; weight [out, k, in].
    pub fn conv1d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv1d, .{ x, w, stride, padding, dilation, groups }); }
    pub fn convTranspose1d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, output_padding: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv_transpose1d, .{ x, w, stride, padding, dilation, output_padding, groups }); }
    /// Channels-last; weight [out, kh, kw, in]; square stride, padding and dilation.
    pub fn conv2d(g: *Graph, x: Array, w: Array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int) !Array { return g.call(mlx.mlx_conv2d, .{ x, w, stride, stride, padding, padding, dilation, dilation, groups }); }

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

    /// `dst` with `update` written over [start, stop), unit strides.
    pub fn sliceUpdate(g: *Graph, dst: Array, update: Array, start: []const c_int, stop: []const c_int) !Array {
        const ones8: [8]c_int = @splat(1);
        return g.call(mlx.mlx_slice_update, .{ dst, update, start.ptr, start.len, stop.ptr, stop.len, &ones8, start.len });
    }

    pub fn concat(g: *Graph, xs: []const Array, axis: c_int) !Array {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        return g.call(mlx.mlx_concatenate_axis, .{ v, axis });
    }

    /// `lo`/`hi` zero rows added on one axis (the pad value is cast to x's dtype by MLX).
    pub fn padAxis(g: *Graph, x: Array, axis: c_int, lo: c_int, hi: c_int) !Array {
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        return g.call(mlx.mlx_pad, .{ x, &[_]c_int{axis}, 1, &[_]c_int{lo}, 1, &[_]c_int{hi}, 1, zero, "constant" });
    }

    /// Two keys from one.
    pub fn splitKey(g: *Graph, key: Array) ![2]Array {
        var k1 = counted(mlx.mlx_array_new());
        var k2 = counted(mlx.mlx_array_new());
        const rc = mlx.mlx_random_split(&k1, &k2, key, g.s);
        g.owned.add(g.a, k1) catch |e| {
            release(k1);
            release(k2);
            return e;
        };
        _ = try g.adoptCounted(k2);
        try mlx.check(rc);
        return .{ k1, k2 };
    }

    // ── scalars: in the operand's dtype, never a bare f32 that promotes bf16 ──

    /// A 0-d `v` in `dt` (cast from f32 only when `dt` is not f32).
    pub fn scalar(g: *Graph, v: f32, dt: Dtype) !Array {
        const raw = mlx.mlx_array_new_float(v);
        if (dt == .float32) return g.adopt(raw);
        defer _ = mlx.mlx_array_free(raw);
        return g.astype(raw, dt);
    }
    pub fn scalarLike(g: *Graph, x: Array, v: f32) !Array {
        return g.scalar(v, dtype(x));
    }
    pub fn addScalar(g: *Graph, x: Array, v: f32) !Array {
        return g.add(x, try g.scalarLike(x, v));
    }
    pub fn mulScalar(g: *Graph, x: Array, v: f32) !Array {
        return g.mul(x, try g.scalarLike(x, v));
    }

    /// A host slice copied into an array of the matching dtype.
    pub fn fromSlice(g: *Graph, comptime T: type, data: []const T, sh: []const c_int) !Array {
        const dt: Dtype = switch (T) {
            f32 => .float32,
            i32 => .int32,
            u32 => .uint32,
            bool => .bool_,
            else => @compileError("fromSlice: no dtype for " ++ @typeName(T)),
        };
        return g.adopt(mlx.mlx_array_new_data(data.ptr, sh.ptr, @intCast(sh.len), dt));
    }

    // ── composites ──

    pub fn silu(g: *Graph, x: Array) !Array {
        var c = g.scope();
        defer c.deinit();
        return g.keep(try c.mul(x, try c.sigmoid(x)));
    }

    /// x @ w^T (+ b): PyTorch Linear over the last axis.
    pub fn linear(g: *Graph, x: Array, w: Array, b: ?Array) !Array {
        var c = g.scope();
        defer c.deinit();
        const y = try c.matmul(x, try c.transpose(w, &.{ 1, 0 }));
        return g.keep(if (b) |bb| try c.add(y, bb) else y);
    }

    /// [B,L,H*D] → [B,H,L,D]
    pub fn splitHeads(g: *Graph, x: Array, h: c_int, d: c_int) !Array {
        var c = g.scope();
        defer c.deinit();
        const sh = shape(x);
        return g.keep(try c.transpose(try c.reshape(x, &.{ sh[0], sh[1], h, d }), &.{ 0, 2, 1, 3 }));
    }

    /// [B,H,L,D] → [B,L,H*D]
    pub fn mergeHeads(g: *Graph, x: Array) !Array {
        var c = g.scope();
        defer c.deinit();
        const sh = shape(x);
        return g.keep(try c.reshape(try c.transpose(x, &.{ 0, 2, 1, 3 }), &.{ sh[0], sh[2], sh[1] * sh[3] }));
    }
};

pub fn shape(x: Array) []const c_int {
    return mlx.getShape(x);
}

pub fn dtype(x: Array) Dtype {
    return mlx.mlx_array_dtype(x);
}

pub fn eval(x: Array) !void {
    try mlx.check(mlx.mlx_array_eval(x));
}

/// One eval over several outputs (they share the work of their common inputs).
pub fn evalMany(xs: []const Array) !void {
    const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
}

// ── tests ──

const testing = std.testing;

fn cpu() Stream {
    return mlx.mlx_default_cpu_stream_new();
}

/// The bytes of `x`, row-major (evaluates it).
fn bytesOf(a: std.mem.Allocator, x: Array) ![]u8 {
    var g = Graph.init(a, cpu());
    defer g.deinit();
    const c = try g.contiguous(x);
    try eval(c);
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return a.dupe(u8, p[0..n]);
}

/// `got` and `want` agree in dtype, shape and every bit, and a float `want` is finite.
pub fn expectBitEqual(got: Array, want: Array) !void {
    try testing.expectEqual(dtype(want), dtype(got));
    try testing.expectEqualSlices(c_int, shape(want), shape(got));
    const a = try bytesOf(testing.allocator, got);
    defer testing.allocator.free(a);
    const b = try bytesOf(testing.allocator, want);
    defer testing.allocator.free(b);
    try testing.expectEqualSlices(u8, b, a);
    switch (dtype(want)) {
        .float16, .bfloat16, .float32 => {
            var g = Graph.init(testing.allocator, cpu());
            defer g.deinit();
            const f = try g.contiguous(try g.astype(want, .float32));
            try eval(f);
            const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
            for (d[0..mlx.mlx_array_size(f)]) |v| try testing.expect(std.math.isFinite(v));
        },
        else => {},
    }
}

/// The wrapper's output equals the raw mlx-c call `f(&out, args..., s)`.
fn expectSameAs(got: Array, comptime f: anytype, args: anytype, s: Stream) !void {
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(@call(.auto, f, .{&want} ++ args ++ .{s}));
    try expectBitEqual(got, want);
}

fn randn(g: *Graph, sh: []const c_int, seed: u64) !Array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    return g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, Dtype.float32, @as(f32, 0), @as(f32, 1), key });
}

fn chain(g: *Graph) !Array {
    const x = try randn(g, &.{ 2, 8 }, 40);
    const w = try randn(g, &.{ 4, 8 }, 41);
    const h = try g.splitHeads(try g.reshape(try g.linear(x, w, null), &.{ 1, 2, 4 }), 2, 2);
    return g.silu(try g.mergeHeads(h));
}
