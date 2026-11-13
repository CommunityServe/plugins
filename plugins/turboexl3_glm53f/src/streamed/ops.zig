//! Op backends for the DeepSeek-V4.1 trunk graphs (`deepseek_v41_graph.zig`).
//! A trunk function is written once against a backend `G`:
//! - `MlxOps` builds real mlx-c graphs on one stream (the inference thread, or
//!   a parity window); intermediates are freed by `reset`, state leaves the
//!   scope only through `keep`.
//! - `TraceOps` builds nothing: it records each op with its output shape and
//!   dtype under MLX's rules (promotion table, fp-mode qmm keeps the input
//!   dtype, fp dequantize yields bf16), so graph construction is checked on the
//!   host without creating an MLX array.
//! Python scalars are weak in MLX (`to_array(v, other.dtype)`); callers pass
//! the dtype explicitly through `scalar`, never an f32 default.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const xk = @import("../exl3/kernels.zig");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const first_cycle = @import("profile/decode_first.zig");
const prefill_timers = @import("profile/prefill_timers.zig");

/// G7: the package's profile probes, injected into the quant and the kernel registry through the backend type
/// (`sdk_ext.profile.of`); every probe compiles to nothing outside the profile builds.
const package_profile: sdk_ext.profile.Hook = .{ .prefill = prefill_timers, .launch = first_cycle };

pub const Dtype = mlx.mlx_dtype;
pub const max_dims = 8;

pub const Shape = struct {
    n: u8 = 0,
    d: [max_dims]c_int = @splat(0),

    pub fn of(dims: []const c_int) Shape {
        var s: Shape = .{ .n = @intCast(dims.len) };
        @memcpy(s.d[0..dims.len], dims);
        return s;
    }

    pub fn slice(self: *const Shape) []const c_int {
        return self.d[0..self.n];
    }

    pub fn dim(self: Shape, axis: c_int) c_int {
        return self.d[normAxis(axis, self.n)];
    }

};

pub fn normAxis(axis: c_int, n: u8) usize {
    return @intCast(if (axis < 0) axis + @as(c_int, n) else axis);
}

pub fn isFloat(d: Dtype) bool {
    return d == .float16 or d == .float32 or d == .float64 or d == .bfloat16;
}

pub fn dtypeSize(d: Dtype) usize {
    return switch (d) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        else => 8,
    };
}

/// MLX's `promote_types` (mlx/dtype.cpp), indexed by `mlx_dtype`.
pub fn promote(a: Dtype, b: Dtype) Dtype {
    const t = [14][14]u8{
        .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
        .{ 1, 1, 2, 3, 4, 6, 6, 7, 8, 9, 10, 11, 12, 13 },
        .{ 2, 2, 2, 3, 4, 7, 7, 7, 8, 9, 10, 11, 12, 13 },
        .{ 3, 3, 3, 3, 4, 8, 8, 8, 8, 9, 10, 11, 12, 13 },
        .{ 4, 4, 4, 4, 4, 10, 10, 10, 10, 9, 10, 11, 12, 13 },
        .{ 5, 6, 7, 8, 10, 5, 6, 7, 8, 9, 10, 11, 12, 13 },
        .{ 6, 6, 7, 8, 10, 6, 6, 7, 8, 9, 10, 11, 12, 13 },
        .{ 7, 7, 7, 8, 10, 7, 7, 7, 8, 9, 10, 11, 12, 13 },
        .{ 8, 8, 8, 8, 10, 8, 8, 8, 8, 9, 10, 11, 12, 13 },
        .{ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 11, 10, 13 },
        .{ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 13 },
        .{ 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13 },
        .{ 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 11, 12, 13 },
        .{ 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13 },
    };
    return @fromBackingInt(@intCast(t[@intCast(@backingInt(a))][@intCast(@backingInt(b))]));
}

/// A point in a backend's scope (`mark`). `resetTo` frees every tracked array
/// built after it; arrays held through `keep` are separate handles and survive.
/// A mark is spent by `reset` or by a `resetTo` to an earlier mark.
pub const Mark = struct { n: usize };

/// Frees `list[from..]` and truncates the list: the one release path of
/// `reset` (from 0) and `resetTo` (from a mark).
fn freeFrom(comptime E: type, list: *std.ArrayList(E), from: usize, comptime free: fn (E) void) void {
    std.debug.assert(from <= list.items.len);
    for (list.items[from..]) |a| free(a);
    list.shrinkRetainingCapacity(from);
}

/// The compiled regions a model builds at construction (`prepareTape`), one
/// closure per region and context (the trunk's config, the draft head's).
pub const Region = enum { attn_core, qkv_prep, out_prep, gate_prefix, moe_combine, hc_attn_prep, hc_ffn_prep, hc_post, seg2, seg3, draft_kv, markov_step, confidence, shared_mid };
const n_regions = std.meta.fieldNames(Region).len;
const contexts_per_region = 2;

// ── MLX backend ──

pub const MlxOps = struct {
    pub const T = mlx.mlx_array;
    pub const profile_hook = package_profile;

    gpa: std.mem.Allocator,
    s: mlx.mlx_stream,
    live: std.ArrayList(mlx.mlx_array) = .empty,
    /// The host's waits on the GPU so far: every evaluation that blocks (`evalAll`) and every read of an array's values
    /// (`hostIds`, `hostU32`, `hostF32`, `hostBool`, `hostArgmax`); `asyncEval` waits for nothing, nor do `readU32` and
    /// `readBool` (they read what an earlier eval produced).
    waits: u64 = 0,
    /// `nn.silu` / `nn.softplus` are `mx.compile(shapeless=True)` regions in
    /// the Python oracle; the same closures are compiled here (a compiled
    /// graph is not output-preserving against the op chain).
    silu_fn: mlx.mlx_closure = .{ .ctx = null },
    softplus_fn: mlx.mlx_closure = .{ .ctx = null },
    stream_box: *mlx.mlx_stream,
    /// Compiled trunk regions, one per (region type, construction context).
    tapes: std.ArrayList(TapeEntry) = .empty,
    /// Each region's closures by context, filled by `prepareTape`.
    regions: [n_regions][contexts_per_region]RegionSlot = @splat(@splat(.{})),
    /// A region's tracing context: borrows the owner's stream and closures.
    is_child: bool = false,
    /// The kernels lane's pinned registry, bound on this stream (`launch`); set
    /// once, before any kernel route is built.
    launcher: ?*const xk.Bound = null,

    const RegionSlot = struct { ctx: ?*const anyopaque = null, compiled: mlx.mlx_closure = .{ .ctx = null } };

    const TapeEntry = struct {
        key: usize,
        ctx: *const anyopaque,
        compiled: mlx.mlx_closure,
        payload: *anyopaque,
        free: *const fn (*anyopaque, std.mem.Allocator) void,
    };

    pub fn init(gpa: std.mem.Allocator, s: mlx.mlx_stream) !MlxOps {
        const box = try gpa.create(mlx.mlx_stream);
        errdefer gpa.destroy(box);
        box.* = s;
        var g: MlxOps = .{ .gpa = gpa, .s = s, .stream_box = box };
        g.silu_fn = try compileUnary(siluBody, box);
        errdefer _ = mlx.mlx_closure_free(g.silu_fn);
        g.softplus_fn = try compileUnary(softplusBody, box);
        return g;
    }

    pub fn deinit(g: *MlxOps) void {
        g.reset();
        g.live.deinit(g.gpa);
        if (g.is_child) return;
        for (g.tapes.items) |t| {
            _ = mlx.mlx_closure_free(t.compiled);
            t.free(t.payload, g.gpa);
        }
        g.tapes.deinit(g.gpa);
        _ = mlx.mlx_closure_free(g.silu_fn);
        _ = mlx.mlx_closure_free(g.softplus_fn);
        g.gpa.destroy(g.stream_box);
    }

    /// One pinned kernel launch (the kernels contract: its `Kernel` tag and
    /// `LaunchConfig`): `Bound.apply`, the declared outputs join this scope.
    pub fn launch(g: *MlxOps, k: xk.Kernel, inputs: []const T, cfg: *const xk.LaunchConfig, out: []T) !void {
        try g.launcher.?.apply(k, inputs, cfg, out);
        for (out[0..cfg.n_out]) |*o| o.* = try g.track(o.*);
    }

    /// A launch's mlx config built once (the kernels' decode routes, at construction).
    pub const Prepared = xk.Prepared;

    pub fn evalAll(g: *MlxOps, xs: []const T) !void {
        g.waits += 1;
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    fn freeArray(a: mlx.mlx_array) void {
        _ = mlx.mlx_array_free(a);
    }

    /// Free every intermediate built since the last reset.
    pub fn reset(g: *MlxOps) void {
        freeFrom(T, &g.live, 0, freeArray);
    }

    /// Return MLX's cached (freed) buffers to the system: the phase boundaries' release (the prompt
    /// fence, the phase change, construction), owned by the backend like every other allocation.
    pub fn clearCache(_: *MlxOps) void {
        _ = mlx.mlx_clear_cache();
    }

    /// The current scope point (a wave's start).
    pub fn mark(g: *const MlxOps) Mark {
        return .{ .n = g.live.items.len };
    }

    /// Free every array tracked since `m` (a wave's intermediates, after its
    /// eval); what the caller kept survives.
    pub fn resetTo(g: *MlxOps, m: Mark) void {
        freeFrom(T, &g.live, m.n, freeArray);
    }

    /// A reference that outlives `reset` (cache state, outputs); the caller
    /// frees it with `release`.
    pub fn keep(_: *MlxOps, x: T) T {
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out, x);
        return out;
    }

    pub fn release(_: *MlxOps, x: T) void {
        _ = mlx.mlx_array_free(x);
    }

    /// Take ownership of an array built outside the backend (freed by `reset`).
    pub fn adopt(g: *MlxOps, x: T) !T {
        return g.track(x);
    }

    /// The bytes of an array made by `hostArray` (its own unified-memory buffer), writable in place: DRAFT_AHEAD's
    /// persistent inputs, written only while no committed command reads them.
    pub fn hostBytes(_: *MlxOps, x: T) ![]u8 {
        const p = mlx.mlx_array_data_uint8(x) orelse return error.MlxNoData;
        return @constCast(p)[0 .. mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)];
    }

    /// A copy of host bytes as an array (`mx.array(numpy)`); freed by `reset`.
    pub fn hostArray(g: *MlxOps, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const a = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    fn track(g: *MlxOps, a: T) !T {
        g.live.append(g.gpa, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }

    pub fn dtypeOf(_: *MlxOps, x: T) Dtype {
        return mlx.mlx_array_dtype(x);
    }

    pub fn shapeOf(_: *MlxOps, x: T) Shape {
        return Shape.of(mlx.getShape(x));
    }

    fn op1(g: *MlxOps, comptime f: anytype, a: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, a, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn op2(g: *MlxOps, comptime f: anytype, a: T, b: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, a, b, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// A 0-d constant of `dt` holding `v` rounded as MLX rounds a Python
    /// scalar (f64 -> f32 -> dt).
    pub fn scalar(g: *MlxOps, v: f64, dt: Dtype) !T {
        const shape = [_]c_int{};
        const a = switch (dt) {
            .float32 => mlx.mlx_array_new_float(@floatCast(v)),
            .bool_ => mlx.mlx_array_new_bool(v != 0),
            .int32 => mlx.mlx_array_new_int(@intFromFloat(v)),
            .bfloat16 => blk: {
                const bits = bf16Bits(@floatCast(v));
                break :blk mlx.mlx_array_new_data(&bits, &shape, 0, .bfloat16);
            },
            .float16 => blk: {
                const h: f16 = @floatCast(@as(f32, @floatCast(v)));
                break :blk mlx.mlx_array_new_data(&h, &shape, 0, .float16);
            },
            else => return error.UnsupportedScalar,
        };
        if (a.ctx == null) return error.MlxError;
        return g.track(a);
    }

    pub fn ones(g: *MlxOps, shape: []const c_int, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_ones(&r, shape.ptr, shape.len, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn zeros(g: *MlxOps, shape: []const c_int, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_zeros(&r, shape.ptr, shape.len, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn astype(g: *MlxOps, x: T, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&r, x, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn add(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_add, a, b);
    }
    pub fn sub(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_subtract, a, b);
    }
    pub fn mul(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_multiply, a, b);
    }
    pub fn maximum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_maximum, a, b);
    }
    pub fn minimum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_minimum, a, b);
    }
    pub fn matmul(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_matmul, a, b);
    }

    pub fn rsqrt(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_rsqrt, x);
    }
    pub fn exp(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_exp, x);
    }
    /// `mx.log` (GLM-5.3's MTP lane: the typical test's entropy of a sampled row).
    pub fn log(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_log, x);
    }
    pub fn sigmoid(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_sigmoid, x);
    }
    pub fn cos(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_cos, x);
    }
    pub fn sin(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_sin, x);
    }
    pub fn abs(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_abs, x);
    }
    /// `.T`: every axis reversed.
    pub fn transpose(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_transpose, x);
    }

    pub fn clip(g: *MlxOps, x: T, lo: T, hi: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_clip(&r, x, lo, hi, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn reshape(g: *MlxOps, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn slice(g: *MlxOps, x: T, start: []const c_int, stop: []const c_int, strides: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice(&r, x, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn concat(g: *MlxOps, xs: []const T, axis: c_int) !T {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn take(g: *MlxOps, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_axis(&r, x, idx, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn reduce(g: *MlxOps, comptime f: anytype, x: T, axis: c_int, keepdims: bool) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, x, axis, keepdims, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }
    pub fn sum(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_sum_axis, x, axis, keepdims);
    }
    pub fn mean(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_mean_axis, x, axis, keepdims);
    }
    pub fn max(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_max_axis, x, axis, keepdims);
    }

    /// `mx.softmax` (precise = False).
    pub fn softmax(g: *MlxOps, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_softmax_axis(&r, x, axis, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `nn.silu`: the compiled `x * sigmoid(x)`.
    pub fn silu(g: *MlxOps, x: T) !T {
        return g.applyUnary(g.silu_fn, x);
    }

    /// `mx.async_eval(xs)`: the GPU starts on `xs`; nothing waits.
    pub fn asyncEval(_: *MlxOps, xs: []const T) !void {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }

    /// The routing barrier (`mx.eval(indices)` + `tolist`): evaluates the integer ids `x` and copies
    /// them, row-major, into `out` (x's size). Read in place (`readInPlace`).
    pub fn hostIds(g: *MlxOps, x: T, out: []u16) ![]const u16 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostIdsSize;
        // One producer dtype: the router's int32 indices (the arm's stand-in emits the same).
        std.debug.assert(mlx.mlx_array_dtype(x) == .int32);
        return readInPlace(i32, u16, x, mlx.mlx_array_data_int32(x), out);
    }

    /// Copies the evaluated `x`'s values, row-major, into `out`, reading its buffer through its own
    /// strides: an array an eval already produced is read with no GPU round trip (a `reshape(-1)`
    /// is a new array, and evaluating it is a fresh eval: a commit and a wait).
    fn readInPlace(comptime S: type, comptime D: type, x: T, data: ?[*]const S, out: []D) ![]const D {
        const nd = mlx.mlx_array_ndim(x);
        copyStrided(S, D, data orelse return error.MlxNoData, mlx.mlx_array_shape(x)[0..nd], mlx.mlx_array_strides(x)[0..nd], out);
        return out;
    }

    pub fn hostF32(g: *MlxOps, x: T, out: []f32) ![]const f32 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .float32);
        return readInPlace(f32, f32, x, mlx.mlx_array_data_float32(x), out);
    }

    fn compileUnary(comptime body: fn (mlx.mlx_array, mlx.mlx_stream) ?mlx.mlx_array, box: *mlx.mlx_stream) !mlx.mlx_closure {
        const Cb = struct {
            fn call(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                const s: *const mlx.mlx_stream = @ptrCast(@alignCast(payload.?));
                var x = mlx.mlx_array_new();
                if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
                defer _ = mlx.mlx_array_free(x);
                const y = body(x, s.*) orelse return -1;
                const out = [_]mlx.mlx_array{y};
                res.* = mlx.mlx_vector_array_new_data(&out, 1);
                _ = mlx.mlx_array_free(y);
                return 0;
            }
        };
        const raw = mlx.mlx_closure_new_func_payload(&Cb.call, @ptrCast(box), null);
        defer _ = mlx.mlx_closure_free(raw);
        var compiled = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&compiled, raw, true));
        return compiled;
    }

    fn siluBody(x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        if (mlx.mlx_sigmoid(&sig, x, s) != 0) return null;
        var y = mlx.mlx_array_new();
        if (mlx.mlx_multiply(&y, x, sig, s) != 0) {
            _ = mlx.mlx_array_free(y);
            return null;
        }
        return y;
    }

    fn softplusBody(x: mlx.mlx_array, s: mlx.mlx_stream) ?mlx.mlx_array {
        // `mx.logaddexp(x, 0)`: the Python int is weak, so 0 takes x's dtype.
        const zero = switch (mlx.mlx_array_dtype(x)) {
            .float32 => mlx.mlx_array_new_float(0),
            else => return null,
        };
        defer _ = mlx.mlx_array_free(zero);
        var y = mlx.mlx_array_new();
        if (mlx.mlx_logaddexp(&y, x, zero, s) != 0) {
            _ = mlx.mlx_array_free(y);
            return null;
        }
        return y;
    }
};

/// The SDK's strided host copy (sdk_ext.ops), the name this package's code reads.
pub const copyStrided = sdk_ext.ops.copyStrided;

/// f32 -> bf16 bits, round to nearest even (what MLX's host conversion does).
pub fn bf16Bits(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return @intCast((u >> 16) | 0x40);
    const rounding: u32 = 0x7FFF + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}

// ── host trace backend ──

pub const Op = enum {
    input,
    host,
    scalar,
    arange,
    ones,
    zeros,
    full,
    astype,
    add,
    sub,
    mul,
    div,
    floor_div,
    maximum,
    minimum,
    power,
    logaddexp,
    less,
    less_equal,
    greater,
    equal,
    logical_and,
    logical_or,
    matmul,
    neg,
    square,
    sqrt,
    rsqrt,
    exp,
    sigmoid,
    cos,
    sin,
    abs,
    transpose,
    where,
    clip,
    einsum,
    qmm,
    dequantize,
    reshape,
    transpose_axes,
    broadcast_to,
    expand_dims,
    slice,
    slice_update,
    concat,
    stack,
    take,
    take_along_axis,
    sum,
    mean,
    max,
    softmax,
    sort,
    argsort,
    argpartition,
    cumsum,
    repeat,
    silu,
    softplus,
    greater_equal,
    quantize,
    tape_begin,
    tape_end,
    hadamard,
    async_eval,
    host_read,
    kernel,
    gather_qmm,
    argmax,
    logsumexp,
    event_wait,
};

pub const TraceOps = struct {
    pub const T = u32;
    pub const profile_hook = package_profile;
    pub const Node = struct { op: Op, dtype: Dtype, shape: Shape };
    /// What a host read returns on the trace backend (a test's script): the
    /// routed ids of each routing barrier and each greedy pick.
    pub const HostValues = struct {
        ctx: *anyopaque,
        ids: *const fn (ctx: *anyopaque, out: []u16) anyerror!void,
        argmax: *const fn (ctx: *anyopaque) anyerror!u32,
        u32s: ?*const fn (ctx: *anyopaque, out: []u32) anyerror!void = null,
        f32s: ?*const fn (ctx: *anyopaque, out: []f32) anyerror!void = null,
        bools: ?*const fn (ctx: *anyopaque, out: []bool) anyerror!void = null,
    };

    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    host_values: ?HostValues = null,
    /// Each event wait's timeline value and dependency count, in build order.
    waits: std.ArrayList(Wait) = .empty,
    /// The event waits the scheduler built (one per `eventWait` call, however many arrays it aliases into `waits`).
    wait_calls: usize = 0,
    freed: std.ArrayList(Freed) = .empty,
    /// The node count at each `evalAll` (where a host sync fell in the build).
    evals: std.ArrayList(usize) = .empty,
    /// Every array handed to `evalAll`, in call order (which arrays a fence settled).
    evaluated: std.ArrayList(T) = .empty,
    /// Every array handed to `asyncEval`, in call order, and each commit's [start, end) span in it (which arrays each
    /// commit started, in order).
    committed: std.ArrayList(T) = .empty,
    commit_spans: std.ArrayList([2]usize) = .empty,
    /// Every kept array handed to `release`, in call order (when a pass lets a kept array go).
    released: std.ArrayList(T) = .empty,
    /// Launches of prepared configs, and the prepared configs not yet released.
    prepared_launches: usize = 0,
    prepared_live: usize = 0,
    /// The kernel of each prepared launch, in order (which text a route reached).
    launched: std.ArrayList(xk.Kernel) = .empty,
    /// The regions `prepareTape` compiled, by context.
    regions: [n_regions][contexts_per_region]?*const anyopaque = @splat(@splat(null)),
    /// When set, `hostArray` keeps a copy of its bytes (`hostBytesOf`), so a
    /// test can compare what a graph was fed; so does a row `concat` of such arrays.
    record_host: bool = false,
    host_data: std.AutoHashMapUnmanaged(u32, []u8) = .empty,
    /// Region traces: a region called with inputs of a signature (shapes and
    /// dtypes) it has not seen is what `mx.compile` traces anew.
    compiles: usize = 0,
    tape_sigs: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// Arrays dropped before their scope ended (`drop`), and the reads of them after (a use after release).
    dropped: std.ArrayList(T) = .empty,
    use_after_drop: u32 = 0,
    /// `track_live` (tests): the kept handles' live counts (keep +1, release -1) and each slice's base array, so a
    /// test can sum the distinct buffers the kept handles pin (a view pins its whole base).
    track_live: bool = false,
    live_kept: std.AutoHashMapUnmanaged(T, i32) = .empty,
    slice_base: std.AutoHashMapUnmanaged(T, T) = .empty,
    /// `track_live`: each `sliceUpdateDyn`'s source and update bytes, and at every `evalAll` each update built since
    /// the previous one judged as MLX would run it: in place (the source's buffer donated: the update's bytes) when no
    /// kept handle pins the source (itself or a view of it), else a copy of the whole source.
    update_src: std.AutoHashMapUnmanaged(T, struct { src: T, bytes: u64 }) = .empty,
    lane_writes: std.ArrayList(LaneWrite) = .empty,
    last_eval_node: usize = 0,
    pub const LaneWrite = struct { node: T, src: T, eval: usize, full_bytes: u64, update_bytes: u64, pinned_by: ?T };
    pub const Wait = struct { value: u64, n_deps: u32 };

    pub fn init(gpa: std.mem.Allocator) TraceOps {
        return .{ .gpa = gpa };
    }

    pub fn deinit(g: *TraceOps) void {
        g.nodes.deinit(g.gpa);
        g.waits.deinit(g.gpa);
        g.freed.deinit(g.gpa);
        g.evals.deinit(g.gpa);
        g.evaluated.deinit(g.gpa);
        g.committed.deinit(g.gpa);
        g.commit_spans.deinit(g.gpa);
        g.released.deinit(g.gpa);
        g.dropped.deinit(g.gpa);
        var it = g.host_data.valueIterator();
        while (it.next()) |v| g.gpa.free(v.*);
        g.host_data.deinit(g.gpa);
        g.tape_sigs.deinit(g.gpa);
        g.launched.deinit(g.gpa);
        g.live_kept.deinit(g.gpa);
        g.slice_base.deinit(g.gpa);
        g.update_src.deinit(g.gpa);
        g.lane_writes.deinit(g.gpa);
    }

    pub fn reset(_: *TraceOps) void {}

    /// Node ranges a `resetTo` released, `[from, to)`, in call order (the trace
    /// keeps its nodes; tests read the ranges).
    pub const Freed = struct { from: u32, to: u32 };

    pub fn mark(g: *const TraceOps) Mark {
        return .{ .n = g.nodes.items.len };
    }

    pub fn resetTo(g: *TraceOps, m: Mark) void {
        std.debug.assert(m.n <= g.nodes.items.len);
        g.freed.append(g.gpa, .{ .from = @intCast(m.n), .to = @intCast(g.nodes.items.len) }) catch @panic("trace: out of memory");
    }

    pub fn keep(g: *TraceOps, x: T) T {
        if (g.track_live) {
            const e = g.live_kept.getOrPut(g.gpa, x) catch @panic("trace: out of memory");
            if (!e.found_existing) e.value_ptr.* = 0;
            e.value_ptr.* += 1;
        }
        return x;
    }
    pub fn release(g: *TraceOps, x: T) void {
        g.released.append(g.gpa, x) catch @panic("trace: out of memory");
        if (g.track_live) if (g.live_kept.getPtr(x)) |c| {
            c.* -= 1;
        };
    }

    /// A leaf the test hands the graph (weights, cache state, inputs).
    pub fn input(g: *TraceOps, shape: []const c_int, dtype: Dtype) !T {
        return g.push(.input, dtype, Shape.of(shape));
    }

    /// Host bytes handed to the graph (a leaf, like `input`).
    pub fn hostArray(g: *TraceOps, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const s = Shape.of(shape);
        if (@as(i64, @intCast(bytes.len)) != s.numel() * @as(i64, @intCast(dtypeSize(dt)))) return error.HostBytes;
        const x = try g.push(.host, dt, s);
        if (g.record_host) {
            const copy = try g.gpa.dupe(u8, bytes);
            errdefer g.gpa.free(copy);
            try g.host_data.put(g.gpa, x, copy);
        }
        return x;
    }

    pub fn dtypeOf(g: *TraceOps, x: T) Dtype {
        g.touch(x);
        return g.nodes.items[x].dtype;
    }

    pub fn shapeOf(g: *TraceOps, x: T) Shape {
        g.touch(x);
        return g.nodes.items[x].shape;
    }

    fn broadcast(a: Shape, b: Shape) !Shape {
        const n = @max(a.n, b.n);
        var out: Shape = .{ .n = n };
        for (0..n) |i| {
            const da: c_int = if (i + a.n >= n) a.d[i + a.n - n] else 1;
            const db: c_int = if (i + b.n >= n) b.d[i + b.n - n] else 1;
            if (da != db and da != 1 and db != 1) return error.BroadcastMismatch;
            out.d[i] = if (da == 1) db else da;
        }
        return out;
    }

    pub fn scalar(g: *TraceOps, _: f64, dt: Dtype) !T {
        return g.push(.scalar, dt, .{});
    }

    /// MLX returns the input itself for a same-dtype astype: no node.
    pub fn astype(g: *TraceOps, x: T, dt: Dtype) !T {
        if (g.dtypeOf(x) == dt) return x;
        return g.push(.astype, dt, g.shapeOf(x));
    }

    pub fn add(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.add, a, b);
    }
    pub fn sub(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.sub, a, b);
    }
    pub fn mul(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.mul, a, b);
    }
    pub fn maximum(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.maximum, a, b);
    }
    pub fn minimum(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.minimum, a, b);
    }

    pub fn matmul(g: *TraceOps, a: T, b: T) !T {
        const sa = g.shapeOf(a);
        const sb = g.shapeOf(b);
        if (sa.n < 2 or sb.n < 2 or sa.dim(-1) != sb.dim(-2)) return error.MatmulShape;
        var ba = sa;
        ba.n -= 2;
        var bb = sb;
        bb.n -= 2;
        var out = try broadcast(ba, bb);
        out.d[out.n] = sa.dim(-2);
        out.d[out.n + 1] = sb.dim(-1);
        out.n += 2;
        return g.push(.matmul, promote(g.dtypeOf(a), g.dtypeOf(b)), out);
    }

    pub fn rsqrt(g: *TraceOps, x: T) !T {
        return g.unary(.rsqrt, x);
    }
    pub fn exp(g: *TraceOps, x: T) !T {
        return g.unary(.exp, x);
    }
    pub fn sigmoid(g: *TraceOps, x: T) !T {
        return g.unary(.sigmoid, x);
    }
    pub fn cos(g: *TraceOps, x: T) !T {
        return g.unary(.cos, x);
    }
    pub fn sin(g: *TraceOps, x: T) !T {
        return g.unary(.sin, x);
    }
    pub fn abs(g: *TraceOps, x: T) !T {
        return g.unary(.abs, x);
    }
    pub fn silu(g: *TraceOps, x: T) !T {
        return g.unary(.silu, x);
    }
    pub fn softmax(g: *TraceOps, x: T, _: c_int) !T {
        return g.unary(.softmax, x);
    }

    pub fn transpose(g: *TraceOps, x: T) !T {
        const s = g.shapeOf(x);
        var out: Shape = .{ .n = s.n };
        for (0..s.n) |i| out.d[i] = s.d[s.n - 1 - i];
        return g.push(.transpose, g.dtypeOf(x), out);
    }

    /// A custom kernel's output (the shape and dtype its launch declares).
    pub fn kernel(g: *TraceOps, shape: []const c_int, dt: Dtype) !T {
        return g.push(.kernel, dt, Shape.of(shape));
    }

    /// A pinned kernel launch: one kernel node per output the launch declares.
    pub fn launch(g: *TraceOps, _: xk.Kernel, _: []const T, cfg: *const xk.LaunchConfig, out: []T) !void {
        for (out[0..cfg.n_out], 0..) |*o, i| o.* = try g.kernel(cfg.out_shapes[i][0..cfg.out_ranks[i]], cfg.out_dtypes[i]);
    }

    /// A prepared launch on the trace backend: the kernel and its config, kept by value.
    pub const Prepared = struct { k: xk.Kernel, cfg: xk.LaunchConfig };

    pub fn clip(g: *TraceOps, x: T, lo: T, hi: T) !T {
        return g.push(.clip, promote(promote(g.dtypeOf(x), g.dtypeOf(lo)), g.dtypeOf(hi)), g.shapeOf(x));
    }

    pub fn reshape(g: *TraceOps, x: T, shape: []const c_int) !T {
        var out = Shape.of(shape);
        const total = g.shapeOf(x).numel();
        var known: i64 = 1;
        var infer: ?usize = null;
        for (shape, 0..) |d, i| {
            if (d == -1) infer = i else known *= d;
        }
        if (infer) |i| out.d[i] = @intCast(@divExact(total, known)) else if (known != total) return error.ReshapeSize;
        return g.push(.reshape, g.dtypeOf(x), out);
    }

    /// Positive-stride slices only (the trunk never reverses with `slice`).
    pub fn slice(g: *TraceOps, x: T, start: []const c_int, stop: []const c_int, strides: []const c_int) !T {
        const s = g.shapeOf(x);
        if (start.len != s.n or stop.len != s.n or strides.len != s.n) return error.SliceRank;
        var out: Shape = .{ .n = s.n };
        for (0..s.n) |i| {
            if (strides[i] <= 0 or start[i] < 0 or stop[i] > s.d[i] or start[i] > stop[i]) return error.SliceBounds;
            out.d[i] = @divFloor(stop[i] - start[i] + strides[i] - 1, strides[i]);
        }
        const y = try g.push(.slice, g.dtypeOf(x), out);
        if (g.track_live) try g.slice_base.put(g.gpa, y, x);
        return y;
    }

    pub fn concat(g: *TraceOps, xs: []const T, axis: c_int) !T {
        var out = g.shapeOf(xs[0]);
        var dt = g.dtypeOf(xs[0]);
        const a = normAxis(axis, out.n);
        for (xs[1..]) |x| {
            const s = g.shapeOf(x);
            if (s.n != out.n) return error.ConcatShape;
            for (0..s.n) |i| if (i != a and s.d[i] != out.d[i]) return error.ConcatShape;
            out.d[a] += s.d[a];
            dt = promote(dt, g.dtypeOf(x));
        }
        const y = try g.push(.concat, dt, out);
        if (g.record_host and a == 0) try g.recordRows(y, xs);
        return y;
    }

    pub fn take(g: *TraceOps, x: T, idx: T, axis: c_int) !T {
        const s = g.shapeOf(x);
        const si = g.shapeOf(idx);
        const a = normAxis(axis, s.n);
        var out: Shape = .{};
        for (s.d[0..a]) |d| {
            out.d[out.n] = d;
            out.n += 1;
        }
        for (si.d[0..si.n]) |d| {
            out.d[out.n] = d;
            out.n += 1;
        }
        for (s.d[a + 1 .. s.n]) |d| {
            out.d[out.n] = d;
            out.n += 1;
        }
        return g.push(.take, g.dtypeOf(x), out);
    }

    pub fn sum(g: *TraceOps, x: T, axis: c_int, keepdims: bool) !T {
        const d = g.dtypeOf(x);
        return g.reduce(.sum, x, axis, keepdims, if (d == .bool_) .int32 else d);
    }
    pub fn mean(g: *TraceOps, x: T, axis: c_int, keepdims: bool) !T {
        const d = g.dtypeOf(x);
        return g.reduce(.mean, x, axis, keepdims, if (isFloat(d)) d else .float32);
    }
    pub fn max(g: *TraceOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(.max, x, axis, keepdims, g.dtypeOf(x));
    }

};

const testing = std.testing;

comptime {
    // The gather quant's adapter binds this backend (its `gatherMatmul` and the activation's ops).
    sdk_ext.quant.checkAccepted(sdk_ext.quant.FromGatherMatmul(sdk_ext.quant.GatherQmm), MlxOps);
}
