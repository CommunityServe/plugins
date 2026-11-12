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
/// Inputs / outputs of one compiled trunk region.
pub const max_tape_io = 24;

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

    pub fn numel(self: Shape) i64 {
        var n: i64 = 1;
        for (self.d[0..self.n]) |v| n *= v;
        return n;
    }

    pub fn eql(a: Shape, b: Shape) bool {
        return a.n == b.n and std.mem.eql(c_int, a.d[0..a.n], b.d[0..b.n]);
    }
};

pub fn normAxis(axis: c_int, n: u8) usize {
    return @intCast(if (axis < 0) axis + @as(c_int, n) else axis);
}

pub fn isFloat(d: Dtype) bool {
    return d == .float16 or d == .float32 or d == .float64 or d == .bfloat16;
}

pub fn isInteger(d: Dtype) bool {
    return switch (d) {
        .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64 => true,
        else => false,
    };
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

pub fn quantBits(mode: sdk.QuantMode) u32 {
    return switch (mode) {
        .mxfp8 => 8,
        .mxfp4, .nvfp4 => 4,
        .affine => 8,
        // ggml blocks are the gguf engine's; a bank's quantization parses through sdk_ext.quant, which refuses them.
        .gguf => unreachable,
    };
}

pub fn quantGroup(mode: sdk.QuantMode) u32 {
    return switch (mode) {
        .nvfp4 => 16,
        else => 32,
    };
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

    fn child(g: *const MlxOps) MlxOps {
        return .{ .gpa = g.gpa, .s = g.s, .silu_fn = g.silu_fn, .softplus_fn = g.softplus_fn, .stream_box = g.stream_box, .is_child = true, .launcher = g.launcher };
    }

    /// One pinned kernel launch (the kernels contract: its `Kernel` tag and
    /// `LaunchConfig`): `Bound.apply`, the declared outputs join this scope.
    pub fn launch(g: *MlxOps, k: xk.Kernel, inputs: []const T, cfg: *const xk.LaunchConfig, out: []T) !void {
        try g.launcher.?.apply(k, inputs, cfg, out);
        for (out[0..cfg.n_out]) |*o| o.* = try g.track(o.*);
    }

    /// A launch's mlx config built once (the kernels' decode routes, at construction).
    pub const Prepared = xk.Prepared;

    pub fn prepareLaunch(g: *MlxOps, k: xk.Kernel, cfg: *const xk.LaunchConfig) !Prepared {
        return g.launcher.?.prepare(k, cfg);
    }

    /// One launch of a prepared config; the outputs join this scope.
    pub fn launchPrepared(g: *MlxOps, p: *const Prepared, inputs: []const T, out: []T) !void {
        try g.launcher.?.applyPrepared(p, inputs, out);
        for (out[0..p.n_out]) |*o| o.* = try g.track(o.*);
    }

    pub fn releasePrepared(_: *MlxOps, p: *Prepared) void {
        p.deinit();
    }

    /// Compiles region `Body` for context `ctx` (at construction; idempotent).
    /// `ctx` carries the region's structural constants and must outlive the backend.
    pub fn prepareTape(g: *MlxOps, comptime Body: type, ctx: *const Body.Ctx) !void {
        const slots = &g.regions[@backingInt(Body.region)];
        for (slots) |sl| if (sl.ctx == @as(*const anyopaque, ctx)) return;
        const free = for (slots) |*sl| {
            if (sl.ctx == null) break sl;
        } else return error.RegionContextsFull;
        free.* = .{ .ctx = ctx, .compiled = try g.buildTape(Body, ctx, @intFromPtr(@typeName(Body).ptr)) };
    }

    /// `mx.compile(fn)` (fixed shape) of one trunk region `Body.run`, prepared
    /// at construction: traced once per input signature, replayed after.
    pub fn tape(g: *MlxOps, comptime Body: type, ctx: *const Body.Ctx, inputs: []const T, out: []T) !void {
        // Construction prepared every region a call can reach (the trace twin refuses the rest by name,
        // which the host tests prove); here the lookup only picks the context's closure (<= 2 slots).
        const compiled = for (g.regions[@backingInt(Body.region)]) |sl| {
            if (sl.ctx == @as(*const anyopaque, ctx)) break sl.compiled;
        } else unreachable;
        if (comptime first_cycle.enabled) first_cycle.region(@tagName(Body.region), @intFromPtr(ctx), inputs);
        const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        defer _ = mlx.mlx_vector_array_free(out_vec);
        if (mlx.mlx_vector_array_size(out_vec) != out.len) return error.TapeOutputs;
        for (out, 0..) |*o, i| {
            var r = mlx.mlx_array_new();
            mlx.check(mlx.mlx_vector_array_get(&r, out_vec, i)) catch |e| {
                _ = mlx.mlx_array_free(r);
                return e;
            };
            o.* = try g.track(r);
        }
    }

    fn buildTape(g: *MlxOps, comptime Body: type, ctx: *const Body.Ctx, key: usize) !mlx.mlx_closure {
        const Payload = struct { proto: MlxOps, ctx: *const Body.Ctx };
        const Cb = struct {
            fn call(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                const p: *const Payload = @ptrCast(@alignCast(payload.?));
                var c = p.proto;
                defer c.deinit();
                const n = mlx.mlx_vector_array_size(input);
                var ins: [max_tape_io]T = undefined;
                if (n > ins.len) return -1;
                for (0..n) |i| {
                    var x = mlx.mlx_array_new();
                    if (mlx.mlx_vector_array_get(&x, input, i) != 0) {
                        _ = mlx.mlx_array_free(x);
                        return -1;
                    }
                    ins[i] = c.track(x) catch return -1;
                }
                var outs: [max_tape_io]T = undefined;
                Body.run(&c, p.ctx, ins[0..n], outs[0..Body.n_out]) catch return -1;
                res.* = mlx.mlx_vector_array_new_data(&outs, Body.n_out);
                return 0;
            }
            fn free(pp: *anyopaque, gpa: std.mem.Allocator) void {
                gpa.destroy(@as(*Payload, @ptrCast(@alignCast(pp))));
            }
        };
        const payload = try g.gpa.create(Payload);
        errdefer g.gpa.destroy(payload);
        payload.* = .{ .proto = g.child(), .ctx = ctx };
        const raw = mlx.mlx_closure_new_func_payload(&Cb.call, payload, null);
        defer _ = mlx.mlx_closure_free(raw);
        var compiled = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&compiled, raw, false));
        errdefer _ = mlx.mlx_closure_free(compiled);
        try g.tapes.append(g.gpa, .{ .key = key, .ctx = ctx, .compiled = compiled, .payload = payload, .free = &Cb.free });
        return compiled;
    }

    /// `mx.eval(arrays)`: one fence for a span's outputs and cache lanes.
    /// MLX's high-water mark from here (`mlx_reset_peak_memory`) and the active
    /// bytes now: a construction-time measurement's two ends (the warm-up).
    pub fn peakFrom(_: *MlxOps) u64 {
        _ = mlx.mlx_reset_peak_memory();
        var n: usize = 0;
        _ = mlx.mlx_get_active_memory(&n);
        return n;
    }

    /// Bytes MLX's high-water mark rose above `base` (a `peakFrom` value).
    pub fn peakAbove(_: *MlxOps, base: u64) u64 {
        var n: usize = 0;
        _ = mlx.mlx_get_peak_memory(&n);
        return n -| base;
    }

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

    /// Free a wave-tracked array before its wave's reset (its slot in the scope holds an empty handle after).
    pub fn drop(g: *MlxOps, x: T) void {
        var i = g.live.items.len;
        while (i > 0) {
            i -= 1;
            if (g.live.items[i].ctx == x.ctx) {
                _ = mlx.mlx_array_free(x);
                g.live.items[i] = mlx.mlx_array_new();
                return;
            }
        }
        unreachable;
    }

    /// Free a kept array now; returns the empty handle its holder keeps until its own release.
    pub fn dropKept(_: *MlxOps, x: T) T {
        _ = mlx.mlx_array_free(x);
        return mlx.mlx_array_new();
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

    pub fn arange(g: *MlxOps, start: f64, stop: f64, step: f64, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_arange(&r, start, stop, step, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
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

    pub fn full(g: *MlxOps, shape: []const c_int, v: T, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_full(&r, shape.ptr, shape.len, v, dt, g.s)) catch |e| {
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
    pub fn div(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_divide, a, b);
    }
    pub fn floorDiv(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_floor_divide, a, b);
    }
    pub fn maximum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_maximum, a, b);
    }
    pub fn minimum(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_minimum, a, b);
    }
    pub fn power(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_power, a, b);
    }
    pub fn logaddexp(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_logaddexp, a, b);
    }
    pub fn less(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_less, a, b);
    }
    pub fn lessEqual(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_less_equal, a, b);
    }
    pub fn greater(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_greater, a, b);
    }
    pub fn equal(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_equal, a, b);
    }
    pub fn greaterEqual(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_greater_equal, a, b);
    }
    pub fn logicalAnd(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_logical_and, a, b);
    }
    pub fn logicalOr(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_logical_or, a, b);
    }
    pub fn matmul(g: *MlxOps, a: T, b: T) !T {
        return g.op2(mlx.mlx_matmul, a, b);
    }

    pub fn neg(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_negative, x);
    }
    pub fn square(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_square, x);
    }
    pub fn sqrt(g: *MlxOps, x: T) !T {
        return g.op1(mlx.mlx_sqrt, x);
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

    pub fn where(g: *MlxOps, c: T, x: T, y: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_where(&r, c, x, y, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn clip(g: *MlxOps, x: T, lo: T, hi: T) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_clip(&r, x, lo, hi, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn einsum(g: *MlxOps, subscripts: [:0]const u8, operands: []const T) !T {
        const vec = mlx.mlx_vector_array_new_data(operands.ptr, operands.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_einsum(&r, subscripts.ptr, vec, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `nn.QuantizedLinear` (transpose, no biases in the fp modes).
    pub fn qmm(g: *MlxOps, x: T, w: T, sc: T, mode: sdk.QuantMode) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_quantized_matmul(&r, x, w, sc, .{}, true, mlx.mlx_optional_int.some(@intCast(quantGroup(mode))), mlx.mlx_optional_int.some(@intCast(quantBits(mode))), mode.cstr(), g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `mx.quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode)` at a tensor's own bits and
    /// group (`nn.QuantizedLinear` with `transpose`; mlx-lm's `QuantizedMultiLinear` either way). Affine carries
    /// its biases; the fp modes pass none.
    pub fn quantizedMatmul(g: *MlxOps, x: T, w: T, sc: T, biases: ?T, transpose_w: bool, bits: u32, group: u32, mode: sdk.QuantMode) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_quantized_matmul(&r, x, w, sc, biases orelse .{}, transpose_w, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `mx.dequantize(w, scales, biases, group_size, bits, mode)` at a tensor's own bits and group, the default
    /// output dtype (the scales' for affine).
    pub fn dequantizeWith(g: *MlxOps, w: T, sc: T, biases: ?T, bits: u32, group: u32, mode: sdk.QuantMode) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_dequantize(&r, w, sc, biases orelse .{}, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{}, .{}, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// The gather quant's backend call (`sdk_ext.quant.GatherQmm`): `mx.gather_qmm(x, w, scales, biases,
    /// rhs_indices=rhs, transpose=True, group_size, bits, mode, sorted_indices)` for a quantized bank, `mx.gather_mm(x,
    /// w.T, rhs_indices=rhs)` for a dense one (mode null; never sorted: MLX's dense gather_mm is wrong with sorted
    /// indices). x [rows, 1, in] -> [rows, 1, out].
    pub fn gatherMatmul(g: *MlxOps, x: T, w: T, sc: ?T, biases: ?T, rhs: T, bits: u32, group: u32, mode: ?sdk.QuantMode, sorted: bool) !T {
        if (mode != null and sc == null) return error.GatherScalesMissing;
        var r = mlx.mlx_array_new();
        if (mode) |m| {
            mlx.check(mlx.mlx_gather_qmm(&r, x, w, sc.?, biases orelse .{}, .{}, rhs, true, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), m.cstr(), sorted, g.s)) catch |e| {
                _ = mlx.mlx_array_free(r);
                return e;
            };
        } else {
            const wt = try g.transposeAxes(w, &.{ 0, 2, 1 });
            mlx.check(mlx.mlx_gather_mm(&r, x, wt, .{}, rhs, false, g.s)) catch |e| {
                _ = mlx.mlx_array_free(r);
                return e;
            };
        }
        return g.track(r);
    }

    /// `mx.dequantize` with the default output dtype (bf16 for the fp modes).
    pub fn dequantize(g: *MlxOps, w: T, sc: T, mode: sdk.QuantMode) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_dequantize(&r, w, sc, .{}, mlx.mlx_optional_int.some(@intCast(quantGroup(mode))), mlx.mlx_optional_int.some(@intCast(quantBits(mode))), mode.cstr(), .{}, .{}, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `mx.quantize(w, group, bits, mode)` of an fp mode: packed words + scales.
    pub fn quantize(g: *MlxOps, w: T, mode: sdk.QuantMode) !struct { w: T, s: T } {
        var vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(@intCast(quantGroup(mode))), mlx.mlx_optional_int.some(@intCast(quantBits(mode))), mode.cstr(), .{}, g.s));
        defer _ = mlx.mlx_vector_array_free(vec);
        var q = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&q, vec, 0)) catch |e| {
            _ = mlx.mlx_array_free(q);
            return e;
        };
        const qw = try g.track(q);
        var sc = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&sc, vec, 1)) catch |e| {
            _ = mlx.mlx_array_free(sc);
            return e;
        };
        return .{ .w = qw, .s = try g.track(sc) };
    }

    pub fn reshape(g: *MlxOps, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn transposeAxes(g: *MlxOps, x: T, axes: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_transpose_axes(&r, x, axes.ptr, axes.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn broadcastTo(g: *MlxOps, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_broadcast_to(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn expandDims(g: *MlxOps, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_expand_dims(&r, x, axis, g.s)) catch |e| {
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

    /// `mx.slice_update(src, update, start_indices, axes=all)` with an int32
    /// start array (the dynamic slice update the Python KV lanes write with).
    pub fn sliceUpdateDyn(g: *MlxOps, src: T, update: T, start: T) !T {
        const n = mlx.getShape(update).len;
        var axes: [max_dims]c_int = undefined;
        for (0..n) |i| axes[i] = @intCast(i);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice_update_dynamic(&r, src, update, start, &axes, n, g.s)) catch |e| {
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

    pub fn stack(g: *MlxOps, xs: []const T, axis: c_int) !T {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_stack_axis(&r, vec, axis, g.s)) catch |e| {
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

    pub fn takeAlongAxis(g: *MlxOps, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_along_axis(&r, x, idx, axis, g.s)) catch |e| {
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

    fn axisOp(g: *MlxOps, comptime f: anytype, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(f(&r, x, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }
    pub fn sort(g: *MlxOps, x: T, axis: c_int) !T {
        return g.axisOp(mlx.mlx_sort_axis, x, axis);
    }
    pub fn argsort(g: *MlxOps, x: T, axis: c_int) !T {
        return g.axisOp(mlx.mlx_argsort_axis, x, axis);
    }

    pub fn argpartition(g: *MlxOps, x: T, kth: c_int, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_argpartition_axis(&r, x, kth, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `mx.cumsum` (inclusive, forward).
    pub fn cumsum(g: *MlxOps, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_cumsum(&r, x, axis, false, true, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn repeat(g: *MlxOps, x: T, repeats: c_int, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_repeat_axis(&r, x, repeats, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    fn applyUnary(g: *MlxOps, f: mlx.mlx_closure, x: T) !T {
        const in_arr = [_]T{x};
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_closure_apply(&out_vec, f, in_vec));
        defer _ = mlx.mlx_vector_array_free(out_vec);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&r, out_vec, 0)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// `nn.silu`: the compiled `x * sigmoid(x)`.
    pub fn silu(g: *MlxOps, x: T) !T {
        return g.applyUnary(g.silu_fn, x);
    }

    /// `nn.softplus`: the compiled `logaddexp(x, 0)`.
    pub fn softplus(g: *MlxOps, x: T) !T {
        return g.applyUnary(g.softplus_fn, x);
    }

    /// `mx.hadamard_transform(x, scale)` over the last axis.
    pub fn hadamard(g: *MlxOps, x: T, scale: f32) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_hadamard_transform(&r, x, mlx.mlx_optional_float.some(scale), g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
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

    /// `mx.gather_qmm(x, w, scales, rhs_indices=idx, transpose=True, mode)` (the
    /// resident `SwitchLinear`; unsorted, no biases).
    pub fn gatherQmm(g: *MlxOps, x: T, w: T, sc: T, idx: T, mode: sdk.QuantMode) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_gather_qmm(&r, x, w, sc, .{}, .{}, idx, true, mlx.mlx_optional_int.some(@intCast(quantGroup(mode))), mlx.mlx_optional_int.some(@intCast(quantBits(mode))), mode.cstr(), false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn argmax(g: *MlxOps, x: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_argmax_axis(&r, x, axis, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn logsumexp(g: *MlxOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(mlx.mlx_logsumexp_axis, x, axis, keepdims);
    }

    /// Evaluates `x` (only a wait when an eval already produced it) and copies its values,
    /// row-major, into `out` (x's size), read in place (`readInPlace`).
    pub fn hostU32(g: *MlxOps, x: T, out: []u32) ![]const u32 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .uint32); // argmax outputs and draft ids
        return readInPlace(u32, u32, x, mlx.mlx_array_data_uint32(x), out);
    }

    pub fn hostF32(g: *MlxOps, x: T, out: []f32) ![]const f32 {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .float32);
        return readInPlace(f32, f32, x, mlx.mlx_array_data_float32(x), out);
    }

    pub fn hostBool(g: *MlxOps, x: T, out: []bool) ![]const bool {
        g.waits += 1;
        try mlx.check(mlx.mlx_array_eval(x));
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .bool_);
        return readInPlace(bool, bool, x, mlx.mlx_array_data_bool(x), out);
    }

    /// The values of `x`, which an earlier eval produced, copied row-major into `out` (x's size): no wait, and
    /// `error.NotEvaluated` when no eval has produced `x`.
    pub fn readU32(_: *MlxOps, x: T, out: []u32) ![]const u32 {
        try produced(x);
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .uint32);
        return readInPlace(u32, u32, x, mlx.mlx_array_data_uint32(x), out);
    }

    /// `readU32` for a bool array.
    pub fn readBool(_: *MlxOps, x: T, out: []bool) ![]const bool {
        try produced(x);
        if (mlx.mlx_array_size(x) != out.len) return error.HostReadSize;
        std.debug.assert(mlx.mlx_array_dtype(x) == .bool_);
        return readInPlace(bool, bool, x, mlx.mlx_array_data_bool(x), out);
    }

    fn produced(x: T) !void {
        var ok = false;
        try mlx.check(mlx._mlx_array_is_available(&ok, x));
        if (!ok) return error.NotEvaluated;
    }

    /// Greedy pick: `mx.argmax` over every logit of `x` (one row), evaluated and read.
    pub fn hostArgmax(g: *MlxOps, x: T) !u32 {
        g.waits += 1;
        const flat = try g.reshape(x, &.{-1});
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_argmax_axis(&r, flat, 0, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        const am = try g.track(r);
        try mlx.check(mlx.mlx_array_eval(am));
        const p = mlx.mlx_array_data_uint32(am) orelse return error.MlxNoData;
        return p[0];
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

    /// The base array a view resolves to (itself when it is no slice).
    pub fn baseOf(g: *const TraceOps, x: T) T {
        var b = x;
        while (g.slice_base.get(b)) |p| b = p;
        return b;
    }

    /// Prepared launches of kernel `k` since the `from`-th.
    pub fn launchesOf(g: *const TraceOps, from: usize, k: xk.Kernel) usize {
        return std.mem.count(xk.Kernel, g.launched.items[from..], &.{k});
    }

    /// The bytes host array `x` was made from (`record_host` set before it was made).
    pub fn hostBytesOf(g: *const TraceOps, x: T) ?[]const u8 {
        return g.host_data.get(x);
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
    /// No device memory on the trace backend.
    pub fn peakFrom(_: *TraceOps) u64 {
        return 0;
    }

    pub fn peakAbove(_: *TraceOps, _: u64) u64 {
        return 0;
    }

    pub fn evalAll(g: *TraceOps, xs: []const T) !void {
        if (g.track_live) try g.judgeUpdates();
        for (xs) |x| g.touch(x);
        try g.evals.append(g.gpa, g.nodes.items.len);
        try g.evaluated.appendSlice(g.gpa, xs);
    }
    /// `track_live`: the updates built since the last eval, each in place or a whole-source copy (`lane_writes`).
    fn judgeUpdates(g: *TraceOps) !void {
        const to = g.nodes.items.len;
        var i = g.last_eval_node;
        while (i < to) : (i += 1) {
            if (g.nodes.items[i].op != .slice_update) continue;
            const u = g.update_src.get(@intCast(i)) orelse continue;
            const src = g.baseOf(u.src);
            var pinned: ?T = null;
            var it = g.live_kept.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* > 0 and g.baseOf(e.key_ptr.*) == src) {
                    pinned = e.key_ptr.*;
                    break;
                }
            }
            const nd = g.nodes.items[i];
            try g.lane_writes.append(g.gpa, .{ .node = @intCast(i), .src = src, .eval = g.evals.items.len, .full_bytes = @as(u64, @intCast(nd.shape.numel())) * dtypeSize(nd.dtype), .update_bytes = u.bytes, .pinned_by = pinned });
        }
        g.last_eval_node = to;
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
    /// An array freed before its scope ends; any later shape, dtype or evaluation of it counts in `use_after_drop`.
    pub fn drop(g: *TraceOps, x: T) void {
        g.dropped.append(g.gpa, x) catch @panic("trace: out of memory");
    }
    pub fn dropKept(g: *TraceOps, x: T) T {
        if (g.track_live) if (g.live_kept.getPtr(x)) |c| {
            c.* -= 1;
        };
        g.drop(x);
        return x;
    }
    fn touch(g: *TraceOps, x: T) void {
        if (std.mem.indexOfScalar(T, g.dropped.items, x) != null) g.use_after_drop += 1;
    }
    pub fn adopt(_: *TraceOps, x: T) !T {
        return x;
    }

    fn push(g: *TraceOps, op: Op, dtype: Dtype, shape: Shape) !T {
        try g.nodes.append(g.gpa, .{ .op = op, .dtype = dtype, .shape = shape });
        return @intCast(g.nodes.items.len - 1);
    }

    /// A leaf the test hands the graph (weights, cache state, inputs).
    pub fn input(g: *TraceOps, shape: []const c_int, dtype: Dtype) !T {
        return g.push(.input, dtype, Shape.of(shape));
    }

    /// A host array's recorded bytes, writable (`record_host` only): the trace's stand-in for `MlxOps.hostBytes`.
    pub fn hostBytes(g: *TraceOps, x: T) ![]u8 {
        return g.host_data.get(x) orelse error.NoHostData;
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

    pub fn node(g: *const TraceOps, x: T) Node {
        return g.nodes.items[x];
    }

    /// Ops recorded after node count `from`, inputs excluded.
    pub fn opsSince(g: *const TraceOps, gpa: std.mem.Allocator, from: usize) ![]Op {
        var out: std.ArrayList(Op) = .empty;
        for (g.nodes.items[from..]) |n| if (n.op != .input) try out.append(gpa, n.op);
        return out.toOwnedSlice(gpa);
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

    pub fn arange(g: *TraceOps, start: f64, stop: f64, step: f64, dt: Dtype) !T {
        const n: c_int = @intFromFloat(@max(0, @ceil((stop - start) / step)));
        return g.push(.arange, dt, Shape.of(&.{n}));
    }

    pub fn ones(g: *TraceOps, shape: []const c_int, dt: Dtype) !T {
        return g.push(.ones, dt, Shape.of(shape));
    }

    pub fn zeros(g: *TraceOps, shape: []const c_int, dt: Dtype) !T {
        return g.push(.zeros, dt, Shape.of(shape));
    }

    pub fn full(g: *TraceOps, shape: []const c_int, _: T, dt: Dtype) !T {
        return g.push(.full, dt, Shape.of(shape));
    }

    /// MLX returns the input itself for a same-dtype astype: no node.
    pub fn astype(g: *TraceOps, x: T, dt: Dtype) !T {
        if (g.dtypeOf(x) == dt) return x;
        return g.push(.astype, dt, g.shapeOf(x));
    }

    fn arith(g: *TraceOps, op: Op, a: T, b: T) !T {
        return g.push(op, promote(g.dtypeOf(a), g.dtypeOf(b)), try broadcast(g.shapeOf(a), g.shapeOf(b)));
    }
    fn compare(g: *TraceOps, op: Op, a: T, b: T) !T {
        return g.push(op, .bool_, try broadcast(g.shapeOf(a), g.shapeOf(b)));
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
    pub fn div(g: *TraceOps, a: T, b: T) !T {
        const d = promote(g.dtypeOf(a), g.dtypeOf(b));
        return g.push(.div, if (isFloat(d)) d else .float32, try broadcast(g.shapeOf(a), g.shapeOf(b)));
    }
    pub fn floorDiv(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.floor_div, a, b);
    }
    pub fn maximum(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.maximum, a, b);
    }
    pub fn minimum(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.minimum, a, b);
    }
    pub fn power(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.power, a, b);
    }
    pub fn logaddexp(g: *TraceOps, a: T, b: T) !T {
        return g.arith(.logaddexp, a, b);
    }
    pub fn less(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.less, a, b);
    }
    pub fn lessEqual(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.less_equal, a, b);
    }
    pub fn greater(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.greater, a, b);
    }
    pub fn equal(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.equal, a, b);
    }
    pub fn logicalAnd(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.logical_and, a, b);
    }
    pub fn greaterEqual(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.greater_equal, a, b);
    }

    /// The region runs inline between two markers (a test pins its boundary).
    /// Records that region `Body` is compiled for `ctx` (the MLX backend's construction step).
    pub fn prepareTape(g: *TraceOps, comptime Body: type, ctx: *const Body.Ctx) !void {
        const slots = &g.regions[@backingInt(Body.region)];
        for (slots) |sl| if (sl == @as(?*const anyopaque, ctx)) return;
        for (slots) |*sl| if (sl.* == null) {
            sl.* = ctx;
            return;
        };
        return error.RegionContextsFull;
    }

    pub fn tape(g: *TraceOps, comptime Body: type, ctx: *const Body.Ctx, inputs: []const T, out: []T) !void {
        for (g.regions[@backingInt(Body.region)]) |sl| {
            if (sl == @as(?*const anyopaque, ctx)) break;
        } else return error.RegionNotPrepared;
        var h = std.hash.Wyhash.init(@backingInt(Body.region));
        h.update(std.mem.asBytes(&@intFromPtr(ctx)));
        for (inputs) |x| {
            const nd = g.nodes.items[x];
            h.update(std.mem.asBytes(&nd.dtype));
            h.update(std.mem.sliceAsBytes(nd.shape.d[0..nd.shape.n]));
        }
        if (!(try g.tape_sigs.getOrPut(g.gpa, h.final())).found_existing) g.compiles += 1;
        _ = try g.push(.tape_begin, .bool_, .{});
        try Body.run(g, ctx, inputs, out);
        _ = try g.push(.tape_end, .bool_, .{});
    }

    /// fp-mode `mx.quantize`: `[out, in]` -> words `[out, in * bits / 32]` u32 + scales `[out, in / group]` u8.
    pub fn quantize(g: *TraceOps, w: T, mode: sdk.QuantMode) !struct { w: T, s: T } {
        const sh = g.shapeOf(w);
        const in_dim = sh.dim(-1);
        var ws = sh;
        ws.d[ws.n - 1] = @divExact(in_dim * @as(c_int, @intCast(quantBits(mode))), 32);
        var ss = sh;
        ss.d[ss.n - 1] = @divExact(in_dim, @as(c_int, @intCast(quantGroup(mode))));
        const qw = try g.push(.quantize, .uint32, ws);
        return .{ .w = qw, .s = try g.push(.quantize, .uint8, ss) };
    }
    pub fn logicalOr(g: *TraceOps, a: T, b: T) !T {
        return g.compare(.logical_or, a, b);
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

    fn unary(g: *TraceOps, op: Op, x: T) !T {
        return g.push(op, g.dtypeOf(x), g.shapeOf(x));
    }
    pub fn neg(g: *TraceOps, x: T) !T {
        return g.unary(.neg, x);
    }
    pub fn square(g: *TraceOps, x: T) !T {
        return g.unary(.square, x);
    }
    pub fn sqrt(g: *TraceOps, x: T) !T {
        return g.unary(.sqrt, x);
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
    pub fn softplus(g: *TraceOps, x: T) !T {
        return g.unary(.softplus, x);
    }
    pub fn softmax(g: *TraceOps, x: T, _: c_int) !T {
        return g.unary(.softmax, x);
    }
    pub fn sort(g: *TraceOps, x: T, _: c_int) !T {
        return g.unary(.sort, x);
    }
    pub fn argsort(g: *TraceOps, x: T, _: c_int) !T {
        return g.push(.argsort, .uint32, g.shapeOf(x));
    }
    pub fn argpartition(g: *TraceOps, x: T, _: c_int, _: c_int) !T {
        return g.push(.argpartition, .uint32, g.shapeOf(x));
    }
    pub fn cumsum(g: *TraceOps, x: T, _: c_int) !T {
        const d = g.dtypeOf(x);
        return g.push(.cumsum, if (d == .bool_) .int32 else d, g.shapeOf(x));
    }

    pub fn transpose(g: *TraceOps, x: T) !T {
        const s = g.shapeOf(x);
        var out: Shape = .{ .n = s.n };
        for (0..s.n) |i| out.d[i] = s.d[s.n - 1 - i];
        return g.push(.transpose, g.dtypeOf(x), out);
    }

    pub fn transposeAxes(g: *TraceOps, x: T, axes: []const c_int) !T {
        const s = g.shapeOf(x);
        if (axes.len != s.n) return error.TransposeAxes;
        var out: Shape = .{ .n = s.n };
        for (axes, 0..) |a, i| out.d[i] = s.d[normAxis(a, s.n)];
        return g.push(.transpose_axes, g.dtypeOf(x), out);
    }

    pub fn where(g: *TraceOps, c: T, x: T, y: T) !T {
        if (g.dtypeOf(c) != .bool_) return error.WhereCondition;
        const s = try broadcast(try broadcast(g.shapeOf(c), g.shapeOf(x)), g.shapeOf(y));
        return g.push(.where, promote(g.dtypeOf(x), g.dtypeOf(y)), s);
    }

    /// MLX takes n = m * 2^k (m in 1, 12, 20, 28) on the last axis.
    pub fn hadamard(g: *TraceOps, x: T, _: f32) !T {
        const sh = g.shapeOf(x);
        if (sh.n == 0) return error.HadamardSize;
        var n: c_int = sh.d[sh.n - 1];
        if (n <= 0) return error.HadamardSize;
        for ([_]c_int{ 12, 20, 28 }) |m| {
            if (@rem(n, m) == 0 and std.math.isPowerOfTwo(@divExact(n, m))) n = @divExact(n, m);
        }
        if (!std.math.isPowerOfTwo(n)) return error.HadamardSize;
        if (!isFloat(g.dtypeOf(x))) return error.HadamardDtype;
        return g.push(.hadamard, g.dtypeOf(x), sh);
    }

    /// A marker: the GPU would start on the arrays here.
    pub fn asyncEval(g: *TraceOps, xs: []const T) !void {
        if (g.track_live) try g.judgeUpdates();
        const start = g.committed.items.len;
        try g.committed.appendSlice(g.gpa, xs);
        try g.commit_spans.append(g.gpa, .{ start, g.committed.items.len });
        _ = try g.push(.async_eval, .bool_, .{});
    }

    /// The routing barrier: a marker, then the script's next ids.
    pub fn hostIds(g: *TraceOps, x: T, out: []u16) ![]const u16 {
        _ = try g.push(.host_read, .bool_, .{});
        if (g.dtypeOf(x) != .int32) return error.HostIdsDtype;
        if (g.shapeOf(x).numel() != @as(i64, @intCast(out.len))) return error.HostIdsSize;
        const hv = g.host_values orelse return error.NoHostValues;
        try hv.ids(hv.ctx, out);
        return out;
    }

    /// A greedy pick: a marker, then the script's next token.
    pub fn hostArgmax(g: *TraceOps, _: T) !u32 {
        _ = try g.push(.host_read, .bool_, .{});
        const hv = g.host_values orelse return error.NoHostValues;
        return hv.argmax(hv.ctx);
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

    pub fn prepareLaunch(g: *TraceOps, k: xk.Kernel, cfg: *const xk.LaunchConfig) !Prepared {
        g.prepared_live += 1;
        return .{ .k = k, .cfg = cfg.* };
    }

    /// Launches `p` (counted apart from per-call launches).
    pub fn launchPrepared(g: *TraceOps, p: *const Prepared, inputs: []const T, out: []T) !void {
        g.prepared_launches += 1;
        try g.launched.append(g.gpa, p.k);
        return g.launch(p.k, inputs, &p.cfg, out);
    }

    pub fn releasePrepared(g: *TraceOps, _: *Prepared) void {
        g.prepared_live -= 1;
    }

    /// The resident switch: x [..., 1, K] x w [E, N, K*bits/32] at rhs indices [...]
    /// -> [indices..., 1, N] at x's dtype.
    pub fn gatherQmm(g: *TraceOps, x: T, w: T, sc: T, idx: T, mode: sdk.QuantMode) !T {
        const sx = g.shapeOf(x);
        const sw = g.shapeOf(w);
        const si = g.shapeOf(idx);
        const in_dim = @divExact(sw.dim(-1) * 32, @as(c_int, @intCast(quantBits(mode))));
        // One row per index, or the batched form (x's batch dims == idx's, M rows each).
        const m_rows = sx.dim(-2);
        const batched = m_rows != 1 and sx.n == si.n + 2 and std.mem.eql(c_int, sx.d[0..si.n], si.d[0..si.n]);
        if (sw.n != 3 or (m_rows != 1 and !batched) or sx.dim(-1) != in_dim or g.shapeOf(sc).dim(-1) * @as(c_int, @intCast(quantGroup(mode))) != in_dim) return error.GatherQmmShape;
        if (!isInteger(g.dtypeOf(idx))) return error.GatherQmmIndices;
        var out = si;
        out.d[out.n] = m_rows;
        out.d[out.n + 1] = sw.dim(1);
        out.n += 2;
        return g.push(.gather_qmm, g.dtypeOf(x), out);
    }

    /// An event wait's alias of `x` (the GPU reads it only after the event).
    /// One array aliased behind an event wait at `value` after `n_deps` arrays.
    pub fn eventAlias(g: *TraceOps, x: T, value: u64, n_deps: usize) !T {
        try g.waits.append(g.gpa, .{ .value = value, .n_deps = @intCast(n_deps) });
        return g.push(.event_wait, g.dtypeOf(x), g.shapeOf(x));
    }

    pub fn argmax(g: *TraceOps, x: T, axis: c_int) !T {
        const s = g.shapeOf(x);
        const a = normAxis(axis, s.n);
        var out: Shape = .{};
        for (0..s.n) |i| if (i != a) {
            out.d[out.n] = s.d[i];
            out.n += 1;
        };
        return g.push(.argmax, .uint32, out);
    }

    pub fn logsumexp(g: *TraceOps, x: T, axis: c_int, keepdims: bool) !T {
        return g.reduce(.logsumexp, x, axis, keepdims, g.dtypeOf(x));
    }

    fn scripted(g: *TraceOps, x: T, n: usize) !TraceOps.HostValues {
        _ = try g.push(.host_read, .bool_, .{});
        if (g.shapeOf(x).numel() != @as(i64, @intCast(n))) return error.HostReadSize;
        return g.host_values orelse error.NoHostValues;
    }

    pub fn hostU32(g: *TraceOps, x: T, out: []u32) ![]const u32 {
        if (g.dtypeOf(x) != .uint32) return error.HostReadDtype;
        const hv = try g.scripted(x, out.len);
        try (hv.u32s orelse return error.NoHostValues)(hv.ctx, out);
        return out;
    }

    pub fn hostF32(g: *TraceOps, x: T, out: []f32) ![]const f32 {
        if (g.dtypeOf(x) != .float32) return error.HostReadDtype;
        const hv = try g.scripted(x, out.len);
        try (hv.f32s orelse return error.NoHostValues)(hv.ctx, out);
        return out;
    }

    pub fn hostBool(g: *TraceOps, x: T, out: []bool) ![]const bool {
        if (g.dtypeOf(x) != .bool_) return error.HostReadDtype;
        const hv = try g.scripted(x, out.len);
        try (hv.bools orelse return error.NoHostValues)(hv.ctx, out);
        return out;
    }

    pub fn clip(g: *TraceOps, x: T, lo: T, hi: T) !T {
        return g.push(.clip, promote(promote(g.dtypeOf(x), g.dtypeOf(lo)), g.dtypeOf(hi)), g.shapeOf(x));
    }

    /// Output shape of an einsum over `a..z` letters and one `...`.
    pub fn einsum(g: *TraceOps, subscripts: [:0]const u8, operands: []const T) !T {
        const arrow = std.mem.indexOf(u8, subscripts, "->") orelse return error.Einsum;
        var sizes: [26]c_int = @splat(-1);
        var ell: Shape = .{};
        var it = std.mem.splitScalar(u8, subscripts[0..arrow], ',');
        var dt: ?Dtype = null;
        for (operands) |opnd| {
            const spec = it.next() orelse return error.Einsum;
            const s = g.shapeOf(opnd);
            dt = if (dt) |d| promote(d, g.dtypeOf(opnd)) else g.dtypeOf(opnd);
            var letters: usize = 0;
            for (spec) |ch| {
                if (ch >= 'a' and ch <= 'z') letters += 1;
            }
            const has_ell = std.mem.indexOf(u8, spec, "...") != null;
            const n_ell: usize = if (has_ell) s.n - letters else 0;
            if (!has_ell and letters != s.n) return error.Einsum;
            var axis: usize = 0;
            var i: usize = 0;
            while (i < spec.len) {
                if (std.mem.startsWith(u8, spec[i..], "...")) {
                    var e: Shape = .{ .n = @intCast(n_ell) };
                    @memcpy(e.d[0..n_ell], s.d[axis .. axis + n_ell]);
                    ell = try broadcast(ell, e);
                    axis += n_ell;
                    i += 3;
                    continue;
                }
                const li = spec[i] - 'a';
                if (sizes[li] >= 0 and sizes[li] != s.d[axis] and sizes[li] != 1 and s.d[axis] != 1) return error.Einsum;
                if (sizes[li] <= 1) sizes[li] = s.d[axis];
                axis += 1;
                i += 1;
            }
        }
        var out: Shape = .{};
        const out_spec = subscripts[arrow + 2 ..];
        var i: usize = 0;
        while (i < out_spec.len) {
            if (std.mem.startsWith(u8, out_spec[i..], "...")) {
                @memcpy(out.d[out.n..][0..ell.n], ell.d[0..ell.n]);
                out.n += ell.n;
                i += 3;
                continue;
            }
            out.d[out.n] = sizes[out_spec[i] - 'a'];
            out.n += 1;
            i += 1;
        }
        return g.push(.einsum, dt orelse return error.Einsum, out);
    }

    /// fp-mode `quantized_matmul(transpose=True)`: `[..., in] -> [..., out]`,
    /// dtype of x (mlx/ops.cpp `quantized_matmul`).
    pub fn qmm(g: *TraceOps, x: T, w: T, sc: T, mode: sdk.QuantMode) !T {
        const sx = g.shapeOf(x);
        const sw = g.shapeOf(w);
        const in_dim = @divExact(sw.dim(-1) * 32, @as(c_int, @intCast(quantBits(mode))));
        if (sx.dim(-1) != in_dim or g.shapeOf(sc).dim(-1) * @as(c_int, @intCast(quantGroup(mode))) != in_dim) return error.QmmShape;
        var out = sx;
        out.d[out.n - 1] = sw.dim(-2);
        return g.push(.qmm, g.dtypeOf(x), out);
    }

    pub fn dequantize(g: *TraceOps, w: T, _: T, mode: sdk.QuantMode) !T {
        var out = g.shapeOf(w);
        out.d[out.n - 1] = @divExact(out.d[out.n - 1] * 32, @as(c_int, @intCast(quantBits(mode))));
        return g.push(.dequantize, .bfloat16, out);
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

    pub fn broadcastTo(g: *TraceOps, x: T, shape: []const c_int) !T {
        const out = try broadcast(g.shapeOf(x), Shape.of(shape));
        if (!out.eql(Shape.of(shape))) return error.BroadcastMismatch;
        return g.push(.broadcast_to, g.dtypeOf(x), out);
    }

    pub fn expandDims(g: *TraceOps, x: T, axis: c_int) !T {
        const s = g.shapeOf(x);
        const a = normAxis(axis, s.n + 1);
        var out: Shape = .{ .n = s.n + 1 };
        var j: usize = 0;
        for (0..out.n) |i| {
            if (i == a) {
                out.d[i] = 1;
            } else {
                out.d[i] = s.d[j];
                j += 1;
            }
        }
        return g.push(.expand_dims, g.dtypeOf(x), out);
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

    pub fn sliceUpdateDyn(g: *TraceOps, src: T, update: T, start: T) !T {
        const ss = g.shapeOf(src);
        const us = g.shapeOf(update);
        if (ss.n != us.n or g.shapeOf(start).n != 1 or g.shapeOf(start).d[0] != us.n) return error.SliceUpdateShape;
        for (0..ss.n) |i| if (us.d[i] > ss.d[i]) return error.SliceUpdateShape;
        const y = try g.push(.slice_update, g.dtypeOf(src), ss);
        if (g.track_live) try g.update_src.put(g.gpa, y, .{ .src = src, .bytes = @as(u64, @intCast(us.numel())) * dtypeSize(g.dtypeOf(update)) });
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

    /// A row concatenate's bytes, when every input has recorded bytes of the output's dtype.
    fn recordRows(g: *TraceOps, y: T, xs: []const T) !void {
        var n: usize = 0;
        for (xs) |x| {
            if (g.dtypeOf(x) != g.dtypeOf(y)) return;
            n += (g.host_data.get(x) orelse return).len;
        }
        const bytes = try g.gpa.alloc(u8, n);
        errdefer g.gpa.free(bytes);
        var at: usize = 0;
        for (xs) |x| {
            const b = g.host_data.get(x).?;
            @memcpy(bytes[at..][0..b.len], b);
            at += b.len;
        }
        try g.host_data.put(g.gpa, y, bytes);
    }

    pub fn stack(g: *TraceOps, xs: []const T, axis: c_int) !T {
        const s = g.shapeOf(xs[0]);
        for (xs[1..]) |x| if (!g.shapeOf(x).eql(s)) return error.StackShape;
        const a = normAxis(axis, s.n + 1);
        var out: Shape = .{ .n = s.n + 1 };
        var j: usize = 0;
        for (0..out.n) |i| {
            if (i == a) {
                out.d[i] = @intCast(xs.len);
            } else {
                out.d[i] = s.d[j];
                j += 1;
            }
        }
        return g.push(.stack, g.dtypeOf(xs[0]), out);
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

    pub fn takeAlongAxis(g: *TraceOps, x: T, idx: T, axis: c_int) !T {
        var sx = g.shapeOf(x);
        const si = g.shapeOf(idx);
        const a = normAxis(axis, sx.n);
        sx.d[a] = 1;
        var out = try broadcast(sx, si);
        out.d[a] = si.d[a];
        return g.push(.take_along_axis, g.dtypeOf(x), out);
    }

    fn reduce(g: *TraceOps, op: Op, x: T, axis: c_int, keepdims: bool, dt: Dtype) !T {
        const s = g.shapeOf(x);
        const a = normAxis(axis, s.n);
        var out: Shape = .{};
        for (0..s.n) |i| {
            if (i == a) {
                if (!keepdims) continue;
                out.d[out.n] = 1;
            } else {
                out.d[out.n] = s.d[i];
            }
            out.n += 1;
        }
        return g.push(op, dt, out);
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

    pub fn repeat(g: *TraceOps, x: T, repeats: c_int, axis: c_int) !T {
        var out = g.shapeOf(x);
        out.d[normAxis(axis, out.n)] *= repeats;
        return g.push(.repeat, g.dtypeOf(x), out);
    }
};

const testing = std.testing;

/// Tests: `mx.quantize(w, group, bits, "affine")` as mlx-lm stores a layer: words, scales and biases, in `g`'s scope.
pub fn quantizeAffine(g: *MlxOps, w: MlxOps.T, group: u32, bits: u32) ![3]MlxOps.T {
    var vec = mlx.mlx_vector_array{ .ctx = null };
    try mlx.check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(@intCast(group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, g.s));
    defer _ = mlx.mlx_vector_array_free(vec);
    var parts: [3]MlxOps.T = undefined;
    for (&parts, 0..) |*p, i| {
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(p, vec, i));
        p.* = try g.adopt(p.*);
    }
    return parts;
}

/// `G.name` takes exactly `params` after `*G` (or `*const G`) and returns
/// `Payload` (through an error union when `errors`): the kernels contract's
/// method shapes, as the integration branch checks them.
fn hasMethod(comptime G: type, comptime name: []const u8, comptime self_const: bool, comptime params: []const type, comptime Payload: type, comptime errors: bool) bool {
    if (!@hasDecl(G, name)) return false;
    const f = @typeInfo(@TypeOf(@field(G, name))).@"fn";
    if (f.param_types.len != params.len + 1) return false;
    const self_t = f.param_types[0] orelse return false;
    if (self_t != (if (self_const) *const G else *G)) return false;
    for (params, f.param_types[1..]) |want, got| if ((got orelse return false) != want) return false;
    const ret = f.return_type orelse return false;
    if (!errors) return ret == Payload;
    const info = @typeInfo(ret);
    return info == .error_union and info.error_union.payload == Payload;
}

comptime {
    // The gather quant's adapter binds this backend (its `gatherMatmul` and the activation's ops).
    sdk_ext.quant.checkAccepted(sdk_ext.quant.FromGatherMatmul(sdk_ext.quant.GatherQmm), MlxOps);
}
