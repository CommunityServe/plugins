//! The C symbols the copied expert streamer calls that mlx-serve serves from C++ shims over MLX's private headers
//! (sdk/weights/streamed/csrc: mlx_event_shim.cpp, mlx_alloc_shim.cpp, mlx_launch_cache.inc, q3_event_shim.mm), as this
//! plugin's own, each over the server's host table (`abi.host`), which serves them from those same shims
//! (src/host/plugin_host_impl.zig). The kit ships no MLX headers: a plugin reaches MLX's internals only through the
//! table. `install` keeps the table at load, before the streamer opens; with none kept, every call refuses (kernels fall
//! back to mlx-c's own).

const std = @import("std");
const kit = @import("mlx_serve");
const host = kit.abi.host;
const common = kit.abi.common;
const mlx = kit.mlx;

var table: ?*const host.Host = null;
var last_error: [256:0]u8 = @splat(0);

/// Keeps the server's host table (the load's `LoadCtx.host`); false when it is absent or older than the kit's.
pub fn install(raw: ?*const anyopaque) bool {
    _ = kit.host.Host.of(raw) catch return false;
    table = @ptrCast(@alignCast(raw.?));
    return true;
}

fn gpu() ?*const host.HostGpu {
    const t = table orelse return null;
    return t.gpu;
}

fn fail(comptime op: []const u8, v: common.Verdict) c_int {
    const text = std.fmt.bufPrint(last_error[0 .. last_error.len - 1], op ++ ": {s}", .{v.why.slice()}) catch op;
    last_error[text.len] = 0;
    return -1;
}

fn noTable(comptime op: []const u8) c_int {
    const text = op ++ ": the server handed this load no host table";
    @memcpy(last_error[0..text.len], text);
    last_error[text.len] = 0;
    return -1;
}

fn arrays(xs: [*]const mlx.mlx_array) [*]const common.Array {
    return @ptrCast(xs);
}

// ── mlx_event_shim.cpp ──

/// The event shim's ABI version (sdk_ext/expert/event.zig `abi_version`).
export fn dsv41ev_abi() i32 {
    return 2026092801;
}

export fn dsv41ev_create_metal(start: u64, object: *u64) i32 {
    const g = gpu() orelse return noTable("event_metal");
    var ev: host.Event = .{};
    const v = g.event_metal(g.ctx, start, &ev);
    if (!v.isOk()) return fail("event_metal", v);
    object.* = ev.object;
    return ev.id;
}

export fn dsv41ev_create_host(word: *i64, timeout_ns: i64) i32 {
    const g = gpu() orelse return noTable("event_host");
    var ev: host.Event = .{};
    const v = g.event_host(g.ctx, word, timeout_ns, &ev);
    if (!v.isOk()) return fail("event_host", v);
    return ev.id;
}

export fn dsv41ev_wait(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, deps: ?[*]const mlx.mlx_array, n_deps: usize, track_inputs: bool, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int {
    const g = gpu() orelse return noTable("wait");
    const d: ?[*]const common.Array = if (deps) |p| arrays(p) else null;
    const v = g.wait(g.ctx, arrays(xs), n, event, value, d, n_deps, track_inputs, .{ .ctx = s.ctx }, @ptrCast(outs));
    return if (v.isOk()) 0 else fail("wait", v);
}

export fn dsv41ev_signal(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int {
    const g = gpu() orelse return noTable("signal");
    const v = g.signal(g.ctx, arrays(xs), n, event, value, .{ .ctx = s.ctx }, @ptrCast(outs));
    return if (v.isOk()) 0 else fail("signal", v);
}

export fn dsv41ev_value(event: i32) u64 {
    const g = gpu() orelse return 0;
    return g.event_value(g.ctx, event);
}

export fn dsv41ev_stats(out: *[8]i64) void {
    const g = gpu() orelse {
        out.* = @splat(0);
        return;
    };
    var st: host.EventStats = .{};
    g.event_stats(g.ctx, &st);
    out.* = .{ st.gpu_waits, st.gpu_signals, st.host_ready, st.host_blocked, st.host_timeouts, st.host_wait_ns, st.cpu_passthrough, st.events };
}

export fn dsv41ev_last_error() [*:0]const u8 {
    return &last_error;
}

// ── mlx_alloc_shim.cpp ──

export fn dsv41_alloc_uninit(out: *mlx.mlx_array, shape: [*]const c_int, ndim: usize, dtype: mlx.mlx_dtype) c_int {
    const g = gpu() orelse return noTable("alloc_uninit");
    const v = g.alloc_uninit(g.ctx, @ptrCast(shape), ndim, @intFromEnum(dtype), @ptrCast(out));
    return if (v.isOk()) 0 else fail("alloc_uninit", v);
}

export fn dsv41_alloc_last_error() [*:0]const u8 {
    return &last_error;
}

// ── q3_event_shim.mm ──

/// The read pool's completion (q3_lookahead4_exl3.c), from its worker threads: the host's `event_set` hands the value to
/// the MTLSharedEvent, as the shim's `setSignaledValue:` does.
export fn q3ld_mtl_signal(event: ?*anyopaque, value: u64) void {
    const g = gpu() orelse return;
    g.event_set(g.ctx, .{ .kind = .metal, .object = @intFromPtr(event) }, value);
}

// ── mlx_launch_cache.inc ──

extern "c" fn mlx_vector_string_size(vec: mlx.mlx_vector_string) usize;
extern "c" fn mlx_vector_string_get(res: *?[*:0]const u8, vec: mlx.mlx_vector_string, idx: usize) c_int;

/// The most input or output names one kernel takes.
const max_names = 32;

/// `mlx_fast_metal_kernel_new` with the launch cache: the server's kernel of the text (one per name and text for the
/// process, launched through its pipeline once MLX has built it); mlx-c's own with no table kept.
export fn mlxlc_kernel_new(name: [*:0]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel {
    const t = table orelse return mlx.mlx_fast_metal_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    var ins: [max_names]common.Str = undefined;
    var outs: [max_names]common.Str = undefined;
    const n_in = names(input_names, &ins) orelse return .{};
    const n_out = names(output_names, &outs) orelse return .{};
    const src = std.mem.span(source);
    const hdr = std.mem.span(header);
    const desc: host.KernelDesc = .{
        .ensure_row_contiguous = ensure_row_contiguous,
        .atomic_outputs = atomic_outputs,
        .digest = kit.host.digest(src, hdr),
        .name = .of(std.mem.span(name)),
        .source = .of(src),
        .header = .of(hdr),
        .inputs = .of(ins[0..n_in]),
        .outputs = .of(outs[0..n_out]),
    };
    return .{ .ctx = t.kernel(t.ctx, &desc) };
}

fn names(v: mlx.mlx_vector_string, out: *[max_names]common.Str) ?usize {
    const n = mlx_vector_string_size(v);
    if (n > max_names) return null;
    for (out[0..n], 0..) |*o, i| {
        var p: ?[*:0]const u8 = null;
        if (mlx_vector_string_get(&p, v, i) != 0) return null;
        o.* = .of(std.mem.span(p orelse return null));
    }
    return n;
}

/// The server keeps its kernels for the process; mlx-c's own are freed.
export fn mlxlc_kernel_free(k: mlx.mlx_fast_metal_kernel) void {
    if (table == null) _ = mlx.mlx_fast_metal_kernel_free(k);
}

/// `mlx_fast_metal_kernel_apply`'s contract through the server's launcher.
export fn mlxlc_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int {
    const g = gpu() orelse return mlx.mlx_fast_metal_kernel_apply(outputs, k, inputs, config, stream);
    const v = g.launch(g.ctx, @ptrCast(outputs), k.ctx, .{ .ctx = inputs.ctx }, .{ .ctx = config.ctx }, .{ .ctx = stream.ctx });
    return if (v.isOk()) 0 else fail("launch", v);
}
