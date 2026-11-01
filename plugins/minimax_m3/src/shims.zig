//! The C entry points the copies call that mlx-serve links from its MLX C++ and Objective-C shims (csrc/: the launch
//! cache, the uninitialised allocation, the NVFP4 gather, the read pool's Metal event). Inside this dylib they are
//! hidden: the kernel cache, its launch and the allocation go through the Host table the server hands every load (the
//! same implementation over MLX's internals mlx-serve's own plugins use), and the gather, which no table carries, is
//! written on the kit's mlx-c. With no table bound (the plugin's own GPU tests, no server) they fall back to mlx-c's
//! own kernel calls, which the table's launch is bit-identical to, and to zeros for the allocation.

const std = @import("std");
const kit = @import("mlx_serve");
const c = @import("mlx_c");
const abi = kit.abi.host;
const ffi = @import("mlx_serve/mlx/ffi.zig");
threadlocal var last_error: [*:0]const u8 = "";

/// Keeps the server's Host table for the calls below and the copies' log lines (`ffi.bind`, the edge's one store); a
/// load without one, or with one older than the kit's, is refused.
pub fn bind(raw: ?*const anyopaque) error{ NoHost, HostTooOld }!void {
    ffi.bind(try kit.host.Host.of(raw));
}

fn gpu() ?*const abi.HostGpu {
    return (ffi.boundHost() orelse return null).gpu();
}

fn alloc_last_error() callconv(.c) [*:0]const u8 {
    return last_error;
}

/// An evaluated array of `shape` over one buffer from MLX's allocator, its bytes unspecified until written.
fn alloc_uninit(out: *c.mlx_array, shape: [*]const c_int, ndim: usize, dtype: c.mlx_dtype) callconv(.c) c_int {
    const g = gpu() orelse {
        const s = c.mlx_default_gpu_stream_new();
        defer _ = c.mlx_stream_free(s);
        if (c.mlx_zeros(out, shape, ndim, dtype, s) == 0 and c.mlx_array_eval(out.*) == 0) return 0;
        last_error = "minimax_m3: mlx_zeros failed";
        return -1;
    };
    var a: abi.Array = .{ .ctx = out.ctx };
    const v = g.alloc_uninit(g.ctx, @ptrCast(shape), ndim, @intCast(dtype), &a);
    out.* = .{ .ctx = a.ctx };
    if (!v.isOk()) {
        last_error = "minimax_m3: the host refused an uninitialised allocation";
        return -1;
    }
    return 0;
}

/// The host's cached kernel for (name, source, header): made on the first ask and kept for the process.
fn kernel_new(name: [*c]const u8, input_names: c.mlx_vector_string, output_names: c.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) callconv(.c) c.mlx_fast_metal_kernel {
    const none: c.mlx_fast_metal_kernel = .{ .ctx = null };
    const h = ffi.boundHost() orelse return c.mlx_fast_metal_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
    var names: [2][64]kit.abi.common.Str = undefined;
    const ins = strings(input_names, &names[0]) orelse return none;
    const outs = strings(output_names, &names[1]) orelse return none;
    const src = std.mem.span(source);
    const hdr: []const u8 = if (header) |p| std.mem.span(p) else "";
    var digest = std.hash.Wyhash.init(0);
    digest.update(src);
    digest.update(&.{0});
    digest.update(hdr);
    const desc: abi.KernelDesc = .{
        .ensure_row_contiguous = ensure_row_contiguous,
        .atomic_outputs = atomic_outputs,
        .digest = digest.final(),
        .name = .of(std.mem.span(name)),
        .source = .of(src),
        .header = .of(hdr),
        .inputs = .{ .ptr = ins.ptr, .len = ins.len },
        .outputs = .{ .ptr = outs.ptr, .len = outs.len },
    };
    return .{ .ctx = h.t.kernel(h.t.ctx, &desc) };
}

fn strings(v: c.mlx_vector_string, out: *[64]kit.abi.common.Str) ?[]const kit.abi.common.Str {
    const n = c.mlx_vector_string_size(v);
    if (n > out.len) return null;
    for (out[0..n], 0..) |*s, i| {
        var p: [*c]u8 = null;
        if (c.mlx_vector_string_get(&p, v, i) != 0 or p == null) return null;
        s.* = .of(std.mem.span(p));
    }
    return out[0..n];
}

/// The host's cache owns every kernel it made; one mlx-c made (no table bound) is freed.
fn kernel_free(k: c.mlx_fast_metal_kernel) callconv(.c) void {
    if (ffi.boundHost() == null) c.mlx_fast_metal_kernel_free(k);
}

/// `mlx_fast_metal_kernel_apply` through the host's launch, bit-identical to it.
fn apply(outputs: *c.mlx_vector_array, k: c.mlx_fast_metal_kernel, inputs: c.mlx_vector_array, config: c.mlx_fast_metal_kernel_config, stream: c.mlx_stream) callconv(.c) c_int {
    const g = gpu() orelse return c.mlx_fast_metal_kernel_apply(outputs, k, inputs, config, stream);
    var o: abi.VectorArray = .{ .ctx = outputs.ctx };
    const v = g.launch(g.ctx, &o, k.ctx, .{ .ctx = inputs.ctx }, .{ .ctx = config.ctx }, .{ .ctx = stream.ctx });
    outputs.* = .{ .ctx = o.ctx };
    return if (v.isOk()) 0 else -1;
}

/// MLX's NVFP4 `gather_qmm` with a per-expert global scale, which mlx-c does not expose: the quantized products in f32
/// (rows of one row each, as every caller passes them), then `* (global_scale / (448 * 6))` and one cast to x's dtype,
/// the order MLX's fp_qmv kernels apply it in.
fn gather_qmm_nvfp4(res: *c.mlx_array, x: c.mlx_array, w: c.mlx_array, scales: c.mlx_array, global_scale: c.mlx_array, rhs_indices: c.mlx_array, sorted: bool, s: c.mlx_stream) callconv(.c) c_int {
    var t: [6]c.mlx_array = undefined;
    for (&t) |*a| a.* = c.mlx_array_new();
    defer for (t) |a| {
        _ = c.mlx_array_free(a);
    };
    const enc = c.mlx_array_new_float(448.0 * 6.0);
    defer _ = c.mlx_array_free(enc);
    const none = c.mlx_array{ .ctx = null };
    const ok = c.mlx_astype(&t[0], x, c.MLX_FLOAT32, s) == 0 and
        c.mlx_gather_qmm(&t[1], t[0], w, scales, none, none, rhs_indices, true, .{ .value = 16, .has_value = true }, .{ .value = 4, .has_value = true }, "nvfp4", sorted, s) == 0 and
        c.mlx_take_axis(&t[2], global_scale, rhs_indices, 0, s) == 0 and
        c.mlx_divide(&t[3], t[2], enc, s) == 0 and
        reshapeRows(&t[4], t[3], s) and
        c.mlx_multiply(&t[5], t[1], t[4], s) == 0 and
        c.mlx_astype(res, t[5], c.mlx_array_dtype(x), s) == 0;
    if (!ok) {
        last_error = "minimax_m3: NVFP4 gather failed";
        return -1;
    }
    return 0;
}

/// `[rows]` to `[rows, 1, 1]`, one scale a gathered row.
fn reshapeRows(out: *c.mlx_array, v: c.mlx_array, s: c.mlx_stream) bool {
    const shape = [_]c_int{ @intCast(c.mlx_array_size(v)), 1, 1 };
    return c.mlx_reshape(out, v, &shape, shape.len, s) == 0;
}

/// The read pool signals a Metal event gate through this; MiniMax registers no event gate, so it is never reached.
fn mtl_signal(_: ?*anyopaque, _: u64) callconv(.c) void {
    @panic("minimax_m3: the read pool signalled a Metal event gate, which this plugin never registers");
}

comptime {
    @export(&mtl_signal, .{ .name = "q3ld_mtl_signal", .visibility = .hidden });
    @export(&alloc_last_error, .{ .name = "dsv41_alloc_last_error", .visibility = .hidden });
    @export(&alloc_uninit, .{ .name = "dsv41_alloc_uninit", .visibility = .hidden });
    @export(&gather_qmm_nvfp4, .{ .name = "glm5_gather_qmm_nvfp4", .visibility = .hidden });
    @export(&kernel_new, .{ .name = "mlxlc_kernel_new", .visibility = .hidden });
    @export(&kernel_free, .{ .name = "mlxlc_kernel_free", .visibility = .hidden });
    @export(&apply, .{ .name = "mlxlc_apply", .visibility = .hidden });
}
