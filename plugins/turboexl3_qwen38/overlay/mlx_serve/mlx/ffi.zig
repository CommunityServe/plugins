//! This plugin's own, in place of mlx-serve's `src/mlx/ffi.zig` (a cut edge of scripts/sync-mlx-serve.sh): mlx-c
//! through the kit's `mlx_c`, which resolves against the server's libmlxc at load, so one MLX runs in the process and
//! its error handler is the server's. The handles are mlx-c's own types; `mlx_dtype` and `mlx_device_type` are
//! exhaustive Zig enums and the optionals carry `some`/`none`, the forms the copies call mlx-c with. The functions below
//! the generated marker come from the kit's headers (scripts/gen-mlx.py); the helpers above it are written here.

const std = @import("std");
const builtin = @import("builtin");
const kit = @import("mlx_serve");
pub const c = @import("mlx_c");

// ── handles ──

pub const mlx_array = c.mlx_array;
pub const mlx_stream = c.mlx_stream;
pub const mlx_device = c.mlx_device;
pub const mlx_device_info = c.mlx_device_info;
pub const mlx_vector_array = c.mlx_vector_array;
pub const mlx_vector_string = c.mlx_vector_string;
pub const mlx_map_string_to_array = c.mlx_map_string_to_array;
pub const mlx_map_string_to_array_iterator = c.mlx_map_string_to_array_iterator;
pub const mlx_map_string_to_string = c.mlx_map_string_to_string;
pub const mlx_closure = c.mlx_closure;
pub const mlx_fast_metal_kernel = c.mlx_fast_metal_kernel;
pub const mlx_fast_metal_kernel_config = c.mlx_fast_metal_kernel_config;
pub const mlx_io_reader = c.mlx_io_reader;

/// A custom reader's callbacks over its own descriptor, with the Zig types the copies' readers are written with
/// (mlx-c's `mlx_io_vtable`, field for field).
pub const mlx_io_vtable = extern struct {
    is_open: *const fn (?*anyopaque) callconv(.c) bool,
    good: *const fn (?*anyopaque) callconv(.c) bool,
    tell: *const fn (?*anyopaque) callconv(.c) usize,
    seek: *const fn (?*anyopaque, i64, c_int) callconv(.c) void,
    read: *const fn (?*anyopaque, [*]u8, usize) callconv(.c) void,
    read_at_offset: *const fn (?*anyopaque, [*]u8, usize, usize) callconv(.c) void,
    write: *const fn (?*anyopaque, [*]const u8, usize) callconv(.c) void,
    label: *const fn (?*anyopaque) callconv(.c) [*:0]const u8,
    free: *const fn (?*anyopaque) callconv(.c) void,
};

comptime {
    if (@sizeOf(mlx_io_vtable) != @sizeOf(c.mlx_io_vtable)) @compileError("mlx_io_vtable differs from mlx-c's");
}

pub fn mlx_io_reader_new(desc: ?*anyopaque, vtable: mlx_io_vtable) mlx_io_reader {
    const v: *const c.mlx_io_vtable = @ptrCast(&vtable);
    return c.mlx_io_reader_new(desc, v.*);
}

/// A closure over `fun(outputs, inputs, payload)` (`mlx_compile`'s input), with the Zig types the copies write it with;
/// `dtor(payload)` runs when MLX drops it.
pub fn mlx_closure_new_func_payload(fun: *const fn (*mlx_vector_array, mlx_vector_array, ?*anyopaque) callconv(.c) c_int, payload: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) mlx_closure {
    return c.mlx_closure_new_func_payload(@ptrCast(fun), payload, dtor);
}

/// The half-precision data accessors, which translate-c cannot declare (`bfloat16_t`): mlx-c's symbols with 16-bit
/// element types.
pub extern "c" fn mlx_array_data_float16(arr: mlx_array) ?[*]const f16;
pub extern "c" fn mlx_array_data_bfloat16(arr: mlx_array) ?[*]const u16;

// ── enums and optionals ──

pub const mlx_dtype = enum(c_int) {
    bool_ = c.MLX_BOOL,
    uint8 = c.MLX_UINT8,
    uint16 = c.MLX_UINT16,
    uint32 = c.MLX_UINT32,
    uint64 = c.MLX_UINT64,
    int8 = c.MLX_INT8,
    int16 = c.MLX_INT16,
    int32 = c.MLX_INT32,
    int64 = c.MLX_INT64,
    float16 = c.MLX_FLOAT16,
    float32 = c.MLX_FLOAT32,
    float64 = c.MLX_FLOAT64,
    bfloat16 = c.MLX_BFLOAT16,
    complex64 = c.MLX_COMPLEX64,
};

pub const mlx_device_type = enum(c_int) { cpu = c.MLX_CPU, gpu = c.MLX_GPU };

pub const mlx_optional_int = extern struct {
    value: c_int = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_int {
        return .{};
    }
    pub fn some(v: c_int) mlx_optional_int {
        return .{ .value = v, .has_value = true };
    }
};

pub const mlx_optional_float = extern struct {
    value: f32 = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_float {
        return .{};
    }
    pub fn some(v: f32) mlx_optional_float {
        return .{ .value = v, .has_value = true };
    }
};

pub const mlx_optional_dtype = extern struct {
    value: mlx_dtype = .float32,
    has_value: bool = false,

    pub fn none() mlx_optional_dtype {
        return .{};
    }
    pub fn some(v: mlx_dtype) mlx_optional_dtype {
        return .{ .value = v, .has_value = true };
    }
};

// ── the host ──

/// The server's host table, bound at load (`bind`): its MLX error latch backs the latch helpers below.
var host: ?kit.host.Host = null;

pub fn bind(h: kit.host.Host) void {
    host = h;
}

pub fn boundHost() ?kit.host.Host {
    return host;
}

// ── helpers the copies call ──

/// MLX calls checked, a relaxed count the profilers read.
pub var op_count: std.atomic.Value(u64) = .init(0);

pub fn check(ret: c_int) !void {
    _ = op_count.fetchAdd(1, .monotonic);
    if (ret != 0) return error.MlxError;
}

pub fn getShape(arr: mlx_array) []const c_int {
    const ndim = c.mlx_array_ndim(arr);
    if (ndim == 0) return &.{};
    return c.mlx_array_shape(arr)[0..ndim];
}

/// An outside plugin runs inside the macOS server, whose MLX has Metal.
pub fn noGpuBackend() bool {
    return false;
}

// Metal only. These are inline so their answers are known at compile time: a copy's CUDA path and its fallback for a
// server without Metal kernels are never analyzed, so they are not built and their CUDA kernels are not copied.

pub inline fn metalKernelsAvailable() bool {
    return comptime builtin.os.tag.isDarwin();
}

pub fn gpuStream() mlx_stream {
    return c.mlx_default_gpu_stream_new();
}

/// Weights load on the CPU stream: unified memory, no device pool to fill.
pub fn weightLoadStream() mlx_stream {
    return c.mlx_default_cpu_stream_new();
}

pub fn weightsInDeviceMemory() bool {
    return false;
}

pub fn streamOnGpuDevice(s: mlx_stream) bool {
    var dev = c.mlx_device_new();
    defer _ = c.mlx_device_free(dev);
    if (c.mlx_stream_get_device(&dev, s) != 0) return false;
    var t: c.mlx_device_type = c.MLX_CPU;
    if (c.mlx_device_get_type(&t, dev) != 0) return false;
    return t == c.MLX_GPU;
}

/// The stream runs custom Metal kernels.
pub fn streamIsGpu(s: mlx_stream) bool {
    return streamOnGpuDevice(s) and metalKernelsAvailable();
}

fn gpuInfoSize(key: [*:0]const u8) usize {
    const dev = c.mlx_device_new_type(c.MLX_GPU, 0);
    defer _ = c.mlx_device_free(dev);
    var info = c.mlx_device_info_new();
    defer _ = c.mlx_device_info_free(info);
    if (c.mlx_device_info_get(&info, dev) != 0) return 0;
    var v: usize = 0;
    if (c.mlx_device_info_get_size(&v, info, key) != 0) return 0;
    return v;
}

var arch_buf: [64]u8 = undefined;
var arch_len: ?usize = null;

/// The GPU's architecture name (`applegpu_g17s`), or empty when MLX names none.
pub fn gpuArchitectureName() []const u8 {
    if (arch_len) |n| return arch_buf[0..n];
    const dev = c.mlx_device_new_type(c.MLX_GPU, 0);
    defer _ = c.mlx_device_free(dev);
    var info = c.mlx_device_info_new();
    defer _ = c.mlx_device_info_free(info);
    var n: usize = 0;
    if (c.mlx_device_info_get(&info, dev) == 0) {
        var s: [*c]const u8 = null;
        if (c.mlx_device_info_get_string(&s, info, "architecture") == 0 and s != null) {
            const t = std.mem.span(s);
            n = @min(t.len, arch_buf.len);
            @memcpy(arch_buf[0..n], t[0..n]);
        }
    }
    arch_len = n;
    return arch_buf[0..n];
}

pub fn maxRecommendedWorkingSet() usize {
    return gpuInfoSize("max_recommended_working_set_size");
}

/// mlx-serve's residual dtype trace, a diagnostic armed by its environment: never armed here.
pub const DtypeTrace = struct {
    pub fn begin(path: []const u8, h: mlx_array, weight: ?mlx_array) DtypeTrace {
        _ = path;
        _ = h;
        _ = weight;
        return .{};
    }
    pub fn layer(self: *DtypeTrace, h: mlx_array, layer_idx: usize) void {
        _ = self;
        _ = h;
        _ = layer_idx;
    }
    pub fn end(self: *DtypeTrace, h: mlx_array) void {
        _ = self;
        _ = h;
    }
};

/// Every Metal kernel the copies make goes to mlx-c as written: no kernel-directory override here.
pub fn mlx_fast_metal_kernel_new(name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx_fast_metal_kernel {
    return c.mlx_fast_metal_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
}

pub fn mlx_fast_metal_kernel_free(k: mlx_fast_metal_kernel) c_int {
    c.mlx_fast_metal_kernel_free(k);
    return 0;
}

pub fn mlx_fast_metal_kernel_config_free(cfg: mlx_fast_metal_kernel_config) c_int {
    c.mlx_fast_metal_kernel_config_free(cfg);
    return 0;
}

// ── generated by scripts/gen-mlx.py from the kit's mlx-c headers: do not edit below ──

pub const mlx_array_new = c.mlx_array_new;
pub const mlx_array_free = c.mlx_array_free;
pub const mlx_array_new_bool = c.mlx_array_new_bool;
pub const mlx_array_new_int = c.mlx_array_new_int;
pub const mlx_array_new_float = c.mlx_array_new_float;
pub fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array {
    return c.mlx_array_new_data(data, shape, dim, @intCast(@intFromEnum(dtype)));
}
pub fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int {
    return c.mlx_array_set(arr, src);
}
pub const mlx_array_itemsize = c.mlx_array_itemsize;
pub const mlx_array_size = c.mlx_array_size;
pub const mlx_array_ndim = c.mlx_array_ndim;
pub const mlx_array_shape = c.mlx_array_shape;
pub const mlx_array_strides = c.mlx_array_strides;
pub fn mlx_array_dtype(arr: mlx_array) mlx_dtype {
    return @enumFromInt(c.mlx_array_dtype(arr));
}
pub const mlx_array_eval = c.mlx_array_eval;
pub fn mlx_array_item_bool(res: *bool, arr: mlx_array) c_int {
    return c.mlx_array_item_bool(res, arr);
}
pub fn mlx_array_item_int32(res: *i32, arr: mlx_array) c_int {
    return c.mlx_array_item_int32(res, arr);
}
pub fn mlx_array_data_uint8(arr: mlx_array) ?[*]const u8 {
    return c.mlx_array_data_uint8(arr);
}
pub fn mlx_array_data_uint32(arr: mlx_array) ?[*]const u32 {
    return c.mlx_array_data_uint32(arr);
}
pub fn mlx_array_data_int32(arr: mlx_array) ?[*]const i32 {
    return c.mlx_array_data_int32(arr);
}
pub fn mlx_array_data_float32(arr: mlx_array) ?[*]const f32 {
    return c.mlx_array_data_float32(arr);
}
pub fn _mlx_array_is_available(res: *bool, arr: mlx_array) c_int {
    return c._mlx_array_is_available(res, arr);
}
pub const mlx_closure_free = c.mlx_closure_free;
pub fn mlx_closure_apply(res: *mlx_vector_array, cls: mlx_closure, input: mlx_vector_array) c_int {
    return c.mlx_closure_apply(res, cls, input);
}
pub fn mlx_compile(res: *mlx_closure, fun: mlx_closure, shapeless: bool) c_int {
    return c.mlx_compile(res, fun, shapeless);
}
pub const mlx_device_new = c.mlx_device_new;
pub const mlx_device_free = c.mlx_device_free;
pub const mlx_device_info_new = c.mlx_device_info_new;
pub fn mlx_device_info_get(info: *mlx_device_info, dev: mlx_device) c_int {
    return c.mlx_device_info_get(info, dev);
}
pub const mlx_device_info_free = c.mlx_device_info_free;
pub const mlx_fast_metal_kernel_config_new = c.mlx_fast_metal_kernel_config_new;
pub fn mlx_fast_metal_kernel_config_add_output_arg(cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int {
    return c.mlx_fast_metal_kernel_config_add_output_arg(cls, shape, size, @intCast(@intFromEnum(dtype)));
}
pub const mlx_fast_metal_kernel_config_set_grid = c.mlx_fast_metal_kernel_config_set_grid;
pub const mlx_fast_metal_kernel_config_set_thread_group = c.mlx_fast_metal_kernel_config_set_thread_group;
pub fn mlx_fast_metal_kernel_config_add_template_arg_dtype(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) c_int {
    return c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cls, name, @intCast(@intFromEnum(dtype)));
}
pub const mlx_fast_metal_kernel_config_add_template_arg_int = c.mlx_fast_metal_kernel_config_add_template_arg_int;
pub const mlx_fast_metal_kernel_config_add_template_arg_bool = c.mlx_fast_metal_kernel_config_add_template_arg_bool;
pub fn mlx_fast_metal_kernel_apply(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, stream: mlx_stream) c_int {
    return c.mlx_fast_metal_kernel_apply(outputs, cls, inputs, config, stream);
}
pub fn mlx_fast_rms_norm(res: *mlx_array, x: mlx_array, weight: mlx_array, eps: f32, s: mlx_stream) c_int {
    return c.mlx_fast_rms_norm(res, x, weight, eps, s);
}
pub fn mlx_fast_rope(res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: c_int, freqs: mlx_array, s: mlx_stream) c_int {
    return c.mlx_fast_rope(res, x, dims, traditional, .{ .value = base.value, .has_value = base.has_value }, scale, offset, freqs, s);
}
pub fn mlx_fast_rope_dynamic(res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: mlx_array, freqs: mlx_array, s: mlx_stream) c_int {
    return c.mlx_fast_rope_dynamic(res, x, dims, traditional, .{ .value = base.value, .has_value = base.has_value }, scale, offset, freqs, s);
}
pub fn mlx_fast_scaled_dot_product_attention(res: *mlx_array, queries: mlx_array, keys: mlx_array, values: mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx_array, sinks: mlx_array, force_fused: bool, s: mlx_stream) c_int {
    return c.mlx_fast_scaled_dot_product_attention(res, queries, keys, values, scale, mask_mode, mask_arr, sinks, force_fused, s);
}
pub fn mlx_load_safetensors_reader(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, in_stream: mlx_io_reader, s: mlx_stream) c_int {
    return c.mlx_load_safetensors_reader(res_0, res_1, in_stream, s);
}
pub fn mlx_load_safetensors(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, file: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_load_safetensors(res_0, res_1, file, s);
}
pub const mlx_io_reader_free = c.mlx_io_reader_free;
pub const mlx_map_string_to_array_new = c.mlx_map_string_to_array_new;
pub const mlx_map_string_to_array_free = c.mlx_map_string_to_array_free;
pub const mlx_map_string_to_array_iterator_new = c.mlx_map_string_to_array_iterator_new;
pub const mlx_map_string_to_array_iterator_free = c.mlx_map_string_to_array_iterator_free;
pub fn mlx_map_string_to_array_iterator_next(key: anytype, value: *mlx_array, it: mlx_map_string_to_array_iterator) c_int {
    return c.mlx_map_string_to_array_iterator_next(@ptrCast(@constCast(key)), value, it);
}
pub const mlx_map_string_to_string_new = c.mlx_map_string_to_string_new;
pub const mlx_map_string_to_string_free = c.mlx_map_string_to_string_free;
pub const mlx_clear_cache = c.mlx_clear_cache;
pub fn mlx_get_active_memory(res: *usize) c_int {
    return c.mlx_get_active_memory(res);
}
pub fn mlx_abs(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_abs(res, a, s);
}
pub fn mlx_add(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_add(res, a, b, s);
}
pub fn mlx_all(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_all(res, a, keepdims, s);
}
pub fn mlx_any_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_any_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_arange(res: *mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_arange(res, start, stop, step, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_argmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_argmax_axis(res, a, axis, keepdims, s);
}
pub fn mlx_argpartition_axis(res: *mlx_array, a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_argpartition_axis(res, a, kth, axis, s);
}
pub fn mlx_argsort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_argsort_axis(res, a, axis, s);
}
pub fn mlx_astype(res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_astype(res, a, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_broadcast_to(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int {
    return c.mlx_broadcast_to(res, a, shape, shape_num, s);
}
pub fn mlx_concatenate_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_concatenate_axis(res, arrays, axis, s);
}
pub fn mlx_contiguous(res: *mlx_array, a: mlx_array, allow_col_major: bool, s: mlx_stream) c_int {
    return c.mlx_contiguous(res, a, allow_col_major, s);
}
pub fn mlx_conv1d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv1d(res, input, weight, stride, padding, dilation, groups, s);
}
pub fn mlx_copy(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_copy(res, a, s);
}
pub fn mlx_cos(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_cos(res, a, s);
}
pub fn mlx_dequantize(res: *mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, dtype: mlx_optional_dtype, s: mlx_stream) c_int {
    return c.mlx_dequantize(res, w, scales, biases, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, global_scale, .{ .value = @intCast(@intFromEnum(dtype.value)), .has_value = dtype.has_value }, s);
}
pub fn mlx_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_divide(res, a, b, s);
}
pub fn mlx_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_equal(res, a, b, s);
}
pub fn mlx_exp(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_exp(res, a, s);
}
pub fn mlx_expand_dims(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_expand_dims(res, a, axis, s);
}
pub fn mlx_floor_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_floor_divide(res, a, b, s);
}
pub fn mlx_full(res: *mlx_array, shape: [*]const c_int, shape_num: usize, vals: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_full(res, shape, shape_num, vals, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_greater(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_greater(res, a, b, s);
}
pub fn mlx_greater_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_greater_equal(res, a, b, s);
}
pub fn mlx_isfinite(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isfinite(res, a, s);
}
pub fn mlx_less(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_less(res, a, b, s);
}
pub fn mlx_less_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_less_equal(res, a, b, s);
}
pub fn mlx_log1p(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_log1p(res, a, s);
}
pub fn mlx_logical_and(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logical_and(res, a, b, s);
}
pub fn mlx_logical_or(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logical_or(res, a, b, s);
}
pub fn mlx_logsumexp_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_logsumexp_axis(res, a, axis, keepdims, s);
}
pub fn mlx_matmul(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_matmul(res, a, b, s);
}
pub fn mlx_maximum(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_maximum(res, a, b, s);
}
pub fn mlx_mean_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_mean_axis(res, a, axis, keepdims, s);
}
pub fn mlx_multiply(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_multiply(res, a, b, s);
}
pub fn mlx_negative(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_negative(res, a, s);
}
pub fn mlx_ones(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_ones(res, shape, shape_num, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_pad(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, low_pad_size: [*]const c_int, low_pad_size_num: usize, high_pad_size: [*]const c_int, high_pad_size_num: usize, pad_value: mlx_array, mode: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_pad(res, a, axes, axes_num, low_pad_size, low_pad_size_num, high_pad_size, high_pad_size_num, pad_value, mode, s);
}
pub fn mlx_put_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, values: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_put_along_axis(res, a, indices, values, axis, s);
}
pub fn mlx_quantize(res: *mlx_vector_array, w: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, s: mlx_stream) c_int {
    return c.mlx_quantize(res, w, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, global_scale, s);
}
pub fn mlx_quantized_matmul(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_quantized_matmul(res, x, w, scales, biases, transpose, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, s);
}
pub fn mlx_reshape(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int {
    return c.mlx_reshape(res, a, shape, shape_num, s);
}
pub fn mlx_sigmoid(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sigmoid(res, a, s);
}
pub fn mlx_sign(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sign(res, a, s);
}
pub fn mlx_sin(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sin(res, a, s);
}
pub fn mlx_slice(res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice(res, a, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_update(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_softmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, precise: bool, s: mlx_stream) c_int {
    return c.mlx_softmax_axis(res, a, axis, precise, s);
}
pub fn mlx_split(res: *mlx_vector_array, a: mlx_array, num_splits: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_split(res, a, num_splits, axis, s);
}
pub fn mlx_sqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sqrt(res, a, s);
}
pub fn mlx_subtract(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_subtract(res, a, b, s);
}
pub fn mlx_sum_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_sum_axis(res, a, axis, keepdims, s);
}
pub fn mlx_sum(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_sum(res, a, keepdims, s);
}
pub fn mlx_take_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_take_axis(res, a, indices, axis, s);
}
pub fn mlx_take_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_take_along_axis(res, a, indices, axis, s);
}
pub fn mlx_tile(res: *mlx_array, arr: mlx_array, reps: [*]const c_int, reps_num: usize, s: mlx_stream) c_int {
    return c.mlx_tile(res, arr, reps, reps_num, s);
}
pub fn mlx_transpose_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_transpose_axes(res, a, axes, axes_num, s);
}
pub fn mlx_transpose(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_transpose(res, a, s);
}
pub fn mlx_view(res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_view(res, a, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_where(res: *mlx_array, condition: mlx_array, x: mlx_array, y: mlx_array, s: mlx_stream) c_int {
    return c.mlx_where(res, condition, x, y, s);
}
pub fn mlx_zeros(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_zeros(res, shape, shape_num, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_random_normal(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: f32, scale: f32, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_normal(res, shape, shape_num, @intCast(@intFromEnum(dtype)), loc, scale, key, s);
}
pub const mlx_stream_new_device = c.mlx_stream_new_device;
pub const mlx_stream_free = c.mlx_stream_free;
pub const mlx_stream_equal = c.mlx_stream_equal;
pub fn mlx_stream_get_device(dev: *mlx_device, stream: mlx_stream) c_int {
    return c.mlx_stream_get_device(dev, stream);
}
pub const mlx_set_default_stream = c.mlx_set_default_stream;
pub const mlx_default_cpu_stream_new = c.mlx_default_cpu_stream_new;
pub const mlx_default_gpu_stream_new = c.mlx_default_gpu_stream_new;
pub const mlx_async_eval = c.mlx_async_eval;
pub const mlx_eval = c.mlx_eval;
pub const mlx_vector_array_new = c.mlx_vector_array_new;
pub const mlx_vector_array_free = c.mlx_vector_array_free;
pub fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array {
    return c.mlx_vector_array_new_data(data, size);
}
pub const mlx_vector_array_append_value = c.mlx_vector_array_append_value;
pub const mlx_vector_array_size = c.mlx_vector_array_size;
pub fn mlx_vector_array_get(res: *mlx_array, vec: mlx_vector_array, idx: usize) c_int {
    return c.mlx_vector_array_get(res, vec, idx);
}
pub const mlx_vector_string_free = c.mlx_vector_string_free;
pub fn mlx_vector_string_new_data(data: anytype, size: usize) mlx_vector_string {
    return c.mlx_vector_string_new_data(@ptrCast(@constCast(data)), size);
}

// Left to the hand-written part (function-pointer or untranslated parameters): _mlx_error, mlx_array_data_bfloat16, mlx_array_data_complex64, mlx_array_data_float16, mlx_array_item_bfloat16, mlx_array_item_complex64, mlx_array_item_float16, mlx_array_new_data_managed, mlx_closure_custom_jvp_new_func, mlx_closure_custom_jvp_new_func_payload, mlx_closure_custom_new_func, mlx_closure_custom_new_func_payload, mlx_closure_custom_vmap_new_func, mlx_closure_custom_vmap_new_func_payload, mlx_closure_kwargs_new_func, mlx_closure_kwargs_new_func_payload, mlx_closure_new_func, mlx_closure_new_unary, mlx_closure_value_and_grad_new_func, mlx_closure_value_and_grad_new_func_payload, mlx_detail_compile, mlx_detail_compile_erase, mlx_export_to_dot, mlx_print_graph, mlx_set_error_handler.
