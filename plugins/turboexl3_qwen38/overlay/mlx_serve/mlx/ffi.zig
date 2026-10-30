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
pub const mlx_string = c.mlx_string;
pub const mlx_vector_array = c.mlx_vector_array;
pub const mlx_vector_int = c.mlx_vector_int;
pub const mlx_vector_string = c.mlx_vector_string;
pub const mlx_vector_vector_array = c.mlx_vector_vector_array;
pub const mlx_map_string_to_array = c.mlx_map_string_to_array;
pub const mlx_map_string_to_array_iterator = c.mlx_map_string_to_array_iterator;
pub const mlx_map_string_to_string = c.mlx_map_string_to_string;
pub const mlx_map_string_to_string_iterator = c.mlx_map_string_to_string_iterator;
pub const mlx_closure = c.mlx_closure;
pub const mlx_fast_metal_kernel = c.mlx_fast_metal_kernel;
pub const mlx_fast_metal_kernel_config = c.mlx_fast_metal_kernel_config;
pub const mlx_fast_cuda_kernel = c.mlx_fast_cuda_kernel;
pub const mlx_fast_cuda_kernel_config = c.mlx_fast_cuda_kernel_config;
pub const mlx_io_reader = c.mlx_io_reader;
pub const mlx_io_writer = c.mlx_io_writer;
pub const mlx_distributed_group = c.mlx_distributed_group;

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

/// An array over the caller's buffer, which `dtor(payload)` releases when MLX is done with it.
pub fn mlx_array_new_data_managed_payload(data: *anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype, payload: ?*anyopaque, dtor: *const fn (?*anyopaque) callconv(.c) void) mlx_array {
    return c.mlx_array_new_data_managed_payload(data, shape, dim, @intCast(@intFromEnum(dtype)), payload, dtor);
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

pub fn metalKernelsAvailable() bool {
    return comptime builtin.os.tag.isDarwin();
}

pub fn cudaAvailable() bool {
    return false;
}

pub fn cudaIntegrated() bool {
    return false;
}

pub fn cudaComputeCapability(part: enum { major, minor }) usize {
    _ = part;
    return 0;
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

/// A virtual Metal GPU (a macOS VM) names no Apple GPU family.
pub fn virtualGpu() bool {
    return !std.mem.startsWith(u8, gpuArchitectureName(), "applegpu_g");
}

pub fn maxRecommendedWorkingSet() usize {
    return gpuInfoSize("max_recommended_working_set_size");
}

pub fn maxBufferLength() usize {
    return gpuInfoSize("max_buffer_length");
}

pub fn freeDeviceMemory() usize {
    return 0;
}

/// The server installed its handler before any plugin loaded.
pub fn installErrorHandler() void {}

pub fn errorPending() bool {
    return if (host) |h| h.latchPending() else false;
}

pub fn dropLatchedErrorUnless(had_error: bool) void {
    if (host) |h| h.dropLatchedErrorUnless(had_error);
}

/// Consumes the latched error, as a path that recovers from it does. The host's latch hands out no message text, so the
/// message is a placeholder.
pub fn takeError(buf: []u8) ?[]const u8 {
    _ = buf;
    const h = host orelse return null;
    if (!h.latchPending()) return null;
    h.dropLatchedErrorUnless(false);
    return "an MLX error (the server's latch holds its text)";
}

/// Consumes the latched error only when its message holds `needle`; the host's latch hands out no message, so no error
/// matches and it stays latched: the recovery the caller would make is not made, and the request fails by name.
pub fn takeErrorIf(needle: []const u8) bool {
    _ = needle;
    return false;
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

pub const mlx_closure_custom = c.mlx_closure_custom;
pub const mlx_closure_custom_jvp = c.mlx_closure_custom_jvp;
pub const mlx_closure_custom_vmap = c.mlx_closure_custom_vmap;
pub const mlx_closure_kwargs = c.mlx_closure_kwargs;
pub const mlx_closure_value_and_grad = c.mlx_closure_value_and_grad;
pub const mlx_function_exporter = c.mlx_function_exporter;
pub const mlx_imported_function = c.mlx_imported_function;
pub const mlx_io_gguf = c.mlx_io_gguf;
pub const mlx_node_namer = c.mlx_node_namer;
pub fn mlx_dtype_size(dtype: mlx_dtype) usize {
    return c.mlx_dtype_size(@intCast(@intFromEnum(dtype)));
}
pub fn mlx_array_tostring(str: *mlx_string, arr: mlx_array) c_int {
    return c.mlx_array_tostring(str, arr);
}
pub const mlx_array_new = c.mlx_array_new;
pub const mlx_array_free = c.mlx_array_free;
pub const mlx_array_new_bool = c.mlx_array_new_bool;
pub const mlx_array_new_int = c.mlx_array_new_int;
pub const mlx_array_new_float32 = c.mlx_array_new_float32;
pub const mlx_array_new_float = c.mlx_array_new_float;
pub const mlx_array_new_float64 = c.mlx_array_new_float64;
pub const mlx_array_new_double = c.mlx_array_new_double;
pub const mlx_array_new_complex = c.mlx_array_new_complex;
pub fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array {
    return c.mlx_array_new_data(data, shape, dim, @intCast(@intFromEnum(dtype)));
}
pub fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int {
    return c.mlx_array_set(arr, src);
}
pub fn mlx_array_set_bool(arr: *mlx_array, val: bool) c_int {
    return c.mlx_array_set_bool(arr, val);
}
pub fn mlx_array_set_int(arr: *mlx_array, val: c_int) c_int {
    return c.mlx_array_set_int(arr, val);
}
pub fn mlx_array_set_float32(arr: *mlx_array, val: f32) c_int {
    return c.mlx_array_set_float32(arr, val);
}
pub fn mlx_array_set_float(arr: *mlx_array, val: f32) c_int {
    return c.mlx_array_set_float(arr, val);
}
pub fn mlx_array_set_float64(arr: *mlx_array, val: f64) c_int {
    return c.mlx_array_set_float64(arr, val);
}
pub fn mlx_array_set_double(arr: *mlx_array, val: f64) c_int {
    return c.mlx_array_set_double(arr, val);
}
pub fn mlx_array_set_complex(arr: *mlx_array, real_val: f32, imag_val: f32) c_int {
    return c.mlx_array_set_complex(arr, real_val, imag_val);
}
pub fn mlx_array_set_data(arr: *mlx_array, data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) c_int {
    return c.mlx_array_set_data(arr, data, shape, dim, @intCast(@intFromEnum(dtype)));
}
pub const mlx_array_itemsize = c.mlx_array_itemsize;
pub const mlx_array_size = c.mlx_array_size;
pub const mlx_array_nbytes = c.mlx_array_nbytes;
pub const mlx_array_ndim = c.mlx_array_ndim;
pub const mlx_array_shape = c.mlx_array_shape;
pub const mlx_array_strides = c.mlx_array_strides;
pub const mlx_array_dim = c.mlx_array_dim;
pub fn mlx_array_dtype(arr: mlx_array) mlx_dtype {
    return @enumFromInt(c.mlx_array_dtype(arr));
}
pub const mlx_array_eval = c.mlx_array_eval;
pub fn mlx_array_item_bool(res: *bool, arr: mlx_array) c_int {
    return c.mlx_array_item_bool(res, arr);
}
pub fn mlx_array_item_uint8(res: *u8, arr: mlx_array) c_int {
    return c.mlx_array_item_uint8(res, arr);
}
pub fn mlx_array_item_uint16(res: *u16, arr: mlx_array) c_int {
    return c.mlx_array_item_uint16(res, arr);
}
pub fn mlx_array_item_uint32(res: *u32, arr: mlx_array) c_int {
    return c.mlx_array_item_uint32(res, arr);
}
pub fn mlx_array_item_uint64(res: *u64, arr: mlx_array) c_int {
    return c.mlx_array_item_uint64(res, arr);
}
pub fn mlx_array_item_int8(res: *i8, arr: mlx_array) c_int {
    return c.mlx_array_item_int8(res, arr);
}
pub fn mlx_array_item_int16(res: *i16, arr: mlx_array) c_int {
    return c.mlx_array_item_int16(res, arr);
}
pub fn mlx_array_item_int32(res: *i32, arr: mlx_array) c_int {
    return c.mlx_array_item_int32(res, arr);
}
pub fn mlx_array_item_int64(res: *i64, arr: mlx_array) c_int {
    return c.mlx_array_item_int64(res, arr);
}
pub fn mlx_array_item_float32(res: *f32, arr: mlx_array) c_int {
    return c.mlx_array_item_float32(res, arr);
}
pub fn mlx_array_item_float64(res: *f64, arr: mlx_array) c_int {
    return c.mlx_array_item_float64(res, arr);
}
pub fn mlx_array_data_bool(arr: mlx_array) ?[*]const bool {
    return c.mlx_array_data_bool(arr);
}
pub fn mlx_array_data_uint8(arr: mlx_array) ?[*]const u8 {
    return c.mlx_array_data_uint8(arr);
}
pub fn mlx_array_data_uint16(arr: mlx_array) ?[*]const u16 {
    return c.mlx_array_data_uint16(arr);
}
pub fn mlx_array_data_uint32(arr: mlx_array) ?[*]const u32 {
    return c.mlx_array_data_uint32(arr);
}
pub fn mlx_array_data_uint64(arr: mlx_array) ?[*]const u64 {
    return c.mlx_array_data_uint64(arr);
}
pub fn mlx_array_data_int8(arr: mlx_array) ?[*]const i8 {
    return c.mlx_array_data_int8(arr);
}
pub fn mlx_array_data_int16(arr: mlx_array) ?[*]const i16 {
    return c.mlx_array_data_int16(arr);
}
pub fn mlx_array_data_int32(arr: mlx_array) ?[*]const i32 {
    return c.mlx_array_data_int32(arr);
}
pub fn mlx_array_data_int64(arr: mlx_array) ?[*]const i64 {
    return c.mlx_array_data_int64(arr);
}
pub fn mlx_array_data_float32(arr: mlx_array) ?[*]const f32 {
    return c.mlx_array_data_float32(arr);
}
pub fn mlx_array_data_float64(arr: mlx_array) ?[*]const f64 {
    return c.mlx_array_data_float64(arr);
}
pub fn _mlx_array_is_available(res: *bool, arr: mlx_array) c_int {
    return c._mlx_array_is_available(res, arr);
}
pub const _mlx_array_wait = c._mlx_array_wait;
pub fn _mlx_array_is_contiguous(res: *bool, arr: mlx_array) c_int {
    return c._mlx_array_is_contiguous(res, arr);
}
pub fn _mlx_array_is_row_contiguous(res: *bool, arr: mlx_array) c_int {
    return c._mlx_array_is_row_contiguous(res, arr);
}
pub fn _mlx_array_is_col_contiguous(res: *bool, arr: mlx_array) c_int {
    return c._mlx_array_is_col_contiguous(res, arr);
}
pub const mlx_closure_new = c.mlx_closure_new;
pub const mlx_closure_free = c.mlx_closure_free;
pub fn mlx_closure_set(cls: *mlx_closure, src: mlx_closure) c_int {
    return c.mlx_closure_set(cls, src);
}
pub fn mlx_closure_apply(res: *mlx_vector_array, cls: mlx_closure, input: mlx_vector_array) c_int {
    return c.mlx_closure_apply(res, cls, input);
}
pub const mlx_closure_kwargs_new = c.mlx_closure_kwargs_new;
pub const mlx_closure_kwargs_free = c.mlx_closure_kwargs_free;
pub fn mlx_closure_kwargs_set(cls: *mlx_closure_kwargs, src: mlx_closure_kwargs) c_int {
    return c.mlx_closure_kwargs_set(cls, src);
}
pub fn mlx_closure_kwargs_apply(res: *mlx_vector_array, cls: mlx_closure_kwargs, input_0: mlx_vector_array, input_1: mlx_map_string_to_array) c_int {
    return c.mlx_closure_kwargs_apply(res, cls, input_0, input_1);
}
pub const mlx_closure_value_and_grad_new = c.mlx_closure_value_and_grad_new;
pub const mlx_closure_value_and_grad_free = c.mlx_closure_value_and_grad_free;
pub fn mlx_closure_value_and_grad_set(cls: *mlx_closure_value_and_grad, src: mlx_closure_value_and_grad) c_int {
    return c.mlx_closure_value_and_grad_set(cls, src);
}
pub fn mlx_closure_value_and_grad_apply(res_0: *mlx_vector_array, res_1: *mlx_vector_array, cls: mlx_closure_value_and_grad, input: mlx_vector_array) c_int {
    return c.mlx_closure_value_and_grad_apply(res_0, res_1, cls, input);
}
pub const mlx_closure_custom_new = c.mlx_closure_custom_new;
pub const mlx_closure_custom_free = c.mlx_closure_custom_free;
pub fn mlx_closure_custom_set(cls: *mlx_closure_custom, src: mlx_closure_custom) c_int {
    return c.mlx_closure_custom_set(cls, src);
}
pub fn mlx_closure_custom_apply(res: *mlx_vector_array, cls: mlx_closure_custom, input_0: mlx_vector_array, input_1: mlx_vector_array, input_2: mlx_vector_array) c_int {
    return c.mlx_closure_custom_apply(res, cls, input_0, input_1, input_2);
}
pub const mlx_closure_custom_jvp_new = c.mlx_closure_custom_jvp_new;
pub const mlx_closure_custom_jvp_free = c.mlx_closure_custom_jvp_free;
pub fn mlx_closure_custom_jvp_set(cls: *mlx_closure_custom_jvp, src: mlx_closure_custom_jvp) c_int {
    return c.mlx_closure_custom_jvp_set(cls, src);
}
pub fn mlx_closure_custom_jvp_apply(res: *mlx_vector_array, cls: mlx_closure_custom_jvp, input_0: mlx_vector_array, input_1: mlx_vector_array, input_2: [*]const c_int, input_2_num: usize) c_int {
    return c.mlx_closure_custom_jvp_apply(res, cls, input_0, input_1, input_2, input_2_num);
}
pub const mlx_closure_custom_vmap_new = c.mlx_closure_custom_vmap_new;
pub const mlx_closure_custom_vmap_free = c.mlx_closure_custom_vmap_free;
pub fn mlx_closure_custom_vmap_set(cls: *mlx_closure_custom_vmap, src: mlx_closure_custom_vmap) c_int {
    return c.mlx_closure_custom_vmap_set(cls, src);
}
pub fn mlx_closure_custom_vmap_apply(res_0: *mlx_vector_array, res_1: *mlx_vector_int, cls: mlx_closure_custom_vmap, input_0: mlx_vector_array, input_1: [*]const c_int, input_1_num: usize) c_int {
    return c.mlx_closure_custom_vmap_apply(res_0, res_1, cls, input_0, input_1, input_1_num);
}
pub fn mlx_compile(res: *mlx_closure, fun: mlx_closure, shapeless: bool) c_int {
    return c.mlx_compile(res, fun, shapeless);
}
pub const mlx_detail_compile_clear_cache = c.mlx_detail_compile_clear_cache;
pub const mlx_disable_compile = c.mlx_disable_compile;
pub const mlx_enable_compile = c.mlx_enable_compile;
pub const mlx_set_compile_mode = c.mlx_set_compile_mode;
pub fn mlx_cuda_is_available(res: *bool) c_int {
    return c.mlx_cuda_is_available(res);
}
pub const mlx_device_new = c.mlx_device_new;
pub fn mlx_device_new_type(@"type": mlx_device_type, index: c_int) mlx_device {
    return c.mlx_device_new_type(@intCast(@intFromEnum(@"type")), index);
}
pub const mlx_device_free = c.mlx_device_free;
pub fn mlx_device_set(dev: *mlx_device, src: mlx_device) c_int {
    return c.mlx_device_set(dev, src);
}
pub fn mlx_device_tostring(str: *mlx_string, dev: mlx_device) c_int {
    return c.mlx_device_tostring(str, dev);
}
pub const mlx_device_equal = c.mlx_device_equal;
pub fn mlx_device_get_index(index: *c_int, dev: mlx_device) c_int {
    return c.mlx_device_get_index(index, dev);
}
pub fn mlx_device_get_type(@"type": *mlx_device_type, dev: mlx_device) c_int {
    return c.mlx_device_get_type(@ptrCast(@"type"), dev);
}
pub fn mlx_get_default_device(dev: *mlx_device) c_int {
    return c.mlx_get_default_device(dev);
}
pub const mlx_set_default_device = c.mlx_set_default_device;
pub fn mlx_device_is_available(avail: *bool, dev: mlx_device) c_int {
    return c.mlx_device_is_available(avail, dev);
}
pub fn mlx_device_count(count: *c_int, @"type": mlx_device_type) c_int {
    return c.mlx_device_count(count, @intCast(@intFromEnum(@"type")));
}
pub const mlx_device_info_new = c.mlx_device_info_new;
pub fn mlx_device_info_get(info: *mlx_device_info, dev: mlx_device) c_int {
    return c.mlx_device_info_get(info, dev);
}
pub const mlx_device_info_free = c.mlx_device_info_free;
pub fn mlx_device_info_has_key(exists: *bool, info: mlx_device_info, key: [*:0]const u8) c_int {
    return c.mlx_device_info_has_key(exists, info, key);
}
pub fn mlx_device_info_is_string(is_string: *bool, info: mlx_device_info, key: [*:0]const u8) c_int {
    return c.mlx_device_info_is_string(is_string, info, key);
}
pub fn mlx_device_info_get_string(value: anytype, info: mlx_device_info, key: [*:0]const u8) c_int {
    return c.mlx_device_info_get_string(@ptrCast(@constCast(value)), info, key);
}
pub fn mlx_device_info_get_size(value: *usize, info: mlx_device_info, key: [*:0]const u8) c_int {
    return c.mlx_device_info_get_size(value, info, key);
}
pub fn mlx_device_info_get_keys(keys: *mlx_vector_string, info: mlx_device_info) c_int {
    return c.mlx_device_info_get_keys(keys, info);
}
pub fn mlx_distributed_all_gather(res: *mlx_array, x: mlx_array, group: mlx_distributed_group, S: mlx_stream) c_int {
    return c.mlx_distributed_all_gather(res, x, group, S);
}
pub fn mlx_distributed_all_max(res: *mlx_array, x: mlx_array, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_all_max(res, x, group, s);
}
pub fn mlx_distributed_all_min(res: *mlx_array, x: mlx_array, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_all_min(res, x, group, s);
}
pub fn mlx_distributed_all_sum(res: *mlx_array, x: mlx_array, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_all_sum(res, x, group, s);
}
pub fn mlx_distributed_recv(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, src: c_int, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_recv(res, shape, shape_num, @intCast(@intFromEnum(dtype)), src, group, s);
}
pub fn mlx_distributed_recv_like(res: *mlx_array, x: mlx_array, src: c_int, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_recv_like(res, x, src, group, s);
}
pub fn mlx_distributed_send(res: *mlx_array, x: mlx_array, dst: c_int, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_send(res, x, dst, group, s);
}
pub fn mlx_distributed_sum_scatter(res: *mlx_array, x: mlx_array, group: mlx_distributed_group, s: mlx_stream) c_int {
    return c.mlx_distributed_sum_scatter(res, x, group, s);
}
pub const mlx_distributed_group_new = c.mlx_distributed_group_new;
pub const mlx_distributed_group_free = c.mlx_distributed_group_free;
pub fn mlx_distributed_init(res: *mlx_distributed_group, strict: bool, bk: [*:0]const u8) c_int {
    return c.mlx_distributed_init(res, strict, bk);
}
pub const mlx_distributed_group_rank = c.mlx_distributed_group_rank;
pub const mlx_distributed_group_size = c.mlx_distributed_group_size;
pub fn mlx_distributed_group_split(res: *mlx_distributed_group, group: mlx_distributed_group, color: c_int, key: c_int) c_int {
    return c.mlx_distributed_group_split(res, group, color, key);
}
pub const mlx_distributed_is_available = c.mlx_distributed_is_available;
pub const mlx_export_function = c.mlx_export_function;
pub const mlx_export_function_kwargs = c.mlx_export_function_kwargs;
pub const mlx_function_exporter_new = c.mlx_function_exporter_new;
pub const mlx_function_exporter_free = c.mlx_function_exporter_free;
pub const mlx_function_exporter_apply = c.mlx_function_exporter_apply;
pub const mlx_function_exporter_apply_kwargs = c.mlx_function_exporter_apply_kwargs;
pub const mlx_imported_function_new = c.mlx_imported_function_new;
pub const mlx_imported_function_free = c.mlx_imported_function_free;
pub fn mlx_imported_function_apply(res: *mlx_vector_array, xfunc: mlx_imported_function, args: mlx_vector_array) c_int {
    return c.mlx_imported_function_apply(res, xfunc, args);
}
pub fn mlx_imported_function_apply_kwargs(res: *mlx_vector_array, xfunc: mlx_imported_function, args: mlx_vector_array, kwargs: mlx_map_string_to_array) c_int {
    return c.mlx_imported_function_apply_kwargs(res, xfunc, args, kwargs);
}
pub const mlx_fast_cuda_kernel_config_new = c.mlx_fast_cuda_kernel_config_new;
pub const mlx_fast_cuda_kernel_config_free = c.mlx_fast_cuda_kernel_config_free;
pub fn mlx_fast_cuda_kernel_config_add_output_arg(cls: mlx_fast_cuda_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int {
    return c.mlx_fast_cuda_kernel_config_add_output_arg(cls, shape, size, @intCast(@intFromEnum(dtype)));
}
pub const mlx_fast_cuda_kernel_config_set_grid = c.mlx_fast_cuda_kernel_config_set_grid;
pub const mlx_fast_cuda_kernel_config_set_thread_group = c.mlx_fast_cuda_kernel_config_set_thread_group;
pub const mlx_fast_cuda_kernel_config_set_init_value = c.mlx_fast_cuda_kernel_config_set_init_value;
pub const mlx_fast_cuda_kernel_config_set_verbose = c.mlx_fast_cuda_kernel_config_set_verbose;
pub fn mlx_fast_cuda_kernel_config_add_template_arg_dtype(cls: mlx_fast_cuda_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) c_int {
    return c.mlx_fast_cuda_kernel_config_add_template_arg_dtype(cls, name, @intCast(@intFromEnum(dtype)));
}
pub const mlx_fast_cuda_kernel_config_add_template_arg_int = c.mlx_fast_cuda_kernel_config_add_template_arg_int;
pub const mlx_fast_cuda_kernel_config_add_template_arg_bool = c.mlx_fast_cuda_kernel_config_add_template_arg_bool;
pub const mlx_fast_cuda_kernel_new = c.mlx_fast_cuda_kernel_new;
pub const mlx_fast_cuda_kernel_free = c.mlx_fast_cuda_kernel_free;
pub fn mlx_fast_cuda_kernel_apply(outputs: *mlx_vector_array, cls: mlx_fast_cuda_kernel, inputs: mlx_vector_array, config: mlx_fast_cuda_kernel_config, stream: mlx_stream) c_int {
    return c.mlx_fast_cuda_kernel_apply(outputs, cls, inputs, config, stream);
}
pub fn mlx_fast_layer_norm(res: *mlx_array, x: mlx_array, weight: mlx_array, bias: mlx_array, eps: f32, s: mlx_stream) c_int {
    return c.mlx_fast_layer_norm(res, x, weight, bias, eps, s);
}
pub const mlx_fast_metal_kernel_config_new = c.mlx_fast_metal_kernel_config_new;
pub fn mlx_fast_metal_kernel_config_add_output_arg(cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int {
    return c.mlx_fast_metal_kernel_config_add_output_arg(cls, shape, size, @intCast(@intFromEnum(dtype)));
}
pub const mlx_fast_metal_kernel_config_set_grid = c.mlx_fast_metal_kernel_config_set_grid;
pub const mlx_fast_metal_kernel_config_set_thread_group = c.mlx_fast_metal_kernel_config_set_thread_group;
pub const mlx_fast_metal_kernel_config_set_init_value = c.mlx_fast_metal_kernel_config_set_init_value;
pub const mlx_fast_metal_kernel_config_set_verbose = c.mlx_fast_metal_kernel_config_set_verbose;
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
pub fn mlx_fft_fft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_fft(res, a, n, axis, norm, s);
}
pub fn mlx_fft_fft2(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_fft2(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_fftfreq(res: *mlx_array, n: c_int, d: f64, s: mlx_stream) c_int {
    return c.mlx_fft_fftfreq(res, n, d, s);
}
pub fn mlx_fft_fftn(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_fftn(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_fftshift(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_fft_fftshift(res, a, axes, axes_num, s);
}
pub fn mlx_fft_ifft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_ifft(res, a, n, axis, norm, s);
}
pub fn mlx_fft_ifft2(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_ifft2(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_ifftn(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_ifftn(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_ifftshift(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_fft_ifftshift(res, a, axes, axes_num, s);
}
pub fn mlx_fft_irfft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_irfft(res, a, n, axis, norm, s);
}
pub fn mlx_fft_irfft2(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_irfft2(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_irfftn(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_irfftn(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_rfft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_rfft(res, a, n, axis, norm, s);
}
pub fn mlx_fft_rfft2(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_rfft2(res, a, n, n_num, axes, axes_num, norm, s);
}
pub fn mlx_fft_rfftfreq(res: *mlx_array, n: c_int, d: f64, s: mlx_stream) c_int {
    return c.mlx_fft_rfftfreq(res, n, d, s);
}
pub fn mlx_fft_rfftn(res: *mlx_array, a: mlx_array, n: [*]const c_int, n_num: usize, axes: [*]const c_int, axes_num: usize, norm: c.mlx_fft_norm, s: mlx_stream) c_int {
    return c.mlx_fft_rfftn(res, a, n, n_num, axes, axes_num, norm, s);
}
pub const mlx_node_namer_new = c.mlx_node_namer_new;
pub const mlx_node_namer_free = c.mlx_node_namer_free;
pub const mlx_node_namer_set_name = c.mlx_node_namer_set_name;
pub fn mlx_node_namer_get_name(name: anytype, namer: mlx_node_namer, arr: mlx_array) c_int {
    return c.mlx_node_namer_get_name(@ptrCast(@constCast(name)), namer, arr);
}
pub fn mlx_load_reader(res: *mlx_array, in_stream: mlx_io_reader, s: mlx_stream) c_int {
    return c.mlx_load_reader(res, in_stream, s);
}
pub fn mlx_load(res: *mlx_array, file: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_load(res, file, s);
}
pub fn mlx_load_gguf(gguf: *mlx_io_gguf, file: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_load_gguf(gguf, file, s);
}
pub fn mlx_load_safetensors_reader(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, in_stream: mlx_io_reader, s: mlx_stream) c_int {
    return c.mlx_load_safetensors_reader(res_0, res_1, in_stream, s);
}
pub fn mlx_load_safetensors(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, file: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_load_safetensors(res_0, res_1, file, s);
}
pub const mlx_save_writer = c.mlx_save_writer;
pub const mlx_save = c.mlx_save;
pub const mlx_save_gguf = c.mlx_save_gguf;
pub const mlx_save_safetensors_writer = c.mlx_save_safetensors_writer;
pub const mlx_save_safetensors = c.mlx_save_safetensors;
pub const mlx_io_reader_descriptor = c.mlx_io_reader_descriptor;
pub fn mlx_io_reader_tostring(str_: *mlx_string, io: mlx_io_reader) c_int {
    return c.mlx_io_reader_tostring(str_, io);
}
pub const mlx_io_reader_free = c.mlx_io_reader_free;
pub const mlx_io_writer_new = c.mlx_io_writer_new;
pub const mlx_io_writer_descriptor = c.mlx_io_writer_descriptor;
pub fn mlx_io_writer_tostring(str_: *mlx_string, io: mlx_io_writer) c_int {
    return c.mlx_io_writer_tostring(str_, io);
}
pub const mlx_io_writer_free = c.mlx_io_writer_free;
pub const mlx_io_gguf_new = c.mlx_io_gguf_new;
pub const mlx_io_gguf_free = c.mlx_io_gguf_free;
pub fn mlx_io_gguf_get_keys(keys: *mlx_vector_string, io: mlx_io_gguf) c_int {
    return c.mlx_io_gguf_get_keys(keys, io);
}
pub fn mlx_io_gguf_get_array(arr: *mlx_array, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_get_array(arr, io, key);
}
pub fn mlx_io_gguf_get_metadata_array(arr: *mlx_array, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_get_metadata_array(arr, io, key);
}
pub fn mlx_io_gguf_get_metadata_string(str: *mlx_string, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_get_metadata_string(str, io, key);
}
pub fn mlx_io_gguf_get_metadata_vector_string(vstr: *mlx_vector_string, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_get_metadata_vector_string(vstr, io, key);
}
pub fn mlx_io_gguf_has_metadata_array(flag: *bool, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_has_metadata_array(flag, io, key);
}
pub fn mlx_io_gguf_has_metadata_string(flag: *bool, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_has_metadata_string(flag, io, key);
}
pub fn mlx_io_gguf_has_metadata_vector_string(flag: *bool, io: mlx_io_gguf, key: [*:0]const u8) c_int {
    return c.mlx_io_gguf_has_metadata_vector_string(flag, io, key);
}
pub const mlx_io_gguf_set_array = c.mlx_io_gguf_set_array;
pub const mlx_io_gguf_set_metadata_array = c.mlx_io_gguf_set_metadata_array;
pub const mlx_io_gguf_set_metadata_string = c.mlx_io_gguf_set_metadata_string;
pub const mlx_io_gguf_set_metadata_vector_string = c.mlx_io_gguf_set_metadata_vector_string;
pub fn mlx_linalg_cholesky(res: *mlx_array, a: mlx_array, upper: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_cholesky(res, a, upper, s);
}
pub fn mlx_linalg_cholesky_inv(res: *mlx_array, a: mlx_array, upper: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_cholesky_inv(res, a, upper, s);
}
pub fn mlx_linalg_cross(res: *mlx_array, a: mlx_array, b: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_linalg_cross(res, a, b, axis, s);
}
pub fn mlx_linalg_eig(res_0: *mlx_array, res_1: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_eig(res_0, res_1, a, s);
}
pub fn mlx_linalg_eigh(res_0: *mlx_array, res_1: *mlx_array, a: mlx_array, UPLO: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_linalg_eigh(res_0, res_1, a, UPLO, s);
}
pub fn mlx_linalg_eigvals(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_eigvals(res, a, s);
}
pub fn mlx_linalg_eigvalsh(res: *mlx_array, a: mlx_array, UPLO: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_linalg_eigvalsh(res, a, UPLO, s);
}
pub fn mlx_linalg_inv(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_inv(res, a, s);
}
pub fn mlx_linalg_lu(res: *mlx_vector_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_lu(res, a, s);
}
pub fn mlx_linalg_lu_factor(res_0: *mlx_array, res_1: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_lu_factor(res_0, res_1, a, s);
}
pub fn mlx_linalg_norm(res: *mlx_array, a: mlx_array, ord: f64, axis: [*]const c_int, axis_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_norm(res, a, ord, axis, axis_num, keepdims, s);
}
pub fn mlx_linalg_norm_matrix(res: *mlx_array, a: mlx_array, ord: [*:0]const u8, axis: [*]const c_int, axis_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_norm_matrix(res, a, ord, axis, axis_num, keepdims, s);
}
pub fn mlx_linalg_norm_l2(res: *mlx_array, a: mlx_array, axis: [*]const c_int, axis_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_norm_l2(res, a, axis, axis_num, keepdims, s);
}
pub fn mlx_linalg_pinv(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_pinv(res, a, s);
}
pub fn mlx_linalg_qr(res_0: *mlx_array, res_1: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_qr(res_0, res_1, a, s);
}
pub fn mlx_linalg_solve(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_linalg_solve(res, a, b, s);
}
pub fn mlx_linalg_solve_triangular(res: *mlx_array, a: mlx_array, b: mlx_array, upper: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_solve_triangular(res, a, b, upper, s);
}
pub fn mlx_linalg_svd(res: *mlx_vector_array, a: mlx_array, compute_uv: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_svd(res, a, compute_uv, s);
}
pub fn mlx_linalg_tri_inv(res: *mlx_array, a: mlx_array, upper: bool, s: mlx_stream) c_int {
    return c.mlx_linalg_tri_inv(res, a, upper, s);
}
pub const mlx_map_string_to_array_new = c.mlx_map_string_to_array_new;
pub fn mlx_map_string_to_array_set(map: *mlx_map_string_to_array, src: mlx_map_string_to_array) c_int {
    return c.mlx_map_string_to_array_set(map, src);
}
pub const mlx_map_string_to_array_free = c.mlx_map_string_to_array_free;
pub const mlx_map_string_to_array_insert = c.mlx_map_string_to_array_insert;
pub fn mlx_map_string_to_array_get(value: *mlx_array, map: mlx_map_string_to_array, key: [*:0]const u8) c_int {
    return c.mlx_map_string_to_array_get(value, map, key);
}
pub const mlx_map_string_to_array_iterator_new = c.mlx_map_string_to_array_iterator_new;
pub const mlx_map_string_to_array_iterator_free = c.mlx_map_string_to_array_iterator_free;
pub fn mlx_map_string_to_array_iterator_next(key: anytype, value: *mlx_array, it: mlx_map_string_to_array_iterator) c_int {
    return c.mlx_map_string_to_array_iterator_next(@ptrCast(@constCast(key)), value, it);
}
pub const mlx_map_string_to_string_new = c.mlx_map_string_to_string_new;
pub fn mlx_map_string_to_string_set(map: *mlx_map_string_to_string, src: mlx_map_string_to_string) c_int {
    return c.mlx_map_string_to_string_set(map, src);
}
pub const mlx_map_string_to_string_free = c.mlx_map_string_to_string_free;
pub const mlx_map_string_to_string_insert = c.mlx_map_string_to_string_insert;
pub fn mlx_map_string_to_string_get(value: anytype, map: mlx_map_string_to_string, key: [*:0]const u8) c_int {
    return c.mlx_map_string_to_string_get(@ptrCast(@constCast(value)), map, key);
}
pub const mlx_map_string_to_string_iterator_new = c.mlx_map_string_to_string_iterator_new;
pub const mlx_map_string_to_string_iterator_free = c.mlx_map_string_to_string_iterator_free;
pub fn mlx_map_string_to_string_iterator_next(key: anytype, value: anytype, it: mlx_map_string_to_string_iterator) c_int {
    return c.mlx_map_string_to_string_iterator_next(@ptrCast(@constCast(key)), @ptrCast(@constCast(value)), it);
}
pub const mlx_clear_cache = c.mlx_clear_cache;
pub fn mlx_get_active_memory(res: *usize) c_int {
    return c.mlx_get_active_memory(res);
}
pub fn mlx_get_cache_memory(res: *usize) c_int {
    return c.mlx_get_cache_memory(res);
}
pub fn mlx_get_memory_limit(res: *usize) c_int {
    return c.mlx_get_memory_limit(res);
}
pub fn mlx_get_peak_memory(res: *usize) c_int {
    return c.mlx_get_peak_memory(res);
}
pub const mlx_reset_peak_memory = c.mlx_reset_peak_memory;
pub fn mlx_set_cache_limit(res: *usize, limit: usize) c_int {
    return c.mlx_set_cache_limit(res, limit);
}
pub fn mlx_set_memory_limit(res: *usize, limit: usize) c_int {
    return c.mlx_set_memory_limit(res, limit);
}
pub fn mlx_set_wired_limit(res: *usize, limit: usize) c_int {
    return c.mlx_set_wired_limit(res, limit);
}
pub fn mlx_metal_is_available(res: *bool) c_int {
    return c.mlx_metal_is_available(res);
}
pub const mlx_metal_start_capture = c.mlx_metal_start_capture;
pub const mlx_metal_stop_capture = c.mlx_metal_stop_capture;
pub fn mlx_abs(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_abs(res, a, s);
}
pub fn mlx_add(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_add(res, a, b, s);
}
pub fn mlx_addmm(res: *mlx_array, c_: mlx_array, a: mlx_array, b: mlx_array, alpha: f32, beta: f32, s: mlx_stream) c_int {
    return c.mlx_addmm(res, c_, a, b, alpha, beta, s);
}
pub fn mlx_all_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_all_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_all_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_all_axis(res, a, axis, keepdims, s);
}
pub fn mlx_all(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_all(res, a, keepdims, s);
}
pub fn mlx_allclose(res: *mlx_array, a: mlx_array, b: mlx_array, rtol: f64, atol: f64, equal_nan: bool, s: mlx_stream) c_int {
    return c.mlx_allclose(res, a, b, rtol, atol, equal_nan, s);
}
pub fn mlx_any_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_any_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_any_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_any_axis(res, a, axis, keepdims, s);
}
pub fn mlx_any(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_any(res, a, keepdims, s);
}
pub fn mlx_arange(res: *mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_arange(res, start, stop, step, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_arccos(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arccos(res, a, s);
}
pub fn mlx_arccosh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arccosh(res, a, s);
}
pub fn mlx_arcsin(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arcsin(res, a, s);
}
pub fn mlx_arcsinh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arcsinh(res, a, s);
}
pub fn mlx_arctan(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arctan(res, a, s);
}
pub fn mlx_arctan2(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arctan2(res, a, b, s);
}
pub fn mlx_arctanh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_arctanh(res, a, s);
}
pub fn mlx_argmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_argmax_axis(res, a, axis, keepdims, s);
}
pub fn mlx_argmax(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_argmax(res, a, keepdims, s);
}
pub fn mlx_argmin_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_argmin_axis(res, a, axis, keepdims, s);
}
pub fn mlx_argmin(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_argmin(res, a, keepdims, s);
}
pub fn mlx_argpartition_axis(res: *mlx_array, a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_argpartition_axis(res, a, kth, axis, s);
}
pub fn mlx_argpartition(res: *mlx_array, a: mlx_array, kth: c_int, s: mlx_stream) c_int {
    return c.mlx_argpartition(res, a, kth, s);
}
pub fn mlx_argsort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_argsort_axis(res, a, axis, s);
}
pub fn mlx_argsort(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_argsort(res, a, s);
}
pub fn mlx_array_equal(res: *mlx_array, a: mlx_array, b: mlx_array, equal_nan: bool, s: mlx_stream) c_int {
    return c.mlx_array_equal(res, a, b, equal_nan, s);
}
pub fn mlx_as_strided(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, strides: [*]const i64, strides_num: usize, offset: usize, s: mlx_stream) c_int {
    return c.mlx_as_strided(res, a, shape, shape_num, strides, strides_num, offset, s);
}
pub fn mlx_astype(res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_astype(res, a, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_atleast_1d(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_atleast_1d(res, a, s);
}
pub fn mlx_atleast_2d(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_atleast_2d(res, a, s);
}
pub fn mlx_atleast_3d(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_atleast_3d(res, a, s);
}
pub fn mlx_bartlett(res: *mlx_array, M: c_int, s: mlx_stream) c_int {
    return c.mlx_bartlett(res, M, s);
}
pub fn mlx_bitwise_and(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_bitwise_and(res, a, b, s);
}
pub fn mlx_bitwise_invert(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_bitwise_invert(res, a, s);
}
pub fn mlx_bitwise_or(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_bitwise_or(res, a, b, s);
}
pub fn mlx_bitwise_xor(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_bitwise_xor(res, a, b, s);
}
pub fn mlx_blackman(res: *mlx_array, M: c_int, s: mlx_stream) c_int {
    return c.mlx_blackman(res, M, s);
}
pub fn mlx_block_masked_mm(res: *mlx_array, a: mlx_array, b: mlx_array, block_size: c_int, mask_out: mlx_array, mask_lhs: mlx_array, mask_rhs: mlx_array, s: mlx_stream) c_int {
    return c.mlx_block_masked_mm(res, a, b, block_size, mask_out, mask_lhs, mask_rhs, s);
}
pub fn mlx_broadcast_arrays(res: *mlx_vector_array, inputs: mlx_vector_array, s: mlx_stream) c_int {
    return c.mlx_broadcast_arrays(res, inputs, s);
}
pub fn mlx_broadcast_to(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int {
    return c.mlx_broadcast_to(res, a, shape, shape_num, s);
}
pub fn mlx_ceil(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_ceil(res, a, s);
}
pub fn mlx_clip(res: *mlx_array, a: mlx_array, a_min: mlx_array, a_max: mlx_array, s: mlx_stream) c_int {
    return c.mlx_clip(res, a, a_min, a_max, s);
}
pub fn mlx_concatenate_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_concatenate_axis(res, arrays, axis, s);
}
pub fn mlx_concatenate(res: *mlx_array, arrays: mlx_vector_array, s: mlx_stream) c_int {
    return c.mlx_concatenate(res, arrays, s);
}
pub fn mlx_conjugate(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_conjugate(res, a, s);
}
pub fn mlx_contiguous(res: *mlx_array, a: mlx_array, allow_col_major: bool, s: mlx_stream) c_int {
    return c.mlx_contiguous(res, a, allow_col_major, s);
}
pub fn mlx_conv1d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv1d(res, input, weight, stride, padding, dilation, groups, s);
}
pub fn mlx_conv2d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv2d(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, groups, s);
}
pub fn mlx_conv3d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv3d(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, groups, s);
}
pub fn mlx_conv_general(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: [*]const c_int, stride_num: usize, padding_lo: [*]const c_int, padding_lo_num: usize, padding_hi: [*]const c_int, padding_hi_num: usize, kernel_dilation: [*]const c_int, kernel_dilation_num: usize, input_dilation: [*]const c_int, input_dilation_num: usize, groups: c_int, flip: bool, s: mlx_stream) c_int {
    return c.mlx_conv_general(res, input, weight, stride, stride_num, padding_lo, padding_lo_num, padding_hi, padding_hi_num, kernel_dilation, kernel_dilation_num, input_dilation, input_dilation_num, groups, flip, s);
}
pub fn mlx_conv_transpose1d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, output_padding: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv_transpose1d(res, input, weight, stride, padding, dilation, output_padding, groups, s);
}
pub fn mlx_conv_transpose2d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, output_padding_0: c_int, output_padding_1: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv_transpose2d(res, input, weight, stride_0, stride_1, padding_0, padding_1, dilation_0, dilation_1, output_padding_0, output_padding_1, groups, s);
}
pub fn mlx_conv_transpose3d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, output_padding_0: c_int, output_padding_1: c_int, output_padding_2: c_int, groups: c_int, s: mlx_stream) c_int {
    return c.mlx_conv_transpose3d(res, input, weight, stride_0, stride_1, stride_2, padding_0, padding_1, padding_2, dilation_0, dilation_1, dilation_2, output_padding_0, output_padding_1, output_padding_2, groups, s);
}
pub fn mlx_copy(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_copy(res, a, s);
}
pub fn mlx_cos(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_cos(res, a, s);
}
pub fn mlx_cosh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_cosh(res, a, s);
}
pub fn mlx_cummax(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int {
    return c.mlx_cummax(res, a, axis, reverse, inclusive, s);
}
pub fn mlx_cummin(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int {
    return c.mlx_cummin(res, a, axis, reverse, inclusive, s);
}
pub fn mlx_cumprod(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int {
    return c.mlx_cumprod(res, a, axis, reverse, inclusive, s);
}
pub fn mlx_cumsum(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int {
    return c.mlx_cumsum(res, a, axis, reverse, inclusive, s);
}
pub fn mlx_degrees(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_degrees(res, a, s);
}
pub fn mlx_depends(res: *mlx_vector_array, inputs: mlx_vector_array, dependencies: mlx_vector_array) c_int {
    return c.mlx_depends(res, inputs, dependencies);
}
pub fn mlx_dequantize(res: *mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, dtype: mlx_optional_dtype, s: mlx_stream) c_int {
    return c.mlx_dequantize(res, w, scales, biases, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, global_scale, .{ .value = @intCast(@intFromEnum(dtype.value)), .has_value = dtype.has_value }, s);
}
pub fn mlx_diag(res: *mlx_array, a: mlx_array, k: c_int, s: mlx_stream) c_int {
    return c.mlx_diag(res, a, k, s);
}
pub fn mlx_diagonal(res: *mlx_array, a: mlx_array, offset: c_int, axis1: c_int, axis2: c_int, s: mlx_stream) c_int {
    return c.mlx_diagonal(res, a, offset, axis1, axis2, s);
}
pub fn mlx_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_divide(res, a, b, s);
}
pub fn mlx_divmod(res: *mlx_vector_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_divmod(res, a, b, s);
}
pub fn mlx_einsum(res: *mlx_array, subscripts: [*:0]const u8, operands: mlx_vector_array, s: mlx_stream) c_int {
    return c.mlx_einsum(res, subscripts, operands, s);
}
pub fn mlx_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_equal(res, a, b, s);
}
pub fn mlx_erf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_erf(res, a, s);
}
pub fn mlx_erfinv(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_erfinv(res, a, s);
}
pub fn mlx_exp(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_exp(res, a, s);
}
pub fn mlx_expand_dims_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_expand_dims_axes(res, a, axes, axes_num, s);
}
pub fn mlx_expand_dims(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_expand_dims(res, a, axis, s);
}
pub fn mlx_expm1(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_expm1(res, a, s);
}
pub fn mlx_eye(res: *mlx_array, n: c_int, m: c_int, k: c_int, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_eye(res, n, m, k, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_flatten(res: *mlx_array, a: mlx_array, start_axis: c_int, end_axis: c_int, s: mlx_stream) c_int {
    return c.mlx_flatten(res, a, start_axis, end_axis, s);
}
pub fn mlx_floor(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_floor(res, a, s);
}
pub fn mlx_floor_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_floor_divide(res, a, b, s);
}
pub fn mlx_from_fp8(res: *mlx_array, x: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_from_fp8(res, x, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_full(res: *mlx_array, shape: [*]const c_int, shape_num: usize, vals: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_full(res, shape, shape_num, vals, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_full_like(res: *mlx_array, a: mlx_array, vals: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_full_like(res, a, vals, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_gather(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, axes: [*]const c_int, axes_num: usize, slice_sizes: [*]const c_int, slice_sizes_num: usize, s: mlx_stream) c_int {
    return c.mlx_gather(res, a, indices, axes, axes_num, slice_sizes, slice_sizes_num, s);
}
pub fn mlx_gather_single(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, slice_sizes: [*]const c_int, slice_sizes_num: usize, s: mlx_stream) c_int {
    return c.mlx_gather_single(res, a, indices, axis, slice_sizes, slice_sizes_num, s);
}
pub fn mlx_gather_mm(res: *mlx_array, a: mlx_array, b: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, sorted_indices: bool, s: mlx_stream) c_int {
    return c.mlx_gather_mm(res, a, b, lhs_indices, rhs_indices, sorted_indices, s);
}
pub fn mlx_gather_qmm(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, transpose: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, sorted_indices: bool, s: mlx_stream) c_int {
    return c.mlx_gather_qmm(res, x, w, scales, biases, lhs_indices, rhs_indices, transpose, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, sorted_indices, s);
}
pub fn mlx_greater(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_greater(res, a, b, s);
}
pub fn mlx_greater_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_greater_equal(res, a, b, s);
}
pub fn mlx_hadamard_transform(res: *mlx_array, a: mlx_array, scale: mlx_optional_float, s: mlx_stream) c_int {
    return c.mlx_hadamard_transform(res, a, .{ .value = scale.value, .has_value = scale.has_value }, s);
}
pub fn mlx_hamming(res: *mlx_array, M: c_int, s: mlx_stream) c_int {
    return c.mlx_hamming(res, M, s);
}
pub fn mlx_hanning(res: *mlx_array, M: c_int, s: mlx_stream) c_int {
    return c.mlx_hanning(res, M, s);
}
pub fn mlx_identity(res: *mlx_array, n: c_int, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_identity(res, n, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_imag(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_imag(res, a, s);
}
pub fn mlx_inner(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_inner(res, a, b, s);
}
pub fn mlx_isclose(res: *mlx_array, a: mlx_array, b: mlx_array, rtol: f64, atol: f64, equal_nan: bool, s: mlx_stream) c_int {
    return c.mlx_isclose(res, a, b, rtol, atol, equal_nan, s);
}
pub fn mlx_isfinite(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isfinite(res, a, s);
}
pub fn mlx_isinf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isinf(res, a, s);
}
pub fn mlx_isnan(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isnan(res, a, s);
}
pub fn mlx_isneginf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isneginf(res, a, s);
}
pub fn mlx_isposinf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_isposinf(res, a, s);
}
pub fn mlx_kron(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_kron(res, a, b, s);
}
pub fn mlx_left_shift(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_left_shift(res, a, b, s);
}
pub fn mlx_less(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_less(res, a, b, s);
}
pub fn mlx_less_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_less_equal(res, a, b, s);
}
pub fn mlx_linspace(res: *mlx_array, start: f64, stop: f64, num: c_int, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_linspace(res, start, stop, num, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_log(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_log(res, a, s);
}
pub fn mlx_log10(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_log10(res, a, s);
}
pub fn mlx_log1p(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_log1p(res, a, s);
}
pub fn mlx_log2(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_log2(res, a, s);
}
pub fn mlx_logaddexp(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logaddexp(res, a, b, s);
}
pub fn mlx_logcumsumexp(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int {
    return c.mlx_logcumsumexp(res, a, axis, reverse, inclusive, s);
}
pub fn mlx_logical_and(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logical_and(res, a, b, s);
}
pub fn mlx_logical_not(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logical_not(res, a, s);
}
pub fn mlx_logical_or(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_logical_or(res, a, b, s);
}
pub fn mlx_logsumexp_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_logsumexp_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_logsumexp_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_logsumexp_axis(res, a, axis, keepdims, s);
}
pub fn mlx_logsumexp(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_logsumexp(res, a, keepdims, s);
}
pub fn mlx_masked_scatter(res: *mlx_array, a: mlx_array, mask: mlx_array, src: mlx_array, s: mlx_stream) c_int {
    return c.mlx_masked_scatter(res, a, mask, src, s);
}
pub fn mlx_matmul(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_matmul(res, a, b, s);
}
pub fn mlx_max_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_max_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_max_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_max_axis(res, a, axis, keepdims, s);
}
pub fn mlx_max(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_max(res, a, keepdims, s);
}
pub fn mlx_maximum(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_maximum(res, a, b, s);
}
pub fn mlx_mean_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_mean_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_mean_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_mean_axis(res, a, axis, keepdims, s);
}
pub fn mlx_mean(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_mean(res, a, keepdims, s);
}
pub fn mlx_median(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_median(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_meshgrid(res: *mlx_vector_array, arrays: mlx_vector_array, sparse: bool, indexing: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_meshgrid(res, arrays, sparse, indexing, s);
}
pub fn mlx_min_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_min_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_min_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_min_axis(res, a, axis, keepdims, s);
}
pub fn mlx_min(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_min(res, a, keepdims, s);
}
pub fn mlx_minimum(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_minimum(res, a, b, s);
}
pub fn mlx_moveaxis(res: *mlx_array, a: mlx_array, source: c_int, destination: c_int, s: mlx_stream) c_int {
    return c.mlx_moveaxis(res, a, source, destination, s);
}
pub fn mlx_multiply(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_multiply(res, a, b, s);
}
pub fn mlx_nan_to_num(res: *mlx_array, a: mlx_array, nan: f32, posinf: mlx_optional_float, neginf: mlx_optional_float, s: mlx_stream) c_int {
    return c.mlx_nan_to_num(res, a, nan, .{ .value = posinf.value, .has_value = posinf.has_value }, .{ .value = neginf.value, .has_value = neginf.has_value }, s);
}
pub fn mlx_negative(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_negative(res, a, s);
}
pub fn mlx_not_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_not_equal(res, a, b, s);
}
pub fn mlx_number_of_elements(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, inverted: bool, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_number_of_elements(res, a, axes, axes_num, inverted, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_ones(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_ones(res, shape, shape_num, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_ones_like(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_ones_like(res, a, s);
}
pub fn mlx_outer(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_outer(res, a, b, s);
}
pub fn mlx_pad(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, low_pad_size: [*]const c_int, low_pad_size_num: usize, high_pad_size: [*]const c_int, high_pad_size_num: usize, pad_value: mlx_array, mode: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_pad(res, a, axes, axes_num, low_pad_size, low_pad_size_num, high_pad_size, high_pad_size_num, pad_value, mode, s);
}
pub fn mlx_pad_symmetric(res: *mlx_array, a: mlx_array, pad_width: c_int, pad_value: mlx_array, mode: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_pad_symmetric(res, a, pad_width, pad_value, mode, s);
}
pub fn mlx_partition_axis(res: *mlx_array, a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_partition_axis(res, a, kth, axis, s);
}
pub fn mlx_partition(res: *mlx_array, a: mlx_array, kth: c_int, s: mlx_stream) c_int {
    return c.mlx_partition(res, a, kth, s);
}
pub fn mlx_power(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_power(res, a, b, s);
}
pub fn mlx_prod_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_prod_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_prod_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_prod_axis(res, a, axis, keepdims, s);
}
pub fn mlx_prod(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_prod(res, a, keepdims, s);
}
pub fn mlx_put_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, values: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_put_along_axis(res, a, indices, values, axis, s);
}
pub fn mlx_qqmm(res: *mlx_array, x: mlx_array, w: mlx_array, w_scales: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale_x: mlx_array, global_scale_w: mlx_array, s: mlx_stream) c_int {
    return c.mlx_qqmm(res, x, w, w_scales, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, global_scale_x, global_scale_w, s);
}
pub fn mlx_quantize(res: *mlx_vector_array, w: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, s: mlx_stream) c_int {
    return c.mlx_quantize(res, w, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, global_scale, s);
}
pub fn mlx_quantized_matmul(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, s: mlx_stream) c_int {
    return c.mlx_quantized_matmul(res, x, w, scales, biases, transpose, .{ .value = group_size.value, .has_value = group_size.has_value }, .{ .value = bits.value, .has_value = bits.has_value }, mode, s);
}
pub fn mlx_radians(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_radians(res, a, s);
}
pub fn mlx_real(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_real(res, a, s);
}
pub fn mlx_reciprocal(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_reciprocal(res, a, s);
}
pub fn mlx_remainder(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_remainder(res, a, b, s);
}
pub fn mlx_repeat_axis(res: *mlx_array, arr: mlx_array, repeats: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_repeat_axis(res, arr, repeats, axis, s);
}
pub fn mlx_repeat(res: *mlx_array, arr: mlx_array, repeats: c_int, s: mlx_stream) c_int {
    return c.mlx_repeat(res, arr, repeats, s);
}
pub fn mlx_reshape(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int {
    return c.mlx_reshape(res, a, shape, shape_num, s);
}
pub fn mlx_right_shift(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_right_shift(res, a, b, s);
}
pub fn mlx_roll_axis(res: *mlx_array, a: mlx_array, shift: [*]const c_int, shift_num: usize, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_roll_axis(res, a, shift, shift_num, axis, s);
}
pub fn mlx_roll_axes(res: *mlx_array, a: mlx_array, shift: [*]const c_int, shift_num: usize, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_roll_axes(res, a, shift, shift_num, axes, axes_num, s);
}
pub fn mlx_roll(res: *mlx_array, a: mlx_array, shift: [*]const c_int, shift_num: usize, s: mlx_stream) c_int {
    return c.mlx_roll(res, a, shift, shift_num, s);
}
pub fn mlx_round(res: *mlx_array, a: mlx_array, decimals: c_int, s: mlx_stream) c_int {
    return c.mlx_round(res, a, decimals, s);
}
pub fn mlx_rsqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_rsqrt(res, a, s);
}
pub fn mlx_scatter(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, updates: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_scatter(res, a, indices, updates, axes, axes_num, s);
}
pub fn mlx_scatter_single(res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_single(res, a, indices, updates, axis, s);
}
pub fn mlx_scatter_add(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, updates: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_scatter_add(res, a, indices, updates, axes, axes_num, s);
}
pub fn mlx_scatter_add_single(res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_add_single(res, a, indices, updates, axis, s);
}
pub fn mlx_scatter_add_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, values: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_add_axis(res, a, indices, values, axis, s);
}
pub fn mlx_scatter_max(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, updates: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_scatter_max(res, a, indices, updates, axes, axes_num, s);
}
pub fn mlx_scatter_max_single(res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_max_single(res, a, indices, updates, axis, s);
}
pub fn mlx_scatter_min(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, updates: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_scatter_min(res, a, indices, updates, axes, axes_num, s);
}
pub fn mlx_scatter_min_single(res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_min_single(res, a, indices, updates, axis, s);
}
pub fn mlx_scatter_prod(res: *mlx_array, a: mlx_array, indices: mlx_vector_array, updates: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_scatter_prod(res, a, indices, updates, axes, axes_num, s);
}
pub fn mlx_scatter_prod_single(res: *mlx_array, a: mlx_array, indices: mlx_array, updates: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_scatter_prod_single(res, a, indices, updates, axis, s);
}
pub fn mlx_segmented_mm(res: *mlx_array, a: mlx_array, b: mlx_array, segments: mlx_array, s: mlx_stream) c_int {
    return c.mlx_segmented_mm(res, a, b, segments, s);
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
pub fn mlx_sinh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sinh(res, a, s);
}
pub fn mlx_slice(res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice(res, a, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_dynamic(res: *mlx_array, a: mlx_array, start: mlx_array, axes: [*]const c_int, axes_num: usize, slice_size: [*]const c_int, slice_size_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_dynamic(res, a, start, axes, axes_num, slice_size, slice_size_num, s);
}
pub fn mlx_slice_update(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_update_dynamic(res: *mlx_array, src: mlx_array, update: mlx_array, start: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update_dynamic(res, src, update, start, axes, axes_num, s);
}
pub fn mlx_slice_update_add(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update_add(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_update_max(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update_max(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_update_min(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update_min(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_slice_update_prod(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int {
    return c.mlx_slice_update_prod(res, src, update, start, start_num, stop, stop_num, strides, strides_num, s);
}
pub fn mlx_softmax_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, precise: bool, s: mlx_stream) c_int {
    return c.mlx_softmax_axes(res, a, axes, axes_num, precise, s);
}
pub fn mlx_softmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, precise: bool, s: mlx_stream) c_int {
    return c.mlx_softmax_axis(res, a, axis, precise, s);
}
pub fn mlx_softmax(res: *mlx_array, a: mlx_array, precise: bool, s: mlx_stream) c_int {
    return c.mlx_softmax(res, a, precise, s);
}
pub fn mlx_sort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_sort_axis(res, a, axis, s);
}
pub fn mlx_sort(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sort(res, a, s);
}
pub fn mlx_split(res: *mlx_vector_array, a: mlx_array, num_splits: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_split(res, a, num_splits, axis, s);
}
pub fn mlx_split_sections(res: *mlx_vector_array, a: mlx_array, indices: [*]const c_int, indices_num: usize, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_split_sections(res, a, indices, indices_num, axis, s);
}
pub fn mlx_sqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_sqrt(res, a, s);
}
pub fn mlx_square(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_square(res, a, s);
}
pub fn mlx_squeeze_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_squeeze_axes(res, a, axes, axes_num, s);
}
pub fn mlx_squeeze_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_squeeze_axis(res, a, axis, s);
}
pub fn mlx_squeeze(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_squeeze(res, a, s);
}
pub fn mlx_stack_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_stack_axis(res, arrays, axis, s);
}
pub fn mlx_stack(res: *mlx_array, arrays: mlx_vector_array, s: mlx_stream) c_int {
    return c.mlx_stack(res, arrays, s);
}
pub fn mlx_std_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_std_axes(res, a, axes, axes_num, keepdims, ddof, s);
}
pub fn mlx_std_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_std_axis(res, a, axis, keepdims, ddof, s);
}
pub fn mlx_std(res: *mlx_array, a: mlx_array, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_std(res, a, keepdims, ddof, s);
}
pub fn mlx_stop_gradient(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_stop_gradient(res, a, s);
}
pub fn mlx_subtract(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int {
    return c.mlx_subtract(res, a, b, s);
}
pub fn mlx_sum_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_sum_axes(res, a, axes, axes_num, keepdims, s);
}
pub fn mlx_sum_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_sum_axis(res, a, axis, keepdims, s);
}
pub fn mlx_sum(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int {
    return c.mlx_sum(res, a, keepdims, s);
}
pub fn mlx_swapaxes(res: *mlx_array, a: mlx_array, axis1: c_int, axis2: c_int, s: mlx_stream) c_int {
    return c.mlx_swapaxes(res, a, axis1, axis2, s);
}
pub fn mlx_take_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_take_axis(res, a, indices, axis, s);
}
pub fn mlx_take(res: *mlx_array, a: mlx_array, indices: mlx_array, s: mlx_stream) c_int {
    return c.mlx_take(res, a, indices, s);
}
pub fn mlx_take_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_take_along_axis(res, a, indices, axis, s);
}
pub fn mlx_tan(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_tan(res, a, s);
}
pub fn mlx_tanh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_tanh(res, a, s);
}
pub fn mlx_tensordot(res: *mlx_array, a: mlx_array, b: mlx_array, axes_a: [*]const c_int, axes_a_num: usize, axes_b: [*]const c_int, axes_b_num: usize, s: mlx_stream) c_int {
    return c.mlx_tensordot(res, a, b, axes_a, axes_a_num, axes_b, axes_b_num, s);
}
pub fn mlx_tensordot_axis(res: *mlx_array, a: mlx_array, b: mlx_array, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_tensordot_axis(res, a, b, axis, s);
}
pub fn mlx_tile(res: *mlx_array, arr: mlx_array, reps: [*]const c_int, reps_num: usize, s: mlx_stream) c_int {
    return c.mlx_tile(res, arr, reps, reps_num, s);
}
pub fn mlx_to_fp8(res: *mlx_array, x: mlx_array, s: mlx_stream) c_int {
    return c.mlx_to_fp8(res, x, s);
}
pub fn mlx_topk_axis(res: *mlx_array, a: mlx_array, k: c_int, axis: c_int, s: mlx_stream) c_int {
    return c.mlx_topk_axis(res, a, k, axis, s);
}
pub fn mlx_topk(res: *mlx_array, a: mlx_array, k: c_int, s: mlx_stream) c_int {
    return c.mlx_topk(res, a, k, s);
}
pub fn mlx_trace(res: *mlx_array, a: mlx_array, offset: c_int, axis1: c_int, axis2: c_int, dtype: mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_trace(res, a, offset, axis1, axis2, @intCast(@intFromEnum(dtype)), s);
}
pub fn mlx_transpose_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int {
    return c.mlx_transpose_axes(res, a, axes, axes_num, s);
}
pub fn mlx_transpose(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_transpose(res, a, s);
}
pub fn mlx_tri(res: *mlx_array, n: c_int, m: c_int, k: c_int, @"type": mlx_dtype, s: mlx_stream) c_int {
    return c.mlx_tri(res, n, m, k, @intCast(@intFromEnum(@"type")), s);
}
pub fn mlx_tril(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int {
    return c.mlx_tril(res, x, k, s);
}
pub fn mlx_triu(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int {
    return c.mlx_triu(res, x, k, s);
}
pub fn mlx_unflatten(res: *mlx_array, a: mlx_array, axis: c_int, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int {
    return c.mlx_unflatten(res, a, axis, shape, shape_num, s);
}
pub fn mlx_var_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_var_axes(res, a, axes, axes_num, keepdims, ddof, s);
}
pub fn mlx_var_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_var_axis(res, a, axis, keepdims, ddof, s);
}
pub fn mlx_var(res: *mlx_array, a: mlx_array, keepdims: bool, ddof: c_int, s: mlx_stream) c_int {
    return c.mlx_var(res, a, keepdims, ddof, s);
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
pub fn mlx_zeros_like(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int {
    return c.mlx_zeros_like(res, a, s);
}
pub fn mlx_random_bernoulli(res: *mlx_array, p: mlx_array, shape: [*]const c_int, shape_num: usize, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_bernoulli(res, p, shape, shape_num, key, s);
}
pub fn mlx_random_bits(res: *mlx_array, shape: [*]const c_int, shape_num: usize, width: c_int, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_bits(res, shape, shape_num, width, key, s);
}
pub fn mlx_random_categorical_shape(res: *mlx_array, logits: mlx_array, axis: c_int, shape: [*]const c_int, shape_num: usize, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_categorical_shape(res, logits, axis, shape, shape_num, key, s);
}
pub fn mlx_random_categorical_num_samples(res: *mlx_array, logits_: mlx_array, axis: c_int, num_samples: c_int, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_categorical_num_samples(res, logits_, axis, num_samples, key, s);
}
pub fn mlx_random_categorical(res: *mlx_array, logits: mlx_array, axis: c_int, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_categorical(res, logits, axis, key, s);
}
pub fn mlx_random_gumbel(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_gumbel(res, shape, shape_num, @intCast(@intFromEnum(dtype)), key, s);
}
pub fn mlx_random_key(res: *mlx_array, seed: u64) c_int {
    return c.mlx_random_key(res, seed);
}
pub fn mlx_random_laplace(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: f32, scale: f32, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_laplace(res, shape, shape_num, @intCast(@intFromEnum(dtype)), loc, scale, key, s);
}
pub fn mlx_random_multivariate_normal(res: *mlx_array, mean: mlx_array, cov: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_multivariate_normal(res, mean, cov, shape, shape_num, @intCast(@intFromEnum(dtype)), key, s);
}
pub fn mlx_random_normal_broadcast(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: mlx_array, scale: mlx_array, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_normal_broadcast(res, shape, shape_num, @intCast(@intFromEnum(dtype)), loc, scale, key, s);
}
pub fn mlx_random_normal(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: f32, scale: f32, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_normal(res, shape, shape_num, @intCast(@intFromEnum(dtype)), loc, scale, key, s);
}
pub fn mlx_random_permutation(res: *mlx_array, x: mlx_array, axis: c_int, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_permutation(res, x, axis, key, s);
}
pub fn mlx_random_permutation_arange(res: *mlx_array, x: c_int, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_permutation_arange(res, x, key, s);
}
pub fn mlx_random_randint(res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_randint(res, low, high, shape, shape_num, @intCast(@intFromEnum(dtype)), key, s);
}
pub const mlx_random_seed = c.mlx_random_seed;
pub fn mlx_random_split_num(res: *mlx_array, key: mlx_array, num: c_int, s: mlx_stream) c_int {
    return c.mlx_random_split_num(res, key, num, s);
}
pub fn mlx_random_split(res_0: *mlx_array, res_1: *mlx_array, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_split(res_0, res_1, key, s);
}
pub fn mlx_random_truncated_normal(res: *mlx_array, lower: mlx_array, upper: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_truncated_normal(res, lower, upper, shape, shape_num, @intCast(@intFromEnum(dtype)), key, s);
}
pub fn mlx_random_uniform(res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int {
    return c.mlx_random_uniform(res, low, high, shape, shape_num, @intCast(@intFromEnum(dtype)), key, s);
}
pub const mlx_stream_new = c.mlx_stream_new;
pub const mlx_stream_new_device = c.mlx_stream_new_device;
pub fn mlx_stream_set(stream: *mlx_stream, src: mlx_stream) c_int {
    return c.mlx_stream_set(stream, src);
}
pub const mlx_stream_free = c.mlx_stream_free;
pub fn mlx_stream_tostring(str: *mlx_string, stream: mlx_stream) c_int {
    return c.mlx_stream_tostring(str, stream);
}
pub const mlx_stream_equal = c.mlx_stream_equal;
pub fn mlx_stream_get_device(dev: *mlx_device, stream: mlx_stream) c_int {
    return c.mlx_stream_get_device(dev, stream);
}
pub fn mlx_stream_get_index(index: *c_int, stream: mlx_stream) c_int {
    return c.mlx_stream_get_index(index, stream);
}
pub const mlx_synchronize = c.mlx_synchronize;
pub fn mlx_get_default_stream(stream: *mlx_stream, dev: mlx_device) c_int {
    return c.mlx_get_default_stream(stream, dev);
}
pub const mlx_set_default_stream = c.mlx_set_default_stream;
pub const mlx_default_cpu_stream_new = c.mlx_default_cpu_stream_new;
pub const mlx_default_gpu_stream_new = c.mlx_default_gpu_stream_new;
pub const mlx_string_new = c.mlx_string_new;
pub const mlx_string_new_data = c.mlx_string_new_data;
pub fn mlx_string_set(str: *mlx_string, src: mlx_string) c_int {
    return c.mlx_string_set(str, src);
}
pub const mlx_string_data = c.mlx_string_data;
pub const mlx_string_free = c.mlx_string_free;
pub const mlx_async_eval = c.mlx_async_eval;
pub fn mlx_checkpoint(res: *mlx_closure, fun: mlx_closure) c_int {
    return c.mlx_checkpoint(res, fun);
}
pub fn mlx_custom_function(res: *mlx_closure, fun: mlx_closure, fun_vjp: mlx_closure_custom, fun_jvp: mlx_closure_custom_jvp, fun_vmap: mlx_closure_custom_vmap) c_int {
    return c.mlx_custom_function(res, fun, fun_vjp, fun_jvp, fun_vmap);
}
pub fn mlx_custom_vjp(res: *mlx_closure, fun: mlx_closure, fun_vjp: mlx_closure_custom) c_int {
    return c.mlx_custom_vjp(res, fun, fun_vjp);
}
pub const mlx_eval = c.mlx_eval;
pub fn mlx_jvp(res_0: *mlx_vector_array, res_1: *mlx_vector_array, fun: mlx_closure, primals: mlx_vector_array, tangents: mlx_vector_array) c_int {
    return c.mlx_jvp(res_0, res_1, fun, primals, tangents);
}
pub fn mlx_value_and_grad(res: *mlx_closure_value_and_grad, fun: mlx_closure, argnums: [*]const c_int, argnums_num: usize) c_int {
    return c.mlx_value_and_grad(res, fun, argnums, argnums_num);
}
pub fn mlx_vjp(res_0: *mlx_vector_array, res_1: *mlx_vector_array, fun: mlx_closure, primals: mlx_vector_array, cotangents: mlx_vector_array) c_int {
    return c.mlx_vjp(res_0, res_1, fun, primals, cotangents);
}
pub fn mlx_detail_vmap_replace(res: *mlx_vector_array, inputs: mlx_vector_array, s_inputs: mlx_vector_array, s_outputs: mlx_vector_array, in_axes: [*]const c_int, in_axes_num: usize, out_axes: [*]const c_int, out_axes_num: usize) c_int {
    return c.mlx_detail_vmap_replace(res, inputs, s_inputs, s_outputs, in_axes, in_axes_num, out_axes, out_axes_num);
}
pub fn mlx_detail_vmap_trace(res_0: *mlx_vector_array, res_1: *mlx_vector_array, fun: mlx_closure, inputs: mlx_vector_array, in_axes: [*]const c_int, in_axes_num: usize) c_int {
    return c.mlx_detail_vmap_trace(res_0, res_1, fun, inputs, in_axes, in_axes_num);
}
pub const mlx_vector_array_new = c.mlx_vector_array_new;
pub fn mlx_vector_array_set(vec: *mlx_vector_array, src: mlx_vector_array) c_int {
    return c.mlx_vector_array_set(vec, src);
}
pub const mlx_vector_array_free = c.mlx_vector_array_free;
pub fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array {
    return c.mlx_vector_array_new_data(data, size);
}
pub const mlx_vector_array_new_value = c.mlx_vector_array_new_value;
pub fn mlx_vector_array_set_data(vec: *mlx_vector_array, data: [*]const mlx_array, size: usize) c_int {
    return c.mlx_vector_array_set_data(vec, data, size);
}
pub fn mlx_vector_array_set_value(vec: *mlx_vector_array, val: mlx_array) c_int {
    return c.mlx_vector_array_set_value(vec, val);
}
pub fn mlx_vector_array_append_data(vec: mlx_vector_array, data: [*]const mlx_array, size: usize) c_int {
    return c.mlx_vector_array_append_data(vec, data, size);
}
pub const mlx_vector_array_append_value = c.mlx_vector_array_append_value;
pub const mlx_vector_array_size = c.mlx_vector_array_size;
pub fn mlx_vector_array_get(res: *mlx_array, vec: mlx_vector_array, idx: usize) c_int {
    return c.mlx_vector_array_get(res, vec, idx);
}
pub const mlx_vector_vector_array_new = c.mlx_vector_vector_array_new;
pub fn mlx_vector_vector_array_set(vec: *mlx_vector_vector_array, src: mlx_vector_vector_array) c_int {
    return c.mlx_vector_vector_array_set(vec, src);
}
pub const mlx_vector_vector_array_free = c.mlx_vector_vector_array_free;
pub fn mlx_vector_vector_array_new_data(data: [*]const mlx_vector_array, size: usize) mlx_vector_vector_array {
    return c.mlx_vector_vector_array_new_data(data, size);
}
pub const mlx_vector_vector_array_new_value = c.mlx_vector_vector_array_new_value;
pub fn mlx_vector_vector_array_set_data(vec: *mlx_vector_vector_array, data: [*]const mlx_vector_array, size: usize) c_int {
    return c.mlx_vector_vector_array_set_data(vec, data, size);
}
pub fn mlx_vector_vector_array_set_value(vec: *mlx_vector_vector_array, val: mlx_vector_array) c_int {
    return c.mlx_vector_vector_array_set_value(vec, val);
}
pub fn mlx_vector_vector_array_append_data(vec: mlx_vector_vector_array, data: [*]const mlx_vector_array, size: usize) c_int {
    return c.mlx_vector_vector_array_append_data(vec, data, size);
}
pub const mlx_vector_vector_array_append_value = c.mlx_vector_vector_array_append_value;
pub const mlx_vector_vector_array_size = c.mlx_vector_vector_array_size;
pub fn mlx_vector_vector_array_get(res: *mlx_vector_array, vec: mlx_vector_vector_array, idx: usize) c_int {
    return c.mlx_vector_vector_array_get(res, vec, idx);
}
pub const mlx_vector_int_new = c.mlx_vector_int_new;
pub fn mlx_vector_int_set(vec: *mlx_vector_int, src: mlx_vector_int) c_int {
    return c.mlx_vector_int_set(vec, src);
}
pub const mlx_vector_int_free = c.mlx_vector_int_free;
pub fn mlx_vector_int_new_data(data: *c_int, size: usize) mlx_vector_int {
    return c.mlx_vector_int_new_data(data, size);
}
pub const mlx_vector_int_new_value = c.mlx_vector_int_new_value;
pub fn mlx_vector_int_set_data(vec: *mlx_vector_int, data: *c_int, size: usize) c_int {
    return c.mlx_vector_int_set_data(vec, data, size);
}
pub fn mlx_vector_int_set_value(vec: *mlx_vector_int, val: c_int) c_int {
    return c.mlx_vector_int_set_value(vec, val);
}
pub fn mlx_vector_int_append_data(vec: mlx_vector_int, data: *c_int, size: usize) c_int {
    return c.mlx_vector_int_append_data(vec, data, size);
}
pub const mlx_vector_int_append_value = c.mlx_vector_int_append_value;
pub const mlx_vector_int_size = c.mlx_vector_int_size;
pub fn mlx_vector_int_get(res: *c_int, vec: mlx_vector_int, idx: usize) c_int {
    return c.mlx_vector_int_get(res, vec, idx);
}
pub const mlx_vector_string_new = c.mlx_vector_string_new;
pub fn mlx_vector_string_set(vec: *mlx_vector_string, src: mlx_vector_string) c_int {
    return c.mlx_vector_string_set(vec, src);
}
pub const mlx_vector_string_free = c.mlx_vector_string_free;
pub fn mlx_vector_string_new_data(data: anytype, size: usize) mlx_vector_string {
    return c.mlx_vector_string_new_data(@ptrCast(@constCast(data)), size);
}
pub const mlx_vector_string_new_value = c.mlx_vector_string_new_value;
pub fn mlx_vector_string_set_data(vec: *mlx_vector_string, data: anytype, size: usize) c_int {
    return c.mlx_vector_string_set_data(vec, @ptrCast(@constCast(data)), size);
}
pub fn mlx_vector_string_set_value(vec: *mlx_vector_string, val: [*:0]const u8) c_int {
    return c.mlx_vector_string_set_value(vec, val);
}
pub fn mlx_vector_string_append_data(vec: mlx_vector_string, data: anytype, size: usize) c_int {
    return c.mlx_vector_string_append_data(vec, @ptrCast(@constCast(data)), size);
}
pub const mlx_vector_string_append_value = c.mlx_vector_string_append_value;
pub const mlx_vector_string_size = c.mlx_vector_string_size;
pub fn mlx_vector_string_get(res: anytype, vec: mlx_vector_string, idx: usize) c_int {
    return c.mlx_vector_string_get(@ptrCast(@constCast(res)), vec, idx);
}
pub fn mlx_version(str_: *mlx_string) c_int {
    return c.mlx_version(str_);
}

// Left to the hand-written part (function-pointer or untranslated parameters): _mlx_error, mlx_array_data_bfloat16, mlx_array_data_complex64, mlx_array_data_float16, mlx_array_item_bfloat16, mlx_array_item_complex64, mlx_array_item_float16, mlx_array_new_data_managed, mlx_closure_custom_jvp_new_func, mlx_closure_custom_jvp_new_func_payload, mlx_closure_custom_new_func, mlx_closure_custom_new_func_payload, mlx_closure_custom_vmap_new_func, mlx_closure_custom_vmap_new_func_payload, mlx_closure_kwargs_new_func, mlx_closure_kwargs_new_func_payload, mlx_closure_new_func, mlx_closure_new_unary, mlx_closure_value_and_grad_new_func, mlx_closure_value_and_grad_new_func_payload, mlx_detail_compile, mlx_detail_compile_erase, mlx_export_to_dot, mlx_print_graph, mlx_set_error_handler.
