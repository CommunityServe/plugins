//! The plugin's one seam for MLX custom Metal kernels (every arch: DeepSeek V4.1's EXL3 texts, MiMo's decode and prompt
//! texts, GLM-5.3's and GLM-5.3-Flash's): mlx-c's `mlx_fast_metal_kernel_new` / `_free` / `_apply`, same signatures and
//! contracts, through the plugin's shim (csrc/mlx_launch_cache.inc). A kernel made here keeps a launch cache: each
//! variant (template arguments, input dtypes and address classes, output dtypes) runs MLX's own apply until MLX has
//! built its library, then launches MLX's cached pipeline directly, without regenerating and rehashing the kernel's
//! source on every call (the pinned MLX's per-launch cost). Same pipeline, same arguments: bit-identical results.
//! Grid, threadgroup and output shapes are read from the config on every launch, as MLX's; a config with an init value
//! always runs MLX's apply.

const mlx = @import("sdk").mlx;

extern "c" fn mlxlc_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel;
extern "c" fn mlxlc_kernel_free(k: mlx.mlx_fast_metal_kernel) void;
extern "c" fn mlxlc_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int;

pub fn mlx_fast_metal_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel {
    return mlxlc_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
}

/// The cache's eviction, then mlx-c's free.
pub fn mlx_fast_metal_kernel_free(k: mlx.mlx_fast_metal_kernel) void {
    mlxlc_kernel_free(k);
}

pub fn mlx_fast_metal_kernel_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int {
    return mlxlc_apply(outputs, k, inputs, config, stream);
}

// ---- CPU tests (no device: graphs are built on an explicit GPU stream and never evaluated) ----

const std = @import("std");
const testing = std.testing;

extern "c" fn mlxlc_test_config(config: mlx.mlx_fast_metal_kernel_config, out: *[9]c_int) void;
extern "c" fn mlxlc_test_bench(k: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, n_in: c_int, ndims: [*]const c_int, shapes: [*]const c_int, dtypes: [*]const mlx.mlx_dtype, iters: c_int, out: *[3]i64) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;
