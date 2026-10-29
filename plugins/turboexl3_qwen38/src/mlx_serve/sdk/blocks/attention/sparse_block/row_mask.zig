//! copied_from: src/sdk/blocks/attention/sparse_block/row_mask.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The dense key-visibility mask of one QSA decode row, written by one kernel. The op chain it
//! replaces (compare, select, scatter, broadcast, pad, three more compares) is about eight dependent
//! launches per attention layer; the mask is just "this token's block is in the sorted selection, or
//! the token is in the incomplete tail" (every key of a one-row step is causal-visible).
const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../util/log.zig");

// grid (kv), one thread a token. `blocks` is sorted ascending with INT_MAX sentinels, so a binary
// search finds a token's block; a sentinel never equals a block id.
pub const SOURCE = kernelSource("kernels/attention/qsa_mask.metal");

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
/// Test seam: false keeps the op chain.
pub var override: ?bool = null;

fn enabled() bool {
    return override orelse true;
}

/// `[1, 1, 1, kv]` bool mask for one decode row over `blocks` `[1, 1, kb]` int32, or null outside
/// the kernel's set (the caller keeps the op chain).
pub fn rowMask(s: mlx.mlx_stream, blocks: mlx.mlx_array, kv: c_int, ratio: c_int) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    const bs = mlx.getShape(blocks);
    if (bs.len != 3 or bs[0] != 1 or bs[1] != 1 or bs[2] < 1 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    if (kv < 1 or ratio < 1) return null;
    if (kernel == null) {
        const ins = [_][*:0]const u8{"blocks"};
        const outs = [_][*:0]const u8{"mask"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_row_mask", in_vec, out_vec, SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    // The mask's width is the step's kv, so the config is built per call.
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 1, kv }, 4, .bool_));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, kv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(kv, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KB", bs[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", ratio));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{blocks}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel.?, v, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    if (!engaged) {
        engaged = true;
        log.info("[qsa] one-row key mask kernel engaged\n", .{});
    }
    return out;
}

const testing = std.testing;
