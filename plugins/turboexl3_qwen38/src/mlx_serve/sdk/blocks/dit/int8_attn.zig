//! copied_from: src/sdk/blocks/dit/int8_attn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Int8 attention for media DiTs on NAX GPUs (default on, `--no-w8a8`, LOSSY), SageAttention's
//! recipe: Q int8 per token, K smoothed (minus its per-channel mean, which no
//! softmax row can see) and int8 per 32-key block, P int8 against each block's
//! row max, V int8 per channel. Both products run on the int8 tensor unit with
//! the online softmax in registers; ~1.3-1.5x MLX's bf16 sdpa at 16k tokens.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");

/// Keys per block: the K scale granularity and the attention kernel's step.
const BK: c_int = 32;
/// Query rows per simdgroup and simdgroups per threadgroup.
const SM: c_int = 16;
const NSG: c_int = 4;

/// One simdgroup per ROWS-row block of a [B, H, L, D] view, SG blocks per threadgroup: int8 codes,
/// row-contiguous, and one fp32 scale (absmax / 127, at least `eps` / 127) per block, less `mean`
/// [B, H, D] when given; DIVIDE rounds x / scale, else x * (1 / scale). Each lane owns D / 32
/// adjacent columns as float4s and stores them as char4. CONTIG (MLX made x row-contiguous, < 2^32
/// elements) loads them as vectors with 32-bit offsets, which per-row Q prep needs to stay at the
/// bandwidth bound; otherwise scalar reads through x's strides.
pub const QUANT_SOURCE = kernelSource("kernels/dit/int8_attn_quant.metal");

/// One simdgroup per (kv head, row) of v [1, KV, L, 128], over its per-channel |max| `vm`.
pub const VPREP_SOURCE = kernelSource("kernels/dit/int8_attn_vprep.metal");

/// One simdgroup per 16 query rows of one head. Element i of an SM x N tensor-op
/// destination sits at column cb + (i&3) + 16*((i>>3)&(N/16-1)) and row
/// rb + 8*slot, slot = ((i>>2)&1) + 2*(i >> (3 + log2(N/16))); S's destination
/// and P's left operand share that layout, so P never leaves registers.
pub const ATTN_SOURCE = kernelSource("kernels/dit/int8_attn.metal");

/// Blocks (one simdgroup each) per threadgroup of `QUANT_SOURCE` when CONTIG; the strided arm
/// keeps one, its long-K prep ran slower in wide threadgroups.
pub const QUANT_SG: c_int = 8;
var quant_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var vprep_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attn_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged_logged = false;

const Out = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

/// Int8 codes of x [B, H, L, D] and its scales [B * H * blocks] per `QUANT_SOURCE`; `contig` lets
/// MLX copy a strided x into rows first, else the kernel reads x's strides. The caller frees both.
pub fn quantRows(x: mlx.mlx_array, rows: c_int, mean: ?mlx.mlx_array, eps: f32, divide: bool, contig_ok: bool, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const contig = contig_ok and mlx.mlx_array_size(x) <= std.math.maxInt(u32);
    const slot = &quant_kernels[@intFromBool(contig)];
    if (slot.* == null) {
        const iv = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "x", "mean", "eps" }, 3);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{ "q", "scales" }, 2);
        defer _ = mlx.mlx_vector_string_free(ov);
        const name = if (contig) "msv_int8_quant_rows_contig" else "msv_int8_quant_rows";
        const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, QUANT_SOURCE, "", contig, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = k;
    }
    const sh = mlx.getShape(x);
    const blocks = sh[0] * sh[1] * @divTrunc(sh[2] + rows - 1, rows);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, 4, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{blocks}, 1, .float32));
    const sg: c_int = if (contig) QUANT_SG else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divTrunc(blocks + sg - 1, sg) * 32 * sg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * sg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", sg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "HAS_MEAN", mean != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "DIVIDE", divide));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "CONTIG", contig));
    const e = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(e);
    const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ x, mean orelse x, e }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, slot.*.?, iv, c, s));
    var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*r, i| try mlx.check(mlx.mlx_vector_array_get(r, ov, i));
    return out;
}

const testing = std.testing;
