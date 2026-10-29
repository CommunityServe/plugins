//! copied_from: src/sdk/blocks/ssm/mamba2_decode.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Fused Mamba2 mixer step for any window (decode, spec verify, prefill): the
//! conv window, depthwise conv + bias + SiLU, dt softplus/clamp, the SSM
//! recurrence, the D skip and the SiLU(z) gate run as ONE dispatch over R
//! consecutive rows; the grouped RMS norm is a second. Same structure as
//! TensorFold's `mamba_step`/`group_norm` (MIT), arithmetic kept to the op
//! chain's: f32 math, bf16 where the chain stores bf16. The SSM state stays
//! f32; `y` comes back in x's dtype.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

/// Rows of B/C the step kernel stages per threadgroup pass; a wider window
/// loops over passes. Also the decode/verify width `add_norm` keys on.
pub const MAX_ROWS: c_int = 16;
const TGY: c_int = 8;
/// Threadgroup memory is an occupancy decision: the step kernel keeps B/C for
/// a row BUCKET, never the 16-row pass on a one-row decode step.
const TG_BYTES_MAX: usize = 16 * 1024;

// grid (32, Dh, B*H), threadgroup (32, TGY, 1): one simdgroup per (b, h, d),
// each lane owning Ds/32 state columns. B/C of the head's group are conv'd
// once per threadgroup into shared memory.
pub const STEP_SOURCE = kernelSource("kernels/ssm/mamba2_decode_step.metal");

// grid (GS/4, NG, rows): one threadgroup per (row, group), 4 elements a thread.
pub const GNORM_SOURCE = kernelSource("kernels/ssm/mamba2_decode_gnorm.metal");

pub const Geometry = struct { batch: c_int, rows: c_int, heads: c_int, groups: c_int, head_dim: c_int, state: c_int, conv_kernel: c_int, proj: c_int };

pub const Inputs = struct {
    proj: mlx.mlx_array, // [B,R,PROJ] in_proj output: z | x B C | dt
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
    ssm_state: mlx.mlx_array, // [B,H,Dh,Ds] f32
    conv_w: mlx.mlx_array, // [CD,KC,1]
    conv_b: mlx.mlx_array, // [CD]
    A_log: mlx.mlx_array, // [H]
    dt_bias: mlx.mlx_array, // [H]
    D: mlx.mlx_array, // [H]
    dt_lo: mlx.mlx_array, // 0-dim f32 (MLX binds a scalar input by value)
    dt_hi: mlx.mlx_array, // 0-dim f32
    rows: mlx.mlx_array, // 0-dim int32
    all_rows: bool, // state after EVERY row (spec capture) or only the last
};

pub const Outputs = struct {
    y: mlx.mlx_array, // [B,R,H*Dh] gated, x dtype
    ssm_state: mlx.mlx_array, // [R,B,H,Dh,Ds] f32 when all_rows, else [1,B,H,Dh,Ds]
    conv_state: mlx.mlx_array, // [B,KC-1,CD]
};

var step_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gnorm_kernel: ?mlx.mlx_fast_metal_kernel = null;
const StepKey = struct { g: Geometry, dt: mlx.mlx_dtype, all_rows: bool };
var step_key: ?StepKey = null;
var step_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
const GnormKey = struct { rows: c_int, xd: c_int, groups: c_int, dt: mlx.mlx_dtype };
var gnorm_key: ?GnormKey = null;
var gnorm_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
