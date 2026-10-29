//! copied_from: src/sdk/blocks/norms/residual_norm.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/norms/residual_norm.
//! Owns add_norm_max_rows, RESIDUAL_NORM_HEADER, RESIDUAL_NORM_SOURCE, residual_norm_kernel, residual_norm_engaged, add_rmsnorm_override, and 12 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mamba2_decode = @import("../ssm/mamba2_decode.zig");
const add_norm = @import("add_norm.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;

pub var add_norm_max_rows: c_int = mamba2_decode.MAX_ROWS; // test seam

// ── Fused residual add + RMSNorm ──

/// One dispatch for the residual seam between blocks:
///   n1  = PRE1 ? w1 * rms(b1, eps1) : b1
///   n2  = B2 ? (PRE2 ? w2 * rms(b2, eps2) : b2) : -
///   c   = B2 ? n1 + n2 : n1
///   c   = PREC ? wc * rms(c, epsc) : c
///   res = (a + c) [* scale]
///   out = POST ? wp * rms(res, eps) : -,  out2 = POST2 ? wp2 * rms(res, eps) : -
/// Every op rounds to T exactly where MLX's separate kernels do, and each
/// reduction keeps MLX's `rms_single_row` geometry (ceil(AXIS/4) threads,
/// four consecutive elements each, simd_sum, a zeroed partials plane, one
/// simd_sum over it) so the tree is MLX's tree, not merely its total.
/// `metal::precise::rsqrt` as upstream.
pub const RESIDUAL_NORM_HEADER = kernelSource("kernels/norms/residual_norm_header.metal");
pub const RESIDUAL_NORM_SOURCE = kernelSource("kernels/norms/residual_norm.metal");

var residual_norm_kernel: ?mlx.mlx_fast_metal_kernel = null;
var residual_norm_engaged: bool = false;

/// The stages of `fusedResidualNorm`; a null-ctx weight switches its stage
/// off. Eps arrays are 0-dim f32 scalars; `eps` serves both post-norms.
pub const ResidualNormOpts = struct {
    b2: mlx.mlx_array = .{ .ctx = null },
    w1: mlx.mlx_array = .{ .ctx = null },
    eps1: mlx.mlx_array = .{ .ctx = null },
    w2: mlx.mlx_array = .{ .ctx = null },
    eps2: mlx.mlx_array = .{ .ctx = null },
    wc: mlx.mlx_array = .{ .ctx = null },
    epsc: mlx.mlx_array = .{ .ctx = null },
    scale: mlx.mlx_array = .{ .ctx = null },
    w_post: mlx.mlx_array = .{ .ctx = null },
    w_post2: mlx.mlx_array = .{ .ctx = null },
    eps: mlx.mlx_array = .{ .ctx = null },
};

const ResidualNormCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype, flags: u8 };
var residual_norm_cfg: [8]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var residual_norm_cfg_key: [8]ResidualNormCfgKey = @splat(std.mem.zeroes(ResidualNormCfgKey));
var residual_norm_cfg_next: usize = 0;

pub const AddNormResult = struct { sum: mlx.mlx_array, normed: mlx.mlx_array, normed2: mlx.mlx_array = .{ .ctx = null } };

// ── tests ──
