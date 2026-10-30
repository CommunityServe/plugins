//! copied_from: src/sdk/blocks/gdn/gated_delta_net.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/gdn/gated_delta_net.
//! Owns LinearAttnWeights, GdnProj, and 1 more.
const mlx = @import("../../../mlx/ffi.zig");
const QW = @import("../linear/fused_rows.zig").QW;
const FusedRows = @import("../linear/fused_rows.zig").FusedRows;
pub const LinearAttnWeights = struct {
    // For separate projections (qwen3_5_moe): qkv=QKV, z=Z, a=A, b=B
    // For combined projections (qwen3_next): qkv=QKVZ, b=BA, z/a unused
    combined_proj: bool = false,
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
    /// Row-joined [qkv | z | a | b] (separate projections only); the fields
    /// above are views of it once built (fuseRowGroup).
    in: FusedRows = .{},
    /// q/k/v quantized at different widths cannot join: `qkv_w` is null and each projects alone.
    qkv_parts: [3]QW = .{ .{}, .{}, .{} },
    /// bf16 in_proj_a / in_proj_b joined at load (qwen4_exp `trunk/gdn_ab.zig`): `a_w` / `b_w` are views of it.
    ab: GdnAb = .{},
};
/// A GDN layer's joined bf16 a / b: `w` the transposed view [hidden, n] of the row-major buffer [a | 0 | b] (null-ctx:
/// not joined), b's first column `b_off`, the parts' widths.
pub const GdnAb = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    n: c_int = 0,
    b_off: c_int = 0,
    widths: [2]c_int = .{ 0, 0 },
};
// ── GatedDeltaNet (linear attention layers) ──
