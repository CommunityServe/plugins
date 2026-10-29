//! copied_from: src/sdk/blocks/rope/mrope.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/rope/mrope.
//! Owns MropeCosSin, and 1 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");

// ── Full Attention for MoE models (with optional output gate) ──

pub const MropeCosSin = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

