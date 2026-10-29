//! copied_from: src/sdk/bases/blocks/row_exact.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The row-exact lanes: while a DFlash drafter is bound (`Forward.row_exact`) and the model's weights fit them, every
//! projection, the MoE router and experts, and attention over a short window read kernels whose bits for a row do not
//! depend on how many rows share the call, so a verify row reproduces serial decode (`Caps.exact_drafted`).
//! copied_from: sdk/blocks/linear/qmatmul.zig (`rowExactQmm`), plugins/nemotron_h/moe.zig (the row-exact arms),
//! sdk/blocks/attention/hybrid_attn.zig (the row-by-row window), sdk/blocks/attention/gated_full.zig (the
//! `lane_attn`/`row_attn` arms), sdk/blocks/mlp/dense.zig (`rowGroupServes`).

const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const Linear = @import("linear.zig").Linear;
const rowqmv = @import("../../blocks/qmm/rowqmv.zig");
const lane_qmm = @import("../../blocks/qmm/lane_qmm.zig");
const simd_qmm = @import("../../blocks/qmm/simd_qmm.zig");
const sdpa = @import("../../blocks/attention/sdpa.zig");
const row_attn = @import("../../blocks/attention/row_attn.zig");
const lane_attn = @import("../../blocks/attention/lane_attn.zig");
const FUSED_ROWS_MAX_M = @import("../../blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;

const Array = ops.Array;
const Graph = ops.Graph;

/// `off`, or the kernel family a model's projections take: `rows` = `rowqmv`'s per-row matvecs (Nemotron-H's thin
/// MoE trunk), `mma` = `lane_qmm` (NAX) then `simd_qmm`, whose multi-row MMA reads each weight once (dense Qwen3.8).
pub const Lane = enum(u8) { off, rows, mma };

/// The widest window the row kernels and the row-by-row attention take.
pub const MAX_ROWS = rowqmv.MAX_ROWS;
