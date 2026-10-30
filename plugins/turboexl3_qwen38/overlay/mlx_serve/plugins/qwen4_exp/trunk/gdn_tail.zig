//! copied_from: src/plugins/qwen4_exp/trunk/gdn_tail.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The GDN layer's tail on the trunk format's linears (prefill), through the quant table's `linear_*` entries:
//! - GDN-OUT: in_proj_z stops at its inner product and one launch (turboexl3's `gdnOutPrep`) runs z's finish, the
//!   norm-gate and out_proj's prepare per (row, head), so neither z nor the gated rows are written; out_proj continues
//!   from the prepared rows. Every element is the chain's expressions in its order with its roundings.
//! - The overlap: the chunk's recurrence runs piece after piece on the main stream, each from the state the piece before
//!   left (the f32 state crosses a piece boundary unrounded, as at a chunk boundary), while piece i's tail runs on the
//!   side stream; out_proj's weight is made once and one finish reads every piece's inner product. The tail launches
//!   are per row with the whole chunk's template arguments, and MLX's GEMM runs the whole call's kernel at every piece's
//!   row count (`gemmSame`), so every row keeps its bytes.

const mlx = @import("../../../mlx/ffi.zig");

// ── the format's parts (`abi.QuantTable`'s `linear_*` entries) ──

// ── GDN-OUT ──


pub const OutProj = struct { qkv: mlx.mlx_array, z: mlx.mlx_array, z_inner: mlx.mlx_array };

// ── the overlap ──

/// The most pieces: the format's finish over pieces reads up to four, and in_proj_z's GEMM shares the side stream.
pub const MAX_PIECES: usize = 4;

/// The kernel MLX's matmul runs for a 2-D row-contiguous f16 / bf16 [m, k] x [k, n] (no transposes, m and n >= 2) on a
/// device of class `devc` (the last letter of its architecture), with or without NAX: split K, else the regular NAX kernel
/// or steel kernel and its tile. Two row counts with one key run one kernel, whose K loop gives every row the same
/// summation order (the M tail's bounds-checked loads zero the rows past M; the swizzle only places tiles).
pub const MlxGemm = struct { split: bool = false, nax: bool = false, bm: u16 = 0, bn: u16 = 0, bk: u16 = 0, wm: u8 = 0, wn: u8 = 0 };

/// A chunk's pieces: piece i is rows [start[i], start[i + 1]).
pub const Plan = struct { n: usize, start: [MAX_PIECES + 1]c_int };


// ── tests ──
