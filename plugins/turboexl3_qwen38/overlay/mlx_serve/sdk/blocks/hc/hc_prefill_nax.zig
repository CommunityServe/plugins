//! copied_from: src/sdk/blocks/hc/hc_prefill_nax.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The hc prefill read on the tensor units, each launch bit for bit the MLX chain it replaces: the down product
//! (`naxDown`) and the up product with the mix as its epilogue (`naxUpMix`); the down with its silu and the inject
//! product with its gate in one launch (`fuseDown`, `readDown`); the read with the normalized rows never written
//! (`normInv`, `fuseDownNorm`, `rwDown`, `upMixNorm`, `rwUpMix`); the up + mix in registers (`readUpMix`). Each
//! declines (null) where it does not replay MLX's chain, and the caller runs the chain.
const mlx = @import("../../../mlx/ffi.zig");
// ── launches ──
// ── plans: tiles of the hc read's geometry (hc 4 x 2560, rank 320); none changes a bit ──
pub const DownPlan = struct { tm: c_int, wm: c_int, wn: c_int, np: c_int, part: c_int, swz: c_int, bar: c_int };
/// The register-form up + mix's tiles (readUpMix, rwUpMix): a simdgroup owns 16 tm rows x 32 columns of every stream.
pub const RegPlan = struct { tm: c_int, wm: c_int, wn: c_int, swz: c_int, bar: c_int };
pub const RwPlan = struct { tm: c_int, wm: c_int, wn: c_int, swz: c_int, sd: c_int };
// ── dispatches (the wrappers' and the contract's) ──
// ── which read takes which launches ──
// ── the read's tables and outputs ──
/// The chain's silu and inject gate of every bf16 pattern ([65536] bf16 each, indexed by the input's bits), built from
/// the trunk's own closures; cached per (silu compiled, gate compiled) for the process (2 x 128 KB a pair).
pub const FuseTables = struct { silu: mlx.mlx_array, gate: mlx.mlx_array };
/// The fused down's outputs: act [B, S, R] (the chain's `act`) and inj [B, S, HC, 1] (its `inj`; null-ctx without an inject).
pub const FuseOut = struct {
    act: mlx.mlx_array,
    inj: mlx.mlx_array,
};
/// normInv's outputs: the stream the read continues with (the written stream with a pending write, else a reference to
/// `x`) and the norm's f32 row scales [B S HC] (row = token HC + stream).
pub const NormInv = struct {
    stream: mlx.mlx_array,
    invs: mlx.mlx_array,
};
// ── the launches ──
// ── tests ──
