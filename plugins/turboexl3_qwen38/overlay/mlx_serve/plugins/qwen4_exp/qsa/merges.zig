//! copied_from: src/plugins/qwen4_exp/qsa/merges.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The QSA indexer's decode-row merges, each the bytes of the chain it replaces: the indexer epilogue (the indexer
//! projection's finish normalizes and rotates the index queries and pools the block key the call completes, through the
//! trunk format's `indexer` entry), the placed row's mask (a placed MTP draft row's bounded select and its key
//! visibility in one mask launch) and the ring rebuild (the key ring's in one launch).

const mlx = @import("../../../mlx/ffi.zig");

// ── the indexer epilogue ──

/// The indexer epilogue's results for one layer (each the holder's): qk [1, S, (n + 1) 128] as `qmatmul` gives it, the
/// index queries normed and rotated [1, n, S, 128] (null-ctx: the chain builds them), the one pooled block key [1, 1,
/// 128] (null-ctx: the chain pools it), and the pooled block count its plan read.
pub const IdxMerge = struct {
    qk: mlx.mlx_array,
    q_rope: mlx.mlx_array = .{ .ctx = null },
    pooled: mlx.mlx_array = .{ .ctx = null },
    pool_nb_cached: c_int = -1,

    pub fn deinit(self: *IdxMerge) void {
        inline for (.{ &self.qk, &self.q_rope, &self.pooled }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// The one block a call completes, as `qsaMaskFromQk` plans it: `ln` ring rows (the history's leftover), then the
/// call's first ratio - ln rows. Null where the call completes no block, more than one, or the chain would raise its
/// gap error.
pub const PoolPlan = struct { nb_cached: c_int, ln: c_int, held: c_int };

// ── the placed row's mask ──

// ── the ring rebuild ──


// ── tests ──
