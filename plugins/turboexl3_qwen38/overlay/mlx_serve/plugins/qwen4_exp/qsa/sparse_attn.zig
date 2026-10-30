//! copied_from: src/plugins/qwen4_exp/qsa/sparse_attn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA sparse attention over the selected blocks (`qsaSparseAttn`): the tile, row and slot kernels and their merge.

const mlx = @import("../../../mlx/ffi.zig");

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──
//
// At 2 <= S < 16 the QSA path spent ~50 dependent ops per layer and attended a fixed union
// of the S rows' selections under a mask (6x the MACs at S=6, plus a 12.6 MB slab). This is
// one dispatch: `msv_attn_qsa256` verbatim with the device K/V reads replaced by an in-kernel
// affine unpack (`qkv_attn_dec`'s dense packing). Each row sees its own blocks then its own
// ragged tail, so causality is automatic and the INT_MAX sentinels are never read. A bf16
// cache runs the same split pass with `msv_attn_qsa256`'s plain uint4 loads: the source is
// built from shared pieces, and only the prologue, the K/V loads and their bases differ per variant.






/// The split pass's partials of a one-view call (each the holder's): pacc f32 [Hq, S, NSPLIT, 256], pml f32 [Hq, S,
/// NSPLIT, 2].
pub const SplitParts = struct {
    pacc: mlx.mlx_array = .{ .ctx = null },
    pml: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *SplitParts) void {
        inline for (.{ &self.pacc, &self.pml }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

// ── tests ──

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──
