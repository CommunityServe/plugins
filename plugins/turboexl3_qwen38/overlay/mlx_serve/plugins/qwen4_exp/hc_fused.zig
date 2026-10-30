//! copied_from: src/plugins/qwen4_exp/hc_fused.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Hyper-connection reads and writes fused into custom kernels (the deferred write folded into the next read, the
//! prefill-width pending write), with their kill switches.

const mlx = @import("../../mlx/ffi.zig");

// ── qwen4_exp fused hyper-connection READ at decode and verify width ──
// The read is two launches (down, then up/mix) in hc_decode2.zig for 1..HC_FUSED_MAX_ROWS rows; anything outside its
// geometry keeps the unfused chain.

pub const HcFusedOut = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// A deferred `hcWrite`: the next read's N kernel applies `stream + out*inj`
/// itself (one dispatch fewer per block). Handles are retained copies.
pub const HcPending = struct {
    out: mlx.mlx_array, // [B,S,hidden]
    inj: mlx.mlx_array, // [B,S,hc,1]
    pub fn deinit(self: *HcPending) void {
        _ = mlx.mlx_array_free(self.out);
        _ = mlx.mlx_array_free(self.inj);
    }
};

/// Fused decode-width hyper-connection read over `batch*seq` rows (1..
/// HC_FUSED_MAX_ROWS: decode, verify widths, batched slots — rows are
/// independent, the grid carries them, weights are read once per row).
/// `x` holds `batch*seq*hc*hidden` elements; `iw` is the row-major dense
/// `[hc*hidden, hc]` inject weight or null-ctx (mixer). With `pend`, the read
/// first applies that pending write and returns the written stream in
/// `.stream` (null-ctx otherwise). Returns `mixed [B,S,hidden]` +
/// `inj [B,S,hc,1]`, or null when the geometry/quant is outside the kernel
/// (caller keeps the chain).
pub const HC_FUSED_MAX_ROWS: c_int = 16;

// ── tests ──
