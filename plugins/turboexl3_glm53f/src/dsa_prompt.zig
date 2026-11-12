//! `TURBOEXL3_GLM53F_DSA_ONEP=1` (opt-in): a DSA layer's prompt pass, in the copied glm5_next forward through
//! patches/turboexl3-model.patch.
//!
//! - `DSA_ONEP` (rounding-class): the sparse latent attention with P meeting the values as one fp16 piece, not the
//!   plain form's hi and lo pieces (dsa_nax.zig `sparseLatentOneP`): a third fewer of the kernel's MMAs (it runs at the
//!   NAX's rate, ~57 TFLOPS of MMA, 103 ms a DSA layer at 16K), P rounded once to fp16. In a prompt pass only (rows past
//!   the verify width); src/dsa_gpu_test.zig bounds its difference from the plain form at the real shapes.
//!
//! It runs only where it is written for (else the stock path runs), prints its line at load, and counts its launches
//! (src/dsa_gpu_test.zig reads that it ran).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const dsa = @import("mlx_serve/plugins/glm5_next/dsa_nax.zig");
const naxCheck = @import("mlx_serve/sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const Array = mlx.mlx_array;

// ── the one-P sparse attention ──

/// `TURBOEXL3_GLM53F_DSA_ONEP=1`, set by the plugin at load, before any forward.
pub var onep_on: bool = false;
/// Launches since the load.
pub var onep_runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const onep_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_ONEP=1: a DSA layer's prompt sparse latent attention with P as one fp16 piece against the values (a third fewer MMAs; P rounded once: not the plain form's bits)\n";

/// `dsa_nax.sparseLatentOneP` where the switch is on and the device has NAX; null otherwise (the caller runs the plain
/// form). The caller calls it for a prompt pass's rows only.
pub fn sparseOneP(s: mlx.mlx_stream, q_abs: Array, latent: Array, idx: Array, scale: f32) !?Array {
    if (!onep_on or !naxCheck()) return null;
    const out = (try dsa.sparseLatentOneP(s, q_abs, latent, idx, scale)) orelse return null;
    onep_runs += 1;
    return out;
}
