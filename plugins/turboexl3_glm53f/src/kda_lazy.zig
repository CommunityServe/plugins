//! `TURBOEXL3_GLM53F_KDA_LAZY_SEQ=1` (opt-in): a KDA layer's verify writes only its final state, and its per-row states
//! (the capture a partial accept's rollback reads) become a launch of their own that MLX runs only when something reads
//! them (patches/turboexl3-model.patch, in kda.zig's capture branch).
//!
//! A verify (kda.zig's `capture` forward: every decode round's trunk forward, an MTP or prompt-lookup verify) runs the
//! stock capture launch, `gated_delta_step_seq_vec`: its output, its final state and the state after every row but the
//! last ([T, B, Hv, Dv, Dk], f32 at GLM-5.3-Flash: 4 MB a row a layer). The host's rollback reads one of those rows on a
//! partial accept (generate/pipeline.zig `rollbackSsmFromCapture`, a lazy slice the next verify reads) and none on a full
//! accept, which keeps the final state. Here the verify runs the stock launch without the per-row stores,
//! `gated_delta_step_vec` (the same text but for them, so the same output and final state bits), and the per-row states
//! come from the stock capture launch as a primitive of its own over the same inputs and initial state, never in an eval
//! set: a full accept frees it unrun; a partial accept's rollback slice runs it when the next verify reads the state, and
//! its rows are the capture launch's bits. A full-accept verify writes 8 MB a layer less at 3 rows; a partial accept's
//! next verify runs the capture launch once more. The initial state is held until the next verify (4 MB a layer).
//!
//! src/kda_gpu_test.zig holds the outputs, windows, states, conv inputs and per-row states of two verifies in a row to
//! KDA_VERIFY's bit for bit at widths 1 to 4, checks the per-row states are not computed when only the rest is read, and
//! holds a verify from a rolled-back state (each partial accept of every width) to KDA_VERIFY's.
const std = @import("std");

/// `TURBOEXL3_GLM53F_KDA_LAZY_SEQ=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_LAZY_SEQ=1: a KDA layer's verify writes its output and final state with the stock launch without the per-row stores, and its per-row states (the rollback's) come from the capture launch as a launch of their own that runs only when a partial accept's rollback reads them: the same bits\n";
/// The verifies that took the lazy branch since the load.
pub var verifies: u64 = 0;

/// The request's line (the plugin's meters): the verifies that took the lazy branch.
pub fn summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_LAZY_SEQ verifies={d}\n", .{verifies}) catch buf[0..0];
}

test "turboexl3_glm53f stream kda_lazy: the summary line counts the lazy verifies" {
    const t = std.testing;
    var buf: [96]u8 = undefined;
    verifies = 68;
    defer verifies = 0;
    try t.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_LAZY_SEQ verifies=68\n", summary(&buf));
}
