//! `TURBOEXL3_GLM53F_ASYNC_EVAL=1` (opt-in, exact): a prompt forward's eval at the host's cadence
//! (`Forward.eval_every`: the stream and every recurrent conv window the slots hold, after every 4 layers and the last,
//! the copy's `Slots.evalWith`) handed to the GPU without the host's wait (`mlx_async_eval`: the sdk base's
//! `pipeline_every` form), through patches/turboexl3-model.patch in the copy's table_forward.zig and
//! slots.zig. The same arrays are evaluated by the same kernels in the same order on the same stream, so nothing served
//! moves; the host builds the next sublayer's graph while the GPU runs the layer's tail, where the blocking eval left
//! the GPU idle while it did (about 8 ms an eval at a 16K prompt, 12 evals of its 45 layers).
//!
//! The eval still cuts the graph where it did, so the graph's bound holds. What moves is when a tail's buffers are
//! freed: they are held by the tail's command buffers until they complete, and the next sublayer's arrays are allocated
//! when its first eval encodes them, which can be while the tail still runs. That overlap is not billed: at a 16K
//! prompt it is at most the tail's own transients beside the stream it replaces (the old [16K, 4, 4096] bf16 stream,
//! 0.54 GB, and the shared expert's and the combine's outputs), about 1 GiB, which MLX's buffer cache can also absorb.
const std = @import("std");
const log = @import("mlx_serve/sdk/util/log.zig");

/// `TURBOEXL3_GLM53F_ASYNC_EVAL=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ASYNC_EVAL=1: a prompt forward's eval every 4 layers and after the last (the stream and the slots' conv windows) handed to the GPU without the host's wait (mlx_async_eval), the same arrays by the same kernels in the same order; not billed: a tail's buffers can be held while the next sublayer's are made (at a 16K prompt about 1 GiB at most)\n";
/// Async evals since the load (the engaged line prints at the first).
pub var runs: u64 = 0;

/// Whether a prompt forward's cadence eval skips the host's wait (the copy's table_forward.zig asks at each), counted,
/// the engaged line at the first.
pub fn take() bool {
    if (!on) return false;
    if (runs == 0) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ASYNC_EVAL engaged: a prompt forward's cadence eval handed to the GPU without the host's wait\n", .{});
    runs += 1;
    return true;
}

// ── tests (MLX-free) ──

test "turboexl3_glm53f stream async eval: off, every cadence eval waits (nothing counted); on, none does, each counted" {
    try std.testing.expect(!on);
    defer {
        on = false;
        runs = 0;
    }
    try std.testing.expect(!take() and !take() and runs == 0);
    on = true;
    try std.testing.expect(take() and take() and runs == 2);
}
