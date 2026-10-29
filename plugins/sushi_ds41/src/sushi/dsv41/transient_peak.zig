//! Cut edge: sushi's `SUSHI_DSV41_TRANSIENT_PEAK` diagnostic reads counters of its patched MLX
//! (`mlx_get_transient_peak`, `mlx_reset_transient_peak`) that the host's MLX does not export, so here the phase marks
//! the model and the vision tower place are no-ops. Ours; the signatures are sushi's.

pub const Phase = enum { decode, dspark_draft, dspark_verify, prefill_chunk, vision_encode };

pub fn begin() void {}

/// `rows` is the phase's width: tokens of the pass, or patches of an image.
pub fn end(phase: Phase, rows: usize) void {
    _ = phase;
    _ = rows;
}
