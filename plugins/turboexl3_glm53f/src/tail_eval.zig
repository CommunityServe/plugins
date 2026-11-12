//! `TURBOEXL3_GLM53F_PF_TAIL_EVAL` (on by default, exact; `0` off): a prompt layer's tail (the routed call's combine, the shared
//! expert and the hyper-connection's stream update) handed to the GPU without the host's wait at the layer's end, where
//! the forward's cadence eval (`Forward.eval_every`, every 4 layers and the last; TURBOEXL3_GLM53F_ASYNC_EVAL's) is not
//! due (patches/turboexl3-model.patch in the copy's table_forward.zig `hidden` and `hiddenSplit`).
//!
//! Between two cadence evals a layer's tail is committed by the next layer's first eval: a KDA layer's routing barrier,
//! or under TURBOEXL3_GLM53F_DSA_HOLD a DSA layer's q_abs, after that layer's start and its attention graph. The routed
//! call's last chunks have completed by then (the combine reads them), so the GPU waits for the host there: g53hmt
//! (PF_INT8=4's host-marks probe, a 16K prompt) put 0.77 ms (median) of GPU idle at each KDA layer's barrier and a DSA
//! layer's start behind its cache trim (TURBOEXL3_GLM53F_PF_RUN_TRIM leaves about 1.5 GB of it). With the switch the
//! tail runs while the host builds the next layer.
//!
//! The same arrays by the same kernels in the same order on the same stream: an eval only cuts the graph where the
//! next one would have, so nothing served moves. Not billed: as ASYNC_EVAL's, a tail's buffers can be held while the
//! next layer's are made (at a 16K prompt about 1 GiB at most).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const log = @import("mlx_serve/sdk/util/log.zig");

/// `TURBOEXL3_GLM53F_PF_TAIL_EVAL=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (an A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_TAIL_EVAL=1: a prompt layer's tail (the routed call's combine, the shared expert and the stream update) handed to the GPU without the host's wait at the layer's end where the cadence's eval is not due, so it runs while the host builds the next layer; the same arrays by the same kernels in the same order\n";
/// The tails committed since the load (the engaged line prints at the first).
pub var runs: u64 = 0;

/// Whether layer `li`'s end commits its tail: the switch on, a forward with a cadence (`eval_every` > 0), and not a
/// layer where the cadence's own eval is due (every `eval_every`th and the forward's last).
pub fn due(eval_every: u32, li: usize, n_layers: usize) bool {
    if (!on or eval_every == 0) return false;
    const next = li + 1;
    return next % eval_every != 0 and next != n_layers;
}

/// The layer's output stream `h` handed to the GPU without the host's wait (its graph, the tail, encoded now).
pub fn commit(h: mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{h}, 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
    if (runs == 0) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PF_TAIL_EVAL engaged: a prompt layer's tail handed to the GPU at the layer's end\n", .{});
    runs += 1;
}

// ── tests (MLX-free) ──

test "turboexl3_glm53f stream tail eval: off, no layer commits; on, every layer between the cadence's evals does (GLM's 45 layers at eval_every 4: 33 of them), none at a cadence layer, the last, or without a cadence" {
    defer on = false;
    try std.testing.expect(!due(4, 0, 45));
    on = true;
    var n: usize = 0;
    for (0..45) |li| {
        if (due(4, li, 45)) {
            n += 1;
            try std.testing.expect((li + 1) % 4 != 0 and li + 1 != 45);
        }
    }
    try std.testing.expectEqual(@as(usize, 45 - 11 - 1), n);
    try std.testing.expect(!due(4, 3, 45) and !due(4, 43, 45) and !due(4, 44, 45) and due(4, 0, 45) and due(4, 42, 45));
    try std.testing.expect(!due(0, 0, 45));
}
