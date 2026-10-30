//! This plugin's own, in place of mlx-serve's `src/engines/mlx/transformer.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the engine's generic transformer is not this plugin's; the copies read only these of it.
//! The overrides are mlx-serve's test seams, never set here: null is the served path.


// What the copied tests (`zig build gpu-test`) read of it.

/// The engine's model as far as `spec/mtp_test.zig`'s rerank fixture builds one: a quantized `lm_head` and the config
/// `headQuantParams` reads, the whole of what the rerank scheme reads of its target (the served path's target is the
/// trunk). The served path never makes one.
pub const Transformer = struct {
    rht: ?*const anyopaque = null,
    config: @import("../../sdk/config/model.zig").ModelConfig = .{},
    s: mlx.mlx_stream = .{ .ctx = null },
    lm_head_w: mlx.mlx_array = .{ .ctx = null },
    lm_head_s: mlx.mlx_array = .{ .ctx = null },
    lm_head_b: mlx.mlx_array = .{ .ctx = null },
};

const mlx = @import("../../mlx/ffi.zig");
