//! This plugin's own, in place of mlx-serve's `src/engines/mlx/transformer.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the engine's generic transformer is not this plugin's; the copies read only these of it.
//! The overrides are mlx-serve's test seams, never set here: null is the served path.


pub const QSA_RING_ROWS = @import("../../plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;

pub const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;

// What the copied tests (`zig build gpu-test`) read of it.

/// The hc decode read, as mlx-serve's transformer re-exports it (`sdk/blocks/hc/hc_decode2_test.zig`).
pub const HcPending = @import("../../plugins/qwen4_exp/hc_fused.zig").HcPending;

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
