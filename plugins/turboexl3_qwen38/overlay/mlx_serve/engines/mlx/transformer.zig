//! This plugin's own, in place of mlx-serve's `src/engines/mlx/transformer.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the engine's generic transformer is not this plugin's; the copies read only these of it.
//! The overrides are mlx-serve's test seams, never set here: null is the served path.

pub var qsa_gather_override: ?bool = null;
pub var qsa_gather_min_kv_override: ?c_int = null;
pub var qsa_decode_gather_override: ?bool = null;
pub var qsa_batched_gather_override: ?bool = null;
pub var qsa_history_share_override: ?bool = null;
pub var qsa_attn_kernel_override: ?bool = null;

pub const QSA_RING_ROWS = @import("../../plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;

pub const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;

/// A weight struct field a loader fills by name.
pub const WeightField = struct { name: [:0]const u8, type: type };

pub const qsaHistoryShareEnabled = @import("../../plugins/qwen4_exp/qsa/gather.zig").qsaHistoryShareEnabled;

// What the copied tests (`zig build gpu-test`) read of it.

/// The hc decode read, as mlx-serve's transformer re-exports it (`sdk/blocks/hc/hc_decode2_test.zig`).
pub const HcPending = @import("../../plugins/qwen4_exp/hc_fused.zig").HcPending;
pub const hcReadFused = @import("../../plugins/qwen4_exp/hc_fused.zig").hcReadFused;

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
