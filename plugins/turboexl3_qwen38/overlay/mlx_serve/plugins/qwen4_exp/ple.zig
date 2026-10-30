//! copied_from: src/plugins/qwen4_exp/ple.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The n-gram PLE layer: the host gather or GPU kernel into a deferred leaf (`ple_defer`, `flushDeferredPle`), the
//! gated value injection and its dilated causal conv, the per-slot history, and the spec capture and rollback.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const scalarOf = @import("../../engines/mlx/load/weights.zig").scalarOf;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("../../engines/mlx/ssm/entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const Trunk = @import("trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const Qwen4TestSlot = @import("forward.zig").Qwen4TestSlot;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const PleWeights = @import("weights.zig").PleWeights;
const cachedScalarInt = @import("../../sdk/blocks/linear/verify_qmm.zig").cachedScalarInt;
const sigmoidTableFor = @import("../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const ShapeKey = @import("../../sdk/blocks/moe/router_kernel.zig").ShapeKey;

pub const PlePending = struct {
    emb: mlx.mlx_array,
    token_ids: mlx.mlx_array,
    entry: *SSMCacheEntry,
    layer: usize,
    seq_len: c_int,
    /// Whether the gather owes `entry.spec_ple_tokens` the verify history.
    /// Decided (and `spec_ple_len` claimed) at BUILD time: the PLE layer's own
    /// conv capture, later in the same forward, keys on `spec_ple_len` while
    /// this gather has not run yet, and `spec_capture_ssm` is already back to
    /// false by the time the flush happens.
    capture: bool,
    /// GPU arm: `emb` is the kernel's output, not a leaf; only the history is owed.
    gpu: bool = false,
};


// ── the PLE layer's elementwise tail in four launches ──
//
// Between its MLX projections, norms, gate sum and conv, the chain is 18 elementwise launches on the layer's critical
// path: the group norms' three weight products and the keys x queries product, the gate's seven ops and its value
// product, the conv input's concat, silu's sigmoid and product, the conv branch's add (and the call site's residual
// add). Four launches carry them with every op's arithmetic and rounding unchanged: mlx Multiply / Add on two T values,
// the concat's copy, and the two unary chains (the gate's seven ops, MLX's sigmoid) read from tables of MLX's own
// results over every 16-bit pattern, so no transcendental is re-spelled. Pack-independent: the same bf16 operands on
// any qwen4 pack, so it serves every one.




// ── tests ──

// ── qwen4 PLE: the GPU arm against the host gather (synthetic table, no model) ──
