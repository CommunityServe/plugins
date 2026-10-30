//! copied_from: src/plugins/qwen4_exp/forward.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The qwen4_exp trunk forward (`forwardQwen4With`): embeddings tiled into hyper-connection streams, the n-gram PLE,
//! GDN and QSA attention layers under hc reads and writes, the final mixer; also the grouped verify rows and a
//! pipeline rank's stream forward. Env-gated fixture tests against the HF reference live here.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const log = @import("../../sdk/util/log.zig");
const mtp = @import("mtp.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../../engines/mlx/eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../../engines/mlx/eval_cadence.zig").evalCadencePoint;
const ladderStepMulti = @import("../../engines/mlx/eval_cadence.zig").ladderStepMulti;
const prefillEvalCadenceApplies = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceFor;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const Trunk = @import("trunk.zig").Trunk;
const HcRead = @import("hc.zig").HcRead;
const HcPending = @import("hc_fused.zig").HcPending;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;

/// qwen4_exp's batched decode builds each step's graph while the GPU idles, so its ladder ships ON at this stride.
/// Serial decode and other arches keep `decode_async_ladder`.
pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT: u32 = 4;

// ── tests ──

// ── qwen4_exp fixture parity (env-gated: QWEN4_TEST_MODEL + QWEN4_FIXTURE) ──


/// One per-request slot for the qwen4 batched-vs-serial test.
pub const Qwen4TestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,
    /// The MTP head's per-request stash, so a committed-state bar can assert the trunk
    /// verify left it where the solo round did.
    head: Trunk.Qwen4MtpState,

};
