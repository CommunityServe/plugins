//! copied_from: src/plugins/qwen4_exp/forward.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The qwen4_exp trunk forward (`forwardQwen4With`): embeddings tiled into hyper-connection streams, the n-gram PLE,
//! GDN and QSA attention layers under hc reads and writes, the final mixer; also the grouped verify rows and a
//! pipeline rank's stream forward. Env-gated fixture tests against the HF reference live here.
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("trunk.zig").Trunk;
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
