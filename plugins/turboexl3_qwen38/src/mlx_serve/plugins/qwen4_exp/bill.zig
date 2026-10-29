//! copied_from: src/plugins/qwen4_exp/bill.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's memory terms: the QSA indexer's per-slot ring, per-token history and score bank, the prefill's QSA
//! transient, the batched tick's effective KV and the SSM checkpoint, each the engine's own arithmetic.

const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const gather_mod = @import("qsa/gather.zig");
const score_mod = @import("../../sdk/blocks/attention/sparse_block/score.zig");
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const model_mod = @import("../../sdk/config/model.zig");
const qconfig = @import("config.zig");
const toModelConfig = @import("model_config.zig").toModelConfig;

const bills = @import("../../engines/mlx/admission/bills.zig");
const prefillTransientReserve = bills.prefillTransientReserve;
const resolvePrefillChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").resolvePrefillChunk;
const verify_qmm_nax = @import("../../sdk/blocks/linear/verify_qmm_nax.zig");

/// Bytes per (query, key) the QSA prefill holds for ONE live layer past the
/// indexer budget: the `[S, kv]` bool mask, its additive copy for the sdpa
/// arm, and the f32 indexer scores over kv/4 blocks (4 bytes/block = 1/key).
const QSA_MASK_BYTES_PER_KEY: u64 = 4;

/// qwen4_exp sparse attention materializes a `[S, kv]` mask per layer that
/// no envelope term models: 4096 x 25k keys = 410 MB, and a tight box met it
/// as a Metal OOM at 25k+. Zero for every arch without an indexer. The sizer
/// bills it at kv = chunk (no prompt yet); the admission guard at the real
/// prompt length.
pub fn qsaMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) u64 {
    if (config.indexer_budget == 0) return 0;
    // Prefill widths gather by block index (row-chunked score sheet, no
    // [S, kv] mask); decode/verify widths still build the dense mask.
    if (fwd >= attn256.FUSED256_MIN_Q_LEN and gather_mod.qsaGatherEnabled())
        return score_mod.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return QSA_MASK_BYTES_PER_KEY * fwd * kv * 5 / 4;
}

/// The engine's config of a parsed pack (no folder: nothing allocated).
fn engineConfig(p: *const abi.Parsed) model_mod.ModelConfig {
    return toModelConfig(std.heap.c_allocator, qconfig.of(p), null) catch unreachable;
}

pub fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const cfg = engineConfig(p);
    out.* = .{
        .fixed_slot_bytes = cfg.qsaRingBytes(),
        .state_per_token = bills.statePerTokenBilled(&cfg),
        .ssm_checkpoint = cfg.ssmCheckpointBytes(),
        .prefill_bytes = &prefillBytes,
        .batched_kv_len = &batchedKvLen,
    };
}

/// The QSA transient a prefill chunk adds beside the dense terms, at the real prompt length.
fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, _: u8, _: u64) callconv(.c) u64 {
    const cfg = engineConfig(p);
    return qsaMaskBytes(&cfg, @min(chunk, @max(seq, 1)), seq);
}

fn batchedKvLen(p: *const abi.Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32 {
    const cfg = engineConfig(p);
    return cfg.batchedEffectiveKvLen(kv, gather_on, min);
}

// ── tests ──
