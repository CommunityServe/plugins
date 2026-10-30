//! copied_from: src/engines/mlx/ssm/entry.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/ssm/entry.
//! Owns SSMCacheEntry, arrayBytes, qsaRawKeyBytes, ssmEntryBytes, SSMCacheEntrySnapshot, ssmSnapshot, and 7 more.

const qwen4_mod = @import("../../../plugins/qwen4_exp/qwen4_exp.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const PersistentSsmGroup = @import("group.zig").PersistentSsmGroup;

// ── SSM Cache (for GatedDeltaNet linear attention layers) ──

pub const SSMCacheEntry = struct {
    conv_state: mlx.mlx_array, // [B, kernel-1, conv_dim]
    ssm_state: mlx.mlx_array, // [B, Hv, Dv, Dk]
    initialized: bool,
    /// PLD spec-decode capture: per-position SSM states recorded by the GDN
    /// verify forward (`Transformer.spec_capture_ssm`). Shape [T, B, Hv, Dv, Dk]
    /// where T = verify length. Lets partial-accept rollback pick the
    /// accepted-position state WITHOUT a re-forward — the verify already ran the
    /// (sequential, expensive on a 48-layer GatedDeltaNet trunk) recurrence, so
    /// re-running it for the accepted prefix is pure waste. Null outside a
    /// capturing verify; freed at the end of every `nextPld` round.
    spec_state_seq: mlx.mlx_array = .{ .ctx = null },
    /// PLD spec-decode capture: the conv1d input `[B, (kernel-1)+T, conv_dim]`
    /// of the verify forward, so the accepted-position `conv_state` is just a
    /// slice (no re-forward). Null outside capture.
    spec_conv_input: mlx.mlx_array = .{ .ctx = null },
    /// A draft-tree verify's input state and per-row recurrence inputs: the
    /// commit replays the kept path from them (`gdn_decode.replay`).
    spec_state_in: mlx.mlx_array = .{ .ctx = null },
    spec_prework: gdn_decode.Prework = .{},
    /// A chain verify left `spec_state_in` / `spec_prework` (the round's input state and its rows' recurrence inputs)
    /// in place of `spec_state_seq`; a partial accept replays rows 0..accepted from them (`ssmRollbackFromCapture`).
    spec_replay: bool = false,
    /// qwen4_exp per-request state beside the GDN pair: the PLE dilated-conv
    /// window `[B, (K-1)*dilation, hc*hidden]` on the PLE layer, the raw QSA
    /// indexer key history `[B, kv, idx_hd]` on full-attention layers.
    /// QSA history is a prefix of ONE buffer — stride checkpoints must NOT
    /// copy it (attachQsaHistoryToLatest puts one copy on the latest snap).
    aux_state: mlx.mlx_array = .{ .ctx = null },
    /// PLE: the (ngram_size-1) tokens preceding the next forward.
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
    /// Spec-verify capture for the PLE layer: the dilated-conv input
    /// `[B, state_len + T, hc*hidden]` and the token history (prev ++ ids)
    /// of the verify forward, so a partial accept slices instead of
    /// re-forwarding. Null / 0 outside a capturing verify.
    spec_ple_input: mlx.mlx_array = .{ .ctx = null },
    spec_ple_tokens: [32]u32 = @splat(0),
    spec_ple_len: u32 = 0,
    /// QSA: pooled + normed + roped block keys `[B, nb, idx_hd]` for the
    /// complete blocks of `aux_state`, extended incrementally per forward.
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    /// QSA raw-key history accelerator: a pre-grown `[B, cap, hd]` capacity buffer whose first
    /// `qsa_key_rows` rows are the history; `aux_state` stays the authority as a tight view.
    /// `qsaAppendKeys` trusts it only when `qsa_key_rows` equals the position being appended
    /// at; live sites that replace `aux_state` go through `ssmFreeQsaState`.
    qsa_key_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_key_rows: c_int = 0,
    /// Pooled-bank accelerators, same discipline: `qsa_pooled_buf` is the capacity buffer
    /// `qsa_pooled` views; `qsa_score_bank` is the f32 `[B, 1, hd, nb]` score operand,
    /// a view of `qsa_score_buf`, which grows by appended columns as blocks complete.
    qsa_pooled_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled_blocks: c_int = 0,
    qsa_score_bank: mlx.mlx_array = .{ .ctx = null },
    qsa_score_buf: mlx.mlx_array = .{ .ctx = null },
    qsa_score_blocks: c_int = 0,
    /// Rows this entry's QSA history is asked to hold up front (the counterpart of
    /// `KVCache.reserve_tokens`); the pooled bank derives its own by `ratio`. 0 = none.
    qsa_reserve_rows: usize = 0,
    /// Logical QSA history length in tokens (`pooled_blocks * ratio + ring_tail`).
    qsa_hist_rows: c_int = 0,
    group: ?*PersistentSsmGroup = null,
    group_row: u16 = 0,
    group_layer: u16 = 0,
    /// Bumped on every write to this entry outside its group; the group refuses a
    /// member whose gen moved. Read only from the entries, never from a ctx copy.
    member_gen: u64 = 0,
};

/// SSM snapshot value. Holds clones of conv_state and ssm_state via refcount —
/// the underlying buffer is shared with the source entry, but the snapshot
/// owns its own array handles and frees them on deinit. Used for
/// speculative-decoding rollback (PLD) where we must be able to revert one
/// decode step on a hybrid model.
pub const SSMCacheEntrySnapshot = struct {
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    initialized: bool,
    /// qwen4_exp per-request state beside the GDN pair (see SSMCacheEntry):
    /// the PLE conv window / QSA key history, the pooled block keys, and the
    /// PLE token history. Null / invalid on every other arch.
    aux_state: mlx.mlx_array = .{ .ctx = null },
    qsa_pooled: mlx.mlx_array = .{ .ctx = null },
    qsa_ratio: c_int = 4,
    qsa_rows: c_int = 0,
    /// The (ngram_size - 1) tokens preceding this entry.
    ple_prev: [qwen4_mod.MAX_NGRAM_SIZE]u32 = @splat(0),
    ple_prev_valid: bool = false,
};

pub fn ssmSnapshotDeinit(snap: *SSMCacheEntrySnapshot) void {
    _ = mlx.mlx_array_free(snap.conv_state);
    _ = mlx.mlx_array_free(snap.ssm_state);
    if (snap.aux_state.ctx != null) _ = mlx.mlx_array_free(snap.aux_state);
    if (snap.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(snap.qsa_pooled);
    snap.aux_state = .{ .ctx = null };
    snap.qsa_pooled = .{ .ctx = null };
}

/// Free the transient PLD spec-decode capture buffers on an SSM entry
/// (`spec_state_seq` / `spec_conv_input`). Idempotent — safe to call on an
/// entry that never captured. Called at the end of every `nextPld` round and
/// in the cache teardown paths so the capture never outlives a round.
pub fn ssmFreeSpecCapture(entry: *SSMCacheEntry) void {
    if (entry.spec_state_seq.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_state_seq);
        entry.spec_state_seq = .{ .ctx = null };
    }
    if (entry.spec_conv_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_conv_input);
        entry.spec_conv_input = .{ .ctx = null };
    }
    inline for (.{ &entry.spec_state_in, &entry.spec_prework.k, &entry.spec_prework.v, &entry.spec_prework.gb }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
    entry.spec_replay = false;
    if (entry.spec_ple_input.ctx != null) {
        _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = .{ .ctx = null };
    }
    entry.spec_ple_len = 0;
}

// ── tests ──
