//! copied_from: src/plugins/qwen4_exp/mtp.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's MTP head (`Qwen4Mtp`, `Qwen4MtpState` swapped onto the module): its load, forward over the pre-mixer
//! stream, QSA layer, history capture and rollback, the draft rerank, and the head-graph meters.

const std = @import("std");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const draft_fuse = @import("../../sdk/spec/mtp/draft_fuse.zig");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const kv_quant = @import("../../engines/mlx/cache/kv_quant.zig");
const KVQuantConfig = @import("../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const Weights = @import("../../sdk/config/model.zig").Weights;
const fused_activations = @import("../../sdk/blocks/mlp/fused_activations.zig");
const verify_projection = @import("../../sdk/blocks/linear/verify_projection.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const expectRowsByteEqual = @import("../../engines/mlx/batched/row_axis_verify.zig").expectRowsByteEqual;
const logicalSsmCapture = @import("../../engines/mlx/batched/row_axis_verify.zig").logicalSsmCapture;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const KVCacheSnapshot = @import("../../engines/mlx/kv/cache.zig").KVCacheSnapshot;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const init = @import("../../engines/mlx/load/init.zig").init;
const initMoeLayers = @import("../../engines/mlx/load/layers_layered.zig").initMoeLayers;
const weightTriple = @import("../../engines/mlx/load/weights.zig").weightTriple;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRestore = @import("../../engines/mlx/ssm/entry.zig").ssmRestore;
const ssmSnapshot = @import("../../engines/mlx/ssm/entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("../../engines/mlx/ssm/entry.zig").ssmSnapshotDeinit;
const Trunk = @import("trunk.zig").Trunk;
const abi = @import("../../abi/quant.zig");
const HcPending = @import("hc_fused.zig").HcPending;
const concatAxis0 = @import("../../mlx/array.zig").concatAxis0;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const SingleVerifyFeature = @import("../../sdk/blocks/linear/verify_projection.zig").SingleVerifyFeature;
const mtpHeadRowsFit = @import("../../sdk/blocks/linear/verify_projection.zig").mtpHeadRowsFit;
const verifySharedHardware = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedHardware;
const VqmmLane = @import("../../sdk/blocks/linear/verify_qmm.zig").VqmmLane;
const vqmmLaneFor = @import("../../sdk/blocks/linear/verify_qmm.zig").vqmmLaneFor;
const fusedSwiGLU = @import("../../sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;
const gatherExpertMm = @import("../../sdk/blocks/moe/experts.zig").gatherExpertMm;
const MoeDecodeDispatchArm = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").MoeDecodeDispatchArm;
const moeDecodeDispatchArm = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").moeDecodeDispatchArm;
const moeRowsAffineBank = @import("../../sdk/blocks/moe/gather_qmv_rows.zig").moeRowsAffineBank;
const MoeMlpWeights = @import("../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const endMropeChunk = @import("../../engines/mlx/methods/mrope.zig").endMropeChunk;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const mlxSettledActiveBytes = @import("../../sdk/testing/parity.zig").mlxSettledActiveBytes;
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;
const Qwen4TestSlot = @import("forward.zig").Qwen4TestSlot;
const loadHcWeights = @import("load.zig").loadHcWeights;
const qsaHistoryRows = @import("qsa/checkpoint.zig").qsaHistoryRows;
const QSA_RING_ROWS = @import("qsa/history.zig").QSA_RING_ROWS;
const QsaHeadMark = @import("qsa/history.zig").QsaHeadMark;
const QsaHeadMarkSet = @import("qsa/history.zig").QsaHeadMarkSet;
const qsaLeftoverAt = @import("qsa/history.zig").qsaLeftoverAt;
const qsaResliceToKeep = @import("qsa/history.zig").qsaResliceToKeep;
const qsaTestAppendPool = @import("qsa/history.zig").qsaTestAppendPool;
const qsaTestMeanBlocks = @import("qsa/history.zig").qsaTestMeanBlocks;
const reserveQsaHistoryWithHead = @import("qsa/history.zig").reserveQsaHistoryWithHead;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const truncatePooled = @import("qsa/history.zig").truncatePooled;
const headPlaceRopePos = @import("qsa/head_place.zig").headPlaceRopePos;
const qsaScoreFusedActive = @import("../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedActive;
const qsaScoreFusedActiveFor = @import("../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreFusedActiveFor;
const qsaArraysAllEqual = @import("../../sdk/testing/parity.zig").qsaArraysAllEqual;
const HcWeights = @import("weights.zig").HcWeights;
const gather_qmv_rows_vars = @import("../../sdk/blocks/moe/gather_qmv_rows.zig");

pub var qwen4_mtp_head_graphs: usize = 0;
pub var mtp_verify_moe_group_graphs: u64 = 0;
pub var mtp_verify_moe_group_last_ops: u64 = 0;
pub var mtp_head_force_batched_override: ?bool = null;

/// qwen4_exp MTP head (vLLM #53896 / SGLang #36497 `residual_linear_shared`):
///   x = fc_hidden(rms_{4H}(stream)) per stream + fc_embedding(rms_H(embed))
///   → one full-attention decoder layer (own hc/QSA/MoE) → mixer → lm_head.
/// Keeps its own KV cache (index `num_hidden_layers`) + indexer key history.
pub const Qwen4Mtp = struct {
    layer: MoeLayerWeights,
    pre_norm_emb: mlx.mlx_array,
    pre_norm_hidden: mlx.mlx_array,
    fc_emb_w: mlx.mlx_array,
    fc_emb_s: mlx.mlx_array,
    fc_emb_b: mlx.mlx_array,
    fc_hid_w: mlx.mlx_array,
    fc_hid_s: mlx.mlx_array,
    fc_hid_b: mlx.mlx_array,
    mixer: HcWeights,
    owned: []mlx.mlx_array,
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    /// The head's QSA leftovers at the trunk's checkpoint positions
    /// (`qwen4MtpMarkQsaLeftover`): what a warm clamp reads once the ring has
    /// moved past them. Per REQUEST — `qwen4MtpReset` clears it and an adopt
    /// replaces it with the committed set.
    qsa_marks: QsaHeadMarkSet = .{},
    /// Absolute position of the head's key row 0 (the first draft position
    /// after a reset).
    pos_base: c_int = 0,
    /// Cross-request EV controller seed — the in-checkpoint head's twin of
    /// `mtp.MtpModel.ev_seed_*`: the last HEALTHY request's per-index
    /// acceptance EMAs and base depth, written by `Generator.deinit` and read
    /// by the first `nextMtp` round of the next request. Without it every
    /// request re-warms the controller (MTP_EV_WARMUP_ROUNDS legacy rounds
    /// plus a +1/round base climb), which is a sizeable share of a short
    /// generation. Per-LOADED-MODEL state, exactly like the sidecar's: it
    /// lives on the head, so it dies with the head and can never reach
    /// another model. `qwen4MtpReset` deliberately does not clear it — a
    /// reset starts a new REQUEST, which is precisely who should inherit it.
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,
    /// Draft rerank (`mtp.rerankSelect`): a low-bit copy of the TRUNK lm_head
    /// that shortlists 32 candidates, which the real head then re-scores.
    /// Built at LOAD by `qwen4BuildDraftRerank` — not in `loadQwen4Mtp`, which
    /// runs while the Trunk is still under construction and `lm_head_w`
    /// is not assigned yet, but from the scheduler's load path once it is.
    /// `tried` makes it (and the refusal) one-shot, so the draft path's own
    /// ask is a pure read; a null `rerank` after it means the full readout is
    /// this head's draft path.
    rerank: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    rerank_tried: bool = false,

    /// Frees what the loader built and the head's per-request state.
    pub fn deinit(m: *Qwen4Mtp, a: std.mem.Allocator) void {
        for (m.owned) |arr| _ = mlx.mlx_array_free(arr);
        a.free(m.owned);
        m.cache.deinit();
        _ = mlx.mlx_array_free(m.entry.conv_state);
        _ = mlx.mlx_array_free(m.entry.ssm_state);
        ssmFreeQsaState(&m.entry);
        m.qsa_marks.deinit();
        if (m.rerank) |*rc| rc.deinit();
    }
};

/// The prefix of the MTP head's weights the pack ships (`<prefix>.fc_hidden` and its layer), or null.
pub fn headPrefix(weights: *const Weights) ?[]const u8 {
    if (weights.get("language_model.mtp.fc_hidden.weight") != null) return "language_model.mtp";
    if (weights.get("mtp.fc_hidden.weight") != null) return "mtp";
    return null;
}

/// Load the qwen4_exp MTP head when the pack ships `mtp.*` (null otherwise).
pub fn loadQwen4Mtp(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !?Qwen4Mtp {
    const mtp_prefix = headPrefix(weights) orelse return null;
    var mcfg = config;
    // The head's experts stay whole on a tensor-parallel rank 0 (`distributed.headOnly`).
    if (mcfg.dist_shard) |sh| {
        mcfg.moe_intermediate_size *= sh.size;
        mcfg.shared_expert_intermediate_size *= sh.size;
        mcfg.dist_shard = null;
    }
    mcfg.weight_prefix = mtp_prefix;
    mcfg.num_hidden_layers = 1;
    mcfg.full_attention_interval = 1; // layer 0 is the full-attention layer
    mcfg.linear_attn_tail_from = 0;
    mcfg.ple_layer_idx = -1;
    const ml = try initMoeLayers(allocator, mcfg, weights, name_buf, s);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer owned.deinit(allocator);
    try owned.appendSlice(allocator, ml.owned_bf16);
    allocator.free(ml.owned_bf16);
    const layer = ml.moe_layers[0];
    allocator.free(ml.moe_layers);
    const entry = ml.ssm_entries[0];
    allocator.free(ml.ssm_entries);
    var fe_buf: [80]u8 = undefined;
    var fh_buf: [80]u8 = undefined;
    var mixer_buf: [80]u8 = undefined;
    var pne_buf: [80]u8 = undefined;
    var pnh_buf: [80]u8 = undefined;
    const fe = try weightTriple(weights, try std.fmt.bufPrint(&fe_buf, "{s}.fc_embedding", .{mtp_prefix}), null, &owned, allocator, s);
    const fh = try weightTriple(weights, try std.fmt.bufPrint(&fh_buf, "{s}.fc_hidden", .{mtp_prefix}), null, &owned, allocator, s);
    const mixer = try loadHcWeights(weights, try std.fmt.bufPrint(&mixer_buf, "{s}.hyper_connection_mixer", .{mtp_prefix}), false, config.hc_count, config.hidden_size, &owned, allocator, s);
    const pne = weights.get(try std.fmt.bufPrint(&pne_buf, "{s}.pre_fc_norm_embedding.weight", .{mtp_prefix})) orelse return error.MissingWeight;
    const pnh = weights.get(try std.fmt.bufPrint(&pnh_buf, "{s}.pre_fc_norm_hidden.weight", .{mtp_prefix})) orelse return error.MissingWeight;
    var cache = try KVCache.init(allocator, config.num_hidden_layers + 1);
    errdefer cache.deinit();
    log.info("[qwen4] MTP head loaded (1 hyper-connected QSA+MoE layer; drafts armed by --mtp)\n", .{});
    return .{
        .layer = layer,
        .pre_norm_emb = pne,
        .pre_norm_hidden = pnh,
        .fc_emb_w = fe.w,
        .fc_emb_s = fe.sc,
        .fc_emb_b = fe.bi,
        .fc_hid_w = fh.w,
        .fc_hid_s = fh.sc,
        .fc_hid_b = fh.bi,
        .mixer = mixer,
        .owned = try owned.toOwnedSlice(allocator),
        .cache = cache,
        .entry = entry,
    };
}

/// The head's PER-REQUEST half. Every request owns one and installs it on the
/// module (`qwen4MtpActivate`) before any head operation; the module's own copy
/// is whatever the last owner swapped out.
pub const Qwen4MtpState = struct {
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    qsa_marks: QsaHeadMarkSet = .{},
    pos_base: c_int = -1,

    pub const Meta = struct {
        stash_len: usize,
        origin: c_int,
        cache_step: usize,
    };

    pub fn meta(self: *const Qwen4MtpState) Meta {
        return .{
            .stash_len = self.seq_offset,
            .origin = self.pos_base,
            .cache_step = self.cache.step,
        };
    }

    pub fn restoreMeta(self: *Qwen4MtpState, m: Meta) void {
        self.seq_offset = m.stash_len;
        self.pos_base = m.origin;
        self.cache.step = m.cache_step;
    }

    pub fn deinit(self: *Qwen4MtpState) void {
        self.cache.deinit();
        ssmFreeQsaState(&self.entry);
        self.qsa_marks.deinit();
    }
};

/// Pointers to one head's live runtime. The tick holds N of these without a module swap.
pub const Qwen4MtpLive = struct {
    cache: *KVCache,
    entry: *SSMCacheEntry,
    seq_offset: *usize,
    pos_base: *c_int,
    qsa_marks: *QsaHeadMarkSet,
};

pub fn qwen4MtpLiveState(st: *Qwen4MtpState) Qwen4MtpLive {
    return .{
        .cache = &st.cache,
        .entry = &st.entry,
        .seq_offset = &st.seq_offset,
        .pos_base = &st.pos_base,
        .qsa_marks = &st.qsa_marks,
    };
}

pub fn qwen4MtpLiveModule(m: *Qwen4Mtp) Qwen4MtpLive {
    return .{
        .cache = &m.cache,
        .entry = &m.entry,
        .seq_offset = &m.seq_offset,
        .pos_base = &m.pos_base,
        .qsa_marks = &m.qsa_marks,
    };
}

/// A fresh, empty per-request head state (what `qwen4MtpReset` leaves behind).
pub fn qwen4MtpStateNew(self: *Trunk) !Qwen4MtpState {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var entry = m.entry;
    entry.conv_state = .{ .ctx = null };
    entry.ssm_state = .{ .ctx = null };
    entry.initialized = false;
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    entry.qsa_hist_rows = 0;
    entry.qsa_pooled = .{ .ctx = null };
    entry.qsa_pooled_buf = .{ .ctx = null };
    entry.qsa_pooled_blocks = 0;
    entry.qsa_score_bank = .{ .ctx = null };
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    entry.spec_state_seq = .{ .ctx = null };
    entry.spec_conv_input = .{ .ctx = null };
    entry.spec_ple_input = .{ .ctx = null };
    const cache = try KVCache.initWithConfig(self.allocator, self.config.num_hidden_layers + 1, m.cache.config);
    return .{ .cache = cache, .entry = entry };
}

pub fn qwen4MtpSwapState(m: *Qwen4Mtp, st: *Qwen4MtpState) void {
    std.mem.swap(KVCache, &m.cache, &st.cache);
    std.mem.swap(SSMCacheEntry, &m.entry, &st.entry);
    std.mem.swap(usize, &m.seq_offset, &st.seq_offset);
    std.mem.swap(QsaHeadMarkSet, &m.qsa_marks, &st.qsa_marks);
    std.mem.swap(c_int, &m.pos_base, &st.pos_base);
}

/// Install `st` as the head's live state (swapping the previous owner's out). Idempotent.
pub fn qwen4MtpActivate(self: *Trunk, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner == st) return;
    if (self.qwen4_mtp_owner) |prev| qwen4MtpSwapState(m, prev);
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = st;
}

/// Take `st` back off the module before it is freed.
pub fn qwen4MtpRelease(self: *Trunk, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner != st) return;
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = null;
}

/// MTP draft logits: `stream_prev` `[B,S,hc*H]` is the trunk's pre-mixer
/// stream at positions p.., `token_ids` `[B,S]` the tokens at p+1.., and
/// `pos_offset` = p+1 of the first row (the draft query's own position).
/// Returns `[B,S,V]` logits for positions p+2... Caller frees.
/// `logits` is the last-row projection (null unless `.all_rows`/`.last_row`),
/// `stream` the pre-mixer hyper-connection stream the next chain step reads,
/// and `mixed` the MIXER output — the vector the lm_head consumes — which
/// only `.mixed_last_row` fills. The three are distinct spaces: `stream` is
/// `[B,S,hc*H]`, `mixed` is `[B,1,H]`, and handing the wrong one to the
/// lm_head is a silent shape error, not a wrong answer.
pub const Qwen4MtpOut = struct {
    logits: mlx.mlx_array,
    stream: mlx.mlx_array,
    mixed: mlx.mlx_array = .{ .ctx = null },
};

/// Owns the rows built so far, so a failure part-way frees exactly them.
pub const Qwen4MtpOutBuf = struct {
    items: []Qwen4MtpOut,
    built: usize = 0,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, n: usize) !Qwen4MtpOutBuf {
        return .{ .items = try allocator.alloc(Qwen4MtpOut, n), .allocator = allocator };
    }

    fn deinit(self: *Qwen4MtpOutBuf) void {
        for (self.items[0..self.built]) |o| {
            if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
            if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
            if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
        }
        self.allocator.free(self.items);
    }

};

/// What `qwen4MtpForward` owes its caller past the head's own layer. The
/// mixer read + the 248320-wide lm_head run over whatever rows `stream`
/// carries, so a caller that consumes ONE row says so instead of paying
/// for `S` of them and slicing (the merged history step after a partial
/// accept is `1 + accepted` rows wide, of which the drafter wants the
/// last).
pub const Qwen4MtpProject = enum {
    /// Every row's logits + the full `[B,S,hc*H]` stream (fixture oracles).
    all_rows,
    /// The LAST row only: `h` is sliced BEFORE the mixer, so `logits` is
    /// `[B,1,V]` and `stream` is that one row.
    last_row,
    /// No logits at all (history append): mixer + lm_head are skipped and
    /// `logits` comes back null; `stream` is the full `[B,S,hc*H]`.
    none,
    /// The LAST row's mixer output in `mixed`, and NO lm_head: the draft
    /// rerank path shortlists through its own coarse head, so the 248320-
    /// wide projection — 675 MB, the single biggest read in a draft step —
    /// never runs. `logits` comes back null; `stream` is the last row.
    mixed_last_row,
};

/// Truncate the head's committed history to `len` rows.
pub fn qwen4MtpTruncate(self: *Trunk, len: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    var live = qwen4MtpLiveModule(m);
    try self.qwen4MtpTruncateLive(&live, len);
}

pub fn qwen4MtpTruncateLive(self: *Trunk, live: *Qwen4MtpLive, len: usize) !void {
    if (len >= live.seq_offset.*) return;
    try live.cache.truncate(len, self.s);
    if (live.entry.aux_state.ctx != null or live.entry.qsa_pooled.ctx != null) {
        if (len == 0) {
            ssmFreeQsaState(live.entry);
        } else {
            const keep: c_int = @intCast(len);
            if (!try qsaResliceToKeep(live.entry, keep, self.s)) {
                if (live.entry.aux_state.ctx == null) {
                    live.entry.qsa_hist_rows = keep;
                    try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                } else {
                    const hist = qsaHistoryRows(live.entry);
                    const drop_tail = hist - keep;
                    const ks = mlx.getShape(live.entry.aux_state);
                    const new_held = ks[1] - drop_tail;
                    if (new_held < @mod(keep, @max(live.entry.qsa_ratio, 1))) {
                        try self.qwen4MtpTruncateToMarkLive(live, keep);
                    } else {
                        const start = [_]c_int{ 0, 0, 0 };
                        const stop = [_]c_int{ ks[0], new_held, ks[2] };
                        const strides = [_]c_int{ 1, 1, 1 };
                        var view = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(view);
                        try mlx.check(mlx.mlx_slice(&view, live.entry.aux_state, &start, 3, &stop, 3, &strides, 3, self.s));
                        const owned = try materializedOwnedCopy(self.s, view);
                        const pooled = live.entry.qsa_pooled;
                        live.entry.qsa_pooled = .{ .ctx = null };
                        ssmFreeQsaState(live.entry);
                        live.entry.qsa_pooled = pooled;
                        live.entry.aux_state = owned;
                        live.entry.qsa_key_rows = new_held;
                        live.entry.qsa_hist_rows = keep;
                        try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                    }
                }
            }
        }
    }
    live.qsa_marks.dropAbove(@intCast(len));
    live.seq_offset.* = len;
}

pub fn qwen4MtpTruncateToMarkLive(self: *Trunk, live: *Qwen4MtpLive, keep: c_int) !void {
    const want_lv = @mod(keep, @max(live.entry.qsa_ratio, 1));
    var rows: mlx.mlx_array = .{ .ctx = null };
    if (want_lv > 0) {
        const marked = live.qsa_marks.find(keep) orelse return error.MtpHeadQsaRingGap;
        if (mlx.getShape(marked)[1] != want_lv) return error.MtpHeadQsaRingGap;
        rows = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&rows, marked)) catch |err| {
            _ = mlx.mlx_array_free(rows);
            return err;
        };
    }
    const pooled = live.entry.qsa_pooled;
    live.entry.qsa_pooled = .{ .ctx = null };
    ssmFreeQsaState(live.entry);
    live.entry.qsa_pooled = pooled;
    live.entry.aux_state = rows;
    live.entry.qsa_key_rows = want_lv;
    live.entry.qsa_hist_rows = keep;
    try truncatePooled(&live.entry.qsa_pooled, keep, live.entry.qsa_ratio, self.s, true);
}

/// Commit a forward's `seq_len` rows to the head's own bookkeeping: `KVCache.update`
/// advances `step` only at layer 0, and the head's layer is `num_hidden_layers`. Without
/// this the committed snapshot's step disagreed with its QSA history and every restore declined.
pub fn qwen4MtpAdvance(cache: *KVCache, seq_offset: *usize, seq_len: c_int) void {
    seq_offset.* += @intCast(seq_len);
    cache.step = seq_offset.*;
}

/// `mrope_ctx` (image turns): the head's rows sit at ABSOLUTE positions
/// `pos_base + seq_offset ..`, so its attention + QSA indexer read the
/// slot's 3-D table there (prompt rows spanning the image) and the
/// scalar `offset + delta` past it — the same two arms as the trunk.
pub fn qwen4MtpForward(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project, null, null);
}

pub fn qwen4MtpForwardOn(self: *Trunk, st: *Qwen4MtpState, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveState(st);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project, null, null);
}

/// A draft row written past a padded history append. Key slots `[live_end, dead_end)`
/// are dead history rows: hidden from its attention and its QSA block selection. The
/// row ropes at `live_end + (slot - dead_end)`, the position it would hold with no pad.
pub const HeadPlace = struct {
    /// 0-d int32 slot one past the last live history row.
    live_end: mlx.mlx_array,
    dead_end: c_int,
};

/// `qwen4MtpForward` with the row's normalized embedding handed in (`front_en`: the previous draft step's fused pick,
/// bit for bit `rmsNorm(embedding(token_ids_in), pre_norm_emb)`, `draft_fuse.zig`); null builds it here.
pub fn qwen4MtpForwardFront(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject, front_en: ?mlx.mlx_array) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project, null, front_en);
}

/// `qwen4MtpForwardPlaced` with the row's normalized embedding handed in (`qwen4MtpForwardFront`).
pub fn qwen4MtpForwardPlacedFront(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, place: *const HeadPlace, project: Qwen4MtpProject, front_en: ?mlx.mlx_array) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, 0, null, project, place, front_en);
}

pub const Qwen4MtpBatchRow = struct {
    st: *Qwen4MtpState,
    stream: mlx.mlx_array,
    token_ids: mlx.mlx_array,
    pos_offset: c_int,
    mrope_ctx: ?mrope.PositionContext = null,
};

var head_defer_said = false;

/// The head's deferred hc write on an EXL3 trunk pack (`hcWriteOrDefer` left a write `pending`), said once.
fn noteHeadDefer(deferred: bool, rows: c_int) void {
    if (!deferred or head_defer_said) return;
    head_defer_said = true;
    log.info("[qwen4] MTP head deferred hc write engaged: the head's write rides its next read's first launch (rows={d})\n", .{rows});
}

pub fn qwen4MtpForwardLive(self: *Trunk, m: *Qwen4Mtp, live: *Qwen4MtpLive, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject, place: ?*const HeadPlace, front_en: ?mlx.mlx_array) !Qwen4MtpOut {
    qwen4_mtp_head_graphs += 1;
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const shape = mlx.getShape(stream_prev);
    const batch: c_int = shape[0];
    const seq_len: c_int = shape[1];
    const is_prefill = seq_len > 1;
    if (place) |p| {
        if (seq_len != 1 or mrope_ctx != null or p.dead_end <= 0 or p.dead_end > @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPlacement;
    } else {
        if (live.seq_offset.* == 0) live.pos_base.* = pos_offset;
        if (pos_offset != live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPositionGap;
    }

    // The previous draft step's fused pick built this row's `embedding` + `rmsNorm(pre_norm_emb)` (`draft_fuse.zig`,
    // bit for bit); a front that is not one [1,1,hidden] bf16 row is ignored.
    const front: ?mlx.mlx_array = if (front_en) |fe| (if (draft_fuse.frontEnFits(fe, batch, seq_len, hidden)) fe else null) else null;
    const en = if (front) |fe| blk: {
        var c = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_array_set(&c, fe));
        draft_fuse.noteFront();
        break :blk c;
    } else blk: {
        var token_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(token_ids);
        try mlx.check(mlx.mlx_reshape(&token_ids, token_ids_in, &[_]c_int{ batch, seq_len }, 2, self.s));
        const e = try self.embedding(token_ids);
        defer _ = mlx.mlx_array_free(e);
        break :blk try self.rmsNorm(e, m.pre_norm_emb);
    };
    defer _ = mlx.mlx_array_free(en);
    const ep = try self.qmatmul(en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try self.rmsNorm(stream_prev, m.pre_norm_hidden); // ONE rms over all hc*H
    defer _ = mlx.mlx_array_free(hn);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &shape4, 4, self.s));
    const hp = try self.qmatmul(hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ batch, seq_len, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ batch, seq_len, hc * hidden }, 3, self.s));

    var ctx: ForwardCtx = .{ .cache = live.cache, .moe_seq_offset = live.seq_offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    defer if (ctx.head_rope_pos.ctx != null) {
        _ = mlx.mlx_array_free(ctx.head_rope_pos);
    };
    if (place) |p| {
        ctx.head_place = p;
        ctx.head_rope_pos = try headPlaceRopePos(self.s, p, live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*)) - p.dead_end);
    }
    if (mrope_ctx) |mc| {
        ctx.mrope_pos = mc.pos;
        ctx.mrope_total = mc.total;
        ctx.mrope_delta = mc.delta;
    }
    try self.beginMropeChunk(&ctx, @intCast(live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer endMropeChunk(&ctx);
    // A history append keeps the KV rows and the QSA history; its callers drop the stream (`fusion.kvOnlySkips`).
    const kv_only_was = self.fused.kv_only;
    self.fused.kv_only = project == .none;
    defer self.fused.kv_only = kv_only_was;
    // On an EXL3 trunk pack the head's two writes defer to its next read, as the trunk's do (the read's first launch
    // applies stream + out * inj, bit for bit the write and then the read); flushed wherever the stream itself is needed.
    // Every other pack writes as main does, and `pending` stays null.
    var pending: ?HcPending = null;
    defer if (pending) |*pd| pd.deinit();
    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    var pre = try self.hcRead(h, &lw.hc_attn.?, batch, seq_len);
    defer pre.deinit();
    const attn_out = switch (lw.attn) {
        .full => |fa| try self.qwen4AttnWith(&ctx, pre.mixed, &fa, live.entry, li, @intCast(live.seq_offset.*), live.pos_base.*, batch, seq_len, is_prefill),
        .linear => return error.MtpLayerNotAttention,
    };
    defer _ = mlx.mlx_array_free(attn_out);
    if (self.config.trunk_format) {
        try self.hcWriteOrDefer(&h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);
        noteHeadDefer(pending != null, batch * seq_len);
    } else h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len);
    var pre2 = try self.hcReadPending(&h, &lw.hc_mlp.?, batch, seq_len, &pending);
    defer pre2.deinit();
    const mlp_out = switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP(pre2.mixed, mw),
        .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
    };
    defer _ = mlx.mlx_array_free(mlp_out);
    if (self.config.trunk_format) try self.hcWriteOrDefer(&h, mlp_out, pre2.inj, batch, seq_len, m.mixer.inject_flat, &pending) else h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len);
    qwen4MtpAdvance(live.cache, live.seq_offset, seq_len);

    // History append wants the stream and nothing else: the mixer read and
    // the vocab-wide projection are pure, so skipping them is free.
    if (project == .none) {
        try self.hcFlush(&h, batch, seq_len, &pending);
        return .{ .logits = .{ .ctx = null }, .stream = h };
    }
    // Last-row-only: cut `h` HERE, so the mixer + lm_head see one row
    // instead of S. Same values as slicing the outputs afterwards.
    var out_seq: c_int = seq_len;
    if ((project == .last_row or project == .mixed_last_row) and seq_len > 1) {
        try self.hcFlush(&h, batch, seq_len, &pending);
        const hs = mlx.getShape(h);
        var h_last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h_last);
        try mlx.check(mlx.mlx_slice(
            &h_last,
            h,
            &[_]c_int{ 0, seq_len - 1, 0 },
            3,
            &[_]c_int{ hs[0], seq_len, hs[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            self.s,
        ));
        _ = mlx.mlx_array_free(h);
        h = h_last;
        out_seq = 1;
    }

    // With a deferred write (one row: the draft chain) the mixer read applies it and hands back the written stream.
    const mix = try self.hcReadPending(&h, &m.mixer, batch, out_seq, &pending);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    // Rerank draft: the mixer output IS the answer. Skipping
    // `lmHeadProject` here is the whole point — the 248320-wide 8-bit
    // projection is 675 MB, more than every other read in a draft step put
    // together, and a greedy draft only ever needed its argmax.
    if (project == .mixed_last_row) {
        return .{ .logits = .{ .ctx = null }, .stream = h, .mixed = mix.mixed };
    }
    defer _ = mlx.mlx_array_free(mix.mixed);
    const logits = try self.lmHeadProject(mix.mixed);
    return .{ .logits = logits, .stream = h };
}

/// Build the qwen4_exp head's coarse rerank head. ONE-SHOT (`rerank_tried`
/// caches the refusal too) and infallible — false keeps the full-vocab
/// draft projection.
///
/// Called at LOAD (`scheduler.doLoadOnInferenceThread`, the site both the
/// boot load and the `/v1/load` cold load route through), the sidecar
/// arm's `maybeBuildDraftRerank`-at-bind twin: a full `requantizeRows` of
/// the trunk lm_head plus a synchronous eval of its ~240 MB is a LOAD
/// cost, and inside the first request's draft chain it was first-token
/// latency on the inference thread with the stream drained mid-round.
pub fn qwen4BuildDraftRerank(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    m.rerank_tried = true;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .awake);
    // A trunk format's head (the EXL3 pack's trellis) has no affine rows: the coarse head is built from what it
    // computes.
    const built = if (self.formats.linearOf(self.lm_head_w)) |fw| formatHeadRerank(self, fw) else mtp_mod.buildRerankCoarse(self.s, self, mtp_mod.DRAFT_HEAD_BITS);
    const rc = built orelse return false;
    m.rerank = rc;
    const h: c_int = @intCast(self.config.hidden_size);
    var bytes = mtp_mod.rerankCoarseBytes(rc.rows, h, rc.bits);
    if (rc.exact != null) bytes += mtp_mod.rerankCoarseBytes(rc.rows, h, mtp_mod.RERANK_EXACT_BITS);
    if (rc.sub != null) bytes += mtp_mod.rerankCoarseBytes(rc.sub_rows, h, rc.bits);
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    log.info(
        "[qwen4] MTP draft rerank: coarse lm_head {d}-bit{s} ({d} MB, {d} ms) + exact top-32 rescoring; single-row drafts read {d} rows\n",
        .{ rc.bits, if (rc.exact != null) " + 8-bit re-score copy" else "", bytes / (1024 * 1024), ms, rc.readout().rows },
    );
    return true;
}

/// The coarse head over a trunk format's lm_head: the rows it computes (`formatHeadRows`) requantized once
/// (`mtp_mod.buildRerankCoarseRows`), then dropped. Proposal side only: the verify reads the head itself.
fn formatHeadRerank(self: *Trunk, fw: *const abi.Weight) ?mtp_mod.RerankCoarse {
    // no byte saving over a head already at the coarse width
    if (fw.bits != 0 and mtp_mod.DRAFT_HEAD_BITS >= fw.bits) {
        log.info("[qwen4] MTP draft rerank declined: the trunk format's lm_head is {d}-bit, no wider than the {d}-bit coarse head; drafts read the full head\n", .{ fw.bits, mtp_mod.DRAFT_HEAD_BITS });
        return null;
    }
    const dense = formatHeadRows(self) catch |err| {
        log.warn("[qwen4] MTP draft rerank: the head's rows failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    defer _ = mlx.mlx_array_free(dense);
    return mtp_mod.buildRerankCoarseRows(self.s, dense, mtp_mod.DRAFT_HEAD_BITS);
}

/// The `[V, H]` rows a trunk format's lm_head computes: the head applied to I_H (row i of the identity reads column i
/// of W through the dispatch every readout takes), transposed. Lazy; a load-time transient of V x H in the activation
/// dtype once evaluated.
fn formatHeadRows(self: *const Trunk) !mlx.mlx_array {
    const h: c_int = @intCast(self.config.hidden_size);
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0, @floatFromInt(h), 1, .int32, self.s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, idx, &[_]c_int{ h, 1 }, 2, self.s));
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, col, idx, self.s));
    var eye = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eye);
    try mlx.check(mlx.mlx_astype(&eye, hit, self.actDtype(), self.s));
    // one H-row chunk [1, H, H], as a prefill chunk reaches the head
    var eye3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eye3);
    try mlx.check(mlx.mlx_reshape(&eye3, eye, &[_]c_int{ 1, h, h }, 3, self.s));
    const cols3 = try self.lmHeadProject(eye3); // [1, H, V] = W^T
    defer _ = mlx.mlx_array_free(cols3);
    const v = mlx.getShape(cols3)[2];
    var cols = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cols);
    try mlx.check(mlx.mlx_reshape(&cols, cols3, &[_]c_int{ h, v }, 2, self.s));
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_transpose(&rows, cols, self.s));
    return rows;
}

/// Is the qwen4_exp head's draft rerank available? A pure read once the
/// load-time build has run; the build is only reached here when it did
/// NOT (an offline path, a directly-constructed Trunk), and it is
/// one-shot either way.
pub fn qwen4DraftRerankReady(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    return self.qwen4BuildDraftRerank();
}

/// One greedy draft id from the mixer output `x` `[B,1,H]`. Shares the
/// sidecar head's scheme verbatim; a coarse-head absence or a shortlist
/// failure falls through to the full trunk-head readout, which is exactly
/// what this path replaced.
pub fn qwen4DraftSelect(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try mtp_mod.rerankSelect(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask)) |tok| return tok;
    return mtp_mod.fullReadoutArgmax(self.s, self, x, suppress_mask);
}

/// The fused draft pick (`draft_fuse.zig`): `qwen4DraftSelect`'s id for one row `x` `[1,1,H]` in fewer launches, bit
/// for bit, plus the next draft step's normalized embedding for `qwen4MtpForwardFront`. Null = off the EXL3 profile, or
/// declined (logged): the caller runs `qwen4DraftSelect`.
pub fn qwen4DraftPick(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?draft_fuse.Pick {
    const m = &(self.qwen4_mtp orelse return null);
    const front = draftFuseFront(self, m) orelse return null;
    if (m.rerank != null) return mtp_mod.rerankPickFused(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask, front);
    // No coarse head (never built, or retired): `fullReadoutArgmax`'s readout, its mask and argmax in the pick.
    const logits = try self.lmHeadForeign(x);
    defer _ = mlx.mlx_array_free(logits);
    return draft_fuse.pickExactChecked(self.s, logits, suppress_mask, front, .int32);
}

/// A greedy draft from the head's own logits `[1,1,V]` (`sampleTokenLazy`'s temperature-0 path): its where + argmax
/// (`[1,1]` uint32), then the next step's normalized embedding in one launch. Null = off the EXL3 profile, or declined
/// (logged): the caller runs `sampleTokenLazy`.
pub fn qwen4GreedyPick(self: *Trunk, logits: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?draft_fuse.Pick {
    const m = &(self.qwen4_mtp orelse return null);
    const front = draftFuseFront(self, m) orelse return null;
    return draft_fuse.pickExactChecked(self.s, logits, suppress_mask, front, .uint32);
}

/// The head's front as `qwen4MtpForwardLive` builds it, when a pick can build it instead: a dense table's row cast to
/// bfloat16, then `rmsNorm(pre_norm_emb)`. The EXL3 profile only (`Formats.profile`): elsewhere the path is unmeasured.
fn draftFuseFront(self: *const Trunk, m: *const Qwen4Mtp) ?draft_fuse.Front {
    if (!self.formats.profile) return null;
    if (!mlx.streamIsGpu(self.s) or mlx.cudaAvailable()) {
        draft_fuse.decline(.no_metal);
        return null;
    }
    if (self.emb_s.ctx != null) {
        draft_fuse.decline(.embedding);
        return null;
    }
    const f: draft_fuse.Front = .{ .emb = self.emb_w, .norm_w = m.pre_norm_emb, .eps = self.config.rms_norm_eps, .hidden = @intCast(self.config.hidden_size) };
    if (draft_fuse.frontReason(f)) |r| {
        draft_fuse.decline(r);
        return null;
    }
    return f;
}

/// Test hook: layer-0 intermediates of the qwen4 forward (fixture bisect).
pub const Qwen4Trace = struct {
    mixed_attn: mlx.mlx_array = .{ .ctx = null },
    inj_attn: mlx.mlx_array = .{ .ctx = null },
    attn_out: mlx.mlx_array = .{ .ctx = null },
    mixed_mlp: mlx.mlx_array = .{ .ctx = null },
    inj_mlp: mlx.mlx_array = .{ .ctx = null },
    mlp_out: mlx.mlx_array = .{ .ctx = null },
    ple_emb: mlx.mlx_array = .{ .ctx = null },
    ple_out: mlx.mlx_array = .{ .ctx = null },
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    attn3_out: mlx.mlx_array = .{ .ctx = null },
    pub fn set(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
        if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
        dst.* = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(dst, src);
    }
    pub fn deinit(self: *Qwen4Trace) void {
        inline for (.{ &self.mixed_attn, &self.inj_attn, &self.attn_out, &self.mixed_mlp, &self.inj_mlp, &self.mlp_out, &self.ple_emb, &self.ple_out, &self.qsa_mask, &self.attn3_out }) |f| {
            if (f.ctx != null) _ = mlx.mlx_array_free(f.*);
        }
    }
};

// ── tests ──

/// A loaded qwen4_exp pack for the head tests; heap-held because the Trunk keeps a pointer to the weights.
const Qwen4HeadRig = struct {
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: Trunk,

    fn deinit(rig: *Qwen4HeadRig, alloc: std.mem.Allocator) void {
        rig.xfm.deinit();
        rig.weights.deinit();
        if (rig.config.ngram_table_path) |p| alloc.free(p);
        alloc.destroy(rig);
    }

    /// `n` seeded pre-mixer stream rows `[1,n,hc*H]`.
    fn stream(rig: *Qwen4HeadRig, seed: u64, n: c_int) !mlx.mlx_array {
        const width: c_int = @intCast(rig.config.hc_count * rig.config.hidden_size);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&out, &[_]c_int{ 1, n, width }, 3, .bfloat16, 0, 1, key, rig.xfm.s));
        return out;
    }

};

const SingleVerifyTestResult = struct {
    logits: mlx.mlx_array,
    last: mlx.mlx_array,
    all: mlx.mlx_array,

    fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.logits);
        _ = mlx.mlx_array_free(self.last);
        _ = mlx.mlx_array_free(self.all);
    }
};
