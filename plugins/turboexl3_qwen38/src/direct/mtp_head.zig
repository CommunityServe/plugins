//! The direct model's MTP head on the K packs: its live forward (the fc projections of the embedding and the previous
//! stream, one hyper-connected QSA + MoE layer, the mixer), the per-request state it keeps (swapped onto the module,
//! truncated on a rewind), its load, and the draft head over the trunk format's lm_head: a 3-bit coarse readout of the
//! vocab subset, its top 32 re-scored by an 8-bit copy. Transcribed from today's `plugins/qwen4_exp/mtp.zig` and
//! `sdk/spec/mtp/mtp.zig`, over their types; the round policy is `src/spec/drafter.zig`'s.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const materializedOwnedCopy = @import("../mlx_serve/mlx/array.zig").materializedOwnedCopy;
const log = @import("../mlx_serve/sdk/util/log.zig");
const abi = @import("../mlx_serve/abi/quant.zig");
const ModelConfig = @import("../mlx_serve/sdk/config/model.zig").ModelConfig;
const Weights = @import("../mlx_serve/sdk/config/model.zig").Weights;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;
const initMoeLayers = @import("../mlx_serve/engines/mlx/load/layers_layered.zig").initMoeLayers;
const weightTriple = @import("../mlx_serve/engines/mlx/load/weights.zig").weightTriple;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const mtp = @import("../mlx_serve/plugins/qwen4_exp/mtp.zig");
const loadHcWeights = @import("../mlx_serve/plugins/qwen4_exp/load.zig").loadHcWeights;
const qsaHistoryRows = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig").qsaHistoryRows;
const QsaHeadMarkSet = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").QsaHeadMarkSet;
const qsaResliceToKeep = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").qsaResliceToKeep;
const ssmFreeQsaState = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const truncatePooled = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").truncatePooled;
const spec_mtp = @import("../mlx_serve/sdk/spec/mtp/mtp.zig");
const hc = @import("hc.zig");
const attn = @import("attn.zig");
const moe = @import("moe.zig");
const linear = @import("linear.zig");

const QLinear = spec_mtp.QLinear;
const RerankCoarse = spec_mtp.RerankCoarse;
const Shortlist = spec_mtp.Shortlist;

// ── The head's forward ──

/// One live head step over `live` (today's `qwen4MtpForwardLive`): `stream_prev` `[B,S,hc*H]` is the trunk's pre-mixer
/// stream at positions p.., `token_ids_in` `[B,S]` the tokens at p+1.., `pos_offset` = p+1 of the first row. `project`
/// is what the caller reads past the head's layer (`Qwen4MtpProject`). Caller frees the result's arrays.
pub fn forwardLive(self: *Trunk, m: *mtp.Qwen4Mtp, live: *mtp.Qwen4MtpLive, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, project: mtp.Qwen4MtpProject) !mtp.Qwen4MtpOut {
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const cfg = &self.config;
    const hc_count: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const shape = mlx.getShape(stream_prev);
    const batch: c_int = shape[0];
    const seq_len: c_int = shape[1];
    const is_prefill = seq_len > 1;
    if (live.seq_offset.* == 0) live.pos_base.* = pos_offset;
    if (pos_offset != live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPositionGap;

    const en = blk: {
        var token_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(token_ids);
        try mlx.check(mlx.mlx_reshape(&token_ids, token_ids_in, &[_]c_int{ batch, seq_len }, 2, self.s));
        const e = try linear.embedding(self, token_ids);
        defer _ = mlx.mlx_array_free(e);
        break :blk try linear.rmsNorm(self, e, m.pre_norm_emb);
    };
    defer _ = mlx.mlx_array_free(en);
    const ep = try linear.qmatmul(self, en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try linear.rmsNorm(self, stream_prev, m.pre_norm_hidden); // ONE rms over all hc*H
    defer _ = mlx.mlx_array_free(hn);
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &[_]c_int{ batch, seq_len, hc_count, hidden }, 4, self.s));
    const hp = try linear.qmatmul(self, hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ batch, seq_len, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ batch, seq_len, hc_count * hidden }, 3, self.s));

    // The head's own ctx: its KV cache and row count.
    var ctx: ForwardCtx = .{ .cache = live.cache, .moe_seq_offset = live.seq_offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    // A history append keeps the KV rows and the QSA history; its callers drop the stream (`fusion.kvOnlySkips`).
    const kv_only_was = self.fused.kv_only;
    self.fused.kv_only = project == .none;
    defer self.fused.kv_only = kv_only_was;
    // Every K pack is a trunk format's (`trunk_format`): the head's two writes defer to its next read, as the trunk's
    // do (the read's first launch applies stream + out * inj, bit for bit the write and then the read), flushed
    // wherever the stream itself is needed.
    var pending: ?hc.HcPending = null;
    defer if (pending) |*pd| pd.deinit();
    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    var pre = try hc.readPlain(self, h, &lw.hc_attn.?, batch, seq_len);
    defer pre.deinit();
    const attn_out = switch (lw.attn) {
        .full => |fa| try attn.layer(self, &ctx, pre.mixed, &fa, live.entry, li, @intCast(live.seq_offset.*), live.pos_base.*, batch, seq_len, is_prefill),
        .linear => return error.MtpLayerNotAttention,
    };
    defer _ = mlx.mlx_array_free(attn_out);
    try hc.writeOrDefer(self, &h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);
    var pre2 = try hc.read(self, &h, &lw.hc_mlp.?, batch, seq_len, &pending);
    defer pre2.deinit();
    const mlp_out = switch (lw.mlp) {
        .moe => |*mw| try moe.mlp(self, pre2.mixed, mw),
        .dense => return error.DirectModelHasNoDenseMlp,
    };
    defer _ = mlx.mlx_array_free(mlp_out);
    try hc.writeOrDefer(self, &h, mlp_out, pre2.inj, batch, seq_len, m.mixer.inject_flat, &pending);
    qwen4MtpAdvance(live.cache, live.seq_offset, seq_len);

    // History append wants the stream and nothing else: the mixer read and
    // the vocab-wide projection are pure, so skipping them is free.
    if (project == .none) {
        try hc.flush(self, &h, batch, seq_len, &pending);
        return .{ .logits = .{ .ctx = null }, .stream = h };
    }
    // Last-row-only: cut `h` HERE, so the mixer + lm_head see one row
    // instead of S. Same values as slicing the outputs afterwards.
    var out_seq: c_int = seq_len;
    if ((project == .last_row or project == .mixed_last_row) and seq_len > 1) {
        try hc.flush(self, &h, batch, seq_len, &pending);
        const hs = mlx.getShape(h);
        var h_last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h_last);
        try mlx.check(mlx.mlx_slice(&h_last, h, &[_]c_int{ 0, seq_len - 1, 0 }, 3, &[_]c_int{ hs[0], seq_len, hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        _ = mlx.mlx_array_free(h);
        h = h_last;
        out_seq = 1;
    }

    // With a deferred write (one row: the draft chain) the mixer read applies it and hands back the written stream.
    const mix = try hc.read(self, &h, &m.mixer, batch, out_seq, &pending);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    // A rerank draft reads the mixer output itself: the 248320-wide projection, more than every other read in a draft
    // step put together, never runs.
    if (project == .mixed_last_row) {
        return .{ .logits = .{ .ctx = null }, .stream = h, .mixed = mix.mixed };
    }
    defer _ = mlx.mlx_array_free(mix.mixed);
    const logits = try linear.lmHeadProject(self, mix.mixed);
    return .{ .logits = logits, .stream = h };
}

/// Commit a forward's `seq_len` rows to the head's own bookkeeping: `KVCache.update` advances `step` only at layer 0,
/// and the head's layer is `num_hidden_layers`. Without this the committed snapshot's step disagreed with its QSA
/// history and every restore declined.
pub fn qwen4MtpAdvance(cache: *KVCache, seq_offset: *usize, seq_len: c_int) void {
    seq_offset.* += @intCast(seq_len);
    cache.step = seq_offset.*;
}

/// `forwardLive` on the module's live state (the installed request's, `qwen4MtpActivate`).
pub fn qwen4MtpForward(self: *Trunk, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, project: mtp.Qwen4MtpProject) !mtp.Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return forwardLive(self, m, &live, stream_prev, token_ids_in, pos_offset, project);
}

/// `forwardLive` on `st`, installed on the module or not.
pub fn qwen4MtpForwardOn(self: *Trunk, st: *mtp.Qwen4MtpState, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, project: mtp.Qwen4MtpProject) !mtp.Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveState(st);
    return forwardLive(self, m, &live, stream_prev, token_ids_in, pos_offset, project);
}

// ── The head's state ──

/// The prefix of the MTP head's weights the pack ships (`<prefix>.fc_hidden` and its layer), or null.
pub fn headPrefix(weights: *const Weights) ?[]const u8 {
    if (weights.get("language_model.mtp.fc_hidden.weight") != null) return "language_model.mtp";
    if (weights.get("mtp.fc_hidden.weight") != null) return "mtp";
    return null;
}

/// Load the MTP head when the pack ships `mtp.*` (null otherwise): one hyper-connected QSA + MoE layer, the fc
/// projections, the mixer and the pre-fc norms, with its own KV cache (index `num_hidden_layers`). The coarse draft head
/// is built later (`qwen4BuildDraftRerank`), once the trunk's lm_head is bound.
pub fn loadQwen4Mtp(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !?mtp.Qwen4Mtp {
    const mtp_prefix = headPrefix(weights) orelse return null;
    // One rank: the head's experts are never a tensor-parallel shard.
    if (config.dist_shard != null) return error.DirectModelIsOneRank;
    var mcfg = config;
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

pub fn qwen4MtpLiveState(st: *mtp.Qwen4MtpState) mtp.Qwen4MtpLive {
    return .{
        .cache = &st.cache,
        .entry = &st.entry,
        .seq_offset = &st.seq_offset,
        .pos_base = &st.pos_base,
        .qsa_marks = &st.qsa_marks,
    };
}

pub fn qwen4MtpLiveModule(m: *mtp.Qwen4Mtp) mtp.Qwen4MtpLive {
    return .{
        .cache = &m.cache,
        .entry = &m.entry,
        .seq_offset = &m.seq_offset,
        .pos_base = &m.pos_base,
        .qsa_marks = &m.qsa_marks,
    };
}

/// A fresh, empty per-request head state (what `qwen4MtpReset` leaves behind).
pub fn qwen4MtpStateNew(self: *Trunk) !mtp.Qwen4MtpState {
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

pub fn qwen4MtpSwapState(m: *mtp.Qwen4Mtp, st: *mtp.Qwen4MtpState) void {
    std.mem.swap(KVCache, &m.cache, &st.cache);
    std.mem.swap(SSMCacheEntry, &m.entry, &st.entry);
    std.mem.swap(usize, &m.seq_offset, &st.seq_offset);
    std.mem.swap(QsaHeadMarkSet, &m.qsa_marks, &st.qsa_marks);
    std.mem.swap(c_int, &m.pos_base, &st.pos_base);
}

/// Install `st` as the head's live state (swapping the previous owner's out). Idempotent.
pub fn qwen4MtpActivate(self: *Trunk, st: *mtp.Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner == st) return;
    if (self.qwen4_mtp_owner) |prev| qwen4MtpSwapState(m, prev);
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = st;
}

/// Take `st` back off the module before it is freed.
pub fn qwen4MtpRelease(self: *Trunk, st: *mtp.Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner != st) return;
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = null;
}

/// Truncate the head's committed history to `len` rows.
pub fn qwen4MtpTruncate(self: *Trunk, len: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    var live = qwen4MtpLiveModule(m);
    try qwen4MtpTruncateLive(self, &live, len);
}

/// `live`'s history cut to its first `len` rows: the KV rows, the QSA key ring and pooled keys (re-sliced in place, or
/// rebuilt from the leftover marked at `len`), and the marks above it.
pub fn qwen4MtpTruncateLive(self: *Trunk, live: *mtp.Qwen4MtpLive, len: usize) !void {
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
                        try qwen4MtpTruncateToMarkLive(self, live, keep);
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

/// The QSA history cut to `keep` rows once the ring no longer holds `keep`'s leftover: the leftover marked at `keep`
/// becomes the ring.
pub fn qwen4MtpTruncateToMarkLive(self: *Trunk, live: *mtp.Qwen4MtpLive, keep: c_int) !void {
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

// ── The draft head ──

/// Build the head's coarse rerank head, once (`rerank_tried` keeps the refusal too) and infallibly: false keeps the
/// full-readout draft path. Called at load, where a requantize of the lm_head's rows and the eval of its ~1 GB are a
/// load cost, not the first draft's latency.
pub fn qwen4BuildDraftRerank(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    m.rerank_tried = true;
    // A K pack's lm_head is the trunk format's (the EXL3 trellis), with no affine rows: the coarse head is built from
    // what it computes.
    const fw = self.formats.linearOf(self.lm_head_w) orelse {
        log.err("[qwen4] MTP draft rerank refused ({s}): the lm_head is not the trunk format's\n", .{@errorName(error.DirectModelHeadNotTrunkFormat)});
        return false;
    };
    m.rerank = formatHeadRerank(self, fw) orelse return false;
    return true;
}

/// The coarse head over a trunk format's lm_head: the rows it computes (`formatHeadRows`) requantized once
/// (`buildRerankCoarseRows`), then dropped. Proposal side only: the verify reads the head itself.
fn formatHeadRerank(self: *Trunk, fw: *const abi.Weight) ?RerankCoarse {
    // no byte saving over a head already at the coarse width: drafts read the full head
    if (fw.bits != 0 and DRAFT_HEAD_BITS >= fw.bits) return null;
    const dense = formatHeadRows(self) catch |err| {
        log.warn("[qwen4] MTP draft rerank: the head's rows failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    defer _ = mlx.mlx_array_free(dense);
    return buildRerankCoarseRows(self.s, dense, DRAFT_HEAD_BITS);
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
    const cols3 = try linear.lmHeadProject(self, eye3); // [1, H, V] = W^T
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

/// Is the head's draft rerank available? A pure read once the load-time build has run; the build runs here only when
/// it did not (a directly-constructed Trunk), one-shot either way.
pub fn qwen4DraftRerankReady(self: *Trunk) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    return qwen4BuildDraftRerank(self);
}

/// One greedy draft id from the mixer output `x` `[B,1,H]`: the rerank scheme, or the full trunk-head readout once the
/// coarse head is gone (never built, or retired by a shortlist failure).
pub fn qwen4DraftSelect(self: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try rerankSelect(self.s, &m.rerank, x, suppress_mask)) |tok| return tok;
    return fullReadoutArgmax(self.s, self, x, suppress_mask);
}

// ── The coarse rerank head (built at load) ──
//
// Proposal side only: a coarse requant of the lm_head scores the vocab, an exact top-32 shortlist is taken in two
// dispatches, and the 32 rows re-score it, so whenever the coarse top-32 holds the head's argmax the draft IS the
// head's choice. A coarse miss costs acceptance, never output: the verify reads the head itself.

/// Coarse-head width at build time. The width a head is READ at is a property of its packed weights
/// (`RerankCoarse.bits`): packed geometry implies K, so a read at another width reads a different vocab entirely.
pub const DRAFT_HEAD_BITS: u32 = 3;

/// Group size every coarse head is packed at.
pub const RERANK_COARSE_GROUP: u32 = 64;

/// Width of a trunk format head's re-score copy (`RerankCoarse.exact`).
pub const RERANK_EXACT_BITS: u32 = 8;

/// The vocab subset a trunk format head's single-row drafts read (`withVocabSubset`), on a vocab of DRAFT_VOCAB_MIN ids
/// or more: the first 96 Ki ids. A byte-level BPE vocab numbers its merges most frequent first, so these carry nearly
/// all text; a draft never proposes a later id and the verify reads the full head, so it costs acceptance, never
/// output.
pub const DRAFT_VOCAB_ROWS: c_int = 98304;
pub const DRAFT_VOCAB_MIN: c_int = 131072;

/// Resident bytes of a coarse head over `rows` x `hidden` at `bits`/gs64: packed weights plus bf16 scales AND biases.
/// Reported at build so the log line carries the number the change is claiming.
pub fn rerankCoarseBytes(rows: c_int, hidden: c_int, bits: u32) u64 {
    const r: u64 = @intCast(rows);
    const h: u64 = @intCast(hidden);
    const packed_bytes = r * h * @as(u64, bits) / 8;
    const groups = h / 64;
    return packed_bytes + 2 * (r * groups * 2);
}

/// A contiguous copy of `q`'s leading `n` rows (packed weights and both group tables), unevaluated; the caller frees
/// it.
fn leadingCopy(s: mlx.mlx_stream, q: QLinear, n: c_int) !QLinear {
    var parts: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (parts) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    for ([_]mlx.mlx_array{ q.w, q.s, q.b }, &parts) |a, *out| {
        const sh = mlx.getShape(a);
        if (sh.len != 2) return error.UnsupportedDraftHeadShape;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, a, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ n, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
        out.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_copy(out, view, s));
    }
    return .{ .w = parts[0], .s = parts[1], .b = parts[2] };
}

/// `rc` plus a copy of its leading `keep` rows (`RerankCoarse.sub`), evaluated. A `keep` under TOP32_MIN_ROWS or not
/// under the vocab, or a refused copy, leaves the whole head alone.
pub fn withVocabSubset(s: mlx.mlx_stream, rc: RerankCoarse, keep: c_int) RerankCoarse {
    if (keep < TOP32_MIN_ROWS or keep >= rc.rows) return rc;
    var sub = leadingCopy(s, rc.q, keep) catch |err| {
        log.warn("[mtp] draft rerank vocab subset failed ({s}) - drafting over the whole vocab\n", .{@errorName(err)});
        return rc;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    for ([_]mlx.mlx_array{ sub.w, sub.s, sub.b }) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        sub.deinit();
        log.warn("[mtp] draft rerank vocab subset eval failed - drafting over the whole vocab\n", .{});
        return rc;
    };
    var out = rc;
    out.sub = sub;
    out.sub_rows = keep;
    return out;
}

/// The coarse head of a trunk format's lm_head, whose rows exist only as what it computes: `dense`, those `[V, H]`
/// rows, requantized to the coarse width and to the re-score copy (`RerankCoarse.exact`), with the vocab subset on a
/// vocab of DRAFT_VOCAB_MIN ids or more. Infallible: every refusal is logged and keeps the full-readout path.
pub fn buildRerankCoarseRows(s: mlx.mlx_stream, dense: mlx.mlx_array, bits: u32) ?RerankCoarse {
    if (bits == 0 or bits >= RERANK_EXACT_BITS) return null;
    mlx.check(mlx.mlx_array_eval(dense)) catch {
        log.warn("[mtp] draft rerank: the head's rows failed to evaluate - keeping the full-readout path\n", .{});
        return null;
    };
    const sh = mlx.getShape(dense);
    if (sh.len != 2 or sh[0] < TOP32_MIN_ROWS) return null;
    var coarse = requantizeRows(s, dense, RERANK_COARSE_GROUP, bits, 32768) catch |err| {
        log.warn("[mtp] draft rerank coarse build failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    var exact = requantizeRows(s, dense, RERANK_COARSE_GROUP, RERANK_EXACT_BITS, 32768) catch |err| {
        coarse.deinit();
        log.warn("[mtp] draft rerank re-score copy failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    for ([_]mlx.mlx_array{ coarse.w, coarse.s, coarse.b, exact.w, exact.s, exact.b }) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        coarse.deinit();
        exact.deinit();
        log.warn("[mtp] draft rerank coarse eval failed - keeping the full-readout path\n", .{});
        return null;
    };
    const rc: RerankCoarse = .{ .q = coarse, .bits = bits, .group_size = RERANK_COARSE_GROUP, .rows = sh[0], .exact = exact };
    return withVocabSubset(s, rc, if (sh[0] >= DRAFT_VOCAB_MIN) DRAFT_VOCAB_ROWS else 0);
}

/// Quantize the dense rows `w` to (`to_gs`, `to_bits`) affine, chunk-wise over rows so the bf16 transient stays bounded
/// (~chunk_rows x in_features x 2 B instead of the whole matrix). Rows quantize independently in MLX's affine packing,
/// so the per-chunk triples concatenate along axis 0 into a valid whole. The rows come in dense: a trunk format head's
/// (`formatHeadRows`); a quantized source is another format's head.
pub fn requantizeRows(s: mlx.mlx_stream, w: mlx.mlx_array, to_gs: u32, to_bits: u32, chunk_rows: c_int) !QLinear {
    const w_shape = mlx.getShape(w);
    if (w_shape.len != 2) return error.UnsupportedDraftHeadShape;
    const rows: c_int = w_shape[0];

    const wv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(wv);
    const sv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(sv);
    const bv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(bv);

    var r0: c_int = 0;
    while (r0 < rows) : (r0 += chunk_rows) {
        const r1: c_int = @min(rows, r0 + chunk_rows);

        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        {
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try sliceRows(&raw, w, r0, r1, s);
            try mlx.check(mlx.mlx_astype(&dense, raw, .bfloat16, s));
        }

        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            dense,
            mlx.mlx_optional_int.some(@intCast(to_gs)),
            mlx.mlx_optional_int.some(@intCast(to_bits)),
            "affine",
            .{}, // global_scale
            s,
        ));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var part = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_vector_array_get(&part[0], triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&part[1], triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&part[2], triple, 2));
        // Realize the chunk so its dense transient can be reclaimed before
        // the next chunk builds (lazy eval would otherwise stack them all).
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for (part) |p| _ = mlx.mlx_vector_array_append_value(ev, p);
            try mlx.check(mlx.mlx_eval(ev));
        }
        _ = mlx.mlx_vector_array_append_value(wv, part[0]);
        _ = mlx.mlx_vector_array_append_value(sv, part[1]);
        _ = mlx.mlx_vector_array_append_value(bv, part[2]);
        for (part) |p| _ = mlx.mlx_array_free(p);
    }

    var out = QLinear{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_concatenate_axis(&out.w, wv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.s, sv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.b, bv, 0, s));
    return out;
}

fn sliceRows(out: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    const start = [_]c_int{ r0, 0 };
    const stop = [_]c_int{ r1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(out, src, &start, 2, &stop, 2, &strides, 2, s));
}

// ── The draft's readout, shortlist and pick ──
//
// What `x` is, is load-bearing: whatever the lm_head consumes, the MIXER output, not the pre-mixer stream the head
// hands back for the next step.

/// `argmax(where(mask, -inf, logits))` as a lazy [.., 1] int32 id - the tail
/// every full-readout draft path shares.
pub fn maskAndArgmax(s: mlx.mlx_stream, logits_in: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_set(&logits, logits_in));
    if (suppress_mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        _ = mlx.mlx_array_free(logits);
        logits = masked;
    }
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, logits, -1, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, amax, .int32, s));
    return out;
}

/// Full trunk-lm_head readout + argmax: the draft path once the coarse head is gone.
pub fn fullReadoutArgmax(
    s: mlx.mlx_stream,
    target: *const Trunk,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    const logits = try linear.lmHeadForeign(target, x);
    defer _ = mlx.mlx_array_free(logits);
    return maskAndArgmax(s, logits, suppress_mask);
}

/// `-inf` on every suppressed id of a coarse readout, in place; a vocab subset's readout (`RerankCoarse.sub`) is
/// narrower than the mask, and its ids are the mask's leading ones.
fn rerankMaskCoarseFormat(s: mlx.mlx_stream, logits: *mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !void {
    // The mask goes on the COARSE row: a reserved id that never enters the
    // shortlist cannot be re-scored back into the draft.
    const full = suppress_mask orelse return;
    const msh = mlx.getShape(full);
    const lsh = mlx.getShape(logits.*);
    var lead = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lead);
    var m = full;
    if (msh.len == 1 and lsh.len >= 1 and msh[0] > lsh[lsh.len - 1]) {
        try mlx.check(mlx.mlx_slice(&lead, full, &[_]c_int{0}, 1, &[_]c_int{lsh[lsh.len - 1]}, 1, &[_]c_int{1}, 1, s));
        m = lead;
    }
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits.*, s));
    _ = mlx.mlx_array_free(logits.*);
    logits.* = masked;
}

/// Shortlist one coarse row `flat` `[rows]` and re-score it through the head's 8-bit copy (`RerankCoarse.exact`: a
/// trunk format's head has no rows to gather); a vocab subset's shortlist still spans the vocab. Null retires the coarse
/// head, so every later step lands on the caller's fallback.
fn rerankShortlistRowFormat(
    s: mlx.mlx_stream,
    coarse: *?RerankCoarse,
    flat: mlx.mlx_array,
    rows: c_int,
    x_row: mlx.mlx_array,
) !?Shortlist {
    // `rows` is the coarse readout's width; a vocab subset's shortlist still spans the vocab
    const vocab: c_int = if (coarse.*) |rc| rc.rows else rows;
    const copy = (if (coarse.*) |rc| rc.exact else null) orelse return error.DirectModelRerankNotTrunkFormat;
    const cands = draftTop32(s, flat, rows) catch |err| {
        log.warn("[mtp] draft rerank shortlist failed ({s}) - dropping to the full readout\n", .{@errorName(err)});
        var dead = coarse.*.?;
        dead.deinit();
        coarse.* = null;
        return null;
    };
    errdefer _ = mlx.mlx_array_free(cands);

    // Gathered packed rows are self-contained (w [V, K*bits/32], scales/biases
    // [V, K/gs]), so a 32-row quantized matmul is exact.
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    var s32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_take_axis(&w32, copy.w, cands, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s32, copy.s, cands, 0, s));
    if (copy.b.ctx != null)
        try mlx.check(mlx.mlx_take_axis(&b32, copy.b, cands, 0, s));

    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(
        &exact,
        x_row,
        w32,
        s32,
        b32,
        true,
        mlx.mlx_optional_int.some(@intCast(RERANK_COARSE_GROUP)),
        mlx.mlx_optional_int.some(@intCast(RERANK_EXACT_BITS)),
        "affine",
        s,
    ));
    return .{ .cands = cands, .exact = exact, .rows = vocab };
}

/// The coarse readout of one row over `head` (`RerankCoarse.readout`: the vocab subset when one is built) as a flat
/// `[rows]` array, reserved ids already at `-inf` - the front half every rerank draft shares.
fn rerankCoarseFlatFormat(
    s: mlx.mlx_stream,
    rc: *const RerankCoarse,
    head: RerankCoarse.Readout,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var coarse_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coarse_logits);
    try mlx.check(mlx.mlx_quantized_matmul(
        &coarse_logits,
        x,
        head.q.w,
        head.q.s,
        head.q.b,
        true,
        mlx.mlx_optional_int.some(@intCast(rc.group_size)),
        // The BUILT width and group, never `DRAFT_HEAD_BITS`: the packed
        // geometry implies K, so a read at another width reads a different vocab.
        mlx.mlx_optional_int.some(@intCast(rc.bits)),
        "affine",
        s,
    ));
    try rerankMaskCoarseFormat(s, &coarse_logits, suppress_mask);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{head.rows};
    try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
    return flat;
}

/// The exact re-scored shortlist behind a rerank draft: the coarse head's top-32 candidate ids plus their re-scores.
/// Null means the coarse head is unavailable (never built, or retired by a shortlist failure).
pub fn rerankShortlist(s: mlx.mlx_stream, coarse: *?RerankCoarse, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?Shortlist {
    const rc = if (coarse.*) |*p| p else return null;
    // A trunk format head's coarse head (`buildRerankCoarseRows`) reads its vocab subset and its re-score copy.
    if (rc.exact == null and rc.sub == null) return error.DirectModelRerankNotTrunkFormat;
    // the vocab subset's rows when one is built
    const head = rc.readout();
    const flat = try rerankCoarseFlatFormat(s, rc, head, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);
    return rerankShortlistRowFormat(s, coarse, flat, head.rows, x);
}

/// The greedy draft's tail: argmax over the re-scored shortlist, mapped back to
/// a vocab id as a lazy int32 array.
pub fn shortlistArgmax(s: mlx.mlx_stream, sl: Shortlist) !mlx.mlx_array {
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, sl.exact, -1, false, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, amax, 0, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, picked, .int32, s));
    return out;
}

/// One greedy draft proposal via the rerank scheme: coarse readout -> exact top-32 shortlist (two dispatches) -> the
/// 32 rows' re-score -> argmax, a lazy [1,1] int32 id. Null when the coarse head is unavailable: never built, or a
/// shortlist-kernel failure just dropped it (so every later step of the chain lands on the caller's fallback too).
pub fn rerankSelect(s: mlx.mlx_stream, coarse: *?RerankCoarse, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mlx.mlx_array {
    var sl = (try rerankShortlist(s, coarse, x, suppress_mask)) orelse return null;
    defer sl.deinit();
    return try shortlistArgmax(s, sl);
}

// ── The top-32 shortlist (the mlx.fast challenge's draft-rerank scheme, submission 942e5ab2) ──
//
// MLX's GPU argpartition is a full multi-block argsort (14 dependent dispatches for 32 read values); these take the
// exact top 32 in two. Ordinal: a monotone map from float into uint32 inducing (value asc, NaN above every number,
// -0.0 == +0.0). Real values never map to 0, so the zero-initialized empty slots can never be selected while a
// simdgroup still has real candidates (guaranteed by rows >= TILES*TG).
const TOP32_ORDINAL_HEADER = kernelSource("kernels/spec/mtp_top32_ordinal_header.metal");

// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32 as (ordinal, index) pairs.
const TOP32_PARTIAL_SRC = kernelSource("kernels/spec/mtp_top32_partial.metal");

// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32 ids (ascending by (value, index)).
const TOP32_FINAL_SRC = kernelSource("kernels/spec/mtp_top32_final.metal");

const TOP32_K: c_int = 32;
const TOP32_TG: c_int = 256;
const TOP32_TILES: c_int = 64;
pub const TOP32_CANDS: c_int = TOP32_TILES * TOP32_K;
/// Minimum row count: below TILES*TG some simdgroups hold fewer than 32 real
/// candidates and the zero-initialized empty slots become reachable.
pub const TOP32_MIN_ROWS: c_int = TOP32_TILES * TOP32_TG;

var top32_partial_cached: ?mlx.mlx_fast_metal_kernel = null;
var top32_final_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getTop32Partial() !mlx.mlx_fast_metal_kernel {
    if (top32_partial_cached) |k| return k;
    const input_names = [_][*:0]const u8{"logits"};
    const output_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_partial",
        in_vec,
        out_vec,
        TOP32_PARTIAL_SRC,
        TOP32_ORDINAL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_partial_cached = kernel;
    return kernel;
}

fn getTop32Final() !mlx.mlx_fast_metal_kernel {
    if (top32_final_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const output_names = [_][*:0]const u8{"token_ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_finalize",
        in_vec,
        out_vec,
        TOP32_FINAL_SRC,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_final_cached = kernel;
    return kernel;
}

/// Exact top-32 ids of `row` ([rows], any float dtype) as a lazy [32] uint32 array, ascending by (value, index) - ties
/// break toward the higher index, NaN above every number: the partial launch (`draftTop32Cands`), then the finalize.
/// `rows` rides as a template int (stable per model, so MLX caches one specialization per vocab width).
pub fn draftTop32(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    const cands = try draftTop32Cands(s, row, rows);
    defer for (cands) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const fk = try getTop32Final();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{TOP32_K};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, cands[0]);
    _ = mlx.mlx_vector_array_append_value(inputs, cands[1]);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, fk, inputs, cfg, s));
    var token_ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&token_ids, outputs, 0));
    return token_ids;
}

/// Whether `draftTop32`'s Metal kernels take a `rows`-wide row.
pub fn top32RowsOk(rows: c_int) bool {
    if (rows < TOP32_MIN_ROWS) return false;
    const per_thread = @divTrunc(rows + TOP32_TILES * TOP32_TG - 1, TOP32_TILES * TOP32_TG);
    return per_thread <= 32;
}

/// `draftTop32`'s first launch: the TOP32_CANDS (ordinal, index) candidates, `[TOP32_CANDS]` uint32 each, that its
/// finalize reduces (the fused pick runs that finalize inside its own launch). A Metal GPU stream (M5-class: CUDA is
/// never present); the caller frees both.
pub fn draftTop32Cands(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) ![2]mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (!top32RowsOk(rows)) return error.UnsupportedTop32Shape;
    const pk = try getTop32Partial();
    var cand_ord = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cand_ord);
    var cand_idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cand_idx);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const cand_shape = [_]c_int{TOP32_CANDS};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TILES * TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RC", rows));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, row);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, pk, inputs, cfg, s));
    try mlx.check(mlx.mlx_vector_array_get(&cand_ord, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&cand_idx, outputs, 1));
    return .{ cand_ord, cand_idx };
}
