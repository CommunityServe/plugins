//! copied_from: src/plugins/qwen4_exp/mtp.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's MTP head (`Qwen4Mtp`, `Qwen4MtpState` swapped onto the module): its load, forward over the pre-mixer
//! stream, QSA layer, history capture and rollback, the draft rerank, and the head-graph meters.

const std = @import("std");
const mtp_mod = @import("../../sdk/spec/mtp/mtp.zig");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const Weights = @import("../../sdk/config/model.zig").Weights;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const init = @import("../../engines/mlx/load/init.zig").init;
const initMoeLayers = @import("../../engines/mlx/load/layers_layered.zig").initMoeLayers;
const weightTriple = @import("../../engines/mlx/load/weights.zig").weightTriple;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("trunk.zig").Trunk;
const loadHcWeights = @import("load.zig").loadHcWeights;
const QsaHeadMarkSet = @import("qsa/history.zig").QsaHeadMarkSet;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const HcWeights = @import("weights.zig").HcWeights;


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

/// A draft row written past a padded history append. Key slots `[live_end, dead_end)`
/// are dead history rows: hidden from its attention and its QSA block selection. The
/// row ropes at `live_end + (slot - dead_end)`, the position it would hold with no pad.
pub const HeadPlace = struct {
    /// 0-d int32 slot one past the last live history row.
    live_end: mlx.mlx_array,
    dead_end: c_int,
};

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
