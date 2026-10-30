//! copied_from: src/engines/mlx/spec/mtp/chain.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MTP draft chain state: history stash, pre-drafts, prompt lookup, lazy pre-draft, padded head history.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const HeadPlace = @import("../../../../plugins/qwen4_exp/mtp.zig").HeadPlace;
const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const transformer_mod = @import("../../transformer.zig");
const tokenizer_mod = @import("../../../../sdk/tokenizer/tokenizer.zig");
const model_mod = @import("../../../../sdk/config/model.zig");
const mtp_lookup = @import("../mtp_lookup.zig");
const group_planner = @import("../mtp_group_planner.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = @import("../../generate/generator.zig").Generator;
const MAX_ROUND_DRAFTS = @import("../mtp_head_ref.zig").MAX_ROUND_DRAFTS;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;
const deinit = @import("../../generate/generator.zig").Generator.deinit;


/// Qwen native-MTP speculative round. Structure mirrors `nextDrafter`
/// (same verify invariant: cache.step = prompt_len + emitted, t1 NOT in
/// cache on entry, verify input `[t1, draft[0..m-1]]`, bonus from row m,
/// partial-accept snapshot/restore + re-forward) with one addition: the
/// MTP head's committed-history KV cache. Draft steps append m temporary
/// entries built from MTP-PREDICTED hiddens; after the verify decision we
/// restore the round-boundary snapshot and re-append the committed pairs
/// from TRUE trunk hiddens, so the history never accumulates drift.
pub const MtpHistStash = struct {
    /// `[n]` int32 committed token ids: `[t1, drafts[0..accepted]]`.
    ids: mlx.mlx_array,
    /// `ids` on the host.
    host_ids: [MAX_ROUND_DRAFTS + 1]u32 = undefined,
    /// `[1, n, H]` trunk hiddens paired 1:1 with `ids` (a lazy concat of
    /// last_hidden + a verify-capture slice — the handle pins the ~90 KB
    /// parent until consumed, deliberately NOT a deep copy).
    hidden: mlx.mlx_array,
    n: usize,
    /// Head-cache position of ids[0]'s entry (the producing round's
    /// mtp_off0). The consume-time truncate drops the producing round's
    /// stale draft tail past it.
    off0: usize,
    /// The whole verify row `[t1, drafts[0..m]]` and its `[1, 1+m, H]` hiddens, for a
    /// padded append (`mtpPaddedHistory`); `ids`/`hidden` are its first `n` rows.
    pad: ?struct { ids: mlx.mlx_array, hidden: mlx.mlx_array, rows: usize } = null,

    pub fn deinit(self: *MtpHistStash) void {
        _ = mlx.mlx_array_free(self.ids);
        _ = mlx.mlx_array_free(self.hidden);
        if (self.pad) |p| {
            _ = mlx.mlx_array_free(p.ids);
            _ = mlx.mlx_array_free(p.hidden);
        }
    }
};

/// One round's lazily-built MTP draft chain — the Phase 0/1 state that
/// cross-round pre-drafting (`mtpMaybePreDraft`) moves into the PREVIOUS
/// round's tail. Owns every handle it holds; `deinit` frees whatever was
/// built so far, so it is safe on partial builds and on a pre-draft left
/// unconsumed at request end.
pub const MtpPreDraft = struct {
    plan: MtpRoundPlan,
    /// Head-cache position of this round's first draft entry.
    off0: usize,
    t1: u32,
    /// `[1]` int32 t1 — the verify input needs it again in Phase 3.
    t1_arr: mlx.mlx_array,
    /// Host draft ids, filled at the Phase 4b sync. len = plan.m_hi.
    drafts: []u32,
    /// Lazy `[1]` int32 draft ids. len = plan.m_hi; [0..n_drafted) valid.
    draft_arrs: []mlx.mlx_array,
    n_drafted: u32,
    /// Generated index the first draft proposes (a coupled draft's key).
    pos0: usize = 0,
    head_input_rows: u32 = 1,
    /// Chunk-A log-confidence graphs (two-chunk plans only). len m_lo.
    conf_arrs: ?[]mlx.mlx_array,
    n_conf: u32,
    /// Proposal densities q, present when depth 0 samples; an argmax
    /// depth's row is its one-hot. len m_hi; [0..n_qp) valid.
    q_probs: ?[]mlx.mlx_array,
    n_qp: u32,
    /// Chain hidden after the last built step (owned) — a chunk-B
    /// extension resumes from it; freed once the chain is complete.
    h_chain: ?mlx.mlx_array,
    /// Tokens actually drafted this round; starts at m_lo, grows to
    /// m_hi iff the confidence gate clears at the chunk boundary.
    m: u32,
    /// Drafts are a context continuation (`mtpLookupChain`): nothing to
    /// build, and no MTP statistics are fed.
    lookup: bool = false,
    /// Set when step 0 consumed a padded history append: every draft step sits past its dead rows.
    place: ?HeadPlace = null,
    /// The normalized embedding of draft `en_step` (owned), built by that step's fused pick; the next step's head
    /// forward reads it instead of `embedding` + `rmsNorm`, and only while it is that step's predecessor's.
    en_next: ?mlx.mlx_array = null,
    en_step: u32 = 0,

    pub fn deinit(self: *MtpPreDraft, allocator: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.t1_arr);
        if (self.place) |p| _ = mlx.mlx_array_free(p.live_end);
        for (self.draft_arrs[0..self.n_drafted]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(self.draft_arrs);
        if (self.conf_arrs) |slots| {
            for (slots[0..self.n_conf]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.q_probs) |slots| {
            for (slots[0..self.n_qp]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.h_chain) |h| _ = mlx.mlx_array_free(h);
        if (self.en_next) |e| _ = mlx.mlx_array_free(e);
        allocator.free(self.drafts);
    }

};

/// Prompt lookup may stand in for the MTP chain (the test seam's false keeps the chain).
pub var mtp_lookup_override: ?bool = null;
pub fn mtpLookupEnabled() bool {
    return mtp_lookup_override orelse true;
}

/// End a round that stays out of the cost table. The regime clock runs from
/// the last observed round, so an unobserved round must end the interval too.
pub fn mtpRoundUntimed(self: *Generator) void {
    if (self.mtp_regime_clock) |*c| c.reset();
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
}

/// Price a lookup round into the table's lookup row: the wall since the previous round
/// ended, the quantity the width grid stores. No sample before the regime clock runs.
pub fn mtpLookupObserve(self: *Generator, drafts: u32, tokens: u32) void {
    const c = if (self.mtp_regime_clock) |*c| c else return;
    const ms = @as(f32, @floatFromInt(c.read())) / @as(f32, std.time.ns_per_ms);
    _ = self.xfm.round_cost.observeLookup(drafts, self.mtpKvLen(), ms, @floatFromInt(tokens), self.spec_cost_solo);
}

/// What this machine measured for an MTP round at `width` and for each lookup size.
pub fn mtpLookupCosts(self: *const Generator, width: u32) mtp_lookup.Costs {
    const t = &self.xfm.round_cost;
    const kv = self.mtpKvLen();
    var costs = mtp_lookup.Costs{ .mtp_ms = if (t.bucketToRead(kv)) |b| t.roundMs(width, b) else null };
    const bucket = t.bucketOf(kv);
    for (&costs.lookup_ms, 0..) |*ms, k| ms.* = t.lookupMs(@intCast(k), bucket);
    return costs;
}

/// Count a round's accepted drafts toward its kind and that kind's EMA.
pub fn mtpRoundAcceptObserve(self: *Generator, lookup: bool, drafted: u32, accepted: u32) void {
    if (lookup) {
        self.mtp_lookup_streak = accepted == drafted;
        self.mtp_lookup_accepted += accepted;
        self.mtp_lookup_ema = mtp_lookup.emaStep(self.mtp_lookup_ema, accepted);
    } else {
        self.mtp_accepted_tokens += accepted;
        self.mtp_round_ema = mtp_lookup.emaStep(self.mtp_round_ema, accepted);
        self.mtp_lookup_ema = mtp_lookup.driftStep(self.mtp_lookup_ema);
    }
}

/// The committed-token index, synced to `generated_ids`.
pub fn mtpLookupIndex(self: *Generator, allocator: std.mem.Allocator) !*mtp_lookup.Index {
    if (self.mtp_lookup_idx == null) {
        var idx = mtp_lookup.Index.init(allocator);
        errdefer idx.deinit();
        try idx.extend(self.prompt_ids_owned);
        self.mtp_lookup_idx = idx;
        self.mtp_lookup_gen = 0;
    }
    const gen = self.generated_ids.items;
    try self.mtp_lookup_idx.?.extend(gen[self.mtp_lookup_gen..]);
    self.mtp_lookup_gen = gen.len;
    return &self.mtp_lookup_idx.?;
}

/// A chain of the context's continuation after `t1`, or null when the gate
/// prefers the MTP head. Single-stream rounds only.
/// How many context-continuation drafts the lookup gate takes after `t1`; 0 = the MTP head.
pub fn mtpLookupDrafts(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !u32 {
    if (!mtpLookupEnabled() or self.mtp_batch_head) return 0;
    if (group_planner.enabled() and self.mtp_planner_owned) return 0;
    const idx = try self.mtpLookupIndex(allocator);
    const remaining: u32 = @intCast(self.max_tokens -| self.completion_tokens -| 1);
    const lookup_cap = depth_bounds.lookupCap(mtp_lookup.MAX_DRAFT_STRONG, depth_bounds.active);
    const got = idx.match(t1, lookup_cap);
    return mtp_lookup.gate(got, remaining, self.mtp_lookup_ema, self.mtp_round_ema, plan.m_lo, self.mtp_lookup_streak, self.mtpLookupCosts(plan.m_lo));
}

/// Round N's accept decision as lazy arrays: `a` (0-d int32) drafts accepted, the next
/// `t1` (`[1]` int32) and the next `h_prev` (`[1,1,H]`), all before any host read.
pub const MtpLazyVerdict = struct {
    a: mlx.mlx_array,
    t1: mlx.mlx_array,
    h_prev: mlx.mlx_array,

    pub fn deinit(self: *MtpLazyVerdict) void {
        _ = mlx.mlx_array_free(self.a);
        _ = mlx.mlx_array_free(self.t1);
        _ = mlx.mlx_array_free(self.h_prev);
    }
};




// ── tests ──


/// A loaded qwen4_exp pack and a byte tokenizer for generator-level lazy pre-draft runs.
const LazyRig = struct {
    config: model_mod.ModelConfig,
    xfm: Transformer,
    tok: Tokenizer,

    fn deinit(rig: *LazyRig, a: std.mem.Allocator) void {
        rig.tok.deinit();
        rig.xfm.deinit();
        rig.config.deinit(a);
        a.destroy(rig);
    }

};
