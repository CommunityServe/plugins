//! This plugin's own: qwen4_exp's MTP head (the copied `plugins/qwen4_exp/mtp.zig`) behind the kit's `NativeDrafter`,
//! on the state the model keeps for each slot, with #1005's round (its `engines/mlx/spec/mtp/`): the same history rows,
//! positions and grouping, the same chain, verify and rewind, and #1005's round policy itself (planner, depth, ev_plan,
//! ev_fix, regime, adaptive, observe, the prompt lookup), copied unchanged and run on a request's state as #1005's
//! engine keeps it (`engines/mlx/generate/generator.zig`, this plugin's edge).
//!
//! The head's history follows every committed token. Row r pairs the trunk's stream at position r with token r + 1
//! and sits at position r + 1. A forward's rows whose next token is known join the history in one head forward: a
//! prompt chunk's rows with the prompt's tokens after them (`resume_prompt` hands the prompt in), as #1005's prefill
//! appends each chunk; the last row's stream waits as `last` for the token after it.
//!
//! A round from t1 at `pos` runs the plan #1005's policy makes for it: the context's own continuation when the lookup
//! gate prefers it, else the head's chain to the plan's base depth (the first step from `last` and t1, each next from the
//! head's own stream; on an EXL3 pack the fused pick hands the next step its embedding), and on a two-chunk plan the
//! chain's confidence read at that boundary and the extension when it clears the plan's bar. Then one trunk forward
//! over [t1, drafts] in `.verify` (each recurrent row's state captured, the n-gram leaf deferred so the graph builds
//! while the drafts run), the drafts the trunk's argmax agrees with kept (at most `accept_cap`), the trunk rewound to
//! them (the KV truncated, the recurrent states from the capture), and the head truncated to the round's origin, its
//! rows waiting as a stash (#1005's deferred history append): the next head chain appends the whole verify row, the
//! rejected drafts too, and drafts past them out of their sight (#1005's padded history, `HeadPlace`); anything else
//! appends the committed rows alone. A greedy head round with budget for another builds the next round's chunk A from
//! its own verdict before the host reads it (#1005's lazy pre-draft): its whole row joins the head at once, the next
//! chain placed past it at the verdict's live end, kept when the read agrees and the request goes on with a head round.
//! A round the policy keeps serial (its adaptive switch, a probe, a disabled request) drafts nothing.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const q4 = @import("../mlx_serve/plugins/qwen4_exp/plugin.zig");
const mtp = @import("../mlx_serve/plugins/qwen4_exp/mtp.zig");
const cost_profile = @import("../mlx_serve/plugins/qwen4_exp/cost_profile.zig");
const draft_fuse = @import("../mlx_serve/sdk/spec/mtp/draft_fuse.zig");
const mtp_mod = @import("../mlx_serve/sdk/spec/mtp/mtp.zig");
const ssm_entry = @import("../mlx_serve/engines/mlx/ssm/entry.zig");
const mtp_lookup = @import("../mlx_serve/engines/mlx/spec/mtp_lookup.zig");
const depth_bounds = @import("../mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const round_cost = @import("../mlx_serve/engines/mlx/spec/round_cost.zig");
const ev_plan = @import("../mlx_serve/engines/mlx/spec/mtp/ev_plan.zig");
const ev_costs = @import("../mlx_serve/engines/mlx/spec/mtp/ev_costs.zig");
const ev_fix = @import("../mlx_serve/engines/mlx/spec/mtp/ev_fix.zig");
const regime = @import("../mlx_serve/engines/mlx/spec/mtp/regime.zig");
const depth = @import("../mlx_serve/engines/mlx/spec/mtp/depth.zig");
const spec_sampling = @import("../mlx_serve/engines/mlx/sampling/spec_sampling.zig");
const generator = @import("../mlx_serve/engines/mlx/generate/generator.zig");
const MtpHeadRef = @import("../mlx_serve/engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const io_util = @import("../mlx_serve/sdk/util/io_util.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const kv_cache = @import("../mlx_serve/sdk/kv/cache.zig");
const checkpoint = @import("../mlx_serve/engines/mlx/ssm/checkpoint.zig");
const qsa_checkpoint = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig");
const QsaHeadMarkSet = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").QsaHeadMarkSet;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const kvBytes = @import("../cache/aux.zig").kvBytes;

const Array = mlx.mlx_array;
const KVCache = @typeInfo(@FieldType(ForwardCtx, "cache")).pointer.child;
const Generator = generator.Generator;
const MtpRoundPlan = ev_plan.MtpRoundPlan;
const Trunk = @FieldType(q4.Model, "trunk");
const OwnSlot = @typeInfo(@FieldType(@FieldType(q4.Model, "own"), "items")).pointer.child;
const gpa = std.heap.c_allocator;
const max_rows = abi.max_round_tokens;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// One slot's head state and history bookkeeping, on the heap: the head module points at the live `st`.
const Slot = struct {
    st: mtp.Qwen4MtpState,
    /// The prompt `resume_prompt` handed in: the tokens after a chunk's rows.
    prompt: std.ArrayList(u32) = .empty,
    /// The trunk's stream `[1, 1, hc*H]` at `next - 1`, whose row waits for the token at `next`.
    last: Array = .{ .ctx = null },
    /// Where the slot's next forward starts for the history to stay whole.
    next: u64 = 0,
    /// The history holds every committed row so far: rounds draft.
    synced: bool,
    /// The request's round policy, from its first round (#1005's per-request state).
    gen: ?Generator = null,
    /// Committed ids the engine's own forwards left on the device, read into `gen.generated_ids` at the next round.
    pending: std.ArrayList(Array) = .empty,
    /// The last drafted round's rows, until the round that reads the head appends them.
    stash: ?Stash = null,
};

/// A drafted round's history before the next round reads it (#1005's `MtpHistStash`, padded): the verify's rows
/// `[t1, drafts]` with the stream before each (`last ++ verify[0..n]`), from the round's origin `off0` (the head's live
/// rows) at position `pos`, of which the first `n` (t1 and the kept drafts) are committed. A head chain appends all of
/// them and drafts past the dead ones (`HeadPlace`, #1005's padded history); anything else appends the committed rows.
const Stash = struct {
    off0: usize,
    pos: u64,
    ids: [max_rows]i32 = undefined,
    rows: u32,
    n: u32,
    hidden: Array,

    fn deinit(st: *Stash) void {
        if (st.hidden.ctx != null) _ = mlx.mlx_array_free(st.hidden);
    }

    /// The head's live rows once the stash is appended.
    fn live(st: *const Stash) usize {
        return st.off0 + st.n;
    }
};

fn dropStash(sl: *Slot) void {
    if (sl.stash) |*st| st.deinit();
    sl.stash = null;
}

/// A round's drafts and verify before its accept: the drafts `[n]` (int32, dispatched), the verify's logits
/// `[1 + n, vocab]` and streams `[1, 1 + n, hc*H]`, what its n-gram leaf still owes, the slot's KV step before it, and
/// the plan it ran.
const Built = struct {
    n: u32 = 0,
    /// The drafts are the context's continuation (`mtp_lookup`), not the head's.
    lookup: bool = false,
    /// The head's row count at the round's start; null: the round drafted nothing.
    off0: ?usize = null,
    plan: MtpRoundPlan = .{ .m_lo = 0, .m_hi = 0, .tau_ln = 0 },
    drafts: Array = .{ .ctx = null },
    logits: Array = .{ .ctx = null },
    streams: Array = .{ .ctx = null },
    owed: abi.Owed = .none,
    step: usize = 0,

    fn deinit(b: *Built) void {
        for ([_]Array{ b.drafts, b.logits, b.streams }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
    }
};

/// A verify the engine accepts itself (`NativeDrafter.verify`), until `finish`.
const Open = struct {
    slot: u32,
    pos: u64,
    t1: u32,
    drafts: [max_rows]u32 = undefined,
    n_drafts: u32,
    lookup: bool,
    off0: ?usize,
    plan: MtpRoundPlan,
    step: usize,
    watch: io_util.Stopwatch,
    /// The verify's streams `[1, 1 + n_drafts, hc*H]`.
    streams: Array = .{ .ctx = null },

    fn deinit(o: *Open) void {
        if (o.streams.ctx != null) _ = mlx.mlx_array_free(o.streams);
    }
};

/// The next round's opening, built and dispatched at this round's tail (#1005's cross-round pre-draft): the head's chunk
/// A for the plan made then, or the lookup drafts the gate took. The next round from (slot, pos, t1) takes it; any other
/// drops it, and its draft rows leave the head.
const Pre = struct {
    slot: u32,
    pos: u64,
    t1: u32,
    plan: MtpRoundPlan,
    /// The head's row count before the chain's rows.
    off0: usize,
    lookup: bool = false,
    n_lookup: u32 = 0,
    lookup_ids: [max_rows]u32 = undefined,
    chain: ?Chain = null,
};

/// #1005's cross-round pre-draft (`mtpPredraftEnabled`, on by default there).
const predraft = true;

/// #1005's lazy pre-draft (`mtpLazyPredraftEnabled`, on by default there): a greedy head round builds the next round's
/// chunk A from its own verdict before the host reads it, so the head drafts while the read and the commit run.
const lazy_predraft = true;

/// A lazy pre-draft (#1005's `mtpLazyPreDraft`): the next round's plan and chunk A, built before this round's read.
/// The head holds this round's whole row from `origin` (its streams `hidden`, kept for a discard to stash), the chain
/// placed past its dead rows at the verdict's live end.
const Lazy = struct {
    plan: MtpRoundPlan,
    chain: ?Chain,
    origin: usize,
    hidden: Array,

    fn deinit(z: *Lazy) void {
        if (z.chain) |*c| c.deinit();
        z.chain = null;
        if (z.hidden.ctx != null) _ = mlx.mlx_array_free(z.hidden);
        z.hidden = .{ .ctx = null };
    }
};

pub const Drafter = struct {
    m: *q4.Model,
    eos: [abi.max_eos]u32 = @splat(0),
    n_eos: u32 = 0,
    pre: ?Pre = null,
    /// The model's share of the round policy (the engine keeps it on its Transformer): the measured round-cost
    /// table, the chunk-A boundary's price, the head's seed.
    pm: generator.Model,
    slots: std.AutoHashMapUnmanaged(u32, *Slot) = .empty,
    open: ?Open = null,

    pub fn init(m: *q4.Model, shell: *const abi.Shell) Drafter {
        var d: Drafter = .{ .m = m, .pm = .{ .trunk = &m.trunk, .round_cost = .{ .layout = round_cost.layoutFor(&m.cfg) } } };
        d.n_eos = @min(shell.eos_count, abi.max_eos);
        @memcpy(d.eos[0..d.n_eos], shell.eos[0..d.n_eos]);
        return d;
    }

    /// The pack carries an MTP head.
    pub fn has(d: *const Drafter) bool {
        return d.m.trunk.qwen4_mtp != null;
    }

    /// A two-chunk round's head path once, at load, on a scratch slot and head state: chunk A's steps with their
    /// confidences, the read at its boundary, chunk B. The policy prices two-chunk plans at the first boundaries it
    /// times and keeps that price for the process (`ev_fix.Model.read_ms`); timed while these pipelines compiled, it
    /// priced them out for good.
    pub fn warmTwoChunk(d: *Drafter) !void {
        const t = &d.m.trunk;
        const s = t.s;
        const lap = io_util.Stopwatch.init(io());
        const id = std.math.maxInt(u32) - 1;
        defer q4.releaseSlot(d.m, id);
        const toks = [_]u32{ 1, 2 };
        const slot_ids = [_]u32{id};
        const positions = [_]u32{0};
        const f: abi.Forward = .{ .mode = .prefill, .want_hidden = true, .rows = 2, .tokens = .of(&toks), .slots = .of(&slot_ids), .positions = .of(&positions) };
        var out: abi.ForwardOut = .{};
        const v = q4.forward(d.m, &f, &out);
        defer for ([_]abi.Array{ out.logits, out.hidden }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
        };
        if (!v.isOk() or out.hidden.ctx == null) return error.MtpWarmForward;
        var h_prev = mlx.mlx_array_new();
        sliceRows(&h_prev, .{ .ctx = out.hidden.ctx }, 1, 2, s) catch |e| {
            _ = mlx.mlx_array_free(h_prev);
            return e;
        };
        var st = t.qwen4MtpStateNew() catch |e| {
            _ = mlx.mlx_array_free(h_prev);
            return e;
        };
        defer st.deinit();
        t.qwen4MtpActivate(&st);
        defer t.qwen4MtpRelease(&st);
        var tok: i32 = 2;
        var c: Chain = .{ .t = t, .rerank = t.qwen4DraftRerankReady(), .mask = null, .h_prev = h_prev, .prev = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32) };
        defer c.deinit();
        const m_lo = 3;
        const m_hi = 5;
        try c.build(1, 0, m_lo, true);
        try c.dispatch(0, m_lo, true);
        _ = try spec_sampling.readChainConfidence(c.ids[0..m_lo], c.confs[0..m_lo], s);
        try c.build(1, m_lo, m_hi, false);
        const vec = mlx.mlx_vector_array_new_data(&c.ids, c.n_ids);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        log.info("[spec-warmup] two-chunk head path {d}->{d} ({d} ms).\n", .{ m_lo, m_hi, lap.read() / std.time.ns_per_ms });
    }

    pub fn deinit(d: *Drafter) void {
        d.dropPre();
        if (d.open) |*o| o.deinit();
        d.open = null;
        var it = d.slots.valueIterator();
        while (it.next()) |sl| d.drop(sl.*);
        d.slots.deinit(gpa);
    }

    fn drop(d: *Drafter, sl: *Slot) void {
        dropStash(sl);
        d.endRequest(sl);
        d.m.trunk.qwen4MtpRelease(&sl.st);
        sl.st.deinit();
        sl.pending.deinit(gpa);
        sl.prompt.deinit(gpa);
        if (sl.last.ctx != null) _ = mlx.mlx_array_free(sl.last);
        gpa.destroy(sl);
    }

    pub fn release(d: *Drafter, id: u32) void {
        if (d.pre) |p| if (p.slot == id) d.dropPre();
        if (d.open) |*o| if (o.slot == id) {
            o.deinit();
            d.open = null;
        };
        if (d.slots.fetchRemove(id)) |kv| d.drop(kv.value);
    }

    /// The slot's state. A forward at position 0 starts its history over (a new prompt); a slot first seen past 0 has
    /// no history.
    fn slotAt(d: *Drafter, id: u32, pos: u64) !*Slot {
        if (d.slots.get(id)) |sl| {
            if (pos == 0 and sl.next != 0) try d.restart(sl);
            return sl;
        }
        const sl = try gpa.create(Slot);
        errdefer gpa.destroy(sl);
        sl.* = .{ .st = try d.m.trunk.qwen4MtpStateNew(), .next = pos, .synced = pos == 0 };
        errdefer sl.st.deinit();
        try d.slots.put(gpa, id, sl);
        return sl;
    }

    fn restart(d: *Drafter, sl: *Slot) !void {
        const t = &d.m.trunk;
        const fresh = try t.qwen4MtpStateNew();
        d.endRequest(sl);
        dropStash(sl);
        t.qwen4MtpRelease(&sl.st);
        sl.st.deinit();
        sl.st = fresh;
        sl.prompt.clearRetainingCapacity();
        setLast(sl, .{ .ctx = null });
        sl.next = 0;
        sl.synced = true;
    }

    /// A request's end: its `[spec-stats]` lines said (`specStatsLines`), and its policy state gone, a healthy run's
    /// acceptance surface seeding the next request's plans (#1005's Generator.deinit). The server releases the slot as
    /// the request finishes, right after its own `[spec-stats]` line.
    fn endRequest(d: *Drafter, sl: *Slot) void {
        _ = d;
        for (sl.pending.items) |a| _ = mlx.mlx_array_free(a);
        sl.pending.clearRetainingCapacity();
        var g = sl.gen orelse return;
        sl.gen = null;
        var stats: [2048]u8 = undefined;
        const lines = specStatsLines(&stats, &g, g.mtpEvFixOn());
        if (lines.len > 0) log.info("{s}", .{lines});
        if (g.mtp) |head| {
            if (regime.mtpPinnedDepth() == null and !g.spec_disabled_runtime and g.mtp_attempted >= 8) head.setEvSeed(g.mtp_ev_accept, g.mtp_ev_m_lo_prev);
        }
        g.deinit(gpa);
    }

    /// The request's policy, made at its first round as #1005's generate init makes it: the depth cap from the head's
    /// cost profile (no `--mtp-max-depth` reaches a native drafter), the free cap, the profile's costs, depth 1.
    fn policyOf(d: *Drafter, sl: *Slot, own: *const OwnSlot) *Generator {
        if (sl.gen) |*g| return g;
        const head: MtpHeadRef = .{ .qwen4 = &d.m.trunk };
        const profile = cost_profile.qwen4G17CostProfileForKv(&d.m.trunk, own.cache.config);
        const cap = @min(ev_plan.resolveMtpDepthCapForProfile(0, profile), head.maxDepth());
        sl.gen = .{
            .xfm = &d.pm,
            .mtp = head,
            .timer = io_util.Stopwatch.init(io()),
            .prompt_ids_owned = sl.prompt.items,
            .model_has_mtp = true,
            .mtp_depth = cap,
            .mtp_depth_free = if (d.pm.mtp_depth_free != 0) d.pm.mtp_depth_free else ev_plan.mtpDepthCapFree(0),
            .mtp_ev_costs = ev_costs.mtpEvCosts(profile),
            .mtp_depth_current = depth_bounds.floorFor(depth_bounds.active, cap),
        };
        return &sl.gen.?;
    }

    /// The head's row count (its state on the module: `qwen4MtpActivate` swapped it there).
    fn headRows(d: *Drafter) usize {
        return d.m.trunk.qwen4_mtp.?.seq_offset;
    }

    /// The head's live rows for `sl` (its state active): the stash's end while one waits, else the head's own.
    fn liveRows(d: *Drafter, sl: *const Slot) usize {
        if (sl.stash) |*st| return st.live();
        return d.headRows();
    }

    /// The waiting stash appended as committed history alone (#1005's `mtpApplyStash`), for whatever reads the head next
    /// that is not a head chain: a lookup round, an engine forward, a snapshot.
    fn applyStash(d: *Drafter, sl: *Slot) !void {
        var st = sl.stash orelse return;
        sl.stash = null;
        defer st.deinit();
        const t = &d.m.trunk;
        t.qwen4MtpActivate(&sl.st);
        try t.qwen4MtpTruncate(st.off0);
        var hidden = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hidden);
        try sliceRows(&hidden, st.hidden, 0, @intCast(st.n), t.s);
        const ids = mlx.mlx_array_new_data(&st.ids, &[_]c_int{@intCast(st.n)}, 1, .int32);
        defer _ = mlx.mlx_array_free(ids);
        freeOut(try t.qwen4MtpForward(hidden, ids, @intCast(st.pos), null, .none));
    }

    /// The waiting stash appended whole for a head chain (#1005's `mtpPaddedHistoryAppend`): its committed rows and the
    /// round's rejected drafts after them, which the returned placement keeps out of the chain's sight. Null: no stash.
    fn padStash(d: *Drafter, sl: *Slot) !?mtp.HeadPlace {
        var st = sl.stash orelse return null;
        sl.stash = null;
        defer st.deinit();
        const t = &d.m.trunk;
        t.qwen4MtpActivate(&sl.st);
        try t.qwen4MtpTruncate(st.off0);
        const ids = mlx.mlx_array_new_data(&st.ids, &[_]c_int{@intCast(st.rows)}, 1, .int32);
        defer _ = mlx.mlx_array_free(ids);
        freeOut(try t.qwen4MtpForward(st.hidden, ids, @intCast(st.pos), null, .none));
        return .{ .live_end = mlx.mlx_array_new_int(@intCast(st.live())), .dead_end = @intCast(st.off0 + st.rows) };
    }

    /// The whole prompt before its prefill from `at` (`begin_prompt`, `resume_prompt`): 0 is a cold prompt, which starts
    /// the slot over; past 0 the slot stands where `restore_at` left it.
    pub fn prompt(d: *Drafter, id: u32, ids: []const u32, at: u64) !void {
        const sl = if (at == 0) try d.slotAt(id, 0) else d.slots.get(id) orelse return;
        sl.prompt.clearRetainingCapacity();
        try sl.prompt.appendSlice(gpa, ids);
    }

    /// A slot's head history at a prompt boundary, copied (`aux.Snap.head`): its KV rows, its own QSA state, where its
    /// rows sit, and its QSA leftover marks (#1005's prefix cache restores the head's cache with the trunk's).
    pub const Head = struct {
        kv: kv_cache.KVCacheSnapshot,
        entry: checkpoint.SSMCheckpoint,
        meta: mtp.Qwen4MtpState.Meta,
        marks: QsaHeadMarkSet,

        pub fn deinit(h: *Head) void {
            h.kv.deinit();
            h.entry.deinit(gpa);
            h.marks.deinit();
        }

        pub fn bytes(h: *const Head) u64 {
            return kvBytes(&h.kv) + checkpoint.ssmCheckpointBytes(&h.entry);
        }

        pub fn clone(h: *const Head) !Head {
            var kv = try h.kv.trimmedCopy(h.kv.step, mlx.gpuStream());
            errdefer kv.deinit();
            var entry = try checkpoint.shareSsmCheckpoint(gpa, &h.entry);
            errdefer entry.deinit(gpa);
            return .{ .kv = kv, .entry = entry, .meta = h.meta, .marks = QsaHeadMarkSet.share(h.marks.slice()) };
        }
    };

    /// Slot `id`'s head history at `n`, copied, or null when the slot keeps none whole there.
    pub fn headSnap(d: *Drafter, id: u32, n: u64) !?Head {
        const sl = d.slots.get(id) orelse return null;
        if (!sl.synced or sl.next != n or sl.last.ctx != null) return null;
        const t = &d.m.trunk;
        const s = t.s;
        try d.applyStash(sl);
        // The live state back on the slot: it sits on the head module while the slot is the active one.
        t.qwen4MtpRelease(&sl.st);
        var shared = try sl.st.cache.snapshot();
        defer shared.deinit();
        var kv = try shared.trimmedCopy(sl.st.cache.step, s);
        errdefer kv.deinit();
        const one = (&sl.st.entry)[0..1];
        var cps = [1]checkpoint.SSMCheckpoint{try checkpoint.captureSsmCheckpoint(gpa, one, @intCast(qsa_checkpoint.qsaHistoryRows(&sl.st.entry)), s)};
        errdefer cps[0].deinit(gpa);
        try qsa_checkpoint.attachQsaHistoryToLatest(&cps, one, s);
        return .{ .kv = kv, .entry = cps[0], .meta = sl.st.meta(), .marks = QsaHeadMarkSet.share(sl.st.qsa_marks.slice()) };
    }

    /// `restore_at`: slot `id` started over at `n` with the head history `head` (its arrays shared), or with none, so its
    /// rounds draft nothing until the next cold prompt.
    pub fn headRestore(d: *Drafter, id: u32, n: u64, head: ?*const Head) !void {
        if (d.pre) |p| if (p.slot == id) d.dropPre();
        if (d.open) |*o| if (o.slot == id) {
            o.deinit();
            d.open = null;
        };
        const t = &d.m.trunk;
        var fresh = try t.qwen4MtpStateNew();
        errdefer fresh.deinit();
        if (head) |h| {
            try fresh.cache.restore(&h.kv);
            try ssm_entry.ssmRestore(&fresh.entry, &h.entry.layers[0]);
            fresh.restoreMeta(h.meta);
            fresh.qsa_marks = QsaHeadMarkSet.share(h.marks.slice());
        }
        const sl = try d.slotAt(id, n);
        d.endRequest(sl);
        dropStash(sl);
        t.qwen4MtpRelease(&sl.st);
        sl.st.deinit();
        sl.st = fresh;
        sl.prompt.clearRetainingCapacity();
        setLast(sl, .{ .ctx = null });
        sl.next = n;
        sl.synced = head != null;
    }

    /// After slot `id`'s forward of `n` rows at `pos` (their ids `ids` on the host, else `ids_dev` on the device) with
    /// the streams `streams` `[1, n, hc*H]` (borrowed): the rows whose next token is known join the history. `keep`: the
    /// forward is the engine's, whose rows past the prompt the request's policy has not counted yet.
    pub fn follow(d: *Drafter, id: u32, pos: u64, n: u32, ids: []const u32, ids_dev: Array, streams: Array, keep: bool) !void {
        const sl = try d.slotAt(id, pos);
        if (!sl.synced or n == 0) return;
        if (pos != sl.next) {
            sl.synced = false;
            return;
        }
        errdefer sl.synced = false;
        try d.applyStash(sl);
        const t = &d.m.trunk;
        const s = t.s;
        const pend = sl.last.ctx != null;
        const p = sl.prompt.items;
        // This forward's ids on the host: the engine's own, else the prompt's while the rows lie inside it.
        const host: ?[]const u32 = if (ids.len == n) ids else if (pos + n <= p.len) p[@intCast(pos)..@intCast(pos + n)] else null;
        // The token after the last row, while the prompt holds it.
        const after: ?u32 = if (pos + n < p.len) p[@intCast(pos + n)] else null;
        const body: u32 = if (after != null) n else n - 1;
        const total: usize = body + @intFromBool(pend);
        // The rows past the prompt are committed tokens the request's policy has not seen.
        const in_prompt: u32 = if (pos < p.len) @intCast(@min(p.len - pos, n)) else 0;
        if (keep and in_prompt < n) try keepPending(sl, if (ids.len == n) ids[in_prompt..] else &.{}, ids_dev, in_prompt, n, s);
        if (total > 0) {
            var row_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row_ids);
            if (host) |h| {
                const tok = try gpa.alloc(i32, total);
                defer gpa.free(tok);
                var k: usize = 0;
                if (pend) {
                    tok[0] = @intCast(h[0]);
                    k = 1;
                }
                for (h[1..]) |v| {
                    tok[k] = @intCast(v);
                    k += 1;
                }
                if (after) |a| tok[k] = @intCast(a);
                const arr = mlx.mlx_array_new_data(tok.ptr, &[_]c_int{@intCast(total)}, 1, .int32);
                defer _ = mlx.mlx_array_free(arr);
                try mlx.check(mlx.mlx_array_set(&row_ids, arr));
            } else {
                // Device ids (a token not read yet): `last`'s row pairs with id 0, row i with id i + 1.
                if (ids_dev.ctx == null) return error.MtpHistoryIds;
                var flat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(flat);
                try mlx.check(mlx.mlx_reshape(&flat, ids_dev, &[_]c_int{@intCast(n)}, 1, s));
                var as32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(as32);
                try mlx.check(mlx.mlx_astype(&as32, flat, .int32, s));
                const from: c_int = if (pend) 0 else 1;
                try mlx.check(mlx.mlx_slice(&row_ids, as32, &[_]c_int{from}, 1, &[_]c_int{@intCast(n)}, 1, &[_]c_int{1}, 1, s));
            }
            var hist = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hist);
            try rowsAfter(&hist, if (pend) sl.last else null, streams, body, n, s);
            t.qwen4MtpActivate(&sl.st);
            // The first row sits at its token's position: `last`'s at `pos`, else row 0's at `pos + 1`.
            const at: c_int = @intCast(if (pend) pos else pos + 1);
            freeOut(try t.qwen4MtpForward(hist, row_ids, at, null, .none));
        }
        if (after == null) {
            var last = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(last);
            try sliceRows(&last, streams, @intCast(n - 1), @intCast(n), s);
            setLast(sl, last);
        } else setLast(sl, .{ .ctx = null });
        sl.next = pos + n;
    }

    /// The slot when a round from t1 at `pos` can draft: its history whole and `last` the stream before `pos`.
    fn ready(d: *Drafter, id: u32, pos: u64) ?*Slot {
        const sl = d.slots.get(id) orelse return null;
        if (!sl.synced or sl.last.ctx == null or sl.next != pos) return null;
        return sl;
    }

    /// The request's policy brought up to t1: the ids the engine's own forwards committed read into its generated
    /// tokens, and its budget (`accept_cap` drafts may still be kept). Null when the slot's history cannot draft.
    fn policyAt(d: *Drafter, id: u32, pos: u64, cap: u32) !?*Generator {
        const sl = d.ready(id, pos) orelse return null;
        const own = ownOf(d.m, id) orelse return null;
        const g = d.policyOf(sl, own);
        for (sl.pending.items) |a| try appendFrom(&g.generated_ids, a);
        for (sl.pending.items) |a| _ = mlx.mlx_array_free(a);
        sl.pending.clearRetainingCapacity();
        g.completion_tokens = @intCast(g.generated_ids.items.len);
        g.max_tokens = g.completion_tokens + 1 + cap;
        return g;
    }

    /// What this round runs, as #1005's round entry decides it: null = serial (the policy's adaptive switch, a probe or
    /// its exit, a request it disabled), else the plan: the pre-draft's, when the last round opened this one.
    fn planOf(d: *Drafter, g: *Generator, pre: ?*const Pre) ?MtpRoundPlan {
        _ = d;
        // Adaptive serial re-entry: sticky, except when the request crossed into a different KV bucket.
        if (g.spec_disabled_runtime and g.spec_disable_reason == .adaptive and g.mtpAdaptiveHeadMayResume()) {
            const b = g.mtpAdaptiveBucket(g.mtpKvLen());
            const prev_bucket = g.mtp_adaptive.bucket;
            const prev_arm = g.mtp_adaptive.arm;
            const action = g.mtp_adaptive.serialTick(b);
            g.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
            if (action == .to_mtp) {
                log.info("  [mtp] adaptive: kv {d} crossed into bucket {s} -> mtp\n", .{ g.mtpKvLen(), round_cost.bucketName(g.xfm.round_cost.layout, b) });
                g.spec_disabled_runtime = false;
                g.spec_disable_reason = .none;
                g.mtp_serial_exit = .drain;
            }
        }
        if (g.spec_disabled_runtime or g.mtp_serial_left > 0 or g.mtp_serial_exit != .none) return null;
        // Cross-request EV seed: the head's last healthy acceptance surface, so the plan starts warm.
        if (g.mtp_ev_rounds == 0 and g.mtp_attempted == 0 and regime.mtpPinnedDepth() == null) {
            if (g.mtp.?.evSeed()) |seed| {
                g.mtp_ev_accept = seed.accept;
                g.mtp_ev_m_lo_prev = @min(@max(seed.m_lo, 1), mtp_mod.MAX_DEPTH);
                g.mtp_ev_rounds = depth.MTP_EV_WARMUP_ROUNDS;
            }
        }
        if (pre) |p| return p.plan;
        return headPlan(g);
    }

    /// The policy's plan for a head round (#1005's `mtpRoundPlan`, solo), held to the head's depth.
    fn headPlan(g: *Generator) MtpRoundPlan {
        var plan = g.mtpRoundPlanInner();
        plan.m_lo = @min(plan.m_lo, mtp_mod.MAX_DEPTH);
        plan.m_hi = @min(plan.m_hi, mtp_mod.MAX_DEPTH);
        return plan;
    }

    /// The plan of a round that drafts `k` lookup tokens instead (#1005's `mtpLookupChain`).
    fn lookupPlan(plan: MtpRoundPlan, k: u32) MtpRoundPlan {
        var p = plan;
        p.m_lo = k;
        p.m_hi = k;
        p.width_trial = false;
        return p;
    }

    /// A serial tick's bookkeeping (#1005's `mtpSerialTick`): the serial cell, the probe's count, the exit ramp (whose
    /// drain and capture ticks are serial ticks here: the slot's state is already the round entry's).
    fn serialDone(g: *Generator, t1: u32) !void {
        try g.generated_ids.append(gpa, t1);
        switch (g.mtp_serial_exit) {
            .drain => {
                g.mtp_serial_exit = .capture;
                return;
            },
            .capture => {
                g.mtp_serial_exit = .none;
                g.mtp_serial_clock = null;
                g.mtp_serial_warm = 0;
                g.mtp_regime_clock = null;
                return;
            },
            .none => {},
        }
        g.observeSerialTick();
        if (g.mtp_serial_left > 0) {
            g.mtp_serial_left -= 1;
            if (g.mtp_serial_left == 0) g.mtp_serial_exit = .drain;
        }
    }

    /// A drafted round's accept, as #1005's round finish counts it before the next round opens: the round toward its
    /// kind, the committed tokens, the acceptance observers and the EV surface. True: `roundEnd` times the round (a
    /// lookup round, and the head round after one, are no width sample).
    fn roundAccept(g: *Generator, lookup: bool, drafted: u32, accepted: u32, t1: u32, drafts: []const u32) !bool {
        const times_round = !lookup and !g.mtp_after_lookup;
        g.mtp_after_lookup = lookup;
        if (lookup) {
            g.mtp_lookup_rounds += 1;
            g.mtp_lookup_drafted += drafted;
        } else {
            g.mtp_attempted += 1;
            g.mtp_drafted_tokens += drafted;
        }
        // A partial accept rolls the trunk's recurrent state back from the verify's capture (`commit`).
        if (accepted < drafted) g.partial_rounds += 1;
        try g.generated_ids.append(gpa, t1);
        try g.generated_ids.appendSlice(gpa, drafts[0..accepted]);
        g.mtpRoundAcceptObserve(lookup, drafted, accepted);
        if (!lookup) g.updateMtpEvRound(drafted, accepted);
        return times_round;
    }

    /// The round's end, after the next round's pre-draft (#1005 times a round through it): its cost into the model's
    /// table and the regime, the chunk-A boundary's price, or a lookup round's own count.
    fn roundEnd(g: *Generator, times_round: bool, lookup: bool, plan: MtpRoundPlan, drafted: u32, accepted: u32, watch: io_util.Stopwatch) void {
        if (times_round) {
            g.mtpRoundEndObserve(drafted, accepted + 1, plan.m_hi > plan.m_lo, plan.m_lo, plan.width_trial, @as(f32, @floatFromInt(watch.read())) / @as(f32, std.time.ns_per_ms));
            if (g.mtpEvFixOn()) g.mtpEvFixObserve(plan, drafted, accepted);
            g.mtp_ev_round_ms = regime.mtpEmaMs(g.mtp_ev_round_ms, watch.read());
        } else {
            if (lookup) g.mtpLookupObserve(drafted, accepted + 1);
            g.mtpRoundUntimed();
        }
    }

    /// The head's chain for `plan` from t1 at `pos`, #1005's (`mtpChainBuild`), opened: chunk A to `m_lo` (on a
    /// two-chunk plan drafting off the head's full logits, each draft's confidence beside it), dispatched.
    fn chainOpen(d: *Drafter, sl: *Slot, t1: u32, pos: u64, plan: MtpRoundPlan, mask: ?Array) !Chain {
        const t = &d.m.trunk;
        var c: Chain = .{ .t = t, .rerank = t.qwen4DraftRerankReady(), .mask = mask, .h_prev = mlx.mlx_array_new(), .prev = mlx.mlx_array_new() };
        errdefer c.deinit();
        c.place = try d.padStash(sl);
        var tok: i32 = @intCast(t1);
        const t1_arr = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        try mlx.check(mlx.mlx_array_set(&c.h_prev, sl.last));
        try mlx.check(mlx.mlx_array_set(&c.prev, t1_arr));
        const two_chunk = plan.m_hi > plan.m_lo;
        try c.build(pos, 0, plan.m_lo, two_chunk);
        try c.dispatch(0, plan.m_lo, two_chunk);
        return c;
    }

    /// The opened chain closed: on a two-chunk plan the chunk's confidence read at that boundary (the round's one bounded
    /// sync) and chunk B to `m_hi` when it clears the plan's bar. The drafts `[m]` int32, dispatched, and `m`.
    fn chainClose(d: *Drafter, c: *Chain, g: *Generator, pos: u64, plan: MtpRoundPlan) !struct { drafts: Array, m: u32 } {
        const s = d.m.trunk.s;
        var m = plan.m_lo;
        if (plan.m_hi > plan.m_lo) {
            var sync_watch = io_util.Stopwatch.init(io());
            const chain_ln = try spec_sampling.readChainConfidence(c.ids[0..plan.m_lo], c.confs[0..plan.m_lo], s);
            g.mtp_ev_sync_ms = regime.mtpEmaMs(g.mtp_ev_sync_ms, sync_watch.read());
            if (chain_ln >= plan.tau_ln) {
                m = plan.m_hi;
                g.mtp_ext_rounds += 1;
                g.mtp_ext_dry_streak = 0;
                try c.build(pos, plan.m_lo, plan.m_hi, false);
                try c.dispatch(plan.m_lo, plan.m_hi, false);
            } else g.mtp_ext_dry_streak +|= 1;
            // A kept plan's boundary as the round sees it (`ev_fix.Model.read_ms`): the read, plus chunk B when it fired.
            if (g.mtpEvFixOn() and plan.fire_ideal > 0 and g.spec_cost_solo) g.xfm.mtp_ev_fix.read_ms = regime.mtpEmaMs(g.xfm.mtp_ev_fix.read_ms, sync_watch.read());
        }
        var cat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cat);
        const vec = mlx.mlx_vector_array_new_data(&c.ids, m);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        return .{ .drafts = cat, .m = m };
    }

    /// The lookup drafts a round from t1 takes (#1005's `mtpLookupChain`), into `out`: 0 = the head's chain.
    fn lookupDrafts(d: *Drafter, g: *Generator, plan: MtpRoundPlan, t1: u32, out: *[max_rows]u32) !u32 {
        _ = d;
        const lookup_cap = depth_bounds.lookupCap(mtp_lookup.MAX_DRAFT_STRONG, depth_bounds.active);
        const k = try g.mtpLookupDrafts(gpa, plan, t1);
        if (k == 0 or k < depth_bounds.floorFor(depth_bounds.active, lookup_cap)) return 0;
        const got = (try g.mtpLookupIndex(gpa)).match(t1, lookup_cap).?;
        if (!lookup_said) {
            lookup_said = true;
            log.info("[mtp] prompt-lookup drafts engaged: k={d} suffix={d}\n", .{ k, got.suffix });
        }
        @memcpy(out[0..k], got.draft[0..k]);
        return k;
    }

    /// The next round's opening at this round's tail (#1005's cross-round pre-draft, `mtpPreDraftOpen` and
    /// `mtpMaybePreDraft`): its plan, made now as the next round's entry would make it, and the lookup drafts or the
    /// head's chunk A, dispatched so the head runs while the engine publishes this round.
    fn preDraft(d: *Drafter, id: u32, pos: u64, t1: u32, mask: ?Array) !void {
        if (!predraft) return;
        const sl = d.ready(id, pos) orelse return;
        const g = if (sl.gen) |*gg| gg else return;
        g.completion_tokens = @intCast(g.generated_ids.items.len);
        if (g.completion_tokens >= g.max_tokens or d.isEos(t1)) return;
        if (g.spec_disabled_runtime) return;
        const plan = headPlan(g);
        // The plan itself can end speculation or park the request on a serial probe.
        if (g.spec_disabled_runtime or g.mtp_serial_left > 0) return;
        d.m.trunk.qwen4MtpActivate(&sl.st);
        var pre: Pre = .{ .slot = id, .pos = pos, .t1 = t1, .plan = plan, .off0 = d.liveRows(sl) };
        const k = try d.lookupDrafts(g, plan, t1, &pre.lookup_ids);
        if (k > 0) {
            try d.applyStash(sl);
            pre.lookup = true;
            pre.n_lookup = k;
            pre.plan = lookupPlan(plan, k);
        } else pre.chain = try d.chainOpen(sl, t1, pos, plan, mask);
        d.pre = pre;
    }

    /// The pre-draft for the round from t1 at `pos` on slot `id`, or null; one for any other round is dropped.
    fn takePre(d: *Drafter, id: u32, pos: u64, t1: u32) ?Pre {
        var p = d.pre orelse return null;
        d.pre = null;
        if (p.slot == id and p.pos == pos and p.t1 == t1 and d.ready(id, pos) != null) return p;
        d.dropPreValue(&p);
        return null;
    }

    /// Drops the pending pre-draft, if any: its chain's draft rows leave the head.
    pub fn dropPre(d: *Drafter) void {
        var p = d.pre orelse return;
        d.pre = null;
        d.dropPreValue(&p);
    }

    fn dropPreValue(d: *Drafter, p: *Pre) void {
        var c = p.chain orelse return;
        c.deinit();
        const sl = d.slots.get(p.slot) orelse return;
        d.m.trunk.qwen4MtpActivate(&sl.st);
        d.m.trunk.qwen4MtpTruncate(p.off0) catch {
            sl.synced = false;
        };
    }

    fn isEos(d: *const Drafter, tok: u32) bool {
        for (d.eos[0..d.n_eos]) |e| if (e == tok) return true;
        return false;
    }

    /// A round from t1 at `pos` (the request's budget in `g`): the lookup drafts when the gate takes them, else the head's
    /// chain for the policy's plan, dispatched (none on a serial round or when the slot's history cannot draft), and the
    /// verify over [t1, drafts] built, its n-gram leaf filled when the graph reads one (the one read of the draft ids).
    fn build(d: *Drafter, id: u32, pos: u64, t1: u32, mask: ?Array, g: ?*Generator) !Built {
        const t = &d.m.trunk;
        const s = t.s;
        const own = ownOf(d.m, id) orelse return error.MtpSlotGone;
        var b: Built = .{ .step = own.cache.step };
        errdefer b.deinit();
        var tok: i32 = @intCast(t1);
        const t1_arr = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        // The round the previous one opened (its plan and drafts), else the plan this round's entry makes.
        var pre = d.takePre(id, pos, t1);
        defer if (pre) |*p| if (p.chain) |*c| c.deinit();
        // This round's own chain, and a lookup round's drafts one id each: open through the verify, which lends them.
        var opened: ?Chain = null;
        defer if (opened) |*c| c.deinit();
        var lookup_ids: [max_rows]Array = undefined;
        var n_lookup_ids: usize = 0;
        defer for (lookup_ids[0..n_lookup_ids]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var lent: []const Array = &.{};
        const planned: ?MtpRoundPlan = if (g) |gg| d.planOf(gg, if (pre) |*p| p else null) else null;
        // A round the policy keeps serial leaves the pre-draft unread (#1005's `mtpDetachHead`).
        if (planned == null) if (pre) |*p| {
            d.dropPreValue(p);
            pre = null;
        };
        if (planned) |plan| {
            const gg = g.?;
            const sl = d.ready(id, pos).?;
            t.qwen4MtpActivate(&sl.st);
            b.off0 = if (pre) |p| p.off0 else d.liveRows(sl);
            b.plan = plan;
            var got: [max_rows]u32 = undefined;
            const k = if (pre) |p| (if (p.lookup) blk: {
                @memcpy(got[0..p.n_lookup], p.lookup_ids[0..p.n_lookup]);
                break :blk p.n_lookup;
            } else 0) else try d.lookupDrafts(gg, plan, t1, &got);
            if (k > 0) {
                // No head chain reads the stash this round: its committed rows join the head now.
                try d.applyStash(sl);
                var ids32: [max_rows]i32 = undefined;
                for (ids32[0..k], got[0..k]) |*o, v| o.* = @intCast(v);
                b.lookup = true;
                b.n = k;
                b.plan = lookupPlan(plan, k);
                b.drafts = mlx.mlx_array_new_data(&ids32, &[_]c_int{@intCast(k)}, 1, .int32);
                for (ids32[0..k]) |*v| {
                    lookup_ids[n_lookup_ids] = mlx.mlx_array_new_data(v, &[_]c_int{1}, 1, .int32);
                    n_lookup_ids += 1;
                }
                lent = lookup_ids[0..k];
            } else {
                const c = if (pre) |*p| &p.chain.? else blk: {
                    opened = try d.chainOpen(sl, t1, pos, plan, mask);
                    break :blk &opened.?;
                };
                const closed = try d.chainClose(c, gg, pos, plan);
                b.drafts = closed.drafts;
                b.n = closed.m;
                lent = c.ids[0..closed.m];
            }
        }
        var input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input);
        if (b.n > 0) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const vec = mlx.mlx_vector_array_new_data(&[_]Array{ t1_arr, b.drafts }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
            try mlx.check(mlx.mlx_reshape(&input, flat, &[_]c_int{ 1, @intCast(1 + b.n) }, 2, s));
        } else try mlx.check(mlx.mlx_reshape(&input, t1_arr, &[_]c_int{ 1, 1 }, 2, s));
        // A drafted round's verify lends its PLE flush the arrays its input concatenates, as #1005's round does.
        var lend_buf: [max_rows + 1]Array = undefined;
        const lend = if (b.n > 0 and d.m.cfg.trunk_format) pleIdsSource(&lend_buf, t1_arr, lent, 1 + b.n) else null;
        b.owed = try d.verifyForward(id, pos, input, 1 + b.n, &b.logits, &b.streams, lend);
        if (b.owed == .input) {
            try d.settle(id, &.{});
            b.owed = .none;
        }
        return b;
    }

    /// The trunk's forward over `input` `[1, rows]` at `pos` (`.verify`; one row is the serial `.decode` of t1) with
    /// its n-gram leaf deferred: the logits `[rows, vocab]`, the streams `[1, rows, hc*H]`, and what the leaf owes.
    /// `ple_ids`: the arrays `input` concatenates, lent to the leaf's flush for this forward (`PleLend`).
    fn verifyForward(d: *Drafter, id: u32, pos: u64, input: Array, rows: u32, logits: *Array, streams: *Array, ple_ids: ?[]const Array) !abi.Owed {
        const slot_ids = [_]u32{id};
        const positions = [_]u32{@intCast(pos)};
        const f: abi.Forward = .{
            .mode = if (rows == 1) .decode else .verify,
            .logits = .all_rows,
            .want_hidden = true,
            .rows = rows,
            .tokens = .{},
            .ids = .{ .ctx = input.ctx },
            .slots = .of(&slot_ids),
            .positions = .of(&positions),
            .defer_host = true,
        };
        var lend: PleLend = undefined;
        if (ple_ids) |ids| lend.begin(&d.m.scratch, ids);
        defer if (ple_ids != null) lend.end();
        var out: abi.ForwardOut = .{};
        const v = q4.forward(d.m, &f, &out);
        if (!v.isOk()) {
            for ([_]abi.Array{ out.logits, out.hidden }) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
            };
            log.err("[turboexl3_qwen38] the round's verify failed: {s}\n", .{v.why.slice()});
            return error.MtpVerifyFailed;
        }
        logits.* = .{ .ctx = out.logits.ctx };
        streams.* = .{ .ctx = out.hidden.ctx };
        return out.owed;
    }

    /// Settles what slot `id`'s verify left owed, from `tokens` (its ids on the host) or, empty, off the device.
    fn settle(d: *Drafter, id: u32, tokens: []const u32) !void {
        const slot = [_]u32{id};
        const toks: abi.Span(u32) = .of(tokens);
        const v = q4.forwardSettle(d.m, .of(&slot), .flush, &toks);
        if (!v.isOk()) {
            log.err("[turboexl3_qwen38] the verify's n-gram rows did not settle: {s}\n", .{v.why.slice()});
            return error.MtpVerifySettle;
        }
    }

    /// A round that failed part-way: what its verify owed dropped, its captures freed, the slot's history no longer
    /// trusted (the request fails with the round).
    fn abandon(d: *Drafter, id: u32) void {
        d.dropPre();
        const slot = [_]u32{id};
        const none: abi.Span(u32) = .{};
        _ = q4.forwardSettle(d.m, .of(&slot), .discard, &none);
        if (ownOf(d.m, id)) |own| for (own.ssm) |*e| ssm_entry.ssmFreeSpecCapture(e);
        if (d.slots.get(id)) |sl| {
            dropStash(sl);
            sl.synced = false;
        }
    }

    /// Slot `id` back to t1 and the first `accepted` of `drafts`, verified at `pos`: the trunk's KV truncated (its
    /// step kept at `kv_step`, the value before the verify) and its recurrent states taken from the verify's capture,
    /// then (with `off0`, the head's row count before the round) the head truncated to it and the committed rows
    /// re-appended, `last` the stream at the last committed position.
    fn commit(d: *Drafter, id: u32, pos: u64, t1: u32, drafts: []const u32, accepted: u32, off0: ?usize, streams: Array, kv_step: usize, padded: bool) !void {
        const t = &d.m.trunk;
        const s = t.s;
        const own = ownOf(d.m, id) orelse return error.MtpSlotGone;
        const n_drafts: u32 = @intCast(drafts.len);
        defer for (own.ssm) |*e| ssm_entry.ssmFreeSpecCapture(e);
        if (accepted < n_drafts) {
            try own.cache.truncate(pos + accepted + 1, s);
            own.cache.step = kv_step;
            for (own.ssm) |*e| try ssm_entry.ssmRollbackFromCapture(e, accepted, n_drafts + 1, s);
        }
        own.pos = pos + accepted + 1;
        const sl = d.slots.get(id) orelse return;
        const origin = off0 orelse {
            // Nothing drafted: t1's row joins the history as a decode forward's would.
            var one = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one);
            try sliceRows(&one, streams, 0, 1, s);
            return d.follow(id, pos, 1, &.{t1}, .{ .ctx = null }, one, false);
        };
        errdefer sl.synced = false;
        if (padded) {
            // The lazy pre-draft already appended the round's whole row: only `last` moves.
            var last = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(last);
            try sliceRows(&last, streams, @intCast(accepted), @intCast(accepted + 1), s);
            setLast(sl, last);
            sl.next = pos + accepted + 1;
            return;
        }
        t.qwen4MtpActivate(&sl.st);
        // The round's dead rows and drafts leave the head; its rows wait for the round that reads the head next.
        try t.qwen4MtpTruncate(origin);
        dropStash(sl);
        var hidden = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(hidden);
        try rowsAfter(&hidden, sl.last, streams, n_drafts, n_drafts + 1, s);
        var last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(last);
        try sliceRows(&last, streams, @intCast(accepted), @intCast(accepted + 1), s);
        var st: Stash = .{ .off0 = origin, .pos = pos, .rows = n_drafts + 1, .n = accepted + 1, .hidden = hidden };
        st.ids[0] = @intCast(t1);
        for (drafts, 1..) |dv, i| st.ids[i] = @intCast(dv);
        sl.stash = st;
        setLast(sl, last);
        sl.next = pos + accepted + 1;
    }

    /// An open verify the engine never finished: discarded, its drafts never kept.
    fn closeOpen(d: *Drafter) !void {
        var o = d.open orelse return;
        d.open = null;
        defer o.deinit();
        try d.commit(o.slot, o.pos, o.t1, o.drafts[0..o.n_drafts], 0, o.off0, o.streams, o.step, false);
        try d.done(o.slot, o.pos + 1, o.lookup, o.plan, o.n_drafts, 0, o.t1, &.{}, o.off0 != null, o.watch);
    }

    /// The request's policy at a round's end, when the slot's history came through the round whole (to `next_pos`).
    fn policyAfter(d: *Drafter, id: u32, next_pos: u64) ?*Generator {
        const sl = d.slots.get(id) orelse return null;
        if (sl.next != next_pos) return null;
        return if (sl.gen) |*g| g else null;
    }

    /// The round's end for the request's policy, with no next round opened: a drafted round's accept and end, else a
    /// serial tick's.
    fn done(d: *Drafter, id: u32, next_pos: u64, lookup: bool, plan: MtpRoundPlan, drafted: u32, accepted: u32, t1: u32, drafts: []const u32, drafted_round: bool, watch: io_util.Stopwatch) !void {
        const g = d.policyAfter(id, next_pos) orelse return;
        if (!drafted_round) return serialDone(g, t1);
        const timed = try roundAccept(g, lookup, drafted, accepted, t1, drafts);
        roundEnd(g, timed, lookup, plan, drafted, accepted, watch);
    }

    /// `NativeDrafter.step`: one greedy round from `f.tokens[0]` at `f.positions[0]`.
    pub fn step(d: *Drafter, f: *const abi.Forward, out: *abi.ForwardOut) !void {
        if (f.tokens.len != 1) return error.MtpRoundInput;
        const id: u32 = if (f.slots.len > 0) f.slots.slice()[0] else 0;
        const pos: u64 = if (f.positions.len > 0) f.positions.slice()[0] else 0;
        const t1 = f.tokens.slice()[0];
        const mask: ?Array = if (f.suppress_mask.ctx != null) .{ .ctx = f.suppress_mask.ctx } else null;
        const s = d.m.trunk.s;
        try d.closeOpen();
        const watch = io_util.Stopwatch.init(io());
        const g = try d.policyAt(id, pos, f.accept_cap);
        var b = d.build(id, pos, t1, mask, g) catch |e| {
            d.abandon(id);
            return e;
        };
        defer b.deinit();
        errdefer d.abandon(id);
        // The trunk's argmax per row (under the sampler's mask) and the drafts, read in one sync.
        var am_u = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(am_u);
        {
            const am = try maskedArgmax(b.logits, mask, s);
            defer _ = mlx.mlx_array_free(am);
            try mlx.check(mlx.mlx_astype(&am_u, am, .uint32, s));
        }
        // #1005's lazy pre-draft (`mtpLazyPredraftAllowedFor`): a greedy head round with budget for one more builds the
        // next round's chunk A from its verdict before the read. A plan that ends speculation leaves no pre-draft at all.
        var lazy: ?Lazy = null;
        defer if (lazy) |*z| z.deinit();
        var planned = false;
        if (lazy_predraft) if (g) |gg| if (b.off0 != null and !b.lookup and b.n > 0 and f.accept_cap > b.n and !gg.spec_disabled_runtime and gg.mtp_serial_left == 0) {
            planned = true;
            lazy = try d.lazyOpen(id, pos, t1, gg, &b, am_u, mask);
            if (lazy != null) gg.mtp_lazy_built += 1;
        };
        {
            const vec = if (b.n > 0) mlx.mlx_vector_array_new_data(&[_]Array{ am_u, b.drafts }, 2) else mlx.mlx_vector_array_new_data(&[_]Array{am_u}, 1);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }
        if (mlx.mlx_array_size(am_u) < b.n + 1) return error.MtpVerifyRows;
        const am = (mlx.mlx_array_data_uint32(am_u) orelse return error.MtpVerifyUnreadable)[0 .. b.n + 1];
        var dr: [max_rows]u32 = undefined;
        try readDrafts(&b, &dr);
        if (b.owed == .history) {
            var toks: [max_rows]u32 = undefined;
            toks[0] = t1;
            @memcpy(toks[1 .. 1 + b.n], dr[0..b.n]);
            try d.settle(id, toks[0 .. 1 + b.n]);
        }
        var raw: u32 = 0;
        while (raw < b.n and dr[raw] == am[raw]) raw += 1;
        const accepted = @min(raw, f.accept_cap);
        try d.commit(id, pos, t1, dr[0..b.n], accepted, b.off0, b.streams, b.step, lazy != null);
        const next_pos = pos + accepted + 1;
        out.round = .{ .n = accepted + 1, .accepted = accepted, .next_token = am[accepted], .rows = b.n + 1, .position = next_pos };
        out.round.tokens[0] = t1;
        @memcpy(out.round.tokens[1 .. accepted + 1], dr[0..accepted]);
        if (b.off0 == null) return d.done(id, next_pos, false, b.plan, 0, 0, t1, &.{}, false, watch);
        // Only a drafted round opens the next (#1005 pre-drafts at a round's tail, never after a serial tick), between
        // its accept and its end.
        const g_end = d.policyAfter(id, next_pos);
        const timed = if (g_end) |ge| try roundAccept(ge, b.lookup, b.n, accepted, t1, dr[0..b.n]) else false;
        if (lazy) |*z| {
            d.lazyResolve(id, pos, t1, dr[0..b.n], accepted, accepted == raw, am[accepted], z, g_end) catch |e| {
                log.warn("[turboexl3_qwen38] the lazy pre-draft did not resolve ({s}): the slot's rounds draft nothing\n", .{@errorName(e)});
                d.dropPre();
                if (d.slots.get(id)) |sl| {
                    dropStash(sl);
                    sl.synced = false;
                }
            };
        } else if (!planned) d.preDraft(id, next_pos, am[accepted], mask) catch |e| {
            log.warn("[turboexl3_qwen38] the next round's pre-draft failed ({s}): it drafts at its own entry\n", .{@errorName(e)});
            d.dropPre();
        };
        if (g_end) |ge| roundEnd(ge, timed, b.lookup, b.plan, b.n, accepted, watch);
    }

    /// #1005's lazy pre-draft (`mtpLazyPreDraft`): this round's whole row into the head from its origin, and the next
    /// round's chunk A placed past it at the verdict's live end, dispatched before the host reads this round. Its plan is
    /// drawn now, before this round's accept, as #1005 draws it (an adaptive depth change lands a round later). Null: the
    /// plan ends speculation.
    fn lazyOpen(d: *Drafter, id: u32, pos: u64, t1: u32, g: *Generator, b: *const Built, am_u: Array, mask: ?Array) !?Lazy {
        const plan = headPlan(g);
        if (g.spec_disabled_runtime or g.mtp_serial_left > 0) return null;
        const sl = d.slots.get(id) orelse return null;
        const t = &d.m.trunk;
        const s = t.s;
        const origin = b.off0.?;
        var v = try lazyVerdict(am_u, b.drafts, b.n, b.streams, s);
        defer v.deinit();
        var z: Lazy = .{ .plan = plan, .chain = null, .origin = origin, .hidden = mlx.mlx_array_new() };
        errdefer z.deinit();
        try rowsAfter(&z.hidden, sl.last, b.streams, b.n, b.n + 1, s);
        var tok: i32 = @intCast(t1);
        const t1_arr = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        {
            const vec = mlx.mlx_vector_array_new_data(&[_]Array{ t1_arr, b.drafts }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&ids, vec, 0, s));
        }
        t.qwen4MtpActivate(&sl.st);
        try t.qwen4MtpTruncate(origin);
        freeOut(try t.qwen4MtpForward(z.hidden, ids, @intCast(pos), null, .none));
        z.chain = .{ .t = t, .rerank = t.qwen4DraftRerankReady(), .mask = mask, .h_prev = mlx.mlx_array_new(), .prev = mlx.mlx_array_new() };
        const c = &z.chain.?;
        const one = mlx.mlx_array_new_int(@intCast(origin + 1));
        defer _ = mlx.mlx_array_free(one);
        var live_end = mlx.mlx_array_new();
        mlx.check(mlx.mlx_add(&live_end, one, v.a, s)) catch |e| {
            _ = mlx.mlx_array_free(live_end);
            return e;
        };
        c.place = .{ .live_end = live_end, .dead_end = @intCast(origin + 1 + b.n) };
        try mlx.check(mlx.mlx_array_set(&c.h_prev, v.h_prev));
        try mlx.check(mlx.mlx_array_set(&c.prev, v.t1));
        const two_chunk = plan.m_hi > plan.m_lo;
        try c.build(pos + b.n + 1, 0, plan.m_lo, two_chunk);
        try c.dispatch(0, plan.m_lo, two_chunk);
        if (!lazy_said) {
            lazy_said = true;
            log.info("[mtp] lazy predraft engaged\n", .{});
        }
        return z;
    }

    /// The lazy chain against the host's read (#1005's `mtpPreDraftResolve`): the next round's pre-draft when the read
    /// kept the raw greedy accept and the request goes on with a head round; else discarded, the head back at this round's
    /// origin with its row waiting as the stash, and the lookup drafts pre-drafted when the gate takes them.
    fn lazyResolve(d: *Drafter, id: u32, pos: u64, t1: u32, drafts: []const u32, accepted: u32, exact: bool, next_t1: u32, z: *Lazy, g: ?*Generator) !void {
        const sl = d.slots.get(id) orelse return;
        const next_pos = pos + accepted + 1;
        var lookup_ids: [max_rows]u32 = undefined;
        var k: u32 = 0;
        var keeps = false;
        if (g) |gg| {
            gg.completion_tokens = @intCast(gg.generated_ids.items.len);
            keeps = exact and gg.completion_tokens < gg.max_tokens and !d.isEos(next_t1) and !gg.spec_disabled_runtime and gg.mtp_serial_left == 0;
            if (keeps) k = try d.lookupDrafts(gg, z.plan, next_t1, &lookup_ids);
        }
        if (keeps and k == 0) {
            d.pre = .{ .slot = id, .pos = next_pos, .t1 = next_t1, .plan = z.plan, .off0 = z.origin + accepted + 1, .chain = z.chain };
            z.chain = null;
            g.?.mtp_lazy_kept += 1;
            return;
        }
        if (z.chain) |*c| c.deinit();
        z.chain = null;
        const t = &d.m.trunk;
        t.qwen4MtpActivate(&sl.st);
        try t.qwen4MtpTruncate(z.origin);
        var st: Stash = .{ .off0 = z.origin, .pos = pos, .rows = @intCast(drafts.len + 1), .n = accepted + 1, .hidden = z.hidden };
        z.hidden = .{ .ctx = null };
        st.ids[0] = @intCast(t1);
        for (drafts, 1..) |dv, i| st.ids[i] = @intCast(dv);
        dropStash(sl);
        sl.stash = st;
        if (k == 0) return;
        try d.applyStash(sl);
        var pre: Pre = .{ .slot = id, .pos = next_pos, .t1 = next_t1, .plan = lookupPlan(z.plan, k), .off0 = d.liveRows(sl), .lookup = true, .n_lookup = k };
        @memcpy(pre.lookup_ids[0..k], lookup_ids[0..k]);
        d.pre = pre;
    }

    /// `NativeDrafter.verify`: the drafts and their verify, left open for `finish`: `out.round.tokens[0..n]` and
    /// `out.logits` `[n, vocab]` (f32).
    pub fn verify(d: *Drafter, f: *const abi.Forward, out: *abi.ForwardOut) !void {
        if (f.tokens.len != 1) return error.MtpRoundInput;
        const id: u32 = if (f.slots.len > 0) f.slots.slice()[0] else 0;
        const pos: u64 = if (f.positions.len > 0) f.positions.slice()[0] else 0;
        const t1 = f.tokens.slice()[0];
        try d.closeOpen();
        const watch = io_util.Stopwatch.init(io());
        const g = try d.policyAt(id, pos, f.accept_cap);
        var b = d.build(id, pos, t1, null, g) catch |e| {
            d.abandon(id);
            return e;
        };
        defer b.deinit();
        errdefer d.abandon(id);
        var o: Open = .{ .slot = id, .pos = pos, .t1 = t1, .n_drafts = b.n, .lookup = b.lookup, .off0 = b.off0, .plan = b.plan, .step = b.step, .watch = watch };
        if (b.n > 0) try mlx.check(mlx.mlx_array_eval(b.drafts));
        try readDrafts(&b, &o.drafts);
        if (b.owed == .history) {
            var toks: [max_rows]u32 = undefined;
            toks[0] = t1;
            @memcpy(toks[1 .. 1 + b.n], o.drafts[0..b.n]);
            try d.settle(id, toks[0 .. 1 + b.n]);
        }
        var f32_logits = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(f32_logits);
        try mlx.check(mlx.mlx_astype(&f32_logits, b.logits, .float32, d.m.trunk.s));
        o.streams = b.streams;
        b.streams = .{ .ctx = null };
        d.open = o;
        out.logits = .{ .ctx = f32_logits.ctx };
        out.round = .{ .n = b.n + 1, .rows = b.n + 1, .position = pos };
        out.round.tokens[0] = t1;
        @memcpy(out.round.tokens[1 .. b.n + 1], o.drafts[0..b.n]);
    }

    /// `NativeDrafter.finish`: keeps t1 and the open verify's first `accepted` drafts.
    pub fn finish(d: *Drafter, accepted: u32, out: *abi.ForwardOut) !void {
        var o = d.open orelse return error.MtpNoOpenVerify;
        d.open = null;
        defer o.deinit();
        const a = @min(accepted, o.n_drafts);
        try d.commit(o.slot, o.pos, o.t1, o.drafts[0..o.n_drafts], a, o.off0, o.streams, o.step, false);
        try d.done(o.slot, o.pos + a + 1, o.lookup, o.plan, o.n_drafts, a, o.t1, o.drafts[0..o.n_drafts], o.off0 != null, o.watch);
        out.round = .{ .n = a + 1, .accepted = a, .rows = o.n_drafts + 1, .position = o.pos + a + 1 };
    }
};

/// The head's draft steps of one round (#1005's `MtpPreDraft` as `mtpChainBuild` fills it): each step's id `[1]`
/// int32, its confidence on a two-chunk plan's chunk A, the stream the next step reads and the fused pick's front.
const Chain = struct {
    t: *Trunk,
    rerank: bool,
    mask: ?Array,
    ids: [max_rows]Array = undefined,
    confs: [max_rows]Array = undefined,
    n_ids: usize = 0,
    n_confs: usize = 0,
    h_prev: Array,
    prev: Array,
    front: Array = .{ .ctx = null },
    /// Where the chain's rows sit when the head holds a padded round's dead rows (`padStash`): past them, out of sight.
    place: ?mtp.HeadPlace = null,

    fn deinit(c: *Chain) void {
        for (c.ids[0..c.n_ids]) |a| _ = mlx.mlx_array_free(a);
        for (c.confs[0..c.n_confs]) |a| _ = mlx.mlx_array_free(a);
        _ = mlx.mlx_array_free(c.h_prev);
        _ = mlx.mlx_array_free(c.prev);
        if (c.front.ctx != null) _ = mlx.mlx_array_free(c.front);
        if (c.place) |p| _ = mlx.mlx_array_free(p.live_end);
    }

    /// Steps `from..to` at positions `pos + i` (placed past the dead rows when the head holds some); `conf`: these steps
    /// draft off the head's full logits, each with its confidence (a two-chunk plan's chunk A, `mtpDraftStepPath`'s
    /// `.full_logits`).
    fn build(c: *Chain, pos: u64, from: u32, to: u32, conf: bool) !void {
        const s = c.t.s;
        var i: u32 = from;
        while (i < to) : (i += 1) {
            const rerank = c.rerank and !conf;
            const project: mtp.Qwen4MtpProject = if (rerank) .mixed_last_row else .last_row;
            const front: ?Array = if (c.front.ctx != null) c.front else null;
            const o = if (c.place) |*p|
                try mtp.qwen4MtpForwardPlacedFront(c.t, c.h_prev, c.prev, p, project, front)
            else
                try mtp.qwen4MtpForwardFront(c.t, c.h_prev, c.prev, @intCast(pos + i), null, project, front);
            defer freeOut(o);
            const pk = try pick(c.t, o, rerank, c.mask);
            defer _ = mlx.mlx_array_free(pk.id);
            if (c.front.ctx != null) _ = mlx.mlx_array_free(c.front);
            c.front = pk.en;
            var as32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(as32);
            try mlx.check(mlx.mlx_astype(&as32, pk.id, .int32, s));
            c.ids[c.n_ids] = mlx.mlx_array_new();
            c.n_ids += 1;
            try mlx.check(mlx.mlx_reshape(&c.ids[c.n_ids - 1], as32, &[_]c_int{1}, 1, s));
            if (conf) {
                // Built from the step's own logits before they are freed (a lazy graph holds its inputs).
                c.confs[c.n_confs] = try spec_sampling.draftConfidenceGraph(o.logits, c.ids[c.n_ids - 1], s);
                c.n_confs += 1;
            }
            try mlx.check(mlx.mlx_array_set(&c.prev, c.ids[c.n_ids - 1]));
            try mlx.check(mlx.mlx_array_set(&c.h_prev, o.stream));
        }
    }

    /// Fires steps `from..to` at the GPU without a sync; chunk A of a two-chunk plan also carries its confidences and
    /// the chain's stream, so the boundary read finds them realized and chunk B resumes from a realized buffer.
    fn dispatch(c: *Chain, from: u32, to: u32, conf: bool) !void {
        var arrs: [2 * max_rows + 1]Array = undefined;
        var n: usize = 0;
        for (c.ids[from..to]) |a| {
            arrs[n] = a;
            n += 1;
        }
        if (conf) {
            for (c.confs[from..to]) |a| {
                arrs[n] = a;
                n += 1;
            }
            arrs[n] = c.h_prev;
            n += 1;
        }
        const vec = mlx.mlx_vector_array_new_data(&arrs, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }
};

/// #1005's verify lend (`ForwardCtx.ple_ids_src`): the arrays a solo verify's input concatenates, which its deferred
/// PLE flush reads each id off after that id's own producer, instead of an op over the input queued behind the whole
/// draft chain, and fills the leaf byte for byte as the gather does. The copy reads it off its engine context
/// (`q4.engineCtx`), which carries nothing else here (no M-RoPE table): set for one forward, and the context before it
/// put back after, so no later forward reads arrays the round frees.
const PleLend = struct {
    ctx: ForwardCtx,
    off: usize,
    prev: ?*const ForwardCtx,

    fn begin(l: *PleLend, kv: *KVCache, ids: []const Array) void {
        l.off = 0;
        l.ctx = .{ .cache = kv, .moe_seq_offset = &l.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null, .ple_ids_src = ids };
        l.prev = q4.engineCtx().*;
        q4.engineCtx().* = &l.ctx;
    }

    fn end(l: *const PleLend) void {
        q4.engineCtx().* = l.prev;
    }
};

/// A request's `[spec-stats]` lines, as #1005's `logSpecStats` says them for an MTP request (`stats_log.zig`): its head
/// rounds, their acceptance, depth and costs against the model's round-cost table at its context, its lookup rounds,
/// its lazy pre-drafts, its chunk-A pricing (where the head prices its plans, `ev_fix_on`, or it priced one) and its
/// lookup table. Empty for a request that ran no round. The server's own `mode=dspark` line counts every round, lookup
/// rounds included; these count them apart, as #1005's do.
fn specStatsLines(buf: []u8, g: *const Generator, ev_fix_on: bool) []const u8 {
    if (g.mtp_attempted == 0 and g.mtp_lookup_rounds == 0) return buf[0..0];
    var n: usize = 0;
    var table_buf: [256]u8 = undefined;
    var lookup_buf: [256]u8 = undefined;
    const table = &g.xfm.round_cost;
    const kv = g.mtpKvLen();
    const bucket = table.bucketOf(kv);
    const avg: f64 = @as(f64, @floatFromInt(g.mtp_accepted_tokens)) / @as(f64, @floatFromInt(@max(g.mtp_attempted, 1)));
    // Depth varies per round under the EV controller: the drafted count is the denominator, not attempts x cap.
    const proposed: u64 = if (g.mtp_drafted_tokens > 0) g.mtp_drafted_tokens else g.mtp_attempted * @as(u64, g.mtp_depth);
    const pct: f64 = if (proposed > 0) 100.0 * @as(f64, @floatFromInt(g.mtp_accepted_tokens)) / @as(f64, @floatFromInt(proposed)) else 0.0;
    n += (std.fmt.bufPrint(buf[n..], "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} partial_rounds={d} runtime_disabled={s} reason={s} adaptive={s} serial_cell={d:.2} sync_ms={d:.2} round_ms={d:.2} two_ms_tok={d:.2} one_ms_tok={d:.2} verdict_round={d} trials={d} width_trials={d} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} serial_drops=t{d}/c{d}/b{d} depth_policy={s} lookup={d}/{d}/{d}\n", .{
        g.mtp_attempted,
        g.mtp_accepted_tokens,
        avg,
        pct,
        g.mtp_depth,
        g.mtp_drafted_tokens,
        g.mtp_ext_rounds,
        g.partial_rounds,
        if (g.spec_disabled_runtime) "true" else "false",
        @tagName(g.spec_disable_reason),
        @tagName(g.mtp_adaptive.arm),
        table.serialMsPerTok(bucket) orelse 0.0,
        g.mtp_ev_sync_ms,
        g.mtp_ev_round_ms,
        if (g.mtp_regime.two_tok > 0) g.mtp_regime.two_ms / g.mtp_regime.two_tok else 0.0,
        if (g.mtp_regime.one_tok > 0) g.mtp_regime.one_ms / g.mtp_regime.one_tok else 0.0,
        g.mtp_regime.verdict_round,
        g.mtp_regime.trials,
        g.mtp_width_trial.trials,
        round_cost.bucketName(table.layout, bucket),
        table.formatBucket(bucket, &table_buf),
        table.dropped_transition,
        table.dropped_contended,
        table.dropped_bad,
        table.dropped_implausible,
        table.serial_dropped_transition,
        table.serial_dropped_contended,
        table.serial_dropped_bad,
        @tagName(ev_plan.mtpDepthPolicyFor(ev_plan.mtpDepthPolicy(), kv)),
        g.mtp_lookup_rounds,
        g.mtp_lookup_drafted,
        g.mtp_lookup_accepted,
    }) catch return buf[0..n]).len;
    n += (std.fmt.bufPrint(buf[n..], "  [spec-stats] lazy={d}/{d}\n", .{ g.mtp_lazy_built, g.mtp_lazy_kept }) catch return buf[0..n]).len;
    if (g.mtp_ev_fix.priced > 0 or ev_fix_on) {
        const fx = g.xfm.mtp_ev_fix;
        const r = ev_fix.ratios(fx);
        n += (std.fmt.bufPrint(buf[n..], "  [spec-stats] ev_fix priced={d} collapsed={d} boundary_ms={d:.2} read_ms={d:.2} excess_ms={d:.2}/{d} gain_ratio={d:.2} fire_ratio={d:.2}\n", .{
            g.mtp_ev_fix.priced, g.mtp_ev_fix.collapsed, ev_fix.boundaryMs(fx) orelse 0, fx.read_ms, fx.excess_ms, fx.excess_n, r.gain, r.fire,
        }) catch return buf[0..n]).len;
    }
    if (g.mtp_lookup_rounds > 0) n += (std.fmt.bufPrint(buf[n..], "  [spec-stats] lookup_table={s}:{s}\n", .{
        round_cost.bucketName(table.layout, bucket),
        table.formatLookupBucket(bucket, &lookup_buf),
    }) catch return buf[0..n]).len;
    return buf[0..n];
}

/// The arrays a solo verify's input concatenates, `[t1, drafts]`, into `buf` (#1005's `pleIdsSource`); null where the
/// input is not that concatenation.
fn pleIdsSource(buf: *[max_rows + 1]Array, t1: Array, drafts: []const Array, rows: u32) ?[]const Array {
    if (drafts.len > max_rows or rows != 1 + drafts.len) return null;
    buf[0] = t1;
    @memcpy(buf[1 .. 1 + drafts.len], drafts);
    return buf[0 .. 1 + drafts.len];
}

fn setLast(sl: *Slot, stream: Array) void {
    if (sl.last.ctx != null) _ = mlx.mlx_array_free(sl.last);
    sl.last = stream;
}

fn ownOf(m: *q4.Model, id: u32) ?*OwnSlot {
    for (m.own.items) |*o| if (o.id == id) return o;
    return null;
}

/// The step's draft: #1005's rerank pick (its coarse shortlist re-scored) or the head's own greedy argmax, through the
/// fused pick on an EXL3 pack (the id and the next step's normalized embedding in one pass, the unfused id bit for
/// bit). `en` is null where the pick ran unfused.
fn pick(t: *Trunk, o: mtp.Qwen4MtpOut, rerank: bool, mask: ?Array) !draft_fuse.Pick {
    if (rerank) {
        if (try mtp.qwen4DraftPick(t, o.mixed, mask)) |pk| return pk;
        return .{ .id = try t.qwen4DraftSelect(o.mixed, mask), .en = .{ .ctx = null } };
    }
    if (try mtp.qwen4GreedyPick(t, o.logits, mask)) |pk| return pk;
    return .{ .id = try maskedArgmax(o.logits, mask, t.s), .en = .{ .ctx = null } };
}

/// The round's drafts (already evaluated) into `out`.
fn readDrafts(b: *const Built, out: *[max_rows]u32) !void {
    if (b.n == 0) return;
    const host = mlx.mlx_array_data_int32(b.drafts) orelse return error.MtpDraftsUnreadable;
    for (out[0..b.n], host[0..b.n]) |*o, v| o.* = @intCast(v);
}

fn freeOut(o: mtp.Qwen4MtpOut) void {
    if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
    if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
    if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
}

/// Rows `from..to` of `x` `[1, n, w]`.
fn sliceRows(out: *Array, x: Array, from: c_int, to: c_int, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(x);
    try mlx.check(mlx.mlx_slice(out, x, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ sh[0], to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
}

/// `last` (when set) and the first `body` rows of `streams` `[1, n, w]`, joined on the row axis.
fn rowsAfter(out: *Array, last: ?Array, streams: Array, body: u32, n: u32, s: mlx.mlx_stream) !void {
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    if (body == n) try mlx.check(mlx.mlx_array_set(&rows, streams)) else if (body > 0) try sliceRows(&rows, streams, 0, @intCast(body), s);
    const l = last orelse return mlx.check(mlx.mlx_array_set(out, rows));
    if (body == 0) return mlx.check(mlx.mlx_array_set(out, l));
    const vec = mlx.mlx_vector_array_new_data(&[_]Array{ l, rows }, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(out, vec, 1, s));
}

/// argmax over the last axis, the suppressed ids never drawn.
fn maskedArgmax(logits: Array, mask: ?Array, s: mlx.mlx_stream) !Array {
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (mask) |mk| {
        const neg = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg);
        var negd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(negd);
        try mlx.check(mlx.mlx_astype(&negd, neg, mlx.mlx_array_dtype(logits), s));
        try mlx.check(mlx.mlx_where(&masked, mk, negd, logits, s));
    } else try mlx.check(mlx.mlx_array_set(&masked, logits));
    var am = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, masked, -1, false, s));
    return am;
}

/// Keeps rows `from..n` of a forward's committed ids for the request's policy: `host` when the engine handed them on
/// the host, else a slice of `dev` (`[n]` or `[1, n]`), read at the next round.
fn keepPending(sl: *Slot, host: []const u32, dev: Array, from: u32, n: u32, s: mlx.mlx_stream) !void {
    var ref = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ref);
    if (host.len > 0) {
        const tok = try gpa.alloc(i32, host.len);
        defer gpa.free(tok);
        for (tok, host) |*o, v| o.* = @intCast(v);
        const arr = mlx.mlx_array_new_data(tok.ptr, &[_]c_int{@intCast(host.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_array_set(&ref, arr));
    } else {
        if (dev.ctx == null) return error.MtpHistoryIds;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, dev, &[_]c_int{@intCast(n)}, 1, s));
        try mlx.check(mlx.mlx_slice(&ref, flat, &[_]c_int{@intCast(from)}, 1, &[_]c_int{@intCast(n)}, 1, &[_]c_int{1}, 1, s));
    }
    try sl.pending.append(gpa, ref);
}

/// `ids` (any integer dtype, any shape) read off the device onto `list`.
fn appendFrom(list: *std.ArrayList(u32), ids: Array) !void {
    var as32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as32);
    try mlx.check(mlx.mlx_astype(&as32, ids, .int32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(as32));
    const n = mlx.mlx_array_size(as32);
    const host = mlx.mlx_array_data_int32(as32) orelse return error.MtpCommittedIdsUnreadable;
    try list.ensureUnusedCapacity(gpa, n);
    for (host[0..n]) |v| list.appendAssumeCapacity(@intCast(v));
}

var lookup_said = false;
var lazy_said = false;

/// A round's verdict as lazy arrays (#1005's `mtpLazyVerdict`): `a` (0-d int32) the drafts the argmax agrees with, the
/// next `t1` (`[1]` int32) and the stream before it (`[1, 1, hc*H]`), all before any host read.
const LazyVerdict = struct {
    a: Array,
    t1: Array,
    h_prev: Array,

    fn deinit(v: *LazyVerdict) void {
        for ([_]Array{ v.a, v.t1, v.h_prev }) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
    }
};

/// `am_u` the verify's argmax `[1 + n]` (uint32), `drafts` `[n]` (int32), `streams` `[1, 1 + n, hc*H]`.
fn lazyVerdict(am_u: Array, drafts: Array, n: u32, streams: Array, s: mlx.mlx_stream) !LazyVerdict {
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, am_u, .int32, s));
    // a = the first mismatch, with a mismatch appended at n: argmax takes the first max.
    var a_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_u);
    {
        var head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(head);
        try mlx.check(mlx.mlx_slice(&head, ids, &[_]c_int{0}, 1, &[_]c_int{@intCast(n)}, 1, &[_]c_int{1}, 1, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_equal(&eq, head, drafts, s));
        var miss = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss);
        try mlx.check(mlx.mlx_logical_not(&miss, eq, s));
        const tail = mlx.mlx_array_new_data(&[_]bool{true}, &[_]c_int{1}, 1, .bool_);
        defer _ = mlx.mlx_array_free(tail);
        var miss_ext = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss_ext);
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{ miss, tail }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&miss_ext, vec, 0, s));
        var miss_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(miss_i);
        try mlx.check(mlx.mlx_astype(&miss_i, miss_ext, .int32, s));
        try mlx.check(mlx.mlx_argmax_axis(&a_u, miss_i, 0, false, s));
    }
    var v: LazyVerdict = .{ .a = mlx.mlx_array_new(), .t1 = mlx.mlx_array_new(), .h_prev = mlx.mlx_array_new() };
    errdefer v.deinit();
    try mlx.check(mlx.mlx_astype(&v.a, a_u, .int32, s));
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    try mlx.check(mlx.mlx_reshape(&a1, v.a, &[_]c_int{1}, 1, s));
    try mlx.check(mlx.mlx_take(&v.t1, ids, a1, s));
    try mlx.check(mlx.mlx_take_axis(&v.h_prev, streams, a1, 1, s));
    return v;
}

// ── tests ──

const testing = std.testing;

test "drafter: a verify lends its [t1, drafts] to the copy's PLE flush, for its own forward alone" {
    const t1: Array = .{ .ctx = @ptrFromInt(0x10) };
    const drafts = [_]Array{ .{ .ctx = @ptrFromInt(0x20) }, .{ .ctx = @ptrFromInt(0x30) } };
    var buf: [max_rows + 1]Array = undefined;
    // A row a group padded is no concatenation of them: nothing is lent.
    try testing.expect(pleIdsSource(&buf, t1, &drafts, 4) == null);
    const ids = pleIdsSource(&buf, t1, &drafts, 3).?;
    try testing.expectEqual(@as(usize, 3), ids.len);
    for ([_]Array{ t1, drafts[0], drafts[1] }, ids) |want, got| try testing.expectEqual(want.ctx, got.ctx);
    var kv: KVCache = undefined;
    var lend: PleLend = undefined;
    lend.begin(&kv, ids);
    const ec = q4.engineCtx().*.?;
    try testing.expectEqual(@as(usize, 3), ec.ple_ids_src.?.len);
    // The copy's forward reads nothing else of it: no M-RoPE table.
    try testing.expect(ec.mrope_pos == null);
    lend.end();
    try testing.expect(q4.engineCtx().* == null);
}

test "drafter: a request's [spec-stats] lines are #1005's: its rounds, lookups, lazy pre-drafts and chunk-A pricing" {
    var model: generator.Model = .{ .trunk = undefined };
    var g: Generator = .{ .xfm = &model, .timer = undefined };
    var buf: [2048]u8 = undefined;
    // A request that ran no round says nothing.
    try testing.expectEqualStrings("", specStatsLines(&buf, &g, true));
    g.mtp_attempted = 3;
    g.mtp_accepted_tokens = 5;
    g.mtp_drafted_tokens = 10;
    g.partial_rounds = 2;
    g.mtp_lookup_rounds = 2;
    g.mtp_lookup_drafted = 9;
    g.mtp_lookup_accepted = 4;
    g.mtp_lazy_built = 4;
    g.mtp_lazy_kept = 3;
    model.mtp_ev_fix.read_ms = 2.1;
    var it = std.mem.splitScalar(u8, specStatsLines(&buf, &g, false), '\n');
    const first = it.next().?;
    try testing.expect(std.mem.startsWith(u8, first, "  [spec-stats] mode=mtp attempts=3 accepts=5 avg_per_round=1.67 per_draft_pct=50.0% depth=3 drafted=10 ext_rounds=0 partial_rounds=2 runtime_disabled=false reason=none adaptive="));
    try testing.expect(std.mem.endsWith(u8, first, " lookup=2/9/4"));
    try testing.expectEqualStrings("  [spec-stats] lazy=4/3", it.next().?);
    // No pricing head and nothing priced: no ev_fix line; lookup rounds: their table.
    try testing.expect(std.mem.startsWith(u8, it.next().?, "  [spec-stats] lookup_table="));
    try testing.expectEqualStrings("", it.next().?);
    try testing.expect(it.next() == null);
    // A head that prices its plans says its chunk-A pricing.
    try testing.expect(std.mem.indexOf(u8, specStatsLines(&buf, &g, true), "  [spec-stats] lazy=4/3\n  [spec-stats] ev_fix priced=0 collapsed=0 boundary_ms=2.10 read_ms=2.10 excess_ms=0.00/0 gain_ratio=1.00 fire_ratio=1.00\n  [spec-stats] lookup_table=") != null);
}
