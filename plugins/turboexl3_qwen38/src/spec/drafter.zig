//! This plugin's own: the MTP head (src/direct/mtp_head.zig) behind the kit's `NativeDrafter`, on the state the model
//! keeps for each slot, with #1005's round (its `engines/mlx/spec/mtp/`): the same history rows, positions, chain,
//! verify and rewind, and #1005's round policy itself (planner, depth, ev_plan, ev_fix, regime, adaptive, observe, the
//! prompt lookup), copied unchanged and run on a request's state as #1005's engine keeps it
//! (`engines/mlx/generate/generator.zig`, this plugin's edge).
//!
//! The head's history follows every committed token. Row r pairs the trunk's stream at position r with token r + 1
//! and sits at position r + 1. A forward's rows whose next token is known join the history in one head forward: a
//! prompt chunk's rows with the prompt's tokens after them (`resume_prompt` hands the prompt in), as #1005's prefill
//! appends each chunk; the last row's stream waits as `last` for the token after it. As in #1005, a prompt chunk before
//! the window `--mtp-history-window` keeps adds none (the history starts late, at the head's next position), nor does
//! anything a request past `--max-mtp-ctx` forwards. A request the engine runs no round for never drafts: the engine
//! drafts from a request's first tick, so a one-row forward past the prompt before any round ends the history and frees
//! the head's state. Its prompt's history is built all the same: nothing says which requests draft before the prefill.
//!
//! A round from t1 at `pos` runs the plan #1005's policy makes for it: the context's own continuation when the lookup
//! gate prefers it, else the head's chain to the plan's base depth (the first step from `last` and t1, each next from the
//! head's own stream), and on a two-chunk plan the chain's confidence read at that boundary and the extension when it
//! clears the plan's bar. Then one trunk forward over [t1, drafts] in `.verify` (each recurrent row's state captured,
//! the n-gram leaf deferred so the graph builds while the drafts run), the drafts the trunk's argmax agrees with kept
//! (at most `accept_cap`), the trunk rewound to them (the KV truncated, the recurrent states from the capture), and the
//! head truncated to the round's origin with the committed rows appended. A round the policy keeps serial (its adaptive
//! switch, a probe, a disabled request, a context past `--max-mtp-ctx`) drafts nothing.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const q4 = @import("../direct/serve.zig");
const mtp = @import("../mlx_serve/plugins/qwen4_exp/mtp.zig");
const mtp_head = @import("../direct/mtp_head.zig");
const cost_profile = @import("../mlx_serve/plugins/qwen4_exp/cost_profile.zig");
const mtp_mod = @import("../mlx_serve/sdk/spec/mtp/mtp.zig");
const mtp_lookup = @import("../mlx_serve/engines/mlx/spec/mtp_lookup.zig");
const round_cost = @import("../mlx_serve/engines/mlx/spec/round_cost.zig");
const ev_plan = @import("../mlx_serve/engines/mlx/spec/mtp/ev_plan.zig");
const ev_costs = @import("../mlx_serve/engines/mlx/spec/mtp/ev_costs.zig");
const regime = @import("../mlx_serve/engines/mlx/spec/mtp/regime.zig");
const depth = @import("../mlx_serve/engines/mlx/spec/mtp/depth.zig");
const depth_bounds = @import("../mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const spec_sampling = @import("../mlx_serve/engines/mlx/sampling/spec_sampling.zig");
const generator = @import("../mlx_serve/engines/mlx/generate/generator.zig");
const MtpHeadRef = @import("../mlx_serve/engines/mlx/spec/mtp_head_ref.zig").MtpHeadRef;
const io_util = @import("../mlx_serve/sdk/util/io_util.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const state = @import("../direct/cache.zig");
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
    /// A forward that ends at or before this adds no rows: a prompt chunk before the window `--mtp-history-window`
    /// keeps, or any forward of a prompt the server admits no MTP for (`--max-mtp-ctx`).
    from: u64 = 0,
    /// Row r sits at the head's position r + 1 - `base`: the rows the forwards before `from` skipped (a restore's own
    /// from its head), so a history that starts late starts at the head's next position, as #1005's cache-relative
    /// positions do.
    base: u64 = 0,
    /// The history holds every committed row so far: rounds draft.
    synced: bool,
    /// The engine ran a round (`step`, `verify`) for this request: it drafts.
    drafts: bool = false,
    /// The request's round policy, from its first round (#1005's per-request state).
    gen: ?Generator = null,
    /// Committed ids the engine's own forwards left on the device, read into `gen.generated_ids` at the next round.
    pending: std.ArrayList(Array) = .empty,
};

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

pub const Drafter = struct {
    m: *q4.Model,
    /// The model's share of the round policy (the engine keeps it on its Transformer): the measured round-cost
    /// table, the chunk-A boundary's price, the exact-depth costs.
    pm: generator.Model,
    /// `--max-mtp-ctx` (0 = no ceiling) and `--mtp-history-window` (0 = the whole history), as src/host_flags.zig reads
    /// them: #1005's `generate.max_mtp_ctx` and `mtp_history_window_override`.
    max_mtp_ctx: u32,
    mtp_history_window: usize,
    slots: std.AutoHashMapUnmanaged(u32, *Slot) = .empty,
    open: ?Open = null,

    pub fn init(m: *q4.Model, max_mtp_ctx: u32, mtp_history_window: usize) Drafter {
        return .{ .m = m, .pm = .{ .trunk = &m.trunk, .round_cost = .{ .layout = round_cost.layoutFor(&m.cfg) } }, .max_mtp_ctx = max_mtp_ctx, .mtp_history_window = mtp_history_window };
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
        var st = mtp_head.qwen4MtpStateNew(t) catch |e| {
            _ = mlx.mlx_array_free(h_prev);
            return e;
        };
        defer st.deinit();
        mtp_head.qwen4MtpActivate(t, &st);
        defer mtp_head.qwen4MtpRelease(t, &st);
        var tok: i32 = 2;
        var c: Chain = .{ .t = t, .rerank = mtp_head.qwen4DraftRerankReady(t), .mask = null, .h_prev = h_prev, .prev = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32) };
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
    }

    pub fn deinit(d: *Drafter) void {
        if (d.open) |*o| o.deinit();
        d.open = null;
        var it = d.slots.valueIterator();
        while (it.next()) |sl| d.drop(sl.*);
        d.slots.deinit(gpa);
    }

    fn drop(d: *Drafter, sl: *Slot) void {
        endRequest(sl);
        mtp_head.qwen4MtpRelease(&d.m.trunk, &sl.st);
        sl.st.deinit();
        sl.pending.deinit(gpa);
        sl.prompt.deinit(gpa);
        if (sl.last.ctx != null) _ = mlx.mlx_array_free(sl.last);
        gpa.destroy(sl);
    }

    pub fn release(d: *Drafter, id: u32) void {
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
        sl.* = .{ .st = try mtp_head.qwen4MtpStateNew(&d.m.trunk), .next = pos, .synced = pos == 0 };
        errdefer sl.st.deinit();
        try d.slots.put(gpa, id, sl);
        return sl;
    }

    fn restart(d: *Drafter, sl: *Slot) !void {
        const t = &d.m.trunk;
        const fresh = try mtp_head.qwen4MtpStateNew(t);
        endRequest(sl);
        mtp_head.qwen4MtpRelease(t, &sl.st);
        sl.st.deinit();
        sl.st = fresh;
        sl.prompt.clearRetainingCapacity();
        setLast(sl, .{ .ctx = null });
        sl.next = 0;
        sl.from = 0;
        sl.base = 0;
        sl.synced = true;
        sl.drafts = false;
    }

    /// A request that decodes plain: its history ends, the head's state is freed, and no round of it drafts.
    fn quiet(d: *Drafter, sl: *Slot) !void {
        const t = &d.m.trunk;
        const fresh = try mtp_head.qwen4MtpStateNew(t);
        endRequest(sl);
        mtp_head.qwen4MtpRelease(t, &sl.st);
        sl.st.deinit();
        sl.st = fresh;
        setLast(sl, .{ .ctx = null });
        sl.synced = false;
    }

    /// The request's policy, made at its first round as #1005's generate init makes it: the depth cap from the head's
    /// cost profile and `--mtp-max-depth` (src/host_flags.zig reads the server's), the free cap, the profile's costs,
    /// and the depth controller's start at its floor, `--mtp-min-depth` (1 without it).
    fn policyOf(d: *Drafter, sl: *Slot, own: *const OwnSlot) *Generator {
        if (sl.gen) |*g| return g;
        const head: MtpHeadRef = .{ .qwen4 = &d.m.trunk };
        const profile = cost_profile.qwen4G17CostProfileForKv(&d.m.trunk, own.cache.config);
        const cap = @min(ev_plan.resolveMtpDepthCapForProfile(depth_bounds.active.max, profile), head.maxDepth());
        sl.gen = .{
            .xfm = &d.pm,
            .mtp = head,
            .timer = io_util.Stopwatch.init(io()),
            .prompt_ids_owned = sl.prompt.items,
            .model_has_mtp = true,
            .mtp_depth = cap,
            .mtp_depth_free = if (d.pm.mtp_depth_free != 0) d.pm.mtp_depth_free else ev_plan.mtpDepthCapFree(depth_bounds.active.max),
            .mtp_ev_costs = ev_costs.mtpEvCosts(profile),
            .mtp_depth_current = depth_bounds.floorFor(depth_bounds.active, cap),
        };
        return &sl.gen.?;
    }

    /// The head's row count (its state on the module: `qwen4MtpActivate` swapped it there).
    fn headRows(d: *Drafter) usize {
        return d.m.trunk.qwen4_mtp.?.seq_offset;
    }

    /// The whole prompt before its prefill from `at` (`begin_prompt`, `resume_prompt`): 0 is a cold prompt, which starts
    /// the slot over; past 0 the slot stands where `restore_at` left it. A prompt past `--max-mtp-ctx` is one the
    /// server admits no MTP for: none of its forwards adds history, as #1005 keeps none for it.
    pub fn prompt(d: *Drafter, id: u32, ids: []const u32, at: u64) !void {
        const sl = if (at == 0) try d.slotAt(id, 0) else d.slots.get(id) orelse return;
        sl.prompt.clearRetainingCapacity();
        try sl.prompt.appendSlice(gpa, ids);
        sl.from = if (mtp_mod.mtpCtxWithinLimit(d.max_mtp_ctx, ids.len)) historyFrom(ids.len, at, d.mtp_history_window) else std.math.maxInt(u64);
    }

    /// Whether slot `id`'s forward of `n` rows at `pos` adds rows to its history, so `follow` reads its streams: one that
    /// starts the slot over, else one on a whole history (`synced`, at `next`) that ends past the history's start
    /// (`from`), unless it is a request decoding plain (`decodesPlain`).
    pub fn takes(d: *const Drafter, id: u32, pos: u64, n: u32) bool {
        const sl = d.slots.get(id) orelse return pos == 0;
        if (pos == 0 and sl.next != 0) return true;
        return sl.synced and pos == sl.next and pos + n > sl.from and !decodesPlain(sl, pos, n);
    }

    /// A slot's head history at a prompt boundary, copied (`aux.Snap.head`): its KV rows, its own QSA state, where its
    /// rows sit, and its QSA leftover marks (#1005's prefix cache restores the head's cache with the trunk's).
    pub const Head = struct {
        kv: state.KVCacheSnapshot,
        entry: state.SSMCheckpoint,
        meta: mtp.Qwen4MtpState.Meta,
        marks: QsaHeadMarkSet,

        pub fn deinit(h: *Head) void {
            h.kv.deinit();
            h.entry.deinit(gpa);
            h.marks.deinit();
        }

        pub fn bytes(h: *const Head) u64 {
            return kvBytes(&h.kv) + state.ssmCheckpointBytes(&h.entry);
        }

        pub fn clone(h: *const Head) !Head {
            var kv = try h.kv.trimmedCopy(h.kv.step, mlx.gpuStream());
            errdefer kv.deinit();
            var entry = try state.shareSsmCheckpoint(gpa, &h.entry);
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
        // The live state back on the slot: it sits on the head module while the slot is the active one.
        mtp_head.qwen4MtpRelease(t, &sl.st);
        var shared = try sl.st.cache.snapshot();
        defer shared.deinit();
        var kv = try shared.trimmedCopy(sl.st.cache.step, s);
        errdefer kv.deinit();
        const one = (&sl.st.entry)[0..1];
        var cps = [1]state.SSMCheckpoint{try state.captureSsmCheckpoint(gpa, one, @intCast(qsa_checkpoint.qsaHistoryRows(&sl.st.entry)), s)};
        errdefer cps[0].deinit(gpa);
        try qsa_checkpoint.attachQsaHistoryToLatest(&cps, one, s);
        return .{ .kv = kv, .entry = cps[0], .meta = sl.st.meta(), .marks = QsaHeadMarkSet.share(sl.st.qsa_marks.slice()) };
    }

    /// `restore_at`: slot `id` started over at `n` with the head history `head` (its arrays shared), or with none, so its
    /// rounds draft nothing until the next cold prompt.
    pub fn headRestore(d: *Drafter, id: u32, n: u64, head: ?*const Head) !void {
        if (d.open) |*o| if (o.slot == id) {
            o.deinit();
            d.open = null;
        };
        const t = &d.m.trunk;
        var fresh = try mtp_head.qwen4MtpStateNew(t);
        errdefer fresh.deinit();
        if (head) |h| {
            try fresh.cache.restore(&h.kv);
            try state.ssmRestore(&fresh.entry, &h.entry.layers[0]);
            fresh.restoreMeta(h.meta);
            fresh.qsa_marks = QsaHeadMarkSet.share(h.marks.slice());
        }
        const sl = try d.slotAt(id, n);
        endRequest(sl);
        mtp_head.qwen4MtpRelease(t, &sl.st);
        sl.st.deinit();
        sl.st = fresh;
        sl.prompt.clearRetainingCapacity();
        setLast(sl, .{ .ctx = null });
        sl.next = n;
        sl.from = 0;
        sl.base = if (head) |h| restoredBase(n, h.meta) else 0;
        sl.synced = head != null;
        sl.drafts = false;
    }

    /// After slot `id`'s forward of `n` rows at `pos` (their ids `ids` on the host, else `ids_dev` on the device) with
    /// the streams `streams` `[1, n, hc*H]` (borrowed; null for a forward the history `takes` nothing of): the rows whose
    /// next token is known join the history. `keep`: the forward is the engine's, whose rows past the prompt the
    /// request's policy has not counted yet.
    pub fn follow(d: *Drafter, id: u32, pos: u64, n: u32, ids: []const u32, ids_dev: Array, streams: Array, keep: bool) !void {
        const sl = try d.slotAt(id, pos);
        if (!sl.synced or n == 0) return;
        if (pos != sl.next) {
            sl.synced = false;
            return;
        }
        if (decodesPlain(sl, pos, n)) return d.quiet(sl);
        // Before the history's start: no rows, and the next row takes the head's next position.
        if (pos + n <= sl.from) {
            sl.next = pos + n;
            sl.base += n;
            return;
        }
        errdefer sl.synced = false;
        if (streams.ctx == null) return error.NoStreams;
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
        // Past `--max-mtp-ctx` the request drafts no more, and its rows join no history: #1005's serial block grows none.
        const held = if (sl.gen) |*g| g.spec_disabled_runtime and g.spec_disable_reason == .max_ctx else false;
        if (total > 0 and !held) {
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
            mtp_head.qwen4MtpActivate(t, &sl.st);
            // The first row sits at its token's position: `last`'s at `pos`, else row 0's at `pos + 1`.
            const at: c_int = @intCast((if (pend) pos else pos + 1) - sl.base);
            freeOut(try mtp_head.qwen4MtpForward(t, hist, row_ids, at, .none));
            // A prefill's rows: the head's history computes with its chunk, as #1005's prefill evaluates it with the
            // chunk's KV, so the prefix-cache snapshot and the first draft find it done.
            if (n > 1) try queueHistory(t);
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
        if (d.slots.get(id)) |s| s.drafts = true;
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

    /// What this round runs, as #1005's round entry decides it: null = serial (a context past `max_mtp_ctx`, the
    /// policy's adaptive switch, a probe or its exit, a request it disabled), else the plan.
    fn planOf(g: *Generator, max_mtp_ctx: u32) ?MtpRoundPlan {
        // `--max-mtp-ctx`, checked against prompt + generated: sticky (the context only grows), through the same
        // `spec_disabled_runtime` fallback.
        if (!g.spec_disabled_runtime and !mtp_mod.mtpCtxWithinLimit(max_mtp_ctx, g.mtpKvLen())) {
            log.info("  mtp=off (ctx {d} > --max-mtp-ctx {d})\n", .{ g.mtpKvLen(), max_mtp_ctx });
            g.spec_disabled_runtime = true;
            g.spec_disable_reason = .max_ctx;
        }
        // Adaptive serial re-entry: sticky, except when the request crossed into a different KV bucket. `.max_ctx` never
        // re-enters.
        if (g.spec_disabled_runtime and g.spec_disable_reason == .adaptive and g.mtpAdaptiveHeadMayResume()) {
            const b = g.mtpAdaptiveBucket(g.mtpKvLen());
            const prev_bucket = g.mtp_adaptive.bucket;
            const prev_arm = g.mtp_adaptive.arm;
            const action = g.mtp_adaptive.serialTick(b);
            g.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
            if (action == .to_mtp) {
                g.spec_disabled_runtime = false;
                g.spec_disable_reason = .none;
                g.mtp_serial_exit = .drain;
            }
        }
        if (g.spec_disabled_runtime or g.mtp_serial_left > 0 or g.mtp_serial_exit != .none) return null;
        // Cross-request EV seed: the head's last healthy acceptance surface, so the plan starts warm. A pinned depth
        // never plans, so it takes none.
        if (g.mtp_ev_rounds == 0 and g.mtp_attempted == 0 and regime.mtpPinnedDepth() == null) {
            if (g.mtp.?.evSeed()) |seed| {
                g.mtp_ev_accept = seed.accept;
                g.mtp_ev_m_lo_prev = @min(@max(seed.m_lo, 1), mtp_mod.MAX_DEPTH);
                g.mtp_ev_rounds = depth.MTP_EV_WARMUP_ROUNDS;
            }
        }
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

    /// The head's chain for `plan` from t1 at the head's position `pos`, #1005's (`mtpChainBuild`), opened: chunk A to
    /// `m_lo` (on a two-chunk plan drafting off the head's full logits, each draft's confidence beside it), dispatched.
    fn chainOpen(d: *Drafter, sl: *Slot, t1: u32, pos: u64, plan: MtpRoundPlan, mask: ?Array) !Chain {
        const t = &d.m.trunk;
        var tok: i32 = @intCast(t1);
        var c: Chain = .{ .t = t, .rerank = mtp_head.qwen4DraftRerankReady(t), .mask = mask, .h_prev = mlx.mlx_array_new(), .prev = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32) };
        errdefer c.deinit();
        try mlx.check(mlx.mlx_array_set(&c.h_prev, sl.last));
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

    /// The lookup drafts a round from t1 takes (#1005's `mtpLookupChain`), into `out`: 0 = the head's chain. A lookup
    /// draft is held to `--mtp-max-depth`, and one shorter than `--mtp-min-depth` is none.
    fn lookupDrafts(g: *Generator, plan: MtpRoundPlan, t1: u32, out: *[max_rows]u32) !u32 {
        const lookup_cap = depth_bounds.lookupCap(mtp_lookup.MAX_DRAFT_STRONG, depth_bounds.active);
        const k = try g.mtpLookupDrafts(gpa, plan, t1);
        if (k == 0 or k < depth_bounds.floorFor(depth_bounds.active, lookup_cap)) return 0;
        const got = (try g.mtpLookupIndex(gpa)).match(t1, lookup_cap).?;
        @memcpy(out[0..k], got.draft[0..k]);
        return k;
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
        // This round's own chain, and a lookup round's drafts one id each: open through the verify, which lends them.
        var opened: ?Chain = null;
        defer if (opened) |*c| c.deinit();
        var lookup_ids: [max_rows]Array = undefined;
        var n_lookup_ids: usize = 0;
        defer for (lookup_ids[0..n_lookup_ids]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var lent: []const Array = &.{};
        const planned: ?MtpRoundPlan = if (g) |gg| planOf(gg, d.max_mtp_ctx) else null;
        if (planned) |plan| {
            const gg = g.?;
            const sl = d.ready(id, pos).?;
            mtp_head.qwen4MtpActivate(t, &sl.st);
            b.off0 = d.headRows();
            b.plan = plan;
            var got: [max_rows]u32 = undefined;
            const k = try lookupDrafts(gg, plan, t1, &got);
            if (k > 0) {
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
                opened = try d.chainOpen(sl, t1, pos - sl.base, plan, mask);
                const c = &opened.?;
                const closed = try d.chainClose(c, gg, pos - sl.base, plan);
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
        const slot = [_]u32{id};
        const none: abi.Span(u32) = .{};
        _ = q4.forwardSettle(d.m, .of(&slot), .discard, &none);
        if (ownOf(d.m, id)) |own| for (own.ssm) |*e| state.ssmFreeSpecCapture(e);
        if (d.slots.get(id)) |sl| sl.synced = false;
    }

    /// Slot `id` back to t1 and the first `accepted` of `drafts`, verified at `pos`: the trunk's KV truncated (its
    /// step kept at `kv_step`, the value before the verify) and its recurrent states taken from the verify's capture,
    /// then (with `off0`, the head's row count before the round) the head truncated to it and the committed rows
    /// appended, `last` the stream at the last committed position.
    fn commit(d: *Drafter, id: u32, pos: u64, t1: u32, drafts: []const u32, accepted: u32, off0: ?usize, streams: Array, kv_step: usize) !void {
        const t = &d.m.trunk;
        const s = t.s;
        const own = ownOf(d.m, id) orelse return error.MtpSlotGone;
        const n_drafts: u32 = @intCast(drafts.len);
        defer for (own.ssm) |*e| state.ssmFreeSpecCapture(e);
        if (accepted < n_drafts) {
            try own.cache.truncate(pos + accepted + 1, s);
            own.cache.step = kv_step;
            for (own.ssm) |*e| try state.ssmRollbackFromCapture(e, accepted, n_drafts + 1, s);
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
        // The round's draft rows leave the head; the committed rows join it from the verify's own streams: `last` with
        // t1, then each kept draft with the stream before it.
        mtp_head.qwen4MtpActivate(t, &sl.st);
        try mtp_head.qwen4MtpTruncate(t, origin);
        var hidden = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hidden);
        try rowsAfter(&hidden, sl.last, streams, accepted, n_drafts + 1, s);
        var ids: [max_rows]i32 = undefined;
        ids[0] = @intCast(t1);
        for (drafts[0..accepted], 1..) |dv, i| ids[i] = @intCast(dv);
        const id_arr = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(accepted + 1)}, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        freeOut(try mtp_head.qwen4MtpForward(t, hidden, id_arr, @intCast(pos - sl.base), .none));
        var last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(last);
        try sliceRows(&last, streams, @intCast(accepted), @intCast(accepted + 1), s);
        setLast(sl, last);
        sl.next = pos + accepted + 1;
    }

    /// An open verify the engine never finished: discarded, its drafts never kept.
    fn closeOpen(d: *Drafter) !void {
        var o = d.open orelse return;
        d.open = null;
        defer o.deinit();
        try d.commit(o.slot, o.pos, o.t1, o.drafts[0..o.n_drafts], 0, o.off0, o.streams, o.step);
        try d.done(o.slot, o.pos + 1, o.lookup, o.plan, o.n_drafts, 0, o.t1, &.{}, o.off0 != null, o.watch);
    }

    /// The round's end for the request's policy, as #1005's round finish counts it, when the slot's history came through
    /// the round whole (to `next_pos`): a serial tick's bookkeeping, else the round toward its kind, the committed
    /// tokens, the acceptance observers and the EV surface, then its cost into the model's table and the regime and the
    /// chunk-A boundary's price. A lookup round, and the head round after one, are no width sample: a lookup round is
    /// priced in its own row.
    fn done(d: *Drafter, id: u32, next_pos: u64, lookup: bool, plan: MtpRoundPlan, drafted: u32, accepted: u32, t1: u32, drafts: []const u32, drafted_round: bool, watch: io_util.Stopwatch) !void {
        const sl = d.slots.get(id) orelse return;
        if (sl.next != next_pos) return;
        const g = if (sl.gen) |*gg| gg else return;
        if (!drafted_round) return serialDone(g, t1);
        const timed = !lookup and !g.mtp_after_lookup;
        g.mtp_after_lookup = lookup;
        if (lookup) g.mtp_lookup_rounds += 1 else g.mtp_attempted += 1;
        try g.generated_ids.append(gpa, t1);
        try g.generated_ids.appendSlice(gpa, drafts[0..accepted]);
        g.mtpRoundAcceptObserve(lookup, drafted, accepted);
        if (!lookup) g.updateMtpEvRound(drafted, accepted);
        if (timed) {
            g.mtpRoundEndObserve(drafted, accepted + 1, plan.m_hi > plan.m_lo, plan.m_lo, plan.width_trial, @as(f32, @floatFromInt(watch.read())) / @as(f32, std.time.ns_per_ms));
            if (g.mtpEvFixOn()) g.mtpEvFixObserve(plan, drafted, accepted);
            g.mtp_ev_round_ms = regime.mtpEmaMs(g.mtp_ev_round_ms, watch.read());
        } else {
            if (lookup) g.mtpLookupObserve(drafted, accepted + 1);
            g.mtpRoundUntimed();
        }
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
        var accepted: u32 = 0;
        while (accepted < b.n and dr[accepted] == am[accepted]) accepted += 1;
        accepted = @min(accepted, f.accept_cap);
        try d.commit(id, pos, t1, dr[0..b.n], accepted, b.off0, b.streams, b.step);
        const next_pos = pos + accepted + 1;
        out.round = .{ .n = accepted + 1, .accepted = accepted, .next_token = am[accepted], .rows = b.n + 1, .position = next_pos };
        out.round.tokens[0] = t1;
        @memcpy(out.round.tokens[1 .. accepted + 1], dr[0..accepted]);
        try d.done(id, next_pos, b.lookup, b.plan, b.n, accepted, t1, dr[0..b.n], b.off0 != null, watch);
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
        try d.commit(o.slot, o.pos, o.t1, o.drafts[0..o.n_drafts], a, o.off0, o.streams, o.step);
        try d.done(o.slot, o.pos + a + 1, o.lookup, o.plan, o.n_drafts, a, o.t1, o.drafts[0..o.n_drafts], o.off0 != null, o.watch);
        out.round = .{ .n = a + 1, .accepted = a, .rows = o.n_drafts + 1, .position = o.pos + a + 1 };
    }
};

/// The head's draft steps of one round (#1005's `MtpPreDraft` as `mtpChainBuild` fills it): each step's id `[1]`
/// int32, its confidence on a two-chunk plan's chunk A, and the stream the next step reads.
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

    fn deinit(c: *Chain) void {
        for (c.ids[0..c.n_ids]) |a| _ = mlx.mlx_array_free(a);
        for (c.confs[0..c.n_confs]) |a| _ = mlx.mlx_array_free(a);
        _ = mlx.mlx_array_free(c.h_prev);
        _ = mlx.mlx_array_free(c.prev);
    }

    /// Steps `from..to` at positions `pos + i`, each draft the rerank pick (the coarse head's shortlist re-scored) or,
    /// with `conf` (a two-chunk plan's chunk A, `mtpDraftStepPath`'s `.full_logits`), the argmax of the head's full
    /// logits with its confidence beside it.
    fn build(c: *Chain, pos: u64, from: u32, to: u32, conf: bool) !void {
        const s = c.t.s;
        const rerank = c.rerank and !conf;
        var i: u32 = from;
        while (i < to) : (i += 1) {
            const o = try mtp_head.qwen4MtpForward(c.t, c.h_prev, c.prev, @intCast(pos + i), if (rerank) .mixed_last_row else .last_row);
            defer freeOut(o);
            const id = if (rerank) try mtp_head.qwen4DraftSelect(c.t, o.mixed, c.mask) else try maskedArgmax(o.logits, c.mask, s);
            defer _ = mlx.mlx_array_free(id);
            var as32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(as32);
            try mlx.check(mlx.mlx_astype(&as32, id, .int32, s));
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

/// A request's end: its `[spec-stats]` line said, and its policy state gone, a healthy run's acceptance surface seeding
/// the next request's plans (#1005's Generator.deinit). The server releases the slot as the request finishes, right
/// after its own `[spec-stats]` line.
fn endRequest(sl: *Slot) void {
    for (sl.pending.items) |a| _ = mlx.mlx_array_free(a);
    sl.pending.clearRetainingCapacity();
    var g = sl.gen orelse return;
    sl.gen = null;
    var buf: [256]u8 = undefined;
    const line = specStatsLine(&buf, &g);
    if (line.len > 0) log.info("{s}", .{line});
    // A pinned depth's surface is not one the policy chose: it seeds nothing.
    if (g.mtp) |head| {
        if (regime.mtpPinnedDepth() == null and !g.spec_disabled_runtime and g.mtp_attempted >= 8) head.setEvSeed(g.mtp_ev_accept, g.mtp_ev_m_lo_prev);
    }
    g.deinit(gpa);
}

/// A request's `[spec-stats]` line, the head of #1005's for an MTP request (`stats_log.zig`): its head rounds and their
/// accepted drafts, whether the policy ended speculation, and its lookup rounds and their accepted drafts. Empty for a
/// request that ran no round. Head and lookup rounds are counted apart, as #1005's line counts them.
fn specStatsLine(buf: []u8, g: *const Generator) []const u8 {
    if (g.mtp_attempted == 0 and g.mtp_lookup_rounds == 0) return buf[0..0];
    const avg: f64 = @as(f64, @floatFromInt(g.mtp_accepted_tokens)) / @as(f64, @floatFromInt(@max(g.mtp_attempted, 1)));
    const disabled: []const u8 = if (g.spec_disabled_runtime) "true" else "false";
    return std.fmt.bufPrint(buf, "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} runtime_disabled={s} reason={s} lookup={d}/{d}\n", .{
        g.mtp_attempted, g.mtp_accepted_tokens, avg, disabled, @tagName(g.spec_disable_reason), g.mtp_lookup_rounds, g.mtp_lookup_accepted,
    }) catch buf[0..0];
}

/// The arrays a solo verify's input concatenates, `[t1, drafts]`, into `buf` (#1005's `pleIdsSource`); null where the
/// input is not that concatenation.
fn pleIdsSource(buf: *[max_rows + 1]Array, t1: Array, drafts: []const Array, rows: u32) ?[]const Array {
    if (drafts.len > max_rows or rows != 1 + drafts.len) return null;
    buf[0] = t1;
    @memcpy(buf[1 .. 1 + drafts.len], drafts);
    return buf[0 .. 1 + drafts.len];
}

/// Where the head history of a prompt of `len` tokens starts when its prefill runs from `at`, as #1005's generate init
/// decides it: with `--mtp-history-window` (`window`), a prefill of more than `HISTORY_WINDOW_THRESHOLD` positions before
/// its last token keeps its last `window` only (`effectiveMtpHistoryWindow`), from the first chunk that ends inside them
/// (`chunkNeedsMtpHistory`); 0 when every chunk adds rows.
fn historyFrom(len: u64, at: u64, window: usize) u64 {
    const prefix = len -| at -| 1;
    if (window == 0 or prefix <= mtp_mod.HISTORY_WINDOW_THRESHOLD) return 0;
    return at + prefix - @min(window, prefix);
}

/// Whether the engine's forward of `n` rows at `pos` is a one-row decode past the prompt for a request it has run no
/// round for: one that never drafts. The engine runs a drafting request's first round at its first tick, where a
/// forced span (a thinking budget's close) comes in one forward of several rows.
fn decodesPlain(sl: *const Slot, pos: u64, n: u32) bool {
    return !sl.drafts and n == 1 and sl.prompt.items.len > 0 and pos >= sl.prompt.items.len;
}

/// The `base` of a history restored at `n` with the head state `meta`: row n takes the head's next position, after its
/// `stash_len` rows from `origin` on (an empty one's first row sits at 1).
fn restoredBase(n: u64, meta: mtp.Qwen4MtpState.Meta) u64 {
    if (meta.stash_len == 0) return n;
    return n + 1 - meta.stash_len - @as(u64, @intCast(meta.origin));
}

fn setLast(sl: *Slot, stream: Array) void {
    if (sl.last.ctx != null) _ = mlx.mlx_array_free(sl.last);
    sl.last = stream;
}

fn ownOf(m: *q4.Model, id: u32) ?*OwnSlot {
    for (m.own.items) |*o| if (o.id == id) return o;
    return null;
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

/// The head's live history, its KV rows and QSA history, queued for evaluation (`mlx_async_eval`).
fn queueHistory(t: *Trunk) !void {
    const m = &(t.qwen4_mtp orelse return);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (m.cache.entries) |e| if (e.initialized) for ([_]Array{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    const q = &m.entry;
    for ([_]Array{ q.aux_state, q.qsa_pooled, q.qsa_key_buf, q.qsa_pooled_buf, q.qsa_score_bank, q.qsa_score_buf }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    if (mlx.mlx_vector_array_size(vec) > 0) try mlx.check(mlx.mlx_async_eval(vec));
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

test "drafter: a request's [spec-stats] line, as the windows parse it" {
    var model: generator.Model = .{ .trunk = undefined };
    var g: Generator = .{ .xfm = &model, .timer = undefined };
    var buf: [256]u8 = undefined;
    // A request that ran no round says nothing.
    try testing.expectEqualStrings("", specStatsLine(&buf, &g));
    g.mtp_attempted = 3;
    g.mtp_accepted_tokens = 5;
    g.spec_disabled_runtime = true;
    g.spec_disable_reason = .acceptance;
    g.mtp_lookup_rounds = 2;
    g.mtp_lookup_accepted = 7;
    try testing.expectEqualStrings("  [spec-stats] mode=mtp attempts=3 accepts=5 avg_per_round=1.67 runtime_disabled=true reason=acceptance lookup=2/7\n", specStatsLine(&buf, &g));
}
