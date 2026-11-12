//! RQ, kept and built in (no option: it always runs): the expert bank's decode rows sized per request, for the KV
//! mlx-serve 66aad949 allocates, in place of one plan for the KV the bill holds (C1: the served context's, src/bill.zig
//! `contextOf`; `Terms.kv_tokens`).
//! MLX-free: the host's capacity schedule replayed over a request's appends (D2's included: an MLA layer's prompt spans
//! past 24,576 in one append, `d2_from`), the plan, the crossing check, the prompt refusal and the slot table.
//! plugin.zig runs them at its re-plan points (a request's first prompt forward, its first decode forward, and any
//! forward whose append would pass the plan's capacity: the trim), src/ling_mtp.zig at the MTP head's appends, and the
//! copied streamer's source (overlay ling_source.zig `setPhase`, `trimDecode`) grows and trims the bank's decode rows to
//! the rows they give. The prompt rows stay the bank's base (stage 1): the copied streamer allocates the base at open and
//! changes only the rows past it.
//!
//! mlx-serve 66aad949 reserves no KV for a whole-prompt prefill (generate/init.zig:335 and :749-773 skip the chunk
//! loop's reserve, :345-384), so each KV cache grows lazily: at each append past its capacity, to `nextCapacity`
//! (src/sdk/kv/cache.zig:238-249). The KV a request holds is that schedule replayed over its appends: the prompt's
//! (`Appends`: its spans, `Forward.prompt_spans`, in D2's groups, cut where C4's segments end), the MTP head's history
//! after them (`replayHistory`: in C4's forwards past a 32,768-token prompt), then decode's rows. A plan bills the trunk's
//! MLA
//! layers at their capacity, the MTP head's latent cache at its own (the head's history grows on its own schedule), and
//! at the plan's last growth one MLA layer's old buffer: the copy a growth makes is bounded to one layer at a time,
//! because each routed layer's routing barrier evaluates the graph up to that layer (the copied experts.zig's
//! `g.hostIds`, ops.zig:832-834).
//!
//! The same text by construction: RQ changes only which experts are resident, by the decode rows' count. A resident row
//! and a streamed row hold the same record bytes, and a decode wave computes each routed pair from the (bank, row) that
//! holds its expert, at weight 1 (src/ling_math.zig `downBanked`, turboexl3's `moeLingBanked`), as a prompt call's
//! pairs take the prefill GEMM whatever the residency split (src/ling_math.zig, the 17-pair padding). So the bank
//! change needs no byte test. That rests on one host fact (the RQ design's section 7): mlx-serve's admission bills this
//! plugin 0 MB (its `[admission] needed=0 MB ... width=4096` line), so the decode rows still allocated when a request
//! is admitted cannot narrow its prefill width. If that bill ever became non-zero, they could (66aad949
//! src/engines/mlx/prefill.zig 490-523), and the text could change.
const std = @import("std");
/// C4: the layer-major pass's segments and the MTP head's history forwards (src/ling_segments.zig).
const ling_segments = @import("ling_segments.zig");

// ── mlx-serve 66aad949's KV capacity schedule ──

/// The capacity mlx-serve 66aad949 grows a KV buffer to so `needed` tokens fit, from `cur`
/// (src/sdk/kv/cache.zig:238-249, `KVCache.nextCapacity`): a quarter of the capacity, at least one 256-token step and
/// at most 8,192 tokens, then up to whole steps; `cur` when the tokens fit. A cache with no reservation takes it at
/// every grow (`nextCapacityReserved`, :281-288): a whole-prompt prefill reserves nothing, and neither does the MTP
/// head's cache.
pub fn nextCapacity(cur: u64, needed: u64) u64 {
    if (needed <= cur) return cur;
    const step = @min(@max(cur / 4, chunk_step), max_grow_tokens);
    const target = @max(needed, cur + step);
    return (target + chunk_step - 1) / chunk_step * chunk_step;
}
const chunk_step: u64 = 256;
const max_grow_tokens: u64 = 8192;

/// A replay's result: the capacity, and the capacity before its last growth (0: it never grew), one MLA layer's buffer
/// at which is the copy that growth makes.
pub const Growth = struct { cap: u64, old: u64 = 0 };

/// A cache's capacity after appends that end at `ends` (ascending absolute positions: a prompt forward's spans), from
/// `from`, its capacity before them.
pub fn replay(from: u64, ends: []const u64) Growth {
    var g: Growth = .{ .cap = from };
    for (ends) |e| {
        const c = nextCapacity(g.cap, e);
        if (c != g.cap) g.old = g.cap;
        g.cap = c;
    }
    return g;
}

/// Decode's growth: a cache at `from` tokens of capacity, appended a few rows at a time (a decode or verify forward's
/// rows, never one growth step's 256), until it holds `end` tokens. Each growth comes at the first append past the
/// capacity and gives the same capacity whatever rows that append adds.
pub fn grownTo(from: u64, end: u64) Growth {
    var g: Growth = .{ .cap = from };
    while (g.cap < end) {
        g.old = g.cap;
        g.cap = nextCapacity(g.cap, g.cap + 1);
    }
    return g;
}

/// mlx-serve's prefill chunk as served (the `[admission] ... width=4096` line), its snapshot backoff (66aad949
/// src/engines/mlx/generate/prefill_chunk.zig:277, `SSM_SNAPSHOT_BACKOFF`) and the tail a chunk absorbs
/// (src/sdk/bases/run.zig:301, `TAIL_MERGE_MAX`).
pub const prefill_chunk: u64 = 4096;
const snapshot_backoff: u64 = 30;
const tail_merge_max: u64 = 512;

/// The span ends mlx-serve 66aad949 names for a whole prompt of `n` tokens from position 0 (generate/init.zig:749,
/// prefill_chunk.zig:445-469 `wholePromptPlan`): the chunk loop's ends at `prefill_chunk` up to the snapshot backoff
/// (30 tokens short of the prompt's last token), a tail under 512 tokens merged into the chunk before it, then `n`.
/// The checkpoint stride is coarsened to the chunk (init.zig:246-248), so no stride boundary falls inside a chunk.
pub const Spans = struct {
    n: u64,
    loop_end: u64,
    pos: u64 = 0,
    last: bool = false,

    pub fn init(n: u64) Spans {
        const prefix_len = n -| 1;
        const backoff: u64 = if (prefix_len <= snapshot_backoff) 0 else snapshot_backoff;
        return .{ .n = n, .loop_end = prefix_len - backoff };
    }

    pub fn next(s: *Spans) ?u64 {
        if (s.pos < s.loop_end) {
            var end = @min(s.pos + prefill_chunk, s.loop_end);
            if (end < s.loop_end and s.loop_end - end < tail_merge_max) end = s.loop_end;
            s.pos = end;
            return end;
        }
        if (s.last or s.n <= s.loop_end) return null;
        s.last = true;
        return s.n;
    }
};

/// `Spans` into `buf`, at most its length.
pub fn spanEnds(buf: []u64, n: u64) []const u64 {
    var it = Spans.init(n);
    var k: usize = 0;
    while (k < buf.len) : (k += 1) buf[k] = it.next() orelse break;
    return buf[0..k];
}

/// The capacity mlx-serve 66aad949 allocates for a whole prompt of `n` tokens from position 0, as this plugin appends
/// it: `nextCapacity` replayed over its spans, the spans from `d2_from` on in one append (`wholePromptCapacityFrom`).
/// Rule 7's limit reads it (src/bill.zig `echoRowsMax`); a served forward replays its own spans (`replayAppends`).
pub fn wholePromptCapacity(n: u64) u64 {
    return wholePromptCapacityFrom(n, d2_from);
}

// ── C1: the served context (`--ctx-size`) and what the bill holds for it ──

/// C1: the KV capacity the bill holds for a served context of `context` tokens (src/bill.zig `contextOf`): the most
/// mlx-serve allocates for any whole prompt up to it (`wholePromptCapacity`, D2's appends), so every whole prompt
/// mlx-serve admits at that `--ctx-size` keeps its KV within the bill. The capacity is not monotone in the prompt's
/// length: a backoff span that ends just under a 256-token step gives the last 31 tokens a quarter's growth (1,290
/// tokens get 1,792, 1,312 get 1,536). It is at most one growth step past the prompt (`nextCapacity`: at most 8,192
/// tokens, rounded up to 256), so only the lengths within that of `context` can pass `context`'s own capacity, which is
/// at least `context`. At 32,768 it is 32,768 (D2's position holds every whole prompt up to it within it).
pub fn promptCapacityMax(context: u64) u64 {
    var most: u64 = 0;
    var n: u64 = @max(1, context -| (max_grow_tokens + chunk_step));
    while (n <= context) : (n += 1) most = @max(most, wholePromptCapacity(n));
    return most;
}

/// C1: the SSM checkpoints mlx-serve names for a whole prompt of `n` tokens from position 0 (`Forward.ssm_checkpoints`;
/// 9a173773 prefill_chunk.zig `wholePromptPlan`, the stride coarsened to the 4,096-token chunk): every span's end but
/// the prompt's, the stride's multiples and the snapshot backoff's point. The layer-major pass holds each until every
/// layer has passed it (plugin.zig `spanMajorOr`). Monotone in `n`: a context's count is its longest prompt's.
pub fn checkpointsOf(n: u64) u64 {
    var it = Spans.init(n);
    var k: u64 = 0;
    while (it.next()) |_| k += 1;
    return k -| 1;
}

// ── D2: one KV append for an MLA layer's prompt spans past 24,576 ──

/// D2's position: an MLA layer's prompt spans that start here or later append their keys and values in one call, two
/// or more of them (the copied mla.zig `mlaGroup`, from the copied table_forward.zig `hiddenSpans`), so mlx-serve's KV
/// grows at most once for them where an append a span grows it once a span. From a cold prompt the capacity here is
/// 25,600 tokens (the growth at the 4,096-token chunk that ends at 24,576), and one append from 25,600 to any P up to
/// the bill's 32,768-token context gives max(P, 32,000) rounded up to 256 tokens, at most 32,768; an append a span grows
/// from 32,000 to 40,192 at the first past 32,000. Of the chunk boundaries it is the one that both holds every whole
/// prompt up to that context within that term and keeps an append a span's capacity for every whole prompt up to
/// 32,000 tokens (a test sweeps them): from 20,480 a 28,000-token prompt would get 28,160, and at 28,672 the KV is
/// already 32,000. One append writes the words the appends a span wrote (each span's projections stay at its own rows),
/// and each span attends over the rows up to its own end, so D2 changes no word.
pub const d2_from: u64 = 24_576;

/// D2's position in a whole-prompt forward from `pos0`: `d2_from`, and past the forward's first span, whose append stays
/// its own. mlx-serve's first append after a restored prefix decides from its rows whether to regrow from the restored
/// rows or keep the restored share's buffer (66aad949 src/sdk/kv/cache.zig `restoredOversized`): a group there could
/// keep a share's larger buffer where RQ's replay regrows from the prefix.
pub fn d2Start(pos0: u64) u64 {
    return @max(d2_from, pos0 + 1);
}

/// D2's group among a layer-major call's spans (`starts`: each span's first position, ascending; table_forward.zig
/// `hiddenSpans` at an MLA layer): the index of the first span that starts at `from` (`d2Start`) or later when two or
/// more do, whose spans append together; else `starts.len` (each span appends alone).
pub fn groupStart(starts: []const u32, from: u64) usize {
    for (starts, 0..) |s, j| if (s >= from) return if (starts.len - j >= 2) j else starts.len;
    return starts.len;
}

/// The positions where a whole-prompt forward's appends end at one MLA layer: the forward's spans `ends` (ascending, the
/// first from `pos0`, as the host names them), each its own append but in D2's groups. plugin.zig `spanMajorOr` forwards
/// a span of `lane` rows or fewer (a decode lane's) on its own and runs each run of longer spans in C4's segments
/// (src/ling_segments.zig `Plan`), each segment one `hiddenSpans` call, where the spans from `groupStart` on append once,
/// at the segment's last end.
pub const Appends = struct {
    pos0: u64,
    ends: []const u64,
    lane: u64,
    from: u64,
    /// C4: a segment's rows at most (src/ling_segments.zig `segment_rows` as served), unless one routed call is wider or a
    /// last group under `segment_tail_rows` rows joins it.
    max_rows: u64,
    j: usize = 0,
    /// The current segment's end (an index into `ends`, past its last span).
    seg_end: usize = 0,

    pub fn init(pos0: u64, ends: []const u64, lane: u64) Appends {
        return initFrom(pos0, ends, lane, d2Start(pos0), ling_segments.segment_rows);
    }

    /// `init` with D2's position at `from` and C4's segments at `max_rows` rows.
    pub fn initFrom(pos0: u64, ends: []const u64, lane: u64, from: u64, max_rows: u64) Appends {
        return .{ .pos0 = pos0, .ends = ends, .lane = lane, .from = from, .max_rows = max_rows };
    }

    fn startOf(a: *const Appends, j: usize) u64 {
        return if (j == 0) a.pos0 else a.ends[j - 1];
    }

    /// Span `j` is longer than a decode lane's rows: the pass runs it in a run's segment, not as a forward of its own.
    fn wide(a: *const Appends, j: usize) bool {
        return a.ends[j] - a.startOf(j) > a.lane;
    }

    /// The end of the segment that starts at span `j`, a wide span at a segment's start: the first segment of the plan of
    /// the run's spans from `j` on. A segment starts at one of the whole run's call ends, where the calls start afresh
    /// (src/ling_segments.zig's test), so this is the whole run's segment there.
    fn segmentEnd(a: *const Appends, j: usize) usize {
        var rows: [ling_segments.max_spans]u32 = undefined;
        var k = j;
        while (k < a.ends.len and k - j < rows.len and a.wide(k)) : (k += 1) rows[k - j] = @intCast(a.ends[k] - a.startOf(k));
        var p: ling_segments.Plan = undefined;
        p.init(rows[0 .. k - j], a.max_rows) catch return k;
        const sg = p.next() orelse return k;
        return j + sg.end;
    }

    pub fn next(a: *Appends) ?u64 {
        while (a.j < a.ends.len) {
            const at = a.j;
            a.j += 1;
            const end = a.ends[at];
            // a decode lane's span: a forward of its own, so an append of its own
            if (!a.wide(at)) return end;
            if (at >= a.seg_end) a.seg_end = a.segmentEnd(at);
            // a grouped span whose segment goes on appends with the spans after it
            if (a.startOf(at) >= a.from and at + 1 < a.seg_end) continue;
            return end;
        }
        return null;
    }
};

/// A whole-prompt forward's KV growth (`replay` over its `Appends`), from `from`, its capacity before them.
pub fn replayAppends(from: u64, pos0: u64, ends: []const u64, lane: u64) Growth {
    var g: Growth = .{ .cap = from };
    var it = Appends.init(pos0, ends, lane);
    while (it.next()) |e| {
        const c = nextCapacity(g.cap, e);
        if (c != g.cap) g.old = g.cap;
        g.cap = c;
    }
    return g;
}

/// `wholePromptCapacity` with D2's position at `from` (from position 0, `d2Start(0)` is `d2_from`): each span that starts
/// before it appends alone, and the spans from it on in one append a C4 segment (one to `n` for a prompt of 32,768
/// tokens or fewer, one segment). A whole prompt's spans past 24,576 each hold 31 rows or more, more than a decode
/// lane's, so none of them is forwarded alone to split the group.
pub fn wholePromptCapacityFrom(n: u64, from: u64) u64 {
    var buf: [ling_segments.max_spans]u64 = undefined;
    var it = Appends.initFrom(0, spanEnds(&buf, n), 0, from, ling_segments.segment_rows);
    var cap: u64 = 0;
    while (it.next()) |e| cap = nextCapacity(cap, e);
    return cap;
}

/// The MTP head's history after a prompt that ends at `p` from the restored prefix `r` (its rows less one: `p - 1 - r`
/// rows from `r`), replayed from `from` as src/ling_mtp.zig appends it: one append, or C4's forwards of
/// `ling_segments.historyRows` rows past a 32,768-token prompt.
pub fn replayHistory(from: u64, r: u64, p: u64) Growth {
    const end = p -| 1;
    const step = ling_segments.historyRows(end -| r);
    var g: Growth = .{ .cap = from };
    var at = r;
    while (at < end) {
        at = @min(at + step, end);
        const c = nextCapacity(g.cap, at);
        if (c != g.cap) g.old = g.cap;
        g.cap = c;
    }
    return g;
}

// ── The plan ──

/// The KV bytes a token by cache (src/bill.zig `ropeOnceLayer`, `latentKv`): one trunk MLA layer's on the rope-once
/// cache, the trunk's MLA layers, and the MTP head's latent cache.
pub const Kv = struct {
    layer: u64,
    layers: u64,
    head: u64,

    pub fn trunk(k: Kv) u64 {
        return k.layer * k.layers;
    }

    /// Both caches' bytes a token: the bill's KV a token (plugin.zig `kvPerToken`).
    pub fn perToken(k: Kv) u64 {
        return k.trunk() + k.head;
    }
};

/// The bank's terms beside its slot rows (ling_source.zig `billBank`): a row of every routed layer, a phase's windows,
/// the read pool's staging past its default workers, and a layer's experts.
pub const Bank = struct { row_bytes: u64, prompt_windows: u64, decode_windows: u64, staging: u64, n_experts: u32 };
/// The phases' budgets for the slot rows and their terms (src/bill.zig `budget`).
pub const Budget = struct { prompt: u64, decode: u64 };
pub const Rows = struct { prompt: u32, decode: u32 };

/// The whole rows `budget` holds past `fixed` bytes of a phase's own terms (its windows and the staging).
pub fn rowsIn(bank: Bank, budget: u64, fixed: u64) u64 {
    return (budget -| fixed) / bank.row_bytes;
}

/// The bank's rows at the budgets it opens with (moved here from overlay ling_source.zig `billBank`): each phase's
/// whole rows past its windows and the staging; the prompt at least one row, decode at least the prompt's, both at most
/// a layer's experts.
pub fn rowsOf(bank: Bank, budget: Budget) error{StreamedRowsDoNotFit}!Rows {
    const n: u64 = bank.n_experts;
    const p = rowsIn(bank, budget.prompt, bank.prompt_windows + bank.staging);
    const d = rowsIn(bank, budget.decode, bank.decode_windows + bank.staging);
    if (p == 0) return error.StreamedRowsDoNotFit;
    return .{ .prompt = @intCast(@min(p, n)), .decode = @intCast(@min(@max(d, p), n)) };
}

/// The plan's terms, fixed when the bank opens (plugin.zig `openBank`).
pub const Terms = struct {
    bank: Bank,
    /// The load-time budgets (src/bill.zig `budget`): each holds the KV of `kv_tokens` tokens at `kv.perToken()` a token.
    budget: Budget,
    /// The served context (C1: `--ctx-size`, src/bill.zig `contextOf`), where mlx-serve clamps generation: the decode
    /// plan's end (`decodeEnd`).
    context: u64,
    /// The KV the budgets hold (C1: `promptCapacityMax(context)`, the most mlx-serve allocates for a whole prompt up to the
    /// context): the term a plan's own KV replaces (`rowsAt`), and the most a prompt forward may grow (`promptRefusal`).
    kv_tokens: u64,
    kv: Kv,
    /// The rows the bank opened with: the prompt rows are its base.
    rows: Rows,
    /// C5: the layer-major pass's working set the prompt budget holds at the served context (src/bill.zig
    /// `passWorkBill`, `prompt_work` included; with no layer-major pass, `prompt_work`), the term a prompt plan's own pass
    /// replaces (`planPrompt`).
    pass_bill: u64 = 0,
    /// C5: what O reads at a fresh request that the pass's fit already holds (the host's resident draft-rerank head,
    /// src/bill.zig `rerankHead`): a prompt plan counts O past it.
    resident: u64 = 0,
    /// C5: the free RAM the bill keeps past the host's wired margin (src/bill.zig `free_ram`): its whole rows are the
    /// most a decode or trim plan may fall under the base and still be served at it (`overVerdict`).
    free_ram: u64 = 0,
};

pub const Phase = enum { prompt, decode };

/// The rows a phase's budget holds for a request: `held`, the whole rows past its windows and the staging once the
/// request's KV (`bill` bytes) replaces the load-time term and O (`other` bytes) is beside it; `rows`, what the bank
/// takes, at most a layer's experts. Under the base rows the plan is over (`over`): the bank stays at the base.
pub const Fit = struct { rows: u32, held: u64, over: bool };

pub fn rowsAt(t: Terms, phase: Phase, bill: u64, other: u64) Fit {
    const budget = switch (phase) {
        .prompt => t.budget.prompt,
        .decode => t.budget.decode,
    };
    const windows = switch (phase) {
        .prompt => t.bank.prompt_windows,
        .decode => t.bank.decode_windows,
    };
    const room = (budget + t.kv_tokens * t.kv.perToken()) -| (bill + other);
    const held = rowsIn(t.bank, room, windows + t.bank.staging);
    if (held < t.rows.prompt) return .{ .rows = t.rows.prompt, .held = held, .over = true };
    return .{ .rows = @intCast(@min(held, t.bank.n_experts)), .held = held, .over = false };
}

/// H, the decode plan's headroom past the request's position, in tokens. mlx-serve 66aad949 hands the plugin no
/// max_tokens (Forward, LoadCtx and the host table carry none), so the plan holds the KV of a fixed headroom, capped
/// where mlx-serve clamps generation: at the context, the prompt and the rest of it (src/host/api/sampling.zig
/// `clampMaxTokens`). It is 4,096 tokens because that is the host's unit, not a benchmark's: one prefill chunk
/// (`prefill_chunk`), and from a capacity of 16,384 tokens on, one growth step of the host's schedule holds it, so a
/// plan's decode there bills one growth at most. It is not the standard test's max_tokens (1,024): a headroom fitted to
/// one benchmark's answers would give its requests rows that a longer answer must give back. A request that decodes
/// past its plan's capacity takes a trim (`crosses`), and a new plan from there.
pub const headroom: u64 = 4096;

/// How far the MTP head's cache may run past the trunk's position: its draft steps, fewer than the widest decode
/// forward's rows (the copied experts.zig `decode_forward_rows`, 8; a verify checks its drafts and one more row). A
/// decode plan holds the head's cache to the trunk's capacity plus these, so the head passes its plan only after the
/// trunk has passed its own.
pub const head_margin: u64 = 8;

/// The decode plan's end: `pos` (the request's position: its prompt's end at the decode start, the crossing forward's
/// first position at a trim) plus the headroom, capped at `context` (mlx-serve's clamp), and at least the forward's own
/// `n_rows` past `pos`.
pub fn decodeEnd(pos: u64, n_rows: u64, context: u64) u64 {
    return @max(@min(pos + headroom, context), pos + n_rows);
}

/// Where a plan is made: a request's prompt start, its decode start, or a trim.
pub const Kind = enum { prompt, decode, trim };

/// A re-plan's result, for the bank and its RQ line.
pub const Plan = struct {
    kind: Kind = .prompt,
    /// The trunk's and the MTP head's capacity the plan holds (tokens).
    cap: u64 = 0,
    head_cap: u64 = 0,
    /// The KV bytes it bills: both caches at those capacities, and its largest growth copy (a prompt plan's copy is its
    /// pass's, `pass`).
    kv: u64 = 0,
    /// A prompt plan's pass working set (C5, src/bill.zig `passWork`); 0 for a decode plan.
    pass: u64 = 0,
    /// O, the other memory beside it (bytes).
    other: u64 = 0,
    /// The rows the bank takes, the whole rows the budget holds, and whether those are under the base (`Fit`). A
    /// prompt plan's rows are the base: stage 1 leaves the prompt rows there.
    rows: u32 = 0,
    held: u64 = 0,
    over: bool = false,
};

/// The KV bytes at the trunk's and the head's capacities, with the larger growth copy: one trunk MLA layer's old
/// buffer, or the head's (its cache grows in its own forwards, never inside the trunk's).
fn kvBill(k: Kv, tg: Growth, hg: Growth) u64 {
    return tg.cap * k.trunk() + hg.cap * k.head + @max(tg.old * k.layer, hg.old * k.head);
}

/// A request's prompt plan at its start (C5): the room the prompt budget holds with the KV and the pass the bill holds
/// (`Terms.kv_tokens`, `Terms.pass_bill`), less the request's KV at its capacity (both caches at the trunk's, as the
/// pass's fit reads them; `s.prompt`, replayed over its spans), its pass's working set (`pass`: src/bill.zig `passWork`
/// at its length, which holds the growth's copy and the host's rerank head) and O past what that fit holds
/// (`Terms.resident`). Before C5 the plan counted the growth's copy in the KV beside the pass's term and the rerank head
/// in O, so a 31,652-token prompt planned 211 rows under the base 214 at a peak 2 GB under the bill. Stage 1 keeps the
/// prompt rows at the base: under it the plan is over (`over`), which C5 refuses by name (`overVerdict`). The head's
/// capacity after the prompt's history (its rows less one, generate/init.zig:802; C4's `replayHistory`: 4,096-row
/// forwards past a 32,768-token prompt) is reported in the plan, not counted in the room.
pub fn planPrompt(t: Terms, s: *const Slot, other: u64, pass: u64) Plan {
    const hg = replayHistory(s.head_cap, s.r, s.p);
    const kv = s.prompt.cap * t.kv.perToken();
    const room = (t.budget.prompt + t.kv_tokens * t.kv.perToken() + t.pass_bill) -| (kv + pass + (other -| t.resident));
    const held = rowsIn(t.bank, room, t.bank.prompt_windows + t.bank.staging);
    return .{ .kind = .prompt, .cap = s.prompt.cap, .head_cap = hg.cap, .kv = kv, .pass = pass, .other = other, .rows = t.rows.prompt, .held = held, .over = held < t.rows.prompt };
}

/// C5 (main's ruling, 2026-11-11): what RQ does with a plan under the bank's base rows (`Plan.over`). The copied
/// streamer cannot take a phase under its base (stage 1), so the bank stays at the base, past the plan:
/// - a prompt plan under the base is refused by name, before any work: a prompt pass at the base past its plan is the
///   OOM case at 32K and up;
/// - a decode or trim plan under the base by at most the rows `free_ram` covers (`coverRows`: 12 of Ling's rows) keeps
///   the bank at the base with a named warning: its excess stays inside the free RAM the bill keeps;
/// - a decode or trim plan under the base by more than that is refused by name: its excess passes the free RAM.
/// The 32K check's decode plan holds 1.7 rows past the base, far from the cover.
pub const OverVerdict = enum { fits, warn, refuse };
pub fn overVerdict(t: Terms, pl: Plan) OverVerdict {
    if (!pl.over) return .fits;
    if (pl.kind == .prompt) return .refuse;
    return if (t.rows.prompt - pl.held <= coverRows(t)) .warn else .refuse;
}

/// The bank's whole rows the bill's free RAM holds (`Terms.free_ram`).
pub fn coverRows(t: Terms) u64 {
    return t.free_ram / t.bank.row_bytes;
}

/// C5's refusal's name and text (`overVerdict`'s `refuse`), written into `buf`.
pub const over_refusal_name = "turboexl3_ling: RQ over plan";
pub fn overRefusal(buf: []u8, t: Terms, s: *const Slot, pl: Plan) []const u8 {
    const why = if (pl.kind == .prompt) "a prompt plan under the base is refused by name" else "past the rows free RAM covers, refused by name";
    return std.fmt.bufPrint(buf, over_refusal_name ++ ": slot {d} P={d}: the {t} plan holds {d} rows, {d} under the bank's base of {d} (free RAM covers {d}), with O at {d} MiB (KV {d} MiB, the pass {d} MiB, the {d}-token context); {s}", .{ s.id, s.p, pl.kind, pl.held, t.rows.prompt -| pl.held, t.rows.prompt, coverRows(t), pl.other >> 20, pl.kv >> 20, pl.pass >> 20, t.context, why }) catch over_refusal_name;
}

/// A request's decode plan from its capacities as allocated (`s`), at `pos` with a forward of `n_rows` there
/// (`decodeEnd`), with O beside it: the trunk's cache, from its prompt's capacity at least (a prompt of a decode lane's
/// rows is planned before its forward appends), grown to the plan's end; the head's from its history (the prompt's
/// rows less one, appended before decode's first forward) to the trunk's capacity plus `head_margin`; against the
/// decode budget.
pub fn planDecode(t: Terms, s: *const Slot, kind: Kind, pos: u64, n_rows: u64, other: u64) Plan {
    const tg = grownTo(@max(s.trunk_cap, s.prompt.cap), decodeEnd(pos, n_rows, t.context));
    const hg = grownTo(replayHistory(s.head_cap, s.r, s.p).cap, tg.cap + head_margin);
    const bill = kvBill(t.kv, tg, hg);
    const f = rowsAt(t, .decode, bill, other);
    return .{ .kind = kind, .cap = tg.cap, .head_cap = hg.cap, .kv = bill, .other = other, .rows = f.rows, .held = f.held, .over = f.over };
}

/// The crossing check: a forward of `n_rows` rows from `pos` appends past the plan's `cap` tokens.
pub fn crosses(pos: u64, n_rows: u64, cap: u64) bool {
    return pos + n_rows > cap;
}

// ── The prompt refusal (option c, with D2; C1: at the served context's KV) ──

/// The refusal's name, the head of its text (rule 7's pattern, src/bill.zig `echoRefusal`).
pub const refusal_name = "turboexl3_ling: a prompt whose KV passes the plan's";

/// RQ's refusal: a prompt forward of `n_rows` rows from `first` whose KV mlx-serve's schedule grows to `cap` tokens
/// (`replayAppends`), past the KV the plan holds (`Terms.kv_tokens`), is refused by name, before any append; the reason
/// is written into `buf`. Any other forward returns null. The plan holds the most mlx-serve allocates for any whole
/// prompt up to the served context (C1, `promptCapacityMax`), so no whole prompt from position 0 that mlx-serve admits
/// at its `--ctx-size` is refused (mlx-serve answers a longer prompt with HTTP 400 before any forward, 66aad949
/// src/host/api/completions.zig:309-316). A forward past it would grow the KV past the plan inside the layer-major pass,
/// and the copied streamer cannot take the prompt phase under its base (sdk_ext/expert/stream.zig: the base is allocated
/// at open, :888; `shrink` takes no count but the base, :2849; `grow` none under the current count, :2760): it is a
/// forward from a restored prefix whose replay from the prefix passes it.
pub fn promptRefusal(buf: []u8, t: Terms, first: u64, n_rows: u64, cap: u64) ?[]const u8 {
    if (cap <= t.kv_tokens) return null;
    return std.fmt.bufPrint(buf, refusal_name ++ " {d} tokens: a {d}-row prompt forward from position {d} grows mlx-serve's KV to {d} tokens; whole prompts from position 0 up to the {d}-token context are served (D2: an MLA layer's spans from position {d} append in one call)", .{ t.kv_tokens, n_rows, first, cap, t.context, d2_from }) catch refusal_name;
}

// ── The slot table ──

/// A request's KV by its slot: what its plans read and set.
pub const Slot = struct {
    id: u32 = 0,
    /// The request's prompt end (P: its last prompt forward's first position plus rows) and the restored prefix its
    /// first prompt forward starts at (r; 0 for a cold prompt).
    p: u64 = 0,
    r: u64 = 0,
    /// The trunk's and the MTP head's KV capacity as mlx-serve allocated them so far (tokens).
    trunk_cap: u64 = 0,
    head_cap: u64 = 0,
    /// The prompt's growth: its capacity once its current forward has appended, and its last growth's old capacity.
    prompt: Growth = .{ .cap = 0 },
    /// The decode plan's trunk and head capacities (0: no decode plan yet), and its rows.
    plan_cap: u64 = 0,
    plan_head_cap: u64 = 0,
    rows: u32 = 0,
    /// The request's engaged line has printed.
    said: bool = false,
    /// The request's interleave line has printed (once a request, however often its forwards interleave).
    said_interleave: bool = false,
    stamp: u64 = 0,
};

/// The slots RQ keeps, at most `max_slots`, the least recently used dropped first. mlx-serve interleaves requests
/// (server.zig:491-508), and this plugin's slots are not exclusive (verdict.zig:177-209).
pub const max_slots = 8;

pub const SlotTable = struct {
    entries: [max_slots]Slot = @splat(.{}),
    used: [max_slots]bool = @splat(false),
    clock: u64 = 0,
    /// The slot of the last trunk forward (`begin`, `touch`): a head forward's append is that request's.
    last: ?u32 = null,

    /// A request starts in slot `id` from the restored prefix `r` (0 for a cold prompt): its entry fresh, both caches
    /// at `r` tokens (the restore's own rows; a restored buffer's capacity past them is O's).
    pub fn begin(t: *SlotTable, id: u32, r: u64) *Slot {
        t.clock += 1;
        t.last = id;
        const i = t.find(id) orelse t.vacant() orelse t.oldest();
        t.used[i] = true;
        t.entries[i] = .{ .id = id, .p = r, .r = r, .trunk_cap = r, .head_cap = r, .prompt = .{ .cap = r }, .stamp = t.clock };
        return &t.entries[i];
    }

    /// Slot `id`'s request, if one started.
    pub fn get(t: *SlotTable, id: u32) ?*Slot {
        const i = t.find(id) orelse return null;
        return &t.entries[i];
    }

    /// A trunk forward of slot `id` that starts no request: true when another slot's forwards ran since this slot's
    /// last (the requests interleave: O then holds the other request's KV as it is, not its growth).
    pub fn touch(t: *SlotTable, id: u32) bool {
        t.clock += 1;
        const prev = t.last;
        t.last = id;
        const i = t.find(id) orelse return false;
        t.entries[i].stamp = t.clock;
        const other = prev orelse return false;
        return other != id;
    }

    fn find(t: *const SlotTable, id: u32) ?usize {
        for (t.entries, t.used, 0..) |e, u, i| if (u and e.id == id) return i;
        return null;
    }

    fn vacant(t: *const SlotTable) ?usize {
        for (t.used, 0..) |u, i| if (!u) return i;
        return null;
    }

    fn oldest(t: *const SlotTable) usize {
        var at: usize = 0;
        for (t.entries, 0..) |e, i| if (e.stamp < t.entries[at].stamp) {
            at = i;
        };
        return at;
    }
};

// ── tests ──

const testing = std.testing;
const mib: u64 = 1 << 20;

// The served plan's terms (the 2.60 pack on m5-max-eigen, dylib 01ca9769; its load lines): the bank's budgets as
// logged, prompt 69,998 MiB and decode 76,329 MiB, `frac` bytes past the MiB the log rounds to; 68 routed layers of
// 4,947,456 B a record (336,427,008 B a row of every layer), 5 prompt windows of 48 records and 1 at decode,
// 118,489,088 B of staging (src/bill.zig's MEM_REFIT test); the rope-once trunk's 9 MLA layers and the MTP head's
// latent cache. C5's terms for that bill: its pass terms (prompt_work, the layer-major term 5,030,789,120 B and
// MEM_REFIT's pass part, 989 MiB) and the host's rerank head (281,673,728 B).
fn servedTerms(frac: u64) Terms {
    const row: u64 = 336_427_008;
    const window: u64 = 48 * (row / 68);
    const bank: Bank = .{ .row_bytes = row, .prompt_windows = 5 * window, .decode_windows = window, .staging = 118_489_088, .n_experts = 512 };
    const budget: Budget = .{ .prompt = 69998 * mib + frac, .decode = 76329 * mib + frac };
    return .{ .bank = bank, .budget = budget, .context = 32768, .kv_tokens = 32768, .kv = .{ .layer = 32_896, .layers = 9, .head = 1_152 }, .rows = rowsOf(bank, budget) catch unreachable, .pass_bill = (2 << 30) + 5_030_789_120 + (989 << 20), .resident = 281_673_728, .free_ram = 4 << 30 };
}

// A request at its decode start: its prompt's capacity, and the head's history (its rows less one) appended.
fn decodeStart(p: u64) Slot {
    return .{ .p = p, .trunk_cap = wholePromptCapacity(p), .head_cap = nextCapacity(0, p - 1) };
}

test "RQ's capacity replay: mlx-serve 66aad949's schedule over its whole-prompt spans (prefill chunk 4,096, the 30-token snapshot backoff) gives memfit's KV allocated at every ladder size: 1,120 -> 1,280, 2,042 -> 2,048, 4,094 -> 4,096, 8,185 -> 8,192, 16,377 -> 16,384, 31,652 -> 32,000" {
    // ladder-01ca9769/memfit-windows.txt: "prompt N tokens (KV allocated C)"
    const ladder = [_][2]u64{ .{ 1120, 1280 }, .{ 2042, 2048 }, .{ 4094, 4096 }, .{ 8185, 8192 }, .{ 16377, 16384 }, .{ 31652, 32000 } };
    for (ladder) |c| try testing.expectEqual(c[1], wholePromptCapacity(c[0]));
    // The standard request's spans: four chunks, the backoff point and the prompt's end; 20,480 tokens, the last growth
    // from 16,384.
    var buf: [16]u64 = undefined;
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 18349, 18380 }, spanEnds(&buf, 18380));
    try testing.expectEqual(Growth{ .cap = 20480, .old = 16384 }, replay(0, spanEnds(&buf, 18380)));
    // The 32K point's: the last growth 25,600 -> 32,000 at the chunk ending at 28,672.
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 28672, 31621, 31652 }, spanEnds(&buf, 31652));
    try testing.expectEqual(Growth{ .cap = 32000, .old = 25600 }, replay(0, spanEnds(&buf, 31652)));
    // A short prompt: one chunk to the backoff point, then the prompt's end.
    try testing.expectEqualSlices(u64, &.{ 1089, 1120 }, spanEnds(&buf, 1120));
}

test "RQ's capacity replay in decode: 32,000 grows to 40,192 at offset 32,001, copying one layer's 32,000-token buffer, and the standard request, 18,380 + 1,024 tokens, never grows from 20,480" {
    try testing.expectEqual(@as(u64, 32000), nextCapacity(32000, 32000));
    try testing.expectEqual(@as(u64, 40192), nextCapacity(32000, 32001));
    try testing.expectEqual(Growth{ .cap = 40192, .old = 32000 }, grownTo(32000, 32001));
    // Its decode ends at 19,404 tokens, and a verify writes at most the widest decode forward's rows past it.
    try testing.expectEqual(Growth{ .cap = 20480, .old = 0 }, grownTo(20480, 18380 + 1024 + head_margin));
    // A 1,120-token request decoding 1,024 tokens: 1,280 -> 1,792 -> 2,304, the last copy 1,792 tokens of one layer.
    try testing.expectEqual(Growth{ .cap = 2304, .old = 1792 }, grownTo(1280, 1120 + 1024));
}

test "RQ's rows: with the bill's 32,768-token KV term and O = 0 the plan gives the 214 prompt and 236 decode rows the bank opens with, at either end of the logged MiB" {
    for ([_]u64{ 0, mib - 1 }) |frac| {
        const t = servedTerms(frac);
        try testing.expectEqual(Rows{ .prompt = 214, .decode = 236 }, t.rows);
        const bill_term = t.kv_tokens * t.kv.perToken();
        try testing.expectEqual(@as(u64, 297_216), t.kv.perToken());
        try testing.expectEqual(Fit{ .rows = 214, .held = 214, .over = false }, rowsAt(t, .prompt, bill_term, 0));
        try testing.expectEqual(Fit{ .rows = 236, .held = 236, .over = false }, rowsAt(t, .decode, bill_term, 0));
    }
}

test "RQ's decode plan at H = 4,096 and O = 0: a 1,120-token request takes 259 decode rows, 18,380 tokens 241 and 31,652 tokens 227 (236 at the bill's 32,768-token KV term), at mlx-serve's capacities for its KV" {
    // P, the decode rows, the trunk's and the head's capacity the plan holds
    const cases = [_][4]u64{ .{ 1120, 259, 6144, 7680 }, .{ 18380, 241, 25600, 28928 }, .{ 31652, 227, 40192, 47872 } };
    for ([_]u64{ 0, mib - 1 }) |frac| {
        const t = servedTerms(frac);
        for (cases) |c| {
            const s = decodeStart(c[0]);
            const plan = planDecode(t, &s, .decode, c[0], 1, 0);
            try testing.expectEqual(c[1], plan.rows);
            try testing.expect(!plan.over);
            try testing.expectEqual(c[2], plan.cap);
            try testing.expectEqual(c[3], plan.head_cap);
        }
    }
    // The standard request's KV: the trunk at 25,600 tokens, the head at 28,928, and one layer's 20,480-token copy.
    const s = decodeStart(18380);
    try testing.expectEqual(@as(u64, 25600 * 296_064 + 28928 * 1_152 + 20480 * 32_896), planDecode(servedTerms(0), &s, .decode, 18380, 1, 0).kv);
}

test "RQ's plan holds its line at every prompt length up to the served maximum and every O: its rows, decode's window and the staging, its KV and O within the decode budget with the bill's KV term, and one row more past it; under the base it is the named over-plan result, at the base" {
    const t = servedTerms(0);
    const line = t.budget.decode + t.kv_tokens * t.kv.perToken();
    const fixed = t.bank.decode_windows + t.bank.staging;
    const others = [_]u64{ 0, 1 << 30, 4 << 30, 8 << 30 };
    var n_over: [others.len]usize = @splat(0);
    var p: u64 = 32;
    while (p <= t.context) : (p += 97) {
        for (others, &n_over) |other, *n| {
            const s = decodeStart(p);
            const plan = planDecode(t, &s, .decode, p, 1, other);
            if (plan.over) {
                n.* += 1;
                try testing.expectEqual(t.rows.prompt, plan.rows);
                try testing.expect(plan.held < t.rows.prompt);
                continue;
            }
            try testing.expect(plan.rows >= t.rows.prompt);
            try testing.expect(@as(u64, plan.rows) * t.bank.row_bytes + fixed + plan.kv + other <= line);
            if (plan.rows < t.bank.n_experts) try testing.expect(@as(u64, plan.rows + 1) * t.bank.row_bytes + fixed + plan.kv + other > line);
        }
    }
    // With D2's served maximum, 32,768 tokens: over plan at 4 GiB of O at 32,333 and 32,430 tokens (213 rows' room; their
    // decode plans bill decode's next grow from 32,512 to 40,704), and at 8 GiB from 21,566 tokens (209 rows' room) to
    // the last length swept: 116 of 338 lengths (to 32,000 tokens, 108 of 330, and none at 4 GiB).
    try testing.expectEqualSlices(usize, &.{ 0, 0, 2, 116 }, &n_over);
    // 10 GiB of O (another slot's KV, or one not yet freed) leaves the standard request 209 rows: the base, over plan.
    const s = decodeStart(18380);
    const over = planDecode(t, &s, .decode, 18380, 1, 10 << 30);
    try testing.expect(over.over);
    try testing.expectEqual(@as(u32, 214), over.rows);
    try testing.expectEqual(@as(u64, 209), over.held);
    // C5's prompt plan at the standard request's start: its KV at its capacity, its pass's working set (4,802,489,944
    // B: src/bill.zig `passWork` at 18,380 tokens, whose test pins it) and O past the rerank head. The base fits with
    // O = 0 (235 rows' room) and with 4 GiB (223); 7 GiB is over (213). Before C5 the plan counted the growth's copy
    // beside the pass's term and the rerank head in O: 223 at O = 0, and over at 4 GiB (210).
    const sp: Slot = .{ .p = 18380, .prompt = .{ .cap = 20480, .old = 16384 } };
    const fits = planPrompt(t, &sp, 0, 4_802_489_944);
    try testing.expectEqual(@as(u32, 214), fits.rows);
    try testing.expectEqual(@as(u64, 235), fits.held);
    try testing.expect(!fits.over);
    try testing.expectEqual(@as(u64, 223), planPrompt(t, &sp, 4 << 30, 4_802_489_944).held);
    const tight = planPrompt(t, &sp, 7 << 30, 4_802_489_944);
    try testing.expect(tight.over);
    try testing.expectEqual(@as(u64, 213), tight.held);
}

test "RQ's crossing check fires exactly when a forward's position plus its rows passes the plan's capacity" {
    try testing.expect(!crosses(25598, 2, 25600));
    try testing.expect(crosses(25599, 2, 25600));
    try testing.expect(!crosses(25599, 1, 25600));
    try testing.expect(crosses(25600, 1, 25600));
    var pos: u64 = 25590;
    while (pos < 25601) : (pos += 1) {
        var n: u64 = 1;
        while (n <= 8) : (n += 1) try testing.expectEqual(pos + n > 25600, crosses(pos, n, 25600));
    }
}

// ── D2's tests ──

/// The ends of a forward's appends as `Appends` gives them, into `buf`.
fn appendEnds(buf: []u64, pos0: u64, ends: []const u64, lane: u64) []const u64 {
    var it = Appends.init(pos0, ends, lane);
    var k: usize = 0;
    while (it.next()) |e| : (k += 1) buf[k] = e;
    return buf[0..k];
}

/// The same appends as plugin.zig `spanMajorOr` and table_forward.zig `hiddenSpans` make them, from `groupStart`: a span
/// of `lane` rows or fewer alone, each run of longer spans in C4's segments (the whole run's plan,
/// src/ling_segments.zig `Plan`), each segment one call, whose spans from `groupStart` on append once, at the
/// segment's last end.
fn segmentEnds(buf: []u64, pos0: u64, ends: []const u64, lane: u64) []const u64 {
    var starts: [64]u32 = undefined;
    var rows: [ling_segments.max_spans]u32 = undefined;
    var k: usize = 0;
    var j: usize = 0;
    while (j < ends.len) {
        const s0 = if (j == 0) pos0 else ends[j - 1];
        if (ends[j] - s0 <= lane) {
            buf[k] = ends[j];
            k += 1;
            j += 1;
            continue;
        }
        var e = j + 1;
        while (e < ends.len and ends[e] - ends[e - 1] > lane) : (e += 1) {}
        for (j..e, 0..) |sj, i| rows[i] = @intCast(ends[sj] - (if (sj == 0) pos0 else ends[sj - 1]));
        var p: ling_segments.Plan = undefined;
        p.init(rows[0 .. e - j], ling_segments.segment_rows) catch unreachable;
        while (p.next()) |sg| {
            const a = j + sg.first;
            const b = j + sg.end;
            for (a..b, 0..) |sj, i| starts[i] = @intCast(if (sj == 0) pos0 else ends[sj - 1]);
            const g = a + groupStart(starts[0 .. b - a], d2Start(pos0));
            for (a..b) |sj| if (sj < g or sj + 1 == b) {
                buf[k] = ends[sj];
                k += 1;
            };
        }
        j = e;
    }
    return buf[0..k];
}

test "D2: a whole prompt's spans from 24,576 on append their keys and values in one call, so each MLA layer's KV grows once for them: 25,600 and 32,000 tokens as an append a span gives up to a 32,000-token prompt, then 32,256, 32,512 and 32,768 up to the bill's 32,768-token context, where an append a span grew it to 40,192" {
    // P, D2's capacity and its last growth's old capacity, an append a span's capacity
    const cases = [_][4]u64{
        .{ 24576, 25600, 20480, 25600 }, .{ 25118, 25600, 20480, 25600 }, .{ 25119, 25600, 20480, 25600 },
        .{ 31652, 32000, 25600, 32000 }, .{ 32000, 32000, 25600, 32000 }, .{ 32001, 32256, 25600, 40192 },
        .{ 32200, 32256, 25600, 40192 }, .{ 32256, 32256, 25600, 40192 }, .{ 32257, 32512, 25600, 40192 },
        .{ 32512, 32512, 25600, 40192 }, .{ 32513, 32768, 25600, 40192 }, .{ 32767, 32768, 25600, 40192 },
        .{ 32768, 32768, 25600, 40192 }, .{ 32769, 33024, 25600, 40192 },
    };
    var buf: [16]u64 = undefined;
    var abuf: [16]u64 = undefined;
    for (cases) |c| {
        const ends = spanEnds(&buf, c[0]);
        try testing.expectEqual(Growth{ .cap = c[1], .old = c[2] }, replayAppends(0, 0, ends, 6));
        try testing.expectEqual(c[1], wholePromptCapacity(c[0]));
        try testing.expectEqual(c[3], replay(0, ends).cap);
    }
    // 32,200 tokens: the three spans from 24,576 (4,096, 3,497 and 31 rows) in one append
    const e200 = spanEnds(&buf, 32200);
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32169, 32200 }, e200);
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 32200 }, appendEnds(&abuf, 0, e200, 6));
    // 25,118 tokens: one span from 24,576 (the 31-row last), so no group; 25,119: two, in one append
    const e118 = spanEnds(&buf, 25118);
    try testing.expectEqualSlices(u64, e118, appendEnds(&abuf, 0, e118, 6));
    const e119 = spanEnds(&buf, 25119);
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 25088, 25119 }, e119);
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 25119 }, appendEnds(&abuf, 0, e119, 6));
}

test "D2 leaves a whole prompt of up to 25,118 tokens on an append a span, and gives every whole prompt up to 32,000 tokens an append a span's KV growth: today's path and today's KV" {
    var buf: [16]u64 = undefined;
    var abuf: [16]u64 = undefined;
    var n: u64 = 1;
    while (n <= 32000) : (n += 1) {
        const ends = spanEnds(&buf, n);
        if (n <= 25118) try testing.expectEqualSlices(u64, ends, appendEnds(&abuf, 0, ends, 6));
        try testing.expectEqual(replay(0, ends), replayAppends(0, 0, ends, 6));
        try testing.expectEqual(replay(0, ends).cap, wholePromptCapacity(n));
    }
}

test "D2's position, 24,576, is the one 4,096-token chunk boundary from which one append holds every whole prompt up to the bill's 32,768-token context within that KV term and keeps an append a span's KV for every whole prompt up to 32,000 tokens" {
    var buf: [16]u64 = undefined;
    try testing.expectEqual(@as(u64, 24576), d2_from);
    try testing.expectEqual(d2_from, d2Start(0));
    var k: u64 = 1;
    while (k <= 8) : (k += 1) {
        const from = k * prefill_chunk;
        var holds = true;
        var keeps = true;
        var n: u64 = 1;
        while (n <= 32768) : (n += 1) {
            const c = wholePromptCapacityFrom(n, from);
            holds = holds and c <= 32768;
            if (n <= 32000) keeps = keeps and c == replay(0, spanEnds(&buf, n)).cap;
        }
        try testing.expectEqual(from == d2_from, holds and keeps);
    }
    // from 20,480 a 28,000-token prompt gets 28,160 tokens, not 32,000; from 28,672 a 32,001-token one grows to 40,192
    try testing.expectEqual(@as(u64, 28160), wholePromptCapacityFrom(28000, 20480));
    try testing.expectEqual(@as(u64, 40192), wholePromptCapacityFrom(32001, 28672));
}

test "D2's groups in a forward's spans: never its first span (the first append after a restored prefix stays its own), split where a decode lane's span is forwarded alone, none for one span; Appends gives the appends that plugin.zig's runs and table_forward.zig's groupStart make" {
    var abuf: [16]u64 = undefined;
    var sbuf: [16]u64 = undefined;
    // a restored prefix of 4,096 tokens, then the host's spans to 32,700: the three spans from 24,576 append once
    const r4 = [_]u64{ 8192, 12288, 16384, 20480, 24576, 28672, 32669, 32700 };
    try testing.expectEqualSlices(u64, &.{ 8192, 12288, 16384, 20480, 24576, 32700 }, appendEnds(&abuf, 4096, &r4, 6));
    try testing.expectEqual(Growth{ .cap = 32768, .old = 25600 }, replayAppends(4096, 4096, &r4, 6));
    try testing.expectEqual(@as(u64, 40192), replay(4096, &r4).cap);
    // a restored prefix of 24,576 tokens: its first span appends alone, the two after it once
    const r24 = [_]u64{ 28672, 32669, 32700 };
    try testing.expectEqual(@as(u64, 24577), d2Start(24576));
    try testing.expectEqualSlices(u64, &.{ 28672, 32700 }, appendEnds(&abuf, 24576, &r24, 6));
    try testing.expectEqual(Growth{ .cap = 38400, .old = 30720 }, replayAppends(24576, 24576, &r24, 6));
    // a 3-row span (a decode lane's, forwarded alone) splits the run: 24,576's span appends alone, the two after it once
    const split = [_]u64{ 4096, 8192, 12288, 16384, 20480, 24576, 28672, 28675, 32700, 32731 };
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 28672, 28675, 32731 }, appendEnds(&abuf, 0, &split, 6));
    // one span from 24,576: no group
    const one = [_]u64{ 4096, 8192, 12288, 16384, 20480, 24576, 24607 };
    try testing.expectEqualSlices(u64, &one, appendEnds(&abuf, 0, &one, 6));
    // groupStart on table_forward.zig's span starts: the first at d2Start or later, when two or more are
    try testing.expectEqual(@as(usize, 6), groupStart(&.{ 0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32169 }, 24576));
    try testing.expectEqual(@as(usize, 7), groupStart(&.{ 0, 4096, 8192, 12288, 16384, 20480, 24576 }, 24576));
    try testing.expectEqual(@as(usize, 1), groupStart(&.{ 28000, 32096, 32669 }, d2Start(28000)));
    // the two agree on these forwards and on whole prompts from 0
    inline for (.{ .{ 4096, &r4 }, .{ 24576, &r24 }, .{ 0, &split }, .{ 0, &one } }) |c| try testing.expectEqualSlices(u64, segmentEnds(&sbuf, c[0], c[1], 6), appendEnds(&abuf, c[0], c[1], 6));
    var buf: [16]u64 = undefined;
    var n: u64 = 1;
    while (n <= 32768) : (n += 97) {
        const ends = spanEnds(&buf, n);
        try testing.expectEqualSlices(u64, segmentEnds(&sbuf, 0, ends, 6), appendEnds(&abuf, 0, ends, 6));
    }
}

test "RQ's refusal with D2 at the default context: every whole prompt up to 32,768 tokens is served, so every prompt mlx-serve admits at --ctx-size 32768, and a prompt forward whose KV would pass the plan's 32,768 tokens is refused by name before any append" {
    var buf: [320]u8 = undefined;
    const t = servedTerms(0);
    try testing.expectEqual(@as(u64, 32768), promptCapacityMax(32768));
    try testing.expectEqual(t.kv_tokens, promptCapacityMax(t.context));
    try testing.expect(promptRefusal(&buf, t, 0, 32768, wholePromptCapacity(32768)) == null);
    const why = promptRefusal(&buf, t, 0, 32769, wholePromptCapacity(32769)) orelse return error.TestExpectedRefusal;
    try testing.expect(std.mem.startsWith(u8, why, refusal_name ++ " 32768 tokens"));
    try testing.expect(std.mem.indexOf(u8, why, "a 32769-row prompt forward from position 0 grows mlx-serve's KV to 33024 tokens") != null);
    try testing.expect(std.mem.indexOf(u8, why, "whole prompts from position 0 up to the 32768-token context are served") != null);
    // a forward from a restored prefix of 24,576 tokens whose replay passes 32,768 (its first span appends alone)
    const ends = [_]u64{ 28672, 32669, 32700 };
    const g = replayAppends(24576, 24576, &ends, 6);
    try testing.expect(promptRefusal(&buf, t, 24576, 32700 - 24576, g.cap) != null);
    var p: u64 = 1;
    while (p <= 40000) : (p += 1) try testing.expectEqual(p > 32768, promptRefusal(&buf, t, 0, p, wholePromptCapacity(p)) != null);
}

// ── C1's tests ──

test "C1: the KV the bill holds for a served context is the most mlx-serve allocates for any whole prompt up to it: 32,768 at the default context (D2's term); more than the context where a shorter prompt's growth passes it (20,224 at 16,384; 1,792 at 1,312, where 1,290 tokens get 1,792 and 1,312 get 1,536); at the sweep's contexts 21,248 / 21,248 / 25,600 / 26,368 / 33,280 / 50,176 / 83,712 / 151,040 tokens" {
    try testing.expectEqual(@as(u64, 1792), wholePromptCapacity(1290));
    try testing.expectEqual(@as(u64, 1536), wholePromptCapacity(1312));
    try testing.expectEqual(@as(u64, 1792), promptCapacityMax(1312));
    try testing.expectEqual(@as(u64, 20224), promptCapacityMax(16384));
    try testing.expectEqual(@as(u64, 32768), promptCapacityMax(32768));
    // ctx = P + 16,384 for the sweep's P = 1,058 / 2,093 / 4,181 / 8,397 / 16,800 / 33,591 / 67,310 / 134,615
    const sweep = [_][2]u64{ .{ 17442, 21248 }, .{ 18477, 21248 }, .{ 20565, 25600 }, .{ 24781, 26368 }, .{ 33184, 33280 }, .{ 49975, 50176 }, .{ 83694, 83712 }, .{ 150999, 151040 } };
    for (sweep) |c| try testing.expectEqual(c[1], promptCapacityMax(c[0]));
}

test "C1: every whole prompt up to a served context keeps mlx-serve's KV within what the bill holds for that context, and some prompt reaches it; a prompt's capacity is never more than one growth step (8,192 tokens, rounded up to 256) past it, so the lengths promptCapacityMax reads miss none" {
    for ([_]u64{ 1312, 17442, 24781, 33184, 49975 }) |ctx| {
        const most = promptCapacityMax(ctx);
        var reached = false;
        var n: u64 = 1;
        while (n <= ctx) : (n += 1) {
            const c = wholePromptCapacity(n);
            try testing.expect(c <= most);
            reached = reached or c == most;
        }
        try testing.expect(reached);
    }
    var n: u64 = 1;
    while (n <= 160000) : (n += 1) try testing.expect(wholePromptCapacity(n) <= n + max_grow_tokens + chunk_step);
}

test "C1: the checkpoints a whole prompt names: none at one token, 1 from 2 tokens (the backoff point), 5 at 18,380, 8 at 31,652 and at 32,768 (one fewer than the 9 billed before C1), and 9 / 13 / 17 / 21 / 33 / 37 at 33,591 / 49,975 / 67,310 / 83,694 / 134,615 / 150,999 tokens; never fewer at a longer prompt" {
    const cases = [_][2]u64{ .{ 1, 0 }, .{ 2, 1 }, .{ 31, 1 }, .{ 1120, 1 }, .{ 18380, 5 }, .{ 31652, 8 }, .{ 32768, 8 }, .{ 33591, 9 }, .{ 49975, 13 }, .{ 67310, 17 }, .{ 83694, 21 }, .{ 134615, 33 }, .{ 150999, 37 } };
    for (cases) |c| try testing.expectEqual(c[1], checkpointsOf(c[0]));
    var prev: u64 = 0;
    var n: u64 = 1;
    while (n <= 160000) : (n += 1) {
        const k = checkpointsOf(n);
        try testing.expect(k >= prev);
        prev = k;
    }
}

test "C1: RQ at a served context past 32,768 (ctx 49,975, the sweep's 32K point): every whole prompt from position 0 up to it is served (49,975 tokens take 50,176 tokens of KV, the plan's), a forward whose KV passes 50,176 is refused by name, and the decode plan ends 4,096 past the position or at the served context" {
    var buf: [320]u8 = undefined;
    var t = servedTerms(0);
    t.context = 49975;
    t.kv_tokens = promptCapacityMax(49975);
    try testing.expectEqual(@as(u64, 50176), t.kv_tokens);
    try testing.expectEqual(@as(u64, 50176), wholePromptCapacity(49975));
    try testing.expect(promptRefusal(&buf, t, 0, 33591, wholePromptCapacity(33591)) == null);
    var p: u64 = 49975 - 9000;
    while (p <= 49975) : (p += 1) try testing.expect(promptRefusal(&buf, t, 0, p, wholePromptCapacity(p)) == null);
    const why = promptRefusal(&buf, t, 0, 50177, wholePromptCapacity(50177)) orelse return error.TestExpectedRefusal;
    try testing.expect(std.mem.indexOf(u8, why, "passes the plan's 50176 tokens") != null);
    try testing.expect(std.mem.indexOf(u8, why, "up to the 49975-token context are served") != null);
    try testing.expectEqual(@as(u64, 37687), decodeEnd(33591, 1, t.context));
    try testing.expectEqual(@as(u64, 49975), decodeEnd(47000, 1, t.context));
}

test "RQ's decode plans after D2's longer prompts at H = 4,096 and O = 0: after 32,200 and 32,512 tokens they bill decode's next grow (32,256 to 40,448, 32,512 to 40,704) at 226 rows; after 32,767 tokens the trunk holds the context, 32,768, at 236 rows, and a forward past it trims to a plan that bills 32,768 to 40,960" {
    for ([_]u64{ 0, mib - 1 }) |frac| {
        const t = servedTerms(frac);
        // P, the trunk's capacity the plan holds and its old one, the head's capacity, the rows
        const cases = [_][5]u64{ .{ 32200, 40448, 32256, 48640, 226 }, .{ 32512, 40704, 32512, 48896, 226 }, .{ 32767, 32768, 0, 40960, 236 } };
        for (cases) |c| {
            const s = decodeStart(c[0]);
            const plan = planDecode(t, &s, .decode, c[0], 1, 0);
            try testing.expectEqual(c[1], plan.cap);
            try testing.expectEqual(c[3], plan.head_cap);
            try testing.expectEqual(c[4], @as(u64, plan.rows));
            try testing.expect(!plan.over);
            // the growth copy billed: one layer's old buffer, or the head's when larger
            const tg = grownTo(wholePromptCapacity(c[0]), decodeEnd(c[0], 1, t.context));
            try testing.expectEqual(c[2], tg.old);
        }
        // after 32,767 tokens a forward past the context (a verify's 2 rows at 32,767) crosses the plan, and its trim
        // bills the trunk at 40,960 with one layer's 32,768-token copy
        const s = decodeStart(32767);
        try testing.expect(crosses(32767, 2, 32768));
        const trim = planDecode(t, &s, .trim, 32767, 2, 0);
        try testing.expectEqual(@as(u64, 40960), trim.cap);
        try testing.expectEqual(@as(u32, 226), trim.rows);
        try testing.expectEqual(@as(u64, 40960 * 296_064 + 49152 * 1_152 + 32768 * 32_896), trim.kv);
    }
    // the KV the 32,200-token request's decode plan bills: the trunk at 40,448, the head at 48,640, one layer's 32,256
    const s = decodeStart(32200);
    try testing.expectEqual(@as(u64, 40448 * 296_064 + 48640 * 1_152 + 32256 * 32_896), planDecode(servedTerms(0), &s, .decode, 32200, 1, 0).kv);
}

test "RQ's slot table: each slot keeps its own request's capacities, a request in a slot starts its entry over, and a forward of a slot after another slot's forwards is flagged interleaved" {
    var t: SlotTable = .{};
    t.begin(0, 0).trunk_cap = 20480;
    try testing.expect(!t.touch(0));
    t.begin(1, 0).trunk_cap = 1280;
    // slot 0's request resumes after slot 1's prompt: interleaved, once
    try testing.expect(t.touch(0));
    try testing.expect(!t.touch(0));
    try testing.expectEqual(@as(u64, 20480), t.get(0).?.trunk_cap);
    try testing.expectEqual(@as(u64, 1280), t.get(1).?.trunk_cap);
    // a new request in slot 0 from a restored prefix of 512 tokens: both caches start at it
    const s = t.begin(0, 512);
    try testing.expectEqual(@as(u64, 512), s.trunk_cap);
    try testing.expectEqual(@as(u64, 512), s.head_cap);
    try testing.expect(t.get(2) == null);
    // past `max_slots` slots the least recently used goes
    for (2..2 + max_slots) |i| _ = t.begin(@intCast(i), 0);
    try testing.expect(t.get(1) == null);
    try testing.expect(t.get(2 + max_slots - 1) != null);
}

// ── C4's tests (src/ling_segments.zig: the layer-major pass in segments, the MTP history in forwards) ──

test "C4: a whole prompt's appends at one MLA layer end where its segments end, D2's group a segment: 40,000 tokens at 40,000 (one segment, the last call joined: 40,192 tokens of KV), 67,310 at 32,768, 49,152 and 67,310 (67,328), 134,615 at 32,768, 65,536, 98,304 and 134,615 (134,656); 32,769 and 33,591 tokens as D2 had them; a restored prefix's and a split run's; Appends gives spanMajorOr's segments' appends on whole prompts to 140,000 tokens" {
    var buf: [64]u64 = undefined;
    var abuf: [64]u64 = undefined;
    var sbuf: [64]u64 = undefined;
    const head = [_]u64{ 4096, 8192, 12288, 16384, 20480, 24576 };
    // P, the appends past 24,576, the capacity and its last growth's old capacity
    const C = struct { p: u64, past: []const u64, cap: u64, old: u64 };
    const cases = [_]C{
        .{ .p = 40000, .past = &.{40000}, .cap = 40192, .old = 25600 },
        .{ .p = 67310, .past = &.{ 32768, 49152, 67310 }, .cap = 67328, .old = 49152 },
        .{ .p = 134615, .past = &.{ 32768, 65536, 98304, 134615 }, .cap = 134656, .old = 98304 },
    };
    for (cases) |c| {
        const ends = spanEnds(&buf, c.p);
        const got = appendEnds(&abuf, 0, ends, 6);
        try testing.expectEqualSlices(u64, &head, got[0..6]);
        try testing.expectEqualSlices(u64, c.past, got[6..]);
        try testing.expectEqual(Growth{ .cap = c.cap, .old = c.old }, replayAppends(0, 0, ends, 6));
        try testing.expectEqual(c.cap, wholePromptCapacity(c.p));
    }
    // two segments from 32,769 tokens, the first ending at 16,384: D2's one group from 24,576, as before
    try testing.expectEqual(@as(u64, 33024), wholePromptCapacity(32769));
    try testing.expectEqual(@as(u64, 33792), wholePromptCapacity(33591));
    // a restored prefix of 4,096 tokens, then the 64K prompt's spans: its segments from 4,096
    var r4: [64]u64 = undefined;
    const e64 = spanEnds(&buf, 67310);
    for (e64, 0..) |e, i| r4[i] = 4096 + e;
    const r4e = r4[0..e64.len];
    try testing.expectEqualSlices(u64, &.{ 8192, 12288, 16384, 20480, 24576, 36864, 53248, 71406 }, appendEnds(&abuf, 4096, r4e, 6));
    try testing.expectEqual(Growth{ .cap = 71424, .old = 53248 }, replayAppends(4096, 4096, r4e, 6));
    try testing.expectEqualSlices(u64, segmentEnds(&sbuf, 4096, r4e, 6), appendEnds(&abuf, 4096, r4e, 6));
    // a 3-row span (a decode lane's, forwarded alone) at 40,960 splits the 64K prompt's run in two
    const split = [_]u64{ 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 40963, 45059, 49155, 53251, 57347, 61443, 65539, 67282, 67313 };
    try testing.expectEqualSlices(u64, &.{ 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 40963, 67313 }, appendEnds(&abuf, 0, &split, 6));
    try testing.expectEqualSlices(u64, segmentEnds(&sbuf, 0, &split, 6), appendEnds(&abuf, 0, &split, 6));
    var n: u64 = 1;
    while (n <= 140_000) : (n += 97) {
        const ends = spanEnds(&buf, n);
        try testing.expectEqualSlices(u64, segmentEnds(&sbuf, 0, ends, 6), appendEnds(&abuf, 0, ends, 6));
    }
}

test "C4's segment tail: a whole prompt's segmented appends grow mlx-serve's KV to the single pass's capacity (D2's one group from 24,576) for every whole prompt from 2 to 150,999 tokens; without the tail rule 32,800 tokens grew it to 40,192 where the single pass gives 33,024" {
    var buf: [64]u64 = undefined;
    var n: u64 = 2;
    while (n <= 150_999) : (n += 1) {
        const ends = spanEnds(&buf, n);
        // the single pass: one segment (no bound), as before C4
        var one = Appends.initFrom(0, ends, 0, d2_from, std.math.maxInt(u64));
        var cap: u64 = 0;
        while (one.next()) |e| cap = nextCapacity(cap, e);
        try testing.expectEqual(cap, wholePromptCapacity(n));
    }
    try testing.expectEqual(@as(u64, 33024), wholePromptCapacity(32800));
    try testing.expectEqual(@as(u64, 33280), wholePromptCapacity(33184));
    try testing.expectEqual(@as(u64, 134656), wholePromptCapacity(134615));
}

test "C4: the MTP head's history in RQ's replay: one append for a prompt of 32,768 tokens or fewer (18,380: 18,432 tokens of the head's KV; 32,768: 32,768), 4,096-row forwards past that (33,591: 40,192 from 32,000; 67,310: 72,960 from 64,768; 134,615: 138,496 from 130,304), from a restored prefix too; the prompt plan reports the head at that capacity" {
    // from, r, P, the capacity and its last growth's old capacity
    const cases = [_][5]u64{
        .{ 0, 0, 18380, 18432, 0 },          .{ 0, 0, 32768, 32768, 0 },           .{ 0, 0, 33591, 40192, 32000 },
        .{ 0, 0, 67310, 72960, 64768 },      .{ 0, 0, 134615, 138496, 130304 },    .{ 4096, 4096, 71406, 72960, 64768 },
        .{ 512, 512, 30000, 30208, 512 },
    };
    for (cases) |c| try testing.expectEqual(Growth{ .cap = c[3], .old = c[4] }, replayHistory(c[0], c[1], c[2]));
    // the 64K point's prompt plan: the trunk at its segments' 67,328, the head at 72,960 (reported, not in C5's room)
    const s: Slot = .{ .p = 67310, .prompt = .{ .cap = 67328, .old = 49152 } };
    try testing.expectEqual(@as(u64, 72960), planPrompt(servedTerms(0), &s, 0, 0).head_cap);
}
