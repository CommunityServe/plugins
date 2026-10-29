//! copied_from: src/engines/mlx/spec/round_cost.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Measured spec round-cost table: what a round at draft width w costs on
//! THIS model, on THIS machine, at the live context length.
//!
//! Every width decision in spec decode is throughput = accepted tokens over
//! round wall time, and every earlier cost source (the hand-typed chip rows,
//! the fitted EV surfaces, the boot ladder) measured only part of that, in
//! one regime, and shipped it for all of them. This table measures the
//! whole fraction from the rounds the server actually runs. Pure data, no
//! MLX — the generator feeds it and reads it, the fitted surface is the
//! cold-start prior.
//!
//! Width = drafts per round (MTP depth m; DFlash block_size - 1; 0 = serial).
//! Buckets = KV length at the round: <2k, 2-4k, 4-8k, 8-16k, 16-32k, 32k+.
//! Each cell holds an EMA of round ms AND an EMA of emitted tokens — cost is
//! never stored without the tokens it bought.
//!
//! Beside the width grid sits one more row: `serial`, the measured ms of a plain decode
//! token per bucket. Kept out of the width grid (a serial tick is not a round), it answers
//! the one question no width can: is speculation worth running here at all?
const std = @import("std");
const env = @import("../../../env.zig");
const crt = @import("../../../sdk/util/crt.zig");
const os_mutex = @import("../../../sdk/util/os_mutex.zig");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// Drafts per round the table covers (MTP depth <= 8, a DFlash block up to 16); index 0 is serial.
pub const MAX_WIDTH: u32 = 16;
/// KV buckets. The long grid splits the old unbounded `32k+` cell at 64k/128k/256k.
pub const N_BUCKETS: usize = 9;
const BUCKET_EDGES = [_]u32{ 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
pub const BUCKET_NAMES = [N_BUCKETS][]const u8{
    "<2k",    "2-4k",    "4-8k",     "8-16k", "16-32k",
    "32-64k", "64-128k", "128-256k", "256k+",
};

/// The label for bucket `b` under `layout`: the grids disagree about bucket 5 (`32k+` on legacy).
pub fn bucketName(layout: Layout, b: usize) []const u8 {
    const n = nBuckets(layout);
    if (b + 1 == n and n < N_BUCKETS) return "32k+";
    return BUCKET_NAMES[b];
}

/// Which bucket grid a table speaks. The split, the serial row and the adaptive switch were
/// measured on qwen4_exp; every other arch keeps the six-bucket grid 26.9.1 wrote and so
/// reads its own persisted file (a store-version bump is a cold table, and a cold table
/// plans every round as a two-chunk round until it matures: -25% decode on the 27B at 8k).
pub const Layout = enum {
    /// Six buckets, top one unbounded at 32k. Store version 1, no serial row.
    legacy,
    /// Nine buckets (64k/128k/256k edges) + the serial row. Store version 3.
    long,
};

/// The one resolver for a model's layout; the layout decides the store version and so
/// which persisted file a model reads. `anytype`: this module imports nothing but std.
pub fn layoutFor(config: anytype) Layout {
    return if (config.longCtxGated()) .long else .legacy;
}

/// Buckets the layout uses; cells past it are never written and never active.
pub fn nBuckets(layout: Layout) usize {
    return switch (layout) {
        .legacy => 6,
        .long => N_BUCKETS,
    };
}

/// EMA weight. MIN is wrong here (thermal soak makes the early rounds the
/// fast ones); an EMA tracks the live machine.
pub const BETA: f32 = 0.10;
/// A cell unsampled for this many offered rounds takes its next sample at
/// RESEED_WEIGHT instead of BETA: whatever it held was measured in another
/// regime (thermal, context). It keeps its sample count — a cell that is
/// only ever re-sampled by trials (period up to EXPLORE_PERIOD_MAX) would
/// otherwise lose trust on every trial, reopen the horizon, and cycle.
pub const RESEED_GAP: u32 = 64;
pub const RESEED_WEIGHT: f32 = 0.5;
/// Measured widths a bucket needs before the table replaces the prior: ONE
/// trusted width anchors the prior's shape (scale) and lets raw cells floor
/// it — two were required first, and with w5 settled as clearly worse
/// after one sample the bucket never reached two, so the prior kept
/// planning the 4 -> 5 extension the table had already priced.
pub const MIN_WIDTHS: u32 = 1;
/// Samples a cell needs before it COUNTS as measured. A one-sample cell is
/// the seed (a legacy-controller round at the warmup boundary seeded a w2
/// cell on the M4 base 9B, activated the table on {w2, w4} and anchored it
/// at 2 — the plan then read w3 as a bargain and lost 6.6%).
pub const MIN_SAMPLES: u32 = 3;
/// Serial probes a bucket may attempt per process before it gives up.
pub const MAX_SERIAL_PROBES: u8 = 3;
/// A width whose FIRST sample already reads this much worse per token than
/// a trusted reference is settled as worse: the plan only needs "not
/// better", and every further trial block of it is a 3-4% hit on the
/// request that carries it (M1 Pro 27B: w5 read 94.7 against w4's 71.2 on
/// sample 1 and never moved; at 3-samples-to-trust that cost -7.1%).
pub const CLEARLY_WORSE: f32 = 0.20;

/// The bucket `kv_len` falls in under `layout`; the legacy grid folds everything past 32k into its last bucket.
pub fn bucketForLayout(kv_len: u32, layout: Layout) usize {
    const edges = nBuckets(layout) - 1;
    for (BUCKET_EDGES[0..edges], 0..) |edge, i| {
        if (kv_len < edge) return i;
    }
    return edges;
}

pub const Cell = struct {
    ms: f32 = 0,
    tok: f32 = 0,
    n: u32 = 0,
    /// `Table.seq` at the last fold — the reseed clock.
    last_seen: u32 = 0,

    pub fn msPerTok(self: Cell) ?f32 {
        if (self.n == 0 or self.tok <= 0) return null;
        return self.ms / self.tok;
    }
};

pub const Verdict = enum { folded, reseeded, contended, transition, bad_sample, out_of_range, implausible };

/// A width-w round is one forward of w+1 rows; it cannot cost more than this
/// times the width-(w-1) round (measured steps run under 1.25x). Anything past
/// it carried foreign work between round ends and would poison the planner.
pub const IMPLAUSIBLE_STEP: f32 = 1.5;
/// The mirror bound for the persisted sweep: a narrower round is strictly less work than a
/// wider one, so a stored cell past this multiple of its nearest trusted WIDER cell carried
/// foreign work. Width 1 has no narrower neighbour and only this bound reaches it (#382).
/// Healthy tables reach 1.08x (M4 Max); a poisoned width-1 cell reads 1.6x to 2.1x.
pub const IMPLAUSIBLE_WIDER: f32 = 1.25;
/// A sample past this multiple of a mature cell's own value is the machine, not the round
/// (another process on the GPU, #369). A stale cell is a regime change and reseeds instead.
pub const SELF_SPIKE: f32 = 3.0;

pub const Table = struct {
    /// Set once at load from the arch, never from a request.
    layout: Layout = .legacy,
    cells: [MAX_WIDTH + 1][N_BUCKETS]Cell = @splat(@splat(.{})),
    /// Measured ms per plain serial decode token, per bucket (`tok` is 1 per sample).
    serial: [N_BUCKETS]Cell = @splat(.{}),
    /// Measured ms of a prompt-lookup round (`mtp_lookup`) per draft count and bucket. Its
    /// own row so lookup rounds never reach the width grid, and runtime only: a stored row
    /// would bump STORE_VERSION and boot every existing table cold.
    lookup: [MAX_WIDTH + 1][N_BUCKETS]Cell = @splat(@splat(.{})),
    /// The lookup row's reseed clock.
    lookup_seq: u32 = 0,
    /// A draft count whose verify shape has run once this process. Spec-warmup compiles MTP
    /// widths only, so the first lookup round at a count pays its kernel compile.
    lookup_compiled: [MAX_WIDTH + 1]bool = @splat(false),
    /// Serial probes attempted per bucket this process. A count, not a flag: an interrupted
    /// probe must not burn the bucket's only chance. Runtime only, never serialized.
    serial_probes: [N_BUCKETS]u8 = @splat(0),
    /// Samples offered (accepted or not): the width grid's reseed clock.
    seq: u32 = 0,
    /// The serial row's own reseed clock: a serial tick is a token, a width sample is a round.
    serial_seq: u32 = 0,
    folded: u32 = 0,
    dropped_transition: u32 = 0,
    dropped_contended: u32 = 0,
    dropped_bad: u32 = 0,
    dropped_implausible: u32 = 0,
    /// The serial row's own fold/drop counters, for the same reason.
    serial_folded: u32 = 0,
    serial_dropped_transition: u32 = 0,
    serial_dropped_contended: u32 = 0,
    serial_dropped_bad: u32 = 0,
    /// One-shot: the first plan that read the table instead of the prior.
    first_use_logged: bool = false,
    /// `folded` at the last store (persistence writes only when it moved).
    stored_at: u32 = 0,
    /// Width cells restored from disk at load (diagnostics).
    restored: u32 = 0,
    /// Serial cells restored from disk at load (diagnostics).
    restored_serial: u32 = 0,
    /// Persisted width cells dropped at load for failing the step bound.
    restored_dropped: u32 = 0,

    /// Feed one realized round. `solo` = this was the only decoding stream
    /// (contention only ever ADDS time, so a busy server stops teaching the
    /// table rather than teaching it a lie); `transition` = the width
    /// differs from the previous round (the width change is a one-off cost
    /// that read the minority shape 5-7% slow in Phase 1).
    pub fn observe(self: *Table, width: u32, kv_len: u32, ms: f32, tokens: f32, solo: bool, transition: bool) Verdict {
        self.seq +%= 1;
        if (width > MAX_WIDTH) return .out_of_range;
        if (!std.math.isFinite(ms) or ms <= 0 or !(tokens > 0)) {
            self.dropped_bad += 1;
            return .bad_sample;
        }
        if (!solo) {
            self.dropped_contended += 1;
            return .contended;
        }
        if (transition) {
            self.dropped_transition += 1;
            return .transition;
        }
        const bucket = self.bucketOf(kv_len);
        if (self.stepImplausible(width, bucket, ms) or selfSpike(self.cells[width][bucket], ms, self.seq)) {
            self.dropped_implausible += 1;
            return .implausible;
        }
        self.folded += 1;
        const verdict = foldInto(&self.cells[width][bucket], ms, tokens, self.seq);
        if (self.cells[width][bucket].n == MIN_SAMPLES) self.dropped_implausible += self.dropWiderAgainst(width, bucket);
        return verdict;
    }

    // A cold boot folds the first wide rounds while every narrower cell is still untrusted, and
    // nothing else re-reads a cell once its narrower neighbour matures: re-apply the step bound
    // to the wider cells the moment `width` becomes trusted.
    fn dropWiderAgainst(self: *Table, width: u32, bucket: usize) u32 {
        var dropped: u32 = 0;
        var w: u32 = width + 1;
        while (w <= MAX_WIDTH) : (w += 1) {
            const c = self.cells[w][bucket];
            if (c.n == 0 or !self.stepImplausible(w, bucket, c.ms)) continue;
            self.cells[w][bucket] = .{};
            dropped += 1;
        }
        return dropped;
    }

    /// `ms` at `width` exceeds IMPLAUSIBLE_STEP per step over the nearest trusted narrower cell.
    fn stepImplausible(self: *const Table, width: u32, bucket: usize, ms: f32) bool {
        var w = width;
        while (w > 1) {
            w -= 1;
            const below = self.cells[w][bucket];
            if (below.n < MIN_SAMPLES) continue;
            return ms > below.ms * std.math.pow(f32, IMPLAUSIBLE_STEP, @floatFromInt(width - w));
        }
        return false;
    }

    /// Feed one realized plain serial decode token into `serial[bucket]`. Same drop rules as
    /// `observe`; `transition` marks the ticks that follow a speculative round.
    pub fn observeSerial(self: *Table, kv_len: u32, ms: f32, solo: bool, transition: bool) Verdict {
        self.serial_seq +%= 1;
        if (!std.math.isFinite(ms) or ms <= 0) {
            self.serial_dropped_bad += 1;
            return .bad_sample;
        }
        if (!solo) {
            self.serial_dropped_contended += 1;
            return .contended;
        }
        if (transition) {
            self.serial_dropped_transition += 1;
            return .transition;
        }
        const bucket = self.bucketOf(kv_len);
        if (selfSpike(self.serial[bucket], ms, self.serial_seq)) {
            self.dropped_implausible += 1;
            return .implausible;
        }
        self.serial_folded += 1;
        return foldInto(&self.serial[bucket], ms, 1.0, self.serial_seq);
    }

    /// Feed one prompt-lookup round of `drafts` drafts into the lookup row. Same drops as
    /// `observe`, except that a transition is the first round at a draft count (its kernel
    /// compile): a lookup's draft count moves every round, so a changed count is no signal.
    pub fn observeLookup(self: *Table, drafts: u32, kv_len: u32, ms: f32, tokens: f32, solo: bool) Verdict {
        self.lookup_seq +%= 1;
        if (drafts > MAX_WIDTH) return .out_of_range;
        if (!self.lookup_compiled[drafts]) {
            self.lookup_compiled[drafts] = true;
            return .transition;
        }
        if (!std.math.isFinite(ms) or ms <= 0 or !(tokens > 0)) return .bad_sample;
        if (!solo) return .contended;
        const cell = &self.lookup[drafts][self.bucketOf(kv_len)];
        if (selfSpike(cell.*, ms, self.lookup_seq)) return .implausible;
        return foldInto(cell, ms, tokens, self.lookup_seq);
    }

    /// Lookup round ms at `drafts`: measured, else linear between the nearest measured draft
    /// counts in the bucket, else null (outside the span the caller's prior applies).
    pub fn lookupMs(self: *const Table, drafts: u32, bucket: usize) ?f32 {
        if (drafts > MAX_WIDTH or bucket >= N_BUCKETS) return null;
        if (self.lookup[drafts][bucket].n >= MIN_SAMPLES) return self.lookup[drafts][bucket].ms;
        var lo: ?u32 = null;
        var hi: ?u32 = null;
        for (0..MAX_WIDTH + 1) |wi| {
            const w: u32 = @intCast(wi);
            if (self.lookup[w][bucket].n < MIN_SAMPLES) continue;
            if (w < drafts) {
                lo = w;
            } else if (hi == null) {
                hi = w;
            }
        }
        const l = lo orelse return null;
        const h = hi orelse return null;
        return lerp(l, self.lookup[l][bucket].ms, h, self.lookup[h][bucket].ms, drafts);
    }

    fn selfSpike(cell: Cell, ms: f32, clock: u32) bool {
        if (cell.n < MIN_SAMPLES or clock -% cell.last_seen > RESEED_GAP) return false;
        return ms > cell.ms * SELF_SPIKE;
    }

    fn foldInto(cell: *Cell, ms: f32, tokens: f32, clock: u32) Verdict {
        defer cell.last_seen = clock;
        if (cell.n == 0) {
            cell.ms = ms;
            cell.tok = tokens;
            cell.n = 1;
            return .reseeded;
        }
        // The first MIN_SAMPLES are a running MEAN (an EMA seeded from
        // sample 1 is still sample 1 at n=3); the EMA takes over after.
        const stale = clock -% cell.last_seen > RESEED_GAP;
        const beta: f32 = if (stale) RESEED_WEIGHT else if (cell.n < MIN_SAMPLES) 1.0 / @as(f32, @floatFromInt(cell.n + 1)) else BETA;
        cell.ms += beta * (ms - cell.ms);
        cell.tok += beta * (tokens - cell.tok);
        cell.n += 1;
        return if (stale) .reseeded else .folded;
    }

    /// Measured ms of one plain serial token in `bucket`, or null. Never interpolated across buckets.
    pub fn serialMsPerTok(self: *const Table, bucket: usize) ?f32 {
        if (bucket >= N_BUCKETS or self.serial[bucket].n < MIN_SAMPLES) return null;
        return self.serial[bucket].msPerTok();
    }

    /// The bucket resolver for anything holding a table: the layout is the table's.
    pub fn bucketOf(self: *const Table, kv_len: u32) usize {
        return bucketForLayout(kv_len, self.layout);
    }

    fn trusted(self: *const Table, width: u32, bucket: usize) bool {
        return width <= MAX_WIDTH and self.cells[width][bucket].n >= MIN_SAMPLES;
    }

    /// Measured round ms at exactly this width (MIN_SAMPLES folded), or null.
    pub fn measuredMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.cells[width][bucket].ms else null;
    }

    /// Measured tokens per round at exactly this width, or null.
    pub fn measuredTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.cells[width][bucket].tok else null;
    }

    /// Tokens per round for PLANNING: the cell's own `tok` or any trusted narrower cell's,
    /// whichever is larger. `tok` is a workload mixture (w2 learned on echo, w3 on prose read
    /// w3 at 2x per token and the plan never widened), and a wider draft never accepts fewer.
    fn planTok(self: *const Table, width: u32, bucket: usize) f32 {
        var best: f32 = self.cells[width][bucket].tok;
        var w: u32 = 1;
        while (w < width) : (w += 1) {
            if (self.trusted(w, bucket)) best = @max(best, self.cells[w][bucket].tok);
        }
        return best;
    }

    fn planMsPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        const c = self.cells[width][bucket];
        const tok = self.planTok(width, bucket);
        if (c.n == 0 or tok <= 0) return null;
        return c.ms / tok;
    }

    pub fn msPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        return if (self.trusted(width, bucket)) self.planMsPerTok(width, bucket) else null;
    }

    /// Round ms / ms per token from ANY folded cell (n >= 1): evidence for
    /// "worse", never for "better".
    pub fn rawMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH or self.cells[width][bucket].n == 0) return null;
        return self.cells[width][bucket].ms;
    }

    pub fn rawMsPerTok(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (width > MAX_WIDTH) return null;
        return self.planMsPerTok(width, bucket);
    }

    /// `width` has at least one sample and reads CLEARLY_WORSE per token
    /// than trusted `ref`.
    pub fn clearlyWorse(self: *const Table, width: u32, ref: u32, bucket: usize) bool {
        const w = self.rawMsPerTok(width, bucket) orelse return false;
        const r = self.msPerTok(ref, bucket) orelse return false;
        return w >= r * (1.0 + CLEARLY_WORSE);
    }

    pub fn measuredCount(self: *const Table, bucket: usize) u32 {
        var n: u32 = 0;
        for (0..MAX_WIDTH + 1) |w| {
            if (self.trusted(@intCast(w), bucket)) n += 1;
        }
        return n;
    }

    pub fn active(self: *const Table, bucket: usize) bool {
        return self.measuredCount(bucket) >= MIN_WIDTHS;
    }

    /// The bucket a plan at `kv_len` reads: its own when active, else the
    /// nearest active one (lower side preferred — cost grows with KV, so a
    /// lower bucket under-bills rather than over-bills). Null = no active
    /// bucket, the prior applies. A bucket boundary crossed mid-generation
    /// must not snap the plan back to the prior.
    pub fn bucketToRead(self: *const Table, kv_len: u32) ?usize {
        const own = self.bucketOf(kv_len);
        if (self.active(own)) return own;
        var d: usize = 1;
        while (d < N_BUCKETS) : (d += 1) {
            if (own >= d and self.active(own - d)) return own - d;
            if (own + d < N_BUCKETS and self.active(own + d)) return own + d;
        }
        return null;
    }

    /// Narrowest measured width in the bucket (the normalization anchor).
    pub fn narrowestMeasured(self: *const Table, bucket: usize) ?u32 {
        for (0..MAX_WIDTH + 1) |w| {
            if (self.trusted(@intCast(w), bucket)) return @intCast(w);
        }
        return null;
    }

    /// Round ms at `width`: measured, else linear between the two nearest
    /// measured widths. OUTSIDE the measured span the answer is null and
    /// the caller's prior fills in — below, because the prior's extended
    /// rounds land on the cliff first and the cliff's slope run downward
    /// reads every narrower width as free; above, because a shallow slope
    /// run upward (w3 -> w4 +6 ms) priced widths 5..8 as nearly free and the
    /// plan raced there in consecutive transition rounds, measuring nothing
    /// (the caller takes max(last slope, prior marginal) per extra width).
    /// Null while the bucket has fewer than MIN_WIDTHS measured widths.
    pub fn roundMs(self: *const Table, width: u32, bucket: usize) ?f32 {
        if (!self.active(bucket)) return null;
        if (self.measuredMs(width, bucket)) |m| return m;
        var lo: ?u32 = null;
        var hi: ?u32 = null;
        for (0..MAX_WIDTH + 1) |wi| {
            const w: u32 = @intCast(wi);
            if (!self.trusted(w, bucket)) continue;
            if (w < width) {
                lo = w;
            } else if (hi == null) {
                hi = w;
            }
        }
        if (lo != null and hi != null) {
            return lerp(lo.?, self.cells[lo.?][bucket].ms, hi.?, self.cells[hi.?][bucket].ms, width);
        }
        return null;
    }

    pub fn widestMeasured(self: *const Table, bucket: usize) ?u32 {
        var i: usize = MAX_WIDTH + 1;
        while (i > 0) {
            i -= 1;
            if (self.trusted(@intCast(i), bucket)) return @intCast(i);
        }
        return null;
    }

    /// ms per width between the two widest measured widths (a cliff's slope
    /// when one was measured), null with fewer than two.
    pub fn lastSlope(self: *const Table, bucket: usize) ?f32 {
        const hi = self.widestMeasured(bucket) orelse return null;
        var lo: ?u32 = null;
        for (0..hi) |wi| {
            if (self.trusted(@intCast(wi), bucket)) lo = @intCast(wi);
        }
        const l = lo orelse return null;
        return (self.cells[hi][bucket].ms - self.cells[l][bucket].ms) / @as(f32, @floatFromInt(hi - l));
    }

    fn lerp(w0: u32, m0: f32, w1: u32, m1: f32, w: u32) f32 {
        const t = (@as(f32, @floatFromInt(w)) - @as(f32, @floatFromInt(w0))) / (@as(f32, @floatFromInt(w1)) - @as(f32, @floatFromInt(w0)));
        return m0 + t * (m1 - m0);
    }

    /// `w3:12.10/5,w4:11.80/2` — ms per emitted token per width in the
    /// bucket, with the sample count (a cell under MIN_SAMPLES is shown but
    /// does not count), for `[spec-stats]`. Empty when nothing was folded.
    pub fn formatBucket(self: *const Table, bucket: usize, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (0..MAX_WIDTH + 1) |wi| {
            const c = self.cells[wi][bucket];
            const mpt = c.msPerTok() orelse continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("w{d}:{d:.2}/{d}", .{ wi, mpt, c.n }) catch break;
        }
        return w.buffered();
    }

    /// The lookup row in `bucket` as `k<drafts>:<round ms>/<samples>`, for the spec-stats line.
    pub fn formatLookupBucket(self: *const Table, bucket: usize, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (0..MAX_WIDTH + 1) |k| {
            const c = self.lookup[k][bucket];
            if (c.n == 0) continue;
            if (!first) w.writeAll(",") catch break;
            first = false;
            w.print("k{d}:{d:.1}/{d}", .{ k, c.ms, c.n }) catch break;
        }
        return w.buffered();
    }
};

// ── Trial schedule + per-round width chooser ─────────────────────────────

/// Rounds between trials while the gap is unknown, the drag the period is
/// sized to once it is, the cap, and the block length. A block is THREE
/// rounds: the transition, the round after it (still elevated — the regime
/// gate measured the majority shape's first round after a block 3-4% slow),
/// and the measurement. The period is twice the regime gate's: a trial's
/// cost is paid per request while its knowledge persists on the model.
pub const EXPLORE_PERIOD: u32 = 16;
/// Period while the trial's target is still untrusted: the table persists,
/// so this is paid once per (chip, model, quant, OS) — measured M4 base 9B
/// cap 6, w5 reached 1-2 samples per 66-round boot at period 16 and the
/// +5% it buys stayed out of reach.
pub const EXPLORE_PERIOD_COLD: u32 = 8;
/// Half the regime gate's: a re-trial of a known cliff (M1 Pro 27B w5 is
/// 34% dearer than w4) costs block * gap / period of throughput, and the
/// measured per-request cost of one such block was 4.3% on a 22-round
/// request. The knowledge persists on the model; the drag is paid per call.
pub const EXPLORE_DRAG: f32 = 0.005;
pub const EXPLORE_PERIOD_MAX: u32 = 256;
pub const EXPLORE_BLOCK: u32 = 3;

/// Explicit trial schedule: a BLOCK of consecutive rounds every period,
/// idempotent per round (a planner may ask twice for the same round).
/// `idx % period` was tried first and chained trials because the block's
/// own observation moved the period.
///
/// Under `reread` the period is re-read every round and a shorter one pulls the next trial
/// in (`armed_at + period`); a longer one never pushes it out. Without it a request in a cold
/// bucket kept the neighbour bucket's long period and ran w1 for the whole request.
/// `reread = false` is the old arm-once schedule, kept only for the characterization tests.
pub const TrialSchedule = struct {
    trial_end: u32 = 0,
    next_trial: u32 = 0,
    armed_at: u32 = 0,
    trials: u32 = 0,
    last_idx: ?u32 = null,
    last_force: bool = false,

    pub fn force(t: *TrialSchedule, round_idx: u32, period: u32, reread: bool) bool {
        if (t.last_idx == round_idx) return t.last_force;
        t.last_idx = round_idx;
        t.last_force = blk: {
            if (round_idx < t.trial_end) break :blk true;
            if (t.next_trial == 0) {
                t.armed_at = round_idx;
                t.next_trial = round_idx + period;
                break :blk false;
            }
            if (reread) t.next_trial = @min(t.next_trial, t.armed_at + period);
            if (round_idx >= t.next_trial) {
                t.trials += 1;
                t.trial_end = round_idx + EXPLORE_BLOCK;
                t.armed_at = t.trial_end;
                t.next_trial = t.trial_end + period;
                break :blk true;
            }
            break :blk false;
        };
        return t.last_force;
    }

    /// Start trialling at `round_idx` instead of one period later (a block
    /// drafter with no serial measurement must not run eight rounds blind).
    pub fn startAt(t: *TrialSchedule, round_idx: u32) void {
        if (t.next_trial == 0) {
            t.armed_at = @max(1, round_idx);
            t.next_trial = t.armed_at;
        }
    }
};

/// Which layouts re-read the trial period every round: all of them (measured on the 27B
/// sidecar pack: 4k 75.1 vs 71.4 tok/s, 32k inside the per-boot swing). A predicate so a
/// layout can opt out with a measurement.
pub fn schedulePeriodReread(layout: Layout) bool {
    _ = layout;
    return true;
}

/// Period from the measured ms/tok gap between two widths (a width G worse,
/// run once in G/DRAG rounds, costs ~DRAG of throughput); the default while
/// either is unmeasured.
pub fn trialPeriod(a: ?f32, b: ?f32) u32 {
    const x = a orelse return EXPLORE_PERIOD_COLD;
    const y = b orelse return EXPLORE_PERIOD_COLD;
    if (!(x > 0) or !(y > 0)) return EXPLORE_PERIOD;
    const gap = @abs(x - y) / @min(x, y);
    const block: f32 = @floatFromInt(EXPLORE_BLOCK);
    const p: u32 = @intFromFloat(@ceil(block * gap / EXPLORE_DRAG - 1e-3));
    return @min(EXPLORE_PERIOD_MAX, @max(EXPLORE_PERIOD, p));
}

/// A standing choice only moves when the challenger beats it by this
/// margin: a width is measured from rounds interleaved with transitions
/// and reads a few percent slow.
pub const SWITCH_MARGIN: f32 = 0.05;

// ── Persistence ──────────────────────────────────────────────────────────
//
// Knowledge is per (chip, model, quant, OS build, engine build): the same
// binary across boots shares a table, a different binary never does.
// Stored under ~/.mlx-serve/round-cost/<key>.txt, restored at load, written
// at request end unless a barrier diagnostic is armed. Stale version or
// unreadable content is a QUIET miss (the kv_disk_cache discipline).
// `MLX_SERVE_ROUND_COST_PERSIST=0` disables both directions.

/// v2 added the `serial` row; v3 split the top bucket (edges 64k/128k/256k). Bucket indices
/// are the file's only spelling of "which context", so a stale version is a quiet miss.
pub const STORE_VERSION: u32 = 3;

var build_id_buf: [64]u8 = undefined;
var build_id_len: usize = 0;
var build_id_mu: os_mutex.Mutex = .{};

// A release build's sha stands for its bytes (the packager owns that promise); a dev build
// has no sha and hashes the executable so an edit-and-rebuild never shares a table.

// Size plus six sampled 64 KiB windows: a byte-identical reinstall keeps its table (mtime is
// not identity), a rebuilt dylib or metallib rotates it, and a 182 MB metallib costs 384 KiB.

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;
