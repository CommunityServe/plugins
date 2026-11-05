//! The DSpark rounds' choices, MLX-free (plugin.zig runs the rounds). The engine makes them for its built-in
//! deepseek_v41 only (`engines/mlx/spec/dspark.zig` `moduleRound`), so this drafter makes them itself, with the
//! engine's own code (the copies in src/mlx_serve/engines/mlx/spec):
//! - a copy round verifies the context's own continuation after `t1` (`mtp_lookup.Index.matchBest`,
//!   `verify.dsparkCopyLen`), with no stage forward;
//! - a stage round verifies the prefix of the stages' block whose calibrated confidence buys the most tokens per ms of
//!   verify (`dflash/policy.zig`), on the verify ladder this model measures (`Ladder`);
//! - where drafting does not pay, the drafting backoff skips the stages and the round verifies `t1` alone;
//! - a round may refresh a ladder cell gone stale instead (`Ladder.due`): `t1` alone for the 1-row cell, the whole
//!   block on a stage round.
//! Each choice is only what to verify: the trunk's accept decides what is kept.

const std = @import("std");
const lookup = @import("mlx_serve/engines/mlx/spec/mtp_lookup.zig");
const dflash = @import("mlx_serve/engines/mlx/spec/dflash/policy.zig");
const verify = @import("mlx_serve/engines/mlx/spec/verify.zig");

/// The two features (the server's environment, read at load): copy rounds, and the verify policy with its backoff. With
/// `fixed` (`MLX_SERVE_DSV41_DSPARK_POLICY=fixed`) the policy reads a fixed cost table, never the ladder or a round's
/// wall time: each choice is a function of the drafts' confidence and the request's own accepts, so a request's text is
/// the same run to run.
pub const Switches = struct {
    copy: bool = true,
    policy: bool = true,
    fixed: ?Fixed = null,
    /// MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY (aw1): the fixed policy cuts and backs off a greedy request's rounds too; off,
    /// it stands aside on them (the whole block). Off unless set (aw1s: the cut lost on the same greedy text).
    fixed_greedy: bool = false,
};

/// The fixed policy's tables when `MLX_SERVE_DSV41_DSPARK_COSTS` gives none: fitted on the M5 Max on the a4k canonical pack
/// (lad2, w-lad2-20261102-1707: every bucket by one rule, machine timings only, kept apart from every benchmark prompt:
/// a diagnostic build forcing a greedy round's verify width 1..6 in turn on the warm
/// request, a card first on the same prompt, at about 1K (the 2048 table), 4K (the 8192 table) and 16K (the unbounded
/// table) prompt tokens of two prompts of their own (Python's asyncio and email sources, a task tail of their own): row w
/// the two prompts' median verify ms at w rows plus the bucket's per-round overhead (the median round less its verify,
/// which a plain round pays as a stage round does), to 0.1 ms and strictly increasing; the draft's ms the greedy rounds'),
/// bounded at 2048 and 8192 positions.
pub const DEFAULT_COSTS = "2048:41.4,47.9,57.2,66.4,77.2,84.4/4.8+8192:40.1,47.1,55.0,64.9,73.2,85.1/4.8+41.0,47.9,57.9,65.8,76.7,85.9/4.9";

/// `MLX_SERVE_DSV41_DSPARK_COSTS`: the fixed policy's tables, fitted once per box and pack (machine tuning, never on a
/// benchmark prompt; `DEFAULT_COSTS` unset): `v1,v2,...,vN/d` for every context, or tables by context `U1:t1+U2:t2+...+tK`, each `t` a table,
/// the bounds `U` ascending positions and the last table unbounded; a round at the state's position n reads the first
/// table whose bound exceeds n. A table: the verify ms by rows from one (`t1` alone) to N, then the ms a stage draft
/// takes. A block's cut never verifies more than N rows (the rows past the table cost +inf); the backoff weighs a drafted
/// round as `d` plus its rows' verify, its tokens as one-row verifies. Sampled requests only (`Drafter.greedy`).
pub const Fixed = struct {
    buckets: [MAX_BUCKETS]Table = undefined,
    n: usize = 0,

    pub const MAX_BUCKETS = 8;
    /// No bound: the last table's.
    pub const UNBOUNDED = std.math.maxInt(u32);
    pub const Table = struct {
        upto: u32 = UNBOUNDED,
        costs: dflash.Costs,
        rows: usize,
        draft_ms: f32,
    };

    pub const Error = error{DsparkCostsSetting};

    /// Every value a positive finite ms (the draft's may be 0), 2 to `dflash.MAX_ROWS` rows a table, the bounds ascending
    /// integers, only the last table unbounded, at most `MAX_BUCKETS` tables; anything else refused.
    pub fn parse(value: []const u8) Error!Fixed {
        var f: Fixed = .{};
        var it = std.mem.splitScalar(u8, value, '+');
        var last: u32 = 0;
        while (it.next()) |item| {
            if (f.n == MAX_BUCKETS) return error.DsparkCostsSetting;
            if (f.n > 0 and f.buckets[f.n - 1].upto == UNBOUNDED) return error.DsparkCostsSetting;
            var t = item;
            var upto: u32 = UNBOUNDED;
            if (std.mem.indexOfScalar(u8, item, ':')) |c| {
                upto = std.fmt.parseInt(u32, item[0..c], 10) catch return error.DsparkCostsSetting;
                if (upto == 0 or upto == UNBOUNDED or upto <= last) return error.DsparkCostsSetting;
                last = upto;
                t = item[c + 1 ..];
            }
            f.buckets[f.n] = try parseTable(t);
            f.buckets[f.n].upto = upto;
            f.n += 1;
        }
        if (f.n == 0 or f.buckets[f.n - 1].upto != UNBOUNDED) return error.DsparkCostsSetting;
        return f;
    }

    fn parseTable(value: []const u8) Error!Table {
        const slash = std.mem.indexOfScalar(u8, value, '/') orelse return error.DsparkCostsSetting;
        var f: Table = .{ .costs = @splat(std.math.inf(f32)), .rows = 0, .draft_ms = 0 };
        f.costs[0] = 0;
        var it = std.mem.splitScalar(u8, value[0..slash], ',');
        while (it.next()) |t| {
            if (f.rows == dflash.MAX_ROWS) return error.DsparkCostsSetting;
            const v = std.fmt.parseFloat(f32, t) catch return error.DsparkCostsSetting;
            if (!std.math.isFinite(v) or v <= 0) return error.DsparkCostsSetting;
            f.rows += 1;
            f.costs[f.rows] = v;
        }
        if (f.rows < 2) return error.DsparkCostsSetting;
        const d = std.fmt.parseFloat(f32, value[slash + 1 ..]) catch return error.DsparkCostsSetting;
        if (!std.math.isFinite(d) or d < 0) return error.DsparkCostsSetting;
        f.draft_ms = d;
        return f;
    }

    /// The table a round at the state's position `n` reads.
    pub fn at(f: *const Fixed, n: usize) *const Table {
        for (f.buckets[0 .. f.n - 1]) |*t| if (n < t.upto) return t;
        return &f.buckets[f.n - 1];
    }

    /// The tables as `1:v1,2:v2,.../d`, each bounded one led by `<U:`, joined by `+`, for the meters.
    pub fn describe(f: *const Fixed, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        for (f.buckets[0..f.n], 0..) |*t, i| {
            if (i > 0) w.writeAll("+") catch break;
            if (t.upto != UNBOUNDED) w.print("<{d}:", .{t.upto}) catch break;
            for (1..t.rows + 1) |r| w.print("{s}{d}:{d:.1}", .{ if (r == 1) "" else ",", r, t.costs[r] }) catch break;
            w.print("/{d:.1}", .{t.draft_ms}) catch {};
        }
        return w.buffered();
    }
};

/// The tokens the model's state holds, by position (the request's prompt, then what each round and forward took), in
/// the lookup's index. A gap (positions it never saw) or a failed forward loses it until the next context.
pub const History = struct {
    idx: lookup.Index,
    lost: bool = false,

    pub fn init(gpa: std.mem.Allocator) History {
        return .{ .idx = lookup.Index.init(gpa) };
    }

    pub fn deinit(self: *History) void {
        self.idx.deinit();
    }

    pub fn size(self: *const History) usize {
        return self.idx.toks.items.len;
    }

    /// Holds each of the state's `n` positions.
    pub fn whole(self: *const History, n: usize) bool {
        return !self.lost and self.size() == n;
    }

    /// A new context holding `kept` (a restored prompt prefix; empty for a cold one).
    pub fn reset(self: *History, kept: []const u32) void {
        self.empty();
        self.lost = false;
        self.idx.extend(kept) catch self.lose();
    }

    /// `ids` entered the state at position `at`; position 0 starts a new context.
    pub fn write(self: *History, at: usize, ids: []const u32) void {
        if (at == 0) return self.reset(ids);
        if (self.lost) return;
        if (at > self.size()) return self.lose();
        // Back to the first `at` positions (a state never rewinds within a request): the index rebuilt over them.
        if (at < self.size()) {
            var kept = lookup.Index.init(self.idx.allocator);
            kept.extend(self.idx.toks.items[0..at]) catch {
                kept.deinit();
                return self.lose();
            };
            self.idx.deinit();
            self.idx = kept;
        }
        self.idx.extend(ids) catch self.lose();
    }

    /// Its positions no longer match the state's: nothing is copied until the next context.
    pub fn lose(self: *History) void {
        self.empty();
        self.lost = true;
    }

    fn empty(self: *History) void {
        const gpa = self.idx.allocator;
        self.idx.deinit();
        self.idx = lookup.Index.init(gpa);
    }
};

/// The drafts a copy round verifies after `t1` at the state's position `n`: the context's continuation where it agrees
/// with the committed tail and `t1` far enough back and runs past the stage `block`, at most `max` and `remaining` (the
/// engine's `dsparkCopyStep`); empty for none. Borrowed from the history until its next write.
pub fn copyDrafts(h: *const History, n: usize, t1: u32, max: usize, remaining: u32, block: usize) []const u32 {
    if (!h.whole(n) or max == 0) return &.{};
    const found = h.idx.matchBest(t1, @intCast(max));
    const k = verify.dsparkCopyLen(found, remaining, block);
    return if (k == 0) &.{} else found.?.draft[0..k];
}

/// Verify ms by rows (`dflash.Costs`, index = rows) as this model measures its own verifies. The engine's prior ladder
/// (`dflash.priorCosts`, GLM-5.3's on an M5 Ultra) holds until the first sample; then:
/// - The 1-row cell is the anchor, in ms; every other cell keeps its rows' cost as a ratio to the anchor (each sample
///   over the anchor's estimate when it came). A level change (a cold box warming, a cache filling) shows first in the
///   anchor, which every plain round samples, and every row moves with it whether or not a round verifies it.
/// - A cell's estimate is the lower median of its last `window` samples and one pseudo-sample from the prior ladder
///   (its 1-row ms for the anchor, its ratio for the others). One sample can lower a cell below the prior but not lift
///   it: a first verify that compiled its kernels moves nothing, two agreeing samples outvote the prior. A full window
///   moves all the way to a new level in 4 or 5 samples, and a single stall in it moves the estimate by one rank.
/// - A cell whose newest sample is more than `fresh_rounds` rounds old may hold cold samples: it counts no dearer than
///   the fresh cells around it put it, and its window restarts at its next sample. A row no fresh cell holds takes the
///   prior ladder's shape, bent to the nearest fresh rows below and above (the anchor at one row): its deviation from
///   the prior's ratio is linear between theirs, and the widest fresh row's beyond it.
/// - Never cheaper for more rows: a row measured dear lifts the rows above it.
/// A cell that errs cheap corrects itself (the policy verifies those rows); one that errs dear is never verified
/// again, which is how the PlainCost cells (2.5% a sample, started from the prior at a cold unit) stuck at
/// `1:7.7,2:27.7,3:32.2,5:35.8` with 2 rows measuring 15.5 ms. So the cells a round must refresh are named by `due`,
/// and the drafter's rounds take them (`Drafter.plan`, `Drafter.cut`).
pub const Ladder = struct {
    pub const window = 8;
    pub const fresh_rounds: u64 = 64;
    cells: [dflash.MAX_ROWS + 1]Cell = @splat(.{}),
    /// Samples taken, one a round: the clock staleness is counted on.
    rounds: u64 = 0,

    const Cell = struct {
        ring: [window]f32 = @splat(0),
        /// Samples held (`ring[0..n]`) and the next one's slot.
        n: usize = 0,
        next: usize = 0,
        /// `Ladder.rounds` at the newest sample, and when a refresh was last asked of the cell.
        at: u64 = 0,
        asked: ?u64 = null,

        fn push(c: *Cell, v: f32, at: u64) void {
            c.ring[c.next] = v;
            c.next = (c.next + 1) % window;
            c.n = @min(c.n + 1, window);
            c.at = at;
        }

        /// The lower median of the samples and `pseudo`.
        fn estimate(c: *const Cell, pseudo: f32) f32 {
            var v: [window + 1]f32 = undefined;
            @memcpy(v[0..c.n], c.ring[0..c.n]);
            v[c.n] = pseudo;
            std.mem.sort(f32, v[0 .. c.n + 1], {}, std.sort.asc(f32));
            return v[c.n / 2];
        }
    };

    fn fresh(self: *const Ladder, rows: usize) bool {
        const c = &self.cells[rows];
        return c.n > 0 and self.rounds - c.at <= fresh_rounds;
    }

    /// The 1-row cell's ms, the prior's before its first sample.
    fn anchor(self: *const Ladder) f32 {
        const c = &self.cells[1];
        return if (c.n == 0) dflash.PRIOR_PLAIN_MS else c.estimate(dflash.PRIOR_PLAIN_MS);
    }

    pub fn costs(self: *const Ladder) dflash.Costs {
        const prior = dflash.priorCosts(dflash.PRIOR_PLAIN_MS);
        if (self.rounds == 0) return prior;
        // The fresh cells' ratios, the anchor's own 1.
        var known: [dflash.MAX_ROWS + 1]?f32 = @splat(null);
        known[1] = 1;
        for (2..dflash.MAX_ROWS + 1) |r| {
            if (self.fresh(r)) known[r] = self.cells[r].estimate(prior[r] / prior[1]);
        }
        const a = self.anchor();
        var out: dflash.Costs = undefined;
        out[0] = 0;
        for (1..dflash.MAX_ROWS + 1) |r| {
            const ratio = known[r] orelse blk: {
                const around = between(&known, &prior, r);
                const c = &self.cells[r];
                break :blk if (c.n > 0) @min(c.estimate(prior[r] / prior[1]), around) else around;
            };
            out[r] = if (r == 1) a else @max(a * ratio, out[r - 1]);
        }
        return out;
    }

    /// The ratio of row `r`, which no fresh cell holds, from the nearest fresh rows below and above (`known`).
    fn between(known: *const [dflash.MAX_ROWS + 1]?f32, prior: *const dflash.Costs, r: usize) f32 {
        var lo = r - 1;
        while (known[lo] == null) lo -= 1;
        var hi = r + 1;
        while (hi <= dflash.MAX_ROWS and known[hi] == null) hi += 1;
        // Each fresh row's ratio over the prior's: 1 where the measured ladder has the prior's shape.
        const dev_lo = known[lo].? * prior[1] / prior[lo];
        const dev = if (hi > dflash.MAX_ROWS) dev_lo else blk: {
            const t = @as(f32, @floatFromInt(r - lo)) / @as(f32, @floatFromInt(hi - lo));
            break :blk dev_lo + (known[hi].? * prior[1] / prior[hi] - dev_lo) * t;
        };
        return dev * prior[r] / prior[1];
    }

    /// A verify of `rows` rows took `ms` (a time that is not positive and finite teaches nothing; the round counts).
    pub fn sample(self: *Ladder, rows: usize, ms: f32) void {
        defer self.rounds += 1;
        if (rows == 0 or rows > dflash.MAX_ROWS or !std.math.isFinite(ms) or ms <= 0) return;
        const v = if (rows == 1) ms else ms / self.anchor();
        const c = &self.cells[rows];
        if (!self.fresh(rows)) {
            c.n = 0;
            c.next = 0;
        }
        c.push(v, self.rounds);
    }

    /// Whether `rows`' cell wants a round to sample it: it holds no fresh sample once the ladder has any, and no
    /// refresh was asked of it in the last `fresh_rounds` rounds.
    pub fn due(self: *const Ladder, rows: usize) bool {
        if (self.rounds == 0 or rows == 0 or rows > dflash.MAX_ROWS or self.fresh(rows)) return false;
        const asked = self.cells[rows].asked orelse return true;
        return self.rounds - asked > fresh_rounds;
    }

    /// A round verifies `rows` rows to refresh their cell.
    pub fn ask(self: *Ladder, rows: usize) void {
        if (rows == 0 or rows > dflash.MAX_ROWS) return;
        self.cells[rows].asked = self.rounds;
    }

    /// `rows:ms` of the cells holding samples, as `costs` reads them (`~` a stale one), for `[spec-stats]`.
    pub fn formatCells(self: *const Ladder, buf: []u8) []const u8 {
        const c = self.costs();
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.cells, 0..) |cell, rows| if (cell.n > 0) {
            w.print("{s}{d}:{d:.1}{s}", .{ if (first) "" else ",", rows, c[rows], if (self.fresh(rows)) "" else "~" }) catch break;
            first = false;
        };
        return w.buffered();
    }
};

pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// The drafts a stage round verifies: the prefix whose calibrated acceptance (the confidence head's logits through
/// `calib`) buys the most tokens per ms of verify on `costs` (the engine's `dsparkVerifyDrafts`, over a given ladder).
pub fn verifyDrafts(conf_logits: []const f32, calib: *const dflash.Calibration, costs: *const dflash.Costs) usize {
    var p: [dflash.MAX_DRAFTS]f32 = undefined;
    const n = @min(conf_logits.len, p.len);
    for (conf_logits[0..n], p[0..n]) |c, *o| o.* = calib.probability(sigmoid(c));
    return dflash.chooseRows(p[0..n], costs).rows - 1;
}

pub const Kind = enum { copy, stage, alone };

/// What a round verifies after `t1`, chosen before any stage forward.
pub const Plan = union(Kind) {
    /// The context's continuation, borrowed from the history until its next write.
    copy: []const u32,
    /// The stages' draft, cut by `Drafter.cut` once drafted.
    stage,
    /// `t1` alone: the backoff skips the stages, or no draft can be kept.
    alone,
};

/// One request's round policy (the engine's `DflashPol`, its plain cost the ladder's 1-row cell) and counters.
pub const Request = struct {
    calib: dflash.Calibration = .{},
    drafting: dflash.Drafting = .{},
    /// The stage rounds' drafts verified and kept by position, and their verify rows.
    stats: dflash.PositionStats = .{},
    stage_rounds: u64 = 0,
    copy_rounds: u64 = 0,
    copy_drafted: u64 = 0,
    copy_accepted: u64 = 0,
    alone_rounds: u64 = 0,
    /// Rounds that verified what a stale ladder cell wanted (`Ladder.due`) instead of the policy's choice.
    refreshes: u64 = 0,
};

/// The drafter's state beside its model: the model's history and ladder, the policy of the request it serves.
pub const Drafter = struct {
    on: Switches,
    history: History,
    ladder: Ladder = .{},
    req: Request = .{},
    /// The round `cut` gave the whole block to refresh its cell: it teaches the ladder and the calibration, not the
    /// backoff, whose judgment is of the policy's own widths.
    refreshing: bool = false,
    /// The round is a greedy request's (`draftStep`, set before `plan`): the fixed policy leaves it the whole block, no
    /// backoff, as with the policy off (its cut changes the verify's shapes, and with them the greedy text, for no gain).
    greedy: bool = false,
    /// The state's position at the round's `plan`: the fixed table `cut` and `learn` read.
    at_n: usize = 0,

    pub fn init(gpa: std.mem.Allocator, on: Switches) Drafter {
        return .{ .on = on, .history = History.init(gpa) };
    }

    pub fn deinit(self: *Drafter) void {
        self.history.deinit();
    }

    /// A request begins on a state holding `kept`, its prompt's restored prefix.
    pub fn begin(self: *Drafter, kept: []const u32) void {
        self.req = .{};
        self.history.reset(kept);
    }

    /// The round from `t1` at the state's position `n`: a copy where the context holds one, else `t1` alone where no
    /// draft can be kept, the ladder's 1-row cell is due a refresh or the backoff skips the stages, else the stages'
    /// draft. A verify carries and the round keeps at most `max` drafts; `remaining` is the tokens still allowed after
    /// `t1`, `block` the stages' block.
    pub fn plan(self: *Drafter, n: usize, t1: u32, max: usize, remaining: u32, block: usize) Plan {
        self.refreshing = false;
        self.at_n = n;
        if (max == 0) return .alone;
        if (self.on.copy) {
            const c = copyDrafts(&self.history, n, t1, max, remaining, block);
            if (c.len > 0) return .{ .copy = c };
        }
        if (!self.on.policy or self.fixedGreedy()) return .stage;
        // The anchor goes stale only while every round verifies more than t1 (the backoff skipping none): a plain round
        // refreshes it, and stands for one the backoff owes when it owes any. The fixed policy has no ladder to refresh.
        if (self.on.fixed == null and self.ladder.due(1)) {
            if (self.req.drafting.owed > 0) _ = self.req.drafting.shouldDraft();
            self.ladder.ask(1);
            self.req.refreshes += 1;
            return .alone;
        }
        if (!self.req.drafting.shouldDraft()) return .alone;
        return .stage;
    }

    /// turboexl3_ds41 (the plugin's aheadLaunch): whether `plan` would return `.stage` for these arguments now,
    /// without its effects: the copy check as `plan` makes it, and the stages' block only where `plan` gives it without
    /// the ladder or the backoff (the policy off, or the fixed policy standing aside on a greedy round).
    pub fn wouldStage(self: *const Drafter, n: usize, t1: u32, max: usize, remaining: u32, block: usize) bool {
        if (max == 0) return false;
        if (self.on.copy and copyDrafts(&self.history, n, t1, max, remaining, block).len > 0) return false;
        if (!self.on.policy or self.fixedGreedy()) return true;
        // aw1: the fixed policy's plan is the stages' unless the backoff owes a plain round (no ladder to refresh).
        return self.on.fixed != null and self.req.drafting.owed == 0;
    }

    /// The drafts of a stage block a round verifies, `conf` the block's confidence logits: the policy's cut, or all.
    /// All as well when the whole block's cell is due a refresh (`Ladder.due`), on any stage round, the backoff's probes
    /// (the rounds that draft within a skip stretch) included, so it never drafts a round the backoff would not. The
    /// whole block is the end the rows between it and the anchor are interpolated from, and a wider verify only adds
    /// tokens to the policy's. No narrower row is refreshed: on a probe that would cost the probe its tokens, and with
    /// them the backoff's way back (a probe that verified two rows of a block worth six lost, and kept the backoff at
    /// its longest period); a stale row counts no dearer than the fresh rows around it instead (`Ladder.costs`).
    pub fn cut(self: *Drafter, conf: []const f32) usize {
        if (!self.on.policy or self.fixedGreedy()) return conf.len;
        if (self.on.fixed) |*f| return verifyDrafts(conf, &self.req.calib, &f.at(self.at_n).costs);
        if (conf.len > 0 and self.ladder.due(conf.len + 1)) {
            self.ladder.ask(conf.len + 1);
            self.req.refreshes += 1;
            self.refreshing = true;
            return conf.len;
        }
        const costs = self.ladder.costs();
        return verifyDrafts(conf, &self.req.calib, &costs);
    }

    /// A round kept `accepted` of the `drafts` it verified (a stage round's confidence logits in `conf`, else empty);
    /// its verify took `verify_ms`, the whole round `wall_ms`. A stage round teaches the calibration each position it
    /// reached, and (but a refresh) the backoff what its tokens would have cost as plain rounds against what it took.
    pub fn learn(self: *Drafter, kind: Kind, drafts: usize, conf: []const f32, accepted: u32, verify_ms: f32, wall_ms: f32) void {
        defer self.refreshing = false;
        const q = &self.req;
        self.ladder.sample(drafts + 1, verify_ms);
        switch (kind) {
            .copy => {
                q.copy_rounds += 1;
                q.copy_drafted += drafts;
                q.copy_accepted += accepted;
            },
            .alone => q.alone_rounds += 1,
            .stage => {
                q.stage_rounds += 1;
                q.stats.record(@intCast(drafts), accepted);
                if (!self.on.policy or self.fixedGreedy()) return;
                for (conf, 0..) |c, k| {
                    if (k > accepted) break;
                    q.calib.observe(sigmoid(c), k < accepted);
                }
                // The fixed policy weighs the round on its table: the draft's ms and its rows' verify, never the clock.
                if (self.on.fixed) |*f| {
                    const t = f.at(self.at_n);
                    q.drafting.record(@as(f32, @floatFromInt(accepted + 1)) * t.costs[1], t.draft_ms + t.costs[@min(drafts + 1, dflash.MAX_ROWS)]);
                } else if (!self.refreshing)
                    q.drafting.record(@as(f32, @floatFromInt(accepted + 1)) * self.ladder.costs()[1], wall_ms);
            },
        }
    }

    /// The fixed policy on a greedy request's round, where it stands aside (`greedy`): unless it covers greedy rounds
    /// (`fixed_greedy`, aw1).
    fn fixedGreedy(self: *const Drafter) bool {
        return self.on.fixed != null and self.greedy and !self.on.fixed_greedy;
    }

    /// The request's `[spec-stats]` lines beside the engine's own, which counts every native round a stage round; then
    /// its counters reset. Empty when the request ran no round.
    pub fn meters(self: *Drafter, buf: []u8) []const u8 {
        const q = &self.req;
        defer {
            q.stats = .{};
            q.stage_rounds = 0;
            q.copy_rounds = 0;
            q.copy_drafted = 0;
            q.copy_accepted = 0;
            q.alone_rounds = 0;
            q.refreshes = 0;
        }
        if (q.stage_rounds + q.copy_rounds + q.alone_rounds == 0) return &.{};
        var w = std.Io.Writer.fixed(buf);
        w.print("[spec-stats] dspark rounds: stage_rounds={d} copy_rounds={d} copy_drafted={d} copy_accepted={d} alone_rounds={d} copy={s} policy={s}\n", .{
            q.stage_rounds, q.copy_rounds, q.copy_drafted, q.copy_accepted, q.alone_rounds, onOff(self.on.copy), if (self.fixedGreedy()) "fixed: greedy, the whole block" else if (self.on.fixed != null) "fixed" else onOff(self.on.policy),
        }) catch return w.buffered();
        if (self.on.fixed) |*f| {
            var tb: [1600]u8 = undefined;
            w.print("[spec-stats] dspark fixed policy: {s} costs_ms={s}\n", .{ if (self.greedy and !self.on.fixed_greedy) "greedy: stood aside" else if (self.greedy) "greedy: cut" else "sampled", f.describe(&tb) }) catch return w.buffered();
        }
        var acc: [128]u8 = undefined;
        var rows: [160]u8 = undefined;
        var cells: [256]u8 = undefined;
        w.print("[spec-stats] dspark policy: skipped={d} probes={d} refreshes={d} plain_ms={d:.2} acc_pos={s} rows_hist={s} ladder_ms={s}\n", .{
            q.drafting.skipped, q.drafting.probes, q.refreshes, self.ladder.costs()[1], q.stats.formatAcc(&acc), q.stats.formatRows(&rows), self.ladder.formatCells(&cells),
        }) catch {};
        return w.buffered();
    }
};

fn onOff(on: bool) []const u8 {
    return if (on) "on" else "off";
}

// ── tests ──

const testing = std.testing;

/// A 30-token passage `p` and a prompt quoting its first `quoted` tokens again after a 3-token break.
const Quote = struct {
    p: [30]u32,
    prompt: [63]u32,
    len: usize,

    fn of(quoted: usize) Quote {
        var q: Quote = .{ .p = undefined, .prompt = undefined, .len = 33 + quoted };
        for (&q.p, 0..) |*x, i| x.* = @intCast(100 + i);
        @memcpy(q.prompt[0..30], &q.p);
        q.prompt[30..33].* = .{ 7, 8, 9 };
        @memcpy(q.prompt[33..][0..quoted], q.p[0..quoted]);
        return q;
    }

    fn ids(q: *const Quote) []const u32 {
        return q.prompt[0..q.len];
    }
};

test "copy rounds: a long agreement drafts the context's continuation past the stage block; a short one, a short budget or a stale history none" {
    var h = History.init(testing.allocator);
    defer h.deinit();
    // The quote has reached p[20]: t1 = p[21] agrees back 22 tokens, so the copy is p[22..], at most 7 drafts.
    const long = Quote.of(21);
    h.reset(long.ids());
    try testing.expectEqualSlices(u32, long.p[22..29], copyDrafts(&h, long.len, long.p[21], 7, 100, 5));
    // The budget left after t1 caps it: 6 still outruns the block, 5 does not.
    try testing.expectEqualSlices(u32, long.p[22..28], copyDrafts(&h, long.len, long.p[21], 7, 6, 5));
    try testing.expectEqual(@as(usize, 0), copyDrafts(&h, long.len, long.p[21], 7, 5, 5).len);
    try testing.expectEqual(@as(usize, 0), copyDrafts(&h, long.len, long.p[21], 5, 100, 5).len);
    // A t1 the context never had after this tail.
    try testing.expectEqual(@as(usize, 0), copyDrafts(&h, long.len, 4000, 7, 100, 5).len);
    // A history one position short of the state's copies nothing.
    try testing.expectEqual(@as(usize, 0), copyDrafts(&h, long.len + 1, long.p[21], 7, 100, 5).len);
    // Quoting 11 tokens agrees back 12, under the 16 a copy needs.
    const short = Quote.of(11);
    h.reset(short.ids());
    try testing.expectEqual(@as(usize, 0), copyDrafts(&h, short.len, short.p[11], 7, 100, 5).len);
}

test "the history follows the state across a request reset, a restored prefix and a gap" {
    var d = Drafter.init(testing.allocator, .{});
    defer d.deinit();
    const q = Quote.of(21);
    // A resumed request: the state kept 40 of the prompt, the prefill forwards the rest, a round keeps three tokens.
    d.begin(q.ids()[0..40]);
    d.history.write(40, q.ids()[40..]);
    try testing.expect(d.history.whole(q.len));
    d.history.write(q.len, &.{ q.p[21], q.p[22], q.p[23] });
    try testing.expect(d.history.whole(q.len + 3));
    try testing.expectEqualSlices(u32, q.p[25..30], copyDrafts(&d.history, q.len + 3, q.p[24], 5, 100, 4));
    d.req.copy_rounds = 3;
    // The next request is another conversation, cold: nothing of the first one is copied, and its counters are gone.
    const other = [_]u32{ 300, 301, 302, 303, 304, 305 };
    d.begin(&.{});
    d.history.write(0, &other);
    try testing.expect(d.history.whole(other.len));
    try testing.expectEqual(@as(u64, 0), d.req.copy_rounds);
    try testing.expectEqual(@as(usize, 0), copyDrafts(&d.history, other.len, q.p[24], 7, 100, 5).len);
    // A forward past positions the history never saw loses it until the next context.
    d.history.write(other.len + 2, &.{ 1, 2 });
    try testing.expect(!d.history.whole(d.history.size()));
    d.history.write(other.len + 4, &.{3});
    try testing.expect(!d.history.whole(d.history.size()));
    d.begin(q.ids());
    try testing.expectEqualSlices(u32, q.p[22..29], copyDrafts(&d.history, q.len, q.p[21], 7, 100, 5));
}

test "the fixed policy's table: verify ms by rows then the draft's, refused by name when malformed" {
    const f = (try Fixed.parse("41.6,59.9,71.3,79.6,94.8,100.5/5")).at(0).*;
    try testing.expectEqual(@as(usize, 6), f.rows);
    try testing.expectEqual(@as(f32, 41.6), f.costs[1]);
    try testing.expectEqual(@as(f32, 100.5), f.costs[6]);
    try testing.expect(std.math.isInf(f.costs[7]));
    try testing.expectEqual(@as(f32, 5), f.draft_ms);
    _ = try Fixed.parse("57,70/0");
    for ([_][]const u8{ "", "57", "57,70", "57/5", "57,70/", "57,70/-1", "57,-1/5", "57,0/5", "57,x/5", "57,,70/5", " 57,70/5", "57,70/5x", "57,inf/5", "57,nan/5", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17/5" }) |v|
        try testing.expectError(error.DsparkCostsSetting, Fixed.parse(v));
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("1:57.0,2:70.0/0.0", (try Fixed.parse("57,70/0")).describe(&buf));
}

test "the fixed policy's tables by context: the first whose bound exceeds the position, the last unbounded" {
    const f = try Fixed.parse("2048:30,40,50/4+8192:35,46,57/4.5+45,60,72/5");
    try testing.expectEqual(@as(usize, 3), f.n);
    try testing.expectEqual(@as(f32, 30), f.at(0).costs[1]);
    try testing.expectEqual(@as(f32, 30), f.at(2047).costs[1]);
    try testing.expectEqual(@as(f32, 35), f.at(2048).costs[1]);
    try testing.expectEqual(@as(f32, 4.5), f.at(8191).draft_ms);
    try testing.expectEqual(@as(f32, 45), f.at(8192).costs[1]);
    try testing.expectEqual(@as(f32, 72), f.at(1 << 20).costs[3]);
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("<2048:1:30.0,2:40.0,3:50.0/4.0+<8192:1:35.0,2:46.0,3:57.0/4.5+1:45.0,2:60.0,3:72.0/5.0", f.describe(&buf));
    // The last table unbounded, the bounds ascending, only the last unbounded, at most eight tables.
    for ([_][]const u8{ "2048:30,40/4", "8192:30,40/4+2048:35,46/4+45,60/5", "2048:30,40/4+2048:35,46/4+45,60/5", "30,40/4+45,60/5", "0:30,40/4+45,60/5", "x:30,40/4+45,60/5", "2048:30,40/4+", "+45,60/5", "1:1,2/0+2:1,2/0+3:1,2/0+4:1,2/0+5:1,2/0+6:1,2/0+7:1,2/0+8:1,2/0+1,2/0" }) |v|
        try testing.expectError(error.DsparkCostsSetting, Fixed.parse(v));
    _ = try Fixed.parse("1:1,2/0+2:1,2/0+3:1,2/0+4:1,2/0+5:1,2/0+6:1,2/0+7:1,2/0+1,2/0");
    // A drafter reads the table of the round's position: a block a 2-row table caps at one draft, the next at the block.
    var d = Drafter.init(testing.allocator, .{ .copy = false, .fixed = try Fixed.parse("100:30,40/4+30,40,50,60,70,80/4") });
    defer d.deinit();
    _ = d.plan(50, 0, 5, 100, 5);
    try testing.expectEqual(@as(usize, 1), d.cut(&.{ 9, 9, 9, 9, 9 }));
    _ = d.plan(500, 0, 5, 100, 5);
    try testing.expectEqual(@as(usize, 5), d.cut(&.{ 9, 9, 9, 9, 9 }));
}

test "the built-in tables parse: three by context, each strictly increasing by rows" {
    const f = try Fixed.parse(DEFAULT_COSTS);
    try testing.expectEqual(@as(usize, 3), f.n);
    try testing.expectEqual(@as(u32, 2048), f.buckets[0].upto);
    try testing.expectEqual(@as(u32, 8192), f.buckets[1].upto);
    try testing.expectEqual(Fixed.UNBOUNDED, f.buckets[2].upto);
    for (f.buckets[0..f.n]) |t| {
        try testing.expectEqual(@as(usize, 6), t.rows);
        for (2..t.rows + 1) |r| try testing.expect(t.costs[r] > t.costs[r - 1]);
    }
    try testing.expectEqual(@as(f32, 41.4), f.at(100).costs[1]);
    try testing.expectEqual(@as(f32, 85.1), f.at(4096).costs[6]);
    try testing.expectEqual(@as(f32, 4.9), f.at(16384).draft_ms);
}

test "the fixed policy stands aside on a greedy round: the whole block, no backoff, as with the policy off" {
    const table = try Fixed.parse("41.6,59.9,71.3,79.6,94.8,100.5/5");
    var g = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table, .fixed_greedy = false });
    defer g.deinit();
    var off = Drafter.init(testing.allocator, .{ .copy = false, .policy = false });
    defer off.deinit();
    g.greedy = true;
    // Rounds that keep nothing would back a sampled request's drafting off; a greedy one drafts the whole block each time.
    for (0..40) |_| {
        const pg = g.plan(1, 0, 5, 100, 5);
        const po = off.plan(1, 0, 5, 100, 5);
        try testing.expect(pg == .stage and po == .stage);
        try testing.expectEqual(@as(usize, 5), g.cut(&.{ -4, -4, -4, -4, -4 }));
        try testing.expectEqual(off.cut(&.{ -4, -4, -4, -4, -4 }), g.cut(&.{ -4, -4, -4, -4, -4 }));
        g.learn(.stage, 5, &.{ -4, -4, -4, -4, -4 }, 0, 80, 85);
        off.learn(.stage, 5, &.{ -4, -4, -4, -4, -4 }, 0, 80, 85);
    }
    try testing.expectEqual(@as(u64, 0), g.req.drafting.skipped);
    // The same drafter on a sampled round takes the table's cut.
    g.greedy = false;
    _ = g.plan(1, 0, 5, 100, 5);
    try testing.expectEqual(@as(usize, 0), g.cut(&.{ -4, -4, -4, -4, -4 }));
}

test "aw1: with FIXED_GREEDY the fixed policy cuts a greedy round as a sampled one, and DRAFT_AHEAD stages while no plain round is owed" {
    const table = try Fixed.parse("41.6,59.9,71.3,79.6,94.8,100.5/5");
    var g = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table, .fixed_greedy = true });
    defer g.deinit();
    var s = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table, .fixed_greedy = true });
    defer s.deinit();
    g.greedy = true;
    for ([_][5]f32{ @splat(9), .{ 9, 9, -4, 9, 9 }, @splat(-4), .{ 6, 5, 4, 3, 2 } }) |conf| {
        _ = g.plan(1, 0, 5, 100, 5);
        _ = s.plan(1, 0, 5, 100, 5);
        try testing.expectEqual(s.cut(&conf), g.cut(&conf));
    }
    try testing.expect(g.wouldStage(1, 0, 5, 100, 5));
    // a backoff owing plain rounds: no stage plan, so no draft launched ahead
    g.req.drafting.owed = 2;
    try testing.expect(!g.wouldStage(1, 0, 5, 100, 5));
    try testing.expect(g.plan(1, 0, 5, 100, 5) == .alone);
}

test "the fixed policy chooses from its table alone: the same confidence and accepts make the same rounds whatever the clock" {
    const table = try Fixed.parse("41.6,59.9,71.3,79.6,94.8,100.5/5");
    var a = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table });
    defer a.deinit();
    var b = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table });
    defer b.deinit();
    const blocks = [_][5]f32{ @splat(9), .{ 9, 9, -4, 9, 9 }, @splat(3), @splat(-4), .{ 6, 5, 4, 3, 2 }, @splat(9) };
    const kept = [_]u32{ 5, 2, 1, 0, 3, 5 };
    for (0..48) |i| {
        const pa = a.plan(1, 0, 5, 100, 5);
        const pb = b.plan(1, 0, 5, 100, 5);
        try testing.expectEqual(std.meta.activeTag(pa), std.meta.activeTag(pb));
        // Never a refresh: the fixed policy has no ladder cell to keep fresh.
        try testing.expect(pa != .alone or a.req.drafting.skipped > 0);
        const conf = &blocks[i % blocks.len];
        const ca = if (pa == .stage) a.cut(conf) else 0;
        const cb = if (pb == .stage) b.cut(conf) else 0;
        try testing.expectEqual(ca, cb);
        const acc = @min(kept[i % kept.len], @as(u32, @intCast(ca)));
        // The clocks disagree (b's ten times a's, a stall now and then): the choices may not.
        const ms: f32 = if (i % 7 == 3) 900 else 60;
        a.learn(if (pa == .stage) .stage else .alone, ca, conf[0..ca], acc, ms, ms + 5);
        b.learn(if (pb == .stage) .stage else .alone, cb, conf[0..cb], acc, 10 * ms, 10 * ms + 50);
    }
    try testing.expectEqual(@as(u64, 0), a.req.refreshes);
    try testing.expectEqual(a.req.drafting.skipped, b.req.drafting.skipped);
    // On the table it is given: a sure block whole, a doubt mid-block cut there, an unsure one t1 alone.
    var c = Drafter.init(testing.allocator, .{ .copy = false, .fixed = table });
    defer c.deinit();
    try testing.expectEqual(@as(usize, 5), c.cut(&.{ 9, 9, 9, 9, 9 }));
    try testing.expectEqual(@as(usize, 2), c.cut(&.{ 9, 9, -4, 9, 9 }));
    try testing.expectEqual(@as(usize, 0), c.cut(&.{ -4, -4, -4, -4, -4 }));
    // A table of three rows never verifies more than two drafts.
    var d = Drafter.init(testing.allocator, .{ .copy = false, .fixed = try Fixed.parse("41.6,59.9,71.3/5") });
    defer d.deinit();
    try testing.expectEqual(@as(usize, 2), d.cut(&.{ 9, 9, 9, 9, 9 }));
}

test "the verify policy cuts the block on the ladder it is given: this plugin's steep verify verifies fewer drafts than the prior" {
    const calib = dflash.Calibration{};
    const prior = dflash.priorCosts(24);
    // About 24 ms at one row, 47.5 at four, 63 at six: each row reads ~6 more distinct experts a layer.
    var steep: dflash.Costs = undefined;
    for (&steep, 0..) |*c, r| c.* = 24 + 7.8 * (@as(f32, @floatFromInt(r)) - 1);
    const mid = [_]f32{ 3, 3, 3, 3, 3 };
    try testing.expectEqual(@as(usize, 3), verifyDrafts(&mid, &calib, &prior));
    try testing.expectEqual(@as(usize, 2), verifyDrafts(&mid, &calib, &steep));
    // A sure head verifies the block on either; an unsure one t1 alone; a doubt mid-block cuts there.
    for ([_]*const dflash.Costs{ &prior, &steep }) |costs| {
        try testing.expectEqual(@as(usize, 5), verifyDrafts(&.{ 9, 9, 9, 9, 9 }, &calib, costs));
        try testing.expectEqual(@as(usize, 0), verifyDrafts(&.{ 0, 0, 0, 0, 0 }, &calib, costs));
        try testing.expectEqual(@as(usize, 2), verifyDrafts(&.{ 9, 9, -4, 9, 9 }, &calib, costs));
    }
}

test "the ladder: the prior until measured, then measured cells in their own ms and the rest the prior's shape bent to them; a first stall moves nothing" {
    var l = Ladder{};
    const prior = dflash.priorCosts(dflash.PRIOR_PLAIN_MS);
    try testing.expectEqualSlices(f32, &prior, &l.costs());
    // A first 6-row verify that compiled its kernels: the cell holds the prior.
    l.sample(6, 900);
    try testing.expectApproxEqRel(prior[6], l.costs()[6], 1e-5);
    for (0..8) |_| {
        l.sample(1, 24);
        l.sample(6, 63);
    }
    const c = l.costs();
    try testing.expectApproxEqAbs(@as(f32, 24), c[1], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 63), c[6], 0.01);
    // Unmeasured rows: the prior's ratio times a deviation linear between the measured rows' (1 at one row), the
    // widest measured row's beyond it.
    const dev6 = 63.0 / 24.0 * prior[1] / prior[6];
    try testing.expectApproxEqRel(24 * (1 + (dev6 - 1) * 2.0 / 5.0) * prior[3] / prior[1], c[3], 1e-4);
    try testing.expectApproxEqRel(63 * prior[8] / prior[6], c[8], 1e-4);
    for (2..dflash.MAX_ROWS + 1) |r| try testing.expect(c[r] >= c[r - 1]);
    // A measured row dearer than the rows above it lifts them.
    for (0..8) |_| l.sample(2, 70);
    const lifted = l.costs();
    try testing.expectApproxEqAbs(@as(f32, 70), lifted[2], 0.01);
    for (3..7) |r| try testing.expect(lifted[r] >= 70);
}

test "the ladder names the cells due a refresh: none before any sample, then each without a fresh sample, asked once a stretch" {
    var l = Ladder{};
    try testing.expect(!l.due(1) and !l.due(6));
    l.sample(6, 18.6);
    try testing.expect(l.due(1) and l.due(2) and !l.due(6));
    l.ask(1);
    try testing.expect(!l.due(1));
    for (0..Ladder.fresh_rounds) |_| l.sample(1, 7.7);
    try testing.expect(!l.due(1) and l.due(6));
    var buf: [128]u8 = undefined;
    try testing.expect(std.mem.endsWith(u8, l.formatCells(&buf), "~"));
    // Its next sample starts the cell over: the one it held (the ratio to a 1-row cell not yet measured) is gone.
    l.sample(6, 18.6);
    try testing.expect(!l.due(6));
    try testing.expectApproxEqRel(@as(f32, 18.6), l.costs()[6], 0.02);
}

test "a 1-row cell gone stale is refreshed by a plain round, which stands for one the backoff owes" {
    var d = Drafter.init(testing.allocator, .{ .copy = false });
    defer d.deinit();
    const sure: [5]f32 = @splat(9);
    // Rounds that verify the whole block and pay never measure the 1-row cell: the next round does.
    d.learn(.stage, 5, &sure, 5, 18.6, 24);
    try testing.expect(d.plan(1, 0, 5, 100, 5) == .alone);
    d.learn(.alone, 0, &.{}, 0, 7.7, 7.7);
    for (0..Ladder.fresh_rounds) |_| {
        try testing.expect(d.plan(1, 0, 5, 100, 5) == .stage);
        d.learn(.stage, 5, &sure, 5, 18.6, 24);
    }
    // Stale again; the backoff owing two rounds, the refresh is one of them.
    d.req.drafting.owed = 2;
    try testing.expect(d.plan(1, 0, 5, 100, 5) == .alone);
    try testing.expectEqual(@as(u32, 1), d.req.drafting.owed);
    try testing.expectEqual(@as(u64, 2), d.req.refreshes);
}

test "a stage round verifies the whole block once its cell is stale, the backoff's probe too, and the backoff does not judge that round" {
    var d = Drafter.init(testing.allocator, .{ .copy = false });
    defer d.deinit();
    const unsure: [5]f32 = @splat(-4);
    // Every row of the block measured cold (three times its warm ms), then t1 alone for fresh_rounds rounds, warm.
    for (1..7) |rows| d.learn(.stage, rows - 1, unsure[0 .. rows - 1], 0, 3 * warm_verify[rows], 3 * warm_verify[rows] + draft_ms);
    for (0..Ladder.fresh_rounds) |_| d.learn(.alone, 0, &.{}, 0, 7.7, 7.7);
    // The stale rows count no dearer than the fresh rows put them: here the anchor alone, in the prior's shape.
    const prior = dflash.priorCosts(dflash.PRIOR_PLAIN_MS);
    for (2..7) |r| try testing.expect(d.ladder.costs()[r] <= 7.7 * prior[r] / prior[1] * 1.001);
    // The backoff's probe drafts: its block is verified whole, and the backoff's debt and period stay as they were.
    d.req.drafting = .{ .owed = 1, .period = 4 };
    try testing.expect(d.plan(1, 0, 5, 100, 5) == .alone);
    try testing.expect(d.plan(1, 0, 5, 100, 5) == .stage);
    try testing.expectEqual(@as(usize, 5), d.cut(&unsure));
    d.learn(.stage, 5, &unsure, 0, 18.6, 24);
    try testing.expectEqual(@as(u32, 0), d.req.drafting.owed);
    try testing.expectEqual(@as(u32, 4), d.req.drafting.period);
    // Fresh again, the next round is the policy's: it cuts an unsure block to t1 alone, and the backoff judges it.
    try testing.expect(d.plan(1, 0, 5, 100, 5) == .stage);
    try testing.expectEqual(@as(usize, 0), d.cut(&unsure));
    d.learn(.stage, 0, &.{}, 0, 7.7, 13.1);
    try testing.expectEqual(@as(u32, 4), d.req.drafting.owed);
    try testing.expectEqual(@as(u64, 1), d.req.refreshes);
}

test "the ladder after a cold start: the rows the backoff stopped verifying follow the 1-row cell down" {
    var l = Ladder{};
    // Cold first rounds (expert reads, kernel compiles): one row ~30 ms, a few multi-row verifies dearer still.
    for ([_]struct { usize, f32 }{ .{ 1, 30 }, .{ 2, 40 }, .{ 1, 30 }, .{ 3, 45 }, .{ 3, 45 }, .{ 1, 30 }, .{ 5, 50 }, .{ 5, 50 } }) |s| l.sample(s[0], s[1]);
    // Then the backoff verifies t1 alone, warm.
    for (0..60) |_| l.sample(1, 7.7);
    // Its probes verify a block now and then, as a full-block run on the same pack measured it.
    for (0..2) |_| {
        l.sample(2, 15.5);
        l.sample(6, 18.6);
    }
    // The stale ladder held 1:7.7,2:27.7,3:32.2,5:35.8 here, and verified t1 alone after every confident block.
    const c = l.costs();
    try testing.expectApproxEqAbs(@as(f32, 7.7), c[1], 0.1);
    try testing.expectApproxEqRel(@as(f32, 15.5), c[2], 0.1);
    try testing.expectApproxEqRel(@as(f32, 18.6), c[6], 0.1);
    for (3..6) |r| try testing.expect(c[r] <= 18.6 * 1.1);
    const calib = dflash.Calibration{};
    try testing.expect(verifyDrafts(&.{ 3, 3, 3, 3, 3 }, &calib, &c) >= 3);
}

test "the ladder follows a level change within ten samples, and one stall moves a warm cell little" {
    var l = Ladder{};
    for (0..16) |_| {
        l.sample(1, 7.7);
        l.sample(6, 18.6);
    }
    l.sample(6, 900);
    try testing.expectApproxEqRel(@as(f32, 18.6), l.costs()[6], 0.03);
    // The 6-row verify gets 40% cheaper (its experts stay resident), then dearer again (the context grew).
    for (0..10) |_| l.sample(6, 11.2);
    try testing.expectApproxEqRel(@as(f32, 11.2), l.costs()[6], 0.03);
    for (0..10) |_| l.sample(6, 16);
    try testing.expectApproxEqRel(@as(f32, 16), l.costs()[6], 0.03);
}

test "the ladder re-anchors: when one row gets cheaper, the rows no round verifies follow it" {
    var l = Ladder{};
    for (0..16) |_| {
        l.sample(1, 15.4);
        l.sample(6, 37.2);
    }
    // Everything halves (the box warmed) while only one-row verifies run (the backoff), ten of them.
    for (0..10) |_| l.sample(1, 7.7);
    try testing.expectApproxEqRel(@as(f32, 7.7), l.costs()[1], 0.03);
    try testing.expectApproxEqRel(@as(f32, 18.6), l.costs()[6], 0.05);
}

/// Warm verify ms by rows as a full-block run on an 8-layer pack measured them, and the draft's.
const warm_verify = [_]f32{ 0, 7.7, 15.5, 16.3, 17.0, 17.8, 18.6 };
const draft_ms: f32 = 5.4;

/// One round of `d` over a 5-draft block whose confidence logits are all `conf`: the trunk keeps every draft it
/// verifies when `keep` is set, none otherwise; `slow` scales every ms (a cold box). Returns the drafts it verified.
fn simRound(d: *Drafter, conf: f32, keep: bool, slow: f32) usize {
    const block: [5]f32 = @splat(conf);
    switch (d.plan(1, 0, block.len, 100, block.len)) {
        .copy => unreachable,
        .alone => {
            const ms = warm_verify[1] * slow;
            d.learn(.alone, 0, &.{}, 0, ms, ms);
            return 0;
        },
        .stage => {
            const drafts = d.cut(&block);
            const ms = warm_verify[drafts + 1] * slow;
            d.learn(.stage, drafts, block[0..drafts], if (keep) @intCast(drafts) else 0, ms, ms + draft_ms * slow);
            return drafts;
        },
    }
}

test "the policy recovers from a stale ladder: after a cold start and a stretch the backoff skipped, a confident block is verified whole again" {
    var d = Drafter.init(testing.allocator, .{ .copy = false });
    defer d.deinit();
    // Ten cold rounds (every verify three times its warm ms), confident and kept.
    for (0..10) |_| _ = simRound(&d, 9, true, 3);
    // A stretch whose drafts lose (unsure and rejected): the backoff verifies t1 alone, warm.
    for (0..150) |_| _ = simRound(&d, -4, false, 1);
    // Confident and kept again: from the backoff's next probe (within its longest period) on, the rounds verify the
    // whole block (the stale ladder verified t1 alone in every one).
    var whole: usize = 0;
    var first: ?usize = null;
    for (0..200) |i| if (simRound(&d, 9, true, 1) == 5) {
        whole += 1;
        if (first == null) first = i;
    };
    try testing.expect(first != null and first.? <= 32 and whole >= 150);
}

test "a round copies where the context holds a match, else verifies t1 alone while the backoff owes, else drafts; each switch turns its feature off" {
    const q = Quote.of(21);
    var d = Drafter.init(testing.allocator, .{});
    defer d.deinit();
    d.begin(q.ids());
    const first = d.plan(q.len, q.p[21], 7, 100, 5);
    try testing.expect(first == .copy);
    try testing.expectEqualSlices(u32, q.p[22..29], first.copy);
    try testing.expect(d.plan(q.len, 4000, 7, 100, 5) == .stage);
    // No draft can be kept: t1 alone.
    try testing.expect(d.plan(q.len, q.p[21], 0, 0, 5) == .alone);
    // A stage round that cost far more than the plain rounds its tokens replace owes one plain round (which also
    // measures the 1-row cell, due its first sample).
    d.learn(.stage, 5, &.{ 1, 1, 1, 1, 1 }, 0, 40, 400);
    try testing.expect(d.plan(q.len, 4000, 7, 100, 5) == .alone);
    try testing.expectEqual(@as(u32, 0), d.req.drafting.owed);
    // The backoff's probe drafts, and cuts an unsure block to t1 alone.
    try testing.expect(d.plan(q.len, 4000, 7, 100, 5) == .stage);
    try testing.expectEqual(@as(usize, 0), d.cut(&.{ 0, 0, 0, 0, 0 }));
    var off = Drafter.init(testing.allocator, .{ .copy = false, .policy = false });
    defer off.deinit();
    off.begin(q.ids());
    try testing.expect(off.plan(q.len, q.p[21], 7, 100, 5) == .stage);
    off.learn(.stage, 5, &.{ 1, 1, 1, 1, 1 }, 0, 40, 400);
    try testing.expect(off.plan(q.len, 4000, 7, 100, 5) == .stage);
    try testing.expectEqual(@as(usize, 5), off.cut(&.{ 0, 0, 0, 0, 0 }));
}
