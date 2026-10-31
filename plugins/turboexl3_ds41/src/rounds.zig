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

/// The two features (the server's environment, read at load): copy rounds, and the verify policy with its backoff.
pub const Switches = struct { copy: bool = true, policy: bool = true };

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
        if (max == 0) return .alone;
        if (self.on.copy) {
            const c = copyDrafts(&self.history, n, t1, max, remaining, block);
            if (c.len > 0) return .{ .copy = c };
        }
        if (!self.on.policy) return .stage;
        // The anchor goes stale only while every round verifies more than t1 (the backoff skipping none): a plain round
        // refreshes it, and stands for one the backoff owes when it owes any.
        if (self.ladder.due(1)) {
            if (self.req.drafting.owed > 0) _ = self.req.drafting.shouldDraft();
            self.ladder.ask(1);
            self.req.refreshes += 1;
            return .alone;
        }
        if (!self.req.drafting.shouldDraft()) return .alone;
        return .stage;
    }

    /// The drafts of a stage block a round verifies, `conf` the block's confidence logits: the policy's cut, or all.
    /// All as well when the whole block's cell is due a refresh (`Ladder.due`), on any stage round, the backoff's probes
    /// (the rounds that draft within a skip stretch) included, so it never drafts a round the backoff would not. The
    /// whole block is the end the rows between it and the anchor are interpolated from, and a wider verify only adds
    /// tokens to the policy's. No narrower row is refreshed: on a probe that would cost the probe its tokens, and with
    /// them the backoff's way back (a probe that verified two rows of a block worth six lost, and kept the backoff at
    /// its longest period); a stale row counts no dearer than the fresh rows around it instead (`Ladder.costs`).
    pub fn cut(self: *Drafter, conf: []const f32) usize {
        if (!self.on.policy) return conf.len;
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
                if (!self.on.policy) return;
                for (conf, 0..) |c, k| {
                    if (k > accepted) break;
                    q.calib.observe(sigmoid(c), k < accepted);
                }
                if (!self.refreshing) q.drafting.record(@as(f32, @floatFromInt(accepted + 1)) * self.ladder.costs()[1], wall_ms);
            },
        }
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
            q.stage_rounds, q.copy_rounds, q.copy_drafted, q.copy_accepted, q.alone_rounds, onOff(self.on.copy), onOff(self.on.policy),
        }) catch return w.buffered();
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
