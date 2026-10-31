//! The DSpark rounds' choices, MLX-free (plugin.zig runs the rounds). The engine makes them for its built-in
//! deepseek_v41 only (`engines/mlx/spec/dspark.zig` `moduleRound`), so this drafter makes them itself, with the
//! engine's own code (the copies in src/mlx_serve/engines/mlx/spec):
//! - a copy round verifies the context's own continuation after `t1` (`mtp_lookup.Index.matchBest`,
//!   `verify.dsparkCopyLen`), with no stage forward;
//! - a stage round verifies the prefix of the stages' block whose calibrated confidence buys the most tokens per ms of
//!   verify (`dflash/policy.zig`), on the verify ladder this model measures (`Ladder`);
//! - where drafting does not pay, the drafting backoff skips the stages and the round verifies `t1` alone.
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

/// Verify ms by rows (`dflash.Costs`, index = rows) as this model measures its own verifies, each cell a
/// `dflash.PlainCost` (a stall or a kernel compile moves it little); a row count not yet measured is the engine's prior
/// ladder (`dflash.priorCosts`) in the measured cells' unit, the prior itself before any.
pub const Ladder = struct {
    cells: [dflash.MAX_ROWS + 1]?dflash.PlainCost = @splat(null),

    pub fn costs(self: *const Ladder) dflash.Costs {
        const prior = dflash.priorCosts(dflash.PRIOR_PLAIN_MS);
        var have: f32 = 0;
        var want: f32 = 0;
        for (self.cells[1..], prior[1..]) |cell, p| if (cell) |c| {
            have += c.ms;
            want += p;
        };
        const unit: f32 = if (want > 0) have / want else 1;
        var out: dflash.Costs = undefined;
        out[0] = 0;
        for (1..dflash.MAX_ROWS + 1) |r| {
            const ms = if (self.cells[r]) |c| c.ms else prior[r] * unit;
            // Never cheaper for more rows: a row measured dear lifts the unmeasured rows above it.
            out[r] = if (r == 1) ms else @max(ms, out[r - 1]);
        }
        return out;
    }

    /// A verify of `rows` rows took `ms`; a cell's first sample starts it from what the ladder held for that row.
    pub fn sample(self: *Ladder, rows: usize, ms: f32) void {
        if (rows == 0 or rows > dflash.MAX_ROWS) return;
        if (self.cells[rows] == null) {
            // Read before the cell is set: the cell is the assignment's result location, which `costs` would read.
            const start = self.costs()[rows];
            self.cells[rows] = .{ .ms = start };
        }
        self.cells[rows].?.sample(ms);
    }

    /// `rows:ms` of the measured cells, for `[spec-stats]`.
    pub fn formatCells(self: *const Ladder, buf: []u8) []const u8 {
        var w = std.Io.Writer.fixed(buf);
        var first = true;
        for (self.cells, 0..) |cell, rows| if (cell) |c| {
            w.print("{s}{d}:{d:.1}", .{ if (first) "" else ",", rows, c.ms }) catch break;
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
};

/// The drafter's state beside its model: the model's history and ladder, the policy of the request it serves.
pub const Drafter = struct {
    on: Switches,
    history: History,
    ladder: Ladder = .{},
    req: Request = .{},

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
    /// draft can be kept or the backoff skips the stages, else the stages' draft. A verify carries and the round keeps
    /// at most `max` drafts; `remaining` is the tokens still allowed after `t1`, `block` the stages' block.
    pub fn plan(self: *Drafter, n: usize, t1: u32, max: usize, remaining: u32, block: usize) Plan {
        if (max == 0) return .alone;
        if (self.on.copy) {
            const c = copyDrafts(&self.history, n, t1, max, remaining, block);
            if (c.len > 0) return .{ .copy = c };
        }
        if (self.on.policy and !self.req.drafting.shouldDraft()) return .alone;
        return .stage;
    }

    /// The drafts of a stage block a round verifies, `conf` the block's confidence logits: the policy's cut, or all.
    pub fn cut(self: *const Drafter, conf: []const f32) usize {
        if (!self.on.policy) return conf.len;
        const costs = self.ladder.costs();
        return verifyDrafts(conf, &self.req.calib, &costs);
    }

    /// A round kept `accepted` of the `drafts` it verified (a stage round's confidence logits in `conf`, else empty);
    /// its verify took `verify_ms`, the whole round `wall_ms`. A stage round teaches the calibration each position it
    /// reached, and the backoff what its tokens would have cost as plain rounds against what it took.
    pub fn learn(self: *Drafter, kind: Kind, drafts: usize, conf: []const f32, accepted: u32, verify_ms: f32, wall_ms: f32) void {
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
                q.drafting.record(@as(f32, @floatFromInt(accepted + 1)) * self.ladder.costs()[1], wall_ms);
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
        }
        if (q.stage_rounds + q.copy_rounds + q.alone_rounds == 0) return &.{};
        var w = std.Io.Writer.fixed(buf);
        w.print("[spec-stats] dspark rounds: stage_rounds={d} copy_rounds={d} copy_drafted={d} copy_accepted={d} alone_rounds={d} copy={s} policy={s}\n", .{
            q.stage_rounds, q.copy_rounds, q.copy_drafted, q.copy_accepted, q.alone_rounds, onOff(self.on.copy), onOff(self.on.policy),
        }) catch return w.buffered();
        var acc: [128]u8 = undefined;
        var rows: [160]u8 = undefined;
        var cells: [256]u8 = undefined;
        w.print("[spec-stats] dspark policy: skipped={d} probes={d} plain_ms={d:.2} acc_pos={s} rows_hist={s} ladder_ms={s}\n", .{
            q.drafting.skipped, q.drafting.probes, self.ladder.costs()[1], q.stats.formatAcc(&acc), q.stats.formatRows(&rows), self.ladder.formatCells(&cells),
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

test "the ladder: the prior until measured, then measured cells in their own ms and the rest in their unit; a stall moves a cell little" {
    var l = Ladder{};
    const prior = dflash.priorCosts(dflash.PRIOR_PLAIN_MS);
    try testing.expectEqualSlices(f32, &prior, &l.costs());
    // A first 6-row verify that compiled its kernels: the cell starts from the prior and moves a tenth of a quarter.
    l.sample(6, 900);
    const stalled = l.costs()[6];
    try testing.expect(stalled > prior[6] and stalled <= prior[6] * 1.026);
    for (0..200) |_| {
        l.sample(1, 24);
        l.sample(6, 63);
    }
    const c = l.costs();
    try testing.expectApproxEqAbs(@as(f32, 24), c[1], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 63), c[6], 0.01);
    const unit = (c[1] + c[6]) / (prior[1] + prior[6]);
    try testing.expectApproxEqAbs(prior[3] * unit, c[3], 0.01);
    for (2..dflash.MAX_ROWS + 1) |r| try testing.expect(c[r] >= c[r - 1]);
    // A measured row dearer than the prior above it lifts those rows to it.
    for (0..200) |_| l.sample(2, 50);
    const lifted = l.costs();
    try testing.expectApproxEqAbs(@as(f32, 50), lifted[3], 0.05);
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
    // A stage round that cost far more than the plain rounds its tokens replace owes one plain round.
    d.learn(.stage, 5, &.{ 1, 1, 1, 1, 1 }, 0, 40, 400);
    try testing.expect(d.plan(q.len, 4000, 7, 100, 5) == .alone);
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
