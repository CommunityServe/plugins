//! The DSpark round's copy arm: when the committed tail plus the round's first token occurred earlier in the prompt or
//! output, the tokens that followed are the draft (an echoed file, an edit's unchanged lines), with no stage forward.
//! Pure data, no MLX. Ported from mlx-serve's `spec/mtp_lookup.zig` (`Index`, `matchBest`) and `spec/verify.zig`
//! (`dsparkCopyLen`), which the built-in `deepseek_v41` engine runs before every stage draft.
const std = @import("std");

pub const NGRAM: usize = 4;
/// A 16-token agreement marks a quoted span; a shorter one lands too few drafts to pay for the wider verify.
pub const STRONG_SUFFIX: u32 = 16;
const SUFFIX_CAP: u32 = 64;
/// Earlier occurrences `matchBest` compares.
const CANDIDATES: u32 = 8;

pub const Match = struct {
    /// Borrowed from the index.
    draft: []const u32,
    /// Tokens the two sites agree on going back, n-gram included, capped.
    suffix: u32,
};

pub const Index = struct {
    toks: std.ArrayList(u32) = .empty,
    /// n-gram -> position of the token after its latest occurrence.
    next: std.AutoHashMapUnmanaged(u128, u32) = .empty,
    /// Position of the token after an earlier occurrence of the same n-gram, by position of the later one.
    prev: std.ArrayList(u32) = .empty,

    const none: u32 = std.math.maxInt(u32);

    pub fn deinit(self: *Index, gpa: std.mem.Allocator) void {
        self.toks.deinit(gpa);
        self.next.deinit(gpa);
        self.prev.deinit(gpa);
    }

    /// Forgets every token (a new request, or a restored prefix whose ids replace the history).
    pub fn clear(self: *Index) void {
        self.toks.clearRetainingCapacity();
        self.next.clearRetainingCapacity();
        self.prev.clearRetainingCapacity();
    }

    /// Indexes the ids `history` holds past the ones already indexed; `history` must extend what was indexed.
    pub fn sync(self: *Index, gpa: std.mem.Allocator, history: []const u32) !void {
        if (history.len < self.toks.items.len) self.clear();
        try self.extend(gpa, history[self.toks.items.len..]);
    }

    pub fn extend(self: *Index, gpa: std.mem.Allocator, ids: []const u32) !void {
        try self.toks.ensureUnusedCapacity(gpa, ids.len);
        for (ids) |id| {
            self.toks.appendAssumeCapacity(id);
            const n = self.toks.items.len;
            if (n <= NGRAM) continue;
            const g = self.toks.items[n - 1 - NGRAM .. n - 1];
            const gop = try self.next.getOrPut(gpa, key(g[0], g[1], g[2], g[3]));
            try self.prev.resize(gpa, n);
            self.prev.items[n - 1] = if (gop.found_existing) gop.value_ptr.* else none;
            gop.value_ptr.* = @intCast(n - 1);
        }
    }

    /// The continuation after the earlier occurrence of the last NGRAM-1 committed tokens plus `t1` (not yet
    /// committed) that agrees furthest back, among the last `CANDIDATES` occurrences; the latest among equals. An edit
    /// that left the latest occurrence on the wrong copy still finds the right one.
    pub fn matchBest(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        var p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        var best: ?Match = null;
        var seen: u32 = 0;
        while (seen < CANDIDATES) : (seen += 1) {
            const suffix = agreement(toks, p, t1);
            if (best == null or suffix > best.?.suffix) best = .{ .draft = toks[p..@min(p + max_draft, toks.len)], .suffix = suffix };
            if (suffix >= SUFFIX_CAP or p >= self.prev.items.len or self.prev.items[p] == none) break;
            p = self.prev.items[p];
        }
        return best;
    }

    /// Tokens before `p` that agree with the committed tail (plus `t1`), capped.
    fn agreement(toks: []const u32, p: usize, t1: u32) u32 {
        var suffix: u32 = 0;
        var a: usize = p;
        var b: usize = toks.len + 1;
        while (suffix < SUFFIX_CAP and a > 0) : (suffix += 1) {
            a -= 1;
            b -= 1;
            if (toks[a] != (if (b == toks.len) t1 else toks[b])) break;
        }
        return suffix;
    }

    fn key(a: u32, b: u32, c: u32, d: u32) u128 {
        return (@as(u128, a) << 96) | (@as(u128, b) << 64) | (@as(u128, c) << 32) | d;
    }
};

/// When a request stops taking copies: a copy round verifies up to 16 rows, so one that is rejected early costs far more
/// than the stage round it replaced. Each copy round is priced against the stage rounds' own ms per token; copies that
/// lose back the next matches off (skipped, the stage drafts instead), doubling to a cap, as the rows policy backs off
/// stage drafts that lose to plain steps.
pub const Backoff = struct {
    /// Matches to skip before the next copy, and the stretch the next losing stretch of copies will owe.
    owed: u32 = 0,
    period: u32 = 1,
    /// Decayed ms a copy round saved against stage rounds for its tokens; negative while copies lose.
    edge: f32 = 0,
    /// Stage rounds' ms per token, smoothed (the plugin's DSpark rounds at a short context land near this).
    stage_ms_per_token: f32 = 12,
    /// Copies kept a copy round, smoothed, and whether the last copy round kept every draft: a round verifies what
    /// copies have been landing plus 2, or the whole cap right after one landed whole (mlx-serve's `mtp_lookup.gate`).
    kept: f32 = 5,
    streak: bool = false,

    const max_period: u32 = 16;
    const keep: f32 = 0.8;

    /// Drafts the next copy round may verify, of `cap`: sized to what copies have been landing, so one that is rejected
    /// early wastes a few rows, not 16. At or under the stage block, `copyLen` leaves the round to the stage draft.
    pub fn width(self: *const Backoff, cap: usize) usize {
        if (self.streak) return cap;
        return @min(cap, @as(usize, @intFromFloat(@ceil(@max(self.kept, 0)))) + 2);
    }

    /// A match worth copying turned up: whether to take it.
    pub fn take(self: *Backoff) bool {
        if (self.owed == 0) return true;
        self.owed -= 1;
        return false;
    }

    /// A stage round (or a plain step) that committed `tokens` in `ms`.
    pub fn observeStage(self: *Backoff, tokens: usize, ms: f32) void {
        if (tokens == 0 or !std.math.isFinite(ms) or ms <= 0) return;
        self.stage_ms_per_token = keep * self.stage_ms_per_token + (1 - keep) * ms / @as(f32, @floatFromInt(tokens));
    }

    /// A copy round that verified `drafted` copies, kept `kept` of them and took `ms`.
    pub fn observeCopy(self: *Backoff, drafted: usize, kept: usize, ms: f32) void {
        self.kept = 0.7 * self.kept + 0.3 * @as(f32, @floatFromInt(kept));
        self.streak = kept == drafted;
        if (!std.math.isFinite(ms) or ms <= 0) return;
        const saved = @as(f32, @floatFromInt(1 + kept)) * self.stage_ms_per_token - ms;
        self.edge = keep * self.edge + (1 - keep) * saved;
        if (self.edge < 0) {
            self.owed = self.period;
            self.period = @min(self.period * 2, max_period);
        } else self.period = 1;
    }
};

/// Drafts a copy round takes from a match, or 0 for the stage draft. The stage block already lands most drafts on
/// repeated text, so only a quoted span (a long agreement) whose copy outruns the block is worth the wider verify.
pub fn copyLen(found: ?Match, cap: usize, block: usize) usize {
    const f = found orelse return 0;
    if (f.suffix < STRONG_SUFFIX) return 0;
    const n = @min(f.draft.len, cap);
    return if (n > block) n else 0;
}

const testing = std.testing;

test "sushi_ds41 copy: latest earlier continuation, keyed on the tail plus t1, with how far back it agrees" {
    var idx: Index = .{};
    defer idx.deinit(testing.allocator);
    // Two earlier "1 2 3 4" sites; the later one agrees back through "0 11 12 13".
    try idx.extend(testing.allocator, &.{ 1, 2, 3, 4, 10, 50, 0, 11, 12, 13, 1, 2, 3, 4, 77, 5 });
    try idx.extend(testing.allocator, &.{ 60, 0, 11, 12, 13, 1, 2, 3 });
    const m = idx.matchBest(4, 3) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 77, 5, 60 }, m.draft);
    try testing.expectEqual(@as(u32, 8), m.suffix);
    try testing.expect(idx.matchBest(9, 3) == null);
}

test "sushi_ds41 copy: an edit that moved the latest site still copies the site that agrees further back" {
    var idx: Index = .{};
    defer idx.deinit(testing.allocator);
    // The quoted span "20 21 22 23 1 2 3 4 5 6" appears first; a later "1 2 3 4" (after other text) continues 99.
    try idx.extend(testing.allocator, &.{ 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 40, 41, 1, 2, 3, 4, 99 });
    try idx.extend(testing.allocator, &.{ 20, 21, 22, 23, 1, 2, 3 });
    const m = idx.matchBest(4, 2) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 5, 6 }, m.draft);
}

test "sushi_ds41 copy: sync indexes only the new ids and starts over on a shorter history" {
    var idx: Index = .{};
    defer idx.deinit(testing.allocator);
    const hist = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3 };
    try idx.sync(testing.allocator, hist[0..6]);
    try idx.sync(testing.allocator, &hist);
    try testing.expectEqual(hist.len, idx.toks.items.len);
    try testing.expectEqualSlices(u32, &.{ 5, 6 }, (idx.matchBest(4, 2) orelse return error.NoMatch).draft);
    try idx.sync(testing.allocator, &.{ 9, 9 });
    try testing.expectEqual(@as(usize, 2), idx.toks.items.len);
    try testing.expect(idx.matchBest(4, 2) == null);
}

test "sushi_ds41 copy: copies that lose to stage rounds back off, doubling; a winning copy clears the stretch" {
    var b: Backoff = .{};
    for (0..10) |_| b.observeStage(3, 30); // stage rounds: 10 ms a token
    try testing.expect(b.take());
    // A 16-row copy rejected after 2 drafts: 3 tokens in 60 ms, 30 ms worse than the stage rounds would have taken.
    b.observeCopy(15, 2, 60);
    try testing.expectEqual(@as(u32, 1), b.owed);
    try testing.expect(!b.take());
    try testing.expect(b.take());
    b.observeCopy(15, 2, 60);
    try testing.expectEqual(@as(u32, 2), b.owed);
    // A copy kept whole: 16 tokens in 60 ms, 100 ms better; the edge turns positive and the next loss owes one again.
    b.owed = 0;
    for (0..3) |_| b.observeCopy(15, 15, 60);
    try testing.expect(b.edge > 0);
    try testing.expectEqual(@as(u32, 1), b.period);
    try testing.expect(b.take());
}

test "sushi_ds41 copy: a round verifies what copies have been landing plus 2, the cap after one landed whole" {
    var b: Backoff = .{};
    try testing.expectEqual(@as(usize, 7), b.width(15)); // prior: 5 kept
    b.observeCopy(7, 7, 50);
    try testing.expectEqual(@as(usize, 15), b.width(15)); // landed whole
    b.observeCopy(15, 1, 50);
    try testing.expect(!b.streak);
    // 0.7 * (0.7 * 5 + 0.3 * 7) + 0.3 * 1 = 4.22, so 5 + 2: still a copy of more than the 5-row block.
    try testing.expectEqual(@as(usize, 7), b.width(15));
    for (0..4) |_| b.observeCopy(7, 1, 50);
    // Copies keep about 1: the width falls to the block, so `copyLen` leaves rounds to the stage draft.
    try testing.expect(b.width(15) <= 5);
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = &(@as([15]u32, @splat(0))), .suffix = STRONG_SUFFIX }, b.width(15), 5));
}

test "sushi_ds41 copy: only a long agreement that outruns the stage block is copied" {
    const draft: [15]u32 = @splat(0);
    try testing.expectEqual(@as(usize, 15), copyLen(.{ .draft = &draft, .suffix = STRONG_SUFFIX }, 400, 5));
    try testing.expectEqual(@as(usize, 9), copyLen(.{ .draft = &draft, .suffix = STRONG_SUFFIX }, 9, 5));
    try testing.expectEqual(@as(usize, 0), copyLen(null, 400, 5));
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = &draft, .suffix = STRONG_SUFFIX - 1 }, 400, 5));
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = draft[0..5], .suffix = STRONG_SUFFIX }, 400, 5));
}
