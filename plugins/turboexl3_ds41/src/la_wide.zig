//! `MLX_SERVE_DSV41_LOOKAHEAD_WIDE` (read at load, before the stream opens; unset or empty: 10; `0`: off; `N`, 1 to 64;
//! anything else is refused by name; with `MLX_SERVE_DSV41_LOOKAHEAD`'s predictor on): the decode lookahead's recall
//! widened by about N records a round past its per-call budget. Merged as measured (zzz9c; the rule: both point
//! estimates >= 0), not as a lever: its extra records go unread (the read below).
//!
//! The decode's lookahead (src/stream_lookahead.zig) reads ahead, at each layer call, the first `budget` (2) experts of
//! the next layer's predicted ranking (each verify row's top 8 on the next layer's gate over this layer's input, ordered
//! by rank over the rows, then score) that the next layer does not hold. On ctl26 (w-20261102-0204 and -0228, the a
//! arms, rounds 2 on) a greedy round has 36.2 misses, 13.9 of them claimed from the lookahead and 22.3 read on demand
//! (12.4 MB each, about 1.0 ms of verify each: 80 ms/GB); the lookahead reads 617 MB a round, 412 of it unclaimed, and
//! those bytes show no verify cost (the round fit's unclaimed term is -15 ms/GB beside the demand reads). The drive
//! moves 10.4 GB/s in decode against the ~13 GB/s it streams a prompt at: about 240 MB a round to spare, ~19 records.
//!
//! On, the stream is sized for the lookahead's widest budget (4: eight staging slots, 51 MiB more in the bill) and a
//! layer call reads its `budget` and, while a credit lasts, up to 2 more of the same ranking: the credit grows N /
//! (layers - 1) at each predicting call (about N a round) and spends one per record read past the budget, at most 2
//! held. The b arms' N is 10, half that spare (the speculative class's queue rule still yields the drive to demand
//! reads). The extra records go out behind the budget's in the same step, so a staging-slot shortfall skips an extra
//! record, never a budget one.
//!
//! On or off, a `[lookahead]` line at each request's end: the records read ahead, the budget's and the extra ones, by
//! the top-k position each was chosen at, and of them those the target call routed (`p`/`r`, `xp`/`xr`; a routed record
//! is claimed when it is in flight or landed, one still queued is cancelled and read on demand). Exact: the lookahead
//! only reads ahead; the served bytes are the same at every setting.
//!
//! The read (la2, M5 Max, w-20261102-0435; code-ctx16384 card then greedy, a = ctl29 6e9c8b0a279a, b = la2 52b6b44a6398 at
//! 10; a1 b1 b2 a2 b3 a3 a4 b4): card 44.50 44.70 44.70 44.40 against 44.50 44.70 44.70 44.70 tok/s (+0.17%; pairs 0.00,
//! 0.00, 0.00, +0.68), greedy 57.10 56.10 57.20 57.10 against 57.00 56.90 56.80 56.90 (+0.04%; pairs -0.18, +1.43, -0.70,
//! -0.35; a2 a slow outlier at 87.01 ms a round); the texts the same bytes in every arm, 150 prompt / 179 decode rows in
//! every arm. The extra records were chosen (card 10.1 a round, greedy 9.8; 17% / 10% of them routed by their target
//! call) but hardly read: against the a arms the b arms' records issued rose by exactly them (card 8122 -> 9261 a
//! request, greedy 7094 -> 8145) while records landed did not move (2643 -> 2650, 1918 -> 1916), claims a round barely
//! (20.14 -> 20.25, 14.16 -> 14.30) and the speculative bytes a round by 3 / 4 MB (640 -> 643, 617 -> 621 MB): the
//! speculative class's two threads were busy with the budget's records (70 / 66 issued a round, 640 / 617 MB read, 369 /
//! 409 MB of it abandoned unrouted), so the extras waited queued behind them and were dropped unstarted at their call's
//! settle, or, routed, cancelled by the demand read that needed them (a queued record is read on demand). Misses a
//! round did not move (56.9 card, 36.9 greedy).

const std = @import("std");

/// Records a round read ahead past the per-call budget (`configure`, at load): 0 off.
pub var per_round: u32 = default_per_round;
/// The setting unset or empty (merged on: zzz9c).
pub const default_per_round: u32 = 10;
/// The most records past the budget a layer call reads (the selector's `max_budget` 4 less the default budget 2).
pub const max_extra: u32 = 2;
/// The top-k positions the histogram keeps (the selector's `max_k`).
pub const max_rank = 12;
/// The positions the `[lookahead]` line prints (the stream's `k`, 8).
pub const shown_ranks = 8;

pub const Error = error{LookaheadWideSetting};

pub fn configure(value: ?[]const u8) Error!void {
    per_round = try parse(value);
}

/// `value`: null or empty the default (10); `0` off (0); a decimal 1 to 64; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!u32 {
    const v = value orelse return default_per_round;
    if (v.len == 0) return default_per_round;
    for (v) |c| if (c < '0' or c > '9') return error.LookaheadWideSetting;
    const n = std.fmt.parseInt(u32, v, 10) catch return error.LookaheadWideSetting;
    if (n > 64) return error.LookaheadWideSetting;
    return n;
}

pub fn on() bool {
    return per_round > 0;
}

/// A layer call's credit, held by the stream: `grow` at each predicting call, `take` the records past the budget the
/// call may read, `spend` the ones it read.
pub const Credit = struct {
    /// Credit grown a predicting call (N / (layers - 1)).
    rate: f32 = 0,
    held: f32 = 0,

    pub fn of(n: u32, layers: usize) Credit {
        return .{ .rate = if (n == 0 or layers < 2) 0 else @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(layers - 1)) };
    }
    /// The call's records past the budget, at most `room` (the selector's room past the budget).
    pub fn take(c: *Credit, room: u32) u32 {
        if (c.rate == 0) return 0;
        c.held = @min(c.held + c.rate, @as(f32, @floatFromInt(max_extra)));
        return @min(room, @as(u32, @intFromFloat(@floor(c.held))));
    }
    /// The call read `n` records past the budget.
    pub fn spend(c: *Credit, n: u32) void {
        c.held = @max(c.held - @as(f32, @floatFromInt(n)), 0);
    }
};

/// The stream's histogram: [extra][top-k position] = {read ahead, routed by the target call}.
pub const Hist = [2][max_rank][2]u64;

var last: Hist = @splat(@splat(.{ 0, 0 }));

/// The request's `[lookahead]` line from the stream's histogram `now` (cumulative since load): this request's counts.
pub fn meters(now: Hist, buf: []u8) []const u8 {
    var d: Hist = undefined;
    for (0..2) |x| for (0..max_rank) |r| for (0..2) |k| {
        d[x][r][k] = now[x][r][k] -| last[x][r][k];
    };
    last = now;
    var tot: [2][2]u64 = @splat(.{ 0, 0 });
    for (0..2) |x| for (0..max_rank) |r| for (0..2) |k| {
        tot[x][k] += d[x][r][k];
    };
    if (tot[0][0] + tot[1][0] == 0) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[lookahead] wide={d} p={d} r={d} xp={d} xr={d}", .{ per_round, tot[0][0], tot[0][1], tot[1][0], tot[1][1] }) catch return buf[0..0];
    for (0..shown_ranks) |r| w.print(" p{d}={d} r{d}={d}", .{ r, d[0][r][0], r, d[0][r][1] }) catch return buf[0..0];
    for (0..shown_ranks) |r| w.print(" xp{d}={d} xr{d}={d}", .{ r, d[1][r][0], r, d[1][r][1] }) catch return buf[0..0];
    w.print("\n", .{}) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "lookahead wide setting: unset or empty 10; 0 off; 1 to 64 on; anything else refused by name" {
    for ([_]?[]const u8{ null, "" }) |v| try testing.expectEqual(@as(u32, 10), try parse(v));
    try testing.expectEqual(@as(u32, 0), try parse("0"));
    try testing.expectEqual(@as(u32, 10), try parse("10"));
    try testing.expectEqual(@as(u32, 64), try parse("64"));
    for ([_][]const u8{ "65", "-1", "+2", "10 ", "on", "1.5" }) |v| try testing.expectError(error.LookaheadWideSetting, parse(v));
}

test "lookahead wide credit: about N a round over the predicting calls, at most 2 a call, spent by the reads" {
    // 10 a round over 43 layers (42 predicting calls): 10 records past the budget a round when every one is read.
    var c = Credit.of(10, 43);
    var taken: u32 = 0;
    for (0..42) |_| {
        const n = c.take(2);
        c.spend(n);
        taken += n;
    }
    try testing.expect(taken >= 9 and taken <= 10);
    // Unread credit is held up to 2, no more.
    var h = Credit.of(10, 43);
    for (0..100) |_| _ = h.take(2);
    try testing.expectEqual(@as(u32, 2), h.take(2));
    try testing.expectEqual(@as(u32, 1), h.take(1));
    // Off: nothing past the budget.
    var z = Credit.of(0, 43);
    try testing.expectEqual(@as(u32, 0), z.take(2));
}

test "lookahead wide meters: this request's counts; nothing read, no line" {
    last = @splat(@splat(.{ 0, 0 }));
    defer last = @splat(@splat(.{ 0, 0 }));
    const saved = per_round;
    defer per_round = saved;
    per_round = 10;
    var h: Hist = @splat(@splat(.{ 0, 0 }));
    var buf: [1024]u8 = undefined;
    try testing.expectEqualStrings("", meters(h, &buf));
    h[0][0] = .{ 5, 4 };
    h[0][1] = .{ 3, 1 };
    h[1][2] = .{ 2, 1 };
    const line = meters(h, &buf);
    try testing.expect(std.mem.startsWith(u8, line, "[lookahead] wide=10 p=8 r=5 xp=2 xr=1 p0=5 r0=4 p1=3 r1=1 p2=0 r2=0"));
    try testing.expect(std.mem.indexOf(u8, line, " xp2=2 xr2=1 ") != null);
    // The next request: only its own counts.
    h[0][0] = .{ 6, 4 };
    const next = meters(h, &buf);
    try testing.expect(std.mem.startsWith(u8, next, "[lookahead] wide=10 p=1 r=0 xp=0 xr=0 p0=1 r0=0"));
}
