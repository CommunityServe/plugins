//! `MLX_SERVE_DSV41_DEMAND_FIRST` (read at load; df1): decode's lookahead reads behind its demand reads, through the
//! copied read pool's `q3ld_demand_first` (patches/turboexl3-zzz9u-demand-first.patch). The source sets the pool's mode at
//! each phase change: the setting's mode for decode, off for the prompt.
//!
//! Inside a decode verify the GPU either works or waits for miss bytes (ctl48's profile, pt6d, M5 Max, code-ctx16384).
//! The card's verify is 44.9 ms of work and 53.2 ms of waits. As a fit, the waits are 6.1 + 0.974 x 36.7 ms of fresh
//! demand reads + 0.551 x 20.5 ms of claimed lookahead records still reading. Two costs come from the lookahead:
//!   - A fresh demand read beside an unclaimed lookahead chunk is slower. 51% of the card's K = 3 gate/up ranges take
//!     1.82 ms against 1.59 ms clean, the downs 1.36 against 1.24, and 58% of the lookahead's bytes are never claimed.
//!     Today lookahead thread 0 starts a chunk beside one demand read, and a chunk (a quarter slot, about 3.3 MB) once
//!     started runs to its end, demand or not.
//!   - A claimed record still reading is read by its one lookahead thread, chunk after chunk, while the demand ranges
//!     that need it wait: 11 ms a round on the card, 7 greedy.
//!
//! Settings:
//!   - unset, empty or `2`: on (merged on: zzz9u). The order: an unclaimed lookahead record starts and continues only
//!     while no demand read is queued or running, on every lookahead thread. It reads in 1 MiB pieces, so a demand read
//!     that starts meanwhile waits behind at most one piece a thread; the record then pauses at its next piece, its
//!     landed pieces kept, until demand drains. And the help: a claimed record's unread pieces go to the demand reads that
//!     need them. A demand range that finds its record claimed and its bytes not all in reads the record's untaken
//!     pieces under its own bytes itself, into the record's staging, beside the lookahead thread. It adopts the range as
//!     soon as every piece under it is in, not only once the record's prefix reaches it.
//!   - `1`: the order alone; claimed records read as before (by their lookahead thread, chunk after chunk).
//!   - `0`: off, the lane's rule as before.
//! Any other value is refused by name.
//!
//! `[demand-first]` at each request's end, when the switch is set, gives the request's counts:
//!   - pieces: unclaimed lookahead reads made in pieces;
//!   - stops: unclaimed records that stopped for demand inside one of the lane's chunks, and stop_mb, the rest of those
//!     chunks (what the lane would still have read beside the demand);
//!   - mode 2: help_runs, help_mb and help_ms, the runs of claimed pieces the demand read and their bytes and time;
//!     help_lands, records their last piece's demand reader landed; releases, claimed records their lookahead thread
//!     left to the demand; early, ranges adopted ahead of their record's prefix; failed, records a reader's piece failed;
//!   - adopt_waits and adopt_wait_ms, the demand ranges that found their record's bytes not all in, and their wait (with
//!     mode 2, the reads they made).
//! Exact: the same file bytes go into the same staging and the same rows. Only which thread reads a piece, and when,
//! changes.
//!
//! The reads (M5 Max, code-ctx16384 card then greedy, every arm on G_KITSP1 a34826975049 with keyed sampling; a1 b1 b2 a2
//! b3 a3 a4 b4):
//!   - `1` alone, df1d (w-20261103-0004; a = ctl48 b528be70e900, b = df1 7e1cad69a020 at 1): card 48.5 48.1 48.4 48.0
//!     against 47.8 46.9 47.7 47.9 tok/s (-1.40%, every pair negative), greedy 58.5 57.8 58.3 57.7 against 58.0 57.4
//!     57.9 57.9 (-0.47%); the texts the same bytes. The claims about the same (20.5 -> 20.4 a round), abandons 373 ->
//!     235, the card's verify 1.6 ms longer: with the lookahead yielding to every demand read the claimed records land
//!     later (adopt_waits 5468, 3669 ms a card request).
//!   - `2` against `0` on one binary (df1, 7e1cad69a020), df2o. Its first window (w-20261103-0019) was invalid on rows:
//!     147 / 176 in a1 b1 b2 a2 b3, 145 / 174 in a3, 146 / 175 in a4 and b4 (the bank bill's ceiling moved at load). The
//!     rerun (w-20261103-0036): the rows 146 / 175 in every arm, the texts the same bytes (card 8074f27375, greedy
//!     0a1c2b8690); card 48.0 47.4 48.0 48.0 against 47.9 48.2 48.4 48.3 tok/s (+0.73%; pairs -0.21, +1.69, +0.83,
//!     +0.62), greedy 58.0 57.3 58.0 58.0 against 58.7 58.7 58.8 58.4 (+1.43%; pairs +1.21, +2.44, +1.38, +0.69).
//!     Merged on at 2 (zzz9u).

const std = @import("std");

pub const Mode = enum { off, first, help };

/// The switch (`configure`, at load).
pub var mode: Mode = .off;

pub const Error = error{DemandFirstSetting};

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .help;
    if (v.len == 0) return .help;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .first;
    if (std.mem.eql(u8, v, "2")) return .help;
    return error.DemandFirstSetting;
}

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// The read pool's mode for decode (`q3ld_demand_first`): 0 off, 1 demand first, 2 demand first and the claimed pieces.
pub fn poolMode() u32 {
    return switch (mode) {
        .off => 0,
        .first => 1,
        .help => 2,
    };
}

/// The pool's df1 counters and its adoption waits (`Source.demandFirstStats`: the C pool's DF_* order, then
/// `adopt_waits`, `adopt_wait_ns`), cumulative.
pub const Snap = struct {
    pieces: u64 = 0,
    stops: u64 = 0,
    stop_bytes: u64 = 0,
    help_runs: u64 = 0,
    help_bytes: u64 = 0,
    help_ns: u64 = 0,
    help_lands: u64 = 0,
    releases: u64 = 0,
    early: u64 = 0,
    failed: u64 = 0,
    adopt_waits: u64 = 0,
    adopt_wait_ns: u64 = 0,

    pub fn of(w: [snap_words]i64) Snap {
        var s: Snap = .{};
        inline for (@typeInfo(Snap).@"struct".field_names, 0..) |f, i| @field(s, f) = @intCast(@max(w[i], 0));
        return s;
    }

    fn minus(a: Snap, b: Snap) Snap {
        var d: Snap = .{};
        inline for (@typeInfo(Snap).@"struct".field_names) |f| @field(d, f) = @field(a, f) -| @field(b, f);
        return d;
    }
};

/// `Snap`'s words (`Source.demandFirstStats`: the pool's df_n, then the two adoption counters).
pub const snap_words = 12;

comptime {
    if (@typeInfo(Snap).@"struct".field_names.len != snap_words) @compileError("demand_first: Snap's fields are not snap_words");
}

/// The counters at the last line (or at load).
var last: Snap = .{};

/// The counters at load: the first request's line counts from here.
pub fn start(s: Snap) void {
    last = s;
}

fn mb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e6;
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// `[demand-first]`: the counters since the last line, "" off.
pub fn meters(now: Snap, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const d = now.minus(last);
    last = now;
    var w: std.Io.Writer = .fixed(buf);
    w.print("[demand-first] mode={d} pieces={d} stops={d} stop_mb={d:.1} help_runs={d} help_mb={d:.1} help_ms={d:.2} help_lands={d} releases={d} early={d} failed={d} adopt_waits={d} adopt_wait_ms={d:.2}\n", .{
        poolMode(), d.pieces, d.stops, mb(d.stop_bytes), d.help_runs, mb(d.help_bytes), ms(d.help_ns), d.help_lands, d.releases, d.early, d.failed, d.adopt_waits, ms(d.adopt_wait_ns),
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "demand_first setting: 2 unset, empty or 2 (merged on); 1 the order alone; 0 off; anything else refused by name" {
    try testing.expectEqual(Mode.help, try parse(null));
    try testing.expectEqual(Mode.help, try parse(""));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.first, try parse("1"));
    try testing.expectEqual(Mode.help, try parse("2"));
    for ([_][]const u8{ "on", "3", "01", "first", "1 ", "-1" }) |v| try testing.expectError(error.DemandFirstSetting, parse(v));
}

test "demand_first: the pool's modes" {
    const saved = mode;
    defer mode = saved;
    mode = .off;
    try testing.expectEqual(@as(u32, 0), poolMode());
    mode = .first;
    try testing.expectEqual(@as(u32, 1), poolMode());
    mode = .help;
    try testing.expectEqual(@as(u32, 2), poolMode());
}

test "demand_first snap: the stats words in order, negatives as 0" {
    const s = Snap.of(.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12 });
    try testing.expectEqual(@as(u64, 1), s.pieces);
    try testing.expectEqual(@as(u64, 3), s.stop_bytes);
    try testing.expectEqual(@as(u64, 6), s.help_ns);
    try testing.expectEqual(@as(u64, 10), s.failed);
    try testing.expectEqual(@as(u64, 11), s.adopt_waits);
    try testing.expectEqual(@as(u64, 0), s.adopt_wait_ns);
    try testing.expectEqual(@as(usize, 12), snap_words);
}

test "demand_first meter: the request's differences; off writes nothing" {
    const saved = mode;
    defer mode = saved;
    mode = .help;
    start(.{ .pieces = 10, .stops = 1, .stop_bytes = 1_000_000, .adopt_waits = 5, .adopt_wait_ns = 2_000_000 });
    var buf: [512]u8 = undefined;
    const out = meters(.{ .pieces = 14, .stops = 3, .stop_bytes = 3_500_000, .help_runs = 6, .help_bytes = 12_000_000, .help_ns = 4_250_000, .help_lands = 1, .releases = 2, .early = 3, .failed = 0, .adopt_waits = 9, .adopt_wait_ns = 7_000_000 }, &buf);
    try testing.expectEqualStrings("[demand-first] mode=2 pieces=4 stops=2 stop_mb=2.5 help_runs=6 help_mb=12.0 help_ms=4.25 help_lands=1 releases=2 early=3 failed=0 adopt_waits=4 adopt_wait_ms=5.00\n", out);
    mode = .off;
    try testing.expectEqualStrings("", meters(.{}, &buf));
}
