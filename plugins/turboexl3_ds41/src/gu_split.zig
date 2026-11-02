//! `MLX_SERVE_DSV41_GU_SPLIT` (read at load; dq1b): decode's demand reads with each missed record's gate/up range read as
//! two range jobs side by side, through the copied read pool's `q3ld_gu_split` (patches/turboexl3-zzz9h-gu-split.patch).
//! The split falls at the part boundary nearest the range's byte midpoint: a classic bank's gate/up boundary, or, in an
//! a4k bank, the up code alone in the second half. One range reads as one preadv at about 3.6 GB/s (dq1a's
//! `[demand-split]`, w-20261102-0303: a K = 3 record's 8.88 MB gate/up span in 2.44 ms on the card, its 4.44 MB down span
//! in 1.35 ms beside it), against about 13 GB/s for the drive at four streams. A missed record's latency was its gate/up
//! span's; with two halves read side by side it is about its down span's. The source sets the pool's mode at each phase
//! change: decode's mode for decode, off for the prompt.
//!
//! Settings: unset, empty or `1` is on (merged on: zzz9h); `0` is off. `check` is on, and before each split ticket is
//! published the pool
//! reads the whole range again and compares it with the rows its halves wrote (`[gu-split] mismatch=`). Any other value
//! is refused by name.
//!
//! `[gu-split]` at each request's end, when the switch is set, gives the request's counts:
//!   - split tickets;
//!   - their halves read, adopted out of a lookahead record, or served by a pre-range of their own;
//!   - half pre-ranges queued;
//!   - per split ticket, the mean span (earlier half's start to later half's end) and the mean time both halves were in
//!     flight;
//!   - in check mode, the tickets compared and those that differed.
//! Exact: the same bytes go into the same rows. Only the preadv calls covering a range change.
//!
//! The read (dq1b, M5 Max, w-20261102-0820; code-ctx16384 card then greedy, a = ctl29 6e9c8b0a279a, b = dq1b
//! 301b9c8d08fe at 1, both on G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 43.8 44.1 43.5 44.2 against 45.5 45.8 45.8 46.0
//! tok/s (+4.27%; pairs +3.88, +3.85, +5.29, +4.07), a round 111.56 -> 107.08 ms; greedy 56.1 56.3 55.7 56.4 against
//! 57.0 57.0 56.5 57.0 (+1.34%; pairs +1.60, +1.24, +1.44, +1.06), a round 87.12 -> 85.88 ms; the texts the same bytes
//! in every arm, check's split tickets all compared with no mismatch, the rows the same, the claims not collapsing.
//! Merged on (zzz9h).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .off;

pub const Error = error{GuSplitSetting};

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.GuSplitSetting;
}

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// The read pool's mode for decode (`q3ld_gu_split`): 0 off, 1 on, 2 check.
pub fn poolMode() u32 {
    return switch (mode) {
        .off => 0,
        .on => 1,
        .check => 2,
    };
}

/// The stream's split counters (`expert_stream.Stats`'s `split_*`), cumulative.
pub const Snap = struct {
    tickets: u64 = 0,
    read: u64 = 0,
    adopted: u64 = 0,
    pre: u64 = 0,
    pre_issued: u64 = 0,
    span_ns: u64 = 0,
    overlap_ns: u64 = 0,
    checked: u64 = 0,
    mismatch: u64 = 0,

    pub fn of(s: anytype) Snap {
        return .{
            .tickets = s.split_tickets,
            .read = s.split_read,
            .adopted = s.split_adopted,
            .pre = s.split_pre,
            .pre_issued = s.split_pre_issued,
            .span_ns = s.split_span_ns,
            .overlap_ns = s.split_overlap_ns,
            .checked = s.split_checked,
            .mismatch = s.split_mismatch,
        };
    }

    fn minus(a: Snap, b: Snap) Snap {
        var d: Snap = .{};
        inline for (@typeInfo(Snap).@"struct".field_names) |f| @field(d, f) = @field(a, f) -| @field(b, f);
        return d;
    }
};

/// The counters at the last line (or at load).
var last: Snap = .{};

/// The counters at load: the first request's line counts from here.
pub fn start(s: Snap) void {
    last = s;
}

fn meanMs(ns: u64, n: u64) f64 {
    if (n == 0) return 0;
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
}

/// `[gu-split]`: the counters since the last line, "" off.
pub fn meters(now: Snap, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const d = now.minus(last);
    last = now;
    var w: std.Io.Writer = .fixed(buf);
    w.print("[gu-split] mode={t} tickets={d} read={d} adopted={d} pre={d} pre_issued={d} span_ms={d:.3} overlap_ms={d:.3} checked={d} mismatch={d}\n", .{
        mode, d.tickets, d.read, d.adopted, d.pre, d.pre_issued, meanMs(d.span_ns, d.tickets), meanMs(d.overlap_ns, d.tickets), d.checked, d.mismatch,
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "gu_split setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "1 " }) |v| try testing.expectError(error.GuSplitSetting, parse(v));
}

test "gu_split: the pool's modes" {
    const saved = mode;
    defer mode = saved;
    mode = .off;
    try testing.expectEqual(@as(u32, 0), poolMode());
    mode = .on;
    try testing.expectEqual(@as(u32, 1), poolMode());
    mode = .check;
    try testing.expectEqual(@as(u32, 2), poolMode());
}

test "gu_split meter: the request's differences, the mean span and overlap a split ticket; off writes nothing" {
    const saved = mode;
    defer mode = saved;
    mode = .check;
    start(.{ .tickets = 10, .read = 15, .span_ns = 9_000_000 });
    var buf: [512]u8 = undefined;
    const out = meters(.{ .tickets = 14, .read = 21, .adopted = 1, .pre = 1, .pre_issued = 2, .span_ns = 13_000_000, .overlap_ns = 3_200_000, .checked = 4, .mismatch = 0 }, &buf);
    try testing.expectEqualStrings("[gu-split] mode=check tickets=4 read=6 adopted=1 pre=1 pre_issued=2 span_ms=1.000 overlap_ms=0.800 checked=4 mismatch=0\n", out);
    mode = .off;
    try testing.expectEqualStrings("", meters(.{}, &buf));
}
