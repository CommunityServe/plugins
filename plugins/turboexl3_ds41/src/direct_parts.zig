//! `MLX_SERVE_DSV41_DIRECT_PARTS` (read at load; on by default, `0` turns it off, `1` on; any other value is refused by
//! name): the read pool's demand ranges part by part. A record is read as two ranges, its gate/up span (the gate's code,
//! rout and rin, then the up projection's) and its down span, each today in one F_NOCACHE preadv into the worker's
//! staging buffer and a copy of every part into its slot row. On, the pool tests each part as it reads it (the copied
//! pool's PARTS route, `q3ld_parts_config`): a part whose file offset, length and row address are multiples of the
//! file's block (fstatfs f_bsize, 4,096 B on APFS) is read straight into its row, the uncached preadv's direct path
//! with no staging and no copy; every run of the other parts takes the staging loop and the copy as before. On this pack
//! only the gate's code passes (at offset 0 of a record the bank pads to 4,096 B, 1,474,560 x K bytes; the up and down
//! codes sit 2,560 and 1,024 B off the block): a third of a record's bytes no longer copied, in every demand read of the
//! prompt phase (its routes, read-ahead and staged loads, the CED decoder's replay reads). The decode phase's reads (the
//! grow's fills, the misses) stage and copy as without it: the copied source switches the route off at each decode phase
//! and on at each prompt (`Stream.directPartsLive`). The pre-read ranges and the lookahead's records, read before their
//! rows exist, stage as before. Exact: the file's bytes land in the rows either way.
//!
//! Read on the M5 Max (dp1, w-pio-20261101-1006, on d451beb with the route in both phases: a throwaway warm server, then
//! 0 / on / on / 0, each a cold code-ctx16384 card request and a new-text 16K one, staging on, policy off): cold prefill
//! 1195.7 and 1192.7 tok/s at 0, 1199.5 and 1200.4 on (+0.48%); the encoder's attention 274.2 and 275.5 ms a layer at 0,
//! 268.0 and 267.5 on (-7.1 ms, a third of staging's copy interference); 35.81 of the cold prompt's 107.8 GB read direct.
//! The texts the same in every arm (the cold one md5 d081845201). Decode (dd1, w-20261101-1015, 16K card + greedy, 1024
//! tokens): the misses are the drive's wait, not the copy's (the same 78.5 and 60.0 a round, verify 114.8 / 99.5 ms a
//! round off against 118.1 / 99.9 on, decode 41.8 / 48.1 tok/s against 40.7 / 47.9), so the route is the prompt's only.
//!
//! A `[direct-parts] prompt` line at each prompt's end and a `[direct-parts] request` line at each request's end (every
//! phase since the last request's, or the load's, end): the parts read direct, their GB, and the GB of the other parts
//! of the ranges the route read (staged and copied), from the stream's counters.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{DirectPartsSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.DirectPartsSetting;
}

/// The stream's counters a line reads.
pub const Snap = struct {
    parts: u64 = 0,
    direct: u64 = 0,
    staged: u64 = 0,

    /// From the stream's `Stats`.
    pub fn of(s: anytype) Snap {
        return .{ .parts = s.parts_direct, .direct = s.parts_direct_bytes, .staged = s.parts_staged_bytes };
    }

    fn sub(a: Snap, b: Snap) Snap {
        return .{ .parts = a.parts -| b.parts, .direct = a.direct -| b.direct, .staged = a.staged -| b.staged };
    }
};

var at_prompt: ?Snap = null;
var at_request: Snap = .{};

/// The counters at the load's end: the first request's line counts from here.
pub fn start(s: Snap) void {
    at_request = s;
    at_prompt = null;
}

/// The source's prompt phase starts (`stats`: the stream's `Stats`).
pub fn promptStart(stats: anytype) void {
    if (!on) return;
    at_prompt = Snap.of(stats);
}

/// The source's decode phase starts: the prompt's line.
pub fn promptEnd(stats: anytype) void {
    if (!on) return;
    const a = at_prompt orelse return;
    at_prompt = null;
    var buf: [192]u8 = undefined;
    std.debug.print("{s}", .{line(&buf, "prompt", Snap.of(stats).sub(a))});
}

/// The request's end: its line into `buf` (empty when off or when it does not fit).
pub fn meters(now: Snap, buf: []u8) []const u8 {
    if (!on) return buf[0..0];
    defer at_request = now;
    return line(buf, "request", now.sub(at_request));
}

fn line(buf: []u8, what: []const u8, d: Snap) []const u8 {
    return std.fmt.bufPrint(buf, "[direct-parts] {s} parts={d} direct_gb={d:.2} staged_gb={d:.2}\n", .{ what, d.parts, gb(d.direct), gb(d.staged) }) catch buf[0..0];
}

fn gb(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

test "direct_parts setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expectEqual(true, try parse(null));
    try std.testing.expectEqual(true, try parse(""));
    try std.testing.expectEqual(true, try parse("1"));
    try std.testing.expectEqual(false, try parse("0"));
    for ([_][]const u8{ "on", "off", "2", "check", " 1", "00" }) |v| try std.testing.expectError(error.DirectPartsSetting, parse(v));
}

test "direct_parts lines: the request's counters since the last request's end" {
    const was = on;
    defer on = was;
    on = true;
    start(.{ .parts = 1, .direct = 4_000_000_000, .staged = 1_000_000_000 });
    var buf: [192]u8 = undefined;
    const s = meters(.{ .parts = 5, .direct = 21_690_000_000, .staged = 44_380_000_000 }, &buf);
    try std.testing.expectEqualStrings("[direct-parts] request parts=4 direct_gb=17.69 staged_gb=43.38\n", s);
    const t = meters(.{ .parts = 5, .direct = 21_690_000_000, .staged = 44_380_000_000 }, &buf);
    try std.testing.expectEqualStrings("[direct-parts] request parts=0 direct_gb=0.00 staged_gb=0.00\n", t);
    on = false;
    try std.testing.expectEqual(@as(usize, 0), meters(.{}, &buf).len);
}
