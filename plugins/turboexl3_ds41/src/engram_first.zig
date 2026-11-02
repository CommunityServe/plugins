//! `MLX_SERVE_DSV41_ENGRAM_FIRST` (read at load; unset, empty or `1`: on; `0`: off; `check`; any other value is refused
//! by name): a prompt chunk's Engram rows read sooner and on more threads.
//!
//! Off, extendIds hashes the chunk's Engram row ids and starts their read thread (`engram.Prefetch`) after the phase
//! change, and the thread reads a prompt grid's table (its distinct rows sorted, about 155K of them a table on a cold 16K
//! card, about 4 KB each, one direct pread a row: MLX_SERVE_DSV41_ENGRAM_DIRECT) on at most 64 threads, the first
//! Engram layer's table first. That layer (L1) waits for table 0: 383-406 ms on the cold card and 290-328 ms on the 15K
//! new-text slice (w-pio-20261102-0840, -1002, -1013: table 0 read in 925-936 / 752-770 ms, L1 reaching it about 536 /
//! 458 ms after the thread starts; table 1 is in long before L14), the GPU idle meanwhile. The reads are small and many
//! (about 167K a second, 0.7 GB/s) beside the expert stream's large ones on the same drive. On (`1`): the ids hashed and
//! the read thread started first, before the phase change, and a prompt grid's tables read on up to `grid_threads`
//! threads, a deeper queue for the drive (a machine parameter, not the cell's). Exact: the same rows, read into the same
//! buffers and served the same; only when the read starts and how many threads run it. `check`: as on, and table 0's
//! rows read again at the stock 64 threads, the two reads' bytes compared (a `[engram-first] check` line a prompt grid).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

/// The threads a prompt grid's table read runs on with the switch on (the stock: 64).
pub const grid_threads = 128;

pub const Error = error{EngramFirstSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.EngramFirstSetting;
}

/// `check`'s line: table 0's rows read at `grid_threads` and at the stock threads, their bytes compared.
pub fn report(rows: usize, bytes: usize, bad: usize) void {
    std.debug.print("[engram-first] check table=0 rows={d} bytes={d} differ={d}\n", .{ rows, bytes, bad });
}

/// The bytes of `a` and `b` (one length) that differ; every byte of the longer one past the shorter's end if not.
pub fn differ(a: []const u8, b: []const u8) usize {
    const n = @min(a.len, b.len);
    var d: usize = @max(a.len, b.len) - n;
    for (a[0..n], b[0..n]) |x, y| d += @intFromBool(x != y);
    return d;
}

test "engram_first setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.on, try parse("1"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "first", "CHECK" }) |v| try std.testing.expectError(error.EngramFirstSetting, parse(v));
}

test "engram_first differ: equal bytes 0; each differing byte counted; a length mismatch counts the extra bytes" {
    try std.testing.expectEqual(@as(usize, 0), differ("abcd", "abcd"));
    try std.testing.expectEqual(@as(usize, 2), differ("abcd", "abxy"));
    try std.testing.expectEqual(@as(usize, 2), differ("abcd", "ab"));
    try std.testing.expect(grid_threads >= 64);
}
