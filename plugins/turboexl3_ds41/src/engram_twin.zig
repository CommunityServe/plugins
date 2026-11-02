//! `MLX_SERVE_DSV41_ENGRAM_TWIN` (read at load; unset, empty or `1`: on; `0`: off; `check`: on, every twin call compared
//! with MLX's one-row matmul of each of its rows; anything else is refused by name): the Engram layers' wkv words the
//! same at any verify width (zzz9r's second patch, eg2, on the shared gemv twin).
//!
//! An Engram layer turns its n-gram rows into one key a copy plus a shared value through `wkv`, a plain half-type linear
//! (the pack's f16 cast to the activation dtype at load: bf16 [L, 6144] x [25600, 6144]^T on DeepSeek-V4.1-Flash). MLX
//! (the kit's lib/mlx-src, 64ea011) runs it as its gemv at one row, as gemv_wide at 2 to 15 rows and as its GEMM from 16:
//! a verify's rows round apart from the same rows decoded one at a time (ip1, w-20261102-0143: the Engram wkv the only
//! break at widths to 6). On, a call of 2 to `twin_rows` rows runs on the batched twin of MLX's one-row gemv
//! (src/gemv_twin.zig, zzz9r-1's: one launch, every row the words MLX's gemv gives that row alone; its served
//! launch for 25600 x 6144 by rows: r2 t1, r3 t1h, r4 t1, r5 and r6 c3t2i, dvi4u), so a row's keys and value are its
//! one-row words at any verify width; one row keeps MLX's gemv (the same words) and a wider call (a prompt's chunk) its
//! matmul. `check` also runs every twin call's rows one at a time through MLX's matmul and compares them word for word
//! (the twin's `[gemv-twin] check N=25600 K=6144 ...` lines, every 128 calls and at a differing word), and prints an
//! `[engram-twin]` line at each request's end: the calls the twin served and declined, and in check mode the Engram
//! shape's check counts over the request (calls, rows, words, differ) with the running max_ulps. Check holds no second
//! weight: per call MLX's own matmul of the call and one M=1 matmul a row (kilobytes), and host syncs.

const std = @import("std");
const gtwin = @import("gemv_twin.zig");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .on;
/// The widest call the twin takes (a decode lane's verify; the copied model's decode-width regions run to 16 rows).
pub const twin_rows: usize = 16;

/// The twin calls the Engram layers' wkv made and declined since the last line (`engramWkv`).
pub var served: u64 = 0;
pub var declined: u64 = 0;
/// The Engram wkv's weight shape (N, K), noted at its first twin call: its check tally's key.
pub var shape: [2]usize = .{ 0, 0 };

pub const Error = error{EngramTwinSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.EngramTwinSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// A twin call of the Engram wkv (`n` x `k` weight): `ok` served, else declined.
pub fn note(n: usize, k: usize, ok: bool) void {
    if (ok) served += 1 else declined += 1;
    shape = .{ n, k };
}

/// The check tally's counts at the last line (the request's are the differences).
var last: gtwin.Tally = .{};

fn tallyOf(n: usize, k: usize) gtwin.Tally {
    for (gtwin.tallies) |t| if (t.n == n and t.k == k and n != 0) return t;
    return .{};
}

/// `[engram-twin]` at a request's end (then the request's counts cleared), "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    defer {
        served = 0;
        declined = 0;
    }
    w.print("[engram-twin] mode={t} served={d} declined={d}", .{ mode, served, declined }) catch return buf[0..0];
    if (mode == .check) {
        const t = tallyOf(shape[0], shape[1]);
        w.print(" N={d} K={d} calls={d} rows={d} words={d} differ={d} max_ulps={d}", .{ shape[0], shape[1], t.calls -| last.calls, t.rows -| last.rows, t.words -| last.words, t.differ -| last.differ, t.max_ulps }) catch return buf[0..0];
        last = t;
    }
    w.writeAll("\n") catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "engram twin setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", "1 ", "true" }) |v| try testing.expectError(error.EngramTwinSetting, parse(v));
}

test "engram twin meter: the request's served and declined calls; in check mode the Engram shape's check counts over the request" {
    const saved_mode = mode;
    const saved_tallies = gtwin.tallies;
    defer {
        mode = saved_mode;
        gtwin.tallies = saved_tallies;
        served = 0;
        declined = 0;
        shape = .{ 0, 0 };
        last = .{};
    }
    var buf: [256]u8 = undefined;
    mode = .off;
    note(25600, 6144, true);
    try testing.expectEqualStrings("", meters(&buf));
    mode = .check;
    last = .{};
    gtwin.tallies = @splat(.{});
    gtwin.tallies[0] = .{ .n = 32, .k = 5120, .calls = 9, .rows = 40, .words = 1280 };
    gtwin.tallies[1] = .{ .n = 25600, .k = 6144, .calls = 3, .rows = 12, .words = 307200 };
    served = 0;
    note(25600, 6144, true);
    note(25600, 6144, true);
    note(25600, 6144, true);
    note(25600, 6144, false);
    try testing.expectEqualStrings("[engram-twin] mode=check served=3 declined=1 N=25600 K=6144 calls=3 rows=12 words=307200 differ=0 max_ulps=0\n", meters(&buf));
    gtwin.tallies[1].calls = 5;
    gtwin.tallies[1].rows = 20;
    gtwin.tallies[1].words = 512000;
    note(25600, 6144, true);
    note(25600, 6144, true);
    try testing.expectEqualStrings("[engram-twin] mode=check served=2 declined=0 N=25600 K=6144 calls=2 rows=8 words=204800 differ=0 max_ulps=0\n", meters(&buf));
    mode = .on;
    note(25600, 6144, true);
    try testing.expectEqualStrings("[engram-twin] mode=on served=1 declined=0\n", meters(&buf));
}
