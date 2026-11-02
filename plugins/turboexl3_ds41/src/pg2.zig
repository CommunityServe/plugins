//! `MLX_SERVE_DSV41_PG2` (read at load; unset, empty or `1`: on; `0`: off; `check`: the chain served and pg2's launch
//! checked against it; anything else is refused by name): a wide prompt layer's dense EXL3 linears' matmul and finish as
//! one launch at the shapes this machine runs faster that way (turboexl3's `pgemm`, src/pg2_t3.zig).
//!
//! turboexl3's `linearOn` at prefill rows runs the prepare, the W_hat decode, MLX's f16 matmul (z half [M, N] written) and
//! the finish (z read, y written: f16(H128(f16(z)) * svh) per 128-column block). pg2 runs the matmul and the finish as one
//! kernel: MLX's fused NAX GEMM body and gemm loop (its statements, SK 32, the 16 x 32 x 16 MMA on f16, k in MLX's order)
//! at the tile the pg2u probe measured per (K, N) on this machine, with the finish's statements as its epilogue (the
//! accumulators rounded to half as MLX's store rounds them, staged in threadgroup memory), launched in MLX's own
//! threadgroup shape. It takes a call only where MLX runs that regular kernel (no split K) on the NAX path, so y is the
//! chain's y word for word; z never reaches memory. The shapes (turboexl3 `pgemm.table`): wq_a, wkv (and the
//! compressors'), wq_b, wo_b, the shared expert's w1, w3, w2. WOA_JOIN's wo_a runs are not taken. `check` serves the chain
//! and, at every call pg2 takes, runs pg2's launch on the same operands and compares the two y word for word (one host
//! read a call). `[pg2]` at each request's end (cumulative for the process).
const std = @import("std");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .on;

pub const Error = error{Pg2Setting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.Pg2Setting;
}

/// check's compares since load: calls, the y words compared and those that differ
pub var checks: u64 = 0;
pub var words: u64 = 0;
pub var differ: u64 = 0;

/// One `check` compare's counts.
pub fn tally(n: u64, same: u64) void {
    checks += 1;
    words += n;
    differ += n - @min(same, n);
}

/// The request's `[pg2]` line (the counts since load: pg2's launches and their output elements; under check the
/// compares), or nothing off.
pub fn meters(launches: u64, elems: u64, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[pg2] mode={t} launches={d} elems={d}", .{ mode, launches, elems }) catch return buf[0..0];
    if (mode == .check) w.print(" checks={d} words={d} differ={d}", .{ checks, words, differ }) catch return buf[0..0];
    w.writeAll("\n") catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "pg2 setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "true", "Check", " 1" }) |v| try testing.expectError(error.Pg2Setting, parse(v));
    defer mode = .on;
    try configure("check");
    try testing.expectEqual(Mode.check, mode);
    try configure("0");
    try testing.expectEqual(Mode.off, mode);
}

test "pg2 meters: nothing off; launches and elements on; the compares under check" {
    var buf: [256]u8 = undefined;
    defer mode = .on;
    mode = .off;
    try testing.expectEqualStrings("", meters(3, 9, &buf));
    mode = .on;
    try testing.expectEqualStrings("[pg2] mode=on launches=3 elems=9\n", meters(3, 9, &buf));
    mode = .check;
    checks = 2;
    words = 18;
    differ = 0;
    defer {
        checks = 0;
        words = 0;
        differ = 0;
    }
    try testing.expectEqualStrings("[pg2] mode=check launches=3 elems=9 checks=2 words=18 differ=0\n", meters(3, 9, &buf));
}
