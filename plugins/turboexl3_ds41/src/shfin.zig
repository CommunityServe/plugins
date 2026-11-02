//! `MLX_SERVE_DSV41_SHFIN` (read at load; unset, empty or `1`: on, merged on as zzz9p; `0`: off, the host's chain;
//! `check`: the host's chain served and F3's launch checked against it; anything else is refused by name): F3, the decode
//! shared expert's EXL3 gate and up finishes, the clamped SwiGLU and the down's prepare in one launch (turboexl3's
//! `sharedSwiglu`, src/shfin_t3.zig).
//!
//! At a verify's rows (5 to 16: DEC-ONE keeps up to 4 as they are) the shared expert runs the gate's and the up's
//! prepare, GEMV and finish, the host's clamped SwiGLU in f32 (one compiled elementwise launch in a decode stretch), then
//! the down's prepare, GEMV and finish. On, the two finishes, the SwiGLU and the down's prepare are one launch
//! (`dense.text_swiglu_prep`: for each 128-wide block, text_dfinish's statements on the gate's and the up's planes, MLX's
//! Minimum, Maximum, Sigmoid and Multiply functors in clampedSwiGLU's order, the activation dtype's rounding, then
//! text_prep_s's statements, nothing between them stored): three launches and two dependent levels fewer a layer, the
//! same words (turboexl3's CPU test holds the statements; its GPU test compares every word at rows 1 to 16 with the
//! host's chain op for op). The shared expert's output is the chain's words, so what reads it is unchanged: the
//! `ffn_shared` stretch's output handed to DECODE_FUSE's kernel or to `ffn_post_shared` (SHARED_HOIST), or `withShared`
//! in `ffn_post`. `check` serves the host's chain and, for every shared expert F3 takes, runs F3 on the same input and
//! compares the down's output word for word: after each compiled decode stretch that holds the shared expert (its body
//! traced) by running the stretch's body once more, eagerly, on its own inputs. `[shfin]` at each request's end
//! (cumulative for the process): on, the calls F3's code ran, which in a compiled stretch are its traces (its replays run
//! F3's launch without passing the count); under check, every call.
//!
//! The reads: the f3u probe (w-f3u, M5 Max): every word the host chain's at rows 1 to 16 (taken at 5 to 16), and the
//! 40-layer trunk chain at rows 5, 6 and 8 -0.056, -0.029 and -0.218 ms a verify (medians, A B B A in 3 rounds). The f3
//! A/B (w-20261102-1413; a = ctl27 42a2a45fae3c, b = f3 at 1, chk at check): card +0.34% (44.0 44.0 44.1 44.3 against
//! 44.4 44.4 44.3 43.9 tok/s; round 111.07 -> 110.75 ms), greedy +0.18% (56.4 56.0 56.2 56.3 against 56.3 56.3 56.4
//! 56.3; round 86.95 -> 86.85 ms); the texts the same bytes in every arm, chk's differ 0 over 4899 + 9503 calls. Merged
//! on (zzz9p).
const std = @import("std");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .on;

pub const Error = error{ShfinSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.ShfinSetting;
}

/// Set while the copied model's compiled decode stretch traces its body (its arrays then hold no values to compare).
pub var tracing: bool = false;

/// Since load: shared experts F3 served (on: the calls its code ran, a compiled stretch's traces) or compared (check),
/// their words, and (check) those that differ.
pub var calls: u64 = 0;
pub var words: u64 = 0;
pub var differ: u64 = 0;

/// One served call (on).
pub fn served(n: u64) void {
    calls += 1;
    words += n;
}

/// One `check` compare's counts.
pub fn tally(n: u64, same: u64) void {
    calls += 1;
    words += n;
    differ += n - @min(same, n);
}

/// The request's `[shfin]` line (the counts since load), or nothing off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[shfin] mode={t} calls={d} words={d}", .{ mode, calls, words }) catch return buf[0..0];
    if (mode == .check) w.print(" differ={d}", .{differ}) catch return buf[0..0];
    w.writeAll("\n") catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "shfin setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "true", "Check", " 1" }) |v| try testing.expectError(error.ShfinSetting, parse(v));
    const saved = mode;
    defer mode = saved;
    try configure("check");
    try testing.expectEqual(Mode.check, mode);
    try configure("0");
    try testing.expectEqual(Mode.off, mode);
    try configure(null);
    try testing.expectEqual(Mode.on, mode);
}

test "shfin meters: nothing off; the served calls on; check's counts" {
    const saved = .{ mode, calls, words, differ };
    defer {
        mode = saved[0];
        calls = saved[1];
        words = saved[2];
        differ = saved[3];
    }
    var buf: [256]u8 = undefined;
    mode = .off;
    try testing.expectEqual(@as(usize, 0), meters(&buf).len);
    mode = .on;
    calls = 0;
    words = 0;
    differ = 0;
    served(30720);
    try testing.expectEqualStrings("[shfin] mode=on calls=1 words=30720\n", meters(&buf));
    calls = 0;
    words = 0;
    mode = .check;
    tally(10, 10);
    tally(6, 4);
    try testing.expectEqualStrings("[shfin] mode=check calls=2 words=16 differ=2\n", meters(&buf));
}
