//! `MLX_SERVE_DSV41_DECODE_FUSE` (read at load; unset, empty or `1`: on, merged on as zzz9l; `0`: off; `check`: the chain
//! served, the kernel compared with it; anything else is refused by name; patches/turboexl3-zzz9l-decode-fuse.patch): a
//! decode-lane routed call's tail in TAIL_FUSE's kernel.
//!
//! A decode call (at most 48 routed ids: a DSpark verify's 6 rows, the prompt's last token) ends in the copied streamer's
//! join (its outputs concatenated, then taken in routed order), the copied source's weight and sum (a multiply, then
//! MLX's column reduce over the six), and the model's `ffn_post_shared` stretch (the shared expert's rows added in f32,
//! rounded to the activation dtype, glm5's hc expand): a few small launches a layer, 40 layers a round. TAIL_FUSE's
//! kernel (src/tail_fuse.zig, kernels/tail_fuse.metal) already computes that chain's words in one pass for a wide prompt
//! call: each routed row read where it lies through a (source, row) table, times its weight, folded in the column
//! reduce's order (`t = v0 + 0`, then `t = (vk + 0) + t`: the reduce at six rows runs six thread rows and folds them in
//! order), the shared row added, rounded once, then the hc expand, uncontracted where the chain is. On (with
//! `MLX_SERVE_DSV41_SHARED_HOIST`), the model builds the shared expert first and hands it to the source, which commits
//! it behind the call's waves (where SHARED_HOIST commits it); the call ends at the join's rows and their table
//! (`Hook.routedHoistLocated`: JOIN_INDEXED's order, as (source 0, row) pairs) and the kernel writes the layer's new
//! streams; its decode instance (`tail_fuse.expandDecode`, NROWS at `tail_fuse.decode_rows`) is one compiled pipeline
//! for every decode call, compiled at load (`warmed`). `check` serves the chain (the take, the weight and sum, the add,
//! the round and the expand from the same rows) and compares the kernel's streams with it on every decode call, word
//! for word. `[decode-fuse] mode= calls= checks= words= differ= max_abs=` at each request's end, since load. Exact: the
//! chain's operations in its order, one launch.
//!
//! The read (jc1, M5 Max, w-20261102-1043; a = ctl25 5ccb44791b50, b = jc1 3ba9ae0d352e at 1, chk at check): card +0.63%
//! (44.1 44.0 43.9 43.7 against 44.3 44.3 43.8 44.4 tok/s), greedy +0.45% (55.7 56.1 55.6 55.7 against 56.0 55.9 56.1
//! 56.1); the texts the same bytes in every arm, chk's differ 0 over 4640 + 8960 checks. Merged on (zzz9l).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .off;
/// The decode instance compiled at load (the model's `warmDecodeFuse`).
pub var warmed: bool = false;

/// Since load, on the model's thread: decode calls that took the kernel; `check`'s compares, their words, those that
/// differ and the largest absolute difference.
pub var calls: u64 = 0;
pub var checks: u64 = 0;
pub var words: u64 = 0;
pub var differ: u64 = 0;
pub var max_abs: f32 = 0;

pub const Error = error{DecodeFuseSetting};

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
    return error.DecodeFuseSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// A decode call takes the kernel: counted; the first one's dtypes logged once (whether the routed rows are f32, the
/// streams bf16, the shared rows bf16 or f32: the instances the load compiled), beside whether the load compiled them.
pub fn firstCall(rows_f32: bool, streams_bf16: bool, shared_bf16: bool, shared_f32: bool) void {
    calls += 1;
    if (calls != 1) return;
    std.debug.print("[decode-fuse] first decode call: rows f32 {s}, streams bf16 {s}, shared {s} (the decode instance compiled at load: {s})\n", .{ if (rows_f32) "yes" else "no", if (streams_bf16) "yes" else "no", if (shared_bf16) "bf16" else if (shared_f32) "f32" else "other", if (warmed) "yes" else "no" });
}

/// One `check`'s compare.
pub fn note(w: u64, d: u64, m: f32) void {
    checks += 1;
    words += w;
    differ += d;
    max_abs = @max(max_abs, m);
}

/// The meters' line (`[decode-fuse]`), "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[decode-fuse] mode={t} calls={d} checks={d} words={d} differ={d} max_abs={e}\n", .{ mode, calls, checks, words, differ, max_abs }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "decode fuse setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", "1 ", "true" }) |v| try testing.expectError(error.DecodeFuseSetting, parse(v));
}

test "decode fuse meters: the counts; none off" {
    const saved = .{ mode, calls, checks, words, differ, max_abs };
    defer {
        mode = saved[0];
        calls = saved[1];
        checks = saved[2];
        words = saved[3];
        differ = saved[4];
        max_abs = saved[5];
    }
    var buf: [192]u8 = undefined;
    mode = .off;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .check;
    calls = 0;
    checks = 0;
    words = 0;
    differ = 0;
    max_abs = 0;
    note(98304, 0, 0);
    note(98304, 0, 0);
    const out = meters(&buf);
    try testing.expect(std.mem.startsWith(u8, out, "[decode-fuse] mode=check calls=0 checks=2 words=196608 differ=0 max_abs="));
    try testing.expect(std.mem.endsWith(u8, out, "\n"));
}
