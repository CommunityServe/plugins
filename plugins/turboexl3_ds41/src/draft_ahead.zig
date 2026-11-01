//! `MLX_SERVE_DSV41_DRAFT_AHEAD` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything
//! else is refused by name): a greedy round's next draft launched at the end of the round before it.
//!
//! A greedy round (`draftStep`) ends with its state committed (`dsparkFinish`) and its next token known; the next round
//! starts after the host's bookkeeping, the engine's turn and the round's own start (its checks, its plan), and only
//! then does its draft's graph go to the GPU: the GPU idles between the verify's last kernel and the draft's first
//! (tp1, w-20261101-1811: 0.63 ms median, the verify -> draft seam). On, `draftStep` launches the next round's draft
//! from its next token right after its bookkeeping, when the next round's plan would be the stages' block
//! (`rounds.Drafter.wouldStage`): the draft's graph built from the committed state and handed to the GPU, not read. The
//! next round's `draft` takes it when it fits (a greedy round, its plan the stages' block, the same token, position and
//! state) and reads it as its own draft would be read; any other entry (a forward, a prompt, a sampled round, the
//! request's end) drops it. `check` makes the round's own draft too, compares the two (ids and confidence words) and
//! serves the round's own. Exact: the same ops on the same state, launched earlier.

const std = @import("std");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .off;

pub const Error = error{DraftAheadSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.DraftAheadSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// Since load, on the model's thread: drafts launched, taken by the next round, dropped (no round took them: another
/// entry, a plan other than the stages' block, another token or state), launches that failed; `check`'s compares and
/// the words (ids and confidences) that differ.
pub var launched: u64 = 0;
pub var taken: u64 = 0;
pub var dropped: u64 = 0;
pub var failed: u64 = 0;
pub var checks: u64 = 0;
pub var words: u64 = 0;
pub var differ: u64 = 0;

/// One `check`: the launched draft's ids, confidences and length against the round's own; differing words counted.
pub fn compare(ids: []const u32, own_ids: []const u32, conf: []const f32, own_conf: []const f32, len: usize, own_len: usize) void {
    checks += 1;
    words += ids.len + conf.len + 1;
    if (ids.len != own_ids.len or conf.len != own_conf.len) {
        differ += @max(ids.len, own_ids.len) + @max(conf.len, own_conf.len) + 1;
        return;
    }
    for (ids, own_ids) |a, b| differ += @intFromBool(a != b);
    for (conf, own_conf) |a, b| differ += @intFromBool(@as(u32, @bitCast(a)) != @as(u32, @bitCast(b)));
    differ += @intFromBool(len != own_len);
}

/// The meters' line (`[draft-ahead]`), "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[draft-ahead] mode={t} launched={d} taken={d} dropped={d} failed={d}", .{ mode, launched, taken, dropped, failed }) catch return buf[0..0];
    if (mode == .check) w.print(" checks={d} words={d} differ={d}", .{ checks, words, differ }) catch return buf[0..0];
    w.writeAll("\n") catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

fn resetForTest() void {
    launched = 0;
    taken = 0;
    dropped = 0;
    failed = 0;
    checks = 0;
    words = 0;
    differ = 0;
}

test "draft ahead setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", "1 ", "true" }) |v| try testing.expectError(error.DraftAheadSetting, parse(v));
}

test "draft ahead check: equal drafts differ nowhere; an id, a confidence or the length each count" {
    defer resetForTest();
    resetForTest();
    const ids = [_]u32{ 7, 11, 13, 17 };
    const conf = [_]f32{ 0.5, -1.25, 3.0 };
    compare(&ids, &ids, &conf, &conf, 3, 3);
    try testing.expectEqual(@as(u64, 1), checks);
    try testing.expectEqual(@as(u64, 8), words);
    try testing.expectEqual(@as(u64, 0), differ);
    const ids2 = [_]u32{ 7, 11, 19, 17 };
    const conf2 = [_]f32{ 0.5, -1.0, 3.0 };
    compare(&ids, &ids2, &conf, &conf, 3, 3);
    try testing.expectEqual(@as(u64, 1), differ);
    compare(&ids, &ids, &conf, &conf2, 3, 3);
    try testing.expectEqual(@as(u64, 2), differ);
    compare(&ids, &ids, &conf, &conf, 3, 2);
    try testing.expectEqual(@as(u64, 3), differ);
}

test "draft ahead meters: the counters, the check's words; none off" {
    const saved = mode;
    defer {
        mode = saved;
        resetForTest();
    }
    resetForTest();
    var buf: [256]u8 = undefined;
    mode = .off;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .on;
    launched = 5;
    taken = 4;
    dropped = 1;
    try testing.expectEqualStrings("[draft-ahead] mode=on launched=5 taken=4 dropped=1 failed=0\n", meters(&buf));
    mode = .check;
    checks = 4;
    words = 44;
    try testing.expectEqualStrings("[draft-ahead] mode=check launched=5 taken=4 dropped=1 failed=0 checks=4 words=44 differ=0\n", meters(&buf));
}
