//! `MLX_SERVE_DSV41_JOINLESS` (read at load; on by default, `0` turns it off, `check` runs both and compares them; any
//! other value is refused by name): a wide prompt call's routed sum and the layer's shared expert in one combine that
//! reads each routed row where its wave wrote it (the streamer's JOINLESS route: `Hook.routedParts` and the kernel set's
//! `dsv41_jl_combine_bf16`).
//!
//! `0`: the copied streamer joins every bank call's wave outputs (a concatenate and a take, each evaluated inside the
//! routed call), then the call's outputs again into `[rows * 6, 5120]` bf16 in routed order; the copied source widens
//! them to f32, multiplies by the routes' weights and sums the six; the model adds the shared expert in f32. At 16K rows
//! that is about 18 GB of reads and writes a layer, 14 GB of it in the layer's last eval; the combine reads and writes
//! under 2 GB.
//!
//! On, the waves' outputs stay as they are (above 24 of them, the smallest are concatenated into the last source,
//! `mergeJoinless`) and one kernel reads each routed row in place through a (source, row) table, weights it, folds the six
//! in the column reduce's order and adds the shared rows: the op chain's f32 words (`t = v0 + 0`, then `t = (vk + 0) + t`,
//! then `+ shared`, uncontracted). `check` builds both from the same outputs, compares them word for word, serves the op
//! chain's and logs one `[joinless] check` line a wide call.

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

pub const Error = error{JoinlessSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.JoinlessSetting;
}

/// `check`'s reference join: each source's first row once the sources are concatenated in order (`rows`: each source's
/// rows), so assignment (s, r) of the combine's table is row `first[s] + r` of the join. Returns the join's rows.
pub fn firstRows(rows: []const u32, first: []i32) u64 {
    std.debug.assert(first.len == rows.len);
    var at: u64 = 0;
    for (rows, first) |r, *f| {
        f.* = @intCast(at);
        at += r;
    }
    return at;
}

/// One wide call's `check`: its rows, the words compared, those that differ and the largest absolute difference.
pub fn report(layer: u32, rows: u64, sources: usize, words: u64, differ: u64, max_abs: f32) void {
    std.debug.print("[joinless] check layer={d} rows={d} sources={d} words={d} differ={d} max_abs={e}\n", .{ layer, rows, sources, words, differ, max_abs });
}

test "joinless setting: on unset or empty, 0 off, check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "1", "on", "2", "00", "Check", "true" }) |v| try std.testing.expectError(error.JoinlessSetting, parse(v));
}

test "joinless check join: each source starts after the rows of the ones before it" {
    var first: [4]i32 = undefined;
    try std.testing.expectEqual(@as(u64, 4100), firstRows(&.{ 7, 4000, 1, 92 }, &first));
    try std.testing.expectEqualSlices(i32, &.{ 0, 7, 4007, 4008 }, &first);
    var one: [1]i32 = undefined;
    try std.testing.expectEqual(@as(u64, 183), firstRows(&.{183}, &one));
    try std.testing.expectEqual(@as(i32, 0), one[0]);
}
