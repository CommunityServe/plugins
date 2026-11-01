//! `MLX_SERVE_DSV41_STAGE_NEXT` (read at load; unset, empty or `1`: on; `0`: off; anything else is refused by name; with
//! `MLX_SERVE_DSV41_STAGE` on): a wide prompt layer's predicted transient loads read during the call before it, into
//! the windows that call's groups let go.
//!
//! Off, staging (`MLX_SERVE_DSV41_STAGE`, src/stage.zig) posts a layer's predicted stream misses at the layer's start,
//! after P1's read-ahead, so they read under its attention; a 16K-row layer's ~4.5 GB take ~340 ms at the drive's rate
//! against ~250 ms of attention, and its call waits on the rest, while the drive idles from the end of the call before's
//! reads to this layer's start (~5.5 s of a cold 16K prompt's encoder). A shorter attention (`MLX_SERVE_DSV41_Q_TAIL`)
//! turned into as much more waiting on bytes.
//!
//! On, the copied model predicts layer N+1's routes from layer N's FFN rows (N+1's gate over N's MoE input, its columns
//! scaled by the two FFN norms' ratio) and hands the ids, unevaluated, to the copied source before N's call; the call
//! starts them behind its routing barrier. Once the call's routes are all made, the source reads the prediction (each
//! expert's votes, hottest first), reads layer N+1's predicted seed ahead into N+1's persistent rows (which N's call never
//! reads) and lays N+1's stream misses out as staging does; each window's records are posted as soon as no route holds
//! the window (a group drains before its release, so no wave reads its rows). Layer N+1's staging then takes them as its
//! own (`Stream.stageGroups`: the windows not posted yet posted then) and its call adopts them as before; its own
//! read-ahead at its start reads whatever its prediction adds. The read pool is first come, first served, and N's routes'
//! reads are all queued before N+1's, so N's calls never wait behind them. The ticket ring doubles again (4096).
//!
//! Exact: the prediction picks which records are read ahead and into which rows, never the routes; a record is the same
//! bytes in any row.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{StageNextSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.StageNextSetting;
}

test "stage_next setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(try parse("1"));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "on", "2", "01", "true", " 1" }) |v| try std.testing.expectError(error.StageNextSetting, parse(v));
}
