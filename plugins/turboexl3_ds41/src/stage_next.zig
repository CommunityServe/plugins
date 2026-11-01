//! `MLX_SERVE_DSV41_STAGE_NEXT` (read at load; unset, empty, `1` or `only`: on, and a layer the call before it read
//! ahead for skips its own read-ahead; `own`: on, and every wide layer also runs its own read-ahead, as before `only`
//! was the default; `0`: off; anything else is refused by name; with `MLX_SERVE_DSV41_STAGE` on): a wide prompt layer's
//! predicted transient loads read during the call before it, into the windows that call's groups let go.
//!
//! `only` (the default): the layer's own read-ahead (RA_FUSED's prediction from its input, read back at its start:
//! ~17-20 ms of host wait a 16K-row layer and its norm and router passes on the GPU) is not run where the call before it
//! read ahead for it; its staging is taken at its call's start (`Stream.promoteFor`), its P1 records are the early ones.
//! The first wide layer of a pass reads ahead as before.
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

pub const Mode = enum { off, own, only };

/// The switch (`configure`, at load).
pub var on: bool = true;
/// `only` (the default): a layer the call before it read ahead for skips its own read-ahead.
pub var skip_own: bool = true;

pub const Error = error{StageNextSetting};

pub fn configure(value: ?[]const u8) Error!void {
    const mode = try parse(value);
    on = mode != .off;
    skip_own = mode == .only;
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .only;
    if (v.len == 0 or std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "only")) return .only;
    if (std.mem.eql(u8, v, "own")) return .own;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.StageNextSetting;
}

test "stage_next setting: only unset, empty, 1 or only; own; 0 off; anything else refused by name" {
    try std.testing.expectEqual(Mode.only, try parse(null));
    try std.testing.expectEqual(Mode.only, try parse(""));
    try std.testing.expectEqual(Mode.only, try parse("1"));
    try std.testing.expectEqual(Mode.only, try parse("only"));
    try std.testing.expectEqual(Mode.own, try parse("own"));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    for ([_][]const u8{ "on", "2", "01", "true", " 1", "Only", "OWN", "off" }) |v| try std.testing.expectError(error.StageNextSetting, parse(v));
}
