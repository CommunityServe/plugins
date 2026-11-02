//! `MLX_SERVE_DSV41_DEC_INVARIANT` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything
//! else is refused by name): the indexer's plain linears give a decode row the same words at any verify width.
//!
//! The indexer's head weights (each index layer's `weights_proj`, over the step's rows) and the KV sources' index keys
//! (`wk`, over the groups the step completes) are plain bf16 linears, which MLX runs as its M=1 gemv for one row,
//! gemv_wide for 2..15 rows and its GEMM past them, so a verify's rows round apart from the same rows decoded one at a
//! time. On, their calls of 2..`twin_rows` rows (a verify's widths: DSpark's drafts and the bonus row) take
//! src/gemv_twin.zig's batched twin of MLX's M=1 gemv: every row that gemv's words, the weight read once a launch, the
//! launch for the weight's shape and the call's rows its `served` table's. One row keeps MLX's matmul (its gemv, the same
//! words), and so do calls wider than `twin_rows` (a prompt's).
//!
//! `check`: on, and every twin call compared with MLX's M=1 matmul of each of its rows, word for word (a host sync a
//! row; `[gemv-twin] check` lines a weight shape every 128 calls, at its first differing word and at the model's end),
//! and the call's own MLX matmul (gemv_wide, what `0` serves) against the same words (`[gemv-twin] wide-vs-m1` lines,
//! reported only).

const std = @import("std");

pub const Mode = enum { off, on, check };
pub var mode: Mode = .on;

pub const Error = error{DecInvariantSetting};

/// The widest call the indexer's linears give the twin: a verify's widest (5 drafts and the bonus row; the twin itself
/// takes up to src/gemv_twin.zig's `max_rows`, its `served` table's launches measured at 2..6).
pub const twin_rows: usize = 6;

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.DecInvariantSetting;
}

pub fn on() bool {
    return mode != .off;
}

test "dec_invariant setting: on unset, empty or 1; 0 off; check; anything else refused" {
    const t = std.testing;
    try t.expectEqual(Mode.on, try parse(null));
    try t.expectEqual(Mode.on, try parse(""));
    try t.expectEqual(Mode.on, try parse("1"));
    try t.expectEqual(Mode.off, try parse("0"));
    try t.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", " 1", "1 ", "true", "check+1" }) |v| try t.expectError(error.DecInvariantSetting, parse(v));
}
