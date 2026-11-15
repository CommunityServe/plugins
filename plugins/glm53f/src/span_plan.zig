//! glm53f's whole-prompt plan, MLX-free: the engine's spans checked (`check`, Ling's), each span's rows (`rowsOf`) and
//! whether the last two spans run as a pair (`pairAt`): the last chunk and the held-back tail, #1196's split.
//! copied_from: mlx-serve src/plugins/bailing_hybrid/span_plan.zig@84271b33 (`check`).

const std = @import("std");

/// The held-back tail's rows at most: the snapshot backoff (30, `SSM_SNAPSHOT_BACKOFF`) and the prompt's last row.
pub const max_tail: u32 = 31;
/// A pair's spans each take more rows than this: a window of 16 rows or fewer runs `glm5.hcPre`'s one-dispatch arm, not
/// the collapse the pair's MLP takes (`glm5.HC_PRE_MAX_ROWS`, #1196's `splitSpans`).
pub const min_rows: u32 = 16;

/// `ends` (absolute span ends, `Forward.prompt_spans`) cover the `rows` rows from `pos0` in ascending order, and every
/// checkpoint position is one of them.
pub fn check(pos0: u64, rows: u64, ends: []const u64, cps: []const u64) !void {
    if (ends.len == 0) return error.PromptSpansEmpty;
    var prev = pos0;
    for (ends) |e| {
        if (e <= prev) return error.PromptSpansNotAscending;
        prev = e;
    }
    if (prev != pos0 + rows) return error.PromptSpansNotThePrompt;
    for (cps) |c| if (std.mem.indexOfScalar(u64, ends, c) == null) return error.PromptCheckpointOffSpan;
}

/// Each span's rows into `out` (`ends.len` of them; `check` holds).
pub fn rowsOf(pos0: u64, ends: []const u64, out: []u32) !void {
    var prev = pos0;
    for (ends, out) |e, *r| {
        r.* = std.math.cast(u32, e - prev) orelse return error.PromptSpanTooWide;
        prev = e;
    }
}

/// The index of the pair's first span (the last chunk), or null: every span runs alone. The last span is the held-back
/// tail (`max_tail` rows or fewer) and both spans are wider than `min_rows`.
pub fn pairAt(rows: []const u32) ?usize {
    if (rows.len < 2) return null;
    const a = rows[rows.len - 2];
    const b = rows[rows.len - 1];
    if (b > max_tail or b <= min_rows or a <= min_rows) return null;
    return rows.len - 2;
}

/// What of a prompt forward decides whether its pair may run: one slot, no capture layers, no media rows, and the head
/// over the last row only (with `.all_rows` every span's logits come from its own forward).
pub const Facts = struct { slots: usize = 1, captures: usize = 0, media: bool = false, all_rows: bool = false };

/// `pairAt` for a forward of `facts`: null (every span alone) where the forward is not a pair's.
pub fn pairOf(rows: []const u32, facts: Facts) ?usize {
    _ = .{ rows, facts };
    return null;
}

// ── tests ──

const testing = std.testing;

test "glm53f span plan: the spans cover the prompt in order and every checkpoint ends one; anything else is refused by name" {
    try check(100, 40, &.{ 110, 130, 140 }, &.{ 110, 130 });
    try check(0, 5, &.{5}, &.{});
    try testing.expectError(error.PromptSpansEmpty, check(0, 5, &.{}, &.{}));
    try testing.expectError(error.PromptSpansNotAscending, check(100, 40, &.{ 130, 110, 140 }, &.{}));
    try testing.expectError(error.PromptSpansNotAscending, check(100, 40, &.{ 100, 140 }, &.{}));
    try testing.expectError(error.PromptSpansNotThePrompt, check(100, 40, &.{ 110, 139 }, &.{}));
    try testing.expectError(error.PromptCheckpointOffSpan, check(100, 40, &.{ 110, 140 }, &.{120}));
}

test "glm53f span plan: each span's rows from the prompt's position" {
    var rows: [3]u32 = undefined;
    try rowsOf(100, &.{ 110, 130, 140 }, &rows);
    try testing.expectEqualSlices(u32, &.{ 10, 20, 10 }, &rows);
    var big: [1]u32 = undefined;
    try testing.expectError(error.PromptSpanTooWide, rowsOf(0, &.{@as(u64, std.math.maxInt(u32)) + 1}, &big));
}

test "glm53f span plan: the last chunk and a held-back tail of 17 to 31 rows are a pair; any other ending runs alone" {
    // the 16K cell's chunk and its tail, as #1196's split took them
    try testing.expectEqual(@as(?usize, 0), pairAt(&.{ 16_356, 31 }));
    try testing.expectEqual(@as(?usize, 2), pairAt(&.{ 16_384, 16_384, 8_000, 17 }));
    // a tail wider than the backoff's (no checkpoints: a full last chunk), one too narrow to collapse, one row (the
    // prefix cache off), a chunk too narrow, a single span
    try testing.expectEqual(@as(?usize, null), pairAt(&.{ 16_384, 32 }));
    try testing.expectEqual(@as(?usize, null), pairAt(&.{ 16_384, 16 }));
    try testing.expectEqual(@as(?usize, null), pairAt(&.{ 16_384, 1 }));
    try testing.expectEqual(@as(?usize, null), pairAt(&.{ 16, 31 }));
    try testing.expectEqual(@as(?usize, null), pairAt(&.{31}));
    try testing.expectEqual(@as(?usize, null), pairAt(&.{}));
}

test "glm53f span plan: a pair only for one slot's prompt with no captures, no media and the last row's logits" {
    const rows = [_]u32{ 16_356, 31 };
    try testing.expectEqual(@as(?usize, 0), pairOf(&rows, .{}));
    try testing.expectEqual(@as(?usize, null), pairOf(&rows, .{ .slots = 2 }));
    try testing.expectEqual(@as(?usize, null), pairOf(&rows, .{ .captures = 1 }));
    try testing.expectEqual(@as(?usize, null), pairOf(&rows, .{ .media = true }));
    try testing.expectEqual(@as(?usize, null), pairOf(&rows, .{ .all_rows = true }));
    try testing.expectEqual(@as(?usize, null), pairOf(&.{ 16_356, 1 }, .{}));
}
