//! C4, the bounded layer-major pass, its plan (MLX-free): where the pass cuts a run of wide prompt spans into segments
//! and the MTP head's history into forwards. plugin.zig `spanMajorOr` runs each run through src/ling_lm_pass.zig one
//! segment at a time, each segment through every layer before the next (the copied table_forward.zig `hiddenSpans`).
//! - A run's routed calls are the whole run's (`routedCalls`, the copied moeSpans' rule before C4, moved here verbatim):
//!   a segment is a run of whole calls, and `hiddenSpans` takes its calls from here, so no segment regroups a call.
//! - A segment takes the next call while its rows stay at or under `segment_rows`, and always at least one call; a last
//!   group of calls shorter than `segment_tail_rows` joins the segment before it. So every whole prompt up to 32,768
//!   tokens is one segment, today's single pass (a test sweeps them), and a segment is at most 40,959 rows.
//! - The tail rule keeps mlx-serve's KV at the single pass's capacity for every whole prompt (src/ling_rq.zig's sweep to
//!   150,999 tokens): an MLA layer's appends end where a segment ends (D2's group a segment), and from 32,768 tokens on a
//!   growth adds at least 8,192 tokens, so a last segment under that would grow the KV past the prompt (at 32,800-33,310
//!   tokens the first segment ended at 28,672 and the KV grew 25,600 -> 32,000 -> 40,192, where one group gives 33,280).
//! - The history of a prompt over 32,768 tokens goes to the MTP head in forwards of at most `history_rows` rows
//!   (`historyRows`); a prompt of 32,768 tokens or fewer keeps today's one history forward.
//! A segment's words are the single pass's: every op whose bytes may depend on its call's rows runs on one span at a
//! time, as in the single pass, and each routed call is the single pass's call (`calls`). The GPU byte test holds the two
//! passes to the same words (src/ling_lm_pass_test.zig).
const std = @import("std");

/// A layer-major routed call's rows: the calls take whole spans up to this many rows (plugin.zig's `layer_major_rows`).
pub const routed_rows: u32 = 16384;
/// A last group of a run's spans shorter than this joins the call before it (it would re-read the experts the call
/// before it held; plugin.zig's `layer_major_tail_rows`).
pub const tail_rows: u32 = 2048;
/// A segment's rows at most, unless one routed call is wider or a last group joins it: the bill's 32,768-token context of
/// today's single pass.
pub const segment_rows: u32 = 32768;
/// A last group of a run's calls shorter than this joins the segment before it: mlx-serve's growth step from 32,768
/// tokens on (src/ling_rq.zig `nextCapacity`: a quarter, at most 8,192), so no segment's end makes the KV grow past the
/// single pass's.
pub const segment_tail_rows: u32 = 8192;
/// The MTP head's history forward of a prompt over 32,768 tokens: forwards of at most this many rows.
pub const history_rows: u32 = 4096;
/// The longest history the head takes in one forward: a 32,768-token prompt's (its rows less one).
pub const history_whole_max: u64 = segment_rows - 1;

/// The whole run's routed calls over its spans `spans` (each span's rows, in order), the copied moeSpans' rule: a call
/// takes the next span while its rows stay at or under `routed`, or while the rows left after it are fewer than `tail`
/// (so a last group that small joins the call before it); a span wider than `routed` is a call alone. Each call's end
/// (an index into `spans`, past its last span) goes into `ends`; past `ends.len` calls, error.LayerMajorTooManyCalls.
pub fn routedCalls(spans: []const u32, routed: u32, tail: u32, ends: []u32) error{LayerMajorTooManyCalls}![]const u32 {
    var left: u64 = 0;
    for (spans) |r| left += r;
    var k: usize = 0;
    var j0: usize = 0;
    const n = spans.len;
    while (j0 < n) {
        var j1 = j0 + 1;
        var rows: u64 = spans[j0];
        while (j1 < n and (rows + spans[j1] <= routed or left - rows < tail)) : (j1 += 1) rows += spans[j1];
        left -= rows;
        if (k == ends.len) return error.LayerMajorTooManyCalls;
        ends[k] = @intCast(j1);
        k += 1;
        j0 = j1;
    }
    return ends[0..k];
}

/// A run's segments over its calls `calls` (`routedCalls`): each segment's end (a call's end), whole calls while the
/// segment's rows stay at or under `max_rows`, at least one call, and no cut while the rows left after the segment are
/// fewer than `tail` (they join it). Past `ends.len` segments, error.LayerMajorTooManySegments.
pub fn segmentsOf(spans: []const u32, calls: []const u32, max_rows: u64, tail: u64, ends: []u32) error{LayerMajorTooManySegments}![]const u32 {
    var total: u64 = 0;
    for (spans) |r| total += r;
    var k: usize = 0;
    // the rows before the open segment, the open segment's rows, and the end of the last call it took
    var done: u64 = 0;
    var rows: u64 = 0;
    var prev: usize = 0;
    for (calls) |c| {
        var call_rows: u64 = 0;
        for (spans[prev..c]) |r| call_rows += r;
        // a cut before this call, unless the rows left after the open segment are fewer than `tail`: they join it
        if (rows > 0 and rows + call_rows > max_rows and total - (done + rows) >= tail) {
            if (k == ends.len) return error.LayerMajorTooManySegments;
            ends[k] = @intCast(prev);
            k += 1;
            done += rows;
            rows = 0;
        }
        rows += call_rows;
        prev = c;
    }
    if (calls.len > 0) {
        if (k == ends.len) return error.LayerMajorTooManySegments;
        ends[k] = @intCast(prev);
        k += 1;
    }
    return ends[0..k];
}

/// One segment of a run: its spans `[first, end)` (indices into the run's spans), its rows and its calls' ends, each
/// counted from `first` (what `hiddenSpans` takes).
pub const Segment = struct { first: usize, end: usize, rows: u64, calls: []const u32 };

/// The calls (and so the segments) a run's plan holds at most: 16 Mi rows at the served calls.
pub const max_calls = 1024;
/// The spans of one run that src/ling_rq.zig's replay plans at most (4 Mi rows at the host's 4,096-row spans).
pub const max_spans = 1024;

/// A run's segments in order (`Plan.init`, then `next`): the whole run's calls, cut into segments of at most
/// `max_rows` rows at call ends. The buffers hold a run of up to `max_calls` calls. A plan is used where it was
/// initialized (its slices point into its own buffers).
pub const Plan = struct {
    spans: []const u32,
    call_buf: [max_calls]u32 = undefined,
    seg_buf: [max_calls]u32 = undefined,
    rel_buf: [max_calls]u32 = undefined,
    calls: []const u32 = &.{},
    segs: []const u32 = &.{},
    i: usize = 0,
    ci: usize = 0,

    /// A run's plan at `max_rows` rows a segment (`segment_rows` as served), a last group under `segment_tail_rows`
    /// joined to the segment before it.
    pub fn init(p: *Plan, spans: []const u32, max_rows: u64) !void {
        p.* = .{ .spans = spans };
        p.calls = try routedCalls(spans, routed_rows, tail_rows, &p.call_buf);
        p.segs = try segmentsOf(spans, p.calls, max_rows, segment_tail_rows, &p.seg_buf);
    }

    /// The run's segments: their count.
    pub fn count(p: *const Plan) usize {
        return p.segs.len;
    }

    /// The next segment, its calls counted from its first span; null after the last. The calls slice is valid until the
    /// next call.
    pub fn next(p: *Plan) ?Segment {
        if (p.i >= p.segs.len) return null;
        const first: usize = if (p.i == 0) 0 else p.segs[p.i - 1];
        const end: usize = p.segs[p.i];
        p.i += 1;
        var n: usize = 0;
        while (p.ci < p.calls.len and p.calls[p.ci] <= end) : (p.ci += 1) {
            p.rel_buf[n] = @intCast(p.calls[p.ci] - first);
            n += 1;
        }
        var rows: u64 = 0;
        for (p.spans[first..end]) |r| rows += r;
        return .{ .first = first, .end = end, .rows = rows, .calls = p.rel_buf[0..n] };
    }
};

/// The MTP head's history forward of `rows` rows: the rows of each of its forwards, `history_rows` a forward past
/// `history_whole_max` rows (a prompt over 32,768 tokens), else one forward of all of them (today's).
pub fn historyRows(rows: u64) u64 {
    return if (rows > history_whole_max) history_rows else rows;
}

// ── tests ──

const testing = std.testing;

/// The copied table_forward.zig moeSpans' grouping before C4, verbatim (its `while` over `j0`), as the oracle: each
/// call's end into `ends`.
fn moeSpansOracle(spans: []const u32, routed: u32, tail: u32, ends: []u32) []const u32 {
    var left: u32 = 0;
    for (spans) |r| left += r;
    var k: usize = 0;
    var j0: usize = 0;
    const n = spans.len;
    while (j0 < n) {
        var j1 = j0 + 1;
        var rows: u32 = spans[j0];
        while (j1 < n and (rows + spans[j1] <= routed or left - rows < tail)) : (j1 += 1) rows += spans[j1];
        left -= rows;
        ends[k] = @intCast(j1);
        k += 1;
        j0 = j1;
    }
    return ends[0..k];
}

/// The host's whole-prompt spans of an `n`-token prompt from position 0 as rows (src/ling_rq.zig `spanEnds`), into `buf`
fn promptRows(buf: []u32, n: u64) []const u32 {
    var ends: [512]u64 = undefined;
    const e = ling_rq.spanEnds(&ends, n);
    var prev: u64 = 0;
    for (e, 0..) |x, j| {
        buf[j] = @intCast(x - prev);
        prev = x;
    }
    return buf[0..e.len];
}

const ling_rq = @import("ling_rq.zig");

/// A run's segment rows into `out`, at `max_rows`
fn segmentRows(out: []u64, spans: []const u32, max_rows: u64) ![]const u64 {
    var p: Plan = .{ .spans = spans };
    try p.init(spans, max_rows);
    var k: usize = 0;
    while (p.next()) |sg| : (k += 1) out[k] = sg.rows;
    return out[0..k];
}

test "C4's routed calls are the copied moeSpans' rule verbatim: the standard request's one call, the 32K check's two, and the host's spans of every whole prompt to 140,000 tokens and of short and odd span lists" {
    var a: [max_calls]u32 = undefined;
    var b: [max_calls]u32 = undefined;
    var rows: [512]u32 = undefined;
    // the standard request (18,380 tokens): one call of 18,380 rows
    const std_spans = [_]u32{ 4096, 4096, 4096, 4096, 1965, 31 };
    try testing.expectEqualSlices(u32, &.{6}, try routedCalls(&std_spans, routed_rows, tail_rows, &a));
    // 31,652 tokens (the 32K check): 16,384 + 15,268
    try testing.expectEqualSlices(u32, &.{ 4, 9 }, try routedCalls(promptRows(&rows, 31652), routed_rows, tail_rows, &a));
    var n: u64 = 2;
    while (n <= 140_000) : (n += 37) {
        const sp = promptRows(&rows, n);
        try testing.expectEqualSlices(u32, moeSpansOracle(sp, routed_rows, tail_rows, &b), try routedCalls(sp, routed_rows, tail_rows, &a));
    }
    // spans the host does not name from position 0: a span wider than a call, a small group in the middle, a tail
    const odd = [_][]const u32{
        &.{ 20000, 100, 4096 },  &.{ 16000, 1000, 16000, 3000 }, &.{ 8, 8, 8 }, &.{16385},
        &.{ 4096, 4096, 4096, 4096, 2047 }, &.{ 4096, 4096, 4096, 4096, 2048 }, &.{ 1, 16384, 1 },
    };
    for (odd) |sp| try testing.expectEqualSlices(u32, moeSpansOracle(sp, routed_rows, tail_rows, &b), try routedCalls(sp, routed_rows, tail_rows, &a));
    // the buffer's bound is a named error
    try testing.expectError(error.LayerMajorTooManyCalls, routedCalls(&.{ 20000, 20000, 20000 }, routed_rows, tail_rows, a[0..2]));
}

test "C4's segments: whole routed calls at or under 32,768 rows, a last group under 8,192 rows joined to the segment before it: one segment, today's single pass, for every whole prompt up to 32,768 tokens, 33,184 (the 16K point's context) and 40,959; two at 32,769 and 40,960; 33,591 tokens (the 32K point) 16,384 + 17,207, 67,310 (64K) 32,768 + 16,384 + 18,158, 134,615 (128K) 3 x 32,768 + 36,311" {
    var rows: [512]u32 = undefined;
    var out: [64]u64 = undefined;
    var n: u64 = 2;
    while (n <= 32768) : (n += 1) try testing.expectEqual(@as(usize, 1), (try segmentRows(&out, promptRows(&rows, n), segment_rows)).len);
    try testing.expectEqualSlices(u64, &.{ 16384, 16385 }, try segmentRows(&out, promptRows(&rows, 32769), segment_rows));
    // calls 16,384 + 12,288 + 4,512: the last joins (the 16K point's longest prompt)
    try testing.expectEqualSlices(u64, &.{33184}, try segmentRows(&out, promptRows(&rows, 33184), segment_rows));
    try testing.expectEqualSlices(u64, &.{ 16384, 17207 }, try segmentRows(&out, promptRows(&rows, 33591), segment_rows));
    // calls 16,384 + 16,384 + 8,191: the last joins; at 8,192 it does not
    try testing.expectEqualSlices(u64, &.{40959}, try segmentRows(&out, promptRows(&rows, 40959), segment_rows));
    try testing.expectEqualSlices(u64, &.{ 32768, 8192 }, try segmentRows(&out, promptRows(&rows, 40960), segment_rows));
    try testing.expectEqualSlices(u64, &.{ 32768, 16384, 18158 }, try segmentRows(&out, promptRows(&rows, 67310), segment_rows));
    try testing.expectEqualSlices(u64, &.{ 32768, 32768, 32768, 36311 }, try segmentRows(&out, promptRows(&rows, 134615), segment_rows));
    try testing.expectEqualSlices(u64, &.{ 32768, 32768, 32768, 32768, 19927 }, try segmentRows(&out, promptRows(&rows, 150999), segment_rows));
    // the byte test's arms at 52,000 tokens: one segment (A, no bound) and 16,384-row segments (B), D2's group cut at
    // 32,768 and the last call (2,848 rows) joined to the third
    try testing.expectEqualSlices(u64, &.{52000}, try segmentRows(&out, promptRows(&rows, 52000), std.math.maxInt(u64)));
    try testing.expectEqualSlices(u64, &.{ 16384, 16384, 19232 }, try segmentRows(&out, promptRows(&rows, 52000), 16384));
    // at 40,000 tokens 16,384-row segments take the last call (7,232 rows) into the second
    try testing.expectEqualSlices(u64, &.{ 16384, 23616 }, try segmentRows(&out, promptRows(&rows, 40000), 16384));
}

test "C4's segments cut only at the whole run's call ends: each segment's calls, from its first span, are the whole run's calls there, every segment is at or under its bound unless one call is wider or it is the run's last and the calls past its bound are under 8,192 rows, and the segments cover the run" {
    var rows: [512]u32 = undefined;
    var whole: [max_calls]u32 = undefined;
    var from: [max_calls]u32 = undefined;
    const bounds = [_]u64{ 16384, 20000, 32768, 50000 };
    var n: u64 = 2;
    while (n <= 140_000) : (n += 211) {
        const sp = promptRows(&rows, n);
        const calls = try routedCalls(sp, routed_rows, tail_rows, &whole);
        for (bounds) |bound| {
            var p: Plan = .{ .spans = sp };
            try p.init(sp, bound);
            var covered: usize = 0;
            var ci: usize = 0;
            while (p.next()) |sg| {
                try testing.expectEqual(covered, sg.first);
                // the whole run's calls from here, rebased
                var k: usize = 0;
                while (ci < calls.len and calls[ci] <= sg.end) : (ci += 1) {
                    try testing.expectEqual(calls[ci] - @as(u32, @intCast(sg.first)), sg.calls[k]);
                    k += 1;
                }
                try testing.expectEqual(k, sg.calls.len);
                try testing.expectEqual(@as(u32, @intCast(sg.end - sg.first)), sg.calls[sg.calls.len - 1]);
                // the rule from this segment's first span alone gives the same calls (a call end is a fresh start)
                try testing.expectEqualSlices(u32, sg.calls[0..k], (try routedCalls(sp[sg.first..], routed_rows, tail_rows, &from))[0..k]);
                if (sg.calls.len > 1 and sg.rows > bound) {
                    // over its bound: the run's last segment, and the calls past its bound fewer than 8,192 rows
                    try testing.expectEqual(sp.len, sg.end);
                    var within: u64 = 0;
                    var cut: usize = 0;
                    for (sg.calls, 0..) |c, ci2| {
                        var r: u64 = 0;
                        for (sp[sg.first + (if (ci2 == 0) 0 else sg.calls[ci2 - 1]) .. sg.first + c]) |x| r += x;
                        if (ci2 > 0 and within + r > bound) break;
                        within += r;
                        cut = c;
                    }
                    var past: u64 = 0;
                    for (sp[sg.first + cut .. sg.end]) |x| past += x;
                    try testing.expect(past < segment_tail_rows);
                }
                covered = sg.end;
            }
            try testing.expectEqual(sp.len, covered);
        }
    }
}

test "C4's MTP history: one head forward for a prompt of 32,768 tokens or fewer (its history 32,767 rows or fewer), forwards of at most 4,096 rows past that" {
    try testing.expectEqual(@as(u64, 18379), historyRows(18379));
    try testing.expectEqual(@as(u64, 32767), historyRows(32767));
    try testing.expectEqual(@as(u64, 4096), historyRows(32768));
    try testing.expectEqual(@as(u64, 4096), historyRows(67309));
    try testing.expectEqual(@as(u64, 4096), historyRows(134614));
    try testing.expectEqual(@as(u64, 1), historyRows(1));
}
