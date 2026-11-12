//! The span bookkeeping of turboexl3_ling's MoE tail at prompt widths (src/ling_pf_tail.zig, PF_MOE_TAIL),
//! MLX-free: for an armed layer-major MoE layer, which of its routed calls each span's rows went through and where they
//! start in that call (the patched table_forward.zig `moeSpans` groups spans into calls in order: a call over spans
//! j0..j1 takes their rows back to back).
const std = @import("std");

/// The spans a pass takes at most (table_forward.zig `moeSpans`' group arrays hold 64)
pub const max_spans: usize = 64;

pub const Plan = struct {
    /// The armed layer's spans (0: not armed)
    n_spans: usize = 0,
    /// Its routed calls so far
    n_calls: usize = 0,
    /// Each span's call (null: no call took it yet)
    call_of: [max_spans]?u8 = @splat(null),
    /// Each span's first row in its call, and its rows
    at: [max_spans]c_int = @splat(0),
    rows: [max_spans]c_int = @splat(0),

    /// A layer of `spans` spans armed, whatever was there before dropped; false (not armed) past `max_spans`
    pub fn arm(p: *Plan, spans: usize) bool {
        p.* = .{};
        if (spans == 0 or spans > max_spans) return false;
        p.n_spans = spans;
        return true;
    }

    /// A routed call over spans `j0`..`j1` of `spans` (each span's rows): its index, each span's first row in it. Null
    /// (no call recorded) when not armed, past the armed spans, or for an empty group.
    pub fn beginCall(p: *Plan, j0: usize, j1: usize, spans: []const u32) ?usize {
        if (p.n_spans == 0 or j0 >= j1 or j1 > p.n_spans or j1 > spans.len or p.n_calls >= max_spans) return null;
        const c = p.n_calls;
        p.n_calls += 1;
        var first: c_int = 0;
        for (j0..j1) |j| {
            p.call_of[j] = @intCast(c);
            p.at[j] = first;
            p.rows[j] = @intCast(spans[j]);
            first += @intCast(spans[j]);
        }
        return c;
    }

    /// Span `j`'s call and its rows' place there; null when no call took it
    pub fn spanOf(p: *const Plan, j: usize) ?struct { call: usize, at: c_int, rows: c_int } {
        if (j >= p.n_spans) return null;
        const c = p.call_of[j] orelse return null;
        return .{ .call = c, .at = p.at[j], .rows = p.rows[j] };
    }
};

const testing = std.testing;

test "the plan: the standard request's spans in one call take their rows back to back; two calls restart at 0; a span no call took has no place" {
    var p: Plan = .{};
    const spans = [_]u32{ 4096, 4096, 4096, 4096, 1965, 31 };
    try testing.expect(p.arm(spans.len));
    try testing.expectEqual(@as(?usize, 0), p.beginCall(0, spans.len, &spans));
    const want_at = [_]c_int{ 0, 4096, 8192, 12288, 16384, 18349 };
    for (want_at, spans, 0..) |a, r, j| {
        const sp = p.spanOf(j).?;
        try testing.expectEqual(@as(usize, 0), sp.call);
        try testing.expectEqual(a, sp.at);
        try testing.expectEqual(@as(c_int, @intCast(r)), sp.rows);
    }
    try testing.expect(p.spanOf(spans.len) == null);
    // two calls (a longer prompt's groups): the second's spans start at its own row 0
    const long = [_]u32{ 4096, 4096, 4096, 4096, 4096, 2500, 31 };
    try testing.expect(p.arm(long.len));
    try testing.expectEqual(@as(?usize, 0), p.beginCall(0, 4, &long));
    try testing.expect(p.spanOf(4) == null);
    try testing.expectEqual(@as(?usize, 1), p.beginCall(4, long.len, &long));
    try testing.expectEqual(@as(usize, 1), p.spanOf(4).?.call);
    try testing.expectEqual(@as(c_int, 0), p.spanOf(4).?.at);
    try testing.expectEqual(@as(c_int, 4096), p.spanOf(5).?.at);
    try testing.expectEqual(@as(c_int, 6596), p.spanOf(6).?.at);
    try testing.expectEqual(@as(c_int, 12288), p.spanOf(3).?.at);
}

test "the plan's refusals: no arm past max_spans or at 0 spans; no call before arm, past the spans, or empty; re-arming drops the old calls" {
    var p: Plan = .{};
    const spans = [_]u32{ 100, 200 };
    try testing.expectEqual(@as(?usize, null), p.beginCall(0, 2, &spans));
    try testing.expect(!p.arm(0));
    try testing.expect(!p.arm(max_spans + 1));
    try testing.expect(p.arm(2));
    try testing.expectEqual(@as(?usize, null), p.beginCall(1, 1, &spans));
    try testing.expectEqual(@as(?usize, null), p.beginCall(0, 3, &spans));
    try testing.expectEqual(@as(?usize, 0), p.beginCall(0, 2, &spans));
    try testing.expect(p.arm(2));
    try testing.expect(p.spanOf(0) == null and p.n_calls == 0);
}
