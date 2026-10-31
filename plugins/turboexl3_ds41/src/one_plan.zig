//! The host plan of `MLX_SERVE_DSV41_ROUTED_ONE` (src/routed_one.zig), MLX-free: a wide prompt call's experts split into
//! the resident ones (a persistent slot of the layer's base bank) and the streamed ones, whether the call's resident
//! pairs run as one turboexl3 call, that call's tables over the router's [rows, top-k] layout, and where each streamed
//! pair's row lands in the combine's source.

const std = @import("std");

/// `MLX_SERVE_DSV41_ROUTED_ONE`:
/// - `off` (unset, empty or `0`): every wide call keeps the streamer's wide lane.
/// - `on` (`1`): a call whose experts are all resident is one turboexl3 call; a call with streamed experts runs its
///   resident pairs as that call (each streamed pair in it at weight 0) and its streamed experts on the streamer's
///   schedule, when the resident pairs are at least half the call's (`takes`).
/// - `resident`: only the calls whose experts are all resident; every other call keeps the wide lane.
/// - `check`: `on`'s calls run both ways, the wide lane's sum served, each comparison logged.
pub const Mode = enum { off, on, resident, check };

/// The setting's mode, or `error.RoutedOneSetting` for a value it does not name (the load refuses it by name).
pub fn modeOf(value: ?[]const u8) error{RoutedOneSetting}!Mode {
    const v = value orelse return .off;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "resident")) return .resident;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.RoutedOneSetting;
}

/// An expert with no base-bank row in this call.
pub const no_slot: u32 = std.math.maxInt(u32);

/// A call's experts, the resident ones first: each part hottest first (pairs descending, ids ascending, the wide lane's
/// feed order), with the call's pairs (rows x top-k) and those of its resident experts.
pub const Split = struct {
    distinct: []const u16,
    n_resident: usize,
    pairs: usize,
    resident_pairs: usize,

    pub fn resident(s: Split) []const u16 {
        return s.distinct[0..s.n_resident];
    }

    pub fn streamed(s: Split) []const u16 {
        return s.distinct[s.n_resident..];
    }

    pub fn allResident(s: Split) bool {
        return s.n_resident == s.distinct.len;
    }
};

/// The call's experts split (`Split`): `ids` its routed ids (rows x top-k, each < `is_resident.len`), `is_resident[e]`
/// whether expert e holds a persistent slot of the layer. `count` [experts] gets each expert's pairs, `out` [experts]
/// holds the split's experts.
pub fn split(ids: []const u16, is_resident: []const bool, count: []u32, out: []u16) Split {
    std.debug.assert(count.len == is_resident.len and out.len == is_resident.len);
    @memset(count, 0);
    for (ids) |e| count[e] += 1;
    var n: usize = 0;
    var resident_pairs: usize = 0;
    for (count, 0..) |c, e| if (c > 0) {
        out[n] = @intCast(e);
        n += 1;
        if (is_resident[e]) resident_pairs += c;
    };
    const Order = struct {
        count: []const u32,
        resident: []const bool,
        fn before(o: @This(), x: u16, y: u16) bool {
            if (o.resident[x] != o.resident[y]) return o.resident[x];
            if (o.count[x] != o.count[y]) return o.count[x] > o.count[y];
            return x < y;
        }
    };
    std.sort.pdq(u16, out[0..n], Order{ .count = count, .resident = is_resident }, Order.before);
    var n_resident: usize = 0;
    while (n_resident < n and is_resident[out[n_resident]]) n_resident += 1;
    return .{ .distinct = out[0..n], .n_resident = n_resident, .pairs = ids.len, .resident_pairs = resident_pairs };
}

/// Whether the call's resident pairs run as one call. Every resident: always (the one call is the whole call). With
/// streamed experts (`on` and `check` only): when the resident pairs are at least half the call's. The one call computes
/// every pair, a streamed one at weight 0, at about twice the wide lane's rate (turboexl3's `moeClamped` over 16384 rows
/// x top-6 in 141 ms a layer, the wide lane's call 302 ms a layer, on the M5 Max), so it pays from about half; below,
/// the call keeps the wide lane.
pub fn takes(mode: Mode, s: Split) bool {
    if (mode == .off or s.n_resident == 0) return false;
    if (s.allResident()) return true;
    if (mode == .resident) return false;
    return 2 * s.resident_pairs >= s.pairs;
}

/// The one call's tables over the [rows, top-k] layout. `slot_of[e]` resident expert e's base-bank row (`no_slot` for
/// every other expert); `resident` the call's resident experts. Pair p gets `slots[p]`, its expert's row, and
/// `keep[p]` 1; a streamed pair gets a resident expert's row (the resident experts in turn, so no one expert takes them
/// all) and `keep[p]` 0, its weight in the call. Returns the streamed (masked) pairs.
pub fn table(ids: []const u16, slot_of: []const u32, resident: []const u16, slots: []u32, keep: []f32) usize {
    std.debug.assert(resident.len > 0 and slots.len == ids.len and keep.len == ids.len);
    var masked: usize = 0;
    for (ids, slots, keep) |e, *s, *kp| {
        if (slot_of[e] != no_slot) {
            s.* = slot_of[e];
            kp.* = 1;
        } else {
            s.* = slot_of[resident[masked % resident.len]];
            kp.* = 0;
            masked += 1;
        }
    }
    return masked;
}

/// The call's pairs bucketed by expert, rows ascending (a stable counting sort): expert e's pairs are
/// `pairs[start[e]..][0..count[e]]`. `count` is `split`'s; `start` [experts], `pairs` [ids.len].
pub fn bucket(ids: []const u16, count: []const u32, start: []u32, pairs: []u32) void {
    std.debug.assert(start.len == count.len and pairs.len == ids.len);
    var at: u32 = 0;
    for (count, start) |c, *s| {
        s.* = at;
        at += c;
    }
    // Each pair at its expert's cursor, then the cursors put back to the starts.
    for (ids, 0..) |e, p| {
        pairs[start[e]] = @intCast(p);
        start[e] += 1;
    }
    for (count, start) |c, *s| s.* -= c;
}

/// The streamed experts' groups of at most `width` (a route's ids), in order.
pub fn groups(n_streamed: usize, width: usize) usize {
    return (n_streamed + width - 1) / width;
}

pub fn group(streamed: []const u16, width: usize, g: usize) []const u16 {
    const lo = g * width;
    return streamed[lo..@min(lo + width, streamed.len)];
}

/// One wave call's rows in the combine's source: `pos` its pairs (routed positions) in the call's row order, the
/// call's first row at `at` of the source (its outputs concatenated in call order). Returns the next call's `at`.
pub fn place(loc: []i32, pos: []const u32, at: u32) u32 {
    for (pos, 0..) |p, j| loc[p] = @intCast(at + j);
    return at + @as(u32, @intCast(pos.len));
}

/// Whether `loc` places every streamed pair once and no resident one: each streamed pair (`slot_of` `no_slot`) a row
/// below `rows`, the rows a permutation of 0..rows, each resident pair -1 (`seen`: scratch, [rows]).
pub fn placedOnce(ids: []const u16, slot_of: []const u32, loc: []const i32, rows: usize, seen: []bool) bool {
    std.debug.assert(seen.len == rows);
    @memset(seen, false);
    var placed: usize = 0;
    for (ids, loc) |e, l| {
        if (slot_of[e] != no_slot) {
            if (l != -1) return false;
            continue;
        }
        if (l < 0 or @as(usize, @intCast(l)) >= rows or seen[@intCast(l)]) return false;
        seen[@intCast(l)] = true;
        placed += 1;
    }
    return placed == rows;
}

// ── tests ──

const testing = std.testing;

test "routed one: the setting names its modes; any other value is refused by name" {
    for ([_]?[]const u8{ null, "", "0" }) |v| try testing.expectEqual(Mode.off, try modeOf(v));
    try testing.expectEqual(Mode.on, try modeOf("1"));
    try testing.expectEqual(Mode.resident, try modeOf("resident"));
    try testing.expectEqual(Mode.check, try modeOf("check"));
    for ([_][]const u8{ "2", "on", "true", "1 ", "Resident", "checks" }) |v| try testing.expectError(error.RoutedOneSetting, modeOf(v));
}

test "routed one: a call's experts split resident first, each part hottest first (ties by id), with their pairs" {
    // 6 experts; 3 rows x top-2: expert 4 x2, 1 x1, 5 x1, 0 x2; resident: 1 and 4 (and 2, which the call does not route)
    const ids = [_]u16{ 4, 1, 0, 5, 4, 0 };
    const res = [_]bool{ false, true, true, false, true, false };
    var count: [6]u32 = undefined;
    var out: [6]u16 = undefined;
    const s = split(&ids, &res, &count, &out);
    try testing.expectEqualSlices(u16, &.{ 4, 1, 0, 5 }, s.distinct);
    try testing.expectEqualSlices(u16, &.{ 4, 1 }, s.resident());
    try testing.expectEqualSlices(u16, &.{ 0, 5 }, s.streamed());
    try testing.expectEqual(@as(usize, 6), s.pairs);
    try testing.expectEqual(@as(usize, 3), s.resident_pairs);
    try testing.expectEqualSlices(u32, &.{ 2, 1, 0, 0, 2, 1 }, &count);
    try testing.expect(!s.allResident());
    // every expert resident: one part, hottest first, ties by id
    const all: [6]bool = @splat(true);
    const t = split(&ids, &all, &count, &out);
    try testing.expectEqualSlices(u16, &.{ 0, 4, 1, 5 }, t.distinct);
    try testing.expect(t.allResident());
    try testing.expectEqual(@as(usize, 6), t.resident_pairs);
}

test "routed one: a call takes the one call when all its experts are resident, or (on, check) half its pairs are" {
    const all: Split = .{ .distinct = &.{ 1, 2 }, .n_resident = 2, .pairs = 12, .resident_pairs = 12 };
    const half: Split = .{ .distinct = &.{ 1, 2 }, .n_resident = 1, .pairs = 12, .resident_pairs = 6 };
    const less: Split = .{ .distinct = &.{ 1, 2 }, .n_resident = 1, .pairs = 12, .resident_pairs = 5 };
    const none: Split = .{ .distinct = &.{ 1, 2 }, .n_resident = 0, .pairs = 12, .resident_pairs = 0 };
    for ([_]Mode{ .on, .resident, .check }) |m| {
        try testing.expect(takes(m, all));
        try testing.expect(!takes(m, less));
        try testing.expect(!takes(m, none));
    }
    try testing.expect(takes(.on, half) and takes(.check, half) and !takes(.resident, half));
    for ([_]Split{ all, half, less, none }) |s| try testing.expect(!takes(.off, s));
}

test "routed one: the one call's table keeps each resident pair's row and weight, spreads the streamed ones at weight 0" {
    const ids = [_]u16{ 4, 1, 0, 5, 4, 0 };
    var slot_of: [6]u32 = @splat(no_slot);
    slot_of[4] = 7;
    slot_of[1] = 2;
    const resident = [_]u16{ 4, 1 };
    var slots: [6]u32 = undefined;
    var keep: [6]f32 = undefined;
    const masked = table(&ids, &slot_of, &resident, &slots, &keep);
    try testing.expectEqual(@as(usize, 3), masked);
    // streamed pairs 2, 3, 5 take experts 4, 1, 4's rows in turn
    try testing.expectEqualSlices(u32, &.{ 7, 2, 7, 2, 7, 7 }, &slots);
    try testing.expectEqualSlices(f32, &.{ 1, 1, 0, 0, 1, 0 }, &keep);
    // all resident: every pair its own row, nothing masked
    slot_of[0] = 0;
    slot_of[5] = 3;
    try testing.expectEqual(@as(usize, 0), table(&ids, &slot_of, &resident, &slots, &keep));
    try testing.expectEqualSlices(u32, &.{ 7, 2, 0, 3, 7, 0 }, &slots);
    try testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1, 1 }, &keep);
}

test "routed one: the pairs bucket by expert in row order, and the streamed groups cover the streamed experts once" {
    const ids = [_]u16{ 4, 1, 0, 5, 4, 0 };
    const count = [_]u32{ 2, 1, 0, 0, 2, 1 };
    var start: [6]u32 = undefined;
    var pairs: [6]u32 = undefined;
    bucket(&ids, &count, &start, &pairs);
    try testing.expectEqualSlices(u32, &.{ 0, 2, 3, 3, 3, 5 }, &start);
    try testing.expectEqualSlices(u32, &.{ 2, 5, 1, 0, 4, 3 }, &pairs);
    const streamed = [_]u16{ 9, 8, 7, 6, 5 };
    try testing.expectEqual(@as(usize, 3), groups(streamed.len, 2));
    try testing.expectEqual(@as(usize, 1), groups(streamed.len, 48));
    try testing.expectEqualSlices(u16, &.{ 9, 8 }, group(&streamed, 2, 0));
    try testing.expectEqualSlices(u16, &.{ 7, 6 }, group(&streamed, 2, 1));
    try testing.expectEqualSlices(u16, &.{5}, group(&streamed, 2, 2));
}

test "routed one: the streamed pairs' rows in the combine's source are each placed once, the resident pairs none" {
    // pairs 2, 3, 5 streamed (experts 0 and 5): a transient call over pairs 3, 2, then a last call over pair 5
    const ids = [_]u16{ 4, 1, 0, 5, 4, 0 };
    var slot_of: [6]u32 = @splat(no_slot);
    slot_of[4] = 7;
    slot_of[1] = 2;
    var loc: [6]i32 = @splat(-1);
    var at = place(&loc, &.{ 3, 2 }, 0);
    try testing.expectEqual(@as(u32, 2), at);
    at = place(&loc, &.{5}, at);
    try testing.expectEqual(@as(u32, 3), at);
    try testing.expectEqualSlices(i32, &.{ -1, -1, 1, 0, -1, 2 }, &loc);
    var seen: [3]bool = undefined;
    try testing.expect(placedOnce(&ids, &slot_of, &loc, 3, &seen));
    // a streamed pair left out, placed twice, or a resident one placed: refused
    var missing = loc;
    missing[5] = -1;
    try testing.expect(!placedOnce(&ids, &slot_of, &missing, 3, &seen));
    var twice = loc;
    twice[5] = 0;
    try testing.expect(!placedOnce(&ids, &slot_of, &twice, 3, &seen));
    var resident_placed = loc;
    resident_placed[0] = 2;
    try testing.expect(!placedOnce(&ids, &slot_of, &resident_placed, 3, &seen));
}
