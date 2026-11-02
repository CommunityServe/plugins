//! `[cold-start]` (dq1b's report, main 2026-11-02; on with `[round-io]`, the DSpark profile), MLX-free: the decode's first
//! three routed calls of each layer (c0 = the first verify's, r0) against the layer's last prompt call. The decode's first
//! rounds miss heavily in every request (r0 100-272 misses against ~32 in a warm round, dx1's timelines). The lever
//! this report sizes would fill the decode rows at the phase change from the prompt's last routes (timing only).
//!
//! What the copied stream hands it:
//!   - every prompt call's routed ids (`prompt`: the call's rows x top-k, in row order). A later call of the same layer
//!     replaces the earlier one, so the prompt's last call stands.
//!   - each decode call's ids and their waves (`decode`: wave 0 means the hit wave, anything else a miss part).
//!
//! What it counts, for decode call c = 0, 1, 2 of each layer:
//!   - the call's distinct experts, and those it missed;
//!   - of the missed, those the layer's last prompt call routed at all, those its last 8 rows routed (the tail), and
//!     those among its 6, 12, 24 and 48 most voted experts (votes = the rows that chose it; ties go to the lower id);
//!   - `nocall`: the misses of layers with no prompt call in the request (a fully cached prompt).
//!
//! `r0_by_layer`: per layer, c0's misses / those in the tail / those in the top 12.
//!
//! At each request's end it prints one line, then clears. Reads only: no byte served moves.

const std = @import("std");

/// Set at load (with `[round-io]`).
pub var on: bool = false;

pub const max_layers = 64;
pub const max_experts = 512;
/// The prompt rows whose experts make the tail set: the last call's last rows.
pub const tail_rows = 8;
/// The decode calls counted a layer.
pub const calls = 3;
/// The ranks among the last prompt call's experts (by votes) counted.
pub const tops = [_]u16{ 6, 12, 24, 48 };

const Set = std.StaticBitSet(max_experts);

const Layer = struct {
    /// The last prompt call's rows choosing each expert.
    votes: [max_experts]u16 = @splat(0),
    /// Each voted expert's rank by votes (computed at the layer's first decode call).
    rank: [max_experts]u16 = @splat(0),
    tail: Set = .empty,
    rows: u32 = 0,
    distinct: u16 = 0,
    have: bool = false,
    ranked: bool = false,
    /// The decode calls counted since the prompt call (or the request's start).
    seen: u8 = 0,
};

pub const Count = struct {
    calls: u64 = 0,
    routed: u64 = 0,
    missed: u64 = 0,
    in_call: u64 = 0,
    in_tail: u64 = 0,
    top: [tops.len]u64 = @splat(0),
    nocall: u64 = 0,
};

var layers: [max_layers]Layer = @splat(.{});
var counts: [calls]Count = @splat(.{});
/// c0 per layer: its misses, those in the tail, those in the top 12.
var r0: [max_layers][3]u16 = @splat(@splat(0));
var r0_seen: [max_layers]bool = @splat(false);

/// A prompt call of `layer`: `ids` its `n` rows x `k` routed ids, row order.
pub fn prompt(layer: u32, ids: []const u16, n: u32, k: u32) void {
    if (layer >= max_layers or n == 0 or k == 0) return;
    const l = &layers[layer];
    const all = ids[0..@min(ids.len, @as(usize, n) * k)];
    l.votes = @splat(0);
    var d: u16 = 0;
    for (all) |e| {
        if (e >= max_experts) continue;
        if (l.votes[e] == 0) d += 1;
        l.votes[e] +|= 1;
    }
    l.tail = .empty;
    const t0 = (@as(usize, n) - @min(n, tail_rows)) * k;
    if (t0 < all.len) {
        for (all[t0..]) |e| {
            if (e < max_experts) l.tail.set(e);
        }
    }
    l.rows = n;
    l.distinct = d;
    l.have = true;
    l.ranked = false;
    l.seen = 0;
}

fn rankOf(l: *Layer) void {
    var order: [max_experts]u16 = undefined;
    var n: usize = 0;
    for (l.votes, 0..) |v, e| {
        if (v == 0) continue;
        order[n] = @intCast(e);
        n += 1;
    }
    const votes: *const [max_experts]u16 = &l.votes;
    std.sort.pdq(u16, order[0..n], votes, struct {
        fn lt(v: *const [max_experts]u16, a: u16, b: u16) bool {
            return if (v[a] != v[b]) v[a] > v[b] else a < b;
        }
    }.lt);
    for (order[0..n], 0..) |e, i| l.rank[e] = @intCast(i);
    l.ranked = true;
}

/// A decode call of `layer`: its routed `ids` and each one's wave (0 the hit wave, else a miss part).
pub fn decode(layer: u32, ids: []const u16, waves: []const u8) void {
    if (layer >= max_layers) return;
    const l = &layers[layer];
    if (l.seen >= calls) return;
    const c = l.seen;
    l.seen += 1;
    var routed: Set = .empty;
    var missed: Set = .empty;
    for (ids, 0..) |e, i| {
        if (e >= max_experts) continue;
        routed.set(e);
        if (i < waves.len and waves[i] > 0) missed.set(e);
    }
    const ct = &counts[c];
    ct.calls += 1;
    ct.routed += routed.count();
    const nm = missed.count();
    ct.missed += nm;
    var in_tail: u16 = 0;
    var top12: u16 = 0;
    if (!l.have) {
        ct.nocall += nm;
    } else {
        if (!l.ranked) rankOf(l);
        var it = missed.iterator(.{});
        while (it.next()) |e| {
            if (l.votes[e] == 0) continue;
            ct.in_call += 1;
            if (l.tail.isSet(e)) {
                ct.in_tail += 1;
                in_tail += 1;
            }
            for (tops, 0..) |t, i| {
                if (l.rank[e] < t) ct.top[i] += 1;
            }
            if (l.rank[e] < 12) top12 += 1;
        }
    }
    if (c == 0) {
        r0[layer] = .{ @intCast(nm), in_tail, top12 };
        r0_seen[layer] = true;
    }
}

/// Every count cleared and every layer's prompt record dropped (a request's end).
pub fn clear() void {
    counts = @splat(.{});
    r0_seen = @splat(false);
    for (&layers) |*l| {
        l.have = false;
        l.ranked = false;
        l.seen = 0;
    }
}

/// `[cold-start]`, the request's (then cleared); "" off or with no decode call.
pub fn meters(buf: []u8) []const u8 {
    if (!on) return buf[0..0];
    defer clear();
    if (counts[0].calls == 0) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    write(&w) catch return w.buffered();
    return w.buffered();
}

fn write(w: *std.Io.Writer) !void {
    var nl: u32 = 0;
    var rmin: u32 = std.math.maxInt(u32);
    var rmax: u32 = 0;
    var dsum: u64 = 0;
    for (layers) |l| {
        if (!l.have) continue;
        nl += 1;
        rmin = @min(rmin, l.rows);
        rmax = @max(rmax, l.rows);
        dsum += l.distinct;
    }
    const mean_d: f64 = if (nl > 0) @as(f64, @floatFromInt(dsum)) / @as(f64, @floatFromInt(nl)) else 0;
    try w.print("[cold-start] layers={d} prompt_rows={d}..{d} prompt_experts={d:.1}", .{ nl, if (nl > 0) rmin else 0, rmax, mean_d });
    for (counts, 0..) |ct, c| {
        try w.print(" c{d}_calls={d} c{d}_routed={d} c{d}_missed={d} c{d}_call={d} c{d}_tail={d}", .{ c, ct.calls, c, ct.routed, c, ct.missed, c, ct.in_call, c, ct.in_tail });
        for (tops, 0..) |t, i| try w.print(" c{d}_top{d}={d}", .{ c, t, ct.top[i] });
        try w.print(" c{d}_nocall={d}", .{ c, ct.nocall });
    }
    try w.writeAll(" r0_by_layer=");
    var first = true;
    for (r0, 0..) |x, li| {
        if (!r0_seen[li]) continue;
        try w.print("{s}{d}:{d}/{d}/{d}", .{ if (first) "" else ",", li, x[0], x[1], x[2] });
        first = false;
    }
    try w.writeAll("\n");
}

const testing = std.testing;

test "cold start: the first decode calls' misses against the last prompt call, by tail and by vote rank" {
    const saved = on;
    defer {
        on = saved;
        clear();
    }
    clear();
    on = true;
    // layer 2, an earlier prompt call (replaced), then the last: 10 rows x 2 ids; expert 7 in every row, 9 in the
    // first 6 rows, 1 in rows 6 and 7, 3 in rows 8 and 9 (the last 8 rows: 7, 9, 1, 3 and the pairs' others)
    prompt(2, &[_]u16{ 100, 101 }, 1, 2);
    var ids: [20]u16 = undefined;
    for (0..10) |r| {
        ids[2 * r] = 7;
        ids[2 * r + 1] = if (r < 6) 9 else if (r < 8) 1 else 3;
    }
    prompt(2, &ids, 10, 2);
    // ranks: 7 (10 votes) 0, 9 (6) 1, 1 (2) 2, 3 (2) 3: ties to the lower id
    // c0: routes 7, 9, 3, 50, 100 (100: the replaced call's); misses 9, 3, 50, 100
    decode(2, &[_]u16{ 7, 9, 3, 50, 100, 7 }, &[_]u8{ 0, 1, 1, 2, 1, 0 });
    // c1: misses 1 only
    decode(2, &[_]u16{ 1, 7 }, &[_]u8{ 1, 0 });
    // a layer with no prompt call: its misses are `nocall`
    decode(5, &[_]u16{ 4, 5 }, &[_]u8{ 1, 1 });
    try testing.expectEqual(@as(u16, 1), layers[2].rank[9]);
    try testing.expectEqual(@as(u16, 2), layers[2].rank[1]);
    try testing.expectEqual(@as(u16, 3), layers[2].rank[3]);
    var buf: [2048]u8 = undefined;
    const out = meters(&buf);
    try testing.expect(std.mem.startsWith(u8, out, "[cold-start] layers=1 prompt_rows=10..10 prompt_experts=4.0 "));
    try testing.expect(std.mem.indexOf(u8, out, " c0_calls=2 c0_routed=7 c0_missed=6 c0_call=2 c0_tail=2 c0_top6=2 c0_top12=2 c0_top24=2 c0_top48=2 c0_nocall=2") != null);
    try testing.expect(std.mem.indexOf(u8, out, " c1_calls=1 c1_routed=2 c1_missed=1 c1_call=1 c1_tail=1 c1_top6=1 c1_top12=1 c1_top24=1 c1_top48=1 c1_nocall=0") != null);
    try testing.expect(std.mem.indexOf(u8, out, " c2_calls=0 c2_routed=0 c2_missed=0") != null);
    try testing.expect(std.mem.endsWith(u8, out, " r0_by_layer=2:4/2/2,5:2/0/0\n"));
    // cleared: nothing until the next request's decode
    try testing.expectEqualStrings("", meters(&buf));
}

test "cold start: a layer's calls past the third are not counted; a new prompt call restarts the count" {
    const saved = on;
    defer {
        on = saved;
        clear();
    }
    clear();
    on = true;
    prompt(0, &[_]u16{ 1, 2 }, 1, 2);
    for (0..5) |_| decode(0, &[_]u16{3}, &[_]u8{1});
    try testing.expectEqual(@as(u64, 1), counts[2].calls);
    prompt(0, &[_]u16{ 1, 2 }, 1, 2);
    decode(0, &[_]u16{3}, &[_]u8{1});
    try testing.expectEqual(@as(u64, 2), counts[0].calls);
}

test "cold start: off writes nothing" {
    const saved = on;
    defer on = saved;
    on = false;
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("", meters(&buf));
}
