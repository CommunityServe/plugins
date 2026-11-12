//! The residency policy's tests (`sdk_ext.expert.policy`), kept in the package's test root under their names.

const std = @import("std");
const env = @import("mlx_host").env;
const policy = @import("../sdk_ext.zig").expert.policy;
const Phase = policy.Phase;
const max_route_ids = policy.max_route_ids;
const no_expert = policy.no_expert;
const no_slot = policy.no_slot;
const Load = policy.Load;
const Eviction = policy.Eviction;
const Plan = policy.Plan;
const LayerPolicy = policy.LayerPolicy;
const rankHottest = policy.rankHottest;
const boundedParts = policy.boundedParts;

// ── Tests ──

const testing = std.testing;
const expert_bank = @import("bank.zig");

/// Every plan leaves a bijective residency map and serves each id exactly as
/// its hit or load says.
fn checkPlan(p: *const LayerPolicy, ids: []const u16, out: *const Plan, transient: u32) !void {
    var occupied: u32 = 0;
    for (p.slot_to_expert[0..p.n_experts], 0..) |e, s| {
        if (e == no_expert) continue;
        try testing.expect(s < p.capacity);
        try testing.expectEqual(@as(u32, @intCast(s)), p.expert_to_slot[e]);
        occupied += 1;
    }
    try testing.expectEqual(p.occupancy, occupied);
    try testing.expect(p.occupancy <= p.capacity);
    for (out.hitsOf()) |h| try testing.expect(p.expert_to_slot[h] != no_slot);
    var transient_seen: [max_route_ids]bool = @splat(false);
    for (out.loadsOf()) |l| {
        if (l.persistent) {
            try testing.expectEqual(l.slot, p.expert_to_slot[l.expert]);
        } else {
            try testing.expect(l.slot >= p.capacity and l.slot < p.capacity + transient);
            try testing.expect(!transient_seen[l.slot - p.capacity]);
            transient_seen[l.slot - p.capacity] = true;
            try testing.expectEqual(no_slot, p.expert_to_slot[l.expert]);
        }
    }
    for (out.evictionsOf()) |ev| {
        try testing.expectEqual(no_slot, p.expert_to_slot[ev.previous]);
        for (out.hitsOf()) |h| try testing.expect(h != ev.previous);
    }
    try testing.expectEqual(out.n_hits + out.n_misses, @as(u32, @intCast(countUnique(ids))));
    try testing.expectEqual(out.n_misses, out.n_loads);
    for (ids, out.slotsOf()) |e, s| {
        if (p.expert_to_slot[e] != no_slot) {
            try testing.expectEqual(p.expert_to_slot[e], s);
        } else {
            var found = false;
            for (out.loadsOf()) |l| if (l.expert == e) {
                try testing.expectEqual(l.slot, s);
                found = true;
            };
            try testing.expect(found);
        }
    }
}

fn countUnique(ids: []const u16) usize {
    var n: usize = 0;
    for (ids, 0..) |e, i| {
        if (std.mem.indexOfScalar(u16, ids[0..i], e) == null) n += 1;
    }
    return n;
}

/// One plan as the reference runtime's dump_phase1_route_fixture.py records it from the
/// Python bank. loads: [expert, slot, persistent, physical skip];
/// evictions: [slot, previous, next]; parts: decode miss parts by expert.
pub const FixPlan = struct {
    ids: []const u16,
    slots: []const u32,
    hits: []const u16,
    misses: []const u16,
    loads: []const [4]u32,
    evictions: []const [3]u32,
    parts: []const []const u16,
};

/// A one-layer replay for the real-bank test: small rows, per-row routes.
pub const BankTrace = struct {
    layer: u32,
    seed: []const u16,
    prefill_rows: u32,
    decode_rows: u32,
    transient: u32,
    prefill: []const FixPlan,
    routes: []const FixPlan,
};

pub fn expectPlan(p: *const Plan, want: FixPlan) !void {
    try testing.expectEqualSlices(u32, want.slots, p.slotsOf());
    try testing.expectEqualSlices(u16, want.hits, p.hitsOf());
    try testing.expectEqualSlices(u16, want.misses, p.missesOf());
    try testing.expectEqual(want.loads.len, p.n_loads);
    for (want.loads, p.loadsOf()) |w, l| {
        try testing.expectEqual(w[0], l.expert);
        try testing.expectEqual(w[1], l.slot);
        try testing.expectEqual(w[2] != 0, l.persistent);
    }
    try testing.expectEqual(want.evictions.len, p.n_evictions);
    for (want.evictions, p.evictionsOf()) |w, ev| {
        try testing.expectEqual(w[0], ev.slot);
        try testing.expectEqual(w[1], ev.previous);
        try testing.expectEqual(w[2], ev.next);
    }
}

/// The decode miss parts of `p` (by expert), in the real bank's placement.
fn partsOf(p: *const Plan, table: []const expert_bank.Layer, layer: u32, out_experts: *[max_route_ids]u16, ends: *[max_route_ids]u32) []u32 {
    const n = p.n_loads;
    for (p.loadsOf(), 0..) |l, i| out_experts[i] = l.expert;
    std.sort.insertion(u16, out_experts[0..n], {}, std.sort.asc(u16));
    var offsets: [max_route_ids]u64 = undefined;
    var lengths: [max_route_ids]u64 = undefined;
    const t = table[layer];
    for (out_experts[0..n], 0..) |e, i| {
        offsets[i] = t.base_offset + @as(u64, e) * t.record_bytes;
        lengths[i] = t.logical_bytes;
    }
    return boundedParts(offsets[0..n], lengths[0..n], 3, ends);
}

/// ExpertSlotPool._prepare_load's reuse rule over physical rows: a load whose
/// row already holds (layer, expert) is not read.
const Physical = struct {
    persistent: []u16,
    experts: u32,
    transient: [max_route_ids]?[2]u32 = @splat(null),

    fn skip(ph: *Physical, layer: u32, capacity: u32, w: [4]u32) bool {
        if (w[2] != 0) {
            const o = &ph.persistent[layer * ph.experts + w[1]];
            const held = o.* == w[0];
            o.* = @intCast(w[0]);
            return held;
        }
        const o = &ph.transient[w[1] - capacity];
        const held = if (o.*) |h| h[0] == layer and h[1] == w[0] else false;
        o.* = .{ layer, w[0] };
        return held;
    }
};

// ── Edges and accounting identities (synthetic traces; capacity 0, 1 and full; both phases; both planners) ──


/// Totals over a trace: every plan's lookups split into hits and misses, every miss loaded once, and the residency the
/// persistent loads and evictions leave.
const Tally = struct {
    unique: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    loads: u64 = 0,
    persistent: u64 = 0,
    evictions: u64 = 0,

    fn add(t: *Tally, ids: []const u16, out: *const Plan) void {
        t.unique += countUnique(ids);
        t.hits += out.n_hits;
        t.misses += out.n_misses;
        t.loads += out.n_loads;
        t.persistent += out.n_persistent;
        t.evictions += out.n_evictions;
    }

    fn expectIdentities(t: *const Tally, occupancy: u32) !void {
        try testing.expectEqual(t.unique, t.hits + t.misses);
        try testing.expectEqual(t.misses, t.loads);
        try testing.expectEqual(@as(u64, occupancy), t.persistent - t.evictions);
    }
};

/// A random prefill trace whose every plan holds two slots of another live route (`PlanOpts.held`): none is ever a victim.
fn heldTrace(phase: Phase) !void {
    var p = try LayerPolicy.init(testing.allocator, 40, 6);
    defer p.deinit(testing.allocator);
    var out: Plan = .{};
    var t: Tally = .{};
    var rng = std.Random.DefaultPrng.init(if (phase == .prefill) 17 else 18);
    const r = rng.random();
    var ids: [max_route_ids]u16 = undefined;
    for (0..300) |step| {
        const n = r.intRangeAtMost(usize, 1, 10);
        for (ids[0..n]) |*e| e.* = @intCast(r.intRangeLessThan(u32, 0, 40));
        if (step % 40 == 0) p.prepareSeed(ids[0..n]);
        const held = [_]u32{ @intCast(step % 6), @intCast((step + 3) % 6) };
        p.planWith(ids[0..n], phase, &out, .{ .held = &held, .transient_base = 4 });
        try checkPlan(&p, ids[0..n], &out, max_route_ids + 4);
        for (out.evictionsOf()) |ev| for (held) |h| testing.expect(ev.slot != h) catch |e| {
            std.debug.print("{t} step {d}: slot {d} held, evicted {d} -> {d}\n", .{ phase, step, h, ev.previous, ev.next });
            return e;
        };
        for (out.loadsOf()) |l| if (!l.persistent) try testing.expect(l.slot >= p.capacity + 4);
        t.add(ids[0..n], &out);
    }
    try t.expectIdentities(p.occupancy);
}


fn policyInitDeinit(a: std.mem.Allocator, n_experts: u32, capacity: u32) !void {
    var p = try LayerPolicy.init(a, n_experts, capacity);
    p.deinit(a);
}

// ── The rate tally (`LayerPolicy.setTally`, `sdk_ext.expert.tally`) ──

const tally = @import("../sdk_ext.zig").expert.tally;

/// The resident experts in ascending order.
fn residentsOf(p: *const LayerPolicy, buf: []u16) []u16 {
    var n: usize = 0;
    for (p.slot_to_expert[0..p.capacity]) |e| if (e != no_expert) {
        buf[n] = e;
        n += 1;
    };
    std.sort.pdq(u16, buf[0..n], {}, std.sort.asc(u16));
    return buf[0..n];
}

fn expectPins(p: *const LayerPolicy, want: []const u16) !void {
    var n: usize = 0;
    for (0..p.n_experts) |e| if (p.isPinned(@intCast(e))) {
        try testing.expect(n < want.len and want[n] == e);
        n += 1;
    };
    try testing.expectEqual(want.len, n);
}

/// Route t of a cycle over experts 3 .. 15, three a route.
fn cycleRoute(t: usize) [3]u16 {
    return .{ @intCast(3 + (3 * t) % 13), @intCast(3 + (3 * t + 1) % 13), @intCast(3 + (3 * t + 2) % 13) };
}
