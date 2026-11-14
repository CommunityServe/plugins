//! A prompt call's pairs by slot bank, MLX-free (src/ling_bank.zig `promptCall`): the call's routes grouped by expert
//! (experts in id order, each expert's pairs in route order), and one group's pairs split into a call per slot bank
//! (base, ext, transient), in that order. Which bank an expert sits in follows the read pool's timing, so the plan must
//! keep every pair's math the same whatever bank its expert lands in: each expert's pairs stay together in one call, in
//! route order, and the combine reads every pair at its route position (`loc`), never by bank.

const std = @import("std");

/// The call's pairs grouped by expert, a counting sort: expert e's pairs are `order[first[e]..first[e + 1]]` in route
/// order; `experts` the call's experts in id order. `first` has `n_experts + 1` entries, `order` and `fill` as named.
pub fn byExpert(ids: []const u16, n_experts: usize, first: []u32, order: []u32, fill: []u32, experts: *std.ArrayList(u16), a: std.mem.Allocator) !void {
    @memset(first, 0);
    for (ids) |e| first[e + 1] += 1;
    for (1..n_experts + 1) |e| first[e] += first[e - 1];
    @memcpy(fill, first[0..n_experts]);
    for (ids, 0..) |e, i| {
        order[fill[e]] = @intCast(i);
        fill[e] += 1;
    }
    experts.clearRetainingCapacity();
    for (0..n_experts) |e| if (first[e + 1] > first[e]) try experts.append(a, @intCast(e));
}

/// One slot bank's pairs of a group (`group`: experts in id order; `bank_of[j]` the bank of `group[j]`): each pair's
/// route position into `pair_at`, its expert's every pair in route order, experts in id order. Returns the count.
pub fn bankPairs(group: []const u16, bank_of: []const u8, bank: u8, first: []const u32, order: []const u32, pair_at: []u32) usize {
    var m: usize = 0;
    for (group, bank_of) |e, b| {
        if (b != bank) continue;
        for (order[first[e]..first[e + 1]]) |i| {
            pair_at[m] = i;
            m += 1;
        }
    }
    return m;
}

// ── tests ──

const testing = std.testing;

/// What one pair's math sees in a plan: the bank call's other pairs of its expert, in order (the pair's own place
/// among them included). A residency-invariant plan gives every pair the same view whatever bank its expert sits in.
fn viewsOf(a: std.mem.Allocator, ids: []const u16, n_experts: usize, group_size: usize, bank_of_expert: []const u8, plan: anytype) ![][]u32 {
    const first = try a.alloc(u32, n_experts + 1);
    defer a.free(first);
    const order = try a.alloc(u32, ids.len);
    defer a.free(order);
    const fill = try a.alloc(u32, n_experts);
    defer a.free(fill);
    var experts: std.ArrayList(u16) = .empty;
    defer experts.deinit(a);
    try byExpert(ids, n_experts, first, order, fill, &experts, a);
    const views = try a.alloc([]u32, ids.len);
    const pair_at = try a.alloc(u32, ids.len);
    defer a.free(pair_at);
    const bank_of = try a.alloc(u8, group_size);
    defer a.free(bank_of);
    var at: usize = 0;
    while (at < experts.items.len) {
        const group = experts.items[at..@min(at + group_size, experts.items.len)];
        at += group.len;
        for (group, bank_of[0..group.len]) |e, *b| b.* = bank_of_expert[e];
        for (0..3) |bank| {
            const m = plan(group, bank_of[0..group.len], @intCast(bank), first, order, pair_at);
            for (pair_at[0..m]) |p| {
                // the pair's expert's pairs in this call, in call order
                var same: std.ArrayList(u32) = .empty;
                for (pair_at[0..m]) |q| if (ids[q] == ids[p]) try same.append(a, q);
                views[p] = try same.toOwnedSlice(a);
            }
        }
    }
    return views;
}

fn freeViews(a: std.mem.Allocator, v: [][]u32) void {
    for (v) |x| a.free(x);
    a.free(v);
}

/// Routes of `rows` rows at top-`k` over `n_experts` experts, drawn from `seed`, distinct within a row.
fn drawRoutes(a: std.mem.Allocator, rows: usize, k: usize, n_experts: usize, seed: u64) ![]u16 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const ids = try a.alloc(u16, rows * k);
    for (0..rows) |row| {
        var j: usize = 0;
        while (j < k) {
            const e: u16 = @intCast(r.uintLessThan(usize, n_experts));
            if (std.mem.indexOfScalar(u16, ids[row * k ..][0..j], e) != null) continue;
            ids[row * k + j] = e;
            j += 1;
        }
    }
    return ids;
}

fn expectSameViews(a: std.mem.Allocator, ids: []const u16, plan: anytype) !void {
    const n_experts = 64;
    const all_base = try a.alloc(u8, n_experts);
    defer a.free(all_base);
    @memset(all_base, 0);
    const base = try viewsOf(a, ids, n_experts, 12, all_base, plan);
    defer freeViews(a, base);
    var prng = std.Random.DefaultPrng.init(7);
    for (0..8) |_| {
        // a different resident / streamed split of the same routes
        const split = try a.alloc(u8, n_experts);
        defer a.free(split);
        for (split) |*b| b.* = @intCast(prng.random().uintLessThan(u8, 3));
        const got = try viewsOf(a, ids, n_experts, 12, split, plan);
        defer freeViews(a, got);
        for (base, got) |x, y| try testing.expectEqualSlices(u32, x, y);
    }
}

test "prompt plan: every pair's call view is the same whatever slot bank its expert sits in" {
    const a = testing.allocator;
    const ids = try drawRoutes(a, 40, 4, 64, 11);
    defer a.free(ids);
    try expectSameViews(a, ids, bankPairs);
}

test "prompt plan: a plan that splits an expert's pairs across bank calls, or reorders them, fails the residency check" {
    const a = testing.allocator;
    const ids = try drawRoutes(a, 40, 4, 64, 11);
    defer a.free(ids);
    // Plant 1: an expert's pairs split by route parity between the bank's call and the base call.
    const Split = struct {
        fn plan(group: []const u16, bank_of: []const u8, bank: u8, first: []const u32, order: []const u32, pair_at: []u32) usize {
            var m: usize = 0;
            for (group, bank_of) |e, b| for (order[first[e]..first[e + 1]]) |i| {
                const to: u8 = if (i % 2 == 0) b else 0;
                if (to != bank) continue;
                pair_at[m] = i;
                m += 1;
            };
            return m;
        }
    };
    try testing.expectError(error.TestExpectedEqual, expectSameViews(a, ids, Split.plan));
    // Plant 2: a bank's pairs in reverse route order when the bank is not the base.
    const Reverse = struct {
        fn plan(group: []const u16, bank_of: []const u8, bank: u8, first: []const u32, order: []const u32, pair_at: []u32) usize {
            const m = bankPairs(group, bank_of, bank, first, order, pair_at);
            if (bank != 0) std.mem.reverse(u32, pair_at[0..m]);
            return m;
        }
    };
    try testing.expectError(error.TestExpectedEqual, expectSameViews(a, ids, Reverse.plan));
}
