//! turboexl3_ling's read-ahead predictor, MLX-free (`zig build test`): Ling's score form over a host graph, and the
//! copied streamer's group-limited pick (`Selector.limitGroups`) against Ling's router rule written out directly.
const std = @import("std");
const testing = std.testing;
const predictor = @import("ling_predictor.zig");
const lookahead = @import("mlx_serve/sdk/weights/streamed/sdk_ext/expert/lookahead.zig");

/// A host graph over row-major f32 tensors with the ops Ling's score form uses; `add` broadcasts a [1, cols] or [1, 1]
/// right operand over the rows, as MLX does.
const HostOps = struct {
    pub const T = struct { data: []f32, rows: usize, cols: usize };
    pub const Dtype = enum { float32, bfloat16 };
    arena: std.mem.Allocator,

    fn new(g: *HostOps, rows: usize, cols: usize) !T {
        return .{ .data = try g.arena.alloc(f32, rows * cols), .rows = rows, .cols = cols };
    }
    fn at(b: T, i: usize, j: usize) f32 {
        return b.data[(if (b.rows == 1) 0 else i) * b.cols + (if (b.cols == 1) 0 else j)];
    }
    pub fn transpose(g: *HostOps, x: T) !T {
        const o = try g.new(x.cols, x.rows);
        for (0..x.rows) |i| for (0..x.cols) |j| {
            o.data[j * x.rows + i] = x.data[i * x.cols + j];
        };
        return o;
    }
    pub fn matmul(g: *HostOps, a: T, b: T) !T {
        const o = try g.new(a.rows, b.cols);
        for (0..a.rows) |i| for (0..b.cols) |j| {
            var s: f32 = 0;
            for (0..a.cols) |k| s += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            o.data[i * b.cols + j] = s;
        };
        return o;
    }
    pub fn astype(_: *HostOps, x: T, _: Dtype) !T {
        return x;
    }
    pub fn sigmoid(g: *HostOps, x: T) !T {
        const o = try g.new(x.rows, x.cols);
        for (o.data, x.data) |*d, v| d.* = sigmoidRef(v);
        return o;
    }
    pub fn add(g: *HostOps, a: T, b: T) !T {
        const o = try g.new(a.rows, a.cols);
        for (0..a.rows) |i| for (0..a.cols) |j| {
            o.data[i * a.cols + j] = a.data[i * a.cols + j] + at(b, i, j);
        };
        return o;
    }
};

fn sigmoidRef(v: f32) f32 {
    return 1 / (1 + @exp(-v));
}

// Bar: Ling's scores are sigmoid of the gate logits plus the bias, each against the logit written out directly.
test "predictor scores: Ling's form is sigmoid(z) + bias" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var g: HostOps = .{ .arena = arena.allocator() };
    var xd = [_]f32{ 0.5, -1.25, 2.0, -0.75, 0.125, 1.5 };
    var wd = [_]f32{ 0.3, -0.2, 0.1, -0.4, 0.6, 0.05, 1.0, 0.0, -1.0, 0.25, 0.25, 0.25 };
    var bd = [_]f32{ 0.01, -0.02, 0.0, 0.03 };
    const xf: HostOps.T = .{ .data = &xd, .rows = 2, .cols = 3 };
    const w: HostOps.T = .{ .data = &wd, .rows = 4, .cols = 3 };
    const bias: HostOps.T = .{ .data = &bd, .rows = 1, .cols = 4 };
    const ling = try predictor.lingScores(HostOps, &g, xf, w, bias);
    for (0..2) |r| for (0..4) |e| {
        var z: f32 = 0;
        for (0..3) |k| z += xd[r * 3 + k] * wd[e * 3 + k];
        try testing.expectEqual(sigmoidRef(z) + bd[e], ling.data[r * 4 + e]);
    };
}

/// Ling's router rule written out directly (router_group_limit.metal, then the top-k): group scores are each group's two
/// best keys summed; the `topk_group` best groups are kept, the lower index on a tie; then the `k` best keys of the kept
/// groups, the lower expert id on a tie.
fn routerPicks(row: []const f32, n_group: usize, topk_group: usize, out: []u16) []u16 {
    const gsz = row.len / n_group;
    var gs: [32]f32 = undefined;
    for (0..n_group) |g| {
        var vals: [512]f32 = undefined;
        @memcpy(vals[0..gsz], row[g * gsz ..][0..gsz]);
        std.mem.sort(f32, vals[0..gsz], {}, std.sort.desc(f32));
        gs[g] = vals[0] + vals[1];
    }
    var kept: [32]bool = @splat(false);
    for (0..topk_group) |_| {
        var best: usize = 0;
        while (kept[best]) best += 1;
        for (0..n_group) |g| if (!kept[g] and gs[g] > gs[best]) {
            best = g;
        };
        kept[best] = true;
    }
    const Cand = struct { v: f32, e: u16 };
    var cands: [512]Cand = undefined;
    var n: usize = 0;
    for (row, 0..) |v, e| if (kept[e / gsz]) {
        cands[n] = .{ .v = v, .e = @intCast(e) };
        n += 1;
    };
    std.mem.sort(Cand, cands[0..n], {}, struct {
        fn less(_: void, a: Cand, b: Cand) bool {
            return a.v > b.v or (a.v == b.v and a.e < b.e);
        }
    }.less);
    for (out, cands[0..out.len]) |*o, c| o.* = c.e;
    return out;
}

fn plainPicks(row: []const f32, out: []u16) []u16 {
    return routerPicks(row, 1, 1, out);
}

fn neverResident(_: void, _: u16) bool {
    return false;
}

fn fillRow(r: std.Random, row: []f32, levels: u32) void {
    for (row) |*v| v.* = @as(f32, @floatFromInt(r.uintLessThan(u32, levels))) / @as(f32, @floatFromInt(levels)) - 0.5;
}

// Bar: on one row the selector's picks, in order, are the router's own (8 groups of 64, 4 kept, top-8), ties included.
test "read-ahead: the group-limited pick is Ling's router rule, ties included" {
    var prng = std.Random.DefaultPrng.init(0x1196_c1);
    const r = prng.random();
    var sel = try lookahead.SelectorOf(8).init(testing.allocator, 512, 8, std.math.inf(f32), 8);
    defer sel.deinit(testing.allocator);
    try sel.limitGroups(8, 4);
    var row: [512]f32 = undefined;
    for (0..3000) |trial| {
        // Coarse grids make tied groups and tied experts common; a fine grid has few.
        fillRow(r, &row, switch (trial % 3) {
            0 => 4,
            1 => 16,
            else => 100_000,
        });
        var got: [8]u16 = undefined;
        var want: [8]u16 = undefined;
        try testing.expectEqualSlices(u16, routerPicks(&row, 8, 4, &want), sel.selectWith(&row, {}, neverResident, &got, null));
    }
}

// Bar: without a group limit (a geometry of one group) the selector keeps the stock plain top-k by score, the lower id
// on a tie: the group limit's branch leaves the stock pick as it was.
test "read-ahead: without a group limit the pick stays the plain top-k" {
    var prng = std.Random.DefaultPrng.init(0x1196_c2);
    const r = prng.random();
    var sel = try lookahead.SelectorOf(8).init(testing.allocator, 512, 8, std.math.inf(f32), 8);
    defer sel.deinit(testing.allocator);
    var row: [512]f32 = undefined;
    for (0..1000) |trial| {
        fillRow(r, &row, if (trial % 2 == 0) 8 else 100_000);
        var got: [8]u16 = undefined;
        var want: [8]u16 = undefined;
        try testing.expectEqualSlices(u16, plainPicks(&row, &want), sel.selectWith(&row, {}, neverResident, &got, null));
    }
}

// Bar: across verify rows every pick is in some row's group-limited top-k, and residents are skipped.
test "read-ahead: a multi-row group-limited pick takes each row's own groups and skips residents" {
    var prng = std.Random.DefaultPrng.init(0x1196_c3);
    const r = prng.random();
    var sel = try lookahead.SelectorOf(8).init(testing.allocator, 512, 8, std.math.inf(f32), 8);
    defer sel.deinit(testing.allocator);
    try sel.limitGroups(8, 4);
    const Res = struct {
        fn odd(_: void, e: u16) bool {
            return e % 2 == 1;
        }
    };
    var rows: [3 * 512]f32 = undefined;
    for (0..500) |_| {
        fillRow(r, &rows, 16);
        var tops: [3][8]u16 = undefined;
        for (0..3) |i| _ = routerPicks(rows[i * 512 ..][0..512], 8, 4, &tops[i]);
        var got: [8]u16 = undefined;
        for (sel.selectWith(&rows, {}, Res.odd, &got, null)) |e| {
            try testing.expect(e % 2 == 0);
            var found = false;
            for (tops) |t| found = found or std.mem.indexOfScalar(u16, &t, e) != null;
            try testing.expect(found);
        }
    }
}

// Bar: a limit the router could not run is refused.
test "read-ahead: limitGroups refuses uneven groups, too many groups and too few kept experts" {
    var sel = try lookahead.SelectorOf(8).init(testing.allocator, 512, 8, std.math.inf(f32), 8);
    defer sel.deinit(testing.allocator);
    try testing.expectError(error.InvalidSelector, sel.limitGroups(7, 4));
    try testing.expectError(error.InvalidSelector, sel.limitGroups(64, 8));
    try testing.expectError(error.InvalidSelector, sel.limitGroups(8, 0));
    try testing.expectError(error.InvalidSelector, sel.limitGroups(8, 9));
    try testing.expectError(error.InvalidSelector, sel.limitGroups(1, 1));
    try sel.limitGroups(32, 1);
    try testing.expectEqual(@as(u32, 32), sel.n_group);
}
