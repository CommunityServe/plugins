//! `MLX_SERVE_DSV41_DEC_PREFETCH` (read at load: `S` or `S:K`; unset or `0`: off): a replaying prompt chunk's CED decoder
//! layers' experts read before the decoder runs. At encoder layer S (its input stream, before its attention), each
//! decoder layer's router runs over the replay tail's rows (the model's `predictDecoder`, the prompt timeline's probe);
//! each decoder layer's experts are ranked by the tail rows' votes (each row's top-k), ties by the lower id. From then
//! on, at the end of each routed call of the prompt, the source posts the next K decoder layers' rankings, in layer order,
//! into those layers' own free persistent rows (the stream's `prefetchLayer`: an empty row, else a usage seed row no route
//! has used whose expert the ranking does not name, `MLX_SERVE_DSV41_SEED_FREE`; never a row a route has used, so a warm
//! layer gets few or no fills; the pool's reserved tickets), and each decoder layer's first route lands its fill. K
//! defaults to 2: S = 10 posts all twenty by the encoder's end; S = 19 keeps the posts about two layers ahead of the
//! decoder.
//!
//! The rows are the decoder layers' own (their prompt slot rows, idle while the encoder runs); the only new memory is the
//! pool's reserved tickets (host words, in the pool's bill).

const std = @import("std");

pub const max_layers = 64;
pub const max_experts = 512;

/// The switch (`configure`, at load): the encoder layer whose input predicts, and the decoder layers posted per call.
pub var on: bool = false;
pub var start: usize = 0;
pub var per_call: usize = 2;

pub const Error = error{DecPrefetchSetting};

/// `S` or `S:K` (S >= 1, K 1..32); unset or starting with `0`: off; anything else refused by name.
pub fn configure(value: ?[]const u8) Error!void {
    on = false;
    start = 0;
    per_call = 2;
    reset();
    const v = value orelse return;
    if (v.len == 0 or v[0] == '0') return;
    var it = std.mem.splitScalar(u8, v, ':');
    const s = std.fmt.parseInt(usize, it.next() orelse return error.DecPrefetchSetting, 10) catch return error.DecPrefetchSetting;
    var k: usize = 2;
    if (it.next()) |ks| k = std.fmt.parseInt(usize, ks, 10) catch return error.DecPrefetchSetting;
    if (it.next() != null or s == 0 or s >= max_layers or k == 0 or k > 32) return error.DecPrefetchSetting;
    on = true;
    start = s;
    per_call = k;
}

/// The plan of the replaying chunk running now: per decoder layer its ranking; the next layer to post.
var ranked: [max_layers][max_experts]u16 = undefined;
var n_ranked: [max_layers]u16 = @splat(0);
var planned = false;
var next_layer: usize = 0;
var end_layer: usize = 0;

/// No plan (a new prompt, or its end).
pub fn reset() void {
    planned = false;
    next_layer = 0;
    end_layer = 0;
}

/// Decoder layer `layer`'s tail rows' top-k (`ids`) ranked by votes, the experts the source will post for it.
pub fn rank(layer: usize, ids: []const u32) void {
    if (layer >= max_layers) return;
    var votes: [max_experts]u32 = @splat(0);
    for (ids) |e| if (e < max_experts) {
        votes[e] += 1;
    };
    var n: usize = 0;
    for (votes, 0..) |c, e| if (c > 0) {
        ranked[layer][n] = @intCast(e);
        n += 1;
    };
    std.sort.pdq(u16, ranked[layer][0..n], &votes, struct {
        fn less(cs: *const [max_experts]u32, x: u16, y: u16) bool {
            return if (cs[x] != cs[y]) cs[x] > cs[y] else x < y;
        }
    }.less);
    n_ranked[layer] = @intCast(n);
}

/// Decoder layers [first, end) were ranked (`rank`): the source posts them from the next call's end on.
pub fn plan(first: usize, end: usize) void {
    planned = first < end and end <= max_layers;
    next_layer = first;
    end_layer = end;
}

pub const Post = struct { layer: u32, experts: []const u16 };

/// The next decoder layer to post, or null when the plan is done (or none is armed).
pub fn next() ?Post {
    if (!planned) return null;
    while (next_layer < end_layer) {
        const l = next_layer;
        next_layer += 1;
        if (n_ranked[l] > 0) return .{ .layer = @intCast(l), .experts = ranked[l][0..n_ranked[l]] };
    }
    planned = false;
    return null;
}

test "dec_prefetch setting: S, S:K, off, refused" {
    try configure(null);
    try std.testing.expect(!on);
    try configure("0");
    try std.testing.expect(!on);
    try configure("10");
    try std.testing.expect(on and start == 10 and per_call == 2);
    try configure("19:3");
    try std.testing.expect(on and start == 19 and per_call == 3);
    for ([_][]const u8{ "x", "10:", "10:0", "10:2:1", "64", "-1" }) |v| try std.testing.expectError(error.DecPrefetchSetting, configure(v));
    try configure(null);
}

test "dec_prefetch plan: rankings by votes, posted in layer order, empty layers skipped" {
    reset();
    rank(20, &.{ 7, 7, 3, 9, 9, 9 });
    n_ranked[21] = 0;
    rank(22, &.{5});
    plan(20, 23);
    const a = next().?;
    try std.testing.expectEqual(@as(u32, 20), a.layer);
    try std.testing.expectEqualSlices(u16, &.{ 9, 7, 3 }, a.experts);
    const b = next().?;
    try std.testing.expectEqual(@as(u32, 22), b.layer);
    try std.testing.expect(next() == null);
    try std.testing.expect(next() == null);
}
