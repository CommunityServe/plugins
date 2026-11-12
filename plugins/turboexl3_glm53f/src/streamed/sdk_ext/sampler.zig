//! mlx-stream's serial sampler: the plugin's token for every row it hands the host. Every arch binds one per module.
//! The host arms a clean request through `draft_lane.arm` before its prompt pass (`arm`). The prompt pass takes the
//! binding (`begin`): a request the host did not arm (logprobs, grammar, penalties, `--no-mtp`) is unbound and keeps
//! the host's sampler.
//!
//! For a bound sampled request every logits row the arch returns becomes the one-hot of the token `acceptance` draws
//! at that position (`chosen`): the prompt's last row (the first token) and each serial step's. That token is
//! `acceptance.sample`: the row's tempered, filtered distribution, drawn at `drawAt(seed, position, token_draw)`. A
//! lane's verify emits the same token at the same position (`sampledRows`, sample-then-verify), so a seeded request
//! yields the same ids serial or drafted, on every arch. The host's own sampler then returns the plugin's token
//! whatever its stream. A greedy request keeps its logits (the host's argmax is the target's).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const acc = @import("acceptance.zig");

/// The sampler over an arch's op backend `G` (its `init(gpa, stream)`, `reset`, `keep` and the acceptance graphs'
/// ops), its own scope reset after each row.
pub fn SamplerOf(comptime G: type) type {
    return struct {
        const Self = @This();
        g: G,
        /// The armed request's sampling, until its prompt pass (`begin`).
        pending: ?acc.Params = null,
        /// The request's: null = unbound or greedy (the logits go to the host as they are).
        active: ?acc.Params = null,

        pub fn init(gpa: std.mem.Allocator, s: mlx.mlx_stream) !Self {
            return .{ .g = try G.init(gpa, s) };
        }

        pub fn deinit(self: *Self) void {
            self.g.deinit();
        }

        /// `draft_lane.arm`: a clean sampled request's sampling, for its prompt pass (greedy: the host's argmax).
        pub fn arm(self: *Self, req: sdk.ArmRequest) void {
            self.pending = if (req.clean and !req.greedy) bindingOf(req.sampling) else null;
        }

        /// The prompt pass's start: the request's binding (none when the host did not arm it). A bound request logs
        /// once, with the decode lane it runs on (`lane`: the arch's lane name, "serial" without one).
        pub fn begin(self: *Self, lane: []const u8) void {
            self.active = self.pending;
            self.pending = null;
            if (self.active) |sm| sdk.log.info("mlx-stream sampler: request bound (the plugin draws every token) lane=\"{s}\" seed={d} temperature={d:.3} top_p={d:.3} top_k={d}\n", .{ lane, sm.seed, sm.temperature, sm.top_p, sm.top_k });
        }

        /// `logits` (owned by the caller; its last axis the vocabulary) predicts the token at absolute position
        /// `pos`. Bound and sampled: a new owned f32 [1, vocab], 0 at the drawn token and -inf elsewhere (`logits`
        /// freed). Otherwise `logits` itself.
        pub fn chosen(self: *Self, logits: mlx.mlx_array, pos: u64) !mlx.mlx_array {
            const sm = self.active orelse return logits;
            const g = &self.g;
            defer g.reset();
            const sh = g.shapeOf(logits);
            const vocab = sh.dim(-1);
            const row = try g.reshape(logits, &.{ 1, vocab });
            const sr = try acc.sampledRows(g, row, sm, pos - 1, &.{});
            const rank = try g.arange(0, @floatFromInt(vocab), 1, .uint32);
            const hot = try g.where(try g.equal(rank, sr.tok), try g.scalar(0, .float32), try g.scalar(-std.math.inf(f64), .float32));
            const out = g.keep(try g.reshape(hot, &.{ 1, vocab }));
            _ = mlx.mlx_array_free(logits);
            return out;
        }
    };
}

/// A request's binding: its sampling when sampled, null when greedy (the host's argmax).
pub fn bindingOf(s: sdk.SamplingParams) ?acc.Params {
    return acc.Params.active(acc.Params.of(s));
}

const testing = std.testing;

/// Seeded rows: each a leader over a spread of logits.
fn fixture(seed: u64, comptime w: usize, comptime v: usize, lg: *[w * v]f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * 1.2;
    for (0..w) |r| lg[r * v + rng.random().uintLessThan(u32, @as(u32, v))] += 2.0;
}

fn argmaxRow(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// The binding's bookkeeping needs no ops (the CPU test above): an empty backend.
const NoOps = struct {};
