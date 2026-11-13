//! G3, speculative decode as an opt-in `arch` capability: `none`, or the arch's own `draft_lane` over its
//! module-owned state.

const std = @import("std");
const check = @import("check.zig");

pub const Spec = union(enum) {
    none,
    draft_lane: DraftLane,
};

/// The request's sampling as the host resolved it (request body > launch flags > generation_config > defaults). A lane
/// that serves sampled requests applies it to the TARGET's logits before acceptance; the host's serial path keeps its
/// own sampler. Penalties, grammar and logprobs never reach a lane (`ArmRequest.clean` is false for those).
pub const SamplingParams = struct {
    /// Divides the target logits before acceptance (logits / temperature). < 0.01 = greedy (argmax).
    temperature: f32 = 1.0,
    /// Nucleus filter on the tempered target distribution: keep the smallest set of tokens whose probability sums to
    /// >= top_p; 1.0 = off.
    top_p: f32 = 1.0,
    /// Keep the k most likely tokens; 0 = off, 1 = greedy.
    top_k: u32 = 0,
    /// Drop tokens below min_p x the top token's probability; null = off.
    min_p: ?f32 = null,
    /// The request's seed, or the host's per-request seed when the request sent none. A lane derives each draw's
    /// randomness from (seed, the drawn token's absolute position), so draws never repeat across rounds and a seeded
    /// request is reproducible.
    seed: u64 = 0,

};

/// What the host knows of a request when it arms a lane: the lane decides which requests it serves.
pub const ArmRequest = struct {
    /// `sampling.greedy()`, kept for lanes that serve only greedy requests
    greedy: bool,
    /// no logprobs, grammar or penalties
    clean: bool,
    /// the request's sampling; the host passes the same value to every `round` of the request
    sampling: SamplingParams = .{},
};

pub const DraftArm = enum { off, greedy, typical, stochastic };

/// One round's result: [t1, <= accepted_cap accepted drafts] (owned by the round's allocator) and the next token,
/// which is not in the state.
pub const DraftRound = struct {
    tokens: []u32,
    accepted: u32,
    next_token: u32,

    pub fn deinit(self: *DraftRound, a: std.mem.Allocator) void {
        a.free(self.tokens);
    }
};

/// A lane's counters over the request, for `[spec-stats]`, `/props` and the receipts.
pub const DraftStats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    generated: u64 = 0,
};

/// The arch drives its own rounds over its module-owned state; the host dispatches, stops and emits. Every round
/// keeps the state at prompt + emitted with t1 not in it, and never stops: EOS, stop strings and the budget are
/// the host's. One indirect call per round.
pub const DraftLane = struct {
    /// Rows verified per round; 0 = serial (the lane's readiness signal).
    block_size: *const fn (m: *const anyopaque) u32,
    /// The lane as installed, for the server log and the receipts.
    lane_name: *const fn (m: *const anyopaque) []const u8,
    /// The request's arming, before its prompt pass: the arch binds its sampler here (mlx-stream: the plugin draws every
    /// token of a clean request), whatever it answers.
    arm: *const fn (m: *anyopaque, req: ArmRequest) DraftArm,
    /// One round; `sampling` is the request's (the value `arm` received).
    round: *const fn (m: *anyopaque, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: SamplingParams) anyerror!DraftRound,
    stats: *const fn (m: *const anyopaque) DraftStats,

};

const testing = std.testing;
