//! sdk_ext.expert.lookahead's tests (the Python lane's oracle traces), in the unit tests; the names it reads are the SDK's.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const lookahead = @import("../sdk_ext.zig").expert.lookahead;
const expert_policy = @import("../sdk_ext.zig").expert.policy;
const LayerPolicy = expert_policy.LayerPolicy;
pub const routed_top_k = @import("bank.zig").routed_top_k;
pub const min_k = routed_top_k;
pub const max_k = lookahead.max_k;
pub const max_budget = lookahead.max_budget;
pub const max_rows = lookahead.max_rows;
pub const max_candidates = lookahead.max_candidates;
pub const Selector = lookahead.SelectorOf(routed_top_k);

const testing = std.testing;

/// A policy holding exactly `residents` (prefill-admitted, capacity = their count).
fn policyWith(a: std.mem.Allocator, n_experts: u32, residents: []const u16) !LayerPolicy {
    var p = try LayerPolicy.init(a, n_experts, @intCast(residents.len));
    errdefer p.deinit(a);
    if (residents.len > 0) {
        var plan: expert_policy.Plan = .{};
        p.plan(residents, .prefill, &plan);
    }
    return p;
}

fn rowsOf(comptime n_experts: usize, comptime rows: usize, spec: [rows][]const struct { u16, f32 }) [rows * n_experts]f32 {
    var s: [rows * n_experts]f32 = @splat(0);
    for (spec, 0..) |row, r| for (row) |ev| {
        s[r * n_experts + ev[0]] = ev[1];
    };
    return s;
}
