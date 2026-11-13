//! sdk_ext.expert.lookahead's tests (the Python lane's oracle traces), in the unit tests; the names it reads are the SDK's.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const lookahead = @import("../sdk_ext.zig").expert.lookahead;
const expert_policy = @import("../sdk_ext.zig").expert.policy;
const LayerPolicy = expert_policy.LayerPolicy;
pub const routed_top_k = @import("bank.zig").routed_top_k;
pub const max_k = lookahead.max_k;
pub const max_budget = lookahead.max_budget;
pub const max_rows = lookahead.max_rows;
pub const max_candidates = lookahead.max_candidates;
pub const Selector = lookahead.SelectorOf(routed_top_k);

const testing = std.testing;
