//! `MLX_SERVE_DSV41_SEED_FREE` (read at load; on by default: unset, empty or `1`; `0` turns it off; anything else refused
//! by name): the rows the
//! load's usage seed fills (`MLX_SERVE_DSV41_RESIDENT_SEED`, src/expert_usage_cfg.zig) stay free until a route uses
//! their expert. The prompt's read-ahead (P1) may take one for an expert its prediction ranks before the seeded one, so a
//! layer's rows hold its call's hottest experts as they would have held them empty; the CED decoder prefetch
//! (`MLX_SERVE_DSV41_DEC_PREFETCH`) may take one whose expert its prediction does not name. A seeded row a route has used
//! is a resident as any other, so the rule changes nothing past the rows no request has touched. The policy is the
//! streamer's (`LayerPolicy.free_seed`, patches/turboexl3-zz5-seed-free.patch); src/seed_free_test.zig tests it.

const std = @import("std");

/// The switch (`configure`, at load, before the stream opens and seeds).
pub var on: bool = true;

pub const Error = error{SeedFreeSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.SeedFreeSetting;
}

test "seed_free setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    for ([_][]const u8{ "on", "2", "yes", "01" }) |v| try std.testing.expectError(error.SeedFreeSetting, parse(v));
}
