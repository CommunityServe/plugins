//! copied_from: src/sdk/quant/block_test.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! `block.zig`'s tests, out of its import graph: a plugin unit that reaches `block.zig` never compiles them.

const block_mod = @import("block.zig");
const std = @import("std");
const abi = @import("../../abi/quant.zig");
const Verdict = abi.Verdict;
const blocksOf = block_mod.blocksOf;
const builtins = block_mod.builtins;
const claimant = block_mod.claimant;
const declared = block_mod.declared;
const hadamardBlock = block_mod.hadamardBlock;
const registry = @import("../../plugins/registry.zig");
const testing = std.testing;

fn expectClaimant(want: ?[]const u8, block_json: []const u8, config_json: []const u8) !void {
    const got = try claimant(block_json, config_json);
    const name: ?[]const u8 = if (got) |i| builtins[i].name else null;
    if (want) |w| {
        try testing.expect(name != null);
        try testing.expectEqualStrings(w, name.?);
    } else try testing.expectEqual(@as(?[]const u8, null), name);
}

test "quant block: blocks are found under their keys; a config without one has none" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bs = try blocksOf(a, "{\"expert_quant\":{\"format\":\"exl3\"},\"quantization\":{\"bits\":4},\"quantization_config\":3}");
    try testing.expectEqual(@as(usize, 2), bs.len);
    try testing.expectEqualStrings("quantization", bs[0].key);
    try testing.expectEqualStrings("{\"bits\":4}", bs[0].json);
    try testing.expectEqualStrings("expert_quant", bs[1].key);
    const full = try blocksOf(a, "{\"trunk_quant\":{\"format\":\"exl3\"},\"expert_quant\":{\"format\":\"exl3\",\"k\":5}}");
    try testing.expectEqual(@as(usize, 2), full.len);
    try testing.expectEqualStrings("expert_quant", full[0].key);
    try testing.expectEqualStrings("trunk_quant", full[1].key);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "{\"model_type\":\"llama\"}")).len);
    try testing.expectEqual(@as(usize, 0), (try blocksOf(a, "not json")).len);
}
