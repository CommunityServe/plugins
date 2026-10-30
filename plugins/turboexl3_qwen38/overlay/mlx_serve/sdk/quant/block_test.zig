//! copied_from: src/sdk/quant/block_test.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! `block.zig`'s tests, out of its import graph: a plugin unit that reaches `block.zig` never compiles them.
const block_mod = @import("block.zig");
const std = @import("std");
const blocksOf = block_mod.blocksOf;
const testing = std.testing;
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
