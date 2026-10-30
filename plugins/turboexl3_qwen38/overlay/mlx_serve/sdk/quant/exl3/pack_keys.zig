//! copied_from: src/sdk/quant/exl3/pack_keys.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! A fully EXL3 pack's config.json keys beside `expert_quant`, read MLX-free (a plugin's claim parse and the engine's
//! parse share them): `trunk_quant` makes the trunk's linears a quant format's (the block its claimant reads; the
//! trunk loads and runs them through that format), and `ngram_table {file}` names the n-gram table's file in the pack.
const std = @import("std");
pub const Error = error{ TrunkQuantNotAnObject, InvalidNgramTableFile };
/// The longest n-gram table file name a pack may give.
pub const max_file_len = 95;
/// Whether the config root names a format for the trunk's linears (`trunk_quant`, an object its format reads); a
/// missing key or a JSON null is a native trunk.
pub fn hasTrunkQuant(root: std.json.ObjectMap) Error!bool {
    const v = root.get("trunk_quant") orelse return false;
    if (v == .null) return false;
    if (v != .object) return error.TrunkQuantNotAnObject;
    return true;
}
/// The n-gram table's file a config root names (`ngram_table.file`), a path inside the pack; null when it names none.
pub fn ngramTableFile(root: std.json.ObjectMap) Error!?[]const u8 {
    const v = root.get("ngram_table") orelse return null;
    if (v != .object) return null;
    const f = v.object.get("file") orelse return null;
    if (f != .string or f.string.len == 0 or f.string.len > max_file_len or f.string[0] == '/' or
        std.mem.indexOf(u8, f.string, "..") != null) return error.InvalidNgramTableFile;
    return f.string;
}
// ── tests ──
const testing = std.testing;
fn rootOf(arena: std.mem.Allocator, json: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{})).object;
}
test "pack keys: trunk_quant is a format's block or nothing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expect(!try hasTrunkQuant(try rootOf(a, "{}")));
    try testing.expect(!try hasTrunkQuant(try rootOf(a, "{\"trunk_quant\":null}")));
    try testing.expect(try hasTrunkQuant(try rootOf(a, "{\"trunk_quant\":{\"format\":\"exl3\",\"codebook\":\"mul1\",\"window\":16}}")));
    try testing.expectError(error.TrunkQuantNotAnObject, hasTrunkQuant(try rootOf(a, "{\"trunk_quant\":\"exl3\"}")));
}
