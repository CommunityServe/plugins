//! copied_from: src/sdk/quant/block.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! A config's quantization blocks and the built-in formats' claims and validation, MLX-free (discovery and the
//! load preflight run them): which block a pack carries, which format serves it, the defaults it declares.

const std = @import("std");

const Value = std.json.Value;

/// The config keys a quantization block sits under (`trunk_quant`: the trunk's linears' format beside `expert_quant`).
pub const keys = [_][]const u8{ "quantization", "quantization_config", "expert_quant", "trunk_quant" };

pub const Block = struct { key: []const u8, json: []const u8 };

/// The blocks of `config_json`, in `keys` order, each re-serialized into `a`; none when it does not parse.
pub fn blocksOf(a: std.mem.Allocator, config_json: []const u8) ![]Block {
    const root = parseObject(a, config_json) orelse return &.{};
    var out: std.ArrayList(Block) = .empty;
    for (keys) |k| {
        const v = root.get(k) orelse continue;
        if (v != .object) continue;
        try out.append(a, .{ .key = k, .json = try std.json.Stringify.valueAlloc(a, v, .{}) });
    }
    return out.items;
}

// ── the built-in formats' claims and validation ──


/// A scratch arena for one call: the C-ABI members take no allocator.
const Scratch = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    fn a(self: *Scratch) std.mem.Allocator {
        return self.arena.allocator();
    }
};

// ── JSON readers ──

fn parseObject(a: std.mem.Allocator, json: []const u8) ?std.json.ObjectMap {
    const v = std.json.parseFromSliceLeaky(Value, a, json, .{}) catch return null;
    return if (v == .object) v.object else null;
}
