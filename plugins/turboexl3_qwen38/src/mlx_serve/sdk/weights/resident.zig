//! copied_from: src/sdk/weights/resident.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The resident weights: a pack's tensors held whole in one map by name (`Weights`, filled by the engine's loader with
//! the adaptations ModelOpt and Sushi packs need), the source the quant formats read it through (`abi.Source`), and
//! the shard set a safetensors index names.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");

pub const Resident = struct {
    weights: Weights,

    pub fn deinit(self: *Resident) void {
        self.weights.deinit();
    }

    /// The whole map behind `src` when it is a resident source, moved out (the source is left empty): an in-tree
    /// model that keeps the loader's map adopts it instead of reading the pack twice. Null for any other source.
    pub fn take(src: *const abi.Source) ?Weights {
        if (src.size < @sizeOf(abi.Source) or !src.resident) return null;
        const self: *Resident = @ptrCast(@alignCast(src.ctx.?));
        const w = self.weights;
        self.weights = Weights.init(w.allocator);
        return w;
    }

    fn get(ctx: ?*anyopaque, name: abi.Str) callconv(.c) abi.Array {
        const self: *const Resident = @ptrCast(@alignCast(ctx.?));
        const a = self.weights.get(name.slice()) orelse return .{};
        return .{ .ctx = a.ctx };
    }
};

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,
    /// Int8 copies of block linears that replace their packed arrays (`--w8a8`); only the media
    /// DiTs that read through a name-keyed map fill it.
    w8: @import("../blocks/dit/w8a8.zig").Overlay = .{},

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
        self.w8.deinit(self.allocator);
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Hand the map a new array under `name`, freeing the one it held (load-time
    /// weight fusion parks its row views here so the originals go away).
    pub fn replace(self: *Weights, name: []const u8, arr: mlx.mlx_array) void {
        if (self.map.getPtr(name)) |p| {
            _ = mlx.mlx_array_free(p.*);
            p.* = arr;
        }
    }

    /// Hand the map `arr` under `name` (the key is copied), freeing an array it held there.
    pub fn put(self: *Weights, name: []const u8, arr: mlx.mlx_array) !void {
        if (self.map.contains(name)) return self.replace(name, arr);
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.map.put(key, arr);
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The set of shard basenames in `model.safetensors.index.json`'s `weight_map`,
/// or null when there is no usable index (single-file packs, media packs).
pub fn indexShardSet(io: std.Io, dir: std.Io.Dir) ?std.StringHashMapUnmanaged(void) {
    const a = std.heap.page_allocator;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024)) catch return null;
    defer a.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const wm = parsed.value.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string) continue;
        if (set.contains(v.string)) continue;
        const key = a.dupe(u8, v.string) catch continue;
        set.put(a, key, {}) catch {
            a.free(key);
            continue;
        };
    }
    // An index none of whose shards exist is stale (the repo was re-sharded
    // after this index was written); the directory is then the set.
    var any_present = false;
    var keys = set.keyIterator();
    while (keys.next()) |k| {
        _ = dir.statFile(io, k.*, .{}) catch continue;
        any_present = true;
        break;
    }
    if (!any_present) {
        log.warn("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead\n", .{});
        freeShardSet(&set);
        return null;
    }
    return set;
}

/// Free a set from `indexShardSet`.
pub fn freeShardSet(set: *std.StringHashMapUnmanaged(void)) void {
    var keys = set.keyIterator();
    while (keys.next()) |k| std.heap.page_allocator.free(k.*);
    set.deinit(std.heap.page_allocator);
}

// ── tests ──

const testing = std.testing;
