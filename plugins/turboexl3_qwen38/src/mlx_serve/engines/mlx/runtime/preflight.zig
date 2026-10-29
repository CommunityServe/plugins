//! This plugin's own, in place of mlx-serve's `src/engines/mlx/runtime/preflight.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the bytes a pack's load reads, which the copies' loader weighs against the machine's
//! RAM to read past the page cache.

const std = @import("std");

/// The safetensors shards the pack's index names (a stray shard beside them is not read), else every shard in the
/// folder; 0 when the folder cannot be read.
pub fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (indexShards(io, dir, a)) |names| {
        var total: u64 = 0;
        for (names) |n| total += fileBytes(io, dir, n);
        return total;
    }
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if ((e.kind == .file or e.kind == .sym_link) and std.mem.endsWith(u8, e.name, ".safetensors")) total += fileBytes(io, dir, e.name);
    }
    return total;
}

fn fileBytes(io: std.Io, dir: std.Io.Dir, name: []const u8) u64 {
    const st = dir.statFile(io, name, .{}) catch return 0;
    return if (st.kind == .file) @intCast(st.size) else 0;
}

/// The distinct shard names of `model.safetensors.index.json`'s `weight_map`, each inside the folder; null without one.
fn indexShards(io: std.Io, dir: std.Io.Dir, a: std.mem.Allocator) ?[]const []const u8 {
    const text = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(256 << 20)) catch return null;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return null;
    if (doc != .object) return null;
    const map = doc.object.get("weight_map") orelse return null;
    if (map != .object) return null;
    var seen: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (map.object.values()) |v| {
        if (v != .string or std.mem.indexOf(u8, v.string, "..") != null or std.fs.path.isAbsolute(v.string)) continue;
        seen.put(a, v.string, {}) catch return null;
    }
    return seen.keys();
}
