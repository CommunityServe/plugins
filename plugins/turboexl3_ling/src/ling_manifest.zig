//! The bank's geometry from `expert-manifest-v2.json`, MLX-free: the routed layers of `experts.bin` (the pack's model
//! layers 4..71 as bank layers 0..67), each a record per expert at `base_offset + expert * record_bytes`, its nine
//! segments back to back in the classic layout (gate code, rout, rin, up code, rout, rin, down code, rout, rin; code
//! I16 [in/16, out/16, 16 K], rout = svh F16 [out], rin = suh F16 [in]), zero-padded to `record_bytes`. The gate/up range
//! is segments 0..5, the down range 6..8, each contiguous: the kit stream's EXL3 topology (9 components, 6 gate/up).

const std = @import("std");

pub const n_components = 9;
pub const gu_components = 6;
pub const Dtype = enum { I16, F16 };
pub const Segment = struct { dtype: Dtype, rank: u8, shape: [3]u64 = @splat(0), offset: u64, length: u64 };
pub const Layer = struct { segments: [n_components]Segment, logical_bytes: u64, record_bytes: u64, base_offset: u64 };
pub const Manifest = struct { layers: []Layer, n_experts: u32, hidden: u32, inter: u32, sidecar: []const u8, sidecar_size: u64 };

/// The classic layout's components, in record order.
pub const components = [n_components][]const u8{
    "gate_proj.code", "gate_proj.rout", "gate_proj.rin",
    "up_proj.code",   "up_proj.rout",   "up_proj.rin",
    "down_proj.code", "down_proj.rout", "down_proj.rin",
};

pub const Error = error{ LingManifestJson, LingManifestFormat, LingManifestLayer, LingManifestSegment, OutOfMemory };

/// `json` (the manifest's text) read into the arena `a`. Every layer's segments are the classic layout's, in order,
/// back to back from offset 0; every record holds its segments (`logical_bytes` = their end, at most `record_bytes`);
/// the layers' records do not overlap and end within the sidecar.
pub fn parse(a: std.mem.Allocator, json: []const u8) Error!Manifest {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.LingManifestJson;
    if (v != .object) return error.LingManifestJson;
    const o = v.object;
    const fmt = o.get("format") orelse return error.LingManifestFormat;
    if (fmt != .string or !std.mem.eql(u8, fmt.string, "mtplx-expert-manifest-v2")) return error.LingManifestFormat;
    const dims = obj(o.get("dims")) orelse return error.LingManifestFormat;
    const n_experts = try int(u32, dims.get("n_experts"));
    const n_layers = try int(u32, dims.get("n_layers"));
    const hidden = try int(u32, dims.get("hidden"));
    const inter = try int(u32, dims.get("inter"));
    const side = obj(o.get("sidecar")) orelse return error.LingManifestFormat;
    const file = side.get("file") orelse return error.LingManifestFormat;
    if (file != .string or file.string.len == 0 or std.mem.indexOfScalar(u8, file.string, '/') != null) return error.LingManifestFormat;
    const size = try int(u64, side.get("size"));
    const ls = o.get("layers") orelse return error.LingManifestFormat;
    if (ls != .array or ls.array.items.len != n_layers or n_layers == 0 or n_experts == 0) return error.LingManifestLayer;
    const layers = try a.alloc(Layer, n_layers);
    for (ls.array.items, layers, 0..) |lv, *l, i| {
        const lo = obj(lv) orelse return error.LingManifestLayer;
        if (try int(u32, lo.get("layer")) != i) return error.LingManifestLayer;
        l.* = .{ .segments = undefined, .logical_bytes = try int(u64, lo.get("logical_bytes")), .record_bytes = try int(u64, lo.get("record_bytes")), .base_offset = try int(u64, lo.get("base_offset")) };
        const segs = lo.get("segments") orelse return error.LingManifestSegment;
        if (segs != .array or segs.array.items.len != n_components) return error.LingManifestSegment;
        var end: u64 = 0;
        for (segs.array.items, &l.segments, components) |sv, *sg, name| {
            const so = obj(sv) orelse return error.LingManifestSegment;
            const comp = so.get("component") orelse return error.LingManifestSegment;
            if (comp != .string or !std.mem.eql(u8, comp.string, name)) return error.LingManifestSegment;
            const dt = so.get("dtype") orelse return error.LingManifestSegment;
            const want: Dtype = if (std.mem.endsWith(u8, name, ".code")) .I16 else .F16;
            if (dt != .string or !std.mem.eql(u8, dt.string, @tagName(want))) return error.LingManifestSegment;
            const sh = so.get("shape") orelse return error.LingManifestSegment;
            if (sh != .array or sh.array.items.len == 0 or sh.array.items.len > 3) return error.LingManifestSegment;
            sg.* = .{ .dtype = want, .rank = @intCast(sh.array.items.len), .offset = try int(u64, so.get("offset")), .length = try int(u64, so.get("length")) };
            var elems: u64 = 1;
            for (sh.array.items, 0..) |d, k| {
                const n = try int(u64, d);
                if (n == 0 or n > std.math.maxInt(c_int)) return error.LingManifestSegment;
                sg.shape[k] = n;
                elems = std.math.mul(u64, elems, n) catch return error.LingManifestSegment;
            }
            if (sg.offset != end or sg.length != std.math.mul(u64, elems, 2) catch return error.LingManifestSegment) return error.LingManifestSegment;
            end += sg.length;
        }
        if (end != l.logical_bytes or l.logical_bytes > l.record_bytes) return error.LingManifestLayer;
        const span = std.math.mul(u64, l.record_bytes, n_experts) catch return error.LingManifestLayer;
        const stop = std.math.add(u64, l.base_offset, span) catch return error.LingManifestLayer;
        if (stop > size) return error.LingManifestLayer;
        if (i > 0) {
            const p = layers[i - 1];
            if (l.base_offset < p.base_offset + p.record_bytes * n_experts) return error.LingManifestLayer;
        }
    }
    return .{ .layers = layers, .n_experts = n_experts, .hidden = hidden, .inter = inter, .sidecar = file.string, .sidecar_size = size };
}

fn obj(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn int(comptime T: type, v: ?std.json.Value) Error!T {
    const x = v orelse return error.LingManifestFormat;
    if (x != .integer or x.integer < 0) return error.LingManifestFormat;
    return std.math.cast(T, x.integer) orelse error.LingManifestFormat;
}

/// The bytes a slot row of every routed layer holds (the records' segments, not their padding).
pub fn rowBytes(m: Manifest) u64 {
    var n: u64 = 0;
    for (m.layers) |l| n += l.logical_bytes;
    return n;
}

/// The largest record's segments (a transient row's bytes: the kit stream sizes its scratch at one geometry).
pub fn widestRecord(m: Manifest) u64 {
    var n: u64 = 0;
    for (m.layers) |l| n = @max(n, l.logical_bytes);
    return n;
}

// ── tests ──

const testing = std.testing;

/// Ling-3.1-flash 2.60bpw's manifest, cut to two layers (each record's nine segments as the pack's, 4,947,456 B of
/// segments in a 4,947,968 B record).
fn twoLayers(comptime base1: []const u8, comptime gate_rout_dtype: []const u8) []const u8 {
    const segs =
        \\[{"component":"gate_proj.code","dtype":"I16","shape":[256,80,40],"offset":0,"length":1638400},
    ++ "{\"component\":\"gate_proj.rout\",\"dtype\":\"" ++ gate_rout_dtype ++ "\",\"shape\":[1280],\"offset\":1638400,\"length\":2560}," ++
        \\{"component":"gate_proj.rin","dtype":"F16","shape":[4096],"offset":1640960,"length":8192},
        \\{"component":"up_proj.code","dtype":"I16","shape":[256,80,40],"offset":1649152,"length":1638400},
        \\{"component":"up_proj.rout","dtype":"F16","shape":[1280],"offset":3287552,"length":2560},
        \\{"component":"up_proj.rin","dtype":"F16","shape":[4096],"offset":3290112,"length":8192},
        \\{"component":"down_proj.code","dtype":"I16","shape":[80,256,40],"offset":3298304,"length":1638400},
        \\{"component":"down_proj.rout","dtype":"F16","shape":[4096],"offset":4936704,"length":8192},
        \\{"component":"down_proj.rin","dtype":"F16","shape":[1280],"offset":4944896,"length":2560}]
    ;
    return "{\"format\":\"mtplx-expert-manifest-v2\",\"dims\":{\"hidden\":4096,\"inter\":1280,\"n_experts\":4,\"n_layers\":2}," ++
        "\"sidecar\":{\"file\":\"experts.bin\",\"alignment\":4096,\"size\":39583744}," ++
        "\"layers\":[{\"layer\":0,\"K\":2.5,\"record_bytes\":4947968,\"logical_bytes\":4947456,\"base_offset\":0,\"segments\":" ++ segs ++ "}," ++
        "{\"layer\":1,\"K\":2.5,\"record_bytes\":4947968,\"logical_bytes\":4947456,\"base_offset\":" ++ base1 ++ ",\"segments\":" ++ segs ++ "}]}";
}

test "ling manifest: the 2.60bpw layout (two layers of four experts): segments in record order, the gate/up range first, records within the sidecar" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const m = try parse(arena.allocator(), twoLayers("19791872", "F16"));
    try testing.expectEqual(@as(usize, 2), m.layers.len);
    try testing.expectEqual(@as(u32, 4), m.n_experts);
    try testing.expectEqualStrings("experts.bin", m.sidecar);
    const l = m.layers[1];
    try testing.expectEqual(@as(u64, 19_791_872), l.base_offset);
    try testing.expectEqual(Dtype.I16, l.segments[0].dtype);
    try testing.expectEqual(@as(u8, 3), l.segments[0].rank);
    try testing.expectEqual([3]u64{ 256, 80, 40 }, l.segments[0].shape);
    try testing.expectEqual(@as(u64, 3_298_304), l.segments[gu_components].offset);
    try testing.expectEqual(@as(u64, 2 * 4_947_456), rowBytes(m));
    // The served bank: 68 layers of 4,947,456 B are 336,427,008 B a row of every layer.
    try testing.expectEqual(@as(u64, 336_427_008), 68 * l.logical_bytes);
    try testing.expectEqual(@as(u64, 4_947_456), widestRecord(m));
}

test "ling manifest: overlapping layers, a mistyped segment or another format is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectError(error.LingManifestLayer, parse(arena.allocator(), twoLayers("4947968", "F16")));
    try testing.expectError(error.LingManifestSegment, parse(arena.allocator(), twoLayers("19791872", "I16")));
    try testing.expectError(error.LingManifestFormat, parse(arena.allocator(), "{\"format\":\"mtplx-expert-manifest-v1\"}"));
    try testing.expectError(error.LingManifestJson, parse(arena.allocator(), "not json"));
}
