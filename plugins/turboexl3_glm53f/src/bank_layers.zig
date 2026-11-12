//! The routed bank's slot-row bytes per layer, MLX-free: each bank layer's record segments summed, the bytes one slot row of
//! that layer holds (the stream allocates every routed layer's rows on the layer's own segments). They come from the pack's
//! `expert-manifest-v2.json` layer table. An SC pack routes its layer 3 at K + 1 (sc_3.10: K4, 12,619,776 B a row) and
//! layers 4-44 at K (K3, 9,474,048 B), so the widest row that `stream_room.rowsFor` bills in every layer over-counts 41 of
//! the 42 layers by a third (`stream_room.rowsByLayer` bills each layer's own).

const std = @import("std");

pub const manifest_file = "expert-manifest-v2.json";

const Segment = struct { length: u64 };
const Layer = struct { layer: u32, logical_bytes: u64, segments: []const Segment };
const Doc = struct { layers: []const Layer };

/// `text` (the manifest's bytes) into `out`, one entry per bank layer in layer order: each layer's segment lengths summed.
/// Refused by name (`Glm53fBankLayers`) when the table is not one of `out.len` layers in order, each with segments whose
/// sum is its `logical_bytes`.
pub fn parse(a: std.mem.Allocator, text: []const u8, out: []u64) error{ Glm53fBankLayers, OutOfMemory }!void {
    const p = std.json.parseFromSlice(Doc, a, text, .{ .ignore_unknown_fields = true }) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.Glm53fBankLayers,
    };
    defer p.deinit();
    const layers = p.value.layers;
    if (layers.len != out.len) return error.Glm53fBankLayers;
    for (layers, out, 0..) |l, *o, i| {
        if (l.layer != i or l.segments.len == 0) return error.Glm53fBankLayers;
        var n: u64 = 0;
        for (l.segments) |s| n = std.math.add(u64, n, s.length) catch return error.Glm53fBankLayers;
        if (n == 0 or n != l.logical_bytes) return error.Glm53fBankLayers;
        o.* = n;
    }
}

/// The pack's manifest in `dir`, parsed into `out` (`parse`).
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, out: []u64) !void {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, manifest_file, a, .limited(256 << 20));
    defer a.free(text);
    try parse(a, text, out);
}

// ── tests ──

const testing = std.testing;

/// A v2 layer table in the converter's form (LAYOUT.md): `K`, the record and its segments, the records list after it.
const sc_like =
    \\{"format":"mtplx-expert-manifest-v2","dims":{"n_layers":3},
    \\ "layers":[
    \\  {"layer":0,"model_layer":3,"K":4,"record_bytes":12619776,"logical_bytes":12619776,"base_offset":0,
    \\   "segments":[{"component":"gate_proj.code","dtype":"I16","shape":[256,128,64],"offset":0,"length":4194304},
    \\               {"component":"gate_proj.rout","dtype":"F16","shape":[2048],"offset":4194304,"length":4096},
    \\               {"component":"gate_proj.rin","dtype":"F16","shape":[4096],"offset":4198400,"length":8192},
    \\               {"component":"up_proj.code","dtype":"I16","shape":[256,128,64],"offset":4206592,"length":4194304},
    \\               {"component":"up_proj.rout","dtype":"F16","shape":[2048],"offset":8400896,"length":4096},
    \\               {"component":"up_proj.rin","dtype":"F16","shape":[4096],"offset":8404992,"length":8192},
    \\               {"component":"down_proj.code","dtype":"I16","shape":[128,256,64],"offset":8413184,"length":4194304},
    \\               {"component":"down_proj.rout","dtype":"F16","shape":[4096],"offset":12607488,"length":8192},
    \\               {"component":"down_proj.rin","dtype":"F16","shape":[2048],"offset":12615680,"length":4096}]},
    \\  {"layer":1,"model_layer":4,"K":3,"record_bytes":9474048,"logical_bytes":9474048,"base_offset":3634495488,
    \\   "segments":[{"length":3145728},{"length":4096},{"length":8192},{"length":3145728},{"length":4096},{"length":8192},
    \\               {"length":3145728},{"length":8192},{"length":4096}]},
    \\  {"layer":2,"model_layer":5,"K":2.5,"record_bytes":7901184,"logical_bytes":7901184,"base_offset":6363029504,
    \\   "segments":[{"length":2621440},{"length":4096},{"length":8192},{"length":2621440},{"length":4096},{"length":8192},
    \\               {"length":2621440},{"length":8192},{"length":4096}]}],
    \\ "records":[{"layer":0,"expert":0,"sidecar_offset":0,"stats":{"rout_max":[1.5,2.0,0.25]}}]}
;

test "turboexl3_glm53f stream bank layers: each layer's slot row is its own record's segments (K4, K3 and K2.5 rows)" {
    var out: [3]u64 = undefined;
    try parse(testing.allocator, sc_like, &out);
    try testing.expectEqual(@as(u64, 12_619_776), out[0]);
    try testing.expectEqual(@as(u64, 9_474_048), out[1]);
    try testing.expectEqual(@as(u64, 7_901_184), out[2]);
}

test "turboexl3_glm53f stream bank layers: a table of another length, out of order or whose segments miss its logical bytes is refused by name" {
    var two: [2]u64 = undefined;
    try testing.expectError(error.Glm53fBankLayers, parse(testing.allocator, sc_like, &two));
    var out: [3]u64 = undefined;
    const swapped = try std.mem.replaceOwned(u8, testing.allocator, sc_like, "{\"layer\":1,", "{\"layer\":2,");
    defer testing.allocator.free(swapped);
    try testing.expectError(error.Glm53fBankLayers, parse(testing.allocator, swapped, &out));
    const short = try std.mem.replaceOwned(u8, testing.allocator, sc_like, "\"logical_bytes\":9474048", "\"logical_bytes\":9474049");
    defer testing.allocator.free(short);
    try testing.expectError(error.Glm53fBankLayers, parse(testing.allocator, short, &out));
    try testing.expectError(error.Glm53fBankLayers, parse(testing.allocator, "{\"layers\":[", &out));
}
