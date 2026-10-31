//! `StqStream.allResident`, which keeps the live (every expert resident) config off `extendState`'s cross-chunk drain:
//! true only when every active layer routes through its resident map. Hermetic (no MLX call, no pack): the engine's
//! layers are built by hand with the four fields the predicate reads. Run by the claims step.
const std = @import("std");
const dstream = @import("sushi/dsv41/dsv41_stream.zig");
const stream = @import("sushi/expert_stream.zig");

const LayerState = @typeInfo(@FieldType(stream.Engine, "layers")).pointer.child;
const GroupedLayer = @typeInfo(@typeInfo(@FieldType(LayerState, "grouped")).optional.child).pointer.child;

/// A layer as `allResident` sees it: `mapped` gives it a grouped pool and a resident map.
fn layer(active: bool, mapped: bool, borrowed: bool, grouped: *GroupedLayer) LayerState {
    var l: LayerState = undefined;
    l.active = active;
    l.borrowed = borrowed;
    l.grouped = if (mapped) grouped else null;
    l.resident_map = .{ .ctx = if (mapped) @ptrFromInt(0x1000) else null };
    return l;
}

fn allResident(layers: []LayerState) bool {
    var engine: stream.Engine = undefined;
    engine.layers = layers;
    var st: dstream.StqStream = .{ .engine = &engine, .cfg = undefined };
    const hook = st.hook();
    return hook.all_resident.?(hook.ctx);
}

test "sushi_ds41 allResident: true only when every active layer routes through its resident map" {
    const t = std.testing;
    var g: GroupedLayer = undefined;
    // Every expert resident (the live config): no drain between prefill chunks.
    var full = [_]LayerState{ layer(true, true, false, &g), layer(true, true, false, &g), layer(true, true, false, &g) };
    try t.expect(allResident(&full));
    // An inactive layer (no experts in this engine) has no map and does not count.
    var dense_first = [_]LayerState{ layer(false, false, false, &g), layer(true, true, false, &g) };
    try t.expect(allResident(&dense_first));
    // A streamed layer (a cache smaller than the pack: no resident map) means routes can refill slabs.
    var streamed = [_]LayerState{ layer(true, true, false, &g), layer(true, false, false, &g), layer(true, true, false, &g) };
    try t.expect(!allResident(&streamed));
    // A layer released for a prefill's working set reads through the union workspace.
    var borrowed = [_]LayerState{ layer(true, true, false, &g), layer(true, true, true, &g) };
    try t.expect(!allResident(&borrowed));
}
