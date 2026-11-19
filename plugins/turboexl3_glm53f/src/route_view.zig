//! `TURBOEXL3_GLM53F_ROUTE_VIEW=1` (on by default, exact): a decode or verify call's route ids handed to the routed call
//! as an int32 view of the router's uint32 ids (src/streamed/glm5n/experts.zig's decode lane), where the stock lane
//! casts them (`astype(ids, int32)`: a copy launch on the path from the routing barrier's router to the hit wave's LUT
//! gather, every MoE layer of every verify).
//!
//! An expert id is below GLM-5.3-Flash's 288 routed experts, so its uint32 and int32 words are the same bits; MLX's View
//! at the same item size shares the ids' buffer (backend/gpu/primitives.cpp `View::eval_gpu`): no launch. Ids in any
//! other type are cast as before. The ids the trunk hands the bank are the router's uint32 (glm53f's NAX router, or the
//! kit block's route).
//! copied_from: CommunityServe plugins/turboexl3_glm53f/src/route_view.zig@ac87fe1 (its request summary line left out:
//! the bank's `meters` print the stream's counters only).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("host/log.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_ROUTE_VIEW`, set by the plugin at its open, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTE_VIEW=1: a decode or verify call's route ids as an int32 view of the router's uint32 ids (ids below 288: the same bits; MLX's View shares the buffer) where the stock lane casts them: one copy launch less a routed layer\n";
/// Calls whose ids took the view since the load.
pub var views: u64 = 0;
var engaged = false;

/// Whether a call's ids take the view (MLX-free): the switch on and uint32 ids; any other type is cast by the caller.
pub fn takes(dt: mlx.mlx_dtype) bool {
    return on and dt == .uint32;
}

/// The int32 view of uint32 `ids` (a new array sharing their buffer), null with the switch off or ids of another type
/// (the caller casts them).
pub fn view(s: mlx.mlx_stream, ids: Array) !?Array {
    if (!takes(mlx.mlx_array_dtype(ids))) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_view(&out, ids, .int32, s));
    views += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTE_VIEW engaged: a {d}-row call's route ids as an int32 view\n", .{mlx.getShape(ids)[0]});
    }
    return out;
}

test "turboexl3_glm53f route_view: with the switch on only uint32 ids take the view (int32 ids need no cast, other types are cast as before); off, none does" {
    const t = std.testing;
    const saved = on;
    defer on = saved;
    on = true;
    try t.expect(takes(.uint32));
    try t.expect(!takes(.int32));
    try t.expect(!takes(.int64));
    try t.expect(!takes(.uint16));
    try t.expect(!takes(.float32));
    on = false;
    try t.expect(!takes(.uint32));
}
