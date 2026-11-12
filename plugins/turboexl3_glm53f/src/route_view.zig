//! `TURBOEXL3_GLM53F_ROUTE_VIEW=1` (opt-in, exact): a decode or verify call's route ids handed to the routed call as an
//! int32 view of the router's uint32 ids (in
//! src/streamed/glm5n/experts.zig's decode lane), where the stock lane casts them (`astype(ids, int32)`: a copy launch on the path
//! from the routing barrier's router to the hit wave's LUT gather, every MoE layer of every verify).
//!
//! An expert id is below GLM-5.3-Flash's 288 routed experts, so its uint32 and int32 words are the same bits; MLX's
//! View at the same item size shares the ids' buffer (backend/gpu/primitives.cpp `View::eval_gpu`): no launch. Ids in
//! any other type are cast as before. src/moe_combine_gpu_test.zig holds the view to the cast's values and checks it
//! shares the router's buffer, at 1 to 4 rows.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_ROUTE_VIEW=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTE_VIEW=1: a decode or verify call's route ids as an int32 view of the router's uint32 ids (ids below 288: the same bits; MLX's View shares the buffer) where the stock lane casts them: one copy launch less a routed layer\n";
/// Calls whose ids took the view since the load (the request's summary line).
pub var views: u64 = 0;
var engaged = false;

/// Whether a call's ids take the view (MLX-free): uint32 ids; any other type is cast by the caller.
pub fn takes(dt: mlx.mlx_dtype) bool {
    return dt == .uint32;
}

/// The int32 view of uint32 `ids` (a new array sharing their buffer), null with the switch off or ids of another type
/// (the caller casts them).
pub fn view(s: mlx.mlx_stream, ids: Array) !?Array {
    if (!on or !takes(mlx.mlx_array_dtype(ids))) return null;
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

/// The request's line (the plugin's meters): the calls whose ids took the view.
pub fn summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTE_VIEW views={d}\n", .{views}) catch buf[0..0];
}

test "turboexl3_glm53f stream route_view: only uint32 ids take the view (int32 ids need no cast, other types are cast as before); the summary counts the views" {
    const t = std.testing;
    try t.expect(takes(.uint32));
    try t.expect(!takes(.int32));
    try t.expect(!takes(.int64));
    try t.expect(!takes(.float32));
    var buf: [96]u8 = undefined;
    views = 7;
    defer views = 0;
    try t.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTE_VIEW views=7\n", summary(&buf));
}
