//! copied_from: src/engines/mlx/methods/mrope.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The legacy `Transformer`'s methods over `sdk/blocks/rope/mrope.zig` (what reads the engine's model state).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;
const yarnMscale = @import("yarn.zig").yarnMscale;
const MropeCosSin = @import("../../../sdk/blocks/rope/mrope.zig").MropeCosSin;

pub fn mropeContext(ctx: *const ForwardCtx) mrope.PositionContext {
    return .{ .pos = ctx.mrope_pos.?, .total = ctx.mrope_total, .delta = ctx.mrope_delta };
}

pub fn endMropeChunk(ctx: *ForwardCtx) void {
    if (ctx.mrope_cos_cur) |c| _ = mlx.mlx_array_free(c);
    if (ctx.mrope_sin_cur) |sn| _ = mlx.mlx_array_free(sn);
    ctx.mrope_cos_cur = null;
    ctx.mrope_sin_cur = null;
}
