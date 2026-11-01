//! copied_from: src/plugins/minimax_m3/nvfp4_format.zig@37e90382b439f163474f99d3c78522ae88caaf12
const std = @import("std");
const mlx = @import("mlx_stream").sdk.mlx;
const ops = @import("mlx_stream").ops;
const bank = @import("nvfp4_bank.zig");
extern "c" fn glm5_gather_qmm_nvfp4(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, scales: mlx.mlx_array, global_scale: mlx.mlx_array, rhs_indices: mlx.mlx_array, sorted: bool, s: mlx.mlx_stream) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;
pub const Format = struct {
    pub fn gather(g: *ops.MlxOps, x: mlx.mlx_array, p: bank.Arrays(mlx.mlx_array), ids: mlx.mlx_array) !mlx.mlx_array {
        const n = g.shapeOf(p.w).d[0];
        const scale = try g.reshape(try g.slice(p.global_scale, &.{ 0, 0 }, &.{ n, 1 }, &.{ 1, 1 }), &.{n});
        var result = mlx.mlx_array_new();
        if (glm5_gather_qmm_nvfp4(&result, x, p.w, p.scales, scale, ids, false, g.s) != 0) {
            _ = mlx.mlx_array_free(result);
            std.log.err("minimax_m3: NVFP4 gather: {s}", .{std.mem.span(dsv41_alloc_last_error())});
            return error.MlxError;
        }
        return g.adopt(result);
    }
};
