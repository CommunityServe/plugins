//! copied_from: src/sdk/weights/load.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! A pack's safetensors read into a `Weights` map: every shard the index names (or one file), with the tensors a load
//! leaves out dropped and f16 side tensors and 1-D tables narrowed to the bf16 activations they meet.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");
const resident = @import("resident.zig");
const affineParamsFromGeometry = @import("../quant/params.zig").affineParamsFromGeometry;
pub const Weights = resident.Weights;

/// A tiny affine-quantized tensor rewritten as the dense `[out, in]` a dense
/// checkpoint ships (untransposed), so the paths keyed on the dense layout
/// (hc prefill, the fused HC read, the softmax_gate fold) serve it too.
pub fn dequantizeSmall(w: *mlx.mlx_array, sc: *mlx.mlx_array, bi: *mlx.mlx_array, in_dim: u32, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    if (sc.ctx == null or bi.ctx == null) return;
    const qp = affineParamsFromGeometry(w.*, sc.*, in_dim) orelse return;
    var d = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_dequantize(&d, w.*, sc.*, bi.*, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", .{ .ctx = null }, .{ .value = mlx.mlx_array_dtype(sc.*), .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(d));
    try owned.append(allocator, d);
    w.* = d;
    sc.* = mlx.mlx_array_new();
    bi.* = mlx.mlx_array_new();
}

// ── tests ──

const testing = std.testing;
