//! Cut edge: the two pieces of sushi's dflash.zig the vision tower's load-time requantization reaches (`DflashLinear`
//! as a holder of MLX affine codes, and `quantizeDense`). Ours; copied from sushi's dflash.zig, not generated. The
//! 4-bit g64 fused matmul of `DflashLinear.apply` is left out: the tower runs 8-bit g128 through
//! `vision_common.quantizedMatmul`.

const mlx = @import("mlx.zig");

/// One linear: `bits == 0` is dense `[in, out]`; quantized keeps the checkpoint's `[out, in]` packing for
/// `mlx_quantized_matmul` with transpose=true.
pub const DflashLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    bits: u32 = 0,
    group_size: u32 = 0,

    pub fn isQuantized(self: *const DflashLinear) bool {
        return self.bits != 0;
    }

    pub fn deinit(self: *DflashLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
    }
};

/// Affine-quantize a dense `[out, in]` weight in place of a transpose — the
/// packed layout is exactly what `mlx_quantized_matmul(transpose=true)` reads.
pub fn quantizeDense(raw: mlx.mlx_array, bits: u32, group: u32, s: mlx.mlx_stream) !DflashLinear {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(
        &triple,
        raw,
        mlx.mlx_optional_int.some(@intCast(group)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale
        s,
    ));
    if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
    var out = DflashLinear{
        .w = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
        .bits = bits,
        .group_size = group,
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, triple, 2));
    return out;
}
