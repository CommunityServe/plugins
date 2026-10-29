//! copied_from: src/sdk/quant/mxfp8.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/quant/mxfp8.
//! Owns lmHeadMxfp8On, quantizeHeadMxfp8.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const Weights = @import("../weights/resident.zig").Weights;

/// LOSSY, so on by default only where it was measured: Flash Next (`qwen4`) on an integrated
/// CUDA GPU (DGX Spark), for a quantized pack whose head ships dense. A bf16 pack keeps its reference head.
pub fn lmHeadMxfp8On(setting: ?bool, cli: ?bool, cuda_integrated: bool, qwen4: bool, quant_bits: u32) bool {
    return setting orelse cli orelse (cuda_integrated and qwen4 and quant_bits > 0);
}

/// Quantize the dense head `w` (the map's `key`) to MXFP8 g32, the bytes vLLM serves under
/// `VLLM_MXFP8_LM_HEAD`, and hand the map the packed weight and its scales in place of the
/// bf16 copy, which is freed. Returns the scales, borrowed from the map.
pub fn quantizeHeadMxfp8(weights: *Weights, key: []const u8, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    try mlx.check(mlx.mlx_eval(pair)); // before the bf16 copy goes
    var qw = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qw);
    var qs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qs);
    try mlx.check(mlx.mlx_vector_array_get(&qw, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qs, pair, 1));
    var name: [176]u8 = undefined;
    try weights.put(try std.fmt.bufPrint(&name, "{s}.scales", .{key[0 .. key.len - ".weight".len]}), qs);
    weights.replace(key, qw);
    return qs;
}

// ── tests ──

const model_mod = @import("../config/model.zig");
const plugin_driver = @import("../../engines/mlx/plugin/driver.zig");
