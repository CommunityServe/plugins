//! turboexl3_ling's routed combine at prompt rows (COMBINE, exact): a prompt call's outputs stay unjoined (one source a
//! slot bank's rows-path call, each pair's (source, row)) and turboexl3's Ling combine sums them over the routes' weights
//! in one launch (`root.ling.combineParts`), the words of the join's take and `sum(multiply(astype(routed, f32),
//! expand_dims(weights, -1)), -2)` (turboexl3's combine_test.zig). src/ling_bank.zig calls it.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// turboexl3's Ling combine over a wide call's JOINLESS parts (`outs` at most `root.ling.combine_max_sources`, `loc`
/// int32 [n, k, 2], `weights` f32 [n, k]) -> f32 [n, hidden]; the caller frees it.
pub fn combineParts(s: mlx.mlx_stream, outs: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    return t3.ling.combineParts(s, outs, loc, weights);
}
