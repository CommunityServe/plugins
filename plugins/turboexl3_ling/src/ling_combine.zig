//! turboexl3_ling's routed combine at prompt rows (COMBINE, exact): a wide (prompt) routed call goes through the copied
//! streamer's JOINLESS route (`Hook.routedParts`: the waves' outputs unjoined, each pair's (source, row)) and
//! turboexl3's Ling combine over them in one launch (`root.ling.combineParts`), in place of the join (`concatenate` +
//! `take`) and the three MLX ops of `Source.rows` (`sum(multiply(astype(routed, f32), expand_dims(weights, -1)), -2)`),
//! whose words it gives (turboexl3's combine_test.zig). Decode calls keep their path. The copied source (the overlay's
//! ling_source.zig `partsSum`) calls it through `mlx_host`.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// turboexl3's Ling combine over a wide call's JOINLESS parts (`outs` at most `root.ling.combine_max_sources`, `loc`
/// int32 [n, k, 2], `weights` f32 [n, k]) -> f32 [n, hidden]; the caller frees it.
pub fn combineParts(s: mlx.mlx_stream, outs: []const mlx.mlx_array, loc: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    return t3.ling.combineParts(s, outs, loc, weights);
}
