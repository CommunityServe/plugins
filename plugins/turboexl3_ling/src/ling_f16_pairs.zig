//! turboexl3_ling's routed prompt pairs in f16 (PF_F16_PAIRS): each chunk of a wide (prompt) routed call on the rows
//! path (src/ling_math.zig `rowsCall`) stores its pairs' rows as f16 in place of f32, through turboexl3's
//! `ling.moeLingRowsPairsF16`. Exact: a chunk is a top-1 call at weight 1 (each pair) or +0 (each pad row of a padded
//! call), so the chain's finish stores `0 + w * float(h)`, an f16 value, in every word; the f16 store holds that value,
//! and every reader widens it back exactly: the combine (src/ling_combine.zig, turboexl3's `combineParts` over f16
//! sources), the join's take and astype, the JOINLESS merge's concatenate. Half the bytes of every pair's row written and
//! read. The chunk's weights go to turboexl3 on the host, and a call whose weights are not all 1 or +0 is refused there
//! (error.Exl3PairsNotExact): that chunk keeps its f32 call. Decode waves and cold rows keep their paths. turboexl3's
//! models/ling/f16pairs_test.zig holds the f16 pairs and their combine to the f32 ones, word for word.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// A rows-path call's weights on the host (src/ling_math.zig `rowsCall`'s, which it hands turboexl3 as an array): 1 for
/// each of `rows` pairs; a call of `rows` <= 16 pairs (`small`) padded to `padded` row-slots with +0 for each pad row.
/// `buf` holds at least `padded` (small) or `rows` values.
pub fn weightsInto(buf: []f32, rows: usize, small: bool, padded: usize) []const f32 {
    const n = if (small) padded else rows;
    for (buf[0..n], 0..) |*w, i| w.* = if (i < rows) 1.0 else 0.0;
    return buf[0..n];
}

/// turboexl3's prompt call over gathered rows with its pairs stored as f16: x bf16 [n, hidden], rows (the row map) and
/// inds u32 [S], `weights` the host weights (`weightsInto`), y f16 [S, hidden] at Ling's SwiGLU `limit`; null when
/// turboexl3 refuses the weights (not f16-exact: the caller keeps its f32 call). `dfin`: PF_DFIN's whole-call choice
/// (src/ling_math.zig), the fused finish or the served chain. The caller frees y.
pub fn call(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: t3.Bank, inds: mlx.mlx_array, weights: []const f32, limit: c_int, dfin: bool) !?mlx.mlx_array {
    return t3.ling.moeLingRowsPairsF16(s, x, rows, bank, inds, weights, .mul1, limit, .{ .dfin = dfin }) catch |e| switch (e) {
        error.Exl3PairsNotExact => null,
        else => return e,
    };
}
