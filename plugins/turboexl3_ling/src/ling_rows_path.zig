//! turboexl3_ling's routed prompt rows (ROWS_PATH, exact): each chunk of a wide (prompt) routed call (src/ling_math.zig
//! `chunks()`) runs turboexl3's prompt call over gathered rows (`moeLingRows`) in place of the chunk's take, its cast to
//! f32 and `moeLing` at top-1: the chunk's act rows are the row map the chain's prepare reads the call's x through, so
//! the gathered bf16 copy, the f32 copy and the prepare's f32 read become one bf16 read in the prepare. The prepare
//! rounds each x value to f16 through f32 whatever x's dtype, and the finish stores the one f32 sum, so every row's bytes
//! are the chain's (turboexl3's models/ling/rows_test.zig holds them to this chunk's take + f32 + `moeLing`, on the
//! loader's views). The call's x is cast to bf16 once where it is not (the chunks' own rounding; the streamer hands
//! every prompt call bf16). Decode waves and cold rows keep their paths.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// turboexl3's prompt call over gathered rows: x bf16 [n, hidden], rows (the row map) and inds u32 [S], scores f32 [S],
/// y f32 [S, hidden] at Ling's SwiGLU `limit`; the caller frees y. `dfin`: PF_DFIN's whole-call choice
/// (src/ling_math.zig), the fused finish or the served chain (`moeLingRowsWith`).
pub fn call(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, bank: t3.Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, limit: c_int, dfin: bool) !mlx.mlx_array {
    return t3.moeLingRowsWith(s, x, rows, bank, inds, scores, .mul1, limit, .float32, .{ .dfin = dfin });
}
