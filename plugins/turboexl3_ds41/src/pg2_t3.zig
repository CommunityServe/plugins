//! pg2: a wide prompt layer's dense EXL3 linears' matmul and finish as one launch at the shapes this machine runs faster
//! that way (turboexl3's `pgemm`): turboexl3's switch set on at load and its tabled tiles' kernels compiled at load.
//!
//! turboexl3's `linearOn` at prefill rows runs the prepare, the W_hat decode, MLX's f16 matmul (z half [M, N] written) and
//! the finish (z read, y written: f16(H128(f16(z)) * svh) per 128-column block). pg2 runs the matmul and the finish as one
//! kernel: MLX's fused NAX GEMM body and gemm loop (its statements, SK 32, the 16 x 32 x 16 MMA on f16, k in MLX's order)
//! at the tile the pg2u probe measured per (K, N) on this machine, with the finish's statements as its epilogue (the
//! accumulators rounded to half as MLX's store rounds them, staged in threadgroup memory), launched in MLX's own
//! threadgroup shape. It takes a call only where MLX runs that regular kernel (no split K) on the NAX path, so y is the
//! chain's y word for word; z never reaches memory. The shapes (turboexl3 `pgemm.table`): wq_a, wkv (and the
//! compressors'), wq_b, wo_b, the shared expert's w1, w3, w2. The joined `wo_a` runs (src/woa_join.zig) are not taken.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// At load, before the model's first prompt: turboexl3's switch on, no check hooked
pub fn configure() void {
    t3.pgemm.mode.get().* = .on;
    t3.pgemm.check_hook.get().* = null;
}

/// After the load: every tabled tile's kernel compiled for the activations' dtype, and the compressors'
/// 5120 x 512 for f32 (the pooling layers run their wkv and wgate on f32 rows, so their y is f32), so a prompt's first
/// call pays no JIT
pub fn warm(s: mlx.mlx_stream, act: mlx.mlx_dtype) !void {
    try t3.pgemm.warm(s, act, null);
    if (act != .float32) try t3.pgemm.warm(s, .float32, .{ 5120, 512 });
}
