//! F3: the decode shared expert's EXL3 gate and up finishes, the clamped SwiGLU and the down's prepare in one launch
//! (turboexl3's `sharedSwiglu`). The copied model's `shared` asks `fused` first (where F3 takes the call: the down's
//! output; else null and the host's chain). F3 takes x [L, hidden] through the EXL3 gate (w1), up (w3) and down (w2) at
//! the decode rows the dense chain takes (`sharedSwigluTakes`), the clamped SwiGLU's limit > 0.
//!
//! At a verify's rows (5 to 16: DEC-ONE keeps up to 4 as they are) the shared expert runs the gate's and the up's
//! prepare, GEMV and finish, the host's clamped SwiGLU in f32 (one compiled elementwise launch in a decode stretch), then
//! the down's prepare, GEMV and finish. F3 makes the two finishes, the SwiGLU and the down's prepare one launch
//! (`dense.text_swiglu_prep`: for each 128-wide block, text_dfinish's statements on the gate's and the up's planes, MLX's
//! Minimum, Maximum, Sigmoid and Multiply functors in clampedSwiGLU's order, the activation dtype's rounding, then
//! text_prep_s's statements, nothing between them stored): three launches and two dependent levels fewer a layer, the
//! same words (turboexl3's CPU test holds the statements; its GPU test compares every word at rows 1 to 16 with the
//! host's chain op for op). The shared expert's output is the chain's words, so what reads it is unchanged.
//!
//! The reads (M5 Max): every word the host chain's at rows 1 to 16 (taken at 5 to 16), and the 40-layer trunk chain at
//! rows 5, 6 and 8 -0.056, -0.029 and -0.218 ms a verify; card +0.34%, greedy +0.18%, the texts the same bytes, F3's
//! words equal to the chain's over 4899 + 9503 calls.

const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

const A = mlx.mlx_array;

fn takes(x: A, w1: t3.Proj, w3: t3.Proj, w2: t3.Proj, limit: f32, act: mlx.mlx_dtype) !bool {
    if (!(limit > 0)) return false;
    return t3.sharedSwigluTakes(x, w1, w3, w2, act, .mul1);
}

/// The shared expert's output [L, hidden] in `act` on F3 where it takes the call; else null (the caller runs the host's
/// chain). The caller owns the result.
pub fn fused(s: mlx.mlx_stream, x: A, w1: t3.Proj, w3: t3.Proj, w2: t3.Proj, limit: f32, act: mlx.mlx_dtype) !?A {
    if (!try takes(x, w1, w3, w2, limit, act)) return null;
    return try t3.sharedSwiglu(s, x, w1, w3, w2, limit, act, .mul1);
}
