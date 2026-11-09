//! A dense EXL3 SwiGLU MLP (DeepSeek-V4.1's shared expert) at a verify's decode rows as one launch: the gate's and the
//! up's finishes, the clamped SwiGLU and the down's prepare (turboexl3's `sharedSwiglu`), the host chain's words; null
//! where turboexl3 does not take the rows (the caller runs its chain).

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
