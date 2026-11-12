//! turboexl3_ling's shared expert at decode rows on turboexl3 (SHARED_F3): the copied MoE block's `sharedDown` hands its
//! three EXL3 projections here first, and turboexl3's `lingSharedExpert` runs them in two dependent launches (the gate
//! and the up in one DEC-ONE launch whose output blocks' pairs run the host's SwiGLU and the down's prepare, then the
//! down's GEMV) where `lingSharedTakes` holds (decode rows up to DEC-ONE's 4); null keeps the block's chain. The composed
//! chain's words: `linear(gate)`, `linear(up)`, the host's SwiGLU (mlx-serve's fusedSwiGLU at a layer without a clamp,
//! reading its own sigmoid table; ling_exl3.zig's swigluClampOut at `shared_limit_out`, with MLX's f32 sigmoid of every
//! 16-bit pattern tabulated once here), `linear(down)` (turboexl3's models/ling/shared_test.zig holds them word for word).
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
/// The run-path line (src/ling_ran.zig): SHARED_F3 marks where its two launches ran.
const ran = @import("ling_ran.zig");

/// MLX's f32 sigmoid of every 16-bit pattern of bf16 and f16, taken to f32: made at the first clamped call, kept.
var f32_tables: [2]?mlx.mlx_array = @splat(null);

fn f32Table(s: mlx.mlx_stream, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    if (f32_tables[slot]) |t| return t;
    const t = try t3.lingSigmoidTable(s, dt, true);
    f32_tables[slot] = t;
    return t;
}

/// The shared expert of x through gate, up and down (EXL3: each its trellis, suh, svh) at the layer's clamp `limit`
/// (0 none), or null where turboexl3's two launches do not serve the call. `host_table` is the MoE block's
/// `fused_act.sigmoidTableFor` (the table its fusedSwiGLU reads). The caller owns the result.
pub fn run(s: mlx.mlx_stream, x: mlx.mlx_array, gate: [3]mlx.mlx_array, up: [3]mlx.mlx_array, down: [3]mlx.mlx_array, limit: f32, comptime host_table: anytype) !?mlx.mlx_array {
    const g: t3.Proj = .{ .trellis = gate[0], .suh = gate[1], .svh = gate[2] };
    const u: t3.Proj = .{ .trellis = up[0], .suh = up[1], .svh = up[2] };
    const d: t3.Proj = .{ .trellis = down[0], .suh = down[1], .svh = down[2] };
    if (!(limit >= 0 and limit <= 128 and limit == @floor(limit))) return null;
    const lim: u32 = @intFromFloat(limit);
    if (!(try t3.lingSharedTakes(x, g, u, d, .mul1))) return null;
    const dt = mlx.mlx_array_dtype(x);
    const tab = if (lim == 0) try host_table(s, dt) else try f32Table(s, dt);
    const y = try t3.lingSharedExpert(s, x, g, u, d, lim, tab, .mul1);
    ran.mark(.SHARED_F3);
    return y;
}
