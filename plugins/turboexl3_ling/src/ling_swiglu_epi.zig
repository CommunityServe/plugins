//! turboexl3_ling's prefill SwiGLU epilogue (SWIGLU_EPI): at a prompt's rows the up projection of the shared expert and
//! of the dense layers' MLP runs on turboexl3's `lingUpSwiglu`, whose finish runs the host's SwiGLU of the gate's value
//! and its own and stores the activation, where the chain stores up and runs the SwiGLU as launches of its own (the
//! copied MoE block's `sharedDown`, the trunk's `denseMlp`). The words are the chain's: the gate is the caller's
//! `linear`, up's inner product is `linear`'s own launches, and the SwiGLU is the host's statements (limit 0: mlx-serve's
//! fusedSwiGLU over the host's own table; a layer's `shared_limit_out`: swigluClampOut in f32 over MLX's f32 sigmoid of
//! every 16-bit pattern, tabulated once here); turboexl3's swiglu_test and src/ling_swiglu_epi_test.zig hold it word for
//! word. Null keeps the chain: the decode GEMVs' rows, a dtype or clamp the epilogue does not take.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ran = @import("ling_ran.zig");
/// LAYER_WHAT: a layer-major dense layer's wide spans over the up W_hat of its first (`up`)
const layer_what = @import("ling_layer_what.zig");

/// MLX's f32 sigmoid of every 16-bit pattern of bf16 and f16, taken to f32: made at the first clamped call, kept.
var f32_tables: [2]?mlx.mlx_array = @splat(null);

/// The f32 table of the clamped SwiGLU (also read by src/ling_shared_layer.zig, so both read one table).
pub fn f32Table(s: mlx.mlx_stream, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    if (f32_tables[slot]) |t| return t;
    const t = try t3.lingSigmoidTable(s, dt, true);
    f32_tables[slot] = t;
    return t;
}

pub const Site = enum { dense, shared };

/// The activation h of x through the gate's `gate` (the caller's `linear` of the gate projection of x) and the up
/// projection (EXL3: its trellis, suh, svh) at the layer's clamp `limit` (0 none), or null where turboexl3's epilogue
/// does not serve the call. `host_table` is the block's `fused_act.sigmoidTableFor` (the table its fusedSwiGLU reads).
/// The caller owns the result. A layer-major dense layer's wide span runs the epilogue over the layer's kept W_hat of up
/// (LAYER_WHAT, src/ling_layer_what.zig: turboexl3's `lingUpSwigluW`), the same words. Each call the epilogue serves marks
/// the run-path line.
pub fn up(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, gate: mlx.mlx_array, limit: f32, comptime host_table: anytype, site: Site) !?mlx.mlx_array {
    if (!(limit >= 0 and limit <= 128 and limit == @floor(limit))) return null;
    const lim: u32 = @intFromFloat(limit);
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const tab = if (lim == 0) try host_table(s, dt) else try f32Table(s, dt);
    const proj: t3.Proj = .{ .trellis = w, .suh = suh, .svh = svh };
    // LAYER_WHAT: a dense layer's up over the layer's kept W_hat where its scope serves the call (the shared expert's is
    // SHARED_LAYER_WHAT's), else its own per call
    const kept = if (site == .dense) try layer_what.upSwiglu(s, x, proj, gate, lim, tab) else null;
    const h = kept orelse ((try t3.lingUpSwiglu(s, x, proj, gate, lim, tab, .mul1)) orelse return null);
    ran.mark(.SWIGLU_EPI);
    return h;
}
