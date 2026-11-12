//! The trunk's EXL3 projections and Ling-3.1's per-layer SwiGLU clamps, read by the plugin's patched copies of
//! mlx-serve's Ling forward (the copied trunk.zig, linear.zig, moe.zig, kda.zig and mla.zig): a `Linear` whose weight
//! is an EXL3 trellis runs on turboexl3, and the late layers' shared expert takes Ling's clamp after the SiLU.

const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const clamps = @import("clamps.zig");
/// LAYER_WHAT: a layer-major layer's dense prefill linears over W_hats decoded once for its spans
const layer_what = @import("ling_layer_what.zig");

/// The mul1 codebook's multiplier, the one turboexl3 decodes (exllamav3's `codebook.cuh`).
pub const mul1_multiplier: u32 = 0x83DCD12D;

/// A weight is an EXL3 trellis: integer 16-bit [in/16, out/16, n], beside its suh and svh (F16).
pub fn isTrellis(w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array) bool {
    if (w.ctx == null or suh.ctx == null or svh.ctx == null) return false;
    const dt = mlx.mlx_array_dtype(w);
    if (dt != .int16 and dt != .uint16) return false;
    if (mlx.mlx_array_ndim(w) != 3) return false;
    return mlx.mlx_array_dtype(suh) == .float16 and mlx.mlx_array_dtype(svh) == .float16;
}

/// `x @ W` for an EXL3 projection: turboexl3's linear (y in x's dtype). A layer-major layer's wide span of a linear its
/// LAYER_WHAT scope holds (src/ling_layer_what.zig) runs `linear`'s launches over the layer's kept W_hat, `linear`'s
/// words; every other call is `linear` as it was.
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array) !mlx.mlx_array {
    if (try layer_what.linear(s, x, .{ .trellis = trellis, .suh = suh, .svh = svh })) |y| return y;
    return t3.linear(s, x, trellis, suh, svh, .mul1);
}

/// The exact verify (plugin.zig `forward`): set around a verify forward, so each row's bytes are serial decode's. The
/// copies read it where several rows would take another kernel than one row: a dense matmul (MLX picks a matrix kernel
/// for several rows, a vector kernel for one), the MLA attention and the KDA recurrence.
pub var exact_rows: bool = false;

/// An exact verify's rows of a dense matmul in one launch, each row serial decode's one-row gemv (src/ling_rows.zig).
pub const denseRows = @import("ling_rows.zig").denseRows;

/// Several rows on axis -2 while the exact verify runs: the caller takes them in one `denseRows` launch.
pub fn splitRows(x: mlx.mlx_array) bool {
    if (!exact_rows) return false;
    const sh = mlx.getShape(x);
    return sh.len >= 2 and sh[sh.len - 2] > 1;
}

/// Model layer `layer`'s shared-expert clamp (0: none), as config.zig read it (the copied trunk.zig's load sets each
/// MoE layer's `shared_limit_out` from it; the routed clamps reach the bank through plugin.zig's `clamps.routed`).
pub fn sharedLimit(layer: u32) u32 {
    return clamps.shared(layer);
}

/// Ling-3.1's clamped SwiGLU of the shared expert: min(silu(g), L) * clamp(u, -L, L), in f32, back in g's dtype.
pub fn swigluClampOut(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array, limit: f32) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(gate);
    var tmp: [10]mlx.mlx_array = @splat(mlx.mlx_array_new());
    defer for (tmp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try mlx.check(mlx.mlx_astype(&tmp[0], gate, .float32, s));
    try mlx.check(mlx.mlx_astype(&tmp[1], up, .float32, s));
    try mlx.check(mlx.mlx_sigmoid(&tmp[2], tmp[0], s));
    try mlx.check(mlx.mlx_multiply(&tmp[3], tmp[0], tmp[2], s));
    tmp[4] = mlx.mlx_array_new_float(limit);
    tmp[5] = mlx.mlx_array_new_float(-limit);
    try mlx.check(mlx.mlx_minimum(&tmp[6], tmp[3], tmp[4], s));
    try mlx.check(mlx.mlx_clip(&tmp[7], tmp[1], tmp[5], tmp[4], s));
    try mlx.check(mlx.mlx_multiply(&tmp[8], tmp[6], tmp[7], s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, tmp[8], dt, s));
    return out;
}
