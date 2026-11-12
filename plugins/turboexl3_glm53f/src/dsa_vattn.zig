//! `TURBOEXL3_GLM53F_DSA_VATTN=1` (opt-in, exact): a DSA layer's verify-window sparse latent attention (glm5_next
//! `sparseLatentAttention` at 2 to 8 rows) without its gather, through patches/turboexl3-model.patch.
//! The stock chain gathers the selected latent rows into keys [n, W, R] (`take` over the clamped indices, a `maximum`
//! before it), then `latentAttendFolded`: the scale's multiply, mlx_matmul(qs, keys^T), the mask's where, the precise
//! softmax and mlx_matmul(p, keys), each matmul MLX's regular NAX GEMM over a batch of n rows. Here the same chain with
//! the two matmuls kernels/dsa_vattn.metal: MLX's NAX GEMM loop (the same tile_matmad_nax on the same operands in the
//! same k order, the same past-the-end zeros), its B tile's rows read from the latent at each row's clamped index, where
//! MLX reads the gathered copy, which holds those values. So the same output, without the gather's write and read and
//! its two launches. src/dsa_gpu_test.zig holds the output to the stock chain's bit for bit at 1 to 4 rows.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const lane_qmm = @import("mlx_serve/sdk/blocks/qmm/lane_qmm.zig");
const naxAvailable = @import("mlx_serve/sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_VATTN=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// Windows attended since the load (src/dsa_gpu_test.zig reads that the gathered GEMMs ran).
pub var runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_VATTN=1: a DSA layer's verify-window sparse latent attention (2 to 8 rows) without the gather: its two matmuls as MLX's NAX GEMM loop (the same op on the same operands in the same k order) with the keys read from the latent at the selected rows, the stock chain's output\n";
/// The rows it takes: below 2 MLX runs other GEMMs (the thin and split-K ones), past 8 the forward takes the NAX
/// sparse kernel (glm5_next `sparseLatentNax`).
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 8;

const source = @embedFile("kernels/dsa_vattn.metal");
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// One gathered GEMM: D [n, m, cols] = a [n, m, k] times the keys, `tb` keys^T (keys [cols = w, k = r]) or keys (keys
/// [k = w, cols = r]), the keys the rows of `lat` [1, T, r] at each row's clamped `idx` [1, n, w].
fn gemm(s: mlx.mlx_stream, a: Array, lat: Array, idx: Array, n: c_int, m: c_int, cols: c_int, k: c_int, r: c_int, w: c_int, tb: bool, dt: mlx.mlx_dtype) !Array {
    const kern = try kda_prompt.kernelOf(&kernel, "turboexl3_glm53f_dsa_vattn", &.{ "a", "lat", "idx" }, &.{"d"}, source, try lane_qmm.prefillHeader());
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, m, cols }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(cols + 127, 128) * 256, @divTrunc(m + 63, 64), n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "VM", m }, .{ "VN", cols }, .{ "VK", k }, .{ "VR", r }, .{ "VW", w }, .{ "VTB", @as(c_int, @intFromBool(tb)) } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    var outs: [1]Array = undefined;
    try kda_prompt.applyKernel(s, kern, cfg, &.{ a, lat, idx }, &outs);
    return outs[0];
}

/// `latentAttendFolded(q4, keys, mask, scale)` over keys = `lat2`[max(`ic`, 0)] without the keys: `q4` [n, h, 1, r],
/// `lat2` [1, T, r], `ic` int32 [1, n, w], `mask` [n, 1, w] (the slots in use). [n, h, 1, r], a new array; null with the
/// switch off, off the GPU or its tensor units, or outside 2 to 8 rows of GLM-5.3-Flash's 64 heads and 512-wide latent.
pub fn attend(s: mlx.mlx_stream, q4: Array, lat2: Array, ic: Array, mask: Array, scale: ?f32) !?Array {
    if (!on or !mlx.streamIsGpu(s) or !naxAvailable()) return null;
    const qs = mlx.getShape(q4);
    if (qs.len != 4 or qs[0] < min_rows or qs[0] > max_rows or qs[1] != 64 or qs[2] != 1 or qs[3] != 512) return null;
    const n = qs[0];
    const h = qs[1];
    const r = qs[3];
    const ls = mlx.getShape(lat2);
    if (ls.len != 3 or ls[0] != 1 or ls[1] < 1 or ls[2] != r) return null;
    const is = mlx.getShape(ic);
    if (is.len != 3 or is[0] != 1 or is[1] != n or is[2] < 1) return null;
    const w = is[2];
    const dt = mlx.mlx_array_dtype(q4);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(lat2) != dt or mlx.mlx_array_dtype(ic) != .int32) return null;
    // latentAttendFolded's statements, the two matmuls the gathered GEMMs
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    try mlx.check(mlx.mlx_reshape(&q3, q4, &[_]c_int{ n, h, r }, 3, s));
    var qsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qsc);
    if (scale) |sv| {
        const sc_f = mlx.mlx_array_new_float(sv);
        defer _ = mlx.mlx_array_free(sc_f);
        var sc_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc_t);
        try mlx.check(mlx.mlx_astype(&sc_t, sc_f, dt, s));
        try mlx.check(mlx.mlx_multiply(&qsc, q3, sc_t, s));
    } else _ = mlx.mlx_array_set(&qsc, q3);
    const scores = try gemm(s, qsc, lat2, ic, n, h, w, r, r, w, true, dt);
    defer _ = mlx.mlx_array_free(scores);
    const ninf_f = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf_f);
    var ninf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ninf);
    try mlx.check(mlx.mlx_astype(&ninf, ninf_f, dt, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, mask, scores, ninf, s));
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, true, s));
    const o = try gemm(s, p, lat2, ic, n, h, r, w, r, w, false, dt);
    defer _ = mlx.mlx_array_free(o);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ n, h, 1, r }, 4, s));
    runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_VATTN engaged: a {d}-row verify window over {d} selected slots, its two matmuls reading the latent at the selected rows\n", .{ n, w });
    }
    return out;
}
