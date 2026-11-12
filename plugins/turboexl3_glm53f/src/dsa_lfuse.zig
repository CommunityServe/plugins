//! `TURBOEXL3_GLM53F_DSA_LFUSE=1` (opt-in, exact): a DSA layer's verify-window sparse latent attention (glm5_next
//! `sparseLatentAttention` at 2 to 8 rows, one chunk) in two launches at GLM-5.3-Flash's dims, through
//! patches/turboexl3-model.patch. The stock chain builds q4 (q_abs with its rows and heads swapped,
//! a copy), the mask (idx >= 0) and the gathered keys, then `latentAttendFolded`: the scale's multiply, mlx_matmul(qs,
//! keys^T), the mask's where, the precise softmax and mlx_matmul(p, keys); with DSA_VATTN the matmuls read the latent
//! at the selected rows. Here kernels/dsa_lfuse_scores.metal takes the copy, the multiply, the first matmul, the mask
//! and the where in one launch (q read from q_abs at the window row, each element times T(scale) in T, the unused slots
//! T(-inf) at the store), and kernels/dsa_lfuse_pv.metal the softmax and the second matmul in one (each head's max and
//! sum as MLX's block softmax reduces them, each p as it writes it, inside MLX's NAX GEMM loop). Each step is the stock
//! op's arithmetic on the same values in the same order, so the same output; src/dsa_lfuse_gpu_test.zig holds it to
//! the stock chain bit for bit at 1 to 8 rows.
//!
//! The dims are GLM-5.3-Flash's, constants in the texts: 64 heads, 512-wide latent rows, 2051 slots a row (the indexer's
//! 512 pools of 4 tokens, the last pool's 3 more); any other shape, a batch past 1, a window of more than one chunk, a
//! call without the scale, rows outside 2 to 8, or a device without tensor units keeps the stock chain (null).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const lane_qmm = @import("mlx_serve/sdk/blocks/qmm/lane_qmm.zig");
const naxAvailable = @import("mlx_serve/sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_LFUSE=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// Windows attended since the load (src/dsa_lfuse_gpu_test.zig reads that the two launches ran).
pub var runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_LFUSE=1: a DSA layer's verify-window sparse latent attention (2 to 8 rows) in two launches at GLM-5.3-Flash's dims: the query copy, scale, first matmul and mask in one, the precise softmax and second matmul in one, each the stock op's arithmetic in its order (the stock chain's output)\n";

/// GLM-5.3-Flash's latent attention, the texts' constants: heads, latent row width, selected slots a row.
pub const heads: c_int = 64;
pub const latent_w: c_int = 512;
pub const slots: c_int = 2051;
/// The rows it takes: below 2 MLX runs other GEMMs (the thin and split-K ones), past 8 the forward takes the NAX
/// sparse kernel (glm5_next `sparseLatentNax`).
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 8;

const scores_source = @embedFile("kernels/dsa_lfuse_scores.metal");
const pv_source = @embedFile("kernels/dsa_lfuse_pv.metal");
var scores_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// The window rows a call of these shapes and dtypes takes (`q` q_abs's [1, 64, n, 512], `l` the latent rows' [1, T,
/// 512], `i` the selection's [1, n, 2051] int32, the attention scale), or null: the stock chain keeps every other call.
pub fn serves(q: []const c_int, l: []const c_int, i: []const c_int, qdt: mlx.mlx_dtype, ldt: mlx.mlx_dtype, idt: mlx.mlx_dtype, scale: ?f32) ?c_int {
    if (scale == null) return null;
    if (q.len != 4 or q[0] != 1 or q[1] != heads or q[3] != latent_w) return null;
    const n = q[2];
    if (n < min_rows or n > max_rows) return null;
    if (l.len != 3 or l[0] != 1 or l[1] < 1 or l[2] != latent_w) return null;
    if (i.len != 3 or i[0] != 1 or i[1] != n or i[2] != slots) return null;
    if ((qdt != .bfloat16 and qdt != .float16) or ldt != qdt or idt != .int32) return null;
    return n;
}

/// `sparseLatentAttention`'s output for a window of one chunk ([B*n, H, 1, R] from `latentAttendFolded`, B 1) in two
/// launches: `q_abs` the forward's absorbed queries [1, 64, n, 512] (the window's rows all of them), `lat2` the latent
/// rows [1, T, 512], `ic` the selection [1, n, 2051] (-1 unused). A new array; null with the switch off, off the GPU or
/// its tensor units, or outside the window `serves` takes.
pub fn attend(s: mlx.mlx_stream, q_abs: Array, lat2: Array, ic: Array, scale: ?f32) !?Array {
    if (!on or !mlx.streamIsGpu(s) or !naxAvailable()) return null;
    const n = serves(mlx.getShape(q_abs), mlx.getShape(lat2), mlx.getShape(ic), mlx.mlx_array_dtype(q_abs), mlx.mlx_array_dtype(lat2), mlx.mlx_array_dtype(ic), scale) orelse return null;
    const dt = mlx.mlx_array_dtype(q_abs);
    const header = try lane_qmm.prefillHeader();
    // launch 1: the masked scores [n, 64, 2051]
    const ks = try kda_prompt.kernelOf(&scores_kernel, "turboexl3_glm53f_dsa_lfuse_scores", &.{ "q", "lat", "idx" }, &.{"d"}, scores_source, header);
    const cfg1 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg1);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &[_]c_int{ n, heads, slots }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg1, @divTrunc(slots + 127, 128) * 256, 1, n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg1, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg1, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "VS", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "VSC", @bitCast(@as(f32, scale.?))));
    var sc: [1]Array = undefined;
    try kda_prompt.applyKernel(s, ks, cfg1, &.{ q_abs, lat2, ic }, &sc);
    defer _ = mlx.mlx_array_free(sc[0]);
    // launch 2: the softmax and the second matmul, [n, 64, 512]
    const kp = try kda_prompt.kernelOf(&pv_kernel, "turboexl3_glm53f_dsa_lfuse_pv", &.{ "sc", "lat", "idx" }, &.{"d"}, pv_source, header);
    const cfg2 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg2);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &[_]c_int{ n, heads, latent_w }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg2, @divTrunc(latent_w, 128) * 256, 1, n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg2, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg2, "T", dt));
    var o: [1]Array = undefined;
    try kda_prompt.applyKernel(s, kp, cfg2, &.{ sc[0], lat2, ic }, &o);
    defer _ = mlx.mlx_array_free(o[0]);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o[0], &[_]c_int{ n, heads, 1, latent_w }, 4, s));
    runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_LFUSE engaged: a {d}-row verify window over {d} selected slots in two launches\n", .{ n, slots });
    }
    return out;
}

test "turboexl3_glm53f stream math: TURBOEXL3_GLM53F_DSA_LFUSE takes GLM-5.3-Flash's verify window only (64 heads, 512-wide latent rows, 2051 slots, 2 to 8 rows, B 1, the scale)" {
    const t = std.testing;
    const q = [_]c_int{ 1, 64, 3, 512 };
    const l = [_]c_int{ 1, 17411, 512 };
    const i = [_]c_int{ 1, 3, 2051 };
    try t.expectEqual(@as(?c_int, 3), serves(&q, &l, &i, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(@as(?c_int, 3), serves(&q, &l, &i, .float16, .float16, .int32, 0.0625));
    try t.expectEqual(@as(?c_int, 2), serves(&.{ 1, 64, 2, 512 }, &l, &.{ 1, 2, 2051 }, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(@as(?c_int, 8), serves(&.{ 1, 64, 8, 512 }, &l, &.{ 1, 8, 2051 }, .bfloat16, .bfloat16, .int32, 0.0625));
    // the stock chain: no scale, 1 or 9 rows, a batch, other heads, latent width or slot count, f32, rows the selection lacks
    const none = @as(?c_int, null);
    try t.expectEqual(none, serves(&q, &l, &i, .bfloat16, .bfloat16, .int32, null));
    try t.expectEqual(none, serves(&.{ 1, 64, 1, 512 }, &l, &.{ 1, 1, 2051 }, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&.{ 1, 64, 9, 512 }, &l, &.{ 1, 9, 2051 }, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&.{ 2, 64, 3, 512 }, &l, &i, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&.{ 1, 32, 3, 512 }, &l, &i, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&.{ 1, 64, 3, 576 }, &.{ 1, 17411, 576 }, &i, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&q, &l, &.{ 1, 3, 2048 }, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&q, &l, &.{ 1, 4, 2051 }, .bfloat16, .bfloat16, .int32, 0.0625));
    try t.expectEqual(none, serves(&q, &l, &i, .float32, .float32, .int32, 0.0625));
    try t.expectEqual(none, serves(&q, &l, &i, .bfloat16, .float16, .int32, 0.0625));
}
