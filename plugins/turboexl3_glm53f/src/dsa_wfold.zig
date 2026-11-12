//! `TURBOEXL3_GLM53F_DSA_WFOLD=1` (opt-in, exact): a decode, verify or draft window's DSA selection reads the indexer's
//! head-weight projection as it is, through patches/turboexl3-model.patch. The stock window
//! (table_forward `dsaAttn`, glm5_next `selectRows`) casts the projection to f32 (an astype launch), multiplies it by
//! 1/sqrt(index heads) (a launch) and then by 1/sqrt(index head dim) (a launch) before the scores launch reads it. Here
//! the scores launch is DSA_VSEL's text (src/dsa_verify.zig: the one-query scores text with the row in the grid) with
//! its head weight computed where it reads it, the stock f32 multiplies in the stock order on the same f32 values; the
//! top-k and its expansion are DSA_VSEL's. The same weights, so the same scores and tokens: 3 launches fewer a DSA layer,
//! and the one-row window (a draft step) in DSA_VSEL's two launches too, which compute what the decode's own two do.
//!
//! GLM-5.3-Flash's dims only: 32 index heads of 128, pools of 4, 2048 selected tokens, B 1, 1 to 16 rows, more pools
//! than it selects (fewer select every pool: the stock path, which reads no weight).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kda_prompt = @import("kda_prompt.zig");
const dsa_verify = @import("dsa_verify.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_WFOLD=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// Windows selected with the weights folded in, since the load.
pub var runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_WFOLD=1: a decode, verify or draft window's DSA selection reads the indexer's head weights as projected, the f32 cast and both scale multiplies done in the scores launch (3 launches fewer a layer, the same tokens)\n";

/// GLM-5.3-Flash's indexer.
pub const heads: c_int = 32;
pub const head_dim: c_int = 128;
pub const kpool: c_int = 4;
pub const topk: c_int = 2048;
/// The rows a window has at most (glm5_next `HC_PRE_MAX_ROWS`).
pub const max_rows: c_int = 16;
/// The stock scores launch's simdgroups and pools a simdgroup (glm5_next `DEC_SCORE_SGS`, `DEC_SCORE_PPS`).
pub const sgs: c_int = 8;
pub const pps: c_int = 4;

/// DSA_VSEL's scores text with the head weight computed from the projection `wp` (WT) as the stock chain does: f32,
/// times the heads' scale, times the head dim's scale (each an f32 passed as its bits).
const scores_source = dsa_verify.edited(dsa_verify.scores_source, .{
    .{ "const device float* w_r = w + size_t(vrow) * 32;", "const device WT* wp_r = wp + size_t(vrow) * 32;", 1 },
    .{ "w_r[lane]", "((float(wp_r[lane]) * as_type<float>(HSB)) * as_type<float>(SCB))", 1 },
});
var scores_kernel: ?mlx.mlx_fast_metal_kernel = null;
var topk_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// The heads' scale table_forward `dsaAttn` multiplies the projection by, and the head dim's that glm5_next `selectRows`
/// multiplies the result by, each as the stock code computes it in f32.
pub fn headsScale(h: c_int) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
}
pub fn dimScale(d: c_int) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
}

/// One window's selection: its rows, the first row's position, the pools and how many it selects.
pub const Plan = struct { rows: c_int, q_start: c_int, p: c_int, select_k: c_int };

/// The plan for `q` the index queries [1, S, 32, 128], `w` the head-weight projection [1, S, 32], `c` the indexer cache
/// [1, T, 256] (the window's rows its last S), or null where the stock path keeps it. MLX-free.
pub fn plan(q: []const c_int, w: []const c_int, c: []const c_int, qdt: mlx.mlx_dtype, wdt: mlx.mlx_dtype, cdt: mlx.mlx_dtype) ?Plan {
    if (q.len != 4 or q[0] != 1 or q[1] < 1 or q[1] > max_rows or q[2] != heads or q[3] != head_dim) return null;
    if (w.len != 3 or w[0] != 1 or w[1] != q[1] or w[2] != heads) return null;
    if (c.len != 3 or c[0] != 1 or c[1] < q[1] or c[2] != 2 * head_dim) return null;
    if ((cdt != .bfloat16 and cdt != .float16) or qdt != cdt or (wdt != .bfloat16 and wdt != .float16)) return null;
    const p = @divTrunc(c[1], kpool);
    const select_k = @min(@divTrunc(topk, kpool), p);
    if (select_k < 1 or select_k >= p) return null;
    return .{ .rows = q[1], .q_start = c[1] - q[1], .p = p, .select_k = select_k };
}

/// glm5_next `selectTokens` for a window of 1 to 16 rows from the head-weight projection `wp` (bf16 or f16, the
/// stock chain's input to its cast): int32 [1, S, 2051], a new array; null with the switch off, off the GPU, or where
/// `plan` declines (the caller runs the stock chain).
pub fn select(s: mlx.mlx_stream, q_idx: Array, wp: Array, idx_cache: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const pl = plan(mlx.getShape(q_idx), mlx.getShape(wp), mlx.getShape(idx_cache), mlx.mlx_array_dtype(q_idx), mlx.mlx_array_dtype(wp), mlx.mlx_array_dtype(idx_cache)) orelse return null;
    const dims = mlx.mlx_array_new_data(&[_]i32{ pl.q_start, pl.p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    // the scores [S, P]: DSA_VSEL's grid
    const sk = try kda_prompt.kernelOf(&scores_kernel, "turboexl3_glm53f_dsa_wfold_scores", &.{ "cache", "q", "wp", "dims" }, &.{"out"}, scores_source, "");
    const scfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(scfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(scfg, &[_]c_int{ pl.rows, pl.p }, 2, .float32));
    const per_tg = sgs * pps;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(scfg, sgs * 32, @divTrunc(pl.p + per_tg - 1, per_tg), pl.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(scfg, sgs * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(scfg, "T", mlx.mlx_array_dtype(idx_cache)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(scfg, "WT", mlx.mlx_array_dtype(wp)));
    inline for (.{ .{ "DH", head_dim }, .{ "KPOOL", kpool }, .{ "SGS", sgs }, .{ "PPS", pps } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(scfg, kv[0], kv[1]));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(scfg, "HSB", @bitCast(headsScale(heads))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(scfg, "SCB", @bitCast(dimScale(head_dim))));
    var sc: [1]Array = undefined;
    try kda_prompt.applyKernel(s, sk, scfg, &.{ idx_cache, q_idx, wp, dims }, &sc);
    defer _ = mlx.mlx_array_free(sc[0]);
    // the top-k and its expansion: DSA_VSEL's text
    const tk = try kda_prompt.kernelOf(&topk_kernel, "turboexl3_glm53f_dsa_wfold_topk", &.{ "scores", "dims" }, &.{"out"}, dsa_verify.topk_source, dsa_verify.topk_header);
    const out_w = topk + kpool - 1;
    const tcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(tcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(tcfg, &[_]c_int{ 1, pl.rows, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(tcfg, dsa_verify.topk_tg, pl.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(tcfg, dsa_verify.topk_tg, 1, 1));
    inline for (.{ .{ "SEL_K", pl.select_k }, .{ "KPOOL", kpool }, .{ "TOPK_W", topk }, .{ "TAIL_W", kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(tcfg, kv[0], kv[1]));
    }
    var o: [1]Array = undefined;
    try kda_prompt.applyKernel(s, tk, tcfg, &.{ sc[0], dims }, &o);
    runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_WFOLD engaged: a {d}-row window over {d} pools, {d} selected, its head weights folded into the scores launch\n", .{ pl.rows, pl.p, pl.select_k });
    }
    return o[0];
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_DSA_WFOLD plans GLM-5.3-Flash's 1- to 16-row windows past the selection budget (the first row at the cache's length less the rows) and leaves prompts, other indexers, dtypes, batches and every-pool contexts to the stock path" {
    const t = std.testing;
    const bf = mlx.mlx_dtype.bfloat16;
    for (1..17) |r| {
        const s: c_int = @intCast(r);
        const pl = plan(&.{ 1, s, 32, 128 }, &.{ 1, s, 32 }, &.{ 1, 16387 + s, 256 }, bf, bf, bf) orelse return error.TestUnexpectedResult;
        try t.expectEqual(s, pl.rows);
        try t.expectEqual(@as(c_int, 16387), pl.q_start);
        try t.expectEqual(@divTrunc(16387 + s, 4), pl.p);
        try t.expectEqual(@as(c_int, 512), pl.select_k);
    }
    // f16, and a bf16 cache with f16 weights
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 256 }, .float16, .float16, .float16) != null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 256 }, bf, .float16, bf) != null);
    // every pool selected (2051 keys and fewer): the stock broadcast, which reads no weight
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 2051, 256 }, bf, bf, bf) == null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 2052, 256 }, bf, bf, bf) != null);
    // a prompt chunk, a batch, other indexers, f32 weights or queries, a mismatched cache dtype
    try t.expect(plan(&.{ 1, 17, 32, 128 }, &.{ 1, 17, 32 }, &.{ 1, 16404, 256 }, bf, bf, bf) == null);
    try t.expect(plan(&.{ 2, 3, 32, 128 }, &.{ 2, 3, 32 }, &.{ 2, 16390, 256 }, bf, bf, bf) == null);
    try t.expect(plan(&.{ 1, 3, 64, 128 }, &.{ 1, 3, 64 }, &.{ 1, 16390, 256 }, bf, bf, bf) == null);
    try t.expect(plan(&.{ 1, 3, 32, 64 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 128 }, bf, bf, bf) == null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 256 }, bf, .float32, bf) == null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 256 }, .float32, bf, .float32) == null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 3, 32 }, &.{ 1, 16390, 256 }, .float16, bf, bf) == null);
    try t.expect(plan(&.{ 1, 3, 32, 128 }, &.{ 1, 2, 32 }, &.{ 1, 16390, 256 }, bf, bf, bf) == null);
}
