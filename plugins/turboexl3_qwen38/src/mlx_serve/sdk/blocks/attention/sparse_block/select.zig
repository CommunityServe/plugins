//! copied_from: src/sdk/blocks/attention/sparse_block/select.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Exact top-k block selection (`msv_qsa_select`: radix select, single, sliced and merged setups) and its 16-way split
//! at decode.

const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../util/log.zig");
const QsaCfgCache = @import("../../../../mlx/kernel_cache.zig").QsaCfgCache;
const attn256MaxDiff = @import("../attn256.zig").attn256MaxDiff;
const OneShotBits = @import("one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("one_shot.zig").widthBucket;
const Blocks = @import("params.zig").Blocks;
const ops = @import("../../../../mlx/ops.zig");
const Graph = ops.Graph;
const testing = std.testing;

/// The block geometry the select tests run at (Qwen3.8-Flash-Next's).
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };

// ── Fused QSA top-k block select (msv_qsa_select) ──
//
// The composed arm is ~10 dependent dispatches per attention layer (~0.31 ms/layer at kv
// 62.7k); this kernel is one dispatch: one threadgroup per query row, an MSD radix select
// over the f32 score's monotone ordinal, then an index-ordered compaction. Semantics are
// `qsaChunkSelect`'s: top-K visible blocks per row, ties to the LOWER index (torch.topk),
// emitted ascending, INT_MAX past the row's count. The tie rule is exact where the composed
// arm's `b * 1e-7` bias only approximates it.
pub const QSA_SELECT_KERNEL_HEADER = kernelSource("kernels/attention/qsa_select_header.metal");

/// One threadgroup per query row. 1. MSD radix select over the 32-bit ordinal (the DIGITS
/// template picks 11/11/10-bit digits at prefill widths or four 8-bit digits at decode widths,
/// where each level's fixed cost, not the walk, is the bill) counting only visible blocks,
/// stopping early when a bin's count is exactly what is still needed. 2. Compaction: selected
/// iff `ord > T`, or `ord == T` with rank below `need_eq`; the output slot is the count of
/// selected elements with a smaller index (a simd prefix scan per chunk), so the output is
/// born sorted. 3. Rows with `bounds[row] <= K` are `0 .. bounds-1` then INT_MAX.
/// Threadgroup memory ~8.3 KiB at 2048 bins, ~1.3 KiB at 256.
const QSA_SELECT_SHARED = kernelSource("kernels/attention/qsa_select_shared.metal");

// One radix-select body composed into three kernels (single row, row slice, merge) so a
// fix to the digit walk or the tie rule lands once; the kernels differ only in setup.
const QSA_SELECT_RADIX_BODY = kernelSource("kernels/attention/qsa_select_radix_body.metal");

const QSA_SELECT_SINGLE_SETUP = kernelSource("kernels/attention/qsa_select_single_setup.metal");

const QSA_SELECT_SLICE_SETUP = kernelSource("kernels/attention/qsa_select_slice_setup.metal");

// Candidates arrive ascending by ORIGINAL index with SENTINEL slots skipped, never sorted
// in; that is what resolves a cross-slice tie at the K-th ordinal to the lowest index. Every
// simd reduction below runs in a uniform base-stepped loop: a divergent simd_sum is undefined.
const QSA_SELECT_MERGE_SETUP = kernelSource("kernels/attention/qsa_select_merge_setup.metal");

pub const QSA_SELECT_KERNEL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SINGLE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_LOCAL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SLICE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_MERGE_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_MERGE_SETUP, QSA_SELECT_RADIX_BODY });

/// Threads per row-threadgroup.
pub const QSA_SELECT_TG_DEFAULT: c_int = 1024;
pub var qsa_select_tg_override: ?c_int = null;

// A row's threadgroup is a per-shape choice: the wide passes of a prefill chunk run more
// rows concurrently at 256 threads, a narrow pass wants 512, and the decode widths keep
// 1024 (the split kernels size their own slices). Outputs are identical at every width.
pub fn qsaSelectTgFor(rows: c_int) c_int {
    if (qsa_select_tg_override) |forced| return forced;
    if (rows >= 256) return 256;
    if (rows >= 64) return 512;
    return QSA_SELECT_TG_DEFAULT;
}

var qsa_select_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select",
        in_vec,
        out_vec,
        QSA_SELECT_KERNEL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true, // the row walk indexes linearly; let mlx materialize a view
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_kernel_cached = kernel;
    return kernel;
}

/// Keyed on what the config reads: output `[1, rows, kb]`, grid `(tg, rows, 1)`, templates
/// TGS and K. `nb` is a runtime shape (it advances every `ratio` decode tokens).
const QsaSelectCfgKey = struct { rows: c_int, k: c_int, tg: c_int, digits: c_int };
var qsa_select_cfgs = QsaCfgCache(QsaSelectCfgKey, 1){};

var qsa_select_engaged_bits: OneShotBits = .{};

pub var qsa_select_kernel_override: ?bool = null;

/// Fused QSA block select; the test seam's false runs the argpartition chain.
pub fn qsaSelectKernelEnabled() bool {
    return qsa_select_kernel_override orelse true;
}

/// Fused replacement for the composed selection tail: f32 `[1, rows, nb]` scores + int32
/// `[rows]` bounds -> sorted int32 `[1, rows, kb]` selection. Null = declined.
pub fn qsaSelectTopBlocks(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
) !?mlx.mlx_array {
    if (!qsaSelectKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (scores.ctx == null or bounds.ctx == null) return null;
    if (mlx.mlx_array_dtype(scores) != .float32 or mlx.mlx_array_dtype(bounds) != .int32) return null;
    if (mlx.mlx_array_ndim(scores) != 3 or mlx.mlx_array_ndim(bounds) != 1) return null;
    const ssh = mlx.getShape(scores);
    const bsh = mlx.getShape(bounds);
    if (ssh[0] != 1) return null;
    const rows = ssh[1];
    const nb = ssh[2];
    if (rows <= 0 or nb <= 0 or bsh[0] != rows) return null;
    if (kb <= 0 or kb > nb) return null;

    qsa_select_used_split = false;
    if (qsaSelectSplitEnabled() and qsaSelectSplitServes(rows, nb, kb)) {
        if (try qsaSelectTopBlocksSplit(s, scores, bounds, kb, rows, nb)) |picks| {
            qsa_select_used_split = true;
            return picks;
        }
    }

    const kernel = getQsaSelectKernel() catch return null;
    const tg = qsaSelectTgFor(rows);
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectCfgKey{ .rows = rows, .k = kb, .tg = tg, .digits = digits };
    const sel_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_select_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TGS", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DIGITS", digits));
        qsa_select_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const inputs_arr = [_]mlx.mlx_array{ scores, bounds };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, sel_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (qsa_select_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (S={d} nb={d} k={d} tg={d})\n",
            .{ rows, nb, kb, tg },
        );
    }
    return out;
}

// ── The composed selection arm (the kernel's reference) ──
// Free functions so the parity test can run both arms on the same scores.

/// argpartition of the visibility-masked scores; a tiny index-ascending bias reproduces
/// torch.topk's lower-index-wins. A null-ctx `vis3` skips the mask op (all visible).
pub fn qsaTopBlocksOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, nb: c_int, block_topk: c_int) !mlx.mlx_array {
    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_subtract(&biased, scores, tie_bias, s));
    if (vis3.ctx == null) {
        var part_all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_argpartition_axis(&part_all, biased, nb - block_topk, -1, s));
        return part_all;
    }
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, vis3, biased, neg_inf, s));
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argpartition_axis(&part, masked, nb - block_topk, -1, s));
    return part;
}

/// Sort a `[1, rows, kb]` pick list ascending, mapping rejected picks to INT_MAX first.
pub fn qsaSortWithInvisible(s: mlx.mlx_stream, top_idx: mlx.mlx_array, vis3: mlx.mlx_array, all_vis: bool) !mlx.mlx_array {
    var sorted = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sorted);
    if (all_vis) {
        try mlx.check(mlx.mlx_sort_axis(&sorted, top_idx, -1, s));
        return sorted;
    }
    const int_max = mlx.mlx_array_new_int(std.math.maxInt(i32));
    defer _ = mlx.mlx_array_free(int_max);
    var picked_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked_vis);
    try mlx.check(mlx.mlx_take_along_axis(&picked_vis, vis3, top_idx, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_where(&sel, picked_vis, top_idx, int_max, s));
    try mlx.check(mlx.mlx_sort_axis(&sorted, sel, -1, s));
    return sorted;
}

/// The composed arm the fused kernel replaces, end to end.
pub fn qsaSelectComposedOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, rows: c_int, nb: c_int, kb: c_int, block_topk: c_int, all_vis: bool) !mlx.mlx_array {
    const strides3 = [_]c_int{ 1, 1, 1 };
    const part = try qsaTopBlocksOps(s, scores, vis3, tie_bias, nb, block_topk);
    defer _ = mlx.mlx_array_free(part);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, part, &[_]c_int{ 0, 0, nb - kb }, 3, &[_]c_int{ 1, rows, nb }, 3, &strides3, 3, s));
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_astype(&top_idx, tail, .int32, s));
    return qsaSortWithInvisible(s, top_idx, vis3, all_vis);
}

// ── Picks under any `Blocks` ──

/// f32 `[nb]` `b * 1e-7`: the index-ascending bias the composed chain breaks ties with (torch.topk's lower index).
pub fn tieBias(s: mlx.mlx_stream, nb: c_int) !mlx.mlx_array {
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0, @floatFromInt(nb), 1.0, .float32, s));
    const eps = mlx.mlx_array_new_float(1e-7);
    defer _ = mlx.mlx_array_free(eps);
    var bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&bias, idx, eps, s));
    return bias;
}

// One threadgroup per row is latency-bound at decode widths (0.72 ms at 215k blocks; ~20x
// under the bandwidth floor), so rows split over G slices, each running the exact select,
// and a second dispatch selects over the G*K candidates. Exact because the order is total
// (ordinal, then lowest original index): every global top-K element is in its slice's
// top-K. The floor is the crossover of the marginal in-graph cost (probe: the split pair
// beats the single kernel from ~24k blocks once the compaction scan and 8-bit digits landed;
// live, +4% at 40k blocks and +18% serial at 200k); 2*G*K <= nb (16k blocks at G=16) keeps
// the candidate list smaller than the row it replaces and is the design's hard minimum.
pub const QSA_SELECT_SPLIT_MIN_NB: c_int = 24576;
pub const QSA_SELECT_SPLIT_MAX_ROWS: c_int = 15;
pub const QSA_SELECT_SPLIT_G: c_int = 16;

pub var qsa_select_digits_override: ?c_int = null;

pub fn qsaSelectDigitBits(rows: c_int) c_int {
    if (qsa_select_digits_override) |v| return v;
    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) return 8;
    return 11;
}

pub var qsa_select_split_override: ?bool = null;
pub var qsa_select_used_split: bool = false;

pub fn qsaSelectSplitEnabled() bool {
    return qsa_select_split_override orelse true;
}

pub fn qsaSelectSplitServes(rows: c_int, nb: c_int, kb: c_int) bool {
    if (rows < 1 or rows > QSA_SELECT_SPLIT_MAX_ROWS) return false;
    if (nb < QSA_SELECT_SPLIT_MIN_NB) return false;
    if (kb <= 0 or kb > nb) return false;
    const span = std.math.mul(c_int, kb, QSA_SELECT_SPLIT_G) catch return false;
    const twice = std.math.mul(c_int, span, 2) catch return false;
    return twice <= nb;
}

var qsa_select_split_local_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_select_split_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectSplitLocalKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_local_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_local",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_LOCAL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_local_kernel_cached = kernel;
    return kernel;
}

fn getQsaSelectSplitMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "local_ids" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_merge",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_MERGE_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_merge_kernel_cached = kernel;
    return kernel;
}

const QsaSelectSplitCfgKey = struct { rows: c_int, k: c_int, slice_tg: c_int, merge_tg: c_int, digits: c_int };
var qsa_select_split_cfgs = QsaCfgCache(QsaSelectSplitCfgKey, 2){};

var qsa_select_split_engaged_bits: OneShotBits = .{};

fn qsaSelectTopBlocksSplit(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
    rows: c_int,
    nb: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const g = QSA_SELECT_SPLIT_G;
    const loc_count = std.math.mul(c_int, kb, g) catch return null;
    const local_kernel = getQsaSelectSplitLocalKernel() catch return null;
    const merge_kernel = getQsaSelectSplitMergeKernel() catch return null;
    const slice_tg: c_int = qsa_select_tg_override orelse 512;
    const merge_tg: c_int = qsa_select_tg_override orelse QSA_SELECT_TG_DEFAULT;
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectSplitCfgKey{ .rows = rows, .k = kb, .slice_tg = slice_tg, .merge_tg = merge_tg, .digits = digits };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_select_split_cfgs.get(key) orelse blk: {
        const local_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(local_cfg);
        const local_shape = [_]c_int{ 1, rows, loc_count };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(local_cfg, &local_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(local_cfg, slice_tg * g, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(local_cfg, slice_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "TGS", slice_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "DIGITS", digits));
        const merge_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge_cfg);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge_cfg, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge_cfg, merge_tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge_cfg, merge_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "TGS", merge_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "DIGITS", digits));
        qsa_select_split_cfgs.put(key, .{ local_cfg, merge_cfg });
        break :blk .{ local_cfg, merge_cfg };
    };

    const local_in = [_]mlx.mlx_array{ scores, bounds };
    const local_in_vec = mlx.mlx_vector_array_new_data(&local_in, local_in.len);
    defer _ = mlx.mlx_vector_array_free(local_in_vec);
    var local_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(local_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&local_out_vec, local_kernel, local_in_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(local_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var local_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_ids);
    try mlx.check(mlx.mlx_vector_array_get(&local_ids, local_out_vec, 0));

    const merge_in = [_]mlx.mlx_array{ scores, local_ids };
    const merge_in_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_in_vec);
    var merge_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(merge_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&merge_out_vec, merge_kernel, merge_in_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(merge_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, merge_out_vec, 0));
    if (qsa_select_split_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (split G={d} S={d} nb={d} k={d} tg={d})\n",
            .{ g, rows, nb, kb, slice_tg },
        );
    }
    return out;
}

// ── tests ──

// ── Fused QSA block select (msv_qsa_select) ──
