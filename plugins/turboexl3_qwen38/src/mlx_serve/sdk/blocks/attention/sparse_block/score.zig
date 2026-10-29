//! copied_from: src/sdk/blocks/attention/sparse_block/score.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The QSA block score sheet: the fused NAX kernel (`msv_qsa_score`), its layouts and arms, and the composed chain.

const std = @import("std");
const env = @import("../../../../env.zig");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../util/log.zig");
const select = @import("select.zig");
const QsaCfgCache = @import("../../../../mlx/kernel_cache.zig").QsaCfgCache;
const attn256MaxDiff = @import("../attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attn256.zig").attn256RandBf16;
const verifyQmmNaxAvailable = @import("../../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const OneShotBits = @import("one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("one_shot.zig").widthBucket;
const qsaProbeBitEqual = @import("probe.zig").qsaProbeBitEqual;
const qsaProbeCorruptFirst = @import("probe.zig").qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = @import("probe.zig").qsaProbeLcgBf16;
const qsaArraysAllEqual = @import("../../../testing/parity.zig").qsaArraysAllEqual;
const qsaSelectComposedOps = @import("select.zig").qsaSelectComposedOps;
const qsaSelectKernelEnabled = @import("select.zig").qsaSelectKernelEnabled;
const qsaSelectTopBlocks = @import("select.zig").qsaSelectTopBlocks;
const tieBias = @import("select.zig").tieBias;
const verify_qmm_nax_vars = @import("../../linear/verify_qmm_nax.zig");
const Blocks = @import("params.zig").Blocks;
const ops = @import("../../../../mlx/ops.zig");
const Graph = ops.Graph;
const testing = std.testing;

/// The block geometry the score tests run at (Qwen3.8-Flash-Next's).
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };

pub const QsaScoreLayout = enum(u8) { base = 0, h4 = 1 };

pub var qsa_score_probe_flip_override: bool = false;
pub var qsa_score_layout_override: ?QsaScoreLayout = null;
pub var qsa_score_tf32_override: ?bool = null;
var qsa_score_tf32_env: ?bool = null;
var qsa_score_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_score_kernel_builds: usize = 0;

/// Test seam: false keeps the composed chain.
pub var qsa_score_fused_override: ?bool = null;

pub fn qsaScoreFusedEnabled() bool {
    return qsa_score_fused_override orelse true;
}

pub fn qsaScoreFusedActive() bool {
    if (!qsaScoreFusedEnabled()) return false;
    if (!verifyQmmNaxAvailable()) return false;
    if (!qsaScoreTf32Enabled()) return false;
    return qsaScoreFusedProbeOk();
}

var qsa_score_probe_state: std.atomic.Value(u8) = .init(0);

fn qsaScoreFusedProbeOk() bool {
    return qsa_score_probe_state.load(.acquire) == 2;
}

pub fn qsaScoreFusedArm() void {
    if (qsa_score_probe_state.load(.acquire) != 0) return;
    if (!qsaScoreFusedEnabled() or !verifyQmmNaxAvailable() or !qsaScoreTf32Enabled()) return;
    const ok = qsaScoreFusedRunProbe();
    qsa_score_probe_state.store(if (ok) 2 else 1, .release);
}

pub fn qsaScoreFusedActiveFor(batch: c_int, n_idx: c_int, idx_hd: c_int) bool {
    return qsaScoreFusedActive() and batch == 1 and n_idx == 4 and idx_hd == 128;
}

fn qsaScoreTf32ValueOn(raw: []const u8) bool {
    var i: usize = 0;
    while (i < raw.len and std.ascii.isWhitespace(raw[i])) : (i += 1) {}
    if (i >= raw.len) return false;
    var sign: i32 = 1;
    if (raw[i] == '-') {
        sign = -1;
        i += 1;
    } else if (raw[i] == '+') {
        i += 1;
    }
    var val: i64 = 0;
    var any = false;
    while (i < raw.len) : (i += 1) {
        const c = raw[i];
        if (c < '0' or c > '9') break;
        any = true;
        val = val * 10 + @as(i64, c - '0');
    }
    if (!any) return false;
    return (if (sign < 0) -val else val) != 0;
}

fn qsaScoreTf32Enabled() bool {
    if (qsa_score_tf32_override) |v| return v;
    if (qsa_score_tf32_env) |v| return v;
    const v = blk: {
        const raw = env.get(.MLX_ENABLE_TF32) orelse break :blk true;
        break :blk qsaScoreTf32ValueOn(std.mem.sliceTo(raw, 0));
    };
    qsa_score_tf32_env = v;
    return v;
}

pub fn qsaScoreFusedEligible(batch: c_int, n_idx: c_int, idx_hd: c_int, q_dt: mlx.mlx_dtype, pooled_dt: mlx.mlx_dtype) bool {
    return qsaScoreFusedActiveFor(batch, n_idx, idx_hd) and
        q_dt == .bfloat16 and pooled_dt == .bfloat16;
}

pub fn qsaScoreFusedEligibleFrom(q: mlx.mlx_array, pooled: mlx.mlx_array) bool {
    if (q.ctx == null or pooled.ctx == null) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(pooled) != 3) return false;
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    if (qs[2] <= 0 or ps[1] <= 0) return false;
    if (qs[3] != ps[2]) return false;
    return qsaScoreFusedEligible(qs[0], qs[1], qs[3], mlx.mlx_array_dtype(q), mlx.mlx_array_dtype(pooled));
}

pub fn qsaPooledRowContiguous(pooled: mlx.mlx_array) bool {
    if (mlx.mlx_array_ndim(pooled) != 3) return false;
    const ps = mlx.getShape(pooled);
    if (ps[0] != 1 or ps[2] != 128) return false;
    const st = mlx.mlx_array_strides(pooled);
    return st[2] == 1 and st[1] == 128;
}

pub fn qsaScoreLayoutFor(rows: c_int) QsaScoreLayout {
    if (qsa_score_layout_override) |v| return v;
    return if (rows >= 128) .h4 else .base;
}

pub fn qsaScoreNsg(layout: QsaScoreLayout) c_int {
    return if (layout == .h4) 8 else 4;
}

pub fn qsaScoreRowsPerTg(layout: QsaScoreLayout, nsg: c_int) c_int {
    return nsg * (if (layout == .h4) @as(c_int, 16) else 4);
}

pub fn qsaScoreNsh(nslab: u32, ngrp: u32, layout: QsaScoreLayout) u32 {
    const slabs_per: u32 = if (layout == .h4) 32 else 1;
    var nsh: u32 = @max(@as(u32, 1), nslab / slabs_per);
    const min_nsh: u32 = @max(@as(u32, 1), (256 + ngrp - 1) / ngrp);
    if (nsh < min_nsh) nsh = min_nsh;
    if (nsh > nslab) nsh = nslab;
    return nsh;
}

pub const QSA_SCORE_KERNEL_HEADER = @import("../kv_quant_attn.zig").QSA_SCORE_KERNEL_HEADER;

// The lane -> (row, col) map (`fm`/`fn`) is the matmul2d cooperative-fragment layout,
// hand-derived from the SDK; every A, B and C fragment must use it. K stays 8 ascending
// 16-deep `op.run` calls: that is the stock tf32 GEMM's order and the bit-identity contract.
pub const QSA_SCORE_KERNEL_SOURCE = kernelSource("kernels/attention/qsa_score.metal");

const QsaScoreCfgKey = struct { nsg: c_int, rows: c_int, nb: c_int, layout: u8 };
var qsa_score_cfgs = QsaCfgCache(QsaScoreCfgKey, 1){};
var qsa_score_engaged_bits: OneShotBits = .{};

fn getQsaScoreKernel() !mlx.mlx_fast_metal_kernel {
    if (!verifyQmmNaxAvailable()) return error.NaxUnavailable;
    if (qsa_score_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "Q", "POOL" };
    const output_names = [_][*:0]const u8{"OUT"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_score",
        in_vec,
        out_vec,
        QSA_SCORE_KERNEL_SOURCE,
        QSA_SCORE_KERNEL_HEADER,
        // row-contiguous is load-bearing: q is an axis-2 slice whose parent head stride is
        // S*128; the copy is what makes Q_shape[2] the physical row stride the kernel indexes by
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_score_kernel_cached = kernel;
    qsa_score_kernel_builds += 1;
    return kernel;
}

pub fn qsaScoreSheetComposed(s: mlx.mlx_stream, q: mlx.mlx_array, k32t: mlx.mlx_array) !mlx.mlx_array {
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, q32, k32t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, sc, zero, s));
    var scores = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&scores, relu, 1, false, s));
    return scores;
}

pub fn qsaScoreK32tFromPooled(s: mlx.mlx_stream, pooled: mlx.mlx_array) !mlx.mlx_array {
    const ps = mlx.getShape(pooled);
    var k4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k4);
    try mlx.check(mlx.mlx_reshape(&k4, pooled, &[_]c_int{ ps[0], 1, ps[1], ps[2] }, 4, s));
    const perm = [_]c_int{ 0, 1, 3, 2 };
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k4, &perm, 4, s));
    var k32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
    return k32;
}

fn qsaScoreFusedFillConfig(config: mlx.mlx_fast_metal_kernel_config, rows: c_int, nb: c_int, nsg: c_int, ngrp: c_int, nsh: c_int, layout: QsaScoreLayout) !void {
    const o_shape = [_]c_int{ 1, rows, nb };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nsh * nsg * 32, ngrp, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, nsg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGVAL", nsg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "USEH4VAL", if (layout == .h4) 1 else 0));
}

fn qsaScoreFusedDispatch(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    const rows = qs[2];
    const nb = ps[1];
    const layout = qsaScoreLayoutFor(rows);
    const nsg = qsaScoreNsg(layout);
    const r_tg = qsaScoreRowsPerTg(layout, nsg);
    const ngrp: c_int = @divTrunc(rows + r_tg - 1, r_tg);
    const nslab: u32 = @intCast(@divTrunc(nb + 31, 32));
    const nsh: c_int = @intCast(qsaScoreNsh(nslab, @intCast(ngrp), layout));
    const kernel = try getQsaScoreKernel();
    const key = QsaScoreCfgKey{
        .nsg = nsg,
        .rows = rows,
        .nb = nb,
        .layout = @backingInt(layout),
    };
    const score_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_score_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try qsaScoreFusedFillConfig(config, rows, nb, nsg, ngrp, nsh, layout);
        qsa_score_cfgs.put(key, .{config});
        break :blk .{config};
    };
    const inputs_arr = [_]mlx.mlx_array{ q, pooled };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, score_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (qsa_score_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-score] engaged (S={d} nb={d} layout={s} nsg={d} nsh={d})\n",
            .{ rows, nb, if (layout == .h4) "h4" else "base", nsg, nsh },
        );
    }
    return out;
}

fn qsaScoreFusedProbeLayout(s: mlx.mlx_stream, rows: c_int, nb: c_int) bool {
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 4, rows, 128 }, 0x51C04E11) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const pooled = qsaProbeLcgBf16(s, &[_]c_int{ 1, nb, 128 }, 0xA5A55A5A) orelse return false;
    defer _ = mlx.mlx_array_free(pooled);
    const k32t = qsaScoreK32tFromPooled(s, pooled) catch return false;
    defer _ = mlx.mlx_array_free(k32t);
    const want = qsaScoreSheetComposed(s, q, k32t) catch return false;
    defer _ = mlx.mlx_array_free(want);
    var got = qsaScoreFusedDispatch(s, q, pooled) catch return false;
    defer _ = mlx.mlx_array_free(got);
    if (qsa_score_probe_flip_override) {
        const flipped = qsaProbeCorruptFirst(s, got) orelse return false;
        _ = mlx.mlx_array_free(got);
        got = flipped;
    }
    if (mlx.mlx_array_eval(got) != 0 or mlx.mlx_array_eval(want) != 0) return false;
    if (!qsaProbeBitEqual(s, got, want)) {
        log.info("[qsa-score] disabled: probe mismatch ({s})\n", .{if (rows >= 128) "h4" else "base"});
        return false;
    }
    return true;
}

fn qsaScoreFusedRunProbe() bool {
    if (mlx.noGpuBackend()) {
        log.info("[qsa-score] disabled: no gpu\n", .{});
        return false;
    }
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) {
        log.info("[qsa-score] disabled: not gpu stream\n", .{});
        return false;
    }
    _ = getQsaScoreKernel() catch |e| {
        log.info("[qsa-score] disabled: {s}\n", .{@errorName(e)});
        return false;
    };
    if (!qsaScoreFusedProbeLayout(s, 16, 700)) return false;
    if (!qsaScoreFusedProbeLayout(s, 128, 700)) return false;
    return true;
}

pub fn qsaScoreFused(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !?mlx.mlx_array {
    qsaScoreFusedArm();
    if (!qsaScoreFusedEligibleFrom(q, pooled)) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (!qsaPooledRowContiguous(pooled)) return null;
    return try qsaScoreFusedDispatch(s, q, pooled);
}

pub fn qsaScoreSheet(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array, k32t: mlx.mlx_array, fused: bool) !mlx.mlx_array {
    if (fused) {
        return (try qsaScoreFused(s, q, pooled)) orelse error.QsaScoreFusedDeclined;
    }
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    return qsaScoreSheetComposed(s, q, k32t);
}

/// Bound on the per-chunk [n_idx, rows, nb] f32 indexer score sheet: the
/// prefill selection runs in row chunks so 4096 rows x 64k blocks x 4 heads
/// (4.3 GB at 256k) never materializes at once.
///
/// At long context it decides the prefill's row chunk: at kv 383k (nb ~95.8k) 256 MB leaves room
/// for ~171 rows, so a 4096-token chunk is re-split into 24 indexer passes. The forward and the
/// memory bill both read it
/// through `qsaScoreRowsPerChunk`, so a wider sheet is billed, never just spent.
pub const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT: u64 = 256;

/// Upper bound on the lever: a fat-fingered value must become a named 400, not a Metal OOM.
/// Test seam: the sheet budget in MB.
pub var qsa_score_sheet_mb_override: ?u64 = null;

/// The live sheet budget in bytes.
pub fn qsaScoreSheetBudget() u64 {
    return (qsa_score_sheet_mb_override orelse QSA_SCORE_SHEET_BUDGET_MB_DEFAULT) << 20;
}

pub fn qsaScoreRowsPerChunkFor(per_row: u64, rows: u64) u64 {
    if (per_row == 0) return rows;
    return @max(16, @min(rows, qsaScoreSheetBudget() / per_row));
}

pub fn qsaScoreRowsPerChunkFused(n_idx: u64, nb: u64, rows: u64, fused: bool) u64 {
    const per_row: u64 = if (fused) nb * 4 else n_idx * nb * 4;
    return qsaScoreRowsPerChunkFor(per_row, rows);
}

/// Prefill transient of the block-gathered QSA path for ONE live layer.
/// Composed: matmul out + relu per head + scores + argpartition temps + selection.
/// Fused: the [rows,nb] f32 sheet (×4 when the select kernel is off) + selection.
pub fn qsaPrefillTransientBytes(n_idx: u64, fwd: u64, kv: u64, ratio: u64, idx_hd: u64) u64 {
    if (ratio == 0) return 0;
    const nb = kv / ratio;
    const fused = qsaScoreFusedActiveFor(1, @intCast(n_idx), @intCast(idx_hd));
    const rows = qsaScoreRowsPerChunkFused(n_idx, nb, fwd, fused);
    if (fused) {
        const sheet = rows * nb * 4;
        const select_mult: u64 = if (qsaSelectKernelEnabled()) 1 else 4;
        return sheet * select_mult + fwd * @min(nb, 512) * 4;
    }
    return rows * nb * 4 * (n_idx * 2 + 4) + fwd * @min(nb, 512) * 4;
}

// ── tests ──
