//! copied_from: src/plugins/qwen4_exp/qsa/sparse_attn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA sparse attention over the selected blocks (`qsaSparseAttn`): the tile, row and slot kernels and their merge.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../sdk/util/log.zig");
const qsa_decode = @import("../../../sdk/blocks/attention/sparse_block/decode.zig");
const qsa_decode_cuda = @import("../../../sdk/blocks/attention/sparse_block/decode_cuda.zig");
const arms_mod = @import("arms.zig");
const transformer = @import("../../../engines/mlx/transformer.zig");
const ProfClock = @import("../../../engines/mlx/diag/profile.zig").ProfClock;
const env = @import("../../../env.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256Cosine = @import("../../../sdk/blocks/attention/attn256.zig").attn256Cosine;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const attn256UlpBar = @import("../../../sdk/blocks/attention/attn256.zig").attn256UlpBar;
const fusedSdpa256Masked = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Masked;
const qsaFusedEnabled = @import("../../../sdk/blocks/attention/attn256.zig").qsaFusedEnabled;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const ShapeKey = @import("../../../sdk/blocks/moe/router_kernel.zig").ShapeKey;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const standinRef = @import("../../../sdk/diag/standin.zig").standinRef;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const OneShotBits = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").OneShotBits;
const QsaArmCounts = @import("arms.zig").QsaArmCounts;
const qsaWidthBucket = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").widthBucket;
const qsaDecodeGatherAttn = @import("decode_gather.zig").qsaDecodeGatherAttn;
const ATTN_QSA256_KERNEL_HEADER = @import("gather.zig").ATTN_QSA256_KERNEL_HEADER;
const gatherQsa256 = @import("gather.zig").gatherQsa256;
const qsaDecodeGatherEnabled = @import("gather.zig").qsaDecodeGatherEnabled;
const qsaGatherBk = @import("gather.zig").qsaGatherBk;
const qsaGatherEnabled = @import("gather.zig").qsaGatherEnabled;
const qsaGatherMinKv = @import("gather.zig").qsaGatherMinKv;
const qsaHistoryShareEnabled = @import("gather.zig").qsaHistoryShareEnabled;
const qsaPrefillGatherMinKv = @import("gather.zig").qsaPrefillGatherMinKv;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaMaskArm = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskArm;
const qsaScoreSheetBudget = @import("../../../sdk/blocks/attention/sparse_block/score.zig").qsaScoreSheetBudget;
const qsaSelectKernelEnabled = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectKernelEnabled;
const qsaSelectSplitEnabled = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectSplitEnabled;
const QSA_VERIFY_GATHER_MIN_KV_DENSE = @import("verify_gather.zig").QSA_VERIFY_GATHER_MIN_KV_DENSE;
const qsaVerifyBlocksHost = @import("verify_gather.zig").qsaVerifyBlocksHost;
const qsaVerifyGatherAttn = @import("verify_gather.zig").qsaVerifyGatherAttn;
const qsaVerifyGatherEnabled = @import("verify_gather.zig").qsaVerifyGatherEnabled;
const qsaVerifyGatherMinKvFor = @import("verify_gather.zig").qsaVerifyGatherMinKvFor;

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──
//
// At 2 <= S < 16 the QSA path spent ~50 dependent ops per layer and attended a fixed union
// of the S rows' selections under a mask (6x the MACs at S=6, plus a 12.6 MB slab). This is
// one dispatch: `msv_attn_qsa256` verbatim with the device K/V reads replaced by an in-kernel
// affine unpack (`qkv_attn_dec`'s dense packing). Each row sees its own blocks then its own
// ragged tail, so causality is automatic and the INT_MAX sentinels are never read. A bf16
// cache runs the same split pass with `msv_attn_qsa256`'s plain uint4 loads: the source is
// built from shared pieces, and only the prologue, the K/V loads and their bases differ per variant.
/// Shared by every variant of the split pass: tile constants.
const QSA_ATTN_CONSTS = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_consts.metal");

/// Quantized variant: unpack constants (BITS is its template).
const QSA_ATTN_Q_CONSTS = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_q_consts.metal");

/// Shared: the row's geometry and grid position. Both variants name their K/V inputs
/// `kq`/`vq`, so `kq_shape` is the key count in either.
const QSA_ATTN_ROW_GRID = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_row_grid.metal");

/// The shared fragments read strides only through `qS`/`blS`/`kqS`/..., which each kernel's prologue
/// defines from its own inputs (these two copy the bound strides; the slot kernel derives them from `meta`).
const QSA_ATTN_Q_STRIDES = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_q_strides.metal");

const QSA_ATTN_D_STRIDES = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_strides.metal");

/// Shared: the row's visible range, its split and its block list.
const QSA_ATTN_ROW = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_row.metal");

/// Quantized variant: per-head bases of the six packed arrays.
const QSA_ATTN_Q_BASES = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_q_bases.metal");

/// Dense variant: per-head bases of the bf16 K and V, and whether each can take uint4
/// loads. A uint4 needs a 16-byte address: the bound pointer (which carries the view's data
/// offset, invisible to the host) and every row start. Anything else — an offset slice, a
/// padded or non-unit innermost layout — reads element by element through its strides.
const QSA_ATTN_D_BASES = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_bases.metal");

/// Shared: staging, Q fragments, and the head of the key-tile loop.
const QSA_ATTN_TILE_OPEN = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_tile_open.metal");

/// Quantized variant: K tile, unpacked in-kernel.
const QSA_ATTN_Q_KLOAD = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_q_kload.metal");

/// Dense variant: K tile, read straight from the bf16 cache.
const QSA_ATTN_D_KLOAD = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_kload.metal");

/// Shared: scores and the ragged-tile mask.
const QSA_ATTN_SCORES = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_scores.metal");

/// Quantized variant: V tile, unpacked in-kernel.
const QSA_ATTN_Q_VLOAD = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_q_vload.metal");

/// Dense variant: V tile, read straight from the bf16 cache.
const QSA_ATTN_D_VLOAD = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_vload.metal");

/// Shared: online softmax, P@V, and the partials for the merge.
const QSA_ATTN_TILE_CLOSE = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_tile_close.metal");

pub const QSA_ATTN_Q_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_Q_CONSTS ++ QSA_ATTN_ROW_GRID ++ QSA_ATTN_Q_STRIDES ++ QSA_ATTN_ROW ++
    QSA_ATTN_Q_BASES ++ QSA_ATTN_TILE_OPEN ++ QSA_ATTN_Q_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_Q_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// The dense (bf16 KV) variant: the same split pass with plain device loads, so a dense
/// cache keeps the verify kernel's split-K grid (never `gatherQsa256`'s prefill grid).
pub const QSA_ATTN_D_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_ROW_GRID ++ QSA_ATTN_D_STRIDES ++ QSA_ATTN_ROW ++
    QSA_ATTN_D_BASES ++ QSA_ATTN_TILE_OPEN ++ QSA_ATTN_D_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_D_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// The slot-batched quantized split pass: slot `bb` on grid z, `meta` = {Hq, Hk, KB, then kv length and
/// parent capacity per slot}.
const QSA_ATTN_SLOTS_GRID = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_slots_grid.metal");

pub const QSA_ATTN_SLOTS_SOURCE = QSA_ATTN_CONSTS ++ QSA_ATTN_Q_CONSTS ++ QSA_ATTN_SLOTS_GRID ++ QSA_ATTN_ROW ++ QSA_ATTN_Q_BASES ++
    QSA_ATTN_TILE_OPEN ++ QSA_ATTN_Q_KLOAD ++ QSA_ATTN_SCORES ++ QSA_ATTN_Q_VLOAD ++ QSA_ATTN_TILE_CLOSE;

/// msv_attn_qsa256_qmerge: the split-K reduction, one threadgroup per (q head, query row).
/// Standard flash-decoding merge in the log2 domain; `l_j == 0` marks an empty split and is
/// skipped. NSPLIT and qL ride `pacc_shape`, so the config depends on nothing per token.
pub const QSA_ATTN_MERGE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_merge.metal");

/// Measured split count (M5 Max, qwen4_exp, kv 62k, 8-bit KV, 12 counterbalanced boots):
/// every count 8..64 beats the union gather by 5-13%, only S=6's 16-vs-32 clears the noise
/// floor, so one flat constant. The un-split kernel (12 threadgroups at S=6) ran +15.6%.
pub const QSA_ATTN_NSPLIT_DEFAULT: c_int = 16;

/// Ceiling on the split count; past it the f32 partial buffer costs more traffic than it buys.
pub const QSA_ATTN_MAX_NSPLIT: c_int = 64;

/// Default lower width bound for the fused kernel.
pub const QSA_ATTN_MIN_S_DEFAULT: c_int = 2;

/// Width floor for a batched decode step of two or more slots: its S=1 rows take the fused kernel,
/// which beats the per-slot decode gather there (a lone stream keeps the decode gather).
pub const QSA_ATTN_BATCHED_MIN_S_DEFAULT: c_int = 1;

pub var qsa_attn_nsplit_override: ?c_int = null;
pub var qsa_attn_min_s_override: ?c_int = null;

/// Splits per (row, kv head): a measured constant, independent of kv and the row's key
/// count (NSPLIT is a template).
pub fn qsaAttnNSplit() c_int {
    const forced = qsa_attn_nsplit_override orelse 0;
    if (forced > 0) {
        var f: c_int = 1;
        while (f < forced and f < QSA_ATTN_MAX_NSPLIT) f *= 2;
        return f;
    }
    return QSA_ATTN_NSPLIT_DEFAULT;
}

/// Lowest query width the fused kernel serves.
pub fn qsaAttnMinS() c_int {
    const v = qsa_attn_min_s_override orelse QSA_ATTN_MIN_S_DEFAULT;
    return @max(1, @min(v, FUSED256_MIN_Q_LEN - 1));
}

/// Lowest width the fused kernel serves for S=1 rows of a batched step with two or more slots.
pub fn qsaAttnBatchedMinS() c_int {
    return @max(1, @min(QSA_ATTN_BATCHED_MIN_S_DEFAULT, FUSED256_MIN_Q_LEN - 1));
}

/// Does the fused sparse-attention kernel serve this call? Dense KV takes the kernel's bf16
/// variant (same split-K grid). It must never fall into `gatherQsa256`, the prefill kernel,
/// whose grid starves at a verify width (16k fp16: 77.7 vs 93.6 tok/s).
pub fn qsaSparseAttnServes(has_quant_triple: bool, seq_len: c_int, min_s: c_int) bool {
    if (seq_len < min_s or seq_len >= FUSED256_MIN_Q_LEN) return false;
    return has_quant_triple or qsaAttnDenseEnabled();
}

/// Will the fused verify kernel take a verify-width (2..15) call on this KV type? Its switch,
/// its width floor and its dense switch; no kv floor, because it reads each row's own
/// O(budget) keys at any kv.
pub fn qsaFusedVerifyServes(quantized: bool, seq_len: c_int) bool {
    if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN) return false;
    return qsaAttnKernelEnabled() and qsaSparseAttnServes(quantized, seq_len, qsaAttnMinS());
}

/// The indexer's choice for one single-slot call: per-row block selection (`true`) or the
/// dense `[S, kv]` mask. Prefill and decode widths gather past `qsaGatherMinKv`. Verify widths
/// take blocks wherever the fused kernel serves them (every kv the indexer is active);
/// otherwise only past the union gather's own floor.
pub fn qsaWantsBlocks(batch: c_int, kv: c_int, seq_len: c_int, quantized: bool) bool {
    if (batch != 1 or !qsaGatherEnabled()) return false;
    if (qsaFusedVerifyServes(quantized, seq_len)) return true;
    if (seq_len >= FUSED256_MIN_Q_LEN) return kv > qsaPrefillGatherMinKv();
    if (kv <= qsaGatherMinKv()) return false;
    if (seq_len == 1) return qsaDecodeGatherEnabled();
    return qsaVerifyGatherEnabled() and kv > qsaVerifyGatherMinKvFor(quantized);
}

var qsa_attn_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "pacc", "pml" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_qmerge",
        in_vec,
        out_vec,
        QSA_ATTN_MERGE_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_merge_kernel_cached = kernel;
    return kernel;
}

/// Threadgroup tile depth: 16 keys stages 12 KiB, half the dense prefill kernel's footprint.
pub const QSA_ATTN_BK: c_int = 16;

pub var qsa_attn_bk_override: ?c_int = null;

/// Threadgroup tile depth (`QSA_ATTN_BK`; the test seam must be a multiple of 8, `KT = BK / 8`). A new value owes a
/// re-run of the NSPLIT sweep.
pub fn qsaAttnBk() c_int {
    const v = qsa_attn_bk_override orelse QSA_ATTN_BK;
    if (v <= 0 or @rem(v, 8) != 0 or v > 32) return QSA_ATTN_BK;
    return v;
}


var qsa_attn_q_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_attn_d_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnDKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_d_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "vq" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_dsplit",
        in_vec,
        out_vec,
        QSA_ATTN_D_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // K/V are cache views (see msv_attn_qsa256)
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_d_kernel_cached = kernel;
    return kernel;
}

fn getQsaAttnQKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_q_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "ksc", "kbi", "vq", "vsc", "vbi" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256_qsplit",
        in_vec,
        out_vec,
        QSA_ATTN_Q_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // packed triples are cache VIEWS — a contiguous copy erases the win
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_q_kernel_cached = kernel;
    return kernel;
}

var qsa_attn_slots_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

/// One pipeline for 2..4 slots: each slot's six triples bind slot-major after q/scl/blocks (so one slot's inputs
/// lay out as the solo kernel's), `meta` last; an absent slot repeats slot 0's.
fn getQsaAttnSlotsKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_slots_kernel_cached) |kk| return kk;
    const input_names = comptime blk: {
        var nm: [28][*:0]const u8 = undefined;
        nm[0..3].* = .{ "q", "scl", "blocks" };
        for (0..24) |i| nm[3 + i] = std.fmt.comptimePrint("{s}{d}", .{ ([_][]const u8{ "kq", "ksc", "kbi", "vq", "vsc", "vbi" })[i % 6], i / 6 });
        nm[27] = "meta";
        break :blk nm;
    };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("msv_attn_qsa256_qsplit_slots", in_vec, out_vec, QSA_ATTN_SLOTS_SOURCE, ATTN_QSA256_KERNEL_HEADER, false, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_slots_kernel_cached = kernel;
    return kernel;
}


/// Fused QSA sparse attention at verify widths and batched decode; the test seam's false runs the gather arms.
pub fn qsaAttnKernelEnabled() bool {
    return transformer.qsa_attn_kernel_override orelse true;
}

pub var qsa_attn_dense_override: ?bool = null;

/// The fused kernel's bf16 (dense KV) variant; the test seam's false hands dense KV back to the mask arm and the
/// union gather.
pub fn qsaAttnDenseEnabled() bool {
    return qsa_attn_dense_override orelse true;
}

const QsaAttnCfgKey = struct {
    /// Its batch axis is the slot count.
    q_shape: ShapeKey,
    /// Shapes the grid and is not recoverable from the q shape.
    h_kv: c_int,
    bits: u8,
    gs: u32,
    nsg: c_int,
    nsplit: c_int,
    bk: c_int,
    ratio: c_int,
    dtype: mlx.mlx_dtype,
};
var qsa_attn_cfgs = QsaCfgCache(QsaAttnCfgKey, 2){};

var qsa_attn_engaged_bits: OneShotBits = .{};

/// The K/V inputs one variant of the split pass reads, validated. `bits == 16` is the dense
/// (bf16) variant; its two arrays bind to `kq`/`vq`.
const QsaAttnKvInputs = struct {
    arrays: [6]mlx.mlx_array,
    n: usize,
    h_kv: c_int,
    kv: c_int,
    bits: u8,
    gs: u32,
};

/// The packed affine triples, or null when the quantized variant cannot read them.
fn qsaAttnQuantInputs(kv_view: *const DenseKVView) ?QsaAttnKvInputs {
    if (kv_view.bits != 4 and kv_view.bits != 8) return null;
    if (kv_view.group_size == 0 or @rem(@as(c_int, 256), @as(c_int, @intCast(kv_view.group_size))) != 0) return null;
    // One quant group per 8-dim staging chunk.
    if (kv_view.group_size % 8 != 0) return null;
    // `has_quant_triple` promises all six handles; a null ctx would fault, so check.
    const arrays = [6]mlx.mlx_array{
        kv_view.k_triple_q, kv_view.k_triple_scales, kv_view.k_triple_biases,
        kv_view.v_triple_q, kv_view.v_triple_scales, kv_view.v_triple_biases,
    };
    for (arrays) |a| {
        if (a.ctx == null) return null;
    }
    if (mlx.mlx_array_ndim(kv_view.k_triple_q) != 4 or mlx.mlx_array_ndim(kv_view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(kv_view.k_triple_q);
    const vs = mlx.getShape(kv_view.v_triple_q);
    const vpw: c_int = @divExact(@as(c_int, 32), @as(c_int, kv_view.bits));
    if (ks.len != 4 or vs.len != 4 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[3] * vpw != 256 or vs[3] * vpw != 256) return null;
    const h_kv = ks[1];
    const kv = ks[2];
    if (vs[1] != h_kv or vs[2] != kv) return null;
    const groups: c_int = @divExact(@as(c_int, 256), @as(c_int, @intCast(kv_view.group_size)));
    const ksc_sh = mlx.getShape(kv_view.k_triple_scales);
    const vsc_sh = mlx.getShape(kv_view.v_triple_scales);
    if (ksc_sh.len != 4 or vsc_sh.len != 4 or ksc_sh[3] != groups or vsc_sh[3] != groups) return null;
    if (ksc_sh[2] != kv or vsc_sh[2] != kv) return null;
    // `ensure_row_contiguous = false` (the packed triples are cache views), so the kernel
    // indexes every innermost axis with unit stride; decline anything else.
    for (arrays) |a| {
        if (mlx.mlx_array_strides(a)[3] != 1) return null;
    }
    return .{ .arrays = arrays, .n = 6, .h_kv = h_kv, .kv = kv, .bits = kv_view.bits, .gs = kv_view.group_size };
}

/// The bf16 K/V views, or null when the dense variant cannot read them. Layout is the
/// kernel's business: a lazy view reports row-contiguous strides until it is evaluated and
/// mlx-c exposes no data offset, so the host cannot judge alignment. The kernel checks the
/// bound pointer and the real strides and falls back to element loads (`k_vec`/`v_vec`).
fn qsaAttnDenseInputs(kv_view: *const DenseKVView) ?QsaAttnKvInputs {
    const k = kv_view.k;
    const v = kv_view.v;
    if (k.ctx == null or v.ctx == null) return null;
    if (mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    if (mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (ks[0] != 1 or ks[3] != 256 or !std.mem.eql(c_int, ks, vs)) return null;
    var arrays: [6]mlx.mlx_array = @splat(.{ .ctx = null });
    arrays[0] = k;
    arrays[1] = v;
    return .{ .arrays = arrays, .n = 2, .h_kv = ks[1], .kv = ks[2], .bits = 16, .gs = 0 };
}

/// One fused sparse-attention dispatch for a verify-width block or a batched decode row: each
/// row indexes its own selected blocks plus its own tail. Quantized KV unpacks its triples
/// in-kernel; dense (bf16) KV takes the same split pass with plain loads. Null = declined.
pub fn qsaSparseAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1, Hq, S, 256] bf16, post-RoPE
    kv_view: *const DenseKVView, // cache AFTER this block's append
    blocks: mlx.mlx_array, // [1, S, kb] int32 ascending, INT_MAX padded
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    return qsaSparseAttnFloor(s, q_rope, kv_view[0..1], &.{blocks}, ratio, attn_scale, qsaAttnMinS());
}

/// `qsaSparseAttn` under an explicit width floor (the batched decode floor for S=1 rows of a batch).
/// One view and block list per q batch row: 2..4 quantized S=1 slots share one launch, exactly as N solo launches.
pub fn qsaSparseAttnFloor(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    min_s: c_int,
) !?mlx.mlx_array {
    return sparseAttnFloorCore(s, q_rope, views, blocks, ratio, attn_scale, min_s, null);
}

/// The split pass's partials of a one-view call (each the holder's): pacc f32 [Hq, S, NSPLIT, 256], pml f32 [Hq, S,
/// NSPLIT, 2].
pub const SplitParts = struct {
    pacc: mlx.mlx_array = .{ .ctx = null },
    pml: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *SplitParts) void {
        inline for (.{ &self.pacc, &self.pml }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// `qsaSparseAttn`'s split pass alone, for a consumer that runs the merge's statements itself (the fused o_proj
/// prepare); null where `qsaSparseAttn` declines.
pub fn qsaSparseAttnSplit(s: mlx.mlx_stream, q_rope: mlx.mlx_array, kv_view: *const DenseKVView, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?SplitParts {
    var parts: SplitParts = .{};
    errdefer parts.deinit();
    if ((try sparseAttnFloorCore(s, q_rope, kv_view[0..1], &.{blocks}, ratio, attn_scale, qsaAttnMinS(), &parts)) == null) return null;
    return parts;
}

/// `qsaSparseAttnFloor`'s body. With `parts` (one view) the split pass's outputs are handed over and the merge does not
/// launch: a served call then returns a null-ctx array (null still means declined).
fn sparseAttnFloorCore(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    min_s: c_int,
    parts: ?*SplitParts,
) !?mlx.mlx_array {
    if (!qsaAttnKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const n = views.len;
    if (n == 0 or n > 4 or blocks.len != n) return null;
    if (parts != null and n != 1) return null;
    const kv_view = &views[0];
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks[0]) != 3) return null;
    const qs = mlx.getShape(q_rope);
    const seq_len = qs[2];
    // `min_s` keeps a lone decode row on `qsaDecodeGatherAttn` (a batched step lowers it);
    // prefill keeps `gatherQsa256`; a dense cache follows `qsaAttnDenseEnabled`.
    if (!qsaSparseAttnServes(kv_view.has_quant_triple, seq_len, min_s)) return null;
    if (qs[0] != n or qs[3] != 256 or (n > 1 and seq_len != 1)) return null;
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks[0]) != .int32) return null;
    const bs = mlx.getShape(blocks[0]);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return null;
    // `kb` rides `blocks_shape[2]` (the slot kernel: `meta[2]`) into the kernel.
    _ = bs[2];
    if (ratio <= 0) return null;
    // MLX binds an input under 8 elements in the `constant` address space, and the kernel's
    // `const device int* blk` then fails to compile (a throw at eval, not a decline).
    if (bs[1] * bs[2] < 8) return null;
    if (mlx.mlx_array_strides(q_rope)[3] != 1) return null;
    if (mlx.mlx_array_strides(blocks[0])[2] != 1) return null;

    const kvin = (if (kv_view.has_quant_triple) qsaAttnQuantInputs(kv_view) else qsaAttnDenseInputs(kv_view)) orelse return null;
    const dense = kvin.bits == 16;
    const h_kv = kvin.h_kv;
    const kv = kvin.kv;
    if (h_kv <= 0 or kv < seq_len) return null;
    if (@rem(qs[1], h_kv) != 0) return null;
    // Slots: one geometry and dtype, `device`-bound triples. The kernel cannot bind each slot's strides (24
    // more buffers, past Metal's 31) and a lazy view reports contiguous strides until evaluated, so `meta`
    // carries each slot's parent capacity; a slot's six buffers are row-contiguous and share it on the kv axis.
    var meta: [11]i32 = .{ qs[1], h_kv, bs[2] } ++ @as([8]i32, @splat(0));
    var inputs_arr: [28]mlx.mlx_array = undefined;
    if (n > 1) for (views, blocks, 0..) |*v, blk, i| {
        const kin = qsaAttnQuantInputs(v) orelse return null;
        if (blk.ctx == null or mlx.mlx_array_dtype(blk) != .int32 or !std.mem.eql(c_int, mlx.getShape(blk), bs) or mlx.mlx_array_strides(blk)[2] != 1) return null;
        if (kin.bits != kvin.bits or kin.gs != kvin.gs or kin.h_kv != h_kv or kin.kv < 1 or v.kv_cap < kin.kv) return null;
        for (kin.arrays, kvin.arrays) |x, x0| if (mlx.mlx_array_size(x) < 8 or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(x0)) return null;
        meta[3 + 2 * i ..][0..2].* = .{ kin.kv, v.kv_cap };
        inputs_arr[3 + 6 * i ..][0..6].* = kin.arrays;
    };
    const gqa = @divExact(qs[1], h_kv);
    if (gqa <= 0 or gqa > 64) return null;

    const kernel = (if (n > 1) getQsaAttnSlotsKernel() else if (dense) getQsaAttnDKernel() else getQsaAttnQKernel()) catch return null;
    const merge_kernel = getQsaAttnMergeKernel() catch return null;
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const nsplit = qsaAttnNSplit();
    const bk = qsaAttnBk();
    const scl_data = [_]f32{attn_scale};
    const one = [_]c_int{1};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);

    const key = QsaAttnCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = h_kv,
        .bits = kvin.bits,
        .gs = kvin.gs,
        .nsg = nsg,
        .nsplit = nsplit,
        .bk = bk,
        .ratio = ratio,
        .dtype = .bfloat16,
    };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_attn_cfgs.get(key) orelse blk: {
        // Split pass: one threadgroup per (row, kv head, split).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const acc_shape = [_]c_int{ qs[0] * qs[1], seq_len, nsplit, 256 };
        const ml_shape = [_]c_int{ qs[0] * qs[1], seq_len, nsplit, 2 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &acc_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, seq_len * 32, h_kv * nsg, qs[0] * nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio));
        if (!dense) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(kvin.bits)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(kvin.gs)));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSPLIT", nsplit));

        // Merge pass: one threadgroup per (q head, row), 256 threads.
        const mconfig = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(mconfig);
        const o_shape = [_]c_int{ qs[0], qs[1], seq_len, 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mconfig, &o_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mconfig, 256, seq_len, qs[0] * qs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mconfig, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mconfig, "T", .bfloat16));

        qsa_attn_cfgs.put(key, .{ config, mconfig });
        break :blk .{ config, mconfig };
    };

    inputs_arr[0] = q_rope;
    inputs_arr[1] = scl;
    inputs_arr[2] = blocks[0];
    @memcpy(inputs_arr[3..][0..kvin.n], kvin.arrays[0..kvin.n]);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    var bst = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bst);
    var meta_arr: mlx.mlx_array = .{ .ctx = null };
    defer if (meta_arr.ctx != null) {
        _ = mlx.mlx_array_free(meta_arr);
    };
    if (n > 1) {
        // The kernel computes the q and block offsets itself: q [N, Hq, 1, 256] and blocks [N, 1, KB], contiguous.
        try mlx.check(mlx.mlx_contiguous(&qc, q_rope, false, s));
        const bvec = mlx.mlx_vector_array_new_data(blocks.ptr, n);
        defer _ = mlx.mlx_vector_array_free(bvec);
        try mlx.check(mlx.mlx_concatenate_axis(&bst, bvec, 0, s));
        meta_arr = mlx.mlx_array_new_data(&meta, &[_]c_int{meta.len}, 1, .int32);
        inputs_arr[0] = qc;
        inputs_arr[2] = bst;
        inputs_arr[27] = meta_arr;
        for (n..4) |i| inputs_arr[3 + 6 * i ..][0..6].* = kvin.arrays;
    }
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, if (n > 1) 28 else 3 + kvin.n);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var parts_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(parts_vec) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts_vec, 1));

    var out: mlx.mlx_array = .{ .ctx = null };
    if (parts) |p| {
        p.pacc = mlx.mlx_array_new();
        p.pml = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&p.pacc, pacc));
        try mlx.check(mlx.mlx_array_set(&p.pml, pml));
    } else {
        const merge_in = [_]mlx.mlx_array{ pacc, pml };
        const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
        defer _ = mlx.mlx_vector_array_free(merge_vec);
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, merge_kernel, merge_vec, cfgs[1], s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
        out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    }
    // One bit per arm: width bucket (+4 dense) for one view, bit 8 for the slot launch.
    if (n > 1) {
        if (qsa_attn_engaged_bits.take(8)) {
            var kv_lo: i32 = meta[3];
            var kv_hi: i32 = meta[3];
            for (1..n) |i| {
                kv_lo = @min(kv_lo, meta[3 + 2 * i]);
                kv_hi = @max(kv_hi, meta[3 + 2 * i]);
            }
            log.info(
                "[qsa-attn] engaged (slots={d} S={d} kv={d}..{d} kv-bits={d}/gs{d} gqa={d} bk={d} nsplit={d} tgs={d})\n",
                .{ n, seq_len, kv_lo, kv_hi, kvin.bits, kvin.gs, gqa, bk, nsplit, seq_len * h_kv * nsplit * @as(c_int, @intCast(n)) },
            );
        }
    } else if (qsa_attn_engaged_bits.take(qsaWidthBucket(seq_len) + @as(u5, if (dense) 4 else 0))) {
        log.info(
            "[qsa-attn] engaged (S={d} kv={d} kv-bits={d}/gs{d} gqa={d} bk={d} nsplit={d} tgs={d})\n",
            .{ seq_len, kv, kvin.bits, kvin.gs, gqa, bk, nsplit, seq_len * h_kv * nsplit },
        );
    }
    return out;
}

/// What served one single-slot QSA call: the output and its cost-trace bit
/// (1 fused, 2 decode gather, 4 union gather, 8 prefill gather, 16 mask).
pub const QsaSlotServed = struct { out: mlx.mlx_array, cost_bit: u8 };

/// One single-slot qwen4_exp QSA attention call: the arms in the order the forward tries
/// them. The indexer set `blocks` (per-row selection) or `mask.*` (dense `[S, kv]`). A
/// declined gather builds the mask here and writes it back so later layers reuse it.
pub fn qsaSlotAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    kv_view: *const DenseKVView,
    blocks: mlx.mlx_array,
    mask: *mlx.mlx_array,
    arms: *QsaArmCounts,
    seq_len: c_int,
    ratio: c_int,
    attn_scale: f32,
) !QsaSlotServed {
    const standin_sdpa = qwen4Standin().attn_sdpa;
    const qsa_ok = blocks.ctx != null and mask.ctx == null and !standin_sdpa;
    const verify_w = seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN;
    // Three arms, three independent widths: `qsaAttnMinS` gates the fused kernel only
    // (gating the whole block on it dropped S=2..5 onto the dense mask).
    if (qsa_ok and seq_len >= qsaAttnMinS() and seq_len < FUSED256_MIN_Q_LEN) {
        // One fused dispatch per layer: each row indexes its own selection + tail.
        if (try qsaSparseAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 1);
    }
    if (qsa_ok and seq_len == 1 and !kv_view.has_quant_triple) {
        const dec = (try qsa_decode.attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale)) orelse
            try qsa_decode_cuda.attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale);
        if (dec) |g| {
            if (arms_mod.qsa_dec_engaged_bits.take(0)) log.info("[qsa-dec] engaged (kv={d})\n", .{mlx.getShape(kv_view.k)[2]});
            return qsaSlotGathered(arms, seq_len, g, 2);
        }
    }
    if (qsa_ok and seq_len == 1) {
        // Decode width: subset triples (or dense rows) → subset
        // dequant -> dense SDPA over K'<<kv. Declines to the mask arm.
        if (try qsaDecodeGatherAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 2);
    }
    if (qsa_ok and verify_w and !kv_view.has_quant_triple) {
        // CUDA: each verify row attends its own picks and tail; the mask arm read the whole cache.
        if (try qsa_decode_cuda.attend(s, q_rope, kv_view.k, kv_view.v, blocks, ratio, attn_scale)) |g| {
            if (arms_mod.qsa_dec_engaged_bits.take(1)) log.info("[qsa-dec] verify rows engaged (S={d} kv={d})\n", .{ seq_len, mlx.getShape(kv_view.k)[2] });
            return qsaSlotGathered(arms, seq_len, g, 4);
        }
    }
    if (qsa_ok and verify_w) {
        // Verify widths 2..15: the union of the block's rows' selections + tail. The
        // fused kernel's fallback for any width it declines.
        if (try qsaVerifyGatherAttn(s, q_rope, kv_view, blocks, ratio, attn_scale)) |g| return qsaSlotGathered(arms, seq_len, g, 4);
    }
    // The prefill kernel has no q_len floor of its own; a verify-width
    // selection must never fall into it (it walks the WHOLE cache).
    if (seq_len >= FUSED256_MIN_Q_LEN and blocks.ctx != null and !standin_sdpa) {
        if (try qsaPrefillGatherProfiled(s, q_rope, kv_view.k, kv_view.v, attn_scale, blocks, ratio)) |g| return qsaSlotGathered(arms, seq_len, g, 8);
    }
    // The dense [S, kv] mask arm reads `k`/`v` — on a quantized cache that is what
    // EVALUATES the whole-range dequant graph `updateAffine` built, per layer per forward.
    arms.note(.mask);
    if (mask.ctx == null) mask.* = try qsaMaskFromBlocks(s, blocks, mlx.getShape(kv_view.k)[2], ratio);
    if (standin_sdpa) return .{ .out = try standinRef(q_rope), .cost_bit = 16 };
    return .{ .out = try qsaMaskArm(s, q_rope, kv_view.k, kv_view.v, attn_scale, mask.*), .cost_bit = 16 };
}

fn qsaSlotGathered(arms: *QsaArmCounts, seq_len: c_int, out: mlx.mlx_array, cost_bit: u8) QsaSlotServed {
    arms.noteGather(seq_len);
    return .{ .out = out, .cost_bit = cost_bit };
}

/// `gatherQsa256` with the `QWEN4_PROFILE_QSA` timer around it.
fn qsaPrefillGatherProfiled(s: mlx.mlx_stream, q_rope: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, attn_scale: f32, blocks: mlx.mlx_array, ratio: c_int) !?mlx.mlx_array {
    const prof = env.on(.QWEN4_PROFILE_QSA);
    var clk: ProfClock = undefined;
    if (prof) {
        try mlx.check(mlx.mlx_array_eval(q_rope));
        clk = ProfClock.init();
    }
    const gathered = try gatherQsa256(s, q_rope, k, v, attn_scale, blocks, ratio);
    if (prof) if (gathered) |g| {
        try mlx.check(mlx.mlx_array_eval(g));
        log.info("[qsa-prof] gather S={d} kv={d}: {d:.2} ms\n", .{ mlx.getShape(q_rope)[2], mlx.getShape(k)[2], @as(f64, @floatFromInt(clk.lap())) / 1e6 });
    };
    return gathered;
}

// ── tests ──

pub fn qsaAttnCfgBuilds() usize {
    return qsa_attn_cfgs.builds;
}

// ── Fused sparse attention at verify widths and batched decode (msv_attn_qsa256_q) ──

/// Build a KVCache whose view is a strided slice of a larger capacity buffer (the shape the serving path hands the kernel).
pub fn qsaAttnCacheFixture(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    k_dense: mlx.mlx_array,
    v_dense: mlx.mlx_array,
    kv: c_int,
    split: c_int,
    cfg: kv_quant.KVQuantConfig,
    cache: *KVCache,
) !DenseKVView {
    _ = alloc;
    const st = [_]c_int{ 1, 1, 1, 1 };
    var k0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k0);
    var v0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v0);
    const ks = mlx.getShape(k_dense);
    try mlx.check(mlx.mlx_slice(&k0, k_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v0, v_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    var first = try cache.update(0, k0, v0, s, 0);
    first.deinit();
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var v1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v1);
    try mlx.check(mlx.mlx_slice(&k1, k_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v1, v_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    _ = cfg;
    return try cache.update(0, k1, v1, s, 0);
}
