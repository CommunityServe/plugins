//! copied_from: src/plugins/qwen4_exp/qsa/verify_gather.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA attention at verify widths (S 2..15) over gathered rows (`qsaVerifyGatherAttn`) and its floors per KV scheme.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../sdk/util/log.zig");
const arms = @import("arms.zig");
const sparse_attn = @import("sparse_attn.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const takeContig = @import("../../../mlx/array.zig").takeContig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256Cosine = @import("../../../sdk/blocks/attention/attn256.zig").attn256Cosine;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const QsaArmCounts = @import("arms.zig").QsaArmCounts;
const QsaDecline = @import("arms.zig").QsaDecline;
const qsaEngagedBit = @import("arms.zig").qsaEngagedBit;
const qsaBatchedGatherFloor = @import("gather.zig").qsaBatchedGatherFloor;
const qsaGatherMinKv = @import("gather.zig").qsaGatherMinKv;
const qsaPrefillGatherMinKv = @import("gather.zig").qsaPrefillGatherMinKv;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAttnCacheFixture = @import("sparse_attn.zig").qsaAttnCacheFixture;
const qsaSlotAttn = @import("sparse_attn.zig").qsaSlotAttn;
const qsaWantsBlocks = @import("sparse_attn.zig").qsaWantsBlocks;


/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN — an MTP/DFlash
/// verify block): read the UNION of the S rows' selections + the union tail
/// instead of the whole cache under a dense [S, kv] mask. With `--kv-quant 8`
/// the dense mask arm also dequantizes the whole stored range per layer per
/// forward; this arm dequantizes only the gathered rows.
/// Test seams: the verify gather on or off, and its KV floor.
pub var qsa_verify_gather_override: ?bool = null;
pub var qsa_verify_gather_min_kv_override: ?c_int = null;

pub fn qsaVerifyGatherEnabled() bool {
    return qsa_verify_gather_override orelse true;
}

/// The union is FIXED-size (S * budget blocks): at the production budget
/// (512 blocks of 4) S=7 is 14336 rows. On dense KV the gather COPIES those
/// rows while the mask arm reads the cache in place, so the copy only pays
/// once the union is well under half the cache (M4 Max, S=7: gather loses 7%
/// at 17k keys, breaks even at 34k). Quantized KV dequantizes only the
/// gathered rows, so its floor stays lower.
pub const QSA_VERIFY_GATHER_MIN_KV_DENSE: c_int = 32768;
pub const QSA_VERIFY_GATHER_MIN_KV_QUANT: c_int = 16384;

/// `quantized` = the arm will dequantize only the gathered rows, i.e. the
/// view carries affine triples (`DenseKVView.has_quant_triple`).
pub fn qsaVerifyGatherMinKvFor(quantized: bool) c_int {
    if (qsa_verify_gather_min_kv_override) |v| return v;
    return if (quantized) QSA_VERIFY_GATHER_MIN_KV_QUANT else QSA_VERIFY_GATHER_MIN_KV_DENSE;
}

/// Geometry of the verify-width QSA gather — pure arithmetic, the ONE owner
/// of R (the gate, the MLX gatherer and the host reference plan all read it).
///
/// The gathered set is two DISJOINT parts:
///   A. the union tail `[tail_start, kv)`, where `tail_start` is row 0's own
///      tail start `ratio * floor((offset+1)/ratio)`. Tail starts are
///      non-decreasing in position and tails are SUFFIXES, so row 0's (the
///      SMALLEST start) is the largest and covers every other row's.
///   B. the union of the rows' selected blocks that lie ENTIRELY below
///      `tail_start` (`b < nb_lo`), deduplicated and padded to a fixed
///      `k_blocks` slots with blocks no row selected. A padded block's tokens
///      are below every row's tail start and in no row's selection, so the
///      per-row mask hides them everywhere — no host sync is needed to learn
///      the real union size.
/// Selected blocks at/above `tail_start` need no slot: their tokens are in A.
pub const QsaVerifyGeom = struct {
    /// Complete blocks in the cache.
    nb: c_int,
    /// Blocks lying entirely below the union tail (`tail_start / ratio`).
    nb_lo: c_int,
    /// Row 0's tail start = the union tail's first token.
    tail_start: c_int,
    tail_len: c_int,
    /// Fixed union slot count (part B).
    k_blocks: c_int,
    /// R — gathered rows total.
    rows: c_int,

    pub fn compute(seq_len: c_int, kv: c_int, ratio: c_int, kb: c_int) ?QsaVerifyGeom {
        if (ratio <= 0 or kb <= 0) return null;
        if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN) return null;
        if (kv <= seq_len) return null;
        const offset = kv - seq_len;
        const nb = @divTrunc(kv, ratio);
        const tail_start = @divTrunc(offset + 1, ratio) * ratio;
        if (tail_start > kv) return null;
        const nb_lo = @divTrunc(tail_start, ratio);
        const k_blocks = @min(nb_lo, seq_len * kb);
        return .{
            .nb = nb,
            .nb_lo = nb_lo,
            .tail_start = tail_start,
            .tail_len = kv - tail_start,
            .k_blocks = k_blocks,
            .rows = k_blocks * ratio + (kv - tail_start),
        };
    }
};

/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN, sorted `blocks`):
/// gather the fixed-size union of the S rows' selections + the union tail
/// (see `QsaVerifyGeom`), dequantizing ONLY those rows on a quantized cache,
/// then dense SDPA under the per-row `[S, R]` additive mask. Null = declined
/// (kill switch, kv floor, no sparsity win, dtype, geometry) — the caller
/// falls back to the dense `[S, kv]` mask arm.
///
/// Parity argument: the gathered rows are STRICTLY ASCENDING (part B blocks
/// are all below the tail's first token and the block ids are sorted), so no
/// key can be counted twice; the mask is the dense formula evaluated at the
/// gathered tokens; and every token the dense mask shows a row is present
/// (a token below the union tail that a row sees is in that row's selection,
/// hence in the union, and `k_blocks >= |union|` by construction). Remaining
/// deltas are kernel tiling order.
pub fn qsaVerifyGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,S,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this block's append
    blocks: mlx.mlx_array, // [1,S,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &arms.qsa_verify_decline_bits, .arm = "qsa-verify-gather" };
    if (!qsaVerifyGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const qs = mlx.getShape(q_rope);
    if (qs[0] != 1) return dq.no(.geometry);
    const seq_len: c_int = qs[2];
    dq.s = seq_len;
    const head_dim: c_int = qs[3];
    const h_q: c_int = qs[1];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return dq.no(.geometry);
    const kb: c_int = bs[2];

    if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
    const ks = mlx.getShape(kv_view.k);
    if (ks.len != 4 or ks[0] != 1) return dq.no(.geometry);
    const h_kv: c_int = ks[1];
    const kv: c_int = ks[2];
    dq.kv = kv;
    if (h_kv <= 0 or kv <= 0 or @rem(h_q, h_kv) != 0) return dq.no(.geometry);
    if (ks[3] != head_dim) return dq.no(.geometry);
    const vs = mlx.getShape(kv_view.v);
    if (vs.len != 4 or vs[0] != 1 or vs[1] != h_kv or vs[2] != kv or vs[3] != head_dim) return dq.no(.geometry);
    if (kv <= qsaVerifyGatherMinKvFor(kv_view.has_quant_triple)) return dq.no(.kv_floor);

    const geom = QsaVerifyGeom.compute(seq_len, kv, ratio, kb) orelse return dq.no(.width);
    if (geom.k_blocks <= 0 or geom.rows <= 0 or geom.rows >= kv) return dq.no(.no_win);
    const offset: c_int = kv - seq_len;
    const nb = geom.nb;
    const sel_w: c_int = nb + 2; // [.. nb-1] real, `nb` = the always-false
    // membership sink for tokens past the last complete block, `nb+1` the
    // sentinel sink.

    // Per-row block selection [S, nb+2] bool.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    const sink_arr = mlx.mlx_array_new_int(nb + 1);
    defer _ = mlx.mlx_array_free(sink_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s));
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, sink_arr, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ 1, seq_len, sel_w }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var sel3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel3);
    try mlx.check(mlx.mlx_put_along_axis(&sel3, falses, safe_b, true_v, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_reshape(&sel, sel3, &[_]c_int{ seq_len, sel_w }, 2, s)); // [S, nb+2]

    // Union over rows of the blocks BELOW the union tail, then a fixed-size
    // ascending id list: argsort a key that ranks union members first
    // (`b` vs `nb_lo + b`), keep k_blocks, sort so the rows come out
    // ascending. No host sync — the padding blocks mask out everywhere.
    const strides2 = [_]c_int{ 1, 1 };
    var sel_lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_lo);
    try mlx.check(mlx.mlx_slice(&sel_lo, sel, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ seq_len, geom.nb_lo }, 2, &strides2, 2, s));
    var union_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(union_b);
    try mlx.check(mlx.mlx_any_axes(&union_b, sel_lo, &[_]c_int{0}, 1, false, s)); // [nb_lo]
    var lo_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_idx);
    try mlx.check(mlx.mlx_arange(&lo_idx, 0, @floatFromInt(geom.nb_lo), 1.0, .int32, s));
    const nb_lo_arr = mlx.mlx_array_new_int(geom.nb_lo);
    defer _ = mlx.mlx_array_free(nb_lo_arr);
    var lo_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_shift);
    try mlx.check(mlx.mlx_add(&lo_shift, lo_idx, nb_lo_arr, s));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_where(&key, union_b, lo_idx, lo_shift, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, key, -1, s));
    var order_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i32);
    try mlx.check(mlx.mlx_astype(&order_i32, order, .int32, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_slice(&picked, order_i32, &[_]c_int{0}, 1, &[_]c_int{geom.k_blocks}, 1, &[_]c_int{1}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_sort_axis(&ids, picked, -1, s)); // [K] ascending

    // Token ids: each picked block's `ratio` tokens, then the union tail.
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var ids2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids2);
    try mlx.check(mlx.mlx_reshape(&ids2, ids, &[_]c_int{ geom.k_blocks, 1 }, 2, s));
    var ids_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_scaled);
    try mlx.check(mlx.mlx_multiply(&ids_scaled, ids2, ratio_arr, s));
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_reshape(&r2, rvec, &[_]c_int{ 1, ratio }, 2, s));
    var tok2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok2d);
    try mlx.check(mlx.mlx_add(&tok2d, ids_scaled, r2, s)); // [K, ratio]
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, tok2d, &[_]c_int{geom.k_blocks * ratio}, 1, s));
    var tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok);
    if (geom.tail_len > 0) {
        var tail_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_ids);
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(geom.tail_start), @floatFromInt(kv), 1.0, .int32, s));
        const parts = [_]mlx.mlx_array{ sel_tok, tail_ids };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&tok, vec, 0, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&tok, sel_tok));
    }
    var tok_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_u32);
    try mlx.check(mlx.mlx_astype(&tok_u32, tok, .uint32, s));

    // [S, R] visibility: the dense formula evaluated at the gathered tokens.
    var blk_of = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of);
    try mlx.check(mlx.mlx_floor_divide(&blk_of, tok, ratio_arr, s)); // [R], <= nb
    var blk_of_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of_u32);
    try mlx.check(mlx.mlx_astype(&blk_of_u32, blk_of, .uint32, s));
    var member = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(member);
    try mlx.check(mlx.mlx_take_axis(&member, sel, blk_of_u32, 1, s)); // [S, R]
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(offset), @floatFromInt(kv), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ seq_len, 1 }, 2, s));
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_arr, s));
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_multiply(&ts, nb_p, ratio_arr, s)); // [S,1]
    var tok_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_row);
    try mlx.check(mlx.mlx_reshape(&tok_row, tok, &[_]c_int{ 1, geom.rows }, 2, s));
    var tailvis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tailvis);
    try mlx.check(mlx.mlx_greater_equal(&tailvis, tok_row, ts, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, tok_row, pos2, s));
    var any_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(any_vis);
    try mlx.check(mlx.mlx_logical_or(&any_vis, member, tailvis, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_logical_and(&vis, any_vis, causal, s));
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vis, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, seq_len, geom.rows }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/8-bit (then
    // dequantize ONLY the subset — the whole point at `--kv-quant 8`), else
    // dense rows. Dequantize groups along D, so subset-first is exactly
    // full-first-restricted (pinned by the take-then-dequantize test).
    var gk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gk);
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    if (kv_view.has_quant_triple and (kv_view.bits == 4 or kv_view.bits == 8)) {
        const kq = kv_view.k_triple_q;
        const vq = kv_view.v_triple_q;
        if (kq.ctx == null or vq.ctx == null) return dq.no(.quant_layout);
        const kqs = mlx.getShape(kq);
        const vqs = mlx.getShape(vq);
        if (kqs.len != 4 or vqs.len != 4 or kqs[0] != 1 or vqs[0] != 1 or
            kqs[1] != h_kv or vqs[1] != h_kv or kqs[2] != kv or vqs[2] != kv) return dq.no(.quant_layout);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, tok_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, tok_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != .bfloat16 or mlx.mlx_array_dtype(gv) != .bfloat16) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != geom.rows or gvs2[2] != geom.rows or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    // hd 256 + an ARRAY mask has no fused arm in MLX: the same row-group
    // split the dense mask arm uses, now over [R] instead of [kv].
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (try splitMaskedSdpa256(s, q_rope, gk, gv, attn_scale, add4)) |split_out| {
        _ = mlx.mlx_array_free(out);
        out = split_out;
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    }
    if (arms.qsa_engaged_bits.take(qsaEngagedBit(.verify_gather, seq_len))) {
        log.info("[qsa-verify-gather] engaged (S={d} kv={d} rows={d})\n", .{ seq_len, kv, geom.rows });
    }
    return out;
}

// ── tests ──

/// Host reference plan for the verify gather: the ascending gathered token
/// ids and the `[S, R]` visibility mask. Executable spec of the MLX arm —
/// `qsaVerifyGatherAttn` builds the same two arrays in ops, and the parity
/// test pins its row ids against this.
pub const QsaVerifyPlan = struct {
    rows: []i32,
    mask: []bool, // [seq_len, rows.len], row-major

};

pub const QsaSlotPlan = struct {
    rows: []i32,
    mask: []bool,

};

// ── Verify-width QSA gather (S = 2 .. FUSED256_MIN_Q_LEN-1) ──
