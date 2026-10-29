//! copied_from: src/plugins/qwen4_exp/qsa/decode_gather.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA attention at decode and in a batched tick over the indexer's gathered blocks (`qsaDecodeGatherAttn`,
//! `qsaBatchedGatherAttn`).

const std = @import("std");
const verify_gather = @import("verify_gather.zig");
const qwen4_mod = @import("../qwen4_exp.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../engines/mlx/cache/kv_quant.zig");
const log = @import("../../../sdk/util/log.zig");
const arms_mod = @import("arms.zig");
const gather = @import("gather.zig");
const sparse_attn = @import("sparse_attn.zig");
const standin = @import("../../../sdk/diag/standin.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const bf16Scalar = @import("../../../engines/mlx/load/weights.zig").bf16Scalar;
const Trunk = @import("../trunk.zig").Trunk;
const takeContig = @import("../../../mlx/array.zig").takeContig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const fusedSdpa256Masked = @import("../../../sdk/blocks/attention/attn256.zig").fusedSdpa256Masked;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const splitMaskedSdpa256 = @import("../../../sdk/blocks/attention/sdpa.zig").splitMaskedSdpa256;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const maxAbsDiffF32 = @import("../../../sdk/testing/parity.zig").maxAbsDiffF32;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const QsaArm = @import("arms.zig").QsaArm;
const QsaDecline = @import("arms.zig").QsaDecline;
const qsaEngagedBit = @import("arms.zig").qsaEngagedBit;
const qsaWidthBucket = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").widthBucket;
const QsaBlockFixture = @import("gather.zig").QsaBlockFixture;
const qsaBatchedGatherEnabled = @import("gather.zig").qsaBatchedGatherEnabled;
const qsaBatchedGatherOn = @import("gather.zig").qsaBatchedGatherOn;
const qsaDecodeGatherEnabled = @import("gather.zig").qsaDecodeGatherEnabled;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAttnBatchedMinS = @import("sparse_attn.zig").qsaAttnBatchedMinS;
const qsaAttnCacheFixture = @import("sparse_attn.zig").qsaAttnCacheFixture;
const qsaAttnCfgBuilds = @import("sparse_attn.zig").qsaAttnCfgBuilds;
const qsaAttnMinS = @import("sparse_attn.zig").qsaAttnMinS;
const qsaSparseAttnFloor = @import("sparse_attn.zig").qsaSparseAttnFloor;
const qsaBatchedPlanHost = @import("verify_gather.zig").qsaBatchedPlanHost;
const qsaDecodeGatherIdsHost = @import("verify_gather.zig").qsaDecodeGatherIdsHost;
const qsaVerifyBlocksHost = @import("verify_gather.zig").qsaVerifyBlocksHost;
const qsaVerifyGatherAttn = @import("verify_gather.zig").qsaVerifyGatherAttn;
const qsaVerifyPlanHost = @import("verify_gather.zig").qsaVerifyPlanHost;

/// Decode-width QSA gather (S==1, sorted `blocks`, no mask): expand the
/// selection + incomplete tail into ascending token indices, gather those rows
/// (quant triples when the view carries them — dequantizing ONLY the subset —
/// else dense rows), then dense SDPA under an additive validity mask. Null =
/// declined (kill switch, dtype, geometry) — the caller keeps the dense mask.
///
/// Parity argument: the mask arm attends exactly the selected complete blocks'
/// tokens ∪ the incomplete tail ∧ causal. At decode the query is the last
/// cache row, so every gathered token is causal-visible and the ascending
/// gather order matches the masked row's summation order; the masked-out slots
/// contributed exact zeros there. Remaining deltas are kernel tiling order.
pub fn qsaDecodeGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,1,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this token's append
    blocks: mlx.mlx_array, // [1,1,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &arms_mod.qsa_decode_decline_bits, .arm = "qsa-decode-gather" };
    if (!qsaDecodeGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    const qs = mlx.getShape(q_rope);
    dq.s = qs[2];
    if (qs[0] != 1) return dq.no(.geometry);
    if (qs[2] != 1) return dq.no(.width);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return dq.no(.dtype);
    if (mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != 1) return dq.no(.geometry);
    const kb: c_int = bs[2];
    if (kb <= 0 or ratio <= 0) return dq.no(.geometry);
    const h_q: c_int = qs[1];
    const head_dim: c_int = qs[3];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);

    // Cache geometry off the dense K (present in every scheme; shapes are
    // known without executing the lazy dequant graph).
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

    const nb: c_int = @divTrunc(kv, ratio);
    const tail_start: c_int = nb * ratio;
    const tail_len: c_int = kv - tail_start;
    if (tail_len < 0) return dq.no(.geometry);
    const sel_len: c_int = kb * ratio;
    const gathered_len: c_int = sel_len + tail_len;
    // No sparsity win (or degenerate): keep the mask arm.
    if (gathered_len <= 0 or gathered_len >= kv) return dq.no(.no_win);

    // Token ids: selected blocks' ratio tokens each (sentinel → 0, masked
    // below) then the incomplete tail. Reachable decode rows have no
    // sentinels — every complete block is visible from the last position, so
    // visible == nb >= kb — and this stays ascending; the mask is the
    // backstop, not the plan.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s)); // [1,1,kb] bool
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, zero_i, s));
    var blk4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk4);
    try mlx.check(mlx.mlx_expand_dims(&blk4, safe_b, -1, s)); // [1,1,kb,1]
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s)); // [ratio]
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, rvec, &[_]c_int{ 1, 1, 1, ratio }, 4, s));
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var blk_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_scaled);
    try mlx.check(mlx.mlx_multiply(&blk_scaled, blk4, ratio_arr, s));
    var tok4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok4);
    try mlx.check(mlx.mlx_add(&tok4, blk_scaled, r4, s)); // [1,1,kb,ratio]
    var tok_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_flat);
    try mlx.check(mlx.mlx_reshape(&tok_flat, tok4, &[_]c_int{sel_len}, 1, s));
    var valid4e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4e);
    try mlx.check(mlx.mlx_expand_dims(&valid4e, valid_b, -1, s)); // [1,1,kb,1]
    var valid4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4);
    try mlx.check(mlx.mlx_broadcast_to(&valid4, valid4e, &[_]c_int{ 1, 1, kb, ratio }, 4, s));
    var valid_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_flat);
    try mlx.check(mlx.mlx_reshape(&valid_flat, valid4, &[_]c_int{sel_len}, 1, s));
    var tail_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ids);
    var tail_ok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ok);
    if (tail_len > 0) {
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(tail_start), @floatFromInt(kv), 1.0, .int32, s));
        try mlx.check(mlx.mlx_ones(&tail_ok, &[_]c_int{tail_len}, 1, .bool_, s));
    } else {
        const eshp = [_]c_int{0};
        try mlx.check(mlx.mlx_zeros(&tail_ids, &eshp, 1, .int32, s));
        try mlx.check(mlx.mlx_zeros(&tail_ok, &eshp, 1, .bool_, s));
    }
    const idx_parts = [_]mlx.mlx_array{ tok_flat, tail_ids };
    const idx_vec = mlx.mlx_vector_array_new_data(&idx_parts, 2);
    defer _ = mlx.mlx_vector_array_free(idx_vec);
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_concatenate_axis(&idx, idx_vec, 0, s)); // [K'] int32
    const vok_parts = [_]mlx.mlx_array{ valid_flat, tail_ok };
    const vok_vec = mlx.mlx_vector_array_new_data(&vok_parts, 2);
    defer _ = mlx.mlx_vector_array_free(vok_vec);
    var vok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vok);
    try mlx.check(mlx.mlx_concatenate_axis(&vok, vok_vec, 0, s)); // [K'] bool
    // take_axis wants unsigned indices; every id is >= 0 by construction.
    var idx_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_u32);
    try mlx.check(mlx.mlx_astype(&idx_u32, idx, .uint32, s));

    // Additive validity mask, same bf16 convention as the sliding masks.
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vok, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, 1, gathered_len }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/8-bit (then
    // dequantize ONLY the subset), else dense rows. Dequantize is per-token
    // along D, so subset-first is exactly full-first-restricted.
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
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, idx_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, idx_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != mlx.mlx_array_dtype(q_rope)) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != gathered_len or gvs2[2] != gathered_len or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    if (arms_mod.qsa_engaged_bits.take(qsaEngagedBit(.decode_gather, qs[2]))) {
        log.info("[qsa-decode-gather] engaged (S={d} kv={d} rows={d})\n", .{ qs[2], kv, gathered_len });
    }
    return out;
}

pub fn qsaBatchedGatherAttn(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    masks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    out_arms: []QsaArm,
) !mlx.mlx_array {
    const qs = mlx.getShape(q_rope);
    const n: usize = @intCast(qs[0]);
    const seq_len: c_int = qs[2];
    const h_count: c_int = qs[1];
    const hd: c_int = qs[3];
    if (views.len != n or blocks.len != n) return error.QsaBatchedGatherLen;
    if (masks.len != 0 and masks.len != n) return error.QsaBatchedGatherLen;
    if (out_arms.len != 0 and out_arms.len != n) return error.QsaBatchedGatherLen;
    const fused_min_s = if (n >= 2 and seq_len == 1) qsaAttnBatchedMinS() else qsaAttnMinS();
    const standin_sdpa = qwen4Standin().attn_sdpa;
    // S=1 slots on block lists share one fused launch; a decline falls through to the per-slot arms.
    const all_blocks = for (blocks, 0..) |b, i| {
        if (b.ctx == null or (i < masks.len and masks[i].ctx != null)) break false;
    } else true;
    if (n >= 2 and seq_len == 1 and all_blocks and !standin_sdpa) {
        if (try qsaSparseAttnFloor(s, q_rope, views, blocks, ratio, attn_scale, fused_min_s)) |out| {
            for (out_arms) |*a| a.* = .decode_gather;
            return out;
        }
    }
    var outs = try allocator.alloc(mlx.mlx_array, n);
    defer allocator.free(outs);
    var built: usize = 0;
    errdefer {
        var fi: usize = 0;
        while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    }
    for (views, blocks, 0..) |*kv_view, blk, i| {
        const slot_mask: mlx.mlx_array = if (i < masks.len) masks[i] else .{ .ctx = null };
        const i_c: c_int = @intCast(i);
        var q_slot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_slot);
        try mlx.check(mlx.mlx_slice(
            &q_slot,
            q_rope,
            &[_]c_int{ i_c, 0, 0, 0 },
            4,
            &[_]c_int{ i_c + 1, h_count, seq_len, hd },
            4,
            &[_]c_int{ 1, 1, 1, 1 },
            4,
            s,
        ));
        var gathered: ?mlx.mlx_array = null;
        const qsa_ok = blk.ctx != null and slot_mask.ctx == null and !standin_sdpa;
        if (qsa_ok and seq_len >= fused_min_s and seq_len < FUSED256_MIN_Q_LEN) {
            gathered = try qsaSparseAttnFloor(s, q_slot, kv_view[0..1], &.{blk}, ratio, attn_scale, fused_min_s);
        }
        if (gathered == null and qsa_ok and seq_len == 1) {
            gathered = try qsaDecodeGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered == null and qsa_ok and seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN) {
            gathered = try qsaVerifyGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered) |g| {
            outs[i] = g;
            built = i + 1;
            if (out_arms.len == n) out_arms[i] = switch (qsaWidthBucket(seq_len)) {
                0 => .decode_gather,
                1 => .verify_gather,
                else => .prefill_gather,
            };
            continue;
        }
        var mask_tmp: mlx.mlx_array = .{ .ctx = null };
        defer if (mask_tmp.ctx != null) {
            _ = mlx.mlx_array_free(mask_tmp);
        };
        const mask: mlx.mlx_array = if (slot_mask.ctx != null) slot_mask else if (blk.ctx != null) blk: {
            mask_tmp = try qsaMaskFromBlocks(s, blk, mlx.getShape(kv_view.k)[2], ratio);
            break :blk mask_tmp;
        } else .{ .ctx = null };
        outs[i] = mlx.mlx_array_new();
        built = i + 1;
        if (out_arms.len == n) out_arms[i] = .mask;
        if (mask.ctx != null) {
            if (try fusedSdpa256Masked(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |fused| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = fused;
            } else if (try splitMaskedSdpa256(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |split_out| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = split_out;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "array", mask, .{ .ctx = null }, false, s));
            }
        } else {
            const none = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(none);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "causal", none, .{ .ctx = null }, false, s));
        }
    }
    const vec = mlx.mlx_vector_array_new_data(outs.ptr, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    var fi: usize = 0;
    while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    return stacked;
}

pub fn qsaBatchedAttn(
    self: *Trunk,
    ctx: *ForwardCtx,
    slots: []const *ForwardCtx,
    q_rope: mlx.mlx_array,
    layer: u32,
    seq_len: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    if (!qsaBatchedGatherEnabled()) return null;
    if (self.qwen4 == null) return null;
    if (ctx.qsa_mask.ctx != null) return null;
    var any_blocks = false;
    for (slots) |sc| if (sc.qsa_blocks.ctx != null) {
        any_blocks = true;
    };
    if (!any_blocks) return null;
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    var views = try self.allocator.alloc(DenseKVView, slots.len);
    defer {
        for (views) |*dv| dv.deinit();
        self.allocator.free(views);
    }
    var blks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(blks);
    var slot_masks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(slot_masks);
    const arms = try self.allocator.alloc(QsaArm, slots.len);
    defer self.allocator.free(arms);
    for (slots, 0..) |slot_ctx, i| {
        views[i] = try slot_ctx.cache.denseView(layer, self.s);
        blks[i] = slot_ctx.qsa_blocks;
        slot_masks[i] = slot_ctx.qsa_mask;
    }
    if (!gather.qsa_batched_gather_logged) {
        gather.qsa_batched_gather_logged = true;
        log.info("[qsa-batched-gather] engaged (slots={d} S={d})\n", .{ slots.len, seq_len });
    }
    const stacked = try qsaBatchedGatherAttn(self.allocator, self.s, q_rope, views, blks, slot_masks, ratio, attn_scale, arms);
    for (slots, arms) |slot_ctx, arm| {
        slot_ctx.qsa_arms.note(arm);
        if (self.cost_trace_active) self.cost_attention |= switch (arm) {
            .mask => @as(u8, 16),
            .decode_gather => 2,
            .verify_gather => 4,
            .prefill_gather => 8,
        };
    }
    return stacked;
}

// ── tests ──
