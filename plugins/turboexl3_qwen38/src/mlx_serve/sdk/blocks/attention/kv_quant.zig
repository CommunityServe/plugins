//! copied_from: src/sdk/blocks/attention/kv_quant.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! KV-cache quantization backend.
//!
//! This module owns the storage / dispatch contract between the cache
//! (`KVCache` in `transformer.zig`) and the attention call sites. SDPA always
//! reads dense `[B, H, T, head_dim]` tensors via `KVCache.denseView` — what the
//! cache buffers actually hold is decided here.
//!
//! One non-trivial scheme: affine group-wise quantization at 4 or 8 bits,
//! mathematically identical to mlx-c's `mlx_quantize` weight path. The buffers
//! grow to 3 arrays per K and V (q, scales, biases); attention is unchanged
//! because `denseView` calls `dequantizeAffine` before returning.
//!
//! Adding a scheme: one `Scheme` variant, a quantize/dequantize pair here,
//! one arm each in `KVCache.update` (write) and `KVCache.denseView` (read).
//! SDPA call sites never change; the cache contract holds. The fused
//! quantized-attention kernels key on the affine triple (`transformer.zig`).

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");

/// KV-cache storage scheme.
///   * `off`      — dense bf16.
///   * `affine`   — group-wise affine quant via `mlx_quantize`/`mlx_dequantize`.
pub const Scheme = enum { off, affine };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }

    /// The wire vocabulary shared by the per-request `kv_quant` body field and
    /// `model-settings.json`: "off"/0, "4", "8". Null = unrecognized.
    pub fn fromJsonValue(v: std.json.Value) ?KVQuantConfig {
        switch (v) {
            .string => |s| {
                if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return dense;
                if (std.mem.eql(u8, s, "4")) return affine(4);
                if (std.mem.eql(u8, s, "8")) return affine(8);
                return null;
            },
            .integer => |i| {
                if (i == 0) return dense;
                if (i == 4) return affine(4);
                if (i == 8) return affine(8);
                return null;
            },
            else => return null,
        }
    }

    /// The same vocabulary, for reporting (`/v1/models` `meta.kv_quant`).
    pub fn wireName(self: KVQuantConfig) []const u8 {
        return switch (self.scheme) {
            .off => "off",
            .affine => if (self.bits == 4) "4" else "8",
        };
    }
};

/// One quantized K or V triple. Layout for input shape `[..., D]`:
///   q      : `[..., D * bits / 32]` uint32   (packed)
///   scales : `[..., D / group_size]` bf16
///   biases : `[..., D / group_size]` bf16
///
/// Owns its three array handles. Caller frees via `deinit`.
pub const QuantizedKV = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,

    pub fn deinit(self: *QuantizedKV) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        self.q = mlx.mlx_array_new();
        self.scales = mlx.mlx_array_new();
        self.biases = mlx.mlx_array_new();
    }
};

/// Affine quantize `dense_x` group-wise along the last axis. Returns a
/// `QuantizedKV` triple owned by the caller. Caller's input `dense_x` is
/// not consumed (refcount semantics — the caller still owns it).
pub fn quantizeAffine(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    try mlx.check(mlx.mlx_quantize(
        &vec,
        dense_x,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        s,
    ));

    // mlx-c convention: the vector contains [q, scales, biases] in that order.
    // Mirror the unpack pattern used elsewhere when consuming a
    // `mlx_vector_array` (e.g. concatenate fan-out).
    const n = mlx.mlx_vector_array_size(vec);
    if (n != 3) return error.UnexpectedQuantizeOutput;

    var out: QuantizedKV = .{
        .q = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
    };
    errdefer out.deinit();

    try mlx.check(mlx.mlx_vector_array_get(&out.q, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, vec, 2));
    return out;
}

/// Affine dequantize a `(q, scales, biases)` triple to the scales dtype. Caller
/// owns the returned array.
pub fn dequantizeAffine(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_dequantize(
        &out,
        q,
        scales,
        biases,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        // The scales carry the dtype the cache was fed: a hardcoded bf16 met an
        // f16 query in attention and widened the whole residual stream to f32.
        .{ .value = mlx.mlx_array_dtype(scales), .has_value = true },
        s,
    ));
    return out;
}

// ── Fused quant-attention path (opt-in via --kv-attn-mode fused) ──
//
// `quantAttention` reads K and V directly from their quantized triples,
// avoiding the dense materialization that `KVCache.denseView` performs in
// the default path. The fusion comes from `mlx_quantized_matmul`, Apple's
// kernel that internally dequantizes one operand and multiplies in a
// single Metal pass — same primitive `qmatmulBits` already uses for weight
// quantization throughout `transformer.zig`.
//
// Tradeoff vs `mlx_fast_scaled_dot_product_attention` (flash-attention):
//   * Dense SDPA fuses Q@K^T → scale → mask → softmax → @V into a single
//     tiled pass with no intermediate HBM writes. Wins at short context.
//   * Hand-rolled qmm × 2 + softmax loses tile-level fusion but skips the
//     dense K/V materialization. Wins at long context where K/V bandwidth
//     dominates.
// The crossover is data-driven; v1 ships behind `--kv-attn-mode fused` so the
// default is unchanged.
//
// Shape contract:
//   q_dense      : [B, H,    T_q, D] bf16 (Q already scaled or not; we apply scale below)
//   k_q/sc/bi    : K affine-quantized along last axis (D). Shape of the
//                  triple is whatever `mlx_quantize` produced.
//   v_q/sc/bi    : V same as K.
// GQA: when H > H_kv, Q is RESHAPED down to H_kv groups of g·T_q rows
// (never the triples expanded up to H_q — that materializes the packed
// bank ×g per layer per token). See the body comment in `quantAttention`.

/// A read-only borrow of a `QuantizedKV` triple — the cache owns the
/// arrays; the call site borrows them for the duration of one attention.
/// Used to thread quant triples through `DenseKVView` without altering
/// refcount semantics.
pub const BorrowedTriple = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
};

/// Hand-rolled attention that consumes K and V triples directly.
/// `scale` is the standard 1/sqrt(D) factor SDPA applies before softmax.
/// `mask_mode` mirrors `mlx_fast_scaled_dot_product_attention`:
///   * "causal": apply a Q-aligned causal mask (upper triangular -inf).
///     For T_q == 1 (decode tick) this is a no-op.
///   * "":       no mask.
///   * "array":  add `mask_arr` to the pre-softmax scores. Must be
///               additive (mlx convention: -inf for masked positions).
/// Returns a `[B, H, T_q, D]` bf16 array; caller owns and frees.
pub fn quantAttention(
    q_dense: mlx.mlx_array,
    k_triple: BorrowedTriple,
    v_triple: BorrowedTriple,
    bits: u8,
    group_size: u32,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    // GQA handling (docs/kv-quant-perf.md Phase 1). mlx_quantized_matmul
    // does not broadcast across the head dim the way
    // mlx_fast_scaled_dot_product_attention does, and expanding the K/V
    // triples to H_q materializes the packed bank ×(H_q/H_kv) per layer per
    // token (mlx_quantized_matmul makes non-contiguous operands contiguous
    // on read). Instead we reshape Q the other way:
    //   [B, H_q, T_q, D] → [B, H_kv, g·T_q, D]   where g = H_q / H_kv.
    // Pure relabel in logical row-major order: element (b, hkv·g+gi, t, d)
    // maps to row r = gi·T_q + t of block hkv. The packed triples are then
    // read IN PLACE with no expansion at all; the output reshapes back.
    // Masking under grouped rows: row r is query position r % T_q, so a
    // [T_q, T_k] mask TILED g times along rows (g-major, matching the row
    // order above) masks every group identically.
    const q_shape = mlx.getShape(q_dense);
    if (q_shape.len < 4) return error.UnexpectedQShape;
    const k_q_shape = mlx.getShape(k_triple.q);
    if (k_q_shape.len < 4) return error.UnexpectedKShape;
    const B: c_int = q_shape[0];
    const H_q: c_int = q_shape[1];
    const T_q: c_int = q_shape[2];
    const D: c_int = q_shape[3];
    const H_kv: c_int = k_q_shape[1];
    if (H_kv <= 0 or @mod(H_q, H_kv) != 0) return error.UnsupportedGqaRatio;
    const g: c_int = @divExact(H_q, H_kv);

    // Group Q down to H_kv blocks. Borrow when g == 1 (no GQA).
    var q_used = q_dense;
    var owns_q = false;
    defer if (owns_q) {
        _ = mlx.mlx_array_free(q_used);
    };
    if (g > 1) {
        const gshape = [_]c_int{ B, H_kv, g * T_q, D };
        var qg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&qg, q_dense, &gshape, 4, s));
        q_used = qg;
        owns_q = true;
    }

    // 1) scores = Q @ K^T. transpose_w=true contracts on K's quantized
    //    last axis (D), the same dim Q contracts. Output: [B, H_kv, g·T_q, T_k].
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_quantized_matmul(
        &scores,
        q_used,
        k_triple.q,
        k_triple.scales,
        k_triple.biases,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));

    // 2) scale. Fold into a single multiply rather than dividing inside
    //    softmax — saves one op per call.
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, scores, scale_arr, s));

    // 3) mask. We build masked_scores into `pre_softmax` and free the
    //    intermediates at scope exit. The branches assign `pre_softmax`
    //    to either an owned array or a borrow of `scaled`; `owns_pre`
    //    tracks which.
    var pre_softmax: mlx.mlx_array = .{ .ctx = null };
    var owns_pre = false;
    defer {
        if (owns_pre) _ = mlx.mlx_array_free(pre_softmax);
    }
    if (std.mem.eql(u8, mask_mode, "causal")) {
        // Causal mask over the UNGROUPED query axis: T_q comes from the
        // caller's Q shape, never from the (possibly grouped) scores rows.
        // For decode (T_q == 1) every grouped row is the same position and
        // the mask is identically zero — skip building it.
        const scores_shape = mlx.getShape(scaled);
        if (scores_shape.len < 2) return error.UnexpectedShape;
        const t_k: c_int = scores_shape[scores_shape.len - 1];
        if (T_q == 1) {
            pre_softmax = scaled;
        } else {
            // Build a `[T_q, T_k]` additive mask: -inf above the diagonal
            // anchored to the right edge (Q position 0 corresponds to
            // K position (T_k - T_q)).
            const offset: c_int = t_k - T_q;
            const shape2 = [_]c_int{ T_q, t_k };
            var ones2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones2);
            try mlx.check(mlx.mlx_ones(&ones2, &shape2, 2, .bfloat16, s));
            var upper = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper);
            // triu(ones, k=offset+1) yields 1s strictly above the causal
            // diagonal (positions Q can't see) and 0s on/below.
            try mlx.check(mlx.mlx_triu(&upper, ones2, offset + 1, s));
            const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(neg_inf);
            var neg_inf_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(neg_inf_bf16);
            try mlx.check(mlx.mlx_astype(&neg_inf_bf16, neg_inf, .bfloat16, s));
            // mask = where(upper, -inf, 0). NEVER `upper * -inf`: the
            // below-diagonal zeros make 0 x -inf = NaN, which poisons the
            // whole softmax (live symptom: gemma answered "<pad><pad>" —
            // caught only once the parity loops became NaN-aware).
            const half = mlx.mlx_array_new_float(0.5);
            defer _ = mlx.mlx_array_free(half);
            var upper_bool = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper_bool);
            try mlx.check(mlx.mlx_greater(&upper_bool, upper, half, s));
            const zero_f = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero_f);
            var zero_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zero_bf16);
            try mlx.check(mlx.mlx_astype(&zero_bf16, zero_f, .bfloat16, s));
            var add_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(add_mask);
            try mlx.check(mlx.mlx_where(&add_mask, upper_bool, neg_inf_bf16, zero_bf16, s));
            // Tile g-major along rows so every Q-group sees the same mask.
            var tiled_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tiled_mask);
            if (g > 1) {
                const reps = [_]c_int{ g, 1 };
                try mlx.check(mlx.mlx_tile(&tiled_mask, add_mask, &reps, 2, s));
            } else {
                try mlx.check(mlx.mlx_array_set(&tiled_mask, add_mask));
            }
            pre_softmax = mlx.mlx_array_new();
            owns_pre = true;
            try mlx.check(mlx.mlx_add(&pre_softmax, scaled, tiled_mask, s));
        }
    } else if (std.mem.eql(u8, mask_mode, "array")) {
        // External additive mask `[..., T_q, T_k]`. When Q is grouped and
        // T_q > 1 the mask's row axis must be tiled ×g (g-major). At
        // T_q == 1 the mask row broadcasts over the g grouped rows as-is.
        var mask_used = mask_arr;
        var owns_mask = false;
        defer if (owns_mask) {
            _ = mlx.mlx_array_free(mask_used);
        };
        if (g > 1 and T_q > 1) {
            const m_shape = mlx.getShape(mask_arr);
            if (m_shape.len < 2) return error.UnexpectedMaskShape;
            var reps_buf: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
            if (m_shape.len > reps_buf.len) return error.UnexpectedMaskShape;
            reps_buf[m_shape.len - 2] = g;
            var tm = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_tile(&tm, mask_arr, &reps_buf, m_shape.len, s));
            mask_used = tm;
            owns_mask = true;
        }
        pre_softmax = mlx.mlx_array_new();
        owns_pre = true;
        try mlx.check(mlx.mlx_add(&pre_softmax, scaled, mask_used, s));
    } else {
        // No mask: borrow `scaled` for the softmax input.
        pre_softmax = scaled;
    }

    // 4) softmax along last axis. `precise=true` matches the fast SDPA
    //    kernel's accumulation order (f32 reductions inside softmax) more
    //    closely than the default; trades a small perf hit for tighter
    //    equivalence to dense SDPA.
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    try mlx.check(mlx.mlx_softmax_axis(&attn, pre_softmax, -1, true, s));

    // 5) out = attn @ V. transpose_w=false contracts attn's last axis
    //    (T_k) with V's second-to-last (T_k) — V's quantized last axis
    //    (D) becomes the output dim. mlx_quantized_matmul dequantizes V
    //    on-the-fly without materializing a dense intermediate.
    var out_grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_grouped);
    try mlx.check(mlx.mlx_quantized_matmul(
        &out_grouped,
        attn,
        v_triple.q,
        v_triple.scales,
        v_triple.biases,
        false,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));

    // 6) Ungroup: [B, H_kv, g·T_q, D] → [B, H_q, T_q, D] (same relabel,
    //    inverted). No-op reshape when g == 1.
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (g > 1) {
        const oshape = [_]c_int{ B, H_q, T_q, D };
        try mlx.check(mlx.mlx_reshape(&out, out_grouped, &oshape, 4, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&out, out_grouped));
    }
    return out;
}

// ── Tests ──

const testing = std.testing;

/// Build a `[1, 1, 8, head_dim]` bf16 tensor whose values vary smoothly per
/// position so quantization error is non-trivial but bounded.
fn buildSmoothBf16(s: mlx.mlx_stream, head_dim: c_int) !mlx.mlx_array {
    const T: usize = 8;
    const D: usize = @intCast(head_dim);
    const buf = try testing.allocator.alloc(f32, T * D);
    defer testing.allocator.free(buf);
    for (0..T) |t| {
        for (0..D) |d| {
            // Range roughly [-1, 1]; smooth so adjacent group elements are
            // close (best case for affine), with some variation across groups.
            const fi: f32 = @floatFromInt(t * D + d);
            const denom: f32 = @floatFromInt(T * D);
            buf[t * D + d] = (fi / denom) * 2.0 - 1.0;
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(T), head_dim };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Read a flat float32 host buffer for a small array (eval-and-copy via
/// `mlx_astype` to float32 then reshape to 1D).
fn readF32Flat(s: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
    var f32_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_view);
    try mlx.check(mlx.mlx_astype(&f32_view, arr, .float32, s));

    const n: c_int = @intCast(mlx.mlx_array_size(f32_view));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    {
        const sh = [_]c_int{n};
        try mlx.check(mlx.mlx_reshape(&flat, f32_view, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
    const out = try allocator.alloc(f32, @intCast(n));
    @memcpy(out, ptr[0..@intCast(n)]);
    return out;
}

// ── Fused-attention validation ──
//
// The two tests below are the validation harness Phase 2 v1 relies on
// before wiring `quantAttention` into transformer.zig SDPA call sites.
// They prove (a) `mlx_quantized_matmul` semantics match dequant+matmul
// in both transpose modes, and (b) the assembled `quantAttention`
// produces logits within the same loose tolerance (0.05 max-abs-diff)
// the existing affine round-trip tests use.

/// Build a `[B, H, T, D]` dense bf16 with smooth ramped values per
/// `(b, h, t, d)` so quantization is non-trivial. Caller frees.
fn buildSmoothBHTD(s: mlx.mlx_stream, B: c_int, H: c_int, T: c_int, D: c_int) !mlx.mlx_array {
    const total: usize = @intCast(B * H * T * D);
    const buf = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(buf);
    var i: usize = 0;
    for (0..@intCast(B)) |b| {
        for (0..@intCast(H)) |h| {
            for (0..@intCast(T)) |t| {
                for (0..@intCast(D)) |d| {
                    const f: f32 = @floatFromInt(b * 17 + h * 5 + t * 3 + d);
                    const denom: f32 = @floatFromInt(total);
                    buf[i] = (f / denom) * 2.0 - 1.0;
                    i += 1;
                }
            }
        }
    }
    const shape = [_]c_int{ B, H, T, D };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Shared harness for the grouped-Q GQA parity tests: build smooth Q at
/// `[1, H_q, T_q, D]` and K/V at `[1, H_kv, T_k, D]`, quantize K/V, then
/// compare `quantAttention` against dense SDPA fed the SAME dequantized K/V
/// (no-worse-than-reference — quantization error folds into both paths;
/// what's under test is the grouped-Q contraction + mask tiling).
fn gqaParityCase(
    H_q: c_int,
    H_kv: c_int,
    T_q: c_int,
    T_k: c_int,
    bits: u8,
    mask_mode: [:0]const u8,
    mask_arr: mlx.mlx_array,
) !void {
    return gqaParityCaseD(H_q, H_kv, T_q, T_k, 64, bits, mask_mode, mask_arr);
}

fn gqaParityCaseD(
    H_q: c_int,
    H_kv: c_int,
    T_q: c_int,
    T_k: c_int,
    D: c_int,
    bits: u8,
    mask_mode: [:0]const u8,
    mask_arr: mlx.mlx_array,
) !void {
    const s = mlx.gpuStream();
    const q = try buildSmoothBHTD(s, 1, H_q, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, bits);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, bits);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, bits);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, bits);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    // MLX's fast SDPA broadcasts GQA natively (H_q vs H_kv).
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        mask_mode,
        mask_arr,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        bits,
        64,
        scale,
        mask_mode,
        mask_arr,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    // Output shape must be the ungrouped [1, H_q, T_q, D].
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(usize, 4), cand_shape.len);
    try testing.expectEqual(H_q, cand_shape[1]);
    try testing.expectEqual(T_q, cand_shape[2]);
    try testing.expectEqual(D, cand_shape[3]);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("gqaParityCase Hq={d} Hkv={d} Tq={d} Tk={d} D={d} bits={d} mode={s}: max_err={d}\n", .{ H_q, H_kv, T_q, T_k, D, bits, mask_mode, max_err });
    }
    try testing.expect(max_err < 0.05);
}
