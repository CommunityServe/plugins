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

// ── Tests ──

const testing = std.testing;

// ── Fused-attention validation ──
//
// The two tests below are the validation harness Phase 2 v1 relies on
// before wiring `quantAttention` into transformer.zig SDPA call sites.
// They prove (a) `mlx_quantized_matmul` semantics match dequant+matmul
// in both transpose modes, and (b) the assembled `quantAttention`
// produces logits within the same loose tolerance (0.05 max-abs-diff)
// the existing affine round-trip tests use.
