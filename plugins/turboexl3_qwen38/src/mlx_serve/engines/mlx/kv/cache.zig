//! copied_from: src/engines/mlx/kv/cache.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/kv/cache: the KV cache lives in the SDK (`sdk/kv/cache.zig`); its tests stay here.

const std = @import("std");
const sdk_cache = @import("../../../sdk/kv/cache.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../cache/kv_quant.zig");
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const qsaArraysAllEqual = @import("../../../sdk/testing/parity.zig").qsaArraysAllEqual;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const DenseKVView = @import("../../../sdk/blocks/attention/kv_view.zig").DenseKVView;
const bf16BitsEqual = @import("../../../sdk/testing/parity.zig").bf16BitsEqual;
const mlxDeviceUsable = @import("../../../sdk/testing/parity.zig").mlxDeviceUsable;
const testRandWeightBf16 = @import("../../../sdk/testing/parity.zig").testRandWeightBf16;
const SLIDING_TRIM_UNBOUNDED = @import("sliding.zig").SLIDING_TRIM_UNBOUNDED;
const slidingTailSpan = @import("sliding.zig").slidingTailSpan;
const slidingViewFor = @import("sliding.zig").slidingViewFor;
const attn256_vars = @import("../../../sdk/blocks/attention/attn256.zig");

pub const KVCacheEntry = sdk_cache.KVCacheEntry;
pub const newEmptyKVEntry = sdk_cache.newEmptyKVEntry;
pub const resetKVEntry = sdk_cache.resetKVEntry;
pub const freeKVEntry = sdk_cache.freeKVEntry;
pub const KVCache = sdk_cache.KVCache;
pub const KVCacheSnapshot = sdk_cache.KVCacheSnapshot;

// ── Tests ──

pub const testing = std.testing;

// ── tests ──

/// Helper: create a dummy K or V tensor of shape [1, 1, seq_len, 1].
fn testKV(seq_len: usize, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 1, sl, 1 };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

fn testKVWide(seq_len: usize, head_dim: c_int, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 2, sl, head_dim };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

// ── Cache snapshot/restore (for spec-decode rollback) ──────────────────────

const kv_append_kv_h: c_int = 2;
const kv_append_hd: c_int = 256;

fn kvAppendPacked(n: c_int, seq: c_int, seed: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const nn: usize = @intCast(n * kv_append_kv_h * seq * kv_append_hd);
    const buf = try testing.allocator.alloc(f32, nn);
    defer testing.allocator.free(buf);
    var st: u32 = seed;
    for (buf) |*x| {
        st = st *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(st >> 16)) / 65536.0 - 0.5;
    }
    const shape = [_]c_int{ n, kv_append_kv_h, seq, kv_append_hd };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
    return out;
}

fn kvAppendPrefill(cache: *KVCache, tokens: usize, seed: u32, s: mlx.mlx_stream) !void {
    const k = try kvAppendPacked(1, @intCast(tokens), seed, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try kvAppendPacked(1, @intCast(tokens), seed ^ 0x9E3779B9, s);
    defer _ = mlx.mlx_array_free(v);
    var dv = try cache.update(0, k, v, s, 0);
    dv.deinit();
}

fn kvAppendOperandsEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.offset, b.offset);
    try testing.expect(try qsaArraysAllEqual(a.keys, b.keys, s));
    try testing.expect(try qsaArraysAllEqual(a.values, b.values, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_scales, b.keys_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_biases, b.keys_biases, s));
    try testing.expect(try qsaArraysAllEqual(a.values_scales, b.values_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.values_biases, b.values_biases, s));
}

fn kvAppendDummyAttn(cache: *KVCache, s: mlx.mlx_stream) !mlx.mlx_array {
    var dv = try cache.denseView(0, s);
    defer dv.deinit();
    const ksh = mlx.getShape(dv.k);
    const q_shape = [_]c_int{ ksh[0], ksh[1], 1, ksh[3] };
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_ones(&q, &q_shape, 4, .bfloat16, s));
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_append_hd)));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &out,
        q,
        dv.k,
        dv.v,
        scale,
        "",
        none,
        .{ .ctx = null },
        false,
        s,
    ));
    _ = mlx.mlx_array_free(q);
    return out;
}

// ── KV cache growth policy (issue #110) ──────────────────────────────────────

/// Bit equality of two strided views (cache slices are not row-contiguous).
fn ringViewsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    const ca = try materializedOwnedCopy(s, a);
    defer _ = mlx.mlx_array_free(ca);
    const cb = try materializedOwnedCopy(s, b);
    defer _ = mlx.mlx_array_free(cb);
    return bf16BitsEqual(ca, cb);
}
