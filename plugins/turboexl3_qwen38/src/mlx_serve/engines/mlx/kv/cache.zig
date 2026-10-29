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

// ── Cache snapshot/restore (for spec-decode rollback) ──────────────────────

const kv_append_kv_h: c_int = 2;
const kv_append_hd: c_int = 256;

// ── KV cache growth policy (issue #110) ──────────────────────────────────────
