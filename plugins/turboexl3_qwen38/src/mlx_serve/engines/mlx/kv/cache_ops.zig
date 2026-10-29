//! copied_from: src/engines/mlx/kv/cache_ops.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The engine's `CacheOps` over its `KVCache` and SSM entries (`sdk/kv/cache_ops.zig`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const abi = @import("mlx_serve").abi.plugin;
const KVCache = @import("cache.zig").KVCache;
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const adapter = @import("../../../sdk/kv/cache_ops.zig").For(SSMCacheEntry);

pub const Slot = adapter.Slot;
pub const ops = adapter.ops;

// ── tests ──

const testing = std.testing;
const conformance = @import("../../../sdk/testing/cache_ops.zig");
