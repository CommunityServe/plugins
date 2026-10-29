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

fn mx(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn ab(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

// ── tests ──

const testing = std.testing;
const conformance = @import("../../../sdk/testing/cache_ops.zig");

fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try testing.allocator.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
    }
    testing.allocator.free(es);
}
