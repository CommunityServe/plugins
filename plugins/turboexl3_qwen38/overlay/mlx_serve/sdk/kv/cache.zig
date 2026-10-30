//! This plugin's own, in place of mlx-serve's `src/sdk/kv/cache.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! direct model's dense KV cache (src/direct/cache.zig), under the names the copies take.

const cache = @import("../../../direct/cache.zig");

pub const KVCache = cache.KVCache;
pub const KVCacheEntry = cache.KVCacheEntry;
pub const KVCacheSnapshot = cache.KVCacheSnapshot;
