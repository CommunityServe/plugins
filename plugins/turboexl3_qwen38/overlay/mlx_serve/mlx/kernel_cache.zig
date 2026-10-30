//! This plugin's own, in place of mlx-serve's `src/mlx/kernel_cache.zig` (a cut edge of scripts/sync-mlx-serve.sh): a
//! few launch configs kept per key, the least recently used one rebuilt.

const std = @import("std");
const mlx = @import("ffi.zig");

/// QSA's: four slots.
pub fn QsaCfgCache(comptime Key: type, comptime NCFG: usize) type {
    return KernelCfgCache(Key, NCFG, 4);
}

/// `slots` keys, each with its `NCFG` configs (owned: a replaced entry's are freed).
pub fn KernelCfgCache(comptime Key: type, comptime NCFG: usize, comptime slots: usize) type {
    return struct {
        const Self = @This();
        pub const SLOTS: usize = slots;
        const Cfg = mlx.mlx_fast_metal_kernel_config;
        const Entry = struct {
            key: Key = std.mem.zeroes(Key),
            cfg: [NCFG]?Cfg = @splat(null),
            stamp: u64 = 0,
        };

        entries: [SLOTS]Entry = @splat(.{}),
        clock: u64 = 0,
        builds: usize = 0,

        pub fn get(self: *Self, key: Key) ?[NCFG]Cfg {
            for (&self.entries) |*e| {
                if (e.cfg[0] == null or !std.meta.eql(e.key, key)) continue;
                self.clock += 1;
                e.stamp = self.clock;
                var out: [NCFG]Cfg = undefined;
                for (e.cfg, &out) |c, *o| o.* = c.?;
                return out;
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: [NCFG]Cfg) void {
            var victim: usize = 0;
            for (&self.entries, 0..) |*e, i| {
                if (e.cfg[0] == null) {
                    victim = i;
                    break;
                }
                if (e.stamp < self.entries[victim].stamp) victim = i;
            }
            const e = &self.entries[victim];
            for (&e.cfg) |*c| if (c.*) |old| {
                _ = mlx.mlx_fast_metal_kernel_config_free(old);
            };
            e.key = key;
            for (cfg, &e.cfg) |c, *slot| slot.* = c;
            self.clock += 1;
            e.stamp = self.clock;
            self.builds += 1;
        }
    };
}
