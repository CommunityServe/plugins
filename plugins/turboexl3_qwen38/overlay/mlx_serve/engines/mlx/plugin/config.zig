//! This plugin's own, in place of mlx-serve's `src/engines/mlx/plugin/config.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the engine's pack-to-table binding, which the copies' config names. This plugin builds
//! its config from its own parse and never binds one (`ModelConfig.plugin` stays null).

const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;

pub const Binding = struct {
    table: *const abi.ModelTable,
    parsed: abi.Parsed = .{},
    outside: bool = false,

    pub fn destroy(b: *Binding, a: std.mem.Allocator) void {
        b.table.free_parsed(&b.parsed);
        a.destroy(b);
    }
};
