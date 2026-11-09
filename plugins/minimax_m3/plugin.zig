//! minimax_m3: MiniMax-M3.1-Flash text from prepared packs, its 128 routed experts streamed from `experts.bin`. The
//! model is the in-tree port's (src/mlx_serve/plugins/minimax_m3) on mlx-serve's expert streamer, both generated from
//! mlx-serve by scripts/prepare.sh; its MLX calls go through the kit's mlx-c (src/mlx_serve/mlx/ffi.zig).

const kit = @import("mlx_serve");
const port = @import("src/mlx_serve/plugins/minimax_m3/plugin.zig");
const shims = @import("src/shims.zig");

pub const config = @import("config.zig");
pub const version = "0.1.0";
pub const plugin = port.plugin;
/// The port's table, its load first keeping the server's Host table for the GPU calls the copies make (src/shims.zig).
pub const table = blk: {
    var t = port.table;
    t.load = &load;
    break :blk t;
};
pub const conformanceFixture = port.conformanceFixture;

// What the copies reach through `mlx_host`, the plugin's root (as mlx-serve's own root exports them).
pub const mlx = @import("src/mlx_serve/mlx/ffi.zig");
pub const log = @import("src/mlx_serve/sdk/util/log.zig");
pub const env = @import("src/mlx_serve/env.zig");
pub const shared = @import("src/mlx_serve/sdk/util/shared.zig");
pub const plugin_profile = false;

fn load(ctx: *const kit.abi.plugin.LoadCtx, out: *kit.abi.plugin.Verdict) callconv(.c) ?*anyopaque {
    shims.bind(ctx.host) catch |e| {
        out.* = switch (e) {
            error.NoHost => kit.abi.plugin.Verdict.refuse(.unsupported, "minimax_m3: the server handed no Host table"),
            error.HostTooOld => kit.abi.plugin.Verdict.refuse(.unsupported, "minimax_m3: the server's Host table is older than this plugin's kit"),
        };
        return null;
    };
    return port.table.load(ctx, out);
}

comptime {
    _ = shims;
}

test {
    _ = port;
    _ = config;
}
