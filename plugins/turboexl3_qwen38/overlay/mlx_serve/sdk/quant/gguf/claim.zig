//! This plugin's own, in place of mlx-serve's `src/sdk/quant/gguf/claim.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): a GGUF pack is never this plugin's (it claims a config.json with a `trunk_quant` block).

const std = @import("std");

pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, out: anytype) !bool {
    _ = io;
    _ = allocator;
    _ = model_dir;
    _ = out;
    return false;
}
