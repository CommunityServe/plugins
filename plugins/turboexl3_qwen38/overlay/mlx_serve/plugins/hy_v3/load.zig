//! This plugin's own, in place of mlx-serve's `src/plugins/hy_v3/load.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! the engine's layered loader asks it for a Hunyuan-V3 pack's expert names, which a Qwen3.8 pack never is.

pub fn hy3ExpertContainer(weights: anytype, buf: *[256]u8, prefix: []const u8, layer: u32) []const u8 {
    _ = weights;
    _ = buf;
    _ = prefix;
    _ = layer;
    return "";
}
