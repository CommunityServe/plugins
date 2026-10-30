//! This plugin's own, in place of mlx-serve's `src/sdk/quant/nvfp4/modelopt_pack.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): an NVIDIA ModelOpt pack is never this plugin's (`ModelConfig.modelopt` stays false).

pub fn adapt(weights: anytype, model_dir: []const u8, mtp_layer: ?u32, clamped: bool, span: anytype, shard: anytype) !void {
    _ = weights;
    _ = model_dir;
    _ = mtp_layer;
    _ = clamped;
    _ = span;
    _ = shard;
    return error.ModelOptPackNotServed;
}

pub fn skipsAtLoad(key: []const u8) bool {
    _ = key;
    return false;
}

pub fn bankExpansionBytes(model_dir: []const u8) u64 {
    _ = model_dir;
    return 0;
}
