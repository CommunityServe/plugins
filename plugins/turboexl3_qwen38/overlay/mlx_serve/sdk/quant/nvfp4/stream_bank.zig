//! This plugin's own, in place of mlx-serve's `src/sdk/quant/nvfp4/stream_bank.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): a streaming GLM pack's NVFP4 expert bank, never a Qwen3.8 pack's.

pub fn load(weights: anytype, dir: []const u8, manifest: []const u8, dst: []const u8) !void {
    _ = weights;
    _ = dir;
    _ = manifest;
    _ = dst;
    return error.StreamBankNotServed;
}
