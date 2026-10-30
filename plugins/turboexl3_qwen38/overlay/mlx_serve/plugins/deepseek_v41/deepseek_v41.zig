//! This plugin's own, in place of mlx-serve's `src/plugins/deepseek_v41/deepseek_v41.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the copies' config names a DeepSeek-V4.1 Thunderbolt pair, which a Qwen3.8 pack never
//! has (`ModelConfig.dist_dsv41` stays null).

pub const Pair = struct {
    lo: usize,
    hi: usize,
    worker: bool,

};
