//! `mlx_host` for turboexl3's own tests (build.zig `gpu-test`): what src/turboexl3_sdk.zig reads from the plugin's root
//! (the kit's MLX layer, the copies' globals, the quant ABI) and nothing else, so the test binary holds none of the
//! plugin's copies. No `sushi_exl3` here: the tests' Sushi comparisons skip.

const kit = @import("mlx_serve");
pub const mlx = kit.mlx;
pub const shared = @import("mlx_serve/sdk/util/shared.zig");
pub const quant_abi = kit.abi.quant;
