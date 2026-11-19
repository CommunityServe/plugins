//! turboexl3's `sdk` (its README's Wiring) in this plugin, a module of its own (build.zig): `mlx` is the kit's MLX layer,
//! the one the copies and the streamer use, read through `mlx_host`, the plugin's root; `quant_abi` is the kit's quant
//! ABI; `kernel_launch` is turboexl3's own launcher over the server's host table (`turboexl3_kit`); `steel` is MLX's steel
//! headers as the kit ships them. `shared` is the plugin's plain globals (src/mlx_serve/sdk/util/shared.zig), one per
//! name, so each of turboexl3's `Global`s has storage of its own.

pub const mlx = @import("mlx_host").mlx;
pub const shared = @import("mlx_host").shared;
pub const quant_abi = @import("mlx_host").quant_abi;
pub const kernel_launch = @import("turboexl3_kit").kernel_launch;
pub const steel = @import("mlx_steel_sources");
