//! turboexl3's `sdk` (its README's Wiring) in this plugin, a module of its own (build.zig): `mlx` is the plugin's
//! mlx-c layer (src/mlx_serve/mlx/ffi.zig), read through `mlx_host`, the plugin's root, so no file is in two modules;
//! `quant_abi` is the quant ABI the copies use (#1005's, over the kit's types); `kernel_launch` is turboexl3's own launcher over the server's host table
//! (`turboexl3_kit`); `steel` is MLX's steel headers at the server's MLX (src/mlx_steel/), as #1005 hands them. `shared`
//! is the plugin's plain globals (src/mlx_serve/sdk/util/shared.zig), one per name: turboexl3's `Global` without it is
//! a struct that captures only its type and initial value, which Zig interns as one storage for every name of that
//! type (`gdn_out_kernel` and `attn_io_prep4_kernel` shared one slot, and the prepare launched the GDN tail's kernel).

pub const mlx = @import("mlx_host").mlx;
pub const shared = @import("mlx_host").shared;
pub const quant_abi = @import("mlx_host").quant_abi;
pub const kernel_launch = @import("turboexl3_kit").kernel_launch;
pub const steel = @import("mlx_steel_sources");
