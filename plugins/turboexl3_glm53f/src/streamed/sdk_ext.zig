//! What the bank adds beside its own `sdk`: the kit's expert source, the routed-expert quant contract, the launch
//! cache for MLX custom kernels and a host-side strided copy.

/// G6: the expert source's record layout, source contract, slot types and the process's one reader.
pub const expert = @import("mlx_serve").streamed.expert;
/// Every MLX custom kernel's launch through the plugin's launch cache (no per-call source rebuild and hash).
pub const metal_launch = @import("sdk_ext/metal_launch.zig");
/// C2: the routed-expert quant contract.
pub const quant = @import("sdk_ext/quant.zig");
/// A strided copy out of an evaluated array's buffer.
pub const ops = @import("sdk_ext/ops.zig");
