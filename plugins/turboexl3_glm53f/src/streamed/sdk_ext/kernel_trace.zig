//! The kernel routes' host test backend (`Trace`) and the helpers their tests share, over a pinned registry `R`
//! (G5): every launch, eval, join and graph op recorded with its inputs, no MLX. Test code only.

const std = @import("std");
const mlx = @import("sdk").mlx;
const kernel_routes = @import("kernel_routes.zig");
const QuantMode = @import("sdk").QuantMode;
