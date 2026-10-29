//! MLX's steel NAX tile headers at the server's MLX (src/mlx_steel/MLX_COMMIT, copied unchanged), a module of its own
//! (`mlx_steel_sources`, build.zig): the kernels that reuse MLX's tile matmul in their own Metal source read them
//! (mlx-serve's lane_qmm, turboexl3's prefill GEMMs). Each header uses only the ones before it; the consumers strip
//! their `#include "mlx/...` lines.

pub const defines: []const u8 = @embedFile("mlx/backend/metal/kernels/steel/defines.h");
pub const type_traits: []const u8 = @embedFile("mlx/backend/metal/kernels/steel/utils/type_traits.h");
pub const integral_constant: []const u8 = @embedFile("mlx/backend/metal/kernels/steel/utils/integral_constant.h");
pub const nax: []const u8 = @embedFile("mlx/backend/metal/kernels/steel/gemm/nax.h");
