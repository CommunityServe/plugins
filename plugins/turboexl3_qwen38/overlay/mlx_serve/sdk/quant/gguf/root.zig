//! This plugin's own, in place of mlx-serve's `src/sdk/quant/gguf/root.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! the copies' `engines/mlx/load/weights.zig` asks `kernels.infoOf` whether a `.biases` weight sits beside a raw ggml
//! table (uint8 weight, bool scales sentinel) before it reads the biases. A turboexl3 pack holds no ggml tables, so the
//! answer is always no, and the GGUF kernels stay out of this dylib.

const mlx = @import("../../../mlx/ffi.zig");

pub const kernels = struct {
    pub fn infoOf(w: mlx.mlx_array, sc: mlx.mlx_array) ?void {
        _ = w;
        _ = sc;
        return null;
    }
};
