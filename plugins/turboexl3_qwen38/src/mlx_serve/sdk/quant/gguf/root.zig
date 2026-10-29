//! This plugin's own, in place of mlx-serve's `src/sdk/quant/gguf/root.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! this plugin loads no GGUF weights (its loader's GGUF branch, sdk/quant/gguf/claim.zig, loads none), so no weight
//! carries GGUF's encoding and the copies' matmuls never dispatch to its kernels.

const mlx = @import("../../../mlx/ffi.zig");

pub const kernels = struct {
    pub const Info = struct {};

    pub fn infoOf(w: mlx.mlx_array, sc: mlx.mlx_array) ?Info {
        _ = w;
        _ = sc;
        return null;
    }

    pub fn linear(info: Info, x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        _ = info;
        _ = x;
        _ = w;
        _ = s;
        return error.GgufNotServed;
    }

    pub fn gatherLinear(info: Info, x: mlx.mlx_array, w: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        _ = info;
        _ = x;
        _ = w;
        _ = lhs_idx;
        _ = rhs_idx;
        _ = s;
        return error.GgufNotServed;
    }
};
