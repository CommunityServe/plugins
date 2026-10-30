//! The plugin's GPU tests (`zig build gpu-test -Dmlx-lib=<dir>`, against the server's own MLX), by name in build.zig's
//! `gpu_filters`: #1005's tests of the EXL3 path the direct model still calls in the copies (the trunk_quant key, the
//! trunk format's load and the side stream). The MLX-free steps build none of these: `mlx-free-test`'s filters leave
//! them out, and `zig build test` roots config.zig.

test {
    _ = @import("mlx_serve/sdk/quant/exl3/pack_keys.zig");
    _ = @import("mlx_serve/plugins/qwen4_exp/trunk/formats.zig");
    _ = @import("mlx_serve/plugins/qwen4_exp/side_stream.zig");
}
