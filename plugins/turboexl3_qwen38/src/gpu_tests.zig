//! The plugin's GPU tests (`zig build gpu-test -Dmlx-lib=<dir>`, against the server's own MLX): #1005's tests over the
//! code this plugin copies from it. The kernel contract holds every kernel of the areas this plugin ships (the
//! registry's `areas`, cut to them by scripts/sync-mlx-serve.sh) to its gates on the Metal device, the controls going
//! red; the GDN fold's replay, the hc decode reads and the MTP head's draft rerank run their own. The copies hold
//! mlx-serve's other tests too: build.zig's `gpu_filters` picks #1005's. The MLX-free steps build none of these:
//! `mlx-free-test`'s filters leave them out, and `zig build test` roots config.zig.

test {
    _ = @import("mlx_serve/sdk/testing/kernel_contract/registry.zig");
    _ = @import("mlx_serve/sdk/blocks/gdn/decode_fused_test.zig");
    _ = @import("mlx_serve/sdk/blocks/hc/hc_decode2_test.zig");
    _ = @import("mlx_serve/engines/mlx/spec/mtp_test.zig");
}
