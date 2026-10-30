//! This plugin's own, in place of mlx-serve's `src/sdk/blocks/attention/sparse_block/decode_cuda.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): QSA's CUDA decode, which declines on Metal, the one backend this plugin runs on.
const mlx = @import("../../../../mlx/ffi.zig");
