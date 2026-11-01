//! This plugin's own, in place of mlx-serve's `src/mlx_stream.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! expert streamer the MiniMax copies import as the module `mlx_stream`, without the archs mlx-serve's root binds.
pub const sdk = @import("sdk");
pub const sdk_ext = @import("sdk/weights/streamed/sdk_ext.zig");
pub const ops = @import("sdk/weights/streamed/ops.zig");
