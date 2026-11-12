//! The plugin root's names that the copied streamer and its `sdk` read through `mlx_host`, as the streamer's MLX-free test
//! builds them (src/ling_streamer_test.zig, build.zig). They come from the same files plugin.zig exports: MLX's types from
//! the plugin kit, and the copies' log, env, io_util, process-wide state and MTP acceptance modes, and the run-path marks
//! (src/ling_ran.zig: the copied stream marks LOOKAHEAD_ROWS). The trace backend reaches no MLX call.
const kit = @import("mlx_serve");
pub const mlx = kit.mlx;
pub const log = @import("mlx_serve/sdk/util/log.zig");
pub const env = @import("mlx_serve/env.zig");
pub const io_util = @import("mlx_serve/sdk/util/io_util.zig");
pub const shared = @import("mlx_serve/sdk/util/shared.zig");
pub const mtp_acceptance = @import("mlx_serve/sdk/spec/mtp_acceptance.zig");
pub const quant_abi = kit.abi.quant;
pub const ling_ran = @import("ling_ran.zig");
