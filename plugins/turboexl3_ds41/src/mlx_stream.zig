//! The expert streamer as this plugin builds it: module `mlx_stream` over its own `sdk` (src/mlx_serve/sdk/weights/
//! streamed/sdk/) and the plugin's root as `mlx_host`. It is mlx-serve's `src/mlx_stream.zig` without GLM-5.3's arch: the
//! sources `sdk/weights/streamed.zig` names, of which DeepSeek-V4.1's trunk opens the EXL3 bank (`exl3_source`).

pub const sdk = @import("sdk");
/// The expert bank source a resident trunk streams its routed experts from (`sdk/weights/streamed.zig` `Streamed`).
pub const bank_source = @import("mlx_serve/sdk/weights/streamed/bank_source.zig");
/// A standard MLX pack's routed experts streamed as downloaded (`Streamed.openPack`).
pub const mlx_pack = @import("mlx_serve/sdk/weights/streamed/mlx_pack.zig");
/// An EXL3 expert bank streamed for a resident trunk (`sdk/weights/streamed.zig` `Exl3`).
pub const exl3_source = @import("mlx_serve/sdk/weights/streamed/exl3_source.zig");
/// The event shim's ABI version the streamer checks at its first event (src/host_shims.zig serves it).
pub const event_abi = @import("mlx_serve/sdk/weights/streamed/sdk_ext/expert/event.zig").abi_version;
/// The decode verify's GPU timeline (profile builds: `zig build -Dplugin-profile=true`), whose layer stamps the routed
/// calls make (`Experts.run`); the plugin arms it and reads it (src/verify_tl.zig).
pub const verify_timeline = @import("mlx_serve/sdk/weights/streamed/profile/verify_timeline.zig");
