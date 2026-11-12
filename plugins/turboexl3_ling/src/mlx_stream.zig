//! The expert streamer as this plugin builds it: module `mlx_stream` over its own `sdk` (src/mlx_serve/sdk/weights/
//! streamed/sdk/) and the plugin's root as `mlx_host`. It is mlx-serve's `src/mlx_stream.zig` without GLM-5.3's arch,
//! plus Ling-3.1's bank source (`ling_source`, overlay/mlx_serve: the EXL3 bank at Ling's geometry, every routed call on
//! turboexl3 through the plugin's src/ling_math.zig), which Ling's trunk opens.

pub const sdk = @import("sdk");
/// The expert bank source a resident trunk streams its routed experts from (`sdk/weights/streamed.zig` `Streamed`).
pub const bank_source = @import("mlx_serve/sdk/weights/streamed/bank_source.zig");
/// A standard MLX pack's routed experts streamed as downloaded (`Streamed.openPack`).
pub const mlx_pack = @import("mlx_serve/sdk/weights/streamed/mlx_pack.zig");
/// An EXL3 expert bank streamed for a resident trunk (`sdk/weights/streamed.zig` `Exl3`).
pub const exl3_source = @import("mlx_serve/sdk/weights/streamed/exl3_source.zig");
/// Ling-3.1's EXL3 bank (the plugin's own, overlay/mlx_serve/sdk/weights/streamed/ling_source.zig).
pub const ling_source = @import("mlx_serve/sdk/weights/streamed/ling_source.zig");
/// The event shim's ABI version the streamer checks at its first event (src/host_shims.zig serves it).
pub const event_abi = @import("mlx_serve/sdk/weights/streamed/sdk_ext/expert/event.zig").abi_version;
