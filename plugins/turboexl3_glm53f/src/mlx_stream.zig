//! GLM-5.3-Flash's EXL3 bank as this plugin builds it: module `mlx_stream` over its own `sdk` (src/streamed/sdk/), the
//! plugin's root as `mlx_host` and the kit (`mlx_serve`), whose expert stream (`kit.streamed`) the bank streams through.
//! The bank is mlx-serve's GLM-5.3-Flash lane (src/streamed/, src/exl3/: moved out of mlx-serve at bc6c5f2c, which no
//! longer carries it): the source the glm5_next trunk's `sdk/weights/streamed.zig` `Glm5n` opens.

pub const sdk = @import("sdk");
/// GLM-5.3-Flash's EXL3 bank for the glm5_next trunk (`sdk/weights/streamed.zig` `Glm5n`).
pub const glm5n_source = @import("streamed/glm5n_source.zig");
/// The kit's stream counters (`kit.streamed.expert.Stats`), for the plugin's `[stream-stats]` lines (src/stream_stats.zig).
pub const expert_stats = @import("mlx_serve").streamed.expert.Stats;
/// The kit's usage profile (`expert-usage-v1.json`), for TURBOEXL3_GLM53F_SEED_DECODE's profile read (its size cap).
pub const expert_usage = @import("mlx_serve").streamed.expert.usage;
/// The kit's read pool (`kit.streamed.expert.io`).
pub const expert_io = @import("mlx_serve").streamed.expert.io;
/// The keep-warm engine (`gpu_warm`, TURBOEXL3_GLM53F_KEEP_WARM), for its trace proofs (src/keep_warm_test.zig).
pub const gpu_warm = @import("streamed/mimo/gpu_warm.zig");
/// The event shim's ABI version the kit's stream checks at its first event (src/host_shims.zig serves it).
pub const event_abi = @import("mlx_serve").streamed.expert.event.abi_version;
