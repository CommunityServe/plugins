//! The expert streamer with its EXL3 lane, as this plugin builds it (module `mlx_stream`, over its own `sdk` in
//! streamed/sdk/): the bank DeepSeek-V4.1's trunk streams its routed experts from (`exl3_source`, its schedule's parts
//! in bank/), and the settings the bank reads at its open.

pub const sdk = @import("sdk");
pub const exl3_source = @import("streamed/exl3_source.zig");
/// The event shim's ABI version the streamer checks at its first event (src/host_shims.zig serves it).
pub const event_abi = @import("mlx_serve").streamed.expert.event.abi_version;
pub const buffer_mb = @import("bank/buffer_mb.zig");
pub const decode_stop = @import("bank/decode_stop.zig");
pub const expert_usage_cfg = @import("bank/expert_usage_cfg.zig");
pub const frac = @import("bank/frac.zig");
pub const l39_last = @import("bank/l39_last.zig");
pub const tail_fuse = @import("bank/tail_fuse.zig");
