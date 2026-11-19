//! The bank's own SDK module (module `sdk` of `mlx_stream`): the host's MLX layer and log, the profile switch, the
//! routed-expert quant's claim descriptions, the slot rows by phase and the read past the page cache.

/// Compile the profile probes in: the host's `zig build -Dplugin-profile=true` publishes the switch as the host root's
/// `pub const plugin_profile`. Off in every served build, and with a host that has no such switch.
pub const plugin_profile: bool = if (@hasDecl(@import("mlx_host"), "plugin_profile")) @import("mlx_host").plugin_profile else false;

pub const mlx = @import("mlx_host").mlx;
pub const log = @import("mlx_host").log;
pub const io_util = @import("io_util.zig");

const peek = @import("peek.zig");
pub const Priority = peek.Priority;
pub const Diag = peek.Diag;
pub const GroupPeek = peek.GroupPeek;
pub const LayerPeek = peek.LayerPeek;
pub const Segment = peek.Segment;

/// Slot rows per routed layer, per phase: prompt <= decode <= the layer's experts.
pub const Rows = struct { prompt: u32, decode: u32 };
