//! The expert streamer's own SDK (module `sdk`): MLX, the host's log, environment and process globals as this plugin
//! serves them, the memory ledgers, the pack peeks, file helpers and the comptime interface checks.

/// The profile probes (mlx-serve's `-Dplugin-profile`), never built into this plugin.
pub const plugin_profile = false;

pub const mlx = @import("mlx_serve").mlx;
pub const log = @import("log.zig");
pub const env = @import("env.zig");
pub const shared = @import("shared.zig");
pub const io_util = @import("io_util.zig");

const peek = @import("peek.zig");
pub const Priority = peek.Priority;
pub const Diag = peek.Diag;
pub const ConfigPeek = peek.ConfigPeek;
pub const GroupPeek = peek.GroupPeek;
pub const LayerPeek = peek.LayerPeek;
pub const Segment = peek.Segment;

/// Process and box memory readings (the kernel's ledgers).
pub const memory = @import("memory.zig");
pub const QuantMode = @import("quant_mode.zig").QuantMode;

/// The comptime interface checks the kinds run on a plugin's namespaces.
pub const check = @import("check.zig");
