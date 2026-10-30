//! This plugin's own, in place of mlx-serve's `src/sdk/blocks/dit/w8a8.zig` (a cut edge of scripts/sync-mlx-serve.sh):
//! the int8 overlay that only the media DiTs fill (`--w8a8`), which a Qwen3.8 pack's weights never carry, and the NAX
//! gate the int8 kernels read (the machine's).

const std = @import("std");

pub const Overlay = struct {
    pub fn deinit(self: *Overlay, a: std.mem.Allocator) void {
        _ = self;
        _ = a;
    }
};

pub fn nax() bool {
    return @import("../../../mlx/device.zig").naxAvailable();
}
