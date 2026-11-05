//! turboexl3_ds41, MLX-free: whether the copied deepseek_v41.zig projects a chunk's last streams through the LM head
//! (`extend`'s and `extendBatched`'s logits). mlx-serve's own rule there is "a head with `w`": a tensor rank past 0 holds
//! none and hands its streams on. This plugin's EXL3 linears (`exl3Lin`) carry their weights in `x3` and leave `w` unset,
//! and the packs' LM head is one of them, so `w` alone would read an EXL3 head as no head and hand back the hidden
//! stream for logits. Std only, tested by `zig build test`.

const std = @import("std");

/// Whether `head` (a `Lin`: `w`, an MLX array, and `x3`, an optional EXL3 projection) is a head to apply.
pub fn projects(head: anytype) bool {
    return head.w.ctx != null or head.x3 != null;
}

// ── tests ──
const testing = std.testing;

/// `Lin`'s two members `projects` reads, with stand-in handles: `w` an MLX array's handle, `x3` an EXL3 projection.
const Lin = struct {
    w: struct { ctx: ?*anyopaque = null } = .{},
    x3: ?u8 = null,
};
var cell: u8 = 0;

test "projects: an MLX head (w set) is applied" {
    try testing.expect(projects(Lin{ .w = .{ .ctx = &cell } }));
}

test "projects: an EXL3 head (x3 set, no w) is applied, not read as a tensor rank's missing head" {
    try testing.expect(projects(Lin{ .x3 = 0 }));
}

test "projects: a tensor rank past 0 (no w, no x3) holds no head" {
    try testing.expect(!projects(Lin{}));
}
