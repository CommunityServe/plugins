//! `MLX_SERVE_DSV41_SHARED_HOIST` (read at load; on by default, `0` turns it off, any other value is refused by name): a
//! decode layer's shared expert committed on its own as soon as the routed call returns, so it queues right behind the
//! device hit wave and runs while a layer with misses waits on their bytes. `0` keeps it in the layer's last stretch,
//! after the miss parts. The copied model splits its `ffn_post` stretch at the shared expert's down projection
//! (patches/turboexl3-zzz4-late-gemv-shared-hoist.patch): the same kernels, so the same bytes.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{SharedHoistSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.SharedHoistSetting;
}

test "shared_hoist setting: on unset, 0 off, anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "00" }) |v| try std.testing.expectError(error.SharedHoistSetting, parse(v));
}
