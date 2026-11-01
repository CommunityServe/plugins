//! `MLX_SERVE_DSV41_T3_GROUP` (read at load; on by default with the t3 route, the default; off with
//! `MLX_SERVE_DSV41_ROUTED=streamer`, which
//! has no t3 hit wave to group; `0` turns it off, any other value is refused by name): the t3 route's device hit wave on
//! turboexl3's grouped banked decode. src/routed_t3.zig sets turboexl3's `Config.banked_group` to `bankedGroup()` before
//! the bank opens, so the banked GEMVs share one weight decode among up to four row-slots of one packed id (two at 2).
//! DEVROUTE's resident LUT then sends a pair whose expert no bank holds to turboexl3's skip id (`banked_skip_id`: no
//! decode, its row 0), where it sent it to base row 0 (patches/turboexl3-zzz8b-t3-group.patch). The miss part's row
//! replaces that row in the join, as it replaced base row 0's, and every other row is the banked call's byte for byte
//! (turboexl3 65c72d1's byte test), so the served bytes are the same.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = false;

pub const Error = error{T3GroupSetting};

/// At load: the setting, refused by name when it is neither unset nor `0`; on only with the t3 route.
pub fn configure(value: ?[]const u8, routed_t3: bool) Error!void {
    on = (try parse(value)) and routed_t3;
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.T3GroupSetting;
}

/// turboexl3's banked GEMVs' row-slots of one packed id per lead at the setting: 4 the grouped texts, 2 the banked ones.
pub fn bankedGroup() u32 {
    return if (on) 4 else 2;
}

test "t3_group setting: on unset, 0 off, anything else refused by name; on only with the t3 route" {
    const t = std.testing;
    try t.expect(try parse(null));
    try t.expect(try parse(""));
    try t.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "00" }) |v| try t.expectError(error.T3GroupSetting, parse(v));
    defer on = false;
    try configure(null, true);
    try t.expect(on);
    try t.expectEqual(@as(u32, 4), bankedGroup());
    try configure(null, false);
    try t.expect(!on);
    try t.expectEqual(@as(u32, 2), bankedGroup());
    try configure("0", true);
    try t.expect(!on);
    try t.expectError(error.T3GroupSetting, configure("1", true));
}
