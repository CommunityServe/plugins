//! `MLX_SERVE_DSV41_VISION_LAZY` (read at load; on by default, `0` turns it off, any other value is refused by
//! name): the vision tower (`vision.*`, `aligner.*`, `image_*`: 0.94 GB in the full pack) left out of the load
//! and read from the pack for each image, then freed. The streamed bank, sized from the memory in use when it opens,
//! takes those bytes as slot rows. For an image, the copied model's media hook first puts the bank at its prompt rows (the
//! prompt's working set, unused until its wide calls, holds the tower; a decode call after it grows the bank back), loads
//! the tower's weights, encodes, evaluates the rows and frees the tower before it returns
//! (patches/turboexl3-zzz8d-embed-vision.patch). The tower's arithmetic is the resident tower's, so an image's rows are
//! the same.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{VisionLazySetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.VisionLazySetting;
}

test "vision_lazy setting: on unset, 0 off, anything else refused by name" {
    const t = std.testing;
    try t.expect(try parse(null));
    try t.expect(try parse(""));
    try t.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "00" }) |v| try t.expectError(error.VisionLazySetting, parse(v));
}
