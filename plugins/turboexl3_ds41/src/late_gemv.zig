//! `MLX_SERVE_DSV41_LATE_GEMV` (read at load; on by default, `0` turns it off, any other value is refused by name): the
//! decode lookahead's next-layer scores (the copied stream's `nextScores`, a verify's rows against the next layer's
//! router weights) as a product in the weights' half type. MLX runs that as its wide GEMV, one weight pass for the
//! verify's rows, where the f32 product took its tiled GEMM at a single row tile
//! (patches/turboexl3-zzz4-late-gemv-shared-hoist.patch). The scores only rank the experts the read-ahead reads: the
//! served bytes are the same, near-ties may rank apart. `0`: the f32 product.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{LateGemvSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.LateGemvSetting;
}

test "late_gemv setting: on unset, 0 off, anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "00" }) |v| try std.testing.expectError(error.LateGemvSetting, parse(v));
}
