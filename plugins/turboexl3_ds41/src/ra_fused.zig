//! `MLX_SERVE_DSV41_RA_FUSED` (read at load; on by default, `0` turns it off, any other value is refused by name): a
//! wide prompt layer's read-ahead prediction on the attention side's fused hc kernels. The prediction (the router over
//! the stream collapsed with the previous mix and normed for the FFN) is the same collapse of the same rows with the same
//! mix as the attention's input, normed by the other weight: the copied model's `wideLayer` runs the mix kernel first and
//! takes the prediction from one more pass of the norm kernel (patches/turboexl3-zz9-ra-fused.patch), where the op chain
//! (the stream to f32, a batched matmul, a norm) built it. The attention's input and every routed call are the same
//! kernels over the same arrays: only the experts the read-ahead reads may round apart. `0`: the op chain.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{RaFusedSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.RaFusedSetting;
}

test "ra_fused setting: on unset, 0 off, anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "00" }) |v| try std.testing.expectError(error.RaFusedSetting, parse(v));
}
