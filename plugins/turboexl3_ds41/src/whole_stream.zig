//! `MLX_SERVE_DSV41_WHOLE_STREAM` (read at load; on by default, `0` turns it off; any other value is refused by name): a
//! wide prompt layer takes the last wide layer's whole stream and FFN mix as they are, not the concatenation of its
//! spans' slices of them.
//!
//! `0`: the copied model's `wideLayer` starts by concatenating its spans' streams (`[L, hc, D]`: 16K x 4 x 5120 words)
//! and mixes into fresh arrays, though the last wide layer's one MoE group left every span holding a row slice of one
//! array: a copy and a fresh buffer a layer. On, the layer's last step also keeps that array and the mix
//! (`Spans.whole`, two held slots), and the next wide layer takes them when every span still holds its slice: this layer
//! writes no Engram rows (layers 1 and 14 do, and concatenate as before) and nothing replaced a span since (an exchange
//! clears the slots; a layer of several MoE groups leaves them empty). The words are the concatenation's.
//!
//! Read on the M5 Max (ws1, w-pio-20261101-0737: a throwaway warm server, then off / on / on / off, each a cold
//! code-ctx16384 card request and a new-text 16K one): cold prefill 1125.7 and 1126.8 tok/s off, 1132.3 and 1127.8 on
//! (+0.3%); new text 1104.9 and 1130.3 off, 1116.8 and 1137.2 on (+0.8%). The seeded text is the same in every arm.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{WholeStreamSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.WholeStreamSetting;
}

test "whole_stream setting: on unset or empty, 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "1", "on", "2", "01", "true" }) |v| try std.testing.expectError(error.WholeStreamSetting, parse(v));
}
