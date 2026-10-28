//! Cut edge: the one helper of sushi's transformer.zig the DeepSeek files reach (diagnostic switches).

const std = @import("std");

pub fn diagEnvOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return v[0] != 0 and v[0] != '0';
}
