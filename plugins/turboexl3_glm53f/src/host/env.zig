//! The process environment the bank reads, by variable name (mlx-serve's registry, `env.zig`, is the host's).

const std = @import("std");

pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    return std.c.getenv(@tagName(name));
}

pub fn named(name: [:0]const u8) ?[*:0]const u8 {
    return std.c.getenv(name);
}
