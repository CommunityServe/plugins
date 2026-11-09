//! The process environment the streamer reads, by variable name (mlx-serve's registry, `env.zig`, is the host's).

const std = @import("std");

pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    return std.c.getenv(@tagName(name));
}

pub fn span(comptime name: @EnumLiteral()) ?[:0]const u8 {
    return if (get(name)) |v| std.mem.span(v) else null;
}
