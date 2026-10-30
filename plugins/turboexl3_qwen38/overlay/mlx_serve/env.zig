//! This plugin's own, in place of mlx-serve's `src/env.zig` (a cut edge of scripts/sync-mlx-serve.sh): no environment
//! variable is a lever here. Every accessor answers as its variable unset would, so each path in the copies takes its
//! coded default, the one #1005 measured. The accessors keep mlx-serve's names and types, so the copies compile unchanged.

const std = @import("std");

pub fn get(comptime name: @EnumLiteral()) ?[*:0]const u8 {
    _ = name;
    return null;
}

pub fn named(comptime n: []const u8) ?[*:0]const u8 {
    _ = n;
    return null;
}

pub fn span(comptime name: @EnumLiteral()) ?[:0]const u8 {
    _ = name;
    return null;
}

pub fn on(comptime name: @EnumLiteral()) bool {
    _ = name;
    return false;
}

pub fn present(comptime name: @EnumLiteral()) bool {
    _ = name;
    return false;
}

pub fn one(comptime name: @EnumLiteral()) bool {
    _ = name;
    return false;
}

pub fn int(comptime name: @EnumLiteral(), comptime T: type, default: T) T {
    _ = name;
    return default;
}

pub fn float(comptime name: @EnumLiteral(), default: f32) f32 {
    _ = name;
    return default;
}

pub fn onCached(cache: *?bool, comptime name: @EnumLiteral()) bool {
    _ = name;
    cache.* = false;
    return false;
}

pub fn byName(name: [*:0]const u8) ?[*:0]const u8 {
    _ = name;
    return null;
}

pub fn diagAny(comptime prefix: []const u8) bool {
    _ = prefix;
    return false;
}

/// mlx-serve's reading of a raw value (the copies pass `get`'s, always null here): set and not starting with `0`.
pub fn onValue(raw: ?[*:0]const u8) bool {
    const v = std.mem.span(raw orelse return false);
    return v.len == 0 or v[0] != '0';
}

test "no variable is a lever: one set in the process environment reads as unset" {
    const c = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    try std.testing.expectEqual(@as(c_int, 0), c.setenv("MLX_SERVE_GDN_STATE_F32", "0", 1));
    defer _ = c.unsetenv("MLX_SERVE_GDN_STATE_F32");
    try std.testing.expectEqual(@as(c_int, 0), c.setenv("LAYER_CAP", "3", 1));
    defer _ = c.unsetenv("LAYER_CAP");
    try std.testing.expect(get(.MLX_SERVE_GDN_STATE_F32) == null and span(.MLX_SERVE_GDN_STATE_F32) == null);
    try std.testing.expect(!on(.MLX_SERVE_GDN_STATE_F32) and !present(.MLX_SERVE_GDN_STATE_F32) and !one(.LAYER_CAP));
    try std.testing.expectEqual(@as(u32, 7), int(.LAYER_CAP, u32, 7));
    try std.testing.expect(byName("LAYER_CAP") == null and named("LAYER_CAP") == null);
}
