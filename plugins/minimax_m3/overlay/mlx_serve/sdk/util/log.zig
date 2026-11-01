//! This plugin's own, in place of mlx-serve's `src/sdk/util/log.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! copies' lines go to the server's log through the host table bound at load (the caller writes the newline), and to
//! stderr before a load binds it.

const std = @import("std");
const ffi = @import("../../mlx/ffi.zig");
const Level = @import("mlx_serve").abi.host.LogLevel;

fn emit(comptime level: Level, comptime fmt: []const u8, args: anytype) void {
    if (ffi.boundHost()) |h| return h.log(level, fmt, args);
    std.debug.print(fmt, args);
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    emit(.err, fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    emit(.warn, fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    emit(.info, fmt, args);
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    emit(.debug, fmt, args);
}
