//! The plugin's log lines (the caller writes the newline) into the server's log, through the host table the plugin
//! keeps at its open (`use`); none before it.

const kit = @import("mlx_serve");

var host: ?kit.host.Host = null;

pub fn use(h: kit.host.Host) void {
    host = h;
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (host) |h| h.log(.err, fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (host) |h| h.log(.warn, fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (host) |h| h.log(.info, fmt, args);
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (host) |h| h.log(.debug, fmt, args);
}
