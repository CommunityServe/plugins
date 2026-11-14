//! The bank's log lines (the caller writes the newline) into the server's log, and the host's headroom (C5), through the
//! host table the plugin keeps at its load (`use`); none before it.

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

/// The host's headroom (`Host.memoryCeiling`: its ceiling less what is resident, its KV included); null before `use`.
pub fn headroom() ?u64 {
    return if (host) |h| h.memoryCeiling() else null;
}
