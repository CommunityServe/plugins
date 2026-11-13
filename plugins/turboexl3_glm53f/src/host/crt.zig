//! The POSIX file calls the MTP bank's reader makes (macOS only, as the bank).

const std = @import("std");

pub fn openRead(path: [*:0]const u8) c_int {
    return std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
}

pub fn close(fd: c_int) void {
    _ = std.c.close(fd);
}

pub fn seekEnd(fd: c_int) i64 {
    return std.c.lseek(fd, 0, std.c.SEEK.END);
}

/// Bytes read, or <= 0 on error.
pub fn pread(fd: c_int, buf: [*]u8, len: usize, off: i64) isize {
    return std.c.pread(fd, buf, len, off);
}
