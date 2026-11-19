//! Reads past the page cache (`noCache`): the plugin's one I/O helper (from #749's host io_util).

const std = @import("std");
const builtin = @import("builtin");

pub const NoCacheOptions = struct {
    /// Keep the kernel's read-ahead (a file streamed front to back).
    read_ahead: bool = false,
    /// Open a symlink's target; false refuses a symlink (ELOOP).
    follow_symlinks: bool = true,
};

/// Reads of `fd` bypass the page cache (F_NOCACHE); read-ahead off unless asked.
/// Darwin only: elsewhere it refuses, so no caller silently reads through the cache.
pub fn noCache(fd: std.c.fd_t, opts: NoCacheOptions) error{ NoCacheFcntl, NoCacheUnsupported }!void {
    if (comptime !builtin.os.tag.isDarwin()) return error.NoCacheUnsupported;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.NoCacheFcntl;
    if (!opts.read_ahead and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
}
