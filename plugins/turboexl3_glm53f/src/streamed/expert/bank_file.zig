//! sdk.expert.bank_file: what a bank's open does around its own manifest rules (its `adopt`): the manifest read and
//! parsed, the sidecar opened for the reader at the records' bytes, the quant claim's peek. Each failure is a refusal
//! by name. GLM-5.3's two banks open through `open`; DeepSeek-V4.1's bank opens its sidecar through `openSidecar`.

const std = @import("std");
const sdk = @import("sdk");
const Diag = sdk.Diag;
const io = @import("mlx_serve").streamed.expert.io;

/// Sets `diag` (when given) and returns `err`: a refusal by name.
pub fn refuse(diag: ?*Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// A file name with no directory part: the sidecar the manifest names, beside it.
pub fn plainName(s: []const u8) bool {
    return s.len > 0 and std.mem.indexOfScalar(u8, s, '/') == null and !std.mem.eql(u8, s, ".") and !std.mem.eql(u8, s, "..");
}

/// How the sidecar file's size must compare with the records' bytes: GLM-5.3's banks hold exactly them, DeepSeek-V4.1's
/// at least them.
pub const SizeRule = enum { exact, at_least };

/// The read pool's descriptor for `path` (`io.openUncached`: no symlink, a regular file, the page cache bypassed), its
/// size against the records' `size` under `rule`.
pub fn openSidecar(path: [:0]const u8, size: u64, rule: SizeRule, diag: ?*Diag) !io.UncachedFd {
    var errno: c_int = 0;
    const f = io.openUncached(path.ptr, &errno) catch |e| return switch (e) {
        error.NotFound => refuse(diag, error.SidecarMissing, "{s}: not found", .{path}),
        error.OpenFailed => refuse(diag, error.SidecarOpen, "{s}: open(O_RDONLY | O_NOFOLLOW) failed, errno {d}", .{ path, errno }),
        error.StatFailed => refuse(diag, error.SidecarOpen, "{s}: fstat failed", .{path}),
        error.NotRegularFile => refuse(diag, error.SidecarGeometry, "{s} is not a regular file", .{path}),
        error.NoCacheRefused => refuse(diag, error.SidecarOpen, "{s}: F_NOCACHE / F_RDAHEAD refused", .{path}),
    };
    errdefer f.close();
    switch (rule) {
        .exact => if (f.size != size) return refuse(diag, error.SidecarGeometry, "{s} is {d} B, the manifest's sidecar {d} B", .{ path, f.size, size }),
        .at_least => if (f.size < size) return refuse(diag, error.SidecarGeometry, "{s} is {d} B < {d}", .{ path, f.size, size }),
    }
    return f;
}
