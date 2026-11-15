//! The plugin's Metal texts (src/kernels/), embedded by name, as mlx-serve's `mlx/kernels.zig` `source` embeds a kernel
//! by its path under `src/`.

/// The file's bytes, comptime-known and typed like a string literal, so `++` and `std.fmt` compose them.
pub inline fn source(comptime name: []const u8) @TypeOf(@embedFile("kernels/" ++ name)) {
    return @embedFile("kernels/" ++ name);
}
