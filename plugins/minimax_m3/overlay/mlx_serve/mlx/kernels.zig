//! This plugin's own, in place of mlx-serve's `src/mlx/kernels.zig`: kernel text embedded at compile time from the
//! files scripts/sync-mlx-serve.sh copies beside the slice, named by their path from mlx-serve's `src/`. No override
//! directory: the text built in is the text MLX compiles.

/// The file's bytes, comptime-known and typed like a string literal, so `++` and `std.fmt` compose them.
pub inline fn source(comptime path: []const u8) @TypeOf(@embedFile("../" ++ path)) {
    return @embedFile("../" ++ path);
}
