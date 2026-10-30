//! This plugin's own, in place of mlx-serve's `src/mlx/dtype.zig` (a cut edge of scripts/sync-mlx-serve.sh): a
//! load-time constant table handed back in the dtype of the activation it scales, so a product never widens the stream.

const mlx = @import("ffi.zig");

/// The one narrowed copy of a table, with its dtype.
pub const DtypeCastCache = struct {
    arr: mlx.mlx_array = .{ .ctx = null },
    dtype: mlx.mlx_dtype = .float32,

    pub fn deinit(self: *DtypeCastCache) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }
};

/// `table` itself when it is in `want`, else its copy in `want`, made (and evaluated) once: borrowed from `cache`.
pub fn constTableAs(table: mlx.mlx_array, want: mlx.mlx_dtype, cache: *DtypeCastCache, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(table) == want) return table;
    if (cache.arr.ctx != null and cache.dtype == want) return cache.arr;
    cache.deinit();
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, table, want, s));
    try mlx.check(mlx.mlx_array_eval(cast));
    cache.* = .{ .arr = cast, .dtype = want };
    return cast;
}
