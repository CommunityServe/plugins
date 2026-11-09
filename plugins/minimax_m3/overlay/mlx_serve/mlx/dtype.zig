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
