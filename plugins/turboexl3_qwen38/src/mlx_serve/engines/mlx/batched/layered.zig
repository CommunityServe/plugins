//! copied_from: src/engines/mlx/batched/layered.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/batched/layered: the engine's batched decode and verify entries, routed to the served plugin.

const std = @import("std");
const plugin_driver = @import("../plugin/driver.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const Transformer = @import("../transformer.zig").Transformer;

pub var per_slot_attn_logged: bool = false;
var gdn_batched_verify_rows_logged: c_int = 0; // widest row total logged so far
var gdn_batched_logged: bool = false;
pub var row_axis_verify_logged: bool = false;
pub var verify_fail_after_flush: ?usize = null;

pub const MAX_BATCH_ROWS = @import("../../../sdk/bases/run.zig").MAX_BATCH_ROWS;
pub var std_batched_logged: bool = false; // one-shot log guard

/// Row `i` of `[N, S, V]` logits as an owned `[1, S, V]` array, for every row.
pub fn sliceBatchRows(allocator: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array, n: usize) ![]mlx.mlx_array {
    const shape = mlx.getShape(arr);
    const out = try allocator.alloc(mlx.mlx_array, n);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    for (out, 0..) |*slot, i| {
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0 };
        const stop = [_]c_int{ i_c + 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        slot.* = mlx.mlx_array_new();
        built = i + 1;
        try mlx.check(mlx.mlx_slice(slot, arr, &start, 3, &stop, 3, &strides, 3, s));
    }
    return out;
}
