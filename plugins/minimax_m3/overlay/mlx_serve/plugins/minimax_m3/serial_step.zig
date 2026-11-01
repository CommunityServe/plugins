//! copied_from: src/plugins/minimax_m3/serial_step.zig@1614df2919f0d76ed61f90b7b8c8bc3e75905144
const std = @import("std");
pub fn run(comptime Tensor: type, executor: anytype, ids: []const u32) !Tensor {
    if (ids.len == 0) return error.MiniMaxEmptyStep;
    var last: ?Tensor = null;
    errdefer if (last) |value| executor.releaseStep(value);
    for (ids, 0..) |id, index| {
        if (last) |value| executor.releaseStep(value);
        last = null;
        last = try executor.token(id, index + 1 == ids.len);
    }
    return last.?;
}
