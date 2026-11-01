//! copied_from: src/plugins/minimax_m3/serial_step.zig@90fb57282d374c0989a46503b12f4fd9be300963
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
