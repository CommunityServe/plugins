//! copied_from: src/plugins/minimax_m3/serial_step.zig@61ec76d7bc1abfb7f588db3bf6a94ffc0cd423c7
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
