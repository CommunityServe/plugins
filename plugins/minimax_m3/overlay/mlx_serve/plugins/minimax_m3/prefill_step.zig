//! copied_from: src/plugins/minimax_m3/prefill_step.zig@37e90382b439f163474f99d3c78522ae88caaf12
const std = @import("std");
pub const tokens_per_chunk = 8;
pub fn run(comptime Tensor: type, executor: anytype, ids: []const u32) !Tensor {
    if (ids.len == 0) return error.MiniMaxEmptyPrompt;
    var last: ?Tensor = null;
    errdefer if (last) |value| executor.releaseStep(value);
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + tokens_per_chunk);
        if (last) |value| executor.releaseStep(value);
        last = null;
        last = try executor.chunk(ids[at..end], end == ids.len);
        at = end;
    }
    return last.?;
}
