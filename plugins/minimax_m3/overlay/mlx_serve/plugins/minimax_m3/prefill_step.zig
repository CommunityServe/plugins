//! copied_from: src/plugins/minimax_m3/prefill_step.zig@61ec76d7bc1abfb7f588db3bf6a94ffc0cd423c7
const std = @import("std");
/// Prompt rows a span holds: one layer-major pass reads each missing expert once a span.
pub const span_tokens = 2048;
/// A span's hidden rows through one layer: the tiles, their concatenations and four routed outputs a row.
pub fn spanBytes(hidden: u64, seq: u64) u64 {
    return @as(u64, @min(seq, span_tokens)) * 40 * hidden;
}
pub fn run(comptime Tensor: type, executor: anytype, ids: []const u32, per: usize) !Tensor {
    if (ids.len == 0) return error.MiniMaxEmptyPrompt;
    var last: ?Tensor = null;
    errdefer if (last) |value| executor.releaseStep(value);
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(ids.len, at + per);
        if (last) |value| executor.releaseStep(value);
        last = null;
        last = try executor.chunk(ids[at..end], end == ids.len);
        at = end;
    }
    return last.?;
}
