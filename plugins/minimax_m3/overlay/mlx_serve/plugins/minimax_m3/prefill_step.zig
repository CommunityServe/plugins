//! copied_from: src/plugins/minimax_m3/prefill_step.zig@b20ebdd16e8e262f5e13f2b24ef33cc4ad09aa1f
const std = @import("std");
/// Prompt rows a span holds: one layer-major pass reads each missing expert once a span.
pub const span_tokens = 4096;
/// A span's hidden rows through one layer: the tiles, their concatenations and four routed outputs a row.
pub fn spanBytes(hidden: u64, seq: u64) u64 {
    return @as(u64, @min(seq, span_tokens)) * 40 * hidden;
}
/// Query rows one sparse attention tile batches at most; `sparse_tile_cap` bounds its scratch.
pub const sparse_tile_rows = 256;
pub const sparse_tile_cap: u64 = 128 << 20;
/// Fewer rows than this a tile and the rows attend one by one.
pub const sparse_tile_min = 8;
/// A sparse row's index scores and key masks (`16 * tokens`) and its block ranks (`10 * blocks^2`), each index head.
fn sparseRowBytes(index_heads: u64, block: u64, tokens: u64) u64 {
    const blocks = (tokens + block - 1) / @max(block, 1);
    return index_heads *| (16 *| tokens +| 10 *| blocks *| blocks);
}
/// Rows a sparse tile takes over `tokens` keys: 0 when fewer than `sparse_tile_min` fit `sparse_tile_cap`.
pub fn sparseTileRows(index_heads: u64, block: u64, tokens: u64, rows: usize) usize {
    const fit = sparse_tile_cap / @max(sparseRowBytes(index_heads, block, tokens), 1);
    const n = @min(@as(u64, rows), @min(sparse_tile_rows, fit));
    return if (n < sparse_tile_min) 0 else @intCast(n);
}
/// The scratch a sparse tile can hold over a context of `seq` tokens (any shorter prompt's tile is smaller).
pub fn sparseTileBytes(index_heads: u64, block: u64, seq: u64) u64 {
    return @min(sparse_tile_cap, sparse_tile_rows *| sparseRowBytes(index_heads, block, seq));
}
test "MiniMax bill prices a sparse tile inside its cap, and past it the rows attend one by one" {
    try std.testing.expectEqual(@as(usize, 256), sparseTileRows(4, 128, 2048 + 300, 1000));
    try std.testing.expectEqual(@as(usize, 40), sparseTileRows(4, 128, 4096, 40));
    try std.testing.expectEqual(@as(usize, 0), sparseTileRows(4, 128, 1 << 20, 256));
    for ([_]u64{ 2049, 4096, 32768, 65536, 1 << 20 }) |seq| {
        const rows = sparseTileRows(4, 128, seq, sparse_tile_rows);
        try std.testing.expect(rows * sparseRowBytes(4, 128, seq) <= sparseTileBytes(4, 128, seq));
        try std.testing.expect(sparseTileBytes(4, 128, seq) <= sparse_tile_cap);
    }
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
