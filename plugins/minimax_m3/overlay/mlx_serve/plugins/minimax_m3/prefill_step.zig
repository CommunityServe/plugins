//! copied_from: src/plugins/minimax_m3/prefill_step.zig@9df2c52b504cce5bbf22f074dab168fd8584f961
const std = @import("std");
const Geometry = @import("geometry.zig").Config;
/// Prompt rows a span holds: one layer-major pass reads each missing expert once a span.
pub const span_tokens = 4096;
/// The bf16 activations one span row holds through one layer, from the geometry: the attention projections and their
/// normed/roped copies, the dense or shared MLP's gate, up and the eight arrays of the SwiGLU-OAI op chain, the current
/// expert batch at its worst (all four routed pairs: input, gate, up, the chain, down), the kept routed outputs and the
/// combine's concat, gather, optional bias, weighting and sums, and the residual stream.
pub fn spanRowBytes(c: *const Geometry) u64 {
    const hidden: u64 = c.hidden_size;
    const attn = 2 * (2 * @as(u64, c.n_heads) * c.head_dim + 2 * @as(u64, c.n_kv_heads) * c.head_dim + (1 + @as(u64, c.index_n_heads)) * c.index_head_dim + hidden);
    const mlp = 10 * @as(u64, @max(c.dense_intermediate_size, c.shared_intermediate_size));
    const batch = 4 * (2 * hidden + 10 * @as(u64, c.intermediate_size));
    const combine = (4 * 4 + 3 + 8) * hidden;
    return 2 * (attn + mlp + batch + combine + 6 * hidden);
}
/// A span's rows through one layer, an upper bound.
pub fn spanBytes(c: *const Geometry, seq: u64) u64 {
    return @as(u64, @min(seq, span_tokens)) *| spanRowBytes(c);
}
test "MiniMax bill prices a span row from every width the geometry accepts" {
    const released: Geometry = .{};
    try std.testing.expectEqual(@as(u64, 1092096), spanRowBytes(&released));
    var wide = released;
    wide.intermediate_size = 1 << 16;
    wide.dense_intermediate_size = 1 << 16;
    try std.testing.expect(spanRowBytes(&wide) > spanRowBytes(&released) + 2 * 4 * 10 * ((1 << 16) - 3072));
    try std.testing.expectEqual(@as(u64, span_tokens) * spanRowBytes(&released), spanBytes(&released, 1 << 20));
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
/// The scratch a sparse tile can hold over a context of `seq` tokens (any shorter prompt's tile is smaller): its rows,
/// and the f32 copy of the index keys the tile shares.
pub fn sparseTileBytes(index_heads: u64, index_dim: u64, block: u64, seq: u64) u64 {
    return @min(sparse_tile_cap, sparse_tile_rows *| sparseRowBytes(index_heads, block, seq)) +| 4 *| index_heads *| index_dim *| seq;
}
/// One row-by-row query's scratch over `tokens` keys: the f32 index keys and scores, and a host-oracle mask a head.
fn fallbackRowBytes(index_heads: u64, index_dim: u64, heads: u64, tokens: u64) u64 {
    return tokens *| (index_heads *| (4 *| index_dim +| 8) +| 2 *| heads);
}
/// Rows a row-by-row group evaluates and frees together: at most 64, at least one, inside `sparse_tile_cap`.
pub fn fallbackRows(index_heads: u64, index_dim: u64, heads: u64, tokens: u64) usize {
    return @intCast(std.math.clamp(sparse_tile_cap / @max(fallbackRowBytes(index_heads, index_dim, heads, tokens), 1), 1, 64));
}
/// The scratch a row-by-row group can hold over a context of `seq` tokens.
pub fn fallbackBytes(index_heads: u64, index_dim: u64, heads: u64, seq: u64) u64 {
    const row = fallbackRowBytes(index_heads, index_dim, heads, seq);
    return @max(row, @min(sparse_tile_cap, 64 *| row));
}
test "MiniMax bill prices a row-by-row group: 64 rows short, one row at a 1M context" {
    try std.testing.expectEqual(@as(usize, 64), fallbackRows(4, 128, 64, 512));
    try std.testing.expectEqual(@as(usize, 29), fallbackRows(4, 128, 64, 2048));
    try std.testing.expectEqual(@as(usize, 1), fallbackRows(4, 128, 64, 1 << 20));
    for ([_]u64{ 2048, 8192, 1 << 20 }) |seq| {
        try std.testing.expect(fallbackRows(4, 128, 64, seq) * fallbackRowBytes(4, 128, 64, seq) <= fallbackBytes(4, 128, 64, seq));
    }
}
test "MiniMax bill prices a sparse tile inside its cap, and past it the rows attend one by one" {
    try std.testing.expectEqual(@as(usize, 256), sparseTileRows(4, 128, 2048 + 300, 1000));
    try std.testing.expectEqual(@as(usize, 40), sparseTileRows(4, 128, 4096, 40));
    try std.testing.expectEqual(@as(usize, 0), sparseTileRows(4, 128, 1 << 20, 256));
    for ([_]u64{ 2049, 4096, 32768, 65536, 1 << 20 }) |seq| {
        const rows = sparseTileRows(4, 128, seq, sparse_tile_rows);
        try std.testing.expect(rows * sparseRowBytes(4, 128, seq) + 4 * 4 * 128 * seq <= sparseTileBytes(4, 128, 128, seq));
    }
}
/// One layer's span: the seed takes every routed occurrence once (its counts are prompt frequencies), and `order` gets
/// each distinct expert in first-use order, the batches its reads follow. Returns the distinct count.
pub fn seedSpan(stream: anytype, layer: u32, ids: []const u16, experts: u32, order: []u16) !usize {
    var seen: [256]bool = @splat(false);
    var distinct: usize = 0;
    for (ids) |id| {
        if (id >= experts or id >= seen.len) return error.MiniMaxRouteWidth;
        if (seen[id]) continue;
        seen[id] = true;
        order[distinct] = id;
        distinct += 1;
    }
    try stream.seedPrefill(layer, ids);
    return distinct;
}
test "MiniMax span seed keeps every routed occurrence: an expert routed 1000 times outweighs one routed once" {
    const Recorder = struct {
        counts: [8]u32 = @splat(0),
        calls: u32 = 0,
        pub fn seedPrefill(self: *@This(), _: u32, ids: []const u16) !void {
            self.calls += 1;
            for (ids) |id| self.counts[id] += 1;
        }
    };
    const ids = try std.testing.allocator.alloc(u16, 4004);
    defer std.testing.allocator.free(ids);
    for (ids[0..4000], 0..) |*id, i| id.* = if (i % 4 == 0) 5 else @intCast(i % 4);
    @memcpy(ids[4000..], &[_]u16{ 7, 1, 2, 3 });
    var r: Recorder = .{};
    var order: [8]u16 = undefined;
    try std.testing.expectEqual(@as(usize, 5), try seedSpan(&r, 0, ids, 8, &order));
    try std.testing.expectEqualSlices(u16, &.{ 5, 1, 2, 3, 7 }, order[0..5]);
    try std.testing.expectEqual(@as(u32, 1), r.calls);
    try std.testing.expectEqual(@as(u32, 1000), r.counts[5]);
    try std.testing.expectEqual(@as(u32, 1), r.counts[7]);
    try std.testing.expectEqual(@as(u32, 1001), r.counts[1]);
    try std.testing.expectError(error.MiniMaxRouteWidth, seedSpan(&r, 0, &.{ 1, 9 }, 8, &order));
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
