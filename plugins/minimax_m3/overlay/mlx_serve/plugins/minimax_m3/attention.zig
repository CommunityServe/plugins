//! copied_from: src/plugins/minimax_m3/attention.zig@b20ebdd16e8e262f5e13f2b24ef33cc4ad09aa1f
//! Serial text attention with per-index-head sparse block masks. Index keys are
//! retained even before sparsity activates, so later queries can select old blocks.
const std = @import("std");
const sdk = @import("mlx_stream").sdk;
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const C = @import("geometry.zig").Config;
const reference = @import("reference.zig");
pub const Weights = struct { q: ops.Linear, k: ops.Linear, v: ops.Linear, o: ops.Linear, q_norm: ops.T, k_norm: ops.T, index_q: ops.Linear, index_k: ops.Linear, index_q_norm: ops.T, index_k_norm: ops.T };
pub const Cache = @import("cache.zig").Cache;
const sparse = @import("sparse_decode.zig");
var sparse_engaged = false;
var q8_engaged = false;
fn heads(g: *ops.G, x: ops.T, n: u32, d: u32) !ops.T {
    return g.reshape(x, &.{ 1, @intCast(n), 1, @intCast(d) });
}
pub fn forward(g: *ops.G, a: std.mem.Allocator, c: *const C, w: Weights, cache: *Cache, x: ops.T, position: u32) !ops.T {
    return forwardChunk(g, a, c, w, cache, x, position);
}
/// Layer-major prompt chunks retain one-row projection/norm/RoPE arithmetic.
/// Only the newly projected rows are concatenated. Capacity storage is appended
/// once, then each query reads its own causal prefix without copying the cache.
pub fn forwardChunk(g: *ops.G, a: std.mem.Allocator, c: *const C, w: Weights, cache: *Cache, x: ops.T, position: u32) !ops.T {
    const sx = g.shapeOf(x);
    if (sx.n != 2 or sx.d[0] < 1 or sx.d[0] > 8 or sx.d[1] != c.hidden_size or position != cache.len) return error.MiniMaxAttentionChunk;
    const n: usize = @intCast(sx.d[0]);
    var q: [8]ops.T = undefined;
    var k: [8]ops.T = undefined;
    var v: [8]ops.T = undefined;
    var ik: [8]ops.T = undefined;
    var iq: [8]?ops.T = @splat(null);
    for (0..n) |i| {
        const row: c_int = @intCast(i);
        const xr = try g.slice(x, &.{ row, 0 }, &.{ row + 1, sx.d[1] }, &.{ 1, 1 });
        const pos = position + @as(u32, @intCast(i));
        q[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.q), c.n_heads, c.head_dim), w.q_norm, c), c.rotary_dim, c, pos);
        k[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.k), c.n_kv_heads, c.head_dim), w.k_norm, c), c.rotary_dim, c, pos);
        v[i] = try heads(g, try ops.linear(g, xr, w.v), c.n_kv_heads, c.head_dim);
        ik[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.index_k), 1, c.index_head_dim), w.index_k_norm, c), @min(c.rotary_dim, c.index_head_dim), c, pos);
        if (pos / c.index_block_size + 1 > c.index_topk_blocks)
            iq[i] = try ops.rope(g, try ops.norm(g, try heads(g, try ops.linear(g, xr, w.index_q), c.index_n_heads, c.index_head_dim), w.index_q_norm, c), @min(c.rotary_dim, c.index_head_dim), c, pos);
    }
    _ = try cache.append(g, if (n == 1) k[0] else try g.concat(k[0..n], 2), if (n == 1) v[0] else try g.concat(v[0..n], 2), if (n == 1) ik[0] else try g.concat(ik[0..n], 2));
    var outputs: [8]ops.T = undefined;
    for (0..n) |i| {
        const pos = position + @as(u32, @intCast(i));
        const view = try cache.view(g, pos + 1);
        const attended = try attend(g, a, c, q[i], iq[i], view, pos);
        outputs[i] = try ops.linear(g, try g.reshape(attended, &.{ 1, @intCast(c.n_heads * c.head_dim) }), w.o);
    }
    return if (n == 1) outputs[0] else g.concat(outputs[0..n], 0);
}
/// `[rows, n * d]` to `[1, n, rows, d]`.
fn headRows(g: *ops.G, x: ops.T, rows: usize, n: u32, d: u32) !ops.T {
    return g.transposeAxes(try g.reshape(x, &.{ 1, @intCast(rows), @intCast(n), @intCast(d) }), &.{ 0, 2, 1, 3 });
}
/// A prompt span's rows with every projection, norm and RoPE batched (M = rows): the weights stream once a span, not
/// once a row. Each query still attends its own causal prefix of the cache.
pub fn forwardRows(g: *ops.G, a: std.mem.Allocator, c: *const C, w: Weights, cache: *Cache, x: ops.T, position: u32) !ops.T {
    const sx = g.shapeOf(x);
    if (sx.n != 2 or sx.d[0] < 1 or sx.d[1] != c.hidden_size or position != cache.len) return error.MiniMaxAttentionChunk;
    const n: usize = @intCast(sx.d[0]);
    const index_rope = @min(c.rotary_dim, c.index_head_dim);
    const q = try ops.rope(g, try ops.norm(g, try headRows(g, try ops.linear(g, x, w.q), n, c.n_heads, c.head_dim), w.q_norm, c), c.rotary_dim, c, position);
    const k = try ops.rope(g, try ops.norm(g, try headRows(g, try ops.linear(g, x, w.k), n, c.n_kv_heads, c.head_dim), w.k_norm, c), c.rotary_dim, c, position);
    const v = try headRows(g, try ops.linear(g, x, w.v), n, c.n_kv_heads, c.head_dim);
    const ik = try ops.rope(g, try ops.norm(g, try headRows(g, try ops.linear(g, x, w.index_k), n, 1, c.index_head_dim), w.index_k_norm, c), index_rope, c, position);
    const last = position + @as(u32, @intCast(n - 1));
    const iq = if (last / c.index_block_size + 1 > c.index_topk_blocks)
        try ops.rope(g, try ops.norm(g, try headRows(g, try ops.linear(g, x, w.index_q), n, c.index_n_heads, c.index_head_dim), w.index_q_norm, c), index_rope, c, position)
    else
        null;
    _ = try cache.append(g, k, v, ik);
    const outputs = try a.alloc(ops.T, n);
    defer a.free(outputs);
    var count: usize = 0;
    // A row whose prefix fits the block budget selects every block, so the rows before the budget are one causal call.
    const budget = @as(u64, c.index_topk_blocks) * c.index_block_size;
    const dense: usize = if (cache.bits == 0 and position < budget) @intCast(@min(@as(u64, n), budget - position)) else 0;
    if (dense > 1) {
        const qd = try g.slice(q, &.{ 0, 0, 0, 0 }, &.{ 1, @intCast(c.n_heads), @intCast(dense), @intCast(c.head_dim) }, &.{ 1, 1, 1, 1 });
        const view = try cache.view(g, position + @as(u32, @intCast(dense)));
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, qd, view.k, view.v, 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), "causal", .{}, .{}, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        outputs[0] = try g.reshape(try g.transposeAxes(try g.adopt(out), &.{ 0, 2, 1, 3 }), &.{ @intCast(dense), @intCast(c.n_heads * c.head_dim) });
        count = 1;
    }
    var i: usize = if (dense > 1) dense else 0;
    var since: usize = 0;
    while (i < n) {
        const pos = position + @as(u32, @intCast(i));
        const r: c_int = @intCast(i);
        const tile = if (cache.bits == 0 and c.index_n_heads == c.n_kv_heads and pos >= budget) @import("prefill_step.zig").sparseTileRows(c.index_n_heads, c.index_block_size, position + n, n - i) else 0;
        if (tile > 0) {
            const m: c_int = @intCast(tile);
            const qs = try g.slice(q, &.{ 0, 0, r, 0 }, &.{ 1, @intCast(c.n_heads), r + m, @intCast(c.head_dim) }, &.{ 1, 1, 1, 1 });
            const iqs = try g.slice(iq.?, &.{ 0, 0, r, 0 }, &.{ 1, @intCast(c.index_n_heads), r + m, @intCast(c.index_head_dim) }, &.{ 1, 1, 1, 1 });
            outputs[count] = try sparseRows(g, c, qs, iqs, try cache.view(g, pos + @as(u32, @intCast(m))), pos);
            try g.evalAll(outputs[count .. count + 1]);
            count += 1;
            i += @intCast(m);
            continue;
        }
        const qi = try g.slice(q, &.{ 0, 0, r, 0 }, &.{ 1, @intCast(c.n_heads), r + 1, @intCast(c.head_dim) }, &.{ 1, 1, 1, 1 });
        const iqi = if (pos / c.index_block_size + 1 > c.index_topk_blocks)
            try g.slice(iq.?, &.{ 0, 0, r, 0 }, &.{ 1, @intCast(c.index_n_heads), r + 1, @intCast(c.index_head_dim) }, &.{ 1, 1, 1, 1 })
        else
            null;
        const attended = try attend(g, a, c, qi, iqi, try cache.view(g, pos + 1), pos);
        outputs[count] = try g.reshape(attended, &.{ 1, @intCast(c.n_heads * c.head_dim) });
        count += 1;
        since += 1;
        i += 1;
        // Bounds the graph a long span builds before its layer's eval.
        if (since % 64 == 0) try g.evalAll(outputs[count - 64 .. count]);
    }
    return ops.linear(g, if (count == 1) outputs[0] else try g.concat(outputs[0..count], 0), w.o);
}
/// Rows past the block budget, each choosing its blocks by `reference.selectSparseBlocks`' rule (block maxima of the
/// causal index scores, the initial and local blocks forced, score descending then the lower block), then one masked
/// attention a KV head. q `[1,H,m,D]` and iq `[1,KV,m,Di]` start at `position`; the view holds `position + m` keys.
fn sparseRows(g: *ops.G, c: *const C, q: ops.T, iq: ops.T, view: Cache.View, position: u32) !ops.T {
    const m = g.shapeOf(q).d[2];
    const kv: c_int = @intCast(c.n_kv_heads);
    const bs: c_int = @intCast(c.index_block_size);
    const tokens: c_int = @as(c_int, @intCast(position)) + m;
    const blocks = @divFloor(tokens - 1, bs) + 1;
    const neg_inf = try g.scalar(-std.math.inf(f64), .float32);
    var scores = try g.matmul(try g.astype(iq, .float32), try g.transposeAxes(try g.astype(view.index_k, .float32), &.{ 0, 1, 3, 2 }));
    if (blocks * bs > tokens) scores = try g.concat(&.{ scores, try g.full(&.{ 1, kv, m, blocks * bs - tokens }, neg_inf, .float32) }, 3);
    const rows = try g.reshape(try g.arange(@floatFromInt(position), @floatFromInt(@as(i64, position) + m), 1, .int32), &.{ m, 1 });
    const visible = try g.lessEqual(try g.arange(0, @floatFromInt(blocks * bs), 1, .int32), rows);
    // A NaN score never wins its block (NaN != NaN).
    scores = try g.where(try g.logicalAnd(visible, try g.equal(scores, scores)), scores, neg_inf);
    var pooled = try g.max(try g.reshape(scores, &.{ 1, kv, m, blocks, bs }), -1, false);
    const ids = try g.arange(0, @floatFromInt(blocks), 1, .int32);
    const current = try g.floorDiv(rows, try g.scalar(@floatFromInt(bs), .int32));
    const seen = try g.lessEqual(ids, current);
    pooled = try g.where(seen, pooled, neg_inf);
    if (c.index_init_blocks > 0) pooled = try g.where(try g.logicalAnd(seen, try g.less(ids, try g.scalar(@floatFromInt(c.index_init_blocks), .int32))), try g.scalar(1e30, .float32), pooled);
    if (c.index_local_blocks > 0) pooled = try g.where(try g.logicalAnd(seen, try g.less(try g.sub(current, ids), try g.scalar(@floatFromInt(c.index_local_blocks), .int32))), try g.scalar(1e29, .float32), pooled);
    // A block's rank is the count of blocks ahead of it: a higher score, or the same score and a lower id.
    const mine = try g.expandDims(pooled, -1);
    const other = try g.expandDims(pooled, -2);
    const lower = try g.less(try g.reshape(ids, &.{ 1, blocks }), try g.reshape(ids, &.{ blocks, 1 }));
    const ahead = try g.logicalOr(try g.greater(other, mine), try g.logicalAnd(try g.equal(other, mine), lower));
    const rank = try g.sum(try g.astype(ahead, .int32), -1, false);
    const chosen = try g.logicalAnd(try g.less(rank, try g.scalar(@floatFromInt(c.index_topk_blocks), .int32)), seen);
    const keys = try g.slice(try g.repeat(chosen, bs, -1), &.{ 0, 0, 0, 0 }, &.{ 1, kv, m, tokens }, &.{ 1, 1, 1, 1 });
    const keep = try g.logicalAnd(keys, try g.slice(visible, &.{ 0, 0 }, &.{ m, tokens }, &.{ 1, 1 }));
    const group: c_int = @intCast(c.n_heads / c.n_kv_heads);
    const d: c_int = @intCast(c.head_dim);
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    var heads_out: [16]ops.T = undefined;
    if (kv > heads_out.len) return error.MiniMaxAttentionChunk;
    for (0..@intCast(kv)) |j_| {
        const j: c_int = @intCast(j_);
        const qj = try g.slice(q, &.{ 0, j * group, 0, 0 }, &.{ 1, (j + 1) * group, m, d }, &.{ 1, 1, 1, 1 });
        const kj = try g.slice(view.k, &.{ 0, j, 0, 0 }, &.{ 1, j + 1, tokens, d }, &.{ 1, 1, 1, 1 });
        const vj = try g.slice(view.v, &.{ 0, j, 0, 0 }, &.{ 1, j + 1, tokens, d }, &.{ 1, 1, 1, 1 });
        const mj = try g.slice(keep, &.{ 0, j, 0, 0 }, &.{ 1, j + 1, m, tokens }, &.{ 1, 1, 1, 1 });
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, qj, kj, vj, scale, "array", mj, .{}, false, g.s)) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        heads_out[j_] = try g.adopt(out);
    }
    const all = if (kv == 1) heads_out[0] else try g.concat(heads_out[0..@intCast(kv)], 1);
    return g.reshape(try g.transposeAxes(all, &.{ 0, 2, 1, 3 }), &.{ m, @intCast(c.n_heads * c.head_dim) });
}
fn attend(g: *ops.G, a: std.mem.Allocator, c: *const C, q: ops.T, iq: ?ops.T, view: Cache.View, position: u32) !ops.T {
    var mask: ops.T = .{};
    const tokens = position + 1;
    const blocks = (tokens + c.index_block_size - 1) / c.index_block_size;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    if (view.bits == 8) {
        const q8 = @import("q8_cache.zig");
        if (c.index_n_heads != c.n_kv_heads or !sparse.supported(c.head_dim, c.index_block_size, c.n_heads, c.n_kv_heads)) return error.MiniMaxQ8Shape;
        const selected = if (blocks > c.index_topk_blocks)
            try q8.select(g, iq.?, .{ .codes = view.index_k, .scales = view.index_scales.? }, position, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks)
        else
            try q8.allBlocks(g, c.n_kv_heads, tokens, position, c.index_topk_blocks);
        const out = try q8.attention(g, q, .{ .codes = view.k, .scales = view.k_scales.? }, .{ .codes = view.v, .scales = view.v_scales.? }, selected, position, scale);
        if (!q8_engaged) {
            sdk.log.info("minimax_m3: Q8 KV/index cache engaged (symmetric group32, portable SIMD)\n", .{});
            q8_engaged = true;
        }
        return out;
    }
    if (blocks > c.index_topk_blocks) {
        const scores = try g.matmul(try g.astype(iq.?, .float32), try g.transposeAxes(try g.astype(view.index_k, .float32), &.{ 0, 1, 3, 2 }));
        if (c.index_n_heads == c.n_kv_heads and c.index_topk_blocks <= 16 and sparse.supported(c.head_dim, c.index_block_size, c.n_heads, c.n_kv_heads)) {
            const selected = try sparse.select(g, scores, c.index_n_heads, tokens, position, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks);
            const out = try sparse.attention(g, q, view.k, view.v, selected, position, scale);
            if (!sparse_engaged) {
                sdk.log.info("minimax_m3: GPU selected-block attention engaged (portable SIMD)\n", .{});
                sparse_engaged = true;
            }
            return out;
        }
        // The small host oracle geometries retain the general masked route.
        const host_scores = try a.alloc(f32, @as(usize, c.index_n_heads) * tokens);
        defer a.free(host_scores);
        _ = try g.hostF32(scores, host_scores);
        const selected = try a.alloc(bool, host_scores.len);
        defer a.free(selected);
        const scratch = try a.alloc(f32, @as(usize, c.index_n_heads) * blocks);
        defer a.free(scratch);
        try reference.selectSparseBlocks(host_scores, c.index_n_heads, tokens, position, c.index_block_size, c.index_topk_blocks, c.index_local_blocks, c.index_init_blocks, selected, scratch);
        const expanded = try a.alloc(bool, @as(usize, c.n_heads) * tokens);
        defer a.free(expanded);
        for (0..c.n_heads) |h| {
            const ih = h / (c.n_heads / c.index_n_heads);
            @memcpy(expanded[h * tokens ..][0..tokens], selected[ih * tokens ..][0..tokens]);
        }
        mask = try g.hostArray(std.mem.sliceAsBytes(expanded), &.{ 1, @intCast(c.n_heads), 1, @intCast(tokens) }, .bool_);
    }
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, view.k, view.v, scale, if (mask.ctx != null) "array" else "", mask, .{}, false, g.s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    return g.adopt(out);
}
// Opt-in local device fixture: verifies the actual graph crosses the block
// budget, drops old blocks, retains index keys and widens each KV group correctly.
test "MiniMax GPU attention crosses dense-to-sparse boundary with retained index keys" {
    if (sdk.mlx.noGpuBackend() or sdk.mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    var c: C = .{};
    c.hidden_size = 4;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 2;
    c.rotary_dim = 2;
    c.index_n_heads = 2;
    c.index_head_dim = 2;
    c.index_block_size = 2;
    c.index_topk_blocks = 1;
    const identity = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const output = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
    const w: Weights = .{
        .q = .{ .weight = try g.zeros(&.{ 8, 4 }, .float32) },
        .k = .{ .weight = try g.zeros(&.{ 4, 4 }, .float32) },
        .v = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&identity), &.{ 4, 4 }, .float32) },
        .o = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&output), &.{ 4, 8 }, .float32) },
        .q_norm = try g.zeros(&.{2}, .float32),
        .k_norm = try g.zeros(&.{2}, .float32),
        .index_q = .{ .weight = try g.zeros(&.{ 4, 4 }, .float32) },
        .index_k = .{ .weight = try g.zeros(&.{ 2, 4 }, .float32) },
        .index_q_norm = try g.zeros(&.{2}, .float32),
        .index_k_norm = try g.zeros(&.{2}, .float32),
    };
    var cache: Cache = .{};
    defer cache.clear(&g);
    const rows = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 } };
    const wants = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 3, 4, 5, 6 }, .{ 9, 10, 11, 12 } };
    for (rows, wants, 0..) |row, want, position| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const x = try g.hostArray(std.mem.sliceAsBytes(&row), &.{ 1, 4 }, .float32);
        const y = try forward(&g, a, &c, w, &cache, x, @intCast(position));
        var actual: [4]f32 = undefined;
        _ = try g.hostF32(y, &actual);
        for (actual, want) |got, expected| try std.testing.expectApproxEqAbs(expected, got, 1e-5);
        try std.testing.expectEqual(@as(u32, @intCast(position + 1)), cache.len);
        const retained = try cache.view(&g, cache.len);
        try std.testing.expectEqual(@as(c_int, @intCast(position + 1)), g.shapeOf(retained.index_k).d[2]);
    }
}
test "MiniMax GPU chunk attention matches serial queries across cache growth and sparse block boundaries" {
    if (sdk.mlx.noGpuBackend() or sdk.mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    var c: C = .{};
    c.hidden_size = 4;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 128;
    c.rotary_dim = 64;
    c.index_n_heads = 2;
    c.index_head_dim = 128;
    c.index_block_size = 128;
    c.index_topk_blocks = 1;
    var v_weights: [256 * 4]f32 = @splat(0);
    for (0..256) |i| v_weights[i * 4 + i % 4] = 1;
    var o_weights: [4 * 512]f32 = @splat(0);
    for (0..4) |i| o_weights[i * 512 + i] = 1;
    const w: Weights = .{
        .q = .{ .weight = try g.zeros(&.{ 512, 4 }, .float32) },
        .k = .{ .weight = try g.zeros(&.{ 256, 4 }, .float32) },
        .v = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&v_weights), &.{ 256, 4 }, .float32) },
        .o = .{ .weight = try g.hostArray(std.mem.sliceAsBytes(&o_weights), &.{ 4, 512 }, .float32) },
        .q_norm = try g.zeros(&.{128}, .float32),
        .k_norm = try g.zeros(&.{128}, .float32),
        .index_q = .{ .weight = try g.zeros(&.{ 256, 4 }, .float32) },
        .index_k = .{ .weight = try g.zeros(&.{ 128, 4 }, .float32) },
        .index_q_norm = try g.zeros(&.{128}, .float32),
        .index_k_norm = try g.zeros(&.{128}, .float32),
    };
    var serial: Cache = .{ .limit = 300 };
    defer serial.clear(&g);
    var chunked: Cache = .{ .limit = 300 };
    defer chunked.clear(&g);
    {
        const mark = g.mark();
        defer g.resetTo(mark);
        const history = try a.alloc(f32, 2 * 250 * 128);
        defer a.free(history);
        for (0..2) |h| for (0..250) |t| for (0..128) |d| {
            history[(h * 250 + t) * 128 + d] = @as(f32, @floatFromInt(t + d % 4)) / 256;
        };
        const k = try g.zeros(&.{ 1, 2, 250, 128 }, .float32);
        const v = try g.hostArray(std.mem.sliceAsBytes(history), &.{ 1, 2, 250, 128 }, .float32);
        const index = try g.zeros(&.{ 1, 1, 250, 128 }, .float32);
        _ = try serial.append(&g, k, v, index);
        _ = try chunked.append(&g, k, v, index);
    }
    var rows: [8 * 4]f32 = undefined;
    for (0..8) |t| for (0..4) |d| {
        rows[t * 4 + d] = @as(f32, @floatFromInt(t * 4 + d + 10)) / 8;
    };
    var expected: [8 * 4]f32 = undefined;
    for (0..8) |i| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const row = rows[i * 4 ..][0..4];
        const x = try g.hostArray(std.mem.sliceAsBytes(row), &.{ 1, 4 }, .float32);
        const y = try forward(&g, a, &c, w, &serial, x, @intCast(250 + i));
        _ = try g.hostF32(y, expected[i * 4 ..][0..4]);
    }
    const input = try g.hostArray(std.mem.sliceAsBytes(&rows), &.{ 8, 4 }, .float32);
    const out = try forwardChunk(&g, a, &c, w, &chunked, input, 250);
    var actual: [8 * 4]f32 = undefined;
    _ = try g.hostF32(out, &actual);
    for (actual, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    try std.testing.expectEqual(@as(u32, 258), chunked.len);
    try std.testing.expectEqual(serial.len, chunked.len);
    try std.testing.expectEqual(serial.capacity, chunked.capacity);
    // The new block begins at token256: query256 must see exactly its own
    // value, despite all eight rows having landed before attention started.
    for (0..4) |d| try std.testing.expectApproxEqAbs(rows[6 * 4 + d], actual[6 * 4 + d], 1e-6);
}
test "MiniMax GPU prompt rows attend batched (dense inside the block budget, sparse tiles past it) and match row-by-row attention" {
    if (sdk.mlx.noGpuBackend() or sdk.mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    const h = 512;
    const rows = 40;
    const Fill = struct {
        fn values(gg: *ops.G, shape: []const c_int, salt: usize) !ops.T {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const v = try gg.gpa.alloc(f32, n);
            defer gg.gpa.free(v);
            for (v, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + salt * 5) % 23)) - 11)) / 64;
            return gg.hostArray(std.mem.sliceAsBytes(v), shape, .float32);
        }
        fn linear(gg: *ops.G, outputs: c_int, inputs: c_int, salt: usize) !ops.Linear {
            return .{ .weight = try values(gg, &.{ outputs, inputs }, salt) };
        }
    };
    const norm = try g.zeros(&.{128}, .float32);
    const w: Weights = .{ .q = try Fill.linear(&g, 4 * 128, h, 1), .k = try Fill.linear(&g, 128, h, 2), .v = try Fill.linear(&g, 128, h, 3), .o = try Fill.linear(&g, h, 4 * 128, 4), .q_norm = norm, .k_norm = norm, .index_q = try Fill.linear(&g, 128, h, 5), .index_k = try Fill.linear(&g, 128, h, 6), .index_q_norm = norm, .index_k_norm = norm };
    const input = try Fill.values(&g, &.{ rows, h }, 8);
    // (start, budget): 28 dense rows then a sparse tile; all 40 dense; rows in block 2 choosing one of blocks 0 and 1.
    for ([_][2]u32{ .{ 100, 1 }, .{ 100, 16 }, .{ 300, 2 } }) |case| {
        const start = case[0];
        const topk = case[1];
        const mark = g.mark();
        defer g.resetTo(mark);
        const prefix = try Fill.values(&g, &.{ 1, 1, @intCast(start), 128 }, 7);
        const c: C = .{ .hidden_size = h, .n_heads = 4, .n_kv_heads = 1, .head_dim = 128, .index_n_heads = 1, .index_head_dim = 128, .index_block_size = 128, .index_topk_blocks = topk };
        var one: Cache = .{ .limit = 512 };
        defer one.clear(&g);
        var many: Cache = .{ .limit = 512 };
        defer many.clear(&g);
        _ = try one.append(&g, prefix, prefix, prefix);
        _ = try many.append(&g, prefix, prefix, prefix);
        var expected: [rows * h]f32 = undefined;
        for (0..rows) |i| {
            const y = try forwardChunk(&g, a, &c, w, &one, try ops.row(&g, input, @intCast(i)), @intCast(start + i));
            _ = try g.hostF32(y, expected[i * h ..][0..h]);
        }
        var actual: [rows * h]f32 = undefined;
        _ = try g.hostF32(try forwardRows(&g, a, &c, w, &many, input, start), &actual);
        for (actual, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-4 * @max(1, @abs(want)));
    }
}
test "MiniMax GPU sparse tile rows choose the reference's blocks and attend as the scalar oracle" {
    if (sdk.mlx.noGpuBackend() or sdk.mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    const H = 8;
    const KV = 2;
    const D = 128;
    const start = 600;
    const m = 24;
    const tokens = start + m;
    // Sixty-fourths: exact in bf16, and their f32 dot products are exact, so both sides see the same scores.
    var seed: u32 = 12345;
    const Gen = struct {
        fn fill(st: *u32, out: []f32) void {
            for (out) |*x| {
                st.* = st.* *% 1664525 +% 1013904223;
                x.* = @as(f32, @floatFromInt(@as(i32, @intCast(st.* >> 26)) - 32)) / 64;
            }
        }
    };
    const q = try a.alloc(f32, H * m * D);
    defer a.free(q);
    const iq = try a.alloc(f32, KV * m * D);
    defer a.free(iq);
    const k = try a.alloc(f32, KV * tokens * D);
    defer a.free(k);
    const v = try a.alloc(f32, k.len);
    defer a.free(v);
    const ik = try a.alloc(f32, k.len);
    defer a.free(ik);
    for ([_][]f32{ q, iq, k, v, ik }) |x| Gen.fill(&seed, x);
    // (top-k, local, initial): a free choice among old blocks, and a forced first block.
    for ([_][3]u32{ .{ 3, 1, 0 }, .{ 2, 1, 1 } }) |case| {
        const c: C = .{ .n_heads = H, .n_kv_heads = KV, .head_dim = D, .index_n_heads = KV, .index_head_dim = D, .index_block_size = 128, .index_topk_blocks = case[0], .index_local_blocks = case[1], .index_init_blocks = case[2] };
        var expected: [m * H * D]f32 = undefined;
        for (0..m) |r| {
            const pos = start + r;
            const seen = pos + 1;
            const scores = try a.alloc(f32, KV * seen);
            defer a.free(scores);
            for (0..KV) |h| for (0..seen) |t| {
                var dot: f32 = 0;
                for (0..D) |d| dot += iq[(h * m + r) * D + d] * ik[(h * tokens + t) * D + d];
                scores[h * seen + t] = dot;
            };
            const keep = try a.alloc(bool, scores.len);
            defer a.free(keep);
            var pooled: [KV * 8]f32 = undefined;
            try reference.selectSparseBlocks(scores, KV, seen, pos, 128, case[0], case[1], case[2], keep, &pooled);
            const kr = try a.alloc(f32, KV * seen * D);
            defer a.free(kr);
            const vr = try a.alloc(f32, kr.len);
            defer a.free(vr);
            for (0..KV) |h| {
                @memcpy(kr[h * seen * D ..][0 .. seen * D], k[h * tokens * D ..][0 .. seen * D]);
                @memcpy(vr[h * seen * D ..][0 .. seen * D], v[h * tokens * D ..][0 .. seen * D]);
            }
            var qr: [H * D]f32 = undefined;
            for (0..H) |h| @memcpy(qr[h * D ..][0..D], q[(h * m + r) * D ..][0..D]);
            const scratch = try a.alloc(f32, seen);
            defer a.free(scratch);
            try reference.groupedAttention(&qr, kr, vr, H, KV, KV, seen, D, keep, expected[r * H * D ..][0 .. H * D], scratch);
        }
        // The decode kernel's bars (`sparse_decode.zig`): f32 accumulation, then one rounding to the activation dtype.
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
            const mark = g.mark();
            defer g.resetTo(mark);
            const host = struct {
                fn arr(gg: *ops.G, x: []const f32, shape: []const c_int, dt: mlx.mlx_dtype) !ops.T {
                    return gg.astype(try gg.hostArray(std.mem.sliceAsBytes(x), shape, .float32), dt);
                }
            };
            const view: Cache.View = .{ .k = try host.arr(&g, k, &.{ 1, KV, tokens, D }, dtype), .v = try host.arr(&g, v, &.{ 1, KV, tokens, D }, dtype), .index_k = try host.arr(&g, ik, &.{ 1, KV, tokens, D }, dtype) };
            const out = try sparseRows(&g, &c, try host.arr(&g, q, &.{ 1, H, m, D }, dtype), try host.arr(&g, iq, &.{ 1, KV, m, D }, dtype), view, start);
            var actual: [m * H * D]f32 = undefined;
            _ = try g.hostF32(try g.astype(out, .float32), &actual);
            const tolerance: f32 = if (dtype == .float32) 2e-5 else 4e-3;
            for (actual, expected) |got, want| {
                try std.testing.expect(std.math.isFinite(got));
                try std.testing.expectApproxEqAbs(want, got, tolerance);
            }
        }
    }
}
