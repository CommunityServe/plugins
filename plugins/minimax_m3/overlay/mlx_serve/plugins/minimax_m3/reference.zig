//! copied_from: src/plugins/minimax_m3/reference.zig@61ec76d7bc1abfb7f588db3bf6a94ffc0cd423c7
//! Scalar reference operations for the released MiniMax M3.1 Flash text path.
//! Sparse selection follows vLLM 7651825993d001759107c66cdf6983535900c874,
//! common/ops/index_topk.py: one independently selected block set per index head.
//! These small CPU routines also specify the contracts of the device graph.
const std = @import("std");
pub const Error = error{InvalidShape};
pub fn sigmoid(x: f32) f32 {
    // Stable for large negative router logits as well as positive ones.
    if (x >= 0) return 1 / (1 + @exp(-x));
    const e = @exp(x);
    return e / (1 + e);
}
/// Gate is upper-clipped only; the linear/up branch is clipped on both sides.
/// The +beta belongs to the up branch, after clipping.
pub fn swigluOai(up: f32, gate: f32, alpha: f32, limit: f32, beta: f32) f32 {
    const g = @min(gate, limit);
    const u = std.math.clamp(up, -limit, limit);
    return g * sigmoid(alpha * g) * (u + beta);
}
/// The checkpoint stores the Gemma offset, so effective gamma is weight + 1.
/// All accumulation is in f32; input and output may alias.
pub fn gemmaRmsNorm(x: []const f32, weight: []const f32, eps: f32, out: []f32) Error!void {
    if (x.len == 0 or x.len != weight.len or x.len != out.len) return error.InvalidShape;
    var sum: f32 = 0;
    for (x) |v| sum += v * v;
    const inv = 1 / @sqrt(sum / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, weight, out) |v, w, *o| o.* = v * inv * (1 + w);
}
/// Nontraditional RoPE pairs the two halves of the first rotary_dim channels.
/// The remaining head channels are untouched. The position is the absolute token
/// position, not the slot modulo the sparse block or a prompt chunk.
pub fn rope(x: []f32, rotary_dim: usize, theta: f32, position: u32) Error!void {
    if (rotary_dim == 0 or rotary_dim % 2 != 0 or rotary_dim > x.len or theta <= 0) return error.InvalidShape;
    const half = rotary_dim / 2;
    for (0..half) |i| {
        const exponent = -2 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rotary_dim));
        const angle = @as(f32, @floatFromInt(position)) * std.math.pow(f32, theta, exponent);
        const c = @cos(angle);
        const s = @sin(angle);
        const a = x[i];
        const b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}
/// Correction bias affects selection only, never the mixture weights. The
/// selected unbiased sigmoid scores sum to routed_scale after normalization.
/// Exact ties use the smaller expert ID; reference argpartition leaves ties
/// unspecified, so callers must not depend on its ordering in a tied case.
pub fn selectExperts(gates: []const f32, correction: []const f32, routed_scale: f32, ids: []u32, weights: []f32) Error!void {
    if (gates.len == 0 or gates.len != correction.len or ids.len == 0 or ids.len > gates.len or weights.len != ids.len) return error.InvalidShape;
    var total: f32 = 0;
    for (ids, 0..) |*id, rank| {
        var best: ?usize = null;
        var best_score: f32 = -std.math.inf(f32);
        for (gates, correction, 0..) |g, bias, e| {
            var used = false;
            for (ids[0..rank]) |previous| if (previous == e) {
                used = true;
                break;
            };
            if (used) continue;
            const score = sigmoid(g) + bias;
            if (best == null or score > best_score) {
                best = e;
                best_score = score;
            }
        }
        id.* = @intCast(best.?);
        weights[rank] = sigmoid(gates[best.?]);
        total += weights[rank];
    }
    for (weights) |*w| w.* = w.* / (total + 1e-20) * routed_scale;
}
/// scores is [index_heads, tokens] for ONE query, already dotted/scaled in f32.
/// selected has the same shape. scratch has at least index_heads*ceil(tokens /
/// block_size) entries. Future keys are excluded BEFORE block max and again in
/// the final token mask. Partial blocks count; local/init blocks consume top-k
/// slots. Each index head selects independently and covers one KV group in the
/// release (four index heads, four KV heads, sixteen query heads per group).
pub fn selectSparseBlocks(scores: []const f32, index_heads: usize, tokens: usize, position: usize, block_size: usize, topk: usize, local_blocks: usize, init_blocks: usize, selected: []bool, scratch: []f32) Error!void {
    if (index_heads == 0 or tokens == 0 or block_size == 0 or topk == 0 or scores.len != index_heads * tokens or selected.len != scores.len) return error.InvalidShape;
    const blocks = (tokens - 1) / block_size + 1;
    if (scratch.len < index_heads * blocks) return error.InvalidShape;
    @memset(selected, false);
    const current = position / block_size;
    const visible = @min(blocks, current + 1);
    for (0..index_heads) |h| {
        const pooled = scratch[h * blocks ..][0..blocks];
        @memset(pooled, -std.math.inf(f32));
        for (0..@min(tokens, position + 1)) |t| {
            const score = scores[h * tokens + t];
            if (!std.math.isNan(score)) pooled[t / block_size] = @max(pooled[t / block_size], score);
        }
        for (0..visible) |b| {
            if (b < init_blocks) pooled[b] = 1e30;
            if (local_blocks > 0 and b <= current and current - b < local_blocks) pooled[b] = 1e29;
        }
        for (0..@min(topk, visible)) |_| {
            var best: ?usize = null;
            var best_score: f32 = -std.math.inf(f32);
            for (pooled[0..visible], 0..) |score, b| {
                if (std.math.isNan(score)) continue; // Already selected.
                if (best == null or score > best_score) {
                    best = b;
                    best_score = score;
                }
            }
            const b = best.?;
            const end = @min(@min((b + 1) * block_size, tokens), position + 1);
            @memset(selected[h * tokens + b * block_size .. h * tokens + end], true);
            pooled[b] = std.math.nan(f32);
        }
    }
}
/// Single-query GQA attention oracle: q=[query_heads,dim], k/v=[kv_heads,tokens,
/// dim], keep=[index_heads,tokens]. The selected mask widens by contiguous head
/// groups, not by interleaving heads. Scratch contains at least tokens floats.
pub fn groupedAttention(q: []const f32, k: []const f32, v: []const f32, query_heads: usize, kv_heads: usize, index_heads: usize, tokens: usize, dim: usize, keep: []const bool, out: []f32, scratch: []f32) Error!void {
    if (query_heads == 0 or kv_heads == 0 or index_heads == 0 or dim == 0 or tokens == 0 or query_heads % kv_heads != 0 or query_heads % index_heads != 0 or q.len != query_heads * dim or out.len != q.len or k.len != kv_heads * tokens * dim or v.len != k.len or keep.len != index_heads * tokens or scratch.len < tokens) return error.InvalidShape;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(dim)));
    for (0..query_heads) |h| {
        const kh = h / (query_heads / kv_heads);
        const ih = h / (query_heads / index_heads);
        var max_score: f32 = -std.math.inf(f32);
        for (0..tokens) |t| {
            var score: f32 = 0;
            for (0..dim) |d| score += q[h * dim + d] * k[(kh * tokens + t) * dim + d];
            scratch[t] = if (keep[ih * tokens + t]) score * scale else -std.math.inf(f32);
            max_score = @max(max_score, scratch[t]);
        }
        if (max_score == -std.math.inf(f32)) return error.InvalidShape;
        var total: f32 = 0;
        for (scratch[0..tokens]) |*s| {
            s.* = @exp(s.* - max_score);
            total += s.*;
        }
        for (0..dim) |d| {
            var sum: f32 = 0;
            for (0..tokens) |t| sum += scratch[t] / total * v[(kh * tokens + t) * dim + d];
            out[h * dim + d] = sum;
        }
    }
}
const testing = std.testing;
