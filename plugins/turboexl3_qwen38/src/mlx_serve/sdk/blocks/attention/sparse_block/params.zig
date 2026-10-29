//! copied_from: src/sdk/blocks/attention/sparse_block/params.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Block-sparse attention's shape: an indexer scores blocks of keys, and each query attends the `top_k` best blocks
//! it can see (its first `init` and last `local` blocks forced among them), plus the keys after its last complete
//! block. Pure host arithmetic; the ops read these answers.

const std = @import("std");

/// Which blocks a query attends.
pub const Blocks = struct {
    size: u32,
    top_k: u32,
    init: u32 = 0,
    /// The blocks ending at the query's own, counted in `top_k`; 0 leaves its incomplete block outside `top_k`.
    local: u32 = 0,

    /// Blocks every key of which a query at cache position `pos` sees; its keys past them are always attended.
    pub fn complete(b: Blocks, pos: c_int) c_int {
        return if (pos < 0) 0 else @divTrunc(pos + 1, @as(c_int, @intCast(b.size)));
    }

    /// The candidate blocks `[0, bound)` of a query at `pos`: its complete blocks, and with local blocks its own.
    pub fn bound(b: Blocks, pos: c_int) c_int {
        if (b.local == 0 or pos < 0) return b.complete(pos);
        return @divTrunc(pos, @as(c_int, @intCast(b.size))) + 1;
    }

    /// Per row `r` (cache position `row0 + r`), the candidate bound clamped to the `nb` scored blocks; `all_vis`
    /// claims every row sees all `nb`, and is honored verbatim.
    pub fn selectBounds(b: Blocks, out: []i32, row0: c_int, nb: c_int, all_vis: bool) void {
        for (out, 0..) |*o, r| o.* = if (all_vis) nb else @min(nb, b.bound(row0 + @as(c_int, @intCast(r))));
    }
};

pub const Score = enum {
    /// sum over the index heads of relu(q_h . pooled block key), one pick list per query (Qwen3.8-Flash-Next QSA:
    /// the pooled key is the mean of the block's keys, normed and roped by the model).
    pooled_relu_sum,
    /// max over the block's visible keys of q_h . k, one pick list per index head (MiniMax-M3 `sparse_score_type`
    /// max; q head h reads index head h / (q heads / index heads)).
    key_max,
};

pub const Params = struct {
    blocks: Blocks,
    index_heads: u32,
    index_dim: u32,
    score: Score,
    /// The attention layers that are sparse; null = every one.
    layers: ?[]const bool = null,

};

const testing = std.testing;
