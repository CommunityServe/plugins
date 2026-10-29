//! copied_from: src/plugins/qwen4_exp/qsa/history.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The per-slot QSA key ring and pooled-block history (`QSA_RING_ROWS`), head marks, and `ssmFreeQsaState`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const round_cost_mod = @import("../../../engines/mlx/spec/round_cost.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const SSMCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").SSMCheckpoint;
const captureSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").captureSsmCheckpoint;
const restoreSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").restoreSsmCheckpoint;
const spanPreservingDropIndex = @import("../../../engines/mlx/ssm/checkpoint.zig").spanPreservingDropIndex;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const qsaRawKeyBytes = @import("../../../engines/mlx/ssm/entry.zig").qsaRawKeyBytes;
const ssmRollbackFromCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const applyQsaHistoryAt = @import("checkpoint.zig").applyQsaHistoryAt;
const attachQsaHistoryToLatest = @import("checkpoint.zig").attachQsaHistoryToLatest;
const qsaHistoryRows = @import("checkpoint.zig").qsaHistoryRows;

/// Raw indexer-key rows kept per layer (tokens): verify width plus one block.
pub const QSA_RING_ROWS: c_int = 32;

/// Widest `indexer_compress_ratio` the ring serves; past it a checkpoint is refused at load.
pub const QSA_MAX_COMPRESS_RATIO: c_int = 8;

comptime {
    // A partial-accept rollback re-slices the ring to `keep`, so the ring must still hold a
    // whole verify width plus the partial block that width reopened.
    std.debug.assert(QSA_RING_ROWS >= @as(c_int, @intCast(round_cost_mod.MAX_WIDTH)) + QSA_MAX_COMPRESS_RATIO + 1);
}

/// The ring is a fixed size, so a config asking for a wider block than it can hold is refused
/// by name instead of serving a leftover the rollback cannot re-pool.
pub fn checkQsaCompressRatio(ratio: u32) !void {
    if (ratio == 0 or ratio > QSA_MAX_COMPRESS_RATIO) return error.UnsupportedQsaCompressRatio;
}

/// The `pos % ratio` raw indexer rows ending at MTP-head row `pos`, copied while the ring
/// still held them. The trunk restores at one of its checkpoint positions and the head is
/// clamped to that same row; by commit time the head's ring is a whole generated tail past
/// it, so the rows the next append must re-pool are kept here instead.
pub const QsaHeadMark = struct {
    pos: c_int = 0,
    rows: mlx.mlx_array = .{ .ctx = null },
};

/// Marks a head keeps. A prefill's checkpoints are capped at `ssm_checkpoint_max` (16) and a
/// warm turn merges the donor's; past this the list thins like the checkpoints themselves.
pub const QSA_HEAD_MARKS_MAX: usize = 32;

/// One owner of a head's marks: the live head holds one, its committed snapshot a
/// refcount-sharing copy. Positions ascend.
pub const QsaHeadMarkSet = struct {
    items: [QSA_HEAD_MARKS_MAX]QsaHeadMark = @splat(.{}),
    len: usize = 0,

    pub fn slice(self: *const QsaHeadMarkSet) []const QsaHeadMark {
        return self.items[0..self.len];
    }

    /// A second owner for every mark; both sets then free independently.
    pub fn share(src: []const QsaHeadMark) QsaHeadMarkSet {
        var out: QsaHeadMarkSet = .{};
        for (src) |m| {
            if (m.rows.ctx == null or out.len == QSA_HEAD_MARKS_MAX) continue;
            var arr = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&arr, m.rows);
            out.items[out.len] = .{ .pos = m.pos, .rows = arr };
            out.len += 1;
        }
        return out;
    }

    pub fn deinit(self: *QsaHeadMarkSet) void {
        for (self.items[0..self.len]) |*m| {
            if (m.rows.ctx != null) _ = mlx.mlx_array_free(m.rows);
            m.* = .{};
        }
        self.len = 0;
    }

    pub fn find(self: *const QsaHeadMarkSet, pos: c_int) ?mlx.mlx_array {
        for (self.items[0..self.len]) |m| {
            if (m.pos == pos and m.rows.ctx != null) return m.rows;
        }
        return null;
    }

    /// Marks above `pos` describe rows a clamp just discarded — the next append writes
    /// different tokens there.
    pub fn dropAbove(self: *QsaHeadMarkSet, pos: c_int) void {
        var i = self.len;
        while (i > 0 and self.items[i - 1].pos > pos) : (i -= 1) {
            if (self.items[i - 1].rows.ctx != null) _ = mlx.mlx_array_free(self.items[i - 1].rows);
            self.items[i - 1] = .{};
        }
        self.len = i;
    }

};

pub fn qsaLeftoverAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return qsaRowsEndingAt(aux, hist, pos, @mod(pos, @max(ratio, 1)), s);
}

/// The `want` raw indexer rows ending at row `pos`, out of a buffer whose rows end at logical
/// row `hist`; null when the buffer no longer physically holds them.
fn qsaRowsEndingAt(aux: mlx.mlx_array, hist: c_int, pos: c_int, want: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (want <= 0 or aux.ctx == null) return .{ .ctx = null };
    const sh = mlx.getShape(aux);
    if (sh.len < 2 or sh[1] < want) return .{ .ctx = null };
    const held = sh[1];
    const ring_start = hist - held;
    const want_start = pos - want;
    if (want_start < ring_start or pos > hist) return .{ .ctx = null };
    const phys_from = want_start - ring_start;
    const phys_to = pos - ring_start;
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, aux, &[_]c_int{ 0, phys_from, 0 }, 3, &[_]c_int{ sh[0], phys_to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

/// Free and null a live entry's whole QSA state: the history, its capacity buffer and the pooled bank move together.
pub fn ssmFreeQsaState(e: *SSMCacheEntry) void {
    if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
    if (e.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_key_buf);
    if (e.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled);
    if (e.qsa_pooled_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_pooled_buf);
    if (e.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_bank);
    if (e.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(e.qsa_score_buf);
    e.aux_state = .{ .ctx = null };
    e.qsa_key_buf = .{ .ctx = null };
    e.qsa_key_rows = 0;
    e.qsa_hist_rows = 0;
    e.qsa_pooled = .{ .ctx = null };
    e.qsa_pooled_buf = .{ .ctx = null };
    e.qsa_pooled_blocks = 0;
    e.qsa_score_bank = .{ .ctx = null };
    e.qsa_score_buf = .{ .ctx = null };
    e.qsa_score_blocks = 0;
}

/// Where the next `add` rows of QSA raw-key history go, and how big the capacity buffer must
/// be. `cap == 0` = no buffer yet. Growth is the KV cache's proportional policy.
const QsaKeyAppendPlan = struct { new_cap: usize, write_at: usize, new_rows: usize };
fn qsaKeyAppendPlan(rows: usize, cap: usize, add: usize, reserve: usize) QsaKeyAppendPlan {
    const need = rows + add;
    // A reservation raises the first grow to the whole request; past it the proportional policy takes over.
    const grown = if (need <= cap) cap else @max(KVCache.nextCapacity(cap, need), reserve);
    return .{
        .new_cap = grown,
        .write_at = rows,
        .new_rows = need,
    };
}

/// How many times a capacity buffer actually allocated.
pub var qsa_cap_buf_allocs: usize = 0;

/// How many times a QSA history accelerator was re-seeded from its authority (`seedCapBuf`).
pub var qsa_history_seeds: usize = 0;

pub var qsa_score_incremental_appends: usize = 0;

/// Append `chunk` into a pre-grown capacity buffer along `axis` and republish `view` as the
/// tight `count`-long prefix. The published view is dropped first: while it lives mlx copies
/// the buffer instead of donating it.
/// Materialize `src` straight into a capacity buffer already `reserve` rows long: the seed
/// half of `capBufAppend`, done once. A tight seed made the next append grow (two full passes
/// over the prefix, ~81 ms at 393k). Byte-identical to the pair it replaces.
pub fn seedCapBuf(s: mlx.mlx_stream, src: mlx.mlx_array, axis: usize, reserve: usize) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const nd: usize = sh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const held: usize = @intCast(sh[axis]);
    // No reservation (or one the history exceeds): the tight copy is the right buffer.
    if (reserve <= held) return materializedOwnedCopy(s, src);
    var shape = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    for (sh, 0..) |d, i| {
        shape[i] = d;
        stop[i] = d;
    }
    shape[axis] = @intCast(reserve);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var grown = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grown);
    try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(src), s));
    var seeded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(seeded);
    try mlx.check(mlx.mlx_slice_update(&seeded, grown, src, &start, nd, &stop, nd, &strides, nd, s));
    // Not counted in `qsa_cap_buf_allocs`: that counter means "capBufAppend had to grow".
    qsa_history_seeds += 1;
    return seeded;
}

pub fn capBufAppend(s: mlx.mlx_stream, buf: *mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, chunk: mlx.mlx_array, axis: usize, reserve: usize) !void {
    const csh = mlx.getShape(chunk);
    const nd: usize = csh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    const add = csh[axis];
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };

    const cap: usize = if (buf.ctx != null) @intCast(mlx.getShape(buf.*)[axis]) else 0;
    const plan = qsaKeyAppendPlan(@intCast(count.*), cap, @intCast(add), reserve);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (csh, 0..) |d, i| stop[i] = d;

    if (plan.new_cap != cap) {
        qsa_cap_buf_allocs += 1;
        var shape = [_]c_int{ 0, 0, 0, 0 };
        for (csh, 0..) |d, i| shape[i] = d;
        shape[axis] = @intCast(plan.new_cap);
        var grown = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(grown);
        try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(chunk), s));
        if (buf.ctx != null) {
            if (count.* > 0) {
                stop[axis] = count.*;
                var old = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(old);
                try mlx.check(mlx.mlx_slice(&old, buf.*, &start, nd, &stop, nd, &strides, nd, s));
                var seeded = mlx.mlx_array_new();
                // Safe unscoped: on failure `grown` is still the old handle, so the two errdefers free two different arrays.
                errdefer _ = mlx.mlx_array_free(seeded);
                try mlx.check(mlx.mlx_slice_update(&seeded, grown, old, &start, nd, &stop, nd, &strides, nd, s));
                _ = mlx.mlx_array_free(grown);
                grown = seeded;
            }
            _ = mlx.mlx_array_free(buf.*);
        }
        buf.* = grown;
    }
    start[axis] = count.*;
    stop[axis] = @intCast(plan.new_rows);
    // Block-scoped errdefer on purpose: `updated` becomes `buf.*` on the last line.
    {
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, chunk, &start, nd, &stop, nd, &strides, nd, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }
    count.* = @intCast(plan.new_rows);

    start[axis] = 0;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf.*, &start, nd, &stop, nd, &strides, nd, s));
}

/// Re-publish an entry's QSA views over the first `keep` rows of the capacity buffers it
/// already owns, instead of copying the history out and re-seeding it. False (the caller
/// falls back to the copy) when there is no capacity buffer: a restore, a cold entry, a
/// post-adopt head.
pub fn qsaResliceToKeep(entry: *SSMCacheEntry, keep: c_int, s: mlx.mlx_stream) !bool {
    const hist = qsaHistoryRows(entry);
    if (keep > hist) return false;
    const drop_tail = hist - keep;
    const held = entry.qsa_key_rows;
    if (drop_tail > held) return false;
    const new_held = held - drop_tail;
    // The retained ring must still hold `keep`'s own leftover for the next append to re-pool
    // it; short of that the clamp belongs to the caller's mark path, or to its named decline.
    if (new_held < @mod(keep, @max(entry.qsa_ratio, 1))) return false;
    if (new_held > 0) {
        const buf_ok = entry.qsa_key_buf.ctx != null and
            mlx.getShape(entry.qsa_key_buf)[1] >= new_held;
        if (!buf_ok) return false;
        try capBufReslice(s, entry.qsa_key_buf, &entry.aux_state, &entry.qsa_key_rows, 1, new_held);
    } else {
        if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
        entry.qsa_key_rows = 0;
    }
    entry.qsa_hist_rows = keep;
    const keep_blocks: c_int = @divTrunc(keep, @max(entry.qsa_ratio, 1));
    if (entry.qsa_pooled_buf.ctx != null) {
        if (keep_blocks > 0 and mlx.getShape(entry.qsa_pooled_buf)[1] >= keep_blocks) {
            try capBufReslice(s, entry.qsa_pooled_buf, &entry.qsa_pooled, &entry.qsa_pooled_blocks, 1, keep_blocks);
        } else {
            if (entry.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(entry.qsa_pooled);
            entry.qsa_pooled = .{ .ctx = null };
            entry.qsa_pooled_blocks = 0;
        }
    } else {
        try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, false);
    }
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    entry.qsa_score_bank = .{ .ctx = null };
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    return true;
}

fn capBufReslice(s: mlx.mlx_stream, buf: mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, axis: usize, new_count: c_int) !void {
    const sh = mlx.getShape(buf);
    const nd: usize = sh.len;
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };
    count.* = 0;
    if (new_count <= 0) return;
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (sh, 0..) |d, i| stop[i] = d;
    stop[axis] = new_count;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf, &start, nd, &stop, nd, &strides, nd, s));
    count.* = new_count;
}

/// Keep the pooled-key rows that are still complete blocks of a key history
/// truncated to `keep_rows` (the block size is implied by the two shapes).
/// Cut the pooled bank to the blocks `keep_rows` completes. `materialize` = a fresh buffer; false = a slice view.
pub fn truncatePooled(pooled: *mlx.mlx_array, keep_rows: c_int, ratio: c_int, s: mlx.mlx_stream, materialize: bool) !void {
    if (pooled.ctx == null) return;
    const ps = mlx.getShape(pooled.*);
    const keep_blocks: c_int = @divTrunc(keep_rows, ratio);
    if (keep_blocks >= ps[1]) return;
    if (keep_blocks <= 0) {
        _ = mlx.mlx_array_free(pooled.*);
        pooled.* = .{ .ctx = null };
        return;
    }
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ ps[0], keep_blocks, ps[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var view = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&view, pooled.*, &start, 3, &stop, 3, &strides, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(view);
        return err;
    };
    const owned = if (materialize) blk: {
        defer _ = mlx.mlx_array_free(view);
        break :blk try materializedOwnedCopy(s, view);
    } else view;
    _ = mlx.mlx_array_free(pooled.*);
    pooled.* = owned;
}

// ── tests ──
