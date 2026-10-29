//! copied_from: src/plugins/qwen4_exp/qsa/checkpoint.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA indexer state inside SSM checkpoints and snapshots: whether an entry's aux holds the key history, how many
//! rows it covers, and the capture, trim and restore of the pooled blocks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const SSMCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").SSMCheckpoint;
const captureSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").captureSsmCheckpoint;
const qsaHistoryBytesHeld = @import("../../../engines/mlx/ssm/checkpoint.zig").qsaHistoryBytesHeld;
const restoreSsmCheckpoint = @import("../../../engines/mlx/ssm/checkpoint.zig").restoreSsmCheckpoint;
const ssmCheckpointBytes = @import("../../../engines/mlx/ssm/checkpoint.zig").ssmCheckpointBytes;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntrySnapshot;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const qsaHistoryShareEnabled = @import("gather.zig").qsaHistoryShareEnabled;
const QSA_RING_ROWS = @import("history.zig").QSA_RING_ROWS;
const qsaLeftoverAt = @import("history.zig").qsaLeftoverAt;
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const truncatePooled = @import("history.zig").truncatePooled;

/// QSA indexer key history lives on full-attention layers: `aux_state` is
/// `[B, kv, idx_hd]` and there is no conv recurrence (`conv_state` null or
/// an empty handle). PLE's dilated-conv window is `aux_state` on a GDN
/// layer that HAS a real `conv_state` — that window is O(1) and stays in
/// every checkpoint.
pub fn ssmAuxIsQsaHistory(e: *const SSMCacheEntry) bool {
    if (e.qsa_pooled.ctx != null) return true;
    if (e.aux_state.ctx == null) return false;
    if (e.conv_state.ctx == null) return true;
    return mlx.mlx_array_size(e.conv_state) == 0;
}

pub fn qsaHistoryRows(e: *const SSMCacheEntry) c_int {
    if (e.qsa_hist_rows > 0) return e.qsa_hist_rows;
    if (!ssmAuxIsQsaHistory(e)) return 0;
    if (e.aux_state.ctx == null) return 0;
    const sh = mlx.getShape(e.aux_state);
    return if (sh.len >= 2) sh[1] else 0;
}

/// `ssmAuxIsQsaHistory` over a snapshot layer.
pub fn snapshotHasQsaHistory(l: *const SSMCacheEntrySnapshot) bool {
    return ssmAuxIsQsaHistory(&.{
        .conv_state = l.conv_state,
        .ssm_state = l.ssm_state,
        .aux_state = l.aux_state,
        .qsa_pooled = l.qsa_pooled,
        .initialized = l.initialized,
    });
}

pub fn checkpointHasQsaHistory(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (snapshotHasQsaHistory(l)) return true;
    }
    return false;
}

/// Whether the pooled indexer bank — the history proper — lives on THIS checkpoint. Every
/// checkpoint carries its own raw leftover, so `checkpointHasQsaHistory` cannot answer "is this
/// the bank's home": a mid-block snap says yes on the leftover alone.
pub fn checkpointHasQsaPooled(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (l.qsa_pooled.ctx != null) return true;
    }
    return false;
}

pub fn checkpointQsaAuxRows(cp: *const SSMCheckpoint) c_int {
    var r: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.qsa_rows > r) r = l.qsa_rows;
    }
    return r;
}

/// Raw leftover rows a checkpoint carries (0 on a non-QSA layer, and on a QSA layer whose
/// position is block-aligned).
fn checkpointQsaLeftoverRows(cp: *const SSMCheckpoint) c_int {
    var rows: c_int = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) continue;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len >= 2 and sh[1] > rows) rows = sh[1];
    }
    return rows;
}

/// Whether a QSA restore at `pos` has its whole indexer history: `cp` — the checkpoint being
/// restored — carries the raw leftover for `pos` (the `pos % ratio` rows the next append
/// re-pools), and `src` a pooled bank reaching `pos`. A ring sliced at another position is a
/// miss, never a silently wrong leftover.
pub fn checkpointQsaCoversPos(cp: *const SSMCheckpoint, src: *const SSMCheckpoint, pos: usize) bool {
    const logical = checkpointQsaAuxRows(src);
    if (logical == 0) return true;
    const p: c_int = @intCast(pos);
    if (logical < p) return false;
    var has_pooled = false;
    var pblocks: c_int = 0;
    var ratio: c_int = 4;
    for (src.layers) |*l| {
        if (l.qsa_ratio > 0) ratio = l.qsa_ratio;
        if (l.qsa_pooled.ctx == null) continue;
        has_pooled = true;
        const ps = mlx.getShape(l.qsa_pooled);
        if (ps.len >= 2 and ps[1] > pblocks) pblocks = ps[1];
    }
    const r = @max(ratio, 1);
    if (has_pooled and pblocks < @divTrunc(p, r)) return false;
    if (!has_pooled) return qsaSourceHoldsRowsAt(src, p, p);
    const want_lv = @mod(p, r);
    const cp_rows = checkpointQsaLeftoverRows(cp);
    if (cp_rows == 0) return qsaSourceHoldsRowsAt(src, p, want_lv);
    if (cp_rows == want_lv) return true;
    return qsaSourceHoldsRowsAt(cp, p, want_lv) or qsaSourceHoldsRowsAt(src, p, want_lv);
}

pub fn qsaEntrySatisfiesForward(e: *const SSMCacheEntry, pos: c_int) bool {
    if (!ssmAuxIsQsaHistory(e)) return true;
    if (qsaHistoryRows(e) != pos) return false;
    const ratio = @max(e.qsa_ratio, 1);
    const blocks: c_int = if (e.qsa_pooled.ctx != null) mlx.getShape(e.qsa_pooled)[1] else 0;
    const held: c_int = if (e.aux_state.ctx != null) mlx.getShape(e.aux_state)[1] else 0;
    return blocks * ratio + held >= pos;
}

pub fn qsaRestoreSatisfiesForward(entries: []const SSMCacheEntry, pos: usize) bool {
    const p: c_int = @intCast(pos);
    for (entries) |*e| {
        if (!qsaEntrySatisfiesForward(e, p)) return false;
    }
    return true;
}

pub fn checkpointListHasQsaPooled(cps: []const SSMCheckpoint) bool {
    for (cps) |*cp| {
        if (checkpointHasQsaPooled(cp)) return true;
    }
    return false;
}

/// Whether every QSA layer of `src` physically holds the `want` raw rows ending at `pos`.
fn qsaSourceHoldsRowsAt(src: *const SSMCheckpoint, pos: c_int, want: c_int) bool {
    if (want <= 0) return true;
    var any = false;
    for (src.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx == null) return false;
        const sh = mlx.getShape(l.aux_state);
        if (sh.len < 2) return false;
        const hist: c_int = if (l.qsa_rows > 0) l.qsa_rows else sh[1];
        if (pos > hist or pos - want < hist - sh[1]) return false;
        any = true;
    }
    return any;
}

/// Copy QSA aux/pooled from the live entries onto the LATEST checkpoint —
/// ONE materialized copy, sliced to that checkpoint's position (a cancel
/// handoff attaches while the live history is ahead of the last stride
/// snap). Older snaps in `cps` drop any history they carried. No-op when
/// `cps` is empty or no layer holds QSA history.
///
/// The copy arm: the copy coexists with the live buffer for the whole decode.
pub fn attachQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .copy);
}

/// Rows a handed-off history may pin past the checkpoint's position before the handoff
/// materializes instead: an entry must never pin much more than `ssmCheckpointBytes` bills.
pub const QSA_HANDOFF_MAX_SLACK_ROWS: c_int = 16384;

/// The commit boundary: hand the slot's live QSA history to the latest checkpoint as a slice
/// view of the live capacity buffer, sliced to that checkpoint's position. The slot dies right
/// after, so the entry ends up the buffer's only owner. Too much slack materializes instead.
pub fn handoffQsaHistoryToLatest(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return attachQsaHistoryToLatestMode(cps, live, s, .share);
}

/// Every "the slot is about to die" seam: share when the switch is on, copy otherwise.
pub fn attachQsaHistoryOnHandoff(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream) !void {
    return if (qsaHistoryShareEnabled())
        handoffQsaHistoryToLatest(cps, live, s)
    else
        attachQsaHistoryToLatest(cps, live, s);
}

pub const QsaAttachMode = enum { copy, share };

/// Does sharing `buf_rows` rows of capacity for a `keep`-row snap pin more slack than `QSA_HANDOFF_MAX_SLACK_ROWS`?
pub fn qsaHandoffExceedsSlack(buf_rows: c_int, keep: c_int) bool {
    return buf_rows - keep > QSA_HANDOFF_MAX_SLACK_ROWS;
}

/// The buffer a share of `src` would pin: the capacity buffer when the accelerator holds one, else the published history.
fn qsaHandoffMustMaterialize(src: *const SSMCacheEntry, keep: c_int) bool {
    const buf_rows: c_int = if (src.qsa_key_buf.ctx != null)
        mlx.getShape(src.qsa_key_buf)[1]
    else if (src.aux_state.ctx != null)
        mlx.getShape(src.aux_state)[1]
    else
        0;
    return qsaHandoffExceedsSlack(buf_rows, keep);
}

fn attachQsaHistoryToLatestMode(cps: []SSMCheckpoint, live: []const SSMCacheEntry, s: mlx.mlx_stream, mode: QsaAttachMode) !void {
    if (cps.len == 0) return;
    const cp = &cps[cps.len - 1];
    if (cp.layers.len != live.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(cp.pos);
    var copied: usize = 0;
    for (cp.layers, live) |*dst, src| {
        if (!ssmAuxIsQsaHistory(&src)) continue;
        const materialize = switch (mode) {
            .copy => true,
            .share => qsaHandoffMustMaterialize(&src, keep) or src.qsa_pooled_buf.ctx == null,
        };
        if (src.qsa_pooled.ctx != null) {
            if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, src.qsa_ratio, s, materialize);
        }
        dst.qsa_ratio = src.qsa_ratio;
        copied += 1;
    }
    if (copied == 0) return;
    recordQsaRowsOnCheckpoints(cps, live);
    keepOnlyLatestQsaHistory(cps);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (cp.layers) |*l| {
        if (!snapshotHasQsaHistory(l)) continue;
        inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

fn recordQsaRowsOnCheckpoints(cps: []SSMCheckpoint, live: []const SSMCacheEntry) void {
    for (cps) |*cp| {
        if (cp.layers.len != live.len) continue;
        const pos_rows: c_int = @intCast(cp.pos);
        for (cp.layers, live) |*dst, src| {
            if (!ssmAuxIsQsaHistory(&src)) continue;
            if (dst.qsa_rows == 0) dst.qsa_rows = pos_rows;
            dst.qsa_ratio = src.qsa_ratio;
        }
    }
}

/// Keep the QSA history on the NEWEST snap that carries it and drop it from
/// every older one. A prefix-extend merge inherits the previous turn's latest
/// snap, so without this an agent session holds one full copy per turn.
pub fn keepOnlyLatestQsaHistory(cps: []SSMCheckpoint) void {
    var i = cps.len;
    var kept = false;
    while (i > 0) {
        i -= 1;
        if (!checkpointHasQsaPooled(&cps[i])) continue;
        if (!kept) {
            kept = true;
            continue;
        }
        for (cps[i].layers) |*l| {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = .{ .ctx = null };
        }
    }
}

/// Any live QSA layer holds indexer history. On a QSA arch a restore that
/// leaves this false cannot prefill (QsaHistoryGap) — the cache treats it
/// as a miss.
pub fn entriesHaveQsaHistory(entries: []const SSMCacheEntry) bool {
    for (entries) |*e| {
        if (ssmAuxIsQsaHistory(e)) return true;
    }
    return false;
}

/// Overlay QSA history from `src_cp` (the entry's latest snap, full length)
/// onto `entries` already restored to `pos`. Slices aux to `pos` rows and
/// pooled to `pos/ratio` blocks. No-op on PLE/GDN layers (they already
/// hold the per-position window from `restoreSsmCheckpoint`).
/// `materialize` false = the destination holds a slice of `src_cp`'s bank (safe when the
/// checkpoint outlives the slot's first append); true = an owned copy, required when the
/// source is a transient such as the SSD tier's overlay.
pub fn applyQsaHistoryAt(entries: []SSMCacheEntry, src_cp: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream, materialize: bool) !void {
    if (entries.len != src_cp.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (entries, src_cp.layers) |*dst, src| {
        if (!snapshotHasQsaHistory(&src)) continue;
        // After restoreSsmCheckpoint a QSA layer has no conv window.
        if (dst.conv_state.ctx != null and mlx.mlx_array_size(dst.conv_state) > 0) continue;
        const src_hist: c_int = if (src.qsa_rows > 0) src.qsa_rows else blk: {
            if (src.qsa_pooled.ctx == null) break :blk 0;
            break :blk keep;
        };
        if (src_hist < keep) return error.QsaHistoryGap;
        const own = dst.aux_state;
        dst.aux_state = .{ .ctx = null };
        if (dst.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(dst.qsa_pooled);
        dst.qsa_pooled = .{ .ctx = null };
        const ratio = @max(src.qsa_ratio, 1);
        if (src.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(src.qsa_pooled);
            const need_blocks = @divTrunc(keep, ratio);
            if (ps.len < 2 or ps[1] < need_blocks) return error.QsaHistoryGap;
            if (materialize) {
                dst.qsa_pooled = try materializedOwnedCopy(s, src.qsa_pooled);
            } else {
                dst.qsa_pooled = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&dst.qsa_pooled, src.qsa_pooled));
            }
            try truncatePooled(&dst.qsa_pooled, keep, ratio, s, materialize);
        }
        const from_src = try qsaLeftoverAt(src.aux_state, src_hist, keep, ratio, s);
        if (from_src.ctx != null) {
            if (own.ctx != null) _ = mlx.mlx_array_free(own);
            dst.aux_state = from_src;
        } else {
            dst.aux_state = if (own.ctx != null) own else .{ .ctx = null };
        }
        dst.qsa_ratio = src.qsa_ratio;
        dst.qsa_hist_rows = keep;
        dst.qsa_key_rows = if (dst.aux_state.ctx != null) mlx.getShape(dst.aux_state)[1] else 0;
        if (dst.qsa_pooled.ctx != null) {
            const ps = mlx.getShape(dst.qsa_pooled);
            dst.qsa_pooled_blocks = if (ps.len >= 2) ps[1] else 0;
        }
    }
}

/// Copy QSA aux/pooled from `src` onto `dst`, sliced to `pos` rows. Used when
/// a byte-budget trim would drop the latest snap — the only one that carries
/// the indexer history. A 122880-token trim that freed that snap left aux
/// empty; the next prefill then reshaped a 4096-row chunk into
/// `(1, nb, 4, 128)` and aborted (MLX reshape, 77% RAM — not an OOM).
/// `pos` is `dst.pos`: a checkpoint's aux is normalized to exactly `pos % ratio` rows at
/// capture, which is what the keep-own arm relies on when the source cannot supply them.
pub fn sliceQsaHistoryOntoCheckpoint(dst: *SSMCheckpoint, src: *const SSMCheckpoint, pos: usize, s: mlx.mlx_stream) !void {
    if (dst.layers.len != src.layers.len) return error.SsmCheckpointLayerMismatch;
    const keep: c_int = @intCast(pos);
    for (dst.layers, src.layers) |*d, s_l| {
        if (!snapshotHasQsaHistory(&s_l)) continue;
        if (d.conv_state.ctx != null and mlx.mlx_array_size(d.conv_state) > 0) continue;
        const ratio = @max(s_l.qsa_ratio, 1);
        if (s_l.qsa_pooled.ctx != null) {
            if (d.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(d.qsa_pooled);
            d.qsa_pooled = try materializedOwnedCopy(s, s_l.qsa_pooled);
            try truncatePooled(&d.qsa_pooled, keep, ratio, s, true);
        }
        const src_hist: c_int = if (s_l.qsa_rows > 0) s_l.qsa_rows else keep;
        const leftover_n = @mod(keep, ratio);
        const leftover = try qsaLeftoverAt(s_l.aux_state, src_hist, keep, ratio, s);
        if (leftover.ctx != null) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = leftover;
        } else if (leftover_n == 0) {
            if (d.aux_state.ctx != null) _ = mlx.mlx_array_free(d.aux_state);
            d.aux_state = .{ .ctx = null };
        }
        d.qsa_ratio = s_l.qsa_ratio;
        d.qsa_rows = keep;
    }
    // Materialize now: an unevaluated copy node keeps `src`'s buffer alive.
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var n: usize = 0;
    for (dst.layers) |*d| {
        if (!snapshotHasQsaHistory(d)) continue;
        inline for (.{ d.aux_state, d.qsa_pooled }) |arr| {
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
                n += 1;
            }
        }
    }
    if (n > 0) _ = mlx.mlx_eval(vec);
}

// ── tests ──

fn qsaCoverTestCp(ring: c_int, blocks: c_int, rows: c_int, ratio: c_int, hd: c_int, s: mlx.mlx_stream) !SSMCheckpoint {
    const layers = try std.testing.allocator.alloc(SSMCacheEntrySnapshot, 1);
    errdefer std.testing.allocator.free(layers);
    layers[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio, .qsa_rows = rows };
    var cp: SSMCheckpoint = .{ .pos = @intCast(rows), .layers = layers };
    errdefer cp.deinit(std.testing.allocator);
    if (ring > 0) {
        layers[0].aux_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].aux_state, &[_]c_int{ 1, ring, hd }, 3, .float32, s));
    }
    if (blocks > 0) {
        layers[0].qsa_pooled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&layers[0].qsa_pooled, &[_]c_int{ 1, blocks, hd }, 3, .float32, s));
    }
    return cp;
}
