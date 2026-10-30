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

/// Whether the pooled indexer bank — the history proper — lives on THIS checkpoint. Every
/// checkpoint carries its own raw leftover, so `checkpointHasQsaHistory` cannot answer "is this
/// the bank's home": a mid-block snap says yes on the leftover alone.
pub fn checkpointHasQsaPooled(cp: *const SSMCheckpoint) bool {
    for (cp.layers) |*l| {
        if (l.qsa_pooled.ctx != null) return true;
    }
    return false;
}

pub fn qsaEntrySatisfiesForward(e: *const SSMCacheEntry, pos: c_int) bool {
    if (!ssmAuxIsQsaHistory(e)) return true;
    if (qsaHistoryRows(e) != pos) return false;
    const ratio = @max(e.qsa_ratio, 1);
    const blocks: c_int = if (e.qsa_pooled.ctx != null) mlx.getShape(e.qsa_pooled)[1] else 0;
    const held: c_int = if (e.aux_state.ctx != null) mlx.getShape(e.aux_state)[1] else 0;
    return blocks * ratio + held >= pos;
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

// ── tests ──
