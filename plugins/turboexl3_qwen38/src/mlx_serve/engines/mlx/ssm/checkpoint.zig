//! copied_from: src/engines/mlx/ssm/checkpoint.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/ssm/checkpoint.
//! Owns SSMCheckpoint, captureSsmCheckpoint, compact_conv_state_override, compact_conv_state_copy_count, cadence_conv_eval_count, compactConvStateEnabled, and 15 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const applyQsaHistoryAt = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").applyQsaHistoryAt;
const attachQsaHistoryToLatest = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").attachQsaHistoryToLatest;
const qsaHistoryRows = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").qsaHistoryRows;
const ssmAuxIsQsaHistory = @import("../../../plugins/qwen4_exp/qsa/checkpoint.zig").ssmAuxIsQsaHistory;
const qsaLeftoverAt = @import("../../../plugins/qwen4_exp/qsa/history.zig").qsaLeftoverAt;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const KVCache = @import("../kv/cache.zig").KVCache;
const testing = @import("../kv/cache.zig").testing;
const SSMCacheEntry = @import("entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("entry.zig").ssmFreeSpecCapture;
const ssmRestore = @import("entry.zig").ssmRestore;
const ssmRollbackFromCapture = @import("entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("entry.zig").ssmSnapshotDeinit;
const ssmLiveAux = @import("group.zig").ssmLiveAux;
const ssmMaterializeLive = @import("group.zig").ssmMaterializeLive;

/// Phase 1 (performance-plan): per-position SSM checkpoint covering ALL hybrid
/// layers in the model. Captures the SSM/conv state after the model has been
/// forwarded over the first `pos` tokens of some prompt. Used by the hot prefix
/// cache to restore mid-sequence SSM state on a multi-turn warm request, so we
/// don't have to cold-prefill the shared prefix every turn.
///
/// A single checkpoint = one `SSMCacheEntrySnapshot` per layer (matching the
/// shape of `ctx.ssm_entries`). Layers whose ssm/conv state is uninitialized
/// at snapshot time hold a null-handle snapshot — `ssmRestore` is null-safe.
///
/// Memory: per GatedDeltaNet layer, the Hv x Dv x Dk state in its own dtype
/// (`ModelConfig.ssmStateDtype`) plus the conv window. Bounded via
/// `HotPrefixCache.max_kv_bytes` (counts toward the same budget as KV).
pub const SSMCheckpoint = struct {
    /// 1-based KV position immediately AFTER forwarding `pos` tokens. The
    /// caller restores the slot's KVCache to exactly this position alongside
    /// the SSM state, so the model behaves as if it had only seen the first
    /// `pos` tokens of the prompt.
    pos: usize,
    /// Per-layer SSM snapshots — same length as `ctx.ssm_entries`. Non-SSM
    /// layers (plain attention) get a null-handle snapshot which is a no-op
    /// in `ssmRestore`.
    layers: []SSMCacheEntrySnapshot,

    pub fn deinit(self: *SSMCheckpoint, allocator: std.mem.Allocator) void {
        for (self.layers) |*l| ssmSnapshotDeinit(l);
        allocator.free(self.layers);
        self.layers = &[_]SSMCacheEntrySnapshot{};
        self.pos = 0;
    }
};

/// Snapshot of every entry in `ssm_entries` at the current point. Caller owns
/// the resulting buffer (free via `SSMCheckpoint.deinit`).
///
/// Unlike the per-decode-step `ssmSnapshot` (PLD rollback — transient, so a
/// cheap refcount-share is right), checkpoints OUTLIVE the request inside the
/// hot prefix cache. The live conv/ssm states are routinely shared-buffer
/// SLICES of much larger parents (the prefill chunk's conv input
/// [B,(k-1)+T,C], the GDN capture sequence), so sharing the handle silently
/// retained the whole parent — the ~3.4x "[hot-cache] resident" under-count
/// on hybrid archs. `materializedOwnedCopy` forces a fresh buffer; the copies
/// are evaluated here, in one batch, so the parents' lifetimes end with the
/// chunk's activation graph instead of with the cache entry.
pub fn captureSsmCheckpoint(
    allocator: std.mem.Allocator,
    ssm_entries: []const SSMCacheEntry,
    pos: usize,
    s: mlx.mlx_stream,
) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        errdefer ssmSnapshotDeinit(&out);
        // Per-field null guards mirror `ssmSnapshot` — either state may
        // legitimately be null (LFM2 gated_conv never sets ssm_state).
        const live_conv = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].conv_state else src.conv_state) else src.conv_state;
        const live_ssm = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) g.layers[src.group_layer].ssm_state else src.ssm_state) else src.ssm_state;
        const live_ple = ssmLiveAux(src);
        if (live_conv.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_conv);
            _ = mlx.mlx_array_free(out.conv_state);
            out.conv_state = c;
        }
        if (live_ssm.ctx != null) {
            const c = try ssmMaterializeLive(s, src, live_ssm);
            _ = mlx.mlx_array_free(out.ssm_state);
            out.ssm_state = c;
        }
        // PLE conv window (O(1), rides a GDN layer) is per-position state and
        // belongs in every checkpoint. QSA indexer history (aux && !conv) is
        // a prefix of one buffer — copying it at every stride is what OOM'd
        // a 400k prefill while auto-context advertised 700k–1M. One copy is
        // attached to the latest checkpoint after prefill.
        const grouped_ple = src.group != null and src.group.?.live and src.group_layer < src.group.?.layers.len and src.group.?.layers[src.group_layer].aux_state.ctx != null;
        if (grouped_ple or (src.aux_state.ctx != null and !ssmAuxIsQsaHistory(src))) {
            out.aux_state = try ssmMaterializeLive(s, src, live_ple);
        }
        out.qsa_ratio = src.qsa_ratio;
        if (!grouped_ple and ssmAuxIsQsaHistory(src)) {
            out.qsa_rows = @intCast(pos);
            if (src.aux_state.ctx != null) {
                out.aux_state = try qsaLeftoverAt(src.aux_state, qsaHistoryRows(src), @intCast(pos), src.qsa_ratio, s);
            }
        }
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    // Materialize all copies in one batch so the parent buffers can be
    // released with the prefill chunk's graph. Without this eval the lazy
    // copy node itself keeps the parent alive.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, arr);
                    count += 1;
                }
            }
        }
        if (count > 0) _ = mlx.mlx_eval(vec);
    }
    return .{ .pos = pos, .layers = layers };
}

pub var cadence_conv_eval_count: u64 = 0;

pub const compactConvStateEnabled = @import("../../../sdk/blocks/conv/depthwise.zig").compactConvStateEnabled;

/// A second owner for `cp`'s state: fresh handles and a fresh layer slice,
/// refcount-SHARING every underlying buffer (`mlx_array_set`, the same idiom
/// `ssmSnapshot` and `KVCache.snapshot` use). Both checkpoints then free
/// independently and the DATA outlives whichever dies first.
///
/// This is what makes checkpoint inheritance cheap enough to be unconditional:
/// a commit built on a restored prefix carries the donor entry's checkpoints
/// without re-copying multi-GB of GDN state, and evicting the donor later
/// frees nothing the inheritor still needs. `captureSsmCheckpoint` must
/// MATERIALIZE (its sources are slices of a live prefill graph); a checkpoint
/// is already materialized, so sharing it aliases nothing transient.
pub fn shareSsmCheckpoint(allocator: std.mem.Allocator, cp: *const SSMCheckpoint) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, cp.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (cp.layers, 0..) |*src, i| {
        var out: SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = src.initialized,
        };
        // Per-field null guards: mlx_array_set on a null source aborts the
        // process via mlx-c's default error handler (see `ssmSnapshot`).
        if (src.conv_state.ctx != null) _ = mlx.mlx_array_set(&out.conv_state, src.conv_state);
        if (src.ssm_state.ctx != null) _ = mlx.mlx_array_set(&out.ssm_state, src.ssm_state);
        if (src.aux_state.ctx != null) {
            out.aux_state = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.aux_state, src.aux_state);
        }
        if (src.qsa_pooled.ctx != null) {
            out.qsa_pooled = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&out.qsa_pooled, src.qsa_pooled);
        }
        out.qsa_ratio = src.qsa_ratio;
        out.qsa_rows = src.qsa_rows;
        out.ple_prev = src.ple_prev;
        out.ple_prev_valid = src.ple_prev_valid;
        layers[i] = out;
        built = i + 1;
    }
    return .{ .pos = cp.pos, .layers = layers };
}

/// Restore every layer of `ssm_entries` from `cp`. Mirrors the per-layer
/// `ssmRestore` pattern — null-safe on either side.
pub fn restoreSsmCheckpoint(
    ssm_entries: []SSMCacheEntry,
    cp: *const SSMCheckpoint,
) !void {
    if (ssm_entries.len != cp.layers.len) return error.SsmCheckpointLayerMismatch;
    for (ssm_entries, cp.layers) |*dst, *src| {
        try ssmRestore(dst, src);
    }
}

/// Total bytes held by an SSM checkpoint (sum of conv_state + ssm_state across
/// all layers). Used for hot-cache memory budgeting alongside `KVCacheSnapshot`
/// bytes.
pub fn ssmCheckpointBytes(cp: *const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cp.layers) |l| {
        inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
            if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
        }
    }
    return total;
}

/// Which retention policy a checkpoint list is thinned with. The span-preserving thin moves
/// where a warm turn restores from and was measured on qwen4_exp only, so the policy is chosen
/// by `longCtxGated()` at each site and the ungated value names that site's previous
/// behaviour: `.oldest` (prefill capture, disk tier), `.min_span` (hot-cache merge/shed),
/// `.min_span_recency` (`.min_span` plus a dense newest quarter; qwen4_exp).
pub const ThinPolicy = enum { oldest, min_span, min_span_recency };

/// The checkpoint index to drop when a retention list is over its cap: the interior position
/// whose removal widens the coverage gap least; index 0 and the last are always kept. One
/// selection for every retention site (#330: end-anchored survivors left no affordable trim point).
pub fn spanPreservingDropIndex(
    comptime T: type,
    items: []const T,
    comptime posOf: fn (*const T) usize,
    policy: ThinPolicy,
    /// An index the caller knows a later turn must restore from (a media boundary): never
    /// selected. When it is the only candidate the oldest goes instead — that one sits below
    /// it, so the protected position survives either way.
    protect: ?usize,
    /// How many LOW positions are anchored as well (#794: thinning must not
    /// migrate the grid floor upward). 0 = the plain policy; a caller that
    /// wants the legacy drop-oldest behaviour also passes 0.
    low_anchors: usize,
) usize {
    std.debug.assert(items.len > 0);
    const last = items.len - 1;
    // The cap outranks the anchors: the newest position is never a drop candidate,
    // so if every remaining index is anchored the lowest anchor gives way.
    if (items.len <= low_anchors + 1) return @min(low_anchors, last);
    if (policy == .oldest) return @min(low_anchors, last);
    if (items.len < 3) return @min(low_anchors, last);
    // Recency bias: the newest quarter stays at capture density (a warm turn that edits near
    // the end restores from there); everything below it is thinned span-preservingly.
    const scan_end = if (policy == .min_span_recency and items.len >= RECENCY_DENSE_MIN)
        items.len - items.len / 4
    else
        items.len;
    // Index 0 is never a candidate (the span is measured across k-1), so the scan
    // starts one past the last anchor — and at 1 when nothing is anchored, which
    // is the position the plain policies have always started from.
    const scan_start = @max(low_anchors, 1);
    var best_at: usize = scan_start;
    var best_span: usize = std.math.maxInt(usize);
    var k: usize = scan_start;
    while (k + 1 < items.len and k < scan_end) : (k += 1) {
        if (protect) |p| if (k == p) continue;
        const span = posOf(&items[k + 1]) -| posOf(&items[k - 1]);
        if (span < best_span) {
            best_span = span;
            best_at = k;
        }
    }
    // Nothing scannable (recency-protected or anchored): the oldest goes, which
    // sits below every anchor, so the anchors still survive.
    if (best_span == std.math.maxInt(usize)) return 0;
    return best_at;
}

/// `spanPreservingDropIndex` over a bare ascending position list, with the first
/// `low_anchors` positions protected from thinning.
pub fn positionDropIndexLowAnchored(positions: []const u32, policy: ThinPolicy, low_anchors: usize) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null, low_anchors);
}

/// Shortest list that reserves a dense newest quarter.
const RECENCY_DENSE_MIN: usize = 8;

fn checkpointPosOf(cp: *const SSMCheckpoint) usize {
    return cp.pos;
}

fn u32PosOf(p: *const u32) usize {
    return p.*;
}

fn usizePosOf(p: *const usize) usize {
    return p.*;
}

/// `spanPreservingDropIndex` over a checkpoint list.
pub fn ssmCheckpointDropIndex(cps: []const SSMCheckpoint, policy: ThinPolicy, protect: ?usize) usize {
    return spanPreservingDropIndex(SSMCheckpoint, cps, checkpointPosOf, policy, protect, 0);
}

/// `ssmCheckpointDropIndex` with the first `low_anchors` positions protected
/// from thinning: generation-side retention must let the dense low grid anchors
/// (#794) survive the per-prefill cap, or the persist-side `ssm_low_anchors`
/// has nothing left to protect. 0 = the plain policy, byte-identical to before.
pub fn ssmCheckpointDropIndexLowAnchored(
    cps: []const SSMCheckpoint,
    policy: ThinPolicy,
    protect: ?usize,
    low_anchors: usize,
) usize {
    return spanPreservingDropIndex(SSMCheckpoint, cps, checkpointPosOf, policy, protect, low_anchors);
}

/// `spanPreservingDropIndex` over a bare ascending position list.
pub fn positionDropIndex(positions: []const u32, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null, 0);
}

pub fn positionDropIndexUsize(positions: []const usize, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(usize, positions, usizePosOf, policy, null, 0);
}

// ── tests ──

pub fn qsaHistoryBytesHeld(cps: []const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cps) |cp| {
        for (cp.layers) |l| {
            inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
            }
        }
    }
    return total;
}
