//! The direct model's per-slot state: the attention layers' KV cache (dense or affine, grown in place), each layer's recurrent
//! entry with the verify's capture and partial-accept rollback, the batched tick's merged group, and the prefix cache's
//! checkpoints. The KV types are this file's, which the copies take through the sdk/kv/cache.zig edge; the recurrent
//! entry, its snapshot and checkpoint and the group stay the copies' types (`ForwardCtx` and the QSA history hold them),
//! so the group's methods are `ssmGroupMatches` and `ssmGroupClear` here.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const materializedOwnedCopy = @import("../mlx_serve/mlx/array.zig").materializedOwnedCopy;
const concatAxis0 = @import("../mlx_serve/mlx/array.zig").concatAxis0;
const kv_quant = @import("../mlx_serve/sdk/blocks/attention/kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const DenseKVView = @import("../mlx_serve/sdk/blocks/attention/kv_view.zig").DenseKVView;
const MAX_BATCH_ROWS = @import("../mlx_serve/engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
// GDN's replay and the QSA history, the copies'; PLE's history, the direct ple.zig's.
const gdn_decode = @import("../mlx_serve/sdk/blocks/gdn/decode.zig");
const plePrevAfterAccept = @import("ple.zig").plePrevAfterAccept;
const qsaHistoryRows = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig").qsaHistoryRows;
const ssmAuxIsQsaHistory = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig").ssmAuxIsQsaHistory;
const qsaLeftoverAt = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").qsaLeftoverAt;
const qsaResliceToKeep = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").qsaResliceToKeep;
const ssmFreeQsaState = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const truncatePooled = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").truncatePooled;

const none: mlx.mlx_array = .{ .ctx = null };
const zeros: [8]c_int = @splat(0);
const ones: [8]c_int = @splat(1);

/// `dst` freed and set to `x`.
fn replace(dst: *mlx.mlx_array, x: mlx.mlx_array) void {
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = x;
}

/// A second handle on `x`'s array (none when `x` holds none).
fn ref(x: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    if (x.ctx != null) try mlx.check(mlx.mlx_array_set(&r, x));
    return r;
}

/// The bounds of index `i` of `axis` (the axis kept) in an array of shape `sh`.
fn indexBounds(sh: []const c_int, axis: usize, i: usize) ![2][8]c_int {
    if (sh.len > 8) return error.SsmRankTooHigh;
    var b: [2][8]c_int = .{ zeros, zeros };
    @memcpy(b[1][0..sh.len], sh);
    b[0][axis] = @intCast(i);
    b[1][axis] = @intCast(i + 1);
    return b;
}

/// Index `i` of `axis` of `arr`, the axis kept (a view).
fn ssmAxisView(s: mlx.mlx_stream, arr: mlx.mlx_array, axis: usize, i: usize) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    const b = try indexBounds(sh, axis, i);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &b[0], sh.len, &b[1], sh.len, &ones, sh.len, s));
    return out;
}

/// Rows [from, to) of axis 1 of the 3-D `x` (a view).
fn rows3(s: mlx.mlx_stream, x: mlx.mlx_array, from: c_int, to: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ sh[0], to, sh[2] }, 3, &ones, 3, s));
    return out;
}

// ── KV cache (the attention layers) ──

pub const KVCacheEntry = struct {
    /// Dense: K and V `[B, H, capacity, D]`. Affine: their packed codes `[B, H, capacity, D * bits / 32]` (uint32), and
    /// each row's scales and biases `[B, H, capacity, D / group_size]` (null on a dense cache).
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    keys_scales: mlx.mlx_array,
    keys_biases: mlx.mlx_array,
    values_scales: mlx.mlx_array,
    values_biases: mlx.mlx_array,
    /// Each buffer's first `offset` rows, which attention reads.
    key_view: mlx.mlx_array,
    value_view: mlx.mlx_array,
    key_scales_view: mlx.mlx_array,
    key_biases_view: mlx.mlx_array,
    value_scales_view: mlx.mlx_array,
    value_biases_view: mlx.mlx_array,
    offset: usize,
    initialized: bool,
};

/// An entry's buffers, and their views in the same order.
const buffers = .{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" };
const view_names = .{ "key_view", "value_view", "key_scales_view", "key_biases_view", "value_scales_view", "value_biases_view" };

/// An entry with no buffers, uninitialized.
pub fn newEntry() KVCacheEntry {
    var e: KVCacheEntry = undefined;
    inline for (buffers ++ view_names) |f| @field(e, f) = mlx.mlx_array_new();
    e.offset = 0;
    e.initialized = false;
    return e;
}

fn freeEntry(e: *KVCacheEntry) void {
    inline for (buffers ++ view_names) |f| _ = mlx.mlx_array_free(@field(e, f));
}

fn dropViews(e: *KVCacheEntry) void {
    inline for (view_names) |f| replace(&@field(e, f), mlx.mlx_array_new());
}

/// The first `rows` rows (axis 2) of the 4-D `x` (a view).
fn headRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: usize) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, x, &zeros, 4, &[_]c_int{ sh[0], sh[1], @intCast(rows), sh[3] }, 4, &ones, 4, s));
    return out;
}

/// `v` set to `buf`'s first `rows` rows: the buffer itself when they fill it, else a slice.
fn setView(s: mlx.mlx_stream, v: *mlx.mlx_array, buf: mlx.mlx_array, rows: usize) !void {
    if (rows == KVCache.bufferCapacity(buf)) return mlx.check(mlx.mlx_array_set(v, buf));
    replace(v, try headRows(s, buf, rows));
}

fn rebuild(e: *KVCacheEntry, s: mlx.mlx_stream) !void {
    inline for (buffers, 0..) |b, i| if (@field(e, b).ctx != null) try setView(s, &@field(e, view_names[i]), @field(e, b), e.offset);
}

pub const KVCache = struct {
    entries: []KVCacheEntry,
    step: usize, // absolute sequence position
    allocator: std.mem.Allocator,
    config: KVQuantConfig,

    pub fn init(allocator: std.mem.Allocator, num_layers: u32) !KVCache {
        return initWithConfig(allocator, num_layers, KVQuantConfig.dense);
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, num_layers: u32, config: KVQuantConfig) !KVCache {
        const entries = try allocator.alloc(KVCacheEntry, num_layers);
        for (entries) |*e| e.* = newEntry();
        return .{ .entries = entries, .step = 0, .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.entries) |*e| freeEntry(e);
        self.allocator.free(self.entries);
    }

    /// This cache built over, empty, under `config`: the new one first, so a failure leaves the old one whole.
    pub fn reinit(self: *KVCache, num_layers: u32, config: KVQuantConfig) !void {
        const fresh = try initWithConfig(self.allocator, num_layers, config);
        self.deinit();
        self.* = fresh;
    }

    /// A second owner of every layer's buffers (refcount-shared, no copy; the views are left out).
    pub fn snapshot(self: *const KVCache) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeEntry(e);
            self.allocator.free(out);
        }
        for (self.entries, out) |src, *dst| {
            dst.* = newEntry();
            built += 1;
            try share(dst, src);
        }
        return .{ .entries = out, .step = self.step, .allocator = self.allocator };
    }

    /// Every layer replaced by `snap`'s, its buffers shared; the views are rebuilt on the next read or write.
    pub fn restore(self: *KVCache, snap: *const KVCacheSnapshot) !void {
        std.debug.assert(self.entries.len == snap.entries.len);
        for (self.entries, snap.entries) |*dst, src| {
            freeEntry(dst);
            dst.* = newEntry();
            try share(dst, src);
        }
        self.step = snap.step;
    }

    fn share(dst: *KVCacheEntry, src: KVCacheEntry) !void {
        dst.offset = src.offset;
        dst.initialized = src.initialized;
        if (!src.initialized) return;
        inline for (buffers) |b| @field(dst, b) = try ref(@field(src, b));
    }

    const chunk_step = 256;
    const max_grow_tokens = 8192;

    /// The capacity a grow to `needed` rows takes: +25%, one chunk at least and `max_grow_tokens` at most, in whole
    /// chunks. MLX reuses a freed buffer only for a request within ~2 pages of it, so a fixed step would strand every
    /// superseded buffer in its pool.
    pub fn nextCapacity(current_cap: usize, needed: usize) usize {
        if (needed <= current_cap) return current_cap;
        const grow_by = @min(@max(current_cap / 4, chunk_step), max_grow_tokens);
        const target = @max(needed, current_cap + grow_by);
        return (target + chunk_step - 1) / chunk_step * chunk_step;
    }

    /// The layer's new K/V rows `[B, H, rows, D]` written at its offset, and its views: an affine cache stores them
    /// quantized and reads them dequantized. The K packs have no sliding window: a tail span (`max_seq`) is refused.
    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        if (max_seq != 0) return error.DirectModelKvHasNoWindow;
        const e = &self.entries[layer];
        if (self.config.scheme == .affine) {
            const c = self.config;
            var kq = try kv_quant.quantizeAffine(s, new_k, c.group_size, c.bits);
            defer kq.deinit();
            var vq = try kv_quant.quantizeAffine(s, new_v, c.group_size, c.bits);
            defer vq.deinit();
            try self.write(e, layer, s, &.{ kq.q, vq.q, kq.scales, kq.biases, vq.scales, vq.biases });
            return self.affineView(e, s);
        }
        try self.write(e, layer, s, &.{ new_k, new_v });
        return .{ .k = e.key_view, .v = e.value_view, .owned = false };
    }

    /// The layer's new rows, one array `[B, H, rows, last]` per buffer in `buffers`' order, written at its offset (the
    /// buffers grown in lockstep where they are full, each at its rows' dtype and width), then its views.
    fn write(self: *KVCache, e: *KVCacheEntry, layer: u32, s: mlx.mlx_stream, rows: []const mlx.mlx_array) !void {
        // The views go first: their references would keep the write from donating the buffer.
        dropViews(e);
        const n: usize = @intCast(mlx.getShape(rows[0])[2]);
        if (!e.initialized or e.offset + n > bufferCapacity(e.keys)) {
            const kept: usize = if (e.initialized) e.offset else 0;
            const cap = nextCapacity(if (e.initialized) bufferCapacity(e.keys) else 0, kept + n);
            inline for (buffers, 0..) |b, i| if (i < rows.len) {
                const sh = mlx.getShape(rows[i]);
                try grow(s, &@field(e, b), cap, kept, sh[0], sh[1], sh[3], mlx.mlx_array_dtype(rows[i]));
            };
            e.initialized = true;
        }
        inline for (buffers, 0..) |b, i| if (i < rows.len) try writeAt(s, &@field(e, b), e.offset, rows[i]);
        e.offset += n;
        if (layer == 0) self.step += n;
        try rebuild(e, s);
    }

    /// An affine layer's read: K and V dequantized from the views (owned; lazy, so paid only where attention reads
    /// them), the packed triples borrowed beside them for the kernels that read those.
    fn affineView(self: *const KVCache, e: *const KVCacheEntry, s: mlx.mlx_stream) !DenseKVView {
        const c = self.config;
        const k = try kv_quant.dequantizeAffine(s, e.key_view, e.key_scales_view, e.key_biases_view, c.group_size, c.bits);
        errdefer _ = mlx.mlx_array_free(k);
        const v = try kv_quant.dequantizeAffine(s, e.value_view, e.value_scales_view, e.value_biases_view, c.group_size, c.bits);
        return .{
            .k = k,
            .v = v,
            .owned = true,
            .k_triple_q = e.key_view,
            .k_triple_scales = e.key_scales_view,
            .k_triple_biases = e.key_biases_view,
            .v_triple_q = e.value_view,
            .v_triple_scales = e.value_scales_view,
            .v_triple_biases = e.value_biases_view,
            .has_quant_triple = true,
            .bits = c.bits,
            .group_size = c.group_size,
            .kv_cap = @intCast(bufferCapacity(e.keys)),
        };
    }

    /// Each slot's row of the stacked K/V `[N, H, rows, D]` written into its own cache. An 8-bit affine group (with
    /// `use_batched`) quantizes the stacked rows once and writes each slot's row of the codes.
    pub fn appendFromStacked(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        use_batched: bool,
    ) !void {
        if (use_batched and caches.len > 0 and caches[0].config.scheme == .affine and caches[0].config.bits == 8) {
            if (max_seq != 0) return error.DirectModelKvHasNoWindow;
            const c = caches[0].config;
            var kq = try kv_quant.quantizeAffine(s, stacked_k, c.group_size, c.bits);
            defer kq.deinit();
            var vq = try kv_quant.quantizeAffine(s, stacked_v, c.group_size, c.bits);
            defer vq.deinit();
            for (caches, 0..) |cache, i| {
                var rows: [6]mlx.mlx_array = undefined;
                var n: usize = 0;
                defer for (rows[0..n]) |r| {
                    _ = mlx.mlx_array_free(r);
                };
                for ([_]mlx.mlx_array{ kq.q, vq.q, kq.scales, kq.biases, vq.scales, vq.biases }) |x| {
                    rows[n] = try ssmAxisView(s, x, 0, i);
                    n += 1;
                }
                try cache.write(&cache.entries[layer], layer, s, &rows);
            }
            return;
        }
        for (caches, 0..) |cache, i| {
            const k = try ssmAxisView(s, stacked_k, 0, i);
            defer _ = mlx.mlx_array_free(k);
            const v = try ssmAxisView(s, stacked_v, 0, i);
            defer _ = mlx.mlx_array_free(v);
            var view = try cache.update(layer, k, v, s, max_seq);
            view.deinit();
        }
    }

    /// The layer's K/V views (dequantized on an affine cache); a restored entry's are rebuilt first.
    pub fn denseView(self: *KVCache, layer: u32, s: mlx.mlx_stream) !DenseKVView {
        const e = &self.entries[layer];
        if (e.initialized and e.key_view.ctx == null) try rebuild(e, s);
        if (self.config.scheme == .affine and e.initialized) return self.affineView(e, s);
        return .{ .k = e.key_view, .v = e.value_view, .owned = false };
    }

    pub fn bufferCapacity(arr: mlx.mlx_array) usize {
        const shape = mlx.getShape(arr);
        if (shape.len < 3) return 0;
        return @intCast(shape[2]);
    }

    /// `buf` reallocated at `cap` rows `[B, H, cap, dim]`, its first `kept` rows copied over.
    fn grow(s: mlx.mlx_stream, buf: *mlx.mlx_array, cap: usize, kept: usize, B: c_int, H: c_int, dim: c_int, dtype: mlx.mlx_dtype) !void {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_zeros(&out, &[_]c_int{ B, H, @intCast(cap), dim }, 4, dtype, s));
        if (kept > 0) {
            const old = try headRows(s, buf.*, kept);
            defer _ = mlx.mlx_array_free(old);
            var copied = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(copied);
            try mlx.check(mlx.mlx_slice_update(&copied, out, old, &zeros, 4, &[_]c_int{ B, H, @intCast(kept), dim }, 4, &ones, 4, s));
            replace(&out, copied);
        }
        replace(buf, out);
    }

    /// `chunk` `[B, H, rows, D]` written into `buf` from row `at`.
    fn writeAt(s: mlx.mlx_stream, buf: *mlx.mlx_array, at: usize, chunk: mlx.mlx_array) !void {
        const sh = mlx.getShape(buf.*);
        const from: c_int = @intCast(at);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_slice_update(&out, buf.*, chunk, &[_]c_int{ 0, 0, from, 0 }, 4, &[_]c_int{ sh[0], sh[1], from + mlx.getShape(chunk)[2], sh[3] }, 4, &ones, 4, s));
        replace(buf, out);
    }

    /// Every layer back to its first `len` rows: the offsets move, the buffers keep their capacity.
    pub fn truncate(self: *KVCache, len: usize, s: mlx.mlx_stream) !void {
        self.step = len;
        for (self.entries) |*e| {
            if (!e.initialized or len >= e.offset) continue;
            if (len == 0) {
                freeEntry(e);
                e.* = newEntry();
                continue;
            }
            dropViews(e);
            e.offset = len;
            try rebuild(e, s);
        }
    }
};

/// A `KVCache`'s state at a point, its arrays its own handles (`KVCache.snapshot`, `trimmedCopy`).
pub const KVCacheSnapshot = struct {
    entries: []KVCacheEntry,
    step: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| freeEntry(e);
        self.allocator.free(self.entries);
    }

    /// The first `len` rows in buffers of their own, evaluated: a shared snapshot pins its source's whole capacity,
    /// and so would the copy's lazy graph.
    pub fn trimmedCopy(self: *const KVCacheSnapshot, len: usize, s: mlx.mlx_stream) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeEntry(e);
            self.allocator.free(out);
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.entries, out) |src, *dst| {
            dst.* = newEntry();
            built += 1;
            dst.initialized = src.initialized;
            dst.offset = @min(src.offset, len);
            if (!src.initialized) continue;
            inline for (buffers) |b| if (@field(src, b).ctx != null) {
                @field(dst, b) = try trimRows(s, @field(src, b), dst.offset);
                _ = mlx.mlx_vector_array_append_value(vec, @field(dst, b));
            };
        }
        if (mlx.mlx_vector_array_size(vec) > 0) _ = mlx.mlx_eval(vec);
        return .{ .entries = out, .step = @min(self.step, len), .allocator = self.allocator };
    }
};

/// `x`'s first `rows` rows (axis 2) in a buffer of its own (a slice alone keeps its parent alive).
fn trimRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: usize) !mlx.mlx_array {
    const view = try headRows(s, x, rows);
    defer _ = mlx.mlx_array_free(view);
    return materializedOwnedCopy(s, view);
}

// ── Recurrent entries (every layer) ──

pub const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;

/// A layer's recurrent state at a point: the GDN pair, the aux state (the PLE window, or a QSA history or leftover),
/// the QSA pooled bank and history rows, and the PLE token history.
pub const SSMCacheEntrySnapshot = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntrySnapshot;

const snapshot_arrays = .{ "conv_state", "ssm_state", "aux_state", "qsa_pooled" };

fn ssmSnapshotDeinit(snap: *SSMCacheEntrySnapshot) void {
    inline for (snapshot_arrays) |f| replace(&@field(snap, f), none);
}

fn rowsOf(x: mlx.mlx_array) c_int {
    const sh = mlx.getShape(x);
    return if (sh.len >= 2) sh[1] else 0;
}

/// `dst` given `snap`'s state, its arrays shared. Restores land on fresh entries, never on a grouped one.
pub fn ssmRestore(dst: *SSMCacheEntry, snap: *const SSMCacheEntrySnapshot) !void {
    if (dst.group != null) return error.DirectModelRestoresUngrouped;
    replace(&dst.conv_state, try ref(snap.conv_state));
    replace(&dst.ssm_state, try ref(snap.ssm_state));
    dst.initialized = snap.initialized;
    // The QSA state is replaced, never merged: the entry may hold the previous request's history.
    ssmFreeQsaState(dst);
    dst.aux_state = try ref(snap.aux_state);
    dst.qsa_pooled = try ref(snap.qsa_pooled);
    dst.qsa_ratio = snap.qsa_ratio;
    dst.qsa_hist_rows = snap.qsa_rows;
    dst.ple_prev = snap.ple_prev;
    dst.ple_prev_valid = snap.ple_prev_valid;
    if (dst.aux_state.ctx != null) dst.qsa_key_rows = rowsOf(dst.aux_state);
    if (dst.qsa_pooled.ctx != null) dst.qsa_pooled_blocks = rowsOf(dst.qsa_pooled);
}

/// The verify's capture freed (idempotent): called at the end of every round, so it never outlives one.
pub fn ssmFreeSpecCapture(e: *SSMCacheEntry) void {
    inline for (.{ &e.spec_state_seq, &e.spec_conv_input, &e.spec_state_in, &e.spec_prework.k, &e.spec_prework.v, &e.spec_prework.gb, &e.spec_ple_input }) |a| replace(a, none);
    e.spec_replay = false;
    e.spec_ple_len = 0;
}

/// Slot `i`'s row of a merged entry's verify captures, onto the slot's own entry `e`.
pub fn splitSpecCaptures(s: mlx.mlx_stream, m: *const SSMCacheEntry, e: *SSMCacheEntry, i: usize) !void {
    if (m.spec_state_seq.ctx != null) replace(&e.spec_state_seq, try ssmAxisView(s, m.spec_state_seq, 1, i));
    if (m.spec_conv_input.ctx != null) replace(&e.spec_conv_input, try ssmAxisView(s, m.spec_conv_input, 0, i));
    if (m.spec_ple_input.ctx != null) replace(&e.spec_ple_input, try ssmAxisView(s, m.spec_ple_input, 0, i));
}

/// A partial accept: the entry's state after verify rows 0..accepted, from what the verify captured (no re-forward),
/// and its group row patched to match.
pub fn ssmRollbackFromCapture(entry: *SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
    var patch: SsmPatchFields = .{};
    const acc: c_int = @intCast(accepted);
    const t: c_int = @intCast(verify_len);
    // The aux state: the PLE window slid to the accepted row (the PLE layer), or the QSA key history cut back to it (an
    // attention layer, which has no conv state).
    if (entry.spec_ple_input.ctx != null) {
        const ci = mlx.getShape(entry.spec_ple_input); // [B, window + T, hc*hidden]
        const view = try rows3(s, entry.spec_ple_input, 1 + acc, 1 + acc + ci[1] - t);
        defer _ = mlx.mlx_array_free(view);
        const owned = try materializedOwnedCopy(s, view);
        ssmFreeQsaState(entry);
        entry.aux_state = owned;
        plePrevAfterAccept(entry, accepted, verify_len);
        patch.aux = true;
    } else if ((entry.aux_state.ctx != null or entry.qsa_pooled.ctx != null) and entry.conv_state.ctx == null) {
        const hist = qsaHistoryRows(entry);
        const keep = hist - t + 1 + acc;
        if (!try qsaResliceToKeep(entry, keep, s)) {
            if (entry.aux_state.ctx != null) {
                const held = mlx.getShape(entry.aux_state)[1];
                // A slice past the ring's own start would clamp to empty, so the rollback is refused by name.
                if (hist - keep > held) return error.QsaRollbackBeyondRing;
                const view = try rows3(s, entry.aux_state, 0, held - (hist - keep));
                defer _ = mlx.mlx_array_free(view);
                const owned = try materializedOwnedCopy(s, view);
                const pooled = entry.qsa_pooled;
                entry.qsa_pooled = none;
                ssmFreeQsaState(entry);
                entry.qsa_pooled = pooled;
                entry.aux_state = owned;
                entry.qsa_key_rows = held - (hist - keep);
            }
            entry.qsa_hist_rows = keep;
            try truncatePooled(&entry.qsa_pooled, keep, entry.qsa_ratio, s, true);
        }
    }
    if (entry.spec_state_seq.ctx != null) {
        // Row `accepted` of the verify's per-row states `[T, B, Hv, Dv, Dk]`.
        const sq = mlx.getShape(entry.spec_state_seq);
        const shape = [_]c_int{ sq[1], sq[2], sq[3], sq[4] };
        const row = try ssmAxisView(s, entry.spec_state_seq, 0, accepted);
        defer _ = mlx.mlx_array_free(row);
        var state = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(state);
        try mlx.check(mlx.mlx_reshape(&state, row, &shape, 4, s));
        replace(&entry.ssm_state, state);
        patch.ssm = true;
    } else if (entry.spec_replay) {
        // A folded verify kept its input state and its rows' recurrence inputs instead: rows 0..accepted replayed.
        const kp = mlx.getShape(entry.spec_prework.k); // [T, Hk, Dk]
        if (accepted + 1 > @as(u32, @intCast(kp[0]))) return error.GdnReplayRange;
        const sh = mlx.getShape(entry.spec_state_in); // [B, Hv, Dv, Dk]
        var path: [gdn_decode.FOLD_MAX_SEQ]i32 = undefined;
        for (path[0 .. accepted + 1], 0..) |*p, i| p.* = @intCast(i);
        const a: i32 = @intCast(accepted);
        const c = (try gdn_decode.replay(.{ .hk = kp[1], .hv = sh[1], .dk = sh[3], .dv = sh[2] }, entry.spec_state_in, entry.spec_prework, path[0 .. accepted + 1], entry.spec_conv_input, .{ 1 + a, 2 + a, 3 + a }, s)) orelse return error.GdnReplayUnavailable;
        _ = mlx.mlx_array_free(c.conv_state);
        replace(&entry.ssm_state, c.ssm_state);
        patch.ssm = true;
    }
    if (entry.spec_conv_input.ctx != null) {
        // The conv window after row `accepted` of the verify's conv input `[B, (kernel-1) + T, C]`.
        const ci = mlx.getShape(entry.spec_conv_input);
        replace(&entry.conv_state, try rows3(s, entry.spec_conv_input, 1 + acc, 1 + acc + ci[1] - t));
        patch.conv = true;
    }
    try ssmPatchGroupRow(s, entry, patch);
}

// ── The batched tick's group ──

/// The batched tick's owner of the merged GDN/conv/PLE state across ticks: slots hold (group, row, member_gen), and
/// any write outside the group bumps the slot's `member_gen`, which `ssmGroupMatches` then refuses.
pub const PersistentSsmGroup = @import("../mlx_serve/engines/mlx/ssm/group.zig").PersistentSsmGroup;
pub const SsmMergeKind = @import("../mlx_serve/engines/mlx/ssm/group.zig").SsmMergeKind;

var ssm_member_gen_src: std.atomic.Value(u64) = .init(1);

pub fn nextSsmMemberGen() u64 {
    return ssm_member_gen_src.fetchAdd(1, .monotonic);
}

const SsmGroupMember = struct { key: usize, gen: u64, entries: []SSMCacheEntry };
const SsmPatchFields = struct { conv: bool = false, ssm: bool = false, aux: bool = false };

/// Whether the live group holds exactly these slots, in order, at their current generations (`.matches`).
pub fn ssmGroupMatches(self: *const PersistentSsmGroup, ctxs: []const *ForwardCtx) bool {
    if (!self.live or self.n != ctxs.len) return false;
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries orelse return false;
        if (self.keys[i] != @intFromPtr(entries.ptr) or entries.len == 0 or self.gens[i] != entries[0].member_gen) return false;
    }
    return true;
}

/// Every member unbound and the merged state freed (`.clear`, and `.deinit`).
pub fn ssmGroupClear(self: *PersistentSsmGroup) void {
    if (!self.live) return;
    ssmUnbindAll(self);
    for (self.layers) |*m| ssmFreeMergedLayer(m);
    inline for (.{ "layers", "keys", "gens", "member_entries", "kinds" }) |f| {
        self.allocator.free(@field(self, f));
        @field(self, f) = &.{};
    }
    self.n = 0;
    self.live = false;
}

fn ssmFreeMergedLayer(m: *SSMCacheEntry) void {
    inline for (.{ &m.conv_state, &m.ssm_state, &m.aux_state }) |a| replace(a, none);
    ssmFreeSpecCapture(m);
    m.initialized = false;
}

fn ssmBindRow(group: *PersistentSsmGroup, entries: []SSMCacheEntry, row: usize) void {
    for (entries, 0..) |*e, l| {
        e.group = group;
        e.group_row = @intCast(row);
        e.group_layer = @intCast(l);
    }
}

fn ssmUnbindEntries(group: *PersistentSsmGroup, entries: []SSMCacheEntry) void {
    for (entries) |*e| {
        if (e.group == group) e.group = null;
    }
}

fn ssmUnbindAll(group: *PersistentSsmGroup) void {
    if (!group.live) return;
    for (group.member_entries) |entries| ssmUnbindEntries(group, entries);
}

/// Row `row` of a merged state copied into `dst` as an array of its own, queued for the export's eval.
fn ssmExportState(s: mlx.mlx_stream, merged: mlx.mlx_array, row: usize, dst: *mlx.mlx_array, vec: mlx.mlx_vector_array) !void {
    const view = try ssmAxisView(s, merged, 0, row);
    defer _ = mlx.mlx_array_free(view);
    replace(dst, try materializedOwnedCopy(s, view));
    _ = mlx.mlx_vector_array_append_value(vec, dst.*);
}

/// A row handed back to its slot as evaluated arrays of its own, the slot's `member_gen` bumped so no later repack
/// can re-adopt it.
fn ssmExportRow(s: mlx.mlx_stream, group: *PersistentSsmGroup, row: usize) !void {
    const entries = group.member_entries[row];
    if (entries.len > 0 and entries[0].group == null) return;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (group.layers, group.kinds, 0..) |*m, kind, l| {
        const e = &entries[l];
        if (kind.linear and m.conv_state.ctx != null) {
            try ssmExportState(s, m.conv_state, row, &e.conv_state, vec);
            try ssmExportState(s, m.ssm_state, row, &e.ssm_state, vec);
            e.initialized = true;
        }
        if (kind.ple and m.aux_state.ctx != null) try ssmExportState(s, m.aux_state, row, &e.aux_state, vec);
        e.group = null;
    }
    const gen = nextSsmMemberGen();
    for (entries) |*e| e.member_gen = gen;
    if (mlx.mlx_vector_array_size(vec) > 0) _ = mlx.mlx_eval(vec);
}

fn ssmFindOldRow(group: *const PersistentSsmGroup, key: usize, gen: u64) ?usize {
    for (group.keys, group.gens, 0..) |k, g, i| {
        if (k == key and g == gen) return i;
    }
    return null;
}

fn ssmRowOf(group: *const PersistentSsmGroup, key: usize) ?usize {
    for (group.keys, 0..) |k, i| {
        if (k == key) return i;
    }
    return null;
}

/// The members but row `r`.
fn ssmSurvivors(group: *const PersistentSsmGroup, r: usize, buf: []SsmGroupMember) []SsmGroupMember {
    var n: usize = 0;
    for (0..group.n) |i| {
        if (i == r) continue;
        buf[n] = .{ .key = group.keys[i], .gen = group.gens[i], .entries = group.member_entries[i] };
        n += 1;
    }
    return buf[0..n];
}

fn ssmMembersFromCtxs(ctxs: []const *ForwardCtx, buf: []SsmGroupMember) ![]SsmGroupMember {
    if (ctxs.len > buf.len) return error.SsmGroupTooWide;
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries.?;
        const gen = if (entries.len > 0) entries[0].member_gen else c.ssm_member_gen;
        buf[i] = .{ .key = @intFromPtr(entries.ptr), .gen = gen, .entries = entries };
    }
    return buf[0..ctxs.len];
}

/// Row `row` of the merged `parent` replaced by `update`.
fn ssmSliceUpdateRow(s: mlx.mlx_stream, parent: *mlx.mlx_array, row: usize, update: mlx.mlx_array) !void {
    const sh = mlx.getShape(parent.*);
    const b = try indexBounds(sh, 0, row);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice_update(&out, parent.*, update, &b[0], sh.len, &b[1], sh.len, &ones, sh.len, s));
    replace(parent, out);
}

/// Layer `l`'s merged `field` for `members`: a survivor's row of the current generation, a joiner's own state.
fn ssmRepackState(s: mlx.mlx_stream, group: *const PersistentSsmGroup, members: []const SsmGroupMember, l: usize, comptime field: []const u8) !mlx.mlx_array {
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var views: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n_views: usize = 0;
    defer for (views[0..n_views]) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (members, 0..) |mem, j| {
        parts[j] = if (ssmFindOldRow(group, mem.key, mem.gen)) |old| blk: {
            views[n_views] = try ssmAxisView(s, @field(group.layers[l], field), 0, old);
            n_views += 1;
            break :blk views[n_views - 1];
        } else @field(mem.entries[l], field);
    }
    return concatAxis0(s, parts[0..members.len]);
}

/// The group's next generation for `members` (a first one when the group is not live): a survivor keeps its row, a
/// joiner brings its own state, a leaver is exported first. Nothing old is freed until the new generation is built.
fn ssmRepackMembers(alloc: std.mem.Allocator, s: mlx.mlx_stream, group: *PersistentSsmGroup, members: []const SsmGroupMember, kinds: []const SsmMergeKind) !void {
    for (0..group.n) |i| {
        const stays = for (members) |m| {
            if (ssmFindOldRow(group, m.key, m.gen) == i) break true;
        } else false;
        if (!stays) try ssmExportRow(s, group, i);
    }
    const kinds_local = try alloc.dupe(SsmMergeKind, kinds);
    errdefer alloc.free(kinds_local);
    const layers = try alloc.alloc(SSMCacheEntry, kinds.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*m| ssmFreeMergedLayer(m);
        alloc.free(layers);
    }
    for (layers, kinds, 0..) |*m, kind, l| {
        m.* = .{ .conv_state = none, .ssm_state = none, .initialized = kind.linear };
        built += 1;
        if (kind.linear) {
            m.conv_state = try ssmRepackState(s, group, members, l, "conv_state");
            m.ssm_state = try ssmRepackState(s, group, members, l, "ssm_state");
        }
        if (kind.ple) m.aux_state = try ssmRepackState(s, group, members, l, "aux_state");
    }
    const n = members.len;
    const keys = try alloc.alloc(usize, n);
    errdefer alloc.free(keys);
    const gens = try alloc.alloc(u64, n);
    errdefer alloc.free(gens);
    const member_entries = try alloc.alloc([]SSMCacheEntry, n);
    for (members, keys, gens, member_entries) |mem, *k, *g, *e| {
        k.* = mem.key;
        g.* = mem.gen;
        e.* = mem.entries;
    }
    const old = group.*;
    ssmUnbindAll(group);
    group.* = .{ .allocator = old.allocator, .n = n, .keys = keys, .gens = gens, .member_entries = member_entries, .layers = layers, .kinds = kinds_local, .live = true };
    for (old.layers) |*m| ssmFreeMergedLayer(m);
    alloc.free(old.layers);
    alloc.free(old.keys);
    alloc.free(old.gens);
    alloc.free(old.member_entries);
    alloc.free(old.kinds);
    for (members, 0..) |mem, row| {
        ssmBindRow(group, mem.entries, row);
        for (mem.entries) |*e| e.member_gen = mem.gen;
    }
}

pub fn ssmGroupDropMember(group: *PersistentSsmGroup, s: mlx.mlx_stream, entries: []SSMCacheEntry) !void {
    if (!group.live) return;
    const r = ssmRowOf(group, @intFromPtr(entries.ptr)) orelse return;
    ssmUnbindEntries(group, entries);
    if (group.n == 1) return ssmGroupClear(group);
    var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    ssmRepackMembers(group.allocator, s, group, ssmSurvivors(group, r, &buf), group.kinds) catch |err| {
        // Every survivor gets its own state back before the group goes, so none reverts to a stale entry.
        var export_err: ?anyerror = null;
        for (0..group.n) |i| {
            if (i == r) continue;
            ssmExportRow(s, group, i) catch |e| {
                export_err = e;
            };
        }
        ssmGroupClear(group);
        return export_err orelse err;
    };
}

/// A member's row back to its own entries, at the entry of every forward outside the group.
pub fn ssmGroupReleaseCtx(group: *PersistentSsmGroup, s: mlx.mlx_stream, ctx: *ForwardCtx) !void {
    const entries = ctx.ssm_entries orelse return;
    if (!group.live) return;
    const r = ssmRowOf(group, @intFromPtr(entries.ptr)) orelse return;
    if (group.n == 1) {
        try ssmExportRow(s, group, r);
        ssmGroupClear(group);
    } else {
        var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
        try ssmRepackMembers(group.allocator, s, group, ssmSurvivors(group, r, &buf), group.kinds);
    }
    if (entries.len > 0) ctx.ssm_member_gen = entries[0].member_gen;
}

/// The tick's merged entries, the group's own: a stable membership takes them as they are, any other repacks once.
pub fn ssmBindTickState(alloc: std.mem.Allocator, s: mlx.mlx_stream, group: *PersistentSsmGroup, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) ![]SSMCacheEntry {
    if (!ssmGroupMatches(group, ctxs) or group.kinds.len != kinds.len) {
        var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
        group.allocator = alloc;
        try ssmRepackMembers(alloc, s, group, try ssmMembersFromCtxs(ctxs, &buf), kinds);
    }
    return group.layers;
}

/// A grouped entry's rewritten states copied into its row of the group's merged layer.
fn ssmPatchGroupRow(s: mlx.mlx_stream, e: *const SSMCacheEntry, fields: SsmPatchFields) !void {
    const g = e.group orelse return;
    if (!g.live or e.group_layer >= g.layers.len) return;
    const m = &g.layers[e.group_layer];
    if (fields.conv and e.conv_state.ctx != null and m.conv_state.ctx != null) try ssmSliceUpdateRow(s, &m.conv_state, e.group_row, e.conv_state);
    if (fields.ssm and e.ssm_state.ctx != null and m.ssm_state.ctx != null) try ssmSliceUpdateRow(s, &m.ssm_state, e.group_row, e.ssm_state);
    if (fields.aux and e.aux_state.ctx != null and m.aux_state.ctx != null) try ssmSliceUpdateRow(s, &m.aux_state, e.group_row, e.aux_state);
}

/// `arr` copied into an array of its own: the entry's row of it when the entry sits in a live group.
fn ssmMaterializeLive(s: mlx.mlx_stream, src: *const SSMCacheEntry, arr: mlx.mlx_array) !mlx.mlx_array {
    if (src.group) |g| {
        if (g.live) {
            const view = try ssmAxisView(s, arr, 0, src.group_row);
            defer _ = mlx.mlx_array_free(view);
            return materializedOwnedCopy(s, view);
        }
    }
    return materializedOwnedCopy(s, arr);
}

// ── The prefix cache's checkpoints ──

/// Every layer's recurrent state after the first `pos` tokens of a prompt (one snapshot per entry), restored with the
/// KV at `pos` on a warm turn: the copy's type, which the QSA history's attach takes.
pub const SSMCheckpoint = @import("../mlx_serve/engines/mlx/ssm/checkpoint.zig").SSMCheckpoint;

/// Every entry's state copied into arrays of its own and evaluated: the live states are often slices of a prefill
/// chunk's much larger arrays, which a shared handle or a lazy copy would keep alive inside the prefix cache.
pub fn captureSsmCheckpoint(allocator: std.mem.Allocator, ssm_entries: []const SSMCacheEntry, pos: usize, s: mlx.mlx_stream) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, ssm_entries.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (ssm_entries, layers) |*src, *out| {
        out.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = src.initialized, .qsa_ratio = src.qsa_ratio, .ple_prev = src.ple_prev, .ple_prev_valid = src.ple_prev_valid };
        built += 1;
        // A grouped entry's live state is its row of the group's merged layer.
        const m: ?*const SSMCacheEntry = if (src.group) |g| (if (g.live and src.group_layer < g.layers.len) &g.layers[src.group_layer] else null) else null;
        const conv = if (m) |x| x.conv_state else src.conv_state;
        const ssm = if (m) |x| x.ssm_state else src.ssm_state;
        if (conv.ctx != null) replace(&out.conv_state, try ssmMaterializeLive(s, src, conv));
        if (ssm.ctx != null) replace(&out.ssm_state, try ssmMaterializeLive(s, src, ssm));
        // The PLE window is per-position state, in every checkpoint. The QSA history is a prefix of one buffer: each
        // checkpoint keeps only its leftover rows, and the history goes on the latest (`attachQsaHistoryToLatest`).
        if (m != null and m.?.aux_state.ctx != null) {
            out.aux_state = try ssmMaterializeLive(s, src, m.?.aux_state);
        } else if (ssmAuxIsQsaHistory(src)) {
            out.qsa_rows = @intCast(pos);
            if (src.aux_state.ctx != null) out.aux_state = try qsaLeftoverAt(src.aux_state, qsaHistoryRows(src), @intCast(pos), src.qsa_ratio, s);
        } else if (src.aux_state.ctx != null) {
            out.aux_state = try ssmMaterializeLive(s, src, src.aux_state);
        }
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (layers) |*l| {
        inline for (snapshot_arrays) |f| {
            if (@field(l, f).ctx != null) _ = mlx.mlx_vector_array_append_value(vec, @field(l, f));
        }
    }
    if (mlx.mlx_vector_array_size(vec) > 0) _ = mlx.mlx_eval(vec);
    return .{ .pos = pos, .layers = layers };
}

/// A second owner of `cp`'s state, every buffer refcount-shared: a checkpoint is already materialized, so sharing it
/// pins nothing transient.
pub fn shareSsmCheckpoint(allocator: std.mem.Allocator, cp: *const SSMCheckpoint) !SSMCheckpoint {
    const layers = try allocator.alloc(SSMCacheEntrySnapshot, cp.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    for (cp.layers, layers) |*src, *out| {
        out.* = src.*;
        inline for (snapshot_arrays) |f| @field(out, f) = none;
        built += 1;
        inline for (snapshot_arrays) |f| @field(out, f) = try ref(@field(src, f));
    }
    return .{ .pos = cp.pos, .layers = layers };
}

pub fn restoreSsmCheckpoint(ssm_entries: []SSMCacheEntry, cp: *const SSMCheckpoint) !void {
    if (ssm_entries.len != cp.layers.len) return error.SsmCheckpointLayerMismatch;
    for (ssm_entries, cp.layers) |*dst, *src| try ssmRestore(dst, src);
}

/// The bytes a checkpoint holds, for the prefix cache's budget.
pub fn ssmCheckpointBytes(cp: *const SSMCheckpoint) u64 {
    var total: u64 = 0;
    for (cp.layers) |l| {
        inline for (snapshot_arrays) |f| {
            const a = @field(l, f);
            if (a.ctx != null) total += @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
        }
    }
    return total;
}
