//! copied_from: src/plugins/qwen4_exp/ple.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The n-gram PLE layer: the host gather or GPU kernel into a deferred leaf (`ple_defer`, `flushDeferredPle`), the
//! gated value injection and its dilated causal conv, the per-slot history, and the spec capture and rollback.

const std = @import("std");
const qwen4_mod = @import("qwen4_exp.zig");
const ple_gpu = @import("ple_gpu.zig");
const mlx = @import("../../mlx/ffi.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const io_util_mod = @import("../../sdk/util/io_util.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const layered = @import("../../engines/mlx/batched/layered.zig");
const prof = @import("prof.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const L10VerifySlots = @import("../../engines/mlx/batched/row_axis_verify.zig").L10VerifySlots;
const ProfClock = @import("../../engines/mlx/diag/profile.zig").ProfClock;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const scalarOf = @import("../../engines/mlx/load/weights.zig").scalarOf;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("../../engines/mlx/ssm/entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const Trunk = @import("trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const Qwen4TestSlot = @import("forward.zig").Qwen4TestSlot;
const Qwen4Trace = @import("mtp.zig").Qwen4Trace;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const qsaArraysAllEqual = @import("../../sdk/testing/parity.zig").qsaArraysAllEqual;
const PleWeights = @import("weights.zig").PleWeights;
const kernelSource = @import("../../mlx/kernels.zig").source;
const cachedScalarInt = @import("../../sdk/blocks/linear/verify_qmm.zig").cachedScalarInt;
const sigmoidTableFor = @import("../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const ShapeKey = @import("../../sdk/blocks/moe/router_kernel.zig").ShapeKey;

pub var verify_fault_after_ple_row: ?usize = null;

pub fn claimPleSpecCapture(entry: *SSMCacheEntry, n: usize, ctx_len: usize, capturing: bool) bool {
    if (capturing and ctx_len + n <= entry.spec_ple_tokens.len) {
        entry.spec_ple_len = @intCast(ctx_len + n);
        return true;
    }
    entry.spec_ple_len = 0;
    return false;
}

/// A partial accept's n-gram history: the `ngram_size - 1` verify tokens ending at `t1` plus
/// `accepted` drafts, read off the capture's `prev ++ ids`.
pub fn plePrevAfterAccept(entry: *SSMCacheEntry, accepted: u32, verify_len: u32) void {
    const ctx_len: usize = @as(usize, entry.spec_ple_len) - @as(usize, verify_len);
    for (0..ctx_len) |i| entry.ple_prev[i] = entry.spec_ple_tokens[1 + accepted + i];
    entry.ple_prev_valid = true;
}

pub fn fillPleSpecTokens(entry: *SSMCacheEntry, prev: []const u32, ids: []const u32) void {
    const ctx_len = prev.len;
    for (0..ctx_len) |i| entry.spec_ple_tokens[i] = prev[i];
    for (ids, 0..) |id, j| entry.spec_ple_tokens[ctx_len + j] = id;
}

pub const PlePending = struct {
    emb: mlx.mlx_array,
    token_ids: mlx.mlx_array,
    entry: *SSMCacheEntry,
    layer: usize,
    seq_len: c_int,
    /// Whether the gather owes `entry.spec_ple_tokens` the verify history.
    /// Decided (and `spec_ple_len` claimed) at BUILD time: the PLE layer's own
    /// conv capture, later in the same forward, keys on `spec_ple_len` while
    /// this gather has not run yet, and `spec_capture_ssm` is already back to
    /// false by the time the flush happens.
    capture: bool,
    /// GPU arm: `emb` is the kernel's output, not a leaf; only the history is owed.
    gpu: bool = false,
};

/// Advance a PLE entry's (ngram_size-1)-token history past `ids`.
pub fn advancePlePrev(entry: *SSMCacheEntry, prev: [8]u32, ids: []const u32, ctx_len: usize) void {
    const n = ids.len;
    const tail_from: usize = if (n >= ctx_len) n - ctx_len else 0;
    var k: usize = 0;
    if (n < ctx_len) {
        // Shift the old history left and append everything.
        for (0..ctx_len - n) |i| entry.ple_prev[i] = prev[n + i];
        k = ctx_len - n;
    }
    for (tail_from..n) |i| {
        entry.ple_prev[k] = ids[i];
        k += 1;
    }
    entry.ple_prev_valid = true;
}

/// Test meter: device reads of a batched tick's deferred PLE ids (one per group flush).
pub var deferred_ple_group_syncs: u64 = 0;

/// A deferred PLE leaf is filled after the build, so a build that reads its
/// routing on the host (the EXL3 prefill GEMM, past `DECODE_ROWS_MAX` rows)
/// would evaluate it empty: those widths gather eagerly.
pub fn pleDeferrable(cfg: *const ModelConfig, rows: usize) bool {
    return cfg.exl3 == null or rows <= exl3.kernels.DECODE_ROWS_MAX;
}

/// Host-side n-gram gather: `[B, S, ple_embed_dim]` bf16 for this chunk's
/// token ids, advancing the token history. Serial: `entry`'s history over
/// `[1, S]`. Batched (`ctx.batch_slots`): `[N, 1]`, each row hashed
/// against ITS slot's history (`slots[i].ssm_entries[layer]`), ONE gather
/// for all N·heads rows.
pub fn pleEmbedding(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int) !mlx.mlx_array {
    const st = self.qwen4.?;
    // A pipelined serial step on the GPU arm leaves its history settle to the next forward:
    // by now those ids are evaluated, so the read waits on nothing.
    if (ctx.ple_pending) |p| if (p.gpu) try self.flushDeferredPle(ctx);
    const emb_dim: usize = st.table.dim * st.hash.n_heads;
    const n: usize = mlx.mlx_array_size(token_ids);
    const batch: c_int = @intCast(n / @as(usize, @intCast(seq_len)));
    const shape = [_]c_int{ batch, seq_len, @intCast(emb_dim) };
    const capture = blk: {
        if (ctx.batch_slots) |slots| {
            if (!self.spec_capture_ssm) break :blk false;
            const per: usize = @intCast(seq_len);
            var any = false;
            for (slots) |sc| {
                if (self.pleClaimSpecCapture(&sc.ssm_entries.?[layer], per)) any = true;
            }
            if (any) _ = self.pleClaimSpecCapture(entry, per);
            break :blk any;
        }
        break :blk self.pleClaimSpecCapture(entry, n);
    };
    if (ctx.batch_slots == null and st.gpu != null and mlx.streamIsGpu(self.s)) {
        return self.pleEmbeddingGpu(ctx, st.gpu.?, token_ids, entry, layer, seq_len, capture, &shape);
    }
    // Packed bf16 on the host (RNE) so the upload is one copy — no
    // mid-graph eval, no GPU sync inside the layer loop.
    const pk = try self.allocator.alloc(u16, n * emb_dim);
    defer self.allocator.free(pk);
    if (ctx.ple_defer and pleDeferrable(&self.config, n)) {
        if (ctx.ple_pending != null) return error.PlePendingAlreadySet;
        @memset(pk, 0);
        const emb = mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
        errdefer _ = mlx.mlx_array_free(emb);
        var emb_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(emb_ref);
        try mlx.check(mlx.mlx_array_set(&emb_ref, emb));
        var ids_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ids_ref);
        try mlx.check(mlx.mlx_array_set(&ids_ref, token_ids));
        ctx.ple_pending = .{ .emb = emb_ref, .token_ids = ids_ref, .entry = entry, .layer = layer, .seq_len = seq_len, .capture = capture };
        return emb;
    }
    try self.pleGatherBf16(ctx, token_ids, entry, layer, seq_len, pk, capture);
    return mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
}

/// GPU arm of a serial forward: the kernel reads the ids itself, lazy or not, so the build
/// needs no host read. The history moves now (eager) or when the caller flushes or settles
/// the pending record (`ple_defer`), before anything reads the history again.
pub fn pleEmbeddingGpu(self: *Trunk, ctx: *ForwardCtx, tbl: ple_gpu.Table, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int, capture: bool, shape: []const c_int) !mlx.mlx_array {
    const st = self.qwen4.?;
    if (ctx.ple_defer and ctx.ple_pending != null) return error.PlePendingAlreadySet;
    const prev = plePrevOf(entry, st.hash.eos);
    const flat = try ple_gpu.embed(self.s, tbl, &st.hash, &st.table, token_ids, prev[0 .. st.hash.ngram_size - 1]);
    defer _ = mlx.mlx_array_free(flat);
    var emb = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_reshape(&emb, flat, shape.ptr, @intCast(shape.len), self.s));
    if (!ctx.ple_defer) {
        try self.pleSettleFromArray(token_ids, entry, capture);
        return emb;
    }
    var ids_ref = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids_ref);
    try mlx.check(mlx.mlx_array_set(&ids_ref, token_ids));
    ctx.ple_pending = .{ .emb = .{ .ctx = null }, .token_ids = ids_ref, .entry = entry, .layer = layer, .seq_len = seq_len, .capture = capture, .gpu = true };
    return emb;
}

/// Claim (or clear) `entry`'s fixed spec-PLE token slot for a gather of
/// `n` ids, reporting whether the verify history fits it. ONE predicate
/// for the eager and the deferred arm: with `ple_defer` the gather that
/// would set `spec_ple_len` runs AFTER the PLE layer's conv capture, which
/// keys on it — so the length is claimed here, at build time, and the
/// gather only fills the tokens.
pub fn pleClaimSpecCapture(self: *Trunk, entry: *SSMCacheEntry, n: usize) bool {
    const ctx_len: usize = self.qwen4.?.hash.ngram_size - 1;
    return claimPleSpecCapture(entry, n, ctx_len, self.spec_capture_ssm);
}

/// Fill the leaf a `ple_defer` forward was built on: the token ids are
/// read NOW (this is the one host sync of a pipelined qwen4 step), the
/// rows gathered straight into the leaf's buffer, and the per-entry
/// n-gram history advanced. No-op without a pending leaf.
pub fn flushDeferredPle(self: *Trunk, ctx: *ForwardCtx) !void {
    const p = ctx.ple_pending orelse return;
    if (p.gpu) {
        try self.pleSettleFromArray(p.token_ids, p.entry, p.capture);
    } else {
        const dst = mlx.mlx_array_data_bfloat16(p.emb) orelse return error.PleLeafUnreadable;
        const out: [*]u16 = @constCast(dst);
        const pk = out[0..mlx.mlx_array_size(p.emb)];
        // a solo verify's ids off the arrays its input was built from, when the caller lent them
        const from_src = if (ctx.ple_ids_src) |src| (if (self.config.trunk_format) try pleFillFromSources(self, ctx, src, p.token_ids, p.entry, pk, p.capture) else false) else false;
        if (!from_src) try self.pleGatherBf16(ctx, p.token_ids, p.entry, p.layer, p.seq_len, pk, p.capture);
    }
    ctx.ple_pending = null;
    releasePlePending(p);
}

/// The deferred leaf filled from `src` (`ForwardCtx.ple_ids_src`), the arrays the verify input concatenates, instead of
/// an op over that input (`pleEvalIds`' `contiguous` queues behind everything already handed to the GPU, under the
/// verify ladder layer 0): each id is read after its OWN producer's event, and while the next one is still running the
/// rows of the ids already held are gathered. The leaf gets byte for byte what `pleGatherBf16` writes (a token's rows
/// depend only on the history and the tokens up to it, and a token-aligned range keeps each row's head), and the
/// history moves are the same call on the same ids. False, nothing read or moved, where `src` cannot stand in for
/// `token_ids`.
fn pleFillFromSources(self: *Trunk, ctx: *ForwardCtx, src: []const mlx.mlx_array, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, pk: []u16, capture: bool) !bool {
    const st = self.qwen4 orelse return false;
    const n: usize = src.len;
    const nh: usize = st.hash.n_heads;
    const emb_dim: usize = st.table.dim * nh;
    if (!pleSourcesFit(n, mlx.mlx_array_size(token_ids), pk.len, emb_dim, ctx.batch_slots == null)) return pleSourcesDeclined("the lent arrays are not this solo verify's ids, one per row");
    if (!pleRangesKeepHeads(nh, st.table.n_heads)) return pleSourcesDeclined("the table's heads do not divide the hash's, so a token range would shift a row's head");
    for (src) |a| if (!pleSourceUsable(mlx.mlx_array_size(a), mlx.mlx_array_dtype(a))) return pleSourcesDeclined("a lent array is not one int32 or uint32 id");
    // anything not yet handed to the GPU goes now, in one eval: reading unscheduled ids one by one would sync per id
    {
        const v = mlx.mlx_vector_array_new_data(src.ptr, src.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }
    const ids = try self.allocator.alloc(u32, n);
    defer self.allocator.free(ids);
    const rows = try self.allocator.alloc(i64, n * nh);
    defer self.allocator.free(rows);
    const host = try self.allocator.alloc(f32, n * emb_dim);
    defer self.allocator.free(host);
    const ctx_len: usize = st.hash.ngram_size - 1;
    // the history `pleAdvanceSerial` hashes against: nothing touches the entry until that call below
    const prev = plePrevOf(entry, st.hash.eos);
    const kv_len: u64 = @intCast(ctx.moe_seq_offset.*);
    var sched: PleSrcSchedule = .{};
    var waited: u64 = 0;
    const io = std.Io.Threaded.global_single_threaded.io();
    for (src, 0..) |a, i| {
        var ready = false;
        _ = mlx._mlx_array_is_available(&ready, a);
        if (sched.beforeRead(ready)) |r| pleGatherRange(st, prev[0..ctx_len], ids, r, rows, host, pk, kv_len);
        var wait_lap = io_util_mod.Stopwatch.init(io);
        ids[i] = try pleSourceId(a);
        waited += wait_lap.read();
        sched.afterRead();
    }
    self.verify_laps.ple_sync_ns = waited;
    if (sched.finish()) |r| pleGatherRange(st, prev[0..ctx_len], ids, r, rows, host, pk, kv_len);
    const moved = pleAdvanceSerial(st, entry, ids, capture);
    std.debug.assert(std.mem.eql(u32, moved[0..ctx_len], prev[0..ctx_len]));
    if (!ple_src_said) {
        ple_src_said = true;
        log.info("[qwen4] PLE id lend engaged: verify PLE ids read off the draft arrays, each after its own event ({d} ids)\n", .{n});
    }
    return true;
}

var ple_src_said = false;
var ple_src_declined_said = false;

/// The id lend's first decline, said once: the flush reads the verify input's ids itself. False, nothing read or moved.
fn pleSourcesDeclined(why: []const u8) bool {
    if (!ple_src_declined_said) {
        ple_src_declined_said = true;
        log.info("[qwen4] PLE id lend declined: {s}; the flush reads the verify input's ids\n", .{why});
    }
    return false;
}

/// Whether `n` source ids can stand in for a deferred leaf: one per token id, the leaf `n` rows of `emb_dim`, and a
/// solo forward (a batched one hashes per slot).
pub fn pleSourcesFit(n: usize, n_ids: usize, leaf_len: usize, emb_dim: usize, solo: bool) bool {
    return solo and n > 0 and n == n_ids and leaf_len == n * emb_dim;
}

/// A trellis gather decodes row `i` of ITS call at head `i % table_heads`; a range that starts at a token boundary (a
/// multiple of the hash's `nh` rows a token) keeps every row's head only when `table_heads` divides `nh`.
pub fn pleRangesKeepHeads(nh: usize, table_heads: usize) bool {
    return table_heads <= 1 or (nh > 0 and nh % table_heads == 0);
}

/// One source id array: a single integer element (the head's `[1]`, a sampler's `[1, 1]`, int32 or uint32), what the
/// verify input concatenates.
pub fn pleSourceUsable(size: usize, dtype: mlx.mlx_dtype) bool {
    return size == 1 and (dtype == .int32 or dtype == .uint32);
}

/// One id off its source after the source's own event (`mlx_array_eval` on an array already handed to the GPU only
/// waits for it). A token id is non-negative and below 2^31, so its int32 and uint32 readings agree with the input's
/// `astype(int32)`.
pub fn pleSourceId(a: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const v: i64 = switch (mlx.mlx_array_dtype(a)) {
        .int32 => (mlx.mlx_array_data_int32(a) orelse return error.TokenIdsUnreadable)[0],
        .uint32 => (mlx.mlx_array_data_uint32(a) orelse return error.TokenIdsUnreadable)[0],
        else => return error.PleSourceDtype,
    };
    if (v < 0 or v > std.math.maxInt(i32)) return error.PleSourceRange;
    return @intCast(v);
}

/// The rows of the held ids `ids[r[0]..r[1]]`, hashed over the history and every id up to `r[1]`, gathered and
/// bf16-packed into their slice of the leaf: the rows, decode and rounding `pleFillFromIds` gives the whole range.
fn pleGatherRange(st: *const qwen4_mod.Qwen4State, prev: []const u32, ids: []const u32, r: [2]usize, rows: []i64, host: []f32, pk: []u16, kv_len: u64) void {
    const lo = r[0];
    const hi = r[1];
    if (lo >= hi) return;
    const nh: usize = st.hash.n_heads;
    const emb_dim: usize = st.table.dim * nh;
    st.hash.rowIds(prev, ids[0..hi], rows[0 .. hi * nh]);
    st.table.gather(rows[lo * nh .. hi * nh], host[lo * emb_dim .. hi * emb_dim], kv_len);
    for (host[lo * emb_dim .. hi * emb_dim], pk[lo * emb_dim .. hi * emb_dim]) |v, *o| o.* = qwen4_mod.bf16Rne(v);
}

/// When `pleFillFromSources` gathers: the ids arrive in order, and before WAITING for the next one the rows of every id
/// held but not gathered are gathered, so the host work runs while the GPU still produces the later ids; the rest goes
/// after the last id. Every id is gathered once, in order, in contiguous ranges; all readable at once is ONE range.
pub const PleSrcSchedule = struct {
    /// Ids read so far.
    held: usize = 0,
    /// Ids whose rows are in the leaf.
    gathered: usize = 0,

    /// Before reading the next id (`ready`: readable without waiting): the token range to gather now, if any.
    pub fn beforeRead(self: *PleSrcSchedule, ready: bool) ?[2]usize {
        if (ready or self.gathered == self.held) return null;
        return self.take();
    }

    pub fn afterRead(self: *PleSrcSchedule) void {
        self.held += 1;
    }

    /// After the last id: the range still owed, if any.
    pub fn finish(self: *PleSrcSchedule) ?[2]usize {
        if (self.gathered == self.held) return null;
        return self.take();
    }

    fn take(self: *PleSrcSchedule) [2]usize {
        const r = [2]usize{ self.gathered, self.held };
        self.gathered = self.held;
        return r;
    }
};

/// Fill every pending deferred-PLE leaf of a group in ONE pass. The caller retains
/// rollback state until the whole round succeeds.
pub fn flushDeferredPleGroup(self: *Trunk, ctxs: []const *ForwardCtx) !void {
    var ids: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (ids[0..n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (ctxs.len > ids.len) return error.PleGroupTooWide;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (ctxs) |ctx| {
        const p = ctx.ple_pending orelse continue;
        ids[n] = try self.plePrepareIds(p.token_ids);
        _ = mlx.mlx_vector_array_append_value(vec, ids[n]);
        n += 1;
    }
    if (n == 0) return;
    var sync_lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    try mlx.check(mlx.mlx_eval(vec));
    self.verify_laps.ple_sync_ns = sync_lap.read();
    deferred_ple_group_syncs += 1;
    var k: usize = 0;
    for (ctxs) |ctx| {
        const p = ctx.ple_pending orelse continue;
        if (p.gpu) {
            const host = try self.pleHostIds(ids[k]);
            defer self.allocator.free(host);
            _ = pleAdvanceSerial(self.qwen4.?, p.entry, host, p.capture);
        } else {
            const dst = mlx.mlx_array_data_bfloat16(p.emb) orelse return error.PleLeafUnreadable;
            const out: [*]u16 = @constCast(dst);
            try self.pleFillFromIds(ctx, ids[k], p.entry, p.layer, p.seq_len, out[0..mlx.mlx_array_size(p.emb)], p.capture);
        }
        k += 1;
        ctx.ple_pending = null;
        releasePlePending(p);
        if (@import("builtin").is_test) {
            if (layered.verify_fail_after_flush == k) return error.InjectedGroupFlushFailure;
        }
    }
}

/// GPU arm: advance a deferred forward's history from ids the caller already read (the
/// solo MTP verify reads its drafts anyway), so the forward costs no host read of its own.
/// Runs before the round rolls the history back or builds on it. No-op without a record.
pub fn settleDeferredPle(self: *Trunk, ctx: *ForwardCtx, ids: []const u32) !void {
    const p = ctx.ple_pending orelse return;
    if (!p.gpu) return error.PleLeafUnfilled;
    if (ids.len != @as(usize, @intCast(p.seq_len))) return error.PleSettleWidth;
    _ = pleAdvanceSerial(self.qwen4.?, p.entry, ids, p.capture);
    ctx.ple_pending = null;
    releasePlePending(p);
}

/// True when the pending record is the GPU arm's: the embedding is already in the graph
/// and only the history is owed, by `settleDeferredPle` or `flushDeferredPle`.
pub fn deferredPleOnGpu(ctx: *const ForwardCtx) bool {
    const p = ctx.ple_pending orelse return false;
    return p.gpu;
}

pub fn releasePlePending(p: PlePending) void {
    if (p.emb.ctx != null) _ = mlx.mlx_array_free(p.emb);
    _ = mlx.mlx_array_free(p.token_ids);
}

/// Drop a pending leaf without filling it (the forward that built on it
/// is being abandoned). The spec-PLE slot the build claimed is released
/// too — no gather ever filled its tokens.
pub fn discardDeferredPle(_: *Trunk, ctx: *ForwardCtx) void {
    const p = ctx.ple_pending orelse return;
    ctx.ple_pending = null;
    p.entry.spec_ple_len = 0;
    if (ctx.batch_slots) |slots| {
        for (slots) |sc| sc.ssm_entries.?[p.layer].spec_ple_len = 0;
    }
    releasePlePending(p);
}

/// The host side of the n-gram PLE embedding: token ids → hashed rows →
/// gathered + bf16-packed into `pk` (`[n][emb_dim]`), advancing the
/// entry's n-gram history. `capture` comes from `pleClaimSpecCapture` at
/// BUILD time (never re-read off `spec_capture_ssm`, which a deferred
/// flush sees already cleared).
pub fn pleGatherBf16(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int, pk: []u16, capture: bool) !void {
    const ids_c = try self.pleEvalIds(token_ids);
    defer _ = mlx.mlx_array_free(ids_c);
    try self.pleKickOff();
    return self.pleFillFromIds(ctx, ids_c, entry, layer, seq_len, pk, capture);
}

/// Starts the layers a prefill chunk built before its PLE layer (`Trunk.ple_kick`) on the GPU, so they run under the host
/// gather that follows; called once the ids are read, whose eval would queue behind them. The same graph, started
/// earlier: no op is added and no value changes.
pub fn pleKickOff(self: *Trunk) !void {
    const kick = self.ple_kick;
    if (kick.ctx == null) return;
    self.ple_kick = .{ .ctx = null };
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{kick}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_async_eval(v));
    if (!ple_kick_said) {
        ple_kick_said = true;
        log.info("[qwen4] PLE kick engaged: a prefill chunk's layers before the PLE layer start on the GPU once its ids are read, under the host gather\n", .{});
    }
}

var ple_kick_said = false;

/// A prefill of `prompt_ids` is about to start on `ctx`: the table reads the prompt's PLE rows ahead of each chunk's
/// gather (`NgramTable.prefetchBegin`). Nothing on the GPU arm (its kernel reads the table in place) or without a PLE
/// layer.
pub fn plePrefetchBegin(self: *Trunk, ctx: *const ForwardCtx, prompt_ids: []const u32) void {
    const st = self.qwen4 orelse return;
    if (st.gpu != null) return;
    const entries = ctx.ssm_entries orelse return;
    for (self.moe_layers, 0..) |*lw, l| {
        if (lw.ple == null) continue;
        if (l >= entries.len) return;
        const prev = plePrevOf(&entries[l], st.hash.eos);
        st.table.prefetchBegin(&st.hash, prev[0 .. st.hash.ngram_size - 1], prompt_ids, ctx.moe_seq_offset.*);
        return;
    }
}

/// Ends the prefill's prefetch session (`plePrefetchBegin`), on every exit of the prefill.
pub fn plePrefetchEnd(self: *Trunk) void {
    const st = self.qwen4 orelse return;
    st.table.prefetchEnd();
}

/// The GPU arm's host half: read the ids (the one sync), advance the history.
pub fn pleSettleFromArray(self: *Trunk, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, capture: bool) !void {
    // A one-token id is read in place after waiting on its own event (no new ops): the
    // cast+contiguous eval below would queue behind whatever the GPU is running, which for a
    // settle deferred past its own forward is the next forward.
    if (mlx.mlx_array_size(token_ids) == 1) {
        try mlx.check(mlx.mlx_array_eval(token_ids));
        const one: ?u32 = switch (mlx.mlx_array_dtype(token_ids)) {
            .int32 => if (mlx.mlx_array_data_int32(token_ids)) |p| @intCast(p[0]) else null,
            .uint32 => if (mlx.mlx_array_data_uint32(token_ids)) |p| p[0] else null,
            else => null,
        };
        if (one) |id| {
            _ = pleAdvanceSerial(self.qwen4.?, entry, &[_]u32{id}, capture);
            return;
        }
    }
    const ids_c = try self.pleEvalIds(token_ids);
    defer _ = mlx.mlx_array_free(ids_c);
    const ids = try self.pleHostIds(ids_c);
    defer self.allocator.free(ids);
    _ = pleAdvanceSerial(self.qwen4.?, entry, ids, capture);
}

/// `plePrepareIds` evaluated now: the host read of the ids, timed as `ple_sync`.
pub fn pleEvalIds(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const ids_c = try self.plePrepareIds(token_ids);
    errdefer _ = mlx.mlx_array_free(ids_c);
    var sync_lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    try mlx.check(mlx.mlx_array_eval(ids_c));
    self.verify_laps.ple_sync_ns = sync_lap.read();
    return ids_c;
}

/// An evaluated int32 id array copied out as tokens. Caller frees.
pub fn pleHostIds(self: *Trunk, ids_c: mlx.mlx_array) ![]u32 {
    const n: usize = mlx.mlx_array_size(ids_c);
    const ids = try self.allocator.alloc(u32, n);
    errdefer self.allocator.free(ids);
    const src = mlx.mlx_array_data_int32(ids_c) orelse return error.TokenIdsUnreadable;
    for (0..n) |i| ids[i] = @intCast(src[i]);
    return ids;
}

/// The gather ids as one contiguous int32 array, still LAZY: a group evaluates every
/// row's ids together so the flush costs ONE host sync, not one per row.
pub fn plePrepareIds(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const s = if (self.ple_ids_stream.ctx != null) self.ple_ids_stream else self.s;
    var ids_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_i32);
    try mlx.check(mlx.mlx_astype(&ids_i32, token_ids, .int32, s));
    var ids_c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids_c);
    try mlx.check(mlx.mlx_contiguous(&ids_c, ids_i32, false, s));
    return ids_c;
}

/// Host-only half of the gather: `ids_c` must already be evaluated. Everything that
/// MOVES state (the spec token slot, the n-gram history) happens here.
pub fn pleFillFromIds(self: *Trunk, ctx: *ForwardCtx, ids_c: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int, pk: []u16, capture: bool) !void {
    const st = self.qwen4.?;
    const ctx_len: usize = st.hash.ngram_size - 1;
    const ids = try self.pleHostIds(ids_c);
    defer self.allocator.free(ids);
    const n: usize = ids.len;
    const nh = st.hash.n_heads;
    const rows = try self.allocator.alloc(i64, n * nh);
    defer self.allocator.free(rows);
    if (ctx.batch_slots) |slots| {
        const S: usize = @intCast(seq_len);
        std.debug.assert(n == slots.len * S);
        for (slots, 0..) |sc, i| {
            const e = &sc.ssm_entries.?[layer];
            std.debug.assert(e.ple_prev_valid); // batchedGdnReady
            const prev = e.ple_prev;
            const ids_i = ids[i * S .. (i + 1) * S];
            if (capture) {
                std.debug.assert(ctx_len + S <= e.spec_ple_tokens.len);
                fillPleSpecTokens(e, prev[0..ctx_len], ids_i);
                e.spec_ple_len = @intCast(ctx_len + S);
            } else e.spec_ple_len = 0;
            st.hash.rowIds(prev[0..ctx_len], ids_i, rows[i * S * nh ..][0 .. S * nh]);
            advancePlePrev(e, prev, ids_i, ctx_len);
        }
    } else {
        const prev = pleAdvanceSerial(st, entry, ids, capture);
        st.hash.rowIds(prev[0..ctx_len], ids, rows);
    }
    const emb_dim: usize = st.table.dim * nh;
    const host = try self.allocator.alloc(f32, n * emb_dim);
    defer self.allocator.free(host);
    var gclk: ProfClock = if (prof.profileFwdOn()) ProfClock.init() else undefined;
    // The pool's arm is a kv-length question; `ctx.moe_seq_offset` is the pre-chunk
    // position (`cache.step` is 0 forever on a GDN trunk).
    st.table.gather(rows, host, @intCast(ctx.moe_seq_offset.*));
    if (prof.profileFwdOn()) log.info("[qwen4-prof] ple gather S={d}: {d:.2} ms\n", .{ seq_len, @as(f64, @floatFromInt(gclk.lap())) / 1e6 });
    std.debug.assert(pk.len == host.len);
    for (host, pk) |v, *o| o.* = qwen4_mod.bf16Rne(v);
}

/// The history a serial entry's next ids hash against: all eos for a fresh sequence.
pub fn plePrevOf(entry: *const SSMCacheEntry, eos: u32) [8]u32 {
    return if (entry.ple_prev_valid) entry.ple_prev else @splat(eos);
}

/// A serial gather's state moves: the verify slot's `prev ++ ids`, then the history past
/// `ids`. Returns the history `ids` hash against. Both arms run exactly this.
pub fn pleAdvanceSerial(st: *const qwen4_mod.Qwen4State, entry: *SSMCacheEntry, ids: []const u32, capture: bool) [8]u32 {
    const ctx_len: usize = st.hash.ngram_size - 1;
    const prev = plePrevOf(entry, st.hash.eos);
    if (capture) {
        std.debug.assert(ctx_len + ids.len <= entry.spec_ple_tokens.len);
        fillPleSpecTokens(entry, prev[0..ctx_len], ids);
        entry.spec_ple_len = @intCast(ctx_len + ids.len);
    } else entry.spec_ple_len = 0;
    advancePlePrev(entry, prev, ids, ctx_len);
    return prev;
}

/// Qwen4ExpTextPLELayer.forward → the `[B,S,hc*hidden]` addend. Batched
/// decode hands in the MERGED conv window as `entry.aux_state`.
pub fn pleForward(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    return pleForwardWith(self, ctx, stream, token_ids, pw, entry, layer, batch, seq_len, false);
}

/// `stream + pleForward(stream, ...)`: the PLE layer with its call site's residual add (mlx Add of the stream), inside
/// the fused tail's last launch where it runs.
pub fn pleForwardResidual(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    return pleForwardWith(self, ctx, stream, token_ids, pw, entry, layer, batch, seq_len, true);
}

fn pleForwardWith(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer: usize, batch: c_int, seq_len: c_int, residual: bool) !mlx.mlx_array {
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const emb = try self.pleEmbedding(ctx, token_ids, entry, layer, seq_len);
    defer _ = mlx.mlx_array_free(emb);
    if (Trunk.qwen4_trace) |tr| Qwen4Trace.set(&tr.ple_emb, emb);

    const key_raw = try self.qmatmul(emb, pw.key_w, pw.key_s, pw.key_b);
    defer _ = mlx.mlx_array_free(key_raw);
    // The fused tail: the elementwise ops below in four launches (`decodePleForward`); null: the chain
    if (decodePleArmed(self, stream, key_raw, pw, entry)) if (try decodePleForward(self, stream, emb, key_raw, pw, entry, batch, seq_len, residual)) |out| return out;
    const key4 = try self.hcGroupNorm(key_raw, pw.norm_key, batch, seq_len);
    defer _ = mlx.mlx_array_free(key4);
    const value = try self.qmatmul(emb, pw.value_w, pw.value_s, pw.value_b);
    defer _ = mlx.mlx_array_free(value);
    const query4 = try self.hcGroupNorm(stream, pw.norm_query, batch, seq_len);
    defer _ = mlx.mlx_array_free(query4);

    var kq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq);
    try mlx.check(mlx.mlx_multiply(&kq, key4, query4, self.s));
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sum_axis(&gate, kq, -1, true, self.s)); // [B,S,hc,1]
    // the gate's seven ops (one function, which the fused tail tabulates)
    const g_sig = try pleGateChain(self.s, gate, hidden);
    defer _ = mlx.mlx_array_free(g_sig);

    const v_shape = [_]c_int{ batch, seq_len, 1, hidden };
    var value4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(value4);
    try mlx.check(mlx.mlx_reshape(&value4, value, &v_shape, 4, self.s));
    var gv4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv4);
    try mlx.check(mlx.mlx_multiply(&gv4, g_sig, value4, self.s)); // [B,S,hc,hidden]
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    try mlx.check(mlx.mlx_reshape(&gv, gv4, &flat_shape, 3, self.s));
    const gvn4 = try self.hcGroupNorm(gv, pw.norm_conv, batch, seq_len);
    defer _ = mlx.mlx_array_free(gvn4);
    var gvn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gvn);
    try mlx.check(mlx.mlx_reshape(&gvn, gvn4, &flat_shape, 3, self.s));

    // Dilated depthwise causal conv over [state | gvn].
    const dilation: c_int = @intCast(cfg.ngram_size);
    const state_len: c_int = (@as(c_int, @intCast(cfg.ple_conv_kernel)) - 1) * dilation;
    var prev_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_state);
    if (entry.aux_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&prev_state, entry.aux_state));
    } else {
        const zshape = [_]c_int{ batch, state_len, hc * hidden };
        try mlx.check(mlx.mlx_zeros(&prev_state, &zshape, 3, .bfloat16, self.s));
    }
    const parts = [_]mlx.mlx_array{ prev_state, gvn };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, self.s));
    var conv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv);
    try mlx.check(mlx.mlx_conv1d(&conv, cat, pw.conv_w, 1, 0, dilation, hc * hidden, self.s));
    const conv_act = try self.silu(conv);
    defer _ = mlx.mlx_array_free(conv_act);
    // New state = the last state_len rows of the concat (materialized —
    // a view outliving `cat` pins the whole buffer).
    {
        const total: c_int = state_len + seq_len;
        const start = [_]c_int{ 0, total - state_len, 0 };
        const stop = [_]c_int{ batch, total, hc * hidden };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, cat, &start, 3, &stop, 3, &strides, 3, self.s));
        const owned = try materializedOwnedCopy(self.s, view);
        ssmFreeQsaState(entry); // PLE layer: no QSA state to lose
        entry.aux_state = owned;
    }
    if (self.spec_capture_ssm and entry.spec_ple_len > 0) {
        if (entry.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&entry.spec_ple_input, cat));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, gv, conv_act, self.s));
    if (Trunk.qwen4_trace) |tr| Qwen4Trace.set(&tr.ple_out, out);
    if (!residual) return out;
    defer _ = mlx.mlx_array_free(out);
    var h_ple = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h_ple);
    try mlx.check(mlx.mlx_add(&h_ple, stream, out, self.s));
    return h_ple;
}

/// The PLE gate: g -> sigmoid(sqrt(max(|g / sqrt(H)|, 1e-6)) * sign(g / sqrt(H))), seven MLX ops in g's dtype. Caller
/// owns the result.
pub fn pleGateChain(s: mlx.mlx_stream, gate: mlx.mlx_array, hidden: c_int) !mlx.mlx_array {
    // Scalars in the activation dtype: an f32 scalar promotes the gate
    // and, through the value product, the whole residual stream.
    const gate_dt = mlx.mlx_array_dtype(gate);
    const inv_sqrt_h = try scalarOf(1.0 / @sqrt(@as(f32, @floatFromInt(hidden))), gate_dt, s);
    defer _ = mlx.mlx_array_free(inv_sqrt_h);
    var gate_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_sc);
    try mlx.check(mlx.mlx_multiply(&gate_sc, gate, inv_sqrt_h, s));
    // sqrt(max(|g|, 1e-6)) · sign(g)
    var g_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_abs);
    try mlx.check(mlx.mlx_abs(&g_abs, gate_sc, s));
    const floor = try scalarOf(1e-6, gate_dt, s);
    defer _ = mlx.mlx_array_free(floor);
    var g_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_max);
    try mlx.check(mlx.mlx_maximum(&g_max, g_abs, floor, s));
    var g_sqrt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sqrt);
    try mlx.check(mlx.mlx_sqrt(&g_sqrt, g_max, s));
    var g_sign = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sign);
    try mlx.check(mlx.mlx_sign(&g_sign, gate_sc, s));
    var g_signed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_signed);
    try mlx.check(mlx.mlx_multiply(&g_signed, g_sqrt, g_sign, s));
    var g_sig = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g_sig);
    try mlx.check(mlx.mlx_sigmoid(&g_sig, g_signed, s));
    return g_sig;
}

// ── the PLE layer's elementwise tail in four launches ──
//
// Between its MLX projections, norms, gate sum and conv, the chain is 18 elementwise launches on the layer's critical
// path: the group norms' three weight products and the keys x queries product, the gate's seven ops and its value
// product, the conv input's concat, silu's sigmoid and product, the conv branch's add (and the call site's residual
// add). Four launches carry them with every op's arithmetic and rounding unchanged: mlx Multiply / Add on two T values,
// the concat's copy, and the two unary chains (the gate's seven ops, MLX's sigmoid) read from tables of MLX's own
// results over every 16-bit pattern, so no transcendental is re-spelled. Pack-independent: the same bf16 operands on
// any qwen4 pack, so it serves every one.

pub const PLE_DEC_KQ_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_kq.metal");
pub const PLE_DEC_GV_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_gv.metal");
pub const PLE_DEC_CAT_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_cat.metal");
pub const PLE_DEC_TAIL_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail.metal") ++ kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail_out.metal");
/// The tail with the call site's mlx Add of the stream too.
pub const PLE_DEC_TAIL_RES_SOURCE = kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail.metal") ++ kernelSource("plugins/qwen4_exp/kernels/ple_dec_tail_res.metal");

pub const PleDecKind = enum(u8) { kq = 0, gv = 1, cat = 2, tail = 3, tail_res = 4 };
var ple_dec_kernels: [5]?mlx.mlx_fast_metal_kernel = @splat(null);
const PleDecInt = struct { name: [*:0]const u8, value: c_int };
const PleDecCfgKey = struct { shape: ShapeKey, t0: c_int, t1: c_int };
var ple_dec_cfg: [5]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var ple_dec_cfg_key: [5]PleDecCfgKey = undefined;

pub fn pleDecInputs(kind: PleDecKind) []const [*:0]const u8 {
    return switch (kind) {
        .kq => &.{ "nk", "wk", "nq", "wq", "N_size" },
        .gv => &.{ "gate", "value", "gtab", "N_size" },
        .cat => &.{ "prev", "nc", "wc", "N_size", "S_rows" },
        .tail => &.{ "conv", "gv", "sigtab", "N_size" },
        .tail_res => &.{ "conv", "gv", "sigtab", "N_size", "h" },
    };
}

pub fn pleDecOutput(kind: PleDecKind) [*:0]const u8 {
    return switch (kind) {
        .kq => "kq",
        .gv => "gv",
        .cat => "cat",
        .tail, .tail_res => "y",
    };
}

pub const ple_dec_names = [5][*:0]const u8{ "mlxserve_ple_dec_kq", "mlxserve_ple_dec_gv", "mlxserve_ple_dec_cat", "mlxserve_ple_dec_tail", "mlxserve_ple_dec_tail_res" };
pub const ple_dec_sources = [5][:0]const u8{ PLE_DEC_KQ_SOURCE, PLE_DEC_GV_SOURCE, PLE_DEC_CAT_SOURCE, PLE_DEC_TAIL_SOURCE, PLE_DEC_TAIL_RES_SOURCE };

fn pleDecKernel(kind: PleDecKind) !mlx.mlx_fast_metal_kernel {
    const i: usize = @intFromEnum(kind);
    if (ple_dec_kernels[i]) |kr| return kr;
    const ins = pleDecInputs(kind);
    const outs = [_][*:0]const u8{pleDecOutput(kind)};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(ple_dec_names[i], in_vec, out_vec, ple_dec_sources[i], "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    ple_dec_kernels[i] = kernel;
    return kernel;
}

/// One elementwise launch: bf16 output `out_shape` (one thread an element, 256 a group), template T bf16 and `ints`,
/// `inputs` in the kernel's order. The caller owns the result.
pub fn pleDecApply(s: mlx.mlx_stream, kind: PleDecKind, inputs: []const mlx.mlx_array, out_shape: []const c_int, ints: []const PleDecInt) !mlx.mlx_array {
    if (out_shape.len == 0 or out_shape.len > 4) return error.DecodePleShape;
    var n: i64 = 1;
    for (out_shape) |d| n *= d;
    if (n <= 0 or n > std.math.maxInt(c_int) - 256) return error.DecodePleShape;
    const ni: c_int = @intCast(n);
    const tg: c_int = 256;
    const i: usize = @intFromEnum(kind);
    // one config a kernel, rebuilt when the shape or a template constant changes: a config per call is graph-build time
    const key = PleDecCfgKey{ .shape = ShapeKey.from(out_shape), .t0 = if (ints.len > 0) ints[0].value else 0, .t1 = if (ints.len > 1) ints[1].value else 0 };
    if (ple_dec_cfg[i] == null or !std.meta.eql(ple_dec_cfg_key[i], key)) {
        if (ple_dec_cfg[i]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        ple_dec_cfg[i] = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, out_shape.ptr, out_shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, t.name, t.value));
        ple_dec_cfg[i] = config;
        ple_dec_cfg_key[i] = key;
    }
    const kernel = try pleDecKernel(kind);
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, ple_dec_cfg[i].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

/// Launch 1: kq = (nk * wk) * (nq * wq) [B, S, hc, hidden] from the keys' and queries' RMS norms and their norm weights
/// [hc, hidden] (the group norms' weight products and the layer's product).
pub fn pleDecKq(s: mlx.mlx_stream, nk: mlx.mlx_array, wk: mlx.mlx_array, nq: mlx.mlx_array, wq: mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(nk));
    return pleDecApply(s, .kq, &.{ nk, wk, nq, wq, cachedScalarInt(n) }, mlx.getShape(nk), &.{.{ .name = "HW", .value = @intCast(mlx.mlx_array_size(wk)) }});
}

/// Launch 2: gv [B, S, hc * hidden] = table[gate] * value, from the gate's sum [B, S, hc, 1], the value projection
/// [B, S, hidden] and the gate table (`pleGateTable`).
pub fn pleDecGv(s: mlx.mlx_stream, gate: mlx.mlx_array, value: mlx.mlx_array, gtab: mlx.mlx_array, batch: c_int, seq_len: c_int, hc: c_int, hidden: c_int) !mlx.mlx_array {
    const shape = [_]c_int{ batch, seq_len, hc * hidden };
    return pleDecApply(s, .gv, &.{ gate, value, gtab, cachedScalarInt(batch * seq_len * hc * hidden) }, &shape, &.{ .{ .name = "H", .value = hidden }, .{ .name = "HC", .value = hc } });
}

/// Launch 3: cat [B, SL + S, C] = [state | gvn], gvn the RMS-normed gv [B, S, C] times the conv norm's weight [C].
pub fn pleDecCat(s: mlx.mlx_stream, prev: mlx.mlx_array, nc: mlx.mlx_array, wc: mlx.mlx_array, batch: c_int, state_len: c_int, seq_len: c_int, width: c_int) !mlx.mlx_array {
    const shape = [_]c_int{ batch, state_len + seq_len, width };
    return pleDecApply(s, .cat, &.{ prev, nc, wc, cachedScalarInt(batch * (state_len + seq_len) * width), cachedScalarInt(seq_len) }, &shape, &.{ .{ .name = "C", .value = width }, .{ .name = "SL", .value = state_len } });
}

/// Launch 4: gv + silu(conv), plus the stream `h` when given (`pleForwardResidual`); [B, S, C].
pub fn pleDecTail(s: mlx.mlx_stream, conv: mlx.mlx_array, gv: mlx.mlx_array, sigtab: mlx.mlx_array, h: ?mlx.mlx_array) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(conv));
    if (h) |hh| return pleDecApply(s, .tail_res, &.{ conv, gv, sigtab, cachedScalarInt(n), hh }, mlx.getShape(conv), &.{});
    return pleDecApply(s, .tail, &.{ conv, gv, sigtab, cachedScalarInt(n) }, mlx.getShape(conv), &.{});
}

/// The gate's table at the hidden width it was built for (`pleGateTable`): a per-process cache.
var ple_gate_tables: struct { table: ?mlx.mlx_array = null, hidden: c_int = 0 } = .{};

/// `pleGateChain` over every bf16 bit pattern (the raw patterns as its input, so the table is indexed by the gate's own
/// bits, NaN payloads included), evaluated once per hidden width; borrowed by the caller.
pub fn pleGateTable(s: mlx.mlx_stream, hidden: c_int) !mlx.mlx_array {
    if (ple_gate_tables.table) |t| if (ple_gate_tables.hidden == hidden) return t;
    const bits = try std.heap.c_allocator.alloc(u16, 1 << 16);
    defer std.heap.c_allocator.free(bits);
    for (bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{1 << 16}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const table = try pleGateChain(s, x, hidden);
    errdefer _ = mlx.mlx_array_free(table);
    if (mlx.mlx_array_dtype(table) != .bfloat16) return error.DecodePleTableDtype;
    try mlx.check(mlx.mlx_array_eval(table));
    if (ple_gate_tables.table) |old| _ = mlx.mlx_array_free(old);
    ple_gate_tables = .{ .table = table, .hidden = hidden };
    return table;
}

var ple_dec_said = [2]bool{ false, false };

/// The fused tail's one decline line (false: the chain serves the call).
fn decodePleDeclined(why: []const u8) bool {
    if (!ple_dec_said[1]) {
        ple_dec_said[1] = true;
        log.info("[qwen4] PLE tail declined: {s}; the chain serves the PLE layer\n", .{why});
    }
    return false;
}

/// The fused tail's operands: bf16 throughout (the stream, the key projection, the three norm weights [hc, hidden], the
/// conv weight, the conv state if any), a conv state (kernel >= 2), no trace (it reads the chain's `out`).
fn decodePleArmed(self: *Trunk, stream: mlx.mlx_array, key_raw: mlx.mlx_array, pw: *const PleWeights, entry: *const SSMCacheEntry) bool {
    // An EXL3 trunk pack's only: every other pack runs main's PLE chain, and says nothing.
    if (!self.config.trunk_format) return false;
    if (Trunk.qwen4_trace != null or !mlx.streamIsGpu(self.s)) return false;
    const hw: usize = @as(usize, self.config.hc_count) * @as(usize, self.config.hidden_size);
    for ([_]mlx.mlx_array{ stream, key_raw, pw.norm_key, pw.norm_query, pw.norm_conv, pw.conv_w }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return decodePleDeclined("an operand is not bf16");
    }
    for ([_]mlx.mlx_array{ pw.norm_key, pw.norm_query, pw.norm_conv }) |w| {
        if (mlx.mlx_array_size(w) != hw) return decodePleDeclined("a norm weight is not [hc, hidden]");
    }
    if (hw == 0 or mlx.mlx_array_size(stream) % hw != 0 or mlx.mlx_array_size(key_raw) != mlx.mlx_array_size(stream)) return decodePleDeclined("the stream / keys are not [B, S, hc * hidden]");
    if (entry.aux_state.ctx != null and mlx.mlx_array_dtype(entry.aux_state) != .bfloat16) return decodePleDeclined("the conv state is not bf16");
    if (self.config.ple_conv_kernel < 2) return decodePleDeclined("no conv state (kernel 1)");
    return true;
}

/// `hcGroupNorm`'s RMS norm alone ([B, S, hc, hidden]); the launch that reads it applies the norm's weight.
fn hcGroupRms(self: *Trunk, x: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, self.s));
    var n4 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4, x4, self.ones_hidden, self.config.rms_norm_eps, self.s));
    return n4;
}

/// The fused tail: `pleForwardWith`'s chain with its elementwise ops in four launches (`pleDecKq`, `pleDecGv`,
/// `pleDecCat`, `pleDecTail`) and every MLX op between them the chain's own (the value projection, the three RMS norms,
/// the gate's sum, the conv, the state's copy and capture): the same bytes, 14 dispatches and 11 dependent levels fewer
/// per PLE layer (13 / 10 without the residual add). Null where the value projection is not bf16 (the chain runs).
fn decodePleForward(self: *Trunk, stream: mlx.mlx_array, emb: mlx.mlx_array, key_raw: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, batch: c_int, seq_len: c_int, residual: bool) !?mlx.mlx_array {
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const value = try self.qmatmul(emb, pw.value_w, pw.value_s, pw.value_b);
    defer _ = mlx.mlx_array_free(value);
    if (mlx.mlx_array_dtype(value) != .bfloat16) {
        _ = decodePleDeclined("the value projection is not bf16");
        return null;
    }
    // the keys' and queries' norms, then their weight products and product in one launch
    const nk = try hcGroupRms(self, key_raw, batch, seq_len);
    defer _ = mlx.mlx_array_free(nk);
    const nq = try hcGroupRms(self, stream, batch, seq_len);
    defer _ = mlx.mlx_array_free(nq);
    const kq = try pleDecKq(self.s, nk, pw.norm_key, nq, pw.norm_query);
    defer _ = mlx.mlx_array_free(kq);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sum_axis(&gate, kq, -1, true, self.s)); // [B,S,hc,1]
    // the gate's seven ops (one table read) and its value product in one launch: gv [B, S, hc * hidden]
    const gv = try pleDecGv(self.s, gate, value, try pleGateTable(self.s, hidden), batch, seq_len, hc, hidden);
    defer _ = mlx.mlx_array_free(gv);
    const nc = try hcGroupRms(self, gv, batch, seq_len);
    defer _ = mlx.mlx_array_free(nc);

    // Dilated depthwise causal conv over [state | gvn]: gvn's weight product inside the concat launch.
    const dilation: c_int = @intCast(cfg.ngram_size);
    const state_len: c_int = (@as(c_int, @intCast(cfg.ple_conv_kernel)) - 1) * dilation;
    var prev_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_state);
    if (entry.aux_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&prev_state, entry.aux_state));
    } else {
        const zshape = [_]c_int{ batch, state_len, hc * hidden };
        try mlx.check(mlx.mlx_zeros(&prev_state, &zshape, 3, .bfloat16, self.s));
    }
    const cat = try pleDecCat(self.s, prev_state, nc, pw.norm_conv, batch, state_len, seq_len, hc * hidden);
    defer _ = mlx.mlx_array_free(cat);
    var conv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv);
    try mlx.check(mlx.mlx_conv1d(&conv, cat, pw.conv_w, 1, 0, dilation, hc * hidden, self.s));
    // New state = the last state_len rows of the concat (materialized), as the chain keeps it.
    {
        const total: c_int = state_len + seq_len;
        const start = [_]c_int{ 0, total - state_len, 0 };
        const stop = [_]c_int{ batch, total, hc * hidden };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, cat, &start, 3, &stop, 3, &strides, 3, self.s));
        const owned = try materializedOwnedCopy(self.s, view);
        ssmFreeQsaState(entry); // PLE layer: no QSA state to lose
        entry.aux_state = owned;
    }
    if (self.spec_capture_ssm and entry.spec_ple_len > 0) {
        if (entry.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&entry.spec_ple_input, cat));
    }
    if (!ple_dec_said[0]) {
        ple_dec_said[0] = true;
        log.info("[qwen4] PLE decode tail engaged: four launches ({d} rows{s})\n", .{ batch * seq_len, if (residual) ", the residual add inside the last" else "" });
    }
    // silu, the conv branch's add and (residual) the call site's add in one launch
    return try pleDecTail(self.s, conv, gv, try sigmoidTableFor(self.s, .bfloat16), if (residual) stream else null);
}

// ── tests ──

const Qwen4BatchedPleRun = struct {
    logits: mlx.mlx_array,
    pending: bool,
};

pub fn qwen4BatchedPleForward(
    xfm: *Trunk,
    slots: []const *Qwen4TestSlot,
    ids: []const i32,
    seq_len: c_int,
    capture: bool,
    defer_ple: bool,
    do_flush: bool,
) !Qwen4BatchedPleRun {
    const N = slots.len;
    const ctxs = try xfm.allocator.alloc(*ForwardCtx, N);
    defer xfm.allocator.free(ctxs);
    var off_buf = try xfm.allocator.alloc(i32, N);
    defer xfm.allocator.free(off_buf);
    for (slots, 0..) |sl, i| {
        ctxs[i] = &sl.ctx;
        off_buf[i] = @intCast(sl.off);
    }
    const ml = xfm.moe_layers;
    const merged = try xfm.allocator.alloc(SSMCacheEntry, ml.len);
    defer xfm.allocator.free(merged);
    for (merged) |*m| m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
    defer for (merged) |*m| {
        if (m.conv_state.ctx != null) _ = mlx.mlx_array_free(m.conv_state);
        if (m.ssm_state.ctx != null) _ = mlx.mlx_array_free(m.ssm_state);
        if (m.aux_state.ctx != null) _ = mlx.mlx_array_free(m.aux_state);
        ssmFreeSpecCapture(m);
    };
    for (ml, 0..) |*lw, l| {
        if (!lw.is_linear and lw.ple == null) continue;
        merged[l] = try xfm.mergeSsmAcrossSlots(ctxs, l);
    }
    const off_shape = [_]c_int{@intCast(N)};
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &off_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);
    var scratch_offset: usize = @intCast(off_buf[0]);
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = merged,
        .capture_hidden = null,
        .capture_ssm_seq = capture,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = defer_ple,
    };
    const tshape = [_]c_int{ @intCast(N), seq_len };
    const tarr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &tshape, 2, .int32);
    defer _ = mlx.mlx_array_free(tarr);
    const logits = try xfm.forwardQwen4With(&bctx, tarr);
    errdefer _ = mlx.mlx_array_free(logits);
    const pending = bctx.ple_pending != null;
    if (do_flush) try xfm.flushDeferredPle(&bctx) else if (bctx.ple_pending != null) xfm.discardDeferredPle(&bctx);
    for (merged, 0..) |*m, li| {
        if (!ml[li].is_linear and ml[li].ple == null) continue;
        try xfm.splitSsmToSlots(m, ctxs, li);
    }
    for (slots) |sl| sl.off += @intCast(seq_len);
    return .{ .logits = logits, .pending = pending };
}

// ── qwen4 PLE: the GPU arm against the host gather (synthetic table, no model) ──

/// A zeroed Trunk carrying only what `pleEmbedding` reads, over a synthetic 4-bit table.
/// With `gate` the table goes through `ple_gpu.load`, so the `--ple-gpu` switch is what is tested;
/// without it the table is wrapped directly, because `load`'s working-set gate (table plus
/// 16 GB headroom) rightly picks the host gather on a small CI runner, and these tests are
/// about the arms, not the gate. `arm(false)` hides the table buffer to run the host gather.
pub const PleArmFixture = struct {
    fx: ple_gpu.Fixture,
    st: qwen4_mod.Qwen4State,
    gpu: ?ple_gpu.Table,
    xfm_bytes: [@sizeOf(Trunk)]u8 align(@alignOf(Trunk)),
    cache_bytes: [@sizeOf(KVCache)]u8 align(@alignOf(KVCache)),
    off: usize,

    pub fn init(self: *PleArmFixture, gate: ?bool) !void {
        const hash = try qwen4_mod.NgramHash.init(1000, 3, 8, 500, 1, 1234, 0, 999);
        self.fx = try ple_gpu.writeFixture(4, hash.total_rows, 64, 32, 21);
        self.gpu = if (gate) |on| ple_gpu.load(&self.fx.table, on, 0) else try ple_gpu.wrap(&self.fx.table);
        self.st = .{ .hash = hash, .table = self.fx.table, .gpu = self.gpu };
        self.xfm_bytes = @splat(0);
        self.cache_bytes = @splat(0);
        self.off = 0;
        const x = self.xfm();
        x.allocator = testing.allocator;
        x.s = mlx.gpuStream();
        x.qwen4 = &self.st;
    }

    pub fn deinit(self: *PleArmFixture) void {
        self.st.gpu = self.gpu;
        self.st.deinit();
        self.fx.td.cleanup();
    }

    pub fn xfm(self: *PleArmFixture) *Trunk {
        return @ptrCast(&self.xfm_bytes);
    }

    fn ctx(self: *PleArmFixture) ForwardCtx {
        return .{ .cache = @ptrCast(&self.cache_bytes), .moe_seq_offset = &self.off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    }

    pub fn arm(self: *PleArmFixture, on_gpu: bool) !void {
        if (on_gpu and self.gpu == null) return error.NoGpuArm;
        self.st.gpu = if (on_gpu) self.gpu else null;
    }
};

pub fn pleArmEntry() SSMCacheEntry {
    return .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
}

pub fn pleArmIds(ids: []const u32, batch: c_int) mlx.mlx_array {
    var buf: [64]i32 = undefined;
    for (ids, 0..) |v, i| buf[i] = @intCast(v);
    const shape = [_]c_int{ batch, @divExact(@as(c_int, @intCast(ids.len)), batch) };
    return mlx.mlx_array_new_data(&buf, &shape, 2, .int32);
}

fn pleArmRead(emb: mlx.mlx_array) ![]u16 {
    try mlx.check(mlx.mlx_array_eval(emb));
    const d = mlx.mlx_array_data_bfloat16(emb) orelse return error.MlxArrayDataNull;
    return testing.allocator.dupe(u16, d[0..mlx.mlx_array_size(emb)]);
}

/// One serial PLE embedding on the chosen arm, read back as bf16 bits.
fn pleArmEmbed(fx: *PleArmFixture, ctx: *ForwardCtx, entry: *SSMCacheEntry, ids: []const u32, on_gpu: bool) ![]u16 {
    try fx.arm(on_gpu);
    const arr = pleArmIds(ids, 1);
    defer _ = mlx.mlx_array_free(arr);
    const emb = try fx.xfm().pleEmbedding(ctx, arr, entry, 0, @intCast(ids.len));
    defer _ = mlx.mlx_array_free(emb);
    return pleArmRead(emb);
}

pub fn expectPleHistoryEqual(want: *const SSMCacheEntry, got: *const SSMCacheEntry) !void {
    try testing.expectEqual(want.ple_prev_valid, got.ple_prev_valid);
    try testing.expectEqualSlices(u32, &want.ple_prev, &got.ple_prev);
    try testing.expectEqual(want.spec_ple_len, got.spec_ple_len);
    try testing.expectEqualSlices(u32, want.spec_ple_tokens[0..want.spec_ple_len], got.spec_ple_tokens[0..got.spec_ple_len]);
}
