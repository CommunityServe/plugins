//! The serving glue on the K path: the model the plugin's table serves (its trunk, each slot's own state and what the
//! slot owes between forwards), the forward at each mode, the batched tick, the n-gram settle and slot release.
//! Today's `plugins/qwen4_exp/plugin.zig` and `trunk/batched.zig`; the plugin's root calls it at the switch. Text only, on state the plugin owns: the host hands no media, no M-RoPE table and no engine slots.

const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const model = @import("../mlx_serve/sdk/config/model.zig");
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const CaptureLayers = @import("../mlx_serve/engines/mlx/forward_ctx.zig").CaptureLayers;
const layered = @import("../mlx_serve/engines/mlx/batched/layered.zig");
const MAX_BATCH_ROWS = layered.MAX_BATCH_ROWS;
const ssmFreeQsaState = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const q4ple = @import("../mlx_serve/plugins/qwen4_exp/ple.zig");
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const QSA_RING_ROWS = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;
const spec_mtp = @import("../mlx_serve/sdk/spec/mtp/mtp.zig");
const config = @import("config.zig");
const load_mod = @import("load.zig");
const state = @import("cache.zig");
const KVCache = state.KVCache;
const KVQuantConfig = @import("../mlx_serve/sdk/blocks/attention/kv_quant.zig").KVQuantConfig;
const SSMCacheEntry = state.SSMCacheEntry;
const fwd = @import("model.zig");
const ple = @import("ple.zig");

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const gpa = std.heap.c_allocator;

/// The context the drafter lends a solo verify's PLE flush (src/spec/drafter.zig `PleLend`): its id arrays, nothing else.
pub fn engineCtx() *?*const ForwardCtx {
    return EngineCtx.get();
}
pub const EngineCtx = @import("../mlx_serve/sdk/util/shared.zig").SharedThreadLocal(?*const ForwardCtx, "qwen4_engine_ctx", null);

/// GatedDeltaNet layers keep a recurrent state; every `full_attention_interval`-th layer caches KV at the shell's
/// geometry (QSA reads a sparse subset of it past the indexer budget).
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isFullLayer(@intCast(li)))
        .{ .kind = .full, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim }
    else
        .{ .kind = .linear };
    return n;
}

/// A slot's state: the KV cache, the per-layer recurrent entries (GDN pair, PLE window and history, QSA keys) and the
/// next position. Position 0 starts it over.
const OwnSlot = struct {
    id: u32,
    cache: KVCache,
    ssm: []SSMCacheEntry,
    pos: u64 = 0,

    fn init(id: u32, layers: u32, kv: KVQuantConfig) !OwnSlot {
        var cache = try KVCache.initWithConfig(gpa, layers, kv);
        errdefer cache.deinit();
        const ssm = try gpa.alloc(SSMCacheEntry, layers);
        const gen = state.nextSsmMemberGen();
        for (ssm) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false, .member_gen = gen };
        return .{ .id = id, .cache = cache, .ssm = ssm };
    }

    fn deinit(self: *OwnSlot, t: *Trunk) void {
        state.ssmGroupDropMember(&t.ssm_group, t.s, self.ssm) catch {};
        for (self.ssm) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        gpa.free(self.ssm);
        self.cache.deinit();
    }
};

/// What a slot id carries between forwards: the n-gram PLE work a `defer_host` forward left owed, the entries its last
/// forward ran on (a batched group may hold them), and whether its request reads an affine cache fused (`kv_attn_fused`).
const Rec = struct {
    pending: ?q4ple.PlePending = null,
    ssm: ?[]SSMCacheEntry = null,
    kv_attn_fused: bool = false,
};

pub const Model = struct {
    cfg: model.ModelConfig,
    weights: model.Weights,
    trunk: Trunk,
    /// The cache a settle-only context names (it reads no cache).
    scratch: KVCache,
    caps: abi.Caps,
    /// The slots' KV scheme, as the server's load resolves it (src/host_flags.zig).
    kv: KVQuantConfig = KVQuantConfig.dense,
    own: std.ArrayList(OwnSlot) = .empty,
    recs: std.AutoHashMapUnmanaged(u32, Rec) = .empty,

    fn rec(m: *Model, id: u32) !*Rec {
        const g = try m.recs.getOrPut(gpa, id);
        if (!g.found_existing) g.value_ptr.* = .{};
        return g.value_ptr;
    }

    /// A context for host work on `r`'s owed record alone (no forward reads its cache).
    fn settleCtx(m: *Model, r: *Rec, offset: *usize) ForwardCtx {
        return .{ .cache = &m.scratch, .moe_seq_offset = offset, .ssm_entries = r.ssm, .capture_hidden = null, .vision_embeddings = null, .ple_pending = r.pending };
    }

    pub fn ownSlot(m: *Model, id: u32, pos: u64) !*OwnSlot {
        for (m.own.items) |*o| if (o.id == id) {
            if (pos == 0) {
                o.deinit(&m.trunk);
                o.* = try OwnSlot.init(id, m.cfg.num_hidden_layers, m.kv);
            } else if (o.pos != pos) return error.PositionDoesNotFollow;
            return o;
        };
        if (pos != 0) return error.PositionDoesNotFollow;
        try m.own.append(gpa, try OwnSlot.init(id, m.cfg.num_hidden_layers, m.kv));
        return &m.own.items[m.own.items.len - 1];
    }

    /// Whether slot `id`'s request reads its affine cache fused below the QSA budget (its prompt's, `handPrompt`).
    pub fn setKvAttnFused(m: *Model, id: u32, fused: bool) !void {
        (try m.rec(id)).kv_attn_fused = fused;
    }

    /// Row `i`'s slot's read mode (`slotOf`'s id).
    fn fusedOf(m: *Model, f: *const abi.Forward, i: usize) bool {
        const r = m.recs.get(if (f.slots.len > i) f.slots.slice()[i] else @intCast(i)) orelse return false;
        return r.kv_attn_fused;
    }
};

/// The K pack's model (`load.zig`): a failure is refused by its name.
pub fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return load_mod.loadModel(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "turboexl3_qwen38: out of memory"),
            else => Verdict.named(.failed, @errorName(e)),
        };
        return null;
    };
}

/// QSA's memory terms as the engine bills them: each slot's raw key ring (`QSA_RING_ROWS` rows per attention layer),
/// the pooled history per token (once: the slots share it; the score kernel keeps no f32 bank) and the SSM checkpoint.
pub fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    out.* = .{
        .fixed_slot_bytes = @as(u64, c.attnCacheLayerCount()) * @as(u64, @intCast(QSA_RING_ROWS)) * c.indexer_head_dim * 2,
        .state_per_token = c.qsaHistoryBytesPerToken(),
        .ssm_checkpoint = c.ssmCheckpointBytes(),
        .prefill_bytes = &prefillBytes,
        .batched_kv_len = &batchedKvLen,
    };
}

/// A prefill chunk's QSA transient at the prompt's length: the score kernel's sheet over the rows its 256 MB budget
/// takes at once, and the select's bounds; a chunk under 16 rows builds the dense mask instead.
fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, _: u8, _: u64) callconv(.c) u64 {
    const c = config.of(p);
    const q_rows: u64 = @min(chunk, @max(seq, 1));
    if (q_rows < 16) return 4 * q_rows * seq * 5 / 4;
    const nb = seq / c.indexer_compress_ratio;
    const rows: u64 = if (nb == 0) q_rows else @max(16, @min(q_rows, (256 << 20) / (nb * 4)));
    return rows * nb * 4 + q_rows * @min(nb, 512) * 4;
}

/// The keys a batched tick reads per slot: past the gather floor, the indexer's budget and its open block.
fn batchedKvLen(p: *const abi.Parsed, kv: u32, gather_on: bool, min: u32) callconv(.c) u32 {
    const c = config.of(p);
    if (!gather_on or kv <= min) return kv;
    return @min(kv, c.indexer_budget + c.indexer_compress_ratio);
}

/// The resident bill: the shards and, unless the load's settings turn MTP off, the rerank head the MTP head builds at
/// load: the coarse copy, the re-score copy and the coarse vocab subset (`mtp_head.qwen4BuildDraftRerank`).
pub fn residentBill(p: *const abi.Parsed, model_dir: abi.Str, settings_json: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    _ = model_dir;
    const c = config.of(p);
    out.* = shards;
    if (mtpSetting(settings_json.slice()) == false) return .ok;
    const v: c_int = @intCast(c.vocab_size);
    const h: c_int = @intCast(c.hidden_size);
    out.* +|= spec_mtp.rerankCoarseBytes(v, h, spec_mtp.DRAFT_HEAD_BITS) + spec_mtp.rerankCoarseBytes(v, h, spec_mtp.RERANK_EXACT_BITS);
    if (v >= spec_mtp.DRAFT_VOCAB_MIN) out.* +|= spec_mtp.rerankCoarseBytes(spec_mtp.DRAFT_VOCAB_ROWS, h, spec_mtp.DRAFT_HEAD_BITS);
    return .ok;
}

/// `mtp` as the load settings state it (`--no-mtp` and a model's own setting); null = the default (on).
pub fn mtpSetting(settings_json: []const u8) ?bool {
    if (settings_json.len == 0) return null;
    const S = struct { mtp: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.mtp;
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    var it = m.recs.iterator();
    while (it.next()) |e| if (e.value_ptr.pending) |p| ple.releasePlePending(p);
    m.recs.deinit(gpa);
    for (m.own.items) |*o| o.deinit(&m.trunk);
    m.own.deinit(gpa);
    m.trunk.deinit();
    m.scratch.deinit();
    m.weights.deinit();
    m.cfg.deinit(gpa);
    gpa.destroy(m);
}

/// Slot `i`'s own state at its position.
fn slotOf(m: *Model, f: *const abi.Forward, i: usize) !*OwnSlot {
    if (f.cache_ops) |ops| if (ops.sdkSlots()) return error.DirectModelOwnsItsSlots;
    const pos: u64 = if (f.positions.len > i) f.positions.slice()[i] else 0;
    const id: u32 = if (f.slots.len > i) f.slots.slice()[i] else @intCast(i);
    return m.ownSlot(id, pos);
}

fn tokenArray(f: *const abi.Forward, slots: usize) !Array {
    if (f.ids.ctx != null) {
        var a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&a, .{ .ctx = f.ids.ctx }));
        return a;
    }
    var buf: [4096]i32 = undefined;
    const t = f.tokens.slice();
    if (t.len > buf.len or t.len != slots * f.rows) return error.TokenCount;
    for (t, 0..) |id, k| buf[k] = @intCast(id);
    return mlx.mlx_array_new_data(&buf, &[_]c_int{ @intCast(slots), @intCast(f.rows) }, 2, .int32);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "turboexl3_qwen38: forward mode not declared");
    if (f.mrope.ctx != null or f.media.ctx != null) return Verdict.refuse(.unsupported, "turboexl3_qwen38: text only");
    if (f.rows == 0) return Verdict.refuse(.failed, "turboexl3_qwen38: a forward of no rows");
    const slots = @max(f.positions.len, 1);
    if (f.mode != .batched_rows and slots != 1) return Verdict.refuse(.unsupported, "turboexl3_qwen38: one slot's rows at a time outside batched rows");
    if (f.ids.ctx == null) for (f.tokens.slice()) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "turboexl3_qwen38: token id past the vocabulary");
    const r = if (f.mode == .batched_rows) runBatched(m, f, out) else runOne(m, f, out);
    r catch |e| return switch (e) {
        error.PositionDoesNotFollow => Verdict.refuse(.failed, "turboexl3_qwen38: the position does not follow the slot's state"),
        else => Verdict.named(.failed, @errorName(e)),
    };
    return .ok;
}

/// `logits` `[B, S, V]` as the table returns it: every row, or each slot's last one.
fn shapeLogits(m: *Model, logits: Array, all: bool) !Array {
    const sh = mlx.getShape(logits);
    const s = m.trunk.s;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (all) {
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ sh[0] * sh[1], sh[2] }, 2, s));
        return out;
    }
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_slice(&last, logits, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ sh[0], sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_reshape(&out, last, &[_]c_int{ sh[0], sh[2] }, 2, s));
    return out;
}

fn runOne(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const o = try slotOf(m, f, 0);
    const ids = try tokenArray(f, 1);
    defer _ = mlx.mlx_array_free(ids);
    var offset: usize = @intCast(if (f.positions.len > 0) f.positions.slice()[0] else 0);
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    var captured: [64]Array = undefined;
    const cap_ids = f.capture_layers.slice();
    if (cap_ids.len > captured.len) return error.TooManyCaptures;
    for (captured[0..cap_ids.len]) |*c| c.* = mlx.mlx_array_new();
    defer for (captured[0..cap_ids.len]) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var cl: CaptureLayers = .{ .ids = cap_ids, .out = captured[0..cap_ids.len] };
    const r = try m.rec(if (f.slots.len > 0) f.slots.slice()[0] else 0);
    r.ssm = o.ssm;
    const lend = engineCtx().*;
    if (lend) |e| if (e.mrope_pos != null) return error.DirectModelIsTextOnly;
    // A verify leaves each recurrent row's state for the partial-accept rollback; the stream is the head's input.
    var ctx: ForwardCtx = .{
        .cache = &o.cache,
        .moe_seq_offset = &offset,
        .ssm_entries = o.ssm,
        .capture_hidden = null,
        .capture_hidden_all = if (f.want_hidden and f.logits != .none) &hidden else null,
        .capture_layers = if (cap_ids.len > 0) &cl else null,
        .capture_ssm_seq = f.mode == .verify,
        .vision_embeddings = null,
        .spec_verify = f.mode == .verify,
        .skip_lm_head = f.logits == .none,
        .ple_defer = f.defer_host,
        .ple_pending = r.pending,
        .ple_ids_src = if (lend) |e| e.ple_ids_src else null,
        .kv_attn_fused = r.kv_attn_fused,
    };
    r.pending = null;
    defer r.pending = ctx.ple_pending;
    try state.ssmGroupReleaseCtx(&m.trunk.ssm_group, m.trunk.s, &ctx);
    const res = fwd.forward(&m.trunk, &ctx, ids) catch |e| {
        ple.discardDeferredPle(&m.trunk, &ctx);
        return e;
    };
    defer _ = mlx.mlx_array_free(res);
    o.pos = offset;
    if (ctx.ple_pending) |p| out.owed = if (p.gpu) .history else .input;
    if (f.logits == .none) {
        // No head: the result is the mixer output, the vector the lm_head reads.
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, res);
        out.hidden = .{ .ctx = h.ctx };
    } else out.logits = .{ .ctx = (try shapeLogits(m, res, f.logits == .all_rows)).ctx };
    if (f.want_hidden and f.logits != .none) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        out.hidden = .{ .ctx = h.ctx };
    }
    if (out.captured) |dst| for (captured[0..cap_ids.len], 0..) |c, k| {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, c);
        dst[k] = .{ .ctx = h.ctx };
    };
}

/// One row per slot at its own position: the batched tick when every slot's recurrent state is live, else each
/// slot's serial step.
fn runBatched(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.positions.len;
    if (n == 0 or n > MAX_BATCH_ROWS) return error.BatchShape;
    if (f.rows > 1) return runBatchedVerify(m, f, out);
    if (f.ids.ctx != null or f.tokens.len != n) return error.BatchShape;
    var own: [MAX_BATCH_ROWS]*OwnSlot = undefined;
    var offsets: [MAX_BATCH_ROWS]usize = undefined;
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (0..n) |i| {
        own[i] = try slotOf(m, f, i);
        offsets[i] = f.positions.slice()[i];
        ctxs[i] = .{ .cache = &own[i].cache, .moe_seq_offset = &offsets[i], .ssm_entries = own[i].ssm, .capture_hidden = null, .vision_embeddings = null, .kv_attn_fused = m.fusedOf(f, i) };
        ptrs[i] = &ctxs[i];
    }
    var rows: [MAX_BATCH_ROWS]Array = undefined;
    var made: usize = 0;
    defer for (rows[0..made]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var hidden: [MAX_BATCH_ROWS]Array = undefined;
    var made_hidden: usize = 0;
    defer for (hidden[0..made_hidden]) |h| {
        _ = mlx.mlx_array_free(h);
    };
    // The tick's PLE gather defers inside it and settles once, before it returns: nothing is left owed.
    if (n > 1 and batchedGdnReady(&m.trunk, ptrs[0..n])) {
        var hrows: ?[]Array = null;
        const got = try batchedDecode(&m.trunk, f.tokens.slice(), ptrs[0..n], f.positions.slice(), if (f.want_hidden) &hrows else null);
        defer gpa.free(got);
        @memcpy(rows[0..n], got);
        made = n;
        if (hrows) |hr| {
            defer gpa.free(hr);
            @memcpy(hidden[0..n], hr);
            made_hidden = n;
        }
    } else for (0..n) |i| {
        var id: i32 = @intCast(f.tokens.slice()[i]);
        const ids = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        try state.ssmGroupReleaseCtx(&m.trunk.ssm_group, m.trunk.s, ptrs[i]);
        if (f.want_hidden) {
            hidden[i] = mlx.mlx_array_new();
            made_hidden += 1;
            ptrs[i].capture_hidden = &hidden[i];
        }
        rows[i] = try fwd.forward(&m.trunk, ptrs[i], ids);
        made += 1;
    }
    for (0..n) |i| {
        own[i].pos = f.positions.slice()[i] + 1;
        if (f.slots.len > i) {
            const r = try m.rec(f.slots.slice()[i]);
            r.ssm = own[i].ssm;
        }
    }
    out.logits = .{ .ctx = (try shapeLogits(m, try concatRows(m, rows[0..n]), false)).ctx };
    if (made_hidden == n and f.want_hidden) out.hidden = .{ .ctx = (try concatRows(m, hidden[0..n])).ctx };
}

/// Every slot's `[t1, drafts…]` row of `f.ids` `[N, S]` at its own position: one batched verify, each recurrent row's
/// state captured for the slot's own rollback; logits for every row, the stream at every row.
fn runBatchedVerify(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.positions.len;
    if (f.ids.ctx == null) return error.BatchShape;
    var own: [MAX_BATCH_ROWS]*OwnSlot = undefined;
    var offsets: [MAX_BATCH_ROWS]usize = undefined;
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (0..n) |i| {
        own[i] = try slotOf(m, f, i);
        offsets[i] = f.positions.slice()[i];
        ctxs[i] = .{ .cache = &own[i].cache, .moe_seq_offset = &offsets[i], .ssm_entries = own[i].ssm, .capture_hidden = null, .vision_embeddings = null, .kv_attn_fused = m.fusedOf(f, i) };
        ptrs[i] = &ctxs[i];
    }
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    const logits = try batchedVerify(&m.trunk, .{ .ctx = f.ids.ctx }, ptrs[0..n], f.positions.slice(), &last, &all);
    defer _ = mlx.mlx_array_free(logits);
    for (0..n) |i| {
        own[i].pos = f.positions.slice()[i] + f.rows;
        if (f.slots.len > i) {
            const r = try m.rec(f.slots.slice()[i]);
            r.ssm = own[i].ssm;
        }
    }
    out.logits = .{ .ctx = (try shapeLogits(m, logits, true)).ctx };
    out.hidden = .{ .ctx = all.ctx };
}

/// `xs` stacked on the batch axis (a new array).
fn concatRows(m: *Model, xs: []const Array) !Array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    var cat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, m.trunk.s));
    return cat;
}

/// Settles what `slots` were left owed: the n-gram rows of a deferred leaf (`input`), off the device; `.discard` drops
/// it and the spec slot it claimed. A K pack's leaves are the host gather's: none settles from the host's `tokens`.
pub fn forwardSettle(mp: *anyopaque, slots: abi.Span(u32), how: abi.Settle, tokens: *const abi.Span(u32)) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    settleSlots(m, slots.slice(), how, tokens.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn settleSlots(m: *Model, ids: []const u32, how: abi.Settle, tokens: []const u32) !void {
    if (ids.len > MAX_BATCH_ROWS) return error.PleGroupTooWide;
    var offs: [MAX_BATCH_ROWS]usize = @splat(0);
    var ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    var n: usize = 0;
    for (ids) |id| {
        const r = m.recs.getPtr(id) orelse continue;
        if (r.pending == null) continue;
        ctxs[n] = m.settleCtx(r, &offs[n]);
        ptrs[n] = &ctxs[n];
        r.pending = null;
        n += 1;
    }
    if (n == 0) return;
    // A failed settle leaves nothing owed behind it: the engine discards the forward.
    defer for (ctxs[0..n]) |*c| if (c.ple_pending) |p| ple.releasePlePending(p);
    switch (how) {
        .discard => for (ctxs[0..n]) |*c| ple.discardDeferredPle(&m.trunk, c),
        .flush => if (tokens.len > 0) {
            if (n != 1) return error.PleSettleWidth;
            try ple.settleDeferredPle(&m.trunk, &ctxs[0], tokens);
        } else if (n == 1) try ple.flushDeferred(&m.trunk, &ctxs[0]) else try ple.flushDeferredPleGroup(&m.trunk, ptrs[0..n]),
    }
}

/// The slot id is gone: what it was owed dies with it, a batched group lets its entries go, and its own state is freed.
pub fn releaseSlot(mp: *anyopaque, slot: u32) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (m.recs.fetchRemove(slot)) |kv| {
        var r = kv.value;
        var off: usize = 0;
        var c = m.settleCtx(&r, &off);
        ple.discardDeferredPle(&m.trunk, &c);
        if (r.ssm) |entries| state.ssmGroupDropMember(&m.trunk.ssm_group, m.trunk.s, entries) catch {};
    }
    m.trunk.markQsaPooledRopeStale();
    for (m.own.items, 0..) |*o, i| if (o.id == slot) {
        o.deinit(&m.trunk);
        _ = m.own.swapRemove(i);
        break;
    };
}

// ── the batched tick: one decode row (or one verify block) per slot through one forward, every slot's GDN state and
// PLE window merged into a batch-N entry (kept merged across ticks by the persistent group), each slot's QSA keys read
// from its own cache ──

/// Every slot's next token `[N, 1]` in one forward; N logits `[1, 1, V]` (and, with `hidden_rows`, each slot's
/// pre-mixer stream row) the caller owns.
fn batchedDecode(t: *Trunk, next_tokens: []const u32, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_rows: ?*?[]Array) ![]Array {
    const token_buf = try t.allocator.alloc(i32, next_tokens.len);
    defer t.allocator.free(token_buf);
    for (next_tokens, token_buf) |id, *b| b.* = @intCast(id);
    const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &[_]c_int{ @intCast(next_tokens.len), 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    const logits = try batchedRows(t, token_arr, ctxs, rope_offsets, false, if (hidden_rows != null) &hidden else null, null);
    defer _ = mlx.mlx_array_free(logits);
    if (hidden_rows) |out| out.* = try layered.sliceBatchRows(t.allocator, t.s, hidden, next_tokens.len);
    return layered.sliceBatchRows(t.allocator, t.s, logits, next_tokens.len);
}

/// `[N, S]` rows (every slot's `[t1, drafts…]`) with each recurrent row's state captured for the slot's own rollback;
/// logits `[N, S, V]`, the stream at the last row and at every row.
fn batchedVerify(t: *Trunk, token_arr: Array, ctxs: []const *ForwardCtx, rope_offsets: []const u32, hidden_last: *Array, hidden_all: *Array) !Array {
    return batchedRows(t, token_arr, ctxs, rope_offsets, true, hidden_last, hidden_all);
}

fn batchedRows(t: *Trunk, token_arr: Array, ctxs: []const *ForwardCtx, rope_offsets: []const u32, capture_ssm: bool, hidden_last: ?*Array, hidden_all: ?*Array) !Array {
    t.fwd_gen +%= 1;
    const ml = t.moe_layers;
    const off_buf = try t.allocator.alloc(i32, rope_offsets.len);
    defer t.allocator.free(off_buf);
    for (rope_offsets, off_buf) |o, *b| b.* = @intCast(o);
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &[_]c_int{@intCast(ctxs.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);

    // Each linear (and PLE) layer's per-slot state merged into one batch-N entry, the group's own across ticks;
    // `batchedGdnReady` gates on every slot carrying it.
    const stable = state.ssmGroupMatches(&t.ssm_group, ctxs);
    var kinds_buf: []state.SsmMergeKind = &.{};
    defer if (kinds_buf.len != 0) t.allocator.free(kinds_buf);
    const kinds: []const state.SsmMergeKind = if (stable) t.ssm_group.kinds else blk: {
        kinds_buf = try t.allocator.alloc(state.SsmMergeKind, ml.len);
        for (ml, kinds_buf) |*lw, *k| k.* = .{ .linear = lw.is_linear, .ple = lw.ple != null };
        break :blk kinds_buf;
    };
    const merged = try state.ssmBindTickState(t.allocator, t.s, &t.ssm_group, ctxs, kinds);

    // `cache` and `moe_seq_offset` are never read on this path (attention reaches each slot's cache through
    // `batch_slots`, rope through the offsets array); slot 0's stand in for the non-optional fields.
    var scratch_offset: usize = rope_offsets[0];
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = merged,
        .capture_hidden = hidden_last,
        .capture_hidden_all = hidden_all,
        .capture_ssm_seq = capture_ssm,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = true,
    };
    // A verify's captures live on the merged entries until split below.
    defer if (capture_ssm) {
        for (merged) |*mm| state.ssmFreeSpecCapture(mm);
    };
    const logits = fwd.forward(t, &bctx, token_arr) catch |e| {
        ple.discardDeferredPle(t, &bctx);
        return e;
    };
    errdefer {
        ple.discardDeferredPle(t, &bctx);
        _ = mlx.mlx_array_free(logits);
    }
    try ple.flushDeferred(t, &bctx);
    if (capture_ssm) {
        for (merged, 0..) |*mm, li| {
            if (!kinds[li].linear and !kinds[li].ple) continue;
            for (ctxs, 0..) |c, i| try state.splitSpecCaptures(t.s, mm, &c.ssm_entries.?[li], i);
        }
    }
    return logits;
}

/// Can these slots take one batched tick? Every slot must carry live state on every linear layer and its PLE window
/// and token history (true after any prefill).
fn batchedGdnReady(t: *const Trunk, ctxs: []const *ForwardCtx) bool {
    const ml = t.moe_layers;
    for (ctxs) |c| {
        const entries = c.ssm_entries orelse return false;
        if (entries.len != ml.len) return false;
        for (ml, entries) |*lw, *e| {
            if (lw.ple != null and (e.aux_state.ctx == null or !e.ple_prev_valid)) return false;
            if (!lw.is_linear) continue;
            if (!e.initialized) return false;
            if (e.conv_state.ctx == null or e.ssm_state.ctx == null) return false;
        }
    }
    return true;
}

// ── tests ──

test "direct serve: a K pack's resident bill holds the rerank heads unless MTP is off" {
    const c: config.Config = .{ .vocab_size = 248320, .hidden_size = 2560 };
    var out: u64 = 0;
    try std.testing.expect(residentBill(&.{ .private = @constCast(&c) }, .of(""), .{}, 1000, &out).isOk());
    // the coarse copy (278.1 MB), the re-score copy (675.4 MB) and the vocab subset (110.1 MB): the load's 1,014 MiB
    try std.testing.expectEqual(@as(u64, 1000 + 1_063_649_280), out);
    try std.testing.expect(residentBill(&.{ .private = @constCast(&c) }, .of(""), .of("{\"mtp\":false}"), 1000, &out).isOk());
    try std.testing.expectEqual(@as(u64, 1000), out);
}

