//! copied_from: src/plugins/qwen4_exp/wave.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The wavefront prefill: a wide prefill chunk as two halves, the later one phase behind the earlier on a second GPU
//! stream, so one half's routed-expert GEMMs run beside the other's recurrence, hc norms and bandwidth-bound launches.
//! A layer is two phases, its mixer side (PLE, attention-side hc read, GDN or QSA mixer) and its MoE side (MLP-side hc
//! read, MoE); slot t builds A's phase t, then B's phase t - 1. Layer i of B needs of A only what a chunk needs of the
//! chunk before it (the K/V and QSA rows, the GDN and PLE states, the n-gram history), all of it inside MLX's graph,
//! so MLX orders B's reads after A's writes across the two queues. Each half runs at its own row count, so GEMM tiles
//! and split-K may differ from the whole chunk's: a reordering path (`Trunk.reordered`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const model = @import("../../sdk/config/model.zig");
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const KVCacheEntry = @import("../../engines/mlx/kv/cache.zig").KVCacheEntry;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const layerCap = @import("../../engines/mlx/diag/profile.zig").layerCap;
const dispatchProbeCount = @import("../../engines/mlx/diag/profile.zig").dispatchProbeCount;
const concatAxis1 = @import("../../mlx/array.zig").concatAxis1;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const profileFwdOn = @import("prof.zig").profileFwdOn;
const HcPending = @import("hc_fused.zig").HcPending;
const SideStream = @import("side_stream.zig").SideStream;
const newStream = @import("side_stream.zig").newStream;
const Trunk = @import("trunk.zig").Trunk;
const pleForwardResidual = @import("ple.zig").pleForwardResidual;

/// The smallest half: below 1024 rows a half's fixed costs (the launches, the routed experts' part-empty M tiles)
/// spread over too few rows.
pub const PF_WAVE_MIN_HALF: usize = 1024;
/// The earlier half ends on a multiple of this: whole NAX M tiles and whole QSA blocks, so the later half starts on a
/// block boundary (no leftover keys to re-pool).
pub const PF_WAVE_ALIGN: usize = 64;
/// A half's phases in flight past the one the host waits on: the transients of two chunk-wide layers, under the chain's
/// four between cadence evals.
pub const PF_WAVE_AHEAD: usize = 4;
/// The widest attention layer's K + V the later half's write may copy while the earlier half still reads it (a
/// 256K-token buffer at 2 KiB a token and layer); past it the chunk runs as one.
pub const PF_WAVE_KV_COPY_MAX: usize = 512 << 20;

/// A chunk's halves: A = its first `a` rows, B = the `b` after them.
pub const PfWaveHalves = struct { a: usize, b: usize };

/// The halves of a `rows`-row chunk: A is rows / 2 rounded down to PF_WAVE_ALIGN, B the rest; null when A falls under
/// `min_half` (the chunk runs as one).
pub fn pfWavePlan(rows: usize, min_half: usize) ?PfWaveHalves {
    const a = rows / 2 / PF_WAVE_ALIGN * PF_WAVE_ALIGN;
    if (a == 0 or a < min_half) return null;
    return .{ .a = a, .b = rows - a };
}

var engaged = false;
var declined_said = false;
/// Chunks the wavefront served (the tests' proof the path ran).
pub var pf_wave_calls: u64 = 0;
/// Test seam: every chunk on the chain (the forward test's reference arm).
pub var chain_override = false;

/// What decides whether a chunk the plan splits takes the wavefront (`pfWaveRefusal`).
pub const PfWaveFacts = struct {
    /// the forward's batch (a prefill chunk's is 1)
    batch: c_int = 1,
    /// the trunk's stream is a Metal GPU stream
    gpu: bool = true,
    /// `forwardPlain`: no stand-in, trace, profiler, layer cap, dispatch probe or distributed rank
    plain: bool = true,
    /// `pfWaveCtxPlain`: the ctx carries a prefill chunk's fields alone
    ctx_plain: bool = true,
    /// `pfWaveHostIds`: the token ids are row-contiguous host data, so each half's ids are a host copy
    host_ids: bool = true,
    /// `pfWaveKvCopyBytes`
    kv_copy_bytes: usize = 0,
};

/// Why a chunk the plan splits declines the wavefront, or null.
pub fn pfWaveRefusal(f: PfWaveFacts) ?[]const u8 {
    if (f.batch != 1) return "a batched forward";
    if (!f.gpu) return "not a Metal GPU stream";
    if (!f.plain) return "a stand-in, trace, profiler, layer cap or distributed rank changes what the forward builds";
    if (!f.ctx_plain) return "the forward carries more than a prefill chunk (batched slots, vision, M-RoPE, layer captures, spec state, a deferred PLE leaf)";
    if (!f.host_ids) return "the token ids are not row-contiguous host data";
    if (f.kv_copy_bytes > PF_WAVE_KV_COPY_MAX) return "an attention layer's K/V buffer is past PF_WAVE_KV_COPY_MAX";
    return null;
}

fn pfWaveDeclined(why: []const u8, rows: usize) ?PfWaveHalves {
    if (!declined_said) {
        declined_said = true;
        log.info("[pf-wave] declined at S={d}: {s}; the chunk runs as one\n", .{ rows, why });
    }
    return null;
}

/// The ctx carries a prefill chunk's fields alone: no batched slots, rank hand-off, vision rows or M-RoPE table, layer
/// captures, spec capture or verify, tree, placed head row, mask term, deferred PLE or its id sources, build ladder or
/// head entry.
pub fn pfWaveCtxPlain(ctx: *const ForwardCtx) bool {
    return ctx.ssm_entries != null and ctx.batch_slots == null and ctx.stream_in == null and ctx.vision_embeddings == null and
        ctx.mrope_pos == null and ctx.mrope_cos_cur == null and ctx.capture_layers == null and !ctx.capture_ssm_seq and
        ctx.prefill_mask_add == null and !ctx.spec_verify and ctx.tree == null and ctx.head_place == null and
        ctx.head_rope_pos.ctx == null and !ctx.ple_defer and ctx.ple_pending == null and ctx.ple_ids_src == null and
        ctx.pipeline_build == 0 and ctx.qsa_entry == null and ctx.batch_rope_offsets == null and ctx.key_pad_mask == null;
}

/// The token ids are [1, S] int32 / uint32 host data, row-contiguous and available, so `pfWaveIds` can copy each half's.
pub fn pfWaveHostIds(token_ids: mlx.mlx_array) bool {
    const dt = mlx.mlx_array_dtype(token_ids);
    if (dt != .int32 and dt != .uint32) return false;
    if (mlx.mlx_array_ndim(token_ids) != 2 or mlx.mlx_array_strides(token_ids)[1] != 1) return false;
    var ready = false;
    if (mlx._mlx_array_is_available(&ready, token_ids) != 0) return false;
    return ready;
}

/// The K + V bytes of the widest attention layer's buffer once this chunk is written (dense bf16 at the config's K/V
/// heads, at least `kv_end` rows and the request's reservation): what the later half's K/V write copies at most.
pub fn pfWaveKvCopyBytes(cache: *const KVCache, cfg: *const model.ModelConfig, kv_end: usize) usize {
    var rows: usize = @max(kv_end, cache.reserve_tokens);
    for (cache.entries) |*e| {
        if (e.initialized) rows = @max(rows, KVCache.bufferCapacity(e.keys));
    }
    return rows * 2 * @as(usize, cfg.num_key_value_heads) * @as(usize, cfg.head_dim) * 2;
}

/// The later half's stream on this inference thread (MLX registers a stream's command encoders on the thread that
/// makes it), made at the first chunk that engages, with its side stream (`SideStream.Tls.wave_side`); `failed`: MLX
/// made none on this thread, so its chunks run as one.
pub const Later = struct { stream: mlx.mlx_stream = .{ .ctx = null }, failed: bool = false };
pub const later_shared = @import("../../sdk/util/shared.zig").SharedThreadLocal(Later, "qwen4_wave_stream", .{});

/// The wavefront's streams beside `main`: false (the chunk declines) when MLX makes no second stream. Without a side
/// stream of its own the later half's side-stream parts stay on its stream.
fn pfWaveStreams(main: mlx.mlx_stream) bool {
    const later = later_shared.get();
    if (later.stream.ctx == null) {
        if (later.failed) return false;
        later.stream = newStream(main) orelse {
            later.failed = true;
            return false;
        };
    }
    const tls = SideStream.tls_shared.get();
    if (tls.wave_side.ctx == null) {
        if (newStream(main)) |st| tls.wave_side = st;
    }
    return true;
}

/// One half of a wavefront chunk: its token ids (a host copy), first row's position, rows and forward generation, the
/// stream its ops are recorded on, its hyper-connection stream and pending write, its last PF_WAVE_AHEAD phases' outputs.
const PfWaveHalf = struct {
    ids: mlx.mlx_array = .{ .ctx = null },
    offset: usize,
    seq: c_int,
    generation: u64,
    stream: mlx.mlx_stream,
    later: bool,
    h: mlx.mlx_array = .{ .ctx = null },
    pending: ?HcPending = null,
    marks: [PF_WAVE_AHEAD]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn deinit(self: *PfWaveHalf) void {
        if (self.pending) |*pd| pd.deinit();
        self.pending = null;
        for (&self.marks) |*m| {
            if (m.ctx != null) _ = mlx.mlx_array_free(m.*);
            m.* = .{ .ctx = null };
        }
        inline for (.{ &self.h, &self.ids }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// Rows [r0, r0 + n) of the [1, S] host token ids as a host array of their own: no op, so no stream (the PLE layer's id
/// read would wait behind a slice recorded on a GPU stream).
fn pfWaveIds(token_ids: mlx.mlx_array, r0: usize, n: usize) !mlx.mlx_array {
    const shape = [_]c_int{ 1, @intCast(n) };
    const out = switch (mlx.mlx_array_dtype(token_ids)) {
        .int32 => mlx.mlx_array_new_data(@ptrCast((mlx.mlx_array_data_int32(token_ids) orelse return error.PfWaveIdsUnreadable) + r0), &shape, 2, .int32),
        .uint32 => mlx.mlx_array_new_data(@ptrCast((mlx.mlx_array_data_uint32(token_ids) orelse return error.PfWaveIdsUnreadable) + r0), &shape, 2, .uint32),
        else => return error.PfWaveIdsUnreadable,
    };
    if (out.ctx == null) return error.MlxError;
    return out;
}

/// Submits half `x`'s phase (`mlx_async_eval`, no wait): its stream and pending write and, on a mixer phase, every state
/// the layer's entries hold; an array left out would be dispatched by the first eval that reads it, on its own stream
/// behind every phase queued there since.
fn pfWaveSubmit(x: *const PfWaveHalf, entry: ?*const SSMCacheEntry, kv: ?*const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x.h);
    if (x.pending) |pd| {
        _ = mlx.mlx_vector_array_append_value(vec, pd.out);
        _ = mlx.mlx_vector_array_append_value(vec, pd.inj);
    }
    if (entry) |e| {
        inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled", "qsa_score_bank" }) |name| {
            if (@field(e, name).ctx != null) _ = mlx.mlx_vector_array_append_value(vec, @field(e, name));
        }
    }
    if (kv) |k| if (k.initialized) {
        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |name| {
            if (@field(k, name).ctx != null) _ = mlx.mlx_vector_array_append_value(vec, @field(k, name));
        }
    };
    try mlx.check(mlx.mlx_async_eval(vec));
}

/// Holds at most PF_WAVE_AHEAD of half `x`'s phases in flight: waits for the phase that far back, then keeps phase `p`'s
/// output in its slot (the pending write's block output where there is one).
fn pfWaveBound(x: *PfWaveHalf, p: usize) !void {
    const slot = &x.marks[p % PF_WAVE_AHEAD];
    if (slot.ctx != null) {
        try mlx.check(mlx.mlx_array_eval(slot.*));
        _ = mlx.mlx_array_free(slot.*);
        slot.* = .{ .ctx = null };
    }
    var m = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_array_set(&m, if (x.pending) |pd| pd.out else x.h));
    slot.* = m;
}

/// The forward `forwardQwen4With` builds is the plain one: on Metal and one rank, with none of the diagnostics that
/// change what a forward builds (stand-ins, the trace, the per-block profilers, a layer cap, the dispatch probe).
pub fn forwardPlain(self: *const Trunk) bool {
    if (self.qwen4 == null or !mlx.streamIsGpu(self.s) or self.dist_peer != null or self.dist_tp != null) return false;
    if (@as(u16, @bitCast(qwen4Standin())) != 0 or Trunk.qwen4_trace != null) return false;
    if (profileFwdOn() or env.on(.QWEN4_OPCOUNT) or env.on(.QWEN4_PROFILE_ATTN) or env.on(.QWEN4_PROFILE_QSA) or dispatchProbeCount() != 0) return false;
    return layerCap(self.config.num_hidden_layers) == self.config.num_hidden_layers;
}

/// The halves the wavefront runs this forward as, or null: the chain, with nothing made or touched (decode, verify and
/// narrow chunks return at the row check).
pub fn pfWavePlanFor(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) ?PfWaveHalves {
    if (chain_override or ctx.stream_in != null or token_ids.ctx == null) return null;
    const sh = mlx.getShape(token_ids);
    if (sh.len != 2 or sh[1] < 0) return null;
    const halves = pfWavePlan(@intCast(sh[1]), PF_WAVE_MIN_HALF) orelse return null;
    if (!self.reordered()) return null;
    const rows = halves.a + halves.b;
    const facts: PfWaveFacts = .{
        .batch = sh[0],
        .gpu = mlx.streamIsGpu(self.s),
        .plain = forwardPlain(self),
        .ctx_plain = pfWaveCtxPlain(ctx),
        .host_ids = pfWaveHostIds(token_ids),
        .kv_copy_bytes = pfWaveKvCopyBytes(ctx.cache, &self.config, ctx.moe_seq_offset.* + rows),
    };
    if (pfWaveRefusal(facts)) |why| return pfWaveDeclined(why, rows);
    if (!pfWaveStreams(self.s)) return pfWaveDeclined("MLX made no second GPU stream", rows);
    return halves;
}

/// `forwardQwen4With` over a prefill chunk run as two halves (`pfWavePlanFor`): each half's embedding and tile, every
/// layer's two phases for A and, one phase behind, for B on the second stream, each half's last write, the chunk's
/// stream in row order and `forwardQwen4With`'s end over it. On a failure the position is where it was and every stream
/// scope is restored.
pub fn forwardQwen4Wave(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, halves: PfWaveHalves) !mlx.mlx_array {
    const cfg = &self.config;
    const n_layers: usize = cfg.num_hidden_layers;
    const hc: c_int = @intCast(cfg.hc_count);
    const rows = halves.a + halves.b;
    const main = self.s;
    // The thread's default GPU stream as it stands: the later half's phases set their own and put this one back.
    const prev_default = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(prev_default);
    const base_offset = ctx.moe_seq_offset.*;
    errdefer ctx.moe_seq_offset.* = base_offset;
    self.spec_capture_ssm = false;
    // The PLE layer's id reads on the CPU stream: a host sync that waits behind no GPU queue.
    const ids_stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(ids_stream);
    self.ple_ids_stream = ids_stream;
    defer self.ple_ids_stream = .{ .ctx = null };
    if (!engaged) {
        engaged = true;
        log.info("[pf-wave] engaged: S={d} halves={d}+{d}, the later half one phase behind on a second GPU stream\n", .{ rows, halves.a, halves.b });
    }

    const generation = self.fwd_gen;
    var hv = [2]PfWaveHalf{
        .{ .offset = base_offset, .seq = @intCast(halves.a), .generation = generation +% 1, .stream = main, .later = false },
        .{ .offset = base_offset + halves.a, .seq = @intCast(halves.b), .generation = generation +% 2, .stream = later_shared.get().stream, .later = true },
    };
    defer for (&hv) |*x| x.deinit();
    hv[0].ids = try pfWaveIds(token_ids, 0, halves.a);
    hv[1].ids = try pfWaveIds(token_ids, halves.a, halves.b);

    for (&hv) |*x| {
        try enter(self, ctx, x);
        defer leave(self, x, main, prev_default);
        const emb = try self.embedding(x.ids);
        defer _ = mlx.mlx_array_free(emb);
        x.h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&x.h, emb, &[_]c_int{ 1, 1, hc }, 3, self.s));
        if (@import("forward.zig").streamF32(self)) {
            var h32 = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(h32);
            try mlx.check(mlx.mlx_astype(&h32, x.h, .float32, self.s));
            _ = mlx.mlx_array_free(x.h);
            x.h = h32;
        }
    }
    var dt = mlx.DtypeTrace.begin("qwen4", hv[0].h, switch (self.moe_layers[0].attn) {
        .full => |f| f.q_w,
        .linear => |la| la.qkv_w,
    });

    // Slot t builds A's phase t, then B's phase t - 1 (layer p / 2: even p its mixer side, odd p its MoE).
    const phases = 2 * n_layers;
    var t: usize = 0;
    while (t <= phases) : (t += 1) {
        if (t < phases) try phase(self, ctx, &hv[0], t, main, prev_default);
        if (t >= 1) try phase(self, ctx, &hv[1], t - 1, main, prev_default);
    }

    for (&hv) |*x| {
        try enter(self, ctx, x);
        defer leave(self, x, main, prev_default);
        try self.hcFlush(&x.h, 1, x.seq, &x.pending);
    }
    var h = try concatAxis1(main, &[_]mlx.mlx_array{ hv[0].h, hv[1].h });
    errdefer if (h.ctx != null) {
        _ = mlx.mlx_array_free(h);
    };
    ctx.moe_seq_offset.* = base_offset + rows;
    self.fwd_gen = hv[1].generation;
    pf_wave_calls +%= 1;
    dt.end(h);

    if (ctx.capture_stream_all) |target| _ = mlx.mlx_array_set(target, h);
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(h);
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, h, &[_]c_int{ 0, hs[1] - 1, 0 }, 3, &[_]c_int{ hs[0], hs[1], hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, h);
    var no_pending: ?HcPending = null;
    const mix = try self.hcReadPending(&h, &self.qwen4_mixer, 1, @intCast(rows), &no_pending);
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) return final;
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try self.lmHeadProject(final);
    _ = mlx.mlx_array_free(final);
    return logits;
}

/// Records the next ops as half `x`'s: its stream as the trunk's, its forward generation and position; for the later half
/// also as the thread's default stream (MLX replays a compiled closure on the default stream, so the hc tails, the GDN
/// gate and the routing closure run on that half's stream) and its side-stream route. `leave` restores the scope.
fn enter(self: *Trunk, ctx: *ForwardCtx, x: *const PfWaveHalf) !void {
    if (x.later) try mlx.check(mlx.mlx_set_default_stream(x.stream));
    self.s = x.stream;
    self.fwd_gen = x.generation;
    ctx.moe_seq_offset.* = x.offset;
    if (x.later) {
        const tls = SideStream.tls_shared.get();
        tls.wave_route = if (tls.wave_side.ctx != null) .own else .main;
    }
}

fn leave(self: *Trunk, x: *const PfWaveHalf, main: mlx.mlx_stream, prev_default: mlx.mlx_stream) void {
    if (x.later) {
        SideStream.tls_shared.get().wave_route = .off;
        _ = mlx.mlx_set_default_stream(prev_default);
    }
    self.s = main;
}

/// Half `x`'s phase `p`, layer p / 2: its mixer side (even p) or its MoE side (odd p), `forwardQwen4With`'s calls in its
/// order at the half's rows and position; then submitted and the half's run-ahead bounded.
fn phase(self: *Trunk, ctx: *ForwardCtx, x: *PfWaveHalf, p: usize, main: mlx.mlx_stream, prev_default: mlx.mlx_stream) !void {
    const ml = self.moe_layers;
    const n_layers: usize = self.config.num_hidden_layers;
    const li = p / 2;
    const lw = &ml[li];
    const entry = &ctx.ssm_entries.?[li];
    try enter(self, ctx, x);
    defer leave(self, x, main, prev_default);
    if (p % 2 == 0) {
        if (lw.ple) |*pw| {
            try self.hcFlush(&x.h, 1, x.seq, &x.pending);
            // no kick: every phase before this one is on the GPU already
            const h_ple = try pleForwardResidual(self, ctx, x.h, x.ids, pw, entry, li, 1, x.seq);
            _ = mlx.mlx_array_free(x.h);
            x.h = h_ple;
        }
        var pre = try self.hcReadPending(&x.h, &lw.hc_attn.?, 1, x.seq, &x.pending);
        defer pre.deinit();
        const attn_out = switch (lw.attn) {
            .linear => |la| try self.gatedDeltaNet(pre.mixed, &la, entry, li, 1, x.seq, true),
            .full => |fa| try self.qwen4AttnWith(ctx, pre.mixed, &fa, entry, @intCast(li), @intCast(x.offset), 0, 1, x.seq, true),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try self.hcWriteOrDefer(&x.h, attn_out, pre.inj, 1, x.seq, lw.hc_mlp.?.inject_flat, &x.pending);
        try pfWaveSubmit(x, entry, if (li < ctx.cache.entries.len) &ctx.cache.entries[li] else null);
    } else {
        var pre2 = try self.hcReadPending(&x.h, &lw.hc_mlp.?, 1, x.seq, &x.pending);
        defer pre2.deinit();
        const mlp_out = switch (lw.mlp) {
            .moe => |*mw| try self.moeMLP(pre2.mixed, mw),
            .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
        };
        defer _ = mlx.mlx_array_free(mlp_out);
        const next_inject: mlx.mlx_array = if (li + 1 < n_layers) ml[li + 1].hc_attn.?.inject_flat else .{ .ctx = null };
        try self.hcWriteOrDefer(&x.h, mlp_out, pre2.inj, 1, x.seq, next_inject, &x.pending);
        try pfWaveSubmit(x, null, null);
    }
    try pfWaveBound(x, p);
}

// ── tests ──

const testing = std.testing;

const fixture = @import("fixture.zig");
const driver = @import("../../engines/mlx/plugin/driver.zig");
const intree = @import("../../engines/mlx/plugin/intree.zig");
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const Qwen4TestSlot = @import("forward.zig").Qwen4TestSlot;

/// One arm of the forward test: the chunk's logits and the slot's states after it (f32 copies, an absent array empty),
/// its position and each layer's n-gram history.
const WaveTestRun = struct {
    arrays: std.ArrayList([]f32) = .empty,
    off: usize = 0,
    prev: [8][8]u32 = undefined,
    prev_valid: [8]bool = undefined,

    fn deinit(r: *WaveTestRun) void {
        for (r.arrays.items) |x| testing.allocator.free(x);
        r.arrays.deinit(testing.allocator);
    }

    fn keep(r: *WaveTestRun, x: mlx.mlx_array, s: mlx.mlx_stream) !void {
        const v = if (x.ctx == null) try testing.allocator.alloc(f32, 0) else try qwen4ReadF32(testing.allocator, x, s);
        errdefer testing.allocator.free(v);
        try r.arrays.append(testing.allocator, v);
    }
};

/// rms(got - want) / rms(want) (the plain rms against a zero `want`), and false on a value that is not finite.
fn waveTestRel(want: []const f32, got: []const f32) ?f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (want, got) |w, g| {
        if (!std.math.isFinite(w) or !std.math.isFinite(g)) return null;
        const d = @as(f64, g) - w;
        num += d * d;
        den += @as(f64, w) * w;
    }
    return if (den == 0) @sqrt(num) else @sqrt(num / den);
}
