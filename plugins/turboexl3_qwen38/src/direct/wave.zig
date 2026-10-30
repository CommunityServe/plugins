//! The wavefront prefill: a wide prefill chunk as two halves, the later one phase behind the earlier on a second GPU
//! stream, so one half's routed-expert GEMMs run beside the other's recurrence, hc norms and bandwidth-bound launches.
//! A layer is two phases, its mixer side (PLE, attention-side hc read, GDN or QSA) and its MoE side (MLP-side hc read,
//! MoE); slot t builds A's phase t, then B's phase t - 1. Layer i of B needs of A only what a chunk needs of the chunk
//! before it (the K/V and QSA rows, the GDN and PLE states, the n-gram history), all of it inside MLX's graph, so MLX
//! orders B's reads after A's writes across the two queues. Each half runs at its own row count, so GEMM tiles and
//! split-K may differ from the whole chunk's: a path of the reordered numerics only.

const mlx = @import("../mlx_serve/mlx/ffi.zig");
const concatAxis1 = @import("../mlx_serve/mlx/array.zig").concatAxis1;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCache;
const KVCacheEntry = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCacheEntry;
const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SideStream = @import("../mlx_serve/plugins/qwen4_exp/side_stream.zig").SideStream;
const newStream = @import("../mlx_serve/plugins/qwen4_exp/side_stream.zig").newStream;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const hc = @import("hc.zig");
const gdn = @import("gdn.zig");
const attn = @import("attn.zig");
const moe = @import("moe.zig");
const ple = @import("ple.zig");
const linear = @import("linear.zig");

/// The smallest half: below 1024 rows a half's fixed costs (the launches, the routed experts' part-empty M tiles)
/// spread over too few rows.
const MIN_HALF: usize = 1024;
/// The earlier half ends on a multiple of this: whole NAX M tiles and whole QSA blocks, so the later half starts on a
/// block boundary (no leftover keys to re-pool).
const ALIGN: usize = 64;
/// A half's phases in flight past the one the host waits on: the transients of two chunk-wide layers, under the chain's
/// four between cadence evals.
const AHEAD: usize = 4;
/// The widest attention layer's K + V the later half's write may copy while the earlier half still reads it (a
/// 256K-token buffer at 2 KiB a token and layer); past it the chunk runs as one.
const KV_COPY_MAX: usize = 512 << 20;

/// A chunk's halves: A is its first `a` rows, B the `b` after them.
pub const Halves = struct { a: usize, b: usize };


/// The halves this forward runs as, or null: the chain, with nothing made or touched (decode, verify and narrow chunks
/// return at the row check).
pub fn planFor(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) ?Halves {
    if (ctx.stream_in != null or token_ids.ctx == null) return null;
    const sh = mlx.getShape(token_ids);
    if (sh.len != 2 or sh[1] < 0) return null;
    const rows: usize = @intCast(sh[1]);
    const a = rows / 2 / ALIGN * ALIGN;
    if (a == 0 or a < MIN_HALF) return null;
    if (!self.reordered()) return null;
    // one prefill chunk of one slot, on the GPU, its ids on the host, its K/V buffers small enough to copy, two streams
    if (sh[0] != 1 or !mlx.streamIsGpu(self.s) or self.qwen4 == null or !ctxPlain(ctx) or !hostIds(token_ids) or
        kvCopyBytes(ctx.cache, self, ctx.moe_seq_offset.* + rows) > KV_COPY_MAX or !streams(self.s)) return null;
    return .{ .a = a, .b = rows - a };
}

/// The ctx carries a prefill chunk's fields alone: no batched slots, rank hand-off, vision rows or M-RoPE table, layer
/// captures, spec capture or verify, tree, mask term, deferred PLE or its id sources, build ladder or head entry.
fn ctxPlain(ctx: *const ForwardCtx) bool {
    return ctx.ssm_entries != null and ctx.batch_slots == null and ctx.stream_in == null and ctx.vision_embeddings == null and
        ctx.mrope_pos == null and ctx.mrope_cos_cur == null and ctx.capture_layers == null and !ctx.capture_ssm_seq and
        ctx.prefill_mask_add == null and !ctx.spec_verify and ctx.tree == null and !ctx.ple_defer and
        ctx.ple_pending == null and ctx.ple_ids_src == null and ctx.pipeline_build == 0 and ctx.qsa_entry == null and
        ctx.batch_rope_offsets == null and ctx.key_pad_mask == null;
}

/// The token ids are [1, S] int32 / uint32 host data, row-contiguous and available, so `ids` can copy each half's.
fn hostIds(token_ids: mlx.mlx_array) bool {
    const dt = mlx.mlx_array_dtype(token_ids);
    if (dt != .int32 and dt != .uint32) return false;
    if (mlx.mlx_array_ndim(token_ids) != 2 or mlx.mlx_array_strides(token_ids)[1] != 1) return false;
    var ready = false;
    if (mlx._mlx_array_is_available(&ready, token_ids) != 0) return false;
    return ready;
}

/// The K + V bytes of the widest attention layer's buffer once this chunk is written (dense bf16 at the config's K/V
/// heads, at least `kv_end` rows): what the later half's K/V write copies at most.
fn kvCopyBytes(cache: *const KVCache, self: *const Trunk, kv_end: usize) usize {
    var rows: usize = kv_end;
    for (cache.entries) |*e| {
        if (e.initialized) rows = @max(rows, KVCache.bufferCapacity(e.keys));
    }
    return rows * 2 * @as(usize, self.config.num_key_value_heads) * @as(usize, self.config.head_dim) * 2;
}

/// The later half's stream on this inference thread (MLX registers a stream's command encoders on the thread that
/// makes it), made at the first chunk that engages, with its side stream (`SideStream.Tls.wave_side`); `failed`: MLX
/// made none on this thread, so its chunks run as one.
const Later = struct { stream: mlx.mlx_stream = .{ .ctx = null }, failed: bool = false };
const later_shared = @import("../mlx_serve/sdk/util/shared.zig").SharedThreadLocal(Later, "qwen4_wave_stream", .{});

/// The wavefront's streams beside `main`: false when MLX makes no second stream. Without a side stream of its own the
/// later half's side-stream parts stay on its stream.
fn streams(main: mlx.mlx_stream) bool {
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

/// One half: its token ids (a host copy), first row's position, rows and forward generation, the stream its ops are
/// recorded on, its hyper-connection stream and pending write, its last AHEAD phases' outputs.
const Half = struct {
    ids: mlx.mlx_array = .{ .ctx = null },
    offset: usize,
    seq: c_int,
    generation: u64,
    stream: mlx.mlx_stream,
    later: bool,
    h: mlx.mlx_array = .{ .ctx = null },
    pending: ?hc.HcPending = null,
    marks: [AHEAD]mlx.mlx_array = @splat(.{ .ctx = null }),

    fn deinit(self: *Half) void {
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
fn ids(token_ids: mlx.mlx_array, r0: usize, n: usize) !mlx.mlx_array {
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
fn submit(x: *const Half, entry: ?*const SSMCacheEntry, kv: ?*const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x.h);
    if (x.pending) |pd| {
        _ = mlx.mlx_vector_array_append_value(vec, pd.out);
        _ = mlx.mlx_vector_array_append_value(vec, pd.inj);
    }
    if (entry) |e| {
        inline for (.{ "conv_state", "ssm_state", "aux_state", "qsa_pooled" }) |name| {
            if (@field(e, name).ctx != null) _ = mlx.mlx_vector_array_append_value(vec, @field(e, name));
        }
    }
    if (kv) |k| if (k.initialized) {
        inline for (.{ "keys", "values" }) |name| {
            if (@field(k, name).ctx != null) _ = mlx.mlx_vector_array_append_value(vec, @field(k, name));
        }
    };
    try mlx.check(mlx.mlx_async_eval(vec));
}

/// Holds at most AHEAD of half `x`'s phases in flight: waits for the phase that far back, then keeps phase `p`'s output
/// in its slot (the pending write's block output where there is one).
fn bound(x: *Half, p: usize) !void {
    const slot = &x.marks[p % AHEAD];
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

/// The model's forward over a prefill chunk run as two halves (`planFor`): each half's embedding and tile, every layer's
/// two phases for A and, one phase behind, for B on the second stream, each half's last write, the chunk's stream in row
/// order and the forward's end over it. On a failure the position is where it was and every stream scope is restored.
pub fn forward(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array, halves: Halves) !mlx.mlx_array {
    const n_layers: usize = self.config.num_hidden_layers;
    const hc_count: c_int = @intCast(self.config.hc_count);
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

    const generation = self.fwd_gen;
    var hv = [2]Half{
        .{ .offset = base_offset, .seq = @intCast(halves.a), .generation = generation +% 1, .stream = main, .later = false },
        .{ .offset = base_offset + halves.a, .seq = @intCast(halves.b), .generation = generation +% 2, .stream = later_shared.get().stream, .later = true },
    };
    defer for (&hv) |*x| x.deinit();
    hv[0].ids = try ids(token_ids, 0, halves.a);
    hv[1].ids = try ids(token_ids, halves.a, halves.b);

    for (&hv) |*x| {
        try enter(self, ctx, x);
        defer leave(self, x, main, prev_default);
        const emb = try linear.embedding(self, x.ids);
        defer _ = mlx.mlx_array_free(emb);
        x.h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&x.h, emb, &[_]c_int{ 1, 1, hc_count }, 3, self.s));
    }

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
        try hc.flush(self, &x.h, 1, x.seq, &x.pending);
    }
    var h = try concatAxis1(main, &[_]mlx.mlx_array{ hv[0].h, hv[1].h });
    errdefer if (h.ctx != null) {
        _ = mlx.mlx_array_free(h);
    };
    ctx.moe_seq_offset.* = base_offset + rows;
    self.fwd_gen = hv[1].generation;

    if (ctx.capture_stream_all) |target| _ = mlx.mlx_array_set(target, h);
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(h);
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, h, &[_]c_int{ 0, hs[1] - 1, 0 }, 3, &[_]c_int{ hs[0], hs[1], hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, h);
    var no_pending: ?hc.HcPending = null;
    const mix = try hc.read(self, &h, &self.qwen4_mixer, 1, @intCast(rows), &no_pending);
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) return final;
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try linear.lmHeadProject(self, final);
    _ = mlx.mlx_array_free(final);
    return logits;
}

/// Records the next ops as half `x`'s: its stream as the trunk's, its forward generation and position; for the later half
/// also as the thread's default stream (MLX replays a compiled closure on the default stream, so the hc tails, the GDN
/// gate and the routing closure run on that half's stream) and its side-stream route. `leave` restores the scope.
fn enter(self: *Trunk, ctx: *ForwardCtx, x: *const Half) !void {
    if (x.later) try mlx.check(mlx.mlx_set_default_stream(x.stream));
    self.s = x.stream;
    self.fwd_gen = x.generation;
    ctx.moe_seq_offset.* = x.offset;
    if (x.later) {
        const tls = SideStream.tls_shared.get();
        tls.wave_route = if (tls.wave_side.ctx != null) .own else .main;
    }
}

fn leave(self: *Trunk, x: *const Half, main: mlx.mlx_stream, prev_default: mlx.mlx_stream) void {
    if (x.later) {
        SideStream.tls_shared.get().wave_route = .off;
        _ = mlx.mlx_set_default_stream(prev_default);
    }
    self.s = main;
}

/// Half `x`'s phase `p`, layer p / 2: its mixer side (even p) or its MoE side (odd p), the forward's calls in its order
/// at the half's rows and position; then submitted and the half's run-ahead bounded.
fn phase(self: *Trunk, ctx: *ForwardCtx, x: *Half, p: usize, main: mlx.mlx_stream, prev_default: mlx.mlx_stream) !void {
    const ml = self.moe_layers;
    const n_layers: usize = self.config.num_hidden_layers;
    const li = p / 2;
    const lw = &ml[li];
    const entry = &ctx.ssm_entries.?[li];
    try enter(self, ctx, x);
    defer leave(self, x, main, prev_default);
    if (p % 2 == 0) {
        if (lw.ple) |*pw| {
            try hc.flush(self, &x.h, 1, x.seq, &x.pending);
            // no kick: every phase before this one is on the GPU already
            const h_ple = try ple.forwardResidual(self, ctx, x.h, x.ids, pw, entry, li, 1, x.seq);
            _ = mlx.mlx_array_free(x.h);
            x.h = h_ple;
        }
        var pre = try hc.read(self, &x.h, &lw.hc_attn.?, 1, x.seq, &x.pending);
        defer pre.deinit();
        const attn_out = switch (lw.attn) {
            .linear => |la| try gdn.layer(self, pre.mixed, &la, entry, li, 1, x.seq, true),
            .full => |fa| try attn.layer(self, ctx, pre.mixed, &fa, entry, @intCast(li), @intCast(x.offset), 0, 1, x.seq, true),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try hc.writeOrDefer(self, &x.h, attn_out, pre.inj, 1, x.seq, lw.hc_mlp.?.inject_flat, &x.pending);
        try submit(x, entry, if (li < ctx.cache.entries.len) &ctx.cache.entries[li] else null);
    } else {
        var pre2 = try hc.read(self, &x.h, &lw.hc_mlp.?, 1, x.seq, &x.pending);
        defer pre2.deinit();
        const mw = switch (lw.mlp) {
            .moe => |*w| w,
            .dense => return error.DirectModelHasNoDenseMlp,
        };
        const mlp_out = try moe.mlp(self, pre2.mixed, mw);
        defer _ = mlx.mlx_array_free(mlp_out);
        const next_inject: mlx.mlx_array = if (li + 1 < n_layers) ml[li + 1].hc_attn.?.inject_flat else .{ .ctx = null };
        try hc.writeOrDefer(self, &x.h, mlp_out, pre2.inj, 1, x.seq, next_inject, &x.pending);
        try submit(x, null, null);
    }
    try bound(x, p);
}
