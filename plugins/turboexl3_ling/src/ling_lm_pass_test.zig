//! C4's byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips; MLX_SERVE_LING_PACK names the pack it
//! loads): the layer-major pass in segments (src/ling_lm_pass.zig `runWide` at a segment bound) gives, word for word,
//! what the single pass gives (one segment), where both fit; and the MTP head's history in forwards of 4,096 rows
//! (src/ling_mtp.zig `history`) leaves the head's cache rows one forward leaves. It is in the restore check's binary
//! (build.zig `restoreTest`): the GPU binary that runs the copied trunk's model.
//! - The model: the pack's first 8 layers through the copied trunk.zig `load`, the pack's config cut to 8 layers (KDA
//!   layers 0-6, the dense MLP at 0-3 and the MoE at 4-6; the MLA layer 7 with the MoE), its final norm and head, the
//!   embedding's rows read from the pack (EMBED_ROWS, src/ling_embed.zig). The MTP head: the pack's NextN layer
//!   (src/ling_mtp.zig `load`), its 512 routed experts resident. Those experts are also each MoE layer's routed source
//!   here (a resident Routed hook: the streamed bank is not in this binary). The layers load as the plugin's parse
//!   serves them, streamed (config.zig `served`), so the loader leaves their experts to that hook, and each routed call
//!   runs turboexl3's routed math on the pack's records. turboexl3 is configured as served (src/ling_math.zig
//!   `configure`).
//! - The prompt: 52,000 tokens (fixed ids) at the host's spans (src/ling_rq.zig `spanEnds`: twelve of 4,096 rows, 2,817
//!   and 31), each wider than a decode lane, with a checkpoint at every span end but the last.
//! - The caches, one set an arm: the trunk's (the MLA layer's KV as mlx-serve 66aad949's dense cache grows it for a
//!   fresh request, as src/ling_d2_test.zig's `Host`; each KDA layer's conv and ssm state) and the head's (its latent).
//! - Arm A: `runWide` with no bound: one segment, the single pass. Arm B: 16,384 rows a segment: three segments (16,384,
//!   16,384 and 19,232 rows), so layer 7's D2 group is cut at 32,768, and the last call (2,848 rows) joins the third by
//!   the segment tail rule (src/ling_segments.zig `segment_tail_rows`). Each arm's hooks do what the plugin's do with
//!   the rows: its checkpoints filed in order, its hidden rows written through `HiddenRows`, evaluated after each
//!   segment.
//! - Word for word, B against A: each span's final hidden, the joined hidden (and A's against its spans' rows joined),
//!   layer 7's K and V (rows [0, 52,000)), every KDA layer's conv and ssm at every checkpoint (the same 13 positions in
//!   order). Then the MTP history, 51,999 rows (each arm's hidden rows with the next tokens): A in one head forward, B
//!   in the forwards the served rule gives it (src/ling_segments.zig `historyRows`: 4,096 rows): what serving reads of
//!   it, the head's cache rows (its latent cache's K and V, every row), and RQ's hook as each calls it (A once with
//!   51,999 rows; B 13 times, 4,096 rows each but 2,847 last). The run-path marks LM_SEGMENTS and MTP_HISTORY_CHUNKS:
//!   set by B, not by A.
//! - The head's last-row output of the history is counted, not checked (the lead's ruling, 2026-11-11): its differing
//!   words go on a `TURBOEXL3_LING_LM_PASS_INFO` line. Serving never evaluates a history forward's output, so only the
//!   cache rows reach a draft. mlx-serve 9a173773 appends a prompt's history through the plugin head's `appendRows`
//!   (src/engines/mlx/plugin/head.zig), which runs the head with `logits = .none` and `want_hidden = false` and frees
//!   what `run` returns unevaluated; `run` frees `out.logits` unevaluated. The plugin's `headForward` (src/ling_mtp.zig)
//!   frees the normed output unevaluated when `want_hidden` is false, and `history` drops each earlier forward's output
//!   unevaluated, evaluating only the cache's view. B's head forwards run their attention and routed calls at other row
//!   counts than A's one forward (4,096 rows against 51,999), so those output words may differ; the cache rows come from
//!   each row alone (kv_a after eh_proj and the norms, the roped key).
//! - Plants, which show the test can fail: arm B again with token 49,500 changed (in span 12, B's third segment): spans
//!   0-11 keep their words and spans 12 and 13 do not, nor layer 7's rows past it; one flipped bit of B's joined hidden is
//!   one differing word; B's checkpoints against A's one position later differ.
//! Each check prints a `TURBOEXL3_LING_LM_PASS` line ending in PASS or FAIL; the count, one
//! `TURBOEXL3_LING_LM_PASS_INFO` line. Nothing is timed.
const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const sdk = kit.sdk;
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const family = @import("mlx_serve/plugins/glm5_next/config.zig");
const ling_config = @import("mlx_serve/plugins/bailing_hybrid/config.zig");
const ling_weights = @import("ling_weights.zig");
const ling_embed = @import("ling_embed.zig");
const ling_mtp = @import("ling_mtp.zig");
const ling_math = @import("ling_math.zig");
const ling_rq = @import("ling_rq.zig");
const lm_pass = @import("ling_lm_pass.zig");
const ling_segments = @import("ling_segments.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const gpa = std.heap.c_allocator;
const tag = "TURBOEXL3_LING_LM_PASS";
const zero4 = [_]c_int{ 0, 0, 0, 0 };
const st4 = [_]c_int{ 1, 1, 1, 1 };

/// The prompt's tokens, the cut model's layers, its MLA layer, arm B's segment bound and the plant's changed token
const n_tokens: u64 = 52_000;
const n_layers: u32 = 8;
const mla_layer: u32 = 7;
const b_rows: u64 = 16384;
const plant_at: u64 = 49_500;
/// What the plan gives at 52,000 tokens: the spans, the segments of each arm, the history's forwards in B
const want_spans = [_]u32{ 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 2817, 31 };
const want_b_segments: usize = 3;
const want_b_history = [_]u64{ 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 2847 };

/// The GPU tests' gate: MLX_SERVE_LING_GPU_TESTS=1 and a GPU stream, else the test skips
fn gpu() !Stream {
    const v = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// A reference of its own on `a` (null stays null)
fn takeRef(a: Array) Array {
    var r = mlx.mlx_array_new();
    if (a.ctx != null) _ = mlx.mlx_array_set(&r, a);
    return r;
}

/// Token `i` of the prompt
fn tokenAt(i: u64) i32 {
    return @intCast((i * 7919 + 101) % 150_000 + 100);
}

/// The prompt's tokens `[1, n]` int32 from token `first`, token `plant_at` replaced by another where `plant` is set
fn tokens(first: u64, n: u64, plant: bool) !Array {
    const buf = try gpa.alloc(i32, n);
    defer gpa.free(buf);
    for (buf, 0..) |*t, k| {
        const i: u64 = first + k;
        t.* = if (plant and i == plant_at) tokenAt(i + 1) else tokenAt(i);
    }
    const a = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(n) }, 2, .int32);
    if (a.ctx == null) return error.MlxError;
    return a;
}

/// Words of `a` (0 for a null array)
fn sizeOf(a: Array) usize {
    return if (a.ctx == null) 0 else mlx.mlx_array_size(a);
}

/// The words in which `a` and `b` differ (16- or 32-bit), counted on the GPU as integers: the words less the equal ones;
/// a shape, dtype or width mismatch counts every word, and so does one array null where the other is not
fn differing(a: Array, b: Array, s: Stream) !usize {
    if (a.ctx == null or b.ctx == null) return @max(sizeOf(a), sizeOf(b));
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return n;
    const w: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(b)) {
        2 => .uint16,
        4 => .uint32,
        else => return n,
    };
    var ua = mlx.mlx_array_new();
    defer free(ua);
    try mlx.check(mlx.mlx_view(&ua, a, w, s));
    var ub = mlx.mlx_array_new();
    defer free(ub);
    try mlx.check(mlx.mlx_view(&ub, b, w, s));
    var eq = mlx.mlx_array_new();
    defer free(eq);
    try mlx.check(mlx.mlx_equal(&eq, ua, ub, s));
    var sum = mlx.mlx_array_new();
    defer free(sum);
    try mlx.check(mlx.mlx_sum(&sum, eq, false, s));
    var cnt = mlx.mlx_array_new();
    defer free(cnt);
    try mlx.check(mlx.mlx_astype(&cnt, sum, .int32, s));
    try mlx.check(mlx.mlx_array_eval(cnt));
    var same: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&same, cnt));
    return n - @as(usize, @intCast(same));
}

// ── the caches ──

/// mlx-serve 66aad949's capacity schedule (src/sdk/kv/cache.zig `KVCache.nextCapacity`)
fn nextCapacity(cur: usize, needed: usize) usize {
    if (needed <= cur) return cur;
    const step = @min(@max(cur / 4, 256), 8192);
    const target = @max(needed, cur + step);
    return (target + 255) / 256 * 256;
}

/// One slot's caches for the cut model: each attention layer's dense KV as mlx-serve 66aad949's cache takes a fresh
/// request's appends (src/sdk/kv/cache.zig `updateDense` with no reservation, ring or restored share: `nextCapacity`,
/// zeros with the kept rows copied in, the rows written at the offset, the view of rows [0, offset), the buffer itself
/// at its capacity: src/ling_d2_test.zig's `Host`), and each KDA layer's conv and ssm state (references of its own).
const Cache = struct {
    s: Stream,
    kv: [n_layers]Kv = @splat(.{}),
    conv: [n_layers]Array = @splat(.{ .ctx = null }),
    ssm: [n_layers]Array = @splat(.{ .ctx = null }),

    const Kv = struct {
        k: Array = .{ .ctx = null },
        v: Array = .{ .ctx = null },
        k_view: Array = .{ .ctx = null },
        v_view: Array = .{ .ctx = null },
        offset: c_int = 0,
        appends: usize = 0,

        fn cap(kv: *const Kv) c_int {
            return if (kv.k.ctx == null) 0 else mlx.getShape(kv.k)[2];
        }
    };

    fn deinit(c: *Cache) void {
        for (&c.kv) |*kv| inline for (.{ kv.k, kv.v, kv.k_view, kv.v_view }) |a| free(a);
        for (c.conv, c.ssm) |a, b| {
            free(a);
            free(b);
        }
    }

    /// `buf` at `to` rows of capacity, shaped as `like` on the other axes: zeros, then its rows [0, offset) copied in
    fn grow(c: *Cache, kv: *Kv, buf: *Array, to: c_int, like: Array) !void {
        const ls = mlx.getShape(like);
        const shape = [_]c_int{ ls[0], ls[1], to, ls[3] };
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &shape, 4, mlx.mlx_array_dtype(like), c.s));
        if (buf.ctx != null and kv.offset > 0) {
            defer free(z);
            const kept = [_]c_int{ ls[0], ls[1], kv.offset, ls[3] };
            var old = mlx.mlx_array_new();
            defer free(old);
            try mlx.check(mlx.mlx_slice(&old, buf.*, &zero4, 4, &kept, 4, &st4, 4, c.s));
            var u = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&u, z, old, &zero4, 4, &kept, 4, &st4, 4, c.s));
            free(buf.*);
            buf.* = u;
            return;
        }
        free(buf.*);
        buf.* = z;
    }

    /// `buf` with `rows` written at the offset
    fn write(c: *Cache, kv: *const Kv, buf: *Array, rows: Array) !void {
        const bs = mlx.getShape(buf.*);
        const n = mlx.getShape(rows)[2];
        const stop = [_]c_int{ bs[0], bs[1], kv.offset + n, bs[3] };
        var u = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&u, buf.*, rows, &[_]c_int{ 0, 0, kv.offset, 0 }, 4, &stop, 4, &st4, 4, c.s));
        free(buf.*);
        buf.* = u;
    }

    /// `dst` = rows [0, offset) of `buf`: the buffer itself at its capacity, else a slice
    fn viewOf(c: *Cache, kv: *const Kv, dst: *Array, buf: Array) !void {
        const bs = mlx.getShape(buf);
        dst.* = mlx.mlx_array_new();
        if (kv.offset == bs[2]) {
            try mlx.check(mlx.mlx_array_set(dst, buf));
        } else {
            try mlx.check(mlx.mlx_slice(dst, buf, &zero4, 4, &[_]c_int{ bs[0], bs[1], kv.offset, bs[3] }, 4, &st4, 4, c.s));
        }
    }

    fn append(c: *Cache, layer: u32, k: Array, v: Array) !void {
        const kv = &c.kv[layer];
        free(kv.k_view);
        kv.k_view = .{ .ctx = null };
        free(kv.v_view);
        kv.v_view = .{ .ctx = null };
        const n = mlx.getShape(k)[2];
        if (kv.k.ctx == null or kv.offset + n > kv.cap()) {
            const to: c_int = @intCast(nextCapacity(@intCast(kv.cap()), @intCast(kv.offset + n)));
            try c.grow(kv, &kv.k, to, k);
            try c.grow(kv, &kv.v, to, v);
        }
        try c.write(kv, &kv.k, k);
        try c.write(kv, &kv.v, v);
        kv.offset += n;
        kv.appends += 1;
        try c.viewOf(kv, &kv.k_view, kv.k);
        try c.viewOf(kv, &kv.v_view, kv.v);
    }

    fn handle(c: *Cache) abi.Handle {
        return @ptrCast(c);
    }

    fn of(h: abi.Handle) *Cache {
        return @ptrCast(@alignCast(h.?));
    }

    const ops: abi.CacheOps = .{
        .len = len,
        .append = appendOp,
        .append_ring = appendRing,
        .view = view,
        .release = release,
        .append_rows = appendRows,
        .state = state,
        .set_state = setState,
        .state_rows = stateRows,
        .set_state_rows = setStateRows,
    };

    fn len(h: abi.Handle, l: u32) callconv(.c) u64 {
        if (l >= n_layers) return 0;
        return @intCast(of(h).kv[l].offset);
    }

    fn appendOp(h: abi.Handle, l: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        if (l >= n_layers or keep != 0) return abi.Verdict.refuse(.unsupported, "the C4 test's cache holds each layer's every row");
        const c = of(h);
        c.append(l, .{ .ctx = k.ctx }, .{ .ctx = v.ctx }) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        out.* = .{ .k = .{ .ctx = c.kv[l].k_view.ctx }, .v = .{ .ctx = c.kv[l].v_view.ctx } };
        return .ok;
    }

    fn appendRing(h: abi.Handle, l: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, ring: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = .{ h, l, k, v, keep, window, ring, s, out };
        return abi.Verdict.refuse(.unsupported, "the C4 test's cache has no ring");
    }

    fn view(h: abi.Handle, l: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        if (l >= n_layers) return abi.Verdict.refuse(.unsupported, "the C4 test's cache has no such layer");
        const c = of(h);
        out.* = .{ .k = .{ .ctx = c.kv[l].k_view.ctx }, .v = .{ .ctx = c.kv[l].v_view.ctx } };
        return .ok;
    }

    fn release(v: *abi.KvView) callconv(.c) void {
        v.* = .{};
    }

    fn appendRows(caches: [*]const abi.Handle, n: u32, l: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, l, k, v, keep, s };
        return abi.Verdict.refuse(.unsupported, "the C4 test's cache is one slot");
    }

    fn state(h: abi.Handle, l: u32, which: abi.StateSlot) callconv(.c) abi.Array {
        if (l >= n_layers) return .{};
        const c = of(h);
        return switch (which) {
            .conv => .{ .ctx = c.conv[l].ctx },
            .ssm => .{ .ctx = c.ssm[l].ctx },
            else => .{},
        };
    }

    fn setState(h: abi.Handle, l: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
        if (l >= n_layers) return abi.Verdict.refuse(.unsupported, "the C4 test's cache has no such layer");
        const c = of(h);
        const slot = switch (which) {
            .conv => &c.conv[l],
            .ssm => &c.ssm[l],
            else => return abi.Verdict.refuse(.unsupported, "the C4 test's cache holds the conv and ssm states only"),
        };
        free(slot.*);
        slot.* = takeRef(.{ .ctx = value.ctx });
        return .ok;
    }

    fn stateRows(caches: [*]const abi.Handle, n: u32, l: u32, which: abi.StateSlot, s: abi.Stream) callconv(.c) abi.Array {
        _ = .{ caches, n, l, which, s };
        return .{};
    }

    fn setStateRows(caches: [*]const abi.Handle, n: u32, l: u32, which: abi.StateSlot, value: abi.Array, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, l, which, value, s };
        return abi.Verdict.refuse(.unsupported, "the C4 test's cache is one slot");
    }
};

// ── an arm ──

/// A checkpoint as an arm's hook took it: its position, every KDA layer's conv and ssm (references of its own; null at
/// the MLA layer)
const Cp = struct {
    pos: u64,
    conv: [n_layers]Array = @splat(.{ .ctx = null }),
    ssm: [n_layers]Array = @splat(.{ .ctx = null }),
};

/// What an arm wrote, each a reference of its own
const Rec = struct {
    outs: std.ArrayList(Array) = .empty,
    cps: std.ArrayList(Cp) = .empty,
    joined: Array = .{ .ctx = null },
    k7: Array = .{ .ctx = null },
    v7: Array = .{ .ctx = null },
    appends7: usize = 0,
    cap7: c_int = 0,
    segments: usize = 0,
    lm_mark: bool = false,
    head_out: Array = .{ .ctx = null },
    head_k: Array = .{ .ctx = null },
    head_v: Array = .{ .ctx = null },
    head_calls: std.ArrayList([2]u64) = .empty,
    hist_mark: bool = false,

    fn deinit(r: *Rec) void {
        for (r.outs.items) |a| free(a);
        r.outs.deinit(gpa);
        for (r.cps.items) |cp| for (cp.conv, cp.ssm) |a, b| {
            free(a);
            free(b);
        };
        r.cps.deinit(gpa);
        inline for (.{ r.joined, r.k7, r.v7, r.head_out, r.head_k, r.head_v }) |a| free(a);
        r.head_calls.deinit(gpa);
    }
};

/// An arm's hooks for `runWide`: as the plugin's (plugin.zig `WideRun`) with the rows, none of the bank's steps: each
/// span's final hidden kept, the hidden rows written through `HiddenRows` and evaluated after each segment, each
/// checkpoint taken in order
const ArmHooks = struct {
    rec: *Rec,
    hidden: *lm_pass.HiddenRows,

    fn hooks(a: *ArmHooks) lm_pass.Hooks {
        return .{ .ctx = a, .begin = &begin, .after = &after, .file = &file };
    }

    fn of(ctx: *anyopaque) *ArmHooks {
        return @ptrCast(@alignCast(ctx));
    }

    fn begin(ctx: *anyopaque, first: bool) anyerror!void {
        _ = .{ ctx, first };
    }

    fn after(ctx: *anyopaque, j: usize, row: u64, outs: []const Array) anyerror!void {
        const a = of(ctx);
        _ = j;
        var r = row;
        for (outs) |hd| {
            try a.rec.outs.append(gpa, takeRef(hd));
            try a.hidden.put(r, hd);
            r += @intCast(mlx.getShape(hd)[1]);
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, a.hidden.dest);
        try mlx.check(mlx.mlx_eval(vec));
    }

    fn file(ctx: *anyopaque, pos: u64, states: []const abi.LayerState) anyerror!void {
        const a = of(ctx);
        var cp: Cp = .{ .pos = pos };
        for (states) |st| {
            if (st.layer >= n_layers) return error.LmPassTestLayer;
            cp.conv[st.layer] = takeRef(.{ .ctx = st.conv.ctx });
            cp.ssm[st.layer] = takeRef(.{ .ctx = st.ssm.ctx });
        }
        try a.rec.cps.append(gpa, cp);
    }
};

/// One arm of the pass over the prompt (`plant`: token `plant_at` changed): `runWide` at `max_rows` rows a segment on a
/// fresh cache, from position 0, its records into `rec`
fn runArm(s: Stream, m: *trunk.Model, rows: []const u32, checkpoints: []const u64, max_rows: u64, plant: bool, rec: *Rec) !void {
    defer _ = mlx.mlx_clear_cache();
    const ids = try tokens(0, n_tokens, plant);
    defer free(ids);
    var cache: Cache = .{ .s = s };
    defer cache.deinit();
    const hs = [_]abi.Handle{cache.handle()};
    var hidden: lm_pass.HiddenRows = .{ .s = s, .rows = n_tokens };
    defer hidden.deinit();
    var ah: ArmHooks = .{ .rec = rec, .hidden = &hidden };
    ran.clearForTests();
    rec.segments = try lm_pass.runWide(m, &Cache.ops, &hs, ids, rows, 0, checkpoints, ling_rq.d2Start(0), max_rows, ah.hooks());
    rec.lm_mark = ran.has(.LM_SEGMENTS);
    rec.joined = (try hidden.take()) orelse return error.LmPassTestNoHidden;
    try mlx.check(mlx.mlx_array_eval(rec.joined));
    const kv = &cache.kv[mla_layer];
    rec.k7 = takeRef(kv.k_view);
    rec.v7 = takeRef(kv.v_view);
    rec.appends7 = kv.appends;
    rec.cap7 = kv.cap();
}

/// RQ's hook as the head's history calls it: each call's (length, rows) into the current arm's record
var head_calls: ?*std.ArrayList([2]u64) = null;

fn headHook(mp: *anyopaque, len: u64, n_rows: u64) anyerror!void {
    _ = mp;
    const calls = head_calls orelse return error.LmPassTestNoHook;
    try calls.append(gpa, .{ len, n_rows });
}

/// The MTP head's history after an arm's pass: the arm's hidden rows but the last with tokens 1 to n - 1, on a fresh
/// head cache, in forwards of at most `chunk` rows (src/ling_mtp.zig `history`): its output, evaluated, and its cache
fn runHistory(s: Stream, m: *trunk.Model, chunk: u64, rec: *Rec) !void {
    defer _ = mlx.mlx_clear_cache();
    const rows: u64 = n_tokens - 1;
    const ids = try tokens(1, rows, false);
    defer free(ids);
    const jsh = mlx.getShape(rec.joined);
    var hid = mlx.mlx_array_new();
    defer free(hid);
    try mlx.check(mlx.mlx_slice(&hid, rec.joined, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(rows), jsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var cache: Cache = .{ .s = s };
    defer cache.deinit();
    const hs = [_]abi.Handle{cache.handle()};
    const pos = [_]u32{0};
    const f: abi.Forward = .{
        .mode = .prefill,
        .logits = .none,
        .want_hidden = true,
        .rows = @intCast(rows),
        .tokens = .{},
        .ids = .{ .ctx = ids.ctx },
        .positions = .of(&pos),
        .caches = .of(&hs),
        .cache_ops = &Cache.ops,
        .hidden_in = .{ .ctx = hid.ctx },
    };
    head_calls = &rec.head_calls;
    defer head_calls = null;
    ran.clearForTests();
    rec.head_out = try ling_mtp.history(@ptrCast(m), m, &m.ling_mtp.?, &Cache.ops, &f, &headHook, chunk);
    rec.hist_mark = ran.has(.MTP_HISTORY_CHUNKS);
    rec.head_k = takeRef(cache.kv[0].k_view);
    rec.head_v = takeRef(cache.kv[0].v_view);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ rec.head_out, rec.head_k, rec.head_v }) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    try mlx.check(mlx.mlx_eval(vec));
}

// ── the model ──

/// The cut model and what it holds: the pack's shards, its config (whole and cut to `n_layers`), the trunk
const Model = struct {
    w: *ling_weights.Weights,
    parsed: abi.Parsed,
    cut: family.Config,
    cut_parsed: abi.Parsed,
    m: *trunk.Model,

    fn deinit(md: *Model) void {
        trunk.Model.deinit(md.m);
        gpa.destroy(md.m);
        ling_embed.close();
        md.w.deinit();
        ling_config.freeParsed(&md.parsed);
    }
};

/// The pack's first `n_layers` layers through the copied loader (the config cut to them), EMBED_ROWS' table opened for
/// row reads first, as plugin.zig `load` opens it; the MTP head (the pack's NextN layer) loaded at the whole config's
/// layer count; every MoE layer's routed source the head's resident experts
fn loadModel(s: Stream, pack: []const u8, md: *Model) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{pack});
    defer gpa.free(cfg_path);
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(json);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    md.parsed = .{};
    const verdict = ling_config.parse(&dir, abi.Str.of(json), .{}, null, true, &md.parsed);
    if (!verdict.isOk()) {
        std.debug.print("{s} FAIL: the parse refused {s}: {s}\n", .{ tag, cfg_path, verdict.why.slice() });
        return error.ConfigRefused;
    }
    errdefer ling_config.freeParsed(&md.parsed);
    // Serving's fact from the plugin's own parse (config.zig `served`): the routed experts stream from the bank, so the
    // loader leaves them to the source bound after the load (here the head's resident experts, below). This built-in
    // parse takes it from the pack's `expert_streaming` block, which this pack does not carry; without it the loader asks
    // the shards for the first MoE layer's experts (c4.FAIL 2026-11-11 19:20:50Z: MissingWeight in trunk.zig `routed`).
    @constCast(family.of(&md.parsed)).streamed = true;
    const full = family.of(&md.parsed);
    md.cut = full.*;
    md.cut.num_hidden_layers = n_layers;
    md.cut_parsed = md.parsed;
    md.cut_parsed.private = &md.cut;
    md.w = try ling_weights.Weights.open(gpa, io, pack, "\x00", s);
    errdefer md.w.deinit();
    // EMBED_ROWS: the table opened before the trunk loads (plugin.zig `openEmbedRows`)
    var kb: [160]u8 = undefined;
    const key = for ([_][]const u8{ full.weight_prefix, "language_model.model", "model.language_model", "model" }) |p| {
        const name = std.fmt.bufPrint(&kb, "{s}.word_embeddings.weight", .{p}) catch continue;
        if (md.w.src.tensor(name).ctx != null) break name;
    } else return error.LmPassTestNoEmbedding;
    try ling_embed.open(pack, key, full.vocab_size, full.hidden_size);
    errdefer ling_embed.close();
    const ctx: abi.LoadCtx = .{ .model_dir = abi.Str.of(pack), .dir = &dir, .parsed = &md.cut_parsed, .stream = .{ .ctx = s.ctx }, .weights = &md.w.src };
    md.m = try trunk.load(&ctx);
    errdefer {
        trunk.Model.deinit(md.m);
        gpa.destroy(md.m);
    }
    if (md.m.layers.len != n_layers or md.m.layers[mla_layer].mixer != .mla) return error.LmPassTestLayers;
    // the head at the whole config (its layer is `num_hidden_layers`), then the cut config again
    md.m.c = full;
    const head_err = ling_mtp.load(md.m, &md.w.src);
    md.m.c = &md.cut;
    try head_err;
    const head = md.m.ling_mtp orelse return error.LmPassTestNoHead;
    const st = head.moe.streamed orelse return error.LmPassTestNoHead;
    const hook = st.routed;
    for (md.m.layers, 0..) |*l, li| if (l.mlp == .moe) {
        l.mlp.moe.streamed = .{ .routed = hook, .layer = @intCast(li) };
    };
}

// ── the checks ──

const Counter = struct {
    lines: usize = 0,
    failed: usize = 0,

    fn check(c: *Counter, ok: bool, comptime fmt: []const u8, args: anytype) void {
        c.lines += 1;
        if (!ok) c.failed += 1;
        std.debug.print(tag ++ " " ++ fmt ++ " {s}\n", args ++ .{if (ok) "PASS" else "FAIL"});
    }
};

/// The words of `xs` against `ys`, item by item, summed; every word of a missing item counted
fn differingAll(xs: []const Array, ys: []const Array, s: Stream) !struct { d: usize, n: usize } {
    var d: usize = 0;
    var n: usize = 0;
    for (xs, 0..) |x, i| {
        n += mlx.mlx_array_size(x);
        d += if (i < ys.len) try differing(x, ys[i], s) else mlx.mlx_array_size(x);
    }
    return .{ .d = d, .n = n };
}

/// Every KDA layer's states at every checkpoint, B against A (`shift`: A's checkpoint i + shift), and whether both took
/// the same positions (unshifted)
fn cpDiffer(a: []const Cp, b: []const Cp, shift: usize, s: Stream) !struct { d: usize, n: usize, same_pos: bool } {
    var d: usize = 0;
    var n: usize = 0;
    var same_pos = a.len == b.len;
    for (b, 0..) |cb, i| {
        const j = i + shift;
        if (j >= a.len) break;
        const ca = a[j];
        if (shift == 0 and ca.pos != cb.pos) same_pos = false;
        for (0..n_layers) |li| {
            if (cb.conv[li].ctx == null and ca.conv[li].ctx == null) continue;
            n += sizeOf(cb.conv[li]) + sizeOf(cb.ssm[li]);
            d += try differing(ca.conv[li], cb.conv[li], s) + try differing(ca.ssm[li], cb.ssm[li], s);
        }
    }
    return .{ .d = d, .n = n, .same_pos = same_pos };
}

/// `a` (16-bit words) with the lowest bit of its first word flipped, a new array of its shape and dtype
fn flipOne(a: Array, s: Stream) !Array {
    var c = mlx.mlx_array_new();
    defer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    if (mlx.mlx_array_itemsize(c) != 2) return error.LmPassTestDtype;
    const n = mlx.mlx_array_size(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const buf = try std.heap.page_allocator.alloc(u8, n * 2);
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, p[0 .. n * 2]);
    buf[0] ^= 1;
    const sh = mlx.getShape(c);
    const out = mlx.mlx_array_new_data(buf.ptr, sh.ptr, @intCast(sh.len), mlx.mlx_array_dtype(c));
    if (out.ctx == null) return error.MlxError;
    return out;
}

test "turboexl3_ling C4 bytes: the layer-major pass in 16,384-row segments (D2's group cut at 32,768, the last call joined by the segment tail rule) gives the single pass's words at 52,000 tokens of the pack's first 8 layers (each span's hidden, the joined hidden, the MLA layer's K and V, every KDA state at every checkpoint), and the MTP head's history in 4,096-row forwards one forward's cache rows (its output counted, not checked: serving never evaluates it), with RQ's hook at each forward and the run-path marks (GPU)" {
    const s = try gpu();
    const pack_raw = std.c.getenv("MLX_SERVE_LING_PACK") orelse {
        std.debug.print("{s} FAIL: set MLX_SERVE_LING_PACK to the pack the test loads\n", .{tag});
        return error.NoPack;
    };
    const pack = std.mem.span(pack_raw);
    // turboexl3 as the plugin's load configures it, restored after
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch {};
    try ling_math.configure();
    const prev_head = @import("ling_mtp_latent.zig").head;
    defer @import("ling_mtp_latent.zig").head = prev_head;
    var md: Model = undefined;
    try loadModel(s, pack, &md);
    defer md.deinit();
    const m = md.m;

    // the host's spans and checkpoints at 52,000 tokens (generate/prefill_chunk.zig `wholePromptPlan` at 4,096)
    var ebuf: [64]u64 = undefined;
    const ends = ling_rq.spanEnds(&ebuf, n_tokens);
    var rows: [64]u32 = undefined;
    var prev: u64 = 0;
    for (ends, 0..) |e, i| {
        rows[i] = @intCast(e - prev);
        prev = e;
    }
    const sp = rows[0..ends.len];
    const cps = ends[0 .. ends.len - 1];
    var c: Counter = .{};
    c.check(std.mem.eql(u32, sp, &want_spans), "the host's spans at {d} tokens: {any} (want {any})", .{ n_tokens, sp, want_spans });

    var a: Rec = .{};
    defer a.deinit();
    var b: Rec = .{};
    defer b.deinit();
    try runArm(s, m, sp, cps, std.math.maxInt(u64), false, &a);
    try runArm(s, m, sp, cps, b_rows, false, &b);
    c.check(a.segments == 1 and b.segments == want_b_segments and !a.lm_mark and b.lm_mark, "segments A={d} B={d} (want 1, {d}); LM_SEGMENTS marked A={} B={} (want false, true)", .{ a.segments, b.segments, want_b_segments, a.lm_mark, b.lm_mark });
    const ho = try differingAll(a.outs.items, b.outs.items, s);
    c.check(a.outs.items.len == sp.len and b.outs.items.len == sp.len and ho.d == 0, "each span's final hidden: spans A={d} B={d} (want {d}), differing={d}/{d}", .{ a.outs.items.len, b.outs.items.len, sp.len, ho.d, ho.n });
    const hj = try differing(a.joined, b.joined, s);
    // A's joined rows against its spans' rows joined by concatenation (the join before C4)
    var cat = mlx.mlx_array_new();
    defer free(cat);
    {
        const vec = mlx.mlx_vector_array_new_data(a.outs.items.ptr, a.outs.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
    }
    const hc = try differing(cat, a.joined, s);
    c.check(hj == 0 and hc == 0, "the joined hidden [1, {d}, {d}]: B against A differing={d}, A against its spans concatenated differing={d} (of {d})", .{ n_tokens, mlx.getShape(a.joined)[2], hj, hc, mlx.mlx_array_size(a.joined) });
    const dk = try differing(a.k7, b.k7, s);
    const dv = try differing(a.v7, b.v7, s);
    c.check(dk == 0 and dv == 0 and mlx.getShape(a.k7)[2] == n_tokens, "layer {d}'s K {any} and V {any}: differing K={d}/{d} V={d}/{d}; appends A={d} B={d}, capacity A={d} B={d}", .{ mla_layer, mlx.getShape(a.k7), mlx.getShape(a.v7), dk, mlx.mlx_array_size(a.k7), dv, mlx.mlx_array_size(a.v7), a.appends7, b.appends7, a.cap7, b.cap7 });
    const dc = try cpDiffer(a.cps.items, b.cps.items, 0, s);
    c.check(a.cps.items.len == cps.len and dc.same_pos and dc.d == 0 and dc.n > 0, "checkpoints A={d} B={d} (want {d}), the same positions in order {}, every KDA layer's conv and ssm differing={d}/{d}", .{ a.cps.items.len, b.cps.items.len, cps.len, dc.same_pos, dc.d, dc.n });

    // the MTP history: A in one forward, B in the forwards the served rule gives a prompt over 32,768 tokens (4,096 rows)
    try runHistory(s, m, n_tokens - 1, &a);
    try runHistory(s, m, ling_segments.historyRows(n_tokens - 1), &b);
    var b_rows_ok = b.head_calls.items.len == want_b_history.len;
    var at: u64 = 0;
    for (b.head_calls.items, 0..) |call, i| {
        if (i < want_b_history.len) b_rows_ok = b_rows_ok and call[0] == at and call[1] == want_b_history[i];
        at += call[1];
    }
    const a_rows_ok = a.head_calls.items.len == 1 and a.head_calls.items[0][0] == 0 and a.head_calls.items[0][1] == n_tokens - 1;
    c.check(a_rows_ok and b_rows_ok and !a.hist_mark and b.hist_mark, "the MTP history's forwards: RQ's hook A {any}, B {d} calls ({any}...); MTP_HISTORY_CHUNKS marked A={} B={} (want false, true)", .{ a.head_calls.items, b.head_calls.items.len, b.head_calls.items[0..@min(3, b.head_calls.items.len)], a.hist_mark, b.hist_mark });
    // What serving reads of the history: the head's cache rows, every one. A difference here fails.
    const hk = try differing(a.head_k, b.head_k, s);
    const hv = try differing(a.head_v, b.head_v, s);
    c.check(hk == 0 and hv == 0 and mlx.getShape(a.head_k)[2] == n_tokens - 1 and mlx.getShape(b.head_k)[2] == n_tokens - 1, "the MTP history's cache rows, what serving reads: K {any} differing={d}/{d}, V differing={d}/{d}", .{ mlx.getShape(a.head_k), hk, mlx.mlx_array_size(a.head_k), hv, mlx.mlx_array_size(a.head_v) });
    // The head's last-row output: counted, not checked. Serving never evaluates a history forward's output (the
    // file's comment: mlx-serve's `appendRows` and `run`, the plugin's `headForward` and `history`), and B's forwards
    // run attention and routed calls at other row counts than A's.
    const hout = try differing(a.head_out, b.head_out, s);
    std.debug.print("{s}_INFO the head's last-row output of the history, information only (serving never evaluates a history forward's output): differing={d}/{d}\n", .{ tag, hout, mlx.mlx_array_size(a.head_out) });

    // plants: B with token 49,500 changed; one flipped bit; the checkpoints one position apart
    var p: Rec = .{};
    defer p.deinit();
    try runArm(s, m, sp, cps, b_rows, true, &p);
    var before: usize = 0;
    var after: usize = 0;
    var pos: u64 = 0;
    for (sp, 0..) |r, j| {
        const d = if (j < p.outs.items.len) try differing(a.outs.items[j], p.outs.items[j], s) else mlx.mlx_array_size(a.outs.items[j]);
        if (pos + r <= plant_at) before += d else if (d > 0) after += 1;
        pos += r;
    }
    var k_head = mlx.mlx_array_new();
    defer free(k_head);
    var k_head_p = mlx.mlx_array_new();
    defer free(k_head_p);
    var v_tail = mlx.mlx_array_new();
    defer free(v_tail);
    var v_tail_p = mlx.mlx_array_new();
    defer free(v_tail_p);
    const ksh = mlx.getShape(a.k7);
    const vsh = mlx.getShape(a.v7);
    const cut: c_int = @intCast(plant_at);
    try mlx.check(mlx.mlx_slice(&k_head, a.k7, &zero4, 4, &[_]c_int{ ksh[0], ksh[1], cut, ksh[3] }, 4, &st4, 4, s));
    try mlx.check(mlx.mlx_slice(&k_head_p, p.k7, &zero4, 4, &[_]c_int{ ksh[0], ksh[1], cut, ksh[3] }, 4, &st4, 4, s));
    try mlx.check(mlx.mlx_slice(&v_tail, a.v7, &[_]c_int{ 0, 0, cut, 0 }, 4, &[_]c_int{ vsh[0], vsh[1], vsh[2], vsh[3] }, 4, &st4, 4, s));
    try mlx.check(mlx.mlx_slice(&v_tail_p, p.v7, &[_]c_int{ 0, 0, cut, 0 }, 4, &[_]c_int{ vsh[0], vsh[1], vsh[2], vsh[3] }, 4, &st4, 4, s));
    const k_head_d = try differing(k_head, k_head_p, s);
    const v_tail_d = try differing(v_tail, v_tail_p, s);
    c.check(before == 0 and after == 2 and k_head_d == 0 and v_tail_d > 0, "plant: B with token {d} changed: the spans before it differing={d} words (want 0), the spans from it differing {d} (want 2); layer {d}'s K before it differing={d} (want 0), V from it differing={d} (must differ)", .{ plant_at, before, after, mla_layer, k_head_d, v_tail_d });
    const flipped = try flipOne(b.joined, s);
    defer free(flipped);
    const df = try differing(b.joined, flipped, s);
    c.check(df == 1, "plant: one flipped bit of B's joined hidden: differing={d} (want 1)", .{df});
    const ds = try cpDiffer(a.cps.items, b.cps.items, 1, s);
    c.check(ds.d > 0, "plant: B's checkpoints against A's one position later: differing={d}/{d} (must differ)", .{ ds.d, ds.n });
    std.debug.print("{s}_DONE {d} lines, {d} failed\n", .{ tag, c.lines, c.failed });
    try std.testing.expectEqual(@as(usize, 0), c.failed);
}
