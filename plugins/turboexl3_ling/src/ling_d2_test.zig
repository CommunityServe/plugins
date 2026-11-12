//! turboexl3_ling's D2 byte test (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips). D2 appends an MLA layer's
//! prompt spans past 24,576 in one CacheOps.append (the copied mla.zig `mlaGroup`, which table_forward.zig
//! `hiddenSpans` calls at the spans src/ling_rq.zig `groupStart` names) where an append a span grew the KV once a span.
//! It is in the restore check's binary (build.zig `restoreTest`): the one GPU binary that runs the copied MLA layer with
//! the copied trunk's model.
//! - The cache is mlx-serve 66aad949's dense KV cache as a fresh request's takes an append (`Host`: src/sdk/kv/cache.zig
//!   `updateDense` with no reservation, ring or restored share: `nextCapacity`, `growQuantBuf`'s zeros and copy of the
//!   kept rows, `writeAtOffset`, `buildSliceView`'s view of rows [0, offset), the buffer itself at its capacity).
//! - One trunk MLA layer (7) at Ling-3.1-flash's geometry (`ling_cfg`), its projections dense bf16 views of random f16
//!   weights as the restore check holds them.
//! - Four whole prompts from position 0 at the served width, 31,652 (the 32K check's), 32,000, 32,200 and 32,768
//!   tokens. Each cache first takes the six 4,096-row spans before 24,576 as their own appends, the same random words in
//!   both, so both hold 24,576 rows at 25,600 tokens of capacity. Then arm A, an append a span: the copied `mlaAttn` on
//!   each of the three spans past 24,576, at its position; arm B, D2: `mlaGroup` on the three.
//! - Word for word: the cache's rows after the spans (its K and V views) and each span's output. Then arm B's one append
//!   (7 in all, arm A's 9), each arm's capacity (D2's one growth from 25,600: 32,000, 32,000, 32,256, 32,768; an append a
//!   span's: 32,000, 32,000, 40,192, 40,192), and the D2 mark, set by arm B and not by arm A.
//! - A red check: arm B with the first grouped span's input from another seed must change its output and the cache.
//! The prompt spans run turboexl3's rope-once flash launch where the device passes the NAX gate (src/ling_math.zig
//! `configure`'s `mla_rope_once`), else the expanded key; both arms take the same. Each case prints a `TURBOEXL3_LING_D2`
//! line ending in PASS or FAIL. Nothing is timed.
const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const slots = @import("mlx_serve/plugins/glm5_next/slots.zig");
const family = @import("mlx_serve/plugins/glm5_next/config.zig");
const mla = @import("mlx_serve/plugins/bailing_hybrid/mla.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const tag = "TURBOEXL3_LING_D2";
const zero4 = [_]c_int{ 0, 0, 0, 0 };
const st4 = [_]c_int{ 1, 1, 1, 1 };

/// The MLA layer the test runs: Ling-3.1-flash's first
const layer: u32 = 7;
/// D2's position (src/ling_rq.zig `d2_from`)
const d2_at: c_int = 24576;

/// Ling-3.1-flash's MLA geometry and rope as the plugin's parse reads tests/fixtures/config-2.60bpw.json: hidden 4,096;
/// 64 heads; q_lora 1,536; kv_lora 512; nope 128, rope 64, v 128; `query_pre_attn_scalar` its qk_head_dim, 192; rope
/// theta 6,000,000, interleaved pairs; the head-wise gate; RMS eps 1e-6.
const ling_cfg: family.Config = .{
    .hidden_size = 4096,
    .num_attention_heads = 64,
    .mla_q_lora_rank = 1536,
    .mla_kv_lora_rank = 512,
    .mla_qk_nope_head_dim = 128,
    .mla_qk_rope_head_dim = 64,
    .mla_v_head_dim = 128,
    .query_pre_attn_scalar = 192,
    .rope_theta = 6_000_000,
    .rope_interleaved_pairs = true,
    .mla_head_gate = true,
    .rms_norm_eps = 1e-6,
};

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

/// normal(0, `sd`) in `dt` from `seed`, evaluated
fn randn(s: Stream, seed: u64, shape: []const c_int, sd: f32, dt: mlx.mlx_dtype) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0, sd, key, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A dense projection stored `[out, in]` (f16, as the checkpoint stores it), held as the loader holds it: the
/// `[in, out]` view that a plain matmul reads, cast to bf16 and evaluated (src/ling_restore_test.zig `dense`)
fn dense(s: Stream, seed: u64, out_dim: c_int, in_dim: c_int) !trunk.Linear {
    const ckpt = try randn(s, seed, &[_]c_int{ out_dim, in_dim }, 1.0 / @sqrt(@as(f32, @floatFromInt(in_dim))), .float16);
    defer free(ckpt);
    var t = mlx.mlx_array_new();
    defer free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, ckpt, &[_]c_int{ 1, 0 }, 2, s));
    var w = mlx.mlx_array_new();
    errdefer free(w);
    try mlx.check(mlx.mlx_astype(&w, t, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    return .{ .w = w };
}

/// An RMSNorm weight as loaded: bf16 ones `[dim]`
fn norm(s: Stream, dim: c_int) !Array {
    var w = mlx.mlx_array_new();
    errdefer free(w);
    try mlx.check(mlx.mlx_ones(&w, &[_]c_int{dim}, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    return w;
}

/// One MLA layer's projections at `ling_cfg`'s geometry (the low-rank query pair and the head-wise gate)
fn mlaWeights(s: Stream, seed: u64) !trunk.Mla {
    const c = &ling_cfg;
    const hidden: c_int = @intCast(c.hidden_size);
    const heads: c_int = @intCast(c.num_attention_heads);
    const nope: c_int = @intCast(c.mla_qk_nope_head_dim);
    const rope: c_int = @intCast(c.mla_qk_rope_head_dim);
    const vdim: c_int = @intCast(c.mla_v_head_dim);
    const q_lora: c_int = @intCast(c.mla_q_lora_rank);
    const kv_lora: c_int = @intCast(c.mla_kv_lora_rank);
    return .{
        .q_a = try dense(s, seed + 1, q_lora, hidden),
        .q_a_norm = try norm(s, q_lora),
        .q_b = try dense(s, seed + 2, heads * (nope + rope), q_lora),
        .kv_a = try dense(s, seed + 3, kv_lora + rope, hidden),
        .kv_a_norm = try norm(s, kv_lora),
        .kv_b = try dense(s, seed + 4, heads * (nope + vdim), kv_lora),
        .g = try dense(s, seed + 5, heads, hidden),
        .o = try dense(s, seed + 6, hidden, heads * vdim),
    };
}

fn freeMla(w: *const trunk.Mla) void {
    inline for (.{ w.q_a.w, w.q_a_norm, w.q_b.w, w.kv_a.w, w.kv_a_norm, w.kv_b.w, w.g.w, w.o.w }) |a| free(a);
}

/// mlx-serve 66aad949's capacity schedule (src/sdk/kv/cache.zig `KVCache.nextCapacity`)
fn nextCapacity(cur: usize, needed: usize) usize {
    if (needed <= cur) return cur;
    const step = @min(@max(cur / 4, 256), 8192);
    const target = @max(needed, cur + step);
    return (target + 255) / 256 * 256;
}

/// mlx-serve 66aad949's dense KV cache, one MLA layer of one slot, as a fresh request's cache takes an append
/// (src/sdk/kv/cache.zig `updateDense`, :596): the stale views freed; a growth past the capacity to `nextCapacity`
/// (`planGrow` with no reservation, ring or restored share), zeros with the kept rows copied in (`growQuantBuf`, :752);
/// the rows written at the offset (`writeAtOffset`, :786); the view of rows [0, offset), the buffer itself at its
/// capacity (`buildSliceView`, :802). It counts its appends.
const Host = struct {
    s: Stream,
    k: Array = .{ .ctx = null },
    v: Array = .{ .ctx = null },
    k_view: Array = .{ .ctx = null },
    v_view: Array = .{ .ctx = null },
    offset: c_int = 0,
    appends: usize = 0,

    fn deinit(h: *Host) void {
        inline for (.{ h.k, h.v, h.k_view, h.v_view }) |a| free(a);
    }

    fn cap(h: *const Host) c_int {
        return if (h.k.ctx == null) 0 else mlx.getShape(h.k)[2];
    }

    /// `buf` at `to` rows of capacity, shaped as `like` on the other axes: zeros, then its rows [0, offset) copied in
    fn grow(h: *Host, buf: *Array, to: c_int, like: Array) !void {
        const ls = mlx.getShape(like);
        const shape = [_]c_int{ ls[0], ls[1], to, ls[3] };
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &shape, 4, mlx.mlx_array_dtype(like), h.s));
        if (buf.ctx != null and h.offset > 0) {
            defer free(z);
            const kept = [_]c_int{ ls[0], ls[1], h.offset, ls[3] };
            var old = mlx.mlx_array_new();
            defer free(old);
            try mlx.check(mlx.mlx_slice(&old, buf.*, &zero4, 4, &kept, 4, &st4, 4, h.s));
            var u = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&u, z, old, &zero4, 4, &kept, 4, &st4, 4, h.s));
            free(buf.*);
            buf.* = u;
            return;
        }
        free(buf.*);
        buf.* = z;
    }

    /// `buf` with `rows` written at the offset
    fn write(h: *Host, buf: *Array, rows: Array) !void {
        const bs = mlx.getShape(buf.*);
        const n = mlx.getShape(rows)[2];
        const stop = [_]c_int{ bs[0], bs[1], h.offset + n, bs[3] };
        var u = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&u, buf.*, rows, &[_]c_int{ 0, 0, h.offset, 0 }, 4, &stop, 4, &st4, 4, h.s));
        free(buf.*);
        buf.* = u;
    }

    /// `dst` = rows [0, offset) of `buf`: the buffer itself at its capacity, else a slice
    fn viewOf(h: *Host, dst: *Array, buf: Array) !void {
        const bs = mlx.getShape(buf);
        dst.* = mlx.mlx_array_new();
        if (h.offset == bs[2]) {
            try mlx.check(mlx.mlx_array_set(dst, buf));
        } else {
            try mlx.check(mlx.mlx_slice(dst, buf, &zero4, 4, &[_]c_int{ bs[0], bs[1], h.offset, bs[3] }, 4, &st4, 4, h.s));
        }
    }

    fn append(h: *Host, k: Array, v: Array) !void {
        free(h.k_view);
        h.k_view = .{ .ctx = null };
        free(h.v_view);
        h.v_view = .{ .ctx = null };
        const n = mlx.getShape(k)[2];
        if (h.k.ctx == null or h.offset + n > h.cap()) {
            const to: c_int = @intCast(nextCapacity(@intCast(h.cap()), @intCast(h.offset + n)));
            try h.grow(&h.k, to, k);
            try h.grow(&h.v, to, v);
        }
        try h.write(&h.k, k);
        try h.write(&h.v, v);
        h.offset += n;
        h.appends += 1;
        try h.viewOf(&h.k_view, h.k);
        try h.viewOf(&h.v_view, h.v);
    }

    fn handle(h: *Host) abi.Handle {
        return @ptrCast(h);
    }

    fn of(hd: abi.Handle) *Host {
        return @ptrCast(@alignCast(hd.?));
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

    fn len(hd: abi.Handle, l: u32) callconv(.c) u64 {
        const h = of(hd);
        return if (l == layer) @intCast(h.offset) else 0;
    }

    fn appendOp(hd: abi.Handle, l: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        if (l != layer or keep != 0) return abi.Verdict.refuse(.unsupported, "the D2 test's cache holds one MLA layer's every row");
        const h = of(hd);
        h.append(.{ .ctx = k.ctx }, .{ .ctx = v.ctx }) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        out.* = .{ .k = .{ .ctx = h.k_view.ctx }, .v = .{ .ctx = h.v_view.ctx } };
        return .ok;
    }

    fn appendRing(hd: abi.Handle, l: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, ring: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = .{ hd, l, k, v, keep, window, ring, s, out };
        return abi.Verdict.refuse(.unsupported, "the D2 test's cache has no ring");
    }

    fn view(hd: abi.Handle, l: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        if (l != layer) return abi.Verdict.refuse(.unsupported, "the D2 test's cache holds one MLA layer");
        const h = of(hd);
        out.* = .{ .k = .{ .ctx = h.k_view.ctx }, .v = .{ .ctx = h.v_view.ctx } };
        return .ok;
    }

    fn release(v: *abi.KvView) callconv(.c) void {
        v.* = .{};
    }

    fn appendRows(caches: [*]const abi.Handle, n: u32, l: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, l, k, v, keep, s };
        return abi.Verdict.refuse(.unsupported, "the D2 test's cache is one slot");
    }

    fn state(hd: abi.Handle, l: u32, which: abi.StateSlot) callconv(.c) abi.Array {
        _ = .{ hd, l, which };
        return .{};
    }

    fn setState(hd: abi.Handle, l: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
        _ = .{ hd, l, which, value };
        return abi.Verdict.refuse(.unsupported, "the D2 test's cache holds no recurrent state");
    }

    fn stateRows(caches: [*]const abi.Handle, n: u32, l: u32, which: abi.StateSlot, s: abi.Stream) callconv(.c) abi.Array {
        _ = .{ caches, n, l, which, s };
        return .{};
    }

    fn setStateRows(caches: [*]const abi.Handle, n: u32, l: u32, which: abi.StateSlot, value: abi.Array, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, l, which, value, s };
        return abi.Verdict.refuse(.unsupported, "the D2 test's cache is one slot");
    }
};

/// The six spans before 24,576 (4,096 rows each): random roped keys `[1, 1, 4096, 64]` and kv_b rows
/// `[1, 1, 4096, 16384]`, bf16, the same words appended to every cache, each its own append
const Prefix = struct {
    k: [6]Array = @splat(.{ .ctx = null }),
    v: [6]Array = @splat(.{ .ctx = null }),

    fn deinit(p: *Prefix) void {
        for (p.k, p.v) |k, v| {
            free(k);
            free(v);
        }
    }
};

/// A cache that holds the prefix: 24,576 rows at 25,600 tokens of capacity
fn filled(s: Stream, p: *const Prefix) !Host {
    var h: Host = .{ .s = s };
    errdefer h.deinit();
    for (p.k, p.v) |k, v| try h.append(k, v);
    if (h.offset != d2_at or h.cap() != 25600) return error.D2TestPrefix;
    return h;
}

/// Every array of `xs` evaluated
fn evalAll(xs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    try mlx.check(mlx.mlx_eval(vec));
}

/// The 16-bit words in which `a` and `b` differ, counted on the GPU as integers: the words less the equal ones (a shape,
/// dtype or width mismatch counts every word)
fn differing(a: Array, b: Array, s: Stream) !usize {
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b) or mlx.mlx_array_itemsize(b) != 2) return n;
    var ua = mlx.mlx_array_new();
    defer free(ua);
    try mlx.check(mlx.mlx_view(&ua, a, .uint16, s));
    var ub = mlx.mlx_array_new();
    defer free(ub);
    try mlx.check(mlx.mlx_view(&ub, b, .uint16, s));
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

/// A whole prompt from position 0 at the served width (4,096): its spans' ends past 24,576 (mlx-serve's
/// generate/prefill_chunk.zig `wholePromptPlan`; src/ling_rq.zig `spanEnds`), and each arm's capacity after them
const Case = struct { p: c_int, ends: [3]c_int, want_a: c_int, want_b: c_int };
const cases = [_]Case{
    .{ .p = 31652, .ends = .{ 28672, 31621, 31652 }, .want_a = 32000, .want_b = 32000 },
    .{ .p = 32000, .ends = .{ 28672, 31969, 32000 }, .want_a = 32000, .want_b = 32000 },
    .{ .p = 32200, .ends = .{ 28672, 32169, 32200 }, .want_a = 40192, .want_b = 32256 },
    .{ .p = 32768, .ends = .{ 28672, 32737, 32768 }, .want_a = 40192, .want_b = 32768 },
};

/// The three grouped spans' rows and positions
const Group = struct { rows: [3]c_int, starts: [3]c_int };

fn groupOf(c: Case) Group {
    var g: Group = undefined;
    var prev: c_int = d2_at;
    for (c.ends, &g.rows, &g.starts) |e, *r, *st| {
        r.* = e - prev;
        st.* = prev;
        prev = e;
    }
    return g;
}

/// Each grouped span's input rows `[1, rows, 4096]`, random bf16 from `seed`
fn inputs(s: Stream, g: Group, seed: u64) ![3]Array {
    var xs: [3]Array = @splat(.{ .ctx = null });
    errdefer for (xs) |x| free(x);
    for (&xs, g.rows, 0..) |*x, r, i| x.* = try randn(s, seed + i, &[_]c_int{ 1, r, 4096 }, 1.0, .bfloat16);
    return xs;
}

/// Arm A, an append a span: the copied `mlaAttn` on each grouped span, at its position
fn armA(model: *const trunk.Model, fa: *const trunk.Mla, h: *Host, g: Group, xs: [3]Array, out: *[3]Array) !void {
    const hs = [_]abi.Handle{h.handle()};
    for (xs, g.rows, g.starts, out) |x, r, st, *o| {
        const pos = [_]u32{@intCast(st)};
        const sl: slots.Slots = .{ .ops = &Host.ops, .caches = &hs, .positions = &pos };
        o.* = try mla.mlaAttn(model, sl, x, fa, layer, st, 1, r, true);
    }
    try evalAll(out);
}

/// Arm B, D2: `mlaGroup` on the grouped spans, from 24,576
fn armB(model: *const trunk.Model, fa: *const trunk.Mla, h: *Host, g: Group, xs: [3]Array, out: *[3]Array) !void {
    const hs = [_]abi.Handle{h.handle()};
    const pos = [_]u32{@intCast(d2_at)};
    const sl: slots.Slots = .{ .ops = &Host.ops, .caches = &hs, .positions = &pos };
    try mla.mlaGroup(model, sl, &xs, fa, layer, d2_at, &g.rows, out);
    try evalAll(out);
}

fn passFail(ok: bool) []const u8 {
    return if (ok) "PASS" else "FAIL";
}

fn runCase(s: Stream, model: *const trunk.Model, fa: *const trunk.Mla, prefix: *const Prefix, c: Case, seed: u64, flash: bool) !bool {
    defer _ = mlx.mlx_clear_cache();
    const g = groupOf(c);
    const xs = try inputs(s, g, seed);
    defer for (xs) |x| free(x);
    ran.clearForTests();
    defer ran.clearForTests();
    var a = try filled(s, prefix);
    defer a.deinit();
    var ya: [3]Array = @splat(.{ .ctx = null });
    defer for (ya) |y| free(y);
    try armA(model, fa, &a, g, xs, &ya);
    const d2_a = ran.has(.D2);
    var b = try filled(s, prefix);
    defer b.deinit();
    var yb: [3]Array = @splat(.{ .ctx = null });
    defer for (yb) |y| free(y);
    try armB(model, fa, &b, g, xs, &yb);
    const d2_b = ran.has(.D2);
    const dk = try differing(a.k_view, b.k_view, s);
    const dv = try differing(a.v_view, b.v_view, s);
    var dy: usize = 0;
    var ny: usize = 0;
    for (ya, yb) |x, y| {
        dy += try differing(x, y, s);
        ny += mlx.mlx_array_size(x);
    }
    const ok = dk == 0 and dv == 0 and dy == 0 and a.appends == 9 and b.appends == 7 and a.cap() == c.want_a and b.cap() == c.want_b and !d2_a and d2_b;
    std.debug.print("{s} P={d}: spans past 24576 of {d}, {d} and {d} rows (flash {s}); appends A={d} B={d} (want 9, 7); capacity A={d} B={d} (want {d}, {d}); K differing={d}/{d}, V differing={d}/{d}, outputs differing={d}/{d}; D2 marked A={} B={} (want false, true) {s}\n", .{
        tag,             c.p,                            g.rows[0],                      g.rows[1], g.rows[2],    if (flash) "on" else "off (no NAX: the expanded key)",
        a.appends,       b.appends,                      a.cap(),                        b.cap(),   c.want_a,     c.want_b,
        dk,              mlx.mlx_array_size(b.k_view),   dv,                             mlx.mlx_array_size(b.v_view), dy, ny,
        d2_a,            d2_b,                           passFail(ok),
    });
    return ok;
}

/// The red check: arm B with the first grouped span's input from another seed, against arm A, must change words
fn redCheck(s: Stream, model: *const trunk.Model, fa: *const trunk.Mla, prefix: *const Prefix, c: Case, seed: u64) !bool {
    defer _ = mlx.mlx_clear_cache();
    const g = groupOf(c);
    const xs = try inputs(s, g, seed);
    defer for (xs) |x| free(x);
    var other = xs;
    other[0] = try randn(s, seed + 1000, &[_]c_int{ 1, g.rows[0], 4096 }, 1.0, .bfloat16);
    defer free(other[0]);
    var a = try filled(s, prefix);
    defer a.deinit();
    var ya: [3]Array = @splat(.{ .ctx = null });
    defer for (ya) |y| free(y);
    try armA(model, fa, &a, g, xs, &ya);
    var r = try filled(s, prefix);
    defer r.deinit();
    var yr: [3]Array = @splat(.{ .ctx = null });
    defer for (yr) |y| free(y);
    try armB(model, fa, &r, g, other, &yr);
    const dy = try differing(ya[0], yr[0], s);
    const dv = try differing(a.v_view, r.v_view, s);
    const ok = dy > 0 and dv > 0;
    std.debug.print("{s}_RED P={d}: the first grouped span's input from another seed: its output differing={d}/{d}, V differing={d}/{d} (must differ) {s}\n", .{ tag, c.p, dy, mlx.mlx_array_size(ya[0]), dv, mlx.mlx_array_size(a.v_view), passFail(ok) });
    return ok;
}

test "turboexl3_ling D2 bytes: an MLA layer's prompt spans past 24,576 appended in one call (mla.zig mlaGroup) give the cache and each span's output the words of an append a span (mlaAttn), at one KV growth from 25,600, and mark D2 (GPU)" {
    const s = try gpu();
    // turboexl3's rope-once flash on the prompt spans, as src/ling_math.zig `configure` sets it (it runs where the device
    // passes turboexl3's NAX gate)
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch {};
    var cf = saved;
    cf.mla_rope_once = true;
    try t3.configure(cf);
    const fa = try mlaWeights(s, 7000);
    defer freeMla(&fa);
    const model: trunk.Model = .{ .c = &ling_cfg, .s = s, .caps = .{}, .embed = .{}, .head = .{}, .final_norm = .{} };
    var prefix: Prefix = .{};
    defer prefix.deinit();
    for (&prefix.k, &prefix.v, 0..) |*k, *v, i| {
        k.* = try randn(s, 900 + i, &[_]c_int{ 1, 1, 4096, 64 }, 1.0, .bfloat16);
        v.* = try randn(s, 950 + i, &[_]c_int{ 1, 1, 4096, 16384 }, 1.0, .bfloat16);
    }
    var failed: usize = 0;
    for (cases, 0..) |c, ci| failed += @intFromBool(!try runCase(s, &model, &fa, &prefix, c, 0x5EED_D200 + 16 * ci, cf.nax));
    failed += @intFromBool(!try redCheck(s, &model, &fa, &prefix, cases[2], 0x5EED_D2FF));
    std.debug.print("{s}_DONE {d} lines, {d} failed\n", .{ tag, cases.len + 1, failed });
    try std.testing.expectEqual(@as(usize, 0), failed);
}
