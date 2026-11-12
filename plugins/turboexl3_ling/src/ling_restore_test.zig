//! turboexl3_ling's restore byte check, part A: the prefill, run by the plugin's own code (GPU; it runs with
//! MLX_SERVE_LING_GPU_TESTS=1 and MLX_SERVE_LING_RESTORE_OUT set to the file it writes, else it skips). Part B
//! (tests/ling_restore_host_test.zig) runs in the served mlx-serve's own unit-test build. It hands what this part wrote
//! to that tree's cache code, commits it to the SSD tier, restores it and compares the bytes. The two parts are
//! separate binaries because one binary cannot hold both: the copied blocks' MLX layer and the host's are the same file
//! twice, and each copy exports the same process-wide state (sdk/util/shared.zig `Shared`).
//! - The served cache, the one layout this part writes: each of the trunk's 9 MLA layers caches the roped key once a
//!   token as K and kv_b's rows as V (the rope-once cache, src/ling_mla_rope1.zig: K [1, 1, S, 64], V [1, 1, S, 16384]
//!   bf16, 296,064 bytes a token), and the MTP head's MLA layer caches its latent (src/ling_mtp_latent.zig: K [1, 1, S,
//!   512], V [1, 1, S, 64] bf16, 1,152 bytes a token).
//! - Two prompts at the served width, 4096, one for each of part B's commit paths. The chunk loop's plan for each
//!   (generate/prefill_chunk.zig `wholePromptPlan`, which part B recomputes and checks):
//!   - `flush`, 16,700 tokens: spans of 4096, 4096, 4096 and 4381 rows, then 31, with checkpoints at 4096, 8192, 12288
//!     and 16669. The 4381-row span is the merged tail: `nextChunkEnd` gives it the 285 rows that are left before the
//!     30-row backoff.
//!   - `spill`, the first 9,000 of the same tokens: spans of 4096, 4096 and 777 rows, then 31, with checkpoints at
//!     4096, 8192 and 8969. The decline spill writes at most 4 GiB a commit, so the 16,700-token prompt's keys and
//!     values (4.9 GB) would stop at 13,312 tokens there (part B states the limit). The 9,000-token prompt and its 3
//!     checkpoints come to 3.1 GB.
//! - Operands as the loader and the runtime build them. The config is Ling-3.1-flash's config.json (the pack's, at
//!   MLX_SERVE_LING_PACK), through the plugin's own parse. Each MLA projection is a dense f16 tensor stored [out, in]
//!   as the checkpoint stores it, held as the loader holds it: the [in, out] view that a plain matmul reads, cast to
//!   bf16 and evaluated. kv_b is plain, as the head's latent cache needs. The words are random, and each span's input
//!   rows are random bf16 [1, S, 4096].
//! - The cache's writes: each span runs through the copied mla.zig `mlaAttn` at each of the 9 MLA layers (7, 15, ...,
//!   71), so each span's keys and values reach CacheOps.append in the served layout. The MTP head's layer, named to
//!   src/ling_mtp_latent.zig as src/ling_mtp.zig's load names it, appends its normed latent and roped shared key to its
//!   own cache, at layer 0. Each KDA layer's state after each span goes in through the copied slots.zig `putEntry`
//!   (CacheOps.set_state): random words in the shapes that kda.zig gives Ling (conv [1, 3, 24576], ssm [1, 64, 128,
//!   128], bf16). At each checkpoint's span end, every KDA layer's state is read back through CacheOps.state, as
//!   plugin.zig `spanMajor` reads it for CacheOps.checkpoint.
//! - The CacheOps here records every append's arrays in call order, the rows that each layer holds (the view that the
//!   attention reads) and each layer's states.
//! - The file holds, for each prompt `<p>` (`spill`, `flush`), every append's K and V (`<p>.k.l<layer>.s<span>`,
//!   `<p>.v.l<layer>.s<span>`, and the head's `<p>.hk.s<span>`, `<p>.hv.s<span>`), every checkpoint's states
//!   (`<p>.conv.c<i>.l<layer>`, `<p>.ssm.c<i>.l<layer>`), its tokens, span ends and checkpoint positions; and the MLA
//!   layers, the KDA layers and the MLA widths (`mla_dims`: heads, nope, rope, v, kv_lora). Its metadata names the
//!   served cache's SSD root namespace as config.zig serves it (`ns`, src/ling_switches.zig `layoutNamespace`). About
//!   8.6 GB, all of it held in memory until the file is written. Each line here starts with
//!   `TURBOEXL3_LING_RESTORE_BYTES A`.
const std = @import("std");
const switches = @import("ling_switches.zig");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const sdk = kit.sdk;
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const slots = @import("mlx_serve/plugins/glm5_next/slots.zig");
const kda = @import("mlx_serve/plugins/glm5_next/kda.zig");
const family = @import("mlx_serve/plugins/glm5_next/config.zig");
const ling_config = @import("mlx_serve/plugins/bailing_hybrid/config.zig");
const mla = @import("mlx_serve/plugins/bailing_hybrid/mla.zig");
const ling_mtp_latent = @import("ling_mtp_latent.zig");

// What turboexl3's `sdk` reads through `mlx_host` in this binary (build.zig), as src/ling_gpu_tests.zig names them.
pub const mlx = kit.mlx;
pub const shared = @import("mlx_serve/sdk/util/shared.zig");
pub const quant_abi = kit.abi.quant;

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const gpa = std.heap.c_allocator;
const tag = "TURBOEXL3_LING_RESTORE_BYTES A";

/// A shape check's word on a line
fn yesNo(ok: bool) []const u8 {
    return if (ok) "yes" else "NO";
}

/// One prompt: its name in the file (the commit path that part B sends it through), its length, and the chunk loop's
/// span ends and checkpoints for it at width 4096 (part B checks them against the host's `wholePromptPlan`).
const Prompt = struct { name: []const u8, n: usize, span_ends: []const u32, checkpoints: []const u32 };
const prompts = [_]Prompt{
    .{ .name = "spill", .n = 9_000, .span_ends = &.{ 4096, 8192, 8969, 9000 }, .checkpoints = &.{ 4096, 8192, 8969 } },
    .{ .name = "flush", .n = 16_700, .span_ends = &.{ 4096, 8192, 12288, 16669, 16700 }, .checkpoints = &.{ 4096, 8192, 12288, 16669 } },
};

/// Token `i` of both prompts (the spill prompt is the first 9,000 of the flush prompt's).
fn tokenAt(i: usize) u32 {
    return @intCast((i * 7919 + 101) % 150_000 + 100);
}

/// The GPU tests' gate, test-only: MLX_SERVE_LING_GPU_TESTS set to 1
fn gpuTestsOn() bool {
    const v = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn gpu() !Stream {
    if (!gpuTestsOn()) return error.SkipZigTest;
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
/// `[in, out]` view that a plain matmul reads, cast to bf16 and evaluated (the copied trunk.zig `transposed`, `lin`)
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

/// One MLA layer's projections at Ling's geometry (the copied trunk.zig `Mla`; the low-rank query pair and the
/// head-wise gate, as Ling-3.1-flash has them)
fn mlaWeights(s: Stream, c: *const family.Config, seed: u64) !trunk.Mla {
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

/// The CacheOps of one slot that records what the plugin hands it: every append in call order, the rows each layer
/// holds (the appends joined on the row axis: the view that the attention reads), and each layer's two states.
const Rec = struct {
    s: Stream,
    k: []Array,
    v: []Array,
    conv: []Array,
    ssm: []Array,
    appends: std.ArrayList(Append) = .empty,

    const Append = struct { layer: u32, k: Array, v: Array };

    fn init(layers: usize, s: Stream) !Rec {
        const r: Rec = .{ .s = s, .k = try gpa.alloc(Array, layers), .v = try gpa.alloc(Array, layers), .conv = try gpa.alloc(Array, layers), .ssm = try gpa.alloc(Array, layers) };
        for ([_][]Array{ r.k, r.v, r.conv, r.ssm }) |xs| @memset(xs, .{ .ctx = null });
        return r;
    }

    fn deinit(r: *Rec) void {
        for ([_][]Array{ r.k, r.v, r.conv, r.ssm }) |xs| {
            for (xs) |a| free(a);
            gpa.free(xs);
        }
        for (r.appends.items) |ap| {
            free(ap.k);
            free(ap.v);
        }
        r.appends.deinit(gpa);
    }

    fn handle(r: *Rec) abi.Handle {
        return @ptrCast(r);
    }

    fn of(h: abi.Handle) *Rec {
        return @ptrCast(@alignCast(h.?));
    }

    fn mx(a: abi.Array) Array {
        return .{ .ctx = a.ctx };
    }

    fn ab(a: Array) abi.Array {
        return .{ .ctx = a.ctx };
    }

    fn failed(e: anyerror) abi.Verdict {
        return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
    }

    const ops: abi.CacheOps = .{
        .len = len,
        .append = append,
        .append_ring = appendRing,
        .view = view,
        .release = release,
        .append_rows = appendRows,
        .state = state,
        .set_state = setState,
        .state_rows = stateRows,
        .set_state_rows = setStateRows,
    };

    fn len(h: abi.Handle, layer: u32) callconv(.c) u64 {
        const k = of(h).k[layer];
        return if (k.ctx == null) 0 else @intCast(mlx.getShape(k)[2]);
    }

    fn join(r: *Rec, held: *Array, add: Array) !void {
        if (held.ctx == null) {
            held.* = takeRef(add);
            return;
        }
        const parts = [_]Array{ held.*, add };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        errdefer free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 2, r.s));
        free(held.*);
        held.* = joined;
    }

    fn appendOr(r: *Rec, layer: u32, k: Array, v: Array) !void {
        try r.appends.append(gpa, .{ .layer = layer, .k = takeRef(k), .v = takeRef(v) });
        try r.join(&r.k[layer], k);
        try r.join(&r.v[layer], v);
    }

    fn append(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        if (keep != 0) return abi.Verdict.refuse(.unsupported, "the restore check's slot keeps every row");
        const r = of(h);
        r.appendOr(layer, mx(k), mx(v)) catch |e| return failed(e);
        out.* = .{ .k = ab(r.k[layer]), .v = ab(r.v[layer]) };
        return .ok;
    }

    fn appendRing(h: abi.Handle, layer: u32, k: abi.Array, v: abi.Array, keep: u32, window: u32, ring: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = .{ h, layer, k, v, keep, window, ring, s, out };
        return abi.Verdict.refuse(.unsupported, "the restore check's slot has no ring");
    }

    fn view(h: abi.Handle, layer: u32, s: abi.Stream, out: *abi.KvView) callconv(.c) abi.Verdict {
        _ = s;
        const r = of(h);
        out.* = .{ .k = ab(r.k[layer]), .v = ab(r.v[layer]) };
        return .ok;
    }

    fn release(v: *abi.KvView) callconv(.c) void {
        v.* = .{};
    }

    fn appendRows(caches: [*]const abi.Handle, n: u32, layer: u32, k: abi.Array, v: abi.Array, keep: u32, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, layer, k, v, keep, s };
        return abi.Verdict.refuse(.unsupported, "the restore check's slot is one slot");
    }

    fn state(h: abi.Handle, layer: u32, which: abi.StateSlot) callconv(.c) abi.Array {
        const r = of(h);
        return switch (which) {
            .conv => ab(r.conv[layer]),
            .ssm => ab(r.ssm[layer]),
            else => .{},
        };
    }

    fn setState(h: abi.Handle, layer: u32, which: abi.StateSlot, value: abi.Array) callconv(.c) abi.Verdict {
        const r = of(h);
        const slot = switch (which) {
            .conv => &r.conv[layer],
            .ssm => &r.ssm[layer],
            else => return abi.Verdict.refuse(.unsupported, "the restore check's slot holds the conv and ssm states only"),
        };
        free(slot.*);
        slot.* = takeRef(mx(value));
        return .ok;
    }

    fn stateRows(caches: [*]const abi.Handle, n: u32, layer: u32, which: abi.StateSlot, s: abi.Stream) callconv(.c) abi.Array {
        _ = .{ caches, n, layer, which, s };
        return .{};
    }

    fn setStateRows(caches: [*]const abi.Handle, n: u32, layer: u32, which: abi.StateSlot, value: abi.Array, s: abi.Stream) callconv(.c) abi.Verdict {
        _ = .{ caches, n, layer, which, value, s };
        return abi.Verdict.refuse(.unsupported, "the restore check's slot is one slot");
    }
};

/// `a` into `map` under `key` (the map takes its own reference)
fn put(map: mlx.mlx_map_string_to_array, comptime fmt: []const u8, args: anytype, a: Array) !void {
    var buf: [64]u8 = undefined;
    const key = try std.fmt.bufPrintSentinel(&buf, fmt, args, 0);
    if (mlx.mlx_map_string_to_array_insert(map, key, a) != 0) return error.MapInsertFailed;
}

/// `xs` as a uint32 array `[n]`
fn u32Array(xs: []const u32) !Array {
    const a = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{@intCast(xs.len)}, 1, .uint32);
    if (a.ctx == null) return error.MlxError;
    return a;
}

fn shapeIs(a: Array, want: []const c_int) bool {
    return a.ctx != null and std.mem.eql(c_int, mlx.getShape(a), want);
}

/// The model's layers: the MLA layers', the KDA layers' and their count
const Layers = struct { mla: []const u32, kda: []const u32, n: usize };

/// What one prompt's prefill wrote: whether every write has the served cache's shape and dtype, its appends (K and V
/// pairs) and its checkpoint states
const Wrote = struct { ok: bool, appends: usize, states: usize };

/// Prompt `pi`'s prefill through the copied MLA layer and slots, as the served prefill runs it, on the served cache: its
/// writes into `out` under the prompt's names. Its recording slots go when it returns; `out` keeps its own reference on
/// every write.
fn prefill(s: Stream, c: *const family.Config, p: Prompt, pi: usize, ly: Layers, trunk_fa: *const trunk.Mla, head_fa: *const trunk.Mla, out: mlx.mlx_map_string_to_array) !Wrote {
    const model: trunk.Model = .{ .c = c, .s = s, .caps = .{}, .embed = .{}, .head = .{}, .final_norm = .{} };
    const hidden: c_int = @intCast(c.hidden_size);
    var trunk_rec = try Rec.init(ly.n, s);
    defer trunk_rec.deinit();
    var head_rec = try Rec.init(1, s);
    defer head_rec.deinit();
    const trunk_h = [_]abi.Handle{trunk_rec.handle()};
    const head_h = [_]abi.Handle{head_rec.handle()};
    var pos = [_]u32{0};
    const sl: slots.Slots = .{ .ops = &Rec.ops, .caches = &trunk_h, .positions = &pos };
    const sl_head: slots.Slots = .{ .ops = &Rec.ops, .caches = &head_h, .positions = &pos };

    const kv_dim: c_int = @intCast(c.linear_num_key_heads * c.linear_key_head_dim);
    const v_dim: c_int = @intCast(c.linear_num_value_heads * c.linear_value_head_dim);
    const conv_shape = [_]c_int{ 1, @as(c_int, @intCast(c.linear_conv_kernel_dim)) - 1, kv_dim * 2 + v_dim };
    const ssm_shape = [_]c_int{ 1, @intCast(c.linear_num_value_heads), @intCast(c.linear_value_head_dim), @intCast(c.linear_key_head_dim) };
    // Each prompt its own random words.
    const base: u64 = 10_000_000 * @as(u64, pi);

    var start: u32 = 0;
    var ci: usize = 0;
    for (p.span_ends, 0..) |end, j| {
        const rows: c_int = @intCast(end - start);
        pos[0] = start;
        for (0..ly.n) |li| {
            const seed: u64 = base + 10_000 * @as(u64, li) + j;
            if (c.isLinearLayer(@intCast(li))) {
                // The layer's state after this span, through the slot (kda.zig's shapes and dtype for Ling).
                var e: kda.Entry = .{ .conv_state = try randn(s, seed, &conv_shape, 1.0, .bfloat16), .ssm_state = try randn(s, seed + 500_000, &ssm_shape, 1.0, .bfloat16), .initialized = true };
                try slots.putEntry(sl, 0, @intCast(li), &e);
                continue;
            }
            const x = try randn(s, seed, &[_]c_int{ 1, rows, hidden }, 1.0, .bfloat16);
            defer free(x);
            const y = try mla.mlaAttn(&model, sl, x, trunk_fa, @intCast(li), @intCast(start), 1, rows, true);
            defer free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
        {
            const x = try randn(s, base + 900_000 + j, &[_]c_int{ 1, rows, hidden }, 1.0, .bfloat16);
            defer free(x);
            const y = try mla.mlaAttn(&model, sl_head, x, head_fa, 0, @intCast(start), 1, rows, true);
            defer free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
        // A checkpoint at this span's end: every KDA layer's state as the slot holds it, read as spanMajor reads it.
        if (ci < p.checkpoints.len and p.checkpoints[ci] == end) {
            for (ly.kda) |li| {
                const conv = takeRef(sl.state(0, li, .conv));
                defer free(conv);
                const ssm = takeRef(sl.state(0, li, .ssm));
                defer free(ssm);
                try put(out, "{s}.conv.c{d}.l{d}", .{ p.name, ci, li }, conv);
                try put(out, "{s}.ssm.c{d}.l{d}", .{ p.name, ci, li }, ssm);
            }
            ci += 1;
        }
        start = end;
    }

    // Every append in call order: each MLA layer's span appends, then the head's.
    const spans_of = try gpa.alloc(u32, ly.n);
    defer gpa.free(spans_of);
    @memset(spans_of, 0);
    for (trunk_rec.appends.items) |ap| {
        try put(out, "{s}.k.l{d}.s{d}", .{ p.name, ap.layer, spans_of[ap.layer] }, ap.k);
        try put(out, "{s}.v.l{d}.s{d}", .{ p.name, ap.layer, spans_of[ap.layer] }, ap.v);
        spans_of[ap.layer] += 1;
    }
    for (head_rec.appends.items, 0..) |ap, j| {
        try put(out, "{s}.hk.s{d}", .{ p.name, j }, ap.k);
        try put(out, "{s}.hv.s{d}", .{ p.name, j }, ap.v);
    }
    var appends_ok = head_rec.appends.items.len == p.span_ends.len;
    for (ly.mla) |li| appends_ok = appends_ok and spans_of[li] == p.span_ends.len;
    for (ly.kda) |li| appends_ok = appends_ok and spans_of[li] == 0;

    const tokens = try gpa.alloc(u32, p.n);
    defer gpa.free(tokens);
    for (tokens, 0..) |*t, i| t.* = tokenAt(i);
    inline for (.{ .{ "{s}.tokens", @as([]const u32, tokens) }, .{ "{s}.span_ends", p.span_ends }, .{ "{s}.checkpoints", p.checkpoints } }) |kv| {
        const a = try u32Array(kv[1]);
        defer free(a);
        try put(out, kv[0], .{p.name}, a);
    }

    // The cache as the attention reads it: every trunk MLA layer's rows on the rope-once cache (the roped key once, K
    // [1, 1, n, rope], and kv_b's rows, V [1, 1, n, heads (nope + v)]), the head's latent K [1, 1, n, kv_lora] and V
    // [1, 1, n, rope], all bf16, and each checkpoint's states in kda.zig's shapes.
    const heads: c_int = @intCast(c.num_attention_heads);
    const nope: c_int = @intCast(c.mla_qk_nope_head_dim);
    const vd: c_int = @intCast(c.mla_v_head_dim);
    const kv_lora: c_int = @intCast(c.mla_kv_lora_rank);
    const rope: c_int = @intCast(c.mla_qk_rope_head_dim);
    const n: c_int = @intCast(p.n);
    const k_want = [4]c_int{ 1, 1, n, rope };
    const v_want = [4]c_int{ 1, 1, n, heads * (nope + vd) };
    var rows_ok = true;
    for (ly.mla) |li| rows_ok = rows_ok and shapeIs(trunk_rec.k[li], &k_want) and shapeIs(trunk_rec.v[li], &v_want) and
        mlx.mlx_array_dtype(trunk_rec.k[li]) == .bfloat16 and mlx.mlx_array_dtype(trunk_rec.v[li]) == .bfloat16;
    const head_ok = shapeIs(head_rec.k[0], &.{ 1, 1, n, kv_lora }) and shapeIs(head_rec.v[0], &.{ 1, 1, n, rope }) and
        mlx.mlx_array_dtype(head_rec.k[0]) == .bfloat16 and mlx.mlx_array_dtype(head_rec.v[0]) == .bfloat16;
    const states_ok = ci == p.checkpoints.len and shapeIs(trunk_rec.conv[ly.kda[0]], &conv_shape) and shapeIs(trunk_rec.ssm[ly.kda[0]], &ssm_shape);
    const ok = rows_ok and head_ok and states_ok and appends_ok;
    std.debug.print("{s} PREFILL {s}, {d} tokens: spans {any}, checkpoints {any}; the trunk's cache, rope-once (the roped key once, kv_b's rows): {d} MLA layers ({d}..{d}), {d} appends each through CacheOps.append, K {any} V {any} bf16: {s}; the MTP head's latent cache: K [1, 1, {d}, {d}] V [1, 1, {d}, {d}] bf16: {s}; {d} KDA layers' states (conv {any}, ssm {any} bf16) at {d} checkpoints: {s} {s}\n", .{
        tag,        p.name,          p.n,              p.span_ends,            p.checkpoints,   ly.mla.len,
        ly.mla[0],  ly.mla[ly.mla.len - 1],            p.span_ends.len,        k_want,          v_want,
        yesNo(rows_ok), n,           kv_lora,          n,                      rope,            yesNo(head_ok),
        ly.kda.len, conv_shape,      ssm_shape,        ci,                     yesNo(states_ok), if (ok) "PASS" else "FAIL",
    });
    return .{ .ok = ok, .appends = trunk_rec.appends.items.len + head_rec.appends.items.len, .states = 2 * ci * ly.kda.len };
}

test "turboexl3_ling restore bytes (A): the prefill on the served cache (the trunk's rope-once MLA cache and the MTP head's latent) writes its rows and the checkpoints' states through CacheOps, to the file part B restores from (MLX_SERVE_LING_GPU_TESTS=1, MLX_SERVE_LING_RESTORE_OUT)" {
    const s = try gpu();
    const out_path = std.c.getenv("MLX_SERVE_LING_RESTORE_OUT") orelse {
        std.debug.print("{s} SKIP: set MLX_SERVE_LING_RESTORE_OUT to the file part B reads\n", .{tag});
        return error.SkipZigTest;
    };
    const pack = std.c.getenv("MLX_SERVE_LING_PACK") orelse {
        std.debug.print("{s} FAIL: set MLX_SERVE_LING_PACK to the pack whose config.json the check parses\n", .{tag});
        return error.NoPack;
    };
    const prev_head = ling_mtp_latent.head;
    defer ling_mtp_latent.head = prev_head;

    // The pack's config.json through the plugin's own parse, as the loader reads it.
    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{std.mem.span(pack)});
    defer gpa.free(cfg_path);
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
    defer gpa.free(json);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var parsed: abi.Parsed = .{};
    const verdict = ling_config.parse(&dir, abi.Str.of(json), .{}, null, true, &parsed);
    if (!verdict.isOk()) {
        std.debug.print("{s} FAIL: the plugin's parse refused {s}: {s}\n", .{ tag, cfg_path, verdict.why.slice() });
        return error.ConfigRefused;
    }
    defer ling_config.freeParsed(&parsed);
    const c = family.of(&parsed);
    const n_layers: usize = c.num_hidden_layers;
    var mla_layers: std.ArrayList(u32) = .empty;
    defer mla_layers.deinit(gpa);
    var kda_layers: std.ArrayList(u32) = .empty;
    defer kda_layers.deinit(gpa);
    for (0..n_layers) |li| try (if (c.isLinearLayer(@intCast(li))) &kda_layers else &mla_layers).append(gpa, @intCast(li));
    const layers_ok = mla_layers.items.len == 9 and kda_layers.items.len == n_layers - 9 and mla_layers.items[0] == 7 and mla_layers.items[8] == 71;

    const trunk_fa = try mlaWeights(s, c, 1000);
    defer freeMla(&trunk_fa);
    // The head's MLA layer: its own weights' address, as src/ling_mtp.zig's load names it to src/ling_mtp_latent.zig.
    const head_fa = try mlaWeights(s, c, 2000);
    defer freeMla(&head_fa);
    ling_mtp_latent.head = &head_fa;

    const out = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(out);
    const ly: Layers = .{ .mla = mla_layers.items, .kda = kda_layers.items, .n = n_layers };
    var all_ok = layers_ok;
    var appends: usize = 0;
    var states: usize = 0;
    for (prompts, 0..) |p, pi| {
        const w = try prefill(s, c, p, pi, ly, &trunk_fa, &head_fa, out);
        all_ok = all_ok and w.ok;
        appends += w.appends;
        states += w.states;
    }
    const dims = [_]u32{ c.num_attention_heads, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim, c.mla_v_head_dim, c.mla_kv_lora_rank };
    inline for (.{ .{ "mla_layers", @as([]const u32, mla_layers.items) }, .{ "kda_layers", @as([]const u32, kda_layers.items) }, .{ "mla_dims", @as([]const u32, &dims) } }) |kv| {
        const a = try u32Array(kv[1]);
        defer free(a);
        try put(out, kv[0], .{}, a);
    }

    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    var nb: [32]u8 = undefined;
    _ = mlx.mlx_map_string_to_string_insert(meta, "layers", try std.fmt.bufPrintSentinel(&nb, "{d}", .{n_layers}, 0));
    _ = mlx.mlx_map_string_to_string_insert(meta, "writer", "turboexl3_ling src/ling_restore_test.zig (part A)");
    // The served cache's SSD root namespace as config.zig serves it.
    _ = mlx.mlx_map_string_to_string_insert(meta, "ns", switches.layoutNamespace().ptr);
    try mlx.check(mlx.mlx_save_safetensors(out_path, out, meta));
    std.debug.print("{s} WROTE {s}: {d} prompts, {d} appends (K and V), {d} checkpoint states, the plans and the tokens; the SSD root's namespace {s}; the layers {s}\n", .{
        tag, std.mem.span(out_path), prompts.len, appends, states, switches.layoutNamespace(), if (layers_ok) "as Ling-3.1-flash has them" else "NOT Ling-3.1-flash's",
    });
    try std.testing.expect(all_ok);
}

// D2's byte test (src/ling_d2_test.zig) and C4's (src/ling_lm_pass_test.zig) are in this binary: they run the copied MLA
// layer, and C4's the copied trunk's load and layer-major pass, with the copied trunk's model, as part A does, and the
// suite's binary keeps that trunk on the expert streamer (build.zig `restoreTest`).
test {
    _ = @import("ling_d2_test.zig");
    _ = @import("ling_lm_pass_test.zig");
}
