//! The speed changes' tests on the Metal device (`zig build gpu-test -Dmlx-lib=<dir>`), one a speed change, each
//! against the stock arm it replaces at GLM-5.3-Flash's shapes: the exact ones bit for bit, the three rounding-class
//! ones (`KDA_PROMPT_PERCORE`, `KDA_PROMPT_CHUNK`, `DSA_PROMPT_ONEP`) within their bound. The KDA ones run a layer
//! through the kit trunk's `kda.forward` with this plugin's stages set, so they hold the stage wiring too.
//! Not here: GLM53F_HEAD_LAST_ROW (its arm is the kit's `moe.mlpLastRow`, tested in mlx-serve, and needs a format
//! bank) and GLM53F_PROMPT_ASYNC_EVAL (the same arrays to `mlx_async_eval`): the standard A/B's greedy text holds both.
//! Ported from #1196's (mlx-serve 4bab19a5) tests of the same names, at fewer cases.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const ops = kit.mlx_ops;
const kh = kit.bases.kda_hybrid;
const kda = kh.kda;
const trunk = kh.trunk;
const options = @import("options.zig");
const stages = @import("stages.zig");
const kda_verify = @import("kda_verify.zig");
const kda_prologue = @import("kda_prologue.zig");
const kda_percore = @import("kda_percore.zig");
const kda_chunk = @import("kda_chunk.zig");
const dsa_prompt = @import("dsa_prompt.zig");
const dsa_append = @import("dsa_append.zig");
const dsa_nax = @import("dsa_nax.zig");
const glm5 = @import("glm5_next.zig");
const router_nax = @import("router_nax.zig");

const testing = std.testing;
const Array = mlx.mlx_array;
const Switch = options.Switch;

fn skipOffGpu() !mlx.mlx_stream {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    return mlx.gpuStream();
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn newRef(a: Array) Array {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, a);
    return out;
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Sets each of `sws` to `on` (the others as the environment has them) until `release`.
const Forced = struct {
    fn set(sws: []const Switch, on: bool) void {
        for (sws) |sw| options.forced[@intFromEnum(sw)] = on;
    }
    fn release() void {
        options.forced = @splat(null);
    }
};

/// Seeded normal values of `dtype` at `scale`, evaluated.
fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Seeded uniform values in [lo, hi) of `dtype`, evaluated.
fn uniform(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, lo: f32, hi: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, u, dtype, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// `got` holds `want`'s bits; with `lead`, over their first `lead` rows only.
fn expectBits(s: mlx.mlx_stream, want: Array, got: Array, lead: ?c_int) !void {
    if (lead == null) return ops.expectBitEqual(got, want);
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| free(a);
    for ([_]Array{ want, got }, &parts) |a, *p| {
        const sh = mlx.getShape(a);
        const start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const stride: [8]c_int = @splat(1);
        for (sh, 0..) |d, i| stop[i] = d;
        stop[0] = lead.?;
        p.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(p, a, &start, sh.len, &stop, sh.len, &stride, sh.len, s));
    }
    return ops.expectBitEqual(parts[1], parts[0]);
}

/// The largest |got - want| over the largest |want|, both in f32; a nonfinite `got` fails.
fn relApart(s: mlx.mlx_stream, want: Array, got: Array) !f32 {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var t: [8]Array = @splat(.{ .ctx = null });
    defer for (t) |a| free(a);
    for (&t) |*a| a.* = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&t[0], want, .float32, s));
    try mlx.check(mlx.mlx_astype(&t[1], got, .float32, s));
    try mlx.check(mlx.mlx_subtract(&t[2], t[1], t[0], s));
    try mlx.check(mlx.mlx_abs(&t[3], t[2], s));
    try mlx.check(mlx.mlx_max(&t[4], t[3], false, s));
    try mlx.check(mlx.mlx_abs(&t[5], t[0], s));
    try mlx.check(mlx.mlx_max(&t[6], t[5], false, s));
    try mlx.check(mlx.mlx_max(&t[7], t[1], false, s));
    try evalAll(&.{ t[4], t[6], t[7] });
    var apart: f32 = 0;
    var scale: f32 = 0;
    var top: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&apart, t[4]));
    try mlx.check(mlx.mlx_array_item_float32(&scale, t[6]));
    try mlx.check(mlx.mlx_array_item_float32(&top, t[7]));
    try testing.expect(scale > 0 and std.math.isFinite(apart) and std.math.isFinite(top));
    return apart / scale;
}

/// bf16 outputs round at 2^-8 of their own magnitude and a summation order moves the f32 state far less, while a wrong
/// gate, head or row moves them by their own size.
const rounding_bound: f32 = 1e-2;

// ── the KDA layer through the trunk, with this plugin's stages ──

/// A GLM-5.3-Flash KDA layer at its served 64 heads of 128, 4 conv taps, the low-rank gates.
const t_hidden: c_int = 512;
const t_heads: c_int = 64;
const t_head: c_int = 128;
const t_rank: c_int = 128;
const t_prompt: c_int = 17;

/// The layer and its model: plain bf16 projections of seeded weights, pre-transposed as the loader leaves them, and
/// this plugin's stages.
const Rig = struct {
    cfg: kh.config.Config,
    m: trunk.Model = undefined,
    la: trunk.Kda = undefined,
    w: [12]Array = @splat(.{ .ctx = null }),
    seed: u64 = 20261112,

    fn init(r: *Rig, s: mlx.mlx_stream) !void {
        r.* = .{ .cfg = .{ .model_type = "glm5_next", .hidden_size = @intCast(t_hidden), .linear_num_key_heads = @intCast(t_heads), .linear_num_value_heads = @intCast(t_heads), .linear_key_head_dim = @intCast(t_head), .linear_value_head_dim = @intCast(t_head), .linear_conv_kernel_dim = 4, .kda_gate_lower_bound = -5.0 } };
        r.m = .{ .c = &r.cfg, .s = s, .caps = .{}, .embed = .{}, .head = .{}, .final_norm = .{} };
        r.m.kda.stages = stages.all;
        errdefer r.deinit();
        const dim = t_heads * t_head;
        const shapes = [_][]const c_int{ &.{ t_hidden, 3 * dim }, &.{ t_hidden, t_rank }, &.{ t_rank, dim }, &.{ t_hidden, t_rank }, &.{ t_rank, dim }, &.{ t_hidden, t_heads }, &.{ dim, t_hidden }, &.{ 3 * dim, 4, 1 } };
        const scales = [_]f32{ 1.0 / 16.0, 1.0 / 16.0, 1.0 / 11.0, 1.0 / 16.0, 1.0 / 11.0, 1.0 / 16.0, 1.0 / 22.0, 0.5 };
        for (shapes, scales, 0..) |sh, sc, i| r.w[i] = try normal(s, &r.seed, sh, .bfloat16, sc);
        // A_log per head and per key channel, dt_bias, f32 as the packs load them; the gated norm's weight bf16.
        r.w[8] = try uniform(s, &r.seed, &.{t_heads}, .float32, 0.0, 2.5);
        r.w[9] = try expandHeads(s, r.w[8]);
        r.w[10] = try normal(s, &r.seed, &.{dim}, .float32, 1.0);
        r.w[11] = try uniform(s, &r.seed, &.{t_head}, .bfloat16, 0.5, 1.5);
        r.la = .{ .qkv = .{ .w = r.w[0] }, .z = .{ .w = r.w[1] }, .z2 = .{ .w = r.w[2] }, .a = .{ .w = r.w[3] }, .a2 = .{ .w = r.w[4] }, .b = .{ .w = r.w[5] }, .out = .{ .w = r.w[6] }, .conv_w = r.w[7], .A_log = r.w[9], .dt_bias = r.w[10], .norm_w = r.w[11], .a_log_h = r.w[8], .dt_f32 = r.w[10] };
    }

    fn deinit(r: *Rig) void {
        for (r.w) |a| free(a);
        r.w = @splat(.{ .ctx = null });
        for ([_]Array{ r.m.kda.q_scale, r.m.kda.k_scale, r.m.kda.ones, r.m.kda.eps }) |a| free(a);
        r.m.kda = .{};
    }

    fn input(r: *Rig, rows: c_int) !Array {
        return normal(r.m.s, &r.seed, &.{ 1, rows, t_hidden }, .bfloat16, 1.0);
    }
};

/// A per-head A_log [h] expanded per key channel [h * 128], as the trunk loads it.
fn expandHeads(s: mlx.mlx_stream, per_head: Array) !Array {
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, per_head, &[_]c_int{ t_heads, 1 }, 2, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ t_heads, t_head }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, bc, &[_]c_int{t_heads * t_head}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, flat, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

const verify_switches = [_]Switch{ .kda_verify, .kda_expsig, .kda_lazy_seq };

/// A cold prompt chunk, then two verifies of `rows` rows with capture, under the switches as set: what each verify
/// leaves (output, window, state, conv input, per-row states), evaluated.
fn verifies(r: *Rig, x0: Array, xs: []const Array, rows: c_int, out: [][5]Array) !void {
    var e: kda.Entry = .{};
    defer e.deinit();
    const y0 = try kda.forward(&r.m, x0, &r.la, &e, 1, t_prompt, true, false);
    defer free(y0);
    try evalAll(&.{ y0, e.conv_state, e.ssm_state });
    for (xs, out) |x, *o| {
        o[0] = try kda.forward(&r.m, x, &r.la, &e, 1, rows, rows > 1, true);
        o[1] = newRef(e.conv_state);
        o[2] = newRef(e.ssm_state);
        o[3] = newRef(e.spec_conv_input);
        o[4] = newRef(e.spec_state_seq);
        try evalAll(o);
    }
}

test "glm-5.3-flash gpu: kda verify: the prework, gate, exp(A_log) and lazy per-row states give the chain's output, window, state, conv input and per-row states bit for bit" {
    const s = try skipOffGpu();
    defer Forced.release();
    var r: Rig = undefined;
    try r.init(s);
    defer r.deinit();
    const x0 = try r.input(t_prompt);
    defer free(x0);
    // 12 rows: the prework declines past 9, the gate and norm-gate still serve.
    for ([_]c_int{ 1, 2, 4, 12 }) |rows| {
        const xs: [2]Array = .{ try r.input(rows), try r.input(rows) };
        defer for (xs) |x| free(x);
        var want: [2][5]Array = @splat(@splat(.{ .ctx = null }));
        defer for (&want) |*o| for (o) |a| free(a);
        Forced.set(&verify_switches, false);
        const before_chain = kda_verify.runs;
        try verifies(&r, x0, &xs, rows, &want);
        try testing.expectEqual(before_chain, kda_verify.runs);
        Forced.set(&verify_switches, true);
        var got: [2][5]Array = @splat(@splat(.{ .ctx = null }));
        defer for (&got) |*o| for (o) |a| free(a);
        const before = kda_verify.runs;
        try verifies(&r, x0, &xs, rows, &got);
        try testing.expectEqual(before.gate + 2, kda_verify.runs.gate);
        try testing.expectEqual(before.lazy + 2, kda_verify.runs.lazy);
        try testing.expectEqual(before.normgate + 2, kda_verify.runs.normgate);
        try testing.expectEqual(before.prework + @as(u64, if (rows <= 9) 2 else 0), kda_verify.runs.prework);
        for (want, got) |w, g| {
            for (w[0..4], g[0..4]) |a, b| try expectBits(s, a, b, null);
            // the capture writes T - 1 per-row states (its last is the final state's); a one-row verify none
            if (rows > 1) try expectBits(s, w[4], g[4], rows - 1);
        }
    }
}

/// A layer's prompt under the switches as set: a cold `rows`-row chunk, then a warm 31-row chunk, each chunk's output,
/// conv window and state, evaluated.
fn prompts(r: *Rig, xa: Array, rows: c_int, xb: Array) ![6]Array {
    var o: [6]Array = @splat(.{ .ctx = null });
    errdefer for (o) |a| free(a);
    var e: kda.Entry = .{};
    defer e.deinit();
    o[0] = try kda.forward(&r.m, xa, &r.la, &e, 1, rows, true, false);
    o[1] = newRef(e.conv_state);
    o[2] = newRef(e.ssm_state);
    try evalAll(o[0..3]);
    o[3] = try kda.forward(&r.m, xb, &r.la, &e, 1, 31, true, false);
    o[4] = newRef(e.conv_state);
    o[5] = newRef(e.ssm_state);
    try evalAll(o[3..6]);
    return o;
}

const prompt_switches = [_]Switch{ .kda_prompt_prologue, .kda_prompt_normgate };
const inexact_kda = [_]Switch{ .kda_prompt_percore, .kda_prompt_chunk };

test "glm-5.3-flash gpu: kda prompt: the prologue and the norm-gate give the chain's output, window and state, 16,356 rows cold then 31 warm" {
    const s = try skipOffGpu();
    defer Forced.release();
    var r: Rig = undefined;
    try r.init(s);
    defer r.deinit();
    const xa = try r.input(16_356);
    defer free(xa);
    const xb = try r.input(31);
    defer free(xb);
    Forced.set(&inexact_kda, false);
    Forced.set(&prompt_switches, false);
    const before_chain = kda_prologue.runs;
    const want = try prompts(&r, xa, 16_356, xb);
    defer for (want) |a| free(a);
    try testing.expectEqual(before_chain, kda_prologue.runs);
    Forced.set(&prompt_switches, true);
    const before = kda_prologue.runs;
    const got = try prompts(&r, xa, 16_356, xb);
    defer for (got) |a| free(a);
    // where the stock per-core route takes the rows (64 cores or more), no stage is asked
    if (!kit.blocks.kda.recurrence.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32)) try testing.expectEqual(before + 2, kda_prologue.runs);
    for (want, got) |w, g| try expectBits(s, w, g, null);
}

/// The stock prompt path's two chunks against `got`'s: the windows bit for bit (no route moves them), the outputs and
/// states within `rounding_bound` of their largest magnitude.
fn expectNearStock(r: *Rig, xa: Array, xb: Array, got: *const [6]Array) !void {
    const s = r.m.s;
    Forced.set(&inexact_kda, false);
    const want = try prompts(r, xa, 16_356, xb);
    defer for (want) |a| free(a);
    for ([_]usize{ 1, 4 }) |i| try expectBits(s, want[i], got[i], null);
    for ([_]usize{ 0, 2, 3, 5 }) |i| {
        const apart = try relApart(s, want[i], got[i]);
        std.debug.print("glm53f kda prompt: output {d} apart from the stock path by {e} of its largest magnitude\n", .{ i, apart });
        try testing.expect(apart <= rounding_bound);
    }
}

test "glm-5.3-flash gpu: kda percore: rounding-class: a layer's prompt stays within the bound of the stock prompt path" {
    const s = try skipOffGpu();
    defer Forced.release();
    if (kit.blocks.kda.recurrence.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32)) return error.SkipZigTest;
    if (!kda_percore.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32)) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init(s);
    defer r.deinit();
    const xa = try r.input(16_356);
    defer free(xa);
    const xb = try r.input(31);
    defer free(xb);
    Forced.set(&.{.kda_prompt_chunk}, false);
    Forced.set(&.{.kda_prompt_percore}, true);
    const before = kda_percore.runs;
    const got = try prompts(&r, xa, 16_356, xb);
    defer for (got) |a| free(a);
    try testing.expectEqual(before + 2, kda_percore.runs);
    try expectNearStock(&r, xa, xb, &got);
}

test "glm-5.3-flash gpu: kda chunk: rounding-class: a layer's prompt on the chunkwise route stays within the bound of the stock prompt path" {
    const s = try skipOffGpu();
    defer Forced.release();
    if (kit.blocks.kda.recurrence.serves(s, t_heads, t_heads, t_head, t_head, .bfloat16, .float32)) return error.SkipZigTest;
    if (!try kda_chunk.probe(s, t_heads, -5.0)) return error.SkipZigTest;
    var r: Rig = undefined;
    try r.init(s);
    defer r.deinit();
    const xa = try r.input(16_356);
    defer free(xa);
    const xb = try r.input(31);
    defer free(xb);
    Forced.set(&.{.kda_prompt_chunk}, true);
    const before = kda_chunk.runs;
    const got = try prompts(&r, xa, 16_356, xb);
    defer for (got) |a| free(a);
    try testing.expectEqual(before + 2, kda_chunk.runs);
    try expectNearStock(&r, xa, xb, &got);
}

// ── DSA ──

const ix = dsa_prompt.test_ix;

/// An indexer selection's inputs: q_idx bf16 [1, rows, 32, 128], the head weights f32 [1, rows, 32], the indexer cache
/// bf16 [1, t, 256].
const Sel = struct {
    q: Array,
    w: Array,
    cache: Array,
    fn init(s: mlx.mlx_stream, seed: *u64, t: c_int, rows: c_int) !Sel {
        const q = try normal(s, seed, &.{ 1, rows, ix.heads, ix.head_dim }, .bfloat16, 1.0);
        errdefer free(q);
        const w = try normal(s, seed, &.{ 1, rows, ix.heads }, .float32, 1.0);
        errdefer free(w);
        return .{ .q = q, .w = w, .cache = try normal(s, seed, &.{ 1, t, 2 * ix.head_dim }, .bfloat16, 1.0) };
    }
    fn deinit(x: *Sel) void {
        for ([_]Array{ x.q, x.w, x.cache }) |a| free(a);
    }
    fn select(x: Sel, s: mlx.mlx_stream, lever: bool) !Array {
        Forced.set(&.{.dsa_prompt_indexer}, lever);
        const out = try glm5.selectTokens(s, x.q, x.w, x.cache, ix);
        errdefer free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }
};

test "glm-5.3-flash gpu: dsa prompt indexer: the one launch selects the stock tokens, the output's one copy is the stock layout, and rows that select every pool decline" {
    const s = try skipOffGpu();
    defer Forced.release();
    if (!kit.blocks.linear.verify_qmm_nax.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const Case = struct { t: c_int, rows: c_int, runs: bool };
    // a 16K prompt's main call and tail, a 64K context's last rows, and a cold prompt within the selection
    const cases = [_]Case{ .{ .t = 16356, .rows = 14305, .runs = true }, .{ .t = 16387, .rows = 31, .runs = true }, .{ .t = 65536, .rows = 256, .runs = true }, .{ .t = 995, .rows = 995, .runs = false } };
    var seed: u64 = 20261109;
    for (cases) |c| {
        var x = try Sel.init(s, &seed, c.t, c.rows);
        defer x.deinit();
        const want = try x.select(s, false);
        defer free(want);
        const before = dsa_prompt.index_runs;
        const got = try x.select(s, true);
        defer free(got);
        try ops.expectBitEqual(got, want);
        try testing.expectEqual(@as(u64, @intFromBool(c.runs)), dsa_prompt.index_runs - before);
    }
    // the output: the dense and sparse parts [1, 64, rows, 256] in the output projection's row order
    Forced.set(&.{.dsa_prompt_indexer}, true);
    for ([_][2]c_int{ .{ 2051, 14305 }, .{ 2051, 17 } }) |rows| {
        const dn = try normal(s, &seed, &.{ 1, dsa_prompt.attn_heads, rows[0], dsa_prompt.attn_v }, .bfloat16, 1.0);
        defer free(dn);
        const sp = try normal(s, &seed, &.{ 1, dsa_prompt.attn_heads, rows[1], dsa_prompt.attn_v }, .bfloat16, 1.0);
        defer free(sp);
        var o = mlx.mlx_array_new();
        defer free(o);
        const two = [_]Array{ dn, sp };
        const vec = mlx.mlx_vector_array_new_data(&two, two.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
        var want = mlx.mlx_array_new();
        defer free(want);
        try mlx.check(mlx.mlx_transpose_axes(&want, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var got = mlx.mlx_array_new();
        defer free(got);
        try testing.expect(try dsa_prompt.assemble(&got, s, dn, sp));
        try ops.expectBitEqual(got, want);
    }
}

test "glm-5.3-flash gpu: dsa prompt one-P: rounding-class: the one-piece P stays within the bound of the stock two-piece text, and runs" {
    const s = try skipOffGpu();
    defer Forced.release();
    if (!kit.blocks.linear.verify_qmm_nax.verifyQmmNaxAvailable()) return error.SkipZigTest;
    var seed: u64 = 20261113;
    // a 16K prompt's last 1,024 rows and its 31-row tail
    for ([_][2]c_int{ .{ 16356, 1024 }, .{ 16387, 31 } }) |c| {
        var x = try Sel.init(s, &seed, c[0], c[1]);
        defer x.deinit();
        const sel = try x.select(s, false);
        defer free(sel);
        const q = try normal(s, &seed, &.{ 1, dsa_prompt.attn_heads, c[1], 512 }, .bfloat16, 1.0);
        defer free(q);
        const lat = try normal(s, &seed, &.{ 1, 1, c[0], 512 }, .bfloat16, 1.0);
        defer free(lat);
        const plain = (try dsa_nax.sparseLatent(s, q, lat, sel, 0.0625)) orelse return error.TestUnexpectedResult;
        defer free(plain);
        Forced.set(&.{.dsa_prompt_onep}, true);
        const before = dsa_prompt.onep_runs;
        const onep = (try dsa_prompt.sparseOneP(s, q, lat, sel, 0.0625)) orelse return error.TestUnexpectedResult;
        defer free(onep);
        try testing.expectEqual(before + 1, dsa_prompt.onep_runs);
        const apart = try relApart(s, plain, onep);
        std.debug.print("glm53f dsa one-P: {e} of the stock text's largest magnitude apart\n", .{apart});
        try testing.expect(apart > 0 and apart <= rounding_bound);
    }
}

/// `Slots` as the tails' `begin` and `take` read it: one slot's cache handle and position, a view per layer.
var mock_cache: u8 = 0;
const MockView = struct {
    v: Array,
    pub fn deinit(self: *MockView) void {
        self.* = .{ .v = .{ .ctx = null } };
    }
};
const MockSlots = struct {
    caches: []const ?*anyopaque,
    positions: []const u32,
    layers: []const u32,
    views: []const Array,
    pub fn view(self: MockSlots, i: usize, layer: u32, s: mlx.mlx_stream) !MockView {
        _ = s;
        if (i != 0) return error.TestUnexpectedResult;
        for (self.layers, self.views) |l, v| if (l == layer) return .{ .v = v };
        return error.TestUnexpectedResult;
    }
};

test "glm-5.3-flash gpu: dsa tails: each DSA layer gets the stock tail's rows bit for bit, and a prompt or a batch reads none" {
    const s = try skipOffGpu();
    defer {
        Forced.release();
        dsa_append.end();
    }
    Forced.set(&.{.dsa_tails}, true);
    var seed: u64 = 20261109;
    const trunk_ids = [_]u32{ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 };
    const handles = [_]?*anyopaque{@ptrCast(&mock_cache)};
    // deep in a 16K cache (3 rows) and at its second row (1), the trunk's layers and the head's
    for ([_]struct { layers: []const u32, t: c_int, rows: c_int }{ .{ .layers = &trunk_ids, .t = 16390, .rows = 3 }, .{ .layers = &trunk_ids, .t = 1, .rows = 2 }, .{ .layers = &.{0}, .t = 16390, .rows = 1 } }) |c| {
        var bufs: [dsa_append.trunk_layers]Array = @splat(.{ .ctx = null });
        var views: [dsa_append.trunk_layers]Array = @splat(.{ .ctx = null });
        defer for (bufs, views) |b, v| {
            free(b);
            free(v);
        };
        for (c.layers, 0..) |_, i| {
            bufs[i] = try normal(s, &seed, &.{ 1, 1, c.t + 61, dsa_append.row_w }, .bfloat16, 1.0);
            views[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&views[i], bufs[i], &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 1, c.t, dsa_append.row_w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        }
        const pos = [_]u32{@intCast(c.t)};
        const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = c.layers, .views = views[0..c.layers.len] };
        const n_prev: c_int = @min(c.t, 3);
        try dsa_append.begin(s, sl, false, &.{ 1, c.rows }, c.layers);
        defer dsa_append.end();
        for (c.layers, 0..) |layer, i| {
            var tail = mlx.mlx_array_new();
            defer free(tail);
            try mlx.check(mlx.mlx_slice(&tail, views[i], &[_]c_int{ 0, 0, c.t - n_prev, 0 }, 4, &[_]c_int{ 1, 1, c.t, dsa_append.row_w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var want = mlx.mlx_array_new();
            defer free(want);
            try mlx.check(mlx.mlx_reshape(&want, tail, &[_]c_int{ 1, n_prev, dsa_append.row_w }, 3, s));
            const got = dsa_append.take(sl, 0, layer, n_prev) orelse return error.TestUnexpectedResult;
            try ops.expectBitEqual(got, want);
        }
        try testing.expect(dsa_append.take(sl, 0, c.layers[0], n_prev + 1) == null);
        try dsa_append.begin(s, sl, false, &.{ 1, 17 }, c.layers);
        try testing.expect(dsa_append.take(sl, 0, c.layers[0], n_prev) == null);
        try dsa_append.begin(s, sl, true, &.{ 1, 1 }, c.layers);
        try testing.expect(dsa_append.take(sl, 0, c.layers[0], n_prev) == null);
    }
}

// ── the NAX router ──

test "glm-5.3-flash gpu: router: the plan's routes and weights are the stock route's bit for bit at 1 to 4 rows" {
    const s = try skipOffGpu();
    defer Forced.release();
    if (!kit.mlx_device.naxAvailable()) return error.SkipZigTest;
    Forced.set(&.{ .router_nax, .router_one1 }, true);
    var seed: u64 = 20261107;
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |wt| {
        const raw = try normal(s, &seed, &.{ router_nax.experts, router_nax.hidden }, wt, 1.0 / 64.0);
        defer free(raw);
        try testing.expect(try router_nax.admit(raw));
        // the loader's f32 table: the transposed view widened, materialized
        var tr = mlx.mlx_array_new();
        defer free(tr);
        try mlx.check(mlx.mlx_transpose_axes(&tr, raw, &[_]c_int{ 1, 0 }, 2, s));
        var wide = mlx.mlx_array_new();
        defer free(wide);
        try mlx.check(mlx.mlx_astype(&wide, tr, .float32, s));
        var zero = mlx.mlx_array_new();
        defer free(zero);
        try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 0, .float32, s));
        var tbl = mlx.mlx_array_new();
        defer free(tbl);
        try mlx.check(mlx.mlx_add(&tbl, wide, zero, s));
        try mlx.check(mlx.mlx_array_eval(tbl));
        const bias = try normal(s, &seed, &.{router_nax.experts}, .float32, 0.1);
        defer free(bias);
        const router: router_nax.Router = .{ .raw = raw, .table = tbl, .plain = true, .bias = bias, .k = 8, .norm = true, .scale = 2.5 };
        for ([_]c_int{ 1, 2, 3, 4 }) |rows| for (0..16) |_| {
            if (rows > 1 and !kit.mlx_device.tf32Matmul()) continue;
            const x = try normal(s, &seed, &.{ 1, rows, router_nax.hidden }, .bfloat16, 1.0);
            defer free(x);
            var logits = mlx.mlx_array_new();
            defer free(logits);
            try mlx.check(mlx.mlx_matmul(&logits, x, tbl, s));
            const want = (try kit.blocks.moe.router_kernel.moeRouterTopK(s, logits, bias, 8, .sigmoid_bias, true, 2.5, .bfloat16, 0, 0, null)) orelse return error.TestStockDeclined;
            defer {
                free(want.inds);
                free(want.norm_scores);
            }
            const got = (try router_nax.route(s, x, router)) orelse return error.TestPlanDeclined;
            defer {
                free(got.inds);
                free(got.norm_scores);
            }
            try ops.expectBitEqual(got.inds, want.inds);
            try ops.expectBitEqual(got.norm_scores, want.norm_scores);
        };
    }
}
