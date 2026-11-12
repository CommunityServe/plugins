//! turboexl3_ling's shared expert over its layer's W_hats (SHARED_LAYER_WHAT, src/ling_shared_layer.zig) against the
//! per-call path (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips).
//! - The oracle (A) is the copied MoE block's `sharedDown` at a prompt span outside a scope, on SWIGLU_EPI's path: the
//!   gate by `ling_exl3.linear`, up's SwiGLU epilogue (`ling_swiglu_epi.up` with the table the block passes it), the
//!   down by `ling_exl3.linear`; each linear decodes its W_hat per call.
//! - The variant (B) is `moeSpans`' scope: a scope around the layer (`begin`, `end`), each span through `runParts` as
//!   `sharedDown` hands it, the block's chain (A's) where it returns null.
//! - Operands as the loader keeps them: gate, up and down at Ling's shared expert (4096 -> 1280 -> 4096), I16 trellises
//!   [256, 80, 96] and [80, 256, 96] (K 6) with F16 suh and svh, each held through a handle of its own and evaluated
//!   (`Loader.keep`, `evalTrunk`). Random words (the gate's and up's outputs scaled to about 8, so the clamp binds), and
//!   with MLX_SERVE_LING_PACK=<pack dir> the pack's own layers 4 (limit 0) and 64 (limit 10), read as
//!   src/ling_weights.zig reads the shards.
//! - Spans in the layer-major pass's order at the standard request (4 of 4,096 rows, 1,965, 31), x bf16 [1, S, 4096],
//!   every fifth row 16 times larger. Two layers one after the other (limits 0 and 10), so the second scope must decode
//!   its own W_hats after the first one's drop. Each layer's outputs are evaluated together after its scope ends, as
//!   `evalLayer` runs after `moeSpans` returns.
//! - Checks: every word of every span's gate, h and y, B against A; the scope's counts (5 wide spans served, 4 of them
//!   over the kept W_hats, 3 W_hats decoded, the 31-row span left to the chain, nothing kept after `end`); the run-path
//!   mark after each layer (src/ling_ran.zig). Red checks in the same run: outside a scope nothing is served; a span over
//!   the other layer's W_hats (a missed drop) differs from its own words; one flipped bit counts as one differing word.
//! Each check prints a `TURBOEXL3_LING_SHARED_LAYER SXL` line ending in PASS or FAIL.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const fused_act = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const ling_exl3 = @import("ling_exl3.zig");
const ling_weights = @import("ling_weights.zig");
const epi = @import("ling_swiglu_epi.zig");
const sl = @import("ling_shared_layer.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Parts = sl.Parts;

const hidden: c_int = 4096;
const width: c_int = 1280;
/// K 6: 96 halfwords a 16 x 16 tile
const n_k6: c_int = 96;
/// The standard request's spans in the layer-major pass's order
const spans = [_]c_int{ 4096, 4096, 4096, 4096, 1965, 31 };
/// The spans the scope serves (past the register path's 32 rows)
const wide: usize = 5;

fn gpu() !Stream {
    // the GPU tests' own variable (test-only): at 1 they run, at any other value or unset they skip
    const run_gpu = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(run_gpu), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// SHARED_LAYER_WHAT's bit of the run-path line (src/ling_ran.zig), set by a scope's `end` whose later spans read its
/// kept W_hats (the bits are the process's: `markStart` clears them before each check)
fn marked() bool {
    return (ran.ranBits() & (@as(u64, 1) << @intFromEnum(ran.Path.SHARED_LAYER_WHAT))) != 0;
}

/// A run-path check's start (the review's follow-up b: a bit an earlier test in the binary set passed the check
/// vacuously): the bits cleared, so the check reads only what the pass or scope after it sets.
fn markStart() void {
    ran.clearForTests();
}

/// A copy of host words as an MLX array
fn hostArray(comptime T: type, data: []const T, shape: []const c_int, dt: mlx.mlx_dtype) !Array {
    const a = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dt);
    if (a.ctx == null) return error.MlxError;
    return a;
}

/// A new handle on `a`, as the loader's `keep` makes the Linear's
fn keep(a: Array) !Array {
    var h = mlx.mlx_array_new();
    errdefer free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

fn freeProj(p: sl.Proj) void {
    free(p.trellis);
    free(p.suh);
    free(p.svh);
}

/// A projection in_dim -> out_dim as the loader keeps it: random trellis words viewed as I16 [in_dim / 16,
/// out_dim / 16, n] (the pack stores I16), suh [in_dim] and svh [out_dim] in F16, each through a handle of its own,
/// evaluated. svh is `sv` x (0.5 + u).
fn randProj(s: Stream, rnd: std.Random, in_dim: c_int, out_dim: c_int, n: c_int, sv: f32) !sl.Proj {
    const a = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(in_dim, 16), @divExact(out_dim, 16), n };
    const words = try a.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer a.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    const u = try hostArray(u16, words, &shape, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    defer free(w);
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    const su = try a.alloc(f16, @intCast(in_dim));
    defer a.free(su);
    for (su) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const svv = try a.alloc(f16, @intCast(out_dim));
    defer a.free(svv);
    for (svv) |*v| v.* = @floatCast(sv * (0.5 + rnd.float(f32)));
    const suh = try hostArray(f16, su, &.{in_dim}, .float16);
    defer free(suh);
    const svh = try hostArray(f16, svv, &.{out_dim}, .float16);
    defer free(svh);
    const kt = try keep(w);
    errdefer free(kt);
    const ks = try keep(suh);
    errdefer free(ks);
    return .{ .trellis = kt, .suh = ks, .svh = try keep(svh) };
}

/// The pack's projection `<base>` as the loader takes it (Loader.triple, then `keep` in Loader.lin): its trellis, suh
/// and svh as stored, its mul1 checked; each evaluated (the load's evalTrunk).
fn packProj(src: *const @import("mlx_serve").abi.quant.Source, base: []const u8) !sl.Proj {
    var nb: [160]u8 = undefined;
    const names = [_][]const u8{ "trellis", "suh", "svh", "mul1" };
    var got: [4]Array = undefined;
    for (names, &got) |nm, *g| {
        const t = src.tensor(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, nm }));
        if (t.ctx == null) return error.MissingWeight;
        g.* = .{ .ctx = t.ctx };
    }
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, got[3]));
    if (@as(u32, @bitCast(v)) != ling_exl3.mul1_multiplier) return error.Exl3CodebookUnsupported;
    if (!ling_exl3.isTrellis(got[0], got[1], got[2])) return error.BadExl3Shape;
    const kt = try keep(got[0]);
    errdefer free(kt);
    const ks = try keep(got[1]);
    errdefer free(ks);
    const kv = try keep(got[2]);
    errdefer free(kv);
    for ([_]Array{ kt, ks, kv }) |x| try mlx.check(mlx.mlx_array_eval(x));
    return .{ .trellis = kt, .suh = ks, .svh = kv };
}

/// x [1, rows, 4096] in bf16: normal draws, every fifth row times 16
fn input(s: Stream, rnd: std.Random, rows: c_int) !Array {
    const a = std.heap.page_allocator;
    const xf = try a.alloc(f32, @intCast(rows * hidden));
    defer a.free(xf);
    for (xf, 0..) |*v, i| v.* = rnd.floatNorm(f32) * @as(f32, if ((i / @as(usize, @intCast(hidden))) % 5 == 4) 16 else 1);
    const f = try hostArray(f32, xf, &.{ 1, rows, hidden }, .float32);
    defer free(f);
    var x = mlx.mlx_array_new();
    errdefer free(x);
    try mlx.check(mlx.mlx_astype(&x, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}

/// `a` made contiguous and evaluated, for reading its words on the host
fn evaluated(a: Array, s: Stream) !Array {
    var c = mlx.mlx_array_new();
    errdefer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word)
fn differing(a: Array, b: Array, s: Stream) !usize {
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return n;
    const ca = try evaluated(a, s);
    defer free(ca);
    const cb = try evaluated(b, s);
    defer free(cb);
    const isz = mlx.mlx_array_itemsize(cb);
    const pa = mlx.mlx_array_data_uint8(ca) orelse return error.NoData;
    const pb = mlx.mlx_array_data_uint8(cb) orelse return error.NoData;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(!std.mem.eql(u8, pa[i * isz ..][0..isz], pb[i * isz ..][0..isz]));
    return d;
}

/// `a` with one bit of its word `at` flipped, a new array of its shape and dtype (the comparison's red check)
fn flipped(a: Array, at: usize, s: Stream) !Array {
    const c = try evaluated(a, s);
    defer free(c);
    const isz = mlx.mlx_array_itemsize(c);
    const n = mlx.mlx_array_size(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const buf = try std.heap.page_allocator.alloc(u8, n * isz);
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, p[0 .. n * isz]);
    buf[at * isz] ^= 1;
    const sh = mlx.getShape(c);
    const out = mlx.mlx_array_new_data(buf.ptr, sh.ptr, @intCast(sh.len), mlx.mlx_array_dtype(c));
    if (out.ctx == null) return error.MlxError;
    return out;
}

/// The elements where the clamp at `limit` binds: the gate's SiLU above it, and up beyond it in size
fn clampedCounts(gate: Array, up: Array, limit: f32, s: Stream) ![2]usize {
    var over: [2]usize = .{ 0, 0 };
    for ([_]Array{ gate, up }, 0..) |v, k| {
        var f = mlx.mlx_array_new();
        defer free(f);
        try mlx.check(mlx.mlx_astype(&f, v, .float32, s));
        const c = try evaluated(f, s);
        defer free(c);
        const p = mlx.mlx_array_data_float32(c) orelse return error.NoData;
        for (p[0..mlx.mlx_array_size(c)]) |e| over[k] += @intFromBool(if (k == 0) e / (1.0 + @exp(-e)) > limit else @abs(e) > limit);
    }
    return over;
}

/// One layer's shared expert: gate, up, down, and the layer's clamp
const Layer = struct {
    p: [3]sl.Proj,
    limit: f32,
    layer: u32,
    fn deinit(l: Layer) void {
        for (l.p) |q| freeProj(q);
    }
};

/// The pack's shards (MLX_SERVE_LING_PACK), opened once for the cases that read them
var pack_weights: ?*ling_weights.Weights = null;

fn layerOf(s: Stream, rnd: std.Random, layer: u32, limit: f32, from_pack: bool) !Layer {
    if (from_pack) {
        const w = pack_weights.?;
        var bb: [96]u8 = undefined;
        var l: Layer = .{ .p = undefined, .limit = limit, .layer = layer };
        var made: usize = 0;
        errdefer for (l.p[0..made]) |q| freeProj(q);
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |nm, i| {
            l.p[i] = try packProj(&w.src, try std.fmt.bufPrint(&bb, "model.layers.{d}.mlp.shared_experts.{s}", .{ layer, nm }));
            made += 1;
        }
        return l;
    }
    // E6-1's scales: gate and up outputs about 8, so the clamp binds on both; the down at 1 / sqrt(1280)
    const g = try randProj(s, rnd, hidden, width, n_k6, 8.0 / 64.0);
    errdefer freeProj(g);
    const u = try randProj(s, rnd, hidden, width, n_k6, 8.0 / 64.0);
    errdefer freeProj(u);
    return .{ .p = .{ g, u, try randProj(s, rnd, width, hidden, n_k6, 1.0 / @sqrt(@as(f32, @floatFromInt(width)))) }, .limit = limit, .layer = layer };
}

/// A: the per-call chain at a span (`sharedDown`'s SWIGLU_EPI path outside a scope). Lazy; the caller owns the three.
fn control(s: Stream, x: Array, l: Layer) !Parts {
    const g = l.p[0];
    const u = l.p[1];
    const d = l.p[2];
    const gate = try ling_exl3.linear(s, x, g.trellis, g.suh, g.svh);
    errdefer free(gate);
    const h = (try epi.up(s, x, u.trellis, u.suh, u.svh, gate, l.limit, fused_act.sigmoidTableFor, .shared)) orelse return error.EpilogueDeclined;
    errdefer free(h);
    return .{ .gate = gate, .h = h, .y = try ling_exl3.linear(s, h, d.trellis, d.suh, d.svh) };
}

/// B at a span, as `sharedDown` takes it: the scope's `runParts`, else the block's chain (A's). `served` says which.
fn variant(s: Stream, x: Array, l: Layer, served: *bool) !Parts {
    if (try sl.runParts(s, x, l.p[0], l.p[1], l.p[2], l.limit, fused_act.sigmoidTableFor)) |p| {
        served.* = true;
        return p;
    }
    served.* = false;
    return control(s, x, l);
}

fn evalAll(parts: []const Parts) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts) |p| for ([_]Array{ p.gate, p.h, p.y }) |a| {
        _ = mlx.mlx_vector_array_append_value(vec, a);
    };
    try mlx.check(mlx.mlx_eval(vec));
}

fn deinitAll(parts: []Parts, n: usize) void {
    for (parts[0..n]) |p| p.deinit();
}

/// The table the block's SWIGLU_EPI path reads at `limit`
fn tableFor(s: Stream, limit: u32) !Array {
    return if (limit == 0) fused_act.sigmoidTableFor(s, .bfloat16) else epi.f32Table(s, .bfloat16);
}

/// `l`'s three W_hats, decoded now by the per-call launch (the stale-W_hat check's operands)
fn whats(s: Stream, l: Layer) ![3]Array {
    var w: [3]Array = @splat(.{ .ctx = null });
    errdefer for (w) |a| free(a);
    for (l.p, &w) |p, *o| {
        const sh = mlx.getShape(p.trellis);
        const d = try t3.denseOf(p, @intCast(sh[0] * 16), .bfloat16, .mul1);
        o.* = try d.dequantize(s, p.trellis);
    }
    return w;
}

test "turboexl3_ling SHARED_LAYER_WHAT (SXL): a layer-major layer's spans over its shared expert's W_hats, decoded once at its first wide span and dropped after its spans, give the per-call path's words (gate, h, y) at the loader's operands, the standard request's spans in layer-major order, limits 0 and 10, the control's SWIGLU_EPI (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    defer sl.end();
    if (std.c.getenv("MLX_SERVE_LING_PACK")) |dir| {
        const io = std.Io.Threaded.global_single_threaded.io();
        pack_weights = try ling_weights.Weights.open(std.heap.c_allocator, io, std.mem.span(dir), "model.layers.72.", s);
    }
    defer if (pack_weights) |w| {
        w.deinit();
        pack_weights = null;
    };
    var prng = std.Random.DefaultPrng.init(0x5E5C);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var words: usize = 0;
    var clamp_bound = false;
    var reds: usize = 0;
    // red: outside a scope (the chunk loop, decode, the NextN head) nothing is served
    {
        const l = try layerOf(s, rnd, 4, 0, false);
        defer l.deinit();
        const x = try input(s, rnd, 64);
        defer free(x);
        const none = (try sl.runParts(s, x, l.p[0], l.p[1], l.p[2], 0, fused_act.sigmoidTableFor)) == null;
        lines += 1;
        failed += @intFromBool(!none);
        reds += @intFromBool(none);
        std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL red: outside a scope a 64-row span is not served: {s} {s}\n", .{ if (none) "yes" else "no", if (none) "PASS" else "FAIL" });
    }
    for ([_]bool{ false, true }) |from_pack| {
        if (from_pack and pack_weights == null) continue;
        const src_name: []const u8 = if (from_pack) "pack" else "random";
        // two layers, one after the other: the second scope opens after the first one's drop
        var layers: [2]Layer = undefined;
        layers[0] = try layerOf(s, rnd, 4, 0, from_pack);
        defer layers[0].deinit();
        layers[1] = try layerOf(s, rnd, 64, 10, from_pack);
        defer layers[1].deinit();
        var first_x: [2]Array = @splat(.{ .ctx = null });
        defer for (first_x) |x| free(x);
        var first_want_y: [2]Array = @splat(.{ .ctx = null });
        defer for (first_want_y) |y| free(y);
        for (layers, 0..) |l, li| {
            var xs: [spans.len]Array = @splat(.{ .ctx = null });
            defer for (xs) |x| free(x);
            for (&xs, spans) |*x, rows| x.* = try input(s, rnd, rows);
            // A: the per-call chain, every span, evaluated together
            var want: [spans.len]Parts = undefined;
            var n_want: usize = 0;
            defer deinitAll(&want, n_want);
            for (xs) |x| {
                want[n_want] = try control(s, x, l);
                n_want += 1;
            }
            try evalAll(want[0..n_want]);
            // B: the scope around the layer, each span as `sharedDown` hands it, evaluated after the scope ends
            var got: [spans.len]Parts = undefined;
            var n_got: usize = 0;
            defer deinitAll(&got, n_got);
            var served: [spans.len]bool = @splat(false);
            markStart();
            sl.begin();
            const at_begin = sl.stats();
            for (xs, 0..) |x, j| {
                got[n_got] = try variant(s, x, l, &served[j]);
                n_got += 1;
            }
            const in_scope = sl.stats();
            sl.end();
            const after = sl.stats();
            const marked_after = marked();
            try evalAll(got[0..n_got]);
            // the scope's counts: 5 wide spans served, 4 over the kept W_hats, 3 decoded; the 31-row span left to the
            // chain; nothing kept after the drop; the run-path mark after the layer
            var served_ok = true;
            for (served, spans) |sv, rows| served_ok = served_ok and (sv == (rows > 32));
            const counts_ok = at_begin.open and at_begin.kept == 0 and in_scope.spans_run == wide and in_scope.spans_reused == wide - 1 and
                in_scope.decoded == 3 and in_scope.kept == 3 and !after.open and after.kept == 0 and marked_after and served_ok;
            lines += 1;
            failed += @intFromBool(!counts_ok);
            std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL {s} layer={d} limit={d} scope: served {d} wide spans (want {d}), {d} over the kept W_hats (want {d}), {d} W_hats decoded (want 3), {d} kept in the scope, {d} after the drop; the 31-row span left to the chain: {s}; the run-path mark set: {s} {s}\n", .{ src_name, l.layer, l.limit, in_scope.spans_run, wide, in_scope.spans_reused, wide - 1, in_scope.decoded, in_scope.kept, after.kept, if (!served[spans.len - 1]) "yes" else "no", if (marked_after) "yes" else "no", if (counts_ok) "PASS" else "FAIL" });
            // every word of every span, B against A
            for (want[0..n_want], got[0..n_got], spans, 0..) |a, b, rows, j| {
                var dw: [3]usize = undefined;
                for ([_]Array{ b.gate, b.h, b.y }, [_]Array{ a.gate, a.h, a.y }, 0..) |gb, ga, k| dw[k] = try differing(gb, ga, s);
                const nel = mlx.mlx_array_size(a.gate) + mlx.mlx_array_size(a.h) + mlx.mlx_array_size(a.y);
                words += nel;
                const ok = dw[0] == 0 and dw[1] == 0 and dw[2] == 0;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL {s} layer={d} limit={d} span={d} rows={d} {s} differing gate={d} h={d} y={d} (of {d}, {d}, {d}) {s}\n", .{ src_name, l.layer, l.limit, j, rows, if (served[j]) "kept-W_hat" else "chain", dw[0], dw[1], dw[2], mlx.mlx_array_size(a.gate), mlx.mlx_array_size(a.h), mlx.mlx_array_size(a.y), if (ok) "PASS" else "FAIL" });
            }
            if (l.limit != 0) {
                const up = try ling_exl3.linear(s, xs[0], l.p[1].trellis, l.p[1].suh, l.p[1].svh);
                defer free(up);
                const over = try clampedCounts(want[0].gate, up, l.limit, s);
                std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL clamp {s} layer={d} rows=4096 limit={d} silu_over={d}/{d} up_over={d}/{d}\n", .{ src_name, l.layer, l.limit, over[0], mlx.mlx_array_size(up), over[1], mlx.mlx_array_size(up) });
                if (!from_pack) clamp_bound = over[0] > 0 and over[1] > 0;
            }
            first_x[li] = try keep(xs[0]);
            first_want_y[li] = try keep(want[0].y);
        }
        // red: the second layer's first span over the first layer's W_hats (what a missed drop under a wrong key would
        // serve) does not give its words
        {
            const w0 = try whats(s, layers[0]);
            defer for (w0) |a| free(a);
            const l1 = layers[1];
            const lim1: u32 = @intFromFloat(l1.limit);
            const stale = try sl.partsOver(s, first_x[1], l1.p[0], l1.p[1], l1.p[2], w0[0], w0[1], w0[2], lim1, try tableFor(s, lim1));
            defer stale.deinit();
            const d = try differing(stale.y, first_want_y[1], s);
            const red_ok = d > 0;
            lines += 1;
            failed += @intFromBool(!red_ok);
            reds += @intFromBool(red_ok);
            std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL red: {s} layer {d}'s first span over layer {d}'s W_hats differs from its own words: {d}/{d} {s}\n", .{ src_name, l1.layer, layers[0].layer, d, mlx.mlx_array_size(stale.y), if (red_ok) "PASS" else "FAIL" });
        }
        _ = mlx.mlx_clear_cache();
    }
    // red: the word comparison counts a one-bit change as one differing word
    {
        const l = try layerOf(s, rnd, 4, 0, false);
        defer l.deinit();
        const x = try input(s, rnd, 64);
        defer free(x);
        const o = try control(s, x, l);
        defer o.deinit();
        try evalAll(&.{o});
        const f = try flipped(o.y, 12345, s);
        defer free(f);
        const d = try differing(f, o.y, s);
        lines += 1;
        failed += @intFromBool(d != 1);
        reds += @intFromBool(d == 1);
        std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL red: one flipped bit of y differing={d} (want 1) {s}\n", .{ d, if (d == 1) "PASS" else "FAIL" });
    }
    // red (the review's follow-up b): a stale SHARED_LAYER_WHAT bit set before the check starts, then a scope whose one
    // wide span reuses nothing (its `end` marks nothing): the check must read the name absent
    {
        const l = try layerOf(s, rnd, 4, 0, false);
        defer l.deinit();
        const x = try input(s, rnd, 64);
        defer free(x);
        ran.mark(.SHARED_LAYER_WHAT);
        markStart();
        var sv = false;
        sl.begin();
        const one = try variant(s, x, l, &sv);
        defer one.deinit();
        sl.end();
        try evalAll(&.{one});
        const absent = !marked();
        lines += 1;
        failed += @intFromBool(!absent);
        reds += @intFromBool(absent);
        std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL red: a stale bit, then a scope whose one wide span reused nothing: the run-path name {s}\n", .{if (absent) "absent PASS" else "set, the stale bit passed the check FAIL"});
    }
    const sources: usize = if (pack_weights != null) 2 else 1;
    const want_lines = 1 + sources * (2 * (1 + spans.len) + 1) + 2;
    std.debug.print("TURBOEXL3_LING_SHARED_LAYER SXL DONE lines={d} failed={d} words={d} reds={d} clamp_bound={s} {s}\n", .{ lines, failed, words, reds, if (clamp_bound) "yes" else "no", if (failed == 0 and clamp_bound) "BYTES_OK" else "BYTES_DIFFER" });
    try std.testing.expectEqual(want_lines, lines);
    try std.testing.expectEqual(3 + sources, reds);
    try std.testing.expect(clamp_bound);
    try std.testing.expectEqual(@as(usize, 0), failed);
}
