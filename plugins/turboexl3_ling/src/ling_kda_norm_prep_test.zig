//! KDA_NORM_PREP's byte test (L5, kept, built in: no option turns it on or off; GPU, `MLX_SERVE_LING_GPU_TESTS=1` runs
//! it, else it skips). The plugin's own call is src/ling_kda_norm_prep.zig `oProj`, with no option set and the operands
//! the copied KDA layer (plugins/glm5_next/kda.zig) gives it at its norm-gate site: turboexl3's `ling.kdaNormGatePrep`,
//! one launch that writes o_proj's prepared rows of the norm-gate's output, then o_proj's matmul against LAYER_WHAT's
//! kept W_hat and its finish from those rows. The oracle is the path that site serves: the host's norm-gate
//! (decode_fused `gdnNormGateFusedTab` with the sigmoid gate, on its table, as the copied layer calls it with the
//! packed prework), then o_proj's `linear` (src/ling_exl3.zig `linear`, what `m.mm` runs), per call outside any scope.
//! - Operands at Ling-3.1-flash's KDA shapes: o_proj 8192 -> 4096 as the loader keeps it (K 6, random I16 trellis
//!   words, F16 suh and svh, each through a handle of its own); y [1, S, 64, 128], z [1, S, 8192] and norm_w [128] in
//!   bf16, every fifth row of the last axis of y and z 16 times larger; eps 1e-6, f32 0-d, as the copied layer makes it.
//! - LAYER_WHAT's kept W_hat: a KDA layer's scope (`begin`; o_proj is the layer's one EXL3 linear here, so the scope
//!   holds it alone) around spans of 4,096 and 1,965 rows (the standard request's full span and its tail span, which
//!   ends inside a tile): each call's output against the per-call served path's, every word, with the run-path name
//!   KDA_NORM_PREP set; then the scope's counts (o_proj decoded once and read by both spans) and the pass's LAYER_WHAT
//!   mark.
//! - The declines, each with no run-path name set and, in a scope, nothing kept, decoded or read: 31 rows in a scope
//!   (the standard request's last span, o_proj's register path, where `linear` makes no W_hat), 4,096 rows outside a
//!   scope, and pg2 on in a scope. After the pg2-on decline, the served path in the same scope with pg2 off, as served
//!   (the norm-gate, then o_proj's `linear` over the scope's W_hat), gives the per-call words.
//! - A red check, which shows that the test can fail and that the call runs turboexl3's launch: a fault planted in the
//!   launch through turboexl3's `kdanorm.kernel_override` (norm_w read at the next channel, as turboexl3's own red check
//!   plants it) must give other words than the served path's, at 4,096 rows over the kept W_hat.
//! Each check prints a `TURBOEXL3_LING_KDA_NORM_PREP` line ending in PASS or FAIL; the DONE line counts them.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const decode_fused = @import("mlx_serve/sdk/blocks/gdn/decode_fused.zig");
const ling_exl3 = @import("ling_exl3.zig");
const norm_prep = @import("ling_kda_norm_prep.zig");
const lwhat = @import("ling_layer_what.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const kdanorm = t3.ling.kdanorm;

const tag = "TURBOEXL3_LING_KDA_NORM_PREP";
/// Ling-3.1-flash's KDA layer: 64 value heads of 128 (the norm-gate's output and o_proj's input, 8,192 wide) and the
/// hidden size 4,096 (o_proj's output); the norm's eps
const hidden: c_int = 4096;
const heads: c_int = 64;
const head_dim: c_int = 128;
const width: c_int = heads * head_dim;
const eps_v: f32 = 1e-6;
/// halfwords a 16 x 16 tile at K 6
const n_k6: c_int = 96;

/// The run-path bits the checks read: KDA_NORM_PREP's and LAYER_WHAT's
const prep_bit: u64 = @as(u64, 1) << @intFromEnum(ran.Path.KDA_NORM_PREP);
const what_bit: u64 = @as(u64, 1) << @intFromEnum(ran.Path.LAYER_WHAT);

fn marked(bit: u64) bool {
    return ran.ranBits() & bit != 0;
}

fn setIf(v: bool) []const u8 {
    return if (v) "set" else "absent";
}

fn setText(bit: u64) []const u8 {
    return setIf(marked(bit));
}

fn yesNo(v: bool) []const u8 {
    return if (v) "yes" else "no";
}

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

/// A copy of host words as an MLX array
fn hostArray(comptime T: type, data: []const T, shape: []const c_int, dt: mlx.mlx_dtype) !Array {
    const a = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dt);
    if (a.ctx == null) return error.MlxError;
    return a;
}

/// A new handle on `a` (as the loader's `keep` makes a Linear's)
fn refOf(a: Array) !Array {
    var h = mlx.mlx_array_new();
    errdefer free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

/// `a` made contiguous and evaluated, for reading its words on the host
fn evaluated(a: Array, s: Stream) !Array {
    var c = mlx.mlx_array_new();
    errdefer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word): equal buffers first, else word
/// by word
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
    if (std.mem.eql(u8, pa[0 .. n * isz], pb[0 .. n * isz])) return 0;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(!std.mem.eql(u8, pa[i * isz ..][0..isz], pb[i * isz ..][0..isz]));
    return d;
}

// ── operands as the loader keeps them ──

/// A `Linear` as the copied layers hold it and LAYER_WHAT's `begin` reads it: the trellis (w), suh (s), svh (b) and
/// the EXL3 flag
const Lin = struct { w: Array = .{ .ctx = null }, s: Array = .{ .ctx = null }, b: Array = .{ .ctx = null }, exl3: bool = false };

fn freeLin(l: Lin) void {
    free(l.w);
    free(l.s);
    free(l.b);
}

/// The trunk's layer with the fields `begin` reads: a KDA mixer whose one EXL3 linear is o_proj (q|k|v, f and g not
/// EXL3 here, so the scope holds o_proj alone) and no MLP of the scope's; the other kinds keep the trunk's shape for
/// `begin`'s switches
const KdaW = struct { qkv: Lin = .{}, qkv_parts: [3]Lin = @splat(.{}), a: Lin = .{}, z: Lin = .{}, out: Lin };
const MlaW = struct { q: Lin = .{}, q_a: Lin = .{}, q_b: Lin = .{}, kv_a: Lin = .{}, o: Lin = .{} };
const DenseW = struct { gate: Lin = .{}, up: Lin = .{}, down: Lin = .{} };
const LayerW = struct {
    mixer: union(enum) { kda: KdaW, dsa: void, mla: MlaW },
    mlp: union(enum) { dense: DenseW, moe: void },
};

/// o_proj as the loader keeps it, 8192 -> 4096: random trellis words viewed as I16 [512, 256, 96] (the pack stores
/// I16), suh [8192] and svh [4096] in F16, each through a handle of its own, evaluated. svh is (0.5 + u) / 90.5 (about
/// 1 / sqrt(8192)), so the outputs stay near 1.
fn oProjLin(s: Stream, rnd: std.Random) !Lin {
    const pa = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(width, 16), @divExact(hidden, 16), n_k6 };
    const words = try pa.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer pa.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    const u = try hostArray(u16, words, &shape, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    defer free(w);
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    const su = try pa.alloc(f16, @intCast(width));
    defer pa.free(su);
    for (su) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const sv = try pa.alloc(f16, @intCast(hidden));
    defer pa.free(sv);
    for (sv) |*v| v.* = @floatCast((0.5 + rnd.float(f32)) / 90.5);
    const suh = try hostArray(f16, su, &.{width}, .float16);
    defer free(suh);
    const svh = try hostArray(f16, sv, &.{hidden}, .float16);
    defer free(svh);
    const kt = try refOf(w);
    errdefer free(kt);
    const ksu = try refOf(suh);
    errdefer free(ksu);
    const ksv = try refOf(svh);
    errdefer free(ksv);
    for ([_]Array{ kt, ksu, ksv }) |t| try mlx.check(mlx.mlx_array_eval(t));
    return .{ .w = kt, .s = ksu, .b = ksv, .exl3 = true };
}

/// bf16 of `shape`: normal draws times `scale`, every fifth row of the last axis 16 times larger when `spiky`, evaluated
fn randBf16(s: Stream, rnd: std.Random, shape: []const c_int, scale: f32, spiky: bool) !Array {
    const pa = std.heap.page_allocator;
    var n: usize = 1;
    for (shape) |e| n *= @intCast(e);
    const row: usize = @intCast(shape[shape.len - 1]);
    const xf = try pa.alloc(f32, n);
    defer pa.free(xf);
    for (xf, 0..) |*e, i| e.* = rnd.floatNorm(f32) * scale * @as(f32, if (spiky and (i / row) % 5 == 4) 16 else 1);
    const f = try hostArray(f32, xf, shape, .float32);
    defer free(f);
    var x = mlx.mlx_array_new();
    errdefer free(x);
    try mlx.check(mlx.mlx_astype(&x, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}

/// One span's norm-gate operands: y [1, S, 64, 128] (the recurrence's output) and z [1, S, 8192] (g's output, the
/// gate), as the copied kda.zig hands them to the norm-gate
const Span = struct {
    y: Array = .{ .ctx = null },
    z: Array = .{ .ctx = null },

    fn deinit(sp: Span) void {
        free(sp.y);
        free(sp.z);
    }
};

fn spanOf(s: Stream, rnd: std.Random, rows: c_int) !Span {
    const y = try randBf16(s, rnd, &.{ 1, rows, heads, head_dim }, 1.0, true);
    errdefer free(y);
    const z = try randBf16(s, rnd, &.{ 1, rows, width }, 2.0, true);
    return .{ .y = y, .z = z };
}

/// What the copied kda.zig passes with every span of the layer: norm_w [128] bf16, the eps it makes (f32 0-d) and
/// o_proj
const Layer = struct { norm_w: Array, eps: Array, o: Lin };

// ── the two paths ──

/// The served path (the oracle), as the copied kda.zig runs its norm-gate site at a prompt span: the host's norm-gate
/// (`gdnNormGateFusedTab`, the sigmoid gate, the table at every width as the layer passes it with the packed prework),
/// then o_proj's `linear` (src/ling_exl3.zig: per call outside a scope, over the scope's kept W_hat in one); null where
/// the norm-gate declines
fn servedPath(s: Stream, sp: Span, ly: Layer) !?Array {
    const sh = mlx.getShape(sp.y);
    const flat = (try decode_fused.gdnNormGateFusedTab(s, sp.y, sp.z, 0, width, ly.norm_w, ly.eps, false, heads, head_dim, sh[0], sh[1], true)) orelse return null;
    defer free(flat);
    return try ling_exl3.linear(s, flat, ly.o.w, ly.o.s, ly.o.b);
}

/// The plugin's own call (src/ling_kda_norm_prep.zig `oProj`), with the copied kda.zig's operands; null where it
/// declines
fn prepCall(s: Stream, sp: Span, ly: Layer) !?Array {
    return norm_prep.oProj(s, .{ .y = sp.y, .z = sp.z, .z_off = 0, .z_stride = @intCast(width), .norm_w = ly.norm_w, .eps = ly.eps }, &ly.o);
}

/// The scope's kept W_hats and counts the same in two reads of LAYER_WHAT's stats: nothing kept, decoded or read
/// between them
fn unchanged(a: lwhat.Stats, b: lwhat.Stats) bool {
    return a.kept == b.kept and std.mem.eql(u32, &a.scope.decoded, &b.scope.decoded) and std.mem.eql(u32, &a.scope.read, &b.scope.read);
}

/// The planted fault (a red check, never served): the launch's text (turboexl3's `kdanorm.text_normprep`) with norm_w
/// read at the next channel, turboexl3's own red check's fault
const plant_what = "norm_w read at the next channel";
const plant_needle = "norm_w[c] * T(xc * inv)";
const plant_with = "norm_w[(c + 1u) % uint(DV)] * T(xc * inv)";

/// The planted kernel: the launch's text with the fault, its operands and output, the prepare's header, as turboexl3
/// makes its own; null where the needle is not once in the text (the plant would change nothing)
fn plantedKernel() !?mlx.mlx_fast_metal_kernel {
    const src: []const u8 = kdanorm.text_normprep;
    if (std.mem.count(u8, src, plant_needle) != 1) return null;
    const ca = std.heap.c_allocator;
    const planted = try std.mem.replaceOwned(u8, ca, src, plant_needle, plant_with);
    defer ca.free(planted);
    const planted_z = try ca.dupeSentinel(u8, planted, 0);
    defer ca.free(planted_z);
    const iv = mlx.mlx_vector_string_new_data(&kdanorm.normprep_in, kdanorm.normprep_in.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&kdanorm.normprep_out, kdanorm.normprep_out.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("turboexl3_ling_test_kdanorm_prep_nw_next", iv, ov, planted_z.ptr, t3.decode.header.ptr, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const Counts = struct {
    lines: usize = 0,
    failed: usize = 0,
    reds: usize = 0,
    words: usize = 0,

    fn check(c: *Counts, ok: bool, red: bool, comptime fmt: []const u8, args: anytype) void {
        c.lines += 1;
        c.failed += @intFromBool(!ok);
        c.reds += @intFromBool(ok and red);
        std.debug.print(tag ++ " " ++ fmt ++ " {s}\n", args ++ .{if (ok) "PASS" else "FAIL"});
    }
};

test "turboexl3_ling KDA_NORM_PREP (L5): the plugin's call, with no option set, writes o_proj's prepared rows in the norm-gate's launch, the served words over LAYER_WHAT's kept W_hat at 4,096 and 1,965 rows, with the scope's counts and the run-path mark; it declines, with nothing marked, decoded or read, at 31 rows, outside a scope and with pg2 on, and the served path after a decline gives its words; a planted fault is caught (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    lwhat.end();
    defer lwhat.end();
    defer ran.clearForTests();
    defer kdanorm.kernel_override.get().* = null;
    // pg2 off, as served (LAYER_WHAT's matmul path); the pg2 decline sets it on for its one call
    const pg_saved = t3.pgemm.mode.get().*;
    defer t3.pgemm.mode.get().* = pg_saved;
    t3.pgemm.mode.get().* = .off;
    var prng = std.Random.DefaultPrng.init(0x15A0_D1F5);
    const rnd = prng.random();
    var c: Counts = .{};
    const o = try oProjLin(s, rnd);
    defer freeLin(o);
    const norm_w = try randBf16(s, rnd, &.{head_dim}, 1.0, false);
    defer free(norm_w);
    const eps = mlx.mlx_array_new_float(eps_v);
    defer free(eps);
    const ly: Layer = .{ .norm_w = norm_w, .eps = eps, .o = o };
    const layer: LayerW = .{ .mixer = .{ .kda = .{ .out = o } }, .mlp = .moe };
    const k_o = @intFromEnum(lwhat.Family.o);
    // o_proj's Dense at the norm-gate's output (`linear`'s own), for the paths the lines name
    const dn = try t3.denseOf(.{ .trellis = o.w, .suh = o.s, .svh = o.b }, @intCast(width), .bfloat16, .mul1);

    // LAYER_WHAT's kept W_hat: a layer's scope around spans of 4,096 and 1,965 rows, each call against the per-call
    // served path, then the scope's counts and the pass's mark
    {
        defer _ = mlx.mlx_clear_cache();
        const spans = [_]c_int{ 4096, 1965 };
        var ins: [spans.len]Span = @splat(.{});
        defer for (ins) |sp| sp.deinit();
        var wants: [spans.len]?Array = @splat(null);
        defer for (wants) |w| if (w) |a| free(a);
        var gots: [spans.len]?Array = @splat(null);
        defer for (gots) |g| if (g) |a| free(a);
        var marks: [spans.len]bool = @splat(false);
        for (spans, &ins, &wants) |rows, *sp, *w| {
            sp.* = try spanOf(s, rnd, rows);
            w.* = try servedPath(s, sp.*, ly);
        }
        lwhat.passBegin();
        lwhat.begin(&layer);
        for (ins, &gots, &marks) |sp, *g, *mk| {
            ran.clearForTests();
            g.* = try prepCall(s, sp, ly);
            mk.* = marked(prep_bit);
        }
        const in_scope = lwhat.stats();
        lwhat.end();
        ran.clearForTests();
        lwhat.passEnd();
        for (spans, wants, gots, marks) |rows, w, g, mk| {
            const want = w orelse {
                c.check(false, false, "kept W_hat rows={d}: the served norm-gate declined, so there is no oracle", .{rows});
                continue;
            };
            const got = g orelse {
                c.check(false, false, "kept W_hat rows={d} (o_proj's path {s}): the call declined", .{ rows, @tagName(dn.pathFor(@intCast(rows))) });
                continue;
            };
            const d = try differing(got, want, s);
            const n = mlx.mlx_array_size(want);
            c.words += n;
            c.check(d == 0 and mk, false, "kept W_hat rows={d}: o_proj's output differing={d}/{d} against the per-call served path (the norm-gate, then o_proj's linear); KDA_NORM_PREP {s}", .{ rows, d, n, setIf(mk) });
        }
        const counts_ok = in_scope.open and in_scope.held == 1 and in_scope.kept == 1 and in_scope.scope.decoded[k_o] == 1 and
            in_scope.scope.read[k_o] == spans.len and marked(what_bit);
        c.check(counts_ok, false, "kept W_hat: the scope held {d} and kept {d} (want 1 and 1); o_proj decoded {d} and read {d} times (want 1 and {d}); LAYER_WHAT's pass mark {s}", .{ in_scope.held, in_scope.kept, in_scope.scope.decoded[k_o], in_scope.scope.read[k_o], spans.len, setText(what_bit) });
    }

    // the declines: no run-path name set, and in a scope nothing kept, decoded or read
    {
        defer _ = mlx.mlx_clear_cache();
        // 31 rows in a scope: the standard request's last span, o_proj's register path (`linear` makes no W_hat there)
        {
            const sp = try spanOf(s, rnd, 31);
            defer sp.deinit();
            lwhat.begin(&layer);
            defer lwhat.end();
            ran.clearForTests();
            const before = lwhat.stats();
            const got = try prepCall(s, sp, ly);
            defer if (got) |a| free(a);
            const after = lwhat.stats();
            c.check(got == null and !marked(prep_bit) and unchanged(before, after), false, "decline rows=31 in a scope (o_proj's path {s}): declined {s}, KDA_NORM_PREP {s}; kept {d} -> {d}, o_proj decoded {d} -> {d} and read {d} -> {d} (want no change)", .{ @tagName(dn.pathFor(31)), yesNo(got == null), setText(prep_bit), before.kept, after.kept, before.scope.decoded[k_o], after.scope.decoded[k_o], before.scope.read[k_o], after.scope.read[k_o] });
        }
        // 4,096 rows outside a scope: no kept W_hat for the call
        {
            const sp = try spanOf(s, rnd, 4096);
            defer sp.deinit();
            lwhat.end();
            ran.clearForTests();
            const got = try prepCall(s, sp, ly);
            defer if (got) |a| free(a);
            c.check(got == null and !marked(prep_bit), false, "decline rows=4096 outside a scope: declined {s}, KDA_NORM_PREP {s}", .{ yesNo(got == null), setText(prep_bit) });
        }
        // pg2 on, in a scope at 4,096 rows: LAYER_WHAT's matmul path is pg2 off; then the served path in the same
        // scope, with pg2 off as served, gives the per-call words
        {
            const sp = try spanOf(s, rnd, 4096);
            defer sp.deinit();
            const want = try servedPath(s, sp, ly);
            defer if (want) |a| free(a);
            lwhat.begin(&layer);
            defer lwhat.end();
            ran.clearForTests();
            const before = lwhat.stats();
            t3.pgemm.mode.get().* = .on;
            const got = prepCall(s, sp, ly) catch |e| {
                t3.pgemm.mode.get().* = .off;
                return e;
            };
            t3.pgemm.mode.get().* = .off;
            defer if (got) |a| free(a);
            const after = lwhat.stats();
            c.check(got == null and !marked(prep_bit) and unchanged(before, after), false, "decline rows=4096 with pg2 on, in a scope: declined {s}, KDA_NORM_PREP {s}; kept {d} -> {d}, o_proj decoded {d} -> {d} and read {d} -> {d} (want no change)", .{ yesNo(got == null), setText(prep_bit), before.kept, after.kept, before.scope.decoded[k_o], after.scope.decoded[k_o], before.scope.read[k_o], after.scope.read[k_o] });
            // the served path after it, in the scope: the norm-gate, then o_proj's `linear` over the scope's W_hat
            const fallback = try servedPath(s, sp, ly);
            defer if (fallback) |a| free(a);
            const in_scope = lwhat.stats();
            const by_scope = in_scope.scope.decoded[k_o] == 1 and in_scope.scope.read[k_o] == 1;
            if (want != null and fallback != null) {
                const d = try differing(fallback.?, want.?, s);
                const n = mlx.mlx_array_size(want.?);
                c.words += n;
                c.check(by_scope and d == 0, false, "after the pg2-on decline, in the scope with pg2 off: the served path decoded {d} and read {d} (want 1 and 1) and gave the per-call words: differing={d}/{d}", .{ in_scope.scope.decoded[k_o], in_scope.scope.read[k_o], d, n });
            } else c.check(false, false, "after the pg2-on decline, in the scope with pg2 off: the served norm-gate declined (per call {s}, in the scope {s}), so there is no oracle", .{ yesNo(want != null), yesNo(fallback != null) });
        }
    }

    // red: the planted fault over the kept W_hat at 4,096 rows, against the served path's words
    {
        defer _ = mlx.mlx_clear_cache();
        const planted = try plantedKernel();
        defer if (planted) |k| {
            _ = mlx.mlx_fast_metal_kernel_free(k);
        };
        const sp = try spanOf(s, rnd, 4096);
        defer sp.deinit();
        const want = try servedPath(s, sp, ly);
        defer if (want) |a| free(a);
        lwhat.begin(&layer);
        defer lwhat.end();
        if (planted == null) {
            c.check(false, true, "red: planted {s}: its needle is not once in the launch's text, so the plant would change nothing", .{plant_what});
        } else if (want == null) {
            c.check(false, true, "red: planted {s}: the served norm-gate declined, so there is no oracle", .{plant_what});
        } else {
            kdanorm.kernel_override.get().* = planted.?;
            const got = prepCall(s, sp, ly) catch |e| {
                kdanorm.kernel_override.get().* = null;
                return e;
            };
            kdanorm.kernel_override.get().* = null;
            if (got) |g| {
                defer free(g);
                const d = try differing(g, want.?, s);
                c.check(d > 0, true, "red: planted {s}, at 4,096 rows over the kept W_hat: o_proj's output differing={d}/{d} against the served path (want > 0)", .{ plant_what, d, mlx.mlx_array_size(want.?) });
            } else c.check(false, true, "red: planted {s}: the call declined", .{plant_what});
        }
    }

    // 2 kept spans and the scope's counts; 3 declines; the served path after the pg2-on decline; 1 planted fault
    const want_lines: usize = 3 + 3 + 1 + 1;
    const want_reds: usize = 1;
    std.debug.print(tag ++ " DONE lines={d} failed={d} reds={d} words={d} {s}\n", .{ c.lines, c.failed, c.reds, c.words, if (c.failed == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try std.testing.expectEqual(want_lines, c.lines);
    try std.testing.expectEqual(want_reds, c.reds);
    try std.testing.expectEqual(@as(usize, 0), c.failed);
}
