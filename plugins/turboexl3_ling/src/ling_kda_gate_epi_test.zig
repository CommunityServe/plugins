//! KDA_GATE_EPI's byte test (L6, S52, built in: no option turns it on or off; GPU, `MLX_SERVE_LING_GPU_TESTS=1` runs
//! it, else it skips). The plugin's own call is src/ling_kdaproj.zig `projPromptGate`, with no option set: a KDA
//! layer's q|k|v, f and g of one input in turboexl3's `ling.kdaProjPromptGate` (form G0), or `ling.kdaProjPromptWGate`
//! over LAYER_WHAT's kept W_hats, with f's output the bounded gate g (f32). The oracle is the path the copied KDA layer
//! (plugins/glm5_next/kda.zig) serves at prompt rows: `projPrompt`, then the gate in one launch over its f output
//! (sdk/blocks/kda/ling.zig `gate`, else `kdaGateChain`), at the bound -5.
//! - Operands as the loader keeps them, at Ling-3.1-flash's KDA shapes: q|k|v 4096 -> 24576, f and g 4096 -> 8192 (K 6,
//!   random I16 trellis words, F16 suh and svh, each through a handle of its own), A_log (one value a value head,
//!   expanded per key channel) and dt_bias f32 [8192]; x bf16 [1, S, 4096], every fifth row 16 times larger.
//! - Per call, no scope open, at S 33 (the first rows past the register path), 512, 1,965 (the standard request's tail
//!   span, which ends inside a tile) and 4,096 (a full span): q|k|v and z against `projPrompt`'s and g against the served
//!   gate's, every word, g f32; the run-path names KDA_GATE_EPI and KDA_PROJ_PROMPT set (P4's launch ran, with the gate
//!   in its store loop).
//! - LAYER_WHAT's kept W_hats: a layer's scope (`begin`) around spans of 4,096 and 1,965 rows, each against the per-call
//!   path's words; the scope's counts (q|k|v, f and g decoded once, read by both spans) and the pass's LAYER_WHAT mark.
//! - The decline fallback: at 31 rows (the standard request's last span) and where turboexl3 refuses the gate (A_log in
//!   bf16, A_log or dt_bias of 4,096 channels, the bound 0), the call declines with no run-path name set; in a scope a
//!   refused call decodes and reads nothing, and the served path after it (`projPrompt` over the scope's W_hats, then
//!   the served gate) gives the per-call path's words.
//! - The stand-in for a: where the gate was made in the projection's launch, the copied kda.zig hands the packed prework
//!   (decode_fused `gdnPreworkFused`, gate off, on its table, as it calls it at prompt rows) a new reference of q|k|v in
//!   a's place. The prework gives the same q, k, v, conv window and beta with it as with the f projection, every word,
//!   at 33, 1,965 and 4,096 rows over a warm window and 4,096 over a cold one.
//! - Red checks, which show the test can fail: three faults planted in G0's store loop through turboexl3's
//!   `gate_kernel_override` (f's word not rounded as P4 stores it, dt_bias of the next channel, the sigmoid's exp on the
//!   fast path), each on the per-call and the kept-W_hat call, must give other g words than the served gate's; so must a
//!   bound of -4 (which the gate takes) and another layer's A_log. The four refusals above count as red checks too.
//! Each check prints a `TURBOEXL3_LING_KDA_GATE_EPI` line ending in PASS or FAIL; the DONE line counts them.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const kdaGateChain = @import("mlx_serve/sdk/blocks/kda/gate.zig").kdaGateChain;
const decode_fused = @import("mlx_serve/sdk/blocks/gdn/decode_fused.zig");
const ling_k = @import("mlx_serve/sdk/blocks/kda/ling.zig");
const ling_kdaproj = @import("ling_kdaproj.zig");
const lwhat = @import("ling_layer_what.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const GateIn = ling_kdaproj.GateIn;
const pf = t3.ling.kdaproj_pf;

const tag = "TURBOEXL3_LING_KDA_GATE_EPI";
/// Ling-3.1-flash's KDA layer: 64 heads of 128 keys and values (q, k and v 8,192 wide each), a 4-tap conv over q|k|v,
/// the bound -5 (the config's kda_gate_lower_bound)
const hidden: c_int = 4096;
const heads: c_int = 64;
const head_dim: c_int = 128;
const width: c_int = heads * head_dim;
const conv_dim: c_int = 3 * width;
const bound: f32 = -5.0;
const qk_eps: f32 = 1e-6;
/// halfwords a 16 x 16 tile at K 6
const n_k6: c_int = 96;

/// The run-path bits the checks read: KDA_GATE_EPI's, KDA_PROJ_PROMPT's and LAYER_WHAT's
const gate_bit: u64 = @as(u64, 1) << @intFromEnum(ran.Path.KDA_GATE_EPI);
const prompt_bit: u64 = @as(u64, 1) << @intFromEnum(ran.Path.KDA_PROJ_PROMPT);
const what_bit: u64 = @as(u64, 1) << @intFromEnum(ran.Path.LAYER_WHAT);

fn marked(bit: u64) bool {
    return ran.ranBits() & bit != 0;
}

fn setText(bit: u64) []const u8 {
    return if (marked(bit)) "set" else "absent";
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

/// A new handle on `a` (as the loader's `keep` makes a Linear's, and the copied kda.zig's `ref`)
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

/// The trunk's layer with the fields `begin` reads: a KDA mixer (q|k|v joined as the pack ships it, f as `a`, g as `z`,
/// no o_proj, so the scope holds those three) and no MLP of the scope's; the other kinds keep the trunk's shape for
/// `begin`'s switches
const KdaW = struct { qkv: Lin, qkv_parts: [3]Lin = @splat(.{}), a: Lin, z: Lin, out: Lin = .{} };
const MlaW = struct { q: Lin = .{}, q_a: Lin = .{}, q_b: Lin = .{}, kv_a: Lin = .{}, o: Lin = .{} };
const DenseW = struct { gate: Lin = .{}, up: Lin = .{}, down: Lin = .{} };
const LayerW = struct {
    mixer: union(enum) { kda: KdaW, dsa: void, mla: MlaW },
    mlp: union(enum) { dense: DenseW, moe: void },
};

/// A projection 4096 -> `out_dim` as the loader keeps it: random trellis words viewed as I16 [256, out_dim / 16, 96]
/// (the pack stores I16), suh [4096] and svh [out_dim] in F16, each through a handle of its own, evaluated. svh is
/// (0.5 + u) / 64, so the outputs stay near 1 (16 on the large rows) and the gate sees both sigmoid arms.
fn randLin(s: Stream, rnd: std.Random, out_dim: c_int) !Lin {
    const pa = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(hidden, 16), @divExact(out_dim, 16), n_k6 };
    const words = try pa.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer pa.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    const u = try hostArray(u16, words, &shape, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    defer free(w);
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    const su = try pa.alloc(f16, @intCast(hidden));
    defer pa.free(su);
    for (su) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const sv = try pa.alloc(f16, @intCast(out_dim));
    defer pa.free(sv);
    for (sv) |*v| v.* = @floatCast((0.5 + rnd.float(f32)) / 64.0);
    const suh = try hostArray(f16, su, &.{hidden}, .float16);
    defer free(suh);
    const svh = try hostArray(f16, sv, &.{out_dim}, .float16);
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

/// A KDA layer's A_log as the loader keeps it: log(1 + 15 u) for each value head, expanded per key channel, f32 [8192]
fn aLogOf(rnd: std.Random) !Array {
    const pa = std.heap.page_allocator;
    const v = try pa.alloc(f32, @intCast(width));
    defer pa.free(v);
    const hd: usize = @intCast(head_dim);
    const nh: usize = @intCast(heads);
    for (0..nh) |h| @memset(v[h * hd ..][0..hd], @log(1.0 + 15.0 * rnd.float(f32)));
    return hostArray(f32, v, &.{width}, .float32);
}

/// f32 [n]: normal draws times `scale` (dt_bias, f32 [8192] as the loader keeps it, and the refusals' other lengths)
fn normalF32(rnd: std.Random, n: c_int, scale: f32) !Array {
    const pa = std.heap.page_allocator;
    const v = try pa.alloc(f32, @intCast(n));
    defer pa.free(v);
    for (v) |*e| e.* = rnd.floatNorm(f32) * scale;
    return hostArray(f32, v, &.{n}, .float32);
}

/// bf16 of `shape`: normal draws times `scale`, every fifth row (the next-to-last axis) times 16 more when `spiky`,
/// evaluated
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

/// x [1, rows, 4096] bf16, the layer's input: normal draws, every fifth row 16 times larger
fn input(s: Stream, rnd: std.Random, rows: c_int) !Array {
    return randBf16(s, rnd, &.{ 1, rows, hidden }, 1.0, true);
}

/// A 0-dim bf16 scalar (the layer's q_scale and k_scale), evaluated
fn scalarBf16(s: Stream, v: f32) !Array {
    const raw = mlx.mlx_array_new_float(v);
    defer free(raw);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// bf16 zeros of `shape` (a cold layer's window, the copied kda.zig's `zero_conv`), evaluated
fn zerosBf16(s: Stream, shape: []const c_int) !Array {
    var z = mlx.mlx_array_new();
    errdefer free(z);
    try mlx.check(mlx.mlx_zeros(&z, shape.ptr, shape.len, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(z));
    return z;
}

// ── the two paths ──

/// A prompt projection's outputs: q|k|v, the gate g (f32) and z
const Out = struct {
    qkv: Array,
    g: Array,
    z: Array,

    fn deinit(o: Out) void {
        free(o.qkv);
        free(o.g);
        free(o.z);
    }
};

/// The gate the copied kda.zig serves at prompt rows over f's output `a`: the bounded gate in one launch
/// (sdk/blocks/kda/ling.zig `gate`), else `kdaGateChain`
fn servedGate(s: Stream, a: Array, gate: GateIn) !Array {
    if (try ling_k.gate(s, gate.A_log, a, gate.dt_bias, gate.bound)) |g1| return g1;
    return kdaGateChain(gate.A_log, a, gate.dt_bias, gate.bound, s);
}

/// The served path (the oracle): `projPrompt`, then the served gate over its f output; null where `projPrompt` declines
fn servedPath(s: Stream, x: Array, la: *const KdaW, gate: GateIn) !?Out {
    const ys = (try ling_kdaproj.projPrompt(s, x, &la.qkv, &la.a, &la.z)) orelse return null;
    defer free(ys[1]);
    const g1 = servedGate(s, ys[1], gate) catch |e| {
        free(ys[0]);
        free(ys[2]);
        return e;
    };
    return .{ .qkv = ys[0], .g = g1, .z = ys[2] };
}

/// The plugin's own call (src/ling_kdaproj.zig `projPromptGate`); null where it declines
fn gateCall(s: Stream, x: Array, la: *const KdaW, gate: GateIn) !?Out {
    const ys = (try ling_kdaproj.projPromptGate(s, x, &la.qkv, &la.a, &la.z, gate)) orelse return null;
    return .{ .qkv = ys[0], .g = ys[1], .z = ys[2] };
}

/// Each output's differing words, and the words compared
const D3 = struct {
    qkv: usize,
    g: usize,
    z: usize,
    n: usize,

    fn same(d: D3) bool {
        return d.qkv == 0 and d.g == 0 and d.z == 0;
    }
};

fn diff3(s: Stream, got: Out, want: Out) !D3 {
    return .{
        .qkv = try differing(got.qkv, want.qkv, s),
        .g = try differing(got.g, want.g, s),
        .z = try differing(got.z, want.z, s),
        .n = mlx.mlx_array_size(want.qkv) + mlx.mlx_array_size(want.g) + mlx.mlx_array_size(want.z),
    };
}

/// The packed prework as the copied kda.zig calls it at prompt rows for Ling's layer (the gate off, the table at every
/// width), with `a` the operand in a's place
fn preworkOf(s: Stream, qkv: Array, a: Array, b: Array, window: Array, conv_w: Array, gate: GateIn, q_scale: Array, k_scale: Array, rows: c_int) !?decode_fused.GdnPrework {
    return decode_fused.gdnPreworkFused(s, .{
        .qkv = qkv,
        .qkv_off = 0,
        .qkv_stride = conv_dim,
        .b = b,
        .b_off = 0,
        .b_stride = heads,
        .a = a,
        .a_off = 0,
        .a_stride = heads,
        .A_log = gate.A_log,
        .dt_bias = gate.dt_bias,
        .conv_state = window,
        .conv_w = conv_w,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .qk_eps = qk_eps,
        .hk = heads,
        .hv = heads,
        .dk = head_dim,
        .dv = head_dim,
        .seq = rows,
        .batch = 1,
        .gate = false,
        .g_dtype = .bfloat16,
        .tab = true,
    });
}

fn freePre(p: decode_fused.GdnPrework) void {
    inline for (.{ p.q, p.k, p.v, p.conv_state, p.g, p.beta }) |a| free(a);
}

/// A fault planted in G0's store loop (turboexl3's `kdaproj_pf.text_pfg0`): `needle`, once in the text, replaced
const Plant = struct { name: [:0]const u8, what: []const u8, needle: []const u8, with: []const u8 };
const plants = [_]Plant{
    .{ .name = "turboexl3_ling_test_kdagate_epi_unrounded", .what = "f's word not rounded as P4 stores it", .needle = "const OUT_T yv = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))));", .with = "const float yv = float(half(v[j] * HAD_SCALE * float(svh[o])));" },
    .{ .name = "turboexl3_ling_test_kdagate_epi_dtb_next", .what = "dt_bias of the next channel", .needle = "static_cast<float>(dt_bias[o])", .with = "static_cast<float>(dt_bias[(o + 1u) % DIMC])" },
    .{ .name = "turboexl3_ling_test_kdagate_epi_fastexp", .what = "the sigmoid's exp on the fast path", .needle = "metal::precise::exp(metal::abs(sc))", .with = "metal::fast::exp(metal::abs(sc))" },
};

/// `pl`'s kernel: G0's text with the fault, G0's operands and outputs, pg2's header, as turboexl3 makes its own; null
/// where the needle is not once in the text (the plant would change nothing)
fn plantedKernel(pl: Plant) !?mlx.mlx_fast_metal_kernel {
    const src: []const u8 = pf.text_pfg0;
    if (std.mem.count(u8, src, pl.needle) != 1) return null;
    const ca = std.heap.c_allocator;
    const planted = try std.mem.replaceOwned(u8, ca, src, pl.needle, pl.with);
    defer ca.free(planted);
    const planted_z = try ca.dupeSentinel(u8, planted, 0);
    defer ca.free(planted_z);
    const iv = mlx.mlx_vector_string_new_data(&pf.pfg0_in, pf.pfg0_in.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&pf.pfg_out, pf.pfg_out.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(pl.name.ptr, iv, ov, planted_z.ptr, t3.pgemm.header.ptr, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// The plugin's call with `k` in the gate launch's place (turboexl3's `gate_kernel_override`, cleared after the call):
/// its g's words against `want`'s; null where the call declined
fn plantedDiff(s: Stream, x: Array, la: *const KdaW, gate: GateIn, k: mlx.mlx_fast_metal_kernel, want: Out) !?usize {
    pf.gate_kernel_override.get().* = k;
    const got = gateCall(s, x, la, gate) catch |e| {
        pf.gate_kernel_override.get().* = null;
        return e;
    };
    pf.gate_kernel_override.get().* = null;
    const o = got orelse return null;
    defer o.deinit();
    return try differing(o.g, want.g, s);
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

fn yesNo(v: bool) []const u8 {
    return if (v) "yes" else "no";
}

test "turboexl3_ling KDA_GATE_EPI (L6): the plugin's call makes the gate in its prompt projection's launch, the served gate's words with projPrompt's q|k|v and z, per call at 33 to 4,096 rows and over LAYER_WHAT's kept W_hats; it declines to the served path at 31 rows and where turboexl3 refuses the gate; the prework's stand-in for a changes no word; planted faults and wrong operands are caught (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    lwhat.end();
    defer lwhat.end();
    defer ran.clearForTests();
    defer pf.gate_kernel_override.get().* = null;
    var prng = std.Random.DefaultPrng.init(0x16E9_A7E5);
    const rnd = prng.random();
    var c: Counts = .{};
    const p_qkv = try randLin(s, rnd, 3 * width);
    defer freeLin(p_qkv);
    const p_f = try randLin(s, rnd, width);
    defer freeLin(p_f);
    const p_g = try randLin(s, rnd, width);
    defer freeLin(p_g);
    const la: KdaW = .{ .qkv = p_qkv, .a = p_f, .z = p_g };
    const a_log = try aLogOf(rnd);
    defer free(a_log);
    const dt_bias = try normalF32(rnd, width, 0.5);
    defer free(dt_bias);
    const gate: GateIn = .{ .A_log = a_log, .dt_bias = dt_bias, .bound = bound };

    // per call, no scope open: every word against the served path's, and the run-path names
    for ([_]c_int{ 33, 512, 1965, 4096 }) |rows| {
        defer _ = mlx.mlx_clear_cache();
        const x = try input(s, rnd, rows);
        defer free(x);
        const want = (try servedPath(s, x, &la, gate)) orelse {
            c.check(false, false, "per call rows={d}: projPrompt declined, so there is no oracle", .{rows});
            continue;
        };
        defer want.deinit();
        ran.clearForTests();
        const got = (try gateCall(s, x, &la, gate)) orelse {
            c.check(false, false, "per call rows={d}: the call declined", .{rows});
            continue;
        };
        defer got.deinit();
        const d = try diff3(s, got, want);
        c.words += d.n;
        const marks_ok = marked(gate_bit) and marked(prompt_bit);
        c.check(d.same() and marks_ok, false, "per call rows={d}: differing qkv={d} g={d} z={d} (of {d} words), g f32 {s}; the run-path names KDA_GATE_EPI {s}, KDA_PROJ_PROMPT {s}", .{ rows, d.qkv, d.g, d.z, d.n, yesNo(mlx.mlx_array_dtype(got.g) == .float32), setText(gate_bit), setText(prompt_bit) });
    }

    // LAYER_WHAT's kept W_hats: a layer's scope around two spans, each against the per-call served path
    {
        defer _ = mlx.mlx_clear_cache();
        const spans = [_]c_int{ 4096, 1965 };
        var xs = [_]Array{ .{ .ctx = null }, .{ .ctx = null } };
        defer for (xs) |x| free(x);
        var wants = [_]?Out{ null, null };
        defer for (wants) |w| if (w) |o| o.deinit();
        var gots = [_]?Out{ null, null };
        defer for (gots) |w| if (w) |o| o.deinit();
        for (spans, &xs, &wants) |rows, *x, *w| {
            x.* = try input(s, rnd, rows);
            w.* = try servedPath(s, x.*, &la, gate);
        }
        const layer: LayerW = .{ .mixer = .{ .kda = la }, .mlp = .moe };
        ran.clearForTests();
        lwhat.passBegin();
        lwhat.begin(&layer);
        for (xs, &gots) |x, *o| o.* = try gateCall(s, x, &la, gate);
        const in_scope = lwhat.stats();
        lwhat.end();
        lwhat.passEnd();
        for (spans, wants, gots) |rows, w, o| {
            const want = w orelse {
                c.check(false, false, "kept W_hats rows={d}: projPrompt declined, so there is no oracle", .{rows});
                continue;
            };
            const got = o orelse {
                c.check(false, false, "kept W_hats rows={d}: the call declined", .{rows});
                continue;
            };
            const d = try diff3(s, got, want);
            c.words += d.n;
            c.check(d.same() and marked(gate_bit), false, "kept W_hats rows={d}: differing qkv={d} g={d} z={d} (of {d} words) against the per-call served path; KDA_GATE_EPI {s}", .{ rows, d.qkv, d.g, d.z, d.n, setText(gate_bit) });
        }
        const k = @intFromEnum(lwhat.Family.kda);
        const counts_ok = in_scope.open and in_scope.held == 3 and in_scope.kept == 3 and in_scope.scope.decoded[k] == 3 and
            in_scope.scope.read[k] == 3 * spans.len and marked(what_bit);
        c.check(counts_ok, false, "kept W_hats: the scope held {d} and kept {d} (want 3 and 3); q|k|v, f and g decoded {d} and read {d} times (want 3 and {d}); LAYER_WHAT's pass mark {s}", .{ in_scope.held, in_scope.kept, in_scope.scope.decoded[k], in_scope.scope.read[k], 3 * spans.len, setText(what_bit) });
    }

    // the decline fallback: the register path's rows, and gate operands turboexl3 refuses
    var a_bf16 = mlx.mlx_array_new();
    defer free(a_bf16);
    try mlx.check(mlx.mlx_astype(&a_bf16, a_log, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(a_bf16));
    const short = try normalF32(rnd, hidden, 0.5);
    defer free(short);
    {
        const x = try input(s, rnd, 31);
        defer free(x);
        ran.clearForTests();
        const got = try gateCall(s, x, &la, gate);
        if (got) |o| o.deinit();
        c.check(got == null and !marked(gate_bit) and !marked(prompt_bit), false, "decline rows=31 (the register path's rows): declined {s}, the run-path names {s} and {s}; the served path runs each projection's linear", .{ yesNo(got == null), setText(gate_bit), setText(prompt_bit) });
    }
    {
        defer _ = mlx.mlx_clear_cache();
        const x = try input(s, rnd, 512);
        defer free(x);
        const Refusal = struct { what: []const u8, gate: GateIn };
        const refusals = [_]Refusal{
            .{ .what = "A_log in bf16", .gate = .{ .A_log = a_bf16, .dt_bias = dt_bias, .bound = bound } },
            .{ .what = "A_log of 4,096 channels", .gate = .{ .A_log = short, .dt_bias = dt_bias, .bound = bound } },
            .{ .what = "dt_bias of 4,096 channels", .gate = .{ .A_log = a_log, .dt_bias = short, .bound = bound } },
            .{ .what = "the bound 0", .gate = .{ .A_log = a_log, .dt_bias = dt_bias, .bound = 0.0 } },
        };
        for (refusals) |r| {
            ran.clearForTests();
            const got = try gateCall(s, x, &la, r.gate);
            if (got) |o| o.deinit();
            c.check(got == null and !marked(gate_bit) and !marked(prompt_bit), true, "red: refused at 512 rows, {s}: declined {s}, the run-path names {s} and {s}", .{ r.what, yesNo(got == null), setText(gate_bit), setText(prompt_bit) });
        }
    }
    {
        defer _ = mlx.mlx_clear_cache();
        const x = try input(s, rnd, 4096);
        defer free(x);
        const want = try servedPath(s, x, &la, gate);
        defer if (want) |o| o.deinit();
        const layer: LayerW = .{ .mixer = .{ .kda = la }, .mlp = .moe };
        lwhat.begin(&layer);
        ran.clearForTests();
        const before = lwhat.stats();
        const refused = try gateCall(s, x, &la, .{ .A_log = a_bf16, .dt_bias = dt_bias, .bound = bound });
        if (refused) |o| o.deinit();
        const after = lwhat.stats();
        // the served path after it, through the scope: projPrompt over the scope's W_hats, then the served gate
        const fallback = try servedPath(s, x, &la, gate);
        defer if (fallback) |o| o.deinit();
        const in_scope = lwhat.stats();
        lwhat.end();
        const k = @intFromEnum(lwhat.Family.kda);
        const untouched = refused == null and !marked(gate_bit) and after.kept == before.kept and
            after.scope.decoded[k] == before.scope.decoded[k] and after.scope.read[k] == before.scope.read[k];
        const by_scope = in_scope.scope.decoded[k] == 3 and in_scope.scope.read[k] == 3;
        if (want != null and fallback != null) {
            const d = try diff3(s, fallback.?, want.?);
            c.words += d.n;
            c.check(untouched and by_scope and d.same(), false, "decline in a scope (A_log in bf16, 4,096 rows): declined {s}, kept {d} -> {d}, decoded {d} -> {d}, read {d} -> {d} (want no change); the served path after it decoded {d} and read {d} (want 3 and 3) and gave the per-call words: differing qkv={d} g={d} z={d} (of {d} words)", .{ yesNo(refused == null), before.kept, after.kept, before.scope.decoded[k], after.scope.decoded[k], before.scope.read[k], after.scope.read[k], in_scope.scope.decoded[k], in_scope.scope.read[k], d.qkv, d.g, d.z, d.n });
        } else c.check(false, false, "decline in a scope (A_log in bf16, 4,096 rows): projPrompt declined (want {s}, after {s}), so there is no oracle", .{ yesNo(want != null), yesNo(fallback != null) });
    }

    // the stand-in for a: the packed prework at GATE 0 with a new reference of q|k|v in a's place, against a's
    {
        const conv_w = try randBf16(s, rnd, &.{ conv_dim, 4, 1 }, 0.5, false);
        defer free(conv_w);
        const q_scale = try scalarBf16(s, 1.0 / @as(f32, @floatFromInt(head_dim)));
        defer free(q_scale);
        const k_scale = try scalarBf16(s, @sqrt(1.0 / @as(f32, @floatFromInt(head_dim))));
        defer free(k_scale);
        const Case = struct { rows: c_int, cold: bool };
        for ([_]Case{ .{ .rows = 33, .cold = false }, .{ .rows = 1965, .cold = false }, .{ .rows = 4096, .cold = false }, .{ .rows = 4096, .cold = true } }) |cs| {
            defer _ = mlx.mlx_clear_cache();
            const qkv_in = try randBf16(s, rnd, &.{ 1, cs.rows, conv_dim }, 1.0, false);
            defer free(qkv_in);
            const a_in = try randBf16(s, rnd, &.{ 1, cs.rows, width }, 6.0, false);
            defer free(a_in);
            const b_in = try randBf16(s, rnd, &.{ 1, cs.rows, heads }, 8.0, false);
            defer free(b_in);
            const window = if (cs.cold) try zerosBf16(s, &.{ 1, 3, conv_dim }) else try randBf16(s, rnd, &.{ 1, 3, conv_dim }, 1.0, false);
            defer free(window);
            const stand_in = try refOf(qkv_in);
            defer free(stand_in);
            const cold_text: []const u8 = if (cs.cold) " cold" else "";
            const pre_a = (try preworkOf(s, qkv_in, a_in, b_in, window, conv_w, gate, q_scale, k_scale, cs.rows)) orelse {
                c.check(false, false, "stand-in for a rows={d}{s}: the prework declined with a", .{ cs.rows, cold_text });
                continue;
            };
            defer freePre(pre_a);
            const pre_b = (try preworkOf(s, qkv_in, stand_in, b_in, window, conv_w, gate, q_scale, k_scale, cs.rows)) orelse {
                c.check(false, false, "stand-in for a rows={d}{s}: the prework declined with the stand-in", .{ cs.rows, cold_text });
                continue;
            };
            defer freePre(pre_b);
            const dq = try differing(pre_b.q, pre_a.q, s);
            const dkk = try differing(pre_b.k, pre_a.k, s);
            const dv = try differing(pre_b.v, pre_a.v, s);
            const dw = try differing(pre_b.conv_state, pre_a.conv_state, s);
            const db = try differing(pre_b.beta, pre_a.beta, s);
            const n = mlx.mlx_array_size(pre_a.q) + mlx.mlx_array_size(pre_a.k) + mlx.mlx_array_size(pre_a.v) + mlx.mlx_array_size(pre_a.conv_state) + mlx.mlx_array_size(pre_a.beta);
            c.words += n;
            c.check(dq == 0 and dkk == 0 and dv == 0 and dw == 0 and db == 0, false, "stand-in for a rows={d}{s}: the prework at GATE 0 with a new reference of q|k|v in a's place, differing q={d} k={d} v={d} conv_window={d} beta={d} (of {d} words) against the f projection's", .{ cs.rows, cold_text, dq, dkk, dv, dw, db, n });
        }
    }

    // red: the planted faults, on the per-call call and over the kept W_hats, each against the served gate's words
    {
        defer _ = mlx.mlx_clear_cache();
        var kernels: [plants.len]?mlx.mlx_fast_metal_kernel = @splat(null);
        defer for (kernels) |kk| if (kk) |kern| {
            _ = mlx.mlx_fast_metal_kernel_free(kern);
        };
        for (plants, &kernels) |pl, *kk| kk.* = try plantedKernel(pl);
        const x = try input(s, rnd, 4096);
        defer free(x);
        const want = try servedPath(s, x, &la, gate);
        defer if (want) |o| o.deinit();
        const layer: LayerW = .{ .mixer = .{ .kda = la }, .mlp = .moe };
        for ([_]bool{ false, true }) |kept| {
            if (kept) lwhat.begin(&layer);
            defer if (kept) lwhat.end();
            const where: []const u8 = if (kept) "over the kept W_hats" else "per call";
            for (plants, kernels) |pl, kk| {
                const kern = kk orelse {
                    c.check(false, true, "red: planted {s}, {s}: its needle is not once in G0's text, so the plant would change nothing", .{ pl.what, where });
                    continue;
                };
                const w = want orelse {
                    c.check(false, true, "red: planted {s}, {s}: projPrompt declined, so there is no oracle", .{ pl.what, where });
                    continue;
                };
                const d = (try plantedDiff(s, x, &la, gate, kern, w)) orelse {
                    c.check(false, true, "red: planted {s}, {s}: the call declined", .{ pl.what, where });
                    continue;
                };
                c.check(d > 0, true, "red: planted {s}, {s} at 4,096 rows: g differing={d}/{d} against the served gate (want > 0)", .{ pl.what, where, d, mlx.mlx_array_size(w.g) });
            }
        }
    }

    // red: a wrong bound the gate takes and another layer's A_log, each against the served gate's words
    {
        defer _ = mlx.mlx_clear_cache();
        const a_log_other = try aLogOf(rnd);
        defer free(a_log_other);
        const x = try input(s, rnd, 512);
        defer free(x);
        const want = try servedPath(s, x, &la, gate);
        defer if (want) |o| o.deinit();
        const Wrong = struct { what: []const u8, gate: GateIn };
        const wrongs = [_]Wrong{
            .{ .what = "the bound -4 (the gate takes it)", .gate = .{ .A_log = a_log, .dt_bias = dt_bias, .bound = -4.0 } },
            .{ .what = "another layer's A_log", .gate = .{ .A_log = a_log_other, .dt_bias = dt_bias, .bound = bound } },
        };
        for (wrongs) |wr| {
            const w = want orelse {
                c.check(false, true, "red: {s}: projPrompt declined, so there is no oracle", .{wr.what});
                continue;
            };
            const got = (try gateCall(s, x, &la, wr.gate)) orelse {
                c.check(false, true, "red: {s}: the call declined", .{wr.what});
                continue;
            };
            defer got.deinit();
            const d = try diff3(s, got, w);
            c.check(d.g > 0 and d.qkv == 0 and d.z == 0, true, "red: {s} at 512 rows: g differing={d}/{d} against the served gate (want > 0), qkv={d} z={d} (want 0)", .{ wr.what, d.g, mlx.mlx_array_size(w.g), d.qkv, d.z });
        }
    }

    // 4 per-call row counts; 2 kept spans and the counts; 1 + 4 + 1 declines; 4 stand-in cases; 3 plants x 2 paths and 2
    // wrong operands
    const want_lines: usize = 4 + 3 + 6 + 4 + 8;
    const want_reds: usize = 4 + 6 + 2;
    std.debug.print(tag ++ " DONE lines={d} failed={d} reds={d} words={d} {s}\n", .{ c.lines, c.failed, c.reds, c.words, if (c.failed == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try std.testing.expectEqual(want_lines, c.lines);
    try std.testing.expectEqual(want_reds, c.reds);
    try std.testing.expectEqual(@as(usize, 0), c.failed);
}
