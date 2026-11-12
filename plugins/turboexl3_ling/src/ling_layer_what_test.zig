//! turboexl3_ling's layer-major layer's dense prefill linears over its W_hats (LAYER_WHAT, src/ling_layer_what.zig)
//! against the per-call path (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips).
//! - The oracle (A) is each linear's per-call path at a prompt span, no scope open, through the entry points the copied
//!   layers call (on KDA_PROJ_PROMPT's and SWIGLU_EPI's paths): a KDA layer's q|k|v, f and g by src/ling_kdaproj.zig
//!   `projPrompt` (each member's `ling_exl3.linear` where it declines, as kda.zig falls back), its o_proj by
//!   `ling_exl3.linear`; an MLA layer's q_a, q_b, kv_a and o_proj by `ling_exl3.linear`; a dense layer's MLP as
//!   trunk.zig's `denseMlp` runs it: the gate and the down by `ling_exl3.linear`, up's SwiGLU by `ling_swiglu_epi.up`.
//!   Each linear decodes its W_hat per call.
//! - The variant (B) is `hiddenSpans`' scope: the scope around the layer (`begin` with the layer, `end` after its spans;
//!   a MoE layer's MLP has no linear of the scope's), every span through the same entry points, which the scope serves.
//! - Operands as the loader keeps them: each projection an I16 trellis [in/16, out/16, n] with F16 suh and svh, each
//!   held through a handle of its own and evaluated (`Loader.keep`, `evalTrunk`), in a `Linear`'s fields (w, s, b,
//!   exl3) inside a layer of the trunk's shape (`mixer`, `mlp`), which `begin` reads as it reads the trunk's. Ling's
//!   shapes: a KDA layer's q|k|v 4096 -> 24576, f and g 4096 -> 8192, o_proj 8192 -> 4096; an MLA layer's q_a
//!   4096 -> 1536, q_b 1536 -> 12288, kv_a 4096 -> 640 (576 used), o_proj 8192 -> 4096 (K 6); a dense layer's MLP
//!   4096 -> 10240 -> 4096 (K 4). Random words, and with MLX_SERVE_LING_PACK=<pack dir> the pack's own layers 0 (KDA,
//!   dense MLP), 4 (KDA) and 7 (MLA), read as src/ling_weights.zig reads the shards.
//! - Spans in the layer-major pass's order at the standard request (4 of 4,096 rows, 1,965, 31), each linear's input
//!   bf16 [1, S, in], every fifth row 16 times larger. Layers 0, 4 and 7 one after another in one pass (`passBegin`,
//!   `passEnd`), so each scope decodes its own W_hats after the one before dropped its. Each layer's outputs are
//!   evaluated together after its scope ends.
//! - Checks: every word of every span's outputs, B against A; each scope's counts by family (each held linear's W_hat
//!   decoded once and read by the 5 wide spans; the 31-row span left to the chain); nothing held or kept after `end`,
//!   and the layer's linears not served then; the pass's counts and its run-path mark (src/ling_ran.zig). Red checks in
//!   the same run, which show that the test can fail: outside a scope a 64-row call is not served; layer 4's q|k|v, f
//!   and g over layer 0's W_hats (a wrong layer's) differ from their own words; layer 4's o_proj over layer 0's o_proj
//!   W_hat (what a W_hat kept past `end` would serve) differs from its own; layer 4's o_proj over its own W_hat gives its
//!   words and, with one bit of that W_hat flipped, does not; one flipped bit of an output counts as one differing word.
//! Each check prints a `TURBOEXL3_LING_LAYER_WHAT P3` line ending in PASS or FAIL.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const fused_act = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const ling_exl3 = @import("ling_exl3.zig");
const ling_kdaproj = @import("ling_kdaproj.zig");
const ling_weights = @import("ling_weights.zig");
const epi = @import("ling_swiglu_epi.zig");
const sl = @import("ling_shared_layer.zig");
const lwhat = @import("ling_layer_what.zig");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const Proj = lwhat.Proj;
const Family = lwhat.Family;
const Counts = lwhat.Counts;
const Source = @import("mlx_serve").abi.quant.Source;

const hidden: c_int = 4096;
/// halfwords a 16 x 16 tile: K 6 (the trunk's linears), K 4 (the dense MLP)
const n_k6: c_int = 96;
const n_k4: c_int = 64;
/// o_proj's input (64 heads x 128), q_b's (q_lora_rank), the dense MLP's width
const attn_width: c_int = 8192;
const q_lora: c_int = 1536;
const ffn_width: c_int = 10240;
/// The standard request's spans in the layer-major pass's order
const spans = [_]c_int{ 4096, 4096, 4096, 4096, 1965, 31 };
/// The spans the scope serves (past the register path's 32 rows)
const wide: u32 = 5;
const tag = "TURBOEXL3_LING_LAYER_WHAT P3";

fn gpu() !Stream {
    // the GPU tests' own variable (test-only): at 1 they run, at any other value or unset they skip
    const run_gpu = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(run_gpu), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

/// turboexl3's `ling.kdaProjPromptW` (the red check runs it over another layer's W_hats)
fn kdaOver(s: Stream, x: Array, p: [3]Proj, w: [3]Array) ![3]Array {
    return t3.ling.kdaProjPromptW(s, x, p, w, .mul1);
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// LAYER_WHAT's bit of the run-path line (src/ling_ran.zig), set by a pass's `passEnd` whose layers read a kept W_hat
/// past its decode (the bits are the process's: `markStart` clears them before each check)
fn marked() bool {
    return (ran.ranBits() & (@as(u64, 1) << @intFromEnum(ran.Path.LAYER_WHAT))) != 0;
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

fn freeProj(p: Proj) void {
    free(p.trellis);
    free(p.suh);
    free(p.svh);
}

/// A projection in_dim -> out_dim as the loader keeps it: random trellis words viewed as I16 [in_dim / 16,
/// out_dim / 16, n] (the pack stores I16), suh [in_dim] and svh [out_dim] in F16, each through a handle of its own,
/// evaluated. svh is `sv` x (0.5 + u): about 1 / sqrt(in_dim) keeps the outputs near 1 (16 on the large rows).
fn randProj(s: Stream, rnd: std.Random, in_dim: c_int, out_dim: c_int, n: c_int, sv: f32) !Proj {
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
    const kv = try keep(svh);
    errdefer free(kv);
    for ([_]Array{ kt, ks, kv }) |t| try mlx.check(mlx.mlx_array_eval(t));
    return .{ .trellis = kt, .suh = ks, .svh = kv };
}

/// The pack's projection `<base>` as the loader takes it (Loader.triple, then `keep` in Loader.lin): its trellis, suh
/// and svh as stored, its mul1 checked; each evaluated (the load's evalTrunk).
fn packBase(src: *const Source, base: []const u8) !Proj {
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
    for ([_]Array{ kt, ks, kv }) |t| try mlx.check(mlx.mlx_array_eval(t));
    return .{ .trellis = kt, .suh = ks, .svh = kv };
}

/// Model layer `layer`'s projection at the first of `leaves` the pack holds (`model.layers.<layer>.<leaf>`)
fn packProj(src: *const Source, layer: u32, leaves: []const []const u8) !Proj {
    var bb: [128]u8 = undefined;
    var nb: [160]u8 = undefined;
    for (leaves) |leaf| {
        const base = try std.fmt.bufPrint(&bb, "model.layers.{d}.{s}", .{ layer, leaf });
        if (src.tensor(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base})).ctx == null) continue;
        return packBase(src, base);
    }
    return error.MissingWeight;
}

/// x [1, rows, width] in bf16: normal draws, every fifth row times 16
fn input(s: Stream, rnd: std.Random, rows: c_int, width: c_int) !Array {
    const a = std.heap.page_allocator;
    const xf = try a.alloc(f32, @intCast(rows * width));
    defer a.free(xf);
    for (xf, 0..) |*v, i| v.* = rnd.floatNorm(f32) * @as(f32, if ((i / @as(usize, @intCast(width))) % 5 == 4) 16 else 1);
    const f = try hostArray(f32, xf, &.{ 1, rows, width }, .float32);
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

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word): equal buffers first (the PASS
/// case at the copy's speed), else word by word
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

/// `a` (16-bit words) with bit `bit_at` of its word `at` flipped, a new array of its shape and dtype (the red checks)
fn flipped(a: Array, at: usize, bit_at: u4, s: Stream) !Array {
    const c = try evaluated(a, s);
    defer free(c);
    if (mlx.mlx_array_itemsize(c) != 2) return error.BadExl3Dtype;
    const n = mlx.mlx_array_size(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const buf = try std.heap.page_allocator.alloc(u8, n * 2);
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, p[0 .. n * 2]);
    const word = std.mem.readInt(u16, buf[at * 2 ..][0..2], .little) ^ (@as(u16, 1) << bit_at);
    std.mem.writeInt(u16, buf[at * 2 ..][0..2], word, .little);
    const sh = mlx.getShape(c);
    const out = mlx.mlx_array_new_data(buf.ptr, sh.ptr, @intCast(sh.len), mlx.mlx_array_dtype(c));
    if (out.ctx == null) return error.MlxError;
    return out;
}

// ── a layer of the trunk's shape, as `begin` reads it ──

/// A `Linear` as `begin` reads the trunk's: the trellis (w), suh (s), svh (b), and the EXL3 flag
const Lin = struct { w: Array = .{ .ctx = null }, s: Array = .{ .ctx = null }, b: Array = .{ .ctx = null }, exl3: bool = false };
/// The trunk's `Kda` (q|k|v joined, as the pack ships it; b_proj plain), `Mla` (the low-rank query pair; kv_b and the
/// gate plain) and `Dense`, with the fields `begin` reads
const KdaW = struct { qkv: Lin, qkv_parts: [3]Lin = @splat(.{}), a: Lin, z: Lin, b: Lin = .{}, out: Lin };
const MlaW = struct { q: Lin = .{}, q_a: Lin, q_b: Lin, kv_a: Lin, kv_b: Lin = .{}, g: Lin = .{}, o: Lin };
const DenseW = struct { gate: Lin, up: Lin, down: Lin };
const LayerW = struct {
    /// The trunk's three mixer kinds; `dsa` (GLM's) is never built here, but keeps the trunk's shape, so `begin`'s
    /// `else` prong has a case
    mixer: union(enum) { kda: KdaW, dsa: void, mla: MlaW },
    mlp: union(enum) { dense: DenseW, moe: void },
    layer: u32,

    fn deinit(l: LayerW) void {
        switch (l.mixer) {
            .kda => |la| for ([_]Lin{ la.qkv, la.a, la.z, la.out }) |x| freeLin(x),
            .mla => |at| for ([_]Lin{ at.q_a, at.q_b, at.kv_a, at.o }) |x| freeLin(x),
            .dsa => {},
        }
        switch (l.mlp) {
            .dense => |dw| for ([_]Lin{ dw.gate, dw.up, dw.down }) |x| freeLin(x),
            .moe => {},
        }
    }
};

fn linOf(p: Proj) Lin {
    return .{ .w = p.trellis, .s = p.suh, .b = p.svh, .exl3 = true };
}

fn projOf(l: Lin) Proj {
    return .{ .trellis = l.w, .suh = l.s, .svh = l.b };
}

fn freeLin(l: Lin) void {
    free(l.w);
    free(l.s);
    free(l.b);
}

/// The kinds of layer the test runs: a dense KDA layer (the first 4), a MoE KDA layer, a MoE MLA layer
const Kind = enum { kda_dense, kda, mla };
/// The pass: layers 0, 4 and 7
const Spec = struct { layer: u32, kind: Kind };
const plan = [_]Spec{ .{ .layer = 0, .kind = .kda_dense }, .{ .layer = 4, .kind = .kda }, .{ .layer = 7, .kind = .mla } };

/// The pack's shards (MLX_SERVE_LING_PACK), opened once for the cases that read them
var pack_weights: ?*ling_weights.Weights = null;

fn kdaRand(s: Stream, rnd: std.Random) !KdaW {
    const qkv = try randProj(s, rnd, hidden, 24576, n_k6, 1.0 / 64.0);
    errdefer freeProj(qkv);
    const f = try randProj(s, rnd, hidden, 8192, n_k6, 1.0 / 64.0);
    errdefer freeProj(f);
    const g = try randProj(s, rnd, hidden, 8192, n_k6, 1.0 / 64.0);
    errdefer freeProj(g);
    const o = try randProj(s, rnd, attn_width, hidden, n_k6, 1.0 / 90.5);
    return .{ .qkv = linOf(qkv), .a = linOf(f), .z = linOf(g), .out = linOf(o) };
}

fn kdaPack(src: *const Source, layer: u32) !KdaW {
    const qkv = try packProj(src, layer, &.{"attention.qkv_proj"});
    errdefer freeProj(qkv);
    const f = try packProj(src, layer, &.{ "attention.f_proj", "attention.forget_gate.f_proj" });
    errdefer freeProj(f);
    const g = try packProj(src, layer, &.{"attention.g_proj"});
    errdefer freeProj(g);
    const o = try packProj(src, layer, &.{"attention.o_proj"});
    return .{ .qkv = linOf(qkv), .a = linOf(f), .z = linOf(g), .out = linOf(o) };
}

fn mlaRand(s: Stream, rnd: std.Random) !MlaW {
    const q_a = try randProj(s, rnd, hidden, q_lora, n_k6, 1.0 / 64.0);
    errdefer freeProj(q_a);
    const q_b = try randProj(s, rnd, q_lora, 12288, n_k6, 1.0 / 39.2);
    errdefer freeProj(q_b);
    const kv_a = try randProj(s, rnd, hidden, 640, n_k6, 1.0 / 64.0);
    errdefer freeProj(kv_a);
    const o = try randProj(s, rnd, attn_width, hidden, n_k6, 1.0 / 90.5);
    return .{ .q_a = linOf(q_a), .q_b = linOf(q_b), .kv_a = linOf(kv_a), .o = linOf(o) };
}

fn mlaPack(src: *const Source, layer: u32) !MlaW {
    const q_a = try packProj(src, layer, &.{"attention.q_a_proj"});
    errdefer freeProj(q_a);
    const q_b = try packProj(src, layer, &.{"attention.q_b_proj"});
    errdefer freeProj(q_b);
    const kv_a = try packProj(src, layer, &.{"attention.kv_a_proj_with_mqa"});
    errdefer freeProj(kv_a);
    const o = try packProj(src, layer, &.{"attention.dense"});
    return .{ .q_a = linOf(q_a), .q_b = linOf(q_b), .kv_a = linOf(kv_a), .o = linOf(o) };
}

fn mlpRand(s: Stream, rnd: std.Random) !DenseW {
    const gate = try randProj(s, rnd, hidden, ffn_width, n_k4, 1.0 / 64.0);
    errdefer freeProj(gate);
    const up = try randProj(s, rnd, hidden, ffn_width, n_k4, 1.0 / 64.0);
    errdefer freeProj(up);
    const down = try randProj(s, rnd, ffn_width, hidden, n_k4, 1.0 / 101.0);
    return .{ .gate = linOf(gate), .up = linOf(up), .down = linOf(down) };
}

fn mlpPack(src: *const Source, layer: u32) !DenseW {
    const gate = try packProj(src, layer, &.{"mlp.gate_proj"});
    errdefer freeProj(gate);
    const up = try packProj(src, layer, &.{"mlp.up_proj"});
    errdefer freeProj(up);
    const down = try packProj(src, layer, &.{"mlp.down_proj"});
    return .{ .gate = linOf(gate), .up = linOf(up), .down = linOf(down) };
}

fn layerOf(s: Stream, rnd: std.Random, spec: Spec, from_pack: bool) !LayerW {
    const src: ?*const Source = if (from_pack) &pack_weights.?.src else null;
    switch (spec.kind) {
        .kda_dense => {
            const la = if (src) |p| try kdaPack(p, spec.layer) else try kdaRand(s, rnd);
            errdefer (LayerW{ .mixer = .{ .kda = la }, .mlp = .moe, .layer = spec.layer }).deinit();
            const dw = if (src) |p| try mlpPack(p, spec.layer) else try mlpRand(s, rnd);
            return .{ .mixer = .{ .kda = la }, .mlp = .{ .dense = dw }, .layer = spec.layer };
        },
        .kda => return .{ .mixer = .{ .kda = if (src) |p| try kdaPack(p, spec.layer) else try kdaRand(s, rnd) }, .mlp = .moe, .layer = spec.layer },
        .mla => return .{ .mixer = .{ .mla = if (src) |p| try mlaPack(p, spec.layer) else try mlaRand(s, rnd) }, .mlp = .moe, .layer = spec.layer },
    }
}

// ── a layer's spans through the served entry points ──

/// One span's inputs: x [1, S, 4096] (the mixer's and the MLP's), y [1, S, 8192] (o_proj's), qn [1, S, 1536] (q_b's)
const SpanIn = struct {
    x: Array,
    y: Array,
    qn: Array,
    fn deinit(in: SpanIn) void {
        inline for (.{ in.x, in.y, in.qn }) |a| free(a);
    }
};

fn spanIn(s: Stream, rnd: std.Random, rows: c_int) !SpanIn {
    const x = try input(s, rnd, rows, hidden);
    errdefer free(x);
    const y = try input(s, rnd, rows, attn_width);
    errdefer free(y);
    return .{ .x = x, .y = y, .qn = try input(s, rnd, rows, q_lora) };
}

/// A span's outputs in the layer's order (a KDA layer: q|k|v, f, g, o_proj, then a dense layer's gate, h and y; an MLA
/// layer: q_a, q_b, kv_a, o_proj); the owner frees them
const max_outs = 7;
const Outs = struct {
    a: [max_outs]Array = @splat(.{ .ctx = null }),
    n: usize = 0,
    fn add(o: *Outs, x: Array) void {
        o.a[o.n] = x;
        o.n += 1;
    }
    fn deinit(o: Outs) void {
        for (o.a[0..o.n]) |x| free(x);
    }
};

const kda_dense_names = [_][]const u8{ "qkv", "f", "g", "o", "gate", "h", "y" };
const mla_names = [_][]const u8{ "q_a", "q_b", "kv_a", "o" };
const mlp_names = [_][]const u8{ "gate", "h", "y" };

fn outNames(l: *const LayerW) []const []const u8 {
    return switch (l.mixer) {
        .kda => if (l.mlp == .dense) kda_dense_names[0..] else kda_dense_names[0..4],
        .mla => mla_names[0..],
        .dsa => mlp_names[0..],
    };
}

/// The span input of `p`'s width (x 4096, y 8192, qn 1536), null for another (the dense MLP's down)
fn inputFor(p: Proj, in: SpanIn) ?Array {
    const w = mlx.getShape(p.trellis)[0] * 16;
    if (w == hidden) return in.x;
    if (w == attn_width) return in.y;
    if (w == q_lora) return in.qn;
    return null;
}

/// `m.mm` of an EXL3 `Linear`: `ling_exl3.linear`
fn lin(s: Stream, x: Array, p: Lin) !Array {
    return ling_exl3.linear(s, x, p.w, p.s, p.b);
}

/// The layer's mixer projections of one span as the copied layers call them: a KDA layer's q|k|v, f and g by
/// `projPrompt` (each member's `linear` where it declines, kda.zig's fallback), its o_proj; an MLA layer's q_a, q_b,
/// kv_a and o_proj
fn mixer(s: Stream, l: *const LayerW, in: SpanIn, o: *Outs) !void {
    switch (l.mixer) {
        .kda => |la| {
            if (try ling_kdaproj.projPrompt(s, in.x, &la.qkv, &la.a, &la.z)) |qaz| {
                for (qaz) |y| o.add(y);
            } else for ([_]Lin{ la.qkv, la.a, la.z }) |p| o.add(try lin(s, in.x, p));
            o.add(try lin(s, in.y, la.out));
        },
        .mla => |at| {
            o.add(try lin(s, in.x, at.q_a));
            o.add(try lin(s, in.qn, at.q_b));
            o.add(try lin(s, in.x, at.kv_a));
            o.add(try lin(s, in.y, at.o));
        },
        .dsa => {},
    }
}

/// A dense layer's MLP of one span as trunk.zig's `denseMlp` runs it under SWIGLU_EPI: the gate, up's SwiGLU, the down
fn denseMlp(s: Stream, dw: DenseW, x: Array, o: *Outs) !void {
    const gate = try lin(s, x, dw.gate);
    o.add(gate);
    const h = (try epi.up(s, x, dw.up.w, dw.up.s, dw.up.b, gate, 0, fused_act.sigmoidTableFor, .dense)) orelse return error.EpilogueDeclined;
    o.add(h);
    o.add(try lin(s, h, dw.down));
}

/// Every span's mixer, then a dense layer's MLP span by span (`hiddenSpans`' order)
fn runSpans(s: Stream, l: *const LayerW, ins: []const SpanIn, outs: []Outs) !void {
    for (ins, outs) |in, *o| try mixer(s, l, in, o);
    switch (l.mlp) {
        .dense => |dw| for (ins, outs) |in, *o| try denseMlp(s, dw, in.x, o),
        .moe => {},
    }
}

fn evalOuts(outs: []const Outs) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |o| for (o.a[0..o.n]) |x| {
        _ = mlx.mlx_vector_array_append_value(vec, x);
    };
    try mlx.check(mlx.mlx_eval(vec));
}

/// The layer's held linears (each family's EXL3 projections) in `begin`'s order
fn heldOf(l: *const LayerW, out: *[8]Proj) []const Proj {
    var n: usize = 0;
    switch (l.mixer) {
        .kda => |la| for ([_]Lin{ la.qkv, la.a, la.z, la.out }) |x| {
            out[n] = projOf(x);
            n += 1;
        },
        .mla => |at| for ([_]Lin{ at.q_a, at.q_b, at.kv_a, at.o }) |x| {
            out[n] = projOf(x);
            n += 1;
        },
        .dsa => {},
    }
    switch (l.mlp) {
        .dense => |dw| for ([_]Lin{ dw.gate, dw.up, dw.down }) |x| {
            out[n] = projOf(x);
            n += 1;
        },
        .moe => {},
    }
    return out[0..n];
}

/// The scope's counts the layer's spans should leave: each held linear's W_hat decoded once, read by every wide span
fn wantOf(l: *const LayerW) Counts {
    var c: Counts = .{};
    switch (l.mixer) {
        .kda => {
            c.decoded[@intFromEnum(Family.kda)] = 3;
            c.decoded[@intFromEnum(Family.o)] = 1;
        },
        .mla => {
            c.decoded[@intFromEnum(Family.mla)] = 3;
            c.decoded[@intFromEnum(Family.o)] = 1;
        },
        .dsa => {},
    }
    if (l.mlp == .dense) c.decoded[@intFromEnum(Family.dense)] = 3;
    for (&c.read, c.decoded) |*r, dc| r.* = dc * wide;
    return c;
}

/// `p`'s W_hat by its `linear`'s decode launch (lazy)
fn whatOf(s: Stream, p: Proj) !Array {
    const sh = mlx.getShape(p.trellis);
    const d = try t3.denseOf(p, @intCast(sh[0] * 16), .bfloat16, .mul1);
    return d.dequantize(s, p.trellis);
}

fn kdaProjs(l: *const LayerW) [3]Proj {
    const la = l.mixer.kda;
    return .{ projOf(la.qkv), projOf(la.a), projOf(la.z) };
}

/// `counts` by family into `buf`: "<family> <decoded>/<read> (want <d>/<r>)", "; " between them
fn countsText(buf: []u8, got: Counts, want: Counts) []const u8 {
    const names = [4][]const u8{ "KDA q|k|v, f and g", "o_proj", "MLA q_a, q_b and kv_a", "dense MLP" };
    var n: usize = 0;
    for (names, 0..) |name, f| {
        const sep: []const u8 = if (f == 0) "" else "; ";
        n += (std.fmt.bufPrint(buf[n..], "{s}{s} {d}/{d} (want {d}/{d})", .{ sep, name, got.decoded[f], got.read[f], want.decoded[f], want.read[f] }) catch return buf[0..n]).len;
    }
    return buf[0..n];
}

test "turboexl3_ling LAYER_WHAT (P3): a layer-major layer's dense prefill linears over W_hats decoded once at its first wide span and dropped after its spans give the per-call path's words (a KDA layer's q|k|v, f, g and o_proj, an MLA layer's q_a, q_b, kv_a and o_proj, a dense layer's gate, h and y) at the loader's operands, the standard request's spans in layer-major order, layers 0, 4 and 7 in one pass (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    defer lwhat.end();
    if (std.c.getenv("MLX_SERVE_LING_PACK")) |dir| {
        const io = std.Io.Threaded.global_single_threaded.io();
        pack_weights = try ling_weights.Weights.open(std.heap.c_allocator, io, std.mem.span(dir), "model.layers.72.", s);
    }
    defer if (pack_weights) |pw| {
        pw.deinit();
        pack_weights = null;
    };
    var prng = std.Random.DefaultPrng.init(0x9A3C);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var words: usize = 0;
    var reds: usize = 0;
    // red: outside a scope (the chunk loop, decode, the NextN head) a 64-row call is not served
    {
        const p = try randProj(s, rnd, attn_width, hidden, n_k6, 1.0 / 90.5);
        defer freeProj(p);
        const y64 = try input(s, rnd, 64, attn_width);
        defer free(y64);
        const none = (try lwhat.linear(s, y64, p)) == null;
        lines += 1;
        failed += @intFromBool(!none);
        reds += @intFromBool(none);
        std.debug.print("{s} red: outside a scope a 64-row call is not served: {s} {s}\n", .{ tag, if (none) "yes" else "no", if (none) "PASS" else "FAIL" });
    }
    // the last source's layer 4 o_proj at its first span (A's words), for the flipped-bit check
    var last_o: Array = .{ .ctx = null };
    defer free(last_o);
    for ([_]bool{ false, true }) |from_pack| {
        if (from_pack and pack_weights == null) continue;
        const src_name: []const u8 = if (from_pack) "pack" else "random";
        // every layer's spans read the same inputs (each layer's own weights make its words)
        var ins: [spans.len]SpanIn = undefined;
        var n_in: usize = 0;
        defer for (ins[0..n_in]) |in| in.deinit();
        for (spans) |rows| {
            ins[n_in] = try spanIn(s, rnd, rows);
            n_in += 1;
        }
        var layers: [plan.len]LayerW = undefined;
        var n_layers: usize = 0;
        defer for (layers[0..n_layers]) |l| l.deinit();
        for (plan) |spec| {
            layers[n_layers] = try layerOf(s, rnd, spec, from_pack);
            n_layers += 1;
        }
        // layer 4's first span: A's q|k|v, f, g and o_proj words, for the red checks
        var first4: [4]Array = @splat(.{ .ctx = null });
        defer for (first4) |a| free(a);
        var pass_want: Counts = .{};
        markStart();
        lwhat.passBegin();
        for (&layers) |*l| {
            const want_c = wantOf(l);
            for (&pass_want.decoded, &pass_want.read, want_c.decoded, want_c.read) |*pd, *pr, wd, wr| {
                pd.* += wd;
                pr.* += wr;
            }
            const held_n = want_c.decodedAll();
            // A: the per-call path, every span, evaluated together
            var want: [spans.len]Outs = @splat(.{});
            defer for (want) |o| o.deinit();
            try runSpans(s, l, &ins, &want);
            try evalOuts(&want);
            // B: the scope around the layer, every span through the same entry points, evaluated after the scope ends
            var got: [spans.len]Outs = @splat(.{});
            defer for (got) |o| o.deinit();
            lwhat.begin(l);
            const at_begin = lwhat.stats();
            try runSpans(s, l, &ins, &got);
            const in_scope = lwhat.stats();
            lwhat.end();
            const after = lwhat.stats();
            // after the drop: nothing held or kept, and the layer's linears not served (their first span's input)
            var held_buf: [8]Proj = undefined;
            const held = heldOf(l, &held_buf);
            var served_after = false;
            for (held) |p| if (inputFor(p, ins[0])) |xin| if (try lwhat.linear(s, xin, p)) |y| {
                free(y);
                served_after = true;
            };
            if (l.mixer == .kda) if (try lwhat.kdaProjPrompt(s, ins[0].x, kdaProjs(l))) |ys| {
                for (ys) |y| free(y);
                served_after = true;
            };
            try evalOuts(&got);
            var counts_ok = at_begin.open and at_begin.held == held_n and at_begin.kept == 0 and in_scope.held == held_n and
                in_scope.kept == held_n and !after.open and after.held == 0 and after.kept == 0 and !served_after;
            for (in_scope.scope.decoded, in_scope.scope.read, want_c.decoded, want_c.read) |gd, gr, wd, wr| counts_ok = counts_ok and gd == wd and gr == wr;
            var cb: [512]u8 = undefined;
            lines += 1;
            failed += @intFromBool(!counts_ok);
            std.debug.print("{s} {s} layer={d} scope, decoded/read by family: {s}; held {d} kept {d} (want {d}); the 31-row span left to the chain; after the drop: open {s}, held {d}, kept {d}, the layer's linears served {s} {s}\n", .{ tag, src_name, l.layer, countsText(&cb, in_scope.scope, want_c), in_scope.held, in_scope.kept, held_n, if (after.open) "yes" else "no", after.held, after.kept, if (served_after) "yes" else "no", if (counts_ok) "PASS" else "FAIL" });
            // every word of every span, B against A
            const names = outNames(l);
            for (want, got, spans, 0..) |wa, gb, rows, j| {
                var dline: [256]u8 = undefined;
                var dn: usize = 0;
                var ok = wa.n == names.len and gb.n == names.len;
                var nel: usize = 0;
                const m = @min(wa.n, gb.n);
                for (wa.a[0..m], gb.a[0..m], 0..) |x_a, x_b, k| {
                    const d = try differing(x_b, x_a, s);
                    ok = ok and d == 0;
                    nel += mlx.mlx_array_size(x_a);
                    const nm: []const u8 = if (k < names.len) names[k] else "?";
                    dn += (std.fmt.bufPrint(dline[dn..], " {s}={d}", .{ nm, d }) catch break).len;
                }
                words += nel;
                lines += 1;
                failed += @intFromBool(!ok);
                std.debug.print("{s} {s} layer={d} span={d} rows={d} {s} differing{s} (of {d} words) {s}\n", .{ tag, src_name, l.layer, j, rows, if (rows > 32) "kept-W_hat" else "chain", dline[0..dn], nel, if (ok) "PASS" else "FAIL" });
            }
            if (l.layer == plan[1].layer) for (&first4, want[0].a[0..4]) |*k4, x| {
                k4.* = try keep(x);
            };
            _ = mlx.mlx_clear_cache();
        }
        // the pass: its counts over the three layers, and the run-path mark `passEnd` sets
        lwhat.passEnd();
        {
            const ps = lwhat.stats();
            const marked_after = marked();
            var ok = ps.layers == plan.len and ps.spans_reused == plan.len * (wide - 1) and marked_after;
            for (ps.pass.decoded, ps.pass.read, pass_want.decoded, pass_want.read) |gd, gr, wd, wr| ok = ok and gd == wd and gr == wr;
            var cb: [512]u8 = undefined;
            lines += 1;
            failed += @intFromBool(!ok);
            std.debug.print("{s} {s} pass: {d} layers (want {d}), {d} wide spans reused (want {d}), decoded/read by family: {s}; {d} decodes saved (want {d}); the run-path mark set: {s} {s}\n", .{ tag, src_name, ps.layers, plan.len, ps.spans_reused, plan.len * (wide - 1), countsText(&cb, ps.pass, pass_want), ps.pass.saved(), pass_want.saved(), if (marked_after) "yes" else "no", if (ok) "PASS" else "FAIL" });
        }
        const l0 = &layers[0];
        const l4 = &layers[1];
        const rows0: usize = @intCast(spans[0]);
        // red: a wrong layer's W_hats: layer 4's first span's q|k|v, f and g over layer 0's give other words
        {
            var w0: [3]Array = @splat(.{ .ctx = null });
            defer for (w0) |a| free(a);
            for (kdaProjs(l0), &w0) |p, *w| w.* = try whatOf(s, p);
            const ys = try kdaOver(s, ins[0].x, kdaProjs(l4), w0);
            defer for (ys) |y| free(y);
            var d: [3]usize = undefined;
            for (ys, first4[0..3], &d) |y, a, *dd| dd.* = try differing(y, a, s);
            const red_ok = d[0] > 0 and d[1] > 0 and d[2] > 0;
            lines += 1;
            failed += @intFromBool(!red_ok);
            reds += @intFromBool(red_ok);
            std.debug.print("{s} red: {s} layer {d}'s first span over layer {d}'s W_hats (a wrong layer's) differs from its own words: qkv={d} f={d} g={d} {s}\n", .{ tag, src_name, l4.layer, l0.layer, d[0], d[1], d[2], if (red_ok) "PASS" else "FAIL" });
        }
        // red: a W_hat kept past `end`: layer 4's first span's o_proj over layer 0's o_proj W_hat gives other words
        {
            const p4 = projOf(l4.mixer.kda.out);
            const d4 = try t3.denseOf(p4, @intCast(attn_width), .bfloat16, .mul1);
            const w0 = try whatOf(s, projOf(l0.mixer.kda.out));
            defer free(w0);
            const y = try sl.linearOver(s, d4, ins[0].y, p4, w0, rows0);
            defer free(y);
            const d = try differing(y, first4[3], s);
            const red_ok = d > 0;
            lines += 1;
            failed += @intFromBool(!red_ok);
            reds += @intFromBool(red_ok);
            std.debug.print("{s} red: {s} layer {d}'s first span's o_proj over layer {d}'s o_proj W_hat (one kept past the drop) differs from its own words: {d}/{d} {s}\n", .{ tag, src_name, l4.layer, l0.layer, d, mlx.mlx_array_size(first4[3]), if (red_ok) "PASS" else "FAIL" });
        }
        // red: one span's matmul fed a perturbed W_hat: layer 4's o_proj over its own W_hat gives its words; with one
        // exponent bit of its first weight flipped, it does not
        {
            const p4 = projOf(l4.mixer.kda.out);
            const d4 = try t3.denseOf(p4, @intCast(attn_width), .bfloat16, .mul1);
            const w4 = try whatOf(s, p4);
            defer free(w4);
            const own = try sl.linearOver(s, d4, ins[0].y, p4, w4, rows0);
            defer free(own);
            const d_own = try differing(own, first4[3], s);
            const wf = try flipped(w4, 0, 13, s);
            defer free(wf);
            const bad = try sl.linearOver(s, d4, ins[0].y, p4, wf, rows0);
            defer free(bad);
            const d_bad = try differing(bad, first4[3], s);
            const red_ok = d_own == 0 and d_bad > 0;
            lines += 1;
            failed += @intFromBool(!red_ok);
            reds += @intFromBool(red_ok);
            std.debug.print("{s} red: {s} layer {d}'s first span's o_proj over its own W_hat differing={d} (want 0), over it with one bit flipped differing={d} (want > 0) {s}\n", .{ tag, src_name, l4.layer, d_own, d_bad, if (red_ok) "PASS" else "FAIL" });
        }
        free(last_o);
        last_o = try keep(first4[3]);
        _ = mlx.mlx_clear_cache();
    }
    // red: the word comparison counts a one-bit change as one differing word
    {
        const f = try flipped(last_o, 12345, 0, s);
        defer free(f);
        const d = try differing(f, last_o, s);
        lines += 1;
        failed += @intFromBool(d != 1);
        reds += @intFromBool(d == 1);
        std.debug.print("{s} red: one flipped bit of o_proj's output differing={d} (want 1) {s}\n", .{ tag, d, if (d == 1) "PASS" else "FAIL" });
    }
    // red (the review's follow-up b): a stale LAYER_WHAT bit set before the check starts, then a pass with no layer (its
    // `passEnd` saves no decode, so it marks nothing): the check must read the name absent
    {
        ran.mark(.LAYER_WHAT);
        markStart();
        lwhat.passBegin();
        lwhat.passEnd();
        const absent = !marked();
        lines += 1;
        failed += @intFromBool(!absent);
        reds += @intFromBool(absent);
        std.debug.print("{s} red: a stale bit, then a pass that saved no decode: the run-path name {s}\n", .{ tag, if (absent) "absent PASS" else "set, the stale bit passed the check FAIL" });
    }
    const sources: usize = if (pack_weights != null) 2 else 1;
    // per source: each layer's counts line and its 6 span lines, the pass line, 3 red checks; then the 3 global reds
    const want_lines = 3 + sources * (plan.len * (1 + spans.len) + 1 + 3);
    std.debug.print("{s} DONE lines={d} failed={d} words={d} reds={d} {s}\n", .{ tag, lines, failed, words, reds, if (failed == 0) "BYTES_OK" else "BYTES_DIFFER" });
    try std.testing.expectEqual(want_lines, lines);
    try std.testing.expectEqual(3 + 3 * sources, reds);
    try std.testing.expectEqual(@as(usize, 0), failed);
}
