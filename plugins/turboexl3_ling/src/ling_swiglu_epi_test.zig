//! turboexl3_ling's prefill SwiGLU epilogue against the chain it replaces, built from the host's own code (GPU;
//! `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips). SWIGLU_EPI's entry, src/ling_swiglu_epi.zig's `up` with the
//! table the copied blocks pass it (`fused_act.sigmoidTableFor`), against the composed chain at a prompt's rows: the
//! gate and up by `ling_exl3.linear`, then mlx-serve's fusedSwiGLU (limit 0) or the plugin's swigluClampOut (limit 10),
//! word for word. The operands are Ling-3.1-flash's as the loader keeps them: the shared expert's gate and up (I16
//! trellis views [256, 80, 96], K 6, with F16 suh and svh) at limits 0 and 10, and the dense MLP's ([256, 640, 64], K 4)
//! at limit 0.
//! x is bf16 [1, S, 4096] at the standard request's span widths (31, 1965 and 4096 rows), every fifth row times 16, so
//! the clamp binds on the gate and on up. A second test holds the host's sigmoid table to turboexl3's
//! (`lingSigmoidTable(s, bf16, false)`) at every pattern that is not a NaN. Each case prints a
//! `TURBOEXL3_LING_SWIGLU_EPI E6-1` line ending in PASS or FAIL.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const fused_act = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const ling_exl3 = @import("ling_exl3.zig");
const epi = @import("ling_swiglu_epi.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

const hidden: c_int = 4096;

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

const Proj = struct { w: Array, suh: Array, svh: Array };

fn freeProj(p: Proj) void {
    free(p.w);
    free(p.suh);
    free(p.svh);
}

/// A projection 4096 -> out as the loader keeps it: random trellis words viewed as I16 [256, out / 16, n], suh [4096]
/// and svh [out] in F16. svh scales the outputs to about 8, so the large rows pass the limit.
fn proj(s: Stream, rnd: std.Random, out: c_int, n: c_int) !Proj {
    const a = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(hidden, 16), @divExact(out, 16), n };
    const words = try a.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer a.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    const u = try hostArray(u16, words, &shape, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    errdefer free(w);
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    const su = try a.alloc(f16, @intCast(hidden));
    defer a.free(su);
    for (su) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const sv = try a.alloc(f16, @intCast(out));
    defer a.free(sv);
    for (sv) |*v| v.* = @floatCast(8.0 * (0.5 + rnd.float(f32)) / 64.0);
    const suh = try hostArray(f16, su, &.{hidden}, .float16);
    errdefer free(suh);
    const svh = try hostArray(f16, sv, &.{out}, .float16);
    return .{ .w = w, .suh = suh, .svh = svh };
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

/// The elements where the clamp at `limit` binds: the gate's SiLU above it, and up beyond it in size
fn clamped(gate: Array, up: Array, limit: f32, s: Stream) ![2]usize {
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

const Case = struct { site: epi.Site, out: c_int, n: c_int, limit: f32 };
/// The shared expert's up (K 6) at limits 0 and 10, and the dense MLP's (K 4) at limit 0
const cases = [_]Case{
    .{ .site = .shared, .out = 1280, .n = 96, .limit = 0 },
    .{ .site = .shared, .out = 1280, .n = 96, .limit = 10 },
    .{ .site = .dense, .out = 10240, .n = 64, .limit = 0 },
};
/// The standard request's span widths (four spans of 4096 rows, then 1965 and 31)
const spans = [_]c_int{ 31, 1965, 4096 };

test "turboexl3_ling SWIGLU_EPI (E6-1): the epilogue gives the served chain's words (the host's fusedSwiGLU, the plugin's swigluClampOut) at the shared expert's and the dense MLP's operands as the loader keeps them (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    var prng = std.Random.DefaultPrng.init(0xE61);
    const rnd = prng.random();
    var lines: usize = 0;
    var failed: usize = 0;
    var clamp_bound = false;
    for (cases) |c| {
        const gate_p = try proj(s, rnd, c.out, c.n);
        defer freeProj(gate_p);
        const up_p = try proj(s, rnd, c.out, c.n);
        defer freeProj(up_p);
        for (spans) |rows| {
            const x = try input(s, rnd, rows);
            defer free(x);
            const gate = try ling_exl3.linear(s, x, gate_p.w, gate_p.suh, gate_p.svh);
            defer free(gate);
            // the served chain: up's linear, then the host's SwiGLU
            const up = try ling_exl3.linear(s, x, up_p.w, up_p.suh, up_p.svh);
            defer free(up);
            const want = if (c.limit == 0)
                (try fused_act.fusedSwiGLU(s, gate, up)) orelse return error.FusedSwiGLUDeclined
            else
                try ling_exl3.swigluClampOut(s, gate, up, c.limit);
            defer free(want);
            // the epilogue: SWIGLU_EPI's entry with the table the copied blocks pass it
            const got = (try epi.up(s, x, up_p.w, up_p.suh, up_p.svh, gate, c.limit, fused_act.sigmoidTableFor, c.site)) orelse return error.EpilogueDeclined;
            defer free(got);
            const d = try differing(got, want, s);
            const nel = mlx.mlx_array_size(want);
            if (c.limit != 0 and rows == 4096) {
                const over = try clamped(gate, up, c.limit, s);
                std.debug.print("TURBOEXL3_LING_SWIGLU_EPI E6-1 clamp {s} rows={d} limit={d} silu_over={d}/{d} up_over={d}/{d}\n", .{ @tagName(c.site), rows, c.limit, over[0], nel, over[1], nel });
                clamp_bound = over[0] > 0 and over[1] > 0;
            }
            lines += 1;
            failed += @intFromBool(d != 0);
            std.debug.print("TURBOEXL3_LING_SWIGLU_EPI E6-1 {s} 4096x{d} n={d} rows={d} limit={d} differing={d}/{d} {s}\n", .{ @tagName(c.site), c.out, c.n, rows, c.limit, d, nel, if (d == 0) "PASS" else "FAIL" });
        }
        _ = mlx.mlx_clear_cache();
    }
    try std.testing.expectEqual(@as(usize, cases.len * spans.len), lines);
    try std.testing.expect(clamp_bound);
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "turboexl3_ling SWIGLU_EPI (E6-1): the host's sigmoid table (fused_act.sigmoidTableFor) is turboexl3's lingSigmoidTable at every bf16 pattern that is not a NaN (MLX_SERVE_LING_GPU_TESTS=1)" {
    const s = try gpu();
    // the host keeps its table for the process: not freed here
    const host = try fused_act.sigmoidTableFor(s, .bfloat16);
    const ours = try t3.lingSigmoidTable(s, .bfloat16, false);
    defer free(ours);
    try std.testing.expectEqual(@as(usize, 1 << 16), mlx.mlx_array_size(host));
    try std.testing.expectEqual(@as(usize, 1 << 16), mlx.mlx_array_size(ours));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(host));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(ours));
    const ch = try evaluated(host, s);
    defer free(ch);
    const co = try evaluated(ours, s);
    defer free(co);
    const ph = mlx.mlx_array_data_uint8(ch) orelse return error.NoData;
    const po = mlx.mlx_array_data_uint8(co) orelse return error.NoData;
    var compared: usize = 0;
    var d: usize = 0;
    for (0..1 << 16) |i| {
        const pat: u16 = @intCast(i);
        // a NaN pattern: every exponent bit set and a mantissa bit set
        if ((pat & 0x7F80) == 0x7F80 and (pat & 0x007F) != 0) continue;
        compared += 1;
        d += @intFromBool(!std.mem.eql(u8, ph[2 * i ..][0..2], po[2 * i ..][0..2]));
    }
    std.debug.print("TURBOEXL3_LING_SWIGLU_EPI E6-1 table bf16 differing={d}/{d} {s}\n", .{ d, compared, if (d == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, (1 << 16) - 254), compared);
    try std.testing.expectEqual(@as(usize, 0), d);
}
