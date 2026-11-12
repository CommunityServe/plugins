//! turboexl3_ling's run-path marks for the paths whose launch is their only evidence (GPU; `MLX_SERVE_LING_GPU_TESTS=1`
//! runs it, else it skips). The strip review's finding 2: PAIR, SHARED_F3, KDA_PROJ_FUSED, ROUTER_F16 and MLA_FLASH's
//! expanded-key launch printed a line when their launch ran, and lost it with no mark. Each case clears the bits, calls
//! the path's entry on operands it takes (Ling-3.1-flash's shapes, random EXL3 trellis words as the loader keeps them)
//! and requires its name set; then clears them, makes the same call decline before the launch (a projection that is
//! not EXL3, more rows than the launch takes, an f32 weight or operand) and requires the name absent: a path that stops
//! running drops its name from the run-path line, which the strip smoke reads. Each case prints a
//! `TURBOEXL3_LING_RAN <name>` line and a `TURBOEXL3_LING_RAN_RED <name>` line, each ending in PASS or FAIL.
//! PROMPT_AHEAD_FILL, READS_WIDE, MTP_HEAD_LATENT and its composed fallback mark inside the served source and the copied
//! MLA layer, which only a host that loaded the plugin opens: the strip smoke requires their names.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const ran = @import("ling_ran.zig");
const ling_pair = @import("ling_pair.zig");
const ling_shared = @import("ling_shared.zig");
const ling_kdaproj = @import("ling_kdaproj.zig");
const ling_router = @import("ling_router.zig");
const ling_flash = @import("ling_flash.zig");
const fused_act = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

const hidden: c_int = 4096;
/// K 6: 96 trellis words a 16 x 16 tile
const n_k6: c_int = 96;

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

/// A projection as the copied `Linear` holds it: `w` the trellis words viewed as I16 [in / 16, out / 16, n], `s` its
/// suh [in] and `b` its svh [out] in F16, `exl3` its kind
const Lin = struct { w: Array, s: Array, b: Array, exl3: bool = true };

fn freeLin(l: Lin) void {
    free(l.w);
    free(l.s);
    free(l.b);
}

/// Random trellis words for a projection in -> out at K 6, svh scaled so the outputs stay near unit size
fn lin(s: Stream, rnd: std.Random, in_dim: c_int, out_dim: c_int) !Lin {
    const a = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(in_dim, 16), @divExact(out_dim, 16), n_k6 };
    const words = try a.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer a.free(words);
    rnd.bytes(std.mem.sliceAsBytes(words));
    const u = try hostArray(u16, words, &shape, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    errdefer free(w);
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    const su = try a.alloc(f16, @intCast(in_dim));
    defer a.free(su);
    for (su) |*v| v.* = @floatCast(0.5 + rnd.float(f32));
    const sv = try a.alloc(f16, @intCast(out_dim));
    defer a.free(sv);
    for (sv) |*v| v.* = @floatCast((0.5 + rnd.float(f32)) / 64.0);
    const suh = try hostArray(f16, su, &.{in_dim}, .float16);
    errdefer free(suh);
    const svh = try hostArray(f16, sv, &.{out_dim}, .float16);
    return .{ .w = w, .s = suh, .b = svh };
}

/// x [1, rows, cols] in `dt`: normal draws
fn input(s: Stream, rnd: std.Random, rows: c_int, cols: c_int, dt: mlx.mlx_dtype) !Array {
    const a = std.heap.page_allocator;
    const xf = try a.alloc(f32, @intCast(rows * cols));
    defer a.free(xf);
    for (xf) |*v| v.* = rnd.floatNorm(f32);
    const f = try hostArray(f32, xf, &.{ 1, rows, cols }, .float32);
    defer free(f);
    var x = mlx.mlx_array_new();
    errdefer free(x);
    try mlx.check(mlx.mlx_astype(&x, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}

/// Every output of a call evaluated (the launch has run), then freed
fn evalFree(outs: []const Array) !void {
    for (outs) |o| try mlx.check(mlx.mlx_array_eval(o));
    for (outs) |o| free(o);
}

/// The case's line: `want` (the name set, or absent for the red) against what the bits show
fn line(comptime tag: []const u8, comptime p: ran.Path, what: []const u8, want_set: bool) usize {
    const set = ran.has(p);
    const ok = set == want_set;
    std.debug.print(tag ++ " " ++ @tagName(p) ++ " {s}: name {s} {s}\n", .{ what, if (set) "set" else "absent", if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

test "turboexl3_ling run-path marks: PAIR, SHARED_F3, KDA_PROJ_FUSED, ROUTER_F16 and MLA_FLASH are set when their launch runs, and absent when the same call declines before it (GPU)" {
    const s = try gpu();
    ran.clearForTests();
    defer ran.clearForTests();
    var prng = std.Random.DefaultPrng.init(0x5EED_2A11);
    const rnd = prng.random();
    var failed: usize = 0;

    // PAIR: the shared expert's gate | up (4096 -> 1280 each) at one decode row
    {
        const g = try lin(s, rnd, hidden, 1280);
        defer freeLin(g);
        const u = try lin(s, rnd, hidden, 1280);
        defer freeLin(u);
        const x = try input(s, rnd, 1, hidden, .bfloat16);
        defer free(x);
        ran.clearForTests();
        if (try ling_pair.pair(s, x, g, u)) |o| try evalFree(&o);
        failed += line("TURBOEXL3_LING_RAN", .PAIR, "gate | up at 1 row", true);
        ran.clearForTests();
        var g_plain = g;
        g_plain.exl3 = false;
        if (try ling_pair.pair(s, x, g_plain, u)) |o| try evalFree(&o);
        failed += line("TURBOEXL3_LING_RAN_RED", .PAIR, "a member that is not EXL3 (the pair declines)", false);
    }

    // SHARED_F3: the shared expert (4096 -> 1280 -> 4096) at 1 row, then at 5 rows, past the two launches' 4
    {
        const g = try lin(s, rnd, hidden, 1280);
        defer freeLin(g);
        const u = try lin(s, rnd, hidden, 1280);
        defer freeLin(u);
        const d = try lin(s, rnd, 1280, hidden);
        defer freeLin(d);
        const gate = [3]Array{ g.w, g.s, g.b };
        const up = [3]Array{ u.w, u.s, u.b };
        const down = [3]Array{ d.w, d.s, d.b };
        const x1 = try input(s, rnd, 1, hidden, .bfloat16);
        defer free(x1);
        ran.clearForTests();
        if (try ling_shared.run(s, x1, gate, up, down, 0, fused_act.sigmoidTableFor)) |y| try evalFree(&.{y});
        failed += line("TURBOEXL3_LING_RAN", .SHARED_F3, "the shared expert at 1 row", true);
        const x5 = try input(s, rnd, 5, hidden, .bfloat16);
        defer free(x5);
        ran.clearForTests();
        if (try ling_shared.run(s, x5, gate, up, down, 0, fused_act.sigmoidTableFor)) |y| try evalFree(&.{y});
        failed += line("TURBOEXL3_LING_RAN_RED", .SHARED_F3, "5 rows (the two launches decline)", false);
    }

    // KDA_PROJ_FUSED: a KDA layer's q|k|v (4096 -> 24576), f and g (4096 -> 8192 each) at 2 rows
    {
        const qkv = try lin(s, rnd, hidden, 24576);
        defer freeLin(qkv);
        const f = try lin(s, rnd, hidden, 8192);
        defer freeLin(f);
        const g = try lin(s, rnd, hidden, 8192);
        defer freeLin(g);
        const x = try input(s, rnd, 2, hidden, .bfloat16);
        defer free(x);
        ran.clearForTests();
        if (try ling_kdaproj.proj(s, x, qkv, f, g)) |o| try evalFree(&o);
        failed += line("TURBOEXL3_LING_RAN", .KDA_PROJ_FUSED, "q|k|v, f and g at 2 rows", true);
        ran.clearForTests();
        var qkv_plain = qkv;
        qkv_plain.exl3 = false;
        if (try ling_kdaproj.proj(s, x, qkv_plain, f, g)) |o| try evalFree(&o);
        failed += line("TURBOEXL3_LING_RAN_RED", .KDA_PROJ_FUSED, "q|k|v not EXL3 (the launch declines)", false);
    }

    // ROUTER_F16: the fused router over the f16 weight's column-major view [4096, 512] at 1 row, and the stock call's
    // f32 table rebuilt from it; then the same calls over an f32 weight, which take no f16 path
    {
        const a = std.heap.page_allocator;
        const n_experts: c_int = 512;
        const wf = try a.alloc(f32, @intCast(n_experts * hidden));
        defer a.free(wf);
        for (wf) |*v| v.* = rnd.floatNorm(f32) / 64.0;
        const w32_rows = try hostArray(f32, wf, &.{ n_experts, hidden }, .float32);
        defer free(w32_rows);
        var w16_rows = mlx.mlx_array_new();
        defer free(w16_rows);
        try mlx.check(mlx.mlx_astype(&w16_rows, w32_rows, .float16, s));
        try mlx.check(mlx.mlx_array_eval(w16_rows));
        // the loader's view: [hidden, 512] column-major, the transpose of the row-major [512, hidden] words. Evaluated,
        // as the loader's is: an unevaluated view reports row-major strides, and `logits` declines that layout.
        var w16 = mlx.mlx_array_new();
        defer free(w16);
        try mlx.check(mlx.mlx_transpose_axes(&w16, w16_rows, &[_]c_int{ 1, 0 }, 2, s));
        try mlx.check(mlx.mlx_array_eval(w16));
        var w32 = mlx.mlx_array_new();
        defer free(w32);
        try mlx.check(mlx.mlx_transpose_axes(&w32, w32_rows, &[_]c_int{ 1, 0 }, 2, s));
        try mlx.check(mlx.mlx_array_eval(w32));
        const x = try input(s, rnd, 1, hidden, .bfloat16);
        defer free(x);
        var x2 = mlx.mlx_array_new();
        defer free(x2);
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ 1, hidden }, 2, s));
        ran.clearForTests();
        if (try ling_router.logits(s, x2, w16, 0)) |y| try evalFree(&.{y});
        ling_router.clear();
        failed += line("TURBOEXL3_LING_RAN", .ROUTER_F16, "the fused router over the f16 weight", true);
        ran.clearForTests();
        if (try ling_router.stockTable(s, w16)) |t| try evalFree(&.{t});
        failed += line("TURBOEXL3_LING_RAN", .ROUTER_F16, "the stock call's f32 table from the f16 weight", true);
        ran.clearForTests();
        if (try ling_router.logits(s, x2, w32, 0)) |y| try evalFree(&.{y});
        ling_router.clear();
        if (try ling_router.stockTable(s, w32)) |t| try evalFree(&.{t});
        failed += line("TURBOEXL3_LING_RAN_RED", .ROUTER_F16, "an f32 weight (no f16 path)", false);
    }

    // MLA_FLASH: the expanded-key flash launch (the MTP head's prompt chunks) over 64 heads, qk 192 and v 128, at 32 rows
    // after none; then the same call in f32, which the launch refuses
    {
        const rows: c_int = 32;
        const scale: f32 = 1.0 / @sqrt(@as(f32, 192));
        const q = try heads(s, rnd, rows, 192, .bfloat16);
        defer free(q);
        const k = try heads(s, rnd, rows, 192, .bfloat16);
        defer free(k);
        const v = try heads(s, rnd, rows, 128, .bfloat16);
        defer free(v);
        var out = mlx.mlx_array_new();
        defer free(out);
        ran.clearForTests();
        if (try ling_flash.attend(&out, q, k, v, scale, s)) try mlx.check(mlx.mlx_array_eval(out));
        failed += line("TURBOEXL3_LING_RAN", .MLA_FLASH, "the expanded-key flash launch at 32 rows", true);
        const q32 = try heads(s, rnd, rows, 192, .float32);
        defer free(q32);
        const k32 = try heads(s, rnd, rows, 192, .float32);
        defer free(k32);
        const v32 = try heads(s, rnd, rows, 128, .float32);
        defer free(v32);
        var out32 = mlx.mlx_array_new();
        defer free(out32);
        ran.clearForTests();
        if (try ling_flash.attend(&out32, q32, k32, v32, scale, s)) try mlx.check(mlx.mlx_array_eval(out32));
        failed += line("TURBOEXL3_LING_RAN_RED", .MLA_FLASH, "f32 operands (the launch refuses them)", false);
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// [1, 64, rows, d] in `dt`: normal draws
fn heads(s: Stream, rnd: std.Random, rows: c_int, d: c_int, dt: mlx.mlx_dtype) !Array {
    const a = std.heap.page_allocator;
    const n: usize = @intCast(64 * rows * d);
    const xf = try a.alloc(f32, n);
    defer a.free(xf);
    for (xf) |*v| v.* = rnd.floatNorm(f32);
    const f = try hostArray(f32, xf, &.{ 1, 64, rows, d }, .float32);
    defer free(f);
    var x = mlx.mlx_array_new();
    errdefer free(x);
    try mlx.check(mlx.mlx_astype(&x, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}
