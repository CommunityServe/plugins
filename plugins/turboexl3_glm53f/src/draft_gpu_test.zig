//! The device check of `TURBOEXL3_GLM53F_HEAD_LAST_ROW` (sdk/bases/blocks/moe.zig `mlpLastRow`,
//! patches/turboexl3-sdk.patch), run by the build step `draft-gpu-test` (linked to the
//! server's MLX) under TURBOEXL3_GLM53F_DRAFT_GPU=1; `zig build test` compiles it and its filter leaves it out.
//!
//! The switch runs the MTP head's merged first draft step's routed experts (turboexl3's `moeClamped` over the resident
//! bank) at the step's last row alone, where the step ran them over every row and read the last. That is exact when a
//! row's output is the same whatever rows share the call: this holds it at the MTP bank's geometry (hidden 4096, inter
//! 2048, K4, the clamped SwiGLU at GLM-5.3-Flash's swiglu_limit 10, mul1 codewords), on seeded random codewords, suh and
//! svh, bf16 x and bf16 routing weights (the head's dtypes), at 2, 3, 4, 5, 8 and 16 rows whose routes share experts
//! (consecutive rows four of their eight): a merged step's rows (the round's committed rows and the draft row, 2 to 4 at
//! the MTP's depth 2, more after a prompt-lookup round) up to the 16 of turboexl3's decode form, past which the switch
//! declines (`mlpLastRow`: the whole call would take the prefill chain). For each row count, the call over every row,
//! then each row alone, its bytes against its row of the call. One line a row, and the run's last line:
//!
//!     TURBOEXL3_GLM53F_DRAFT_BYTES rows=<n> row=<r> differing=<n>/<total> PASS|FAIL
//!     TURBOEXL3_GLM53F_DRAFT_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const Array = mlx.mlx_array;

const hidden: u32 = 4096;
const inter: u32 = 2048;
/// The MTP layer's experts' rate (mtp-experts.bin: 64 halfwords a tile)
const k: u32 = 4;
/// A bank of this many experts (the MTP layer has 288; a row's bytes do not read the others)
const experts: u32 = 32;
const top_k: u32 = 8;
/// GLM-5.3-Flash's swiglu_limit (the clamp `moe.zig` passes the head's bank)
const limit: c_int = 10;
const rows_tested = [_]c_int{ 2, 3, 4, 5, 8, 16 };
/// The byte lines a run prints: one a row of each row count
pub const byte_lines = 2 + 3 + 4 + 5 + 8 + 16;

fn words(w: []const u16, shape: []const c_int, dtype: mlx.mlx_dtype) Array {
    return mlx.mlx_array_new_data(w.ptr, shape.ptr, @intCast(shape.len), dtype);
}

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn bf16Bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// A bank projection `in -> out` at rate `k` as the MTP bank stores it: random codewords [E, in/16, out/16, 16k] u16,
/// suh f16 [E, in] in [0.5, 1.5], svh f16 [E, out] in [0.5, 1.5] / sqrt(in).
fn proj(a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32) !t3.Proj {
    const n = 16 * k;
    const tw = try a.alloc(u16, @as(usize, experts) * (in / 16) * (out / 16) * n);
    defer a.free(tw);
    rnd.bytes(std.mem.sliceAsBytes(tw));
    const su = try a.alloc(u16, @as(usize, experts) * in);
    defer a.free(su);
    for (su) |*v| v.* = f16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, @as(usize, experts) * out);
    defer a.free(sv);
    for (sv) |*v| v.* = f16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: t3.Proj = .{
        .trellis = words(tw, &.{ @intCast(experts), @intCast(in / 16), @intCast(out / 16), @intCast(n) }, .uint16),
        .suh = words(su, &.{ @intCast(experts), @intCast(in) }, .float16),
        .svh = words(sv, &.{ @intCast(experts), @intCast(out) }, .float16),
    };
    try evalAll(&.{ p.trellis, p.suh, p.svh });
    return p;
}

fn freeProj(p: t3.Proj) void {
    for ([_]Array{ p.trellis, p.suh, p.svh }) |x| _ = mlx.mlx_array_free(x);
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Row `r` of `x` [1, rows, w]: [1, 1, w] (the slice the head's last-row call takes)
fn rowOf(s: mlx.mlx_stream, x: Array, r: c_int) !Array {
    const sh = mlx.getShape(x);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// The bf16 words of `got` that differ from `want`'s (both made contiguous: copies of the same bits), of the total
fn differing(s: mlx.mlx_stream, want: Array, got: Array) !struct { n: usize, total: usize } {
    if (mlx.mlx_array_dtype(want) != .bfloat16 or mlx.mlx_array_dtype(got) != .bfloat16) return error.DraftGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.DraftGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.DraftGpuData)[0..n];
    const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.DraftGpuData)[0..n];
    var d: usize = 0;
    for (pw, pg) |x, y| d += @intFromBool(x != y);
    return .{ .n = d, .total = n };
}

test "turboexl3_glm53f draft gpu: the MTP bank's routed experts over one row give that row of the call over every row bit for bit, at 2 to 16 rows whose routes share experts" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DRAFT_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(20261109);
    const rnd = prng.random();
    const bank: t3.Bank = .{ .gate = try proj(a, rnd, hidden, inter), .up = try proj(a, rnd, hidden, inter), .down = try proj(a, rnd, inter, hidden) };
    defer for ([_]t3.Proj{ bank.gate, bank.up, bank.down }) |p| freeProj(p);
    var lines: usize = 0;
    var fails: usize = 0;
    for (rows_tested) |rows| {
        const nr: usize = @intCast(rows);
        // the routes: a shuffle of the bank, row r its entries 4r to 4r + 8 (consecutive rows share four experts)
        var perm: [experts]u32 = undefined;
        for (&perm, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(u32, &perm);
        const ids = try a.alloc(u32, nr * top_k);
        defer a.free(ids);
        for (0..nr) |r| for (0..top_k) |j| {
            ids[r * top_k + j] = perm[(4 * r + j) % experts];
        };
        const wts = try a.alloc(u16, nr * top_k);
        defer a.free(wts);
        for (wts) |*v| v.* = bf16Bits(0.05 + 0.35 * rnd.float(f32));
        const xs = try a.alloc(u16, nr * hidden);
        defer a.free(xs);
        for (xs) |*v| v.* = bf16Bits(rnd.floatNorm(f32));
        const x = words(xs, &.{ 1, rows, @intCast(hidden) }, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const inds = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, rows, @intCast(top_k) }, 3, .uint32);
        defer _ = mlx.mlx_array_free(inds);
        const scores = words(wts, &.{ 1, rows, @intCast(top_k) }, .bfloat16);
        defer _ = mlx.mlx_array_free(scores);
        // the call over every row (the merged step's MoE as the head ran it)
        const want = try t3.moeClamped(s, x, bank, inds, scores, .mul1, limit);
        defer _ = mlx.mlx_array_free(want);
        try evalAll(&.{want});
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            const xr = try rowOf(s, x, r);
            defer _ = mlx.mlx_array_free(xr);
            const ir = try rowOf(s, inds, r);
            defer _ = mlx.mlx_array_free(ir);
            const sr = try rowOf(s, scores, r);
            defer _ = mlx.mlx_array_free(sr);
            // the row alone (HEAD_LAST_ROW's call, at the last row; every row here)
            const got = try t3.moeClamped(s, xr, bank, ir, sr, .mul1, limit);
            defer _ = mlx.mlx_array_free(got);
            const wr = try rowOf(s, want, r);
            defer _ = mlx.mlx_array_free(wr);
            const d = try differing(s, wr, got);
            const ok = d.total == hidden and d.n == 0;
            lines += 1;
            fails += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_GLM53F_DRAFT_BYTES rows={d} row={d} differing={d}/{d} {s}\n", .{ rows, r, d.n, d.total, if (ok) "PASS" else "FAIL" });
        }
    }
    const ok = fails == 0 and lines == byte_lines;
    std.debug.print("TURBOEXL3_GLM53F_DRAFT_PARITY lines={d} fails={d} {s}\n", .{ lines, fails, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}
