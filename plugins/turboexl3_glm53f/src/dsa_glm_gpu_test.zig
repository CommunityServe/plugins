//! TURBOEXL3_GLM53F_DSA_GLM's device checks (`zig build dsa-gpu-test`, run with TURBOEXL3_GLM53F_DSA_GPU=1 beside
//! src/dsa_gpu_test.zig's), each bit for bit against today's prompt path:
//! - **The tokens:** glm5_next `selectTokens` over src/dsa_gpu_test.zig's inputs (random bf16 queries and cache rows,
//!   f32 head weights) with the switch off (the NAX indexer scores, then MLX's argpartition chain and
//!   `expandSelection`) and on (src/dsa_glm.zig's one launch): a 16K prompt's main call (16,356 keys, its 14,305 sparse
//!   rows) and its 31-row tail (16,387 keys), 995 and 17 rows after a 16,356-token prompt, a 32K, a 64K and a 128K
//!   context's last rows (8,192, 16,384 and 32,768 pools), the main call with every head weight zero (each visible pool
//!   ties at 0) and with keys repeating every 7 pools (pools tie in groups); then 995- and 17-token cold prompts, whose
//!   rows select every pool (ran=0: the launch never runs).
//! - **The scores:** the launch at one row a threadgroup (its scratch rows the rows' scores) against dsa_nax
//!   `indexerScores` over the same inputs, every score a row can see, as f32 bits.
//! - **The output:** `assemble` against `dsaAttn`'s concatenate, transpose and reshape, as bf16 bits: a 16K prompt's
//!   main call (2,051 dense and 14,305 sparse rows; again with the dense part a transposed view, as an attention kernel
//!   may lay it out), a 4,096-token prompt and one of 2,068 tokens (17 sparse rows).
//! One line each, then the summary:
//!     TURBOEXL3_GLM53F_DSA_BYTES glm case=<name> rows=<S> pools=<P> differing=<n>/<total> ran=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DSA_BYTES glm_scores case=<name> rows=<S> pools=<P> differing=<n>/<total> PASS|FAIL
//!     TURBOEXL3_GLM53F_DSA_BYTES glm_out case=<name> dense=<n> sparse=<n> differing=<n>/<total> ran=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DSA_GLM_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const dsa = @import("mlx_serve/plugins/glm5_next/dsa_nax.zig");
const dsa_gpu_test = @import("dsa_gpu_test.zig");
const dsa_glm = @import("dsa_glm.zig");
const Array = mlx.mlx_array;

const ix = dsa_gpu_test.ix;

/// The selection cases (the cache's keys, the query rows: its last ones) and whether the launch runs on them.
const Case = struct { sel: dsa_gpu_test.SelCase, runs: bool };
pub const cases = [_]Case{
    .{ .sel = .{ .name = "main16k", .t = 16356, .rows = 14305, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "tail16k", .t = 16387, .rows = 31, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "warm995", .t = 16356 + 995, .rows = 995, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "warm17", .t = 16356 + 17, .rows = 17, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "last4k_of_32k", .t = 32768, .rows = 4096, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "last256_of_64k", .t = 65536, .rows = 256, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "last64_of_128k", .t = 131072, .rows = 64, .fill = .random }, .runs = true },
    .{ .sel = .{ .name = "main16k_zero_weights", .t = 16356, .rows = 14305, .fill = .zero_weights }, .runs = true },
    .{ .sel = .{ .name = "main16k_repeating_keys", .t = 16356, .rows = 14305, .fill = .repeating_keys }, .runs = true },
    .{ .sel = .{ .name = "cold995", .t = 995, .rows = 995, .fill = .random }, .runs = false },
    .{ .sel = .{ .name = "cold17", .t = 17, .rows = 17, .fill = .random }, .runs = false },
};
/// The output cases: the dense and sparse rows, and whether the dense part is a transposed view.
const OutCase = struct { name: []const u8, dense: c_int, sparse: c_int, strided: bool };
pub const out_cases = [_]OutCase{
    .{ .name = "main16k", .dense = 2051, .sparse = 14305, .strided = false },
    .{ .name = "main16k_strided", .dense = 2051, .sparse = 14305, .strided = true },
    .{ .name = "cold4k", .dense = 2051, .sparse = 2045, .strided = false },
    .{ .name = "cold2068", .dense = 2051, .sparse = 17, .strided = false },
};
fn countRuns() usize {
    var n: usize = 0;
    for (cases) |c| n += @intFromBool(c.runs);
    return n;
}
/// The lines a run prints: a token line a case, a scores line a case the launch runs on, an output line a case.
pub const lines = cases.len + countRuns() + out_cases.len;

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// `a` made contiguous and evaluated, a new array.
fn dense(s: mlx.mlx_stream, a: Array) !Array {
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

const Diff = struct { differing: usize = 0, total: usize = 0 };

/// int32 arrays of one shape, element by element.
fn diffI32(s: mlx.mlx_stream, a: Array, b: Array) !Diff {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return error.DsaGlmGpuShape;
    if (mlx.mlx_array_dtype(a) != .int32 or mlx.mlx_array_dtype(b) != .int32) return error.DsaGlmGpuDtype;
    const ac = try dense(s, a);
    defer _ = mlx.mlx_array_free(ac);
    const bc = try dense(s, b);
    defer _ = mlx.mlx_array_free(bc);
    const n = mlx.mlx_array_size(ac);
    const pa = (mlx.mlx_array_data_int32(ac) orelse return error.DsaGlmGpuData)[0..n];
    const pb = (mlx.mlx_array_data_int32(bc) orelse return error.DsaGlmGpuData)[0..n];
    var d: Diff = .{ .total = n };
    for (pa, pb) |x, y| d.differing += @intFromBool(x != y);
    return d;
}

/// bf16 arrays of one shape, as bits.
fn diffBf16(s: mlx.mlx_stream, a: Array, b: Array) !Diff {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return error.DsaGlmGpuShape;
    if (mlx.mlx_array_dtype(a) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return error.DsaGlmGpuDtype;
    const ac = try dense(s, a);
    defer _ = mlx.mlx_array_free(ac);
    const bc = try dense(s, b);
    defer _ = mlx.mlx_array_free(bc);
    const n = mlx.mlx_array_size(ac);
    const pa = (mlx.mlx_array_data_bfloat16(ac) orelse return error.DsaGlmGpuData)[0..n];
    const pb = (mlx.mlx_array_data_bfloat16(bc) orelse return error.DsaGlmGpuData)[0..n];
    var d: Diff = .{ .total = n };
    for (pa, pb) |x, y| d.differing += @intFromBool(x != y);
    return d;
}

/// The stock selection, or the switch's, evaluated.
fn select(s: mlx.mlx_stream, x: dsa_gpu_test.SelInputs, on: bool) !Array {
    dsa_glm.on = on;
    defer dsa_glm.on = false;
    const out = try glm5.selectTokens(s, x.q, x.w, x.cache, ix);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The launch's scores at one row a threadgroup against the NAX indexer's, every score a row can see, as f32 bits.
fn diffScores(s: mlx.mlx_stream, x: dsa_gpu_test.SelInputs, c: dsa_gpu_test.SelCase) !Diff {
    const q_start = c.t - c.rows;
    const p = @divTrunc(c.t, ix.kpool);
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_multiply(&ws, x.w, scale, s));
    // the pools' keys as glm5_next `storedPoolKeys` slices them: each pool's last row's gate half
    var pk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pk);
    try mlx.check(mlx.mlx_slice(&pk, x.cache, &[_]c_int{ 0, ix.kpool - 1, ix.head_dim }, 3, &[_]c_int{ 1, p * ix.kpool, 2 * ix.head_dim }, 3, &[_]c_int{ 1, ix.kpool, 1 }, 3, s));
    const want = (try dsa.indexerScores(s, x.q, ws, pk, q_start, ix.kpool)) orelse return error.DsaGlmGpuNoNax;
    defer _ = mlx.mlx_array_free(want);
    const got = (try dsa_glm.index(s, x.q, ws, x.cache, q_start, c.rows)) orelse return error.DsaGlmGpuDeclined;
    defer {
        _ = mlx.mlx_array_free(got.tokens);
        _ = mlx.mlx_array_free(got.scores);
    }
    try evalAll(&.{ want, got.tokens, got.scores });
    const wc = try dense(s, want);
    defer _ = mlx.mlx_array_free(wc);
    const gc = try dense(s, got.scores);
    defer _ = mlx.mlx_array_free(gc);
    const pu: usize = @intCast(p);
    const rows: usize = @intCast(c.rows);
    const pw = (mlx.mlx_array_data_float32(wc) orelse return error.DsaGlmGpuData)[0 .. rows * pu];
    const pg = (mlx.mlx_array_data_float32(gc) orelse return error.DsaGlmGpuData)[0 .. rows * pu];
    var d: Diff = .{};
    for (0..rows) |r| {
        const q_pos = q_start + @as(c_int, @intCast(r));
        const pv: usize = @intCast(@min(p, @divTrunc(q_pos + 1, ix.kpool)));
        for (pw[r * pu ..][0..pv], pg[r * pu ..][0..pv]) |a, b| {
            d.differing += @intFromBool(@as(u32, @bitCast(a)) != @as(u32, @bitCast(b)));
        }
        d.total += pv;
    }
    return d;
}

/// `dsaAttn`'s output without the switch: the parts concatenated along the rows, transposed and reshaped.
fn stockOut(s: mlx.mlx_stream, dn: Array, sp: Array, rows: c_int) !Array {
    const two = [_]Array{ dn, sp };
    const vec = mlx.mlx_vector_array_new_data(&two, two.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ot, &[_]c_int{ 1, rows, dsa_glm.attn_heads * dsa_glm.attn_v }, 3, s));
    return flat;
}

test "turboexl3_glm53f dsa gpu: the GLM-5.3-Flash indexer launch writes the stock selection's tokens from the NAX indexer's scores bit for bit, and the one-copy output the stock concatenate and transpose's bits" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer dsa_glm.on = false;
    const s = mlx.gpuStream();
    var seed: u64 = 20261109;
    var fails: usize = 0;
    var printed: usize = 0;
    for (cases) |c| {
        var x = try dsa_gpu_test.selInputs(s, &seed, c.sel);
        defer x.deinit();
        const off = try select(s, x, false);
        defer _ = mlx.mlx_array_free(off);
        const before = dsa_glm.index_runs;
        const on = try select(s, x, true);
        defer _ = mlx.mlx_array_free(on);
        const ran = dsa_glm.index_runs - before;
        const d = try diffI32(s, off, on);
        const ok = d.differing == 0 and d.total > 0 and ran == @intFromBool(c.runs);
        if (!ok) fails += 1;
        printed += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES glm case={s} rows={d} pools={d} differing={d}/{d} ran={d} {s}\n", .{ c.sel.name, c.sel.rows, @divTrunc(c.sel.t, ix.kpool), d.differing, d.total, ran, if (ok) "PASS" else "FAIL" });
        if (!c.runs) continue;
        const ds = try diffScores(s, x, c.sel);
        const ok_s = ds.differing == 0 and ds.total > 0;
        if (!ok_s) fails += 1;
        printed += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES glm_scores case={s} rows={d} pools={d} differing={d}/{d} {s}\n", .{ c.sel.name, c.sel.rows, @divTrunc(c.sel.t, ix.kpool), ds.differing, ds.total, if (ok_s) "PASS" else "FAIL" });
    }
    for (out_cases) |c| {
        const rows = c.dense + c.sparse;
        const h = dsa_glm.attn_heads;
        const v = dsa_glm.attn_v;
        // the dense part [1, 64, dense, 256], or the same shape viewed from [1, dense, 64, 256]
        const dn_shape: [4]c_int = if (c.strided) .{ 1, c.dense, h, v } else .{ 1, h, c.dense, v };
        const dn_src = try dsa_gpu_test.normal(s, &seed, &dn_shape, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(dn_src);
        var dn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dn);
        if (c.strided) {
            try mlx.check(mlx.mlx_transpose_axes(&dn, dn_src, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        } else _ = mlx.mlx_array_set(&dn, dn_src);
        const sp = try dsa_gpu_test.normal(s, &seed, &.{ 1, h, c.sparse, v }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(sp);
        const want = try stockOut(s, dn, sp, rows);
        defer _ = mlx.mlx_array_free(want);
        dsa_glm.on = true;
        const before = dsa_glm.assemble_runs;
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        const took = try dsa_glm.assemble(&o, s, dn, sp);
        dsa_glm.on = false;
        const ran = dsa_glm.assemble_runs - before;
        var d: Diff = .{};
        if (took) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, o, &[_]c_int{ 1, rows, h * v }, 3, s));
            d = try diffBf16(s, want, flat);
        }
        const ok = took and d.differing == 0 and d.total > 0 and ran == 1;
        if (!ok) fails += 1;
        printed += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES glm_out case={s} dense={d} sparse={d} differing={d}/{d} ran={d} {s}\n", .{ c.name, c.dense, c.sparse, d.differing, d.total, ran, if (ok) "PASS" else "FAIL" });
    }
    // another model's attention (32 heads) is left to the stock path
    {
        const dn = try dsa_gpu_test.normal(s, &seed, &.{ 1, 32, 64, dsa_glm.attn_v }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(dn);
        const sp = try dsa_gpu_test.normal(s, &seed, &.{ 1, 32, 64, dsa_glm.attn_v }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(sp);
        dsa_glm.on = true;
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        const took = try dsa_glm.assemble(&o, s, dn, sp);
        dsa_glm.on = false;
        if (took) fails += 1;
    }
    std.debug.print("TURBOEXL3_GLM53F_DSA_GLM_PARITY lines={d} fails={d} {s}\n", .{ printed, fails, if (fails == 0 and printed == lines) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
    try std.testing.expectEqual(lines, printed);
}
