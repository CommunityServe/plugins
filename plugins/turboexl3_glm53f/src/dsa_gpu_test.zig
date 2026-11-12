//! The device checks of the DSA prompt texts (src/dsa_prompt.zig), run by the build step `dsa-gpu-test` (linked to the
//! server's MLX) under TURBOEXL3_GLM53F_DSA_GPU=1; `zig build test` compiles them and its filter leaves them out.
//!
//! - `DSA_ONEP`, bounded: the one-P sparse attention against the plain form at the real shapes (64 heads, a 512-wide
//!   latent, the 2,051 slots the stock selection picks), a 16K prompt's main chunk (16,356 keys, its 14,305 sparse rows)
//!   and its 31-row tail (16,387 keys), with random weights:
//!       TURBOEXL3_GLM53F_DSA_ONEP case=<name> rows=<L> keys=<K> max_abs=<x> max_ref=<x> rel=<x> finite=<0|1> ran=<n>
//!       PASS|FAIL
//!   PASS: finite output and max_abs / max_ref at most 1e-2 (the bound for a rounding-class prefill text).
//! The run's last line: `TURBOEXL3_GLM53F_DSA_PARITY onep=<n> fails=<n> PASS|FAIL`.
//! - `DSA_VSEL`, bit for bit: `selectTokens` over a verify window (1 to 4 rows, the cache's last ones) with the switch
//!   off (one-query launches a row, joined) and on (two launches), at a 16K context (the window's pools completing
//!   between its rows, a partial tail pool), its end (17,411 keys), 2K and 32K contexts, and with every head weight zero
//!   and keys repeating every 7 pools. The switch runs at 2 rows or more (ran=1) and leaves one row to the decode's own
//!   launches (ran=0). One line each, then the summary:
//!       TURBOEXL3_GLM53F_DSA_BYTES vsel case=<name> rows=<S> pools=<P> differing=<n>/<total> ran=<n> PASS|FAIL
//!       TURBOEXL3_GLM53F_DSA_VSEL_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const dsa = @import("mlx_serve/plugins/glm5_next/dsa_nax.zig");
const dsa_prompt = @import("dsa_prompt.zig");
const dsa_verify = @import("dsa_verify.zig");
const Array = mlx.mlx_array;

/// GLM-5.3-Flash's indexer and latent attention.
pub const ix: glm5.Indexer = .{ .topk = 2048, .kpool = 4, .heads = 32, .head_dim = 128 };
const heads: c_int = 64;
const latent_w: c_int = 512;
const attn_scale: f32 = 0.0625;

/// The selection cases: the keys in the cache, the query rows (the last ones), and what the inputs hold.
pub const Fill = enum { random, zero_weights, repeating_keys };
pub const SelCase = struct { name: []const u8, t: c_int, rows: c_int, fill: Fill };
pub const sel_cases = [_]SelCase{
    .{ .name = "main16k", .t = 16356, .rows = 14305, .fill = .random },
    .{ .name = "tail16k", .t = 16387, .rows = 31, .fill = .random },
};
/// The verify-window cases: the cache's keys (the window's rows the last of them), the rows, what the inputs hold.
pub const vsel_cases = [_]SelCase{
    .{ .name = "verify16k", .t = 16390, .rows = 3, .fill = .random },
    .{ .name = "verify16k_rows2", .t = 16389, .rows = 2, .fill = .random },
    .{ .name = "verify16k_rows4", .t = 16391, .rows = 4, .fill = .random },
    .{ .name = "verify16k_rows1", .t = 16388, .rows = 1, .fill = .random },
    .{ .name = "verify16k_end", .t = 17411, .rows = 3, .fill = .random },
    .{ .name = "verify2k", .t = 2103, .rows = 3, .fill = .random },
    .{ .name = "verify32k", .t = 32771, .rows = 4, .fill = .random },
    .{ .name = "verify16k_zero_weights", .t = 16390, .rows = 3, .fill = .zero_weights },
    .{ .name = "verify16k_repeating_keys", .t = 16390, .rows = 3, .fill = .repeating_keys },
};
/// The one-P cases: the selection case whose inputs they reuse.
pub const onep_cases = [_][]const u8{ "main16k", "tail16k" };

pub fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
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

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// The selection's inputs for `c`: q_idx bf16 [1, rows, 32, 128], the head weights f32 [1, rows, 32], the indexer cache
/// bf16 [1, t, 256] (each row's key | gate halves; a pool's key is its completing row's gate half).
pub const SelInputs = struct {
    q: Array,
    w: Array,
    cache: Array,
    pub fn deinit(x: *SelInputs) void {
        for ([_]Array{ x.q, x.w, x.cache }) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn selInputs(s: mlx.mlx_stream, seed: *u64, c: SelCase) !SelInputs {
    const q = try normal(s, seed, &.{ 1, c.rows, ix.heads, ix.head_dim }, .bfloat16, 1.0);
    errdefer _ = mlx.mlx_array_free(q);
    var w = try normal(s, seed, &.{ 1, c.rows, ix.heads }, .float32, 1.0);
    errdefer _ = mlx.mlx_array_free(w);
    if (c.fill == .zero_weights) {
        _ = mlx.mlx_array_free(w);
        w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 1, c.rows, ix.heads }, 3, .float32, s));
        try mlx.check(mlx.mlx_array_eval(w));
    }
    const width = 2 * ix.head_dim;
    const cache = if (c.fill == .repeating_keys) blk: {
        // 7 pools' rows tiled over the cache: pool p's key is pool (p mod 7)'s
        const period: c_int = 7 * ix.kpool;
        const base = try normal(s, seed, &.{ 1, period, width }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(base);
        var tiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tiled);
        try mlx.check(mlx.mlx_tile(&tiled, base, &[_]c_int{ 1, @divTrunc(c.t + period - 1, period), 1 }, 3, s));
        var cut = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cut);
        try mlx.check(mlx.mlx_slice(&cut, tiled, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, c.t, width }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var cont = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cont);
        try mlx.check(mlx.mlx_contiguous(&cont, cut, false, s));
        try mlx.check(mlx.mlx_array_eval(cont));
        break :blk cont;
    } else try normal(s, seed, &.{ 1, c.t, width }, .bfloat16, 1.0);
    return .{ .q = q, .w = w, .cache = cache };
}

/// int32 arrays `a` and `b` (the same shape) compared element by element: the count that differ, and the total.
const I32Diff = struct { differing: usize, total: usize };
fn diffI32(s: mlx.mlx_stream, a: Array, b: Array) !I32Diff {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return error.DsaGpuShape;
    if (mlx.mlx_array_dtype(a) != .int32 or mlx.mlx_array_dtype(b) != .int32) return error.DsaGpuDtype;
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    try evalAll(&.{ ac, bc });
    const n = mlx.mlx_array_size(ac);
    const pa = (mlx.mlx_array_data_int32(ac) orelse return error.DsaGpuData)[0..n];
    const pb = (mlx.mlx_array_data_int32(bc) orelse return error.DsaGpuData)[0..n];
    var d: usize = 0;
    for (pa, pb) |x, y| d += @intFromBool(x != y);
    return .{ .differing = d, .total = n };
}

/// glm5_next's selection, evaluated.
fn select(s: mlx.mlx_stream, x: SelInputs) !Array {
    const out = try glm5.selectTokens(s, x.q, x.w, x.cache, ix);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// glm5_next's selection with DSA_VSEL as `on`, evaluated.
fn selectVsel(s: mlx.mlx_stream, x: SelInputs, on: bool) !Array {
    dsa_verify.vsel_on = on;
    defer dsa_verify.vsel_on = false;
    const out = try glm5.selectTokens(s, x.q, x.w, x.cache, ix);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The one-P form against the plain one: the largest absolute difference, the plain output's largest magnitude, and
/// whether the one-P output is finite (its sum is).
const OnePStats = struct { max_abs: f32, max_ref: f32, finite: bool };
fn onePStats(s: mlx.mlx_stream, plain: Array, onep: Array) !OnePStats {
    var pf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_astype(&pf, plain, .float32, s));
    var of = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(of);
    try mlx.check(mlx.mlx_astype(&of, onep, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, of, pf, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, s));
    var ap = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ap);
    try mlx.check(mlx.mlx_abs(&ap, pf, s));
    var mr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mr);
    try mlx.check(mlx.mlx_max(&mr, ap, false, s));
    var sm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sm);
    try mlx.check(mlx.mlx_sum(&sm, of, false, s));
    try evalAll(&.{ mx, mr, sm });
    var r: OnePStats = .{ .max_abs = 0, .max_ref = 0, .finite = false };
    var total: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&r.max_abs, mx));
    try mlx.check(mlx.mlx_array_item_float32(&r.max_ref, mr));
    try mlx.check(mlx.mlx_array_item_float32(&total, sm));
    r.finite = std.math.isFinite(total) and std.math.isFinite(r.max_abs);
    return r;
}

test "turboexl3_glm53f dsa gpu: the one-P sparse attention stays within 1e-2 of the plain form at GLM-5.3-Flash's prompt shapes" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer dsa_prompt.onep_on = false;
    const s = mlx.gpuStream();
    var seed: u64 = 20261107;
    var fails: usize = 0;
    var onep_lines: usize = 0;
    for (sel_cases) |c| {
        var x = try selInputs(s, &seed, c);
        defer x.deinit();
        const off = try select(s, x);
        defer _ = mlx.mlx_array_free(off);
        // the one-P sparse attention over the slots this selection picked, at the case's rows and keys
        for (onep_cases) |name| if (std.mem.eql(u8, name, c.name)) {
            const q = try normal(s, &seed, &.{ 1, heads, c.rows, latent_w }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(q);
            const lat = try normal(s, &seed, &.{ 1, 1, c.t, latent_w }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(lat);
            const plain = (try dsa.sparseLatent(s, q, lat, off, attn_scale)) orelse return error.DsaGpuNoNax;
            defer _ = mlx.mlx_array_free(plain);
            dsa_prompt.onep_on = true;
            const before_p = dsa_prompt.onep_runs;
            const onep = (try dsa_prompt.sparseOneP(s, q, lat, off, attn_scale)) orelse return error.DsaGpuNoNax;
            defer _ = mlx.mlx_array_free(onep);
            dsa_prompt.onep_on = false;
            try evalAll(&.{ plain, onep });
            const st = try onePStats(s, plain, onep);
            const rel = if (st.max_ref > 0) st.max_abs / st.max_ref else std.math.inf(f32);
            const ok_p = st.finite and rel <= 1e-2 and dsa_prompt.onep_runs - before_p == 1;
            if (!ok_p) fails += 1;
            onep_lines += 1;
            std.debug.print("TURBOEXL3_GLM53F_DSA_ONEP case={s} rows={d} keys={d} max_abs={e:.3} max_ref={e:.3} rel={e:.3} finite={d} ran={d} {s}\n", .{ c.name, c.rows, c.t, st.max_abs, st.max_ref, rel, @intFromBool(st.finite), dsa_prompt.onep_runs - before_p, if (ok_p) "PASS" else "FAIL" });
        };
    }
    std.debug.print("TURBOEXL3_GLM53F_DSA_PARITY onep={d} fails={d} {s}\n", .{ onep_lines, fails, if (fails == 0 and onep_lines == onep_cases.len) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
    try std.testing.expectEqual(onep_cases.len, onep_lines);
}

test "turboexl3_glm53f dsa gpu: the verify window's two-launch selection writes the per-row launches' tokens bit for bit at 1 to 4 rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer dsa_verify.vsel_on = false;
    const s = mlx.gpuStream();
    var seed: u64 = 20261108;
    var fails: usize = 0;
    for (vsel_cases) |c| {
        var x = try selInputs(s, &seed, c);
        defer x.deinit();
        const off = try selectVsel(s, x, false);
        defer _ = mlx.mlx_array_free(off);
        const before = dsa_verify.vsel_runs;
        const on = try selectVsel(s, x, true);
        defer _ = mlx.mlx_array_free(on);
        const ran = dsa_verify.vsel_runs - before;
        const d = try diffI32(s, off, on);
        const ok = d.differing == 0 and d.total > 0 and ran == @intFromBool(c.rows >= 2);
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES vsel case={s} rows={d} pools={d} differing={d}/{d} ran={d} {s}\n", .{ c.name, c.rows, @divTrunc(c.t, ix.kpool), d.differing, d.total, ran, if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_GLM53F_DSA_VSEL_PARITY lines={d} fails={d} {s}\n", .{ vsel_cases.len, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
