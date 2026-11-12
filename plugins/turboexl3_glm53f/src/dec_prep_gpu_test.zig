//! The prepared-input producers' device check (src/dec_prep.zig), on the server's MLX: `zig build dec-prep-gpu-test
//! -Dmlx-lib=<dir>` under TURBOEXL3_GLM53F_DEC_PREP_GPU=1 (unset: skipped). Each producer against the stock GLM chain on
//! the same inputs, bit for bit, at 2, 3 and 4 rows, under DEC-ONE's default GEMV and g1 (TURBOEXL3_GLM53F_DEC=g1, the
//! control's), each variant's engines made anew:
//! - KDA_OPREP: kda_prompt.normGateText's out projection input, then `t3.linear` (prep_sc + DEC-ONE's GEMV) at
//!   GLM-5.3-Flash's 8192 -> 4096, against `dec_prep.kdaOut`;
//! - DSA_OPREP: the per-head values [1, 64, rows, 256] transposed and reshaped into row order (MLX's copy), then
//!   `t3.linear` at 16384 -> 4096, against `dec_prep.dsaOut`.
//! One and five rows decline (null). Lines:
//!
//!     TURBOEXL3_GLM53F_DEC_PREP_BYTES dec=<default|g1> piece=<kda_o|dsa_o> rows=<n> differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DEC_PREP_DECLINED piece=<kda|dsa_o> rows=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_DEC_PREP_PARITY lines=<n> fails=<n> kda_oprep=<n> dsa_oprep=<n> declined=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const dec_prep = @import("dec_prep.zig");
const kda_prompt = @import("kda_prompt.zig");
const routed_t3 = @import("routed_t3.zig");
const Array = mlx.mlx_array;

/// GLM-5.3-Flash's norm epsilon and the trunk's K6
const rms_eps: f32 = 1e-6;
const rate_k: u32 = 6;
const hidden: u32 = 4096;
const prep_rows = [_]c_int{ 2, 3, 4 };
const variants = [_][]const u8{ "", "g1" };
/// the byte lines a variant prints a row count: the KDA out projection and the DSA out projection
const pieces_per_rows: usize = 2;
pub const byte_lines = variants.len * prep_rows.len * pieces_per_rows;
pub const decline_lines = 2 * 2;

fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, .bfloat16, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A norm weight [d] bf16 in [0.5, 1.5]
fn weight(s: mlx.mlx_stream, seed: *u64, d: c_int) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    const lo = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.5);
    defer _ = mlx.mlx_array_free(hi);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_uniform(&f, lo, hi, &[_]c_int{d}, 1, .float32, key, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn f16Bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// A projection `in -> out` at rate `k` as the pack stores it: random codewords [in/16, out/16, 16k], suh f16 [in] in
/// [0.5, 1.5], svh f16 [out] in [0.5, 1.5] / sqrt(in) (src/smallop_gpu_test.zig's)
fn proj(a: std.mem.Allocator, rnd: std.Random, in: u32, out: u32, k: u32) !t3.Proj {
    const n = 16 * k;
    const t = try a.alloc(u16, (in / 16) * (out / 16) * n);
    defer a.free(t);
    rnd.bytes(std.mem.sliceAsBytes(t));
    const su = try a.alloc(u16, in);
    defer a.free(su);
    for (su) |*v| v.* = f16Bits(0.5 + rnd.float(f32));
    const sv = try a.alloc(u16, out);
    defer a.free(sv);
    for (sv) |*v| v.* = f16Bits((0.5 + rnd.float(f32)) / @sqrt(@as(f32, @floatFromInt(in))));
    const p: t3.Proj = .{
        .trellis = mlx.mlx_array_new_data(t.ptr, &[_]c_int{ @intCast(in / 16), @intCast(out / 16), @intCast(n) }, 3, .uint16),
        .suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{@intCast(in)}, 1, .float16),
        .svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{@intCast(out)}, 1, .float16),
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

const Diff = struct { differing: usize, total: usize, nonfinite: usize };

/// `got` against `want` element by element, as bits (both made contiguous: copies of the same bits)
fn diff(s: mlx.mlx_stream, want: Array, got: Array) !Diff {
    if (mlx.mlx_array_dtype(want) != .bfloat16 or mlx.mlx_array_dtype(got) != .bfloat16) return error.DecPrepGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.DecPrepGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    var d: Diff = .{ .differing = 0, .total = n, .nonfinite = 0 };
    const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.DecPrepGpuData)[0..n];
    const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.DecPrepGpuData)[0..n];
    for (pw, pg) |x, y| {
        if (x != y) d.differing += 1;
        if ((y & 0x7F80) == 0x7F80) d.nonfinite += 1;
    }
    return d;
}

const Tally = struct {
    lines: usize = 0,
    fails: usize = 0,
    fn check(t: *Tally, s: mlx.mlx_stream, dec: []const u8, piece: []const u8, rows: c_int, want: Array, got: Array) !void {
        const d = try diff(s, want, got);
        const ok = d.total > 0 and d.differing == 0 and d.nonfinite == 0;
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_DEC_PREP_BYTES dec={s} piece={s} rows={d} differing={d}/{d} nonfinite={d} {s}\n", .{ if (dec.len == 0) "default" else dec, piece, rows, d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
    }
    fn declined(t: *Tally, piece: []const u8, rows: c_int, got: ?Array) void {
        if (got) |x| _ = mlx.mlx_array_free(x);
        t.declinedIf(piece, rows, got == null);
    }
    fn declinedIf(t: *Tally, piece: []const u8, rows: c_int, ok: bool) void {
        t.lines += 1;
        t.fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_DEC_PREP_DECLINED piece={s} rows={d} {s}\n", .{ piece, rows, if (ok) "PASS" else "FAIL" });
    }
};

fn linear(s: mlx.mlx_stream, x: Array, p: t3.Proj) !Array {
    return t3.linear(s, x, p.trellis, p.suh, p.svh, .mul1);
}

/// DEC-ONE's GEMV variant `name` ("" today's) for the engines made from here on, and every engine made so far dropped.
fn useVariant(s: mlx.mlx_stream, name: []const u8) !void {
    try mlx.check(mlx.mlx_synchronize(s));
    const v = if (name.len == 0) 0 else (t3.dense.decVariantOf(name) orelse return error.DecPrepGpuVariant);
    try routed_t3.DecSchedule.apply(.{ .variant = v });
    t3.releaseEngines();
}

test "turboexl3_glm53f dec_prep gpu: KDA_OPREP's and DSA_OPREP's producers give the stock GLM chain's bytes (the gated norm and the per-head values' copy, each then prep_sc and DEC-ONE's GEMV) at 2, 3 and 4 rows under the default and g1 GEMVs, and decline one and five rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DEC_PREP_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var seed: u64 = 20261109 + 101;
    var prng = std.Random.DefaultPrng.init(20261109 + 102);
    const rnd = prng.random();
    const saved = .{ dec_prep.kda_calls, dec_prep.dsa_o_calls, dec_prep.declined };
    dec_prep.kda_calls = 0;
    dec_prep.dsa_o_calls = 0;
    dec_prep.declined = 0;
    dec_prep.kda_on = true;
    dec_prep.dsa_o_on = true;
    defer {
        dec_prep.kda_on = false;
        dec_prep.dsa_o_on = false;
        dec_prep.kda_calls = saved[0];
        dec_prep.dsa_o_calls = saved[1];
        dec_prep.declined = saved[2];
    }
    defer useVariant(s, "") catch {};
    var t: Tally = .{};
    // GLM-5.3-Flash's projections at K6: the KDA out projection and the DSA out projection
    const kda_o = try proj(a, rnd, dec_prep.kda_in, hidden, rate_k);
    defer freeProj(kda_o);
    const dsa_o = try proj(a, rnd, dec_prep.dsa_in, hidden, rate_k);
    defer freeProj(dsa_o);
    const norm_w = try weight(s, &seed, dec_prep.kda_dv);
    defer _ = mlx.mlx_array_free(norm_w);
    for (variants) |vname| {
        try useVariant(s, vname);
        for (prep_rows) |rows| {
            // KDA_OPREP: the gated norm's text, then the out projection
            {
                const y = try normal(s, &seed, &.{ 1, rows, dec_prep.kda_heads, dec_prep.kda_dv }, 1.0);
                defer _ = mlx.mlx_array_free(y);
                const z = try normal(s, &seed, &.{ 1, rows, dec_prep.kda_heads * dec_prep.kda_dv }, 2.0);
                defer _ = mlx.mlx_array_free(z);
                const flat = (try kda_prompt.normGateText(s, y, z, norm_w, rms_eps, dec_prep.kda_heads, dec_prep.kda_dv)) orelse return error.DecPrepGpuNormGateDeclined;
                defer _ = mlx.mlx_array_free(flat);
                const want = try linear(s, flat, kda_o);
                defer _ = mlx.mlx_array_free(want);
                const got = (try dec_prep.kdaOut(s, y, z, norm_w, rms_eps, dec_prep.kda_heads, dec_prep.kda_dv, kda_o)) orelse return error.DecPrepGpuKdaDeclined;
                defer _ = mlx.mlx_array_free(got);
                try t.check(s, vname, "kda_o", rows, want, got);
            }
            // DSA_OPREP: the per-head values copied into row order, then the out projection
            {
                const o = try normal(s, &seed, &.{ 1, dec_prep.dsa_heads, rows, dec_prep.dsa_hd }, 1.0);
                defer _ = mlx.mlx_array_free(o);
                var ot = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ot);
                try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
                var flat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(flat);
                try mlx.check(mlx.mlx_reshape(&flat, ot, &[_]c_int{ 1, rows, dec_prep.dsa_heads * dec_prep.dsa_hd }, 3, s));
                const want = try linear(s, flat, dsa_o);
                defer _ = mlx.mlx_array_free(want);
                const got = (try dec_prep.dsaOut(s, o, dsa_o)) orelse return error.DecPrepGpuDsaODeclined;
                defer _ = mlx.mlx_array_free(got);
                try t.check(s, vname, "dsa_o", rows, want, got);
            }
        }
    }
    // one row (DEC-ONE's prepare inside its launch) and five (past DEC-ONE) decline
    for ([_]c_int{ 1, 5 }) |rows| {
        const y = try normal(s, &seed, &.{ 1, rows, dec_prep.kda_heads, dec_prep.kda_dv }, 1.0);
        defer _ = mlx.mlx_array_free(y);
        const z = try normal(s, &seed, &.{ 1, rows, dec_prep.kda_heads * dec_prep.kda_dv }, 1.0);
        defer _ = mlx.mlx_array_free(z);
        t.declined("kda", rows, try dec_prep.kdaOut(s, y, z, norm_w, rms_eps, dec_prep.kda_heads, dec_prep.kda_dv, kda_o));
        const o = try normal(s, &seed, &.{ 1, dec_prep.dsa_heads, rows, dec_prep.dsa_hd }, 1.0);
        defer _ = mlx.mlx_array_free(o);
        t.declined("dsa_o", rows, try dec_prep.dsaOut(s, o, dsa_o));
    }
    const n = variants.len * prep_rows.len;
    const counts_ok = dec_prep.kda_calls == n and dec_prep.dsa_o_calls == n and dec_prep.declined == 0;
    const ok = t.fails == 0 and t.lines == byte_lines + decline_lines and counts_ok;
    std.debug.print("TURBOEXL3_GLM53F_DEC_PREP_PARITY lines={d} fails={d} kda_oprep={d} dsa_oprep={d} declined={d} {s}\n", .{ t.lines, t.fails, dec_prep.kda_calls, dec_prep.dsa_o_calls, dec_prep.declined, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}
