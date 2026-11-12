//! turboexl3_ling's latent MLA decode at the served state (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs it, else it skips).
//! It checks src/ling_mla_latent.zig `decode` against the expanded cache's attention. `decode` is the composed chain that
//! the copied mla.zig's `latentAttn` runs for the MTP head's latent cache where turboexl3's launches decline. The head
//! loads kv_b with the trunk's loader (trunk.zig `fn mla`), so these operands are the head's.
//! It then holds turboexl3's fused launches (src/ling_mla_fused.zig `attend`, as the load configures them: launch 2 on
//! the matrix units, MLA_LATENT_NAX, and staged in f32 at the calls NAX does not take, MLA_LATENT_STAGE) to `decode`,
//! the composed chain they replace, at the head's calls.
//! It also checks `kvRows`, the operand turboexl3's launches read. The operands are built the way the loader and the
//! runtime build them:
//! - kv_b: a dense f16 tensor stored `[H (nope + v), kv_lora]` as the checkpoint stores it, then held as the loader holds
//!   it: the `[kv_lora, H (nope + v)]` view that a plain matmul reads, cast to bf16 and evaluated (trunk.zig
//!   `transposed`, `lin`).
//! - The expanded keys and values: made from that kv_b as upstream mla.zig's expanded cache makes them (`latent @ kv_b`,
//!   the heads cut from it, the shared rope key broadcast), then MLX's SDPA, each row over the keys up to it.
//! - The latent cache: the first T rows of a wider buffer, the call's rows its last S. Past T the buffer holds other
//!   rows, as it does after a rejected draft is rolled back.
//! - q_nope and q_rot: strided views of the call's transposed query heads, as mlaAttn cuts them. RoPE runs before the
//!   cache's branch on both paths, so it is not part of this test.
//! Ling-3.1-flash's MLA: 64 heads, nope 128, rope 64, v 128, kv_lora 512, bf16. Each case prints a
//! TURBOEXL3_LING_MLA_LATENT_SERVED (or, for the fused launches, TURBOEXL3_LING_MLA_LATENT_FUSED) line ending in PASS or
//! FAIL. Nothing is timed.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const lml = @import("ling_mla_latent.zig");
const fused = @import("ling_mla_fused.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

const H: c_int = 64;
const DN: c_int = 128;
const DR: c_int = 64;
const DV: c_int = 128;
const DL: c_int = 512;
const ROWS: c_int = H * (DN + DV);
const geo: lml.Geometry = .{ .heads = H, .nope = DN, .v = DV, .kv_lora = DL };

/// MLX_SERVE_LING_GPU_TESTS=1 (the tests' own variable) and a GPU stream, else the test skips
fn gpu() !Stream {
    const gate = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(gate), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// normal(0, `sd`) in `dt` from `seed`, evaluated
fn randn(s: Stream, seed: u64, shape: []const c_int, sd: f32, dt: mlx.mlx_dtype) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0, sd, key, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// `ckpt` (a dense f16 `[out, in]` checkpoint tensor) as the loader holds it: trunk.zig `transposed` (`transposeLast`,
/// the `[in, out]` view a plain matmul reads), then `lin`'s cast of a plain f16 weight to bf16, evaluated (`astypeEval`)
fn loaderHeld(ckpt: Array, s: Stream) !Array {
    var t = mlx.mlx_array_new();
    defer free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, ckpt, &[_]c_int{ 1, 0 }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_astype(&out, t, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// a[..., start:stop, :] along axis 2 (a view)
fn rows4(a: Array, start: c_int, stop: c_int, s: Stream) !Array {
    const sh = mlx.getShape(a);
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(mlx.mlx_slice(&r, a, &[_]c_int{ 0, 0, start, 0 }, 4, &[_]c_int{ sh[0], sh[1], stop, sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return r;
}

/// a[..., start:stop] along axis 3 (a view)
fn cols4(a: Array, start: c_int, stop: c_int, s: Stream) !Array {
    const sh = mlx.getShape(a);
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(mlx.mlx_slice(&r, a, &[_]c_int{ 0, 0, 0, start }, 4, &[_]c_int{ sh[0], sh[1], sh[2], stop }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return r;
}

/// A contiguous copy of `a`, evaluated
fn copy(a: Array, s: Stream) !Array {
    var c = mlx.mlx_array_new();
    errdefer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

/// `a` and `b` joined on the last axis
fn cat(a: Array, b: Array, s: Stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer free(r);
    const parts = [_]Array{ a, b };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, -1, s));
    return r;
}

fn sdpa(q: Array, k: Array, v: Array, scale: f32, s: Stream) !Array {
    const none = mlx.mlx_array_new();
    defer free(none);
    var o = mlx.mlx_array_new();
    errdefer free(o);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", none, .{ .ctx = null }, false, s));
    return o;
}

/// mla.zig's expanded cache at the call's rows: each query row r of q `[1, H, S, D]` in one MLX SDPA call over the
/// first T - S + r + 1 keys (`rowAttn`, as the exact verify runs a head call of several rows; one row is MLX's one-row
/// SDPA, as a draft step runs)
fn rowAttn(q: Array, k: Array, v: Array, scale: f32, s: Stream) !Array {
    const qs = mlx.getShape(q);
    const seq_len = qs[2];
    const first = mlx.getShape(k)[2] - seq_len;
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var r: c_int = 0;
    while (r < seq_len) : (r += 1) {
        var qr = mlx.mlx_array_new();
        defer free(qr);
        try mlx.check(mlx.mlx_slice(&qr, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ qs[0], qs[1], r + 1, qs[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const kr = try rows4(k, 0, first + r + 1, s);
        defer free(kr);
        const vr = try rows4(v, 0, first + r + 1, s);
        defer free(vr);
        const o = try sdpa(qr, kr, vr, scale, s);
        defer free(o);
        _ = mlx.mlx_vector_array_append_value(parts, o);
    }
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 2, s));
    return out;
}

/// The expanded cache's attention of the call's rows (the head's attention on upstream mla.zig's expanded cache): the
/// latent rows through the loader's kv_b as that form makes them (`self.mm(&fa.kv_b, latent)`, a plain matmul of the
/// `[in, out]` view), the heads' keys and values cut from them, the shared rope key broadcast, then `rowAttn`
fn expanded(q_nope: Array, q_rot: Array, lat: Array, rot: Array, held: Array, scale: f32, s: Stream) !Array {
    const t = mlx.getShape(lat)[2];
    var lat3 = mlx.mlx_array_new();
    defer free(lat3);
    try mlx.check(mlx.mlx_reshape(&lat3, lat, &[_]c_int{ 1, t, DL }, 3, s));
    var kv_flat = mlx.mlx_array_new();
    defer free(kv_flat);
    try mlx.check(mlx.mlx_matmul(&kv_flat, lat3, held, s));
    var kv_heads = mlx.mlx_array_new();
    defer free(kv_heads);
    try mlx.check(mlx.mlx_reshape(&kv_heads, kv_flat, &[_]c_int{ 1, t, H, DN + DV }, 4, s));
    var kv_t = mlx.mlx_array_new();
    defer free(kv_t);
    try mlx.check(mlx.mlx_transpose_axes(&kv_t, kv_heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const k_nope = try cols4(kv_t, 0, DN, s);
    defer free(k_nope);
    const v_t = try cols4(kv_t, DN, DN + DV, s);
    defer free(v_t);
    var k_rot_b = mlx.mlx_array_new();
    defer free(k_rot_b);
    try mlx.check(mlx.mlx_broadcast_to(&k_rot_b, rot, &[_]c_int{ 1, H, t, DR }, 4, s));
    const keys = try cat(k_nope, k_rot_b, s);
    defer free(keys);
    const v = try copy(v_t, s);
    defer free(v);
    const queries = try cat(q_nope, q_rot, s);
    defer free(queries);
    return rowAttn(queries, keys, v, scale, s);
}

/// The relative error of `a` against `b`, ||a - b|| / ||b||, and the largest word's, max |a - b| / max |b|, in f64 on
/// the host (a shape mismatch is infinite)
fn relErr(a: Array, b: Array, s: Stream) ![2]f64 {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return .{ std.math.inf(f64), std.math.inf(f64) };
    var fa = mlx.mlx_array_new();
    defer free(fa);
    try mlx.check(mlx.mlx_astype(&fa, a, .float32, s));
    var fb = mlx.mlx_array_new();
    defer free(fb);
    try mlx.check(mlx.mlx_astype(&fb, b, .float32, s));
    const ca = try copy(fa, s);
    defer free(ca);
    const cb = try copy(fb, s);
    defer free(cb);
    const n = mlx.mlx_array_size(cb);
    const pa = mlx.mlx_array_data_float32(ca) orelse return error.NoData;
    const pb = mlx.mlx_array_data_float32(cb) orelse return error.NoData;
    var num: f64 = 0;
    var den: f64 = 0;
    var max_d: f64 = 0;
    var max_b: f64 = 0;
    for (0..n) |i| {
        const x: f64 = pa[i];
        const y: f64 = pb[i];
        num += (x - y) * (x - y);
        den += y * y;
        max_d = @max(max_d, @abs(x - y));
        max_b = @max(max_b, @abs(y));
    }
    return .{ @sqrt(num / den), max_d / max_b };
}

/// The words of `a` and `b` that differ (a shape or dtype mismatch counts every word)
fn differing(a: Array, b: Array, s: Stream) !usize {
    const n = mlx.mlx_array_size(b);
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b)) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return n;
    const ca = try copy(a, s);
    defer free(ca);
    const cb = try copy(b, s);
    defer free(cb);
    const isz = mlx.mlx_array_itemsize(cb);
    const pa = mlx.mlx_array_data_uint8(ca) orelse return error.NoData;
    const pb = mlx.mlx_array_data_uint8(cb) orelse return error.NoData;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(!std.mem.eql(u8, pa[i * isz ..][0..isz], pb[i * isz ..][0..isz]));
    return d;
}

/// The checkpoint's kv_b, f16 `[H (nope + v), kv_lora]`, about the scale of a trained projection's words
fn checkpointKvB(s: Stream, seed: u64) !Array {
    return randn(s, seed, &.{ ROWS, DL }, 1.0 / @sqrt(@as(f32, @floatFromInt(DL))), .float16);
}

/// The relative error past which `decode`, or a fused launch against it, is wrong, not rounded: the expanded cache rounds
/// each key and value to bf16, and the NAX launch the absorbed query (turboexl3's own test holds it to 2^-8 of an f64
/// reference's RMS), which leaves well under 1 %.
const tolerance: f64 = 0.02;

/// One call of `rows` rows over a cache of `t` rows, the first t of a `cap`-row buffer: `decode` against `expanded`.
/// The control is `decode` given kv_b's loader view read as rows (what the copied mla.zig's latent decode did before
/// this file), which `expanded` must tell apart.
fn runCase(s: Stream, held: Array, t: c_int, rows: c_int, cap: c_int, seed: u64, what: []const u8) !bool {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DN + DR)));
    const lat_buf = try randn(s, seed, &.{ 1, 1, cap, DL }, 1.0, .bfloat16);
    defer free(lat_buf);
    const rot_buf = try randn(s, seed + 1, &.{ 1, 1, cap, DR }, 1.0, .bfloat16);
    defer free(rot_buf);
    const lat = try rows4(lat_buf, 0, t, s);
    defer free(lat);
    const rot = try rows4(rot_buf, 0, t, s);
    defer free(rot);
    // mlaAttn's cut: the query heads [1, S, H, nope + rope] transposed to [1, H, S, nope + rope], then each half a view
    const q_heads = try randn(s, seed + 2, &.{ 1, rows, H, DN + DR }, 1.0, .bfloat16);
    defer free(q_heads);
    var q_t = mlx.mlx_array_new();
    defer free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const q_nope = try cols4(q_t, 0, DN, s);
    defer free(q_nope);
    const q_rot = try cols4(q_t, DN, DN + DR, s);
    defer free(q_rot);
    const want = try expanded(q_nope, q_rot, lat, rot, held, scale, s);
    defer free(want);
    var got = mlx.mlx_array_new();
    defer free(got);
    try lml.decode(&got, q_nope, q_rot, lat, rot, held, geo, scale, s);
    const e = try relErr(got, want, s);
    // the control: the loader's view read in row-major order as [H (nope + v), kv_lora] rows
    var misread = mlx.mlx_array_new();
    defer free(misread);
    try mlx.check(mlx.mlx_reshape(&misread, held, &[_]c_int{ ROWS, DL }, 2, s));
    var ctl = mlx.mlx_array_new();
    defer free(ctl);
    try lml.decode(&ctl, q_nope, q_rot, lat, rot, misread, geo, scale, s);
    const c = try relErr(ctl, want, s);
    const ok = e[0] <= tolerance and c[0] > 25 * tolerance;
    std.debug.print("TURBOEXL3_LING_MLA_LATENT_SERVED {s}: keys={d} of a {d}-row buffer, rows={d}: decode against the expanded cache rel_l2={d:.5} rel_max={d:.5} (tolerance {d}); control (kv_b's loader view read as rows) rel_l2={d:.3} {s}\n", .{ what, t, cap, rows, e[0], e[1], tolerance, c[0], if (ok) "PASS" else "FAIL" });
    return ok;
}

test "turboexl3_ling latent MLA: kv_b as the loader holds it; kvRows gives back the checkpoint's stored rows (GPU)" {
    const s = try gpu();
    const ckpt = try checkpointKvB(s, 0x5EED_1A7E);
    defer free(ckpt);
    const held = try loaderHeld(ckpt, s);
    defer free(held);
    const r = try lml.kvRows(held, geo, s);
    defer free(r);
    try mlx.check(mlx.mlx_array_eval(r));
    var want = mlx.mlx_array_new();
    defer free(want);
    try mlx.check(mlx.mlx_astype(&want, ckpt, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(want));
    const hs = mlx.getShape(held);
    const rs = mlx.getShape(r);
    const d = try differing(r, want, s);
    // Where the rows sit: a row-contiguous view of the loader's buffer is what turboexl3's launches read as it is.
    const st = mlx.mlx_array_strides(r);
    const pr = mlx.mlx_array_data_uint8(r) orelse return error.NoData;
    const ph = mlx.mlx_array_data_uint8(held) orelse return error.NoData;
    const same = @intFromPtr(pr) == @intFromPtr(ph);
    const ok = hs[0] == DL and hs[1] == ROWS and rs[0] == ROWS and rs[1] == DL and d == 0;
    std.debug.print("TURBOEXL3_LING_MLA_LATENT_KVB the loader holds [{d}, {d}]; kvRows gives [{d}, {d}], strides [{d}, {d}], the loader's buffer: {}; words differing from the checkpoint's rows in bf16: {d}/{d} {s}\n", .{ hs[0], hs[1], rs[0], rs[1], st[0], st[1], same, d, mlx.mlx_array_size(want), if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}

test "turboexl3_ling latent MLA: the latent decode at the served state (the MTP head's calls) is the expanded cache's attention within rounding, kv_b and the cache built as the loader and the runtime build them (GPU)" {
    const s = try gpu();
    const ckpt = try checkpointKvB(s, 0x5EED_1A7F);
    defer free(ckpt);
    const held = try loaderHeld(ckpt, s);
    defer free(held);
    var failed: usize = 0;
    // 18,380 keys: the standard prompt's history at the first decode step. 1 to 4 rows: a head draft step (1) and the
    // head's merged step (2 to 4). A cache rolled back after a rejected draft: its first 18,381 rows, with other rows
    // past them in the buffer. Short and middle caches. The buffers are as the host's growth rounds them (256-row
    // steps).
    var seed: u64 = 11;
    for ([_]c_int{ 1, 2, 3, 4 }) |rows| {
        failed += @intFromBool(!try runCase(s, held, 18380, rows, 18432, seed, "the standard prompt's 18,380 keys"));
        seed += 10;
    }
    failed += @intFromBool(!try runCase(s, held, 18381, 2, 18432, 61, "after a rollback (other rows past the cache's)"));
    failed += @intFromBool(!try runCase(s, held, 18381, 4, 18432, 71, "after a rollback (other rows past the cache's)"));
    failed += @intFromBool(!try runCase(s, held, 300, 1, 512, 81, "a short cache"));
    failed += @intFromBool(!try runCase(s, held, 4100, 3, 4352, 91, "a middle cache"));
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// The text the load runs launch 2 of turboexl3's fused launches on (src/ling_math.zig `configure` sets turboexl3's
/// Config so): `nax`, the matrix units (MLA_LATENT_NAX), at every call of a bf16 or f16 cache; `stage`, staged in f32
/// (MLA_LATENT_STAGE), at the calls NAX does not take (here with NAX off, so every call).
const Arm = enum { nax, stage };

/// One head call of `rows` rows over a cache of `t` rows, the first t of a `cap`-row buffer: turboexl3's fused launches
/// (`fused.attend` over kv_b's stored rows, as mla.zig's `latentAttn` hands them) under `arm` against the composed chain
/// they replace (`lml.decode`), and the text launch 2 ran. The red check is the control: `attend` given kv_b's loader
/// view read as rows, which must land outside the tolerance.
fn runFused(s: Stream, held: Array, t: c_int, rows: c_int, cap: c_int, seed: u64, arm: Arm, what: []const u8) !bool {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DN + DR)));
    const lat_buf = try randn(s, seed, &.{ 1, 1, cap, DL }, 1.0, .bfloat16);
    defer free(lat_buf);
    const rot_buf = try randn(s, seed + 1, &.{ 1, 1, cap, DR }, 1.0, .bfloat16);
    defer free(rot_buf);
    const lat = try rows4(lat_buf, 0, t, s);
    defer free(lat);
    const rot = try rows4(rot_buf, 0, t, s);
    defer free(rot);
    // mlaAttn's cut, as runCase makes it
    const q_heads = try randn(s, seed + 2, &.{ 1, rows, H, DN + DR }, 1.0, .bfloat16);
    defer free(q_heads);
    var q_t = mlx.mlx_array_new();
    defer free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const q_nope = try cols4(q_t, 0, DN, s);
    defer free(q_nope);
    const q_rot = try cols4(q_t, DN, DN + DR, s);
    defer free(q_rot);
    // the oracle: the composed latent chain
    var want = mlx.mlx_array_new();
    defer free(want);
    try lml.decode(&want, q_nope, q_rot, lat, rot, held, geo, scale, s);
    const w_rows = try lml.kvRows(held, geo, s);
    defer free(w_rows);
    var got = mlx.mlx_array_new();
    defer free(got);
    const took = try fused.attend(&got, s, q_nope, q_rot, lat, rot, w_rows, scale);
    // the text launch 2 ran for this call (turboexl3's record), read before the control's call
    const ran_attn = t3.ling.mla_latent.lastAttn();
    // the arm's own text: a NAX text for `nax`, the staged text for `stage` (the base text fails the stage arm)
    const arm_text = switch (arm) {
        .nax => t3.ling.mla_latent.naxOf(ran_attn) != null,
        .stage => ran_attn == .stage,
    };
    const e: [2]f64 = if (took) try relErr(got, want, s) else .{ std.math.inf(f64), std.math.inf(f64) };
    // the control: the loader's view read in row-major order as [H (nope + v), kv_lora] rows
    var misread = mlx.mlx_array_new();
    defer free(misread);
    try mlx.check(mlx.mlx_reshape(&misread, held, &[_]c_int{ ROWS, DL }, 2, s));
    var ctl = mlx.mlx_array_new();
    defer free(ctl);
    const ctl_took = try fused.attend(&ctl, s, q_nope, q_rot, lat, rot, misread, scale);
    const c: [2]f64 = if (ctl_took) try relErr(ctl, want, s) else .{ 0, 0 };
    const ok = took and arm_text and e[0] <= tolerance and ctl_took and c[0] > 25 * tolerance;
    std.debug.print("TURBOEXL3_LING_MLA_LATENT_FUSED {s} {s}: keys={d} of a {d}-row buffer, rows={d}: took={} launch 2 ran {s}; against the composed chain rel_l2={d:.5} rel_max={d:.5} (tolerance {d}); control (kv_b's loader view read as rows) rel_l2={d:.3} {s}\n", .{ @tagName(arm), what, t, cap, rows, took, @tagName(ran_attn), e[0], e[1], tolerance, c[0], if (ok) "PASS" else "FAIL" });
    return ok;
}

test "turboexl3_ling latent MLA: the MTP head's fused launches as the load configures them (launch 2 on the matrix units; staged in f32 where NAX does not take a call) are the composed latent chain's attention within the tolerance at the head's calls, and kv_b's loader view read as rows is outside it (GPU)" {
    const s = try gpu();
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch {};
    if (!saved.nax) {
        std.debug.print("TURBOEXL3_LING_MLA_LATENT_FUSED skipped: the device fails turboexl3's NAX gate\n", .{});
        return error.SkipZigTest;
    }
    const ckpt = try checkpointKvB(s, 0x5EED_1A80);
    defer free(ckpt);
    const held = try loaderHeld(ckpt, s);
    defer free(held);
    var failed: usize = 0;
    for ([_]Arm{ .nax, .stage }) |arm| {
        var cf = saved;
        cf.mla_latent_stage = true;
        cf.mla_latent_nax = arm == .nax;
        try t3.configure(cf);
        // The served-state test's calls: a draft step (1 row) and the merged step (2 to 4) over the standard
        // prompt's 18,380 keys, a rollback, a short and a middle cache.
        var seed: u64 = 211;
        for ([_]c_int{ 1, 2, 3, 4 }) |rows| {
            failed += @intFromBool(!try runFused(s, held, 18380, rows, 18432, seed, arm, "the standard prompt's 18,380 keys"));
            seed += 10;
        }
        failed += @intFromBool(!try runFused(s, held, 18381, 2, 18432, 261, arm, "after a rollback (other rows past the cache's)"));
        failed += @intFromBool(!try runFused(s, held, 300, 1, 512, 281, arm, "a short cache"));
        failed += @intFromBool(!try runFused(s, held, 4100, 3, 4352, 291, arm, "a middle cache"));
    }
    failed += try stageRed(s, held, saved);
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// Whether launch 2 ran the base text, by name (the review's follow-up c: "anything but the staged text" took NAX's
/// text for the base, as stageRed's NAX case shows).
fn isBase(a: @TypeOf(t3.ling.mla_latent.lastAttn())) bool {
    return a == .base;
}

/// The stage arm's red check: with MLA_LATENT_STAGE and NAX off, launch 2 runs the base text (`isBase`), which the stage
/// arm's text check must reject (before the review's finding 4 the arm took any text but NAX's); with NAX on and the
/// stage off, launch 2 runs NAX's text, which is not the base text. 0 when both hold.
fn stageRed(s: Stream, held: Array, saved: anytype) !usize {
    var missed: usize = 0;
    for ([_]bool{ false, true }) |nax_on| {
        var cf = saved;
        cf.mla_latent_stage = false;
        cf.mla_latent_nax = nax_on;
        try t3.configure(cf);
        missed += try stageRedOne(s, held, nax_on);
    }
    return missed;
}

fn stageRedOne(s: Stream, held: Array, nax_on: bool) !usize {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DN + DR)));
    const lat = try randn(s, 401, &.{ 1, 1, 300, DL }, 1.0, .bfloat16);
    defer free(lat);
    const rot = try randn(s, 402, &.{ 1, 1, 300, DR }, 1.0, .bfloat16);
    defer free(rot);
    const q_heads = try randn(s, 403, &.{ 1, 1, H, DN + DR }, 1.0, .bfloat16);
    defer free(q_heads);
    var q_t = mlx.mlx_array_new();
    defer free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const q_nope = try cols4(q_t, 0, DN, s);
    defer free(q_nope);
    const q_rot = try cols4(q_t, DN, DN + DR, s);
    defer free(q_rot);
    const w_rows = try lml.kvRows(held, geo, s);
    defer free(w_rows);
    var got = mlx.mlx_array_new();
    defer free(got);
    const took = try fused.attend(&got, s, q_nope, q_rot, lat, rot, w_rows, scale);
    const ran_attn = t3.ling.mla_latent.lastAttn();
    if (!nax_on) {
        const caught = took and isBase(ran_attn) and ran_attn != .stage;
        std.debug.print("TURBOEXL3_LING_MLA_LATENT_STAGE_RED stage and NAX off: took={} launch 2 ran {s}, the base text: {s}; the stage arm's text check {s}\n", .{ took, @tagName(ran_attn), if (isBase(ran_attn)) "yes" else "no", if (caught) "rejects it PASS" else "accepts it, MISSED" });
        return @intFromBool(!caught);
    }
    const named = took and !isBase(ran_attn);
    std.debug.print("TURBOEXL3_LING_MLA_LATENT_STAGE_RED stage off, NAX on: took={} launch 2 ran {s}; the base-text check {s}\n", .{ took, @tagName(ran_attn), if (named) "says it is not the base text PASS" else "takes it for the base text, MISSED" });
    return @intFromBool(!named);
}
