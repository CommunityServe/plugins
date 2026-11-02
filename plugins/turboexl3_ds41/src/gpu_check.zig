//! sx4's GPU check (`zig build gpu-check -Dgpu-check-mlx=<host tree>/lib/mlx`, an executable over the kit's `mlx` and
//! that tree's staged MLX; run only inside a GPU window): the two in-place kernels `MLX_SERVE_DSV41_SCRATCH_TAIL` serves,
//! compiled on the GPU at the served template arguments (K 6, D 5120, HC 4, bf16 streams, sources and shared rows, 24
//! source slots; rows 16,383 by default, then 4,097) and their words checked against the stock kernels' over synthetic
//! inputs:
//! - tail: TAIL_FUSE's stock kernel's new streams (a fresh output) against its write-into twin's over a second copy of
//!   the streams, in place: every word equal (and the copy's words changed: the twin wrote it, not a temporary); then
//!   `check`'s path (`expandIntoChecked`: the stock, then the twin over a third copy, compared in the one call) with the
//!   stock's words served.
//! - hc: SCRATCH's hc expand into a region (kernels/hc_expand_into.metal) out of place, against the same kernel over a
//!   second copy of the streams in place (its streams the region itself, as the layer after a twin reads them): every word
//!   equal (and the copy's words changed).
//! - hc-fuse (o1a2, `MLX_SERVE_DSV41_HC_FUSE`, src/hc_fuse.zig): the three fused hc kernels compiled on the GPU as the
//!   copied model builds and launches them (HC_TG 1,024 threads, hc_mix's 4-row by 6-mix tile, HC 4, 20 Sinkhorn
//!   iterations, bf16 streams) at 1, 2, 4,097 and 16,383 rows (one row binds `post` and `cpre`, 4 words each, in
//!   MLX's constant space; two rows in the device space), every output checked word for word against the stock chain's
//!   (the copied model's own kernel files): hc_mix_norm (with and without the read-ahead's second norm) against hc_mix and
//!   hc_norm (twice), its split by hc_split against hc_norm's; hc_expand_norm (a fresh new stream) and
//!   hc_expand_norm_into (into a region of its own, and into the stream's own buffer in place) against SCRATCH's expand
//!   into a region then hc_mix and hc_norm over it, the fused stream's split by hc_mix and hc_split against hc_norm's.
//! Lines `GPUCHECK start`, `GPUCHECK tail`, `GPUCHECK tail-check`, `GPUCHECK hc`, `GPUCHECK hc-fuse` (or `... error=` and
//! `GPUCHECK mlx-error` with MLX's message, a kernel that does not compile among them), `GPUCHECK DONE rc=` (0: every
//! check passed).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const tail_fuse = @import("tail_fuse.zig");
const sx1 = @import("scratch_x1.zig");
const scratch = @import("scratch.zig");
const hcf = @import("hc_fuse.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const K = 6;
const D = 5120;
const HC = 4;
const n_sources = tail_fuse.sources;

fn onError(msg: [*:0]const u8, _: ?*anyopaque) callconv(.c) void {
    std.debug.print("GPUCHECK mlx-error {s}\n", .{msg});
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn drop(x: *A) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
    x.* = none;
}

fn evalOne(x: A) !void {
    try mlx.check(mlx.mlx_array_eval(x));
}

/// A new evaluated array of normal draws (the same seed: the same words, in a buffer of its own).
fn normal(s: mlx.mlx_stream, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, scale: f32) !A {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dt, 0, scale, key, s));
    try evalOne(out);
    return out;
}

fn words(s: mlx.mlx_stream, a: A, b: A) !scratch.Tally {
    var t: scratch.Tally = .{};
    try scratch.compare(s, a, b, &t);
    return t;
}

fn tailCheck(s: mlx.mlx_stream, n: usize) !bool {
    const rows = ci(n);
    const per: usize = (n * K + n_sources - 1) / n_sources;
    var srcs: [n_sources]A = @splat(none);
    defer for (&srcs) |*x| drop(x);
    for (&srcs, 0..) |*x, i| x.* = try normal(s, &.{ ci(per), D }, .bfloat16, 100 + i, 1.0);
    // Assignment j = row * K + slot reads source j % 24 at its row j / 24.
    const a = std.heap.page_allocator;
    const host = try a.alloc(i32, n * K * 2);
    defer a.free(host);
    for (0..n * K) |j| {
        host[2 * j] = @intCast(j % n_sources);
        host[2 * j + 1] = @intCast(j / n_sources);
    }
    var loc = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ rows, K, 2 }, 3, .int32);
    defer drop(&loc);
    try evalOne(loc);
    var w = try normal(s, &.{ rows, K }, .float32, 200, 0.5);
    defer drop(&w);
    var sh = try normal(s, &.{ rows, D }, .bfloat16, 201, 1.0);
    defer drop(&sh);
    var pc = try normal(s, &.{ rows, HC + HC * HC }, .float32, 202, 0.5);
    defer drop(&pc);
    var res = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res);
    // The stock kernel's new streams, a fresh output.
    var fresh = try tail_fuse.expand(s, &srcs, loc, w, sh, res, pc);
    defer drop(&fresh);
    try evalOne(fresh);
    // The twin over a second copy of the streams, in place.
    var res2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res2);
    const copy0 = try words(s, res2, res);
    var carrier = try tail_fuse.expandInto(s, &srcs, loc, w, sh, res2, pc);
    defer drop(&carrier);
    try evalOne(carrier);
    const t = try words(s, res2, fresh);
    const changed = try words(s, res2, res);
    std.debug.print("GPUCHECK tail rows={d} copy_differ={d} words={d} differ={d} changed={d}\n", .{ n, copy0.differ, t.words, t.differ, changed.differ });
    var ok = copy0.differ == 0 and t.words > 0 and t.differ == 0 and changed.differ > 0;
    drop(&carrier);
    drop(&res2);
    _ = mlx.mlx_clear_cache();
    // `check`'s path over a third copy: the stock kernel's words served, the twin's compared with them in the call.
    var res3 = try normal(s, &.{ rows, HC, D }, .bfloat16, 203, 1.0);
    defer drop(&res3);
    const r = try tail_fuse.expandIntoChecked(s, &srcs, loc, w, sh, res3, pc);
    var served = r.fresh;
    defer drop(&served);
    const t2 = try words(s, served, fresh);
    std.debug.print("GPUCHECK tail-check rows={d} words={d} differ={d} served_words={d} served_differ={d}\n", .{ n, r.words, r.differ, t2.words, t2.differ });
    ok = ok and r.words > 0 and r.differ == 0 and t2.differ == 0;
    return ok;
}

fn hcCheck(s: mlx.mlx_stream, n: usize) !bool {
    const rows = ci(n);
    var y4 = try normal(s, &.{ 1, rows, D }, .bfloat16, 300, 1.0);
    defer drop(&y4);
    var post = try normal(s, &.{ rows, HC }, .float32, 301, 0.5);
    defer drop(&post);
    var comb = try normal(s, &.{ rows, HC, HC }, .float32, 302, 0.5);
    defer drop(&comb);
    // Out of place: the streams expanded into a region of their own.
    var res = try normal(s, &.{ rows, HC, D }, .bfloat16, 303, 1.0);
    defer drop(&res);
    var dst = try normal(s, &.{ rows, HC, D }, .bfloat16, 304, 1.0);
    defer drop(&dst);
    var x4 = mlx.mlx_array_new();
    defer drop(&x4);
    try mlx.check(mlx.mlx_reshape(&x4, res, &[_]c_int{ 1, rows, HC, D }, 4, s));
    var c1 = try sx1.expandInto(s, y4, x4, post, comb, dst, x4);
    defer drop(&c1);
    try evalOne(c1);
    // In place: a second copy of the streams expanded over itself.
    var res2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 303, 1.0);
    defer drop(&res2);
    var x4b = mlx.mlx_array_new();
    defer drop(&x4b);
    try mlx.check(mlx.mlx_reshape(&x4b, res2, &[_]c_int{ 1, rows, HC, D }, 4, s));
    var c2 = try sx1.expandInto(s, y4, x4b, post, comb, res2, x4b);
    defer drop(&c2);
    try evalOne(c2);
    const t = try words(s, res2, dst);
    const changed = try words(s, res2, res);
    std.debug.print("GPUCHECK hc rows={d} words={d} differ={d} changed={d}\n", .{ n, t.words, t.differ, changed.differ });
    return t.words > 0 and t.differ == 0 and changed.differ > 0;
}

// ── hc-fuse (MLX_SERVE_DSV41_HC_FUSE) ───────────────────────────────────────────────────────────────────────────────
/// The copied model's stock hc kernels (deepseek_v41.zig's HC_MIX_SOURCE, HC_NORM_SOURCE and HC_SPLIT_SOURCE: the same
/// files) and its served launch values: HC_TG threads (the load's probe halves none on the M5 Max), hc_mix's tile of
/// `hc_mix_rows` rows by the first of 6, 4, 3, 2, 1 mixes dividing 24, the pack's 20 Sinkhorn iterations.
const hc_mix_source = @embedFile("mlx_serve/plugins/deepseek_v41/kernels/hc_mix.metal");
const hc_norm_source = @embedFile("mlx_serve/plugins/deepseek_v41/kernels/hc_norm.metal");
const hc_split_source = @embedFile("mlx_serve/plugins/deepseek_v41/kernels/hc_split.metal");
const HC_TG = 1024;
const MIX = (2 + HC) * HC;
const MIX_ROWS = 4;
const MIX_COLS = 6;
const ITERS = 20;

const Kern = mlx.mlx_fast_metal_kernel;

/// The copied model's `buildKernel`.
fn buildKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !Kern {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// The six kernels with the copied model's names, inputs, outputs, sources and headers.
const HcKernels = struct { mix: Kern, norm: Kern, split: Kern, mix_norm: Kern, exp_norm: Kern, exp_norm_into: Kern };

fn hcKernels() !HcKernels {
    return .{
        .mix = try buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, hc_mix_source, ""),
        .norm = try buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, hc_norm_source, ""),
        .split = try buildKernel("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, hc_split_source, ""),
        .mix_norm = try buildKernel("dsv41_hc_mix_norm", &.{ "x", "fn", "scale", "base", "eps", "cpre", "w", "w2" }, &.{ "z", "h_out", "h2" }, hcf.mix_norm_source, ""),
        .exp_norm = try buildKernel("dsv41_hc_expand_norm", &.{ "y", "res", "post", "comb", "cpre", "w", "eps" }, &.{ "h_out", "x1" }, hcf.expand_norm_source, hcf.expand_norm_fresh_header),
        .exp_norm_into = try buildKernel("dsv41_hc_expand_norm_into", &.{ "y", "res", "post", "comb", "cpre", "w", "eps", "dst", "after" }, &.{ "h_out", "carrier" }, hcf.expand_norm_source, hcf.expand_norm_into_header),
    };
}

const OutArg = struct { shape: []const c_int, dt: mlx.mlx_dtype };
const TArg = struct { name: [*:0]const u8, v: c_int };

/// One launch of `k` as the copied model's: outputs `outs`, a grid of `gx` by `gy` threads in threadgroups of `tg`, the
/// int template arguments in the model's order, then `T` when given; the outputs into `got` (new handles).
fn launch(s: mlx.mlx_stream, k: Kern, ins: []const A, outs: []const OutArg, gx: usize, gy: usize, tg: usize, ints: []const TArg, t: ?mlx.mlx_dtype, got: []A) !void {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(gx), ci(gy), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    for (ints) |p| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, p.name, p.v));
    if (t) |dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (got, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
}

/// Arrays let go together.
const Bag = struct {
    xs: [48]A = @splat(none),
    n: usize = 0,
    fn put(b: *Bag, x: A) A {
        b.xs[b.n] = x;
        b.n += 1;
        return x;
    }
    fn deinit(b: *Bag) void {
        for (b.xs[0..b.n]) |*x| drop(x);
        b.n = 0;
    }
};

/// A wide layer's hc inputs, synthetic: the streams, the mix weights, the norms' weights, the previous split's `pre`, the
/// block output and this stage's split (the FFN side's expand).
const HcIn = struct { x: A, fn_w: A, scale: A, base: A, eps: A, cpre: A, w: A, w2: A, y: A, post: A, comb: A };

/// hcMixWide: the mixes [L, 24] f32.
fn stockMix(s: mlx.mlx_stream, ks: *const HcKernels, b: *Bag, x: A, in: *const HcIn, L: usize) !A {
    const zs = [_]c_int{ ci(L), MIX };
    var o: [1]A = undefined;
    try launch(s, ks.mix, &.{ x, in.fn_w, in.scale, in.base, in.eps }, &.{.{ .shape = &zs, .dt = .float32 }}, MIX / MIX_COLS * HC_TG, (L + MIX_ROWS - 1) / MIX_ROWS, HC_TG, &.{ .{ .name = "TG", .v = HC_TG }, .{ .name = "R", .v = MIX_ROWS }, .{ .name = "M", .v = MIX_COLS } }, null, &o);
    return b.put(o[0]);
}

/// hcNorm: pre, post, comb (f32) and the normed rows [L, D] (bf16).
fn stockNorm(s: mlx.mlx_stream, ks: *const HcKernels, b: *Bag, z: A, x: A, in: *const HcIn, w: A, L: usize) ![4]A {
    const ps = [_]c_int{ ci(L), HC };
    const cs = [_]c_int{ ci(L), HC, HC };
    const hs = [_]c_int{ ci(L), D };
    var o: [4]A = undefined;
    try launch(s, ks.norm, &.{ z, x, in.cpre, w, in.eps }, &.{ .{ .shape = &ps, .dt = .float32 }, .{ .shape = &ps, .dt = .float32 }, .{ .shape = &cs, .dt = .float32 }, .{ .shape = &hs, .dt = .bfloat16 } }, HC_TG, L, HC_TG, &.{ .{ .name = "TG", .v = HC_TG }, .{ .name = "HC", .v = HC }, .{ .name = "ITERS", .v = ITERS } }, .bfloat16, &o);
    for (o) |a| _ = b.put(a);
    return o;
}

/// hcSplitWide: pre, post, comb of the mixes `z` (hc_split, one simdgroup a row).
fn split(s: mlx.mlx_stream, ks: *const HcKernels, b: *Bag, z: A, in: *const HcIn, L: usize) ![3]A {
    const ps = [_]c_int{ ci(L), HC };
    const cs = [_]c_int{ ci(L), HC, HC };
    var o: [3]A = undefined;
    try launch(s, ks.split, &.{ z, in.eps }, &.{ .{ .shape = &ps, .dt = .float32 }, .{ .shape = &ps, .dt = .float32 }, .{ .shape = &cs, .dt = .float32 } }, 32, L, 32, &.{ .{ .name = "HC", .v = HC }, .{ .name = "ITERS", .v = ITERS } }, null, &o);
    for (o) |a| _ = b.put(a);
    return o;
}

/// hcMixNormFused: z, the normed rows and (h2) the rows normed again by `w2`, one launch.
fn mixNorm(s: mlx.mlx_stream, ks: *const HcKernels, b: *Bag, in: *const HcIn, h2: bool, L: usize) ![3]A {
    const zs = [_]c_int{ ci(L), MIX };
    const hs = [_]c_int{ ci(L), D };
    const one = [_]c_int{1};
    const h2s: []const c_int = if (h2) &hs else &one;
    var o: [3]A = undefined;
    try launch(s, ks.mix_norm, &.{ in.x, in.fn_w, in.scale, in.base, in.eps, in.cpre, in.w, if (h2) in.w2 else in.w }, &.{ .{ .shape = &zs, .dt = .float32 }, .{ .shape = &hs, .dt = .bfloat16 }, .{ .shape = h2s, .dt = .bfloat16 } }, (MIX / MIX_COLS + 1) * HC_TG, (L + MIX_ROWS - 1) / MIX_ROWS, HC_TG, &.{ .{ .name = "TG", .v = HC_TG }, .{ .name = "R", .v = MIX_ROWS }, .{ .name = "M", .v = MIX_COLS }, .{ .name = "HC", .v = HC }, .{ .name = "H2", .v = @intFromBool(h2) } }, .bfloat16, &o);
    for (o) |a| _ = b.put(a);
    return o;
}

/// hcExpandNormLaunch: the normed rows and the new stream (fresh), or the normed rows and the carrier (into `dst`).
fn expNorm(s: mlx.mlx_stream, ks: *const HcKernels, b: *Bag, in: *const HcIn, res: A, dst: ?A, after: A, L: usize) ![2]A {
    const hs = [_]c_int{ ci(L), D };
    const xs = [_]c_int{ ci(L), HC, D };
    const one = [_]c_int{1};
    var o: [2]A = undefined;
    const ints = [_]TArg{ .{ .name = "TG", .v = HC_TG }, .{ .name = "HC", .v = HC } };
    if (dst) |d| {
        try launch(s, ks.exp_norm_into, &.{ in.y, res, in.post, in.comb, in.cpre, in.w, in.eps, d, after }, &.{ .{ .shape = &hs, .dt = .bfloat16 }, .{ .shape = &one, .dt = .uint32 } }, HC_TG, L, HC_TG, &ints, .bfloat16, &o);
    } else {
        try launch(s, ks.exp_norm, &.{ in.y, res, in.post, in.comb, in.cpre, in.w, in.eps }, &.{ .{ .shape = &hs, .dt = .bfloat16 }, .{ .shape = &xs, .dt = .bfloat16 } }, HC_TG, L, HC_TG, &ints, .bfloat16, &o);
    }
    for (o) |a| _ = b.put(a);
    return o;
}

fn tally(s: mlx.mlx_stream, pairs: []const [2]A) !scratch.Tally {
    var t: scratch.Tally = .{};
    for (pairs) |p| try scratch.compare(s, p[0], p[1], &t);
    return t;
}

/// The attention side: hc_mix_norm (h2 1 and 0) and its split by hc_split against hc_mix, hc_norm and hc_norm by w2.
fn hcFuseAttn(s: mlx.mlx_stream, ks: *const HcKernels, in: *const HcIn, L: usize) !bool {
    var b: Bag = .{};
    defer b.deinit();
    const z = try stockMix(s, ks, &b, in.x, in, L);
    const n = try stockNorm(s, ks, &b, z, in.x, in, in.w, L);
    const n2 = try stockNorm(s, ks, &b, z, in.x, in, in.w2, L);
    var ok = true;
    for ([_]bool{ true, false }) |h2| {
        const f = try mixNorm(s, ks, &b, in, h2, L);
        const sp = try split(s, ks, &b, f[0], in, L);
        var t = try tally(s, &.{ .{ f[0], z }, .{ sp[0], n[0] }, .{ sp[1], n[1] }, .{ sp[2], n[2] }, .{ f[1], n[3] } });
        if (h2) try scratch.compare(s, f[2], n2[3], &t);
        std.debug.print("GPUCHECK hc-fuse kernel=mix_norm h2={d} rows={d} words={d} differ={d}\n", .{ @intFromBool(h2), L, t.words, t.differ });
        ok = ok and t.words > 0 and t.differ == 0;
    }
    return ok;
}

/// The FFN side: hc_expand_norm fresh, hc_expand_norm_into out of place and in place (the stream's own buffer, as after
/// SCRATCH_TAIL's in-place tail), each new stream's split by hc_mix and hc_split, against SCRATCH's expand into a region
/// (the stock served path) and hc_mix and hc_norm over it.
fn hcFuseFfn(s: mlx.mlx_stream, ks: *const HcKernels, in: *const HcIn, L: usize) !bool {
    const rows = ci(L);
    var b: Bag = .{};
    defer b.deinit();
    var y4 = mlx.mlx_array_new();
    defer drop(&y4);
    try mlx.check(mlx.mlx_reshape(&y4, in.y, &[_]c_int{ 1, rows, D }, 3, s));
    var x4 = mlx.mlx_array_new();
    defer drop(&x4);
    try mlx.check(mlx.mlx_reshape(&x4, in.x, &[_]c_int{ 1, rows, HC, D }, 4, s));
    var dst = try normal(s, &.{ rows, HC, D }, .bfloat16, 410, 1.0);
    defer drop(&dst);
    var c1 = try sx1.expandInto(s, y4, x4, in.post, in.comb, dst, x4);
    defer drop(&c1);
    try evalOne(c1);
    const z1 = try stockMix(s, ks, &b, dst, in, L);
    const n1 = try stockNorm(s, ks, &b, z1, dst, in, in.w, L);
    var ok = true;
    // Fresh.
    {
        var bf: Bag = .{};
        defer bf.deinit();
        const f = try expNorm(s, ks, &bf, in, in.x, null, none, L);
        const sp = try split(s, ks, &bf, try stockMix(s, ks, &bf, f[1], in, L), in, L);
        const t = try tally(s, &.{ .{ f[1], dst }, .{ f[0], n1[3] }, .{ sp[0], n1[0] }, .{ sp[1], n1[1] }, .{ sp[2], n1[2] } });
        std.debug.print("GPUCHECK hc-fuse kernel=expand_norm rows={d} words={d} differ={d}\n", .{ L, t.words, t.differ });
        ok = ok and t.words > 0 and t.differ == 0;
    }
    _ = mlx.mlx_clear_cache();
    // Into a region of its own (other words in it first).
    {
        var bf: Bag = .{};
        defer bf.deinit();
        var d2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 411, 1.0);
        defer drop(&d2);
        const f = try expNorm(s, ks, &bf, in, in.x, d2, in.x, L);
        try evalOne(f[1]);
        const sp = try split(s, ks, &bf, try stockMix(s, ks, &bf, d2, in, L), in, L);
        const t = try tally(s, &.{ .{ d2, dst }, .{ f[0], n1[3] }, .{ sp[0], n1[0] }, .{ sp[1], n1[1] }, .{ sp[2], n1[2] } });
        std.debug.print("GPUCHECK hc-fuse kernel=expand_norm_into place=out rows={d} words={d} differ={d}\n", .{ L, t.words, t.differ });
        ok = ok and t.words > 0 and t.differ == 0;
    }
    _ = mlx.mlx_clear_cache();
    // Into the stream's own buffer: a second copy of the streams, expanded over itself.
    {
        var bf: Bag = .{};
        defer bf.deinit();
        var r2 = try normal(s, &.{ rows, HC, D }, .bfloat16, 400, 1.0);
        defer drop(&r2);
        const copy0 = try words(s, r2, in.x);
        const f = try expNorm(s, ks, &bf, in, r2, r2, r2, L);
        try evalOne(f[1]);
        const sp = try split(s, ks, &bf, try stockMix(s, ks, &bf, r2, in, L), in, L);
        const t = try tally(s, &.{ .{ r2, dst }, .{ f[0], n1[3] }, .{ sp[0], n1[0] }, .{ sp[1], n1[1] }, .{ sp[2], n1[2] } });
        const changed = try words(s, r2, in.x);
        std.debug.print("GPUCHECK hc-fuse kernel=expand_norm_into place=in rows={d} copy_differ={d} words={d} differ={d} changed={d}\n", .{ L, copy0.differ, t.words, t.differ, changed.differ });
        ok = ok and copy0.differ == 0 and t.words > 0 and t.differ == 0 and changed.differ > 0;
    }
    return ok;
}

fn hcFuseCheck(s: mlx.mlx_stream, ks: *const HcKernels, L: usize) !bool {
    const rows = ci(L);
    var in: HcIn = undefined;
    const all = [_]*A{ &in.x, &in.fn_w, &in.scale, &in.base, &in.eps, &in.cpre, &in.w, &in.w2, &in.y, &in.post, &in.comb };
    for (all) |p| p.* = none;
    defer for (all) |p| drop(p);
    in.x = try normal(s, &.{ rows, HC, D }, .bfloat16, 400, 1.0);
    in.fn_w = try normal(s, &.{ MIX, HC * D }, .float32, 401, 0.02);
    in.scale = try normal(s, &.{MIX}, .float32, 402, 0.5);
    in.base = try normal(s, &.{MIX}, .float32, 403, 0.5);
    in.eps = mlx.mlx_array_new_data(&[_]f32{ 1e-6, 1e-6 }, &[_]c_int{2}, 1, .float32);
    try evalOne(in.eps);
    in.cpre = try normal(s, &.{ rows, HC }, .float32, 404, 0.5);
    in.w = try normal(s, &.{D}, .bfloat16, 405, 1.0);
    in.w2 = try normal(s, &.{D}, .bfloat16, 406, 1.0);
    in.y = try normal(s, &.{ rows, D }, .bfloat16, 407, 1.0);
    in.post = try normal(s, &.{ rows, HC }, .float32, 408, 0.5);
    in.comb = try normal(s, &.{ rows, HC, HC }, .float32, 409, 0.5);
    const ok_attn = hcFuseAttn(s, ks, &in, L) catch |e| blk: {
        std.debug.print("GPUCHECK hc-fuse side=attn rows={d} error={s}\n", .{ L, @errorName(e) });
        break :blk false;
    };
    _ = mlx.mlx_clear_cache();
    const ok_ffn = hcFuseFfn(s, ks, &in, L) catch |e| blk: {
        std.debug.print("GPUCHECK hc-fuse side=ffn rows={d} error={s}\n", .{ L, @errorName(e) });
        break :blk false;
    };
    _ = mlx.mlx_clear_cache();
    return ok_attn and ok_ffn;
}

pub fn main() u8 {
    mlx.mlx_set_error_handler(&onError, null, null);
    const s = mlx.mlx_default_gpu_stream_new();
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, 1 << 30);
    const rows_list = [_]usize{ 16383, 4097 };
    std.debug.print("GPUCHECK start sources={d} buffers={d} of {d} K={d} D={d} HC={d} T=bfloat16 S=bfloat16 hc_tg={d} mix={d}x{d} iters={d}\n", .{ n_sources, tail_fuse.kernel_buffers, tail_fuse.metal_buffers, K, D, HC, HC_TG, MIX_ROWS, MIX_COLS, ITERS });
    var rc: u8 = 0;
    for (rows_list) |n| {
        const ok_tail = tailCheck(s, n) catch |e| blk: {
            std.debug.print("GPUCHECK tail rows={d} error={s}\n", .{ n, @errorName(e) });
            break :blk false;
        };
        _ = mlx.mlx_clear_cache();
        const ok_hc = hcCheck(s, n) catch |e| blk: {
            std.debug.print("GPUCHECK hc rows={d} error={s}\n", .{ n, @errorName(e) });
            break :blk false;
        };
        _ = mlx.mlx_clear_cache();
        if (!ok_tail or !ok_hc) rc = 1;
    }
    // hc-fuse: one row (`post` and `cpre` bound in MLX's constant space), two rows, then the wide layers' sizes.
    if (hcKernels()) |ks| {
        for ([_]usize{ 1, 2, 4097, 16383 }) |n| {
            const ok = hcFuseCheck(s, &ks, n) catch |e| blk: {
                std.debug.print("GPUCHECK hc-fuse rows={d} error={s}\n", .{ n, @errorName(e) });
                break :blk false;
            };
            if (!ok) rc = 1;
        }
    } else |e| {
        std.debug.print("GPUCHECK hc-fuse error={s}\n", .{@errorName(e)});
        rc = 1;
    }
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("GPUCHECK DONE rc={d} peak_mb={d}\n", .{ rc, peak >> 20 });
    return rc;
}
