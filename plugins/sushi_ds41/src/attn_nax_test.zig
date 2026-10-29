//! The Metal test of `attn_nax.zig`: `attnFusedNax` against the slice's `attnFused` at the release's attention shape.
//! The root of `zig build gpu-test -Dmlx-lib=<dir>` only (it links that MLX and sets SUSHI_DS41_GPU_TEST); skipped
//! without an M5.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41_scope.zig");
const kernels = @import("sushi/dsv41_kernels.zig");
const nax = @import("attn_nax.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const ci = sc_mod.ci;

const Case = struct { L: usize, wp: usize, K: usize, nc: usize, dt: mlx.mlx_dtype = .bfloat16, packed_rows: bool = false, holes: bool = false, all_neg: bool = false, oob: bool = false };

fn randn(sc: *Scope, shape: []const c_int, seed: u64, dt: mlx.mlx_dtype) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    const x = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1.5, key, sc.s), &o);
    return sc.astype(x, dt);
}

fn maxDiff(got: []const f32, want: []const f32) struct { err: f64, peak: f64 } {
    var err: f64 = 0;
    var peak: f64 = 1e-30;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - @as(f64, w));
        err = if (std.math.isNan(e)) std.math.inf(f64) else @max(err, e);
        peak = @max(peak, @abs(w));
    }
    return .{ .err = err, .peak = peak };
}

test "sushi_ds41 gpu: attnFusedNax gives attnFused's heads at the release shapes and the edge rows" {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    if (!nax.naxAvailable()) return error.SkipZigTest;
    const testing = std.testing;
    const H = 64;
    const D = 512;
    const window = 128;
    const scale: f32 = 1.0 / @sqrt(@as(f32, D));
    const cases = [_]Case{
        .{ .L = 256, .wp = 127, .K = 512, .nc = 256 * 512, .packed_rows = true }, // a production span, packed picks
        .{ .L = 256, .wp = 127, .K = 512, .nc = 256 * 512, .packed_rows = true, .dt = .float16 },
        .{ .L = 128, .wp = 127, .K = 512, .nc = 128 * 512, .packed_rows = true, .holes = true }, // the CED tail
        .{ .L = 128, .wp = 40, .K = 512, .nc = 900, .holes = true }, // a floored band
        .{ .L = 256, .wp = 0, .K = 512, .nc = 60, .holes = true }, // a prompt's first rows: bands shorter than the window
        .{ .L = 256, .wp = 127, .K = 0, .nc = 0 }, // no compressed rows yet
        .{ .L = 64, .wp = 5, .K = 512, .nc = 100, .all_neg = true }, // every pick < 0
        .{ .L = 256, .wp = 127, .K = 512, .nc = 300, .oob = true }, // picks at nc and beyond
        .{ .L = 77, .wp = 93, .K = 33, .nc = 500, .holes = true }, // a span offset, K not a tile multiple
    };
    for (cases, 0..) |c, ci_| {
        var sc = Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var prng = std.Random.DefaultPrng.init(0xa77 + ci_);
        const r = prng.random();
        const seed: u64 = 0xab00 + ci_ * 8;
        const q = try randn(&sc, &.{ ci(c.L), H, D }, seed, c.dt);
        const keys = try randn(&sc, &.{ ci(c.wp + c.L), D }, seed + 1, c.dt);
        const sink = try randn(&sc, &.{ H, 1 }, seed + 2, c.dt);
        var comp = sc_mod.none;
        var topk = sc_mod.none;
        if (c.K > 0) {
            comp = try randn(&sc, &.{ ci(c.nc), D }, seed + 3, c.dt);
            const picks = try testing.allocator.alloc(i32, c.L * c.K);
            defer testing.allocator.free(picks);
            for (picks, 0..) |*v, j| {
                const l = j / c.K;
                v.* = if (c.packed_rows) @intCast(j) else @intCast(r.uintLessThan(u32, @intCast(c.nc)));
                if (c.oob and r.uintLessThan(u32, 5) == 0) v.* = @intCast(c.nc + r.uintLessThan(u32, 1 << 20));
                if (c.all_neg or (c.holes and (l % 7 == 3 or r.uintLessThan(u32, 9) == 0))) v.* = -1;
            }
            topk = try sc.ints(picks, &.{ ci(c.L), ci(c.K) });
        }
        // f32 heads within the bound the slice's fused-kernel test holds attnFused to (3e-5 of the peak), and the
        // rounded heads within one ulp of the peak.
        const want = try sc_mod.hostF32(try kernels.attnFused(&sc, sink, q, keys, c.wp, window, comp, topk, scale, .float32));
        const got = try sc_mod.hostF32(try nax.attnFusedNax(&sc, sink, q, keys, c.wp, window, comp, topk, scale, .float32));
        const f = maxDiff(got, want);
        try testing.expect(f.err <= 3e-5 * f.peak);
        const want_r = try sc_mod.hostF32(try sc.astype(try kernels.attnFused(&sc, sink, q, keys, c.wp, window, comp, topk, scale, c.dt), .float32));
        const got_r = try sc_mod.hostF32(try sc.astype(try nax.attnFusedNax(&sc, sink, q, keys, c.wp, window, comp, topk, scale, c.dt), .float32));
        const g = maxDiff(got_r, want_r);
        const mant: i32 = if (c.dt == .float16) 10 else 7;
        try testing.expect(g.err <= std.math.ldexp(@as(f64, 1.0), std.math.frexp(g.peak).exponent - 1 - mant));
    }
}
