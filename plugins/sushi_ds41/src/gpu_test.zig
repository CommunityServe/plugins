//! The Metal tests of `index_nax.zig` (`indexScoresNax` against `indexScoresTile`). The root of
//! `zig build gpu-test -Dmlx-lib=<dir>` only (it links that MLX and sets SUSHI_DS41_GPU_TEST); skipped without an M5.
//! It also runs `vision_key.zig`'s test, which needs MLX but no M5. The attention kernel's own Metal tests are the
//! slice's (`dsv41_model.zig`).
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const inax = @import("index_nax.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const ci = sc_mod.ci;

test {
    _ = @import("vision_key.zig");
}

fn randn(sc: *Scope, shape: []const c_int, seed: u64, dt: mlx.mlx_dtype) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    const x = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1.5, key, sc.s), &o);
    return sc.astype(x, dt);
}

test "sushi_ds41 gpu: indexScoresNax gives indexScoresTile's scores, and a slice scores the whole list's bits" {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    if (!inax.fits(32, 128, .bfloat16)) return error.SkipZigTest;
    const testing = std.testing;
    const IH = 32;
    const IHD = 128;
    const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
    const ICase = struct { L: usize, p0: usize, ratio: usize, nb: usize, dt: mlx.mlx_dtype = .bfloat16 };
    const cases = [_]ICase{
        .{ .L = 256, .p0 = 47104, .ratio = 2, .nb = 23680 }, // a release span at 48k
        .{ .L = 256, .p0 = 47104, .ratio = 2, .nb = 23680, .dt = .float16 },
        .{ .L = 77, .p0 = 931, .ratio = 2, .nb = 504 }, // rows not a multiple of R, keys not of the tile
        .{ .L = 128, .p0 = 0, .ratio = 2, .nb = 64 }, // a prompt's first rows: most keys unseen
        .{ .L = 5, .p0 = 3000, .ratio = 4, .nb = 751 },
    };
    for (cases, 0..) |c, ci_| {
        var sc = Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const seed: u64 = 0x1dd0 + ci_ * 8;
        const q = try sc.astype(try randn(&sc, &.{ ci(c.L), IH, IHD }, seed, c.dt), .float32);
        const keys = try randn(&sc, &.{ ci(c.nb), IHD }, seed + 1, c.dt);
        const w = try randn(&sc, &.{ ci(c.L), IH }, seed + 2, c.dt);
        const want = try sc_mod.hostF32(try kernels.indexScoresTile(&sc, q, keys, w, wscale, c.p0, c.ratio, 0));
        const got = try sc_mod.hostF32(try inax.indexScoresNax(&sc, q, keys, w, wscale, c.p0, c.ratio, 0));
        // The same f32 sums of the same bf16 products in another order: within a few ulps of the row's peak, and the
        // same -inf columns.
        for (0..c.L) |l| {
            var peak: f64 = 1e-30;
            for (want[l * c.nb ..][0..c.nb]) |v| if (!std.math.isInf(v)) {
                peak = @max(peak, @abs(@as(f64, v)));
            };
            for (got[l * c.nb ..][0..c.nb], want[l * c.nb ..][0..c.nb]) |g, v| {
                errdefer std.debug.print("case {d} row {d}: got {e} want {e} peak {e}\n", .{ ci_, l, g, v, peak });
                try testing.expectEqual(std.math.isInf(v), std.math.isInf(g));
                if (!std.math.isInf(v)) try testing.expect(@abs(@as(f64, g) - @as(f64, v)) <= 1e-6 * peak);
            }
        }
        // A slice from a tile boundary scores the bits of those columns of the whole list.
        const t0 = (c.nb / 2) / 32 * 32;
        if (t0 > 0) {
            const part = try sc_mod.hostF32(try inax.indexScoresNax(&sc, q, try sc.slice(keys, 0, t0, c.nb), w, wscale, c.p0, c.ratio, t0));
            const w2 = c.nb - t0;
            for (0..c.L) |l| try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[l * c.nb + t0 ..][0..w2]), std.mem.sliceAsBytes(part[l * w2 ..][0..w2]));
        }
    }
}
