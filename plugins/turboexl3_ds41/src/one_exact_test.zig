//! src/one_exact.zig against turboexl3's own MLX-free reference (its src/format.zig, a module of its own here: build.zig
//! adds this test to `zig build test`).

const std = @import("std");
const format = @import("turboexl3_format");
const exact = @import("one_exact.zig");
const E = exact.Exact(format);
const testing = std.testing;

test "routed one exact: the f64 Hadamard is the Sylvester matrix's product, natural order, over sqrt(128)" {
    var prng = std.Random.DefaultPrng.init(0x0E1);
    const rnd = prng.random();
    var v: [128]f64 = undefined;
    for (&v) |*x| x.* = rnd.floatNorm(f64);
    const x0 = v;
    exact.hadamard(&v);
    for (0..128) |r| {
        var want: f64 = 0;
        for (0..128) |k| want += (if (@popCount(r & k) % 2 == 0) @as(f64, 1) else -1) * x0[k];
        try testing.expectApproxEqAbs(want / @sqrt(128.0), v[r], 1e-12);
    }
    // the map is its own inverse
    exact.hadamard(&v);
    for (v, x0) |got, want| try testing.expectApproxEqAbs(want, got, 1e-12);
}

test "routed one exact: the clamped SwiGLU clamps the gate from above and the up both ways" {
    const silu = struct {
        fn f(x: f64) f64 {
            return x / (1.0 + @exp(-x));
        }
    }.f;
    try testing.expectApproxEqAbs(silu(3) * 2, exact.swigluClamped(3, 2, 10), 1e-15);
    try testing.expectApproxEqAbs(silu(10) * 2, exact.swigluClamped(25, 2, 10), 1e-15);
    // a gate far below is not clamped (from above only)
    try testing.expectApproxEqAbs(silu(-30) * 2, exact.swigluClamped(-30, 2, 10), 1e-15);
    try testing.expectApproxEqAbs(silu(3) * 10, exact.swigluClamped(3, 40, 10), 1e-15);
    try testing.expectApproxEqAbs(silu(3) * -10, exact.swigluClamped(3, -40, 10), 1e-15);
}

/// A random projection in -> out at `rate`, drawn as turboexl3's `ref.synthBank` draws one (every halfword a valid
/// codeword; suh in [0.5, 1.5]; svh in [0.5, 1.5] / sqrt(in)), and an input of f16 values (so `format.project`'s input
/// rounding is exact and its other two roundings are the whole difference).
const Case = struct {
    trellis: []u16,
    suh: []u16,
    svh: []u16,
    x: []f32,

    fn draw(a: std.mem.Allocator, rnd: std.Random, in: usize, out: usize, rate: format.Rate) !Case {
        const c: Case = .{
            .trellis = try a.alloc(u16, (in / 16) * (out / 16) * rate.halfwords()),
            .suh = try a.alloc(u16, in),
            .svh = try a.alloc(u16, out),
            .x = try a.alloc(f32, in),
        };
        for (c.trellis) |*w| w.* = rnd.int(u16);
        for (c.suh) |*s| s.* = format.f32ToF16Bits(0.5 + rnd.float(f32));
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
        for (c.svh) |*s| s.* = format.f32ToF16Bits((0.5 + rnd.float(f32)) * scale);
        for (c.x) |*v| v.* = format.f16BitsToF32(format.f32ToF16Bits(rnd.floatNorm(f32)));
        return c;
    }

    fn free(c: Case, a: std.mem.Allocator) void {
        a.free(c.trellis);
        a.free(c.suh);
        a.free(c.svh);
        a.free(c.x);
    }
};

test "routed one exact: a projection in f64 agrees with turboexl3's reference projection within its f16 roundings, K3 and K2, two users at once" {
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0E2);
    const rnd = prng.random();
    const e = E.init();
    for ([_]struct { in: usize, out: usize, k: u32 }{ .{ .in = 256, .out = 384, .k = 3 }, .{ .in = 384, .out = 256, .k = 2 } }) |sh| {
        const rate = format.Rate.fromK(sh.k);
        const c = try Case.draw(a, rnd, sh.in, sh.out, rate);
        defer c.free(a);
        // the reference: format.project rounds the transformed input, the inner product and the output to f16
        const transformed = try a.alloc(f32, sh.in);
        defer a.free(transformed);
        const inner = try a.alloc(f32, sh.out);
        defer a.free(inner);
        const want = try a.alloc(f32, sh.out);
        defer a.free(want);
        format.project(c.x, c.trellis, c.suh, c.svh, sh.in, sh.out, rate, .{ .codebook = .mul1 }, transformed, inner, want);
        // two users of the one decode: the case's input and its negation (the map is linear: the outputs negate)
        const x64 = try a.alloc(f64, 2 * sh.in);
        defer a.free(x64);
        for (c.x, 0..) |v, i| {
            x64[i] = v;
            x64[sh.in + i] = -@as(f64, v);
        }
        const t64 = try a.alloc(f64, 2 * sh.in);
        defer a.free(t64);
        const y64 = try a.alloc(f64, 2 * sh.out);
        defer a.free(y64);
        const xs = [_][]const f64{ x64[0..sh.in], x64[sh.in..] };
        const ts = [_][]f64{ t64[0..sh.in], t64[sh.in..] };
        const ys = [_][]f64{ y64[0..sh.out], y64[sh.out..] };
        e.project(c.trellis, c.suh, c.svh, sh.in, sh.out, rate, &xs, &ts, &ys);
        var ss: f64 = 0;
        for (want) |v| ss += @as(f64, v) * v;
        const rms = @sqrt(ss / @as(f64, @floatFromInt(sh.out)));
        try testing.expect(rms > 1e-2);
        // within 8 f16 half-ulps of the element and the rms: a wrong tile order, permutation, Hadamard or scale is O(1)
        const bar = 8.0 / 2048.0;
        for (want, ys[0], ys[1]) |w, got, neg| {
            try testing.expect(@abs(got - @as(f64, w)) <= bar * (@abs(@as(f64, w)) + rms));
            try testing.expectEqual(-got, neg);
        }
    }
}
