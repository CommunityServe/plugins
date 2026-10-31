//! `MLX_SERVE_DSV41_ROUTED_ONE=check`'s exact reference, MLX-free: an EXL3 projection and the clamped SwiGLU in f64 with
//! no rounding anywhere, over turboexl3's format (`Exact(format)`: its tile decode, rates and codebooks). A projection
//! is the linear map every kernel approximates: suh times the input, H128 over each 128-block, the decoded inner weight
//! (f16 values, exact in f64), H128 over each output block, times svh. turboexl3's own reference (`format.project`)
//! rounds to f16 at three points of the same map; the test (src/one_exact_test.zig) holds this one to it.

const std = @import("std");

pub const had_dim = 128;

/// The Sylvester Hadamard in natural order, scaled 1/sqrt(128), over `v` (128 values) in place: H[r][k] = (-1)^popcount(r & k)
/// / sqrt(128), as butterflies.
pub fn hadamard(v: []f64) void {
    std.debug.assert(v.len == had_dim);
    var span: usize = 1;
    while (span < had_dim) : (span *= 2) {
        var i: usize = 0;
        while (i < had_dim) : (i += 2 * span) {
            for (i..i + span) |j| {
                const p = v[j];
                const q = v[j + span];
                v[j] = p + q;
                v[j + span] = p - q;
            }
        }
    }
    const scale = 1.0 / @sqrt(@as(f64, had_dim));
    for (v) |*x| x.* *= scale;
}

/// DeepSeek-V4's clamped SwiGLU (`limit` > 0): the gate clamped from above, the up to [-limit, limit], then silu(gate) * up.
pub fn swigluClamped(gate: f64, up: f64, limit: f64) f64 {
    const gc = if (gate > limit) limit else gate;
    const uc = if (up > limit) limit else if (up < -limit) -limit else up;
    return gc / (1.0 + @exp(-gc)) * uc;
}

pub fn f16Value(bits: u16) f64 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// The projections over turboexl3's `format` (src/format.zig: `Rate`, `unpackTile`, `tensorCorePerm`, `decodeCodeword`).
pub fn Exact(comptime format: type) type {
    return struct {
        const Self = @This();
        /// Code position p of a tile holds the row-major tile element perm[p] (`format.tensorCorePerm`).
        perm: [256]usize,

        pub fn init() Self {
            var s: Self = undefined;
            format.tensorCorePerm(&s.perm);
            return s;
        }

        /// Each `xs[u]` [in] through one EXL3 projection into `ys[u]` [out], in f64: `trellis` its [in/16, out/16, n]
        /// halfwords (mul1, window 16), `suh` [in] and `svh` [out] f16 bits. `ts[u]` [in] is scratch (the transformed
        /// input). The users share each tile's decode.
        pub fn project(self: *const Self, trellis: []const u16, suh: []const u16, svh: []const u16, in: usize, out: usize, rate: format.Rate, xs: []const []const f64, ts: []const []f64, ys: []const []f64) void {
            const n = rate.halfwords();
            const in_t = in / 16;
            const out_t = out / 16;
            std.debug.assert(trellis.len == in_t * out_t * n and suh.len == in and svh.len == out);
            for (xs, ts) |x, t| {
                for (t[0..in], x[0..in], suh) |*d, v, s| d.* = v * f16Value(s);
                var b: usize = 0;
                while (b < in) : (b += had_dim) hadamard(t[b..][0..had_dim]);
            }
            for (ys) |y| @memset(y[0..out], 0);
            var cw: [256]u16 = undefined;
            var w: [256]f64 = undefined;
            for (0..in_t) |tk| for (0..out_t) |tn| {
                const off = (tk * out_t + tn) * n;
                format.unpackTile(trellis[off..][0..n], rate, &cw);
                for (cw, 0..) |c, p| w[self.perm[p]] = f16Value(format.decodeCodeword(c, .mul1));
                for (ts, ys) |t, y| {
                    for (0..16) |r| {
                        const xv = t[tk * 16 + r];
                        const wr = w[r * 16 ..][0..16];
                        const yr = y[tn * 16 ..][0..16];
                        for (yr, wr) |*o, wv| o.* += xv * wv;
                    }
                }
            };
            for (ys) |y| {
                var b: usize = 0;
                while (b < out) : (b += had_dim) hadamard(y[b..][0..had_dim]);
                for (y[0..out], svh) |*o, s| o.* *= f16Value(s);
            }
        }
    };
}
