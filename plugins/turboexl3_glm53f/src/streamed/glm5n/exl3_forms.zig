//! GLM-5.3-Flash's EXL3 K=3 decode forms: the decode chain of the seam (`exl3_kernels.zig` gateUp / down: in_rin, the
//! gate and up GEMVs, gu_epi, din_rin, the down GEMV, dpost; DeepSeek-V4.1's banked mul1h texts at GLM's shape) in other
//! launch geometries, each computing the seam's words (the bench's device parity checks every output word). A form is a
//! construction-time route: `install` builds the chosen form's launches for every wave width once; `gateUp` / `down`
//! keep the seam's signatures and run them with no per-call choice.
//!
//! Forms (texts in kernels/exl3_forms/):
//!   gate / up (`GuForm`): `seam` (two GEMV launches + gu_epi), `guone` (DeepSeek's gu_one: both GEMVs in one launch).
//!   down (`DnForm`): `seam`, `pair` (DeepSeek's down_pair).
//! Every form keeps the seam's arithmetic and order per output word (fma chains per lane over the k-tiles, the column
//! sums, the 128-point butterflies, the scales, the clamped SwiGLU, one rounding on dpost's store); only where values
//! live and which thread computes them change. A form that fails the device parity is not installed.
//! Wave widths: P = 1 .. `max_pairs` (64 = 8 tokens x top-8, the NVFP4 texts' widest wave; the decode lane forms at most
//! 4 rows x 8 = 32 pairs, and the scheduler's route at most `policy.max_route_ids` = 48).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const mtl = @import("../sdk_ext/metal_launch.zig");
const kset = @import("../sdk_ext/kernel_set.zig");
const xk = @import("exl3_kernels.zig");

pub const G = xk.G;
pub const T = xk.T;
const BankArrays = xk.BankArrays;
const n_banks = xk.n_banks;
const hidden = xk.hidden;
const inter = xk.inter;

pub const max_pairs: u32 = 64;

comptime {
    if (max_pairs < @import("../sdk_ext.zig").expert.policy.max_route_ids) @compileError("glm5_next exl3 forms: the widest wave is narrower than the scheduler's route");
}

pub const GuForm = enum { seam, guone };
pub const DnForm = enum { seam, pair };
pub const Forms = struct {
    gu: GuForm = .seam,
    dn: DnForm = .seam,
};

// ── The texts ──

pub const Kernel = enum { g5k3_guone, g5k3_pair };
const n_kernels = std.meta.fieldNames(Kernel).len;
const dir = "kernels/exl3_forms/";
pub const texts: [n_kernels][:0]const u8 = blk: {
    var out: [n_kernels][:0]const u8 = undefined;
    for (std.meta.fieldNames(Kernel), 0..) |n, i| out[i] = @embedFile(dir ++ n ++ ".metal");
    break :blk out;
};
/// sha256 of each text.
pub const text_sha256 = [n_kernels]*const [64]u8{
    "85dcbd286beb00d7ec1f80b298967859090c0f8aedf29824380a5c5027df3374",
    "0732d4840889255ad9499b30e1431fc2ff78ec19d72ebc89f5a1d9df31440110",
};

const Sig = struct { ins: []const [*:0]const u8, outs: []const [*:0]const u8 };

// ── Launches (pure: the tests read them) ──

pub const Launch = struct {
    grid: [3]u32,
    tg: [3]u32,
    ints: [4]struct { name: [*:0]const u8, v: u32 } = undefined,
    n_int: u8 = 0,
    outs: [2]struct { shape: [2]c_int, dtype: mlx.mlx_dtype } = undefined,
    n_out: u8 = 0,

    fn out(l: Launch, rows: u32, cols: u32, dtype: mlx.mlx_dtype) Launch {
        var o = l;
        o.outs[o.n_out] = .{ .shape = .{ @intCast(rows), @intCast(cols) }, .dtype = dtype };
        o.n_out += 1;
        return o;
    }
};

fn apply(g: *G, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const T, outs: []T) !void {
    const in = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k, in, cfg, g.s));
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = try g.adopt(x);
    }
}

// ── Install ──

const Cfg = mlx.mlx_fast_metal_kernel_config;
const none: Cfg = .{ .ctx = null };
/// One wave width's prepared launches.
const WaveCfg = struct {
    in_rin: Cfg = none,
    g1: Cfg = none,
    epi: Cfg = none,
    din: Cfg = none,
    d1: Cfg = none,
    post: Cfg = none,

    fn free(w: WaveCfg) void {
        for ([_]Cfg{ w.in_rin, w.g1, w.epi, w.din, w.d1, w.post }) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
    }
};

pub const Installed = struct {
    forms: Forms,
    out_dtype: mlx.mlx_dtype,
    seam: *const xk.Kernels,
    k: [n_kernels]mlx.mlx_fast_metal_kernel = @splat(.{ .ctx = null }),
    w: [max_pairs]WaveCfg = @splat(.{}),
    tok: [max_pairs]T = @splat(.{ .ctx = null }),

    pub fn deinit(self: *Installed) void {
        for (self.k) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (self.w) |w| w.free();
        for (self.tok) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
    }

};

fn widthOf(g: *G, x: T) !u32 {
    const p: u32 = @intCast(g.shapeOf(x).d[0]);
    if (p < 1 or p > max_pairs) return error.Glm5NextExl3WaveWidth;
    return p;
}

/// The seam's `gateUp` in the installed form: x bf16 [P, hidden], ids u32 [P] packed `bank << 24 | slot row` -> the
/// clamped SwiGLU f32 [P, inter].
pub fn gateUp(f: *const Installed, g: *G, x: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, x);
    const w = &f.w[p - 1];
    var xs: [2]T = undefined;
    try apply(g, f.seamK(.in_rin), w.in_rin, &.{ x, f.tok[p - 1], banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
    var z: [2]T = undefined;
    switch (f.forms.gu) {
        .seam => {
            const k = f.seamK(.gemv);
            try apply(g, k, w.g1, &.{ xs[0], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, z[0..1]);
            try apply(g, k, w.g1, &.{ xs[1], ids, banks[0].up.code, banks[1].up.code, banks[2].up.code }, z[1..2]);
        },
        .guone => try apply(g, f.kern(.g5k3_guone), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
    }
    var out: [1]T = undefined;
    try apply(g, f.seamK(.gu_epi), w.epi, &.{ z[0], z[1], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
    return out[0];
}

/// The seam's `down` in the installed form: h f32 [P, inter] (gateUp's), the same ids -> [P, hidden] in the installed
/// output dtype (the seam's bf16).
pub fn down(f: *const Installed, g: *G, h: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, h);
    const w = &f.w[p - 1];
    var out: [1]T = undefined;
    var hd: [1]T = undefined;
    try apply(g, f.seamK(.din_rin), w.din, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
    const k = if (f.forms.dn == .seam) f.seamK(.gemv) else f.kern(.g5k3_pair);
    var zd: [1]T = undefined;
    try apply(g, k, w.d1, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, &zd);
    try apply(g, f.seamK(.dpost), w.post, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
    return out[0];
}

// ── Tests (host only) ──

const testing = std.testing;
