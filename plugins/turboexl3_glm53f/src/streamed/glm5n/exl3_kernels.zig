//! GLM-5.3-Flash's EXL3 K=3 expert kernels (the seam of the `exl3` expert family): DeepSeek-V4.1's mul1h texts at
//! GLM's expert shape (hidden 4096, inter 2048) as compile-time instantiations. The texts in kernels/exl3/ are
//! DeepSeek's with each shape literal a template constant (HID, INTER, IT, OT, KD, ND, SGS); the arithmetic, its order
//! and the bank reads stay DeepSeek's. The
//! DIG-X GEMMs read DeepSeek's `header_dig_mul1h_k3.metal` as it is (no shape literal in it).
//!
//! The seam (what the glm5_next decode stack and prompt path call; a kernel variant keeps these signatures):
//!   - Slot arrays, per projection and slot bank (the stream's base, ext and transient banks): `code` i16
//!     [rows, in / 16, out / 16, 48] (the trellis, 16 x 16 tiles of K = 3, 48 halfwords a tile), `rout` f16 [rows, out]
//!     (exl3 svh), `rin` f16 [rows, in] (exl3 suh). gate / up: in = hidden, out = inter; down: in = inter, out = hidden.
//!     Each array is row-contiguous, one slot per row. The record has no global-scale word (the codebook multiplier is
//!     the bank manifest's, fixed in the texts): no scale-line isolation applies.
//!   - Decode (`gateUp`, `down`): P routed pairs, 1 <= P <= `max_pairs` (one route: the MTP verify's rows 1..4 x top-8
//!     at most); `ids` u32 [P] packed `bank << 24 | slot row`; x bf16 [P, hidden] (each pair's token row, taken);
//!     `gateUp` -> the clamped SwiGLU f32 [P, inter]; `down` (h f32 [P, inter]) -> bf16 [P, hidden] (no route weight; the
//!     f32 word rounded once on store, the expert output type the combines read).
//!     Six launches: in_rin, the gate GEMV, the up GEMV, gu_epi, din_rin, the down GEMV, dpost.
//!   - Prompt (DIG-X, one wave of <= 16 experts of one bank, rows grouped by expert): `take2` -> `gemmGateUp` ->
//!     `onePass` -> `gemmDown` -> `widen1`; the wave table is `exl3/quant.zig digTableBm` at `m_tile` rows with
//!     `gu_tiles` / `dn_tiles` threadgroups per M tile.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const mtl = @import("../sdk_ext/metal_launch.zig");
const kset = @import("../sdk_ext/kernel_set.zig");
const quant = @import("../sdk_ext.zig").quant;
const xq = @import("../../exl3/quant.zig");
const ops = @import("../ops.zig");

pub const G = ops.MlxOps;
pub const T = G.T;

// ── The geometry the texts are instantiated at ──

pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
/// Bits per weight of the bank's trellis (every layer): 16 x K halfwords a 16 x 16 tile.
pub const k_bits: u32 = 3;
pub const tile_words: u32 = 16 * k_bits;
/// The codebook the texts decode (exl3 mul1, the multiplier in the texts).
pub const codebook = xq.format.codebook;
pub const codebook_multiplier: u64 = xq.format.multiplier;
/// The swiglu clamp the texts fuse (gu_epi, onepass): GLM-5.3-Flash's `swiglu_limit` is DeepSeek's.
pub const swiglu_limit: f64 = xq.fused_act.swiglu_clamped;
/// The widest decode wave: one route (the scheduler's `max_route_ids`).
pub const max_pairs: u32 = @import("../sdk_ext.zig").expert.policy.max_route_ids;
/// The expert stream's slot banks: base, ext, transient.
pub const n_banks: u32 = std.meta.fieldNames(@import("../sdk_ext.zig").expert.BankKind).len;
/// DIG-X: the GEMMs' M tile and threadgroups per M tile (gate|up: both operands' 64-column tiles; down: its own).
pub const m_tile: u32 = 128;
pub const gu_tiles: u32 = 2 * inter / 64;
pub const dn_tiles: u32 = hidden / 64;
/// The onepass epilogue's simdgroups per threadgroup (inter / 128 blocks in whole threadgroups).
pub const onepass_sgs: u32 = 8;

comptime {
    if (hidden % 128 != 0 or inter % 128 != 0) @compileError("glm5_next exl3: the rotations take 128-wide blocks");
    if ((inter / 128) % onepass_sgs != 0 or (hidden / 128) % 8 != 0) @compileError("glm5_next exl3: the rotation launches take whole threadgroups");
    if (tile_words != 48) @compileError("glm5_next exl3: the texts read K = 3 tiles (48 halfwords)");
}

/// One projection's slot arrays (DeepSeek's `ProjArrays`: the stream binds them, the scheduler's default projection).
pub const Proj = xq.ProjArrays(T);
pub const BankArrays = quant.BankArrays(Proj);

/// The dims of a projection's slot arrays (without the leading rows): code, rout, rin.
pub const ProjDims = struct { code: [3]u32, rout: u32, rin: u32 };
pub const Which = enum { gate_up, down };
pub fn projDims(which: Which) ProjDims {
    const in: u32, const out: u32 = if (which == .down) .{ inter, hidden } else .{ hidden, inter };
    return .{ .code = .{ in / 16, out / 16, tile_words }, .rout = out, .rin = in };
}

// ── The texts ──

pub const Kernel = enum { in_rin, gemv, gu_epi, din_rin, dpost, take2v, gemm_gu, gemm_dn, onepass, widen1 };
const n_kernels = std.meta.fieldNames(Kernel).len;

const dir = "kernels/exl3/";
pub const texts: [n_kernels][:0]const u8 = .{
    @embedFile(dir ++ "glm5n_exl3_in_rin.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemv.metal"),
    @embedFile(dir ++ "glm5n_exl3_gu_epi.metal"),
    @embedFile(dir ++ "glm5n_exl3_din_rin.metal"),
    @embedFile(dir ++ "glm5n_exl3_dpost.metal"),
    @embedFile(dir ++ "glm5n_exl3_take2v.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemm_gu.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemm_dn.metal"),
    @embedFile(dir ++ "glm5n_exl3_onepass.metal"),
    @embedFile(dir ++ "glm5n_exl3_widen1.metal"),
};

/// sha256 of each text (checked once at install and in the tests).
pub const text_sha256 = [n_kernels]*const [64]u8{
    "6b5d6987c06750e44ff16cd4d92142ede9b1693643cae6c1ec6391cfad8162d2",
    "632af1fde45f91dd64b1c7c4ba05562d50bdcf4587fbe4e6ca5a91d64a5c6bbe",
    "f9526c59b09561032dad3b048a78e50b4d2c77cef487fd924a65ab6e1dab3388",
    "4cfa833c6427f68000eaac5b77d3964f5975875d95c1634720ae915fcbb4b735",
    "ba6f173e5e190177314cbadd85c6b0c8398253ce76392f1aba93454e873a35d5",
    "27755f90ed5379a9254172c794dd0f8eaa4df85fc2cee4bd0587554a333edfc8",
    "c72444223bf1b65d86ec05f681fb24672180f538474a191a0359ea8a2e02d291",
    "537d8c33523b8d703cca2a5aa8729476fb805ee12fc50170ff572e9bff4a122e",
    "155be055f25f0d138713a187c7b7915178e7feecfe6f645613abe69e9bb56e4f",
    "66885ec42a841c9f578d3e81db9b191d79d83c39126f6b92450fc14fbbbf4b55",
};

/// A text's MLX signature: inputs, outputs, whether MLX makes the inputs row-contiguous (the texts that read strides
/// do not), its header.
const Sig = struct { ins: []const [*:0]const u8, outs: []const [*:0]const u8, contiguous: bool, header: [:0]const u8 = "" };

const MKernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

/// One launch's geometry: grid, threadgroup, template ints, outputs (shape, dtype), an optional template dtype T. Fixed
/// arrays: a launch is a value (no slice into a caller's frame).
pub const IntArg = struct { name: [*:0]const u8, v: c_int };
pub const OutArg = struct { shape: [3]c_int, ndim: u8, dtype: mlx.mlx_dtype };
pub const Launch = struct {
    grid: [3]c_int,
    tg: [3]c_int,
    ints: [2]IntArg = undefined,
    n_int: u8 = 0,
    t_dtype: ?mlx.mlx_dtype = null,
    outs: [2]OutArg = undefined,
    n_out: u8 = 0,

    fn out(l: Launch, shape: []const u32, dtype: mlx.mlx_dtype) Launch {
        var o = l;
        var sh: [3]c_int = @splat(0);
        for (shape, 0..) |d, i| sh[i] = @intCast(d);
        o.outs[o.n_out] = .{ .shape = sh, .ndim = @intCast(shape.len), .dtype = dtype };
        o.n_out += 1;
        return o;
    }
};

fn apply(g: *G, k: MKernel, cfg: Config, ins: []const T, outs: []T) !void {
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

// ── The launches (pure: the tests read them) ──

/// The decode configs of one wave width (prepared at install).
const DecodeCfg = struct { in_rin: Config, gemv_gu: Config, gu_epi: Config, din_rin: Config, gemv_dn: Config, dpost: Config };

pub const Kernels = struct {
    k: [n_kernels]MKernel = @splat(.{ .ctx = null }),
    /// [P - 1]: the decode launches of a P-pair wave.
    dec: [max_pairs]?DecodeCfg = @splat(null),
    /// [P - 1]: int32 [P] 0..P-1 (in_rin's row map: the pairs' rows are taken already), kept.
    tok: [max_pairs]T = @splat(.{ .ctx = null }),

    pub fn deinit(self: *Kernels) void {
        for (self.k) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (self.dec) |d| if (d) |c| for ([_]Config{ c.in_rin, c.gemv_gu, c.gu_epi, c.din_rin, c.gemv_dn, c.dpost }) |x| {
            _ = mlx.mlx_fast_metal_kernel_config_free(x);
        };
        for (self.tok) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
        self.* = .{};
    }

};

// ── Decode: the banked PREP=rin chain ──

/// x bf16 [P, hidden] (each pair's token row), ids u32 [P] packed `bank << 24 | slot row` -> the clamped SwiGLU f32
/// [P, inter]: in_rin, the gate GEMV, the up GEMV, gu_epi.
pub fn gateUp(k: *const Kernels, g: *G, x: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p: u32 = @intCast(g.shapeOf(x).d[0]);
    const c = k.dec[p - 1].?;
    var xs: [2]T = undefined;
    try apply(g, k.get(.in_rin), c.in_rin, &.{ x, k.tok[p - 1], banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
    var zg: [1]T = undefined;
    var zu: [1]T = undefined;
    try apply(g, k.get(.gemv), c.gemv_gu, &.{ xs[0], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, &zg);
    try apply(g, k.get(.gemv), c.gemv_gu, &.{ xs[1], ids, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &zu);
    var out: [1]T = undefined;
    try apply(g, k.get(.gu_epi), c.gu_epi, &.{ zg[0], zu[0], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
    return out[0];
}

/// h f32 [P, inter] (gateUp's), the same packed ids -> bf16 [P, hidden]: din_rin, the down GEMV, dpost (its f32 word
/// rounded once on store).
pub fn down(k: *const Kernels, g: *G, h: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p: u32 = @intCast(g.shapeOf(h).d[0]);
    const c = k.dec[p - 1].?;
    var hd: [1]T = undefined;
    try apply(g, k.get(.din_rin), c.din_rin, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
    var zd: [1]T = undefined;
    try apply(g, k.get(.gemv), c.gemv_dn, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, &zd);
    var out: [1]T = undefined;
    try apply(g, k.get(.dpost), c.dpost, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
    return out[0];
}

// ── Prompt: DIG-X over one wave (one bank, <= 16 experts, rows grouped by expert in table order) ──

const testing = std.testing;
