//! copied_from: src/sdk/blocks/moe/moe_fp4.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MoE decode over fp4 expert banks (MXFP4 / NVFP4), read in place: one
//! dispatch for gate+up+SwiGLU across the top-k experts, one for down and the
//! score-weighted sum. Each simdgroup owns ROWS output rows and keeps its 16
//! values of x in registers across them (MLX's `fp_qmv_fast` shape), so x is
//! read once per row group rather than once per row. ModelOpt NVFP4 banks add a
//! per-expert gate scale (before the SiLU); their up x down scales ride the scores.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const parity = @import("../../testing/parity.zig");
const moe_affine4 = @import("moe_affine4.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;

pub const ROWS: c_int = 4;
pub const SGS: c_int = 2;
/// Widest token count worth sending here: a row re-reads its own experts, so
/// past two rows MLX's sorted gather (one read per expert per row group) wins.
pub const MAX_ROWS: c_int = 2;

/// e2m1 nibble -> its value times 2^-14, by bit placement into a half.
pub const E2M1_HEADER = kernelSource("kernels/moe/moe_fp4_e2m1_header.metal");
/// Group scale -> its value times 2^-8: e4m3 (NVFP4) by bit placement into a
/// half, e8m0 (MXFP4) as MLX's `fp8_e8m0` reads it (the byte is a float
/// exponent, 0 is 2^-127). With the e2m1 2^-14, one 2^22 multiply restores both.
pub const NV_SCALE = kernelSource("kernels/moe/moe_fp4_nv_scale.metal");
pub const MX_SCALE = kernelSource("kernels/moe/moe_fp4_mx_scale.metal");

/// Nibbles j and j+4 of a word decode together: one shift puts them in the two halves of a
/// half2, then the e2m1 bit placement serves both lanes.
pub const E2M1X2_HEADER = kernelSource("kernels/moe/moe_fp4_e2m1x2_header.metal");

pub const DOT16 = E2M1X2_HEADER ++ kernelSource("kernels/moe/moe_fp4_dot16.metal");

// grid (32, N/ROWS, R*TOPK), threadgroup (32, SGS, 1): slot e is token e / TOPK's
// e % TOPK-th expert. USCALE and CLAMP are the per-expert up scale and GLM's whole-number SwiGLU clamp. The e2m1 and scale decodes each scale by a power of two;
// 2^22 folds both back once per row.
pub const GATEUP_SOURCE = kernelSource("kernels/moe/moe_fp4_gateup.metal");

// grid (32, H/ROWS, R), threadgroup (32, SGS, 1): every expert's down row for
// token tok, weighted by its routing score, accumulated in the same registers.
pub const DOWNRED_SOURCE = kernelSource("kernels/moe/moe_fp4_downred.metal");

/// Routes each pair's weight reads to its slot bank (`decodeBanks`).
pub const BANKS_HEADER = kernelSource("kernels/moe/moe_fp4_banks_header.metal");

const Kernels = struct { gateup: ?mlx.mlx_fast_metal_kernel = null, downred: ?mlx.mlx_fast_metal_kernel = null };
/// Indexed by format: nvfp4, mxfp4; `banked` the slot-bank builds.
var kernels: [2]Kernels = .{ .{}, .{} };
var banked: [2]Kernels = .{ .{}, .{} };
/// Dispatches over several slot banks (the streamed proofs' engagement count).
pub var banked_dispatches: u32 = 0;
var engaged = false;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, out_shape: []const c_int, dt: mlx.mlx_dtype, grid: [3]c_int, rows: c_int, tmpl: []const struct { [*:0]const u8, c_int }, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
    for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    return y;
}

pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array };
/// Per-expert [E] scales in x's dtype on the gate and up outputs, then `clamp` (0 = none, else
/// GLM's whole-number limit: silu(min(g, c)) * clip(u, -c, c)).
pub const Scales = struct { gate: ?mlx.mlx_array = null, up: ?mlx.mlx_array = null, clamp: u32 = 0 };

/// y [R, hidden] = sum_k scores[r,k] * down_k(silu(gate_k(x_r)) * up_k(x_r)) for
/// R tokens. `x` [R, hidden] bf16/f16, banks [E, out, in/8] u32 + [E, out, in/GS]
/// u8, `inds` [R, TOPK] u32, `scores` [R, TOPK] in x's dtype, `sigtab` the SwiGLU
/// table. `views` (of the same banks) are what the kernels bind; `scales` [E] in
/// x's dtype multiply each expert's gate (before the SiLU) and up (before the clamp); `shared` (one token only)
/// adds the gated shared expert in the epilogue. Null outside the kernels' set.
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, views: ?moe_affine4.Views, scales: Scales, shared: ?moe_affine4.Shared, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, mode: QuantMode, group_size: u32) !?mlx.mlx_array {
    const b = if (views) |v| [3]Bank{ .{ .w = v.gate.w, .s = v.gate.s }, .{ .w = v.up.w, .s = v.up.s }, .{ .w = v.down.w, .s = v.down.s } } else [3]Bank{ gate, up, down };
    return run(s, x, .{ gate, up, down }, &.{b}, null, scales, shared, inds, scores, sigtab, mode, group_size);
}

/// The two dispatches over `bind` (one bank, or the slot banks with `kinds`); `geo` is a bank of the geometry.
fn run(s: mlx.mlx_stream, x: mlx.mlx_array, geo: [3]Bank, bind: []const [3]Bank, kinds: ?mlx.mlx_array, scales: Scales, shared: ?moe_affine4.Shared, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, mode: QuantMode, group_size: u32) !?mlx.mlx_array {
    const gate = geo[0];
    const up = geo[1];
    const down = geo[2];
    if (!mlx.streamIsGpu(s)) return null;
    const ki: usize = switch (mode) {
        .nvfp4 => if (group_size == 16) 0 else return null,
        .mxfp4 => if (group_size == 32) 1 else return null,
        else => return null,
    };
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(scores) != dt) return null;
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3 or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    const inter = gsh[1];
    const hidden = gsh[2] * 8;
    const rows_per_sg = ROWS;
    if (dsh[1] != hidden or dsh[2] * 8 != inter) return null;
    if (@rem(hidden, 16) != 0 or @rem(inter, 16) != 0 or @rem(inter, rows_per_sg * SGS) != 0 or @rem(hidden, rows_per_sg * SGS) != 0) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2 or xsh[1] != hidden) return null;
    const rows = xsh[0];
    const ish = mlx.getShape(inds);
    if (ish.len != 2 or ish[0] != rows or !std.mem.eql(c_int, ish, mlx.getShape(scores))) return null;
    const topk: c_int = ish[1];
    inline for (.{ scales.gate, scales.up }) |sc| if (sc) |a| if (mlx.mlx_array_dtype(a) != dt) return null;
    if (shared) |sh| {
        if (rows != 1 or mlx.mlx_array_dtype(sh.down) != dt or mlx.mlx_array_dtype(sh.logit) != dt or mlx.mlx_array_size(sh.logit) != 1 or mlx.mlx_array_size(sh.down) != @as(usize, @intCast(hidden))) return null;
    }

    const full_header = [2][:0]const u8{ E2M1_HEADER ++ NV_SCALE ++ DOT16, E2M1_HEADER ++ MX_SCALE ++ DOT16 };
    const banks_header = [2][:0]const u8{ full_header[0] ++ BANKS_HEADER, full_header[1] ++ BANKS_HEADER };
    const set = if (kinds != null) &banked[ki] else &kernels[ki];
    if (set.gateup == null) {
        const names = [2][2][*:0]const u8{ .{ "mlxserve_moe_fp4_gateup_nv", "mlxserve_moe_fp4_gateup_mx" }, .{ "mlxserve_moe_fp4_gateup_nv_banks", "mlxserve_moe_fp4_gateup_mx_banks" } };
        const one = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "gsc", "usc" };
        const several = [_][*:0]const u8{ "x", "wg_q0", "g_scales0", "wg_q1", "g_scales1", "wg_q2", "g_scales2", "wu_q0", "u_scales0", "wu_q1", "u_scales1", "wu_q2", "u_scales2", "inds", "kinds", "sigtab", "gsc", "usc" };
        set.gateup = if (kinds == null)
            try makeKernel(names[0][ki], &one, GATEUP_SOURCE, full_header[ki].ptr)
        else
            try makeKernel(names[1][ki], &several, GATEUP_SOURCE, banks_header[ki].ptr);
    }
    if (set.downred == null) {
        const names = [2][2][*:0]const u8{ .{ "mlxserve_moe_fp4_downred_nv", "mlxserve_moe_fp4_downred_mx" }, .{ "mlxserve_moe_fp4_downred_nv_banks", "mlxserve_moe_fp4_downred_mx_banks" } };
        const one = [_][*:0]const u8{ "act", "wd_q", "d_scales", "inds", "scores", "sdown", "glog" };
        const several = [_][*:0]const u8{ "act", "wd_q0", "d_scales0", "wd_q1", "d_scales1", "wd_q2", "d_scales2", "inds", "kinds", "scores", "sdown", "glog" };
        set.downred = if (kinds == null)
            try makeKernel(names[0][ki], &one, DOWNRED_SOURCE, full_header[ki].ptr)
        else
            try makeKernel(names[1][ki], &several, DOWNRED_SOURCE, banks_header[ki].ptr);
    }
    const gs: c_int = @intCast(group_size);
    // Unused operands bind any array of the right type; the template flag keeps them unread.
    const gsc = scales.gate orelse scores;
    const usc = scales.up orelse scores;
    // Per projection, each bank's words and scales (a missing bank binds the first's; `kinds` never reads it).
    var w: [3][6]mlx.mlx_array = undefined;
    for (0..3) |p| for (0..3) |i| {
        const bk = bind[if (i < bind.len) i else 0][p];
        w[p][2 * i] = bk.w;
        w[p][2 * i + 1] = bk.s;
    };
    const gu_tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "K", hidden }, .{ "N", inter }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "GSCALE", @intFromBool(scales.gate != null) }, .{ "USCALE", @intFromBool(scales.up != null) }, .{ "CLAMP", @intCast(scales.clamp) } };
    if (kinds != null) banked_dispatches +%= 1;
    const act = if (kinds) |k|
        try apply(set.gateup.?, &(.{x} ++ w[0] ++ w[1] ++ .{ inds, k, sigtab, gsc, usc }), &.{ rows * topk, inter }, dt, .{ 32, @divExact(inter, rows_per_sg), rows * topk }, rows_per_sg, &gu_tmpl, s)
    else
        try apply(set.gateup.?, &.{ x, w[0][0], w[0][1], w[1][0], w[1][1], inds, sigtab, gsc, usc }, &.{ rows * topk, inter }, dt, .{ 32, @divExact(inter, rows_per_sg), rows * topk }, rows_per_sg, &gu_tmpl, s);
    defer _ = mlx.mlx_array_free(act);
    const sh = shared orelse moe_affine4.Shared{ .down = scores, .logit = scores };
    const dr_tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "I", inter }, .{ "H", hidden }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "SHARED", @intFromBool(shared != null) } };
    const y = if (kinds) |k|
        try apply(set.downred.?, &(.{act} ++ w[2] ++ .{ inds, k, scores, sh.down, sh.logit }), &.{ rows, hidden }, dt, .{ 32, @divExact(hidden, rows_per_sg), rows }, rows_per_sg, &dr_tmpl, s)
    else
        try apply(set.downred.?, &.{ act, w[2][0], w[2][1], inds, scores, sh.down, sh.logit }, &.{ rows, hidden }, dt, .{ 32, @divExact(hidden, rows_per_sg), rows }, rows_per_sg, &dr_tmpl, s);
    if (!engaged) {
        engaged = true;
        @import("../../util/log.zig").info("[moe] fp4 decode kernels engaged: {s} rows={d} topk={d} inter={d} hidden={d} gate_scale={} shared={} banks={d}\n", .{ @tagName(mode), rows, topk, inter, hidden, scales.gate != null, shared != null, bind.len });
    }
    return y;
}

const testing = std.testing;
