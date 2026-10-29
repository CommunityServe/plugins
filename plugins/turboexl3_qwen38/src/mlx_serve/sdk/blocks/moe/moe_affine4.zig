//! copied_from: src/sdk/blocks/moe/moe_affine4.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MoE decode over 4-bit AFFINE expert banks (Qwen3.8-Flash-Next's routed experts), read in
//! place: one dispatch for gate+up+SwiGLU across the top-k experts, one for down and the
//! score-weighted sum. The shape is `moe_fp4`'s: a simdgroup owns ROWS output rows and keeps
//! its 16 values of x in registers across them, loads are 8 bytes a lane, no threadgroup
//! memory, and the down kernel accumulates every expert of a token in registers. The affine
//! bias rides a per-group sum of x: `sum((s*q + b) * x) = s * dot(q, x) + b * sum(x)`.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

/// One output row a simdgroup: more rows share the x registers but their live state cuts occupancy.
const ROWS: c_int = 1;
const SGS: c_int = 2;

pub const DOT16 = kernelSource("kernels/moe/moe_affine4_dot16.metal");

// grid (32, N/ROWS, TOPK + SE), threadgroup (32, SGS, 1): one token, slot e = its e-th expert;
// slots past TOPK are N-row slices of the ungated shared expert (`sg_*`/`su_*`, SE = its rows / N).
// GS >= 16 and 16 | GS, so a lane's 16 values share one scale and bias.
pub const GATEUP_SOURCE = kernelSource("kernels/moe/moe_affine4_gateup.metal");

// grid (32, H/ROWS, 1), threadgroup (32, SGS, 1): every expert's down row, weighted by its
// routing score, accumulated in the same registers. SHARED: the gated shared expert's output
// (`sdown`, gate logit `glog`) is added in the epilogue. SI > 0: the ungated shared expert's
// down row (`sd_*`, SI inputs after the routed activations) joins the sum at weight 1.
pub const DOWNRED_SOURCE = kernelSource("kernels/moe/moe_affine4_downred.metal");

var gateup_kernel: ?mlx.mlx_fast_metal_kernel = null;
var downred_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, DOT16, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, out_shape: []const c_int, dt: mlx.mlx_dtype, grid: [3]c_int, tmpl: []const struct { [*:0]const u8, c_int }, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, out_shape.ptr, out_shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", ROWS));
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

pub const Bank = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array };

/// One-expert views of a layer's banks, bound in place of them: MLX bills a command buffer for the
/// `data_size` of every array a kernel binds, and a whole bank (419 MB) forced a commit per kernel.
/// The kernels address experts from the buffer pointer, so the view only changes the billing.
pub const Views = struct { gate: Bank, up: Bank, down: Bank };

fn oneExpert(a: mlx.mlx_array, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    if (sh.len != 3) return a;
    var v = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_slice(&v, a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try owned.append(allocator, v);
    return v;
}

/// Views for quantized banks (null for dense ones); fp banks carry no biases and their views none.
/// The caller keeps `owned` alive as long as the banks.
pub fn bindViews(gate: Bank, up: Bank, down: Bank, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !?Views {
    inline for (.{ gate, up, down }) |bk| {
        if (bk.s.ctx == null) return null;
    }
    var out: Views = undefined;
    inline for (.{ "gate", "up", "down" }, .{ gate, up, down }) |name, bk| {
        @field(out, name) = .{ .w = try oneExpert(bk.w, owned, allocator, s), .s = try oneExpert(bk.s, owned, allocator, s), .b = if (bk.b.ctx == null) bk.b else try oneExpert(bk.b, owned, allocator, s) };
    }
    return out;
}

/// The shared expert's down output [hidden] and its gate logit [1], both in x's dtype: the down
/// kernel adds `sigmoid(logit) * down` to the routed sum.
pub const Shared = struct { down: mlx.mlx_array, logit: mlx.mlx_array };

/// An UNGATED shared expert (Flint): 2-D affine-4 weights at the routed banks' group size, run
/// as extra gate+up slices and joined to the down sum at weight 1. Its rows are a multiple of N.
pub const Ungated = struct { gate: Bank, up: Bank, down: Bank };

/// y [hidden] = sum_k scores[k] * down_k(silu(gate_k(x)) * up_k(x)) for ONE token. `x` [hidden]
/// bf16/f16, banks [E, out, in/8] u32 with scales and biases [E, out, in/GS] in x's dtype,
/// `inds` [TOPK] u32, `scores` [TOPK] in x's dtype, `sigtab` the SwiGLU table. Null outside
/// the kernels' set (caller keeps its path). `views` (of the same banks) are what the kernels bind.
pub fn decode(s: mlx.mlx_stream, x: mlx.mlx_array, gate: Bank, up: Bank, down: Bank, views: ?Views, shared: ?Shared, ungated: ?Ungated, inds: mlx.mlx_array, scores: mlx.mlx_array, sigtab: mlx.mlx_array, group_size: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (group_size < 16 or group_size % 16 != 0) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(scores) != dt) return null;
    inline for (.{ gate, up, down }) |bk| {
        if (bk.s.ctx == null or bk.b.ctx == null or mlx.mlx_array_dtype(bk.s) != dt or mlx.mlx_array_dtype(bk.b) != dt or mlx.mlx_array_dtype(bk.w) != .uint32) return null;
    }
    const gsh = mlx.getShape(gate.w);
    const dsh = mlx.getShape(down.w);
    if (gsh.len != 3 or dsh.len != 3 or !std.mem.eql(c_int, gsh, mlx.getShape(up.w))) return null;
    const inter = gsh[1];
    const hidden = gsh[2] * 8;
    const gs: c_int = @intCast(group_size);
    if (dsh[1] != hidden or dsh[2] * 8 != inter) return null;
    if (@rem(hidden, gs) != 0 or @rem(inter, gs) != 0 or @rem(inter, ROWS * SGS) != 0 or @rem(hidden, ROWS * SGS) != 0) return null;
    if (@rem(hidden, 16) != 0 or @rem(inter, 16) != 0) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 1 or xsh[0] != hidden) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1 or !std.mem.eql(c_int, ish, mlx.getShape(scores))) return null;
    const topk: c_int = ish[0];
    if (shared) |sh| {
        if (mlx.mlx_array_dtype(sh.down) != dt or mlx.mlx_array_dtype(sh.logit) != dt or mlx.mlx_array_size(sh.logit) != 1) return null;
        if (mlx.mlx_array_size(sh.down) != @as(usize, @intCast(hidden))) return null;
    }
    var si: c_int = 0;
    if (ungated) |u| {
        if (shared != null) return null;
        inline for (.{ u.gate, u.up, u.down }) |bk| {
            if (bk.s.ctx == null or bk.b.ctx == null or mlx.mlx_array_dtype(bk.s) != dt or mlx.mlx_array_dtype(bk.b) != dt or mlx.mlx_array_dtype(bk.w) != .uint32) return null;
        }
        const ush = mlx.getShape(u.gate.w);
        const udsh = mlx.getShape(u.down.w);
        if (ush.len != 2 or udsh.len != 2 or !std.mem.eql(c_int, ush, mlx.getShape(u.up.w))) return null;
        si = ush[0];
        if (ush[1] * 8 != hidden or udsh[0] != hidden or udsh[1] * 8 != si) return null;
        if (@rem(si, inter) != 0 or @rem(si, gs) != 0) return null;
        // One group size for the whole layer: the scales' width says it.
        if (mlx.getShape(u.gate.s)[1] * gs != hidden or mlx.getShape(u.down.s)[1] * gs != si) return null;
    }

    if (gateup_kernel == null) {
        const ins = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "sg_q", "sg_scales", "sg_biases", "su_q", "su_scales", "su_biases" };
        gateup_kernel = try makeKernel("mlxserve_moe_a4_gateup", &ins, GATEUP_SOURCE);
    }
    if (downred_kernel == null) {
        const ins = [_][*:0]const u8{ "act", "wd_q", "d_scales", "d_biases", "inds", "scores", "sdown", "glog", "sd_q", "sd_scales", "sd_biases" };
        downred_kernel = try makeKernel("mlxserve_moe_a4_downred", &ins, DOWNRED_SOURCE);
    }
    const bound = views orelse Views{ .gate = gate, .up = up, .down = down };
    // Without a shared expert its operands are never read; any array of the right type binds.
    const ug = ungated orelse Ungated{ .gate = bound.gate, .up = bound.up, .down = bound.down };
    const se = @divExact(si, inter);
    const act = try apply(gateup_kernel.?, &.{ x, bound.gate.w, bound.gate.s, bound.gate.b, bound.up.w, bound.up.s, bound.up.b, inds, sigtab, ug.gate.w, ug.gate.s, ug.gate.b, ug.up.w, ug.up.s, ug.up.b }, &.{ topk + se, inter }, dt, .{ 32, @divExact(inter, ROWS), topk + se }, &.{ .{ "K", hidden }, .{ "N", inter }, .{ "GS", gs }, .{ "TOPK", topk } }, s);
    defer _ = mlx.mlx_array_free(act);
    const sh = shared orelse Shared{ .down = scores, .logit = scores };
    const y = try apply(downred_kernel.?, &.{ act, bound.down.w, bound.down.s, bound.down.b, inds, scores, sh.down, sh.logit, ug.down.w, ug.down.s, ug.down.b }, &.{hidden}, dt, .{ 32, @divExact(hidden, ROWS), 1 }, &.{ .{ "I", inter }, .{ "H", hidden }, .{ "GS", gs }, .{ "TOPK", topk }, .{ "SHARED", @intFromBool(shared != null) }, .{ "SI", si } }, s);
    if (!engaged) {
        engaged = true;
        log.info("[moe] affine-4 decode kernels engaged: topk={d} inter={d} hidden={d} gs={d} ungated_shared={d}\n", .{ topk, inter, hidden, gs, si });
    }
    return y;
}

const testing = std.testing;
