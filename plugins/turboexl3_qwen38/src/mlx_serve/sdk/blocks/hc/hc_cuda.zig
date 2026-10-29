//! copied_from: src/sdk/blocks/hc/hc_cuda.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! `hc_decode2`'s two-launch hyper-connection read on MLX's CUDA backend: the stream norm (and a
//! pending hcWrite) folds into the down launch, the stream partials meet in the up/mix launch.
//! Rounding sites are the composed chain's (stream', xn, silu, up, gate, mix); only the f32
//! accumulation order differs. Dense bf16 and 2/4/8-bit affine weights.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const hc_decode2 = @import("hc_decode2.zig");

/// Down rows a block owns, one per warp.
const ROWS_TG: c_int = 8;
/// Up columns a block owns, one per warp.
const COLS_TG: c_int = 16;

const HEADER = kernelSource("kernels/hc/hc_cuda_header.cu");

const ND_SOURCE = kernelSource("kernels/hc/hc_cuda_nd.cu");

const UP_SOURCE = kernelSource("kernels/hc/hc_cuda_up.cu");

var nd_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var up_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var engaged = false;
/// Reads served by the two launches (the tests assert the path engaged).
pub var served: u64 = 0;

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

fn launch(kernel: mlx.mlx_fast_cuda_kernel, outs: []const struct { c_int, mlx.mlx_dtype }, grid: [3]c_int, tg: c_int, tmpl: []const struct { [*:0]const u8, c_int }, dt: mlx.mlx_dtype, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{o[0]}, 1, o[1]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", dt));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, kernel, v, c, s));
    if (mlx.mlx_vector_array_size(o) != res.len) return error.CudaKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (res[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(r, o, i));
    }
}

/// `hc_decode2.read`'s contract on CUDA, or null outside its envelope (the caller keeps the
/// unfused chain).
pub fn read(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: mlx.mlx_array,
    rows: c_int,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?hc_decode2.Pending,
) !?hc_decode2.Out {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const dense = bits == 16;
    if (dense and dt != .bfloat16) return null;
    if (bits != 2 and bits != 4 and bits != 8 and !dense) return null;
    if (nw.ctx == null or mlx.mlx_array_dtype(nw) != dt) return null;
    if (!dense) inline for (.{ ds, db, us, ub }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != dt) return null;
    };
    if (iw.ctx != null and mlx.mlx_array_dtype(iw) != dt) return null;
    const vpw: c_int = @intCast(32 / bits);
    const b: c_int = @intCast(bits);
    const gs: c_int = @intCast(group_size);
    const K = hc * hidden;
    if (hc < 1 or hc > 8 or @rem(hidden, 256) != 0 or @rem(gs, 16) != 0 or @rem(hidden, gs) != 0) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2 or dsh[1] * 32 != K * b or ush[0] != K) return null;
    const R = dsh[0];
    if (ush[1] * 32 != R * b or @rem(R, vpw) != 0 or @rem(R, ROWS_TG) != 0 or @rem(R, gs) != 0 or @rem(hidden, COLS_TG) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (rows < 1 or rows > 16 or mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    if (pend) |p| {
        if (mlx.mlx_array_dtype(p.out) != dt or mlx.mlx_array_dtype(p.inj) != dt) return null;
    }

    if (nd_kernel == null) nd_kernel = try makeKernel("msv_hc_read_nd", &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" }, &.{ "xn_out", "parts_out", "xs_out" }, ND_SOURCE);
    if (up_kernel == null) up_kernel = try makeKernel("msv_hc_read_up", &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" }, &.{ "mixed_out", "inj_out" }, UP_SOURCE);

    const pr = R + hc;
    const groups = @divExact(R, ROWS_TG) + inj;
    const tmpl = [_]struct { [*:0]const u8, c_int }{ .{ "HC", hc }, .{ "H", hidden }, .{ "R", R }, .{ "GS", gs }, .{ "BITS", b }, .{ "INJ", inj }, .{ "WR", wr }, .{ "COLS", COLS_TG } };
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    // Dense weights carry no scales; the kernel never reads the operands bound in their place.
    const dsb: [4]mlx.mlx_array = if (dense) .{ nw, nw, nw, nw } else .{ ds, db, us, ub };
    var nd_out: [3]mlx.mlx_array = undefined;
    try launch(nd_kernel.?, &.{ .{ rows * K, dt }, .{ rows * hc * pr, .float32 }, .{ if (wr == 1) rows * K else 1, dt } }, .{ 256, groups * hc, 1 }, 256, &tmpl, dt, &.{ x, nw, if (inj == 1) iw else nw, eps, wo, wi, dw, dsb[0], dsb[1] }, s, &nd_out);
    defer _ = mlx.mlx_array_free(nd_out[0]);
    defer _ = mlx.mlx_array_free(nd_out[1]);
    errdefer _ = mlx.mlx_array_free(nd_out[2]);
    var upmix: [2]mlx.mlx_array = undefined;
    try launch(up_kernel.?, &.{ .{ rows * hidden, dt }, .{ rows * hc, dt } }, .{ 32 * COLS_TG, @divExact(hidden, COLS_TG), 1 }, 32 * COLS_TG, &tmpl, dt, &.{ nd_out[0], nd_out[1], uw, dsb[2], dsb[3] }, s, &upmix);
    var stream = nd_out[2];
    if (wr == 0) {
        _ = mlx.mlx_array_free(stream);
        stream = .{ .ctx = null };
    }
    var inj_out = upmix[1];
    if (inj == 0) {
        _ = mlx.mlx_array_free(inj_out);
        inj_out = .{ .ctx = null };
    }
    if (!engaged) {
        engaged = true;
        log.info("[qwen4] CUDA two-launch hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit\n", .{ hc, hidden, R, bits });
    }
    served += 1;
    return .{ .mixed = upmix[0], .inj = inj_out, .stream = stream };
}

const testing = std.testing;

/// The arrays of one test case; `free` releases every one.
pub const Bag = struct {
    items: [256]mlx.mlx_array = undefined,
    n: usize = 0,
};
