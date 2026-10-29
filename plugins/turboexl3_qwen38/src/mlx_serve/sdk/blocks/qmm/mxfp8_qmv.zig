//! copied_from: src/sdk/blocks/qmm/mxfp8_qmv.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MXFP8 linears at decode and verify widths (1..8 rows): one dispatch, the weight row read
//! once and applied to every activation row. MLX's fp kernel decodes e4m3 per element; here two
//! bytes become a half2 by bit placement.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const MAX_ROWS: c_int = 8;
const SGS: c_int = 4;

/// Bytes 0 and 2 of `w` as a half2 (value x 2^-8), the e4m3 fields shifted into half's.
pub const HEADER = kernelSource("kernels/qmm/mxfp8_qmv_header.metal");

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// y [.., N] = x [.., K] . W^T, or null outside the kernel (caller keeps MLX's matmul).
/// `w` [N, K/4] u32 e4m3 words, `sc` [N, K/32] e8m0.
pub fn linear(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    // Measured on M4 (g16) and the M5 Ultra (g17); other generations keep MLX's kernel until someone measures them.
    const g = @import("../../../mlx/device.zig").deviceGeneration();
    if ((g.gen != 16 and g.gen != 17) or g.phone) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(sc);
    if (wsh.len != 2 or ssh.len != 2 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .uint8) return null;
    const N = wsh[0];
    const K = wsh[1] * 4;
    if (ssh[0] != N or ssh[1] * 32 != K) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len == 0 or xsh.len > 8 or xsh[xsh.len - 1] != K) return null;
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;

    if (kernel == null) {
        const ins = [_][*:0]const u8{ "x", "w_q", "scales" };
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_mxfp8_qmv", in_vec, out_vec, kernelSource("kernels/qmm/mxfp8_qmv.metal"), HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xsh.len], xsh);
    out_shape[xsh.len - 1] = N;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, xsh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32, N, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, SGS, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "M", rows } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    const inputs = [_]mlx.mlx_array{ x, w, sc };
    const v = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel.?, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    if (!engaged) {
        engaged = true;
        log.info("[mxfp8] qmv kernel engaged: rows={d} N={d} K={d}\n", .{ rows, N, K });
    }
    return y;
}

const testing = std.testing;
