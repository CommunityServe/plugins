//! copied_from: src/sdk/blocks/qmm/qmv_cuda.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! 2..8 rows of x against a 4-bit affine weight on MLX's CUDA backend, through tensor cores.
//! One `mma.m16n8k16` takes 16 weight rows by 8 x rows, so every width up to 8 costs one weight
//! read. Inside each 128-wide k block a lane owns 32 consecutive k (its packed words and its x
//! slice load contiguously); the dot product is the same under that k permutation as long as
//! A and B use it alike. Story and measurements: docs/CUDA.md.
const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const errVsTruth = @import("qmv2.zig").errVsTruth;

pub const MAX_ROWS = 8;
/// Warps of a block split K over the same 16 weight rows, then reduce through shared memory.
const SPLIT_K = 16;

const HEADER = kernelSource("kernels/qmm/qmv_cuda_header.cu");

// Lane (gid = lane/4, tig = lane%4) owns k = blk*128 + tig*32 .. +31 of weight rows gid and
// gid+8 and of x row gid. Mma step s feeds k offsets s*4+{0,1} as the fragment's tig*2+{0,1}
// and s*4+{2,3} as its tig*2+8+{0,1}.
const SOURCE = kernelSource("kernels/qmm/qmv_cuda.cu");

/// Measured on sm_120 (RTX 5060 Ti); every other device keeps MLX's own kernels unless
/// `MLX_SERVE_CUDA_QMV_TC=1`. `=0` turns it off everywhere. The bf16/f16 mma needs sm_80.
pub fn defaultOn(major: usize, minor: usize, raw: ?[]const u8) bool {
    if (major < 8) return false;
    if (raw) |e| if (e.len > 0) return !std.mem.eql(u8, e, "0");
    return major == 12 and minor == 0;
}

var enabled_cache: ?bool = null;

fn enabled() bool {
    if (enabled_cache) |v| return v;
    const v = mlx.cudaAvailable() and defaultOn(mlx.cudaComputeCapability(.major), mlx.cudaComputeCapability(.minor), env.span(.MLX_SERVE_CUDA_QMV_TC));
    enabled_cache = v;
    return v;
}

/// Whether the kernel serves this call: rows of x, its inner dim, the weight's quantization and dtype.
pub fn eligible(rows: usize, k: usize, bits: u32, group_size: u32, has_bias: bool, dt: mlx.mlx_dtype) bool {
    if (rows < 2 or rows > MAX_ROWS or k == 0 or k % 128 != 0) return false;
    if (bits != 4 or !has_bias) return false;
    if (group_size != 32 and group_size != 64 and group_size != 128) return false;
    return dt == .bfloat16 or dt == .float16;
}

/// `x @ dequant(w).T` for eligible calls on a device where the kernel is on; null = decline.
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamOnGpuDevice(s)) return null;
    const y = try run(x, w, sc, bi, bits, group_size, s) orelse return null;
    if (!engaged_logged) {
        engaged_logged = true;
        const xs = mlx.getShape(x);
        log.info("[qmv-cuda] tensor-core rows kernel engaged: first call rows={d} N={d} K={d} (MLX_SERVE_CUDA_QMV_TC=0 restores stock)\n", .{ @divExact(mlx.mlx_array_size(x), @as(usize, @intCast(xs[xs.len - 1]))), mlx.getShape(w)[0], xs[xs.len - 1] });
    }
    return y;
}

var engaged_logged = false;

fn run(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len == 0 or xs.len > 8 or ws.len != 2) return null;
    const k: usize = @intCast(xs[xs.len - 1]);
    var rows: usize = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= @intCast(d);
    const dt = mlx.mlx_array_dtype(x);
    if (!eligible(rows, k, bits, group_size, bi.ctx != null, dt)) return null;
    if (mlx.mlx_array_dtype(sc) != dt or mlx.mlx_array_dtype(bi) != dt) return null;
    if (@as(usize, @intCast(ws[1])) * 8 != k) return null;
    const n = ws[0];
    const groups: c_int = @intCast(k / group_size);
    for ([_]mlx.mlx_array{ sc, bi }) |a| {
        const as = mlx.getShape(a);
        if (as.len != 2 or as[0] != n or as[1] != groups) return null;
    }

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, s));

    const config = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(config);
    const out_shape = [_]c_int{ @intCast(rows), n };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &out_shape, 2, dt));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(config, 32, SPLIT_K, @divFloor(n + 15, 16)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(config, 32, SPLIT_K, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "T", dt));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "GROUP", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "SK", SPLIT_K));

    const ins = [_]mlx.mlx_array{ x2, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&outs, try kernel(), in_vec, config, s));
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outs, 0));

    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = n;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, xs.len, s));
    return y;
}

var kernel_cache: ?mlx.mlx_fast_cuda_kernel = null;

fn kernel() !mlx.mlx_fast_cuda_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "x", "w", "scales", "biases" };
    const out_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new("msv_qmv_tc", in_vec, out_vec, SOURCE, HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    kernel_cache = k;
    return k;
}
