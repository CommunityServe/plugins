//! copied_from: src/sdk/quant/hadamard/rht.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Prism Hadamard packs (prism-ml Ternary-Bonsai-2, `prism_hadamard_qwen35`):
//! every packed linear stores its codes in a rotated basis, so the matmul reads
//! `H(signs * x)` per `block`-wide stripe, H the Sylvester Walsh-Hadamard
//! butterfly scaled 1/sqrt(block). The embedding table is the inverse side:
//! the gathered rows become `signs * H(row)`. Signs ship as `<linear>.signs`.
//! `Registry` keys them on the weight handle; `Transformer.qmatmul` and
//! `rawEmbedding` apply them. The input transform is memoized on (x, signs):
//! q/k/v, gate/up and the GDN qkv/z share one normed activation.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const qmm_int8 = @import("../../blocks/qmm/qmm_int8.zig");

pub const SIGNS_SUFFIX = ".signs";

/// One 128-thread group per `block`-wide stripe, element `(sg*P + p)*32 + lane`
/// (P = J/4): lane bits butterfly via `simd_shuffle_xor`, p bits in registers,
/// the two simdgroup bits through one threadgroup exchange.
pub const KERNEL_SOURCE = kernelSource("kernels/quant/hadamard_block.metal");

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "x", "signs" };
    const out_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("msv_hadamard_block", in_vec, out_vec, KERNEL_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = k;
    return k;
}

/// Blocks the butterfly kernels (this one and gdn_decode's) hold, the four Prism's runtime
/// accepts: J/4 floats per lane, J*32 floats of threadgroup memory.
pub fn kernelBlock(block: c_int) bool {
    return block == 512 or block == 1024 or block == 2048 or block == 4096;
}

fn transformKernel(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, n_total: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const shape = mlx.getShape(x);
    const dt = mlx.mlx_array_dtype(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divExact(n_total, block) * 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "J", @divExact(block, 32)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "INV", inverse));
    const inputs = [_]mlx.mlx_array{ x, signs };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(), in_vec, config, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

/// Input side: `H(signs * x)`; inverse: `signs * H(x)`. Computed in f32 like
/// the reference, returned in x's dtype. Caller owns the result.
pub fn transform(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    if (kernelBlock(block) and mlx.streamIsGpu(s)) {
        const sh = mlx.getShape(x);
        var total: c_int = 1;
        for (sh) |d| total *= d;
        if (sh.len > 0 and @rem(sh[sh.len - 1], block) == 0) return transformKernel(x, signs, block, inverse, total, s);
    }
    return transformComposed(x, signs, block, inverse, s);
}

fn transformComposed(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    if (shape.len == 0 or shape.len >= 16) return error.RhtShape;
    const n = shape[shape.len - 1];
    if (block <= 0 or @rem(n, block) != 0) return error.RhtShape;

    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    if (inverse) {
        try mlx.check(mlx.mlx_astype(&src, x, .float32, s));
    } else {
        try mlx.check(mlx.mlx_multiply(&src, x, signs, s)); // f32 signs promote
    }

    var blocked_shape: [16]c_int = undefined;
    @memcpy(blocked_shape[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    blocked_shape[shape.len - 1] = @divExact(n, block);
    blocked_shape[shape.len] = block;
    var blocked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blocked);
    try mlx.check(mlx.mlx_reshape(&blocked, src, &blocked_shape, shape.len + 1, s));

    var had = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(had);
    try mlx.check(mlx.mlx_hadamard_transform(&had, blocked, mlx.mlx_optional_float.none(), s));

    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, had, shape.ptr, shape.len, s));

    var signed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(signed);
    const pre_cast = if (inverse) blk: {
        try mlx.check(mlx.mlx_multiply(&signed, flat, signs, s));
        break :blk signed;
    } else flat;

    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, pre_cast, mlx.mlx_array_dtype(x), s));
    return out;
}

/// Residual add + RMS norm + input rotation: `sum = a + b`, `normed = w * rms(sum)`,
/// `rot = H(signs * normed)`. One 128-thread group per 1024 block; each group
/// re-reads the whole row for the sum of squares instead of syncing with the
/// others. Rounds through T exactly where the composed ops would.
pub const NORM_ROTATE_SOURCE = kernelSource("kernels/quant/norm_rotate.metal");

var norm_rotate_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const NormRotate = struct {
    sum: ?mlx.mlx_array,
    normed: ?mlx.mlx_array,
    rot: mlx.mlx_array,
};

/// Every threadgroup re-reads its whole row for the sum of squares (NB× read
/// amplification), so the kernel serves decode/verify widths only; prefill
/// keeps the composed chain.
pub const NORM_ROTATE_MAX_ROWS: c_int = 16;

pub const Registry = struct {
    allocator: std.mem.Allocator,
    block: c_int,
    /// Every rotated weight's biases are exactly -scales (checked at load).
    bias_is_neg_scale: bool = true,
    map: std.AutoHashMapUnmanaged(usize, mlx.mlx_array) = .{},
    /// One retained handle per DISTINCT sign vector, so siblings reading one
    /// activation hit the memo below.
    pool: std.AutoHashMapUnmanaged(u64, mlx.mlx_array) = .{},
    /// The pooled sign vector per input width, for fused producers. A width
    /// with two distinct vectors is unusable there: a producer seeds the memo
    /// under one vector and the consumer under the other would rotate twice.
    by_width: std.AutoHashMapUnmanaged(c_int, mlx.mlx_array) = .{},
    width_conflict: std.AutoHashMapUnmanaged(c_int, void) = .{},
    memo_key: usize = 0,
    memo_signs: ?*anyopaque = null,
    memo_x: mlx.mlx_array = .{ .ctx = null },
    memo_out: mlx.mlx_array = .{ .ctx = null },
    /// The LOSSY `int8_prefill` setting: prompt-width projections quantize their activation to int8 (`qmm_int8`);
    /// owned, null = off.
    int8: ?*qmm_int8.Cache = null,

};

// ── tests ──
