//! copied_from: src/sdk/blocks/attention/sparse_block/decode_cuda.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA attention for a decode token or the rows of a verify block over dense bf16 KV on MLX's CUDA
//! backend, the CUDA twin of `qsa_decode.zig`: each row's block picks go straight to a split pass and
//! a merge pass, with no index arithmetic, gather, mask or SDPA between them. A warp owns one q head and `CH`
//! of the visible keys, eight keys at a time with the head dim over the lanes; scores, softmax and
//! the value sum stay in f32. Not MLX's summation order: held to no more error than the masked SDPA
//! against an f64 truth.
const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");

const CH: c_int = 32; // visible keys per warp, a multiple of 8 and of RATIO
const BD: c_int = 256;
/// Verify widths the row grid serves (the dense mask arm takes wider blocks).
pub const MAX_ROWS: c_int = 16;

const HEADER = kernelSource("kernels/attention/qsa_decode_header.cu");

// Grid (NS splits, Hk, rows): a block of GQA warps, warp w the q head hk * GQA + w. Row r is the
// cache position kv - rows + r: it sees its own picks that are complete blocks before it, and its
// own incomplete tail.
const PARTS_SOURCE = kernelSource("kernels/attention/qsa_decode_parts.cu");

// Block per (head, row), thread per dim. An empty split writes l = 0, the skip flag.
const MERGE_SOURCE = kernelSource("kernels/attention/qsa_decode_merge.cu");

var parts_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var merge_kernel: ?mlx.mlx_fast_cuda_kernel = null;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_cuda_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: the kernel reads their strides.
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, HEADER, false, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    return k;
}

/// `[1, Hq, S, 256]` bf16 attention of the query rows `q`, the last S cache positions, each over
/// its own picks in `blocks` `[1, S, kb]` int32 (RATIO tokens each) plus its incomplete tail; `k`/`v`
/// are the dense cache views `[1, Hk, kv, 256]`. Null outside the kernel's geometry (the caller
/// keeps its arm).
pub fn attend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s) or q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    const rows = qs[2];
    if (qs[0] != 1 or rows < 1 or rows > MAX_ROWS or qs[3] != BD or ks[0] != 1 or ks[3] != BD or bs[0] != 1 or bs[1] != rows or ks[2] < rows) return null;
    const hq = qs[1];
    const hk = ks[1];
    if (hk <= 0 or ratio <= 0 or bs[2] < 1 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(k)[3] != 1 or mlx.mlx_array_strides(v)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;
    const gqa = @divExact(hq, hk);
    const kb = bs[2];

    if (parts_kernel == null) {
        parts_kernel = try newKernel("msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, PARTS_SOURCE);
        merge_kernel = try newKernel("msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, MERGE_SOURCE);
    }
    const ns = @divTrunc(kb * ratio + ratio - 1 + CH - 1, CH);
    const pc = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(pc);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(pc, &[_]c_int{ rows * hq, ns, BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(pc, &[_]c_int{ rows * hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(pc, ns * gqa * 32, hk, rows));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(pc, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(pc, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", kb }, .{ "RATIO", ratio }, .{ "CH", CH }, .{ "NS", ns } }) |a|
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(pc, a[0], a[1]));

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const ins = [_]mlx.mlx_array{ q, scl, blocks, k, v };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&parts, parts_kernel.?, in_vec, pc, s));
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts, 1));

    const mc = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(mc, &[_]c_int{ 1, hq, rows, BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(mc, BD * hq, rows, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(mc, BD, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(mc, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(mc, "NS", ns));
    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, merge_kernel.?, merge_vec, mc, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

const testing = std.testing;
