//! copied_from: src/sdk/blocks/attention/sparse_block/decode.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! QSA attention for ONE decode token over dense bf16 KV: the indexer's sorted block picks go
//! straight to a split pass and a merge pass, with no mask, gather or SDPA between them.
//!
//! Split pass: a simdgroup owns one q head and `CH` of the visible keys, eight keys at a time with
//! the head dim spread over the lanes. The eight partial dots reduce in one butterfly that leaves
//! key `lane >> 2` on lanes `4k..4k+3` (9 shuffles, not 8 x 5). More heads per simdgroup lose to
//! register pressure.
const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");

const CH: c_int = 32; // visible keys per simdgroup, a multiple of 8 and of RATIO
const BD: c_int = 256;

pub const HEADER = kernelSource("kernels/attention/qsa_decode_header.metal");

pub const PARTS_SOURCE = kernelSource("kernels/attention/qsa_decode_parts.metal");

// One threadgroup a head. An empty split writes l = 0, the skip flag.
pub const MERGE_SOURCE = kernelSource("kernels/attention/qsa_decode_merge.metal");

var parts_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

const Key = struct { hq: c_int, hk: c_int, kb: c_int, ratio: c_int };
const Cfgs = struct { key: Key, parts: mlx.mlx_fast_metal_kernel_config, merge: mlx.mlx_fast_metal_kernel_config };
var decode_cfgs: ?Cfgs = null;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: the kernel reads their strides.
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn configsFor(key: Key, ns: c_int) !Cfgs {
    if (decode_cfgs) |c| {
        if (std.meta.eql(c.key, key)) return c;
        _ = mlx.mlx_fast_metal_kernel_config_free(c.parts);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.merge);
        decode_cfgs = null;
    }
    const gqa = @divExact(key.hq, key.hk);
    const parts = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(parts, ns * gqa * 32, key.hk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(parts, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(parts, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", key.kb }, .{ "RATIO", key.ratio }, .{ "CH", CH }, .{ "NS", ns } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(parts, kv[0], kv[1]));
    }
    const merge = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge, &[_]c_int{ 1, key.hq, 1, BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge, BD * key.hq, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge, BD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(merge, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge, "NS", ns));
    decode_cfgs = .{ .key = key, .parts = parts, .merge = merge };
    return decode_cfgs.?;
}

/// `[1, Hq, 1, 256]` bf16 attention of one query row `q` over the keys `blocks` `[1, 1, kb]` int32
/// selects (RATIO tokens each) plus the incomplete tail; `k`/`v` are the dense cache views
/// `[1, Hk, kv, 256]`. Null when the geometry is outside the kernel's set (the caller keeps its arm).
pub fn attend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    if (qs[0] != 1 or qs[2] != 1 or qs[3] != BD or ks[0] != 1 or ks[3] != BD or bs[0] != 1 or bs[1] != 1) return null;
    const hq = qs[1];
    const hk = ks[1];
    // An input under 8 elements binds in `constant` space, where the kernel's `device int*` fails to compile.
    if (hk <= 0 or ratio <= 0 or bs[2] < 8 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;

    if (parts_kernel == null) {
        parts_kernel = try newKernel("msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, PARTS_SOURCE);
        merge_kernel = try newKernel("msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, MERGE_SOURCE);
    }
    const ns = @divTrunc(bs[2] * ratio + ratio - 1 + CH - 1, CH);
    const c = try configsFor(.{ .hq = hq, .hk = hk, .kb = bs[2], .ratio = ratio }, ns);

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const ins = [_]mlx.mlx_array{ q, scl, blocks, k, v };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, parts_kernel.?, in_vec, c.parts, s));
    if (mlx.mlx_vector_array_size(parts) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, merge_kernel.?, merge_vec, c.merge, s));
    if (mlx.mlx_vector_array_size(o) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

const testing = std.testing;
