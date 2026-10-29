//! copied_from: src/sdk/blocks/attention/row_attn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Decode attention whose every row has the bits of the one-row step at its
//! position, ported from TensorFold's `row_attention.py` (MIT, see NOTICE).
//! One threadgroup per (kv head, window row, key chunk). Keys are cut into
//! CK-position chunks counted from 0 and a chunk's keys
//! interleave over SPLIT simdgroups; per simdgroup an online softmax in
//! ascending position, the simdgroups merged in order, then the chunks. What
//! rides in the window changes nothing a row computes, so serial decoding
//! through this kernel is the reference a verify window reproduces.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

const CK = 128;
const SPLIT = 4;
const BLK = 4;
pub const MAX_ROWS = 16;

// K/V are the KV cache's views [1, HKV, L, D]: indexed through their strides,
// never copied. The window's rows sit at cache rows P .. P + W - 1 (P = L - W);
// row w is at position P + depth[w] and its keys past P are the window rows
// path[w][0 .. depth[w]] (a chain: depth w, path 0..w; a tree: its ancestors).
pub const PARTIAL = kernelSource("kernels/attention/row_attn_partial.metal");

pub const MERGE = kernelSource("kernels/attention/row_attn_merge.metal");

var partial_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, row_contig: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, "", row_contig, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Whether the kernel serves this attention: bf16 q/k/v, head dim a multiple
/// of 32, whole kv-head groups in one threadgroup, 1..MAX_ROWS query rows.
pub fn fits(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array) bool {
    for ([_]mlx.mlx_array{ q, k, v }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (qs.len != 4 or ks.len != 4 or qs[0] != 1 or ks[0] != 1) return false;
    if (@rem(qs[3], 32) != 0 or ks[3] != qs[3] or @rem(qs[1], ks[1]) != 0) return false;
    const g = @divExact(qs[1], ks[1]);
    return g * SPLIT * 32 <= 1024 and qs[2] >= 1 and qs[2] <= MAX_ROWS and ks[2] >= qs[2];
}

/// A draft tree over a window's rows: `depth [W]` int32 and `path [W, MAXD]`
/// int32 (row w's window rows at depths 0 .. depth[w]), `max_depth` = MAXD - 1.
pub const Tree = struct { depth: mlx.mlx_array, path: mlx.mlx_array, max_depth: c_int };

/// q [1, H, W, D] of a window whose rows sit at the last W positions of k/v
/// [1, HKV, L, D] (the cache views after this step's update) -> [1, H, W, D].
/// `tree` null: a chain.
pub fn sdpa(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, tree: ?Tree, s: mlx.mlx_stream) !bool {
    if (!mlx.streamIsGpu(s) or !fits(q, k, v)) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const h = qs[1];
    const w = qs[2];
    const d = qs[3];
    const hkv = ks[1];
    const g = @divExact(h, hkv);
    const p = ks[2] - w;
    const nch = @divTrunc(ks[2] + CK - 1, CK);

    const pk = try kernel(&partial_kernel, "msv_row_attn_partial", &.{ "Q", "K", "V", "depth", "path", "scale", "dims" }, &.{ "PM", "PL", "PO" }, PARTIAL, false);
    const mk = try kernel(&merge_kernel, "msv_row_attn_merge", &.{ "PM", "PL", "PO", "dims" }, &.{"OUT"}, MERGE, true);

    var chain: [MAX_ROWS * MAX_ROWS]i32 = undefined;
    var chain_depth: [MAX_ROWS]i32 = undefined;
    const t = tree orelse blk: {
        const wu: usize = @intCast(w);
        for (0..wu) |r| {
            chain_depth[r] = @intCast(r);
            for (0..wu) |i| chain[r * wu + i] = @intCast(i);
        }
        break :blk Tree{
            .depth = mlx.mlx_array_new_data(&chain_depth, &[_]c_int{w}, 1, .int32),
            .path = mlx.mlx_array_new_data(&chain, &[_]c_int{ w, w }, 2, .int32),
            .max_depth = w - 1,
        };
    };
    defer if (tree == null) {
        _ = mlx.mlx_array_free(t.depth);
        _ = mlx.mlx_array_free(t.path);
    };
    const dims_v = [_]i32{ p, w, nch, t.max_depth + 1 };
    const dims = mlx.mlx_array_new_data(&dims_v, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const scale_arr = mlx.mlx_array_new_data(&scale, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale_arr);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    try mlx.check(mlx.mlx_contiguous(&qc, q, false, s));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const slots = h * w * nch;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{slots}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{slots}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ slots, d }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * g * SPLIT, nch, hkv * w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * g * SPLIT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", g));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CK", CK));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPLIT", SPLIT));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BLK", BLK));
    const ins = [_]mlx.mlx_array{ qc, k, v, t.depth, t.path, scale_arr, dims };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, pk, in_vec, cfg, s));

    const mcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mcfg, &[_]c_int{ 1, h, w, d }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mcfg, 32, h, w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mcfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mcfg, "D", d));
    var pm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pm);
    var pl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pl);
    var po = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(po);
    try mlx.check(mlx.mlx_vector_array_get(&pm, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pl, parts, 1));
    try mlx.check(mlx.mlx_vector_array_get(&po, parts, 2));
    const mins = [_]mlx.mlx_array{ pm, pl, po, dims };
    const mvec = mlx.mlx_vector_array_new_data(&mins, mins.len);
    defer _ = mlx.mlx_vector_array_free(mvec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, mk, mvec, mcfg, s));
    try mlx.check(mlx.mlx_vector_array_get(out, res, 0));
    return true;
}

// ── tests ──

const testing = std.testing;

fn randBf16(shape: []const c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn slice4(a: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &start, 4, &stop, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return out;
}
