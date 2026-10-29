//! copied_from: src/sdk/blocks/moe/moe_prefill_cuda.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MoE prefill over fp expert banks (nvfp4, mxfp8) on MLX's CUDA backend: the sorted gather GEMM,
//! gate+up+SwiGLU in one launch, and the weighted top-k sum. Weights stream through registers four
//! 128-wide K stages ahead, each warp load reading 64+ contiguous bytes of a row (32-byte row pieces,
//! as `qmm_sm80` reads them, run at half the DRAM bandwidth); the CTA turns each stage into a bf16
//! tile once, and `ldmatrix` feeds it to the same `mma.m16n8k16` in the same K order, so every output
//! is bit-identical to the chain's (the weights are exact in bf16). Where MLX would not run `qmm_sm80`
//! the kernels decline.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");

pub const Format = enum(c_int) { mxfp8 = 1, nvfp4 = 3 };

const BM = 64;
const BN = 128;
const BK = 128;
const STAGES = 2;
/// 128-wide K stages of weights held in registers ahead of their use.
const PREFETCH = 4;
const THREADS = 256;

fn sharedBytes() c_int {
    return STAGES * BM * BK * 2 + BN * BK * 2 + 1024;
}

const HEADER = kernelSource("kernels/moe/moe_prefill_cuda_header.cu");

// Grid (THREADS * N/128, E, splits): CTA (nt, e, z) computes y columns [nt*128, +128) for expert e's
// rows in 64-row tiles z, z+splits, ...
const SOURCE = kernelSource("kernels/moe/moe_prefill_cuda.cu");

// Grid (THREADS * I/64, E, splits): CTA (nt, e, z) computes act columns [nt*64, +64) of expert e's rows,
// B rows 0-63 from the gate bank and 64-127 from the up bank; x rows come through the routing map `rows`.
// The epilogue is the chain's: each projection rounded to T, gate times its expert scale, then
// silu(g) * u with MLX's own functors in that order.
const GU_SOURCE = kernelSource("kernels/moe/moe_prefill_cuda_gu.cu");

/// MLX's sorted gather runs `qmm_sm80` only from 16 rows and 4 rows per expert on average
/// (`supports_gather_qmm_rhs_sm80`); below that its mat-vec sums in another order, so we decline too.
fn mlxTakesExpertPath(rows: c_int, experts: c_int) bool {
    return rows >= 16 and experts > 0 and @divTrunc(rows, experts) >= 4;
}

var kernels: [2]?mlx.mlx_fast_cuda_kernel = .{ null, null };
var engaged = false;
/// Gathers served (the tests assert the path engaged).
pub var served: u64 = 0;

fn kernelFor(fmt: Format) !mlx.mlx_fast_cuda_kernel {
    const slot = &kernels[if (fmt == .mxfp8) 0 else 1];
    if (slot.*) |k| return k;
    const ins = [_][*:0]const u8{ "x", "w", "s", "inds" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const outs = [_][*:0]const u8{"y"};
    const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const name: [*:0]const u8 = if (fmt == .mxfp8) "msv_moe_prefill_mxfp8" else "msv_moe_prefill_nvfp4";
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, SOURCE, HEADER, true, sharedBytes());
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    slot.* = k;
    return k;
}

/// `gather_qmm(x, w, s, rhs_indices=inds, sorted)` for bf16 rows: x [R, 1, K] or [R, K], banks
/// w [E, N, K*bits/32] uint32 and s [E, N, K/group] uint8, inds [R] uint32 sorted. Returns
/// [R, 1, N] (x's rank), or null outside the kernel's envelope.
pub fn sortedGather(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, inds: mlx.mlx_array, fmt: Format) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(inds) != .uint32) return null;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .uint8) return null;
    const bits: c_int = if (fmt == .mxfp8) 8 else 4;
    const group: c_int = if (fmt == .nvfp4) 16 else 32;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const is = mlx.getShape(inds);
    if (is.len != 1 or ws.len != 3 or (xs.len != 2 and !(xs.len == 3 and xs[1] == 1))) return null;
    const R = is[0];
    const K = xs[xs.len - 1];
    const E = ws[0];
    const N = ws[1];
    if (xs[0] != R or ws[2] * 32 != K * bits or !mlxTakesExpertPath(R, E)) return null;
    if (@rem(K, BK) != 0 or @rem(N, BN) != 0) return null;
    if (mlx.getShape(sc)[2] * group != K) return null;

    const n_tiles = @divExact(N, BN);
    const tiles_m = @divTrunc(R + BM - 1, BM);
    const splits = std.math.clamp(@divTrunc(16384, n_tiles * E), 1, 8);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ R, N }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, THREADS * n_tiles, E, @min(splits, tiles_m)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, THREADS, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "FMT", @intFromEnum(fmt)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "K", K));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "N", N));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "STG", STAGES));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "BDEP", PREFETCH));
    const ins = [_]mlx.mlx_array{ x, w, sc, inds };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, try kernelFor(fmt), v, c, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (xs.len == 3) try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ R, 1, N }, 3, s)) else try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ R, N }, 2, s));
    if (!engaged) {
        engaged = true;
        log.info("[moe] CUDA prefill expert GEMM engaged: {s} K={d} N={d} experts={d}\n", .{ @tagName(fmt), K, N, E });
    }
    served += 1;
    return out;
}

var gu_kernels: [2]?mlx.mlx_fast_cuda_kernel = .{ null, null };
var gu_engaged = false;
/// Fused gate+up launches served (the tests assert the path engaged).
pub var gu_served: u64 = 0;

/// The sorted prefill's `silu(gate(x) * gscale) * up(x)` in one launch: x [T, K] token rows read
/// through `rows` [R] (each routed row's token), `inds` [R] sorted experts, gate/up banks
/// [E, I, K*bits/32] uint32 with uint8 scales, `gscale` [E] or null. Returns act [R, I], or null
/// outside the kernel's envelope.
pub fn gateUp(s: mlx.mlx_stream, x: mlx.mlx_array, rows: mlx.mlx_array, inds: mlx.mlx_array, gw: mlx.mlx_array, gsc: mlx.mlx_array, uw: mlx.mlx_array, usc: mlx.mlx_array, gscale: ?mlx.mlx_array, fmt: Format) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(inds) != .uint32 or mlx.mlx_array_dtype(rows) != .uint32) return null;
    if (gscale) |g| if (mlx.mlx_array_dtype(g) != .bfloat16) return null;
    inline for (.{ gw, uw }) |b| if (mlx.mlx_array_dtype(b) != .uint32) return null;
    inline for (.{ gsc, usc }) |b| if (mlx.mlx_array_dtype(b) != .uint8) return null;
    const bits: c_int = if (fmt == .mxfp8) 8 else 4;
    const group: c_int = if (fmt == .nvfp4) 16 else 32;
    const xs = mlx.getShape(x);
    const gsh = mlx.getShape(gw);
    const is = mlx.getShape(inds);
    if (xs.len != 2 or gsh.len != 3 or is.len != 1 or mlx.getShape(rows).len != 1 or mlx.getShape(rows)[0] != is[0]) return null;
    if (!std.mem.eql(c_int, gsh, mlx.getShape(uw))) return null;
    const R = is[0];
    const K = xs[1];
    const E = gsh[0];
    const I = gsh[1];
    if (gsh[2] * 32 != K * bits or @rem(K, BK) != 0 or @rem(I, 64) != 0 or !mlxTakesExpertPath(R, E)) return null;
    if (mlx.getShape(gsc)[2] * group != K or mlx.getShape(usc)[2] * group != K) return null;

    const slot = &gu_kernels[if (fmt == .mxfp8) 0 else 1];
    if (slot.* == null) {
        const ins = [_][*:0]const u8{ "x", "rows", "inds", "gw", "gs", "uw", "us", "gscale" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const outs = [_][*:0]const u8{"act"};
        const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const name: [*:0]const u8 = if (fmt == .mxfp8) "msv_moe_prefill_gate_up_mxfp8" else "msv_moe_prefill_gate_up_nvfp4";
        const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, GU_SOURCE, HEADER, true, sharedBytes());
        if (k.ctx == null) return error.CudaKernelCreateFailed;
        slot.* = k;
    }
    const n_tiles = @divExact(I, 64);
    const tiles_m = @divTrunc(R + BM - 1, BM);
    const splits = std.math.clamp(@divTrunc(16384, n_tiles * E), 1, 8);
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ R, I }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, THREADS * n_tiles, E, @min(splits, tiles_m)));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, THREADS, 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    inline for (.{ .{ "FMT", @intFromEnum(fmt) }, .{ "K", K }, .{ "I", I }, .{ "STG", STAGES }, .{ "BDEP", PREFETCH }, .{ "GSCALE", @as(c_int, @intFromBool(gscale != null)) } }) |a|
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, a[0], a[1]));
    const ins = [_]mlx.mlx_array{ x, rows, inds, gw, gsc, uw, usc, gscale orelse x };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, slot.*.?, v, c, s));
    var act = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_vector_array_get(&act, o, 0));
    if (!gu_engaged) {
        gu_engaged = true;
        log.info("[moe] CUDA prefill gate+up+SwiGLU engaged: {s} K={d} I={d} experts={d}\n", .{ @tagName(fmt), K, I, E });
    }
    gu_served += 1;
    return act;
}

// Block per token, thread per 8 output columns: the chain's unsort, routing-weight Multiply and
// sequential bf16 Sum over the top-k (MLX's col_reduce_small), k in order.
const REDUCE_SOURCE = kernelSource("kernels/moe/moe_prefill_cuda_reduce.cu");

var reduce_kernel: ?mlx.mlx_fast_cuda_kernel = null;
var reduce_engaged = false;

/// sum_k scores[b,s,k] * down[inverse[(b*S+s)*K + k]]: the routed rows back in token order,
/// weighted and summed over the top-k, [B, S, H]. down [B*S*K, H] and scores [B, S, K] bf16,
/// inverse [B*S*K] uint32. Null outside the kernel's envelope.
pub fn weightedSum(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const is = mlx.getShape(inverse);
    if (ds.len != 2 or ss.len != 3 or is.len != 1) return null;
    const tokens = ss[0] * ss[1];
    const K = ss[2];
    const H = ds[1];
    if (tokens < 1 or ds[0] != tokens * K or is[0] != ds[0] or @rem(H, 8) != 0 or @divExact(H, 8) > 1024) return null;
    if (reduce_kernel == null) {
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const outs = [_][*:0]const u8{"y"};
        const out_vec = mlx.mlx_vector_string_new_data(&outs, 1);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_cuda_kernel_new("msv_moe_prefill_sum", in_vec, out_vec, REDUCE_SOURCE, HEADER, true, 0);
        if (k.ctx == null) return error.CudaKernelCreateFailed;
        reduce_kernel = k;
    }
    const c = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(c, &[_]c_int{ ss[0], ss[1], H }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(c, tokens * @divExact(H, 8), 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(c, @divExact(H, 8), 1, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "TOPK", K));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(c, "H", H));
    const ins = [_]mlx.mlx_array{ down, inverse, scores };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&o, reduce_kernel.?, v, c, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, o, 0));
    if (!reduce_engaged) {
        reduce_engaged = true;
        log.info("[moe] CUDA prefill weighted sum engaged: top-{d} H={d}\n", .{ K, H });
    }
    return y;
}

const testing = std.testing;
