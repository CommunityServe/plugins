//! copied_from: src/sdk/blocks/moe/moe_gather_ubench.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Env-gated microbench for the two fused decode MoE kernels at Qwen3.8-Flash-Next's
//! shapes (E 512, top-10, hidden 2560, intermediate 640, 4-bit affine g64, bf16):
//! `gatherQmvGateUp` and `gatherQmvDownReduce`, against MLX's stock `gather_qmm`
//! chains, so a kernel variant is timed in isolation instead of behind a server boot.
//!   MOE_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="moe gather kernels"

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const down_reduce = @import("down_reduce.zig");
const gather_qmv = @import("gather_qmv.zig");
const gather_qmv_rows = @import("gather_qmv_rows.zig");
const io_util = @import("../../util/io_util.zig");

const E: c_int = 512;
const TOPK: c_int = 10;
const H: c_int = 2560;
const I: c_int = 640;
const BITS: u32 = 4;
const GS: u32 = 64;
const WARM = 5;
const BATCHES = 20;
const CALLS_SMALL = 10;
const MAX_CALLS = 40;

const Bank = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,

};

const Inputs = struct {
    s: mlx.mlx_stream,
    x: mlx.mlx_array, // [H] bf16
    x_down: mlx.mlx_array, // [TOPK, I] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [TOPK], distinct experts each, cycled by call index
    lhs_zero: mlx.mlx_array, // uint32 [TOPK] zeros: every expert reads the one token
    lhs_iota: mlx.mlx_array, // uint32 [TOPK] 0..9: every expert reads its own row
    scores: mlx.mlx_array, // [TOPK] bf16, sums to ~1
};

const Arm = enum { gateup, downred, pair, stock_gateup, stock_downred };

// ── Verify-width rows arm ──────────────────────────────────────────────────────────────────────
// A solo verify of S rows reaches the MoE through `gatherQmvGateUpRows` + `gatherQmvDownReduceRows`
// (S <= 8). Bytes are counted over the DISTINCT experts the S rows pick, since rows that share an
// expert read its weights once.

const RowsIn = struct {
    s: mlx.mlx_stream,
    rows: c_int,
    x: mlx.mlx_array, // [S, H] bf16
    x_act: mlx.mlx_array, // [S, TOPK, I] bf16, the activation the down kernel reads
    scores: mlx.mlx_array, // [S, TOPK] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [S, TOPK], TOPK distinct experts per row
    distinct_avg: f64, // mean distinct experts per index table over the cycled tables
};

const RowsArm = enum { gateup, downred, pair, decode_pair_x_rows };

// The same rows pair, but each call's input depends on the previous call's output (through a zero that the
// graph cannot fold), so the calls cannot overlap: the shape of a forward, where a kernel's ramp-up and tail
// are on the critical path. The three small ops that carry the dependency are the same in every variant.
const ChainMode = enum { pair, gateup, downred, ops };

// A streaming-read reference for the GB/s above: sum over a 2 GiB bf16 array reads every byte once, far
// larger than any cache, so its rate is the DRAM read rate this harness can reach.
// The read ceiling with wide loads: one simdgroup streams its own 64 KiB span with 16-byte loads, UNROLL
// loads in flight per lane, and folds what it read into one word so nothing is dead code. A DRAM rate here
// that beats the reduction above says the reduction was the limit, not the memory.
pub const STREAM_WIDE_SOURCE = kernelSource("kernels/moe/ubench_stream_wide.metal");

/// The stream-wide launch: `nsg` simdgroups over 64 KiB spans, `unroll` loads in flight, grid-stride when `strided`.
pub fn streamWideConfig(nsg: c_int, unroll: c_int, strided: c_int) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const oshape = [_]c_int{nsg};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &oshape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nsg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPAN4", 4096));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "UNROLL", unroll));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "STRIDED", strided));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", nsg));
    return cfg;
}
