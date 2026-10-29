//! copied_from: src/sdk/blocks/linear/prefill_dq_gemm.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/linear/prefill_dq_gemm.
//! Owns PREFILL_DQ_GEMM_MIN_M, prefillDqGemmMinRows, prefillDqGemmRowFloor, prefillDqGemmRowFloorNow, prefill_dq_gemm_env_cached, prefillDqGemmEnabled, and 2 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const testing = std.testing;
const lastDim = @import("../../../mlx/array.zig").lastDim;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const dualDieArch = @import("verify_projection.zig").dualDieArch;
const verifyArch = @import("verify_projection.zig").verifyArch;

// ── Prefill-width dequant+GEMM qmm route ──
// Stock MLX qmm_t runs a 32x32x32 steel tile; at prefill widths (M >= 2048)
// on the qwen 27B shapes that underuses the compute units by ~10% (µbench,
// M4 Max: gate/up q4 M=2048 stock 28.05 ms vs dequant+bf16-GEMM 25.36 ms
// INCLUDING the per-call dequant; pre-dequantized floor 24.64 ms). oMLX
// closes the same gap by re-instantiating qmm_t at bigger tiles (bm 64-128,
// their qwen35_q4_mlp patch); we get within ~3% of that with zero custom
// kernel: dequantize the weight to a bf16 [N,K] transient and run MLX's
// steel GEMM. The transient repeats identical sizes per layer, so the MLX
// allocator cache recycles it; numerics differ from in-kernel dequant only
// by the bf16 rounding of w = s*q + b (pinned no-worse-than-stock by test).
// Decode (M=1) and spec-verify widths never route.
pub const PREFILL_DQ_GEMM_MIN_M: usize = 2048;

/// Rows from which `prefillDqGemm` takes over from stock qmm. 2-bit qmm loses
/// from 384 rows (M4 Max, 5120x17408: +5% at 384, +10% at 1024); other widths
/// keep the 2048 they were tuned on.
pub fn prefillDqGemmMinRows(bits: u32) usize {
    return if (bits == 2) 384 else PREFILL_DQ_GEMM_MIN_M;
}

/// Rows from which one [M,K] x [N,K]^T call takes the route. Pure over its inputs.
/// `prefillDqGemmMinRows` stays the smallest floor any shape gets, which is what the
/// admission bill needs.
pub fn prefillDqGemmRowFloor(bits: u32, ultra_nax: bool, n: usize, k: usize) usize {
    const base = prefillDqGemmMinRows(bits);
    // Off NAX stock qmm is the 32x32 steel tile the base floors were tuned against. On the Ultra's NAX
    // stock qmm is the fused kernel, which holds out to ~0.9 * min(N, K) rows once both N and K are
    // 4096 or more. Below that (Flash-Next's K 640, K 2560 and N 2560 shapes) and at K >= 3N, where
    // the dequantized GEMM runs MLX's split-K kernel, dequant+GEMM wins from the base floor.
    if (!ultra_nax or bits == 2 or @min(n, k) < 4096 or k >= 3 * n) return base;
    return @max(base, @min(n, k) * 9 / 10);
}

/// The shape-aware floor is measured on the M5 Ultra only. The M5 Max crosses the other way on
/// K 5120 (dequant+GEMM wins from 2048 rows; MLX picks different GEMM tiles per arch suffix), so
/// every other chip keeps the base floor.
fn prefillDqGemmRowFloorNow(bits: u32, n: usize, k: usize) usize {
    const ultra_nax = naxAvailable() and dualDieArch(verifyArch() orelse "");
    return prefillDqGemmRowFloor(bits, ultra_nax, n, k);
}

/// Off on CUDA: there it measured slower than stock qmm (3845 vs 4400 tok/s, Llama 3.2 3B, RTX 5060 Ti) and its
/// bf16 weight transients stacked ~5 GB into the prefill peak.
pub fn prefillDqGemmEnabled() bool {
    return prefill_dq_gemm_override orelse !mlx.cudaAvailable();
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var prefill_dq_gemm_engaged: u64 = 0;

/// The dequant+GEMM prefill route (see the block comment above
/// prefillDqGemmEnabled). Returns null when the call must stay on stock qmm.
pub fn prefillDqGemm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!prefillDqGemmEnabled()) return null;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    const last = lastDim(x) orelse return null;
    if (last == 0) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(last));
    const n: usize = @intCast(mlx.getShape(w)[0]);
    if (rows < prefillDqGemmRowFloorNow(bits, n, @intCast(last))) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;

    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = xd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, dq_t, s));
    prefill_dq_gemm_engaged += 1;
    return out;
}

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

// ── tests ──

const qmatmulBits = @import("qmatmul.zig").qmatmulBits;
