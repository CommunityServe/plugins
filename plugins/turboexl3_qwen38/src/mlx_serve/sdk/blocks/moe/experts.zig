//! copied_from: src/sdk/blocks/moe/experts.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/moe/experts.
//! Owns gatherExpertMm.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const mlx_gguf = @import("../../quant/gguf/root.zig");
const gather_qmm_nax = @import("gather_qmm_nax.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const testing = std.testing;
const splitPackedGateUp = @import("../../../mlx/array.zig").splitPackedGateUp;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const verifyQmmNaxAvailable = @import("../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;

/// Gathered matmul for MoE expert dispatch — handles both quantized and dense bf16.
/// Quantized (sc.ctx != null): mlx_gather_qmm with transpose_w=true, w stored [E,out,in].
/// Dense bf16 (sc.ctx == null): mlx_gather_mm; w was pre-transposed to [E,in,out] at load
/// (maybeTransposeForBf16 + generalized transposeBf16Weight), so x @ w is correct with no
/// transpose flag — mirrors mlx-lm's `gather_mm(x, weight.swapaxes(-1,-2))`.
pub fn gatherExpertMm(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, sorted: bool, s: mlx.mlx_stream) !void {
    if (mlx_gguf.kernels.infoOf(w, sc)) |info| {
        const y = try mlx_gguf.kernels.gatherLinear(info, x, w, lhs_idx, rhs_idx, s);
        _ = mlx.mlx_array_free(res.*);
        res.* = y;
        return;
    }
    if (sc.ctx == null) {
        // mlx 0.31.2's `mlx_gather_mm` returns WRONG results with sorted_indices=true
        // for the dense (non-quantized) path — the quantized `mlx_gather_qmm` honors
        // the flag correctly, but the dense kernel does not. The sorted-indices flag is
        // only a performance hint (rhs_indices ARE sorted here), so forcing false is
        // always numerically correct; it just forgoes the sorted-stream optimization.
        // Without this, dense bf16 MoE prefill (the sorted path, S>1) produced fluent
        // but semantically-wrong output (e.g. Qwen3.6-35B-A3B-bf16 calling clean prompts
        // "jumbled"). Verified byte-identical to mlx-lm once forced false. `sorted` is
        // still honored by the quantized branch below, where gather_qmm handles it
        // correctly.
        try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_idx, rhs_idx, false, s));
    } else {
        if (sorted and lhs_idx.ctx == null and (mode == .affine or mode.isFp4())) {
            if (try gather_qmm_nax.sortedGather(x, w, sc, bi, rhs_idx, bits, group_size, mode.isFp4(), verifyQmmNaxAvailable(), s)) |out| {
                _ = mlx.mlx_array_free(res.*);
                res.* = out;
                return;
            }
        }
        const prefill_cuda = @import("moe_prefill_cuda.zig");
        const cuda_fmt: ?prefill_cuda.Format = switch (mode) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
            else => null,
        };
        if (sorted and lhs_idx.ctx == null and bi.ctx == null) if (cuda_fmt) |fmt| {
            if (try prefill_cuda.sortedGather(s, x, w, sc, rhs_idx, fmt)) |out| {
                _ = mlx.mlx_array_free(res.*);
                res.* = out;
                return;
            }
        };
        try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, bi, lhs_idx, rhs_idx, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), sorted, s));
    }
}

// ── tests ──
