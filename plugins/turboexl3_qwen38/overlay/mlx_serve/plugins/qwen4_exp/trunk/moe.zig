//! copied_from: src/plugins/qwen4_exp/trunk/moe.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The trunk's MoE: router logits, top-k routing (fused kernel, compiled closure or chain), the routed experts
//! (fp4 and affine-4 decode kernels, gather-qmv rows, CUDA banks, the sorted NAX prefill, EXL3 banks), the gated
//! shared expert, grouped verify rows and the tensor-parallel split.
//! copied_from: sdk/blocks/moe/{moe,decode,experts,routing}.zig (the qwen4_exp arms: no GLM decode, Hy3 sigmoid
//! routing, sigma-MoE scales, clamped SwiGLU, ungated shared expert or fp32 logit select).

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const Trunk = @import("../trunk.zig").Trunk;
const routing = @import("../../../sdk/blocks/moe/routing.zig");
const MoeRouting = routing.MoeRouting;
const moeRoutingChain = routing.moeRoutingChain;

/// Reference implementation of the MoE routing chain (used both as fallback
/// and as the body the compiled closure traces). Returns owned `inds` +
/// `norm_scores` arrays — caller must free both.
pub fn moeRoutingUncompiled(self: *const Trunk, router_logits: mlx.mlx_array, k: c_int) !MoeRouting {
    return moeRoutingChain(router_logits, k, self.s);
}

/// Compile MoE routing (negate→argpartition→slice→softmax→take_along_axis→sum→expand→divide)
/// into a single fused kernel. Input: router_logits. Outputs: inds, norm_scores.
/// shapeless=false: slice bounds derive from input ndim, so the closure must
/// re-trace per input shape. MoE inference only sees two shapes in practice
/// (decode seq_len=1, prefill seq_len=N), so the trace cost amortizes after
/// the first prefill + first decode.
pub fn compileMoeRouting(self: *Trunk) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &moeRoutingClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, false);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_moe_routing = compiled;
        log.info("MoE routing compiled (kernel fusion enabled)\n", .{});
    }
}

/// Pure subgraph for MoE routing. Inputs:
///   [0] router_logits — shape [..., num_experts]
/// Outputs:
///   [0] inds         — shape [..., K], int32 expert indices (top-K)
///   [1] norm_scores  — shape [..., K], renormalized top-K softmax weights
///
/// The sigma-MoE per-expert-scale path stays outside the closure because it
/// branches on per-layer weights at model-load time.
pub fn moeRoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const k: c_int = @intCast(self.config.num_experts_per_tok);

    var router_logits = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(router_logits);

    const inds_norm = self.moeRoutingUncompiled(router_logits, k) catch return -1;
    defer _ = mlx.mlx_array_free(inds_norm.inds);
    defer _ = mlx.mlx_array_free(inds_norm.norm_scores);

    const out_arr = [_]mlx.mlx_array{ inds_norm.inds, inds_norm.norm_scores };
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
    return 0;
}
