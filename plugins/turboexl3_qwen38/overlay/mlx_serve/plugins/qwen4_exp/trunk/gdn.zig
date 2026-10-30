//! copied_from: src/plugins/qwen4_exp/trunk/gdn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The trunk's GatedDeltaNet layer over a slot's recurrent entry: the joined or split projections, the fused decode
//! and verify recurrences (with the per-row state capture, or the rows' recurrence inputs, a partial accept rolls back
//! from), the fused or composed prefill prework, the recurrence kernels, and the sigmoid-gated norm (qwen4_exp's
//! `output_gate_type`). Prefill chunks take the GDN core kernels (`gdn_core.zig`), the joined a | b GEMM (`gdn_ab.zig`)
//! and on the trunk format's linears the fused tail and the overlap (`gdn_tail.zig`).
//! copied_from: sdk/blocks/gdn/{gated_delta_net,gate}.zig, sdk/blocks/conv/conv1d_cache.zig (the qwen4_exp arms: no
//! combined projection, KDA, low-rank gates, Hadamard, ANE or spec-tree paths).

const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const gdn_decode = @import("../../../sdk/blocks/gdn/decode.zig");
const Trunk = @import("../trunk.zig").Trunk;
const gdnGateChain = @import("../../../sdk/blocks/gdn/gate.zig").gdnGateChain;
const gdnChainGateDtype = @import("../../../sdk/blocks/gdn/gate.zig").gdnChainGateDtype;

/// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
/// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel.
/// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
pub fn compileGdnGate(self: *Trunk) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gdnGateClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gdn_gate = compiled;
        log.info("GDN gate compiled (kernel fusion enabled); recurrent state {s}, gate {s}\n", .{ @tagName(self.ssmStateDtype()), @tagName(gdn_decode.gateDtype(self.config.actDtype(), self.ssmStateDtype())) });
    }
}

pub fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    var A_log = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(A_log);
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(a);
    var dt_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = gdnGateChain(A_log, a, dt_bias, gdnChainGateDtype(mlx.mlx_array_dtype(a), self.ssmStateDtype()), self.s) catch return -1;
    const out_arr = [_]mlx.mlx_array{g};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(g);
    return 0;
}
