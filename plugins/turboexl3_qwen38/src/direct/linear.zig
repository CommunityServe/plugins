//! The trunk's shared helpers on the K path: its linears, the lm_head, the norms and activations, the token
//! embedding. Today's `trunk/linear.zig`, which the pieces call in its place.
//! A K pack's linears are EXL3 in the trunk format or bf16 (pre-transposed to `[in, out]` at load, one matmul); its
//! lm_head and attention projections are EXL3 and its embedding is bf16. Anything else is a named refusal.

const mlx = @import("../mlx_serve/mlx/ffi.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const FUSED_ROWS_MAX_M = @import("../mlx_serve/sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const fusedSwiGLU = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;

/// `x @ w`: an EXL3 linear through the trunk format, a bf16 one as one matmul.
pub inline fn qmatmul(self: *const Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    _ = bi;
    if (self.formats.linearOf(w)) |fw| return self.formats.matmul(fw, x, self.s);
    return dense(x, w, sc, self.s);
}

/// A bf16 linear on `s`: today's `qmatmulBits` for a weight with no scales, one matmul.
pub fn dense(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (sc.ctx != null) {
        log.err("[direct] a quantized linear outside the trunk format: a K pack has none\n", .{});
        return error.DirectLinearIsExl3OrBf16;
    }
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, w, s));
    return y;
}

/// The logits projection, an EXL3 linear on a K pack.
pub inline fn lmHeadProject(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    const fw = self.formats.linearOf(self.lm_head_w) orelse {
        log.err("[direct] an lm_head outside the trunk format: a K pack's is EXL3\n", .{});
        return error.DirectHeadIsExl3;
    };
    return self.formats.matmul(fw, x, self.s);
}

/// The head over a hidden of another dtype (a draft's), cast to the trunk's activation dtype first.
pub fn lmHeadForeign(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(x) == self.actDtype()) return lmHeadProject(self, x);
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, self.actDtype(), self.s));
    return lmHeadProject(self, xc);
}

/// A joined row group serves one-row steps.
pub fn rowGroupServes(self: *const Trunk, rows: c_int) bool {
    _ = self;
    return rows <= FUSED_ROWS_MAX_M;
}

/// An attention projection, an EXL3 linear on a K pack (today's decode-only requant serves dense ones only).
pub inline fn attnProj(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, decode_shape: bool, layer: u32) !mlx.mlx_array {
    _ = .{ sc, bi, decode_shape, layer };
    const fw = self.formats.linearOf(w) orelse {
        log.err("[direct] an attention projection outside the trunk format: a K pack's are EXL3\n", .{});
        return error.DirectAttnProjIsExl3;
    };
    return self.formats.matmul(fw, x, self.s);
}

pub inline fn rmsNorm(self: *const Trunk, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
    return result;
}

pub fn silu(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
    return result;
}

/// silu(gate) * up: the one fused kernel, else the two ops.
pub fn computeGeglu(self: *const Trunk, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSwiGLU(self.s, gate, up)) |fused| return fused;
    const activated = try silu(self, gate);
    defer _ = mlx.mlx_array_free(activated);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, up, self.s));
    return result;
}

/// The token rows `[B, L, hidden]`, gathered from the bf16 table.
pub fn embedding(self: *const Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    if (self.emb_w.ctx == null) return error.MissingWeight;
    if (self.emb_s.ctx != null) {
        log.err("[direct] a quantized embedding: a K pack's is bf16\n", .{});
        return error.DirectEmbeddingIsBf16;
    }
    const id_shape = mlx.getShape(token_ids);
    const batch = id_shape[0];
    const seq_len = id_shape[1];
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &[_]c_int{batch * seq_len}, 1, self.s));
    var taken_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_w);
    try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_astype(&emb, taken_w, .bfloat16, self.s));
    var reshaped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&reshaped, emb, &[_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) }, 3, self.s));
    return reshaped;
}
