//! copied_from: src/plugins/qwen4_exp/trunk/linear.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The trunk's projections, norms, activations and embedding: per-weight quant params, the lm_head, the decode-only
//! attention requant, the dense MLP, the token embedding and the vision splice.
//! copied_from: sdk/blocks/linear/qmatmul.zig, sdk/blocks/linear/verify_projection.zig, sdk/blocks/attention/proj.zig,
//! sdk/blocks/norms/rms.zig, sdk/blocks/mlp/{activations,dense}.zig, sdk/blocks/embedding.zig, sdk/vision/splice.zig
//! (the qwen4_exp arms: no Hadamard, ternary, row-exact, GGUF, grouped-norm or Gemma activation paths).
const mlx = @import("../../../mlx/ffi.zig");
const sushi_pack = @import("../../../sdk/quant/exl3/sushi_pack.zig");
const decode_attn_quant = @import("../../../sdk/blocks/attention/decode_attn_quant.zig");
const Trunk = @import("../trunk.zig").Trunk;
const lastDim = @import("../../../engines/mlx/load/weights.zig").lastDim;
const transposeBf16Weight = @import("../../../engines/mlx/load/weights.zig").transposeBf16Weight;
const BITS_CACHE_CAP = @import("../../../sdk/quant/params.zig").BITS_CACHE_CAP;
const BitsCache = @import("../../../sdk/quant/params.zig").BitsCache;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const computeQuantParams = @import("../../../sdk/quant/params.zig").computeQuantParams;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const qmv2 = @import("../../../sdk/blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../../sdk/blocks/qmm/qmv_nax2.zig");
pub inline fn qmatmul(self: *const Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    // A linear the pack's trunk format loaded (`trunk/formats.zig`) runs through that format.
    if (self.formats.linearOf(w)) |fw| return self.formats.matmul(fw, x, self.s);
    if (sushi_pack.isRowScaledFp8(w, sc)) return sushi_pack.fp8Linear(x, w, sc, self.s);
    const qp = self.quantParamsHinted(w, sc, lastDim(x));
    // A ternary 2-bit pack (bias == -scale on every weight) reads the 2-bit kernels; they decline other shapes.
    if (self.ternary_2bit) {
        if (try qmv_nax2.qmm(x, w, sc, bi, qp.bits, qp.group_size, self.s)) |y| return y;
        if (try qmv2.qmm(x, w, sc, bi, qp.bits, qp.group_size, true, false, self.s)) |y| return y;
    }
    return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
}
/// The logits projection: a dense bf16 head is `[vocab, hidden]`, projected through a transposed view.
pub inline fn lmHeadProject(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    if (self.lm_head_s.ctx == null) {
        const wt = try transposeBf16Weight(self.lm_head_w, self.s);
        defer _ = mlx.mlx_array_free(wt);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&result, x, wt, self.s));
        return result;
    }
    return self.qmatmul(x, self.lm_head_w, self.lm_head_s, self.lm_head_b);
}
/// Per-weight (bits, group_size, mode), cached by the scales array; `in_dim` pins an override's geometry.
pub inline fn quantParamsHinted(self: *const Trunk, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const key_raw = sc.ctx orelse return .{
        .bits = self.config.quant_bits,
        .group_size = self.config.quant_group_size,
        .mode = self.config.quant_mode,
    };
    const cache = @constCast(&self.bits_cache);
    const start = BitsCache.slot(key_raw);
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const idx = (start + i) & (BITS_CACHE_CAP - 1);
        const entry = cache.keys[idx];
        if (entry == key_raw) {
            return .{
                .bits = cache.vals_bits[idx],
                .group_size = @as(u32, cache.vals_gs_div8[idx]) * 8,
                .mode = @fromBackingInt(@intCast(cache.vals_mode[idx])),
            };
        }
        if (entry == null) {
            const detected = computeQuantParams(.of(&self.config), w, sc, in_dim);
            _ = cache.put(key_raw, detected);
            return detected;
        }
    }
    return computeQuantParams(.of(&self.config), w, sc, in_dim);
}
/// Attention projection with the decode-only requant (`batch == 1 and !is_prefill`: serial decode and verify).
pub inline fn attnProj(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, decode_shape: bool, layer: u32) !mlx.mlx_array {
    if (decode_shape and sc.ctx == null and decode_attn_quant.decodeAttnQuantEnabled()) {
        if (self.attnDqFor(w, layer) catch null) |dq| {
            return qmatmulBits(x, dq.w, dq.s, dq.b, dq.bits, dq.group_size, dq.mode, self.s);
        }
    }
    return self.qmatmul(x, w, sc, bi);
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
