//! turboexl3_ling: the latent cache's MLA attention at decode and verify rows, composed in MLX: the copied mla.zig's
//! `latentAttn` runs it at the MTP head's decode-width calls (MTP_HEAD_LATENT, src/ling_mtp_latent.zig) where
//! turboexl3's fused launches (src/ling_mla_fused.zig) do not take the call, and the GPU test holds those launches to
//! it. q_nope goes through kv_b's key half into the latent (q_lat . latent = q_nope . k_nope). The scores are over the
//! latent and the shared rope key, the latent is the value, then kv_b's value half is applied to the output. It is f32
//! from the fold to the value half (the ops mla.zig's latent decode had, moved here unchanged).
//! kv_b comes as the loader holds it. The loader holds a dense projection as the `[in, out]` view that a plain matmul
//! reads (trunk.zig `proj`, `transposed`), so kv_b is `[kv_lora, H (nope + v)]`. The fold and turboexl3's launches read
//! its stored rows `[H (nope + v), kv_lora]`, which `kvRows` gives back. The GPU test (src/ling_mla_latent_test.zig)
//! builds kv_b the loader's way.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const ling_mtp_latent = @import("ling_mtp_latent.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// One MLA layer's widths.
pub const Geometry = struct { heads: c_int, nope: c_int, v: c_int, kv_lora: c_int };

/// kv_b's stored rows `[H (nope + v), kv_lora]` (owned), from `kv_b` as the loader holds it: the transpose of the
/// loader's `[kv_lora, H (nope + v)]` view, or kv_b itself where it already has the rows' shape. Any other shape is
/// error.LingLatentKvBShape. The transpose is a view. The GPU test prints whether it reads the loader's own buffer
/// row-contiguous, which is what turboexl3's launches take without a copy.
pub fn kvRows(kv_b: Array, g: Geometry, s: Stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    switch (ling_mtp_latent.kvbOrient(mlx.getShape(kv_b), g.heads * (g.nope + g.v), g.kv_lora) orelse return error.LingLatentKvBShape) {
        .rows => try mlx.check(mlx.mlx_array_set(&r, kv_b)),
        .loader_view => try mlx.check(mlx.mlx_transpose(&r, kv_b, s)),
    }
    return r;
}

/// `out` (the caller's handle) = the attention `[B, H, S, v]` in q_nope's dtype of the call's rows `q_nope`
/// `[B, H, S, nope]` and `q_rot` `[B, H, S, rope]` (roped). It runs over the latent cache view `lat` `[B, 1, T, kv_lora]`
/// and `rot` `[B, 1, T, rope]`, after this call's rows are appended: the call's rows are the last S of the T, and each
/// row attends to the keys up to its own. `kv_b` is as the loader holds it (`kvRows`).
pub fn decode(out: *Array, q_nope: Array, q_rot: Array, lat: Array, rot: Array, kv_b: Array, g: Geometry, scale: f32, s: Stream) !void {
    const h_count = g.heads;
    const nope = g.nope;
    const v_dim = g.v;
    const kv_lora = g.kv_lora;
    const seq_len: c_int = mlx.getShape(q_nope)[2];
    const total: c_int = mlx.getShape(lat)[2];
    const strides3 = [_]c_int{ 1, 1, 1 };
    const w = try kvRows(kv_b, g, s);
    defer _ = mlx.mlx_array_free(w);
    var w3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w3);
    try mlx.check(mlx.mlx_reshape(&w3, w, &[_]c_int{ h_count, nope + v_dim, kv_lora }, 3, s));
    var wk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wk);
    try mlx.check(mlx.mlx_slice(&wk, w3, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ h_count, nope, kv_lora }, 3, &strides3, 3, s));
    var wv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wv);
    try mlx.check(mlx.mlx_slice(&wv, w3, &[_]c_int{ 0, nope, 0 }, 3, &[_]c_int{ h_count, nope + v_dim, kv_lora }, 3, &strides3, 3, s));
    var wv_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wv_t);
    try mlx.check(mlx.mlx_transpose_axes(&wv_t, wv, &[_]c_int{ 0, 2, 1 }, 3, s));
    // In f32 from the fold to the value half: the scores [B, H, S, kv_lora] . [B, 1, kv_lora, T] + the rope half, scaled.
    const qnf = try toF32(q_nope, s);
    defer _ = mlx.mlx_array_free(qnf);
    const wkf = try toF32(wk, s);
    defer _ = mlx.mlx_array_free(wkf);
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_matmul(&qf, qnf, wkf, s));
    const qrf = try toF32(q_rot, s);
    defer _ = mlx.mlx_array_free(qrf);
    const lf = try toF32(lat, s);
    defer _ = mlx.mlx_array_free(lf);
    const rf = try toF32(rot, s);
    defer _ = mlx.mlx_array_free(rf);
    var lf_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lf_t);
    try mlx.check(mlx.mlx_swapaxes(&lf_t, lf, -1, -2, s));
    var rf_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf_t);
    try mlx.check(mlx.mlx_swapaxes(&rf_t, rf, -1, -2, s));
    var s_lat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s_lat);
    try mlx.check(mlx.mlx_matmul(&s_lat, qf, lf_t, s));
    var s_rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s_rot);
    try mlx.check(mlx.mlx_matmul(&s_rot, qrf, rf_t, s));
    var s_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s_sum);
    try mlx.check(mlx.mlx_add(&s_sum, s_lat, s_rot, s));
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_multiply(&scores, s_sum, scale_arr, s));
    // A verify's rows see the keys up to their own position (the last S keys are theirs).
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (seq_len > 1) {
        var qp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qp);
        try mlx.check(mlx.mlx_arange(&qp, @floatFromInt(total - seq_len), @floatFromInt(total), 1, .int32, s));
        var qp2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qp2);
        try mlx.check(mlx.mlx_reshape(&qp2, qp, &[_]c_int{ seq_len, 1 }, 2, s));
        var kp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kp);
        try mlx.check(mlx.mlx_arange(&kp, 0, @floatFromInt(total), 1, .int32, s));
        var future = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(future);
        try mlx.check(mlx.mlx_greater(&future, kp, qp2, s));
        const neg = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_where(&masked, future, neg, scores, s));
    } else {
        _ = mlx.mlx_array_set(&masked, scores);
    }
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, masked, -1, true, s));
    var o_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_f);
    try mlx.check(mlx.mlx_matmul(&o_f, probs, lf, s));
    const wvf = try toF32(wv_t, s);
    defer _ = mlx.mlx_array_free(wvf);
    var o_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_v);
    try mlx.check(mlx.mlx_matmul(&o_v, o_f, wvf, s));
    try mlx.check(mlx.mlx_astype(out, o_v, mlx.mlx_array_dtype(q_nope), s));
}

/// `a` as f32.
fn toF32(a: Array, s: Stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_astype(&r, a, .float32, s));
    return r;
}
