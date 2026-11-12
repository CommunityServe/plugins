//! turboexl3_ling: a causal prompt call's MLA attention in spans of `span_rows` query rows, each span one MLX SDPA call
//! over the keys up to its own last row (MLA_SPANS, from mla.zig's `spanAttn`, where the flash launch does not take the
//! call), so the scores of keys that every row of a span masks are never made. Each row gets `spanAttn`'s words. MLX
//! composes this attention, because a query head of 192 with a value head of 128 has no fused prompt kernel
//! (mlx/fast.cpp): the scores q x k^T, the causal mask (the dtype's finite minimum), a softmax in f32, then P x v.
//! - Both products give a row the same words over fewer keys. Each output word sums its K in one fixed order (32-wide
//!   steps on the NAX GEMM, 8-wide on the others, whatever the block size), and the keys a span leaves out are keys its
//!   rows mask, where P is exactly 0.
//! - The softmax gives a row the same words over fewer keys within one of MLX's two softmax kernels
//!   (mlx/backend/metal/softmax.cpp: one pass for a row of at most 4096 keys, a loop for a longer row). A masked or
//!   missing key adds 0 to the same thread's sum. Across the two kernels the words can differ, so a span whose call's
//!   rows take the loop extends until its own rows take the loop too (`spanEnd`).
//! - MLX runs a call of fewer than `min_rows` query rows on other kernels (gemv_wide for the scores at 2-15 rows, the
//!   fused vector attention at 1-8 rows), so no span is that short: a short tail joins the span before it.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;

/// Query rows a span. The last span of a call can be up to `min_rows - 1` rows longer.
pub const span_rows: c_int = 512;
/// MLX's softmax kernel boundary (softmax.cpp `SOFTMAX_LOOPED_LIMIT`): a row of more keys than this takes the loop.
pub const softmax_one_pass_keys: c_int = 4096;
/// The fewest query rows MLX composes on its GEMMs (fewer: gemv_wide for the scores, and the vector attention at 1-8).
pub const min_rows: c_int = 16;

/// The spans the last call `attend` split took
pub var last_spans: u32 = 0;

/// The end of `spanAttn`'s span that starts at `c0` in a call of `seq_len` rows: the whole call up to `ctrl_rows` rows,
/// else `ctrl_rows` rows each.
fn ctrlEnd(c0: c_int, seq_len: c_int, ctrl_rows: c_int) c_int {
    return if (seq_len <= ctrl_rows) seq_len else @min(c0 + ctrl_rows, seq_len);
}

/// The end of the span that starts at query row `r0` of `spanAttn`'s span ending at row `c1`, with `first` keys before
/// the call's first row: `span_rows` rows on, or further, so that the span's rows take the softmax kernel that
/// `spanAttn`'s rows take (keys `first + c1`) and no tail shorter than `min_rows` rows is left.
pub fn spanEnd(first: c_int, r0: c_int, c1: c_int) c_int {
    var r1 = @min(r0 + span_rows, c1);
    if (first + c1 > softmax_one_pass_keys) r1 = @max(r1, @min(c1, softmax_one_pass_keys + 1 - first));
    if (c1 - r1 < min_rows) r1 = c1;
    return r1;
}

/// How many spans `attend` makes of a call of `seq_len` rows after `first` keys, and how many `spanAttn` makes
pub const Count = struct { spans: u32, ctrl: u32 };
pub fn count(first: c_int, seq_len: c_int, ctrl_rows: c_int) Count {
    var n: Count = .{ .spans = 0, .ctrl = 0 };
    var c0: c_int = 0;
    while (c0 < seq_len) : (n.ctrl += 1) {
        const c1 = ctrlEnd(c0, seq_len, ctrl_rows);
        var r0 = c0;
        while (r0 < c1) : (n.spans += 1) r0 = spanEnd(first, r0, c1);
        c0 = c1;
    }
    return n;
}

/// `out` = `spanAttn`'s attention (MLX SDPA, causal) of `queries` `[B, H, S, D]` over `keys` and `values`
/// (`[B, Hk, T, *]`, the S = `seq_len` query rows the last S of the T), each of `spanAttn`'s spans (`ctrl_rows`) in
/// this file's spans. False, with nothing made, when it would make no more calls than `spanAttn` makes.
pub fn attend(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, scale: f32, seq_len: c_int, ctrl_rows: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const ks = mlx.getShape(keys);
    const vs = mlx.getShape(values);
    const first = ks[2] - seq_len;
    const n = count(first, seq_len, ctrl_rows);
    if (n.spans <= n.ctrl) return false;
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    const st4 = [_]c_int{ 1, 1, 1, 1 };
    var c0: c_int = 0;
    while (c0 < seq_len) {
        const c1 = ctrlEnd(c0, seq_len, ctrl_rows);
        var r0 = c0;
        while (r0 < c1) {
            const r1 = spanEnd(first, r0, c1);
            var q = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q);
            try mlx.check(mlx.mlx_slice(&q, queries, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ qs[0], qs[1], r1, qs[3] }, 4, &st4, 4, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_slice(&k, keys, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], first + r1, ks[3] }, 4, &st4, 4, s));
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_slice(&v, values, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ vs[0], vs[1], first + r1, vs[3] }, 4, &st4, 4, s));
            var o = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "causal", none_mask, .{ .ctx = null }, false, s));
            _ = mlx.mlx_vector_array_append_value(parts, o);
            r0 = r1;
        }
        c0 = c1;
    }
    try mlx.check(mlx.mlx_concatenate_axis(out, parts, 2, s));
    last_spans = n.spans;
    return true;
}

test "attn_spans: spans of 512 rows; a span whose call's rows take MLX's softmax loop takes it too; no tail under 16 rows" {
    const t = std.testing;
    // a first chunk (0 keys before): 8 spans of 512, all in the one-pass softmax as the call
    try t.expectEqual(Count{ .spans = 8, .ctrl = 1 }, count(0, 4096, 4096));
    try t.expectEqual(@as(c_int, 512), spanEnd(0, 0, 4096));
    // a later chunk (4096 keys before): every row takes the loop at any span, 8 spans
    try t.expectEqual(Count{ .spans = 8, .ctrl = 1 }, count(4096, 4096, 4096));
    // a warm chunk (3000 keys before, the call's rows 7096 keys: the loop): the first span runs to 1097 rows (4097 keys)
    try t.expectEqual(@as(c_int, 1097), spanEnd(3000, 0, 4096));
    try t.expectEqual(@as(c_int, 1609), spanEnd(3000, 1097, 4096));
    // a short tail joins the span before it: 600 rows = 512 + 88; 520 rows = one span (an 8-row tail)
    try t.expectEqual(Count{ .spans = 2, .ctrl = 1 }, count(0, 600, 4096));
    try t.expectEqual(Count{ .spans = 1, .ctrl = 1 }, count(0, 520, 4096));
    try t.expectEqual(@as(c_int, 527), spanEnd(0, 0, 527));
    try t.expectEqual(@as(c_int, 512), spanEnd(0, 0, 528));
    // calls too short to split, and the backoff span, take spanAttn's one call
    try t.expectEqual(Count{ .spans = 1, .ctrl = 1 }, count(18349, 31, 4096));
    try t.expectEqual(Count{ .spans = 1, .ctrl = 1 }, count(0, 512, 4096));
    // a layer-major call (8192 rows): spanAttn's two 4096-row spans, each in 8
    try t.expectEqual(Count{ .spans = 16, .ctrl = 2 }, count(0, 8192, 4096));
    // the standard prompt's last chunk (1996 rows after 16384 keys): 512, 512, 512, 460
    try t.expectEqual(Count{ .spans = 4, .ctrl = 1 }, count(16384, 1996, 4096));
}
