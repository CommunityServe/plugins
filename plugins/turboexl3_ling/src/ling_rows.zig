//! turboexl3_ling: an exact verify's rows of a dense matmul in one launch, each row with serial decode's bytes (lever 8).
//! Serial decode runs a dense projection `x @ w` (`w` [in, out] as loaded: the router in f32, the plain bf16 ones) at
//! one row on MLX's gemv; a verify's S rows in one matmul take MLX's GEMM, another accumulation order. As a batch of S
//! one-row matmuls they run that same gemv, S times in one launch:
//! - `w` as [1, in, out]: `matmul` flattens a batched `x` into M rows when `w` has two dimensions;
//! - `x` as [S, 1, in] at a batch stride of two rows (each row stored twice, the first viewed): Matmul's evaluation
//!   collapses a batch whose stride is one row into M rows;
//! - `x` already in the product's dtype: `matmul`'s own astype would hand it a contiguous copy.
//! src/ling_rows_test.zig holds every row to the one-row matmul's bytes at Ling's dense shapes.
const mlx = @import("mlx_serve").mlx;

/// The dtype `x @ w` computes in (MLX's promotion over bf16, f16 and f32): the shared one, else f32.
fn promoted(a: mlx.mlx_dtype, b: mlx.mlx_dtype) mlx.mlx_dtype {
    return if (a == b) a else .float32;
}

/// `x @ w` for `x` [..., in] (every leading row its own one-row gemv) and `w` [in, out] -> [..., out].
pub fn denseRows(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    const ws = mlx.getShape(w);
    if (nd < 2 or nd > 8 or ws.len != 2 or ws[0] != sh[nd - 1]) return error.LingDenseRowsShape;
    const k = sh[nd - 1];
    const n = ws[1];
    var rows: c_int = 1;
    for (sh[0 .. nd - 1]) |d| rows *= d;
    var tmp: [6]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (tmp) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    tmp[0] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&tmp[0], x, &[_]c_int{ rows, 1, k }, 3, s));
    var x3 = tmp[0];
    const dt = promoted(mlx.mlx_array_dtype(x), mlx.mlx_array_dtype(w));
    if (mlx.mlx_array_dtype(x3) != dt) {
        tmp[1] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&tmp[1], x3, dt, s));
        x3 = tmp[1];
    }
    {
        const parts = [_]mlx.mlx_array{ x3, x3 };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        tmp[2] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&tmp[2], vec, 1, s));
    }
    tmp[3] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&tmp[3], tmp[2], &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ rows, 1, k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    tmp[4] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&tmp[4], w, 0, s));
    tmp[5] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&tmp[5], tmp[3], tmp[4], s));
    var osh: [8]c_int = undefined;
    @memcpy(osh[0 .. nd - 1], sh[0 .. nd - 1]);
    osh[nd - 1] = n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, tmp[5], osh[0..nd].ptr, nd, s));
    return out;
}
