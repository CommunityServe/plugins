//! The turboexl3 side of `MLX_SERVE_DSV41_SHFIN` (src/shfin.zig): F3 on turboexl3's `sharedSwiglu`. The copied model's
//! `shared` asks `fused` first (on, where F3 takes the call: the down's output; else null and the host's chain as before)
//! and, under `check`, hands the chain's output to `check`. F3 takes x [L, hidden] through the EXL3 gate (w1), up (w3)
//! and down (w2) at the decode rows the dense chain takes (`sharedSwigluTakes`), the clamped SwiGLU's limit > 0.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const shfin = @import("shfin.zig");

const A = mlx.mlx_array;

fn takes(x: A, w1: t3.Proj, w3: t3.Proj, w2: t3.Proj, limit: f32, act: mlx.mlx_dtype) !bool {
    if (shfin.mode == .off or !(limit > 0)) return false;
    return t3.sharedSwigluTakes(x, w1, w3, w2, act, .mul1);
}

/// On: the shared expert's output [L, hidden] in `act` on F3 where it takes the call; else null (the caller runs the
/// host's chain). The caller owns the result.
pub fn fused(s: mlx.mlx_stream, x: A, w1: t3.Proj, w3: t3.Proj, w2: t3.Proj, limit: f32, act: mlx.mlx_dtype) !?A {
    if (shfin.mode != .on or !try takes(x, w1, w3, w2, limit, act)) return null;
    const y = try t3.sharedSwiglu(s, x, w1, w3, w2, limit, act, .mul1);
    shfin.served(mlx.mlx_array_size(y));
    return y;
}

/// `check`, outside a traced stretch: where F3 takes the call, its output beside the host chain's (`want`, which the
/// caller serves), compared word for word on the device (one host read), into F3's counts.
pub fn check(s: mlx.mlx_stream, x: A, w1: t3.Proj, w3: t3.Proj, w2: t3.Proj, limit: f32, act: mlx.mlx_dtype, want: A) !void {
    if (shfin.mode != .check or shfin.tracing or !try takes(x, w1, w3, w2, limit, act)) return;
    const got = try t3.sharedSwiglu(s, x, w1, w3, w2, limit, act, .mul1);
    defer _ = mlx.mlx_array_free(got);
    const wt: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(want)) {
        2 => .uint16,
        4 => .uint32,
        else => return error.ShfinCheckWords,
    };
    var tmp: [5]A = undefined;
    for (&tmp) |*a| a.* = mlx.mlx_array_new();
    defer for (tmp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try mlx.check(mlx.mlx_view(&tmp[0], want, wt, s));
    try mlx.check(mlx.mlx_view(&tmp[1], got, wt, s));
    try mlx.check(mlx.mlx_equal(&tmp[2], tmp[0], tmp[1], s));
    try mlx.check(mlx.mlx_astype(&tmp[3], tmp[2], .uint32, s));
    try mlx.check(mlx.mlx_sum(&tmp[4], tmp[3], false, s));
    try mlx.check(mlx.mlx_array_eval(tmp[4]));
    const same = (mlx.mlx_array_data_uint32(tmp[4]) orelse return error.NoData)[0];
    shfin.tally(mlx.mlx_array_size(want), same);
}
