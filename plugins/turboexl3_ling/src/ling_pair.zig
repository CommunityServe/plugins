//! turboexl3_ling's same-input EXL3 pairs in one launch (PAIR; the same bytes): a KDA layer's z | a, the shared
//! expert's gate | up and the dense FFN's gate | up run at decode rows through turboexl3's `ling.linearPair` (each member
//! in its own buffers, one GEMV launch in the DEC-ONE form each member's own `linear` takes), each member's output a
//! column view of the pair's. The copied KDA layer (kda.zig), MoE block (moe.zig `sharedDown`) and dense FFN (trunk.zig
//! `denseMlp`) call `pair`; where the pair declines, they run each projection as before. turboexl3's
//! models/ling/ling_test.zig holds each member to its own `linear` words.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
/// The run-path line (src/ling_ran.zig): PAIR marks where its launch ran.
const ran = @import("ling_ran.zig");

/// `a(x)` and `b(x)` in one launch for two EXL3 projections of one input (`a`, `b`: the copied `Linear`s, `w` the
/// trellis, `s` its suh, `b` its svh), x [..., in]; null where either is not EXL3, or turboexl3's pair declines (past the
/// decode rows, members of two K splits): the caller then runs each. Owned results: a's output [..., out_a] and b's
/// [..., out_b], column views of the pair's.
pub fn pair(s: mlx.mlx_stream, x: mlx.mlx_array, a: anytype, b: anytype) !?[2]mlx.mlx_array {
    if (!a.exl3 or !b.exl3) return null;
    const pa: t3.Proj = .{ .trellis = a.w, .suh = a.s, .svh = a.b };
    const pb: t3.Proj = .{ .trellis = b.w, .suh = b.s, .svh = b.b };
    const y2 = t3.ling.linearPair(s, x, pa, pb, .mul1) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    defer _ = mlx.mlx_array_free(y2);
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    const oa: c_int = mlx.getShape(a.w)[1] * 16;
    const ob: c_int = mlx.getShape(b.w)[1] * 16;
    var shape: [8]c_int = undefined;
    for (xsh, 0..) |d, i| shape[i] = d;
    const nd = xsh.len;
    shape[nd - 1] = oa + ob;
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, nd, s));
    var lo: [8]c_int = @splat(0);
    var hi: [8]c_int = undefined;
    for (shape[0..nd], 0..) |d, i| hi[i] = d;
    const st: [8]c_int = @splat(1);
    var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    hi[nd - 1] = oa;
    try mlx.check(mlx.mlx_slice(&out[0], y, &lo, nd, &hi, nd, &st, nd, s));
    lo[nd - 1] = oa;
    hi[nd - 1] = oa + ob;
    try mlx.check(mlx.mlx_slice(&out[1], y, &lo, nd, &hi, nd, &st, nd, s));
    ran.mark(.PAIR);
    return out;
}
