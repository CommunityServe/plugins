//! turboexl3_ling's KDA projections on turboexl3's fused launches, each member's output its own `linear`'s bytes:
//! - `proj` (KDA_PROJ_FUSED): a KDA layer's q|k|v, its forget gate (`a`, f_proj) and its output gate (`z`, g_proj) of
//!   one input at decode rows (1 to 16) through turboexl3's `ling.kdaProj`: one prepare launch, then one DEC-ONE GEMV
//!   launch, each member in its own buffers at its own K split.
//! - `projB` (KDA_BPROJ): that launch with b_proj (a plain bf16 matrix) as a fourth member (turboexl3's
//!   `ling.kdaProjB`): its threadgroups run MLX's gemv over each row, the kernel MLX's matmul runs for b's one-row
//!   product, so b's output is serial decode's matmul's at every row, in place of b_proj's own matmul (one row) or the
//!   exact verify's denseRows (a concatenate and a batched matmul). turboexl3's models/ling/bproj_test.zig holds b and
//!   the three members to the served launches word for word at rows 1 to 8.
//! - `projPrompt` (KDA_PROJ_PROMPT, P4): the same three projections at prompt rows, their products and finishes in one
//!   launch past each member's own prepare and W_hat (turboexl3's `ling.kdaProjPrompt`).
//! - `projPromptGate` (KDA_GATE_EPI, S52): `projPrompt`'s launch with f's output its bounded gate, made in the launch's
//!   store loop (turboexl3's `ling.kdaProjPromptGate`, form G0), so the served gate's own launch does not run.
//! The copied KDA layer (plugins/glm5_next/kda.zig) calls `projB`, then `proj` where it declines, in serial decode, in a
//! verify and in an exact verify's rows, and `projPromptGate`, then `projPrompt` where it declines, outside an exact
//! verify's rows; where turboexl3 declines them all (a projection not EXL3, rows none of them takes), it runs each
//! projection as before.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
/// the exact verify's flag (`exact_rows`): b_proj's rows run as one-row matmuls only then
const ling_exl3 = @import("ling_exl3.zig");
/// LAYER_WHAT (src/ling_layer_what.zig): a layer-major layer's wide spans over the W_hats of its first (`projPrompt`)
const layer_what = @import("ling_layer_what.zig");
/// The run-path line (src/ling_ran.zig): KDA_PROJ_FUSED, KDA_BPROJ and KDA_PROJ_PROMPT mark where they run.
const ran = @import("ling_ran.zig");

/// `qkv(x)`, `f(x)` and `g(x)` in one launch for a KDA layer's three EXL3 projections of one input (the copied
/// `Linear`s: `w` the trellis, `s` its suh, `b` its svh), x [..., in]; null where any of them is not EXL3, or
/// turboexl3 declines: the caller then runs each. Owned results: each member's output [..., out_m].
pub fn proj(s: mlx.mlx_stream, x: mlx.mlx_array, qkv: anytype, f: anytype, g: anytype) !?[3]mlx.mlx_array {
    if (!qkv.exl3 or !f.exl3 or !g.exl3) return null;
    const p = [3]t3.Proj{
        .{ .trellis = qkv.w, .suh = qkv.s, .svh = qkv.b },
        .{ .trellis = f.w, .suh = f.s, .svh = f.b },
        .{ .trellis = g.w, .suh = g.s, .svh = g.b },
    };
    const ys = t3.ling.kdaProj(s, x, p, .mul1) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    const owned = ys;
    defer for (owned) |y| if (y.ctx != null) {
        _ = mlx.mlx_array_free(y);
    };
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    var shape: [8]c_int = undefined;
    for (xsh, 0..) |d, i| shape[i] = d;
    const nd = xsh.len;
    var out: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (&out, owned) |*o, y| {
        shape[nd - 1] = mlx.getShape(y)[1];
        try mlx.check(mlx.mlx_reshape(o, y, &shape, nd, s));
    }
    ran.mark(.KDA_PROJ_FUSED);
    return out;
}

/// `proj` with the layer's b_proj (a plain bf16 `Linear`: `w` the loaded [in, out] column-major view) as a fourth
/// member of the GEMV launch: the three outputs as `proj` returns them, b's [..., out_b] into `b_out` (owned); null
/// where a projection is not as turboexl3's `ling.kdaProjB` takes it, or turboexl3 declines: the caller then runs `proj`
/// and b's matmul apart, nothing launched.
pub fn projB(s: mlx.mlx_stream, x: mlx.mlx_array, qkv: anytype, f: anytype, g: anytype, b: anytype, b_out: *mlx.mlx_array) !?[3]mlx.mlx_array {
    if (!qkv.exl3 or !f.exl3 or !g.exl3) return null;
    if (b.exl3 or b.s.ctx != null or b.w.ctx == null or mlx.mlx_array_dtype(b.w) != .bfloat16 or mlx.getShape(b.w).len != 2) return null;
    // the member is serial decode's one-row matmul at every row: at one row, and at an exact verify's rows (denseRows'
    // per-row gemv); several rows outside an exact verify take MLX's GEMM, which it is not
    const xsh0 = mlx.getShape(x);
    if (xsh0.len >= 2 and xsh0[xsh0.len - 2] > 1 and !ling_exl3.exact_rows) return null;
    // b's weight as the member reads it: the loaded [in, out] view transposed back, [out, in] row-major (no copy)
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_transpose(&wb, b.w, s));
    const p = [3]t3.Proj{
        .{ .trellis = qkv.w, .suh = qkv.s, .svh = qkv.b },
        .{ .trellis = f.w, .suh = f.s, .svh = f.b },
        .{ .trellis = g.w, .suh = g.s, .svh = g.b },
    };
    const ys = t3.ling.kdaProjB(s, x, p, wb, .mul1) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    const owned = ys;
    defer for (owned) |y| if (y.ctx != null) {
        _ = mlx.mlx_array_free(y);
    };
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return error.BadExl3Shape;
    var shape: [8]c_int = undefined;
    for (xsh, 0..) |d, i| shape[i] = d;
    const nd = xsh.len;
    var out: [4]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (&out, owned) |*o, y| {
        shape[nd - 1] = mlx.getShape(y)[1];
        try mlx.check(mlx.mlx_reshape(o, y, &shape, nd, s));
    }
    ran.mark(.KDA_BPROJ);
    b_out.* = out[3];
    return .{ out[0], out[1], out[2] };
}

/// P4: q|k|v, f and g of x at prompt rows, each member's prepare and W_hat by its own `linear`'s launches, then their
/// three products and finishes in one launch (turboexl3's `ling.kdaProjPrompt`: pg2's fused NAX GEMM with the dense
/// finish over the members), each member's output its `linear`'s bytes. Null where a projection is not EXL3, or
/// turboexl3 declines (decode rows, off NAX, a shape MLX runs another kernel at): the caller then runs each projection.
/// Owned results: each member's [..., out_m]. Under LAYER_WHAT (src/ling_layer_what.zig), a layer-major layer's wide
/// span runs the launch over the layer's kept W_hats of the three (turboexl3's `ling.kdaProjPromptW`), the same words.
pub fn projPrompt(s: mlx.mlx_stream, x: mlx.mlx_array, qkv: anytype, f: anytype, g: anytype) !?[3]mlx.mlx_array {
    if (!qkv.exl3 or !f.exl3 or !g.exl3) return null;
    const p = [3]t3.Proj{
        .{ .trellis = qkv.w, .suh = qkv.s, .svh = qkv.b },
        .{ .trellis = f.w, .suh = f.s, .svh = f.b },
        .{ .trellis = g.w, .suh = g.s, .svh = g.b },
    };
    // LAYER_WHAT: over the layer's kept W_hats where its scope serves the call, else the per-call W_hats
    const ys = blk: {
        if (try layer_what.kdaProjPrompt(s, x, p)) |kept| break :blk kept;
        break :blk t3.ling.kdaProjPrompt(s, x, p, .mul1) catch |e| switch (e) {
            error.Exl3EngineNotImplemented => return null,
            else => return e,
        };
    };
    ran.mark(.KDA_PROJ_PROMPT);
    return ys;
}

/// KDA_GATE_EPI's gate operands: the layer's A_log and dt_bias as the loader keeps them (f32 [N1], N1 f's width: one
/// value a key channel) and the config's kda_gate_lower_bound
pub const GateIn = struct { A_log: mlx.mlx_array, dt_bias: mlx.mlx_array, bound: f32 };

/// KDA_GATE_EPI (S52; the copied kda.zig checks the layer's own conditions before it calls this): `projPrompt`'s launch
/// with f's output its bounded gate g, f32 [..., N1], made in the launch's store loop (turboexl3's
/// `ling.kdaProjPromptGate`, form G0: exp(A_log) for each element, as the served gate takes it), so the served gate's
/// own launch (sdk/blocks/kda/ling.zig `gate`) does not run. q|k|v and z are `projPrompt`'s words and g the served
/// gate's of `projPrompt`'s f output. Under LAYER_WHAT, a layer-major layer's wide span runs it over the layer's kept
/// W_hats (turboexl3's `ling.kdaProjPromptWGate`), the same words. Owned results: q|k|v, g and z. Null, nothing
/// launched, where a projection is not EXL3, turboexl3 refuses the gate's operands (`kdaproj_pf.gateTakes`: A_log and
/// dt_bias f32 [N1], the bound finite and not 0), or it declines the call as `projPrompt`'s (decode rows, the register
/// path's 32, off NAX): the caller then runs `projPrompt` and the served gate. A call that takes it marks
/// KDA_PROJ_PROMPT (P4's launch ran) and KDA_GATE_EPI; the first one prints the engagement line.
pub fn projPromptGate(s: mlx.mlx_stream, x: mlx.mlx_array, qkv: anytype, f: anytype, g: anytype, gate: GateIn) !?[3]mlx.mlx_array {
    if (!qkv.exl3 or !f.exl3 or !g.exl3) return null;
    const p = [3]t3.Proj{
        .{ .trellis = qkv.w, .suh = qkv.s, .svh = qkv.b },
        .{ .trellis = f.w, .suh = f.s, .svh = f.b },
        .{ .trellis = g.w, .suh = g.s, .svh = g.b },
    };
    const tg: t3.ling.kdaproj_pf.Gate = .{ .form = .g0, .a = gate.A_log, .dt_bias = gate.dt_bias, .bound = gate.bound };
    // the gate's operands first, so a refused gate leaves LAYER_WHAT's scope as it was (nothing decoded or counted)
    if (!t3.ling.kdaproj_pf.gateTakes(p, tg)) return null;
    // LAYER_WHAT: over the layer's kept W_hats where its scope serves the call, else the per-call W_hats
    const ys = blk: {
        if (try layer_what.kdaProjPromptGate(s, x, p, tg)) |kept| break :blk kept;
        break :blk t3.ling.kdaProjPromptGate(s, x, p, tg, .mul1) catch |e| switch (e) {
            error.Exl3EngineNotImplemented => return null,
            else => return e,
        };
    };
    ran.mark(.KDA_PROJ_PROMPT);
    ran.mark(.KDA_GATE_EPI);
    if (!gate_said) {
        gate_said = true;
        const xsh = mlx.getShape(x);
        var rows: usize = 1;
        for (xsh[0 .. xsh.len - 1]) |d| rows *= @intCast(d);
        std.log.info("[turboexl3_ling] KDA_GATE_EPI engaged: S={d}", .{rows});
    }
    return ys;
}

/// KDA_GATE_EPI's engagement line, once (the run-path names mark every call)
var gate_said = false;
