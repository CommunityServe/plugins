//! turboexl3_ling's KDA_NORM_PREP (L5, kept, built in: no option turns it on or off): a KDA layer's gated norm at a
//! wide prompt span written straight into o_proj's prepared rows. At prompt rows the copied KDA layer
//! (plugins/glm5_next/kda.zig) runs the host's fused norm-gate (decode_fused `gdnNormGateFusedTab` on its sigmoid
//! table, TAB 1, with the sigmoid gate), whose output o bf16 [B, S, 8192] goes to GPU memory, and then o_proj's `linear`
//! of o, whose first launch reads o back and prepares it for the matmul (in the layer-major pass over LAYER_WHAT's kept
//! W_hat: src/ling_layer_what.zig `linear`). `oProj` runs turboexl3's `ling.kdaNormGatePrep` in their place: one launch,
//! the prepare with the norm-gate's statements where it loads o, so o never reaches GPU memory and the prepared rows are
//! the prepare's words. o_proj's matmul against the kept W_hat and its finish then run from those rows, `linear`'s words.
//! The copied layer calls `oProj` first at its norm-gate site; where it declines, the norm-gate and o_proj run as before.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const layer_what = @import("ling_layer_what.zig");
const ran = @import("ling_ran.zig");
/// The host's sigmoid table, the copied fused_activations.zig's cache: the array its norm-gate reads at TAB 1
const sigmoidTableFor = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;

const kdanorm = t3.ling.kdanorm;

/// The norm-gate's operands as the copied kda.zig hands them to the host's norm-gate: y [B, S, HV, 128] (the
/// recurrence's output), z [B, S, z_stride] (the gate's rows: head h's at z_off + h * 128), norm_w [128], all bf16 (or
/// all f16), and eps f32 0-d
pub const NormGate = struct {
    y: mlx.mlx_array,
    z: mlx.mlx_array,
    z_off: u32,
    z_stride: u32,
    norm_w: mlx.mlx_array,
    eps: mlx.mlx_array,
};

/// KDA_NORM_PREP: o_proj's output y [B, S, out] of the norm-gate of `ng`, what `m.mm(&la.out, o)` returns for the
/// norm-gate's output o; `o` is o_proj's `Linear` (w, s, b, exl3). The caller owns y. The launch writes o_proj's
/// prepared rows (turboexl3's `ling.kdaNormGatePrep`: dense's prepare of the norm-gate's output, word for word), then
/// o_proj's matmul against LAYER_WHAT's kept W_hat and its finish run from them (`layer_what.linearPrepared`), so y is
/// `linear`'s words. Null where it declines, every check before the launch, so nothing is launched, decoded or counted:
/// not a GPU stream, o_proj not EXL3, o_proj not held in the open scope or not on `linear`'s matmul path at the call's
/// rows (`layer_what.preparedDense`: the register path's rows, pg2 on, an un-rotated resident trellis), or operands the
/// launch does not take (`kdanorm.shapeOk` with the table's shape before the table is made, then `kdanorm.takes`). The
/// caller then runs the norm-gate and o_proj. A call that takes it marks KDA_NORM_PREP; the first one prints the
/// engagement line.
pub fn oProj(s: mlx.mlx_stream, ng: NormGate, o: anytype) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !o.exl3) return null;
    const ysh = mlx.getShape(ng.y);
    if (ysh.len != 4 or ysh[0] < 1 or ysh[1] < 1 or ysh[2] < 1 or ysh[3] < 1) return null;
    const rows = @as(usize, @intCast(ysh[0])) * @as(usize, @intCast(ysh[1]));
    const in = @as(u32, @intCast(ysh[2])) * @as(u32, @intCast(ysh[3]));
    // the launch's dtypes, which are the table's too (`sigmoidTableFor` makes bf16 and f16 tables)
    const dt = mlx.mlx_array_dtype(ng.y);
    if (dt != .bfloat16 and dt != .float16) return null;
    const p: t3.Proj = .{ .trellis = o.w, .suh = o.s, .svh = o.b };
    const d = layer_what.preparedDense(p, rows, in, dt) orelse return null;
    if (!operandsOk(ng, dt, p.suh)) return null;
    const op: kdanorm.Op = .{ .y = ng.y, .z = ng.z, .z_off = ng.z_off, .z_stride = ng.z_stride, .norm_w = ng.norm_w, .eps = ng.eps, .sigtab = try sigmoidTableFor(s, dt), .suh = p.suh };
    if (!kdanorm.takes(d, op)) return null;
    const xp = t3.ling.kdaNormGatePrep(s, d, op) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    defer _ = mlx.mlx_array_free(xp);
    const out_sh = [_]c_int{ ysh[0], ysh[1], @intCast(d.shape.out) };
    const y = try layer_what.linearPrepared(s, d, xp, p, rows, &out_sh);
    ran.mark(.KDA_NORM_PREP);
    if (!said) {
        said = true;
        std.log.info("[turboexl3_ling] KDA_NORM_PREP engaged: S={d}", .{rows});
    }
    return y;
}

/// `kdanorm.shapeOk` of the call's operands with the table as `sigmoidTableFor` makes it (65,536 words in y's dtype),
/// so a call the launch refuses makes no table
fn operandsOk(ng: NormGate, dt: mlx.mlx_dtype, suh: mlx.mlx_array) bool {
    const tab_sh = [_]c_int{1 << 16};
    return kdanorm.shapeOk(.{
        .y = .{ .dt = dt, .sh = mlx.getShape(ng.y) },
        .z = .{ .dt = mlx.mlx_array_dtype(ng.z), .sh = mlx.getShape(ng.z) },
        .z_off = ng.z_off,
        .z_stride = ng.z_stride,
        .norm_w = .{ .dt = mlx.mlx_array_dtype(ng.norm_w), .sh = mlx.getShape(ng.norm_w) },
        .eps = .{ .dt = mlx.mlx_array_dtype(ng.eps), .sh = mlx.getShape(ng.eps) },
        .sigtab = .{ .dt = dt, .sh = &tab_sh },
        .suh = .{ .dt = mlx.mlx_array_dtype(suh), .sh = mlx.getShape(suh) },
    }) != null;
}

/// KDA_NORM_PREP's engagement line, once (the run-path names mark every call)
var said = false;
