//! This plugin's own, in place of mlx-serve's `src/engines/mlx/load/exl3_bank.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): a loaded `MoeMlpWeights` checked against its config through turboexl3.

const turboexl3 = @import("turboexl3");
const mlx = @import("../../../mlx/ffi.zig");
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;

/// An EXL3 bank's suh/svh bind optionally beside the affine names, so their absence and a trellis turboexl3 cannot
/// decode at the config's rate refuse here. `config` is the copies' `ModelConfig` or the direct model's config.
pub fn checkExl3Bank(mw: *const MoeMlpWeights, config: anytype, spec: turboexl3.Spec) !void {
    const h = config.hidden_size;
    const inter = config.moe_intermediate_size;
    const projs = [_]struct { p: turboexl3.Proj, in: u32, out: u32 }{
        .{ .p = .{ .trellis = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b }, .in = h, .out = inter },
        .{ .p = .{ .trellis = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b }, .in = h, .out = inter },
        .{ .p = .{ .trellis = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b }, .in = inter, .out = h },
    };
    for (projs) |p| {
        if (p.p.suh.ctx == null or p.p.svh.ctx == null) return error.MissingWeight;
        if (!turboexl3.trellisAdmitted(mlx.getShape(p.p.trellis), config.num_experts, p.in, p.out, spec.rate)) return error.Exl3TrellisGeometry;
    }
}
