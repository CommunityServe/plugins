//! copied_from: src/plugins/minimax_m3/affine_format.zig@b20ebdd16e8e262f5e13f2b24ef33cc4ad09aa1f
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
