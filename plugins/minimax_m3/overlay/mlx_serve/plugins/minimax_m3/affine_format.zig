//! copied_from: src/plugins/minimax_m3/affine_format.zig@90fb57282d374c0989a46503b12f4fd9be300963
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
