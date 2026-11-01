//! copied_from: src/plugins/minimax_m3/affine_format.zig@2d42b6c7e8b4b0b0b1372789c425b966d73a9dbd
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
