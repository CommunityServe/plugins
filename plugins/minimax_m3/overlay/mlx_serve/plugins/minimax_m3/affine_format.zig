//! copied_from: src/plugins/minimax_m3/affine_format.zig@9df2c52b504cce5bbf22f074dab168fd8584f961
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
