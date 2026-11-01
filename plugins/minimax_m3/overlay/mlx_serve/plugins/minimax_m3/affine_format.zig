//! copied_from: src/plugins/minimax_m3/affine_format.zig@dfd8741b6296d23ebb3c27f172409fcabcb67ed8
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
