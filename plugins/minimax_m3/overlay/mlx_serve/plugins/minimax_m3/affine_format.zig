//! copied_from: src/plugins/minimax_m3/affine_format.zig@61ec76d7bc1abfb7f588db3bf6a94ffc0cd423c7
const ops = @import("ops.zig");
const bank = @import("affine_bank.zig");
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays, ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, projection.scales, projection.biases, ids, bank.bits, bank.group_size, .affine, false);
}
