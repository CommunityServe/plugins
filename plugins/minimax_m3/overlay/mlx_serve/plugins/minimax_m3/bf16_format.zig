//! copied_from: src/plugins/minimax_m3/bf16_format.zig@dfd8741b6296d23ebb3c27f172409fcabcb67ed8
const ops = @import("ops.zig");
const bank = @import("bf16_bank.zig");
/// x [rows, 1, in] through the slots `ids` of one BF16 projection [slots, out, in] (MLX's gather_mm).
pub fn gather(g: *ops.G, x: ops.T, projection: bank.Arrays(ops.T), ids: ops.T) !ops.T {
    return g.gatherMatmul(x, projection.w, null, null, ids, 0, 0, null, false);
}
