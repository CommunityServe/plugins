//! `TURBOEXL3_GLM53F_HOST_ZERO=1` (opt-in, exact): DEC-ONE's arrival counters made on the host (turboexl3's
//! `dense.dec_host_zero`). A DEC-ONE call whose prepare runs inside its launch (a decode row, or a linear of at most
//! 1,024 outputs at decode and verify rows) and the joined launches of SMALLOP_FUSE (a DSA layer's q_a and kv_a,
//! glm53f.pair_ks; a dense layer's gate and up, glm53f.shared_gu) zero their counters with a
//! launch of their own (`turboexl3_dense_dec_zero`) before the GEMV. With the switch the counters are a new array of
//! zeros copied from the host a call (`dense.hostCounters`, ROUTER_ONE's counters' way), so the GEMV reads the same
//! zeros and writes the same bytes, one launch fewer a call. src/smallop_gpu_test.zig holds each such call's outputs to
//! the launch-zeroed call's bit for bit at 1 to 4 rows.
const std = @import("std");
const t3 = @import("turboexl3");

/// `TURBOEXL3_GLM53F_HOST_ZERO=1`, set by the plugin at load (`apply`), before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_HOST_ZERO=1: DEC-ONE's arrival counters a new array of zeros copied from the host a call where a launch zeroed them (prepare-inside calls, the joined q_a | kv_a and gate | up launches): the same zeros, one launch fewer a call\n";

/// turboexl3's flag, set for the process before any dense call.
pub fn apply() void {
    t3.dense.dec_host_zero.get().* = true;
}

/// The calls that took host-made counters since the process began (turboexl3's count).
pub fn calls() u64 {
    return t3.dense.dec_host_zeroed.get().*;
}

/// The request's line (the plugin's meters), over `n` calls.
pub fn summary(buf: []u8, n: u64) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_HOST_ZERO calls={d}\n", .{n}) catch buf[0..0];
}

test "turboexl3_glm53f stream host_zero: turboexl3's flag is off until the plugin applies the switch, and the request's line counts the calls" {
    const t = std.testing;
    try t.expect(!t3.dense.dec_host_zero.get().*);
    try t.expect(t3.dense.host_counters_max >= 768);
    var buf: [96]u8 = undefined;
    try t.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_HOST_ZERO calls=168\n", summary(&buf, 168));
}
