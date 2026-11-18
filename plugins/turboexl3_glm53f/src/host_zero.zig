//! `TURBOEXL3_GLM53F_HOST_ZERO=1` (on by default, exact): DEC-ONE's arrival counters made on the host (turboexl3's
//! `dense.dec_host_zero`). A DEC-ONE call whose prepare runs inside its launch (a trunk linear at a decode row, or one of
//! at most 1,024 outputs at decode and verify rows, `matmul`'s `t3.linear` under TURBOEXL3_GLM53F_DEC) zeroes its
//! counters with a launch of its own (`turboexl3_dense_dec_zero`) before the GEMV. With the switch the counters are a new
//! array of zeros copied from the host a call (`dense.hostCounters`: no two calls share one, so no reset and no order
//! between calls), so the GEMV reads the same zeros and writes the same bytes, one launch fewer a call. A call past
//! `dense.host_counters_max` counters launches the zeroing as before.
//! copied_from: CommunityServe plugins/turboexl3_glm53f/src/host_zero.zig@ac87fe1 (its request summary line left out:
//! the bank's `meters` print the stream's counters only).
const std = @import("std");
const t3 = @import("turboexl3");

/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_HOST_ZERO=1: DEC-ONE's arrival counters a new array of zeros copied from the host a call where a launch zeroed them (prepare-inside calls): the same zeros, one launch fewer a call\n";

/// turboexl3's flag for the process, set before any dense call (`on` false puts the zeroing launch back).
pub fn apply(on: bool) void {
    t3.dense.dec_host_zero.get().* = on;
}

/// The calls that took host-made counters since the process began (turboexl3's count).
pub fn calls() u64 {
    return t3.dense.dec_host_zeroed.get().*;
}

test "turboexl3_glm53f host_zero: turboexl3's flag is off until the plugin applies the switch, its counters cover GLM-5.3-Flash's widest DEC-ONE call, and off puts the launch back" {
    const t = std.testing;
    const saved = t3.dense.dec_host_zero.get().*;
    defer t3.dense.dec_host_zero.get().* = saved;
    try t.expect(!saved);
    // the dense gate | up at 16 rows takes 768 counters: every served call is under the host's cap
    try t.expect(t3.dense.host_counters_max >= 768);
    apply(true);
    try t.expect(t3.dense.dec_host_zero.get().*);
    apply(false);
    try t.expect(!t3.dense.dec_host_zero.get().*);
    try t.expectEqual(@as(u64, 0), calls());
}
