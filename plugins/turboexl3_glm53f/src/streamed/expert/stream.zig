//! The streamer's stream over the EXL3 bank (`sdk_ext.expert.stream.StreamOf`): the slot, gate and residency types the
//! routed-expert executor binds, and DEVROUTE's resident map.
const sdk_ext = @import("../sdk_ext.zig");
const bo = @import("../build_flags.zig");
const expert_bank = @import("bank.zig");
const expert_policy = sdk_ext.expert.policy;

const S = sdk_ext.expert.stream.StreamOf(expert_bank, bo.dsv41_prefill_timers);
pub const BankKind = S.BankKind;
pub const SlotRef = S.SlotRef;
pub const Residents = S.Residents;
pub const Gates = S.Gates;
pub const Stats = S.Stats;
pub const Error = S.Error;
pub const max_wide_depth = S.max_wide_depth;

/// DEVROUTE's resident map of one layer from its policy: each resident expert's persistent slot as its packed
/// (bank << 24 | row) through `src.slotRef`; non-resident experts 0.
pub fn lutOf(pol: *const expert_policy.LayerPolicy, src: anytype, layer: u32, out: []u32) void {
    for (out, 0..) |*o, e| {
        o.* = 0;
        if (e >= pol.n_experts) continue;
        const s = pol.slotOf(@intCast(e)) orelse continue;
        if (s >= pol.capacity) continue;
        const ref = src.slotRef(layer, s);
        o.* = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
    }
}
