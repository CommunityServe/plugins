//! A replaying prompt chunk's last decoder layer routes its last row only (the model's `moeLast`).
//!
//! A replaying chunk (CED, the model's `cedTail`) runs the decoder layers (from the last KV source on: 20..39 on
//! the pack) over its last `sliding_window` rows (128). Every decoder layer's output rows are read on by the next
//! layer's window KV (each layer keeps its own ring, which decode reads) and, at the DSpark targets (layers 37, 38, 39:
//! their inputs), by the drafter's rings; the last layer's output rows are read on by nothing but the chunk's logits,
//! which take the last row (`want == .last`). Its window KV comes from its input, as every layer's does. So the last
//! layer's routed call over all 128 rows reads records for 127 rows nothing reads: on the cold code-ctx16384 card layer
//! 39 routes about 138 experts, 61 of them misses (0.68 GB), of the decoder tail's 1060.
//!
//! `moeLast` serves that layer's MoE: the router and the shared expert over every row as `moe` runs them, the routed
//! call over the last row only (its 6 routes), forced through the wide lane (`force`: the experts' joinless
//! `routedParts`, which a call of at most 48 ids otherwise refuses), the same waves and combine a wide call runs; its row
//! becomes the last row of the layer's routed output (the other rows zero: nothing reads them). The window KV, the
//! drafter's rows and the decode state are the stock ones by construction; the one-row call's words are the 128-row
//! call's for that row (measured word for word on the server). The layer's residency differs: its call admits the last
//! row's misses, not the 128 rows'.

/// The copied model's `runLayers` arms the replaying chunk's last decoder layer's last span (`block` then takes `moeLast`).
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    armed_flag: bool = false,
    force: bool = false,
};
pub const shared_state = @import("sdk").shared.Shared(State, "dsv41_l39_last", .{});
pub inline fn ps() *State {
    return shared_state.get();
}
pub fn arm(v: bool) void {
    ps().armed_flag = v;
}
pub fn armed() bool {
    return ps().armed_flag;
}

/// The routed call in flight is `moeLast`'s one-row call, forced through the wide lane (the experts' `routedParts`).
pub fn forced() bool {
    return ps().force;
}
