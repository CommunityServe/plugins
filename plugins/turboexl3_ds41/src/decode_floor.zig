//! The streamed bank's decode bill with a decode phase's runtime floor in place of a prompt pass's (the generated bill's
//! `decode_floor`). The bill's decode budget (`streamBudget`) takes the decode phase's working set out: the state at
//! `context + stream_generation` positions, one row's arrays and a runtime floor, the sum times 5/4. A prompt pass's
//! floor (`PREFILL_RUNTIME_FLOOR_BYTES`, 512 MiB) is 640 of the 804 MiB at the bank's 32768-token rung; decode's runtime
//! need beside its state is MLX's peak over its active bytes, 51 MiB on the 16K card and 29 on greedy. The decode phase's
//! working set is billed with `floor_bytes` (128 MiB) instead: 325 MiB, 480 MiB back to the decode slot rows (one more
//! row at 470 MiB); the prompt's bill and its rows are unchanged. Exact: only the decode rows' count moves (a route's
//! bytes are the records' in any row). A request's peak stays its prompt's, which decode stays under.

const std = @import("std");

/// The decode phase's runtime floor (the bill's `decode_floor`): 2.5 times MLX's largest measured peak over active at decode.
pub const floor_bytes: u64 = 128 << 20;

const testing = std.testing;

test "decode floor: the 32K rung's decode working set 480 MiB less, one decode row at the pack's 470 MiB" {
    // prefillBytes at 40960 positions, one row (the pack's config: 3200 B a token, 40 layers' 144-row window rings of
    // 512 halves, a row's held, attention, index and MoE bytes), times 5/4
    const state: u64 = 40960 * 3200 + 40 * 144 * 512 * 2;
    const row_bytes: u64 = 119_808 + 376_832 + 491_520 + 144_384;
    const prompt_floor: u64 = 512 << 20;
    const stock = (state + row_bytes + prompt_floor) * 5 / 4;
    const floored = (state + row_bytes + floor_bytes) * 5 / 4;
    try testing.expectEqual(@as(u64, 843_717_120), stock);
    try testing.expectEqual(@as(u64, 503_316_480), stock - floored);
    try testing.expectEqual(@as(u64, 1), (stock - floored) / 492_810_240);
}
