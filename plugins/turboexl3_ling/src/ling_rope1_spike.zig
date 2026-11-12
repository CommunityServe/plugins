//! C6 (MLX-free): the rope-once cache's fallback past the bill's point, refused by name. A trunk MLA call that neither
//! rope-once path takes rebuilds the expanded key from the cache's views (src/ling_mla_rope1.zig `expand`): `heads` x
//! (nope + rope) bf16 a cached token, 24 KiB on Ling-3.1-flash (src/bill.zig `ropeOnceSpike`), alive for the call's
//! attention. The bill holds no term for it: the spike sits inside the prompt pass's idle working set (`prompt_work`,
//! 2 GiB), which it fits up to 87,381 cached tokens (`point`). C1 serves contexts past that, so a forward with a call
//! that would rebuild the key past the point is refused by name before any append (plugin.zig `forward`, beside RQ's
//! refusal): the spike is never allocated, and no term is billed for it.
//! - Which calls rebuild it (the copied mla.zig `rope1Read`): a call of 2 to 16 rows that is not an exact verify's. A
//!   decode row and an exact verify's rows take the verify kernels' rows path, and a prompt call of 17 rows or more the
//!   flash launch (src/ling_flash.zig `min_rows`).
//! - A forward's calls: each span the host names, else the forward's rows at once. A span of a decode lane's rows or
//!   fewer runs as a forward of its own, and a wider one as a call of its own rows at each MLA layer.
//! - No whole prompt from position 0 is refused: past 512 tokens its spans are the 4,096-token chunks (a tail under 512
//!   merged into the chunk before it) and its last 31 tokens. What C6 refuses is a short forward on a long cache, such
//!   as a follow-up turn of 2 to 16 tokens on a restored prefix past 87,381 tokens.
//! - Not covered: a prompt call of 17 rows or more that turboexl3's flash launch refuses (its shape, dtype or the NAX
//!   gate) falls back as well. No served shape is refused (src/ling_flash.zig), and that refusal is only known inside
//!   the call, after its append.
const std = @import("std");
const bill = @import("bill.zig");

/// The fewest and most rows of a call that rebuilds the key: one row takes the rows path, 17 or more the flash launch.
pub const min_rows: u64 = 2;
pub const max_rows: u64 = 16;

/// What the refusal's reason starts with.
pub const refusal_name = "turboexl3_ling: a call that would rebuild the rope-once key past the bill's point";

/// The most cached tokens whose rebuilt key fits `prompt_work`: 87,381 on Ling-3.1-flash (64 heads x (128 + 64) bf16).
pub fn point(heads: u64, nope: u64, rope: u64) u64 {
    return bill.prompt_work / bill.ropeOnceSpike(heads, nope, rope);
}

/// A call of `rows` rows that leaves the cache at `end` rows (the key count its attention reads): its reason in `buf`
/// when it would rebuild the key (`spike` bytes a cached token) past `pt` tokens, else null.
pub fn callRefusal(buf: []u8, rows: u64, end: u64, pt: u64, spike: u64) ?[]const u8 {
    if (rows < min_rows or rows > max_rows or end <= pt) return null;
    return std.fmt.bufPrint(buf, refusal_name ++ ": a call of {d} rows ending at position {d} would rebuild {d} MiB of key ({d} B a cached token), past the {d} MiB the prompt pass's idle working set holds (up to {d} cached tokens)", .{ rows, end, (spike * end) >> 20, spike, bill.prompt_work >> 20, pt }) catch refusal_name;
}

/// The forward's kind, as the attention takes its calls: a decode forward's row (and a batched forward's, one a lane)
/// and an exact verify's rows never rebuild the key.
pub const Kind = enum { prefill, verify, exact_verify, decode };

/// A forward's calls (`ends`, the host's spans from `pos0`, else its `rows` from `pos0` at once): the first that would
/// rebuild the key past `pt` tokens, as a reason in `buf`, else null.
pub fn forwardRefusal(buf: []u8, kind: Kind, pos0: u64, rows: u64, ends: []const u64, pt: u64, spike: u64) ?[]const u8 {
    switch (kind) {
        .decode, .exact_verify => return null,
        .prefill, .verify => {},
    }
    if (ends.len == 0) return callRefusal(buf, rows, pos0 + rows, pt, spike);
    var prev = pos0;
    for (ends) |e| {
        if (callRefusal(buf, e -| prev, e, pt, spike)) |why| return why;
        prev = e;
    }
    return null;
}

// ── tests ──

const testing = std.testing;
const ling_rq = @import("ling_rq.zig");

/// Ling-3.1-flash's MLA heads, nope and rope dims
const ling_heads: u64 = 64;
const ling_nope: u64 = 128;
const ling_rope: u64 = 64;

test "C6's point: the rope-once fallback's rebuilt key, 24 KiB a cached token, fits the prompt pass's idle working set (2 GiB) up to 87,381 cached tokens and no further" {
    const spike = bill.ropeOnceSpike(ling_heads, ling_nope, ling_rope);
    try testing.expectEqual(@as(u64, 24_576), spike);
    try testing.expectEqual(@as(u64, 2 << 30), bill.prompt_work);
    const pt = point(ling_heads, ling_nope, ling_rope);
    try testing.expectEqual(@as(u64, 87_381), pt);
    try testing.expect(spike * pt <= bill.prompt_work);
    try testing.expect(spike * (pt + 1) > bill.prompt_work);
    // the sweep's points: the 64K point's context (83,694) is within it, the 128K point's (150,999) is not
    try testing.expect(83_694 <= pt);
    try testing.expect(150_999 > pt);
}

test "C6: a call of 2 to 16 rows that ends past the point is refused by name; one row, 17 rows or more, and every call that ends at the point or before it are not" {
    const spike = bill.ropeOnceSpike(ling_heads, ling_nope, ling_rope);
    const pt = point(ling_heads, ling_nope, ling_rope);
    var buf: [384]u8 = undefined;
    var rows: u64 = min_rows;
    while (rows <= max_rows) : (rows += 1) {
        try testing.expect(callRefusal(&buf, rows, pt + 1, pt, spike) != null);
        try testing.expect(callRefusal(&buf, rows, 150_999, pt, spike) != null);
        try testing.expectEqual(@as(?[]const u8, null), callRefusal(&buf, rows, pt, pt, spike));
    }
    try testing.expectEqual(@as(?[]const u8, null), callRefusal(&buf, 1, 150_999, pt, spike));
    try testing.expectEqual(@as(?[]const u8, null), callRefusal(&buf, 17, 150_999, pt, spike));
    // the reason: its name, the call, the key it would rebuild and the bill's line
    const why = callRefusal(&buf, 11, 90_011, pt, spike) orelse return error.TestExpectedRefusal;
    try testing.expect(std.mem.startsWith(u8, why, refusal_name));
    try testing.expect(std.mem.indexOf(u8, why, ": a call of 11 rows ending at position 90011 would rebuild 2109 MiB of key (24576 B a cached token), past the 2048 MiB the prompt pass's idle working set holds (up to 87381 cached tokens)") != null);
}

test "C6 over a forward's calls: no whole prompt from position 0 up to the 128K point's context (150,999 tokens) is refused, and a short forward on a restored prefix past the point is, unless it is a decode row or an exact verify" {
    const spike = bill.ropeOnceSpike(ling_heads, ling_nope, ling_rope);
    const pt = point(ling_heads, ling_nope, ling_rope);
    var buf: [384]u8 = undefined;
    var ends_buf: [64]u64 = undefined;
    // every whole prompt from position 0, at the host's spans
    var n: u64 = 2;
    while (n <= 150_999) : (n += 1) {
        const ends = ling_rq.spanEnds(&ends_buf, n);
        try testing.expectEqual(@as(?[]const u8, null), forwardRefusal(&buf, .prefill, 0, n, ends, pt, spike));
    }
    // a 12-token follow-up on a restored 90,000-token prefix: its spans (11 rows to the snapshot backoff, then 1)
    try testing.expect(forwardRefusal(&buf, .prefill, 90_000, 12, &.{ 90_011, 90_012 }, pt, spike) != null);
    // the same at 80,000, within the point: served
    try testing.expectEqual(@as(?[]const u8, null), forwardRefusal(&buf, .prefill, 80_000, 12, &.{ 80_011, 80_012 }, pt, spike));
    // a prompt forward with no spans named: its rows at once
    try testing.expect(forwardRefusal(&buf, .prefill, 90_000, 12, &.{}, pt, spike) != null);
    try testing.expectEqual(@as(?[]const u8, null), forwardRefusal(&buf, .prefill, 90_000, 17, &.{}, pt, spike));
    // a verify's 8 rows past the point: the exact verify's rows path, else the fallback
    try testing.expectEqual(@as(?[]const u8, null), forwardRefusal(&buf, .exact_verify, 140_000, 8, &.{}, pt, spike));
    try testing.expect(forwardRefusal(&buf, .verify, 140_000, 8, &.{}, pt, spike) != null);
    // a decode row past the point
    try testing.expectEqual(@as(?[]const u8, null), forwardRefusal(&buf, .decode, 150_000, 1, &.{}, pt, spike));
}
