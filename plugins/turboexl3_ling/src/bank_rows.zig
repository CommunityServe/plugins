//! The expert bank's slot rows, MLX-free: each phase's whole rows at the budgets it opens with, past its windows and the
//! read pool's staging (src/ling_bank.zig `Bill`), and C5's plans against the host's real KV at each phase (below).

const std = @import("std");

/// The bank's terms beside its slot rows: a row of every routed layer, a phase's windows, the read pool's staging past
/// its default workers, and a layer's experts.
pub const Bank = struct { row_bytes: u64, prompt_windows: u64, decode_windows: u64, staging: u64, n_experts: u32 };
/// The phases' budgets for the slot rows and their terms (the host's `abi.BankBudget`).
pub const Budget = struct { prompt: u64, decode: u64 };
pub const Rows = struct { prompt: u32, decode: u32 };

/// The whole rows `budget` holds past `fixed` bytes of a phase's own terms (its windows and the staging).
pub fn rowsIn(bank: Bank, budget: u64, fixed: u64) u64 {
    return (budget -| fixed) / bank.row_bytes;
}

/// A bank phase.
pub const Phase = enum { prompt, decode };

/// The bank's bytes in `phase` (`b`: the bank's bill, `ling_bank.Bill`): the phase's rows, its windows and the read
/// staging, as `rowsOf` budgets them. A bank-fit calibration subtracts these from the phase's measured peak.
pub fn phaseBytes(b: anytype, phase: Phase) u64 {
    return switch (phase) {
        .prompt => @as(u64, b.rows.prompt) * b.row_bytes + b.prompt_windows + b.staging,
        .decode => @as(u64, b.rows.decode) * b.row_bytes + b.decode_windows + b.staging,
    };
}

/// Each phase's whole rows past its windows and the staging, both at most a layer's experts, and each within its own
/// budget. The kit's stream keeps the prompt's rows through decode (`Stream.grow` only adds rows, `Stream.shrink` only
/// returns to the prompt's), so the prompt takes at most the decode budget's rows: decode never holds more than its
/// own fit. At least one row in each phase.
pub fn rowsOf(bank: Bank, budget: Budget) error{StreamedRowsDoNotFit}!Rows {
    const n: u64 = bank.n_experts;
    const d = @min(rowsIn(bank, budget.decode, bank.decode_windows + bank.staging), n);
    const p = @min(rowsIn(bank, budget.prompt, bank.prompt_windows + bank.staging), d);
    if (p == 0) return error.StreamedRowsDoNotFit;
    return .{ .prompt = @intCast(p), .decode = @intCast(d) };
}

// ── C5: the bank against the host's real KV ──
//
// The host bills a streamed model's prompt at 0 (mlx-serve `prefill_chunk.zig` `prefillNeededAtChunk`: "a longer prompt
// is the plugin's refusal by name"), and its room for the bank (`driver.zig` `streamRoomAt`) holds one prompt of 16K + 1K.
// So the bank reads what the host has really allocated, KV included, from its headroom (`Host.memoryCeiling`: the
// host's ceiling less what is resident), and plans each phase from it, as the model plugin's RQ did (C5, main's ruling
// 2026-11-11):
// - a prompt forward whose headroom holds less than one slot row (the pass is at the host's ceiling) is under the base
//   and refused by name, before its calls run;
// - a decode or trim plan under the base by at most the rows `free_ram` covers keeps the bank at its base with a named
//   warning; past that cover it is refused by name;
// - a decode plan above the base grows to at most the decode rows the headroom holds past `free_ram` (a trim below the
//   load's decode rows is logged).

/// The free RAM a decode plan keeps past the host's headroom (the model plugin's `free_ram`).
pub const free_ram: u64 = 4 << 30;

pub const Verdict = enum { fits, warn, refuse };
/// A plan's rows (what the bank takes), the whole rows the headroom holds (`held`, under the base when below it) and
/// its verdict.
pub const Plan = struct { rows: u32, held: i64, verdict: Verdict };

/// The bank's whole rows `free_ram` holds: 12 of Ling-3.1's 336,427,008 B rows.
pub fn coverRows(row_bytes: u64) u64 {
    return free_ram / row_bytes;
}

/// The rows the bank may hold with `now` rows held and `headroom` bytes past them, keeping `reserve` free.
pub fn heldAt(row_bytes: u64, now: u32, headroom: u64, reserve: u64) i64 {
    const n: i64 = now;
    if (headroom >= reserve) return n + @as(i64, @intCast((headroom - reserve) / row_bytes));
    return n - @as(i64, @intCast(std.math.divCeil(u64, reserve - headroom, row_bytes) catch unreachable));
}

/// A decode plan (at the phase change, `now` the base rows) or a trim (at a decode forward's settle, `now` the rows
/// grown): `rows` the decode rows to grow to, at most the load's.
pub fn decodePlan(rows: Rows, row_bytes: u64, now: u32, headroom: u64) Plan {
    const held = heldAt(row_bytes, now, headroom, free_ram);
    const base: i64 = rows.prompt;
    if (held >= base) return .{ .rows = @intCast(@min(held, rows.decode)), .held = held, .verdict = .fits };
    const under: u64 = @intCast(base - held);
    return .{ .rows = rows.prompt, .held = held, .verdict = if (under <= coverRows(row_bytes)) .warn else .refuse };
}

/// A prompt forward's plan at its calls: under the base when its headroom holds less than one row.
pub fn promptPlan(rows: Rows, row_bytes: u64, headroom: u64) Plan {
    const held = heldAt(row_bytes, rows.prompt, headroom, row_bytes);
    return .{ .rows = rows.prompt, .held = held, .verdict = if (held < rows.prompt) .refuse else .fits };
}

// ── tests ──

const testing = std.testing;
const mib: u64 = 1 << 20;

/// Ling-3.1-flash 2.60bpw's bank (68 routed layers of 4,947,456 B a record, 5 prompt windows of 48 records, one at
/// decode, 118,489,088 B of staging: the served bank's terms).
const served: Bank = .{
    .row_bytes = 336_427_008,
    .prompt_windows = 5 * 48 * 2 * 4_947_456,
    .decode_windows = 48 * 2 * 4_947_456,
    .staging = 118_489_088,
    .n_experts = 512,
};

test "bank rows: each phase's whole rows past its windows and staging, the prompt at most decode's, both at most 512" {
    const r = try rowsOf(served, .{ .prompt = 69_998 * mib, .decode = 76_329 * mib });
    // (budget - windows - staging) / row, rounded down.
    try testing.expectEqual(@as(u32, @intCast((69_998 * mib - served.prompt_windows - served.staging) / served.row_bytes)), r.prompt);
    try testing.expectEqual(@as(u32, @intCast((76_329 * mib - served.decode_windows - served.staging) / served.row_bytes)), r.decode);
    try testing.expect(r.decode >= r.prompt);
    // A budget past every expert stops at 512.
    const huge = try rowsOf(served, .{ .prompt = 1 << 50, .decode = 1 << 50 });
    try testing.expectEqual(Rows{ .prompt = 512, .decode = 512 }, huge);
}

test "bank rows: a prompt or decode budget that holds no row past its windows and staging is refused" {
    try testing.expectError(error.StreamedRowsDoNotFit, rowsOf(served, .{ .prompt = served.prompt_windows + served.staging + served.row_bytes - 1, .decode = 1 << 40 }));
    try testing.expectError(error.StreamedRowsDoNotFit, rowsOf(served, .{ .prompt = 1 << 40, .decode = served.decode_windows + served.staging + served.row_bytes - 1 }));
}

test "bank rows: a prompt budget past the decode fit takes decode's rows, and the decode bank stays within its budget" {
    // A served load's terms and budgets (bfc3, run 2: the bank's bytes 62,839,508,992 B at 186 prompt rows and
    // 63,076,986,880 B at 186 decode rows, against budgets that hold 186 and 182).
    const bank: Bank = .{ .row_bytes = 336_427_008, .prompt_windows = 145_596_416, .decode_windows = 383_074_304, .staging = 118_489_088, .n_experts = 512 };
    const budget: Budget = .{ .prompt = 62_934_480_832, .decode = 61_978_492_736 };
    try testing.expectEqual(@as(u64, 186), rowsIn(bank, budget.prompt, bank.prompt_windows + bank.staging));
    try testing.expectEqual(@as(u64, 182), rowsIn(bank, budget.decode, bank.decode_windows + bank.staging));
    const r = try rowsOf(bank, budget);
    try testing.expectEqual(Rows{ .prompt = 182, .decode = 182 }, r);
    const b = .{ .row_bytes = bank.row_bytes, .prompt_windows = bank.prompt_windows, .decode_windows = bank.decode_windows, .staging = bank.staging, .rows = r };
    try testing.expect(phaseBytes(b, .decode) <= budget.decode);
    try testing.expect(phaseBytes(b, .prompt) <= budget.prompt);
    // Every split of the two budgets: each phase's bytes within its own.
    var prng = std.Random.DefaultPrng.init(5);
    for (0..1000) |_| {
        const bb: Budget = .{ .prompt = prng.random().uintLessThan(u64, 200 << 30), .decode = prng.random().uintLessThan(u64, 200 << 30) };
        const rr = rowsOf(served, bb) catch continue;
        const x = .{ .row_bytes = served.row_bytes, .prompt_windows = served.prompt_windows, .decode_windows = served.decode_windows, .staging = served.staging, .rows = rr };
        try testing.expect(rr.prompt <= rr.decode);
        try testing.expect(phaseBytes(x, .decode) <= bb.decode and phaseBytes(x, .prompt) <= bb.prompt);
    }
}

test "C5: a decode plan grows to the rows the headroom holds past free RAM, at most the load's decode rows" {
    const rows: Rows = .{ .prompt = 214, .decode = 236 };
    const row = served.row_bytes;
    // Room for 30 rows past free RAM: the load's 236.
    try testing.expectEqual(Plan{ .rows = 236, .held = 244, .verdict = .fits }, decodePlan(rows, row, 214, free_ram + 30 * row));
    // Room for 10: a trim to 224.
    try testing.expectEqual(Plan{ .rows = 224, .held = 224, .verdict = .fits }, decodePlan(rows, row, 214, free_ram + 10 * row + 5));
    // Exactly free RAM: the base.
    try testing.expectEqual(Plan{ .rows = 214, .held = 214, .verdict = .fits }, decodePlan(rows, row, 214, free_ram));
}

test "C5: a decode or trim plan under the base by at most the rows free RAM covers (12) warns; past it, refused" {
    const rows: Rows = .{ .prompt = 214, .decode = 236 };
    const row = served.row_bytes;
    try testing.expectEqual(@as(u64, 12), coverRows(row));
    // 12 rows short of free RAM: 12 under the base, a warning at the base.
    try testing.expectEqual(Plan{ .rows = 214, .held = 202, .verdict = .warn }, decodePlan(rows, row, 214, free_ram - 12 * row));
    // 13 rows short: refused.
    try testing.expectEqual(Verdict.refuse, decodePlan(rows, row, 214, free_ram - 12 * row - 1).verdict);
    try testing.expectEqual(Verdict.refuse, decodePlan(rows, row, 214, 0).verdict);
    // A trim with the bank grown to 230: the headroom past those rows.
    try testing.expectEqual(Plan{ .rows = 232, .held = 232, .verdict = .fits }, decodePlan(rows, row, 230, free_ram + 2 * row));
    try testing.expectEqual(Verdict.warn, decodePlan(rows, row, 214, free_ram - 1).verdict);
}

test "C5: a prompt forward with less than one row of headroom is under the base and refused" {
    const rows: Rows = .{ .prompt = 214, .decode = 236 };
    const row = served.row_bytes;
    try testing.expectEqual(Verdict.fits, promptPlan(rows, row, row).verdict);
    try testing.expectEqual(Verdict.fits, promptPlan(rows, row, 40 << 30).verdict);
    try testing.expectEqual(Plan{ .rows = 214, .held = 213, .verdict = .refuse }, promptPlan(rows, row, row - 1));
    try testing.expectEqual(Verdict.refuse, promptPlan(rows, row, 0).verdict);
}

test "bank rows: a phase's bytes are its rows, its windows and the staging, as rowsOf budgets them" {
    const bank: Bank = .{ .row_bytes = 100, .prompt_windows = 30, .decode_windows = 60, .staging = 7, .n_experts = 512 };
    const rows = try rowsOf(bank, .{ .prompt = 1037, .decode = 2067 });
    const b = .{ .row_bytes = bank.row_bytes, .prompt_windows = bank.prompt_windows, .decode_windows = bank.decode_windows, .staging = bank.staging, .rows = rows };
    try std.testing.expectEqual(@as(u64, 10 * 100 + 30 + 7), phaseBytes(b, .prompt));
    try std.testing.expectEqual(@as(u64, 20 * 100 + 60 + 7), phaseBytes(b, .decode));
}
