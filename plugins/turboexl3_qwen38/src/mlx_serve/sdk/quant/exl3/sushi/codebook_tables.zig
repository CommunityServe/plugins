//! copied_from: src/sdk/quant/exl3/sushi/codebook_tables.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Named, immutable codebook constant tables: what a pack's `expert_quant` points at with `codebook_table` and
//! `codebook_digest`. One row of u32 constants per rate n = 16K, every even n of the table's own range (within 16 to
//! 128), per codebook.
//!
//! A SPEC1 table's window is the one it was tuned at, nothing more: any window the engine admits for MCG decodes it,
//! and the kernels read the row of each bank's own rate from the table (`Decode.table`). PAIR1's pair codec reads
//! the whole 16-bit window, so a PAIR1 table holds only at its own window, and its rows are the ones the kernels
//! have compiled (`format.pair1Pair`).
//!
//! STD1 and SPEC1 are table-driven only: a pack names one of `shipped`, and a pack naming none this build holds is
//! refused. `std1` is one pair at every rate.
//!
//! To add a SPEC1 table: append one entry to `shipped` (name, window, rows, digest); no kernel changes.
const std = @import("std");
const builtin = @import("builtin");
const format = @import("expert_exl3.zig");

/// The range of the tables from before K8 (n = 24..64), which PAIR1's compiled rows still cover.
const k4_lo: u32 = 24;
const k4_count = (64 - k4_lo) / 2 + 1;

pub const Refusal = error{
    Exl3CodebookTableUnknown,
    Exl3CodebookTableDigestMismatch,
    Exl3CodebookTableWindowMismatch,
    Exl3CodebookTableCodebookMismatch,
    Exl3CodebookTableRateMissing,
    Exl3CodebookTableLegacyConfig,
    Exl3CodebookTableUndecodable,
};

pub const Row = [4]u32;

pub const Table = struct {
    name: []const u8,
    codebook: format.Codebook,
    window: format.Window,
    /// First 8 hex digits of sha256 over the canonical text, pinned so a shipped table cannot change.
    digest: []const u8,
    /// How many of a row's constants the codebook uses ((std1 and spec1 2, pair1 4)).
    fields: usize,
    /// The rate of `rows[0]`; row i is rate n_lo + 2i.
    n_lo: u32 = k4_lo,
    rows: []const Row,
    /// Digests of this table's earlier, shorter versions: their rows are this table's at their rates, and a pack
    /// stamped with one is refused as a legacy config.
    superseded: []const []const u8 = &.{},

    pub fn rateOf(self: *const Table, i: usize) format.Rate {
        return .{ .n = self.n_lo + 2 * @as(u32, @intCast(i)) };
    }

    /// The rates this table has a row for, as the bit set `Rate.in` reads (bit n/2).
    pub fn rates(self: *const Table) u128 {
        var set: u128 = 0;
        for (0..self.rows.len) |i| set |= @as(u128, 1) << @intCast(self.rateOf(i).n / 2);
        return set;
    }

    /// Whether a pack at `window` may name this table: STD1 and SPEC1 decode at any window, PAIR1 only at its own.
    pub fn admits(self: *const Table, window: format.Window) bool {
        return self.codebook.tabledHashMask() or self.window == window;
    }

    /// Whether the kernels decode this table's rows: STD1's and SPEC1's are data, PAIR1's are the ones compiled in.
    pub fn decodable(self: *const Table) bool {
        if (self.codebook.tabledHashMask()) return true;
        for (self.rows, 0..) |r, i| {
            const want = compiledRow(self.codebook, self.rateOf(i)) orelse return false;
            if (!std.mem.eql(u32, r[0..self.fields], want[0..self.fields])) return false;
        }
        return true;
    }
};

fn compiledRow(codebook: format.Codebook, rate: format.Rate) ?Row {
    switch (codebook) {
        .pair1 => {
            const p = format.pair1Pair(rate) orelse return null;
            return .{ p.mult, p.rot, p.mask0, p.mask1 };
        },
        else => return null,
    }
}

fn compiledRows(codebook: format.Codebook) [k4_count]Row {
    var out: [k4_count]Row = undefined;
    for (&out, 0..) |*r, i| r.* = compiledRow(codebook, .{ .n = k4_lo + 2 * @as(u32, @intCast(i)) }).?;
    return out;
}

const Pair = struct { u32, u32 };

/// `pair1-w16-v1` is, by definition, what the kernels have compiled since before the tables existed.
const shipped = [_]Table{
    .{ .name = "pair1-w16-v1", .codebook = .pair1, .window = .w16, .digest = "c089b4eb", .fields = 4, .rows = &compiledRows(.pair1) },
    .{ .name = "spec1-w15", .codebook = .spec1, .window = .w15, .digest = "37f72431", .fields = 2, .n_lo = 16, .rows = &tabledRows(w15Pairs()), .superseded = &.{"2ac2594d"} },
    .{ .name = "spec1-w16", .codebook = .spec1, .window = .w16, .digest = "753ffa03", .fields = 2, .rows = &tabledRows(w16Pairs()) },
    .{ .name = "std1", .codebook = .std1, .window = .w15, .digest = "c5dba171", .fields = 2, .rows = &tabledRows(@as([k4_count]Pair, @splat(global_pair))) },
};

/// The set at window 15, n = 16..128. Its rows
/// n = 24..64 are the 21-row table it extends (digest `2ac2594d`), so a pack stamped with that digest decodes the same.
fn w15Pairs() [57]Pair {
    return .{
        .{ 0x84d1cf89, 0x8fe08947 },
        .{ 0x84c1cf49, 0x8fe48a47 },
        .{ 0x84d1cfe9, 0x8fe08a47 },
        .{ 0x84d1cfc9, 0x8ff08b67 },
        .{ 0x84c1cf49, 0x8fe08b47 },
        .{ 0x84c1cf99, 0x8fa08b66 },
        .{ 0x862c1d37, 0x8ff28c50 },
        .{ 0x84641fb7, 0x8fa08c40 },
        .{ 0x846c1fa7, 0x8fe08c80 },
        .{ 0x844c1fd7, 0x8ffe8c98 },
        .{ 0x846c1fa7, 0x8ffe8d10 },
        .{ 0x845c1f77, 0x8ffa8e10 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x842c1fc7, 0x8fff8f00 },
        .{ 0x843c1fb7, 0x8ffe8f10 },
        .{ 0x84181fb5, 0x8fe08f80 },
        .{ 0x84081f75, 0x8ffa8f72 },
        .{ 0x84081f75, 0x8fd68fd0 },
        .{ 0x84241f27, 0x8ffc8fc0 },
        .{ 0x84081f73, 0x8fe08f80 },
        .{ 0x84081f75, 0x8fbe8ff0 },
        .{ 0x3fed4cef, 0x8fe08fe0 },
        .{ 0x3fed4cef, 0x8fc18ff2 },
        .{ 0x3fed4cef, 0x8ff08ff0 },
        .{ 0x3fed4cef, 0x8fe28ff0 },
        .{ 0x3fed4cef, 0x8fe08fe0 },
        .{ 0x80081b7b, 0x8ff08fc8 },
        .{ 0x80081b7b, 0x8ff48fc0 },
        .{ 0x400036cb, 0x8f448fff },
        .{ 0x400036cb, 0x8fe48fff },
        .{ 0x400036cb, 0x8f448ff9 },
        .{ 0x8000387b, 0x8fe08ff7 },
        .{ 0x8000387b, 0x8fc08fee },
        .{ 0x80003779, 0x8fe88ffe },
        .{ 0x80003779, 0x8fc88ffc },
        .{ 0x80003779, 0x8fca8ffc },
        .{ 0x80003779, 0x8fcc8ffe },
        .{ 0x80003879, 0x8ffc8fef },
        .{ 0x50003851, 0x8ff98fff },
        .{ 0xd0003751, 0x8ff98fff },
        .{ 0x50003851, 0x8ff98ff6 },
        .{ 0xd0003931, 0x8ff98fff },
        .{ 0xd0003931, 0x8ff98ffa },
        .{ 0xd0003931, 0x8ff48ffa },
        .{ 0x300031b1, 0x8ffb8ffc },
        .{ 0x300031b1, 0x8ffa8ffe },
        .{ 0x300031b1, 0x8ffb8ffc },
        .{ 0xc00031f1, 0x8ffe8ffa },
        .{ 0x400031f1, 0x8ffe8ffe },
        .{ 0x400031f1, 0x8ffe8ffe },
        .{ 0x40003931, 0x8ffc8ffc },
        .{ 0x40003931, 0x8ffc8ffc },
        .{ 0x40003931, 0x8ff88ffe },
        .{ 0x40003931, 0x8ff88ffc },
        .{ 0x80003811, 0x8ffc8ffc },
        .{ 0x80003811, 0x8ffc8ffc },
    };
}

/// The set at window 16, n = 24..64.
fn w16Pairs() [k4_count]Pair {
    return .{
        .{ 0x862d1c19, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8fee8c30 },
        .{ 0x862d1d37, 0x8ffe8c10 },
        .{ 0x842f1f37, 0x8ffc8d10 },
        .{ 0x840f1f67, 0x8ffc8d10 },
        .{ 0x844f1f77, 0x8ffe8d40 },
        .{ 0x840d1f37, 0x8ffe8e10 },
        .{ 0x84091f33, 0x8fe88e00 },
        .{ 0x844d1f77, 0x8fe08e00 },
        .{ 0x843c1fa7, 0x8ffe8e40 },
        .{ 0x844c1f7f, 0x8ffc8f02 },
        .{ 0x841c1fb9, 0x8fe88f04 },
        .{ 0x842c1fc7, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8ffe8fc1 },
        .{ 0x83fc1fa3, 0x8ffe8fd0 },
        .{ 0x83fc1fa3, 0x8fee8fc0 },
        .{ 0x84061f77, 0x8fd08f80 },
        .{ 0x84081f6f, 0x8ffe8fc0 },
        .{ 0x84081f25, 0x8ffe8fe0 },
        .{ 0x83f81f6f, 0x8ff48fc0 },
    };
}

/// The one pair `std1` decodes with, at every rate.
const global_pair: Pair = .{ 0x840C1F37, 0x8FFE8FD0 };

fn tabledRows(pairs: anytype) [pairs.len]Row {
    var out: [pairs.len]Row = undefined;
    for (&out, pairs) |*r, p| r.* = .{ p[0], p[1], 0, 0 };
    return out;
}

/// Test-only SPEC1 table whose pair differs at every rate, so a decoder that read one pair for every bank fails.
fn syntheticPairs() [57]Pair {
    const masks = [_]u32{ 0x8FFE8FD0, 0x8FE08FC0, 0x8FF08FE0, 0x8F8E8FD8 };
    var out: [57]Pair = undefined;
    for (&out, 0..) |*p, i| p.* = .{ 0x840C1F37 +% (16 + 2 * @as(u32, @intCast(i))) *% 0x2E0B5, masks[i % masks.len] };
    return out;
}

const test_only = [_]Table{
    .{ .name = "spec1-test-per-rate", .codebook = .spec1, .window = .w12, .digest = "a2f2fe40", .fields = 2, .n_lo = 16, .rows = &tabledRows(syntheticPairs()) },
};

pub const tables = shipped ++ if (builtin.is_test) test_only else [0]Table{};

pub fn index(t: *const Table) u8 {
    return @intCast(t - &tables[0]);
}

pub fn lookup(name: []const u8) ?*const Table {
    for (&tables) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

/// The table `block` names for `codebook` at `window`, refusing by name. Only std1, spec1 and pair1 have tables.
pub fn resolve(block: std.json.ObjectMap, codebook: format.Codebook, window: format.Window) Refusal!*const Table {
    if (!codebook.hasTable()) return error.Exl3CodebookTableCodebookMismatch;
    const name = block.get("codebook_table") orelse return error.Exl3CodebookTableUnknown;
    if (name != .string) return error.Exl3CodebookTableUnknown;
    const table = lookup(name.string) orelse return error.Exl3CodebookTableUnknown;
    const digest = block.get("codebook_digest") orelse return error.Exl3CodebookTableDigestMismatch;
    if (digest != .string) return error.Exl3CodebookTableDigestMismatch;
    for (table.superseded) |old| if (std.mem.eql(u8, digest.string, old)) return error.Exl3CodebookTableLegacyConfig;
    if (!std.mem.eql(u8, digest.string, table.digest)) return error.Exl3CodebookTableDigestMismatch;
    if (table.codebook != codebook) return error.Exl3CodebookTableCodebookMismatch;
    if (!table.admits(window)) return error.Exl3CodebookTableWindowMismatch;
    if (!table.decodable()) return error.Exl3CodebookTableUndecodable;
    return table;
}
