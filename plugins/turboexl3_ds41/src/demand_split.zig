//! `[demand-split]` (dq1a's, carried by dq1b; on with `[round-io]`, the DSpark profile `MLX_SERVE_DIAG=dspark-profile`),
//! MLX-free: the decode's demand reads split by class at their settle, from the copied read pool's result words (with
//! `MLX_SERVE_DSV41_GU_SPLIT` a split gate/up ticket's words are its two halves': its time their span). Each settled range
//! of a decode route goes by its layer's record size (the pack's K = 3 and K = 2 records), its part (the gate/up span or
//! the down span) and how it was served: read with no unclaimed lookahead chunk in flight at its start and none started
//! while it read (`clean`: one preadv's own speed beside the other demand reads), read beside the lookahead (`spec`: an
//! unclaimed chunk in flight at its start or started while it ran), or copied out of a lookahead record (`adopt`: no
//! preadv; its time the copy, its wait the record's remaining chunks). Per class: the ranges, MB, mean and p90 ms (the
//! range's first preadv to its end; an adopted range's copy), the lookahead chunks started beside the spec ranges, and
//! the adopted ranges' mean and p90 wait. At each request's end the request's decode, then cleared. Reads only: no
//! byte served moves.

const std = @import("std");

/// Set at load (with `[round-io]`).
pub var on: bool = false;

pub const Kind = enum(u2) { clean, spec, adopt };
const n_kinds = 3;
/// Distinct record sizes kept apart (a fifth size's ranges are counted, not split).
const max_sizes = 4;
/// 25 µs bins to 10 ms; the last bin holds 10 ms and over.
const bin_ns: u64 = 25_000;
const n_bins = 400;

const Hist = struct {
    n: u64 = 0,
    sum_ns: u64 = 0,
    bins: [n_bins]u32 = @splat(0),

    fn add(h: *Hist, ns: u64) void {
        h.n += 1;
        h.sum_ns += ns;
        h.bins[@min(ns / bin_ns, n_bins - 1)] += 1;
    }

    fn meanMs(h: *const Hist) f64 {
        if (h.n == 0) return 0;
        return @as(f64, @floatFromInt(h.sum_ns)) / @as(f64, @floatFromInt(h.n)) / 1e6;
    }

    /// The upper edge of the bin that holds the 90th percentile range (ms).
    fn p90Ms(h: *const Hist) f64 {
        if (h.n == 0) return 0;
        const want = (h.n * 9 + 9) / 10;
        var seen: u64 = 0;
        for (h.bins, 0..) |c, i| {
            seen += c;
            if (seen >= want) return @as(f64, @floatFromInt((i + 1) * bin_ns)) / 1e6;
        }
        return @as(f64, @floatFromInt(n_bins * bin_ns)) / 1e6;
    }
};

const Cell = struct {
    bytes: u64 = 0,
    /// The range's own time (read or copy).
    time: Hist = .{},
    /// spec: the unclaimed lookahead chunks started while the ranges read.
    chunks: u64 = 0,
    /// adopt: the wait for the record's chunks before the copy.
    wait: Hist = .{},
};

var sizes: [max_sizes]u64 = @splat(0);
var n_sizes: usize = 0;
var cells: [max_sizes][2][n_kinds]Cell = @splat(@splat(@splat(.{})));
/// Ranges of a record size past `max_sizes`.
pub var unsplit: u64 = 0;

fn sizeIndex(record_bytes: u64) ?usize {
    for (sizes[0..n_sizes], 0..) |s, i| if (s == record_bytes) return i;
    if (n_sizes == max_sizes) return null;
    sizes[n_sizes] = record_bytes;
    n_sizes += 1;
    return n_sizes - 1;
}

/// The class of a settled range: copied out of a lookahead record (no preadv, a payload), else read beside the
/// lookahead when its flags word says so (bit 0 a chunk in flight at its start, `>> 1` the chunks started since).
pub fn kindOf(adopted: bool, lookahead: i64) Kind {
    if (adopted) return .adopt;
    return if (lookahead > 0) .spec else .clean;
}

/// One settled demand range of a decode route: its layer's record bytes, its part (`down`: the down span), whether a
/// lookahead record served it, its lookahead flags, its payload, its own time and (adopted) its wait.
pub fn note(record_bytes: u64, down: bool, adopted: bool, lookahead: i64, bytes: u64, ns: u64, wait_ns: u64) void {
    const si = sizeIndex(record_bytes) orelse {
        unsplit += 1;
        return;
    };
    const kind = kindOf(adopted, lookahead);
    const c = &cells[si][@intFromBool(down)][@intFromEnum(kind)];
    c.bytes += bytes;
    c.time.add(ns);
    switch (kind) {
        .spec => c.chunks += @intCast(@max(lookahead, 0) >> 1),
        .adopt => c.wait.add(wait_ns),
        .clean => {},
    }
}

/// Clears every class (the start of a request's count).
pub fn clear() void {
    sizes = @splat(0);
    n_sizes = 0;
    cells = @splat(@splat(@splat(.{})));
    unsplit = 0;
}

/// One `[demand-split]` line a (record size, part) with any range, the larger records first, then cleared; "" off or
/// with no range.
pub fn meters(buf: []u8) []const u8 {
    if (!on) return buf[0..0];
    defer clear();
    var order: [max_sizes]usize = .{ 0, 1, 2, 3 };
    std.mem.sort(usize, order[0..n_sizes], {}, struct {
        fn gt(_: void, a: usize, b: usize) bool {
            return sizes[a] > sizes[b];
        }
    }.gt);
    var w: std.Io.Writer = .fixed(buf);
    for (order[0..n_sizes]) |si| for (0..2) |part| {
        const cs = &cells[si][part];
        if (cs[0].time.n + cs[1].time.n + cs[2].time.n == 0) continue;
        const cl = &cs[@intFromEnum(Kind.clean)];
        const sp = &cs[@intFromEnum(Kind.spec)];
        const ad = &cs[@intFromEnum(Kind.adopt)];
        w.print("[demand-split] rec_mb={d:.2} down={d} clean_n={d} clean_mb={d:.1} clean_ms={d:.3} clean_p90={d:.3} spec_n={d} spec_mb={d:.1} spec_ms={d:.3} spec_p90={d:.3} spec_chunks={d} adopt_n={d} adopt_mb={d:.1} adopt_ms={d:.3} adopt_p90={d:.3} adopt_wait_ms={d:.3} adopt_wait_p90={d:.3}\n", .{
            @as(f64, @floatFromInt(sizes[si])) / 1e6, part,
            cl.time.n,                                  @as(f64, @floatFromInt(cl.bytes)) / 1e6,
            cl.time.meanMs(),                           cl.time.p90Ms(),
            sp.time.n,                                  @as(f64, @floatFromInt(sp.bytes)) / 1e6,
            sp.time.meanMs(),                           sp.time.p90Ms(),
            sp.chunks,                                  ad.time.n,
            @as(f64, @floatFromInt(ad.bytes)) / 1e6,    ad.time.meanMs(),
            ad.time.p90Ms(),                            ad.wait.meanMs(),
            ad.wait.p90Ms(),
        }) catch return w.buffered();
    };
    if (unsplit > 0) w.print("[demand-split] unsplit={d}\n", .{unsplit}) catch {};
    return w.buffered();
}

const testing = std.testing;

test "demand split: the class of a range from its adoption and its lookahead flags" {
    try testing.expectEqual(Kind.adopt, kindOf(true, 3));
    try testing.expectEqual(Kind.clean, kindOf(false, 0));
    try testing.expectEqual(Kind.spec, kindOf(false, 1));
    try testing.expectEqual(Kind.spec, kindOf(false, 4));
}

test "demand split: counts, means and p90 by record size and part, the larger records first, cleared after" {
    const saved = on;
    defer {
        on = saved;
        clear();
    }
    clear();
    on = true;
    // K2 gate/up: four clean reads of 1 ms and one of 2 ms; K3 down: one spec read beside 2 started chunks
    for (0..4) |_| note(8_891_904, false, false, 0, 5_923_000, 1_000_000, 0);
    note(8_891_904, false, false, 0, 5_923_000, 2_000_000, 0);
    note(13_315_584, true, false, 1 | (2 << 1), 4_438_000, 1_500_000, 0);
    note(13_315_584, true, true, 0, 4_438_000, 100_000, 400_000);
    var buf: [1024]u8 = undefined;
    const out = meters(&buf);
    const k3 = std.mem.indexOf(u8, out, "rec_mb=13.32 down=1").?;
    const k2 = std.mem.indexOf(u8, out, "rec_mb=8.89 down=0").?;
    try testing.expect(k3 < k2);
    try testing.expect(std.mem.indexOf(u8, out, "clean_n=5 clean_mb=29.6 clean_ms=1.200 clean_p90=2.025") != null);
    try testing.expect(std.mem.indexOf(u8, out, "spec_n=1 spec_mb=4.4 spec_ms=1.500 spec_p90=1.525 spec_chunks=2") != null);
    try testing.expect(std.mem.indexOf(u8, out, "adopt_n=1 adopt_mb=4.4 adopt_ms=0.100 adopt_p90=0.125 adopt_wait_ms=0.400 adopt_wait_p90=0.425") != null);
    try testing.expectEqual(@as(usize, 0), n_sizes);
    try testing.expectEqualStrings("", meters(&buf));
}

test "demand split: off writes nothing" {
    const saved = on;
    defer on = saved;
    on = false;
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("", meters(&buf));
}
