//! `[round-io]` (on with the DSpark profile, `MLX_SERVE_DIAG=dspark-profile`), MLX-free: each DSpark round's expert
//! stream traffic beside its verify time, and a request's traffic split into its rounds (the decode) and the rest (the
//! prompt pass, the first request's warm-up, forwards outside the rounds). The stream's own counters (`Snap`, read at a
//! round's draft and at its commit): unique experts missed and hit, the bytes its demand reads served (read, or copied
//! out of a lookahead record), the lookahead's physical bytes and those of its records no read claimed, the records it
//! issued and those claimed, the demand bytes copied out of lookahead records, the wall time with a demand read in flight
//! and the host's time blocked on reads; and the process's physical disk reads and page-ins (`proc_pid_rusage`): every
//! byte the drive moved for it, and how many of its pages it read back in through a mapping. A decode's misses wait on
//! GPU event gates, not on the host: its `read_wait` stays near 0 and the drive's bytes are the measure.

const std = @import("std");
const builtin = @import("builtin");

/// Cumulative counters at one point.
pub const Snap = struct {
    misses: u64 = 0,
    hits: u64 = 0,
    demand: u64 = 0,
    spec: u64 = 0,
    abandoned: u64 = 0,
    claimed: u64 = 0,
    issued: u64 = 0,
    wall_ns: u64 = 0,
    wait_ns: u64 = 0,
    adopt: u64 = 0,
    phys: u64 = 0,
    pageins: u64 = 0,

    /// `b - a`, field by field (0 where a counter went back).
    pub fn sub(b: Snap, a: Snap) Snap {
        var d: Snap = .{};
        inline for (@typeInfo(Snap).@"struct".field_names) |f| @field(d, f) = @field(b, f) -| @field(a, f);
        return d;
    }

    fn add(a: *Snap, b: Snap) void {
        inline for (@typeInfo(Snap).@"struct".field_names) |f| @field(a, f) += @field(b, f);
    }
};

/// The process's physical disk reads (bytes) and page-ins so far (0 where the platform does not say).
pub const Proc = struct { phys: u64 = 0, pageins: u64 = 0 };
pub fn proc() Proc {
    if (comptime !builtin.os.tag.isDarwin()) return .{};
    var info: RusageInfoV2 = undefined;
    if (proc_pid_rusage(std.c.getpid(), rusage_info_v2, &info) != 0) return .{};
    return .{ .phys = info.v[16], .pageins = info.v[4] };
}
/// `struct rusage_info_v2`: the uuid, then 18 counters: `ri_pageins` the 5th, `ri_diskio_bytesread` the 17th.
const RusageInfoV2 = extern struct { uuid: [16]u8, v: [18]u64 };
const rusage_info_v2: c_int = 2;
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;

/// Set at load.
pub var on: bool = false;

/// One round: its verify's rows, verify and round time, and its traffic.
const Row = struct { rows: u8, verify_us: u32, round_us: u32, d: Snap };

const max_rounds = 512;
var rounds: [max_rounds]Row = undefined;
var n_rounds: usize = 0;
/// Past `max_rounds`: counted in the decode's sums, not listed.
var sum: Snap = .{};
var sum_rows: u64 = 0;
var sum_verify_ns: u64 = 0;
var sum_round_ns: u64 = 0;
var n_all: usize = 0;
/// The open round's start, and the counters at the last request's end (or the load).
var at: ?Snap = null;
var req0: ?Snap = null;

/// The counters at the load (the first request's rest counts from here).
pub fn start(s: Snap) void {
    req0 = s;
    at = null;
    clear();
}

fn clear() void {
    n_rounds = 0;
    n_all = 0;
    sum = .{};
    sum_rows = 0;
    sum_verify_ns = 0;
    sum_round_ns = 0;
}

/// A round's draft starts.
pub fn begin(s: Snap) void {
    if (!on) return;
    at = s;
}

/// A round ended without its commit, or a forward outside the rounds runs.
pub fn abandon() void {
    at = null;
}

/// The round's commit: its verify of `rows` rows took `verify_ns`, the round `round_ns`.
pub fn end(s: Snap, rows: usize, verify_ns: u64, round_ns: u64) void {
    const a = at orelse return;
    at = null;
    const d = s.sub(a);
    sum.add(d);
    sum_rows += rows;
    sum_verify_ns += verify_ns;
    sum_round_ns += round_ns;
    n_all += 1;
    if (n_rounds == max_rounds) return;
    rounds[n_rounds] = .{
        .rows = @intCast(@min(rows, 255)),
        .verify_us = @intCast(@min(verify_ns / 1000, std.math.maxInt(u32))),
        .round_us = @intCast(@min(round_ns / 1000, std.math.maxInt(u32))),
        .d = d,
    };
    n_rounds += 1;
}

fn mb(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e6;
}

fn traffic(w: *std.Io.Writer, d: Snap) void {
    w.print("misses={d} hits={d} demand_mb={d:.1} adopt_mb={d:.1} phys_mb={d:.1} spec_mb={d:.1} abandoned_mb={d:.1} claimed={d} issued={d} pageins={d} read_wall_ms={d:.1} read_wait_ms={d:.1}", .{
        d.misses,  d.hits,   mb(d.demand), mb(d.adopt), mb(d.phys), mb(d.spec), mb(d.abandoned), d.claimed, d.issued,
        d.pageins, @as(f64, @floatFromInt(d.wall_ns)) / 1e6, @as(f64, @floatFromInt(d.wait_ns)) / 1e6,
    }) catch {};
}

/// At a request's end (`now`: the counters then): the decode's sums and the rest's, then one line a round (the first
/// `max_rounds`); the rounds cleared.
pub fn meters(now: Snap, buf: []u8) []const u8 {
    if (!on) return buf[0..0];
    var w = std.Io.Writer.fixed(buf);
    defer {
        req0 = now;
        clear();
    }
    w.print("[round-io] decode rounds={d} rows={d} verify_ms={d:.1} round_ms={d:.1} ", .{ n_all, sum_rows, @as(f64, @floatFromInt(sum_verify_ns)) / 1e6, @as(f64, @floatFromInt(sum_round_ns)) / 1e6 }) catch {};
    traffic(&w, sum);
    w.writeAll("\n") catch {};
    if (req0) |r0| {
        w.writeAll("[round-io] rest ") catch {};
        traffic(&w, now.sub(r0).sub(sum));
        w.writeAll("\n") catch {};
    }
    for (rounds[0..n_rounds], 0..) |r, i| {
        w.print("[round-io] r={d} rows={d} verify_ms={d:.2} round_ms={d:.2} ", .{ i, r.rows, @as(f64, @floatFromInt(r.verify_us)) / 1000, @as(f64, @floatFromInt(r.round_us)) / 1000 }) catch {};
        traffic(&w, r.d);
        w.writeAll("\n") catch {};
    }
    return w.buffered();
}

// ── tests ──

const testing = std.testing;

test "round io: a round's traffic is its commit's counters less its draft's; an abandoned round is not kept" {
    const saved = on;
    defer {
        on = saved;
        start(.{});
    }
    on = true;
    start(.{ .misses = 100, .demand = 1_000_000_000, .phys = 2_000_000_000 });
    begin(.{ .misses = 150, .hits = 10, .demand = 1_500_000_000, .phys = 2_600_000_000, .wall_ns = 5_000_000 });
    end(.{ .misses = 190, .hits = 40, .demand = 1_980_000_000, .phys = 3_500_000_000, .spec = 300_000_000, .claimed = 3, .issued = 9, .wall_ns = 37_000_000 }, 6, 120_500_000, 126_000_000);
    // abandoned: neither listed nor summed
    begin(.{ .misses = 190 });
    abandon();
    end(.{ .misses = 999 }, 6, 1, 1);
    var buf: [4096]u8 = undefined;
    const out = meters(.{ .misses = 230, .hits = 40, .demand = 2_500_000_000, .phys = 4_000_000_000, .spec = 300_000_000, .claimed = 3, .issued = 9, .wall_ns = 37_000_000 }, &buf);
    try testing.expect(std.mem.indexOf(u8, out, "[round-io] r=0 rows=6 verify_ms=120.50 round_ms=126.00 misses=40 hits=30 demand_mb=480.0 adopt_mb=0.0 phys_mb=900.0 spec_mb=300.0 abandoned_mb=0.0 claimed=3 issued=9 pageins=0 read_wall_ms=32.0 read_wait_ms=0.0\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "[round-io] decode rounds=1 rows=6 verify_ms=120.5 round_ms=126.0 misses=40 hits=30 demand_mb=480.0 adopt_mb=0.0 phys_mb=900.0") != null);
    // the rest: the request's 130 misses, 1500 MB demand and 2000 MB physical, less the round's
    try testing.expect(std.mem.indexOf(u8, out, "[round-io] rest misses=90 hits=10 demand_mb=1020.0 adopt_mb=0.0 phys_mb=1100.0 spec_mb=0.0") != null);
    try testing.expect(std.mem.indexOf(u8, out, "r=1 ") == null);
    // the next request counts from this one's end
    const next = meters(.{ .misses = 231, .hits = 40, .demand = 2_500_000_000, .phys = 4_000_000_000, .spec = 300_000_000, .claimed = 3, .issued = 9, .wall_ns = 37_000_000 }, &buf);
    try testing.expect(std.mem.indexOf(u8, next, "[round-io] decode rounds=0 rows=0") != null);
    try testing.expect(std.mem.indexOf(u8, next, "[round-io] rest misses=1 hits=0") != null);
}

test "round io: off, nothing is kept or printed; a counter that went back reads 0" {
    const saved = on;
    defer on = saved;
    on = false;
    abandon();
    begin(.{ .misses = 1 });
    end(.{ .misses = 5 }, 6, 1, 1);
    var buf: [256]u8 = undefined;
    try testing.expectEqual(@as(usize, 0), meters(.{}, &buf).len);
    const d = (Snap{ .misses = 3, .phys = 10 }).sub(.{ .misses = 5, .phys = 4 });
    try testing.expectEqual(@as(u64, 0), d.misses);
    try testing.expectEqual(@as(u64, 6), d.phys);
}

test "round io: the process's disk reads and page-ins read back, never going down" {
    const a = proc();
    const b = proc();
    try testing.expect(b.phys >= a.phys and b.pageins >= a.pageins);
}
