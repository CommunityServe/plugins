//! `MLX_SERVE_DSV41_STAGE_VOTES` (read at load; unset or empty: the floor 8; `N`, 1 to 65535: the floor N; `0`: off;
//! anything else is refused by name; with `MLX_SERVE_DSV41_STAGE` on): a wide prompt call's staging takes only the
//! predicted stream misses that at least N rows of the prediction name.
//!
//! Staging (src/stage.zig, src/stage_next.zig) reads the prediction's stream misses past the seed ranks, hottest first,
//! into the windows the call's first stream groups take: the coldest experts the prediction names, so its tail is the
//! experts a few rows name, the ones the call routes least often. On a cold 16K card (ctl25's a arms, w-pio-20261102-0114)
//! 786 staged records were never adopted: 10.47 GB of the prompt pass's 102.6 GB. On, the staging plan (`plan`, for the
//! stream's `stageGroups` and STAGE_NEXT's `stageNext`) stops at the first expert of the ranking (votes descending) under
//! the floor; P1's and STAGE_NEXT's read-ahead into the persistent rows keep the whole ranking (the row choice sees every
//! named expert). An expert the floor takes out that the call routes is read by its own route, as before staging.
//!
//! On or off, every staged record's votes are kept with it: a `[stage-votes] floor= posted= adopted= cut= p1= a1= c1= p2=
//! ... c64=` line at the end of each prompt that staged, with `MLX_SERVE_DSV41_PROMPT_IO`, after its `[stream-stats
//! prompt]` line (the counts reset at every prompt's end). Bucket N holds the records N to 2N - 1 rows named (64: 64 and
//! more): pN staged (posted), aN adopted by a route of the call, cN the floor's (taken out of a staging plan; a plan's
//! records, posted or not). `posted` is every record the prompt staged: `[stream-stats prompt]`'s counters start at the
//! prompt's first routed call, after the first wide layer's own read-ahead has staged (on a warm server, where that
//! layer's rows are taken), so its stage_posted can be lower by that layer's staging (89 records on the new-text slice of
//! w-pio-20261102-0252); its stage_adopted counts their adoptions, and `adopted` agrees with it. Then the staged jobs
//! (each a read-pool submission, up to `expert_io.max_items` records, posted in STAGE_TRIM's order) by the records they
//! read that no route adopted, counted where their staging is let go: `jobs=` (jobs that read; a job STAGE_TRIM
//! withdrew read nothing), `jwaste=` (their read records no route adopted), `jall=` (jobs none of whose read records a
//! route adopted: what withdrawing whole jobs could save at most), `j0= j1= j2= j4= j8=` (jobs with 0, 1, 2 to 3, 4 to
//! 7, 8 or more such records).
//!
//! Read on the M5 Max (sv1, w-pio-20261102-0252, two binaries: a = ctl27 (f6aa81b, zzz8y), b = this code on zzz8y with
//! the floor off by default, `=8` in the b arms; a warm server, then a1 b1 b2 a2, each a cold code-ctx16384 card then the
//! new-text 16K slice 3): cold 16K a 1419.7/1415.2 -> b 1429.0/1428.0 tok/s (+0.78%, pairs +0.66% and +0.90%), new text
//! 1500.8/1490.4 -> 1513.7/1497.5 (+0.67%, pairs +0.86% and +0.48%); 150 prompt / 179 decode rows in every arm, the texts
//! the same in every arm. Read 102.25 -> 100.27 GB cold, 70.32 -> 68.52 new; the staging's waste (posted - adopted)
//! 775 -> 624 records cold, 212 -> 109 new. On the cold card the floor took 957 records out (4268 -> 3311 posted), 806 of
//! which the call adopted (3493 -> 2687): 808 more demand reads, about 151 records saved. What stays wasted is mostly
//! high-vote (64 and more: 750 of 1104 adopted; 32 to 63: 151 wasted): the prediction's misses, which a higher floor
//! does not reach.
//!
//! Exact: the floor picks which records are staged, never the routes; a record is the same bytes in any row, and a
//! wave's rows and the combine's fold do not depend on the groups' composition.

const std = @import("std");

/// The floor unset (sv1's measured setting).
pub const default_floor: u32 = 8;
/// The floor (`configure`, at load): 0 off, else a staged record's least votes.
pub var floor: u32 = default_floor;

pub const Error = error{StageVotesSetting};

pub fn configure(value: ?[]const u8) Error!void {
    floor = try parse(value);
}

/// `value`: null or empty the default floor; `0` off (0); a decimal 1 to 65535 the floor; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!u32 {
    const v = value orelse return default_floor;
    if (v.len == 0) return default_floor;
    for (v) |c| if (c < '0' or c > '9') return error.StageVotesSetting;
    const n = std.fmt.parseInt(u32, v, 10) catch return error.StageVotesSetting;
    if (n > std.math.maxInt(u16)) return error.StageVotesSetting;
    return n;
}

/// An expert's votes in a prediction counted by expert (`votes`; empty: none known, 0).
pub fn of(votes: []const u32, e: u16) u32 {
    return if (e < votes.len) votes[e] else 0;
}

/// A record whose votes are known (`v` > 0) under the floor.
pub fn under(v: u32) bool {
    return floor > 0 and v > 0 and v < floor;
}

/// A staged record's votes as its row keeps them: a u16 (a prediction names an expert at most once a row, and a call has
/// up to 16K rows), capped below `stage.no_expert` (the layout's empty row).
pub fn keep(v: u32) u16 {
    return @intCast(@min(v, std.math.maxInt(u16) - 1));
}

/// A staging plan's stream misses into `out`: `experts` (the prediction past the seed ranks, hottest first) that `pol` holds
/// no row for (`slotOf`), up to `cap`, each one's votes (`votes`, by expert; empty: not known) beside it in `mv`. The
/// misses under the floor are taken out (counted, not staged: the ranking's votes descend, so they are its tail). Returns
/// how many are staged.
pub fn plan(pol: anytype, experts: []const u16, votes: []const u32, cap: usize, out: []u16, mv: []u16) usize {
    var nm: usize = 0;
    var cut: usize = 0;
    for (experts) |e| {
        if (nm + cut == cap) break;
        if (pol.slotOf(e) != null) continue;
        const v = of(votes, e);
        if (under(v)) {
            noteCut(v);
            cut += 1;
            continue;
        }
        out[nm] = e;
        mv[nm] = keep(v);
        nm += 1;
    }
    return nm;
}

pub const n_buckets = 7;
/// Each bucket's least votes.
pub const bucket_floor = [n_buckets]u32{ 1, 2, 4, 8, 16, 32, 64 };

/// The bucket of `v` > 0 votes: floor(log2 v), 64 and more the last.
pub fn bucket(v: u32) usize {
    std.debug.assert(v > 0);
    return @min(std.math.log2_int(u32, v), n_buckets - 1);
}

/// The jobs' buckets: 0, 1, 2 to 3, 4 to 7, 8 and more records no route adopted.
pub const n_job_buckets = 5;

/// The job bucket of `wasted` records.
pub fn jobBucket(wasted: u32) usize {
    if (wasted == 0) return 0;
    return @min(@as(usize, std.math.log2_int(u32, wasted)) + 1, n_job_buckets - 1);
}

const Counts = struct {
    posted: [n_buckets]u64 = @splat(0),
    adopted: [n_buckets]u64 = @splat(0),
    cut: [n_buckets]u64 = @splat(0),
    jobs: u64 = 0,
    jwaste: u64 = 0,
    jall: u64 = 0,
    job: [n_job_buckets]u64 = @splat(0),
};
var counts: Counts = .{};

/// A staged record posted with `v` votes (0: not known, not counted).
pub fn notePosted(v: u16) void {
    if (v > 0) counts.posted[bucket(v)] += 1;
}

/// A staged record with `v` votes adopted by a route.
pub fn noteAdopted(v: u16) void {
    if (v > 0) counts.adopted[bucket(v)] += 1;
}

/// A stream miss with `v` votes the floor took out of a staging plan.
pub fn noteCut(v: u32) void {
    if (v > 0) counts.cut[bucket(v)] += 1;
}

/// A staged job let go: `read` of its records read, `wasted` of those no route adopted (a job that read none: not counted).
pub fn noteJob(read: u32, wasted: u32) void {
    if (read == 0) return;
    counts.jobs += 1;
    counts.jwaste += wasted;
    counts.jall += @intFromBool(wasted == read);
    counts.job[jobBucket(wasted)] += 1;
}

fn sum(xs: [n_buckets]u64) u64 {
    var s: u64 = 0;
    for (xs) |x| s += x;
    return s;
}

/// The prompt's end: its `[stage-votes]` line when `print` and it staged (or the floor took a record out), then the
/// counts reset.
pub fn report(print: bool) void {
    defer counts = .{};
    if (!print or sum(counts.posted) + sum(counts.cut) == 0) return;
    const p = counts.posted;
    const a = counts.adopted;
    const c = counts.cut;
    const j = counts.job;
    // Two format calls (32 arguments at most a call), one line written.
    var buf: [1024]u8 = undefined;
    const votes = std.fmt.bufPrint(&buf, "[stage-votes] floor={d} posted={d} adopted={d} cut={d} p1={d} a1={d} c1={d} p2={d} a2={d} c2={d} p4={d} a4={d} c4={d} p8={d} a8={d} c8={d} p16={d} a16={d} c16={d} p32={d} a32={d} c32={d} p64={d} a64={d} c64={d}", .{ floor, sum(p), sum(a), sum(c), p[0], a[0], c[0], p[1], a[1], c[1], p[2], a[2], c[2], p[3], a[3], c[3], p[4], a[4], c[4], p[5], a[5], c[5], p[6], a[6], c[6] }) catch return;
    std.debug.print("{s} jobs={d} jwaste={d} jall={d} j0={d} j1={d} j2={d} j4={d} j8={d}\n", .{ votes, counts.jobs, counts.jwaste, counts.jall, j[0], j[1], j[2], j[3], j[4] });
}

const testing = std.testing;

test "stage_votes setting: unset or empty the floor 8; 0 off; 1 to 65535 the floor; anything else refused by name" {
    for ([_]?[]const u8{ null, "" }) |v| try testing.expectEqual(@as(u32, 8), try parse(v));
    try testing.expectEqual(@as(u32, 0), try parse("0"));
    try testing.expectEqual(@as(u32, 1), try parse("1"));
    try testing.expectEqual(@as(u32, 8), try parse("8"));
    try testing.expectEqual(@as(u32, 65535), try parse("65535"));
    for ([_][]const u8{ "65536", "-1", "+8", "8 ", " 8", "on", "8.0", "0x8", "99999999999" }) |v| try testing.expectError(error.StageVotesSetting, parse(v));
}

test "stage_votes buckets: N to 2N - 1 votes, 64 and more the last" {
    const want = [_][2]u32{ .{ 1, 0 }, .{ 2, 1 }, .{ 3, 1 }, .{ 4, 2 }, .{ 7, 2 }, .{ 8, 3 }, .{ 15, 3 }, .{ 16, 4 }, .{ 31, 4 }, .{ 32, 5 }, .{ 63, 5 }, .{ 64, 6 }, .{ 65535, 6 } };
    for (want) |p| try testing.expectEqual(@as(usize, p[1]), bucket(p[0]));
    for (bucket_floor, 0..) |b, i| try testing.expectEqual(i, bucket(b));
}

test "stage_votes floor: off takes nothing out; on, known votes under it; unknown votes never" {
    const saved = floor;
    defer floor = saved;
    floor = 0;
    try testing.expect(!under(1));
    floor = 8;
    try testing.expect(under(1) and under(7));
    try testing.expect(!under(8) and !under(9) and !under(0));
    const votes = [_]u32{ 3, 0, 12 };
    try testing.expectEqual(@as(u32, 3), of(&votes, 0));
    try testing.expectEqual(@as(u32, 12), of(&votes, 2));
    try testing.expectEqual(@as(u32, 0), of(&votes, 3));
    try testing.expectEqual(@as(u32, 0), of(&.{}, 0));
    try testing.expectEqual(@as(u16, 16384), keep(16384));
    try testing.expectEqual(@as(u16, std.math.maxInt(u16) - 1), keep(1 << 20));
}

/// A layer's rows for `plan`'s tests: the experts in `resident` hold rows.
const FakePolicy = struct {
    resident: []const u16,
    pub fn slotOf(p: *const FakePolicy, e: u16) ?u32 {
        for (p.resident, 0..) |x, i| if (x == e) return @intCast(i);
        return null;
    }
};

test "stage_votes plan: off, the first cap misses; on, the misses under the floor taken out and counted, the cap the same" {
    const saved = floor;
    defer floor = saved;
    counts = .{};
    defer counts = .{};
    const pol = FakePolicy{ .resident = &.{ 2, 5 } };
    // The ranking (votes descending) and each expert's votes.
    const ranked = [_]u16{ 3, 2, 7, 1, 5, 0, 6 };
    var votes: [8]u32 = @splat(0);
    for ([_][2]u32{ .{ 3, 40 }, .{ 2, 20 }, .{ 7, 9 }, .{ 1, 8 }, .{ 5, 6 }, .{ 0, 3 }, .{ 6, 1 } }) |x| votes[x[0]] = x[1];
    var out: [8]u16 = undefined;
    var mv: [8]u16 = undefined;
    floor = 0;
    try testing.expectEqualSlices(u16, &.{ 3, 7, 1, 0, 6 }, out[0..plan(&pol, &ranked, &votes, 10, &out, &mv)]);
    try testing.expectEqualSlices(u16, &.{ 40, 9, 8, 3, 1 }, mv[0..5]);
    try testing.expectEqualSlices(u16, &.{ 3, 7 }, out[0..plan(&pol, &ranked, &votes, 2, &out, &mv)]);
    try testing.expectEqual(@as(u64, 0), sum(counts.cut));
    floor = 8;
    try testing.expectEqualSlices(u16, &.{ 3, 7, 1 }, out[0..plan(&pol, &ranked, &votes, 10, &out, &mv)]);
    try testing.expectEqual(@as(u64, 1), counts.cut[0]);
    try testing.expectEqual(@as(u64, 1), counts.cut[1]);
    counts = .{};
    // The cap counts the misses the floor takes out: 3, 7, 1 staged and 0 cut fill a cap of 4; 6 is past it.
    try testing.expectEqual(@as(usize, 3), plan(&pol, &ranked, &votes, 4, &out, &mv));
    try testing.expectEqual(@as(u64, 1), sum(counts.cut));
    try testing.expectEqual(@as(u64, 1), counts.cut[1]);
    // Votes not known: nothing is taken out, the rows keep 0.
    counts = .{};
    try testing.expectEqualSlices(u16, &.{ 3, 7, 1, 0, 6 }, out[0..plan(&pol, &ranked, &.{}, 10, &out, &mv)]);
    try testing.expectEqualSlices(u16, &.{ 0, 0, 0, 0, 0 }, mv[0..5]);
    try testing.expectEqual(@as(u64, 0), sum(counts.cut));
}

test "stage_votes jobs: by the records no route adopted; a job that read none not counted" {
    counts = .{};
    defer counts = .{};
    const want = [_][2]u32{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 2 }, .{ 4, 3 }, .{ 7, 3 }, .{ 8, 4 }, .{ 48, 4 } };
    for (want) |x| try testing.expectEqual(@as(usize, x[1]), jobBucket(x[0]));
    noteJob(8, 0);
    noteJob(8, 3);
    noteJob(4, 4);
    noteJob(0, 0);
    try testing.expectEqual(@as(u64, 3), counts.jobs);
    try testing.expectEqual(@as(u64, 7), counts.jwaste);
    try testing.expectEqual(@as(u64, 1), counts.jall);
    try testing.expectEqualSlices(u64, &.{ 1, 0, 1, 1, 0 }, &counts.job);
}

test "stage_votes counts: by bucket, unknown votes not counted, reset at the report" {
    counts = .{};
    notePosted(1);
    notePosted(5);
    notePosted(5);
    notePosted(0);
    noteAdopted(5);
    noteCut(2);
    noteCut(100);
    try testing.expectEqual(@as(u64, 1), counts.posted[0]);
    try testing.expectEqual(@as(u64, 2), counts.posted[2]);
    try testing.expectEqual(@as(u64, 3), sum(counts.posted));
    try testing.expectEqual(@as(u64, 1), counts.adopted[2]);
    try testing.expectEqual(@as(u64, 1), counts.cut[1]);
    try testing.expectEqual(@as(u64, 1), counts.cut[6]);
    report(false);
    try testing.expectEqual(@as(u64, 0), sum(counts.posted) + sum(counts.adopted) + sum(counts.cut));
}
