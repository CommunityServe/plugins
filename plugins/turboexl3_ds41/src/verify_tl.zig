//! The decode verify's GPU timeline (`MLX_SERVE_DSV41_TIMELINE`), MLX-free: its setting, each recorded round's own
//! counts and the request's file. The timeline is mlx-serve's `profile/verify_timeline.zig` in the copied stream: every
//! command buffer MLX commits from a request's first DSpark round to its end, with its host commit time and Metal's GPU
//! start and end; each verify's window; each routed layer's host stamps in the verify (`Experts.run`). It compiles into
//! profile builds only (`zig build -Dplugin-profile=true`, the root's `plugin_profile`, which also compiles the copied
//! stream's other profile probes in), so a served build has no hook and refuses the setting. Its shim hooks the class of
//! MLX's command buffers through a buffer of the plugin's own queue (src/tl_buffer.m) and takes the first recorded
//! buffer's queue as MLX's (patches/turboexl3-zzz2-verify-timeline.patch). At each request's end the plugin logs the
//! timeline's `VERIFY_GPU_TIMELINE` line and writes `ds41-timeline-<pid>-<seq>.json` in the setting's directory:
//! `{"timeline": <its buffers, verifies and the read pool's signals>, "round_fields": [...], "rounds": [...]}`, the
//! rounds in the timeline's cycle order (`Round`).

const std = @import("std");

pub const max_path = 1024;

pub const Setting = struct {
    dir_buf: [max_path]u8 = undefined,
    dir_len: usize = 0,

    /// The directory the files go to; null: the timeline is off.
    pub fn dirPath(s: *const Setting) ?[]const u8 {
        return if (s.dir_len > 0) s.dir_buf[0..s.dir_len] else null;
    }
};

/// Set at load.
pub var setting: Setting = .{};

pub const Error = error{ TimelineSetting, TimelineNotBuilt };

/// `value` (null or empty: off): an absolute directory. `built`: a profile build; a set value in another build is
/// refused by name, as is a relative or overlong path.
pub fn parse(value: ?[]const u8, built: bool) Error!Setting {
    var s: Setting = .{};
    const v = value orelse return s;
    if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return s;
    if (!built) return error.TimelineNotBuilt;
    if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.TimelineSetting;
    @memcpy(s.dir_buf[0..v.len], v);
    s.dir_len = v.len;
    return s;
}

/// The stream's cumulative counters a round's are the difference of: unique experts missed and hit, demand bytes.
pub const Counters = struct { misses: u64 = 0, hits: u64 = 0, demand: u64 = 0 };

/// One recorded round: the verify's rows and the drafts kept, the stream's counters from its draft's start to its
/// commit, and its host times (ns): the draft (the round's start to the verify's begin), the verify (the timeline's
/// window: its forward to the read of its last eval) and the rest (the verify's end to the commit).
pub const Round = struct { rows: u32 = 0, accepted: u32 = 0, misses: u64 = 0, hits: u64 = 0, demand: u64 = 0, draft_ns: u64 = 0, verify_ns: u64 = 0, rest_ns: u64 = 0 };

/// The rounds per request (the timeline's `max_cycles`).
pub const max_rounds = 1024;

/// A request's rounds by cycle, and the open one's stamps. A round whose commit never comes stays zero.
pub const Rounds = struct {
    items: [max_rounds]Round = undefined,
    n: u32 = 0,
    cur: u32 = 0,
    t_begin: u64 = 0,
    t_verify: u64 = 0,
    t_verified: u64 = 0,
    c0: Counters = .{},

    pub fn reset(r: *Rounds) void {
        r.n = 0;
    }
    /// Cycle `cycle`'s round begins at `t` with the stream's counters `c`.
    pub fn begin(r: *Rounds, cycle: u32, t: u64, c: Counters) void {
        std.debug.assert(cycle < max_rounds);
        r.cur = cycle;
        r.items[cycle] = .{};
        while (r.n <= cycle) : (r.n += 1) r.items[r.n] = .{};
        r.t_begin = t;
        r.t_verify = t;
        r.t_verified = t;
        r.c0 = c;
    }
    pub fn verifyBegin(r: *Rounds, t: u64) void {
        r.t_verify = t;
    }
    pub fn verifyEnd(r: *Rounds, t: u64) void {
        r.t_verified = t;
    }
    /// The open round's commit at `t`: its rows, the drafts kept and the counters now.
    pub fn end(r: *Rounds, t: u64, c: Counters, rows: u32, accepted: u32) void {
        r.items[r.cur] = .{
            .rows = rows,
            .accepted = accepted,
            .misses = c.misses -| r.c0.misses,
            .hits = c.hits -| r.c0.hits,
            .demand = c.demand -| r.c0.demand,
            .draft_ns = r.t_verify -| r.t_begin,
            .verify_ns = r.t_verified -| r.t_verify,
            .rest_ns = t -| r.t_verified,
        };
    }
    pub fn slice(r: *const Rounds) []const Round {
        return r.items[0..r.n];
    }
};

/// The file's `round_fields` and `rounds` members (no braces), times in us.
pub fn writeRounds(w: *std.Io.Writer, rs: []const Round) !void {
    try w.writeAll("\"round_fields\": [\"rows\", \"accepted\", \"misses\", \"hits\", \"demand_bytes\", \"draft_us\", \"verify_us\", \"rest_us\"], \"rounds\": [");
    for (rs, 0..) |r, i| try w.print("{s}[{d}, {d}, {d}, {d}, {d}, {d:.1}, {d:.1}, {d:.1}]", .{ if (i == 0) "" else ", ", r.rows, r.accepted, r.misses, r.hits, r.demand, us(r.draft_ns), us(r.verify_ns), us(r.rest_ns) });
    try w.writeAll("]");
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

/// `<dir>/ds41-timeline-<pid>-<seq>.json` into `buf`.
pub fn filePath(buf: []u8, dir: []const u8, pid: i64, seq: u32) error{NoSpaceLeft}![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/ds41-timeline-{d}-{d}.json", .{ dir, pid, seq });
}

// ── tests ──

const testing = std.testing;

test "timeline setting: off unless set; an absolute directory in a profile build; refused by name otherwise" {
    for ([_]?[]const u8{ null, "", "0", "off" }) |v| {
        try testing.expect((try parse(v, false)).dirPath() == null);
        try testing.expect((try parse(v, true)).dirPath() == null);
    }
    try testing.expectEqualStrings("/data/tl", (try parse("/data/tl", true)).dirPath().?);
    try testing.expectError(error.TimelineNotBuilt, parse("/data/tl", false));
    try testing.expectError(error.TimelineSetting, parse("tl", true));
    var long: [max_path + 1]u8 = @splat('a');
    long[0] = '/';
    try testing.expectError(error.TimelineSetting, parse(&long, true));
}

test "timeline rounds: each cycle's counts and phases, a round never committed left zero" {
    var r: Rounds = .{};
    r.begin(0, 1_000, .{ .misses = 10, .hits = 100, .demand = 1 << 20 });
    r.verifyBegin(6_000);
    r.verifyEnd(70_000);
    r.end(71_500, .{ .misses = 75, .hits = 160, .demand = 3 << 20 }, 6, 4);
    r.begin(1, 80_000, .{});
    r.verifyBegin(85_000);
    r.begin(2, 90_000, .{ .misses = 75, .hits = 160, .demand = 3 << 20 });
    r.end(91_000, .{ .misses = 76, .hits = 160, .demand = 3 << 20 }, 1, 0);
    try testing.expectEqual(@as(usize, 3), r.slice().len);
    try testing.expectEqual(Round{ .rows = 6, .accepted = 4, .misses = 65, .hits = 60, .demand = 2 << 20, .draft_ns = 5_000, .verify_ns = 64_000, .rest_ns = 1_500 }, r.slice()[0]);
    try testing.expectEqual(Round{}, r.slice()[1]);
    try testing.expectEqual(Round{ .rows = 1, .accepted = 0, .misses = 1, .rest_ns = 1_000 }, r.slice()[2]);
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeRounds(&w, r.slice()[0..1]);
    try testing.expectEqualStrings("\"round_fields\": [\"rows\", \"accepted\", \"misses\", \"hits\", \"demand_bytes\", \"draft_us\", \"verify_us\", \"rest_us\"], \"rounds\": [[6, 4, 65, 60, 2097152, 5.0, 64.0, 1.5]]", w.buffered());
    r.reset();
    try testing.expectEqual(@as(usize, 0), r.slice().len);
    var pb: [64]u8 = undefined;
    try testing.expectEqualStrings("/t/ds41-timeline-42-3.json", try filePath(&pb, "/t", 42, 3));
}
