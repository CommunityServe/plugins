//! `[stream-stats]`: the streamed bank's counters by phase in the request's summary (the bank's `meters`, which
//! glm5_next's `ModelTable.meters` logs beside `[spec-stats]` at the request's end), turboexl3_ds41's line split at the
//! phase change. The kit's stream counters (`kit.streamed.expert.Stats`) are read at the first forward of each phase
//! (`Meter.mark`, from the bank's `phase`, which glm5_next calls before each forward: a prompt pass after any other
//! forward, any other forward after a prompt pass) and at the request's end; each line is the difference
//! (`Stats.since`). The decode line also splits the lookahead's unclaimed records by why (the stream's own `spec_*`
//! counters), counts the decode's forwards (the bank sees a forward's phase, not its mode) and ends with the adopted
//! ranges' copy time. The counters of the
//! switches whose mechanism is a seam of the kit's stream (src/streamed/glm5n/experts.zig `seams`: the gate/up spans read
//! in pieces, `TURBOEXL3_GLM53F_DEMAND_SPLIT`; the demand ranges' waits and the demand's reads of claimed records; the
//! copies in pieces, `TURBOEXL3_GLM53F_ADOPT_PIECES`; the lookahead's I/O shape) print only where the kit's `Stats` has
//! them. Where the kit's `Stats` lacks the demand ranges' waits, the decode line ends with the read pool's own counters
//! (`Extra`: the waits, and the seams' side arrays where the kit's stream has them). Host counters only: nothing here
//! times, reads or waits for anything (the copies' time is the read pool's own clock, in the stream's counters).

const std = @import("std");

/// The read pool's counters the kit's `Stats` does not carry, each since the pool's start: the demand ranges that
/// waited for a lookahead record still in flight and that wait; DEMAND_SPLIT's ranges split and their pieces;
/// ADOPT_PIECES' ranges copied in pieces and the pieces another worker copied. Without a seam its pair reads 0.
pub const Extra = struct {
    adopt_waits: i64 = 0,
    adopt_wait_ns: i64 = 0,
    split: [2]i64 = .{ 0, 0 },
    adopt: [2]i64 = .{ 0, 0 },

    /// `src.poolCounters()` (src/streamed/glm5n/experts.zig) where the source has it, else zeros.
    pub fn of(src: anytype) Extra {
        const S = std.meta.Child(@TypeOf(src));
        if (comptime !@hasDecl(S, "poolCounters")) return .{};
        const c = src.poolCounters();
        return .{ .adopt_waits = c.adopt_waits, .adopt_wait_ns = c.adopt_wait_ns, .split = c.split, .adopt = c.adopt };
    }

    fn since(a: Extra, b: Extra) Extra {
        return .{
            .adopt_waits = a.adopt_waits - b.adopt_waits,
            .adopt_wait_ns = a.adopt_wait_ns - b.adopt_wait_ns,
            .split = .{ a.split[0] - b.split[0], a.split[1] - b.split[1] },
            .adopt = .{ a.adopt[0] - b.adopt[0], a.adopt[1] - b.adopt[1] },
        };
    }
};

/// `Stats`: the stream's counters (`kit.streamed.expert.Stats`).
pub fn Meter(comptime Stats: type) type {
    return struct {
        const Self = @This();

        phase: enum { none, prompt, decode } = .none,
        at_prompt: Stats = .{},
        at_decode: Stats = .{},
        px_prompt: Extra = .{},
        px_decode: Extra = .{},
        /// The decode phase's forwards.
        forwards: u64 = 0,
        buf: [2048]u8 = undefined,

        /// Before each forward: a prompt pass (`prompt`) or not. `src.stats()` and `Extra.of(src)` are read at a phase's
        /// first forward only.
        pub fn mark(self: *Self, prompt: bool, src: anytype) void {
            if (prompt) {
                if (self.phase != .prompt) {
                    self.at_prompt = src.stats();
                    self.px_prompt = Extra.of(src);
                    self.phase = .prompt;
                    self.forwards = 0;
                }
                return;
            }
            if (self.phase == .prompt) {
                self.at_decode = src.stats();
                self.px_decode = Extra.of(src);
                self.phase = .decode;
            }
            self.forwards += 1;
        }

        /// The request's lines at its end (`now` and `px`: the stream's and the read pool's counters then), in the
        /// meter's buffer.
        pub fn lines(self: *Self, now: Stats, px: Extra) []const u8 {
            var w: std.Io.Writer = .fixed(&self.buf);
            switch (self.phase) {
                .none => {},
                .prompt => line(&w, "prompt", now.since(self.at_prompt), null, .{}),
                .decode => {
                    line(&w, "prompt", self.at_decode.since(self.at_prompt), null, .{});
                    line(&w, "decode", now.since(self.at_decode), self.forwards, px.since(self.px_decode));
                },
            }
            return w.buffered();
        }

        fn line(w: *std.Io.Writer, phase: []const u8, d: Stats, forwards: ?u64, px: Extra) void {
            w.print("[stream-stats] {s}: routes={d} hits={d} misses={d} evictions={d} persistent_loads={d} transient_loads={d} skipped={d} read_gb={d:.3} preadv={d} read_s={d:.3} read_wall_ms={d:.1} read_wait_ms={d:.1} ahead={d}/{d}/{d} ahead_dropped={d} promoted={d} spec={d}/{d}/{d} gates_forced={d}", .{
                phase,
                d.route_calls,
                d.expert_cache_hits,
                d.expert_cache_misses,
                d.expert_cache_evictions,
                d.persistent_loads,
                d.transient_loads,
                d.loads_skipped,
                @as(f64, @floatFromInt(d.expert_bytes_read)) / 1e9,
                d.preadv_calls,
                d.expert_read_seconds,
                @as(f64, @floatFromInt(d.read_wall_ns)) / 1e6,
                @as(f64, @floatFromInt(d.read_wait_ns)) / 1e6,
                d.ahead_posted,
                d.ahead_hits,
                d.ahead_demand,
                d.ahead_dropped,
                d.promoted,
                d.spec_issued,
                d.spec_landed,
                d.claimed,
                d.gates_forced,
            }) catch return;
            // The decode line: the lookahead's records no demand read claimed, by why: dropped while queued, dropped
            // while read, read and never served (each a wrong prediction), or still queued when their call's demand read
            // them (a right one not started); the claimed still in flight at their claim; the speculative reads' and the
            // abandoned records' physical bytes; then the gate/up spans of lone decode reads read in pieces.
            if (forwards != null) w.print(" spec_expired={d} spec_abandoned={d} spec_discarded={d} spec_cancelled={d} spec_inflight={d} spec_gb={d:.3} spec_abandoned_gb={d:.3}", .{
                d.spec_expired,
                d.spec_abandoned,
                d.spec_discarded,
                d.spec_cancelled,
                d.spec_claimed_inflight,
                @as(f64, @floatFromInt(d.spec_bytes)) / 1e9,
                @as(f64, @floatFromInt(d.spec_abandoned_bytes)) / 1e9,
            }) catch return;
            if (comptime @hasField(Stats, "split_ranges")) {
                if (forwards != null) w.print(" split={d}", .{d.split_ranges}) catch return;
            }
            if (forwards) |f| w.print(" forwards={d}", .{f}) catch return;
            if (forwards != null) w.print(" adopt_copy_ms={d:.1}", .{d.adopt_copy_seconds * 1e3}) catch return;
            // The decode line's end: the demand ranges that waited for a lookahead record's bytes still in flight and
            // that wait (ms, its help reads included); the runs of a claimed record's pieces read by a demand worker
            // (runs/GB/ms), the records their last reader landed, and the ranges adopted over done pieces; the
            // adopted ranges' copy time (ms, summed over the ranges) and ADOPT_PIECES' ranges copied in pieces / pieces
            // another worker copied; the lookahead's records by pick order: each call's first pick read ahead / routed
            // by its target call, then the rest read ahead / routed (the precision of each).
            if (comptime @hasField(Stats, "adopt_waits")) {
                if (forwards != null) w.print(" adopt_waits={d} adopt_wait_ms={d:.1} help={d}/{d:.3}/{d:.1} help_lands={d} early_adopts={d} adopt_split={d}/{d} ahead_picks={d}/{d}/{d}/{d}", .{
                    d.adopt_waits,
                    @as(f64, @floatFromInt(d.adopt_wait_ns)) / 1e6,
                    d.help_runs,
                    @as(f64, @floatFromInt(d.help_bytes)) / 1e9,
                    @as(f64, @floatFromInt(d.help_ns)) / 1e6,
                    d.help_lands,
                    d.early_adopts,
                    d.adopt_split,
                    d.adopt_helped,
                    d.ahead_first_read,
                    d.ahead_first_routed,
                    d.ahead_rest_read,
                    d.ahead_rest_routed,
                }) catch return;
            }
            // turboexl3_glm53f: the lookahead's I/O shape, printed once a record started: the staged records' (read into
            // their staging slots) records / preadv calls / iovecs / iovecs starting or ending off a page.
            if (comptime @hasField(Stats, "io_st_recs")) {
                if (forwards != null and d.io_st_recs != 0) w.print(" spec_io={d}/{d}/{d}/{d}", .{ d.io_st_recs, d.io_st_preadv, d.io_st_iovecs, d.io_st_offpage }) catch return;
            }
            // The read pool's own counters (`Extra`) where the kit's `Stats` lacks the waits: the demand ranges that
            // waited for a lookahead record in flight and that wait (ms); DEMAND_SPLIT's ranges split / pieces;
            // ADOPT_PIECES' ranges copied in pieces / pieces another worker copied.
            if (comptime !@hasField(Stats, "adopt_waits")) {
                if (forwards != null) w.print(" pool_adopt_waits={d} pool_adopt_wait_ms={d:.1} split_pieces={d}/{d} adopt_pieces={d}/{d}", .{
                    px.adopt_waits,
                    @as(f64, @floatFromInt(px.adopt_wait_ns)) / 1e6,
                    px.split[0],
                    px.split[1],
                    px.adopt[0],
                    px.adopt[1],
                }) catch return;
            }
            w.writeAll("\n") catch return;
        }
    };
}

// ── tests ──

const testing = std.testing;

/// The stream's and the read pool's counters at each read, in order (a stream that only counts).
fn Counters(comptime Stats: type) type {
    return struct {
        seq: []const Stats,
        pool: []const Extra,
        at: usize = 0,
        pool_at: usize = 0,
        pub fn stats(self: *@This()) Stats {
            const s = self.seq[self.at];
            self.at += 1;
            return s;
        }
        pub fn poolCounters(self: *@This()) Extra {
            const c = self.pool[self.pool_at];
            self.pool_at += 1;
            return c;
        }
    };
}

/// The plugin-rooted test's check over the stream's own `Stats` (plugin.zig's test).
pub fn expectPhases(comptime Stats: type) !void {
    var m: Meter(Stats) = .{};
    var s0: Stats = .{};
    s0.route_calls = 100;
    s0.expert_cache_misses = 40;
    var s1 = s0; // the request's prompt starts
    s1.route_calls = 142;
    var s2 = s1; // its decode starts: the prompt routed 42 calls, read 3 GB, missed 500
    s2.route_calls = 184;
    s2.expert_cache_misses = 540;
    s2.expert_bytes_read = 3_000_000_000;
    s2.spec_expired = 7; // the lookahead's records before the decode: not the decode's
    var s3 = s2; // its end: the decode's 420 routes and 80 misses
    s3.route_calls = 604;
    s3.expert_cache_misses = 620;
    s3.read_wait_ns = 1_500_000;
    // the decode's lookahead: 100 records issued, 30 claimed (12 still in flight at their claim), 9 still queued when
    // their call's demand read them, 50 dropped while queued, 8 while read (0.32 GB read), 3 read and never served;
    // 1.9 GB read speculatively
    s3.spec_issued = 100;
    s3.claimed = 30;
    s3.spec_claimed_inflight = 12;
    s3.spec_cancelled = 9;
    s3.spec_expired = 57;
    s3.spec_abandoned = 8;
    s3.spec_discarded = 3;
    s3.spec_bytes = 1_900_000_000;
    s3.spec_abandoned_bytes = 320_000_000;
    // the decode's adopted ranges: 12.5 ms of copies
    s3.adopt_copy_seconds = 0.0125;
    const seq = [_]Stats{ s0, s0, s1, s2 };
    // the read pool: 2 waits (1 ms) before the decode; in it 3 more (2.5 ms), 4 ranges split into 12 pieces, 5 ranges
    // copied in pieces with 9 pieces another worker copied
    const x0: Extra = .{ .adopt_waits = 2, .adopt_wait_ns = 1_000_000 };
    const x1: Extra = .{ .adopt_waits = 2, .adopt_wait_ns = 1_000_000, .split = .{ 1, 3 }, .adopt = .{ 2, 2 } };
    const x2: Extra = .{ .adopt_waits = 5, .adopt_wait_ns = 3_500_000, .split = .{ 5, 15 }, .adopt = .{ 7, 11 } };
    const pool = [_]Extra{ x0, x0, x0, x1 };
    var src: Counters(Stats) = .{ .seq = &seq, .pool = &pool };
    // the warm-up (a prompt pass, a decode), then the request: two prompt chunks, eleven decode forwards
    m.mark(true, &src);
    m.mark(false, &src);
    m.mark(true, &src);
    m.mark(true, &src);
    for (0..11) |_| m.mark(false, &src);
    try testing.expectEqual(@as(usize, 4), src.at);
    try testing.expectEqual(@as(usize, 4), src.pool_at);
    const text = m.lines(s3, x2);
    var it = std.mem.splitScalar(u8, text, '\n');
    const p = it.next().?;
    const d = it.next().?;
    try testing.expect(std.mem.startsWith(u8, p, "[stream-stats] prompt: routes=42 "));
    try testing.expect(std.mem.indexOf(u8, p, " misses=500 ") != null and std.mem.indexOf(u8, p, " read_gb=3.000 ") != null);
    try testing.expect(std.mem.startsWith(u8, d, "[stream-stats] decode: routes=420 "));
    try testing.expect(std.mem.indexOf(u8, d, " misses=80 ") != null and std.mem.indexOf(u8, d, " read_wait_ms=1.5 ") != null);
    // the decode's lookahead: issued / landed / claimed, then why the unclaimed were not claimed, the claimed in flight
    // and the bytes, each since the decode's first forward
    try testing.expect(std.mem.indexOf(u8, d, " spec=100/0/30 ") != null);
    try testing.expect(std.mem.indexOf(u8, d, " gates_forced=0 spec_expired=50 spec_abandoned=8 spec_discarded=3 spec_cancelled=9 spec_inflight=12 spec_gb=1.900 spec_abandoned_gb=0.320 forwards=") != null);
    try testing.expect(std.mem.indexOf(u8, p, "spec_expired=") == null and std.mem.indexOf(u8, p, "adopt_copy_ms=") == null);
    if (comptime @hasField(Stats, "adopt_waits")) {
        try testing.expect(std.mem.indexOf(u8, d, " forwards=11 adopt_copy_ms=12.5 ") != null);
        try testing.expect(std.mem.indexOf(u8, d, "pool_adopt_waits=") == null);
    } else {
        try testing.expect(std.mem.endsWith(u8, d, " forwards=11 adopt_copy_ms=12.5 pool_adopt_waits=3 pool_adopt_wait_ms=2.5 split_pieces=4/12 adopt_pieces=5/9"));
    }
    try testing.expect(std.mem.indexOf(u8, p, "pool_adopt_waits=") == null);
    try testing.expectEqualStrings("", it.next().?);
    // a request that ends in its prompt has the prompt's line only
    var m2: Meter(Stats) = .{};
    const seq2 = [_]Stats{s1};
    const pool2 = [_]Extra{x0};
    var src2: Counters(Stats) = .{ .seq = &seq2, .pool = &pool2 };
    m2.mark(true, &src2);
    const one = m2.lines(s2, x1);
    try testing.expect(std.mem.startsWith(u8, one, "[stream-stats] prompt: routes=42 ") and std.mem.count(u8, one, "\n") == 1);
}
