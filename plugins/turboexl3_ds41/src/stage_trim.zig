//! `MLX_SERVE_DSV41_STAGE_TRIM` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything else
//! is refused by name): a wide prompt call's staged records its routes do not take let go before the drive reaches them.
//!
//! MLX_SERVE_DSV41_STAGE reads a wide layer's predicted transient loads into its windows ahead of its call (STAGE_NEXT:
//! during the call before it); the call's routes take the staged records they route and the rest are read for nothing.
//! On the cold code-ctx16384 card (pt3, w-pio-20261101-2207) the encoder staged 4,168 records and its routes took 3,373:
//! 795 wasted, 207 of them at layer 0 (of 323) and 115 at layer 1, whose staging is posted at the prompt's start, the
//! first wide layer's routes arriving with part of it still queued in front of their own demand reads. On:
//! - a staged window's records are posted in jobs in rank order (the layout's, hottest first), not file order, so the
//!   jobs the call is likeliest to leave are the last posted, the likeliest still queued when its routes are known;
//! - once the call has its routes and has taken its staged groups (`Stream.trimStaged`, from the wide lane's adoption),
//!   each staged job holding none of the call's experts that no read worker has started is withdrawn from the read pool
//!   (`Pool.withdraw`: its tickets published unread, no row written), its rows let go (a reading row emptied, every row
//!   unpinned, named no longer); a job a worker has started reads as posted.
//! `check` also reads each withdrawn ticket's status back (`withdrawn`: nothing read). `[stage-trim] mode= parts=
//! records= started= verified= bad=` at each request's end, since load. Exact: only reads no route of the call takes are
//! dropped; a record is the same bytes in any row.
//!
//! Read on the M5 Max (tr1, w-pio-20261102-0152, two binaries: a = ctl25 (2ff23cb, zzz8w), b = this code on zzz8w, `=1` in
//! the b arms; a warm server, then a1 b1 b2 a2, each a cold code-ctx16384 card then the new-text 16K slice 3): cold 16K
//! a 1407.6/1407.4 -> b 1409.9/1408.2 tok/s (+0.11%), new text 1496.3/1492.5 -> 1498.7/1493.5 (+0.11%); 152 prompt / 177
//! decode rows in every arm, the texts the same in every arm. No job was withdrawn: every b request's line reads parts=0
//! records=0 (started=0 on the cards, 1 on the slices), so every staged job held one of its call's experts, the waste
//! record by record inside them (`MLX_SERVE_DSV41_STAGE_VOTES` cuts records at the plan); the b arms posted in rank order,
//! their stream stats the a arms' (read 101.75 / 70.41 GB, 6049 misses, 4228 staged, 3441 adopted).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .on;

/// The most experts a layer has that the trim tracks (a larger bank is left untrimmed).
pub const max_experts = 512;

/// Since load, on the model's thread: staged jobs withdrawn and their records; jobs that held none of the call's experts
/// but a worker had started (read as posted); `check`'s withdrawn tickets read back as withdrawn, and any not.
pub var parts: u64 = 0;
pub var records: u64 = 0;
pub var started: u64 = 0;
pub var verified: u64 = 0;
pub var bad: u64 = 0;

pub const Error = error{StageTrimSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.StageTrimSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// The meters' line (`[stage-trim]`), "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[stage-trim] mode={t} parts={d} records={d} started={d} verified={d} bad={d}\n", .{ mode, parts, records, started, verified, bad }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "stage trim setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", "1 ", "true" }) |v| try testing.expectError(error.StageTrimSetting, parse(v));
}

test "stage trim meters: the counts; none off" {
    const saved = .{ mode, parts, records, started, verified, bad };
    defer {
        mode = saved[0];
        parts = saved[1];
        records = saved[2];
        started = saved[3];
        verified = saved[4];
        bad = saved[5];
    }
    var buf: [160]u8 = undefined;
    mode = .off;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .check;
    parts = 3;
    records = 12;
    started = 1;
    verified = 24;
    bad = 0;
    try testing.expectEqualStrings("[stage-trim] mode=check parts=3 records=12 started=1 verified=24 bad=0\n", meters(&buf));
}
