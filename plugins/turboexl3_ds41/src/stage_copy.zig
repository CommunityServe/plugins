//! `MLX_SERVE_DSV41_STAGE_COPY` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything else
//! is refused by name): a wide prompt call's persistent load whose record a staged window row already holds is filled by
//! copying that row, not by reading the record from the file again.
//!
//! MLX_SERVE_DSV41_STAGE (src/stage.zig) reads the prediction's experts past its seed ranks into the windows the call's
//! stream groups take. The call's own seed (`LayerPolicy.prepareSeed`, over its actual counts) can rank a staged expert
//! inside the persistent room: it is then a seed miss, the plan gives it a persistent row (`planWith`'s held rows serve
//! transient misses only) and the route reads it from the file again while its staged copy sits unused in a window. On
//! the cold code-ctx16384 card (pt6, w-pio-20261102-2311) 618 of 3,371 staged records went unadopted, 147 at layer 0
//! and 134 at layer 1, and the hottest vote bins lost the most (the experts nearest the seed boundary); on the 16K
//! new-text slice 100 of 2,573. The boundary experts sit in the first stream group's window, which no seed group's
//! route takes, so the copy reads from whichever window holds the record.
//!
//! On, a prefill route with staging (`Stream.routeWith`, after its plan) looks up each persistent load it would read
//! (the slot not already holding the record) in the transient windows: a row of the layer's staging that names the
//! expert (staged, read or still reading), or a row whose state is the record ready. Such a load reads nothing; it is
//! filled at its part's wait (`waitGu`, `waitDown`, a flush's settle, or the route's gate): the staged part holding the
//! row waited for if it is still reading (only that part, its window left pending), then the row's bytes copied into
//! the persistent row on the copy threads (`copyParallel`). A route's transient read into a row a pending copy reads
//! from, and a staging post into one, fill that copy first. The staged row counts as adopted. Exact: the plans, slots,
//! victims and protection are the stock's (the plan is untouched); a record is the same bytes in any row, and the
//! copied row holds the file's bytes (its staged pread). A row found holding another record at the copy (never
//! expected) has the record read from the file instead (`lost=`).
//!
//! `check`: as `1`, and each copied record is read again from the file into a scratch row and compared with the
//! persistent row byte for byte (`checked= differ=`: the differing bytes, 0 expected).
//!
//! It also counts, with the switch on, the in-call evictions of a seeded prompt call: a plan's eviction of a resident
//! whose expert a later route of the same call routes (the call's counts name it and no route of the call has routed it
//! yet) reads that record again later in the call. `[stage-copy] calls: evictions= by_seed= later= later_seed=
//! remiss= remiss_mb=`: the plans' evictions, those a seed admission made, those whose victim the call routes later,
//! and the misses of experts an earlier route of the same call evicted (their records read again).
//!
//! `[stage-copy] mode= copied= mb= waited= wait_ms= copy_ms= lost= checked= differ=` at each request's end.

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load; on unless `0`).
pub var mode: Mode = .off;

pub const Error = error{StageCopySetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
    reset();
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.StageCopySetting;
}

pub fn on() bool {
    return mode != .off;
}

// The request's meters, on the model's thread (`meters` clears them).
/// Records copied into persistent rows, and their bytes.
pub var copied: u64 = 0;
pub var bytes: u64 = 0;
/// Staged parts a copy waited for (still reading at the copy), and the host's time in those waits.
pub var waited: u64 = 0;
pub var wait_ns: u64 = 0;
/// The host's time copying.
pub var copy_ns: u64 = 0;
/// Copies whose row held another record at the copy: read from the file instead.
pub var lost: u64 = 0;
/// `check`: records read again and compared, and the bytes that differ.
pub var checked: u64 = 0;
pub var differ: u64 = 0;
/// In-call evictions (see above).
pub var evictions: u64 = 0;
pub var by_seed: u64 = 0;
pub var later: u64 = 0;
pub var later_seed: u64 = 0;
pub var remiss: u64 = 0;
pub var remiss_bytes: u64 = 0;

fn reset() void {
    copied = 0;
    bytes = 0;
    waited = 0;
    wait_ns = 0;
    copy_ns = 0;
    lost = 0;
    checked = 0;
    differ = 0;
    evictions = 0;
    by_seed = 0;
    later = 0;
    later_seed = 0;
    remiss = 0;
    remiss_bytes = 0;
}

/// One plan's eviction in a seeded call: `seed`, a seed admission made it; `routed_later`, the call routes the victim
/// after this route.
pub fn noteEviction(seed: bool, routed_later: bool) void {
    evictions += 1;
    by_seed += @intFromBool(seed);
    later += @intFromBool(routed_later);
    later_seed += @intFromBool(routed_later and seed);
}

/// A miss of an expert an earlier route of the same call evicted: its record of `record_bytes` read again.
pub fn noteRemiss(record_bytes: u64) void {
    remiss += 1;
    remiss_bytes += record_bytes;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] wide prompt calls' persistent loads read from the file, a staged copy of the record or not (MLX_SERVE_DSV41_STAGE_COPY=0)\n", .{}),
        .on => w.print("[turboexl3_ds41] wide prompt calls' persistent loads whose record a staged window row holds filled from that row at their part's wait, not read again (MLX_SERVE_DSV41_STAGE_COPY; =0 reads them); [stage-copy] at each request's end\n", .{}),
        .check => w.print("[turboexl3_ds41] wide prompt calls' persistent loads filled from their staged rows, checked (MLX_SERVE_DSV41_STAGE_COPY=check): each copied record read again from the file into a scratch row and compared byte for byte; [stage-copy] checked= differ= at each request's end\n", .{}),
    }) catch return buf[0..0];
    return w.buffered();
}

/// The request's `[stage-copy]` lines ("" off), then the meters cleared.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    defer reset();
    var w: std.Io.Writer = .fixed(buf);
    const mb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
    const rmb = @as(f64, @floatFromInt(remiss_bytes)) / (1024.0 * 1024.0);
    w.print("[stage-copy] mode={t} copied={d} mb={d:.1} waited={d} wait_ms={d:.1} copy_ms={d:.1} lost={d} checked={d} differ={d}\n", .{ mode, copied, mb, waited, ms(wait_ns), ms(copy_ns), lost, checked, differ }) catch return buf[0..0];
    w.print("[stage-copy] calls: evictions={d} by_seed={d} later={d} later_seed={d} remiss={d} remiss_mb={d:.1}\n", .{ evictions, by_seed, later, later_seed, remiss, rmb }) catch return buf[0..0];
    return w.buffered();
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

const testing = std.testing;

test "stage copy setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", " 1", "true" }) |v| try testing.expectError(error.StageCopySetting, parse(v));
}

test "stage copy meters: the request's counts, then cleared; off prints nothing" {
    const was = mode;
    defer {
        mode = was;
        reset();
    }
    var buf: [512]u8 = undefined;
    mode = .off;
    copied = 3;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .check;
    reset();
    copied = 2;
    bytes = 3 * 1024 * 1024;
    waited = 1;
    wait_ns = 2_500_000;
    copy_ns = 500_000;
    checked = 2;
    noteEviction(true, true);
    noteEviction(false, false);
    noteRemiss(1024 * 1024);
    try testing.expectEqualStrings("[stage-copy] mode=check copied=2 mb=3.0 waited=1 wait_ms=2.5 copy_ms=0.5 lost=0 checked=2 differ=0\n" ++
        "[stage-copy] calls: evictions=2 by_seed=1 later=1 later_seed=1 remiss=1 remiss_mb=1.0\n", meters(&buf));
    try testing.expectEqualStrings("[stage-copy] mode=check copied=0 mb=0.0 waited=0 wait_ms=0.0 copy_ms=0.0 lost=0 checked=0 differ=0\n" ++
        "[stage-copy] calls: evictions=0 by_seed=0 later=0 later_seed=0 remiss=0 remiss_mb=0.0\n", meters(&buf));
}
