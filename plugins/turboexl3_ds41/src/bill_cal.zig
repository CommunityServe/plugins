//! `MLX_SERVE_DSV41_BILL_CAL` (read at load; unset, empty or `1`: on; `0`: off, the bill as before; anything else is
//! refused by name): the streamed bank's bill calibrated against what the box measured (David 2026-11-04: "we need an
//! accurate calibration of memory estimates so we can fit in the max amount of memory for performance"; the model over
//! eight windows and 172 servers, w-20261104-0419 .. w-20261104-1515). Each phase's budget is MLX's recommended working
//! set less `margin_bytes`, the memory in use (the window's live pin: its first server's reading) and the phase's
//! measured working set at the bank's context:
//!   prompt: `wp(context)`, a prompt of `context` tokens: the prompt phase's peak over its bank's bytes and the load's
//!           non-bank memory (one 16K call's working set plus what grows with the context);
//!   decode: `wd(context)`, a decode at `context` positions: the decode phase's peak over the load's footprint.
//! The terms the bill reserved beside its own working sets (the wired margin, the host's bytes, the Engram LRU's, the
//! decode stop's cut) are inside what was measured (the load's footprint holds them, or the phase's peak does), so they
//! drop out. `margin_bytes` is the stated safety under the guard's 112 GiB less the smallest over-read a live reading
//! showed of the box's non-file memory and the load's non-bank memory: 1.5 GiB - 1.1 GiB = 0.4 GiB (a window's own
//! over-read adds to it). The measured points hold for the cell form they came from: `--prefill-chunk 16384`, MLX's
//! cache limit 2 GiB (MLX_SERVE_CACHE_LIMIT), the DSpark profile's env; other settings need their own points, so the
//! calibration engages only on the server it was measured on: `--prefill-chunk 16384` given, MLX_SERVE_CACHE_LIMIT 2 GiB,
//! an Apple M5 Max with 128 GB and a 112 GiB wired limit (iogpu.wired_limit_mb 114688); with any of these different the
//! bill is as before and the load line names what differs (`unmeasured`). Exact: only the row counts move (a route's
//! bytes are the records' in any row).

const std = @import("std");

pub const Mode = enum { off, on, unmeasured };
/// Set at load (`configure`), before the bank opens.
pub var mode: Mode = .off;

pub const Error = error{BillCalSetting};

const mib: u64 = 1 << 20;

/// The safety the code keeps: S (1.5 GiB) less the live reading's smallest over-read (1.1 GiB), MiB rounded.
pub const margin_bytes: u64 = 410 * mib;

/// One measured point: tokens and MiB.
pub const Point = struct { tokens: u64, mib: u64 };

/// The prompt phase's working set by prompt tokens (after the comp-once fix; 2.81 at the ladder's sizes, the other
/// packs' within 0.2 GiB before it): card peak - prompt bank - (load peak - decode bank).
pub const wp_points = [_]Point{
    .{ .tokens = 16384, .mib = 12616 },
    .{ .tokens = 32768, .mib = 13373 },
    .{ .tokens = 65536, .mib = 13804 },
    .{ .tokens = 131072, .mib = 13599 },
    .{ .tokens = 262144, .mib = 13926 },
};

/// The decode phase's working set by context tokens: the peak over the load's footprint (up to 8K, where decode binds)
/// and the footprint after a request plus 0.1 GiB above it (from 16K, where the prompt's peak hides decode's).
pub const wd_points = [_]Point{
    .{ .tokens = 1024, .mib = 164 },
    .{ .tokens = 2048, .mib = 215 },
    .{ .tokens = 4096, .mib = 276 },
    .{ .tokens = 8192, .mib = 430 },
    .{ .tokens = 16384, .mib = 686 },
    .{ .tokens = 32768, .mib = 1096 },
    .{ .tokens = 65536, .mib = 1403 },
    .{ .tokens = 131072, .mib = 1761 },
    .{ .tokens = 262144, .mib = 2642 },
};

/// The table's MiB at `n` tokens: linear between the points, never below a point at or under `n` (the prompt's table
/// dips from 64K to 128K), the first point's value under it, the last segment's slope past the end (never below the last
/// point).
pub fn at(points: []const Point, n: u64) u64 {
    var floor: u64 = 0;
    for (points) |p| {
        if (p.tokens <= n) floor = @max(floor, p.mib);
    }
    if (n <= points[0].tokens) return points[0].mib;
    var lin: u64 = 0;
    var i: usize = 1;
    while (i < points.len) : (i += 1) {
        const a = points[i - 1];
        const b = points[i];
        if (n <= b.tokens or i == points.len - 1) {
            const span: i128 = @intCast(b.tokens - a.tokens);
            const dv: i128 = @as(i128, @intCast(b.mib)) - @as(i128, @intCast(a.mib));
            const dn: i128 = @as(i128, @intCast(n)) - @as(i128, @intCast(a.tokens));
            const v: i128 = @as(i128, @intCast(a.mib)) + @divFloor(dv * dn + span - 1, span);
            lin = if (v < 0) 0 else @intCast(v);
            break;
        }
    }
    return @max(lin, floor);
}

/// The prompt phase's working set for a prompt of `context` tokens, bytes (a prompt under one 16K call billed as one).
pub fn wp(context: u64) u64 {
    return at(&wp_points, @max(context, wp_points[0].tokens)) * mib;
}

/// The decode phase's working set at `context` positions, bytes.
pub fn wd(context: u64) u64 {
    return at(&wd_points, context) * mib;
}

/// The generated bill's terms (`bill.CalTerms`): the margin and each phase's working set at `context`.
pub const Terms = struct { margin: u64, wp: u64, wd: u64 };

pub fn terms(context: u64) Terms {
    return .{ .margin = margin_bytes, .wp = wp(context), .wd = wd(context) };
}

/// `value`: null, empty or `1` on; `0` off; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.BillCalSetting;
}

/// What the calibration was measured with (the box: an Apple M5 Max with 128 GB at a 112 GiB wired limit; the cells:
/// `--prefill-chunk 16384`, MLX_SERVE_CACHE_LIMIT 2 GiB), and what a server reads of each at its load (null: not read).
pub const Conditions = struct {
    prefill_chunk: ?u64,
    cache_limit: ?u64,
    memsize: ?u64,
    chip: ?[]const u8,
    wired_limit_mb: ?u64,
};
pub const measured: Conditions = .{ .prefill_chunk = 16384, .cache_limit = 2 << 30, .memsize = 128 << 30, .chip = "Apple M5 Max", .wired_limit_mb = 114688 };

var why_buf: [256]u8 = undefined;
/// Why the calibration is off on this server (`unmeasured`), for the load line.
pub var why: []const u8 = "";

/// null when `c` is the calibration's conditions, else what differs, written into `buf`.
pub fn mismatch(c: Conditions, buf: []u8) ?[]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const m = measured;
    if (c.prefill_chunk == null or c.prefill_chunk.? != m.prefill_chunk.?) {
        if (c.prefill_chunk) |v| {
            w.print("--prefill-chunk {d}, not 16384; ", .{v}) catch {};
        } else {
            w.print("no --prefill-chunk 16384; ", .{}) catch {};
        }
    }
    if (c.cache_limit == null or c.cache_limit.? != m.cache_limit.?) {
        if (c.cache_limit) |v| {
            w.print("MLX_SERVE_CACHE_LIMIT {d} B, not 2 GiB; ", .{v}) catch {};
        } else {
            w.print("MLX_SERVE_CACHE_LIMIT unset or not bytes, not 2 GiB; ", .{}) catch {};
        }
    }
    if (c.memsize == null or c.memsize.? != m.memsize.?) {
        w.print("hw.memsize {d} B, not 128 GiB; ", .{c.memsize orelse 0}) catch {};
    }
    if (c.chip == null or !std.mem.eql(u8, c.chip.?, m.chip.?)) {
        w.print("chip {s}, not an Apple M5 Max; ", .{c.chip orelse "unread"}) catch {};
    }
    if (c.wired_limit_mb == null or c.wired_limit_mb.? != m.wired_limit_mb.?) {
        w.print("iogpu.wired_limit_mb {d}, not 114688; ", .{c.wired_limit_mb orelse 0}) catch {};
    }
    const out = w.buffered();
    if (out.len == 0) return null;
    return if (std.mem.endsWith(u8, out, "; ")) out[0 .. out.len - 2] else out;
}

/// `value` as `parse` reads it; on, the server's `conditions` must be the calibration's, else `unmeasured` (the bill as
/// before) with `why` naming what differs.
pub fn configure(value: ?[]const u8, conditions: Conditions) Error!void {
    mode = try parse(value);
    why = "";
    if (mode == .on) {
        if (mismatch(conditions, &why_buf)) |w| {
            mode = .unmeasured;
            why = w;
        }
    }
}

pub fn on() bool {
    return mode == .on;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] the streamed bank's bill as before: the wired margin, the host's and the Engram LRU's bytes and each phase's estimated working set (MLX_SERVE_DSV41_BILL_CAL=0)\n", .{}),
        .unmeasured => w.print("[turboexl3_ds41] the streamed bank's bill as before (MLX_SERVE_DSV41_BILL_CAL on, but the calibration's conditions are not this server's: {s})\n", .{why}),
        .on => w.print("[turboexl3_ds41] the streamed bank's bill calibrated (MLX_SERVE_DSV41_BILL_CAL; =0: as before): each phase's budget the recommended working set less {d} MiB, the memory in use and the phase's measured working set at the context (prompt {d}..{d} MiB, decode {d}..{d} MiB); [bank-bill] cal=on\n", .{ margin_bytes >> 20, wp_points[0].mib, wp_points[wp_points.len - 1].mib, wd_points[0].mib, wd_points[wd_points.len - 1].mib }),
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "bill cal setting: unset, empty or 1 on; 0 off; anything else refused by name" {
    const was = mode;
    defer mode = was;
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    for ([_][]const u8{ "on", "2", "cal", " 1" }) |v| try testing.expectError(error.BillCalSetting, parse(v));
    try configure("0", measured);
    try testing.expect(!on());
    try configure(null, measured);
    try testing.expect(on());
}

test "bill cal tables: the points (or a higher earlier one), linear between, never below an earlier point, the last slope past the end" {
    // at each point its value, or an earlier point's when that is higher (the prompt's 128K point: the 64K point's 13804)
    var run: u64 = 0;
    for (wp_points) |p| {
        run = @max(run, p.mib);
        try testing.expectEqual(run, at(&wp_points, p.tokens));
    }
    try testing.expectEqual(@as(u64, 13804), at(&wp_points, 131072));
    run = 0;
    for (wd_points) |p| {
        run = @max(run, p.mib);
        try testing.expectEqual(run, at(&wd_points, p.tokens));
    }
    // under one call: one call's working set
    try testing.expectEqual(@as(u64, 12616) * mib, wp(1024));
    try testing.expectEqual(@as(u64, 12616) * mib, wp(16384));
    // between 16K and 32K: a 17408-token context (the 1K cell) bills 12616 + 757 * 1024 / 16384, rounded up
    try testing.expectEqual(@as(u64, 12664), at(&wp_points, 17408));
    // 64K to 128K dips (13804 -> 13599): the 64K point holds
    try testing.expectEqual(@as(u64, 13804), at(&wp_points, 81920));
    try testing.expectEqual(@as(u64, 13804), at(&wp_points, 147456));
    // from about 213K the 128K-256K line passes the 64K point
    try testing.expectEqual(@as(u64, 13859), at(&wp_points, 235000));
    // past 256K: the last segment's slope (327 MiB a 131072 tokens) from the 128K point: 278528 -> 13599 + 368 = 13967
    try testing.expectEqual(@as(u64, 13967), at(&wp_points, 278528));
    // decode: under the first point its value; between points linear
    try testing.expectEqual(@as(u64, 164), at(&wd_points, 512));
    try testing.expectEqual(@as(u64, 712), at(&wd_points, 17408));
    try testing.expectEqual(@as(u64, 2753), at(&wd_points, 278528));
}

test "bill cal rows: the model's rows for the 2.81 pack at the 16K and 256K cells" {
    // the 2.81 pack: a row 492810240 B, the prompt's windows 5082 MiB, decode's 84, the staging 102 (its bill lines);
    // the recommended working set 114688 MiB; a window's pin at 26450 MiB (B 10.6 + Res 13.36 + the live over-read 1.87)
    const row: u64 = 492810240;
    const rec: u64 = 114688 * mib;
    const in_use: u64 = 26450 * mib;
    const Rows = struct {
        fn of(budget: u64, windows: u64, staging: u64, r: u64) u64 {
            return (budget -| (windows + staging)) / r;
        }
    };
    const t16 = terms(16384 + 16384);
    const box16 = rec - t16.margin - in_use;
    try testing.expectEqual(@as(u64, 147), Rows.of(box16 - t16.wp, 5082 * mib, 102 * mib, row));
    try testing.expectEqual(@as(u64, 184), Rows.of(box16 - t16.wd, 84 * mib, 102 * mib, row));
    const t256 = terms(262144 + 16384);
    const box256 = rec - t256.margin - in_use;
    try testing.expectEqual(@as(u64, 146), Rows.of(box256 - t256.wp, 5082 * mib, 102 * mib, row));
    try testing.expectEqual(@as(u64, 180), Rows.of(box256 - t256.wd, 84 * mib, 102 * mib, row));
}

test "bill cal conditions: on only under the calibration's; each difference named; the bill as before" {
    const was = mode;
    defer mode = was;
    var buf: [256]u8 = undefined;
    try testing.expect(mismatch(measured, &buf) == null);
    try configure(null, measured);
    try testing.expectEqual(Mode.on, mode);
    var c = measured;
    c.prefill_chunk = null;
    try testing.expectEqualStrings("no --prefill-chunk 16384", mismatch(c, &buf).?);
    c = measured;
    c.prefill_chunk = 8192;
    try testing.expectEqualStrings("--prefill-chunk 8192, not 16384", mismatch(c, &buf).?);
    c = measured;
    c.cache_limit = 4 << 30;
    c.chip = "Apple M4 Max";
    try testing.expectEqualStrings("MLX_SERVE_CACHE_LIMIT 4294967296 B, not 2 GiB; chip Apple M4 Max, not an Apple M5 Max", mismatch(c, &buf).?);
    c = measured;
    c.memsize = 64 << 30;
    c.wired_limit_mb = null;
    try configure("1", c);
    try testing.expectEqual(Mode.unmeasured, mode);
    try testing.expect(!on());
    try testing.expectEqualStrings("hw.memsize 68719476736 B, not 128 GiB; iogpu.wired_limit_mb 0, not 114688", why);
    var lb: [512]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, loadLine(&lb), "conditions are not this server's: hw.memsize") != null);
    try configure("0", c);
    try testing.expectEqual(Mode.off, mode);
}
