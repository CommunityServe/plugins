//! `MLX_SERVE_DSV41_L39_LAST` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything else
//! is refused by name): a replaying prompt chunk's last decoder layer routes its last row only.
//!
//! A replaying chunk (CED, the copied model's `cedTail`) runs the decoder layers (from the last KV source on: 20..39 on
//! the pack) over its last `sliding_window` rows (128). Every decoder layer's output rows are read on by the next
//! layer's window KV (each layer keeps its own ring, which decode reads) and, at the DSpark targets (layers 37, 38, 39:
//! their inputs), by the drafter's rings; the last layer's output rows are read on by nothing but the chunk's logits,
//! which take the last row (`want == .last`). Its window KV comes from its input, as every layer's does. So the last
//! layer's routed call over all 128 rows reads records for 127 rows nothing reads: on the cold code-ctx16384 card (ctl48,
//! the dtf1 window's a arms) layer 39 routes about 138 experts, 61 of them misses (0.68 GB), of the decoder tail's 1060.
//!
//! On, the copied model's `moeLast` serves that layer's MoE: the router and the shared expert over every row as `moe`
//! runs them, the routed call over the last row only (its 6 routes), forced through the wide lane (`force`: the experts'
//! joinless `routedParts`, which a call of at most 48 ids otherwise refuses), the same waves and combine a wide call runs;
//! its row becomes the last row of the layer's routed output (the other rows zero: nothing reads them). The window KV,
//! the drafter's rows and the decode state are the stock ones by construction; the logits are the stock ones when the
//! one-row call's words are the 128-row call's for that row, which `check` measures. The layer's residency differs: its
//! call admits the last row's misses, not the 128 rows'.
//!
//! `check`: as on, and the stock 128-row call made too (first, so the residency is the stock one), its last row compared
//! with the one-row call's word for word (f32 bits); the stock call's served. `[l39-last] mode= calls= words= differ=` at
//! each request's end (calls: one-row calls made; words and differ: compared words and those that differ, check only).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load; on unless `0`).
pub var mode: Mode = .off;

pub const Error = error{L39LastSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
    armed_flag = false;
    force = false;
    resetMeters();
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.L39LastSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// The copied model's `runLayers` arms the replaying chunk's last decoder layer's last span (`block` then takes `moeLast`).
var armed_flag: bool = false;
pub fn arm(v: bool) void {
    armed_flag = v and on();
}
pub fn armed() bool {
    return armed_flag;
}

/// The routed call in flight is `moeLast`'s one-row call, forced through the wide lane (the copied experts' `routedParts`).
pub var force: bool = false;
pub fn forced() bool {
    return force;
}

// The request's meters, on the model's thread (`meters` clears them).
pub var calls: u64 = 0;
pub var words: u64 = 0;
pub var differ: u64 = 0;

fn resetMeters() void {
    calls = 0;
    words = 0;
    differ = 0;
}

/// `check`: one comparison of `n` words, `d` of them differing.
pub fn noteCheck(n: u64, d: u64) void {
    words += n;
    differ += d;
}

/// The differing words of two f32 rows, compared as bits.
pub fn differWords(a: []const f32, b: []const f32) u64 {
    var d: u64 = 0;
    for (a, b) |x, y| d += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    return d;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] a replaying prompt chunk's last decoder layer routes every tail row (MLX_SERVE_DSV41_L39_LAST=0)\n", .{}),
        .on => w.print("[turboexl3_ds41] a replaying prompt chunk's last decoder layer routes its last row only, through the wide lane (MLX_SERVE_DSV41_L39_LAST; =0 routes every tail row); [l39-last] at each request's end\n", .{}),
        .check => w.print("[turboexl3_ds41] a replaying prompt chunk's last decoder layer: its 128-row call served, its last row routed again alone through the wide lane and compared word for word (MLX_SERVE_DSV41_L39_LAST=check); [l39-last] differ= at each request's end\n", .{}),
    }) catch return buf[0..0];
    return w.buffered();
}

/// The request's `[l39-last]` line ("" off), then the meters cleared.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    defer resetMeters();
    var w: std.Io.Writer = .fixed(buf);
    w.print("[l39-last] mode={t} calls={d} words={d} differ={d}\n", .{ mode, calls, words, differ }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "l39 last setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "39", "Check", " 1", "true" }) |v| try testing.expectError(error.L39LastSetting, parse(v));
}

test "l39 last: armed only with the switch on; words compared as bits" {
    const was = mode;
    defer {
        mode = was;
        armed_flag = false;
        resetMeters();
    }
    mode = .off;
    arm(true);
    try testing.expect(!armed());
    mode = .on;
    arm(true);
    try testing.expect(armed());
    arm(false);
    try testing.expect(!armed());
    // +0 and -0 are equal as floats, not as words; a NaN equals its own bits.
    const nan: f32 = std.math.nan(f32);
    try testing.expectEqual(@as(u64, 1), differWords(&.{ 0.0, 1.5, nan }, &.{ -0.0, 1.5, nan }));
    try testing.expectEqual(@as(u64, 0), differWords(&.{ 2.0, 3.0 }, &.{ 2.0, 3.0 }));
}

test "l39 last meters: the request's counts, then cleared; off prints nothing" {
    const was = mode;
    defer {
        mode = was;
        resetMeters();
    }
    var buf: [256]u8 = undefined;
    mode = .off;
    calls = 2;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .check;
    resetMeters();
    calls = 1;
    noteCheck(4096, 0);
    try testing.expectEqualStrings("[l39-last] mode=check calls=1 words=4096 differ=0\n", meters(&buf));
    try testing.expectEqualStrings("[l39-last] mode=check calls=0 words=0 differ=0\n", meters(&buf));
}
