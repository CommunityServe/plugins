//! `MLX_SERVE_DSV41_PROMPT_TL` (profile builds only, `zig build -Dplugin-profile=true`; an absolute directory, read at
//! load; needs `MLX_SERVE_DSV41_PROMPT_IO`, refused with `MLX_SERVE_DSV41_TIMELINE`): a prompt pass's GPU timeline. From
//! the prompt's first stage to its end every command buffer MLX commits is recorded by the copied stream's verify
//! timeline (its shim: host commit, Metal's GPU start and end; the read pool's event signals), tagged with the layer and
//! the host's stage that committed it, and the host's stage marks are kept; at the request's end the plugin writes
//! `<dir>/ds41-prompt-tl-<pid>-<seq>.json`: `{"timeline": <buffers (their cycle is the layer, their phase the stage),
//! signals>, "stages": [...], "origin_us": ..., "mark_fields": [...], "marks": [...]}`. No fence is added: each mark
//! sits at a hook `[prompt-io]` already has (`MLX_SERVE_DSV41_PROMPT_IO`), at an eval the pass makes anyway.
//!
//! Stages (each from its host mark to the next; a stage's GPU time is its buffers', wherever they run):
//!   pre   a wide layer's start to its routed call: its input (the Engram rows read on layers 1 and 14), the read-ahead
//!         (the stream's rows, their collapse and the predictor's routes read back; the decoder probe's routers when it
//!         runs) and the graph built until the call
//!   attn  the routed call's barrier: its routes evaluated (the attention, wo_a and wo_b, the FFN's mix, the router)
//!   moe   the routed call: its groups' reads and waves, each drained
//!   rest  from the call's end to the next layer's start: the layer's last eval (the routed tail, the shared expert, the
//!         HC expansion); a narrow layer's (the CED decoder's) graph until its own call
//! The last mark (`end`) is the prompt's end (the source's phase change to decode); no buffer is recorded after it.

const std = @import("std");

pub const max_path = 1024;

pub const Setting = struct {
    dir_buf: [max_path]u8 = undefined,
    dir_len: usize = 0,

    /// The directory the files go to; null: off.
    pub fn dirPath(s: *const Setting) ?[]const u8 {
        return if (s.dir_len > 0) s.dir_buf[0..s.dir_len] else null;
    }
};

/// Set at load.
pub var setting: Setting = .{};

pub const Error = error{ PromptTimelineSetting, PromptTimelineNotBuilt };

/// `value` (null, empty, `0` or `off`: off): an absolute directory. `built`: a profile build; a set value in another
/// build is refused by name, as is a relative or overlong path.
pub fn parse(value: ?[]const u8, built: bool) Error!Setting {
    var s: Setting = .{};
    const v = value orelse return s;
    if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return s;
    if (!built) return error.PromptTimelineNotBuilt;
    if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.PromptTimelineSetting;
    @memcpy(s.dir_buf[0..v.len], v);
    s.dir_len = v.len;
    return s;
}

/// The host's stages, in the file's `stages` order (0: the prompt's end).
pub const Stage = enum(u8) { end = 0, pre = 1, attn = 2, moe = 3, rest = 4 };
pub const stage_names = [_][]const u8{ "end", "pre", "attn", "moe", "rest" };

/// The plugin's hook (set at load when the setting is on): a stage of `layer` starts now. Null: off.
pub var hook: ?*const fn (layer: u32, stage: Stage) void = null;

pub fn stage(layer: usize, s: Stage) void {
    if (hook) |f| f(@intCast(layer), s);
}

/// One host mark: its time (the timeline's clock, ns), the layer and the stage that starts.
pub const Mark = struct { t: u64, layer: u32, stage: Stage };

pub const max_marks = 4096;

/// A request's marks; past `max_marks` they are counted, not kept.
pub const Marks = struct {
    items: [max_marks]Mark = undefined,
    n: u32 = 0,
    dropped: u32 = 0,

    pub fn reset(m: *Marks) void {
        m.n = 0;
        m.dropped = 0;
    }
    pub fn add(m: *Marks, t: u64, layer: u32, s: Stage) void {
        if (m.n == max_marks) {
            m.dropped += 1;
            return;
        }
        m.items[m.n] = .{ .t = t, .layer = layer, .stage = s };
        m.n += 1;
    }
    pub fn slice(m: *const Marks) []const Mark {
        return m.items[0..m.n];
    }
};

/// The file's `stages`, `origin_us`, `mark_fields`, `marks` and `marks_dropped` members (no braces), times in us from
/// `origin` (the timeline's install, as its buffers' times are).
pub fn writeMarks(w: *std.Io.Writer, ms: []const Mark, dropped: u32, origin: u64) !void {
    try w.writeAll("\"stages\": [");
    for (stage_names, 0..) |s, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", s });
    try w.print("], \"origin_us\": {d:.1}, \"mark_fields\": [\"us\", \"layer\", \"stage\"], \"marks\": [", .{us(origin)});
    for (ms, 0..) |m, i| try w.print("{s}[{d:.1}, {d}, {d}]", .{ if (i == 0) "" else ", ", us(m.t -| origin), m.layer, @intFromEnum(m.stage) });
    try w.print("], \"marks_dropped\": {d}", .{dropped});
}

fn us(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

/// `<dir>/ds41-prompt-tl-<pid>-<seq>.json` into `buf`.
pub fn filePath(buf: []u8, dir: []const u8, pid: i64, seq: u32) error{NoSpaceLeft}![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/ds41-prompt-tl-{d}-{d}.json", .{ dir, pid, seq });
}

// ── tests ──

const testing = std.testing;

test "prompt timeline setting: off unless set; an absolute directory in a profile build; refused by name otherwise" {
    for ([_]?[]const u8{ null, "", "0", "off" }) |v| {
        try testing.expect((try parse(v, false)).dirPath() == null);
        try testing.expect((try parse(v, true)).dirPath() == null);
    }
    try testing.expectEqualStrings("/data/ptl", (try parse("/data/ptl", true)).dirPath().?);
    try testing.expectError(error.PromptTimelineNotBuilt, parse("/data/ptl", false));
    try testing.expectError(error.PromptTimelineSetting, parse("ptl", true));
    var long: [max_path + 1]u8 = @splat('a');
    long[0] = '/';
    try testing.expectError(error.PromptTimelineSetting, parse(&long, true));
}

test "prompt timeline marks: kept in order from the install, counted past the cap, written as the file's members" {
    var m: Marks = .{};
    m.add(5_000, 0, .pre);
    m.add(7_500, 0, .attn);
    m.add(9_000, 39, .end);
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeMarks(&w, m.slice(), m.dropped, 1_000);
    try testing.expectEqualStrings("\"stages\": [\"end\", \"pre\", \"attn\", \"moe\", \"rest\"], \"origin_us\": 1.0, \"mark_fields\": [\"us\", \"layer\", \"stage\"], \"marks\": [[4.0, 0, 1], [6.5, 0, 2], [8.0, 39, 0]], \"marks_dropped\": 0", w.buffered());
    m.n = max_marks;
    m.add(10_000, 1, .moe);
    try testing.expectEqual(@as(u32, 1), m.dropped);
    m.reset();
    try testing.expectEqual(@as(usize, 0), m.slice().len);
    var pb: [64]u8 = undefined;
    try testing.expectEqualStrings("/t/ds41-prompt-tl-42-3.json", try filePath(&pb, "/t", 42, 3));
}
