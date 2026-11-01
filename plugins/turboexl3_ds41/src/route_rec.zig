//! The decode route recorder's setting and its steps' token log (`MLX_SERVE_DSV41_ROUTES`), MLX-free. The recorder is
//! mlx-serve's `expert.routes` (the copied stream records each decode route, its plan's flags and the lookahead's
//! records; patches/turboexl3-zz7-route-rec.patch arms it): it compiles into route-recording builds only
//! (`zig build -Droute-rec=true`, the root's `route_recording`), so a served build has no recorder branch. At each
//! request's end the plugin writes, in the setting's directory:
//! - `ds41-routes-<pid>-<seq>.bin`: the recorder's file (`expert.routes`' format: the handover's residents and prompt
//!   counts, then the steps and routes);
//! - `ds41-routes-<pid>-<seq>.steps.txt`: one line a recorder step, in its order: the step's kind (`stage`, `copy` or
//!   `alone` for a DSpark round, `fwd` for a plain forward that routed), the position of its first row, its rows' tokens
//!   (a round's: t1, then its drafts; a plain forward's are not kept), the drafts accepted and the tokens emitted.

const std = @import("std");

pub const max_path = 1024;
/// A round's rows at most (t1 and its drafts).
pub const max_rows = 16;

pub const Setting = struct {
    dir_buf: [max_path]u8 = undefined,
    dir_len: usize = 0,

    /// The directory the files go to; null: the recorder is off.
    pub fn dirPath(s: *const Setting) ?[]const u8 {
        return if (s.dir_len > 0) s.dir_buf[0..s.dir_len] else null;
    }
};

/// Set at load.
pub var setting: Setting = .{};

pub const Error = error{ RoutesSetting, RoutesNotBuilt };

/// `value` (null or empty: off): an absolute directory. `built`: the build has the recorder; a set value in a build
/// without it is refused by name, as is a relative or overlong path.
pub fn parse(value: ?[]const u8, built: bool) Error!Setting {
    var s: Setting = .{};
    const v = value orelse return s;
    if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return s;
    if (!built) return error.RoutesNotBuilt;
    if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.RoutesSetting;
    @memcpy(s.dir_buf[0..v.len], v);
    s.dir_len = v.len;
    return s;
}

/// The steps' token log of one request (`steps.txt`), in the recorder's step order.
pub const Log = struct {
    buf: std.ArrayList(u8) = .empty,
    /// The open round (`round`): its kind, first row's position and rows' tokens.
    kind: []const u8 = "fwd",
    pos: usize = 0,
    rows: [max_rows]u32 = undefined,
    n_rows: usize = 0,
    steps: u32 = 0,

    /// A request's start: the log emptied, its first line the request's prompt and the prefix its state kept.
    pub fn begin(l: *Log, a: std.mem.Allocator, prompt: usize, kept: usize) !void {
        l.buf.clearRetainingCapacity();
        l.steps = 0;
        l.close();
        try l.buf.print(a, "request prompt={d} kept={d}\n", .{ prompt, kept });
    }
    /// A round's verify rows (t1, then its drafts) at position `pos`, before it runs.
    pub fn round(l: *Log, kind: []const u8, pos: usize, rows: []const u32) void {
        const n = @min(rows.len, max_rows);
        l.kind = kind;
        l.pos = pos;
        @memcpy(l.rows[0..n], rows[0..n]);
        l.n_rows = n;
    }
    /// The step the recorder just closed: the open round's (`round`; none open: a plain forward at `pos`, its rows not
    /// kept), its accepted drafts and emitted tokens.
    pub fn step(l: *Log, a: std.mem.Allocator, pos: usize, accepted: usize, emitted: []const u32) !void {
        defer l.close();
        const at = if (l.n_rows > 0) l.pos else pos;
        try l.buf.print(a, "step={d} kind={s} pos={d} rows=", .{ l.steps, l.kind, at });
        try ints(&l.buf, a, l.rows[0..l.n_rows]);
        try l.buf.print(a, " accepted={d} emitted=", .{accepted});
        try ints(&l.buf, a, emitted);
        try l.buf.append(a, '\n');
        l.steps += 1;
    }
    fn close(l: *Log) void {
        l.kind = "fwd";
        l.pos = 0;
        l.n_rows = 0;
    }
    pub fn deinit(l: *Log, a: std.mem.Allocator) void {
        l.buf.deinit(a);
    }
};

fn ints(buf: *std.ArrayList(u8), a: std.mem.Allocator, xs: []const u32) !void {
    for (xs, 0..) |x, i| try buf.print(a, "{s}{d}", .{ if (i == 0) "" else ",", x });
}

/// `<dir>/ds41-routes-<pid>-<seq><ext>` into `buf`.
pub fn filePath(buf: []u8, dir: []const u8, pid: i64, seq: u32, ext: []const u8) error{NoSpaceLeft}![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/ds41-routes-{d}-{d}{s}", .{ dir, pid, seq, ext });
}

// ── tests ──

const testing = std.testing;

test "route recorder setting: off unless set; an absolute directory in a recording build; refused by name otherwise" {
    for ([_]?[]const u8{ null, "", "0", "off" }) |v| {
        try testing.expect((try parse(v, false)).dirPath() == null);
        try testing.expect((try parse(v, true)).dirPath() == null);
    }
    try testing.expectEqualStrings("/data/routes", (try parse("/data/routes", true)).dirPath().?);
    try testing.expectError(error.RoutesNotBuilt, parse("/data/routes", false));
    try testing.expectError(error.RoutesSetting, parse("routes", true));
    var long: [max_path + 1]u8 = @splat('a');
    long[0] = '/';
    try testing.expectError(error.RoutesSetting, parse(&long, true));
}

test "route recorder log: a line a step, a round's rows and a plain forward's none" {
    const a = testing.allocator;
    var l: Log = .{};
    defer l.deinit(a);
    try l.begin(a, 16384, 0);
    l.round("stage", 16384, &.{ 7, 8, 9 });
    try l.step(a, 0, 1, &.{ 8, 11 });
    try l.step(a, 16386, 0, &.{});
    l.round("copy", 16386, &.{12});
    try l.step(a, 0, 0, &.{13});
    try testing.expectEqualStrings(
        \\request prompt=16384 kept=0
        \\step=0 kind=stage pos=16384 rows=7,8,9 accepted=1 emitted=8,11
        \\step=1 kind=fwd pos=16386 rows= accepted=0 emitted=
        \\step=2 kind=copy pos=16386 rows=12 accepted=0 emitted=13
        \\
    , l.buf.items);
    try l.begin(a, 5, 5);
    try testing.expectEqualStrings("request prompt=5 kept=5\n", l.buf.items);
    var pb: [64]u8 = undefined;
    try testing.expectEqualStrings("/r/ds41-routes-42-3.bin", try filePath(&pb, "/r", 42, 3, ".bin"));
}
