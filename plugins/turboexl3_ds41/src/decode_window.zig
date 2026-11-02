//! `MLX_SERVE_DSV41_DECODE_WINDOW` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything
//! else is refused by name): decode's window 0 at `window_rows` rows, not the widest decode route's 48, its memory the
//! bank's decode slot rows.
//!
//! At the phase change the stream frees the prompt's transient scratch and allocates decode's window 0: the rows a
//! decode plan's transient loads take, sized for the widest decode route (48 ids, one window a geometry: 48 x (13.3 + 8.9
//! MB) = 1.066 GB on the 2.81 bpw pack, billed before the decode slot rows). A transient load is a miss the decode
//! admission declines (it ranks below every resident it could evict); the 16K card's greedy decode declines none (ed1's
//! b1: 4,165 misses, 4,165 evictions), so the window holds nothing. On, window 0 holds `window_rows` rows, and a decode
//! plan admits its declined misses past them over the lowest-ranked residents the route does not use
//! (`LayerPolicy.admitPast`); the bill gives the freed bytes to the decode slot rows (about two more a layer). Taken only
//! where every layer's decode rows are 2 * 48 or more (a resident is always left to take), else the window stays the
//! widest route's. `check` also counts the plans left past the window (`overflow`, 0 by construction). Exact: a route's
//! bytes are the records' wherever they sit; only which rows hold them moves, and only on a route past the window. The
//! `[mem-decode]` line (each request's end, either mode) and the `[bank-bill]` load line give the memory it frees and
//! the decode's own: the bill's terms, MLX's active, cache and peak bytes over the request's decode, the process
//! footprint, the declined, forced and overflow counts.

const std = @import("std");
const builtin = @import("builtin");

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .off;

/// Window 0's rows on: room for a few declined misses a plan before any is admitted.
pub const window_rows: u32 = 4;

pub const Error = error{DecodeWindowSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.DecodeWindowSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// Window 0's rows for the bill and the stream: `window_rows` on, 0 off (the widest decode route's).
pub fn rows() u32 {
    return if (on()) window_rows else 0;
}

/// The `[mem-decode]` line: MLX's bytes over the request's decode (active and cache now, the peak since its first
/// round), the process footprint, every layer's decode plans' declined misses, those admitted past window 0, the plans
/// left past it, and window 0's rows. MiB for bytes.
pub const Snap = struct { active: u64, cache: u64, peak: u64, footprint: u64, declined: u64, forced: u64, overflow: u64, window: u32 };

/// The process's physical footprint (`proc_pid_rusage`'s `ri_phys_footprint`, rusage_info_v2's 8th counter); 0 where the
/// platform does not say.
pub fn footprint() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var info: RusageInfoV2 = undefined;
    if (proc_pid_rusage(std.c.getpid(), rusage_info_v2, &info) != 0) return 0;
    return info.v[7];
}
/// `struct rusage_info_v2`: the uuid, then 18 counters, `ri_phys_footprint` the 8th (src/round_io.zig reads the same).
const RusageInfoV2 = extern struct { uuid: [16]u8, v: [18]u64 };
const rusage_info_v2: c_int = 2;
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;

pub fn meters(s: Snap, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[mem-decode] mode={t} window={d} declined={d} forced={d} overflow={d} mlx_active_mb={d} mlx_cache_mb={d} mlx_peak_mb={d} footprint_mb={d}\n", .{ mode, s.window, s.declined, s.forced, s.overflow, s.active >> 20, s.cache >> 20, s.peak >> 20, s.footprint >> 20 }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "decode window setting: unset, empty or 1 on; 0 off (the widest route's window); check; anything else refused by name" {
    const saved = mode;
    defer mode = saved;
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "4", "Check", "1 " }) |v| try testing.expectError(error.DecodeWindowSetting, parse(v));
    mode = .off;
    try testing.expectEqual(@as(u32, 0), rows());
    mode = .on;
    try testing.expectEqual(window_rows, rows());
    mode = .check;
    try testing.expectEqual(window_rows, rows());
}

test "decode window meters: the counts and MiB; none off" {
    const saved = mode;
    defer mode = saved;
    var buf: [256]u8 = undefined;
    const s: Snap = .{ .active = 3 << 30, .cache = 512 << 20, .peak = 4 << 30, .footprint = 96 << 30, .declined = 2, .forced = 1, .overflow = 0, .window = 4 };
    mode = .off;
    try testing.expectEqualStrings("", meters(s, &buf));
    mode = .on;
    try testing.expectEqualStrings("[mem-decode] mode=on window=4 declined=2 forced=1 overflow=0 mlx_active_mb=3072 mlx_cache_mb=512 mlx_peak_mb=4096 footprint_mb=98304\n", meters(s, &buf));
}
