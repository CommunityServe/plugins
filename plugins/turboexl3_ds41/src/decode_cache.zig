//! `MLX_SERVE_DSV41_DECODE_CACHE` (read at load, before DECODE_STOP's cut reaches the bill; unset, empty or `1`: on;
//! `0`: off; anything else is refused by name): decode's MLX cache without the prompt's leftovers, its cap and DECODE_STOP's
//! cut sized from decode's own use.
//!
//! A prompt's last chunk lets the SCRATCH_ATTN ring go at the change to decode (its S and P slots, 125.8 MB, and
//! SCRATCH_HEADS' two phase-B run regions, 2 x 268,435,456 B: 632 MiB), but after the phase change (`src.phase`), whose
//! clearCache, grow and DECODE_STOP cap (1 GiB) have run: those buffers land in decode's capped cache and stay until
//! decode's own churn evicts them (MLX's cache at decode 867-884 MiB at a 32-token decode's end, 71-100 at a 524-1024-token
//! one: wm1p, pt7, pt7d, l39d). On: (1) the ring's regions are let go before the phase change, as SCRATCH's are, so its
//! synchronize and clearCache take them; (2) MLX's cache at decode is capped at `cache_bytes` (512 MiB, twice decode's
//! own early use of 235-252 MiB; db1's 256 MiB cap churned with the leftovers still in it) and (3) DECODE_STOP's cut takes
//! `extra_cut` (512 MiB) more of the stop back for decode's slot rows (one more row at 470 MiB). Exact: buffer lifetimes,
//! the cache's size and the decode rows' count move; every op and its words are the same. Decode holds the row and less
//! cache: a prompted request's early decode about 160 MiB under today's, its late decode 470 MiB over (under its prompt's
//! peak: David's "server's peak" ruling); the next prompt allocates its ring fresh (about 10 ms).

const std = @import("std");
const decode_stop = @import("decode_stop.zig");

pub const Mode = enum { off, on };
/// Set at load (`configure`), before the bill reads DECODE_STOP's cut.
pub var mode: Mode = .off;

pub const Error = error{DecodeCacheSetting};

/// MLX's cache cap at decode, on.
pub const cache_bytes: usize = 512 << 20;
/// The stop's bytes the decode budget takes back past DECODE_STOP's, on: what the lower cap frees.
pub const extra_cut: u64 = (1 << 30) - (512 << 20);

/// `value`: null, empty or `1` on; `0` off; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.DecodeCacheSetting;
}

/// At load, before `decode_stop.cutBytes()` reaches the bill: DECODE_STOP's cut and cap moved when on.
pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
    decode_stop.cut_now = decode_stop.cut_bytes + if (mode == .on) extra_cut else 0;
    decode_stop.cache_now = if (mode == .on) cache_bytes else decode_stop.cache_bytes;
}

/// The copied model's: the ring's regions let go before the phase change.
pub fn on() bool {
    return mode == .on;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] the prompt's attention ring let go after the phase change to decode, into decode's cache (MLX_SERVE_DSV41_DECODE_CACHE=0; on: before it, the cap {d} MiB)\n", .{cache_bytes >> 20}),
        .on => w.print("[turboexl3_ds41] the prompt's attention ring let go before the phase change to decode (its clearCache takes it), decode's cache capped at {d} MiB and DECODE_STOP's cut {d} MiB more for slot rows (MLX_SERVE_DSV41_DECODE_CACHE; =0 off)\n", .{ cache_bytes >> 20, extra_cut >> 20 }),
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "decode cache setting: unset, empty or 1 on; 0 off; anything else refused by name; on moves DECODE_STOP's cut and cap" {
    const was = mode;
    defer configure(if (was == .on) null else "0") catch unreachable;
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    for ([_][]const u8{ "on", "512", "check", " 1" }) |v| try testing.expectError(error.DecodeCacheSetting, parse(v));
    try configure("1");
    try testing.expectEqual(@as(u64, 1_000_000_000 + (512 << 20)), decode_stop.cut_now);
    try testing.expectEqual(@as(usize, 512 << 20), decode_stop.cache_now);
    // one more decode row at the pack's 470 MiB, at any remainder
    try testing.expect(extra_cut >= 492_810_240);
    try configure("0");
    try testing.expectEqual(decode_stop.cut_bytes, decode_stop.cut_now);
    try testing.expectEqual(decode_stop.cache_bytes, decode_stop.cache_now);
}
