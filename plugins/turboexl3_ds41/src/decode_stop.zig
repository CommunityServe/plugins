//! `MLX_SERVE_DSV41_DECODE_STOP` (read at load; unset, empty or `1`: on; `0`: off; `check`: as `1`; anything else is
//! refused by name): the decode phase's budget takes part of the stop under the ceiling back for its slot rows, MLX's
//! cache capped at decode instead.
//!
//! The bank's bill keeps `stream_host_bytes` unplanned beside the host's wired-limit margin: a 2.0e9 B stop under the
//! ceiling and a 1 GiB host side, in both phases. The stop is there for what MLX holds beyond the bill (its buffer
//! cache, a phase's transients). At decode it holds little: on wz1's b arms (w-20261101-2308, 16K) MLX's cache was 97 to
//! 109 MiB and its peak 51 MiB over its active bytes; at a 32K context on ctl25 (md1, w-20261102-0003) the cache 0 and
//! the peak again 51 MiB over active, the footprint 3.51 GiB under the 102,400 MiB ceiling. On, the decode budget takes
//! `cut_bytes` of the stop back (two more decode slot rows at 16K and at 32K on the M5 Max: 470 MiB a row), and MLX's
//! cache is capped at `cache_bytes` from the phase change to decode (after the grow), the cap in effect before restored
//! at the change back to a prompt (before decode's rows are let go); the prompt's budget, rows and cache are as before.
//! The 2 GiB wired margin is untouched. `[decode-stop] mode= cut_mb= cache_mb= prompt_cache_mb= enters= exits=` at each
//! request's end. Exact: a route's bytes are the records' in any row; only which rows hold them moves.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

pub const Mode = enum { off, on, check };

/// Set at load.
pub var mode: Mode = .on;
/// The stop's bytes the decode budget takes back.
pub const cut_bytes: u64 = 1_000_000_000;
/// MLX's cache cap at decode.
pub const cache_bytes: usize = 1 << 30;
/// turboexl3_ds41 (MLX_SERVE_DSV41_DECODE_CACHE, src/decode_cache.zig): the cut and the cap in force, set at load before
/// the bill reads the cut; `cut_bytes` and `cache_bytes` unless DECODE_CACHE moves them.
pub var cut_now: u64 = cut_bytes;
pub var cache_now: usize = cache_bytes;

/// The cap in effect before decode's (restored at the change back to a prompt); null outside decode.
var prompt_cap: ?usize = null;
/// The last prompt cap seen (for the meters).
pub var last_prompt_cap: usize = 0;
/// Phase changes into decode and back, since load.
pub var enters: u64 = 0;
pub var exits: u64 = 0;

pub const Error = error{DecodeStopSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.DecodeStopSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// The bill's decode cut (`bill.decode_stop_cut`): `cut_bytes` on, 0 off.
pub fn cutBytes() u64 {
    return if (on()) cut_now else 0;
}

/// The phase change to decode, after the grow: MLX's cache capped at `cache_bytes` (never raised: a tighter cap stays),
/// the cap before it kept for the change back.
pub fn enterDecode() void {
    if (!on() or prompt_cap != null) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, cache_now);
    if (prev < cache_now) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
    }
    prompt_cap = prev;
    last_prompt_cap = prev;
    enters += 1;
}

/// The phase change back to a prompt, before decode's rows are let go: the cap before decode's restored.
pub fn enterPrompt() void {
    const prev = prompt_cap orelse return;
    var tmp: usize = 0;
    _ = mlx.mlx_set_cache_limit(&tmp, prev);
    prompt_cap = null;
    exits += 1;
}

/// The meters' line (`[decode-stop]`), "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[decode-stop] mode={t} cut_mb={d} cache_mb={d} prompt_cache_mb={d} enters={d} exits={d}\n", .{ mode, cut_now >> 20, cache_now >> 20, last_prompt_cap >> 20, enters, exits }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "decode stop setting: unset, empty or 1 on; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", "1 ", "true" }) |v| try testing.expectError(error.DecodeStopSetting, parse(v));
}

test "decode stop cut: the stop's bytes on, none off" {
    const saved = mode;
    defer mode = saved;
    mode = .off;
    try testing.expectEqual(@as(u64, 0), cutBytes());
    mode = .on;
    try testing.expectEqual(cut_bytes, cutBytes());
    // two 492,810,240 B rows, with the 16K bill's 59 to 76 MiB remainder, and never a third
    const row: u64 = 492_810_240;
    for ([_]u64{ 59 << 20, 76 << 20, 165 << 20 }) |rem| try testing.expectEqual(@as(u64, 2), (rem + cut_bytes) / row);
}

test "decode stop meters: the terms; none off" {
    const saved = .{ mode, enters, exits, last_prompt_cap };
    defer {
        mode = saved[0];
        enters = saved[1];
        exits = saved[2];
        last_prompt_cap = saved[3];
    }
    var buf: [160]u8 = undefined;
    mode = .off;
    try testing.expectEqualStrings("", meters(&buf));
    mode = .on;
    enters = 2;
    exits = 1;
    last_prompt_cap = 2 << 30;
    try testing.expectEqualStrings("[decode-stop] mode=on cut_mb=953 cache_mb=1024 prompt_cache_mb=2048 enters=2 exits=1\n", meters(&buf));
}
