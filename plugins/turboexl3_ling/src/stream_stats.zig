//! A bank phase's `[stream-stats]` line (src/ling_bank.zig `meters`), MLX-free: the stream's counters over the phase.

const std = @import("std");

/// One phase's line: routes resident or read, the bytes read and the read pool's busy wall time, the time waited on
/// reads, and the prompt's read-ahead. `st` has the kit's `expert.Stats` counter names.
pub fn line(buf: []u8, name: []const u8, st: anytype) ![]u8 {
    const looked = st.expert_cache_hits + st.expert_cache_misses;
    const rate: f64 = if (looked == 0) 0 else 100.0 * @as(f64, @floatFromInt(st.expert_cache_hits)) / @as(f64, @floatFromInt(looked));
    const wall: f64 = @as(f64, @floatFromInt(st.read_wall_ns)) / 1e9;
    const gbps: f64 = if (wall == 0) 0 else @as(f64, @floatFromInt(st.expert_bytes_read)) / 1e9 / wall;
    const wait: f64 = @as(f64, @floatFromInt(st.read_wait_ns)) / 1e9;
    return std.fmt.bufPrint(buf, "[stream-stats] {s} hits={d} misses={d} hit_rate={d:.2}% read_bytes={d} read_wall_s={d:.3} read_GBps={d:.2} read_wait_s={d:.3} ahead_posted={d} ahead_hits={d} ahead_demand={d}", .{
        name, st.expert_cache_hits, st.expert_cache_misses, rate, st.expert_bytes_read, wall, gbps, wait, st.ahead_posted, st.ahead_hits, st.ahead_demand,
    });
}

test "stream stats: a phase's line reads its counters" {
    const S = struct { expert_cache_hits: u64 = 0, expert_cache_misses: u64 = 0, expert_bytes_read: u64 = 0, read_wall_ns: u64 = 0, read_wait_ns: u64 = 0, ahead_posted: u64 = 0, ahead_hits: u64 = 0, ahead_demand: u64 = 0 };
    var buf: [512]u8 = undefined;
    const got = try line(&buf, "prompt", S{ .expert_cache_hits = 3, .expert_cache_misses = 1, .expert_bytes_read = 2_000_000_000, .read_wall_ns = 1_000_000_000, .ahead_posted = 7 });
    try std.testing.expectEqualStrings("[stream-stats] prompt hits=3 misses=1 hit_rate=75.00% read_bytes=2000000000 read_wall_s=1.000 read_GBps=2.00 read_wait_s=0.000 ahead_posted=7 ahead_hits=0 ahead_demand=0", got);
}
