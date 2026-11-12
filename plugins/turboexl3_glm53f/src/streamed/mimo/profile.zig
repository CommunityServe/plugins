//! Diagnostic builds only. No clock, counter, or branch in served builds.
const std = @import("std");
const sdk = @import("sdk");
pub const enabled = sdk.plugin_profile;
pub const Bucket = @import("../sdk_ext.zig").profile.PrefillBucket;
pub const Stamp = if (enabled) u64 else void;
const n = @typeInfo(Bucket).@"enum".field_names.len;
var ns: [n]u64 = @splat(0);
var calls: u64 = 0;
pub inline fn now() Stamp {
    if (comptime !enabled) return {};
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
pub inline fn charge(bucket: Bucket, start: Stamp) void {
    if (comptime enabled) ns[@backingInt(bucket)] += now() - start;
}
pub inline fn call() void {
    if (comptime enabled) calls += 1;
}

fn seconds(bucket: Bucket) f64 {
    return @as(f64, @floatFromInt(ns[@backingInt(bucket)])) / std.time.ns_per_s;
}
