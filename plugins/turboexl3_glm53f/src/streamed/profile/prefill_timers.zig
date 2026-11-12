//! The DSV4.1 prompt pass's routed-call split, for PROFILE builds only (`-Ddsv41-prefill-timers=true`): host
//! time in the wide lane's steps and the DIG-X dispatcher's waves, accumulated per process. In every
//! other build `enabled` is false and each call below compiles to nothing (the timed path carries no
//! timer, branch or counter).

const std = @import("std");
const bo = @import("../build_flags.zig");
const sdk_ext = @import("../sdk_ext.zig");

pub const enabled: bool = if (@hasDecl(bo, "dsv41_prefill_timers")) bo.dsv41_prefill_timers else false;

/// Where the routed call's host time goes (the SDK's probe shape, G7): the routing barrier, the stream's route,
/// the read waits, the waves' encode, the drains, the join.
pub const Bucket = sdk_ext.profile.PrefillBucket;
const n_buckets = @typeInfo(Bucket).@"enum".field_names.len;

pub var ns: [n_buckets]u64 = @splat(0);
pub var waves: u64 = 0;
pub var calls: u64 = 0;
pub var launches: u64 = 0;

/// Whose time a charge is (PREFILL_PROFILE_ROUTED_SPLIT): the stream groups', or the deferred base call's at the seed
/// (P1b) or at the call's end. The wide lane sets it around the base call; the dispatcher's charges follow it.
pub const Mode = enum { stream, base_seed, base_end };
const n_modes = @typeInfo(Mode).@"enum".field_names.len;
pub var mode: Mode = .stream;
pub var ns_mode: [n_modes][n_buckets]u64 = @splat(@splat(0));
pub var wide_calls: u64 = 0;

/// P1's read-ahead, one record per layer (PREFILL_PROFILE_READAHEAD): the predictor's counts, the predicted seed
/// (the top of its ranking the layer's empty rows hold), the admissions and reads, the predicted experts no empty row
/// took (blocked: every row held by a resident), and at the barrier the hits, the seed's demand records and their
/// predicted counts against the cut (the last predicted rank's count): within x0.8-1.25 (near) or beyond (far).
pub const max_layers = @import("decode_timers.zig").max_layers;
pub const max_experts = 512;

const clock = sdk_ext.profile.Clock(enabled);
pub const Stamp = clock.Stamp;
pub const now = clock.now;

/// Charge the time since `t0` to `b` (and to the current mode's split).
pub inline fn charge(b: Bucket, t0: Stamp) void {
    if (comptime !enabled) return;
    const d = now() - t0;
    ns[@intFromEnum(b)] += d;
    ns_mode[@intFromEnum(mode)][@intFromEnum(b)] += d;
}

pub inline fn count(w: u64, l: u64) void {
    if (comptime !enabled) return;
    waves += w;
    launches += l;
}

pub inline fn countCall() void {
    if (comptime !enabled) return;
    calls += 1;
}

pub fn seconds(b: Bucket) f64 {
    return @as(f64, @floatFromInt(ns[@intFromEnum(b)])) / 1e9;
}
