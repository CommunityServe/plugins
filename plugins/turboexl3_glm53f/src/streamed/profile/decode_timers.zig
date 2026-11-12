//! The DSV4.1 DSpark cycle's host split, for PROFILE builds only (`-Ddsv41-decode-timers=true`): host time
//! by cycle phase and, inside the verify forward, the routed calls' waits, accumulated per process and
//! printed once (`DECODE_TIMERS`). In every other build `enabled` is false and each call below compiles
//! to nothing (the timed path carries no timer, branch or counter).

const std = @import("std");
const bo = @import("../build_flags.zig");
const sdk_ext = @import("../sdk_ext.zig");

pub const enabled: bool = if (@hasDecl(bo, "dsv41_decode_timers")) bo.dsv41_decode_timers else false;
/// Routed layers a per-layer profile row holds.
pub const max_layers = 64;

/// Where a cycle's host time goes. The verify forward's own share (`verify`) holds the routed calls'
/// `barrier`, `route` and `read_wait`; the rest of it is the forward's encode (graph builds, wave
/// commits, the in-place reads).
pub const Bucket = enum {
    /// The draft block's graph.
    draft_build,
    /// Blocked on the draft's eval: the GPU's window update, main row and draft block.
    draft_wait,
    /// The draft's reads, the confidence stop and the lookup's extension.
    draft_host,
    /// The whole verify forward (its graph and every routed call, up to its last layer).
    verify,
    /// Inside `verify`: blocked in a routing barrier's eval (the GPU up to the layer's router; with event
    /// gates also its wait on the previous layer's bytes).
    barrier,
    /// Inside `verify`: the stream's route (plan, pins, read submission, read-ahead).
    route,
    /// Inside `verify`: blocked on reads (waitGu / waitDown; the host-waits arm only).
    read_wait,
    /// Blocked on the verify's last eval (the last layer, the head, the main taps).
    verify_eval,
    /// The decision: its graph, eval and reads, and the acceptance.
    accept,
    /// The target's trim.
    trim,
    /// The draft windows' seed (graph).
    seed,
    /// The window update's dispatch (its GPU time lands in the next `draft_wait`).
    window,
    /// The round's tail: the lookup, the next main row, the stream's flush, the backend reset.
    tail,
};
const n_buckets = @typeInfo(Bucket).@"enum".field_names.len;

pub var ns: [n_buckets]u64 = @splat(0);
pub var cycles: u64 = 0;
pub var routed_calls: u64 = 0;

const clock = sdk_ext.profile.Clock(enabled);
pub const Stamp = clock.Stamp;
pub const now = clock.now;

/// Charge the time since `t0` to `b`; returns now (the next span's start).
pub inline fn charge(b: Bucket, t0: Stamp) Stamp {
    if (comptime !enabled) return {};
    const t = now();
    ns[@intFromEnum(b)] += t - t0;
    return t;
}

pub inline fn countCall() void {
    if (comptime !enabled) return;
    routed_calls += 1;
}
