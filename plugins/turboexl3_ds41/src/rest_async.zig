//! `MLX_SERVE_DSV41_REST_ASYNC` (read at load; unset, empty or `1`: on; `0`: off; anything else is refused by name): a
//! wide prompt layer followed by another wide one has its last eval committed, not waited for, so the next layer's start
//! (its graph built, its read-ahead's first buffers encoded) runs while the GPU finishes this layer's tail.
//!
//! Off, the copied model's `runLayers` evaluates each wide layer's held arrays and its attention state and waits. Since
//! REST_EARLY (the tail committed before the teardown) and HC_FUSE (the expand and the next attention's norm in the tail),
//! the tail is ~17.5 ms of GPU work a 16K-row encoder layer and the host waits out most of it there (pt5,
//! w-pio-20261102-1502: the rest stage 17.7 ms of host a layer, 17.5 ms of it GPU busy); then the next layer's start keeps
//! the GPU idle 5.9 ms a regular wide layer (its graph and the read-ahead's encode before its first commit; ~0.11 s a
//! cold 16K card over 18 regular wide layers).
//!
//! On, the same arrays are committed (`mlx_async_eval`) when the next layer in the pass is wide too, and the host goes on
//! to that layer at once: its start's host work runs under the tail's remaining GPU time, and its first commit lands
//! behind the tail on the one stream. The pass's last wide layer is waited for as before. The same kernels in the same
//! order on the one stream, so the same words (a text gate: no in-process comparison exists, the graph is the same either
//! way). Memory: the layer's teardown (its scopes' frees) has already run when the last eval is made, so what the next
//! layer's first arrays overlap is the tail's own inputs and outputs (the routed parts it sums, the shared expert's rows,
//! the new streams), which the old flow freed only once the tail was done; the next layer's start makes its hc mixes and
//! norms and the read-ahead's small arrays there, far under the attention's own peak (no bill term; the window's peak
//! gate checks it).

//!
//! Read as ra2 (w-pio-20261102-1943, a = ctl46, b = this at 1, a1 b1 b2 a2): cold 16K +0.76%, new text +0.90%, the
//! texts the same bytes, the bank rows and the guard's peak unchanged; merged on (zzz9s).
const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{RestAsyncSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.RestAsyncSetting;
}

test "rest_async setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(try parse("1"));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "on", "2", "00", "check", "true", " 1" }) |v| try std.testing.expectError(error.RestAsyncSetting, parse(v));
}
