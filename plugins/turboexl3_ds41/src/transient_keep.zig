//! `MLX_SERVE_DSV41_TRANSIENT_KEEP` (read at load; unset, empty or `1`: on; `0`: off; `check`: on, and each wide layer
//! counts the kept buffers its own projections took back; anything else is refused by name): a wide prompt layer's
//! attention projections' large outputs (its attention norm's rows, `wq_b`'s heads, their roped tail: [L, 5120],
//! [L, 64, 512] and [L, 64, 64] at L rows, 1.4 GB at 16K) kept past the layer's teardown by an extra reference and handed
//! to MLX's buffer cache at the next wide layer's start, so that layer's same arrays take those buffers back instead of
//! new Metal allocations.
//!
//! Off, the layer's scope frees them at its end. MLX recycles a freed buffer only while its cache is under the limit, and
//! at a wide layer's end the cache already holds the routed call's freed transients, so the projections' buffers go back
//! to Metal and the next layer allocates them anew when it encodes them (~15 ms of host a GB). The attention stage is
//! host-bound (pt5 w-pio-20261102-1502: 177 ms of host a layer for 158 ms of GPU), and in pt5's one-op-a-buffer arm the
//! encode of `wq_b`'s 1.07 GB output takes 19.2 ms of host a regular wide layer with the GPU idle 5.1 ms of it (the
//! stock arm's buffer holding it: 18.9 ms, 4.9 ms idle).
//!
//! On, a wide layer with another wide one after it in the pass (`layerEnd`, at its end) takes a reference to each of those
//! arrays of at least `min_bytes` (one per buffer: `wq_b`'s heads are a view of its output). The next wide layer, at its
//! start (`release`), lifts MLX's cache limit, frees the references (each buffer goes to the cache, the newest there),
//! and puts the limit back at once: MLX's next allocation trims the cache to the limit from its oldest buffers, and the
//! layer's same-shaped arrays (the same shapes every wide layer) take these (MLX hands out a cached buffer within two
//! pages of the size asked). The same kernels over the same inputs: only which buffer an output lands in moves, so the
//! same words (a text gate; MLX's cache hands out used buffers already).
//!
//! Memory: the kept arrays were alive through the layer's routed call either way (its scope held them until its end), so
//! from the teardown to the next layer's start they add only to the tail's in-flight buffers, which the routed call's own
//! encode already held beside them. At the next layer's start the cache is over its limit by their bytes until that
//! layer's first allocation trims it (the layer's graph build, no allocations, the footprint's low point), and the limit
//! itself never moves past an instant: no other freed buffer is cached beyond the limit. `[transient-keep] mode= layers=
//! arrays= mb=` (and `reused=`, check) at each request's end.
//!
//! Read as tk1 (w-pio-20261102-1746, a = ctl44, b = this at 1, a1 b1 b2 a2): cold 16K +0.63%, new text +1.12%, the
//! texts the same bytes, the bank rows and the guard's peak unchanged; merged on (zzz9q).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

const A = mlx.mlx_array;

pub const Mode = enum { off, on, check };

/// The setting (`configure`, at load).
pub var mode: Mode = .on;

/// The smallest array kept: a fresh allocation of it costs the host ~1 ms (MLX's new Metal buffer, ~15 ms a GB).
pub const min_bytes: u64 = 64 << 20;
/// Arrays a layer keeps at most (its attention norm's rows, `wq_b`'s heads, their roped tail).
pub const max_kept = 4;

pub const Error = error{TransientKeepSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.TransientKeepSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// The references a layer keeps (`layerEnd`) until the next wide layer's `release`.
var kept: [max_kept]A = undefined;
var n_kept: usize = 0;
var kept_bytes: u64 = 0;
/// `check`: the buffers the last `release` handed to the cache (their data pointers), for the layer's `layerEnd` to look for.
var given: [max_kept]usize = @splat(0);
var n_given: usize = 0;
// The meters since the last request's end.
var layers: u64 = 0;
var arrays: u64 = 0;
var mb: u64 = 0;
var reused: u64 = 0;

fn bytesOf(x: A) u64 {
    return @as(u64, mlx.mlx_array_size(x)) * @as(u64, mlx.mlx_array_itemsize(x));
}

/// The buffer's data pointer (0 unevaluated).
fn dataOf(x: A) usize {
    const p = mlx.mlx_array_data_uint8(x) orelse return 0;
    return @intFromPtr(p);
}

/// A wide layer's end (its arrays evaluated by the routed call, still held by its scope). `check`: each of `xs` whose
/// buffer the last `release` gave the cache counted. `keep` (the next layer of the pass wide too): a reference to each of
/// `xs` of at least `min_bytes`, one per buffer, for that layer's `release`.
pub fn layerEnd(xs: []const A, keep: bool) void {
    if (!on()) return;
    drop();
    if (mode == .check) {
        for (xs) |x| {
            if (x.ctx == null) continue;
            const d = dataOf(x);
            if (d != 0 and std.mem.indexOfScalar(usize, given[0..n_given], d) != null) reused += 1;
        }
    }
    n_given = 0;
    if (!keep) return;
    for (xs) |x| {
        if (x.ctx == null or n_kept == max_kept or bytesOf(x) < min_bytes) continue;
        const d = dataOf(x);
        // One reference a buffer (a view of a kept array's buffer at its start is the same buffer).
        if (d != 0 and (for (kept[0..n_kept]) |k| {
            if (dataOf(k) == d) break true;
        } else false)) continue;
        var h = mlx.mlx_array_new();
        if (mlx.mlx_array_set(&h, x) != 0) {
            _ = mlx.mlx_array_free(h);
            continue;
        }
        kept[n_kept] = h;
        n_kept += 1;
        kept_bytes += bytesOf(x);
    }
}

/// The next wide layer's start: the kept buffers to MLX's cache (the limit lifted while they are freed, then put back),
/// for this layer's arrays of their sizes.
pub fn release() void {
    if (n_kept == 0) return;
    if (mode == .check) {
        for (kept[0..n_kept], 0..) |k, j| given[j] = dataOf(k);
        n_given = n_kept;
    }
    var prev: usize = 0;
    const lifted = mlx.mlx_set_cache_limit(&prev, std.math.maxInt(usize)) == 0;
    // `wq_b`'s heads (the largest, kept first) freed last: the newest in the cache, the last a trim would take.
    var i = n_kept;
    while (i > 0) {
        i -= 1;
        _ = mlx.mlx_array_free(kept[i]);
    }
    if (lifted) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
    }
    layers += 1;
    arrays += n_kept;
    mb += kept_bytes >> 20;
    n_kept = 0;
    kept_bytes = 0;
}

/// A pass's end (or a kept set no layer took): the references let go as the layer's scope would have.
pub fn drop() void {
    for (kept[0..n_kept]) |k| _ = mlx.mlx_array_free(k);
    n_kept = 0;
    kept_bytes = 0;
}

/// The meters' line (`[transient-keep]`) since the last request's end, "" off.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[transient-keep] mode={t} layers={d} arrays={d} mb={d}", .{ mode, layers, arrays, mb }) catch return buf[0..0];
    if (mode == .check) w.print(" reused={d}", .{reused}) catch return buf[0..0];
    w.print("\n", .{}) catch return buf[0..0];
    layers = 0;
    arrays = 0;
    mb = 0;
    reused = 0;
    return w.buffered();
}

const testing = std.testing;

test "transient_keep setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "true", " 1" }) |v| try testing.expectError(error.TransientKeepSetting, parse(v));
}

test "transient_keep meters: empty off, the line on" {
    const was = mode;
    defer mode = was;
    var buf: [128]u8 = undefined;
    mode = .off;
    try testing.expectEqual(@as(usize, 0), meters(&buf).len);
    mode = .on;
    try testing.expect(std.mem.startsWith(u8, meters(&buf), "[transient-keep] mode=on layers=0 arrays=0 mb=0\n"));
}
