//! TURBOEXL3_GLM53F_KEEP_WARM's engine (the copied src/mlx_serve/sdk/weights/streamed/mimo/gpu_warm.zig) proved on its
//! trace backend, no device: mlx-serve's own CPU proofs of the engine (the copy's tests, which the sync leaves out),
//! named for the stream test's filter (build.zig `stream_options`). The engine warms only while the stream's event is
//! below the newest armed value or a host hold is in progress, stops at the event (at most the chunks in flight after
//! it), adapts its chunk length to the target, gives up on an event standing still, stops for good on a failed commit,
//! and retires every chunk in flight when destroyed.
const std = @import("std");
const gpu_warm = @import("mlx_stream").gpu_warm;
const Trace = gpu_warm.Trace;
const TraceEngine = gpu_warm.TraceEngine;
const testing = std.testing;

fn waitFor(comptime f: anytype, args: anytype, timeout_ms: u64) !void {
    const t0 = gpu_warm.nowNs();
    while (!@call(.auto, f, args)) {
        if (gpu_warm.nowNs() - t0 > timeout_ms * std.time.ns_per_ms) return error.TestTimeout;
        sleepUs(200);
    }
}

fn atLeast(l: *const Trace.Log, n: u64) bool {
    return l.commits.load(.seq_cst) >= n;
}

fn parkedNow(e: *const TraceEngine) bool {
    return e.isParked();
}

fn sleepUs(us: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(us / std.time.us_per_s), .nsec = @intCast((us % std.time.us_per_s) * std.time.ns_per_us) };
    _ = std.c.nanosleep(&ts, null);
}

test "turboexl3_glm53f stream keep warm: the routes \"off\" and \"waits\" parse; anything else is invalid" {
    try testing.expectEqual(gpu_warm.Route.off, gpu_warm.parseRoute("off"));
    try testing.expectEqual(gpu_warm.Route.waits, gpu_warm.parseRoute("waits"));
    try testing.expect(gpu_warm.Route.waits.waitsOn() and !gpu_warm.Route.off.waitsOn());
    for ([_][]const u8{ "", "on", "Waits", "invalid", "wait", "always", "tick", "all" }) |s| try testing.expectEqual(gpu_warm.Route.invalid, gpu_warm.parseRoute(s));
}

test "turboexl3_glm53f stream keep warm: the engine warms only while the event is below the armed value, stops at the event (at most one commit after it), parks and re-arms (trace backend)" {
    const a = testing.allocator;
    var ev = std.atomic.Value(u64).init(0);
    var l: Trace.Log = .{};
    const e = try TraceEngine.create(a, Trace.word(&ev, &l), .{ .target_chunk_ns = 50 * std.time.ns_per_us, .first_iters = 25_000 });
    defer e.destroy();
    // parked with nothing armed: no chunk
    try waitFor(parkedNow, .{e}, 2000);
    sleepUs(2000);
    try testing.expectEqual(@as(u64, 0), l.commits.load(.seq_cst));
    // an arm the event already satisfies: still nothing
    ev.store(4, .seq_cst);
    e.arm(3);
    sleepUs(2000);
    try testing.expectEqual(@as(u64, 0), l.commits.load(.seq_cst));
    // armed past the event: chunks until the event reaches it
    e.arm(6);
    try waitFor(atLeast, .{ &l, 6 }, 2000);
    ev.store(6, .seq_cst);
    const at_store = l.commits.load(.seq_cst);
    try waitFor(parkedNow, .{e}, 2000);
    const after = l.commits.load(.seq_cst);
    try testing.expect(after - at_store <= 1);
    sleepUs(3000);
    try testing.expectEqual(after, l.commits.load(.seq_cst));
    // a lower or equal value never re-arms (the targets are monotonic)
    e.arm(5);
    e.arm(6);
    sleepUs(2000);
    try testing.expectEqual(after, l.commits.load(.seq_cst));
    // a newer value resumes; its event stops it again
    e.arm(9);
    try waitFor(atLeast, .{ &l, after + 4 }, 2000);
    ev.store(9, .seq_cst);
    try waitFor(parkedNow, .{e}, 2000);
    const st = e.stats();
    try testing.expectEqual(@as(u64, 2), st.episodes);
    try testing.expect(!st.failed);
    try testing.expectEqual(@as(u64, 0), st.stuck);
    // at most `max_inflight` chunks are ever unretired
    try testing.expect(st.chunks - st.retired <= gpu_warm.max_inflight);
}

test "turboexl3_glm53f stream keep warm: the chunk length converges to the target, an event standing still stops the warming until a newer value, a failed commit stops the engine for good (trace backend)" {
    const a = testing.allocator;
    var ev = std.atomic.Value(u64).init(0);
    var l: Trace.Log = .{};
    // 2 ns an iteration from 1,000,000 iterations (2 ms chunks): the 100 us target is 50,000 iterations
    const e = try TraceEngine.create(a, Trace.word(&ev, &l), .{ .target_chunk_ns = 100 * std.time.ns_per_us, .first_iters = 1_000_000, .stuck_ns = 40 * std.time.ns_per_ms });
    defer e.destroy();
    e.arm(1);
    try waitFor(atLeast, .{ &l, 1 }, 2000);
    try waitFor(parkedNow, .{e}, 2000); // the event never moves: the stuck bound (40 ms) ends the episode
    var st = e.stats();
    try testing.expectEqual(@as(u64, 1), st.stuck);
    try testing.expect(st.iters >= 40_000 and st.iters <= 62_500);
    const parked_at = l.commits.load(.seq_cst);
    // the same value again: parked (given up); a newer one: warms again
    e.arm(1);
    sleepUs(3000);
    try testing.expectEqual(parked_at, l.commits.load(.seq_cst));
    e.arm(2);
    try waitFor(atLeast, .{ &l, parked_at + 3 }, 2000);
    ev.store(2, .seq_cst);
    try waitFor(parkedNow, .{e}, 2000);
    st = e.stats();
    try testing.expectEqual(@as(u64, 1), st.stuck);
    // a failing backend: the engine stops for the process; the arms after it commit nothing; its chunks are retired
    var ev2 = std.atomic.Value(u64).init(0);
    var l2: Trace.Log = .{};
    var tb = Trace.word(&ev2, &l2);
    tb.fail_at = 3;
    const f = try TraceEngine.create(a, tb, .{ .target_chunk_ns = 50 * std.time.ns_per_us, .first_iters = 25_000 });
    defer f.destroy();
    f.arm(1);
    try waitFor(atLeast, .{ &l2, 2 }, 2000);
    try waitFor(parkedNow, .{f}, 2000);
    try testing.expect(f.stats().failed);
    const n = l2.commits.load(.seq_cst);
    f.arm(5);
    sleepUs(3000);
    try testing.expectEqual(n, l2.commits.load(.seq_cst));
    try testing.expectEqual(n, l2.retired.load(.seq_cst));
}

test "turboexl3_glm53f stream keep warm: destroy while warming retires every chunk in flight and joins (trace backend)" {
    const a = testing.allocator;
    var ev = std.atomic.Value(u64).init(0);
    var l: Trace.Log = .{};
    const e = try TraceEngine.create(a, Trace.word(&ev, &l), .{ .target_chunk_ns = 50 * std.time.ns_per_us, .first_iters = 25_000 });
    e.arm(100);
    try waitFor(atLeast, .{ &l, 5 }, 2000);
    e.destroy();
    try testing.expect(l.commits.load(.seq_cst) >= 5);
    try testing.expectEqual(l.commits.load(.seq_cst), l.retired.load(.seq_cst));
}

test "turboexl3_glm53f stream keep warm: a host hold warms until released, with no stuck bound (trace backend)" {
    const a = testing.allocator;
    var ev = std.atomic.Value(u64).init(0);
    var l: Trace.Log = .{};
    const e = try TraceEngine.create(a, Trace.word(&ev, &l), .{ .target_chunk_ns = 50 * std.time.ns_per_us, .first_iters = 25_000, .stuck_ns = 10 * std.time.ns_per_ms });
    defer e.destroy();
    try waitFor(parkedNow, .{e}, 2000);
    e.hold(true);
    try waitFor(atLeast, .{ &l, 4 }, 2000);
    sleepUs(30_000); // past the stuck bound: a hold has none
    try testing.expect(!e.isParked());
    try testing.expectEqual(@as(u64, 0), e.stats().stuck);
    e.hold(false);
    try waitFor(parkedNow, .{e}, 2000);
    const n = l.commits.load(.seq_cst);
    sleepUs(5000);
    try testing.expectEqual(n, l.commits.load(.seq_cst));
}

test "turboexl3_glm53f stream keep warm: the profile line names the route and the request's counters" {
    var buf: [512]u8 = undefined;
    const now: gpu_warm.Stats = .{ .episodes = 50, .chunks = 410, .retired = 409, .stuck = 0, .warm_gpu_ns = 102_500_000, .iters = 70_000, .last_chunk_ns = 251_000 };
    const then: gpu_warm.Stats = .{ .episodes = 3, .chunks = 10, .retired = 9, .warm_gpu_ns = 2_500_000 };
    const ln = gpu_warm.line(&buf, .waits, now.since(then));
    try testing.expectEqualStrings("MIMO_GPU_WARM {\"route\": \"waits\", \"episodes\": 47, \"chunks\": 400, \"retired\": 400, \"stuck\": 0, \"warm_gpu_ms\": 100.000, \"iters\": 70000, \"last_chunk_us\": 251.0, \"failed\": false}", ln);
}
