//! MiMo X12 `gpu_keep_warm` (mimo/gpu-warm): keep the GPU busy while the decode lane's event-gated miss parts wait for
//! their expert bytes, so the clock the serial GPU parts (miss parts, attention, the next hit wave) run at is not the
//! one a ~2 ms idle gap leaves behind. A construction-time route, read once (settings `"gpu_keep_warm"`):
//!   "off" (the control): nothing is installed; `StreamSource.gate` is the stock call.
//!   "waits" (the default): a trivial resident kernel (one dependent FMA chain per thread, ONE simdgroup, its own
//!            32 KiB buffer) on its own Metal command queue (csrc/mimo_gpu_warm.inc: MLX's device, not MLX's
//!            stream), committed in
//!            short chunks (~500 us, adapted from each chunk's GPU time) by a warm thread while the stream's
//!            MTLSharedEvent is below the value the newest committed gated wait needs. The decode lane arms it from
//!            `StreamSource.gate` (the decode phase only: the stream's phase, one field read): one atomic max, and a
//!            condition signal when the thread is parked. The thread stops committing at the event (the bytes landed);
//!            at most the chunks already in flight (two, each one simdgroup on one core) outlive the wait.
//! Exact by construction: the kernel's only memory is its own buffer; no model array, graph, stream, command-buffer
//! limit or event value is touched (the event is only read); the warm thread makes no MLX call. Costs: GPU power during
//! the waits (measured, not assumed: the `gpuwarm` family and the ABBA's GPU energy), one parked thread, the bill's
//! `gpu_warm` term (`bill_bytes`). An event standing still past `stuck_ns` (a read the pool's 10 s watchdog has not
//! failed yet) stops the warming until a newer value is armed.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;

const log = std.log.scoped(.mimo);

/// The warm engine's counters (`Stats`) are written in profile and test builds only: their readers are the MTP
/// profile's `MIMO_GPU_WARM` line and the CPU proofs. A served build's warm thread keeps only its working state.
const counting = @import("profile.zig").enabled or builtin.is_test;

/// `off`: nothing. `waits`: warm while a committed gated wait (decode miss parts, prompt group reads) is below the
/// stream's event, and while the host waits on a read-ahead (`StreamSource.awaitReadAhead`).
pub const Route = enum {
    off,
    waits,
    invalid,

};

/// The route's grid: one threadgroup of one simdgroup. A chunk still running when the bytes land occupies one
/// simdgroup slot of one core beside the model's kernels.
pub const route_groups: u32 = 1;
pub const threads: u32 = 32;
/// A chunk's target GPU length. After the event, at most the chunks in flight run on (two: <= ~1 ms of one simdgroup).
/// The profile timeline records every committed buffer (`dsv41_verify_timeline.max_buffers` = 262,144 per request): a
/// 1,024-token cell's ~116,700 model buffers (cells/k2-20261009) plus ~38 gated layers x ~2.4 ms / 0.5 ms x 270 rounds
/// = ~49,000 warm chunks stays inside it.
pub const chunk_target_ns: u64 = 500 * std.time.ns_per_us;
/// The first chunk's FMAs per thread; each retired chunk's GPU time moves it toward the target (within 20 % it stays).
pub const initial_iters: u32 = 65_536;
pub const min_iters: u32 = 256;
pub const max_iters: u32 = 1 << 24;
/// Chunks in flight: the next chunk is queued while one runs (no gap between chunks on the warm queue).
pub const max_inflight: usize = 2;
/// The event standing still this long while warming (a read the pool's 10 s watchdog has not failed yet; a decode wait
/// is ~2 ms per layer) stops the episode until a newer value is armed.
pub const stuck_ns: u64 = 200 * std.time.ns_per_ms;
/// The bill's `gpu_warm` term: the 32 KiB output buffer, the compiled library and pipeline, the queue, two command
/// buffers in flight and the parked thread's touched stack (declared, not measured: < 1 MiB).
pub const bill_bytes: u64 = 1 << 20;

// ── the shim (csrc/mimo_gpu_warm.inc) ──
const c = if (builtin.os.tag == .macos) struct {
    extern fn mimo_warm_create(groups: u32, threads: u32, out: *?*anyopaque) c_int;
    extern fn mimo_warm_destroy(h: ?*anyopaque) void;
    extern fn mimo_warm_commit(h: ?*anyopaque, iters: u32, cb: *?*anyopaque) c_int;
    extern fn mimo_warm_retire(cb: ?*anyopaque, gpu_start: *f64, gpu_end: *f64) c_int;
    extern fn mimo_warm_event_value(event: u64) u64;
    extern fn mimo_warm_now_ns() u64;
    extern fn mimo_warm_last_error() [*:0]const u8;
} else struct {
    fn mimo_warm_destroy(_: ?*anyopaque) void {}
};

/// The host clock Metal's GPU timestamps use (mach-absolute ns).
pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The warm queue, kernel and buffer on MLX's GPU device, and the stream's event it reads.
pub const Metal = struct {
    pub const Chunk = ?*anyopaque;
    h: ?*anyopaque = null,
    /// The stream's id<MTLSharedEvent> (`expert_event.Event.object`).
    event: u64,

    /// Construction only, under the caller's lease: compiles the kernel and makes the queue and buffer (no kernel runs).
    pub fn init(groups: u32, event_object: u64) !Metal {
        var h: ?*anyopaque = null;
        if (c.mimo_warm_create(groups, threads, &h) != 0) {
            log.warn("gpu_keep_warm=waits refused: {s}", .{std.mem.span(c.mimo_warm_last_error())});
            return error.MimoGpuKeepWarmDevice;
        }
        return .{ .h = h, .event = event_object };
    }

    pub fn deinit(self: *Metal) void {
        c.mimo_warm_destroy(self.h);
        self.h = null;
    }

    pub fn commit(self: *Metal, iters: u32) !Chunk {
        var cb: ?*anyopaque = null;
        if (c.mimo_warm_commit(self.h, iters, &cb) != 0) return error.MimoGpuWarmCommit;
        return cb;
    }

    /// Waits for the chunk; its GPU time (ns; 0 when Metal gave no times).
    pub fn retire(_: *Metal, ch: Chunk) !u64 {
        var s: f64 = 0;
        var e: f64 = 0;
        if (c.mimo_warm_retire(ch, &s, &e) != 0) return error.MimoGpuWarmChunk;
        return if (e > s and s > 0) @intFromFloat((e - s) * 1e9) else 0;
    }

    pub fn eventValue(self: *const Metal) u64 {
        return c.mimo_warm_event_value(self.event);
    }

    pub fn lastError() []const u8 {
        return std.mem.span(c.mimo_warm_last_error());
    }
};

/// The warm engine's counters (process-wide; the profile prints a request's difference). Written by the warm thread
/// (episodes, chunks, retired, stuck, GPU time, the current iterations) in profile and test builds only (`counting`),
/// and nowhere else: the decode lane's arm adds no counter. `failed` is the engine's own state, in every build.
pub const Stats = struct {
    episodes: u64 = 0,
    chunks: u64 = 0,
    retired: u64 = 0,
    stuck: u64 = 0,
    warm_gpu_ns: u64 = 0,
    iters: u32 = 0,
    last_chunk_ns: u64 = 0,
    failed: bool = false,

};

/// Tunables (tests shorten them).
pub const Params = struct {
    target_chunk_ns: u64 = chunk_target_ns,
    stuck_ns: u64 = stuck_ns,
    first_iters: u32 = initial_iters,
};

/// The warm loop over backend `B` (`Metal` served; a trace backend in the CPU tests): `commit(iters) !Chunk`,
/// `retire(Chunk) !u64` (the chunk's GPU ns; blocks until it completed), `eventValue() u64`.
pub fn Engine(comptime B: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        backend: B,
        params: Params,
        thread: std.Thread = undefined,
        /// The newest value a committed gated wait needs (monotonic: the stream's gate values only grow).
        target: std.atomic.Value(u64) = .init(0),
        parked: std.atomic.Value(u32) = .init(0),
        quit: std.atomic.Value(bool) = .init(false),
        /// Host waits in progress (`hold`): warm while any.
        holds: std.atomic.Value(u32) = .init(0),
        mu: std.c.pthread_mutex_t = .{},
        cv: std.c.pthread_cond_t = .{},
        /// Under `mu`: an arm (or the stop) signalled the parked thread.
        wake: bool = false,
        // ── the warm thread's own state ──
        ring: [max_inflight]B.Chunk = undefined,
        /// Each ring chunk's iterations (the adaptation scales the chunk's own count, not the current one).
        ring_iters: [max_inflight]u32 = undefined,
        ring_head: usize = 0,
        ring_n: usize = 0,
        iters: u32,
        /// A target the stuck bound gave up on: parked until a newer one.
        stuck_at: u64 = 0,
        // ── counters (the warm thread writes; anyone reads) ──
        n_episodes: std.atomic.Value(u64) = .init(0),
        n_chunks: std.atomic.Value(u64) = .init(0),
        n_retired: std.atomic.Value(u64) = .init(0),
        n_stuck: std.atomic.Value(u64) = .init(0),
        gpu_ns: std.atomic.Value(u64) = .init(0),
        cur_iters: std.atomic.Value(u32) = .init(0),
        last_ns: std.atomic.Value(u64) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),

        /// Owns `backend` (deinit on destroy) and starts the parked warm thread.
        pub fn create(a: std.mem.Allocator, backend: B, params: Params) !*Self {
            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.* = .{ .a = a, .backend = backend, .params = params, .iters = params.first_iters };
            if (comptime counting) self.cur_iters.store(params.first_iters, .monotonic);
            self.thread = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, main, .{self});
            return self;
        }

        /// Stops the thread (it retires its chunks in flight), joins it, releases the backend.
        pub fn destroy(self: *Self) void {
            self.quit.store(true, .seq_cst);
            self.signal();
            self.thread.join();
            if (comptime @hasDecl(B, "deinit")) self.backend.deinit();
            _ = std.c.pthread_cond_destroy(&self.cv);
            _ = std.c.pthread_mutex_destroy(&self.mu);
            self.a.destroy(self);
        }

        /// The decode lane's gated miss parts wait for `value` (`StreamSource.gate`): warm until the stream's event
        /// reaches it. Never blocks on the warm thread: one atomic max, and a signal only when the thread is parked.
        pub fn arm(self: *Self, value: u64) void {
            var cur = self.target.load(.seq_cst);
            while (value > cur) cur = self.target.cmpxchgWeak(cur, value, .seq_cst, .seq_cst) orelse break;
            if (self.parked.load(.seq_cst) != 0 and !self.failed.load(.monotonic)) self.signal();
        }

        /// A host wait on bytes begins (`on`) or ends: warm while any is in progress (the read-ahead's, whose GPU work
        /// is not gated on a value). Never blocks on the warm thread.
        pub fn hold(self: *Self, on: bool) void {
            if (on) {
                _ = self.holds.fetchAdd(1, .seq_cst);
                if (self.parked.load(.seq_cst) != 0 and !self.failed.load(.monotonic)) self.signal();
            } else _ = self.holds.fetchSub(1, .seq_cst);
        }

        pub fn stats(self: *const Self) Stats {
            return .{ .episodes = self.n_episodes.load(.monotonic), .chunks = self.n_chunks.load(.monotonic), .retired = self.n_retired.load(.monotonic), .stuck = self.n_stuck.load(.monotonic), .warm_gpu_ns = self.gpu_ns.load(.monotonic), .iters = self.cur_iters.load(.monotonic), .last_chunk_ns = self.last_ns.load(.monotonic), .failed = self.failed.load(.monotonic) };
        }

        fn signal(self: *Self) void {
            _ = std.c.pthread_mutex_lock(&self.mu);
            self.wake = true;
            _ = std.c.pthread_cond_signal(&self.cv);
            _ = std.c.pthread_mutex_unlock(&self.mu);
        }

        /// A target past the event's value and past a target given up as stuck, and no failure.
        fn wants(self: *Self, t: u64) bool {
            if (self.failed.load(.monotonic)) return false;
            if (self.holds.load(.seq_cst) > 0) return true;
            return t > self.stuck_at and self.backend.eventValue() < t;
        }

        fn main(self: *Self) void {
            while (!self.quit.load(.seq_cst)) {
                if (self.wants(self.target.load(.seq_cst))) {
                    self.warm();
                    continue;
                }
                self.park();
            }
            self.drain();
        }

        /// Parks until an arm or the stop. `parked` is published before the target is re-read: an arm either sees
        /// the thread parked (and signals) or its target is seen here (both seq_cst).
        fn park(self: *Self) void {
            _ = std.c.pthread_mutex_lock(&self.mu);
            self.parked.store(1, .seq_cst);
            while (!self.wake and !self.quit.load(.seq_cst) and !self.wants(self.target.load(.seq_cst)))
                _ = std.c.pthread_cond_wait(&self.cv, &self.mu);
            self.wake = false;
            self.parked.store(0, .seq_cst);
            _ = std.c.pthread_mutex_unlock(&self.mu);
        }

        /// One episode: chunks while the event is below the newest target. The event is read before every commit, so
        /// after the bytes land at most the chunks already in flight run on.
        fn warm(self: *Self) void {
            if (comptime counting) _ = self.n_episodes.fetchAdd(1, .monotonic);
            var seen = self.backend.eventValue();
            var moved = nowNs();
            while (true) {
                const t = self.target.load(.seq_cst);
                const v = self.backend.eventValue();
                const held = self.holds.load(.seq_cst) > 0;
                if ((v >= t and !held) or self.quit.load(.seq_cst)) return;
                const now = nowNs();
                if (held or v >= t) {
                    // a host wait (no value to watch): no stuck bound; it ends when the host returns
                    seen = v;
                    moved = now;
                } else if (v != seen) {
                    seen = v;
                    moved = now;
                } else if (now -| moved > self.params.stuck_ns) {
                    self.stuck_at = t;
                    if (comptime counting) _ = self.n_stuck.fetchAdd(1, .monotonic);
                    return;
                }
                if (self.ring_n == max_inflight) {
                    // the oldest chunk first (the queue already holds the next one); then the event again
                    self.retireOldest() catch |e| return self.fail(e);
                    continue;
                }
                const ch = self.backend.commit(self.iters) catch |e| return self.fail(e);
                const slot = (self.ring_head + self.ring_n) % max_inflight;
                self.ring[slot] = ch;
                self.ring_iters[slot] = self.iters;
                self.ring_n += 1;
                if (comptime counting) _ = self.n_chunks.fetchAdd(1, .monotonic);
            }
        }

        fn retireOldest(self: *Self) !void {
            const ch = self.ring[self.ring_head];
            const iters = self.ring_iters[self.ring_head];
            self.ring_head = (self.ring_head + 1) % max_inflight;
            self.ring_n -= 1;
            const ns = try self.backend.retire(ch);
            if (comptime counting) _ = self.n_retired.fetchAdd(1, .monotonic);
            if (ns == 0) return;
            if (comptime counting) {
                _ = self.gpu_ns.fetchAdd(ns, .monotonic);
                self.last_ns.store(ns, .monotonic);
            }
            self.adapt(iters, ns);
        }

        /// Moves the iterations toward the target chunk length when a chunk (of `iters`) is off by more than 20 %.
        fn adapt(self: *Self, iters: u32, ns: u64) void {
            const target = self.params.target_chunk_ns;
            if (ns * 5 >= target * 4 and ns * 4 <= target * 5) return;
            const next = @as(u128, iters) * target / ns;
            self.iters = @intCast(std.math.clamp(next, min_iters, max_iters));
            if (comptime counting) self.cur_iters.store(self.iters, .monotonic);
        }

        fn drain(self: *Self) void {
            while (self.ring_n > 0) self.retireOldest() catch {};
        }

        /// A backend error: the warming stops for the process (the model is unaffected), said once.
        fn fail(self: *Self, e: anyerror) void {
            if (!self.failed.swap(true, .monotonic)) log.warn("gpu_keep_warm: the warm thread stopped for the process: {s} ({s})", .{ @errorName(e), if (B == Metal) Metal.lastError() else "trace" });
            self.drain();
        }
    };
}

/// The served engine.
pub const Warm = Engine(Metal);

/// What `StreamSource.gate` calls (decode phase): an engine's `arm`, bound once at construction (the served engine, or a
/// CPU proof's trace engine).
pub const Arm = struct {
    ctx: *anyopaque,
    f: *const fn (*anyopaque, u64) void,
    h: *const fn (*anyopaque, bool) void,

    pub fn of(comptime E: type, e: *E) Arm {
        const W = struct {
            fn f(p: *anyopaque, v: u64) void {
                const x: *E = @ptrCast(@alignCast(p));
                x.arm(v);
            }
            fn h(p: *anyopaque, on: bool) void {
                const x: *E = @ptrCast(@alignCast(p));
                x.hold(on);
            }
        };
        return .{ .ctx = e, .f = W.f, .h = W.h };
    }

    pub inline fn call(self: Arm, value: u64) void {
        self.f(self.ctx, value);
    }

    pub inline fn holdOn(self: Arm, on: bool) void {
        self.h(self.ctx, on);
    }
};

/// The stream runs on the GPU (the route's precondition: a CPU stream's waits are host waits; nothing to keep warm).
pub fn gpuStream(s: mlx.mlx_stream) bool {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_stream_get_device(&dev, s) != 0) return false;
    defer _ = mlx.mlx_device_free(dev);
    var kind: mlx.mlx_device_type = .cpu;
    return mlx.mlx_device_get_type(&kind, dev) == 0 and kind == .gpu;
}

/// Construction (`Source.initWith`, under the caller's lease): `.waits` on a GPU stream with the stream's Metal event
/// makes the warm queue, kernel and buffer and starts the parked thread; `.off` installs nothing; anything else refuses
/// by name before any device work.
pub fn install(a: std.mem.Allocator, route: Route, stream: mlx.mlx_stream, event_object: u64) !?*Warm {
    switch (route) {
        .off => return null,
        .invalid => return error.MimoGpuKeepWarm,
        .waits => {},
    }
    if (!gpuStream(stream)) {
        log.warn("gpu_keep_warm=waits refused: the decode stream is not a GPU stream (its waits are host waits)", .{});
        return error.MimoGpuKeepWarmStream;
    }
    if (event_object == 0) {
        log.warn("gpu_keep_warm=waits refused: the stream's event is not a Metal event", .{});
        return error.MimoGpuKeepWarmEvent;
    }
    var backend = try Metal.init(route_groups, event_object);
    errdefer backend.deinit();
    return Warm.create(a, backend, .{});
}

// ── the trace backend (CPU proofs): chunks are log entries; a retire sleeps the chunk's emulated GPU time ──

pub const Trace = struct {
    pub const Chunk = struct { id: u64, iters: u32 };
    /// The test's record (it outlives the engine): commits and retires.
    pub const Log = struct {
        commits: std.atomic.Value(u64) = .init(0),
        retired: std.atomic.Value(u64) = .init(0),
    };
    /// The event value the warm loop reads (a test word, or the stream's host event word).
    value_fn: *const fn (?*const anyopaque) u64,
    ctx: ?*const anyopaque,
    log: *Log,
    /// The emulated GPU time of a chunk: iters x ns_per_iter.
    ns_per_iter: u64 = 2,
    /// A commit that fails (`fail_at` > 0: the commit with that number).
    fail_at: u64 = 0,

};

// ── CPU proofs (no device, no lock) ──

const testing = std.testing;
