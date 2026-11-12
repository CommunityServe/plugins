//! The verify's GPU timeline, for PROFILE builds only (`dsv41_decode_timers.enabled`; in every other build each entry
//! below compiles to nothing and the shim, lib/expert_io/dsv41_cb_timeline.mm, is not linked). Active in a
//! decode-profile run (`install`): every command buffer MLX's GPU queue commits during the decode is recorded with
//! its host commit time, Metal's GPUStartTime / GPUEndTime and its completion handler's host time; each verify's
//! window and, per routed layer, the host stamps of the layer's submits (`Point`). All storage is static; the
//! summary (`VERIFY_GPU_TIMELINE`) and the raw rows (`writeJson`) are made after the decode.
const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const dt = @import("decode_timers.zig");

pub const enabled = dt.enabled;

/// Set by `install`, cleared by `uninstall`.
pub var active: bool = false;

/// Cycles <= max_tokens - 1 (the cell's default cap: 1,024 ids).
pub const max_cycles = 1024;
pub const max_layers = @import("decode_timers.zig").max_layers;
/// Buffers past this are counted (`dropped`), not recorded.
pub const max_buffers = 1 << 18;

/// One committed command buffer (the shim's `Dsv41tlRow`). Times: mach_absolute_time ns.
/// `queue`: 1 for the queue of the buffer the install saw (MLX's stream), 0 for another queue.
pub const Row = extern struct { host_commit: u64 = 0, gpu_start: u64 = 0, gpu_end: u64 = 0, host_done: u64 = 0, tag: u64 = 0, status: u64 = 0, queue: u64 = 0 };

pub const Phase = enum(u8) { draft = 1, verify = 2, rest = 3 };

/// A routed layer's host stamps in a verify (`Experts.run`): the barrier's eval submitted (`call`), the ids read
/// (`barrier`), the route planned (`route`), the hit wave and hoist submitted (`hit`), the call's last submit (`end`).
pub const Point = enum(u8) { call, barrier, route, hit, end };
/// `gate_gu` / `gate_down`: the stream event's values the layer's gated miss parts wait for (its gate/up value and its
/// last down part's), 0 without a gated part.
pub const Layer = struct { t: [5]u64 = @splat(0), read_call: u64 = 0, read_end: u64 = 0, missed: bool = false, gate_gu: u64 = 0, gate_down: u64 = 0 };

/// The read pool's event-signal log (`q3ld_test_ev_log`, compiled in the profile and test builds): per signal the
/// value handed to the event, the host time of the signal call, the time the satisfied prefix advanced (the bytes
/// landed) and the call's ns. Past the cap the pool stops logging (`signals_full`).
pub const max_signals = 1 << 17;
pub const Signal = [4]i64;

var rows: [max_buffers]Row = undefined;
var layers: [max_cycles][max_layers]Layer = undefined;
var signals: [max_signals]Signal = undefined;
var n_signals: usize = 0;
/// Cycles begun (`cycleBegin`), and the one in progress.
pub var cycles: u32 = 0;
var cur: u32 = 0;
var in_verify = false;
pub var n_layers: u32 = 0;
var t0: u64 = 0;
/// The stream whose current buffer every cycle's begin re-checks (a buffer of an unhooked class is hooked).
var stream: ?mlx.mlx_stream = null;

const c = if (enabled) struct {
    pub extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
    pub extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
    pub extern fn dsv41tl_tag(tag: u64) void;
    pub extern fn dsv41tl_counts(out: *[3]u32) void;
    pub extern fn dsv41tl_rehook(buf: ?*anyopaque) c_int;
    pub extern fn dsv41tl_hook_stats(out: *[4]u64) void;
    pub extern fn dsv41tl_hooked_names(out: [*]u8, cap: usize) usize;
    pub extern fn dsv41tl_selftest(rows: [*]Row, cap: u32) c_int;
    pub extern fn q3ld_test_ev_log(buf: ?[*]i64, cap: i64) i64;
} else struct {};
// The guards below (`@hasDecl(c, ...)`) see only pub declarations: a profile build must reach every shim call.
comptime {
    if (enabled) for (.{ "dsv41tl_tag", "dsv41tl_rehook", "dsv41tl_counts", "dsv41tl_hook_stats", "dsv41tl_hooked_names", "q3ld_test_ev_log" }) |n| {
        if (!@hasDecl(c, n)) @compileError("dsv41 verify timeline: the shim's " ++ n ++ " is not reachable");
    };
}

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn arm(layer_count: u32) void {
    n_layers = layer_count;
    cycles = 0;
    cur = 0;
    in_verify = false;
    n_signals = 0;
    t0 = now();
    active = true;
}

/// A routed layer's gated miss parts in the verify: the event values its gate/up waves and its last down wave wait for.
pub fn gate(layer: u32, gu: u64, down_first: u64, n_parts: u32) void {
    if (comptime !enabled) return;
    if (!active or !in_verify or layer >= max_layers or n_parts == 0) return;
    layers[cur][layer].gate_gu = gu;
    layers[cur][layer].gate_down = down_first + n_parts - 1;
}

/// (committed, completed, dropped) from the shim.
pub fn counts() [3]u32 {
    var out: [3]u32 = .{ 0, 0, 0 };
    if (comptime @hasDecl(c, "dsv41tl_counts")) c.dsv41tl_counts(&out);
    return out;
}

// ── the summary (pure over the stored rows; the tests feed it by hand) ──

/// A host- or GPU-clock interval [start, end) (the prompt timeline's too, `mimo_v2/prefill_timeline.zig`).
pub const Span = [2]u64;

fn median(xs: []f64) f64 {
    if (xs.len == 0) return 0;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}
