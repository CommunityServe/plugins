//! A wide prompt layer's attention spans write their scores and probabilities into a ring of two per-prompt slots
//! (src/scratch.zig's `Region`) instead of fresh MLX buffers.
//!
//! The gather's three launches a span (attn_qk, or Q_TAIL's attn_qk_t; attn_sm; attn_pv) each made a fresh output:
//! the scores `s` f32 [L, H, NK] (up to 41.9 MB at 256 rows, 64 heads, 640 columns), the probabilities `p` f16 (21 MB)
//! and the heads `o`. A layer's spans are all encoded inside its one eval, so their arrays are live together and MLX's
//! buffer cache cannot hand one span's to the next whatever its cap: 66% of p missed it in a cold 16K prompt and 82% in
//! the new-text one, at 2 GiB as at 6 GiB (prf2, w-pio-20261101-1419: p0 and p6); clearing it at each layer's start
//! (LAYER_CLEAR) only added misses.
//!
//! The ring: S (f32) and P (f16), each two slots of `span x H x nk_max` words (256 x 64 x 640: 126 MB together); a
//! wide layer's span k takes slot k mod 2. The launches are write-into twins of the stock texts (src/attn_twins.zig,
//! derived and compiled at load): qk writes s through `s_dst + soff[0]`, sm reads it through `s_rg + soff[0]` and
//! writes p through `p_dst + soff[0]`, pv reads p through `p_rg + soff[0]` (its heads MLX's own output); each also
//! writes a one-word carrier. sm binds qk's carrier as its `dep` and pv binds sm's (MLX's barrier or fence orders each
//! after the one it reads), and a slot's next qk binds the carrier of the last pv that read the slot (span k - 2's): every
//! read of both of the slot's regions is behind that pv, since it follows the slot's sm. The regions are bound whole
//! (evaluated arrays, so no `contiguous` copy stands in for them) and are never an output, so MLX tracks no hazard on
//! them and the carriers alone order them. Exact: the stock statements write and read the same words, only where s and
//! p live changes.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const scratch = @import("scratch.zig");
/// The twins' text derivation (std only).
pub const twins = @import("attn_twins.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const slots = 2;

/// The ring's regions, made at a wide layer's start and kept while their shape holds; freed at the decode phase change.
pub var s_region: scratch.Region = .{};
pub var p_region: scratch.Region = .{};
/// Each slot's `soff` (a `[1]` uint32: the slot's first word, the same in S and P), made with the regions.
var soffs: [slots]A = .{ none, none };
/// A `[1]` uint32 zeros: the `dep` of a slot's first qk (nothing read the slot yet).
var no_dep: A = none;
/// Each slot's last pv's carrier: the slot's next qk binds it.
var last_read: [slots]A = .{ none, none };
/// MLX_SERVE_DSV41_FLASH_ATTN on (src/flash_attn.zig) clears it at load: the spans write no scores or probabilities, so
/// a layer keeps the ring's span bookkeeping and the heads regions but makes no S or P regions.
pub var ring_regions: bool = true;
/// The slot geometry the regions hold: rows, heads, columns a slot.
var g_rows: usize = 0;
var g_heads: usize = 0;
var g_nk: usize = 0;
/// The layer's next span's slot, and whether its spans take the ring.
var next: u1 = 0;
var live = false;

/// A wide prompt layer starts: the regions made at `rows x heads x nk` words a slot (or kept), the ring live for its
/// spans; false (the fresh path) on a CPU stream or without a streamed bank (`eligible`), or past the regions' cap.
pub fn layerBegin(eligible: bool, rows: usize, heads: usize, nk: usize) !bool {
    live = false;
    next = 0;
    if (!eligible) return false;
    if (!ring_regions) {
        s_region.free();
        p_region.free();
        g_rows = rows;
        g_heads = heads;
        g_nk = nk;
        try ensureNoDep();
        live = true;
        return true;
    }
    const words = rows * heads * nk;
    const n: c_int = @intCast(slots * words);
    const had = s_region.arr.ctx != null and p_region.arr.ctx != null and g_rows == rows and g_heads == heads and g_nk == nk;
    if ((try s_region.ensure(&.{n}, .float32)) == null) return false;
    if ((try p_region.ensure(&.{n}, .float16)) == null) return false;
    if (!had) {
        // new regions: no reader of their words yet
        for (&last_read) |*x| drop(x);
        for (&soffs, 0..) |*x, j| {
            drop(x);
            const w: u32 = @intCast(j * words);
            x.* = mlx.mlx_array_new_data(&w, &[_]c_int{1}, 1, .uint32);
            if (x.ctx == null) return error.ScratchAttnSoff;
        }
        g_rows = rows;
        g_heads = heads;
        g_nk = nk;
    }
    try ensureNoDep();
    live = true;
    return true;
}

fn ensureNoDep() !void {
    if (no_dep.ctx != null) return;
    const z: u32 = 0;
    no_dep = mlx.mlx_array_new_data(&z, &[_]c_int{1}, 1, .uint32);
    if (no_dep.ctx == null) return error.ScratchAttnDep;
}

/// A `[1]` uint32 zeros: what a launch with nothing to follow binds as its `dep`.
pub fn noDep() A {
    return no_dep;
}

/// The layer's end: no span takes the ring until the next wide layer begins.
pub fn layerEnd() void {
    live = false;
}

/// The slot a span of `rows` rows, `heads` heads and `nk` columns takes (the next one), or null (the fresh path): not
/// live, or the span past a slot's geometry.
pub fn take(rows: usize, heads: usize, nk: usize) ?u1 {
    if (!live or !ring_regions) return null;
    if (rows == 0 or rows > g_rows or heads != g_heads or nk > g_nk) return null;
    const j = next;
    next +%= 1;
    return j;
}

/// Slot `j`'s offset input, and what its next qk binds as `dep` (the slot's last pv's carrier, else zeros).
pub fn soff(j: u1) A {
    return soffs[j];
}
/// Slot `j`'s first word (in S and in P).
pub fn slotWord(j: u1) u32 {
    return @intCast(@as(usize, j) * g_rows * g_heads * g_nk);
}
pub fn depFor(j: u1) A {
    return if (last_read[j].ctx != null) last_read[j] else no_dep;
}
/// Slot `j`'s pv's carrier: the slot's next qk binds it.
pub fn readBy(j: u1, carrier: A) !void {
    try hold(&last_read[j], carrier);
}

/// The decode phase change (or the model's end): the regions and the ring's carriers let go.
pub fn releaseAll() void {
    s_region.free();
    p_region.free();
    for (&run_regions) |*r| r.free();
    for (&run_release) |*x| drop(x);
    for (&run_last) |*x| drop(x);
    for (run_all[0..run_n]) |*x| drop(x);
    run_n = 0;
    h_rows = 0;
    h_groups = 0;
    h_in = 0;
    heads_live = false;
    run_live = false;
    span_live = false;
    for (&last_read) |*x| drop(x);
    for (&soffs) |*x| drop(x);
    drop(&no_dep);
    g_rows = 0;
    g_heads = 0;
    g_nk = 0;
    live = false;
}

fn hold(slot: *A, x: A) !void {
    var held = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(held);
    try mlx.check(mlx.mlx_array_set(&held, x));
    drop(slot);
    slot.* = held;
}
fn drop(slot: *A) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = none;
}

// ── the run regions (phase B) ───────────────────────────────────────────────────────────────────────────────────────
// A wide layer's WOA_ROWS run (`woa.rows` rows: 16 spans of 256) joined its spans' grouped heads with `concat` ([og, rows, in], 268 MB
// fresh a run), and turboexl3's grouped linear then made the `[rows, og, in]` it reads from that join's transpose (another
// copy). Phase B's pv twin writes each span's heads straight into its run's region in `[rows, og, in]` at its rows (the
// stock grouped store moved to that layout, the same words), and the run's `wo_a` reads the region as it is: no join, no
// transpose. Two run regions by the run's parity a layer; a run's region is published (the host's event wait over the
// region and the last pv carrier of each span parity) for its `wo_a`, whose output the region's next writers bind
// (`dep2`). A run that is not full (the layer's last, 4095 rows at 16383) keeps the join.

pub var run_regions: [slots]scratch.Region = .{ .{}, .{} };
/// Each run region's last reader (its run's `wo_a` output): the region's next writers bind it as `dep2`.
var run_release: [slots]A = .{ none, none };
/// The run regions' geometry: rows, groups, a group's columns.
var h_rows: usize = 0;
var h_groups: usize = 0;
var h_in: usize = 0;
/// The layer's runs take the regions; the next run's region; the current run's and span's targets.
var heads_live = false;
var run_next: u1 = 0;
var run_cur: u1 = 0;
var run_live = false;
var span_live = false;
var span_row: u32 = 0;
/// The current run's last pv carrier of each span parity (the publish gates both).
var run_last: [slots]A = .{ none, none };
/// MLX_SERVE_DSV41_FLASH_ATTN: every span's carrier of the current run (its spans are not chained; the publish gates all).
const max_run_spans = 32;
var run_all: [max_run_spans]A = @splat(none);
var run_n: usize = 0;

/// A wide layer's runs (after `layerBegin`): the run regions made at `rows x groups x in` of `dt` (or kept); false (the
/// join) not `eligible`, without the ring live, or past the regions' cap.
pub fn headsLayerBegin(eligible: bool, rows: usize, groups: usize, in: usize, dt: mlx.mlx_dtype) !bool {
    heads_live = false;
    run_next = 0;
    run_live = false;
    span_live = false;
    if (!eligible or !live) return false;
    const shape = [_]c_int{ @intCast(rows), @intCast(groups), @intCast(in) };
    for (&run_regions) |*r| if ((try r.ensure(&shape, dt)) == null) return false;
    if (h_rows != rows or h_groups != groups or h_in != in) {
        for (&run_release) |*x| drop(x);
        h_rows = rows;
        h_groups = groups;
        h_in = in;
    }
    heads_live = true;
    return true;
}

/// Before a span's attention: whether its run takes a region (`full`: the run has its `woa.rows` rows), whether the span
/// starts the run, and its first row in the run.
pub fn headsSpan(full: bool, starts_run: bool, row: usize) void {
    if (starts_run) {
        for (&run_last) |*x| drop(x);
        for (run_all[0..run_n]) |*x| drop(x);
        run_n = 0;
    }
    headsSpanAt(full, starts_run, row);
}

/// `headsSpan`'s bookkeeping (no MLX call: the plugin's tests link no MLX).
fn headsSpanAt(full: bool, starts_run: bool, row: usize) void {
    if (starts_run) {
        run_cur = run_next;
        run_live = heads_live and full;
    }
    span_live = run_live;
    span_row = @intCast(row);
}

/// The span's target when its heads go into the run region: the region, the span's first row, what its writer binds as
/// `dep2` (the region's last reader, else zeros); null: the span's heads are MLX's own (the join).
pub const Target = struct { region: A, row: u32, dep2: A };
pub fn headsTarget() ?Target {
    if (!span_live) return null;
    return .{ .region = run_regions[run_cur].arr, .row = span_row, .dep2 = if (run_release[run_cur].ctx != null) run_release[run_cur] else no_dep };
}
/// A span's heads pv wrote the region: its carrier, by the span's ring slot `j` (the span parity).
pub fn headsWrote(j: u1, carrier: A) !void {
    try hold(&run_last[j], carrier);
}
/// MLX_SERVE_DSV41_FLASH_ATTN: a span's flash kernel wrote the region (no chain between spans): its carrier joins the
/// run's publish.
pub fn headsWroteFree(carrier: A) !void {
    if (run_n == max_run_spans) return error.ScratchHeadsRunSpans;
    try hold(&run_all[run_n], carrier);
    run_n += 1;
}

/// A run's end: its region published for its `wo_a` (a new handle the caller owns; the host's event wait over the
/// region and the last pv carrier of each span parity), the next run on the other region; null: the run kept the join.
pub fn headsRunEnd(s: mlx.mlx_stream) !?A {
    if (!run_live) {
        run_next +%= 1;
        return null;
    }
    run_live = false;
    span_live = false;
    run_next +%= 1;
    var gated: [3 + max_run_spans]A = undefined;
    var n: usize = 1;
    gated[0] = run_regions[run_cur].arr;
    for (run_last) |c| {
        if (c.ctx != null) {
            gated[n] = c;
            n += 1;
        }
    }
    for (run_all[0..run_n]) |c| {
        gated[n] = c;
        n += 1;
    }
    if (n == 1) return error.ScratchHeadsUnwritten;
    var outs: [3 + max_run_spans]A = undefined;
    for (outs[0..n]) |*x| x.* = mlx.mlx_array_new();
    defer for (outs[1..n]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    errdefer _ = mlx.mlx_array_free(outs[0]);
    if (dsv41ev_wait(&gated, n, try readyEvent(), 1, null, 0, true, s, &outs) != 0) return error.ScratchHeadsPublish;
    return outs[0];
}
/// The run region `run`'s reader (its `wo_a` output): the region's next writers bind it.
pub fn headsReleased(token: A) !void {
    try hold(&run_release[run_cur], token);
}
/// The layer's end.
pub fn headsLayerEnd() void {
    heads_live = false;
    run_live = false;
    span_live = false;
}

/// The host's event wait (src/host_shims.zig) over an event already at 1 (scratch.zig's `publish`, with more gated
/// carriers).
extern fn dsv41ev_create_metal(start: u64, object: *u64) i32;
extern fn dsv41ev_wait(xs: [*]const A, n: usize, event: i32, value: u64, deps: ?[*]const A, n_deps: usize, track_inputs: bool, s: mlx.mlx_stream, outs: [*]A) c_int;
var ready_event: i32 = -1;
fn readyEvent() !i32 {
    if (ready_event < 0) {
        var object: u64 = 0;
        const id = dsv41ev_create_metal(1, &object);
        if (id < 0) return error.ScratchHeadsEvent;
        ready_event = id;
    }
    return ready_event;
}

test "scratch_heads spans: a full run's spans take its region, a run that is not full keeps the join" {
    // (pure: no MLX call; the regions' arrays stay null)
    const saved = .{ heads_live, run_next, run_cur, run_live, span_live, span_row };
    defer {
        heads_live = saved[0];
        run_next = saved[1];
        run_cur = saved[2];
        run_live = saved[3];
        span_live = saved[4];
        span_row = saved[5];
    }
    heads_live = true;
    run_next = 1;
    headsSpanAt(true, true, 0);
    try std.testing.expect(headsTarget() != null);
    try std.testing.expectEqual(@as(u1, 1), run_cur);
    headsSpanAt(true, false, 256);
    try std.testing.expectEqual(@as(u32, 256), headsTarget().?.row);
    headsSpanAt(false, true, 0);
    try std.testing.expect(headsTarget() == null);
}

test "scratch_attn take: none unless live; a span past a slot's geometry fresh; the slots alternate" {
    // (pure: the plugin's tests link no MLX, so `layerBegin` itself is not called here)
    const saved = .{ live, next, g_rows, g_heads, g_nk };
    defer {
        live = saved[0];
        next = saved[1];
        g_rows = saved[2];
        g_heads = saved[3];
        g_nk = saved[4];
    }
    live = false;
    try std.testing.expect(take(256, 64, 640) == null);
    live = true;
    next = 0;
    g_rows = 256;
    g_heads = 64;
    g_nk = 640;
    try std.testing.expectEqual(@as(?u1, 0), take(256, 64, 640));
    try std.testing.expectEqual(@as(?u1, 1), take(255, 64, 128));
    try std.testing.expectEqual(@as(?u1, 0), take(1, 64, 512));
    try std.testing.expect(take(257, 64, 640) == null);
    try std.testing.expect(take(256, 32, 640) == null);
    try std.testing.expect(take(256, 64, 641) == null);
    try std.testing.expect(take(0, 64, 640) == null);
    try std.testing.expectEqual(@as(?u1, 1), take(256, 64, 640));
}
