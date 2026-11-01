//! `MLX_SERVE_DSV41_SCRATCH_ATTN` (read at load; unset, empty or `1`: on; `0`: off; `check`: the fresh path served,
//! every ring span's scores, probabilities and heads compared with the fresh ones; anything else is refused by name): a
//! wide prompt layer's attention spans write their scores and probabilities into a ring of two per-prompt slots
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

pub const Mode = scratch.Mode;
pub var mode: Mode = .on;

pub fn configure(value: ?[]const u8) scratch.Error!void {
    mode = try parse(value);
}

/// On unless the setting says otherwise (src/scratch.zig's `parse` is off by default, for its other levers).
pub fn parse(value: ?[]const u8) scratch.Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.ScratchSetting;
}

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
/// The slot geometry the regions hold: rows, heads, columns a slot.
var g_rows: usize = 0;
var g_heads: usize = 0;
var g_nk: usize = 0;
/// The layer's next span's slot, and whether its spans take the ring.
var next: u1 = 0;
var live = false;

/// Counts since the load: spans on the ring, spans a live layer sent to the fresh path (rows, heads or columns past a
/// slot's).
pub var spans_in: u64 = 0;
pub var spans_fresh: u64 = 0;

/// A wide prompt layer starts: the regions made at `rows x heads x nk` words a slot (or kept), the ring live for its
/// spans; false (the fresh path) off, on a CPU stream or without a streamed bank (`eligible`), or past the regions' cap.
pub fn layerBegin(eligible: bool, rows: usize, heads: usize, nk: usize) !bool {
    live = false;
    next = 0;
    if (mode == .off or !eligible) return false;
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
    if (no_dep.ctx == null) {
        const z: u32 = 0;
        no_dep = mlx.mlx_array_new_data(&z, &[_]c_int{1}, 1, .uint32);
        if (no_dep.ctx == null) return error.ScratchAttnDep;
    }
    live = true;
    return true;
}

/// The layer's end: no span takes the ring until the next wide layer begins.
pub fn layerEnd() void {
    live = false;
}

/// The slot a span of `rows` rows, `heads` heads and `nk` columns takes (the next one), or null (the fresh path): not
/// live, or the span past a slot's geometry.
pub fn take(rows: usize, heads: usize, nk: usize) ?u1 {
    if (!live) return null;
    if (rows == 0 or rows > g_rows or heads != g_heads or nk > g_nk) {
        spans_fresh += 1;
        return null;
    }
    const j = next;
    next +%= 1;
    spans_in += 1;
    return j;
}

/// Slot `j`'s offset input, and what its next qk binds as `dep` (the slot's last pv's carrier, else zeros).
pub fn soff(j: u1) A {
    return soffs[j];
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
    for (&last_read) |*x| drop(x);
    for (&soffs) |*x| drop(x);
    drop(&no_dep);
    g_rows = 0;
    g_heads = 0;
    g_nk = 0;
    live = false;
    // a check run's layer stats are reported at each wide layer's end; any left (the load's warm launches) dropped
    layer_stats = .initFill(.{});
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

/// `check`: a span's ring result against the fresh one, word for word: the words, those that differ, the largest
/// absolute difference, the largest distance in units in the last place (the words' integer distance) and the words
/// more than one ulp apart.
pub const Stats = struct { words: u64 = 0, differ: u64 = 0, max_abs: f64 = 0, max_ulps: u64 = 0, over1: u64 = 0 };
pub const Out = enum { s, p, o };
/// The layer's, and the totals since the load.
pub var layer_stats: std.enums.EnumArray(Out, Stats) = .initFill(.{});
pub var total_stats: std.enums.EnumArray(Out, Stats) = .initFill(.{});

/// `check`: slot `j` of `region` (its words from the slot's first, at `fresh`'s shape; `carrier`: the launch that wrote
/// it) against `fresh`, into the layer's stats `out`.
pub fn compareSlot(s: mlx.mlx_stream, out: Out, region: *scratch.Region, j: u1, carrier: A, fresh: A) !void {
    try region.wrote(carrier);
    const seen = try region.publish(s);
    defer _ = mlx.mlx_array_free(seen);
    const n: c_int = @intCast(mlx.mlx_array_size(fresh));
    const lo: c_int = @intCast(@as(usize, j) * g_rows * g_heads * g_nk);
    var sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl);
    try mlx.check(mlx.mlx_slice(&sl, seen, &[_]c_int{lo}, 1, &[_]c_int{lo + n}, 1, &[_]c_int{1}, 1, s));
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, sl, mlx.mlx_array_shape(fresh), mlx.mlx_array_ndim(fresh), s));
    try compare(s, out, r, fresh);
}

/// `check`: `a` (the ring's) against `fresh` (one shape, 2- or 4-byte words), into the layer's stats `out`.
pub fn compare(s: mlx.mlx_stream, out: Out, a: A, fresh: A) !void {
    const wt: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(fresh)) {
        2 => .uint16,
        4 => .uint32,
        else => return error.ScratchCheckWords,
    };
    var t: [14]A = undefined;
    for (&t) |*x| x.* = mlx.mlx_array_new();
    defer for (t) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_view(&t[0], a, wt, s));
    try mlx.check(mlx.mlx_view(&t[1], fresh, wt, s));
    try mlx.check(mlx.mlx_equal(&t[2], t[0], t[1], s));
    try mlx.check(mlx.mlx_sum(&t[3], t[2], false, s));
    try mlx.check(mlx.mlx_astype(&t[4], t[0], .int64, s));
    try mlx.check(mlx.mlx_astype(&t[5], t[1], .int64, s));
    try mlx.check(mlx.mlx_subtract(&t[6], t[4], t[5], s));
    try mlx.check(mlx.mlx_abs(&t[7], t[6], s));
    try mlx.check(mlx.mlx_max(&t[8], t[7], false, s));
    const one = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_greater(&t[9], t[7], one, s));
    try mlx.check(mlx.mlx_sum(&t[10], t[9], false, s));
    try mlx.check(mlx.mlx_astype(&t[11], a, .float32, s));
    try mlx.check(mlx.mlx_astype(&t[12], fresh, .float32, s));
    try mlx.check(mlx.mlx_subtract(&t[13], t[11], t[12], s));
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, t[13], s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    // read back: the counts as uint32 (a span's array is under 2^31 words), the distances as f32
    var c: [4]A = undefined;
    for (&c) |*x| x.* = mlx.mlx_array_new();
    defer for (c) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_astype(&c[0], t[3], .uint32, s));
    try mlx.check(mlx.mlx_astype(&c[1], t[10], .uint32, s));
    try mlx.check(mlx.mlx_astype(&c[2], t[8], .float32, s));
    try mlx.check(mlx.mlx_astype(&c[3], mx, .float32, s));
    const ev = mlx.mlx_vector_array_new_data(&c, c.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const n: u64 = mlx.mlx_array_size(fresh);
    const same: u64 = (mlx.mlx_array_data_uint32(c[0]) orelse return error.NoData)[0];
    const over1: u64 = (mlx.mlx_array_data_uint32(c[1]) orelse return error.NoData)[0];
    const ulps: f32 = (mlx.mlx_array_data_float32(c[2]) orelse return error.NoData)[0];
    const max_abs: f64 = (mlx.mlx_array_data_float32(c[3]) orelse return error.NoData)[0];
    const st = layer_stats.getPtr(out);
    st.words += n;
    st.differ += n - same;
    st.over1 += over1;
    st.max_ulps = @max(st.max_ulps, @as(u64, @intFromFloat(@min(ulps, 1.8e19))));
    if (max_abs > st.max_abs or std.math.isNan(max_abs)) st.max_abs = max_abs;
}

/// `check`: the layer's `[scratch-attn] check` line (each of s, p and o: words, differing words, largest absolute
/// difference, largest ulp distance, words more than one ulp apart; the totals since the load), the layer's stats then
/// added to the totals and cleared.
pub fn report(layer: usize) void {
    var words: u64 = 0;
    var differ: u64 = 0;
    for (std.enums.values(Out)) |o| {
        const st = layer_stats.get(o);
        const tot = total_stats.getPtr(o);
        tot.words += st.words;
        tot.differ += st.differ;
        tot.over1 += st.over1;
        tot.max_ulps = @max(tot.max_ulps, st.max_ulps);
        if (st.max_abs > tot.max_abs or std.math.isNan(st.max_abs)) tot.max_abs = st.max_abs;
        words += tot.words;
        differ += tot.differ;
    }
    const sl = layer_stats.get(.s);
    const pl = layer_stats.get(.p);
    const ol = layer_stats.get(.o);
    std.debug.print("[scratch-attn] check layer={d} s:words={d},differ={d},max_abs={e},max_ulps={d},over1={d} p:words={d},differ={d},max_abs={e},max_ulps={d},over1={d} o:words={d},differ={d},max_abs={e},max_ulps={d},over1={d} spans_in={d} spans_fresh={d} total_words={d} total_differ={d}\n", .{ layer, sl.words, sl.differ, sl.max_abs, sl.max_ulps, sl.over1, pl.words, pl.differ, pl.max_abs, pl.max_ulps, pl.over1, ol.words, ol.differ, ol.max_abs, ol.max_ulps, ol.over1, spans_in, spans_fresh, words, differ });
    layer_stats = .initFill(.{});
}

test "scratch_attn setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| {
        try configure(v);
        try std.testing.expectEqual(Mode.on, mode);
    }
    try configure("0");
    try std.testing.expectEqual(Mode.off, mode);
    try configure("check");
    try std.testing.expectEqual(Mode.check, mode);
    for ([_][]const u8{ "on", "2", "Check", " 1" }) |v| try std.testing.expectError(error.ScratchSetting, configure(v));
    try configure(null);
}

test "scratch_attn take: none unless live; a span past a slot's geometry fresh; the slots alternate" {
    // (pure: the plugin's tests link no MLX, so `layerBegin` itself is not called here)
    const saved = .{ live, next, g_rows, g_heads, g_nk, spans_in, spans_fresh };
    defer {
        live = saved[0];
        next = saved[1];
        g_rows = saved[2];
        g_heads = saved[3];
        g_nk = saved[4];
        spans_in = saved[5];
        spans_fresh = saved[6];
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
