//! `MLX_SERVE_DSV41_SCRATCH_WAVES` (read at load; on by default: unset, empty or `1`; `0`: off; `check`: the fresh
//! waves served, each region-backed launch's output compared with the fresh one's; anything else is refused by name): a
//! wide prompt call's DIG-X waves (the EXL3 stream quant's prefill lane) write their intermediates into per-prompt
//! regions (src/scratch.zig's `Region`: one MTLBuffer each, from the host's `alloc_uninit`) instead of fresh MLX
//! buffers.
//!
//! A wave of n rows runs five launches, each output a fresh array in the stock lane: take2's x0 and x1 (f16
//! [n,1,5120]), the gate|up GEMM's zg and zu (f32 [n,2304]), onepass's hd (f16 [n,1,2304]), the down GEMM's z (f32
//! [n,5120]) and rot_widen1's y (bf16 [n,1,5120]). Two waves are in flight (the tier shape's `inflight`): a wave is
//! submitted once all but one older wave have been drained to completion on the host. So two sets of the six
//! intermediates (by the wave's parity, sized for the shape's 7168-row budget, 2 x 459 MB) carry every wave, a set
//! reused only after the wave two before drained. y stays MLX's own output (the combine's part, 10,240 B of a wave
//! row's 74,240 B).
//!
//! Each launch writing a set is its stock text's write-into twin (src/wave_twins.zig, derived at construction from the
//! registry's pinned stock texts and compiled then, by the copied quant's `IntoTwins`): its outputs come from the set's
//! regions, bound whole as inputs (evaluated arrays, so no `contiguous` copy stands in for them), and its only MLX
//! output is a one-word carrier; each launch binds the previous launch's carrier (`dep`), so MLX's barrier or fence
//! orders the wave's launches as their data flow. rot_widen1 is a dep twin: the stock text with `dep` bound, its y
//! MLX's output. The kernels index only the wave's rows (the grid and the wave table), so a whole region serves every
//! wave size. A carried call (its last waves left in flight past it), a solo wave (rows above a set's) and a call the
//! regions do not fit (`scratch.cap_bytes`) take the fresh path. Exact: the same texts' words, only where they land
//! changes.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const scratch = @import("scratch.zig");
/// The twins' text derivation (std only), for the copied quant.
pub const twins = @import("wave_twins.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Mode = scratch.Mode;
pub var mode: Mode = .on;

pub const Error = error{ScratchWavesSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// On unless the setting says otherwise (src/scratch.zig's `parse` is off by default, for its other levers).
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.ScratchWavesSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// A set's rows: the served tier shape's row budget (exl3 stream quant `PrefillShape.tier.row_budget`); a wave above it
/// (solo) takes the fresh path.
pub const set_rows: u32 = 7168;
const hidden: c_int = 5120;
const inter: c_int = 2304;

/// One wave's six intermediates' regions.
pub const Set = struct {
    x0: scratch.Region = .{},
    x1: scratch.Region = .{},
    zg: scratch.Region = .{},
    zu: scratch.Region = .{},
    hd: scratch.Region = .{},
    z: scratch.Region = .{},

    fn ensure(s: *Set) !bool {
        const r: c_int = @intCast(set_rows);
        const wide = [_]c_int{ r, 1, hidden };
        const gu = [_]c_int{ r, inter };
        const mid = [_]c_int{ r, 1, inter };
        const dn = [_]c_int{ r, hidden };
        if ((try s.x0.ensure(&wide, .float16)) == null) return false;
        if ((try s.x1.ensure(&wide, .float16)) == null) return false;
        if ((try s.zg.ensure(&gu, .float32)) == null) return false;
        if ((try s.zu.ensure(&gu, .float32)) == null) return false;
        if ((try s.hd.ensure(&mid, .float16)) == null) return false;
        if ((try s.z.ensure(&dn, .float32)) == null) return false;
        return true;
    }
    fn free(s: *Set) void {
        inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| @field(s, f).free();
    }
};

pub var sets: [2]Set = .{ .{}, .{} };
/// Whether the call's waves may take the regions.
var call_live: bool = false;
/// A `[1]` uint32 zeros the first launch of a wave binds as its `dep` (nothing to order: the set's last user drained).
var no_dep: A = none;

/// Counts since the load (the meters' line): waves written into the regions, waves on the fresh path.
pub var waves_in: u64 = 0;
pub var waves_fresh: u64 = 0;

/// `check`: one launch's output against the fresh launch's, word for word: the words, those that differ, the largest
/// absolute difference, the largest distance in units in the last place (the words' integer distance) and the words
/// more than one ulp apart.
pub const Stats = struct { words: u64 = 0, differ: u64 = 0, max_abs: f64 = 0, max_ulps: u64 = 0, over1: u64 = 0 };
/// The compared outputs, by launch.
pub const Out = enum { x0, x1, zg, zu, hd, z, y };
pub var stats: std.enums.EnumArray(Out, Stats) = .initFill(.{});

/// A wide routed call begins (the source, before its routes). The regions are made at their first use and kept while
/// their shapes hold; false when they would pass the regions' cap (the call's waves take the fresh path).
pub fn callBegin() !bool {
    call_live = false;
    if (mode == .off) return false;
    if (!try sets[0].ensure() or !try sets[1].ensure()) return false;
    if (no_dep.ctx == null) {
        no_dep = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32);
        if (no_dep.ctx == null) return error.ScratchWavesDep;
    }
    call_live = true;
    return true;
}

/// A call's end (the source, after its combine is built): no wave takes the regions until the next call begins.
pub fn callEnd() void {
    call_live = false;
}

/// Whether a wave of `rows` takes the regions now (`views`' check, pure): a live call, not a solo wave (rows above a
/// set's).
fn fits(rows: u32) bool {
    return call_live and rows > 0 and rows <= set_rows;
}

/// One arena wave's destinations: set `parity`'s whole regions (new handles the caller owns, of evaluated arrays: a
/// row-contiguous kernel binds them as they are) and the first launch's `dep`. Null: the fresh path (no live call, a
/// solo wave).
pub const Views = struct { x0: A, x1: A, zg: A, zu: A, hd: A, z: A, dep: A };
pub fn views(parity: u1, rows: u32) !?Views {
    if (!fits(rows)) return null;
    const st = &sets[parity];
    var v: Views = .{ .x0 = none, .x1 = none, .zg = none, .zu = none, .hd = none, .z = none, .dep = no_dep };
    errdefer inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| {
        if (@field(v, f).ctx != null) _ = mlx.mlx_array_free(@field(v, f));
    };
    inline for (.{ "x0", "x1", "zg", "zu", "hd", "z" }) |f| @field(v, f) = try handle(@field(st, f).arr);
    return v;
}
fn handle(x: A) !A {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, x));
    return h;
}

/// The decode phase change (or the model's end): every region let go.
pub fn releaseAll() void {
    sets[0].free();
    sets[1].free();
    if (no_dep.ctx != null) _ = mlx.mlx_array_free(no_dep);
    no_dep = none;
    call_live = false;
}

/// `check`: the region's first `rows` rows (`region`, a whole region) against the fresh output `fresh` (one shape
/// after the slice, 2- or 4-byte words), into `stats[out]`: on the device the words compared, those that differ, the
/// largest |a - b| in f32, the largest integer distance of the words and the words more than one apart; read back.
pub fn compare(s: mlx.mlx_stream, out: Out, region: A, rows: u32, fresh: A) !void {
    const item = mlx.mlx_array_itemsize(fresh);
    const wt: mlx.mlx_dtype = switch (item) {
        2 => .uint16,
        4 => .uint32,
        else => return error.ScratchCheckWords,
    };
    var t: [16]A = undefined;
    for (&t) |*x| x.* = mlx.mlx_array_new();
    defer for (t) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // the region's rows, at the fresh output's shape
    const start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const step: [4]c_int = @splat(1);
    const nd: usize = mlx.mlx_array_ndim(region);
    if (nd == 0 or nd > 4) return error.ScratchCheckShape;
    const rshape = mlx.mlx_array_shape(region);
    for (0..nd) |i| stop[i] = rshape[i];
    stop[0] = @intCast(rows);
    try mlx.check(mlx.mlx_slice(&t[0], region, &start, nd, &stop, nd, &step, nd, s));
    const fshape = mlx.mlx_array_shape(fresh);
    try mlx.check(mlx.mlx_reshape(&t[1], t[0], fshape, mlx.mlx_array_ndim(fresh), s));
    // the words: equal (summed as MLX sums a bool array), and as integers (the ulp distance of same-sign floats)
    try mlx.check(mlx.mlx_view(&t[2], t[1], wt, s));
    try mlx.check(mlx.mlx_view(&t[3], fresh, wt, s));
    try mlx.check(mlx.mlx_equal(&t[4], t[2], t[3], s));
    try mlx.check(mlx.mlx_sum(&t[5], t[4], false, s));
    try mlx.check(mlx.mlx_astype(&t[6], t[2], .int64, s));
    try mlx.check(mlx.mlx_astype(&t[7], t[3], .int64, s));
    try mlx.check(mlx.mlx_subtract(&t[8], t[6], t[7], s));
    try mlx.check(mlx.mlx_abs(&t[9], t[8], s));
    try mlx.check(mlx.mlx_max(&t[10], t[9], false, s));
    const one = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_greater(&t[11], t[9], one, s));
    try mlx.check(mlx.mlx_sum(&t[12], t[11], false, s));
    // the values: the largest |a - b| in f32
    try mlx.check(mlx.mlx_astype(&t[13], t[1], .float32, s));
    try mlx.check(mlx.mlx_astype(&t[14], fresh, .float32, s));
    try mlx.check(mlx.mlx_subtract(&t[15], t[13], t[14], s));
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, t[15], s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    // read back: the counts as uint32 (a wave's output is under 2^31 words), the distances as f32 (the ulp count
    // exact to 2^24)
    var c: [4]A = undefined;
    for (&c) |*x| x.* = mlx.mlx_array_new();
    defer for (c) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_astype(&c[0], t[5], .uint32, s));
    try mlx.check(mlx.mlx_astype(&c[1], t[12], .uint32, s));
    try mlx.check(mlx.mlx_astype(&c[2], t[10], .float32, s));
    try mlx.check(mlx.mlx_astype(&c[3], mx, .float32, s));
    const ev = mlx.mlx_vector_array_new_data(&c, c.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const n: u64 = mlx.mlx_array_size(fresh);
    const same: u64 = (mlx.mlx_array_data_uint32(c[0]) orelse return error.NoData)[0];
    const over1: u64 = (mlx.mlx_array_data_uint32(c[1]) orelse return error.NoData)[0];
    const ulps: f32 = (mlx.mlx_array_data_float32(c[2]) orelse return error.NoData)[0];
    const max_abs: f64 = (mlx.mlx_array_data_float32(c[3]) orelse return error.NoData)[0];
    const st = stats.getPtr(out);
    st.words += n;
    st.differ += n - same;
    st.over1 += over1;
    st.max_ulps = @max(st.max_ulps, @as(u64, @intFromFloat(@min(ulps, 1.8e19))));
    if (max_abs > st.max_abs or std.math.isNan(max_abs)) st.max_abs = max_abs;
}

/// The meters' line (`[scratch-waves]`), "" off; `check` adds every launch's words, differing words, largest absolute
/// difference, largest ulp distance and words more than one ulp apart.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[scratch-waves] mode={t} waves_in={d} waves_fresh={d}", .{ mode, waves_in, waves_fresh }) catch return buf[0..0];
    if (mode == .check) {
        var words: u64 = 0;
        var differ: u64 = 0;
        for (std.enums.values(Out)) |o| {
            const st = stats.get(o);
            words += st.words;
            differ += st.differ;
            w.print(" {t}:words={d},differ={d},max_abs={e},max_ulps={d},over1={d}", .{ o, st.words, st.differ, st.max_abs, st.max_ulps, st.over1 }) catch return buf[0..0];
        }
        w.print(" check_words={d} check_differ={d}", .{ words, differ }) catch return buf[0..0];
    }
    w.writeAll("\n") catch return buf[0..0];
    return w.buffered();
}

test "scratch_waves setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| {
        try configure(v);
        try std.testing.expectEqual(Mode.on, mode);
    }
    try configure("0");
    try std.testing.expectEqual(Mode.off, mode);
    try configure("check");
    try std.testing.expectEqual(Mode.check, mode);
    for ([_][]const u8{ "on", "2", "Check", " 1" }) |v| try std.testing.expectError(error.ScratchWavesSetting, configure(v));
    try configure(null);
}

test "scratch_waves fits: none without a live call or for a solo wave" {
    // (pure: the plugin's tests link no MLX, so `views` itself is not called here)
    const saved = call_live;
    defer call_live = saved;
    call_live = false;
    try std.testing.expect(!fits(16));
    call_live = true;
    try std.testing.expect(fits(16));
    try std.testing.expect(fits(set_rows));
    try std.testing.expect(!fits(0));
    try std.testing.expect(!fits(set_rows + 1));
}
