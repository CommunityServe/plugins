//! `MLX_SERVE_DSV41_HW4B` (read at load; hw4b): the t3 route's banked calls (the decode hit wave's and each miss
//! part's) on turboexl3's grouped banked texts with an M = 3 member path (`Config.banked_m3`): a lead with three members
//! of an expert runs it, where the grouped texts ran the M = 4 path with a fourth member's FMAs padding. Every member is
//! fed in the one-member order (turboexl3's `turboexl3_gemv_group`), so a row-slot's bytes are the grouped texts' (the
//! EXL3 expert-kernels lane's hw4u microbench: 256 records, every word of hd, hs and y at rows 1-64, one id on 1 to 16
//! row-slots; turboexl3's src/banked_grp_test.zig on the texts).
//!
//! Settings: unset, empty or `1` is on (merged on: zzz9j) with the grouped banked decode (the t3 route and
//! `MLX_SERVE_DSV41_T3_GROUP` on, the defaults); unset or empty is off without it. `0` is off. `check` runs the grouped
//! texts' call and the M = 3 one for every banked call, compares their f32 words on the device (a sync a call: its
//! timings mean nothing) and serves the grouped texts' rows. `1` or `check` without the grouped banked decode, and any
//! other value, are refused by name.
//!
//! `[hw4b]` at each request's end, on or check: the mode and turboexl3's `banked_m3` as configured (1 on); in check the
//! request's calls, words, differing words, the largest ulp distance and the largest absolute difference.
//!
//! The reads (M5 Max): hw4u's microbench (w-hw4u-20261102-0359), against the grouped texts: -1.5 to -15 µs a banked call
//! on the recorded routes' profiles, the 3-member sweep -10%; hw5u's (w-hw5u-20261102-0648) reproduced it: -3.2 to
//! -16.8 µs a call at K3 and K2, the 3-member sweeps -9.8% to -10.9%, flat with no 3-member lead. Served as hw4b
//! (w-20261102-0923; code-ctx16384 card then greedy, a = ctl29 6e9c8b0a279a, b = hw4b 502c5238bc37 at 1, both on
//! G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 44.1 44.2 44.0 44.3 against 44.0 44.4 44.4 44.1 tok/s (+0.17%; pairs
//! -0.23, +0.45, +0.91, -0.45); greedy 56.2 56.4 56.0 56.2 against 56.1 56.6 56.6 56.3 (+0.36%; pairs -0.18, +0.35,
//! +1.07, +0.18); the texts the same bytes in every arm, check's `[hw4b]` differ 0 and max_ulps 0 over 8498 and 6949
//! calls. Merged on (zzz9j).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

pub const Mode = enum { off, on, check };

/// Set at load (`configure`); off until then.
pub var mode: Mode = .off;

pub const Error = error{ Hw4bSetting, Hw4bNeedsT3Group };

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.Hw4bSetting;
}

/// The mode a load runs (`t3_grouped`: the t3 route with MLX_SERVE_DSV41_T3_GROUP on): the setting (`parse`); unset or
/// empty off without the grouped banked decode, `1` or `check` without it refused.
pub fn resolve(value: ?[]const u8, t3_grouped: bool) Error!Mode {
    const m = try parse(value);
    if (m == .off or t3_grouped) return m;
    const set = if (value) |v| v.len != 0 else false;
    return if (set) error.Hw4bNeedsT3Group else .off;
}

/// At load, after turboexl3's banked group is set (`routed_t3.configureGroup`: 4) and before the bank opens (its warm-up
/// builds the banked kernels): the mode (`resolve`); turboexl3's `banked_m3` set on (the M = 3 texts) and off otherwise
/// (off and check keep the grouped texts, check's M = 3 call made beside them).
pub fn configure(value: ?[]const u8, t3_grouped: bool) (Error || error{InvalidTurboexl3Config})!void {
    const m = try resolve(value, t3_grouped);
    const m3 = m == .on;
    if (t3.config.get().banked_m3 != m3) {
        var c = t3.config.get().*;
        c.banked_m3 = m3;
        try t3.configure(c);
    }
    mode = m;
}

/// The t3 route's banked call as the setting runs it (src/routed_t3.zig's `downBanked`): off and on turboexl3's call as
/// configured (`moeClampedBanked`: the grouped texts off, the M = 3 ones on), check the grouped texts' call and the
/// M = 3 one (their words compared, the grouped texts' rows returned). The caller owns the result.
pub fn banked(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]t3.Bank, ids: mlx.mlx_array, dec: t3.format.Decode, limit: c_int) !mlx.mlx_array {
    switch (mode) {
        .off, .on => return t3.moeClampedBanked(s, x, banks, ids, dec, limit),
        .check => {
            const y0 = try t3.moeClampedBankedSel(s, x, banks, ids, dec, limit, .g4);
            errdefer _ = mlx.mlx_array_free(y0);
            const yw = try t3.moeClampedBankedSel(s, x, banks, ids, dec, limit, .g4m3);
            defer _ = mlx.mlx_array_free(yw);
            try compare(s, yw, y0);
            return y0;
        },
    }
}

/// `check`'s totals since the last request's end.
const Stats = struct { calls: u64 = 0, words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0, max_abs: f64 = 0 };
var stats: Stats = .{};

/// The M = 3 call's f32 words against the grouped texts', on the device: equal words (summed as MLX sums a bool array),
/// the largest integer distance of the words (the ulp distance of same-sign floats), the largest absolute difference;
/// read back at once (check only).
fn compare(s: mlx.mlx_stream, got: mlx.mlx_array, want: mlx.mlx_array) !void {
    var t: [12]mlx.mlx_array = undefined;
    for (&t) |*x| x.* = mlx.mlx_array_new();
    defer for (t) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_view(&t[0], got, .uint32, s));
    try mlx.check(mlx.mlx_view(&t[1], want, .uint32, s));
    try mlx.check(mlx.mlx_equal(&t[2], t[0], t[1], s));
    try mlx.check(mlx.mlx_sum(&t[3], t[2], false, s));
    try mlx.check(mlx.mlx_astype(&t[4], t[0], .int64, s));
    try mlx.check(mlx.mlx_astype(&t[5], t[1], .int64, s));
    try mlx.check(mlx.mlx_subtract(&t[6], t[4], t[5], s));
    try mlx.check(mlx.mlx_abs(&t[7], t[6], s));
    try mlx.check(mlx.mlx_max(&t[8], t[7], false, s));
    try mlx.check(mlx.mlx_subtract(&t[9], got, want, s));
    try mlx.check(mlx.mlx_abs(&t[10], t[9], s));
    try mlx.check(mlx.mlx_max(&t[11], t[10], false, s));
    // read back: the equal count as uint32 (a call's output is under 2^31 words), the distances as f32 (the ulp count
    // exact to 2^24)
    var c: [3]mlx.mlx_array = undefined;
    for (&c) |*x| x.* = mlx.mlx_array_new();
    defer for (c) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_astype(&c[0], t[3], .uint32, s));
    try mlx.check(mlx.mlx_astype(&c[1], t[8], .float32, s));
    try mlx.check(mlx.mlx_astype(&c[2], t[11], .float32, s));
    const ev = mlx.mlx_vector_array_new_data(&c, c.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    const n: u64 = mlx.mlx_array_size(want);
    const same: u64 = (mlx.mlx_array_data_uint32(c[0]) orelse return error.NoData)[0];
    const ulps: f32 = (mlx.mlx_array_data_float32(c[1]) orelse return error.NoData)[0];
    const max_abs: f64 = (mlx.mlx_array_data_float32(c[2]) orelse return error.NoData)[0];
    stats.calls += 1;
    stats.words += n;
    stats.differ += n - same;
    stats.max_ulps = @max(stats.max_ulps, @as(u64, @intFromFloat(@min(ulps, 1.8e19))));
    if (max_abs > stats.max_abs or std.math.isNan(max_abs)) stats.max_abs = max_abs;
}

/// The meters' line (`[hw4b]`), "" off: the mode and turboexl3's `banked_m3` as configured (1 on); in check the totals
/// since the last request's end.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const st = stats;
    stats = .{};
    var w = std.Io.Writer.fixed(buf);
    w.print("[hw4b] mode={t} m3={d} calls={d} words={d} differ={d} max_ulps={d} max_abs={e:.3}\n", .{ mode, @intFromBool(t3.config.get().banked_m3), st.calls, st.words, st.differ, st.max_ulps, st.max_abs }) catch return buf[0..0];
    return w.buffered();
}

test "hw4b settings: on unset, empty or 1, off without the grouped banked decode unless set; 0 off; check; anything else refused by name" {
    const tt = std.testing;
    try tt.expectEqual(Mode.on, try parse(null));
    try tt.expectEqual(Mode.on, try parse(""));
    try tt.expectEqual(Mode.on, try parse("1"));
    try tt.expectEqual(Mode.off, try parse("0"));
    try tt.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "1 " }) |v| try tt.expectError(error.Hw4bSetting, parse(v));
    // with the grouped banked decode: the setting
    try tt.expectEqual(Mode.on, try resolve(null, true));
    try tt.expectEqual(Mode.on, try resolve("", true));
    try tt.expectEqual(Mode.on, try resolve("1", true));
    try tt.expectEqual(Mode.off, try resolve("0", true));
    try tt.expectEqual(Mode.check, try resolve("check", true));
    // without it: unset, empty and 0 off; 1 and check refused; anything else refused by name
    try tt.expectEqual(Mode.off, try resolve(null, false));
    try tt.expectEqual(Mode.off, try resolve("", false));
    try tt.expectEqual(Mode.off, try resolve("0", false));
    try tt.expectError(error.Hw4bNeedsT3Group, resolve("1", false));
    try tt.expectError(error.Hw4bNeedsT3Group, resolve("check", false));
    try tt.expectError(error.Hw4bSetting, resolve("on", false));
    // `configure` reaches turboexl3's config; this test module has no turboexl3
}
