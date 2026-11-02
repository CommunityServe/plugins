//! `MLX_SERVE_DSV41_HW2` (read at load; hw2): the t3 route's banked calls (the decode hit wave's and each miss part's)
//! with turboexl3's grouped banked GEMVs at hw2u's picked variants (`pick_gu` for the gate|up GEMV, `pick_dn` for the
//! down GEMV; turboexl3's `group_variants`, `Config.banked_variant_gu` and `banked_variant_dn`), on the grouped texts the
//! call runs (hw4b's M = 3 texts with `MLX_SERVE_DSV41_HW4B` on, the default; today's grouped texts with it off): the
//! texts under Metal defines that change a GEMV's schedule (when a tile's words load, how many lanes load them, the
//! registers the loop holds, the passes over a tile, the instructions from a codeword to t), never a word read, a decoded
//! value or a sum: the same codewords, the same decoded values, the same FMA sequence into the same f32 accumulators and
//! the same folds, so the same bytes (turboexl3's banked_ilp_test.zig: every variant's words on both texts at rows
//! 1-48, and every route to t over all 65536 codeword states).
//!
//! Settings: unset, empty or `1` is on (merged on: zzz9m) with the grouped banked decode (the t3 route and
//! `MLX_SERVE_DSV41_T3_GROUP` on, the defaults); unset or empty is off without it. `0` is off (variant 0). `check` runs
//! the call at variant 0 and at the picks for every banked call, compares their f32 words on the device (a sync a call:
//! its timings mean nothing) and serves variant 0's rows. `1` or `check` without the grouped banked decode, `check`
//! beside `MLX_SERVE_DSV41_HW4B=check` (one check mode at a time) and any other value are refused by name.
//!
//! `[hw2]` at each request's end, on or check: the mode and the picks; in check the request's calls, words, differing
//! words, the largest ulp distance and the largest absolute difference.
//!
//! The reads (M5 Max): hw2u's microbench (w-hw2u-20261102-0023; the whole banked call at a verify's 36 pairs on the
//! recorded p10, median and p90 profiles, chain medians, µs a call): today's 902.0, 1029.1 and 1239.3 at K3, 888.3,
//! 981.6 and 1162.9 at K2; t4 at both GEMVs -24.3, -21.5 and -21.3 at K3, -17.4, -21.1 and -12.6 at K2, every other
//! variant slower in every cell; every variant's words today's (968 records, 100,859,088 words), every route to t exact
//! over all 65536 codeword states. Served as hw2 (w-20261102-1025; code-ctx16384 card then greedy, a = ctl25
//! 5ccb44791b50, b = hw2 d43b09222a94 at 1, both on G_KIT1271; a1 b1 b2 a2 b3 a3 a4 b4): card 43.9 43.9 43.6 43.9
//! against 44.0 43.9 43.6 44.1 tok/s (+0.17%; pairs +0.23, 0.00, 0.00, +0.46); greedy 55.8 55.9 55.6 55.6 against 56.0
//! 56.1 55.2 55.8 (+0.09%; pairs +0.36, +0.36, -0.72, +0.36); the texts the same bytes in every arm, check's `[hw2]`
//! differ 0 and max_ulps 0 over 8531 and 7018 calls, the bank rows 150 / 174 in every arm. Measured on today's grouped
//! texts (before hw4b); merged on both (zzz9m).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// hw2u's pick (turboexl3 `group_variants` names; "" today's): the gate|up GEMV's variant and the down GEMV's.
pub const pick_gu = "t4";
pub const pick_dn = "t4";

pub const Mode = enum { off, on, check };

/// Set at load (`configure`); off until then.
pub var mode: Mode = .off;
/// The picks' variant indexes (`configure`)
var vgu: u8 = 0;
var vdn: u8 = 0;

pub const Error = error{ Hw2Setting, Hw2NeedsT3Group, Hw2Variant, Hw2CheckBesideHw4bCheck };

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.Hw2Setting;
}

/// The mode a load runs (`t3_grouped`: the t3 route with MLX_SERVE_DSV41_T3_GROUP on; `hw4b_check`:
/// MLX_SERVE_DSV41_HW4B=check): the setting (`parse`); unset or empty off without the grouped banked decode, `1` or
/// `check` without it refused, `check` beside hw4b's check refused.
pub fn resolve(value: ?[]const u8, t3_grouped: bool, hw4b_check: bool) Error!Mode {
    const m = try parse(value);
    if (m == .check and hw4b_check) return error.Hw2CheckBesideHw4bCheck;
    if (m == .off or t3_grouped) return m;
    const set = if (value) |v| v.len != 0 else false;
    return if (set) error.Hw2NeedsT3Group else .off;
}

/// At load, after turboexl3's banked group and hw4b's texts are set (`routed_t3.configureGroup`, `hw4b.configure`) and
/// before the bank opens (its warm-up builds the banked kernels): the mode (`resolve`); turboexl3's variants set from it
/// (on: the picks; off and check: variant 0, check's picked call made beside it).
pub fn configure(value: ?[]const u8, t3_grouped: bool, hw4b_check: bool) (Error || error{InvalidTurboexl3Config})!void {
    const m = try resolve(value, t3_grouped, hw4b_check);
    const g = t3.groupVariantOf(pick_gu) orelse return error.Hw2Variant;
    const d = t3.groupVariantOf(pick_dn) orelse return error.Hw2Variant;
    var c = t3.config.get().*;
    c.banked_variant_gu = if (m == .on) g else 0;
    c.banked_variant_dn = if (m == .on) d else 0;
    try t3.configure(c);
    mode = m;
    vgu = g;
    vdn = d;
}

/// The t3 route's banked call as the setting runs it (src/routed_t3.zig's `downBanked`, in check; hw4b.banked otherwise):
/// off and on turboexl3's call at the configured variants (`moeClampedBanked`: variant 0 off, the picks on), check the
/// call at variant 0 and at the picks on the configured texts (their words compared, variant 0's rows returned). The
/// caller owns the result.
pub fn banked(s: mlx.mlx_stream, x: mlx.mlx_array, banks: *const [3]t3.Bank, ids: mlx.mlx_array, dec: t3.format.Decode, limit: c_int) !mlx.mlx_array {
    switch (mode) {
        .off, .on => return t3.moeClampedBanked(s, x, banks, ids, dec, limit),
        .check => {
            const y0 = try t3.moeClampedBankedV(s, x, banks, ids, dec, limit, 0, 0);
            errdefer _ = mlx.mlx_array_free(y0);
            const yv = try t3.moeClampedBankedV(s, x, banks, ids, dec, limit, vgu, vdn);
            defer _ = mlx.mlx_array_free(yv);
            try compare(s, yv, y0);
            return y0;
        },
    }
}

/// `check`'s totals since the last request's end.
const Stats = struct { calls: u64 = 0, words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0, max_abs: f64 = 0 };
var stats: Stats = .{};

/// The picked call's f32 words against today's, on the device: equal words (summed as MLX sums a bool array), the largest
/// integer distance of the words (the ulp distance of same-sign floats), the largest absolute difference; read back at
/// once (check only).
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

/// The meters' line (`[hw2]`), "" off: the mode and the picks; in check the totals since the last request's end.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const st = stats;
    stats = .{};
    var w = std.Io.Writer.fixed(buf);
    w.print("[hw2] mode={t} gu={s} dn={s} calls={d} words={d} differ={d} max_ulps={d} max_abs={e:.3}\n", .{ mode, if (pick_gu.len == 0) "0" else pick_gu, if (pick_dn.len == 0) "0" else pick_dn, st.calls, st.words, st.differ, st.max_ulps, st.max_abs }) catch return buf[0..0];
    return w.buffered();
}

test "hw2 settings: on unset, empty or 1, off without the grouped banked decode unless set; 0 off; check, not beside hw4b's; anything else refused by name" {
    const tt = std.testing;
    try tt.expectEqual(Mode.on, try parse(null));
    try tt.expectEqual(Mode.on, try parse(""));
    try tt.expectEqual(Mode.on, try parse("1"));
    try tt.expectEqual(Mode.off, try parse("0"));
    try tt.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "1 " }) |v| try tt.expectError(error.Hw2Setting, parse(v));
    // with the grouped banked decode: the setting; check beside hw4b's check refused
    try tt.expectEqual(Mode.on, try resolve(null, true, false));
    try tt.expectEqual(Mode.on, try resolve("1", true, true));
    try tt.expectEqual(Mode.off, try resolve("0", true, true));
    try tt.expectEqual(Mode.check, try resolve("check", true, false));
    try tt.expectError(error.Hw2CheckBesideHw4bCheck, resolve("check", true, true));
    // without it: unset, empty and 0 off; 1 and check refused
    try tt.expectEqual(Mode.off, try resolve(null, false, false));
    try tt.expectEqual(Mode.off, try resolve("", false, false));
    try tt.expectEqual(Mode.off, try resolve("0", false, false));
    try tt.expectError(error.Hw2NeedsT3Group, resolve("1", false, false));
    try tt.expectError(error.Hw2NeedsT3Group, resolve("check", false, false));
    // the picks name turboexl3 variants: `configure` refuses the load by name otherwise (Hw2Variant); this test module
    // has no turboexl3
}
