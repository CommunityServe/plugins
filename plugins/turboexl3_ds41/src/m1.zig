//! `MLX_SERVE_DSV41_M1` (read at load; m1): a wide prompt call's DIG-X waves (the copied EXL3 stream quant's prefill
//! lane) run their 128-row routed GEMMs (K3 and K2, gate|up and down) on texts with the B stage double-buffered
//! (`dsv41_prefill_dig_gemm_*_m128d`, patches/turboexl3-zzz9k-m1.patch): stage 0 decoded before the K loop, then per
//! stage the MMA on its buffer and the next stage decoded into the other, one barrier a stage. Every output's A
//! fragments, decoded B values and tile_matmad_nax sequence are the 128-row text's, so the same z words (the registry's
//! twin check at load, bitwise; the EXL3 expert-kernels lane's m1u microbench: 28 of 28 wave records equal word for
//! word). On, SCRATCH_WAVES' write-into twins derive from the double-buffered texts (the copied quant's `IntoTwins`).
//!
//! Settings: unset, empty or `1` is on (merged on: zzz9k); `0` is off (the 128-row texts). `check` serves the 128-row
//! texts and launches the double-buffered ones beside them on every wave's inputs (the waves off the scratch regions,
//! so every GEMM is compared) and compares their f32 words on the device (a sync a GEMM: its timings mean nothing). Any
//! other value is refused by name.
//!
//! `[m1]` at each request's end, on or check: the mode; in check the request's GEMMs, words, differing words and the
//! largest ulp distance, per gate|up and down.
//!
//! The reads (M5 Max): m1u's microbench (w-m1u-20261102-0007; 8 experts a wave, 1 warm and 9 timed runs, median µs a
//! launch, 128-row text against the double-buffered one): the 16K shape's wave (8 x 256 rows) 1789.5 -> 1707.3 (-4.6%)
//! on K3's gate|up, 886.6 -> 878.1 (-1.0%) on its down, 1755.6 -> 1702.4 (-3.0%) and 872.9 -> 861.8 (-1.3%) on K2's;
//! every other wave -0.9% to -6.5%. Served as m1 (w-pio-20261102-0941; a cold code-ctx16384 card then the new-text
//! slice 3, 32 tokens, a = ctl25 5ccb44791b50, b = m1 3466f8330cbd at 1, both on G_KIT1271; a1 b1 b2 a2): cold
//! 1397.4 1399.6 against 1397.1 1405.7 tok/s (+0.21%; pairs -0.02, +0.44); new text 1489.8 1474.6 against 1487.4
//! 1501.7 tok/s (+0.83%; pairs -0.16, +1.84); the cold card's encoder MoE stage 4102 4097 against 4087 4057 ms; the
//! texts the same bytes in every arm, check's `[m1]` differ 0 and max_ulps 0 over 1428 and 1334 GEMMs of each kind,
//! the bank rows 149 / 174 in every measured arm, the guard's peak not up (b 106.7 against a 107.8 GiB). Merged on
//! (zzz9k).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;

pub const Mode = enum { off, on, check };

/// Set at load (`configure`); off until then.
pub var mode: Mode = .off;

pub const Error = error{M1Setting};

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.M1Setting;
}

/// At load, before the bank opens (the copied quant's `Accepted.routeM1` reads `mode` at construction).
pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// The two GEMMs of a wave.
pub const Gemm = enum { gate_up, down };

/// `check`'s totals per GEMM since the last request's end.
const Stats = struct { calls: u64 = 0, words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0 };
var stats: std.enums.EnumArray(Gemm, Stats) = .initFill(.{});

/// check: the double-buffered GEMM's outputs (`got`) against the 128-row text's (`want`), on the device: equal words
/// (summed as MLX sums a bool array) and the largest integer distance of the words (the ulp distance of same-sign
/// floats), read back at once.
pub fn compare(s: mlx.mlx_stream, which: Gemm, got: []const mlx.mlx_array, want: []const mlx.mlx_array) !void {
    std.debug.assert(got.len == want.len);
    for (got, want) |gx, wx| {
        var t: [9]mlx.mlx_array = undefined;
        for (&t) |*x| x.* = mlx.mlx_array_new();
        defer for (t) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try mlx.check(mlx.mlx_view(&t[0], gx, .uint32, s));
        try mlx.check(mlx.mlx_view(&t[1], wx, .uint32, s));
        try mlx.check(mlx.mlx_equal(&t[2], t[0], t[1], s));
        try mlx.check(mlx.mlx_sum(&t[3], t[2], false, s));
        try mlx.check(mlx.mlx_astype(&t[4], t[0], .int64, s));
        try mlx.check(mlx.mlx_astype(&t[5], t[1], .int64, s));
        try mlx.check(mlx.mlx_subtract(&t[6], t[4], t[5], s));
        try mlx.check(mlx.mlx_abs(&t[7], t[6], s));
        try mlx.check(mlx.mlx_max(&t[8], t[7], false, s));
        // read back: the equal count as uint32 (a GEMM's output is under 2^31 words), the distance as f32 (exact to 2^24)
        var c: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (c) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try mlx.check(mlx.mlx_astype(&c[0], t[3], .uint32, s));
        try mlx.check(mlx.mlx_astype(&c[1], t[8], .float32, s));
        const ev = mlx.mlx_vector_array_new_data(&c, c.len);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
        const n: u64 = mlx.mlx_array_size(wx);
        const same: u64 = (mlx.mlx_array_data_uint32(c[0]) orelse return error.NoData)[0];
        const ulps: f32 = (mlx.mlx_array_data_float32(c[1]) orelse return error.NoData)[0];
        const st = stats.getPtr(which);
        st.words += n;
        st.differ += n - same;
        st.max_ulps = @max(st.max_ulps, @as(u64, @intFromFloat(@min(ulps, 1.8e19))));
    }
    stats.getPtr(which).calls += 1;
}

/// The meters' line (`[m1]`), "" off: the mode; in check the totals since the last request's end.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const gu = stats.get(.gate_up);
    const dn = stats.get(.down);
    stats = .initFill(.{});
    var w = std.Io.Writer.fixed(buf);
    w.print("[m1] mode={t} gu_calls={d} gu_words={d} gu_differ={d} gu_max_ulps={d} dn_calls={d} dn_words={d} dn_differ={d} dn_max_ulps={d}\n", .{ mode, gu.calls, gu.words, gu.differ, gu.max_ulps, dn.calls, dn.words, dn.differ, dn.max_ulps }) catch return buf[0..0];
    return w.buffered();
}

test "m1 settings: on unset, empty or 1; 0 off; check; anything else refused by name" {
    const tt = std.testing;
    try tt.expectEqual(Mode.on, try parse(null));
    try tt.expectEqual(Mode.on, try parse(""));
    try tt.expectEqual(Mode.on, try parse("1"));
    try tt.expectEqual(Mode.off, try parse("0"));
    try tt.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "1 " }) |v| try tt.expectError(error.M1Setting, parse(v));
}
