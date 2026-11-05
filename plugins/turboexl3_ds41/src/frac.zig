//! `MLX_SERVE_DSV41_FRAC` (read at load; unset, empty or `1`: on; `0`: off; `check`: on, and each fractional call
//! compared with the exact f64 expert; anything else is refused by name): routed layers at a half-integer EXL3 rate on
//! turboexl3, the rest on the streamer's own lanes as before.
//!
//! exllamav3 1.6 quantizes a tensor at a half-integer rate K = KA + 0.5 (exllamav3_ext/quant/frac.cu): its 256-weight
//! tile is a ring of 16 K halfwords whose weights take KA and KA + 1 fresh bits in turn (mask 0xAAAA), each weight's
//! codeword the 16 ring bits ending at the running sum, MSB-first in 32-bit words, mul1 after. That is turboexl3's
//! `Rate` at n = 16 K halfwords (bitEnd(t) = floor((t + 1) n / 16)), so its kernels take such a tile as they take an
//! integer one (only its integer-rate reader is off). turboderp's DeepSeek-V4.1-Flash-1017 EXL3 branches route layers
//! at n 40 (K 2.5: 2.25bpw's layers 0-5 and 35-39, 2.68bpw's 6-34) and n 24 (K 1.5: 1.68bpw's 6-34).
//!
//! The streamer's DIG-X lane and its decode GEMVs have texts for n 32 and 48 only (K 2 and 3); a code array of another
//! width there is refused by name (`kIndexOf`). On, the bank admits a layer at a width in `frac_n` and the t3 math
//! (src/routed_t3.zig) runs every call of such a layer on turboexl3: a wide prompt call's pairs, sorted by slot, as
//! `moeClamped` calls at top-1 over the bank's slot arrays, at most `prefill_rows` pairs each (two in flight, the stock
//! waves' lifecycle), a cold-row chunk the same way, every prompt call on turboexl3's prefill GEMM (a call of 16 pairs or
//! fewer padded into it with zero-score rows: `gemm_min_rows`), so a pair's bytes, and the text, do not depend on the
//! bank's rows; the banked decode as before. A layer at
//! n 32 or 48 runs exactly as it did (the same launches, the same words): a pack without a fractional layer is served
//! byte for byte as before. Needs the t3 route (`MLX_SERVE_DSV41_ROUTED` unset or turboexl3): with the streamer's own
//! route the switch is off and a fractional bank is refused at its claim, by name.
//!
//! `check`: each fractional call's output compared, on up to `check_rows` sampled rows, with each row's expert in f64
//! from the bank's own bytes (src/one_exact.zig over turboexl3's format), at ROUTED_ONE's bars (rms of the difference
//! within `bar_rms` of the reference's rms, the largest within `bar_ratio` of it). `[frac]` at each request's end: the
//! calls, rows and checks (`meters`).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

pub const Error = error{ FracSetting, FracNeedsT3 };

/// The fractional tile widths (halfwords a 256-weight tile) the turboexl3 lane serves: K 1.5 and K 2.5, exllamav3's
/// half-integer rates at turboderp's branches.
pub const frac_n = [_]u32{ 24, 40 };
/// The widths the streamer's DIG-X lane and decode GEMVs have texts for: K 2 and K 3.
pub const digx_n = [_]u32{ 32, 48 };

/// `check`'s bars (src/routed_one.zig's: turboexl3's for one EXL3 GEMM in two summation orders).
pub const bar_rms: f64 = 3.9e-3;
pub const bar_ratio: f64 = 8e-2;
/// `check`: rows compared a call.
pub const check_rows = 4;
/// A fractional wide call's pairs a `moeClamped` call: bounds a chunk's working set (its rows in bf16 and f32, the
/// turboexl3 chain's intermediates, its f32 output) at about half a GiB, two chunks in flight, about the stock lane's two
/// 7,168-row waves; not a speed setting.
pub const prefill_rows = 4096;
/// The fewest pairs a fractional prompt call takes to moeClamped: turboexl3 runs a call of `DECODE_ROWS_MAX` (16) rows or
/// fewer on its decode kernels and a larger one on its prefill GEMM, which agree within tolerance, not byte for byte; a
/// call's pair count follows the bank's residency, so a smaller call is padded to this many with zero-score copies of
/// its first pair (src/routed_t3.zig `fracGemm`, which checks this against turboexl3's constant at compile time).
pub const gemm_min_rows = 17;

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name. `t3_route`: the t3 math is on
/// (`MLX_SERVE_DSV41_ROUTED`): without it an explicit 1 or check is refused (`FracNeedsT3`), unset is off.
pub fn configure(value: ?[]const u8, t3_route: bool) Error!void {
    const m = try parse(value);
    stats = .{};
    if (!t3_route) {
        if (value != null and value.?.len > 0 and m != .off) return error.FracNeedsT3;
        mode = .off;
        return;
    }
    mode = m;
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.FracSetting;
}

pub fn on() bool {
    return mode != .off;
}

/// The fractional widths the bank admits now (`Implemented.frac_n`): `frac_n` on, none off.
pub fn admitted() []const u32 {
    return if (on()) &frac_n else &.{};
}

/// Whether a code array `n` halfwords a tile runs on the DIG-X lane (n 32 or 48).
pub fn isDigx(n: u64) bool {
    return n == 32 or n == 48;
}

/// Whether the t3 math runs a code array of `n` halfwords a tile on turboexl3 (a fractional width the switch admits).
pub fn served(n: u64) bool {
    if (!on()) return false;
    for (frac_n) |f| if (f == n) return true;
    return false;
}

/// A v2 manifest layer's K (an integer, or exllamav3's half-integer) as halfwords a tile: 16 K, whole, even, within
/// turboexl3's 16..128; null otherwise.
pub fn nOfK(k: f64) ?u32 {
    if (!(k >= 1.0 and k <= 8.0)) return null;
    const x = k * 16.0;
    const n: u32 = @intFromFloat(x);
    if (@as(f64, @floatFromInt(n)) != x or n % 2 != 0 or n < 16 or n > 128) return null;
    return n;
}

/// `check`: calls checked a layer a request (each fractional layer's first ones; the f64 expert is slow).
pub const check_calls = 2;
var checked: [64]u8 = @splat(0);
var checked_cold: u8 = 0;

/// Whether `check` compares this call (the layer's first `check_calls` of the request; cold-row chunks: the request's
/// first `check_calls`).
pub fn takeCheck(layer: u32, cold: bool) bool {
    if (mode != .check) return false;
    if (cold) {
        if (checked_cold >= check_calls) return false;
        checked_cold += 1;
        return true;
    }
    if (layer >= checked.len or checked[layer] >= check_calls) return false;
    checked[layer] += 1;
    return true;
}

/// A checked call's outcome into the request's stats.
pub fn noteCheck(layer: u32, rows: usize, fails: u32, rms: f64, ratio: f64) void {
    stats.checks += 1;
    stats.check_rows += rows;
    if (fails > 0) stats.check_fail += 1;
    if (!(rms <= stats.worst_rms)) stats.worst_rms = rms;
    if (!(ratio <= stats.worst_ratio)) {
        stats.worst_ratio = ratio;
        stats.worst_layer = layer;
    }
}

/// The fractional calls since the last request's end.
pub const Stats = struct {
    /// wide prompt calls, their rows (routed pairs) and their `moeClamped` chunks (`prefill_rows` pairs at most)
    prefill_calls: u64 = 0,
    prefill_rows: u64 = 0,
    prefill_chunks: u64 = 0,
    /// moeClamped calls of 16 pairs or fewer padded into the GEMM (`gemm_min_rows`), hot chunks and cold chunks
    padded_calls: u64 = 0,
    /// cold-row chunks (one banked call each) and their rows
    cold_calls: u64 = 0,
    cold_rows: u64 = 0,
    /// host time in the prefill calls (encoding, and waiting for the chunk two back)
    prefill_ns: u64 = 0,
    /// `check`
    checks: u64 = 0,
    check_rows: u64 = 0,
    check_fail: u64 = 0,
    worst_rms: f64 = 0,
    worst_ratio: f64 = 0,
    worst_layer: u32 = 0,
};
pub var stats: Stats = .{};

/// `[frac]` at a request's end ("" when off or idle), then cleared.
pub fn meters(buf: []u8) []const u8 {
    if (!on()) return buf[0..0];
    const s = stats;
    stats = .{};
    checked = @splat(0);
    checked_cold = 0;
    if (s.prefill_calls == 0 and s.cold_calls == 0 and s.checks == 0) return buf[0..0];
    var w = std.Io.Writer.fixed(buf);
    w.print("[frac] mode={t} prefill_calls={d} prefill_rows={d} prefill_host_ms={d:.1} cold_calls={d} cold_rows={d} prefill_chunks={d} padded_calls={d}", .{ mode, s.prefill_calls, s.prefill_rows, @as(f64, @floatFromInt(s.prefill_ns)) / 1e6, s.cold_calls, s.cold_rows, s.prefill_chunks, s.padded_calls }) catch {};
    if (mode == .check) w.print(" checks={d} rows={d} fail={d} worst_rms={e:.3} worst_ratio={e:.3} worst_layer={d} bar={e:.1}/{e:.1}", .{ s.checks, s.check_rows, s.check_fail, s.worst_rms, s.worst_ratio, s.worst_layer, bar_rms, bar_ratio }) catch {};
    w.writeAll("\n") catch {};
    return w.buffered();
}

/// The load line: each routed layer's width, the fractional ones named (`widths[L]` halfwords a tile, 0 unknown).
pub fn loadLine(buf: []u8, widths: []const u32) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.print("[turboexl3_ds41] MLX_SERVE_DSV41_FRAC={t}: ", .{mode}) catch {};
    var any = false;
    for (frac_n) |f| {
        var first = true;
        var l: usize = 0;
        while (l < widths.len) : (l += 1) {
            if (widths[l] != f) continue;
            var e = l;
            while (e + 1 < widths.len and widths[e + 1] == f) e += 1;
            if (first) {
                w.print("{s}n{d} (K {d}.5) at layers ", .{ if (any) "; " else "", f, f / 16 }) catch {};
            } else {
                w.writeAll(",") catch {};
            }
            if (e == l) {
                w.print("{d}", .{l}) catch {};
            } else {
                w.print("{d}-{d}", .{ l, e }) catch {};
            }
            first = false;
            any = true;
            l = e;
        }
    }
    if (!any) w.writeAll("no fractional layer (every layer on its lane as before)") catch {} else w.writeAll(" on turboexl3; n32 / n48 layers on the DIG-X lane") catch {};
    w.writeAll("\n") catch {};
    return w.buffered();
}

/// The cold rows' gate|up of a fractional layer runs at its down, whole (turboexl3's banked call reads the down's suh in
/// its gate|up finish): `gateUp` returns its rows and parks the call's gate and up arrays here, keyed by the rows'
/// array; `down` takes them back. A small ring: the hook runs a chunk's down right after its gate|up, or a wave's groups'
/// downs after their gate|ups (`ring` of them at most).
pub fn Parked(comptime Proj: type) type {
    return struct {
        const Self = @This();
        pub const ring = 16;
        keys: [ring]?*anyopaque = @splat(null),
        gate: [ring]Proj = undefined,
        up: [ring]Proj = undefined,
        next: usize = 0,

        pub fn put(self: *Self, key: ?*anyopaque, gate: Proj, up: Proj) void {
            const i = self.next % ring;
            self.keys[i] = key;
            self.gate[i] = gate;
            self.up[i] = up;
            self.next += 1;
        }

        pub fn take(self: *Self, key: ?*anyopaque) ?struct { gate: Proj, up: Proj } {
            for (&self.keys, 0..) |*k, i| if (k.* != null and k.* == key) {
                k.* = null;
                return .{ .gate = self.gate[i], .up = self.up[i] };
            };
            return null;
        }
    };
}

const testing = std.testing;

test "frac: the switch" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "2", "on", "Check", " 1" }) |v| try testing.expectError(error.FracSetting, parse(v));
    try configure(null, false);
    try testing.expect(!on());
    try testing.expectError(error.FracNeedsT3, configure("1", false));
    try testing.expectError(error.FracNeedsT3, configure("check", false));
    try configure("0", false);
    try configure(null, true);
    try testing.expect(on() and served(40) and served(24) and !served(32) and !served(48) and !served(56));
    try configure("0", true);
    try testing.expect(!on() and !served(40) and admitted().len == 0);
    try configure(null, true);
}

test "frac: K to halfwords a tile" {
    try testing.expectEqual(@as(?u32, 48), nOfK(3));
    try testing.expectEqual(@as(?u32, 32), nOfK(2));
    try testing.expectEqual(@as(?u32, 40), nOfK(2.5));
    try testing.expectEqual(@as(?u32, 24), nOfK(1.5));
    try testing.expectEqual(@as(?u32, 16), nOfK(1));
    try testing.expectEqual(@as(?u32, 128), nOfK(8));
    try testing.expectEqual(@as(?u32, 36), nOfK(2.25));
    for ([_]f64{ 0, 0.5, 8.5, 9, 2.3, 2.0625, -2, std.math.nan(f64), std.math.inf(f64) }) |k| try testing.expectEqual(@as(?u32, null), nOfK(k));
    try testing.expect(isDigx(32) and isDigx(48) and !isDigx(40) and !isDigx(24));
}

test "frac: the load line names the fractional runs" {
    try configure(null, true);
    var w: [40]u32 = @splat(32);
    for (0..6) |l| w[l] = 40;
    for (35..40) |l| w[l] = 40;
    var buf: [512]u8 = undefined;
    const line = loadLine(&buf, &w);
    try testing.expect(std.mem.indexOf(u8, line, "n40 (K 2.5) at layers 0-5,35-39 on turboexl3") != null);
    var z: [40]u32 = @splat(48);
    z[23] = 32;
    try testing.expect(std.mem.indexOf(u8, loadLine(&buf, &z), "no fractional layer") != null);
}

test "frac: the request meter" {
    try configure(null, true);
    stats = .{ .prefill_calls = 2, .prefill_rows = 9000, .prefill_chunks = 3, .prefill_ns = 1_500_000, .cold_calls = 1, .cold_rows = 7, .padded_calls = 2 };
    var buf: [512]u8 = undefined;
    const line = meters(&buf);
    try testing.expect(std.mem.startsWith(u8, line, "[frac] mode=on prefill_calls=2 prefill_rows=9000 prefill_host_ms=1.5 cold_calls=1 cold_rows=7 prefill_chunks=3 padded_calls=2\n"));
    try testing.expectEqual(@as(u64, 0), stats.prefill_calls);
    try testing.expectEqual(@as(usize, 0), meters(&buf).len);
}

test "frac: the parked gate|up ring" {
    const P = struct { id: u32 };
    var r: Parked(P) = .{};
    var a: u8 = 0;
    var b: u8 = 0;
    r.put(&a, .{ .id = 1 }, .{ .id = 2 });
    r.put(&b, .{ .id = 3 }, .{ .id = 4 });
    const got = r.take(&a).?;
    try testing.expectEqual(@as(u32, 1), got.gate.id);
    try testing.expectEqual(@as(u32, 2), got.up.id);
    try testing.expect(r.take(&a) == null);
    try testing.expectEqual(@as(u32, 4), r.take(&b).?.up.id);
}
