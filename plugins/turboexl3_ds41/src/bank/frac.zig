//! `TURBOEXL3_DS41_FRAC` (read at load; unset, empty or `1`: on; `0`: off, a fractional bank refused by name; anything
//! else is refused by name): routed layers at a half-integer EXL3 rate on turboexl3, the rest on the streamer's own
//! lanes as before.
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
//! (routed.zig) runs every call of such a layer on turboexl3: a wide prompt call's pairs, sorted by slot, as
//! `moeClamped` calls at top-1 over the bank's slot arrays, at most `prefill_rows` pairs each (two in flight, the stock
//! waves' lifecycle), a cold-row chunk the same way, every prompt call on turboexl3's prefill GEMM (a call of 16 pairs or
//! fewer padded into it with zero-score rows: `gemm_min_rows`), so a pair's bytes, and the text, do not depend on the
//! bank's rows; the banked decode as before. A layer at
//! n 32 or 48 runs exactly as it did (the same launches, the same words): a pack without a fractional layer is served
//! byte for byte as before.
//!
//! TURBOEXL3_DS41_PF_INT8 (src/pf_int8.zig; not exact) moves the prompt calls of the layers at n 32 (K 2) onto the same
//! turboexl3 path too (routed.zig, `int8_n`), where turboexl3's int8 prefill GEMMs serve them; their decode waves stay
//! banked, and n 48 stays on the DIG-X lane. Unset, nothing here changes.

const std = @import("std");

pub const Mode = enum { off, on };

/// The switch (`configure`, at load).
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    mode: Mode = .on,
};
pub const shared_state = @import("sdk").shared.Shared(State, "dsv41_frac", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

pub const Error = error{FracSetting};

/// The fractional tile widths (halfwords a 256-weight tile) the turboexl3 lane serves: K 1.5 and K 2.5, exllamav3's
/// half-integer rates at turboderp's branches.
pub const frac_n = [_]u32{ 24, 40 };
/// The widths the streamer's DIG-X lane and decode GEMVs have texts for: K 2 and K 3.
pub const digx_n = [_]u32{ 32, 48 };
/// The DIG-X width whose prompt calls TURBOEXL3_DS41_PF_INT8 moves onto turboexl3 (K 2).
pub const int8_n: u32 = 32;

/// A fractional wide call's pairs a `moeClamped` call: bounds a chunk's working set (its rows in bf16 and f32, the
/// turboexl3 chain's intermediates, its f32 output) at about half a GiB, two chunks in flight, about the stock lane's two
/// 7,168-row waves; not a speed setting.
pub const prefill_rows = 4096;
/// The fewest pairs a fractional prompt call takes to moeClamped: turboexl3 runs a call of `DECODE_ROWS_MAX` (16) rows or
/// fewer on its decode kernels and a larger one on its prefill GEMM, which agree within tolerance, not byte for byte; a
/// call's pair count follows the bank's residency, so a smaller call is padded to this many with zero-score copies of
/// its first pair (routed.zig `fracGemm`, which checks this against turboexl3's constant at compile time).
pub const gemm_min_rows = 17;

/// `value`: null, empty or `1` on; `0` off; anything else refused by name.
pub fn configure(value: ?[]const u8) Error!void {
    ps().mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.FracSetting;
}

pub fn on() bool {
    return ps().mode != .off;
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

/// The load line: each routed layer's width, the fractional ones named (`widths[L]` halfwords a tile, 0 unknown).
pub fn loadLine(buf: []u8, widths: []const u32) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.print("deepseek_v41: TURBOEXL3_DS41_FRAC={t}: ", .{ps().mode}) catch {};
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

/// The load line of TURBOEXL3_DS41_PF_INT8's move: the layers at `int8_n` (`widths[L]` halfwords a tile).
pub fn int8LoadLine(buf: []u8, widths: []const u32) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.print("deepseek_v41: TURBOEXL3_DS41_PF_INT8: the n{d} (K {d}) layers' prompt calls on turboexl3's int8 prefill too, layers ", .{ int8_n, int8_n / 16 }) catch {};
    var first = true;
    var l: usize = 0;
    while (l < widths.len) : (l += 1) {
        if (widths[l] != int8_n) continue;
        var e = l;
        while (e + 1 < widths.len and widths[e + 1] == int8_n) e += 1;
        if (!first) w.writeAll(",") catch {};
        if (e == l) w.print("{d}", .{l}) catch {} else w.print("{d}-{d}", .{ l, e }) catch {};
        first = false;
        l = e;
    }
    w.writeAll(" (their decode waves banked as before)\n") catch {};
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
    for ([_][]const u8{ "2", "on", "check", "Check", " 1" }) |v| try testing.expectError(error.FracSetting, parse(v));
    try configure(null);
    try testing.expect(on() and served(40) and served(24) and !served(32) and !served(48) and !served(56));
    try configure("0");
    try testing.expect(!on() and !served(40) and admitted().len == 0);
    try configure(null);
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
    try configure(null);
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

test "frac: TURBOEXL3_DS41_PF_INT8's load line names the K 2 layers' runs" {
    var w: [40]u32 = @splat(32);
    for (0..6) |l| w[l] = 40;
    for (35..40) |l| w[l] = 40;
    w[20] = 48;
    var buf: [512]u8 = undefined;
    const line = int8LoadLine(&buf, &w);
    try testing.expect(std.mem.indexOf(u8, line, "the n32 (K 2) layers' prompt calls on turboexl3's int8 prefill too, layers 6-19,21-34 (") != null);
    try testing.expect(std.mem.endsWith(u8, line, "\n"));
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
