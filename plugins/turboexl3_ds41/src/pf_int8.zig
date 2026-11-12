//! `TURBOEXL3_DS41_PF_INT8` (a plugin option, read at the format's open; not exact; unset, empty or `0`: the exact
//! prefill): turboexl3's int8 prefill GEMMs. Each weight's trellis is decoded to int8, round((t - 510) / 4) of its MUL1
//! byte sum t, and rebuilt in the GEMM's epilogue at the centered 509.5 (turboexl3's `Config.pf_int8_c2` 1019); each
//! activation row goes in as int8; the products are int8 MMAs into int32 sums.
//! - 1: the routed experts turboexl3 serves at prompt rows on int8, a scale each activation 128-block (`Config.pf_int8`
//!   1): the layers at a half-integer rate (their prompt calls on turboexl3's prefill GEMMs), the DSpark stages' banks,
//!   and the K 2 layers' prompt calls, which move off the streamer's DIG-X lane onto turboexl3 under this option
//!   (bank/routed.zig `int8_k2`; their decode waves stay on turboexl3's banked decode). K 3 layers stay on DIG-X, exact.
//! - 3: 1, and the attention projections at prompt rows on int8 too, one scale a row (turboexl3's trunk int8 at
//!   `t3.deepseek_v41.trunkShapes(3)`: wq_a, wkv and the compressors', wq_b, wo_a, wo_b and the indexer's wq_b), with
//!   the inner products the joined wo_a run splits off (`Config.pf_trunk_int8_inner`, src/woa_join.zig).
//! - 4: 3 with every trunk linear (`trunkShapes(4)`: the shared expert's and DSpark's main_proj too).
//! `2` (one scale a row on the routed experts) is refused by name: that scheme holds a row of hidden in one threadgroup,
//! at most 32 blocks of 128, and DeepSeek-V4.1's hidden 5120 is 40. Decode and verify rows stay exact. turboexl3's
//! src/models/deepseek_v41/deepseek_int8_test.zig holds each path to the exact one at these shapes (n 40 and n 32 routed).
//! Quality is not measured on DeepSeek-V4.1: GLM-5.3-Flash's measurements of the same int8 math carry the evidence
//! (turboexl3_glm53f). MLX-free.

const std = @import("std");

/// `TURBOEXL3_DS41_PF_INT8`'s level: 1, 3 or 4
pub const PfInt8 = struct {
    level: u8,

    /// The level a value names: `1`, `3` or `4`; `2` error.Ds41PfInt8OneScaleARow; anything else error.Ds41PfInt8Switch.
    pub fn parse(s: []const u8) !PfInt8 {
        if (s.len == 1) switch (s[0]) {
            '1', '3', '4' => return .{ .level = s[0] - '0' },
            '2' => return error.Ds41PfInt8OneScaleARow,
            else => {},
        };
        return error.Ds41PfInt8Switch;
    }

    /// The option's level: null when unset, empty or `0`; a value `parse` refuses refuses the load by name.
    pub fn read(env: *const fn ([:0]const u8) ?[]const u8) !?PfInt8 {
        const v = env("TURBOEXL3_DS41_PF_INT8") orelse return null;
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return null;
        return try parse(v);
    }

    /// The level in the load line's words.
    pub fn words(p: PfInt8) []const u8 {
        return switch (p.level) {
            3 => "the routed experts turboexl3 serves at a scale each activation 128-block, and the attention projections at one scale a row",
            4 => "the routed experts turboexl3 serves at a scale each activation 128-block, and every trunk linear at one scale a row",
            else => "the routed experts turboexl3 serves at a scale each activation 128-block",
        };
    }
};

/// The load line: `{d}` the level, `{s}` its words, `{d}` the trunk shapes on int8 (0 at level 1)
pub const line = "[turboexl3_ds41] TURBOEXL3_DS41_PF_INT8={d}: the prefill's int8 GEMMs, {s} ({d} trunk shapes); the weights rebuilt at 509.5 (turboexl3's Config.pf_int8, pf_trunk_int8 and pf_trunk_int8_inner; not exact; decode and verify rows exact)\n";

test "turboexl3_ds41 pf int8: TURBOEXL3_DS41_PF_INT8 takes 1, 3 or 4 (unset, empty and 0: exact); 2 refuses by name (one scale a row: hidden 5120 is past 32 blocks), anything else by the switch's name" {
    const t = std.testing;
    for ([_][]const u8{ "1", "3", "4" }) |v| try t.expectEqual(PfInt8{ .level = v[0] - '0' }, try PfInt8.parse(v));
    try t.expectError(error.Ds41PfInt8OneScaleARow, PfInt8.parse("2"));
    for ([_][]const u8{ "", "0", "5", "12", "01", " 1", "-1", "on", "int8" }) |v| try t.expectError(error.Ds41PfInt8Switch, PfInt8.parse(v));
    const Env = struct {
        var v: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TURBOEXL3_DS41_PF_INT8")) v else null;
        }
    };
    try t.expectEqual(@as(?PfInt8, null), try PfInt8.read(&Env.get));
    for ([_][]const u8{ "", "0" }) |off| {
        Env.v = off;
        try t.expectEqual(@as(?PfInt8, null), try PfInt8.read(&Env.get));
    }
    Env.v = "4";
    try t.expectEqual(@as(?PfInt8, .{ .level = 4 }), try PfInt8.read(&Env.get));
    Env.v = "2";
    try t.expectError(error.Ds41PfInt8OneScaleARow, PfInt8.read(&Env.get));
    Env.v = "x";
    try t.expectError(error.Ds41PfInt8Switch, PfInt8.read(&Env.get));
}
