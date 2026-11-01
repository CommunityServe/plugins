//! The host plan of `MLX_SERVE_DSV41_MARKOV` (src/markov.zig), MLX-free: the setting, the drafts the fused chain takes,
//! how its step kernel covers the vocabulary, when the confidence head's logit is the last kernel's (every product of its
//! dot exact), and `check`'s tally.

const std = @import("std");

/// `MLX_SERVE_DSV41_MARKOV`:
/// - `on` (unset, empty or `1`): the fused kernels, B + 1 launches a draft with the op chain's bytes: a launch a
///   position (its bias row, the add, the threadgroups' argmax pairs, the rows in zig-zag order) and one for the tokens
///   and the confidence head.
/// - `off` (`0`): the draft's Markov chain as the copied model's op chain, five op groups a position.
/// - `flat`: `on` with every position reading the rows in one order (no zig-zag).
/// - `check`: both every draft; the op chain's draft served, the two compared bit for bit (tokens, confidence logits,
///   biased rows) and the tally logged at each request's end.
pub const Mode = enum { off, on, flat, check };

/// The setting's mode, or `error.MarkovSetting` for a value it does not name (the load refuses it by name).
pub fn modeOf(value: ?[]const u8) error{MarkovSetting}!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "flat")) return .flat;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.MarkovSetting;
}

/// The step kernel: threadgroups of 256 threads, each of its 8 simdgroups summing a quad (4 rows) at a time; the last
/// kernel: a threadgroup of 512 threads a position (MLX's dot_product geometry).
pub const step_threads: u32 = 256;
pub const step_simdgroups: u32 = step_threads / 32;
pub const final_threads: u32 = 512;
/// Threadgroups of a step: each later step's threadgroups reduce this many pairs first. The fastest of 253, 505 and 1010
/// in window 17 (M5 Max, the draft's chain after a 512 MB read: 575.6 us a draft at the median, 587.4 at 505, 600.3 at
/// 253; the op chain 842.0).
pub const step_groups: u32 = 1010;
/// The most draft positions a chain takes (the copied draft's id arrays).
pub const max_block: usize = 16;
/// The largest confidence dot MLX runs in one dot_product threadgroup (512 threads x 32 elements).
pub const max_conf_len: usize = final_threads * 32;

/// The quad a simdgroup's turn reads at position `k`: `q` (the turns' order), or with zig-zag at an odd position the
/// quad as far from the end (the rows the position before read last, first).
pub fn quadAt(q: u32, quads: u32, k: u32, zigzag: bool) u32 {
    return if (zigzag and k % 2 == 1) quads - 1 - q else q;
}

/// A draft's dtypes as the plan tells them apart.
pub const Dt = enum { bf16, f16, f32, other };

pub const Shape = struct {
    /// the activation dtype (markov_head's), markov_embed's, the stages' collapse (`hout`)
    act: Dt,
    embed: Dt,
    hout: Dt,
    vocab: usize,
    rank: usize,
    dim: usize,
    block: usize,
};

/// Why a draft keeps the op chain.
pub const Refusal = enum {
    /// f32 rows and vectors: their products are not exact, so a fused sum could round where MLX's gemv does not
    act_dtype,
    embed_dtype,
    /// a vocabulary the quads do not tile
    vocab,
    /// a rank MLX's gemv sums with a remainder (a loop the fused sum does not have)
    rank,
    block,
};

fn narrow(d: Dt) bool {
    return d == .bf16 or d == .f16;
}

/// Null when the fused kernels take the draft.
pub fn refusal(s: Shape) ?Refusal {
    if (!narrow(s.act)) return .act_dtype;
    if (!narrow(s.embed)) return .embed_dtype;
    if (s.vocab == 0 or s.vocab % 4 != 0 or s.vocab / 4 > std.math.maxInt(u32)) return .vocab;
    if (s.rank == 0 or s.rank % 128 != 0 or s.rank > 1024) return .rank;
    if (s.block == 0 or s.block > max_block) return .block;
    return null;
}

/// Whether the last kernel gives the confidence logit (else MLX's own ops do, on the fused tokens): MLX's dot_product
/// in one threadgroup, and every product of the dot exact, so where the sum fuses a multiply into an add changes no bit:
/// the activations and markov_embed 16-bit floats (8 or 11 significant bits), the head's weights 16-bit floats' values
/// (`exact16`).
pub fn confFused(s: Shape, weights_exact: bool) bool {
    return narrow(s.hout) and narrow(s.embed) and s.dim + s.rank <= max_conf_len and weights_exact;
}

/// Whether every value is an f16's or a bf16's (11 or 8 significant bits).
pub fn exact16(ws: []const f32) bool {
    for (ws) |w| {
        const h: f32 = @floatCast(@as(f16, @floatCast(w)));
        const b: f32 = @bitCast(@as(u32, @as(u16, @truncate(@as(u32, @bitCast(w)) >> 16))) << 16);
        if (h != w and b != w) return false;
    }
    return true;
}

/// `check`'s tally since the last request's end: the drafts, those the fused kernels took, the compared ones, and what
/// differed (tokens, confidence logits by bits, biased rows by element bits).
pub const Tally = struct {
    drafts: u64 = 0,
    fused: u64 = 0,
    refused: u64 = 0,
    checked: u64 = 0,
    token_drafts: u64 = 0,
    conf_drafts: u64 = 0,
    row_elems: u64 = 0,
    elems: u64 = 0,

    pub fn clean(t: Tally) bool {
        return t.token_drafts == 0 and t.conf_drafts == 0 and t.row_elems == 0;
    }

    /// `[markov] ...`, one line.
    pub fn line(t: Tally, mode: Mode, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "[markov] mode={s} drafts={d} fused={d} refused={d} checked={d} token_differ={d} conf_differ={d} rows_differ={d}/{d}{s}\n", .{
            @tagName(mode), t.drafts, t.fused, t.refused, t.checked, t.token_drafts, t.conf_drafts, t.row_elems, t.elems,
            if (t.checked == 0) "" else if (t.clean()) " IDENTICAL" else " DIFFER",
        }) catch buf[0..0];
    }
};

// ── tests ──

const testing = std.testing;

test "markov: the fused chain is on unless the setting is 0; any value it does not name is refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try modeOf(v));
    try testing.expectEqual(Mode.off, try modeOf("0"));
    try testing.expectEqual(Mode.flat, try modeOf("flat"));
    try testing.expectEqual(Mode.check, try modeOf("check"));
    for ([_][]const u8{ "on", "2", "1 ", "Check", "off" }) |v| try testing.expectError(error.MarkovSetting, modeOf(v));
}

/// Every (threadgroup, simdgroup, turn) of a step as the kernel walks them: how often each quad is read.
fn coverage(quads: u32, groups: u32, k: u32, zigzag: bool, seen: []u8) void {
    @memset(seen, 0);
    const ns = groups * step_simdgroups;
    for (0..groups) |tg| for (0..step_simdgroups) |sg| {
        var q: u32 = @as(u32, @intCast(tg)) * step_simdgroups + @as(u32, @intCast(sg));
        while (q < quads) : (q += ns) seen[quadAt(q, quads, k, zigzag)] += 1;
    };
}

test "markov: a step's simdgroups read every quad of the vocabulary once, in either order, at any threadgroup count" {
    const quads: u32 = 129280 / 4;
    const seen = try testing.allocator.alloc(u8, quads);
    defer testing.allocator.free(seen);
    for ([_]u32{ 253, 505, step_groups, 1, 4040, 5000 }) |groups| for (0..5) |k| for ([_]bool{ false, true }) |zz| {
        coverage(quads, groups, @intCast(k), zz, seen);
        for (seen) |c| try testing.expectEqual(@as(u8, 1), c);
    };
    // zig-zag: an odd position starts at the far end, an even one at the start
    try testing.expectEqual(quads - 1, quadAt(0, quads, 1, true));
    try testing.expectEqual(@as(u32, 0), quadAt(0, quads, 2, true));
    try testing.expectEqual(@as(u32, 0), quadAt(0, quads, 1, false));
}

const served: Shape = .{ .act = .bf16, .embed = .bf16, .hout = .bf16, .vocab = 129280, .rank = 256, .dim = 5120, .block = 5 };

test "markov: the fused kernels take DeepSeek-V4.1's draft and refuse by name what their sums would not match" {
    try testing.expectEqual(@as(?Refusal, null), refusal(served));
    var s = served;
    s.act = .f16;
    try testing.expectEqual(@as(?Refusal, null), refusal(s));
    s = served;
    s.act = .f32;
    try testing.expectEqual(@as(?Refusal, .act_dtype), refusal(s));
    s = served;
    s.embed = .f32;
    try testing.expectEqual(@as(?Refusal, .embed_dtype), refusal(s));
    s = served;
    s.vocab = 129282;
    try testing.expectEqual(@as(?Refusal, .vocab), refusal(s));
    s = served;
    s.rank = 192;
    try testing.expectEqual(@as(?Refusal, .rank), refusal(s));
    s = served;
    s.block = max_block + 1;
    try testing.expectEqual(@as(?Refusal, .block), refusal(s));
}

test "markov: the confidence logit is the last kernel's only where every product of its dot is exact" {
    try testing.expect(confFused(served, true));
    try testing.expect(!confFused(served, false));
    var s = served;
    s.hout = .f32;
    try testing.expect(!confFused(s, true));
    s = served;
    s.dim = max_conf_len;
    try testing.expect(!confFused(s, true));
    // the pack's head (F16 weights cast to f32) is exact; a weight with more bits, or a NaN, is not
    const f16s = [_]f32{ 0.0, -0.0, 1.0, @floatCast(@as(f16, 0.1)), @floatCast(@as(f16, -3.75e-5)), 65504.0, std.math.inf(f32) };
    try testing.expect(exact16(&f16s));
    // a bf16's value past f16's range
    try testing.expect(exact16(&.{@as(f32, @bitCast(@as(u32, 0x47800000) | 0x10000))}));
    try testing.expect(!exact16(&.{0.1}));
    try testing.expect(!exact16(&.{ 1.0, std.math.nan(f32) }));
}

test "markov: check's line says what differed, and IDENTICAL only once a draft was compared" {
    var buf: [256]u8 = undefined;
    var t: Tally = .{ .drafts = 3, .fused = 3 };
    try testing.expectEqualStrings("[markov] mode=on drafts=3 fused=3 refused=0 checked=0 token_differ=0 conf_differ=0 rows_differ=0/0\n", t.line(.on, &buf));
    t.checked = 3;
    t.elems = 300;
    try testing.expect(std.mem.endsWith(u8, t.line(.check, &buf), " IDENTICAL\n"));
    t.row_elems = 2;
    try testing.expect(std.mem.endsWith(u8, t.line(.check, &buf), "rows_differ=2/300 DIFFER\n"));
}
