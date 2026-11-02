//! `MLX_SERVE_DSV41_WIDE_EARLY` (read at load; unset, empty or `1`: on; `0`: off; `check`: on with the old path served
//! and compared; anything else is refused by name): what a wide prompt call's MoE does not read, committed behind its
//! routes before the routing barrier waits on them.
//!
//! Off, the call's barrier evaluates its routes (the layer's attention, its FFN mix and the router) and waits; the next
//! wide layer's predicted routes (`MLX_SERVE_DSV41_STAGE_NEXT`) start behind the barrier once its ids are on the host,
//! and the layer's shared expert and attention state are evaluated only by the layer's last eval, after the call's
//! groups: the GPU idles through the barrier's host work and the call's first read wait, then runs the shared expert's
//! GEMMs (about 25 ms of a 16K-row layer) at the layer's end.
//!
//! On, the copied model's `moeExpand` (the default TAIL_FUSE path) commits three evaluations before the call, each its
//! own commit: the routes, cast to the int32 ids the streamer reads (its own cast, made here instead); the next layer's
//! predicted routes, when there are any (STAGE_NEXT's start then finds them under way); and the shared expert with the
//! layer's attention state. The barrier waits on the first only. The GPU runs the other two during the barrier's host
//! work and the call's first read wait, and the layer's last eval is left the tail kernel. The same kernels run over
//! the same inputs; only independent ones run earlier, so the same words. A prompt call's MoE stage then also holds the
//! shared expert's evaluation (every array of it is made when the host encodes it ahead of the call: 1.03 GB at 16K
//! rows, `bytes`), which the prompt bill counts (the copied model's `prefillBytes`, read from the setting before the load;
//! only for calls wider than a span, so never the decode phase's bill).
//!
//! Measured (hg1, w-pio-20261102-0114, on zzz8w against ctl25; a1 b1 b2 a2, the cold 16K card then the new-text slice 3):
//! cold 1414.0 / 1418.6 / 1425.1 / 1400.3 tok/s (+1.04%, pairs +0.33% / +1.77%), new text 1490.4 / 1498.1 / 1502.6 /
//! 1486.2 (+0.81%, pairs +0.52% / +1.10%), with 150 prompt rows against 153 (the bill's 1.03 GB) and 177 decode rows in
//! every arm; the check arm's 41 lines compared 3,205,017,600 words, 0 differing, max_ulps 0; the texts the same bytes.
//!
//! `check`: as on, and the shared expert also built the old way and served (evaluated after the call's groups, as off
//! does); the early one compared with it word for word, one `[wide-early] check layer= words= differ= max_ulps=
//! max_abs=` line a wide call.

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

pub const Error = error{WideEarlySetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.WideEarlySetting;
}

/// The bytes a wide call's early shared expert holds through its MoE stage, for the prompt bill (the copied model's
/// `prefillBytes`): every array of its evaluation, allocated when the host encodes it ahead of the call (MLX makes each
/// output then; each is freed only when its consumers have run): the gate/up linear's prepared rows (f16 [rows, d]), its
/// decoded weight (f16 [d, 2n]), its product (f16 [rows, 2n]) and finish (the activation dtype, [rows, 2n]), the activation
/// ([rows, n], and its one f32 weight a row), the down linear's prepared rows (f16 [rows, n]), weight (f16 [n, d]),
/// product (f16 [rows, d]) and finish, the shared rows the tail reads ([rows, d]). `d` the hidden width, `n` the shared
/// expert's (2-byte activations). 1.03 GB at 16K rows of DeepSeek-V4.1 (d 5120, n 2304).
pub fn bytes(rows: u64, d: u64, n: u64) u64 {
    return rows * (2 * d + 4 * n + 4 * n + 2 * n + 4 + 2 * n + 2 * d + 2 * d) + 2 * d * 2 * n + 2 * n * d;
}

/// One wide call's `check`: the shared expert's words compared, those that differ, the largest distance in units of
/// the last place (the words' integer distance) and the largest absolute difference.
pub fn report(layer: usize, words: u64, differ: u64, max_ulps: u64, max_abs: f32) void {
    std.debug.print("[wide-early] check layer={d} words={d} differ={d} max_ulps={d} max_abs={e}\n", .{ layer, words, differ, max_ulps, max_abs });
}

test "wide_early bytes: the shared expert's evaluation at 16K rows of DeepSeek-V4.1, about 1.03 GB" {
    const b = bytes(16384, 5120, 2304);
    try std.testing.expectEqual(@as(u64, 16384 * (6 * 5120 + 12 * 2304 + 4) + 6 * 5120 * 2304), b);
    try std.testing.expect(b > 1_020_000_000 and b < 1_035_000_000);
    try std.testing.expectEqual(@as(u64, 6 * 5120 * 2304), bytes(0, 5120, 2304));
}

test "wide_early setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.on, try parse("1"));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "00", "Check", "true", " 1" }) |v| try std.testing.expectError(error.WideEarlySetting, parse(v));
}
