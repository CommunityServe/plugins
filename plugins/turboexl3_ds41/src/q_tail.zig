//! `MLX_SERVE_DSV41_Q_TAIL` (read at load; unset, empty or `1`: on; `0`: off; `check`: off's heads served, the tail
//! checked against them; anything else is refused by name): a wide prompt layer's query heads are not copied whole to
//! rope their last channels.
//!
//! Off, the copied model's `qkvProj` writes the chunk's roped heads `q` [L, 64, 512] bf16 (1.07 GB at 16K rows) from
//! `wq_b`'s output (another 1.07 GB): the rope kernel copies the 448 unroped channels of every head as they are and ropes
//! the last 64. On, the wide layer keeps `wq_b`'s output as the unroped heads and ropes only the last 64 channels of each
//! head into `qt` [L, 64, 64] (kernels/rope_tail_only.metal: rope_tail.metal's arithmetic on the same pairs); the
//! gather's score kernel (kernels/attn_qk_t.metal) reads the unroped heads for its first seven 64-deep steps and `qt` for
//! the last. The scores' operands are the same bf16 words in the same steps on the same tensor op, so the same scores. A
//! span the gather declines (another shape, the f32 core) gets the whole roped heads as before. One fresh 0.94 GB buffer
//! and its copy pass fewer a layer. `check` serves off's heads and also ropes the tail apart and runs the two-part score
//! kernel over every span, each compared word for word with off's (the tail with the roped heads' last channels, the
//! scores with the gather's): a `[q-tail] check` line a wide layer, ~0.13 GB and a span's scores (~40 MB) more.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

const A = mlx.mlx_array;

pub const Mode = enum { off, on, check };
pub var mode: Mode = .on;

pub const Error = error{QTailSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.QTailSetting;
}

/// The kernels' text (the copied model builds them at load).
pub const rope_tail_only_source = @embedFile("kernels/rope_tail_only.metal");
pub const attn_qk_t_source = @embedFile("kernels/attn_qk_t.metal");

/// `check`'s count over one wide layer: the words compared and those that differ.
pub const Tally = struct { words: u64 = 0, differ: u64 = 0 };
pub var tail: Tally = .{};
pub var scores: Tally = .{};

/// `check`: `a` against `b` (one shape, 2- or 4-byte words), word for word on the device, into `t`.
pub fn compare(s: mlx.mlx_stream, a: A, b: A, t: *Tally) !void {
    const wt: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(a)) {
        2 => .uint16,
        4 => .uint32,
        else => return error.QTailCheckWords,
    };
    var tmp: [5]A = undefined;
    for (&tmp) |*x| x.* = mlx.mlx_array_new();
    defer for (tmp) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_view(&tmp[0], a, wt, s));
    try mlx.check(mlx.mlx_view(&tmp[1], b, wt, s));
    try mlx.check(mlx.mlx_equal(&tmp[2], tmp[0], tmp[1], s));
    try mlx.check(mlx.mlx_astype(&tmp[3], tmp[2], .uint32, s));
    try mlx.check(mlx.mlx_sum(&tmp[4], tmp[3], false, s));
    try mlx.check(mlx.mlx_array_eval(tmp[4]));
    const same = (mlx.mlx_array_data_uint32(tmp[4]) orelse return error.NoData)[0];
    const n: u64 = mlx.mlx_array_size(a);
    t.words += n;
    t.differ += n - same;
}

/// `check`: the layer's line; the counts start over.
pub fn report(layer: usize) void {
    std.debug.print("[q-tail] check layer={d} tail_words={d} tail_differ={d} score_words={d} score_differ={d}\n", .{ layer, tail.words, tail.differ, scores.words, scores.differ });
    tail = .{};
    scores = .{};
}

test "q_tail setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.on, try parse("1"));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "true", "Check", " 1" }) |v| try std.testing.expectError(error.QTailSetting, parse(v));
}
