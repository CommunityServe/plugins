//! The turboexl3 side of `MLX_SERVE_DSV41_PG2` (src/pg2.zig): turboexl3's `pgemm` switch set at load, its tabled tiles'
//! kernels compiled at load (on or check), and `check`'s compare (`compare`: pg2's y beside the chain's, word for word on
//! the device, one host read a call, into pg2's counts), which turboexl3 calls at every call pg2 takes under check.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const pg2 = @import("pg2.zig");

const A = mlx.mlx_array;

/// At load, before the model's first prompt: turboexl3's switch at pg2's setting, check's compare hooked
pub fn configure() void {
    t3.pgemm.mode.get().* = switch (pg2.mode) {
        .off => .off,
        .on => .on,
        .check => .check,
    };
    t3.pgemm.check_hook.get().* = if (pg2.mode == .check) &compare else null;
}

/// After the load (on or check): every tabled tile's kernel compiled for the activations' dtype, and the compressors'
/// 5120 x 512 for f32 (the pooling layers run their wkv and wgate on f32 rows, so their y is f32), so a prompt's first
/// call pays no JIT
pub fn warm(s: mlx.mlx_stream, act: mlx.mlx_dtype) !void {
    if (pg2.mode == .off) return;
    try t3.pgemm.warm(s, act, null);
    if (act != .float32) try t3.pgemm.warm(s, .float32, .{ 5120, 512 });
}

/// The shapes turboexl3's table holds (the load line)
pub fn shapes() usize {
    return t3.pgemm.table.len;
}

/// The launches since load and their output elements (`[pg2]`)
pub fn counts() [2]u64 {
    return .{ t3.pgemm.calls.get().*.load(.monotonic), t3.pgemm.elems.get().*.load(.monotonic) };
}

/// check: pg2's y `got` beside the chain's `want` (one shape and dtype), compared word for word; a failed compare counts
/// every word as differing
fn compare(s: mlx.mlx_stream, want: A, got: A) void {
    const n: u64 = mlx.mlx_array_size(want);
    const same = sameWords(s, want, got) catch 0;
    pg2.tally(n, same);
}

fn sameWords(s: mlx.mlx_stream, want: A, got: A) !u64 {
    if (mlx.mlx_array_size(want) != mlx.mlx_array_size(got) or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) return 0;
    const word: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(want)) {
        2 => .uint16,
        4 => .uint32,
        else => return 0,
    };
    var tmp: [5]A = undefined;
    for (&tmp) |*a| a.* = mlx.mlx_array_new();
    defer for (tmp) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try mlx.check(mlx.mlx_view(&tmp[0], want, word, s));
    try mlx.check(mlx.mlx_view(&tmp[1], got, word, s));
    try mlx.check(mlx.mlx_equal(&tmp[2], tmp[0], tmp[1], s));
    try mlx.check(mlx.mlx_astype(&tmp[3], tmp[2], .uint32, s));
    try mlx.check(mlx.mlx_sum(&tmp[4], tmp[3], false, s));
    try mlx.check(mlx.mlx_array_eval(tmp[4]));
    return (mlx.mlx_array_data_uint32(tmp[4]) orelse return error.NoData)[0];
}
