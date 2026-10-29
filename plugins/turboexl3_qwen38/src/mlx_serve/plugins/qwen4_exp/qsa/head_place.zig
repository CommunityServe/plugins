//! copied_from: src/plugins/qwen4_exp/qsa/head_place.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The MTP head's placed draft rows: their RoPE positions, block scores and key visibility.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const Trunk = @import("../trunk.zig").Trunk;

/// `[1]` int32 RoPE position of a placed draft row: `live_end + (slot - dead_end)`, with the
/// head's absolute origin folded into `shift` (`pos_base + slot - dead_end`).
pub fn headPlaceRopePos(s: mlx.mlx_stream, p: *const Trunk.HeadPlace, shift: c_int) !mlx.mlx_array {
    const shift_v = mlx.mlx_array_new_int(shift);
    defer _ = mlx.mlx_array_free(shift_v);
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_add(&pos, p.live_end, shift_v, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, pos, &[_]c_int{1}, 1, s));
    return out;
}

/// Block scores with every block that reaches a dead row (or lies past one) at -inf, so a
/// dead row's pooled key can never win or displace a top-k pick.
pub fn headPlaceScores(s: mlx.mlx_stream, scores: mlx.mlx_array, p: *const Trunk.HeadPlace, ratio: c_int, nb: c_int) !mlx.mlx_array {
    const ratio_v = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_v);
    var eligible_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eligible_n);
    try mlx.check(mlx.mlx_floor_divide(&eligible_n, p.live_end, ratio_v, s));
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_arange(&blk, 0, @floatFromInt(nb), 1, .int32, s));
    var eligible = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eligible);
    try mlx.check(mlx.mlx_less(&eligible, blk, eligible_n, s));
    const floor_v = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(floor_v);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&out, eligible, scores, floor_v, s));
    return out;
}

/// Key visibility `[1,1,1,kv]` for a placed draft row: every row but the dead ones, narrowed
/// by the QSA selection when there is one. Rows from the first block that reaches a dead row
/// on stay visible, since `headPlaceScores` keeps those blocks out of the selection.
pub fn headPlaceVisibility(s: mlx.mlx_stream, qsa_mask: mlx.mlx_array, p: *const Trunk.HeadPlace, ratio: c_int, kv: c_int) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_arange(&r, 0, @floatFromInt(kv), 1, .int32, s));
    var live = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(live);
    try mlx.check(mlx.mlx_less(&live, r, p.live_end, s));
    const dead_end_v = mlx.mlx_array_new_int(p.dead_end);
    defer _ = mlx.mlx_array_free(dead_end_v);
    var past = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(past);
    try mlx.check(mlx.mlx_greater_equal(&past, r, dead_end_v, s));
    var not_dead = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(not_dead);
    try mlx.check(mlx.mlx_logical_or(&not_dead, live, past, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    if (qsa_mask.ctx == null) {
        try mlx.check(mlx.mlx_array_set(&vis, not_dead));
    } else {
        const ratio_v = mlx.mlx_array_new_int(ratio);
        defer _ = mlx.mlx_array_free(ratio_v);
        var blocks = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blocks);
        try mlx.check(mlx.mlx_floor_divide(&blocks, p.live_end, ratio_v, s));
        var recent_from = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recent_from);
        try mlx.check(mlx.mlx_multiply(&recent_from, blocks, ratio_v, s));
        var recent = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recent);
        try mlx.check(mlx.mlx_greater_equal(&recent, r, recent_from, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_logical_or(&picked, qsa_mask, recent, s));
        try mlx.check(mlx.mlx_logical_and(&vis, picked, not_dead, s));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, vis, &[_]c_int{ 1, 1, 1, kv }, 4, s));
    return out;
}
