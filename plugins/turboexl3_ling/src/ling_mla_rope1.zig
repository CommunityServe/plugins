//! turboexl3_ling's rope-once MLA cache (MLA_ROPE_ONCE, S43; the same bytes), the layout every trunk MLA layer caches.
//! Upstream's mla.zig caches the expanded key, each head's 128 nope words ++ the token's 64 roped words for every one of
//! the 64 heads, beside the values: 64 x 320 words a token and layer. The copied mla.zig's `rope1Attn` caches kv_b's
//! rows as kv_b writes them, each head's nope keys then its values (`[B, 1, T, H (nope + v)]`, the V slot), and the
//! roped key once (`[B, 1, T, rope]`, the K slot): 16,448 words a token and layer, not 20,480 (src/bill.zig
//! `ropeOnceLayer`: 2,268 MiB less over the 9 MLA layers at 32,768 tokens). The attention reads the cache in place
//! through `views` (each head's nope keys and values strided views of the rows, the roped keys as cached):
//! - a decode row and an exact verify's rows (1 to 16): verify_rows.zig `attendRope1`, the verify kernels' rope-once
//!   texts, each row its one-row MLX SDPA call's words over the expanded key (what decode and the exact verify give on
//!   the expanded cache);
//! - a prompt call the flash launch takes (src/ling_flash.zig `takes`): turboexl3's `ling.mlaFlashPrefillRope1` (`flash`;
//!   src/ling_math.zig `configure` sets `Config.mla_rope_once`), the flash launch's words over the expanded key;
//! - any other call, or one those refuse: `expand` builds the expanded key from the views (the same words) and mla.zig's
//!   `verifyAttn` or `spanAttn` runs it. That rebuilt key is the fallback's spike (the values stay views: MLX reads them
//!   in place): 24 KiB a cached token, 768 MiB at 32,768 (src/bill.zig `ropeOnceSpike`). The rope-once flash launch
//!   takes the wide calls, so only a call of 16 rows or fewer falls back, inside the prompt pass's idle working set, and
//!   the prompt pass bills nothing for it (src/bill.zig, the test "the rope-once fallback's spike: ...").
//! The layout (plugin.zig `layout`), the bill (plugin.zig `kvPerToken`) and the SSD root (src/ling_switches.zig
//! `layoutNamespace`) are this cache's, so a prefix stored under another layout never restores into it. The cache's
//! first append marks MLA_ROPE_ONCE's run-path bit (mla.zig); the first call of each other kind logs a line (`engaged`).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

/// The cache's views for the attention: `nope` `[B, H, T, nope]` and `v` `[B, H, T, v]`, strided views of kv_b's cached
/// rows (head h's nope keys at h (nope + v) words, its values after them; a token's rows H (nope + v) words apart), and
/// `rope`, the cached roped keys `[B, 1, T, rope]` as they are (the cache's own view: `deinit` leaves it).
pub const Views = struct {
    nope: mlx.mlx_array,
    v: mlx.mlx_array,
    rope: mlx.mlx_array,

    pub fn deinit(self: *Views) void {
        _ = mlx.mlx_array_free(self.nope);
        _ = mlx.mlx_array_free(self.v);
    }
};

/// `Views` of the cache's `rows` `[B, 1, T, heads (nope_d + v_d)]` and roped keys `rope` `[B, 1, T, *]`: no copy (a
/// reshape of the rows' view, a transpose and two slices), each last axis contiguous.
pub fn views(rows: mlx.mlx_array, rope: mlx.mlx_array, heads: c_int, nope_d: c_int, v_d: c_int, s: mlx.mlx_stream) !Views {
    const rs = mlx.getShape(rows);
    if (rs.len != 4 or rs[1] != 1 or rs[3] != heads * (nope_d + v_d)) return error.LingRopeOnceBadRows;
    const b = rs[0];
    const t = rs[2];
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, rows, &[_]c_int{ b, t, heads, nope_d + v_d }, 4, s));
    var rt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rt);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    try mlx.check(mlx.mlx_transpose_axes(&rt, r4, &perm, 4, s));
    const st4 = [_]c_int{ 1, 1, 1, 1 };
    var nv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nv);
    try mlx.check(mlx.mlx_slice(&nv, rt, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, heads, t, nope_d }, 4, &st4, 4, s));
    var vv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(vv);
    try mlx.check(mlx.mlx_slice(&vv, rt, &[_]c_int{ 0, 0, 0, nope_d }, 4, &[_]c_int{ b, heads, t, nope_d + v_d }, 4, &st4, 4, s));
    return .{ .nope = nv, .v = vv, .rope = rope };
}

/// The expanded key of the views, `[B, heads, T, nope + rope]`: each head's nope keys ++ the token's roped key, the words
/// the expanded cache holds (upstream mla.zig's broadcast and concatenation). The caller frees it.
pub fn expand(vw: Views, heads: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const ns = mlx.getShape(vw.nope);
    const rs = mlx.getShape(vw.rope);
    var rb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rb);
    try mlx.check(mlx.mlx_broadcast_to(&rb, vw.rope, &[_]c_int{ ns[0], heads, ns[2], rs[3] }, 4, s));
    const parts = [_]mlx.mlx_array{ vw.nope, rb };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var k = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_concatenate_axis(&k, vec, -1, s));
    return k;
}

var refused_said = false;

/// `out` (the caller's handle) = turboexl3's rope-once causal flash launch of `q` over the views (a prompt call the
/// flash launch takes, the caller's check); false where turboexl3 refuses the call (its shape, dtype,
/// `Config.mla_rope_once` or the NAX gate), so the caller expands the key.
pub fn flash(out: *mlx.mlx_array, q: mlx.mlx_array, vw: Views, scale: f32, s: mlx.mlx_stream) !bool {
    const o = t3.ling.mlaFlashPrefillRope1(s, q, vw.nope, vw.rope, vw.v, scale) catch |e| switch (e) {
        error.BadExl3Shape, error.BadExl3Dtype, error.Exl3EngineNotImplemented => {
            if (!refused_said) {
                refused_said = true;
                std.log.info("[turboexl3_ling] MLA_ROPE_ONCE: turboexl3's rope-once flash launch refused a prompt call ({s}); the key is expanded for the served attention", .{@errorName(e)});
            }
            return false;
        },
        else => return e,
    };
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_array_set(out, o));
    return true;
}

/// Where a call ran: the verify kernels' rope-once texts, the rope-once flash launch, the expanded key on mla.zig's
/// `verifyAttn` or `spanAttn`
pub const Kind = enum { rows, flash, expand };
var said: [3]bool = @splat(false);

/// One line at the first call of each `Kind` (`seq_len` query rows over `total` cached keys)
pub fn engaged(kind: Kind, seq_len: c_int, total: c_int) void {
    const i = @intFromEnum(kind);
    if (said[i]) return;
    said[i] = true;
    switch (kind) {
        .rows => std.log.info("[turboexl3_ling] MLA_ROPE_ONCE: {d} decode or verify rows over {d} keys on the verify kernels' rope-once texts", .{ seq_len, total }),
        .flash => std.log.info("[turboexl3_ling] MLA_ROPE_ONCE: a prompt call of {d} rows over {d} keys on turboexl3's rope-once flash launch", .{ seq_len, total }),
        .expand => std.log.info("[turboexl3_ling] MLA_ROPE_ONCE: a call of {d} rows over {d} keys on the expanded key (the served attention)", .{ seq_len, total }),
    }
}
