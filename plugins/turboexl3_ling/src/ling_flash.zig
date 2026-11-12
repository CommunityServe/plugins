//! turboexl3_ling's MLA prompt attention in one causal flash launch (MLA_FLASH, S38): the copied mla.zig's prompt
//! attention (first in `spanAttn`, ahead of attn_spans.zig's spans) attends in turboexl3's one causal flash launch
//! (`root.ling.mlaFlashPrefill`: MLX's NAX attention kernel at qk 192, v 128, the cache's views read in place) instead
//! of MLX's composed SDPA, which MLX takes at a query head dim of 192. A reordering (f32 scores, running max and sum, one
//! output rounding): within tolerance of the composed path, never its bytes (src/ling_mla_spans_test.zig holds it to
//! the composed span attention; turboexl3's own test to an f64 reference). Decode and verify widths (fewer than
//! `min_rows` query rows) and an exact verify's rows keep MLX's kernels. A call turboexl3 refuses (its shape, dtype or
//! the NAX gate) logs once and runs MLX's SDPA.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
/// The run-path line (src/ling_ran.zig): MLA_FLASH marks where its launch ran.
const ran = @import("ling_ran.zig");

/// The fewest query rows a call takes the kernel at: past the decode and verify widths (16 rows or fewer).
pub const min_rows: c_int = 17;

var refused_said = false;

/// Whether a call of `seq_len` query rows in `mode` takes the kernel: causal (a prompt's), past the decode and verify
/// widths.
pub fn takes(mode: [*:0]const u8, seq_len: c_int) bool {
    return seq_len >= min_rows and std.mem.eql(u8, std.mem.span(mode), "causal");
}

/// `out` (the caller's handle) = turboexl3's causal flash attention of `q` over `k`, `v`; false where turboexl3 refuses
/// the call (the caller then runs MLX's SDPA).
pub fn attend(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !bool {
    const o = t3.ling.mlaFlashPrefill(s, q, k, v, scale) catch |e| switch (e) {
        error.BadExl3Shape, error.BadExl3Dtype, error.Exl3EngineNotImplemented => {
            if (!refused_said) {
                refused_said = true;
                std.log.info("[turboexl3_ling] MLA prompt attention: turboexl3's flash launch refused a call ({s}); MLX's SDPA runs it", .{@errorName(e)});
            }
            return false;
        },
        else => return e,
    };
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_array_set(out, o));
    ran.mark(.MLA_FLASH);
    return true;
}
