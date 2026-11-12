//! turboexl3_ling's fused latent MLA decode, for the MTP head on its latent cache (src/ling_mtp_latent.zig). The head's
//! decode or verify call of at most `absorbed_max_rows` rows attends in three of turboexl3's launches
//! (`turboexl3.ling.mlaLatentDecode`: the query absorbed into the latent, the attention's partials over the latent
//! cache, the merge and the value half) in place of the composed f32 chain in the copied mla.zig's `latentAttn`
//! (src/ling_mla_latent.zig `decode`); a prompt call keeps the composed expansion, and a call the kernel does not take
//! (a batch, a wider verify) keeps the composed chain. Within tolerance of the composed chain, not byte for byte
//! (src/ling_mla_latent_test.zig holds it to `decode`). A row's bits are its own (the key groups and sub-chunks count
//! from position 0), so an exact verify's rows take the one call. The head's load line names what turboexl3's plan
//! makes of its calls (`declines`).
//! The load configures turboexl3 (src/ling_math.zig `configure`) to run the attention launch on the matrix units where
//! it takes the call (`Config.mla_latent_nax`, MLA_LATENT_NAX, S47), else staged in f32 (`Config.mla_latent_stage`,
//! MLA_LATENT_STAGE, S40). A head call whose attention ran on the matrix units marks MLA_LATENT_NAX's run-path bit
//! (`markNax`), from turboexl3's record of the launch the call ran (`ling.mla_latent.lastAttn`).
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ran = @import("ling_ran.zig");

/// MLA_LATENT_NAX's run-path mark: after a head call whose attention ran on the matrix units.
fn markNax() void {
    if (t3.ling.mla_latent.naxOf(t3.ling.mla_latent.lastAttn()) != null) ran.mark(.MLA_LATENT_NAX);
}

/// What `attend` makes of a decode call of one row at `heads` heads, with kv_b's stored rows of shape `rows`
/// `[H (nope + v), kv_lora]` (src/ling_mla_latent.zig `kvRows`) in `dw`, and the call's query and cache in `act`. Null
/// where turboexl3's launches take it, else the reason they do not. turboexl3's own plan decides
/// (`ling.mla_latent.planOf`), so a load line built on this says what the calls run.
pub fn declines(heads: c_int, rows: [2]c_int, act: mlx.mlx_dtype, dw: mlx.mlx_dtype) ?[]const u8 {
    const ml = t3.ling.mla_latent;
    const dn: c_int = @intCast(ml.dn);
    const dr: c_int = @intCast(ml.dr);
    const dl: c_int = @intCast(ml.dl);
    _ = ml.planOf(&[_]c_int{ 1, heads, 1, dn }, &[_]c_int{ 1, heads, 1, dr }, &[_]c_int{ 1, 1, 1, dl }, &[_]c_int{ 1, 1, 1, dr }, &rows, .{ act, act }, .{ act, act }, dw) catch |e| return @errorName(e);
    return null;
}

/// `out` (the caller's handle) = the fused latent attention of the head call's rows over the head's latent cache view
/// (`lat` `[B, 1, T, kv_lora]`, `rot` `[B, 1, T, rope]`) with kv_b's stored rows `w` `[H (nope + v), kv_lora]` (not the
/// loader's transposed view: src/ling_mla_latent.zig `kvRows`); false where the kernel does not take the call (the
/// caller then runs the composed chain).
pub fn attend(out: *mlx.mlx_array, s: mlx.mlx_stream, q_nope: mlx.mlx_array, q_rot: mlx.mlx_array, lat: mlx.mlx_array, rot: mlx.mlx_array, w: mlx.mlx_array, scale: f32) !bool {
    if (!t3.ling.mlaLatentTakes(q_nope, q_rot, lat, rot, w)) return false;
    const r = try t3.ling.mlaLatentDecode(s, q_nope, q_rot, lat, rot, w, scale);
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_array_set(out, r));
    markNax();
    return true;
}
