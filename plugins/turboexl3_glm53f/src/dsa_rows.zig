//! `TURBOEXL3_GLM53F_DSA_ROWS=1` (opt-in, exact): a decode or verify window's DSA indexer rows assembled by the
//! pooled-keys launch itself, through patches/turboexl3-model.patch. The stock forward
//! (table_forward `dsaAttn`) casts the f32 pool gate to the key's dtype (an astype launch), concatenates the key and
//! the gate into rows (a copy launch each) and hands the rows to glm5_next's pooled-keys kernel. Here
//! kernels/dsa_rows.metal reads the LayerNorm's key and the f32 gate directly and casts the gate as the astype does, so
//! it writes the stock kernel's bits: 3 launches fewer a DSA layer.
//!
//! GLM-5.3-Flash's dims only: 128-wide keys and gates, pools of 4, B 1, 1 to 16 rows (a decode, verify or draft
//! window; a prompt keeps the stock path), a bf16 or f16 key, an f32 gate and pool weights.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_ROWS=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// Windows whose rows the pooled-keys launch assembled, since the load.
pub var runs: u64 = 0;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_ROWS=1: a decode or verify window's DSA indexer rows assembled by the pooled-keys launch from the LayerNorm's key and the f32 gate (the gate's cast and the rows' concatenate folded in: 3 launches fewer a layer, the same bits)\n";

/// GLM-5.3-Flash's indexer: key and gate width, pool size.
pub const head_dim: c_int = 128;
pub const kpool: c_int = 4;
/// The rows a decode, verify or draft window has at most (glm5_next `HC_PRE_MAX_ROWS`).
pub const max_rows: c_int = 16;

const source = @embedFile("kernels/dsa_rows.metal");
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// Whether the launch takes these shapes and dtypes: `ikn` the key [1, S, 128], `gate` [1, S, 128], `ape` [4, 128]
/// (any shape of 512 values), `prev` [1, prev_len, 256] when prev_len > 0. MLX-free.
pub fn serves(ikn: []const c_int, gate: []const c_int, ape_size: usize, prev: []const c_int, prev_len: c_int, kdt: mlx.mlx_dtype, gdt: mlx.mlx_dtype, adt: mlx.mlx_dtype, pdt: mlx.mlx_dtype) bool {
    if (ikn.len != 3 or ikn[0] != 1 or ikn[1] < 1 or ikn[1] > max_rows or ikn[2] != head_dim) return false;
    if (!std.mem.eql(c_int, ikn, gate)) return false;
    if (ape_size != @as(usize, @intCast(kpool * head_dim))) return false;
    if ((kdt != .bfloat16 and kdt != .float16) or gdt != .float32 or adt != .float32) return false;
    if (prev_len < 0 or prev_len > kpool - 1) return false;
    if (prev_len > 0 and (prev.len != 3 or prev[0] != 1 or prev[1] != prev_len or prev[2] != 2 * head_dim or pdt != kdt)) return false;
    return true;
}

/// The window's indexer rows [1, S, 256] as glm5_next `withPooledKeys` writes them from the concatenated rows, built
/// from their parts: `ikn` the LayerNorm's key, `gate` the f32 projection, `ape` the pools' f32 position weights, the
/// rows at positions `pos0..`, `prev` the `prev_len` cached rows before `pos0` (any array of the key's dtype when 0: not
/// read). A new array; null with the switch off, off the GPU, or for shapes `serves` declines.
pub fn pooled(s: mlx.mlx_stream, prev: Array, prev_len: c_int, ikn: Array, gate: Array, ape: Array, pos0: c_int) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const ksh = mlx.getShape(ikn);
    const kdt = mlx.mlx_array_dtype(ikn);
    if (!serves(ksh, mlx.getShape(gate), mlx.mlx_array_size(ape), mlx.getShape(prev), prev_len, kdt, mlx.mlx_array_dtype(gate), mlx.mlx_array_dtype(ape), mlx.mlx_array_dtype(prev))) return null;
    const k = try kda_prompt.kernelOf(&kernel, "turboexl3_glm53f_dsa_rows", &.{ "prev", "ikn", "gate", "ape", "dims" }, &.{"out"}, source, "");
    const dims = mlx.mlx_array_new_data(&[_]i32{ pos0, prev_len }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, ksh[1], 2 * head_dim }, 3, kdt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, head_dim, ksh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, head_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", kdt));
    inline for (.{ .{ "KPOOL", kpool }, .{ "DH", head_dim } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    var o: [1]Array = undefined;
    try kda_prompt.applyKernel(s, k, cfg, &.{ prev, ikn, gate, ape, dims }, &o);
    runs += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_ROWS engaged: a {d}-row window's indexer rows assembled by the pooled-keys launch ({d} cached rows before it)\n", .{ ksh[1], prev_len });
    }
    return o[0];
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_DSA_ROWS takes GLM-5.3-Flash's window rows (1 to 16 rows of 128-wide bf16 or f16 keys, f32 gates and pool weights, up to 3 cached rows of the key's dtype) and declines a prompt's rows, other widths, dtypes and batches" {
    const t = std.testing;
    const bf = mlx.mlx_dtype.bfloat16;
    const f32t = mlx.mlx_dtype.float32;
    for (1..17) |r| {
        const s: c_int = @intCast(r);
        for (0..4) |p| {
            const pl: c_int = @intCast(p);
            try t.expect(serves(&.{ 1, s, 128 }, &.{ 1, s, 128 }, 512, &.{ 1, pl, 256 }, pl, bf, f32t, f32t, bf));
        }
    }
    // f16 keys and their cached rows
    try t.expect(serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 256 }, 3, .float16, f32t, f32t, .float16));
    // no cached rows: `prev` is not read (any array)
    try t.expect(serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 128 }, 0, bf, f32t, f32t, bf));
    // a prompt chunk, a batch, other widths, a gate of another shape or dtype, f32 keys, other pool weights
    try t.expect(!serves(&.{ 1, 17, 128 }, &.{ 1, 17, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 4096, 128 }, &.{ 1, 4096, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 2, 3, 128 }, &.{ 2, 3, 128 }, 512, &.{ 2, 3, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 64 }, &.{ 1, 3, 64 }, 256, &.{ 1, 3, 128 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 2, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, bf, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 256 }, 3, f32t, f32t, f32t, f32t));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 1024, &.{ 1, 3, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, f32t, bf, bf));
    // cached rows that are not the stock tail: more than 3, another count, width or dtype
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 4, 256 }, 4, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 2, 256 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 128 }, 3, bf, f32t, f32t, bf));
    try t.expect(!serves(&.{ 1, 3, 128 }, &.{ 1, 3, 128 }, 512, &.{ 1, 3, 256 }, 3, bf, f32t, f32t, .float16));
}
