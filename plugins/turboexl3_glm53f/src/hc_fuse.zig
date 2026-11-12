//! `TURBOEXL3_GLM53F_HC_FUSE=1` (opt-in, exact): a prompt sublayer's hyper-connection expand and the next sublayer's
//! collapse as one text (src/kernels/hc_fuse.metal), through patches/turboexl3-model.patch in the copy's
//! table_forward.zig (`sublayer`, `mlpSublayerSplit`): the new stream, the next sublayer's input and its post and comb
//! gates in one dispatch over each 8 rows, where the copy runs three (its expand, mixes and sinkhorn_collapse texts).
//!
//! At a 16K prompt each sublayer boundary (two a layer) moves the [16K, 4, 4096] bf16 stream, 0.54 GB, five times:
//! the expand reads it and writes the new one, the mixes and the collapse each read the new one. The fused text writes
//! it once and reads it back from this threadgroup's own 8 rows while they are in the cache.
//!
//! Each phase is the copy's own statements in their order and types (expand.metal, mixes.metal with its 8-row blocks
//! and its 8 simdgroups splitting K, gates_header.metal's `msv_hc_gates` and collapse.metal's sum), so its outputs are
//! the composed kernels' bits; src/hc_gpu_test.zig holds them to those bit for bit on the device. It serves where the
//! copy's mixes is its text: the copy runs MLX's rms_norm and matmul under `min_rows` rows (glm5_next.zig's
//! `MIXES_KERNEL_MIN_ROWS`), whose sums this text does not reproduce, so a short span (a split's 31-row tail) keeps the
//! copy's three kernels.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kernel_text = @import("mlx_serve/mlx/kernels.zig");
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_HC_FUSE=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_HC_FUSE=1: a prompt sublayer's hyper-connection expand and the next sublayer's mixes, Sinkhorn gates and collapse as one text over each 8 rows (the copy's expand, mixes and sinkhorn_collapse statements, in their order and types), the new stream written once and read back from the cache\n";

const source = @embedFile("kernels/hc_fuse.metal");
/// The text's header: Metal's simdgroup matrices and the copy's gates (`msv_hc_gates`, its collapse text's header).
const header = "#include <metal_simdgroup_matrix>\n" ++ kernel_text.source("plugins/glm5_next/kernels/gates_header.metal");
/// The rows at which the copy's mixes is its text (glm5_next.zig's `MIXES_KERNEL_MIN_ROWS`); below them it is MLX's ops.
pub const min_rows: c_int = 512;
/// The mixes text's simdgroups a threadgroup (glm5_next.zig's `MIXES_SG`), each a K slice; the fused text gives each
/// row of its 8-row block one of them for its gates.
pub const simdgroups: c_int = 8;

/// Whether the text takes a stream of `sh` (`dt`) at `rows` rows: one [B, S, HC, D] stream of a float dtype with D a
/// whole number of 8-column steps in each simdgroup's K slice, at `min_rows` rows or more.
pub fn serves(sh: []const c_int, dt: mlx.mlx_dtype) bool {
    if (sh.len != 4 or sh[0] < 1 or sh[1] < 1 or sh[2] != glm5.HC or sh[3] < 8) return false;
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return false;
    if (@rem(glm5.HC * sh[3], simdgroups * 8) != 0) return false;
    return sh[0] * sh[1] >= min_rows;
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
/// Launches since the load (src/hc_gpu_test.zig reads that the text ran).
pub var runs: u64 = 0;

/// The new stream and the next sublayer's collapse (its input x, post and comb gates), each a new array.
pub const Out = struct { stream: Array, next: glm5.Collapsed };

/// `glm5.expand(x, stream, c)` and then `glm5.collapse` of the new stream under the next sublayer's `fn_rows`, `scale`
/// and `base`, as one text. Null with the switch off or where it does not serve (the caller runs the two).
pub fn expandCollapse(s: mlx.mlx_stream, x: Array, stream: Array, c: *const glm5.Collapsed, fn_rows: Array, scale: Array, base: Array, iters: u32, hc_eps: f32, norm_eps: f32) !?Out {
    if (!on) return null;
    const o = (try text(s, x, stream, c, fn_rows, scale, base, iters, hc_eps, norm_eps)) orelse return null;
    runs += 1;
    if (!engaged) {
        engaged = true;
        const sh = mlx.getShape(stream);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_HC_FUSE engaged: {d} rows of {d} streams of {d} in one text a sublayer boundary\n", .{ sh[0] * sh[1], sh[2], sh[3] });
    }
    return o;
}

/// The text's launch, the switch aside (src/hc_gpu_test.zig calls it with the switch on): as `glm5.expand` casts the
/// sublayer output to the stream's dtype, and as `glm5.collapse` passes its gates' epsilon to the text (in units of
/// 1e-9) and the norm's epsilon as an f32 [1].
pub fn text(s: mlx.mlx_stream, x: Array, stream: Array, c: *const glm5.Collapsed, fn_rows: Array, scale: Array, base: Array, iters: u32, hc_eps: f32, norm_eps: f32) !?Out {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(stream);
    const dt = mlx.mlx_array_dtype(stream);
    if (!serves(sh, dt)) return null;
    const rows = sh[0] * sh[1];
    const mix: c_int = (2 + glm5.HC) * glm5.HC;
    // the tables and gates as the copy's texts read them: f32, the sizes their indexing assumes
    for ([_]Array{ fn_rows, scale, base, c.post, c.comb }) |a| if (mlx.mlx_array_dtype(a) != .float32) return null;
    if (mlx.mlx_array_size(fn_rows) != @as(usize, @intCast(mix * glm5.HC * sh[3])) or mlx.mlx_array_size(scale) != 3 or mlx.mlx_array_size(base) != @as(usize, @intCast(mix))) return null;
    if (mlx.mlx_array_size(c.post) != @as(usize, @intCast(rows * glm5.HC)) or mlx.mlx_array_size(c.comb) != @as(usize, @intCast(rows * glm5.HC * glm5.HC))) return null;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(rows * sh[3]))) return null;
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, dt, s));
    const params = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(params);
    const ev = [1]f32{norm_eps};
    const eps_a = mlx.mlx_array_new_data(&ev, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps_a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], glm5.HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], glm5.HC, glm5.HC }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(rows + 7, 8) * 32 * simdgroups, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * simdgroups, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "HC", glm5.HC }, .{ "D", sh[3] }, .{ "SG", simdgroups }, .{ "ITERS", @as(c_int, @intCast(iters)) }, .{ "EPS_INT", @as(c_int, @intFromFloat(@round(hc_eps / 1e-9))) } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ xc, stream, c.post, c.comb, fn_rows, params, eps_a, scale, base };
    const in_names = [_][*:0]const u8{ "x", "res", "post", "comb", "fn", "params", "eps", "scale", "base" };
    const out_names = [_][*:0]const u8{ "out", "collapsed", "post_out", "comb_out" };
    const kern = kernel orelse blk: {
        const iv = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("turboexl3_glm53f_hc_fuse", iv, ov, source, header, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
        break :blk k;
    };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 4) return error.MetalKernelBadOutputCount;
    var got: [4]Array = @splat(.{ .ctx = null });
    errdefer for (got) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&got, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    }
    return .{ .stream = got[0], .next = .{ .x = got[1], .post = got[2], .comb = got[3] } };
}

// ── tests (MLX-free) ──

const testing = std.testing;

test "turboexl3_glm53f stream hc fuse: the text takes a 16K prompt's 4 streams of 4096 and a split's main span, and declines below the copy's mixes-text rows, another stream count or a dtype it is not written for" {
    // a 16K prompt (16,387 rows), a split's main span, the copy's first mixes-text row count
    for ([_]c_int{ 16387, 16356, 995, 512 }) |rows| try testing.expect(serves(&.{ 1, rows, 4, 4096 }, .bfloat16));
    try testing.expect(serves(&.{ 2, 256, 4, 4096 }, .float16));
    // a split's 31-row tail and the warm-up's rows run the copy's ops for the mixes; one row less than the text's
    for ([_]c_int{ 31, 8, 511 }) |rows| try testing.expect(!serves(&.{ 1, rows, 4, 4096 }, .bfloat16));
    // 2 streams, a flat stream, columns past the simdgroups' 8-column steps, an integer dtype
    try testing.expect(!serves(&.{ 1, 1024, 2, 4096 }, .bfloat16));
    try testing.expect(!serves(&.{ 1024, 16384 }, .bfloat16));
    try testing.expect(!serves(&.{ 1, 1024, 4, 4100 }, .bfloat16));
    try testing.expect(!serves(&.{ 1, 1024, 4, 4096 }, .int32));
}

test "turboexl3_glm53f stream hc fuse: the switch is off until the plugin sets it" {
    try testing.expect(!on);
}
