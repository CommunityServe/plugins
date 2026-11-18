//! `TURBOEXL3_GLM53F_MOE_COMBINE1=1` (on by default, exact): a decode or verify call's routed join and combine as one
//! launch (src/streamed/glm5n/experts.zig's decode lane).
//!
//! The stock decode lane ends a routed call with the join (src/streamed/experts.zig `join`: the outputs' concatenate,
//! then the take of each routed position's row by the join's order) and `combineStock` (astype f32, the multiply by the
//! f32 route weight, the sum over the 8 routes, astype bf16): a take and four launches after the concatenate, every MoE
//! layer of every verify, each on the path from the call's last wave to the next sublayer. Here the call ends before
//! the take (`routedHoistIndexed`, JOIN_INDEXED: the concatenated outputs and the order) and one text
//! (src/kernels/moe_combine1.metal) reads each position's row in place through the order and runs the stock chain's
//! arithmetic: each product f32(row) * f32(w) rounded to f32 (MLX's Multiply, never contracted), added to the +0 init,
//! folded k = 0..7 in router order (MLX's col_reduce_small at threadgroup y = 8, reduce_col.h), bf16 once. The prompt
//! pass's combine text (src/streamed/mimo/kernels/mimo_prefill_combine.metal) states the same arithmetic. The weights
//! are read in their stored type (the bank's `rows` hands them f32; bf16 widens exactly). So the same bits, in one launch
//! where the stock lane runs five (the take, astype, multiply, sum, astype).
//!
//! The order matches MLX's reduction at lib/mlx-src c69303b0 (the frozen line's MLX and mlx-serve main a5e30916's); the
//! frozen line's src/moe_combine_gpu_test.zig held the output to the stock chain's bit for bit at 1 to 4 rows on it.
//! GLM-5.3-Flash's dims only (hidden 4096, top-8; rows bf16 or f32, weights bf16 or f32, the order u32): any other call
//! runs the stock chain over the same joined rows.
//! copied_from: CommunityServe plugins/turboexl3_glm53f/src/moe_combine1.zig@ac87fe1 (its launch helpers from
//! src/kda_prompt.zig@ac87fe1 `kernelOf` and `applyKernel`; its request summary line left out).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("host/log.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_MOE_COMBINE1`, set by the plugin at its open, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_MOE_COMBINE1=1: a decode or verify call's routed join and combine (the join's take, astype f32, the weights' multiply, the sum over the 8 routes, astype bf16) as one launch reading each route's row in place through the join's order (src/kernels/moe_combine1.metal: MLX's Multiply and col_reduce_small statements in their order): the stock chain's bits\n";
/// Calls the text combined since the load.
pub var calls: u64 = 0;
var engaged = false;

/// GLM-5.3-Flash's routed dims: the hidden width and the routes a token.
pub const hidden: c_int = 4096;
pub const top_k: c_int = 8;
/// The most rows a call it takes (the decode lane's 4, with room: the text holds at any count).
pub const max_rows: c_int = 16;

const source = @embedFile("kernels/moe_combine1.metal");
const header = @embedFile("kernels/moe_combine1_header.metal");
var kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The shapes and types a call hands the text: `rows` the joined outputs [R, hidden], `order` the join's order u32
/// [n * top_k], `w` the route weights [n, top_k] (or flat).
pub const Shapes = struct {
    rows: []const c_int,
    rows_dt: mlx.mlx_dtype,
    order: []const c_int,
    order_dt: mlx.mlx_dtype,
    w: []const c_int,
    w_dt: mlx.mlx_dtype,
};

/// The rows (tokens) of a call the text takes, null where it does not (MLX-free): rows 2-D [R >= 1, 4096] in bf16 or
/// f32, the order 1-D u32 of n * 8, the weights bf16 or f32 of n * 8 (2-D [n, 8] or flat), 1 <= n <= `max_rows`.
pub fn serves(sh: Shapes) ?c_int {
    if (sh.rows.len != 2 or sh.rows[0] < 1 or sh.rows[1] != hidden) return null;
    if (sh.rows_dt != .bfloat16 and sh.rows_dt != .float32) return null;
    if (sh.order.len != 1 or sh.order_dt != .uint32) return null;
    const ids = sh.order[0];
    if (ids < top_k or @rem(ids, top_k) != 0) return null;
    const n = @divExact(ids, top_k);
    if (n > max_rows) return null;
    if (sh.w_dt != .bfloat16 and sh.w_dt != .float32) return null;
    const w_ok = (sh.w.len == 2 and sh.w[0] == n and sh.w[1] == top_k) or (sh.w.len == 1 and sh.w[0] == ids);
    return if (w_ok) n else null;
}

fn kernelOf(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The combine of a decode-lane call (the switch on): [n, 4096] bf16, a new array; null with the switch off or on a call
/// `serves` does not take (the caller runs the stock join and combine over the same rows).
pub fn combine(s: mlx.mlx_stream, rows: Array, order: Array, w: Array) !?Array {
    if (!on) return null;
    const n = serves(.{
        .rows = mlx.getShape(rows),
        .rows_dt = mlx.mlx_array_dtype(rows),
        .order = mlx.getShape(order),
        .order_dt = mlx.mlx_array_dtype(order),
        .w = mlx.getShape(w),
        .w_dt = mlx.mlx_array_dtype(w),
    }) orelse return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, hidden }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, n * @divExact(hidden, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(rows)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "WT", mlx.mlx_array_dtype(w)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", top_k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    const kern = try kernelOf("turboexl3_glm53f_moe_combine1", &.{ "rows", "order", "w" }, &.{"y"});
    const ins = [_]Array{ rows, order, w };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    calls += 1;
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_MOE_COMBINE1 engaged: a {d}-row call's join and combine in one launch over {d} joined rows ({s} weights)\n", .{ n, mlx.getShape(rows)[0], @tagName(mlx.mlx_array_dtype(w)) });
    }
    return y;
}

test "turboexl3_glm53f moe_combine1: the text takes GLM-5.3-Flash's decode-lane calls (4096 wide, top-8, bf16 or f32 rows and weights, u32 order) and declines every other shape" {
    const t = std.testing;
    const ok = Shapes{ .rows = &.{ 48, 4096 }, .rows_dt = .bfloat16, .order = &.{24}, .order_dt = .uint32, .w = &.{ 3, 8 }, .w_dt = .bfloat16 };
    try t.expectEqual(@as(?c_int, 3), serves(ok));
    // the bank's `rows` hands the combine f32 weights [n, 8]
    var s = ok;
    s.w_dt = .float32;
    try t.expectEqual(@as(?c_int, 3), serves(s));
    s.w = &.{24};
    s.rows_dt = .float32;
    try t.expectEqual(@as(?c_int, 3), serves(s));
    s = ok;
    s.order = &.{8};
    s.w = &.{ 1, 8 };
    s.rows = &.{ 8, 4096 };
    try t.expectEqual(@as(?c_int, 1), serves(s));
    // declined: another width, a top-k the order does not divide, an i32 order, f16 weights, weights of another count,
    // a 3-D rows array, more rows than the text takes
    s = ok;
    s.rows = &.{ 48, 2048 };
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.order = &.{20};
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.order_dt = .int32;
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.w_dt = .float16;
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.w = &.{ 4, 8 };
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.rows = &.{ 3, 8, 4096 };
    try t.expectEqual(@as(?c_int, null), serves(s));
    s = ok;
    s.order = &.{17 * 8};
    s.w = &.{17 * 8};
    try t.expectEqual(@as(?c_int, null), serves(s));
}
