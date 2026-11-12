//! `TURBOEXL3_GLM53F_DRAFT_S0=1` (opt-in, exact, with SMALLOP_FUSE): the MTP draft's step 0 (the engine's merged step, 2 to
//! `max_rows` rows: the history rows and the draft row) on SMALLOP's texts where the stock step runs MLX's small ops,
//! through patches/turboexl3-model.patch (head.zig's `step`). SMALLOP takes these pieces at
//! one row (step 1) only; here, at GLM-5.3-Flash's hidden (4096) and bf16:
//! - the front (`front`, kernels/draft_s0_front.metal): every row's embedding row and hidden row under their norms and
//!   their concatenation in one launch, smallop_front.metal's statements a row (the stock: take, two `fast::rms_norm`,
//!   the concatenation);
//! - the first residual and the post-norm through the kit's addNorm at the step's rows (head.zig, `narrowAddNorm`: the
//!   stock `mlx_add` and `fast::rms_norm`);
//! - the last residual and the final norm at the last row, the only row the step returns: without HEAD_LAST_ROW the MoE's
//!   two terms (`add2NormLast`, kernels/draft_s0_add2_norm_last.metal: smallop_add2_norm.metal's statements at that row;
//!   the stock: the two terms' `mlx_add` inside the MoE, the residual's `mlx_add` over every row, the last row taken and
//!   `fast::rms_norm`); with HEAD_LAST_ROW its last-row MoE output (`addNormLast`, kernels/draft_s0_add_norm_last.metal;
//!   the stock `tailLast`: the residual's last row taken, `mlx_add`, `fast::rms_norm`).
//! Each piece is the stock op's arithmetic on the same values in the same order a row (src/smallop_gpu_test.zig holds
//! them to the stock ops bit for bit at 2 to 4 rows). Any other row count, width or dtype keeps the stock ops (null).
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DRAFT_S0`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DRAFT_S0=1: the MTP draft's step 0 (2 to 4 rows) on SMALLOP's texts: its front in one launch, its first residual and post-norm through the kit's addNorm, its last residual and final norm at the last row in one launch; each the stock ops' arithmetic a row, the same bytes\n";

/// The rows step 0 takes (the engine's merged step: a_prev + 2 at MTP depth 2) and GLM-5.3-Flash's hidden.
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 4;
pub const width: c_int = 4096;

/// Launches since the load, by piece; `posts`: the step's addNorm calls taken; `declined`: calls a piece did not take.
pub var fronts: u64 = 0;
pub var posts: u64 = 0;
pub var tails: u64 = 0;
pub var tails_last_row: u64 = 0;
pub var declined: u64 = 0;
var engaged_said = false;

fn engaged(what: []const u8, rows: c_int) void {
    if (engaged_said) return;
    engaged_said = true;
    log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DRAFT_S0 engaged: first piece {s} at {d} rows\n", .{ what, rows });
}

// ── admission (MLX-free) ──

/// Whether step 0's pieces take `rows` rows of `d` in `dt`: 2 to 4 rows, GLM-5.3-Flash's hidden, bf16.
pub fn serves(rows: c_int, d: c_int, dt: mlx.mlx_dtype) bool {
    return rows >= min_rows and rows <= max_rows and d == width and dt == .bfloat16;
}

/// Whether `front` takes the step: `serves`, an unquantized bf16 table of `d`, a bf16 hidden, int32 or uint32 ids.
pub fn frontServes(rows: c_int, d: c_int, table_dt: mlx.mlx_dtype, quantized: bool, hidden_dt: mlx.mlx_dtype, ids_dt: mlx.mlx_dtype) bool {
    return serves(rows, d, hidden_dt) and table_dt == .bfloat16 and !quantized and (ids_dt == .int32 or ids_dt == .uint32);
}

/// The norm mirror's threads a row: four elements a thread (MLX's single-row kernel at `width`: 1024).
pub const threads: c_int = @divExact(width, 4);

// ── the kernels ──

const front_source = @embedFile("kernels/draft_s0_front.metal");
const tail_source = @embedFile("kernels/draft_s0_add2_norm_last.metal");
const last_source = @embedFile("kernels/draft_s0_add_norm_last.metal");
var front_kernel: ?mlx.mlx_fast_metal_kernel = null;
var tail_kernel: ?mlx.mlx_fast_metal_kernel = null;
var last_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn run(kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array, s: mlx.mlx_stream) !Array {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// A step's rows: its residual's middle dimension ([1, rows, d]), or 0 off that shape.
fn stepRows(sh: []const c_int) c_int {
    return if (sh.len == 3 and sh[0] == 1) sh[1] else 0;
}

/// Step 0's front: `cat[enorm(embed[ids]), hnorm(hidden)]` [1, rows, 2 width] as the stock step makes it, one launch.
/// `table` the trunk's embedding as loaded (bf16 [vocab, width], unquantized), `ids` int32 or uint32 [1, rows], `hidden`
/// bf16 [1, rows, width], `eps` the trunk's f32 scalar. Null where it does not serve.
pub fn front(s: mlx.mlx_stream, table: Array, quantized: bool, ids: Array, hidden: Array, we: Array, wh: Array, eps: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const tsh = mlx.getShape(table);
    const hsh = mlx.getShape(hidden);
    const rows = stepRows(hsh);
    if (tsh.len != 2 or rows == 0 or mlx.mlx_array_size(ids) != @as(usize, @intCast(rows))) return null;
    const d = tsh[1];
    if (hsh[2] != d or mlx.getShape(we)[0] != d or mlx.getShape(wh)[0] != d or mlx.mlx_array_dtype(we) != .bfloat16 or mlx.mlx_array_dtype(wh) != .bfloat16) return null;
    if (!frontServes(rows, d, mlx.mlx_array_dtype(table), quantized, mlx.mlx_array_dtype(hidden), mlx.mlx_array_dtype(ids))) {
        declined += 1;
        return null;
    }
    const k = try kernelOf(&front_kernel, "turboexl3_glm53f_draft_s0_front", &.{ "ids", "E", "H", "WE", "WH", "eps" }, &.{"CAT"}, front_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, 2 * d }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 2 * threads, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TN", threads));
    const out = try run(k, cfg, &.{ ids, table, hidden, we, wh, eps }, s);
    fronts += 1;
    engaged("the front", rows);
    return out;
}

/// The last-row pieces' launch: one threadgroup over row R0 of `h` ([1, rows, d]) and the given inputs, [1, 1, d].
fn lastRow(s: mlx.mlx_stream, slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, names: []const [*:0]const u8, src: [*:0]const u8, ins: []const Array, rows: c_int, d: c_int) !Array {
    const k = try kernelOf(slot, name, names, &.{"HN"}, src);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, d }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TN", threads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R0", rows - 1));
    return run(k, cfg, ins, s);
}

/// Step 0's last residual and final norm at its last row: `rms_norm(h + (a + b), w)` at row rows - 1, each add in T as
/// `mlx_add` rounds it (the MoE's routed sum `a` and shared expert `b`, then the residual `h`, all [1, rows, width]), one
/// launch: [1, 1, width]. Null where it does not serve.
pub fn add2NormLast(s: mlx.mlx_stream, h: Array, a: Array, b: Array, w: Array, eps: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(h);
    const rows = stepRows(sh);
    if (rows == 0) return null;
    const d = sh[2];
    const dt = mlx.mlx_array_dtype(h);
    for ([_]Array{ a, b }) |x| {
        if (mlx.mlx_array_dtype(x) != dt or !std.mem.eql(c_int, mlx.getShape(x), sh)) return null;
    }
    if (mlx.mlx_array_dtype(w) != dt or mlx.getShape(w)[0] != d) return null;
    if (!serves(rows, d, dt)) {
        declined += 1;
        return null;
    }
    const out = try lastRow(s, &tail_kernel, "turboexl3_glm53f_draft_s0_add2_norm_last", &.{ "H", "A", "B", "W", "eps" }, tail_source, &.{ h, a, b, w, eps }, rows, d);
    tails += 1;
    engaged("the last residual", rows);
    return out;
}

/// With HEAD_LAST_ROW: step 0's last residual over the MoE's last-row output `d1` ([1, 1, width]) and final norm,
/// `rms_norm(h[rows - 1] + d1, w)`, the add in T as `mlx_add` rounds it, one launch: [1, 1, width] (head.zig's `tailLast`).
/// Null where it does not serve.
pub fn addNormLast(s: mlx.mlx_stream, h: Array, d1: Array, w: Array, eps: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(h);
    const rows = stepRows(sh);
    if (rows == 0) return null;
    const d = sh[2];
    const dt = mlx.mlx_array_dtype(h);
    if (mlx.mlx_array_dtype(d1) != dt or !std.mem.eql(c_int, mlx.getShape(d1), &[_]c_int{ 1, 1, d })) return null;
    if (mlx.mlx_array_dtype(w) != dt or mlx.getShape(w)[0] != d) return null;
    if (!serves(rows, d, dt)) {
        declined += 1;
        return null;
    }
    const out = try lastRow(s, &last_kernel, "turboexl3_glm53f_draft_s0_add_norm_last", &.{ "H", "D1", "W", "eps" }, last_source, &.{ h, d1, w, eps }, rows, d);
    tails_last_row += 1;
    engaged("the last residual over the last-row MoE", rows);
    return out;
}

/// The counts' line at a request's end.
pub fn summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_DRAFT_S0 fronts={d} posts={d} tails={d} tails_last_row={d} declined={d}\n", .{ fronts, posts, tails, tails_last_row, declined }) catch buf[0..0];
}

const testing = std.testing;

test "turboexl3_glm53f stream draft_s0: step 0's pieces take 2 to 4 rows of GLM-5.3-Flash's 4096-wide bf16 rows (an unquantized bf16 table, int32 or uint32 ids) and decline one row, five rows, another width or dtype; the request line counts each piece" {
    for ([_]c_int{ 2, 3, 4 }) |r| {
        try testing.expect(serves(r, 4096, .bfloat16));
        try testing.expect(frontServes(r, 4096, .bfloat16, false, .bfloat16, .int32));
        try testing.expect(frontServes(r, 4096, .bfloat16, false, .bfloat16, .uint32));
    }
    try testing.expect(!serves(1, 4096, .bfloat16));
    try testing.expect(!serves(5, 4096, .bfloat16));
    try testing.expect(!serves(3, 2048, .bfloat16));
    try testing.expect(!serves(3, 4096, .float16));
    try testing.expect(!frontServes(3, 4096, .bfloat16, true, .bfloat16, .int32));
    try testing.expect(!frontServes(3, 4096, .float16, false, .bfloat16, .int32));
    try testing.expect(!frontServes(3, 4096, .bfloat16, false, .bfloat16, .int64));
    try testing.expectEqual(@as(c_int, 1024), threads);
    try testing.expectEqual(@as(c_int, 3), stepRows(&.{ 1, 3, 4096 }));
    try testing.expectEqual(@as(c_int, 0), stepRows(&.{ 2, 3, 4096 }));
    try testing.expectEqual(@as(c_int, 0), stepRows(&.{ 3, 4096 }));
    const saved = .{ fronts, posts, tails, tails_last_row, declined };
    defer {
        fronts = saved[0];
        posts = saved[1];
        tails = saved[2];
        tails_last_row = saved[3];
        declined = saved[4];
    }
    fronts = 356;
    posts = 356;
    tails = 300;
    tails_last_row = 56;
    declined = 2;
    var buf: [160]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_DRAFT_S0 fronts=356 posts=356 tails=300 tails_last_row=56 declined=2\n", summary(&buf));
}
