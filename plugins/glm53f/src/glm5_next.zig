//! GLM-5-Next (glm5_next) pieces outside the trunk's shared layers: DeepSeek-V4's
//! 4-stream Sinkhorn hyper-connection (mHC) around every sublayer, and the pooled
//! indexer + absorbed latent attention of its DeepSeek sparse attention layers.
//! Numerics follow mlx-vlm's `glm5_next` (the reference this port is held to); the
//! fused Sinkhorn-collapse kernel is mlx-vlm's `hc_sinkhorn_collapse`.
//! copied_from: mlx-serve src/plugins/glm5_next/glm5_next.zig@385c3957 (its tests, its affine MoE decode and its per-head
//! qmv left out: an EXL3 pack takes neither), with #1196's (4bab19a5) prompt indexer call.
const std = @import("std");
const kit = @import("mlx_serve");
const kernelSource = @import("kernels.zig").source;
const mlx = kit.mlx;
const dsa = @import("dsa_nax.zig");
const dsa_prompt = @import("dsa_prompt.zig");
const Scope = kit.mlx_scope.Scope;

pub const HC: c_int = 4;
const MIX: c_int = (2 + HC) * HC;

// ── mHC ────────────────────────────────────────────────────────────────

/// mlx-vlm `deepseek_v4/hyper_connection.py` `_make_hc_sinkhorn_collapse_kernel`
/// (Copyright (c) 2026 Apple Inc., MIT): sigmoid pre/post gates and the Sinkhorn-normalized
/// comb from one row's mixes, on simdgroup 0.
const GATES_HEADER = kernelSource("gates_header.metal");

/// The gates, then the pre-weighted collapse of the 4 streams, one row per threadgroup.
const COLLAPSE_SOURCE = kernelSource("collapse.metal");

var collapse_kernel: ?mlx.mlx_fast_metal_kernel = null;
pub const Collapsed = struct {
    /// [B, S, D] in the stream's dtype: the sublayer input before its norm.
    x: mlx.mlx_array,
    /// f32 [B, S, HC]
    post: mlx.mlx_array,
    /// f32 [B, S, HC, HC]
    comb: mlx.mlx_array,

    pub fn deinit(self: *Collapsed) void {
        for ([_]mlx.mlx_array{ self.x, self.post, self.comb }) |a| _ = mlx.mlx_array_free(a);
    }
};

/// `rms_norm(flatten(stream)) @ fn^T` per row, straight from the bf16 stream: 8 rows per
/// threadgroup on simdgroup matrices, SG simdgroups splitting K = HC*D, the RMS scale applied
/// after the dot products.
const MIXES_SOURCE = kernelSource("mixes.metal");
const MIXES_SG: c_int = 8;
const MIXES_KERNEL_MIN_ROWS: c_int = 512;
var mixes_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The per-token mixes `rms_norm(flatten(stream)) @ fn^T`, f32 [B, S, MIX]: the op chain below
/// `MIXES_KERNEL_MIN_ROWS`, the kernel above.
fn mixes(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, eps: f32) !mlx.mlx_array {
    const sh = mlx.getShape(stream);
    // A threadgroup per 8 rows streams the whole `fn`: at decode widths that is one core
    // reading 1.5 MB per sublayer, and MLX's GEMV spreads it instead.
    if (sh[0] * sh[1] < MIXES_KERNEL_MIN_ROWS or !mlx.streamIsGpu(s)) {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_astype(&y, stream, .float32, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{ sh[0], sh[1], sh[2] * sh[3] }, 3, s));
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{sh[2] * sh[3]}, 1, .float32, s));
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_fast_rms_norm(&z, flat, ones, eps, s));
        var fn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fn_t);
        try mlx.check(mlx.mlx_transpose(&fn_t, fn_rows, s));
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&m, z, fn_t, s));
        return m;
    }
    const rows = sh[0] * sh[1];
    const kern = mixes_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "stream", "fn", "params", "eps" };
        const outs = [_][*:0]const u8{"mixes"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_mixes", iv, ov, MIXES_SOURCE, "#include <metal_simdgroup_matrix>\n", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        mixes_kernel = k;
        break :blk k;
    };
    const params = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(params);
    const ev = [1]f32{eps};
    const eps_a = mlx.mlx_array_new_data(&ev, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps_a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], MIX }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(rows + 7, 8) * 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(stream)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SG", MIXES_SG));
    const inputs = [_]mlx.mlx_array{ stream, fn_rows, params, eps_a };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var m = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_vector_array_get(&m, ov, 0));
    return m;
}

/// Collapse the [B, S, HC, D] stream to one sublayer input, with the gates the
/// matching `expand` needs.
pub fn collapse(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, iters: u32, hc_eps: f32, norm_eps: f32) !Collapsed {
    const sh = mlx.getShape(stream);
    if (sh.len != 4 or sh[2] != HC or @rem(sh[3], 4) != 0) return error.Glm5HcShape;
    const m = try mixes(s, stream, fn_rows, norm_eps);
    defer _ = mlx.mlx_array_free(m);
    if (!mlx.streamIsGpu(s)) return collapseOps(s, stream, m, scale, base, iters, hc_eps);
    const kernel = collapse_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x_in", "mixes", "scale", "base" };
        const outs = [_][*:0]const u8{ "collapsed", "post", "comb" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_sinkhorn_collapse", iv, ov, COLLAPSE_SOURCE, GATES_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        collapse_kernel = k;
        break :blk k;
    };
    const rows = sh[0] * sh[1];
    const dt = mlx.mlx_array_dtype(stream);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC, HC }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "U", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS_INT", @intFromFloat(@round(hc_eps / 1e-9))));
    const inputs = [_]mlx.mlx_array{ stream, m, scale, base };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out: Collapsed = .{ .x = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.x, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
    return out;
}

/// `collapse` as MLX ops, where custom Metal kernels cannot run (CUDA).
fn collapseOps(s: mlx.mlx_stream, stream: mlx.mlx_array, m: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, iters: u32, eps: f32) !Collapsed {
    var sc = Scope.init(std.heap.page_allocator, s);
    defer sc.deinit();
    const sh = mlx.getShape(stream);
    const Gate = struct {
        fn z(c: *Scope, mix: mlx.mlx_array, sv: mlx.mlx_array, bv: mlx.mlx_array, k: c_int, lo: c_int, hi: c_int) !mlx.mlx_array {
            return c.add(try c.mul(try c.slice(mix, 2, lo, hi), try c.slice(sv, 0, k, k + 1)), try c.slice(bv, 0, lo, hi));
        }
        /// Each row (axis -1) or column (axis -2) divided by its sum + eps.
        fn norm(c: *Scope, r: mlx.mlx_array, axis: c_int, e: f32) !mlx.mlx_array {
            return c.div(r, try c.addS(try c.sumAxis(r, axis), e));
        }
    };
    const pre = try sc.addS(try sc.sigmoid(try Gate.z(&sc, m, scale, base, 0, 0, HC)), eps);
    const post = try sc.mulS(try sc.sigmoid(try Gate.z(&sc, m, scale, base, 1, HC, 2 * HC)), 2.0);
    const v = try sc.reshape(try Gate.z(&sc, m, scale, base, 2, 2 * HC, MIX), &.{ sh[0], sh[1], HC, HC });
    var r = try sc.addS(try Gate.norm(&sc, try sc.exp(try sc.sub(v, try sc.maxAxis(v, -1))), -1, eps), eps);
    r = try Gate.norm(&sc, r, -2, eps);
    for (1..iters) |_| r = try Gate.norm(&sc, try Gate.norm(&sc, r, -1, eps), -2, eps);
    // Four f32 multiply-adds, as the kernel does: an f32 matmul rounds through TF32 on CUDA.
    const sf = try sc.astype(stream, .float32);
    var x = try sc.zeros(&.{ sh[0], sh[1], sh[3] }, .float32);
    for (0..HC) |i| {
        const ic: c_int = @intCast(i);
        x = try sc.add(x, try sc.mul(try sc.slice(pre, 2, ic, ic + 1), try sc.reshape(try sc.slice(sf, 2, ic, ic + 1), &.{ sh[0], sh[1], sh[3] })));
    }
    return .{ .x = sc.out(try sc.astype(x, mlx.mlx_array_dtype(stream))), .post = sc.out(post), .comb = sc.out(r) };
}

// One token, after oMLX `decode_kernels.py` `hc_pre_fused` (Apache-2.0, see NOTICE): the
// only dispatch between two sublayers. HC_PRE_NT threadgroups each take a K-slice of all the
// mixes; the last to arrive on its row's word of the dispatch's zeroed counter reduces them,
// collapses and norms, then writes the post/comb gates. The sublayer's expand is left to the
// next `hcPre`, which rebuilds each stream element from (y, post, comb, h) as `expand` does.
const HC_PRE_TPG: c_int = 1024;
const HC_PRE_NT: c_int = 16;
/// Rows one `hcPre` dispatch takes (a verify window): one counter word each.
pub const HC_PRE_MAX_ROWS: c_int = 16;
/// Threads per threadgroup the GPU grants this kernel: Metal caps it per compiled
/// kernel by register use (M1/M2 below 1024), so the first eval probes and halves.
var hc_pre_tpg: c_int = HC_PRE_TPG;
var hc_pre_probed = [2]bool{ false, false };
const HC_PRE_BODY = kernelSource("hc_pre_body.metal");
const HC_DEFER_HVAL = "#define HVAL(k) msv_hc_hval<T, HC, D>((k), y + row * D, post_in + row * HC, comb_in + row * HC * HC, h_in + row * HC * D)\n";
const HC_PRE_SOURCE = kernelSource("hc_pre_direct_defines.metal") ++ HC_PRE_BODY;
// Stream pairs cross threadgroups as 32-bit atomics: the even lane stores its odd neighbour's
// element with its own, and the tail loads two elements per atomic.
const HC_PRE_DEFER_SOURCE = HC_DEFER_HVAL ++ kernelSource("hc_pre_defer_defines.metal") ++ HC_PRE_BODY;
const HC_MATERIALIZE_SOURCE = HC_DEFER_HVAL ++ kernelSource("hc_materialize.metal");
var hc_pre_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hc_pre_defer_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hc_materialize_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// A sublayer's expand not applied yet: the stream is `expand(y, h)` under post/comb.
pub const Deferred = struct {
    /// [1, R, D] the sublayer's output.
    y: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]: the sublayer's gates.
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the sublayer's input stream.
    h: mlx.mlx_array,

    pub fn deinit(self: *Deferred) void {
        inline for (.{ self.y, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }

    /// The [1, R, HC, D] stream this stands for, as `expand` writes it.
    pub fn materialize(self: *const Deferred, s: mlx.mlx_stream) !mlx.mlx_array {
        const d = mlx.getShape(self.y)[2];
        const rows = mlx.getShape(self.y)[1];
        const dt = mlx.mlx_array_dtype(self.y);
        const kern = try hcKernel(&hc_materialize_kernel, "glm5_hc_materialize", &.{ "y", "post_in", "comb_in", "h_in" }, &.{"h"}, HC_MATERIALIZE_SOURCE);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, d }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, HC * d, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
        const ins = [_]mlx.mlx_array{ self.y, self.post, self.comb, self.h };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
        return out;
    }
};

pub const Pre = struct {
    /// [1, R, D] the sublayer's normed input.
    normed: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the stream rebuilt from a `Deferred`, else empty.
    h: mlx.mlx_array,

    pub fn deinit(self: *Pre) void {
        inline for (.{ self.normed, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }

    /// The `Deferred` this sublayer leaves: its output `y` over its input stream `h`.
    pub fn defer_(self: *const Pre, y: mlx.mlx_array, h: mlx.mlx_array) Deferred {
        var out: Deferred = .{ .y = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
        _ = mlx.mlx_array_set(&out.y, y);
        _ = mlx.mlx_array_set(&out.post, self.post);
        _ = mlx.mlx_array_set(&out.comb, self.comb);
        _ = mlx.mlx_array_set(&out.h, h);
        return out;
    }
};

/// Whether `hcPre` serves a [1, R, HC, d] stream, R up to `HC_PRE_MAX_ROWS`.
pub fn hcRowsServes(s: mlx.mlx_stream, sh: []const c_int, dt: mlx.mlx_dtype, fn_rows: mlx.mlx_array, norm_w: mlx.mlx_array) bool {
    return mlx.streamIsGpu(s) and sh.len == 4 and sh[0] == 1 and sh[1] >= 1 and sh[1] <= HC_PRE_MAX_ROWS and sh[2] == HC and @rem(sh[3], 8) == 0 and
        @divTrunc(sh[3], 4) <= HC_PRE_TPG and @rem(HC * sh[3], HC_PRE_NT) == 0 and (dt == .bfloat16 or dt == .float16) and
        mlx.mlx_array_dtype(norm_w) == dt and mlx.mlx_array_dtype(fn_rows) == .float32;
}

fn hcKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, GATES_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The normed sublayer input and gates of R rows, from their stream `x` [1, R, HC, D] or
/// from the previous sublayer's `Deferred` (then `Pre.h` is the stream). Callers check
/// `hcRowsServes` first.
pub fn hcPre(s: mlx.mlx_stream, x: ?mlx.mlx_array, deferred: ?*const Deferred, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, norm_w: mlx.mlx_array, iters: u32, hc_eps: f32, norm_eps: f32, rms_eps: f32) !Pre {
    const d = mlx.getShape(norm_w)[0];
    const dt = mlx.mlx_array_dtype(norm_w);
    // Rows ride the grid's y: each has its own K-slices, counter word and gates.
    const rows: c_int = if (x) |xa| mlx.getShape(xa)[1] else mlx.getShape(deferred.?.y)[1];
    if (rows < 1 or rows > HC_PRE_MAX_ROWS) return error.Glm5HcShape;
    const kern = if (deferred != null)
        try hcKernel(&hc_pre_defer_kernel, "glm5_hc_pre_defer", &.{ "y", "post_in", "comb_in", "h_in", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts", "h" }, HC_PRE_DEFER_SOURCE)
    else
        try hcKernel(&hc_pre_kernel, "glm5_hc_pre", &.{ "x", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts" }, HC_PRE_SOURCE);
    // Each dispatch its own counter, zeroed on the host (no dispatch): MLX never sees the kernel write it, so one
    // shared by two dispatches with no data path between them (a batched tick's row groups) mixed their arrivals.
    // One word per row; 8 or more words also keep the input out of the read-only constant address space.
    const zero_words: [HC_PRE_MAX_ROWS]u32 = @splat(0);
    const counter = mlx.mlx_array_new_data(&zero_words, &[_]c_int{HC_PRE_MAX_ROWS}, 1, .uint32);
    defer _ = mlx.mlx_array_free(counter);
    const ev = [2]f32{ norm_eps, rms_eps };
    const eps = mlx.mlx_array_new_data(&ev, &[_]c_int{2}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps);
    var ins_buf: [10]mlx.mlx_array = undefined;
    const ins: []const mlx.mlx_array = if (deferred) |df| blk: {
        ins_buf[0..10].* = .{ df.y, df.post, df.comb, df.h, fn_rows, scale, base, norm_w, eps, counter };
        break :blk ins_buf[0..10];
    } else blk: {
        ins_buf[0..7].* = .{ x.?, fn_rows, scale, base, norm_w, eps, counter };
        break :blk ins_buf[0..7];
    };
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    while (true) {
        const tpg = hc_pre_tpg;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, d }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, HC }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows * HC_PRE_NT, MIX + 1 }, 2, .float32));
        if (deferred != null) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, d }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tpg * HC_PRE_NT, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tpg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        inline for (.{ .{ "HC", HC }, .{ "D", d }, .{ "NT", HC_PRE_NT }, .{ "TPG", tpg }, .{ "ITERS", @as(c_int, @intCast(iters)) }, .{ "EPS_INT", @as(c_int, @intFromFloat(@round(hc_eps / 1e-9))) } }) |kv| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
        }
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
        var out: Pre = .{ .normed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
        errdefer out.deinit();
        try mlx.check(mlx.mlx_vector_array_get(&out.normed, ov, 0));
        try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
        try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
        if (deferred != null) try mlx.check(mlx.mlx_vector_array_get(&out.h, ov, 4));
        const probed = &hc_pre_probed[@intFromBool(deferred != null)];
        if (probed.*) return out;
        // First launch: where Metal caps this kernel's threadgroup below `tpg`, halve and relaunch.
        mlx.check(mlx.mlx_array_eval(out.normed)) catch |e| {
            if (e != error.MlxError or tpg <= 256 or !mlx.takeErrorIf("Thread group size")) return e;
            out.deinit();
            hc_pre_tpg = @divTrunc(tpg, 2);
            kit.log.info("[glm5] hcPre: {d} threads per threadgroup on this GPU\n", .{hc_pre_tpg});
            continue;
        };
        probed.* = true;
        return out;
    }
}

/// `post * x + comb^T @ stream` in f32, one thread per 4 columns of a row (all HC streams).
const EXPAND_SOURCE = kernelSource("expand.metal");
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `post * x + comb^T @ stream`, in f32, back to the stream's dtype: the new stream.
pub fn expand(s: mlx.mlx_stream, x: mlx.mlx_array, stream: mlx.mlx_array, c: *const Collapsed) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return expandOps(s, x, stream, c);
    const sh = mlx.getShape(stream);
    const dt = mlx.mlx_array_dtype(stream);
    const kern = expand_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "x", "res", "post", "comb" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_hc_expand", iv, ov, EXPAND_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        expand_kernel = k;
        break :blk k;
    };
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, dt, s));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(sh[3], 4), sh[0] * sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divTrunc(sh[3], 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    const inputs = [_]mlx.mlx_array{ xc, stream, c.post, c.comb };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `expand` as MLX ops: post * x + comb^T @ stream as f32 multiply-adds (no TF32 matmul),
/// rounded once to the stream's dtype.
fn expandOps(s: mlx.mlx_stream, x: mlx.mlx_array, stream: mlx.mlx_array, c: *const Collapsed) !mlx.mlx_array {
    var sc = Scope.init(std.heap.page_allocator, s);
    defer sc.deinit();
    const sh = mlx.getShape(stream);
    const sf = try sc.astype(stream, .float32);
    var out = try sc.mul(try sc.reshape(c.post, &.{ sh[0], sh[1], HC, 1 }), try sc.reshape(try sc.astype(x, .float32), &.{ sh[0], sh[1], 1, sh[3] }));
    for (0..HC) |j| {
        const jc: c_int = @intCast(j);
        const comb_j = try sc.reshape(try sc.slice(c.comb, 2, jc, jc + 1), &.{ sh[0], sh[1], HC, 1 });
        out = try sc.add(out, try sc.mul(comb_j, try sc.slice(sf, 2, jc, jc + 1)));
    }
    return sc.out(try sc.astype(out, mlx.mlx_array_dtype(stream)));
}

// ── DeepSeek sparse attention: token selection ─────────────────────────

pub const Indexer = struct {
    topk: c_int,
    kpool: c_int,
    heads: c_int,
    head_dim: c_int,
};

fn iota(s: mlx.mlx_stream, n: c_int, start: c_int) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_arange(&a, @floatFromInt(start), @floatFromInt(start + n), 1, .int32, s));
    return a;
}

fn scalarI(v: i32) mlx.mlx_array {
    return mlx.mlx_array_new_int(v);
}

/// `dsa_nax.indexerScores` where the device has NAX.
fn indexerScoresNax(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, kpool: c_int) !?mlx.mlx_array {
    if (!kit.blocks.linear.verify_qmm_nax.verifyQmmNaxAvailable()) return null;
    return dsa.indexerScores(s, q_idx, w, pk, q_start, kpool);
}

/// A forward's indexer rows (key | gate) as the cache stores them: the row that completes a
/// pool (position % kpool == kpool - 1) carries the pool's key in its gate half, the gates'
/// softmax-weighted keys (only pooled keys are read once a pool is whole). One thread per
/// (row, channel); `prev` holds the up-to kpool - 1 cached rows before the forward.
const POOLED_ROWS_SOURCE = kernelSource("pooled_rows.metal");
var pooled_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `chunk` [B, S, 2*Dh] (rows at positions `pos0..`) with each completed pool's key in its
/// completing row's gate half; `prev` [B, n, 2*Dh] are the n <= kpool - 1 cached rows
/// before `pos0` (any one row when n is 0).
pub fn withPooledKeys(s: mlx.mlx_stream, prev: mlx.mlx_array, chunk: mlx.mlx_array, ape: mlx.mlx_array, pos0: c_int, ix: Indexer) !mlx.mlx_array {
    const csh = mlx.getShape(chunk);
    const dt = mlx.mlx_array_dtype(chunk);
    const prev_len: c_int = @min(pos0, mlx.getShape(prev)[1]);
    if (!mlx.streamIsGpu(s)) return withPooledKeysOps(s, prev, chunk, ape, pos0, prev_len, ix);
    const kern = pooled_rows_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "prev", "chunk", "ape", "dims" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_pooled_rows", iv, ov, POOLED_ROWS_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        pooled_rows_kernel = k;
        break :blk k;
    };
    const dims = mlx.mlx_array_new_data(&[_]i32{ pos0, prev_len }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, csh.ptr, csh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ix.head_dim, csh[1], csh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ix.head_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "KPOOL", ix.kpool }, .{ "DH", ix.head_dim } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, kern, cfg, &.{ prev, chunk, ape, dims });
}

/// `withPooledKeys` as MLX ops: every row pools the `kpool` rows ending at it (rows before
/// the cache read as zeros), and only a pool's completing row keeps the result.
fn withPooledKeysOps(s: mlx.mlx_stream, prev: mlx.mlx_array, chunk: mlx.mlx_array, ape: mlx.mlx_array, pos0: c_int, prev_len: c_int, ix: Indexer) !mlx.mlx_array {
    var sc = Scope.init(std.heap.page_allocator, s);
    defer sc.deinit();
    const csh = mlx.getShape(chunk);
    const dt = mlx.mlx_array_dtype(chunk);
    const lead = try sc.zeros(&.{ csh[0], ix.kpool - 1 - prev_len, csh[2] }, dt);
    const rows = try sc.concat(&.{ lead, try sc.slice(prev, 1, 0, prev_len), chunk }, 1);
    var win: [16]mlx.mlx_array = undefined;
    if (ix.kpool > win.len) return error.Glm5HcShape;
    for (win[0..@intCast(ix.kpool)], 0..) |*w, j| w.* = try sc.reshape(try sc.slice(rows, 1, @intCast(j), @as(c_int, @intCast(j)) + csh[1]), &.{ csh[0], csh[1], 1, csh[2] });
    const w4 = try sc.concat(win[0..@intCast(ix.kpool)], 2);
    const logits = try sc.add(try sc.astype(try sc.slice(w4, 3, ix.head_dim, 2 * ix.head_dim), .float32), ape);
    var o = mlx.mlx_array_new();
    const probs = try sc.res(mlx.mlx_softmax_axis(&o, logits, 2, true, s), &o);
    const prod = try sc.mul(try sc.astype(probs, dt), try sc.slice(w4, 3, 0, ix.head_dim));
    const pooled = try sc.reshape(try sc.astype(try sc.sumAxis(try sc.astype(prod, .float32), 2), dt), &.{ csh[0], csh[1], ix.head_dim });
    var done = mlx.mlx_array_new();
    const pos = try sc.res(mlx.mlx_arange(&done, @floatFromInt(pos0 + 1), @floatFromInt(pos0 + 1 + csh[1]), 1, .int32, s), &done);
    var rem = mlx.mlx_array_new();
    const r = try sc.res(mlx.mlx_remainder(&rem, pos, try sc.keep(mlx.mlx_array_new_int(ix.kpool)), s), &rem);
    var is0 = mlx.mlx_array_new();
    const completes = try sc.reshape(try sc.res(mlx.mlx_equal(&is0, r, try sc.keep(mlx.mlx_array_new_int(0)), s), &is0), &.{ 1, csh[1], 1 });
    var g = mlx.mlx_array_new();
    const gate = try sc.res(mlx.mlx_where(&g, completes, pooled, try sc.slice(chunk, 2, ix.head_dim, 2 * ix.head_dim), s), &g);
    return sc.out(try sc.concat(&.{ try sc.slice(chunk, 2, 0, ix.head_dim), gate }, 2));
}

/// [B, P, Dh] keys of the first `p` pools: the completing rows' gate halves.
fn storedPoolKeys(s: mlx.mlx_stream, idx_cache: mlx.mlx_array, p: c_int, ix: Indexer) !mlx.mlx_array {
    const b = mlx.getShape(idx_cache)[0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, idx_cache, &[_]c_int{ 0, ix.kpool - 1, ix.head_dim }, 3, &[_]c_int{ b, p * ix.kpool, 2 * ix.head_dim }, 3, &[_]c_int{ 1, ix.kpool, 1 }, 3, s));
    return out;
}

/// oMLX `decode_kernels.py` `_DSA_EXPAND_SOURCE` (Apache-2.0, see NOTICE): the selected pools
/// [S, SEL_K] of queries at `qpos..` to their token indices, the query's partial tail pool
/// after them, -1 for an unusable slot or the padding up to the output width.
const EXPAND_SEL_SOURCE = kernelSource("expand_sel.metal");
var expand_sel_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// int32 [B, S, topk + kpool - 1] token indices from the selected pools `sel` [B, S, select_k].
fn expandSelection(s: mlx.mlx_stream, sel: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !mlx.mlx_array {
    const ssh = mlx.getShape(sel);
    if (!mlx.streamIsGpu(s)) return expandSelectionOps(s, sel, q_start, p, ix);
    const kern = expand_sel_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "selected", "qpos" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_expand_sel", iv, ov, EXPAND_SEL_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        expand_sel_kernel = k;
        break :blk k;
    };
    const out_w = ix.topk + ix.kpool - 1;
    const qv = [2]i32{ q_start, p };
    const qpos = mlx.mlx_array_new_data(&qv, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(qpos);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ssh[0], ssh[1], out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_w, ssh[0] * ssh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, out_w), 1, 1));
    inline for (.{ .{ "SEL_K", ssh[2] }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    const inputs = [_]mlx.mlx_array{ sel, qpos };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `expandSelection` as MLX ops: each selected pool's tokens where the pool is complete by
/// the row's position, -1 up to `topk`, then the row's own unfinished pool.
fn expandSelectionOps(s: mlx.mlx_stream, sel: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !mlx.mlx_array {
    var sc = Scope.init(std.heap.page_allocator, s);
    defer sc.deinit();
    const ssh = mlx.getShape(sel);
    const I = struct {
        fn n(c: *Scope, v: c_int) !mlx.mlx_array {
            return c.keep(mlx.mlx_array_new_int(v));
        }
        fn iota(c: *Scope, len: c_int, start: c_int, shape: []const c_int) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            return c.reshape(try c.res(mlx.mlx_arange(&o, @floatFromInt(start), @floatFromInt(start + len), 1, .int32, c.s), &o), shape);
        }
        fn op(c: *Scope, comptime f: anytype, x: mlx.mlx_array, y: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            return c.res(f(&o, x, y, c.s), &o);
        }
        fn where(c: *Scope, cond: mlx.mlx_array, x: mlx.mlx_array) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            return c.res(mlx.mlx_where(&o, cond, x, try n(c, -1), c.s), &o);
        }
    };
    const kp = try I.n(&sc, ix.kpool);
    const sel_w = ssh[2] * ix.kpool;
    const qp = try I.iota(&sc, ssh[0] * ssh[1], q_start, &.{ ssh[0], ssh[1], 1 });
    var o = mlx.mlx_array_new();
    const pool = try sc.res(mlx.mlx_repeat_axis(&o, try sc.astype(sel, .int32), ix.kpool, 2, s), &o);
    const tok = try sc.add(try sc.mul(pool, kp), try I.op(&sc, mlx.mlx_remainder, try I.iota(&sc, sel_w, 0, &.{ 1, 1, sel_w }), kp));
    const pool_end = try sc.sub(try sc.mul(try sc.add(pool, try I.n(&sc, 1)), kp), try I.n(&sc, 1));
    const valid = try I.op(&sc, mlx.mlx_logical_and, try I.op(&sc, mlx.mlx_less, pool, try I.n(&sc, p)), try I.op(&sc, mlx.mlx_less_equal, pool_end, qp));
    var parts: [3]mlx.mlx_array = undefined;
    var np: usize = 0;
    parts[np] = try I.where(&sc, valid, tok);
    np += 1;
    if (ix.topk > sel_w) {
        parts[np] = try sc.add(try sc.zeros(&.{ ssh[0], ssh[1], ix.topk - sel_w }, .int32), try I.n(&sc, -1));
        np += 1;
    }
    const tail = try I.op(&sc, mlx.mlx_remainder, try sc.add(qp, try I.n(&sc, 1)), kp);
    const t = try I.iota(&sc, ix.kpool - 1, 0, &.{ 1, 1, ix.kpool - 1 });
    const tail_tok = try sc.add(try sc.sub(try sc.add(qp, try I.n(&sc, 1)), tail), t);
    parts[np] = try sc.broadcast(try I.where(&sc, try I.op(&sc, mlx.mlx_less, t, tail), tail_tok), &.{ ssh[0], ssh[1], ix.kpool - 1 });
    np += 1;
    return sc.out(try sc.concat(parts[0..np], 2));
}

/// `x` that also evaluates `dep`: a cache row nothing in this forward reads (the indexer below
/// its budget) would otherwise stay a lazy chain across tokens until the first read.
pub fn withDependency(x: mlx.mlx_array, dep: mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    const dv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dep}, 1);
    defer _ = mlx.mlx_vector_array_free(dv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_depends(&ov, iv, dv));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether every query of a forward reaching `kv_len` keys reads every earlier key: the
/// pools all fit the budget, so the selection is plain causal attention.
pub fn selectsAll(kv_len: c_int, ix: Indexer) bool {
    return @divTrunc(kv_len, ix.kpool) <= @divTrunc(ix.topk, ix.kpool);
}

/// The token indices each query reads, int32 [B, S, topk + kpool - 1] with -1 for an
/// unused slot: the `topk/kpool` best complete pools ending at or before the query, in
/// pool order, padded to `topk`, then the query's partial tail pool.
/// `q_idx` [B, S, H, Dh] and `w` f32 [B, S, H] (already scaled by H^-0.5) score pools;
/// `idx_cache` [B, T, 2*Dh] holds each token's indexer key and pool gate. The queries are
/// the last S tokens; rows are scored in chunks of at most 2^27 (query, pool) scores.
pub fn selectTokens(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) !mlx.mlx_array {
    const t = mlx.getShape(idx_cache)[1];
    const sq = mlx.getShape(q_idx)[1];
    const rows = @max(64, @divTrunc(@divTrunc(@as(c_int, 1 << 27), @max(1, @divTrunc(t, ix.kpool))), 64) * 64);
    if (sq <= rows) return selectRows(s, q_idx, w, idx_cache, ix, t - sq);
    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    var r0: c_int = 0;
    while (r0 < sq) : (r0 += rows) {
        const r1 = @min(sq, r0 + rows);
        const qsh = mlx.getShape(q_idx);
        var qc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_idx, &[_]c_int{ 0, r0, 0, 0 }, 4, &[_]c_int{ qsh[0], r1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var wc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wc);
        try mlx.check(mlx.mlx_slice(&wc, w, &[_]c_int{ 0, r0, 0 }, 3, &[_]c_int{ qsh[0], r1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try parts.append(std.heap.c_allocator, try selectRows(s, qc, wc, idx_cache, ix, t - sq + r0));
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

/// One query's indexer scores in one dispatch: each simdgroup loads PPS pool keys in turn
/// (`withPooledKeys`' layout) and lane h scores head h against each; the head sum is the
/// pool's score, -FLT_MAX for a pool the query cannot see.
const DEC_SCORE_SGS: c_int = 8;
const DEC_SCORE_PPS: c_int = 4;
const DEC_SCORE_SOURCE = kernelSource("dec_score.metal");
var dec_score_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The top SEL_K of one query's pool scores as token indices in `expandSelection`'s layout,
/// in one threadgroup: a 4-pass 8-bit radix select finds the SEL_K-th largest score, then an
/// ordered compaction (ties to the lower index) writes the pools chronologically.
const TOPK_EXPAND_SOURCE = kernelSource("topk_expand.metal");
const TOPK_EXPAND_HEADER = kernelSource("topk_expand_header.metal");
var topk_expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn decodeKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn applyOne(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether one query's selection runs on `decodeSelect`.
fn decodeSelectServes(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    return mlx.getShape(q_idx)[1] == 1 and decodeRowsServe(s, q_idx, idx_cache, ix);
}

/// `decodeSelect`'s shape conditions short of the row count.
fn decodeRowsServe(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    const qs = mlx.getShape(q_idx);
    const dt = mlx.mlx_array_dtype(idx_cache);
    return mlx.streamIsGpu(s) and qs[0] == 1 and ix.heads == 32 and ix.head_dim == 128 and
        (dt == .bfloat16 or dt == .float16) and mlx.mlx_array_dtype(q_idx) == dt;
}

/// f32 [p] scores of one query at `q_pos` (`ws` carries the Dh^-0.5).
fn decodeScores(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int) !mlx.mlx_array {
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const sk = try decodeKernel(&dec_score_kernel, "glm5_dsa_decode_scores", &.{ "cache", "q", "w", "dims" }, DEC_SCORE_SOURCE, "");
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{p}, 1, .float32));
    const per_tg = DEC_SCORE_SGS * DEC_SCORE_PPS;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, DEC_SCORE_SGS * 32, @divTrunc(p + per_tg - 1, per_tg), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, DEC_SCORE_SGS * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(idx_cache)));
    inline for (.{ .{ "DH", ix.head_dim }, .{ "KPOOL", ix.kpool }, .{ "SGS", DEC_SCORE_SGS }, .{ "PPS", DEC_SCORE_PPS } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, sk, cfg, &.{ idx_cache, q_idx, ws, dims });
}

/// `selectRows` for one query at `q_pos` with `select_k` < `p` pools: int32 [1, 1, topk + kpool - 1].
fn decodeSelect(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const scores = try decodeScores(s, q_idx, ws, idx_cache, ix, q_pos, p);
    defer _ = mlx.mlx_array_free(scores);
    return topkExpand(s, scores, ix, q_pos, p, select_k);
}

/// The chronological top `select_k` of `scores` [p] expanded to token indices.
fn topkExpand(s: mlx.mlx_stream, scores: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const tk = try decodeKernel(&topk_expand_kernel, "glm5_dsa_topk_expand", &.{ "scores", "dims" }, TOPK_EXPAND_SOURCE, TOPK_EXPAND_HEADER);
    const out_w = ix.topk + ix.kpool - 1;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    inline for (.{ .{ "SEL_K", select_k }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return applyOne(s, tk, cfg, &.{ scores, dims });
}

/// `selectTokens` for queries at positions `q_start .. q_start + S`.
fn selectRows(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_start: c_int) !mlx.mlx_array {
    const csh = mlx.getShape(idx_cache);
    const b = csh[0];
    const t = csh[1];
    const sq = mlx.getShape(q_idx)[1];
    const p = @divTrunc(t, ix.kpool);
    const select_k = @min(@divTrunc(ix.topk, ix.kpool), p);

    var sel = mlx.mlx_array_new(); // int32 [B, S, select_k] pool ids, chronological
    defer _ = mlx.mlx_array_free(sel);
    if (select_k == p) {
        // Every complete pool (none yet: pool 0, which the expansion finds unusable).
        const pi = try iota(s, @max(p, 1), 0);
        defer _ = mlx.mlx_array_free(pi);
        try mlx.check(mlx.mlx_broadcast_to(&sel, pi, &[_]c_int{ b, sq, @max(p, 1) }, 3, s));
    } else if (decodeSelectServes(s, q_idx, idx_cache, ix)) {
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        return decodeSelect(s, q_idx, ws, idx_cache, ix, q_start, p, select_k);
    } else if (sq <= HC_PRE_MAX_ROWS and decodeRowsServe(s, q_idx, idx_cache, ix)) {
        // A verify window: the one-query select per row (its own position), joined.
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        const qsh = mlx.getShape(q_idx);
        var rows_out: [HC_PRE_MAX_ROWS]mlx.mlx_array = undefined;
        var n: usize = 0;
        defer for (rows_out[0..n]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        while (n < @as(usize, @intCast(sq))) : (n += 1) {
            const i: c_int = @intCast(n);
            var qi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qi);
            try mlx.check(mlx.mlx_slice(&qi, q_idx, &[_]c_int{ 0, i, 0, 0 }, 4, &[_]c_int{ 1, i + 1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var wi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wi);
            try mlx.check(mlx.mlx_slice(&wi, ws, &[_]c_int{ 0, i, 0 }, 3, &[_]c_int{ 1, i + 1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            rows_out[n] = try decodeSelect(s, qi, wi, idx_cache, ix, q_start + i, p, select_k);
        }
        const vec = mlx.mlx_vector_array_new_data(&rows_out, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
        return out;
    } else {
        // GLM53F_DSA_PROMPT_INDEXER: the scores, their top-k and its expansion in one launch (the same tokens).
        if (try dsa_prompt.selectRows(s, q_idx, w, idx_cache, q_start, p, select_k, ix)) |tokens| return tokens;
        // Scores: sum_h w_h * relu(q_h . pool_key) * Dh^-0.5, f32; pools a query cannot see lose.
        const pk = try storedPoolKeys(s, idx_cache, p, ix);
        defer _ = mlx.mlx_array_free(pk);
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        // One row pads a whole NAX tile: the composed scores are cheaper there.
        if (if (sq > 1) try indexerScoresNax(s, q_idx, ws, pk, q_start, ix.kpool) else null) |nax| {
            _ = mlx.mlx_array_free(masked);
            masked = nax;
        } else try maskedScores(s, &masked, q_idx, ws, pk, q_start, p, ix);
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, masked, s));
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, neg, select_k - 1, -1, s));
        var top = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top);
        try mlx.check(mlx.mlx_slice(&top, part, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ b, sq, select_k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var top32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top32);
        try mlx.check(mlx.mlx_astype(&top32, top, .int32, s));
        try mlx.check(mlx.mlx_sort_axis(&sel, top32, -1, s));
    }
    return expandSelection(s, sel, q_start, p, ix);
}

/// The composed scores off NAX: [B, S, P] f32, -max for a pool past the query.
fn maskedScores(s: mlx.mlx_stream, out: *mlx.mlx_array, q_idx: mlx.mlx_array, ws: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !void {
    const sq = mlx.getShape(q_idx)[1];
    var pkf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pkf);
    try mlx.check(mlx.mlx_astype(&pkf, pk, .float32, s));
    var pk_t = mlx.mlx_array_new(); // [B, 1, Dh, P]
    defer _ = mlx.mlx_array_free(pk_t);
    {
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_expand_dims(&e, pkf, 1, s));
        try mlx.check(mlx.mlx_swapaxes(&pk_t, e, -1, -2, s));
    }
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_astype(&qf, q_idx, .float32, s));
    var sc = mlx.mlx_array_new(); // [B, S, H, P]
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, qf, pk_t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var rl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rl);
    try mlx.check(mlx.mlx_maximum(&rl, sc, zero, s));
    var ws4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws4);
    try mlx.check(mlx.mlx_expand_dims(&ws4, ws, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rl, ws4, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sum_axis(&scores, weighted, 2, false, s));
    // Pool p is visible once its last token (p*kpool + kpool - 1) is at or before the query.
    const pi = try iota(s, p, 0);
    defer _ = mlx.mlx_array_free(pi);
    const kp = scalarI(ix.kpool);
    defer _ = mlx.mlx_array_free(kp);
    var pend = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pend);
    try mlx.check(mlx.mlx_multiply(&pend, pi, kp, s));
    const off = scalarI(ix.kpool - 1 - q_start);
    defer _ = mlx.mlx_array_free(off);
    var pe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe);
    try mlx.check(mlx.mlx_add(&pe, pend, off, s)); // pool end relative to the first query
    var pe2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe2);
    try mlx.check(mlx.mlx_reshape(&pe2, pe, &[_]c_int{ 1, 1, p }, 3, s));
    const qi = try iota(s, sq, 0);
    defer _ = mlx.mlx_array_free(qi);
    var qi2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qi2);
    try mlx.check(mlx.mlx_reshape(&qi2, qi, &[_]c_int{ 1, sq, 1 }, 3, s));
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_less_equal(&cand, pe2, qi2, s));
    const lowest = mlx.mlx_array_new_float(-std.math.floatMax(f32));
    defer _ = mlx.mlx_array_free(lowest);
    try mlx.check(mlx.mlx_where(out, cand, scores, lowest, s));
}

// ── absorbed latent attention ──────────────────────────────────────────

/// Single-query latent attention whose heads share one key set: q [N, H, 1, R] against keys
/// [N, W, R] (values = keys), `valid` [N, 1, W] bool or null; a null `scale` means q arrives
/// scaled. The heads fold into the GEMM's M: MLX's sdpa declines a GQA factor past 32 and
/// its fallback reads the keys once per head.
/// bool [1, heads * rows, t]: query row i (the last `rows` positions of `t`) sees keys up to its
/// own position, for `latentAttendFolded` over a verify window folded into the head axis.
pub fn latentCausalMask(s: mlx.mlx_stream, heads: c_int, rows: c_int, t: c_int) !mlx.mlx_array {
    const ks = try iota(s, t, 0);
    defer _ = mlx.mlx_array_free(ks);
    const qs = try iota(s, rows, t - rows);
    defer _ = mlx.mlx_array_free(qs);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    try mlx.check(mlx.mlx_reshape(&k2, ks, &[_]c_int{ 1, 1, t }, 3, s));
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_reshape(&q2, qs, &[_]c_int{ 1, rows, 1 }, 3, s));
    var valid = mlx.mlx_array_new(); // [1, rows, t]
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less_equal(&valid, k2, q2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_broadcast_to(&wide, valid, &[_]c_int{ heads, rows, t }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, wide, &[_]c_int{ 1, heads * rows, t }, 3, s));
    return out;
}

pub fn latentAttendFolded(s: mlx.mlx_stream, q: mlx.mlx_array, keys: mlx.mlx_array, valid: ?mlx.mlx_array, scale: ?f32) !mlx.mlx_array {
    const qsh = mlx.getShape(q);
    const n = qsh[0];
    const h = qsh[1];
    const r = qsh[3];
    const dt = mlx.mlx_array_dtype(q);
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    try mlx.check(mlx.mlx_reshape(&q3, q, &[_]c_int{ n, h, r }, 3, s));
    var qs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qs);
    if (scale) |sv| {
        const sc_f = mlx.mlx_array_new_float(sv);
        defer _ = mlx.mlx_array_free(sc_f);
        var sc_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc_t);
        try mlx.check(mlx.mlx_astype(&sc_t, sc_f, dt, s));
        try mlx.check(mlx.mlx_multiply(&qs, q3, sc_t, s));
    } else _ = mlx.mlx_array_set(&qs, q3);
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_swapaxes(&kt, keys, 1, 2, s));
    var scores = mlx.mlx_array_new(); // [N, H, W]
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_matmul(&scores, qs, kt, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (valid) |v| {
        const ninf_f = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(ninf_f);
        var ninf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ninf);
        try mlx.check(mlx.mlx_astype(&ninf, ninf_f, dt, s));
        try mlx.check(mlx.mlx_where(&masked, v, scores, ninf, s));
    } else _ = mlx.mlx_array_set(&masked, scores);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, true, s));
    var o = mlx.mlx_array_new(); // [N, H, R]
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_matmul(&o, p, keys, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ n, h, 1, r }, 4, s));
    return out;
}

/// Attention of absorbed queries over the latent rows each query selects.
/// `q_abs` [B, H, S, R], `latent` [B, 1, T, R], `idx` int32 [B, S, W] (-1 = unused).
/// Returns [B, H, S, R]. Rows are processed `chunk` queries at a time so the
/// gathered keys stay bounded.
pub fn sparseLatentAttention(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: ?f32, chunk: c_int) !mlx.mlx_array {
    const qsh = mlx.getShape(q_abs);
    const b = qsh[0];
    const h = qsh[1];
    const sq = qsh[2];
    const r = qsh[3];
    const t = mlx.getShape(latent)[2];
    const wdt = mlx.getShape(idx)[2];
    var lat2 = mlx.mlx_array_new(); // [B, T, R]
    defer _ = mlx.mlx_array_free(lat2);
    try mlx.check(mlx.mlx_reshape(&lat2, latent, &[_]c_int{ b, t, r }, 3, s));

    var outs: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (outs.items) |a| _ = mlx.mlx_array_free(a);
        outs.deinit(std.heap.c_allocator);
    }
    var at: c_int = 0;
    while (at < sq) : (at += chunk) {
        const n = @min(chunk, sq - at);
        var ic = mlx.mlx_array_new(); // [B, n, W]
        defer _ = mlx.mlx_array_free(ic);
        try mlx.check(mlx.mlx_slice(&ic, idx, &[_]c_int{ 0, at, 0 }, 3, &[_]c_int{ b, at + n, wdt }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const zero = mlx.mlx_array_new_int(0);
        defer _ = mlx.mlx_array_free(zero);
        var valid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(valid);
        try mlx.check(mlx.mlx_greater_equal(&valid, ic, zero, s));
        // Unused slots are -1, every other index is a cached row.
        var safe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe);
        try mlx.check(mlx.mlx_maximum(&safe, ic, zero, s));
        // Gather [B, n*W, R] rows, then fold each query into the batch.
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        if (b == 1) {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{n * wdt}, 1, s));
            try mlx.check(mlx.mlx_take_axis(&rows, lat2, flat_i, 1, s));
        } else {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{ b, n * wdt, 1 }, 3, s));
            try mlx.check(mlx.mlx_take_along_axis(&rows, lat2, flat_i, 1, s));
        }
        var keys = mlx.mlx_array_new(); // [B*n, W, R]
        defer _ = mlx.mlx_array_free(keys);
        try mlx.check(mlx.mlx_reshape(&keys, rows, &[_]c_int{ b * n, wdt, r }, 3, s));
        var qc = mlx.mlx_array_new(); // [B, H, n, R]
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_abs, &[_]c_int{ 0, 0, at, 0 }, 4, &[_]c_int{ b, h, at + n, r }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var qt = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qc, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var q4 = mlx.mlx_array_new(); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(q4);
        try mlx.check(mlx.mlx_reshape(&q4, qt, &[_]c_int{ b * n, h, 1, r }, 4, s));
        var mask = mlx.mlx_array_new(); // [B*n, 1, W]
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_reshape(&mask, valid, &[_]c_int{ b * n, 1, wdt }, 3, s));
        const o = try latentAttendFolded(s, q4, keys, mask, scale); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(o);
        var o4 = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(o4);
        try mlx.check(mlx.mlx_reshape(&o4, o, &[_]c_int{ b, n, h, r }, 4, s));
        var ot = mlx.mlx_array_new(); // [B, H, n, R]
        try mlx.check(mlx.mlx_transpose_axes(&ot, o4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try outs.append(std.heap.c_allocator, ot);
    }
    if (outs.items.len == 1) {
        const only = outs.items[0];
        outs.items.len = 0;
        return only;
    }
    const vec = mlx.mlx_vector_array_new_data(outs.items.ptr, outs.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

/// `sparseLatentAttention` on the tensor units (`dsa_nax.sparseLatent`) where the device has NAX.
pub fn sparseLatentNax(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!kit.blocks.linear.verify_qmm_nax.verifyQmmNaxAvailable()) return null;
    return dsa.sparseLatent(s, q_abs, latent, idx, scale);
}
