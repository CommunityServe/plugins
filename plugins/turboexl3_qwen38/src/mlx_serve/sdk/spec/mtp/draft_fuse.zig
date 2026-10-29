//! copied_from: src/sdk/spec/mtp/draft_fuse.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The fused draft pick: a greedy draft step's pick in fewer launches, the last of which also builds the NEXT draft
//! step's front (that token's normalized embedding, which the next head forward reads instead of building `embedding` +
//! `rmsNorm(pre_norm_emb)`). Bit for bit the chain it replaces and proposal side only, so the verify's bytes stand.
//! The argmax stays MLX's own op on the chain's own values (which index it returns for a row holding NaN is MLX's).
//! - rerank (`mtp.rerankPickFused`): one launch runs the top-32 finalize over the partial launch's candidates and the
//!   32-row re-score through the head's rows read in place; MLX's argmax over those values; one launch for
//!   shortlistArgmax's take + astype and the front.
//! - exact (a head's own logits, or the full readout once a coarse head is gone): MLX's where, argmax (+ astype) as in
//!   the chain, then the front launch.
//! Anything outside the envelope declines ("[draft-fuse] declined: <reason>", once per reason) and the chain runs.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const log = @import("../../util/log.zig");

/// The front's threadgroup: rms_single_row's at hidden 2560 (ceil(2560 / 4) threads).
pub const TGN: c_int = 640;
/// The scores launch's threadgroup: the top-32 finalize's own; its 8 simdgroups also re-score.
pub const SCORES_TG: c_int = 256;
const N_READS: c_int = 4;

/// The rerank pick's first launch: the finalize (spliced as is) between the shortlist's threadgroup ids and the
/// re-score.
pub const SCORES_SOURCE = kernelSource("kernels/spec/draft_fuse_scores_head.metal") ++ kernelSource("kernels/spec/mtp_top32_final.metal") ++ "\n" ++ kernelSource("kernels/spec/draft_fuse_scores.metal");
/// The front launch: the pick's id and the next step's normalized embedding.
pub const FRONT_SOURCE = kernelSource("kernels/spec/draft_fuse_front.metal");

pub const scores_inputs = [_][*:0]const u8{ "cand_ord", "cand_idx", "x", "w_q", "scales", "biases" };
pub const scores_outputs = [_][*:0]const u8{ "cands", "scores" };
pub const front_inputs = [_][*:0]const u8{ "sel", "cands", "emb", "nw", "eps" };
pub const front_outputs = [_][*:0]const u8{ "id", "en" };

/// Why a call keeps the unfused chain.
pub const Reason = enum {
    no_metal,
    embedding,
    norm_weight,
    width,
    input,
    rescore_rows,
    rescore_quant,
    shortlist,
    vocab,
    logits,
    mask,

    pub fn text(r: Reason) []const u8 {
        return switch (r) {
            .no_metal => "not a Metal GPU stream",
            .embedding => "the embedding is not a dense bf16/f16/f32 table read as is (quantized, scaled, normed, rotated or grouped)",
            .norm_weight => "pre_norm_emb is not a bf16 [hidden] weight",
            .width => "hidden is outside the 640-thread norm (1..2560)",
            .input => "the draft vector is not a bf16/f16 [1,1,hidden] row",
            .rescore_rows => "no quantized re-score rows (a dense trunk head)",
            .rescore_quant => "the re-score rows are outside MLX qmv_fast's envelope (bits, group, dtype or shape)",
            .shortlist => "the shortlist geometry or its partial launch",
            .vocab => "the vocab is wider than the embedding table",
            .logits => "the logits are not one float [1,1,V] row",
            .mask => "the suppress mask is not a bool [V] row",
        };
    }
};

var declined_said: [@typeInfo(Reason).@"enum".field_names.len]bool = @splat(false);
/// Calls that fell back (tests read it).
pub var calls_declined: usize = 0;

/// Log a fallback, once per reason.
pub fn decline(r: Reason) void {
    calls_declined += 1;
    const i = @intFromEnum(r);
    if (declined_said[i]) return;
    declined_said[i] = true;
    log.info("[draft-fuse] declined: {s} (the unfused chain serves the call)\n", .{r.text()});
}

/// Picks served per path, and next-step fronts read from a pick (tests read them).
pub const Path = enum { rerank, exact };
pub var picks_served: [2]usize = .{ 0, 0 };
pub var fronts_used: usize = 0;
var engaged_said: [2]bool = .{ false, false };

/// A forward took its front from the previous pick.
pub fn noteFront() void {
    fronts_used += 1;
}

var scores_kernel: ?mlx.mlx_fast_metal_kernel = null;
var front_kernel: ?mlx.mlx_fast_metal_kernel = null;

const Which = enum { scores, front };

fn kernelFor(which: Which) !mlx.mlx_fast_metal_kernel {
    const slot = switch (which) {
        .scores => &scores_kernel,
        .front => &front_kernel,
    };
    if (slot.*) |k| return k;
    const ins: []const [*:0]const u8 = if (which == .scores) &scores_inputs else &front_inputs;
    const outs: []const [*:0]const u8 = if (which == .scores) &scores_outputs else &front_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = if (which == .scores)
        mlx.mlx_fast_metal_kernel_new("msv_draft_fuse_scores", in_vec, out_vec, SCORES_SOURCE, "", true, false)
    else
        mlx.mlx_fast_metal_kernel_new("msv_draft_fuse_front", in_vec, out_vec, FRONT_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The next step's front, as `embedding` + `rmsNorm` build it for this config.
pub const Front = struct {
    /// The dense `[V, hidden]` embedding table (a row gather, cast to bfloat16).
    emb: mlx.mlx_array,
    /// pre_norm_emb `[hidden]`.
    norm_w: mlx.mlx_array,
    /// The config's rms_norm_eps, the float `rmsNorm` hands mlx_fast_rms_norm.
    eps: f32,
    hidden: c_int,
};

/// One pick: the draft id (`[1,1]` in the chain's dtype) and its normalized embedding (`[1,1,hidden]` bf16).
pub const Pick = struct { id: mlx.mlx_array, en: mlx.mlx_array };

/// What keeps the front out of the kernel, or null.
pub fn frontReason(f: Front) ?Reason {
    if (f.hidden < 1 or f.hidden > N_READS * TGN) return .width;
    if (f.emb.ctx == null) return .embedding;
    const esh = mlx.getShape(f.emb);
    if (esh.len != 2 or esh[0] < 1 or esh[1] != f.hidden) return .embedding;
    const ed = mlx.mlx_array_dtype(f.emb);
    if (ed != .bfloat16 and ed != .float16 and ed != .float32) return .embedding;
    if (f.norm_w.ctx == null) return .norm_weight;
    const nsh = mlx.getShape(f.norm_w);
    if (nsh.len != 1 or nsh[0] != f.hidden or mlx.mlx_array_dtype(f.norm_w) != .bfloat16) return .norm_weight;
    return null;
}

/// Whether a forward can take `en` as its front: one row of one sequence, `[1,1,hidden]` bf16 (what a pick returns).
pub fn frontEnFits(en: mlx.mlx_array, batch: c_int, seq_len: c_int, hidden: c_int) bool {
    if (en.ctx == null or batch != 1 or seq_len != 1) return false;
    if (mlx.mlx_array_dtype(en) != .bfloat16) return false;
    return std.mem.eql(c_int, mlx.getShape(en), &.{ 1, 1, hidden });
}

/// The re-score's rows: the packed affine rows a quantized_matmul over the gathered shortlist would read.
pub const Rescore = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32 };

/// Values one qmv_rows / qmv_fast thread holds per block, by bits.
pub fn valuesPerThread(bits: u32) c_int {
    return switch (bits) {
        3, 5 => 16,
        6 => 8,
        else => @intCast((32 / bits) * 2),
    };
}

/// What keeps the re-score out of the kernel, or null: x a bf16/f16 `[1,1,hidden]` row, the rows in MLX qmv_fast's
/// envelope (the kernel a one-row quantized_matmul at these shapes takes), at least `rows` of them.
pub fn rescoreReason(x: mlx.mlx_array, q: Rescore, hidden: c_int, rows: c_int) ?Reason {
    if (x.ctx == null or !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 1, hidden })) return .input;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return .input;
    if (q.bits < 3 or q.bits > 8 or q.bits == 7) return .rescore_quant;
    if (q.w.ctx == null or q.sc.ctx == null or q.bi.ctx == null) return .rescore_quant;
    if (mlx.mlx_array_dtype(q.w) != .uint32 or mlx.mlx_array_dtype(q.sc) != xd or mlx.mlx_array_dtype(q.bi) != xd) return .rescore_quant;
    const vpt = valuesPerThread(q.bits);
    const block = vpt * 32;
    const gs: c_int = @intCast(q.group_size);
    if (@rem(hidden, block) != 0) return .rescore_quant;
    if (gs < vpt or @rem(gs, vpt) != 0 or @rem(block, gs) != 0 or @rem(hidden, gs) != 0) return .rescore_quant;
    const wsh = mlx.getShape(q.w);
    if (wsh.len != 2 or wsh[0] < rows or @as(i64, wsh[1]) * 32 != @as(i64, hidden) * @as(i64, q.bits)) return .rescore_quant;
    for ([_]mlx.mlx_array{ q.sc, q.bi }) |a| {
        if (!std.mem.eql(c_int, mlx.getShape(a), &.{ wsh[0], @divExact(hidden, gs) })) return .rescore_quant;
    }
    return null;
}

/// What keeps an exact pick out of the kernel, or null: one float `[1,1,V]` row no wider than the table, a bool `[V]`
/// mask.
pub fn exactReason(logits: mlx.mlx_array, mask: ?mlx.mlx_array, f: Front) ?Reason {
    if (logits.ctx == null) return .logits;
    const lsh = mlx.getShape(logits);
    if (lsh.len != 3 or lsh[0] != 1 or lsh[1] != 1 or lsh[2] < 1) return .logits;
    const ld = mlx.mlx_array_dtype(logits);
    if (ld != .bfloat16 and ld != .float16 and ld != .float32) return .logits;
    if (lsh[2] > mlx.getShape(f.emb)[0]) return .vocab;
    if (mask) |m| {
        if (m.ctx == null or mlx.mlx_array_dtype(m) != .bool_ or !std.mem.eql(c_int, mlx.getShape(m), &.{lsh[2]})) return .mask;
    }
    return null;
}

/// The 0-d f32 eps operand and its value (the float MLX's rms_norm binds), a per-process cache.
var rms_eps: struct { arr: ?mlx.mlx_array = null, val: f32 = 0 } = .{};

fn epsArray(v: f32) mlx.mlx_array {
    if (rms_eps.arr == null or rms_eps.val != v) {
        if (rms_eps.arr) |e| _ = mlx.mlx_array_free(e);
        rms_eps = .{ .arr = mlx.mlx_array_new_float(v), .val = v };
    }
    return rms_eps.arr.?;
}

/// Eight unread ids for HAS_CANDS 0 (eight, so MLX binds them in device memory like real candidates).
var no_cands_placeholder: ?mlx.mlx_array = null;

fn noCands() mlx.mlx_array {
    if (no_cands_placeholder == null) {
        const z: [8]u32 = @splat(0);
        no_cands_placeholder = mlx.mlx_array_new_data(&z, &[_]c_int{8}, 1, .uint32);
    }
    return no_cands_placeholder.?;
}

/// Apply `which` and take its two outputs.
fn launch2(which: Which, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const k = try kernelFor(which);
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, in_vec, cfg, s));
    var o: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (o) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try mlx.check(mlx.mlx_vector_array_get(&o[0], outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&o[1], outs, 1));
    return o;
}

/// The front launch's config: id `[1,1]` id_dtype and en `[1,1,hidden]` bf16.
pub fn frontConfig(hidden: c_int, id_dtype: mlx.mlx_dtype, has_cands: bool) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1 }, 2, id_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, hidden }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TGN, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TGN, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "NT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "ID_T", id_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HAS_CANDS", @intFromBool(has_cands)));
    return cfg;
}

/// The scores launch's config: cands `[32]` uint32 and scores `[1,1,32]` in x's dtype.
pub fn scoresConfig(xd: mlx.mlx_dtype, hidden: c_int, bits: u32, group_size: u32) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{32}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 32 }, 3, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, SCORES_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, SCORES_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
    return cfg;
}

/// The front launch: id `[1,1]` id_dtype and en `[1,1,hidden]` bf16 from the argmax `sel` (through `cands` when given).
pub fn front(s: mlx.mlx_stream, sel: mlx.mlx_array, cands: ?mlx.mlx_array, f: Front, id_dtype: mlx.mlx_dtype) !Pick {
    const cfg = try frontConfig(f.hidden, id_dtype, cands != null);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const o = try launch2(.front, cfg, &.{ sel, cands orelse noCands(), f.emb, f.norm_w, epsArray(f.eps) }, s);
    return .{ .id = o[0], .en = o[1] };
}

/// The scores launch: draftTop32's ids `[32]` uint32 and their re-scores `[1,1,32]` (the 32-row quantized_matmul's).
pub fn scores(s: mlx.mlx_stream, cand_ord: mlx.mlx_array, cand_idx: mlx.mlx_array, x: mlx.mlx_array, q: Rescore, hidden: c_int) ![2]mlx.mlx_array {
    const cfg = try scoresConfig(mlx.mlx_array_dtype(x), hidden, q.bits, q.group_size);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    return launch2(.scores, cfg, &.{ cand_ord, cand_idx, x, q.w, q.sc, q.bi }, s);
}

/// The rerank pick over the partial launch's candidates (`mtp.draftTop32Cands`), the caller having checked
/// `frontReason` and `rescoreReason`: the scores launch, MLX's argmax over its values (shortlistArgmax's), the front
/// launch (shortlistArgmax's take + astype to id_dtype, int32 on the chain, and the next step's front).
pub fn pickRerank(s: mlx.mlx_stream, cand_ord: mlx.mlx_array, cand_idx: mlx.mlx_array, x: mlx.mlx_array, q: Rescore, f: Front, id_dtype: mlx.mlx_dtype) !Pick {
    const cs = try scores(s, cand_ord, cand_idx, x, q, f.hidden);
    defer for (cs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, cs[1], -1, false, s));
    const p = try front(s, amax, cs[0], f, id_dtype);
    picks_served[@intFromEnum(Path.rerank)] += 1;
    if (!engaged_said[0]) {
        engaged_said[0] = true;
        log.info("[draft-fuse] engaged: the rerank pick in three launches ({d}-bit/gs{d} re-score, hidden {d})\n", .{ q.bits, q.group_size, f.hidden });
    }
    return p;
}

/// The exact pick over `logits` `[1,1,V]`, the caller having checked `frontReason`: the chain's own where (with
/// `mask`), argmax and (id_dtype int32: maskAndArgmax's) astype give the id, then the front launch builds the next
/// step's front from the argmax. id_dtype is the chain's: uint32 (sampleTokenLazy) or int32 (maskAndArgmax).
pub fn pickExact(s: mlx.mlx_stream, logits: mlx.mlx_array, mask: ?mlx.mlx_array, f: Front, id_dtype: mlx.mlx_dtype) !Pick {
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
    } else try mlx.check(mlx.mlx_array_set(&masked, logits));
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, masked, -1, false, s));
    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    if (id_dtype == .uint32) try mlx.check(mlx.mlx_array_set(&id, am)) else try mlx.check(mlx.mlx_astype(&id, am, id_dtype, s));
    const fr = try front(s, am, null, f, .uint32);
    _ = mlx.mlx_array_free(fr.id);
    picks_served[@intFromEnum(Path.exact)] += 1;
    if (!engaged_said[1]) {
        engaged_said[1] = true;
        log.info("[draft-fuse] engaged: the exact pick with the next step's front in one launch (vocab {d}, hidden {d})\n", .{ mlx.getShape(logits)[2], f.hidden });
    }
    return .{ .id = id, .en = fr.en };
}

/// `pickExact` behind `exactReason`: null (logged) when the row is outside the envelope.
pub fn pickExactChecked(s: mlx.mlx_stream, logits: mlx.mlx_array, mask: ?mlx.mlx_array, f: Front, id_dtype: mlx.mlx_dtype) !?Pick {
    if (exactReason(logits, mask, f)) |r| {
        decline(r);
        return null;
    }
    return try pickExact(s, logits, mask, f, id_dtype);
}

// ── tests ──
//
// The GPU byte tests hold both picks to the MLX chains they replace, ties and NaN placements included.

const testing = std.testing;
const mtp = @import("mtp.zig");

/// Small host arrays for the envelope tests (no evaluation, no GPU).
fn hostArr(comptime E: type, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(E, n);
    defer testing.allocator.free(buf);
    @memset(buf, std.mem.zeroes(E));
    return mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), dt);
}

test "draft fuse declined: the envelope names each reason, and a declined exact call launches nothing" {
    const D: c_int = 2560;
    const emb = try hostArr(u16, &.{ 64, D }, .bfloat16);
    defer _ = mlx.mlx_array_free(emb);
    const emb_q = try hostArr(u32, &.{ 64, 320 }, .uint32);
    defer _ = mlx.mlx_array_free(emb_q);
    const nw = try hostArr(u16, &.{D}, .bfloat16);
    defer _ = mlx.mlx_array_free(nw);
    const nw32 = try hostArr(f32, &.{D}, .float32);
    defer _ = mlx.mlx_array_free(nw32);
    const f: Front = .{ .emb = emb, .norm_w = nw, .eps = 1e-6, .hidden = D };
    try testing.expectEqual(@as(?Reason, null), frontReason(f));
    try testing.expectEqual(@as(?Reason, .width), frontReason(.{ .emb = emb, .norm_w = nw, .eps = 1e-6, .hidden = 4096 }));
    try testing.expectEqual(@as(?Reason, .embedding), frontReason(.{ .emb = emb_q, .norm_w = nw, .eps = 1e-6, .hidden = D }));
    try testing.expectEqual(@as(?Reason, .norm_weight), frontReason(.{ .emb = emb, .norm_w = nw32, .eps = 1e-6, .hidden = D }));

    // the re-score: 8-bit / gs64 rows of the copy
    const x = try hostArr(u16, &.{ 1, 1, D }, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const x2 = try hostArr(u16, &.{ 1, 2, D }, .bfloat16);
    defer _ = mlx.mlx_array_free(x2);
    const w8 = try hostArr(u32, &.{ 64, @divExact(D * 8, 32) }, .uint32);
    defer _ = mlx.mlx_array_free(w8);
    const sc = try hostArr(u16, &.{ 64, @divExact(D, 64) }, .bfloat16);
    defer _ = mlx.mlx_array_free(sc);
    const sc16 = try hostArr(u16, &.{ 64, @divExact(D, 64) }, .float16);
    defer _ = mlx.mlx_array_free(sc16);
    const q: Rescore = .{ .w = w8, .sc = sc, .bi = sc, .bits = 8, .group_size = 64 };
    try testing.expectEqual(@as(?Reason, null), rescoreReason(x, q, D, 64));
    try testing.expectEqual(@as(?Reason, .input), rescoreReason(x2, q, D, 64));
    try testing.expectEqual(@as(?Reason, .rescore_quant), rescoreReason(x, .{ .w = w8, .sc = sc16, .bi = sc16, .bits = 8, .group_size = 64 }, D, 64));
    try testing.expectEqual(@as(?Reason, .rescore_quant), rescoreReason(x, .{ .w = w8, .sc = sc, .bi = sc, .bits = 7, .group_size = 64 }, D, 64));
    // the packed width is 8-bit's
    try testing.expectEqual(@as(?Reason, .rescore_quant), rescoreReason(x, .{ .w = w8, .sc = sc, .bi = sc, .bits = 4, .group_size = 64 }, D, 64));
    // fewer rows than the shortlist spans
    try testing.expectEqual(@as(?Reason, .rescore_quant), rescoreReason(x, q, D, 65));
    try testing.expectEqual(@as(c_int, 8), valuesPerThread(8));
    try testing.expectEqual(@as(c_int, 16), valuesPerThread(3));

    // the exact pick: one [1,1,V] row, a bool [V] mask, V within the table
    const lg = try hostArr(u16, &.{ 1, 1, 64 }, .bfloat16);
    defer _ = mlx.mlx_array_free(lg);
    const lg2 = try hostArr(u16, &.{ 1, 2, 64 }, .bfloat16);
    defer _ = mlx.mlx_array_free(lg2);
    const lg_wide = try hostArr(u16, &.{ 1, 1, 65 }, .bfloat16);
    defer _ = mlx.mlx_array_free(lg_wide);
    const mk = try hostArr(bool, &.{64}, .bool_);
    defer _ = mlx.mlx_array_free(mk);
    const mk_short = try hostArr(bool, &.{63}, .bool_);
    defer _ = mlx.mlx_array_free(mk_short);
    try testing.expectEqual(@as(?Reason, null), exactReason(lg, mk, f));
    try testing.expectEqual(@as(?Reason, null), exactReason(lg, null, f));
    try testing.expectEqual(@as(?Reason, .logits), exactReason(lg2, null, f));
    try testing.expectEqual(@as(?Reason, .vocab), exactReason(lg_wide, null, f));
    try testing.expectEqual(@as(?Reason, .mask), exactReason(lg, mk_short, f));

    // a declined call logs, counts and returns null before any kernel is built or launched
    const before_picks = picks_served;
    const before_declines = calls_declined;
    const got = try pickExactChecked(mlx.gpuStream(), lg2, null, f, .uint32);
    try testing.expect(got == null);
    try testing.expectEqual(before_picks, picks_served);
    try testing.expectEqual(before_declines + 1, calls_declined);

    // a forward takes a pick's front only for one row of one sequence, [1,1,hidden] bf16
    const en = try hostArr(u16, &.{ 1, 1, D }, .bfloat16);
    defer _ = mlx.mlx_array_free(en);
    try testing.expect(frontEnFits(en, 1, 1, D));
    try testing.expect(!frontEnFits(en, 1, 2, D));
    try testing.expect(!frontEnFits(en, 2, 1, D));
    try testing.expect(!frontEnFits(en, 1, 1, 2048));
    try testing.expect(!frontEnFits(x2, 1, 1, D));
}

fn gpuOn() bool {
    return !mlx.noGpuBackend() and !mlx.virtualGpu() and mlx.streamIsGpu(mlx.gpuStream()) and !mlx.cudaAvailable();
}

fn tRandn(shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, scale, key, s));
    return o;
}

fn tReshape(a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, a, shape.ptr, shape.len, s));
    return o;
}

fn tAstype(a: mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, a, dt, s));
    return o;
}

fn tTake(a: mlx.mlx_array, idx: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_take_axis(&o, a, idx, 0, s));
    return o;
}

/// Same dtype, shape and 16-bit patterns in element order (array_equal alone takes -0 for +0).
fn bitEqual16(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    if (mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b) or !std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return false;
    const n: c_int = @intCast(mlx.mlx_array_size(a));
    var vs: [2]mlx.mlx_array = undefined;
    for ([_]mlx.mlx_array{ a, b }, &vs) |src, *v| {
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, src, false, s));
        const flat = try tReshape(c, &.{n}, s);
        defer _ = mlx.mlx_array_free(flat);
        v.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(v, flat, .uint16, s));
    }
    defer for (vs) |v| {
        _ = mlx.mlx_array_free(v);
    };
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_array_equal(&e, vs[0], vs[1], false, s));
    try mlx.check(mlx.mlx_array_eval(e));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, e));
    return same;
}

/// An id array's value (any integer dtype, one element).
fn idValue(a: mlx.mlx_array, s: mlx.mlx_stream) !i64 {
    const c = try tAstype(a, .int32, s);
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, c));
    return v;
}

/// The chain's next-step front for draft `id`: the embedding row (reshape to [B*S], take, astype to bfloat16, reshape
/// to [B,S,H]) and rmsNorm (mlx_fast_rms_norm with pre_norm_emb and the config's eps).
fn refFront(emb: mlx.mlx_array, id: mlx.mlx_array, nw: mlx.mlx_array, eps: f32, hidden: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const ids2 = try tReshape(id, &.{ 1, 1 }, s);
    defer _ = mlx.mlx_array_free(ids2);
    const flat = try tReshape(ids2, &.{1}, s);
    defer _ = mlx.mlx_array_free(flat);
    const rows = try tTake(emb, flat, s);
    defer _ = mlx.mlx_array_free(rows);
    const e = try tAstype(rows, .bfloat16, s);
    defer _ = mlx.mlx_array_free(e);
    const e3 = try tReshape(e, &.{ 1, 1, hidden }, s);
    defer _ = mlx.mlx_array_free(e3);
    var en = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(en);
    try mlx.check(mlx.mlx_fast_rms_norm(&en, e3, nw, eps, s));
    return en;
}

fn expectPickEqual(got: Pick, want_id: mlx.mlx_array, want_en: mlx.mlx_array, s: mlx.mlx_stream) !void {
    try testing.expectEqual(mlx.mlx_array_dtype(want_id), mlx.mlx_array_dtype(got.id));
    try testing.expectEqualSlices(c_int, mlx.getShape(want_id), mlx.getShape(got.id));
    try testing.expectEqual(try idValue(want_id, s), try idValue(got.id, s));
    try testing.expect(try bitEqual16(got.en, want_en, s));
}

/// 8-bit (or `bits`)/gs64 affine rows of `w_bf` as a re-score source (owned arrays).
fn quantRows(w_bf: mlx.mlx_array, bits: u32, s: mlx.mlx_stream) !Rescore {
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w_bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    var q: Rescore = .{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .bi = mlx.mlx_array_new(), .bits = bits, .group_size = 64 };
    errdefer freeRescore(q);
    try mlx.check(mlx.mlx_vector_array_get(&q.w, qv, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q.sc, qv, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q.bi, qv, 2));
    return q;
}

fn freeRescore(q: Rescore) void {
    _ = mlx.mlx_array_free(q.w);
    _ = mlx.mlx_array_free(q.sc);
    _ = mlx.mlx_array_free(q.bi);
}

/// The chain's rerank pick over coarse row `flat`: draftTop32, the three gathers, the 32-row quantized_matmul,
/// shortlistArgmax. Returns the id and the re-scores (owned).
fn chainRerank(flat: mlx.mlx_array, rows: c_int, x: mlx.mlx_array, q: Rescore, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const cands = try mtp.draftTop32(s, flat, rows);
    defer _ = mlx.mlx_array_free(cands);
    const w32 = try tTake(q.w, cands, s);
    defer _ = mlx.mlx_array_free(w32);
    const s32 = try tTake(q.sc, cands, s);
    defer _ = mlx.mlx_array_free(s32);
    const b32 = try tTake(q.bi, cands, s);
    defer _ = mlx.mlx_array_free(b32);
    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(&exact, x, w32, s32, b32, true, mlx.mlx_optional_int.some(@intCast(q.group_size)), mlx.mlx_optional_int.some(@intCast(q.bits)), "affine", s));
    const id = try mtp.shortlistArgmax(s, .{ .cands = cands, .exact = exact, .rows = rows });
    return .{ id, exact };
}

test "draft fuse GPU: the rerank pick is shortlistArgmax's id and the next step's front, bit for bit (random rows, planted ties)" {
    if (!gpuOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const D: c_int = 2560;
    const rows: c_int = mtp.TOP32_MIN_ROWS + 96;
    const eps: f32 = 1e-6;
    const nw = try tRandn(&.{D}, .bfloat16, 0xD5F0, 1.0, s);
    defer _ = mlx.mlx_array_free(nw);
    const x = try tRandn(&.{ 1, 1, D }, .bfloat16, 0xD5F1, 1.0, s);
    defer _ = mlx.mlx_array_free(x);
    const w_base = try tRandn(&.{ rows, D }, .bfloat16, 0xD5F2, 0.02, s);
    defer _ = mlx.mlx_array_free(w_base);
    // the "hot" row: x itself, the largest re-score by far; planted rows that copy it tie exactly
    const hot = try tReshape(x, &.{ 1, D }, s);
    defer _ = mlx.mlx_array_free(hot);
    var w_cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cat);
    {
        const parts = [_]mlx.mlx_array{ w_base, hot };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&w_cat, vec, 0, s));
    }
    // planted shortlist members (coarse values 10 + k: above every random value, so all in the top 32, in this order)
    const planted = [_]u32{ 7, 389, 4103, 9001, 12345, 16470 };
    const flat_host = try testing.allocator.alloc(f32, @intCast(rows));
    defer testing.allocator.free(flat_host);
    var prng = std.Random.DefaultPrng.init(0xD5F3);
    for (flat_host) |*v| v.* = prng.random().floatNorm(f32);
    for (planted, 0..) |p, k| flat_host[p] = 10.0 + @as(f32, @floatFromInt(k));
    const flat = mlx.mlx_array_new_data(flat_host.ptr, &[_]c_int{rows}, 1, .float32);
    defer _ = mlx.mlx_array_free(flat);

    const map_host = try testing.allocator.alloc(i32, @intCast(rows));
    defer testing.allocator.free(map_host);
    for ([_]bool{ false, true }) |ties| {
        for (map_host, 0..) |*m, i| m.* = @intCast(i);
        // ties: planted members 0, 2 and 4 all re-score as the hot row; shortlistArgmax takes the first in shortlist
        // order
        if (ties) {
            for ([_]usize{ 0, 2, 4 }) |k| map_host[planted[k]] = rows;
        }
        const map = mlx.mlx_array_new_data(map_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(map);
        const w_bf = try tTake(w_cat, map, s);
        defer _ = mlx.mlx_array_free(w_bf);
        for ([_]u32{ 8, 4, 6 }) |bits| {
            const q = try quantRows(w_bf, bits, s);
            defer freeRescore(q);
            for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |edt| {
                const emb = try tRandn(&.{ rows, D }, edt, 0xD5F4, 0.5, s);
                defer _ = mlx.mlx_array_free(emb);
                const f: Front = .{ .emb = emb, .norm_w = nw, .eps = eps, .hidden = D };
                try testing.expectEqual(@as(?Reason, null), frontReason(f));
                try testing.expectEqual(@as(?Reason, null), rescoreReason(x, q, D, rows));
                const want = try chainRerank(flat, rows, x, q, s);
                defer for (want) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const want_en = try refFront(emb, want[0], nw, eps, D, s);
                defer _ = mlx.mlx_array_free(want_en);
                const parts = try mtp.draftTop32Cands(s, flat, rows);
                defer for (parts) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const got = try pickRerank(s, parts[0], parts[1], x, q, f, .int32);
                defer {
                    _ = mlx.mlx_array_free(got.id);
                    _ = mlx.mlx_array_free(got.en);
                }
                if (ties) {
                    // the chain really saw a tie at the top of the re-score
                    const ef = try tAstype(want[1], .float32, s);
                    defer _ = mlx.mlx_array_free(ef);
                    try mlx.check(mlx.mlx_array_eval(ef));
                    const vals = mlx.mlx_array_data_float32(ef).?[0..32];
                    var top: f32 = -std.math.inf(f32);
                    for (vals) |v| top = @max(top, v);
                    var n_top: usize = 0;
                    for (vals) |v| n_top += @intFromBool(v == top);
                    try testing.expect(n_top >= 3);
                }
                try expectPickEqual(got, want[0], want_en, s);
                if (ties) try testing.expectEqual(@as(i64, planted[0]), try idValue(got.id, s));
            }
        }
    }
}

fn named(name: []const u8, want: []const u8) bool {
    return std.mem.eql(u8, name, want);
}

/// The chain's exact pick: `sampleTokenLazy`'s greedy fast path (where with a mask, argmax: [1,1] uint32) or
/// `maskAndArgmax` (the same, astype int32).
fn chainExact(logits: mlx.mlx_array, mask: ?mlx.mlx_array, id_dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    if (id_dtype == .int32) return mtp.maskAndArgmax(s, logits, mask);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
    } else try mlx.check(mlx.mlx_array_set(&masked, logits));
    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    try mlx.check(mlx.mlx_argmax_axis(&id, masked, -1, false, s));
    return id;
}

test "draft fuse GPU: the exact pick is sampleTokenLazy's and maskAndArgmax's id and the next step's front, bit for bit (ties, NaN, masked)" {
    if (!gpuOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const D: c_int = 2560;
    const V: c_int = 20003; // not a multiple of the scan's 2560-wide pass: the last pass is ragged
    const eps: f32 = 1e-6;
    const nw = try tRandn(&.{D}, .bfloat16, 0xE5F0, 1.0, s);
    defer _ = mlx.mlx_array_free(nw);
    const emb = try tRandn(&.{ V, D }, .bfloat16, 0xE5F1, 0.5, s);
    defer _ = mlx.mlx_array_free(emb);
    const f: Front = .{ .emb = emb, .norm_w = nw, .eps = eps, .hidden = D };
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    const neg_nan: f32 = @bitCast(@as(u32, 0xFFC00000));

    const Case = struct { name: []const u8, masked: bool, dt: mlx.mlx_dtype };
    const cases = [_]Case{
        .{ .name = "random", .masked = false, .dt = .bfloat16 },
        .{ .name = "random+mask", .masked = true, .dt = .bfloat16 },
        .{ .name = "ties", .masked = false, .dt = .bfloat16 },
        .{ .name = "ties+mask-first", .masked = true, .dt = .bfloat16 },
        .{ .name = "nan", .masked = false, .dt = .bfloat16 },
        .{ .name = "neg-nan", .masked = false, .dt = .bfloat16 },
        .{ .name = "all-nan", .masked = false, .dt = .bfloat16 },
        .{ .name = "nan+posinf", .masked = false, .dt = .bfloat16 },
        .{ .name = "nan-masked", .masked = true, .dt = .bfloat16 },
        .{ .name = "all-inf+nan", .masked = false, .dt = .bfloat16 },
        .{ .name = "all-masked", .masked = true, .dt = .bfloat16 },
        .{ .name = "ties-f32", .masked = false, .dt = .float32 },
        .{ .name = "nan-f32", .masked = false, .dt = .float32 },
        .{ .name = "neg-zero-ties", .masked = false, .dt = .bfloat16 },
    };
    const lh = try testing.allocator.alloc(f32, @intCast(V));
    defer testing.allocator.free(lh);
    const mh = try testing.allocator.alloc(bool, @intCast(V));
    defer testing.allocator.free(mh);
    var prng = std.Random.DefaultPrng.init(0xE5F2);
    for (cases) |c| {
        for (lh) |*v| v.* = prng.random().floatNorm(f32) * 3.0;
        for (mh, 0..) |*m, i| m.* = (i % 10) == 3;
        if (std.mem.startsWith(u8, c.name, "ties")) {
            // equal maxima inside one thread's 4-read chunk, across simdgroups and in the ragged last pass
            for ([_]usize{ 5, 8, 9, 2561, 19999, 20002 }) |i| {
                lh[i] = 50.0;
                mh[i] = false;
            }
            if (c.masked) mh[5] = true; // the first maximum masked: the next one is the pick
        }
        // NaN before, after and at the largest number; negative NaN; all NaN; NaN beside +inf; masked NaN
        if (named(c.name, "nan") or named(c.name, "nan-f32") or named(c.name, "nan-masked")) {
            lh[0] = nan;
            lh[3] = nan;
            lh[777] = 40.0;
            lh[12000] = nan;
            if (c.masked) {
                mh[0] = true;
                mh[3] = true;
            }
        }
        if (named(c.name, "neg-nan")) {
            lh[1] = neg_nan;
            lh[4097] = neg_nan;
            lh[300] = 40.0;
        }
        if (named(c.name, "all-nan")) @memset(lh, nan);
        if (named(c.name, "nan+posinf")) {
            lh[10] = nan;
            lh[11] = nan;
            lh[4000] = inf;
        }
        if (named(c.name, "all-inf+nan")) {
            for (lh) |*v| v.* = -inf;
            lh[2] = nan;
        }
        if (named(c.name, "all-masked")) @memset(mh, true);
        if (named(c.name, "neg-zero-ties")) {
            for (lh) |*v| v.* = -1.0 - @abs(v.*);
            lh[900] = -0.0;
            lh[901] = 0.0;
            lh[3000] = -0.0;
        }
        const l32 = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{ 1, 1, V }, 3, .float32);
        defer _ = mlx.mlx_array_free(l32);
        const logits = try tAstype(l32, c.dt, s);
        defer _ = mlx.mlx_array_free(logits);
        const mask_arr = mlx.mlx_array_new_data(mh.ptr, &[_]c_int{V}, 1, .bool_);
        defer _ = mlx.mlx_array_free(mask_arr);
        const mask: ?mlx.mlx_array = if (c.masked) mask_arr else null;
        try testing.expectEqual(@as(?Reason, null), exactReason(logits, mask, f));
        for ([_]mlx.mlx_dtype{ .uint32, .int32 }) |idt| {
            const want_id = try chainExact(logits, mask, idt, s);
            defer _ = mlx.mlx_array_free(want_id);
            const want_en = try refFront(emb, want_id, nw, eps, D, s);
            defer _ = mlx.mlx_array_free(want_en);
            const got = (try pickExactChecked(s, logits, mask, f, idt)).?;
            defer {
                _ = mlx.mlx_array_free(got.id);
                _ = mlx.mlx_array_free(got.en);
            }
            try expectPickEqual(got, want_id, want_en, s);
            if (idt == .int32 and named(c.name, "ties")) try testing.expectEqual(@as(i64, 5), try idValue(got.id, s));
            if (idt == .int32 and named(c.name, "ties+mask-first")) try testing.expectEqual(@as(i64, 8), try idValue(got.id, s));
            // all -inf with no NaN: index 0 under any rule (a NaN row's index is MLX's; the byte comparison holds it)
            if (idt == .int32 and named(c.name, "all-masked")) try testing.expectEqual(@as(i64, 0), try idValue(got.id, s));
        }
    }
}

test "draft fuse GPU: a NaN or infinite mixer value re-scores to NaN, and the rerank pick still takes the chain's argmax" {
    if (!gpuOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const D: c_int = 2560;
    const rows: c_int = mtp.TOP32_MIN_ROWS + 96;
    const eps: f32 = 1e-6;
    const nw = try tRandn(&.{D}, .bfloat16, 0xF5F0, 1.0, s);
    defer _ = mlx.mlx_array_free(nw);
    var prng = std.Random.DefaultPrng.init(0xF5F2);
    const flat_host = try testing.allocator.alloc(f32, @intCast(rows));
    defer testing.allocator.free(flat_host);
    for (flat_host) |*v| v.* = prng.random().floatNorm(f32);
    const flat = mlx.mlx_array_new_data(flat_host.ptr, &[_]c_int{rows}, 1, .float32);
    defer _ = mlx.mlx_array_free(flat);
    const w_bf = try tRandn(&.{ rows, D }, .bfloat16, 0xF5F3, 0.02, s);
    defer _ = mlx.mlx_array_free(w_bf);
    const q = try quantRows(w_bf, 8, s);
    defer freeRescore(q);
    const remb = try tRandn(&.{ rows, D }, .bfloat16, 0xF5F4, 0.5, s);
    defer _ = mlx.mlx_array_free(remb);
    const rf: Front = .{ .emb = remb, .norm_w = nw, .eps = eps, .hidden = D };
    const xh = try testing.allocator.alloc(f32, @intCast(D));
    defer testing.allocator.free(xh);
    const neg_nan: f32 = @bitCast(@as(u32, 0xFFC00000));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), neg_nan }) |poison| {
        for (xh) |*v| v.* = prng.random().floatNorm(f32);
        xh[100] = poison;
        const x32 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, 1, D }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        const x = try tAstype(x32, .bfloat16, s);
        defer _ = mlx.mlx_array_free(x);
        const want = try chainRerank(flat, rows, x, q, s);
        defer for (want) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const want_en = try refFront(remb, want[0], nw, eps, D, s);
        defer _ = mlx.mlx_array_free(want_en);
        const parts = try mtp.draftTop32Cands(s, flat, rows);
        defer for (parts) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const got = try pickRerank(s, parts[0], parts[1], x, q, rf, .int32);
        defer {
            _ = mlx.mlx_array_free(got.id);
            _ = mlx.mlx_array_free(got.en);
        }
        try expectPickEqual(got, want[0], want_en, s);
    }
}
