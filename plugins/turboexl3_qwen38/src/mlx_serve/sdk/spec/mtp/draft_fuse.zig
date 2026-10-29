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
