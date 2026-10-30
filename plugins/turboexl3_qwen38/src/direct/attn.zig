//! The direct model's QSA attention layer (12 of the 48) up to the attention core: q|gate, k and v through turboexl3's
//! fused entries, the q/k norm and RoPE, and the indexer (pooled block keys, the key ring, the block score sheet and the
//! top-512 select); attn_core.zig takes it from the KV append. The K path of `attention.zig`, `trunk/attn.zig`,
//! `trunk/fusion.zig`'s attention-input decisions, `qsa/{indexer,merges,pooled,history,checkpoint}.zig` and
//! the sdk launchers they reach: text on the unscaled rope over dense KV, on an M5-class GPU's Metal stream.

const std = @import("std");
const abi = @import("../mlx_serve/abi/quant.zig");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const QsaCfgCache = @import("../mlx_serve/mlx/kernel_cache.zig").QsaCfgCache;
const concatAxis0 = @import("../mlx_serve/mlx/array.zig").concatAxis0;
const materializedOwnedCopy = @import("../mlx_serve/mlx/array.zig").materializedOwnedCopy;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const Formats = @import("../mlx_serve/plugins/qwen4_exp/trunk/formats.zig").Formats;
const fusion = @import("../mlx_serve/plugins/qwen4_exp/trunk/fusion.zig");
const IdxMerge = @import("../mlx_serve/plugins/qwen4_exp/qsa/merges.zig").IdxMerge;
const PoolPlan = @import("../mlx_serve/plugins/qwen4_exp/qsa/merges.zig").PoolPlan;
const QSA_RING_ROWS = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCache;
const FullAttnWeights = @import("../mlx_serve/engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;
const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT = @import("../mlx_serve/sdk/blocks/attention/sparse_block/score.zig").QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
const Blocks = @import("../mlx_serve/sdk/blocks/attention/sparse_block/params.zig").Blocks;
const MropeCosSin = @import("../mlx_serve/sdk/blocks/rope/mrope.zig").MropeCosSin;
const attn_core = @import("attn_core.zig");
const linear = @import("linear.zig");

/// The layer's output for `x`, its KV, key ring and pooled keys advanced.
pub fn layer(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer_idx: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    // Registered above the fallible builders: `qsaMask` can throw with `ctx.qsa_blocks` already written.
    defer {
        if (ctx.qsa_mask.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_mask);
        ctx.qsa_mask = .{ .ctx = null };
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        if (ctx.batch_slots) |slots| {
            for (slots) |sc| {
                if (sc.qsa_mask.ctx != null) _ = mlx.mlx_array_free(sc.qsa_mask);
                sc.qsa_mask = .{ .ctx = null };
                if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
                sc.qsa_blocks = .{ .ctx = null };
            }
        }
        self.fused.releaseLayer();
    }
    if (ctx.mrope_pos != null or ctx.mrope_cos_cur != null) return error.DirectModelIsTextOnly;
    if (ctx.tree != null) return error.DirectModelHasNoSpecTree;
    if (self.config.rope_yarn) return error.DirectModelHasNoYarn;
    // IDXPREP: at prefill rows the indexer's prepare is the fourth member of the fused attention input's prepare launch.
    ctx.qsa_mask = (try idxPrepMask(self, ctx, x, fa, entry, cache_len, pos_base, batch, seq_len, is_prefill)) orelse
        try qsaMask(self, ctx, x, fa, entry, layer_idx, cache_len, pos_base, batch, seq_len);
    return gatedFullAttnProjected(self, ctx, x, fa, layer_idx, pos_base + cache_len, batch, seq_len, is_prefill);
}

// ── q|gate, k and v, normed and rotated ──

/// q|gate, k and v (the fused attention input, the query planes, or the projections and the split), v head-major, q
/// and k normed and rotated (the fused input's finish, the hd-256 QK-norm + RoPE kernel, or the chain); then the core.
fn gatedFullAttnProjected(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, layer_idx: u32, offset: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const rope_dims = ropeDims(self);
    const decode_shape = batch == 1 and !is_prefill;

    const fused_in: ?fusion.AttnIn = if (kvOnlySkips(self, seq_len)) null else try attnInFused(self, x, fa, offset, batch, seq_len, is_prefill);
    defer if (fused_in) |f| f.deinit();
    const q_planes: ?[2]mlx.mlx_array = if (fused_in == null) qPlanes(self, x, fa, batch, seq_len) else null;
    defer if (q_planes) |pl| for (pl) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const q_proj: mlx.mlx_array = if (fused_in != null or q_planes != null) .{ .ctx = null } else try linear.attnProj(self, x, fa.q_w, fa.q_s, fa.q_b, decode_shape, layer_idx);
    defer if (q_proj.ctx != null) {
        _ = mlx.mlx_array_free(q_proj);
    };

    var queries = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    if (fused_in) |f| {
        try mlx.check(mlx.mlx_array_set(&gate, f.gate));
    } else if (q_planes) |pl| {
        try mlx.check(mlx.mlx_array_set(&queries, pl[0]));
        try mlx.check(mlx.mlx_array_set(&gate, pl[1]));
    } else {
        // [B, S, 2 H D] as [B, S, H, 2 D], then one split into queries and the gate (mlx-lm qwen3_next). The gate stays
        // 4-D: flattening the split view would copy it.
        var q_gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_gate);
        try mlx.check(mlx.mlx_reshape(&q_gate, q_proj, &[_]c_int{ batch, seq_len, h_count, hd * 2 }, 4, self.s));
        var split = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split);
        try mlx.check(mlx.mlx_split(&split, q_gate, 2, -1, self.s));
        if (mlx.mlx_vector_array_size(split) != 2) return error.UnexpectedSplitCount;
        try mlx.check(mlx.mlx_vector_array_get(&queries, split, 0));
        try mlx.check(mlx.mlx_vector_array_get(&gate, split, 1));
    }

    const perm = [_]c_int{ 0, 2, 1, 3 };
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    if (fused_in) |f| {
        try mlx.check(mlx.mlx_array_set(&v_t, f.v));
    } else {
        const k_proj = try linear.attnProj(self, x, fa.k_w, fa.k_s, fa.k_b, decode_shape, layer_idx);
        defer _ = mlx.mlx_array_free(k_proj);
        const v_proj = try linear.attnProj(self, x, fa.v_w, fa.v_s, fa.v_b, decode_shape, layer_idx);
        defer _ = mlx.mlx_array_free(v_proj);
        const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
        try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));
    }

    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    if (fused_in) |f| {
        try mlx.check(mlx.mlx_array_set(&q_rope, f.q));
        try mlx.check(mlx.mlx_array_set(&k_rope, f.k));
    } else if (batch == 1 and seq_len <= 32) {
        // Decode and verify widths: the norms, transposes and partial RoPE of q and k in one launch, bit for bit the
        // chain below, over MLX's rope's own angles.
        const angles = try qkAngleRowsFor(self, rope_dims, offset, seq_len);
        const pair = try fusedQkNormRope256(self.s, queries, k_r, fa.q_norm, fa.k_norm, angles, self.rms_eps_arr, h_count, kv_h, seq_len, rope_dims);
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
    } else {
        const q_normed = try linear.rmsNorm(self, queries, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try linear.rmsNorm(self, k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));
        // A batched tick's slots each at their position, else the scalar offset.
        const base = mlx.mlx_optional_float{ .value = cfg.rope_theta, .has_value = true };
        const no_freqs = mlx.mlx_array{ .ctx = null };
        const positions: ?mlx.mlx_array = ctx.batch_rope_offsets;
        if (positions) |pos| {
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, base, 1.0, pos, no_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, base, 1.0, pos, no_freqs, self.s));
        } else {
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, base, 1.0, offset, no_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, base, 1.0, offset, no_freqs, self.s));
        }
    }

    const gate_src = gateSrc(if (fused_in) |f| f.gate else if (q_planes) |pl| pl[1] else null, if (q_proj.ctx != null) q_proj else null, h_count, hd);
    return attn_core.core(self, ctx, q_rope, k_rope, v_t, gate, gate_src, fa, layer_idx, batch, seq_len, is_prefill);
}

/// The rows `attn_in` and IDXPREP's prepare serve: one sequence's prefill chunk past the decode rows.
fn attnInRows(batch: c_int, seq_len: c_int, is_prefill: bool) bool {
    return batch == 1 and is_prefill and seq_len >= fusion.prefill_min_rows;
}

fn qkvOf(f: *const Formats, fa: *const FullAttnWeights) ?[3]*const abi.Weight {
    return .{ f.linearOf(fa.q_w) orelse return null, f.linearOf(fa.k_w) orelse return null, f.linearOf(fa.v_w) orelse return null };
}

/// ATTN: q|gate, k and v at prefill rows through `attn_in`, the split, the q/k RMSNorm and the partial RoPE in its
/// finish, head-major, over IDXPREP's prepared copies where this layer made them. Null: the chain.
fn attnInFused(self: *Trunk, x: mlx.mlx_array, fa: *const FullAttnWeights, offset: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !?fusion.AttnIn {
    if (!attnInRows(batch, seq_len, is_prefill)) return null;
    const ws = qkvOf(&self.formats, fa) orelse return null;
    const cfg = &self.config;
    const rope_dims = ropeDims(self);
    const angles = try qkAngleRowsFor(self, rope_dims, offset, seq_len);
    var opts: abi.AttnIn = .{
        .heads = cfg.num_attention_heads,
        .kv_heads = cfg.num_key_value_heads,
        .rope_dims = @intCast(rope_dims),
        .q_norm = fusion.arr(fa.q_norm),
        .k_norm = fusion.arr(fa.k_norm),
        .eps = fusion.arr(self.rms_eps_arr),
        .angles = fusion.arr(angles),
    };
    const prep = &self.fused.prep;
    if (prep[0].ctx != null) opts.prep = .{ fusion.arr(prep[0]), fusion.arr(prep[1]), fusion.arr(prep[2]) };
    return fusion.callAttnIn(&self.formats, ws, x, &opts, self.s);
}

/// IDXPREP: q's, k's, v's and the indexer projection's prepare in one launch (x read once); the indexer's product over
/// its copy, then the indexer, and `attnInFused` takes the other three copies. Null: `qsaMask`.
fn idxPrepMask(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !?mlx.mlx_array {
    if (!attnInRows(batch, seq_len, is_prefill)) return null;
    const f = &self.formats;
    const ws = qkvOf(f, fa) orelse return null;
    const wi = f.linearOf(fa.idx_qk_w) orelse return null;
    const xp = fusion.callAttnPrep(f, ws, wi, x, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(xp[3]);
    for (&self.fused.prep, xp[0..3]) |*dst, p| {
        if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
        dst.* = p;
    }
    // a refusal here leaves the attention input its three copies and `qsaMask` the indexer
    const qk = fusion.callLinearPreparedIdx(f, wi, xp[3], x, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(qk);
    return try qsaMaskFromQk(self, ctx, qk, fa, entry, cache_len, pos_base, batch, seq_len, true, null);
}

/// Up to these rows the hd-256 launch computes q with k anyway.
const joint_rope_rows: c_int = 32;

/// MTPQ: the MTP head's history append (`Scratch.kv_only`, whose stream is dropped) past the joint-rope widths skips
/// the fused input's q|gate; the chain's q is built and never evaluated.
fn kvOnlySkips(self: *const Trunk, rows: c_int) bool {
    return self.fused.kv_only and rows > joint_rope_rows;
}

/// QPLANES: q_proj at decode and verify rows through `q_planes`, its finish writing [query | gate] as two contiguous
/// planes [1, S, H, D], so neither the split nor the hd-256 kernel's copy of a strided query view runs. Null: q_proj.
fn qPlanes(self: *Trunk, x: mlx.mlx_array, fa: *const FullAttnWeights, batch: c_int, seq_len: c_int) ?[2]mlx.mlx_array {
    if (batch != 1 or seq_len > fusion.decode_rows_max) return null;
    const q = self.formats.linearOf(fa.q_w) orelse return null;
    const opts: abi.QPlanes = .{ .heads = self.config.num_attention_heads, .head_dim = self.config.head_dim };
    return fusion.callQPlanes(&self.formats, q, x, &opts, self.s);
}

/// Where o_proj's fused prepare reads the gate: a gate plane [1, S, H, D], else the q_proj [1, S, 2 H D] this layer
/// projected. Null: the core's tail.
fn gateSrc(plane: ?mlx.mlx_array, own_q_proj: ?mlx.mlx_array, heads: c_int, hd: c_int) ?fusion.GateSrc {
    if (plane) |g| return .{ .base = g, .row_stride = @intCast(heads * hd), .head_stride = @intCast(hd), .off = 0 };
    if (own_q_proj) |q| return .{ .base = q, .row_stride = @intCast(2 * heads * hd), .head_stride = @intCast(2 * hd), .off = @intCast(hd) };
    return null;
}

// ── RoPE ──

fn ropeDims(self: *const Trunk) c_int {
    return @intFromFloat(@as(f32, @floatFromInt(self.config.head_dim)) * self.config.partial_rotary_factor);
}

/// The f32 [rows * rd] cos|sin rows at positions offset.., kept in rope family 0's slot while the key holds (every layer
/// of a forward reads the same rows).
fn qkAngleRowsFor(self: *Trunk, rd: c_int, offset: c_int, rows: c_int) !mlx.mlx_array {
    const slot = &self.qk_angle_cache[0];
    if (slot.arr.ctx != null and slot.offset == offset and slot.rd == rd and slot.rows == rows and slot.scale == 1.0) return slot.arr;
    if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
    slot.arr = .{ .ctx = null };
    slot.arr = try ropeAngleRows(self.s, rd, .{ .value = self.config.rope_theta, .has_value = true }, offset, rows, self.allocator);
    slot.offset = offset;
    slot.rd = rd;
    slot.rows = rows;
    slot.scale = 1.0;
    return slot.arr;
}

/// A [1, 1, rows, rd] stack of ones|zeros probe rows through the stock `mlx_fast_rope`, which stamps each row at its own
/// position: the kernels rotate with its own floats, never re-derived ones.
fn ropeAngleRows(s: mlx.mlx_stream, rd: c_int, base: mlx.mlx_optional_float, offset: c_int, rows: c_int, allocator: std.mem.Allocator) !mlx.mlx_array {
    const n: usize = @intCast(rd);
    const buf = try allocator.alloc(f32, n * @as(usize, @intCast(rows)));
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const probe = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, 1, rows, rd }, 4, .float32);
    defer _ = mlx.mlx_array_free(probe);
    var rotated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rotated);
    try mlx.check(mlx.mlx_fast_rope(&rotated, probe, rd, false, base, 1.0, offset, .{ .ctx = null }, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, rotated, &[_]c_int{rows * rd}, 1, s));
    return flat;
}

/// The f32 [half] spectrum theta^(-2j/rope_dims), in-graph. Caller owns the result.
fn ropeInvFreq(self: *Trunk, rope_dims: c_int, theta: f32) !mlx.mlx_array {
    var j = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(j);
    try mlx.check(mlx.mlx_arange(&j, 0, @floatFromInt(@divExact(rope_dims, 2)), 1.0, .float32, self.s));
    const neg = mlx.mlx_array_new_float(-2.0 * @log(theta) / @as(f32, @floatFromInt(rope_dims)));
    defer _ = mlx.mlx_array_free(neg);
    var jl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(jl);
    try mlx.check(mlx.mlx_multiply(&jl, j, neg, self.s));
    var inv_freq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&inv_freq, jl, self.s));
    return inv_freq;
}

/// cos/sin [n, rope_dims] in `dt` at positions base + step*i. Takes `inv_freq`.
fn ropeCosSinFromFreqs(self: *Trunk, rope_dims: c_int, inv_freq: mlx.mlx_array, base: f32, step: f32, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
    defer _ = mlx.mlx_array_free(inv_freq);
    std.debug.assert(mlx.getShape(inv_freq)[0] == @divExact(rope_dims, 2));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, base, base + step * @as(f32, @floatFromInt(n)) - step * 0.5, step, .float32, self.s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ n, 1 }, 2, self.s));
    var ang = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ang);
    try mlx.check(mlx.mlx_multiply(&ang, pos2, inv_freq, self.s));
    const parts = [_]mlx.mlx_array{ ang, ang };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    try mlx.check(mlx.mlx_concatenate_axis(&emb, vec, 1, self.s));
    var cos_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_f);
    try mlx.check(mlx.mlx_cos(&cos_f, emb, self.s));
    var sin_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_f);
    try mlx.check(mlx.mlx_sin(&sin_f, emb, self.s));
    var cosv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cosv);
    try mlx.check(mlx.mlx_astype(&cosv, cos_f, dt, self.s));
    var sinv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sinv);
    try mlx.check(mlx.mlx_astype(&sinv, sin_f, dt, self.s));
    return .{ .cos = cosv, .sin = sinv };
}

/// A Metal kernel built once into its module `var`.
fn cachedKernel(cache: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, row_contiguous: bool) !mlx.mlx_fast_metal_kernel {
    if (cache.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, row_contiguous, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    cache.* = k;
    return k;
}

/// A launch's outputs, owned.
fn kernelOuts(comptime n: usize, k: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) ![n]mlx.mlx_array {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(out_vec) != n) return error.MetalKernelBadOutputCount;
    var outs: [n]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (outs) |o| {
        if (o.ctx != null) _ = mlx.mlx_array_free(o);
    };
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, out_vec, i));
    }
    return outs;
}

// The hd-256 QK-norm + RoPE kernel: 64 threads per (head, position) row, the RMS of `rms_single_row`, the normed row
// staged bf16 (the separate norm kernel's rounding), then the rotation against the per-position angle rows; head-major
// output, so the transposes vanish. Bit-identical to the chain.
const QK_NORM_ROPE_256_SOURCE = kernelSource("kernels/rope/qk_norm_rope_256.metal");
var qk256_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk256_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qk256_cfg_seq: c_int = 0;

/// q [1, S, HQ, 256] (a strided view is materialized) and k [1, S, HK, 256] normed and rotated by the [S * rd] angle
/// rows into [1, H, S, 256]. The kernel's v stage is Gemma's: `HV` 0.
fn fusedQkNormRope256(s: mlx.mlx_stream, q_in: mlx.mlx_array, k_in: mlx.mlx_array, q_norm_w: mlx.mlx_array, k_norm_w: mlx.mlx_array, angles: mlx.mlx_array, eps_arr: mlx.mlx_array, hq: c_int, hk: c_int, seq: c_int, rd: c_int) ![2]mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(q_in);
    if (qk256_cfg == null or qk256_cfg_seq != seq) {
        if (qk256_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        qk256_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hq, seq, 256 }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hk, seq, 256 }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, 1, 1, 256 }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk) * seq * 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HK", hk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HV", 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "S", seq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "NORM", true));
        qk256_cfg = config;
        qk256_cfg_seq = seq;
    }
    const kernel = try cachedKernel(&qk256_kernel, "mlxserve_qk_norm_rope_256", &.{ "q", "k", "v", "qw", "kw", "vw", "angles", "eps" }, &.{ "oq", "ok", "ov" }, QK_NORM_ROPE_256_SOURCE, "", true);
    // the unbound v stage never reads its slots: k stands in
    const outs = try kernelOuts(3, kernel, &.{ q_in, k_in, k_in, q_norm_w, k_norm_w, k_in, angles, eps_arr }, qk256_cfg.?, s);
    _ = mlx.mlx_array_free(outs[2]);
    return .{ outs[0], outs[1] };
}

// ── the indexer ──

/// This chunk's raw index keys appended and, past the budget, the block picks (`ctx.qsa_blocks`) or the bool
/// [B, 1, S, kv] mask; null-ctx otherwise.
fn qsaMask(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer_idx: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    if (ctx.batch_slots == null) if (try indexerMerged(self, x, fa, entry, cache_len, pos_base, batch, seq_len)) |im| {
        var idx = im;
        defer idx.deinit();
        return qsaMaskFromQk(self, ctx, idx.qk, fa, entry, cache_len, pos_base, batch, seq_len, true, &idx);
    };
    const qk = try linear.qmatmul(self, x, fa.idx_qk_w, fa.idx_qk_s, fa.idx_qk_b); // [B, S, (n + 1) hd]
    defer _ = mlx.mlx_array_free(qk);
    if (ctx.batch_slots) |slots| return qsaMaskBatched(self, slots, qk, fa, layer_idx);
    return qsaMaskFromQk(self, ctx, qk, fa, entry, cache_len, pos_base, batch, seq_len, true, null);
}

/// A batched tick: the projection ran once on [N, 1, ·]; each slot's row runs against its own history and position.
/// Where a slot has picks, the core's batched gather takes every slot's picks or mask as they are; otherwise the masks
/// are false-padded to the longest kv and stacked to [N, 1, 1, kv_max] (null-ctx: no slot past the budget).
fn qsaMaskBatched(self: *Trunk, slots: []const *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, layer_idx: u32) !mlx.mlx_array {
    const N = slots.len;
    // The per-slot ctxs have no other net: without this one throw orphaned a slot's selection.
    errdefer for (slots) |sc| {
        if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
        sc.qsa_blocks = .{ .ctx = null };
    };
    const qk_sh = mlx.getShape(qk);
    const seq_len: c_int = qk_sh[1];
    const w: c_int = qk_sh[2];
    const masks = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(masks);
    for (masks) |*m| m.* = .{ .ctx = null };
    defer for (masks) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const kv_lens = try self.allocator.alloc(c_int, N);
    defer self.allocator.free(kv_lens);
    var kv_max: c_int = 0;
    var any = false;
    var any_blocks = false;
    for (slots, 0..) |sc, i| {
        const e = if (sc.qsa_entry) |qe| qe else &sc.ssm_entries.?[layer_idx];
        const cache_len: c_int = @intCast(sc.moe_seq_offset.*);
        const i_c: c_int = @intCast(i);
        var qk_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qk_i);
        try mlx.check(mlx.mlx_slice(&qk_i, qk, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seq_len, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        masks[i] = try qsaMaskFromQk(self, sc, qk_i, fa, e, cache_len, sc.qsa_pos_base, 1, seq_len, false, null);
        kv_lens[i] = cache_len + seq_len;
        kv_max = @max(kv_max, kv_lens[i]);
        any = any or masks[i].ctx != null;
        any_blocks = any_blocks or sc.qsa_blocks.ctx != null;
    }
    if (any_blocks) {
        for (slots, masks) |sc, *m| {
            if (sc.qsa_blocks.ctx == null and m.ctx != null) {
                sc.qsa_mask = m.*;
                m.* = .{ .ctx = null };
            }
        }
        return mlx.mlx_array_new();
    }
    if (!any) return mlx.mlx_array_new();
    const padded = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(padded);
    for (padded) |*m| m.* = .{ .ctx = null };
    defer for (padded) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const false_v = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(false_v);
    for (0..N) |i| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        if (masks[i].ctx != null) {
            try mlx.check(mlx.mlx_array_set(&row, masks[i]));
        } else {
            try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 1, seq_len, kv_lens[i] }, 4, .bool_, self.s));
        }
        try mlx.check(mlx.mlx_pad(&padded[i], row, &[_]c_int{3}, 1, &[_]c_int{0}, 1, &[_]c_int{kv_max - kv_lens[i]}, 1, false_v, "constant", self.s));
    }
    return concatAxis0(self.s, padded);
}

/// One sequence's indexer over its projected `qk` [1, S, (n + 1) hd]: the raw index keys appended (the pooled blocks
/// they complete, then the key ring) and, past the budget, the picks in `ctx.qsa_blocks` or the bool [1, 1, S, kv]
/// mask; null-ctx otherwise. `cache_len`: key rows already held; `pos_base`: the absolute position of key row 0 (the MTP
/// head's history starts at its first draft position). `idx_merge`: what IDXEPI built.
fn qsaMaskFromQk(self: *Trunk, ctx: *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, solo: bool, idx_merge: ?*const IdxMerge) !mlx.mlx_array {
    const offset = cache_len;
    const cfg = &self.config;
    const n_idx: c_int = @intCast(cfg.indexer_n_heads);
    const idx_hd: c_int = @intCast(cfg.indexer_head_dim);
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);
    const budget: c_int = @intCast(cfg.indexer_budget);
    const block_topk: c_int = @divTrunc(budget, ratio);
    const rope_dims = ropeDims(self);
    const strides3 = [_]c_int{ 1, 1, 1 };
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    try mlx.check(mlx.mlx_slice(&k_raw, qk, &[_]c_int{ 0, 0, n_idx * idx_hd }, 3, &[_]c_int{ batch, seq_len, (n_idx + 1) * idx_hd }, 3, &strides3, 3, self.s));

    const kv: c_int = offset + seq_len;
    entry.qsa_ratio = ratio;
    if (!qsaEntrySatisfiesForward(entry, offset)) return error.QsaHistoryGap;
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    // the blocks this call completes: the one IDXEPI pooled, else the ring's leftover rows and this call's first rows
    if (nb > nb_cached and !(try appendPooled(self, entry, idx_merge, nb, nb_cached))) {
        const leftover_n = offset - nb_cached * ratio;
        const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
        const n_new = nb - nb_cached;
        const need = n_new * ratio;
        const from_k = need - leftover_n;
        if (leftover_n < 0 or leftover_n > held or from_k < 0 or from_k > seq_len) return error.QsaHistoryGap;
        var kb_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb_flat);
        if (leftover_n == 0) {
            try mlx.check(mlx.mlx_slice(&kb_flat, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, need, idx_hd }, 3, &strides3, 3, self.s));
        } else {
            const ash = mlx.getShape(entry.aux_state);
            var leftover = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(leftover);
            try mlx.check(mlx.mlx_slice(&leftover, entry.aux_state, &[_]c_int{ 0, held - leftover_n, 0 }, 3, &[_]c_int{ ash[0], held, ash[2] }, 3, &strides3, 3, self.s));
            var k_part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_part);
            try mlx.check(mlx.mlx_slice(&k_part, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, from_k, idx_hd }, 3, &strides3, 3, self.s));
            const parts = [_]mlx.mlx_array{ leftover, k_part };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&kb_flat, vec, 1, self.s));
        }
        if (mlx.mlx_array_size(kb_flat) != @as(usize, @intCast(batch * need * idx_hd))) return error.QsaHistoryGap;
        var kb4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb4);
        try mlx.check(mlx.mlx_reshape(&kb4, kb_flat, &[_]c_int{ batch, n_new, ratio, idx_hd }, 4, self.s));
        const new3 = try qsaPooledKeys(self, kb4, fa.idx_k_norm, rope_dims, pos_base + nb_cached * ratio);
        defer _ = mlx.mlx_array_free(new3);
        try qsaAppendPooled(self, entry, new3, nb_cached);
    }

    try qsaAppendKeys(self, entry, k_raw, offset);
    if (kv <= budget + ratio - 1) return mlx.mlx_array_new();
    std.debug.assert(batch == 1);

    // The index queries: IDXEPI's, else the per-head norm, [1, n, S, hd] and the partial RoPE at the chunk's positions.
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    const epi_q: ?mlx.mlx_array = if (idx_merge) |im| (if (im.q_rope.ctx != null) im.q_rope else null) else null;
    if (epi_q) |qr| {
        try mlx.check(mlx.mlx_array_set(&q_rope, qr));
    } else {
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        try mlx.check(mlx.mlx_slice(&q, qk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, n_idx * idx_hd }, 3, &strides3, 3, self.s));
        var q4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q4);
        try mlx.check(mlx.mlx_reshape(&q4, q, &[_]c_int{ batch, seq_len, n_idx, idx_hd }, 4, self.s));
        const qn = try linear.rmsNorm(self, q4, fa.idx_q_norm);
        defer _ = mlx.mlx_array_free(qn);
        var qt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qn, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
        const base = mlx.mlx_optional_float{ .value = cfg.rope_theta, .has_value = true };
        try mlx.check(mlx.mlx_fast_rope(&q_rope, qt, rope_dims, false, base, 1.0, pos_base + offset, .{ .ctx = null }, self.s));
    }

    // Picks for the gather arms (decode and verify rows, prefill past the gather's floor) and for a solo decode row at
    // every kv.
    if (attn_core.qsaWantsBlocks(batch, kv, seq_len) or (solo and seq_len == 1)) {
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        ctx.qsa_blocks = try qsaSelectBlocks(self, q_rope, entry.qsa_pooled, offset, seq_len, nb, block_topk);
        return mlx.mlx_array_new();
    }
    const sheet = try qsaScoreSheet(self.s, q_rope, entry.qsa_pooled);
    defer _ = mlx.mlx_array_free(sheet);
    // a call starting past every block's end needs no visibility sheet
    var vis3 = mlx.mlx_array{ .ctx = null };
    defer if (vis3.ctx != null) {
        _ = mlx.mlx_array_free(vis3);
    };
    if (!qsaAllBlocksVisible(offset, nb, ratio)) vis3 = try qsaBlockVisibility(self, offset, seq_len, nb, ratio);
    return qsaSelectMaskOps(self.allocator, self.s, sheet, vis3, offset, seq_len, kv, nb, ratio, block_topk);
}

// ── IDXEPI: the indexer projection's epilogue ──

/// The one block a call completes, as `qsaMaskFromQk` plans it: the history's `ln` leftover ring rows, then the call's
/// first ratio - ln rows. Null: no block, more than one, or a gap.
fn poolPlan(entry: *const SSMCacheEntry, offset: c_int, seq_len: c_int, ratio: c_int) ?PoolPlan {
    const nb = @divTrunc(offset + seq_len, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    if (nb - nb_cached != 1) return null;
    const leftover_n = offset - nb_cached * ratio;
    const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
    if (leftover_n < 0 or leftover_n > held or leftover_n >= ratio) return null;
    const from_k = ratio - leftover_n;
    if (from_k < 1 or from_k > seq_len) return null;
    return .{ .nb_cached = nb_cached, .ln = leftover_n, .held = held };
}

/// The indexer projection at decode and verify rows through `indexer`: qk as `qmatmul` gives it, the index queries
/// normed and rotated in its finish past the budget, and the pooled key of the one block the call completes (the
/// pooled-key kernel's statements over the ring's leftover rows and the call's first rows). Null: the chain.
fn indexerMerged(self: *Trunk, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int) !?IdxMerge {
    if (batch != 1 or seq_len > fusion.decode_rows_max) return null;
    const w = self.formats.linearOf(fa.idx_qk_w) orelse return null;
    const cfg = &self.config;
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);
    const rope_dims = ropeDims(self);
    const offset = cache_len;
    const kv = offset + seq_len;
    // the queries past the budget only
    const qepi = kv > @as(c_int, @intCast(cfg.indexer_budget)) + ratio - 1;
    const plan = poolPlan(entry, offset, seq_len, ratio);
    if (!qepi and plan == null) return null;
    var opts: abi.Indexer = .{
        .heads = cfg.indexer_n_heads,
        .rope_dims = @intCast(rope_dims),
        .ratio = @intCast(ratio),
        .q_rope = qepi,
        .pool = plan != null,
        .ln = 0,
        .eps = cfg.rms_norm_eps,
        .q_norm = fusion.arr(fa.idx_q_norm),
        .k_norm = fusion.arr(fa.idx_k_norm),
    };
    if (qepi) opts.angles = fusion.arr(try qkAngleRowsFor(self, rope_dims, pos_base + offset, seq_len));
    var ring = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ring);
    if (plan) |pl| {
        opts.ln = @intCast(pl.ln);
        if (pl.ln > 0) {
            const ash = mlx.getShape(entry.aux_state);
            try mlx.check(mlx.mlx_slice(&ring, entry.aux_state, &[_]c_int{ 0, pl.held - pl.ln, 0 }, 3, &[_]c_int{ ash[0], pl.held, ash[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            opts.ring = fusion.arr(ring);
        }
        const cs = try qsaPooledCosSin(self, rope_dims, pos_base + pl.nb_cached * ratio, ratio, 1, .bfloat16);
        opts.pcos = fusion.arr(cs.cos);
        opts.psin = fusion.arr(cs.sin);
    }
    const res = fusion.callIndexer(&self.formats, w, x, &opts, self.s) orelse return null;
    return .{ .qk = res[0], .q_rope = res[1], .pooled = res[2], .pool_nb_cached = if (plan) |pl| pl.nb_cached else -1 };
}

/// IDXEPI's pooled key appended where it is the block this call pools; false: the chain pools.
fn appendPooled(self: *Trunk, entry: *SSMCacheEntry, im: ?*const IdxMerge, nb: c_int, nb_cached: c_int) !bool {
    const w = im orelse return false;
    if (w.pooled.ctx == null or w.pool_nb_cached != nb_cached or nb - nb_cached != 1) return false;
    try qsaAppendPooled(self, entry, w.pooled, nb_cached);
    return true;
}

// ── the pooled block keys ──

/// `new3` appended to the pooled bank, `entry.qsa_pooled` republished as its tight view; a `nb_cached` mismatch
/// re-seeds the capacity buffer.
fn qsaAppendPooled(self: *Trunk, entry: *SSMCacheEntry, new3: mlx.mlx_array, nb_cached: c_int) !void {
    const ratio: usize = @max(@as(usize, @intCast(entry.qsa_ratio)), 1);
    if (!(entry.qsa_pooled_buf.ctx != null and entry.qsa_pooled_blocks == nb_cached)) {
        if (entry.qsa_pooled_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_pooled_buf);
        entry.qsa_pooled_buf = .{ .ctx = null };
        entry.qsa_pooled_blocks = 0;
        if (nb_cached > 0 and entry.qsa_pooled.ctx != null) {
            entry.qsa_pooled_buf = try seedCapBuf(self.s, entry.qsa_pooled, 1, entry.qsa_reserve_rows / ratio);
            entry.qsa_pooled_blocks = nb_cached;
        }
    }
    // one row per completed block: the row reservation divided by the ratio
    try capBufAppend(self.s, &entry.qsa_pooled_buf, &entry.qsa_pooled, &entry.qsa_pooled_blocks, new3, 1, entry.qsa_reserve_rows / ratio);
}

/// The pooled-block cos/sin [n, rope_dims] at positions base + step*i, shared by every layer while the key holds.
/// Borrowed.
fn qsaPooledCosSin(self: *Trunk, rope_dims: c_int, base: c_int, step: c_int, n: c_int, dt: mlx.mlx_dtype) !MropeCosSin {
    const c = &self.qsa_pooled_rope;
    if (c.stale.load(.acquire)) c.deinit();
    if (c.cos.ctx != null and c.base == base and c.step == step and c.n == n and c.dtype == dt) return .{ .cos = c.cos, .sin = c.sin };
    c.deinit();
    const cs = try ropeCosSinFromFreqs(self, rope_dims, try ropeInvFreq(self, rope_dims, self.config.rope_theta), @floatFromInt(base), @floatFromInt(step), n, dt);
    c.cos = cs.cos;
    c.sin = cs.sin;
    c.base = base;
    c.step = step;
    c.n = n;
    c.dtype = dt;
    return cs;
}

/// Is every block complete for every row of a call starting at cache position `offset`? (Always at decode width.)
fn qsaAllBlocksVisible(offset: c_int, nb: c_int, ratio: c_int) bool {
    return offset >= nb * ratio - 1;
}

// The pooled-key kernel, one simdgroup per pooled row and each rounding of the chain it replaces: the f32 block mean
// in row order (MLX's order up to 8 rows), bf16, `rms_single_row`, then the bf16 cos/sin rotation with each product
// and the sum rounded as MLX's binary kernels do. The block count is an input; T, R, RD, HALF name the pipeline.
const QSA_POOL_ROPE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_pool_rope.metal");
var qsa_pool_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_pool_rope_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qsa_pool_rope_cfg_nb: c_int = 0;

/// Pooled block keys [1, nb, 128] bf16 from raw index keys `kb4` [1, nb, ratio, 128] in one launch: the f32 block mean,
/// bf16, idx_k_norm and the partial RoPE at the block-start positions base + i*ratio.
fn qsaPooledKeys(self: *Trunk, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, rope_dims: c_int, base: c_int) !mlx.mlx_array {
    const ksh = mlx.getShape(kb4);
    const nb = ksh[1];
    const r = ksh[2];
    const cs = try qsaPooledCosSin(self, rope_dims, base, r, nb, .bfloat16);
    if (qsa_pool_rope_cfg == null or qsa_pool_rope_cfg_nb != nb) {
        if (qsa_pool_rope_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        qsa_pool_rope_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ ksh[0], nb, 128 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, ksh[0] * nb, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "R", r));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rope_dims));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HALF", @divExact(rope_dims, 2)));
        qsa_pool_rope_cfg = config;
        qsa_pool_rope_cfg_nb = nb;
    }
    const kernel = try cachedKernel(&qsa_pool_rope_kernel, "mlxserve_qsa_pool_rope", &.{ "kb", "norm_w", "cosv", "sinv", "eps", "inv_r", "nblk" }, &.{"out"}, QSA_POOL_ROPE_SOURCE, "", true);
    const eps = mlx.mlx_array_new_float(self.config.rms_norm_eps);
    defer _ = mlx.mlx_array_free(eps);
    // MLX's mean multiplies by NumberOfElements: f32(1.0 / n) computed in f64
    const inv_r = mlx.mlx_array_new_float(@floatCast(1.0 / @as(f64, @floatFromInt(r))));
    defer _ = mlx.mlx_array_free(inv_r);
    const nblk = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nblk);
    return (try kernelOuts(1, kernel, &.{ kb4, norm_w, cs.cos, cs.sin, eps, inv_r, nblk }, qsa_pool_rope_cfg.?, self.s))[0];
}

// ── the key ring and the capacity buffers ──

/// This forward's raw index keys [1, S, hd] appended to the layer's ring, `entry.aux_state` republished as its tight
/// view. `offset` is the authority the capacity buffer is checked against.
fn qsaAppendKeys(self: *Trunk, entry: *SSMCacheEntry, k_raw: mlx.mlx_array, offset: c_int) !void {
    const s = self.s;
    const add = mlx.getShape(k_raw)[1];
    const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
    if (!(entry.qsa_key_buf.ctx != null and qsaHistoryRows(entry) == offset and held == entry.qsa_key_rows)) {
        if (entry.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_key_buf);
        entry.qsa_key_buf = .{ .ctx = null };
        entry.qsa_key_rows = 0;
        if (held > 0) {
            entry.qsa_key_buf = try seedCapBuf(s, entry.aux_state, 1, @intCast(QSA_RING_ROWS));
            entry.qsa_key_rows = held;
        }
    }
    if (add >= QSA_RING_ROWS) {
        try qsaInstallRing(entry, k_raw, s);
    } else if (held + add <= QSA_RING_ROWS) {
        try capBufAppend(s, &entry.qsa_key_buf, &entry.aux_state, &entry.qsa_key_rows, k_raw, 1, @intCast(QSA_RING_ROWS));
    } else {
        try ringRebuild(self, entry, k_raw);
    }
    entry.qsa_hist_rows = offset + add;
}

/// The ring rebuilt from `src`'s last `QSA_RING_ROWS` rows (all of them when it has fewer): an owned capacity buffer and
/// `entry.aux_state` its tight view.
fn qsaInstallRing(entry: *SSMCacheEntry, src: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(src);
    const rows = sh[1];
    const n: c_int = @min(rows, QSA_RING_ROWS);
    var tmp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tmp);
    var view = src;
    if (n < rows) {
        try mlx.check(mlx.mlx_slice(&tmp, src, &[_]c_int{ 0, rows - n, 0 }, 3, &[_]c_int{ sh[0], rows, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        view = tmp;
    }
    if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
    if (entry.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    entry.qsa_key_buf = try seedCapBuf(s, view, 1, @intCast(QSA_RING_ROWS));
    entry.qsa_key_rows = n;
    entry.aux_state = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(&entry.aux_state, entry.qsa_key_buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], n, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
}

// The rebuilt ring: out [B, RING, D] = (old rows [held - keep, held) ++ kr) + zero, keep = RING - add, each input read
// through its own strides and `zero` the 0-d +0 `materializedOwnedCopy` adds: the concat's and the Add's bytes.
const RING_CAT_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_ring_cat.metal");
var ring_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// The ring kernel's 0-d +0, data-backed, so it costs no launch.
var ring_kernel_zero: ?mlx.mlx_array = null;

/// RINGCAT: the ring's last rows and this forward's keys past `QSA_RING_ROWS` rebuilt in one launch, with
/// `qsaInstallRing`'s bookkeeping on the result (the rebuilt ring is `QSA_RING_ROWS` rows: no tail slice).
fn ringRebuild(self: *Trunk, entry: *SSMCacheEntry, k_raw: mlx.mlx_array) !void {
    const s = self.s;
    const ash = mlx.getShape(entry.aux_state);
    const b = ash[0];
    const d = ash[2];
    const dt = mlx.mlx_array_dtype(k_raw);
    const kernel = try cachedKernel(&ring_kernel, "mlxserve_qsa_ring_cat", &.{ "old", "kr", "zero" }, &.{"out"}, RING_CAT_SOURCE, "", false);
    const zero = ring_kernel_zero orelse blk: {
        const bits = [_]u16{0};
        const z = mlx.mlx_array_new_data(&bits, &[_]c_int{1}, 0, dt);
        if (z.ctx == null) return error.MlxArrayNew;
        ring_kernel_zero = z;
        break :blk z;
    };
    const n: c_int = b * QSA_RING_ROWS * d;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, QSA_RING_ROWS, d }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(n, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RING", QSA_RING_ROWS));
    const ring = (try kernelOuts(1, kernel, &.{ entry.aux_state, k_raw, zero }, cfg, s))[0];
    _ = mlx.mlx_array_free(entry.aux_state);
    if (entry.qsa_key_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_key_buf);
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = ring;
    entry.qsa_key_rows = QSA_RING_ROWS;
    entry.aux_state = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(entry.aux_state);
        entry.aux_state = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(&entry.aux_state, entry.qsa_key_buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ b, QSA_RING_ROWS, d }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
}

/// Where the next `add` rows go and how big the capacity buffer must be (`cap` 0: none yet), the KV cache's growth
/// policy with the first grow raised to the reservation.
const QsaKeyAppendPlan = struct { new_cap: usize, new_rows: usize };
fn qsaKeyAppendPlan(rows: usize, cap: usize, add: usize, reserve: usize) QsaKeyAppendPlan {
    const need = rows + add;
    return .{ .new_cap = if (need <= cap) cap else @max(KVCache.nextCapacity(cap, need), reserve), .new_rows = need };
}

/// `src` materialized straight into a capacity buffer already `reserve` rows long (the tight copy when it holds them).
fn seedCapBuf(s: mlx.mlx_stream, src: mlx.mlx_array, axis: usize, reserve: usize) !mlx.mlx_array {
    const sh = mlx.getShape(src);
    const nd: usize = sh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    if (reserve <= @as(usize, @intCast(sh[axis]))) return materializedOwnedCopy(s, src);
    var shape = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    for (sh, 0..) |d, i| {
        shape[i] = d;
        stop[i] = d;
    }
    shape[axis] = @intCast(reserve);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var grown = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grown);
    try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(src), s));
    var seeded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(seeded);
    try mlx.check(mlx.mlx_slice_update(&seeded, grown, src, &start, nd, &stop, nd, &strides, nd, s));
    return seeded;
}

/// `chunk` appended into a capacity buffer along `axis` (grown first where needed), `view` republished as the tight
/// `count`-long prefix. The view is dropped first: while it lives mlx copies the buffer instead of donating it.
fn capBufAppend(s: mlx.mlx_stream, buf: *mlx.mlx_array, view: *mlx.mlx_array, count: *c_int, chunk: mlx.mlx_array, axis: usize, reserve: usize) !void {
    const csh = mlx.getShape(chunk);
    const nd: usize = csh.len;
    std.debug.assert(nd <= 4 and axis < nd);
    if (view.ctx != null) _ = mlx.mlx_array_free(view.*);
    view.* = .{ .ctx = null };

    const cap: usize = if (buf.ctx != null) @intCast(mlx.getShape(buf.*)[axis]) else 0;
    const plan = qsaKeyAppendPlan(@intCast(count.*), cap, @intCast(csh[axis]), reserve);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    for (csh, 0..) |d, i| stop[i] = d;
    if (plan.new_cap != cap) {
        var shape = [_]c_int{ 0, 0, 0, 0 };
        for (csh, 0..) |d, i| shape[i] = d;
        shape[axis] = @intCast(plan.new_cap);
        var grown = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(grown);
        try mlx.check(mlx.mlx_zeros(&grown, &shape, nd, mlx.mlx_array_dtype(chunk), s));
        if (buf.ctx != null) {
            if (count.* > 0) {
                stop[axis] = count.*;
                var old = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(old);
                try mlx.check(mlx.mlx_slice(&old, buf.*, &start, nd, &stop, nd, &strides, nd, s));
                var seeded = mlx.mlx_array_new();
                // on failure `grown` is still the old handle: the two errdefers free two different arrays
                errdefer _ = mlx.mlx_array_free(seeded);
                try mlx.check(mlx.mlx_slice_update(&seeded, grown, old, &start, nd, &stop, nd, &strides, nd, s));
                _ = mlx.mlx_array_free(grown);
                grown = seeded;
            }
            _ = mlx.mlx_array_free(buf.*);
        }
        buf.* = grown;
    }
    start[axis] = count.*;
    stop[axis] = @intCast(plan.new_rows);
    {
        // block-scoped: `updated` becomes `buf.*` on the last line
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, chunk, &start, nd, &stop, nd, &strides, nd, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }
    count.* = @intCast(plan.new_rows);
    start[axis] = 0;
    view.* = mlx.mlx_array_new();
    errdefer {
        _ = mlx.mlx_array_free(view.*);
        view.* = .{ .ctx = null };
    }
    try mlx.check(mlx.mlx_slice(view, buf.*, &start, nd, &stop, nd, &strides, nd, s));
}

/// Does `aux_state` hold the QSA key history (a full-attention layer: no conv recurrence)?
fn ssmAuxIsQsaHistory(e: *const SSMCacheEntry) bool {
    if (e.qsa_pooled.ctx != null) return true;
    if (e.aux_state.ctx == null) return false;
    if (e.conv_state.ctx == null) return true;
    return mlx.mlx_array_size(e.conv_state) == 0;
}

fn qsaHistoryRows(e: *const SSMCacheEntry) c_int {
    if (e.qsa_hist_rows > 0) return e.qsa_hist_rows;
    if (!ssmAuxIsQsaHistory(e) or e.aux_state.ctx == null) return 0;
    const sh = mlx.getShape(e.aux_state);
    return if (sh.len >= 2) sh[1] else 0;
}

/// Does the history end at `pos`, its pooled blocks and ring holding every row up to it?
fn qsaEntrySatisfiesForward(e: *const SSMCacheEntry, pos: c_int) bool {
    if (!ssmAuxIsQsaHistory(e)) return true;
    if (qsaHistoryRows(e) != pos) return false;
    const blocks: c_int = if (e.qsa_pooled.ctx != null) mlx.getShape(e.qsa_pooled)[1] else 0;
    const held: c_int = if (e.aux_state.ctx != null) mlx.getShape(e.aux_state)[1] else 0;
    return blocks * @max(e.qsa_ratio, 1) + held >= pos;
}

// ── the block score sheet ──

// The score kernel (`msv_qsa_score`, NAX tf32): the lane -> (row, col) map is the matmul2d cooperative-fragment layout
// and K runs 8 ascending 16-deep steps, the stock tf32 GEMM's order, so the sheet is the composed chain's bit for bit.
const QSA_SCORE_KERNEL_SOURCE = kernelSource("kernels/attention/qsa_score.metal");
const QSA_SCORE_KERNEL_HEADER = kernelSource("kernels/attention/qsa_score_header.metal");
var qsa_score_kernel: ?mlx.mlx_fast_metal_kernel = null;
const QsaScoreCfgKey = struct { rows: c_int, nb: c_int };
var qsa_score_cfgs = QsaCfgCache(QsaScoreCfgKey, 1){};

/// The f32 sheet [1, S, nb] = sum over the 4 index heads of relu(q_h . pooled key) over q [1, 4, S, 128] and the pooled
/// bank [1, nb, 128], bf16; the 1/sqrt(hd) scale is dropped (monotone for the top-k). 128+ rows take the h4 layout.
fn qsaScoreSheet(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !mlx.mlx_array {
    const rows = mlx.getShape(q)[2];
    const nb = mlx.getShape(pooled)[1];
    const h4 = rows >= 128;
    const nsg: c_int = if (h4) 8 else 4;
    const r_tg: c_int = nsg * @as(c_int, if (h4) 16 else 4);
    const ngrp: c_int = @divTrunc(rows + r_tg - 1, r_tg);
    const nslab: u32 = @intCast(@divTrunc(nb + 31, 32));
    var nsh: u32 = @max(1, nslab / @as(u32, if (h4) 32 else 1));
    nsh = @min(@max(nsh, @max(1, (256 + @as(u32, @intCast(ngrp)) - 1) / @as(u32, @intCast(ngrp)))), nslab);
    // row-contiguous is load-bearing: q is an axis-2 slice whose parent head stride is S*128; the copy makes Q_shape[2]
    // the physical row stride the kernel indexes by
    const kernel = try cachedKernel(&qsa_score_kernel, "msv_qsa_score", &.{ "Q", "POOL" }, &.{"OUT"}, QSA_SCORE_KERNEL_SOURCE, QSA_SCORE_KERNEL_HEADER, true);
    const key = QsaScoreCfgKey{ .rows = rows, .nb = nb };
    const cfg = if (qsa_score_cfgs.get(key)) |c| c[0] else blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, rows, nb }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @as(c_int, @intCast(nsh)) * nsg * 32, ngrp, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, nsg * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGVAL", nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "USEH4VAL", if (h4) 1 else 0));
        qsa_score_cfgs.put(key, .{config});
        break :blk config;
    };
    return (try kernelOuts(1, kernel, &.{ q, pooled }, cfg, s))[0];
}

/// The selection's row chunk: the [rows, nb] f32 sheet stays under the sheet budget, so a long context's prefill chunk
/// is scored in passes instead of materializing gigabytes.
fn qsaScoreRowsPerChunk(nb: u64, rows: u64) u64 {
    return @max(16, @min(rows, (QSA_SCORE_SHEET_BUDGET_MB_DEFAULT << 20) / (nb * 4)));
}

// ── the block selection ──

/// Picks for the gather arms: per row the top-k visible blocks ascending, INT_MAX past the row's count, [1, S, kb]
/// int32, scored and selected in row chunks under the sheet budget.
fn qsaSelectBlocks(self: *Trunk, q_rope: mlx.mlx_array, pooled: mlx.mlx_array, offset: c_int, seq_len: c_int, nb: c_int, block_topk: c_int) !mlx.mlx_array {
    const n_idx: c_int = @intCast(self.config.indexer_n_heads);
    const idx_hd: c_int = @intCast(self.config.indexer_head_dim);
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    const kb: c_int = @min(nb, block_topk);
    const rows_per: c_int = @intCast(qsaScoreRowsPerChunk(@intCast(nb), @intCast(seq_len)));
    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(self.allocator);
    }
    var r0: c_int = 0;
    while (r0 < seq_len) : (r0 += rows_per) {
        const r1: c_int = @min(r0 + rows_per, seq_len);
        const rows = r1 - r0;
        var q_chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_chunk);
        try mlx.check(mlx.mlx_slice(&q_chunk, q_rope, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, n_idx, r1, idx_hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, self.s));
        const scores = try qsaScoreSheet(self.s, q_chunk, pooled);
        defer _ = mlx.mlx_array_free(scores);
        const bounds_host = try self.allocator.alloc(i32, @intCast(rows));
        defer self.allocator.free(bounds_host);
        (Blocks{ .size = @intCast(ratio), .top_k = @intCast(block_topk) }).selectBounds(bounds_host, offset + r0, nb, qsaAllBlocksVisible(offset + r0, nb, ratio));
        const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const sorted = try qsaSelectTopBlocks(self.s, scores, bounds, kb);
        errdefer _ = mlx.mlx_array_free(sorted);
        try parts.append(self.allocator, sorted);
    }
    if (parts.items.len == 1) return parts.pop().?;
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var merged = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&merged, vec, 1, self.s));
    return merged;
}

/// Block visibility [1, S, nb]: block b is complete at cache position p iff b*ratio + ratio - 1 <= p.
fn qsaBlockVisibility(self: *Trunk, offset: c_int, seq_len: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    const s = self.s;
    // the block-end row [1, nb], built once per (nb, ratio)
    const c = &self.qsa_consts;
    if (c.blk_end2.ctx == null or c.nb != nb or c.ratio != ratio) {
        c.deinit();
        var blk_end = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_end);
        try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
        c.blk_end2 = mlx.mlx_array_new();
        errdefer c.deinit();
        try mlx.check(mlx.mlx_reshape(&c.blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
        c.nb = nb;
        c.ratio = ratio;
    }
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, c.blk_end2, pos2, s));
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s));
    return vis3;
}

/// The mask arm past the budget: the select kernel's picks to the dense [1, 1, S, kv] mask, one row's in one more
/// launch. `vis3` null-ctx: every block complete (mlx-c aborts on a null handle, so its ops are skipped).
fn qsaSelectMaskOps(alloc: std.mem.Allocator, s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int, block_topk: c_int) !mlx.mlx_array {
    const bounds_host = try alloc.alloc(i32, @intCast(seq_len));
    defer alloc.free(bounds_host);
    (Blocks{ .size = @intCast(ratio), .top_k = @intCast(block_topk) }).selectBounds(bounds_host, offset, nb, vis3.ctx == null);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{seq_len}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const picks = try qsaSelectTopBlocks(s, scores, bounds, block_topk);
    defer _ = mlx.mlx_array_free(picks);
    if (seq_len == 1) return rowMask(s, picks, kv, ratio);
    // a short row's INT_MAX padding clamped to 0: the `logical_and` with the sheet clears block 0 when it is not visible
    const nb_v = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_v);
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, picks, nb_v, s));
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_where(&top_idx, valid, picks, zero_i, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ 1, seq_len, nb }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_put_along_axis(&picked, falses, top_idx, true_v, -1, s));
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    if (vis3.ctx == null) {
        try mlx.check(mlx.mlx_array_set(&blk_sel, picked));
    } else {
        try mlx.check(mlx.mlx_logical_and(&blk_sel, picked, vis3, s));
    }
    return qsaMaskFromBlockSel(s, blk_sel, offset, seq_len, kv, nb, ratio);
}

// The select kernel (`msv_qsa_select`): one threadgroup per row, an MSD radix select over the f32 score's ordinal, then
// an index-ordered compaction: the top-K visible blocks, ties to the lower index, ascending, INT_MAX past the row's
// count. One radix body in three kernels (single row, row slice, merge) that differ only in setup.
const QSA_SELECT_KERNEL_HEADER = kernelSource("kernels/attention/qsa_select_header.metal");
const QSA_SELECT_SHARED = kernelSource("kernels/attention/qsa_select_shared.metal");
const QSA_SELECT_RADIX_BODY = kernelSource("kernels/attention/qsa_select_radix_body.metal");
const QSA_SELECT_KERNEL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, kernelSource("kernels/attention/qsa_select_single_setup.metal"), QSA_SELECT_RADIX_BODY });
const QSA_SELECT_SPLIT_LOCAL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, kernelSource("kernels/attention/qsa_select_slice_setup.metal"), QSA_SELECT_RADIX_BODY });
// the merge's candidates arrive ascending by original index, so a cross-slice tie at the K-th ordinal goes to the
// lowest index
const QSA_SELECT_SPLIT_MERGE_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, kernelSource("kernels/attention/qsa_select_merge_setup.metal"), QSA_SELECT_RADIX_BODY });

/// Decode and verify widths split each row over G slices past this many blocks (latency-bound at one threadgroup per
/// row), and a second launch selects over the G*K candidates: exact, as every global top-K element is in its slice's.
const QSA_SELECT_SPLIT_MIN_NB: c_int = 24576;
const QSA_SELECT_SPLIT_MAX_ROWS: c_int = 15;
const QSA_SELECT_SPLIT_G: c_int = 16;

var qsa_select_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_select_split_local_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_select_split_merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
const QsaSelectCfgKey = struct { rows: c_int, k: c_int };
var qsa_select_cfgs = QsaCfgCache(QsaSelectCfgKey, 1){};
var qsa_select_split_cfgs = QsaCfgCache(QsaSelectCfgKey, 2){};

/// Eight-bit digits at decode and verify widths (each level's fixed cost is the bill there), 11-bit at prefill widths.
fn qsaSelectDigitBits(rows: c_int) c_int {
    return if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) 8 else 11;
}

/// The sorted int32 [1, rows, kb] picks of f32 [1, rows, nb] scores under int32 [rows] bounds.
fn qsaSelectTopBlocks(s: mlx.mlx_stream, scores: mlx.mlx_array, bounds: mlx.mlx_array, kb: c_int) !mlx.mlx_array {
    const ssh = mlx.getShape(scores);
    const rows = ssh[1];
    const nb = ssh[2];
    const key = QsaSelectCfgKey{ .rows = rows, .k = kb };
    const digits = qsaSelectDigitBits(rows);
    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS and nb >= QSA_SELECT_SPLIT_MIN_NB and 2 * QSA_SELECT_SPLIT_G * kb <= nb) {
        const g = QSA_SELECT_SPLIT_G;
        const local_kernel = try cachedKernel(&qsa_select_split_local_kernel, "msv_qsa_select_split_local", &.{ "scores", "bounds" }, &.{"ids"}, QSA_SELECT_SPLIT_LOCAL_SOURCE, QSA_SELECT_KERNEL_HEADER, true);
        const merge_kernel = try cachedKernel(&qsa_select_split_merge_kernel, "msv_qsa_select_split_merge", &.{ "scores", "local_ids" }, &.{"ids"}, QSA_SELECT_SPLIT_MERGE_SOURCE, QSA_SELECT_KERNEL_HEADER, true);
        const cfgs = qsa_select_split_cfgs.get(key) orelse blk: {
            const local_cfg = try selectConfig(rows, kb * g, 512, g, digits);
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(local_cfg);
            const merge_cfg = try selectConfig(rows, kb, 1024, 1, digits);
            qsa_select_split_cfgs.put(key, .{ local_cfg, merge_cfg });
            break :blk [2]mlx.mlx_fast_metal_kernel_config{ local_cfg, merge_cfg };
        };
        const local_ids = (try kernelOuts(1, local_kernel, &.{ scores, bounds }, cfgs[0], s))[0];
        defer _ = mlx.mlx_array_free(local_ids);
        return (try kernelOuts(1, merge_kernel, &.{ scores, local_ids }, cfgs[1], s))[0];
    }
    // the row walk indexes linearly: mlx materializes a view
    const kernel = try cachedKernel(&qsa_select_kernel, "msv_qsa_select", &.{ "scores", "bounds" }, &.{"ids"}, QSA_SELECT_KERNEL_SOURCE, QSA_SELECT_KERNEL_HEADER, true);
    // a row's threadgroup: the wide passes of a prefill chunk run more rows at once at 256 threads, a narrow pass wants
    // 512, decode and verify widths 1024; the picks are the same at every width
    const tg: c_int = if (rows >= 256) 256 else if (rows >= 64) 512 else 1024;
    const cfg = if (qsa_select_cfgs.get(key)) |c| c[0] else blk: {
        const config = try selectConfig(rows, kb, tg, 1, digits);
        qsa_select_cfgs.put(key, .{config});
        break :blk config;
    };
    return (try kernelOuts(1, kernel, &.{ scores, bounds }, cfg, s))[0];
}

/// A select launch's config: int32 [1, rows, width] out, `slices` threadgroups of `tg` per row, templates TGS, K, DIGITS.
fn selectConfig(rows: c_int, width: c_int, tg: c_int, slices: c_int, digits: c_int) !mlx.mlx_fast_metal_kernel_config {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, rows, width }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg * slices, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TGS", tg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", @divExact(width, slices)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DIGITS", digits));
    return config;
}

// ── the masks ──

// One decode row's key mask in one launch: a token is visible iff its block is among the sorted picks (a binary search;
// the INT_MAX sentinels never match) or it lies in the incomplete tail. Grid (kv), one thread a token.
const ROW_MASK_SOURCE = kernelSource("kernels/attention/qsa_mask.metal");
var row_mask_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// [1, 1, 1, kv] bool for one row's picks [1, 1, kb] int32.
fn rowMask(s: mlx.mlx_stream, picks: mlx.mlx_array, kv: c_int, ratio: c_int) !mlx.mlx_array {
    const kernel = try cachedKernel(&row_mask_kernel, "mlxserve_qsa_row_mask", &.{"blocks"}, &.{"mask"}, ROW_MASK_SOURCE, "", true);
    // the mask's width is the step's kv: a config per call
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 1, kv }, 4, .bool_));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, kv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(kv, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KB", mlx.getShape(picks)[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", ratio));
    return (try kernelOuts(1, kernel, &.{picks}, cfg, s))[0];
}

/// A [1, S, nb] bool block selection to the [1, 1, S, kv] token mask: the selected blocks' tokens (zero-padded to kv)
/// ∪ each row's own incomplete tail (tokens from ratio*floor((p+1)/ratio)), ∧ causal. `offset`: row 0's position.
fn qsaMaskFromBlockSel(s: mlx.mlx_stream, blk_sel: mlx.mlx_array, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var sel4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel4);
    try mlx.check(mlx.mlx_expand_dims(&sel4, blk_sel, -1, s));
    var sel_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_b);
    try mlx.check(mlx.mlx_broadcast_to(&sel_b, sel4, &[_]c_int{ 1, seq_len, nb, ratio }, 4, s));
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, sel_b, &[_]c_int{ 1, seq_len, nb * ratio }, 3, s));
    var sel_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_full);
    const pad: c_int = kv - nb * ratio;
    if (pad > 0) {
        var zeros_pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros_pad);
        try mlx.check(mlx.mlx_zeros(&zeros_pad, &[_]c_int{ 1, seq_len, pad }, 3, .bool_, s));
        const parts = [_]mlx.mlx_array{ sel_tok, zeros_pad };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&sel_full, vec, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&sel_full, sel_tok));
    }
    var kv_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx);
    try mlx.check(mlx.mlx_arange(&kv_idx, 0, @floatFromInt(kv), 1.0, .int32, s));
    var kv_idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx2);
    try mlx.check(mlx.mlx_reshape(&kv_idx2, kv_idx, &[_]c_int{ 1, kv }, 2, s));
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    const ratio_i = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_i, s));
    var tail_start = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_start);
    try mlx.check(mlx.mlx_multiply(&tail_start, nb_p, ratio_i, s));
    var tail_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_mask);
    try mlx.check(mlx.mlx_greater_equal(&tail_mask, kv_idx2, tail_start, s));
    var tok_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_mask);
    try mlx.check(mlx.mlx_logical_or(&tok_mask, sel_full, tail_mask, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, kv_idx2, pos2, s));
    var mask3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask3);
    try mlx.check(mlx.mlx_logical_and(&mask3, tok_mask, causal, s));
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask, mask3, &[_]c_int{ 1, 1, seq_len, kv }, 4, s));
    return mask;
}
