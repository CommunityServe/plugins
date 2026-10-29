//! copied_from: src/plugins/qwen4_exp/qsa/merges.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The QSA indexer's decode-row merges, each the bytes of the chain it replaces: the indexer epilogue (the indexer
//! projection's finish normalizes and rotates the index queries and pools the block key the call completes, through the
//! trunk format's `indexer` entry), the placed row's mask (a placed MTP draft row's bounded select and its key
//! visibility in one mask launch) and the ring rebuild (the key ring's in one launch).

const std = @import("std");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const fusion = @import("../trunk/fusion.zig");
const prof_mod = @import("../prof.zig");
const Trunk = @import("../trunk.zig").Trunk;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const FullAttnWeights = @import("../../../engines/mlx/layer_weights.zig").FullAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Blocks = @import("../../../sdk/blocks/attention/sparse_block/params.zig").Blocks;
const qsaSelectTopBlocks = @import("../../../sdk/blocks/attention/sparse_block/select.zig").qsaSelectTopBlocks;
const QSA_RING_ROWS = @import("history.zig").QSA_RING_ROWS;
const qsaEntrySatisfiesForward = @import("checkpoint.zig").qsaEntrySatisfiesForward;
const pooled = @import("pooled.zig");

// ── the indexer epilogue ──

/// The indexer epilogue's results for one layer (each the holder's): qk [1, S, (n + 1) 128] as `qmatmul` gives it, the
/// index queries normed and rotated [1, n, S, 128] (null-ctx: the chain builds them), the one pooled block key [1, 1,
/// 128] (null-ctx: the chain pools it), and the pooled block count its plan read.
pub const IdxMerge = struct {
    qk: mlx.mlx_array,
    q_rope: mlx.mlx_array = .{ .ctx = null },
    pooled: mlx.mlx_array = .{ .ctx = null },
    pool_nb_cached: c_int = -1,

    pub fn deinit(self: *IdxMerge) void {
        inline for (.{ &self.qk, &self.q_rope, &self.pooled }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// What the indexer epilogue needs of a call (`idxRefusal`'s record).
pub const IdxFacts = struct {
    batch: c_int,
    rows: c_int,
    /// The indexer projection is the trunk format's linear.
    linear: bool,
    /// x is bf16 (the entry's epilogue is bf16).
    bf16: bool,
    idx_hd: u32,
    ratio: c_int,
    rope_dims: c_int,
    /// The scalar text RoPE: no M-RoPE, no spec tree, no YaRN.
    text_rope: bool,
    /// The index q / k norm weights present, bf16.
    norms: bool,
};

/// The first reason the indexer epilogue does not serve a call, or null.
pub fn idxRefusal(f: IdxFacts) ?[]const u8 {
    if (f.batch != 1) return "batched rows";
    if (f.rows < 1 or f.rows > fusion.decode_rows_max) return "not decode / verify rows";
    if (!f.linear) return "the indexer projection is not the trunk format's linear";
    if (!f.bf16) return "activations not bf16";
    if (f.idx_hd != 128) return "indexer head_dim is not 128";
    if (f.ratio < 1 or f.ratio > pooled.QSA_POOL_SERIAL_MAX_R) return "block ratio past the pooled kernel's serial order";
    if (f.rope_dims <= 0 or f.rope_dims > 128 or @mod(f.rope_dims, 8) != 0) return "rotary width not served";
    if (!f.text_rope) return "not the scalar text RoPE (M-RoPE, spec tree or YaRN)";
    if (!f.norms) return "no bf16 index q / k norm";
    return null;
}

/// The one block a call completes, as `qsaMaskFromQk` plans it: `ln` ring rows (the history's leftover), then the
/// call's first ratio - ln rows. Null where the call completes no block, more than one, or the chain would raise its
/// gap error.
pub const PoolPlan = struct { nb_cached: c_int, ln: c_int, held: c_int };

pub fn poolPlan(entry: *const SSMCacheEntry, offset: c_int, seq_len: c_int, ratio: c_int) ?PoolPlan {
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

/// The indexer epilogue: the indexer projection at decode rows through `indexer`, byte for byte the chain `qsaMask` /
/// `qsaMaskFromQk` run: qk as `qmatmul` gives it; the index queries normed and rotated in the finish (the hd-128
/// QK-norm + RoPE kernel's statements, MLX's rope's own floats from `qkAngleRowsFor`); the pooled block key the call
/// completes (the pooled-key kernel's statements over the ring's leftover rows and the call's first rows, the cos / sin
/// row `qsaPooledCosSin` gives that kernel). Null: the chain.
pub fn indexerMerged(self: *Trunk, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int) !?IdxMerge {
    // An EXL3 trunk pack's only: every other pack runs main's indexer chain, and says nothing.
    if (!self.config.trunk_format) return null;
    const cfg = &self.config;
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);
    const rope_dims = fusion.ropeDims(self);
    const w = self.formats.linearOf(fa.idx_qk_w);
    const facts: IdxFacts = .{
        .batch = batch,
        .rows = seq_len,
        .linear = w != null,
        .bf16 = mlx.mlx_array_dtype(x) == .bfloat16,
        .idx_hd = cfg.indexer_head_dim,
        .ratio = ratio,
        .rope_dims = rope_dims,
        .text_rope = ctx.mrope_pos == null and ctx.mrope_cos_cur == null and ctx.tree == null and !self.yarnActive(),
        .norms = fa.idx_q_norm.ctx != null and fa.idx_k_norm.ctx != null and mlx.mlx_array_dtype(fa.idx_q_norm) == .bfloat16 and
            mlx.mlx_array_dtype(fa.idx_k_norm) == .bfloat16,
    };
    if (idxRefusal(facts)) |why| {
        if (seq_len >= 1 and seq_len <= fusion.decode_rows_max) fusion.declined(.indexer, why);
        return null;
    }
    const offset = cache_len;
    const kv = offset + seq_len;
    const budget: c_int = @intCast(cfg.indexer_budget);
    // the index queries exist past the budget only (qsaMaskFromQk returns its dense mask first); a placed draft row
    // rotates by `head_rope_pos`, which the chain keeps
    const qepi = kv > budget + ratio - 1 and ctx.head_rope_pos.ctx == null;
    // the pooled key: the one block this call completes, where the chain's fused pooled-key kernel would build it
    const plan = poolPlan(entry, offset, seq_len, ratio);
    const pool = plan != null and pooled.qsaIdxRopeFusedEnabled() and qsaEntrySatisfiesForward(entry, offset) and
        !env.onCached(&prof_mod.qwen4_no_pooled_env, .QWEN4_NO_POOLED);
    if (!qepi and !pool) return null;
    var opts: abi.Indexer = .{
        .heads = cfg.indexer_n_heads,
        .rope_dims = @intCast(rope_dims),
        .ratio = @intCast(ratio),
        .q_rope = qepi,
        .pool = pool,
        .ln = 0,
        .eps = cfg.rms_norm_eps,
        .q_norm = fusion.arr(fa.idx_q_norm),
        .k_norm = fusion.arr(fa.idx_k_norm),
    };
    if (qepi) opts.angles = fusion.arr(try self.qkAngleRowsFor(0, rope_dims, .{ .value = cfg.rope_theta, .has_value = true }, .{ .ctx = null }, pos_base + offset, seq_len, 1.0, 1.0));
    var ring = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ring);
    if (pool) {
        const pl = plan.?;
        opts.ln = @intCast(pl.ln);
        if (pl.ln > 0) {
            // qsaMaskFromQk's leftover: the ring's last ln rows
            const ash = mlx.getShape(entry.aux_state);
            try mlx.check(mlx.mlx_slice(&ring, entry.aux_state, &[_]c_int{ 0, pl.held - pl.ln, 0 }, 3, &[_]c_int{ ash[0], pl.held, ash[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            opts.ring = fusion.arr(ring);
        }
        const cs = try self.qsaPooledCosSin(ctx, rope_dims, pos_base + pl.nb_cached * ratio, ratio, 1, .bfloat16);
        opts.pcos = fusion.arr(cs.cos);
        opts.psin = fusion.arr(cs.sin);
    }
    const res = fusion.callIndexer(&self.formats, w.?, x, &opts, self.s) orelse return null;
    fusion.served(.indexer, "the index queries' norm and rope and the pooled block key in the indexer projection's finish (rows={d} kv={d} queries={} pooled={})", .{ seq_len, kv, qepi, pool });
    return .{ .qk = res[0], .q_rope = res[1], .pooled = res[2], .pool_nb_cached = if (pool) plan.?.nb_cached else -1 };
}

/// The indexer epilogue: append the pooled block key the indexer's finish built in the chain's place, when it is the
/// block the chain pools here (one new block, the plan's pooled count); false: the chain pools it.
pub fn appendPooled(self: *Trunk, entry: *SSMCacheEntry, im: ?*const IdxMerge, nb: c_int, nb_cached: c_int) !bool {
    const w = im orelse return false;
    if (w.pooled.ctx == null or w.pool_nb_cached != nb_cached or nb - nb_cached != 1) return false;
    try self.qsaAppendPooled(entry, w.pooled, nb_cached);
    return true;
}

/// The indexer epilogue's index queries for `qsaMaskFromQk`, or null (the chain builds them).
pub fn qRope(im: ?*const IdxMerge) ?mlx.mlx_array {
    const w = im orelse return null;
    return if (w.q_rope.ctx != null) w.q_rope else null;
}

// ── the placed row's mask ──

/// The row mask's statements over the sorted picks (INT_MAX padded), then `headPlaceVisibility`'s ops on the row: live
/// = t < live_end, past = t >= dead_end, recent = t >= ratio floor_divide(live_end, ratio) (MLX's signed floor_divide),
/// mask = (row mask | recent) & (live | past). Grid (kv), one thread a token.
pub const PLACE_VIS_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_place_vis.metal");

var place_vis_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The placed row's mask over `picks` [1, 1, kb] int32 (the select kernel's sorted picks): [1, 1, 1, kv] bool.
pub fn placeVis(s: mlx.mlx_stream, picks: mlx.mlx_array, live_end: mlx.mlx_array, dead_end: c_int, kv: c_int, ratio: c_int) !mlx.mlx_array {
    const bs = mlx.getShape(picks);
    if (bs.len != 3 or bs[0] != 1 or bs[1] != 1 or bs[2] < 1 or mlx.mlx_array_dtype(picks) != .int32 or kv < 1 or ratio < 1) return error.BadShape;
    if (place_vis_kernel == null) {
        const ins = [_][*:0]const u8{ "blocks", "live_end", "dead_end" };
        const outs = [_][*:0]const u8{"mask"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_place_vis", in_vec, out_vec, PLACE_VIS_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        place_vis_kernel = k;
    }
    // the mask's width is the step's kv (as the row mask's): a config per call; dead_end rides an input, so no template
    // value changes per step
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 1, kv }, 4, .bool_));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, kv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(kv, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KB", bs[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", ratio));
    const dead = mlx.mlx_array_new_int(dead_end);
    defer _ = mlx.mlx_array_free(dead);
    const ins = [_]mlx.mlx_array{ picks, live_end, dead };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, place_vis_kernel.?, v, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

/// One placed row over its score sheet [1, 1, nb] f32: the select bounded at min(the row's visible bound,
/// floor_divide(live_end, ratio)) (both on the device: live_end is the lazy draft's), the first block `headPlaceScores`
/// sets to -inf, then `placeVis`. Every score it keeps is a sum of relus (>= 0, or NaN: the select ranks NaN highest
/// and -inf lowest), so below that bound the chain's top-k is this top-k, and the -inf blocks the chain may add hold
/// only rows `headPlaceVisibility` shows as recent either way: the chain's mask bit for bit. Null: the select declined.
pub fn placedRowMask(s: mlx.mlx_stream, sheet: mlx.mlx_array, live_end: mlx.mlx_array, dead_end: c_int, offset: c_int, kv: c_int, nb: c_int, ratio: c_int, block_topk: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    var bound_host: [1]i32 = undefined;
    (Blocks{ .size = @intCast(ratio), .top_k = @intCast(block_topk) }).selectBounds(&bound_host, offset, nb, pooled.qsaAllBlocksVisible(offset, nb, ratio));
    const bh = mlx.mlx_array_new_data(&bound_host, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(bh);
    const ratio_v = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_v);
    var eligible_n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eligible_n);
    try mlx.check(mlx.mlx_floor_divide(&eligible_n, live_end, ratio_v, s));
    var bounds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bounds);
    try mlx.check(mlx.mlx_minimum(&bounds, bh, eligible_n, s));
    const picks = (qsaSelectTopBlocks(s, sheet, bounds, block_topk) catch null) orelse return null;
    defer _ = mlx.mlx_array_free(picks);
    return try placeVis(s, picks, live_end, dead_end, kv, ratio);
}

/// The placed row's mask: a placed MTP draft row's QSA mask as the select launch + one mask launch (`placedRowMask`),
/// byte for byte `headPlaceScores` + `qsaSelectMaskOps` + `headPlaceVisibility`, so the -inf `where` before the select
/// and the visibility's ops after it leave the critical path. Marks the layer (`Scratch.place_vis`:
/// `qwen4AttnProjected` then skips `headPlaceVisibility`). Null: the chain.
pub fn placedMask(self: *Trunk, sheet: mlx.mlx_array, p: *const Trunk.HeadPlace, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int, block_topk: c_int, batch: c_int, solo: bool) !?mlx.mlx_array {
    // An EXL3 trunk pack's only: every other pack runs main's visibility chain, and says nothing.
    if (!self.config.trunk_format) return null;
    if (!solo or batch != 1 or seq_len != 1 or nb <= block_topk) return null;
    if (env.onCached(&prof_mod.qwen4_debug_scores_env, .QWEN4_DEBUG_SCORES)) return null;
    if (p.live_end.ctx == null or mlx.mlx_array_ndim(p.live_end) != 0 or mlx.mlx_array_dtype(p.live_end) != .int32) return placedDeclined("the row's live end is not a 0-d int32");
    if (mlx.mlx_array_ndim(sheet) != 3 or mlx.mlx_array_dtype(sheet) != .float32) return placedDeclined("the score sheet is not an f32 [1, 1, nb] row");
    if (!mlx.streamIsGpu(self.s)) return placedDeclined("not a Metal stream");
    const m = (try placedRowMask(self.s, sheet, p.live_end, p.dead_end, offset, kv, nb, ratio, block_topk)) orelse return placedDeclined("the bounded select declined the row");
    self.fused.place_vis = true;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[qsa] placed-row visibility engaged: a placed draft row's bounded select and its key visibility in one launch (kv={d} top-{d})\n", .{ kv, block_topk });
    }
    return m;
}

var placed_declined_said = false;

/// The placed row's mask's first decline, said once. Null: the chain.
fn placedDeclined(why: []const u8) ?mlx.mlx_array {
    if (!placed_declined_said) {
        placed_declined_said = true;
        log.info("[qsa] placed-row visibility declined: {s}; the chain and headPlaceVisibility run\n", .{why});
    }
    return null;
}

// ── the ring rebuild ──

/// The rebuilt ring: out [B, RING, D] = (old rows [held - keep, held) ++ kr) + zero, keep = RING - add, each input read
/// through its own strides (kr is a strided slice of the indexer projection) and `zero` the 0-d +0 that
/// `materializedOwnedCopy` adds: the bytes of concatenate + MLX's Add (x + y in T), -0 keys reading +0 as there.
pub const RING_CAT_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_ring_cat.metal");

var ring_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// The ring kernel's 0-d +0 per dtype (bf16, f16), a per-process cache: data-backed, so it costs no launch (the chain's
/// zeros is one).
var ring_kernel_zero: [2]?mlx.mlx_array = .{ null, null };

pub fn ringCat(s: mlx.mlx_stream, old: mlx.mlx_array, kr: mlx.mlx_array, b: c_int, d: c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    if (ring_kernel == null) {
        const ins = [_][*:0]const u8{ "old", "kr", "zero" };
        const outs = [_][*:0]const u8{"out"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        // strided reads: no row-contiguous copy of the keys
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_ring_cat", in_vec, out_vec, RING_CAT_SOURCE, "", false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        ring_kernel = k;
    }
    const slot: usize = if (dt == .bfloat16) 0 else 1;
    const zero = ring_kernel_zero[slot] orelse blk: {
        const bits = [_]u16{0};
        const z = mlx.mlx_array_new_data(&bits, &[_]c_int{1}, 0, dt);
        if (z.ctx == null) return error.MlxArrayNew;
        ring_kernel_zero[slot] = z;
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
    const ins = [_]mlx.mlx_array{ old, kr, zero };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, ring_kernel.?, v, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

/// The ring rebuild: the key ring's rebuild in one launch (`ringCat`), byte for byte `qsaAppendKeys`'s concat of the
/// ring's last rows and this forward's keys + `qsaInstallRing`'s owned copy (MLX's zeros + Add), with
/// `qsaInstallRing`'s bookkeeping on the result. False: the chain.
pub fn ringRebuild(self: *Trunk, entry: *SSMCacheEntry, k_raw: mlx.mlx_array, held: c_int, add: c_int) !bool {
    // An EXL3 trunk pack's only: every other pack runs main's chain, and says nothing.
    if (!self.config.trunk_format) return false;
    const s = self.s;
    if (!mlx.streamIsGpu(s)) return ringDeclined("not a Metal stream", entry.aux_state, k_raw, held, add);
    if (entry.aux_state.ctx == null) return ringDeclined("no ring", entry.aux_state, k_raw, held, add);
    const ksh = mlx.getShape(k_raw);
    const ash = mlx.getShape(entry.aux_state);
    if (ksh.len != 3 or ash.len != 3 or ksh[0] != ash[0] or ksh[2] != ash[2] or ksh[1] != add or ash[1] != held)
        return ringDeclined("the ring's and the keys' shapes", entry.aux_state, k_raw, held, add);
    const dt = mlx.mlx_array_dtype(k_raw);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(entry.aux_state) != dt)
        return ringDeclined("the keys are not bf16 / f16 in the ring's dtype", entry.aux_state, k_raw, held, add);
    if (add < 1 or add >= QSA_RING_ROWS or held + add <= QSA_RING_ROWS)
        return ringDeclined("rows", entry.aux_state, k_raw, held, add);
    // read before the ring's handle is freed below
    const b = ash[0];
    const d = ash[2];
    const ring = try ringCat(s, entry.aux_state, k_raw, b, d, dt);
    // qsaInstallRing on its owned copy: the rebuilt ring is QSA_RING_ROWS rows, so no tail slice
    if (entry.aux_state.ctx != null) _ = mlx.mlx_array_free(entry.aux_state);
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
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[qsa] key-ring concat engaged: the ring's concat and owned copy in one launch (held={d} add={d})\n", .{ held, add });
    }
    return true;
}

/// The ring rebuild's first decline, logged once with the ring's and the keys' shapes and dtypes. False: the chain.
fn ringDeclined(why: []const u8, ring: mlx.mlx_array, keys: mlx.mlx_array, held: c_int, add: c_int) bool {
    const Once = struct {
        var declined = false;
    };
    if (Once.declined) return false;
    Once.declined = true;
    const rsh: []const c_int = if (ring.ctx != null) mlx.getShape(ring) else &.{};
    const rdt: []const u8 = if (ring.ctx != null) @tagName(mlx.mlx_array_dtype(ring)) else "none";
    log.info("[qsa] the key ring's one-launch rebuild declined: {s} (ring {any} {s}, keys {any} {s}, held={d} add={d}); " ++
        "the chain runs\n", .{ why, rsh, rdt, mlx.getShape(keys), @tagName(mlx.mlx_array_dtype(keys)), held, add });
    return false;
}

// ── tests ──

const testing = std.testing;

fn testIdxFacts() IdxFacts {
    return .{ .batch = 1, .rows = 1, .linear = true, .bf16 = true, .idx_hd = 128, .ratio = 4, .rope_dims = 64, .text_rope = true, .norms = true };
}

test "qsa merges: the indexer epilogue serves decode rows only, and each fact refuses alone" {
    try testing.expectEqual(@as(?[]const u8, null), idxRefusal(testIdxFacts()));
    var f = testIdxFacts();
    f.rows = fusion.decode_rows_max + 1;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.batch = 2;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.linear = false;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.bf16 = false;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.idx_hd = 64;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.ratio = pooled.QSA_POOL_SERIAL_MAX_R + 1;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.rope_dims = 60;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.text_rope = false;
    try testing.expect(idxRefusal(f) != null);
    f = testIdxFacts();
    f.norms = false;
    try testing.expect(idxRefusal(f) != null);
}
