//! The direct model's attention core, after attn.zig's projections, norms and RoPE: the KV append, attention over the
//! indexer's block picks (the decode kernel, the split pass with its merge folded into the gated o_proj, the prefill
//! gather) or under its mask, the batched tick's reads, SDPA below the QSA budget (int8 Q and K under `--int8-qk`), and
//! the gate and o_proj. Cut from today's `trunk/attn.zig`, `trunk/fusion.zig`, `qsa/` and the `sdk/blocks/attention`
//! launchers to what the K packs run: a dense bf16 KV cache (a quantized one is refused), an M5-class GPU (NAX), every
//! launch on a GPU stream (the trunk's, a wavefront half's or a side stream), so the copies' `streamIsGpu` and NAX
//! declines are gone.

const std = @import("std");
const abi = @import("../mlx_serve/abi/quant.zig");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const fusion = @import("../mlx_serve/plugins/qwen4_exp/trunk/fusion.zig");
const SplitParts = @import("../mlx_serve/plugins/qwen4_exp/qsa/sparse_attn.zig").SplitParts;
const MAX_BATCH_ROWS = @import("../mlx_serve/engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCache;
const FullAttnWeights = @import("../mlx_serve/engines/mlx/layer_weights.zig").FullAttnWeights;
const DenseKVView = @import("../mlx_serve/sdk/blocks/attention/kv_view.zig").DenseKVView;
const ShapeKey = @import("../mlx_serve/sdk/blocks/moe/router_kernel.zig").ShapeKey;
const QsaCfgCache = @import("../mlx_serve/mlx/kernel_cache.zig").QsaCfgCache;
const axisView = @import("../mlx_serve/mlx/array.zig").axisView;
const bf16Scalar = @import("../mlx_serve/mlx/array.zig").bf16Scalar;
const scalarOf = @import("../mlx_serve/mlx/array.zig").scalarOf;
const macosProductVersion = @import("../mlx_serve/mlx/device.zig").macosProductVersion;
const macosVersionAtLeast = @import("../mlx_serve/mlx/device.zig").macosVersionAtLeast;
const swigluSigTable = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig").swigluSigTable;
const stridedSigmoidGateMul = @import("../mlx_serve/sdk/blocks/attention/gates.zig").stridedSigmoidGateMul;
const linear = @import("linear.zig");

/// From this many rows a call is a prefill block (the copies' `FUSED256_MIN_Q_LEN`); decode and verify rows are fewer.
const PREFILL_ROWS: c_int = 16;
/// The split pass's splits per (row, kv head) and its key tile; the prefill gathers' key tile.
const QSA_ATTN_NSPLIT: c_int = 16;
const QSA_ATTN_BK: c_int = 16;
const QSA_GATHER_BK: c_int = 32;
/// The kv past which a decode row, and a prefill block on the stock gather, take block picks; the tensor-unit gather's
/// prefill floor, (top-512 + 1) * 4 keys (at or below it every block is visible).
const QSA_GATHER_MIN_KV: c_int = 8192;
const QSA_GATHER_MIN_KV_NAX: c_int = 2052;
/// The decode kernel's visible keys per simdgroup (a multiple of 8 and of RATIO), and its head dim.
const QSA_DEC_CH: c_int = 32;
const QSA_DEC_BD: c_int = 256;
/// The hd-256 mask kernel's work per dispatch (batch·Hq·qL·keys); past it the key axis splits into dispatches.
const FUSED256_DISPATCH_BUDGET: i64 = 250_000_000;
/// The int8 Q/K kernel's query tile, 16 rows a simdgroup and `Q8_WM` simdgroups down (the rows one Q scale covers, and
/// the fewest a block takes it with), and its key block (the keys one K scale covers).
const Q8_WM: c_int = 4;
const Q8_BQ: c_int = 16 * Q8_WM;
const Q8_BK: c_int = 32;

const ATTN256_KERNEL_HEADER = kernelSource("kernels/attention/attn256_header.metal");
const ATTN256_KERNEL_SOURCE = kernelSource("kernels/attention/attn256.metal");
const ATTN_QSA256_KERNEL_HEADER = ATTN256_KERNEL_HEADER ++ kernelSource("plugins/qwen4_exp/kernels/attn_qsa256_header.metal");
const ATTN_QSA256_KERNEL_SOURCE = kernelSource("plugins/qwen4_exp/kernels/attn_qsa256.metal");
const QSA_NAX_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_nax.metal");
const QSA_NAX_HEADER = kernelSource("plugins/qwen4_exp/kernels/qsa_nax_header.metal");
/// The split pass over a dense cache: the shared pieces with the bf16 loads.
const QSA_ATTN_D_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_consts.metal") ++
    kernelSource("plugins/qwen4_exp/kernels/qsa_attn_row_grid.metal") ++ kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_strides.metal") ++
    kernelSource("plugins/qwen4_exp/kernels/qsa_attn_row.metal") ++ kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_bases.metal") ++
    kernelSource("plugins/qwen4_exp/kernels/qsa_attn_tile_open.metal") ++ kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_kload.metal") ++
    kernelSource("plugins/qwen4_exp/kernels/qsa_attn_scores.metal") ++ kernelSource("plugins/qwen4_exp/kernels/qsa_attn_d_vload.metal") ++
    kernelSource("plugins/qwen4_exp/kernels/qsa_attn_tile_close.metal");
const QSA_ATTN_MERGE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_attn_merge.metal");
const QSA_DEC_HEADER = kernelSource("kernels/attention/qsa_decode_header.metal");
const QSA_DEC_PARTS_SOURCE = kernelSource("kernels/attention/qsa_decode_parts.metal");
const QSA_DEC_MERGE_SOURCE = kernelSource("kernels/attention/qsa_decode_merge.metal");
/// The int8 Q/K hd-256 entry of MLX's tensor-unit flash attention (ported from oMLX), and int8_attn.zig's row quantizer.
const NAX_ATTN_256Q8_SOURCE = kernelSource("kernels/attention/nax_attention_256q8_entry.metal");
const NAX_ATTN_HEADER = kernelSource("kernels/attention/nax_tiles.metal") ++ kernelSource("kernels/attention/nax_attention.metal");
const INT8_QUANT_ROWS_SOURCE = kernelSource("kernels/dit/int8_attn_quant.metal");

/// The server's `--int8-qk` (src/host_flags.zig sets it at load): a dense prefill block of 64 rows or more attends on
/// int8 Q and K (lossy).
pub var int8_qk = false;
var int8_qk_engaged = false;

var attn256_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attn_qsa256_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_attn_d_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_attn_merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_dec_parts_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_dec_merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
var nax_attn_256q8_kernel: ?mlx.mlx_fast_metal_kernel = null;
var int8_quant_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

const QsaAttnCfgKey = struct { q_shape: ShapeKey, h_kv: c_int, ratio: c_int };
var qsa_attn_cfgs = QsaCfgCache(QsaAttnCfgKey, 2){};
const QsaGatherCfgKey = struct { q_shape: ShapeKey, h_kv: c_int, ratio: c_int, nax: bool };
var qsa_gather_cfgs = QsaCfgCache(QsaGatherCfgKey, 1){};
const QsaDecKey = struct { hq: c_int, hk: c_int, kb: c_int, ratio: c_int };
const QsaDecCfgs = struct { key: QsaDecKey, parts: mlx.mlx_fast_metal_kernel_config, merge: mlx.mlx_fast_metal_kernel_config };
var qsa_dec_cfgs: ?QsaDecCfgs = null;
var qsa_nax_os_cached: ?bool = null;

/// The layer's output from its roped queries `q_rope` `[B, H, S, D]`, this forward's roped keys `k_rope` and values
/// `v_t` (head-major), the output gate `gate` and where it lies for the fused o_proj (`gate_src`): the KV appended to
/// the slot's cache (each slot's, on a batched tick), attention over it, then the gate and o_proj.
pub fn core(self: *Trunk, ctx: *ForwardCtx, q_rope: mlx.mlx_array, k_rope: mlx.mlx_array, v_t: mlx.mlx_array, gate: mlx.mlx_array, gate_src: ?fusion.GateSrc, fa: *const FullAttnWeights, layer: u32, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);

    // A batched tick: each slot owns its cache and kv length, so the append is each slot's and so are the reads.
    if (ctx.batch_slots) |slots| {
        var cache_buf: [MAX_BATCH_ROWS]*KVCache = undefined;
        var picks = false;
        for (slots, 0..) |sc, i| {
            cache_buf[i] = sc.cache;
            picks = picks or sc.qsa_blocks.ctx != null;
        }
        try KVCache.appendFromStacked(k_rope, v_t, cache_buf[0..slots.len], layer, self.s, 0, true);
        // The indexer leaves a group mask only where no slot holds block picks (`qsaMaskBatched`).
        const out = if (ctx.qsa_mask.ctx == null and picks)
            try slotsAttn(self, slots, q_rope, layer, seq_len, ratio, attn_scale)
        else
            try stackedAttn(self, slots, q_rope, layer, seq_len, attn_scale, ctx.qsa_mask);
        defer _ = mlx.mlx_array_free(out);
        return gatedAttnTail(self, out, gate, fa, flat_shape, batch, is_prefill, layer, null);
    }

    var kv_view = try ctx.cache.update(layer, k_rope, v_t, self.s, 0);
    defer kv_view.deinit();
    const out = if (ctx.qsa_mask.ctx != null)
        try maskAttn(self.s, q_rope, kv_view.k, kv_view.v, attn_scale, ctx.qsa_mask)
    else if (ctx.qsa_blocks.ctx != null) blk: {
        // A verify block's split pass, then its merge, the gate's product and o_proj's prepare in one launch.
        if (gate_src) |gs| if (try mergedGatedOut(self, q_rope, &kv_view, ctx.qsa_blocks, gs, fa, flat_shape, batch, seq_len, ratio, attn_scale)) |y| return y;
        break :blk try picksAttn(self.s, q_rope, &kv_view, ctx.qsa_blocks, seq_len, ratio, attn_scale, true);
    } else try denseAttn(self.s, q_rope, kv_view.k, kv_view.v, attn_scale, is_prefill);
    defer _ = mlx.mlx_array_free(out);
    return gatedAttnTail(self, out, gate, fa, flat_shape, batch, is_prefill, layer, gate_src);
}

/// The indexer's choice for one single-sequence call (the QSA front asks): block picks, or the dense mask. A verify
/// block takes picks at any kv (the split pass reads each row's own O(budget) keys), a prefill block past its gather's
/// floor, a decode row past `QSA_GATHER_MIN_KV`.
pub fn qsaWantsBlocks(batch: c_int, kv: c_int, seq_len: c_int) bool {
    if (batch != 1) return false;
    if (seq_len >= PREFILL_ROWS) return kv > qsaPrefillGatherMinKv();
    if (seq_len >= 2) return true;
    return kv > QSA_GATHER_MIN_KV;
}

/// The prefill gather's kv floor: the tensor-unit kernel's where the OS has it, else the stock kernel's.
fn qsaPrefillGatherMinKv() c_int {
    return if (qsaNaxOsOk()) QSA_GATHER_MIN_KV_NAX else QSA_GATHER_MIN_KV;
}

/// The tensor-unit gather wants macOS 26.3 or later (the plugin loads from 26.2).
fn qsaNaxOsOk() bool {
    if (qsa_nax_os_cached) |cached| return cached;
    var buf: [64]u8 = undefined;
    const ok = macosVersionAtLeast(macosProductVersion(&buf) orelse "", 26, 3);
    qsa_nax_os_cached = ok;
    return ok;
}

// ── the gate and o_proj ──

/// The tail: o_proj over the gated product, in o_proj's prepare (`gated_out`) where the gate's source is known, else the
/// attention output back to [B, S, H, D] times sigmoid(gate) (strided views read copy-free: one launch up to 16 rows, a
/// batched tick's), flattened (a free view of the contiguous product) and projected. qwen4_exp always gates its
/// attention output (model_config.zig).
fn gatedAttnTail(self: *Trunk, attn_out: mlx.mlx_array, gate: mlx.mlx_array, fa: *const FullAttnWeights, flat_shape: [3]c_int, batch: c_int, is_prefill: bool, layer: u32, gate_src: ?fusion.GateSrc) !mlx.mlx_array {
    if (gate_src) |gs| if (try gatedOut(self, attn_out, gs, fa, flat_shape, batch)) |y| return y;
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &[_]c_int{ 0, 2, 1, 3 }, 4, self.s));
    var gated_4d: mlx.mlx_array = .{ .ctx = null };
    defer if (gated_4d.ctx != null) {
        _ = mlx.mlx_array_free(gated_4d);
    };
    // The residual norm's 16 rows (`mamba2_decode.MAX_ROWS`): bit for bit the pair below.
    if (flat_shape[0] * flat_shape[1] <= 16) gated_4d = (try stridedSigmoidGateMul(self.s, attn_t, gate)) orelse gated_4d;
    if (gated_4d.ctx == null) {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, self.s));
        gated_4d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gated_4d, attn_t, gate_sig, self.s));
    }
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    try mlx.check(mlx.mlx_reshape(&gated, gated_4d, &flat_shape, 3, self.s));
    return linear.attnProj(self, gated, fa.o_w, fa.o_s, fa.o_b, batch == 1 and !is_prefill, layer);
}

/// The gate's operand for `gated_out` and `merged_gated_out`, with MLX's sigmoid table; null where the call is not one
/// sequence's, or a decode block's rows are past the entries' 16.
fn gateFor(self: *Trunk, dt: mlx.mlx_dtype, gs: fusion.GateSrc, rows: c_int, batch: c_int, flat_batch: c_int, decode: bool) !?abi.Gate {
    if (batch != 1 or flat_batch != 1) return null;
    if (decode and (rows < 1 or rows > fusion.decode_rows_max)) return null;
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(gs.base) != dt) return null;
    const sigtab = try swigluSigTable(self.s, dt, std.heap.c_allocator);
    return .{ .base = fusion.arr(gs.base), .row_stride = gs.row_stride, .head_stride = gs.head_stride, .off = gs.off, .sigtab = fusion.arr(sigtab) };
}

/// The fused attention output: o_proj over T(attn * sigmoid(gate)) through `gated_out`, the product formed in o_proj's
/// prepare. Null: the tail.
fn gatedOut(self: *Trunk, attn_out: mlx.mlx_array, gs: fusion.GateSrc, fa: *const FullAttnWeights, flat_shape: [3]c_int, batch: c_int) !?mlx.mlx_array {
    const rows = flat_shape[1];
    const o = self.formats.linearOf(fa.o_w) orelse return null;
    const ash = mlx.getShape(attn_out);
    if (ash.len != 4 or ash[0] != 1 or ash[2] != rows or ash[1] * ash[3] != flat_shape[2]) return null;
    const gate = (try gateFor(self, mlx.mlx_array_dtype(attn_out), gs, rows, batch, flat_shape[0], rows <= fusion.decode_rows_max)) orelse return null;
    return fusion.callGatedOut(&self.formats, o, attn_out, &gate, self.s);
}

/// The merged attention output at a verify block's rows: the split pass as `picksAttn` runs it, then `merged_gated_out`
/// takes its merge, the gate's product and o_proj's prepare in one launch. Null: the split pass with its merge, then the
/// tail.
fn mergedGatedOut(self: *Trunk, q_rope: mlx.mlx_array, kv_view: *const DenseKVView, blocks: mlx.mlx_array, gs: fusion.GateSrc, fa: *const FullAttnWeights, flat_shape: [3]c_int, batch: c_int, seq_len: c_int, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (seq_len < 2 or seq_len >= PREFILL_ROWS) return null;
    const t = self.formats.trunk orelse return null;
    const call = t.merged_gated_out orelse return null;
    const o = self.formats.linearOf(fa.o_w) orelse return null;
    const gate = (try gateFor(self, mlx.mlx_array_dtype(q_rope), gs, flat_shape[1], batch, flat_shape[0], true)) orelse return null;
    var parts: SplitParts = .{};
    defer parts.deinit();
    if ((try sparseAttn(self.s, q_rope, kv_view, blocks, ratio, attn_scale, &parts)) == null) return null;
    var out: abi.Array = .{};
    if (!call(o, fusion.arr(parts.pacc), fusion.arr(parts.pml), &gate, fusion.strm(self.s), &out).isOk()) return null;
    return fusion.own(out);
}

// ── the arms ──

/// One slot's attention over its block picks: a decode row on the decode kernel (`dec_row`; a batched tick's rows take
/// the split pass), up to 15 rows the split pass and its merge, from 16 the prefill gather. Each serves every K-path
/// shape: a decline is a shape the K packs never make.
fn picksAttn(s: mlx.mlx_stream, q: mlx.mlx_array, kv_view: *const DenseKVView, blocks: mlx.mlx_array, seq_len: c_int, ratio: c_int, attn_scale: f32, dec_row: bool) !mlx.mlx_array {
    const out = if (dec_row and seq_len == 1)
        try qsaDecAttend(s, q, kv_view.k, kv_view.v, blocks, ratio, attn_scale)
    else if (seq_len < PREFILL_ROWS)
        try sparseAttn(s, q, kv_view, blocks, ratio, attn_scale, null)
    else
        try gatherQsa256(s, q, kv_view.k, kv_view.v, attn_scale, blocks, ratio);
    return out orelse error.DirectModelQsaArmDeclined;
}

/// Attention under the indexer's mask (a placed draft row's visibility, or a block of 16 rows or more at its gather's
/// floor): the hd-256 kernel's mask arm from 16 rows, MLX's SDPA otherwise.
fn maskAttn(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSdpa256Masked(s, q, k, v, scale, mask)) |out| return out;
    return sdpa(s, q, k, v, scale, "array", mask, false);
}

/// Attention over the whole cache below the QSA budget: a block of rows causal (under `--int8-qk` the int8 Q/K kernel
/// from 64 rows; else row groups the vector kernel serves at 3..8 rows, MLX's fused NAX kernel from 9), a decode row
/// plain.
fn denseAttn(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, is_prefill: bool) !mlx.mlx_array {
    if (!is_prefill) return sdpa(s, q, k, v, scale, "", .{ .ctx = null }, false);
    if (try int8QkAttn(s, q, k, v, scale)) |out| return out;
    if (try splitCausalSdpa(s, q, k, v, scale)) |out| return out;
    return sdpa(s, q, k, v, scale, "causal", .{ .ctx = null }, sdpaForceFused(q, k));
}

/// MLX's SDPA in `mode` ("", "causal", or "array" under `mask`).
fn sdpa(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mode: [*:0]const u8, mask: mlx.mlx_array, force_fused: bool) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, v, scale, mode, mask, .{ .ctx = null }, force_fused, s));
    return out;
}

// ── the batched tick ──

/// A batched tick where some slot holds block picks: each slot's rows over its own view, through its mask, its picks (the
/// split pass even for a decode row), or causal SDPA below the budget; stacked on the batch axis.
fn slotsAttn(self: *Trunk, slots: []const *ForwardCtx, q_rope: mlx.mlx_array, layer: u32, seq_len: c_int, ratio: c_int, attn_scale: f32) !mlx.mlx_array {
    var outs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (slots, 0..) |sc, i| {
        var view = try sc.cache.denseView(layer, self.s);
        defer view.deinit();
        const q_i = try axisView(self.s, q_rope, 0, i);
        defer _ = mlx.mlx_array_free(q_i);
        outs[n] = if (sc.qsa_mask.ctx != null)
            try maskAttn(self.s, q_i, view.k, view.v, attn_scale, sc.qsa_mask)
        else if (sc.qsa_blocks.ctx != null)
            try picksAttn(self.s, q_i, &view, sc.qsa_blocks, seq_len, ratio, attn_scale, false)
        else
            try sdpa(self.s, q_i, view.k, view.v, attn_scale, "causal", .{ .ctx = null }, false);
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, self.s));
    return stacked;
}

/// The batched tick's dense read: every slot's view zero-padded to the longest and stacked, one SDPA under the
/// additive `[N, 1, S, kv_max]` mask, narrowed by the indexer's group mask `narrow` where it set one.
fn stackedAttn(self: *Trunk, slots: []const *ForwardCtx, q: mlx.mlx_array, layer: u32, seq_len: c_int, attn_scale: f32, narrow: mlx.mlx_array) !mlx.mlx_array {
    var views: [MAX_BATCH_ROWS]DenseKVView = undefined;
    var n: usize = 0;
    defer for (views[0..n]) |*dv| dv.deinit();
    var kv_lens: [MAX_BATCH_ROWS]i32 = undefined;
    var kv_max: c_int = 0;
    for (slots) |sc| {
        views[n] = try sc.cache.denseView(layer, self.s);
        kv_lens[n] = mlx.getShape(views[n].k)[2];
        kv_max = @max(kv_max, kv_lens[n]);
        n += 1;
    }
    const stacked_k = try padAndStack(self.s, views[0..n], true, kv_max);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try padAndStack(self.s, views[0..n], false, kv_max);
    defer _ = mlx.mlx_array_free(stacked_v);
    var mask = try batchedMask(self.s, kv_lens[0..n], kv_max, seq_len, mlx.mlx_array_dtype(q));
    defer _ = mlx.mlx_array_free(mask);
    if (narrow.ctx != null) {
        const neg_inf = try scalarOf(-std.math.inf(f32), mlx.mlx_array_dtype(mask), self.s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&narrowed, narrow, mask, neg_inf, self.s));
        _ = mlx.mlx_array_free(mask);
        mask = narrowed;
    }
    return sdpa(self.s, q, stacked_k, stacked_v, attn_scale, "array", mask, false);
}

/// Each slot's k (or v) view `[1, kv_h, kv_len_i, hd]` zero-padded to `kv_max` along axis 2, concatenated on axis 0.
fn padAndStack(s: mlx.mlx_stream, views: []const DenseKVView, key_not_value: bool, kv_max: c_int) !mlx.mlx_array {
    const pad_value = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(pad_value);
    var padded: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (padded[0..n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (views) |dv| {
        const view = if (key_not_value) dv.k else dv.v;
        padded[n] = mlx.mlx_array_new();
        n += 1;
        try mlx.check(mlx.mlx_pad(&padded[n - 1], view, &[_]c_int{2}, 1, &[_]c_int{0}, 1, &[_]c_int{kv_max - mlx.getShape(view)[2]}, 1, pad_value, "constant", s));
    }
    const vec = mlx.mlx_vector_array_new_data(&padded, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    return stacked;
}

/// The additive `[N, 1, S, kv_max]` mask in `dt` (the queries' dtype: SDPA rejects a mask that does not promote to its
/// output type): row j of slot n sees positions below `kv_len_n - S + 1 + j`, its own causal prefix, never the pad.
fn batchedMask(s: mlx.mlx_stream, kv_lens: []const i32, kv_max: c_int, seq_len: c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_max), 1, .int32, s));
    var pos_4d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_4d);
    try mlx.check(mlx.mlx_reshape(&pos_4d, positions, &[_]c_int{ 1, 1, 1, kv_max }, 4, s));
    var lim_buf: [MAX_BATCH_ROWS * 16]i32 = undefined;
    const rows: usize = @intCast(seq_len);
    if (kv_lens.len * rows > lim_buf.len) return error.BatchedMaskTooWide;
    for (kv_lens, 0..) |kl, n| {
        for (0..rows) |j| lim_buf[n * rows + j] = kl - seq_len + 1 + @as(i32, @intCast(j));
    }
    const lens_4d = mlx.mlx_array_new_data(&lim_buf, &[_]c_int{ @intCast(kv_lens.len), 1, seq_len, 1 }, 4, .int32);
    defer _ = mlx.mlx_array_free(lens_4d);
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, pos_4d, lens_4d, s));
    const zero = try scalarOf(0.0, dt, s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = try scalarOf(-std.math.inf(f32), dt, s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&mask, valid, zero, neg_inf, s));
    return mask;
}

// ── the QSA kernels ──

/// The split pass (`msv_attn_qsa256_dsplit`) over a dense bf16 cache for 1..15 rows, each row its own picks then its own
/// ragged tail, one threadgroup per (row, kv head, split); then the merge (`msv_attn_qsa256_qmerge`), or with `parts`
/// the partials handed over for the fused o_proj prepare's own merge (a served call then returns a null-ctx array).
/// Null: a shape outside the kernel's set.
fn sparseAttn(s: mlx.mlx_stream, q_rope: mlx.mlx_array, kv_view: *const DenseKVView, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32, parts: ?*SplitParts) !?mlx.mlx_array {
    const k = kv_view.k;
    const v = kv_view.v;
    if (k.ctx == null or v.ctx == null or mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q_rope);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    const seq_len = qs[2];
    if (seq_len < 1 or seq_len >= PREFILL_ROWS or qs[0] != 1 or qs[3] != 256 or ks[0] != 1 or ks[3] != 256 or !std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    // `kb` rides `blocks_shape[2]` into the kernel; an input under 8 elements binds in the `constant` address space,
    // where the kernel's `const device int* blk` fails to compile.
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0 or bs[1] * bs[2] < 8 or ratio <= 0) return null;
    if (mlx.mlx_array_strides(q_rope)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;
    const h_kv = ks[1];
    if (h_kv <= 0 or ks[2] < seq_len or @rem(qs[1], h_kv) != 0 or @divTrunc(qs[1], h_kv) > 64) return null;
    const nsg: c_int = @divTrunc(@divTrunc(qs[1], h_kv) + 7, 8);
    // K/V are cache views: the kernel reads their strides, and falls back to element loads off a 16-byte base.
    const kernel = cachedKernel(&qsa_attn_d_kernel, "msv_attn_qsa256_dsplit", &.{ "q", "scl", "blocks", "kq", "vq" }, &.{ "pacc", "pml" }, QSA_ATTN_D_SOURCE, ATTN_QSA256_KERNEL_HEADER) catch return null;
    const merge_kernel = cachedKernel(&qsa_attn_merge_kernel, "msv_attn_qsa256_qmerge", &.{ "pacc", "pml" }, &.{"out"}, QSA_ATTN_MERGE_SOURCE, ATTN_QSA256_KERNEL_HEADER) catch return null;

    const key = QsaAttnCfgKey{ .q_shape = ShapeKey.from(qs), .h_kv = h_kv, .ratio = ratio };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_attn_cfgs.get(key) orelse blk: {
        // The split pass: one threadgroup per (row, kv head, split).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ qs[1], seq_len, QSA_ATTN_NSPLIT, 256 }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ qs[1], seq_len, QSA_ATTN_NSPLIT, 2 }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, seq_len * 32, h_kv * nsg, QSA_ATTN_NSPLIT));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        inline for (.{ .{ "NSG", nsg }, .{ "BK", QSA_ATTN_BK }, .{ "RATIO", ratio }, .{ "NSPLIT", QSA_ATTN_NSPLIT } }) |ta| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, ta[0], ta[1]));
        }
        // The merge: one threadgroup of 256 per (q head, row).
        const mconfig = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(mconfig);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mconfig, &[_]c_int{ 1, qs[1], seq_len, 256 }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mconfig, 256, seq_len, qs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mconfig, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mconfig, "T", .bfloat16));
        qsa_attn_cfgs.put(key, .{ config, mconfig });
        break :blk .{ config, mconfig };
    };

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    var pm: [2]mlx.mlx_array = undefined;
    try launch(s, kernel, cfgs[0], &.{ q_rope, scl, blocks, k, v }, &pm);
    if (parts) |p| {
        p.pacc = pm[0];
        p.pml = pm[1];
        return mlx.mlx_array{ .ctx = null };
    }
    defer for (pm) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var out: [1]mlx.mlx_array = undefined;
    try launch(s, merge_kernel, cfgs[1], &pm, &out);
    return out[0];
}

/// The decode kernel's two configs for `key` (the last key's are kept; a new key frees them).
fn qsaDecConfigs(key: QsaDecKey, ns: c_int) !QsaDecCfgs {
    if (qsa_dec_cfgs) |c| {
        if (std.meta.eql(c.key, key)) return c;
        _ = mlx.mlx_fast_metal_kernel_config_free(c.parts);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.merge);
        qsa_dec_cfgs = null;
    }
    const gqa = @divExact(key.hq, key.hk);
    const parts = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, QSA_DEC_BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(parts, ns * gqa * 32, key.hk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(parts, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(parts, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", key.kb }, .{ "RATIO", key.ratio }, .{ "CH", QSA_DEC_CH }, .{ "NS", ns } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(parts, ta[0], ta[1]));
    }
    const merge = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge, &[_]c_int{ 1, key.hq, 1, QSA_DEC_BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge, QSA_DEC_BD * key.hq, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge, QSA_DEC_BD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(merge, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge, "NS", ns));
    qsa_dec_cfgs = .{ .key = key, .parts = parts, .merge = merge };
    return qsa_dec_cfgs.?;
}

/// One decode row's attention `[1, Hq, 1, 256]` over its picks `blocks` `[1, 1, kb]` (RATIO tokens each) and its
/// incomplete tail, straight from the dense cache views `[1, Hk, kv, 256]`: a split pass in which a simdgroup owns one q
/// head and `CH` of the visible keys, then the merge. Null: a shape outside the kernel's set.
fn qsaDecAttend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    if (qs[0] != 1 or qs[2] != 1 or qs[3] != QSA_DEC_BD or ks[0] != 1 or ks[3] != QSA_DEC_BD or bs[0] != 1 or bs[1] != 1) return null;
    const hq = qs[1];
    const hk = ks[1];
    // An input under 8 elements binds in `constant` space, where the kernel's `device int*` fails to compile.
    if (hk <= 0 or ratio <= 0 or bs[2] < 8 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(QSA_DEC_CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;
    // K/V are cache views: the kernels read their strides.
    const parts_kernel = try cachedKernel(&qsa_dec_parts_kernel, "msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, QSA_DEC_PARTS_SOURCE, QSA_DEC_HEADER);
    const merge_kernel = try cachedKernel(&qsa_dec_merge_kernel, "msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, QSA_DEC_MERGE_SOURCE, QSA_DEC_HEADER);
    const ns = @divTrunc(bs[2] * ratio + ratio - 1 + QSA_DEC_CH - 1, QSA_DEC_CH);
    const c = try qsaDecConfigs(.{ .hq = hq, .hk = hk, .kb = bs[2], .ratio = ratio }, ns);

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    var pm: [2]mlx.mlx_array = undefined;
    try launch(s, parts_kernel, c.parts, &.{ q, scl, blocks, k, v }, &pm);
    defer for (pm) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var out: [1]mlx.mlx_array = undefined;
    try launch(s, merge_kernel, c.merge, &pm, &out);
    return out[0];
}

/// The prefill gather over the block picks (`blocks` [B, S, KB] int32, each row ascending with INT_MAX past its count,
/// `ratio` tokens a block): O(budget) keys a token, the tensor-unit `msv_qsa_nax_precise` where the call fits it on macOS
/// 26.3 or later, `msv_attn_qsa256` otherwise. Null: a shape outside the kernels' set.
fn gatherQsa256(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, blocks: mlx.mlx_array, ratio: c_int) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(blocks) != 3 or mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const bs = mlx.getShape(blocks);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256 or ks[1] <= 0 or @rem(qs[1], ks[1]) != 0 or @divTrunc(qs[1], ks[1]) > 64) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (bs[0] != qs[0] or bs[1] != qs[2] or bs[2] <= 0 or ratio <= 0) return null;
    // The stock kernel stages K/V through `uint4` loads: an unaligned materialized view base reads wrong values.
    if (!attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;
    const qst = mlx.mlx_array_strides(q);
    const kst = mlx.mlx_array_strides(k);
    const vst = mlx.mlx_array_strides(v);
    const nax_aligned = qst[3] == 1 and kst[3] == 1 and vst[3] == 1 and mlx.mlx_array_strides(blocks)[2] == 1 and
        qst[1] % 8 == 0 and qst[2] % 8 == 0 and kst[1] % 8 == 0 and kst[2] % 8 == 0 and vst[1] % 8 == 0 and vst[2] % 8 == 0;
    // The copy's one-time probe of the NAX kernel (against the stock kernel and an f32 reference, at load) is pinned to
    // its result on M5: 677 served logs engage msv_qsa_nax_precise and none says "probe mismatch". The OS floor stays.
    const nax = nax_aligned and ratio == 4 and qs[0] == 1 and qs[1] == 24 and ks[1] == 2 and bs[2] == 512 and qs[2] >= PREFILL_ROWS and qsaNaxOsOk();
    const names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks" };
    const kernel = (if (nax)
        cachedKernel(&qsa_nax_kernel, "msv_qsa_nax_precise", &names, &.{"out"}, QSA_NAX_SOURCE, QSA_NAX_HEADER)
    else
        cachedKernel(&attn_qsa256_kernel, "msv_attn_qsa256", &names, &.{"out"}, ATTN_QSA256_KERNEL_SOURCE, ATTN_QSA256_KERNEL_HEADER)) catch return null;
    const nsg: c_int = @divTrunc(@divTrunc(qs[1], ks[1]) + 7, 8);
    const key = QsaGatherCfgKey{ .q_shape = ShapeKey.from(qs), .h_kv = ks[1], .ratio = ratio, .nax = nax };
    const cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_gather_cfgs.get(key) orelse blk: {
        // One threadgroup per (query token, kv head, batch).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ qs[0], qs[1], qs[2], 256 }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, qs[2] * 32, ks[1] * nsg, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        inline for (.{ .{ "NSG", nsg }, .{ "BK", QSA_GATHER_BK }, .{ "RATIO", ratio } }) |ta| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, ta[0], ta[1]));
        }
        qsa_gather_cfgs.put(key, .{config});
        break :blk .{config};
    };
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    var out: [1]mlx.mlx_array = undefined;
    try launch(s, kernel, cfgs[0], &.{ q, k, v, scl, blocks }, &out);
    return out[0];
}

// ── msv_attn_p256's mask arm (attn256.zig) ──

/// msv_attn_p256 under the indexer's mask for a block of 16 rows or more: `mask` the [B, 1, qL, kL] bool visibility with
/// causal folded in; key blocks no row of a 64-row q tile sees are skipped through a per-tile any-visible table. The key
/// axis runs in budget-sized dispatches whose fp32 m / l / O carry keeps the result one dispatch's. On macOS 26.2 this
/// serves every prefill block at kv 2052..8192 (the stock gather's floor): MLX's SDPA there would build an unbilled
/// [Hq, S, kv] score tensor. Null under 16 rows (MLX's vector kernel's) or outside its shapes.
fn fusedSdpa256Masked(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: mlx.mlx_array) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4 or mlx.mlx_array_ndim(mask) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const ms = mlx.getShape(mask);
    if (mlx.mlx_array_dtype(mask) != .bool_ or ms[0] != qs[0] or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256 or qs[2] < PREFILL_ROWS or ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return null;
    // K/V stage through `uint4` loads: an unaligned materialized view base reads wrong values.
    if (!attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;
    // ensure_row_contiguous off: K/V are cache views, and a contiguous copy of the whole cache per layer erases the win.
    const kernel = cachedKernel(&attn256_kernel, "msv_attn_p256", &.{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" }, &.{ "out", "m_out", "l_out", "o_out" }, ATTN256_KERNEL_SOURCE, ATTN256_KERNEL_HEADER) catch return null;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    // qwen has no sliding band (window 0); a 1-element f32 stands in for the absent carries and learned sinks.
    const win_data = [_]i32{0};
    const win = mlx.mlx_array_new_data(&win_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(win);
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    const skip = try qsaSkipTable(s, mask);
    defer _ = mlx.mlx_array_free(skip);

    const kL = ks[2];
    const chunk_len = fused256KvChunkLen(qs[0], qs[1], qs[2], kL);
    // m, l and O of the dispatches so far.
    var carry: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (carry) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    var result: mlx.mlx_array = .{ .ctx = null };
    var k0: c_int = 0;
    while (k0 < kL) : (k0 += chunk_len) {
        const k1: c_int = @min(k0 + chunk_len, kL);
        const final = k1 == kL;
        const has_carry = k0 > 0;
        const kr_data = [_]i32{ @intCast(k0), @intCast(k1) };
        const kr = mlx.mlx_array_new_data(&kr_data, &[_]c_int{2}, 1, .int32);
        defer _ = mlx.mlx_array_free(kr);
        const phase_data = [_]i32{(if (has_carry) @as(i32, 1) else 0) | (if (final) @as(i32, 2) else 0)};
        const phase = mlx.mlx_array_new_data(&phase_data, &one, 1, .int32);
        defer _ = mlx.mlx_array_free(phase);

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
        const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
        if (final) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .float32));
        }
        // One threadgroup (32, 8, 1) per 64-row q tile per head per batch; the grid in threads, padded to whole tiles
        // (the cooperative staging loops assume 256 threads).
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(qs[2] + 63, 64) * 32, qs[1] * 8, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        inline for (.{ .{ "QSA", 1 }, .{ "BDK", 256 }, .{ "BDV", 256 }, .{ "SINK", 0 } }) |ta| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, ta[0], ta[1]));
        }
        const inputs = [_]mlx.mlx_array{
            q,                                  k,                                  v,     scl, win, kr, phase,
            if (has_carry) carry[0] else dummy, if (has_carry) carry[1] else dummy, if (has_carry) carry[2] else dummy,
            mask,                               skip,                               dummy,
        };
        var outs: [4]mlx.mlx_array = undefined;
        try launch(s, kernel, config, &inputs, &outs);
        if (final) {
            result = outs[0];
            for (outs[1..]) |a| _ = mlx.mlx_array_free(a);
        } else {
            _ = mlx.mlx_array_free(outs[0]);
            for (&carry, outs[1..]) |*c, a| {
                if (c.ctx != null) _ = mlx.mlx_array_free(c.*);
                c.* = a;
            }
        }
    }
    return result;
}

/// The mask arm's per-(batch, q tile, key block) any-visible table: [B, ceil(qL/64), ceil(kL/32)] bool.
fn qsaSkipTable(s: mlx.mlx_stream, mask: mlx.mlx_array) !mlx.mlx_array {
    const ms = mlx.getShape(mask);
    const nq: c_int = @divTrunc(ms[2] + 63, 64);
    const nk: c_int = @divTrunc(ms[3] + 31, 32);
    const falsev = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(falsev);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, mask, &[_]c_int{ 2, 3 }, 2, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ nq * 64 - ms[2], nk * 32 - ms[3] }, 2, falsev, "constant", s));
    var r5 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r5);
    try mlx.check(mlx.mlx_reshape(&r5, padded, &[_]c_int{ ms[0], nq, 64, nk, 32 }, 5, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_any_axes(&out, r5, &[_]c_int{ 2, 4 }, 2, false, s));
    return out;
}

/// The key-axis chunk of one dispatch: the largest 32-multiple keeping batch·hq·ql·chunk within the budget, at least 32,
/// at most kl.
fn fused256KvChunkLen(batch: c_int, hq: c_int, ql: c_int, kl: c_int) c_int {
    const per_key: i64 = @as(i64, batch) * hq * ql;
    const chunk = @divTrunc(@divTrunc(FUSED256_DISPATCH_BUDGET, @max(per_key, 1)), 32) * 32;
    return @intCast(@min(@max(chunk, 32), kl));
}

/// A bf16 view's bound address is 16-byte aligned (every `uint4` staging load's bar). A lazy array has no base yet (the
/// kernel binds its buffer at dispatch); an available one returns the address the kernel reads.
fn attn256Base16Aligned(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0 or !avail) return true;
    const p = mlx.mlx_array_data_uint32(a) orelse return true;
    return @intFromPtr(p) % 16 == 0;
}

// ── `--int8-qk`: msv_nax_attn_256_q8 (nax_attention.zig) ──

/// Causal hd-256 attention on the tensor units with int8 Q and K (SageAttention; `--int8-qk`, lossy): Q quantized per
/// 64-row tile, K per 32-key block after removing each KV head's mean key (a constant offset on every key moves each
/// score row by a constant, which softmax ignores), P and V in q's dtype, one dispatch over every key; the output a
/// [B, H, qL, 256] view like SDPA's. Null: the switch off, or a block under one Q tile (verify widths and short blocks
/// keep the kernels below). Any other shape it declines is one the K packs never make.
fn int8QkAttn(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!int8_qk) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return error.DirectModelInt8QkDeclined;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256) return error.DirectModelInt8QkDeclined;
    if (ks[0] != qs[0] or vs[0] != qs[0] or vs[1] != ks[1] or vs[2] != ks[2]) return error.DirectModelInt8QkDeclined;
    if (ks[1] == 0 or @rem(qs[1], ks[1]) != 0 or ks[2] < qs[2]) return error.DirectModelInt8QkDeclined;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return error.DirectModelInt8QkDeclined;
    if (qs[2] < Q8_BQ) return null;

    // Each KV head's mean key; then Q's int8 codes and scales, and those of K less that mean.
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, k, 2, true, s));
    const qq = try int8QuantRows(s, q, Q8_BQ, null);
    defer for (qq) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const kq = try int8QuantRows(s, k, Q8_BK, mean);
    defer for (kq) |a| {
        _ = mlx.mlx_array_free(a);
    };

    const b = qs[0];
    const h = qs[1];
    const ql = qs[2];
    const kl = ks[2];
    const nq = @divTrunc(ql + Q8_BQ - 1, Q8_BQ);
    const nq_aligned = @divTrunc(ql, Q8_BQ);
    const nk = @divTrunc(kl + Q8_BK - 1, Q8_BK);
    const nk_aligned = @divTrunc(kl, Q8_BK);
    // No sinks and no row state carried in: one-element stand-ins, the sinks' in q's dtype.
    const one = [_]c_int{1};
    const zero = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&zero, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    var sinks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks);
    try mlx.check(mlx.mlx_astype(&sinks, dummy, dt, s));
    // omlx_nax::AttnParams: B H D qL kL gqa scale NQ NK NQ_aligned NK_aligned qL_rem kL_rem qL_off kb_begin kb_end.
    const pdata = [16]i32{ b, h, 256, ql, kl, @divExact(h, ks[1]), @bitCast(scale), nq, nk, nq_aligned, nk_aligned, ql - nq_aligned * Q8_BQ, kl - nk_aligned * Q8_BK, kl - ql, 0, nk };
    const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{16}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ b, ql, h, 256 }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
    // One threadgroup of 4 x 2 simdgroups per 64-row q tile per head per batch; the grid in threads.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq * 32 * Q8_WM * 2, h, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * Q8_WM * 2, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "WM", Q8_WM));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "ALIGN_Q", @rem(ql, Q8_BQ) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "ALIGN_K", @rem(kl, Q8_BK) == 0));
    const kernel = try cachedKernel(&nax_attn_256q8_kernel, "msv_nax_attn_256_q8", &.{ "q", "k", "v", "sinks", "state", "params", "qi", "ki", "qs", "ks" }, &.{ "out", "state_out" }, NAX_ATTN_256Q8_SOURCE, NAX_ATTN_HEADER);
    var outs: [2]mlx.mlx_array = undefined;
    try launch(s, kernel, config, &.{ q, k, v, sinks, dummy, params, qq[0], kq[0], qq[1], kq[1] }, &outs);
    defer for (outs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    // [B, qL, H, 256] rows viewed as [B, H, qL, 256].
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_transpose_axes(&view, outs[0], &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    log.debug("[attn] int8 QK call qL={d} kL={d}\n", .{ ql, kl });
    if (!int8_qk_engaged) {
        int8_qk_engaged = true;
        log.info("[attn] int8 QK hd-256 prefill attention engaged (--int8-qk; lossy)\n", .{});
    }
    return view;
}

/// int8_attn.zig's row quantizer `msv_int8_quant_rows`, its strided arm (x read through its strides, one block a
/// threadgroup): the int8 codes of x [B, H, L, D] less `mean` [B, H, 1, D] where given, and one fp32 scale per `rows`-row
/// block, [B * H * blocks], absmax / 127 and at least 1e-6 / 127; a code is x * (1 / scale) rounded. The caller frees
/// both.
fn int8QuantRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, mean: ?mlx.mlx_array) ![2]mlx.mlx_array {
    const kernel = try cachedKernel(&int8_quant_rows_kernel, "msv_int8_quant_rows", &.{ "x", "mean", "eps" }, &.{ "q", "scales" }, INT8_QUANT_ROWS_SOURCE, "");
    const sh = mlx.getShape(x);
    const blocks = sh[0] * sh[1] * @divTrunc(sh[2] + rows - 1, rows);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, sh.ptr, 4, .int8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{blocks}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, blocks * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SG", 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "ROWS", rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "HAS_MEAN", mean != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "DIVIDE", false));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "CONTIG", false));
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    var out: [2]mlx.mlx_array = undefined;
    try launch(s, kernel, config, &.{ x, mean orelse x, eps }, &out);
    return out;
}

// ── SDPA below the QSA budget (sdpa.zig) ──

/// MLX's fused full-attention kernel on NAX at hd 256 (`force_fused`), on every K-path GPU: legal from 9 query rows (8
/// or fewer is the vector kernel, whose qL * gqa <= 32 wall makes force_fused throw) with q <= k.
fn sdpaForceFused(q: mlx.mlx_array, k: mlx.mlx_array) bool {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    return qs[3] == 256 and qs[2] > 8 and qs[2] <= ks[2];
}

/// A causal block of 3..8 rows past the vector kernel's qL * gqa <= 32 wall (hd 256; measured at gqa 12): row groups of
/// 32 / gqa that it serves, group [r0, r1) "causal" over keys[0 .. kL - (qL - r1)] (bottom-right alignment gives row r
/// the window kL - qL + r), concatenated. Null where one call is already fast (2 rows; from 9, the fused kernel).
fn splitCausalSdpa(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256 or qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[1] < 1 or @rem(qs[1], ks[1]) != 0) return null;
    const qL = qs[2];
    const kL = ks[2];
    const gqa = @divTrunc(qs[1], ks[1]);
    if (qL < 2 or qL > 8 or qL * gqa <= 32) return null;
    const group: c_int = @min(8, @max(1, @divTrunc(32, gqa)));
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var parts: [8]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (parts[0..n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        const k_end: c_int = kL - (qL - r1);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var k_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_g);
        var v_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, 256 }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&k_g, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], k_end, 256 }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&v_g, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, vs[1], k_end, 256 }, 4, &strides, 4, s));
        parts[n] = try sdpa(s, q_g, k_g, v_g, scale, "causal", .{ .ctx = null }, false);
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

// ── kernels ──

/// A Metal kernel over `ins` / `outs`, built once into `cache`: `ensure_row_contiguous` and `atomic_outputs` off (K/V
/// are cache views; each kernel reads its inputs' strides).
fn cachedKernel(cache: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (cache.*) |built| return built;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    cache.* = kernel;
    return kernel;
}

/// `kernel` over `inputs` under `config`: its outputs into `outs`, each the caller's.
fn launch(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, config, s));
    if (mlx.mlx_vector_array_size(out_vec) != outs.len) return error.MetalKernelBadOutputCount;
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, out_vec, i)) catch |e| {
            for (outs[0..i]) |a| _ = mlx.mlx_array_free(a);
            return e;
        };
    }
}
