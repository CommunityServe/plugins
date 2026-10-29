//! copied_from: src/plugins/qwen4_exp/qsa/pooled.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The pooled indexer keys: their rope (`QsaPooledRope`, the pool-rope kernel) and the pointer-keyed cache invalidated
//! by an atomic mark.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const testRandUniformBf16 = @import("../../../sdk/testing/parity.zig").testRandUniformBf16;
const qsaProbeAllFinite = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeAllFinite;
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const tieBias = @import("../../../sdk/blocks/attention/sparse_block/select.zig").tieBias;

/// The QSA pooled-block RoPE tables, shared by every full-attention layer and across
/// forwards while the key holds. The key is the block position (`base`, `step`, `n`, dtype)
/// plus, on the M-RoPE arm, the position table compared by pointer and length; the pointer
/// is never dereferenced. The owner slot frees that table only after
/// `markQsaPooledRopeStale`, so a same-address table cannot hit. Connection threads write
/// only the atomic `stale`; the inference thread frees.
pub const QsaPooledRope = struct {
    base: c_int = 0,
    step: c_int = 0,
    n: c_int = 0,
    dtype: mlx.mlx_dtype = .float32,
    mrope: bool = false,
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    cos: mlx.mlx_array = .{ .ctx = null },
    sin: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,
    stale: std.atomic.Value(bool) = .init(false),

    pub fn deinit(self: *QsaPooledRope) void {
        if (self.cos.ctx != null) _ = mlx.mlx_array_free(self.cos);
        if (self.sin.ctx != null) _ = mlx.mlx_array_free(self.sin);
        self.cos = .{ .ctx = null };
        self.sin = .{ .ctx = null };
        self.mrope_pos = null;
        self.stale.store(false, .monotonic);
    }
};

pub fn qsaPooledTableEq(a: ?[]const i32, b: ?[]const i32) bool {
    const aa = a orelse return b == null;
    const bb = b orelse return false;
    return aa.ptr == bb.ptr and aa.len == bb.len;
}

/// Does every complete-block visibility check pass for every query row of a call starting at
/// cache position `offset`? Block `b` is complete at `p` iff `b*ratio + ratio - 1 <= p`, so
/// the whole sheet is all-true iff `offset >= nb*ratio-1`. Always true at decode width.
pub fn qsaAllBlocksVisible(offset: c_int, nb: c_int, ratio: c_int) bool {
    return offset >= nb * ratio - 1;
}

/// The nb-keyed constants of the QSA block selection, built once per (nb, ratio) instead of
/// once per full-attention layer. A key mismatch rebuilds.
pub const QsaBlockConsts = struct {
    nb: c_int = 0,
    ratio: c_int = 0,
    /// int32 `[1, nb]`: the last cache position of each complete block.
    blk_end2: mlx.mlx_array = .{ .ctx = null },
    /// f32 `[nb]`: `b * 1e-7`, the index-descending bias reproducing torch.topk's lower-index-wins.
    tie_bias: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,

    pub fn deinit(self: *QsaBlockConsts) void {
        if (self.blk_end2.ctx != null) _ = mlx.mlx_array_free(self.blk_end2);
        if (self.tie_bias.ctx != null) _ = mlx.mlx_array_free(self.tie_bias);
        self.blk_end2 = .{ .ctx = null };
        self.tie_bias = .{ .ctx = null };
        self.nb = 0;
        self.ratio = 0;
    }

    /// After this returns, `blk_end2`/`tie_bias` are borrowed handles valid until the next `ensure` with a different key.
    pub fn ensure(self: *QsaBlockConsts, s: mlx.mlx_stream, nb: c_int, ratio: c_int) !void {
        if (self.blk_end2.ctx != null and self.nb == nb and self.ratio == ratio) return;
        self.deinit();
        errdefer self.deinit();
        var blk_end = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_end);
        try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
        self.blk_end2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&self.blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
        self.tie_bias = try tieBias(s, nb);
        self.nb = nb;
        self.ratio = ratio;
        self.builds += 1;
    }
};

pub fn markQsaPooledRopeStale(self: *Trunk) void {
    self.qsa_pooled_rope.stale.store(true, .release);
}

// ── Fused QSA pooled-key upkeep ──
pub var qsa_idx_rope_fused_override: ?bool = null;

/// The test seam's false runs the composed pooled-key chain.
pub fn qsaIdxRopeFusedEnabled() bool {
    return qsa_idx_rope_fused_override orelse true;
}

// Mirrors each rounding of the composed chain it replaces, so it is bit-identical:
//   mean  = f32 sum in row order r = 0..R-1, times f32(1/R) (MLX
//           NumberOfElements), then bf16. That order is MLX's only while
//           R <= QSA_POOL_SERIAL_MAX_R (see there);
//   norm  = `rms_single_row` at D = 128 (32 lanes x 4 reads, simd_sum, precise
//           rsqrt, `w * T(x * inv)`);
//   rope  = the bf16 cos/sin table and bf16(bf16(x*cos) + bf16(rot*sin)), with
//           each product and the sum rounded as MLX's separate binary kernels do.
// One simdgroup per pooled row; lane l owns dims 4l..4l+3, and rotate_half's
// partner (d +- HALF) is HALF/4 lanes away at the same slot. The block count
// `nblk` varies per call (1 at decode, up to 512 per prefill chunk), so it is a
// scalar INPUT; the templates (T, R, RD, HALF) are fixed per model and name the
// one pipeline MLX compiles.
pub const QSA_POOL_ROPE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_pool_rope.metal");

/// Largest block ratio whose f32 block sum MLX folds in row order 0..R-1, the
/// order the kernel uses. The mean over axis 2 of `[B, nb, R, 128]` is a strided
/// reduce; below 32 rows `strided_reduce_general_dispatch` picks
/// `col_reduce_small` with threadgroup_y = min(8, R). Up to 8 rows each y-thread
/// owns ONE row and lid.y 0 folds them 0..R-1 (R == 1 is a copy). At 9..31 each
/// thread sums rows y, y+8, ... first, and at 32+ the looped/2pass kernels
/// reduce in simdgroup trees, so a sum that cancels can differ in f32.
pub const QSA_POOL_SERIAL_MAX_R: c_int = 8;

var qsa_pool_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_pool_rope_engaged: bool = false;
const QsaPoolRopeCfgKey = struct { b: c_int, nb: c_int, r: c_int, rd: c_int };
/// Int template values handed to MLX so far (T is always bf16); each distinct
/// tuple is a pipeline compile.
const QsaPoolRopeTpl = [4]c_int;
var qsa_pool_rope_tpls: [64]QsaPoolRopeTpl = undefined;
var qsa_pool_rope_tpl_count: usize = 0;

fn qsaPoolRopeNoteTemplate(t: QsaPoolRopeTpl) void {
    const kept = @min(qsa_pool_rope_tpl_count, qsa_pool_rope_tpls.len);
    for (qsa_pool_rope_tpls[0..kept]) |seen| {
        if (std.meta.eql(seen, t)) return;
    }
    // Past the table, every unseen tuple still counts: an overcount, never a miss.
    if (kept < qsa_pool_rope_tpls.len) qsa_pool_rope_tpls[kept] = t;
    qsa_pool_rope_tpl_count += 1;
}
var qsa_pool_rope_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qsa_pool_rope_cfg_key: QsaPoolRopeCfgKey = std.mem.zeroes(QsaPoolRopeCfgKey);

fn getQsaPoolRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_pool_rope_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "kb", "norm_w", "cosv", "sinv", "eps", "inv_r", "nblk" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_pool_rope", in_vec, out_vec, QSA_POOL_ROPE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_pool_rope_kernel = kernel;
    return kernel;
}

/// Pooled block keys in one launch: f32 block mean → bf16 → rms_norm(w) →
/// partial RoPE with the bf16 cos/sin table `[nb, rope_dims]`. Null → the
/// caller runs the composed chain (`qsaPoolNorm` + `qsaPooledRopeComposed`).
pub fn qsaPoolNormRopeFused(
    s: mlx.mlx_stream,
    kb4: mlx.mlx_array, // [B, nb, R, D] bf16
    norm_w: mlx.mlx_array, // [D] bf16
    eps: f32,
    cosv: mlx.mlx_array, // [nb, rope_dims] bf16
    sinv: mlx.mlx_array,
    rope_dims: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ksh = mlx.getShape(kb4);
    const wsh = mlx.getShape(norm_w);
    const csh = mlx.getShape(cosv);
    if (ksh.len != 4 or ksh[3] != 128 or ksh[1] < 1 or ksh[2] < 1 or ksh[2] > QSA_POOL_SERIAL_MAX_R) return null;
    if (wsh.len != 1 or wsh[0] != 128) return null;
    // rotate_half's partner must sit whole lanes away: HALF a multiple of 4.
    if (rope_dims <= 0 or rope_dims > 128 or @mod(rope_dims, 8) != 0) return null;
    if (csh.len != 2 or csh[0] != ksh[1] or csh[1] != rope_dims) return null;
    if (!std.mem.eql(c_int, csh, mlx.getShape(sinv))) return null;
    inline for (.{ kb4, norm_w, cosv, sinv }) |arr| {
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return null;
    }
    const key = QsaPoolRopeCfgKey{ .b = ksh[0], .nb = ksh[1], .r = ksh[2], .rd = rope_dims };
    if (qsa_pool_rope_cfg == null or !std.meta.eql(qsa_pool_rope_cfg_key, key)) {
        if (qsa_pool_rope_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        qsa_pool_rope_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const out_shape = [_]c_int{ ksh[0], ksh[1], 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, ksh[0] * ksh[1], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        const tpl_names = [_][*:0]const u8{ "R", "RD", "HALF" };
        const tpl_vals = [_]c_int{ ksh[2], rope_dims, @divExact(rope_dims, 2) };
        var noted: QsaPoolRopeTpl = .{ 0, 0, 0, 0 };
        for (tpl_names, tpl_vals, 0..) |n, v, i| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, n, v));
            noted[i] = v;
        }
        qsaPoolRopeNoteTemplate(noted);
        qsa_pool_rope_cfg = config;
        qsa_pool_rope_cfg_key = key;
    }
    const kernel = try getQsaPoolRopeKernel();
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    // MLX's mean multiplies by NumberOfElements: f32(1.0 / n) computed in f64.
    const inv_r = mlx.mlx_array_new_float(@floatCast(1.0 / @as(f64, @floatFromInt(ksh[2]))));
    defer _ = mlx.mlx_array_free(inv_r);
    const nblk = mlx.mlx_array_new_int(ksh[1]);
    defer _ = mlx.mlx_array_free(nblk);
    const inputs_arr = [_]mlx.mlx_array{ kb4, norm_w, cosv, sinv, eps_a, inv_r, nblk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qsa_pool_rope_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!qsa_pool_rope_engaged) {
        qsa_pool_rope_engaged = true;
        log.info("[qsa] fused pooled-key upkeep engaged (blocks={d} ratio={d} rope={d})\n", .{ ksh[1], ksh[2], rope_dims });
    }
    return out;
}

// ── tests ──
