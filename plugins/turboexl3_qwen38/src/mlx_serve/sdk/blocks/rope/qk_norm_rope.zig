//! copied_from: src/sdk/blocks/rope/qk_norm_rope.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/rope/qk_norm_rope.
//! Owns QkAngleSlot, qk_fused_kernel, qk_fused_engaged, qk_fused_override, qk_fused_env, qkNormRopeFusedEnabled, and 18 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const mlxSettledActiveBytes = @import("../../testing/parity.zig").mlxSettledActiveBytes;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;

pub const QkAngleSlot = struct {
    offset: c_int = -1,
    rd: c_int = 0,
    rows: c_int = 0,
    scale: f32 = 1.0,
    arr: mlx.mlx_array = .{ .ctx = null },
};

// ── Fused decode QK-norm+RoPE (mlxfast 9e06de6 class) ──
//
// At decode the chain q/k reshape → per-head RMSNorm → transpose → RoPE
// (→ YaRN mscale on Laguna full layers) is strictly serial — SDPA needs its
// output and nothing overlaps it — and costs 4-6 dispatches per layer for a
// few KB of work. One 32-thread-per-head kernel does all of it. cos/sin are
// never re-derived: a per-token PROBE row (ones|zeros through the stock
// `mlx_fast_rope` at the token's offset) extracts the exact floats the stock
// kernel would use, cached per rope family for the forward (2 probe
// dispatches replace ~160 removed ones). Bit-identity with the composed
// chain is the acceptance bar (`qk norm rope fused` parity tests).
var qk_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk_fused_engaged: bool = false;
pub var qk_fused_override: ?bool = null;

pub fn qkNormRopeFusedEnabled() bool {
    return qk_fused_override orelse true;
}

// The RMS half mirrors `rms_single_row` at axis 128 / N_READS 4 / one
// simdgroup (lane l owns [4l, 4l+4), f32 squares in index order, `simd_sum`,
// `precise::rsqrt`); the T(x*inv) rounding inside the w-multiply is the value
// the separate kernel would have written and the rotation would have read
// back. The rotation mirrors `rope_single_impl<T, false>`: pair p couples
// elements p and p+RD/2, partner values arrive by simd_shuffle (RD/8 lanes
// apart), f32 arithmetic, one T rounding — then the optional mscale multiply
// reproduces the composed path's separate broadcast multiply (one more T
// rounding, table value 1.0 on the pass-through tail).
pub const QK_NORM_ROPE_SOURCE = kernelSource("kernels/rope/qk_norm_rope.metal");

fn getQkNormRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qk_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "qw", "kw", "angles", "ms", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_qk_norm_rope",
        in_vec,
        out_vec,
        QK_NORM_ROPE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk_fused_kernel = kernel;
    return kernel;
}

const QkFusedCfgKey = struct { hq: c_int, hk: c_int, rd: c_int, scaled: bool, dtype: mlx.mlx_dtype };
/// Two configs: NoPE (rd 0) and rotary layers alternate within one forward (Flint, Kolibri-1).
var qk_fused_cfgs: [2]struct { cfg: ?mlx.mlx_fast_metal_kernel_config = null, key: QkFusedCfgKey = std.mem.zeroes(QkFusedCfgKey) } = .{ .{}, .{} };
var qk_fused_next: usize = 0;

/// One fused dispatch: per-head RMSNorm + RoPE (+ optional mscale) for q AND
/// k, emitting the transposed [1,H,1,128] layout attention consumes (at L==1
/// that is the flat buffer relabeled, so the transposes vanish too).
/// `angles` is the [RD] f32 cos|sin row from `ropeAngleRow`. Null → caller
/// keeps the composed chain. Shared with tts.zig's talker decode (hd 128).
pub fn fusedQkNormRope(
    s: mlx.mlx_stream,
    q_flat: mlx.mlx_array, // [1, 1, HQ*128]
    k_flat: mlx.mlx_array, // [1, 1, HK*128]
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    ms: ?mlx.mlx_array, // bf16 [128] mscale table, null = unscaled
    eps_arr: mlx.mlx_array, // 0-dim f32 SCALAR (binds `constant float&`; a [1] array binds a pointer and the kernel fails to compile)
    hq: c_int,
    hk: c_int,
    rd: c_int,
) !?[2]mlx.mlx_array {
    if (!qkNormRopeFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(q_flat);
    if (dt != .bfloat16 or mlx.mlx_array_dtype(k_flat) != .bfloat16) return null;
    // rd=32 is the qwen3.5/3.6 partial-rotary geometry (0.25 x 128); the
    // shuffle mapping (lane ^ (half_rd/4)) is exact for all three widths.
    // rd=0 is a NoPE layer: every element takes the pass-through branch, `angles` is never read.
    if (rd != 0 and rd != 32 and rd != 64 and rd != 128) return null;
    if (q_norm_w.ctx == null or k_norm_w.ctx == null) return null;

    const key = QkFusedCfgKey{ .hq = hq, .hk = hk, .rd = rd, .scaled = ms != null, .dtype = dt };
    const slot = for (&qk_fused_cfgs) |*c| {
        if (c.cfg != null and std.meta.eql(c.key, key)) break c;
    } else blk: {
        const c = &qk_fused_cfgs[qk_fused_next];
        qk_fused_next ^= 1;
        if (c.cfg) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
        c.cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, 1, 128 };
        const ok_shape = [_]c_int{ 1, hk, 1, 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk) * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SCALED", if (ms != null) 1 else 0));
        c.* = .{ .cfg = config, .key = key };
        break :blk c;
    };

    const kernel = try getQkNormRopeKernel();
    // The unscaled variant never reads `ms` (SCALED == 0 guards the index),
    // but the input slot must be bound — the norm weight stands in.
    const inputs_arr = [_]mlx.mlx_array{ q_flat, k_flat, q_norm_w, k_norm_w, angles, ms orelse q_norm_w, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, slot.cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var oq = mlx.mlx_array_new();
    var ok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&oq, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ok, outputs_vec, 1));
    if (!qk_fused_engaged) {
        qk_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE engaged: hq={d} hk={d} rd={d} scaled={}\n", .{ hq, hk, rd, ms != null });
    }
    return .{ oq, ok };
}

/// Multi-position variant: [rows * rd] f32, row p = the cos|sin row at
/// position `offset + p` — the probe is a [1,1,rows,rd] stack of ones|zeros
/// rows, so `mlx_fast_rope` stamps each row at its own position exactly as it
/// would the real tensor.
///
/// `mscale` multiplies the rows (YaRN's attention factor; 1.0 leaves the probe
/// exactly as the stock kernel computed it). Both halves of a rotary pair tile
/// the same frequency, so scaling the rows scales the rotation of the pair —
/// the same product the reference gets from `cos*af, sin*af`.
pub fn ropeAngleRows(
    s: mlx.mlx_stream,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    rows: c_int,
    allocator: std.mem.Allocator,
    mscale: f32,
    scale: f32,
) !mlx.mlx_array {
    const n: usize = @intCast(rd);
    const nrows: usize = @intCast(rows);
    const buf = try allocator.alloc(f32, n * nrows);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = if ((i % n) < n / 2) 1.0 else 0.0;
    const shape = [_]c_int{ 1, 1, rows, rd };
    const probe = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(probe);
    var rotated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rotated);
    try mlx.check(mlx.mlx_fast_rope(&rotated, probe, rd, false, base, scale, offset, freqs, s));
    if (mscale == 1.0) {
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{rows * rd};
        try mlx.check(mlx.mlx_reshape(&flat, rotated, &flat_shape, 1, s));
        return flat;
    }
    // `defer`, not `errdefer`: the reshape below is a view, so the handle is ours to drop on the
    // success path too (an errdefer leaked one array per decode tick to the MTLSharedEvent cap).
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    const ms = mlx.mlx_array_new_float(mscale);
    defer _ = mlx.mlx_array_free(ms);
    try mlx.check(mlx.mlx_multiply(&scaled, rotated, ms, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rows * rd};
    try mlx.check(mlx.mlx_reshape(&flat, scaled, &flat_shape, 1, s));
    return flat;
}

// ── Fused QK-norm+RoPE, hd-256 (mlxfast-challenge qwen35_attention_qk_rms_rope port) ──
//
// The hd-128 kernel above never fires on qwen3.5/3.6 full-attention layers
// (every checkpoint in the family is hd 256), so their decode AND verify
// widths paid the composed per-head RMSNorm → transpose → partial-RoPE chain.
// This is the 256-wide sibling: 64 threads per (head, position) row, RMS via
// the same simdgroup-partials tree `rms_single_row` uses, normalized row
// staged bf16 in threadgroup memory (one rounding — the value the separate
// rms kernel would have written), then the RD-dim rotation reads the staged
// row against a PER-POSITION angle table (`ropeAngleRows` probe — the stock
// kernel's own floats, never re-derived; the challenge computes cos/sin
// in-kernel, which trusts JIT transcendentals our house rule forbids).
// Output is head-major [B,H,S,D], so the transposes vanish. Covers S 1..32
// (verify widths + one-chunk tiny-prompt prefills; challenge raised the same
// gate 16 -> 32 in 033f622).
// Bit-identity with the composed chain is the acceptance bar.
pub const QK_NORM_ROPE_256_SOURCE = kernelSource("kernels/rope/qk_norm_rope_256.metal");

var qk256_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk256_fused_engaged: bool = false;

fn getQkNormRope256Kernel() !mlx.mlx_fast_metal_kernel {
    if (qk256_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "qw", "kw", "vw", "angles", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok", "ov" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_qk_norm_rope_256",
        in_vec,
        out_vec,
        QK_NORM_ROPE_256_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk256_fused_kernel = kernel;
    return kernel;
}

const Qk256CfgKey = struct { hq: c_int, hk: c_int, hv: c_int, seq: c_int, rd: c_int, normed: bool, dtype: mlx.mlx_dtype };
var qk256_fused_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qk256_fused_cfg_key: Qk256CfgKey = std.mem.zeroes(Qk256CfgKey);

/// One fused dispatch for hd-256 heads at S in 1..32: per-head RMSNorm +
/// partial RoPE for q AND k (and, when `v_in`/`v_norm_w` are bound, the
/// norm-only v of Gemma), emitting the transposed [1,H,S,256] layout. Null
/// q/k norm weights (both, and no v) mean rotation only (Spark).
/// `angles` is the [S*RD] f32 table from `ropeAngleRows` (row p = position
/// offset+p). Null → caller keeps the composed chain. The third result is
/// null-ctx without v.
pub fn fusedQkNormRope256(
    s: mlx.mlx_stream,
    q_in: mlx.mlx_array, // [1, S, HQ, 256] (view ok — ensure_row_contiguous materializes)
    k_in: mlx.mlx_array, // [1, S, HK, 256]
    v_in: mlx.mlx_array, // [1, S, HV, 256] or null-ctx
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    v_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    eps_arr: mlx.mlx_array, // 0-dim f32 scalar
    hq: c_int,
    hk: c_int,
    seq: c_int,
    rd: c_int,
) !?[3]mlx.mlx_array {
    if (!qkNormRopeFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    if (seq < 1 or seq > 32) return null;
    const dt = mlx.mlx_array_dtype(q_in);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k_in) != dt) return null;
    const normed = q_norm_w.ctx != null;
    if (normed != (k_norm_w.ctx != null)) return null;
    if (normed and (mlx.mlx_array_dtype(q_norm_w) != dt or mlx.mlx_array_dtype(k_norm_w) != dt)) return null;
    // The shuffle-free staged rotation is exact for any even rd <= 256 whose
    // pair partner stays inside the row (qwen3.5's 64, Gemma's 256).
    if (rd != 32 and rd != 64 and rd != 128 and rd != 256) return null;
    const with_v = v_in.ctx != null;
    if (with_v and !normed) return null;
    var hv: c_int = 0;
    if (with_v) {
        if (v_norm_w.ctx == null or mlx.mlx_array_dtype(v_in) != dt or mlx.mlx_array_dtype(v_norm_w) != dt) return null;
        const vsh = mlx.getShape(v_in);
        if (vsh.len != 4 or vsh[0] != 1 or vsh[1] != seq or vsh[3] != 256) return null;
        hv = vsh[2];
    }

    const key = Qk256CfgKey{ .hq = hq, .hk = hk, .hv = hv, .seq = seq, .rd = rd, .normed = normed, .dtype = dt };
    if (qk256_fused_cfg == null or !std.meta.eql(qk256_fused_cfg_key, key)) {
        if (qk256_fused_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, seq, 256 };
        const ok_shape = [_]c_int{ 1, hk, seq, 256 };
        const ov_shape = [_]c_int{ 1, @max(hv, 1), if (with_v) seq else 1, 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ov_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk + hv) * seq * 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HK", hk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HV", hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "S", seq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "NORM", normed));
        qk256_fused_cfg = config;
        qk256_fused_cfg_key = key;
    }

    const kernel = try getQkNormRope256Kernel();
    // Unbound stages (no v, no norm) never read their slots; bind k / q so
    // every input is an array.
    const inputs_arr = [_]mlx.mlx_array{ q_in, k_in, if (with_v) v_in else k_in, if (normed) q_norm_w else q_in, if (normed) k_norm_w else k_in, if (with_v) v_norm_w else k_in, angles, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qk256_fused_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var outs: [3]mlx.mlx_array = undefined;
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, outputs_vec, i));
    }
    if (!with_v) {
        _ = mlx.mlx_array_free(outs[2]);
        outs[2] = .{ .ctx = null };
    }
    if (!qk256_fused_engaged) {
        qk256_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE (hd-256) engaged: hq={d} hk={d} hv={d} S={d} rd={d}\n", .{ hq, hk, hv, seq, rd });
    }
    return outs;
}

// ── tests ──
