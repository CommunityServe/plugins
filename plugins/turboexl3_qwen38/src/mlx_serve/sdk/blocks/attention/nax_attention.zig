//! copied_from: src/sdk/blocks/attention/nax_attention.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Tensor-unit (NAX) prefill attention at q/k 192 or 128, v 128 (MiMo-V2, Flint): MLX's own
//! NAX flash-attention kernel with a separate value head dim, ported from oMLX
//! (`omlx/utils/nax_attention.py`, kernels in `kernels/nax_attention.metal`).
//! Causal, optionally banded to a sliding window, with learned sinks. Long key
//! ranges run as consecutive key-range passes that hand the fp32 row state on
//! (bit-identical to one dispatch), so a pass's K/V stays in the on-chip caches;
//! those passes also split the head dims over simdgroup pairs.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const Scope = @import("../../../mlx/scope.zig").Scope;

const BK: c_int = 32;
const THREADS: c_int = 128;
/// Keys per pass, and the grid size from which passes pay (oMLX's measured defaults).
const PASS_KEYS: c_int = 8192;
const PASS_MIN_GROUPS: c_int = 512;
const MAX_PASSES = 64;

pub const SOURCE = kernelSource("kernels/attention/nax_attention_entry.metal");

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;

/// Head dim 256 with int8 Q and K (SageAttention): Q per 64-row tile, K per
/// 32-key block after removing each KV head's mean key (a constant offset on
/// every key moves each score row by a constant, which softmax ignores).
/// P and V stay in the activation dtype. Opt-in and lossy (`--int8-qk`).
pub const SOURCE_256Q8 = kernelSource("kernels/attention/nax_attention_256q8_entry.metal");
const Q8_WM: c_int = 4;
const Q8_BQ: c_int = 16 * Q8_WM;

var kernel_256q8: ?mlx.mlx_fast_metal_kernel = null;
var engaged_256q8 = false;
/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const NaxCliState = struct {
    /// Set by `--int8-qk`.
    int8_qk_cli: bool = false,
};
pub const naxCli_shared = @import("../../util/shared.zig").Shared(NaxCliState, "nax_attention_cli", .{});
pub inline fn naxCli() *NaxCliState {
    return naxCli_shared.get();
}

fn int8QkEnabled() bool {
    return naxCli().int8_qk_cli;
}

fn getKernel256Q8() !mlx.mlx_fast_metal_kernel {
    if (kernel_256q8) |k| return k;
    const ins = [_][*:0]const u8{ "q", "k", "v", "sinks", "state", "params", "qi", "ki", "qs", "ks" };
    const outs = [_][*:0]const u8{ "out", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const header = kernelSource("kernels/attention/nax_tiles.metal") ++ kernelSource("kernels/attention/nax_attention.metal");
    const k = mlx.mlx_fast_metal_kernel_new("msv_nax_attn_256_q8", in_vec, out_vec, SOURCE_256Q8, header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_256q8 = k;
    return k;
}

/// Causal hd-256 prefill attention with int8 Q/K, or null (switch off, not NAX,
/// another shape, or a verify-width query that keeps its own kernels).
pub fn attention256Int8(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!int8QkEnabled()) return null;
    const out = (try attention256Int8Impl(s, q, k, v, scale)) orelse return null;
    @import("../../util/log.zig").debug("[attn] int8 QK call qL={d} kL={d}\n", .{ mlx.getShape(q)[2], mlx.getShape(k)[2] });
    if (!engaged_256q8) {
        engaged_256q8 = true;
        @import("../../util/log.zig").info("[attn] int8 QK hd-256 prefill attention engaged (--int8-qk; lossy)\n", .{});
    }
    return out;
}

pub fn attention256Int8Impl(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("../linear/verify_qmm_nax.zig").verifyQmmNaxAvailable()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256) return null;
    if (ks[0] != qs[0] or vs[0] != qs[0] or vs[1] != ks[1] or vs[2] != ks[2]) return null;
    if (ks[1] == 0 or @rem(qs[1], ks[1]) != 0 or qs[2] < Q8_BQ or ks[2] < qs[2]) return null;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return null;

    var sc = Scope.init(std.heap.c_allocator, s);
    defer sc.deinit();
    var mean = mlx.mlx_array_new();
    const mk = try sc.res(mlx.mlx_mean_axes(&mean, k, &[_]c_int{2}, 1, true, s), &mean);
    const qq = try quantRows(&sc, q, Q8_BQ, null);
    const kq = try quantRows(&sc, k, BK, mk);

    const b = qs[0];
    const h = qs[1];
    const ql = qs[2];
    const kl = ks[2];
    const nq = @divTrunc(ql + Q8_BQ - 1, Q8_BQ);
    const nq_aligned = @divTrunc(ql, Q8_BQ);
    const nk = @divTrunc(kl + BK - 1, BK);
    const nk_aligned = @divTrunc(kl, BK);
    const zero = [_]f32{0};
    const one = [_]c_int{1};
    const dummy = try sc.keep(mlx.mlx_array_new_data(&zero, &one, 1, .float32));
    const sinks_t = try sc.astype(dummy, dt);
    const pdata = [16]i32{ b, h, 256, ql, kl, @divExact(h, ks[1]), @bitCast(scale), nq, nk, nq_aligned, nk_aligned, ql - nq_aligned * Q8_BQ, kl - nk_aligned * BK, kl - ql, 0, nk };
    const params = try sc.keep(mlx.mlx_array_new_data(&pdata, &[_]c_int{16}, 1, .int32));

    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ b, ql, h, 256 }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &one, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, nq * 32 * Q8_WM * 2, h, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * Q8_WM * 2, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WM", Q8_WM));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "ALIGN_Q", @rem(ql, Q8_BQ) == 0));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "ALIGN_K", @rem(kl, BK) == 0));
    const ins = [_]mlx.mlx_array{ q, k, v, sinks_t, dummy, params, qq.vals, kq.vals, qq.scales, kq.scales };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel256Q8(), in_vec, c, s));
    var o = mlx.mlx_array_new();
    _ = try sc.res(mlx.mlx_vector_array_get(&o, outs, 0), &o);
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&view, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    return view;
}

const Quantized = struct { vals: mlx.mlx_array, scales: mlx.mlx_array };

fn quantRows(sc: *Scope, x: mlx.mlx_array, rows: c_int, mean: ?mlx.mlx_array) !Quantized {
    const p = try @import("../dit/int8_attn.zig").quantRows(x, rows, mean, 1e-6, false, false, sc.s);
    const vals = sc.keep(p[0]) catch |e| {
        _ = mlx.mlx_array_free(p[1]);
        return e;
    };
    return .{ .vals = vals, .scales = try sc.keep(p[1]) };
}

const testing = std.testing;
