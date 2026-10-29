//! copied_from: src/sdk/blocks/gdn/kernels.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/gdn/kernels.
//! Owns gdnKernelSource, GdnOpsResult, gdnRecurOps, GDN_CUDA_HEADER, gdnCudaSource, getGdnCudaKernel, and 41 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const gdn_decode = @import("decode.zig");
const testing = std.testing;
const naxAvailable = @import("../../../mlx/device.zig").naxAvailable;
const evalToF32 = @import("../../testing/parity.zig").evalToF32;
const hostF32 = @import("../../testing/parity.zig").hostF32;
const maxAbsDiff = @import("../../testing/parity.zig").maxAbsDiff;
const rmsVsTruth = @import("../../testing/parity.zig").rmsVsTruth;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const attn256RandBf16Scaled = @import("../attention/attn256.zig").attn256RandBf16Scaled;
const gdnChainGateDtype = @import("gate.zig").gdnChainGateDtype;
const gdnGateChain = @import("gate.zig").gdnGateChain;

// ── GatedDeltaNet fused Metal kernel ──
// Ported from mlx-lm/models/gated_delta.py: `_make_gated_delta_kernel(has_mask=False, vectorized=False)`.
// Processes the entire T-step delta recurrence in a single kernel dispatch, eliminating
// the per-token kernel-launch overhead that otherwise caps prefill at ~330 tok/s on
// Qwen 3.5/3.6 MoE. Template args (Dk, Dv, Hk, Hv) specialize the kernel; inputs carry
// the runtime shapes. State math runs in float32 for numerical stability regardless
// of the input/state storage dtype.
// The recurrence exists ONCE, specialized on two independent axes:
//
//   `vectorized` — where the forget gate lives. qwen3.5's GatedDeltaNet has one
//     scalar per (token, value head), so `g` is [B,T,Hv]. KDA (Kimi Delta
//     Attention, bailing_hybrid) gates per KEY CHANNEL, so `g` is [B,T,Hv,Dk]
//     and each state element decays by its own factor. Mirrors mlx-lm's
//     `_make_gated_delta_kernel(vectorized=…)`, which parameterizes the same
//     three lines.
//
//   `capture_seq` — spec-decode support, used ONLY on the verify forward (small
//     T). Besides the FINAL state (`state_out`, written from registers like the
//     stock kernel) it records the state after every INTERMEDIATE timestep into
//     `state_seq` ([T, B, Hv, Dv, Dk]), so partial-accept rollback restores the
//     accepted position by SLICING — no re-forward of the accepted prefix (the
//     costly part on a 48-layer trunk). Decode and prefill never pay for it.
//     `seq_stride` = (B*Hv)*Dv*Dk = the per-timestep element stride into
//     `state_seq` (passed as a scalar, mirroring `T`).
//     Capture-tail trim: state_seq[T-1] is deliberately NEVER written — a
//     partial accept reads index `accepted` <= T-2 (index T-1 would be a full
//     accept, which keeps `state_out` via the normal flow), so the last row was
//     a redundant global write + forced the engine's final state to be a slice
//     VIEW pinning the whole [T,...] capture buffer across rounds.
pub fn gdnKernelSource(comptime vectorized: bool, comptime capture_seq: bool) [:0]const u8 {
    const g_setup = if (vectorized)
        "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
    else
        "auto g_ = g + b_idx * T * Hv;";
    const g_access = if (vectorized) "g_[s_idx]" else "g_[hv_idx]";
    const g_advance = if (vectorized) "  g_ += Hv * Dk;" else "  g_ += Hv;";
    const seq_base = if (capture_seq)
        "auto seq_base = (n * Dv + dv_idx) * Dk;\n"
    else
        "";
    const seq_write = if (capture_seq)
        kernelSource("kernels/gdn/gated_delta_seq_write.metal")
    else
        "";
    return std.fmt.comptimePrint(kernelSource("kernels/gdn/gated_delta.fmt.metal"), .{ seq_base, g_setup, g_access, seq_write, g_advance });
}

const GdnOpsResult = struct { y: mlx.mlx_array, state: mlx.mlx_array, state_seq: mlx.mlx_array };

/// The `gated_delta_step` recurrence as MLX ops, one step per token, for GPUs
/// that run no Metal kernels (CUDA). Same contract and f32 state math as the
/// kernel; `state_seq` ([T,B,Hv,Dv,Dk], every step) only when `capture`.
pub fn gdnRecurOps(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, vectorized: bool, capture: bool, out_dtype: mlx.mlx_dtype) !GdnOpsResult {
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    const b = qs[0];
    const t_len = qs[1];
    const hk = qs[2];
    const dk = qs[3];
    const hv = vs[2];
    const dv = vs[3];
    const state_dtype = mlx.mlx_array_dtype(state_in);

    // f32, k/q heads repeated onto the value heads, shaped for per-step matmuls.
    const Prep = struct {
        fn run(st: mlx.mlx_stream, a: mlx.mlx_array, rep: c_int, shape: []const c_int) !mlx.mlx_array {
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, a, .float32, st));
            var r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(r);
            if (rep > 1) try mlx.check(mlx.mlx_repeat_axis(&r, f, rep, 2, st)) else try mlx.check(mlx.mlx_copy(&r, f, st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&out, r, shape.ptr, shape.len, st));
            return out;
        }
        fn step(st: mlx.mlx_stream, a: mlx.mlx_array, t: c_int) !mlx.mlx_array {
            const sh = mlx.getShape(a);
            var start = [5]c_int{ 0, 0, 0, 0, 0 };
            var stop: [5]c_int = undefined;
            const strides = [5]c_int{ 1, 1, 1, 1, 1 };
            for (sh, 0..) |d, i| stop[i] = d;
            start[1] = t;
            stop[1] = t + 1;
            var sl = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sl);
            try mlx.check(mlx.mlx_slice(&sl, a, &start, sh.len, &stop, sh.len, &strides, sh.len, st));
            var shape: [4]c_int = undefined;
            shape[0] = sh[0];
            for (sh[2..], 1..) |d, i| shape[i] = d;
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&out, sl, &shape, sh.len - 1, st));
            return out;
        }
    };
    const rep = @divExact(hv, hk);
    const q_col = try Prep.run(s, q, rep, &.{ b, t_len, hv, dk, 1 });
    defer _ = mlx.mlx_array_free(q_col);
    const k_col = try Prep.run(s, k, rep, &.{ b, t_len, hv, dk, 1 });
    defer _ = mlx.mlx_array_free(k_col);
    const k_row = try Prep.run(s, k, rep, &.{ b, t_len, hv, 1, dk });
    defer _ = mlx.mlx_array_free(k_row);
    const v_col = try Prep.run(s, v, 1, &.{ b, t_len, hv, dv, 1 });
    defer _ = mlx.mlx_array_free(v_col);
    const beta_b = try Prep.run(s, beta, 1, &.{ b, t_len, hv, 1, 1 });
    defer _ = mlx.mlx_array_free(beta_b);
    const g_b = try Prep.run(s, g, 1, if (vectorized) &.{ b, t_len, hv, 1, dk } else &.{ b, t_len, hv, 1, 1 });
    defer _ = mlx.mlx_array_free(g_b);

    var state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&state, state_in, .float32, s));
    defer _ = mlx.mlx_array_free(state);
    const n: usize = @intCast(t_len);
    const ys = try std.heap.c_allocator.alloc(mlx.mlx_array, n);
    defer std.heap.c_allocator.free(ys);
    var n_ys: usize = 0;
    defer for (ys[0..n_ys]) |y| {
        _ = mlx.mlx_array_free(y);
    };
    const seq = try std.heap.c_allocator.alloc(mlx.mlx_array, if (capture) n else 0);
    defer std.heap.c_allocator.free(seq);
    var n_seq: usize = 0;
    defer for (seq[0..n_seq]) |x| {
        _ = mlx.mlx_array_free(x);
    };

    var t: c_int = 0;
    while (t < t_len) : (t += 1) {
        const gt = try Prep.step(s, g_b, t);
        defer _ = mlx.mlx_array_free(gt);
        const kc = try Prep.step(s, k_col, t);
        defer _ = mlx.mlx_array_free(kc);
        const kr = try Prep.step(s, k_row, t);
        defer _ = mlx.mlx_array_free(kr);
        const qc = try Prep.step(s, q_col, t);
        defer _ = mlx.mlx_array_free(qc);
        const vc = try Prep.step(s, v_col, t);
        defer _ = mlx.mlx_array_free(vc);
        const bt = try Prep.step(s, beta_b, t);
        defer _ = mlx.mlx_array_free(bt);

        var decayed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(decayed);
        try mlx.check(mlx.mlx_multiply(&decayed, state, gt, s));
        var kv_mem = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kv_mem);
        try mlx.check(mlx.mlx_matmul(&kv_mem, decayed, kc, s));
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, vc, kv_mem, s));
        var delta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(delta);
        try mlx.check(mlx.mlx_multiply(&delta, diff, bt, s));
        var upd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(upd);
        try mlx.check(mlx.mlx_matmul(&upd, delta, kr, s));
        var next = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&next, decayed, upd, s));
        _ = mlx.mlx_array_free(state);
        state = next;

        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&y, state, qc, s));
        ys[n_ys] = y;
        n_ys += 1;
        if (capture) {
            var c = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&c, state, state_dtype, s));
            seq[n_seq] = c;
            n_seq += 1;
        }
    }

    var y_stack = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_stack);
    {
        const vec = mlx.mlx_vector_array_new_data(ys.ptr, n_ys);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_stack_axis(&y_stack, vec, 1, s));
    }
    var y_shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_shaped);
    const y_shape = [_]c_int{ b, t_len, hv, dv };
    try mlx.check(mlx.mlx_reshape(&y_shaped, y_stack, &y_shape, 4, s));
    var out: GdnOpsResult = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = .{ .ctx = null } };
    errdefer {
        _ = mlx.mlx_array_free(out.y);
        _ = mlx.mlx_array_free(out.state);
    }
    try mlx.check(mlx.mlx_astype(&out.y, y_shaped, out_dtype, s));
    try mlx.check(mlx.mlx_astype(&out.state, state, state_dtype, s));
    if (capture) {
        const vec = mlx.mlx_vector_array_new_data(seq.ptr, n_seq);
        defer _ = mlx.mlx_vector_array_free(vec);
        out.state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_stack_axis(&out.state_seq, vec, 0, s));
    }
    return out;
}

const GDN_CUDA_HEADER = kernelSource("kernels/gdn/gated_delta_header.cu");

/// `gdnKernelSource` for MLX's CUDA backend: one warp per (head, RPW Dv rows), the Dk lanes reduced
/// by shuffle. A row's arithmetic does not depend on RPW; rows of one head share the step's loads.
/// Scalars arrive as pointers (`T_arr`).
fn gdnCudaSource(comptime vectorized: bool, comptime capture_seq: bool) [:0]const u8 {
    // Step t+1's inputs load while step t computes; each converts to float as it did in place.
    const ng = if (vectorized) "n_per_t" else "1";
    const g_load = if (vectorized) "gn[i] = static_cast<float>(g_[(long)tt * Hv * Dk + s_idx]);" else "if (i == 0) gn[0] = static_cast<float>(g_[(long)tt * Hv + hv_idx]);";
    const g_use = if (vectorized) "gc[i]" else "gc[0]";
    const g_setup = if (vectorized) "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;" else "auto g_ = g + b_idx * T * Hv;";
    const seq_write = if (capture_seq)
        kernelSource("kernels/gdn/gated_delta_seq_write.cu")
    else
        "";
    return std.fmt.comptimePrint(kernelSource("kernels/gdn/gated_delta.fmt.cu"), .{ g_setup, ng, g_load, ng, ng, g_use, seq_write });
}

fn getGdnCudaKernel(comptime vectorized: bool, comptime capture_seq: bool) !mlx.mlx_fast_cuda_kernel {
    const Cache = struct {
        const is_vectorized = vectorized;
        const is_capture_seq = capture_seq;
        var kernel: ?mlx.mlx_fast_cuda_kernel = null;
    };
    if (Cache.kernel) |k| return k;
    const input_names = if (capture_seq)
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T_arr", "seq_stride_arr" }
    else
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T_arr" };
    const output_names = if (capture_seq)
        [_][*:0]const u8{ "y", "state_seq", "state_out" }
    else
        [_][*:0]const u8{ "y", "state_out" };
    const name = "gated_delta_step_cuda" ++ (if (capture_seq) "_seq" else "") ++ (if (vectorized) "_vec" else "");
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, gdnCudaSource(vectorized, capture_seq), GDN_CUDA_HEADER, true, 0);
    if (k.ctx == null) return error.CudaKernelCreateFailed;
    Cache.kernel = k;
    return k;
}

/// Dv rows per warp: a prefill-length recurrence shares each step's loads over 16 rows (fastest of
/// 1-16 on GB10); a decode step keeps one row per warp for the parallelism.
fn gdnCudaRowsPerWarp(t_len: c_int, dv: c_int) c_int {
    return if (t_len >= 64 and @rem(dv, 64) == 0) 16 else 1;
}

/// Test seam: rows per warp for `gdnRecurCuda` (null = `gdnCudaRowsPerWarp`).
var gdn_cuda_rows_per_warp: ?c_int = null;

/// The GDN recurrence on MLX's CUDA backend, same contract as `gdnRecurOps`;
/// null when not on CUDA or Dk is not a warp multiple.
pub fn gdnRecurCuda(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, vectorized: bool, capture: bool, out_dtype: mlx.mlx_dtype) !?GdnOpsResult {
    if (!mlx.cudaAvailable() or !mlx.streamOnGpuDevice(s)) return null;
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    const b = qs[0];
    const t_len = qs[1];
    const hk = qs[2];
    const dk = qs[3];
    const hv = vs[2];
    const dv = vs[3];
    if (@rem(dk, 32) != 0 or @rem(hv, hk) != 0) return null;
    const state_dtype = mlx.mlx_array_dtype(state_in);

    const config = mlx.mlx_fast_cuda_kernel_config_new();
    defer mlx.mlx_fast_cuda_kernel_config_free(config);
    const y_shape = [_]c_int{ b, t_len, hv, dv };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &y_shape, 4, out_dtype));
    if (capture) {
        const seq_shape = [_]c_int{ t_len, b, hv, dv, dk };
        try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &seq_shape, 5, state_dtype));
    }
    const state_shape = [_]c_int{ b, hv, dv, dk };
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_output_arg(config, &state_shape, 4, state_dtype));
    const rpw = gdn_cuda_rows_per_warp orelse gdnCudaRowsPerWarp(t_len, dv);
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_grid(config, 32, @divExact(dv, rpw), b * hv));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "RPW", rpw));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "StT", state_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_cuda_kernel_config_add_template_arg_int(config, "Hv", hv));

    const t_arr = mlx.mlx_array_new_int(t_len);
    defer _ = mlx.mlx_array_free(t_arr);
    const stride_arr = mlx.mlx_array_new_int(b * hv * dv * dk);
    defer _ = mlx.mlx_array_free(stride_arr);
    const ins = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, t_arr, stride_arr };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, if (capture) 8 else 7);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    const kernel = switch (vectorized) {
        inline else => |vec| switch (capture) {
            inline else => |cap| try getGdnCudaKernel(vec, cap),
        },
    };
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_cuda_kernel_apply(&outs, kernel, in_vec, config, s));
    var r: GdnOpsResult = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = .{ .ctx = null } };
    try mlx.check(mlx.mlx_vector_array_get(&r.y, outs, 0));
    if (capture) {
        r.state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&r.state_seq, outs, 1));
    }
    try mlx.check(mlx.mlx_vector_array_get(&r.state, outs, if (capture) 2 else 1));
    return r;
}

/// One compiled kernel per (vectorized, capture_seq) pair. Each comptime
/// instantiation gets its own cache slot; the Metal name must differ per
/// variant or MLX hands back the wrong compiled function.
fn getGdnKernelFor(comptime vectorized: bool, comptime capture_seq: bool) !mlx.mlx_fast_metal_kernel {
    // The two `const`s are load-bearing: a struct declared inside a function
    // that captures NOTHING from the comptime parameters is memoized to ONE
    // type across every instantiation, so all four variants would share a
    // single cache slot and the first kernel compiled would be handed back for
    // the rest (a seq kernel answering a non-seq call = wrong output count).
    const Cache = struct {
        const is_vectorized = vectorized;
        const is_capture_seq = capture_seq;
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    if (Cache.kernel) |k| return k;
    const input_names = if (capture_seq)
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" }
    else
        [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = if (capture_seq)
        [_][*:0]const u8{ "y", "state_seq", "state_out" }
    else
        [_][*:0]const u8{ "y", "state_out" };
    const name = "gated_delta_step" ++
        (if (capture_seq) "_seq" else "") ++
        (if (vectorized) "_vec" else "");
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        gdnKernelSource(vectorized, capture_seq),
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = kernel;
    return kernel;
}

pub fn getGdnKernel(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, false) else getGdnKernelFor(false, false);
}

pub fn getGdnKernelSeq(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, true) else getGdnKernelFor(false, true);
}

// ── GatedDeltaNet blocked-seq PREFILL kernel ──
// Port of oMLX's `gated_delta_blocked_seq` (custom_kernels/qwen35_prefill/
// gdn.py, Apache-2.0, oMLX by jundot — keep this provenance chain). Same
// EXACT recurrence as GDN_KERNEL_SOURCE — no chunked/WY reformulation —
// restructured for Apple-GPU efficiency: k/q/v staged into threadgroup
// memory in TB-token blocks with coalesced cooperative loads (the stock
// kernel re-reads k/q from device once per (Dv/4)-slice threadgroup => 32x
// redundant traffic, ~13 GB per 16K-token layer), state register-resident
// as (dv row, 16-wide d segment) fragments, and the k·state / q·state
// contractions reduced across the 8 segment-threads of a dv row via
// simd_shuffle_down — no threadgroup barriers in the hot token loop. Their
// measurement: 14.9 ms vs 29.7 ms per layer @16K = ~2x over the stock
// kernel shape; 48 of the 27B's 64 layers are GDN.
// Geometry contract (checked by gdnBlockedEligible): Dk == 128 exactly
// (8 threads x 16-wide fragments), Dv % 32 == 0 (DB=32 dv rows per
// threadgroup). Anything else — and decode (T==1), PLD/MTP verify, and the
// per-position-state capture path — stays on the stock kernels.
// Block size 32 for bf16 (Metal's 32 KiB threadgroup limit governs,
// and it is a function of the INPUT WIDTH, not a constant: fp32 activations
// stage twice the bytes and clamp to 16 via gdnBlockTFor. GDN inputs are NOT
// always bf16 — an f16 checkpoint promotes its activations to f32).

/// Test seam: forces the blocked-prefill route on/off.
pub var gdn_blocked_override: ?bool = null;

pub fn gdnBlockedEnabled() bool {
    return gdn_blocked_override orelse true;
}

/// Prefill-width floor for the blocked kernel (mirrors oMLX's OMLX_GDN_MIN_T
/// default). Below this the per-token stock kernel wins on launch overhead.
pub const GDN_BLOCKED_MIN_T: c_int = 64;

/// Pure routing predicate: geometry + width gate for the blocked-seq kernel.
pub fn gdnBlockedEligible(seq_len: c_int, dk: c_int, dv: c_int, num_k_heads: c_int, num_v_heads: c_int) bool {
    if (seq_len < GDN_BLOCKED_MIN_T) return false;
    if (dk != 128) return false;
    if (dv < 32 or @rem(dv, 32) != 0) return false;
    if (num_k_heads <= 0 or @rem(num_v_heads, num_k_heads) != 0) return false;
    return true;
}

/// Metal's threadgroup memory budget per threadgroup on Apple GPUs. The
/// blocked kernel stages k/q/v in threadgroup memory, so this is a HARD cap on
/// TB x input width — exceed it and the kernel does not compile.
const METAL_TG_BYTES: usize = 32 * 1024;

/// Threadgroup bytes the blocked kernel stages per TB-token block:
/// `k_s`+`q_s` [TB][Dk+8] and `v_s` [TB][DB+8] at the input width, plus the
/// `g_s`/`b_s` [TB] f32 rows. Mirrors the declarations in
/// GDN_KERNEL_BLOCKED_BODY — change one, change the other.
pub fn gdnBlockedTgBytes(tb: u32, dk: c_int, in_elem_size: usize) usize {
    const DB: usize = 32; // dv rows per threadgroup
    const t: usize = tb;
    const k: usize = @intCast(dk);
    return t * (2 * (k + 8) * in_elem_size + (DB + 8) * in_elem_size + 2 * @sizeOf(f32));
}

/// Largest supported block size <= `requested` whose staging fits the budget
/// at this input width, or null when even the smallest does not (decline the
/// blocked route and stay on the stock kernel).
///
/// The width is NOT a constant: bf16 and f16 checkpoints stage 2 bytes and
/// keep the requested size, but an f16 checkpoint promotes its activations to
/// f32 (f16 (x) f32-scalar -> f32), and at 4 bytes TB=32 needs 40 KiB. Block
/// size and input dtype are therefore ONE decision — picking them apart is how
/// a valid-looking config produces a kernel that cannot compile.
pub fn gdnBlockTFor(requested: u32, dk: c_int, in_elem_size: usize) ?u32 {
    var best: ?u32 = null;
    for (GDN_BLOCKED_TBS) |cand| {
        if (cand > requested) continue;
        if (gdnBlockedTgBytes(cand, dk, in_elem_size) > METAL_TG_BYTES) continue;
        if (best == null or cand > best.?) best = cand;
    }
    return best;
}

const GDN_BLOCKED_TBS = [_]u32{ 16, 32, 48 };
var gdn_blocked_cached: [GDN_BLOCKED_TBS.len]?mlx.mlx_fast_metal_kernel = @splat(null);
/// The blocked kernel's time block.
pub const GDN_BLOCK_T: u32 = 32;

// Body of the blocked-seq kernel; the `constexpr int TB = N;` line is
// prepended per block size by gdnBlockedSource (one cached kernel per TB,
// DISTINCT names — two specializations sharing a name bind the wrong binary).
// Faithful transcription of oMLX's _KERNEL_S_SRC with one generalization:
// state loads/stores go through vec<StT,4> (theirs hard-codes float4 — their
// state is fp32, ours rides the ssm_state buffer's own dtype exactly like the
// stock kernel, so cross-chunk behavior is unchanged).
const GDN_KERNEL_BLOCKED_BODY = kernelSource("kernels/gdn/gated_delta_blocked_body.metal");

fn gdnBlockedSource(comptime tb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("constexpr int TB = {d};\n", .{tb}) ++ GDN_KERNEL_BLOCKED_BODY;
}

fn getGdnKernelBlocked(tb: u32) !mlx.mlx_fast_metal_kernel {
    inline for (GDN_BLOCKED_TBS, 0..) |cand, i| {
        if (cand == tb) {
            if (gdn_blocked_cached[i]) |k| return k;
            const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
            const output_names = [_][*:0]const u8{ "y", "state_out" };
            const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
            defer _ = mlx.mlx_vector_string_free(in_vec);
            const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
            defer _ = mlx.mlx_vector_string_free(out_vec);
            const kernel = mlx.mlx_fast_metal_kernel_new(
                std.fmt.comptimePrint("gated_delta_blocked_tb{d}", .{cand}),
                in_vec,
                out_vec,
                gdnBlockedSource(cand),
                "",
                true,
                false,
            );
            if (kernel.ctx == null) return error.MetalKernelCompileFailed;
            gdn_blocked_cached[i] = kernel;
            return kernel;
        }
    }
    return error.UnsupportedGdnBlockT;
}

/// Which recurrence kernel a GDN call runs: the per-token stock kernel, the blocked-seq
/// prefill kernel, or oMLX's software-pipelined one, which takes every call the blocked
/// kernel would.
pub const GdnRoute = enum { stock, blocked, pipelined };

var gdn_pipelined_cached: ?mlx.mlx_fast_metal_kernel = null;
pub var gdn_pipelined_override: ?bool = null; // test seam

/// Archs whose prefill was A/B'd against the blocked kernel on an M5 Ultra: Flash Next
/// (`qwen4_exp`, +2.6%) and dense Qwen3.8-27B (`qwen3_5`, +1.1 to 1.5%), both 48 value heads.
/// The kernel is issue- and latency-bound, so an unmeasured arch or chip keeps the blocked
/// kernel until someone measures it.
const GDN_PIPELINED_MEASURED = [_][]const u8{ "qwen4_exp", "qwen3_5" };

/// Whether a GDN prefill takes the pipelined kernel. It sums the recurrence's two 128-wide
/// dots in another fp32 order than the blocked one, so greedy text can move.
/// Only a measured arch on an M5-class GPU takes it.
pub fn gdnPipelinedFor(model_type: []const u8, nax: bool) bool {
    if (!nax) return false;
    for (GDN_PIPELINED_MEASURED) |t| {
        if (std.mem.eql(u8, t, model_type)) return true;
    }
    return false;
}

/// Decided per call, since the registry serves several model types in one process.
pub fn gdnPipelinedEnabled(model_type: []const u8) bool {
    if (gdn_pipelined_override) |v| return v;
    return gdnPipelinedFor(model_type, naxAvailable());
}

fn getGdnKernelPipelined() !mlx.mlx_fast_metal_kernel {
    if (gdn_pipelined_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = [_][*:0]const u8{ "y", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("gated_delta_pipelined_tb12", in_vec, out_vec, kernelSource("kernels/gdn/gdn_pipelined.metal"), "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_pipelined_cached = kernel;
    return kernel;
}

pub fn gdnRouteKernel(route: GdnRoute, tb: u32, vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return switch (route) {
        .stock => getGdnKernel(vector_gate),
        .blocked => getGdnKernelBlocked(tb),
        .pipelined => getGdnKernelPipelined(),
    };
}

pub fn gdnSetGrid(config: mlx.mlx_fast_metal_kernel_config, route: GdnRoute, dv: c_int, hv: c_int, batch: c_int) !void {
    switch (route) {
        // (32, Dv, B*Hv) threads in (32, 4, 1) groups, as mlx-lm.
        .stock => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
        },
        // 256 threads own 32 value rows.
        .blocked => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256 * @divExact(dv, 32), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        },
        // 128 threads own 16 value rows, 8 lanes each; Dv is a multiple of 32 on this route.
        .pipelined => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128 * @divExact(dv, 16), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        },
    }
}

// ── tests ──

// Helper for the GDN seq-kernel parity test below: run the recurrence via the
// requested kernel variant and return the FINAL state [B,Hv,Dv,Dk] as f32.

// Like gdnTestRun(seq=true) but returns the state at an arbitrary position `pos`
// (not just the last) — mirrors what `ssmRollbackFromCapture` slices out.

// ── GDN blocked-seq kernel test harness ──

const GdnHostRef = struct { y: []f32, state: []f32 };

const GdnRunOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// Run the GDN recurrence on one route and return BOTH outputs.
pub fn gdnRunYState(route: GdnRoute, tb_requested: u32, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !GdnRunOut {
    // Mirror the production route exactly: dtypes read off the arrays, block
    // size clamped to what the staging budget allows at that width.
    const in_dtype = mlx.mlx_array_dtype(q);
    const st_dtype = mlx.mlx_array_dtype(state_in);
    const tb = gdnBlockTFor(tb_requested, Dk, mlx.mlx_array_itemsize(q)) orelse return error.GdnBlockedDeclined;
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    // The state leaves at the width it came in (f32 for Hadamard packs), as in the forward.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, st_dtype));
    try gdnSetGrid(config, route, Dv, Hv, B);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", in_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try gdnRouteKernel(route, tb, false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var yo = mlx.mlx_array_new();
    var so = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&yo, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&so, outputs_vec, 1));
    return .{ .y = yo, .state = so };
}

/// `in_dtype` is the ACTIVATION width q/k/v arrive in. bf16 is the common
/// case; an f16 checkpoint promotes its activations to f32 (f16 ⊕ f32 scalar),
/// which is a different kernel specialization AND a different threadgroup
/// budget — both covered by the sweep below.
const GdnCase = struct { B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dv: c_int, tb: u32, in_dtype: mlx.mlx_dtype = .bfloat16, state_dtype: mlx.mlx_dtype = .bfloat16 };
