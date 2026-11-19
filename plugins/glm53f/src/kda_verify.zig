//! GLM-5.3-Flash's KDA verify rows that the joined decode step leaves to the op chain (a pack without the row-joined
//! projection, or gates it does not take): the bounded gate in one launch (exp(A_log) in it too), the capture's conv
//! input, and the per-row states as a launch MLX runs only when a rollback reads them. kda.zig runs them with the
//! packed prework and the norm-gate epilogue; each is the chain's bytes.
//! copied_from: mlx-serve src/plugins/glm5_next/kda_verify.zig@4bab19a5 (#1196), its tests on the GPU in src/gpu_tests.zig.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const kernelSource = @import("kernels.zig").source;
const log = kit.log;
const kernels = kit.blocks.gdn.kernels;
const kda = kit.bases.kda_hybrid.kda;

const Array = mlx.mlx_array;

/// The verify pieces that ran since the load (the tests read them).
pub const Runs = struct { prework: u64 = 0, gate: u64 = 0, normgate: u64 = 0, expsig: u64 = 0, lazy: u64 = 0 };
pub var runs: Runs = .{};

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn isFloat(d: mlx.mlx_dtype) bool {
    return d == .float16 or d == .bfloat16 or d == .float32;
}

/// MLX's promotion of two float dtypes: the dtype when both are it, else f32.
fn promote(a: mlx.mlx_dtype, b: mlx.mlx_dtype) ?mlx.mlx_dtype {
    if (!isFloat(a) or !isFloat(b)) return null;
    return if (a == b) a else .float32;
}

const GATE_SOURCE = kernelSource("kda_verify_gate.metal");
/// The gate's threadgroup: a row's channels are a whole number of them.
const gate_tg: c_int = 256;
var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var expsig_said = false;

fn gateKernel() !mlx.mlx_fast_metal_kernel {
    if (gate_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "a", "dt", "ea", "bound" };
    const outs = [_][*:0]const u8{"g"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_glm_kda_verify_gate", iv, ov, GATE_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    gate_kernel = k;
    return k;
}

/// `kdaGateChain(a_log, a, dt_bias, bound)` over one slot's rows `a` [1, rows, n] in one launch: the f32 gate
/// [1, rows, n]. `in_exp`: exp(A_log) made in the launch, else by the chain's ops. Null off the GPU or on a shape or
/// dtype it does not take (the caller runs the chain).
pub fn gate(s: mlx.mlx_stream, a: Array, dt_bias: Array, a_log: Array, bound: f32, in_exp: bool) !?Array {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(a);
    if (sh.len != 3 or sh[0] != 1 or sh[1] < 1 or sh[2] < gate_tg or @rem(sh[2], gate_tg) != 0) return null;
    const n: usize = @intCast(sh[2]);
    if (mlx.mlx_array_size(dt_bias) != n or mlx.mlx_array_size(a_log) != n) return null;
    const add_t = promote(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(dt_bias)) orelse return null;
    if (!isFloat(mlx.mlx_array_dtype(a_log))) return null;
    // Without `in_exp`, the chain's exp(astype(A_log, f32)), made by its ops.
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    if (in_exp) _ = mlx.mlx_array_set(&ea, a_log) else {
        var alf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(alf);
        try mlx.check(mlx.mlx_astype(&alf, a_log, .float32, s));
        try mlx.check(mlx.mlx_exp(&ea, alf, s));
    }
    const bound_arr = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(bound_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sh[1], sh[2] }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, sh[2], sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, gate_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "AddT", add_t));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", sh[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EXP", @intFromBool(in_exp)));
    const ins = [_]Array{ a, dt_bias, ea, bound_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try gateKernel(), iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 1) return error.MetalKernelBadOutputCount;
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_vector_array_get(&g, ov, 0));
    runs.gate += 1;
    if (in_exp) {
        runs.expsig += 1;
        if (!expsig_said) {
            expsig_said = true;
            log.info("[kda] verify gate exp(A_log) in its launch engaged: S={d} N={d}\n", .{ sh[1], sh[2] });
        }
    }
    return g;
}

/// conv1d.run's `input` for a verify, which a partial accept's rollback reads: [window | qkv] along the rows (MLX's
/// concatenate, as the chain makes it), a new array.
pub fn convInput(s: mlx.mlx_stream, window: Array, qkv: Array) !Array {
    const parts = [_]Array{ window, qkv };
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

pub const Lazy = struct { y: Array, state: Array, state_seq: Array };
var lazy_said = false;

/// kda.zig's capture branch as two launches over its inputs (q, k, v, the gate, beta, the state, T): y and the final
/// state from the stock recurrence (the capture text without its per-row stores, so the same bits), and the per-row
/// states [T, B, Hv, Dv, Dk] from the capture launch, kept alone: MLX runs it only when a rollback reads them.
pub fn captureLazy(s: mlx.mlx_stream, ins: [7]Array, in_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, rows: c_int) !Lazy {
    const st_dt = mlx.mlx_array_dtype(ins[5]);
    const y_shape = [_]c_int{ batch, rows, hv, dv };
    const st_shape = [_]c_int{ batch, hv, dv, dk };
    var out: Lazy = .{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new(), .state_seq = mlx.mlx_array_new() };
    errdefer inline for (.{ out.y, out.state, out.state_seq }) |a| free(a);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 4, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &st_shape, 4, st_dt));
        try kernels.gdnSetGrid(cfg, .stock, dv, hv, batch);
        try kda.templateArgs(cfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernels.gdnRouteKernel(.stock, 0, true), iv, cfg, s));
        if (mlx.mlx_vector_array_size(ov) != 2) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&out.y, ov, 0));
        try mlx.check(mlx.mlx_vector_array_get(&out.state, ov, 1));
    }
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 4, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, batch, hv, dv, dk }, 5, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &st_shape, 4, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dv, batch * hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
        try kda.templateArgs(cfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
        const seq_stride = mlx.mlx_array_new_int(batch * hv * dv * dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const seq_ins = ins ++ [_]Array{seq_stride};
        const iv = mlx.mlx_vector_array_new_data(&seq_ins, seq_ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernels.getGdnKernelSeq(true), iv, cfg, s));
        if (mlx.mlx_vector_array_size(ov) != 3) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&out.state_seq, ov, 1));
    }
    runs.lazy += 1;
    if (!lazy_said) {
        lazy_said = true;
        log.info("[kda] verify lazy per-row states engaged: S={d} Hv={d}\n", .{ rows, hv });
    }
    return out;
}
