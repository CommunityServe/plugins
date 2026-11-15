//! A KDA layer's prompt prologue as one launch (GLM53F_KDA_PROMPT_PROLOGUE): the chain's depthwise conv, SiLU, q's
//! and k's rms_norm and scales, and v, each the MLX kernel the chain runs in its operand order and types, the sigmoid
//! with the op's own precise exp; the conv window as conv1d.run keeps it. One simdgroup a (row, section, head), eight a
//! threadgroup. It takes GLM's prompt rows where the per-core route declines.
//! copied_from: mlx-serve src/plugins/glm5_next/kda_prologue.zig@4bab19a5 (#1196), its tests on the GPU in src/gpu_tests.zig.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const kernelSource = @import("kernels.zig").source;
const log = kit.log;
const PREFILL_EVAL_MIN_SEQ = kit.bases.run.PREFILL_EVAL_MIN_SEQ;
const materializedOwnedCopy = kit.mlx_array.materializedOwnedCopy;
const depthwise = kit.blocks.conv.depthwise;

const Array = mlx.mlx_array;

const SOURCE = kernelSource("kda_prologue.metal");
const HEADER = kernelSource("kda_prologue_header.metal");

/// Simdgroups a threadgroup.
const groups: c_int = 8;
/// The head width the text is written for (rms_single_row's 32 lanes x 4 reads).
const head_dim: c_int = 128;

/// What the chain leaves for the recurrence: q and k normed and scaled and v, each [1, rows, heads, 128] contiguous,
/// and the conv window the layer keeps; each a new array.
pub const Prologue = struct { q: Array, k: Array, v: Array, window: Array };

/// Launches since the load (the tests read that the text ran).
pub var runs: u64 = 0;
var engaged_said = false;
var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernelOf() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "qkv", "prev", "w", "q_scale", "k_scale", "eps" };
    const outs = [_][*:0]const u8{ "q_out", "k_out", "v_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("msv_glm_kda_prologue", iv, ov, SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// Whether the text takes one slot's qkv rows of `shape` ([1, rows, 3 x hk x dk]) at `hk` x `dk` key and `hv` x `dv`
/// value heads and `taps` conv taps: heads alike and 128 wide, whole threadgroups of (section, head) simdgroups.
pub fn serves(shape: []const c_int, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) bool {
    if (hk != hv or dk != dv or dk != head_dim or hk < 1 or taps < 2) return false;
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 3 * hk * dk) return false;
    return @rem(3 * hk, groups) == 0;
}

/// The chain's prologue over `qkv` after `prev` (the layer's conv window; null: a cold layer, zeros as the chain takes
/// them), `conv_w` [cdim, taps, 1], the chain's own 0-dim scales and its rms epsilon. Null off the GPU or on a shape or
/// dtype the text does not take (the caller runs the chain).
pub fn prologue(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    if (!mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(qkv);
    if (!serves(sh, hk, dk, hv, dv, taps)) return null;
    const rows = sh[1];
    const cdim = sh[2];
    const dt = mlx.mlx_array_dtype(qkv);
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    // The chain's conv weights, window and scales in qkv's dtype (else MLX promotes and the text is not the chain).
    const wsh = mlx.getShape(conv_w);
    if (mlx.mlx_array_dtype(conv_w) != dt or wsh.len != 3 or wsh[0] != cdim or wsh[1] != taps or wsh[2] != 1) return null;
    for ([_]Array{ q_scale, k_scale }) |x| if (mlx.mlx_array_dtype(x) != dt or mlx.mlx_array_ndim(x) != 0) return null;
    if (prev) |p| {
        const psh = mlx.getShape(p);
        if (psh.len != 3 or psh[0] != 1 or psh[1] != taps - 1 or psh[2] != cdim or mlx.mlx_array_dtype(p) != dt) return null;
    }
    var cold: Array = .{ .ctx = null };
    defer if (cold.ctx != null) {
        _ = mlx.mlx_array_free(cold);
    };
    const before = prev orelse blk: {
        cold = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&cold, &[_]c_int{ 1, taps - 1, cdim }, 3, dt, s));
        break :blk cold;
    };
    const window = try windowOf(s, before, qkv, rows, cdim, taps);
    errdefer _ = mlx.mlx_array_free(window);

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ 1, rows, hk, dk };
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 3 * hk, rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "DK", dk }, .{ "H", hk }, .{ "DIM", hk * dk }, .{ "CDIM", cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const ins = [_]Array{ qkv, before, conv_w, q_scale, k_scale, eps_arr };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernelOf(), iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != 3) return error.MetalKernelBadOutputCount;
    var outs: [3]Array = @splat(.{ .ctx = null });
    errdefer for (outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
    runs += 1;
    if (!engaged_said) {
        engaged_said = true;
        log.info("[kda] prompt prologue engaged: S={d} Hk={d} Hv={d} cold={}\n", .{ rows, hk, hv, prev == null });
    }
    return .{ .q = outs[0], .k = outs[1], .v = outs[2], .window = window };
}

/// conv1d.run's window over [before | qkv]: its rows [rows, rows + taps - 1), an owned copy past PREFILL_EVAL_MIN_SEQ
/// rows where the chain makes one (`materializedOwnedCopy`), else a slice.
fn windowOf(s: mlx.mlx_stream, before: Array, qkv: Array, rows: c_int, cdim: c_int, taps: c_int) !Array {
    const keep = taps - 1;
    var tail = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tail);
    if (rows >= keep) {
        try mlx.check(mlx.mlx_slice(&tail, qkv, &[_]c_int{ 0, rows - keep, 0 }, 3, &[_]c_int{ 1, rows, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    } else {
        const parts = [_]Array{ before, qkv };
        const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        try mlx.check(mlx.mlx_slice(&tail, cat, &[_]c_int{ 0, rows, 0 }, 3, &[_]c_int{ 1, rows + keep, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    }
    if (depthwise.compactConvStateEnabled() and rows >= PREFILL_EVAL_MIN_SEQ) {
        const owned = try materializedOwnedCopy(s, tail);
        _ = mlx.mlx_array_free(tail);
        depthwise.compact_conv_state_copy_count += 1;
        return owned;
    }
    return tail;
}
