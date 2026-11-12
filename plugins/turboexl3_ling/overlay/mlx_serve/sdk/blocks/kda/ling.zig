//! turboexl3_ling's own (not a copy): Ling-3.1-flash's KDA kernels for the copied layer (plugins/glm5_next/kda.zig),
//! each the arithmetic of the ops or kernels it replaces, so a layer's bytes are the composed path's:
//! - `gate`: the per-channel gate in one launch (kernels/kda/ling_kda_gate.metal), `kdaGateChain`'s eight ops;
//! - `decodeStep`: a decode step of up to 8 rows in one launch per layer (kernels/kda/ling_kda_decode.metal): the
//!   packed prework's and the norm-gate's arithmetic on their table arm (MLX's own sigmoid of every 16-bit pattern), the
//!   gate's and the stock vector-gate recurrence's;
//! - `verifyRows`: an exact verify's 2 to 8 rows in one launch (the same kernel text with ROWS_EXACT): each row the
//!   bytes of a serial one-row step, the state's store and load between rows as serial decode has them, and the state
//!   after every row and the window's inputs kept for a partial accept's rollback;
//! - `recurBlocked`: the stock vector-gate recurrence over prompt rows with each block of tokens staged per threadgroup
//!   of value rows (kernels/kda/ling_kda_blocked_rows.metal on `rec_shape`: 8 simdgroups a threadgroup, 16-token
//!   blocks, 2 value rows a simdgroup). It takes the gate per head, [B, T, hv, 128]; `gateHeads` views the prompt path's
//!   flat gate that way.
//! - `decodeStepEpi` / `verifyRowsEpi` (src/ling_kda_epi.zig): the same launch with an epilogue
//!   (kernels/kda/ling_kda_epi.metal, its header kernels/kda/ling_kda_oprep.metal) that also writes what the launches
//!   after it made: a verify's window inputs (its concatenate's copy) and o_proj's DEC-ONE prepared rows, block sums and
//!   zeroed arrival counters (turboexl3's prep_sc statements), so o_proj runs its GEMV alone.
//! Each returns null where it does not serve (the caller keeps its path), checked before any launch.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("mlx_serve").mlx;
const log = @import("../../util/log.zig");
const sigmoidTableFor = @import("../mlp/fused_activations.zig").sigmoidTableFor;
/// The plugin's run-path line (src/ling_ran.zig): the blocked recurrence marks KDA_REC_SHAPE where it runs.
const ran = @import("../../../../ling_ran.zig");

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// `ins` and `outs`: pointers to arrays of names (as the copied blocks pass them).
fn newKernel(comptime name: [:0]const u8, ins: anytype, outs: anytype, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return newKernelH(name, ins, outs, src, "");
}

/// `newKernel` with a header (the text MLX puts before the kernel's function: helpers its body calls).
fn newKernelH(comptime name: [:0]const u8, ins: anytype, outs: anytype, src: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, iv, ov, src.ptr, header.ptr, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn apply(k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != outs.len) return error.MetalKernelBadOutputCount;
    for (outs) |*o| o.* = .{ .ctx = null };
    errdefer for (outs) |o| free(o);
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
}

fn isAct(dt: mlx.mlx_dtype) bool {
    return dt == .bfloat16 or dt == .float16;
}

// ── the gate ──

var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gate_engaged = false;

/// `kdaGateChain(A_log, a, dt_bias, bound)` in one launch: g f32 in a's shape. `A_log` and `dt_bias` f32 [W] (W, the
/// key channels, a's last dimension), `a` bf16 or f16. Null outside that.
pub fn gate(s: mlx.mlx_stream, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, bound: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ash = mlx.getShape(a);
    if (ash.len < 1 or !isAct(mlx.mlx_array_dtype(a))) return null;
    const w = ash[ash.len - 1];
    if (mlx.mlx_array_dtype(A_log) != .float32 or mlx.mlx_array_dtype(dt_bias) != .float32) return null;
    if (mlx.getShape(A_log).len != 1 or mlx.getShape(A_log)[0] != w or mlx.getShape(dt_bias).len != 1 or mlx.getShape(dt_bias)[0] != w) return null;
    const n_total: c_int = @intCast(mlx.mlx_array_size(a));
    if (n_total == 0) return null;
    const k = gate_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "A_log", "a", "dt_bias", "lb", "n_total" };
        const outs = [_][*:0]const u8{"g"};
        const nk = try newKernel("ling_kda_gate", &ins, &outs, kernelSource("kernels/kda/ling_kda_gate.metal"));
        gate_kernel = nk;
        break :blk nk;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, ash.ptr, ash.len, .float32));
    const tg: c_int = 256;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(n_total + tg - 1, tg) * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "W", w));
    const lb = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(lb);
    const nt = mlx.mlx_array_new_int(n_total);
    defer _ = mlx.mlx_array_free(nt);
    var out: [1]mlx.mlx_array = undefined;
    try apply(k, cfg, s, &.{ A_log, a, dt_bias, lb, nt }, &out);
    if (!gate_engaged) {
        gate_engaged = true;
        log.info("[kda] Ling's gate in one launch engaged: {d} channels\n", .{w});
    }
    return out[0];
}

// ── the decode step ──

/// The widest decode step one launch takes (its threadgroup arrays hold this many rows).
pub const decode_max_rows: c_int = 8;
var decode_kernel: ?mlx.mlx_fast_metal_kernel = null;
var decode_engaged = false;
var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rows_engaged = false;
pub const DecodeOut = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, state: mlx.mlx_array };
/// An exact verify's rows: `DecodeOut`'s, the state after every row `state_seq` [S, 1, hv, dv, dk] in the state's dtype
/// and `conv_input` [1, 3 + S, C] (the window's 3 inputs, then the rows'), as plugins/glm5_next/kda.zig's `exactRows`
/// keeps them.
pub const RowsOut = struct { y: mlx.mlx_array, conv_state: mlx.mlx_array, state: mlx.mlx_array, state_seq: mlx.mlx_array, conv_input: mlx.mlx_array };
pub const DecodeIn = struct {
    /// [1, S, C] q|k|v projections, C = 2 hk dk + hv dv
    qkv: mlx.mlx_array,
    /// [1, S, hv dv] output gate, [1, S, hv dk] gate input, [1, S, hv] beta input, all in the activation dtype
    z: mlx.mlx_array,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    /// [1, 3, C] the conv window, [C, 4, 1] the depthwise conv
    conv_state: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    /// f32 [hv dk] each
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    /// [1, hv, dv, dk] in the state dtype
    state: mlx.mlx_array,
    /// [dv]
    norm_w: mlx.mlx_array,
    /// 0-dim, the activation dtype: 1/dk and 1/sqrt(dk) as the layer's composed chain multiplies them
    q_scale: mlx.mlx_array,
    k_scale: mlx.mlx_array,
    qk_eps: f32,
    norm_eps: f32,
    bound: f32,
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
};

/// The layer's decode step over S <= `decode_max_rows` rows in one launch: y [1, S, hv dv] (the out projection's input),
/// the next conv window and state. Null outside Ling's geometry (hk = hv, dk = dv = 128, a 4-tap conv, bf16 or f16
/// activations, f32 A_log and dt_bias).
pub fn decodeStep(s: mlx.mlx_stream, in: DecodeIn) !?DecodeOut {
    var out: [3]mlx.mlx_array = undefined;
    if (!try launchRows(s, in, false, &out)) return null;
    if (!decode_engaged) {
        decode_engaged = true;
        log.info("[kda] Ling's decode step in one launch engaged: {d} heads, S={d}\n", .{ in.hv, mlx.getShape(in.qkv)[1] });
    }
    return .{ .y = out[0], .conv_state = out[1], .state = out[2] };
}

/// An exact verify's S rows (2 to `decode_max_rows`) in one launch: each row's y, the next window and state, the state
/// after every row and the window's inputs, with the bytes of S serial one-row steps (the composed layer, or
/// `decodeStep` at one row): between two rows the state takes the store to its dtype and the float load that serial
/// decode puts between two forwards. Null where `decodeStep` would be, and at one row.
pub fn verifyRows(s: mlx.mlx_stream, in: DecodeIn) !?RowsOut {
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[1] < 2) return null;
    var out: [4]mlx.mlx_array = undefined;
    if (!try launchRows(s, in, true, &out)) return null;
    errdefer for (out) |o| free(o);
    var conv_input = mlx.mlx_array_new();
    errdefer free(conv_input);
    {
        const parts = [_]mlx.mlx_array{ in.conv_state, in.qkv };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
    }
    if (!rows_engaged) {
        rows_engaged = true;
        log.info("[kda] Ling's exact verify rows in one launch engaged: {d} heads, S={d}\n", .{ in.hv, qsh[1] });
    }
    return .{ .y = out[0], .conv_state = out[1], .state = out[2], .state_seq = out[3], .conv_input = conv_input };
}

// ── the decode step's epilogue (src/ling_kda_epi.zig) ──

var decode_epi_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rows_epi_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The epilogue's o_proj: its suh (f16 [hv dv], o_proj's input width) and the arrival counters its DEC-ONE GEMV reads
/// (at least 8: MLX binds a smaller input in constant space).
pub const OProj = struct { suh: mlx.mlx_array, ncnt: c_int };

/// `decodeStep`'s or `verifyRows`' outputs with the epilogue's: o_proj's prepared rows `xp` f16 [S, hv dv] (lane order),
/// their block sums `xs` f32 [S, hv dv / 128] and `cnt` u32 [ncnt] zeroed; a verify's `state_seq` and `conv_input` (the
/// launch's own copy of the window's inputs), empty at decode. The caller frees each.
pub const EpiOut = struct {
    y: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    state: mlx.mlx_array,
    state_seq: mlx.mlx_array = .{ .ctx = null },
    conv_input: mlx.mlx_array = .{ .ctx = null },
    xp: mlx.mlx_array,
    xs: mlx.mlx_array,
    cnt: mlx.mlx_array,
};

/// `decodeStep` with the epilogue: the same y, window and state, and o_proj's prepared rows of y, their block sums and
/// its zeroed counters. Null where `decodeStep` would be, or where `op` does not fit (suh not f16 [hv dv], fewer than 8
/// counters).
pub fn decodeStepEpi(s: mlx.mlx_stream, in: DecodeIn, op: OProj) !?EpiOut {
    var out: [6]mlx.mlx_array = undefined;
    if (!try launch(s, in, false, true, op, &out)) return null;
    return .{ .y = out[0], .conv_state = out[1], .state = out[2], .xp = out[3], .xs = out[4], .cnt = out[5] };
}

/// `verifyRows` with the epilogue: the same outputs, the window's inputs written by the launch itself (no concatenate),
/// and o_proj's prepared rows of y, their block sums and its zeroed counters. Null where `verifyRows` would be, or where
/// `op` does not fit.
pub fn verifyRowsEpi(s: mlx.mlx_stream, in: DecodeIn, op: OProj) !?EpiOut {
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[1] < 2) return null;
    var out: [8]mlx.mlx_array = undefined;
    if (!try launch(s, in, true, true, op, &out)) return null;
    return .{ .y = out[0], .conv_state = out[1], .state = out[2], .state_seq = out[3], .conv_input = out[4], .xp = out[5], .xs = out[6], .cnt = out[7] };
}

/// The kernel over `in`'s rows, ROWS_EXACT `exact`: y, the next window, the state and (exact) the state after every row
/// into `out`. False where it does not serve, checked before any launch.
fn launchRows(s: mlx.mlx_stream, in: DecodeIn, comptime exact: bool, out: *[if (exact) 4 else 3]mlx.mlx_array) !bool {
    return launch(s, in, exact, false, undefined, out);
}

/// `launchRows`' outputs, then with the epilogue (`epi`) a verify's window inputs and o_proj's xp, xs and counters
fn outCount(comptime exact: bool, comptime epi: bool) usize {
    return (if (exact) 4 else 3) + (if (epi) (if (exact) 4 else 3) else 0);
}

/// `launchRows`' kernel, or with `epi` the same text with the epilogue appended (its operand `op`, unread without it)
fn launch(s: mlx.mlx_stream, in: DecodeIn, comptime exact: bool, comptime epi: bool, op: OProj, out: *[outCount(exact, epi)]mlx.mlx_array) !bool {
    if (!mlx.streamIsGpu(s)) return false;
    if (in.hk != in.hv or in.dk != 128 or in.dv != 128) return false;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != 1 or qsh[1] < 1 or qsh[1] > decode_max_rows) return false;
    const rows = qsh[1];
    const c_dim = 2 * in.hk * in.dk + in.hv * in.dv;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (!isAct(dt) or qsh[2] != c_dim) return false;
    inline for (.{ in.z, in.a, in.b, in.conv_state, in.conv_w, in.norm_w, in.q_scale, in.k_scale }) |x| {
        if (mlx.mlx_array_dtype(x) != dt) return false;
    }
    if (!std.mem.eql(c_int, mlx.getShape(in.z), &[_]c_int{ 1, rows, in.hv * in.dv })) return false;
    if (!std.mem.eql(c_int, mlx.getShape(in.a), &[_]c_int{ 1, rows, in.hv * in.dk })) return false;
    if (!std.mem.eql(c_int, mlx.getShape(in.b), &[_]c_int{ 1, rows, in.hv })) return false;
    if (!std.mem.eql(c_int, mlx.getShape(in.conv_state), &[_]c_int{ 1, 3, c_dim })) return false;
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[0] != c_dim or wsh[1] != 4 or mlx.mlx_array_size(in.conv_w) != @as(usize, @intCast(c_dim * 4))) return false;
    if (mlx.mlx_array_size(in.norm_w) != @as(usize, @intCast(in.dv))) return false;
    if (mlx.mlx_array_dtype(in.A_log) != .float32 or mlx.mlx_array_dtype(in.dt_bias) != .float32) return false;
    if (mlx.mlx_array_size(in.A_log) != @as(usize, @intCast(in.hv * in.dk)) or mlx.mlx_array_size(in.dt_bias) != @as(usize, @intCast(in.hv * in.dk))) return false;
    const st_dt = mlx.mlx_array_dtype(in.state);
    if (!std.mem.eql(c_int, mlx.getShape(in.state), &[_]c_int{ 1, in.hv, in.dv, in.dk }) or (st_dt != .float32 and st_dt != .bfloat16 and st_dt != .float16)) return false;
    // the epilogue's o_proj: one 128-block of its input a head (dv 128 above), its suh f16 [hv dv], its counters
    if (epi and (mlx.mlx_array_dtype(op.suh) != .float16 or mlx.mlx_array_size(op.suh) != @as(usize, @intCast(in.hv * in.dv)) or op.ncnt < 8)) return false;
    const slot = if (epi) (if (exact) &rows_epi_kernel else &decode_epi_kernel) else (if (exact) &rows_kernel else &decode_kernel);
    const k = slot.* orelse blk: {
        const ins = [_][*:0]const u8{ "qkv", "z", "a", "b", "conv_state", "conv_w", "A_log", "dt_bias", "state_in", "norm_w", "q_scale", "k_scale", "qk_eps", "norm_eps", "lb", "sigtab" };
        const ins_epi = ins ++ [_][*:0]const u8{"osuh"};
        const nk = if (epi and exact)
            try newKernelH("ling_kda_verify_rows_epi", &ins_epi, &[_][*:0]const u8{ "y", "conv_out", "state_out", "state_seq", "conv_in", "oxp", "oxs", "ocnt" }, "#define ROWS_EXACT 1\n" ++ kernelSource("kernels/kda/ling_kda_decode.metal") ++ kernelSource("kernels/kda/ling_kda_epi.metal"), kernelSource("kernels/kda/ling_kda_oprep.metal"))
        else if (epi)
            try newKernelH("ling_kda_decode_epi", &ins_epi, &[_][*:0]const u8{ "y", "conv_out", "state_out", "oxp", "oxs", "ocnt" }, "#define ROWS_EXACT 0\n" ++ kernelSource("kernels/kda/ling_kda_decode.metal") ++ kernelSource("kernels/kda/ling_kda_epi.metal"), kernelSource("kernels/kda/ling_kda_oprep.metal"))
        else if (exact)
            try newKernel("ling_kda_verify_rows", &ins, &[_][*:0]const u8{ "y", "conv_out", "state_out", "state_seq" }, "#define ROWS_EXACT 1\n" ++ kernelSource("kernels/kda/ling_kda_decode.metal"))
        else
            try newKernel("ling_kda_decode", &ins, &[_][*:0]const u8{ "y", "conv_out", "state_out" }, "#define ROWS_EXACT 0\n" ++ kernelSource("kernels/kda/ling_kda_decode.metal"));
        slot.* = nk;
        break :blk nk;
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, in.hv * in.dv }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c_dim }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, in.hv, in.dv, in.dk }, 4, st_dt));
    if (exact) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, 1, in.hv, in.dv, in.dk }, 5, st_dt));
    if (epi) {
        if (exact) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3 + rows, c_dim }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, in.hv * in.dv }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, @divExact(in.hv * in.dv, 128) }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{op.ncnt}, 1, .uint32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * in.hv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st_dt));
    inline for (.{ .{ "TOK", rows }, .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "C", c_dim }, .{ "NKEEP", @as(c_int, 3) } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    if (epi) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NCNT", op.ncnt));
    const qe = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(qe);
    const ne = mlx.mlx_array_new_float(in.norm_eps);
    defer _ = mlx.mlx_array_free(ne);
    const lb = mlx.mlx_array_new_float(in.bound);
    defer _ = mlx.mlx_array_free(lb);
    // MLX's own sigmoid of every 16-bit pattern (cached per dtype): the step's sigmoids of T values read it.
    const sigtab = try sigmoidTableFor(s, dt);
    if (epi) {
        try apply(k, cfg, s, &.{ in.qkv, in.z, in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.state, in.norm_w, in.q_scale, in.k_scale, qe, ne, lb, sigtab, op.suh }, out);
    } else {
        try apply(k, cfg, s, &.{ in.qkv, in.z, in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.state, in.norm_w, in.q_scale, in.k_scale, qe, ne, lb, sigtab }, out);
    }
    return true;
}

// ── the blocked prompt recurrence ──

/// The prompt rows from which the blocked recurrence takes a call (below, the stock kernel's per-token launch is cheap).
pub const blocked_min_rows: c_int = 64;
/// A launch shape of the blocked recurrence: simdgroups a threadgroup, tokens a staged block, value rows a simdgroup.
pub const RecShape = struct { ns: c_int, tb: c_int, rows: c_int };
/// The served launch (kernels/kda/ling_kda_blocked_rows.metal at R = `rows`): 8 simdgroups a threadgroup, each carrying
/// 2 value rows (16 a threadgroup), and blocks of 16 tokens whose k, q (activation dtype) and gate (f32) at TB x 128
/// each, v at TB x 16 and beta at TB are staged once a threadgroup: 16.5 KB of the 32 KB threadgroup budget. Each value
/// row's arithmetic is the stock kernel's statement for statement (its 32 dk lanes 4 channels each in the same order, the
/// decay and the k product summed per lane, simd_sum, the delta, the update and the q product, simd_sum, y rounded
/// once), so the words do not depend on the shape: src/ling_kda_test.zig holds this launch to the one it replaced (16
/// simdgroups of one value row, 16-token blocks, kernels/kda/ling_kda_blocked.metal) and to the stock kernel.
pub const rec_shape: RecShape = .{ .ns = 8, .tb = 16, .rows = 2 };
comptime {
    // Ling's 128 value rows split evenly over a threadgroup's simdgroups, each simdgroup 1, 2 or 4 rows, at least one
    // simdgroup and one token a block (the checks the deleted shape table's refusals made: a shape that broke them would
    // make `recurBlocked` decline, and the stock kernel run every prompt, the same words, slower)
    if (!(rec_shape.rows == 1 or rec_shape.rows == 2 or rec_shape.rows == 4)) @compileError("rec_shape: rows must be 1, 2 or 4");
    if (rec_shape.ns < 1 or rec_shape.tb < 1) @compileError("rec_shape: ns and tb must be at least 1");
    if (@rem(128, rec_shape.ns * rec_shape.rows) != 0) @compileError("rec_shape: Ling's 128 value rows must split evenly into ns * rows a threadgroup");
    // 32 lanes a simdgroup, at most 1,024 threads a threadgroup; the staging (16-bit k, q, v and beta, the f32 gate) in
    // the 32 KB of threadgroup memory
    std.debug.assert(32 * rec_shape.ns <= 1024);
    std.debug.assert(rec_shape.tb * (2 * 128 * 2 + 128 * 4 + rec_shape.ns * rec_shape.rows * 2 + 2) <= 32 * 1024);
}
/// The served kernel's name: ling_kda_blocked_r2_ns8_tb16.
const rec_kernel_name = std.fmt.comptimePrint("ling_kda_blocked_r{d}_ns{d}_tb{d}", .{ rec_shape.rows, rec_shape.ns, rec_shape.tb });
var blocked_kernel: ?mlx.mlx_fast_metal_kernel = null;
var blocked_engaged = false;
/// y [B, T, hv, dv] and the new state.
pub const RecurOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// The prompt path's gate as `recurBlocked` takes it: kdaGateChain's (and `gate`'s) per-channel gate, flat f32
/// [B, T, hv * dk], viewed as [B, T, hv, dk]. A reshape of a row-contiguous array, so no copy; the caller frees it.
/// Null where the sizes disagree (the caller keeps the stock kernel).
pub fn gateHeads(s: mlx.mlx_stream, g: mlx.mlx_array, b: c_int, t: c_int, hv: c_int, dk: c_int) !?mlx.mlx_array {
    const n = @as(usize, @intCast(b)) * @as(usize, @intCast(t)) * @as(usize, @intCast(hv)) * @as(usize, @intCast(dk));
    if (mlx.mlx_array_size(g) != n) return null;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, g, &[_]c_int{ b, t, hv, dk }, 4, s));
    return out;
}

/// The vector-gate delta rule over q, k [B, T, hk, 128], v [B, T, hv, dv] (the activation dtype), g f32 [B, T, hv, 128],
/// beta [B, T, hv] (the activation dtype) from `state` [B, hv, dv, 128]: y [B, T, hv, dv] in `out_dtype` and the new
/// state, the stock kernel's bytes, on the served launch (`rec_shape`). Null outside that geometry: under
/// `blocked_min_rows` rows, a head other than 128 keys, value rows that a threadgroup's 16 do not divide.
pub fn recurBlocked(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state: mlx.mlx_array, out_dtype: mlx.mlx_dtype) !?RecurOut {
    if (!mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    if (qs.len != 4 or vs.len != 4 or qs[3] != 128) return null;
    const b = qs[0];
    const t = qs[1];
    const hk = qs[2];
    const hv = vs[2];
    const dv = vs[3];
    if (t < blocked_min_rows or vs[0] != b or vs[1] != t or hk < 1 or @rem(hv, hk) != 0 or @rem(dv, rec_shape.ns * rec_shape.rows) != 0) return null;
    const in_dt = mlx.mlx_array_dtype(q);
    if (!isAct(in_dt) or mlx.mlx_array_dtype(k) != in_dt or mlx.mlx_array_dtype(v) != in_dt or mlx.mlx_array_dtype(beta) != in_dt) return null;
    if (!std.mem.eql(c_int, mlx.getShape(k), qs)) return null;
    if (mlx.mlx_array_dtype(g) != .float32 or !std.mem.eql(c_int, mlx.getShape(g), &[_]c_int{ b, t, hv, 128 })) return null;
    if (!std.mem.eql(c_int, mlx.getShape(beta), &[_]c_int{ b, t, hv })) return null;
    if (!std.mem.eql(c_int, mlx.getShape(state), &[_]c_int{ b, hv, dv, 128 })) return null;
    const kern = blocked_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
        const outs = [_][*:0]const u8{ "y", "state_out" };
        const nk = try newKernel(rec_kernel_name, &ins, &outs, kernelSource("kernels/kda/ling_kda_blocked_rows.metal"));
        blocked_kernel = nk;
        break :blk nk;
    };
    const out = try launchBlocked(kern, s, q, k, v, g, beta, state, out_dtype);
    if (!blocked_engaged) {
        blocked_engaged = true;
        log.info("[kda] Ling's blocked prompt recurrence engaged: T={d}, {d} heads, {d} simdgroups a threadgroup, {d}-token blocks, {d} value rows a simdgroup\n", .{ t, hv, rec_shape.ns, rec_shape.tb, rec_shape.rows });
    }
    ran.mark(.KDA_REC_SHAPE);
    return .{ .y = out[0], .state = out[1] };
}

/// One launch of a blocked recurrence kernel `kern` (ling_kda_blocked_rows.metal's text, or a test's planted copy of it)
/// on the served launch (`rec_shape`): grid (32, dv / R, B * hv), threadgroup (32, NS, 1), the templates InT, StT, OutT,
/// Dk, Dv, Hk, Hv, NS, TB and R in that order. The operands are as `recurBlocked` checked them; y and the new state.
pub fn launchBlocked(kern: mlx.mlx_fast_metal_kernel, s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state: mlx.mlx_array, out_dtype: mlx.mlx_dtype) ![2]mlx.mlx_array {
    const qs = mlx.getShape(q);
    const vs = mlx.getShape(v);
    const b = qs[0];
    const t = qs[1];
    const hk = qs[2];
    const hv = vs[2];
    const dv = vs[3];
    const in_dt = mlx.mlx_array_dtype(q);
    const st_dt = mlx.mlx_array_dtype(state);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, hv, dv }, 4, out_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, hv, dv, 128 }, 4, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(dv, rec_shape.rows), b * hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, rec_shape.ns, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", out_dtype));
    inline for (.{ .{ "Dk", @as(c_int, 128) }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv }, .{ "NS", rec_shape.ns }, .{ "TB", rec_shape.tb }, .{ "R", rec_shape.rows } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    var out: [2]mlx.mlx_array = undefined;
    try apply(kern, cfg, s, &.{ q, k, v, g, beta, state, t_arr }, &out);
    return out;
}
