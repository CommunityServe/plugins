//! copied_from: src/sdk/blocks/moe/gather_qmv_rows.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/moe/gather_qmv_rows.
//! Owns gatherQmvGateUpRowsSource, GQMV_GATEUP_ROWS_SOURCES, GQMV_GATEUP_ROWS_NAMES, gqmv_gateup_rows_kernels, getGatherQmvGateUpRowsKernel, GateUpRowsCfgKey, and 19 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const standin = @import("../../diag/standin.zig");
const testing = std.testing;
const qwen4Standin = @import("../../diag/standin.zig").qwen4Standin;
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const gatherQmvDownReduce = @import("down_reduce.zig").gatherQmvDownReduce;
const gatherQmvDownReduceRows = @import("down_reduce.zig").gatherQmvDownReduceRows;
const GQMV_AFFINE_HEADER = @import("gather_qmv.zig").GQMV_AFFINE_HEADER;
const GQMV_GATEUP_BODY_AFFINE = @import("gather_qmv.zig").GQMV_GATEUP_BODY_AFFINE;
const GQMV_GATEUP_BODY_NVFP4 = @import("gather_qmv.zig").GQMV_GATEUP_BODY_NVFP4;
const GQMV_NVFP4_HEADER = @import("gather_qmv.zig").GQMV_NVFP4_HEADER;
const affineQmvPackUnits = @import("gather_qmv.zig").affineQmvPackUnits;
const affineQmvUnitOk = @import("gather_qmv.zig").affineQmvUnitOk;
const gatherQmvGateUp = @import("gather_qmv.zig").gatherQmvGateUp;
const gatherQmvGateUpEnabled = @import("gather_qmv.zig").gatherQmvGateUpEnabled;

fn gatherQmvGateUpRowsSource(comptime nvfp4: bool) [:0]const u8 {
    return std.fmt.comptimePrint(kernelSource("kernels/moe/gather_qmv_gateup_rows.fmt.metal"), .{
        if (nvfp4) GQMV_GATEUP_BODY_NVFP4 else GQMV_GATEUP_BODY_AFFINE,
        if (nvfp4) " * 4194304.0f" else "",
        if (nvfp4) " * 4194304.0f" else "",
        // ModelOpt banks: each expert's gate scale lands before the SiLU, rounded as the chain does.
        if (nvfp4) "GSCALE != 0 ? T(T(acc_g) * gsc[eid]) : T(acc_g)" else "T(acc_g)",
    });
}

pub const GQMV_GATEUP_ROWS_SOURCES = [2][:0]const u8{ gatherQmvGateUpRowsSource(false), gatherQmvGateUpRowsSource(true) };
const GQMV_GATEUP_ROWS_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv_gateup_rows", "mlxserve_moe_gather_qmv_gateup_rows_nvfp4" };
var gqmv_gateup_rows_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvGateUpRowsKernel(nvfp4: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    if (gqmv_gateup_rows_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "K_size", "N_size", "gsc" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_GATEUP_ROWS_NAMES[mi],
        in_vec,
        out_vec,
        GQMV_GATEUP_ROWS_SOURCES[mi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_rows_kernels[mi] = kernel;
    return kernel;
}

const GateUpRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, gscale: bool = false };
var gateup_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_rows_cfg_key: GateUpRowsCfgKey = std.mem.zeroes(GateUpRowsCfgKey);
var gqmv_gateup_rows_engaged: bool = false;

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const MoeStatsState = struct {
    moe_rows_fused_layers: u32 = 0,
    moe_sorted_layers: u32 = 0,
};
pub const moeStats_shared = @import("../../util/shared.zig").Shared(MoeStatsState, "moe_rows_stats", .{});
const device = @import("../../../mlx/device.zig");
const gather_qmv_vars = @import("gather_qmv.zig");
const down_reduce_vars = @import("down_reduce.zig");
pub inline fn moeStats() *MoeStatsState {
    return moeStats_shared.get();
}

pub fn moeRowsFusedEnabled() bool {
    return moe_rows_fused_override orelse true;
}

pub var moe_verify_rows_override: ?bool = null; // test seam
pub var moe_verify_rows_logged: bool = false;

/// Single-slot verify widths (B == 1, 2 <= S <= 8) through the rows arm (M4 Max Flash Next S=2 26.9 -> 23.5 ms);
/// the test seam false keeps the sorted path.
fn moeVerifyRowsEnabled() bool {
    return moe_verify_rows_override orelse true;
}

/// The CUDA fused MoE decode's format for these banks, or null where it declines: one fp mode on
/// every bank, at most 16 rows, plain SiLU, and no per-expert bias, up scale or clamp.
pub fn moeCudaFormat(rows: c_int, gate: QuantMode, up: QuantMode, down: QuantMode, silu: bool, has_bias: bool, has_up_scale: bool, clamped: bool) ?@import("moe_cuda.zig").Format {
    if (rows < 1 or rows > 16 or !silu or has_bias or has_up_scale or clamped) return null;
    if (gate != up or gate != down) return null;
    return switch (gate) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => null,
    };
}

pub const MoeDecodeDispatchArm = enum { rows, sorted, gather_qmv };

/// ONE place decides the decode MoE arm: fused rows for S == 1 and
/// 2 <= B <= 8, or B == 1 and 2 <= S <= 8 (verify), without expert bias or a clamp; the
/// single-row gatherQmv at B == 1; the sorted chain otherwise, and whenever a
/// stand-in diagnostic wants the stock path.
pub fn moeDecodeDispatchArm(B: c_int, S: c_int, K: c_int, has_expert_bias: bool, clamped: bool) MoeDecodeDispatchArm {
    // A per-expert bias or a clamped SwiGLU: the fused rows kernels apply neither, only the composed chain does.
    const composed = has_expert_bias or clamped;
    const total_inds: c_int = B * S * K;
    const do_sort = B * S > 1 or total_inds >= 64 or composed;
    const stock = composed or qwen4Standin().moe_gateup or qwen4Standin().moe_down;
    if (S == 1 and B >= 2 and B <= 8 and moeRowsFusedEnabled() and !stock) return .rows;
    if (B == 1 and S >= 2 and S <= 8 and moeVerifyRowsEnabled() and !stock) return .rows;
    if (do_sort) return .sorted;
    return .gather_qmv;
}

/// Row-axis twin of `gatherQmvGateUp` for the batched decode tick (S = 1,
/// 2 <= B <= 8): the grid runs over output tiles x (row, expert choice) and
/// each row is the single-row arithmetic unchanged, so the result is
/// bit-identical per row to B solo calls. Shares the single-row Metal body.
pub fn gatherQmvGateUpRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    gate_scale: ?mlx.mlx_array,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (gate_scale != null and mode != .nvfp4) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(gs_arr) != .uint8 or mlx.mlx_array_dtype(us_arr) != .uint8) return null;
        if (gb.ctx != null or ub.ctx != null) return null;
    } else {
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (gb.ctx == null or ub.ctx == null) return null;
    }
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 8 or topk < 1 or topk > 16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2 or xsh[0] != nrows or xsh[1] != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    if (gate_scale) |g| if (mlx.mlx_array_dtype(g) != xd) return null;
    const key = GateUpRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .gscale = gate_scale != null };
    if (gateup_rows_cfg == null or !std.meta.eql(gateup_rows_cfg_key, key)) {
        if (gateup_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 3, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, nrows * topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPK", topk));
        if (nvfp4) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSCALE", @intFromBool(gate_scale != null)));
        gateup_rows_cfg = config;
        gateup_rows_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    // Without a gate scale the kernel never reads `gsc`; any array binds.
    const nvfp4_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, uw, us_arr, inds, sigtab, K_arr, N_arr, gate_scale orelse sigtab };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpRowsKernel(nvfp4);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_rows_engaged) {
        gqmv_gateup_rows_engaged = true;
        log.info("[moe] fused rows gate+up+SwiGLU kernel engaged: B={d} E_topk={d} inter={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

pub var moe_rows_fused_override: ?bool = null;

// ── tests ──
