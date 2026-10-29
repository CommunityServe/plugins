//! copied_from: src/sdk/blocks/attention/gates.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/attention/gates.
//! Owns ATTN_GATE_SOURCE, attn_gate_kernel, attn_gate_engaged, attn_gate_fused_override, attn_gate_fused_env, and 18 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const testing = std.testing;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const gdnParityRand = @import("../gdn/decode_fused.zig").gdnParityRand;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const swigluFusedEnabled = @import("../mlp/fused_activations.zig").swigluFusedEnabled;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const ShapeKey = @import("../moe/router_kernel.zig").ShapeKey;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;

/// `x * sigmoid(g)`: the sigmoid from the per-dtype table (mlx_sigmoid's own
/// values), one multiply. With GROUP > 1 each gate value covers GROUP
/// consecutive x elements (Spark's per-head gate).
pub const SIGMOID_GATE_MUL_SOURCE = kernelSource("kernels/attention/sigmoid_gate_mul.metal");

var sig_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
const SigGateCfgKey = struct { shape: ShapeKey, group: c_int, dtype: mlx.mlx_dtype };
var sig_gate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var sig_gate_cfg_key: SigGateCfgKey = std.mem.zeroes(SigGateCfgKey);
var sig_gate_engaged: bool = false;

/// `x * sigmoid(g)` in one dispatch; `heads` > 0 means g is [.., heads] and
/// gates x's last dim per head, 0 means g has x's shape. Bit-identical to
/// mlx_sigmoid + mlx_multiply (pinned by the `fused sigmoid gate` test).
/// Null → caller keeps the ops.
// `x * sigmoid(gate)` over two equal-shape 4-D views read through their own
// strides (a transposed attention output, a split q|gate projection), written
// contiguous [B, S, H, D]; sigmoid from MLX's own table, so bit-equal to the pair.
pub const STRIDED_SIGMOID_GATE_MUL_SOURCE = kernelSource("kernels/attention/strided_sigmoid_gate_mul.metal");

var strided_sig_gate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var strided_sig_gate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var strided_sig_gate_key: SigGateCfgKey = std.mem.zeroes(SigGateCfgKey);

pub fn stridedSigmoidGateMul(s: mlx.mlx_stream, x: mlx.mlx_array, g: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(g) != dt) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 4 or !std.mem.eql(c_int, xsh, mlx.getShape(g))) return null;
    const n: i64 = @as(i64, xsh[0]) * xsh[1] * xsh[2] * xsh[3];
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const tg: c_int = @min(@as(c_int, 256), @divTrunc(ni + 31, 32) * 32);
    const key = SigGateCfgKey{ .shape = ShapeKey.from(xsh), .group = 1, .dtype = dt };
    if (strided_sig_gate_cfg == null or !std.meta.eql(strided_sig_gate_key, key)) {
        if (strided_sig_gate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, xsh.ptr, xsh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        strided_sig_gate_cfg = config;
        strided_sig_gate_key = key;
    }
    if (strided_sig_gate_kernel == null) {
        const input_names = [_][*:0]const u8{ "x", "gate", "sigtab", "N_size" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_strided_sigmoid_gate_mul", in_vec, out_vec, STRIDED_SIGMOID_GATE_MUL_SOURCE, "", false, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        strided_sig_gate_kernel = kh;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const inputs_arr = [_]mlx.mlx_array{ x, g, sigtab, cachedScalarInt(ni) };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, strided_sig_gate_kernel.?, inputs_vec, strided_sig_gate_cfg.?, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

pub fn fusedSigmoidGateMul(s: mlx.mlx_stream, x: mlx.mlx_array, g: mlx.mlx_array, heads: c_int) !?mlx.mlx_array {
    if (!swigluFusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(g) != dt) return null;
    const xsh = mlx.getShape(x);
    const gsh = mlx.getShape(g);
    if (xsh.len == 0 or xsh.len != gsh.len or xsh.len > 5) return null;
    var n: i64 = 1;
    var gn: i64 = 1;
    for (xsh, gsh, 0..) |a, b, i| {
        n *= a;
        gn *= b;
        if (i + 1 < xsh.len and a != b) return null;
    }
    const last_x = xsh[xsh.len - 1];
    const last_g = gsh[gsh.len - 1];
    const group: c_int = if (heads > 0) @divTrunc(last_x, heads) else 1;
    if (group < 1 or last_g * group != last_x) return null;
    if (n <= 0 or n > std.math.maxInt(c_int)) return null;
    const ni: c_int = @intCast(n);
    const tg: c_int = @min(@as(c_int, 256), @max(@as(c_int, 32), @divTrunc(ni + 31, 32) * 32));
    const key = SigGateCfgKey{ .shape = ShapeKey.from(xsh), .group = group, .dtype = dt };
    if (sig_gate_cfg == null or !std.meta.eql(sig_gate_cfg_key, key)) {
        if (sig_gate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, xsh.ptr, xsh.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(ni + tg - 1, tg) * tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GROUP", group));
        sig_gate_cfg = config;
        sig_gate_cfg_key = key;
    }
    if (sig_gate_kernel == null) {
        const input_names = [_][*:0]const u8{ "x", "gate", "sigtab", "N_size" };
        const output_names = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kh = mlx.mlx_fast_metal_kernel_new("mlxserve_sigmoid_gate_mul", in_vec, out_vec, SIGMOID_GATE_MUL_SOURCE, "", true, false);
        if (kh.ctx == null) return error.MetalKernelCompileFailed;
        sig_gate_kernel = kh;
    }
    const sigtab = try swigluSigTable(s, dt, std.heap.c_allocator);
    const inputs_arr = [_]mlx.mlx_array{ x, g, sigtab, cachedScalarInt(ni) };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, sig_gate_kernel.?, inputs_vec, sig_gate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!sig_gate_engaged) {
        sig_gate_engaged = true;
        log.info("[attn] fused sigmoid gate engaged: n={d} group={d}\n", .{ ni, group });
    }
    return y;
}

// ── tests ──
