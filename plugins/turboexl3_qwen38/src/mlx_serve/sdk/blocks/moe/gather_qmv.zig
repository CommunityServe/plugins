//! copied_from: src/sdk/blocks/moe/gather_qmv.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/moe/gather_qmv.
//! Owns gatherQmvSource, GQMV_BODY_AFFINE, GQMV_BODY_NVFP4, GQMV_AFFINE_HEADER, GQMV_NVFP4_HEADER, gatherQmvGateUpSource, and 28 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const moe_fp4 = @import("moe_fp4.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const testing = std.testing;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const clampedSwiGLU = @import("../mlp/activations.zig").clampedSwiGLU;
const fusedSwiGLU = @import("../mlp/fused_activations.zig").fusedSwiGLU;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const moeCudaFormat = @import("gather_qmv_rows.zig").moeCudaFormat;

// ── MoE decode gather-qmv kernel ──
// MLX's `gather_qmm` at decode width costs O(EXPERT BANK SIZE), not O(work), in
// any C/C++ process (it is flat under pip-Python with the identical dylib —
// measured E-sweep in the `MoE decode gather` µbench: 30/50/77/300 us at banks
// of 8/25/50/201 MB, all reading the SAME 10 experts). 141 gathers/token x 300 us
// is the whole 42 ms/token Laguna decode. `take` from the same 201 MB bank does
// NOT pay it, so indexed reads are fine; the penalty is specific to gather_qmm's
// addressing.
//
// The shipped workaround (`batchedExpertMm`: take + broadcast + batched qmm)
// dodges it but pays a materialization round-trip — read 9.8 MB of experts,
// write 9.8 MB, read 9.8 MB back = 3x the ideal traffic, which is exactly its
// measured 70 us against a 67 us bandwidth prediction. It cannot go faster.
//
// This kernel reads the bank IN PLACE with GPU-resident indices: one simdgroup
// per (expert slot, output row), each lane striding the packed row and reducing
// with simd_sum. Ideal traffic, no copy, no host sync for the top-K indices.
//
// `X_PER_EXPERT` picks the input layout: 0 = one shared x [K] (gate/up
// projections, every expert sees the same token), 1 = x [TOPK, K] (the down
// projection, where each expert consumes its own activation).
//
// `NVFP4` picks the dequant arithmetic: false = affine (q * scale + bias, the
// mlx-quantized checkpoints), true = NVFP4 (e2m1 nibbles + uint8 e4m3 group
// scales, no biases — poolside's Laguna checkpoints).
fn gatherQmvSource(comptime nvfp4: bool, comptime x_per_expert: bool) [:0]const u8 {
    // Everything is indexed off the raw buffer names rather than through
    // explicitly address-spaced pointers: MLX generates the signature from each
    // input's ACTUAL dtype and puts arrays with fewer than 8 elements in
    // `constant` instead of `device` (backend/common/metal_kernel.cpp), so a
    // hand-declared `const device T*` would mismatch on both counts. `x` in
    // particular arrives as float32 on Laguna while scales/biases are bf16.
    //
    // A uint4 (128-bit) load variant was measured and is NOT kept: 38.9 us vs
    // 37.3 for the scalar form. At ~5 ALU ops per dequantized value this loop
    // is ALU-bound, not load-width bound, so widening the loads only adds code.
    return std.fmt.comptimePrint(kernelSource("kernels/moe/gather_qmv.fmt.metal"), .{
        if (x_per_expert) "(size_t)e * (size_t)K" else "0",
        if (nvfp4) GQMV_BODY_NVFP4 else GQMV_BODY_AFFINE,
        // NVFP4 decodes both halves scaled down by a power of two (2^14 for the
        // e2m1 nibble, 2^8 for the e4m3 scale). Multiplying by a power of two is
        // exact in float, so the two constants fold into ONE 2^22 multiply on
        // the finished dot product instead of two per value.
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_BODY_AFFINE = kernelSource("kernels/moe/gqmv_body_affine.metal");

// The group scale is constant across a whole packed word (GS 16 >= VPW 8), so
// the word's four partial dot products are summed unscaled and the scale is
// applied once per word rather than once per value.
pub const GQMV_BODY_NVFP4 = kernelSource("kernels/moe/gqmv_body_nvfp4.metal");

// An e2m1 nibble and an e4m3 byte each become a `half` by shifting their bits
// straight into the half exponent/mantissa/sign fields — no LUT, no branches,
// exact for all codes including zero, subnormals and negative zero. The e2m1
// field layout (2-bit exp, 1-bit mantissa) lands in half's low exponent bits
// and top mantissa bit, which yields the true value scaled by 2^-14; e4m3's
// (4-bit exp, 3-bit mantissa) yields it scaled by 2^-8. Callers fold both back
// in with one multiply. Matches MLX's own `fp4.h` / `fp8.h` conversions, whose
// sign handling is a branch instead of a shift.
// One packed unit of quantized values as a uint32: a word for 2/4/8-bit; at
// 3-bit, mx.quantize packs 8 values into 3 bytes (value i at bit 3i), so the
// unit is a byte triple and the body's `>> (i * BITS)` walk is unchanged.
pub const GQMV_AFFINE_HEADER = kernelSource("kernels/moe/gqmv_affine_header.metal");

pub const GQMV_NVFP4_HEADER = GQMV_AFFINE_HEADER ++ moe_fp4.E2M1_HEADER ++ moe_fp4.E2M1X2_HEADER ++ kernelSource("kernels/moe/gqmv_nvfp4_header.metal");

// ── Fused routed gate+up+SwiGLU (one kernel for what was three) ──
//
// The decode expert path ran gate gather -> up gather -> activation as three
// dispatches, all on the critical chain into down_proj. This computes BOTH dot
// products in the SAME simdgroup and applies the activation before it writes,
// so the x element loaded for the gate FMA is reused for the up FMA and the two
// dequant chains interleave instead of running as separate launches.
//
// It halves the simdgroup count rather than the work (each does two rows'
// worth), which is the right trade here: 4096 simdgroups over 40 cores was ~100
// per core, so there is occupancy to spend on instruction-level parallelism.
//
// Bit-exactness comes from rounding each accumulator to T FIRST — exactly where
// `gatherQmv` writes `T(acc)` — and then running the same table-lookup SwiGLU
// `fusedSwiGLU` does. No f32 shortcut through the activation.
fn gatherQmvGateUpSource(comptime nvfp4: bool) [:0]const u8 {
    return std.fmt.comptimePrint(kernelSource("kernels/moe/gather_qmv_gateup.fmt.metal"), .{
        if (nvfp4) GQMV_GATEUP_BODY_NVFP4 else GQMV_GATEUP_BODY_AFFINE,
        if (nvfp4) " * 4194304.0f" else "",
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_GATEUP_BODY_AFFINE = kernelSource("kernels/moe/gqmv_gateup_body_affine.metal");

pub const GQMV_GATEUP_BODY_NVFP4 = kernelSource("kernels/moe/gqmv_gateup_body_nvfp4.metal");

pub const GQMV_GATEUP_SOURCES = [2][:0]const u8{ gatherQmvGateUpSource(false), gatherQmvGateUpSource(true) };
const GQMV_GATEUP_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv_gateup", "mlxserve_moe_gather_qmv_gateup_nvfp4" };
var gqmv_gateup_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvGateUpKernel(nvfp4: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    if (gqmv_gateup_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_GATEUP_NAMES[mi],
        in_vec,
        out_vec,
        GQMV_GATEUP_SOURCES[mi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_kernels[mi] = kernel;
    return kernel;
}

/// One quant group per pack: a word for 2/4/8-bit, a byte triple (8 values) for 3-bit.
pub fn affineQmvUnitOk(bits: u32, group_size: u32) bool {
    return switch (bits) {
        2, 4, 8 => group_size % (32 / bits) == 0,
        3 => group_size % 8 == 0,
        else => false,
    };
}

pub fn affineQmvPackUnits(K: c_int, bits: u32) c_int {
    return if (bits == 3) @divExact(K, 8) else @divExact(K * @as(c_int, @intCast(bits)), 32);
}

const GateUpCfgKey = struct { topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, limit: u32 };
var gateup_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_cfg_key: GateUpCfgKey = std.mem.zeroes(GateUpCfgKey);
var gqmv_gateup_engaged: bool = false;


pub fn gatherQmvGateUpEnabled() bool {
    return gqmv_gateup_override orelse true;
}

/// One kernel for `silu(gather(gate) @ x) * (gather(up) @ x)` at decode width.
/// Returns [TOPK, N] in x's dtype, or null when anything is outside the
/// supported set (caller falls back to the two separate gathers + activation).
pub fn gatherQmvGateUp(
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
    /// A whole-number `swiglu_limit` the activation clamps at (0 = none).
    limit: u32,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
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
    // The activation is a 16-bit table lookup, exactly as `fusedSwiGLU` gates
    // it; f32 activations keep the unfused path so the two can never disagree.
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    // gate and up must be the SAME geometry — the kernel walks one index set
    // through both banks.
    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpCfgKey{ .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .limit = limit };
    if (gateup_cfg == null or !std.meta.eql(gateup_cfg_key, key)) {
        if (gateup_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "LIMIT", @intCast(limit)));
        gateup_cfg = config;
        gateup_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, uw, us_arr, inds, sigtab, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpKernel(nvfp4);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_engaged) {
        gqmv_gateup_engaged = true;
        log.info("[moe] fused gate+up+SwiGLU kernel engaged: E_topk={d} inter={d} bits={d} gs={d}\n", .{ topk, N, bits, group_size });
    }
    return y;
}

// Indexed [nvfp4][x_per_expert].
pub const GQMV_SOURCES = [2][2][:0]const u8{
    .{ gatherQmvSource(false, false), gatherQmvSource(false, true) },
    .{ gatherQmvSource(true, false), gatherQmvSource(true, true) },
};
const GQMV_NAMES = [2][2][*:0]const u8{
    .{ "mlxserve_moe_gather_qmv", "mlxserve_moe_gather_qmv_px" },
    .{ "mlxserve_moe_gather_qmv_nvfp4", "mlxserve_moe_gather_qmv_nvfp4_px" },
};
var gqmv_kernels: [2][2]?mlx.mlx_fast_metal_kernel = @splat(@splat(null));

fn getGatherQmvKernel(nvfp4: bool, x_per_expert: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    const xi: usize = @intFromBool(x_per_expert);
    if (gqmv_kernels[mi][xi]) |k| return k;
    // NVFP4 carries no biases, so that input is absent from its signature
    // rather than passed as a dummy — MLX builds the signature from the input
    // list, and a phantom array would be one more buffer bound per dispatch.
    const affine_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_NAMES[mi][xi],
        in_vec,
        out_vec,
        GQMV_SOURCES[mi][xi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_kernels[mi][xi] = kernel;
    return kernel;
}

/// Everything a `gatherQmv` config bakes in; one slot per input layout, since
/// the gate/up (shared-x) and down (per-expert-x) calls alternate every layer.
const GqmvCfgKey = struct {
    topk: c_int,
    n: c_int,
    bits: u32,
    gs: u32,
    dtype: mlx.mlx_dtype,
    x_per_expert: bool,
    act2: bool = false,
};
const GqmvCfgSlot = struct {
    cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    key: GqmvCfgKey = std.mem.zeroes(GqmvCfgKey),
};
var gqmv_cfg_cache: [2]GqmvCfgSlot = .{ .{}, .{} };

pub var gqmv_engaged: bool = false;
pub var decode_ladder_engaged: bool = false;

/// Decode-width gathered qmv over an expert bank, read in place.
/// `x`: [K] shared (x_per_expert=false) or [TOPK, K] (true), bf16/fp16.
/// `w`/`sc`/`bi`: the affine bank, [E, N, K*BITS/32] / [E, N, K/GS].
/// `inds`: [TOPK] uint32 bank rows. Returns [TOPK, N], or null when the shape
/// or quant geometry is outside the supported set (caller falls back).
pub fn gatherQmv(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
) !?mlx.mlx_array {
    return gatherQmvAct(s, x, w, sc, bi, inds, bits, group_size, mode, x_per_expert, false);
}

/// `gatherQmv` with an optional ReLU^2 on the rounded output (`act_relu2`):
/// `T(max(T(acc), 0)^2)`, the composed chain's bits.
pub fn gatherQmvAct(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
    act_relu2: bool,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    // Two dequant arithmetics, one kernel shape. Any other mode (mxfp4/mxfp8,
    // whose scales are e8m0 over different group sizes) falls back — running a
    // bank through the wrong dequant is silently wrong, not a crash.
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        // NVFP4 is bits 4 / group 16 by definition, with uint8 e4m3 scales and
        // no biases; anything else claiming the mode is not a bank we can read.
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(sc) != .uint8) return null;
        if (bi.ctx != null) return null;
    } else {
        // Only widths where one uint32 holds a whole number of values (5/6-bit
        // affine is byte-packed differently — those fall back).
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (bi.ctx == null) return null;
    }
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    const want: i64 = if (x_per_expert) @as(i64, topk) * K else K;
    if (xelems != want) return null;

    // One simdgroup per (slot, row); 8 simdgroups to a threadgroup.
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    // CACHED, not rebuilt per call. This runs three times per MoE layer, and a
    // fresh config is ~10 FFI calls: measured `ops/forward` 3226 vs 2329 and
    // +0.4 ms of CPU graph build per token on Laguna XS, which is roughly 60%
    // of the margin this path was being charged against stock `gather_qmm`.
    const key = GqmvCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd, .x_per_expert = x_per_expert, .act2 = act_relu2 };
    const slot = &gqmv_cfg_cache[@intFromBool(x_per_expert)];
    if (slot.cfg == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ACT2", act_relu2));
        slot.* = .{ .cfg = cfg, .key = key };
    }
    const config = slot.cfg.?;

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, w, sc, bi, inds, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, w, sc, inds, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvKernel(nvfp4, x_per_expert);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

pub var gqmv_gateup_override: ?bool = null;

// ── tests ──
