//! copied_from: src/sdk/blocks/attention/decode_attn_quant.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/blocks/attention/decode_attn_quant.
//! Owns AttnDqCopy, attnDqUseNvfp4, attn_dq_engaged, attn_dq_nvfp4_engaged, and 11 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const Weights = @import("../../weights/resident.zig").Weights;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("attn256.zig").attn256RandBf16;

// ── Decode-only dense-attention requant (mlxfast native-affine class) ──
//
// A dense bf16/f16 attention projection reads its whole weight per decode
// token at the DRAM wall; on Laguna (bf16 attention over nvfp4 experts) the
// four projections are ~3 GB/token of a ~4 GB/token read. An INT8 group-32
// affine side copy halves that traffic. LOSSY BY DESIGN — a real
// requantization of weights the checkpoint ships dense, so this is a
// user-facing toggle (--decode-attn-quant / app Settings), never a sanctioned
// bit-identical class. Prefill, spec-verify and every other multi-token
// forward keep the authoritative dense weight (a quality anchor: KV built by
// prefill stays exact; only decode-step projections carry the perturbation).
pub const AttnDqCopy = struct {
    w: mlx.mlx_array, // packed codes, [out, in*bits/32] u32
    s: mlx.mlx_array,
    b: mlx.mlx_array, // null-ctx for nvfp4 (no biases by construction)
    // The dispatch reads the copy's OWN geometry — a tail layer's nvfp4 copy
    // and an early layer's int8 copy ride the same qmatmulBits call.
    bits: u32 = 8,
    group_size: u32 = 32,
    mode: QuantMode = .affine,
};

/// Which side-copy quantization a layer gets under --decode-attn-quant. Late
/// layers amplify quantization error far less (mlxfast measured ~15x lower on
/// layers >= 32 of 40), so the tail can afford real nvfp4-g16 (half the int8
/// bytes) while early layers keep int8-g32. Default tail = the last 20% of
/// layers (`attnDqNvfp4From`).
pub fn attnDqUseNvfp4(layer: u32, num_layers: u32) bool {
    if (num_layers == 0) return false;
    return layer >= attnDqNvfp4From(num_layers);
}

pub fn attnDqNvfp4From(num_layers: u32) u32 {
    return (num_layers * 4) / 5;
}

pub var attn_dq_engaged: bool = false;
pub var attn_dq_nvfp4_engaged: bool = false;

/// Default ON ( characterization on Laguna XS: decode forward
/// 13.285 -> 10.205 ms median, 3/3 paired boots both orders; greedy outputs
/// across six code/reasoning prompts reached identical answers with
/// wording-level divergence only). Flip together with the mirrored Swift
/// `ServerOptions` default.
pub const DECODE_ATTN_QUANT_DEFAULT = true;

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const CliState = struct {
    /// Set by the CLI (`--decode-attn-quant` / `--no-decode-attn-quant`) before any load, so every serve path
    /// (headless included) sees one value.
    decode_attn_quant_flag: ?bool = null,
};
pub const cli_shared = @import("../../util/shared.zig").Shared(CliState, "decode_attn_quant_cli", .{});
pub inline fn cli() *CliState {
    return cli_shared.get();
}

/// Whether the user EXPLICITLY asked for --decode-attn-quant (CLI flag, app
/// toggle) as opposed to riding the silent default. Archs whose
/// characterization showed ANSWER MOVEMENT under the requant gate on this
/// form (dsv4 comp_in: one duplicated paragraph on 1/7 greedy prompts,
/// ); laguna's clean characterization keeps the plain default.
pub fn decodeAttnQuantExplicit() bool {
    if (decode_attn_quant_override) |v| return v; // test seam counts as explicit
    return cli().decode_attn_quant_flag orelse false;
}

pub fn decodeAttnQuantEnabled() bool {
    if (decode_attn_quant_override) |v| return v;
    return cli().decode_attn_quant_flag orelse DECODE_ATTN_QUANT_DEFAULT;
}

/// Whether `--decode-attn-quant` has anything to requantize: a text-trunk attention
/// projection shipped dense. Fully quantized packs decode exactly as without it.
pub fn weightsHaveDenseAttnProj(weights: *const Weights) bool {
    const projs = [_][]const u8{ ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight", ".q_k_v_proj.weight", ".wq_du.weight", ".wk_dv.weight", ".wv_dv.weight", ".wo_ud.weight", ".wr_du.weight" };
    var it = weights.map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (std.mem.indexOf(u8, name, "attn") == null) continue;
        if (std.mem.indexOf(u8, name, "vision") != null or std.mem.indexOf(u8, name, "visual") != null or std.mem.indexOf(u8, name, "audio") != null) continue;
        for (projs) |suffix| {
            if (!std.mem.endsWith(u8, name, suffix)) continue;
            var buf: [512]u8 = undefined;
            const scales = std.fmt.bufPrint(&buf, "{s}.scales", .{name[0 .. name.len - ".weight".len]}) catch break;
            if (decodeAttnQuantEligible(mlx.getShape(e.value_ptr.*), mlx.mlx_array_dtype(e.value_ptr.*), weights.map.contains(scales))) return true;
        }
    }
    return false;
}

/// Whether a loaded dense projection weight (pre-transposed [in, out]) can
/// carry an INT8-g32 side copy. The conditions are the quantized dispatch's
/// own: dense, a float dtype the requant helps, 2-D, in-dim in whole groups.
pub fn decodeAttnQuantEligible(w_shape: []const c_int, dtype: mlx.mlx_dtype, has_scales: bool) bool {
    if (has_scales) return false; // already quantized
    if (dtype != .bfloat16 and dtype != .float16) return false;
    if (w_shape.len != 2) return false;
    return @rem(w_shape[0], 32) == 0;
}

/// Quantize a dense pre-transposed [in, out] weight into a side copy
/// ([out, in] packed — served by `mlx_quantized_matmul` with transpose=true,
/// same as any checkpoint-quantized linear). `nvfp4` picks real nvfp4-g16
/// (half the int8 bytes, e4m3 scales, no biases) over the default INT8-g32
/// affine. Materialized eagerly so the build never rides a token's graph.

pub fn buildAttnDqCopy(s: mlx.mlx_stream, w: mlx.mlx_array, nvfp4: bool) !?AttnDqCopy {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    const perm = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&wt, w, &perm, 2, s));
    var wt_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt_c);
    try mlx.check(mlx.mlx_contiguous(&wt_c, wt, false, s));
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    const gs: c_int = if (nvfp4) 16 else 32;
    const bits: c_int = if (nvfp4) 4 else 8;
    try mlx.check(mlx.mlx_quantize(
        &parts,
        wt_c,
        mlx.mlx_optional_int.some(gs),
        mlx.mlx_optional_int.some(bits),
        if (nvfp4) "nvfp4" else "affine",
        .{ .ctx = null },
        s,
    ));
    if (mlx.mlx_vector_array_size(parts) != @as(usize, if (nvfp4) 2 else 3)) return null;
    var copy = AttnDqCopy{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = if (nvfp4) .{ .ctx = null } else mlx.mlx_array_new(),
        .bits = @intCast(bits),
        .group_size = @intCast(gs),
        .mode = if (nvfp4) .nvfp4 else .affine,
    };
    try mlx.check(mlx.mlx_vector_array_get(&copy.w, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&copy.s, parts, 1));
    if (!nvfp4) try mlx.check(mlx.mlx_vector_array_get(&copy.b, parts, 2));
    _ = mlx.mlx_array_eval(copy.w);
    _ = mlx.mlx_array_eval(copy.s);
    if (copy.b.ctx != null) _ = mlx.mlx_array_eval(copy.b);
    return copy;
}

pub var decode_attn_quant_override: ?bool = null; // test seam

// ── tests ──

const qmatmulBits = @import("../linear/qmatmul.zig").qmatmulBits;
