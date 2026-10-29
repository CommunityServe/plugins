//! copied_from: src/engines/mlx/load/weights.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/load/weights.
//! Owns structFields, transposeBf16WeightPub, transposeBf16Weight, depthwiseConvNeedsTranspose, depthwise_conv_layout_logged, maybeTransposeDepthwiseConv, and 41 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const WeightField = @import("../transformer.zig").WeightField;
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const Weights = @import("../../../sdk/config/model.zig").Weights;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const loadHcWeights = @import("../../../plugins/qwen4_exp/load.zig").loadHcWeights;
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const MoeMlpWeights = @import("../../../sdk/blocks/moe/weights.zig").MoeMlpWeights;
const checkExl3Bank = @import("exl3_bank.zig").checkExl3Bank;
const affineParamsFromGeometry = @import("../../../sdk/quant/params.zig").affineParamsFromGeometry;
const weights_load = @import("../../../sdk/weights/load.zig");
const maxAbsDiffF32 = @import("../../../sdk/testing/parity.zig").maxAbsDiffF32;
const HybridMlpWeights = @import("../layer_weights.zig").HybridMlpWeights;
const testing = @import("../kv/cache.zig").testing;
const FullAttnWeights = @import("../layer_weights.zig").FullAttnWeights;
const QW = @import("../layer_weights.zig").QW;

pub fn structFields(comptime T: type) []const WeightField {
    return comptime blk: {
        const names = std.meta.fieldNames(T);
        const types = std.meta.fieldTypes(T);
        var result: [names.len]WeightField = undefined;
        for (names, types, 0..) |name, ty, i| result[i] = .{ .name = name, .type = ty };
        const final = result;
        break :blk &final;
    };
}

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
/// Exported for module-owned archs that load a dense weight outside
/// `initMoeLayers`' owned-list machinery.
pub fn transposeBf16WeightPub(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return transposeBf16Weight(w, s);
}

pub const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;

/// A depthwise conv weight is `[C, K, 1]` in MLX and `[C, 1, K]` in torch, and
/// which one a checkpoint ships is a CONVERTER choice, not an arch property:
/// mlx_lm's LFM2 conversion writes the MLX layout, while LiquidAI's own bf16
/// repos (LFM2.5-1.2B-Instruct, LFM2.5-2.6B) ship the torch one straight out of
/// the training code. This is the `patchProjLayout` class, with one mercy —
/// reading it wrong is not a plausible-but-wrong number, it is a hard abort:
/// mlx_conv1d solves in_channels off the LAST axis, so a [2048, 1, 3] weight at
/// groups=2048 demands 6144 input channels and kills the load at warmup.
///
/// `K == 1` is the one shape the layouts share, and it is its own transpose.
pub fn depthwiseConvNeedsTranspose(shape: []const c_int) bool {
    if (shape.len != 3) return false;
    return shape[1] == 1 and shape[2] > 1;
}

/// One-shot so a mixed-converter library says which layout it found, once.
var depthwise_conv_layout_logged: bool = false;

/// In-place: swap a torch-layout depthwise conv weight into the layout the
/// forward's `mlx_conv1d` contracts against, recording the new array in `owned`
/// so `Transformer.deinit` frees it.
pub fn maybeTransposeDepthwiseConv(
    w: *mlx.mlx_array,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !void {
    if (w.ctx == null) return;
    if (!depthwiseConvNeedsTranspose(mlx.getShape(w.*))) return;
    const perm = [_]c_int{ 0, 2, 1 };
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w.*, &perm, 3, s));
    // A transpose is a VIEW; the conv reads raw row-major bytes, so materialize.
    var w_c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&w_c, w_t, false, s));
    _ = mlx.mlx_array_free(w_t);
    w.* = w_c;
    try owned.append(allocator, w_c);
    if (!depthwise_conv_layout_logged) {
        depthwise_conv_layout_logged = true;
        log.info("[conv] depthwise weights are torch-layout [C, 1, K] — transposed at load\n", .{});
    }
}

/// In-place: if `*sc` is null-ctx, we treat the matching `*w` as plain bf16.
/// Replace `*w` with its pre-transposed `[in, out]` form and record the new
/// array in `owned` so we can free it on Transformer.deinit.
pub fn maybeTransposeForBf16(
    w: *mlx.mlx_array,
    sc: mlx.mlx_array,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !void {
    if (sc.ctx != null) return; // quantized weight — leave as-is
    if (w.ctx == null) return; // absent weight (e.g. KV-shared layer) — nothing to transpose
    const transposed = try transposeBf16Weight(w.*, s);
    w.* = transposed;
    try owned.append(allocator, transposed);
}

/// One quantized-or-dense projection triple by full tensor base name. With
/// `dense_in` (the input dim) an affine-quantized weight is dequantized first.
pub fn weightTriple(weights: *const Weights, base: []const u8, dense_in: ?u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
    var nb: [160]u8 = undefined;
    const w = weights.get(std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch unreachable) orelse {
        log.err("MISSING WEIGHT: {s}.weight\n", .{base});
        return error.MissingWeight;
    };
    var sc = weights.get(std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    var bi = weights.get(std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    var wm = w;
    if (dense_in) |in_dim| try dequantizeSmall(&wm, &sc, &bi, in_dim, owned_bf16, allocator, s);
    try maybeTransposeForBf16(&wm, sc, owned_bf16, allocator, s);
    return .{ .w = wm, .sc = sc, .bi = bi };
}

pub const dequantizeSmall = weights_load.dequantizeSmall;

/// A grouped-norm weight `[hc*hidden]` viewed as `[hc, hidden]` (owned).
pub fn groupedNormView(weights: *const Weights, name: []const u8, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
    const shape = [_]c_int{ @intCast(hc), @intCast(hidden) };
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, raw, &shape, 2, s));
    try owned_bf16.append(allocator, view);
    return view;
}

/// A small table widened to f32 (and transposed for `x @ t`) once, at load.
pub fn f32Table(w: mlx.mlx_array, transpose: bool, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    if (transpose) try mlx.check(mlx.mlx_transpose(&t, w, s)) else _ = mlx.mlx_array_set(&t, w);
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, t, .float32, s));
    const c = try materializedOwnedCopy(s, f);
    _ = mlx.mlx_array_free(f);
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    try owned.append(allocator, c);
    return c;
}

/// Multiply a projection by a scalar: scales+biases of an affine weight, or
/// the dense weight itself. Replacement arrays are owned (and evaluated).
pub fn scaleTriple(w: *mlx.mlx_array, sc: *mlx.mlx_array, bi: *mlx.mlx_array, k: mlx.mlx_array, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    // The scalar must carry the TARGET's dtype: an f32 scalar promotes the
    // bf16 weight/scales, and a widened inject gate then widened the whole
    // residual stream to f32 (1/hc is exact in bf16 for hc <= 8).
    const Scaled = struct {
        fn f(a: mlx.mlx_array, k_raw: mlx.mlx_array, st: mlx.mlx_stream) !mlx.mlx_array {
            var kt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kt);
            try mlx.check(mlx.mlx_astype(&kt, k_raw, mlx.mlx_array_dtype(a), st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&out, a, kt, st));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.f;
    if (sc.ctx != null) {
        const s2 = try Scaled(sc.*, k, s);
        try owned_bf16.append(allocator, s2);
        sc.* = s2;
        if (bi.ctx != null) {
            const b2 = try Scaled(bi.*, k, s);
            try owned_bf16.append(allocator, b2);
            bi.* = b2;
        }
    } else {
        const w2 = try Scaled(w.*, k, s);
        try owned_bf16.append(allocator, w2);
        w.* = w2;
    }
}

/// Append every non-null mlx_array field of a plain weight struct.
pub fn appendStructArrays(vec: mlx.mlx_vector_array, w: anytype) void {
    inline for (comptime structFields(@TypeOf(w.*))) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        const arr = @field(w, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (comptime structFields(FullAttnWeights)) |field| {
        // Dense bf16 full-attn layers carry null-ctx scales/biases — skip those
        // so null arrays don't poison the eval batch. Mirrors the linear/mlp paths.
        if (comptime field.type != mlx.mlx_array) continue;
        const arr = @field(fa, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (comptime structFields(LinearAttnWeights)) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        const za_field = comptime std.mem.startsWith(u8, field.name, "z_") or std.mem.startsWith(u8, field.name, "a_");
        const skip_za = za_field and la.combined_proj;
        if (!skip_za) {
            const arr = @field(la, field.name);
            // Plain-bf16 layers (Unsloth Dynamic) carry null scales/biases — skip those.
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        }
    }
}

pub fn appendHybridMlpWeights(vec: mlx.mlx_vector_array, hw: *const HybridMlpWeights) void {
    // Plain-bf16 layers (Unsloth Dynamic) carry null-ctx scales/biases — skip those
    // so they don't pollute the eval batch. Mirrors `appendLinearAttnWeights`.
    switch (hw.*) {
        .moe => |*mw| {
            inline for (comptime structFields(MoeMlpWeights)) |field| {
                if (field.type == ?mlx.mlx_array) {
                    if (@field(mw, field.name)) |arr| {
                        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                } else if (field.type == mlx.mlx_array) {
                    const arr = @field(mw, field.name);
                    if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                }
            }
        },
        .dense => |*dw| {
            inline for (comptime structFields(DenseMlpWeights)) |field| {
                if (field.type == mlx.mlx_array) {
                    const arr = @field(dw, field.name);
                    if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                }
            }
        },
    }
}

// ── Utility functions ──

pub const detectQuantBits = @import("../../../sdk/quant/params.zig").detectQuantBits;

pub const lastDim = @import("../../../mlx/array.zig").lastDim;

/// Detect a quantized weight's (bits, group_size, mode) from its scales tensor
/// plus the model config:
/// - uint8 scales → fp8-family. In an nvfp4/mxfp4/mxfp8 model the config's
///   mode/bits/group_size apply (the fp8 scheme is uniform within a model).
///   Under an affine config this is the legacy "mxfp8 tensor inside an affine
///   checkpoint" case: bits is always 8, group size from the shape ratio.
/// - float scales → affine. In an affine model, bits via detectQuantBits
///   against the config group_size (mixed-precision models share one gs). For
///   affine OVERRIDES inside a non-affine model (e.g. nvfp4 QAT checkpoints
///   keep the shared MLP at affine 8-bit/gs64) the config group_size doesn't
///   apply: w_cols*32/s_cols only pins bits×gs, so solve exactly with the
///   caller's `in_dim` (activation inner dim); without a hint, assume mlx-lm's
///   override default group size 64.
/// Solve an affine weight's exact (bits, group_size) from packed-column
/// geometry when the input dimension is known:
///   bits = w_cols * 32 / in_dim,  group_size = in_dim / s_cols
/// Returns null unless both divide EXACTLY and land on values MLX's affine
/// packing supports — callers then fall back to config-based detection.
/// This is what lets a sidecar (e.g. an MTP head quantized 5-bit/gs-128 over
/// a 4-bit/gs-64 trunk) resolve per-weight instead of inheriting the trunk's
/// group size.
/// True when an embedding/lm_head TABLE is stored dense (float dtype) even
/// though the config declares global quantization. Mixed checkpoints quantize
/// per-tensor — hy_v3 2-bit ships a bf16 embed_tokens beside 2-bit experts —
/// so scales-presence must be decided by the WEIGHT's dtype, never the
/// config's bits (a packed uint32 table missing scales still crashes
/// honestly at the mandatory fetch).
pub fn floatDtypeTable(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float16 or dtype == .float32;
}

pub const splitPackedGateUp = @import("../../../mlx/array.zig").splitPackedGateUp;

pub fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// `{prefix}.{base}.{suffix}` with a RUNTIME base — the embedding table's name
/// is the checkpoint's (embed_tokens / word_embeddings / embed / embeddings)
/// and its three lookups must agree, which a comptime format can't express
/// without repeating the arch chain per suffix.
pub fn getNamedWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, base: []const u8, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}.{s}", .{ prefix, base, suffix }) catch unreachable;
    return weights.get(name);
}

pub fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

pub fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

/// `getLayerWeight` for `<base>.<rest>`, formatted into its own buffer.
/// `<prefix>.layers.<li>`: the name base of one trunk layer.
pub fn layerBase(buf: *[96]u8, prefix: []const u8, li: u32) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.layers.{d}", .{ prefix, li }) catch unreachable;
}

pub fn getWeightAt(weights: *const Weights, buf: *[256]u8, base: []const u8, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}", .{ base, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

pub fn getWeightAtOpt(weights: *const Weights, buf: *[256]u8, base: []const u8, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}", .{ base, suffix }) catch unreachable;
    return weights.get(name);
}

pub fn getLayerWeightAt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, base: []const u8, rest: []const u8) error{MissingWeight}!mlx.mlx_array {
    var nb: [128]u8 = undefined;
    return getLayerWeight(weights, buf, prefix, layer, std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, rest }) catch unreachable);
}

pub fn getLayerWeightAtOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, base: []const u8, rest: []const u8) ?mlx.mlx_array {
    var nb: [128]u8 = undefined;
    return getLayerWeightOpt(weights, buf, prefix, layer, std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, rest }) catch unreachable);
}

/// A quantized (or dense) projection `<base>.{weight,scales,biases}`.
pub fn loadQW(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, base: []const u8) error{MissingWeight}!QW {
    return .{
        .w = try getLayerWeightAt(weights, buf, prefix, layer, base, "weight"),
        .s = getLayerWeightAtOpt(weights, buf, prefix, layer, base, "scales") orelse mlx.mlx_array_new(),
        .b = getLayerWeightAtOpt(weights, buf, prefix, layer, base, "biases") orelse mlx.mlx_array_new(),
    };
}

pub fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// "<container>.<proj>.<leaf>" for an expert bank, built into `buf`.
fn expertLeaf(buf: []u8, container: []const u8, proj: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}.{s}", .{ container, proj, leaf }) catch unreachable;
}

/// Build a "<container>.<leaf>" layer-weight suffix into `buf`. Used where the
/// stacked-MoE-expert container is named differently across converters — hy_v3
/// experts are `mlp.experts.*` in ox-ox-style MLX builds and `mlp.switch_mlp.*`
/// in mlx-lm builds (mlx-community Hy3-oQ2*, pipenetwork Hy3-REAP*), same
/// [E, out, in] tensor either way. Uses its own buffer so it can compose with
/// getLayerWeight's separate key buffer in one expression.
pub fn moeExpertSuffix(buf: []u8, container: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ container, leaf }) catch unreachable;
}

/// Fetch a quantization scale/bias tensor, tolerant of dense bf16 models.
/// quant_bits == 0 ⇒ dense bf16 (no .scales/.biases exist anywhere) ⇒ return a
/// null-ctx array, which downstream code (qmatmulBits, gatherExpertMm, append
/// guards) reads as "this weight is plain bf16". quant_bits > 0 ⇒ a genuinely
/// quantized model, so fetch mandatorily — a missing scale stays a clear
/// MISSING WEIGHT error rather than silently degrading to a dense path.
pub fn getLayerScaleOrEmpty(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) error{MissingWeight}!mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return try getLayerWeight(weights, buf, prefix, layer, suffix);
}

/// Fetch a `.biases` tensor with mode-aware mandatoriness:
/// - dense bf16 (quant_bits == 0): null-ctx placeholder, like the scales.
/// - affine mode: mandatory — a missing bias is a clear MISSING WEIGHT error.
/// - bias-less modes (nvfp4/mxfp4/mxfp8, issue #24): OPTIONAL. The fp8
///   tensors ship no biases, but mixed QAT checkpoints override some layers
///   to affine (e.g. shared MLP at 8-bit/gs64) and those overrides DO carry
///   biases that the affine matmul needs.
pub fn getLayerBias(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, config: *const ModelConfig) error{MissingWeight}!mlx.mlx_array {
    // A pack with no `quantization` block (Sushi's) still stores affine trunk linears beside dense ones.
    if (config.quant_bits == 0) return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
    if (config.quant_mode.hasBiases()) return try getLayerWeight(weights, buf, prefix, layer, suffix);
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
}

/// Optional-typed variant for fields stored as `?mlx_array` (e.g. PLE projections).
/// Dense bf16 ⇒ `some(null-ctx)` so call sites that unwrap with `.?` still get a
/// valid (empty) array that qmatmul reads as bf16. Quantized ⇒ optional fetch.
pub fn getLayerScaleOrEmptyOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) ?mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix);
}

pub const scalarOf = @import("../../../mlx/array.zig").scalarOf;

pub const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;

pub fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}

// ── tests ──
