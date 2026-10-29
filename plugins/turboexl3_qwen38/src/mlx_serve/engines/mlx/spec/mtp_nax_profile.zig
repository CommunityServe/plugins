//! copied_from: src/engines/mlx/spec/mtp_nax_profile.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/spec/mtp_nax_profile.
//! Owns MtpNaxProfileInputs, mtpNaxBf16EmbeddingMatchesFrom, mtpNaxBf16EmbeddingMatches, mtpNaxCalibratedModelFrom, mtpNaxProfileEnabledFrom, mtpNaxAffineProjectionMatchesQuant, and 20 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const mixedNaxShapeEnabled = @import("../../../sdk/blocks/linear/verify_qmm.zig").mixedNaxShapeEnabled;
const verifyQmmEnabled = @import("../../../sdk/blocks/linear/verify_qmm.zig").verifyQmmEnabled;
const naxMinMForBits = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").naxMinMForBits;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verifyQmmNaxEnabledForMFrom = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxEnabledForMFrom;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const affineParamsFromGeometry = @import("../../../sdk/quant/params.zig").affineParamsFromGeometry;
const MoeLayerWeights = @import("../layer_weights.zig").MoeLayerWeights;
const Weights = @import("../../../sdk/weights/resident.zig").Weights;
const testing = @import("../kv/cache.zig").testing;
const Transformer = @import("../transformer.zig").Transformer;

/// Pure inputs for the calibrated M5 MTP cost profile. The profile was
/// measured on a dense trunk whose actual lm_head takes the NAX lane at both
/// verify widths produced by depth 8 (M=8 and M=9); every field below mirrors
/// a production dispatch precondition rather than trusting model-wide config.
const MtpNaxProfileInputs = struct {
    dense_model: bool,
    calibrated_model: bool,
    profiled_affine_trunk: bool,
    model_quant: QuantParams,
    weight_present: bool,
    packed_weight: bool,
    scales_present: bool,
    biases_present: bool,
    quant: QuantParams,
    K: c_int,
    N: c_int,
    packed_k: c_int,
    verify_on: bool,
    available: bool,
    min_m: c_int,
};

pub fn mtpNaxCalibratedModelFrom(config: *const ModelConfig, lm_head_n: c_int) bool {
    return std.mem.eql(u8, config.model_type, "qwen3_5_moe") and
        !config.isMoe() and
        config.hidden_size == 5120 and
        config.intermediate_size == 17408 and
        config.num_hidden_layers == 64 and
        config.vocab_size == 248320 and
        lm_head_n == 248320 and
        config.num_attention_heads == 24 and
        config.num_key_value_heads == 4 and
        config.head_dim == 256 and
        config.full_attention_interval == 4 and
        config.linear_num_key_heads == 16 and
        config.linear_num_value_heads == 48 and
        config.linear_key_head_dim == 128 and
        config.linear_value_head_dim == 128 and
        config.attn_output_gate and
        !config.tie_word_embeddings;
}

fn mtpNaxAffineProjectionMatchesQuant(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    expected: QuantParams,
) bool {
    if (w.ctx == null or sc.ctx == null or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or
        mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    if (in_dim == 0 or out_dim == 0 or
        in_dim > std.math.maxInt(c_int) or out_dim > std.math.maxInt(c_int)) return false;
    const out: c_int = @intCast(out_dim);
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const b_shape = mlx.getShape(bi);
    if (w_shape.len != 2 or s_shape.len != 2 or b_shape.len != 2) return false;
    if (w_shape[0] != out or s_shape[0] != out or b_shape[0] != out) return false;
    if (s_shape[1] != b_shape[1]) return false;
    const qp = affineParamsFromGeometry(w, sc, in_dim) orelse return false;
    if (qp.bits != expected.bits or
        qp.group_size != expected.group_size or
        qp.mode != expected.mode) return false;

    // Every material projection on the calibrated surface must take NAX at
    // both depth-8 verify widths. Tiny per-head A/B gates and the measured
    // narrow q5/q6 regressions remain on stock by design.
    if (out_dim >= 512 and mixedNaxShapeEnabled(expected.bits, expected.group_size, @intCast(out_dim), 8)) {
        const K: c_int = @intCast(in_dim);
        const N: c_int = @intCast(out_dim);
        if (!verifyQmmNaxEnabledForMFrom(8, K, N, true, true, 8) or
            !verifyQmmNaxEnabledForMFrom(9, K, N, true, true, 8)) return false;
    }
    return true;
}

fn mtpNaxAffineProjectionMatches(
    config: *const ModelConfig,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = config.quant_bits,
        .group_size = config.quant_group_size,
        .mode = config.quant_mode,
    });
}

pub const OQE_MLP_DOWN_BITS = "5555555555555555555555555556555644444444444444444444444444444444";
pub const OQE_LINEAR_Z_BITS = "5554444444545554555454444444444444444444444444444444444444444444";
pub const OQE_LINEAR_AB_BITS = "5554544444545554555455545554555444444444444444444444444444444444";
pub const OQE_LINEAR_OUT_BITS = "5554555455545554555455545554555455545554555455545554555455545554";
pub const OQE_FULL_QK_BITS = "4444444444444444444444444445444544444444444444444444444444444444";
pub const OQE_FULL_V_BITS = "4444444444444444444444444446444444444444444444444444444444444444";
pub const OQE_FULL_O_BITS = "4444444444454445444544444444444444444444444444444444444444444444";

pub fn oqeLayerBits(pattern: *const [64:0]u8, layer: usize) u32 {
    std.debug.assert(layer < 64);
    return pattern[layer] - '0';
}

fn mtpNaxOqeProjectionMatches(
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    in_dim: u32,
    out_dim: u32,
    bits: u32,
) bool {
    return mtpNaxAffineProjectionMatchesQuant(w, sc, bi, in_dim, out_dim, .{
        .bits = bits,
        .group_size = 64,
        .mode = .affine,
    });
}

/// One projection's packed triple as loaded.
const Proj = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array };

/// What a trunk fingerprint reads of each layer: the full attention's q, k, v, o or the linear attention's
/// qkv, z, a, b, out (`attn[4]` unused on a full layer), then the dense MLP's gate, up, down.
const LayerProjs = struct { linear: bool, attn: [5]Proj, mlp: [3]Proj };

/// The trunk's layers as the legacy load built them, or as a served pack's tensors name them (`<prefix>.layers.<i>`).
pub const TrunkSource = union(enum) {
    layers: ?[]MoeLayerWeights,
    weights: struct { w: *const Weights, prefix: []const u8 },

    fn len(src: TrunkSource, config: *const ModelConfig) ?usize {
        return switch (src) {
            .layers => |l| if (l) |ls| ls.len else null,
            .weights => @intCast(config.num_hidden_layers),
        };
    }

    /// Layer `i`'s projections; null where it is not the separate-projection dense layout the fingerprints read
    /// (fused qkvz, a MoE MLP, a missing tensor).
    pub fn at(src: TrunkSource, config: *const ModelConfig, i: usize) ?LayerProjs {
        const linear = config.isLinearLayer(@intCast(i));
        switch (src) {
            .layers => |l| {
                const layer = &l.?[i];
                if (layer.is_linear != linear) return null;
                var p: LayerProjs = .{ .linear = linear, .attn = undefined, .mlp = undefined };
                switch (layer.attn) {
                    .full => |*a| p.attn = .{ .{ .w = a.q_w, .s = a.q_s, .b = a.q_b }, .{ .w = a.k_w, .s = a.k_s, .b = a.k_b }, .{ .w = a.v_w, .s = a.v_s, .b = a.v_b }, .{ .w = a.o_w, .s = a.o_s, .b = a.o_b }, undefined },
                    .linear => |*a| {
                        if (a.combined_proj) return null;
                        p.attn = .{ .{ .w = a.qkv_w, .s = a.qkv_s, .b = a.qkv_b }, .{ .w = a.z_w, .s = a.z_s, .b = a.z_b }, .{ .w = a.a_w, .s = a.a_s, .b = a.a_b }, .{ .w = a.b_w, .s = a.b_s, .b = a.b_b }, .{ .w = a.out_w, .s = a.out_s, .b = a.out_b } };
                    },
                }
                switch (layer.mlp) {
                    .dense => |*m| p.mlp = .{ .{ .w = m.gate_w, .s = m.gate_s, .b = m.gate_b }, .{ .w = m.up_w, .s = m.up_s, .b = m.up_b }, .{ .w = m.down_w, .s = m.down_s, .b = m.down_b } },
                    .moe => return null,
                }
                return p;
            },
            .weights => |src_w| {
                const full = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj" };
                const lin = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.out_proj" };
                const mlp = [_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
                var p: LayerProjs = .{ .linear = linear, .attn = undefined, .mlp = undefined };
                const names: []const []const u8 = if (linear) &lin else &full;
                for (names, 0..) |n, k| p.attn[k] = projNamed(src_w.w, src_w.prefix, i, n) orelse return null;
                for (mlp, 0..) |n, k| p.mlp[k] = projNamed(src_w.w, src_w.prefix, i, n) orelse return null;
                return p;
            },
        }
    }
};

fn projNamed(w: *const Weights, prefix: []const u8, i: usize, name: []const u8) ?Proj {
    var buf: [256]u8 = undefined;
    var out: Proj = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, [_]*mlx.mlx_array{ &out.w, &out.s, &out.b }) |leaf, dst| {
        const key = std.fmt.bufPrint(&buf, "{s}.layers.{d}.{s}.{s}", .{ prefix, i, name, leaf }) catch return null;
        dst.* = w.get(key) orelse if (std.mem.eql(u8, leaf, "weight")) return null else mlx.mlx_array{ .ctx = null };
    }
    return out;
}

/// Projection roles, in `LayerProjs` order per kind.
const Role = enum { q, k, v, o, qkv, z, a, b, out, gate, up, down };

pub fn trunkMatches(config: *const ModelConfig, src: TrunkSource, comptime oqe: bool) bool {
    if (config.isMoe() or config.full_attention_interval == 0) return false;
    if (config.hidden_size == 0 or config.intermediate_size == 0) return false;
    if (oqe and config.num_hidden_layers != 64) return false;
    const n = src.len(config) orelse return false;
    if (n != @as(usize, @intCast(config.num_hidden_layers))) return false;

    const full_out_wide = @as(u64, config.num_attention_heads) * config.head_dim;
    const full_q_out_wide = full_out_wide * 2;
    const kv_out_wide = @as(u64, config.num_key_value_heads) * config.head_dim;
    const linear_key_out_wide = @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim;
    const linear_out_wide = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const linear_qkv_out_wide = 2 * linear_key_out_wide + linear_out_wide;
    if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
        full_q_out_wide > std.math.maxInt(u32) or
        kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32) or
        (!oqe and (linear_key_out_wide == 0 or linear_key_out_wide > std.math.maxInt(u32))) or
        linear_out_wide == 0 or linear_out_wide > std.math.maxInt(u32) or
        linear_qkv_out_wide > std.math.maxInt(u32)) return false;
    const h = config.hidden_size;
    const full_out: u32 = @intCast(full_out_wide);
    const linear_out: u32 = @intCast(linear_out_wide);
    const heads_v = config.linear_num_value_heads;

    for (0..n) |i| {
        const p = src.at(config, i) orelse return false;
        const roles: []const Role = if (p.linear) &.{ .qkv, .z, .a, .b, .out } else &.{ .q, .k, .v, .o };
        for (roles, 0..) |role, k| {
            const dims: [2]u32 = switch (role) {
                .q => .{ h, @intCast(full_q_out_wide) },
                .k, .v => .{ h, @intCast(kv_out_wide) },
                .o => .{ full_out, h },
                .qkv => .{ h, @intCast(linear_qkv_out_wide) },
                .z => .{ h, linear_out },
                .a, .b => .{ h, heads_v },
                .out => .{ linear_out, h },
                else => unreachable,
            };
            if (!roleMatches(config, p.attn[k], dims, role, i, oqe)) return false;
        }
        const inter = config.intermediate_size;
        for ([_]Role{ .gate, .up, .down }, [_][2]u32{ .{ h, inter }, .{ h, inter }, .{ inter, h } }, 0..) |role, dims, k|
            if (!roleMatches(config, p.mlp[k], dims, role, i, oqe)) return false;
    }
    return true;
}

fn roleMatches(config: *const ModelConfig, p: Proj, dims: [2]u32, role: Role, i: usize, comptime oqe: bool) bool {
    if (!oqe) return mtpNaxAffineProjectionMatches(config, p.w, p.s, p.b, dims[0], dims[1]);
    const bits: u32 = switch (role) {
        .q, .k => oqeLayerBits(OQE_FULL_QK_BITS, i),
        .v => oqeLayerBits(OQE_FULL_V_BITS, i),
        .o => oqeLayerBits(OQE_FULL_O_BITS, i),
        .qkv, .gate, .up => 4,
        .z => oqeLayerBits(OQE_LINEAR_Z_BITS, i),
        .a, .b => oqeLayerBits(OQE_LINEAR_AB_BITS, i),
        .out => oqeLayerBits(OQE_LINEAR_OUT_BITS, i),
        .down => oqeLayerBits(OQE_MLP_DOWN_BITS, i),
    };
    return mtpNaxOqeProjectionMatches(p.w, p.s, p.b, dims[0], dims[1], bits);
}

/// Exact resident-tensor fingerprint for Jundot's Qwen3.6-27B-oQ4e trunk.
/// The checkpoint is globally affine-4/gs64 with a fixed layer/role map of
/// affine q5/q6 overrides. Matching the full map avoids granting cap 8 to
/// unrelated "mixed 4-bit" checkpoints with a different round-cost surface.
pub fn mtpNaxOqeAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    return trunkMatches(config, .{ .layers = maybe_layers }, true);
}

/// The depth-8 controller constants describe the measured homogeneous affine
/// Qwen3.6-27B trunk, not merely its architecture. Same-shape mixed checkpoints
/// (notably Unsloth Dynamic) may store individual projections as BF16 or mxfp8
/// while retaining a model-wide affine-4 config. Inspect every resident linear
/// used by the dense trunk so those checkpoints stay on the conservative cap 6.
pub fn mtpNaxUniformAffineTrunkFrom(config: *const ModelConfig, maybe_layers: ?[]MoeLayerWeights) bool {
    return trunkMatches(config, .{ .layers = maybe_layers }, false);
}

/// `Transformer.mtp_uniform_affine_trunk` and `mtp_oqe_affine_trunk` for a trunk whose head is `lm_head_w`.
pub const TrunkFlags = struct { uniform: bool, oqe: bool };

// ── tests ──
