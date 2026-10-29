//! copied_from: src/engines/mlx/load/sushi_adapt.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Sushi packs at load: the tensor moves `sdk/quant/exl3/sushi_pack.zig` provides, applied per the model's config
//! (GLM-5-Next and MiMo layouts).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const Weights = @import("../../../sdk/weights/resident.zig").Weights;
const ModelConfig = model_mod.ModelConfig;
const adaptMimoHeads = @import("../../../sdk/quant/exl3/sushi_pack.zig").adaptMimoHeads;
const decodeDenseMlp = @import("../../../sdk/quant/exl3/sushi_pack.zig").decodeDenseMlp;
const renameHyperConnections = @import("../../../sdk/quant/exl3/sushi_pack.zig").renameHyperConnections;
const splitFusedQkv = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitFusedQkv;
const splitKvB = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitKvB;

/// Also serves NVIDIA's ModelOpt GLM pack and GLM's streaming packs (`expert_bank`), which ship the same source layout.
pub fn adapt(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.exl3 == null and !config.modelopt and !(config.expert_bank and config.isGlm5())) return;
    if (config.isMimo() and config.exl3 != null) {
        try dequantFp8Trunk(config, weights, s);
    }
    if (config.isGlm5() and config.expert_bank) {
        // A streaming pack: the NextN layer under the release's names, its experts in their own NVFP4 bank.
        try @import("../../../sdk/quant/exl3/sushi_pack.zig").renameGlmMtp(weights, "model.language_model", config.num_hidden_layers);
        if (config.pack_dir) |dir| try @import("../../../sdk/quant/nvfp4/stream_bank.zig").load(weights, dir, "mtp-expert-manifest-nvfp4.json", "model.language_model.mtp.0.block");
    }
    if (config.isGlm5()) {
        try renameHyperConnections(weights);
        try splitKvB(weights, .{
            .heads = config.num_attention_heads,
            .nope = config.dsa_head_dim,
            .v = config.dsa_head_dim,
            .latent = config.dsa_kv_lora_rank,
        }, s);
    }
}

/// MiMo's trunk is block-FP8 as the source stores it; this engine reads dense projections, so
/// the fused rank-local `qkv_proj` becomes q/k/v and the layer-0 MLP dense, both bf16.
fn dequantFp8Trunk(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    var buf: [160]u8 = undefined;
    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const base = try std.fmt.bufPrint(&buf, "{s}.layers.{d}", .{ config.weight_prefix, li });
        try splitFusedQkv(weights, base, qkvRows(config, li), s);
        if (li < config.first_k_dense_replace) try decodeDenseMlp(weights, base, s);
    }
    // The MTP heads are stored the same way and attend in a sliding layer's geometry.
    try adaptMimoHeads(weights, qkvRows(config, config.firstSlidingLayer() orelse return), s);
}

fn qkvRows(config: *const ModelConfig, li: u32) [3]u64 {
    return .{
        @as(u64, config.layerNumHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerVHeadDim(li),
    };
}

// ── tests ──

const testing = std.testing;
