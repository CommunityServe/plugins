//! The trunk's tensors as this plugin reads them (the host loads none: `keep_key` refuses every key): the copied resident
//! loader (`sdk/weights/load.zig`) over the pack's nonexpert shards, then the moves the engine makes for a GLM-5.3-Flash
//! pack in turboderp's names (`sdk/quant/exl3/sushi_pack.zig`, the engine's `sushi_adapt.adapt` for a streaming pack):
//! the hyper-connection tables under `attn_hc.` / `ffn_hc.`, the MTP layer (`layers.45`) under `mtp.0` and
//! `mtp.0.block`, and `kv_b_proj` split into `embed_q` / `unembed_out`. One rule differs from the engine's load: an EXL3
//! linear's `suh` and `svh` stay f16 (patches/turboexl3-sdk.patch, `narrowsLoadedF16`), where the engine narrows
//! every 1-D
//! f16 table to bf16. The trunk's plain f16 matrices are cast to bf16 by its loader (`Loader.lin`,
//! patches/turboexl3-model.patch).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const load = @import("mlx_serve/sdk/weights/load.zig");
const resident = @import("mlx_serve/sdk/weights/resident.zig");
const sushi_pack = @import("mlx_serve/sdk/quant/exl3/sushi_pack.zig");

pub const Weights = resident.Weights;

/// The text model's prefix in turboderp's names.
pub const prefix = "model.language_model";

/// The pack's attention geometry the `kv_b_proj` split reads.
pub const KvB = struct { heads: u32, head_dim: u32, latent: u32 };

/// The pack's trunk tensors in the layout the copied glm5_next loader reads. `trunk_layers`: the decoder layers (the MTP
/// layer is the next one). The vision tower is not read.
pub fn read(io: std.Io, a: std.mem.Allocator, dir: []const u8, trunk_layers: u32, kvb: KvB) !Weights {
    var w = try load.loadWeightsOpt(io, a, dir, .{ .vision = false });
    errdefer w.deinit();
    try sushi_pack.renameGlmMtp(&w, prefix, trunk_layers);
    try sushi_pack.renameHyperConnections(&w);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try sushi_pack.splitKvB(&w, .{ .heads = kvb.heads, .nope = kvb.head_dim, .v = kvb.head_dim, .latent = kvb.latent }, s);
    return w;
}

// ── tests ──

const testing = std.testing;

test "turboexl3_glm53f weights: an EXL3 linear's suh and svh stay f16; the other 1-D f16 tables narrow as the engine's" {
    for ([_][]const u8{ "model.language_model.layers.0.self_attn.qkv_proj.suh", "lm_head.svh", "model.language_model.layers.45.eh_proj.suh" }) |k|
        try testing.expect(!load.narrowsLoadedF16(k, 1, .float16));
    try testing.expect(load.narrowsLoadedF16("model.language_model.layers.3.self_attn.indexer.k_norm.weight", 1, .float16));
    try testing.expect(load.narrowsLoadedF16("x.scales", 2, .float16));
    try testing.expect(!load.narrowsLoadedF16("model.language_model.layers.0.self_attn.b_proj.weight", 2, .float16));
    try testing.expect(!load.narrowsLoadedF16("model.language_model.layers.0.input_layernorm.weight", 1, .bfloat16));
}
