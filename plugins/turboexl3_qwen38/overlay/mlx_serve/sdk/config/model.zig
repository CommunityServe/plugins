//! This plugin's own, in place of mlx-serve's `src/sdk/config/model.zig` (a cut edge of scripts/sync-mlx-serve.sh): the
//! model config with the fields the K packs' code reads; its methods are mlx-serve's at 1b3e624b.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const abi = @import("mlx_serve").abi.plugin;
const exl3 = @import("../quant/exl3/format.zig");
const distributed = @import("../../engines/mlx/cluster/distributed.zig");
pub const HiddenAct = @import("../blocks/mlp/activations.zig").HiddenAct;
pub const QuantMode = @import("../quant/params.zig").QuantMode;
pub const Qwen4NormConvention = enum { delta, folded };
pub const ModelConfig = struct {
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "language_model.model",
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    rms_norm_eps: f32 = 1e-6,
    hadamard_block: u32 = 0,
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    has_sliding_window: bool = true,
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,
    query_pre_attn_scalar: u32 = 256,
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,
    swiglu_limit: f32 = 0.0,
    swiglu_clamp: bool = false,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    first_k_dense_replace: u32 = 0,
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    ssm_state_f32: bool = false,
    kda_sigmoid_out_gate: bool = false,
    mla_kv_lora_rank: u32 = 0,
    mla_qk_nope_head_dim: u32 = 0,
    mla_qk_rope_head_dim: u32 = 0,
    full_attention_interval: u32 = 0,
    linear_attn_tail_from: u32 = 0,
    dist_span: ?distributed.LayerSpan = null,
    dist_shard: ?distributed.Shard = null,
    partial_rotary_factor: f32 = 1.0,
    attn_output_gate: bool = false,
    hc_count: u32 = 0,
    hc_lowrank: u32 = 0,
    ple_layer_idx: i32 = -1, // 0-based; the config lists 1-based ids
    ple_embed_dim: u32 = 0,
    ple_conv_kernel: u32 = 4,
    ngram_size: u32 = 3,
    heads_per_ngram: u32 = 8,
    ngram_vocab_base: u64 = 20_000_000,
    ngram_vocab_divisor: u32 = 128,
    ngram_seed: u64 = 1234,
    split_ngram_parts: u32 = 128,
    qwen4_norm_convention: ?Qwen4NormConvention = null,
    indexer_n_heads: u32 = 0, // 0 = dense attention
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    exl3: ?exl3.Spec = null,
    trunk_format: bool = false,
    ngram_table_file_buf: [96]u8 = @splat(0),
    ngram_table_file_len: u8 = 0,
    modelopt: bool = false,
    ngram_eos: u32 = 0,
    ngram_table_path: ?[]const u8 = null,
    pack_dir: ?[]const u8 = null,
    rope_yarn: bool = false,
    bos_token_id: ?u32 = null,
    max_position_embeddings: u32 = 0,
    drafter_override: ?[]const u8 = null,
    plugin: ?*plugin_config.Binding = null,
    legacy: struct { caps: abi.Caps = .{}, thinking: abi.ThinkingDefault = .unset } = .{},
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,
    gen_enable_thinking: ?bool = null,
    image_token_id: u32 = 0, // 0 = no image token
    qwen_vision: bool = false,
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,
    has_final_norm: bool = true, // false for LFM2 (no model.norm.weight)
    /// The PLE's layer among this process's layers; null when another pipeline rank holds it.
    /// `ple_layer_idx` stays the checkpoint's absolute index (it names the n-gram shards).
    pub fn pleLocalLayer(self: *const ModelConfig) ?u32 {
        if (self.ple_layer_idx < 0) return null;
        const abs: u32 = @intCast(self.ple_layer_idx);
        const st: u32 = if (self.dist_span) |sp| sp.start else 0;
        if (abs < st or abs - st >= self.num_hidden_layers) return null;
        return abs - st;
    }
    pub fn isLinearLayer(self: ModelConfig, layer_idx: u32) bool {
        if (self.full_attention_interval == 0) return false;
        if (self.linear_attn_tail_from != 0 and layer_idx >= self.linear_attn_tail_from) return false;
        return ((layer_idx + 1) % self.full_attention_interval) != 0;
    }
    /// Activation dtype: f16 for a Prism Hadamard pack (its own contract: f16
    /// activations over f16 scales), bf16 everywhere else.
    pub fn actDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.hadamard_block > 0) .float16 else .bfloat16;
    }
    /// GatedDeltaNet recurrent state dtype, which the decay gate is computed in too:
    /// f32 where the reference runtime keeps it (Hadamard packs, GLM-5, qwen4_exp),
    /// since a gate near one rounds to 1.0 in bf16 and that head stops forgetting.
    pub fn ssmStateDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.hadamard_block > 0 or self.isGlm5() or self.ssm_state_f32) .float32 else .bfloat16;
    }
    pub fn rowExactArch(self: *const ModelConfig) bool {
        if (self.caps().exact_drafted) return true;
        return std.mem.startsWith(u8, self.model_type, "qwen3_5") and !self.isMoe() and self.hadamard_block == 0;
    }
    pub fn isGlm5(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
    }
    pub fn isDsv41(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "deepseek_v41");
    }
    pub fn isMimo(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "mimo_v2");
    }
    pub fn isMoe(self: *const ModelConfig) bool {
        return self.caps().moe or self.num_experts > 0;
    }
    /// The model's caps: its plugin's when served, else what the legacy parse declares.
    pub fn caps(self: *const ModelConfig) abi.Caps {
        return if (self.plugin) |b| b.parsed.caps else self.legacy.caps;
    }
    /// True when the full-attention layers are Multi-head Latent Attention
    /// (compressed KV latent + low-rank Q), not plain GQA projections.
    pub fn isMla(self: *const ModelConfig) bool {
        return self.mla_kv_lora_rank > 0;
    }
    /// MLA query/key head dim = the non-positional part plus the rope part.
    /// This — not head_dim — is what the attention scale and the cached K's
    /// last dim are measured in.
    pub fn mlaQkHeadDim(self: *const ModelConfig) u32 {
        return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim;
    }
    /// Epsilon of the per-head q/k RMS norm in the linear-attention prework. GLM's reference
    /// normalizes with fla's l2norm (1e-6 on the SUM of squares), which is 1e-6/dk on the mean;
    /// its post-conv keys are small enough that 1e-6 on the mean shrank them ~2%.
    pub fn gdnQkNormEps(self: *const ModelConfig) f32 {
        return if (self.isGlm5()) 1e-6 / @as(f32, @floatFromInt(self.linear_key_head_dim)) else 1e-6;
    }
    /// Qwen3.8-Flash-Next (`qwen4_exp`): the qwen3_5 GDN + MoE trunk wrapped
    /// in hyper-connection residual streams, with the n-gram PLE and QSA.
    pub fn isQwen4(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "qwen4_exp");
    }
    /// The long-context blast-radius predicate: every long-context mechanism (KV
    /// reservation, pad-waste cap, checkpoint thinning, admission terms, chunk bar) was
    /// measured on qwen4_exp only, so they are opt-in by arch. Never hand-roll it at a site.
    pub fn longCtxGated(self: *const ModelConfig) bool {
        return self.caps().long_ctx;
    }
    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers).
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4");
    }
    /// The head width the PREFILL SCORE tensor is actually built at. Normally
    /// `head_dim`, but an arch can score at a different width than it stores
    /// values at (an MLA q.k can contract over nope+rope widths while
    /// `head_dim` stays the value width) — reading `head_dim` there puts such
    /// an arch under the `<= 128` "fused SDPA covers it" early-out, so the
    /// score budget that exists for exactly this materializing path never
    /// applies. A new arch scoring wider than it stores adds its arm here.
    /// 0 = no score tensor at all: MiMo's prefill attention is always a fused
    /// kernel (`fusedSinkAttnPrefill`), whatever its width.
    pub fn prefillScoreHeadDim(self: *const ModelConfig) u32 {
        if (self.isMimo()) return 0;
        // GLM-5-Next: the sparse rows never form a score matrix (indexed kernel), and the
        // dense ones end at the indexer budget (~2k), whatever the chunk.
        if (self.isGlm5()) return 0;
        // DeepSeek-V4.1: a query scores its window and its index picks, never the prompt.
        if (self.isDsv41()) return 0;
        if (self.isMla()) return self.mlaQkHeadDim();
        return self.head_dim;
    }
    /// Free the allocator-owned fields (`ngram_table_path`, allocPrint'd by
    /// `parseConfig`, and `drafter_override`); everything else is plain data or a borrowed slice. Every
    /// `destroy` of a parsed config pairs with this, or a qwen4 load leaks the
    /// path. Idempotent.
    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        if (self.ngram_table_path) |p| allocator.free(p);
        self.ngram_table_path = null;
        if (self.pack_dir) |p| allocator.free(p);
        self.pack_dir = null;
        if (self.drafter_override) |p| allocator.free(p);
        self.drafter_override = null;
        if (self.plugin) |b| b.destroy(allocator);
        self.plugin = null;
    }
};
/// True when the layer loop installed the PLE on exactly the layer the config names. A negative
/// index asks for no PLE (the MTP head's own layer) and is satisfied by a loop that installed none.
pub fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}
// Optional config nulls have the same fallback and precedence as omitted keys.
pub const Weights = @import("../weights/resident.zig").Weights;
// ── Tests ──
// ── qwen4_exp YaRN context extension (262144 → 1048576) ──────────────────
//
// Both documents below are the SHIPPED checkpoint's text config (Qwen3.8-Flash-
// Next, `model_type: qwen4_exp`) — one as it ships (plain rope, 262144) and one
// with the YaRN block vLLM's `--hf-overrides` recipe writes. Keeping them as
// literals means the parser is tested against the real file shape, braces and
// all, rather than a synthesized one.
