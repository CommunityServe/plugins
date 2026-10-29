//! copied_from: src/sdk/config/model.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const fp8_grid = @import("../quant/fp8_block128_gemv.zig");
const env = @import("../../env.zig");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const abi = @import("mlx_serve").abi.plugin;
const mlx_gguf = @import("../quant/gguf/claim.zig");
const exl3 = @import("../quant/exl3/format.zig");
const exl3_pack_keys = @import("../quant/exl3/pack_keys.zig");
const sushi_pack = @import("../../engines/mlx/load/sushi_adapt.zig");
const modelopt_pack = @import("../quant/nvfp4/modelopt_pack.zig");
const model_discovery = @import("../../host/registry/discovery.zig");
const tokenizer_mod = @import("../tokenizer/tokenizer.zig");
const transcription = @import("../../host/api/audio/transcription.zig");
const qwen4_exp = @import("../../plugins/qwen4_exp/qwen4_exp.zig");
const distributed = @import("../../engines/mlx/cluster/distributed.zig");
const qwen4_ple = @import("../../plugins/qwen4_exp/ple_table.zig");
const kv_quant_mod = @import("../blocks/attention/kv_quant.zig");
const mtp_acceptance_mod = @import("../spec/mtp_acceptance.zig");
const weights_load = @import("../weights/load.zig");

pub const HiddenAct = @import("../blocks/mlp/activations.zig").HiddenAct;

/// Upper bound on a vision tower's per-layer type table (muse ships 50).
pub const MAX_VISION_LAYERS = 64;

/// `MuseGlimmerImageProcessor.max_image_tokens` — MERGED tokens, not pixels.
pub const MUSE_MAX_IMAGE_TOKENS = 4096;

/// Default fraction of RADIO video tokens removed by EVS.
pub const RADIO_VIDEO_PRUNING_RATE_DEFAULT: f64 = 0.7;

pub const QuantMode = @import("../quant/params.zig").QuantMode;

pub const LayerBlockType = enum { attention, gated_conv, mamba2, mlp, moe };

/// Sentence-transformers pooling operation for embedding requests (issue
/// #116): masked mean over real positions, the CLS token (position 0), or the
/// last real (non-padding) token. Every mode is followed by L2 normalization.
pub const PoolingMode = enum {
    mean,
    cls,
    last_token,

    pub fn fromString(s: []const u8) ?PoolingMode {
        if (std.mem.eql(u8, s, "mean")) return .mean;
        if (std.mem.eql(u8, s, "cls")) return .cls;
        if (std.mem.eql(u8, s, "last_token")) return .last_token;
        return null;
    }
};

/// Parse a sentence-transformers `1_Pooling/config.json`. Returns the pooling
/// mode when the file declares one we implement, null when the content isn't a
/// pooling config at all (malformed JSON, unrelated object — best-effort, like
/// generation_config.json), and `error.UnsupportedPoolingMode` when the file
/// DOES declare pooling but only modes we don't implement (weighted-mean,
/// max): serving those checkpoints mean-pooled would be silent corruption.
pub fn parsePoolingSidecar(content: []const u8) !?PoolingMode {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{}) catch return null;
    if (parsed != .object) return null;
    const obj = parsed.object;

    const getBool = struct {
        fn get(o: std.json.ObjectMap, key: []const u8) bool {
            if (o.get(key)) |v| {
                if (v == .bool) return v.bool;
            }
            return false;
        }
    }.get;
    // ST configs set exactly one mode true; check ours most-specific first.
    if (getBool(obj, "pooling_mode_lasttoken")) return .last_token;
    if (getBool(obj, "pooling_mode_cls_token")) return .cls;
    if (getBool(obj, "pooling_mode_mean_tokens")) return .mean;
    // ST 6 names the mode in one string instead of a boolean per mode.
    if (obj.get("pooling_mode")) |v| if (v == .string) {
        if (std.mem.eql(u8, v.string, "mean")) return .mean;
        if (std.mem.eql(u8, v.string, "cls")) return .cls;
        if (std.mem.eql(u8, v.string, "lasttoken")) return .last_token;
        return error.UnsupportedPoolingMode;
    };
    // Declares pooling, but none we support → refuse rather than mean-pool.
    var it = obj.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.UnsupportedPoolingMode;
    }
    return null;
}

/// Known-family pooling fallback for checkpoints that ship neither an explicit
/// `pooling_mode` nor the ST sidecar (the mlx-community conversions strip it).
/// Gated on the arch so a directory name can never flip an unrelated model:
/// qwen3* named *embedding* → last-token (Qwen3-Embedding's contract), BERT
/// bge-/mxbai-embed → CLS (their model cards' contract). Everything else null
/// → the mean default.
pub fn poolingFromDirName(dir_basename: []const u8, model_type: []const u8) ?PoolingMode {
    var lower_buf: [256]u8 = undefined;
    if (dir_basename.len > lower_buf.len) return null;
    const lower = std.ascii.lowerString(&lower_buf, dir_basename);
    if (std.mem.startsWith(u8, model_type, "qwen3")) {
        if (std.mem.indexOf(u8, lower, "embedding") != null) return .last_token;
        return null;
    }
    if (std.mem.eql(u8, model_type, "bert")) {
        if (std.mem.indexOf(u8, lower, "bge-") != null) return .cls;
        if (std.mem.indexOf(u8, lower, "mxbai-embed") != null) return .cls;
        return null;
    }
    return null;
}

pub const Qwen4NormConvention = enum { delta, folded };

pub const Llama3Rope = @import("../blocks/rope/rope.zig").Llama3Rope;

/// `thinker_config.audio_config` of a Qwen3-ASR pack (mlx-audio's `AudioEncoderConfig`).
pub const QwenAudioConfig = struct {
    d_model: u32 = 1024,
    layers: u32 = 24,
    heads: u32 = 16,
    ffn: u32 = 4096,
    mel_bins: u32 = 128,
    downsample: u32 = 480,
    /// Half a conv chunk in mel frames.
    n_window: u32 = 50,
    /// Attention window in mel frames.
    n_window_infer: u32 = 800,
    output_dim: u32 = 2048,
};

/// Qwen3-ForcedAligner: the trunk's head classifies each `<timestamp>` row
/// into steps of `step_ms` (src/plugins/qwen3_asr/forced_align.zig).
pub const ForcedAlignerConfig = struct {
    timestamp_token_id: u32 = 151705,
    step_ms: f32 = 80,
    /// `support_languages`, as `transcription.languageBit`s.
    languages: u32 = 0,
};

pub const ModelConfig = struct {
    // Architecture identity
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "language_model.model",

    // Core dimensions
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    /// Whether `intermediate_size` came from the JSON or is the struct default
    /// above. MoE checkpoints routinely omit the key (DSV4 ships none at all),
    /// and a consumer that cannot tell the two apart bills the 15360 default as
    /// though the model declared it — 3.96 GB of phantom MLP transient in the
    /// prefill guard. Only readers that need the DISTINCTION should look here;
    /// everyone else keeps using `intermediate_size` and its fallback value.
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    rms_norm_eps: f32 = 1e-6,
    /// K2-Horizon: every RMS norm normalizes `hidden_size / norm_groups`-wide channel groups on their own rms, then applies the full weight.
    norm_groups: u32 = 1,
    /// Prism Hadamard packs: every `<linear>.signs` weight reads `H_block(signs * x)` (rht.zig); 0 = none.
    hadamard_block: u32 = 0,

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false, // Gemma 4: full attention uses proportional RoPE
    rope_proportional_factor: f32 = 1.0,
    rope_llama3: ?Llama3Rope = null,

    // Sliding window attention
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,

    // Quantization. 0 = dense bf16 (config.json has no "quantization" key);
    // quantized checkpoints always set this from that key (see parseConfig).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    // Attention scale: 1/sqrt(query_pre_attn_scalar) for Gemma, 1/sqrt(head_dim) for others
    query_pre_attn_scalar: u32 = 256,

    // Architectural differences between model families
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,
    // Learned per-head attention sinks (gpt_oss `self_attn.sinks`, [n_heads]):
    // one extra logit column that lands in the softmax DENOMINATOR only, so
    // every head can attend to "nothing". mlx's fused SDPA takes them
    // natively (`mlx_fast_scaled_dot_product_attention`'s `sinks` argument);
    // this flag is what makes the loader fetch the weight and the forward
    // pass it instead of the null array.
    has_attn_sinks: bool = false,
    // gpt_oss clamped SwiGLU. Non-zero limit selects
    //   clip(gate, max=limit) * sigmoid(alpha*gate) * (clip(up, ±limit) + 1)
    // over the standard silu(gate)*up. The `+ 1` on the linear branch and the
    // asymmetric clip are both load-bearing; `hidden_act` in a gpt_oss
    // config.json says "silu" and is never read by the reference.
    swiglu_limit: f32 = 0.0,
    swiglu_alpha: f32 = 1.702,
    /// GLM-5-Next's form under the same limit: silu(min(gate, limit)) * clip(up, ±limit).
    swiglu_clamp: bool = false,

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    /// Nemotron-H Puzzle: routed experts run in this width, not `hidden_size`.
    /// 0 means the uniform trunk (no fc1/fc2_latent_proj).
    moe_latent_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    // DeepSeek-V3-style sigmoid routing (hy_v3): scores = sigmoid(logits) in
    // f32; top-k SELECTED on scores + expert_bias but WEIGHTED by the unbiased
    // scores; optional renorm (/(sum+1e-20)) then × router_scaling_factor.
    moe_sigmoid_router: bool = false,
    // kolibri1: top-k SELECTED on logits + expert_bias (no sigmoid before the
    // bias), WEIGHTED by sigmoid(logits) of the picks; requires moe_sigmoid_router.
    moe_logit_select: bool = false,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    // Layers [0, first_k_dense_replace) use a dense MLP instead of MoE
    // (hy_v3: layer 0 dense at intermediate_size, the rest MoE).
    first_k_dense_replace: u32 = 0,
    // Laguna: MoE router logit soft-capping (tanh). 0 = off (the shipped
    // Laguna-S-2.1 checkpoint sets moe_router_logit_softcapping: 0.0).
    moe_router_logit_softcapping: f32 = 0.0,

    // Grouped ("noaux_tc") expert routing: split the biased scores into
    // moe_n_group equal groups, keep the moe_topk_group best by their top-2
    // sum, then take the global top-k inside the survivors. 1/1 = ungrouped.
    moe_n_group: u32 = 1,
    moe_topk_group: u32 = 1,

    // Linear attention (GatedDeltaNet)
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    /// Set at parse for qwen4_exp (`gdnStateF32Enabled`); read through `ssmStateDtype`.
    ssm_state_f32: bool = false,

    // KDA (Kimi Delta Attention, bailing_hybrid) variations on the
    // GatedDeltaNet recurrence:
    //   - the forget gate is PER CHANNEL ([B,T,H,Dk]) rather than per head, so
    //     the fused kernel indexes `g` by the key channel (kda_vector_gate);
    //   - a non-zero lower bound replaces the softplus gate entirely with
    //     `g = bound * sigmoid(exp(A_log) * (a + dt_bias))`, which is bounded
    //     in (bound, 0) instead of (-inf, 0) — it is NOT a clamp on the
    //     softplus form (fla/ops/kda/fused_recurrent.py);
    //   - the output gate is a plain sigmoid, not SiLU/swish.
    kda_vector_gate: bool = false,
    kda_gate_lower_bound: f32 = 0.0, // 0 = plain -exp(A_log)·softplus form
    kda_sigmoid_out_gate: bool = false,

    // Multi-head Latent Attention (bailing_hybrid's full-attention layers,
    // DeepSeek-V3 shape): low-rank Q (q_a_proj → q_a_layernorm → q_b_proj) and
    // a single compressed KV latent (kv_a_proj_with_mqa → kv_lora_rank latent
    // + qk_rope_head_dim shared rope key) expanded per head by kv_b_proj.
    // Query/key head dim is nope+rope; the value head dim is SMALLER, so the
    // KV cache holds asymmetric K/V (MLX's SDPA has a 192/128 vector kernel).
    // mla_head_gate: per-head sigmoid gate on the attention output (the
    // checkpoint's `head_wise` gated_attention_proj_granularity_type).
    mla_q_lora_rank: u32 = 0, // 0 = not an MLA arch
    mla_kv_lora_rank: u32 = 0,
    mla_qk_nope_head_dim: u32 = 0,
    mla_qk_rope_head_dim: u32 = 0,
    mla_v_head_dim: u32 = 0,
    mla_head_gate: bool = false,
    // GLM-5-Next DeepSeek sparse attention: absorbed NoPE MLA over a `dsa_kv_lora_rank`
    // latent (per-head q/k and v widths `dsa_head_dim`) plus a pooled indexer that picks
    // `dsa_index_topk` tokens in `dsa_index_kpool`-token pools.
    dsa_q_lora_rank: u32 = 0,
    dsa_kv_lora_rank: u32 = 0,
    dsa_head_dim: u32 = 0,
    dsa_index_topk: u32 = 0,
    dsa_index_kpool: u32 = 0,
    dsa_index_heads: u32 = 0,
    dsa_index_head_dim: u32 = 0,
    // RoPE rotates ADJACENT PAIRS (x[2i], x[2i+1]) instead of halves — mlx's
    // `traditional` rope. Set by rope_interleave.
    rope_interleaved_pairs: bool = false,

    // Hybrid attention
    full_attention_interval: u32 = 0,
    // Hybrid archs: layers at or past this index are ALWAYS full
    // attention, whatever the interval says. The reference's rule is
    // `(idx+1) % group == 0 OR idx >= n_layers // group * group`, i.e. the
    // ragged tail after the last WHOLE group never gets linear attention.
    // 0 = no tail bound, which is every other hybrid arch (qwen3_next, lfm2).
    linear_attn_tail_from: u32 = 0,
    /// Pipeline-parallel rank: the trunk layers this process holds (`applyLayerSpan`).
    dist_span: ?distributed.LayerSpan = null,
    /// DeepSeek-V4.1 on a two-Mac pair: the worker's layers, kept under their own numbers.
    dist_dsv41: ?@import("../../plugins/deepseek_v41/deepseek_v41.zig").Pair = null,
    /// Tensor-parallel rank: its slice of each MoE expert (`applyShard`).
    dist_shard: ?distributed.Shard = null,
    partial_rotary_factor: f32 = 1.0,
    attn_output_gate: bool = false,

    // Qwen4-Exp (Qwen3.8-Flash-Next): gated residual streams ("hyper
    // connections", hc_count x hidden wide), a hashed n-gram embedding
    // injected at ONE layer (PLE), and Qwen Sparse Attention (indexer-selected
    // 4-token blocks past `indexer_budget` tokens). hc_count 0 = none.
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
    embedded_ple_payload_bytes: ?u64 = null,
    qwen4_norm_convention: ?Qwen4NormConvention = null,
    indexer_n_heads: u32 = 0, // 0 = dense attention
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    /// Routed experts in EXL3 (Sushi packs, `expert_quant` in config.json),
    /// served by src/sdk/quant/exl3; null = the pack's own affine banks.
    exl3: ?exl3.Spec = null,
    /// A fully EXL3 pack (`trunk_quant`): the trunk's linears are a quant format's, loaded and run through the claimant of
    /// that block (turboexl3's, linked in: no built-in format serves a dense EXL3 linear).
    trunk_format: bool = false,
    /// The n-gram table's file in the pack (`ngram_table.file`); empty = `ngram_table.bin`.
    ngram_table_file_buf: [96]u8 = @splat(0),
    ngram_table_file_len: u8 = 0,
    /// An NVIDIA ModelOpt checkpoint served as published (`modelopt_pack`).
    modelopt: bool = false,
    /// The TEXT config's own eos (its first entry): the n-gram hash's segment
    /// reset token, independent of the generation-time stop set.
    ngram_eos: u32 = 0,
    /// `<model_dir>/ngram_table.bin` (or the file `ngram_table_file_buf` names) for the PLE table (mmapped by the
    /// engine, never mlx-loaded). Set by `parseConfig`; lives as long as the config does.
    ngram_table_path: ?[]const u8 = null,
    /// The pack's directory: deepseek_v41's Engram tables sit beside the
    /// weights, an mlx-stream pack (`glm_stream`) loads from it, and an `expert_bank`
    /// pack streams its experts from it. Set by `parseConfig`.
    pack_dir: ?[]const u8 = null,

    // Laguna: softplus per-head attention output gate. self_attn.g_proj →
    // softplus(fp32) → per-head scalar × attn output (reshaped [..,H,D]) before
    // o_proj. Distinct from attn_output_gate (qwen3-next sigmoid + doubled
    // q_proj) — separate weight, softplus activation, per-head broadcast.
    laguna_attn_gate: bool = false,
    // Laguna: per-layer Q-head count (full-attention layers 48, sliding 72;
    // KV heads uniform at num_key_value_heads). 0 = uniform num_attention_heads.
    num_attention_heads_per_layer: [128]u32 = @splat(0),
    has_per_layer_heads: bool = false,

    // MuseGlimmer (muse_glimmer): weight-less shared QK RMS-norm with Q scaled
    // by qk_scale_factor (folded into attnScale — a scalar commutes through
    // RoPE and QK^T), elementwise sigmoid attention output gate from a
    // separate self_attn.gate_proj read off the post-input-norm hidden,
    // RMS-normed embeddings (NO sqrt(hidden) scale), Gemma2-centered sandwich
    // norms whose POST norms use post_norm_eps while the FINAL norm is
    // plain-scale (ones-init), logits scaled by output_multiplier before the
    // tanh softcap, and NoPE on layers whose layer_rope_theta entry is 0
    // (exactly the full-attention layers in the released checkpoint).
    qk_scale_factor: f32 = 0.0, // 0 = off
    output_multiplier: f32 = 0.0, // 0 = off
    post_norm_eps: f32 = 0.0, // 0 = same as rms_norm_eps
    final_norm_plain: bool = false, // final norm skips the (1+w) fold
    qk_norm_weightless: bool = false, // param-free RMS on Q and K heads
    normed_embeddings: bool = false, // param-free RMS after embedding lookup
    attn_sigmoid_gate: bool = false, // attn_out *= sigmoid(gate_proj(normed))
    attn_gate_headwise: bool = false, // the gate is ONE scalar per head (g_proj [heads, hidden])
    attn_fused_qkv: bool = false, // checkpoint ships self_attn.q_k_v_proj = [q | k | v] rows
    layer_no_rope: [128]bool = @splat(false),

    // Laguna YaRN RoPE (full-attention layers only; sliding layers use default
    // RoPE at rope_local_base_freq). rope_yarn gates the freqs + mscale
    // precompute at model load; the sliding/full split is by isGlobalLayer.
    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_attention_factor: f32 = 1.0,
    /// HF's `truncate` (default true): floor/ceil the ramp correction bounds to
    /// whole dims. Only the flat-`rope_parameters` readers set it.
    yarn_truncate: bool = true,

    // Inkling (inkling_mm_model, Thinking Machines Inkling Small). NO RoPE:
    // position = the RelativeLogits bias (per-layer wr_du → [heads, d_rel]
    // relative states × a learned [d_rel, extent] profile bank → additive bias
    // over backward distances) + four depthwise causal short-convolutions per
    // layer + log-scaling on global layers past inkling_log_n_floor tokens.
    inkling_d_rel: u32 = 0, // 0 = not an inkling arch
    inkling_rel_extent: u32 = 0, // global-layer bias extent; sliding layers use their window
    inkling_log_n_floor: u32 = 0, // 0 = log-scaling off (exact no-op below the floor)
    inkling_log_alpha: f32 = 0.1,
    inkling_sconv_kernel: u32 = 0, // 0 = no short convolutions
    // Router-gated stacked shared experts (the routing "sink": their weights
    // come from the same softmax as the routed top-k). Distinct from qwen/hy3
    // shared experts (ungated always-added).
    inkling_n_shared_experts: u32 = 0,
    // muP logit scaling: hidden /= this before the unembed matmul (1 = off).
    logits_mup_width_multiplier: f32 = 1.0,
    // Slice logits to the first N rows (vocab padding; 0 = full vocab).
    unpadded_vocab_size: u32 = 0,

    // DeepSeek V4 Flash (deepseek_v4). MQA over ONE head_dim-wide latent
    // (num_key_value_heads == 1): low-rank Q (wq_a → q_norm → wq_b, then an
    // UNWEIGHTED per-head RMS), grouped low-rank O (o_groups slabs of
    // o_lora_rank), rope on the last dsv4_rope_head_dim dims with INVERSE
    // rope on the attention output; per-head attn_sink joins the softmax
    // denominator only. Every layer slides over the last sliding_window raw
    // latents; layers with dsv4_compress_ratios[i] != 0 add learned
    // gated-pooling compression of the history (ratio 4 = overlapping windows
    // + a top-dsv4_index_topk indexer over its own fp4/Hadamard-simulated
    // compressed keys; other ratios plain, all compressed slots visible).
    // YaRN applies ONLY on compressed layers at dsv4_compress_rope_theta
    // (ratio-0 layers run plain rope_theta, no yarn, and there is NO yarn
    // mscale anywhere — the reference applies none). The residual stream is
    // dsv4_hc_mult copies mixed per token by Sinkhorn-normalized
    // hyper-connections. The first dsv4_hash_layers MoE layers route by
    // TOKEN ID (gate.tid2eid). Reference: the release's own
    // inference/{model,kernel}.py; full notes in memory dsv4-port.
    dsv4_q_lora_rank: u32 = 0, // 0 = not a deepseek_v4 arch
    dsv4_o_lora_rank: u32 = 0,
    dsv4_o_groups: u32 = 0,
    dsv4_rope_head_dim: u32 = 0,
    dsv4_hash_layers: u32 = 0,
    dsv4_index_n_heads: u32 = 0,
    dsv4_index_head_dim: u32 = 0,
    dsv4_index_topk: u32 = 0,
    dsv4_hc_mult: u32 = 0,
    dsv4_hc_sinkhorn_iters: u32 = 0,
    dsv4_hc_eps: f32 = 1e-6,
    dsv4_swiglu_limit: f32 = 0.0,
    dsv4_compress_rope_theta: f32 = 0.0,
    // Per-layer compression ratio (0 = pure sliding window). Entries beyond
    // num_hidden_layers describe the MTP module(s).
    dsv4_compress_ratios: [128]u8 = @splat(0),
    dsv4_n_compress_ratios: u32 = 0,
    dsv4_mtp_layers: u32 = 0,
    // DSpark block-parallel speculative decoding (0731 and later). The draft
    // stages live under the SAME `mtp.*` namespace as the preview's single
    // MTP module — `dspark_block_size != 0` is what tells the two apart:
    // stage 0 projects the concatenated hidden states of
    // `dspark_target_layer_ids` (main_proj/main_norm, replacing the preview's
    // e_proj/h_proj) and drafts a whole block of `dspark_block_size` slots
    // seeded with `dspark_noise_token_id`, the last stage adding a rank-
    // `dspark_markov_rank` bigram bias plus a confidence head. Parsed here so
    // the engine can tell a DSpark checkpoint from a preview one BEFORE
    // touching weights; the draft path itself is not wired yet.
    dsv4_dspark_block_size: u32 = 0,
    dsv4_dspark_noise_token_id: u32 = 0,
    dsv4_dspark_markov_rank: u32 = 0,
    dsv4_dspark_target_layers: [8]u8 = @splat(0),
    dsv4_n_dspark_target_layers: u32 = 0,

    // DeepSeek-V4.1 (deepseek_v41) on top of the dsv4_* block: only the
    // `kv_sources` layers compress KV and only the `index_sources` run an
    // indexer; every other compressed layer reads the latest source at or
    // above it. The candidate source pools its scores into blocks that later
    // index sources mask with. Engram adds an n-gram table lookup to the
    // stream before its layers; the DSpark head has its own MoE size.
    dsv41_kv_sources: u64 = 0, // bit i = layer i
    dsv41_index_sources: u64 = 0,
    dsv41_candidate_source: i32 = -1,
    dsv41_candidate_topk_blocks: u32 = 0,
    dsv41_candidate_block_size: u32 = 0,
    dsv41_engram_layers: [4]u8 = @splat(0),
    dsv41_engram_rows: [4]u64 = @splat(0),
    dsv41_n_engram_layers: u32 = 0,
    dsv41_engram_max_ngram: u32 = 0,
    dsv41_engram_vocab_size: u64 = 0,
    dsv41_engram_n_heads: u32 = 0,
    dsv41_engram_head_dim: u32 = 0,
    dsv41_engram_pad_id: u32 = 0,
    dsv41_engram_compressed_vocab: u32 = 0,
    dsv41_dspark_experts: u32 = 0,
    dsv41_dspark_top_k: u32 = 0,
    /// The pack's routed experts are an mlx-stream `experts.bin` bank (a MiMo pack whose config declares
    /// `expert_streaming`): the plugin streams them into slot rows (`stream_experts`, else the fill,
    /// `bills.expertBankFillRows`). Set by `parseConfig`.
    expert_bank: bool = false,
    /// A GLM-5.3 pack (`glm_moe_dsa`, its routed experts in an `experts.bin` bank): mlx-stream serves it
    /// (`arch/mlx_stream.zig`) and loads its weights. Set by `parseConfig`.
    glm_stream: bool = false,

    // BERT encoder-only
    is_encoder_only: bool = false,
    layer_norm_eps: f32 = 1e-12,
    type_vocab_size: u32 = 0,

    /// Sentence-transformers pooling for /v1/embeddings (issue #116). null =
    /// no explicit signal → masked mean (the historical behavior, correct for
    /// MiniLM-class BERTs and EmbeddingGemma). Set from config.json
    /// `pooling_mode`, the ST `1_Pooling/config.json` sidecar, or the
    /// known-family name fallback (`poolingFromDirName`). A non-null mode on a
    /// decoder arch (Qwen3-Embedding) also advertises the `embeddings`
    /// capability WITHOUT flipping `is_encoder_only` — the forward stays the
    /// arch's own causal pass.
    pooling_mode: ?PoolingMode = null,

    // Bidirectional-attention embedding models (EmbeddingGemma): a decoder
    // arch (gemma3_text) trained as an encoder. Implies is_encoder_only.
    use_bidirectional_attention: bool = false,
    // BOS id from config.json (embedding models wrap inputs <bos>…<eos>).
    bos_token_id: ?u32 = null,

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    /// Auto-context, FROZEN at model-load time (`server.pinAutoContext`).
    /// 0 = not pinned yet.
    ///
    /// Without `--ctx-size` the effective context used to be recomputed from
    /// LIVE memory on every request, so the number the server advertised drifted
    /// as other processes took RAM (measured: 92,387–94,883 across one session).
    /// Agent CLIs budget their own `max_tokens` against that advertised value,
    /// so it has to hold still for the model's whole residency. Explicit
    /// `--ctx-size` still wins over this.
    pinned_context: u32 = 0,

    /// Per-model settings from `model-settings.json`, set at the load
    /// construction site. `ctx_override` 0 = the process `--ctx-size`/auto;
    /// `kv_quant_override` null = the process `--kv-quant`; `mtp_override`
    /// true = head loaded AND on by default (the `--mtp` force, per model).
    ctx_override: u32 = 0,
    kv_quant_override: ?kv_quant_mod.KVQuantConfig = null,
    mtp_override: ?bool = null,
    /// null = the process `--mtp-typical`/`--mtp-tokenv3` (exact when neither).
    mtp_acceptance_override: ?mtp_acceptance_mod.Mode = null,
    /// null = the process `--mtp-greedy-tail` (off when absent).
    mtp_greedy_tail_override: ?bool = null,
    /// null = off (the route changes numerics).
    int8_prefill_override: ?bool = null,
    /// LOSSY MXFP8 copy of a dense head; null = `--lm-head-mxfp8`, else the integrated-CUDA default.
    lm_head_mxfp8_override: ?bool = null,
    /// The model's `ane` setting: serve it on the Neural Engine; null follows `--ane`.
    ane_override: ?bool = null,
    /// The model's `stream_experts` setting: its routed experts read from the pack into slot rows; null = resident.
    stream_experts_override: ?@import("../weights/streamed.zig").Setting = null,
    /// Dense context K/V a loaded DFlash drafter keeps per trunk token, per request. Stamped at load.
    drafter_ctx_bytes_per_token: u64 = 0,
    /// `drafter` setting: null or "auto" = the in-dir probe, "off", or a path. Owned.
    drafter_override: ?[]const u8 = null,
    /// The plugin serving this pack and its parse (`engines/mlx/plugin/config.zig`); null = the legacy path.
    plugin: ?*plugin_config.Binding = null,
    /// The facts the legacy parse declares for its family; a served config reads its plugin's (`caps`,
    /// `defaultEnableThinking`).
    legacy: struct { caps: abi.Caps = .{}, thinking: abi.ThinkingDefault = .unset } = .{},
    /// Every quantized projection fits the row-exact kernels (`Transformer.init`).
    row_exact_covered: bool = false,
    /// A DFlash drafter is bound (`DflashModel.bind`): its block tree needs exact verify rows.
    dflash_bound: bool = false,

    /// The prefill chunk this model was sized for, FROZEN at load
    /// (`server.pinPrefillChunk`). 0 = not pinned yet, which keeps the
    /// launch/base chunk.
    ///
    /// The chunk is the multiplier on the biggest transient in the memory bill
    /// (`8 x chunk x max(hidden, ffn) x 2`, three of them). Nothing used to size
    /// it to the MACHINE, so a 16 GB Mac reserved the same 5-7 GB envelope a
    /// 128 GB one does, which is most of its budget: the sizer then reported a
    /// 1024-token context and the admission guard refused prompts whose real
    /// peak was a third of the bill. The sizer, `checkAttentionMemory` and
    /// `generate.effectivePrefillChunk` all read THIS field, so the bill and the
    /// forward can never disagree. Explicit `--prefill-chunk` still wins.
    pinned_prefill_chunk: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    // Model-author sampling recommendations from generation_config.json
    // (e.g. Qwen 3.6: temp 1.0 / top_p 0.95 / top_k 20; Gemma 4: top_k 64).
    // null = the file or key is absent. Used as defaults for request fields
    // the client OMITTED — Claude Code sends no sampling params at all, and
    // pre-2026-06 it sampled the full untruncated distribution at temp 1.0,
    // well outside the model card's intended envelope.
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,

    // The checkpoint's OWN thinking default, from generation_config.json's
    // `default_chat_template_kwargs.enable_thinking`. null = the file or key
    // is absent. Read by `defaultEnableThinking` for requests that name no
    // thinking preference; an explicit request value still outranks it.
    gen_enable_thinking: ?bool = null,

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
    /// EmbeddingGemma 2's processors (`processor_config.json`): soft tokens per image and per video FRAME, and
    /// the most frames kept from a video. The processors' own defaults until the file says otherwise.
    gv_image_tokens: u32 = 280,
    gv_video_tokens: u32 = 70,
    gv_video_frames: u32 = 32,
    /// Those processors serve this pack: each image sized to its own budget (the tower's grid is its sequence,
    /// unpadded), each video frame an image wrapped as one; the encoder embeds `messages`.
    gv_budgeted: bool = false,
    vision_hidden_size: u32 = 768,
    vision_num_layers: u32 = 16,
    vision_num_heads: u32 = 12,
    vision_head_dim: u32 = 64,
    vision_intermediate_size: u32 = 3072,
    vision_patch_size: u32 = 16,
    vision_pooling_kernel: u32 = 3,
    vision_soft_tokens: u32 = 280,
    vision_position_embedding_size: u32 = 10240,
    vision_rope_theta: f32 = 100.0,
    vision_use_clipped_linears: bool = true,
    image_token_id: u32 = 0, // 0 = no image token
    boi_token_id: u32 = 0, // beginning of image
    eoi_token_id: u32 = 0, // end of image

    // Gemma 4 12B "unified" (encoder-free) multimodal. Instead of the SigLIP
    // transformer tower, vision is a single patch embedder
    // (LN → Dense → LN → +factorized 2D posemb → LN → RMSNorm → Linear) and
    // audio is raw 640-sample frames projected straight to text space. Set
    // when model_type is gemma4_unified*. See src/engines/mlx/vision_encoder.zig (UnifiedEmbedder).
    is_gemma4_unified: bool = false,
    vision_mm_embed_dim: u32 = 0, // unified: mm_embed_dim (3840 for 12B = text hidden)
    vision_model_patch_size: u32 = 0, // unified: 48px merged "model patch" (16px teacher × 3 pool)
    vision_mm_posemb_size: u32 = 0, // unified: factorized position table size per axis (1120)
    // Audio (gemma4_unified). embed_audio projects audio_embed_dim → text hidden.
    audio_token_id: u32 = 0, // 0 = no audio token
    boa_token_id: u32 = 0, // beginning of audio
    eoa_token_id: u32 = 0, // end of audio
    audio_embed_dim: u32 = 0, // unified: raw samples per token (640)
    audio_samples_per_token: u32 = 640, // 40ms @ 16kHz
    /// Qwen3-ASR's audio tower (src/plugins/qwen3_asr/audio_tower.zig); the trunk is a plain qwen3.
    qwen_audio: ?QwenAudioConfig = null,
    /// Set on Qwen3-ForcedAligner, the ASR shape with a timestamp head.
    forced_aligner: ?ForcedAlignerConfig = null,

    // Qwen3.5/3.6 vision (Qwen3-VL ViT). Distinct from the Gemma SigLIP fields
    // above: Qwen ships a fused-qkv ViT with a patch merger, and the text trunk
    // uses INTERLEAVED M-RoPE (image tokens get 2D grid positions). Populated in
    // the qwen3_5 arm below. Encoder: src/sdk/vision/qwen_vision.zig; M-RoPE: src/sdk/vision/mrope.zig.
    qwen_vision: bool = false,
    /// Nemotron RADIO tower, selected by `vision_config.model_type`.
    radio_vision: bool = false,
    radio_num_heads: u32 = 0,
    radio_final_layernorm: bool = false,
    radio_ln_eps: f32 = 1e-6,
    /// RADIO video EVS drop fraction. 0 keeps every tubelet token. The Bolt
    /// checkpoint uses 0.7; other models leave this at 0.
    video_pruning_rate: f64 = 0,
    qv_depth: u32 = 0, // ViT transformer blocks
    qv_hidden: u32 = 0, // ViT hidden size
    qv_heads: u32 = 0, // ViT attention heads
    qv_head_dim: u32 = 0, // = qv_hidden / qv_heads
    qv_intermediate: u32 = 0, // ViT MLP intermediate
    qv_patch: u32 = 16, // pixel patch size
    qv_temporal_patch: u32 = 2, // frames folded per patch (still image duplicated)
    qv_merge: u32 = 2, // spatial merge: merge×merge patches → one LLM token
    qv_num_pos_emb: u32 = 0, // learned pos table entries (e.g. 2304 = 48×48)
    qv_out_hidden: u32 = 0, // merger output dim (= text hidden_size)
    // Image-area bounds from processor_config.json / preprocessor_config.json.
    // 0 means absent: the Qwen processor defaults remain the fallback.
    qv_min_pixels: u32 = 0,
    qv_max_pixels: u32 = 0,
    // Muse-Glimmer vision (src/plugins/muse_glimmer/vision.zig). Shares the qv_* geometry but
    // NOT the Qwen ViT: split qkv, learned pos table resampled per image,
    // window/full attention per layer, and plain 1D text positions (no M-RoPE).
    muse_vision: bool = false,
    mv_pos_side: u32 = 0, // learned pos table is pos_side x pos_side
    mv_projector_hidden: u32 = 0, // vision_adapter width
    mv_ln_eps: f32 = 1e-5,
    mv_rope_theta: f64 = 10000.0,
    mv_max_image_tokens: u32 = 0, // processor cap, in MERGED tokens
    mv_full_attn: [MAX_VISION_LAYERS]bool = @splat(false),
    // LFM2-VL vision (src/plugins/lfm2/vision.zig). The tower's geometry comes from the
    // generic vision_* fields above (it is a stock SigLIP2); these are LFM2-VL's
    // own wrapper — the projector, and the NaFlex processor's token budget.
    lfm2_vision: bool = false,
    lv_pos_side: u32 = 0, // learned pos table is pos_side x pos_side
    lv_downsample: u32 = 2, // projector pixel-unshuffle factor
    lv_projector_hidden: u32 = 0,
    lv_ln_eps: f32 = 1e-6,
    lv_min_image_tokens: u32 = 64,
    lv_max_image_tokens: u32 = 256,
    lv_tile_size: u32 = 512,
    lv_min_tiles: u32 = 2,
    lv_max_tiles: u32 = 10,
    lv_split_images: bool = true,
    lv_use_thumbnail: bool = true,
    lv_pixels_tolerance: f32 = 2.0,
    lv_thumbnail_token_id: u32 = 0,
    lv_row_col_base_id: u32 = 0, // id of `<|img_row_1_col_1|>`; the block is row-major
    // GLM-5.3-Flash vision (src/plugins/glm5_next/vision.zig). Geometry rides the qv_* fields (same
    // vision_config keys as Qwen's); these are the tower's own extras and the
    // processor's token budget, in MERGED tokens.
    glm_vision: bool = false,
    glmv_proj_intermediate: u32 = 10240,
    glmv_eps: f32 = 1e-5,
    glmv_swiglu_limit: f32 = 10.0,
    glmv_min_image_tokens: u32 = 16,
    glmv_max_image_tokens: u32 = 8000,

    // DeepSeek-V4.1 vision tower (src/plugins/deepseek_v41/vision.zig): the release's `vision_config`, on top of the
    // generic vision_* geometry. `dv_max_wh_ratio` 0 = no cap (the release's null).
    dsv41_vision: bool = false,
    /// The serving plugin runs its own towers (`abi.MediaHooks`): the host hands them decoded RGB frames and samples.
    plugin_media: bool = false,
    dv_downsample: u32 = 3,
    dv_max_image_tokens: u32 = 1024,
    dv_min_pixels: u32 = 295936,
    dv_max_wh_ratio: u32 = 0,
    // MiniCPM-V 4.7 vision (src/plugins/qwen3_5/minicpm_vision.zig): the SigLIP2 geometry rides the vision_*
    // fields; these are its window merger, slicing and placeholder tokens. The token ids
    // that are not in config.json come from the tokenizer (`populateMiniCpmImageTokens`).
    minicpm_vision: bool = false,
    mcv_insert_layer: u32 = 6,
    mcv_downsample_4x: bool = false,
    mcv_max_slice_nums: u32 = 9,
    mcv_eps: f32 = 1e-6,
    mcv_image_start_id: u32 = 0,
    mcv_image_end_id: u32 = 0,
    mcv_slice_start_id: u32 = 0,
    mcv_slice_end_id: u32 = 0,
    mcv_newline_id: u32 = 0,
    mcv_image_id_start: u32 = 0,
    mcv_image_id_end: u32 = 0,
    mcv_digit_ids: [10]u32 = @splat(0),
    // Interleaved M-RoPE sections [t, h, w]; sum = rotary_dim/2 (e.g. [11,11,10]).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    // Qwen vision token ids (top-level config.json). image_token_id reuses the
    // shared field above (parsed generically at the image_token_id block).
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
    v_head_dim: u32 = 0, // 0 = same as the layer's head_dim (MiMo: K 192 / V 128)
    attention_value_scale: f32 = 1.0, // V multiplied before caching (MiMo-V2.6: 0.707)
    num_global_key_value_heads: u32 = 0, // 0 = same as num_key_value_heads
    num_kv_shared_layers: u32 = 0,
    final_logit_softcapping: f32 = 0.0, // 0 = disabled
    hidden_size_per_layer_input: u32 = 0, // >0 enables PLE
    partial_rotary_factor_global: f32 = 1.0, // for global/full attention layers
    has_v_norm: bool = false, // parameter-free RMS norm on values
    // Gemma 4 (31B): full_attention layers share V with K (no v_proj stored)
    attention_k_eq_v: bool = false,

    // Hybrid layers (LFM2, Nemotron-H): per-layer type dispatch
    has_hybrid_layers: bool = false,
    layer_block_types: [128]LayerBlockType = @splat(.attention),
    has_embedding_norm: bool = false, // LFM2: RMS norm applied to embeddings
    has_final_norm: bool = true, // false for LFM2 (no model.norm.weight)

    // LFM2 gated convolution
    lfm_conv_kernel: u32 = 3,
    /// LFM2.5-8B-A1B (`lfm2_moe`): the hybrid trunk's per-layer MLP is a
    /// sparse MoE from `num_dense_layers` on. `model_type` collapses to
    /// "lfm2" (same conv/attention mixers), so this flag is what tells the
    /// layer loader which feed-forward to bind.
    lfm2_moe: bool = false,
    /// First N layers keep a DENSE feed-forward; the rest are MoE.
    num_dense_layers: u32 = 0,
    lfm_conv_dim: u32 = 0, // 0 = hidden_size

    // Mamba2 SSM (Nemotron-H)
    mamba_num_heads: u32 = 0,
    mamba_head_dim: u32 = 0,
    mamba_n_groups: u32 = 8,
    ssm_state_size: u32 = 128,
    mamba_conv_kernel: u32 = 4,
    mamba_expand: u32 = 2,
    time_step_min: f32 = 0.0,
    time_step_max: f32 = std.math.inf(f32),
    mamba_chunk_size: u32 = 256,
    mamba_mlp_act: HiddenAct = .relu_sq, // Nemotron-H MLP uses ReLU^2

    /// How many keys ONE query actually reads during prefill at this prompt
    /// length. `seq` (dense causal) unless the architecture BOUNDS its
    /// attention — the prefill admission guard's score term multiplies by this,
    /// and billing a dense key axis for a sparse arch is a spurious 400.
    ///
    /// deepseek_v4 reads a raw sliding window plus at most ONE compressed arm
    /// per layer (`deepseek_v4.zig`: `tk = wk + n_sel`, `wk = @min(m.window,
    /// seq_total)`), so the widest layer is what the guard must bill:
    ///   - `compress_ratios[i] == 0` → window only
    ///   - `== 4` → top-`index_topk` of the `seq/4` compressed slots (the
    ///     literal 4 mirrors the engine's own `if (ratio == 4)` branch)
    ///   - otherwise → ALL `seq/ratio` slots, visibility-masked
    /// plus one sink column. The all-visible arm is seq-scaled, so it overtakes
    /// the top-k arm at long context and the bound must track it rather than
    /// freezing at `index_topk`. A checkpoint declaring no ratios stays dense —
    /// an arch we cannot bound must never be billed as though we had.
    pub fn prefillAttnKeys(self: *const ModelConfig, seq: u64) u64 {
        if (self.isDsv41()) return @min(seq, @as(u64, self.sliding_window) + self.dsv4_index_topk + 1);
        if (!std.mem.eql(u8, self.model_type, "deepseek_v4")) return seq;
        const n = @min(self.dsv4_n_compress_ratios, self.dsv4_compress_ratios.len);
        if (n == 0) return seq;
        const window: u64 = @min(@as(u64, self.sliding_window), seq);
        var widest: u64 = 0;
        for (self.dsv4_compress_ratios[0..n]) |r| {
            if (r == 0) continue;
            const slots: u64 = seq / r;
            const cols: u64 = if (r == 4) @min(@as(u64, self.dsv4_index_topk), slots) else slots;
            widest = @max(widest, cols);
        }
        return @min(seq, window + widest + 1);
    }

    pub fn firstSlidingLayer(self: ModelConfig) ?u32 {
        for (0..self.num_hidden_layers) |i| if (!self.isGlobalLayer(@intCast(i))) return @intCast(i);
        return null;
    }

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        // HF/mlx-lm convention (Gemma 3): the GLOBAL layer closes each group —
        // global when `(idx + 1) % pattern == 0` (layers 5, 11, … for pattern 6).
        return (layer_idx % self.sliding_window_pattern) == self.sliding_window_pattern - 1;
    }

    /// For Gemma 4 KV sharing: get the source layer index for a shared layer.
    /// Returns null if the layer computes its own KV (not shared).
    pub fn getKVSourceLayer(self: ModelConfig, layer_idx: u32) ?u32 {
        if (self.num_kv_shared_layers == 0) return null;
        const first_shared = self.num_hidden_layers - self.num_kv_shared_layers;
        if (layer_idx < first_shared) return null;
        const is_global = self.isGlobalLayer(layer_idx);
        // Find last concrete layer of the same type (scanning downward)
        var j: u32 = first_shared;
        while (j > 0) {
            j -= 1;
            if (self.isGlobalLayer(j) == is_global) return j;
        }
        return null;
    }

    /// Get effective head_dim for a layer (global layers may use global_head_dim).
    pub fn layerHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_head_dim > 0 and self.isGlobalLayer(layer_idx)) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }

    /// Per-layer Q-head count (Laguna: 48 on full-attention layers, 72 on
    /// sliding). Every other arch has uniform heads, so this falls back to
    /// num_attention_heads. KV heads stay uniform (layerKVHeads).
    pub fn layerNumHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.has_per_layer_heads and layer_idx < 128 and self.num_attention_heads_per_layer[layer_idx] > 0) {
            return self.num_attention_heads_per_layer[layer_idx];
        }
        return self.num_attention_heads;
    }

    /// The cached V width of a layer (MiMo stores K 192 / V 128).
    pub fn layerVHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        return if (self.v_head_dim > 0) self.v_head_dim else self.layerHeadDim(layer_idx);
    }

    /// Leading channels of a head that RoPE rotates: `int(head_dim * partial)`.
    pub fn layerRopeDims(self: ModelConfig, layer_idx: u32) u32 {
        const partial = if (self.isGlobalLayer(layer_idx)) self.partial_rotary_factor_global else self.partial_rotary_factor;
        return @intFromFloat(@as(f32, @floatFromInt(self.layerHeadDim(layer_idx))) * partial);
    }

    /// Get effective num_kv_heads for a layer.
    pub fn layerKVHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.num_global_key_value_heads > 0 and self.isGlobalLayer(layer_idx)) {
            return self.num_global_key_value_heads;
        }
        return self.num_key_value_heads;
    }

    /// Turn this config into the short model one pipeline rank serves: layers
    /// `[span.start, span.end)` renumbered from 0, every per-layer table shifted with them.
    pub fn applyLayerSpan(self: *ModelConfig, span: distributed.LayerSpan) !void {
        if (span.total != self.num_hidden_layers or span.end > span.total or span.start >= span.end) return error.DistLayerSpanMismatch;
        // The interval rule `(i + 1) % interval` stays true only for a seam on a group boundary.
        if (self.full_attention_interval != 0 and span.start % self.full_attention_interval != 0) return error.DistSplitOffInterval;
        const st = span.start;
        inline for (.{ "num_attention_heads_per_layer", "layer_no_rope", "dsv4_compress_ratios", "layer_is_global", "layer_block_types" }) |f| {
            const arr = &@field(self, f);
            if (st < arr.len) std.mem.copyForwards(@TypeOf(arr[0]), arr[0 .. arr.len - st], arr[st..]);
        }
        if (self.linear_attn_tail_from != 0) self.linear_attn_tail_from -|= st;
        self.num_hidden_layers = span.len();
        self.dist_span = span;
    }

    /// The expert widths one tensor-parallel rank holds.
    pub fn applyShard(self: *ModelConfig, sh: distributed.Shard) !void {
        const quantum = sh.size * 64; // every shipped group size divides a 64-aligned part
        if (self.moe_intermediate_size % quantum != 0 or self.shared_expert_intermediate_size % quantum != 0) return error.DistShardOffGroup;
        self.moe_intermediate_size /= sh.size;
        self.shared_expert_intermediate_size /= sh.size;
        self.dist_shard = sh;
    }

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

    /// Which `partial_rotary_factor` the YaRN table covers: laguna/gemma4 scale
    /// only their full-attention layers and spell that one
    /// `partial_rotary_factor_global`; every other YaRN arch (qwen4_exp) has a
    /// single rope for the whole trunk.
    pub fn yarnPartial(self: *const ModelConfig) f32 {
        return if (self.isQwen4()) self.partial_rotary_factor else self.partial_rotary_factor_global;
    }

    /// `int(head_dim × yarnPartial())` — the rotating slice of a head, i.e. the
    /// dims the YaRN frequency table covers (qwen4_exp: 256 × 0.25 = 64, whose
    /// 32 frequencies are what `mrope_section` [11,11,10] sums to).
    pub fn yarnRotaryDim(self: *const ModelConfig) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.yarnPartial());
    }

    /// The longest sequence the rope can actually resolve. Plain:
    /// `max_position_embeddings`. YaRN: `original_max_position_embeddings ×
    /// factor` — the window HF and vLLM both derive `max_model_len` from — since
    /// a position past it aliases back inside the ramp. 0 = no rope-derived cap.
    pub fn contextCap(self: *const ModelConfig) u32 {
        if (self.plugin) |b| return std.math.cast(u32, b.parsed.shell.context_cap) orelse std.math.maxInt(u32);
        const declared = self.max_position_embeddings;
        if (!self.rope_yarn) return declared;
        const orig: f64 = @floatFromInt(self.yarn_orig_max_pos);
        const factor: f64 = @floatCast(self.yarn_factor);
        const scaled: f64 = @floor(orig * factor);
        const max_u32: f64 = @floatFromInt(std.math.maxInt(u32));
        const window: u32 = if (scaled >= max_u32) std.math.maxInt(u32) else @intFromFloat(scaled);
        return if (declared == 0) window else @min(window, declared);
    }

    /// How many layers hold an attention KV cache. A hybrid arch interleaves
    /// linear-attention layers, which carry a FIXED-SIZE recurrent state
    /// instead of a per-token cache — billing them as attention layers made
    /// the memory model charge a uniform arch's footprint for a model
    /// carrying a fraction of it (bailing_hybrid: 6 of 24).
    pub fn attnCacheLayerCount(self: *const ModelConfig) u32 {
        // A `layer_block_types` hybrid (LFM2 via `layer_types`, Nemotron-H via
        // `hybrid_override_pattern`) never sets `full_attention_interval`, so
        // the interval arm below counted EVERY layer: lfm2 caches 8 of 30 and
        // was billed 3.75x, Nemotron-H worse. Only the `.attention` blocks
        // reach `ctx.cache` in the hybrid forward — gated_conv and mamba2 hold
        // a fixed-size recurrent state in `ssm_entries` instead. Layers past
        // the 128-entry table keep the array's `.attention` default, which is
        // the direction that over-bills rather than OOMs.
        if (self.has_hybrid_layers) {
            var n: u32 = if (self.num_hidden_layers > self.layer_block_types.len)
                self.num_hidden_layers - @as(u32, self.layer_block_types.len)
            else
                0;
            var li: u32 = 0;
            while (li < self.num_hidden_layers and li < self.layer_block_types.len) : (li += 1) {
                if (self.layer_block_types[li] == .attention) n += 1;
            }
            return n;
        }
        if (self.full_attention_interval == 0) return self.num_hidden_layers;
        var n: u32 = 0;
        var i: u32 = 0;
        while (i < self.num_hidden_layers) : (i += 1) {
            if (!self.isLinearLayer(i)) n += 1;
        }
        return n;
    }

    /// Whether the prefill chunk is resolved per request (by the admission bill) instead of
    /// once at load. qwen4_exp only: a 1M session's load-time reserve pins every ordinary
    /// prompt to a narrow rung.
    pub fn perRequestPrefillChunk(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// V4.1's layers split with a worker Mac: the worker keeps ONE decode state and its exchange names no slot, so the
    /// model leaves per-slot state and batched decode and serves one request at a time.
    pub fn pairDsv41(self: *ModelConfig, p: @import("../../plugins/deepseek_v41/deepseek_v41.zig").Pair) void {
        self.dist_dsv41 = p;
        if (self.plugin) |b| {
            b.parsed.caps.slot_state = false;
            b.parsed.caps.batches_decode = false;
        }
    }

    /// The widest prompt chunk the arch's prompt pass gains from (0 = the ladder's): DeepSeek-V4.1 was tuned at 2048,
    /// and its Engram layer slows per row past 4096.
    pub fn prefillChunkMax(self: *const ModelConfig) u32 {
        return if (self.isDsv41()) 2048 else 0;
    }

    /// Rows a sliding layer keeps past its window on an arch that caches only the window
    /// (`slidingRing`): spec rollback plus the prefix-cache matches that land near an entry's end.
    pub const SLIDING_RING_SLACK: u32 = 2048;

    /// Recurrent (SSM / conv) layers beside attention: a cached state restores only at a checkpoint.
    pub fn hasSsmLayers(self: *const ModelConfig) bool {
        return self.has_hybrid_layers or self.full_attention_interval > 0;
    }

    /// Sliding layers cache a bounded ring of rows, as mlx-lm's `RotatingKVCache`, not every token.
    pub fn slidingRing(self: *const ModelConfig) bool {
        return self.isMimo();
    }

    pub fn slidingKeepRows(self: *const ModelConfig) u32 {
        return self.sliding_window + SLIDING_RING_SLACK;
    }

    /// Dense bytes of ONE row across every ring layer (0 off `slidingRing`).
    pub fn slidingRowBytes(self: *const ModelConfig) u64 {
        if (!self.slidingRing()) return 0;
        var sum: u64 = 0;
        for (0..self.num_hidden_layers) |i| {
            const li: u32 = @intCast(i);
            if (!self.isGlobalLayer(li)) sum += self.layerKvRowBytes(li);
        }
        return sum;
    }

    fn layerKvRowBytes(self: *const ModelConfig, li: u32) u64 {
        return @as(u64, self.layerKVHeads(li)) * (@as(u64, self.layerHeadDim(li)) + self.layerVHeadDim(li)) * 2;
    }

    /// Dense (bf16) KV-cache bytes ONE token occupies across the whole model.
    /// The uniform `layers × 2 × kv_heads × head_dim` formula is wrong on a
    /// hybrid MLA arch in both terms: only `attnCacheLayerCount` layers cache
    /// at all, and MLA's key (nope+rope) is WIDER than its value. Every
    /// memory estimate that sizes a KV cache reads this one helper so the
    /// auto-context sizer and the prefill admission guard cannot disagree.
    pub fn kvBytesPerToken(self: *const ModelConfig) u64 {
        // DeepSeek-V4.1 grows only its KV sources' compressed latents and index
        // keys (bf16); the window is a fixed ring per slot.
        if (self.isDsv41()) {
            var sum: u64 = 0;
            for (0..self.num_hidden_layers) |l| {
                const r: u64 = self.dsv4_compress_ratios[l];
                if (r == 0 or (self.dsv41_kv_sources >> @intCast(l)) & 1 == 0) continue;
                sum += self.head_dim / r;
                if ((self.dsv41_index_sources >> @intCast(l)) & 1 != 0) sum += self.dsv4_index_head_dim / r;
            }
            return sum * 2;
        }
        // GLM-5-Next stores one latent row plus the indexer's key and gate per token.
        if (self.isGlm5()) return @as(u64, self.attnCacheLayerCount()) * (self.dsa_kv_lora_rank + 2 * self.dsa_index_head_dim) * 2;
        // An arch that declares its own V width (MiMo) also varies KV heads per
        // layer type: billed layer by layer. Ring layers are a fixed per-slot term.
        if (self.v_head_dim > 0) {
            var sum: u64 = 0;
            for (0..self.num_hidden_layers) |i| {
                const li: u32 = @intCast(i);
                if (self.slidingRing() and !self.isGlobalLayer(li)) continue;
                sum += self.layerKvRowBytes(li);
            }
            return sum;
        }
        const widths: u64 = if (self.isMla())
            @as(u64, self.mlaQkHeadDim()) + @as(u64, self.mla_v_head_dim)
        else
            2 * @as(u64, self.head_dim);
        // MLA decompresses its latent to EVERY attention head before the write
        // (`mlaAttnWith` broadcasts the MQA rope key to `num_attention_heads`
        // and caches `[B, num_attention_heads, S, qk_dim]`), so its cache has
        // no grouping to save on — `num_key_value_heads` is the GQA question
        // and this arch never asks it. Equal on Ling 3.0 (16/16), so the
        // spelling is invisible today and would UNDER-bill the first MLA
        // checkpoint that groups — the direction that ends in an uncatchable
        // Metal OOM rather than a 400.
        const heads: u64 = if (self.isMla())
            @as(u64, self.num_attention_heads)
        else
            @as(u64, self.num_key_value_heads);
        return @as(u64, self.attnCacheLayerCount()) * heads * widths * 2;
    }

    /// Dense bf16 bytes of QSA indexer history ONE token occupies: the pooled
    /// blocks `[kv/ratio, idx_hd]` per full-attn layer. The raw keys are a fixed
    /// ring (`qsaRingBytes`, billed once per slot), not per token. Not
    /// kv-quantized. Zero on archs without an indexer. ONE copy; the billed width
    /// (copies + score bank) is `server.statePerTokenBilled`.
    pub fn qsaHistoryBytesPerToken(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 2 / ratio;
    }

    /// The raw indexer keys every live slot holds: `QSA_RING_ROWS` rows per
    /// full-attn layer, context-independent, billed once per slot.
    pub fn qsaRingBytes(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const rows = @as(u64, @intCast(@import("../../engines/mlx/transformer.zig").QSA_RING_ROWS));
        return n * rows * hd * 2;
    }

    /// f32 bytes per token of the QSA block-score operand a live slot holds
    /// (`SSMCacheEntry.qsa_score_bank`). Never in an entry. Zero without an indexer.
    pub fn qsaScoreBankBytesPerToken(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        if (@import("../blocks/attention/sparse_block/score.zig").qsaScoreFusedActiveFor(1, @intCast(self.indexer_n_heads), @intCast(self.indexer_head_dim))) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 4 / ratio;
    }

    /// Bytes one SSM checkpoint holds: recurrent state + conv window of every linear layer.
    /// The QSA key history is not here (it lands on the newest checkpoint only).
    pub fn ssmCheckpointBytes(self: *const ModelConfig) u64 {
        if (self.linear_num_value_heads == 0) return 0;
        const linear_layers: u64 = @as(u64, self.num_hidden_layers) -| self.attnCacheLayerCount();
        if (linear_layers == 0) return 0;
        const state_elem: u64 = if (self.ssmStateDtype() == .float32) 4 else 2;
        const state: u64 = @as(u64, self.linear_num_value_heads) *
            @as(u64, self.linear_value_head_dim) * @as(u64, self.linear_key_head_dim) * state_elem;
        const conv_dim: u64 = 2 * @as(u64, self.linear_num_key_heads) * self.linear_key_head_dim +
            @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim;
        const conv: u64 = @as(u64, self.linear_conv_kernel_dim) -| 1;
        return linear_layers * (state + conv * conv_dim * 2);
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

    /// Configured MTP depth (0 = auto). Hadamard packs run a grafted head that
    /// over-speculates in auto mode; 2 measured best (code 71 vs 58 tok/s).
    pub fn mtpDepth(self: *const ModelConfig, configured: u32) u32 {
        if (self.plugin) |b| if (configured == 0 and b.parsed.spec.mtp_default_depth > 0) return b.parsed.spec.mtp_default_depth;
        if (configured == 0 and self.hadamard_block > 0) return 2;
        return configured;
    }

    /// SSD prefix-cache layout marker (`kv_disk_cache.modelFingerprintWithLayout`).
    /// Nemotron-H keys were RoPE-rotated before its attention became NoPE; the
    /// marker gives it a fresh root. Null keeps every other arch's root.
    pub fn cacheLayoutNamespace(self: *const ModelConfig) ?[]const u8 {
        if (self.plugin) |b| return if (b.parsed.cache.layout_namespace.len > 0) b.parsed.cache.layout_namespace.slice() else null;
        if (std.mem.eql(u8, self.model_type, "nemotron_h")) return "nemotron-h-nope-v1";
        if (self.rope_llama3 != null) return "llama3-rope-v1";
        // A pool's completing indexer row stores the pool's key in its gate half; v2: the KDA q/k norm epsilon.
        if (self.dsa_kv_lora_rank > 0) return "glm5-pooled-keys-v2";
        // A restored bf16 state would keep running bf16: every GDN kernel takes its state's dtype.
        if (self.ssm_state_f32) return "gdn-f32-state-v1";
        return null;
    }

    /// Decode-width projections go through the row-exact kernels, whose rows
    /// get the same bits at any window width: a verify row reproduces serial
    /// decoding. Only where every projection fits them AND a DFlash drafter is
    /// bound: serial and MTP keep the faster stock paths otherwise.
    pub fn rowExactDecode(self: *const ModelConfig) bool {
        return self.row_exact_covered and self.dflash_bound and self.rowExactArch();
    }

    /// MTP acceptance for this load: exact while a DFlash drafter is bound, so
    /// the MTP rounds it yields to stay byte-exact too.
    pub fn mtpAcceptance(self: *const ModelConfig, default: mtp_acceptance_mod.Mode) mtp_acceptance_mod.Mode {
        if (self.dflash_bound) return .exact;
        return self.mtp_acceptance_override orelse default;
    }

    pub fn rowExactArch(self: *const ModelConfig) bool {
        if (self.caps().exact_drafted) return true;
        return std.mem.startsWith(u8, self.model_type, "qwen3_5") and !self.isMoe() and self.hadamard_block == 0;
    }

    /// Vocab rows a spec drafter proposes from (0 = all): Qwen3.8's tokenizer
    /// puts 99.64% of committed tokens below id 98304, so a draft reads 40% of
    /// the head. A token past it is never drafted: speed only, never output.
    pub fn draftVocab(self: *const ModelConfig) c_int {
        if (self.plugin) |b| if (b.parsed.spec.draft_vocab > 0) return if (self.rowExactDecode()) std.math.cast(c_int, b.parsed.spec.draft_vocab) orelse 0 else 0;
        if (self.rowExactDecode() and std.mem.startsWith(u8, self.model_type, "qwen3_5") and self.vocab_size >= 248320) return 98304;
        return 0;
    }

    pub fn isGlm5(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
    }

    /// The served model's speech output (`Parsed.speech`); null = it does not speak.
    pub fn speech(self: *const ModelConfig) ?abi.SpeechFacts {
        const b = self.plugin orelse return null;
        return if (b.parsed.speech.codebooks != 0) b.parsed.speech else null;
    }

    /// A plugin running its own towers whose caps take speech: its `audio_tower.*` loads.
    pub fn pluginHears(self: *const ModelConfig) bool {
        const b = self.plugin orelse return false;
        return self.plugin_media and b.parsed.caps.audio;
    }

    pub fn isDsv41(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "deepseek_v41");
    }

    pub fn isMimo(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "mimo_v2");
    }

    /// The pack is mlx-stream's (`arch/mlx_stream.zig`): a GLM-5.3 or GLM-5.3-Flash pack.
    pub fn mlxStream(self: *const ModelConfig) bool {
        return self.glm_stream;
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

    /// Does Q go through a low-rank pair, or straight from the hidden state?
    ///
    /// `q_lora_rank: null` is DeepSeek-V3's documented option and what the
    /// whole Ling 3.0 FLASH line ships (tiny ships 256). Those checkpoints
    /// carry a plain `attention.q_proj` instead of
    /// q_a_proj/q_a_layernorm/q_b_proj. 0 is the signal, since a real rank is
    /// always positive.
    pub fn mlaHasQLora(self: *const ModelConfig) bool {
        return self.mla_q_lora_rank > 0;
    }

    /// MLA query/key head dim = the non-positional part plus the rope part.
    /// This — not head_dim — is what the attention scale and the cached K's
    /// last dim are measured in.
    pub fn mlaQkHeadDim(self: *const ModelConfig) u32 {
        return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim;
    }

    /// Which of fla's two KDA gate arms this checkpoint declares. A non-zero
    /// `kda_lower_bound` REPLACES the softplus form with the bounded sigmoid;
    /// absent (0) means the softplus form, which the shared GatedDeltaNet chain
    /// already computes elementwise and therefore serves a per-channel gate
    /// unchanged. Feeding bound 0 to the bounded chain yields exp(0) = 1 — a
    /// gate that never forgets — so the arm must be chosen, never defaulted.
    pub fn kdaUsesBoundedGate(self: *const ModelConfig) bool {
        return self.kda_vector_gate and self.kda_gate_lower_bound != 0.0;
    }

    /// Epsilon of the per-head q/k RMS norm in the linear-attention prework. GLM's reference
    /// normalizes with fla's l2norm (1e-6 on the SUM of squares), which is 1e-6/dk on the mean;
    /// its post-conv keys are small enough that 1e-6 on the mean shrank them ~2%.
    pub fn gdnQkNormEps(self: *const ModelConfig) f32 {
        return if (self.isGlm5()) 1e-6 / @as(f32, @floatFromInt(self.linear_key_head_dim)) else 1e-6;
    }

    /// The pooling op /v1/embeddings runs: the explicit signal, else masked
    /// mean (the historical default — correct for MiniLM and EmbeddingGemma).
    pub fn effectivePooling(self: *const ModelConfig) PoolingMode {
        return self.pooling_mode orelse .mean;
    }

    /// Whether this model serves /v1/embeddings meaningfully: encoder-only
    /// (BERT, EmbeddingGemma) or a decoder with a declared pooling contract
    /// (Qwen3-Embedding). Drives capability advertising, never dispatch.
    pub fn hasEmbeddingCapability(self: *const ModelConfig) bool {
        return self.is_encoder_only or self.pooling_mode != null;
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

    /// The keys a batched tick reads per slot: past the gather floor, QSA's indexer budget and its open block.
    pub fn batchedEffectiveKvLen(self: *const ModelConfig, kv: u32, gather_on: bool, gather_min_kv: u32) u32 {
        if (!gather_on) return kv;
        if (kv <= gather_min_kv) return kv;
        const cap = self.indexer_budget + self.indexer_compress_ratio;
        if (cap == 0) return kv;
        return @min(kv, cap);
    }

    /// SSD-first prefix cache arch predicate; delegates to `longCtxGated`.
    pub fn ssdFirstCapable(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// True when per-request SSM/conv cache entries must exist: hybrid
    /// recurrence (LFM2/Nemotron/GDN) or a served model's recurrent entries
    /// (Inkling's four per-layer short convolutions). Shared by Transformer.init and the scheduler's per-slot
    /// allocation — the two predicates MUST agree or slots crash on a null
    /// `ctx.ssm_entries` (the Qwen3.5-MoE class).
    pub fn needsSsmEntries(self: *const ModelConfig) bool {
        if (self.plugin) |b| if (b.parsed.cache.ssm_state.entries > 0) return true;
        return self.has_hybrid_layers or self.full_attention_interval > 0;
    }

    /// Pure-config half of "can this arch ride the batched GatedDeltaNet
    /// decode kernel?" (`Transformer.forwardMoeBatchedDecode`) — a dense
    /// GDN trunk with periodic full attention, i.e. the qwen3_5 family.
    ///
    /// This exists because the answer is needed in TWO places that see
    /// different things: `server.zig` decides whether `--max-concurrent`
    /// clamps to 1 with only a ModelConfig in hand, while
    /// `Transformer.supportsBatchedGdnDecode` also checks the built layer
    /// set. Both MUST read this predicate — when they were hand-rolled
    /// separately, the server kept clamping qwen3_5 to serial decode while
    /// the scheduler was happily batching it, so `--max-concurrent 4` (the
    /// obvious serving config) silently DISABLED the batched path.
    ///
    /// Says nothing about MoE/hybrid archs that merely share the same
    /// forward — those stay serial, by name, in both callers.
    pub fn supportsBatchedGdnDecode(self: *const ModelConfig) bool {
        // A served model says so itself; its recurrent state rides the row-axis tick.
        if (self.plugin) |b| return b.parsed.caps.batches_decode and b.parsed.cache.ssm_state.entries > 0;
        if (self.full_attention_interval == 0) return false; // not a GDN trunk
        if (self.has_hybrid_layers) return false; // lfm2 / nemotron_h
        if (self.is_encoder_only) return false;
        // Routed experts are row-generic; a MoE trunk batches when its per-slot
        // state is what the path merges (GDN pair, qwen4's PLE window + QSA keys).
        if (self.isMoe() and !std.mem.eql(u8, self.model_type, "qwen3_5_moe")) return false;
        if (self.isMla() or self.isGemma4Layers()) return false;
        if (self.kda_vector_gate) return false; // bailing KDA: its own gate shape
        if (std.mem.eql(u8, self.model_type, "deepseek_v4")) return false;
        return true;
    }

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers).
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4");
    }

    /// Takes the sampling recommendations + extra eos ids of a generation_config.json.
    fn applyGenerationDefaults(self: *ModelConfig, gen_content: []const u8) void {
        const gd = parseGenerationDefaultsFromJson(gen_content);
        self.gen_temperature = gd.temperature;
        self.gen_top_p = gd.top_p;
        self.gen_top_k = gd.top_k;
        self.gen_min_p = gd.min_p;
        self.gen_enable_thinking = gd.enable_thinking;
        self.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    }

    /// Additive + dedup-guarded, like every terminator merge here.
    pub fn mergeEosTokens(self: *ModelConfig, ids: []const u32) void {
        for (ids) |id| if (!self.isEosToken(id)) self.addEosToken(id);
    }

    pub fn addEosToken(self: *ModelConfig, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    /// Gemma's chat template always ends turns with `<end_of_turn>` (id 106)
    /// and emits `<eos>` (id 1) at sequence end, so both must be stop tokens
    /// for EVERY Gemma family (gemma3 / gemma4). Some
    /// checkpoints declare only a SCALAR `eos_token_id: 1` (e.g. the
    /// abliterated text-only `-lm-` builds) — gating the 106 add on
    /// `num_eos_tokens == 0` then leaves it out and it leaks into output as
    /// repeated `<end_of_turn>`. Merge both ADDITIVELY + dedup-guarded (never
    /// removes a config-declared stop). Same leak class as the Qwen2.5-Coder
    /// `<|im_end|>` merge performed at load time (main.zig / scheduler doLoad).
    pub fn ensureGemmaTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(1)) self.addEosToken(1);
        if (!self.isEosToken(106)) self.addEosToken(106);
    }

    /// MuseGlimmer terminators: <|end_of_text|> = 200001 and <|eot|> = 200008
    /// (the chat template's turn terminator; <|eom|> 200007 is deliberately
    /// NOT an eos — generation continues across channel segments). Additive +
    /// dedup-guarded like ensureGemmaTerminators.
    pub fn ensureMuseTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(200001)) self.addEosToken(200001);
        if (!self.isEosToken(200008)) self.addEosToken(200008);
    }

    /// NoPE layers (muse_glimmer: layer_rope_theta[i] == 0). The released
    /// checkpoint's NoPE layers are exactly its full-attention layers, but the
    /// two facts stay independently parsed — layer_types drives masking,
    /// layer_rope_theta drives rotation.
    pub fn layerSkipsRope(self: *const ModelConfig, layer_idx: u32) bool {
        return layer_idx < 128 and self.layer_no_rope[layer_idx];
    }

    /// Post-attention / post-feedforward norm epsilon (muse_glimmer separates
    /// it from rms_norm_eps; everyone else shares one value).
    pub fn postNormEps(self: *const ModelConfig) f32 {
        return if (self.post_norm_eps > 0) self.post_norm_eps else self.rms_norm_eps;
    }

    /// SDPA softmax scale for the standard dense forward. MuseGlimmer
    /// multiplies the unit-RMS Q by qk_scale_factor on top of the standard
    /// 1/sqrt(head_dim); Gemma 4's QK-norm handles normalization (scale 1.0);
    /// everything else keys on query_pre_attn_scalar.
    pub fn attnScale(self: *const ModelConfig) f32 {
        if (self.qk_scale_factor > 0)
            return self.qk_scale_factor / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        if (std.mem.eql(u8, self.model_type, "gemma4")) return 1.0;
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.query_pre_attn_scalar)));
    }

    /// Hy3 (hy_v3) family terminator: <｜hy_eos:opensource｜> = 120025. Real
    /// MLX conversions (ox-ox 2-bit) ship NO eos in config.json and NO
    /// generation_config.json, so without this merge generation never halts.
    /// Additive + dedup-guarded like ensureGemmaTerminators — never gate a
    /// known chat-terminator on "config provided no eos".
    pub fn ensureHy3Terminators(self: *ModelConfig) void {
        if (!self.isEosToken(120025)) self.addEosToken(120025);
    }

    /// gpt_oss / harmony terminators, merged ADDITIVELY onto whatever the
    /// config declared. A harmony assistant turn can end two ways and the
    /// config names only one of them:
    ///   <|return|> (200002) — the declared eos, ends a normal answer.
    ///   <|call|>   (200012) — ends a TOOL CALL. Never the eos, so a
    ///                         tools request that stops only on eos runs to
    ///                         max_tokens with the call already complete.
    /// <|end|> (200007) is deliberately NOT here: it closes the analysis
    /// channel MID-turn, immediately before `<|start|>assistant<|channel|>final`
    /// opens. Terminating on it would truncate every thinking response to its
    /// reasoning and never emit an answer.
    pub fn ensureGptOssTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(200002)) self.addEosToken(200002);
        if (!self.isEosToken(200012)) self.addEosToken(200012);
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

    /// Whether a chat request that names NO thinking preference should render
    /// with thinking on. Our server always passes `enable_thinking` explicitly,
    /// so a template whose own default is 'on' is silently overridden to off
    /// for every client that omits the field — the vendor's default mode
    /// becomes unreachable without a vendor-specific flag. An EXPLICIT request
    /// value always outranks this (see `server.resolveEnableThinking`); it
    /// only fills a silent request.
    ///
    /// First the checkpoint's OWN declaration
    /// (`generation_config.json` -> `default_chat_template_kwargs.enable_thinking`),
    /// then the family's default (its plugin's `ChatFacts.thinking_default`), which
    /// stays opt-in and only where the vendor documents thinking-on AND the shipped
    /// template agrees — never inferred from "the template mentions enable_thinking".
    pub fn defaultEnableThinking(self: *const ModelConfig, has_tools: bool) bool {
        // The checkpoint's own declared default outranks the family's:
        // it is the model author speaking, not our guess about the family.
        if (self.gen_enable_thinking) |v| return v;
        const plugin: abi.ThinkingDefault = if (self.plugin) |b| b.parsed.chat.thinking_default else .unset;
        return switch (if (plugin != .unset) plugin else self.legacy.thinking) {
            .unset, .off => false,
            .on => true,
            .on_with_tools => has_tools,
        };
    }

    /// Fill still-null sampling recommendations with the FAMILY's documented
    /// upstream defaults. Community re-quants/distills routinely ship no
    /// generation_config.json (a Qwen3.6-35B distill served
    /// to pi resolved omitted fields to the hardcoded 1.0/1.0/off — full
    /// untruncated tail sampling on a 4-bit MoE — and a 16K-token agent turn
    /// degenerated into word salad). Same pattern as the gemma3 head-count
    /// gotcha: when resolution relies on per-arch defaults a minimal
    /// checkpoint may omit, fill them explicitly.
    ///
    /// Deliberately fills ONLY the truncation knobs (top_k/top_p — what keeps
    /// the tail out of the sample space), never temperature: a null temp stays
    /// the neutral 1.0, and explicit request/flag/file values always win
    /// (this runs AFTER generation_config.json parse, nulls only).
    pub fn applyFamilySamplingDefaults(self: *ModelConfig) void {
        const t = self.model_type;
        // Qwen 3.x family only — Qwen2.5's upstream defaults differ (top_p
        // 0.8); never guess numbers the family didn't document.
        const is_qwen = std.mem.eql(u8, t, "qwen3") or
            std.mem.eql(u8, t, "qwen3_moe") or
            std.mem.eql(u8, t, "qwen3_5_moe") or
            std.mem.eql(u8, t, "qwen4_exp") or
            std.mem.eql(u8, t, "qwen3_next");
        const is_gemma = std.mem.eql(u8, t, "gemma3") or
            std.mem.eql(u8, t, "gemma4");
        if (is_qwen) {
            if (self.gen_top_k == null) self.gen_top_k = 20;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        } else if (is_gemma) {
            if (self.gen_top_k == null) self.gen_top_k = 64;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        } else if (std.mem.eql(u8, t, "inkling_mm_model")) {
            // Thinking Machines publishes NO recommendation (no
            // generation_config.json in any Inkling repo; their bundled
            // tooling samples greedily), so top_p 0.95 is OUR choice to cut
            // the untruncated tail — the first real pi agent session
            // ran the hardcoded 1.0/1.0/off and degenerated
            // into duplicated tool calls.
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        }
    }

    /// DeepSeek-V4 releases ship generation_config.json with the WILD
    /// signature (temp 1.0 / top_p 1.0) that their own inference/generate.py
    /// IGNORES — its default is temperature 0.6, the value our converter
    /// writes into our mirrors. External conversions (pipenetwork REAP) copy
    /// the source file verbatim, and an agent CLI that omits temperature then
    /// samples the untruncated tail (pi against REAP37
    /// degenerated into token loops on its FIRST turn). When the reference
    /// implementation deliberately ignores a config field, that field is not
    /// the source of truth (the laguna YaRN class): the EXACT untouched
    /// signature resolves to the reference's default; anything an author
    /// actually tuned is left alone, and request/flag values always win.
    pub fn applyDsv4ReferenceSampling(self: *ModelConfig) void {
        if (!std.mem.eql(u8, self.model_type, "deepseek_v4")) return;
        const t = self.gen_temperature orelse return;
        const p = self.gen_top_p orelse return;
        if (t == 1.0 and p == 1.0) {
            log.info("deepseek_v4: generation_config carries the source's wild 1.0/1.0 signature — resolving to the reference default temp 0.6\n", .{});
            self.gen_temperature = 0.6;
        }
    }

    pub fn isEosToken(self: *const ModelConfig, id: u32) bool {
        for (self.eos_token_ids[0..self.num_eos_tokens]) |eos| {
            if (id == eos) return true;
        }
        return false;
    }

    pub fn eosTokenSlice(self: *const ModelConfig) []const u32 {
        return self.eos_token_ids[0..self.num_eos_tokens];
    }

    /// Everything a load derives from the tokenizer; both load paths call it.
    pub fn applyTokenizer(
        self: *ModelConfig,
        tok: *const tokenizer_mod.Tokenizer,
        eos_token: ?[]const u8,
    ) void {
        // Always merge the chat terminator, even when config.json names an EOS:
        // Qwen2.5-Coder's config says <|endoftext|> but turns end on <|im_end|>.
        if (eos_token) |eos_str| {
            if (tok.special_tokens.get(eos_str)) |eos_id| {
                if (!self.isEosToken(eos_id)) {
                    self.addEosToken(eos_id);
                    log.info("EOS token from tokenizer: {s} (id={d})\n", .{ eos_str, eos_id });
                }
            }
        }
        if (tok.special_tokens.get("<|endoftext|>")) |eot_id| {
            if (!self.isEosToken(eot_id)) self.addEosToken(eot_id);
        }
        // Id 0 can be produced spuriously under long/confusing prompts.
        if (tok.special_tokens.get("<pad>")) |pad_id| {
            if (pad_id > 0 and !self.isEosToken(pad_id)) {
                self.addEosToken(pad_id);
                log.info("Added <pad> as stop token (id={d})\n", .{pad_id});
            }
        }
        self.populateLfm2ImageTokens(tok);
        self.populateMiniCpmImageTokens(tok);
        if (self.radio_vision and self.image_token_id == 0) {
            if (tok.special_tokens.get("<image>")) |id| self.image_token_id = id;
        }
    }

    /// MiniCPM-V's `<image_id>N</image_id>` tags and the digits of N are tokenizer-only ids.
    /// A missing one leaves 0, which the run builder reads as "no image-id tag".
    pub fn populateMiniCpmImageTokens(self: *ModelConfig, tok: *const tokenizer_mod.Tokenizer) void {
        if (!self.minicpm_vision) return;
        if (tok.special_tokens.get("</image_id>")) |id| self.mcv_image_id_end = id;
        if (tok.special_tokens.get("<image_id>")) |id| self.mcv_image_id_start = id;
        // Digits are single tokens in a byte-level BPE vocab (the pre-tokenizer splits every \p{N}).
        for (&self.mcv_digit_ids, 0..) |*slot, digit| {
            if (tok.vocab.get(&[_]u8{@intCast('0' + digit)})) |id| slot.* = id else self.mcv_image_id_start = 0;
        }
    }

    /// LFM2-VL wraps its image-token run in `<|image_start|>`/`<|image_end|>`,
    /// labels every tile with `<|img_row_R_col_C|>` and marks the thumbnail
    /// with `<|img_thumbnail|>`. NONE of those ids appear in config.json — the
    /// tokenizer is the only place they exist — so they are resolved by STRING
    /// at load. A missing marker leaves its id 0,
    /// which every consumer reads as "this checkpoint has no such token".
    pub fn populateLfm2ImageTokens(self: *ModelConfig, tok: *const tokenizer_mod.Tokenizer) void {
        if (!self.lfm2_vision) return;
        if (tok.special_tokens.get("<|image_start|>")) |id| self.boi_token_id = id;
        if (tok.special_tokens.get("<|image_end|>")) |id| self.eoi_token_id = id;
        if (tok.special_tokens.get("<|img_thumbnail|>")) |id| self.lv_thumbnail_token_id = id;
        // The row/col markers are one contiguous block laid out row-major over
        // the max tile grid, so the first one plus (row, col) locates them all.
        if (tok.special_tokens.get("<|img_row_1_col_1|>")) |id| self.lv_row_col_base_id = id;
        if (self.image_token_id == 0) {
            if (tok.special_tokens.get("<image>")) |id| self.image_token_id = id;
        }
        log.info("LFM2-VL image tokens: <image>={d} start={d} end={d} thumbnail={d} row_col_base={d}\n", .{
            self.image_token_id, self.boi_token_id, self.eoi_token_id, self.lv_thumbnail_token_id, self.lv_row_col_base_id,
        });
    }

    /// The n-gram table's file name in the pack: `ngram_table.file`, else `ngram_table.bin`.
    pub fn ngramTableFile(self: *const ModelConfig) []const u8 {
        return if (self.ngram_table_file_len > 0) self.ngram_table_file_buf[0..self.ngram_table_file_len] else "ngram_table.bin";
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

/// A MiMo pack prepared for mlx-stream declares its expert bank (`"expert_streaming": {...}`); the plugin's own parse
/// checks the declaration at load. An upstream MiMo checkpoint declares none and stays on the resident arch.
pub fn declaresExpertStreaming(allocator: std.mem.Allocator, content: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const v = parsed.value.object.get("expert_streaming") orelse return false;
    return v == .object;
}

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    return parseConfigWith(io, allocator, model_dir, true);
}

/// The legacy parse alone, for an engine that runs its own forward over the config's every field (the ANE).
pub fn parseConfigLegacy(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    return parseConfigWith(io, allocator, model_dir, false);
}

fn parseConfigWith(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, served: bool) !ModelConfig {
    const content = (try mlx_gguf.sidecar(io, allocator, model_dir, .config)) orelse blk: {
        const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
        defer allocator.free(path);

        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);

        var read_buf: [4096]u8 = undefined;
        var reader_state = file.reader(io, &read_buf);
        break :blk try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    };
    defer allocator.free(content);

    // A served pack's engine config is the legacy parse with the plugin's binding attached, so every engine read
    // (bills, cache sizing, batching, sampling and thinking defaults) stays as it was; the plugin's facts alone
    // only for a type the legacy parse cannot read.
    const binding = if (served) try plugin_config.bindServed(io, allocator, model_dir, content) else null;
    errdefer if (binding) |b| b.destroy(allocator);
    if (binding) |b| if (b.outside) return plugin_config.adapted(b);
    var config = parseConfigFromJson(allocator, content) catch |e| {
        const b = binding orelse return e;
        log.info("[plugin] {s}: the legacy parse refused {s} ({s}); the engine config is the plugin's facts\n", .{ b.table.name.slice(), b.parsed.shell.model_type.slice(), @errorName(e) });
        return plugin_config.adapted(b);
    };
    // A type the legacy parse knows only as a llama guess is the plugin's facts alone (the layout, the recurrent
    // states), never a full-attention llama.
    if (binding) |b| if (std.mem.eql(u8, config.model_type, "unknown")) {
        config.deinit(allocator);
        return plugin_config.adapted(b);
    };
    errdefer config.deinit(allocator);
    if (config.isDsv41()) config.pack_dir = try allocator.dupe(u8, model_dir);
    if (config.isMimo() and declaresExpertStreaming(allocator, content)) {
        config.pack_dir = try allocator.dupe(u8, model_dir);
        config.expert_bank = true;
    }
    if (config.isGlm5() and declaresExpertStreaming(allocator, content)) {
        config.pack_dir = try allocator.dupe(u8, model_dir);
        config.expert_bank = true;
    }
    if (std.mem.eql(u8, config.model_type, "glm_moe_dsa")) {
        // No in-tree arch serves the 744B: only an mlx-stream pack (the trunk beside an experts.bin bank) loads.
        const bank = try std.fmt.allocPrint(allocator, "{s}/experts.bin", .{model_dir});
        defer allocator.free(bank);
        std.Io.Dir.accessAbsolute(io, bank, .{}) catch {
            log.err("glm_moe_dsa: {s} has no experts.bin; GLM-5.3 runs only from an mlx-stream pack (the trunk beside an experts.bin bank)\n", .{model_dir});
            return error.GlmMoeDsaNeedsExpertBank;
        };
        config.pack_dir = try allocator.dupe(u8, model_dir);
        config.glm_stream = true;
    }
    if (config.isQwen4()) {
        if (config.ngram_table_file_len > 0)
            config.ngram_table_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, config.ngramTableFile() })
        else
            config.ngram_table_path = try std.fmt.allocPrint(allocator, "{s}/ngram_table.bin", .{model_dir});
        if (try qwen4_exp.inspectEmbedded(model_dir, try qwen4EmbeddedSpec(&config))) |info| {
            config.embedded_ple_payload_bytes = info.payload_bytes;
        }
    }

    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    const gen_sidecar: ?[]u8 = mlx_gguf.sidecar(io, allocator, model_dir, .generation_config) catch null;
    if (gen_sidecar) |gen_content| {
        defer allocator.free(gen_content);
        config.applyGenerationDefaults(gen_content);
    } else if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            config.applyGenerationDefaults(gen_content);
        } else |_| {}
    } else |_| {}
    // Pooling (issue #116), priority: explicit config.json `pooling_mode`
    // (already parsed) > the ST `1_Pooling/config.json` sidecar > the
    // known-family name fallback. A sidecar declaring only unsupported modes
    // fails the load here — explicitly, never a silent mean-pool.
    if (config.pooling_mode == null) {
        const pool_path = try std.fmt.allocPrint(allocator, "{s}/1_Pooling/config.json", .{model_dir});
        defer allocator.free(pool_path);
        if (std.Io.Dir.openFileAbsolute(io, pool_path, .{})) |pool_file| {
            defer pool_file.close(io);
            var pool_buf: [4096]u8 = undefined;
            var pool_reader = pool_file.reader(io, &pool_buf);
            if (pool_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |pool_content| {
                defer allocator.free(pool_content);
                config.pooling_mode = try parsePoolingSidecar(pool_content);
                if (config.pooling_mode) |m|
                    log.info("[embed] pooling from 1_Pooling/config.json: {s}\n", .{@tagName(m)});
            } else |_| {}
        } else |_| {}
    }
    if (config.pooling_mode == null) {
        if (poolingFromDirName(std.fs.path.basename(model_dir), config.model_type)) |m| {
            config.pooling_mode = m;
            log.info("[embed] pooling inferred from checkpoint name: {s}\n", .{@tagName(m)});
        }
    }

    // Community re-quants often ship NO generation_config.json; fill the
    // still-null truncation knobs with the family's documented defaults so
    // omitted-field resolution never bottoms out at untruncated sampling.
    config.applyFamilySamplingDefaults();
    // ... and a dsv4 generation_config carrying the source's verbatim wild
    // signature resolves to the reference implementation's own default.
    config.applyDsv4ReferenceSampling();

    // Qwen image sizing is processor metadata rather than an architecture
    // constant. Prefer processor_config.json and fill any missing field from
    // the older preprocessor_config.json layout.
    if (config.qwen_vision or config.muse_vision or config.glm_vision) {
        var vision_defaults = VisionProcessorDefaults{};
        const processor_files = [_][]const u8{
            "processor_config.json",
            "preprocessor_config.json",
        };
        for (processor_files) |name| {
            const processor_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, name });
            defer allocator.free(processor_path);
            if (std.Io.Dir.openFileAbsolute(io, processor_path, .{})) |processor_file| {
                defer processor_file.close(io);
                var processor_buf: [4096]u8 = undefined;
                var processor_reader = processor_file.reader(io, &processor_buf);
                if (processor_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |processor_content| {
                    defer allocator.free(processor_content);
                    const parsed_defaults = parseVisionProcessorDefaultsFromJson(processor_content);
                    if (vision_defaults.min_pixels == null)
                        vision_defaults.min_pixels = parsed_defaults.min_pixels;
                    if (vision_defaults.max_pixels == null)
                        vision_defaults.max_pixels = parsed_defaults.max_pixels;
                    if (vision_defaults.max_image_tokens == null)
                        vision_defaults.max_image_tokens = parsed_defaults.max_image_tokens;
                    if (vision_defaults.min_image_tokens == null)
                        vision_defaults.min_image_tokens = parsed_defaults.min_image_tokens;
                } else |_| {}
            } else |_| {}
        }
        if (vision_defaults.min_pixels != null and
            vision_defaults.max_pixels != null and
            vision_defaults.min_pixels.? > vision_defaults.max_pixels.?)
        {
            vision_defaults = .{};
        }
        config.qv_min_pixels = vision_defaults.min_pixels orelse 0;
        config.qv_max_pixels = vision_defaults.max_pixels orelse 0;
        config.mv_max_image_tokens = vision_defaults.max_image_tokens orelse MUSE_MAX_IMAGE_TOKENS;
        if (config.glm_vision) {
            const lo = vision_defaults.min_image_tokens orelse config.glmv_min_image_tokens;
            const hi = vision_defaults.max_image_tokens orelse config.glmv_max_image_tokens;
            // The processor raises on an inverted budget; keep its defaults instead.
            if (lo <= hi) {
                config.glmv_min_image_tokens = lo;
                config.glmv_max_image_tokens = hi;
            }
        }
    }

    if (config.minicpm_vision) {
        const path = try std.fmt.allocPrint(allocator, "{s}/preprocessor_config.json", .{model_dir});
        defer allocator.free(path);
        if (std.Io.Dir.openFileAbsolute(io, path, .{})) |file| {
            defer file.close(io);
            var buf: [4096]u8 = undefined;
            var reader = file.reader(io, &buf);
            if (reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |json| {
                defer allocator.free(json);
                config.mcv_max_slice_nums = parseMaxSliceNumsFromJson(json) orelse config.mcv_max_slice_nums;
            } else |_| {}
        } else |_| {}
    }

    if (config.gv_budgeted) {
        const processor_path = try std.fmt.allocPrint(allocator, "{s}/processor_config.json", .{model_dir});
        defer allocator.free(processor_path);
        if (std.Io.Dir.openFileAbsolute(io, processor_path, .{})) |processor_file| {
            defer processor_file.close(io);
            var processor_buf: [4096]u8 = undefined;
            var processor_reader = processor_file.reader(io, &processor_buf);
            if (processor_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |processor_content| {
                defer allocator.free(processor_content);
                const budgets = parseSoftTokenBudgetsFromJson(allocator, processor_content);
                config.gv_image_tokens = budgets.image orelse config.gv_image_tokens;
                config.gv_video_tokens = budgets.video orelse config.gv_video_tokens;
                config.gv_video_frames = budgets.max_frames orelse config.gv_video_frames;
            } else |_| {}
        } else |_| {}
        log.info("[vision] EmbeddingGemma 2 processors: {d} soft tokens per image, {d} per video frame, at most {d} frames\n", .{ config.gv_image_tokens, config.gv_video_tokens, config.gv_video_frames });
    }

    config.plugin = binding;
    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    /// `default_chat_template_kwargs.enable_thinking` — the checkpoint's own
    /// thinking default. null when absent or not a bool.
    enable_thinking: ?bool = null,
    /// `eos_token_id` (scalar or list): HF stops generation on these, and a
    /// checkpoint may name the chat terminator ONLY here (K2-Horizon's
    /// `<|ifm|im_end|>` rides beside config.json's `<|ifm|endoftext|>`).
    eos_token_ids: [8]u32 = @splat(0),
    num_eos: usize = 0,
};

/// Image-area limits parsed from a Qwen processor configuration.
pub const VisionProcessorDefaults = struct {
    min_pixels: ?u32 = null,
    max_pixels: ?u32 = null,
    /// Muse and GLM: the cap is on MERGED tokens, not pixels.
    max_image_tokens: ?u32 = null,
    /// GLM: the floor a small image is upscaled to.
    min_image_tokens: ?u32 = null,
};

fn positiveJsonU32(value: ?std.json.Value) ?u32 {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

/// Parse both processor layouts used by Qwen checkpoints:
/// `image_processor.{min_pixels,max_pixels}` and
/// `size.{shortest_edge,longest_edge}`.
pub fn parseVisionProcessorDefaultsFromJson(content: []const u8) VisionProcessorDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};

    const root = parsed.value.object;
    const processor = if (root.get("image_processor")) |value|
        if (value == .object) value.object else root
    else
        root;

    var defaults = VisionProcessorDefaults{
        .min_pixels = positiveJsonU32(processor.get("min_pixels")),
        .max_pixels = positiveJsonU32(processor.get("max_pixels")),
        .max_image_tokens = positiveJsonU32(processor.get("max_image_tokens")),
        .min_image_tokens = positiveJsonU32(processor.get("min_image_tokens")),
    };
    if (processor.get("size")) |value| {
        if (value == .object) {
            if (defaults.min_pixels == null)
                defaults.min_pixels = positiveJsonU32(value.object.get("shortest_edge"));
            if (defaults.max_pixels == null)
                defaults.max_pixels = positiveJsonU32(value.object.get("longest_edge"));
        }
    }
    if (defaults.min_pixels != null and
        defaults.max_pixels != null and
        defaults.min_pixels.? > defaults.max_pixels.?)
    {
        return .{};
    }
    return defaults;
}

pub const SoftTokenBudgets = struct { image: ?u32 = null, video: ?u32 = null, max_frames: ?u32 = null };

/// The soft-token budget each EmbeddingGemma 2 processor accepts; any other value makes the processor raise.
fn validSoftTokenBudget(value: ?std.json.Value) ?u32 {
    const n = positiveJsonU32(value) orelse return null;
    return switch (n) {
        70, 140, 280, 560, 1120 => n,
        else => null,
    };
}

/// `processor_config.json` of an EmbeddingGemma 2 checkpoint: the image and video processors' budgets and the
/// video frame cap. A missing or unusable field stays null so the processors' defaults hold.
pub fn parseSoftTokenBudgetsFromJson(allocator: std.mem.Allocator, content: []const u8) SoftTokenBudgets {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;
    var out = SoftTokenBudgets{};
    if (root.get("image_processor")) |v| {
        if (v == .object) out.image = validSoftTokenBudget(v.object.get("max_soft_tokens"));
    }
    if (root.get("video_processor")) |v| {
        if (v == .object) {
            out.video = validSoftTokenBudget(v.object.get("max_soft_tokens"));
            out.max_frames = positiveJsonU32(v.object.get("max_frames"));
        }
    }
    return out;
}

/// Pure parser for generation_config.json content. Total: malformed JSON or
/// out-of-range values yield nulls — a corrupt config must never pin
/// sampling to an extreme. (`do_sample` is deliberately ignored: HF uses it
/// for greedy-vs-sample mode selection, which the request's own temperature
/// already expresses.)
pub fn parseGenerationDefaultsFromJson(content: []const u8) GenerationDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;

    var gd = GenerationDefaults{};
    if (root.get("temperature")) |v| {
        const t: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (t) |tv| {
            if (tv >= 0.0 and tv <= 2.0) gd.temperature = tv;
        }
    }
    if (root.get("top_p")) |v| {
        const p: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (p) |pv| {
            if (pv > 0.0 and pv <= 1.0) gd.top_p = pv;
        }
    }
    if (root.get("top_k")) |v| {
        switch (v) {
            .integer => |i| if (i > 0 and i <= 1000) {
                gd.top_k = @intCast(i);
            },
            else => {},
        }
    }
    if (root.get("min_p")) |v| {
        const m: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (m) |mv| {
            if (mv >= 0.0 and mv <= 1.0) gd.min_p = mv;
        }
    }
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| if (i >= 0) {
                gd.eos_token_ids[0] = @intCast(i);
                gd.num_eos = 1;
            },
            .array => |arr| for (arr.items) |item| {
                if (item == .integer and item.integer >= 0 and gd.num_eos < gd.eos_token_ids.len) {
                    gd.eos_token_ids[gd.num_eos] = @intCast(item.integer);
                    gd.num_eos += 1;
                }
            },
            else => {},
        }
    }
    // The checkpoint's own chat-template kwargs. Only a real bool counts —
    // anything else leaves the field null and the arch default in charge.
    if (root.get("default_chat_template_kwargs")) |v| {
        if (v == .object) {
            if (v.object.get("enable_thinking")) |et| {
                if (et == .bool) gd.enable_thinking = et.bool;
            }
        }
    }
    return gd;
}

/// One qwen4_exp integer bound, read strictly: wrong-typed or negative refuses.
fn qwen4ConfigU64(cfg_obj: std.json.ObjectMap, key: []const u8) !?u64 {
    const v = jsonField(cfg_obj, key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

fn qwen4ConfigU32(cfg_obj: std.json.ObjectMap, key: []const u8) !?u32 {
    const v = try qwen4ConfigU64(cfg_obj, key) orelse return null;
    if (v > std.math.maxInt(u32)) return error.InvalidQwen4ConfigField;
    return @intCast(v);
}

/// A fully EXL3 pack's top-level keys (`exl3_pack_keys`): the trunk's format block and, in that pack only, the n-gram
/// table's file.
fn parseQwen4FullExl3(root: std.json.ObjectMap, config: *ModelConfig) !void {
    config.trunk_format = try exl3_pack_keys.hasTrunkQuant(root);
    if (!config.trunk_format) return;
    if (try exl3_pack_keys.ngramTableFile(root)) |f| {
        @memcpy(config.ngram_table_file_buf[0..f.len], f);
        config.ngram_table_file_len = @intCast(f.len);
    }
}

pub var gdn_state_f32_override: ?bool = null; // test seam
var gdn_state_f32_env: ?bool = null;

/// qwen4_exp's GatedDeltaNet state and decay gate in f32, as its reference keeps them.
/// MLX_SERVE_GDN_STATE_F32=0 restores bf16 (half the state memory, the old numerics).
fn gdnStateF32Enabled() bool {
    if (gdn_state_f32_override) |v| return v;
    if (gdn_state_f32_env) |v| return v;
    const raw = env.get(.MLX_SERVE_GDN_STATE_F32);
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    gdn_state_f32_env = on;
    return on;
}

/// Range-check every qwen4_exp bound the forward indexes a fixed array with or divides by.
/// Names travel to the client as "Model load failed: <name>".
fn validateQwen4Config(config: *const ModelConfig) !void {
    // `NgramHash.multipliers` is [MAX_NGRAM_SIZE]i64; `ple_prev` is written ngram_size-1 deep.
    if (config.ngram_size < 2 or config.ngram_size > qwen4_exp.MAX_NGRAM_SIZE) {
        return error.InvalidQwen4NgramSize;
    }
    // `vocab`/`offsets` are [MAX_HEADS]i64, written n_heads deep.
    if (config.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (@as(u64, config.ngram_size - 1) * config.heads_per_ngram > qwen4_exp.MAX_HEADS) {
        return error.InvalidQwen4NgramHeads;
    }
    if (config.ngram_vocab_divisor == 0 or config.ngram_vocab_base < 2) {
        return error.InvalidQwen4NgramVocab;
    }
    if (config.ple_embed_dim == 0 or config.ple_embed_dim % ((config.ngram_size - 1) * config.heads_per_ngram) != 0 or config.split_ngram_parts == 0) {
        return error.InvalidQwen4PleGeometry;
    }
    // The forward divides kv by the ratio and selects `budget / ratio` blocks.
    if (config.indexer_n_heads > 0) {
        if (config.indexer_head_dim == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_budget < config.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    const layers: i64 = if (config.dist_span) |sp| sp.total else config.num_hidden_layers;
    if (config.ple_layer_idx < 0 or config.ple_layer_idx >= layers) {
        return error.InvalidQwen4PleLayer;
    }
}

pub fn qwen4EmbeddedSpec(config: *const ModelConfig) !qwen4_exp.EmbeddedSpec {
    const hash = try qwen4_exp.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, config.ngram_eos);
    if (config.ple_embed_dim == 0 or config.ple_embed_dim % hash.n_heads != 0) return error.InvalidQwen4PleGeometry;
    if (config.ple_layer_idx < 0) return error.InvalidQwen4PleLayer;
    return .{ .rows = hash.total_rows, .dim = config.ple_embed_dim / hash.n_heads, .shards = config.split_ngram_parts, .layer_index = @intCast(config.ple_layer_idx) };
}

/// True when the layer loop installed the PLE on exactly the layer the config names. A negative
/// index asks for no PLE (the MTP head's own layer) and is satisfied by a loop that installed none.
pub fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
/// Qwen3-VL-family vision + M-RoPE fields, shared by the qwen3_5 and
/// qwen4_exp arms (same `vision_config` keys, `rope_parameters.mrope_*`,
/// vision token ids). The generic vision_config block already set
/// `has_vision`; this reads Qwen's own keys into `qv_*`.
/// One block for every packed module; mixed blocks are not a layout we serve.
fn prismHadamardBlock(root: std.json.ObjectMap) !u32 {
    const modules = (try jsonValue(.array, root.get("modules") orelse return error.UnsupportedHadamardLayout)).items;
    var block: u32 = 0;
    for (modules) |m| {
        const module = try jsonValue(.object, m);
        const b = try jsonU32(module.get("block") orelse return error.UnsupportedHadamardLayout);
        if (b <= 0 or (block != 0 and b != block)) return error.UnsupportedHadamardLayout;
        block = b;
    }
    if (block == 0) return error.UnsupportedHadamardLayout;
    return block;
}

/// The Qwen-VL `vision_config` geometry keys into `qv_*`; GLM-5.3-Flash's tower names them the same.
fn parseQvGeometry(config: *ModelConfig, vc: std.json.ObjectMap) !void {
    if (jsonField(vc, "depth")) |v| {
        if (v == .integer) config.qv_depth = try jsonU32(v);
    }
    if (jsonField(vc, "hidden_size")) |v| {
        if (v == .integer) config.qv_hidden = try jsonU32(v);
    }
    if (jsonField(vc, "num_heads")) |v| {
        if (v == .integer) config.qv_heads = try jsonU32(v);
    }
    if (jsonField(vc, "intermediate_size")) |v| {
        if (v == .integer) config.qv_intermediate = try jsonU32(v);
    }
    if (jsonField(vc, "patch_size")) |v| {
        if (v == .integer) config.qv_patch = try jsonU32(v);
    }
    if (jsonField(vc, "temporal_patch_size")) |v| {
        if (v == .integer) config.qv_temporal_patch = try jsonU32(v);
    }
    if (jsonField(vc, "spatial_merge_size")) |v| {
        if (v == .integer) config.qv_merge = try jsonU32(v);
    }
    if (jsonField(vc, "num_position_embeddings")) |v| {
        if (v == .integer) config.qv_num_pos_emb = try jsonU32(v);
    }
    if (jsonField(vc, "out_hidden_size")) |v| {
        if (v == .integer) config.qv_out_hidden = try jsonU32(v);
    }
    if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
    if (config.qv_out_hidden == 0) config.qv_out_hidden = config.hidden_size;
}

fn parseQwenVisionFields(config: *ModelConfig, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    if (jsonField(root, "vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.qwen_vision = true;
            try parseQvGeometry(config, vc_val.object);
        }
    }
    // Interleaved M-RoPE sections (text_config.rope_parameters). rope_theta /
    // partial_rotary_factor already parsed in the generic rope block above.
    if (jsonField(cfg_obj, "rope_parameters")) |rp| {
        if (rp == .object) {
            if (jsonField(rp.object, "mrope_interleaved")) |v| {
                if (v == .bool) config.mrope_interleaved = v.bool;
            }
            if (jsonField(rp.object, "mrope_section")) |v| {
                if (v == .array) {
                    for (v.array.items, 0..) |item, i| {
                        if (i >= 3) break;
                        if (item == .integer) config.mrope_section[i] = try jsonU32(item);
                    }
                }
            }
        }
    }
    // Qwen vision token ids (top-level).
    if (jsonField(root, "video_token_id")) |v| {
        if (v == .integer) config.video_token_id = try jsonU32(v);
    }
    if (jsonField(root, "vision_start_token_id")) |v| {
        if (v == .integer) config.vision_start_token_id = try jsonU32(v);
    }
    if (jsonField(root, "vision_end_token_id")) |v| {
        if (v == .integer) config.vision_end_token_id = try jsonU32(v);
    }
}

/// MiniCPM-V 4.7's tower options (`src/plugins/qwen3_5/minicpm_vision.zig`). The generic `vision_config` block has
/// already read the SigLIP2 geometry; this adds the window merger, the downsample mode and the
/// token ids config.json carries. One round of 2x2 merging in each merger is the only shape served.
fn parseMiniCpmVision(config: *ModelConfig, root: std.json.ObjectMap) !void {
    const vc_val = jsonField(root, "vision_config") orelse return;
    if (vc_val != .object) return;
    const vc = vc_val.object;
    config.minicpm_vision = true;
    // Video frames are not served: no `<|video_pad|>` placeholder, no "video" modality.
    config.video_token_id = 0;
    if (jsonField(root, "insert_layer_id")) |v| config.mcv_insert_layer = try jsonU32(v);
    if (jsonField(vc, "layer_norm_eps")) |v| config.mcv_eps = try jsonFloat(v);
    if (config.mcv_insert_layer >= config.vision_num_layers) return error.InvalidConfigField;
    if (jsonField(root, "downsample_mode")) |v| {
        const mode = try jsonValue(.string, v);
        if (std.mem.eql(u8, mode, "4x")) {
            config.mcv_downsample_4x = true;
        } else if (!std.mem.eql(u8, mode, "16x")) return error.InvalidConfigField;
    }
    if (jsonField(root, "merger_times")) |v| {
        if (try jsonU32(v) != 1) return error.InvalidConfigField;
    }
    for ([_]std.json.ObjectMap{ root, vc }, [_][]const u8{ "merge_kernel_size", "window_kernel_size" }) |obj, key| {
        const v = jsonField(obj, key) orelse continue;
        const kernel = try jsonValue(.array, v);
        if (kernel.items.len != 2 or try jsonU32(kernel.items[0]) != 2 or try jsonU32(kernel.items[1]) != 2) return error.InvalidConfigField;
    }
    const ids = .{
        .{ "image_start_id", &config.mcv_image_start_id },
        .{ "image_end_id", &config.mcv_image_end_id },
        .{ "slice_start_id", &config.mcv_slice_start_id },
        .{ "slice_end_id", &config.mcv_slice_end_id },
        .{ "newline_id", &config.mcv_newline_id },
    };
    inline for (ids) |entry| {
        if (jsonField(root, entry[0])) |v| entry[1].* = try jsonU32(v);
    }
}

/// `max_slice_nums` of the checkpoint's `preprocessor_config.json` (null when absent or 0).
pub fn parseMaxSliceNumsFromJson(content: []const u8) ?u32 {
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    return positiveJsonU32(parsed.value.object.get("max_slice_nums"));
}

/// Flat HF `rope_parameters` carrying `rope_type: "yarn"` — the YaRN context
/// extension, i.e. exactly what vLLM's `--hf-overrides` recipe for Qwen3.5
/// writes:
///
///   {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
///     "original_max_position_embeddings": 262144, "rope_theta": 10000000,
///     "partial_rotary_factor": 0.25, "mrope_interleaved": true,
///     "mrope_section": [11,11,10]}}}
///
/// `factor` may be omitted, in which case HF derives it from
/// `max_position_embeddings / original_max_position_embeddings` (as vLLM's
/// `_get_and_verify_max_len` does). `attention_factor` is HF's key and
/// REPLACES the computed mscale; `attn_factor` is vLLM's and MULTIPLIES
/// `yarnMscale(factor)`. Neither is present in a vendor config, and per HF's
/// default the mscale is then COMPUTED as 0.1·ln(factor)+1 — the value the
/// scaling was calibrated with. Nested per-layer-type `rope_parameters`
/// (laguna/gemma4) never reach here: they have no top-level `rope_type`.
fn parseYarnRopeParameters(config: *ModelConfig, cfg_obj: std.json.ObjectMap) !void {
    const rp_val = jsonField(cfg_obj, "rope_parameters") orelse return;
    if (rp_val != .object) return;
    const rp = rp_val.object;
    const rt = jsonField(rp, "rope_type") orelse return;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "yarn"))) return;

    if (jsonField(rp, "original_max_position_embeddings")) |v| {
        if (v == .integer) config.yarn_orig_max_pos = try jsonU32(v);
    }
    // A YaRN block with no window to scale FROM is not a scaling we can
    // reproduce: the ramp bounds (and so every mid-band frequency) come from
    // it. Refuse the load rather than serve a silently-wrong rotation.
    if (config.yarn_orig_max_pos == 0) return error.YarnRopeNeedsOriginalMaxPos;
    if (jsonField(rp, "factor")) |v| config.yarn_factor = try jsonFloat(v);
    if (jsonField(rp, "beta_fast")) |v| config.yarn_beta_fast = try jsonFloat(v);
    if (jsonField(rp, "beta_slow")) |v| config.yarn_beta_slow = try jsonFloat(v);
    if (jsonField(rp, "truncate")) |v| {
        if (v == .bool) config.yarn_truncate = v.bool;
    }
    if (config.yarn_factor <= 0.0) return error.InvalidRopeScalingFactor;
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block leaves it out (the config then only states the window).
    if (jsonField(rp, "factor") == null and config.max_position_embeddings > config.yarn_orig_max_pos) {
        config.yarn_factor = @as(f32, @floatFromInt(config.max_position_embeddings)) /
            @as(f32, @floatFromInt(config.yarn_orig_max_pos));
    }
    // HF `attention_factor` replaces; vLLM `attn_factor` multiplies the
    // computed 0.1·ln(factor)+1. Both present → HF wins.
    if (jsonField(rp, "attention_factor")) |v| {
        config.yarn_attention_factor = try jsonFloat(v);
    } else if (jsonField(rp, "attn_factor")) |v| {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor) * try jsonFloat(v);
    } else {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor);
    }
    if (!std.math.isFinite(config.yarn_attention_factor)) return error.InvalidConfigField;
    config.rope_yarn = true;
}

/// HF's default YaRN mscale (`attention_factor`) for a scaling `factor`.
fn yarnMscale(factor: f32) f32 {
    if (factor <= 1.0) return 1.0;
    return 0.1 * @log(@as(f32, factor)) + 1.0;
}

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ModelCliState = struct {
    /// Launch-time JSON deep-merged into every `config.json` before it is parsed,
    /// set once from `--config-overrides`. vLLM's `--hf-overrides` analogue: the
    /// only way to re-shape a checkpoint's declared geometry — most often to scale
    /// its rope and widen the context — without editing the model directory, and
    /// therefore the way to A/B a scaling experiment on identical weights.
    config_overrides: ?[]const u8 = null,
};
pub const modelCli_shared = @import("../util/shared.zig").Shared(ModelCliState, "model_cli", .{});
inline fn modelCli() *ModelCliState {
    return modelCli_shared.get();
}

pub fn setConfigOverrides(raw: ?[]const u8) void {
    modelCli().config_overrides = raw;
}

pub fn getConfigOverrides() ?[]const u8 {
    return modelCli().config_overrides;
}

pub const mergeConfigJson = @import("merge.zig").mergeConfigJson;

/// EmbeddingGemma 2's context window as its model card states it. Kept in sync with
/// `model_discovery.parseStubMeta`, which cannot import this file.
pub const EMBEDDING_GEMMA2_WINDOW: u32 = 8192;

/// `per_layer_config` gives the full-attention layers a head width and KV head count of their own. The loader
/// keeps one such geometry (`global_head_dim`, `num_global_key_value_heads`), so entries that name a sliding
/// layer, disagree with each other or leave a full layer out are refused. An absent block is HF's default.
fn parseEmbeddingGemma2Geometry(cfg_obj: std.json.ObjectMap, config: *ModelConfig) !void {
    if (config.num_hidden_layers > config.layer_is_global.len) return error.UnsupportedLayerGeometry;
    config.global_head_dim = 512;
    config.num_global_key_value_heads = 1;
    const block = jsonField(cfg_obj, "per_layer_config") orelse return;
    var listed: @TypeOf(config.layer_is_global) = @splat(false);
    var first = true;
    var it = (try jsonValue(.object, block)).iterator();
    while (it.next()) |e| {
        const layer = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch return error.UnsupportedLayerGeometry;
        if (layer >= config.num_hidden_layers or !config.isGlobalLayer(layer)) return error.UnsupportedLayerGeometry;
        const entry = try jsonValue(.object, e.value_ptr.*);
        const head_dim = try jsonU32(jsonField(entry, "head_dim") orelse return error.UnsupportedLayerGeometry);
        const kv_heads = try jsonU32(jsonField(entry, "num_key_value_heads") orelse return error.UnsupportedLayerGeometry);
        if (!first and (head_dim != config.global_head_dim or kv_heads != config.num_global_key_value_heads)) return error.UnsupportedLayerGeometry;
        config.global_head_dim = head_dim;
        config.num_global_key_value_heads = kv_heads;
        listed[layer] = true;
        first = false;
    }
    for (0..config.num_hidden_layers) |layer| {
        if (!first and config.isGlobalLayer(@intCast(layer)) and !listed[layer]) return error.UnsupportedLayerGeometry;
    }
}

/// The MiMo release (FP8 trunk, per-expert MXFP4 bytes) is not a pack; mlx packs
/// carry an mlx-style quantization_config with no `quant_method`.
fn refuseUnconvertedMimo(cfg_obj: std.json.ObjectMap) !void {
    const qc = jsonField(cfg_obj, "quantization_config") orelse return;
    if (qc == .object and jsonField(qc.object, "quant_method") != null) {
        log.err("mimo_v2: this is the original release; convert it first with tests/convert_mimo_v2.py\n", .{});
        return error.UnconvertedMimoCheckpoint;
    }
}

/// A Sushi pack's `expert_quant` makes its routed experts EXL3 banks.
fn parseSushiExperts(root: std.json.ObjectMap, config: *ModelConfig) !void {
    if (jsonField(root, "expert_quant") == null) return;
    config.exl3 = try exl3.parse(root, config.num_experts_per_tok);
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    // The launch-time overrides apply to EVERY parse (primary load, on-demand
    // load, discovery stubs), so the advertised context and the loaded model
    // can never disagree about what window the checkpoint has.
    const merged: ?[]const u8 = if (modelCli().config_overrides) |ov| blk: {
        break :blk try mergeConfigJson(allocator, content, ov);
    } else null;
    defer if (merged) |m| allocator.free(m);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, merged orelse content, .{});
    defer parsed.deinit();

    const outer = try jsonValue(.object, parsed.value);
    const root = qwenAsrThinker(outer) orelse outer;
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    var model_type: []const u8 = if (jsonField(root, "model_type")) |v| try jsonValue(.string, v) else "gemma3";

    // Determine which object to read config from: text_config (nested) or root (flat).
    // Nemotron omni wraps the language tower in llm_config; the outer object has
    // no hidden size or layer list.
    var cfg_obj: std.json.ObjectMap = if (jsonField(root, "text_config")) |tc_val| try jsonValue(.object, tc_val) else root;
    if (std.mem.eql(u8, model_type, "nemotron_h_omni")) {
        if (jsonField(root, "llm_config")) |lc| {
            if (lc == .object) {
                cfg_obj = lc.object;
                if (jsonField(lc.object, "model_type")) |mt| {
                    if (mt == .string) model_type = mt.string;
                }
            }
        }
    }

    // Parse common fields
    if (jsonField(cfg_obj, "vocab_size")) |v| config.vocab_size = try jsonU32(v);
    if (jsonField(cfg_obj, "hidden_size")) |v| config.hidden_size = try jsonU32(v);
    if (jsonField(cfg_obj, "intermediate_size")) |v| {
        config.intermediate_size = try jsonU32(v);
        config.intermediate_size_declared = true;
    }
    if (jsonField(cfg_obj, "num_hidden_layers")) |v| config.num_hidden_layers = try jsonU32(v);
    if (jsonField(cfg_obj, "num_attention_heads")) |v| config.num_attention_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "num_key_value_heads")) |v| config.num_key_value_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "head_dim")) |v| config.head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "max_position_embeddings")) |v| config.max_position_embeddings = try jsonU32(v);
    if (jsonField(cfg_obj, "rms_norm_eps")) |v| config.rms_norm_eps = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
    if (jsonField(cfg_obj, "query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = try jsonU32(v);

    // MoE fields (guard against JSON null values)
    if (jsonField(cfg_obj, "num_experts")) |v| {
        if (v == .integer) config.num_experts = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_experts_per_tok")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "top_k_experts")) |v| {
        if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "moe_intermediate_size")) |v| {
        if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "shared_expert_intermediate_size")) |v| {
        if (v == .integer) config.shared_expert_intermediate_size = try jsonU32(v);
    }

    // Linear attention (GatedDeltaNet) fields
    if (jsonField(cfg_obj, "linear_num_key_heads")) |v| config.linear_num_key_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_num_value_heads")) |v| config.linear_num_value_heads = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_key_head_dim")) |v| config.linear_key_head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_value_head_dim")) |v| config.linear_value_head_dim = try jsonU32(v);
    if (jsonField(cfg_obj, "linear_conv_kernel_dim")) |v| config.linear_conv_kernel_dim = try jsonU32(v);

    // Hybrid attention
    if (jsonField(cfg_obj, "full_attention_interval")) |v| config.full_attention_interval = try jsonU32(v);
    if (jsonField(cfg_obj, "attn_output_gate")) |v| {
        if (v == .bool) config.attn_output_gate = v.bool;
    }

    // Bidirectional-attention embedding models (EmbeddingGemma): a decoder
    // arch trained as an encoder. Routes to the encoder forward + the
    // /v1/embeddings surface; chat surfaces reject it.
    if (jsonField(cfg_obj, "use_bidirectional_attention")) |v| {
        if (v == .bool and v.bool) {
            config.use_bidirectional_attention = true;
            config.is_encoder_only = true;
        }
    }
    // Explicit pooling contract (issue #116): "mean" | "cls" | "last_token" in
    // config.json marks a checkpoint as an embedding model and picks the pool
    // op. An unknown value is a parse error, never a silent mean-pool —
    // wrong-semantics vectors are harder to detect than a refused load.
    if (jsonField(root, "pooling_mode")) |v| {
        if (v == .string) {
            config.pooling_mode = PoolingMode.fromString(v.string) orelse
                return error.UnsupportedPoolingMode;
        }
    }
    if (jsonField(cfg_obj, "bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) config.bos_token_id = try jsonU32(v);
    }

    // Rope parameters (nested for Qwen3.5)
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = try jsonU32(v);
            config.has_sliding_window = true;
        }
    }
    if (jsonField(cfg_obj, "sliding_window_pattern")) |v| config.sliding_window_pattern = try jsonU32(v);

    // Gemma-specific: dual RoPE bases
    if (jsonField(cfg_obj, "rope_local_base_freq")) |v| config.rope_local_base_freq = try jsonFloat(v);
    if (jsonField(cfg_obj, "rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (jsonField(rs_val.object, "factor")) |v| config.rope_scaling_factor = try jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (jsonField(cfg_obj, "layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
            }
        }
    }

    // Gemma 4: dual head dimensions and KV sharing
    if (jsonField(cfg_obj, "global_head_dim")) |v| {
        if (v == .integer) config.global_head_dim = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_global_key_value_heads")) |v| {
        if (v == .integer) config.num_global_key_value_heads = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "num_kv_shared_layers")) |v| {
        if (v == .integer) config.num_kv_shared_layers = try jsonU32(v);
    }
    if (jsonField(cfg_obj, "attention_k_eq_v")) |v| {
        if (v == .bool) config.attention_k_eq_v = v.bool;
    }
    if (jsonField(cfg_obj, "final_logit_softcapping")) |v| {
        config.final_logit_softcapping = try jsonFloat(v);
    }
    if (jsonField(cfg_obj, "hidden_size_per_layer_input")) |v| {
        if (v == .integer) config.hidden_size_per_layer_input = try jsonU32(v);
    }

    // Gemma 4: nested rope_parameters with per-attention-type config
    if (jsonField(cfg_obj, "rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            // Gemma 4 style: { "full_attention": {...}, "sliding_attention": {...} }
            if (jsonField(rp_val.object, "full_attention")) |fa| {
                if (fa == .object) {
                    if (jsonField(fa.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
                    if (jsonField(fa.object, "partial_rotary_factor")) |v| config.partial_rotary_factor_global = try jsonFloat(v);
                    if (jsonField(fa.object, "rope_type")) |v| {
                        if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                            config.rope_proportional = true;
                            if (jsonField(fa.object, "factor")) |fv| config.rope_proportional_factor = try jsonFloat(fv);
                        }
                    }
                }
            }
            if (jsonField(rp_val.object, "sliding_attention")) |sa| {
                if (sa == .object) {
                    if (jsonField(sa.object, "rope_theta")) |v| config.rope_local_base_freq = try jsonFloat(v);
                }
            }
            // Qwen3.5 style: { "rope_theta": ..., "partial_rotary_factor": ... }
            if (jsonField(rp_val.object, "rope_theta")) |v| config.rope_theta = try jsonFloat(v);
            if (jsonField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        }
    }

    // Tie word embeddings
    if (jsonField(root, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (jsonField(cfg_obj, "tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (jsonField(root, "max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = try jsonU32(v);
        }
    }

    // Parse quantization from top level
    if (jsonField(outer, "quantization")) |q_val| {
        const q = try jsonValue(.object, q_val);
        if (jsonField(q, "bits")) |v| config.quant_bits = try jsonU32(v);
        if (jsonField(q, "group_size")) |v| config.quant_group_size = try jsonU32(v);
        if (jsonField(q, "mode")) |v| {
            if (v == .string) {
                config.quant_mode = QuantMode.fromString(v.string) orelse {
                    log.err("unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{v.string});
                    return error.UnsupportedQuantMode;
                };
            }
        }
        // Per-weight overrides carry their own mode. An unknown one (EigenLabs' `iq2_xxs` experts) was read
        // as the top-level affine layout and failed at the FIRST request's matmul; refuse it here, by name.
        var overrides = q.iterator();
        while (overrides.next()) |kv| {
            if (kv.value_ptr.* != .object) continue;
            const mode = kv.value_ptr.object.get("mode") orelse continue;
            if (mode != .string) continue;
            // A release's e4m3 projection on a 64 x 128 grid (`fp8_block128`) is read by its dtypes.
            if (QuantMode.fromString(mode.string) == null and !fp8_grid.isEntry(kv.value_ptr.object)) {
                log.err("unsupported quantization mode '{s}' on '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{ mode.string, kv.key_ptr.* });
                return error.UnsupportedQuantMode;
            }
        }
        // MLX ships affine kernels only for bits {2,3,4,5,6,8} (ops.cpp
        // rejects the rest at quantize() time, but an ALREADY-quantized
        // checkpoint skips that check and dies at Metal kernel load during
        // warmup — an uncatchable process kill). Reject at parse instead.
        if (config.quant_mode == .affine and config.quant_bits != 0) {
            switch (config.quant_bits) {
                2, 3, 4, 5, 6, 8 => {},
                else => {
                    log.err("unsupported affine quantization: {d}-bit (this MLX runtime supports 2, 3, 4, 5, 6, 8)\n", .{config.quant_bits});
                    return error.UnsupportedQuantBits;
                },
            }
        }
    }

    if (modelopt_pack.declared(outer)) {
        config.modelopt = true;
        config.quant_bits = 4;
        config.quant_group_size = 16;
        config.quant_mode = .nvfp4;
    }

    // EOS tokens
    if (jsonField(root, "eos_token_id")) |v| {
        switch (v) {
            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(try jsonU32(item));
                }
            },
            else => {},
        }
    }

    // Vision config (Gemma 4 SigLIP)
    if (jsonField(root, "vision_config")) |vc_val| {
        if (vc_val == .object) {
            config.has_vision = true;
            const vc = vc_val.object;
            if (jsonField(vc, "hidden_size")) |v| {
                if (v == .integer) config.vision_hidden_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_hidden_layers")) |v| {
                if (v == .integer) config.vision_num_layers = try jsonU32(v);
            }
            if (jsonField(vc, "num_attention_heads")) |v| {
                if (v == .integer) config.vision_num_heads = try jsonU32(v);
            }
            if (jsonField(vc, "head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "global_head_dim")) |v| {
                if (v == .integer) config.vision_head_dim = try jsonU32(v);
            }
            if (jsonField(vc, "intermediate_size")) |v| {
                if (v == .integer) config.vision_intermediate_size = try jsonU32(v);
            }
            if (jsonField(vc, "patch_size")) |v| {
                if (v == .integer) config.vision_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "pooling_kernel_size")) |v| {
                if (v == .integer) config.vision_pooling_kernel = try jsonU32(v);
            }
            if (jsonField(vc, "default_output_length")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "position_embedding_size")) |v| {
                if (v == .integer) config.vision_position_embedding_size = try jsonU32(v);
            }
            if (jsonField(vc, "rope_parameters")) |rp| {
                if (rp == .object) {
                    if (jsonField(rp.object, "rope_theta")) |v| config.vision_rope_theta = try jsonFloat(v);
                }
            }
            if (jsonField(vc, "use_clipped_linears")) |v| {
                if (v == .bool) config.vision_use_clipped_linears = v.bool;
            }
            // vision_config.standardize is presence-only — the actual `std_scale`/`std_bias`
            // safetensors presence drives behavior in `VisionEncoder.init`, so the config
            // flag needs no field.

            // Gemma 4 12B unified (encoder-free) vision fields. Distinct names
            // from the SigLIP tower: mm_embed_dim (vs hidden_size),
            // model_patch_size (48px merged patch vs 16px teacher patch_size),
            // num_soft_tokens (vs default_output_length), mm_posemb_size.
            if (jsonField(vc, "mm_embed_dim")) |v| {
                if (v == .integer) config.vision_mm_embed_dim = try jsonU32(v);
            }
            if (jsonField(vc, "model_patch_size")) |v| {
                if (v == .integer) config.vision_model_patch_size = try jsonU32(v);
            }
            if (jsonField(vc, "num_soft_tokens")) |v| {
                if (v == .integer) config.vision_soft_tokens = try jsonU32(v);
            }
            if (jsonField(vc, "mm_posemb_size")) |v| {
                if (v == .integer) config.vision_mm_posemb_size = try jsonU32(v);
            }
        }
    }
    // Audio config (Gemma 4 12B unified — raw-waveform projection, no conformer)
    if (jsonField(root, "audio_config")) |ac_val| {
        if (ac_val == .object) {
            const ac = ac_val.object;
            if (jsonField(ac, "audio_embed_dim")) |v| {
                if (v == .integer) config.audio_embed_dim = try jsonU32(v);
            }
            // audio_samples_per_token lives in processor_config, not config.json;
            // default 640 (40ms @ 16kHz) matches the only shipped unified checkpoint.
            if (jsonField(ac, "audio_samples_per_token")) |v| {
                if (v == .integer) config.audio_samples_per_token = try jsonU32(v);
            }
        }
    }
    if (jsonField(root, "audio_token_id")) |v| {
        if (v == .integer) config.audio_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boa_token_id")) |v| {
        if (v == .integer) config.boa_token_id = try jsonU32(v);
    }
    // eoa lives under `eoa_token_index` in the unified config.json.
    if (jsonField(root, "eoa_token_index")) |v| {
        if (v == .integer) config.eoa_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoa_token_id")) |v| {
        if (v == .integer and config.eoa_token_id == 0) config.eoa_token_id = try jsonU32(v);
    }
    // Image token ID (top-level or in mm_tokens_per_image config)
    if (jsonField(root, "image_token_id")) |v| {
        if (v == .integer) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = try jsonU32(v);
    }
    if (jsonField(root, "boi_token_id")) |v| {
        if (v == .integer) config.boi_token_id = try jsonU32(v);
    }
    if (jsonField(root, "eoi_token_id")) |v| {
        if (v == .integer) config.eoi_token_id = try jsonU32(v);
    }

    // Set model-family defaults based on model_type
    if (std.mem.eql(u8, model_type, "gemma3") or
        std.mem.eql(u8, model_type, "gemma3_text"))
    {
        // "gemma3_text" is the FLAT text-only checkpoint (Gemma3ForCausalLM,
        // e.g. the abliterated -lm- builds): no vision tower, weights under
        // "model.*". The multimodal "gemma3" nests them under
        // "language_model.model.*" and carries a text_config. Collapse both
        // onto "gemma3" so every downstream gemma3 comparison fires.
        config.model_type = "gemma3";
        // gemma-3-4b-it's text_config omits num_attention_heads / num_key_value_heads
        // / head_dim and leans on the HF Gemma3TextConfig defaults (8 q-heads,
        // 4 kv-heads, head_dim 256). Our struct defaults are the 12b/27b shape
        // (16/8), so apply the HF defaults explicitly when the config is silent —
        // otherwise the Q projection (8*256) reshapes against 16 heads and the
        // model crashes at warmup (issue #43). The 12b ships these fields, so its
        // values are read at lines 458-460 and these fills never fire.
        if (jsonField(cfg_obj, "num_attention_heads") == null) config.num_attention_heads = 8;
        if (jsonField(cfg_obj, "num_key_value_heads") == null) config.num_key_value_heads = 4;
        if (jsonField(cfg_obj, "head_dim") == null) config.head_dim = 256;
        // Multimodal checkpoint → "language_model.model"; flat text-only
        // (gemma3_text, no text_config) → "model" (mirrors the LFM2 VL split).
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        // Gemma always ties word embeddings; the abliterated text-only build
        // omits the flag (would default false) and ships no lm_head tensor, so
        // force it on. An explicit lm_head tensor, if present, still wins in
        // transformer.zig's resolution.
        config.tie_word_embeddings = true;
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = true;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        if (config.rope_scaling_factor == 1.0) {
            if (jsonField(cfg_obj, "rope_scaling")) |rs_val| {
                if (rs_val == .object) {
                    if (jsonField(rs_val.object, "factor")) |_| {} else {
                        config.rope_scaling_factor = 8.0;
                    }
                } else {
                    config.rope_scaling_factor = 8.0;
                }
            } else {
                config.rope_scaling_factor = 8.0;
            }
        }
        config.ensureGemmaTerminators();
    } else if (std.mem.eql(u8, model_type, "gemma4") or
        std.mem.eql(u8, model_type, "gemma4_text") or
        std.mem.eql(u8, model_type, "gemma4_unified") or
        std.mem.eql(u8, model_type, "gemma4_unified_text"))
    {
        // Per the Gemma 4 12B developer guide, `gemma4_unified` "contains the
        // same advanced decoder structure as the Gemma 4 31B Dense model" —
        // the unified-ness lives in a tiny vision/audio embedder we don't
        // wire here. So we collapse the internal tag onto plain "gemma4" so
        // every downstream model_type comparison in transformer.zig/drafter.zig
        // (attn_scale gate, recommendedBlockSize, etc.) treats it identically
        // to 31B Dense without per-arch fan-out.
        // Detect the unified (12B encoder-free multimodal) variant before we
        // collapse the tag — drives vision_embedder/embed_audio weight loading
        // and the encoder-free forward in src/engines/mlx/vision_encoder.zig.
        config.is_gemma4_unified = std.mem.indexOf(u8, model_type, "unified") != null;
        config.model_type = "gemma4";
        config.weight_prefix = "language_model.model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false; // Gemma 4 norms have NO offset (plain weight, not 1+weight)
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true; // Parameter-free RMS norm on values
        config.rope_scaling_factor = 1.0; // No scaling, uses proportional RoPE via theta
        // [1, 106, 50] from the config array (when present) plus an additive
        // guarantee of <eos>(1) + <end_of_turn>(106) for checkpoints that
        // declare only a scalar eos_token_id.
        config.ensureGemmaTerminators();
        // Note on `attention_k_eq_v` for the 12B unified checkpoint: the
        // weights ship separate v_proj for SLIDING layers and omit them for
        // FULL_ATTENTION layers — i.e. K==V alias is a per-layer choice
        // keyed on global-layer-ness. The existing bindModelWeights logic
        // (transformer.zig:5677) already encodes that:
        //   `k_eq_v = config.attention_k_eq_v and isGlobalLayer(li)`.
        // So we leave the parsed flag intact.
    } else if (std.mem.eql(u8, model_type, "embedding_gemma2")) {
        // Google EmbeddingGemma 2: the Gemma 4 text trunk trained as a bidirectional encoder, with per-layer
        // inputs derived from the embeddings alone (no `embed_tokens_per_layer`) and an `embedding_projection`
        // head. The image tower is a Gemma 4 SigLIP tower; the audio tower is not served.
        config.model_type = "embedding_gemma2";
        config.weight_prefix = "language_model";
        config.hidden_act = .gelu_approx;
        config.norm_has_offset = false;
        config.scale_embeddings = true;
        config.has_pre_ff_norm = true;
        config.has_qk_norm = true;
        config.has_v_norm = true;
        config.tie_word_embeddings = true;
        config.is_encoder_only = true;
        config.use_bidirectional_attention = true;
        config.gv_budgeted = true;
        // The model card's window, not the trunk's 262144-position rope table.
        config.max_position_embeddings = if (config.max_position_embeddings == 0) EMBEDDING_GEMMA2_WINDOW else @min(config.max_position_embeddings, EMBEDDING_GEMMA2_WINDOW);
        try parseEmbeddingGemma2Geometry(cfg_obj, &config);
        if (jsonField(root, "video_token_id")) |v| {
            if (v == .integer) config.video_token_id = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "eos_token_id")) |v| switch (v) {
            .integer => |i| config.addEosToken(std.math.cast(u32, i) orelse return error.InvalidConfigField),
            .array => |arr| for (arr.items) |item| {
                if (item == .integer) config.addEosToken(try jsonU32(item));
            },
            else => {},
        };
    } else if (std.mem.eql(u8, model_type, "muse_glimmer") or
        std.mem.eql(u8, model_type, "muse_glimmer_text"))
    {
        // Muse-Glimmer-30B (meta-models). Dense GQA trunk (32/2 heads, hd 128)
        // with Gemma2-style sandwich-norm layers; every 4th layer counted
        // backward from the last is full-attention AND NoPE (layer_rope_theta
        // 0), the rest slide at 2048. "muse_glimmer_text" is the flat
        // text-only sibling (bare "model" prefix, no text_config).
        config.model_type = "muse_glimmer";
        config.legacy.thinking = .on_with_tools;
        config.weight_prefix = if (jsonField(root, "text_config") != null) "model.language_model" else "model";
        config.hidden_act = .silu;
        config.norm_has_offset = true; // sandwich norms are Gemma2-centered (1+w)…
        config.final_norm_plain = true; // …but model.norm is plain-scale (ones-init)
        config.scale_embeddings = false; // embeddings are RMS-normed, not sqrt(hidden)-scaled
        config.has_pre_ff_norm = true;
        config.has_qk_norm = false; // no q_norm/k_norm tensors in the checkpoint —
        config.qk_norm_weightless = true; // shared weight-less RMS on Q and K instead
        config.normed_embeddings = true;
        config.attn_sigmoid_gate = true;
        // Reference-config class defaults, overridden by explicit keys below.
        config.qk_scale_factor = 3.87;
        config.output_multiplier = 0.19611613513818404;
        config.post_norm_eps = 1e-8;
        if (jsonField(cfg_obj, "qk_scale_factor")) |v| config.qk_scale_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "output_multiplier")) |v| config.output_multiplier = try jsonFloat(v);
        if (jsonField(cfg_obj, "post_norm_eps")) |v| config.post_norm_eps = try jsonFloat(v);
        // ONE theta for every roped layer: sliding layers read
        // rope_local_base_freq in the forward, but muse ships its base only as
        // rope_parameters.rope_theta — without this the Gemma-flavored 10000
        // default mis-rotates all 39 roped layers (the first-turn
        // repetition-loop root cause; global layers are NoPE so EVERY rotated
        // layer ran at the wrong base).
        if (jsonField(cfg_obj, "rope_local_base_freq") == null)
            config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "layer_rope_theta")) |lrt| {
            if (lrt == .array) {
                for (lrt.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    config.layer_no_rope[i] = try jsonFloat(item) == 0;
                }
            }
        }
        // Vision tower (src/plugins/muse_glimmer/vision.zig). Muse names its geometry keys its
        // own way and the window/full pattern is a per-layer list, not a stride.
        if (jsonField(root, "vision_config")) |vc_val| {
            if (vc_val == .object) {
                const vc = vc_val.object;
                config.muse_vision = true;
                if (jsonField(vc, "num_hidden_layers")) |v| {
                    if (v == .integer) config.qv_depth = try jsonU32(v);
                }
                if (jsonField(vc, "hidden_size")) |v| {
                    if (v == .integer) config.qv_hidden = try jsonU32(v);
                }
                if (jsonField(vc, "num_attention_heads")) |v| {
                    if (v == .integer) config.qv_heads = try jsonU32(v);
                }
                if (jsonField(vc, "intermediate_size")) |v| {
                    if (v == .integer) config.qv_intermediate = try jsonU32(v);
                }
                if (jsonField(vc, "patch_size")) |v| {
                    if (v == .integer) config.qv_patch = try jsonU32(v);
                }
                if (jsonField(vc, "patch_temporal")) |v| {
                    if (v == .integer) config.qv_temporal_patch = try jsonU32(v);
                }
                if (jsonField(vc, "merge_size")) |v| {
                    if (v == .integer) config.qv_merge = try jsonU32(v);
                }
                if (jsonField(vc, "pos_emb_height")) |v| {
                    if (v == .integer) config.mv_pos_side = try jsonU32(v);
                }
                if (jsonField(vc, "layer_norm_eps")) |v| config.mv_ln_eps = try jsonFloat(v);
                if (jsonField(vc, "rope_parameters")) |rp| {
                    if (rp == .object) {
                        if (jsonField(rp.object, "rope_theta")) |v| config.mv_rope_theta = try jsonFloat(v);
                    }
                }
                if (jsonField(vc, "layer_types")) |lt| {
                    if (lt == .array) for (lt.array.items, 0..) |item, i| {
                        if (i >= MAX_VISION_LAYERS) break;
                        if (item == .string) config.mv_full_attn[i] = std.mem.eql(u8, item.string, "full_attention");
                    };
                }
                if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
                config.qv_out_hidden = config.hidden_size;
                if (jsonField(root, "projector_hidden_size")) |v| {
                    if (v == .integer) config.mv_projector_hidden = try jsonU32(v);
                }
                // The processor wraps the pad run in <|image_start|>/<|image_end|>;
                // config.json carries neither, so the ids come from the vocab.
                config.boi_token_id = 200080;
                config.eoi_token_id = 200081;
            }
        }
        config.ensureMuseTerminators();
    } else if (std.mem.eql(u8, model_type, "spark2_5")) {
        // XHToken Spark-X2.5 (1.7B / 4B): dense GQA, hd 256, 3:1 sliding(512)/
        // full layers with per-type RoPE (sliding: full rotary at 1e4; full:
        // 25% rotary at 5e6), exact-erf GELU gated MLP, plain RMS norms,
        // per-head sigmoid attention output gate, fused q_k_v_proj, tied head.
        config.model_type = "spark2_5";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .gelu;
        config.attn_sigmoid_gate = true;
        config.attn_gate_headwise = true;
        config.attn_fused_qkv = true;
        config.rope_scaling_factor = 1.0;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_5_moe") or
        std.mem.eql(u8, model_type, "qwen3_5") or
        std.mem.eql(u8, model_type, "qwen3_5_moe_text") or
        std.mem.eql(u8, model_type, "qwen3_5_text") or
        std.mem.eql(u8, model_type, "prism_hadamard_qwen35") or
        // MiniCPM-V 4.7: a Qwen3.5 trunk behind a SigLIP2 tower (src/plugins/qwen3_5/minicpm_vision.zig).
        std.mem.eql(u8, model_type, "minicpmv4_7"))
    {
        config.model_type = "qwen3_5_moe";
        if (std.mem.eql(u8, model_type, "prism_hadamard_qwen35")) config.hadamard_block = try prismHadamardBlock(root);
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        try parseQwenVisionFields(&config, root, cfg_obj);
        if (std.mem.eql(u8, model_type, "minicpmv4_7")) {
            // Its vision_config is a SigLIP2 tower, not Qwen3-VL's.
            config.qwen_vision = false;
            try parseMiniCpmVision(&config, root);
        }
    } else if (std.mem.eql(u8, model_type, "qwen4_exp") or
        std.mem.eql(u8, model_type, "qwen4_exp_text"))
    {
        config.model_type = "qwen4_exp";
        config.legacy.caps.long_ctx = true;
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false;
        if (root.get("qwen4_norm_convention")) |v| {
            if (v != .string) return error.InvalidQwen4NormConvention;
            config.qwen4_norm_convention = std.meta.stringToEnum(Qwen4NormConvention, v.string) orelse return error.InvalidQwen4NormConvention;
        }
        config.has_final_norm = false; // hyper_connection_mixer replaces model.norm
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.kda_sigmoid_out_gate = true; // output_gate_type "sigmoid"
        // Keyed on the arch: the HF reference ignores `mamba_ssm_dtype` and is f32 regardless.
        config.ssm_state_f32 = gdnStateF32Enabled();
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.hc_count = 4;
        config.hc_lowrank = 320;
        config.ple_embed_dim = config.hidden_size;
        try parseQwenVisionFields(&config, root, cfg_obj);
        // YaRN (262144 → e.g. 1048576) rides ONE rotary table for the whole
        // trunk: attention, the QSA indexer and the MTP head all read it, so
        // a scaled rotation cannot desync the block selector from attention.
        try parseYarnRopeParameters(&config, cfg_obj);
        // Read strictly; range-checked in `validateQwen4Config` once every field is in.
        if (try qwen4ConfigU32(cfg_obj, "hc_count")) |v| config.hc_count = v;
        if (try qwen4ConfigU32(cfg_obj, "hc_lowrank")) |v| config.hc_lowrank = v;
        {
            const v = jsonField(cfg_obj, "ple_layer_ids") orelse return error.InvalidQwen4PleLayer;
            if (v != .array or v.array.items.len != 1 or v.array.items[0] != .integer) {
                return error.InvalidQwen4PleLayer;
            }
            const id = v.array.items[0].integer;
            if (id < 1 or id > @as(i64, config.num_hidden_layers)) return error.InvalidQwen4PleLayer;
            config.ple_layer_idx = std.math.cast(i32, id - 1) orelse return error.InvalidQwen4PleLayer;
        }
        if (try qwen4ConfigU32(cfg_obj, "ple_embed_dim")) |v| config.ple_embed_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "ple_conv_kernel_size")) |v| config.ple_conv_kernel = v;
        if (try qwen4ConfigU32(cfg_obj, "ngram_size")) |v| config.ngram_size = v;
        if (try qwen4ConfigU32(cfg_obj, "heads_per_ngram")) |v| config.heads_per_ngram = v;
        if (try qwen4ConfigU64(cfg_obj, "ngram_vocab_size_base")) |v| config.ngram_vocab_base = v;
        if (try qwen4ConfigU32(cfg_obj, "make_ngram_vocab_size_divisible_by")) |v| config.ngram_vocab_divisor = v;
        if (try qwen4ConfigU64(cfg_obj, "seed")) |v| config.ngram_seed = v;
        if (try qwen4ConfigU32(cfg_obj, "split_ngram_parts")) |v| config.split_ngram_parts = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_n_heads")) |v| config.indexer_n_heads = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_head_dim")) |v| config.indexer_head_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_budget")) |v| config.indexer_budget = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_compress_ratio")) |v| config.indexer_compress_ratio = v;
        if (jsonField(cfg_obj, "eos_token_id")) |v| {
            switch (v) {
                .integer => |i| config.ngram_eos = std.math.cast(u32, i) orelse return error.InvalidConfigField,
                .array => |arr| if (arr.items.len > 0 and arr.items[0] == .integer) {
                    config.ngram_eos = try jsonU32(arr.items[0]);
                },
                else => {},
            }
            if (config.num_eos_tokens == 0) config.addEosToken(config.ngram_eos);
        }
        try validateQwen4Config(&config);
        if (jsonField(root, "expert_quant") != null) {
            config.exl3 = try exl3.parse(root, config.num_experts_per_tok);
        }
        try parseQwen4FullExl3(root, &config);
    } else if (std.mem.eql(u8, model_type, "qwen3_moe") or
        std.mem.eql(u8, model_type, "qwen3_moe_text"))
    {
        // Qwen3-30B-A3B / Qwen3-Coder-30B-A3B. Shares qwen3_5_moe's weight
        // layout (`mlp.gate` router + stacked `mlp.switch_mlp.*` experts) and
        // its MoE forward, but differs in three ways that make it its OWN
        // model_type rather than a remap onto qwen3_5_moe:
        //   1. No GatedDeltaNet — every layer is full attention
        //      (full_attention_interval stays 0 ⇒ isLinearLayer == false).
        //   2. No attention output gate (attn_output_gate stays false; the
        //      qwen3_5 split-Q path would mis-shape the projection here).
        //   3. No shared expert (shared_expert_intermediate_size: 0, no
        //      mlp.shared_expert.* weights). The MoE binding in
        //      transformer.zig loads those optionally and the forward skips
        //      the shared branch when shared_expert_gate_w is null.
        // weight_prefix is plain "model" (no language_model nesting).
        config.model_type = "qwen3_moe";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "gpt_oss")) {
        // OpenAI gpt-oss (20B-A3.6B / 120B-A5.1B). A plain dense-attention MoE
        // that rides the qwen3_moe forward arms — no linear/SSM layers, no
        // module-owned decode state, so prefix cache, batched decode and spec
        // decode all apply normally. Family-specific pieces:
        //   1. Learned per-head attention SINKS (self_attn.sinks) — an extra
        //      softmax-denominator column. mlx's fused SDPA takes them.
        //   2. Clamped SwiGLU (swiglu_limit, alpha 1.702, +1 on the linear
        //      branch) instead of silu(gate)*up.
        //   3. Additive biases everywhere: q/k/v/o_proj.bias, mlp.router.bias
        //      and per-expert gate/up/down bias — all living BESIDE the affine
        //      quantizer's `.biases` tensors in the same checkpoint.
        //   4. No QK norm.
        // Router is softmax-over-top-k, which is algebraically identical to
        // the existing softmax-all → top-k → renorm chain, so moe_route_norm
        // stays true and moe_sigmoid_router stays false.
        config.model_type = "gpt_oss";
        config.legacy.thinking = .on;
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_attn_sinks = true;
        // ONE theta for both layer types. The Gemma-flavored default of 10000
        // would silently mis-base every sliding layer — the muse
        // first-turn-repetition class (deterministic "coherent then loops").
        config.rope_local_base_freq = config.rope_theta;
        config.rope_scaling_factor = 1.0;
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Expert count rides `num_local_experts`; the generic block only knows
        // `num_experts`. Top-k has two spellings and both appear in shipped
        // configs (`num_experts_per_tok` is generic, `experts_per_token` is not).
        if (jsonField(cfg_obj, "num_local_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "experts_per_token")) |v| {
            if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
        }
        // There is no moe_intermediate_size key: the expert width IS
        // intermediate_size (2880 on both sizes).
        if (config.moe_intermediate_size == 0) {
            config.moe_intermediate_size = config.intermediate_size;
        }
        if (jsonField(cfg_obj, "swiglu_limit")) |v| config.swiglu_limit = try jsonFloat(v);
        // Flat YaRN block. mscale is COMPUTED, never read: the config ships no
        // "attention_factor" at all, and mlx-lm's YarnRoPE defaults
        // (mscale 1 / mscale_all_dim 0) give 0.1*ln(factor) + 1. Laguna
        // precedent — but note dsv4 is the same shape with the OPPOSITE
        // answer, so this stays a per-arch decision.
        if (jsonField(cfg_obj, "rope_scaling")) |rs| {
            if (rs == .object) {
                const is_yarn = if (jsonField(rs.object, "rope_type")) |rt|
                    (rt == .string and std.mem.eql(u8, rt.string, "yarn"))
                else
                    false;
                if (is_yarn) {
                    config.rope_yarn = true;
                    if (jsonField(rs.object, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                    if (jsonField(rs.object, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                    if (jsonField(rs.object, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                    if (jsonField(rs.object, "original_max_position_embeddings")) |x| {
                        if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                    }
                    if (config.yarn_factor > 1.0) {
                        config.yarn_attention_factor = 0.1 * @log(config.yarn_factor) + 1.0;
                    }
                }
            }
        }
        config.ensureGptOssTerminators();
    } else if (std.mem.eql(u8, model_type, "glm5_next") or std.mem.eql(u8, model_type, "glm5_next_text")) {
        // Z.ai GLM-5.3-Flash (Glm5NextForConditionalGeneration) in mlx-vlm's glm5_next layout
        // (TensorFold's MLX packs). Three Kimi-Delta-Attention layers per DeepSeek sparse
        // attention layer, every sublayer inside DeepSeek-V4's 4-stream Sinkhorn
        // hyper-connection. The sparse attention is absorbed NoPE MLA over a 512 latent; a
        // pooled indexer picks the tokens each query reads. Sigmoid noaux_tc MoE with one
        // ungated shared expert and a clamped SwiGLU; dense MLP on the leading layers.
        config.model_type = "glm5_next";
        config.legacy.thinking = .on;
        config.legacy.caps.batch_pads_kv = false;
        config.legacy.caps.dflash_policy = true;
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "n_routed_experts")) |v| config.num_experts = try jsonU32(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "n_group")) |v| config.moe_n_group = try jsonU32(v);
        if (jsonField(cfg_obj, "topk_group")) |v| config.moe_topk_group = try jsonU32(v);
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size =
                std.math.mul(u32, try jsonU32(v), config.moe_intermediate_size) catch return error.InvalidConfigField;
        }
        if (jsonField(cfg_obj, "swiglu_limit")) |v| {
            if (v != .null) config.swiglu_limit = try jsonFloat(v);
        }
        config.swiglu_clamp = true;

        // Sparse attention: NoPE, q/k and v at one width.
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer and v.integer != 0) {
                log.err("glm5_next: qk_rope_head_dim {d} (only NoPE sparse attention is served)\n", .{v.integer});
                return error.UnsupportedGlm5Config;
            }
        }
        if (jsonField(cfg_obj, "q_lora_rank")) |v| config.dsa_q_lora_rank = try jsonU32(v);
        if (jsonField(cfg_obj, "kv_lora_rank")) |v| config.dsa_kv_lora_rank = try jsonU32(v);
        if (jsonField(cfg_obj, "qk_nope_head_dim")) |v| config.dsa_head_dim = try jsonU32(v);
        if (jsonField(cfg_obj, "v_head_dim")) |v| {
            if ((try jsonU32(v)) != config.dsa_head_dim) {
                log.err("glm5_next: v_head_dim differs from qk_nope_head_dim (not supported)\n", .{});
                return error.UnsupportedGlm5Config;
            }
        }
        config.head_dim = config.dsa_head_dim;
        if (config.num_key_value_heads != config.num_attention_heads) config.num_key_value_heads = config.num_attention_heads;
        if (jsonField(cfg_obj, "index_topk")) |v| config.dsa_index_topk = try jsonU32(v);
        if (jsonField(cfg_obj, "index_kpool")) |v| config.dsa_index_kpool = try jsonU32(v);
        if (jsonField(cfg_obj, "index_n_heads")) |v| config.dsa_index_heads = try jsonU32(v);
        if (jsonField(cfg_obj, "index_head_dim")) |v| config.dsa_index_head_dim = try jsonU32(v);
        inline for (.{ "index_kpool_compress", "index_kpool_always_select_tail", "mla_use_nope" }) |key| {
            if (jsonField(cfg_obj, key)) |v| {
                if (v == .bool and !v.bool) {
                    log.err("glm5_next: {s}=false (not supported)\n", .{key});
                    return error.UnsupportedGlm5Config;
                }
            }
        }
        if (config.dsa_index_kpool == 0 or config.dsa_index_topk % config.dsa_index_kpool != 0) {
            log.err("glm5_next: index_topk must be a multiple of index_kpool\n", .{});
            return error.UnsupportedGlm5Config;
        }
        // Every indexer scores its own layer (`shared` reuses the previous one's top-k).
        if (jsonField(cfg_obj, "indexer_types")) |v| {
            for ((try jsonValue(.array, v)).items) |t| {
                if (!std.mem.eql(u8, try jsonValue(.string, t), "full")) {
                    log.err("glm5_next: a shared indexer layer (not supported)\n", .{});
                    return error.UnsupportedGlm5Config;
                }
            }
        }

        // Layer pattern: sparse attention every 4th layer, the rest KDA.
        config.full_attention_interval = 4;
        if (jsonField(cfg_obj, "layer_types")) |v| {
            for ((try jsonValue(.array, v)).items, 0..) |t, i| {
                const name = try jsonValue(.string, t);
                const sparse = std.mem.eql(u8, name, "deepseek_sparse_attention") or std.mem.eql(u8, name, "full_attention");
                if (sparse != !config.isLinearLayer(@intCast(i))) {
                    log.err("glm5_next: layer {d} is {s}, off the every-4th pattern (not supported)\n", .{ i, name });
                    return error.UnsupportedGlm5Config;
                }
            }
        }
        if (jsonField(cfg_obj, "mlp_layer_types")) |v| {
            config.first_k_dense_replace = 0;
            for ((try jsonValue(.array, v)).items, 0..) |t, i| {
                if (std.mem.eql(u8, try jsonValue(.string, t), "dense")) {
                    if (config.first_k_dense_replace != i) {
                        log.err("glm5_next: dense layer {d} after a MoE layer (not supported)\n", .{i});
                        return error.UnsupportedGlm5Config;
                    }
                    config.first_k_dense_replace += 1;
                }
            }
        } else if (jsonField(cfg_obj, "first_k_dense_replace")) |v| config.first_k_dense_replace = try jsonU32(v);

        // KDA: per-channel bounded gate, sigmoid output gate, low-rank gate projections.
        config.linear_num_key_heads = config.num_attention_heads;
        config.linear_key_head_dim = 128;
        if (jsonField(cfg_obj, "linear_attn_config")) |lac_v| {
            const lac = try jsonValue(.object, lac_v);
            if (jsonField(lac, "num_heads")) |v| config.linear_num_key_heads = try jsonU32(v);
            if (jsonField(lac, "head_dim")) |v| config.linear_key_head_dim = try jsonU32(v);
            if (jsonField(lac, "short_conv_kernel_size")) |v| config.linear_conv_kernel_dim = try jsonU32(v);
            if (jsonField(lac, "gate_lower_bound")) |v| {
                if (v != .null) config.kda_gate_lower_bound = try jsonFloat(v);
            }
        }
        if (config.kda_gate_lower_bound == 0.0) config.kda_gate_lower_bound = -5.0;
        config.linear_num_value_heads = config.linear_num_key_heads;
        config.linear_value_head_dim = config.linear_key_head_dim;
        config.kda_vector_gate = true;
        config.kda_sigmoid_out_gate = true;

        // DeepSeek-V4's Sinkhorn hyper-connection around every sublayer.
        config.dsv4_hc_mult = 4;
        if (jsonField(cfg_obj, "hc_mult")) |v| config.dsv4_hc_mult = try jsonU32(v);
        if (config.dsv4_hc_mult != 4) {
            log.err("glm5_next: hc_mult {d} (only 4 is served)\n", .{config.dsv4_hc_mult});
            return error.UnsupportedGlm5Config;
        }
        config.dsv4_hc_sinkhorn_iters = 20;
        if (jsonField(cfg_obj, "hc_sinkhorn_iters")) |v| config.dsv4_hc_sinkhorn_iters = try jsonU32(v);
        if (jsonField(cfg_obj, "hc_eps")) |v| config.dsv4_hc_eps = try jsonFloat(v);
        // The tower (src/plugins/glm5_next/vision.zig) splices at `<|image|>` with plain 1-D positions: the trunk is NoPE.
        if (config.has_vision) {
            const vc = jsonField(root, "vision_config").?.object;
            config.glm_vision = true;
            try parseQvGeometry(&config, vc);
            if (jsonField(vc, "projection_intermediate_size")) |v| config.glmv_proj_intermediate = try jsonU32(v);
            if (jsonField(vc, "rms_norm_eps")) |v| config.glmv_eps = try jsonFloat(v);
            if (jsonField(vc, "swiglu_limit")) |v| config.glmv_swiglu_limit = try jsonFloat(v);
            // The 2-D rotary splits a head into h and w halves of rotate_half pairs.
            if (config.qv_heads == 0 or config.qv_hidden % config.qv_heads != 0 or config.qv_head_dim % 4 != 0 or
                config.qv_patch == 0 or config.qv_merge == 0 or config.qv_temporal_patch == 0 or
                config.qv_out_hidden != config.hidden_size) return error.InvalidConfigField;
        }
        try parseSushiExperts(root, &config);
    } else if (std.mem.eql(u8, model_type, "mimo_v2") or std.mem.eql(u8, model_type, "mimo_v2_flash")) {
        // Xiaomi MiMo-V2.6-Flash (309B-A15B) in the mlx-lm layout (tests/convert_mimo_v2.py,
        // the community packs); `mimo_v2_flash` is MiMo-V2-Flash and TensorFold's V2.6 label.
        // gpt_oss-shaped attention (learned sinks, 128-token sliding window) with
        // per-layer-type geometry: global layers 4 KV heads at theta 1e7, sliding
        // layers 8 KV heads at theta 1e4 with sinks; K 192 / V 128, RoPE on the
        // first int(192 * 0.334) = 64 channels. Experts: laguna's sigmoid routing
        // with a selection-only bias and no shared expert; dense layers lead.
        // A Sushi pack's `quant_method: fp8` describes its trunk; its experts are EXL3.
        if (jsonField(root, "expert_quant") == null) try refuseUnconvertedMimo(cfg_obj);
        config.model_type = "mimo_v2";
        config.legacy.thinking = .on;
        config.legacy.caps.verify_truncates = true;
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.hidden_act = .silu;
        config.has_attn_sinks = true;
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "layernorm_epsilon")) |v| config.rms_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "attention_value_scale")) |v| {
            if (v != .null) config.attention_value_scale = try jsonFloat(v);
        }
        if (jsonField(cfg_obj, "n_routed_experts")) |v| config.num_experts = try jsonU32(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| {
            if (v != .null) config.router_scaling_factor = try jsonFloat(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| config.moe_n_group = try jsonU32(v);
        if (jsonField(cfg_obj, "topk_group")) |v| config.moe_topk_group = try jsonU32(v);
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer and v.integer != 0) {
                log.err("mimo_v2: n_shared_experts {d} not supported\n", .{v.integer});
                return error.UnsupportedMimoConfig;
            }
        }

        // Pattern 0 = global, 1 = sliding; the dense MLP layers lead the stack.
        const pattern = try jsonValue(.array, jsonField(cfg_obj, "hybrid_layer_pattern") orelse return error.UnsupportedMimoConfig);
        const freq = try jsonValue(.array, jsonField(cfg_obj, "moe_layer_freq") orelse return error.UnsupportedMimoConfig);
        if (pattern.items.len != config.num_hidden_layers or freq.items.len != config.num_hidden_layers or config.num_hidden_layers > 128) {
            log.err("mimo_v2: hybrid_layer_pattern/moe_layer_freq must list all {d} layers\n", .{config.num_hidden_layers});
            return error.UnsupportedMimoConfig;
        }
        config.has_explicit_layer_types = true;
        for (pattern.items, freq.items, 0..) |p, f, i| {
            config.layer_is_global[i] = (try jsonU32(p)) == 0;
            const moe = (try jsonU32(f)) != 0;
            if (!moe and config.first_k_dense_replace != i) {
                log.err("mimo_v2: dense layer {d} after a MoE layer is not supported\n", .{i});
                return error.UnsupportedMimoConfig;
            }
            if (!moe) config.first_k_dense_replace += 1;
        }

        // Root KV heads / theta are the GLOBAL layers'; `swa_*` the sliding ones.
        config.num_global_key_value_heads = config.num_key_value_heads;
        if (jsonField(cfg_obj, "swa_num_key_value_heads")) |v| config.num_key_value_heads = try jsonU32(v);
        if (jsonField(cfg_obj, "swa_rope_theta")) |v| config.rope_local_base_freq = try jsonFloat(v);
        if (jsonField(cfg_obj, "v_head_dim")) |v| config.v_head_dim = try jsonU32(v);
        const same = .{ .{ "swa_head_dim", config.head_dim }, .{ "swa_v_head_dim", config.v_head_dim }, .{ "swa_num_attention_heads", config.num_attention_heads } };
        inline for (same) |kv| {
            if (jsonField(cfg_obj, kv[0])) |v| {
                if ((try jsonU32(v)) != kv[1]) {
                    log.err("mimo_v2: {s} differs from the global layers' (not supported)\n", .{kv[0]});
                    return error.UnsupportedMimoConfig;
                }
            }
        }
        config.partial_rotary_factor_global = config.partial_rotary_factor;
        config.query_pre_attn_scalar = config.head_dim;
        config.rope_scaling_factor = 1.0;
        // The MiMo-ViT tower is not wired: the generic vision_config block must not arm SigLIP.
        config.has_vision = false;
        try parseSushiExperts(root, &config);
    } else if (std.mem.eql(u8, model_type, "hy_v3")) {
        // Tencent Hunyuan 3 (Hy3, 295B-A21B MoE; July 2026). Pure
        // full-attention MoE that rides the qwen3_moe forward arms: GQA with
        // per-head QK RMS-norm, full rotary, scale = head_dim^-0.5, no output
        // gate, plain "model" prefix. Family-specific pieces handled
        // explicitly: DeepSeek-V3-style SIGMOID router with expert bias
        // (mlp.expert_bias, f32) + top-k renorm + router_scaling_factor, an
        // UNGATED always-added shared expert (mlp.shared_mlp.*), and
        // first_k_dense_replace dense bottom layers. Reference: mlx-lm PR
        // #1211 (converted repos ship hy_v3.py alongside the weights).
        // NOTE: tencent's generation_config documents temp 0.9 / top_k -1 /
        // top_p 1 — untruncated by design, so applyFamilySamplingDefaults
        // deliberately has no hy_v3 fill.
        config.model_type = "hy_v3";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.moe_sigmoid_router = true;
        // qk_norm / route_norm default TRUE when absent (mlx-lm ModelArgs
        // defaults) but an explicit false must win.
        if (jsonField(cfg_obj, "qk_norm")) |v| {
            if (v == .bool) config.has_qk_norm = v.bool;
        }
        if (jsonField(cfg_obj, "route_norm")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "router_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "first_k_dense_replace")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        // Expert width may ride as expert_hidden_dim when moe_intermediate_size
        // is absent (both = 1536 on the 295B).
        if (config.moe_intermediate_size == 0) {
            if (jsonField(cfg_obj, "expert_hidden_dim")) |v| {
                if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
            }
        }
        // No explicit shared_expert_intermediate_size key in hy_v3 configs:
        // derive num_shared_experts × expert width. 0 shared experts leaves it
        // 0 and the binder/forward skip the shared branch.
        if (jsonField(cfg_obj, "num_shared_experts")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size =
                std.math.mul(u32, try jsonU32(v), config.moe_intermediate_size) catch return error.InvalidConfigField;
        }
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.ensureHy3Terminators();
    } else if (std.mem.eql(u8, model_type, "bailing_hybrid")) {
        // inclusionAI Ling 3.0 (bailing_hybrid; BailingMoeV3ForCausalLM). A
        // KDA + MLA hybrid MoE: three Kimi-Delta-Attention linear layers for
        // every Multi-head-Latent-Attention layer (layer_group_size 4, so
        // layers 3/7/11/… are full attention), DeepSeek-V3-style grouped
        // sigmoid routing with an expert bias, one ungated shared expert, and
        // a dense MLP on the bottom first_k_dense_replace layers.
        //
        // Three things here are NOT the qwen3.5 GDN defaults and each has its
        // own config field: the forget gate is per CHANNEL (kda_vector_gate),
        // it uses the bounded sigmoid form rather than softplus
        // (kda_gate_lower_bound), and the output gate is a plain sigmoid.
        // RoPE covers only the qk_rope_head_dim slice of each query/key head
        // and rotates ADJACENT PAIRS (rope_interleave).
        //
        // References: the checkpoint's own modeling_bailing_moe_v3.py, and
        // fla/ops/kda/fused_recurrent.py for the gate. Mirror:
        // rapid-mlx/Ling-3.0-tiny-MLX-4bit.
        config.model_type = "bailing_hybrid";
        config.legacy.thinking = .on;
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false; // MLA norms the LATENTS, not the heads
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;

        // Hybrid layout. `layer_group_size` counts layers per group with the
        // LAST one full — exactly isLinearLayer's `(idx+1) % interval != 0`.
        if (jsonField(cfg_obj, "layer_group_size")) |v| {
            if (v == .integer) config.full_attention_interval = try jsonU32(v);
        }

        // KDA: one linear head per attention head, key dim = value dim = head_dim.
        config.linear_num_key_heads = config.num_attention_heads;
        config.linear_num_value_heads = config.num_attention_heads;
        config.linear_key_head_dim = config.head_dim;
        config.linear_value_head_dim = config.head_dim;
        if (jsonField(cfg_obj, "short_conv_kernel_size")) |v| {
            if (v == .integer) config.linear_conv_kernel_dim = try jsonU32(v);
        }
        // A per-head KDA (`num_kv_heads_for_linear_attn`) would give the linear
        // layers their own head count instead of the attention heads' — the
        // three lines above would then be wrong, so refuse rather than size the
        // recurrent state off the wrong geometry.
        if (jsonField(cfg_obj, "num_kv_heads_for_linear_attn")) |v| {
            if (v == .integer and v.integer != 0 and v.integer != @as(i64, config.num_attention_heads)) {
                log.err("bailing_hybrid: num_kv_heads_for_linear_attn {d} != num_attention_heads {d} (per-head KDA not supported)\n", .{ v.integer, config.num_attention_heads });
                return error.UnsupportedBailingConfig;
            }
        }
        config.kda_vector_gate = true;
        config.kda_sigmoid_out_gate = true;
        // fla's kernel has TWO gate arms and this key selects between them: a
        // negative bound takes `exp(bound·σ(exp(A_log)·(a+dt_bias)))`, an absent
        // key the plain `exp(-exp(A_log)·softplus(·))` (which the shared
        // GatedDeltaNet chain already serves, elementwise, so a per-channel gate
        // needs nothing new). A bound of exactly 0 would degenerate the bounded
        // form to exp(0) = 1 — a gate that never forgets — so it is refused
        // rather than served as the other arm by accident.
        if (jsonField(cfg_obj, "kda_lower_bound")) |v| {
            if (v != .null) {
                const lb = try jsonFloat(v);
                if (lb >= 0.0) {
                    log.err("bailing_hybrid: kda_lower_bound must be negative (got {d})\n", .{lb});
                    return error.UnsupportedBailingConfig;
                }
                config.kda_gate_lower_bound = lb;
            }
        }
        // `kda_safe_gate` is a numerics detail of the reference's own kernel
        // launch, not a change of formula — both arms above are already
        // evaluated in f32 here, so it is read as satisfied and ignored.
        //
        // The KDA LoRA gate variants (f_a_proj/f_b_proj, g_a_proj/g_b_proj)
        // are a different weight layout; the shipped checkpoint sets
        // no_kda_lora. Refuse rather than fail with a MISSING WEIGHT crash —
        // and accept BOTH spellings, since a checkpoint that states only the
        // positive one otherwise slips straight through to that crash.
        if (jsonField(cfg_obj, "no_kda_lora")) |v| {
            if (v == .bool and !v.bool) {
                log.err("bailing_hybrid: low-rank KDA gates (no_kda_lora=false) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }
        if (jsonField(cfg_obj, "use_kda_lora")) |v| {
            if (v == .bool and v.bool) {
                log.err("bailing_hybrid: low-rank KDA gates (use_kda_lora=true) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }

        // MLA.
        if (jsonField(cfg_obj, "q_lora_rank")) |v| {
            if (v == .integer) config.mla_q_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "kv_lora_rank")) |v| {
            if (v == .integer) config.mla_kv_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_nope_head_dim")) |v| {
            if (v == .integer) config.mla_qk_nope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer) config.mla_qk_rope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "v_head_dim")) |v| {
            if (v == .integer) config.mla_v_head_dim = try jsonU32(v);
        }
        if (config.mla_v_head_dim == 0) config.mla_v_head_dim = config.head_dim;
        // `q_lora_rank: null` is a plain q_proj — a different weight layout,
        // now served (see `mlaHasQLora`). The KV latent has no such fallback.
        if (config.mla_kv_lora_rank == 0) {
            log.err("bailing_hybrid: kv_lora_rank is required\n", .{});
            return error.UnsupportedBailingConfig;
        }
        // The declared qk_head_dim must agree with nope+rope: everything
        // downstream (the cached K's last dim, the attention scale, the q_b
        // split) is derived from the two halves.
        const qk_head_dim = std.math.add(u32, config.mla_qk_nope_head_dim, config.mla_qk_rope_head_dim) catch return error.InvalidConfigField;
        if (jsonField(cfg_obj, "qk_head_dim")) |v| {
            if (v == .integer and try jsonU32(v) != qk_head_dim) {
                log.err("bailing_hybrid: qk_head_dim {d} != qk_nope_head_dim + qk_rope_head_dim ({d})\n", .{ v.integer, qk_head_dim });
                return error.UnsupportedBailingConfig;
            }
        }
        // Attention scale is 1/sqrt(qk_head_dim) — the FULL query width,
        // wider than head_dim. Not derivable from head_dim on this arch.
        config.query_pre_attn_scalar = qk_head_dim;
        if (jsonField(cfg_obj, "gated_attention_proj_granularity_type")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "head_wise")) {
                    config.mla_head_gate = true;
                } else {
                    // element_wise gating is a differently-shaped g_proj.
                    log.err("bailing_hybrid: only head_wise attention gating supported (got '{s}')\n", .{v.string});
                    return error.UnsupportedBailingConfig;
                }
            }
        }
        if (jsonField(cfg_obj, "rope_interleave")) |v| {
            if (v == .bool) config.rope_interleaved_pairs = v.bool;
        }
        // `use_mla_nope` makes the MLA layers positionless (Kimi-Linear ships
        // exactly that: `rotary_emb=None`). `mlaAttnWith` always ropes the rope
        // slice, so a NoPE checkpoint would be served with positions its
        // reference never applies — refuse by name instead.
        if (jsonField(cfg_obj, "use_mla_nope")) |v| {
            if (v == .bool and v.bool) {
                log.err("bailing_hybrid: NoPE MLA (use_mla_nope=true) not supported\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }
        // partial_rotary_factor is stated against head_dim but the reference's
        // rotary module overrides it to 1.0 over qk_rope_head_dim — rope covers
        // that slice ENTIRELY, so the generic partial factor must not leak in.
        config.partial_rotary_factor = 1.0;

        // Three optional norms the forward does NOT implement. Each is a real
        // BailingMoeV3 switch and each is FALSE in every shipped checkpoint, so
        // they cost nothing here — but a variant flipping one would be served
        // silently without it, which is the failure mode this whole block of
        // named refusals exists to prevent.
        const unsupported_flags = [_][]const u8{ "value_norm", "up_proj_norm", "use_nGPT" };
        for (unsupported_flags) |key| {
            if (jsonField(cfg_obj, key)) |v| {
                if (v == .bool and v.bool) {
                    log.err("bailing_hybrid: {s}=true not supported\n", .{key});
                    return error.UnsupportedBailingConfig;
                }
            }
        }
        // The KDA conv activation. True everywhere shipped; the shared
        // GatedDeltaNet path applies silu after the causal conv unconditionally,
        // so a checkpoint declaring otherwise would get an activation it never
        // trained with.
        if (jsonField(cfg_obj, "linear_silu")) |v| {
            if (v == .bool and !v.bool) {
                log.err("bailing_hybrid: linear_silu=false not supported (the conv activation is silu)\n", .{});
                return error.UnsupportedBailingConfig;
            }
        }

        // MoE: grouped sigmoid routing (noaux_tc) + one ungated shared expert.
        config.moe_sigmoid_router = true;
        if (jsonField(cfg_obj, "first_k_dense_replace")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| {
            if (v == .integer) config.moe_n_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "topk_group")) |v| {
            if (v == .integer) config.moe_topk_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        // Shared expert width = num_shared_experts × its own intermediate size
        // (which falls back to the routed expert width when absent).
        if (config.shared_expert_intermediate_size == 0) {
            var shared_width: u32 = config.moe_intermediate_size;
            if (jsonField(cfg_obj, "moe_shared_expert_intermediate_size")) |v| {
                if (v == .integer) shared_width = try jsonU32(v);
            }
            var n_shared: u32 = 0;
            if (jsonField(cfg_obj, "num_shared_experts")) |v| {
                if (v == .integer) n_shared = try jsonU32(v);
            }
            config.shared_expert_intermediate_size = std.math.mul(u32, shared_width, n_shared) catch return error.InvalidConfigField;
        }
        // Softmax routing is a different score function; only sigmoid ships.
        if (jsonField(cfg_obj, "score_function")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) {
                log.err("bailing_hybrid: only sigmoid score_function supported (got '{s}')\n", .{v.string});
                return error.UnsupportedBailingConfig;
            }
        }
        // The MTP head ships disabled (num_nextn_predict_layers 0) and no
        // mtp.* weights are in the checkpoint. The single terminator
        // (`<|role_end|>`) rides the root eos_token_id — no additive merge.
    } else if (std.mem.eql(u8, model_type, "laguna")) {
        // poolside Laguna S 2.1 (117.6B-A8.5B MoE coder; nvfp4 experts, 256K
        // ctx). Pure-attention MoE that rides the qwen3.5/hy_v3 MoE forward
        // arms. Family-specific pieces: (1) per-layer Q-head counts
        // (num_attention_heads_per_layer: 48 full / 72 sliding; KV uniform 8),
        // (2) a softplus per-head attention OUTPUT gate (self_attn.g_proj), and
        // (3) YaRN RoPE on full-attention layers (theta 5e5, factor 32) with
        // default RoPE (theta 1e4, full rotary) on the 512-window sliding
        // layers. MoE routing is DeepSeek-V3 SIGMOID + expert bias
        // (mlp.gate.e_score_correction_bias) exactly like hy_v3, but the weight
        // NAMING matches qwen3_moe (mlp.gate router, mlp.switch_mlp experts,
        // mlp.shared_expert UNGATED always-added). Dense MLP on mlp_only_layers
        // (layer 0 on the shipped checkpoint), resolved by per-layer weight
        // presence probe at load. Reference: modeling_laguna.py (Apache-2.0).
        config.model_type = "laguna";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.laguna_attn_gate = true;
        config.moe_sigmoid_router = true;
        config.rope_scaling_factor = 1.0;
        // scale = head_dim^-0.5 (query_pre_attn_scalar absent in Laguna config).
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        // Router: norm_topk_prob → route_norm; moe_routed_scaling_factor → scale.
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "moe_routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "moe_router_logit_softcapping")) |v| config.moe_router_logit_softcapping = try jsonFloat(v);
        // Router logit soft-capping is off on the shipped checkpoint and untested;
        // reject a >0 value rather than silently ignore it (honest reject).
        if (config.moe_router_logit_softcapping > 0.0) {
            log.err("laguna: moe_router_logit_softcapping > 0 not supported in v1 (got {d})\n", .{config.moe_router_logit_softcapping});
            return error.UnsupportedLagunaConfig;
        }
        // Only per-head gating is implemented (assert uniform; honest reject).
        if (jsonField(cfg_obj, "gating")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "per-head") and !std.mem.eql(u8, v.string, "per_head")) {
                log.err("laguna: only per-head attention gating supported (got '{s}')\n", .{v.string});
                return error.UnsupportedLagunaConfig;
            }
        }
        // Per-layer Q-head count (cap 128 like layer_is_global).
        if (jsonField(cfg_obj, "num_attention_heads_per_layer")) |v| {
            if (v == .array) {
                config.has_per_layer_heads = true;
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.num_attention_heads_per_layer[i] = try jsonU32(item);
                }
            }
        }
        // YaRN on full-attention layers. The generic nested-rope block above
        // already set rope_theta (5e5), partial_rotary_factor_global (0.5) from
        // full_attention and rope_local_base_freq (1e4) from sliding_attention;
        // here we pull the YaRN-specific fields and flag the precompute.
        if (jsonField(cfg_obj, "rope_parameters")) |rp| {
            if (rp == .object) {
                if (jsonField(rp.object, "full_attention")) |fa_val| {
                    if (fa_val == .object) {
                        const fa = fa_val.object;
                        const is_yarn = if (jsonField(fa, "rope_type")) |rt|
                            (rt == .string and std.mem.eql(u8, rt.string, "yarn"))
                        else
                            false;
                        if (is_yarn) {
                            config.rope_yarn = true;
                            if (jsonField(fa, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                            if (jsonField(fa, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                            if (jsonField(fa, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                            // mscale is COMPUTED, never read from the config's
                            // "attention_factor". Both vendored MLX Laguna
                            // implementations drop that field and take MLX's
                            // YaRN default (mscale 1 / mscale_all_dim 0 =>
                            // 0.1*ln(factor) + 1); poolside's fused kernel
                            // hardcodes the result. S ships the computed value
                            // literally, XS ships 1.0 — honouring the field
                            // would run XS's full-attention layers unscaled.
                            // Generic YaRN readers still honour it; only this
                            // arch pins the value the checkpoint was trained on.
                            if (config.yarn_factor > 1.0) {
                                config.yarn_attention_factor = 0.1 * @log(config.yarn_factor) + 1.0;
                            }
                            if (jsonField(fa, "original_max_position_embeddings")) |x| {
                                if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                            }
                        }
                    }
                }
            }
        }
        // eos [2, 24] (〈|EOS|〉, </assistant>) parsed generically from
        // eos_token_id above; no additive terminator merge needed.
    } else if (std.mem.eql(u8, model_type, "kolibri1")) {
        // Aleph Alpha Kolibri-1: the laguna attention arm minus the output gate, with NoPE
        // full-attention layers, sandwich norms and an ungated shared expert on every layer.
        // Reference: Aleph-Alpha/aleph-alpha-inference kolibri1.py.
        config.model_type = "kolibri1";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.moe_sigmoid_router = true;
        config.moe_logit_select = true;
        config.moe_route_norm = false;
        config.router_scaling_factor = 1.0;
        config.rope_scaling_factor = 1.0;
        config.query_pre_attn_scalar = config.head_dim;
        // Sliding layers rotate at rope_theta (the config has no local base).
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool and v.bool) {
                log.err("kolibri1: norm_topk_prob=true not supported\n", .{});
                return error.UnsupportedKolibriConfig;
            }
        }
        // Full-attention layers carry no positional encoding.
        if (config.has_explicit_layer_types) {
            for (0..@min(config.num_hidden_layers, 128)) |i| config.layer_no_rope[i] = config.layer_is_global[i];
        }
    } else if (std.mem.eql(u8, model_type, "flint")) {
        // Recursive Flint: the laguna attention arm without the output gate, plus learned
        // per-head sinks (`self_attn.softmax_offset`); laguna's MoE (sigmoid top-k, renormed,
        // ungated shared expert, dense `mlp_only_layers`). Reference: recursive-oss/rec rec_mlx/model.py.
        config.model_type = "flint";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_attn_sinks = true;
        config.moe_sigmoid_router = true;
        config.moe_route_norm = true;
        config.router_scaling_factor = 1.0;
        config.rope_scaling_factor = 1.0;
        config.query_pre_attn_scalar = config.head_dim;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "no_rope_layer_indices")) |v| {
            if (v == .array) for (v.array.items) |item| {
                const i = try jsonU32(item);
                if (i < 128) config.layer_no_rope[i] = true;
            };
        }
    } else if (std.mem.eql(u8, model_type, "inkling_mm_model")) {
        // Thinking Machines Inkling Small (276B-A12B MoE, natively multimodal;
        // REAP builds prune n_routed_experts). NO RoPE anywhere: position =
        // RelativeLogits bias + 4 short convs/layer + log-scaling on global
        // layers. Per-head q/k RMSNorm with scale 1/head_dim; hybrid
        // sliding(512)/global from local_layer_ids; dense SwiGLU bottom layers
        // then sigmoid-routed MoE whose selected+shared logits share one
        // logsigmoid-softmax (the shared-expert "sink"); untied quantized
        // embed/unembed with muP logit scaling and a padded vocab. Reference:
        // the checkpoint's bundled inkling_mlx/ (Apache-2.0, parity-validated).
        config.model_type = "inkling_mm_model";
        config.weight_prefix = "model.llm";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.rope_scaling_factor = 1.0;
        // q/k are per-head RMS-normalized → scale = 1/head_dim, expressed via
        // the shared 1/sqrt(query_pre_attn_scalar) convention.
        config.query_pre_attn_scalar = std.math.mul(u32, config.head_dim, config.head_dim) catch return error.InvalidConfigField;
        // The checkpoint labels the MoE expert width `intermediate_size` (read
        // by the generic block above) and the dense bottom-layer width
        // `dense_intermediate_size` — opposite of our field meanings. Swap.
        config.moe_intermediate_size = config.intermediate_size;
        if (jsonField(cfg_obj, "dense_intermediate_size")) |v| {
            if (v == .integer) {
                config.intermediate_size = try jsonU32(v);
                config.intermediate_size_declared = true;
            }
        }
        if (jsonField(cfg_obj, "dense_mlp_idx")) |v| {
            if (v == .integer) config.first_k_dense_replace = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer) config.inkling_n_shared_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "route_scale")) |v| config.router_scaling_factor = try jsonFloat(v);
        // Position machinery.
        if (jsonField(cfg_obj, "d_rel")) |v| {
            if (v == .integer) config.inkling_d_rel = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "rel_extent")) |v| {
            if (v == .integer) config.inkling_rel_extent = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "log_scaling_n_floor")) |v| {
            if (v == .integer) config.inkling_log_n_floor = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "log_scaling_alpha")) |v| config.inkling_log_alpha = try jsonFloat(v);
        if (jsonField(cfg_obj, "sconv_kernel_size")) |v| {
            if (v == .integer) config.inkling_sconv_kernel = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "use_sconv")) |v| {
            if (v == .bool and !v.bool) config.inkling_sconv_kernel = 0;
        }
        // Embedding norm (use_embed_norm, default true for this family).
        config.has_embedding_norm = true;
        if (jsonField(cfg_obj, "use_embed_norm")) |v| {
            if (v == .bool) config.has_embedding_norm = v.bool;
        }
        // Hybrid sliding/global: the config names LOCAL (sliding) layers and
        // uses `sliding_window_size` (the generic block reads `sliding_window`).
        if (jsonField(cfg_obj, "sliding_window_size")) |v| {
            if (v == .integer) {
                config.sliding_window = try jsonU32(v);
                config.has_sliding_window = true;
            }
        }
        if (jsonField(cfg_obj, "local_layer_ids")) |v| {
            if (v == .array) {
                config.has_explicit_layer_types = true;
                for (config.layer_is_global[0..@min(config.num_hidden_layers, 128)]) |*g| g.* = true;
                for (v.array.items) |item| {
                    if (item == .integer and item.integer >= 0 and item.integer < 128) {
                        config.layer_is_global[@intCast(item.integer)] = false;
                    }
                }
            }
        }
        // muP logits + padded vocab.
        if (jsonField(cfg_obj, "logits_mup_width_multiplier")) |v| config.logits_mup_width_multiplier = try jsonFloat(v);
        if (jsonField(cfg_obj, "unpadded_vocab_size")) |v| {
            if (v == .integer) config.unpadded_vocab_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "model_max_length")) |v| {
            if (v == .integer) config.max_position_embeddings = try jsonU32(v);
        }
        // v1 is text-only: the hMLP vision_config must not arm the SigLIP path
        // (the generic vision_config block above set has_vision = true).
        config.has_vision = false;
        // Honest rejects: the forward implements exactly the shipped geometry
        // and router formula. A checkpoint that diverges must refuse to load,
        // not run silently wrong.
        const swa_heads: u32 = if (jsonField(cfg_obj, "swa_num_attention_heads")) |v| try jsonU32(v) else config.num_attention_heads;
        const swa_kv: u32 = if (jsonField(cfg_obj, "swa_num_key_value_heads")) |v| try jsonU32(v) else config.num_key_value_heads;
        const swa_hd: u32 = if (jsonField(cfg_obj, "swa_head_dim")) |v| try jsonU32(v) else config.head_dim;
        if (swa_heads != config.num_attention_heads or swa_kv != config.num_key_value_heads or swa_hd != config.head_dim) {
            log.err("inkling: sliding-attention geometry {d}/{d}/{d} differs from global {d}/{d}/{d} — not supported\n", .{ swa_heads, swa_kv, swa_hd, config.num_attention_heads, config.num_key_value_heads, config.head_dim });
            return error.UnsupportedInklingConfig;
        }
        if (jsonField(cfg_obj, "gate_activation")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sigmoid")) {
                log.err("inkling: gate_activation '{s}' not supported (sigmoid only)\n", .{v.string});
                return error.UnsupportedInklingConfig;
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4") or std.mem.eql(u8, model_type, "deepseek_v41")) {
        // DeepSeek V4 Flash (284B-A13B, 1M ctx). See the dsv4_* field block
        // for the architecture summary; reference is the release's own
        // inference/{model,kernel}.py (torch). Loaded from OUR converted
        // mixed-quant mirror (tests/convert_dsv4_weights.py) — bare
        // inference-style tensor names, stacked expert banks. V4.1 shares
        // these fields and adds the dsv41_* block below.
        const is_v41 = std.mem.eql(u8, model_type, "deepseek_v41");
        config.model_type = if (is_v41) "deepseek_v41" else "deepseek_v4";
        config.legacy.caps.prefix_cache = false;
        if (is_v41) config.legacy.thinking = .on;
        config.weight_prefix = ""; // release ships bare names (embed.weight, layers.N....)
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false; // q-norm is on the lora rank + unweighted per-head RMS, handled in-arch
        config.hidden_act = .silu;
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "num_hash_layers")) |v| {
            if (v == .integer) config.dsv4_hash_layers = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "q_lora_rank")) |v| {
            if (v == .integer) config.dsv4_q_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "o_lora_rank")) |v| {
            if (v == .integer) config.dsv4_o_lora_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "o_groups")) |v| {
            if (v == .integer) config.dsv4_o_groups = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "qk_rope_head_dim")) |v| {
            if (v == .integer) config.dsv4_rope_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_n_heads")) |v| {
            if (v == .integer) config.dsv4_index_n_heads = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_head_dim")) |v| {
            if (v == .integer) config.dsv4_index_head_dim = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "index_topk")) |v| {
            if (v == .integer) config.dsv4_index_topk = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_mult")) |v| {
            if (v == .integer) config.dsv4_hc_mult = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_sinkhorn_iters")) |v| {
            if (v == .integer) config.dsv4_hc_sinkhorn_iters = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "hc_eps")) |v| config.dsv4_hc_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "swiglu_limit")) |v| config.dsv4_swiglu_limit = try jsonFloat(v);
        if (jsonField(cfg_obj, "compress_rope_theta")) |v| config.dsv4_compress_rope_theta = try jsonFloat(v);
        if (jsonField(cfg_obj, "num_nextn_predict_layers")) |v| {
            if (v == .integer) config.dsv4_mtp_layers = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_block_size")) |v| {
            if (v == .integer) config.dsv4_dspark_block_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_noise_token_id")) |v| {
            if (v == .integer) config.dsv4_dspark_noise_token_id = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_markov_rank")) |v| {
            if (v == .integer) config.dsv4_dspark_markov_rank = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "dspark_target_layer_ids")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= config.dsv4_dspark_target_layers.len) break;
                    if (item == .integer) config.dsv4_dspark_target_layers[i] = std.math.cast(u8, try jsonU32(item)) orelse return error.InvalidConfigField;
                }
                config.dsv4_n_dspark_target_layers = @intCast(@min(v.array.items.len, config.dsv4_dspark_target_layers.len));
            }
        }
        if (jsonField(cfg_obj, "compress_ratios")) |v| {
            if (v == .array) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item == .integer) config.dsv4_compress_ratios[i] = std.math.cast(u8, try jsonU32(item)) orelse return error.InvalidConfigField;
                }
                config.dsv4_n_compress_ratios = @intCast(@min(v.array.items.len, 128));
            }
        }
        // YaRN on compressed layers only; the reference applies NO mscale
        // (softmax scale stays head_dim^-0.5 everywhere), so
        // yarn_attention_factor stays 1.0 — do not compute the 0.1·ln(f)+1
        // default here (laguna-class trap in the other direction).
        if (jsonField(cfg_obj, "rope_scaling")) |rs| {
            if (rs == .object) {
                config.rope_yarn = true;
                if (jsonField(rs.object, "factor")) |x| config.yarn_factor = try jsonFloat(x);
                if (jsonField(rs.object, "beta_fast")) |x| config.yarn_beta_fast = try jsonFloat(x);
                if (jsonField(rs.object, "beta_slow")) |x| config.yarn_beta_slow = try jsonFloat(x);
                if (jsonField(rs.object, "original_max_position_embeddings")) |x| {
                    if (x == .integer) config.yarn_orig_max_pos = try jsonU32(x);
                }
            }
        }
        // Honest rejects: the forward implements exactly sqrt(softplus)
        // scoring with selection-only bias (noaux_tc), ONE always-on shared
        // expert, and a single shared KV latent. Divergent checkpoints must
        // refuse to load, not run silently wrong.
        if (jsonField(cfg_obj, "scoring_func")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "sqrtsoftplus")) {
                log.err("{s}: scoring_func '{s}' not supported (sqrtsoftplus only)\n", .{ config.model_type, v.string });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (jsonField(cfg_obj, "topk_method")) |v| {
            if (v == .string and !std.mem.eql(u8, v.string, "noaux_tc")) {
                log.err("{s}: topk_method '{s}' not supported (noaux_tc only)\n", .{ config.model_type, v.string });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (jsonField(cfg_obj, "n_shared_experts")) |v| {
            if (v == .integer and v.integer != 1) {
                log.err("{s}: n_shared_experts {d} not supported (exactly 1)\n", .{ config.model_type, v.integer });
                return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
            }
        }
        if (config.num_key_value_heads != 1) {
            log.err("{s}: num_key_value_heads {d} not supported (single shared KV latent)\n", .{ config.model_type, config.num_key_value_heads });
            return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
        }
        // The July-31 release supersedes the preview, and the preview's
        // single next-token MTP module is no longer supported — its draft
        // path (e_proj/h_proj over one stage) shares nothing with DSpark's
        // block-parallel stages beyond the `mtp.*` namespace, so carrying it
        // would mean maintaining a second architecture for a checkpoint the
        // vendor withdrew. A preview config announces itself by declaring MTP
        // layers with no DSpark descriptor; say so instead of loading a model
        // whose draft weights we would silently ignore.
        if (config.dsv4_mtp_layers > 0 and config.dsv4_dspark_block_size == 0) {
            log.err("{s}: this is the superseded PREVIEW checkpoint (num_nextn_predict_layers={d}, no dspark_* config). Use DeepSeek-V4-Flash-0731 or later.\n", .{ config.model_type, config.dsv4_mtp_layers });
            return if (is_v41) error.UnsupportedDsv41Config else error.UnsupportedDsv4Config;
        }
        if (is_v41) {
            // The release's tower (src/plugins/deepseek_v41/vision.zig) splices at `<｜deepseek_image｜>`: a pack
            // without `vision_config` or the image token serves text.
            config.dsv41_vision = config.has_vision and config.image_token_id != 0;
            config.has_vision = config.dsv41_vision;
            if (config.dsv41_vision) {
                const vc = jsonField(root, "vision_config").?.object;
                config.vision_rope_theta = 10000.0;
                if (jsonField(vc, "rope_theta")) |v| config.vision_rope_theta = try jsonFloat(v);
                if (jsonField(vc, "downsample_ratio")) |v| config.dv_downsample = try jsonU32(v);
                if (jsonField(vc, "max_image_tokens")) |v| config.dv_max_image_tokens = try jsonU32(v);
                if (jsonField(vc, "min_pixels")) |v| config.dv_min_pixels = try jsonU32(v);
                if (jsonField(vc, "max_wh_ratio")) |v| {
                    if (v == .integer) config.dv_max_wh_ratio = try jsonU32(v);
                }
                // The 2-D rotary splits a head into h and w halves of rotate_half pairs.
                if (config.vision_num_heads == 0 or config.vision_hidden_size % config.vision_num_heads != 0 or
                    (config.vision_hidden_size / config.vision_num_heads) % 4 != 0 or config.vision_patch_size == 0 or
                    config.dv_downsample == 0 or config.dv_max_image_tokens < 4) return error.InvalidConfigField;
            }
            try parseDsv41Fields(cfg_obj, &config);
        }
    } else if (std.mem.eql(u8, model_type, "glm_moe_dsa")) {
        // Z.ai GLM-5.3 (glm_moe_dsa), served only by mlx-stream (`parseConfig`). The
        // host reads the shell fields the common block parsed: vocab, hidden, layers,
        // eos ids, context, untied head.
        config.model_type = "glm_moe_dsa";
        config.legacy.thinking = .on;
        config.has_sliding_window = false;
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
    } else if (std.mem.eql(u8, model_type, "qwen3_next")) {
        config.model_type = "qwen3_next";
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (jsonField(cfg_obj, "partial_rotary_factor")) |v| config.partial_rotary_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
    } else if (std.mem.eql(u8, model_type, "lfm2") or std.mem.startsWith(u8, model_type, "lfm2")) {
        config.model_type = "lfm2";
        // VL variant nests text weights under language_model.model (like Gemma 4)
        config.weight_prefix = if (jsonField(root, "text_config") != null) "language_model.model" else "model";
        config.hidden_act = .silu;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_embedding_norm = false;
        config.has_final_norm = true; // embedding_norm IS the final norm
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (config.head_dim == 256) { // default from gemma3, override
            if (config.num_attention_heads == 0) return error.InvalidConfigField;
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        config.query_pre_attn_scalar = config.head_dim;
        // tie_embedding (LFM2 name) -> tie_word_embeddings; default true for LFM2
        config.tie_word_embeddings = true;
        if (jsonField(cfg_obj, "tie_embedding")) |v| {
            if (v == .bool) config.tie_word_embeddings = v.bool;
        }
        if (jsonField(cfg_obj, "norm_eps")) |v| config.rms_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "conv_L_cache")) |v| {
            if (v == .integer) config.lfm_conv_kernel = try jsonU32(v);
        }
        if (std.mem.eql(u8, model_type, "lfm2_moe")) {
            config.lfm2_moe = true;
            if (jsonField(cfg_obj, "num_experts")) |v| {
                if (v == .integer) config.num_experts = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "num_experts_per_tok")) |v| {
                if (v == .integer) config.num_experts_per_tok = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "moe_intermediate_size")) |v| {
                if (v == .integer) config.moe_intermediate_size = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "num_dense_layers")) |v| {
                if (v == .integer) config.num_dense_layers = try jsonU32(v);
            }
            if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
                if (v == .bool) config.moe_route_norm = v.bool;
            }
            if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
            if (config.num_experts == 0 or config.num_experts_per_tok == 0 or config.moe_intermediate_size == 0) {
                return error.IncompleteLfm2MoeConfig;
            }
        }
        if (jsonField(cfg_obj, "conv_dim")) |v| config.lfm_conv_dim = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        // Parse layer_types array: ["conv", "full_attention", ...]
        if (jsonField(cfg_obj, "layer_types")) |lt_val| {
            if (lt_val == .array) {
                for (lt_val.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item != .string) continue;
                    config.layer_block_types[i] = if (std.mem.eql(u8, item.string, "conv"))
                        .gated_conv
                    else
                        .attention;
                }
            }
        }
        if (config.num_eos_tokens == 0) {
            if (jsonField(cfg_obj, "eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(try jsonU32(v));
            }
        }
        // LFM2-VL: a stock `siglip2_vision_model` tower (src/plugins/lfm2/vision.zig)
        // plus LFM2-VL's own projector. The generic vision_config block above
        // already read the tower's geometry; everything here is the wrapper.
        // A `lfm2` checkpoint with no vision_config stays text-only.
        if (jsonField(root, "vision_config")) |vc_val| {
            if (vc_val == .object and std.mem.eql(u8, model_type, "lfm2_vl")) {
                config.lfm2_vision = true;
                const vc = vc_val.object;
                config.lv_ln_eps = 1e-6;
                if (jsonField(vc, "layer_norm_eps")) |v| config.lv_ln_eps = try jsonFloat(v);
                // The stored table is square: num_patches = pos_side².
                var num_patches: u32 = 256;
                if (jsonField(vc, "num_patches")) |v| {
                    if (v == .integer) num_patches = try jsonU32(v);
                }
                config.lv_pos_side = std.math.sqrt(num_patches);
                if (jsonField(root, "downsample_factor")) |v| {
                    if (v == .integer) config.lv_downsample = try jsonU32(v);
                }
                if (jsonField(root, "projector_hidden_size")) |v| {
                    if (v == .integer) config.lv_projector_hidden = try jsonU32(v);
                }
                if (jsonField(root, "min_image_tokens")) |v| {
                    if (v == .integer) config.lv_min_image_tokens = try jsonU32(v);
                }
                if (jsonField(root, "max_image_tokens")) |v| {
                    if (v == .integer) config.lv_max_image_tokens = try jsonU32(v);
                }
                if (jsonField(root, "tile_size")) |v| {
                    if (v == .integer) config.lv_tile_size = try jsonU32(v);
                }
                if (jsonField(root, "min_tiles")) |v| {
                    if (v == .integer) config.lv_min_tiles = try jsonU32(v);
                }
                if (jsonField(root, "max_tiles")) |v| {
                    if (v == .integer) config.lv_max_tiles = try jsonU32(v);
                }
                if (jsonField(root, "do_image_splitting")) |v| {
                    if (v == .bool) config.lv_split_images = v.bool;
                }
                if (jsonField(root, "use_thumbnail")) |v| {
                    if (v == .bool) config.lv_use_thumbnail = v.bool;
                }
                if (jsonField(root, "max_pixels_tolerance")) |v| config.lv_pixels_tolerance = try jsonFloat(v);
                if (config.lv_projector_hidden == 0) config.lv_projector_hidden = config.hidden_size;
            } else {
                // `vision_config` present without the VL tag (mlx-community's
                // text-only LFM2.5 packs ship an EMPTY one): never advertise a
                // tower we have no weights for.
                config.has_vision = false;
            }
        }
    } else if (std.mem.eql(u8, model_type, "nemotron_h") or std.mem.eql(u8, model_type, "nemotron_h_puzzle")) {
        // Puzzle is the heterogeneous latent-MoE trunk. The runtime still
        // keys every kernel and weight prefix on "nemotron_h".
        config.model_type = "nemotron_h";
        config.weight_prefix = "backbone";
        config.radio_vision = model_discovery.hasRadioVision(root);
        config.has_vision = config.radio_vision;
        if (config.radio_vision) {
            const vc = jsonField(root, "vision_config").?.object;
            if (jsonField(vc, "num_attention_heads")) |v| config.radio_num_heads = try jsonU32(v);
            if (jsonField(cfg_obj, "num_nextn_predict_layers")) |v| config.radio_final_layernorm = (try jsonU32(v)) > 0;
            if (jsonField(vc, "layer_norm_eps")) |v| config.radio_ln_eps = try jsonFloat(v);
            if (config.radio_ln_eps <= 0) return error.InvalidConfigField;
            config.video_pruning_rate = RADIO_VIDEO_PRUNING_RATE_DEFAULT;
            if (jsonField(root, "video_pruning_rate")) |v| {
                config.video_pruning_rate = try jsonFloat64(v);
            } else if (jsonField(vc, "video_pruning_rate")) |v| {
                config.video_pruning_rate = try jsonFloat64(v);
            }
            if (config.video_pruning_rate < 0 or config.video_pruning_rate > 1) return error.InvalidConfigField;
        }
        config.hidden_act = .silu;
        config.mamba_mlp_act = .relu_sq;
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.has_sliding_window = false;
        config.has_hybrid_layers = true;
        config.has_final_norm = true;
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        config.query_pre_attn_scalar = config.head_dim;
        // NoPE: the reference attention never rotates q/k, whatever
        // rope_theta the config carries. Covers the MTP head's layer too.
        config.layer_no_rope = @splat(true);
        // Bolt's llm_config omits num_hidden_layers; the block list is the depth.
        if (jsonField(cfg_obj, "num_hidden_layers") == null) {
            const listed: usize = if (jsonField(cfg_obj, "layers_block_type")) |v|
                (if (v == .array) v.array.items.len else 0)
            else if (jsonField(cfg_obj, "block_configs")) |v|
                (if (v == .array) v.array.items.len else 0)
            else if (jsonField(cfg_obj, "hybrid_override_pattern")) |v|
                (if (v == .string) v.string.len else 0)
            else
                0;
            if (listed > config.layer_block_types.len) return error.UnsupportedLayerGeometry;
            if (listed > 0) config.num_hidden_layers = @intCast(listed);
        }
        if (jsonField(cfg_obj, "rms_norm_eps")) |v| {
            config.rms_norm_eps = try jsonFloat(v);
        } else if (jsonField(cfg_obj, "layer_norm_epsilon")) |v| {
            config.rms_norm_eps = try jsonFloat(v);
        }
        // Mamba2-specific config
        if (jsonField(cfg_obj, "mamba_num_heads")) |v| config.mamba_num_heads = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        if (jsonField(cfg_obj, "mamba_head_dim")) |v| config.mamba_head_dim = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 0,
        };
        if (jsonField(cfg_obj, "n_groups")) |v| config.mamba_n_groups = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 8,
        };
        if (jsonField(cfg_obj, "ssm_state_size")) |v| config.ssm_state_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 128,
        };
        if (jsonField(cfg_obj, "conv_kernel")) |v| config.mamba_conv_kernel = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 4,
        };
        if (jsonField(cfg_obj, "expand")) |v| config.mamba_expand = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 2,
        };
        // time_step_limit: Python defaults to (0.0, inf) if not in config.
        // config.json may have time_step_min/time_step_max fields but Python ignores them
        // for SSM clipping — only time_step_limit (a 2-element array) is used.
        if (jsonField(cfg_obj, "time_step_limit")) |v| {
            if (v != .array or v.array.items.len != 2) return error.InvalidConfigField;
            const items = v.array.items;
            config.time_step_min = try jsonSsmBound(items[0], false);
            config.time_step_max = try jsonSsmBound(items[1], true);
            if (config.time_step_min < 0 or config.time_step_max < config.time_step_min) return error.InvalidConfigField;
        }
        if (jsonField(cfg_obj, "chunk_size")) |v| config.mamba_chunk_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 256,
        };
        // Parse hybrid_override_pattern: "M-M-M-MM-M-M*-..."
        if (jsonField(cfg_obj, "hybrid_override_pattern")) |v| {
            if (v == .string) {
                for (v.string, 0..) |ch, i| {
                    if (i >= 128) break;
                    config.layer_block_types[i] = switch (ch) {
                        'M' => .mamba2,
                        '-' => .mlp,
                        '*' => .attention,
                        'E' => .moe,
                        else => .attention,
                    };
                }
            }
        }
        // Nemotron 3.5 configs spell the same pattern as a list of names; the
        // string wins when both are present (mlx-lm's order).
        if (jsonField(cfg_obj, "layers_block_type")) |v| {
            if (v == .array and jsonField(cfg_obj, "hybrid_override_pattern") == null) {
                for (v.array.items, 0..) |item, i| {
                    if (i >= 128) break;
                    if (item != .string) continue;
                    const name = item.string;
                    config.layer_block_types[i] = if (std.mem.eql(u8, name, "mamba"))
                        .mamba2
                    else if (std.mem.eql(u8, name, "mlp"))
                        .mlp
                    else if (std.mem.eql(u8, name, "moe"))
                        .moe
                    else
                        .attention;
                }
            }
        }
        // MoE blocks ('E' / "moe"): sigmoid router with a selection-only
        // score-correction bias, ReLU^2 routed experts, one shared expert.
        if (jsonField(cfg_obj, "n_routed_experts")) |v| {
            if (v == .integer) config.num_experts = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "moe_shared_expert_intermediate_size")) |v| {
            if (v == .integer) config.shared_expert_intermediate_size = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "n_group")) |v| {
            if (v == .integer) config.moe_n_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "topk_group")) |v| {
            if (v == .integer) config.moe_topk_group = try jsonU32(v);
        }
        if (jsonField(cfg_obj, "norm_topk_prob")) |v| {
            if (v == .bool) config.moe_route_norm = v.bool;
        }
        if (jsonField(cfg_obj, "routed_scaling_factor")) |v| config.router_scaling_factor = try jsonFloat(v);
        if (jsonField(cfg_obj, "moe_latent_size")) |v| {
            if (v == .integer) config.moe_latent_size = try jsonU32(v);
        }
        // Per-layer expert banks. The forward reads each tensor's own shape;
        // these scalars are the maxima, for memory billing.
        if (jsonField(cfg_obj, "block_configs")) |v| {
            if (v == .array) {
                var max_experts: u32 = 0;
                var max_inter: u32 = 0;
                for (v.array.items) |item| {
                    if (item != .object) continue;
                    if (jsonField(item.object, "n_routed_experts")) |e| {
                        if (e == .integer) max_experts = @max(max_experts, try jsonU32(e));
                    }
                    if (jsonField(item.object, "moe_intermediate_size")) |e| {
                        if (e == .integer) max_inter = @max(max_inter, try jsonU32(e));
                    }
                    if (jsonField(item.object, "num_experts_per_tok")) |e| {
                        if (e == .integer) config.num_experts_per_tok = try jsonU32(e);
                    }
                }
                if (max_experts > 0) config.num_experts = max_experts;
                if (max_inter > 0) config.moe_intermediate_size = max_inter;
            }
        }
        if (config.num_eos_tokens == 0) {
            if (jsonField(cfg_obj, "eos_token_id")) |v| {
                if (v == .integer) config.addEosToken(try jsonU32(v));
            }
        }
    } else if (std.mem.eql(u8, model_type, "deepseek_v4")) {
        // MLX-format DSV4 is not supported in this build. Users should load
        // the GGUF checkpoint via the ds4 engine (`*.gguf` early-branch in
        // main.zig / Swift app). Fall through to the unknown-arch error
        // path so the failure message points at the right thing.
        return error.UnsupportedDsv4MlxFormat;
    } else if (std.mem.eql(u8, model_type, "bert")) {
        config.model_type = "bert";
        config.is_encoder_only = true;
        config.weight_prefix = "";
        config.hidden_act = .gelu_approx;
        config.tie_word_embeddings = true;
        config.has_sliding_window = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.scale_embeddings = false;
        config.norm_has_offset = false;
        if (config.num_attention_heads == 0) return error.InvalidConfigField;
        config.head_dim = config.hidden_size / config.num_attention_heads;
        config.num_key_value_heads = config.num_attention_heads;
        config.query_pre_attn_scalar = config.head_dim;
        if (jsonField(cfg_obj, "layer_norm_eps")) |v| config.layer_norm_eps = try jsonFloat(v);
        if (jsonField(cfg_obj, "type_vocab_size")) |v| config.type_vocab_size = switch (v) {
            .integer => |i| std.math.cast(u32, i) orelse return error.InvalidConfigField,
            else => 2,
        };
    } else {
        // Llama-family defaults (qwen3, llama, mistral, etc.)
        if (std.mem.eql(u8, model_type, "qwen3")) {
            config.model_type = "qwen3";
        } else if (std.mem.eql(u8, model_type, "qwen3_asr") or std.mem.eql(u8, model_type, "qwen3_forced_aligner")) {
            config.model_type = "qwen3";
            config.qwen_audio = try parseQwenAudioConfig(root);
            config.has_vision = true;
            if (std.mem.eql(u8, model_type, "qwen3_forced_aligner")) config.forced_aligner = try parseForcedAligner(outer);
        } else if (std.mem.eql(u8, model_type, "qwen2")) {
            // Qwen2.5 family: dense Llama-style attention, NO QK-norm (left
            // false below), additive qkv-projection biases applied in the
            // forward when present.
            config.model_type = "qwen2";
        } else if (std.mem.eql(u8, model_type, "llama")) {
            config.model_type = "llama";
        } else if (std.mem.eql(u8, model_type, "mistral")) {
            config.model_type = "mistral";
        } else if (std.mem.eql(u8, model_type, "k2_horizon")) {
            // IFM K2-Horizon dense (0.9B/3.7B/7B/32B): a Llama trunk whose
            // RMS norms are GROUPED (`layernorm_num_groups`). The MoVA MoE
            // sizes (`mova_num_experts` > 0) are a different attention and
            // are not served.
            config.model_type = "k2_horizon";
            config.legacy.thinking = .on;
            if (jsonField(cfg_obj, "layernorm_num_groups")) |v| {
                if (v == .integer and v.integer > 1) config.norm_groups = try jsonU32(v);
            }
        } else {
            config.model_type = "unknown";
        }
        config.weight_prefix = "model";
        config.norm_has_offset = false;
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = false;
        config.rope_scaling_factor = 1.0;
        config.rope_llama3 = try parseLlama3Rope(cfg_obj);
        config.rope_local_base_freq = config.rope_theta;
        // Every layer attends the whole context, as in mlx-lm's llama/qwen2/mistral:
        // the struct default is Gemma's 5-sliding-to-1-global pattern at 1024.
        config.has_sliding_window = false;
        // Llama-family models (qwen2, llama, mistral) usually omit `head_dim`;
        // the HF default is hidden_size / num_attention_heads. Without this the
        // stale 256 sentinel (line 53) would corrupt attention for any such
        // checkpoint that doesn't ship an explicit head_dim (e.g. Qwen2.5).
        // qwen3 ships an explicit head_dim, so this leaves it untouched.
        if (jsonField(cfg_obj, "head_dim") == null) {
            if (config.num_attention_heads == 0) return error.InvalidConfigField;
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        if (jsonField(cfg_obj, "hidden_act")) |v| {
            if (v == .string) {
                if (std.mem.eql(u8, v.string, "silu")) {
                    config.hidden_act = .silu;
                } else if (std.mem.eql(u8, v.string, "gelu_pytorch_tanh")) {
                    config.hidden_act = .gelu_approx;
                }
            }
        }

        if (jsonField(cfg_obj, "query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }

        if (std.mem.eql(u8, config.model_type, "qwen3")) {
            config.has_qk_norm = true;
        }
    }

    if (jsonField(cfg_obj, "linear_num_key_heads") != null or jsonField(cfg_obj, "linear_num_value_heads") != null or
        jsonField(cfg_obj, "linear_key_head_dim") != null or jsonField(cfg_obj, "linear_value_head_dim") != null or
        jsonField(cfg_obj, "linear_conv_kernel_dim") != null)
    {
        if (config.linear_num_key_heads == 0 or config.linear_num_value_heads == 0 or
            config.linear_num_value_heads % config.linear_num_key_heads != 0)
            return error.InvalidConfigField;
    }
    return config;
}

/// V4.1's fields beyond the shared dsv4 block, refused by name where the
/// forward could not serve them: every compressed layer must find a source of
/// its own ratio at or above it (the reference reads whichever source ran
/// last, so a mismatch would silently attend over the wrong cache).
fn parseDsv41Fields(cfg_obj: std.json.ObjectMap, config: *ModelConfig) !void {
    const refuse = struct {
        fn f(comptime fmt: []const u8, args: anytype) error{UnsupportedDsv41Config} {
            log.err("deepseek_v41: " ++ fmt ++ "\n", args);
            return error.UnsupportedDsv41Config;
        }
    }.f;
    const n_layers = config.num_hidden_layers;
    if (n_layers == 0 or n_layers > 64 or config.dsv4_n_compress_ratios < n_layers) return refuse("{d} layers with {d} compress ratios", .{ n_layers, config.dsv4_n_compress_ratios });
    if (config.dsv4_hc_mult != 4) return refuse("hc_mult {d} (only 4 is served)", .{config.dsv4_hc_mult});
    const layerMask = struct {
        fn f(obj: std.json.ObjectMap, key: []const u8, n: u32) !u64 {
            var mask: u64 = 0;
            const v = jsonField(obj, key) orelse return 0;
            for ((try jsonValue(.array, v)).items) |item| {
                const l = try jsonU32(item);
                if (l >= n) return refuse("{s} names layer {d} of {d}", .{ key, l, n });
                mask |= @as(u64, 1) << @intCast(l);
            }
            return mask;
        }
    }.f;
    config.dsv41_kv_sources = try layerMask(cfg_obj, "kv_source_layer_ids", n_layers);
    config.dsv41_index_sources = try layerMask(cfg_obj, "index_source_layer_ids", n_layers);
    if (jsonField(cfg_obj, "candidate_source_layer_id")) |v| config.dsv41_candidate_source = std.math.cast(i32, try jsonValue(.integer, v)) orelse return error.InvalidConfigField;
    if (jsonField(cfg_obj, "candidate_topk_blocks")) |v| config.dsv41_candidate_topk_blocks = try jsonU32(v);
    if (jsonField(cfg_obj, "candidate_block_size")) |v| config.dsv41_candidate_block_size = try jsonU32(v);

    const ratios = config.dsv4_compress_ratios[0..n_layers];
    const has = struct {
        fn f(mask: u64, l: usize) bool {
            return (mask >> @intCast(l)) & 1 != 0;
        }
    }.f;
    var last_kv: ?usize = null;
    var last_owner: ?usize = null;
    var last_index: ?usize = null;
    for (ratios, 0..) |r, l| {
        if (r > 2) return refuse("layer {d}: compress ratio {d} (modes 0, 1, 2 are served)", .{ l, r });
        const kv = has(config.dsv41_kv_sources, l);
        const ix = has(config.dsv41_index_sources, l);
        if (r == 0) {
            if (kv or ix) return refuse("layer {d}: a source without compression", .{l});
            continue;
        }
        if (kv) last_kv = l;
        if (kv and ix) last_owner = l;
        if (ix) last_index = l;
        const src = last_kv orelse return refuse("layer {d}: no KV source at or above it", .{l});
        if (ratios[src] != r) return refuse("layer {d} (ratio {d}) would read layer {d}'s ratio-{d} cache", .{ l, r, src, ratios[src] });
        if (ix) {
            const owner = last_owner orelse return refuse("layer {d}: no index-key owner at or above it", .{l});
            if (ratios[owner] != r) return refuse("layer {d} would score layer {d}'s keys", .{ l, owner });
        } else if (last_index == null) return refuse("layer {d}: no index source at or above it", .{l});
    }
    if (config.dsv41_candidate_source >= 0) {
        const c: usize = @intCast(config.dsv41_candidate_source);
        if (c >= n_layers or !has(config.dsv41_index_sources, c)) return refuse("candidate source {d} is not an index source", .{c});
        if (config.dsv41_candidate_topk_blocks == 0 or config.dsv41_candidate_block_size == 0) return refuse("candidate source without blocks", .{});
    }

    if (jsonField(cfg_obj, "engram_layer_ids")) |v| {
        const items = (try jsonValue(.array, v)).items;
        if (items.len > config.dsv41_engram_layers.len) return refuse("{d} engram layers", .{items.len});
        for (items, 0..) |item, i| {
            const l = try jsonU32(item);
            if (l >= n_layers) return refuse("engram layer {d} of {d}", .{ l, n_layers });
            config.dsv41_engram_layers[i] = @intCast(l);
        }
        config.dsv41_n_engram_layers = @intCast(items.len);
    }
    if (config.dsv41_n_engram_layers > 0) {
        const rows_v = jsonField(cfg_obj, "engram_num_embeddings") orelse return refuse("engram without engram_num_embeddings", .{});
        const rows = (try jsonValue(.array, rows_v)).items;
        if (rows.len != config.dsv41_n_engram_layers) return refuse("{d} engram tables for {d} layers", .{ rows.len, config.dsv41_n_engram_layers });
        for (rows, 0..) |item, i| config.dsv41_engram_rows[i] = std.math.cast(u64, try jsonValue(.integer, item)) orelse return error.InvalidConfigField;
        if (jsonField(cfg_obj, "engram_max_ngram_size")) |x| config.dsv41_engram_max_ngram = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_vocab_size")) |x| config.dsv41_engram_vocab_size = std.math.cast(u64, try jsonValue(.integer, x)) orelse return error.InvalidConfigField;
        if (jsonField(cfg_obj, "engram_n_heads")) |x| config.dsv41_engram_n_heads = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_head_dim")) |x| config.dsv41_engram_head_dim = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_pad_token_id")) |x| config.dsv41_engram_pad_id = try jsonU32(x);
        if (jsonField(cfg_obj, "engram_compressed_vocab_size")) |x| config.dsv41_engram_compressed_vocab = try jsonU32(x);
        if (config.dsv41_engram_max_ngram < 2 or config.dsv41_engram_max_ngram > 8 or
            config.dsv41_engram_n_heads == 0 or config.dsv41_engram_n_heads > 16 or
            config.dsv41_engram_head_dim == 0 or config.dsv41_engram_head_dim % 32 != 0 or
            config.dsv41_engram_compressed_vocab == 0 or config.dsv41_engram_vocab_size == 0)
            return refuse("engram geometry ({d}-grams, {d} heads x {d})", .{ config.dsv41_engram_max_ngram, config.dsv41_engram_n_heads, config.dsv41_engram_head_dim });
    }
    if (jsonField(cfg_obj, "dspark_n_routed_experts")) |v| config.dsv41_dspark_experts = try jsonU32(v);
    if (jsonField(cfg_obj, "dspark_num_experts_per_tok")) |v| config.dsv41_dspark_top_k = try jsonU32(v);
    for (config.dsv4_dspark_target_layers[0..config.dsv4_n_dspark_target_layers]) |l| {
        if (l >= n_layers) return refuse("DSpark target layer {d} of {d}", .{ l, n_layers });
    }
}

fn parseQwenAudioConfig(root: std.json.ObjectMap) !QwenAudioConfig {
    var qa = QwenAudioConfig{};
    const ac = jsonField(root, "audio_config") orelse return qa;
    const o = try jsonValue(.object, ac);
    inline for (.{
        .{ "d_model", "d_model" },     .{ "encoder_layers", "layers" },         .{ "encoder_attention_heads", "heads" },
        .{ "encoder_ffn_dim", "ffn" }, .{ "num_mel_bins", "mel_bins" },         .{ "downsample_hidden_size", "downsample" },
        .{ "n_window", "n_window" },   .{ "n_window_infer", "n_window_infer" }, .{ "output_dim", "output_dim" },
    }) |kv| {
        if (jsonField(o, kv[0])) |v| @field(qa, kv[1]) = try jsonU32(v);
    }
    if (qa.heads == 0 or qa.d_model % qa.heads != 0 or qa.n_window == 0 or qa.n_window_infer < 2 * qa.n_window or qa.downsample == 0)
        return error.InvalidConfigField;
    return qa;
}

/// The aligner's timestamp fields sit at the pack's root, beside `thinker_config`.
fn parseForcedAligner(outer: std.json.ObjectMap) !ForcedAlignerConfig {
    var fa = ForcedAlignerConfig{};
    if (jsonField(outer, "timestamp_token_id")) |v| fa.timestamp_token_id = try jsonU32(v);
    if (jsonField(outer, "timestamp_segment_time")) |v| fa.step_ms = try jsonFloat(v);
    if (jsonField(outer, "support_languages")) |v| {
        for ((try jsonValue(.array, v)).items) |l| fa.languages |= transcription.languageBit(try jsonValue(.string, l));
    }
    if (fa.step_ms <= 0) return error.InvalidConfigField;
    return fa;
}

/// Qwen3-ASR nests the whole model under `thinker_config`; the pack's quantization stays outside it.
fn qwenAsrThinker(outer: std.json.ObjectMap) ?std.json.ObjectMap {
    const mt = jsonField(outer, "model_type") orelse return null;
    if (mt != .string or !std.mem.eql(u8, mt.string, "qwen3_asr")) return null;
    const t = jsonField(outer, "thinker_config") orelse return null;
    return if (t == .object) t.object else null;
}

fn jsonValue(comptime tag: std.meta.Tag(std.json.Value), v: std.json.Value) !@TypeOf(@field(@as(std.json.Value, undefined), @tagName(tag))) {
    if (v != tag) return error.InvalidConfigField;
    return @field(v, @tagName(tag));
}

fn jsonU32(v: std.json.Value) !u32 {
    return std.math.cast(u32, try jsonValue(.integer, v)) orelse error.InvalidConfigField;
}

// Optional config nulls have the same fallback and precedence as omitted keys.
fn jsonField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// `rope_scaling` (or v5's `rope_parameters`) when it names the llama3 schedule.
fn parseLlama3Rope(cfg_obj: std.json.ObjectMap) !?Llama3Rope {
    const rs = jsonField(cfg_obj, "rope_scaling") orelse jsonField(cfg_obj, "rope_parameters") orelse return null;
    if (rs != .object) return null;
    const t = jsonField(rs.object, "rope_type") orelse jsonField(rs.object, "type") orelse return null;
    if (t != .string or !std.mem.eql(u8, t.string, "llama3")) return null;
    var r = Llama3Rope{ .factor = try jsonFloat(jsonField(rs.object, "factor") orelse return error.InvalidConfigField) };
    if (jsonField(rs.object, "low_freq_factor")) |v| r.low_freq_factor = try jsonFloat(v);
    if (jsonField(rs.object, "high_freq_factor")) |v| r.high_freq_factor = try jsonFloat(v);
    if (jsonField(rs.object, "original_max_position_embeddings")) |v| r.original_max_position_embeddings = try jsonFloat(v);
    if (r.factor <= 0 or r.high_freq_factor <= r.low_freq_factor) return error.InvalidConfigField;
    return r;
}

fn jsonSsmBound(v: std.json.Value, allow_infinity: bool) !f32 {
    switch (v) {
        .integer, .float => return jsonFloat(v),
        .string => |name| if (allow_infinity and std.mem.eql(u8, name, "Infinity")) return std.math.inf(f32),
        .object => |obj| {
            if (allow_infinity and obj.count() == 1) {
                if (obj.get("__float__")) |value| {
                    if (value == .string and std.mem.eql(u8, value.string, "Infinity")) return std.math.inf(f32);
                }
            }
        },
        else => {},
    }
    return error.InvalidConfigField;
}

fn jsonFloat64(v: std.json.Value) !f64 {
    const value: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.InvalidConfigField,
    };
    if (!std.math.isFinite(value)) return error.InvalidConfigField;
    return value;
}

fn jsonFloat(v: std.json.Value) !f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

pub const Weights = @import("../weights/resident.zig").Weights;

/// The generic nestings a text trunk ships under: flat, mlx-community's
/// re-nest, and meta's VL original (Muse-Glimmer). `parseConfigFromJson`
/// picks from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";
const VL_NESTED_PREFIX = "model.language_model";

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// Re-point `config.weight_prefix` at the nesting the CHECKPOINT actually uses.
///
/// Which nesting a converter emits is not reliably declared in config.json,
/// so `parseConfigFromJson` guesses from `text_config` presence — wrong for any
/// checkpoint that nests without declaring one (mlx-community LFM2.5-2.6B:
/// `Lfm2ForCausalLM`, an EMPTY `vision_config`, every weight under
/// `language_model.model.*`; the guess picked `model` and the load died on
/// `MISSING WEIGHT: model.embed_tokens.weight`). The class has now shipped in
/// both directions, so the weights get the last word.
///
/// Conservative by construction: generic trunks share one spelling set;
/// Nemotron's `backbone` has its own flat/nested pair. A swap
/// happens only when the configured one holds NOTHING, so every checkpoint
/// that already loaded binds byte-identically. Scan order puts the most
/// specific spelling first: a `model.language_model.*` checkpoint also
/// satisfies the bare "model" probe.
/// Prism Hadamard packs run f16 activations over f16 scales (the pack's own
/// contract) and ship the unpacked tensors f32 (norms, conv, the dense GDN a/b
/// rows, A_log, dt_bias): an f32 table widens the f16 residual, so those
/// narrow to f16. Signs stay f32: the rotation runs in f32.
pub fn narrowHadamardPackTables(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.hadamard_block == 0) return;
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const v = kv.value_ptr.*;
        if (mlx.mlx_array_dtype(v) != .float32) continue;
        if (std.mem.endsWith(u8, kv.key_ptr.*, ".signs")) continue;
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, v, config.actDtype(), s));
        _ = mlx.mlx_array_free(v);
        kv.value_ptr.* = cast;
    }
}

pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const nemotron_backbone = std.mem.eql(u8, config.model_type, "nemotron_h") and
        (std.mem.eql(u8, config.weight_prefix, "backbone") or std.mem.eql(u8, config.weight_prefix, "language_model.backbone"));
    const candidates: []const []const u8 = if (nemotron_backbone)
        &.{ "backbone", "language_model.backbone" }
    else
        &.{ NESTED_PREFIX, VL_NESTED_PREFIX, FLAT_PREFIX };
    var known = false;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) known = true;
    }
    if (!known) return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, p });
        config.weight_prefix = p;
        return;
    }
}

pub const loadWeights = weights_load.loadWeights;

pub const LoadOpts = weights_load.LoadOpts;

pub const Owners = weights_load.Owners;

pub const indexOwners = weights_load.indexOwners;

/// Sushi's EXL3 expert banks run on its Metal kernels only: without them (CUDA) the pack is
/// refused by name before any weight loads.
fn admitExl3Backend(config: *const ModelConfig, metal: bool) !void {
    if (config.trunk_format and !metal) {
        log.err("this pack's EXL3 trunk linears run on turboexl3's Metal kernels only; this backend cannot serve it\n", .{});
        return error.Exl3NeedsMetal;
    }
    if (config.exl3 == null or metal) return;
    log.err("this Sushi pack's EXL3 experts run on Metal kernels only; this backend cannot serve it\n", .{});
    return error.Exl3NeedsMetal;
}

/// Whether the weights load reads an EXL3 trunk pack's shards past the page cache (`nocache_reader`), as a DeepSeek-V4.1
/// pack's always does: when what it loads is more than half the machine's RAM (`nocacheFor`). MLX's own reader leaves
/// each shard's pages cached beside the arrays, so such a pack peaks near twice its weights until the kernel evicts the
/// cache. The embedded n-gram table is read per lookup, never loaded, so it does not count. Every other pack loads as
/// before.
fn nocacheLoad(io: std.Io, model_dir: []const u8, config: *const ModelConfig) bool {
    if (!config.trunk_format) return false;
    const shards = @import("../../engines/mlx/runtime/preflight.zig").modelDiskBytes(io, model_dir);
    return nocacheFor(shards -| (config.embedded_ple_payload_bytes orelse 0), @import("../util/system.zig").getTotalMemBytes());
}

/// `weights` bytes loaded on a machine of `ram` bytes read past the page cache when they are more than half of it (an
/// unknown RAM size, 0, never does).
pub fn nocacheFor(weights: u64, ram: u64) bool {
    return ram > 0 and weights > ram / 2;
}

/// A tensor worker never runs the MTP head or the towers.
const TensorWorker = struct {
    fn remap(_: TensorWorker, key: []const u8, _: []u8) ?[]const u8 {
        return if (distributed.headOnly(key)) null else key;
    }
};

/// Keeps the tensors one rank holds, under the names `keep.remap` gives them
/// (`LayerSpan`: its layers renumbered; `TensorWorker`: all but the head-only towers).
/// DeepSeek-V4.1 on a pair: each Mac's trunk layers under their own numbers (the model reads
/// sources and Engram layers by absolute index); the worker keeps nothing else, and neither
/// keeps the DSpark stages, which stay off on a pair.
const Dsv41Keep = struct {
    pair: @import("../../plugins/deepseek_v41/deepseek_v41.zig").Pair,

    pub fn remap(self: Dsv41Keep, key: []const u8, _: []u8) ?[]const u8 {
        if (std.mem.startsWith(u8, key, "layers.")) {
            const end = std.mem.indexOfScalarPos(u8, key, 7, '.') orelse return null;
            const li = std.fmt.parseInt(usize, key[7..end], 10) catch return null;
            return if (self.pair.holds(li)) key else null;
        }
        return if (self.pair.worker) null else key;
    }
};

fn retainPart(weights: *Weights, keep: anytype) !void {
    const a = weights.allocator;
    var kept = std.StringHashMap(mlx.mlx_array).init(a);
    errdefer kept.deinit();
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        var buf: [512]u8 = undefined;
        const name = keep.remap(kv.key_ptr.*, &buf) orelse {
            _ = mlx.mlx_array_free(kv.value_ptr.*);
            a.free(kv.key_ptr.*);
            continue;
        };
        if (name.ptr == kv.key_ptr.*.ptr) {
            try kept.put(kv.key_ptr.*, kv.value_ptr.*);
        } else {
            try kept.put(try a.dupe(u8, name), kv.value_ptr.*);
            a.free(kv.key_ptr.*);
        }
    }
    weights.map.deinit();
    weights.map = kept;
}

/// Keeps this tensor-parallel rank's slice of every expert projection (still lazy, so a
/// tensor is read once and only its part survives the batch eval). ModelOpt's routed
/// banks arrive already sliced (`modelopt_pack.buildBanks`).
fn shardExperts(weights: *Weights, sh: distributed.Shard, banks_sliced: bool) !void {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const axis = distributed.Shard.expertAxis(kv.key_ptr.*) orelse continue;
        if (banks_sliced and std.mem.indexOf(u8, kv.key_ptr.*, ".switch_mlp.") != null) continue;
        const shape = mlx.getShape(kv.value_ptr.*);
        const ax: usize = if (axis == .rows) shape.len - 2 else shape.len - 1;
        const part: c_int = @divExact(shape[ax], @as(c_int, @intCast(sh.size)));
        var start: [4]c_int = @splat(0);
        var stop: [4]c_int = undefined;
        const strides: [4]c_int = @splat(1);
        @memcpy(stop[0..shape.len], shape);
        start[ax] = part * @as(c_int, @intCast(sh.rank));
        stop[ax] = start[ax] + part;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, kv.value_ptr.*, &start, shape.len, &stop, shape.len, &strides, shape.len, s));
        const owned = try @import("../../engines/mlx/transformer.zig").materializedOwnedCopy(s, view);
        _ = mlx.mlx_array_free(kv.value_ptr.*);
        kv.value_ptr.* = owned;
    }
}

/// The text model's weights for `config`.
pub fn loadModelWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *ModelConfig, load_vision: bool) !Weights {
    if (config.mlxStream()) return Weights.init(allocator); // mlx-stream loads its own
    try admitExl3Backend(config, mlx.metalKernelsAvailable());
    var gguf_weights = Weights.init(allocator);
    errdefer gguf_weights.deinit();
    if (try mlx_gguf.loadWeights(io, allocator, model_dir, &gguf_weights.map)) return gguf_weights;
    if (config.embedded_ple_payload_bytes != null) {
        const info = (try qwen4_exp.inspectEmbedded(model_dir, try qwen4EmbeddedSpec(config))) orelse return error.MissingEmbeddedNgramTable;
        if (info.payload_bytes != config.embedded_ple_payload_bytes.?) return error.EmbeddedNgramTableChanged;
    }
    if (config.dist_span == null and config.dist_shard == null and config.dist_dsv41 == null) if (try distributed.planFor(model_dir, config)) |part| switch (part) {
        .span => |sp| try config.applyLayerSpan(sp),
        .shard => |sh| try config.applyShard(sh),
        .remote => |r| config.pairDsv41(.{ .lo = r.start, .hi = r.end, .worker = false }),
    };
    var weights = try loadWeightsOpt(io, allocator, model_dir, .{ .vision = load_vision, .audio_tower = config.qwen_audio != null or config.pluginHears(), .keep_f16 = config.actDtype() == .float16, .skip = if (config.isQwen4() and config.embedded_ple_payload_bytes != null) qwen4_exp.embeddedTensorName else null, .defer_qwen4_norms = config.isQwen4(), .dsv41 = config.isDsv41(), .index_owners = config.exl3 != null or config.isDsv41(), .modelopt = config.modelopt, .trunk_format = config.trunk_format, .nocache = nocacheLoad(io, model_dir, config) });
    errdefer weights.deinit();
    // Still lazy here, so the other ranks' tensors are dropped before any byte is read.
    if (config.dist_span) |sp| try retainPart(&weights, sp);
    if (config.dist_dsv41) |p| try retainPart(&weights, Dsv41Keep{ .pair = p });
    if (config.dist_shard) |sh| if (sh.rank != 0) try retainPart(&weights, TensorWorker{});
    if (config.modelopt) try modelopt_pack.adapt(&weights, model_dir, if (config.dist_span) |sp| sp.total else config.num_hidden_layers, config.swiglu_limit > 0.0, config.dist_span, config.dist_shard);
    if (config.dist_shard) |sh| try shardExperts(&weights, sh, config.modelopt);
    if (config.isDsv41() and !(if (config.dist_dsv41) |p| p.worker else false)) try loadDsv41Residents(io, allocator, &weights, model_dir);
    if (config.isQwen4()) {
        defer reportF16Narrowing();
        resolveWeightPrefix(config, &weights);
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        try resolveAndFoldQwen4Norms(config, &weights, s, model_dir);
    }
    if (config.exl3 != null or config.modelopt or (config.expert_bank and config.isGlm5())) {
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        try sushi_pack.adapt(config, &weights, s);
    }
    return weights;
}

pub const dsv41VisionKey = weights_load.dsv41VisionKey;

/// The repack keeps the Engram projections beside its tables, under `engram/`.
fn loadDsv41Residents(io: std.Io, allocator: std.mem.Allocator, weights: *Weights, model_dir: []const u8) !void {
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/engram/engram-residents.safetensors", .{model_dir}, 0);
    defer allocator.free(path);
    std.Io.Dir.accessAbsolute(io, path, .{}) catch return;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try loadSafetensorsFile(allocator, weights, path, s, .{});
}

pub const loadWeightsSingleFile = weights_load.loadWeightsSingleFile;

pub const loadWeightsWithVision = weights_load.loadWeightsWithVision;

const loadWeightsOpt = weights_load.loadWeightsOpt;

const loadWeightsFromOpenDir = weights_load.loadWeightsFromOpenDir;

pub const narrowsLoadedF16 = weights_load.narrowsLoadedF16;

pub const narrowsLoadedF32Norm = weights_load.narrowsLoadedF32Norm;

const narrow1dDefault = weights_load.narrow1dDefault;

const narrow1dEnabled = weights_load.narrow1dEnabled;

pub const reportF16Narrowing = weights_load.reportF16Narrowing;

pub const loadWeightsFile = @import("../weights/resident.zig").loadWeightsFile;

pub const loadSafetensorsFile = weights_load.loadSafetensorsFile;

pub const dsv41_text_prefix = weights_load.dsv41_text_prefix;

fn foldQwen4Norm(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try mlx.check(mlx.mlx_add(&shifted, value, one, s));
    var folded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(folded);
    try mlx.check(mlx.mlx_astype(&folded, shifted, .bfloat16, s));
    return folded;
}

fn inferLoadedQwen4Norms(config: *const ModelConfig, weights: *const Weights, s: mlx.mlx_stream) !Qwen4NormConvention {
    if (config.num_hidden_layers > 4096 or config.indexer_head_dim == 0) return error.InvalidQwen4NormAnchor;
    var inferred: ?Qwen4NormConvention = null;
    var total_bytes: u64 = 0;
    for (0..config.num_hidden_layers) |layer| {
        if (config.isLinearLayer(@intCast(layer))) continue;
        for ([_][]const u8{ "q_layernorm.weight", "k_layernorm.weight" }) |suffix| {
            var name_buf: [256]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "{s}.layers.{d}.self_attn.indexer.{s}", .{ config.weight_prefix, layer, suffix }) catch return error.InvalidQwen4NormAnchor;
            const value = weights.get(name) orelse {
                log.err("[qwen4] missing norm anchor {s}\n", .{name});
                return error.MissingQwen4NormAnchor;
            };
            const shape = mlx.getShape(value);
            if (shape.len != 1 or shape[0] != config.indexer_head_dim) return error.InvalidQwen4NormAnchor;
            const dtype = mlx.mlx_array_dtype(value);
            if (dtype != .float32 and dtype != .float16 and dtype != .bfloat16) return error.InvalidQwen4NormAnchor;
            const bytes = std.math.mul(u64, @as(u64, @intCast(mlx.mlx_array_size(value))), @as(u64, @intCast(mlx.mlx_array_itemsize(value)))) catch return error.InvalidQwen4NormAnchor;
            total_bytes = std.math.add(u64, total_bytes, bytes) catch return error.InvalidQwen4NormAnchor;
            if (total_bytes > 1024 * 1024) return error.InvalidQwen4NormAnchor;

            var f32_value = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32_value);
            try mlx.check(mlx.mlx_astype(&f32_value, value, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32_value));
            const data = mlx.mlx_array_data_float32(f32_value) orelse return error.InvalidQwen4NormAnchor;
            var sum: f64 = 0;
            for (data[0..@intCast(config.indexer_head_dim)]) |number| {
                if (!std.math.isFinite(number)) return error.AmbiguousQwen4NormConvention;
                sum += number;
            }
            const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(config.indexer_head_dim)));
            const current: Qwen4NormConvention = if (mean >= -0.2 and mean <= 0.2) .delta else if (mean >= 0.8 and mean <= 1.2) .folded else return error.AmbiguousQwen4NormConvention;
            if (inferred) |prior| {
                if (prior != current) return error.AmbiguousQwen4NormConvention;
            } else inferred = current;
        }
    }
    return inferred orelse error.InvalidQwen4NormAnchor;
}

fn resolveAndFoldQwen4Norms(config: *ModelConfig, weights: *Weights, s: mlx.mlx_stream, model_dir: []const u8) !void {
    const marked = config.qwen4_norm_convention != null;
    const inferred = config.qwen4_norm_convention == null and config.embedded_ple_payload_bytes != null;
    const convention = if (config.qwen4_norm_convention) |marker| marker else if (inferred)
        inferLoadedQwen4Norms(config, weights, s) catch |err| {
            log.err("[qwen4] cannot infer norm convention for {s}: {s}; set qwen4_norm_convention to delta or folded in this checkpoint's config.json.\n", .{ model_dir, @errorName(err) });
            return err;
        }
    else
        Qwen4NormConvention.folded;

    var it = weights.map.iterator();
    while (it.next()) |entry| {
        if (!qwen4NormNeedsFold(entry.key_ptr.*)) continue;
        const value = entry.value_ptr.*;
        if (convention == .delta) {
            entry.value_ptr.* = try foldQwen4Norm(value, s);
        } else if (config.actDtype() != .float16 and narrowsLoadedF16(entry.key_ptr.*, mlx.mlx_array_ndim(value), mlx.mlx_array_dtype(value)) and narrow1dEnabled()) {
            var cast = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            entry.value_ptr.* = cast;
            weights_load.narrowed_1d += 1;
        } else continue;
        _ = mlx.mlx_array_free(value);
    }
    config.qwen4_norm_convention = convention;
    if (inferred) {
        log.info("[qwen4] inferred {s} norm convention from loaded trunk indexer norms: {s}\n", .{ @tagName(convention), model_dir });
    } else if (!marked) {
        log.info("[qwen4] using folded norm convention (legacy external-table default): {s}\n", .{model_dir});
    } else {
        log.info("[qwen4] using checkpoint-local {s} norm convention: {s}\n", .{ @tagName(convention), model_dir });
    }
}

pub const qwen4NormNeedsFold = weights_load.qwen4NormNeedsFold;

fn addQwen4NormTestArray(weights: *Weights, name: []const u8, values: []const f32, shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    const raw = mlx.mlx_array_new_data(values.ptr, shape.ptr, @intCast(shape.len), .float32);
    var value = raw;
    if (dtype != .float32) {
        value = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&value, raw, dtype, s));
        _ = mlx.mlx_array_free(raw);
    }
    errdefer _ = mlx.mlx_array_free(value);
    const key = try weights.allocator.dupe(u8, name);
    errdefer weights.allocator.free(key);
    try weights.map.put(key, value);
}

const keepsTensor = weights_load.keepsTensor;

pub const shouldKeepWeightKey = weights_load.shouldKeepWeightKey;

// ── Tests ──

const testing = std.testing;

/// MiMo-V2.6-Flash in the mlx-lm layout (`tests/convert_mimo_v2.py`, mlx-community's
/// mxfp4-q8), layers trimmed to 6.
const mimo_v2_pack_json =
    \\{
    \\  "model_type": "mimo_v2",
    \\  "add_full_attention_sink_bias": false, "add_swa_attention_sink_bias": true,
    \\  "attention_projection_layout": "fused_qkv", "attention_value_scale": 0.707,
    \\  "eos_token_id": 151645, "head_dim": 192, "hidden_size": 4096,
    \\  "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0],
    \\  "intermediate_size": 16384, "layernorm_epsilon": 1e-06,
    \\  "max_position_embeddings": 1048576, "moe_intermediate_size": 2048,
    \\  "moe_layer_freq": [0, 1, 1, 1, 1, 1],
    \\  "n_group": 1, "n_routed_experts": 256, "n_shared_experts": null, "norm_topk_prob": true,
    \\  "num_attention_heads": 64, "num_experts_per_tok": 8, "num_hidden_layers": 6,
    \\  "num_key_value_heads": 4, "num_nextn_predict_layers": 3, "partial_rotary_factor": 0.334,
    \\  "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default"},
    \\  "rope_theta": 10000000.0, "routed_scaling_factor": null, "scoring_func": "sigmoid",
    \\  "sliding_window": 128, "swa_head_dim": 192, "swa_num_attention_heads": 64,
    \\  "swa_num_key_value_heads": 8, "swa_rope_theta": 10000.0, "swa_v_head_dim": 128,
    \\  "tie_word_embeddings": false, "topk_group": 1, "topk_method": "noaux_tc",
    \\  "v_head_dim": 128, "vocab_size": 152576,
    \\  "quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"},
    \\  "quantization_config": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    \\}
;

/// GLM-5.3-Flash (TensorFold's MLX packs), layers trimmed to 8.
const glm5_next_pack_json =
    \\{
    \\  "model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
    \\  "tie_word_embeddings": false,
    \\  "text_config": {
    \\    "model_type": "glm5_next_text", "vocab_size": 154880, "hidden_size": 4096,
    \\    "intermediate_size": 12288, "moe_intermediate_size": 2048, "num_hidden_layers": 8,
    \\    "num_nextn_predict_layers": 1, "num_attention_heads": 64, "num_key_value_heads": 64,
    \\    "n_shared_experts": 1, "n_routed_experts": 288, "routed_scaling_factor": 2.5,
    \\    "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 0, "v_head_dim": 256,
    \\    "qk_nope_head_dim": 256, "qk_head_dim": 256, "head_dim": 0, "n_group": 1, "topk_group": 1,
    \\    "num_experts_per_tok": 8, "norm_topk_prob": true, "max_position_embeddings": 1048576,
    \\    "rms_norm_eps": 1e-05, "eos_token_id": [154820, 154827, 154829], "pad_token_id": 154820,
    \\    "mlp_layer_types": ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
    \\    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
    \\                    "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
    \\    "indexer_types": ["full", "full", "full", "full", "full", "full", "full", "full"],
    \\    "index_topk": 2048, "index_head_dim": 128, "index_n_heads": 32, "index_kpool": 4,
    \\    "index_kpool_always_select_tail": true, "index_kpool_compress": true, "mla_use_nope": true,
    \\    "swiglu_limit": 10.0, "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20, "mhc": true,
    \\    "first_k_dense_replace": 3, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    \\    "moe_router_dtype": "float32",
    \\    "linear_attn_config": {"num_heads": 64, "head_dim": 128, "gate_lower_bound": -5.0,
    \\                           "short_conv_kernel_size": 4, "full_attn_layers": [3, 7]}
    \\  },
    \\  "image_token_id": 154854, "image_start_token_id": 154830, "image_end_token_id": 154831,
    \\  "vision_config": {"model_type": "glm5_next_vision", "depth": 24, "hidden_size": 1024, "num_heads": 16,
    \\    "intermediate_size": 4096, "out_hidden_size": 4096, "patch_size": 14, "temporal_patch_size": 2,
    \\    "spatial_merge_size": 2, "projection_intermediate_size": 10240, "rms_norm_eps": 1e-05, "swiglu_limit": 10.0}
    \\}
;

/// DeepSeek's V4.1-Flash release config (rev dba1be0a), text_config abridged
/// to the keys the engine reads, vision_config dropped (text-only).
pub const dsv41_release_config =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
    \\ "bos_token_id": 0, "eos_token_id": 1,
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120,
    \\  "moe_intermediate_size": 2304, "num_hidden_layers": 40, "num_attention_heads": 64,
    \\  "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
    \\  "o_lora_rank": 1024, "o_groups": 8, "hidden_act": "silu", "swiglu_limit": 10.0,
    \\  "rms_norm_eps": 1e-20, "max_position_embeddings": 1048576, "rope_theta": 10000,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
    \\   "original_max_position_embeddings": 65536},
    \\  "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
    \\  "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
    \\  "routed_scaling_factor": 1.5, "sliding_window": 128,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    \\  "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 8, 14, 20],
    \\  "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36], "index_n_heads": 32,
    \\  "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
    \\  "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 14],
    \\  "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4,
    \\  "engram_vocab_size": 16000000, "engram_n_heads": 8, "engram_head_dim": 256,
    \\  "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
    \\  "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
    \\  "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256,
    \\  "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3},
    \\ "image_token_id": 129264,
    \\ "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024,
    \\  "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000,
    \\  "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null}}
;

const glm_moe_dsa_config =
    \\{"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "vocab_size": 154880,
    \\ "hidden_size": 6144, "num_hidden_layers": 78, "num_attention_heads": 64, "num_key_value_heads": 64,
    \\ "head_dim": 192, "max_position_embeddings": 1048576, "tie_word_embeddings": false, "n_routed_experts": 256,
    \\ "eos_token_id": [154820, 154827, 154829],
    \\ "quantization": {"group_size": 64, "bits": 4, "model.embed_tokens": {"group_size": 64, "bits": 8}}}
;

const embedding_gemma2_config_json =
    \\{
    \\  "architectures": ["EmbeddingGemma2Model"],
    \\  "model_type": "embedding_gemma2",
    \\  "audio_config": {"model_type": "gemma4_audio", "hidden_size": 1024, "output_proj_dims": 1536},
    \\  "audio_token_id": 258881,
    \\  "image_token_id": 258880,
    \\  "text_config": {
    \\    "model_type": "embedding_gemma2_text",
    \\    "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    \\    "embedding_dim": 768, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh",
    \\    "hidden_size": 512, "hidden_size_per_layer_input": 512, "intermediate_size": 2048,
    \\    "layer_types": [
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
    \\      "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    \\    ],
    \\    "max_position_embeddings": 262144,
    \\    "num_attention_heads": 4, "num_hidden_layers": 24, "num_key_value_heads": 2,
    \\    "per_layer_config": {
    \\      "05": {"head_dim": 512, "num_key_value_heads": 1}, "11": {"head_dim": 512, "num_key_value_heads": 1},
    \\      "17": {"head_dim": 512, "num_key_value_heads": 1}, "23": {"head_dim": 512, "num_key_value_heads": 1}
    \\    },
    \\    "rms_norm_eps": 1e-06,
    \\    "rope_parameters": {
    \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
    \\      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
    \\    },
    \\    "sliding_window": 512, "vocab_size": 262144
    \\  },
    \\  "boi_token_id": 255999, "eoi_token_id": 258882,
    \\  "video_token_id": 258884,
    \\  "vision_config": {
    \\    "model_type": "gemma4_vision", "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16,
    \\    "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64, "global_head_dim": 64,
    \\    "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280, "position_embedding_size": 10240,
    \\    "use_clipped_linears": false, "standardize": false, "rms_norm_eps": 1e-06,
    \\    "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}
    \\  },
    \\  "quantization": {"group_size": 64, "bits": 8, "mode": "affine"}
    \\}
;

// ── qwen4_exp YaRN context extension (262144 → 1048576) ──────────────────
//
// Both documents below are the SHIPPED checkpoint's text config (Qwen3.8-Flash-
// Next, `model_type: qwen4_exp`) — one as it ships (plain rope, 262144) and one
// with the YaRN block vLLM's `--hf-overrides` recipe writes. Keeping them as
// literals means the parser is tested against the real file shape, braces and
// all, rather than a synthesized one.

/// The checkpoint as it ships: `rope_type: "default"`, a 262144 window.
const QWEN4_SHIPPED =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 262144,
    \\    "rope_parameters": {
    \\      "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// The same checkpoint with its rope scaled 4× and the window widened — exactly
/// `vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {...}}}'
/// --max-model-len 1010000` expressed as config instead of a flag.
const QWEN4_YARN =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 1048576,
    \\    "rope_parameters": {
    \\      "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
    \\      "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// QWEN4_SHIPPED's GatedDeltaNet geometry: 36 linear layers of 48 value heads over Dk = Dv = 128.
const QWEN4_GDN_FIELDS = ",\"linear_num_key_heads\":16,\"linear_num_value_heads\":48,\"linear_key_head_dim\":128," ++
    "\"linear_value_head_dim\":128,\"linear_conv_kernel_dim\":4";

/// One qwen4_exp config document with `extra` fields spliced in.
fn qwen4CaseJson(comptime extra: []const u8) []const u8 {
    return "{\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48," ++
        "\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"ple_conv_kernel_size\":4," ++
        "\"num_experts\":512,\"num_experts_per_tok\":10,\"moe_intermediate_size\":640," ++
        "\"eos_token_id\":248044,\"vocab_size\":248320,\"rms_norm_eps\":1e-6," ++
        extra ++ "}";
}

const QWEN4_GOOD_FIELDS =
    "\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8," ++
    "\"ngram_vocab_size_base\":20000000,\"make_ngram_vocab_size_divisible_by\":128," ++
    "\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048,\"indexer_compress_ratio\":4";
