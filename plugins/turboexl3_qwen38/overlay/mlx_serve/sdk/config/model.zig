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

};

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

    /// V4.1's layers split with a worker Mac: the worker keeps ONE decode state and its exchange names no slot, so the
    /// model leaves per-slot state and batched decode and serves one request at a time.
    pub fn pairDsv41(self: *ModelConfig, p: @import("../../plugins/deepseek_v41/deepseek_v41.zig").Pair) void {
        self.dist_dsv41 = p;
        if (self.plugin) |b| {
            b.parsed.caps.slot_state = false;
            b.parsed.caps.batches_decode = false;
        }
    }

    /// Rows a sliding layer keeps past its window on an arch that caches only the window
    /// (`slidingRing`): spec rollback plus the prefix-cache matches that land near an entry's end.
    pub const SLIDING_RING_SLACK: u32 = 2048;

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

    pub fn rowExactArch(self: *const ModelConfig) bool {
        if (self.caps().exact_drafted) return true;
        return std.mem.startsWith(u8, self.model_type, "qwen3_5") and !self.isMoe() and self.hadamard_block == 0;
    }

    pub fn isGlm5(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
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

    /// The keys a batched tick reads per slot: past the gather floor, QSA's indexer budget and its open block.
    pub fn batchedEffectiveKvLen(self: *const ModelConfig, kv: u32, gather_on: bool, gather_min_kv: u32) u32 {
        if (!gather_on) return kv;
        if (kv <= gather_min_kv) return kv;
        const cap = self.indexer_budget + self.indexer_compress_ratio;
        if (cap == 0) return kv;
        return @min(kv, cap);
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

pub const SoftTokenBudgets = struct { image: ?u32 = null, video: ?u32 = null, max_frames: ?u32 = null };

pub var gdn_state_f32_override: ?bool = null; // test seam
var gdn_state_f32_env: ?bool = null;

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

pub const mergeConfigJson = @import("merge.zig").mergeConfigJson;

/// EmbeddingGemma 2's context window as its model card states it. Kept in sync with
/// `model_discovery.parseStubMeta`, which cannot import this file.
pub const EMBEDDING_GEMMA2_WINDOW: u32 = 8192;

// Optional config nulls have the same fallback and precedence as omitted keys.

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

const QWEN4_GOOD_FIELDS =
    "\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8," ++
    "\"ngram_vocab_size_base\":20000000,\"make_ngram_vocab_size_divisible_by\":128," ++
    "\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048,\"indexer_compress_ratio\":4";
