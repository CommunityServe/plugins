//! copied_from: src/sdk/config/model.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const plugin_config = @import("../../engines/mlx/plugin/config.zig");
const abi = @import("mlx_serve").abi.plugin;
const exl3 = @import("../quant/exl3/format.zig");
const distributed = @import("../../engines/mlx/cluster/distributed.zig");
const kv_quant_mod = @import("../blocks/attention/kv_quant.zig");
const mtp_acceptance_mod = @import("../spec/mtp_acceptance.zig");
pub const HiddenAct = @import("../blocks/mlp/activations.zig").HiddenAct;
/// Upper bound on a vision tower's per-layer type table (muse ships 50).
pub const MAX_VISION_LAYERS = 64;
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
