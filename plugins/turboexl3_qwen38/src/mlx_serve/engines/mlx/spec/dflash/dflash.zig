//! copied_from: src/engines/mlx/spec/dflash/dflash.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! DFlash block-drafter — generic block-parallel speculative decoding.
//!
//! A small assistant transformer drafts `block_size - 1` tokens in ONE
//! forward, conditioned on the trunk's intermediate hidden states. Mirrors
//! transformers' `DFlashTokenCandidateGenerator` + `DFlashCache` and the
//! `muse_glimmer_assistant` reference model. DFlash is a METHOD, not a
//! model-family feature: detection keys on the CONFIG CONTRACT
//! (`block_size` + `mask_token_id` + `target_layer_ids`), never on the
//! `model_type` string.
//!
//! Protocol per round (assistant borrows the trunk's embed table):
//!   1. Context delta = trunk hiddens at the OUTPUT of layers
//!      `target_layer_ids` for the tokens the trunk just accepted,
//!      concatenated on features → `encoder.fc` → RMS norm → per-layer K/V
//!      appended to the assistant cache at those tokens' absolute positions.
//!   2. `noise_embeds` = RAW trunk embedding lookup (no norm, no scale) of
//!      `[anchor(t1), mask_token × (block_size-1)]`.
//!   3. Assistant forward: Q from the block only; K/V = cached context +
//!      fresh block K/V. Block attends bidirectionally among itself,
//!      sliding-windowed/full to context.
//!   4. Draft logits use the declared own head or the trunk
//!      `lm_head(hidden)[:, 1:]` — drop the anchor → block_size-1 drafts.
//!   5. Trunk verify over `[t1, drafts...]` (standard spec verify invariant);
//!      the verify forward doubles as the next round's context producer.
//!   6. Block K/V are NEVER cached (reference evicts them via
//!      `crop(-block_size)`; we append them into spare capacity and
//!      truncate back — same math, no copy).

const std = @import("std");
const env = @import("../../../../env.zig");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const model_mod = @import("../../../../sdk/config/model.zig");
const transformer_mod = @import("../../transformer.zig");
const plugin_driver = @import("../../plugin/driver.zig");
// The chunked row requantizer + its packed-triple handle are shared with the
// MTP head — both sidecars shrink the SAME trunk lm_head for drafts only, and
// one requantizer with one chunking discipline is the point.
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const simd_qmm = @import("../../../../sdk/blocks/qmm/simd_qmm.zig");
const lane_qmm = @import("../../../../sdk/blocks/qmm/lane_qmm.zig");
const ane_mod = @import("../../../../sdk/ane/offload.zig");
const lilicorr_mod = @import("lilicorr.zig");

const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

pub const LayerType = enum {
    sliding_attention,
    full_attention,

};

pub const DflashConfig = struct {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    sliding_window: u32,
    layer_types: []LayerType, // owned
    // ── The DFlash contract fields ──
    block_size: u32,
    mask_token_id: u32,
    target_layer_ids: []u32, // owned, ascending
    // ── DFlash2 extension (all 0/absent on a v1 assistant) ──
    selector_rank: u32 = 0,
    selector_top_k: u32 = 0,
    conv_kernel_size: u32 = 0,
    conv_group_size: u32 = 0,
    /// Reference `compute_logits`: `logits * output_multiplier`, then
    /// `tanh(l/cap)*cap`. The selected head is the BARE Linear — an
    /// argmax draft is invariant to these (monotone), but the selector SUMS
    /// unary logits with codebook edges, so the muse sidecar's declared
    /// scale is load-bearing there. 0 cap / 1.0 multiplier = no transform.
    logit_softcap: f32 = 0,
    output_multiplier: f32 = 1.0,
    // ── DSpark extension (0/false on a v1/v2 assistant) ──
    /// Rank of the vanilla Markov head's low-rank bigram bias. > 0 means the
    /// block is drafted SEMI-autoregressively: each step's logits get
    /// `markov_w2(markov_w1[prev_token])` added before its own draft is
    /// picked, chaining the block without a second assistant forward.
    markov_rank: u32 = 0,
    /// Draft row mapping. SpecForge DFlash exports (muse, z-lab DFlash2) DROP
    /// the anchor row — mask row j predicts draft j. DSpark exports read ALL
    /// rows and the ANCHOR row emits draft 0, so their `block_size` counts
    /// DRAFTS, not verify width. Reading the wrong mapping is a silent
    /// 0%-acceptance drafter, not an error.
    anchor_row_drafts: bool = false,
    /// GPT-J interleaved rope (`rope_is_neox_style: false`). Every DFlash
    /// v1/v2 sidecar so far is neox (half-split); DSpark's is not, and the
    /// wrong half rotates silently.
    rope_traditional: bool = false,
    /// Bolt LiLiCorr head. Capture ids are the declared targets minus one:
    /// this engine stores a layer's output, and the head was trained on its input.
    lilicorr: bool = false,
    candidate_topk: u32 = 0,
    lilicorr_hidden: u32 = 0,
    lilicorr_layers: u32 = 0,
    lilicorr_heads: u32 = 0,
    factor_dim: u32 = 0,
    vector_eps: f32 = 0.0001,
    logit_scale: f32 = 8,
    has_attn_sinks: bool = false,
    /// A trained sidecar head uses the target's token ids without remapping.
    has_own_lm_head: bool = false,
    draft_vocab_size: u32 = 0,
    /// Owned when `lilicorr` (same length as `target_layer_ids`). Empty otherwise.
    capture_ids: []u32 = &.{},

};

/// Does this config.json declare the DFlash contract? ALL THREE fields must
/// be present — `model_type` (`*_assistant`) is only the discovery-level
/// "this is a drafter" signal, never the DFlash detection.
pub const isDflashConfigJson = dflash_contract.isDflashConfigJson;

/// The object holding the DFlash contract triple: DFlash2 checkpoints nest it
/// under `dflash_config`, v1 assistants declare it at the root. Null when
/// neither shape declares all three fields.
/// The contract fields, looked up NESTED-FIRST then at the root. DSpark
/// splits the triple across both (`block_size` at the root, the rest under
/// `dflash_config`), so neither object alone answers the question and a
/// nested-only reader silently classifies the sidecar as "not a drafter".
const dflash_contract = @import("../../../../sdk/spec/dflash_contract.zig");
const Contract = dflash_contract.Contract;
const dflashContractObject = dflash_contract.object;

/// Conventional in-checkpoint home for a DFlash assistant: a subdirectory of
/// the model's own directory declaring the config contract. Mirrors
/// `mtp.resolveMtpSource`'s sidecar-or-in-checkpoint shape, and it is what
/// turns the drafter from a LAUNCH-flag dependency into a LOAD-time one — a
/// hot model switch brings its own drafter, no pairing table decides which
/// sidecar goes with which checkpoint, and a mismatched pair is unbuildable.
pub const IN_DIR_SUBDIR = "drafter";

/// Standalone sushi writes a quantized copy of a pack's drafter beside it, marked by this manifest.
const SUSHI_RUNTIME_CACHE = "sushi-runtime-cache.json";

/// Widest verify a machine without the NAX m16 tile serves well: the block
/// IS the verify width, and `transformer.vqmmLaneFor`'s split-K lane stops at
/// M=7. Past it MLX's own 32x32-tiled `qmm_splitk` takes over and a trunk
/// forward jumps from ~1.3x a serial step to ~4.2x, for barely more accepted
/// tokens. Measured on Muse 4-bit / applegpu_g16s, 160-token generations,
/// serial reference in the same boot (28.2 tok/s):
///   block 16 → 0.92x   8 → 1.16x   7 → 1.43x   6 → 1.79x   5 → 1.97x
///   4 → 1.81x   3 → 1.77x
/// The block is a REQUEST cost, not a training constant — the assistant
/// drafts against the same mask token at any width. NAX-capable machines
/// (M5-class) have a real M 8..16 lane and keep the checkpoint's block.
pub const NO_WIDE_LANE_BLOCK_CAP: u32 = 5;
pub const TREE_BLOCK_CAP: u32 = 8;
/// Positions a draft tree's lattice spans on the tensor units.
pub const TREE_NAX_BLOCK: u32 = 16;

/// A no-wide-lane block cap with the machine row it came from, for the
/// `DFlash drafter ready` line — a capped block must say WHY in tester logs.
pub const BlockCap = struct {
    cap: u32,
    label: []const u8,
    /// True when a HUMAN measured this chip. A measured row wins over the
    /// probe: the probe times a forward, while these rows were measured as
    /// realized throughput (acceptance included), which the forward ladder
    /// cannot see.
    measured: bool = false,
};

// ── Weight precision ──

/// Default affine width the dense bf16 checkpoint is quantized to at load.
/// The assistant weight read is the per-round cost the block forward is
/// bound by (5.11 GB bf16 on the Muse sidecar); 8-bit halves it. Only DRAFT
/// quality rides on it — the trunk verify is exact either way, so the worst
/// a bad width can do is cost acceptance.
pub const DEFAULT_QUANT_BITS: u32 = 8;

/// Draft-only lm_head width — DEFAULT OFF (drafts use the trunk's own head).
/// A narrower head shrinks a read the round barely notices and pays for it in
/// ACCEPTANCE, which is the scarce resource: measured on Muse 4-bit at the
/// resolved block (5), 160-token generations, same serial reference —
///   3-bit draft head: 55.4 tok/s, 2.19 accepted/round
///   trunk head:       57.8 tok/s, 2.37 accepted/round
/// The lever was defaulted on at block 16, where the head was 10 ms of a
/// 175 ms round; at block 5 it is under 2 ms of ~60 and the acceptance it
/// costs is worth more than the bytes it saves; `bindWithDraftBits` selects a width for an A/B.
pub const DEFAULT_DRAFT_HEAD_BITS: u32 = 0;
/// Preferred affine group size, narrowed to 32 when 64 does not divide the
/// contraction dim, dense when neither does.
pub const QUANT_GROUP: u32 = 64;

/// One assistant linear. Dense weights are pre-transposed to `[in, out]` and
/// contracted with a plain matmul (drafter.zig's convention); quantized ones
/// keep the checkpoint's `[out, in]` packing and ride `mlx_quantized_matmul`
/// with transpose=true, exactly like the trunk. `bits == 0` IS the dense
/// discriminator — never probe the scales handle.
pub const DflashLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    bits: u32 = 0,
    group_size: u32 = 0,

};

// ── Model ──

/// DFlash2 grouped dynamic causal conv (one per sublayer): two-tap depthwise
/// conv whose kernels are `base + dynamic`, the dynamic part projected
/// per-position from the sublayer's normed input. `base_kernel` axes are
/// `[prepare|finish, tap, channel]` — BOTH leading dims are 2 at ksize 2, so
/// the order is pinned by the parity fixture, never by shape.
pub const DynConv = struct {
    base_kernel: mlx.mlx_array, // [2, ksize, hidden] bf16
    kernel_projection: DflashLinear, // [2*ksize*groups ← hidden]

};

/// DFlash2 path selector: per-position top-k candidates scored pairwise
/// through two per-token codebooks + a hidden→rank projection. Codebooks are
/// GATHER-read tables — bf16, never quantized (the NEVER_QUANTIZE class).
pub const Selector = struct {
    pred_codebook: mlx.mlx_array, // [vocab, rank] bf16
    succ_codebook: mlx.mlx_array, // [vocab, rank] bf16
    hidden_projection: DflashLinear, // [rank ← hidden]

};

/// Per-layer assistant weights.
/// DSpark's vanilla Markov head: a rank-`markov_rank` bigram bias added to
/// each block position's logits from the token drafted at the PREVIOUS
/// position. It is what makes the block semi-autoregressive off ONE assistant
/// forward — the base logits are position-parallel, the bias chains them.
///
/// `markov_w1` is a gather table (`[vocab, rank]`, one row per previous
/// token) and stays DENSE: a packed table read by `mlx_take_axis` gathers
/// uint32 words, not rows. `markov_w2` is an ordinary `rank → vocab` linear
/// and rides the same load-time quantization as the rest of the sidecar —
/// only the DRAFT sees it, so a lossier bias costs acceptance, never a token.
pub const MarkovHead = struct {
    w1: mlx.mlx_array,
    w2: DflashLinear,

};

pub const DflashLayer = struct {
    layer_type: LayerType,

    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,

    q: DflashLinear, // [hidden → n_heads*head_dim]
    q_norm: mlx.mlx_array, // [head_dim]
    k: DflashLinear, // [hidden → kv_heads*head_dim]
    k_norm: mlx.mlx_array, // [head_dim]
    v: DflashLinear, // [hidden → kv_heads*head_dim]
    o: DflashLinear, // [n_heads*head_dim → hidden]

    gate: DflashLinear, // [hidden → intermediate]
    up: DflashLinear, // [hidden → intermediate]
    down: DflashLinear, // [intermediate → hidden]

    // DFlash2 only — null on a v1 assistant.
    attention_conv: ?DynConv = null,
    mlp_conv: ?DynConv = null,
    /// Learned per-query-head softmax sink. Null when the config has no sinks.
    attn_sink: mlx.mlx_array = .{ .ctx = null },

};

pub const DflashModel = struct {
    config: DflashConfig,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    fc: DflashLinear, // encoder.fc [n_targets*hidden → hidden]
    enc_norm: mlx.mlx_array, // encoder.output_norm_enc [hidden]
    final_norm: mlx.mlx_array, // norm [hidden]
    /// Trained by the sidecar; only draft logits use this projection.
    own_lm_head: ?DflashLinear = null,
    layers: []DflashLayer,

    /// Optional DRAFT-ONLY low-bit lm_head, requantized from the trunk's at
    /// bind time (`DEFAULT_DRAFT_HEAD_BITS`, 0 disables).
    /// Only the block's draft argmax projects through it — VERIFICATION is a
    /// trunk forward and never touches it, so the emitted distribution is
    /// untouched; drafts just read ~⅓ of the bytes of a full-vocab head.
    draft_head: ?mtp_mod.QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,

    /// DFlash2 path selector — null on a v1 assistant.
    selector: ?Selector = null,

    /// DSpark Markov head — null unless the config declares `markov_rank`.
    markov: ?MarkovHead = null,

    /// Bolt correlation head. Null on a stock DFlash assistant.
    lilicorr: ?lilicorr_mod.Head = null,

};

// ── Weight loading ──

// ── Per-request context cache ──

/// The assistant's context K/V, one entry per ASSISTANT layer, dense.
/// Grows with committed trunk tokens; block K/V transit through spare
/// capacity and are truncated straight back out (never cached). Invariant
/// at round start: `base_pos + cache.step == trunk cache.step`.
pub const DflashCtx = struct {
    cache: KVCache,
    /// Absolute trunk position of cache index 0. Nonzero on hot-prefix-cache
    /// reuse, where only the freshly forwarded tail produced captures — a
    /// late-starting context is self-consistent (sliding-window semantics,
    /// same rule as the MTP history).
    base_pos: usize,
    /// Last encoded context row. LiLiCorr scores slot 0 against this feature,
    /// which is the token before the next draft anchor.
    anchor: mlx.mlx_array = .{ .ctx = null },

};

// ── Forward helpers ──

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

// ── DFlash2 grouped dynamic causal conv (forward) ──

/// The reference's `_grouped_dynamic_convolve`, transcribed verbatim:
/// `out_t = Σ_tap (base[tap] + dyn_t[tap]) ⊙ x_{t-tap}`, with positions
/// before the block start reading ZERO — the block is self-contained, so the
/// anchor's predecessor tap is the reference's zero pad, never the previous
/// block. `base` is per-CHANNEL `[ksize, H]`; `dynamic` is per-GROUP
/// `[1, L, ksize, groups]`, each coefficient broadcasting over `group_size`
/// channels. Two separate multiply-adds per tap keep the reference's bf16
/// rounding order. One kernel on the GPU (`dynConvFused`), the op chain
/// elsewhere.
pub fn groupedDynConv(
    hidden: mlx.mlx_array, // [1, L, H]
    dynamic: mlx.mlx_array, // [1, L, ksize, groups]
    base: mlx.mlx_array, // [ksize, H]
    group_size: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    if (try dynConvFused(hidden, dynamic, base, group_size, s)) |y| return y;
    return groupedDynConvOps(hidden, dynamic, base, group_size, s);
}

// Each tap adds base[tap] * x_{t-tap}, then dyn_t[tap] * x_{t-tap}, every
// product and sum rounded to T as the op chain's elementwise kernels do.
const DYN_CONV_SOURCE = kernelSource("kernels/spec/dflash_dyn_conv.metal");
var dyn_conv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const DynConvKey = struct { l: c_int, h: c_int, ks: c_int, gs: c_int, dt: mlx.mlx_dtype };
var dyn_conv_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var dyn_conv_key: ?DynConvKey = null;

fn dynConvFused(hidden: mlx.mlx_array, dynamic: mlx.mlx_array, base: mlx.mlx_array, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(hidden);
    if (mlx.mlx_array_dtype(dynamic) != dt or mlx.mlx_array_dtype(base) != dt) return null;
    const hsh = mlx.getShape(hidden);
    const bsh = mlx.getShape(base);
    const dsh = mlx.getShape(dynamic);
    if (hsh.len != 3 or hsh[0] != 1 or bsh.len != 2 or dsh.len != 4) return null;
    const gs: c_int = @intCast(group_size);
    const key = DynConvKey{ .l = hsh[1], .h = hsh[2], .ks = bsh[0], .gs = gs, .dt = dt };
    if (@rem(key.h, gs) != 0 or bsh[1] != key.h or dsh[0] != 1 or dsh[1] != key.l or dsh[2] != key.ks or dsh[3] != @divExact(key.h, gs)) return null;
    if (dyn_conv_kernel == null) {
        const ins = [_][*:0]const u8{ "x", "dyn", "base" };
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("msv_dflash_dyn_conv", in_vec, out_vec, DYN_CONV_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        dyn_conv_kernel = k;
    }
    if (dyn_conv_key == null or !std.meta.eql(dyn_conv_key.?, key)) {
        if (dyn_conv_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        dyn_conv_cfg = null;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.l, key.h }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, key.l * key.h, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        inline for (.{ .{ "L", key.l }, .{ "H", key.h }, .{ "KS", key.ks }, .{ "GS", key.gs } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
        dyn_conv_cfg = cfg;
        dyn_conv_key = key;
    }
    const ins = [_]mlx.mlx_array{ hidden, dynamic, base };
    const vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, dyn_conv_kernel.?, vec, dyn_conv_cfg.?, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

pub fn groupedDynConvOps(
    hidden: mlx.mlx_array,
    dynamic: mlx.mlx_array,
    base: mlx.mlx_array,
    group_size: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const hsh = mlx.getShape(hidden);
    // A lost GPU context hands back empty arrays: refuse them by name, never index past their shape.
    if (hsh.len != 3 or mlx.getShape(base).len != 2) return error.DynConvShape;
    const len = hsh[1];
    const h = hsh[2];
    const groups = @divExact(h, @as(c_int, @intCast(group_size)));
    const ksize: usize = @intCast(mlx.getShape(base)[0]);
    const dtype = mlx.mlx_array_dtype(hidden);

    const blk_shape = [_]c_int{ 1, len, groups, @intCast(group_size) };
    var blocks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blocks);
    try mlx.check(mlx.mlx_reshape(&blocks, hidden, &blk_shape, 4, s));

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_zeros(&out, &blk_shape, 4, dtype, s));

    var tap: usize = 0;
    while (tap < ksize) : (tap += 1) {
        // values = hidden shifted right by `tap`, zero-padded at the front.
        var values = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(values);
        if (tap == 0) {
            try mlx.check(mlx.mlx_array_set(&values, blocks));
        } else {
            const pad_shape = [_]c_int{ 1, @intCast(tap), groups, @intCast(group_size) };
            var pad = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad);
            try mlx.check(mlx.mlx_zeros(&pad, &pad_shape, 4, dtype, s));
            var head = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(head);
            const start = [_]c_int{ 0, 0, 0, 0 };
            const stop = [_]c_int{ 1, len - @as(c_int, @intCast(tap)), groups, @intCast(group_size) };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&head, blocks, &start, 4, &stop, 4, &strides, 4, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, pad);
            _ = mlx.mlx_vector_array_append_value(vec, head);
            try mlx.check(mlx.mlx_concatenate_axis(&values, vec, 1, s));
        }

        // base[tap] as [1, 1, groups, group_size] (per-channel).
        var kernel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kernel);
        {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            const start = [_]c_int{ @intCast(tap), 0 };
            const stop = [_]c_int{ @as(c_int, @intCast(tap)) + 1, h };
            const strides = [_]c_int{ 1, 1 };
            try mlx.check(mlx.mlx_slice(&row, base, &start, 2, &stop, 2, &strides, 2, s));
            const k_shape = [_]c_int{ 1, 1, groups, @intCast(group_size) };
            try mlx.check(mlx.mlx_reshape(&kernel, row, &k_shape, 4, s));
        }
        // dynamic[:, :, tap, :] as [1, L, groups, 1] (per-group).
        var dslice = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dslice);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 0, @intCast(tap), 0 };
            const stop = [_]c_int{ 1, len, @as(c_int, @intCast(tap)) + 1, groups };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, dynamic, &start, 4, &stop, 4, &strides, 4, s));
            const d_shape = [_]c_int{ 1, len, groups, 1 };
            try mlx.check(mlx.mlx_reshape(&dslice, cut, &d_shape, 4, s));
        }

        inline for (.{ kernel, dslice }) |coeff| {
            var term = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(term);
            try mlx.check(mlx.mlx_multiply(&term, coeff, values, s));
            var next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&next, out, term, s));
            _ = mlx.mlx_array_free(out);
            out = next;
        }
    }

    var flat = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, len, h };
    try mlx.check(mlx.mlx_reshape(&flat, out, &out_shape, 3, s));
    _ = mlx.mlx_array_free(out);
    return flat;
}

const ConvPrep = struct {
    hidden: mlx.mlx_array, // conv'd sublayer input
    finish_dyn: mlx.mlx_array, // [1, L, ksize, groups] — kernels for finish()
};

// ── DFlash2 path selector (forward + host trace) ──

pub const SelectedPath = struct {
    ids: []u32, // [m] chosen draft token ids
    chosen_idx: []u32, // [m] index of the choice within its candidate row
    cand_ids: []i32, // [m * top_k] candidate token ids per position
    /// [m * top_k] softmax(scores / temperature) per step — the proposal
    /// density q over the candidate set (zero everywhere else). Null on the
    /// greedy trace.
    q: ?[]f32,
    /// [m] the chosen candidate's share of its step's softmax (temperature 1, or the
    /// sampling temperature): how sure the selector is, which `dflash_policy` calibrates.
    conf: []f32,

};

/// The selector's candidate lattice on the host: `cands`/`unary` `[m, k]`
/// (candidate ids and their draft logits per position), `e0` `[k]` the anchor's
/// edges to position 0, `e` `[m-1, k, k]` edges between adjacent positions.
pub const Lattice = struct {
    m: usize,
    k: usize,
    cands: []i32,
    unary: []f32,
    e0: []f32,
    e: []f32,

};

// Top K of each row in one threadgroup: every thread keeps its own sorted K
// (a compare-and-select chain), each simdgroup merges its lanes' lists, then
// simdgroup 0 merges the NT / 32 lists. Equal values go to the lower index
// within a thread's own list; across lanes the lower lane wins, so a tie can
// pick the higher id (drafts only).
// 256 threads: M1/M2 cap threadgroups below 1024 for kernels this heavy.
const TOPK_SOURCE = kernelSource("kernels/spec/dflash_topk.metal");
var topk_kernel: ?mlx.mlx_fast_metal_kernel = null;
const TopKKey = struct { m: c_int, v: c_int, k: c_int, dt: mlx.mlx_dtype };
var topk_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var topk_key: ?TopKKey = null;

/// Each row's top `k` of `logits` [1, M, V]: ids [1, M, k] int32 and values
/// [1, M, k] f32, largest first. Null off the GPU or past 32 per row.
pub fn topKRows(logits: mlx.mlx_array, k: usize, s: mlx.mlx_stream) !?[2]mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or k == 0 or k > 32) return null;
    const sh = mlx.getShape(logits);
    const dt = mlx.mlx_array_dtype(logits);
    if (sh.len != 3 or sh[0] != 1 or sh[2] < 256 * @as(c_int, @intCast(k)) or (dt != .bfloat16 and dt != .float16 and dt != .float32)) return null;
    const key = TopKKey{ .m = sh[1], .v = sh[2], .k = @intCast(k), .dt = dt };
    if (topk_kernel == null) {
        const ins = [_][*:0]const u8{"logits"};
        const outs = [_][*:0]const u8{ "idx", "val" };
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kern = mlx.mlx_fast_metal_kernel_new("msv_dflash_topk_rows", in_vec, out_vec, TOPK_SOURCE, "", true, false);
        if (kern.ctx == null) return error.MetalKernelCompileFailed;
        topk_kernel = kern;
    }
    if (topk_key == null or !std.meta.eql(topk_key.?, key)) {
        if (topk_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        topk_cfg = null;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.m, key.k }, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.m, key.k }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, key.m, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", key.v));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
        topk_cfg = cfg;
        topk_key = key;
    }
    const ins = [_]mlx.mlx_array{logits};
    const vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, topk_kernel.?, vec, topk_cfg.?, s));
    var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, outs, i));
    return out;
}

/// A best-first draft tree over the lattice: node values are path sums of
/// log-softmax((unary + edge_w * pairwise) / temperature / tau) over siblings,
/// the `children` best candidates of each expanded node queued, the best queued
/// node taken next. Parameters follow TensorFold's fit (MIT): the head's raw
/// scores are overconfident and its pairwise term too strong.
pub const TreeParams = struct {
    max_nodes: usize,
    children: usize = 4,
    tau: f32 = 1.5,
    edge_w: f32 = 0.6,
    temperature: f32 = 1.0,
    /// `[m, k]` Gumbel noise the verify rows draw at the candidates, weighted
    /// into the scores (null for a greedy target).
    noise: ?[]const f32 = null,
    noise_w: f32 = 0.7,
};

/// Nodes in the order taken (a parent always before its children): `tokens`,
/// `parents` (node index, -1 = under the anchor) and `depth` (0 = position 0).
pub const DraftTree = struct {
    tokens: []u32,
    parents: []i32,
    depth: []u32,

};

// ── Context append + block forward ──

// ── Tests ──

const testing = std.testing;

const MUSE_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "architectures": ["MuseGlimmerAssistantModel"],
    \\  "block_size": 16,
    \\  "head_dim": 128,
    \\  "hidden_act": "silu",
    \\  "hidden_size": 6656,
    \\  "intermediate_size": 19968,
    \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    \\  "mask_token_id": 201818,
    \\  "max_position_embeddings": 131072,
    \\  "model_type": "muse_glimmer_assistant",
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 5,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-05,
    \\  "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    \\  "sliding_window": 2048,
    \\  "target_layer_ids": [1, 13, 25, 37, 49]
    \\}
;

// Gemma 4 assistant drafter shape (cross-attention drafter, NOT DFlash):
// no block_size / mask_token_id / target_layer_ids.
const GEMMA_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "model_type": "gemma4_assistant",
    \\  "backbone_hidden_size": 2560,
    \\  "num_centroids": 512,
    \\  "centroid_intermediate_top_k": 16,
    \\  "use_ordered_embeddings": true,
    \\  "text_config": {
    \\    "hidden_size": 256,
    \\    "num_hidden_layers": 4,
    \\    "num_attention_heads": 4,
    \\    "head_dim": 256,
    \\    "intermediate_size": 1024,
    \\    "sliding_window": 512,
    \\    "vocab_size": 262144,
    \\    "rms_norm_eps": 1e-06,
    \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    \\  }
    \\}
;

// The real incoai/Qwen3.8-27B-DFlash2 config shape (fetched):
// the contract nests under `dflash_config`, model_type is a bare "qwen3".
const DFLASH2_CONFIG_JSON =
    \\{
    \\  "architectures": ["DFlash2DraftModel"],
    \\  "is_causal": false,
    \\  "dflash_config": {
    \\    "block_size": 8,
    \\    "conv_group_size": 16,
    \\    "conv_kernel_size": 2,
    \\    "mask_token_id": 248070,
    \\    "selector_rank": 256,
    \\    "selector_top_k": 16,
    \\    "target_layer_ids": [5, 19, 33, 47, 61]
    \\  },
    \\  "head_dim": 128,
    \\  "hidden_size": 5120,
    \\  "intermediate_size": 17408,
    \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    \\  "max_position_embeddings": 262144,
    \\  "model_type": "qwen3",
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 5,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-06,
    \\  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    \\  "sliding_window": 2048,
    \\  "vocab_size": 248320
    \\}
;

// LiquidAI/LFM2.5-2.6B-DSpark, fetched. The contract is SPLIT:
// `block_size` sits at the root while `mask_token_id` / `target_layer_ids`
// nest under `dflash_config`, rope theta is a ROOT `rope_theta`, and the
// drafter's own rope is GPT-J interleaved (`rope_is_neox_style: false`).
const DSPARK_LFM2_CONFIG_JSON =
    \\{
    \\  "architectures": ["Lfm2DSparkDraftModel"],
    \\  "model_type": "qwen3",
    \\  "hidden_size": 2048,
    \\  "num_hidden_layers": 5,
    \\  "num_attention_heads": 32,
    \\  "num_key_value_heads": 8,
    \\  "head_dim": 64,
    \\  "intermediate_size": 6144,
    \\  "rms_norm_eps": 1e-05,
    \\  "vocab_size": 128000,
    \\  "rope_theta": 10000000.0,
    \\  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
    \\  "block_size": 9,
    \\  "dflash_config": {"mask_token_id": 125017, "target_layer_ids": [2, 9, 17, 21, 27], "num_target_layers": 30},
    \\  "markov_rank": 256,
    \\  "rope_is_neox_style": false,
    \\  "enable_confidence_head": true,
    \\  "markov_head_type": "vanilla"
    \\}
;

// ── Hermetic fixtures: tiny llama trunk + tiny DFlash assistant ──
// pub: generate.zig's nextDflash equivalence test builds the same pair.

pub const TinyFix = struct {
    // Dims are small but affine-quantizable: every contraction dim is a
    // multiple of 32, so the load-time quantization and the draft-only head
    // are exercisable hermetically (MLX affine groups are 32/64/128).
    pub const HIDDEN: usize = 64;
    pub const VOCAB: usize = 128;
    pub const HEAD_DIM: usize = 16;
    pub const N_HEADS: usize = 4;
    pub const N_KV: usize = 2;
    pub const INTER: usize = 128;
    pub const TRUNK_LAYERS: usize = 4;
    pub const ASSISTANT_LAYERS: usize = 2;
    pub const N_TARGETS: usize = 2;
    pub const BLOCK: usize = 4;

    // Tiny llama trunk: vocab 128, hidden 64, 4 layers, 4 q / 2 kv heads,
    // hd 16, inter 128, tied embeddings.
    // The trunk SLIDES at 8 on purpose: the greedy-equivalence test runs a
    // 12-token prompt out to 16 tokens, so every decode step and every verify
    // block sits past the window. That makes serial (width 1, view trimmed to
    // the window) vs dflash (width 4, trimmed to window + 3) the pairing guard
    // for `slidingTailSpan` and the relative mask offsets — a wrong trim or a
    // mask built against the untrimmed length diverges the two streams.
    pub const TRUNK_CONFIG =
        \\{
        \\  "model_type": "llama",
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "vocab_size": 128,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "tie_word_embeddings": true,
        \\  "sliding_window": 8,
        \\  "max_position_embeddings": 2048,
        \\  "torch_dtype": "bfloat16"
        \\}
    ;

    // Tiny assistant: hidden 64 (== trunk), 2 layers (sliding + full),
    // 4 q / 2 kv heads, hd 16, inter 128, window 8, block 4, mask id 127,
    // targets [0, 2].
    pub const ASSISTANT_CONFIG =
        \\{
        \\  "model_type": "tiny_assistant",
        \\  "block_size": 4,
        \\  "mask_token_id": 127,
        \\  "target_layer_ids": [0, 2],
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        \\  "sliding_window": 8,
        \\  "layer_types": ["sliding_attention", "full_attention"]
        \\}
    ;

    // DFlash2 tiny fixture: v1 dims + nested contract, selector rank 16 /
    // top_k 4 and 2-tap convs at group 8 (hidden 64 → 8 groups). model_type
    // is a bare "qwen3" — exactly the real checkpoint's shape.
    pub const SEL_RANK: usize = 16;
    pub const SEL_TOPK: usize = 4;
    pub const CONV_K: usize = 2;
    pub const CONV_GS: usize = 8;

    pub const ASSISTANT2_CONFIG =
        \\{
        \\  "architectures": ["DFlash2DraftModel"],
        \\  "model_type": "qwen3",
        \\  "is_causal": false,
        \\  "dflash_config": {
        \\    "block_size": 4,
        \\    "mask_token_id": 127,
        \\    "target_layer_ids": [0, 2],
        \\    "conv_kernel_size": 2,
        \\    "conv_group_size": 8,
        \\    "selector_rank": 16,
        \\    "selector_top_k": 4
        \\  },
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        \\  "sliding_window": 8,
        \\  "vocab_size": 128,
        \\  "layer_types": ["sliding_attention", "full_attention"]
        \\}
    ;

};

const OwnHeadFix = struct {
    const seed = 777;
};

// ── Env-gated real-checkpoint parity vs the executable reference ──
// Fixtures: tests/dump_dflash_fixtures.py (transformers 5.15 CPU fp32).

const ParityStats = struct { cos: f64, rms_ratio: f64 };
