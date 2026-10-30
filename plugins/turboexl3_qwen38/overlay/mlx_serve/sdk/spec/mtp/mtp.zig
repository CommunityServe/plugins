//! copied_from: src/sdk/spec/mtp/mtp.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Qwen 3.5/3.6 native MTP (multi-token prediction) head.
//!
//! Some Qwen 3.6 checkpoints ship a trained one-layer "MTP" sidecar
//! (`mtp/weights.safetensors`, ~15 tensors) that predicts the token AFTER the
//! next one from `(trunk_hidden, next_token)`. Chaining it K times drafts K
//! tokens which the trunk verifies in one batched forward — same
//! draft/verify contract as the Gemma 4 assistant drafter, but the drafter
//! is the model's own head, so acceptance stays high even on novel content.
//!
//! Architecture (matches mlx-lm `qwen3_5` MTP contract):
//!   x      = fc(concat([rmsnorm_e(embed(token)), rmsnorm_h(hidden)]))   [bf16 fc]
//!   x      = full-attention decoder layer(x)    — own 1-layer KV cache,
//!            q/gate split + sigmoid output gate, q/k per-head RMS norm,
//!            partial RoPE (rotary_factor * head_dim dims) at explicit offset
//!   post   = rmsnorm(x, mtp.norm)
//!   logits = trunk lm_head(post);  next-depth hidden = post
//!
//! The MTP layer keeps a COMMITTED-HISTORY KV cache: entry j pairs the trunk
//! hidden at position p_j with the token at p_j+1, built over the prompt at
//! prefill and maintained over committed tokens each decode round (drafts
//! append temporary entries; the round's commit restores the snapshot and
//! re-appends from true verify hiddens). Text-only RoPE offsets are
//! cache-relative ("cache" position mode). Multimodal requests additionally
//! map those cache positions to the trunk's absolute M-RoPE positions so the
//! sidecar and trunk agree on image-token geometry.
//!
//! Everything MTP-specific lives in this file plus `Generator.nextMtp`
//! (src/engines/mlx/generate.zig); deleting the feature is removing those two.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const weights_mod = @import("../../blocks/moe/weights.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const draft_fuse = @import("draft_fuse.zig");
/// Default draft depth (tokens drafted per round). Flipped 1 -> 3 after the
/// round-v2 rebuild made rejected drafts ~free (scalar-anchor rollback + the
/// 3-bit draft-only lm_head replaced the old full trunk re-forward): the old
/// cost model was what made depth 1 optimal. validation matrix on
/// Qwen3.6-27B 4-bit (M4 Max, adaptive controller active, decode tok/s
/// depth-3 vs depth-1): code 54.3 vs 43.3 (+25%), coding-agent ladder 2K
/// 52.1 vs 41.4 (+26%) and 16K 43.9 vs 38.1 (+15%), 2-turn pi agentic
/// (weighted) 48.6 vs 40.9 (+19%), creative temp-0.8 39.1 vs 37.9 (+3% —
/// the class that REGRESSED under the old cost model now holds even at ~30%
/// per-draft acceptance because the controller demotes without churn).
/// Users can cap rounds with `--mtp-max-depth`; the Generator's adaptive
/// controller demotes/promotes within [1, configured].
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;
/// `--max-mtp-ctx`: MTP stays off past this many prompt tokens (inclusive; 0 = unlimited).
/// A machine limit: `enable_mtp:true` in the body does not lift it. MTP only.
pub fn mtpCtxWithinLimit(max: u32, ctx_tokens: usize) bool {
    if (max == 0) return true;
    return ctx_tokens <= @as(usize, max);
}
/// Per-silicon adaptive depth cap for machines on the `.generic` cost
/// surface. The cap is a MACHINE measurement, so each row is one, never
/// interpolated between chips; an unmeasured chip keeps the default row.
///   M1 Pro: 4 (Qwen3.8-27B iQ-3.8bpw, forced-depth sweep:
///     13.01 tok/s at depth 4 vs 10.78/9.63 at 5/6 — the verify width 6
///     cliff; auto at cap 6 measured 10.64, barely over --no-mtp's 10.57).
/// `chip` is sysctl machdep.cpu.brand_string via `ane_mod.chipBrand`
/// (the GPU arch string cannot tell Ultra from Max); "" lands on default.
/// The row carries its own LABEL so the resolve site can say which one it
/// applied: a bare depth=4 in the spec-stats line is indistinguishable from
/// the EV controller having picked 4 on its own, or from `--mtp-max-depth 4`.
/// `measured` marks a row a HUMAN swept as realized throughput. Those beat the
/// boot probe's cost ladder, which cannot see acceptance or the extension sync
/// — see `generate.mtpDepthCapResolved`.
pub const DepthCap = struct { cap: u32, label: []const u8, measured: bool = false };
pub fn adaptiveDepthCapForMachine(chip: []const u8, default_cap: u32) DepthCap {
    if (std.mem.indexOf(u8, chip, "M1 Pro") != null) return .{ .cap = 4, .label = "m1-pro", .measured = true };
    // Base M4 (tester report, Qwen3.5-9B-MTPLX 6-bit, 16 GB):
    // saturated echo, median of 5, decode tok/s by forced depth —
    //   3: 47.44   4: 55.50   5: 47.50   6: 47.28
    // Depth 4 is the only width where the cap BINDS (m_lo == cap), so the
    // plan collapses to one chunk and pays no extension sync; from 5 on every
    // round pays it to buy ~1 more accepted token (4.00 -> 4.96/round) and
    // that trade is a 17% net LOSS there. Novel content sits at depth ~2
    // whatever the cap is, so capping at 4 costs nothing outside the echo
    // regime. The boot probe picks 6 on this chip — a cost ladder cannot see
    // either half of that sentence.
    if (std.mem.indexOf(u8, chip, "M4") != null and
        std.mem.indexOf(u8, chip, "M4 Pro") == null and
        std.mem.indexOf(u8, chip, "M4 Max") == null and
        std.mem.indexOf(u8, chip, "M4 Ultra") == null) return .{ .cap = 4, .label = "m4-base", .measured = true };
    if (std.mem.indexOf(u8, chip, "M4 Max") != null) return .{ .cap = default_cap, .label = "m4-max", .measured = true };
    // Base M5 only — the Pro/Max/Ultra dies are their own (unmeasured) rows.
    if (std.mem.indexOf(u8, chip, "M5") != null and
        std.mem.indexOf(u8, chip, "M5 Pro") == null and
        std.mem.indexOf(u8, chip, "M5 Max") == null and
        std.mem.indexOf(u8, chip, "M5 Ultra") == null) return .{ .cap = 4, .label = "m5", .measured = true };
    return .{ .cap = default_cap, .label = "default" };
}
/// Exact full-round cost surfaces known to the adaptive MTP controller.
/// Selection is based on runtime tensor geometry, never a model/repository
/// name. `generic` retains the conservative M1-M4 surface and auto cap.
pub const MtpCostProfile = enum {
    generic,
    g17_nax_q8_gs32,
    g17_nax_q4_gs32,
    g17_nax_q4_gs64,
    g17_nax_q6_gs64,
    g17_nax_q8_gs64,
    g17_nax_oq4e_q4_gs64,
    g17_nax_qwen4_q4_gs64,
    g17_nax_qwen4_mixed_4_8_gs64,
};
pub const HISTORY_WINDOW_THRESHOLD: usize = 16384;
/// One linear: quantized (w packed u32, s/b bf16) when `s.ctx != null`,
/// otherwise a pre-transposed bf16 weight `[in, out]` for plain matmul.
pub const QLinear = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    pub fn deinit(self: *QLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.s);
        _ = mlx.mlx_array_free(self.b);
    }
};
/// MTP MLP: dense SwiGLU (0.8B/27B-class sidecars) or the sparse MoE of a
/// qwen3_5_moe trunk (35B-A3B-class sidecars: router `mlp.gate` + packed
/// `switch_mlp` experts + shared expert + shared-expert gate). The MoE arm
/// stores the trunk's own `MoeMlpWeights` shape and forwards through
/// `Transformer.moeMLP` — same math, same gather-sort path, same per-weight
/// quant resolution (the sidecar mixes bits AND group sizes, e.g. 8-bit/gs-128
/// shared expert over a 4-bit/gs-64 trunk — `affineParamsFromGeometry`).
const MtpMlp = union(enum) {
    dense: struct {
        gate: QLinear,
        up: QLinear,
        down: QLinear,
    },
    moe: weights_mod.MoeMlpWeights,
    fn deinit(self: *MtpMlp) void {
        switch (self.*) {
            .dense => |*d| {
                d.gate.deinit();
                d.up.deinit();
                d.down.deinit();
            },
            .moe => |*m| {
                const arrs = [_]mlx.mlx_array{
                    m.router_w,      m.router_s,      m.router_b,
                    m.switch_gate_w, m.switch_gate_s, m.switch_gate_b,
                    m.switch_up_w,   m.switch_up_s,   m.switch_up_b,
                    m.switch_down_w, m.switch_down_s, m.switch_down_b,
                    m.shared_gate_w, m.shared_gate_s, m.shared_gate_b,
                    m.shared_up_w,   m.shared_up_s,   m.shared_up_b,
                    m.shared_down_w, m.shared_down_s, m.shared_down_b,
                };
                for (arrs) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_w) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_s) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_b) |a| _ = mlx.mlx_array_free(a);
                if (m.expert_bias) |a| _ = mlx.mlx_array_free(a);
            },
        }
    }
};
/// Which head shape the sidecar loaded as. Everything the shapes share reads
/// the same fields; the few forks (attention gate, QK norms, MLP kind) key here
/// or on `eh_proj`.
pub const Layout = enum { qwen, hy3 };
pub const MtpModel = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    /// Quant params for the MTP layer's own linears — inferred from tensor
    /// geometry at load (sidecars are often quantized differently from the
    /// trunk, e.g. group 32 over a group-64 trunk, or 8-bit over 4-bit).
    quant_bits: u32,
    quant_group_size: u32,
    /// Quant MODE the sidecar's packed weights are in. A sidecar file carries
    /// no config, so this is solved from geometry at load
    /// (`params_mod.quantParamsFromGeometry`) and handed to every
    /// `mlx_quantized_matmul` — passing "affine" over fp8 scales throws.
    quant_mode: QuantMode = .affine,
    /// Qwen heads' concat projection (empty for Hy3). Dense bf16 ships
    /// pre-transposed `[2H, H]` for plain matmul; a checkpoint that ships it
    /// QUANTIZED (avlp12 Alis) keeps `(w, scales, biases)` verbatim — packed
    /// `[out=H, in=2H]` is already what quantized_matmul(transpose) wants.
    fc: QLinear,
    /// Hy3 (hy_v3) heads: the concat projection ships QUANTIZED as
    /// `mtp.eh_proj` instead of Qwen's bf16 `mtp.fc`. Non-null selects the
    /// Hy3 layer shape everywhere it differs: eh_proj replaces the fc matmul
    /// and the attention has NO output gate (FrontOut.gate stays a null-ctx
    /// handle; backChain skips the sigmoid multiply).
    eh_proj: ?QLinear = null,
    pre_fc_norm_emb: mlx.mlx_array, // Qwen pre_fc_norm_embedding / Hy3 enorm
    pre_fc_norm_hidden: mlx.mlx_array, // Qwen pre_fc_norm_hidden / Hy3 hnorm
    final_norm: mlx.mlx_array, // mtp.norm
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    mlp: MtpMlp,
    layout: Layout = .qwen,
    /// Optional DRAFT-ONLY low-bit lm_head, requantized from the trunk's at
    /// bind time (`DRAFT_HEAD_BITS`).
    /// Only draft steps project through it — VERIFICATION always uses the
    /// trunk head, so the output distribution is untouched; drafts just read
    /// ~40% fewer bytes per full-vocab projection (the dominant draft cost).
    draft_head: ?QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,
    /// Draft-rerank scheme: a coarse 2-bit/gs64
    /// requant of the trunk lm_head; greedy drafts shortlist its top-32 and
    /// re-score them through the trunk head's own rows (`draftSelect`). When
    /// built, the 3-bit draft head is dropped — non-greedy draft paths fall
    /// back to the trunk head.
    rerank_coarse: ?RerankCoarse = null,
    rerank_logged: bool = false,
    /// Cross-request EV controller seed (inference thread only, like every
    /// mutable field here): the last HEALTHY request's per-index acceptance
    /// EMAs + base depth, written by `Generator.deinit`, consumed by the
    /// first `nextMtp` round of the next request. A fresh controller burns
    /// ~10 legacy-warmup rounds plus a +1/round base climb per request —
    /// a third of a short protocol-style generation; seeding restores the
    /// learned surface from round 1. Never written by disabled/short runs.
    ev_seed_accept: ?[MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,
    pub fn deinit(self: *MtpModel) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        if (self.draft_head) |*dh| dh.deinit();
        if (self.eh_proj) |*ep| ep.deinit();
        self.fc.deinit();
        _ = mlx.mlx_array_free(self.pre_fc_norm_emb);
        _ = mlx.mlx_array_free(self.pre_fc_norm_hidden);
        _ = mlx.mlx_array_free(self.final_norm);
        _ = mlx.mlx_array_free(self.input_norm);
        _ = mlx.mlx_array_free(self.post_attn_norm);
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.mlp.deinit();
    }
};
// ── Draft rerank: the head-agnostic half ──
//
// Nothing in this scheme is MTP-head-shaped. It reads the TARGET's lm_head
// and a vector `x` in that head's input space, and answers with a token id —
// so the qwen3.5/3.6 sidecar (`MtpModel`) and the qwen4_exp in-checkpoint
// head (`Transformer.qwen4_mtp`) share it verbatim instead of the qwen4 arm
// growing a second copy that could drift from the measured one.
//
// What `x` IS, is per-arm and load-bearing: whatever the lm_head consumes.
// On the sidecar that is the head's own post-norm hidden; on qwen4_exp it is
// the MIXER output, NOT the pre-mixer stream the head hands back as
// `hidden_next`.
/// Coarse-head width AT BUILD TIME (0 disables the build). A coarser head is fewer bytes and a longer tail of
/// shortlist misses, and a miss costs acceptance, never output. The width a head is READ at is a property of its
/// packed weights (`RerankCoarse.bits`): packed geometry implies K, so a read at another width reads a different
/// vocab entirely.
pub const DRAFT_HEAD_BITS: u32 = 3;
/// A BUILT coarse readout head: the packed weights plus the geometry they
/// were packed with. Both live together because `mlx_quantized_matmul` needs
/// all three and only the build knows them.
pub const RerankCoarse = struct {
    q: QLinear,
    bits: u32,
    group_size: u32,
    rows: c_int,
    /// The re-score's rows when the head's own cannot be gathered (a trunk format's head, `buildRerankCoarseRows`): a
    /// `RERANK_EXACT_BITS`/gs64 copy of the rows it computes. Null = re-score through `target.lm_head_*`.
    exact: ?QLinear = null,
    /// A copy of `q`'s leading `sub_rows` rows (`withVocabSubset`), which a single-row draft reads instead of `q`. A
    /// shortlist's `rows` (a dense proposal row's width) stays `rows`, the vocab.
    sub: ?QLinear = null,
    sub_rows: c_int = 0,
    pub const Readout = struct { q: QLinear, rows: c_int };
    /// The coarse rows a single-row draft reads: the vocab subset when one is built, else the whole head.
    pub fn readout(self: *const RerankCoarse) Readout {
        if (self.sub) |q| return .{ .q = q, .rows = self.sub_rows };
        return .{ .q = self.q, .rows = self.rows };
    }
    pub fn deinit(self: *RerankCoarse) void {
        self.q.deinit();
        if (self.exact) |*e| e.deinit();
        if (self.sub) |*e| e.deinit();
    }
};
/// Width of a trunk format head's re-score copy (`RerankCoarse.exact`).
pub const RERANK_EXACT_BITS: u32 = 8;
/// The vocab subset a trunk format head's single-row drafts read (`withVocabSubset`), on a vocab of DRAFT_VOCAB_MIN ids
/// or more: the first 96 Ki ids. A byte-level BPE vocab numbers its merges most frequent first, so these carry nearly
/// all text; a draft never proposes a later id and the verify reads the full head, so it costs acceptance, never
/// output.
pub const DRAFT_VOCAB_ROWS: c_int = 98304;
pub const DRAFT_VOCAB_MIN: c_int = 131072;
/// Resident bytes of a coarse head over `rows` x `hidden` at `bits`/gs64:
/// packed weights plus bf16 scales AND biases. Reported at build so the log
/// line carries the number the change is claiming.
pub fn rerankCoarseBytes(rows: c_int, hidden: c_int, bits: u32) u64 {
    const r: u64 = @intCast(rows);
    const h: u64 = @intCast(hidden);
    const packed_bytes = r * h * @as(u64, bits) / 8;
    const groups = h / 64;
    return packed_bytes + 2 * (r * groups * 2);
}
/// The exact re-scored shortlist behind every rerank draft: the coarse head's
/// top-32 candidate ids plus their trunk-head logits. A greedy draft is the
/// argmax over `exact`; a sampled draft is drawn from the filtered softmax over
/// it, so q's support is exactly these ids.
pub const Shortlist = struct {
    /// `[32]` uint32 vocab ids.
    cands: mlx.mlx_array,
    /// `[.., 32]` trunk-head logits for those ids.
    exact: mlx.mlx_array,
    /// The head's vocab rows — the width a dense proposal row must have.
    rows: c_int,
    pub fn deinit(self: *Shortlist) void {
        _ = mlx.mlx_array_free(self.cands);
        _ = mlx.mlx_array_free(self.exact);
    }
};
/// Load a `<prefix>.{weight,scales?,biases?}` triple raw (no transpose) —
/// the shape the trunk's gather/qmatmul paths expect for MoE tensors.
const MoeTriple = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    fn deinit(self: *MoeTriple) void {
        for ([_]mlx.mlx_array{ self.w, self.s, self.b }) |a| _ = mlx.mlx_array_free(a);
    }
};
// ── Forward ──
/// Outputs of the pre-rope half of the MTP layer. All arrays owned.
const FrontOut = struct {
    q_t: mlx.mlx_array, // [1, H, L, D] normed, pre-rope
    k_t: mlx.mlx_array, // [1, Hkv, L, D] normed, pre-rope
    v_t: mlx.mlx_array, // [1, Hkv, L, D]
    gate: mlx.mlx_array, // [1, L, H, D] raw output gate (pre-sigmoid, strided split view)
    x: mlx.mlx_array, // [1, L, H] fc output — the residual input
    fn deinit(self: *FrontOut) void {
        _ = mlx.mlx_array_free(self.q_t);
        _ = mlx.mlx_array_free(self.k_t);
        _ = mlx.mlx_array_free(self.v_t);
        _ = mlx.mlx_array_free(self.gate);
        _ = mlx.mlx_array_free(self.x);
    }
};
// ── Draft-rerank shortlist (port of the mlx.fast challenge draft-rerank
// scheme, submission 942e5ab2 in mlx.fast-qwen-3.8-mtp-challenge) ──
//
// PROPOSAL SIDE ONLY: a coarse 2-bit requant of the trunk lm_head scores the
// full vocab, an exact top-32 shortlist is taken in TWO dispatches (MLX's
// GPU argpartition is a full multi-block argsort — 14 dependent dispatches
// for 32 read values), and the trunk head's own 32 rows re-score the
// shortlist. The draft is the trunk's argmax over those rows, so whenever
// the coarse top-32 contains the trunk argmax the draft IS the trunk's
// choice — acceptance rides the trunk's order at ~2-bit readout cost.
// Verification is untouched: a coarse miss costs acceptance, never output.
//
// Ordinal: a monotone map from float into uint32 inducing (value asc, NaN
// above every number, -0.0 == +0.0). Real values never map to 0, so the
// zero-initialized empty slots below can never be selected while a
// simdgroup still has real candidates (guaranteed by rows >= TILES*TG).
// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32
// as (ordinal, index) pairs — TILES * 32 candidates. Selection is 32 rounds
// of simd_max over per-thread slots with a `taken` bitmask (hence the
// PER_THREAD <= 32 static_assert), then one simdgroup reduces the
// threadgroup's per-simd lists the same way.
// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32
// ids (written ascending by (value, index) — consumers are order-blind).
// ── Tests ──
test {
    _ = draft_fuse;
}
// Test-only: run the negFraction threshold on a single 1-D norm array.
// Test-only: fold only when the single array reads as delta-encoded.
// Test-only: max |a-b| over all elements.
