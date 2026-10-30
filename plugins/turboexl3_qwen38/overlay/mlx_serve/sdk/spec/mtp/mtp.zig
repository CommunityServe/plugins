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
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");
const weights_load = @import("../../weights/load.zig");
const axisView = @import("../../../mlx/array.zig").axisView;
const kv_quant = @import("../../blocks/attention/kv_quant.zig");
const cache_mod = @import("../../kv/cache.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa_mod = @import("../../blocks/attention/sdpa.zig");
const verify_projection = @import("../../blocks/linear/verify_projection.zig");
const verify_qmm = @import("../../blocks/linear/verify_qmm.zig");
const verify_qmm_nax = @import("../../blocks/linear/verify_qmm_nax.zig");
const weights_mod = @import("../../blocks/moe/weights.zig");
const params_mod = @import("../../quant/params.zig");
const log = @import("../../util/log.zig");
const io_util_mod = @import("../../util/io_util.zig");
const ane_mod = @import("../../ane/offload.zig");
const sushi_pack = @import("../../quant/exl3/sushi_pack.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;

const KVCache = cache_mod.KVCache;
const Weights = @import("../../weights/resident.zig").Weights;
const mtp_source = @import("source.zig");
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

/// Target-side tensors that contribute materially to a complete MTP round.
/// This is deliberately a single classification rather than independent
/// booleans: one bound target can match at most one measured cost surface.
pub const MtpNaxTargetSurface = enum {
    none,
    uniform_quantized_embedding,
    uniform_bf16_embedding,
    uniform_q6_quantized_embedding,
    uniform_q8_bf16_embedding,
    oqe_quantized_embedding,
};

/// The trunk geometry a head's fingerprint is read against.
pub const HeadDims = struct { hidden: u32, heads: u32, kv_heads: u32, head_dim: u32, intermediate: u32 };

/// A head that is the measured dense layout, at its projections' width.
pub const HeadFingerprint = struct { bits: u32, group_size: u32 };

/// Prefill history windowing (OPT-IN via `--mtp-history-window <n>`; mirrors
/// others `last_window 8192` above a 16384-token threshold): prompts whose
/// forwarded tail exceeds the threshold only build MTP history for the LAST
/// n positions — earlier chunks skip the full-hidden capture AND the head
/// forward entirely (and become eligible for the compiled trunk forward).
/// A history that starts mid-sequence is already a supported state: warm
/// hot-cache hits produce exactly that (RoPE offsets are cache-relative).
/// DEFAULT IS FULL HISTORY (0): the A/B failed for windowing on the stock
/// Qwen head — 64K ctx measured 68.2% -> 54.0% per-draft acceptance and
/// -4.2 decode tok/s for zero prefill benefit.
/// `SUGGESTED_HISTORY_WINDOW` is what to pass when
/// experimenting with window-trained sidecars.
pub const SUGGESTED_HISTORY_WINDOW: usize = 8192;
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

const M5NaxDenseSidecarLinears = struct {
    q: *const QLinear,
    k: *const QLinear,
    v: *const QLinear,
    o: *const QLinear,
    gate: *const QLinear,
    up: *const QLinear,
    down: *const QLinear,
};

const M5NaxDenseSidecarGeometry = struct {
    hidden: u32,
    q_out: u32,
    kv_out: u32,
    full_out: u32,
    intermediate: u32,
    bits: u32,
    group_size: u32,
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

/// Group size every coarse head is packed at.
pub const RERANK_COARSE_GROUP: u32 = 64;

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

/// A contiguous copy of `q`'s leading `n` rows (packed weights and both group tables), unevaluated; the caller frees
/// it.
fn leadingCopy(s: mlx.mlx_stream, q: QLinear, n: c_int) !QLinear {
    var parts: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (parts) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    for ([_]mlx.mlx_array{ q.w, q.s, q.b }, &parts) |a, *out| {
        const sh = mlx.getShape(a);
        if (sh.len != 2) return error.UnsupportedDraftHeadShape;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, a, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ n, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
        out.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_copy(out, view, s));
    }
    return .{ .w = parts[0], .s = parts[1], .b = parts[2] };
}

/// `rc` plus a copy of its leading `keep` rows (`RerankCoarse.sub`), evaluated. A `keep` under TOP32_MIN_ROWS or not
/// under the vocab, or a refused copy, leaves the whole head alone.
pub fn withVocabSubset(s: mlx.mlx_stream, rc: RerankCoarse, keep: c_int) RerankCoarse {
    if (keep < TOP32_MIN_ROWS or keep >= rc.rows) return rc;
    var sub = leadingCopy(s, rc.q, keep) catch |err| {
        log.warn("[mtp] draft rerank vocab subset failed ({s}) - drafting over the whole vocab\n", .{@errorName(err)});
        return rc;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    for ([_]mlx.mlx_array{ sub.w, sub.s, sub.b }) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        sub.deinit();
        log.warn("[mtp] draft rerank vocab subset eval failed - drafting over the whole vocab\n", .{});
        return rc;
    };
    var out = rc;
    out.sub = sub;
    out.sub_rows = keep;
    return out;
}

/// `buildRerankCoarse` for a head whose rows exist only as what it computes (a trunk format's head): `dense`, those
/// `[V, H]` rows, requantized to the coarse width and to the re-score copy (`RerankCoarse.exact`), with the vocab
/// subset on a vocab of DRAFT_VOCAB_MIN ids or more. Proposal side only: the verify reads the head itself, so its bytes
/// stand.
pub fn buildRerankCoarseRows(s: mlx.mlx_stream, dense: mlx.mlx_array, bits: u32) ?RerankCoarse {
    if (bits == 0 or bits >= RERANK_EXACT_BITS) return null;
    mlx.check(mlx.mlx_array_eval(dense)) catch {
        log.warn("[mtp] draft rerank: the head's rows failed to evaluate - keeping the full-readout path\n", .{});
        return null;
    };
    const sh = mlx.getShape(dense);
    if (sh.len != 2 or sh[0] < TOP32_MIN_ROWS) return null;
    const none: mlx.mlx_array = .{ .ctx = null };
    var coarse = requantizeRows(s, dense, none, none, 0, 16, "affine", RERANK_COARSE_GROUP, bits, 32768) catch |err| {
        log.warn("[mtp] draft rerank coarse build failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    var exact = requantizeRows(s, dense, none, none, 0, 16, "affine", RERANK_COARSE_GROUP, RERANK_EXACT_BITS, 32768) catch |err| {
        coarse.deinit();
        log.warn("[mtp] draft rerank re-score copy failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    for ([_]mlx.mlx_array{ coarse.w, coarse.s, coarse.b, exact.w, exact.s, exact.b }) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        coarse.deinit();
        exact.deinit();
        log.warn("[mtp] draft rerank coarse eval failed - keeping the full-readout path\n", .{});
        return null;
    };
    const rc: RerankCoarse = .{ .q = coarse, .bits = bits, .group_size = RERANK_COARSE_GROUP, .rows = sh[0], .exact = exact };
    return withVocabSubset(s, rc, if (sh[0] >= DRAFT_VOCAB_MIN) DRAFT_VOCAB_ROWS else 0);
}

/// The coarse readout head: a low-bit/gs64 requantized copy of the target's
/// lm_head, materialized so the first draft does not pay for it. Infallible —
/// every refusal returns null and the caller keeps its full-readout path.
pub fn buildRerankCoarse(s: mlx.mlx_stream, target: anytype, bits: u32) ?RerankCoarse {
    if (bits == 0) return null;
    if (comptime @hasField(@TypeOf(target.*), "rht")) {
        if (target.rht != null) return null;
    }
    if (sushi_pack.isRowScaledFp8(target.lm_head_w, target.lm_head_s)) return null; // e4m3 head: no affine rows to gather
    const w_shape = mlx.getShape(target.lm_head_w);
    if (w_shape.len != 2 or w_shape[0] < TOP32_MIN_ROWS) return null;
    // A dense bf16 head ([V, H], 16 bits) always reads more than the coarse copy.
    const dense = target.lm_head_s.ctx == null;
    const head_qp: params_mod.QuantParams = if (dense) .{ .bits = 16, .group_size = 0, .mode = .affine } else headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    if (bits >= head_qp.bits) return null; // no byte saving over the head we already read

    var rc = requantizeRows(
        s,
        target.lm_head_w,
        target.lm_head_s,
        target.lm_head_b,
        head_qp.group_size,
        head_qp.bits,
        head_qp.mode.cstr(),
        RERANK_COARSE_GROUP,
        bits,
        32768,
    ) catch |err| {
        log.warn("[mtp] draft rerank coarse build failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.w);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.s);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.b);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        rc.deinit();
        log.warn("[mtp] draft rerank coarse eval failed - keeping the full-readout path\n", .{});
        return null;
    };
    return .{ .q = rc, .bits = bits, .group_size = RERANK_COARSE_GROUP, .rows = w_shape[0] };
}

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

/// `argmax(where(mask, -inf, logits))` as a lazy [.., 1] int32 id - the tail
/// every full-readout draft path shares.
pub fn maskAndArgmax(s: mlx.mlx_stream, logits_in: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_set(&logits, logits_in));
    if (suppress_mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        _ = mlx.mlx_array_free(logits);
        logits = masked;
    }
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, logits, -1, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, amax, .int32, s));
    return out;
}

/// Full trunk-lm_head readout + argmax. The qwen4 arm's fallback: unlike the
/// sidecar's, there is no low-bit draft head to prefer.
pub fn fullReadoutArgmax(
    s: mlx.mlx_stream,
    target: anytype,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    const logits = try target.lmHeadForeign(x);
    defer _ = mlx.mlx_array_free(logits);
    return maskAndArgmax(s, logits, suppress_mask);
}

/// `-inf` on every suppressed id of a coarse readout, in place.
fn rerankMaskCoarse(s: mlx.mlx_stream, logits: *mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !void {
    // The mask goes on the COARSE row: a reserved id that never enters the
    // shortlist cannot be re-scored back into the draft.
    const m = suppress_mask orelse return;
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits.*, s));
    _ = mlx.mlx_array_free(logits.*);
    logits.* = masked;
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

/// Shortlist one coarse row `flat` `[rows]` and re-score it through the trunk head.
/// Null retires the coarse head, so every later step lands on the caller's fallback.
fn rerankShortlistRow(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    flat: mlx.mlx_array,
    rows: c_int,
    x_row: mlx.mlx_array,
) !?Shortlist {
    const cands = draftTop32(s, flat, rows) catch |err| {
        log.warn("[mtp] draft rerank shortlist failed ({s}) - dropping to the full readout\n", .{@errorName(err)});
        var dead = coarse.*.?;
        dead.deinit();
        coarse.* = null;
        return null;
    };
    errdefer _ = mlx.mlx_array_free(cands);

    // Re-score the shortlist through the trunk head's own rows: gathered
    // packed rows are self-contained (w [V, K*bits/32], scales/biases
    // [V, K/gs]), so a 32-row quantized matmul is exact; dense rows are a plain matmul.
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    if (target.lm_head_s.ctx == null) {
        try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w32, s));
        var dense = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_matmul(&dense, x_row, wt, s));
        return .{ .cands = cands, .exact = dense, .rows = rows };
    }
    const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    var s32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s32, target.lm_head_s, cands, 0, s));
    if (target.lm_head_b.ctx != null)
        try mlx.check(mlx.mlx_take_axis(&b32, target.lm_head_b, cands, 0, s));

    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(
        &exact,
        x_row,
        w32,
        s32,
        b32,
        true,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        s,
    ));
    return .{ .cands = cands, .exact = exact, .rows = rows };
}

/// The greedy draft's tail: argmax over the re-scored shortlist, mapped back to
/// a vocab id as a lazy int32 array.
pub fn shortlistArgmax(s: mlx.mlx_stream, sl: Shortlist) !mlx.mlx_array {
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, sl.exact, -1, false, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, amax, 0, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, picked, .int32, s));
    return out;
}

/// The coarse full-vocab readout of one row as a flat `[rows]` array, reserved
/// ids already at `-inf` — the front half every rerank draft shares.
fn rerankCoarseFlat(
    s: mlx.mlx_stream,
    rc: *const RerankCoarse,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var coarse_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coarse_logits);
    try mlx.check(mlx.mlx_quantized_matmul(
        &coarse_logits,
        x,
        rc.q.w,
        rc.q.s,
        rc.q.b,
        true,
        mlx.mlx_optional_int.some(@intCast(rc.group_size)),
        // The BUILT width and group, never `DRAFT_HEAD_BITS`: the packed
        // geometry implies K, so a per-step env re-read that disagrees with the
        // build does not read a coarser vocab - it reads a different one.
        mlx.mlx_optional_int.some(@intCast(rc.bits)),
        "affine",
        s,
    ));
    try rerankMaskCoarse(s, &coarse_logits, suppress_mask);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rc.rows};
    try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
    return flat;
}

/// `rerankSelect` without its argmax tail: the caller owns the shortlist and
/// decides the proposal over it (argmax for a greedy request, a draw from the
/// draft sampler's filtered softmax for a sampled one). Null means the coarse
/// head is unavailable, exactly as in `rerankSelect`.
pub fn rerankShortlist(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?Shortlist {
    // A trunk format head's coarse head (`buildRerankCoarseRows`) reads its vocab subset and its re-score copy.
    if (coarse.*) |c| if (c.exact != null or c.sub != null) return rerankShortlistFormat(s, target, coarse, logged, x, suppress_mask);
    const rc = if (coarse.*) |*p| p else return null;
    const coarse_bits = rc.bits;
    const rows = rc.rows;

    const flat = try rerankCoarseFlat(s, rc, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);

    const sl = (try rerankShortlistRow(s, target, coarse, flat, rows, x)) orelse return null;
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{coarse_bits});
        logged.* = true;
    }
    return sl;
}

/// One greedy draft proposal via the rerank scheme: coarse full-vocab readout
/// -> exact top-32 shortlist (two dispatches) -> the trunk head's own 32 rows
/// re-score -> argmax. Returns a lazy [1,1] int32 token id, or NULL when the
/// coarse head is unavailable - never built, or a shortlist-kernel failure
/// just dropped it (`coarse` is freed and nulled in that case, so every later
/// step of the same chain lands on the caller's fallback too).
///
/// PROPOSAL SIDE ONLY: verification still reads the trunk forward's logits, so
/// a coarse miss costs acceptance, never output.
pub fn rerankSelect(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?mlx.mlx_array {
    var sl = (try rerankShortlist(s, target, coarse, logged, x, suppress_mask)) orelse return null;
    defer sl.deinit();
    return try shortlistArgmax(s, sl);
}

/// `rerankMaskCoarse` on a trunk format head's readout: a vocab subset's (`RerankCoarse.sub`) is narrower than the
/// mask, and its ids are the mask's leading ones.
fn rerankMaskCoarseFormat(s: mlx.mlx_stream, logits: *mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !void {
    // The mask goes on the COARSE row: a reserved id that never enters the
    // shortlist cannot be re-scored back into the draft.
    const full = suppress_mask orelse return;
    const msh = mlx.getShape(full);
    const lsh = mlx.getShape(logits.*);
    var lead = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lead);
    var m = full;
    if (msh.len == 1 and lsh.len >= 1 and msh[0] > lsh[lsh.len - 1]) {
        try mlx.check(mlx.mlx_slice(&lead, full, &[_]c_int{0}, 1, &[_]c_int{lsh[lsh.len - 1]}, 1, &[_]c_int{1}, 1, s));
        m = lead;
    }
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits.*, s));
    _ = mlx.mlx_array_free(logits.*);
    logits.* = masked;
}

/// `rerankShortlistRow` for a trunk format head: re-scored through its copy (`RerankCoarse.exact`), a vocab subset's
/// shortlist still spanning the vocab.
fn rerankShortlistRowFormat(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    flat: mlx.mlx_array,
    rows: c_int,
    x_row: mlx.mlx_array,
) !?Shortlist {
    // `rows` is the coarse readout's width; a vocab subset's shortlist still spans the vocab
    const vocab: c_int = if (coarse.*) |rc| rc.rows else rows;
    // A trunk format's head has no rows to gather: its re-score copy stands in (`RerankCoarse.exact`).
    const copy: ?QLinear = if (coarse.*) |rc| rc.exact else null;
    const cands = draftTop32(s, flat, rows) catch |err| {
        log.warn("[mtp] draft rerank shortlist failed ({s}) - dropping to the full readout\n", .{@errorName(err)});
        var dead = coarse.*.?;
        dead.deinit();
        coarse.* = null;
        return null;
    };
    errdefer _ = mlx.mlx_array_free(cands);

    // Re-score the shortlist through the trunk head's own rows: gathered
    // packed rows are self-contained (w [V, K*bits/32], scales/biases
    // [V, K/gs]), so a 32-row quantized matmul is exact; dense rows are a plain matmul.
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    if (copy == null and target.lm_head_s.ctx == null) {
        try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w32, s));
        var dense = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_matmul(&dense, x_row, wt, s));
        return .{ .cands = cands, .exact = dense, .rows = vocab };
    }
    const src: QLinear = copy orelse .{ .w = target.lm_head_w, .s = target.lm_head_s, .b = target.lm_head_b };
    const qp: params_mod.QuantParams = if (copy != null)
        .{ .bits = RERANK_EXACT_BITS, .group_size = RERANK_COARSE_GROUP, .mode = .affine }
    else
        headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    var s32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_take_axis(&w32, src.w, cands, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s32, src.s, cands, 0, s));
    if (src.b.ctx != null)
        try mlx.check(mlx.mlx_take_axis(&b32, src.b, cands, 0, s));

    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(
        &exact,
        x_row,
        w32,
        s32,
        b32,
        true,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        s,
    ));
    return .{ .cands = cands, .exact = exact, .rows = vocab };
}

/// `rerankCoarseFlat` over `head` (`RerankCoarse.readout`: a trunk format head's vocab subset when one is built).
fn rerankCoarseFlatFormat(
    s: mlx.mlx_stream,
    rc: *const RerankCoarse,
    head: RerankCoarse.Readout,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var coarse_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coarse_logits);
    try mlx.check(mlx.mlx_quantized_matmul(
        &coarse_logits,
        x,
        head.q.w,
        head.q.s,
        head.q.b,
        true,
        mlx.mlx_optional_int.some(@intCast(rc.group_size)),
        // The BUILT width and group, never `DRAFT_HEAD_BITS`: the packed
        // geometry implies K, so a per-step env re-read that disagrees with the
        // build does not read a coarser vocab - it reads a different one.
        mlx.mlx_optional_int.some(@intCast(rc.bits)),
        "affine",
        s,
    ));
    try rerankMaskCoarseFormat(s, &coarse_logits, suppress_mask);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{head.rows};
    try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
    return flat;
}

/// `rerankShortlist` for a trunk format head (`buildRerankCoarseRows`): its vocab subset's readout, re-scored through
/// its copy.
fn rerankShortlistFormat(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?Shortlist {
    const rc = if (coarse.*) |*p| p else return null;
    const coarse_bits = rc.bits;
    // the vocab subset's rows when one is built
    const head = rc.readout();

    const flat = try rerankCoarseFlatFormat(s, rc, head, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);

    const sl = (try rerankShortlistRowFormat(s, target, coarse, flat, head.rows, x)) orelse return null;
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{coarse_bits});
        logged.* = true;
    }
    return sl;
}

/// The fused draft pick (`draft_fuse.zig`): `rerankSelect` with the finalize and the re-score in one launch and MLX's
/// own argmax, then one launch for the take + astype that also builds the next draft step's front. The coarse readout
/// and its mask are `rerankCoarseFlat`'s (a trunk format head's: `rerankCoarseFlatFormat`'s), the candidates
/// `draftTop32`'s first launch, the re-score rows `rerankShortlistRow`'s (`rerankShortlistRowFormat`'s); the id is
/// `shortlistArgmax`'s ([1,1] int32), bit for bit. Proposal side only, like
/// `rerankSelect`. Null = no coarse head, or a declined call (logged): the caller runs `rerankSelect`.
pub fn rerankPickFused(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
    front: draft_fuse.Front,
) !?draft_fuse.Pick {
    const rc = if (coarse.*) |*p| p else return null;
    const head = rc.readout();
    const copy: ?QLinear = rc.exact;
    if (copy == null and target.lm_head_s.ctx == null) {
        draft_fuse.decline(.rescore_rows);
        return null;
    }
    const src: QLinear = copy orelse .{ .w = target.lm_head_w, .s = target.lm_head_s, .b = target.lm_head_b };
    const qp: params_mod.QuantParams = if (copy != null)
        .{ .bits = RERANK_EXACT_BITS, .group_size = RERANK_COARSE_GROUP, .mode = .affine }
    else
        headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    if (qp.mode != .affine) {
        draft_fuse.decline(.rescore_quant);
        return null;
    }
    const rescore: draft_fuse.Rescore = .{ .w = src.w, .sc = src.s, .bi = src.b, .bits = qp.bits, .group_size = qp.group_size };
    if (draft_fuse.rescoreReason(x, rescore, front.hidden, head.rows)) |r| {
        draft_fuse.decline(r);
        return null;
    }
    if (!top32RowsOk(head.rows)) {
        draft_fuse.decline(.shortlist);
        return null;
    }
    // the shortlist's ids index the embedding table in the pick
    if (head.rows > mlx.getShape(front.emb)[0]) {
        draft_fuse.decline(.vocab);
        return null;
    }
    const flat = if (rc.exact != null or rc.sub != null) try rerankCoarseFlatFormat(s, rc, head, x, suppress_mask) else try rerankCoarseFlat(s, rc, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);
    const cands = draftTop32Cands(s, flat, head.rows) catch {
        draft_fuse.decline(.shortlist);
        return null;
    };
    defer for (cands) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const pick = try draft_fuse.pickRerank(s, cands[0], cands[1], x, rescore, front, .int32);
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{rc.bits});
        logged.* = true;
    }
    return pick;
}


pub const sidecar_rel_paths = mtp_source.sidecar_rel_paths;
pub const MtpSource = mtp_source.MtpSource;
pub const dirAdvertisesMtp = mtp_source.dirAdvertisesMtp;
pub const resolveMtpSidecarInDir = mtp_source.resolveMtpSidecarInDir;
const checkpoint_header_limit = mtp_source.checkpoint_header_limit;
const mtpShardsFromIndexJson = mtp_source.mtpShardsFromIndexJson;
const readDirFileAlloc = mtp_source.readDirFileAlloc;
pub const resolveMtpSource = mtp_source.resolveMtpSource;
pub const hasMtpHead = mtp_source.hasMtpHead;

/// Negative-fraction bar that separates a delta-encoded gamma (~30-50%
/// negative) from a folded one (positive by construction).
pub const NORM_DELTA_NEG_FRACTION: f32 = 0.05;

/// oMLX's `norm_repair` margin: a head RMSNorm whose mean sits more than this
/// below its backbone anchor is missing the `+1` zero-centered-gamma shift.
/// Mirrors `_REPAIR_MARGIN` in oMLX `patches/mlx_lm_mtp/norm_repair.py`.
pub const MTP_NORM_REPAIR_MARGIN: f32 = 0.4;

/// Head-trunk requantization width: dense
/// bf16 trunk weights (q/k/v/o, gate/up/down) are affine-quantized to this
/// width at group 64 during load — the head only PROPOSES tokens (verify
/// corrects), so the cost is acceptance, never output. 0 disables. Idea from
/// Layr-Labs/qwen-3.8-mtp-challenge @ deb63ad (see NOTICE). Test seam:
/// `head_quant_override`.
pub const DEFAULT_HEAD_QUANT_BITS: u32 = 4;
pub const HEAD_QUANT_GROUP: u32 = 64;
pub var head_quant_override: ?u32 = null;

/// Per-load accounting for the head-trunk requant log line.
const HeadQuantStats = struct {
    n: u32 = 0,
    before_bytes: u64 = 0,
    after_bytes: u64 = 0,
};

/// Requantize a row-quantized affine weight `(w, scales, biases)` from
/// `(from_gs, from_bits)` to `(to_gs, to_bits)`, chunk-wise over rows so the
/// dequantized bf16 transient stays bounded (~chunk_rows × in_features × 2 B
/// instead of the whole matrix — a 248K×5120 lm_head would otherwise
/// materialize 2.5 GB). Rows quantize independently in MLX's affine packing,
/// so per-chunk triples concatenate along axis 0 into a valid whole.
///
/// A DENSE source (`scales.ctx == null`, e.g. a bf16 lm_head) skips the
/// dequantize and quantizes the row chunk directly — same chunking, one
/// requantizer.
pub fn requantizeRows(
    s: mlx.mlx_stream,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    from_gs: u32,
    from_bits: u32,
    from_mode: [*:0]const u8,
    to_gs: u32,
    to_bits: u32,
    chunk_rows: c_int,
) !QLinear {
    const w_shape = mlx.getShape(w);
    if (w_shape.len != 2) return error.UnsupportedDraftHeadShape;
    const rows: c_int = w_shape[0];

    const wv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(wv);
    const sv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(sv);
    const bv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(bv);

    var r0: c_int = 0;
    while (r0 < rows) : (r0 += chunk_rows) {
        const r1: c_int = @min(rows, r0 + chunk_rows);

        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        if (scales.ctx == null) {
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try sliceRows(&raw, w, r0, r1, s);
            try mlx.check(mlx.mlx_astype(&dense, raw, .bfloat16, s));
        } else {
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sq);
            var bq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bq);
            try sliceRows(&wq, w, r0, r1, s);
            try sliceRows(&sq, scales, r0, r1, s);
            if (biases.ctx != null) try sliceRows(&bq, biases, r0, r1, s);
            try mlx.check(mlx.mlx_dequantize(
                &dense,
                wq,
                sq,
                bq,
                mlx.mlx_optional_int.some(@intCast(from_gs)),
                mlx.mlx_optional_int.some(@intCast(from_bits)),
                from_mode,
                .{}, // global_scale
                .{ .value = .bfloat16, .has_value = true },
                s,
            ));
        }

        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            dense,
            mlx.mlx_optional_int.some(@intCast(to_gs)),
            mlx.mlx_optional_int.some(@intCast(to_bits)),
            "affine",
            .{}, // global_scale
            s,
        ));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var part = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_vector_array_get(&part[0], triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&part[1], triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&part[2], triple, 2));
        // Realize the chunk so its dense transient can be reclaimed before
        // the next chunk builds (lazy eval would otherwise stack them all).
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for (part) |p| _ = mlx.mlx_vector_array_append_value(ev, p);
            try mlx.check(mlx.mlx_eval(ev));
        }
        _ = mlx.mlx_vector_array_append_value(wv, part[0]);
        _ = mlx.mlx_vector_array_append_value(sv, part[1]);
        _ = mlx.mlx_vector_array_append_value(bv, part[2]);
        for (part) |p| _ = mlx.mlx_array_free(p);
    }

    var out = QLinear{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_concatenate_axis(&out.w, wv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.s, sv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.b, bv, 0, s));
    return out;
}

fn sliceRows(out: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    const start = [_]c_int{ r0, 0 };
    const stop = [_]c_int{ r1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(out, src, &start, 2, &stop, 2, &strides, 2, s));
}

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

/// The head is the layer after the trunk's last; a NoPE arch (Nemotron-H)
/// marks it like the trunk layers.
inline fn headSkipsRope(cfg: anytype) bool {
    return cfg.layerSkipsRope(cfg.num_hidden_layers);
}

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// The trunk lm_head's TRUE quant params, via the same dtype-gated resolver
/// the trunk itself uses: uint8 scales → the config's non-affine mode (a raw
/// geometry solve mis-reads mxfp8 8-bit/gs32 as AFFINE 8-bit/gs32 — issue
/// #81, "Biases must be provided" crash on biasless heads); float scales →
/// exact per-geometry affine solve (the head's bits routinely differ from
/// the trunk global on mixed checkpoints — hy_v3 ships an 8-bit head over a
/// 2-bit trunk).
pub fn headQuantParams(config: anytype, w: mlx.mlx_array, sc: mlx.mlx_array) params_mod.QuantParams {
    return params_mod.computeQuantParams(.of(config), w, sc, config.hidden_size);
}

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

pub const StepOut = struct {
    /// `[1, 1, vocab]` LAST-row logits, or `.ctx == null` when the caller did
    /// not ask for `.logits` (multi-row calls never project the history rows —
    /// only the last row feeds the draft chain).
    logits: mlx.mlx_array,
    /// MTP post-norm hidden — the next depth's `hidden` input. `[1, 1, H]`
    /// (last row) under `.logits`, the full `[1, L, H]` otherwise.
    hidden_next: mlx.mlx_array,
    /// The vector the TARGET's lm_head consumes, for `.mixed` steps. Null on
    /// every arm where that is just `hidden_next` (the sidecar); non-null on
    /// qwen4_exp, whose `hidden_next` is the PRE-mixer stream and whose
    /// lm_head input is the mixer output. Owned by the caller when non-null.
    rerank_x: mlx.mlx_array = .{ .ctx = null },
};

/// What a head step owes its caller. `want_logits: bool` could not tell the
/// two zero-logits cases apart: a history append wants nothing past the
/// stream, while a rerank draft still needs the vector the lm_head consumes.
pub const StepWant = enum {
    /// Last-row logits (and the last-row hidden).
    logits,
    /// No logits, but the lm_head's INPUT vector for the last row — the
    /// rerank draft path.
    mixed,
    /// Nothing past the stream: committed-history appends.
    none,
};

pub const MropeContext = mrope.PositionContext;

/// MTP forward with an optional mapping from the head's cache-relative offsets
/// to the target's M-RoPE positions. Image-prompt positions use the explicit
/// three-axis table. Once decoding moves past that table, generated text uses
/// the scalar `absolute + delta` fast-RoPE path.
pub var mtp_kv_only_override: ?bool = null; // test seam

pub const Lane = struct {
    cache: *KVCache,
    /// One int32 id, may be lazy.
    id: mlx.mlx_array,
    /// `[1, 1, H]`.
    hidden: mlx.mlx_array,
    rope_offset: c_int,
};

pub const MAX_LANES = 32;

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
const TOP32_ORDINAL_HEADER = kernelSource("kernels/spec/mtp_top32_ordinal_header.metal");

// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32
// as (ordinal, index) pairs — TILES * 32 candidates. Selection is 32 rounds
// of simd_max over per-thread slots with a `taken` bitmask (hence the
// PER_THREAD <= 32 static_assert), then one simdgroup reduces the
// threadgroup's per-simd lists the same way.
const TOP32_PARTIAL_SRC = kernelSource("kernels/spec/mtp_top32_partial.metal");

// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32
// ids (written ascending by (value, index) — consumers are order-blind).
const TOP32_FINAL_SRC = kernelSource("kernels/spec/mtp_top32_final.metal");

const TOP32_K: c_int = 32;
const TOP32_TG: c_int = 256;
const TOP32_TILES: c_int = 64;
pub const TOP32_CANDS: c_int = TOP32_TILES * TOP32_K;
/// Minimum row count: below TILES*TG some simdgroups hold fewer than 32 real
/// candidates and the zero-initialized empty slots become reachable.
pub const TOP32_MIN_ROWS: c_int = TOP32_TILES * TOP32_TG;

var top32_partial_cached: ?mlx.mlx_fast_metal_kernel = null;
var top32_final_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getTop32Partial() !mlx.mlx_fast_metal_kernel {
    if (top32_partial_cached) |k| return k;
    const input_names = [_][*:0]const u8{"logits"};
    const output_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_partial",
        in_vec,
        out_vec,
        TOP32_PARTIAL_SRC,
        TOP32_ORDINAL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_partial_cached = kernel;
    return kernel;
}

fn getTop32Final() !mlx.mlx_fast_metal_kernel {
    if (top32_final_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const output_names = [_][*:0]const u8{"token_ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_finalize",
        in_vec,
        out_vec,
        TOP32_FINAL_SRC,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_final_cached = kernel;
    return kernel;
}

/// `draftTop32` through MLX's argpartition (CUDA, where the Metal kernels do not run): the same
/// set of ids, in no promised order or tie break. Proposal side only, so neither changes output.
fn draftTop32Partition(s: mlx.mlx_stream, row: mlx.mlx_array) !mlx.mlx_array {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, row, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg, TOP32_K - 1, 0, s));
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, part, &[_]c_int{0}, 1, &[_]c_int{TOP32_K}, 1, &[_]c_int{1}, 1, s));
    var ids = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, head, .uint32, s));
    return ids;
}

/// Exact top-32 ids of `row` ([rows], any float dtype) as a lazy [32] uint32
/// array, ascending by (value, index) — ties break toward the higher index,
/// NaN above every number. `rows` rides as a template int (stable per model,
/// so MLX caches one specialization per vocab width).
pub fn draftTop32(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) {
        if (mlx.cudaAvailable()) return draftTop32Partition(s, row);
        return error.MetalKernelNeedsGpuStream;
    }
    if (rows < TOP32_MIN_ROWS) return error.UnsupportedTop32Shape;
    const per_thread = @divTrunc(rows + TOP32_TILES * TOP32_TG - 1, TOP32_TILES * TOP32_TG);
    if (per_thread > 32) return error.UnsupportedTop32Shape;

    const pk = try getTop32Partial();
    var cand_ord = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ord);
    var cand_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_idx);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const cand_shape = [_]c_int{TOP32_CANDS};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TILES * TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RC", rows));
        const inputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(inputs);
        _ = mlx.mlx_vector_array_append_value(inputs, row);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, pk, inputs, cfg, s));
        try mlx.check(mlx.mlx_vector_array_get(&cand_ord, outputs, 0));
        try mlx.check(mlx.mlx_vector_array_get(&cand_idx, outputs, 1));
    }

    const fk = try getTop32Final();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{TOP32_K};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_ord);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_idx);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, fk, inputs, cfg, s));
    var token_ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&token_ids, outputs, 0));
    return token_ids;
}

/// Whether `draftTop32`'s Metal kernels take a `rows`-wide row.
pub fn top32RowsOk(rows: c_int) bool {
    if (rows < TOP32_MIN_ROWS) return false;
    const per_thread = @divTrunc(rows + TOP32_TILES * TOP32_TG - 1, TOP32_TILES * TOP32_TG);
    return per_thread <= 32;
}

/// `draftTop32`'s first launch: the TOP32_CANDS (ordinal, index) candidates, `[TOP32_CANDS]` uint32 each, that its
/// finalize reduces (the fused draft pick runs that finalize inside its own launch). Metal GPU streams; the caller
/// frees both.
pub fn draftTop32Cands(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) ![2]mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (!top32RowsOk(rows)) return error.UnsupportedTop32Shape;
    const pk = try getTop32Partial();
    var cand_ord = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cand_ord);
    var cand_idx = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cand_idx);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const cand_shape = [_]c_int{TOP32_CANDS};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TILES * TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RC", rows));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, row);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, pk, inputs, cfg, s));
    try mlx.check(mlx.mlx_vector_array_get(&cand_ord, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&cand_idx, outputs, 1));
    return .{ cand_ord, cand_idx };
}

/// `draftTop32`'s second launch: the 32 ids of the first's candidates, `[32]` uint32 (lazy).
pub fn draftTop32Final(s: mlx.mlx_stream, cand_ord: mlx.mlx_array, cand_idx: mlx.mlx_array) !mlx.mlx_array {
    const fk = try getTop32Final();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{TOP32_K};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_ord);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_idx);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, fk, inputs, cfg, s));
    var token_ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&token_ids, outputs, 0));
    return token_ids;
}

// ── Tests ──

const testing = std.testing;

test {
    _ = draft_fuse;
}

/// Builds a head's weights in memory (evaluated, so a leaked handle pins a
/// live buffer) with tensor `skip` left out; `skip == cases.len` keeps all.
const HeadLoadLeakProbe = struct {
    const Case = struct { key: []const u8, shape: []const c_int, dt: mlx.mlx_dtype = .bfloat16 };

};

// Test-only: run the negFraction threshold on a single 1-D norm array.

// Test-only: fold only when the single array reads as delta-encoded.

// Test-only: max |a-b| over all elements.
