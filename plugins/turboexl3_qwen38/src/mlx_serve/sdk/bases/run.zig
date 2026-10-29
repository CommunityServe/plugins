//! copied_from: src/sdk/bases/run.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! What the bases share around their layer loops: one forward's arguments (`Run`), its results (`Out`), the head and
//! the logits rows, layer capture, and the model table's `forward` over any base (`tableForward`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("mlx_serve").abi.plugin;
const block = @import("block.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const spliceRows = @import("../vision/splice.zig").spliceRows;
const row_exact = @import("blocks/row_exact.zig");
const prefillHeadDimFused = @import("../blocks/attention/attn256.zig").prefillHeadDimFused;

/// One forward's arguments.
pub const Run = struct {
    mode: abi.Mode,
    /// Tokens per slot.
    rows: c_int,
    slots: c_int = 1,
    /// Each slot's first position.
    positions: []const u32,
    caches: []const abi.Handle = &.{},
    cache_ops: ?*const abi.CacheOps = null,
    logits: abi.Logits = .last_row,
    /// Return the final-normed hidden rows too.
    want_hidden: bool = false,
    /// Layers whose output rows the forward publishes into `captured` (owned by the forward's graph).
    capture_layers: []const u32 = &.{},
    captured: []Array = &.{},
    /// An encoder batch's additive key mask; null = none.
    mask: Array = .{},
    /// `Ctx.mask_add` (`Forward.mask_add`).
    mask_add: Array = .{},
    /// Evaluate the residual after every this many layers and after the last (0 = never; `Forward.eval_every`).
    eval_every: u32 = 0,
    /// `Ctx.capture_spec`.
    capture_spec: bool = false,
    /// `Ctx.row_exact`.
    row_exact: row_exact.Lane = .off,
    /// `Ctx.mrope`: the chunk's M-RoPE cos/sin (`Forward.mrope`).
    mrope: ?[2]Array = null,
    /// `Ctx.tree` (`Forward.tree`).
    tree: ?block.Tree = null,
    /// Hand the residual to the GPU every this many layers (`Forward.pipeline_every`; 0 = never).
    pipeline_every: u32 = 0,
    /// Media rows spliced after the embedding at the placeholder ids, from row `media_offset` (`Forward.media`).
    media: Array = .{},
    media_offset: u32 = 0,
    placeholders: abi.PlaceholderIds = .{},
    /// `Forward.embeds`.
    embeds: Array = .{},
    /// `Ctx.offload` (`Forward.offload`).
    offload: ?*const abi.Offload = null,

};

/// A forward's results, owned by its graph: logits `[slots, vocab]` (last row) or `[slots * rows, vocab]`, and the
/// final-normed hidden `[slots, rows, hidden]` when asked.
pub const Out = struct {
    logits: Array = .{},
    hidden: Array = .{},
};

/// A loaded model of base `Base`: its weights, the shared values, and where its forwards run.
pub fn Model(comptime Base: type) type {
    return struct {
        weights: Base.Weights,
        shared: block.Shared,
        a: std.mem.Allocator,
        s: ops.Stream,
        /// The placeholder ids media rows splice at (`Parsed.placeholder_ids`).
        placeholders: abi.PlaceholderIds = .{},
        /// The lane `Forward.row_exact` turns on; `off` = the weights are outside the row kernels.
        row_exact: row_exact.Lane = .off,

        pub fn of(m: *anyopaque) *@This() {
            return @ptrCast(@alignCast(m));
        }
    };
}

/// Forwards narrower than this skip the mid-loop eval cadence entirely.
/// Spec-decode VERIFY forwards (PLD/drafter/MTP: seq 2..~9) ride the
/// prefill layer loops because they're multi-token, but their lazy-graph
/// transients are decode-scale (KB-MB) — the periodic eval() that bounds
/// GB-scale prefill-chunk transients only costs them synchronous
/// pipeline drains. Measured (qwen3.6-27B, 64 layers, cadence 4 = 16
/// drains per MTP round): depth-1 rounds 48 ms where the AR forward is
/// ~34 ms, and the drain tax grows superlinearly with draft depth.
/// seq-1 decode never ran cadence evals, so exempting verify-width
/// forwards bounds nothing that wasn't already unbounded at decode.
pub const PREFILL_EVAL_MIN_SEQ: c_int = 32;

pub const RECURRENCE_EVAL_INTERVAL: usize = 32;

pub const EVAL_EVERY_N_LAYERS: u32 = 48;
/// One value for the prefill loops AND the admission bill: at most `cadence + 1`
/// linear layers hold a chunk-wide stream between cadence points.
pub const PREFILL_EVAL_CADENCE_DEFAULT: u32 = 4;

/// Per-layer prefill transient (bytes) above which the layer loop eval()s
/// after EVERY layer instead of the coarse default cadence. Between eval
/// points ~3 layers' transients coexist under MLX lazy eval, so a
/// multi-GB transient triples; eval-per-layer bounds peak to ~1 layer.
/// Two transients count (both scale with total_kv, so short prompts keep
/// the coarse cadence and pay nothing):
///   - the composed-SDPA score tensor [heads, chunk, total_kv] — only for
///     head_dims no fused kernel covers (prefillHeadDimFused: <= 128 via
///     MLX, 256 via msv_attn_p256; unfused only via the kill switch or an
///     exotic dim);
///   - the dense-fp16 rebuild of a quantized KV cache (denseView runs
///     per layer under --kv-quant, over the FULL cache).
/// Budget calibrated LIVE on gemma-4-26B-A4B-qat-4bit (M4 Max):
/// eval-per-layer costs ~4.5% prefill at a 5K prompt (845 MB scores —
/// below this budget, coarse cadence kept) and ~0% at 102K (3.4 GB
/// scores — flips to per-layer, peak 51.6 -> 27.0 GB, prefill +14%).
/// 2 GiB puts the flip at ~8K tokens on 16-head/256-hd geometry, where
/// the ~3% sync cost buys a 13.7 -> 4.3 GB transient bound — the margin
/// that keeps a 32 GB Mac serving a 26B alive.
pub const PREFILL_EVAL_TRANSIENT_BUDGET: u64 = 2 << 30;

/// Pure cadence pick for the prefill layer loops (standard/MoE/hybrid).
pub fn prefillEvalCadence(
    default_cadence: u32,
    score_head_dim: u32,
    head_dim: u32,
    n_heads: u32,
    kv_heads: u32,
    chunk_len: u64,
    total_kv: u64,
    kv_dequant: bool,
) u32 {
    const scores: u64 = if (score_head_dim > 0 and !prefillHeadDimFused(score_head_dim)) @as(u64, n_heads) * chunk_len * total_kv * 2 else 0;
    const dequant: u64 = if (kv_dequant) 2 * total_kv * @as(u64, kv_heads) * @as(u64, head_dim) * 2 else 0;
    return if (scores + dequant > PREFILL_EVAL_TRANSIENT_BUDGET) 1 else default_cadence;
}

/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

/// Slots one batched forward carries (twin of `scheduler.MAX_BATCH_GROUP`).
pub const MAX_BATCH_ROWS = 32;
