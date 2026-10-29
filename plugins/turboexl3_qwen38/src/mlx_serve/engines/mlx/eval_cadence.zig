//! copied_from: src/engines/mlx/eval_cadence.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/eval_cadence.
//! Owns EVAL_EVERY_N_LAYERS, PREFILL_EVAL_CADENCE_DEFAULT, MOE_EVAL_EVERY_N_LAYERS, RECURRENCE_EVAL_INTERVAL, PREFILL_EVAL_TRANSIENT_BUDGET, PREFILL_EVAL_MIN_SEQ, and 8 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ModelConfig = @import("../../sdk/config/model.zig").ModelConfig;
const checkpoint = @import("ssm/checkpoint.zig");
const gather_qmv = @import("../../sdk/blocks/moe/gather_qmv.zig");
const transformer = @import("transformer.zig");
const prefillHeadDimFused = @import("../../sdk/blocks/attention/attn256.zig").prefillHeadDimFused;
const testing = @import("kv/cache.zig").testing;
const SSMCacheEntry = @import("ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("transformer.zig").Transformer;
const attn256_vars = @import("../../sdk/blocks/attention/attn256.zig");
const run = @import("../../sdk/bases/run.zig");

// ── Forward dispatch ──

pub const EVAL_EVERY_N_LAYERS = run.EVAL_EVERY_N_LAYERS;
pub const PREFILL_EVAL_CADENCE_DEFAULT = run.PREFILL_EVAL_CADENCE_DEFAULT;
pub const MOE_EVAL_EVERY_N_LAYERS: u32 = PREFILL_EVAL_CADENCE_DEFAULT;
pub const RECURRENCE_EVAL_INTERVAL = @import("../../sdk/bases/run.zig").RECURRENCE_EVAL_INTERVAL;

pub const PREFILL_EVAL_TRANSIENT_BUDGET = run.PREFILL_EVAL_TRANSIENT_BUDGET;

pub const PREFILL_EVAL_MIN_SEQ = @import("../../sdk/bases/run.zig").PREFILL_EVAL_MIN_SEQ;

/// Pure gate: does the mid-loop eval cadence apply to a forward of this
/// width? (True = real prefill chunk; false = decode/spec-verify shape.)
pub fn prefillEvalCadenceApplies(seq_len: c_int) bool {
    return seq_len >= PREFILL_EVAL_MIN_SEQ;
}

/// The cadence eval must NAME every conv_state beside `h`: the owned tail copy is
/// not in the residual's graph, so `mlx_array_eval(h)` alone would leave it lazy
/// and still holding its chunk parent. Re-appending an evaluated array is free.
pub fn evalCadencePoint(h: mlx.mlx_array, ssm_entries: ?[]SSMCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, h);
    var extra: u64 = 0;
    if (ssm_entries) |entries| {
        for (entries) |*e| {
            if (e.initialized and e.conv_state.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, e.conv_state);
                extra += 1;
            }
        }
    }
    if (extra == 0) {
        try mlx.check(mlx.mlx_array_eval(h));
        return;
    }
    checkpoint.cadence_conv_eval_count += extra;
    try mlx.check(mlx.mlx_eval(vec));
}

/// Decode-side intra-step async-eval ladder.
///
/// A decode step builds ~40 layers of graph on the CPU before anything is
/// evaluated, so the GPU sits idle through the build and the CPU sits idle
/// through the execution. `mlx_async_eval` at a layer boundary hands the
/// prefix of the graph to the GPU WITHOUT blocking, so the two overlap
/// inside one token. It adds no op and changes no value — the same graph is
/// evaluated in the same order, just started earlier — so output is
/// bit-identical with the ladder on or off.
///
/// This is distinct from the BLOCKING prefill cadence above, which exists
/// to bound transient memory on wide forwards; the ladder is additive and
/// never replaces it.
///
/// DEFAULT OFF — measured, not assumed. The mlxfast-challenge tree gets
/// +9.7% from exactly this (their `DARKBLOOM_DECODE_ASYNC_STAGE`), but
/// their worker cannot pipeline BETWEEN steps: its protocol is one token
/// per request, so the ladder is the only overlap it can buy. mlx-serve
/// already overlaps at the step boundary (`generate.zig`'s build-next →
/// async_eval → resolve-pending), so the overlap is already collected and
/// the ladder only adds submissions. Swept on Laguna XS 2.1 NVFP4 / M4 Max,
/// 512-token prompt / 128 decode steps, medians of 2:
///   off 39.88 | stride 8 40.22 | 4 40.37 | 2 41.08 | 1 41.49 ms/token
/// Monotonically worse as the stride tightens, so it ships off (`Transformer.decode_async_ladder` 0) except on
/// qwen4_exp batched decode (`QWEN4_DECODE_ASYNC_LADDER_DEFAULT`). Only decode-width forwards ladder: a spec-verify
/// or prefill forward already has enough GPU work per layer to keep the device busy.

// A PREFILL-side async-eval ladder (mlxfast-challenge prefillLadder:
// asyncEval every 3rd-4th layer at seq >= 512) was ported and measured
// (qwen3.8-27B-4bit, 43.7k-token prompt, M4 Max): off 227.9
// vs ladder 225.9 tok/s median — null-to-negative, same reason the
// decode ladder loses here: chunked prefill + the eval cadence already
// collect the overlap. Removed rather than shipped as a dead knob.

/// Fire the ladder at this layer boundary, if it is one. `h` is the live
/// residual; handing it to `mlx_async_eval` starts every op it depends on.
pub fn ladderStep(self: *Transformer, h: mlx.mlx_array, layer_idx: usize, seq_len: c_int) void {
    ladderStepMulti(&.{h}, self.decode_async_ladder, layer_idx, seq_len);
}

pub fn ladderStepMulti(arrs: []const mlx.mlx_array, stride: u32, layer_idx: usize, seq_len: c_int) void {
    if (seq_len != 1) return;
    if (stride == 0) return;
    if ((layer_idx + 1) % stride != 0) return;
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    // Best-effort: a failed async eval is a missed overlap, never a wrong
    // answer — the value is computed by the step's own terminal eval.
    _ = mlx.mlx_async_eval(vec);
    if (!gather_qmv.decode_ladder_engaged) {
        gather_qmv.decode_ladder_engaged = true;
        log.info("[decode] async-eval ladder engaged: stride={d}\n", .{stride});
    }
}

pub const prefillEvalCadence = run.prefillEvalCadence;

pub fn prefillEvalCadenceFor(default_cadence: u32, cfg: *const ModelConfig, chunk_len: u64, total_kv: u64, kv_dequant: bool) u32 {
    return prefillEvalCadence(default_cadence, cfg.prefillScoreHeadDim(), cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, chunk_len, total_kv, kv_dequant);
}

// ── tests ──
