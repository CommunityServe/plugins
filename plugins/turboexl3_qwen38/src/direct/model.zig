//! The direct Qwen3.8 model for the K packs (EXL3 K2–K5) on M5-class GPUs, text only: the trunk's forward over the
//! loaded `Trunk` (weights, caches, `ForwardCtx`), launching what the copied forward launches, in its order, with
//! turboexl3 doing the EXL3 math. `forward` is prefill, decode, a solo verify and the batched tick; the MTP head's step
//! is `mtp_head.forwardLive`. The grouped verify of several slots is gone: on the plugin only the load's warm-up ran it,
//! as the drafter verifies one slot at a time. The layer kinds sit beside it: hc.zig (the hyper-connections), gdn.zig,
//! attn.zig (QSA: projections, RoPE, indexer) with attn_core.zig (the KV append, the sparse core, SDPA, the gate and
//! o_proj), moe.zig (routed and shared experts), ple.zig and mtp_head.zig. wave.zig runs a wide prefill chunk as two
//! halves on two streams; load.zig, config.zig and cache.zig load, parse and hold each slot's state, and serve.zig is
//! the glue the plugin's table calls. The MTP round policy is the copies' under ../mlx_serve.

const mlx = @import("../mlx_serve/mlx/ffi.zig");
const eval_cadence = @import("../mlx_serve/engines/mlx/eval_cadence.zig");
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const MoeLayerWeights = @import("../mlx_serve/engines/mlx/layer_weights.zig").MoeLayerWeights;
const MoeMlpWeights = @import("../mlx_serve/sdk/blocks/moe/weights.zig").MoeMlpWeights;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const hc = @import("hc.zig");
const gdn = @import("gdn.zig");
const attn = @import("attn.zig");
const moe = @import("moe.zig");
const ple = @import("ple.zig");
const wave = @import("wave.zig");
const linear = @import("linear.zig");

/// The logits of `token_ids` `[B, S]` (or the mixer's output when `ctx.skip_lm_head`): a prefill chunk, a decode
/// step, a solo verify, or every slot's row of a batched tick (`ctx.batch_slots`).
pub fn forward(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    try plain(self, ctx);
    // A wide prefill chunk runs as the wavefront's two halves.
    if (wave.planFor(self, ctx, token_ids)) |halves| return wave.forward(self, ctx, token_ids, halves);
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const ml = self.moe_layers;
    const cfg = &self.config;
    const n_layers: usize = cfg.num_hidden_layers;
    const offset = ctx.moe_seq_offset.*;
    const entries = ctx.ssm_entries orelse return error.MissingSsmEntries;

    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    var h = try streams(self, token_ids);
    errdefer _ = mlx.mlx_array_free(h);
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    const cadence = eval_cadence.prefillEvalCadenceFor(
        eval_cadence.PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );
    var pending: ?hc.HcPending = null;
    defer if (pending) |*pd| pd.deinit();

    const ladder = if (ctx.batch_slots != null) self.decode_async_ladder_qwen4 else self.decode_async_ladder;
    var ladder_blocked = false;
    // A spec verify (2..16 rows) goes to the GPU every 4th layer while the rest is built, its first layer at once:
    // otherwise the GPU idles through the build, the PLE flush and eval's walk of the graph.
    const vladder: usize = if (self.formats.profile and ctx.spec_verify and batch == 1 and seq_len > 1 and seq_len <= 16) 4 else 0;
    for (0..n_layers) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];
        const entry = &entries[layer_idx];

        if (lw.ple) |*pw| {
            try hc.flush(self, &h, batch, seq_len, &pending);
            // A prefill chunk's host gather runs with the GPU idle: the layers built so far start once its ids are read.
            const kick = is_prefill and eval_cadence.prefillEvalCadenceApplies(seq_len) and self.qwen4 != null and ple.prefetchKick(&self.qwen4.?.table);
            if (kick) self.ple_kick = h;
            defer self.ple_kick = .{ .ctx = null };
            const h_ple = try ple.forwardResidual(self, ctx, h, token_ids, pw, entry, layer_idx, batch, seq_len);
            _ = mlx.mlx_array_free(h);
            h = h_ple;
        }

        var pre = try hc.read(self, &h, &lw.hc_attn.?, batch, seq_len, &pending);
        defer pre.deinit();
        const attn_out = switch (lw.attn) {
            .linear => |la| try gdn.layer(self, pre.mixed, &la, entry, layer_idx, batch, seq_len, is_prefill),
            .full => |fa| try attn.layer(self, ctx, pre.mixed, &fa, entry, li, @intCast(offset), 0, batch, seq_len, is_prefill),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try hc.writeOrDefer(self, &h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);

        var pre2 = try hc.read(self, &h, &lw.hc_mlp.?, batch, seq_len, &pending);
        defer pre2.deinit();
        const mlp_out = try moe.mlp(self, pre2.mixed, try moeOf(lw));
        defer _ = mlx.mlx_array_free(mlp_out);
        const next_inject: mlx.mlx_array = if (layer_idx + 1 < n_layers) ml[layer_idx + 1].hc_attn.?.inject_flat else .{ .ctx = null };
        try hc.writeOrDefer(self, &h, mlp_out, pre2.inj, batch, seq_len, next_inject, &pending);

        if (ladder != 0 and seq_len == 1 and (layer_idx + 1) % ladder == 0 and !ladder_blocked) {
            // A host-filled PLE leaf reads as zeros until it is flushed, so fill it before the first early eval;
            // while its ids are still a lazy sample, skip the ladder for this forward rather than sync.
            if (ctx.ple_pending) |pp| if (!pp.gpu) {
                var ids_avail = false;
                if (mlx._mlx_array_is_available(&ids_avail, pp.token_ids) == 0 and ids_avail) try ple.flushDeferred(self, ctx) else ladder_blocked = true;
            };
            // `mlp_out` rides along: the hc write that folds it into `h` is deferred to the next read.
            if (!ladder_blocked) eval_cadence.ladderStepMulti(&.{ h, mlp_out }, ladder, layer_idx, seq_len);
        }
        if (vladder != 0 and layer_idx + 1 < n_layers and (layer_idx == 0 or (layer_idx + 1) % vladder == 0)) {
            // past the PLE layer its host leaf is filled first, so the draft ids sync here instead of after the build
            if (layer_idx > 0) if (ctx.ple_pending) |pp| if (!pp.gpu) try ple.flushDeferred(self, ctx);
            const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ h, mlp_out }, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
        }

        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) {
                    try hc.flush(self, &h, batch, seq_len, &pending);
                    _ = mlx.mlx_array_set(slot, h);
                }
            }
        }
        if (is_prefill and eval_cadence.prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % cadence == 0 or layer_idx + 1 == n_layers)) {
            try eval_cadence.evalCadencePoint(h, ctx.ssm_entries);
        }
    }

    if (ctx.capture_stream_all != null or ctx.capture_hidden != null or ctx.capture_hidden_all != null) try hc.flush(self, &h, batch, seq_len, &pending);
    ctx.moe_seq_offset.* += @intCast(seq_len);
    if (ctx.capture_stream_all) |target| _ = mlx.mlx_array_set(target, h);
    // The spec "hidden" is the pre-mixer stream: the MTP head reads `[B, L, hc*hidden]`, never the mixed width.
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(h);
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, h, &[_]c_int{ 0, hs[1] - 1, 0 }, 3, &[_]c_int{ hs[0], hs[1], hs[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, h);

    const mix = try hc.read(self, &h, &self.qwen4_mixer, batch, seq_len, &pending);
    // The function-scope errdefer reads `h` at unwind time, so surrender the handle where it is released.
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) return final;
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try linear.lmHeadProject(self, final);
    _ = mlx.mlx_array_free(final);
    return logits;
}

/// The model serves text on one rank: no image rows, no M-RoPE table, no stream handed in, no peer.
fn plain(self: *const Trunk, ctx: *const ForwardCtx) !void {
    if (ctx.vision_embeddings != null or ctx.mrope_pos != null) return error.DirectModelIsTextOnly;
    if (ctx.stream_in != null or self.dist_peer != null or self.dist_tp != null) return error.DirectModelIsOneRank;
}

/// `token_ids`' embeddings tiled into the hyper-connection streams, `[B, S, hc*hidden]`.
fn streams(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const emb = try linear.embedding(self, token_ids);
    defer _ = mlx.mlx_array_free(emb);
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_tile(&h, emb, &[_]c_int{ 1, 1, @intCast(self.config.hc_count) }, 3, self.s));
    return h;
}

/// The layer's routed MLP; the K packs have no dense one.
fn moeOf(lw: *const MoeLayerWeights) !*const MoeMlpWeights {
    return switch (lw.mlp) {
        .moe => |*mw| mw,
        .dense => error.DirectModelHasNoDenseMlp,
    };
}
