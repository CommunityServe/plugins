//! copied_from: src/plugins/qwen4_exp/forward.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The qwen4_exp trunk forward (`forwardQwen4With`): embeddings tiled into hyper-connection streams, the n-gram PLE,
//! GDN and QSA attention layers under hc reads and writes, the final mixer; also the grouped verify rows and a
//! pipeline rank's stream forward. Env-gated fixture tests against the HF reference live here.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const gather_qmv = @import("../../sdk/blocks/moe/gather_qmv.zig");
const mtp = @import("mtp.zig");
const standin = @import("../../sdk/diag/standin.zig");
const verify_projection = @import("../../sdk/blocks/linear/verify_projection.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const layerCap = @import("../../engines/mlx/diag/profile.zig").layerCap;
const env = @import("../../env.zig");
const profileFwdOn = @import("prof.zig").profileFwdOn;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../../engines/mlx/eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../../engines/mlx/eval_cadence.zig").evalCadencePoint;
const ladderStepMulti = @import("../../engines/mlx/eval_cadence.zig").ladderStepMulti;
const prefillEvalCadenceApplies = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceFor;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmLiveAux = @import("../../engines/mlx/ssm/group.zig").ssmLiveAux;
const Trunk = @import("trunk.zig").Trunk;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const verifyLmHeadProjection = @import("../../sdk/blocks/linear/verify_projection.zig").verifyLmHeadProjection;
const moeRouterFusedEnabled = @import("../../sdk/blocks/moe/router_kernel.zig").moeRouterFusedEnabled;
const endMropeChunk = @import("../../engines/mlx/methods/mrope.zig").endMropeChunk;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const standinRef = @import("../../sdk/diag/standin.zig").standinRef;
const Qwen4TiedCmp = @import("../../sdk/testing/parity.zig").Qwen4TiedCmp;
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4CompareTied = @import("../../sdk/testing/parity.zig").qwen4CompareTied;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const HcRead = @import("hc.zig").HcRead;
const HcPending = @import("hc_fused.zig").HcPending;
const Qwen4Trace = @import("mtp.zig").Qwen4Trace;
const PleArmFixture = @import("ple.zig").PleArmFixture;
const expectPleHistoryEqual = @import("ple.zig").expectPleHistoryEqual;
const pleArmEntry = @import("ple.zig").pleArmEntry;
const pleArmIds = @import("ple.zig").pleArmIds;
const qwen4BatchedPleForward = @import("ple.zig").qwen4BatchedPleForward;
const pleForwardResidual = @import("ple.zig").pleForwardResidual;
const Qwen4AttnProf = @import("prof.zig").Qwen4AttnProf;
const Qwen4FwdProf = @import("prof.zig").Qwen4FwdProf;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;
const wave = @import("wave.zig");

/// qwen4_exp's batched decode builds each step's graph while the GPU idles, so its ladder ships ON at this stride.
/// Serial decode and other arches keep `decode_async_ladder`.
pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT: u32 = 4;

pub fn qwen4VerifyRowsEligible(self: *Trunk, token_rows: []const mlx.mlx_array, ctxs: []const *ForwardCtx) bool {
    // A tensor shard's MoE is half of each expert until `moeTensorParallel` adds the
    // peer's half; the grouped arm calls `moeVerifyRows` directly and would skip it.
    if (self.dist_tp != null) return false;
    if (self.qwen4 == null or !mlx.streamIsGpu(self.s) or !moeRouterFusedEnabled() or
        token_rows.len < 2 or token_rows.len > 8 or token_rows.len != ctxs.len or
        self.config.hidden_size < 32 or self.config.num_experts_per_tok >= 32)
    {
        return false;
    }
    if (Trunk.qwen4_trace != null or profileFwdOn()) return false;
    const si = qwen4Standin();
    if (si.gdn or si.attn or si.mlp or si.gdn_recur or si.gdn_proj or si.attn_qsa or si.attn_sdpa or si.hc or si.moe_shared or si.moe_router or si.moe_gateup or si.moe_down) return false;
    for (self.moe_layers) |lw| switch (lw.mlp) {
        .moe => |mw| if (mw.expert_bias != null or mw.router_scale != null or mw.per_expert_scale != null) return false,
        else => return false,
    };
    var total_tokens: u64 = 0;
    for (token_rows, ctxs) |tokens, ctx| {
        const shape = mlx.getShape(tokens);
        if (shape.len != 2 or shape[0] != 1 or shape[1] < 1 or ctx.ssm_entries == null or ctx.mrope_pos != null or ctx.vision_embeddings != null or
            ctx.capture_hidden != null or ctx.capture_hidden_all != null or ctx.capture_stream_all != null or ctx.capture_layers != null or ctx.batch_slots != null)
        {
            return false;
        }
        total_tokens += @intCast(shape[1]);
    }
    if (self.config.num_experts == 0 or self.config.num_experts_per_tok == 0) return false;
    return (total_tokens * self.config.num_experts_per_tok) / self.config.num_experts < 4;
}

pub fn forwardQwen4VerifyRows(
    self: *Trunk,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
) !void {
    const op_start = mlx.op_count.load(.monotonic);
    const cfg = &self.config;
    const layers = self.moe_layers;
    const hc: c_int = @intCast(cfg.hc_count);
    const Row = struct {
        ctx: *ForwardCtx,
        tokens: mlx.mlx_array,
        entries: []SSMCacheEntry,
        offset: usize,
        seq: c_int,
        generation: u64,
        h: mlx.mlx_array,
        pending: ?HcPending = null,
    };
    var rows: [MAX_BATCH_ROWS]Row = undefined;
    var initialized: usize = 0;
    defer for (rows[0..initialized]) |*row| {
        if (row.pending) |*pending| pending.deinit();
        if (row.h.ctx != null) _ = mlx.mlx_array_free(row.h);
    };
    var generation = self.fwd_gen;
    for (token_rows, ctxs, 0..) |tokens, ctx, i| {
        generation +%= 1;
        self.fwd_gen = generation;
        const seq = mlx.getShape(tokens)[1];
        const emb = try self.embedding(tokens);
        defer _ = mlx.mlx_array_free(emb);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&h, emb, &.{ 1, 1, hc }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(h);
        if (self.qwen4_stream_f32 or env.onCached(&qwen4_stream_f32_env, .QWEN4_STREAM_F32)) {
            var h32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
            _ = mlx.mlx_array_free(h);
            h = h32;
        }
        rows[i] = .{
            .ctx = ctx,
            .tokens = tokens,
            .entries = ctx.ssm_entries.?,
            .offset = ctx.moe_seq_offset.*,
            .seq = seq,
            .generation = generation,
            .h = h,
        };
        initialized += 1;
    }

    self.spec_capture_ssm = true;
    defer self.spec_capture_ssm = false;
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const layer = &layers[layer_idx];
        var group_projection = true;
        if (group_projection) {
            var total: c_int = 0;
            for (rows[0..initialized]) |row| {
                group_projection = group_projection and row.seq >= 2 and row.seq <= 6;
                total += row.seq;
            }
            group_projection = group_projection and total <= 12;
        }
        var attn_reads: [MAX_BATCH_ROWS]HcRead = undefined;
        var attn_count: usize = 0;
        defer for (attn_reads[0..attn_count]) |*read| read.deinit();
        var attn_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var attn_entries: [MAX_BATCH_ROWS]*SSMCacheEntry = undefined;
        var generations: [MAX_BATCH_ROWS]u64 = undefined;
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            if (layer.ple) |*pw| try self.addVerifyPle(row, pw, layer_idx, i);
        }
        const joined_attn = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_attn.?);
        defer if (joined_attn) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            var read = if (joined_attn) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_attn.?, 1, row.seq, &row.pending);
            if (group_projection) {
                attn_reads[i] = read;
                attn_count += 1;
                attn_inputs[i] = read.mixed;
                attn_entries[i] = &row.entries[layer_idx];
                generations[i] = row.generation;
            } else {
                defer read.deinit();
                const attn = switch (layer.attn) {
                    .linear => |la| try self.gatedDeltaNet(read.mixed, &la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |fa| try self.qwen4AttnWith(row.ctx, read.mixed, &fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }
        if (group_projection) {
            const projected = switch (layer.attn) {
                .linear => |*la| try self.gdnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], generations[0..initialized], layer_idx, la),
                .full => |*fa| try self.attnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], ctxs, generations[0..initialized], layer_idx, fa),
            };
            defer if (projected) |values| {
                for (values) |value| _ = mlx.mlx_array_free(value);
                self.allocator.free(values);
            };
            for (rows[0..initialized], attn_reads[0..initialized], 0..) |*row, read, i| {
                self.fwd_gen = row.generation;
                const attn = if (projected) |values| try standinRef(values[i]) else switch (layer.attn) {
                    .linear => |*la| try self.gatedDeltaNet(read.mixed, la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |*fa| try self.qwen4AttnWith(row.ctx, read.mixed, fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }

        var reads: [MAX_BATCH_ROWS]HcRead = undefined;
        var read_count: usize = 0;
        defer for (reads[0..read_count]) |*read| read.deinit();
        var inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        const joined_mlp = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_mlp.?);
        defer if (joined_mlp) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            reads[i] = if (joined_mlp) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_mlp.?, 1, row.seq, &row.pending);
            read_count += 1;
            inputs[i] = reads[i].mixed;
        }
        const mw = switch (layer.mlp) {
            .moe => |*weights| weights,
            else => unreachable,
        };
        const mlp = try self.moeVerifyRows(inputs[0..initialized], mw);
        defer {
            for (mlp) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(mlp);
        }
        for (rows[0..initialized], mlp, reads[0..initialized]) |*row, value, read| {
            self.fwd_gen = row.generation;
            try self.hcWriteOrDefer(&row.h, value, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
        }
    }

    var join_lmhead = true;
    var lmhead_widths: [MAX_BATCH_ROWS]c_int = undefined;
    if (join_lmhead) {
        var total: c_int = 0;
        for (rows[0..initialized], 0..) |row, i| {
            join_lmhead = join_lmhead and !row.ctx.skip_lm_head and row.seq >= 2 and row.seq <= 6;
            lmhead_widths[i] = row.seq;
            total += row.seq;
        }
        join_lmhead = join_lmhead and total <= 12;
    }
    var lmhead_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{});
    defer for (lmhead_inputs[0..initialized]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    var completed: usize = 0;
    errdefer for (0..completed) |i| {
        _ = mlx.mlx_array_free(out_logits[i]);
        _ = mlx.mlx_array_free(out_last[i]);
        _ = mlx.mlx_array_free(out_all[i]);
    };
    for (rows[0..initialized], 0..) |*row, i| {
        self.fwd_gen = row.generation;
        try self.hcFlush(&row.h, 1, row.seq, &row.pending);
        row.ctx.moe_seq_offset.* += @intCast(row.seq);
        var all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&all, row.h));
        errdefer _ = mlx.mlx_array_free(all);
        const shape = mlx.getShape(row.h);
        var last = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&last, row.h, &.{ 0, shape[1] - 1, 0 }, 3, &.{ 1, shape[1], shape[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(last);
        const mix = try self.hcReadPending(&row.h, &self.qwen4_mixer, 1, row.seq, &row.pending);
        _ = mlx.mlx_array_free(row.h);
        row.h = .{ .ctx = null };
        if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
        const logits: mlx.mlx_array = if (join_lmhead) blk: {
            lmhead_inputs[i] = mix.mixed;
            break :blk .{};
        } else if (row.ctx.skip_lm_head)
            mix.mixed
        else blk: {
            defer _ = mlx.mlx_array_free(mix.mixed);
            break :blk try self.lmHeadProject(mix.mixed);
        };
        out_logits[i] = logits;
        out_last[i] = last;
        out_all[i] = all;
        completed += 1;
    }
    if (join_lmhead) {
        const vec = mlx.mlx_vector_array_new_data(&lmhead_inputs, initialized);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
        if (try verifyLmHeadProjection(self.s, joined, self.lm_head_w, self.lm_head_s, self.lm_head_b, lmhead_widths[0..initialized])) |logits| {
            // The block stays published for the accept side; `out_logits`
            // are views of it either way.
            self.verify_joined_logits = logits;
            var offset: c_int = 0;
            for (rows[0..initialized], 0..) |row, i| {
                try mlx.check(mlx.mlx_slice(&out_logits[i], logits, &.{ 0, offset, 0 }, 3, &.{ 1, offset + row.seq, mlx.getShape(logits)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
                offset += row.seq;
            }
            verify_projection.mtp_verify_lmhead_rows_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] vocabulary verify projection engaged (slots={d})\n", .{initialized});
            }
        } else {
            for (0..initialized) |i| out_logits[i] = try self.lmHeadProject(lmhead_inputs[i]);
        }
    }
    self.fwd_gen = generation;
    const group_ops = mlx.op_count.load(.monotonic) - op_start;
    mtp.mtp_verify_moe_group_graphs +%= 1;
    mtp.mtp_verify_moe_group_last_ops = group_ops;
}

/// Qwen3.8-Flash-Next forward: embeddings tiled into `hc` residual
/// streams; every block reads a gated mix of the streams and writes back
/// through per-stream scalar gates; the n-gram PLE adds to the streams
/// before its layer; the final mixer replaces model.norm.
/// A later pipeline rank's forward: its layers over the `[B, L, hc*hidden]`
/// stream the previous rank handed over, returning the stream for the next one.
pub fn forwardQwen4Stream(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array) !mlx.mlx_array {
    ctx.stream_in = stream;
    defer ctx.stream_in = null;
    return self.forwardQwen4With(ctx, .{ .ctx = null });
}

pub fn forwardQwen4With(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    // A wide prefill chunk in reordered numerics runs as the wavefront's two halves (`wave.zig`).
    if (wave.pfWavePlanFor(self, ctx, token_ids)) |halves| return wave.forwardQwen4Wave(self, ctx, token_ids, halves);
    const stream_in = ctx.stream_in;
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const ml = self.moe_layers;
    const cfg = &self.config;
    const offset = ctx.moe_seq_offset.*;
    const entries = ctx.ssm_entries orelse return error.MissingSsmEntries;
    const hc: c_int = @intCast(cfg.hc_count);

    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    if (stream_in) |st| {
        try mlx.check(mlx.mlx_array_set(&h, st));
    } else {
        // Vision rows splice into the `hidden`-wide embeddings BEFORE the
        // hyper-connection tile (the reference masked_scatters, then repeats).
        var emb = try self.embedding(token_ids);
        {
            errdefer _ = mlx.mlx_array_free(emb);
            emb = try self.applyVisionEmbeddingsWith(ctx, emb, token_ids);
        }
        defer _ = mlx.mlx_array_free(emb);
        const reps = [_]c_int{ 1, 1, hc };
        try mlx.check(mlx.mlx_tile(&h, emb, &reps, 3, self.s));
    }
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    if (streamF32(self)) {
        var h32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
        _ = mlx.mlx_array_free(h);
        h = h32;
    }
    // M-RoPE chunk tables: read by every full-attn layer AND the QSA
    // indexer's queries (its pooled block keys take a strided build).
    try self.beginMropeChunk(ctx, @intCast(offset), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer endMropeChunk(ctx);
    var dt = mlx.DtypeTrace.begin("qwen4", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        .linear => |la| la.qkv_w,
    });

    const eval_cadence = prefillEvalCadenceFor(
        PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    var prof = try Qwen4FwdProf.init(seq_len, h);
    var pending: ?HcPending = null;
    defer if (pending) |*pd| pd.deinit();

    const ladder = if (ctx.batch_slots != null) self.decode_async_ladder_qwen4 else self.decode_async_ladder;
    var ladder_blocked = false;
    // A spec verify (2..16 rows) on the EXL3 pack goes to the GPU every 4th layer while the rest is built, its first
    // layer at once: otherwise the GPU idles through the build, the PLE flush and eval's walk of the graph.
    const vladder: usize = if (self.formats.profile and ctx.spec_verify and batch == 1 and seq_len > 1 and seq_len <= 16) 4 else 0;
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];
        const entry = &entries[layer_idx];

        if (lw.ple) |*pw| {
            try self.hcFlush(&h, batch, seq_len, &pending);
            // A prefill chunk's host gather runs with the GPU idle: the layers built so far start once its ids are read.
            const kick = is_prefill and prefillEvalCadenceApplies(seq_len) and self.qwen4 != null and self.qwen4.?.table.prefetchKick();
            if (kick) self.ple_kick = h;
            defer self.ple_kick = .{ .ctx = null };
            const h_ple = try pleForwardResidual(self, ctx, h, token_ids, pw, entry, layer_idx, batch, seq_len);
            _ = mlx.mlx_array_free(h);
            h = h_ple;
            try prof.lap(h, .ple);
        }

        const si = qwen4Standin();
        var pre: HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_attn.?, batch, seq_len, &pending);
        defer pre.deinit();
        try prof.lap(pre.mixed, .hc_read);
        const attn_out = switch (lw.attn) {
            .linear => |la| if (si.gdn) try standinRef(pre.mixed) else try self.gatedDeltaNet(pre.mixed, &la, entry, layer_idx, batch, seq_len, is_prefill),
            .full => |fa| if (si.attn) try standinRef(pre.mixed) else try self.qwen4AttnWith(ctx, pre.mixed, &fa, entry, li, @intCast(offset), 0, batch, seq_len, is_prefill),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try prof.lap(attn_out, if (lw.attn == .linear) .gdn else .attn);
        if (layer_idx == 3) if (Trunk.qwen4_trace) |tr| Qwen4Trace.set(&tr.attn3_out, attn_out);
        if (layer_idx == 0) if (Trunk.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_attn, pre.mixed);
            Qwen4Trace.set(&tr.inj_attn, pre.inj);
            Qwen4Trace.set(&tr.attn_out, attn_out);
        };
        if (si.hc) h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);

        var pre2: HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_mlp.?, batch, seq_len, &pending);
        defer pre2.deinit();
        try prof.lap(pre2.mixed, .hc_read);
        const mlp_out = if (si.mlp) try standinRef(pre2.mixed) else switch (lw.mlp) {
            .moe => |*mw| if (self.dist_tp) |tp| try self.moeTensorParallel(tp, pre2.mixed, mw, li) else try self.moeMLP(pre2.mixed, mw),
            .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
        };
        defer _ = mlx.mlx_array_free(mlp_out);
        try prof.lap(mlp_out, .mlp);
        if (layer_idx == 0) if (Trunk.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_mlp, pre2.mixed);
            Qwen4Trace.set(&tr.inj_mlp, pre2.inj);
            Qwen4Trace.set(&tr.mlp_out, mlp_out);
        };
        if (si.hc) h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, mlp_out, pre2.inj, batch, seq_len, if (layer_idx + 1 < layerCap(cfg.num_hidden_layers)) ml[layer_idx + 1].hc_attn.?.inject_flat else .{ .ctx = null }, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);
        prof.endLayer(if (lw.ple != null) .ple else if (lw.attn == .linear) .gdn else .attn);
        if (ladder != 0 and seq_len == 1 and (layer_idx + 1) % ladder == 0 and !ladder_blocked) {
            // A host-filled PLE leaf reads as zeros until `flushDeferredPle`, so fill it before the first early
            // eval; while its ids are still a lazy sample, skip the ladder for this forward rather than sync.
            if (ctx.ple_pending) |pp| if (!pp.gpu) {
                var ids_avail = false;
                if (mlx._mlx_array_is_available(&ids_avail, pp.token_ids) == 0 and ids_avail) try self.flushDeferredPle(ctx) else ladder_blocked = true;
            };
            // `mlp_out` rides along: the HC write that folds it into `h` is deferred to the next read.
            if (!ladder_blocked) ladderStepMulti(&.{ h, mlp_out }, ladder, layer_idx, seq_len);
        }
        if (vladder != 0 and layer_idx + 1 < layerCap(cfg.num_hidden_layers) and (layer_idx == 0 or (layer_idx + 1) % vladder == 0)) {
            // past the PLE layer its host leaf is filled first, so the draft ids sync here instead of after the build
            if (layer_idx > 0) if (ctx.ple_pending) |pp| if (!pp.gpu) try self.flushDeferredPle(ctx);
            const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ h, mlp_out }, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
            if (!verify_ladder_said) {
                verify_ladder_said = true;
                log.info("[qwen4] verify ladder engaged: stride {d} ({d} rows)\n", .{ vladder, seq_len });
            }
        }

        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) {
                    try self.hcFlush(&h, batch, seq_len, &pending);
                    _ = mlx.mlx_array_set(slot, h);
                }
            }
        }
        if (is_prefill and prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % eval_cadence == 0 or layer_idx + 1 == layerCap(cfg.num_hidden_layers))) {
            try evalCadencePoint(h, ctx.ssm_entries);
        }
        dt.layer(h, layer_idx);
    }

    if (stream_in != null or self.dist_peer != null) try self.hcFlush(&h, batch, seq_len, &pending);
    if (stream_in != null) {
        ctx.moe_seq_offset.* += @intCast(seq_len);
        return h;
    }
    if (self.dist_peer) |peer| {
        // The hop's sends are queued before its receive, so the PLE leaf must be filled now.
        if (ctx.ple_pending != null) try self.flushDeferredPle(ctx);
        const back = try peer.hop(h, offset);
        _ = mlx.mlx_array_free(h);
        h = back;
    }
    if (ctx.capture_stream_all != null or ctx.capture_hidden != null or ctx.capture_hidden_all != null) try self.hcFlush(&h, batch, seq_len, &pending);
    ctx.moe_seq_offset.* += @intCast(seq_len);
    dt.end(h);
    prof.report(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)), ctx.capture_ssm_seq, cfg.num_hidden_layers - cfg.attnCacheLayerCount(), cfg.attnCacheLayerCount());
    Qwen4AttnProf.flush(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)));
    if (ctx.capture_stream_all) |target| _ = mlx.mlx_array_set(target, h);
    // On this arch the spec "hidden" IS the pre-mixer stream: the MTP head
    // consumes `[B, L, hc*hidden]`, never the mixed 2560 (vLLM/SGLang).
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(h);
        const last = hs[1] - 1;
        const start = [_]c_int{ 0, last, 0 };
        const stop = [_]c_int{ hs[0], hs[1], hs[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, h, &start, 3, &stop, 3, &strides, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, h);

    const mix = try self.hcReadPending(&h, &self.qwen4_mixer, batch, seq_len, &pending);
    // The function-scope errdefer reads `h` at unwind time, so surrender the handle where it is released.
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) return final;
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try self.lmHeadProject(final);
    _ = mlx.mlx_array_free(final);
    return logits;
}

/// Forward-pass diagnostics for qwen4_exp. QWEN4_PROFILE_FWD=1 (S 2..16) or
/// =all: per-block GPU ms with a sync per block, never on by default;
/// QWEN4_OPCOUNT=1: FFI op attribution per block, no syncs.
var qwen4_stream_f32_env: ?bool = null;

/// The verify ladder's one engaged line.
var verify_ladder_said = false;

/// The residual streams run in f32 (the fixture oracle's bar, or the `QWEN4_STREAM_F32` diag).
pub fn streamF32(self: *const Trunk) bool {
    return self.qwen4_stream_f32 or env.onCached(&qwen4_stream_f32_env, .QWEN4_STREAM_F32);
}

// ── tests ──

// ── qwen4_exp fixture parity (env-gated: QWEN4_TEST_MODEL + QWEN4_FIXTURE) ──

pub const Qwen4RowCmp = @import("../../sdk/testing/parity.zig").Qwen4RowCmp;
pub const Qwen4Ties = @import("../../sdk/testing/parity.zig").Qwen4Ties;

/// One per-request slot for the qwen4 batched-vs-serial test.
pub const Qwen4TestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,
    /// The MTP head's per-request stash, so a committed-state bar can assert the trunk
    /// verify left it where the solo round did.
    head: Trunk.Qwen4MtpState,

    pub fn init(alloc: std.mem.Allocator, n_layers: u32) !*Qwen4TestSlot {
        const sl = try alloc.create(Qwen4TestSlot);
        const cache = try KVCache.init(alloc, n_layers);
        const head_cache = try KVCache.init(alloc, 1);
        sl.* = .{
            .cache = cache,
            .entries = try alloc.alloc(SSMCacheEntry, n_layers),
            .head = .{
                .cache = head_cache,
                .entry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            },
        };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{ .cache = &sl.cache, .moe_seq_offset = &sl.off, .ssm_entries = sl.entries, .ssm_member_gen = nextSsmMemberGen(), .capture_hidden = null, .vision_embeddings = null };
        return sl;
    }
    pub fn deinit(sl: *Qwen4TestSlot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        _ = mlx.mlx_array_free(sl.head.entry.conv_state);
        _ = mlx.mlx_array_free(sl.head.entry.ssm_state);
        sl.head.cache.deinit();
        sl.cache.deinit();
        alloc.destroy(sl);
    }
    pub fn forward(sl: *Qwen4TestSlot, xfm: anytype, ids: []const i32) !mlx.mlx_array {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        return xfm.forwardWith(&sl.ctx, arr);
    }
};

fn qwen4Rows(arr: mlx.mlx_array) c_int {
    return if (arr.ctx != null) mlx.getShape(arr)[1] else -1;
}

/// A decode-width `forwardQwen4With` over dense synthetic weights with every block a stand-in: the graph is the
/// embedding, one host-gathered PLE layer, the final mixer and the lm_head. Ladder strides start at the shipped
/// defaults (serial off, batched `QWEN4_DECODE_ASYNC_LADDER_DEFAULT`).
const Qwen4LadderRig = struct {
    fx: PleArmFixture,
    layer_bytes: [LAYERS * @sizeOf(MoeLayerWeights)]u8 align(@alignOf(MoeLayerWeights)),
    arrs: [16]mlx.mlx_array,
    n: usize,

    const LAYERS = 4;
    const HIDDEN = 128;
    const HC = 2;
    const RANK = 16;
    const VOCAB = 64;

    const Slot = struct {
        entries: [LAYERS]SSMCacheEntry,
        off: usize,

        fn init() Slot {
            var s: Slot = .{ .entries = undefined, .off = 0 };
            for (&s.entries) |*e| e.* = pleArmEntry();
            return s;
        }

        fn deinit(self: *Slot) void {
            for (&self.entries) |*e| ssmFreeQsaState(e);
        }
    };

    const Step = struct { logits: []f32, pending: bool };

    fn init(self: *Qwen4LadderRig) !void {
        try self.fx.init(null);
        self.n = 0;
        try self.fx.arm(false);
        const x = self.fx.xfm();
        const st = x.qwen4.?;
        const emb_dim: c_int = @intCast(st.table.dim * st.hash.n_heads);
        x.config = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .hidden_size = HIDDEN, .hc_count = HC, .num_hidden_layers = LAYERS, .vocab_size = VOCAB };
        var prng = std.Random.DefaultPrng.init(59);
        const r = prng.random();
        x.emb_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        x.lm_head_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        var ones = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{HIDDEN}, 1, .bfloat16, x.s));
        self.keep(ones);
        x.ones_hidden = ones;
        const none: mlx.mlx_array = .{ .ctx = null };
        x.qwen4_mixer = .{
            .norm_w = try self.weight(r, &.{ HC, HIDDEN }),
            .down_w = try self.weight(r, &.{ HC * HIDDEN, RANK }),
            .down_s = none,
            .down_b = none,
            .up_w = try self.weight(r, &.{ RANK, HC * HIDDEN }),
            .up_s = none,
            .up_b = none,
        };
        self.layer_bytes = @splat(0);
        const layers: [*]MoeLayerWeights = @ptrCast(&self.layer_bytes);
        for (layers[0..LAYERS]) |*lw| {
            lw.hc_attn = null;
            lw.hc_mlp = null;
            lw.ple = null;
        }
        layers[0].ple = .{
            .key_w = try self.weight(r, &.{ emb_dim, HC * HIDDEN }),
            .key_s = none,
            .key_b = none,
            .value_w = try self.weight(r, &.{ emb_dim, HIDDEN }),
            .value_s = none,
            .value_b = none,
            .norm_key = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_query = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_conv = try self.weight(r, &.{ HC, HIDDEN }),
            .conv_w = try self.weight(r, &.{ HC * HIDDEN, @intCast(x.config.ple_conv_kernel), 1 }),
        };
        x.moe_layers = layers[0..LAYERS];
        x.decode_async_ladder = 0;
        x.decode_async_ladder_qwen4 = QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    }

    fn deinit(self: *Qwen4LadderRig) void {
        for (self.arrs[0..self.n]) |a| _ = mlx.mlx_array_free(a);
        self.fx.deinit();
    }

    fn keep(self: *Qwen4LadderRig, a: mlx.mlx_array) void {
        self.arrs[self.n] = a;
        self.n += 1;
    }

    fn weight(self: *Qwen4LadderRig, r: std.Random, shape: []const c_int) !mlx.mlx_array {
        const a = try attn256RandBf16(r, shape, self.fx.xfm().s);
        self.keep(a);
        return a;
    }

    /// One serial decode step on a deferred PLE leaf at serial ladder stride `ladder`; reports whether the leaf was
    /// still pending when the build returned, then fills it and reads the logits.
    fn step(self: *Qwen4LadderRig, slot: *Slot, ids: mlx.mlx_array, ladder: u32) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder = ladder;
        defer x.decode_async_ladder = 0;
        var ctx: ForwardCtx = .{ .cache = @ptrCast(&self.fx.cache_bytes), .moe_seq_offset = &slot.off, .ssm_entries = &slot.entries, .capture_hidden = null, .vision_embeddings = null, .ple_defer = true };
        const logits = try x.forwardQwen4With(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        const pending = ctx.ple_pending != null;
        try x.flushDeferredPle(&ctx);
        return .{ .logits = try qwen4ReadF32(testing.allocator, logits, x.s), .pending = pending };
    }

    fn prefill(self: *Qwen4LadderRig, sl: *Qwen4TestSlot, ids: []const i32) !void {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const logits = try self.fx.xfm().forwardQwen4With(&sl.ctx, arr);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }

    /// One batched decode step over `slots` at batched ladder stride `ladder`, filling (or discarding) the leaf.
    fn batchedStep(self: *Qwen4LadderRig, slots: []const *Qwen4TestSlot, ids: []const i32, ladder: u32, fill: bool) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder_qwen4 = ladder;
        defer x.decode_async_ladder_qwen4 = QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
        const run = try qwen4BatchedPleForward(x, slots, ids, 1, false, true, fill);
        defer _ = mlx.mlx_array_free(run.logits);
        return .{ .logits = try qwen4ReadF32(testing.allocator, run.logits, x.s), .pending = run.pending };
    }
};

fn expectLogitBitsEqual(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) return;
    var n: usize = 0;
    var max: f32 = 0;
    for (want, got) |a, b| {
        if (@as(u32, @bitCast(a)) == @as(u32, @bitCast(b))) continue;
        n += 1;
        max = @max(max, @abs(a - b));
    }
    std.debug.print("logits differ in {d}/{d} entries, max |diff| {d}\n", .{ n, want.len, max });
    return error.TestExpectedEqual;
}
