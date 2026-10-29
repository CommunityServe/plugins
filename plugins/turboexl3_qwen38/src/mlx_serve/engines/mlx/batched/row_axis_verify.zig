//! copied_from: src/engines/mlx/batched/row_axis_verify.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/batched/row_axis_verify.
//! Owns VerifyLaps, VerifyRollback, forwardRowAxisVerify, verifyLaps, takeJoinedVerifyLogits, expectRowsByteEqual, and 9 more.

const std = @import("std");
const modules = @import("../modules/call.zig");
const env = @import("../../../env.zig");
const distributed_mod = @import("../cluster/distributed.zig");
const mlx = @import("../../../mlx/ffi.zig");
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util_mod = @import("../../../sdk/util/io_util.zig");
const gather_qmv_rows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig");
const layered = @import("layered.zig");
const intree = @import("../plugin/intree.zig");
const mtp = @import("../../../plugins/qwen4_exp/mtp.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const Qwen4TestSlot = @import("../../../plugins/qwen4_exp/forward.zig").Qwen4TestSlot;
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const qsaArraysAllEqual = @import("../../../sdk/testing/parity.zig").qsaArraysAllEqual;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const mtpHeadRowsFit = @import("../../../sdk/blocks/linear/verify_projection.zig").mtpHeadRowsFit;
const qwen4ReadF32 = @import("../../../sdk/testing/parity.zig").qwen4ReadF32;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const KVCacheEntry = @import("../kv/cache.zig").KVCacheEntry;
const freeKVEntry = @import("../kv/cache.zig").freeKVEntry;
const newEmptyKVEntry = @import("../kv/cache.zig").newEmptyKVEntry;
const testing = @import("../kv/cache.zig").testing;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../ssm/entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../ssm/entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("../ssm/entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("../ssm/entry.zig").ssmSnapshotDeinit;
const nextSsmMemberGen = @import("../ssm/group.zig").nextSsmMemberGen;
const Transformer = @import("../transformer.zig").Transformer;
const deinit = @import("../transformer.zig").Transformer.deinit;
const MAX_BATCH_ROWS = @import("layered.zig").MAX_BATCH_ROWS;

pub const VerifyLaps = struct {
    build_ns: u64 = 0,
    ple_ns: u64 = 0,
    ple_sync_ns: u64 = 0,
};

pub const VerifyRollback = struct {
    const KvMark = struct { offset: usize, initialized: bool };

    allocator: std.mem.Allocator,
    /// The trunk's per-forward QSA scratch key, bumped on restore.
    fwd_gen: *u64,
    ctx: *ForwardCtx,
    kv_marks: []KvMark,
    cache_step: usize,
    ssm: []SSMCacheEntrySnapshot,
    offset: usize,

    pub fn capture(allocator: std.mem.Allocator, fwd_gen: *u64, ctx: *ForwardCtx) !@This() {
        const marks = try allocator.alloc(KvMark, ctx.cache.entries.len);
        errdefer allocator.free(marks);
        for (ctx.cache.entries, marks) |entry, *mark| mark.* = .{ .offset = entry.offset, .initialized = entry.initialized };
        const entries = ctx.ssm_entries orelse &.{};
        const states = try allocator.alloc(SSMCacheEntrySnapshot, entries.len);
        for (entries, states) |*entry, *state| state.* = ssmSnapshot(entry);
        return .{ .allocator = allocator, .fwd_gen = fwd_gen, .ctx = ctx, .kv_marks = marks, .cache_step = ctx.cache.step, .ssm = states, .offset = ctx.moe_seq_offset.* };
    }

    pub fn restore(self: *@This()) void {
        for (self.ctx.cache.entries, self.kv_marks) |*current, saved| {
            if (!saved.initialized) {
                freeKVEntry(current);
                current.* = newEmptyKVEntry();
            } else {
                // As with KVCache.restore, the next update rebuilds views from the retained prefix.
                inline for (.{ "key_view", "value_view", "key_scales_view", "key_biases_view", "value_scales_view", "value_biases_view" }) |field| {
                    _ = mlx.mlx_array_free(@field(current, field));
                    @field(current, field) = mlx.mlx_array_new();
                }
            }
            current.offset = saved.offset;
        }
        self.ctx.cache.step = self.cache_step;
        self.ctx.moe_seq_offset.* = self.offset;
        if (self.ctx.ssm_entries) |entries| {
            for (entries, self.ssm) |*entry, *saved| {
                _ = mlx.mlx_array_free(entry.conv_state);
                _ = mlx.mlx_array_free(entry.ssm_state);
                ssmFreeSpecCapture(entry);
                ssmFreeQsaState(entry);
                entry.conv_state = saved.conv_state;
                entry.ssm_state = saved.ssm_state;
                entry.aux_state = saved.aux_state;
                entry.qsa_pooled = saved.qsa_pooled;
                saved.conv_state = .{ .ctx = null };
                saved.ssm_state = .{ .ctx = null };
                saved.aux_state = .{ .ctx = null };
                saved.qsa_pooled = .{ .ctx = null };
                entry.initialized = saved.initialized;
                entry.qsa_ratio = saved.qsa_ratio;
                entry.qsa_hist_rows = saved.qsa_rows;
                entry.ple_prev = saved.ple_prev;
                entry.ple_prev_valid = saved.ple_prev_valid;
                if (entry.aux_state.ctx != null) entry.qsa_key_rows = mlx.getShape(entry.aux_state)[1];
                if (entry.qsa_pooled.ctx != null) entry.qsa_pooled_blocks = mlx.getShape(entry.qsa_pooled)[1];
            }
        }
        self.ctx.ssm_member_gen = nextSsmMemberGen();
        self.fwd_gen.* +%= 1;
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.kv_marks);
        for (self.ssm) |*state| ssmSnapshotDeinit(state);
        self.allocator.free(self.ssm);
    }
};

// ── tests ──

pub const L10VerifySlots = struct {
    solo: [2]*Qwen4TestSlot,
    group: [2]*Qwen4TestSlot,

    pub fn deinit(self: *L10VerifySlots, alloc: std.mem.Allocator) void {
        for (0..2) |i| {
            self.solo[i].deinit(alloc);
            self.group[i].deinit(alloc);
        }
    }
};
