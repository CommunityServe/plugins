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

pub fn forwardRowAxisVerify(
    self: *Transformer,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
    row_ns: ?[]u64,
    rollback_out: ?[]VerifyRollback,
) !void {
    std.debug.assert(token_rows.len == ctxs.len);
    std.debug.assert(out_logits.len >= token_rows.len and out_last.len >= token_rows.len and out_all.len >= token_rows.len);
    // qwen4_exp's grouped verify runs on its trunk module (a single row keeps the plugin forward below), owed PLE work
    // settled first.
    if (token_rows.len > 1) if (intree.inner(self)) |t| {
        try self.flushDeferredPleGroup(ctxs);
        return modules.call("qwen4_row_axis_verify", .{ t, token_rows, ctxs, out_logits, out_last, out_all, row_ns, rollback_out });
    };
    const outer_group_rows = verify_projection.verify_group_rows;
    verify_projection.verify_group_rows = token_rows.len;
    defer verify_projection.verify_group_rows = outer_group_rows;
    if (token_rows.len > 1 and !layered.row_axis_verify_logged) {
        layered.row_axis_verify_logged = true;
        log.info("[batched] row-axis mtp verify engaged (slots={d}, width={d})\n", .{ token_rows.len, mlx.getShape(token_rows[0])[1] });
    }
    if (row_ns) |ns| @memset(ns[0..token_rows.len], 0);
    self.verify_laps = .{};
    var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    var built: usize = 0;
    var touched: usize = 0;
    errdefer {
        for (0..built) |i| {
            _ = mlx.mlx_array_free(out_logits[i]);
            _ = mlx.mlx_array_free(out_last[i]);
            _ = mlx.mlx_array_free(out_all[i]);
        }
        for (0..touched) |i| {
            self.discardDeferredPle(ctxs[i]);
            if (ctxs[i].ssm_entries) |entries| for (entries) |*e| ssmFreeSpecCapture(e);
        }
    }
    for (token_rows, ctxs, 0..) |row, ctx, i| {
        touched = i + 1;
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        // Per-position SSM capture lets a partial accept roll back without a re-forward; a served model's deferred
        // host work keeps out of the graph build.
        ctx.capture_ssm_seq = ctx.ssm_entries != null;
        ctx.ple_defer = true;
        const logits = self.forwardWithCaptureAll(ctx, row, &last, &all) catch |e| {
            ctx.ple_defer = false;
            ctx.capture_ssm_seq = false;
            _ = mlx.mlx_array_free(last);
            _ = mlx.mlx_array_free(all);
            return e;
        };
        ctx.ple_defer = false;
        ctx.capture_ssm_seq = false;
        out_logits[i] = logits;
        out_last[i] = last;
        out_all[i] = all;
        built = i + 1;
        const row_build_ns = lap.read();
        self.verify_laps.build_ns += row_build_ns;
        lap.reset();
        if (row_ns) |ns| ns[i] += row_build_ns;
    }
    // Settles the deferred host work after every row's build, before the first eval downstream of it.
    try self.flushDeferredPleGroup(ctxs[0..token_rows.len]);
    const ple_ns = lap.read();
    self.verify_laps.ple_ns = ple_ns;
    if (row_ns) |ns| ns[0] += ple_ns;
}

/// The laps of the last grouped verify: qwen4_exp's trunk keeps its own.
pub fn verifyLaps(self: *Transformer) *VerifyLaps {
    if (intree.trunk(self)) |t| return &t.verify_laps;
    return &self.verify_laps;
}

/// Hand the joined verify block to the caller (null when the projection ran per row; only qwen4_exp's grouped verify
/// joins one).
pub fn takeJoinedVerifyLogits(self: *Transformer) mlx.mlx_array {
    const t = intree.trunk(self) orelse return .{ .ctx = null };
    return t.takeJoinedVerifyLogits();
}

// ── tests ──

pub fn expectRowsByteEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, tag: []const u8) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    if (ok) return;
    const ha = try qwen4ReadF32(alloc, a, s);
    defer alloc.free(ha);
    const hb = try qwen4ReadF32(alloc, b, s);
    defer alloc.free(hb);
    var max_diff: f64 = 0;
    var first: usize = std.math.maxInt(usize);
    var n_diff: usize = 0;
    for (ha, hb, 0..) |x, y, i| {
        if (x == y) continue;
        n_diff += 1;
        if (first == std.math.maxInt(usize)) first = i;
        const d = @abs(@as(f64, x) - @as(f64, y));
        if (d > max_diff) max_diff = d;
    }
    std.debug.print("[mtp verify] {s}: NOT byte-equal — {d}/{d} elements differ, first at {d} ({d} vs {d}), max |diff| {d}\n", .{ tag, n_diff, ha.len, first, ha[first], hb[first], max_diff });
    return error.RowNotByteEqual;
}

pub const L10VerifySlots = struct {
    solo: [2]*Qwen4TestSlot,
    group: [2]*Qwen4TestSlot,

    pub fn init(alloc: std.mem.Allocator, xfm: anytype, n_layers: u32, prompts: [2][]const i32) !L10VerifySlots {
        var out: L10VerifySlots = undefined;
        for (0..2) |i| {
            out.solo[i] = try Qwen4TestSlot.init(alloc, n_layers);
            out.group[i] = try Qwen4TestSlot.init(alloc, n_layers);
            _ = mlx.mlx_array_free(try out.solo[i].forward(xfm, prompts[i]));
            _ = mlx.mlx_array_free(try out.group[i].forward(xfm, prompts[i]));
        }
        return out;
    }
    pub fn deinit(self: *L10VerifySlots, alloc: std.mem.Allocator) void {
        for (0..2) |i| {
            self.solo[i].deinit(alloc);
            self.group[i].deinit(alloc);
        }
    }
};

/// The solo MTP verify: capture armed, PLE deferred, one `forwardWithCaptureAll`, flush
/// after. `forwardRowAxisVerify` at one row must match it.
pub fn logicalSsmCapture(s: mlx.mlx_stream, entry: *const SSMCacheEntry) !mlx.mlx_array {
    const shape = mlx.getShape(entry.spec_state_seq);
    if (shape.len != 5 or shape[0] < 1) return error.SpecCaptureShape;
    var final = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(final);
    try mlx.check(mlx.mlx_expand_dims(&final, entry.ssm_state, 0, s));
    if (shape[0] == 1) return final;
    defer _ = mlx.mlx_array_free(final);
    var prefix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prefix);
    var stop: [5]c_int = undefined;
    @memcpy(&stop, shape);
    stop[0] -= 1;
    const start: [5]c_int = @splat(0);
    const stride: [5]c_int = @splat(1);
    try mlx.check(mlx.mlx_slice(&prefix, entry.spec_state_seq, &start, 5, &stop, 5, &stride, 5, s));
    return Transformer.concatAxis0(s, &.{ prefix, final });
}

fn rollbackRowToAccepted(sl: *Qwen4TestSlot, accepted: u32, verify_len: u32, off_snap: usize, step_snap: usize, s: mlx.mlx_stream) !void {
    const accepted_len: usize = 1 + @as(usize, accepted);
    try sl.cache.truncate(off_snap + accepted_len, s);
    sl.cache.step = step_snap;
    // Full acceptance keeps the dedicated final state; the capture tail is unwritten.
    if (accepted_len < verify_len) for (sl.entries) |*e| {
        try ssmRollbackFromCapture(e, accepted, verify_len, s);
    };
    sl.off = off_snap + accepted_len;
    for (sl.entries) |*e| ssmFreeSpecCapture(e);
}

fn expectSlotStateEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: *Qwen4TestSlot, b: *Qwen4TestSlot, row: usize) !void {
    try testing.expectEqual(a.off, b.off);
    try testing.expectEqual(a.cache.step, b.cache.step);
    try testing.expectEqual(a.entries.len, b.entries.len);
    var buf: [96]u8 = undefined;
    for (a.entries, b.entries, 0..) |*ea, *eb, l| {
        try testing.expectEqual(ea.initialized, eb.initialized);
        try testing.expectEqual(ea.ple_prev_valid, eb.ple_prev_valid);
        try testing.expectEqualSlices(u32, &ea.ple_prev, &eb.ple_prev);
        try testing.expectEqual(a.cache.entries[l].offset, b.cache.entries[l].offset);
        inline for (.{ "conv_state", "ssm_state", "qsa_pooled", "aux_state" }) |field| {
            const xa = @field(ea, field);
            const xb = @field(eb, field);
            try testing.expectEqual(xa.ctx == null, xb.ctx == null);
            if (xa.ctx != null) {
                const tag = try std.fmt.bufPrint(&buf, "row {d} layer {d} {s}", .{ row, l, field });
                try expectRowsByteEqual(alloc, s, xa, xb, tag);
            }
        }
    }
    const ma = a.head.meta();
    const mb = b.head.meta();
    try testing.expectEqual(ma.stash_len, mb.stash_len);
    try testing.expectEqual(ma.origin, mb.origin);
    try testing.expectEqual(ma.cache_step, mb.cache_step);
    try testing.expectEqual(a.head.cache.entries[0].offset, b.head.cache.entries[0].offset);
    try testing.expectEqual(a.head.entry.initialized, b.head.entry.initialized);
    try testing.expectEqualSlices(u32, &a.head.entry.ple_prev, &b.head.entry.ple_prev);
}

fn verifyRollbackEvalStorage(entry: *const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const value = @field(entry, field);
        if (value.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, value);
    }
    try mlx.check(mlx.mlx_eval(vec));
}

fn verifyRollbackStorageAddress(value: mlx.mlx_array) !usize {
    return switch (mlx.mlx_array_dtype(value)) {
        .bfloat16 => @intFromPtr(mlx.mlx_array_data_bfloat16(value) orelse return error.KvUnreadable),
        .uint32 => @intFromPtr(mlx.mlx_array_data_uint32(value) orelse return error.KvUnreadable),
        else => error.KvUnreadable,
    };
}

fn verifyRollbackValidPrefixEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.initialized, b.initialized);
    try testing.expectEqual(a.offset, b.offset);
    if (!a.initialized) return;
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const av = @field(a, field);
        const bv = @field(b, field);
        try testing.expectEqual(av.ctx == null, bv.ctx == null);
        if (av.ctx != null) {
            const ash = mlx.getShape(av);
            const bsh = mlx.getShape(bv);
            var aslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(aslice);
            var bslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bslice);
            try mlx.check(mlx.mlx_slice(&aslice, av, &.{ 0, 0, 0, 0 }, 4, &.{ ash[0], ash[1], @intCast(a.offset), ash[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&bslice, bv, &.{ 0, 0, 0, 0 }, 4, &.{ bsh[0], bsh[1], @intCast(b.offset), bsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try testing.expect(try qsaArraysAllEqual(aslice, bslice, s));
        }
    }
}

fn verifyRollbackKvCase(config: KVQuantConfig, prefix_len: usize, force_growth: bool) !void {
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(allocator, 1, config);
    defer cache.deinit();
    var reference = try KVCache.initWithConfig(allocator, 1, config);
    defer reference.deinit();
    var prng = std.Random.DefaultPrng.init(1911 + prefix_len + @intFromBool(force_growth));
    if (prefix_len != 0) {
        const prefix = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(prefix_len), 64 }, stream);
        defer _ = mlx.mlx_array_free(prefix);
        var cv = try cache.update(0, prefix, prefix, stream, 0);
        cv.deinit();
        var rv = try reference.update(0, prefix, prefix, stream, 0);
        rv.deinit();
        try verifyRollbackEvalStorage(&cache.entries[0]);
        try verifyRollbackEvalStorage(&reference.entries[0]);
    }
    const tail_len = if (force_growth and prefix_len != 0)
        KVCache.bufferCapacity(cache.entries[0].keys) - cache.entries[0].offset + 1
    else
        4;
    const tail = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(tail_len), 64 }, stream);
    defer _ = mlx.mlx_array_free(tail);
    var addresses: [6]usize = @splat(0);
    if (prefix_len != 0 and !force_growth) {
        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }, 0..) |field, i| {
            const value = @field(cache.entries[0], field);
            if (value.ctx != null) addresses[i] = try verifyRollbackStorageAddress(value);
        }
    }
    var offset = prefix_len;
    var ctx = ForwardCtx{ .cache = &cache, .moe_seq_offset = &offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var fwd_gen: u64 = 0;
    var rollback = try VerifyRollback.capture(allocator, &fwd_gen, &ctx);
    defer rollback.deinit();
    var appended = try cache.update(0, tail, tail, stream, 0);
    appended.deinit();
    offset += tail_len;
    try verifyRollbackEvalStorage(&cache.entries[0]);
    if (prefix_len != 0 and !force_growth) {
        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }, 0..) |field, i| {
            const value = @field(cache.entries[0], field);
            if (value.ctx != null) try testing.expectEqual(addresses[i], try verifyRollbackStorageAddress(value));
        }
    }
    rollback.restore();
    try testing.expectEqual(prefix_len, offset);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
    var resumed = try cache.update(0, tail, tail, stream, 0);
    resumed.deinit();
    var expected = try reference.update(0, tail, tail, stream, 0);
    expected.deinit();
    try verifyRollbackEvalStorage(&cache.entries[0]);
    try verifyRollbackEvalStorage(&reference.entries[0]);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
}
