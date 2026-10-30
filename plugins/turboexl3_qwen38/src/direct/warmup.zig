//! The MTP verify warm-up at load: every draft width's solo verify and the head's own step compile their Metal
//! pipelines before the first measured round. Today's `trunk/warmup.zig` with the solo arm of the row-axis verify it
//! runs (`trunk/verify.zig` `forwardRowAxisVerify`). Its grouped rows are gone with the grouped verify: the drafter
//! verifies one slot at a time, so serving never ran the pipelines they compiled.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const KVQuantConfig = @import("../mlx_serve/engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const ForwardCtx = @import("../mlx_serve/engines/mlx/forward_ctx.zig").ForwardCtx;
const ssmFreeQsaState = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const state = @import("cache.zig");
const KVCache = state.KVCache;
const SSMCacheEntry = state.SSMCacheEntry;
const fwd = @import("model.zig");
const mtp_head = @import("mtp_head.zig");
const ple = @import("ple.zig");

const MAX_WIDTH_WARM: usize = 8;
const MAX_WARM_SEQ: usize = MAX_WIDTH_WARM + 1;

const Slot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,

    fn init(t: *Trunk, kv_config: KVQuantConfig) !*Slot {
        const alloc = t.allocator;
        const n_layers = t.config.num_hidden_layers;
        const sl = try alloc.create(Slot);
        errdefer alloc.destroy(sl);
        var cache = try KVCache.initWithConfig(alloc, n_layers, kv_config);
        errdefer cache.deinit();
        sl.* = .{ .cache = cache, .entries = try alloc.alloc(SSMCacheEntry, n_layers) };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{
            .cache = &sl.cache,
            .moe_seq_offset = &sl.off,
            .ssm_entries = sl.entries,
            .ssm_member_gen = state.nextSsmMemberGen(),
            .capture_hidden = null,
            .vision_embeddings = null,
        };
        return sl;
    }

    fn deinit(sl: *Slot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            state.ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        sl.cache.deinit();
        alloc.destroy(sl);
    }
};

/// The serial forward: a slot that sat in a batched group gets its own entries back first.
fn forwardWith(t: *Trunk, ctx: *ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    try state.ssmGroupReleaseCtx(&t.ssm_group, t.s, ctx);
    return fwd.forward(t, ctx, ids);
}

/// One solo verify of `seq` rows, as a round's: every recurrent row's state captured, the PLE leaf deferred and then
/// filled through the group flush the row-axis verify runs.
fn verifyAt(t: *Trunk, sl: *Slot, seq: usize) !void {
    var ids: [MAX_WARM_SEQ]i32 = undefined;
    const row = warmIds(ids[0..seq]);
    defer _ = mlx.mlx_array_free(row);
    const ctx = &sl.ctx;
    errdefer {
        ple.discardDeferredPle(t, ctx);
        if (ctx.ssm_entries) |entries| for (entries) |*e| state.ssmFreeSpecCapture(e);
    }
    var last = mlx.mlx_array_new();
    var all = mlx.mlx_array_new();
    ctx.capture_ssm_seq = ctx.ssm_entries != null;
    ctx.ple_defer = true;
    ctx.capture_hidden = &last;
    ctx.capture_hidden_all = &all;
    const logits = forwardWith(t, ctx, row);
    ctx.capture_hidden = null;
    ctx.capture_hidden_all = null;
    ctx.ple_defer = false;
    ctx.capture_ssm_seq = false;
    const lg = logits catch |e| {
        _ = mlx.mlx_array_free(last);
        _ = mlx.mlx_array_free(all);
        return e;
    };
    ple.flushDeferredPleGroup(t, &.{ctx}) catch |e| {
        for ([_]mlx.mlx_array{ lg, last, all }) |a| _ = mlx.mlx_array_free(a);
        return e;
    };
    evalFree(&.{ lg, last, all });
}

pub fn warmupSpecVerify(self: *Trunk, max_width: u32, kv_config: KVQuantConfig) !void {
    if (self.qwen4_mtp == null) return;
    const width_cap: usize = @min(@max(max_width, 1), MAX_WIDTH_WARM);

    const sl = try Slot.init(self, kv_config);
    defer {
        state.ssmGroupReleaseCtx(&self.ssm_group, self.s, &sl.ctx) catch {};
        sl.deinit(self.allocator);
        _ = mlx.mlx_clear_cache();
    }
    var head_in = mlx.mlx_array_new();
    defer if (head_in.ctx != null) {
        _ = mlx.mlx_array_free(head_in);
    };
    {
        var prompt_ids: [8]i32 = undefined;
        const prompt = warmIds(&prompt_ids);
        defer _ = mlx.mlx_array_free(prompt);
        var stream = mlx.mlx_array_new();
        sl.ctx.capture_stream_all = &stream;
        const logits = try forwardWith(self, &sl.ctx, prompt);
        sl.ctx.capture_stream_all = null;
        if (stream.ctx != null) {
            const sh = mlx.getShape(stream);
            try mlx.check(mlx.mlx_slice(&head_in, stream, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        }
        evalFree(&[_]mlx.mlx_array{ logits, stream });
    }

    var w: usize = 1;
    while (w <= width_cap) : (w += 1) try verifyAt(self, sl, w + 1);

    if (head_in.ctx != null) {
        var st = try mtp_head.qwen4MtpStateNew(self);
        defer st.deinit();
        var id_buf: [1]i32 = undefined;
        const id_arr = warmIds(&id_buf);
        defer _ = mlx.mlx_array_free(id_arr);
        for ([_]Trunk.Qwen4MtpProject{ .mixed_last_row, .none }, 0..) |project, i| {
            const out = try mtp_head.qwen4MtpForwardOn(self, &st, head_in, id_arr, @intCast(1 + i), project);
            evalFree(&[_]mlx.mlx_array{ out.logits, out.stream, out.mixed });
        }
    }
}

fn warmIds(buf: []i32) mlx.mlx_array {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const shape = [_]c_int{ 1, @intCast(buf.len) };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 2, .int32);
}

fn evalFree(arrays: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, a);
    }
    _ = mlx.mlx_eval(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    }
}
