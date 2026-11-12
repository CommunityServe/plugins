//! The split prompt forward's refusals and spans, MLX-free (patches/turboexl3-model.patch and
//! patches/turboexl3-sdk.patch). With SSM checkpointing on,
//! mlx-serve's prompt loop holds back a prompt's last 30 tokens and a final forward runs them with the last token: a 16K
//! prompt is a 16,356-row forward, then a 31-row one, whose routed layers read again records the first one read. A split
//! forward (`Forward.split_rows`) runs the two spans layer by layer with both spans' routed experts in one streamed call
//! a layer. The copy serves it only where
//! each span's arithmetic is its own forward's and refuses any other by name before a cache is touched: `splitSpans`
//! (one slot's prompt rows, no captures or media rows, a cache for span A's states, two spans of more than
//! `HC_PRE_MAX_ROWS` rows, every MoE layer on the streamed source) and `splitServes` (a layer's routed experts on the
//! bank's `.source` arm, and the `fused` arm for neither span). `splitOf` reads the fields only from a kit and a host
//! whose `Forward` has them.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = kit.mlx;
const host = @import("mlx_serve/plugins/glm5_next/host.zig");
const fwd = @import("mlx_serve/plugins/glm5_next/table_forward.zig");
const moe = @import("mlx_serve/sdk/bases/blocks/moe.zig");
const routed = @import("mlx_serve/sdk/weights/routed.zig");

const testing = std.testing;

/// A prompt forward of `toks` from position `pos[0]` on the one cache `cache[0]`.
fn prompt(toks: []const u32, pos: *const [1]u32, cache: *const [1]abi.Handle) abi.Forward {
    return .{ .mode = .prefill, .rows = @intCast(toks.len), .tokens = abi.Span(u32).of(toks), .positions = abi.Span(u32).of(pos), .caches = abi.Span(abi.Handle).of(cache) };
}

test "turboexl3_glm53f stream split: a split prompt forward is refused by name unless each span takes its own forward's arms; else its spans, span B from span A's end" {
    var toks: [71]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i);
    const pos = [1]u32{100};
    const cache = [1]abi.Handle{@ptrFromInt(0x1000)};
    const f = prompt(&toks, &pos, &cache);
    // 40 + 31 rows (a 16K prompt's tail span), span B at 140; then both spans one row past hcPre's
    try testing.expectEqual(fwd.Spans{ .a = 40, .b = 31, .pos_a = 100, .pos_b = 140 }, try fwd.splitSpans(&f, 40, true, true));
    const f34 = prompt(toks[0..34], &pos, &cache);
    try testing.expectEqual(fwd.Spans{ .a = 17, .b = 17, .pos_a = 100, .pos_b = 117 }, try fwd.splitSpans(&f34, 17, true, true));
    // the engine's device ids in place of host tokens
    var ids = f;
    ids.tokens = .{};
    ids.ids = .{ .ctx = @ptrFromInt(0x2000) };
    try testing.expectEqual(fwd.Spans{ .a = 40, .b = 31, .pos_a = 100, .pos_b = 140 }, try fwd.splitSpans(&ids, 40, true, true));
    // not a prompt
    for ([_]abi.Mode{ .decode, .verify, .batched_rows }) |mode| {
        var x = f;
        x.mode = mode;
        try testing.expectError(error.Glm53fSplitMode, fwd.splitSpans(&x, 40, true, true));
    }
    // two slots
    const caches2 = [2]abi.Handle{ @ptrFromInt(0x1000), @ptrFromInt(0x3000) };
    var two = f;
    two.caches = abi.Span(abi.Handle).of(&caches2);
    try testing.expectError(error.Glm53fSplitSlots, fwd.splitSpans(&two, 40, true, true));
    // DFlash captures, media rows, no cache for span A's states
    const layers = [_]u32{10};
    var cap = f;
    cap.capture_layers = abi.Span(u32).of(&layers);
    try testing.expectError(error.Glm53fSplitCapture, fwd.splitSpans(&cap, 40, true, true));
    var media = f;
    media.media = .{ .ctx = @ptrFromInt(0x4000) };
    try testing.expectError(error.Glm53fSplitMedia, fwd.splitSpans(&media, 40, true, true));
    try testing.expectError(error.Glm53fSplitStates, fwd.splitSpans(&f, 40, false, true));
    // no span B, host tokens other than the rows, span B past the last position
    try testing.expectError(error.Glm53fSplitRows, fwd.splitSpans(&f, 71, true, true));
    var less = f;
    less.tokens = abi.Span(u32).of(toks[0..70]);
    try testing.expectError(error.Glm53fSplitRows, fwd.splitSpans(&less, 40, true, true));
    const pos_hi = [1]u32{std.math.maxInt(u32) - 10};
    const hi = prompt(&toks, &pos_hi, &cache);
    try testing.expectError(error.Glm53fSplitRows, fwd.splitSpans(&hi, 40, true, true));
    // a span of 16 rows or fewer: its sublayers would take hcPre's deferred arm
    try testing.expectError(error.Glm53fSplitShortSpan, fwd.splitSpans(&f, 16, true, true));
    const f56 = prompt(toks[0..56], &pos, &cache);
    try testing.expectError(error.Glm53fSplitShortSpan, fwd.splitSpans(&f56, 40, true, true));
    // a MoE layer off the streamed source arm
    try testing.expectError(error.Glm53fSplitRoutedPath, fwd.splitSpans(&f, 40, true, false));
    // the fields: read only from a kit and a host whose Forward has them; split_rows 0 is an ordinary forward
    var sf = f;
    try testing.expect(host.splitOf(&sf) == null);
    if (@hasField(abi.Forward, "split_rows") and @hasField(abi.Forward, "split_states")) {
        sf.split_rows = 40;
        sf.split_states = @ptrFromInt(0x5000);
        const sp = host.splitOf(&sf) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(u32, 40), sp.rows_a);
        try testing.expectEqual(sf.split_states, sp.states);
        sf.size = @offsetOf(abi.Forward, "split_rows");
        try testing.expect(host.splitOf(&sf) == null);
    }
}

/// A streamed layer's seams, never called: the test reads which arm a layer's weights name.
const Seams = struct {
    var ctx: u8 = 0;
    fn source(_: *anyopaque, _: u32, _: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_array) routed.Ret(mlx.mlx_array) {
        return .{};
    }
    fn trunk(_: *anyopaque, _: u32, _: []const u16, _: []const f32) routed.Ret(routed.Fetched) {
        return .{};
    }
    fn settle(_: *anyopaque) routed.Ret(void) {
        return .{};
    }
    fn fused(_: *const moe.Fused, _: *const moe.Moe.Weights, _: mlx.mlx_stream, _: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_array) anyerror!?mlx.mlx_array {
        return null;
    }
};

test "turboexl3_glm53f stream split: a MoE layer serves a split only where both spans take the streamed bank's source arm and neither the fused arm" {
    const none: mlx.mlx_array = .{ .ctx = null };
    const bank: moe.Bank = .{ .w = none, .qp = .{ .bits = 3, .group_size = 64 } };
    const source: routed.Routed = .{ .ctx = &Seams.ctx, .math = .{ .source = &Seams.source }, .settle_fn = &Seams.settle };
    var w: moe.Moe.Weights = .{ .router = .{}, .route = .softmax, .gate = bank, .up = bank, .down = bank, .top_k = 8, .experts = 288, .streamed = .{ .routed = source, .layer = 3 } };
    // the plugin's routed layers: the bank's source runs the math (a 16K prompt's spans)
    try testing.expect(moe.splitServes(&w, 16_356, 31));
    try testing.expect(!moe.splitServes(&w, 16_356, 0));
    // a model's one-kernel arm of up to 16 rows (the built-in's resident GLM decode): a span it may take is refused
    w.fused = .{ .rows = 16, .limit = 10, .run = &Seams.fused };
    try testing.expect(moe.splitServes(&w, 40, 17));
    try testing.expect(!moe.splitServes(&w, 40, 16));
    try testing.expect(!moe.splitServes(&w, 16, 31));
    w.fused = null;
    // the trunk's own gathers on the fetched slots (an MXFP4 bank's math), and resident experts: refused
    w.streamed.?.routed.math = .{ .trunk = &Seams.trunk };
    try testing.expect(!moe.splitServes(&w, 40, 31));
    w.streamed = null;
    try testing.expect(!moe.splitServes(&w, 40, 31));
}
