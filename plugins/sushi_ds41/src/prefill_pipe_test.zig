//! The prefill pipeline (`pipe_depth`, `SUSHI_DS41_PREFILL_PIPELINE`) against the waiting schedule on the tiny
//! checkpoint: the same ids give the same logits byte for byte, and no pipelined layer is left in flight where the
//! plugin reads the state. Run by `zig build gpu-test -Dmlx-lib=<dir> -Ddsv41-tiny=<dir>` (sushi's
//! tests/dump_dsv41_fixtures.py), which sets `DSV41_TINY`; skipped without it.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");
const dstream = @import("sushi/dsv41/dsv41_stream.zig");
const stream = @import("sushi/expert_stream.zig");
const exl3 = @import("sushi_exl3");

const testing = std.testing;
const Scope = sc_mod.Scope;

/// A logits array's f32 bytes (bf16 widens exactly, so equal bytes here are equal bf16 bytes). Non-empty and finite:
/// two arms that both broke the same way would otherwise compare equal.
fn bytesOf(sc: *Scope, a: sc_mod.A) ![]u8 {
    defer _ = mlx.mlx_array_free(a);
    const rows = try sc_mod.hostF32(try sc.astype(a, .float32));
    try testing.expect(rows.len > 0);
    for (rows) |v| try testing.expect(std.math.isFinite(v));
    return testing.allocator.dupe(u8, std.mem.sliceAsBytes(rows));
}

const Arm = struct {
    whole: []u8,
    chunked: []u8,

    fn deinit(self: Arm) void {
        testing.allocator.free(self.whole);
        testing.allocator.free(self.chunked);
    }
};

/// One prompt whole, then the same prompt as a state-only chunk (`extendState`, as the plugin's `run` does) and a
/// logits chunk, at `depth` layers in flight. `streamed`: the experts hook may refill slabs, so `extendState` drains
/// itself; resident experts (in banks, or a hook that reports every layer resident) leave its last layers queued for
/// the caller.
fn arm(m: *dsv.Model, sc: *Scope, ids: []const u32, cut: usize, depth: usize, streamed: bool) !Arm {
    m.pipe_depth = depth;
    dsv.resetState(m);
    const whole = try bytesOf(sc, try dsv.extendSpan(m, ids, .all, dsv.SPAN));
    errdefer testing.allocator.free(whole);
    try testing.expectEqual(@as(usize, 0), m.pipe_n);

    dsv.resetState(m);
    try dsv.extendState(m, ids[0..cut], dsv.SPAN, null, &.{});
    if (depth == 0 or streamed) {
        try testing.expectEqual(@as(usize, 0), m.pipe_n);
    } else {
        // The pipeline engaged: the encoder's last layers are still queued.
        try testing.expect(m.pipe_n > 0 and m.pipe_n <= depth);
    }
    try dsv.pipeDrain(m);
    try testing.expectEqual(@as(usize, 0), m.pipe_n);
    const chunked = try bytesOf(sc, try dsv.extendSpan(m, ids[cut..], .all, dsv.SPAN));
    try testing.expectEqual(@as(usize, 0), m.pipe_n);
    return .{ .whole = whole, .chunked = chunked };
}

fn expectSameBytes(m: *dsv.Model, sc: *Scope, ids: []const u32, cut: usize, streamed: bool) !void {
    const waited = try arm(m, sc, ids, cut, 0, streamed);
    defer waited.deinit();
    const piped = try arm(m, sc, ids, cut, 2, streamed);
    defer piped.deinit();
    try testing.expectEqualSlices(u8, waited.whole, piped.whole);
    try testing.expectEqualSlices(u8, waited.chunked, piped.chunked);
}

test "sushi_ds41 gpu: pipelined prefill layers give the waiting schedule's logits byte for byte and leave none in flight (DSV41_TINY)" {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    try mlx.requireGpu();
    const dir = try dsv.fixtureDir();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    // The pack only (its reference logits are not read): a fixed prompt over the tiny vocabulary, BOS first.
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = if (i == 0) 2 else 3 + @as(u32, @intCast((i * 37) % 90));
    // Both chunks wider than a decode pass, so both pipeline, and the first wider than the window (`extendState`).
    const cut = ids.len / 2;
    try testing.expect(cut > exl3.kernels.DECODE_ROWS_MAX and ids.len - cut > exl3.kernels.DECODE_ROWS_MAX);

    {
        const m = try dsv.tinyModel(dir, .{ .act = .bfloat16 }, s);
        defer m.deinit();
        try testing.expect(m.ced_split != null and cut > m.cfg.window);
        try expectSameBytes(m, &sc, &ids, cut, false);
    }
    {
        // Streamed experts: 3 slots a layer, so the routes refill slabs, across the chunk boundary too.
        const m = try dsv.tinyModel(dir, .{ .act = .bfloat16, .experts = false }, s);
        defer m.deinit();
        const per_expert = try dstream.expertBytes(&m.cfg);
        var engine = try stream.Engine.initWithOptions(testing.allocator, dir, dstream.geometry(&m.cfg), 3 * m.cfg.layers * per_expert, s, .{
            .layout = .mxfp4_w123,
            .io_workers = 2,
            .bounce_size = 1 << 20,
        });
        defer engine.deinit();
        var st = dstream.Stream{ .engine = &engine, .cfg = &m.cfg, .act = .bfloat16 };
        m.experts = .{ .hook = st.hook() };
        try expectSameBytes(m, &sc, &ids, cut, true);
    }
    {
        // A hook that reports every layer resident (`Hook.all_resident`, as `StqStream` does on the live pack) keeps
        // the cross-chunk overlap: no drain in `extendState`. The cache holds every expert, and this hook evaluates its
        // sum before it returns, so the claim is safe here.
        const m = try dsv.tinyModel(dir, .{ .act = .bfloat16, .experts = false }, s);
        defer m.deinit();
        const per_expert = try dstream.expertBytes(&m.cfg);
        var engine = try stream.Engine.initWithOptions(testing.allocator, dir, dstream.geometry(&m.cfg), m.cfg.experts * m.cfg.layers * per_expert, s, .{
            .layout = .mxfp4_w123,
            .io_workers = 2,
            .bounce_size = 1 << 20,
        });
        defer engine.deinit();
        var st = dstream.Stream{ .engine = &engine, .cfg = &m.cfg, .act = .bfloat16 };
        var hook = st.hook();
        hook.all_resident = everyLayerResident;
        m.experts = .{ .hook = hook };
        try expectSameBytes(m, &sc, &ids, cut, false);
    }
}

fn everyLayerResident(_: *anyopaque) bool {
    return true;
}
