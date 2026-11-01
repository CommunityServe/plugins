//! copied_from: src/plugins/minimax_m3/parity_test.zig@dfd8741b6296d23ebb3c27f172409fcabcb67ed8
//! The plugin against `tests/dump_minimax_m3_fixtures.py` (float64, Hugging Face's MiniMax-M3 text math) on the
//! conformance fixture over a prompt past 16 index blocks, so the sparse block selection decides. Env-gated:
//! `MINIMAX_M3_FIXTURE_DUMP=<dir>` writes the fixture and the prompt; `MINIMAX_M3_REFERENCE=<dir>` compares every
//! row's logits (the serial decode, one token a forward) and the whole-prompt prefill's last row.
const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const parity = @import("../../sdk/testing/parity.zig");
const plugin = @import("plugin.zig");
const fixture = @import("fixture.zig");
const testing = std.testing;
/// 17 blocks of 128 and part of an 18th: past the 16 top-k blocks.
const prompt_len = 2200;
fn promptIds(out: []u32, vocab: u32) void {
    for (out, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % vocab);
}
/// `MINIMAX_M3_PARITY_FIXTURE=decided` picks the parity fixture (`fixture.Conditioning`); unset, the conformance one.
fn conditioning() fixture.Conditioning {
    const raw = std.c.getenv("MINIMAX_M3_PARITY_FIXTURE") orelse return .conformance;
    return if (std.mem.eql(u8, std.mem.span(raw), "decided")) .decided else .conformance;
}
fn writeDump(io: std.Io, a: std.mem.Allocator, dir_path: []const u8) !void {
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, dir_path, .{});
    defer dir.close(io);
    for (try fixture.variantOf(a, 7, .nvfp4, conditioning())) |f| try dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    // The same experts as a BF16 bank, for the converter's cross-check (`--experts bf16`).
    const bf = try fixture.filesOf(a, 7, .bf16);
    try dir.writeFile(io, .{ .sub_path = "experts-bf16.bin", .data = bf[3].bytes });
    try dir.writeFile(io, .{ .sub_path = "expert-manifest-bf16.json", .data = bf[4].bytes });
    var ids: [prompt_len]u32 = undefined;
    promptIds(&ids, 512);
    try dir.writeFile(io, .{ .sub_path = "prompt.json", .data = try std.json.Stringify.valueAlloc(a, .{ .ids = ids }, .{}) });
}
fn cosine(x: []const f32, y: []const f32) f64 {
    var xy: f64 = 0;
    var xx: f64 = 0;
    var yy: f64 = 0;
    for (x, y) |p, q| {
        xy += @as(f64, p) * q;
        xx += @as(f64, p) * p;
        yy += @as(f64, q) * q;
    }
    return xy / @sqrt(xx * yy);
}
fn rmsRatio(x: []const f32, y: []const f32) f64 {
    var xx: f64 = 0;
    var yy: f64 = 0;
    for (x, y) |p, q| {
        xx += @as(f64, p) * p;
        yy += @as(f64, q) * q;
    }
    return @sqrt(xx / yy);
}
/// Query positions past 16 blocks (17 and 32 blocks visible), the boundaries around them, and short contexts.
const selection_positions = [_]u32{ 0, 127, 128, 500, 2047, 2048, 2100, 2175, 2176, 2199, 3000, 4095 };
test "minimax_m3 parity: decode block selection, scalar and Metal, against the reference rule past 16 blocks (MINIMAX_M3_SELECTION_DIR)" {
    const out_dir = std.mem.span(std.c.getenv("MINIMAX_M3_SELECTION_DIR") orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const sdk = @import("mlx_stream").sdk;
    const ops = @import("ops.zig");
    const sparse = @import("sparse_decode.zig");
    const reference = @import("reference.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, out_dir, .{});
    defer dir.close(io);
    const s = sdk.mlx.mlx_default_gpu_stream_new();
    defer _ = sdk.mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    const heads = 4;
    var prng = std.Random.DefaultPrng.init(0x5e1ec7);
    for (selection_positions, 0..) |position, ci| {
        const tokens = position + 1;
        const blocks = (tokens - 1) / 128 + 1;
        const scores = try a.alloc(f32, heads * tokens);
        for (scores) |*v| v.* = prng.random().floatNorm(f32) * 4;
        // Case 3k+2 plants an exact tie between two block maxima (a boundary the reference leaves to topk's order).
        if (ci % 3 == 2 and blocks > 2) for (0..heads) |h| {
            scores[h * tokens + 5] = 50;
            scores[h * tokens + 128 + 5] = 50;
        };
        const mask = try a.alloc(bool, scores.len);
        const scratch = try a.alloc(f32, heads * blocks);
        try reference.selectSparseBlocks(scores, heads, tokens, position, 128, 16, 1, 0, mask, scratch);
        const arr = try g.hostArray(std.mem.sliceAsBytes(scores), &.{ 1, heads, 1, @intCast(tokens) }, .float32);
        const picked = try sparse.select(&g, arr, heads, tokens, position, 16, 1, 0);
        try sdk.mlx.check(sdk.mlx.mlx_array_eval(picked));
        const ids = (sdk.mlx.mlx_array_data_int32(picked) orelse return error.NoData)[0 .. heads * 16];
        var name: [64]u8 = undefined;
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "case{d}.scores.f32", .{ci}), .data = std.mem.sliceAsBytes(scores) });
        const mask_bytes = try a.alloc(u8, mask.len);
        for (mask, mask_bytes) |m, *b| b.* = @intFromBool(m);
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "case{d}.scalar.u8", .{ci}), .data = mask_bytes });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "case{d}.kernel.i32", .{ci}), .data = std.mem.sliceAsBytes(ids) });
        g.reset();
    }
    var meta: std.ArrayList(u8) = .empty;
    try meta.print(a, "{{\"heads\":{d},\"block\":128,\"topk\":16,\"local\":1,\"positions\":[", .{heads});
    for (selection_positions, 0..) |p, i| try meta.print(a, "{s}{d}", .{ if (i == 0) "" else ",", p });
    try meta.appendSlice(a, "]}");
    try dir.writeFile(io, .{ .sub_path = "selection.json", .data = meta.items });
}
test "minimax_m3 parity: expert routing against the reference rule on identical inputs (MINIMAX_M3_ROUTING_DIR)" {
    const out_dir = std.mem.span(std.c.getenv("MINIMAX_M3_ROUTING_DIR") orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const sdk = @import("mlx_stream").sdk;
    const ops = @import("ops.zig");
    const chunk_layer = @import("chunk_layer.zig");
    const geometry = @import("geometry.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = testing.io;
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, out_dir, .{});
    defer dir.close(io);
    const c = try geometry.Config.parse(a, fixture.config_json, null);
    const s = sdk.mlx.mlx_default_gpu_stream_new();
    defer _ = sdk.mlx.mlx_stream_free(s);
    var g = try ops.G.init(a, s);
    defer g.deinit();
    const H = c.hidden_size;
    const E = c.n_routed_experts;
    const rows = 8;
    var prng = std.Random.DefaultPrng.init(0x407e);
    for (0..6) |ci| {
        // Inputs exact in bf16 (multiples of 1/64 under 4), the router and bias f32.
        const x = try a.alloc(f32, rows * H);
        for (x) |*v| v.* = @as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -255, 255))) / 64;
        const router = try a.alloc(f32, E * H);
        for (router) |*v| v.* = prng.random().floatNorm(f32) * 0.02;
        const corr = try a.alloc(f32, E);
        for (corr) |*v| v.* = prng.random().floatNorm(f32) * @as(f32, if (ci % 2 == 0) 0.1 else 1.0);
        const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(x), &.{ rows, @intCast(H) }, .float32), .bfloat16);
        const rt = try g.hostArray(std.mem.sliceAsBytes(router), &.{ @intCast(E), @intCast(H) }, .float32);
        const cr = try g.hostArray(std.mem.sliceAsBytes(corr), &.{@intCast(E)}, .float32);
        const routes = try chunk_layer.routeRows(&g, &c, xb, rt, cr);
        var name: [64]u8 = undefined;
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.x.f32", .{ci}), .data = std.mem.sliceAsBytes(x) });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.router.f32", .{ci}), .data = std.mem.sliceAsBytes(router) });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.corr.f32", .{ci}), .data = std.mem.sliceAsBytes(corr) });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.ids.u16", .{ci}), .data = std.mem.sliceAsBytes(routes.ids[0..routes.n]) });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name, "r{d}.weights.f32", .{ci}), .data = std.mem.sliceAsBytes(routes.weights[0..routes.n]) });
        g.reset();
    }
    try dir.writeFile(io, .{ .sub_path = "routing.json", .data = try std.fmt.allocPrint(a, "{{\"cases\":6,\"rows\":{d},\"hidden\":{d},\"experts\":{d},\"topk\":4,\"scaling\":{d}}}", .{ rows, H, E, c.routed_scaling_factor }) });
}
