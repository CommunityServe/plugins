//! copied_from: src/plugins/minimax_m3/fused_activation.zig@dc508a5da248397f6c952e75839f1de58aab5be2
//! Decode-sized SwiGLU-OAI, with a rounding point after every original operation.
const std = @import("std");
const sdk = @import("mlx_stream").sdk;
const ops = @import("ops.zig");
const C = @import("geometry.zig").Config;
const sparse = @import("sparse_decode.zig");
var kernel: ?sdk.mlx.mlx_fast_metal_kernel = null;
pub fn supports(g: *ops.G, gate: ops.T, up: ops.T) bool {
    const shape = g.shapeOf(gate);
    if (shape.n < 2 or !std.mem.eql(c_int, shape.slice(), g.shapeOf(up).slice())) return false;
    const dtype = g.dtypeOf(gate);
    if (dtype != g.dtypeOf(up) or (dtype != .bfloat16 and dtype != .float16 and dtype != .float32)) return false;
    var rows: u64 = 1;
    for (shape.slice()[0 .. shape.n - 1]) |dim| {
        if (dim <= 0 or dim > 4) return false;
        rows *= @intCast(dim);
    }
    return rows <= 4 and shape.d[shape.n - 1] > 0 and shape.d[shape.n - 1] <= 12288;
}
pub fn run(g: *ops.G, gate: ops.T, up: ops.T, c: *const C) !ops.T {
    if (!supports(g, gate, up)) return error.MiniMaxActivationShape;
    if (kernel == null) kernel = try sparse.make("minimax_decode_activation", &.{ "gate", "up" }, &.{"out"}, @embedFile("kernels/minimax_activation.metal"));
    const shape = g.shapeOf(gate);
    var n: c_int = 1;
    for (shape.slice()) |dim| n *= dim;
    const params = [_]sparse.Template{
        .{ .name = "N", .value = n },
        .{ .name = "LIMIT_BITS", .value = @bitCast(c.swiglu_limit) },
        .{ .name = "ALPHA_BITS", .value = @bitCast(c.swiglu_alpha) },
        .{ .name = "BETA_BITS", .value = @bitCast(c.swiglu_beta) },
    };
    return (try sparse.launch(g, kernel.?, &.{ try sparse.contiguous(g, gate), try sparse.contiguous(g, up) }, &.{.{ .shape = shape.slice(), .dtype = g.dtypeOf(gate) }}, .{ @divTrunc(n + 127, 128) * 128, 1, 1 }, .{ 128, 1, 1 }, g.dtypeOf(gate), &params))[0];
}
test "MiniMax GPU fused activation preserves intermediate rounding and clipping" {
    if (sdk.mlx.noGpuBackend() or sdk.mlx.virtualGpu()) return error.SkipZigTest;
    const mlx = sdk.mlx;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.G.init(std.testing.allocator, s);
    defer g.deinit();
    const n = 4 * 3072;
    const gates = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(gates);
    const ups = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ups);
    var random: u32 = 0x9238106a;
    for (gates, ups) |*a, *b| {
        random = random *% 1664525 +% 1013904223;
        a.* = @as(f32, @floatFromInt(@as(i32, @intCast(random >> 16)) - 32768)) / 2048;
        random = random *% 1664525 +% 1013904223;
        b.* = @as(f32, @floatFromInt(@as(i32, @intCast(random >> 16)) - 32768)) / 2048;
    }
    const edges = [_]f32{ 0, -0.0, -1, 1, -7, 7, -7.03125, 7.03125, 80, -80, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) };
    for (edges, 0..) |value, i| {
        gates[i] = value;
        ups[edges.len + i] = value;
    }
    const actual = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(actual);
    const expected = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(expected);
    const shapes = [_][3]c_int{ .{ 4, 1, 3072 }, .{ 1, 1, 12288 }, .{ 3, 1, 65 } };
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| for (shapes) |shape| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const size: usize = @intCast(shape[0] * shape[1] * shape[2]);
        const gate = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gates[0..size]), &shape, .float32), dtype);
        const up = try g.astype(try g.hostArray(std.mem.sliceAsBytes(ups[0..size]), &shape, .float32), dtype);
        const config: C = .{};
        const want = try ops.activateReference(&g, gate, up, &config);
        const got = try run(&g, gate, up, &config);
        _ = try g.hostF32(try g.astype(want, .float32), expected[0..size]);
        _ = try g.hostF32(try g.astype(got, .float32), actual[0..size]);
        for (actual[0..size], expected[0..size], 0..) |a, e, i| {
            if (std.math.isNan(e)) {
                try std.testing.expect(std.math.isNan(a));
            } else if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(e))) {
                std.debug.print("activation mismatch dtype={t} index={d}: {d} vs {d}\n", .{ dtype, i, a, e });
                return error.TestExpectedEqual;
            }
        }
    };
}
