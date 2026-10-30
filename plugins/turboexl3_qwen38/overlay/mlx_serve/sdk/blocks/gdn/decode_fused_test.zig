//! copied_from: src/sdk/blocks/gdn/decode_fused_test.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! `decode_fused.zig`'s tests, out of its import graph: a plugin unit that reaches `decode_fused.zig` never compiles them.

const decode_fused = @import("decode_fused.zig");
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const rht = @import("../../quant/hadamard/rht.zig");
const qmv2 = @import("../qmm/qmv2.zig");
const gdn_decode = @import("decode.zig");
const testing = std.testing;
const bf16Scalar = @import("../../../mlx/array.zig").bf16Scalar;
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const sliceLastDim = @import("../../../mlx/array.zig").sliceLastDim;
const attn256MaxDiff = @import("../attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../attention/attn256.zig").attn256RandBf16;
const attn256RandBf16Scaled = @import("../attention/attn256.zig").attn256RandBf16Scaled;
const sliceQkvPart = @import("../linear/fused_rows.zig").sliceQkvPart;
const gatherQmvAct = @import("../moe/gather_qmv.zig").gatherQmvAct;
const moeRouterTopK = @import("../moe/router_kernel.zig").moeRouterTopK;
const gdnGateChain = @import("gate.zig").gdnGateChain;
const LinearAttnWeights = @import("gated_delta_net.zig").LinearAttnWeights;
const getGdnKernel = @import("kernels.zig").getGdnKernel;
const getGdnKernelSeq = @import("kernels.zig").getGdnKernelSeq;
const GdnPreworkArgs = decode_fused.GdnPreworkArgs;
const gdnNormGateFused = decode_fused.gdnNormGateFused;
const gdnParityRand = decode_fused.gdnParityRand;
const gdnPreworkFused = decode_fused.gdnPreworkFused;
const ModelConfig = @import("../../config/model.zig").ModelConfig;
const model_mod = @import("../../config/model.zig");
const SpecTree = @import("../../../engines/mlx/forward_ctx.zig").SpecTree;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmCommitTreePath = @import("../../../engines/mlx/ssm/entry.zig").ssmCommitTreePath;
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;

test "gdn_decode.recurSeqFoldReplay: the fold's outputs, and replaying rows 0..a gives its state after row a and its conv window" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    decode_fused.gdn_decode_fused_override = true;
    defer decode_fused.gdn_decode_fused_override = null;
    for ([_][2]c_int{ .{ 2, 8 }, .{ 16, 48 } }) |geo|
        for ([_][2]mlx.mlx_dtype{ .{ .bfloat16, .bfloat16 }, .{ .bfloat16, .float32 } }) |dts|
            for ([_]bool{ false, true }) |swish| for ([_]c_int{ 1, 2, 3, 4, 5, 8 }) |t| try gdnDecodeFoldReplayCase(s, dts[0], dts[1], geo[0], geo[1], t, swish);
}

fn gdnDecodeFoldReplayCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, hk: c_int, hv: c_int, t_len: c_int, swish: bool) !void {
    var prng = std.Random.DefaultPrng.init(0x4E91 + @as(u64, @intCast(t_len * 131 + hv)));
    const rnd = prng.random();
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = dk, .dv = dv };
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    var arrs: [11]mlx.mlx_array = undefined;
    const shapes = [_][]const c_int{ &.{hv}, &.{hv}, &.{ 1, t_len, c_dim }, &.{ 1, t_len, value_dim }, &.{ 1, t_len, hv }, &.{ 1, t_len, hv }, &.{ 1, 3, c_dim }, &.{ 1, hv, dv, dk }, &.{ c_dim, 4, 1 }, &.{dv}, &.{} };
    const scales = [_]f32{ 1, 1, 1, 4, 16, 16, 1, 1, 1, 1, 0 };
    for (&arrs, shapes, scales, 0..) |*o, sh, sc, i| o.* = if (i == 10) mlx.mlx_array_new_float(1e-6) else try gdnParityRand(rnd, sh, sc, if (i == 7) st else dt, s);
    defer for (arrs) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const in = gdn_decode.Inputs{ .A_log = arrs[0], .dt_bias = arrs[1], .qkv = arrs[2], .z = arrs[3], .b = arrs[4], .a = arrs[5], .conv_state = arrs[6], .ssm_state = arrs[7], .conv_w = arrs[8], .norm_w = arrs[9], .eps = arrs[10], .q_scale = q_scale, .k_scale = k_scale, .signs = .{ .ctx = null } };
    const ref = (try gdn_decode.recurSeqFold(g, t_len, in, swish, s)) orelse {
        if (gdn_decode.foldDeclined(t_len)) return;
        return error.FoldDeclined;
    };
    defer inline for (.{ ref.gated, ref.conv_state, ref.ssm_state, ref.state_seq, ref.conv_input }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const got = (try gdn_decode.recurSeqFoldReplay(g, t_len, in, swish, s)) orelse return error.FoldReplayDeclined;
    defer inline for (.{ got.gated, got.conv_state, got.ssm_state, got.conv_input, got.prework.k, got.prework.v, got.prework.gb }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for ([_][2]mlx.mlx_array{ .{ ref.gated, got.gated }, .{ ref.conv_state, got.conv_state }, .{ ref.ssm_state, got.ssm_state }, .{ ref.conv_input, got.conv_input } }) |pair|
        try testing.expect(try sameBits(pair[0], pair[1], s));
    // Every accept count: the replay commits the capture's state after row a (the last row: the final state) and the
    // conv rows the rollback slices.
    var a: c_int = 0;
    while (a < t_len) : (a += 1) {
        var buf: [gdn_decode.FOLD_MAX_SEQ]i32 = undefined;
        for (buf[0..@intCast(a + 1)], 0..) |*p, i| p.* = @intCast(i);
        const c = (try gdn_decode.replay(g, arrs[7], got.prework, buf[0..@intCast(a + 1)], got.conv_input, .{ 1 + a, 2 + a, 3 + a }, s)) orelse return error.ReplayDeclined;
        defer inline for (.{ c.ssm_state, c.conv_state }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        if (a + 1 == t_len) {
            try mlx.check(mlx.mlx_array_set(&want, ref.ssm_state));
        } else {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, ref.state_seq, &[_]c_int{ a, 0, 0, 0, 0 }, 5, &[_]c_int{ a + 1, 1, hv, dv, dk }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
            try mlx.check(mlx.mlx_reshape(&want, row, &[_]c_int{ 1, hv, dv, dk }, 4, s));
        }
        try testing.expect(try sameBits(want, c.ssm_state, s));
        var win = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(win);
        try mlx.check(mlx.mlx_slice(&win, ref.conv_input, &[_]c_int{ 0, 1 + a, 0 }, 3, &[_]c_int{ 1, 4 + a, c_dim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try testing.expect(try sameBits(win, c.conv_state, s));
    }
}

/// Same dtype, shape and bit patterns (`array_equal` would take -0 for +0 and fail NaN).
fn sameBits(x: mlx.mlx_array, y: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y) or !std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y))) return false;
    const u: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(x)) {
        2 => .uint16,
        4 => .uint32,
        else => .uint8,
    };
    var vx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vx);
    var vy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vy);
    try mlx.check(mlx.mlx_view(&vx, x, u, s));
    try mlx.check(mlx.mlx_view(&vy, y, u, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, vx, vy, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

/// A qwen4_exp GatedDeltaNet layer at test scale (Hk 2, Hv 8, Dk = Dv = 128) whose config is
/// parsed, so its state dtype is the policy's. Hidden 1: a projection row is one product, the
/// same bits at any row count, so only the GDN arms themselves can differ.
const Qwen4GdnRig = struct {
    xfm: Transformer,
    la: LinearAttnWeights,

    const hv: c_int = 8;
    const conv_dim: c_int = 2 * 2 * 128 + hv * 128;

    /// The arms a token can take: serial decode, the verify kernels, batched slots, prefill.
    const Arm = enum { serial, fold, recur_seq, tree, batched_decode, batched_verify, prefill, prefill_composed };

};
