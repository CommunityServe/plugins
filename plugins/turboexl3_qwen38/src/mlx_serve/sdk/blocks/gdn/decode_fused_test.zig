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

fn gdnDecodeRowsCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, t_len: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0x0E0 + @as(u64, @intCast(t_len)));
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 8;
    const c_dim: c_int = hk * 128 * 2 + hv * 128;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 };
    const seq = [_][]const c_int{ &.{ 1, t_len, c_dim }, &.{ 1, t_len, hv * 128 }, &.{ 1, t_len, hv }, &.{ 1, t_len, hv } };
    var rows: [4]mlx.mlx_array = undefined;
    for (&rows, seq, [_]f32{ 1, 1, 16, 16 }) |*r, sh, scale| r.* = try gdnParityRand(rnd, sh, scale, dt, s);
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, 128, 128 }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{128}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    var in = gdn_decode.Inputs{
        .qkv = rows[0],
        .z = rows[1],
        .a = rows[2],
        .b = rows[3],
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    };
    const all = (try gdn_decode.recurSeq(g, t_len, in, s)) orelse return error.FusedDeclined;
    defer for ([_]mlx.mlx_array{ all.y, all.conv_state, all.ssm_state, all.state_seq }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var conv = conv_state;
    var state = ssm_state;
    var owned: ?gdn_decode.RecurSeq = null;
    defer if (owned) |o| for ([_]mlx.mlx_array{ o.y, o.conv_state, o.ssm_state, o.state_seq }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var r: c_int = 0;
    while (r < t_len) : (r += 1) {
        var one: [4]mlx.mlx_array = undefined;
        for (&one, rows) |*o, full| {
            const sh = mlx.getShape(full);
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(o, full, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        }
        defer for (one) |o| {
            _ = mlx.mlx_array_free(o);
        };
        in.qkv = one[0];
        in.z = one[1];
        in.a = one[2];
        in.b = one[3];
        in.conv_state = conv;
        in.ssm_state = state;
        const step = (try gdn_decode.recurSeq(g, 1, in, s)) orelse return error.FusedDeclined;
        const ys = mlx.getShape(all.y);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, all.y, &[_]c_int{ 0, r, 0, 0 }, 4, &[_]c_int{ 1, r + 1, ys[2], ys[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want, step.y, s));
        if (owned) |o| for ([_]mlx.mlx_array{ o.y, o.conv_state, o.ssm_state, o.state_seq }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        owned = step;
        conv = step.conv_state;
        state = step.ssm_state;
    }
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(all.ssm_state, state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(all.conv_state, conv, s));
}

fn gdnDecodeFoldParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, hk: c_int, hv: c_int, t_len: c_int, swish: bool) !void {
    var prng = std.Random.DefaultPrng.init(0xF01D + @as(u64, @intCast(t_len * 131 + hv)));
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
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, value_dim }, 4.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const in = gdn_decode.Inputs{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    };

    // Today's verify path: recurSeq, the norm-gate kernel, the conv-input concat.
    const ref = (try gdn_decode.recurSeq(g, t_len, in, s)) orelse return error.FusedDeclined;
    defer {
        inline for (.{ ref.y, ref.conv_state, ref.ssm_state, ref.state_seq }) |a| _ = mlx.mlx_array_free(a);
    }
    const ref_gated = (try gdnNormGateFused(s, ref.y, z, 0, value_dim, norm_w, eps_arr, swish, hv, dv, 1, t_len)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(ref_gated);
    var ref_ci = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_ci);
    {
        const arr = [_]mlx.mlx_array{ conv_state, qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&ref_ci, vec, 1, s));
    }

    const got = (try gdn_decode.recurSeqFold(g, t_len, in, swish, s)) orelse {
        if (gdn_decode.foldDeclined(t_len)) return;
        return error.FoldDeclined;
    };
    defer {
        inline for (.{ got.gated, got.conv_state, got.ssm_state, got.state_seq, got.conv_input }) |a| _ = mlx.mlx_array_free(a);
    }

    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_gated, got.gated, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref.ssm_state, got.ssm_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_ci, got.conv_input, s));
    var ref_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_head);
    var got_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_head);
    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ t_len - 1, 1, hv, dv, dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&ref_head, ref.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try mlx.check(mlx.mlx_slice(&got_head, got.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(ref_head, got_head, s));
}

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

fn gdnDecodeTreeCase(st: mlx.mlx_dtype, nodes: usize) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x7EE);
    const rnd = prng.random();
    const dt: mlx.mlx_dtype = .bfloat16;
    const hk: c_int = 2;
    const hv: c_int = 8;
    const c_dim: c_int = hk * 128 * 2 + hv * 128;
    const g = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 };
    // Rows 1 and 2 follow the root, 3 and 4 follow row 1, 5 follows 2, 6 and 7 follow 3;
    // every prefix is a tree of its own.
    const all_parents = [_]i32{ -1, 0, 0, 1, 1, 2, 3, 3 };
    const all_paths = [_][]const i32{ &.{0}, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 0, 1, 3 }, &.{ 0, 1, 4 }, &.{ 0, 2, 5 }, &.{ 0, 1, 3, 6 }, &.{ 0, 1, 3, 7 } };
    const parents = all_parents[0..nodes];
    const paths = all_paths[0..nodes];
    const w: c_int = @intCast(nodes);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, w, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const a = try gdnParityRand(rnd, &[_]c_int{ 1, w, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try gdnParityRand(rnd, &[_]c_int{ 1, w, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b);
    var fixed: [7]mlx.mlx_array = undefined;
    const fixed_shapes = [_][]const c_int{ &.{ 1, 3, c_dim }, &.{ 1, hv, 128, 128 }, &.{ c_dim, 4, 1 }, &.{hv}, &.{hv}, &.{128}, &.{ 1, w, hv * 128 } };
    for (&fixed, fixed_shapes, 0..) |*f, sh, i| f.* = try gdnParityRand(rnd, sh, 1.0, if (i == 1) st else dt, s);
    defer for (fixed) |f| {
        _ = mlx.mlx_array_free(f);
    };
    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const inputs = struct {
        fn of(q: mlx.mlx_array, aa: mlx.mlx_array, bb: mlx.mlx_array, f: [7]mlx.mlx_array, qs: mlx.mlx_array, e: mlx.mlx_array) gdn_decode.Inputs {
            return .{ .qkv = q, .z = f[6], .a = aa, .b = bb, .conv_state = f[0], .ssm_state = f[1], .conv_w = f[2], .A_log = f[3], .dt_bias = f[4], .q_scale = qs, .k_scale = qs, .norm_w = f[5], .eps = e, .signs = .{ .ctx = null } };
        }
    }.of;
    var table: [6 * all_parents.len]i32 = undefined;
    gdn_decode.treeTable(parents, table[0 .. 6 * nodes]);
    const tab = mlx.mlx_array_new_data(&table, &[_]c_int{6 * w}, 1, .int32);
    defer _ = mlx.mlx_array_free(tab);
    const tree = (try gdn_decode.recurTree(g, w, inputs(qkv, a, b, fixed, q_scale, eps_arr), null, tab, s)) orelse return error.TreeDeclined;
    defer inline for (.{ tree.y, tree.conv_input, tree.prework.k, tree.prework.v, tree.prework.gb }) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // Read in place from a joined [qkv | z | a | b] projection, the same bits.
    {
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        const arr = [_]mlx.mlx_array{ qkv, fixed[6], a, b };
        const vec = mlx.mlx_vector_array_new_data(&arr, arr.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 2, s));
        const a_off = c_dim + hv * 128;
        var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (parts) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try sliceQkvPart(&parts[0], all, 0, c_dim, s);
        try sliceQkvPart(&parts[1], all, a_off, hv, s);
        try sliceQkvPart(&parts[2], all, a_off + hv, hv, s);
        const j = (try gdn_decode.recurTree(g, w, inputs(parts[0], parts[1], parts[2], fixed, q_scale, eps_arr), .{ .all = all, .a_off = a_off, .b_off = a_off + hv }, tab, s)) orelse return error.JoinedDeclined;
        defer inline for (.{ j.y, j.conv_input, j.prework.k, j.prework.v, j.prework.gb }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        inline for (.{ .{ tree.y, j.y }, .{ tree.conv_input, j.conv_input }, .{ tree.prework.k, j.prework.k }, .{ tree.prework.v, j.prework.v }, .{ tree.prework.gb, j.prework.gb } }) |pair|
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], pair[1], s));
    }
    // The commit slices its conv window from [conv_state; window rows].
    {
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        const arr = [_]mlx.mlx_array{ fixed[0], qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&want, vec, 1, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want, tree.conv_input, s));
    }
    for (paths, 0..) |path, node| {
        const idx = mlx.mlx_array_new_data(path.ptr, &[_]c_int{@intCast(path.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var rows: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        const srcs = [_]mlx.mlx_array{ qkv, a, b };
        for (srcs, &rows) |src, *dst| try mlx.check(mlx.mlx_take_axis(dst, src, idx, 1, s));
        const n: c_int = @intCast(path.len);
        var chain_out: [4]mlx.mlx_array = undefined;
        var chain_state: mlx.mlx_array = undefined;
        if (n == 1) {
            const r = (try gdn_decode.recur(g, inputs(rows[0], rows[1], rows[2], fixed, q_scale, eps_arr), s)) orelse return error.ChainDeclined;
            chain_out = .{ r.y, r.conv_state, r.ssm_state, .{ .ctx = null } };
            chain_state = r.ssm_state;
        } else {
            const r = (try gdn_decode.recurSeq(g, n, inputs(rows[0], rows[1], rows[2], fixed, q_scale, eps_arr), s)) orelse return error.ChainDeclined;
            chain_out = .{ r.y, r.conv_state, r.ssm_state, r.state_seq };
            chain_state = r.ssm_state;
        }
        defer for (chain_out) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        // The node's y row equals the chain's last row.
        var want_y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want_y);
        try mlx.check(mlx.mlx_slice(&want_y, chain_out[0], &[_]c_int{ 0, n - 1, 0, 0 }, 4, &[_]c_int{ 1, n, hv, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var got_y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got_y);
        const nd: c_int = @intCast(node);
        try mlx.check(mlx.mlx_slice(&got_y, tree.y, &[_]c_int{ 0, nd, 0, 0 }, 4, &[_]c_int{ 1, nd + 1, hv, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(want_y, got_y, s));
        // Replaying the node's path from the round's input state gives the chain's
        // state, and the conv window its path's last three conv inputs.
        var conv_rows: [3]i32 = undefined;
        for (0..3) |i| {
            const dd: i32 = n - 3 + @as(i32, @intCast(i));
            conv_rows[i] = if (dd < 0) dd + 3 else 3 + path[@intCast(dd)];
        }
        const replayed = (try gdn_decode.replay(g, fixed[1], tree.prework, path, tree.conv_input, conv_rows, s)) orelse return error.ReplayDeclined;
        defer inline for (.{ replayed.ssm_state, replayed.conv_state }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(chain_state, replayed.ssm_state, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(chain_out[1], replayed.conv_state, s));
    }
}

/// A qwen4_exp GatedDeltaNet layer at test scale (Hk 2, Hv 8, Dk = Dv = 128) whose config is
/// parsed, so its state dtype is the policy's. Hidden 1: a projection row is one product, the
/// same bits at any row count, so only the GDN arms themselves can differ.
const Qwen4GdnRig = struct {
    xfm: Transformer,
    la: LinearAttnWeights,

    const hv: c_int = 8;
    const conv_dim: c_int = 2 * 2 * 128 + hv * 128;

    /// In place: the compiled gate closure points at `xfm`. Takes ownership of A_log and dt_bias.
    fn init(rig: *Qwen4GdnRig, A_log: mlx.mlx_array, dt_bias: mlx.mlx_array, rnd: std.Random, s: mlx.mlx_stream) !void {
        const cfg = try model_mod.parseConfigFromJson(testing.allocator,
            \\{"model_type": "qwen4_exp", "text_config": {"hidden_size": 1, "ple_embed_dim": 16,
            \\ "num_hidden_layers": 4, "full_attention_interval": 4, "ple_layer_ids": [1],
            \\ "linear_num_key_heads": 2, "linear_num_value_heads": 8, "linear_key_head_dim": 128,
            \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"}}
        );
        const cache = try KVCache.init(testing.allocator, 0);
        rig.xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache, .config = cfg });
        rig.xfm.compileGdnGate();
        var w: [6]mlx.mlx_array = undefined;
        for (&w, [_][]const c_int{ &.{ 1, conv_dim }, &.{ 1, hv * 128 }, &.{ 1, hv }, &.{ 1, hv }, &.{ conv_dim, 4, 1 }, &.{128} }) |*x, sh| x.* = try attn256RandBf16(rnd, sh, s);
        const none = mlx.mlx_array{ .ctx = null };
        rig.la = .{ .qkv_w = w[0], .qkv_s = none, .qkv_b = none, .z_w = w[1], .z_s = none, .z_b = none, .a_w = w[2], .a_s = none, .a_b = none, .b_w = w[3], .b_s = none, .b_b = none, .conv1d_w = w[4], .A_log = A_log, .dt_bias = dt_bias, .norm_w = w[5], .out_w = none, .out_s = none, .out_b = none };
    }

    fn deinit(rig: *Qwen4GdnRig) void {
        for ([_]mlx.mlx_array{ rig.la.qkv_w, rig.la.z_w, rig.la.a_w, rig.la.b_w, rig.la.conv1d_w, rig.la.A_log, rig.la.dt_bias, rig.la.norm_w }) |a| _ = mlx.mlx_array_free(a);
        if (rig.xfm.compiled_gdn_gate) |c| _ = mlx.mlx_closure_free(c);
        for ([_]?mlx.mlx_array{ rig.xfm.gdn_ones_w, rig.xfm.gdn_q_scale, rig.xfm.gdn_k_scale, rig.xfm.gdn_eps }) |o| if (o) |a| {
            _ = mlx.mlx_array_free(a);
        };
        rig.xfm.cache.deinit();
    }

    /// The arms a token can take: serial decode, the verify kernels, batched slots, prefill.
    const Arm = enum { serial, fold, recur_seq, tree, batched_decode, batched_verify, prefill, prefill_composed };

    /// `x` [1, n, 1] through `arm` from `state` [1, Hv, 128, 128] (cast to the policy's dtype):
    /// the outputs [1, n, Hv*128] and the final state, row 0 of a batched arm.
    fn run(rig: *Qwen4GdnRig, arm: Arm, x: mlx.mlx_array, state: mlx.mlx_array) !struct { y: mlx.mlx_array, state: mlx.mlx_array } {
        const s = rig.xfm.s;
        const n = mlx.getShape(x)[1];
        const rows: c_int = if (arm == .batched_decode or arm == .batched_verify) 2 else 1;
        const width: c_int = switch (arm) {
            .serial, .batched_decode => 1,
            .prefill, .prefill_composed => n,
            else => 8,
        };
        const xs = [2]mlx.mlx_array{ x, x };
        const sts = [2]mlx.mlx_array{ state, state };
        var xb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb);
        var st = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(st);
        inline for (.{ .{ &xb, &xs }, .{ &st, &sts } }) |p| {
            const vec = mlx.mlx_vector_array_new_data(p[1], @intCast(rows));
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(p[0], vec, 0, s));
        }
        var e = SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
        defer inline for (.{ e.conv_state, e.ssm_state, e.spec_state_seq, e.spec_conv_input, e.spec_state_in, e.spec_prework.k, e.spec_prework.v, e.spec_prework.gb }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        try mlx.check(mlx.mlx_zeros(&e.conv_state, &[_]c_int{ rows, 3, conv_dim }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_astype(&e.ssm_state, st, rig.xfm.ssmStateDtype(), s));

        rig.xfm.spec_capture_ssm = switch (arm) {
            .fold, .recur_seq, .tree, .batched_verify => true,
            else => false,
        };
        defer rig.xfm.spec_capture_ssm = false;
        if (arm == .recur_seq) decode_fused.gdn_verify_fold_override = false;
        defer decode_fused.gdn_verify_fold_override = null;
        if (arm == .prefill_composed) decode_fused.gdn_prework_override = false;
        defer decode_fused.gdn_prework_override = null;
        // A chain of 8 as a draft tree, committed whole.
        const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6 };
        var table: [6 * chain.len]i32 = undefined;
        gdn_decode.treeTable(&chain, &table);
        const tab = mlx.mlx_array_new_data(&table, &[_]c_int{table.len}, 1, .int32);
        defer _ = mlx.mlx_array_free(tab);
        const tree = SpecTree{ .parents = tab, .attn = .{ .depth = .{ .ctx = null }, .path = .{ .ctx = null }, .max_depth = 0 } };
        if (arm == .tree) rig.xfm.spec_tree = &tree;
        defer rig.xfm.spec_tree = null;

        const ys = try testing.allocator.alloc(mlx.mlx_array, @intCast(@divExact(n, width)));
        defer testing.allocator.free(ys);
        var n_ys: usize = 0;
        defer for (ys[0..n_ys]) |y| {
            _ = mlx.mlx_array_free(y);
        };
        var t: c_int = 0;
        while (t < n) : (t += width) {
            var xw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xw);
            try mlx.check(mlx.mlx_slice(&xw, xb, &[_]c_int{ 0, t, 0 }, 3, &[_]c_int{ rows, t + width, 1 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            ys[n_ys] = try rig.xfm.gatedDeltaNetProjected(xw, &rig.la, &e, 0, rows, width, arm == .prefill or arm == .prefill_composed, null, true);
            n_ys += 1;
            if (arm == .tree) try ssmCommitTreePath(&e, &[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 }, .{ 8, 9, 10 }, s);
            if (@rem(t, 64) == 0) try mlx.check(mlx.mlx_array_eval(e.ssm_state));
        }
        var y_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y_all);
        const vec = mlx.mlx_vector_array_new_data(ys.ptr, n_ys);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&y_all, vec, 1, s));
        var out: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        for (&out, [_]mlx.mlx_array{ y_all, e.ssm_state }) |*o, a| {
            const sh = mlx.getShape(a);
            var stop: [4]c_int = undefined;
            for (sh, 0..) |d, i| stop[i] = d;
            stop[0] = 1;
            try mlx.check(mlx.mlx_slice(o, a, &[_]c_int{ 0, 0, 0, 0 }, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
        }
        return .{ .y = out[0], .state = out[1] };
    }
};

fn gdnDecodeSeqParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, t_len: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0x5EC0 + @as(u64, @intCast(t_len)));
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 8;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;

    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, value_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, t_len, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);

    // Composed chain: fused prework at seq = T, then the capture recurrence.
    const pre = (try gdnPreworkFused(s, .{
        .qkv = qkv,
        .qkv_off = 0,
        .qkv_stride = c_dim,
        .b = b_in,
        .b_off = 0,
        .b_stride = hv,
        .a = a_in,
        .a_off = 0,
        .a_stride = hv,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .conv_state = conv_state,
        .conv_w = conv_w,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .hk = hk,
        .hv = hv,
        .dk = dk,
        .dv = dv,
        .seq = t_len,
        .g_dtype = gdn_decode.gateDtype(dt, st),
    })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(pre.q);
    defer _ = mlx.mlx_array_free(pre.k);
    defer _ = mlx.mlx_array_free(pre.v);
    defer _ = mlx.mlx_array_free(pre.conv_state);
    defer _ = mlx.mlx_array_free(pre.g);
    defer _ = mlx.mlx_array_free(pre.beta);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, t_len, hv, dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ t_len, 1, hv, dv, dk }, 5, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv));
    const T_scalar = mlx.mlx_array_new_int(t_len);
    defer _ = mlx.mlx_array_free(T_scalar);
    const seq_stride = mlx.mlx_array_new_int(hv * dv * dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const inputs_arr = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.g, pre.beta, ssm_state, T_scalar, seq_stride };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGdnKernelSeq(false), inputs_vec, config, s));
    var y_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_ref);
    var seq_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_ref);
    var state_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(state_ref);
    try mlx.check(mlx.mlx_vector_array_get(&y_ref, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&seq_ref, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&state_ref, outputs_vec, 2));

    const got = (try gdn_decode.recurSeq(.{ .hk = hk, .hv = hv, .dk = dk, .dv = dv }, t_len, .{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = .{ .ctx = null },
    }, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got.y);
    defer _ = mlx.mlx_array_free(got.conv_state);
    defer _ = mlx.mlx_array_free(got.ssm_state);
    defer _ = mlx.mlx_array_free(got.state_seq);

    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(y_ref, got.y, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, got.ssm_state, s));
    // state_seq[T-1] is never written (capture-tail trim); compare the rest.
    var seq_ref_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_ref_head);
    var seq_got_head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seq_got_head);
    const start = [_]c_int{ 0, 0, 0, 0, 0 };
    const stop = [_]c_int{ t_len - 1, 1, hv, dv, dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&seq_ref_head, seq_ref, &start, 5, &stop, 5, &strides, 5, s));
    try mlx.check(mlx.mlx_slice(&seq_got_head, got.state_seq, &start, 5, &stop, 5, &strides, 5, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(seq_ref_head, seq_got_head, s));
}

fn gdnDecodeParityCase(s: mlx.mlx_stream, dt: mlx.mlx_dtype, st: mlx.mlx_dtype, block: c_int) !void {
    var prng = std.Random.DefaultPrng.init(0xB0B5A1);
    const rnd = prng.random();
    const hk: c_int = 2;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const hv: c_int = @divExact(2 * block, dv);
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const value_dim: c_int = hv * dv;

    const q_scale = try scalarOf(1.0 / 128.0, dt, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = try scalarOf(@sqrt(1.0 / 128.0), dt, s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnParityRand(rnd, &[_]c_int{hv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const qkv = try gdnParityRand(rnd, &[_]c_int{ 1, 1, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnParityRand(rnd, &[_]c_int{ 1, 1, value_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(z);
    const b_in = try gdnParityRand(rnd, &[_]c_int{ 1, 1, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(b_in);
    const a_in = try gdnParityRand(rnd, &[_]c_int{ 1, 1, hv }, 16.0, dt, s);
    defer _ = mlx.mlx_array_free(a_in);
    const conv_state = try gdnParityRand(rnd, &[_]c_int{ 1, 3, c_dim }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const ssm_state = try gdnParityRand(rnd, &[_]c_int{ 1, hv, dv, dk }, 1.0, st, s);
    defer _ = mlx.mlx_array_free(ssm_state);
    const conv_w = try gdnParityRand(rnd, &[_]c_int{ c_dim, 4, 1 }, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const norm_w = try gdnParityRand(rnd, &[_]c_int{dv}, 1.0, dt, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    const sg = try testing.allocator.alloc(f32, @intCast(value_dim));
    defer testing.allocator.free(sg);
    for (sg) |*e| e.* = if (rnd.boolean()) 1.0 else -1.0;
    const signs = mlx.mlx_array_new_data(sg.ptr, &[_]c_int{value_dim}, 1, .float32);
    defer _ = mlx.mlx_array_free(signs);

    // Composed chain: the three production kernels + the input rotation.
    const pre = (try gdnPreworkFused(s, .{
        .qkv = qkv,
        .qkv_off = 0,
        .qkv_stride = c_dim,
        .b = b_in,
        .b_off = 0,
        .b_stride = hv,
        .a = a_in,
        .a_off = 0,
        .a_stride = hv,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .conv_state = conv_state,
        .conv_w = conv_w,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .hk = hk,
        .hv = hv,
        .dk = dk,
        .dv = dv,
        .seq = 1,
        .g_dtype = gdn_decode.gateDtype(dt, st),
    })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(pre.q);
    defer _ = mlx.mlx_array_free(pre.k);
    defer _ = mlx.mlx_array_free(pre.v);
    defer _ = mlx.mlx_array_free(pre.conv_state);
    defer _ = mlx.mlx_array_free(pre.g);
    defer _ = mlx.mlx_array_free(pre.beta);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, 1, hv, dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", hv));
    const T_scalar = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(T_scalar);
    const inputs_arr = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.g, pre.beta, ssm_state, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, try getGdnKernel(false), inputs_vec, config, s));
    var y_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_ref);
    var state_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(state_ref);
    try mlx.check(mlx.mlx_vector_array_get(&y_ref, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&state_ref, outputs_vec, 1));
    const flat_ref = (try gdnNormGateFused(s, y_ref, z, 0, value_dim, norm_w, eps_arr, true, hv, dv, 1, 1)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(flat_ref);
    const rot_ref = try rht.transform(flat_ref, signs, block, false, s);
    defer _ = mlx.mlx_array_free(rot_ref);

    const geo = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = dk, .dv = dv };
    const in = gdn_decode.Inputs{
        .qkv = qkv,
        .z = z,
        .a = a_in,
        .b = b_in,
        .conv_state = conv_state,
        .ssm_state = ssm_state,
        .conv_w = conv_w,
        .A_log = A_log,
        .dt_bias = dt_bias,
        .q_scale = q_scale,
        .k_scale = k_scale,
        .norm_w = norm_w,
        .eps = eps_arr,
        .signs = signs,
    };
    const got = (try gdn_decode.step(geo, block, in, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got.rot);
    defer _ = mlx.mlx_array_free(got.conv_state);
    defer _ = mlx.mlx_array_free(got.ssm_state);

    try std.testing.expectEqual(st, mlx.mlx_array_dtype(got.ssm_state));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, got.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, got.ssm_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(rot_ref, got.rot, s));

    // Packs without the rotation run the first kernel alone, then the norm-gate.
    const rec = (try gdn_decode.recur(geo, in, s)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rec.y);
    defer _ = mlx.mlx_array_free(rec.conv_state);
    defer _ = mlx.mlx_array_free(rec.ssm_state);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(y_ref, rec.y, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, rec.conv_state, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(state_ref, rec.ssm_state, s));
}
