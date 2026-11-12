//! turboexl3_ling's KDA byte tests (GPU; `MLX_SERVE_LING_GPU_TESTS=1` runs them, else each skips): each served KDA
//! path of sdk/blocks/kda/ling.zig and plugins/glm5_next/kda.zig against the composed chain or launch it replaces, at
//! Ling-3.1-flash's KDA shape (64 heads of 128, a 4-tap conv over q|k|v, the bounded gate at -5, a bf16 state), word
//! for word on every output. Each case prints a `TURBOEXL3_LING_KDA` line ending in PASS or FAIL and asserts no word
//! differs; nothing is timed.
//! - KDA_GATE_FUSED: `sdk/blocks/kda/ling.zig` `gate` against `kdaGateChain`.
//! - KDA_PREFILL_FUSED: `recurBlocked` against the stock vector-gate kernel (`gdn` kernels.zig's
//!   `getGdnKernel(true)` on its stock grid) at T = 64, 300 and 4096: y and the final state. Then the prompt path's own
//!   call at the same T: kdaGateChain's gate, flat f32 [1, T, hv * dk] as plugins/glm5_next/kda.zig holds it, viewed by
//!   `gateHeads` and passed to `recurBlocked`, against the stock kernel reading the flat gate (a decline fails).
//! - KDA_DECODE_FUSED: `decodeStep` against the layer's composed path (conv1d.run, the q/k rms norms and scales,
//!   kdaGateChain, sigmoid(b), the stock kernel, rms_norm(y) * sigmoid(z)) at 1 to 8 rows: y, the conv window and the
//!   state.
//! - KDA_PREWORK (its table arm, as Ling's layer calls it): `gdnPreworkFused` with `.tab = true` against the composed
//!   prework (q, k, v, the conv window, beta) and `gdnNormGateFusedTab(.., true)` against rms_norm(y) * sigmoid(z), at
//!   1 to 9 rows and at the prompt route's widths.
//! - VERIFY_KDA_ROWS: `verifyRows` over S rows against S serial one-row steps of the composed layer (the path serial
//!   decode takes: Ling's bf16 state stored and loaded between two steps), at 2 to 8 rows: y, the state after every row
//!   (`spec_state_seq`), the window's inputs (`spec_conv_input`), the final window and state.
//! - KDA_EPILOGUE: `decodeStepEpi` and `verifyRowsEpi` against `decodeStep` and `verifyRows` on the same operands.
//! - KDA_REC_SHAPE: `recurBlocked`'s served launch (8 simdgroups a threadgroup, 16-token blocks, 2 value rows a
//!   simdgroup) against the launch it replaced, kept here as the oracle (`oldServedRecur`: kernels/kda/ling_kda_blocked.metal
//!   at 16 simdgroups of one value row, 16-token blocks), at T = 64, 1965 and 4096 (y and the final state; 31 rows
//!   declines to the stock kernel), with a red check: the served kernel planted to sum dk in another lane order must
//!   differ from the oracle.
//! Built by `zig build gpu-test -Dmlx-lib=<the server's staged MLX: libmlxc, libmlx, mlx.metallib>` (build.zig), which
//! links that MLX; the plugin's `zig build test` stays MLX-free.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const ops = @import("mlx_serve/mlx/ops.zig");
const conv1d = @import("mlx_serve/sdk/bases/blocks/conv1d.zig");
const kdaGateChain = @import("mlx_serve/sdk/blocks/kda/gate.zig").kdaGateChain;
const decode_fused = @import("mlx_serve/sdk/blocks/gdn/decode_fused.zig");
const gdn_kernels = @import("mlx_serve/sdk/blocks/gdn/kernels.zig");
const ling_k = @import("mlx_serve/sdk/blocks/kda/ling.zig");
/// turboexl3, for o_proj's Dense in the epilogue test's 5-to-8-row byte check
const t3 = @import("turboexl3");
const kernelSource = @import("mlx_serve/mlx/kernels.zig").source;

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// Ling-3.1-flash's KDA layer
const hk: c_int = 64;
const hv: c_int = 64;
const dk: c_int = 128;
const dv: c_int = 128;
const key_dim: c_int = hk * dk;
const value_dim: c_int = hv * dv;
const conv_dim: c_int = 2 * key_dim + value_dim;
const bound: f32 = -5.0;
const qk_eps: f32 = 1e-6;
const norm_eps: f32 = 1e-6;

fn gpu() !Stream {
    // the GPU tests' own variable (test-only): they run at MLX_SERVE_LING_GPU_TESTS=1
    const asked = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(asked), "1")) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    return s;
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// The arrays a test made, freed at its end.
const Arena = struct {
    items: [512]Array = undefined,
    n: usize = 0,
    fn keep(self: *Arena, a: Array) Array {
        std.debug.assert(self.n < self.items.len);
        self.items[self.n] = a;
        self.n += 1;
        return a;
    }
    fn deinit(self: *Arena) void {
        for (self.items[0..self.n]) |a| free(a);
        self.n = 0;
    }
};

/// normal(loc, scale) in `dt` from the key's seed (MLX's own generator on the GPU), evaluated
fn randn(ar: *Arena, s: Stream, seed: u64, shape: []const c_int, loc: f32, scale: f32, dt: mlx.mlx_dtype) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, loc, scale, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

/// uniform(lo, hi) f32 from the key's seed, evaluated
fn randu(ar: *Arena, s: Stream, seed: u64, shape: []const c_int, lo: f32, hi: f32) !Array {
    var key = mlx.mlx_array_new();
    defer free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const l = mlx.mlx_array_new_float(lo);
    defer free(l);
    const h = mlx.mlx_array_new_float(hi);
    defer free(h);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_uniform(&out, l, h, shape.ptr, shape.len, .float32, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

/// Every word of `a`, as u32 values in row order (16-bit words zero-extended), evaluated
fn words(a: Array, s: Stream) !Array {
    const n: c_int = @intCast(mlx.mlx_array_size(a));
    var v = mlx.mlx_array_new();
    defer free(v);
    switch (mlx.mlx_array_dtype(a)) {
        .float32, .uint32, .int32 => try mlx.check(mlx.mlx_view(&v, a, .uint32, s)),
        .bfloat16, .float16, .uint16 => {
            var v16 = mlx.mlx_array_new();
            defer free(v16);
            try mlx.check(mlx.mlx_view(&v16, a, .uint16, s));
            try mlx.check(mlx.mlx_astype(&v, v16, .uint32, s));
        },
        else => return error.UnsupportedDtype,
    }
    // flat: a view of a row-contiguous `v` (its data pointer at the view's offset), else a row-contiguous copy
    var flat = mlx.mlx_array_new();
    errdefer free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, v, &[_]c_int{n}, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    return flat;
}

const Diff = struct { differing: usize, n: usize };

/// The words in which `got` and `want` differ (shapes and dtypes must match: a mismatch counts every word)
fn diff(got: Array, want: Array, s: Stream) !Diff {
    const n = mlx.mlx_array_size(want);
    if (!std.mem.eql(c_int, mlx.getShape(got), mlx.getShape(want)) or mlx.mlx_array_dtype(got) != mlx.mlx_array_dtype(want)) return .{ .differing = n, .n = n };
    const wg = try words(got, s);
    defer free(wg);
    const ww = try words(want, s);
    defer free(ww);
    const pg = mlx.mlx_array_data_uint32(wg) orelse return error.NoData;
    const pw = mlx.mlx_array_data_uint32(ww) orelse return error.NoData;
    var d: usize = 0;
    for (pg[0..n], pw[0..n]) |x, y| d += @intFromBool(x != y);
    return .{ .differing = d, .n = n };
}

/// Every word of `x` finite (MLX's isfinite, then all), evaluated
fn allFinite(ar: *Arena, x: Array, s: Stream) !bool {
    var fin = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_isfinite(&fin, x, s));
    _ = ar.keep(fin);
    var all = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_all(&all, fin, false, s));
    _ = ar.keep(all);
    try mlx.check(mlx.mlx_array_eval(all));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, all));
    return ok;
}

/// Prints the case's line; returns 1 on a failure
fn report(case: []const u8, what: []const u8, r: Diff) usize {
    const ok = r.differing == 0;
    std.debug.print("TURBOEXL3_LING_KDA {s} {s} differing={d}/{d} {s}\n", .{ case, what, r.differing, r.n, if (ok) "PASS" else "FAIL" });
    return @intFromBool(!ok);
}

fn declined(case: []const u8) usize {
    std.debug.print("TURBOEXL3_LING_KDA {s} declined FAIL\n", .{case});
    return 1;
}

/// A_log per value head (log of 1..16), expanded per key channel as the loader keeps it: f32 [hv * dk]
fn aLogExpanded(ar: *Arena, s: Stream, seed: u64) !Array {
    const u = try randu(ar, s, seed, &[_]c_int{hv}, 1.0, 16.0);
    var lg = mlx.mlx_array_new();
    defer free(lg);
    try mlx.check(mlx.mlx_log(&lg, u, s));
    var col = mlx.mlx_array_new();
    defer free(col);
    try mlx.check(mlx.mlx_reshape(&col, lg, &[_]c_int{ hv, 1 }, 2, s));
    var bc = mlx.mlx_array_new();
    defer free(bc);
    try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ hv, dk }, 2, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, bc, &[_]c_int{hv * dk}, 1, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

/// A 0-dim scalar in `dt` (the layer's q_scale / k_scale)
fn scalarIn(ar: *Arena, v: f32, dt: mlx.mlx_dtype, s: Stream) !Array {
    const raw = mlx.mlx_array_new_float(v);
    defer free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

/// The stock vector-gate recurrence as plugins/glm5_next/kda.zig launches it (no capture): y in `out_dt`, the final
/// state in the state's dtype
fn stockRecur(ar: *Arena, s: Stream, q: Array, k: Array, v: Array, g: Array, beta: Array, state: Array, out_dt: mlx.mlx_dtype) !struct { y: Array, state: Array } {
    const qs = mlx.getShape(q);
    const b = qs[0];
    const t = qs[1];
    const kcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(kcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ b, t, hv, dv }, 4, out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ b, hv, dv, dk }, 4, mlx.mlx_array_dtype(state)));
    try gdn_kernels.gdnSetGrid(kcfg, .stock, dv, hv, b);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "InT", mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "StT", mlx.mlx_array_dtype(state)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "OutT", out_dt));
    inline for (.{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, ta[0], ta[1]));
    }
    const T_scalar = mlx.mlx_array_new_int(t);
    defer free(T_scalar);
    const inputs = [_]Array{ q, k, v, g, beta, state, T_scalar };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, try gdn_kernels.gdnRouteKernel(.stock, 0, true), in_vec, kcfg, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, out_vec, 0));
    var st = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&st, out_vec, 1));
    return .{ .y = ar.keep(y), .state = ar.keep(st) };
}

/// The composed prework of plugins/glm5_next/kda.zig (no fused prework): conv1d.run over the window and the rows,
/// q / k / v split into heads, q and k rms-normed over a weight of ones and scaled, the next window, beta = sigmoid(b)
const Prework = struct { q: Array, k: Array, v: Array, window: Array, beta: Array };
fn composedPrework(ar: *Arena, s: Stream, qkv: Array, conv_state: Array, conv_w: Array, b_in: Array, q_scale: Array, k_scale: Array) !Prework {
    const seq = mlx.getShape(qkv)[1];
    var g = ops.Graph.init(std.heap.c_allocator, s);
    defer g.deinit();
    const conv = try conv1d.run(&g, qkv, conv_state, conv_w, null, 1, conv_dim, 4, true);
    const q_flat = try g.slice(conv.y, &.{ 0, 0, 0 }, &.{ 1, seq, key_dim });
    const k_flat = try g.slice(conv.y, &.{ 0, 0, key_dim }, &.{ 1, seq, key_dim * 2 });
    const v_flat = try g.slice(conv.y, &.{ 0, 0, key_dim * 2 }, &.{ 1, seq, key_dim * 2 + value_dim });
    const q_heads = try g.reshape(q_flat, &.{ 1, seq, hk, dk });
    const k_heads = try g.reshape(k_flat, &.{ 1, seq, hk, dk });
    const v_heads = try g.reshape(v_flat, &.{ 1, seq, hv, dv });
    var ones = mlx.mlx_array_new();
    defer free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{dk}, 1, mlx.mlx_array_dtype(qkv), s));
    var q_norm = mlx.mlx_array_new();
    defer free(q_norm);
    try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, ones, qk_eps, s));
    var q_scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, q_scale, s));
    var k_norm = mlx.mlx_array_new();
    defer free(k_norm);
    try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, ones, qk_eps, s));
    var k_scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, k_scale, s));
    var beta = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&beta, b_in, s));
    var v_copy = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&v_copy, v_heads);
    var w_copy = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&w_copy, conv.window);
    const out: Prework = .{ .q = ar.keep(q_scaled), .k = ar.keep(k_scaled), .v = ar.keep(v_copy), .window = ar.keep(w_copy), .beta = ar.keep(beta) };
    inline for (.{ out.q, out.k, out.v, out.window, out.beta }) |x| try mlx.check(mlx.mlx_array_eval(x));
    return out;
}

/// The composed norm-gate: rms_norm(y, norm_w) * sigmoid(z), flattened to [1, S, hv * dv]
fn composedNormGate(ar: *Arena, s: Stream, y: Array, z: Array, norm_w: Array) !Array {
    const seq = mlx.getShape(y)[1];
    var z_heads = mlx.mlx_array_new();
    defer free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z, &[_]c_int{ 1, seq, hv, dv }, 4, s));
    var normed = mlx.mlx_array_new();
    defer free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, y, norm_w, norm_eps, s));
    var gate = mlx.mlx_array_new();
    defer free(gate);
    try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, s));
    var gated = mlx.mlx_array_new();
    defer free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, normed, gate, s));
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, gated, &[_]c_int{ 1, seq, value_dim }, 3, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    return ar.keep(flat);
}

test "ling kda gpu: the gate in one launch is kdaGateChain's word for word (KDA_GATE_FUSED)" {
    const s = try gpu();
    var ar: Arena = .{};
    defer ar.deinit();
    var failed: usize = 0;
    const A_log = try aLogExpanded(&ar, s, 0x6A7E);
    const dt_bias = try randn(&ar, s, 0x6A7F, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    for ([_]c_int{ 1, 7, 4096 }, 0..) |seq, i| {
        // a wide range, so the sigmoid's two arms and its saturation are all reached
        const a = try randn(&ar, s, 0x6A80 + i, &[_]c_int{ 1, seq, hv * dk }, 0.0, 6.0, .bfloat16);
        const want = ar.keep(try kdaGateChain(A_log, a, dt_bias, bound, s));
        try mlx.check(mlx.mlx_array_eval(want));
        var buf: [32]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "gate S={d}", .{seq}) catch "gate";
        const got = (try ling_k.gate(s, A_log, a, dt_bias, bound)) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(got);
        failed += report(case, "g", try diff(got, want, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: the blocked prompt recurrence is the stock vector-gate kernel's word for word at T 64, 300 and 4096 (KDA_PREFILL_FUSED)" {
    const s = try gpu();
    var failed: usize = 0;
    for ([_]c_int{ 64, 300, 4096 }, 0..) |t, i| {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0xB10C + 16 * @as(u64, i);
        const q = try randn(&ar, s, seed, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
        const k = try randn(&ar, s, seed + 1, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
        const v = try randn(&ar, s, seed + 2, &[_]c_int{ 1, t, hv, dv }, 0.0, 1.0, .bfloat16);
        // the bounded gate's range, (exp(-5), 1)
        const u = try randu(&ar, s, seed + 3, &[_]c_int{ 1, t, hv, dk }, 0.0, 1.0);
        const lb = mlx.mlx_array_new_float(bound);
        defer free(lb);
        var gl = mlx.mlx_array_new();
        defer free(gl);
        try mlx.check(mlx.mlx_multiply(&gl, u, lb, s));
        var g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_exp(&g, gl, s));
        try mlx.check(mlx.mlx_array_eval(g));
        _ = ar.keep(g);
        const bn = try randn(&ar, s, seed + 4, &[_]c_int{ 1, t, hv }, 0.0, 2.0, .bfloat16);
        var beta = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&beta, bn, s));
        try mlx.check(mlx.mlx_array_eval(beta));
        _ = ar.keep(beta);
        const state = try randn(&ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
        const want = try stockRecur(&ar, s, q, k, v, g, beta, state, .bfloat16);
        var buf: [32]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "blocked T={d}", .{t}) catch "blocked";
        const got = (try ling_k.recurBlocked(s, q, k, v, g, beta, state, .bfloat16)) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(got.y);
        _ = ar.keep(got.state);
        failed += report(case, "y", try diff(got.y, want.y, s));
        failed += report(case, "state", try diff(got.state, want.state, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: the prompt path's blocked recurrence takes kdaGateChain's flat gate through gateHeads and is the stock kernel's word for word at T 64, 300 and 4096 (KDA_PREFILL_FUSED)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const A_log = try aLogExpanded(&ar0, s, 0x5E61);
    const dt_bias = try randn(&ar0, s, 0x5E62, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    for ([_]c_int{ 64, 300, 4096 }, 0..) |t, i| {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0x5E70 + 16 * @as(u64, i);
        const q = try randn(&ar, s, seed, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
        const k = try randn(&ar, s, seed + 1, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
        const v = try randn(&ar, s, seed + 2, &[_]c_int{ 1, t, hv, dv }, 0.0, 1.0, .bfloat16);
        // the served gate: kdaGateChain over an `a` projection, flat f32 [1, T, hv * dk], the shape kda.zig holds
        const a = try randn(&ar, s, seed + 3, &[_]c_int{ 1, t, hv * dk }, 0.0, 1.0, .bfloat16);
        const g = ar.keep(try kdaGateChain(A_log, a, dt_bias, bound, s));
        try mlx.check(mlx.mlx_array_eval(g));
        const bn = try randn(&ar, s, seed + 4, &[_]c_int{ 1, t, hv }, 0.0, 2.0, .bfloat16);
        var beta = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&beta, bn, s));
        try mlx.check(mlx.mlx_array_eval(beta));
        _ = ar.keep(beta);
        const state = try randn(&ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
        // the stock kernel reads the flat gate, as the prompt path launches it
        const want = try stockRecur(&ar, s, q, k, v, g, beta, state, .bfloat16);
        var buf: [48]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "served-gate blocked T={d}", .{t}) catch "served-gate blocked";
        // the prompt path's call (plugins/glm5_next/kda.zig): the gate through gateHeads, then recurBlocked
        const g_heads = (try ling_k.gateHeads(s, g, 1, t, hv, dk)) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(g_heads);
        const got = (try ling_k.recurBlocked(s, q, k, v, g_heads, beta, state, .bfloat16)) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(got.y);
        _ = ar.keep(got.state);
        failed += report(case, "y", try diff(got.y, want.y, s));
        failed += report(case, "state", try diff(got.state, want.state, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: the decode step in one launch is the composed layer's word for word at 1 to 8 rows (KDA_DECODE_FUSED)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try randn(&ar0, s, 0xDEC0, &[_]c_int{ conv_dim, 4, 1 }, 0.0, 0.5, .bfloat16);
    const A_log = try aLogExpanded(&ar0, s, 0xDEC1);
    const dt_bias = try randn(&ar0, s, 0xDEC2, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0xDEC3, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    var rows: c_int = 1;
    while (rows <= ling_k.decode_max_rows) : (rows += 1) {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0xDE00 + 16 * @as(u64, @intCast(rows));
        const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, rows, conv_dim }, 0.0, 1.0, .bfloat16);
        const z = try randn(&ar, s, seed + 1, &[_]c_int{ 1, rows, value_dim }, 0.0, 6.0, .bfloat16);
        const a = try randn(&ar, s, seed + 2, &[_]c_int{ 1, rows, hv * dk }, 0.0, 6.0, .bfloat16);
        const b_in = try randn(&ar, s, seed + 3, &[_]c_int{ 1, rows, hv }, 0.0, 6.0, .bfloat16);
        const conv_state = try randn(&ar, s, seed + 4, &[_]c_int{ 1, 3, conv_dim }, 0.0, 1.0, .bfloat16);
        const state = try randn(&ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
        // the composed layer, op for op
        const pre = try composedPrework(&ar, s, qkv, conv_state, conv_w, b_in, q_scale, k_scale);
        const g = ar.keep(try kdaGateChain(A_log, a, dt_bias, bound, s));
        const rec = try stockRecur(&ar, s, pre.q, pre.k, pre.v, g, pre.beta, state, .bfloat16);
        const y_want = try composedNormGate(&ar, s, rec.y, z, norm_w);
        var buf: [32]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "decode S={d}", .{rows}) catch "decode";
        const got = (try ling_k.decodeStep(s, .{
            .qkv = qkv,
            .z = z,
            .a = a,
            .b = b_in,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .state = state,
            .norm_w = norm_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .qk_eps = qk_eps,
            .norm_eps = norm_eps,
            .bound = bound,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
        })) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(got.y);
        _ = ar.keep(got.conv_state);
        _ = ar.keep(got.state);
        failed += report(case, "y", try diff(got.y, y_want, s));
        failed += report(case, "conv_window", try diff(got.conv_state, pre.window, s));
        failed += report(case, "state", try diff(got.state, rec.state, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: the packed prework and the norm-gate on their table arm are the composed chain's word for word at 1 to 9 rows (KDA_PREWORK)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try randn(&ar0, s, 0x7AB0, &[_]c_int{ conv_dim, 4, 1 }, 0.0, 0.5, .bfloat16);
    const A_log = try aLogExpanded(&ar0, s, 0x7AB1);
    const dt_bias = try randn(&ar0, s, 0x7AB2, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0x7AB3, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    const eps_arr = ar0.keep(mlx.mlx_array_new_float(norm_eps));
    var seq: c_int = 1;
    while (seq <= 9) : (seq += 1) {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0x7A00 + 16 * @as(u64, @intCast(seq));
        const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, seq, conv_dim }, 0.0, 1.0, .bfloat16);
        const a = try randn(&ar, s, seed + 1, &[_]c_int{ 1, seq, hv * dk }, 0.0, 6.0, .bfloat16);
        const b_in = try randn(&ar, s, seed + 2, &[_]c_int{ 1, seq, hv }, 0.0, 8.0, .bfloat16);
        const conv_state = try randn(&ar, s, seed + 3, &[_]c_int{ 1, 3, conv_dim }, 0.0, 1.0, .bfloat16);
        const y = try randn(&ar, s, seed + 4, &[_]c_int{ 1, seq, hv, dv }, 0.0, 8.0, .bfloat16);
        const z = try randn(&ar, s, seed + 5, &[_]c_int{ 1, seq, value_dim }, 0.0, 8.0, .bfloat16);
        const want = try composedPrework(&ar, s, qkv, conv_state, conv_w, b_in, q_scale, k_scale);
        var buf: [32]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "prework_tab S={d}", .{seq}) catch "prework_tab";
        // as plugins/glm5_next/kda.zig calls it for Ling's layer: the gate off, the table forced
        if (try decode_fused.gdnPreworkFused(s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .qk_eps = qk_eps,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .batch = 1,
            .gate = false,
            .g_dtype = .bfloat16,
            .tab = true,
        })) |pre| {
            inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta }) |x| _ = ar.keep(x);
            failed += report(case, "q", try diff(pre.q, want.q, s));
            failed += report(case, "k", try diff(pre.k, want.k, s));
            failed += report(case, "v", try diff(pre.v, want.v, s));
            failed += report(case, "conv_window", try diff(pre.conv_state, want.window, s));
            failed += report(case, "beta", try diff(pre.beta, want.beta, s));
        } else failed += declined(case);
        const ng_case = std.fmt.bufPrint(&buf, "normgate_tab S={d}", .{seq}) catch "normgate_tab";
        const ng_want = try composedNormGate(&ar, s, y, z, norm_w);
        if (try decode_fused.gdnNormGateFusedTab(s, y, z, 0, value_dim, norm_w, eps_arr, false, hv, dv, 1, seq, true)) |got| {
            _ = ar.keep(got);
            failed += report(ng_case, "out", try diff(got, ng_want, s));
        } else failed += declined(ng_case);
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: the packed prework and the norm-gate on their table arm are the composed chain's word for word at the prompt route's widths (17 rows up), a cold chunk's zero window included (KDA_PREWORK)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try randn(&ar0, s, 0x7AC0, &[_]c_int{ conv_dim, 4, 1 }, 0.0, 0.5, .bfloat16);
    const A_log = try aLogExpanded(&ar0, s, 0x7AC1);
    const dt_bias = try randn(&ar0, s, 0x7AC2, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0x7AC3, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    const eps_arr = ar0.keep(mlx.mlx_array_new_float(norm_eps));
    // The prompt route's widths (gdnPrefillFusedFor; 10 to 16 rows decline by design and keep the composed chain): the
    // snapshot backoff's span, the standard prompt's tail chunks with and without the backoff, a full chunk, each warm
    // and cold (a cold layer's zero window, the served path's `zero_conv`).
    for ([_]c_int{ 17, 31, 1965, 1996, 4096 }, 0..) |seq, i| {
        for ([_]bool{ false, true }) |cold| {
            var ar: Arena = .{};
            defer ar.deinit();
            const seed: u64 = 0x7B00 + 32 * @as(u64, i) + 16 * @as(u64, @intFromBool(cold));
            const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, seq, conv_dim }, 0.0, 1.0, .bfloat16);
            const a = try randn(&ar, s, seed + 1, &[_]c_int{ 1, seq, hv * dk }, 0.0, 6.0, .bfloat16);
            const b_in = try randn(&ar, s, seed + 2, &[_]c_int{ 1, seq, hv }, 0.0, 8.0, .bfloat16);
            const conv_state = if (cold) blk: {
                var zc = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&zc, &[_]c_int{ 1, 3, conv_dim }, 3, .bfloat16, s));
                try mlx.check(mlx.mlx_array_eval(zc));
                break :blk ar.keep(zc);
            } else try randn(&ar, s, seed + 3, &[_]c_int{ 1, 3, conv_dim }, 0.0, 1.0, .bfloat16);
            const y = try randn(&ar, s, seed + 4, &[_]c_int{ 1, seq, hv, dv }, 0.0, 8.0, .bfloat16);
            const z = try randn(&ar, s, seed + 5, &[_]c_int{ 1, seq, value_dim }, 0.0, 8.0, .bfloat16);
            const want = try composedPrework(&ar, s, qkv, conv_state, conv_w, b_in, q_scale, k_scale);
            var buf: [48]u8 = undefined;
            const case = std.fmt.bufPrint(&buf, "prework_tab S={d}{s}", .{ seq, if (cold) " cold" else "" }) catch "prework_tab";
            if (try decode_fused.gdnPreworkFused(s, .{
                .qkv = qkv,
                .qkv_off = 0,
                .qkv_stride = conv_dim,
                .b = b_in,
                .b_off = 0,
                .b_stride = hv,
                .a = a,
                .a_off = 0,
                .a_stride = hv,
                .A_log = A_log,
                .dt_bias = dt_bias,
                .conv_state = conv_state,
                .conv_w = conv_w,
                .q_scale = q_scale,
                .k_scale = k_scale,
                .qk_eps = qk_eps,
                .hk = hk,
                .hv = hv,
                .dk = dk,
                .dv = dv,
                .seq = seq,
                .batch = 1,
                .gate = false,
                .g_dtype = .bfloat16,
                .tab = true,
            })) |pre| {
                inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta }) |x| _ = ar.keep(x);
                failed += report(case, "q", try diff(pre.q, want.q, s));
                failed += report(case, "k", try diff(pre.k, want.k, s));
                failed += report(case, "v", try diff(pre.v, want.v, s));
                failed += report(case, "conv_window", try diff(pre.conv_state, want.window, s));
                failed += report(case, "beta", try diff(pre.beta, want.beta, s));
            } else failed += declined(case);
            if (cold) continue;
            const ng_case = std.fmt.bufPrint(&buf, "normgate_tab S={d}", .{seq}) catch "normgate_tab";
            const ng_want = try composedNormGate(&ar, s, y, z, norm_w);
            if (try decode_fused.gdnNormGateFusedTab(s, y, z, 0, value_dim, norm_w, eps_arr, false, hv, dv, 1, seq, true)) |got| {
                _ = ar.keep(got);
                failed += report(ng_case, "out", try diff(got, ng_want, s));
            } else failed += declined(ng_case);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// Row `r` of a [1, S, W] array, [1, 1, W]
fn rowOf(ar: *Arena, s: Stream, a: Array, r: c_int) !Array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ sh[0], r + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return ar.keep(out);
}

test "ling kda gpu: an exact verify's 2 to 8 rows in one launch are S serial one-row steps of the composed layer word for word (VERIFY_KDA_ROWS)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try randn(&ar0, s, 0x5E10, &[_]c_int{ conv_dim, 4, 1 }, 0.0, 0.5, .bfloat16);
    const A_log = try aLogExpanded(&ar0, s, 0x5E11);
    const dt_bias = try randn(&ar0, s, 0x5E12, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0x5E13, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    var rows: c_int = 2;
    while (rows <= ling_k.decode_max_rows) : (rows += 1) {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0x5E00 + 16 * @as(u64, @intCast(rows));
        const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, rows, conv_dim }, 0.0, 1.0, .bfloat16);
        const z = try randn(&ar, s, seed + 1, &[_]c_int{ 1, rows, value_dim }, 0.0, 6.0, .bfloat16);
        const a = try randn(&ar, s, seed + 2, &[_]c_int{ 1, rows, hv * dk }, 0.0, 6.0, .bfloat16);
        const b_in = try randn(&ar, s, seed + 3, &[_]c_int{ 1, rows, hv }, 0.0, 6.0, .bfloat16);
        const conv_state = try randn(&ar, s, seed + 4, &[_]c_int{ 1, 3, conv_dim }, 0.0, 1.0, .bfloat16);
        const state = try randn(&ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
        // serial decode, one row a step: the composed layer, its window and its bf16 state carried between steps
        var window = conv_state;
        var st = state;
        var ys: [8]Array = undefined;
        var states: [8]Array = undefined;
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            const pre = try composedPrework(&ar, s, try rowOf(&ar, s, qkv, r), window, conv_w, try rowOf(&ar, s, b_in, r), q_scale, k_scale);
            const g = ar.keep(try kdaGateChain(A_log, try rowOf(&ar, s, a, r), dt_bias, bound, s));
            const rec = try stockRecur(&ar, s, pre.q, pre.k, pre.v, g, pre.beta, st, .bfloat16);
            ys[@intCast(r)] = try composedNormGate(&ar, s, rec.y, try rowOf(&ar, s, z, r), norm_w);
            states[@intCast(r)] = rec.state;
            window = pre.window;
            st = rec.state;
        }
        const n: usize = @intCast(rows);
        var y_want = mlx.mlx_array_new();
        {
            const vec = mlx.mlx_vector_array_new_data(&ys, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&y_want, vec, 1, s));
        }
        _ = ar.keep(y_want);
        var seq_want = mlx.mlx_array_new();
        {
            const vec = mlx.mlx_vector_array_new_data(&states, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_stack_axis(&seq_want, vec, 0, s));
        }
        _ = ar.keep(seq_want);
        var input_want = mlx.mlx_array_new();
        {
            const parts = [_]Array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&input_want, vec, 1, s));
        }
        _ = ar.keep(input_want);
        var buf: [32]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "verify_rows S={d}", .{rows}) catch "verify_rows";
        const got = (try ling_k.verifyRows(s, .{
            .qkv = qkv,
            .z = z,
            .a = a,
            .b = b_in,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .state = state,
            .norm_w = norm_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .qk_eps = qk_eps,
            .norm_eps = norm_eps,
            .bound = bound,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
        })) orelse {
            failed += declined(case);
            continue;
        };
        inline for (.{ got.y, got.conv_state, got.state, got.state_seq, got.conv_input }) |x| _ = ar.keep(x);
        failed += report(case, "y", try diff(got.y, y_want, s));
        failed += report(case, "spec_state_seq", try diff(got.state_seq, seq_want, s));
        failed += report(case, "spec_conv_input", try diff(got.conv_input, input_want, s));
        failed += report(case, "conv_window", try diff(got.conv_state, window, s));
        failed += report(case, "state", try diff(got.state, st, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// The conv weight as the loader makes it from the pack's HF layout (plugins/glm5_next/trunk.zig): `[C, 1, K]` bf16,
/// swapped to `[C, K, 1]`, then `materializedOwnedCopy`'s add of a zero scalar (the swapped view's strides kept)
fn servedConvW(ar: *Arena, s: Stream, seed: u64) !Array {
    const raw = try randn(ar, s, seed, &[_]c_int{ conv_dim, 1, 4 }, 0.0, 0.5, .bfloat16);
    var t = mlx.mlx_array_new();
    defer free(t);
    try mlx.check(mlx.mlx_swapaxes(&t, raw, 1, 2, s));
    var zero = mlx.mlx_array_new();
    defer free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 0, .bfloat16, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, t, zero, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

/// A view of rows [lo, hi) of `a` [B, R, W] (how a conv window arrives: the tail of a concatenation, or a rollback's
/// slice of `spec_conv_input`), evaluated as the view
fn rowsView(ar: *Arena, s: Stream, a: Array, lo: c_int, hi: c_int) !Array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &[_]c_int{ 0, lo, 0 }, 3, &[_]c_int{ sh[0], hi, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return ar.keep(out);
}

test "ling kda gpu, served layout: an exact verify's 2 to 8 rows in one launch are S serial one-row steps over the operands as served (the loader's swapped conv weight, a window and a state that are views at an offset) word for word (VERIFY_KDA_ROWS)" {
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try servedConvW(&ar0, s, 0x6E10);
    const A_log = try aLogExpanded(&ar0, s, 0x6E11);
    const dt_bias = try randn(&ar0, s, 0x6E12, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0x6E13, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    const cw_st = mlx.mlx_array_strides(conv_w);
    std.debug.print("TURBOEXL3_LING_KDA served conv_w strides=({d},{d},{d})\n", .{ cw_st[0], cw_st[1], cw_st[2] });
    var rows: c_int = 2;
    while (rows <= ling_k.decode_max_rows) : (rows += 1) {
        var ar: Arena = .{};
        defer ar.deinit();
        const seed: u64 = 0x6E00 + 16 * @as(u64, @intCast(rows));
        const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, rows, conv_dim }, 0.0, 1.0, .bfloat16);
        const z = try randn(&ar, s, seed + 1, &[_]c_int{ 1, rows, value_dim }, 0.0, 6.0, .bfloat16);
        const a = try randn(&ar, s, seed + 2, &[_]c_int{ 1, rows, hv * dk }, 0.0, 6.0, .bfloat16);
        const b_in = try randn(&ar, s, seed + 3, &[_]c_int{ 1, rows, hv }, 0.0, 6.0, .bfloat16);
        // the window: the last 3 rows of a 5-row input (an offset view, as conv1d.run's tail or a rollback leaves it)
        const prev = try randn(&ar, s, seed + 4, &[_]c_int{ 1, 5, conv_dim }, 0.0, 1.0, .bfloat16);
        const conv_state = try rowsView(&ar, s, prev, 2, 5);
        // the state: row 1 of a stacked [3, 1, hv, dv, dk] sequence, viewed as [1, hv, dv, dk] (a rollback's state)
        const seq = try randn(&ar, s, seed + 5, &[_]c_int{ 3, 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
        var one = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&one, seq, &[_]c_int{ 1, 0, 0, 0, 0 }, 5, &[_]c_int{ 2, 1, hv, dv, dk }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
        _ = ar.keep(one);
        var state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&state, one, &[_]c_int{ 1, hv, dv, dk }, 4, s));
        try mlx.check(mlx.mlx_array_eval(state));
        _ = ar.keep(state);
        // serial decode, one row a step: the composed layer, its window and its bf16 state carried between steps
        var window = conv_state;
        var st = state;
        var ys: [8]Array = undefined;
        var states: [8]Array = undefined;
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            const pre = try composedPrework(&ar, s, try rowOf(&ar, s, qkv, r), window, conv_w, try rowOf(&ar, s, b_in, r), q_scale, k_scale);
            const g = ar.keep(try kdaGateChain(A_log, try rowOf(&ar, s, a, r), dt_bias, bound, s));
            const rec = try stockRecur(&ar, s, pre.q, pre.k, pre.v, g, pre.beta, st, .bfloat16);
            ys[@intCast(r)] = try composedNormGate(&ar, s, rec.y, try rowOf(&ar, s, z, r), norm_w);
            states[@intCast(r)] = rec.state;
            window = pre.window;
            st = rec.state;
        }
        const n: usize = @intCast(rows);
        var y_want = mlx.mlx_array_new();
        {
            const vec = mlx.mlx_vector_array_new_data(&ys, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&y_want, vec, 1, s));
        }
        _ = ar.keep(y_want);
        var seq_want = mlx.mlx_array_new();
        {
            const vec = mlx.mlx_vector_array_new_data(&states, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_stack_axis(&seq_want, vec, 0, s));
        }
        _ = ar.keep(seq_want);
        var input_want = mlx.mlx_array_new();
        {
            const parts = [_]Array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&input_want, vec, 1, s));
        }
        _ = ar.keep(input_want);
        var buf: [40]u8 = undefined;
        const case = std.fmt.bufPrint(&buf, "verify_rows_served S={d}", .{rows}) catch "verify_rows_served";
        const got = (try ling_k.verifyRows(s, .{
            .qkv = qkv,
            .z = z,
            .a = a,
            .b = b_in,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .state = state,
            .norm_w = norm_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .qk_eps = qk_eps,
            .norm_eps = norm_eps,
            .bound = bound,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
        })) orelse {
            failed += declined(case);
            continue;
        };
        inline for (.{ got.y, got.conv_state, got.state, got.state_seq, got.conv_input }) |x| _ = ar.keep(x);
        failed += report(case, "y", try diff(got.y, y_want, s));
        failed += report(case, "spec_state_seq", try diff(got.state_seq, seq_want, s));
        failed += report(case, "spec_conv_input", try diff(got.conv_input, input_want, s));
        failed += report(case, "conv_window", try diff(got.conv_state, window, s));
        failed += report(case, "state", try diff(got.state, st, s));
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// o_proj (kda.o, 8,192 -> 4,096) for the epilogue test's 5-to-8-row byte check: a random K6 trellis
/// [value_dim / 16, 4096 / 16, 96] as the pack's I16 words, `suh` the epilogue's (its prepare reads it), svh random
fn oProjOf(ar: *Arena, s: Stream, suh: Array) !t3.Proj {
    const out_dim: c_int = 4096;
    const pa = std.heap.page_allocator;
    const shape = [_]c_int{ @divExact(value_dim, 16), @divExact(out_dim, 16), 96 };
    const words_u16 = try pa.alloc(u16, @intCast(shape[0] * shape[1] * shape[2]));
    defer pa.free(words_u16);
    var prng = std.Random.DefaultPrng.init(0x0E0F_5A11);
    prng.random().bytes(std.mem.sliceAsBytes(words_u16));
    const u = mlx.mlx_array_new_data(words_u16.ptr, &shape, 3, .uint16);
    defer free(u);
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&w, u, .int16, s));
    try mlx.check(mlx.mlx_array_eval(w));
    _ = ar.keep(w);
    const svh = try randn(ar, s, 0x7E15, &[_]c_int{out_dim}, 0.0, 1.0 / 64.0, .float16);
    return .{ .trellis = w, .suh = suh, .svh = svh };
}

/// o_proj from the step's prepared rows (`got.xp`, `got.xs`, `got.cnt`) against `linear` over its y at `rows` rows, and
/// the check's red: `linear` over y with one word's sign flipped must differ. Returns the failures.
fn oProjFromPrepared(ar: *Arena, s: Stream, case: []const u8, got: anytype, o: t3.Proj, rows: c_int) !usize {
    const d = try t3.denseOf(o, @intCast(value_dim), .bfloat16, .mul1);
    const r: usize = @intCast(rows);
    if (!d.preparedTakes(r) or d.decTakes(r)) {
        std.debug.print("TURBOEXL3_LING_KDA {s} o_proj_from_prepared: turboexl3 takes no chain from prepared rows at {d} rows FAIL\n", .{ case, rows });
        return 1;
    }
    var y2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&y2, got.y, &[_]c_int{ rows, value_dim }, 2, s));
    _ = ar.keep(y2);
    const from_prep = ar.keep(try d.linearFromPrepared(s, got.xp, got.xs, got.cnt, o.trellis, o.svh, r));
    const plain = ar.keep(try d.linear(s, y2, o.trellis, o.suh, o.svh, r));
    var failed = report(case, "o_proj_from_prepared_vs_linear", try diff(from_prep, plain, s));
    // red: y with its word 17 of row 1 negated, through `linear`, gives other words than the prepared rows' o_proj
    const pa = std.heap.page_allocator;
    const sign = try pa.alloc(f32, r * @as(usize, @intCast(value_dim)));
    defer pa.free(sign);
    @memset(sign, 1.0);
    sign[@as(usize, @intCast(value_dim)) + 17] = -1.0;
    const sf = ar.keep(mlx.mlx_array_new_data(sign.ptr, &[_]c_int{ rows, value_dim }, 2, .float32));
    var sb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&sb, sf, mlx.mlx_array_dtype(y2), s));
    _ = ar.keep(sb);
    var bad_y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&bad_y, y2, sb, s));
    _ = ar.keep(bad_y);
    const bad = ar.keep(try d.linear(s, bad_y, o.trellis, o.suh, o.svh, r));
    const dr = try diff(from_prep, bad, s);
    const red_ok = dr.differing > 0;
    std.debug.print("TURBOEXL3_LING_KDA {s} o_proj_from_prepared_red: one word of y negated, through linear, differing={d}/{d} (want > 0) {s}\n", .{ case, dr.differing, dr.n, if (red_ok) "PASS" else "FAIL" });
    failed += @intFromBool(!red_ok);
    return failed;
}

test "ling kda gpu, served layout: the step with the epilogue keeps the step's y, window, state and state after every row, writes the verify's window inputs as the concatenate does, and zeroes o_proj's counters, decode 1 to 8 rows and exact verify 2 to 8 (KDA_EPILOGUE)" {
    // The plugin's own launch (`decodeStepEpi` / `verifyRowsEpi`) against `decodeStep` / `verifyRows` on the same operands:
    // every output the step had keeps its words, the window's inputs are the concatenate's, every counter is 0 and the
    // prepared rows and block sums are finite. o_proj's output from the prepared rows against `linear`'s is turboexl3's
    // models/ling/ling_test.zig's ("gpu ling kda epilogue o_proj", TURBOEXL3_LING_KDA_EPI_OPROJ, 1 to 4 rows); this test
    // holds the plugin's wiring of the epilogue.
    const s = try gpu();
    var failed: usize = 0;
    var ar0: Arena = .{};
    defer ar0.deinit();
    const conv_w = try servedConvW(&ar0, s, 0x7E10);
    const A_log = try aLogExpanded(&ar0, s, 0x7E11);
    const dt_bias = try randn(&ar0, s, 0x7E12, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const norm_w = try randn(&ar0, s, 0x7E13, &[_]c_int{dv}, 1.0, 0.2, .bfloat16);
    const q_scale = try scalarIn(&ar0, 1.0 / @as(f32, @floatFromInt(dk)), .bfloat16, s);
    const k_scale = try scalarIn(&ar0, @sqrt(1.0 / @as(f32, @floatFromInt(dk))), .bfloat16, s);
    const osuh = try randn(&ar0, s, 0x7E14, &[_]c_int{value_dim}, 0.0, 1.0, .float16);
    const oproj = try oProjOf(&ar0, s, osuh);
    const ncnt: c_int = 64;
    for ([_]bool{ false, true }) |exact| {
        var rows: c_int = if (exact) 2 else 1;
        while (rows <= ling_k.decode_max_rows) : (rows += 1) {
            var ar: Arena = .{};
            defer ar.deinit();
            const seed: u64 = 0x7E00 + 16 * @as(u64, @intCast(rows)) + @as(u64, @intFromBool(exact)) * 0x1000;
            const qkv = try randn(&ar, s, seed, &[_]c_int{ 1, rows, conv_dim }, 0.0, 1.0, .bfloat16);
            const z = try randn(&ar, s, seed + 1, &[_]c_int{ 1, rows, value_dim }, 0.0, 6.0, .bfloat16);
            const a = try randn(&ar, s, seed + 2, &[_]c_int{ 1, rows, hv * dk }, 0.0, 6.0, .bfloat16);
            const b_in = try randn(&ar, s, seed + 3, &[_]c_int{ 1, rows, hv }, 0.0, 6.0, .bfloat16);
            const prev = try randn(&ar, s, seed + 4, &[_]c_int{ 1, 5, conv_dim }, 0.0, 1.0, .bfloat16);
            const conv_state = try rowsView(&ar, s, prev, 2, 5);
            const state = try randn(&ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
            const in: ling_k.DecodeIn = .{
                .qkv = qkv,
                .z = z,
                .a = a,
                .b = b_in,
                .conv_state = conv_state,
                .conv_w = conv_w,
                .A_log = A_log,
                .dt_bias = dt_bias,
                .state = state,
                .norm_w = norm_w,
                .q_scale = q_scale,
                .k_scale = k_scale,
                .qk_eps = qk_eps,
                .norm_eps = norm_eps,
                .bound = bound,
                .hk = hk,
                .hv = hv,
                .dk = dk,
                .dv = dv,
            };
            var buf: [40]u8 = undefined;
            const case = std.fmt.bufPrint(&buf, "epilogue_{s} S={d}", .{ if (exact) "verify" else "decode", rows }) catch "epilogue";
            const got = (if (exact) try ling_k.verifyRowsEpi(s, in, .{ .suh = osuh, .ncnt = ncnt }) else try ling_k.decodeStepEpi(s, in, .{ .suh = osuh, .ncnt = ncnt })) orelse {
                failed += declined(case);
                continue;
            };
            inline for (.{ got.y, got.conv_state, got.state, got.state_seq, got.conv_input, got.xp, got.xs, got.cnt }) |x| if (x.ctx != null) {
                _ = ar.keep(x);
            };
            if (exact) {
                const want = (try ling_k.verifyRows(s, in)) orelse {
                    failed += declined(case);
                    continue;
                };
                inline for (.{ want.y, want.conv_state, want.state, want.state_seq, want.conv_input }) |x| _ = ar.keep(x);
                failed += report(case, "y", try diff(got.y, want.y, s));
                failed += report(case, "conv_window", try diff(got.conv_state, want.conv_state, s));
                failed += report(case, "state", try diff(got.state, want.state, s));
                failed += report(case, "spec_state_seq", try diff(got.state_seq, want.state_seq, s));
                failed += report(case, "spec_conv_input", try diff(got.conv_input, want.conv_input, s));
            } else {
                const want = (try ling_k.decodeStep(s, in)) orelse {
                    failed += declined(case);
                    continue;
                };
                inline for (.{ want.y, want.conv_state, want.state }) |x| _ = ar.keep(x);
                failed += report(case, "y", try diff(got.y, want.y, s));
                failed += report(case, "conv_window", try diff(got.conv_state, want.conv_state, s));
                failed += report(case, "state", try diff(got.state, want.state, s));
            }
            // every counter 0 (u32 words against zeros), the prepared rows and block sums finite and shaped
            var zeros = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ncnt}, 1, .uint32, s));
            _ = ar.keep(zeros);
            failed += report(case, "counters_zero", try diff(got.cnt, zeros, s));
            const shapes_ok = std.mem.eql(c_int, mlx.getShape(got.xp), &[_]c_int{ rows, value_dim }) and std.mem.eql(c_int, mlx.getShape(got.xs), &[_]c_int{ rows, @divExact(value_dim, 128) });
            const xp_fin = try allFinite(&ar, got.xp, s);
            const xs_fin = try allFinite(&ar, got.xs, s);
            // the finiteness check's red: the block sums times inf (a planted non-finite value) must be caught
            const inf = ar.keep(mlx.mlx_array_new_float(std.math.inf(f32)));
            var planted = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&planted, got.xs, inf, s));
            _ = ar.keep(planted);
            const red_caught = !(try allFinite(&ar, planted, s));
            const ok = shapes_ok and xp_fin and xs_fin and red_caught;
            std.debug.print("TURBOEXL3_LING_KDA {s} prepared_rows shapes_ok={} xp_finite={} xs_finite={} planted_non_finite_caught={} {s}\n", .{ case, shapes_ok, xp_fin, xs_fin, red_caught, if (ok) "PASS" else "FAIL" });
            failed += @intFromBool(!ok);
            // the review's follow-up e: o_proj from the step's prepared rows at 5 to 8 rows, past DEC-ONE's 4 (the chain's
            // GEMV and finish, `linearFromPrepared`, as src/ling_kda_epi.zig `finish` serves the exact verify there),
            // against `linear` over the step's y, word for word; turboexl3's models/ling/ling_test.zig holds 1 to 4 rows.
            if (rows >= 5) failed += try oProjFromPrepared(&ar, s, case, got, oproj, rows);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

// ── KDA_REC_SHAPE: the blocked recurrence's served launch against the launch it replaced ──

/// The prompt path's operands at T rows, as the served-gate case builds them: q and k scaled bf16 [1, T, hk, 128], v bf16
/// [1, T, hv, 128], kdaGateChain's flat f32 gate viewed per head by `gateHeads`, beta = sigmoid(b) bf16 [1, T, hv], and
/// a bf16 state [1, hv, dv, 128]
const RecIn = struct { q: Array, k: Array, v: Array, g: Array, beta: Array, state: Array };
fn recIn(ar: *Arena, s: Stream, seed: u64, t: c_int) !RecIn {
    const A_log = try aLogExpanded(ar, s, seed + 100);
    const dt_bias = try randn(ar, s, seed + 101, &[_]c_int{hv * dk}, 0.0, 0.5, .float32);
    const q = try randn(ar, s, seed, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
    const k = try randn(ar, s, seed + 1, &[_]c_int{ 1, t, hk, dk }, 0.0, 0.09, .bfloat16);
    const v = try randn(ar, s, seed + 2, &[_]c_int{ 1, t, hv, dv }, 0.0, 1.0, .bfloat16);
    const a = try randn(ar, s, seed + 3, &[_]c_int{ 1, t, hv * dk }, 0.0, 1.0, .bfloat16);
    const g = ar.keep(try kdaGateChain(A_log, a, dt_bias, bound, s));
    try mlx.check(mlx.mlx_array_eval(g));
    const g_heads = (try ling_k.gateHeads(s, g, 1, t, hv, dk)) orelse return error.GateHeadsDeclined;
    _ = ar.keep(g_heads);
    const bn = try randn(ar, s, seed + 4, &[_]c_int{ 1, t, hv }, 0.0, 2.0, .bfloat16);
    var beta = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&beta, bn, s));
    try mlx.check(mlx.mlx_array_eval(beta));
    _ = ar.keep(beta);
    const state = try randn(ar, s, seed + 5, &[_]c_int{ 1, hv, dv, dk }, 0.0, 0.1, .bfloat16);
    return .{ .q = q, .k = k, .v = v, .g = g_heads, .beta = beta, .state = state };
}

/// A kernel over the blocked recurrence's operands (q, k, v, g, beta, state_in, T; y, state_out) from `src`, named
/// `name` (a planted text gets a name of its own)
fn kernelOf(name: [:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const outs = [_][*:0]const u8{ "y", "state_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, iv, ov, src.ptr, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// The launch KDA_REC_SHAPE replaced, kept here as the oracle of the served one (`ling_k.rec_shape`):
/// kernels/kda/ling_kda_blocked.metal (one value row a simdgroup) at 16 simdgroups a threadgroup and 16-token blocks, as
/// sdk/blocks/kda/ling.zig launched it until the strip: kernel ling_kda_blocked_ns16_tb16, grid (32, dv, B * hv),
/// threadgroup (32, 16, 1), the templates InT, StT, OutT, Dk, Dv, Hk, Hv, NS and TB. y and the new state.
const old_served: ling_k.RecShape = .{ .ns = 16, .tb = 16, .rows = 1 };
var old_served_kernel: ?mlx.mlx_fast_metal_kernel = null;
fn oldServedRecur(ar: *Arena, s: Stream, in: RecIn) !struct { y: Array, state: Array } {
    const kern = old_served_kernel orelse blk: {
        const nk = try kernelOf("ling_kda_blocked_ns16_tb16", kernelSource("kernels/kda/ling_kda_blocked.metal"));
        old_served_kernel = nk;
        break :blk nk;
    };
    const qs = mlx.getShape(in.q);
    const b = qs[0];
    const t = qs[1];
    const st_dt = mlx.mlx_array_dtype(in.state);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, hv, dv }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, hv, dv, dk }, 4, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dv, b * hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, old_served.ns, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx.mlx_array_dtype(in.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    inline for (.{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv }, .{ "NS", old_served.ns }, .{ "TB", old_served.tb } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const T_scalar = mlx.mlx_array_new_int(t);
    defer free(T_scalar);
    const inputs = [_]Array{ in.q, in.k, in.v, in.g, in.beta, in.state, T_scalar };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kern, in_vec, cfg, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, out_vec, 0));
    var st = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&st, out_vec, 1));
    return .{ .y = ar.keep(y), .state = ar.keep(st) };
}

fn shapeCase(buf: []u8, what: []const u8, t: c_int, sh: ling_k.RecShape) []const u8 {
    return std.fmt.bufPrint(buf, "{s} T={d} ns={d} tb={d} rows={d}", .{ what, t, sh.ns, sh.tb, sh.rows }) catch what;
}

test "ling kda gpu: KDA_REC_SHAPE: the blocked recurrence's served launch (8 simdgroups a threadgroup, 16-token blocks, 2 value rows a simdgroup) is the launch it replaced (16, 16, 1) word for word at T 64, 1965 and 4096, and 31 rows keeps the stock kernel" {
    const s = try gpu();
    var failed: usize = 0;
    for ([_]c_int{ 64, 1965, 4096 }, 0..) |t, i| {
        var ar: Arena = .{};
        defer ar.deinit();
        const in = try recIn(&ar, s, 0x4EC0 + 64 * @as(u64, i), t);
        const want = try oldServedRecur(&ar, s, in);
        var buf: [96]u8 = undefined;
        const case = shapeCase(&buf, "rec-shape", t, ling_k.rec_shape);
        const got = (try ling_k.recurBlocked(s, in.q, in.k, in.v, in.g, in.beta, in.state, .bfloat16)) orelse {
            failed += declined(case);
            continue;
        };
        _ = ar.keep(got.y);
        _ = ar.keep(got.state);
        failed += report(case, "y", try diff(got.y, want.y, s));
        failed += report(case, "state", try diff(got.state, want.state, s));
    }
    // 31 rows (the standard request's last span): under blocked_min_rows the served launch declines and the stock
    // kernel runs
    {
        var ar: Arena = .{};
        defer ar.deinit();
        const in = try recIn(&ar, s, 0x4EF0, 31);
        const r = try ling_k.recurBlocked(s, in.q, in.k, in.v, in.g, in.beta, in.state, .bfloat16);
        if (r) |x| {
            free(x.y);
            free(x.state);
        }
        const ok = r == null;
        std.debug.print("TURBOEXL3_LING_KDA rec-shape T=31 declined={} (the stock kernel) {s}\n", .{ ok, if (ok) "PASS" else "FAIL" });
        failed += @intFromBool(!ok);
    }
    std.debug.print("TURBOEXL3_LING_KDA_REC_SHAPE_DONE failed={d}\n", .{failed});
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "ling kda gpu: KDA_REC_SHAPE red check: the served kernel planted to sum dk in another lane order, on the served launch, differs from the oracle's words" {
    const s = try gpu();
    var ar: Arena = .{};
    defer ar.deinit();
    const t: c_int = 4096;
    const in = try recIn(&ar, s, 0x4F00, t);
    const want = try oldServedRecur(&ar, s, in);
    // each lane's 4 channels strided across the lanes (channel i * 32 + lane, not lane * 4 + i), so every dk sum groups
    // other terms before its simd_sum: the same recurrence, another reduction order. The byte test must see it.
    const gpa = std.heap.c_allocator;
    const src = kernelSource("kernels/kda/ling_kda_blocked_rows.metal");
    const needle = "auto s_idx = n_per_t * dk_idx + i;";
    const n_sites = std.mem.count(u8, src, needle);
    const planted = try std.mem.replaceOwned(u8, gpa, src, needle, "auto s_idx = i * 32 + dk_idx;");
    defer gpa.free(planted);
    const planted_z = try gpa.dupeSentinel(u8, planted, 0);
    defer gpa.free(planted_z);
    const kern = try kernelOf("ling_kda_blocked_rows_planted_dk_order", planted_z);
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const got = try ling_k.launchBlocked(kern, s, in.q, in.k, in.v, in.g, in.beta, in.state, .bfloat16);
    defer free(got[0]);
    defer free(got[1]);
    const dy = try diff(got[0], want.y, s);
    const ds = try diff(got[1], want.state, s);
    const caught = n_sites == 4 and dy.differing > 0 and ds.differing > 0;
    std.debug.print("TURBOEXL3_LING_KDA_REC_SHAPE red planted-dk-order sites={d} y differing={d}/{d} state differing={d}/{d} {s}\n", .{ n_sites, dy.differing, dy.n, ds.differing, ds.n, if (caught) "PASS" else "FAIL" });
    try std.testing.expect(caught);
}
