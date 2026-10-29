//! copied_from: src/sdk/blocks/moe/moe_gather_ubench.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Env-gated microbench for the two fused decode MoE kernels at Qwen3.8-Flash-Next's
//! shapes (E 512, top-10, hidden 2560, intermediate 640, 4-bit affine g64, bf16):
//! `gatherQmvGateUp` and `gatherQmvDownReduce`, against MLX's stock `gather_qmm`
//! chains, so a kernel variant is timed in isolation instead of behind a server boot.
//!   MOE_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="moe gather kernels"

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const down_reduce = @import("down_reduce.zig");
const gather_qmv = @import("gather_qmv.zig");
const gather_qmv_rows = @import("gather_qmv_rows.zig");
const io_util = @import("../../util/io_util.zig");

const E: c_int = 512;
const TOPK: c_int = 10;
const H: c_int = 2560;
const I: c_int = 640;
const BITS: u32 = 4;
const GS: u32 = 64;
const WARM = 5;
const BATCHES = 20;
const CALLS_SMALL = 10;
const MAX_CALLS = 40;

const Bank = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,

    fn deinit(b: Bank) void {
        _ = mlx.mlx_array_free(b.w);
        _ = mlx.mlx_array_free(b.sc);
        _ = mlx.mlx_array_free(b.bi);
    }

    /// Bytes one call reads from this bank: TOPK rows of packed words + bf16 scale and bias per group.
    fn bytesPerCall(n: c_int, k: c_int) f64 {
        const words = @divExact(@as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(BITS)), 32.0);
        const groups = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(GS));
        return @as(f64, @floatFromInt(TOPK)) * @as(f64, @floatFromInt(n)) * (words * 4.0 + groups * 4.0);
    }
};

fn randUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

/// Any bit pattern is a valid 4-bit row, so the words come straight from the RNG.
fn affineBank(rnd: std.Random, n: c_int, k: c_int, s: mlx.mlx_stream) !Bank {
    const words: c_int = @intCast(@divExact(@as(u32, @intCast(k)) * BITS, 32));
    const groups: c_int = @divExact(k, @as(c_int, @intCast(GS)));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const w_shape = [_]c_int{ E, n, words };
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &w_shape, 3, 4, key, s));
    const s_shape = [_]c_int{ E, n, groups };
    const sc = try randUniformBf16(rnd, &s_shape, 0.01, 0.06, s);
    const bi = try randUniformBf16(rnd, &s_shape, -0.5, 0.5, s);
    return .{ .w = w, .sc = sc, .bi = bi };
}

fn evalAll(outs: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |o| _ = mlx.mlx_vector_array_append_value(vec, o);
    try mlx.check(mlx.mlx_eval(vec));
}

fn allFinite(a: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    try mlx.check(mlx.mlx_isfinite(&fin, a, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, fin, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var v = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, all));
    return v;
}

const Inputs = struct {
    s: mlx.mlx_stream,
    x: mlx.mlx_array, // [H] bf16
    x_down: mlx.mlx_array, // [TOPK, I] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [TOPK], distinct experts each, cycled by call index
    lhs_zero: mlx.mlx_array, // uint32 [TOPK] zeros: every expert reads the one token
    lhs_iota: mlx.mlx_array, // uint32 [TOPK] 0..9: every expert reads its own row
    scores: mlx.mlx_array, // [TOPK] bf16, sums to ~1
};

const Arm = enum { gateup, downred, pair, stock_gateup, stock_downred };

fn stockGateUp(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x, &.{ 1, 1, H }, 3, s));
    var xg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xg);
    try mlx.check(mlx.mlx_broadcast_to(&xg, x3, &.{ TOPK, 1, H }, 3, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_gather_qmm(&g, xg, in.gate.w, in.gate.sc, in.gate.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_gather_qmm(&u, xg, in.up.w, in.up.sc, in.up.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_multiply(&out, silu, u, s));
    return out;
}

fn stockDownReduce(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x_down, &.{ TOPK, 1, I }, 3, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_gather_qmm(&d, x3, in.down.w, in.down.sc, in.down.bi, in.lhs_iota, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, in.scores, &.{ TOPK, 1, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

/// One call of `arm`. `pair` is the in-situ per-layer chain: gate+up, then down+reduce over
/// the activation it produced, so its two dispatches are dependent like a real layer's.
fn call(in: *const Inputs, arm: Arm, i: usize) !mlx.mlx_array {
    const inds = in.inds[i % CALLS_SMALL];
    switch (arm) {
        .gateup => return (try gather_qmv.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse error.GateUpDeclined,
        .downred => return (try down_reduce.gatherQmvDownReduce(in.s, in.x_down, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined,
        .pair => {
            const act = (try gather_qmv.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse return error.GateUpDeclined;
            defer _ = mlx.mlx_array_free(act);
            return (try down_reduce.gatherQmvDownReduce(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined;
        },
        .stock_gateup => return stockGateUp(in, inds),
        .stock_downred => return stockDownReduce(in, inds),
    }
}

/// Returns the sorted wall times of BATCHES batches of `n` independent calls (distinct index
/// vectors per call, one eval per batch), after WARM warm-up batches.
fn timeBatches(in: *const Inputs, arm: Arm, n: usize, out: *[BATCHES]u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var outs: [MAX_CALLS]mlx.mlx_array = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        for (outs[0..n], 0..) |*o, i| o.* = try call(in, arm, i);
        try evalAll(outs[0..n]);
        if (b >= WARM) out[b - WARM] = sw.read();
        for (outs[0..n]) |o| _ = mlx.mlx_array_free(o);
    }
    std.mem.sort(u64, out, {}, std.sort.asc(u64));
}

/// Prints the per-call time at CALLS_SMALL calls per batch and the MARGINAL per-call time (the
/// slope between CALLS_SMALL and MAX_CALLS calls per batch, which cancels the fixed eval/submit
/// cost each batch pays once), with the GB/s the marginal implies from the bank bytes a call reads.
fn bench(in: *const Inputs, arm: Arm, label: []const u8, bytes_per_call: f64) !void {
    var small: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, CALLS_SMALL, &small);
    var large: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, MAX_CALLS, &large);
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_us = @as(f64, @floatFromInt(small[BATCHES / 2])) / @as(f64, CALLS_SMALL) / 1000.0;
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-ubench] {s:<22} per-call@{d} {d:6.1} us | marginal median {d:6.1} us  min {d:6.1} us  ({d:5.1} MB/call -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        label, CALLS_SMALL, med_us, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// ── Verify-width rows arm ──────────────────────────────────────────────────────────────────────
// A solo verify of S rows reaches the MoE through `gatherQmvGateUpRows` + `gatherQmvDownReduceRows`
// (S <= 8). Bytes are counted over the DISTINCT experts the S rows pick, since rows that share an
// expert read its weights once.

const RowsIn = struct {
    s: mlx.mlx_stream,
    rows: c_int,
    x: mlx.mlx_array, // [S, H] bf16
    x_act: mlx.mlx_array, // [S, TOPK, I] bf16, the activation the down kernel reads
    scores: mlx.mlx_array, // [S, TOPK] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [S, TOPK], TOPK distinct experts per row
    distinct_avg: f64, // mean distinct experts per index table over the cycled tables
};

const RowsArm = enum { gateup, downred, pair, decode_pair_x_rows };

fn rowsCall(in: *const RowsIn, arm: RowsArm, i: usize) !mlx.mlx_array {
    const inds = in.inds[i % CALLS_SMALL];
    switch (arm) {
        .gateup => return (try gather_qmv_rows.gatherQmvGateUpRows(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, null)) orelse error.GateUpRowsDeclined,
        .downred => return (try down_reduce.gatherQmvDownReduceRows(in.s, in.x_act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownRowsDeclined,
        .pair => {
            const act = (try gather_qmv_rows.gatherQmvGateUpRows(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, null)) orelse return error.GateUpRowsDeclined;
            defer _ = mlx.mlx_array_free(act);
            return (try down_reduce.gatherQmvDownReduceRows(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownRowsDeclined;
        },
        // The same S tokens through the one-token decode kernels, one pair per row, as a solo verify would
        // run them without the rows arm. Each row's own [1, TOPK] expert vector is a slice of the table.
        .decode_pair_x_rows => {
            var acc = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(acc);
            var r: c_int = 0;
            while (r < in.rows) : (r += 1) {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_slice(&xr, in.x, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, H }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var x1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x1);
                try mlx.check(mlx.mlx_reshape(&x1, xr, &[_]c_int{H}, 1, in.s));
                var ir = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ir);
                try mlx.check(mlx.mlx_slice(&ir, inds, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, TOPK }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var ind1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ind1);
                try mlx.check(mlx.mlx_reshape(&ind1, ir, &[_]c_int{TOPK}, 1, in.s));
                var sr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sr);
                try mlx.check(mlx.mlx_slice(&sr, in.scores, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, TOPK }, 2, &[_]c_int{ 1, 1 }, 2, in.s));
                var s1 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(s1);
                try mlx.check(mlx.mlx_reshape(&s1, sr, &[_]c_int{TOPK}, 1, in.s));
                const act = (try gather_qmv.gatherQmvGateUp(in.s, x1, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, ind1, BITS, GS, .affine, 0)) orelse return error.GateUpDeclined;
                defer _ = mlx.mlx_array_free(act);
                const d = (try down_reduce.gatherQmvDownReduce(in.s, act, in.down.w, in.down.sc, in.down.bi, ind1, s1, BITS, GS, .affine)) orelse return error.DownReduceDeclined;
                if (r == 0) {
                    _ = mlx.mlx_array_free(acc);
                    acc = d;
                } else {
                    _ = mlx.mlx_array_free(d);
                }
            }
            return acc;
        },
    }
}

fn timeRowsBatches(in: *const RowsIn, arm: RowsArm, n: usize, out: *[BATCHES]u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var outs: [MAX_CALLS]mlx.mlx_array = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        for (outs[0..n], 0..) |*o, i| o.* = try rowsCall(in, arm, i);
        try evalAll(outs[0..n]);
        if (b >= WARM) out[b - WARM] = sw.read();
        for (outs[0..n]) |o| _ = mlx.mlx_array_free(o);
    }
    std.mem.sort(u64, out, {}, std.sort.asc(u64));
}

fn benchRows(in: *const RowsIn, arm: RowsArm, label: []const u8, bytes_per_call: f64) !void {
    var small: [BATCHES]u64 = undefined;
    try timeRowsBatches(in, arm, CALLS_SMALL, &small);
    var large: [BATCHES]u64 = undefined;
    try timeRowsBatches(in, arm, MAX_CALLS, &large);
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-rows-ubench] S={d} {s:<22} marginal median {d:7.1} us  min {d:7.1} us  ({d:5.1} MB distinct -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        in.rows, label, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// The same rows pair, but each call's input depends on the previous call's output (through a zero that the
// graph cannot fold), so the calls cannot overlap: the shape of a forward, where a kernel's ramp-up and tail
// are on the critical path. The three small ops that carry the dependency are the same in every variant.
const ChainMode = enum { pair, gateup, downred, ops };

fn chainBatch(in: *const RowsIn, n: usize, mode: ChainMode) !void {
    var prev = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev);
    var have = false;
    for (0..n) |i| {
        // The array this call's input is perturbed through: x for gate+up and the pair, x_act for down.
        const base = if (mode == .downred) in.x_act else in.x;
        var xk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xk);
        if (have) {
            var sm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sm);
            try mlx.check(mlx.mlx_sum(&sm, prev, false, in.s));
            const zf = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zf);
            var z = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(z);
            try mlx.check(mlx.mlx_multiply(&z, sm, zf, in.s));
            var zb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zb);
            try mlx.check(mlx.mlx_astype(&zb, z, .bfloat16, in.s));
            try mlx.check(mlx.mlx_add(&xk, base, zb, in.s));
        } else {
            try mlx.check(mlx.mlx_array_set(&xk, base));
        }
        const inds = in.inds[i % CALLS_SMALL];
        var out = mlx.mlx_array_new();
        switch (mode) {
            .ops => try mlx.check(mlx.mlx_array_set(&out, xk)),
            .gateup => {
                _ = mlx.mlx_array_free(out);
                out = (try gather_qmv_rows.gatherQmvGateUpRows(in.s, xk, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, null)) orelse return error.GateUpRowsDeclined;
            },
            .downred => {
                _ = mlx.mlx_array_free(out);
                out = (try down_reduce.gatherQmvDownReduceRows(in.s, xk, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse return error.DownRowsDeclined;
            },
            .pair => {
                const act = (try gather_qmv_rows.gatherQmvGateUpRows(in.s, xk, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, null)) orelse return error.GateUpRowsDeclined;
                defer _ = mlx.mlx_array_free(act);
                _ = mlx.mlx_array_free(out);
                out = (try down_reduce.gatherQmvDownReduceRows(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse return error.DownRowsDeclined;
            },
        }
        _ = mlx.mlx_array_free(prev);
        prev = out;
        have = true;
    }
    try mlx.check(mlx.mlx_array_eval(prev));
}

fn benchChain(in: *const RowsIn, mode: ChainMode, label: []const u8, bytes_per_call: f64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var small: [BATCHES]u64 = undefined;
    var large: [BATCHES]u64 = undefined;
    for ([_]usize{ CALLS_SMALL, MAX_CALLS }, [_]*[BATCHES]u64{ &small, &large }) |n, dst| {
        for (0..WARM + BATCHES) |b| {
            var sw = io_util.Stopwatch.init(io);
            try chainBatch(in, n, mode);
            if (b >= WARM) dst[b - WARM] = sw.read();
        }
        std.mem.sort(u64, dst, {}, std.sort.asc(u64));
    }
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-rows-ubench] S={d} {s:<22} marginal median {d:7.1} us  min {d:7.1} us  ({d:5.1} MB distinct -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        in.rows, label, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// A streaming-read reference for the GB/s above: sum over a 2 GiB bf16 array reads every byte once, far
// larger than any cache, so its rate is the DRAM read rate this harness can reach.
// The read ceiling with wide loads: one simdgroup streams its own 64 KiB span with 16-byte loads, UNROLL
// loads in flight per lane, and folds what it read into one word so nothing is dead code. A DRAM rate here
// that beats the reduction above says the reduction was the limit, not the memory.
pub const STREAM_WIDE_SOURCE = kernelSource("kernels/moe/ubench_stream_wide.metal");

/// The stream-wide launch: `nsg` simdgroups over 64 KiB spans, `unroll` loads in flight, grid-stride when `strided`.
pub fn streamWideConfig(nsg: c_int, unroll: c_int, strided: c_int) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const oshape = [_]c_int{nsg};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &oshape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nsg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SPAN4", 4096));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "UNROLL", unroll));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "STRIDED", strided));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", nsg));
    return cfg;
}
