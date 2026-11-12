//! The device parity of TURBOEXL3_GLM53F_ROUTER (src/fused_router.zig), run by the build step `router-gpu-test` (linked
//! to the server's MLX) under TURBOEXL3_GLM53F_ROUTER_GPU=1; `zig build test` compiles it and its filter leaves it out.
//!
//! The stock route as the trunk runs it for a streamed GLM-5.3-Flash pack: each router an f16 [288, 4096] (the pack's
//! type; bf16 too) made into the loader's f32 table (a transposed view widened and materialized: trunk.zig `transposed`, `f32Table`), the logits
//! `mlx_matmul(x, table)` (the Linear's dense arm, `qmatmulBits`) and `router_kernel.moeRouterTopK` (sigmoid_bias, top
//! 8, normalized, scale 2.5, bf16 weights); against the plan's phase 1 (`fused_router.partials`, over the raw router as
//! stored) and phase 2 (`fused_router.topK`). For each weight type (f16, bf16) and 2, 3 and 4 rows, `calls` calls each,
//! x seeded normal bf16 and the routers (`routers` of them a type, taken in turn) seeded at a router's scale with seeded
//! f32 correction biases:
//!
//!     TURBOEXL3_GLM53F_ROUTER_BYTES wt=<type> M=<rows> calls=<n> rows=<n * rows> logits_differing=<n>/<total>
//!         inds_differing=<n>/<total> scores_differing=<n>/<total> nonfinite=<n> PASS|FAIL
//!
//! (logits: MLX's matmul against (0 + P0) + P1 in f32, as bits; inds and scores: the routes and their bf16 weights as
//! bits); each type's stock table layout, the one phase 1 reproduces MLX's `nt` kernel for:
//!
//!     TURBOEXL3_GLM53F_ROUTER_LAYOUT wt=<type> strides=<s0>,<s1> PASS|FAIL
//!
//! the calls the plan does not take (1 and 5 rows: `route` routes as before, `partials` refuses):
//!
//!     TURBOEXL3_GLM53F_ROUTER_DECLINE M=<rows> route=<null|routed> partials=<refused|ran> PASS|FAIL
//!
//! and the run's last line, `TURBOEXL3_GLM53F_ROUTER_PARITY lines=<n> fails=<n> PASS|FAIL`.
//!
//! TURBOEXL3_GLM53F_PREDICT's parity (its own test, the same binary and switch): the stock predictor as the bank runs
//! it (glm5n/experts.zig `scoresBf16`: `add(sigmoid(astype(x @ router^T, f32)), correction)`, MLX promoting bf16 x and
//! the f16 router to f32) against `fused_router.predict` for f16 routers (the pack's) at 2, 3 and 4 rows, 512 calls
//! each, x in turn 2-D (the routed call's) and 3-D, the scores compared as f32 bits; a bf16 router's call (the stock
//! matmul stays bf16) and 1 and 5 rows score as before:
//!
//!     TURBOEXL3_GLM53F_PREDICT_BYTES wt=float16 M=<rows> calls=<n> scores_differing=<n>/<total> nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_PREDICT_DECLINE wt=<type> M=<rows> predict=<null|scored> PASS|FAIL
//!     TURBOEXL3_GLM53F_PREDICT_PARITY lines=<n> fails=<n> PASS|FAIL
//!
//! TURBOEXL3_GLM53F_ROUTER_ONE's parity (its own test, the same binary and switch): `fused_router.routeOne` against the
//! stock route for each weight type at 2, 3 and 4 rows and `fused_router.predictOne` against the stock predictor for f16
//! routers at 2, 3 and 4 rows, 512 calls each, built eight at a time and evaluated together (independent calls, each its
//! own arrival counter, the GPU free to run them at once), the routes, weights and scores compared as bits:
//!
//!     TURBOEXL3_GLM53F_ROUTER_ONE_BYTES kind=route wt=<type> M=<rows> calls=<n> inds_differing=<n>/<total>
//!         scores_differing=<n>/<total> nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTER_ONE_BYTES kind=predict wt=float16 M=<rows> calls=<n> scores_differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTER_ONE_PARITY lines=<n> fails=<n> PASS|FAIL
//!
//! TURBOEXL3_GLM53F_ROUTER_ONE1's parity (its own test, the same binary and switch): `fused_router.routeOne1` against
//! the stock one-row route (MLX's matmul of x over the loader's f32 table: its gemv; then `moeRouterTopK`) for each
//! weight type, 512 calls each, x in turn 3-D (a decode call's) and 2-D, built eight at a time and evaluated together,
//! the logits, routes and weights compared as bits; 2 rows refused by the launch, and a one-row call on a layer not
//! registered routing as before:
//!
//!     TURBOEXL3_GLM53F_ROUTER_ONE1_BYTES wt=<type> calls=<n> logits_differing=<n>/<total> inds_differing=<n>/<total>
//!         scores_differing=<n>/<total> nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTER_ONE1_DECLINE M=2 one1=<refused|ran> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTER_ONE1_DECLINE M=1 layer=unregistered route=<null|routed> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTER_ONE1_PARITY lines=<n> fails=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const router_kernel = @import("mlx_serve/sdk/blocks/moe/router_kernel.zig");
const moe = @import("mlx_serve/sdk/bases/blocks/moe.zig");
const fused_router = @import("fused_router.zig");
const Array = mlx.mlx_array;

pub const rows_tested = [_]c_int{ 2, 3, 4 };
/// The routers' stored types: the pack's f16, and bf16.
pub const weight_types = [_]mlx.mlx_dtype{ .float16, .bfloat16 };
pub const calls: usize = 512;
pub const routers: usize = 8;
pub const top_k: c_int = 8;
/// GLM-5.3-Flash's routing (config.json: norm_topk_prob, routed_scaling_factor).
pub const route_norm = true;
pub const route_scale: f32 = 2.5;
pub const declined_rows = [_]c_int{ 1, 5 };
/// The lines a run prints: a byte line a (type, row count), a layout line a type, a decline line a declined row count.
pub const lines = weight_types.len * (rows_tested.len + 1) + declined_rows.len;
/// The predictor test's lines: a byte line a row count for f16 routers, a decline line a row count for bf16 routers
/// (the stock matmul stays bf16), a decline line a declined row count.
pub const predict_lines = 2 * rows_tested.len + declined_rows.len;
/// ROUTER_ONE's test's lines: a route byte line a (type, row count), a predict byte line a row count (f16 routers).
pub const one_lines = weight_types.len * rows_tested.len + rows_tested.len;
/// The calls ROUTER_ONE's test builds before one evaluation of them all.
pub const one_batch: usize = 8;
/// ROUTER_ONE1's test's lines: a byte line a weight type, the two decline lines.
pub const one1_lines = weight_types.len + 2;

fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The loader's f32 table of a router `raw` [288, 4096] as stored: its transposed view widened, then materialized (`x + 0`),
/// evaluated (trunk.zig `transposed` and `f32Table`, array.zig `materializedOwnedCopy`).
fn table(s: mlx.mlx_stream, raw: Array) !Array {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, raw, &[_]c_int{ 1, 0 }, 2, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, t, .float32, s));
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 0, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, f, zero, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

const Count = struct { differing: usize = 0, total: usize = 0, nonfinite: usize = 0 };

/// `got` against `want` as 32- or 16-bit words (both made contiguous: copies of the same bits).
fn countBits(s: mlx.mlx_stream, want: Array, got: Array, c: *Count) !void {
    const dt = mlx.mlx_array_dtype(want);
    if (dt != mlx.mlx_array_dtype(got)) return error.RouterGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.RouterGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    c.total += n;
    switch (dt) {
        .float32 => {
            const pw = (mlx.mlx_array_data_float32(wc) orelse return error.RouterGpuData)[0..n];
            const pg = (mlx.mlx_array_data_float32(gc) orelse return error.RouterGpuData)[0..n];
            for (pw, pg) |a, b| {
                if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) c.differing += 1;
                if (!std.math.isFinite(b)) c.nonfinite += 1;
            }
        },
        .uint32 => {
            const pw = (mlx.mlx_array_data_uint32(wc) orelse return error.RouterGpuData)[0..n];
            const pg = (mlx.mlx_array_data_uint32(gc) orelse return error.RouterGpuData)[0..n];
            for (pw, pg) |a, b| if (a != b) {
                c.differing += 1;
            };
        },
        .bfloat16 => {
            const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.RouterGpuData)[0..n];
            const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.RouterGpuData)[0..n];
            for (pw, pg) |a, b| {
                if (a != b) c.differing += 1;
                if ((b & 0x7F80) == 0x7F80) c.nonfinite += 1;
            }
        },
        else => return error.RouterGpuDtype,
    }
}

/// (0 + P0) + P1 over phase 1's partials `p` [2, rows, 288], as MLX's split-K accumulate sums them.
fn summed(s: mlx.mlx_stream, p: Array, rows: c_int) ![2]Array {
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    errdefer for (parts) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&parts, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        const lo: c_int = @intCast(i);
        try mlx.check(mlx.mlx_slice(o, p, &[_]c_int{ lo, 0, 0 }, 3, &[_]c_int{ lo + 1, rows, fused_router.experts }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    }
    return parts;
}

fn accumulate(s: mlx.mlx_stream, parts: [2]Array, rows: c_int) !Array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{ 1, rows, fused_router.experts }, 3, .float32, s));
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_add(&a, zero, parts[0], s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, a, parts[1], s));
    return out;
}

test "turboexl3_glm53f router gpu: TURBOEXL3_GLM53F_ROUTER's two launches give the stock route's logits, routes and weights bit for bit at 2, 3 and 4 rows, 512 calls each, and route nothing at 1 or 5 rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_ROUTER_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261107;
    var fails: usize = 0;
    // the routers of the last type, for the declines
    var last_raw: Array = .{ .ctx = null };
    defer if (last_raw.ctx != null) {
        _ = mlx.mlx_array_free(last_raw);
    };
    for (weight_types) |wt| {
        // the routers: raw [288, 4096] of the type at about a router's scale, the loader's f32 tables, the correction biases
        var raws: [routers]Array = @splat(.{ .ctx = null });
        var tables: [routers]Array = @splat(.{ .ctx = null });
        var biases: [routers]Array = @splat(.{ .ctx = null });
        defer for ([_][]Array{ &raws, &tables, &biases }) |set| for (set) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (0..routers) |i| {
            raws[i] = try normal(s, &seed, &.{ fused_router.experts, fused_router.hidden }, wt, 1.0 / 64.0);
            tables[i] = try table(s, raws[i]);
            biases[i] = try normal(s, &seed, &.{fused_router.experts}, .float32, 0.1);
        }
        // the stock table's layout: the transposed view MLX's matmul reads as transpose_b
        {
            const st = mlx.mlx_array_strides(tables[0]);
            const ok = fused_router.tableLayout(mlx.getShape(tables[0]), st[0..2]);
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_ROUTER_LAYOUT wt={s} strides={d},{d} {s}\n", .{ @tagName(wt), st[0], st[1], if (ok) "PASS" else "FAIL" });
        }
        for (rows_tested) |rows| {
            var lg: Count = .{};
            var ind: Count = .{};
            var sco: Count = .{};
            for (0..calls) |c| {
                const r = c % routers;
                const x = try normal(s, &seed, &.{ 1, rows, fused_router.hidden }, .bfloat16, 1.0);
                defer _ = mlx.mlx_array_free(x);
                // the stock route
                var logits = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(logits);
                try mlx.check(mlx.mlx_matmul(&logits, x, tables[r], s));
                const want = (try router_kernel.moeRouterTopK(s, logits, biases[r], top_k, .sigmoid_bias, route_norm, route_scale, .bfloat16, 0, 0, null)) orelse return error.RouterGpuStockDeclined;
                defer {
                    _ = mlx.mlx_array_free(want.inds);
                    _ = mlx.mlx_array_free(want.norm_scores);
                }
                // the plan
                const p = try fused_router.partials(s, x, raws[r], rows);
                defer _ = mlx.mlx_array_free(p);
                const got = try fused_router.topK(s, p, mlx.getShape(x), biases[r], top_k, route_norm, route_scale);
                defer {
                    _ = mlx.mlx_array_free(got.inds);
                    _ = mlx.mlx_array_free(got.norm_scores);
                }
                const parts = try summed(s, p, rows);
                defer for (parts) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const sum = try accumulate(s, parts, rows);
                defer _ = mlx.mlx_array_free(sum);
                try countBits(s, logits, sum, &lg);
                try countBits(s, want.inds, got.inds, &ind);
                try countBits(s, want.norm_scores, got.norm_scores, &sco);
            }
            const ok = lg.differing == 0 and ind.differing == 0 and sco.differing == 0 and lg.nonfinite == 0 and sco.nonfinite == 0 and lg.total == calls * @as(usize, @intCast(rows * fused_router.experts)) and ind.total == calls * @as(usize, @intCast(rows * top_k));
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_ROUTER_BYTES wt={s} M={d} calls={d} rows={d} logits_differing={d}/{d} inds_differing={d}/{d} scores_differing={d}/{d} nonfinite={d} {s}\n", .{ @tagName(wt), rows, calls, calls * @as(usize, @intCast(rows)), lg.differing, lg.total, ind.differing, ind.total, sco.differing, sco.total, lg.nonfinite + sco.nonfinite, if (ok) "PASS" else "FAIL" });
        }
        if (last_raw.ctx != null) _ = mlx.mlx_array_free(last_raw);
        last_raw = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&last_raw, raws[0]);
    }
    // the calls the plan does not take: `route` returns before it reads the layer (any pointer), `partials` refuses
    {
        const was = fused_router.on;
        fused_router.on = true;
        defer fused_router.on = was;
        const stand_in: moe.Moe.Weights = undefined;
        for (declined_rows) |rows| {
            const x = try normal(s, &seed, &.{ 1, rows, fused_router.hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x);
            const before = fused_router.other_rows;
            const routed = try fused_router.route(s, &stand_in, x);
            if (routed) |r| {
                _ = mlx.mlx_array_free(r.inds);
                _ = mlx.mlx_array_free(r.norm_scores);
            }
            const refused = if (fused_router.partials(s, x, last_raw, rows)) |p| blk: {
                _ = mlx.mlx_array_free(p);
                break :blk false;
            } else |_| true;
            const ok = routed == null and refused and fused_router.other_rows == before + 1;
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_ROUTER_DECLINE M={d} route={s} partials={s} {s}\n", .{ rows, if (routed == null) "null" else "routed", if (refused) "refused" else "ran", if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_GLM53F_ROUTER_PARITY lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}

/// The stock predictor as the bank runs it (glm5n/experts.zig `scoresBf16`, MlxOps' ops: transpose reverses every
/// axis): f32 of x's leading dimensions by 288.
fn stockPredict(s: mlx.mlx_stream, x: Array, raw: Array, correction: Array) !Array {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose(&t, raw, s));
    var lg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lg);
    try mlx.check(mlx.mlx_matmul(&lg, x, t, s));
    var l32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l32);
    try mlx.check(mlx.mlx_astype(&l32, lg, .float32, s));
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, l32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, sg, correction, s));
    return out;
}

test "turboexl3_glm53f router gpu: TURBOEXL3_GLM53F_PREDICT's two launches give the stock predictor's scores bit for bit at 2, 3 and 4 rows, 512 calls each, and score nothing at 1 or 5 rows" {
    if (std.c.getenv("TURBOEXL3_GLM53F_ROUTER_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261108;
    var fails: usize = 0;
    const was = fused_router.predict_on;
    fused_router.predict_on = true;
    defer fused_router.predict_on = was;
    // an f16 router and correction, for the declines of 1 and 5 rows
    var last_raw: Array = .{ .ctx = null };
    var last_corr: Array = .{ .ctx = null };
    defer for ([_]Array{ last_raw, last_corr }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (weight_types) |wt| {
        var raws: [routers]Array = @splat(.{ .ctx = null });
        var corrs: [routers]Array = @splat(.{ .ctx = null });
        defer for ([_][]Array{ &raws, &corrs }) |set| for (set) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (0..routers) |i| {
            raws[i] = try normal(s, &seed, &.{ fused_router.experts, fused_router.hidden }, wt, 1.0 / 64.0);
            corrs[i] = try normal(s, &seed, &.{fused_router.experts}, .float32, 0.1);
        }
        // a bf16 router: the stock matmul stays bf16, so the plan declines and the caller scores as before
        if (wt != .float16) {
            for (rows_tested) |rows| {
                const x = try normal(s, &seed, &.{ rows, fused_router.hidden }, .bfloat16, 1.0);
                defer _ = mlx.mlx_array_free(x);
                const before = fused_router.predict_declined;
                const got = try fused_router.predict(s, x, raws[0], corrs[0]);
                if (got) |a| _ = mlx.mlx_array_free(a);
                const ok = got == null and fused_router.predict_declined == before + 1;
                if (!ok) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_PREDICT_DECLINE wt={s} M={d} predict={s} {s}\n", .{ @tagName(wt), rows, if (got == null) "null" else "scored", if (ok) "PASS" else "FAIL" });
            }
            continue;
        }
        for (rows_tested) |rows| {
            var sc: Count = .{};
            for (0..calls) |c| {
                const r = c % routers;
                // x as the routed call's ([rows, hidden]) and, in turn, with a leading batch of 1
                const x = if (c % 2 == 0)
                    try normal(s, &seed, &.{ rows, fused_router.hidden }, .bfloat16, 1.0)
                else
                    try normal(s, &seed, &.{ 1, rows, fused_router.hidden }, .bfloat16, 1.0);
                defer _ = mlx.mlx_array_free(x);
                const want = try stockPredict(s, x, raws[r], corrs[r]);
                defer _ = mlx.mlx_array_free(want);
                const got = (try fused_router.predict(s, x, raws[r], corrs[r])) orelse return error.RouterGpuPredictDeclined;
                defer _ = mlx.mlx_array_free(got);
                try countBits(s, want, got, &sc);
            }
            const ok = sc.differing == 0 and sc.nonfinite == 0 and sc.total == calls * @as(usize, @intCast(rows * fused_router.experts));
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_PREDICT_BYTES wt={s} M={d} calls={d} scores_differing={d}/{d} nonfinite={d} {s}\n", .{ @tagName(wt), rows, calls, sc.differing, sc.total, sc.nonfinite, if (ok) "PASS" else "FAIL" });
        }
        for ([_]*Array{ &last_raw, &last_corr }, [_]Array{ raws[0], corrs[0] }) |dst, src| {
            if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
            dst.* = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(dst, src);
        }
    }
    // the calls the plan does not take: 1 and 5 rows score as before
    for (declined_rows) |rows| {
        const x = try normal(s, &seed, &.{ rows, fused_router.hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x);
        const before = fused_router.predict_other;
        const got = try fused_router.predict(s, x, last_raw, last_corr);
        if (got) |a| _ = mlx.mlx_array_free(a);
        const ok = got == null and fused_router.predict_other == before + 1;
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_PREDICT_DECLINE wt=float16 M={d} predict={s} {s}\n", .{ rows, if (got == null) "null" else "scored", if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_GLM53F_PREDICT_PARITY lines={d} fails={d} {s}\n", .{ predict_lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}

fn freeAll(arrs: []Array) void {
    for (arrs) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
}

test "turboexl3_glm53f router gpu: TURBOEXL3_GLM53F_ROUTER_ONE's one launch gives the stock route's routes and weights and the stock predictor's scores bit for bit at 2, 3 and 4 rows, 512 calls each, eight calls evaluated together" {
    if (std.c.getenv("TURBOEXL3_GLM53F_ROUTER_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261109;
    var fails: usize = 0;
    comptime std.debug.assert(calls % one_batch == 0);
    for (weight_types) |wt| {
        var raws: [routers]Array = @splat(.{ .ctx = null });
        var tables: [routers]Array = @splat(.{ .ctx = null });
        var biases: [routers]Array = @splat(.{ .ctx = null });
        var corrs: [routers]Array = @splat(.{ .ctx = null });
        defer for ([_][]Array{ &raws, &tables, &biases, &corrs }) |set| freeAll(set);
        for (0..routers) |i| {
            raws[i] = try normal(s, &seed, &.{ fused_router.experts, fused_router.hidden }, wt, 1.0 / 64.0);
            tables[i] = try table(s, raws[i]);
            biases[i] = try normal(s, &seed, &.{fused_router.experts}, .float32, 0.1);
            corrs[i] = try normal(s, &seed, &.{fused_router.experts}, .float32, 0.1);
        }
        for (rows_tested) |rows| {
            var ind: Count = .{};
            var sco: Count = .{};
            var c: usize = 0;
            while (c < calls) : (c += one_batch) {
                var xs: [one_batch]Array = @splat(.{ .ctx = null });
                var want_i: [one_batch]Array = @splat(.{ .ctx = null });
                var want_s: [one_batch]Array = @splat(.{ .ctx = null });
                var got_i: [one_batch]Array = @splat(.{ .ctx = null });
                var got_s: [one_batch]Array = @splat(.{ .ctx = null });
                defer for ([_][]Array{ &xs, &want_i, &want_s, &got_i, &got_s }) |set| freeAll(set);
                for (0..one_batch) |b| {
                    const r = (c + b) % routers;
                    xs[b] = try normal(s, &seed, &.{ 1, rows, fused_router.hidden }, .bfloat16, 1.0);
                    // the stock route
                    var logits = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(logits);
                    try mlx.check(mlx.mlx_matmul(&logits, xs[b], tables[r], s));
                    const want = (try router_kernel.moeRouterTopK(s, logits, biases[r], top_k, .sigmoid_bias, route_norm, route_scale, .bfloat16, 0, 0, null)) orelse return error.RouterGpuStockDeclined;
                    want_i[b] = want.inds;
                    want_s[b] = want.norm_scores;
                    // the one launch
                    const got = try fused_router.routeOne(s, xs[b], raws[r], biases[r], top_k, route_norm, route_scale);
                    got_i[b] = got.inds;
                    got_s[b] = got.norm_scores;
                }
                // the batch's one launches evaluated together, then each against the stock route
                var batch: [2 * one_batch]Array = undefined;
                for (0..one_batch) |b| {
                    batch[2 * b] = got_i[b];
                    batch[2 * b + 1] = got_s[b];
                }
                try evalAll(&batch);
                for (0..one_batch) |b| {
                    try countBits(s, want_i[b], got_i[b], &ind);
                    try countBits(s, want_s[b], got_s[b], &sco);
                }
            }
            const ok = ind.differing == 0 and sco.differing == 0 and sco.nonfinite == 0 and ind.total == calls * @as(usize, @intCast(rows * top_k)) and sco.total == ind.total;
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE_BYTES kind=route wt={s} M={d} calls={d} inds_differing={d}/{d} scores_differing={d}/{d} nonfinite={d} {s}\n", .{ @tagName(wt), rows, calls, ind.differing, ind.total, sco.differing, sco.total, sco.nonfinite, if (ok) "PASS" else "FAIL" });
        }
        // the predictor: f16 routers (a bf16 router's stock matmul stays bf16, which the plan declines)
        if (wt != .float16) continue;
        for (rows_tested) |rows| {
            var sc: Count = .{};
            var c: usize = 0;
            while (c < calls) : (c += one_batch) {
                var xs: [one_batch]Array = @splat(.{ .ctx = null });
                var wants: [one_batch]Array = @splat(.{ .ctx = null });
                var gots: [one_batch]Array = @splat(.{ .ctx = null });
                defer for ([_][]Array{ &xs, &wants, &gots }) |set| freeAll(set);
                for (0..one_batch) |b| {
                    const r = (c + b) % routers;
                    // x as the routed call's ([rows, hidden]) and, in turn, with a leading batch of 1
                    xs[b] = if ((c + b) % 2 == 0)
                        try normal(s, &seed, &.{ rows, fused_router.hidden }, .bfloat16, 1.0)
                    else
                        try normal(s, &seed, &.{ 1, rows, fused_router.hidden }, .bfloat16, 1.0);
                    wants[b] = try stockPredict(s, xs[b], raws[r], corrs[r]);
                    gots[b] = try fused_router.predictOne(s, xs[b], raws[r], corrs[r]);
                }
                try evalAll(&gots);
                for (0..one_batch) |b| try countBits(s, wants[b], gots[b], &sc);
            }
            const ok = sc.differing == 0 and sc.nonfinite == 0 and sc.total == calls * @as(usize, @intCast(rows * fused_router.experts));
            if (!ok) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE_BYTES kind=predict wt={s} M={d} calls={d} scores_differing={d}/{d} nonfinite={d} {s}\n", .{ @tagName(wt), rows, calls, sc.differing, sc.total, sc.nonfinite, if (ok) "PASS" else "FAIL" });
        }
    }
    std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE_PARITY lines={d} fails={d} {s}\n", .{ one_lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}

test "turboexl3_glm53f router gpu: TURBOEXL3_GLM53F_ROUTER_ONE1's one launch gives the stock one-row route's logits, routes and weights bit for bit, 512 calls a weight type, eight calls evaluated together; it refuses 2 rows, and an unregistered layer's one-row call routes as before" {
    if (std.c.getenv("TURBOEXL3_GLM53F_ROUTER_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261110;
    var fails: usize = 0;
    comptime std.debug.assert(calls % one_batch == 0);
    // a router and its bias of the last type, for the 2-row decline
    var last_raw: Array = .{ .ctx = null };
    var last_bias: Array = .{ .ctx = null };
    defer for ([_]Array{ last_raw, last_bias }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (weight_types) |wt| {
        var raws: [routers]Array = @splat(.{ .ctx = null });
        var tables: [routers]Array = @splat(.{ .ctx = null });
        var biases: [routers]Array = @splat(.{ .ctx = null });
        defer for ([_][]Array{ &raws, &tables, &biases }) |set| freeAll(set);
        for (0..routers) |i| {
            raws[i] = try normal(s, &seed, &.{ fused_router.experts, fused_router.hidden }, wt, 1.0 / 64.0);
            tables[i] = try table(s, raws[i]);
            biases[i] = try normal(s, &seed, &.{fused_router.experts}, .float32, 0.1);
        }
        var lg: Count = .{};
        var ind: Count = .{};
        var sco: Count = .{};
        var c: usize = 0;
        while (c < calls) : (c += one_batch) {
            var xs: [one_batch]Array = @splat(.{ .ctx = null });
            var want_l: [one_batch]Array = @splat(.{ .ctx = null });
            var want_i: [one_batch]Array = @splat(.{ .ctx = null });
            var want_s: [one_batch]Array = @splat(.{ .ctx = null });
            var got_l: [one_batch]Array = @splat(.{ .ctx = null });
            var got_i: [one_batch]Array = @splat(.{ .ctx = null });
            var got_s: [one_batch]Array = @splat(.{ .ctx = null });
            defer for ([_][]Array{ &xs, &want_l, &want_i, &want_s, &got_l, &got_i, &got_s }) |set| freeAll(set);
            for (0..one_batch) |b| {
                const r = (c + b) % routers;
                // x as a decode call's ([1, 1, hidden]) and, in turn, 2-D ([1, hidden])
                xs[b] = if ((c + b) % 2 == 0)
                    try normal(s, &seed, &.{ 1, 1, fused_router.hidden }, .bfloat16, 1.0)
                else
                    try normal(s, &seed, &.{ 1, fused_router.hidden }, .bfloat16, 1.0);
                // the stock route: MLX's matmul over the f32 table (its gemv at one row), then the stock router kernel
                want_l[b] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_matmul(&want_l[b], xs[b], tables[r], s));
                const want = (try router_kernel.moeRouterTopK(s, want_l[b], biases[r], top_k, .sigmoid_bias, route_norm, route_scale, .bfloat16, 0, 0, null)) orelse return error.RouterGpuStockDeclined;
                want_i[b] = want.inds;
                want_s[b] = want.norm_scores;
                // the one launch
                const got = try fused_router.routeOne1(s, xs[b], raws[r], biases[r], top_k, route_norm, route_scale);
                got_l[b] = got.logits;
                got_i[b] = got.routing.inds;
                got_s[b] = got.routing.norm_scores;
            }
            // the batch's one launches evaluated together, then each against the stock route
            var batch: [3 * one_batch]Array = undefined;
            for (0..one_batch) |b| {
                batch[3 * b] = got_l[b];
                batch[3 * b + 1] = got_i[b];
                batch[3 * b + 2] = got_s[b];
            }
            try evalAll(&batch);
            for (0..one_batch) |b| {
                try countBits(s, want_l[b], got_l[b], &lg);
                try countBits(s, want_i[b], got_i[b], &ind);
                try countBits(s, want_s[b], got_s[b], &sco);
            }
        }
        const ok = lg.differing == 0 and ind.differing == 0 and sco.differing == 0 and lg.nonfinite == 0 and sco.nonfinite == 0 and lg.total == calls * @as(usize, @intCast(fused_router.experts)) and ind.total == calls * @as(usize, @intCast(top_k)) and sco.total == ind.total;
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE1_BYTES wt={s} calls={d} logits_differing={d}/{d} inds_differing={d}/{d} scores_differing={d}/{d} nonfinite={d} {s}\n", .{ @tagName(wt), calls, lg.differing, lg.total, ind.differing, ind.total, sco.differing, sco.total, lg.nonfinite + sco.nonfinite, if (ok) "PASS" else "FAIL" });
        for ([_]*Array{ &last_raw, &last_bias }, [_]Array{ raws[0], biases[0] }) |dst, src| {
            if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
            dst.* = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(dst, src);
        }
    }
    // 2 rows: the one-row launch refuses (a call of 2 to 4 rows is ROUTER's)
    {
        const x = try normal(s, &seed, &.{ 1, 2, fused_router.hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x);
        const refused = if (fused_router.routeOne1(s, x, last_raw, last_bias, top_k, route_norm, route_scale)) |r| blk: {
            for ([_]Array{ r.logits, r.routing.inds, r.routing.norm_scores }) |a| {
                _ = mlx.mlx_array_free(a);
            }
            break :blk false;
        } else |e| e == error.Glm53fRouterRows;
        if (!refused) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE1_DECLINE M=2 one1={s} {s}\n", .{ if (refused) "refused" else "ran", if (refused) "PASS" else "FAIL" });
    }
    // a one-row call on a layer not registered, ROUTER and ROUTER_ONE1 on: `route` finds no raw router for it (any
    // pointer: it is not read) and the call routes as before
    {
        const was_on = fused_router.on;
        const was_one1 = fused_router.one1_on;
        fused_router.on = true;
        fused_router.one1_on = true;
        defer {
            fused_router.on = was_on;
            fused_router.one1_on = was_one1;
        }
        const stand_in: moe.Moe.Weights = undefined;
        const x = try normal(s, &seed, &.{ 1, 1, fused_router.hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x);
        const before = fused_router.other_rows;
        const before_one1 = fused_router.one1_routes + fused_router.one1_head_routes;
        const routed = try fused_router.route(s, &stand_in, x);
        if (routed) |r| {
            _ = mlx.mlx_array_free(r.inds);
            _ = mlx.mlx_array_free(r.norm_scores);
        }
        const ok = routed == null and fused_router.other_rows == before + 1 and fused_router.one1_routes + fused_router.one1_head_routes == before_one1;
        if (!ok) fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE1_DECLINE M=1 layer=unregistered route={s} {s}\n", .{ if (routed == null) "null" else "routed", if (ok) "PASS" else "FAIL" });
    }
    std.debug.print("TURBOEXL3_GLM53F_ROUTER_ONE1_PARITY lines={d} fails={d} {s}\n", .{ one1_lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
