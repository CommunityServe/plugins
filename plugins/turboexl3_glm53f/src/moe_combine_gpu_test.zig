//! The device check of `TURBOEXL3_GLM53F_MOE_COMBINE1` (src/moe_combine1.zig) and `TURBOEXL3_GLM53F_ROUTE_VIEW`
//! (src/route_view.zig), run by the build step `moe-combine-gpu-test` (linked to the server's MLX) under
//! TURBOEXL3_GLM53F_MOE_COMBINE_GPU=1; `zig build test` compiles it and its filter leaves it out.
//!
//! - The combine text against the stock decode lane's join and combine (glm5n/experts.zig: the take of each position's
//!   row by the join's order, then `combineStock`: astype f32, the multiply by the f32 weights, the sum over the 8
//!   routes, astype bf16), bit for bit, at 1, 2, 3 and 4 rows, over joined outputs as long as the positions and 24 rows
//!   longer (a device wave and a miss part), the order a random injective pick of their rows, bf16 weights (the router's,
//!   which the stock lane casts to f32 first) and f32 ones, rows with exact +0 and -0 among normal values and weights
//!   with exact zeros: `draws` calls a line, each its own seeded values. Then three calls the text declines (a 2048-wide
//!   row, an int32 order, f16 weights): no output, the caller's stock chain runs.
//! - The route ids' int32 view against the stock cast, at 1 to 4 rows of 8 ids below 288: the same values, and the
//!   view's data the router's buffer (no copy); int32 ids take no view.
//! One line a case, and the run's last line:
//!
//!     TURBOEXL3_GLM53F_MOE_COMBINE1_BYTES rows=<n> joined=<R> wt=<bf16|f32> calls=<n> differing=<n>/<total>
//!         nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_MOE_COMBINE1_DECLINED case=<case> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTE_VIEW_BYTES rows=<n> differing=<n>/<total> shared=<0|1> PASS|FAIL
//!     TURBOEXL3_GLM53F_ROUTE_VIEW_DECLINED dtype=int32 PASS|FAIL
//!     TURBOEXL3_GLM53F_MOE_COMBINE_PARITY lines=<n> fails=<n> calls=<n> views=<n> PASS|FAIL
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const moe_combine1 = @import("moe_combine1.zig");
const route_view = @import("route_view.zig");
const Array = mlx.mlx_array;

const hidden = moe_combine1.hidden;
const top_k = moe_combine1.top_k;
/// GLM-5.3-Flash's routed experts: every id is below it
const experts: u32 = 288;
const rows_tested = [_]c_int{ 1, 2, 3, 4 };
/// the joined outputs past the positions: none, and a 24-row device wave's worth more
const extra_rows = [_]c_int{ 0, 24 };
const weight_types = [_]mlx.mlx_dtype{ .bfloat16, .float32 };
/// calls a combine line, each with its own values
const draws: usize = 16;
pub const combine_lines = rows_tested.len * extra_rows.len * weight_types.len;
pub const decline_lines = 3;
pub const view_lines = rows_tested.len + 1;

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

fn freeAll(arrs: []const Array) void {
    for (arrs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

/// A host array of `vals` with `shape`
fn hostU32(vals: []const u32, shape: []const c_int, dt: mlx.mlx_dtype) !Array {
    const a = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), dt);
    if (a.ctx == null) return error.MoeCombineGpuHost;
    return a;
}

/// [R, hidden] bf16: seeded normal values with about 1 in 32 set to +0 and 1 in 32 to -0 (the products' sign of zero)
fn routedRows(s: mlx.mlx_stream, seed: *u64, r: c_int) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    try mlx.check(mlx.mlx_random_normal(&vals, &[_]c_int{ r, hidden }, 2, .bfloat16, 0, 1.0, key, s));
    var key2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key2);
    try mlx.check(mlx.mlx_random_key(&key2, seed.*));
    seed.* += 1;
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    const lo = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    try mlx.check(mlx.mlx_random_uniform(&u, lo, hi, &[_]c_int{ r, hidden }, 2, .float32, key2, s));
    const pz_t = mlx.mlx_array_new_float(1.0 / 32.0);
    defer _ = mlx.mlx_array_free(pz_t);
    const nz_t = mlx.mlx_array_new_float(2.0 / 32.0);
    defer _ = mlx.mlx_array_free(nz_t);
    var pz_m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pz_m);
    try mlx.check(mlx.mlx_less(&pz_m, u, pz_t, s));
    var nz_m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nz_m);
    try mlx.check(mlx.mlx_less(&nz_m, u, nz_t, s));
    const pz_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(pz_f);
    const nz_f = mlx.mlx_array_new_float(-0.0);
    defer _ = mlx.mlx_array_free(nz_f);
    var pz = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pz);
    try mlx.check(mlx.mlx_astype(&pz, pz_f, .bfloat16, s));
    var nz = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nz);
    try mlx.check(mlx.mlx_astype(&nz, nz_f, .bfloat16, s));
    // u < 1/32: +0; 1/32 <= u < 2/32: -0; else the normal value
    var mid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mid);
    try mlx.check(mlx.mlx_where(&mid, nz_m, nz, vals, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_where(&out, pz_m, pz, mid, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// [n, top_k] bf16 route weights in [0, 2) with about 1 in 16 exactly 0 (host-made from the PRNG)
fn routeWeights(rnd: std.Random, n: c_int) !Array {
    var vals: [@as(usize, @intCast(rows_tested[rows_tested.len - 1] * top_k))]u16 = undefined;
    const cnt: usize = @intCast(n * top_k);
    for (vals[0..cnt]) |*v| {
        const f: f32 = if (rnd.uintLessThan(u32, 16) == 0) 0.0 else rnd.float(f32) * 2.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16); // truncation: any bf16 value serves as a weight
    }
    const a = mlx.mlx_array_new_data(&vals, &[_]c_int{ n, top_k }, 2, .bfloat16);
    if (a.ctx == null) return error.MoeCombineGpuHost;
    return a;
}

/// The stock decode lane's join take and `combineStock` over `rows` by `order` with `w` (cast to f32 first, as the
/// source does): [n, hidden] bf16
fn stock(s: mlx.mlx_stream, rows: Array, order: Array, w: Array, n: c_int) !Array {
    var tk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tk);
    try mlx.check(mlx.mlx_take_axis(&tk, rows, order, 0, s));
    var routed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed);
    try mlx.check(mlx.mlx_reshape(&routed, tk, &[_]c_int{ n, top_k, hidden }, 3, s));
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, w, .float32, s));
    var expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expanded);
    try mlx.check(mlx.mlx_reshape(&expanded, w32, &[_]c_int{ n, top_k, 1 }, 3, s));
    var rf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf);
    try mlx.check(mlx.mlx_astype(&rf, routed, .float32, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, rf, expanded, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_sum_axis(&sum, prod, 1, false, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, sum, .bfloat16, s));
    return out;
}

const Diff = struct { differing: usize = 0, total: usize = 0, nonfinite: usize = 0 };

/// `got` against `want` (both bf16, the same shape) as bits
fn diffBf16(s: mlx.mlx_stream, want: Array, got: Array, d: *Diff) !void {
    if (mlx.mlx_array_dtype(want) != .bfloat16 or mlx.mlx_array_dtype(got) != .bfloat16) return error.MoeCombineGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.MoeCombineGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.MoeCombineGpuData)[0..n];
    const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.MoeCombineGpuData)[0..n];
    for (pw, pg) |a, b| {
        if (a != b) d.differing += 1;
        if ((b & 0x7F80) == 0x7F80) d.nonfinite += 1;
    }
    d.total += n;
}

test "turboexl3_glm53f moe combine gpu: TURBOEXL3_GLM53F_MOE_COMBINE1's one launch gives the stock join and combine's bits at 1 to 4 rows with bf16 and f32 weights, and TURBOEXL3_GLM53F_ROUTE_VIEW's ids view gives the cast's values on the router's buffer" {
    if (std.c.getenv("TURBOEXL3_GLM53F_MOE_COMBINE_GPU") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var seed: u64 = 20261109 + 41;
    var prng = std.Random.DefaultPrng.init(20261109 + 42);
    const rnd = prng.random();
    moe_combine1.on = true;
    defer moe_combine1.on = false;
    route_view.on = true;
    defer route_view.on = false;
    const calls0 = moe_combine1.calls;
    const views0 = route_view.views;
    var lines: usize = 0;
    var fails: usize = 0;

    // the combine against the stock join and combine
    for (rows_tested) |n| for (extra_rows) |extra| for (weight_types) |wt| {
        const r: c_int = n * top_k + extra;
        var d: Diff = .{};
        const before = moe_combine1.calls;
        for (0..draws) |_| {
            const rows = try routedRows(s, &seed, r);
            defer _ = mlx.mlx_array_free(rows);
            // the order: a random injective pick of the joined rows, as the join's (each position its own row)
            var pick: [@as(usize, @intCast(4 * 8 + 24))]u32 = undefined;
            const rr: usize = @intCast(r);
            for (pick[0..rr], 0..) |*p, i| p.* = @intCast(i);
            rnd.shuffle(u32, pick[0..rr]);
            const ids: usize = @intCast(n * top_k);
            const order = try hostU32(pick[0..ids], &[_]c_int{@intCast(ids)}, .uint32);
            defer _ = mlx.mlx_array_free(order);
            const w16 = try routeWeights(rnd, n);
            defer _ = mlx.mlx_array_free(w16);
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            if (wt == .float32) try mlx.check(mlx.mlx_astype(&w, w16, .float32, s)) else _ = mlx.mlx_array_set(&w, w16);
            const want = try stock(s, rows, order, w16, n);
            defer _ = mlx.mlx_array_free(want);
            const got = (try moe_combine1.combine(s, rows, order, w)) orelse return error.MoeCombineGpuDeclined;
            defer _ = mlx.mlx_array_free(got);
            try diffBf16(s, want, got, &d);
        }
        const ran = moe_combine1.calls - before;
        const ok = d.total > 0 and d.differing == 0 and d.nonfinite == 0 and ran == draws;
        lines += 1;
        fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_MOE_COMBINE1_BYTES rows={d} joined={d} wt={s} calls={d} differing={d}/{d} nonfinite={d} {s}\n", .{ n, r, @tagName(wt), ran, d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
    };

    // the calls the text declines: no output (the caller's stock chain runs)
    {
        var pick: [24]u32 = undefined;
        for (&pick, 0..) |*p, i| p.* = @intCast(i);
        const order = try hostU32(&pick, &[_]c_int{24}, .uint32);
        defer _ = mlx.mlx_array_free(order);
        const order_i = try hostU32(&pick, &[_]c_int{24}, .int32);
        defer _ = mlx.mlx_array_free(order_i);
        const rows = try routedRows(s, &seed, 24);
        defer _ = mlx.mlx_array_free(rows);
        var narrow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(narrow);
        try mlx.check(mlx.mlx_slice(&narrow, rows, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 24, 2048 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const w16 = try routeWeights(rnd, 3);
        defer _ = mlx.mlx_array_free(w16);
        var wh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wh);
        try mlx.check(mlx.mlx_astype(&wh, w16, .float16, s));
        const cases = [_]struct { name: []const u8, rows: Array, order: Array, w: Array }{
            .{ .name = "width_2048", .rows = narrow, .order = order, .w = w16 },
            .{ .name = "order_int32", .rows = rows, .order = order_i, .w = w16 },
            .{ .name = "weights_f16", .rows = rows, .order = order, .w = wh },
        };
        for (cases) |c| {
            const before = moe_combine1.calls;
            const got = try moe_combine1.combine(s, c.rows, c.order, c.w);
            if (got) |g| _ = mlx.mlx_array_free(g);
            const ok = got == null and moe_combine1.calls == before;
            lines += 1;
            fails += @intFromBool(!ok);
            std.debug.print("TURBOEXL3_GLM53F_MOE_COMBINE1_DECLINED case={s} {s}\n", .{ c.name, if (ok) "PASS" else "FAIL" });
        }
    }

    // the route ids' view against the cast, and on the router's buffer
    for (rows_tested) |n| {
        var vals: [32]u32 = undefined;
        const cnt: usize = @intCast(n * top_k);
        for (vals[0..cnt]) |*v| v.* = rnd.uintLessThan(u32, experts);
        const ids = try hostU32(vals[0..cnt], &[_]c_int{ n, top_k }, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        var cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, ids, .int32, s));
        const view = (try route_view.view(s, ids)) orelse return error.MoeCombineGpuNoView;
        defer _ = mlx.mlx_array_free(view);
        try evalAll(&.{ ids, cast, view });
        const pc = (mlx.mlx_array_data_int32(cast) orelse return error.MoeCombineGpuData)[0..cnt];
        const pv = (mlx.mlx_array_data_int32(view) orelse return error.MoeCombineGpuData)[0..cnt];
        const pi = mlx.mlx_array_data_uint32(ids) orelse return error.MoeCombineGpuData;
        var differing: usize = 0;
        for (pc, pv) |a, b| differing += @intFromBool(a != b);
        const shared = @intFromPtr(pv.ptr) == @intFromPtr(pi);
        const ok = differing == 0 and shared and mlx.mlx_array_dtype(view) == .int32 and std.mem.eql(c_int, mlx.getShape(view), mlx.getShape(ids));
        lines += 1;
        fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_ROUTE_VIEW_BYTES rows={d} differing={d}/{d} shared={d} {s}\n", .{ n, differing, cnt, @intFromBool(shared), if (ok) "PASS" else "FAIL" });
    }
    {
        const vals: [8]u32 = @splat(3);
        const ids = try hostU32(&vals, &[_]c_int{ 1, 8 }, .int32);
        defer _ = mlx.mlx_array_free(ids);
        const v = try route_view.view(s, ids);
        if (v) |a| _ = mlx.mlx_array_free(a);
        const ok = v == null;
        lines += 1;
        fails += @intFromBool(!ok);
        std.debug.print("TURBOEXL3_GLM53F_ROUTE_VIEW_DECLINED dtype=int32 {s}\n", .{if (ok) "PASS" else "FAIL"});
    }

    const expect_lines = combine_lines + decline_lines + view_lines;
    const total_calls = moe_combine1.calls - calls0;
    const total_views = route_view.views - views0;
    const ok = fails == 0 and lines == expect_lines and total_calls == combine_lines * draws and total_views == rows_tested.len;
    std.debug.print("TURBOEXL3_GLM53F_MOE_COMBINE_PARITY lines={d} fails={d} calls={d} views={d} {s}\n", .{ lines, fails, total_calls, total_views, if (ok) "PASS" else "FAIL" });
    try std.testing.expect(ok);
}
