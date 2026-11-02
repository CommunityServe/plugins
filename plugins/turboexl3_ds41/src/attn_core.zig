//! `MLX_SERVE_DSV41_ATTN_CORE` (read at load; unset, empty or `1`: on; `0`: off; `check`: off's launches served, the
//! new ones run beside every gather call and compared word for word; anything else is refused by name): a wide prompt
//! layer's attention gather (the copied model's `attnGather`) on two exact changes. Each moves only which threads
//! compute a word, never how it is computed.
//!
//! pv at 64 output columns a threadgroup. The stock attn_pv.metal gives a threadgroup one row's 128 output columns of all
//! 64 heads: four simdgroups of 16 heads on the 16 x 128 x 32 tensor op, 32 keys a step staged in threadgroup memory a
//! step ahead (each thread 64 bytes of a key), a lane's accumulator 64 floats. On, its text at DT 64 (`derivePv64`, at
//! load, from the stock text the plugin compiles: the tile constant, the staging registers, and the fetch and the store
//! at 32 bytes a thread): the op 16 x 64 x 32, D / 64 threadgroups a row, a lane's accumulator 32 floats, the staging
//! half. Every output word is the same contraction: the same 32 keys a step in the stock order on the same tensor op,
//! accumulated in f32, then the same epilogue on it (the inverse-roped tail and the grouped store, or the plain store).
//! The SCRATCH_ATTN ring twin and SCRATCH_HEADS' pv twin derive from the DT-64 text as they do from the stock one (their
//! prologue, and the grouped store's one statement, which the DT-64 text keeps).
//!
//! The 3D launch. The gather's tensor-op launches (qk, Q_TAIL's qk_t and pv, fresh and as their ring and heads twins)
//! go as MLX's own NAX GEMMs do: threadgroup (32, 1, 4) over (x / 4, rows, 4) threads, not (128, 1, 1) over (x, rows,
//! 1). The threadgroups' positions are the same, a thread's `thread_index_in_threadgroup` is x + 32 z (its 1D index),
//! its simdgroup is z, and the twins' carrier is still written by the grid's one first thread. sm keeps its launch.
//!
//! pv1u (w-pv1u-20261102-0644: L5's late span, 8192 compressed rows, NK 640, GROUPED as served; A B B A twice): 0
//! differing words against the stock at DT 64, in 1D and in 3D; 425.6 -> 336.6 us a call at DT 64 (1.264x), 332.1 with
//! the 3D launch (1.285x). ls1u (w-ls1u-20261102-0630): the 3D launch alone 0 differing words, qk 1.066x, qk_t 1.064x,
//! pv 1.016x.
//!
//! `check` serves the stock texts at the stock launches and, beside every gather call, runs its scores launch (qk, or
//! qk_t as served) at the stock launch and in 3D, its sm, and its pv on the stock text at the stock launch and on the
//! DT-64 text in 3D, in a scope of its own; each pair compared word for word on the device: `[attn-core] check kind=
//! calls= words= differ= max_ulps=`, a kind every 128 calls, at its first differing word and at the model's end.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

const A = mlx.mlx_array;

pub const Mode = enum { off, on, check };
pub var mode: Mode = .on;

pub const Error = error{
    /// MLX_SERVE_DSV41_ATTN_CORE is none of 0, 1, check (unset or empty: on).
    AttnCoreSetting,
    /// A statement `derivePv64` rewrites is not in the stock attn_pv text exactly once.
    AttnCorePvStatement,
};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.AttnCoreSetting;
}

/// The gather's tensor-op launches go three-dimensional (on; off and check serve the stock launches).
pub fn three() bool {
    return mode == .on;
}

/// pv's output columns a threadgroup as served: 64 on, the stock 128 off and in check.
pub fn pvCols() usize {
    return if (mode == .on) 64 else 128;
}

/// A gather launch's grid and threadgroup. `grid`: the stock launch's threads (x, rows), at (128, 1, 1); `three_d`: MLX's
/// NAX GEMMs' shape, (32, 1, 4) over (x / 4, rows, 4), the same threadgroups.
pub const Launch = struct { grid: [3]c_int, tg: [3]c_int };
pub fn launch(grid: [2]c_int, three_d: bool) Launch {
    if (!three_d) return .{ .grid = .{ grid[0], grid[1], 1 }, .tg = .{ 128, 1, 1 } };
    return .{ .grid = .{ @divExact(grid[0], 4), grid[1], 4 }, .tg = .{ 32, 1, 4 } };
}

/// One statement `derivePv64` replaces and its DT-64 form.
pub const Sub = struct { old: []const u8, new: []const u8 };
/// The stock attn_pv statements at DT 64: the tile constant; the staging registers (2 uint4: 16 of a key's columns a
/// thread, the four `part`s of the 32 keys' `kk` covering 64); the fetch and the store at 16 columns a `part`.
pub const pv64_subs = [_]Sub{
    .{ .old = "constexpr int DT = 128;\n", .new = "constexpr int DT = 64;\n" },
    .{
        .old = "const int kk = int(lid >> 2);\nconst int part = int(lid & 3);\nuint4 reg[4];\n",
        .new = "const int kk = int(lid >> 2);\nconst int part = int(lid & 3);\nuint4 reg[2];\n",
    },
    .{
        .old = "  const device uint4* src = (const device uint4*)(vrow + dt * DT + part * 32);\n#pragma clang loop unroll(full)\n  for (int i = 0; i < 4; i++) reg[i] = live ? src[i] : uint4(0);\n",
        .new = "  const device uint4* src = (const device uint4*)(vrow + dt * DT + part * 16);\n#pragma clang loop unroll(full)\n  for (int i = 0; i < 2; i++) reg[i] = live ? src[i] : uint4(0);\n",
    },
    .{
        .old = "  threadgroup uint4* dst = (threadgroup uint4*)(Vs[buf] + kk * DT + part * 32);\n#pragma clang loop unroll(full)\n  for (int i = 0; i < 4; i++) dst[i] = reg[i];\n",
        .new = "  threadgroup uint4* dst = (threadgroup uint4*)(Vs[buf] + kk * DT + part * 16);\n#pragma clang loop unroll(full)\n  for (int i = 0; i < 2; i++) dst[i] = reg[i];\n",
    },
};

/// pv's DT-64 text from the stock one (the caller frees it): every `pv64_subs` statement replaced, the rest unchanged.
/// A statement missing, or there more than once, refuses by name (the load: the copied model builds pv at load).
pub fn derivePv64(a: std.mem.Allocator, stock: []const u8) (Error || error{OutOfMemory})![:0]u8 {
    for (pv64_subs) |sb| if (std.mem.count(u8, stock, sb.old) != 1) return error.AttnCorePvStatement;
    var body = try a.dupe(u8, stock);
    for (pv64_subs) |sb| {
        const next = std.mem.replaceOwned(u8, a, body, sb.old, sb.new) catch |e| {
            a.free(body);
            return e;
        };
        a.free(body);
        body = next;
    }
    defer a.free(body);
    return a.dupeSentinel(u8, body, 0);
}

// ── check ──

/// The launches `check` compares: the scores (qk, or Q_TAIL's qk_t, as served) and pv.
pub const Kind = enum { qk, qk_t, pv };
pub const Tally = struct { calls: u64 = 0, words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0 };
pub var tallies: std.enums.EnumArray(Kind, Tally) = .initFill(.{});
const report_every: u64 = 128;

/// `check`: `got` (the new launch) against `want` (the stock one) word for word, counted under `kind`.
pub fn compare(s: mlx.mlx_stream, kind: Kind, got: A, want: A) !void {
    const st = try compareWords(s, got, want);
    const t = tallies.getPtr(kind);
    const first_bad = t.differ == 0 and st.differ > 0;
    t.calls += 1;
    t.words += st.words;
    t.differ += st.differ;
    t.max_ulps = @max(t.max_ulps, st.max_ulps);
    if (first_bad or t.calls % report_every == 0) line(kind, t);
}

fn line(kind: Kind, t: *const Tally) void {
    std.debug.print("[attn-core] check kind={s} calls={d} words={d} differ={d} max_ulps={d}\n", .{ @tagName(kind), t.calls, t.words, t.differ, t.max_ulps });
}

/// Every compared kind's line (the model's deinit).
pub fn report() void {
    for (std.enums.values(Kind)) |k| {
        const t = tallies.getPtrConst(k);
        if (t.calls > 0) line(k, t);
    }
}

const Stats = struct { words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0 };

/// `a` against `b` (one size, 2- or 4-byte float words, any shapes of that size), word for word on the device; the
/// largest distance in units in the last place only when a word differs.
fn compareWords(s: mlx.mlx_stream, a: A, b: A) !Stats {
    const isz = mlx.mlx_array_itemsize(b);
    if (mlx.mlx_array_itemsize(a) != isz or mlx.mlx_array_size(a) != mlx.mlx_array_size(b)) return error.AttnCoreCheckShape;
    const st: mlx.mlx_dtype, const ut: mlx.mlx_dtype, const lo: i32 = switch (isz) {
        2 => .{ .int16, .uint16, -32768 },
        4 => .{ .int32, .uint32, std.math.minInt(i32) },
        else => return error.AttnCoreCheckWords,
    };
    const n: u64 = mlx.mlx_array_size(b);
    const flat = [_]c_int{-1};
    var f: [2]A = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (f) |v| {
        _ = mlx.mlx_array_free(v);
    };
    try mlx.check(mlx.mlx_reshape(&f[0], a, &flat, 1, s));
    try mlx.check(mlx.mlx_reshape(&f[1], b, &flat, 1, s));
    var t: [5]A = undefined;
    for (&t) |*v| v.* = mlx.mlx_array_new();
    defer for (t) |v| {
        _ = mlx.mlx_array_free(v);
    };
    try mlx.check(mlx.mlx_view(&t[0], f[0], ut, s));
    try mlx.check(mlx.mlx_view(&t[1], f[1], ut, s));
    try mlx.check(mlx.mlx_equal(&t[2], t[0], t[1], s));
    try mlx.check(mlx.mlx_astype(&t[3], t[2], .uint32, s));
    try mlx.check(mlx.mlx_sum(&t[4], t[3], false, s));
    try mlx.check(mlx.mlx_array_eval(t[4]));
    const same: u64 = (mlx.mlx_array_data_uint32(t[4]) orelse return error.NoData)[0];
    if (same == n) return .{ .words = n };
    var u: [12]A = undefined;
    for (&u) |*v| v.* = mlx.mlx_array_new();
    defer for (u) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    const low = mlx.mlx_array_new_int(lo);
    defer _ = mlx.mlx_array_free(low);
    try mlx.check(mlx.mlx_view(&u[0], f[0], st, s));
    try mlx.check(mlx.mlx_view(&u[1], f[1], st, s));
    try mlx.check(mlx.mlx_astype(&u[2], u[0], .int64, s));
    try mlx.check(mlx.mlx_astype(&u[3], u[1], .int64, s));
    for ([_]usize{ 2, 3 }, [_]usize{ 4, 7 }) |src, dst| {
        try mlx.check(mlx.mlx_less(&u[dst], u[src], zero, s));
        try mlx.check(mlx.mlx_subtract(&u[dst + 1], low, u[src], s));
        try mlx.check(mlx.mlx_where(&u[dst + 2], u[dst], u[dst + 1], u[src], s));
    }
    try mlx.check(mlx.mlx_subtract(&u[10], u[6], u[9], s));
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, u[10], s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    try mlx.check(mlx.mlx_astype(&u[11], mx, .float32, s));
    try mlx.check(mlx.mlx_array_eval(u[11]));
    const ulps: f32 = (mlx.mlx_array_data_float32(u[11]) orelse return error.NoData)[0];
    return .{ .words = n, .differ = n - same, .max_ulps = @intFromFloat(@min(@max(ulps, 0), 1.8e19)) };
}

const testing = std.testing;
/// The stock text the plugin compiles (the copied model's `ATTN_PV_SOURCE`) and its header (`MPP_HEADER`).
const stock_pv = @embedFile("mlx_serve/plugins/deepseek_v41/kernels/attn_pv.metal");
const mpp_header = @embedFile("mlx_serve/kernels/common/mpp_tensor.metal");
const twins = @import("attn_twins.zig");

test "attn_core setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "true", "Check", " 1" }) |v| try testing.expectError(error.AttnCoreSetting, parse(v));
}

test "attn_core launch: the 3D launch keeps the stock threadgroups, 128 threads each" {
    for ([_][2]c_int{ .{ 1280, 1 }, .{ 1280, 256 }, .{ 1024, 7 }, .{ 512, 256 } }) |g| {
        const flat = launch(g, false);
        const tri = launch(g, true);
        try testing.expectEqual([3]c_int{ g[0], g[1], 1 }, flat.grid);
        try testing.expectEqual([3]c_int{ 128, 1, 1 }, flat.tg);
        try testing.expectEqual([3]c_int{ 32, 1, 4 }, tri.tg);
        for (0..3) |i| try testing.expectEqual(@divExact(flat.grid[i], flat.tg[i]), @divExact(tri.grid[i], tri.tg[i]));
        try testing.expectEqual(flat.tg[0] * flat.tg[1] * flat.tg[2], tri.tg[0] * tri.tg[1] * tri.tg[2]);
    }
}

test "attn_core pv64: the stock text at DT 64, every other statement unchanged" {
    const t = try derivePv64(testing.allocator, stock_pv);
    defer testing.allocator.free(t);
    try testing.expectEqual(@as(u8, 0), t.ptr[t.len]);
    for (pv64_subs) |sb| {
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, sb.old));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, sb.new));
    }
    // the four statements put back give the stock text, byte for byte
    var back = try testing.allocator.dupe(u8, t);
    defer testing.allocator.free(back);
    for (pv64_subs) |sb| {
        const next = try std.mem.replaceOwned(u8, testing.allocator, back, sb.new, sb.old);
        testing.allocator.free(back);
        back = next;
    }
    try testing.expectEqualStrings(stock_pv, back);
    // what the rewrite rests on: 32 keys a step (kk, lid >> 2 of 128 threads), the stock op's shape from DT, the tile's
    // columns from DT in the epilogue and the plain store
    for ([_][]const u8{
        "constexpr int KS = 32;\n",
        "    HS, DT, KS, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n",
        "    const int d = dt * DT + int(ix[0]);\n",
        "  auto oo = O.template slice<DT, HS>(dt * DT, int(sg) * HS);\n",
        "threadgroup T Vs[2][KS * DT];\n",
    }) |w| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, w));
    // a text without one of the statements is refused
    try testing.expectError(error.AttnCorePvStatement, derivePv64(testing.allocator, "constexpr int DT = 128;\n"));
    try testing.expectError(error.AttnCorePvStatement, derivePv64(testing.allocator, t));
}

test "attn_core pv64: the SCRATCH_ATTN ring twin and SCRATCH_HEADS' twin derive from the DT-64 text" {
    const t = try derivePv64(testing.allocator, stock_pv);
    defer testing.allocator.free(t);
    const p_io = [_]twins.Io{.{ .name = "p", .mtype = "float16_t" }};
    const ring = try twins.derive(testing.allocator, t, mpp_header, &.{}, &p_io);
    defer testing.allocator.free(ring);
    try testing.expect(std.mem.endsWith(u8, ring, t));
    // the copied model's heads twin (`GatherAttn.RingTwins.headsTwin`): the grouped store's one statement, kept at DT 64
    const subs = [_]twins.Sub{.{
        .old = "    device OT* dst = o + (size_t(h / hg) * L + l) * size_t(hg * D) + size_t(h % hg) * D + d;\n",
        .new = "    device OT* dst = o + (size_t(soff[1] + l) * G + size_t(h / hg)) * size_t(hg * D) + size_t(h % hg) * D + d;\n",
    }};
    const heads = try twins.deriveWith(testing.allocator, t, mpp_header, &.{.{ .name = "o", .mtype = "OT", .off = null }}, &p_io, &subs, &.{"dep2"});
    defer testing.allocator.free(heads);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, heads, subs[0].new));
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, heads, subs[0].old));
    for (pv64_subs) |sb| try testing.expectEqual(@as(usize, 1), std.mem.count(u8, heads, sb.new));
}
