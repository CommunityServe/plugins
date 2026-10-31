//! The Metal tests of the directional ablation's write sites (`dsv41_model.ablate`, `ablateExpanded`,
//! `ablateEngramValue`): run under `zig build gpu-test` (rooted at `attn_nax_test.zig`).
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const ci = sc_mod.ci;
const testing = std.testing;

const D = 256;
const HC = 4;

fn randn(sc: *Scope, shape: []const c_int, seed: u64) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1, key, sc.s), &o);
}

/// An ablation along a random unit `r` at strength `s`.
fn armed(sc: *Scope, s: f32) !dsv.Ablation {
    const raw = try randn(sc, &.{D}, 7);
    const r = try sc.div(raw, try sc.op1(mlx.mlx_sqrt, try sc.sum(try sc.mul(raw, raw), 0, false)));
    return .{ .r = r, .col = try sc.reshape(r, &.{ D, 1 }), .s = try sc.f(s) };
}

/// The component of every row of `x` (`[..., D]`) along `r`, as host floats.
fn along(sc: *Scope, x: A, r: A) ![]const f32 {
    return sc_mod.hostF32(try sc.matmul(try sc.astype(x, .float32), try sc.reshape(r, &.{ D, 1 })));
}

fn maxAbs(v: []const f32) f32 {
    var m: f32 = 0;
    for (v) |x| m = @max(m, @abs(x));
    return m;
}

fn skipNoGpu() !void {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
}

test "sushi_ds41 gpu: ablation removes the strength's share of a write's component along r, f32 and bf16" {
    try skipNoGpu();
    var sc = Scope.init(testing.allocator, mlx.mlx_default_gpu_stream_new());
    defer sc.deinit();
    const y = try randn(&sc, &.{ 9, D }, 11);
    const ab = try armed(&sc, 1);
    const r = ab.r;
    const before = try testing.allocator.dupe(f32, try along(&sc, y, r));
    defer testing.allocator.free(before);
    try testing.expect(maxAbs(before) > 0.1);
    try testing.expect(maxAbs(try along(&sc, try dsv.ablateWith(&sc, ab, y), r)) < 1e-5);
    const yb = try sc.astype(y, .bfloat16);
    const out_b = try dsv.ablateWith(&sc, ab, yb);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out_b));
    // bf16 rounding of the result leaves a residue of a few ulps of the row's norm (~16 here).
    try testing.expect(maxAbs(try along(&sc, out_b, r)) < 0.2);
    const half = try along(&sc, try dsv.ablateWith(&sc, try armed(&sc, 0.5), y), r);
    for (half, before) |h, b| try testing.expectApproxEqAbs(b / 2, h, 1e-4);
}

test "sushi_ds41 gpu: the fused decode stream's ablation equals expanding the ablated output" {
    try skipNoGpu();
    var sc = Scope.init(testing.allocator, mlx.mlx_default_gpu_stream_new());
    defer sc.deinit();
    const ab = try armed(&sc, 1);
    const r = ab.r;
    const L = 3;
    const x1 = try randn(&sc, &.{ L, HC, D }, 21);
    const y = try randn(&sc, &.{ L, D }, 22);
    const post = try randn(&sc, &.{ L, HC }, 23);
    const comb = try randn(&sc, &.{ L, HC, HC }, 24);
    const x2 = try dsv.hcExpandOps(&sc, x1, y, post, comb);
    const got = try sc_mod.hostF32(try dsv.ablateExpandedWith(&sc, ab, x1, x2, comb));
    const want = try sc_mod.hostF32(try dsv.hcExpandOps(&sc, x1, try dsv.ablateWith(&sc, ab, y), post, comb));
    // The two orders round differently on the GPU's f32 matmuls: within 1e-3 of the stream's scale.
    for (got, want) |g, w| try testing.expectApproxEqAbs(w, g, 1e-3 * @max(1, @abs(w)));
    // An r-free stream stays r-free through an expand of an ablated output.
    const x1_free = try sc.reshape(try dsv.ablateWith(&sc, ab, try sc.reshape(x1, &.{ L * HC, D })), &.{ L, HC, D });
    const out = try dsv.ablateExpandedWith(&sc, ab, x1_free, try dsv.hcExpandOps(&sc, x1_free, y, post, comb), comb);
    try testing.expect(maxAbs(try along(&sc, out, r)) < 1e-4);
}

test "sushi_ds41 gpu: the Engram ablation leaves the gate's keys and clears the value along r" {
    try skipNoGpu();
    var sc = Scope.init(testing.allocator, mlx.mlx_default_gpu_stream_new());
    defer sc.deinit();
    const ab = try armed(&sc, 1);
    const r = ab.r;
    const kv = try sc.astype(try randn(&sc, &.{ 5, (HC + 1) * D }, 31), .bfloat16);
    const out = try dsv.ablateValueWith(&sc, ab, kv, HC * D);
    const keys_in = try testing.allocator.dupe(f32, try sc_mod.hostF32(try sc.astype(try sc.slice(kv, -1, 0, HC * D), .float32)));
    defer testing.allocator.free(keys_in);
    try testing.expectEqualSlices(f32, keys_in, try sc_mod.hostF32(try sc.astype(try sc.slice(out, -1, 0, HC * D), .float32)));
    const before = maxAbs(try along(&sc, try sc.slice(kv, -1, HC * D, (HC + 1) * D), r));
    const after = maxAbs(try along(&sc, try sc.slice(out, -1, HC * D, (HC + 1) * D), r));
    try testing.expect(before > 0.1 and after < before / 10);
}

test "sushi_ds41 gpu: the stream tap keeps the hc mean of each pass's last rows and frees them on clear" {
    try skipNoGpu();
    var sc = Scope.init(testing.allocator, mlx.mlx_default_gpu_stream_new());
    defer sc.deinit();
    var tap = try dsv.StreamTap.init(testing.allocator, 2, 4);
    defer tap.deinit();
    const x = try randn(&sc, &.{ 6, HC, D }, 41);
    tap.on = true;
    try tap.record(&sc, 0, x);
    try tap.record(&sc, 2, try sc.slice(x, 0, 5, 6));
    tap.on = false;
    try tap.record(&sc, 1, x);
    try testing.expectEqual(@as(usize, 1), tap.sites[0].items.len);
    try testing.expectEqual(@as(usize, 0), tap.sites[1].items.len);
    const got = try testing.allocator.dupe(f32, try sc_mod.hostF32(tap.sites[0].items[0]));
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(usize, 4 * D), got.len);
    const want = try sc_mod.hostF32(try sc.mean(try sc.slice(x, 0, 2, 6), 1, false));
    for (got, want) |g, w| try testing.expectApproxEqAbs(w, g, 1e-5);
    try testing.expectEqual(@as(usize, 1 * D), mlx.mlx_array_size(tap.sites[2].items[0]));
    tap.clear();
    try testing.expectEqual(@as(usize, 0), tap.sites[0].items.len);
}
