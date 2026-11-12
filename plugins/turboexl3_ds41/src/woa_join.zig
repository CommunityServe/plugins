//! A grouped EXL3 linear's prompt rows (DeepSeek-V4.1's `wo_a`: x [rows, og, in], read in place) on one prepare and
//! one finish launch for every group, each group's W_hat decoded once a layer and kept across its runs (`Hats`, 67 MB
//! at the pack's shape), MLX's matmul per group between: 10 launches a run where the per-group `linear` took 41. The
//! same words: every value statement is turboexl3's prep.metal / finish.metal, re-addressed; decode rows keep the stock
//! call. Where turboexl3's trunk int8 takes the inner products a host splits off at the group's shape
//! (TURBOEXL3_DS41_PF_INT8 3 or 4: `t3.trunk_q8.innerTakes`), each group's inner product is its int8 one instead, as
//! turboexl3's split-off pieces run it: the group's int8 weights decoded once a layer and kept (`Hats.int8`, half the
//! bytes), then per run the int8 prepare of the joined prepare's plane and the int8 GEMM to z (`trunk_q8.piecePrepared`),
//! the joined finish over z as before.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

/// The bytes a layer's kept W_hat hold (`Hats`), billed in the prompt term: every group's
/// in x out halves, `heads * head_dim * o_lora * 2` (og x (heads / og x head_dim) x o_lora x 2).
pub fn heldBytes(heads: u64, head_dim: u64, o_lora: u64) u64 {
    return heads * head_dim * o_lora * 2;
}

/// The most groups a run takes (the finish binds one input a group).
pub const max_groups = 8;

// ── the kernels: turboexl3's texts at the pin, re-addressed ──

const Sub = struct { old: []const u8, new: []const u8 };

/// prep.metal's three index expressions on group gz's columns of x [rows, G, DIM], its range of suh [G, DIM] and its
/// plane of xp [G, rows, DIM].
const prep_subs = [_]Sub{
    .{ .old = "x[row * DIMC + i]", .new = "x[row * (uint(G) * DIMC) + gz * DIMC + i]" },
    .{ .old = "suh[i]", .new = "suh[gz * DIMC + i]" },
    .{ .old = "xp[row * DIMC + dst]", .new = "xp[(gz * threadgroups_per_grid.y + row) * DIMC + dst]" },
};
const prep_prologue = "const uint gz = threadgroup_position_in_grid.z;\n";

/// finish.metal's z read from group gz's inner product (inputs z0..z7, [rows, DIM] each), its range of svh [G, DIM], and
/// its y row's group gz columns of y [rows, G DIM].
const finish_subs = [_]Sub{
    .{ .old = "svh[o]", .new = "svh[gz * DIMC + o]" },
    .{ .old = "y[row * DIMC + o]", .new = "y[row * (uint(G) * DIMC) + gz * DIMC + o]" },
};
const finish_prologue = "const uint gz = threadgroup_position_in_grid.z;\n" ++
    "const device float16_t* z = gz == 0u ? z0 : gz == 1u ? z1 : gz == 2u ? z2 : gz == 3u ? z3 : gz == 4u ? z4 : gz == 5u ? z5 : gz == 6u ? z6 : z7;\n";

/// `stock` with each `subs` expression (present exactly once, or refused) replaced, after `prologue`.
fn derive(a: std.mem.Allocator, stock: []const u8, prologue: []const u8, subs: []const Sub) ![:0]u8 {
    for (subs) |sb| if (std.mem.count(u8, stock, sb.old) != 1) return error.WoaJoinDerive;
    var body = stock;
    var owned: ?[]u8 = null;
    defer if (owned) |o| a.free(o);
    for (subs) |sb| {
        const next = try std.mem.replaceOwned(u8, a, body, sb.old, sb.new);
        if (owned) |o| a.free(o);
        owned = next;
        body = next;
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, prologue);
    try out.appendSlice(a, body);
    return out.toOwnedSliceSentinel(a, 0);
}

pub const Kernels = struct { prep: ?mlx.mlx_fast_metal_kernel = null, finish: ?mlx.mlx_fast_metal_kernel = null };
pub var kernels: Kernels = .{};

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const prep_ins: []const [*:0]const u8 = &.{ "x", "suh" };
const prep_outs: []const [*:0]const u8 = &.{"xp"};
const finish_ins: []const [*:0]const u8 = &.{ "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "svh" };
const finish_outs: []const [*:0]const u8 = &.{"y"};

/// The two kernels from turboexl3's texts (built once; `deinit` at the model's).
pub fn build(a: std.mem.Allocator) !void {
    if (kernels.prep != null) return;
    const pt = try derive(a, t3.dense.text_prep, prep_prologue, &prep_subs);
    defer a.free(pt);
    const ft = try derive(a, t3.dense.text_finish, finish_prologue, &finish_subs);
    defer a.free(ft);
    const p = try newKernel("dsv41_woa_join_prep", prep_ins, prep_outs, pt.ptr, t3.decode.header.ptr);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(p);
    const f = try newKernel("dsv41_woa_join_finish", finish_ins, finish_outs, ft.ptr, t3.decode.header.ptr);
    kernels = .{ .prep = p, .finish = f };
}

pub fn deinit() void {
    inline for (.{ kernels.prep, kernels.finish }) |k| if (k) |kk| {
        _ = mlx.mlx_fast_metal_kernel_free(kk);
    };
    kernels = .{};
}

// ── a run ──

/// A layer's kept W_hat, one per group (turboexl3's `Dense.dequantize` of the group's trellis), decoded at the layer's
/// first joined run and freed at its end (`deinit`); with `int8`, each group's int8 weights instead (`trunk_q8.decodeW`).
pub const Hats = struct {
    trellis: A = none,
    w: [max_groups]A = @splat(none),
    n: usize = 0,
    int8: bool = false,
    pub fn deinit(h: *Hats) void {
        for (h.w[0..h.n]) |x| _ = mlx.mlx_array_free(x);
        h.* = .{};
    }
};

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// An int template argument.
const TArg = struct { name: [*:0]const u8, v: c_int };

fn launch(k: mlx.mlx_fast_metal_kernel, ins: []const A, out_shape: []const c_int, out_dt: mlx.mlx_dtype, grid: [3]c_int, ints: []const TArg, dt_name: ?[*:0]const u8, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, out_shape.ptr, out_shape.len, out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.v));
    if (dt_name) |n| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, n, dt));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&o, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return o;
}

/// `t3.linearGrouped(s, x, p, .mul1)` for a run's rows `x` [rows, G, in] (G <= `max_groups`) as y [rows, G out] in x's
/// dtype (a new array the caller owns), each group's W_hat taken from `hats` (decoded there at the layer's first run).
/// Null (nothing launched, `hats` untouched) where the stock call takes another route.
pub fn run(s: mlx.mlx_stream, x: A, p: t3.Proj, hats: *Hats) !?A {
    const kp = kernels.prep orelse return null;
    const kf = kernels.finish orelse return null;
    if (!mlx.streamIsGpu(s)) return null;
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(p.trellis);
    if (xsh.len != 3 or tsh.len != 4) return null;
    const g: usize = @intCast(tsh[0]);
    const in: usize = @intCast(tsh[1] * 16);
    const out: usize = @intCast(tsh[2] * 16);
    if (g < 1 or g > max_groups or xsh[1] != tsh[0] or xsh[2] != tsh[1] * 16 or xsh[0] < 1) return null;
    const rows: usize = @intCast(xsh[0]);
    const shape: t3.grouped.Shape = .{ .groups = @intCast(g), .in = @intCast(in), .out = @intCast(out), .rate = t3.format.Rate.fromN(@intCast(tsh[3])) orelse return null, .dec = .mul1, .out_dtype = mlx.mlx_array_dtype(x) };
    if (!shape.admitted()) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return null;
    // The stock call's route at these rows: groupedPrefill past the decode rows, each group's `linear` on the matmul path
    // (not the register path or turboexl3's fused-GEMM arm).
    if (rows <= t3.kernels.DECODE_ROWS_MAX) return null;
    const d = try t3.denseCached(.{ .in = @intCast(in), .out = @intCast(out), .rate = shape.rate, .dec = .mul1, .out_dtype = xd });
    if (d.pathFor(rows) != .matmul) return null;
    // turboexl3's trunk int8 on the split-off inner products at the group's shape (TURBOEXL3_DS41_PF_INT8 3 or 4)
    const q8 = t3.trunk_q8.innerTakes(d);
    if (hats.n != 0 and (hats.trellis.ctx != p.trellis.ctx or hats.int8 != q8)) return error.WoaJoinHats;

    // One prepare launch: xp half [G, rows, in] (prep.metal's natural order, LANE_ORDER 0).
    const xp = try launch(kp, &.{ x, p.suh }, &.{ ci(g), ci(rows), ci(in) }, .float16, .{ ci(in / 128 * 32), ci(rows), ci(g) }, &.{ .{ .name = "DIM", .v = ci(in) }, .{ .name = "LANE_ORDER", .v = 0 }, .{ .name = "G", .v = ci(g) } }, null, .float16, s);
    defer _ = mlx.mlx_array_free(xp);

    // The groups' W_hat, decoded once a layer (the per-group `linear`'s own dequantize of its trellis slice), or on
    // trunk int8 its int8 weights (`root.linearPiecesBegin`'s int8 decode).
    if (hats.n == 0) {
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_reshape(&t, p.trellis, &[_]c_int{ tsh[0] * tsh[1], tsh[2], tsh[3] }, 3, s));
        hats.trellis = p.trellis;
        hats.int8 = q8;
        for (0..g) |gi| {
            const lo: c_int = @intCast(gi);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_slice(&v, t, &[_]c_int{ lo * tsh[1], 0, 0 }, 3, &[_]c_int{ (lo + 1) * tsh[1], tsh[2], tsh[3] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            hats.w[gi] = if (q8) try t3.trunk_q8.decodeW(s, d, v) else try d.dequantize(s, v);
            hats.n = gi + 1;
        }
    }

    // MLX's matmul of each group: xp[g] [rows, in] (a contiguous plane) x W_hat [in, out], half inner [rows, out].
    var z: [max_groups]A = @splat(none);
    var made: usize = 0;
    defer for (z[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..g) |gi| {
        const lo: c_int = @intCast(gi);
        var plane = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(plane);
        try mlx.check(mlx.mlx_slice(&plane, xp, &[_]c_int{ lo, 0, 0 }, 3, &[_]c_int{ lo + 1, ci(rows), ci(in) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var xg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xg);
        try mlx.check(mlx.mlx_reshape(&xg, plane, &[_]c_int{ ci(rows), ci(in) }, 2, s));
        if (q8) {
            // the plane's int8 prepare and the int8 GEMM to z half [rows, out], trunk int8's own launches
            z[gi] = try t3.trunk_q8.piecePrepared(d, s, xg, hats.w[gi], rows);
            made = gi + 1;
            continue;
        }
        z[gi] = mlx.mlx_array_new();
        made = gi + 1;
        try mlx.check(mlx.mlx_matmul(&z[gi], xg, hats.w[gi], s));
    }
    // Groups past G bind group 0's inner product (never read).
    var zin: [max_groups + 1]A = undefined;
    for (0..max_groups) |gi| zin[gi] = if (gi < g) z[gi] else z[0];
    zin[max_groups] = p.svh;

    // One finish launch: y [rows, G out] in x's dtype, group g's columns g out .. (g + 1) out of each row.
    return try launch(kf, &zin, &.{ ci(rows), ci(g * out) }, xd, .{ ci(out / 128 * 32), ci(rows), ci(g) }, &.{ .{ .name = "DIM", .v = ci(out) }, .{ .name = "G", .v = ci(g) } }, "OUT_T", xd, s);
}

/// One evaluated joined run on `rows` zero rows of `p`'s shape (its W_hat decoded and dropped): the launches compiled at
/// the load, so a prompt's first wide layer pays no JIT.
pub fn warm(s: mlx.mlx_stream, p: t3.Proj, dt: mlx.mlx_dtype) !void {
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 4) return;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 64, tsh[0], tsh[1] * 16 }, 3, dt, s));
    var h: Hats = .{};
    defer h.deinit();
    const y = (try run(s, x, p, &h)) orelse return;
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
}

