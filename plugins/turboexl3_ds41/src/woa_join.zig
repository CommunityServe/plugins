//! `MLX_SERVE_DSV41_WOA_JOIN` (read at load; unset, empty or `1`: on; `0`: off; `check`: on and checked; anything else
//! is refused by name): a wide prompt layer's EXL3 `wo_a` run (MLX_SERVE_DSV41_WOA_ROWS' run of spans, read from its
//! MLX_SERVE_DSV41_SCRATCH_HEADS region [rows, og, in]) without turboexl3's per-group copies, launches and decodes.
//!
//! turboexl3's grouped linear at prefill rows (root.zig `groupedPrefill`) runs `linear` on each of the og groups: a
//! strided column slice of the run's rows, which MLX copies whole for the prepare (its input is read row-contiguous);
//! the prepare; the group's W_hat decoded from its trellis; MLX's matmul; the finish; then one concatenation of the og
//! outputs. At the pack's shape (og 8, in 4096, out 1024) that is 41 launches a run, every group's rows copied once, and
//! each of a 16K layer's 4 runs decoding the same 8 W_hat again.
//!
//! On: one prepare launch for every group, reading the run's rows in place (`prep_text`: turboexl3's prep.metal at the
//! pin, its value statements unchanged, its three index expressions moved to the group's columns, suh range and output
//! plane); each group's W_hat decoded once a layer by turboexl3's own `Dense.dequantize` (the launch and config the
//! per-group `linear` uses) and kept across the layer's runs (`Hats`); MLX's matmul of each group on the same operands
//! (M, N, K and bytes the per-group call's); one finish launch for every group writing straight into the run's
//! [rows, og out] (`finish_text`: finish.metal, likewise re-addressed). No slice copy, no concatenation: 10 launches a
//! run, 18 at a layer's first. The same words: every value statement is turboexl3's and every operand the same bytes
//! (prep and finish have no multiply-add a compiler could contract). A run another path would take (decode or
//! register-path rows, turboexl3's fused-GEMM arm) keeps the stock call.
//!
//! The kept W_hat (heads x head_dim x o_lora halves, 67 MB at the pack's shape) is billed in the prompt term only
//! (`heldBytes`, the copied model's `prefillBytes`, for calls wider than a span: decode's calls are never wide, so the
//! decode bill holds none). `check`: both on every run, the stock run served, its words and the joined run's compared, a
//! `[woa-join] check` line a layer.
//!
//! On against `=0` (wj1, w-pio-20261102-0405, two binaries: a = ctl26 (2d0a33f, zzz8x), b = this code on zzz8x with the
//! switch off by default, dylib 061842336690, `=1` in the b arms): cold 16K a 1409.1/1406.1 -> b 1446.9/1440.6 tok/s
//! (+2.57%, pairs +2.68% and +2.45%), new text (slice 3) 1496.6/1495.2 -> 1518.5/1536.0 (+2.10%, pairs +1.46% and
//! +2.73%); 152 prompt rows in every arm; the guard's peak 111.0 GiB. Its bill then also counted the kept W_hat in the
//! decode term (ws_decode_mb 804 -> 884, decode rows 179 -> 178); billed in the prompt term only here, decode keeps 179.
//! Exact: the texts the same bytes in every arm, and the check arm's 41 wide-layer lines compared 4,026,531,840 words,
//! none different (max_ulps 0).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Mode = enum { off, on, check };
pub var mode: Mode = .on;

pub const Error = error{WoaJoinSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.WoaJoinSetting;
}

/// The bytes a layer's kept W_hat hold (`Hats`), billed in the prompt term while the switch is on: every group's
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
/// first joined run and freed at its end (`deinit`).
pub const Hats = struct {
    trellis: A = none,
    w: [max_groups]A = @splat(none),
    n: usize = 0,
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
    if (hats.n != 0 and hats.trellis.ctx != p.trellis.ctx) return error.WoaJoinHats;

    // One prepare launch: xp half [G, rows, in] (prep.metal's natural order, LANE_ORDER 0).
    const xp = try launch(kp, &.{ x, p.suh }, &.{ ci(g), ci(rows), ci(in) }, .float16, .{ ci(in / 128 * 32), ci(rows), ci(g) }, &.{ .{ .name = "DIM", .v = ci(in) }, .{ .name = "LANE_ORDER", .v = 0 }, .{ .name = "G", .v = ci(g) } }, null, .float16, s);
    defer _ = mlx.mlx_array_free(xp);

    // The groups' W_hat, decoded once a layer (the per-group `linear`'s own dequantize of its trellis slice).
    if (hats.n == 0) {
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_reshape(&t, p.trellis, &[_]c_int{ tsh[0] * tsh[1], tsh[2], tsh[3] }, 3, s));
        hats.trellis = p.trellis;
        for (0..g) |gi| {
            const lo: c_int = @intCast(gi);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_slice(&v, t, &[_]c_int{ lo * tsh[1], 0, 0 }, 3, &[_]c_int{ (lo + 1) * tsh[1], tsh[2], tsh[3] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            hats.w[gi] = try d.dequantize(s, v);
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

// ── check ──

/// A layer's `check` count: its runs, those equal word for word, the words that differ, the largest distance between two
/// words as sign-ordered integers.
pub const Tally = struct { runs: u64 = 0, equal: u64 = 0, words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0 };
pub var tally: Tally = .{};

/// `got` (the joined run) against `want` (the stock call's output, the same words in [rows, G, out]) into `tally`.
pub fn compare(s: mlx.mlx_stream, got: A, want: A) !void {
    // The stock call's [rows, G, out] as the joined [rows, G out] (the same row-major words).
    const gs = mlx.getShape(got);
    var w2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w2);
    try mlx.check(mlx.mlx_reshape(&w2, want, gs.ptr, gs.len, s));
    const st = try compareWords(s, got, w2);
    tally.runs += 1;
    tally.equal += @intFromBool(st.differ == 0);
    tally.words += st.words;
    tally.differ += st.differ;
    tally.max_ulps = @max(tally.max_ulps, st.max_ulps);
}

/// The layer's line, then its count reset.
pub fn report(layer: usize) void {
    std.debug.print("[woa-join] check layer={d} runs={d} equal={d} words={d} differ={d} max_ulps={d}\n", .{ layer, tally.runs, tally.equal, tally.words, tally.differ, tally.max_ulps });
    tally = .{};
}

const Stats = struct { words: u64 = 0, differ: u64 = 0, max_ulps: u64 = 0 };

/// `a` against `b` (one size, 2- or 4-byte float words), word for word on the device: the equal words counted first, the
/// ulp distance only when some differ.
fn compareWords(s: mlx.mlx_stream, a: A, b: A) !Stats {
    const isz = mlx.mlx_array_itemsize(b);
    if (mlx.mlx_array_itemsize(a) != isz or mlx.mlx_array_size(a) != mlx.mlx_array_size(b)) return error.WoaJoinCheckShape;
    const st: mlx.mlx_dtype, const ut: mlx.mlx_dtype, const lo: i32 = switch (isz) {
        2 => .{ .int16, .uint16, -32768 },
        4 => .{ .int32, .uint32, std.math.minInt(i32) },
        else => return error.WoaJoinCheckWords,
    };
    const n: u64 = mlx.mlx_array_size(b);
    var t: [5]A = undefined;
    for (&t) |*v| v.* = mlx.mlx_array_new();
    defer for (t) |v| {
        _ = mlx.mlx_array_free(v);
    };
    try mlx.check(mlx.mlx_view(&t[0], a, ut, s));
    try mlx.check(mlx.mlx_view(&t[1], b, ut, s));
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
    try mlx.check(mlx.mlx_view(&u[0], a, st, s));
    try mlx.check(mlx.mlx_view(&u[1], b, st, s));
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
