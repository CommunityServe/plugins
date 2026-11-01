//! `MLX_SERVE_DSV41_TAIL_FUSE` (read at load; unset, empty or `1`: on; `0`: off; `check`: the chain served, the kernel
//! checked against it): a wide prompt layer's tail as one kernel (kernels/tail_fuse.metal). Off, the layer's routed call
//! ends in the joinless combine (the routed rows weighted and folded, plus the shared expert widened to f32, into an f32
//! [rows, D]), the model rounds that to the activation dtype and the hc expand (glm5's kernel) mixes it into the layer's
//! HC streams: three GB-sized transients a 16K-row layer (the widened shared rows, the f32 sum, its rounded copy, ~840
//! MB) and four passes over them. On, the source runs one kernel over the same inputs (the waves' outputs where they
//! wrote them, the routing weights, the shared expert as it came, the streams, each row's post and comb) and writes the
//! new streams: every rounding where the chain takes it, so the same words. `check` serves the chain's streams and runs
//! the kernel over each wide call's first and last `check_rows` rows, compared word for word with the chain's: a
//! `[tail-fuse] check` line a call (the words compared, those that differ, the largest difference); the chain's memory
//! and a few hundred MB more, so it fits a window's guard. The kernel is the joinless combine's: the setting follows
//! `MLX_SERVE_DSV41_JOINLESS` (unset, off with it; set, refused without it).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");

const A = mlx.mlx_array;

pub const Mode = enum { off, on, check };
pub var mode: Mode = .on;
/// The setting was given (`configure`): with JOINLESS off the load refuses it; unset, it follows JOINLESS.
pub var explicit: bool = false;

pub const Error = error{TailFuseSetting};

pub fn on() bool {
    return mode != .off;
}

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
    explicit = if (value) |v| v.len > 0 else false;
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.TailFuseSetting;
}

/// The routed sources' slots in the kernel (the joinless combine's).
pub const sources = 24;

const header = @embedFile("kernels/tail_fuse_header.metal");
const body = @embedFile("kernels/tail_fuse.metal");
var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dimOf(x: A, axis: usize) usize {
    const sh = mlx.getShape(x);
    return if (axis < sh.len) @intCast(sh[axis]) else 0;
}

fn theKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    var names: [sources + 5][*:0]const u8 = undefined;
    const src_names = [_][*:0]const u8{ "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23" };
    for (names[0..sources], src_names) |*n, s| n.* = s;
    names[sources..].* = .{ "loc", "w", "sh", "res", "pc" };
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const outs = [_][*:0]const u8{"out"};
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("turboexl3_ds41_tail_fuse", iv, ov, body, header, true, false);
    if (k.ctx == null) return error.TailFuseKernel;
    kernel = k;
    return k;
}

/// The new streams [n, HC, D] (`res`'s dtype), a new array the caller owns: the joinless combine of `outs` (1..24 routed
/// output arrays, one dtype) at `loc` (int32 [n, K, 2]) with `w` (f32 [n, K]) plus `shared` ([n, D]), rounded to `res`'s
/// dtype, then the hc expand of `res` ([n, HC, D]) by `pc` (f32 [n, HC + HC * HC]: each row's post, then comb).
pub fn expand(s: mlx.mlx_stream, outs: []const A, loc: A, w: A, shared: A, res: A, pc: A) !A {
    if (outs.len == 0 or outs.len > sources) return error.TailFuseSources;
    const n = dimOf(res, 0);
    const hc = dimOf(res, 1);
    const d = dimOf(res, 2);
    const k = dimOf(w, 1);
    if (n == 0 or d % 4 != 0 or dimOf(w, 0) != n or dimOf(shared, 0) != n or dimOf(shared, 1) != d or dimOf(pc, 0) != n or dimOf(pc, 1) != hc + hc * hc)
        return error.TailFuseShape;
    if (mlx.mlx_array_dtype(w) != .float32 or mlx.mlx_array_dtype(pc) != .float32 or mlx.mlx_array_dtype(loc) != .int32)
        return error.TailFuseDtype;
    const kern = try theKernel();
    var ins: [sources + 5]A = undefined;
    for (ins[0..sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
    ins[sources..].* = .{ loc, w, shared, res, pc };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ ci(n), ci(hc), ci(d) };
    const dt = mlx.mlx_array_dtype(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(n * (d / 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(k)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", ci(d)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", ci(hc)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NROWS", ci(n)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "S", mlx.mlx_array_dtype(outs[0])));
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// The chain's new streams for `check`, a new array the caller owns: the combine's f32 sum `y` ([n, D]) rounded to
/// `res`'s dtype, then glm5's hc expand of `res` by `pc`'s post and comb (the model's `hcExpand`).
pub fn chain(s: mlx.mlx_stream, y: A, res: A, pc: A) !A {
    const n = dimOf(res, 0);
    const hc = dimOf(res, 1);
    const d = dimOf(res, 2);
    const dt = mlx.mlx_array_dtype(res);
    var tmp: [6]A = undefined;
    for (&tmp) |*t| t.* = mlx.mlx_array_new();
    defer for (tmp) |t| {
        _ = mlx.mlx_array_free(t);
    };
    try mlx.check(mlx.mlx_astype(&tmp[0], y, dt, s));
    try mlx.check(mlx.mlx_expand_dims(&tmp[1], tmp[0], 0, s));
    try mlx.check(mlx.mlx_reshape(&tmp[2], res, &[_]c_int{ 1, ci(n), ci(hc), ci(d) }, 4, s));
    try mlx.check(mlx.mlx_slice(&tmp[3], pc, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), ci(hc) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var comb_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(comb_flat);
    try mlx.check(mlx.mlx_slice(&comb_flat, pc, &[_]c_int{ 0, ci(hc) }, 2, &[_]c_int{ ci(n), ci(hc + hc * hc) }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_reshape(&tmp[4], comb_flat, &[_]c_int{ ci(n), ci(hc), ci(hc) }, 3, s));
    const c: glm5.Collapsed = .{ .x = .{ .ctx = null }, .post = tmp[3], .comb = tmp[4] };
    const o = try glm5.expand(s, tmp[1], tmp[2], &c);
    defer _ = mlx.mlx_array_free(o);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ ci(n), ci(hc), ci(d) }, 3, s));
    return out;
}

/// `check`'s rows at each end of a wide call.
pub const check_rows: usize = 2048;

fn sliceRows(s: mlx.mlx_stream, x: A, lo: usize, hi: usize) !A {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    var step: [4]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[0] = ci(lo);
    stop[0] = ci(hi);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &step, sh.len, s));
    return o;
}

/// `check`: the kernel over the first and the last `check_rows` rows of the call (its inputs' rows, the same sources),
/// each compared with the chain's streams `ref` there, word for word on the device; one `[tail-fuse] check` line.
pub fn checkRows(s: mlx.mlx_stream, layer: u32, outs: []const A, loc: A, w: A, shared: A, res: A, pc: A, ref: A) !void {
    const n = dimOf(ref, 0);
    var words: u64 = 0;
    var differ: u64 = 0;
    var max_abs: f32 = 0;
    const first = @min(n, check_rows);
    const windows = [2][2]usize{ .{ 0, first }, .{ @max(first, n -| check_rows), n } };
    for (windows) |win| {
        const lo, const hi = win;
        if (hi <= lo) continue;
        var ins: [6]A = @splat(.{ .ctx = null });
        defer for (ins) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (&ins, [_]A{ loc, w, shared, res, pc, ref }) |*o, x| o.* = try sliceRows(s, x, lo, hi);
        const fused = try expand(s, outs, ins[0], ins[1], ins[2], ins[3], ins[4]);
        defer _ = mlx.mlx_array_free(fused);
        var tmp: [9]A = @splat(.{ .ctx = null });
        defer for (tmp) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (&tmp) |*x| x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(&tmp[0], fused, .uint16, s));
        try mlx.check(mlx.mlx_view(&tmp[1], ins[5], .uint16, s));
        try mlx.check(mlx.mlx_equal(&tmp[2], tmp[0], tmp[1], s));
        try mlx.check(mlx.mlx_astype(&tmp[3], tmp[2], .uint32, s));
        try mlx.check(mlx.mlx_sum(&tmp[4], tmp[3], false, s));
        try mlx.check(mlx.mlx_astype(&tmp[5], fused, .float32, s));
        try mlx.check(mlx.mlx_astype(&tmp[6], ins[5], .float32, s));
        try mlx.check(mlx.mlx_subtract(&tmp[7], tmp[5], tmp[6], s));
        var ad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ad);
        try mlx.check(mlx.mlx_abs(&ad, tmp[7], s));
        try mlx.check(mlx.mlx_max(&tmp[8], ad, false, s));
        try mlx.check(mlx.mlx_array_eval(tmp[4]));
        try mlx.check(mlx.mlx_array_eval(tmp[8]));
        const same = (mlx.mlx_array_data_uint32(tmp[4]) orelse return error.NoData)[0];
        const m = (mlx.mlx_array_data_float32(tmp[8]) orelse return error.NoData)[0];
        const nw: u64 = mlx.mlx_array_size(ins[5]);
        words += nw;
        differ += nw - same;
        max_abs = @max(max_abs, m);
    }
    std.debug.print("[tail-fuse] check layer={d} rows={d} words={d} differ={d} max_abs={e}\n", .{ layer, n, words, differ, max_abs });
}

test "tail_fuse setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.on, try parse("1"));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "true", "Check", " 1" }) |v| try std.testing.expectError(error.TailFuseSetting, parse(v));
}
