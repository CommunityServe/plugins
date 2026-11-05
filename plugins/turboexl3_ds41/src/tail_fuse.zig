//! A wide prompt layer's tail as one kernel (kernels/tail_fuse.metal). Unfused, the layer's routed call ends in the
//! joinless combine (the routed rows weighted and folded, plus the shared expert widened to f32, into an f32 [rows, D]),
//! the model rounds that to the activation dtype and the hc expand (glm5's kernel) mixes it into the layer's HC streams:
//! three GB-sized transients a 16K-row layer (the widened shared rows, the f32 sum, its rounded copy, ~840 MB) and four
//! passes over them. Here the source runs one kernel over the same inputs (the waves' outputs where they wrote them, the
//! routing weights, the shared expert as it came, the streams, each row's post and comb) and writes the new streams:
//! every rounding where the chain takes it, so the same words.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");

const A = mlx.mlx_array;

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

/// turboexl3_ds41 (the decode tail fuse): a decode call's rows take one instance of the kernel, its NROWS (the grid
/// guard's bound, never reached: the grid is the rows' columns exactly) at the most rows a decode stretch carries.
pub const decode_rows: usize = 16;

/// `expand` for a decode call (at most `decode_rows` rows): the same kernel and inputs, its NROWS `decode_rows` whatever
/// the call's rows, so every decode call shares one compiled instance (`expand`'s NROWS is the call's rows: a prompt
/// call's own). The same words: NROWS bounds the grid's rows only.
pub fn expandDecode(s: mlx.mlx_stream, outs: []const A, loc: A, w: A, shared: A, res: A, pc: A) !A {
    if (outs.len == 0 or outs.len > sources) return error.TailFuseSources;
    const n = dimOf(res, 0);
    const hc = dimOf(res, 1);
    const d = dimOf(res, 2);
    const k = dimOf(w, 1);
    if (n == 0 or n > decode_rows or d % 4 != 0 or dimOf(w, 0) != n or dimOf(shared, 0) != n or dimOf(shared, 1) != d or dimOf(pc, 0) != n or dimOf(pc, 1) != hc + hc * hc)
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
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NROWS", ci(decode_rows)));
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

/// The kernel's write-into twin, its text derived from the stock
/// one at its first use: each row's new streams stored over `res` itself (the model's x1 region: every thread reads its
/// row's HC copies of `res` into registers before it stores the row's HC outputs at those words, and no thread reads
/// another's), and a one-word `carrier` (the output the region's publish waits behind) zeroed by the first thread; every
/// other statement the stock text's. It binds the stock kernel's inputs and one output (`kernel_buffers`, 30, as the
/// stock's): Metal's argument table holds 31 (sx3's twin bound a `dst` and an `after` beside them, 32, and did not
/// compile). A missed anchor refuses by name.
var into_kernel: ?mlx.mlx_fast_metal_kernel = null;
const into_head_old = "const uint g = thread_position_in_grid.x;\n";
const into_head_new = "const uint g = thread_position_in_grid.x;\nif (g == 0) carrier[0] = 0;\n";
const into_store_old = "        out[(ulong(row) * ulong(HC) + ulong(k2)) * ulong(D) + ulong(col) + i] = static_cast<T>(metal::fma(pr[k2], y[i], acc));";
const into_store_new = "        ((device T*)res)[(ulong(row) * ulong(HC) + ulong(k2)) * ulong(D) + ulong(col) + i] = static_cast<T>(metal::fma(pr[k2], y[i], acc));";

/// The buffers one Metal kernel binds: its argument table's indices 0 to 30.
pub const metal_buffers = 31;
/// What the stock kernel and its twin each bind: the sources, loc, w, sh, res and pc, and one output.
pub const kernel_buffers = sources + 5 + 1;
comptime {
    if (kernel_buffers > metal_buffers) @compileError("tail_fuse: the kernel binds more buffers than Metal's argument table holds");
}

pub const IntoError = error{ TailFuseIntoAnchor, OutOfMemory };

fn replaceOnce(a: std.mem.Allocator, s: []const u8, old: []const u8, new: []const u8) IntoError![]u8 {
    if (std.mem.count(u8, s, old) != 1) return error.TailFuseIntoAnchor;
    const at = std.mem.indexOf(u8, s, old).?;
    return std.mem.concat(a, u8, &.{ s[0..at], new, s[at + old.len ..] });
}

/// The twin's text from the stock one (each anchor exactly once).
pub fn intoSource(a: std.mem.Allocator, stock: []const u8) IntoError![:0]u8 {
    const s1 = try replaceOnce(a, stock, into_head_old, into_head_new);
    defer a.free(s1);
    const s2 = try replaceOnce(a, s1, into_store_old, into_store_new);
    defer a.free(s2);
    return a.dupeSentinel(u8, s2, 0);
}

fn intoKernel() !mlx.mlx_fast_metal_kernel {
    if (into_kernel) |k| return k;
    const a = std.heap.page_allocator;
    const text = try intoSource(a, body);
    defer a.free(text);
    var names: [sources + 5][*:0]const u8 = undefined;
    const src_names = [_][*:0]const u8{ "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23" };
    for (names[0..sources], src_names) |*n, s| n.* = s;
    names[sources..].* = .{ "loc", "w", "sh", "res", "pc" };
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const outs = [_][*:0]const u8{"carrier"};
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("turboexl3_ds41_tail_fuse_into", iv, ov, text.ptr, header, true, false);
    if (k.ctx == null) return error.TailFuseKernel;
    into_kernel = k;
    return k;
}

/// `expand`'s new streams written over `res` in place (`res` row-contiguous in a buffer of
/// its own: the model's published x1 region), ordered after its inputs' writers by its bindings. Returns the one-word
/// carrier (a new array the caller owns); `res`'s words are the new streams once the carrier is evaluated.
pub fn expandInto(s: mlx.mlx_stream, outs: []const A, loc: A, w: A, shared: A, res: A, pc: A) !A {
    if (outs.len == 0 or outs.len > sources) return error.TailFuseSources;
    const n = dimOf(res, 0);
    const hc = dimOf(res, 1);
    const d = dimOf(res, 2);
    const k = dimOf(w, 1);
    if (n == 0 or d % 4 != 0 or dimOf(w, 0) != n or dimOf(shared, 0) != n or dimOf(shared, 1) != d or dimOf(pc, 0) != n or dimOf(pc, 1) != hc + hc * hc)
        return error.TailFuseShape;
    if (mlx.mlx_array_dtype(w) != .float32 or mlx.mlx_array_dtype(pc) != .float32 or mlx.mlx_array_dtype(loc) != .int32)
        return error.TailFuseDtype;
    const kern = try intoKernel();
    var ins: [sources + 5]A = undefined;
    for (ins[0..sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
    ins[sources..].* = .{ loc, w, shared, res, pc };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const one = [_]c_int{1};
    const dt = mlx.mlx_array_dtype(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &one, one.len, .uint32));
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

test "tail_fuse into twin: each anchor once in the stock text; the store goes over res and the carrier is set" {
    const t = try intoSource(std.testing.allocator, body);
    defer std.testing.allocator.free(t);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, "((device T*)res)["));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, "if (g == 0) carrier[0] = 0;"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, "        out[("));
    try std.testing.expectEqual(body.len + (into_head_new.len - into_head_old.len) + (into_store_new.len - into_store_old.len), t.len);
    try std.testing.expectError(error.TailFuseIntoAnchor, intoSource(std.testing.allocator, "no anchors here"));
}

test "tail_fuse: the stock kernel and its twin bind 30 buffers, within Metal's 31" {
    try std.testing.expectEqual(@as(usize, 30), kernel_buffers);
    try std.testing.expect(kernel_buffers <= metal_buffers);
}
