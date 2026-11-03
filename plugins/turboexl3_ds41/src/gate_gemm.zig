//! `MLX_SERVE_DSV41_GATE_GEMM` (read at load; unset, empty or `1`: on; `0`: off; `check`: the chain served and the
//! launch checked against it; anything else is refused by name): the wide router's f32 logits as one launch of MLX's
//! fused NAX GEMM body at this machine's tile (hgf3u's pick on the M5 Max).
//!
//! The copied model's `gateLogits` past the decode width runs `matmul(astype(x, f32), transpose(astype(gate_w, f32)))`,
//! x bf16 [rows, 5,120] and gate_w [384, 5,120] (F16 on the pack): two casts and MLX's NAX matmul (MLX_ENABLE_TF32
//! unset: 1), its regular fused kernel from 1,707 rows (K < 3 max(M, N)), its split K below. GATE_GEMM runs the chain as
//! one launch: MLX's fused NAX GEMM kernel at nt (steel/gemm/kernels/steel_gemm_fused_nax.h `gemm`, MLX 0.32.3:
//! src/kernels/gate_gemm_nt.metal) and its gemm_loop (src/kernels/gate_gemm_loop.metal: verbatim but A's and B's memory
//! types), reading x and gate_w in their stored types and static_casting each element to float as it loads (the values
//! astype makes, bf16 and f16 alike: no cast passes), the statements of MLX's kernel at this machine's tile (bm 64, bn 128,
//! bk 256, wm 2, wn 4, swizzle 1; MLX's own here is swizzle 2): each simdgroup's tile whole 16-row fragments and N pairs
//! of 16 columns and BK whole SK 32 steps, so every output's k order is MLX's (the 16 x 32 x 16 MMA from k = 0, whatever
//! the tile) and the words are the chain's. hgf3u (w-hgf3u-20261103-1359, src/f3_bench.zig in hg1 lane's bench tree, bf16
//! operands): 0 differing words at 16,383, 4,097 and 2,048 rows; per call 1.773x at 16,383 rows (the chain 2,719 us, the
//! launch 1,536), 1.731x at 4,097 and 1.525x at 2,048. A call is taken only where MLX runs its regular kernel on the NAX
//! path; elsewhere (fewer rows: decode's GEMV and the CED decoder's 128 rows on MLX's split K; no NAX; other shapes or
//! types) the chain. `check`: the chain served and, at every call GATE_GEMM would take, its launch beside it compared word
//! for word (one host read a call): `[gate-gemm] mode= calls= checks= words= differ=` at each request's end (calls:
//! launches; checks, words, differ: check only). On against ctl55 (hggg2, w-pio-20261103-1626): 21 calls a 16K card,
//! the check arm's words none different, cold +0.65% and new +0.21% (README).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const steel = @import("mlx_steel_sources");

const A = mlx.mlx_array;

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

pub const Error = error{GateGemmSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
    reset();
}

/// `value`: null, empty or `1` on; `0` off; `check`; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.GateGemmSetting;
}

/// MLX's tile: BM x BN, WM x WN simdgroups, BK, swizzle_log.
pub const Tile = struct { bm: u16, bn: u16, bk: u16, wm: u8, wn: u8, swz: u8 };

/// hgf3u's pick (w-hgf3u-20261103-1359): the widening launch at this tile.
pub const tile: Tile = .{ .bm = 64, .bn = 128, .bk = 256, .wm = 2, .wn = 4, .swz = 1 };

/// Exact by construction: each simdgroup's tile whole 16-row fragments and N pairs (SN a multiple of 32: the MMA's
/// 16 x 32 pairs as MLX's), BK whole SK 32 steps, within 1,024 threads.
pub fn admitted(t: Tile) bool {
    if (t.wm == 0 or t.wn == 0 or t.bm % t.wm != 0 or t.bn % t.wn != 0) return false;
    const sm = t.bm / t.wm;
    const sn = t.bn / t.wn;
    return sm % 16 == 0 and sn % 32 == 0 and t.bk % 32 == 0 and t.bk > 0 and @as(u32, t.wm) * t.wn * 32 <= 1024 and t.swz <= 4;
}

comptime {
    if (!admitted(tile)) @compileError("gate_gemm: the tile is not exact by construction");
}

/// Whether MLX runs [m, k] x [k, n] on its regular fused NAX kernel, not its split K (turboexl3's `pgemm.mlxRegular`).
pub fn mlxRegular(m: u32, n: u32, k: u32) bool {
    const mx = @max(m, n);
    return !(k >= 3 * mx or (mx <= 1024 and k > 2 * mx));
}

fn kept(line: []const u8) bool {
    return !std.mem.startsWith(u8, line, "#pragma once") and !std.mem.startsWith(u8, line, "#include \"mlx/");
}

/// turboexl3's steel.zig `inlined`: `texts` in order, one line at a time, without the include guards and MLX's own
/// includes.
fn inlined(comptime texts: []const []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(20_000_000);
        var n: usize = 0;
        for (texts) |text| {
            var it = std.mem.splitScalar(u8, text, '\n');
            while (it.next()) |line| {
                if (kept(line)) n += line.len + 1;
            }
        }
        var out: [n]u8 = undefined;
        var at: usize = 0;
        for (texts) |text| {
            var it = std.mem.splitScalar(u8, text, '\n');
            while (it.next()) |line| {
                if (!kept(line)) continue;
                @memcpy(out[at..][0..line.len], line);
                out[at + line.len] = '\n';
                at += line.len + 1;
            }
        }
        const final = out;
        return &final;
    }
}

fn sentinel(comptime s: []const u8) [:0]const u8 {
    comptime {
        var buf: [s.len:0]u8 = undefined;
        @memcpy(buf[0..s.len], s);
        const final = buf;
        return &final;
    }
}

/// pg2's arrangement: MLX's steel NAX headers (the kit's), the loop, then MLX's kernel attribute on the generated
/// definition (`[[kernel]]` becomes `[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]`); the body's last line
/// undefines the macro, so the explicit instantiation MLX appends keeps its plain `[[kernel]]`.
const header: [:0]const u8 = sentinel(inlined(&.{ steel.defines, steel.type_traits, steel.integral_constant, steel.nax }) ++ "using namespace mlx::steel;\n" ++ @embedFile("kernels/gate_gemm_loop.metal") ++ "\n#define kernel kernel, max_total_threads_per_threadgroup(WM * WN * 32)\n");
const body: [:0]const u8 = sentinel(@embedFile("kernels/gate_gemm_nt.metal") ++ "\n#undef kernel\n");

var kern: ?mlx.mlx_fast_metal_kernel = null;

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kern) |k| return k;
    const ins = [_][*:0]const u8{ "a", "b" };
    const outs = [_][*:0]const u8{"y"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("turboexl3_ds41_gate_gemm", iv, ov, body.ptr, header.ptr, true, false);
    if (k.ctx == null) return error.GateGemmKernel;
    kern = k;
    return k;
}

/// A type the launch reads and widens to float exactly as astype does (`static_cast<float>`).
fn floatType(dt: mlx.mlx_dtype) bool {
    return dt == .bfloat16 or dt == .float16 or dt == .float32;
}

/// `a` [M, K] and `b` [N, K] (2-D, K equal, each in a float type of its own): (M, N, K); null otherwise.
fn shapeOf(a: A, b: A) ?[3]u32 {
    if (!floatType(mlx.mlx_array_dtype(a)) or !floatType(mlx.mlx_array_dtype(b))) return null;
    const as_ = mlx.getShape(a);
    const bs = mlx.getShape(b);
    if (as_.len != 2 or bs.len != 2 or as_[1] != bs[1] or as_[0] < 1 or bs[0] < 1 or as_[1] < 1) return null;
    return .{ @intCast(as_[0]), @intCast(bs[0]), @intCast(as_[1]) };
}

/// The launch at tile `t`: y f32 [M, N] = a [M, K] b [N, K]^T, the MMA in `mma`, the accumulators rounded to `out_r` and
/// widened, MLX's grid (tiles_n << swz by ceil(tiles_m >> swz) threadgroups) in MLX's threadgroup shape (32, WN, WM);
/// lazy, the caller owns y.
fn launch(s: mlx.mlx_stream, a: A, b: A, mma: mlx.mlx_dtype, out_r: mlx.mlx_dtype, t: Tile, mnk: [3]u32) !A {
    const m, const n, const kd = mnk;
    const swz: u32 = @as(u32, 1) << @intCast(t.swz);
    const tiles_m = (m + t.bm - 1) / t.bm;
    const tiles_n = (n + t.bn - 1) / t.bn;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const ys = [_]c_int{ @intCast(m), @intCast(n) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &ys, ys.len, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(tiles_n * swz * 32), @intCast(((tiles_m + swz - 1) / swz) * t.wn), @intCast(t.wm)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, @intCast(t.wn), @intCast(t.wm)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mma));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TA", mlx.mlx_array_dtype(a)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TB", mlx.mlx_array_dtype(b)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OUT_R", out_r));
    const ints = [_]struct { [*:0]const u8, u32 }{
        .{ "BM", t.bm },                         .{ "BN", t.bn },                         .{ "BK", t.bk },
        .{ "WM", t.wm },                         .{ "WN", t.wn },                         .{ "SWZ", t.swz },
        .{ "AM", @intFromBool(m % t.bm == 0) }, .{ "AN", @intFromBool(n % t.bn == 0) }, .{ "AK", @intFromBool(kd % t.bk == 0) },
    };
    for (ints) |p| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, p[0], @intCast(p[1])));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try kernel(), iv, cfg, s));
    var y = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&y, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return y;
}

fn routerLaunch(s: mlx.mlx_stream, x: A, w: A, nax: bool) !?A {
    if (!nax or !mlx.streamIsGpu(s)) return null;
    const mnk = shapeOf(x, w) orelse return null;
    if (!mlxRegular(mnk[0], mnk[1], mnk[2])) return null;
    const y = try launch(s, x, w, .float32, .float32, tile, mnk);
    calls += 1;
    return y;
}

/// On: the router's f32 logits [rows, E] for x [rows, K] by gate_w [E, K] (each in its stored float type: x bf16 and
/// gate_w f16 on the pack) as the launch (lazy, the caller owns them); null where it declines (off or check, no NAX,
/// fewer rows than MLX's regular kernel takes, other shapes or types: the chain).
pub fn router(s: mlx.mlx_stream, x: A, w: A, nax: bool) !?A {
    if (mode != .on) return null;
    return routerLaunch(s, x, w, nax);
}

/// Check: the launch beside the chain's logits `want` (served), compared word for word (one host read).
pub fn checkRouter(s: mlx.mlx_stream, x: A, w: A, want: A, nax: bool) !void {
    if (mode != .check) return;
    const got = (try routerLaunch(s, x, w, nax)) orelse return;
    defer _ = mlx.mlx_array_free(got);
    try compare(s, want, got);
}

/// `a` and `b` (f32, one shape) word for word: equal u32 views summed on the device, one host read.
fn compare(s: mlx.mlx_stream, a: A, b: A) !void {
    var tmp: [5]A = undefined;
    for (&tmp) |*x| x.* = mlx.mlx_array_new();
    defer for (tmp) |x| {
        _ = mlx.mlx_array_free(x);
    };
    try mlx.check(mlx.mlx_view(&tmp[0], a, .uint32, s));
    try mlx.check(mlx.mlx_view(&tmp[1], b, .uint32, s));
    try mlx.check(mlx.mlx_equal(&tmp[2], tmp[0], tmp[1], s));
    try mlx.check(mlx.mlx_sum(&tmp[3], tmp[2], false, s));
    try mlx.check(mlx.mlx_astype(&tmp[4], tmp[3], .uint32, s));
    try mlx.check(mlx.mlx_array_eval(tmp[4]));
    const same = (mlx.mlx_array_data_uint32(tmp[4]) orelse return error.NoData)[0];
    const n: u64 = mlx.mlx_array_size(a);
    checks += 1;
    words += n;
    differ += n - same;
}

// The meters since the last line.
var calls: u64 = 0;
var checks: u64 = 0;
var words: u64 = 0;
var differ: u64 = 0;

fn reset() void {
    calls = 0;
    checks = 0;
    words = 0;
    differ = 0;
}

/// `[gate-gemm] mode= calls= checks= words= differ=` (since the last line; "" off), then reset.
pub fn meters(buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const s = std.fmt.bufPrint(buf, "[gate-gemm] mode={t} calls={d} checks={d} words={d} differ={d}\n", .{ mode, calls, checks, words, differ }) catch return buf[0..0];
    reset();
    return s;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    const t = tile;
    return switch (mode) {
        .off => std.fmt.bufPrint(buf, "[turboexl3_ds41] wide router logits on MLX's matmul chain (MLX_SERVE_DSV41_GATE_GEMM=0)\n", .{}) catch buf[0..0],
        .on => std.fmt.bufPrint(buf, "[turboexl3_ds41] wide router logits as one launch of MLX's fused NAX GEMM body, x and gate_w read as stored, where MLX runs its regular kernel (MLX_SERVE_DSV41_GATE_GEMM; =0: the chain): bm {d} bn {d} bk {d} wm {d} wn {d} swizzle {d}, the same words; [gate-gemm] at each request's end\n", .{ t.bm, t.bn, t.bk, t.wm, t.wn, t.swz }) catch buf[0..0],
        .check => std.fmt.bufPrint(buf, "[turboexl3_ds41] wide router logits checked (MLX_SERVE_DSV41_GATE_GEMM=check): the chain served, the launch run beside every call it would take and compared word for word; [gate-gemm] checks= words= differ= at each request's end\n", .{}) catch buf[0..0],
    };
}

test "gate_gemm setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "1" }) |v| try std.testing.expectEqual(Mode.on, try parse(v));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "Check", " 1", "true" }) |v| try std.testing.expectError(error.GateGemmSetting, parse(v));
}

test "gate_gemm tiles: MLX's own and the pick are exact by construction; SN must be N pairs, BK whole steps" {
    try std.testing.expect(admitted(tile));
    try std.testing.expect(admitted(.{ .bm = 64, .bn = 128, .bk = 256, .wm = 2, .wn = 4, .swz = 2 }));
    try std.testing.expect(admitted(.{ .bm = 32, .bn = 384, .bk = 256, .wm = 1, .wn = 6, .swz = 2 }));
    try std.testing.expect(!admitted(.{ .bm = 64, .bn = 64, .bk = 256, .wm = 2, .wn = 4, .swz = 2 }));
    try std.testing.expect(!admitted(.{ .bm = 64, .bn = 128, .bk = 48, .wm = 2, .wn = 4, .swz = 2 }));
    try std.testing.expect(!admitted(.{ .bm = 8, .bn = 128, .bk = 256, .wm = 1, .wn = 4, .swz = 2 }));
}

test "gate_gemm: MLX's regular kernel from 1,707 rows at K 5,120 and N 384; its split K below" {
    try std.testing.expect(mlxRegular(16383, 384, 5120));
    try std.testing.expect(mlxRegular(2048, 384, 5120));
    try std.testing.expect(mlxRegular(1707, 384, 5120));
    try std.testing.expect(!mlxRegular(1706, 384, 5120));
    try std.testing.expect(!mlxRegular(128, 384, 5120));
}
