//! copied_from: src/plugins/qwen4_exp/trunk/gdn_tail.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The GDN layer's tail on the trunk format's linears (prefill), through the quant table's `linear_*` entries:
//! - GDN-OUT: in_proj_z stops at its inner product and one launch (turboexl3's `gdnOutPrep`) runs z's finish, the
//!   norm-gate and out_proj's prepare per (row, head), so neither z nor the gated rows are written; out_proj continues
//!   from the prepared rows. Every element is the chain's expressions in its order with its roundings.
//! - The overlap: the chunk's recurrence runs piece after piece on the main stream, each from the state the piece before
//!   left (the f32 state crosses a piece boundary unrounded, as at a chunk boundary), while piece i's tail runs on the
//!   side stream; out_proj's weight is made once and one finish reads every piece's inner product. The tail launches
//!   are per row with the whole chunk's template arguments, and MLX's GEMM runs the whole call's kernel at every piece's
//!   row count (`gemmSame`), so every row keeps its bytes.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const abi = @import("../../../abi/quant.zig");
const decode_fused = @import("../../../sdk/blocks/gdn/decode_fused.zig");
const kernels = @import("../../../sdk/blocks/gdn/kernels.zig");
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const sigmoidTableFor = @import("../../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const turboexl3 = @import("turboexl3");
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const Trunk = @import("../trunk.zig").Trunk;
const LinearAttnWeights = @import("../../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const SideStream = @import("../side_stream.zig").SideStream;
const gdn_core = @import("gdn_core.zig");
const splitK = @import("gdn_ab.zig").splitK;

/// A z read in place: columns [off, off + Hv*Dv) of `a`'s rows `stride` wide.
pub const Cols = struct { a: mlx.mlx_array, off: c_int, stride: c_int };

// ── the format's parts (`abi.QuantTable`'s `linear_*` entries) ──

/// The trunk format and its weight of the map's `.weight` array `w`; null for a linear the trunk runs itself.
const Part = struct { t: *const abi.QuantTable, w: *const abi.Weight };

fn partOf(self: *const Trunk, w: mlx.mlx_array) ?Part {
    const t = self.formats.trunk orelse return null;
    return .{ .t = t, .w = self.formats.linearOf(w) orelse return null };
}

fn arrayOf(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

fn dtypeArg(dt: mlx.mlx_dtype) i32 {
    return @intFromEnum(dt);
}

var refused_said = false;

/// A part's result, or null on a refusal (said once).
fn took(p: Part, v: abi.Verdict, out: abi.Array, what: []const u8) ?mlx.mlx_array {
    if (v.isOk()) return arrayOf(out);
    if (!refused_said) {
        refused_said = true;
        log.info("[gdn-tail] {s} refused {s} ({s}); the chain runs\n", .{ p.t.name.slice(), what, v.why.slice() });
    }
    return null;
}

fn linearSplit(p: Part, x: mlx.mlx_array, side: ?mlx.mlx_stream, inner: bool, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_split orelse return null;
    const opts: abi.LinearParts = .{ .side = .{ .ctx = if (side) |r| r.ctx else null }, .inner = inner };
    var out: abi.Array = .{};
    return took(p, f(p.w, .{ .ctx = x.ctx }, &opts, .{ .ctx = s.ctx }, &out), out, "a split linear");
}

fn linearFinish(p: Part, parts: []const mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_finish orelse return null;
    var buf: [MAX_PIECES]abi.Array = undefined;
    for (parts, buf[0..parts.len]) |a, *o| o.* = .{ .ctx = a.ctx };
    const opts: abi.LinearParts = .{ .dtype = dtypeArg(dt), .parts = .of(buf[0..parts.len]) };
    var out: abi.Array = .{};
    return took(p, f(p.w, &opts, .{ .ctx = s.ctx }, &out), out, "a finish");
}

fn linearPrepared(p: Part, xp: mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_prepared orelse return null;
    const opts: abi.LinearParts = .{ .dtype = dtypeArg(dt) };
    var out: abi.Array = .{};
    return took(p, f(p.w, .{ .ctx = xp.ctx }, &opts, .{ .ctx = s.ctx }, &out), out, "prepared rows");
}

fn linearPieces(p: Part, rows: []const u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_pieces orelse return null;
    const opts: abi.LinearParts = .{ .dtype = dtypeArg(dt), .rows = .of(rows) };
    var out: abi.Array = .{};
    return took(p, f(p.w, &opts, .{ .ctx = s.ctx }, &out), out, "row pieces");
}

fn linearPiece(p: Part, x: mlx.mlx_array, held: mlx.mlx_array, prepared: bool, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_piece orelse return null;
    const opts: abi.LinearParts = .{ .dtype = dtypeArg(dt), .prepared = prepared, .held = .{ .ctx = held.ctx } };
    var out: abi.Array = .{};
    return took(p, f(p.w, .{ .ctx = x.ctx }, &opts, .{ .ctx = s.ctx }, &out), out, "a row piece");
}

/// `qmatmul` recorded on stream `s` (the side stream): every op it builds takes `self.s`, so it is swapped for the call.
pub fn qmatmulOn(self: *Trunk, s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    const main = self.s;
    self.s = s;
    defer self.s = main;
    return self.qmatmul(x, w, sc, bi);
}

/// The side stream's `gdnz` part: in_proj_z after x's prepare (a format linear: `linear_split`; any other: the whole
/// projection) recorded on the side stream, so it runs beside the prework and the recurrence; null where the part does
/// not run this call. The same ops in the same graph as the one-stream projection.
pub fn projZSide(self: *Trunk, x: mlx.mlx_array, z_w: mlx.mlx_array, z_s: mlx.mlx_array, z_b: mlx.mlx_array) !?mlx.mlx_array {
    const rows = SideStream.rowsOf(x);
    const side = SideStream.forPart(.gdnz, self, rows) orelse return null;
    const z = if (partOf(self, z_w)) |p| (linearSplit(p, x, side, false, self.s) orelse return null) else try qmatmulOn(self, side, x, z_w, z_s, z_b);
    SideStream.note("gdnz", rows);
    return z;
}

// ── GDN-OUT ──

/// The facts `outPlan` decides on: pure, so the CPU test walks every arm.
pub const OutFacts = struct {
    /// the packed prework's prefill arm runs here (17 <= S, B <= 2, a GPU stream): its norm-gate is the one replaced
    prefill_prework: bool = false,
    /// the separate-projection arm projects, with one joined qkv weight (no supplied projections, combined qkvz,
    /// stand-in or row-joined group)
    separate_proj: bool = false,
    /// conv kernel 4, Dk = Dv = 128, no spec capture
    geometry: bool = false,
    /// x bf16 / f16 [B, S, hidden]
    act_ok: bool = false,
    /// in_proj_z and out_proj are the trunk format's, which serves the parts
    z_format: bool = false,
    out_format: bool = false,
    /// the caller takes the gated rows (a stand-in), not out_proj's output
    skip_output: bool = false,
};

/// Null = engage, else the first failed fact.
pub fn outPlan(f: OutFacts) ?[]const u8 {
    if (!f.prefill_prework) return "not the packed prefill prework (17 <= S, B <= 2)";
    if (!f.separate_proj) return "not the separate-projection arm (one qkv weight)";
    if (!f.geometry) return "conv kernel, head dims or spec capture";
    if (!f.act_ok) return "activation dtype or shape";
    if (!f.z_format) return "in_proj_z is not a trunk-format linear with the GDN tail's parts";
    if (!f.out_format) return "out_proj is not a trunk-format linear with the GDN tail's parts";
    if (f.skip_output) return "the caller takes the gated rows";
    return null;
}

pub fn outFacts(self: *const Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, batch: c_int, seq_len: c_int, is_prefill: bool, projected: bool, skip_output: bool) OutFacts {
    const cfg = &self.config;
    const xd = mlx.mlx_array_dtype(x);
    const xsh = mlx.getShape(x);
    const row_joined = la.in.w.ctx != null and self.rowGroupServes(batch * seq_len);
    const zp = partOf(self, la.z_w);
    const op = partOf(self, la.out_w);
    return .{
        .prefill_prework = is_prefill and decode_fused.gdnPrefillFusedFor(seq_len, batch) and (decode_fused.gdn_prework_override orelse true) and mlx.streamIsGpu(self.s),
        .separate_proj = !projected and !la.combined_proj and !qwen4Standin().gdn_proj and !row_joined and la.qkv_w.ctx != null,
        .geometry = cfg.linear_conv_kernel_dim == 4 and cfg.linear_key_head_dim == 128 and cfg.linear_value_head_dim == 128 and !self.spec_capture_ssm,
        .act_ok = (xd == .bfloat16 or xd == .float16) and xsh.len == 3 and xsh[0] == batch and xsh[1] == seq_len,
        .z_format = if (zp) |p| p.t.linear_split != null and p.t.linear_finish != null else false,
        .out_format = if (op) |p| p.t.linear_prepared != null else false,
        .skip_output = skip_output,
    };
}

var out_said = false;
var out_declined_said = false;

/// Says once why a prefill chunk keeps the chain.
pub fn outDeclined(why: []const u8, seq_len: c_int, batch: c_int) void {
    if (out_declined_said or seq_len < 17) return;
    out_declined_said = true;
    log.info("[gdn-out] declined at S={d} B={d}: {s}; the chain runs\n", .{ seq_len, batch, why });
}

pub const OutProj = struct { qkv: mlx.mlx_array, z: mlx.mlx_array, z_inner: mlx.mlx_array };

/// GDN-OUT's projections: qkv as the separate arm projects it, in_proj_z stopped at its inner product (the work after
/// the prepare on the side stream's `gdnz`), and the chain's z: the format's finish of that inner product, lazily
/// (nothing reads it while the fused tail engages). Null where the format refuses z's rows (nothing launched).
pub fn outProj(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, batch: c_int, seq_len: c_int) !?OutProj {
    const zp = partOf(self, la.z_w) orelse return null;
    const rows = SideStream.rowsOf(x);
    const side = SideStream.forPart(.gdnz, self, rows);
    const qkv = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
    errdefer _ = mlx.mlx_array_free(qkv);
    const zi = linearSplit(zp, x, side, true, self.s) orelse {
        _ = mlx.mlx_array_free(qkv);
        return null;
    };
    errdefer _ = mlx.mlx_array_free(zi);
    const z2 = linearFinish(zp, &.{zi}, mlx.mlx_array_dtype(x), side orelse self.s) orelse {
        _ = mlx.mlx_array_free(zi);
        _ = mlx.mlx_array_free(qkv);
        return null;
    };
    defer _ = mlx.mlx_array_free(z2);
    var z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_reshape(&z, z2, &[_]c_int{ batch, seq_len, mlx.getShape(z2)[1] }, 3, self.s));
    if (side != null) SideStream.note("gdnz", rows);
    return .{ .qkv = qkv, .z = z, .z_inner = zi };
}

/// GDN-OUT's tail: out_proj(norm-gate(y, z)) from z's inner product through the fused prepare (`outPrep`) and the
/// format's prepared-rows linear, [B, S, hidden] in the activation dtype `act`. Null where either declines; the
/// prepare's output is then dropped unevaluated and the chain's norm-gate reads the lazy z.
pub fn outFused(self: *Trunk, y: mlx.mlx_array, zi: mlx.mlx_array, la: *const LinearAttnWeights, act: mlx.mlx_dtype, hv: c_int, dv: c_int, batch: c_int, seq_len: c_int) !?mlx.mlx_array {
    const zp = partOf(self, la.z_w) orelse return null;
    const op = partOf(self, la.out_w) orelse return null;
    // the norm-gate reads y and z in one dtype, and z's finish rounds to the activation's
    if (mlx.mlx_array_dtype(y) != act) return null;
    const xp = (try outPrep(self.s, y, zi, arrayOf(zp.w.biases), la.norm_w, self.gdn_eps.?, arrayOf(op.w.scales), false, hv, dv, batch, seq_len)) orelse return null;
    defer _ = mlx.mlx_array_free(xp);
    const y2 = linearPrepared(op, xp, act, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(y2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y2, &[_]c_int{ batch, seq_len, mlx.getShape(y2)[1] }, 3, self.s));
    if (!out_said) {
        out_said = true;
        log.info("[gdn-out] engaged: S={d} B={d} Hv={d} (z's finish, the norm-gate and out_proj's prepare in one launch)\n", .{ seq_len, batch, hv });
    }
    return out;
}

/// GDN-OUT's launch, turboexl3's `gdnOutPrep` (the format's own Hadamard prepare and finish): out_proj's prepared rows
/// xp f16 [B*S, Hv*Dv] (natural order) from the recurrence's y [B, S, Hv, Dv], in_proj_z's inner product zq f16
/// [B*S, Hv*Dv] with its finish vector svh f16 [Hv*Dv], the norm weight, eps (0-dim f32) and out_proj's prepare vector
/// suh f16 [Hv*Dv]. Prefill widths only; null where the fused norm-gate itself would decline.
pub fn outPrep(s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, svh: mlx.mlx_array, norm_w: mlx.mlx_array, eps: mlx.mlx_array, suh: mlx.mlx_array, swish: bool, hv: c_int, dv: c_int, batch: c_int, seq: c_int) !?mlx.mlx_array {
    // turboexl3 serves the sigmoid gate from the table (`gdnNormGateFused`'s source from 17 rows) at prefill widths; its
    // own checks decline every other call, which the caller then serves unfused.
    if (swish or dv != 128 or !mlx.streamIsGpu(s) or !decode_fused.gdnPrefillFusedFor(seq, batch)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(norm_w) != dt) return null;
    const ysh = mlx.getShape(y);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv) return null;
    const none: mlx.mlx_array = .{ .ctx = null };
    const z: turboexl3.Proj = .{ .trellis = none, .suh = none, .svh = svh };
    const o: turboexl3.Proj = .{ .trellis = none, .suh = suh, .svh = none };
    const norm: turboexl3.gdn_out.Norm = .{ .w = norm_w, .eps = eps, .sigtab = try sigmoidTableFor(s, dt) };
    return turboexl3.gdnOutPrep(s, y, zq, z, norm, o) catch |e| switch (e) {
        error.Exl3EngineNotImplemented, error.BadExl3Shape, error.BadExl3Dtype => return null,
        else => return e,
    };
}

// ── the overlap ──

/// The smallest piece: out_proj's GEMM at 1024 rows still launches 16 M tiles x out/128 N tiles.
pub const MIN_PIECE: c_int = 1024;
/// The most pieces: the format's finish over pieces reads up to four, and in_proj_z's GEMM shares the side stream.
pub const MAX_PIECES: usize = 4;
/// Piece boundaries fall on the NAX GEMM's M tile (64 on the large devices, 128 on the others), so every row sits where
/// it sits in the whole call's tiles.
pub const ALIGN: c_int = 128;

/// The kernel MLX's matmul runs for a 2-D row-contiguous f16 / bf16 [m, k] x [k, n] (no transposes, m and n >= 2) on a
/// device of class `devc` (the last letter of its architecture), with or without NAX: split K, else the regular NAX kernel
/// or steel kernel and its tile. Two row counts with one key run one kernel, whose K loop gives every row the same
/// summation order (the M tail's bounds-checked loads zero the rows past M; the swizzle only places tiles).
pub const MlxGemm = struct { split: bool = false, nax: bool = false, bm: u16 = 0, bn: u16 = 0, bk: u16 = 0, wm: u8 = 0, wn: u8 = 0 };

pub fn mlxGemm(m: c_int, n: c_int, k: c_int, nax: bool, devc: u8) MlxGemm {
    const big = devc == 's' or devc == 'c' or devc == 'd';
    if (nax) {
        if (splitK(m, n, k)) return .{ .split = true, .nax = true };
        if (!big) return .{ .nax = true, .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
        return .{ .nax = true, .bm = 64, .bn = 128, .bk = if (k >= 8192 and k > m + n) 64 else 256, .wm = 2, .wn = 4 };
    }
    // a small output over a long K splits K (Max and Ultra up to a larger output)
    const tm: i64 = @divTrunc(m + 15, 16);
    const tn: i64 = @divTrunc(n + 15, 16);
    const thr: i64 = if (devc == 's' or devc == 'd') 2048 else 1024;
    if (tm * tn <= thr and @divTrunc(k, 16) >= 8 and k >= @max(m, n)) return .{ .split = true };
    if (devc == 'g' or devc == 'p') return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    if (devc == 'd') {
        if (@as(i64, m) * n >= 1 << 20 and 2 * @max(m, n) <= k) return .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
        return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    }
    return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 };
}

/// Whether out_proj's GEMM on a piece of `m` rows runs the kernel the whole call of `full` rows runs, with NAX and
/// without (both answers, so a mismatch of the NAX gate cannot pass), on this device's class or, unknown (0), on every
/// class; and neither splits K (a split's partitions follow the row count).
pub fn gemmSame(m: c_int, full: c_int, n: c_int, k: c_int, devc: u8) bool {
    const this_class = [_]u8{devc};
    const classes: []const u8 = if (devc != 0) &this_class else "gpsdm";
    for (classes) |c| for ([_]bool{ true, false }) |nax| {
        const a = mlxGemm(m, n, k, nax, c);
        const b = mlxGemm(full, n, k, nax, c);
        if (a.split or b.split or !std.meta.eql(a, b)) return false;
    };
    return true;
}

var devc_cached: ?u8 = null;

/// The device class letter MLX's GEMM routing reads: the architecture's last letter, 0 when unknown.
fn deviceClass() u8 {
    if (devc_cached) |c| return c;
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse "";
    const c: u8 = if (arch.len > 0) arch[arch.len - 1] else 0;
    devc_cached = c;
    return c;
}

/// A chunk's pieces: piece i is rows [start[i], start[i + 1]).
pub const Plan = struct { n: usize, start: [MAX_PIECES + 1]c_int };

/// The pieces of a `rows`-row chunk whose out_proj is [rows, k] x [k, n]: rows / MIN_PIECE of them, at most `max` and
/// MAX_PIECES, equal pieces on ALIGN-row boundaries with the remainder in the last; the most such pieces for which every
/// piece's GEMM runs the whole call's kernel. Null below two.
pub fn plan(rows: c_int, n: c_int, k: c_int, dc: u8, max: usize) ?Plan {
    if (rows < 2 * MIN_PIECE) return null;
    var np: usize = @min(@min(max, MAX_PIECES), @as(usize, @intCast(@divTrunc(rows, MIN_PIECE))));
    while (np >= 2) : (np -= 1) {
        const base = @divTrunc(@divTrunc(rows, @as(c_int, @intCast(np))), ALIGN) * ALIGN;
        var p = Plan{ .n = np, .start = @splat(rows) };
        var same = true;
        for (0..np) |i| {
            p.start[i] = @as(c_int, @intCast(i)) * base;
            const len = if (i + 1 < np) base else rows - p.start[i];
            same = same and gemmSame(len, rows, n, k, dc);
        }
        if (same) return p;
    }
    return null;
}

/// Rows [r0, r1) of a [1, S, ...] array along S (a view), recorded on `s`.
pub fn rowsView(dst: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(src);
    if (sh.len < 2 or sh.len > 4) return error.BadGdnOverlapShape;
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(0);
    const strides: [4]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[1] = r0;
    stop[1] = r1;
    try mlx.check(mlx.mlx_slice(dst, src, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
}

/// One launch of the pipelined recurrence over q, k [1, T, Hk, Dk], v [1, T, Hv, Dv], g, beta [1, T, Hv] from `state`
/// [1, Hv, Dv, Dk]: the forward's launch on the pipelined route (its outputs, grid and template arguments in its order),
/// y in `out_dtype` and the state after the last row in the state's dtype.
pub fn pipelinedRun(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state: mlx.mlx_array, hk: c_int, hv: c_int, dk: c_int, dv: c_int, out_dtype: mlx.mlx_dtype) !gdn_core.RecurOut {
    const t = mlx.getShape(q)[1];
    const st_dtype = mlx.mlx_array_dtype(state);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, t, hv, dv }, 4, out_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, st_dtype));
    try kernels.gdnSetGrid(config, .pipelined, dv, hv, 1);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    inline for (.{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv } }) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
    const t_scalar = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_scalar);
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state, t_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try kernels.gdnRouteKernel(.pipelined, 0, false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    var st = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_vector_array_get(&st, outputs_vec, 1));
    return .{ .y = y, .state = st };
}

/// `arrs` submitted (no wait): each stream they are recorded on starts them once its earlier evals are done, and a later
/// eval that reads one on another stream waits for this eval's event there alone.
fn submit(arrs: []const mlx.mlx_array) !void {
    const ev = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_async_eval(ev));
}

var overlap_said: [2]bool = .{ false, false };
var overlap_declined_said = false;

/// A decline, said once: null for `overlap` to return.
fn decline(why: []const u8, seq_len: c_int) ?mlx.mlx_array {
    if (!overlap_declined_said) {
        overlap_declined_said = true;
        log.info("[gdn-overlap] declined at S={d}: {s}; the whole chunk runs\n", .{ seq_len, why });
    }
    return null;
}

/// A prefill chunk's pipelined recurrence and its norm-gate + out_proj tail over row pieces (`plan`). Piece i's
/// recurrence runs on this stream from the state piece i - 1 left; its tail up to out_proj's GEMM on the side stream
/// (the last piece's here: nothing is left to overlap it). A piece's tail is the one the whole chunk would take:
/// with GDN-OUT's `z_inner` (in_proj_z's inner product f16 [S, Hv*Dv]), the fused prepare on the piece's rows and
/// out_proj from those prepared rows; else the fused norm-gate on the piece's rows of z (`z_cols`) and out_proj's
/// inner product.
/// Submission: W alone as soon as it is built; each piece but the last as soon as it is built (MLX's cross-stream wait
/// takes the producer stream's latest fence when the consumer is encoded, and one eval encodes the whole recurrence
/// chain before any tail); the last piece, its tail and the finish as soon as built. `core_rec`: each piece's recurrence
/// on the GDN core kernel (the pipelined kernel's bytes).
/// The layer's output [1, S, hidden] with the final state installed in `ssm`, or null with `ssm` untouched: the caller
/// runs the whole chunk.
pub fn overlap(self: *Trunk, la: *const LinearAttnWeights, ssm: *SSMCacheEntry, z_cols: Cols, z_inner: ?mlx.mlx_array, act: mlx.mlx_dtype, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, out_dtype: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, seq_len: c_int, core_rec: bool) !?mlx.mlx_array {
    // a chunk too short for two pieces is the plain chunk, not a decline
    if (seq_len < 2 * MIN_PIECE) return null;
    const op = partOf(self, la.out_w) orelse return decline("out_proj is not a trunk-format linear", seq_len);
    if (op.t.linear_pieces == null or op.t.linear_piece == null or op.t.linear_finish == null) return decline("the trunk format takes no row pieces", seq_len);
    // the pieces carry the state exactly only in f32: a narrower state would round at every piece boundary
    if (mlx.mlx_array_dtype(ssm.ssm_state) != .float32) return decline("the recurrent state is not f32", seq_len);
    const value_dim = hv * dv;
    // GDN-OUT's tail where `outFused` would take the whole chunk's
    const zq: ?mlx.mlx_array = if (z_inner != null and out_dtype == act) z_inner else null;
    if ((out_dtype != .bfloat16 and out_dtype != .float16) or mlx.mlx_array_dtype(la.norm_w) != out_dtype or dv != 128)
        return decline("the norm-gate's operands", seq_len);
    if (zq) |zi| {
        const zsh = mlx.getShape(zi);
        if (mlx.mlx_array_dtype(zi) != .float16 or zsh.len != 2 or zsh[0] != seq_len or zsh[1] != value_dim)
            return decline("in_proj_z's inner product", seq_len);
    } else {
        const zsh = mlx.getShape(z_cols.a);
        if (mlx.mlx_array_dtype(z_cols.a) != out_dtype or zsh.len != 3 or zsh[0] != 1 or zsh[1] != seq_len or zsh[2] != z_cols.stride or z_cols.off + value_dim > z_cols.stride)
            return decline("the norm-gate's operands", seq_len);
    }
    const osh = mlx.getShape(la.out_w);
    if (osh.len != 3 or osh[0] * 16 != value_dim) return decline("out_proj's weight shape", seq_len);
    const p = plan(seq_len, osh[1] * 16, value_dim, deviceClass(), MAX_PIECES) orelse
        return decline("no piece split keeps MLX's GEMM kernel", seq_len);
    // the tails' stream: the side stream (this half's, under the prefill wavefront)
    const side = SideStream.get(self.s) orelse return decline("no second GPU stream", seq_len);
    const n = p.n;
    // out_proj's W, made once on the side stream (refused before any launch where a row count leaves the format's path)
    var piece_rows: [MAX_PIECES]u32 = undefined;
    for (0..n) |i| piece_rows[i] = @intCast(p.start[i + 1] - p.start[i]);
    const W: mlx.mlx_array = linearPieces(op, piece_rows[0..n], out_dtype, side) orelse
        return decline("the trunk format refused the row pieces", seq_len);
    defer _ = mlx.mlx_array_free(W);
    // W alone now, so it takes the side stream ahead of in_proj_z's GEMM and the first tail
    try submit(&[_]mlx.mlx_array{W});
    if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(self.config.rms_norm_eps);
    const zp: ?Part = if (zq != null) partOf(self, la.z_w) else null;
    if (zq != null and zp == null) return decline("in_proj_z is not a trunk-format linear", seq_len);
    // per piece: the recurrence's y and state, out_proj's input (the gated rows; GDN-OUT: the prepared rows), its inner
    // product
    var ys: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var states: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var flats: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var inners: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for ([_][]mlx.mlx_array{ &ys, &states, &flats, &inners }) |set| for (set) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (0..n) |i| {
        const r0 = p.start[i];
        const r1 = p.start[i + 1];
        // the piece's rows of the prework's outputs: views (one batch row, so each is row-contiguous and read in place)
        var rows: [5]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (rows) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]mlx.mlx_array{ q, k, v, g, beta }, &rows) |src, *dst| try rowsView(dst, src, r0, r1, self.s);
        const state_in = if (i == 0) ssm.ssm_state else states[i - 1];
        const r = if (core_rec)
            try gdn_core.recur(self.s, rows[0], rows[1], rows[2], rows[3], rows[4], state_in, hk, hv, dk, dv, out_dtype)
        else
            try pipelinedRun(self.s, rows[0], rows[1], rows[2], rows[3], rows[4], state_in, hk, hv, dk, dv, out_dtype);
        ys[i] = r.y;
        states[i] = r.state;
        // the piece's tail from its norm-gate on: the side stream, the last piece's on this one
        const ts = if (i + 1 < n) side else self.s;
        var zi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zi);
        if (zq) |zall| {
            try mlx.check(mlx.mlx_slice(&zi, zall, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r1, value_dim }, 2, &[_]c_int{ 1, 1 }, 2, ts));
            flats[i] = (try outPrep(ts, ys[i], zi, arrayOf(zp.?.w.biases), la.norm_w, self.gdn_eps.?, arrayOf(op.w.scales), false, hv, dv, 1, r1 - r0)) orelse
                return decline("the fused prepare declined a piece", seq_len);
            inners[i] = linearPiece(op, flats[i], W, true, out_dtype, ts) orelse return decline("the trunk format refused a piece", seq_len);
        } else {
            try rowsView(&zi, z_cols.a, r0, r1, ts);
            flats[i] = (try decode_fused.gdnNormGateFused(ts, ys[i], zi, z_cols.off, z_cols.stride, la.norm_w, self.gdn_eps.?, false, hv, dv, 1, r1 - r0)) orelse
                return decline("the fused norm-gate declined a piece", seq_len);
            inners[i] = linearPiece(op, flats[i], W, false, out_dtype, ts) orelse return decline("the trunk format refused a piece", seq_len);
        }
        if (i + 1 < n) try submit(&[_]mlx.mlx_array{ inners[i], states[i] });
    }
    // the finish over every piece's inner product
    const y2: mlx.mlx_array = linearFinish(op, inners[0..n], out_dtype, self.s) orelse return decline("the trunk format refused the finish", seq_len);
    defer _ = mlx.mlx_array_free(y2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const y2sh = mlx.getShape(y2);
    try mlx.check(mlx.mlx_reshape(&out, y2, &[_]c_int{ 1, seq_len, y2sh[y2sh.len - 1] }, 3, self.s));
    // the last piece's state is the chunk's
    _ = mlx.mlx_array_free(ssm.ssm_state);
    ssm.ssm_state = states[n - 1];
    states[n - 1] = .{ .ctx = null };
    // the last piece, its tail and the finish to the GPU now, not with whatever eval comes next
    try submit(&[_]mlx.mlx_array{ out, ssm.ssm_state });
    const tail = @intFromBool(zq != null);
    if (!overlap_said[tail]) {
        overlap_said[tail] = true;
        log.info("[gdn-overlap] engaged: S={d} pieces={d} ({d}..{d} rows) tail={s} on the side stream beside the next piece's recurrence\n", .{ seq_len, n, p.start[1], seq_len - p.start[n - 1], if (zq != null) "gdn-out" else "norm-gate" });
    }
    return out;
}

// ── tests ──

const testing = std.testing;
