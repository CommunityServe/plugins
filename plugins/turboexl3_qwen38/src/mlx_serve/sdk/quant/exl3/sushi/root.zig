//! copied_from: src/sdk/quant/exl3/sushi/root.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! EXL3 trellis experts: routed MoE banks in turboderp's EXL3 format, decoded
//! and multiplied by Metal kernels. A module, so another MLX host (mlx-serve)
//! can serve EXL3 packs through the same code: it needs an `mlx_host` import
//! whose root exposes `pub const mlx`, `log` and `io_util`.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
pub const format = @import("expert_exl3.zig");
pub const kernels = @import("expert_exl3_kernels.zig");
pub const glm_group2 = @import("glm_group2.zig");
pub const glm_prefill_grid = @import("glm_prefill_grid.zig");
pub const codebook_tables = @import("codebook_tables.zig");

/// One projection's bank, expert e on axis 0: `trellis` U16 [E, in/16, out/16, n],
/// `suh` F16 [E, in], `svh` F16 [E, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const group_layout = @import("layout.zig");
pub const GroupLayout = group_layout.Layout;
pub const rate_groups = @import("rate_groups.zig");
pub const Ragged = rate_groups.Ragged;
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K],
/// returned in x's shape and dtype. Up to `kernels.DECODE_ROWS_MAX` rows (or any
/// verify rows) take the decode chain, wider the prefill GEMM.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    return moeOutput(s, x, bank, inds, scores, dec, verify_rows, mlx.mlx_array_dtype(x), null, null);
}

fn moeOutput(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, shared_gate: ?mlx.mlx_array) !mlx.mlx_array {
    return moeOutputClamped(s, x, bank, inds, scores, dec, verify_rows, out_dtype, shared, shared_gate, 0);
}

/// GLM's gate is upper-clamped and its up branch is symmetrically clamped,
/// before SwiGLU and the down projection. Packed expert rates remain 2–4 bpw.
pub fn moeClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    return moeClampedWithShared(s, x, bank, inds, scores, dec, limit, null);
}

/// A decode row may fold an already computed shared expert (same dtype and size as the output) into the finish reduce.
pub fn moeClampedWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int, shared: ?mlx.mlx_array) !mlx.mlx_array {
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    const sh = mlx.getShape(x);
    const ix = mlx.getShape(inds);
    if (sh.len != 3 or ix.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[0] != ix[0] or sh[1] != ix[1] or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return error.BadExl3Shape;
    if (sh[2] <= 0 or @mod(sh[2], 128) != 0 or !std.mem.eql(c_int, ix, mlx.getShape(scores))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    const id = mlx.mlx_array_dtype(inds);
    const sd = mlx.mlx_array_dtype(scores);
    if ((xd != .bfloat16 and xd != .float32 and xd != .float16) or
        (id != .uint32 and id != .int32) or
        (sd != .float32 and sd != .bfloat16 and sd != .float16)) return error.BadExl3Dtype;
    const rows = std.math.mul(c_int, sh[0], sh[1]) catch return error.BadExl3Shape;
    _ = std.math.mul(c_int, rows, ix[2]) catch return error.BadExl3Shape;
    const gs = mlx.getShape(bank.gate.trellis);
    if (gs.len != 4 or gs[0] <= 0 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return error.BadExl3Shape;
    const width = gs[2] * 16;
    try validateClampedProjection(bank.gate, gs[0], sh[2], width);
    try validateClampedProjection(bank.up, gs[0], sh[2], width);
    try validateClampedProjection(bank.down, gs[0], width, sh[2]);
    return moeOutputClamped(s, x, bank, inds, scores, dec, false, mlx.mlx_array_dtype(x), shared, null, limit);
}

pub fn validateClampedProjection(p: Proj, experts: c_int, input: c_int, output: c_int) !void {
    const sh = mlx.getShape(p.trellis);
    if (input <= 0 or output <= 0 or @mod(input, 128) != 0 or @mod(output, 128) != 0 or
        sh.len != 4 or sh[0] != experts or sh[1] != @divExact(input, 16) or sh[2] != @divExact(output, 16) or sh[3] < 0 or
        format.kFromPackedDim(@intCast(sh[3])) == null or
        !std.mem.eql(c_int, &.{ experts, input }, mlx.getShape(p.suh)) or
        !std.mem.eql(c_int, &.{ experts, output }, mlx.getShape(p.svh))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.trellis) != .uint16 or mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
}

fn moeOutputClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, shared_gate: ?mlx.mlx_array, limit: c_int) !mlx.mlx_array {
    try dec.requireTable();
    // Which kernel a dispatch picks is read off ONE process-global codebook,
    // and several EXL3 packs can be resident at once: set it per call.
    kernels.setDecodeParams(dec);
    const xsh = mlx.getShape(x);
    const xd = out_dtype;
    const B = xsh[0];
    const S = xsh[1];
    const D = xsh[xsh.len - 1];
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ B * S, D }, 2, s));
    const ish = mlx.getShape(inds);
    const K = ish[ish.len - 1];
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_reshape(&slots, inds, &[_]c_int{B * S * K}, 1, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{B * S * K}, 1, s));
    var slots_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_u);
    try mlx.check(mlx.mlx_astype(&slots_u, slots, .uint32, s));
    const rows: usize = @intCast(B * S);
    if (rows >= 2) {
        kernels.dumpUnionHist(slots_u, rows, @intCast(K)) catch {};
    }
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const y = if (limit > 0)
        try kernels.moeSwigluClampedWithShared(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, K, limit, xd, shared)
    else if (rows <= kernels.DECODE_ROWS_MAX or verify_rows)
        if (mlx.getShape(g.trellis)[3] == mlx.getShape(u.trellis)[3])
            try kernels.moeSwigluFusedWithShared(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, xd, shared, shared_gate)
        else
            try moeMixedDecode(s, x2, bank, slots_u, sc, K, xd)
    else
        try kernels.moePrefill(s, x2, g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh, slots_u, sc, K);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xsh.ptr, @intCast(xsh.len), s));
    if (mlx.mlx_array_dtype(out) != xd) {
        var cast = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&cast, out, xd, s));
        _ = mlx.mlx_array_free(out);
        return cast;
    }
    return out;
}

pub const Spec = struct {
    rate: format.Rate,
    codebook: format.Codebook,
    window: format.Window = .w16,
    /// The bank rates the pack's codebook table has a row for (bit n/2); null admits every rate.
    rates: ?u128 = null,
    /// The pack's STD1 or SPEC1 table (`Decode.table`); null for every other codebook.
    table: ?u8 = null,
};

/// `k` is a rate, integer or fractional: admitted only when 16k is an even
/// whole number of halfwords per tile, which is what every reader indexes by.
fn rateFromConfigK(k_v: std.json.Value) ?format.Rate {
    const scaled: f64 = switch (k_v) {
        .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
        .float => |f| f * 16.0,
        else => return null,
    };
    const rounded = @round(scaled);
    if (@abs(scaled - rounded) > 1e-6) return null;
    if (rounded < 0 or rounded > 1024) return null;
    return format.kFromPackedDim(@intFromFloat(rounded));
}

/// `window` is the codeword width the pack's search hashed. Absent means 16,
/// the whole sliding window; a width this build cannot decode is refused, not
/// rounded — the same bitstream decodes to different weights at each width.
fn windowFromConfig(v: ?std.json.Value) ?format.Window {
    const raw = v orelse return format.Window.w16;
    if (raw != .integer) return null;
    return format.Window.fromBits(raw.integer);
}

pub fn parseExpertQuant(obj: std.json.ObjectMap) !Spec {
    const block = obj.get("expert_quant") orelse return error.ExpertLayoutUnsupported;
    if (block != .object) return error.ExpertLayoutUnsupported;
    const fmt_v = block.object.get("format") orelse return error.ExpertLayoutUnsupported;
    if (fmt_v != .string) return error.ExpertLayoutUnsupported;
    const stq = std.mem.eql(u8, fmt_v.string, "stq");
    if (!stq and !std.mem.eql(u8, fmt_v.string, "exl3")) return error.ExpertLayoutUnsupported;
    const k_v = block.object.get("k") orelse return error.ExpertLayoutUnsupported;
    const rate = rateFromConfigK(k_v) orelse return error.ExpertLayoutUnsupported;
    const cb_v = block.object.get("codebook") orelse return error.ExpertLayoutUnsupported;
    if (cb_v != .string) return error.ExpertLayoutUnsupported;
    const codebook = format.Codebook.fromName(cb_v.string) orelse return error.ExpertLayoutUnsupported;
    const tabled = codebook.hasTable();
    if (tabled != stq) return error.ExpertLayoutUnsupported;
    const window = windowFromConfig(block.object.get("window")) orelse return error.Exl3WindowUnsupported;
    if (!tabled) {
        if (block.object.contains("codebook_table")) return error.Exl3CodebookTableCodebookMismatch;
        return .{ .rate = rate, .codebook = codebook, .window = window };
    }
    const table = try codebook_tables.resolve(block.object, codebook, window);
    if (!rate.in(table.rates())) return error.Exl3CodebookTableRateMissing;
    return .{ .rate = rate, .codebook = codebook, .window = window, .rates = table.rates(), .table = codebook_tables.index(table) };
}

const SPEC1_TABLE = ",\"codebook_table\":\"spec1-w15\",\"codebook_digest\":\"37f72431\"";
const PAIR1_TABLE = ",\"codebook_table\":\"pair1-w16-v1\",\"codebook_digest\":\"c089b4eb\"";
const STD1_TABLE = ",\"codebook_table\":\"std1\",\"codebook_digest\":\"c5dba171\"";

pub fn admitTopK(topk: u32) !void {
    if (topk > kernels.REDUCE_MAX_TOPK) return error.Exl3TopKExceedsReduceBank;
}

pub fn trellisAdmitted(shape: []const c_int, experts: u32, in_dim: u32, out_dim: u32, rate: format.Rate) bool {
    if (shape.len != 4) return false;
    for (shape) |d| if (d <= 0) return false;
    if (in_dim % 16 != 0 or out_dim % 16 != 0) return false;
    const want = [3]u32{ experts, in_dim / 16, out_dim / 16 };
    for (want, 0..) |w, i| if (w != @as(u32, @intCast(shape[i]))) return false;
    const packed_rate = format.kFromPackedDim(@intCast(shape[3])) orelse return false;
    return packed_rate.n <= rate.n;
}

test {
    _ = format;
    _ = kernels;
    _ = codebook_tables;
}

fn moeMixedDecode(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, slots: mlx.mlx_array, scores: mlx.mlx_array, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const hidden = mlx.getShape(x)[1];
    const rows = mlx.getShape(x)[0];
    const inter = mlx.getShape(bank.gate.trellis)[2] * 16;
    const nslots = rows * topk;
    const g = bank.gate;
    const u = bank.up;
    const d = bank.down;
    const gate = try kernels.pairGemv(s, x, g.suh, g.suh, g.trellis, g.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(gate[0]);
    defer _ = mlx.mlx_array_free(gate[1]);
    const up = try kernels.pairGemv(s, x, u.suh, u.suh, u.trellis, u.trellis, slots, hidden, inter, nslots, topk, 0);
    defer _ = mlx.mlx_array_free(up[0]);
    defer _ = mlx.mlx_array_free(up[1]);
    const inner = try kernels.downGemvFusedMid(s, gate[0], up[0], d.trellis, g.svh, u.svh, d.suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(inner);
    return kernels.downFinishReduce(s, inner, d.svh, slots, scores, hidden, rows, topk, dtype);
}

test { _ = glm_group2; }

test {
    _ = rate_groups;
}

test {
    _ = glm_prefill_grid;
}
