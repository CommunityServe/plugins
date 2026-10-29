//! copied_from: src/sdk/quant/exl3/format.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The EXL3 quant format: Sushi's `expert_quant` routed-expert banks of 16x16 trellis tiles, `format: exl3` (MCG,
//! MUL1) and its STQ codebook family (`format: stq`: STD1, SPEC1, PAIR1 under a named, digest-pinned table). The
//! decode is `sushi/` (Sushi 1.3.0 as upstream ships it, module `sushi_exl3`). Experts only: a layer's three banks
//! run as one routed SwiGLU dispatch, never per projection.

const std = @import("std");
/// Sushi's decode itself, for the plugins that run the routed chain's stages on their own (mlx-stream's GLM-5.3).
pub const sushi = @import("sushi_exl3");
const abi = @import("../../../abi/quant.zig");
const mlx = @import("../../../mlx/ffi.zig");
const block = @import("../block.zig");

const Str = abi.Str;
const Verdict = abi.Verdict;

pub const Spec = sushi.Spec;
pub const Bank = sushi.Bank;
pub const Proj = sushi.Proj;
pub const Decode = sushi.format.Decode;
pub const Codebook = sushi.format.Codebook;
pub const Window = sushi.format.Window;
/// The host's row planning around `routed` (`DECODE_ROWS_MAX`, `usesPrefillArm`, `rowsOfShape`).
pub const kernels = sushi.kernels;

/// `expert_quant` of a config root, refused by name when this build has no kernels for its codebook, window, table
/// or for `topk` routed experts.
pub fn parse(root: std.json.ObjectMap, topk: u32) !Spec {
    const spec = try sushi.parseExpertQuant(root);
    try sushi.admitTopK(topk);
    return spec;
}

/// What a dispatch decodes with: an STQ table rides along (a table-keyed codebook without it is refused).
pub fn decode(spec: Spec) Decode {
    return .{ .codebook = spec.codebook, .window = spec.window, .table = spec.table };
}

/// One projection's bank at load: its suh/svh present, a trellis the kernels decode at the pack's rate, and a rate
/// the pack's codebook table has a row for.
pub fn admit(p: Proj, experts: u32, in_dim: u32, out_dim: u32, spec: Spec) !void {
    if (p.suh.ctx == null or p.svh.ctx == null) return error.MissingWeight;
    const shape = mlx.getShape(p.trellis);
    if (!sushi.trellisAdmitted(shape, experts, in_dim, out_dim, spec.rate)) return error.Exl3TrellisGeometry;
    const rate = sushi.format.kFromPackedDim(@intCast(shape[3])) orelse return error.Exl3TrellisGeometry;
    if (!rate.in(spec.rates)) return error.Exl3CodebookTableRateMissing;
}

pub const Routed = struct {
    /// > 0: GLM's clamped SwiGLU at this limit.
    limit: c_int = 0,
    verify_rows: bool = false,
};

/// Routed SwiGLU of `x` [B, S, D] through `bank`'s experts `inds` [B, S, K], weighted by `scores`.
pub fn routed(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, spec: Spec, opts: Routed) !mlx.mlx_array {
    const dec = decode(spec);
    return if (opts.limit > 0)
        sushi.moeClamped(s, x, bank, inds, scores, dec, opts.limit)
    else
        sushi.moe(s, x, bank, inds, scores, dec, opts.verify_rows);
}

// ── the quant table ──

fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}
fn wrap(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn failed(e: anyerror) Verdict {
    return Verdict.named(.failed, @errorName(e));
}

fn specOf(ctx: *const abi.LoadCtx) ?*const Spec {
    return @ptrCast(@alignCast(ctx.state orelse return null));
}

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const root = block.expertQuantRoot(arena.allocator(), ctx.block_json.slice()) orelse return Verdict.refuse(.failed, "expert_quant block does not parse");
    const spec = sushi.parseExpertQuant(root) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const st = std.heap.page_allocator.create(Spec) catch |e| return failed(e);
    st.* = spec;
    ctx.state = st;
    return .ok;
}

fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    if (ctx.state) |p| std.heap.page_allocator.destroy(@as(*Spec, @ptrCast(@alignCast(p))));
    ctx.state = null;
}

fn noTensor(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: routed expert banks only");
}

fn loadBank(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadBankInto(ctx, name.slice(), in_dim, out) catch |e| return failed(e);
    return .ok;
}

/// `<base>.{trellis,suh,svh}`: trellis U16 [E, in/16, out/16, n], suh F16 [E, in], svh F16 [E, out]. The weight
/// carries its decode: `layout` the codebook, `flags` the window's bits, `block` the rate (halfwords per tile),
/// `state` the pack's table index + 1 (null: none).
fn loadBankInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    const spec = specOf(ctx) orelse return error.QuantFormatNotOpened;
    const t = tensor(ctx, base, "trellis");
    if (t.ctx == null) return error.MissingWeight;
    const shape = mlx.getShape(t);
    if (shape.len != 4) return error.Exl3TrellisGeometry;
    const p: Proj = .{ .trellis = t, .suh = tensor(ctx, base, "suh"), .svh = tensor(ctx, base, "svh") };
    const experts: u32 = @intCast(shape[0]);
    const in: u32 = @intCast(shape[1] * 16);
    if (in_dim != 0 and in_dim != in) return error.Exl3TrellisGeometry;
    try admit(p, experts, in, @intCast(shape[2] * 16), spec.*);
    out.* = .{};
    errdefer release(out);
    out.w = wrap(try share(p.trellis));
    out.scales = wrap(try share(p.suh));
    out.biases = wrap(try share(p.svh));
    out.layout = @backingInt(spec.codebook);
    out.flags = @intCast(spec.window.bits());
    out.block = @intCast(shape[3]);
    out.state = if (spec.table) |i| @ptrFromInt(@as(usize, i) + 1) else null;
}

fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

/// A new handle on a source array (the source keeps its own).
fn share(a: mlx.mlx_array) !mlx.mlx_array {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(arr(a));
    };
    w.* = .{};
}

fn noMatmul(_: *const abi.Weight, _: abi.Array, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: routed expert banks only");
}

fn noMoe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "exl3: a layer's banks run as one routed SwiGLU (swiglu)");
}

fn projOf(w: *const abi.Weight) Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

/// The decode spec a loaded bank carries; the three banks of a layer share it.
fn decodeOf(w: *const abi.Weight) ?Decode {
    const window = Window.fromBits(w.flags) orelse return null;
    const index: ?u8 = if (w.state) |p| @intCast(@intFromPtr(p) - 1) else null;
    return .{ .codebook = @fromBackingInt(@intCast(w.layout)), .window = window, .table = index };
}

fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const dec = decodeOf(gate) orelse return Verdict.refuse(.failed, "exl3: a bank without its decode spec");
    if (!std.meta.eql(decodeOf(up), dec) or !std.meta.eql(decodeOf(down), dec)) return Verdict.refuse(.unsupported, "exl3: a layer's banks decode differently");
    const bank: Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (opts.limit > 0)
        sushi.moeClamped(st, arr(x), bank, arr(inds), arr(scores), dec, opts.limit)
    else
        sushi.moe(st, arr(x), bank, arr(inds), arr(scores), dec, opts.verify_rows)) catch |e| return failed(e);
    out.* = wrap(y);
    return .ok;
}

/// Banks load as stored.
fn bill(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}

fn builtin() block.Builtin {
    for (block.builtins) |b| if (std.mem.eql(u8, b.name, "exl3")) return b;
    unreachable;
}

pub const table: abi.QuantTable = .{
    .name = Str.of("exl3"),
    .claims = builtin().claims,
    .validate = builtin().validate,
    .bill = &bill,
    .open = &open,
    .close = &close,
    .load_tensor = &noTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &noMatmul,
    .moe = &noMoe,
    .swiglu = &swiglu,
};

// ── tests ──

const testing = std.testing;
const ops = @import("../../../mlx/ops.zig");
const resident = @import("../../weights/resident.zig");

const D: c_int = 128;
const I: c_int = 128;
const E: c_int = 4;
const K: c_int = 2;

/// One projection's random bank under `<base>.{trellis,suh,svh}` in `w`, at `n` halfwords per tile.
fn putBank(w: *resident.Weights, rnd: std.Random, base: []const u8, in: c_int, out: c_int, n: c_int) !void {
    const words = try testing.allocator.alloc(u16, @intCast(E * @divExact(in, 16) * @divExact(out, 16) * n));
    defer testing.allocator.free(words);
    for (words) |*v| v.* = rnd.int(u16);
    var nb: [128]u8 = undefined;
    try w.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), mlx.mlx_array_new_data(words.ptr, &[_]c_int{ E, @divExact(in, 16), @divExact(out, 16), n }, 4, .uint16));
    for ([_]struct { leaf: []const u8, dim: c_int }{ .{ .leaf = "suh", .dim = in }, .{ .leaf = "svh", .dim = out } }) |side| {
        const h = try testing.allocator.alloc(u16, @intCast(E * side.dim));
        defer testing.allocator.free(h);
        for (h) |*v| v.* = sushi.format.f32ToF16Bits(if (rnd.boolean()) 0.125 else -0.125);
        try w.put(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, side.leaf }), mlx.mlx_array_new_data(h.ptr, &[_]c_int{ E, side.dim }, 2, .float16));
    }
}

const projs = [_][]const u8{ "l.gate_proj", "l.up_proj", "l.down_proj" };

const Case = struct {
    block: []const u8,
    /// Halfwords per tile of the gate, up and down banks.
    n: [3]c_int,
    /// What the engine built before the format: `Decode{ codebook, window }`, plus the table STQ needs.
    want: Decode,
    limit: c_int = 0,
};
