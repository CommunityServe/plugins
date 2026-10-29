//! copied_from: src/sdk/quant/native.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The MLX-native formats (`affine`, `fp8`, `mxfp8`, `nvfp4`): one load and one dispatch for all four, delegating to
//! the engine's per-weight resolution (`computeQuantParams`), `qmatmulBits` and `gatherExpertMm`. The formats differ
//! in what they claim, validate and bill (`block.zig`), and in the defaults their block declares.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const block = @import("block.zig");
const params = @import("params.zig");
const sushi_pack = @import("exl3/sushi_pack.zig");
const modelopt_pack = @import("nvfp4/modelopt_pack.zig");
const fp8_block128 = @import("fp8_block128.zig");
const qmatmulBits = @import("../blocks/linear/qmatmul.zig").qmatmulBits;
const gatherExpertMm = @import("../blocks/moe/experts.zig").gatherExpertMm;
const transposeBf16Weight = @import("../../mlx/array.zig").transposeBf16Weight;

const Str = abi.Str;
const Verdict = abi.Verdict;
const QuantMode = @import("params.zig").QuantMode;

/// `Weight.layout`: dense (pre-transposed `[in, out]`), an MLX mode, row-scaled e4m3 (`sushi_pack.fp8Linear`), or
/// e4m3 on a 64 x 128 scale grid (`fp8_block128.linear`).
pub const Layout = enum(u8) { dense, affine, nvfp4, mxfp4, mxfp8, fp8_rows, fp8_block128 };

/// The per-load state: the block's declaration as the engine's resolver reads it.
pub const State = struct {
    model: params.QuantParams = .{ .bits = 0, .group_size = 0 },
    /// Hadamard: the rotation block.
    block: u32 = 0,
};

pub fn arr(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}
pub fn wrap(a: mlx.mlx_array) abi.Array {
    return .{ .ctx = a.ctx };
}
fn stream(s: abi.Stream) mlx.mlx_stream {
    return .{ .ctx = s.ctx };
}

/// A Zig error as a refusal (an error's name is static text).
pub fn failed(e: anyerror) Verdict {
    return .{ .code = .failed, .why = Str.of(@errorName(e)) };
}

pub fn modeOf(w: *const abi.Weight) QuantMode {
    return switch (@as(Layout, @enumFromInt(w.layout))) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => .affine,
    };
}

pub fn state(ctx: *const abi.LoadCtx) ?*const State {
    return @ptrCast(@alignCast(ctx.state orelse return null));
}

pub fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const d = block.declared(arena.allocator(), ctx.block_json.slice());
    const mode = QuantMode.fromString(d.mode) orelse return Verdict.refuse(.unsupported, "quantization mode other than affine, nvfp4, mxfp4, mxfp8");
    const st = std.heap.page_allocator.create(State) catch return failed(error.OutOfMemory);
    st.* = .{};
    st.model = .{ .bits = d.bits, .group_size = d.group_size, .mode = mode };
    if (block.hadamardBlock(arena.allocator(), ctx.config_json.slice())) |b| st.block = b;
    ctx.state = st;
    return .ok;
}

pub fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    if (ctx.state) |p| std.heap.page_allocator.destroy(@as(*State, @ptrCast(@alignCast(p))));
    ctx.state = null;
}

/// A new handle on a source array (the source keeps its own).
fn share(a: mlx.mlx_array) !mlx.mlx_array {
    if (a.ctx == null) return a;
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

/// `<base>.<leaf>` from the source, empty when absent.
pub fn tensor(ctx: *const abi.LoadCtx, base: []const u8, leaf: []const u8) mlx.mlx_array {
    var nb: [320]u8 = undefined;
    const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }) catch return .{};
    return arr(ctx.source.tensor(name));
}

/// The triple at `base`, its quant params resolved per weight as the engine's `quantParamsHinted` does.
pub fn loadInto(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) !void {
    const st = state(ctx) orelse return error.QuantFormatNotOpened;
    const w = tensor(ctx, base, "weight");
    if (w.ctx == null) return error.MissingWeight;
    if (mlx.getShape(w).len < 2) return error.NotAProjection;
    const sc = tensor(ctx, base, "scales");
    const bi = tensor(ctx, base, "biases");
    out.* = .{};
    errdefer release(out);
    const grid = tensor(ctx, base, "weight_scale_inv");
    if (sc.ctx == null and fp8_block128.isBlockScaled(w, grid)) {
        out.w = wrap(try share(w));
        out.scales = wrap(try share(grid));
        out.layout = @intFromEnum(Layout.fp8_block128);
        out.bits = 8;
        out.group_size = fp8_block128.block_cols;
        out.block = fp8_block128.block_rows;
        return;
    }
    if (sc.ctx == null) {
        out.w = wrap(try transposeBf16Weight(w, stream(ctx.stream)));
        out.layout = @intFromEnum(Layout.dense);
        return;
    }
    out.w = wrap(try share(w));
    out.scales = wrap(try share(sc));
    out.biases = wrap(try share(bi));
    if (sushi_pack.isRowScaledFp8(w, sc)) {
        out.layout = @intFromEnum(Layout.fp8_rows);
        out.bits = 8;
        return;
    }
    const qp = params.computeQuantParams(st.model, w, sc, if (in_dim > 0) in_dim else null);
    out.bits = @intCast(qp.bits);
    out.group_size = qp.group_size;
    out.layout = @intFromEnum(switch (qp.mode) {
        .affine => Layout.affine,
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        .gguf => return error.GgufIsNotAnMlxMode,
    });
}

pub fn loadTensor(ctx: *const abi.LoadCtx, name: Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    loadInto(ctx, name.slice(), in_dim, out) catch |e| return failed(e);
    return .ok;
}

pub fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(arr(a));
    };
    w.* = .{};
}

pub fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const y = qmatmulBits(arr(x), arr(w.w), arr(w.scales), arr(w.biases), w.bits, w.group_size, modeOf(w), stream(s)) catch |e| return failed(e);
    out.* = wrap(y);
    return .ok;
}

pub fn moe(bank: *const abi.Weight, x: abi.Array, lhs: abi.Array, rhs: abi.Array, sorted: bool, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (bank.layout == @intFromEnum(Layout.fp8_rows) or bank.layout == @intFromEnum(Layout.fp8_block128)) return Verdict.refuse(.unsupported, "fp8: e4m3 expert banks");
    var res = mlx.mlx_array_new();
    gatherExpertMm(&res, arr(x), arr(bank.w), arr(bank.scales), arr(bank.biases), arr(lhs), arr(rhs), bank.bits, bank.group_size, modeOf(bank), sorted, stream(s)) catch |e| {
        _ = mlx.mlx_array_free(res);
        return failed(e);
    };
    out.* = wrap(res);
    return .ok;
}

/// Stored as the pack holds it.
pub fn billStored(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}

/// A ModelOpt pack decodes its csf-compressed expert scales at load.
fn billNvfp4(block_json: Str, model_dir: Str, stored: u64) callconv(.c) u64 {
    if (!block.isModelOpt(block_json.slice())) return stored;
    return stored +| modelopt_pack.bankExpansionBytes(model_dir.slice());
}

fn builtin(comptime name: []const u8) block.Builtin {
    for (block.builtins) |b| if (std.mem.eql(u8, b.name, name)) return b;
    unreachable;
}

/// The table of the built-in `name` over the shared load and dispatch.
pub fn table(comptime name: []const u8, comptime bill: anytype) abi.QuantTable {
    const b = comptime builtin(name);
    return .{
        .name = Str.of(name),
        .claims = b.claims,
        .validate = b.validate,
        .bill = bill,
        .open = &open,
        .close = &close,
        .load_tensor = &loadTensor,
        .load_bank = &loadTensor,
        .release = &release,
        .matmul = &matmul,
        .moe = &moe,
    };
}

pub const affine = table("affine", &billStored);
pub const fp8 = table("fp8", &billStored);
pub const mxfp8 = table("mxfp8", &billStored);
pub const nvfp4 = table("nvfp4", &billNvfp4);

// ── tests ──

const testing = std.testing;
const ops = @import("../../mlx/ops.zig");
const parity = @import("../testing/parity.zig");
const resident = @import("../weights/resident.zig");

pub const Triple = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array = .{ .ctx = null },
    bi: mlx.mlx_array = .{ .ctx = null },
};

/// `w` quantized by MLX in `mode` (affine carries biases; the fp modes do not).
pub fn quantized(w: mlx.mlx_array, mode: [*:0]const u8, bits: u32, gs: u32, s: mlx.mlx_stream) !Triple {
    var v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_quantize(&v, w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), mode, .{}, s));
    var t: Triple = .{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&t.w, v, 0));
    try mlx.check(mlx.mlx_vector_array_get(&t.sc, v, 1));
    if (mlx.mlx_vector_array_size(v) > 2) {
        t.bi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&t.bi, v, 2));
    }
    return t;
}

/// Hands `t` to `weights` under `base` (the map owns it from here).
pub fn putTriple(weights: *resident.Weights, base: []const u8, t: Triple) !void {
    var nb: [256]u8 = undefined;
    try weights.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), t.w);
    if (t.sc.ctx != null) try weights.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), t.sc);
    if (t.bi.ctx != null) try weights.put(try std.fmt.bufPrint(&nb, "{s}.biases", .{base}), t.bi);
}

pub const Case = struct { base: []const u8, layout: Layout, bits: u8 = 0, group_size: u32 = 0 };

/// Loads every case of `r` through `t` and checks the resolved params, then that `t.matmul` at 1, 3 and 9 rows is
/// bit-equal to the direct path (`quantParamsOrDense` from the activation, then `qmatmulBits`).
fn expectParity(t: *const abi.QuantTable, r: *const resident.Resident, block_json: []const u8, cases: []const Case, k: c_int, s: mlx.mlx_stream) !void {
    const src = r.source();
    var ctx: abi.LoadCtx = .{ .block_json = .of(block_json), .source = &src, .stream = .{ .ctx = s.ctx } };
    try testing.expect(t.open(&ctx).isOk());
    defer t.close(&ctx);
    var prng = std.Random.DefaultPrng.init(0x9a41);
    for (cases) |c| {
        var w: abi.Weight = .{};
        const v = t.load_tensor(&ctx, .of(c.base), @intCast(k), &w);
        if (!v.isOk()) std.debug.print("{s}: {s}\n", .{ c.base, v.why.slice() });
        try testing.expect(v.isOk());
        defer t.release(&w);
        try testing.expectEqual(@intFromEnum(c.layout), w.layout);
        if (c.bits != 0) try testing.expectEqual(c.bits, w.bits);
        if (c.group_size != 0) try testing.expectEqual(c.group_size, w.group_size);
        const direct = tensor(&ctx, c.base, "weight");
        const dense_t = if (c.layout == .dense) try transposeBf16Weight(direct, s) else direct;
        defer if (c.layout == .dense) {
            _ = mlx.mlx_array_free(dense_t);
        };
        for ([_]c_int{ 1, 3, 9 }) |rows| {
            const x = try parity.testRandWeightBf16(prng.random(), &.{ 1, rows, k }, s);
            defer _ = mlx.mlx_array_free(x);
            var got: abi.Array = .{};
            try testing.expect(t.matmul(&w, wrap(x), .{ .ctx = s.ctx }, &got).isOk());
            defer _ = mlx.mlx_array_free(arr(got));
            const sc = tensor(&ctx, c.base, "scales");
            const qp = params.quantParamsOrDense(state(&ctx).?.model, dense_t, sc, @intCast(k));
            const want = try qmatmulBits(x, dense_t, sc, tensor(&ctx, c.base, "biases"), qp.bits, qp.group_size, qp.mode, s);
            defer _ = mlx.mlx_array_free(want);
            try ops.expectBitEqual(arr(got), want);
        }
    }
}

const N: c_int = 128;
const K: c_int = 256;

fn randW(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return parity.testRandWeightBf16(rnd, shape, s);
}

/// One weight quantized in `mode` under `base` in `weights`.
fn addQuantized(weights: *resident.Weights, rnd: std.Random, base: []const u8, mode: [*:0]const u8, bits: u32, gs: u32, s: mlx.mlx_stream) !void {
    const w = try randW(rnd, &.{ N, K }, s);
    defer _ = mlx.mlx_array_free(w);
    try putTriple(weights, base, try quantized(w, mode, bits, gs, s));
}

/// A row-scaled e4m3 weight as Sushi's adaptation leaves a QKV part: codes `[n, k]` (finite, both signs) and one f32
/// scale row per code row.
fn rowScaledFp8(rnd: std.Random, n: c_int, k: c_int) !Triple {
    const codes = try testing.allocator.alloc(u8, @intCast(n * k));
    defer testing.allocator.free(codes);
    for (codes) |*c| c.* = rnd.uintLessThan(u8, 0x78) | (if (rnd.boolean()) @as(u8, 0x80) else 0);
    const sv = try testing.allocator.alloc(f32, @intCast(n * @divExact(k, 128)));
    defer testing.allocator.free(sv);
    for (sv) |*v| v.* = 0.001 + 0.01 * rnd.float(f32);
    return .{
        .w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, k }, 2, .uint8),
        .sc = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32),
    };
}
