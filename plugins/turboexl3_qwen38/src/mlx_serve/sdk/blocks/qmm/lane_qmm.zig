//! copied_from: src/sdk/blocks/qmm/lane_qmm.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Row-exact 4-, 5-, 6- and 8-bit matmul on the M5 tensor units for
//! 1..MAX_ROWS rows, ported from TensorFold's `lane_qmm.py` and `lane_widen.py`
//! (MIT, see NOTICE). Every 64- (or 32-) input group runs one fixed 16-row
//! `matmul2d` (32 rows past 16, in 32-row blocks across threadgroups) over the
//! packed codes (8-bit as bytes, 5- and 6-bit widened to bytes per group), then
//! `C = s * P + b * XS` in group order in fp32, the K slices summed in slice
//! order; the slice count follows the weight's shape only. A row's bits never
//! depend on how many rows ride with it, so a drafted window verifies with the
//! one-row step's bits. 4-bit weights whose N is a multiple of 64 take the
//! 64-column tile two simdgroups run together (`COOP`), the rest 32 columns (`NARROW`),
//! both over MLX's packed layout or, once `tileInPlace` re-ordered a weight in
//! its own buffer, the tiled one (each column tile's group one contiguous block,
//! scales and biases group-major); all four give the same bits. A tiled weight
//! is read by nothing else: past MAX_ROWS rows (a prompt) it takes `PREFILL`,
//! MLX's NAX qmm with the tiled addressing.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const steel = @import("mlx_steel_sources");

pub const MAX_ROWS = 128;

pub const HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");

// XS (a row's sum over a group) is summed in the kernel in a fixed order per
// row: lane 2r + h adds half h of row r, and the two halves join low + high.
const ROW_SUMS = kernelSource("kernels/qmm/lane_qmm_row_sums.metal");
const ROW_SUM_G = kernelSource("kernels/qmm/lane_qmm_row_sum_g.metal");

// 32 output columns a simdgroup; SBt holds (s, b) bf16 pairs group-major [K/GS][N][2].
// A threadgroup covers 16 x TMR rows from `rb` with one op per group; every row
// gets the 16-row op's bits.
pub const NARROW = kernelSource("kernels/qmm/lane_qmm_narrow_head.metal") ++ "\n" ++ ROW_SUMS ++ "  for (int g = g_begin; g < g_end; g++) {\n" ++ ROW_SUM_G ++ kernelSource("kernels/qmm/lane_qmm_narrow_body.metal");

// 64 output columns run by two simdgroups together (N % 64 == 0); a pair per K slice.
pub const COOP = kernelSource("kernels/qmm/lane_qmm_coop_head.metal") ++ "\n" ++ ROW_SUMS ++ "  for (int g = g_begin; g < g_end; g++) {\n" ++ ROW_SUM_G ++ kernelSource("kernels/qmm/lane_qmm_coop_body.metal");

// MLX's affine `qmm_t_nax` (lib/mlx-src/.../quantized_nax.h, MIT) for a tiled
// weight: the same BM x 64 x 64 blocks over 2 x 2 simdgroups, the same
// dequantization into threadgroup memory and the same NAX tile matmuls; only
// the weight and scale addresses differ (and rows past N load zeros).
pub const PREFILL = kernelSource("kernels/qmm/lane_qmm_prefill.metal");

/// MLX's steel NAX headers in include order, their own includes and
/// include guards dropped: they sit inline in one `metal_kernel` header.
pub fn prefillHeader() ![:0]const u8 {
    if (prefill_header) |h| return h;
    const a = std.heap.c_allocator;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n");
    for ([_][]const u8{ steel.defines, steel.type_traits, steel.integral_constant, steel.nax }) |src| {
        var lines = std.mem.splitScalar(u8, src, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "#include \"mlx/") or std.mem.startsWith(u8, line, "#pragma once")) continue;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
        }
    }
    try out.appendSlice(a, "using namespace metal;\nusing namespace mlx::steel;\n");
    const h = try out.toOwnedSliceSentinel(a, 0);
    prefill_header = h;
    return h;
}
var prefill_header: ?[:0]const u8 = null;

/// K slices for an (n, k) weight: fixed by the shape, never by the row count.
fn splitK(n: c_int, k: c_int) c_int {
    const tiles = @divTrunc(n + 31, 32);
    var sk: c_int = 1;
    while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divTrunc(k, 64), sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The pick follows the weight's shape only: 64-column tiles where N allows.
fn coopFor(n: c_int) bool {
    return @rem(n, 64) == 0;
}

const Kind = enum { narrow, coop, prefill };
/// `tmr` is the prefill block's BM there; `ntot`/`r0` place a tiled view in its buffer.
const KernelKey = struct { kind: Kind, tiled: bool, tmr: c_int, k: c_int, n: c_int, gs: c_int, sk: c_int, ntot: c_int = 0, r0: c_int = 0, nt: c_int = 0, bits: c_int = 4 };

/// 16-row blocks a threadgroup's op covers: one up to 16 rows, two past it.
fn tmrFor(rows: c_int) c_int {
    return if (rows <= 16) 1 else 2;
}
var kernels: std.AutoHashMapUnmanaged(KernelKey, mlx.mlx_fast_metal_kernel) = .{};
const PlanKey = struct { kind: Kind, rows: c_int, n: c_int, k: c_int };
/// Launch plans and row counts at lane widths (rows <= MAX_ROWS). A
/// prompt-width call builds its own and frees it: a long session's chunk
/// tails take every row count there is.
var plans: std.AutoHashMapUnmanaged(PlanKey, mlx.mlx_fast_metal_kernel_config) = .{};
var mdims_cache: std.AutoHashMapUnmanaged(c_int, mlx.mlx_array) = .{};

/// (s, b) pairs group-major per (scales, biases) of a weight read in MLX's
/// layout, built on first use (an eighth of the weight's bytes: the lm_head
/// and the ragged GDN gates on a tiled trunk). Keyed by the data the handles
/// point at (a handle's address is reused once its caller frees it) plus the
/// shape (a joined weight and its first part start at the same address); an
/// entry holds its sources alive, so no key is reused.
const DerivedKey = struct { a: usize, b: usize, n: c_int, w: c_int };
const Derived = struct { src_a: mlx.mlx_array, src_b: mlx.mlx_array, out: mlx.mlx_array };
var packed_scales: std.AutoHashMapUnmanaged(DerivedKey, Derived) = .{};

/// A weight (or a row view of one) whose buffer `tileInPlace` re-ordered: the
/// base handles it lives in, its first row there, and the column tile width.
/// The entry holds the base alive, so its addresses are never reused.
const Tiled = struct { owner: usize, w: mlx.mlx_array, st: mlx.mlx_array, bt: mlx.mlx_array, ntot: c_int, r0: c_int, nt: c_int, bits: c_int };
const TiledKey = struct { ptr: usize, n: c_int, kw: c_int };
var tiled: std.AutoHashMapUnmanaged(TiledKey, Tiled) = .{};
/// Registered weights per shape: a weight of any other shape is answered
/// without a data read (a lazy weight's eval mid-graph).
const TiledShape = struct { n: c_int, kw: c_int };
var tiled_shapes: std.AutoHashMapUnmanaged(TiledShape, u32) = .{};

fn shapeReleased(key: TiledKey) void {
    const count = tiled_shapes.getPtr(.{ .n = key.n, .kw = key.kw }) orelse return;
    count.* -= 1;
    if (count.* == 0) _ = tiled_shapes.remove(.{ .n = key.n, .kw = key.kw });
}

/// Frees `owner`'s tiled entries and EVERY model's derived (s, b) copies (a
/// model unload): a copy holds its sources alive and nothing says whose they
/// are, and another model rebuilds its own on the next read. That model's
/// tiled weights stay registered.
pub fn release(owner: usize) void {
    var it = packed_scales.valueIterator();
    while (it.next()) |e| {
        _ = mlx.mlx_array_free(e.src_a);
        if (e.src_b.ctx != null) _ = mlx.mlx_array_free(e.src_b);
        _ = mlx.mlx_array_free(e.out);
    }
    packed_scales.clearAndFree(std.heap.c_allocator);
    var keys: std.ArrayList(TiledKey) = .empty;
    defer keys.deinit(std.heap.c_allocator);
    var ti = tiled.iterator();
    while (ti.next()) |e| if (e.value_ptr.owner == owner) keys.append(std.heap.c_allocator, e.key_ptr.*) catch {};
    for (keys.items) |k| if (tiled.fetchRemove(k)) |kv| {
        for ([_]mlx.mlx_array{ kv.value.w, kv.value.st, kv.value.bt }) |h| _ = mlx.mlx_array_free(h);
        shapeReleased(k);
    };
}

fn tiledEntry(w: mlx.mlx_array) !?Tiled {
    if (tiled.count() == 0 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const sh = mlx.getShape(w);
    if (sh.len != 2 or !tiled_shapes.contains(.{ .n = sh[0], .kw = sh[1] })) return null;
    try mlx.check(mlx.mlx_array_eval(w));
    const p = @intFromPtr(mlx.mlx_array_data_uint32(w) orelse return null);
    return tiled.get(.{ .ptr = p, .n = sh[0], .kw = sh[1] });
}

fn kernelFor(key: KernelKey) !mlx.mlx_fast_metal_kernel {
    if (kernels.get(key)) |k| return k;
    const a = std.heap.c_allocator;
    const consts = try std.fmt.allocPrint(a, "#define BITS {d}\n#define TILED {d}\n  constexpr int TMR = {d};\n  constexpr int BM = {d};\n  constexpr int K = {d};\n  constexpr int N = {d};\n  constexpr int GS = {d};\n  constexpr int SK = {d};\n  constexpr int NTOT = {d};\n  constexpr int R0 = {d};\n  constexpr int NT = {d};\n  constexpr bool ALIGNED_N = {};\n", .{ key.bits, @intFromBool(key.tiled), key.tmr, key.tmr, key.k, key.n, key.gs, key.sk, key.ntot, key.r0, key.nt, @rem(key.n, 64) == 0 });
    defer a.free(consts);
    const body = switch (key.kind) {
        .narrow => NARROW,
        .coop => COOP,
        .prefill => PREFILL,
    };
    const source = try std.mem.concatWithSentinel(a, u8, &.{ consts, body }, 0);
    defer a.free(source);
    const name = try std.fmt.allocPrintSentinel(a, "msv_lane_qmm_{t}{s}_b{d}_t{d}_k{d}_n{d}_g{d}_s{d}_o{d}_{d}", .{ key.kind, if (key.tiled) "_tiled" else "", key.bits, key.tmr, key.k, key.n, key.gs, key.sk, key.ntot, key.r0 }, 0);
    defer a.free(name);
    const in_plain = [_][*:0]const u8{ "X", "W", "SBt", "mdims" };
    const in_tiled = [_][*:0]const u8{ "X", "W", "ST", "BT", "mdims" };
    const in_names: []const [*:0]const u8 = if (key.tiled) &in_tiled else &in_plain;
    const out_names = [_][*:0]const u8{"Y"};
    const in_vec = mlx.mlx_vector_string_new_data(in_names.ptr, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const header: [*:0]const u8 = if (key.kind == .prefill) (try prefillHeader()).ptr else HEADER;
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    try kernels.put(a, key, k);
    return k;
}

fn planFor(key: PlanKey) !mlx.mlx_fast_metal_kernel_config {
    if (plans.get(key)) |p| return p;
    const cfg = try buildPlan(key);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try plans.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

fn buildPlan(key: PlanKey) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, key.n }, 2, .bfloat16));
    if (key.kind == .prefill) {
        const bm = prefillBm(key.rows);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + 63, 64) * 128, @divTrunc(key.rows + bm - 1, bm), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    } else {
        const sk = splitK(key.n, key.k);
        const width: c_int = if (key.kind == .coop) 64 else 32;
        const block = 16 * tmrFor(key.rows);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + width - 1, width) * width * sk, @divTrunc(key.rows + block - 1, block), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, width * sk, 1, 1));
    }
    return cfg;
}

/// MLX's block height: 32 rows when one block covers them all.
fn prefillBm(rows: c_int) c_int {
    return if (rows <= 32) 32 else 64;
}

fn mdimsFor(rows: c_int) !mlx.mlx_array {
    if (mdims_cache.get(rows)) |m| return m;
    const m = newMdims(rows);
    errdefer _ = mlx.mlx_array_free(m);
    try mdims_cache.put(std.heap.c_allocator, rows, m);
    return m;
}

fn newMdims(rows: c_int) mlx.mlx_array {
    const d = [_]i32{rows};
    return mlx.mlx_array_new_data(&d, &[_]c_int{1}, 1, .int32);
}

fn run(key: KernelKey, rows: c_int, x: mlx.mlx_array, xs: []const c_int, weights: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, key.k }, 2, s));
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x2, false, s));
    const mk = try kernelFor(key);
    const cached = rows <= MAX_ROWS;
    const pkey: PlanKey = .{ .kind = key.kind, .rows = rows, .n = key.n, .k = key.k };
    const mcfg = if (cached) try planFor(pkey) else try buildPlan(pkey);
    defer if (!cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(mcfg);
    };
    const mdims = if (cached) try mdimsFor(rows) else newMdims(rows);
    defer if (!cached) {
        _ = mlx.mlx_array_free(mdims);
    };
    var ins: [5]mlx.mlx_array = undefined;
    ins[0] = xc;
    @memcpy(ins[1 .. 1 + weights.len], weights);
    ins[1 + weights.len] = mdims;
    const in_vec = mlx.mlx_vector_array_new_data(&ins, weights.len + 2);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, mk, in_vec, mcfg, s));
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outs, 0));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    shape[xs.len - 1] = key.n;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, xs.len, s));
    return y;
}

/// `x [..., K] @ w.T` for a weight `tileInPlace` re-ordered, at any row count;
/// null for every other weight. Up to MAX_ROWS rows a tile-aligned view takes
/// the lane kernel's bits, past it (or unaligned) MLX's prompt-width qmm.
pub fn tiledQmm(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    const t = (try tiledEntry(w)) orelse return null;
    if (!mlx.streamIsGpu(s)) return error.TiledWeightOffGpu;
    const xs = mlx.getShape(x);
    const n = mlx.getShape(w)[0];
    const k = @divExact(mlx.getShape(w)[1] * 32, t.bits);
    if (xs.len == 0 or xs.len > 8 or xs[xs.len - 1] != k or mlx.mlx_array_dtype(x) != .bfloat16) return error.TiledWeightShape;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    const coop = t.nt == 64;
    const lane = rows <= MAX_ROWS and @rem(t.r0, t.nt) == 0 and (!coop or coopFor(n));
    const key: KernelKey = if (lane)
        .{ .kind = if (coop) .coop else .narrow, .tiled = true, .tmr = tmrFor(rows), .k = k, .n = n, .gs = 64, .sk = splitK(n, k), .ntot = t.ntot, .r0 = t.r0, .bits = t.bits }
    else
        .{ .kind = .prefill, .tiled = true, .tmr = prefillBm(rows), .k = k, .n = n, .gs = 64, .sk = 1, .ntot = t.ntot, .r0 = t.r0, .nt = t.nt, .bits = t.bits };
    return try run(key, rows, x, xs, &.{ t.w, t.st, t.bt }, s);
}

const testing = std.testing;
