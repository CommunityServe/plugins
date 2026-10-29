//! copied_from: src/sdk/quant/exl3/sushi/expert_exl3_kernels.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The EXL3 Metal kernels and their dispatch: decode GEMVs, prefill GEMMs and the routed SwiGLU chains.
const std = @import("std");
const host_env = @import("mlx_host").env;
const mlx = @import("mlx_host").mlx;
const log = @import("mlx_host").log;
const io_util = @import("mlx_host").io_util;
const exl3 = @import("expert_exl3.zig");
const codebook_tables = @import("codebook_tables.zig");

var ubench_mute: bool = false;
var ubench_env: ?bool = null;
var pair_splits_force: ?u32 = null;
var swiglu_maxabs_env: ?bool = null;
var swiglu_maxabs_dumped: u32 = 0;

fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

fn pairSplitCount() u32 {
    if (pair_splits_force) |v| return v;
    return 2;
}

/// A pair-GEMV threadgroup prepares its own slice of x, so the split must cut
/// the hidden dim on a 128-wide Hadamard block boundary.
pub fn pairSplitCountFor(in_dim: c_int) u32 {
    const n = pairSplitCount();
    if (@rem(in_dim, @as(c_int, @intCast(n)) * 128) != 0) return 1;
    return n;
}

/// Routes the decode GEMVs through the generic window reader, one tile per
/// threadgroup: the byte reference the lane funnel's layout is held to.
var funnel_off_for_test: bool = false;

/// Every prefill reader below K4 reads its weights through funnels whose shifts follow
/// from n; K4 keeps its own packed prefill branch, the same reads at a word-aligned rate.
fn funnelReads(n: u32) bool {
    return n != 64;
}

/// How a decode GEMV threadgroup (`FUNNEL`, `OTPT`) reads the bank. The lane
/// funnel carries two output tiles per threadgroup, sharing index math, inputs
/// and the prepare; the other readers carry one.
pub const GemvLayout = struct { funnel: bool, tiles: c_int };

pub fn gemvLayout(out_tiles: c_int) GemvLayout {
    const funnel = !funnel_off_for_test;
    return .{ .funnel = funnel, .tiles = if (funnel and @rem(out_tiles, 2) == 0) 2 else 1 };
}

pub fn addGemvLayout(cfg: mlx.mlx_fast_metal_kernel_config, layout: GemvLayout) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FUNNEL", @intFromBool(layout.funnel)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OTPT", layout.tiles));
}

/// Slots one expert-grouped threadgroup leads: a wider group's accumulators spill.
const DECODE_GROUP_MEMBERS: c_int = 2;
/// The grouped kernels find an expert's slots with one 64-bit ballot mask.
const DECODE_GROUP_MAX_SLOTS: c_int = 64;
/// Routes a multi-row block through the single-slot kernels: the grouped kernels' byte reference.
var grouped_off_for_test: bool = false;

fn addGroup(cfg: mlx.mlx_fast_metal_kernel_config, group: c_int, nslots: c_int) !void {
    if (group == 0) return;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GROUP", group));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSLOTS", nslots));
}

fn logGroupedEngaged(group: c_int) void {
    if (group == 0 or grouped_engaged) return;
    grouped_engaged = true;
    log.info("[exl3-decode] expert-grouped rows engaged group={d}\n", .{group});
}

fn exl3UbenchOn() bool {
    if (ubench_mute) return false;
    if (ubench_env) |v| return v;
    const v = diagEnvValueOn(host_env.get(.SUSHI_EXL3_LAYER_UBENCH));
    ubench_env = v;
    return v;
}

fn benchPrint(comptime fmt: []const u8, args: anytype) void {
    if (!exl3UbenchOn()) return;
    std.debug.print(fmt, args);
}

fn ubenchEval(a: mlx.mlx_array, name: []const u8) !void {
    if (!exl3UbenchOn()) return;
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(a));
    const ns = sw.read();
    benchPrint("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
    log.info("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
}

fn swigluMaxabsOn() bool {
    if (swiglu_maxabs_env) |v| return v;
    const v = diagEnvValueOn(host_env.get(.SUSHI_EXL3_SWIGLU_MAXABS));
    swiglu_maxabs_env = v;
    return v;
}

fn dumpAbsMax(s: mlx.mlx_stream, a: mlx.mlx_array, name: []const u8) !void {
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, a, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    try mlx.check(mlx.mlx_array_eval(mx));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx));
    log.info("[exl3-maxabs] {s} {d:.6}\n", .{ name, v });
}

pub const DECODE_ROWS_MAX: usize = 16;

/// The trellis rate a packed last dim carries: n halfwords per 256-weight tile,
/// K = n/16. Every kernel keys on n, never on an integer K.
fn packedRate(last: c_int) !exl3.Rate {
    if (last < 0) return error.BadExl3Shape;
    return exl3.kFromPackedDim(@intCast(last)) orelse error.BadExl3Shape;
}

pub const UNION_EXPERTS: usize = 512;

pub fn unionMultiplicity(eids: []const u32, counts: []u32) u32 {
    var unique: u32 = 0;
    for (eids) |e| {
        if (e >= counts.len) continue;
        if (counts[e] == 0) unique += 1;
        counts[e] += 1;
    }
    return unique;
}

var union_hist_env: ?bool = null;

fn unionHistOn() bool {
    if (union_hist_env) |v| return v;
    const v = diagEnvValueOn(host_env.get(.SUSHI_EXL3_UNION_HIST));
    union_hist_env = v;
    return v;
}

pub fn dumpUnionHist(slots_u: mlx.mlx_array, S: usize, K: usize) !void {
    if (!unionHistOn()) return;
    if (S < 2 or K == 0) return;
    // The caller swallows this error, so the latch this eval may raise is ours
    // to drop: left standing it becomes the next decode tick's `MlxFailure`.
    const had_error = mlx.errorPending();
    mlx.check(mlx.mlx_array_eval(slots_u)) catch |e| {
        mlx.dropLatchedErrorUnless(had_error);
        return e;
    };
    const n = S * K;
    const ptr = mlx.mlx_array_data_uint32(slots_u) orelse return;
    const slice = ptr[0..n];
    var counts: [UNION_EXPERTS]u32 = @splat(0);
    const unique = unionMultiplicity(slice, &counts);
    var shared: u32 = 0;
    var max_m: u32 = 0;
    for (counts) |c| {
        if (c >= 2) shared += 1;
        if (c > max_m) max_m = c;
    }
    log.info("[exl3-union] S={d} K={d} assignments={d} unique={d} shared={d} max_mult={d}\n", .{ S, K, n, unique, shared, max_m });
    if (S <= DECODE_ROWS_MAX) return;
    // A prefill layer's per-expert row counts, the input `SUSHI_EXL3_GEMM_COUNTS` replays.
    var buf: [UNION_EXPERTS * 6]u8 = undefined;
    var used: usize = 0;
    for (counts) |c| used += (std.fmt.bufPrint(buf[used..], "{d},", .{c}) catch break).len;
    log.info("[exl3-counts] {s}\n", .{buf[0..used]});
}

pub const RunTable = struct {
    start: []u32,
    len: []u32,
    eid: []u32,
    n: u32,
};

pub fn buildRuns(alloc: std.mem.Allocator, experts: []const u32) !RunTable {
    if (experts.len == 0) return .{ .start = &.{}, .len = &.{}, .eid = &.{}, .n = 0 };
    var n: u32 = 1;
    var i: usize = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) n += 1;
    }
    const start = try alloc.alloc(u32, n);
    const len = try alloc.alloc(u32, n);
    const eid = try alloc.alloc(u32, n);
    var r: u32 = 0;
    start[0] = 0;
    eid[0] = experts[0];
    i = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) {
            len[r] = @intCast(i - start[r]);
            r += 1;
            start[r] = @intCast(i);
            eid[r] = experts[i];
        }
    }
    len[r] = @intCast(experts.len - start[r]);
    return .{ .start = start, .len = len, .eid = eid, .n = n };
}

const GEMV_SOURCE: [:0]const u8 =
    @embedFile("kernels/gemv.metal");

const GEMM_SORTED_SOURCE: [:0]const u8 =
    @embedFile("kernels/gemm_sorted.metal");
/// The simdgroup-matrix body's 8x8 fragment readers. A lane at fragment coordinate (fm, fn)
/// owns tile slot group g = 4*fm + fn/2: its pair j, slots (8g+2j, 8g+2j+1), is W^T block
/// (n half j>>1, k half j&1) at (fm, fn) and (fm, fn+1).
const GEMM_SIMDMAT_FRAGS: [:0]const u8 =
    @embedFile("kernels/gemm_simdmat_frags.metal");
/// A data-dependent block count spills the accumulators, so use compile-time WIN.
/// Short runs repeat their last row; padded rows are never stored.
const GEMM_SIMDMAT_SOURCE: [:0]const u8 =
    @embedFile("kernels/gemm_simdmat.metal");
const GEMM_NAX_INCLUDES: [:0]const u8 =
    @embedFile("kernels/gemm_nax_includes.metal");
const GEMM_NAX_FRAGS: [:0]const u8 =
    @embedFile("kernels/gemm_nax_frags.metal");

/// The branch-guarded NAX body GEMM_NAX_SOURCE must reproduce byte for byte (tests only).
const GEMM_NAX_REFERENCE_SOURCE: [:0]const u8 =
    @embedFile("kernels/gemm_nax_reference.metal");

/// The NAX body. A lane's x rows are clamped into the input, so a padded row reads a live
/// neighbour whose product is never stored and the loads carry no bounds branch; the k loop is
/// unswitched on the window's second 16-row block. Bytes equal GEMM_NAX_REFERENCE_SOURCE's.
const GEMM_NAX_SOURCE: [:0]const u8 = blk: {
    const decode_lo =
        @embedFile("kernels/gemm_nax_decode_lo.metal");
    const open_loop =
        @embedFile("kernels/gemm_nax_open_loop.metal");
    break :blk @embedFile("kernels/gemm_nax.metal") ++ open_loop ++ decode_lo ++
        \\  nax_left_rows(left, xh0 + tk * TILE, xh1 + tk * TILE);
        \\  op.run(left, right, dest_hi);
        \\}
        \\} else {
        \\
    ++ open_loop ++ decode_lo ++
        \\}
        \\}
        \\nax_store(destination, y, run0, uint(ODIM), output_base, origin, nlive);
        \\if (n_hi > 0u) nax_store(dest_hi, y, run0 + TILE, uint(ODIM), output_base, origin, n_hi);
        \\row = run_end;
        \\}
    ;
};

/// Test and prefill-ubench seam: dispatch GEMM_NAX_REFERENCE_SOURCE instead.
pub var nax_reference_override: bool = false;
var gemm_nax_ref_kernel: KernelSlots = no_kernels;

const TOKEN_PREPARE_SOURCE: [:0]const u8 =
    @embedFile("kernels/token_prepare.metal");

const TOKEN_PAIR_PREPARE_SOURCE: [:0]const u8 =
    @embedFile("kernels/token_pair_prepare.metal");

const TOKEN_SCATTER_SOURCE: [:0]const u8 =
    @embedFile("kernels/token_scatter.metal");

const TOKEN_REDUCE_SOURCE: [:0]const u8 =
    @embedFile("kernels/token_reduce.metal");

const PREPARE_SOURCE: [:0]const u8 =
    @embedFile("kernels/prepare.metal");

const FINISH_SOURCE: [:0]const u8 =
    @embedFile("kernels/finish.metal");

var gemm_sorted_kernel: KernelSlots = no_kernels;
var gemm_simdmat_kernel: KernelSlots = no_kernels;
/// Test seam: false sends a non-NAX GEMM to the scalar SIMD body.
var gemm_simdmat_force: ?bool = null;
var gemm_simdmat_engaged: bool = false;
var prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var finish_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_kernel: KernelSlots = no_kernels;
var gemv_engaged: bool = false;

/// How the next dispatch decodes: the pack's codebook and codeword window.
/// A weight kernel is built per (codebook, window), so the slots below are
/// indexed by both; the model's own pair is asserted at each dispatch entry,
/// since several packs can be resident.
var active_decode: exl3.Decode = .mul1;
pub fn setDecodeParams(dec: exl3.Decode) void {
    active_decode = dec;
}
/// Test seam: a hash-mask or PAIR1 codebook's weight kernels decode the scale into every weight and store
/// their sums as they are (the converter's form), built into slot row `N_CB + cbIndex(cb)`.
pub var unfolded_for_test: bool = false;
const N_CB = exl3.Codebook.count;
const N_WIN = exl3.Window.count;
const N_TABLES = codebook_tables.tables.len;
const TABLED_ROW0 = 2 * N_CB;
/// A SPEC1 kernel carries its table's rows, so each (table, folded or unfolded) pair has its own row after the codebooks'.
pub const SLOT_ROWS = TABLED_ROW0 + 2 * N_TABLES;
const KernelSlots = [SLOT_ROWS][N_WIN]?mlx.mlx_fast_metal_kernel;
const no_kernels: KernelSlots = @splat(@splat(null));
fn cbIndex(cb: exl3.Codebook) usize {
    return @backingInt(cb);
}
fn tableOf(dec: exl3.Decode) usize {
    return dec.table.?;
}
/// The slot row of codebook `cb` under `active_decode`, `unfolded` for the test seam's scale-in-weights variant.
pub fn slotRow(cb: exl3.Codebook, unfolded: bool) usize {
    if (cb.tabledHashMask()) return TABLED_ROW0 + 2 * tableOf(active_decode) + @intFromBool(unfolded);
    const row: usize = if (unfolded) N_CB else 0;
    return row + cbIndex(cb);
}
fn winSuffix(comptime win: exl3.Window) [:0]const u8 {
    return comptime if (win == .w16) "" else std.fmt.comptimePrint("_w{d}", .{win.bits()});
}
fn cbSuffix(comptime cb: exl3.Codebook) [:0]const u8 {
    return switch (cb) {
        .mul1 => "",
        .mcg => "_mcg",
        .spec1 => "_spec1",
        .std1 => "_std1",
        .pair1 => "_pair1",
    };
}

/// The power of two `exl3_decode2` leaves out of a codebook's weights: a hash-mask codebook's
/// scale, restored by `exl3_out` once per stored sum instead of once per decoded pair.
fn foldScale(comptime cb: exl3.Codebook) u32 {
    return if (cb.tabledHashMask()) exl3.hash_mask_scale else if (cb == .pair1) PAIR1_SCALE else 1;
}

/// Every PAIR1 row's scale; `pair1Helpers` refuses a row that differs.
const PAIR1_SCALE: u32 = 4096;

/// The one decode every weight kernel calls, two codewords to two halves. A
/// pack whose search hashed a narrower window masks the sliding window down to
/// it first; w16 masks nothing and emits the source it always did.
/// `exl3_pairh` is the weight itself, an MMA operand. `exl3_decode1`/`exl3_decode2` are the weight
/// over `foldScale`, and a kernel that reads them passes every f32 sum it stores through
/// `exl3_out`: a power of two, so its sums and stored bytes are the unfolded ones.
pub fn codebookHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return decodeHelpers(cb, win, foldScale(cb));
}

const HELPERS_TAIL: [:0]const u8 =
    @embedFile("kernels/helpers_tail.metal");

fn decodeHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    if (cb == .pair1) return comptime pair1Helpers(win, fold) ++ HELPERS_TAIL;
    if (cb.tabledHashMask()) return comptime tabledHelpers(win, fold) ++ HELPERS_TAIL;
    const body: [:0]const u8 = comptime switch (cb) {
        .mul1 =>
        \\  const uint2 mixed = cw * uint2(0x83DCD12Du);
        \\  const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
        \\  const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
        \\  const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
        \\  return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
        ,
        .mcg => @embedFile("kernels/decode_helpers_mcg.metal"),
        else => unreachable,
    };
    return comptime helpersWithBody(body, win);
}

fn helpersWithBody(comptime body: [:0]const u8, comptime win: exl3.Window) [:0]const u8 {
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    return comptime "static inline half2 exl3_pairh(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n" ++
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
        \\static inline float exl3_out(float v) { return v; }
        \\static inline float exl3_decode1(uint cw) { return exl3_decode2(uint2(cw, 0u)).x; }
        \\// A generic reader's weight `slot` of a tile's unpacked codewords.
        \\static inline float exl3_slot(thread const ushort *cw, uint slot) { return exl3_decode1(uint(cw[slot])); }
        \\
    ++ HELPERS_TAIL;
}

/// What the row-driven codebooks (SPEC1, PAIR1) build over their `exl3_unscaled_n<N>`: the weight, the decode that
/// leaves `scale` out or keeps it, `slot` (a generic reader's weight), and the macros that hand every kernel's rate
/// `N` to them, so each row constant is a compile-time immediate of its own kernel instance.
fn rowedHelpers(comptime fold: u32, comptime scale: u32, comptime slot: [:0]const u8) [:0]const u8 {
    const out: [:0]const u8 = comptime if (fold == 1)
        "template<uint N> static inline float2 exl3_decode2_n(uint2 cw) { return float2(exl3_pairh_n<N>(cw)); }\n" ++
            "static inline float exl3_out(float v) { return v; }\n"
    else if (fold == scale)
        "template<uint N> static inline float2 exl3_decode2_n(uint2 cw) { return float2(exl3_unscaled_n<N>(cw)); }\n" ++
            std.fmt.comptimePrint("static inline float exl3_out(float v) {{ return v * {d}.0f; }}\n", .{fold})
    else
        @compileError("a fold is the codebook's scale or none");
    return comptime std.fmt.comptimePrint("template<uint N> static inline half2 exl3_pairh_n(uint2 cw) {{ return exl3_unscaled_n<N>(cw) * half2({d}.0h); }}\n", .{scale}) ++
        out ++ slot ++
        "#define exl3_pairh(cw) exl3_pairh_n<N>((cw))\n" ++
        "#define exl3_decode2(cw) exl3_decode2_n<N>((cw))\n" ++
        "#define exl3_slot(cw, slot) exl3_slot_n<N>((cw), (slot))\n";
}

fn windowMask(comptime win: exl3.Window) [:0]const u8 {
    return if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
}

/// PAIR1's pair decode: weights 2j and 2j+1 both come from the window of weight 2j+1 (`cw.y`; `cw.x` is
/// never read). A bank's rate is the kernel's `N`, which the macros of `rowedHelpers` look the constants up by in
/// `exl3_pair_row_at`; the row table is generated from `exl3.pair1Pair`, the one definition the host decodes with.
fn pair1Helpers(comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(2_000_000);
        var rows: [:0]const u8 = "";
        var n: u32 = exl3.Rate.min_n;
        while (n <= exl3.Rate.max_n) : (n += 2) {
            const p = exl3.pair1Pair(.{ .n = n }) orelse continue;
            if (p.scale != PAIR1_SCALE) @compileError("a PAIR1 row scales by 4096");
            rows = rows ++ std.fmt.comptimePrint("    case {d}u: return {{0x{X}u, {d}u, 0x{X}u, 0x{X}u}};\n", .{ n, p.mult, p.rot, p.mask0, p.mask1 });
        }
        return "#define EXL3_LANE_FIRST 1u\nstruct exl3_pair_row { uint mult; uint rot; uint mask0; uint mask1; };\n" ++
            "static inline constexpr exl3_pair_row exl3_pair_row_at(uint n) {\n  switch (n) {\n" ++ rows ++
            "  }\n  return {0u, 0u, 0u, 0u};\n}\n" ++
            "template<uint N> static inline half2 exl3_unscaled_n(uint2 cw) {\n" ++
            "  constexpr exl3_pair_row r = exl3_pair_row_at(N);\n" ++
            "  static_assert(r.mult != 0u, \"PAIR1 has no pair row at this rate\");\n" ++ windowMask(win) ++
            "  const uint p = cw.y * r.mult;\n" ++
            "  const half4 h = as_type<half4>(uint2(p & r.mask0, rotate(p, r.rot) & r.mask1));\n" ++
            "  return half2(h.x + h.y, h.z + h.w);\n}\n" ++
            rowedHelpers(fold, PAIR1_SCALE, "template<uint N> static inline float exl3_slot_n(thread const ushort *cw, uint slot) {\n" ++
                "  const float2 w = exl3_decode2_n<N>(uint2(0u, uint(cw[slot | 1u])));\n" ++
                "  return (slot & 1u) != 0u ? w.y : w.x;\n}\n");
    }
}

/// SPEC1's hash-mask decode with the (multiplier, mask) of the bank's rate: `exl3_hash_row_at(N)` is the table's row
/// lookup, which `withTableRows` puts ahead of this text because the rows are the pack's data, not the kernel's.
fn tabledHelpers(comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    return comptime "template<uint N> static inline half2 exl3_unscaled_n(uint2 cw) {\n" ++
        "  constexpr exl3_hash_row r = exl3_hash_row_at(N);\n" ++
        "  static_assert(r.mult != 0u, \"the codebook table has no row at this rate\");\n" ++ windowMask(win) ++
        "  const half4 h = as_type<half4>((cw * uint2(r.mult)) & uint2(r.mask));\n" ++
        "  return half2(h.x + h.y, h.z + h.w);\n}\n" ++
        rowedHelpers(fold, exl3.hash_mask_scale, "template<uint N> static inline float exl3_slot_n(thread const ushort *cw, uint slot) { return exl3_decode2_n<N>(uint2(uint(cw[slot]), 0u)).x; }\n");
}

/// The row lookup `tabledHelpers` reads, generated from SPEC1 table `which`.
fn tableRowsText(alloc: std.mem.Allocator, which: u8) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const table = &codebook_tables.tables[which];
    try out.print(alloc, "struct exl3_hash_row {{ uint mult; uint mask; }};\nstatic inline constexpr exl3_hash_row exl3_hash_row_at(uint n) {{\n  switch (n) {{\n", .{});
    for (table.rows, 0..) |r, i| try out.print(alloc, "    case {d}u: return {{0x{X}u, 0x{X}u}};\n", .{ table.rateOf(i).n, r[0], r[1] });
    try out.print(alloc, "  }}\n  return {{0u, 0u}};\n}}\n", .{});
    return out.toOwnedSliceSentinel(alloc, 0);
}

/// `header` as a kernel of `cb` under `active_decode` needs it: SPEC1's gets its table's rows ahead of the helpers.
/// Allocated for the life of the process, so call it only where a kernel is built.
pub fn withTableRows(comptime cb: exl3.Codebook, header: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return header;
    const alloc = std.heap.page_allocator;
    const rows = try tableRowsText(alloc, active_decode.table.?);
    defer alloc.free(rows);
    return std.mem.concatWithSentinel(alloc, u8, &.{ rows, header }, 0);
}

/// `getNamedKernel` for a kernel of codebook `cb` under `active_decode`: a SPEC1 kernel carries its table's rows and
/// spells the table in its name, since MLX keys a library by name.
fn tableKernel(slot: *?mlx.mlx_fast_metal_kernel, comptime cb: exl3.Codebook, comptime name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    return getNamedKernel(slot, (try tableKernelName(cb, name)).ptr, ins, outs, source, try withTableRows(cb, header));
}

/// MLX keys a kernel library by name, so a SPEC1 kernel's name spells its table.
fn tableKernelName(comptime cb: exl3.Codebook, comptime name: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return name;
    return std.fmt.allocPrintSentinel(std.heap.page_allocator, "{s}_t{s}", .{ name, codebook_tables.tables[tableOf(active_decode)].digest }, 0);
}

fn naxHeader(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return comptime GEMM_NAX_INCLUDES ++ codebookHelpers(cb, win) ++ GEMM_NAX_FRAGS;
}

/// A weight kernel under the active decode parameters: its own slot, name and
/// header.
fn codebookKernel(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return codebookKernelWith(slots, base, ins, outs, source, "");
}

/// `codebookKernel` with helpers of its own appended to the codebook header.
fn codebookKernelWith(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime frags: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| {
                if (@import("builtin").is_test and foldScale(cb) != 1 and unfolded_for_test) return tableKernel(
                    &slots[slotRow(cb, true)][win.index()],
                    cb,
                    comptime base ++ cbSuffix(cb) ++ "_unfolded" ++ winSuffix(win),
                    ins,
                    outs,
                    source,
                    comptime decodeHelpers(cb, win, 1) ++ frags,
                );
                return tableKernel(
                    &slots[slotRow(cb, false)][win.index()],
                    cb,
                    comptime base ++ cbSuffix(cb) ++ winSuffix(win),
                    ins,
                    outs,
                    source,
                    comptime codebookHelpers(cb, win) ++ frags,
                );
            },
        },
    }
}

const GemvKey = struct { in_dim: c_int, out_dim: c_int, n: u32 };

pub fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        pub fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.cfgs, 0..) |c, i| {
                if (c == null) {
                    victim = i;
                    break;
                }
                if (self.used[i] < oldest) {
                    oldest = self.used[i];
                    victim = i;
                }
            }
            if (self.cfgs[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
            self.cfgs[victim] = cfg;
            self.keys[victim] = key;
            self.tick += 1;
            self.used[victim] = self.tick;
        }
    };
}

const IndexedKey = struct { in_dim: c_int, out_dim: c_int, topk: c_int, n: u32 };
const UnaryKey = struct { dim: c_int, topk: c_int };
const GemmSortedKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, win: c_int, n: u32 };

const GEMM_WINDOW_ROWS: c_int = 32;

/// Maximum run supported by all sorted-GEMM bodies; larger runs leave tail rows unwritten.
const GEMM_WINDOW_MAX_ROWS: c_int = 32;

/// Test seams.
var gemm_win_override: ?c_int = null;
var gemm_align_override: ?bool = null;

/// null = not a window these kernels run, so the default stands.
pub fn gemmWindowRows() c_int {
    return gemm_win_override orelse GEMM_WINDOW_ROWS;
}

pub fn gemmWindowAligned() bool {
    return gemm_align_override orelse true;
}
var indexed_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var prepare_cfgs: CfgCache(UnaryKey, 8) = .{};
var finish_cfgs: CfgCache(UnaryKey, 8) = .{};
var gemm_sorted_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_simdmat_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_kernel: KernelSlots = no_kernels;
var gemm_nax_failed: bool = false;
var gemm_nax_cached: ?bool = null;
const PairPrepKey = struct { in_dim: c_int, nslots: c_int, topk: c_int };
const PairGemvKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, topk: c_int, n: u32, layout: GemvLayout, group: c_int };
const DownFusedKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, n: u32, layout: GemvLayout, group: c_int = 0 };
const MidKey = struct { dim: c_int, nslots: c_int, limit: c_int = 0 };
const ReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int };
const DecodeReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype };
var pair_gemv_cfgs: CfgCache(PairGemvKey, 8) = .{};
var mid_cfgs: CfgCache(MidKey, 8) = .{};
var reduce_cfgs: CfgCache(DecodeReduceKey, 8) = .{};
var down_fused_cfgs: CfgCache(DownFusedKey, 8) = .{};
var token_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_pair_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_reduce_cfgs: CfgCache(ReduceKey, 8) = .{};
const ScatterKey = struct { dim: c_int, nslots: c_int };
var token_scatter_cfgs: CfgCache(ScatterKey, 8) = .{};
var token_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_pair_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_pair_unsorted_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_scatter_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fused_dispatches: u32 = 0;
var apply_host_ns: u64 = 0;
var apply_host_n: u32 = 0;
var apply_host_layers: u32 = 0;
var apply_host_dumps: u32 = 0;
var apply_ubench_env: ?bool = null;

fn applyUbenchOn() bool {
    if (apply_ubench_env) |v| return v;
    const v = diagEnvValueOn(host_env.get(.SUSHI_DECODE_TICK_UBENCH));
    apply_ubench_env = v;
    return v;
}

fn gpuArch(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}


/// Test seam: forces the SIMD fallback where the NAX arm would run.
var force_family_fallback: bool = false;
fn forceFamilyFallback() bool {
    return force_family_fallback;
}

fn gemmNaxOn() bool {
    return !gemm_nax_failed and gemmNaxAvailable();
}

// Capability is independent of the dispatch failure latch: tests must still
// probe the real kernel after an earlier dispatch fell back to SIMD.
fn gemmNaxAvailable() bool {
    if (forceFamilyFallback()) return false;
    if (gemm_nax_cached) |v| return v;
    var buf: [128]u8 = undefined;
    const arch = gpuArch(&buf) orelse {
        gemm_nax_cached = false;
        return false;
    };
    var i: usize = 0;
    while (i + 2 < arch.len) : (i += 1) {
        const a = arch[i] | 32;
        const b = arch[i + 1] | 32;
        if (a == 'g' and b == '1' and arch[i + 2] >= '7' and arch[i + 2] <= '9') {
            gemm_nax_cached = true;
            return true;
        }
    }
    gemm_nax_cached = false;
    return false;
}

fn getGemmNaxKernel() !mlx.mlx_fast_metal_kernel {
    @setEvalBranchQuota(20_000);
    const ref = nax_reference_override;
    const slots = if (ref) &gemm_nax_ref_kernel else &gemm_nax_kernel;
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| {
                const slot = &slots[slotRow(cb, false)][win.index()];
                if (slot.*) |k| return k;
                const header = try withTableRows(cb, naxHeader(cb, win));
                const kernel = (if (ref)
                    buildNaxGemmKernel(GEMM_NAX_REFERENCE_SOURCE, header, (try tableKernelName(cb, comptime "sushi_exl3_k4_gemm_nax_ref" ++ cbSuffix(cb) ++ winSuffix(win))).ptr)
                else
                    buildNaxGemmKernel(GEMM_NAX_SOURCE, header, (try tableKernelName(cb, comptime "sushi_exl3_k4_gemm_nax" ++ cbSuffix(cb) ++ winSuffix(win))).ptr)) orelse {
                    gemm_nax_failed = true;
                    log.info("[exl3-gemm] NAX arm declined (kernel probe failed); the sorted GEMM serves prefill\n", .{});
                    return error.MetalKernelCompileFailed;
                };
                slot.* = kernel;
                return kernel;
            },
        },
    }
}

/// Metal JIT-compiles a kernel at its first EVAL, not at apply, so a source the
/// toolchain rejects is proven here on a one-tile problem before any prefill
/// depends on it. null = the arm is unusable; the latch it raised is dropped.
fn buildNaxGemmKernel(source: [:0]const u8, header: [:0]const u8, name: [*:0]const u8) ?mlx.mlx_fast_metal_kernel {
    const input_names = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (kernel.ctx == null) return null;
    if (probeNaxGemm(kernel)) return kernel;
    _ = mlx.mlx_fast_metal_kernel_free(kernel);
    return null;
}

fn probeNaxGemm(kernel: mlx.mlx_fast_metal_kernel) bool {
    const s = mlx.gpuStream();
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const in_dim: c_int = 16;
    const out_dim: c_int = 128;
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, out_dim }, 2, .float16) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIN", gemmWindowRows()) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", 64) != 0) return false;
    const xh = std.mem.zeroes([16]u16);
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, in_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x);
    const th = std.mem.zeroes([8 * 64]u16);
    const trellis = mlx.mlx_array_new_data(&th, &[_]c_int{ 1, 1, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trellis);
    const zero = [_]u32{0};
    const one = [_]u32{1};
    const eids = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eids);
    const starts = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts);
    const nlive = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlive);
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, starts, nlive };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, cfg, s) != 0) return false;
    if (mlx.mlx_vector_array_size(outputs) != 1) return false;
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    if (mlx.mlx_vector_array_get(&out, outputs, 0) != 0) return false;
    if (mlx.mlx_array_eval(out) != 0) return false;
    return !mlx.errorPending();
}

fn getGemmSortedKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&gemm_sorted_kernel, "sushi_exl3_k4_gemm_sorted", &ins, &outs, GEMM_SORTED_SOURCE);
}

fn getGemmSimdmatKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernelWith(&gemm_simdmat_kernel, "sushi_exl3_k4_gemm_simdmat", &ins, &outs, GEMM_SIMDMAT_SOURCE, GEMM_SIMDMAT_FRAGS);
}

fn sortedGemmCfg(cache: *CfgCache(GemmSortedKey, 8), key: GemmSortedKey) !mlx.mlx_fast_metal_kernel_config {
    if (cache.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ key.rows, key.out_dim }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", key.in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", key.out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", key.win));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(key.n)));
    cache.put(key, c);
    return c;
}

fn applySortedGemm(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, x: mlx.mlx_array, trellis: mlx.mlx_array, eids: mlx.mlx_array, tab: WindowTable) !mlx.mlx_array {
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub const WindowTable = struct { starts: mlx.mlx_array, nlives: mlx.mlx_array, nwin: c_int };

var gemm_sel_logged: bool = false;

fn logGemmSelector(win: c_int, aligned: bool, nwin: c_int, n: c_int) void {
    if (n < 2048) return;
    if (gemm_sel_logged) return;
    gemm_sel_logged = true;
    log.info("[exl3-gemm] win={d} aligned={d} nwin={d} mixed={d} n={d}\n", .{
        win,
        @intFromBool(aligned),
        nwin,
        @as(u32, if (aligned) 0 else 1),
        n,
    });
}

fn buildWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    return buildWindowTableHost(s, eids, n, win);
}

fn gemmWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int, aligned: bool) !WindowTable {
    return if (aligned) buildWindowTable(s, eids, n, win) else buildStrideTable(n, win);
}

fn buildWindowTableHost(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    const ids = try std.heap.page_allocator.alloc(u32, @intCast(n));
    defer std.heap.page_allocator.free(ids);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, eids, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    switch (mlx.mlx_array_dtype(contig)) {
        .uint32 => {
            const p = mlx.mlx_array_data_uint32(contig) orelse return error.F16Unreadable;
            @memcpy(ids, p[0..ids.len]);
        },
        .int32 => {
            const p = mlx.mlx_array_data_int32(contig) orelse return error.F16Unreadable;
            for (ids, 0..) |*d, i| d.* = @intCast(p[i]);
        },
        else => return error.BadExl3Shape,
    }
    const runs = try buildRuns(std.heap.page_allocator, ids);
    defer std.heap.page_allocator.free(runs.start);
    defer std.heap.page_allocator.free(runs.len);
    defer std.heap.page_allocator.free(runs.eid);
    const w: u32 = @intCast(win);
    var nwin_u: u32 = 0;
    var r: u32 = 0;
    while (r < runs.n) : (r += 1) {
        nwin_u += (runs.len[r] + w - 1) / w;
    }
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var k: u32 = 0;
    r = 0;
    while (r < runs.n) : (r += 1) {
        var off: u32 = 0;
        while (off < runs.len[r]) {
            const live = @min(w, runs.len[r] - off);
            sh[k] = runs.start[r] + off;
            lh[k] = live;
            k += 1;
            off += live;
        }
    }
    const starts = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    errdefer _ = mlx.mlx_array_free(starts);
    const nlives = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_host nwin={d} n={d} win={d} eval=eids\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

fn buildStrideTable(n: c_int, win: c_int) !WindowTable {
    const w: u32 = @intCast(win);
    const nn: u32 = @intCast(n);
    const nwin_u = (nn + w - 1) / w;
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var i: u32 = 0;
    while (i < nwin_u) : (i += 1) {
        const st = i * w;
        sh[i] = st;
        lh[i] = @min(w, nn - st);
    }
    const starts = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    errdefer _ = mlx.mlx_array_free(starts);
    const nlives = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_stride nwin={d} n={d} win={d}\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

/// The window table depends only on the sorted slots, so a layer's three
/// projections share one: each build is a host eval that drains the GPU.
pub fn innerGemmSortedTable(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
    aligned: bool,
    tab: WindowTable,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 2 or tsh.len != 4) return error.BadExl3Shape;
    if (win <= 0 or win > GEMM_WINDOW_MAX_ROWS) return error.BadExl3Shape;
    const n = xsh[0];
    const in_dim = xsh[1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    if (tab.nwin <= 0) return error.BadExl3Shape;
    logGemmSelector(win, aligned, tab.nwin, n);
    const key = GemmSortedKey{ .in_dim = in_dim, .out_dim = out_dim, .rows = n, .win = win, .n = rate.n };
    if (gemmNaxOn() and @rem(out_dim, 128) == 0) {
        // The fallback below answers a NAX build or dispatch that failed, so the
        // failure's latch is ours to drop: left standing it becomes the next
        // decode tick's `MlxFailure`.
        const had_error = mlx.errorPending();
        if (getGemmNaxKernel()) |nk| {
            const ncfg = try sortedGemmCfg(&gemm_nax_cfgs, key);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ncfg, out_dim, tab.nwin, 1));
            const ninputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
            const ninputs_vec = mlx.mlx_vector_array_new_data(&ninputs, ninputs.len);
            defer _ = mlx.mlx_vector_array_free(ninputs_vec);
            var noutputs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(noutputs);
            if (mlx.mlx_fast_metal_kernel_apply(&noutputs, nk, ninputs_vec, ncfg, s) == 0 and mlx.mlx_vector_array_size(noutputs) == 1) {
                var nout = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(nout);
                try mlx.check(mlx.mlx_vector_array_get(&nout, noutputs, 0));
                logFunnel(funnelReads(rate.n), rate.n, .nax, mlx.mlx_array_dtype(x));
                return nout;
            }
            gemm_nax_failed = true;
        } else |_| {
            gemm_nax_failed = true;
        }
        mlx.dropLatchedErrorUnless(had_error);
    }
    // x feeds `simdgroup_matrix<half>` as stored, so only f16 activations take the matrix body.
    if ((gemm_simdmat_force orelse true) and @rem(out_dim, 128) == 0 and mlx.mlx_array_dtype(x) == .float16) {
        const cfg = try sortedGemmCfg(&gemm_simdmat_cfgs, key);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, tab.nwin, 1));
        const out = try applySortedGemm(s, try getGemmSimdmatKernel(), cfg, x, trellis, eids, tab);
        if (!gemm_simdmat_engaged) {
            gemm_simdmat_engaged = true;
            log.info("[exl3-gemm] simdgroup-matrix body engaged n={d}\n", .{rate.n});
        }
        logFunnel(funnelReads(rate.n), rate.n, .simdmat, mlx.mlx_array_dtype(x));
        return out;
    }
    const cfg = try sortedGemmCfg(&gemm_sorted_cfgs, key);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_tiles * 128, tab.nwin, 1));
    return applySortedGemm(s, try getGemmSortedKernel(), cfg, x, trellis, eids, tab);
}

var inner_gemv_cfgs: CfgCache(GemvKey, 8) = .{};

const INDEXED_COOP_SOURCE: [:0]const u8 =
    @embedFile("kernels/indexed_coop.metal");

var indexed_coop_kernel: KernelSlots = no_kernels;

fn getIndexedCoopKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "slots" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&indexed_coop_kernel, "sushi_exl3_k4_mul1_gemv_indexed", &ins, &outs, INDEXED_COOP_SOURCE);
}

const INDEXED_PAIR_COOP_SOURCE: [:0]const u8 = blk: {
    const split = "uint split = uint(threadgroup_position_in_grid.z);";
    const at = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, split).?;
    const pointers =
        @embedFile("kernels/indexed_pair_coop_pointers.metal");
    // Grid Z selects a projection, never a K split. Keep the original loop
    // and four-simdgroup reduction exactly as the standalone kernel.
    break :blk pointers ++ INDEXED_COOP_SOURCE[0..at] ++ "uint split = 0u;" ++ INDEXED_COOP_SOURCE[at + split.len ..];
};
var indexed_pair_coop_kernel: KernelSlots = no_kernels;
var indexed_pair_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var indexed_pair_coop_engaged: bool = false;

fn indexedPairCoopF16(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array) !?[2]mlx.mlx_array {
    const xs = mlx.getShape(xg);
    const ts = mlx.getShape(tg);
    const ss = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or xs.len != 2 or ts.len != 4 or ss.len != 1 or
        !std.mem.eql(c_int, xs, mlx.getShape(xu)) or !std.mem.eql(c_int, ts, mlx.getShape(tu)) or
        xs[0] <= 0 or xs[1] <= 0 or ts[0] <= 0 or ts[1] <= 0 or ts[2] <= 0 or
        ts[1] > std.math.maxInt(c_int) / 16 or ts[2] > std.math.maxInt(c_int) / 128 or
        xs[0] != ss[0] or ts[1] * 16 != xs[1] or
        mlx.mlx_array_dtype(xg) != .float16 or mlx.mlx_array_dtype(xu) != .float16 or
        mlx.mlx_array_dtype(tg) != .uint16 or mlx.mlx_array_dtype(tu) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    const rate = try packedRate(ts[3]);
    const output = ts[2] * 16;
    const key = IndexedKey{ .in_dim = xs[1], .out_dim = output, .topk = ss[0], .n = rate.n };
    const cfg = indexed_pair_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const shape = [_]c_int{ ss[0], output };
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, ts[2] * 128, ss[0], 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", xs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", output));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        indexed_pair_coop_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try codebookKernel(&indexed_pair_coop_kernel, "sushi_exl3_indexed_pair_coop", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, INDEXED_PAIR_COOP_SOURCE);
    const outputs = try applyOuts(s, kernel, &.{ xg, xu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (result) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (&result, 0..) |*value, i| try mlx.check(mlx.mlx_vector_array_get(value, outputs, i));
    if (!indexed_pair_coop_engaged) {
        indexed_pair_coop_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-decode] paired cooperative gate/up engaged inner=f16 k_splits=1\n", .{});
    }
    return result;
}

pub fn indexedGemvCoopF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk, .n = rate.n };
    const cfg = indexed_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        indexed_coop_cfgs.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getIndexedCoopKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub const DOWN_FUSED_SOURCE: [:0]const u8 =
    @embedFile("kernels/down_fused.metal");

pub fn downGemvFusedMid(
    s: mlx.mlx_stream,
    ig: mlx.mlx_array,
    iu: mlx.mlx_array,
    trellis: mlx.mlx_array,
    svhg: mlx.mlx_array,
    svhu: mlx.mlx_array,
    suhd: mlx.mlx_array,
    slots: mlx.mlx_array,
    in_dim: c_int,
    out_dim: c_int,
    nslots: c_int,
) !mlx.mlx_array {
    const tsh = mlx.getShape(trellis);
    const rate = try packedRate(tsh[tsh.len - 1]);
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(out_dim));
    const layout = gemvLayout(out_tiles);
    const key = DownFusedKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .n = rate.n, .layout = layout };
    const cfg = down_fused_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        try addGemvLayout(c, layout);
        down_fused_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try codebookKernel(&down_fused_kernel, "sushi_exl3_k4_down_fused", &ins, &outs, DOWN_FUSED_SOURCE);
    const ov = try applyOuts(s, kernel, &.{ ig, iu, trellis, svhg, svhu, suhd, slots }, cfg, 1);
    logFunnel(layout.funnel, rate.n, .fused_mid_down, mlx.mlx_array_dtype(ig));
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

/// The suh scale + H128 Hadamard that feeds this GEMV is applied here, over the
/// k range this threadgroup owns, rather than in a dispatch of its own: the tile
/// loop then reads its operands from threadgroup memory instead of device.
pub const PAIR_GEMV_SOURCE: [:0]const u8 =
    @embedFile("kernels/pair_gemv.metal");

const MID_SOURCE: [:0]const u8 =
    @embedFile("kernels/mid.metal");

pub const REDUCE_SOURCE: [:0]const u8 =
    @embedFile("kernels/reduce.metal");

pub const REDUCE_SHARED_SOURCE: [:0]const u8 = blk: {
    const stores =
        \\  y[yb + lane] = T(a0);
        \\  y[yb + lane + 32u] = T(a1);
        \\  y[yb + lane + 64u] = T(a2);
        \\  y[yb + lane + 96u] = T(a3);
    ;
    const at = std.mem.indexOf(u8, REDUCE_SOURCE, stores).?;
    // Preserve the separate routed store's rounding before the shared addition.
    break :blk REDUCE_SOURCE[0..at] ++
        \\  y[yb + lane] = T(float(T(a0)) + float(shared[yb + lane]));
        \\  y[yb + lane + 32u] = T(float(T(a1)) + float(shared[yb + lane + 32u]));
        \\  y[yb + lane + 64u] = T(float(T(a2)) + float(shared[yb + lane + 64u]));
        \\  y[yb + lane + 96u] = T(float(T(a3)) + float(shared[yb + lane + 96u]));
    ++ REDUCE_SOURCE[at + stores.len ..];
};

/// `shared` is the shared expert's raw down projection and `gate[0]` its gate logit, both T = bf16. The gate and the
/// product round to T at each step MLX's bf16 Sigmoid and Multiply do, so the sum equals the three-op chain's.
pub const REDUCE_GATED_SHARED_SOURCE: [:0]const u8 = blk: {
    const stores =
        \\  y[yb + lane] = T(a0);
        \\  y[yb + lane + 32u] = T(a1);
        \\  y[yb + lane + 64u] = T(a2);
        \\  y[yb + lane + 96u] = T(a3);
    ;
    const at = std.mem.indexOf(u8, REDUCE_SOURCE, stores).?;
    break :blk REDUCE_SOURCE[0..at] ++
        @embedFile("kernels/reduce_gated_shared.metal") ++ REDUCE_SOURCE[at + stores.len ..];
};

const FunnelArm = enum { pair, fused_mid_down, prepared_down, nax, simdmat };
var funnel_engaged: [5]bool = @splat(false);

fn logFunnel(on: bool, n: u32, arm: FunnelArm, dtype: mlx.mlx_dtype) void {
    if (!on or funnel_engaged[@backingInt(arm)]) return;
    funnel_engaged[@backingInt(arm)] = true;
    log.info("[exl3] n{d} funnel engaged arm={s} codebook={s} dtype={s} window={d}\n", .{ n, @tagName(arm), @tagName(active_decode.codebook), @tagName(dtype), active_decode.window.bits() });
}

var pair_gemv_kernel: KernelSlots = no_kernels;
var pair_gemv_grouped_kernel: KernelSlots = no_kernels;
var mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_shared_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_shared_lead_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_gated_shared_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_fused_kernel: KernelSlots = no_kernels;

pub fn getNamedKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header.ptr, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = kernel;
    return kernel;
}

pub fn applyOuts(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, n_out: usize) !mlx.mlx_vector_array {
    fused_dispatches += 1;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(outputs_vec);
    const host_on = applyUbenchOn();
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = if (host_on) io_util.Stopwatch.init(io) else undefined;
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (host_on) {
        apply_host_ns += sw.read();
        apply_host_n += 1;
    }
    if (mlx.mlx_vector_array_size(outputs_vec) != n_out) return error.MetalKernelBadOutputCount;
    return outputs_vec;
}

/// `group_ask` > 0 asks for the expert-grouped kernel; only the lane funnel has one.
fn pairGemvConfig(in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, rate: exl3.Rate, layout: GemvLayout, group: c_int) !mlx.mlx_fast_metal_kernel_config {
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(in_dim));
    const key = PairGemvKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .topk = topk, .n = rate.n, .layout = layout, .group = group };
    if (pair_gemv_cfgs.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sh = [_]c_int{ nslots * nsplit, out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, nsplit));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
    try addGemvLayout(c, layout);
    try addGroup(c, group, nslots);
    pair_gemv_cfgs.put(key, c);
    return c;
}

pub fn pairGemv(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, group_ask: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const tsh = mlx.getShape(tg);
    const ush = mlx.getShape(tu);
    const rate = try packedRate(tsh[tsh.len - 1]);
    if (ush[ush.len - 1] != tsh[tsh.len - 1]) return error.BadExl3Shape;
    const layout = gemvLayout(@divExact(out_dim, 16));
    const group: c_int = if (layout.funnel) group_ask else 0;
    const cfg = try pairGemvConfig(in_dim, out_dim, nslots, topk, rate, layout, group);
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (group > 0)
        try codebookKernel(&pair_gemv_grouped_kernel, "sushi_exl3_pair_gemv_grouped", &ins, &outs, PAIR_GEMV_GROUPED_SOURCE)
    else
        try codebookKernel(&pair_gemv_kernel, "sushi_exl3_pair_gemv", &ins, &outs, PAIR_GEMV_SOURCE);
    logGroupedEngaged(group);
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, tg, tu, slots }, cfg, 2);
    logFunnel(layout.funnel, rate.n, .pair, mlx.mlx_array_dtype(x));
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

pub fn midSwigluPrep(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    return midSwigluPrepWithLimit(s, ig, iu, svhg, svhu, suhd, slots, dim, nslots, 0);
}

fn midSwigluPrepWithLimit(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, limit: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots, .limit = limit };
    const cfg = mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "CLAMP_LIMIT", limit));
        mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&mid_kernel, "sushi_exl3_mid_swiglu", &ins, &outs, MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

/// One simdgroup per (row, k) slot, so the threadgroup cannot hold more slots
/// than Metal allows threads.
pub const REDUCE_MAX_TOPK: c_int = 32;

pub fn downFinishReduce(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return downFinishReduceWithShared(s, inner, svh, slots, scores, out_dim, rows, topk, out_dtype, null, null, false);
}

pub fn downFinishReduceWithShared(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array, gate: ?mlx.mlx_array, shared_leads: bool) !mlx.mlx_array {
    if (topk < 1 or topk > REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = DecodeReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk, .dtype = out_dtype };
    const cfg = reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = if (rows == 1) [_]c_int{out_dim} ++ [_]c_int{0} else [_]c_int{ rows, out_dim };
        const ndim: usize = if (rows == 1) 1 else 2;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, ndim, out_dtype));
        const blocks = @divExact(out_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * blocks, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "inner", "svh", "slots", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const ov = if (shared) |value| blk: {
        if (gate) |logit| {
            const gated_ins = ins ++ [_][*:0]const u8{ "shared", "gate" };
            const kernel = try getNamedKernel(&reduce_gated_shared_kernel, "sushi_exl3_down_reduce_gated_shared", &gated_ins, &outs, REDUCE_GATED_SHARED_SOURCE, "");
            break :blk try applyOuts(s, kernel, &.{ inner, svh, slots, scores, value, logit }, cfg, 1);
        }
        if (shared_leads) {
            // MLX encodes inputs in order: a serial decode layer then runs its shared chain before the routed one.
            const lead_ins = [_][*:0]const u8{"shared"} ++ ins;
            const kernel = try getNamedKernel(&reduce_shared_lead_kernel, "sushi_exl3_down_reduce_shared_lead", &lead_ins, &outs, REDUCE_SHARED_SOURCE, "");
            break :blk try applyOuts(s, kernel, &.{ value, inner, svh, slots, scores }, cfg, 1);
        }
        const shared_ins = ins ++ [_][*:0]const u8{"shared"};
        const kernel = try getNamedKernel(&reduce_shared_kernel, "sushi_exl3_down_reduce_shared", &shared_ins, &outs, REDUCE_SHARED_SOURCE, "");
        break :blk try applyOuts(s, kernel, &.{ inner, svh, slots, scores, value }, cfg, 1);
    } else blk: {
        const kernel = try getNamedKernel(&reduce_kernel, "sushi_exl3_down_reduce", &ins, &outs, REDUCE_SOURCE, "");
        break :blk try applyOuts(s, kernel, &.{ inner, svh, slots, scores }, cfg, 1);
    };
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn pairPrepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, order: ?mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots", "order" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (order != null)
        try getNamedKernel(&token_pair_prepare_kernel, "sushi_exl3_token_pair_prepare", &ins, &outs, TOKEN_PAIR_PREPARE_SOURCE, "")
    else
        try getNamedKernel(&token_pair_unsorted_kernel, "sushi_exl3_token_pair_unsorted", ins[0..4], &outs, metalReplaceFound(TOKEN_PAIR_PREPARE_SOURCE, "const uint orig = uint(order[slot]);", "const uint orig = slot;"), "");
    const ov = if (order) |sorted|
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots, sorted }, cfg, 2)
    else
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn scatterSorted(s: mlx.mlx_stream, x: mlx.mlx_array, order: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    const key = ScatterKey{ .dim = dim, .nslots = nslots };
    const cfg = token_scatter_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, dim, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "DIM", dim));
        token_scatter_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_scatter_kernel, "sushi_exl3_token_scatter", &ins, &outs, TOKEN_SCATTER_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn moeSwigluFusedWithShared(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    out_dtype: mlx.mlx_dtype,
    shared: ?mlx.mlx_array,
    shared_gate: ?mlx.mlx_array,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const ssh = mlx.getShape(slots);
    const tsh = mlx.getShape(gate_t);
    const nslots = ssh[0];
    const hidden: c_int = if (xsh.len == 1) xsh[0] else xsh[xsh.len - 1];
    const rows: c_int = if (xsh.len == 1) 1 else xsh[0];
    const topk = @divExact(nslots, rows);
    const inter = tsh[2] * 16;
    const prepared = preparedMidOn(hidden, inter, tsh[0], topk, rows, out_dtype) and mlx.mlx_array_dtype(x) == .bfloat16;
    // Only MiMo's verify rows share enough experts for grouping to pay.
    const group: c_int = if (prepared and rows >= 2 and nslots <= DECODE_GROUP_MAX_SLOTS and !grouped_off_for_test) DECODE_GROUP_MEMBERS else 0;
    // A one-row pair keeps preparing its own K span: the extra prepare dispatch costs it more than it saves.
    const inners = if (prepared and rows >= 2) blk: {
        const prep = try pairPrepare(s, x, gate_suh, up_suh, slots, hidden, nslots, topk);
        defer _ = mlx.mlx_array_free(prep);
        break :blk try pairGemvPrepared(s, prep, gate_t, up_t, slots, hidden, inter, nslots, topk, group);
    } else try pairGemv(s, x, gate_suh, up_suh, gate_t, up_t, slots, hidden, inter, nslots, topk, group);
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try ubenchEval(inners[0], "pair_gemv");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(inners[1]));
    const maxabs = swigluMaxabsOn() and swiglu_maxabs_dumped < 96;
    if (maxabs) {
        try dumpAbsMax(s, inners[0], "ig");
        try dumpAbsMax(s, inners[1], "iu");
    }
    const down_inner = if (prepared) blk: {
        const y = try downGemvPreparedMid(s, inners[0], inners[1], down_t, gate_svh, up_svh, down_suh, slots, inter, hidden, nslots, group);
        if (!prepared_mid_engaged) {
            prepared_mid_engaged = true;
            log.info("[exl3-decode] prepared mid engaged dtype=bfloat16 middle=f16 after down-input Hadamard\n", .{});
        }
        break :blk y;
    } else try downGemvFusedMid(s, inners[0], inners[1], down_t, gate_svh, up_svh, down_suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(down_inner);
    try ubenchEval(down_inner, "down_gemv");
    // The SwiGLU product and the down inner plane are the f16 stores that can
    // saturate: a non-finite here is the mid overflow, not a decode fault.
    if (maxabs) {
        try dumpAbsMax(s, down_inner, "down_inner");
        swiglu_maxabs_dumped += 1;
    }
    const out = try downFinishReduceWithShared(s, down_inner, down_svh, slots, scores, hidden, rows, topk, out_dtype, shared, shared_gate, false);
    errdefer _ = mlx.mlx_array_free(out);
    try ubenchEval(out, "reduce");
    if (applyUbenchOn()) {
        apply_host_layers += 1;
        if (apply_host_layers == 48) {
            if (apply_host_dumps < 8) {
                const ms = @as(f64, @floatFromInt(apply_host_ns)) / 1e6;
                const n: f64 = @floatFromInt(@max(apply_host_n, 1));
                log.info("[exl3-apply] host {d:.3} ms n={d} us/apply={d:.1}\n", .{
                    ms,
                    apply_host_n,
                    (ms * 1e3) / n,
                });
                apply_host_dumps += 1;
            }
            apply_host_ns = 0;
            apply_host_n = 0;
            apply_host_layers = 0;
            if (apply_host_dumps >= 8) apply_ubench_env = false;
        }
    }
    return out;
}

pub fn moePrefill(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const rows = xsh[0];
    const hidden = xsh[1];
    const nslots = rows * topk;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    try ubenchEval(sorted_slots, "sort");
    const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, sorted_slots, order_i, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    try ubenchEval(prep[0], "token_prepare");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(prep[1]));
    const win = gemmWindowRows();
    const aligned = gemmWindowAligned();
    const gt = mlx.getShape(gate_t);
    const optimized = mimoPrefillOn(hidden, gt[2] * 16, gt[0], topk) and aligned;
    const metadata: ?MimoWindowTable = if (optimized) try buildMimoWindowTable(s, sorted_slots, order_i, nslots, win, gt[0]) else null;
    defer if (metadata) |m| {
        _ = mlx.mlx_array_free(m.inverse);
    };
    const tab = if (metadata) |m| m.table else try gemmWindowTable(s, sorted_slots, nslots, win, aligned);
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    const g_inner = try innerGemmSortedTable(s, prep[0], gate_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(g_inner);
    try ubenchEval(g_inner, "gemm_gate");
    const u_inner = try innerGemmSortedTable(s, prep[1], up_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(u_inner);
    try ubenchEval(u_inner, "gemm_up");
    const down_x = try midSwigluPrep(s, g_inner, u_inner, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(g_inner)[1], nslots);
    defer _ = mlx.mlx_array_free(down_x);
    try ubenchEval(down_x, "mid");
    const d_inner = try innerGemmSortedTable(s, down_x, down_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(d_inner);
    try ubenchEval(d_inner, "gemm_down");
    if (metadata) |m| {
        const out = try finishMimoSorted(s, d_inner, m.inverse, down_svh, slots, scores, hidden, rows, topk, mlx.mlx_array_dtype(x));
        try ubenchEval(out, "token_reduce");
        return out;
    }
    const d_unsorted = try scatterSorted(s, d_inner, order_i, hidden, nslots);
    defer _ = mlx.mlx_array_free(d_unsorted);
    const out = try downFinishReduce(s, d_unsorted, down_svh, slots, scores, hidden, rows, topk, mlx.mlx_array_dtype(x));
    try ubenchEval(out, "token_reduce");
    return out;
}

fn clampedWindowTable(s: mlx.mlx_stream, ids: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int, experts: c_int, aligned: bool) !MimoWindowTable {
    if (aligned and experts > 0 and experts <= MIMO_WINDOW_MAX_EXPERTS)
        return buildMimoWindowTable(s, ids, order, n, win, experts);
    return .{ .table = try gemmWindowTable(s, ids, n, win, aligned), .inverse = .{ .ctx = null } };
}

/// Decode rows add `shared` in the finish reduce after the routed sum's own rounding; wider rows refuse it.
pub fn moeSwigluClampedWithShared(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
    limit: c_int,
    out_dtype: mlx.mlx_dtype,
    shared: ?mlx.mlx_array,
) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const rows = sh[0];
    const hidden = sh[1];
    const nslots = rows * topk;
    if (rows <= DECODE_ROWS_MAX) {
        const pair = (try laneClampedPair(s, x, gate_t, up_t, gate_suh, up_suh, slots, hidden, rows, topk)) orelse blk: {
            const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, slots, null, hidden, nslots, topk);
            const prep_g = prep[0];
            const prep_u = prep[1];
            defer _ = mlx.mlx_array_free(prep_g);
            defer _ = mlx.mlx_array_free(prep_u);
            break :blk (try indexedPairCoopF16(s, prep_g, prep_u, gate_t, up_t, slots)) orelse fallback: {
                const gate = try indexedGemvCoopF16(s, prep_g, gate_t, slots);
                errdefer _ = mlx.mlx_array_free(gate);
                break :fallback [2]mlx.mlx_array{ gate, try indexedGemvCoopF16(s, prep_u, up_t, slots) };
            };
        };
        const gate = pair[0];
        const up = pair[1];
        defer for (pair) |value| {
            _ = mlx.mlx_array_free(value);
        };
        const down = blk: {
            const gs = mlx.getShape(gate_t);
            const us = mlx.getShape(up_t);
            const ds = mlx.getShape(down_t);
            if (out_dtype == .bfloat16 and active_decode.codebook.servesGlmFastPaths() and
                gs[3] == us[3] and gs[3] == ds[3] and laneServesRate(gs[3]))
            {
                const prepared = try downLanePrepare(s, gate, up, gate_svh, up_svh, down_suh, slots, mlx.getShape(gate)[1], nslots, limit);
                defer _ = mlx.mlx_array_free(prepared);
                lane_decode_dispatches += 1;
                break :blk try downLaneCoopWith(s, prepared, down_t, slots, rows == 1 and oneRowCoopServes(ds[1], ds[2]));
            }
            const prepared = try midSwigluPrepWithLimit(s, gate, up, gate_svh, up_svh, down_suh, slots, mlx.getShape(gate)[1], nslots, limit);
            defer _ = mlx.mlx_array_free(prepared);
            break :blk try indexedGemvCoopF16(s, prepared, down_t, slots);
        };
        defer _ = mlx.mlx_array_free(down);
        if (shared) |value| return downFinishReduceWithShared(s, down, down_svh, slots, scores, hidden, rows, topk, out_dtype, value, null, true);
        return downFinishReduce(s, down, down_svh, slots, scores, hidden, rows, topk, out_dtype);
    }
    if (shared != null) return error.Exl3SharedNeedsDecodeRows;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, sorted_slots, order_i, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const win = gemmWindowRows();
    const aligned = gemmWindowAligned();
    const metadata = try clampedWindowTable(s, sorted_slots, order_i, nslots, win, mlx.getShape(gate_t)[0], aligned);
    defer if (metadata.inverse.ctx != null) {
        _ = mlx.mlx_array_free(metadata.inverse);
    };
    const tab = metadata.table;
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    const gate = try innerGemmSortedTable(s, prep[0], gate_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(gate);
    const up = try innerGemmSortedTable(s, prep[1], up_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(up);
    const prepared = try midSwigluPrepWithLimit(s, gate, up, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(gate)[1], nslots, limit);
    defer _ = mlx.mlx_array_free(prepared);
    const down = try innerGemmSortedTable(s, prepared, down_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(down);
    if (metadata.inverse.ctx != null)
        return finishMimoSorted(s, down, metadata.inverse, down_svh, slots, scores, hidden, rows, topk, out_dtype);
    const original_order = try scatterSorted(s, down, order_i, hidden, nslots);
    defer _ = mlx.mlx_array_free(original_order);
    return downFinishReduce(s, original_order, down_svh, slots, scores, hidden, rows, topk, out_dtype);
}

/// What a GEMM/GEMV arm's output may differ from the exact dot product of the
/// same decoded weights.
///
/// A trellis dot product cancels: the result lands about four orders below
/// sum|w_i x_i|, so a bar written against the RESULT measures noise and passes
/// or fails on the seed. Every bar here is relative to the SUMMANDS, which is
/// the scale the arithmetic runs at.
const Exl3GemmParity = struct {
    /// f16 unit roundoff. An arm rounds its result to f16 exactly once and
    /// |result| <= sum|w_i x_i|, so that store costs at most U_F16 of the sum.
    const U_F16: f64 = 0x1p-11;
    /// f32 unit roundoff. Every arm accumulates its products in f32 planes, so
    /// a `depth`-long accumulation costs at most depth * U_F32 of the sum.
    const U_F32: f64 = 0x1p-24;
    /// How much of the composite's RMS error the arm may carry.
    const RMS_FACTOR: f64 = 3.0;

};

/// A defect injected between the arm and its reference, so the bar is shown to
/// FAIL on a wrong arm and not only to pass on a right one. `.none` is the
/// real test; the others make the arm's decode disagree with the reference's
/// exactly as a miscoded kernel would.
const Exl3Mutation = enum {
    none,
    /// The arm reads the codeword window the pack names, the reference reads
    /// another — the same defect as a kernel built for the wrong mask.
    window,
    /// One bit of one trellis halfword, which moves the weights whose sliding
    /// window covers it.
    codeword,
};

/// Why a GEMM parity check failed, or null when it passed. Pure, so the
/// decision is unit-testable without a GPU.
const Exl3GemmParityFail = enum { nonfinite, gross_element, systematic };

const Exl3GemmParityStats = struct {
    n: usize = 0,
    finite: bool = true,
    kern_max: f64 = 0,
    comp_max: f64 = 0,
    rms_kern: f64 = 0,
    rms_comp: f64 = 0,
    ceiling: f64 = 0,
};

/// Unequal widths catch a swapped stride a square shape hides; `aligned = false` packs
/// several runs into one window.
const ParityShape = struct { in_dim: usize = 128, out_dim: usize = 128, aligned: bool = true };

/// Seeds are a sample, not a choice: the bar holds for any trellis, so every
/// case sweeps a fixed run of them rather than one that happened to pass.
const PARITY_SEEDS: usize = 8;

/// One MiMo-V2.6-Flash MoE layer's shape and routing on the GPU arms: hidden
/// != inter, every expert holding its OWN bank, and a routing that leaves some
/// experts unrouted while giving others more rows than one GEMM window holds.
const MimoMoeCase = struct {
    e: usize,
    hidden: usize,
    inter: usize,
    topk: usize,
    rows: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
    seed: u64,
    /// |suh_h|, |svh_i|, |suh_i|, |svh_h|. The default is the synthetic 0.9 the
    /// parity cases use; the served pack's own magnitudes are `MIMO_BANKS`.
    banks: [4]f32 = @splat(0.9),
    x_scale: f32 = 0.05,
};

/// The served w12 pack's own scale magnitudes: |suh| is ~0.01 and |svh| ~1,
/// so the residual's size reaches the arm's f16 planes through the GEMMs.
const MIMO_BANKS = [4]f32{ 0.0126, 1.03, 0.0083, 1.009 };

const MimoMoeFixture = struct {
    arrays: [10]mlx.mlx_array,
    gate_t: []u16,
    up_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    slots: []u32,
    scores: []f32,

};

/// The SwiGLU every EXL3 arm approximates, carried end to end in f32: the
/// stored weights are f16 but nothing between them is. `moeSwigluHost` mirrors
/// the kernels' own f16 stores, so it cannot say whether one of them saturated.
const Exl3F32Peaks = struct { g: f64 = 0, u: f64 = 0, mid: f64 = 0, down_inner: f64 = 0 };

/// Forces the non-NAX arms for one test and puts back the prior setting.
const FallbackEnv = struct {
    prior: bool = false,
};

const ReaderArm = enum { lane, nax, simdmat };

const MimoWindowTable = struct { table: WindowTable, inverse: mlx.mlx_array };

const MIMO_WINDOW_SOURCE: [:0]const u8 =
    @embedFile("kernels/mimo_window.metal");

const MimoWindowKey = struct { rows: c_int, win: c_int, experts: c_int };
var mimo_window_cfgs: CfgCache(MimoWindowKey, 8) = .{};
var mimo_window_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mimo_window_engaged: bool = false;

/// One thread per expert, in one threadgroup.
const MIMO_WINDOW_MAX_EXPERTS: c_int = 512;

fn buildMimoWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int, experts: c_int) !MimoWindowTable {
    if (experts < 1 or experts > MIMO_WINDOW_MAX_EXPERTS or win < 1 or n < 1) return error.BadExl3Shape;
    const capacity = @divTrunc(n + win - 1, win) + experts;
    const key = MimoWindowKey{ .rows = n, .win = win, .experts = experts };
    const cfg = mimo_window_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{n}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", win));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "EXPERTS", experts));
        mimo_window_cfgs.put(key, c);
        break :blk c;
    };
    const nn: u32 = @intCast(n);
    const count = mlx.mlx_array_new_data(&nn, &.{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(count);
    const kernel = try getNamedKernel(&mimo_window_kernel, "sushi_exl3_mimo_windows", &.{ "eids", "order", "count" }, &.{ "starts", "nlives", "inverse" }, MIMO_WINDOW_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ eids, order, count }, cfg, 3);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    var inverse = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inverse);
    try mlx.check(mlx.mlx_vector_array_get(&starts, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&nlives, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&inverse, outputs, 2));
    if (!mimo_window_engaged) {
        mimo_window_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] GPU window metadata engaged experts={d} win={d}\n", .{ experts, win });
    }
    return .{ .table = .{ .starts = starts, .nlives = nlives, .nwin = capacity }, .inverse = inverse };
}

const MIMO_REDUCE_SOURCE: [:0]const u8 = blk: {
    const old = "const size_t xb = (size_t)slot * (size_t)(ODIM) + base;";
    const at = std.mem.indexOf(u8, REDUCE_SOURCE, old).?;
    break :blk REDUCE_SOURCE[0..at] ++ "const size_t xb = (size_t)inverse[slot] * (size_t)(ODIM) + base;" ++ REDUCE_SOURCE[at + old.len ..];
};
var mimo_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mimo_reduce_cfgs: CfgCache(DecodeReduceKey, 8) = .{};
var mimo_reduce_engaged: bool = false;

pub fn finishMimoSorted(s: mlx.mlx_stream, inner: mlx.mlx_array, inverse: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    if (topk < 1 or topk > REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = DecodeReduceKey{ .out_dim = dim, .rows = rows, .topk = topk, .dtype = dtype };
    const cfg = mimo_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, dim }, 2, dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * @divExact(dim, 128), rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        mimo_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&mimo_reduce_kernel, "sushi_exl3_mimo_sorted_reduce", &.{ "inner", "inverse", "svh", "slots", "sc" }, &.{"y"}, MIMO_REDUCE_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ inner, inverse, svh, slots, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    if (!mimo_reduce_engaged) {
        mimo_reduce_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] sorted finish/reduce engaged dtype={s}\n", .{@tagName(dtype)});
    }
    return y;
}

var mimo_prefill_force: ?bool = null;

/// The GPU window metadata and the inverse-indexed finish, at the two served geometries (MiMo, Flash-Next).
fn mimoPrefillOn(hidden: c_int, inter: c_int, experts: c_int, topk: c_int) bool {
    if (mimo_prefill_force) |v| return v;
    if (experts < 1) return false;
    return (hidden == 4096 and inter == 2048 and experts <= 256 and topk == 8) or
        (hidden == 2560 and inter == 640 and experts <= MIMO_WINDOW_MAX_EXPERTS and topk == 10);
}

var prepared_mid_force: ?bool = null;

/// `metalReplaceAll` for a rewrite a kernel depends on: a pattern the source no longer holds is a
/// compile error, never a silently unchanged kernel.
fn metalReplaceFound(comptime source: []const u8, comptime old: []const u8, comptime replacement: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    if (comptime std.mem.indexOf(u8, source, old) == null) @compileError("kernel rewrite pattern not found: " ++ old);
    return metalReplaceAll(source, old, replacement);
}

fn metalReplaceAll(comptime source: []const u8, comptime old: []const u8, comptime replacement: []const u8) [:0]const u8 {
    @setEvalBranchQuota(200000);
    const at = comptime std.mem.indexOf(u8, source, old) orelse return source ++ "";
    return comptime source[0..at] ++ replacement ++ metalReplaceAll(source[at + old.len ..], old, replacement);
}

const DECODE_MID_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(200000);
    const start = std.mem.indexOf(u8, DOWN_FUSED_SOURCE, "const float sc =").?;
    const end = start + std.mem.indexOf(u8, DOWN_FUSED_SOURCE[start..], "threadgroup_barrier(mem_flags::mem_threadgroup);").?;
    const head =
        @embedFile("kernels/decode_mid_head.metal");
    break :blk head ++ "\n" ++ metalReplaceAll(DOWN_FUSED_SOURCE[start..end], "prepared[base", "prepared[(size_t)slot * uint(IDIM) + base");
};

const DOWN_PREPARED_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(200000);
    const start = std.mem.indexOf(u8, DOWN_FUSED_SOURCE, "const float sc =").?;
    const barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    const end = start + std.mem.indexOf(u8, DOWN_FUSED_SOURCE[start..], barrier).? + barrier.len;
    const head = metalReplaceAll(DOWN_FUSED_SOURCE[0..start], "threadgroup half prepared[uint(IDIM)];", "");
    const tail = metalReplaceAll(DOWN_FUSED_SOURCE[end..], "const threadgroup half *pp", "const device half *pp");
    break :blk head ++ "const device half *prepared = middle + (size_t)slot * uint(IDIM);\n" ++ tail;
};

/// A multi-row block's slots routed to one expert share a threadgroup: the slot whose rank
/// among that expert's slots is a multiple of `GROUP` leads itself and the next `GROUP - 1`
/// (the other threadgroups return). Each weight is read and decoded once, then fed to every
/// member's accumulators in the single-slot kernel's order, so every slot's bytes equal the
/// lane-funnel kernel's.
const GROUP_MEMBERS_SOURCE =
    @embedFile("kernels/group_members.metal");

/// One funnel iteration (two k-tiles, `OTPT` output tiles) over `mc` members; `member_ptr`
/// points at member j's inputs for this iteration's first k-tile.
fn groupFunnelStep(comptime mc: []const u8, comptime member_ptr: []const u8) []const u8 {
    return "exl3_lane_bits<N> merged[2][OTPT];\n" ++
        @embedFile("kernels/group_funnel_step.metal") ++ "  float ins[" ++ mc ++ "][4];\n  for (uint j = 0u; j < " ++ mc ++ "; j++) {\n    const auto q = " ++ member_ptr ++ " + u * SGS * TILE;\n" ++
        @embedFile("kernels/group_funnel_step_2.metal") ++ "      for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        \\        acc[j][o][p * 2u] = fma(ins[j][(p * 2u) & 3u], w.x, acc[j][o][p * 2u]);
        \\        acc[j][o][p * 2u + 1u] = fma(ins[j][(p * 2u + 1u) & 3u], w.y, acc[j][o][p * 2u + 1u]);
        \\      }
        \\    }
        \\  }
        \\}
        \\wp += 2u * SGS * OT * PACKED_W;
        \\
    ;
}

/// The single-slot kernels' tile reduction for each of `mc` members, the first `m` stored.
fn groupEpilogue(comptime mc: []const u8, comptime y_at: []const u8, comptime cast: []const u8) []const u8 {
    return "for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        @embedFile("kernels/group_epilogue.metal") ++ "  " ++ y_at ++ " = " ++ cast ++ "(exl3_out(sum));\n" ++
        @embedFile("kernels/group_epilogue_2.metal");
}

/// A lone member takes a one-member copy of the loop: the `GROUP` copy would spend
/// `GROUP - 1` idle FMAs per weight.
fn groupBySize(comptime body: fn (comptime []const u8) []const u8) []const u8 {
    return "if (m == 1u) {\n" ++ body("1u") ++ "} else {\n" ++ body("uint(GROUP)") ++ "}\n";
}

fn pairGroupedBody(comptime mc: []const u8) []const u8 {
    return @embedFile("kernels/pair_grouped_body.metal") ++ "  float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        @embedFile("kernels/pair_grouped_body_2.metal") ++ "    for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        @embedFile("kernels/pair_grouped_body_3.metal") ++ groupFunnelStep(mc, "(pp + j * CHUNK)") ++
        \\      pp += 2u * SGS * TILE;
        \\    }
        \\  }
        \\
    ++ groupEpilogue(mc, "y[(size_t)(members[j] * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "") ++
        \\}
        \\
    ;
}

/// `PAIR_GEMV_SOURCE`'s lane-funnel arm, grouped. Members' prepared inputs are staged one
/// projection and one k chunk at a time, so threadgroup memory stays the single-slot 8 KiB.
const PAIR_GEMV_GROUPED_SOURCE: [:0]const u8 =
    @embedFile("kernels/pair_gemv_grouped.metal") ++ GROUP_MEMBERS_SOURCE ++ groupBySize(pairGroupedBody);

fn downGroupedBody(comptime mc: []const u8) []const u8 {
    return "float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        \\const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
        \\const device uint *wp = trellis_e + ((size_t)sg * (size_t)OT + ot) * PACKED_W;
        \\
    ++ "const device half *pm[" ++ mc ++ "];\nfor (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] = middle + (size_t)members[j] * uint(IDIM) + sg * TILE;\n" ++
        @embedFile("kernels/down_grouped_body.metal") ++ groupFunnelStep(mc, "pm[j]") ++
        "  for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] += 2u * SGS * TILE;\n}\n" ++
        groupEpilogue(mc, "y[(size_t)members[j] * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "half");
}

/// `DOWN_PREPARED_SOURCE`'s lane-funnel arm, grouped; each member reads its own prepared middle.
const DOWN_PREPARED_GROUPED_SOURCE: [:0]const u8 =
    @embedFile("kernels/down_prepared_grouped.metal") ++ GROUP_MEMBERS_SOURCE ++ groupBySize(downGroupedBody);

/// Where lane `lane`'s value of a 128-block lands in lane order: each 16-row tile keeps rows
/// (2q, 2q+1, 2q+8, 2q+9) at 4q..4q+3, the four a GEMV lane with `lane & 3 == q` reads.
const LANE_ORDER_SLOT = "((lane & 16u) + ((lane & 7u) >> 1u) * 4u + ((lane & 15u) >> 3u) * 2u + (lane & 1u))";

/// A decode GEMV source reading its prepared input as one device half4 per lane and k-tile from a
/// lane-ordered buffer instead of four halves; the values and their order are unchanged.
fn laneQuadReads(comptime src: [:0]const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var out: [:0]const u8 = src;
    for ([_][2][]const u8{
        .{ "pp[u * SGS * TILE + row", "pp + u * SGS * TILE" },
        .{ "prepared[xb + (tk - tk0) * TILE + row", "prepared + xb + (tk - tk0) * TILE" },
        .{ "prepared[tk * TILE + row", "prepared + tk * TILE" },
    }) |r| {
        if (std.mem.indexOf(u8, out, "const float in0 = float(" ++ r[0] ++ "0]);") == null) continue;
        out = metalReplaceAll(out, "const float in0 = float(" ++ r[0] ++ "0]);", "const float4 in4 = float4(*((const device half4 *)(" ++ r[1] ++ " + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
        out = metalReplaceFound(out, "const float in1 = float(" ++ r[0] ++ "1]);", "const float in1 = in4.y;");
        out = metalReplaceFound(out, "const float in2 = float(" ++ r[0] ++ "2]);", "const float in2 = in4.z;");
        out = metalReplaceFound(out, "const float in3 = float(" ++ r[0] ++ "3]);", "const float in3 = in4.w;");
    }
    if (std.mem.indexOf(u8, out, "ins[j][0] = float(q[prow]);") != null) {
        out = metalReplaceAll(out, "ins[j][0] = float(q[prow]);", "const float4 in4 = float4(*((const device half4 *)(q + (lane & 3u) * 4u)));\n    ins[j][0] = in4.x;");
        out = metalReplaceFound(out, "ins[j][1] = float(q[prow + 1u]);", "ins[j][1] = in4.y;");
        out = metalReplaceFound(out, "ins[j][2] = float(q[prow + 8u]);", "ins[j][2] = in4.z;");
        out = metalReplaceFound(out, "ins[j][3] = float(q[prow + 9u]);", "ins[j][3] = in4.w;");
    }
    if (std.mem.indexOf(u8, out, "in4") == null) @compileError("no prepared read was rewritten");
    if (std.mem.indexOf(u8, out, "row0]") != null or std.mem.indexOf(u8, out, "q[prow]") != null) @compileError("a prepared read kept the natural layout");
    return out;
}

/// A slot's Hadamard-prepared input for both projections, `[slot][gate|up][IDIM]` f16: the values
/// `PAIR_GEMV_SOURCE` otherwise re-derives in every threadgroup of the slot, in lane order.
const PAIR_PREP_SOURCE: [:0]const u8 = metalReplaceFound(PAIR_PREP_BODY, "LANE_SLOT", LANE_ORDER_SLOT);
const PAIR_PREP_BODY: [:0]const u8 =
    @embedFile("kernels/pair_prep_body.metal");

/// `PAIR_GEMV_SOURCE` reading its slot's prepared input from device memory instead of
/// preparing its K span in threadgroup memory; every arm reads the same values in the same order.
const PAIR_GEMV_PREPARED_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(400000);
    const start = std.mem.indexOf(u8, PAIR_GEMV_SOURCE, "threadgroup half prepared[2u * KSPAN];").?;
    const barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const end = start + std.mem.indexOf(u8, PAIR_GEMV_SOURCE[start..], barrier).? + barrier.len;
    const body = PAIR_GEMV_SOURCE[0..start] ++ "const device half *prepared = prep + (size_t)slot * 2u * uint(IDIM) + tk0 * TILE;\n" ++ PAIR_GEMV_SOURCE[end..];
    break :blk laneQuadReads(metalReplaceFound(metalReplaceFound(body, "const uint xb = proj * KSPAN;", "const uint xb = proj * uint(IDIM);"), "const threadgroup half *pp", "const device half *pp"));
};

fn pairPreparedGroupedBody(comptime mc: []const u8) []const u8 {
    return @embedFile("kernels/pair_prepared_grouped_body.metal") ++ "  float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
        \\  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
        \\
    ++ "  const device half *pm[" ++ mc ++ "];\n  for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] = prep + ((size_t)members[j] * 2u + proj) * uint(IDIM) + (tk0 + sg) * TILE;\n" ++
        @embedFile("kernels/pair_prepared_grouped_body_2.metal") ++ groupFunnelStep(mc, "pm[j]") ++
        "    for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] += 2u * SGS * TILE;\n  }\n" ++
        groupEpilogue(mc, "y[(size_t)(members[j] * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "") ++
        \\}
        \\
    ;
}

/// `PAIR_GEMV_GROUPED_SOURCE` on the prepared input: members read their own prepared rows.
const PAIR_GEMV_PREPARED_GROUPED_SOURCE: [:0]const u8 =
    @embedFile("kernels/pair_gemv_prepared_grouped.metal") ++ laneQuadReads(GROUP_MEMBERS_SOURCE ++ groupBySize(pairPreparedGroupedBody));

const PairPrepKeyD = struct { in_dim: c_int, nslots: c_int, topk: c_int };
var pair_prep_cfgs: CfgCache(PairPrepKeyD, 8) = .{};
var pair_prep_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pair_gemv_prepared_kernel: KernelSlots = no_kernels;
var pair_gemv_prepared_grouped_kernel: KernelSlots = no_kernels;

/// Each slot's gate and up inputs, Hadamard-prepared once: `[nslots * 2, in_dim]` f16.
pub fn pairPrepare(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !mlx.mlx_array {
    const key = PairPrepKeyD{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots * 2, in_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(in_dim, 128) * 32, nslots, 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&pair_prep_kernel, "sushi_exl3_pair_prepare", &.{ "x", "suhg", "suhu", "slots" }, &.{"prep"}, PAIR_PREP_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var prep = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(prep);
    try mlx.check(mlx.mlx_vector_array_get(&prep, ov, 0));
    return prep;
}

/// `pairGemv` on `pairPrepare`'s output: the same planes, bit for bit.
pub fn pairGemvPrepared(s: mlx.mlx_stream, prep: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, group_ask: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const tsh = mlx.getShape(tg);
    const rate = try packedRate(tsh[tsh.len - 1]);
    if (mlx.getShape(tu)[mlx.getShape(tu).len - 1] != tsh[tsh.len - 1]) return error.BadExl3Shape;
    const layout = gemvLayout(@divExact(out_dim, 16));
    const group: c_int = if (layout.funnel) group_ask else 0;
    const cfg = try pairGemvConfig(in_dim, out_dim, nslots, topk, rate, layout, group);
    const ins = [_][*:0]const u8{ "prep", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (group > 0)
        try codebookKernel(&pair_gemv_prepared_grouped_kernel, "sushi_exl3_pair_gemv_prepared_grouped", &ins, &outs, PAIR_GEMV_PREPARED_GROUPED_SOURCE)
    else
        try codebookKernel(&pair_gemv_prepared_kernel, "sushi_exl3_pair_gemv_prepared", &ins, &outs, PAIR_GEMV_PREPARED_SOURCE);
    logGroupedEngaged(group);
    const ov = try applyOuts(s, kernel, &.{ prep, tg, tu, slots }, cfg, 2);
    logFunnel(layout.funnel, rate.n, .pair, .float16);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

const DecodeMidKey = struct { dim: c_int, nslots: c_int, nsplit: c_int };
var decode_mid_cfgs: CfgCache(DecodeMidKey, 8) = .{};
var decode_mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_prepared_cfgs: CfgCache(DownFusedKey, 8) = .{};
var down_prepared_kernel: KernelSlots = no_kernels;
var down_prepared_grouped_kernel: KernelSlots = no_kernels;
var prepared_mid_engaged: bool = false;
var grouped_engaged: bool = false;

fn prepareDecodeMid(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, nsplit: c_int) !mlx.mlx_array {
    const key = DecodeMidKey{ .dim = dim, .nslots = nslots, .nsplit = nsplit };
    const cfg = decode_mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots, dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(dim, 128) * 32, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        decode_mid_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&decode_mid_kernel, "sushi_exl3_decode_mid", &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots" }, &.{"prepared"}, DECODE_MID_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var middle = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(middle);
    try mlx.check(mlx.mlx_vector_array_get(&middle, outputs, 0));
    return middle;
}

fn downGemvPreparedMid(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, trellis: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, group_ask: c_int) !mlx.mlx_array {
    const nsplit: c_int = @intCast(pairSplitCountFor(out_dim));
    const middle = try prepareDecodeMid(s, ig, iu, svhg, svhu, suhd, slots, in_dim, nslots, nsplit);
    defer _ = mlx.mlx_array_free(middle);
    const tsh = mlx.getShape(trellis);
    const rate = try packedRate(tsh[tsh.len - 1]);
    const out_tiles = @divExact(out_dim, 16);
    const layout = gemvLayout(out_tiles);
    const group: c_int = if (layout.funnel) group_ask else 0;
    const key = DownFusedKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .n = rate.n, .layout = layout, .group = group };
    const cfg = down_prepared_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots, out_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        try addGemvLayout(c, layout);
        try addGroup(c, group, nslots);
        down_prepared_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = if (group > 0)
        try codebookKernel(&down_prepared_grouped_kernel, "sushi_exl3_down_prepared_grouped", &.{ "middle", "trellis", "slots" }, &.{"y"}, DOWN_PREPARED_GROUPED_SOURCE)
    else
        try codebookKernel(&down_prepared_kernel, "sushi_exl3_down_prepared", &.{ "middle", "trellis", "slots" }, &.{"y"}, DOWN_PREPARED_SOURCE);
    const outputs = try applyOuts(s, kernel, &.{ middle, trellis, slots }, cfg, 1);
    logFunnel(layout.funnel, rate.n, .prepared_down, mlx.mlx_array_dtype(middle));
    defer _ = mlx.mlx_vector_array_free(outputs);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    return y;
}

fn preparedMidOn(hidden: c_int, inter: c_int, experts: c_int, topk: c_int, rows: c_int, dtype: mlx.mlx_dtype) bool {
    if (prepared_mid_force) |v| return v;
    return hidden == 4096 and inter == 2048 and experts > 0 and experts <= 256 and topk == 8 and rows >= 1 and rows <= 8 and dtype == .bfloat16;
}

// Both helpers, every codeword: the MMA operand `exl3_pairh`, and a GEMV's `exl3_decode2` as its
// f32 sums carry it (the weight over the fold) and as `exl3_out` stores it, each against the host
// decode. Spec1's halves are often subnormal and its half sum rounds, alike on both.

const SPEC1_W15 = exl3.Decode.tabled(.spec1, .w15);
const STD1_W15 = exl3.Decode.tabled(.std1, .w15);

const PAIR1_W16: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };
const PAIR1_NS = [_]u32{ 24, 28, 32, 36, 40, 42, 48, 56, 64 };

const FoldArm = struct { prepared: ?bool = null, grouped_off: bool = false, funnel_off: bool = false, scalar_gemm: bool = false };

// A SPEC1 table whose pair differs at every rate, decoded by every reader at the bank's own rate. A reader that
// took one pair for every bank, or the legacy pair, or another rate's row, fails against the host decode.
const SPEC1_PER_RATE: exl3.Decode = .{ .codebook = .spec1, .window = .w12, .table = codebook_tables.indexOf("spec1-test-per-rate") };
const PER_RATE_NS = [_]u32{ 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64 };

// The rates beyond the K1.5 to K4 run, K1 to K1.375 and K4.125 to K8, with SPEC1's own table rows and the test
// table's: each reader, the prefill GEMM, the decode chain and the f32-truth bar, as at the rates above.
const SPEC1_WIDE_NS = [_]u32{ 16, 18, 20, 22, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128 };

/// A copy of `a` that waits on `dep`, so the timed GEMV steps run one after another.
const LANE_LINK_SOURCE: [:0]const u8 =
    @embedFile("kernels/lane_link.metal");
var lane_link_kernel: ?mlx.mlx_fast_metal_kernel = null;

// Staged oracle keeps the original projection and reduction ordering.

const CLAMPED_MIDDLE_DOWN_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(100000);
    const mid_begin = std.mem.indexOf(u8, MID_SOURCE, "const float sc =").?;
    const mid_end = std.mem.indexOf(u8, MID_SOURCE, "yd[xb + lane]").?;
    const ot = "uint ot = uint(threadgroup_position_in_grid.x);";
    const coop_begin = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, ot).? + ot.len;
    const xb = "const size_t xb = (size_t)slot * (size_t)(IDIM);";
    const coop_xb = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, xb).?;
    const prefix =
        @embedFile("kernels/clamped_middle_down_prefix.metal");
    const bridge =
        @embedFile("kernels/clamped_middle_down_bridge.metal");
    break :blk prefix ++ MID_SOURCE[mid_begin..mid_end] ++ bridge ++ INDEXED_COOP_SOURCE[coop_begin..coop_xb] ++
        "const size_t xb = 0u;" ++ INDEXED_COOP_SOURCE[coop_xb + xb.len ..] ++ "\n}\n";
};
const ClampedMiddleKey = struct { input: c_int, output: c_int, slots: c_int, n: u32, limit: c_int, tiles: c_int };
var clamped_middle_cfgs: CfgCache(ClampedMiddleKey, 8) = .{};
var clamped_middle_kernels: KernelSlots = no_kernels;

fn clampedMiddleDownCoop(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, down: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, limit: c_int, tiles: c_int) !?mlx.mlx_array {
    const x = mlx.getShape(ig);
    const w = mlx.getShape(down);
    const ids = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or x.len != 2 or w.len != 4 or ids.len != 1 or
        !std.mem.eql(c_int, x, mlx.getShape(iu)) or x[0] != ids[0] or x[0] < 1 or x[1] < 128 or x[1] > 8192 or @mod(x[1], 128) != 0 or
        w[0] < 1 or w[1] < 1 or w[1] > std.math.maxInt(c_int) / 16 or w[1] * 16 != x[1] or w[2] < 1 or w[2] > std.math.maxInt(c_int) / 128 or (tiles != 1 and tiles != 4 and tiles != 8) or @mod(w[2], tiles) != 0 or limit < 1 or limit > 128 or
        mlx.mlx_array_dtype(ig) != .float16 or mlx.mlx_array_dtype(iu) != .float16 or mlx.mlx_array_dtype(down) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    for ([_]mlx.mlx_array{ svhg, svhu, suhd }) |a|
        if (mlx.mlx_array_dtype(a) != .float16 or !std.mem.eql(c_int, &.{ w[0], x[1] }, mlx.getShape(a))) return null;
    const rate = try packedRate(w[3]);
    const key = ClampedMiddleKey{ .input = x[1], .output = w[2] * 16, .slots = x[0], .n = rate.n, .limit = limit, .tiles = tiles };
    const cfg = clamped_middle_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ x[0], key.output }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(w[2], tiles) * 128, x[0], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        inline for (.{ .{ "IDIM", key.input }, .{ "ODIM", key.output }, .{ "NHW", @as(c_int, @intCast(rate.n)) }, .{ "CLAMP_LIMIT", limit }, .{ "OTPT", tiles } }) |p|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, p[0], p[1]));
        clamped_middle_cfgs.put(key, c);
        break :blk c;
    };
    const k = try codebookKernel(&clamped_middle_kernels, "sushi_exl3_clamped_middle_down", &.{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" }, &.{"y"}, CLAMPED_MIDDLE_DOWN_SOURCE);
    const outputs = try applyOuts(s, k, &.{ ig, iu, down, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, outputs, 0));
    return result;
}

// Lane-ordered half4 gate/up pair; each row keeps the indexed pair's arithmetic.

const LANE_PREP_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(1000000);
    break :blk metalReplaceAll(TOKEN_PAIR_PREPARE_SOURCE, "[yb + lane", "[yb + " ++ LANE_ORDER_SLOT);
};
const LANE_PAIR_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(1000000);
    var body = INDEXED_PAIR_COOP_SOURCE;
    body = metalReplaceAll(body, "const float in0 = float(x[xb + tk * TILE + row0]);", "const float4 in4 = float4(*((const device half4 *)(x + xb + tk * TILE + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
    body = metalReplaceAll(body, "const float in1 = float(x[xb + tk * TILE + row1]);", "const float in1 = in4.y;");
    body = metalReplaceAll(body, "const float in2 = float(x[xb + tk * TILE + row2]);", "const float in2 = in4.z;");
    body = metalReplaceAll(body, "const float in3 = float(x[xb + tk * TILE + row3]);", "const float in3 = in4.w;");
    break :blk body;
};
var lane_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var lane_prep_sorted: ?mlx.mlx_fast_metal_kernel = null;
var lane_prep_unsorted: ?mlx.mlx_fast_metal_kernel = null;
var lane_pair_cfgs: CfgCache(IndexedKey, 8) = .{};
var lane_pair_kernel: KernelSlots = no_kernels;
pub fn lanePairPrepare(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, order: ?mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = lane_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        lane_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots", "order" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (order != null)
        try getNamedKernel(&lane_prep_sorted, "sushi_glm_lane_prepare", &ins, &outs, LANE_PREP_SOURCE, "")
    else
        try getNamedKernel(&lane_prep_unsorted, "sushi_glm_lane_unsorted", ins[0..4], &outs, metalReplaceFound(LANE_PREP_SOURCE, "const uint orig = uint(order[slot]);", "const uint orig = slot;"), "");
    const ov = if (order) |sorted|
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots, sorted }, cfg, 2)
    else
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}
pub fn lanePairCoop(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array) !?[2]mlx.mlx_array {
    return lanePairCoopWith(s, xg, xu, tg, tu, slots, false);
}

fn lanePairCoopWith(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, one_row: bool) !?[2]mlx.mlx_array {
    const xs = mlx.getShape(xg);
    const ts = mlx.getShape(tg);
    const ss = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or xs.len != 2 or ts.len != 4 or ss.len != 1 or
        !std.mem.eql(c_int, xs, mlx.getShape(xu)) or !std.mem.eql(c_int, ts, mlx.getShape(tu)) or
        xs[0] <= 0 or xs[1] <= 0 or ts[0] <= 0 or ts[1] <= 0 or ts[2] <= 0 or
        ts[1] > std.math.maxInt(c_int) / 16 or ts[2] > std.math.maxInt(c_int) / 128 or
        xs[0] != ss[0] or ts[1] * 16 != xs[1] or
        mlx.mlx_array_dtype(xg) != .float16 or mlx.mlx_array_dtype(xu) != .float16 or
        mlx.mlx_array_dtype(tg) != .uint16 or mlx.mlx_array_dtype(tu) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    if (one_row and !oneRowCoopServes(ts[1], ts[2])) return null;
    const rate = try packedRate(ts[3]);
    const output = ts[2] * 16;
    const key = IndexedKey{ .in_dim = xs[1], .out_dim = output, .topk = ss[0], .n = rate.n };
    const cache = if (one_row) &lane_pair_one_cfgs else &lane_pair_cfgs;
    const cfg = cache.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const shape = [_]c_int{ ss[0], output };
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, ts[2] * (if (one_row) @as(c_int, 64) else 128), ss[0], 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", xs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", output));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        cache.put(key, c);
        break :blk c;
    };
    const kernel = if (one_row)
        try codebookKernel(&lane_pair_one_kernel, "sushi_glm_lane_pair_one", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, ONE_ROW_PAIR_SOURCE)
    else
        try codebookKernel(&lane_pair_kernel, "sushi_glm_lane_pair", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, LANE_PAIR_SOURCE);
    if (one_row) countOneRow(rate.n);
    const outputs = try applyOuts(s, kernel, &.{ xg, xu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (result) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (&result, 0..) |*value, i| try mlx.check(mlx.mlx_vector_array_get(value, outputs, i));
    return result;
}

// Kernels shared with the DFlash group-of-two expert path in glm_group2.zig.
pub const Group2Support = struct {
    pub const Slots = KernelSlots;
    pub const empty = no_kernels;
    pub const cooperative_source = INDEXED_COOP_SOURCE;
    pub const prepare = pairPrepareFromTokens;
    pub const middle = midSwigluPrepWithLimit;
    pub const lane_middle = downLanePrepare;
    pub const lane_down = downLaneCoop;
    pub const fused_middle_down = clampedMiddleDownCoop;
};

/// NAX source and helpers shared by the GLM prefill grid permutation.
pub const PrefillGridSupport = struct {
    pub const source = GEMM_NAX_SOURCE;
    pub const prepare = pairPrepareFromTokens;
    pub const windows = clampedWindowTable;
    pub const middle = midSwigluPrepWithLimit;
    pub const finish = finishMimoSorted;
};

var lane_decode_dispatches: usize = 0;
/// The GLM lane kernels read through `exl3_lane`, whose words and shifts follow from n.
fn laneServesRate(n: c_int) bool {
    return n >= 0 and exl3.kFromPackedDim(@intCast(n)) != null;
}
fn laneClampedPair(s: mlx.mlx_stream, x: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, sg: mlx.mlx_array, su: mlx.mlx_array, slots: mlx.mlx_array, hidden: c_int, rows: c_int, topk: c_int) !?[2]mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !active_decode.codebook.servesGlmFastPaths() or mlx.mlx_array_dtype(x) != .bfloat16 or rows < 1 or rows > 16 or hidden < 128 or @mod(hidden, 128) != 0) return null;
    const gs = mlx.getShape(tg);
    if (gs.len != 4 or !std.mem.eql(c_int, gs, mlx.getShape(tu)) or !laneServesRate(gs[3])) return null;
    const prepared = try lanePairPrepare(s, x, sg, su, slots, null, hidden, rows * topk, topk);
    defer _ = mlx.mlx_array_free(prepared[0]);
    defer _ = mlx.mlx_array_free(prepared[1]);
    const pair = (if (rows == 1) try lanePairCoopWith(s, prepared[0], prepared[1], tg, tu, slots, true) else null) orelse try lanePairCoop(s, prepared[0], prepared[1], tg, tu, slots);
    if (pair != null) lane_decode_dispatches += 1;
    return pair;
}

// Lane-ordered half4 reads for the clamped middle and the down projection.
fn downLaneReads(comptime source: [:0]const u8, comptime space: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var body = source;
    body = metalReplaceAll(body, "const float in0 = float(x[xb + tk * TILE + row0]);", "const float4 in4 = float4(*((const " ++ space ++ " half4 *)(x + xb + tk * TILE + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
    body = metalReplaceAll(body, "const float in1 = float(x[xb + tk * TILE + row1]);", "const float in1 = in4.y;");
    body = metalReplaceAll(body, "const float in2 = float(x[xb + tk * TILE + row2]);", "const float in2 = in4.z;");
    body = metalReplaceAll(body, "const float in3 = float(x[xb + tk * TILE + row3]);", "const float in3 = in4.w;");
    return body;
}
const DOWN_LANE_MID_SOURCE = metalReplaceAll(MID_SOURCE, "yd[xb + lane", "yd[xb + " ++ LANE_ORDER_SLOT);
const DOWN_LANE_COOP_SOURCE = downLaneReads(INDEXED_COOP_SOURCE, "device");
var down_lane_mid_cfgs: CfgCache(MidKey, 8) = .{};
var down_lane_mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_lane_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var down_lane_coop_kernel: KernelSlots = no_kernels;
fn getDownLaneCoopKernel() !mlx.mlx_fast_metal_kernel {
    return codebookKernel(&down_lane_coop_kernel, "sushi_glm_down_lane_coop", &.{ "x", "trellis", "slots" }, &.{"y"}, DOWN_LANE_COOP_SOURCE);
}
fn downLanePrepare(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, limit: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots, .limit = limit };
    const cfg = down_lane_mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "CLAMP_LIMIT", limit));
        down_lane_mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&down_lane_mid_kernel, "sushi_glm_down_lane_prepare", &ins, &outs, DOWN_LANE_MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}
/// One-row decode GEMV body. A threadgroup carries two output tiles and a simdgroup two k-tiles per
/// iteration, so four tile reads are in flight and each input quad serves both tiles. Every output cell
/// keeps its k order (tile sg, sg+4, ...) and the epilogue its r-then-simdgroup sum, so the bytes are the
/// lane kernels'. Three or more tiles per group or k-tiles per iteration spill.
pub const ONE_ROW_COOP_SOURCE: [:0]const u8 =
    @embedFile("kernels/one_row_coop.metal");
const ONE_ROW_PAIR_SOURCE: [:0]const u8 =
    @embedFile("kernels/one_row_pair.metal") ++ ONE_ROW_COOP_SOURCE;
var lane_pair_one_cfgs: CfgCache(IndexedKey, 8) = .{};
var lane_pair_one_kernel: KernelSlots = no_kernels;
var down_one_cfgs: CfgCache(IndexedKey, 8) = .{};
var down_one_kernel: KernelSlots = no_kernels;
/// Dispatches of the one-row kernels; a test seam, like the other engagement counters.
var one_row_coop_dispatches: usize = 0;
var one_row_coop_off: bool = false;
var one_row_coop_logged: bool = false;
fn countOneRow(n: u32) void {
    one_row_coop_dispatches += 1;
    if (one_row_coop_logged) return;
    one_row_coop_logged = true;
    log.info("[exl3-decode] one-row coop GEMV engaged n{d}\n", .{n});
}
/// Whole H128 blocks give 8k-tile iterations; an even tile count gives the tile pairs.
fn oneRowCoopServes(in_tiles: c_int, out_tiles: c_int) bool {
    return !one_row_coop_off and in_tiles > 0 and out_tiles > 0 and @mod(in_tiles, 8) == 0 and @mod(out_tiles, 2) == 0;
}
fn downLaneCoop(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    return downLaneCoopWith(s, x, trellis, slots, false);
}
pub fn downLaneCoopWith(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array, one_row: bool) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    if (one_row and !oneRowCoopServes(tsh[1], tsh[2])) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk, .n = rate.n };
    const cache = if (one_row) &down_one_cfgs else &down_lane_coop_cfgs;
    const cfg = cache.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * (if (one_row) @as(c_int, 64) else 128), topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        cache.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = if (one_row)
        try codebookKernel(&down_one_kernel, "sushi_glm_down_one", &.{ "x", "trellis", "slots" }, &.{"y"}, ONE_ROW_COOP_SOURCE)
    else
        try getDownLaneCoopKernel();
    if (one_row) countOneRow(rate.n);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}
const DownLaneArgs = struct { s: mlx.mlx_stream, g: mlx.mlx_array, u: mlx.mlx_array, v: [10]mlx.mlx_array, up_scale: mlx.mlx_array, inter: c_int, slots: c_int, tiles: c_int };

// GLM serial-decode routed chain at the served geometry (E288, 4096/2048, top-8, clamp 10, MCG w14), one row.
// Prints only under SUSHI_GLM_T1_UBENCH (a diagnostic, never a test).
const T1_ROUTES = 24;
