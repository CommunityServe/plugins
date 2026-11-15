//! The NAX router plan: a sigmoid router at GLM-5.3-Flash's dims (K 4096, N 288) in fewer launches, its routes and
//! weights the stock route's bit for bit. The stock route (`moe.mlp`) casts x (bf16) to f32 and multiplies it by the f32
//! table (the stored f16 or bf16 router, widened): at 2 to 4 rows with TF32 on MLX takes its split-K NAX GEMM
//! (`steel_gemm_splitk_nax_nt`, K split at 2048, one chain a 32-expert block and a partition) and sums the partials as
//! (0 + P0) + P1; at one row its gemv. Then `router_kernel.moeRouterTopK` (`.sigmoid_bias`).
//! - 2 to 4 rows: phase 1 (kernels/moe/router_nax_p1.metal) runs the same 18 chains on the same operands in the same k
//!   order over the router as stored; phase 2 is the stock router text with its one logits read summing the partials.
//! - One row (`GLM53F_ROUTER_ONE1`): MLX's own `GEMVKernel` (the pinned MLX's gemv.h, its operand types and its
//!   store edited), then the stock router text in the threadgroup that arrives last.
//! Every edit to a stock text is found exactly once at compile time. A call outside the plan returns null: the caller
//! routes as before.
//! copied_from: mlx-serve src/sdk/blocks/moe/router_nax.zig@4bab19a5 (#1196), its tests on the GPU in src/gpu_tests.zig;
//! the trunk's router hook (`fill`, `run`) is this plugin's.

const std = @import("std");
const kit = @import("mlx_serve");
const kernelSource = @import("kernels.zig").source;
const options = @import("options.zig");
const mlx = kit.mlx;
const device = kit.mlx_device;
const log = kit.log;
const lane_qmm = kit.blocks.qmm.lane_qmm;
const steel = @import("mlx_steel_sources");
const router_kernel = kit.blocks.moe.router_kernel;
const MoeRouting = kit.blocks.moe.routing.MoeRouting;
const moe = kit.bases.blocks.moe;

const Array = mlx.mlx_array;

/// The router's geometry the stock route's MLX plan is read for: the hidden (MLX splits K <= 4096 into two partitions
/// of 2048) and the routed experts (whole 32-expert blocks).
pub const hidden: c_int = 4096;
pub const experts: c_int = 288;
pub const partition: c_int = 2048;
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 4;
/// Phase 1's threadgroup (32 experts by 64 k a stage, 16 B a thread) and its chains.
pub const p1_threads: c_int = 256;
pub const p1_groups: c_int = @divExact(experts, 32) * @divExact(hidden, partition);
/// MLX's gemv at bm4 bn1 sm1 sn32 tm4 tn4: 16 outputs a threadgroup of 128 threads.
pub const one1_threads: c_int = 128;
pub const one1_groups: c_int = @divExact(experts, 16);

comptime {
    // phase 2 runs on phase 1's threadgroup: the stock router's own launch at 288 experts is 256 threads too
    if (router_kernel.moeRouterThreadGroup(experts) != p1_threads) @compileError("router_nax: the stock router's threadgroup at 288 experts is not phase 1's 256 threads");
}

// ── switches ──

pub const Switches = struct { nax: bool, one1: bool };

fn switches() Switches {
    return .{ .nax = options.on(.router_nax), .one1 = options.on(.router_one1) };
}

/// Whether the load keeps a router as stored for the plan (`fill`): its switch on, a NAX GPU.
pub fn enabled() bool {
    return options.on(.router_nax) and device.naxAvailable();
}

// ── the texts ──

const p1_source: [:0]const u8 = kernelSource("router_nax_p1.metal");
const sigmoid_text: [:0]const u8 = router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias)];

/// The stock router text's logits read and phase 2's: MLX's split-K accumulate (`out = 0; out += C0; out += C1`).
const stock_read = "float l = float(logits[rbase + e]);";
const sum_read = "float l = (0.0f + float(logits[rbase + e])) + float(logits[PSTRIDE + rbase + e]);";
const p2_source: [:0]const u8 = replaceOnce(sigmoid_text, stock_read, sum_read);

fn replaceOnce(comptime s: []const u8, comptime old: []const u8, comptime new: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, s, old) orelse @compileError("router_nax: the stock text has no `" ++ old ++ "`");
    if (std.mem.indexOfPos(u8, s, i + old.len, old) != null) @compileError("router_nax: the stock text holds `" ++ old ++ "` twice");
    return std.fmt.comptimePrint("{s}{s}{s}", .{ s[0..i], new, s[i + old.len ..] });
}

/// Each (old, new) of `edits` in turn, each `old` found exactly once.
fn editsOnce(comptime s: []const u8, comptime edits: []const [2][]const u8) [:0]const u8 {
    @setEvalBranchQuota(1_000_000);
    comptime var out: [:0]const u8 = std.fmt.comptimePrint("{s}", .{s});
    inline for (edits) |e| out = replaceOnce(out, e[0], e[1]);
    return out;
}

/// `s` between `open` and `close`, each found exactly once, `close` after `open`.
fn between(comptime s: []const u8, comptime open: []const u8, comptime close: []const u8) []const u8 {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, s, open) orelse @compileError("router_nax: the stock text has no `" ++ open ++ "`");
    if (std.mem.indexOfPos(u8, s, i + open.len, open) != null) @compileError("router_nax: the stock text holds `" ++ open ++ "` twice");
    const j = std.mem.indexOfPos(u8, s, i + open.len, close) orelse @compileError("router_nax: the stock text has no `" ++ close ++ "` after `" ++ open ++ "`");
    if (std.mem.indexOfPos(u8, s, j + close.len, close) != null) @compileError("router_nax: the stock text holds `" ++ close ++ "` twice");
    return s[i + open.len .. j];
}

/// A logit read device-coherently (`pload`) by the one-row launch's last threadgroup.
const pload_text: []const u8 = kernelSource("router_nax_pload.metal");

// The one-row launch: MLX's GEMVKernel from its constant macro to the vector-matrix banner, its loaders and `run` taking
// the operands' stored types (x bf16, the router f16 or bf16, widened by the loaders' own static_cast: the stock cast's
// and f32 copy's exact values) and its output stored atomically. MLX puts utils.h before every custom kernel.
const gemv_open = "#define MLX_MTL_CONST static constant constexpr const\n";
const gemv_close = "///////////////////////////////////////////////////////////////////////////////\n/// Vector matrix multiplication\n";
const gemv_kernel: [:0]const u8 = gemvKernel(withoutCr(steel.gemv));
fn gemvKernel(comptime gemv: []const u8) [:0]const u8 {
    @setEvalBranchQuota(200_000_000);
    return comptime editsOnce(gemv_open ++ between(gemv, gemv_open, gemv_close), &.{
        .{ "template <typename U = T>\n  static METAL_FUNC void\n  load_unsafe(const device T* src,", "template <typename U = T, typename S = T>\n  static METAL_FUNC void\n  load_unsafe(const device S* src," },
        .{ "template <typename U = T>\n  static METAL_FUNC void load_safe(\n      const device T* src,", "template <typename U = T, typename S = T>\n  static METAL_FUNC void load_safe(\n      const device S* src," },
        .{ "  static METAL_FUNC void run(\n      const device T* mat [[buffer(0)]],\n      const device T* in_vec [[buffer(1)]],", "  template <typename MT, typename VT>\n  static METAL_FUNC void run(\n      const device MT* mat [[buffer(0)]],\n      const device VT* in_vec [[buffer(1)]]," },
        .{ "out_vec[out_row + tm] = static_cast<T>(result[tm]);", "atomic_store_explicit(reinterpret_cast<device atomic_uint*>(out_vec + out_row + tm), as_type<uint>(static_cast<T>(result[tm])), memory_order_relaxed);" },
    });
}

/// `s` without its carriage returns. MLX's gemv.h is a submodule file that .gitattributes does not pin to LF, so a
/// checkout that converts line endings (Windows) gives it CRLF lines; the searches above are over LF text.
fn withoutCr(comptime s: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(20_000_000);
        var n: usize = 0;
        for (s) |c| n += @intFromBool(c != '\r');
        var out: [n]u8 = undefined;
        var i: usize = 0;
        for (s) |c| if (c != '\r') {
            out[i] = c;
            i += 1;
        };
        const final = out;
        return &final;
    }
}
const one1_header: [:0]const u8 = blk: {
    @setEvalBranchQuota(200_000_000);
    const args = std.fmt.comptimePrint(kernelSource("router_nax_one1_header.fmt.metal"), .{ hidden, experts, hidden });
    break :blk std.fmt.comptimePrint("using namespace metal;\n{s}\n{s}\n{s}", .{ gemv_kernel, args, pload_text });
};
const one1_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(200_000_000);
    break :blk std.fmt.comptimePrint(kernelSource("router_nax_one1.fmt.metal"), .{ one1_groups, replaceOnce(sigmoid_text, stock_read, "float l = pload(logits, rbase + e);") });
};

// ── the plan's checks (MLX-free over shapes, dtypes and strides) ──

/// The rows of a call over `x` of shape `sh` (every leading dimension), or null when its last is not the hidden.
pub fn rowsOf(sh: []const c_int) ?c_int {
    if (sh.len < 2 or sh.len > 4 or sh[sh.len - 1] != hidden) return null;
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| rows *= d;
    return rows;
}

/// Row-contiguous: every dimension of more than one element at its row-major stride.
pub fn stridesRowMajor(sh: []const c_int, st: []const usize) bool {
    var want: usize = 1;
    var i = sh.len;
    while (i > 0) {
        i -= 1;
        if (sh[i] > 1 and st[i] != want) return false;
        want *= @intCast(@max(sh[i], 1));
    }
    return true;
}

fn rowContiguous(a: Array) bool {
    const sh = mlx.getShape(a);
    return stridesRowMajor(sh, mlx.mlx_array_strides(a)[0..sh.len]);
}

/// The f32 table as the stock matmul reads it: [4096, 288] with strides [1, 4096] (the transposed view of a
/// row-contiguous [288, 4096]), which MLX's matmul takes as transpose_b (the `nt` kernel phase 1 reproduces).
pub fn tableLayout(sh: []const c_int, st: []const usize) bool {
    return sh.len == 2 and sh[0] == hidden and sh[1] == experts and st[0] == 1 and st[1] == @as(usize, @intCast(hidden));
}

/// The weight types phase 1 and the one-row gemv read as stored (2 bytes).
pub fn weightServes(dt: mlx.mlx_dtype) bool {
    return dt == .float16 or dt == .bfloat16;
}

/// Which launch a call of `rows` takes, or null: the stock route.
pub const Arm = enum { two, one1 };

pub fn armFor(rows: c_int, sw: Switches) ?Arm {
    if (!sw.nax) return null;
    if (rows == 1) return if (sw.one1) .one1 else null;
    if (rows < min_rows or rows > max_rows) return null;
    return .two;
}

pub const Reason = enum { top_k, x, table, bias, tf32, stream };
/// A call's facts as the plan reads them: x bf16 row-contiguous, the stock table a plain f32 matmul in `tableLayout`,
/// the selection bias f32 [288] row-contiguous, the route's k, MLX's TF32 matmul, a GPU stream.
pub const Facts = struct { x: bool, table: bool, bias: bool, k: c_int, tf32: bool, gpu: bool };

/// Why a call of `arm` routes as before, or null. The one-row gemv takes no TF32 route, so only the others ask it.
pub fn decline(f: Facts, arm: Arm) ?Reason {
    if (f.k < 1 or f.k > 32) return .top_k;
    if (!f.x) return .x;
    if (!f.table) return .table;
    if (!f.bias) return .bias;
    if (arm != .one1 and !f.tf32) return .tf32;
    if (!f.gpu) return .stream;
    return null;
}

/// A sigmoid router (no groups) as the MoE block holds it: `raw` as stored (`admit`), `table` the stock route's f32
/// operand (`plain`: a dense matmul with no scales, bias or other arm), the selection bias, and the route's k, norm and
/// scale.
pub const Router = struct { raw: Array, table: Array, plain: bool, bias: Array, k: c_int, norm: bool, scale: f32 };

fn tableServes(t: Array) bool {
    if (t.ctx == null or mlx.mlx_array_dtype(t) != .float32) return false;
    const sh = mlx.getShape(t);
    return sh.len == 2 and tableLayout(sh, mlx.mlx_array_strides(t)[0..2]);
}

fn factsOf(s: mlx.mlx_stream, x: Array, r: Router) Facts {
    return .{
        .x = mlx.mlx_array_dtype(x) == .bfloat16 and rowContiguous(x),
        .table = r.plain and tableServes(r.table),
        .bias = r.bias.ctx != null and mlx.mlx_array_dtype(r.bias) == .float32 and mlx.mlx_array_size(r.bias) == @as(usize, @intCast(experts)) and rowContiguous(r.bias),
        .k = r.k,
        .tf32 = device.tf32Matmul(),
        .gpu = mlx.streamIsGpu(s),
    };
}

/// Whether `raw`, a router as stored, is the plan's: f16 or bf16 [288, 4096], row-contiguous, 16-byte aligned (phase 1
/// reads it 16 B a thread). Evaluates it.
pub fn admit(raw: Array) !bool {
    if (raw.ctx == null or !weightServes(mlx.mlx_array_dtype(raw)) or !std.mem.eql(c_int, mlx.getShape(raw), &.{ experts, hidden }) or !rowContiguous(raw)) return false;
    try mlx.check(mlx.mlx_array_eval(raw));
    const addr: usize = switch (mlx.mlx_array_dtype(raw)) {
        .float16 => @intFromPtr(mlx.mlx_array_data_float16(raw) orelse return false),
        else => @intFromPtr(mlx.mlx_array_data_bfloat16(raw) orelse return false),
    };
    return addr % 16 == 0;
}

// ── the route ──

var declined_said: [@typeInfo(Reason).@"enum".field_names.len]bool = @splat(false);
var nax_said = false;
var one1_said = false;

/// The routes of `x` (the MoE block's input) by `r` on the plan, or null: the caller routes as before. The arrays are
/// new (the caller adopts them as the stock route's).
pub fn route(s: mlx.mlx_stream, x: Array, r: Router) !?MoeRouting {
    if (r.raw.ctx == null) return null;
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse return null;
    const arm = armFor(rows, switches()) orelse return null;
    if (decline(factsOf(s, x, r), arm)) |why| {
        if (!declined_said[@backingInt(why)]) {
            declined_said[@backingInt(why)] = true;
            log.info("[moe] router NAX declined: a {d}-row call routes as before ({s})\n", .{ rows, @tagName(why) });
        }
        return null;
    }
    switch (arm) {
        .one1 => {
            const o = try routeOne1(s, x, r.raw, r.bias, r.k, r.norm, r.scale);
            // the logits are the launch's scratch: the graph keeps them until the launch has run
            _ = mlx.mlx_array_free(o.logits);
            if (!one1_said) {
                one1_said = true;
                log.info("[moe] router one-row engaged: K={d} N={d} ({d} threadgroups)\n", .{ hidden, experts, one1_groups });
            }
            return o.routing;
        },
        .two => {
            const p = try partials(s, x, r.raw, rows);
            defer _ = mlx.mlx_array_free(p);
            const o = try topK(s, p, xsh, r.bias, r.k, r.norm, r.scale);
            naxSaid(rows);
            return o;
        },
    }
}

fn naxSaid(rows: c_int) void {
    if (nax_said) return;
    nax_said = true;
    log.info("[moe] router NAX engaged: rows={d} K={d} N={d}\n", .{ rows, hidden, experts });
}

// ── the launches ──

fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, contiguous: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, contiguous, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn applyKernel(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array, outs: []Array) !void {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = x;
        got += 1;
    }
}

/// A call's shape and its route's constants.
const Key = struct { wt: u8 = 0, dims: [4]c_int = @splat(0), len: u8 = 0, k: c_int = 0, norm: bool = false, scale: u32 = 0 };

/// A config cache keyed by everything a config bakes in (the call's full shape among them).
const Cache = struct {
    const Slot = struct { key: Key = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null, scale: Array = .{ .ctx = null } };
    slots: [4]Slot = @splat(.{}),
    next: usize = 0,

    /// The slot for `key`, or a fresh one (its config and scale freed) the caller fills.
    fn find(self: *Cache, key: Key) struct { slot: *Slot, hit: bool } {
        for (&self.slots) |*sl| if (sl.cfg != null and std.meta.eql(sl.key, key)) return .{ .slot = sl, .hit = true };
        const sl = &self.slots[self.next];
        self.next = (self.next + 1) % self.slots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        if (sl.scale.ctx != null) _ = mlx.mlx_array_free(sl.scale);
        sl.* = .{};
        return .{ .slot = sl, .hit = false };
    }
};

fn keyOf(wt: mlx.mlx_dtype, xsh: []const c_int, k: c_int, norm: bool, scale: f32) Key {
    var key: Key = .{ .wt = @intFromBool(wt == .bfloat16), .len = @intCast(xsh.len), .k = k, .norm = norm, .scale = @bitCast(scale) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    return key;
}

/// `xsh` with its last dimension `last`.
fn shapeWith(xsh: []const c_int, last: c_int) [4]c_int {
    var out: [4]c_int = @splat(0);
    for (xsh[0 .. xsh.len - 1], 0..) |d, i| out[i] = d;
    out[xsh.len - 1] = last;
    return out;
}

const Cfg = mlx.mlx_fast_metal_kernel_config;

fn templateInt(cfg: Cfg, name: [*:0]const u8, v: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, v));
}

/// Phase 1's template arguments: the weight's type, the rows, the dims.
fn p1Templates(cfg: Cfg, wt: mlx.mlx_dtype, rows: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "WT", wt));
    try templateInt(cfg, "M", rows);
    try templateInt(cfg, "N", experts);
    try templateInt(cfg, "K", hidden);
    try templateInt(cfg, "PS", partition);
}

/// The stock router text's template arguments (phase 2).
fn routeTemplates(cfg: Cfg, tg: c_int, k: c_int, norm: bool) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TOUT", .bfloat16));
    try templateInt(cfg, "NE", experts);
    try templateInt(cfg, "NK", k);
    try templateInt(cfg, "TG", tg);
    try templateInt(cfg, "NORM", @intFromBool(norm));
}

fn gridOf(cfg: Cfg, threads: c_int, groups: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads * groups, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
}

fn partialsOut(cfg: Cfg, rows: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @divExact(hidden, partition), rows, experts }, 3, .float32));
}

var p1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var p1_cfgs: [2][@as(usize, @intCast(max_rows)) + 1]?Cfg = @splat(@splat(null));

/// Phase 1 alone: MLX's split-K partials of `x` (bf16 [.., rows, 4096], 2 to 4 rows) against `raw` (f16 or bf16
/// [288, 4096] as stored), f32 [2, rows, 288].
fn partials(s: mlx.mlx_stream, x: Array, raw: Array, rows: c_int) !Array {
    if (rows < min_rows or rows > max_rows) return error.RouterNaxRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.RouterNaxWeightType;
    const slot = &p1_cfgs[@intFromBool(wt == .bfloat16)][@intCast(rows)];
    if (slot.* == null) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try partialsOut(cfg, rows);
        try gridOf(cfg, p1_threads, p1_groups);
        try p1Templates(cfg, wt, rows);
        slot.* = cfg;
    }
    const kern = try kernelOf(&p1_kernel, "mlxserve_moe_router_nax_p1", &.{ "x", "w" }, &.{"p"}, p1_source, try lane_qmm.prefillHeader(), false);
    var outs: [1]Array = undefined;
    try applyKernel(s, kern, slot.*.?, &.{ x, raw }, &outs);
    return outs[0];
}

var p2_kernel: ?mlx.mlx_fast_metal_kernel = null;
var p2_cfgs: Cache = .{};

/// Phase 2 alone: the stock router text over `p` (phase 1's partials of a call over x of shape `xsh`), inds u32 and
/// scores bf16 of shape [.., rows, k] as the stock route's.
fn topK(s: mlx.mlx_stream, p: Array, xsh: []const c_int, bias: Array, k: c_int, norm: bool, scale: f32) !MoeRouting {
    const rows = rowsOf(xsh) orelse return error.RouterNaxRows;
    const f = p2_cfgs.find(keyOf(.float32, xsh, k, norm, scale));
    if (!f.hit) {
        const tg = router_kernel.moeRouterThreadGroup(experts);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const out_shape = shapeWith(xsh, k);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
        try routeTemplates(cfg, tg, k, norm);
        try templateInt(cfg, "PSTRIDE", rows * experts);
        f.slot.* = .{ .key = keyOf(.float32, xsh, k, norm, scale), .cfg = cfg, .scale = mlx.mlx_array_new_float(scale) };
    }
    const kern = try kernelOf(&p2_kernel, "mlxserve_moe_router_nax_p2", &.{ "logits", "bias", "scale" }, &.{ "inds", "scores" }, p2_source, "", true);
    var outs: [2]Array = undefined;
    try applyKernel(s, kern, f.slot.cfg.?, &.{ p, bias, f.slot.scale }, &outs);
    return .{ .inds = outs[0], .norm_scores = outs[1] };
}

/// A call's arrival counter: a fresh zeroed array, the call's own (no two calls share one). 16 words: MLX binds an
/// input of 8 or more elements in device memory, fewer in constant memory.
fn arrivals() !Array {
    const zeros: [16]u32 = @splat(0);
    const a = mlx.mlx_array_new_data(&zeros, &[_]c_int{zeros.len}, 1, .uint32);
    if (a.ctx == null) return error.RouterNaxCounter;
    return a;
}

var one1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var one1_cfgs: Cache = .{};

/// A one-row call's launch outputs: the logits f32 [.., 1, 288] (MLX's gemv's) and the routes.
const One1 = struct { logits: Array, routing: MoeRouting };

/// A one-row router call in one launch: the routes of `x` (bf16 [.., 1, 4096]) by `raw` (f16 or bf16 [288, 4096] as
/// stored) and `bias`, inds u32 and scores bf16 of shape [.., 1, k] as the stock route's, and its logits.
fn routeOne1(s: mlx.mlx_stream, x: Array, raw: Array, bias: Array, k: c_int, norm: bool, scale: f32) !One1 {
    const xsh = mlx.getShape(x);
    if ((rowsOf(xsh) orelse return error.RouterNaxRows) != 1) return error.RouterNaxRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.RouterNaxWeightType;
    const f = one1_cfgs.find(keyOf(wt, xsh, k, norm, scale));
    if (!f.hit) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const logits_shape = shapeWith(xsh, experts);
        const out_shape = shapeWith(xsh, k);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &logits_shape, xsh.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .bfloat16));
        try gridOf(cfg, one1_threads, one1_groups);
        try routeTemplates(cfg, one1_threads, k, norm);
        f.slot.* = .{ .key = keyOf(wt, xsh, k, norm, scale), .cfg = cfg, .scale = mlx.mlx_array_new_float(scale) };
    }
    const kern = try kernelOf(&one1_kernel, "mlxserve_moe_router_nax_one1", &.{ "x", "w", "bias", "scale", "counter" }, &.{ "logits", "inds", "scores" }, one1_source, one1_header, false);
    const cnt = try arrivals();
    defer _ = mlx.mlx_array_free(cnt);
    var outs: [3]Array = undefined;
    try applyKernel(s, kern, f.slot.cfg.?, &.{ x, raw, bias, f.slot.scale, cnt }, &outs);
    return .{ .logits = outs[0], .routing = .{ .inds = outs[1], .norm_scores = outs[2] } };
}

// ── the trunk's router hook (`trunk.Hooks.router`) ──

/// The routers kept as stored for the plan, one a routed layer, freed at unload (`release`). GLM-5.3-Flash has 43.
const max_layers = 128;
var kept: [max_layers]Array = @splat(.{ .ctx = null });
var kept_n: usize = 0;

/// `trunk.Hooks.Router.fill`: a routed layer's router as stored (`[288, 4096]`, before the trunk widens it), kept for
/// the plan where `admit` takes it; null (the block's route) with the switch off, off NAX, or a router the plan does
/// not take.
pub fn fill(_: ?*anyopaque, stored: Array) anyerror!?moe.RouterHook {
    if (!enabled() or !(try admit(stored))) return null;
    if (kept_n == kept.len) return error.RouterNaxTooManyLayers;
    var raw = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&raw, stored);
    kept[kept_n] = raw;
    kept_n += 1;
    return .{ .ctx = @ptrCast(&kept[kept_n - 1]), .run = &run };
}

/// The layer's routes on the plan (`route`), or null: the block's own route.
fn run(ctx: ?*const anyopaque, w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) anyerror!?MoeRouting {
    const raw: *const Array = @ptrCast(@alignCast(ctx orelse return null));
    const sb = switch (w.route) {
        .sigmoid_bias => |r| r,
        else => return null,
    };
    if (sb.n_group > 1) return null;
    const l = &w.router;
    const plain = l.s.ctx == null and l.bias.ctx == null and l.ext == null and l.rht == null and !l.ternary;
    return route(s, x, .{ .raw = raw.*, .table = l.w, .plain = plain, .bias = sb.expert_bias, .k = w.top_k, .norm = sb.norm, .scale = sb.scale });
}

/// Frees the kept routers (the model's unload).
pub fn release() void {
    for (kept[0..kept_n]) |*a| {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    }
    kept_n = 0;
}

// ── tests ──

const testing = std.testing;

test "router_nax: a call's rows are its leading dimensions over the hidden; the f32 table is taken only as the transposed view MLX's matmul reads as transpose_b" {
    try testing.expectEqual(@as(?c_int, 3), rowsOf(&.{ 1, 3, 4096 }));
    try testing.expectEqual(@as(?c_int, 4), rowsOf(&.{ 4, 4096 }));
    try testing.expectEqual(@as(?c_int, 2), rowsOf(&.{ 2, 1, 4096 }));
    try testing.expectEqual(@as(?c_int, null), rowsOf(&.{ 1, 3, 2048 }));
    try testing.expectEqual(@as(?c_int, null), rowsOf(&.{4096}));
    try testing.expect(tableLayout(&.{ 4096, 288 }, &.{ 1, 4096 }));
    try testing.expect(!tableLayout(&.{ 4096, 288 }, &.{ 288, 1 }));
    try testing.expect(!tableLayout(&.{ 4096, 256 }, &.{ 1, 4096 }));
    try testing.expect(weightServes(.float16) and weightServes(.bfloat16) and !weightServes(.float32));
    try testing.expect(stridesRowMajor(&.{ 1, 3, 4096 }, &.{ 99, 4096, 1 }));
    try testing.expect(!stridesRowMajor(&.{ 1, 3, 4096 }, &.{ 12288, 1, 3 }));
}

test "router_nax: a call takes its launch by its rows and the switches, and routes as before outside the plan" {
    const all: Switches = .{ .nax = true, .one1 = true };
    try testing.expectEqual(@as(?Arm, .one1), armFor(1, all));
    try testing.expectEqual(@as(?Arm, .two), armFor(2, all));
    try testing.expectEqual(@as(?Arm, .two), armFor(4, all));
    try testing.expectEqual(@as(?Arm, null), armFor(5, all));
    try testing.expectEqual(@as(?Arm, .two), armFor(3, .{ .nax = true, .one1 = false }));
    try testing.expectEqual(@as(?Arm, null), armFor(1, .{ .nax = true, .one1 = false }));
    for ([_]c_int{ 1, 2, 4 }) |rows| try testing.expectEqual(@as(?Arm, null), armFor(rows, .{ .nax = false, .one1 = true }));

    const ok: Facts = .{ .x = true, .table = true, .bias = true, .k = 8, .tf32 = true, .gpu = true };
    for ([_]Arm{ .two, .one1 }) |arm| try testing.expectEqual(@as(?Reason, null), decline(ok, arm));
    var f = ok;
    f.tf32 = false;
    // the one-row gemv takes no TF32 route; the split-K GEMM does
    try testing.expectEqual(@as(?Reason, null), decline(f, .one1));
    try testing.expectEqual(@as(?Reason, .tf32), decline(f, .two));
    f = ok;
    f.k = 33;
    try testing.expectEqual(@as(?Reason, .top_k), decline(f, .two));
    f = ok;
    f.x = false;
    try testing.expectEqual(@as(?Reason, .x), decline(f, .two));
    f = ok;
    f.table = false;
    try testing.expectEqual(@as(?Reason, .table), decline(f, .one1));
    f = ok;
    f.bias = false;
    try testing.expectEqual(@as(?Reason, .bias), decline(f, .two));
    f = ok;
    f.gpu = false;
    try testing.expectEqual(@as(?Reason, .stream), decline(f, .one1));
}
