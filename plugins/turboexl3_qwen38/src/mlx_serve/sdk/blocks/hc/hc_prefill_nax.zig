//! copied_from: src/sdk/blocks/hc/hc_prefill_nax.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The hc prefill read on the tensor units, each launch bit for bit the MLX chain it replaces: the down product
//! (`naxDown`) and the up product with the mix as its epilogue (`naxUpMix`); the down with its silu and the inject
//! product with its gate in one launch (`fuseDown`, `readDown`); the read with the normalized rows never written
//! (`normInv`, `fuseDownNorm`, `rwDown`, `upMixNorm`, `rwUpMix`); the up + mix in registers (`readUpMix`). Each
//! declines (null) where it does not replay MLX's chain, and the caller runs the chain.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const log = @import("../../util/log.zig");
const hp = @import("hc_prefill.zig");
const lane_qmm = @import("../qmm/lane_qmm.zig");

/// Test seam: every launch here declines, so the read runs MLX's chain.
pub var disabled_for_test = false;

// ── launches ──

pub const Kernel = enum { nax_down, nax_upmix, fuse_down, fuse_down_norm, fuse_upmix, norm_inv, read_down, read_upmix, rw_down, rw_upmix };
const n_kernels = @typeInfo(Kernel).@"enum".field_names.len;

/// The header a kernel sits on: MLX's steel NAX headers, then each layer of the read's helpers one file more.
pub const Header = enum { none, nax, fuse, read };

/// A launch as the wrappers build it and the kernel contract declares it.
pub const KernelText = struct { name: [:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8, header: Header };

pub fn text(k: Kernel) KernelText {
    return switch (k) {
        .nax_down => .{ .name = "msv_hc_prefill_nax_down", .inputs = &.{ "x", "w", "rows" }, .outputs = &.{"y"}, .source = kernelSource("kernels/hc/hc_prefill_nax_down.metal"), .header = .nax },
        .nax_upmix => .{ .name = "msv_hc_prefill_nax_upmix", .inputs = &.{ "act", "w", "normed", "sigtab", "rows" }, .outputs = &.{"out"}, .source = kernelSource("kernels/hc/hc_prefill_nax_upmix.metal"), .header = .nax },
        .fuse_down => .{ .name = "msv_hc_prefill_fuse_down", .inputs = &.{ "x", "w", "iw", "stab", "gtab", "rows" }, .outputs = &.{ "act", "inj" }, .source = kernelSource("kernels/hc/hc_prefill_fuse_down.metal"), .header = .fuse },
        .fuse_down_norm => .{ .name = "msv_hc_prefill_fuse_down_norm", .inputs = &.{ "xs", "invs", "nw", "w", "iw", "stab", "gtab", "rows" }, .outputs = &.{ "act", "inj" }, .source = kernelSource("kernels/hc/hc_prefill_fuse_down_norm.metal"), .header = .fuse },
        .fuse_upmix => .{ .name = "msv_hc_prefill_fuse_upmix", .inputs = &.{ "act", "w", "xs", "invs", "nw", "sigtab", "rows" }, .outputs = &.{"out"}, .source = kernelSource("kernels/hc/hc_prefill_fuse_upmix.metal"), .header = .fuse },
        .norm_inv => .{ .name = "msv_hc_prefill_norm_inv", .inputs = &.{ "x", "eps", "wo", "wi" }, .outputs = &.{ "stream", "invs" }, .source = kernelSource("kernels/hc/hc_prefill_norm_inv.metal"), .header = .none },
        .read_down => .{ .name = "msv_hc_prefill_read_down", .inputs = &.{ "x", "w", "iw", "stab", "gtab", "rows" }, .outputs = &.{ "act", "inj" }, .source = kernelSource("kernels/hc/hc_prefill_read_down.metal"), .header = .read },
        .read_upmix => .{ .name = "msv_hc_prefill_read_upmix", .inputs = &.{ "act", "w", "normed", "sigtab", "rows" }, .outputs = &.{"out"}, .source = kernelSource("kernels/hc/hc_prefill_read_upmix.metal"), .header = .read },
        .rw_down => .{ .name = "msv_hc_prefill_rw_down", .inputs = &.{ "xs", "invs", "nw", "w", "iw", "stab", "gtab", "rows" }, .outputs = &.{ "act", "inj" }, .source = kernelSource("kernels/hc/hc_prefill_rw_down.metal"), .header = .read },
        .rw_upmix => .{ .name = "msv_hc_prefill_rw_upmix", .inputs = &.{ "act", "w", "xs", "invs", "nw", "sigtab", "rows" }, .outputs = &.{"out"}, .source = kernelSource("kernels/hc/hc_prefill_rw_upmix.metal"), .header = .read },
    };
}

var kernel_headers: [3]?[:0]const u8 = @splat(null);

pub fn header(h: Header) ![:0]const u8 {
    if (h == .none) return "";
    var base: [:0]const u8 = try lane_qmm.prefillHeader();
    inline for (.{ Header.nax, Header.fuse, Header.read }, .{ "nax", "fuse", "read" }, 0..) |layer, file, i| {
        if (kernel_headers[i] == null) kernel_headers[i] = try std.mem.concatWithSentinel(std.heap.c_allocator, u8, &.{ base, kernelSource("kernels/hc/hc_prefill_" ++ file ++ ".metal") }, 0);
        base = kernel_headers[i].?;
        if (h == layer) return base;
    }
    unreachable;
}

var built_kernels: [n_kernels]?mlx.mlx_fast_metal_kernel = @splat(null);

fn kernelFor(k: Kernel) !mlx.mlx_fast_metal_kernel {
    const i = @intFromEnum(k);
    if (built_kernels[i]) |x| return x;
    const t = text(k);
    const ins = mlx.mlx_vector_string_new_data(t.inputs.ptr, t.inputs.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(t.outputs.ptr, t.outputs.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const x = mlx.mlx_fast_metal_kernel_new(t.name.ptr, ins, outs, t.source.ptr, (try header(t.header)).ptr, true, false);
    if (x.ctx == null) return error.MetalKernelCompileFailed;
    built_kernels[i] = x;
    return x;
}

/// Launches served, by kernel (the tests' proof that a path engaged).
pub var served: [n_kernels]u64 = @splat(0);
var launch_said: [n_kernels]bool = @splat(false);

/// A launch's dispatch as its wrapper sets it: grid (threads), threadgroup width, template args in the kernel's order
/// (`T` = bf16 first where the kernel takes one), and the threadgroup bytes that instantiation declares.
pub const Dispatch = struct {
    grid: [3]c_int,
    threads: c_int,
    t: bool = false,
    ints: [12]Int = undefined,
    n: usize = 0,
    tg_bytes: u32 = 0,

    pub const Int = struct { name: [*:0]const u8, v: c_int };

    fn args(d: *Dispatch, names: []const [*:0]const u8, vals: []const c_int) void {
        for (names, vals) |name, v| {
            d.ints[d.n] = .{ .name = name, .v = v };
            d.n += 1;
        }
    }
};

/// The grid of a launch over BM-row x (32 TILES_N)-column blocks with MLX's threadblock swizzle (2^swz row tiles a column).
fn tiled(m: c_int, bm: c_int, tiles_n: c_int, swz: c_int, threads: c_int) [3]c_int {
    const tile = @as(c_int, 1) << @intCast(swz);
    const tiles_m = @divTrunc(m + bm - 1, bm);
    return .{ tiles_n * tile * threads, @divTrunc(tiles_m + tile - 1, tile), 1 };
}

const Out = struct { shape: []const c_int, dt: mlx.mlx_dtype = .bfloat16 };

/// One launch of `k`; its outputs land in `res` (owned by the caller).
fn launch(s: mlx.mlx_stream, k: Kernel, d: Dispatch, outs: []const Out, ins: []const mlx.mlx_array, res: []mlx.mlx_array, seq: c_int) !void {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, d.grid[0], d.grid[1], d.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, d.threads, 1, 1));
    if (d.t) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    for (d.ints[0..d.n]) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, a.name, a.v));
    const in_vec = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, try kernelFor(k), in_vec, cfg, s));
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(r, out_vec, i)) catch |err| {
            for (res[0 .. i + 1]) |x| _ = mlx.mlx_array_free(x);
            return err;
        };
    }
    const i = @intFromEnum(k);
    served[i] += 1;
    // Past a server's readiness probe, once a kernel.
    if (!launch_said[i] and seq >= 512) {
        launch_said[i] = true;
        log.info("[hc-prefill] {s} engaged at S={d}\n", .{ text(k).name, seq });
    }
}

// ── plans: tiles of the hc read's geometry (hc 4 x 2560, rank 320); none changes a bit ──

/// MLX's split-K pick for a bf16 NAX matmul [m, k] x [k, n], batch folded into m (matmul.cpp's steel_matmul_axpby and its
/// split kernel's partition width): the partition width where MLX splits K, null where it runs one chain over K.
/// `ultra`: the 'd' device class, whose partitions are 2048 wide for m, n <= 2048.
pub fn mlxSplitK(m: c_int, n: c_int, k: c_int, ultra: bool) ?c_int {
    const mx = @max(m, n);
    if (!(k >= 3 * mx or (mx <= 1024 and k > 2 * mx))) return null;
    var part: c_int = 4096;
    if (ultra and m <= 2048 and n <= 2048) part = 2048;
    if (k <= 1024) {
        part = @divTrunc(k, 2);
    } else if (k <= 2048) {
        part = 1024;
    } else if (k <= 4096) {
        part = 2048;
    }
    return part;
}

var ultra_arch: ?bool = null;

fn ultraGpu() bool {
    if (ultra_arch) |v| return v;
    var buf: [128]u8 = undefined;
    const arch = device.gpuArchitecture(&buf) orelse "";
    const v = arch.len > 0 and arch[arch.len - 1] == 'd';
    ultra_arch = v;
    return v;
}

pub const DownPlan = struct { tm: c_int, wm: c_int, wn: c_int, np: c_int, part: c_int, swz: c_int, bar: c_int };

/// At most 512 threads: MLX's own NAX ceiling.
fn tileOk(tm: c_int, wm: c_int, wn: c_int, np: c_int, cols: c_int) bool {
    return (tm == 1 or tm == 2 or tm == 4) and wm >= 1 and wn >= 1 and @rem(cols, wn) == 0 and 32 * np * wm * wn <= 512;
}

/// The split mode's parked partials: (np - 1) x wm x wn tiles of 16 tm x 36 f32.
fn splitPartialBytes(np: c_int, tm: c_int, wm: c_int, wn: c_int) c_int {
    return (np - 1) * wm * wn * 16 * tm * 36 * 4;
}

/// naxDown's tiles (`n` whole 32-column runs): one chain over K in 64 x 160 tiles (R 320 in two column tiles), a barrier
/// every 2 steps; where MLX splits K, 32 x 64 tiles a partition, the partials within 28 KB.
pub fn downPlan(m: c_int, n: c_int, k: c_int) ?DownPlan {
    const part = mlxSplitK(m, n, k, ultraGpu()) orelse return chainPlan(n, k);
    if (part < 32 or @rem(part, 32) != 0) return null;
    const np = @divTrunc(k + part - 1, part);
    if (np < 2 or np > 8) return null;
    var p: DownPlan = .{ .tm = 2, .wm = 1, .wn = 2, .np = np, .part = part, .swz = 0, .bar = 0 };
    if (!tileOk(p.tm, p.wm, p.wn, np, @divTrunc(n, 32)) or splitPartialBytes(np, p.tm, p.wm, p.wn) > 28 * 1024) {
        p.tm = 1;
        p.wm = 1;
        p.wn = 1;
    }
    if (splitPartialBytes(np, p.tm, p.wm, p.wn) > 28 * 1024) return null;
    return p;
}

/// The one-chain tiles, whatever MLX does at this row count (the launches that only run one chain take them).
pub fn chainPlan(n: c_int, k: c_int) DownPlan {
    const cols = @divTrunc(n, 32);
    var p: DownPlan = .{ .tm = 2, .wm = 2, .wn = 5, .np = 1, .part = k, .swz = 0, .bar = 2 };
    if (!tileOk(p.tm, p.wm, p.wn, 1, cols)) p.wn = if (@rem(cols, 2) == 0) 2 else 1;
    return p;
}

/// Whether MLX runs one chain over K for the down product and (with `inj`) the inject product at `m` rows: the launches
/// that have no split mode serve only there.
pub fn oneChain(m: c_int, n: c_int, k: c_int, hc: c_int, inj: bool) bool {
    const ultra = ultraGpu();
    return mlxSplitK(m, n, k, ultra) == null and (!inj or mlxSplitK(m, hc, k, ultra) == null);
}

/// The inject partials' threadgroup memory in split mode: (np - 1) x wm row blocks of 16 tm rows x hc f32.
fn injPartialBytes(np: c_int, tm: c_int, wm: c_int, hc: c_int) c_int {
    return (np - 1) * wm * 16 * tm * hc * 4;
}

/// downPlan for the fused launch: the inject product must split K where the down product does, at the same width (MLX
/// decides each from max(m, n)); in split mode its partials share the threadgroup budget.
pub fn fusePlan(m: c_int, n: c_int, k: c_int, hc: c_int, inj: bool) ?DownPlan {
    if (inj) {
        const ultra = ultraGpu();
        const down_split = mlxSplitK(m, n, k, ultra);
        const inj_split = mlxSplitK(m, hc, k, ultra);
        if ((down_split == null) != (inj_split == null)) return null;
        if (down_split != null and down_split.? != inj_split.?) return null;
    }
    var p = downPlan(m, n, k) orelse return null;
    if (inj and p.np > 1 and splitPartialBytes(p.np, p.tm, p.wm, p.wn) + injPartialBytes(p.np, p.tm, p.wm, hc) > 28 * 1024) {
        p.tm = 1;
        p.wm = 1;
        p.wn = 1;
        if (splitPartialBytes(p.np, p.tm, p.wm, p.wn) + injPartialBytes(p.np, p.tm, p.wm, hc) > 28 * 1024) return null;
    }
    return p;
}

pub const UpPlan = struct { tm: c_int, wm: c_int, swz: c_int, bar: c_int, pf: c_int };

/// naxUpMix's tiles: HC divides 16 tm rows; at most 512 threads; wm x hc bf16 tiles of 16 tm x 40 within 32 KB.
fn upTileOk(tm: c_int, wm: c_int, hc: c_int) bool {
    return (tm == 1 or tm == 2 or tm == 4) and wm >= 1 and @rem(16 * tm, hc) == 0 and 32 * wm * hc <= 512 and wm * hc * 16 * tm * 40 * 2 <= 32 * 1024;
}

pub fn upPlan(hc: c_int) UpPlan {
    var p: UpPlan = .{ .tm = 2, .wm = 2, .swz = 2, .bar = 8, .pf = 0 };
    if (!upTileOk(p.tm, p.wm, hc)) p.wm = 1;
    return p;
}

pub const NormPlan = struct { tm: c_int, wm: c_int, wn: c_int, swz: c_int };

/// At most 1024 threads (MLX's custom-kernel ceiling); the two staged A buffers (2 x BM x 40 bf16) within 32 KB.
fn normTileOk(tm: c_int, wm: c_int, wn: c_int, cols: c_int) bool {
    if (!(tm == 1 or tm == 2 or tm == 4) or wm < 1 or wn < 1 or @rem(cols, wn) != 0) return false;
    return 32 * wm * wn <= 1024 and 2 * 16 * tm * wm * 40 * 2 <= 32 * 1024;
}

/// fuseDownNorm's tiles: 64 x 160 (each value normalized in 2 threadgroups at R 320).
pub fn normPlan(n: c_int) ?NormPlan {
    if (n < 32 or @rem(n, 32) != 0) return null;
    const cols = @divExact(n, 32);
    var p: NormPlan = .{ .tm = 2, .wm = 2, .wn = 5, .swz = 2 };
    if (!normTileOk(p.tm, p.wm, p.wn, cols)) {
        p.wn = if (@rem(cols, 2) == 0) 2 else 1;
        if (!normTileOk(p.tm, p.wm, p.wn, cols)) return null;
    }
    return p;
}

/// The register-form up + mix's tiles (readUpMix, rwUpMix): a simdgroup owns 16 tm rows x 32 columns of every stream.
pub const RegPlan = struct { tm: c_int, wm: c_int, wn: c_int, swz: c_int, bar: c_int };

/// At most 512 threads; tm hc accumulators of 16 floats a lane at most 128 (tm hc <= 8); wn divides the H / 32 runs.
fn regTileOk(p: RegPlan, hc: c_int, hidden: c_int) bool {
    if (!(p.tm == 1 or p.tm == 2 or p.tm == 4) or p.tm * hc > 8 or p.wm < 1 or p.wn < 1) return false;
    return @rem(@divTrunc(hidden, 32), p.wn) == 0 and 32 * p.wm * p.wn <= 512;
}

pub fn regPlan(hc: c_int, hidden: c_int) ?RegPlan {
    const p: RegPlan = .{ .tm = 1, .wm = 4, .wn = 2, .swz = 2, .bar = 0 };
    if (regTileOk(p, hc, hidden)) return p;
    const safe: RegPlan = .{ .tm = 1, .wm = 2, .wn = 1, .swz = 2, .bar = 0 };
    return if (regTileOk(safe, hc, hidden)) safe else null;
}

/// readDown's and rwDown's inject simdgroups a threadgroup: the wm row blocks' chains spread over the column tiles.
pub fn injPerTg(wm: c_int, tiles_n: c_int) c_int {
    return @divTrunc(wm + tiles_n - 1, tiles_n);
}

/// The row block inject simdgroup `j` of column tile `tile_x` runs (>= wm: idle), as the kernels pick it.
pub fn injBlock(j: c_int, tile_x: c_int, tiles_n: c_int) c_int {
    return j * tiles_n + tile_x;
}

pub const RwPlan = struct { tm: c_int, wm: c_int, wn: c_int, swz: c_int, sd: c_int };

/// rwDown's two stages: 2 x BM rows x (32 sd + 8) bf16.
pub fn rwStageBytes(tm: c_int, wm: c_int, sd: c_int) c_int {
    return 2 * 16 * tm * wm * (32 * sd + 8) * 2;
}

/// A stage inside one stream (H % 32 sd == 0: one row scale a run); at most 512 threads with the inject simdgroups; the
/// two stages within 32 KB.
fn rwTileOk(p: RwPlan, n: c_int, hidden: c_int) bool {
    if (!(p.tm == 1 or p.tm == 2 or p.tm == 4) or !(p.sd == 1 or p.sd == 2 or p.sd == 4) or p.wm < 1 or p.wn < 1) return false;
    const cols = @divTrunc(n, 32);
    if (@rem(cols, p.wn) != 0 or @rem(hidden, 32 * p.sd) != 0) return false;
    return 32 * (p.wm * p.wn + injPerTg(p.wm, @divExact(cols, p.wn))) <= 512 and rwStageBytes(p.tm, p.wm, p.sd) <= 32 * 1024;
}

/// rwDown's tiles: 64 x 160 with one inject simdgroup, 64-deep stages; where they do not fit, 32-deep stages at 2 x 5,
/// 2 x 2, then 2 x 1 column tiles.
pub fn rwPlan(n: c_int, hidden: c_int) ?RwPlan {
    if (n < 32 or @rem(n, 32) != 0) return null;
    const p: RwPlan = .{ .tm = 2, .wm = 2, .wn = 5, .swz = 2, .sd = 2 };
    if (rwTileOk(p, n, hidden)) return p;
    for ([_]c_int{ 5, 2, 1 }) |wn| {
        const f: RwPlan = .{ .tm = 2, .wm = 2, .wn = wn, .swz = 2, .sd = 1 };
        if (rwTileOk(f, n, hidden)) return f;
    }
    return null;
}

// ── dispatches (the wrappers' and the contract's) ──

/// naxDown at x [m, k] . w^T, w [n, k].
pub fn naxDownDispatch(m: c_int, n: c_int, k: c_int) ?Dispatch {
    if (n < 32 or @rem(n, 32) != 0 or k < 32 or @rem(k, 32) != 0) return null;
    const p = downPlan(m, n, k) orelse return null;
    const threads = 32 * p.np * p.wm * p.wn;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, @divExact(n, 32 * p.wn), p.swz, threads), .threads = threads, .tg_bytes = @intCast(if (p.np > 1) splitPartialBytes(p.np, p.tm, p.wm, p.wn) else 4) };
    d.args(&.{ "K", "N", "TM", "WM", "WN", "NP", "PART", "SWZ", "BAR" }, &.{ k, n, p.tm, p.wm, p.wn, p.np, p.part, p.swz, p.bar });
    return d;
}

/// naxUpMix and upMixNorm at act [m, r] . w^T, w [hc hidden, r].
pub fn upDispatch(m: c_int, r: c_int, hc: c_int, hidden: c_int) ?Dispatch {
    if ((hc != 1 and hc != 2 and hc != 4 and hc != 8) or hidden < 32 or @rem(hidden, 32) != 0 or r < 32 or @rem(r, 32) != 0) return null;
    const p = upPlan(hc);
    const threads = 32 * p.wm * hc;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, @divExact(hidden, 32), p.swz, threads), .threads = threads, .t = true, .tg_bytes = @intCast(p.wm * hc * 16 * p.tm * 40 * 2) };
    d.args(&.{ "K", "H", "HC", "TM", "WM", "SWZ", "BAR", "PF" }, &.{ r, hidden, hc, p.tm, p.wm, p.swz, p.bar, p.pf });
    return d;
}

/// fuseDown at x [m, k] . w^T (w [n, k]) and, with `inj`, x . iw (iw [k, hc]).
pub fn fuseDownDispatch(m: c_int, n: c_int, k: c_int, hc: c_int, inj: bool) ?Dispatch {
    if (hc < 1 or hc > 8 or n < 32 or @rem(n, 32) != 0 or k < 32 or @rem(k, 32) != 0) return null;
    const p = fusePlan(m, n, k, hc, inj) orelse return null;
    const threads = 32 * p.np * p.wm * p.wn;
    const parts = if (p.np > 1) splitPartialBytes(p.np, p.tm, p.wm, p.wn) + (if (inj) injPartialBytes(p.np, p.tm, p.wm, hc) else 4) else 8;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, @divExact(n, 32 * p.wn), p.swz, threads), .threads = threads, .tg_bytes = @intCast(parts) };
    d.args(&.{ "K", "N", "HC", "INJ", "TM", "WM", "WN", "NP", "PART", "SWZ", "BAR" }, &.{ k, n, hc, @intFromBool(inj), p.tm, p.wm, p.wn, p.np, p.part, p.swz, p.bar });
    return d;
}

/// normInv over `m` tokens of hc streams of `hidden` (one threadgroup a stream row, four values a thread).
pub fn normInvDispatch(m: c_int, hc: c_int, hidden: c_int, wr: bool) ?Dispatch {
    if (hc < 1 or hc > 8 or hidden < 128 or hidden > 4096 or @mod(hidden, 128) != 0) return null;
    var d: Dispatch = .{ .grid = .{ m * hc * @divTrunc(hidden, 4), 1, 1 }, .threads = @divTrunc(hidden, 4), .t = true, .tg_bytes = 128 };
    d.args(&.{ "H", "HC", "WR" }, &.{ hidden, hc, @intFromBool(wr) });
    return d;
}

/// fuseDownNorm over hc streams of `hidden` (K = hc hidden): one chain over K, at any row count (its wrapper serves it
/// only where `oneChain`).
pub fn fuseDownNormDispatch(m: c_int, n: c_int, hc: c_int, hidden: c_int, inj: bool) ?Dispatch {
    if (hc < 1 or hc > 8 or hidden < 32 or @rem(hidden, 32) != 0) return null;
    const k = hc * hidden;
    const p = normPlan(n) orelse return null;
    const threads = 32 * p.wm * p.wn;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, @divExact(n, 32 * p.wn), p.swz, threads), .threads = threads, .tg_bytes = @intCast(2 * 16 * p.tm * p.wm * 40 * 2) };
    d.args(&.{ "K", "N", "H", "HC", "INJ", "TM", "WM", "WN", "SWZ" }, &.{ k, n, hidden, hc, @intFromBool(inj), p.tm, p.wm, p.wn, p.swz });
    return d;
}

/// readDown: one chain over K for the down and the inject products, the inject chains in simdgroups of their own; at any
/// row count (its wrapper serves it only where `oneChain`).
pub fn readDownDispatch(m: c_int, n: c_int, k: c_int, hc: c_int) ?Dispatch {
    if (hc < 1 or hc > 8 or n < 32 or @rem(n, 32) != 0 or k < 32 or @rem(k, 32) != 0) return null;
    const p = chainPlan(n, k);
    const tiles_n = @divExact(n, 32 * p.wn);
    const threads = 32 * (p.wm * p.wn + injPerTg(p.wm, tiles_n));
    if (threads > 512) return null;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, tiles_n, p.swz, threads), .threads = threads };
    d.args(&.{ "K", "N", "HC", "TM", "WM", "WN", "SWZ", "BAR" }, &.{ k, n, hc, p.tm, p.wm, p.wn, p.swz, p.bar });
    return d;
}

/// readUpMix (`rw`: rwUpMix, which also takes PF) at act [m, r] . w^T, w [hc hidden, r].
pub fn regDispatch(m: c_int, r: c_int, hc: c_int, hidden: c_int, rw: bool) ?Dispatch {
    if ((hc != 1 and hc != 2 and hc != 4 and hc != 8) or hidden < 32 or @rem(hidden, 32) != 0 or r < 32 or @rem(r, 32) != 0) return null;
    const p = regPlan(hc, hidden) orelse return null;
    const threads = 32 * p.wm * p.wn;
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, @divExact(hidden, 32 * p.wn), p.swz, threads), .threads = threads, .t = true };
    d.args(&.{ "K", "H", "HC", "TM", "WM", "WN", "SWZ", "BAR" }, &.{ r, hidden, hc, p.tm, p.wm, p.wn, p.swz, p.bar });
    if (rw) d.args(&.{"PF"}, &.{0});
    return d;
}

/// rwDown over hc streams of `hidden` (K = hc hidden): one chain over K, at any row count (its wrapper serves it only
/// where `oneChain`).
pub fn rwDownDispatch(m: c_int, n: c_int, hc: c_int, hidden: c_int) ?Dispatch {
    if (hc < 1 or hc > 8 or hidden < 32 or @rem(hidden, 32) != 0) return null;
    const k = hc * hidden;
    const p = rwPlan(n, hidden) orelse return null;
    const tiles_n = @divExact(n, 32 * p.wn);
    const threads = 32 * (p.wm * p.wn + injPerTg(p.wm, tiles_n));
    var d: Dispatch = .{ .grid = tiled(m, 16 * p.tm * p.wm, tiles_n, p.swz, threads), .threads = threads, .tg_bytes = @intCast(rwStageBytes(p.tm, p.wm, p.sd)) };
    d.args(&.{ "K", "N", "H", "HC", "TM", "WM", "WN", "SWZ", "SD" }, &.{ k, n, hidden, hc, p.tm, p.wm, p.wn, p.swz, p.sd });
    return d;
}

// ── which read takes which launches ──

/// A GPU whose MLX matmuls are the NAX kernels these launches replay.
fn naxGpu(s: mlx.mlx_stream) bool {
    return !disabled_for_test and mlx.streamIsGpu(s) and device.naxAvailable();
}

/// Whether a read's down and up take `naxDown` and `naxUpMix`: the prefill geometry, dense bf16 hc rows and bf16
/// normalized streams. Asked once a read, so both launches agree.
pub fn naxOn(s: mlx.mlx_stream, batch: c_int, seq: c_int, normalized: mlx.mlx_array, down_rows: mlx.mlx_array, up_rows: mlx.mlx_array) bool {
    if (down_rows.ctx == null or up_rows.ctx == null or !hp.geometry(batch, seq)) return false;
    return mlx.mlx_array_dtype(normalized) == .bfloat16 and naxGpu(s);
}

/// Whether a read's down, silu, inject and gate take `readDown` / `fuseDown`. The caller also requires the chain's inject
/// to be hp.norm's matmul of `inject_flat`, or no inject (the mixer's read).
pub fn fuseOn(s: mlx.mlx_stream, batch: c_int, seq: c_int, normalized: mlx.mlx_array, down_rows: mlx.mlx_array) bool {
    if (down_rows.ctx == null or !hp.geometry(batch, seq)) return false;
    return mlx.mlx_array_dtype(normalized) == .bfloat16 and naxGpu(s);
}

/// Whether a read's up + mix takes `readUpMix`.
pub fn readUpOn(s: mlx.mlx_stream, batch: c_int, seq: c_int, normalized: mlx.mlx_array, up_rows: mlx.mlx_array) bool {
    if (up_rows.ctx == null or !hp.geometry(batch, seq)) return false;
    return mlx.mlx_array_dtype(normalized) == .bfloat16 and naxGpu(s);
}

/// Whether a read takes the norm-folded launches (`normInv`, `rwDown` / `fuseDownNorm`, `rwUpMix` / `upMixNorm`): a bf16
/// stream, dense bf16 down and up rows and inject weight, and a shape where MLX runs one chain over K for the down and the
/// inject products (the staged loader has no split mode). Asked once a read, before the norm.
pub fn fuseNormOn(s: mlx.mlx_stream, batch: c_int, seq: c_int, stream: mlx.mlx_array, down_rows: mlx.mlx_array, up_rows: mlx.mlx_array, iw: mlx.mlx_array, hc: c_int, hidden: c_int) bool {
    if (down_rows.ctx == null or up_rows.ctx == null or iw.ctx == null or !hp.geometry(batch, seq)) return false;
    if (mlx.mlx_array_dtype(stream) != .bfloat16 or mlx.mlx_array_dtype(iw) != .bfloat16 or !naxGpu(s)) return false;
    if ((hc != 1 and hc != 2 and hc != 4 and hc != 8) or @rem(hidden, 32) != 0) return false;
    const ds = mlx.getShape(down_rows);
    const us = mlx.getShape(up_rows);
    if (ds.len != 2 or ds[1] * 2 != hc * hidden or us.len != 2 or us[0] != hc * hidden or us[1] * 2 != ds[0]) return false;
    if (mlx.mlx_array_dtype(down_rows) != .uint32 or mlx.mlx_array_dtype(up_rows) != .uint32) return false;
    return oneChain(batch * seq, ds[0], hc * hidden, hc, true) and normPlan(ds[0]) != null;
}

// ── the read's tables and outputs ──

/// The chain's silu and inject gate of every bf16 pattern ([65536] bf16 each, indexed by the input's bits), built from
/// the trunk's own closures; cached per (silu compiled, gate compiled) for the process (2 x 128 KB a pair).
pub const FuseTables = struct { silu: mlx.mlx_array, gate: mlx.mlx_array };
pub var fuse_tables: [4]?FuseTables = @splat(null);

fn tablesOk(t: FuseTables) bool {
    for ([_]mlx.mlx_array{ t.silu, t.gate }) |a| {
        if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16 or mlx.mlx_array_size(a) != 65536) return false;
    }
    return true;
}

/// The fused down's outputs: act [B, S, R] (the chain's `act`) and inj [B, S, HC, 1] (its `inj`; null-ctx without an inject).
pub const FuseOut = struct {
    act: mlx.mlx_array,
    inj: mlx.mlx_array,

    pub fn deinit(self: FuseOut) void {
        _ = mlx.mlx_array_free(self.act);
        if (self.inj.ctx != null) _ = mlx.mlx_array_free(self.inj);
    }
};

/// normInv's outputs: the stream the read continues with (the written stream with a pending write, else a reference to
/// `x`) and the norm's f32 row scales [B S HC] (row = token HC + stream).
pub const NormInv = struct {
    stream: mlx.mlx_array,
    invs: mlx.mlx_array,

    pub fn deinit(self: NormInv) void {
        _ = mlx.mlx_array_free(self.stream);
        _ = mlx.mlx_array_free(self.invs);
    }
};

fn sizeIs(a: mlx.mlx_array, n: c_int) bool {
    return mlx.mlx_array_size(a) == @as(usize, @intCast(n));
}

// ── the launches ──

/// The read's down product: y [B, S, R] = x [B, S, K] . W^T, W = `down_rows` (u32 words of the [R, K] bf16 rows), MLX's
/// NAX matmul of `x` and W's transposed view bit for bit.
pub fn naxDown(s: mlx.mlx_stream, x: mlx.mlx_array, down_rows: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s) or !hp.geometry(batch, seq)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(down_rows) != .uint32) return null;
    const ws = mlx.getShape(down_rows);
    if (ws.len != 2) return null;
    const n = ws[0];
    const k = ws[1] * 2;
    const m = batch * seq;
    if (!sizeIs(x, m * k)) return null;
    const d = naxDownDispatch(m, n, k) orelse return null;
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    var y: [1]mlx.mlx_array = undefined;
    try launch(s, .nax_down, d, &.{.{ .shape = &.{ batch, seq, n } }}, &.{ x, down_rows, rows }, &y, seq);
    return y[0];
}

/// The read's up product with the mix as its epilogue: mixed [B, S, H] from act [B, S, R], W = `up_rows` (u32 words of
/// the [HC H, R] bf16 rows) and the normalized streams [B, S, HC, H]; bit for bit `mix(act . W^T, normalized)` with MLX's
/// NAX matmul, the [B, S, HC H] product never written.
pub fn naxUpMix(s: mlx.mlx_stream, act: mlx.mlx_array, up_rows: mlx.mlx_array, normalized: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    return upMix(s, .nax_upmix, act, up_rows, normalized, null, batch, seq);
}

/// `naxUpMix` and its register form (`readUpMix`), or with `norm` their normalized values from the stream (`upMixNorm`,
/// `rwUpMix`).
fn upMix(s: mlx.mlx_stream, k: Kernel, act: mlx.mlx_array, up_rows: mlx.mlx_array, normalized: mlx.mlx_array, norm: ?[3]mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s) or !hp.geometry(batch, seq)) return null;
    if (mlx.mlx_array_dtype(act) != .bfloat16 or mlx.mlx_array_dtype(up_rows) != .uint32) return null;
    // normalized [B, S, HC, H], or the stream xs [B, S, HC H], the row scales invs [B S HC] and the [HC, H] weight nw
    var hc: c_int = 0;
    var hidden: c_int = 0;
    if (norm) |nm| {
        if (mlx.mlx_array_dtype(nm[0]) != .bfloat16 or mlx.mlx_array_dtype(nm[1]) != .float32 or mlx.mlx_array_dtype(nm[2]) != .bfloat16) return null;
        const nsh = mlx.getShape(nm[2]);
        if (nsh.len != 2) return null;
        hc = nsh[0];
        hidden = nsh[1];
    } else {
        if (mlx.mlx_array_dtype(normalized) != .bfloat16) return null;
        const shape4 = mlx.getShape(normalized);
        if (shape4.len != 4) return null;
        hc = shape4[2];
        hidden = shape4[3];
    }
    const ws = mlx.getShape(up_rows);
    if (ws.len != 2 or ws[0] != hc * hidden) return null;
    const r = ws[1] * 2;
    const m = batch * seq;
    if (!sizeIs(act, m * r)) return null;
    if (norm) |nm| {
        if (!sizeIs(nm[0], m * hc * hidden) or !sizeIs(nm[1], m * hc)) return null;
    } else if (!sizeIs(normalized, m * hc * hidden)) return null;
    const d = switch (k) {
        .nax_upmix, .fuse_upmix => upDispatch(m, r, hc, hidden),
        .read_upmix => regDispatch(m, r, hc, hidden, false),
        .rw_upmix => regDispatch(m, r, hc, hidden, true),
        else => unreachable,
    } orelse return null;
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    const tab = try hp.sigmoidTable(s);
    var out: [1]mlx.mlx_array = undefined;
    const shape = [_]c_int{ batch, seq, hidden };
    if (norm) |nm| {
        try launch(s, k, d, &.{.{ .shape = &shape }}, &.{ act, up_rows, nm[0], nm[1], nm[2], tab, rows }, &out, seq);
    } else try launch(s, k, d, &.{.{ .shape = &shape }}, &.{ act, up_rows, normalized, tab, rows }, &out, seq);
    return out[0];
}

/// The read's down product, silu, inject product and gate as one launch: act [B, S, R] = the chain's silu of naxDown's
/// product, inj [B, S, HC, 1] = the chain's gate of MLX's matmul of `x` and `iw` (the row-major [K, HC] bf16
/// `inject_flat`; null-ctx: no inject, the mixer's read). `tabs`: the chain's silu and gate tables.
pub fn fuseDown(s: mlx.mlx_stream, x: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int, hc: c_int) !?FuseOut {
    return down(s, .fuse_down, x, down_rows, iw, tabs, batch, seq, hc);
}

/// fuseDown with the inject chain in simdgroups of its own: fuseDown's act and inj bit for bit, same operands. Null where
/// MLX splits K (the down or the inject product), without an inject, or past 512 threads: fuseDown then serves.
pub fn readDown(s: mlx.mlx_stream, x: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int, hc: c_int) !?FuseOut {
    if (iw.ctx == null) return null;
    return down(s, .read_down, x, down_rows, iw, tabs, batch, seq, hc);
}

fn down(s: mlx.mlx_stream, k: Kernel, x: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int, hc: c_int) !?FuseOut {
    if (disabled_for_test or !mlx.streamIsGpu(s) or !hp.geometry(batch, seq) or hc < 1 or hc > 8 or !tablesOk(tabs)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(down_rows) != .uint32) return null;
    const ws = mlx.getShape(down_rows);
    if (ws.len != 2) return null;
    const n = ws[0];
    const kk = ws[1] * 2;
    const m = batch * seq;
    if (!sizeIs(x, m * kk)) return null;
    const inj = iw.ctx != null;
    if (inj) {
        const ish = mlx.getShape(iw);
        if (mlx.mlx_array_dtype(iw) != .bfloat16 or ish.len != 2 or ish[0] != kk or ish[1] != hc) return null;
    }
    if (k == .read_down and !oneChain(m, n, kk, hc, true)) return null;
    const d = (if (k == .read_down) readDownDispatch(m, n, kk, hc) else fuseDownDispatch(m, n, kk, hc, inj)) orelse return null;
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    // Without an inject the kernel never reads `iw`; the silu table stands in (a device bf16 array).
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, k, d, &.{ .{ .shape = &.{ batch, seq, n } }, .{ .shape = if (inj) &.{ batch, seq, hc, 1 } else &.{1} } }, &.{ x, down_rows, if (inj) iw else tabs.silu, tabs.silu, tabs.gate, rows }, &res, seq);
    if (!inj) {
        _ = mlx.mlx_array_free(res[1]);
        res[1] = .{ .ctx = null };
    }
    return .{ .act = res[0], .inj = res[1] };
}

/// `hp.norm` without the normalized rows: the pending write applied and stored as `norm` stores it, and the norm's own
/// f32 row scales. `w` is the [HC, H] norm weight (its shape only: the consumers apply it).
pub fn normInv(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, pending: ?hp.Pending) !?NormInv {
    if (disabled_for_test or !mlx.streamIsGpu(s) or !hp.geometry(batch, seq)) return null;
    const ws = mlx.getShape(w);
    if (ws.len != 2) return null;
    const hc = ws[0];
    const hidden = ws[1];
    const width = hc * hidden;
    if (!sizeIs(x, batch * seq * width)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return null;
    if (pending) |p| if (mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16) return null;
    const d = normInvDispatch(batch * seq, hc, hidden, pending != null) orelse return null;
    const xsh = if (pending != null) [_]c_int{ batch, seq, width } else [_]c_int{ 1, 1, 1 };
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, .norm_inv, d, &.{ .{ .shape = &xsh }, .{ .shape = &.{batch * seq * hc}, .dt = .float32 } }, &.{ x, eps, if (pending) |p| p.out else x, if (pending) |p| p.inj else x }, &res, seq);
    if (pending == null) {
        _ = mlx.mlx_array_free(res[0]);
        res[0] = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&res[0], x)) catch |err| {
            _ = mlx.mlx_array_free(res[0]);
            _ = mlx.mlx_array_free(res[1]);
            return err;
        };
    }
    return .{ .stream = res[0], .invs = res[1] };
}

/// fuseDown with the norm in its A loader: fuseDown's act and inj from the written stream `xs` [B, S, HC H], the row
/// scales `invs` (normInv) and the [HC, H] norm weight `nw`, the normalized rows never written. Null where MLX splits K.
pub fn fuseDownNorm(s: mlx.mlx_stream, xs: mlx.mlx_array, invs: mlx.mlx_array, nw: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int) !?FuseOut {
    return downNorm(s, .fuse_down_norm, xs, invs, nw, down_rows, iw, tabs, batch, seq);
}

/// fuseDownNorm with the inject chain in simdgroups of its own and two-step stages: fuseDownNorm's act and inj bit for
/// bit, same operands. Null where fuseDownNorm declines or without an inject: fuseDownNorm then serves.
pub fn rwDown(s: mlx.mlx_stream, xs: mlx.mlx_array, invs: mlx.mlx_array, nw: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int) !?FuseOut {
    if (iw.ctx == null) return null;
    return downNorm(s, .rw_down, xs, invs, nw, down_rows, iw, tabs, batch, seq);
}

fn downNorm(s: mlx.mlx_stream, k: Kernel, xs: mlx.mlx_array, invs: mlx.mlx_array, nw: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: FuseTables, batch: c_int, seq: c_int) !?FuseOut {
    if (disabled_for_test or !mlx.streamIsGpu(s) or !hp.geometry(batch, seq) or !tablesOk(tabs)) return null;
    if (mlx.mlx_array_dtype(xs) != .bfloat16 or mlx.mlx_array_dtype(nw) != .bfloat16 or mlx.mlx_array_dtype(invs) != .float32 or mlx.mlx_array_dtype(down_rows) != .uint32) return null;
    const nsh = mlx.getShape(nw);
    const ws = mlx.getShape(down_rows);
    if (nsh.len != 2 or ws.len != 2) return null;
    const hc = nsh[0];
    const hidden = nsh[1];
    const kk = hc * hidden;
    const n = ws[0];
    const m = batch * seq;
    if (ws[1] * 2 != kk or !sizeIs(xs, m * kk) or !sizeIs(invs, m * hc)) return null;
    const inj = iw.ctx != null;
    if (inj) {
        const ish = mlx.getShape(iw);
        if (mlx.mlx_array_dtype(iw) != .bfloat16 or ish.len != 2 or ish[0] != kk or ish[1] != hc) return null;
    }
    if (!oneChain(m, n, kk, hc, inj)) return null;
    const d = (if (k == .rw_down) rwDownDispatch(m, n, hc, hidden) else fuseDownNormDispatch(m, n, hc, hidden, inj)) orelse return null;
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, k, d, &.{ .{ .shape = &.{ batch, seq, n } }, .{ .shape = if (inj) &.{ batch, seq, hc, 1 } else &.{1} } }, &.{ xs, invs, nw, down_rows, if (inj) iw else tabs.silu, tabs.silu, tabs.gate, rows }, &res, seq);
    if (!inj) {
        _ = mlx.mlx_array_free(res[1]);
        res[1] = .{ .ctx = null };
    }
    return .{ .act = res[0], .inj = res[1] };
}

/// naxUpMix with the normalized values recomputed in its mix epilogue from the written stream `xs`, the row scales
/// `invs` and the [HC, H] norm weight `nw`: naxUpMix's mixed bit for bit, the normalized rows never read.
pub fn upMixNorm(s: mlx.mlx_stream, act: mlx.mlx_array, up_rows: mlx.mlx_array, xs: mlx.mlx_array, invs: mlx.mlx_array, nw: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    return upMix(s, .fuse_upmix, act, up_rows, .{ .ctx = null }, .{ xs, invs, nw }, batch, seq);
}

/// naxUpMix with every stream's up tile in one simdgroup's registers: naxUpMix's mixed bit for bit, same operands.
pub fn readUpMix(s: mlx.mlx_stream, act: mlx.mlx_array, up_rows: mlx.mlx_array, normalized: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    return upMix(s, .read_upmix, act, up_rows, normalized, null, batch, seq);
}

/// upMixNorm's mix in readUpMix's register form: upMixNorm's mixed bit for bit, same operands.
pub fn rwUpMix(s: mlx.mlx_stream, act: mlx.mlx_array, up_rows: mlx.mlx_array, xs: mlx.mlx_array, invs: mlx.mlx_array, nw: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    return upMix(s, .rw_upmix, act, up_rows, .{ .ctx = null }, .{ xs, invs, nw }, batch, seq);
}

// ── tests ──

const testing = std.testing;
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;

fn tOp(comptime f: anytype, args: anytype) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(@call(.auto, f, .{&o} ++ args));
    return o;
}

/// The chain's silu (sigmoid, then multiply) and inject gate (2 sigmoid) as uncompiled ops.
fn tSilu(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sig = try tOp(mlx.mlx_sigmoid, .{ x, s });
    defer _ = mlx.mlx_array_free(sig);
    return tOp(mlx.mlx_multiply, .{ x, sig, s });
}

fn tGate(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sig = try tOp(mlx.mlx_sigmoid, .{ x, s });
    defer _ = mlx.mlx_array_free(sig);
    const two_f = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two_f);
    const two = try tOp(mlx.mlx_astype, .{ two_f, mlx.mlx_array_dtype(x), s });
    defer _ = mlx.mlx_array_free(two);
    return tOp(mlx.mlx_multiply, .{ sig, two, s });
}

/// Every bf16 pattern through tSilu and tGate: the launches only index a table, so any function checks them.
fn testTables(s: mlx.mlx_stream) !FuseTables {
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(&bits, &[_]c_int{65536}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const silu = try tSilu(x, s);
    errdefer _ = mlx.mlx_array_free(silu);
    const gate = try tGate(x, s);
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_array_eval(silu));
    try mlx.check(mlx.mlx_array_eval(gate));
    return .{ .silu = silu, .gate = gate };
}

fn freeTables(t: FuseTables) void {
    _ = mlx.mlx_array_free(t.silu);
    _ = mlx.mlx_array_free(t.gate);
}

/// Same dtype and the same 16-bit patterns in element order (array_equal alone takes -0 for +0), any strides.
fn bitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    if (mlx.mlx_array_size(a) != mlx.mlx_array_size(b) or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b)) return false;
    const n = [_]c_int{@intCast(mlx.mlx_array_size(a))};
    var views: [2]mlx.mlx_array = undefined;
    for ([_]mlx.mlx_array{ a, b }, &views) |x, *v| {
        const c = try tOp(mlx.mlx_contiguous, .{ x, false, s });
        defer _ = mlx.mlx_array_free(c);
        const flat = try tOp(mlx.mlx_reshape, .{ c, &n, n.len, s });
        defer _ = mlx.mlx_array_free(flat);
        v.* = try tOp(mlx.mlx_view, .{ flat, if (mlx.mlx_array_dtype(a) == .float32) mlx.mlx_dtype.uint32 else mlx.mlx_dtype.uint16, s });
    }
    defer for (views) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const e = try tOp(mlx.mlx_array_equal, .{ views[0], views[1], false, s });
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_array_eval(e));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, e));
    return same;
}

fn freeAll(xs: []const mlx.mlx_array) void {
    for (xs) |x| _ = mlx.mlx_array_free(x);
}

/// Dense bf16 hc weights as the loader binds them: the down rows [R, K] (u32 words for the launches, a transposed view
/// for the chain), the up rows [K, R] likewise, the inject weight's contiguous [K, HC] copy and the [HC, H] norm weight.
const TestWeights = struct {
    wd_t: mlx.mlx_array,
    wd_rows: mlx.mlx_array,
    wu_t: mlx.mlx_array,
    wu_rows: mlx.mlx_array,
    iw: mlx.mlx_array,
    nw: mlx.mlx_array,
    own: [4]mlx.mlx_array,

    fn init(rnd: std.Random, s: mlx.mlx_stream) !TestWeights {
        const wd = try testRandUniformBf16(rnd, &.{ R, K }, -0.04, 0.04, s);
        errdefer _ = mlx.mlx_array_free(wd);
        const wu = try testRandUniformBf16(rnd, &.{ K, R }, -0.1, 0.1, s);
        errdefer _ = mlx.mlx_array_free(wu);
        const wi = try testRandUniformBf16(rnd, &.{ HC, K }, -0.05, 0.05, s);
        errdefer _ = mlx.mlx_array_free(wi);
        const nw = try testRandUniformBf16(rnd, &.{ HC, HID }, 0.5, 1.5, s);
        errdefer _ = mlx.mlx_array_free(nw);
        const wi_t = try tOp(mlx.mlx_transpose, .{ wi, s });
        defer _ = mlx.mlx_array_free(wi_t);
        var t: TestWeights = .{
            .wd_t = try tOp(mlx.mlx_transpose, .{ wd, s }),
            .wd_rows = try tOp(mlx.mlx_view, .{ wd, mlx.mlx_dtype.uint32, s }),
            .wu_t = try tOp(mlx.mlx_transpose, .{ wu, s }),
            .wu_rows = try tOp(mlx.mlx_view, .{ wu, mlx.mlx_dtype.uint32, s }),
            .iw = try tOp(mlx.mlx_contiguous, .{ wi_t, false, s }),
            .nw = nw,
            .own = .{ wd, wu, wi, .{ .ctx = null } },
        };
        for ([_]mlx.mlx_array{ t.wd_rows, t.wu_rows, t.iw, t.nw }) |a| try mlx.check(mlx.mlx_array_eval(a));
        t.own[3] = .{ .ctx = null };
        return t;
    }

    fn deinit(t: *TestWeights) void {
        freeAll(&.{ t.wd_t, t.wd_rows, t.wu_t, t.wu_rows, t.iw, t.nw });
        for (t.own[0..3]) |a| _ = mlx.mlx_array_free(a);
    }
};

const HC: c_int = 4;
const HID: c_int = 2560;
const K: c_int = HC * HID;
const R: c_int = 320;

fn nax() bool {
    return !mlx.noGpuBackend() and !mlx.virtualGpu() and device.naxAvailable();
}
