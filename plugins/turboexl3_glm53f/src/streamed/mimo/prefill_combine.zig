//! Lever 1 of prefill-attr-20261008: the prompt pass's routed combine as ONE text per routed layer.
//!
//! The stock route (`Route.stock`, math.zig `combineParts`, the control): per gather tile of 1,024 tokens, up to 24
//! takes + a concatenate + a permuting take build the tile's [count, 8, H] bf16 routed rows (word copies), evaluated;
//! per reduction tile of 256 tokens (`reduceGraph`) f32(row) * f32 w, MLX's axis-1 sum, bf16 once, evaluated and
//! synchronized; then the tiles' outputs concatenated [nt, H], evaluated. 16 + 64 + 1 evals at 16K tokens.
//!
//! The fused route (`Route.fused`): `mimo_prefill_combine` reads each token's 8 routed rows in place from the wide
//! call's <= 24 sources (`deepseek_v41_experts.mergeJoinless`) through the (source, row) table, and computes the stock
//! reduction tile's Metal arithmetic word for word: per column, each product `f32(row_k) * w_k` rounded to f32 (no FMA
//! contraction), added to the +0 init, folded k = 0..7 in router order as `product_k + total`, rounded to bf16 once.
//! One launch and one eval per routed layer; no gathered rows, no f32 transients, no output concatenate.
//!
//! Exactness: the stock reduction is MLX's `col_reduce_small` (reduce.cpp `strided_reduce_general_dispatch`: reduction
//! size 8 x 1 < 32) with threadgroup y = 8: thread k computes `op(p_k, 0)`, lid.y 0 then folds
//! `op(shared_k, total)` for k = 1..7 (reduce_col.h) -- the order this text and `combineRef` state. The residual add
//! (module.zig `next = residual + mlp`) is its own bf16 op after the combine and is not folded in; MiMo has no shared
//! expert (config.zig refuses `n_shared_experts`), so there is no shared term.
const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../sdk_ext/metal_launch.zig");
const ops = @import("../ops.zig");
const bank_mod = @import("bank.zig");

const T = mlx.mlx_array;

/// The prompt pass's routed combine, fixed at construction (`settings.Config.prefill_combine`): `stock` (the control),
/// `fused` (the text over <= 24 bound sources, the call's smallest outputs merged into one by the shared scheduler), or
/// P7 `inplace` (the SAME text reading every source of the call in place through a table of their GPU addresses: no
/// merge concatenate, no merged buffer); any other name is refused (`error.MimoPrefillCombine`).
pub const Route = enum { stock, fused, inplace, invalid };

/// P7 `inplace`: the most sources a call's combine reads in place (the scheduler's source index is a byte); a call with
/// more falls back to the merge at `sources` (the stock rule) -- a MiMo prompt call has at most ~140 tiles.
pub const inplace_sources: usize = 255;

/// P7 `inplace`: the fewest entries the table hands MLX. MLX 64ea011c (backend/common/metal_kernel.cpp
/// `max_constant_array_size`) declares a custom-kernel input of fewer than 8 elements in the `constant` address space,
/// which `mimo_pc_pick`'s table overload (`const device uint64_t*`) does not take: the text failed to compile ("no
/// matching function for call to 'mimo_pc_pick'") for every call of 1..7 sources. The table is padded to this many
/// entries, so every call is the one `device` form (one compiled library for every source count).
pub const table_min_entries: usize = 8;

comptime {
    if (inplace_sources < table_min_entries) @compileError("mimo prefill combine: the table bound must hold the padding");
}

/// The table of `n` sources' addresses (`addr[0..n]`) as MLX is handed it: padded to `table_min_entries` with source
/// 0's address. The padding is never read: every `loc` source index is < `n`.
fn paddedTable(addr: *[inplace_sources]u64, n: usize) []const u64 {
    std.debug.assert(n > 0 and n <= inplace_sources);
    const len = @max(n, table_min_entries);
    for (addr[n..len]) |*x| x.* = addr[0];
    return addr[0..len];
}

/// MiMo's and GLM-5.3-Flash's expert shape (hidden 4096, inter 2048, top-8).
pub const hidden: u32 = 4096;
const inter: u32 = 2048;
pub const top_k: u32 = 8;
/// The text's source slots (`deepseek_v41_experts.joinless_sources`); unused slots alias source 0.
pub const sources: usize = @import("../experts.zig").joinless_sources;
/// Columns per thread (one 8-byte bf16 load per routed row), threads per threadgroup: 256 x 4 = 1,024 columns of one
/// token per threadgroup, H / 1,024 = 4 threadgroups per token.
pub const cols_per_thread: u32 = 4;
pub const threads: u32 = 256;

comptime {
    if (hidden % (cols_per_thread * threads) != 0) @compileError("mimo prefill combine: H must be whole threadgroups of columns");
    if (sources + 2 + 1 > 31) @compileError("mimo prefill combine: the text's buffers exceed Metal's 31 argument slots");
}

/// The texts (pinned by `kernels/manifest-prefill-combine.json`; `manifest_sha256` pins the manifest).
pub const texts = struct {
    pub const header = @embedFile("kernels/header_mimo_prefill_combine.metal");
    pub const combine = @embedFile("kernels/mimo_prefill_combine.metal");
    pub const manifest = @embedFile("kernels/manifest-prefill-combine.json");
    /// P7 `inplace`: the same text with its 24 bound sources replaced by the table (`mimo_pc_pick`'s table overload).
    pub const combine_inplace = tableText(combine);
};

/// The bound sources' argument list as the text passes it to `mimo_pc_pick`.
const bound_args = blk: {
    var t: []const u8 = "s0";
    for (1..sources) |i| t = t ++ std.fmt.comptimePrint(", s{d}", .{i});
    break :blk t;
};

/// `text` with every `mimo_pc_pick` argument list of the 24 bound sources replaced by `tbl` (comptime).
fn tableText(comptime text: []const u8) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        var out: []const u8 = "";
        var rest: []const u8 = text;
        while (std.mem.indexOf(u8, rest, bound_args)) |i| {
            out = out ++ rest[0..i] ++ "tbl";
            rest = rest[i + bound_args.len ..];
        }
        const full = out ++ rest ++ [_]u8{0};
        return full[0 .. full.len - 1 :0];
    }
}

/// The table form's buffers: the sources' GPU addresses, then the text's own.
const inplace_input_names = [_][*:0]const u8{ "tbl", "loc", "w" };

/// P7 `inplace` (csrc/mimo_gpu_address.inc): each evaluated source's GPU address (its MTLBuffer's plus its offset).
extern fn mimo_gpu_addresses(arrs: [*]const T, n: usize, out: [*]u64) c_int;
extern fn dsv41_alloc_last_error() [*:0]const u8;
pub const manifest_sha256 = "ed3b2e1af2a2803791f337adc73605b0d08f0e733e298da86ef5a90ae323b3f3";

const input_names = blk: {
    var names: [sources + 2][*:0]const u8 = undefined;
    for (0..sources) |i| names[i] = std.fmt.comptimePrint("s{d}", .{i});
    names[sources] = "loc";
    names[sources + 1] = "w";
    break :blk names;
};

/// The fused route's text, built once at construction (no device work: it compiles at its first dispatch; the
/// byte-equality self-check is the pre-ship gate's `prefill_combine` family).
pub const Kernel = struct {
    k: mlx.mlx_fast_metal_kernel = .{ .ctx = null },
    /// P7 `inplace`: the table form of the same text (`initInplace`).
    k_inplace: mlx.mlx_fast_metal_kernel = .{ .ctx = null },

    /// Construction boundary: the routed geometry the text is instantiated for (any bank's geometry: its `hidden_size`).
    pub fn init(geometry: anytype) !Kernel {
        if (geometry.hidden_size != hidden) return error.MimoPrefillCombineGeometry;
        return .{ .k = try newKernel("mimo_prefill_combine", &input_names, texts.combine) };
    }

    /// P7 `inplace`: the text's table form, built once at construction (no device work: it compiles at its first
    /// dispatch; the byte-equality self-check is the pre-ship family `prefill_combine`).
    pub fn initInplace(geometry: anytype) !Kernel {
        if (geometry.hidden_size != hidden) return error.MimoPrefillCombineGeometry;
        return .{ .k_inplace = try newKernel("mimo_prefill_combine_inplace", &inplace_input_names, texts.combine_inplace) };
    }

    fn newKernel(name: [*:0]const u8, names: []const [*:0]const u8, text: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
        const outs = [_][*:0]const u8{"y"};
        const in_vec = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mtl.mlx_fast_metal_kernel_new(name, in_vec, out_vec, text, texts.header, true, false);
        if (k.ctx == null) return error.MimoPrefillCombineKernel;
        return k;
    }

    pub fn deinit(self: *Kernel) void {
        if (self.k.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(self.k);
        if (self.k_inplace.ctx != null) _ = mtl.mlx_fast_metal_kernel_free(self.k_inplace);
        self.* = .{};
    }

    /// P7 `inplace`: `outs` the wide call's evaluated bf16 sources [r_i, H] (any number up to `inplace_sources`, none
    /// merged), `loc` int32 [nt, 8, 2] each routed position's (source, row), `weights` f32 [nt, 8] -> bf16 [nt, H],
    /// unevaluated. The sources are read through `tbl` (their GPU addresses): the caller keeps them alive until the
    /// output is evaluated (`Source.forward`: `releaseParts` after it).
    pub fn callInplace(self: *const Kernel, g: *ops.MlxOps, outs: []const T, loc: T, weights: T, nt: u32) !T {
        std.debug.assert(outs.len > 0 and outs.len <= inplace_sources);
        var addr: [inplace_sources]u64 = undefined;
        if (mimo_gpu_addresses(outs.ptr, outs.len, &addr) != 0) {
            std.log.err("MiMo prefill_combine=inplace: {s}", .{dsv41_alloc_last_error()});
            return error.MimoPrefillCombineSource;
        }
        const table = paddedTable(&addr, outs.len);
        const tbl = try g.hostArray(std.mem.sliceAsBytes(table), &.{@intCast(table.len)}, .uint64);
        const in = [_]T{ tbl, loc, weights };
        return self.apply(g, self.k_inplace, &in, nt);
    }

    fn apply(_: *const Kernel, g: *ops.MlxOps, k: mlx.mlx_fast_metal_kernel, in: []const T, nt: u32) !T {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const shape = [_]c_int{ @intCast(nt), @intCast(hidden) };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(nt * (hidden / cols_per_thread)), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", @intCast(top_k)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", @intCast(hidden)));
        const in_vec = mlx.mlx_vector_array_new_data(in.ptr, in.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mtl.mlx_fast_metal_kernel_apply(&outputs, k, in_vec, cfg, g.s));
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0)) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        return g.adopt(out);
    }

    /// `outs`: the wide call's evaluated bf16 sources [r_i, H] (1..24), `loc` int32 [nt, 8, 2] each routed position's
    /// (source, row), `weights` f32 [nt, 8] -> bf16 [nt, H], unevaluated. The launch's grid follows the call's logical
    /// M (`nt`, the one runtime-varying value): nt x H / 4 threads.
    pub fn call(self: *const Kernel, g: *ops.MlxOps, outs: []const T, loc: T, weights: T, nt: u32) !T {
        std.debug.assert(outs.len > 0 and outs.len <= sources);
        var in: [sources + 2]T = undefined;
        for (in[0..sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
        in[sources] = loc;
        in[sources + 1] = weights;
        return self.apply(g, self.k, &in, nt);
    }
};

// ── Host reference of the text's arithmetic (CPU proofs; the device self-check compares the text) ──

// ── Fixtures: real-shaped wide-call sources (CPU proofs and the device self-check) ──

/// A wide call's combine inputs: `outs` (KEPT bf16 [r_i, H] sources), `loc` (host + array), `weights` (host + array);
/// `bits` the sources' host words when `keep_host` (the reference reads them).
const Fixture = struct {
    outs: std.ArrayList(T) = .empty,
    bits: std.ArrayList([]const u16) = .empty,
    loc: []i32 = &.{},
    w: []f32 = &.{},
    loc_arr: T = .{},
    w_arr: T = .{},

    fn deinit(f: *Fixture, a: std.mem.Allocator, g: *ops.MlxOps) void {
        for (f.outs.items) |x| g.release(x);
        f.outs.deinit(a);
        for (f.bits.items) |b| a.free(b);
        f.bits.deinit(a);
        a.free(f.loc);
        a.free(f.w);
        if (f.loc_arr.ctx != null) g.release(f.loc_arr);
        if (f.w_arr.ctx != null) g.release(f.w_arr);
    }
};

// ── CPU tests ──

const testing = std.testing;

const shaHex = @import("../sdk_ext/kernel_set.zig").shaHex;
