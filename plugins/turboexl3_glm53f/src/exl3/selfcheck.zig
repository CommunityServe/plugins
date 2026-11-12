//! The self-checks a GPU run runs before the EXL3 kernel registry is accepted: one
//! executor per (kernel, check) of the manifest's plan. Every trellis state decoded through
//! the GEMVs, golden tiles through the DIG-X decode, row invariance of the decode kernels,
//! bitwise equality with the eager MLX chains the fused kernels replace, float64 parity of
//! the rounding-class kernels, wave composition of the DIG GEMMs, and the rcproj layout guard
//! against MLX's own mxfp8 route. GPU only (DSV41_KERNELS_GPU=1 in a lock-holding window).

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const xk = @import("kernels.zig");
const kset = @import("../streamed/sdk_ext/kernel_set.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Check = xk.Check;
const Entry = xk.Entry;
const Vars = xk.Vars;
const Var = xk.Var;

const scaffold = kset.Checks(Kernel, Check);
pub const Result = scaffold.Result;
pub const Report = scaffold.Report;

/// How much of each kernel's manifest plan runs.
/// - `compile` (the construction's acceptance): every kernel compiles, binds and completes one launch on generated inputs;
///   no numeric comparison (David 10-05: the server's startup runs no numeric probes). A kernel that does not compile or
///   bind refuses the load by name (SelfCheckFailed).
/// - `full` (the device test, `DSV41_SELFCHECK_DEVICE`, and the window smoke tests): every check of the manifest: the
///   exact references (host f64, the MLX chain, the decode table, the join, golden tiles, compositions), the per-site row
///   invariance and the twins. The numeric probes live there only.
pub const Depth = kset.CheckDepth;

/// The checks of `want`'s entries at `depth`, per kernel, in registry order (`stubPlan` runs the same plan).
pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
    var out = scaffold.plan(&reg.entries, want, depth);
    // layout_guard is never run: its oracle was MLX's own mxfp8 quantized_matmul (MLX's to verify, not ours);
    // the rcproj kernels' f64 checks cover them.
    for (&reg.entries) |*e| {
        var cs = out.get(e.kernel);
        cs.remove(.layout_guard);
        out.set(e.kernel, cs);
    }
    return out;
}

const t128_scale: f32 = @bitCast(@as(u32, 0x3DB504F3));

const H = struct {
    a: Allocator,
    reg: *const xk.Registry,
    bound: *const xk.Bound,
    s: mlx.mlx_stream,
    rng: std.Random.DefaultPrng,
    report: *Report,
    table: *const [65536]u16,

};

/// MLX arrays created by one check, freed together.
const Scope = kset.Scope;

fn op(sc: *Scope, comptime f: anytype, args: anytype) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    mlx.check(@call(.auto, f, .{&r} ++ args)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return sc.keep(r);
}

const bf16Bits = @import("sdk").io_util.bf16Rne;

/// A DIG wave: `n` experts, expert j at slot `slots[j]` with `rows[j]` consecutive rows.
const Wave = struct {
    n: usize = 0,
    slots: [16]u32 = @splat(0),
    rows: [16]u32 = @splat(0),

    /// q3_prefill_dig_candidate.wave_table: slot, first row, rows, first threadgroup per expert
    /// (unused threadgroup entries INT32_MAX), then [n, threadgroups], at `bm`-row M tiles. Returns the threadgroups.
    fn table(w: *const Wave, tiles: u32, bm: u32, out: *[80]i32) u64 {
        @memset(out, 0);
        var row0: i64 = 0;
        var tg: i64 = 0;
        for (0..16) |j| {
            if (j >= w.n) {
                out[48 + j] = std.math.maxInt(i32);
                continue;
            }
            out[j] = @intCast(w.slots[j]);
            out[16 + j] = @intCast(row0);
            out[32 + j] = @intCast(w.rows[j]);
            out[48 + j] = @intCast(tg);
            row0 += w.rows[j];
            tg += @as(i64, @intCast((w.rows[j] + bm - 1) / bm)) * tiles;
        }
        out[64] = @intCast(w.n);
        out[65] = @intCast(tg);
        return @intCast(tg);
    }
};

/// One launch of `k`, evaluated; outputs belong to `sc`.
fn launch(h: *H, sc: *Scope, k: Kernel, ins: []const mlx.mlx_array, vars: *const Vars, site: ?[]const u8) ![xk.max_outputs]mlx.mlx_array {
    const e = h.reg.get(k);
    const cfg = try xk.launchFor(e, vars, site);
    var outs: [xk.max_outputs]mlx.mlx_array = @splat(.{});
    try h.bound.apply(k, ins, &cfg, outs[0..cfg.n_out]);
    for (outs[0..cfg.n_out]) |o| _ = try sc.keep(o);
    const v = mlx.mlx_vector_array_new_data(&outs, cfg.n_out);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    if (mlx.errorPending()) return error.MlxError;
    return outs;
}

// ── MLX op helpers (the eager chains the fused kernels replace) ──

fn mul(sc: *Scope, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_multiply, .{ a, b, s });
}

fn reshape(sc: *Scope, a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_reshape, .{ a, shape.ptr, shape.len, s });
}

// ── compile ──

// ── row invariance: every M = 1..rows_max at every slot == those rows of the full call ──

// ── decode_table: one-hot rows through the GEMV read W_hat exactly, every state covered ──

// ── golden_tiles: the DIG-X loader's decode (B^T) == W_hat^T, every word ──

// ── mlx_chain: bitwise equal to the eager MLX chain the kernel replaces ──

// ── f64: parity with a float64 host reference (the rounding-class kernels) ──

fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

// ── DIG-X GEMMs: wave composition and float64 parity ──

// ── Tests ──

const testing = std.testing;
