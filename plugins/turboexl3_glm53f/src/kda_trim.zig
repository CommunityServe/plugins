//! `TURBOEXL3_GLM53F_KDA_CONVIN=1` and `TURBOEXL3_GLM53F_KDA_EXPSIG=1` (opt-in, exact): launches a KDA layer's verify
//! (kda.zig's `capture` forward, every decode round's trunk forward, with TURBOEXL3_GLM53F_KDA_VERIFY's texts,
//! src/kda_verify.zig) spends on a copy and on values it makes again every verify, removed:
//!
//! - `KDA_CONVIN`: the prologue text (kernels/kda_prologue.metal, its statements unchanged) writes the capture's conv
//!   input [window | qkv] itself (kernels/kda_convin.metal after it: each thread copies its 4 channels of its row, the
//!   first row's threads the window's rows), where KDA_VERIFY makes it with MLX's concatenate (a copy launch an input: 2
//!   a layer, 68 a verify), and the layer's new window is that conv input's last `taps - 1` rows, the rows KDA_VERIFY's
//!   window holds (at a 2-row verify its window's own concatenate goes too). The same values, copied.
//! - `KDA_EXPSIG`: the gate text's exp(A_log) (the chain's exp of astype(A_log, f32), a launch a layer a verify) made
//!   once a layer, evaluated and kept (`expALog`); and beta's sigmoid (MLX's sigmoid of b, a launch a layer a verify)
//!   made in the capture recurrence's launch: the copy's stock texts (sdk/blocks/gdn/kernels.zig `gdnKernelSource`, the
//!   capture launch, or beside TURBOEXL3_GLM53F_KDA_LAZY_SEQ its launch without the per-row stores and the per-row
//!   states' lazy launch) with their one beta read taking MLX's own sigmoid of b, tabulated over every 16-bit value
//!   (fused_activations' `sigmoidTableFor`, the table KDA_VERIFY's gated-norm text reads), at b's bits
//!   (patches/turboexl3-model.patch in kda.zig). The same exp and sigmoid values, so the same
//!   gate, recurrence, output, state and per-row states.
//!
//! src/kda_gpu_test.zig holds two verifies in a row with each switch, and with both beside KDA_LAZY_SEQ (and the verify
//! after each partial accept's rollback), to KDA_VERIFY's outputs, windows, states, conv inputs and per-row states, bit
//! for bit, at verify widths 1 to 4.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const gdn_kernels = @import("mlx_serve/sdk/blocks/gdn/kernels.zig");
const fused_activations = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

// ── the switches ──

/// `TURBOEXL3_GLM53F_KDA_CONVIN=1`, set by the plugin at load (with KDA_VERIFY), before any forward.
pub var convin_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const convin_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CONVIN=1: a KDA layer's verify prologue text writes the capture's conv input itself (the window's and qkv's rows copied, MLX's concatenate's values), the layer's new window that input's last rows: no concatenate launches, the same bits\n";
/// `TURBOEXL3_GLM53F_KDA_EXPSIG=1`, set by the plugin at load, before any forward.
pub var expsig_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const expsig_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_EXPSIG=1: a KDA layer's verify takes exp(A_log) made once a layer (the chain's exp of A_log as f32, kept) and beta's sigmoid in its capture recurrence's launch (the stock texts reading MLX's own sigmoid, tabulated, at b's bits): the same gate, beta and recurrence bits\n";

/// Since the load: the conv-input prologues; exp(A_log) made and read from the cache; the recurrence launches that made
/// beta's sigmoid themselves (src/kda_gpu_test.zig reads them).
pub var convin_runs: u64 = 0;
pub var ea_made: u64 = 0;
pub var ea_uses: u64 = 0;
pub var bsig_runs: u64 = 0;
var convin_engaged = false;
var bsig_engaged = false;

/// The request's lines (the plugin's meters), one a switch that is on.
pub fn summary(buf: []u8) []const u8 {
    var n: usize = 0;
    if (convin_on) n += (std.fmt.bufPrint(buf[n..], "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CONVIN prologues={d}\n", .{convin_runs}) catch return buf[0..0]).len;
    if (expsig_on) n += (std.fmt.bufPrint(buf[n..], "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_EXPSIG exp_made={d} exp_uses={d} recurrences={d}\n", .{ ea_made, ea_uses, bsig_runs }) catch return buf[0..0]).len;
    return buf[0..n];
}

/// The load's check: KDA_CONVIN edits KDA_VERIFY's prologue text, so it needs it.
pub fn refusal(convin: bool, verify: bool) ?[]const u8 {
    if (convin and !verify) return "Glm53fKdaConvinNeedsVerify";
    return null;
}

// ── KDA_CONVIN: the prologue text writing the conv input ──

const prologue_source = @embedFile("kernels/kda_prologue.metal");
/// The copy after the prologue text (kernels/kda_convin.metal).
pub const convin_copy = @embedFile("kernels/kda_convin.metal");
/// The prologue text with the conv input's copy after it.
pub const convin_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk std.fmt.comptimePrint("{s}{s}", .{ prologue_source, convin_copy });
};
/// The texts' header (kda_sigmoid: unary_ops.h `Sigmoid`, the prologue text's).
const header_source = @embedFile("kernels/kda_prompt_header.metal");
var convin_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// What KDA_VERIFY leaves a verify: q and k normed and scaled and v (each [1, rows, heads, 128] contiguous), the window
/// the layer keeps and the conv input [window | qkv] the capture keeps; each a new array.
pub const Prologue = struct { q: Array, k: Array, v: Array, window: Array, input: Array };

fn isFloat(d: mlx.mlx_dtype) bool {
    return d == .float16 or d == .bfloat16 or d == .float32;
}

/// KDA_VERIFY's prologue (`kda_prompt.prologueText` over `qkv` after `prev`, the layer's window; null: a cold layer,
/// zeros as the chain takes them) with the capture's conv input written by the same launch and the window its last
/// `taps - 1` rows. Null with the switch off or on a shape or dtype the prologue text does not take (the caller runs
/// KDA_VERIFY's).
pub fn prologueConvIn(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    if (!convin_on) return null;
    // prologueText's checks
    const ps = kda_prompt.prologueServes(mlx.getShape(qkv), hk, dk, hv, dv, taps) orelse return null;
    const dt = mlx.mlx_array_dtype(qkv);
    if (!isFloat(dt)) return null;
    const wsh = mlx.getShape(conv_w);
    if (mlx.mlx_array_dtype(conv_w) != dt or wsh.len != 3 or wsh[0] != ps.cdim or wsh[1] != taps or wsh[2] != 1) return null;
    for ([_]Array{ q_scale, k_scale }) |x| if (mlx.mlx_array_dtype(x) != dt or mlx.mlx_array_ndim(x) != 0) return null;
    if (prev) |p| {
        const psh = mlx.getShape(p);
        if (psh.len != 3 or psh[0] != 1 or psh[1] != taps - 1 or psh[2] != ps.cdim or mlx.mlx_array_dtype(p) != dt) return null;
    }
    var cold = Array{ .ctx = null };
    defer if (cold.ctx != null) {
        _ = mlx.mlx_array_free(cold);
    };
    const before = prev orelse blk: {
        cold = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&cold, &[_]c_int{ 1, taps - 1, ps.cdim }, 3, dt, s));
        break :blk cold;
    };
    // prologueText's launch, with the conv input its fourth output
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ 1, ps.rows, hk, dk };
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, taps - 1 + ps.rows, ps.cdim }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 3 * hk, ps.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, kda_prompt.prologue_groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "DK", dk }, .{ "H", hk }, .{ "DIM", hk * dk }, .{ "CDIM", ps.cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const ins = [_]Array{ qkv, before, conv_w, q_scale, k_scale, eps_arr };
    var outs: [4]Array = undefined;
    try kda_prompt.applyKernel(s, try kda_prompt.kernelOf(&convin_kernel, "turboexl3_glm53f_kda_prologue_convin", &.{ "qkv", "prev", "w", "q_scale", "k_scale", "eps" }, &.{ "q_out", "k_out", "v_out", "conv_in" }, convin_source, header_source), cfg, &ins, &outs);
    errdefer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    // the window: the conv input's last taps - 1 rows (conv1d.run's window over [prev | qkv])
    var window = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(window);
    try mlx.check(mlx.mlx_slice(&window, outs[3], &[_]c_int{ 0, ps.rows, 0 }, 3, &[_]c_int{ 1, ps.rows + taps - 1, ps.cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    convin_runs += 1;
    if (!convin_engaged) {
        convin_engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CONVIN engaged: a {d}-row verify's prologue text wrote its conv input ({d} rows of {d} channels)\n", .{ ps.rows, taps - 1 + ps.rows, ps.cdim });
    }
    return .{ .q = outs[0], .k = outs[1], .v = outs[2], .window = window, .input = outs[3] };
}

// ── KDA_EXPSIG: exp(A_log) once a layer ──

/// The most layers the exp(A_log) cache holds (GLM-5.3-Flash has 34 KDA layers); past it a layer makes its own.
pub const max_layers = 64;
const EaEntry = struct { addr: usize, n: usize, dt: mlx.mlx_dtype, a_log: Array, ea: Array };
var ea_cache: [max_layers]EaEntry = undefined;
var ea_count: usize = 0;

/// An evaluated array's data address (its buffer and offset), or null for one not yet evaluated or not a float.
fn dataAddr(a: Array) ?usize {
    var avail = false;
    mlx.check(mlx._mlx_array_is_available(&avail, a)) catch return null;
    if (!avail) return null;
    return switch (mlx.mlx_array_dtype(a)) {
        .float32 => if (mlx.mlx_array_data_float32(a)) |p| @intFromPtr(p) else null,
        .float16 => if (mlx.mlx_array_data_float16(a)) |p| @intFromPtr(p) else null,
        .bfloat16 => if (mlx.mlx_array_data_bfloat16(a)) |p| @intFromPtr(p) else null,
        else => null,
    };
}

/// The gate text's exp(A_log) for a layer's `a_log` (the chain's exp(astype(A_log, f32)), made as the gate text makes
/// it: the same op over the same input, so the same values), made once a layer and evaluated, a reference the cache
/// keeps (the caller does not free it). Keyed by A_log's data address, length and dtype; the cache holds A_log too, so
/// its buffer stays the key's until `reset`. Null with the switch off, for an A_log not evaluated or not a float, or
/// past `max_layers` layers (the caller makes it).
pub fn expALog(s: mlx.mlx_stream, a_log: Array) !?Array {
    if (!expsig_on) return null;
    const addr = dataAddr(a_log) orelse return null;
    const n = mlx.mlx_array_size(a_log);
    const dt = mlx.mlx_array_dtype(a_log);
    for (ea_cache[0..ea_count]) |e| if (e.addr == addr and e.n == n and e.dt == dt) {
        ea_uses += 1;
        return e.ea;
    };
    if (ea_count == ea_cache.len) return null;
    var alf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alf);
    try mlx.check(mlx.mlx_astype(&alf, a_log, .float32, s));
    var ea = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ea);
    try mlx.check(mlx.mlx_exp(&ea, alf, s));
    try mlx.check(mlx.mlx_array_eval(ea));
    var held = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&held, a_log);
    ea_cache[ea_count] = .{ .addr = addr, .n = n, .dt = dt, .a_log = held, .ea = ea };
    ea_count += 1;
    ea_made += 1;
    ea_uses += 1;
    return ea;
}

/// The exp(A_log) cache emptied (the model's unload; a test's end).
pub fn reset() void {
    for (ea_cache[0..ea_count]) |e| {
        _ = mlx.mlx_array_free(e.a_log);
        _ = mlx.mlx_array_free(e.ea);
    }
    ea_count = 0;
}

// ── KDA_EXPSIG: beta's sigmoid in the capture recurrence's launch ──

/// The stock texts' beta read, and KDA_EXPSIG's: MLX's sigmoid of b (the launch's `beta` input holds b) from the table
/// at b's bits, the value the chain's `mlx_sigmoid` stores, in its type.
pub const beta_read = "auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];";
pub const beta_read_tab = "auto delta = (v_[dv_idx] - kv_mem) * sigtab[as_type<ushort>(beta_[hv_idx])];";

/// `s` with `old` replaced, `old` found exactly once.
fn replaceOnce(comptime s: []const u8, comptime old: []const u8, comptime new: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, s, old) orelse @compileError("kda_trim: the stock text has no `" ++ old ++ "`");
    if (std.mem.indexOfPos(u8, s, i + old.len, old) != null) @compileError("kda_trim: the stock text holds `" ++ old ++ "` twice");
    return std.fmt.comptimePrint("{s}{s}{s}", .{ s[0..i], new, s[i + old.len ..] });
}

/// The stock recurrence at the vector gate without and with the per-row stores (`gated_delta_step_vec`,
/// `gated_delta_step_seq_vec`), and KDA_EXPSIG's: the same texts with the beta read edited.
pub const stock_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk gdn_kernels.gdnKernelSource(true, false);
};
pub const stock_seq_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk gdn_kernels.gdnKernelSource(true, true);
};
pub const bsig_source: [:0]const u8 = replaceOnce(stock_source, beta_read, beta_read_tab);
pub const bsig_seq_source: [:0]const u8 = replaceOnce(stock_seq_source, beta_read, beta_read_tab);
var bsig_kernel: ?mlx.mlx_fast_metal_kernel = null;
var bsig_seq_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// MLX's sigmoid of every 16-bit value of b's dtype (fused_activations' table, evaluated once, kept by it: the caller
/// does not free it) for a verify's b [batch, rows, hv]: KDA_EXPSIG's capture launches read it at b's bits. Null with
/// the switch off, off the GPU, or for another shape or a dtype without a table (the caller runs MLX's sigmoid).
pub fn betaTable(s: mlx.mlx_stream, b: Array, batch: c_int, rows: c_int, hv: c_int) !?Array {
    if (!expsig_on or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(b);
    if (dt != .bfloat16 and dt != .float16) return null;
    const sh = mlx.getShape(b);
    if (sh.len != 3 or sh[0] != batch or sh[1] != rows or sh[2] != hv) return null;
    return try fused_activations.sigmoidTableFor(s, dt);
}

pub const Recur = struct { y: Array, state: Array, state_seq: Array };

fn templateArgs(cfg: mlx.mlx_fast_metal_kernel_config, in_dt: mlx.mlx_dtype, st_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, dk: c_int, dv: c_int, hk: c_int, hv: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", hv));
}

/// kda.zig's capture branch (or, with `lazy`, KDA_LAZY_SEQ's branch: the launch without the per-row stores, and the
/// per-row states from the capture launch as a primitive of its own that MLX runs only when a rollback reads them) with
/// beta's sigmoid made in the launches: `b` the beta input before its sigmoid, `tab` `betaTable`'s. The same grid,
/// threadgroups, template arguments and inputs as kda.zig's launches but beta: y [batch, rows, hv, dv] (`out_dt`), the
/// final state and the per-row states [rows, batch, hv, dv, dk] (the last row unwritten, as the capture launch leaves
/// it), each a new array.
pub fn captureRecur(s: mlx.mlx_stream, q: Array, k: Array, v: Array, g: Array, b: Array, tab: Array, state: Array, in_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, rows: c_int, lazy: bool) !Recur {
    const st_dt = mlx.mlx_array_dtype(state);
    const y_shape = [_]c_int{ batch, rows, hv, dv };
    const st_shape = [_]c_int{ batch, hv, dv, dk };
    const t_arr = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(t_arr);
    var r: Recur = undefined;
    var stock: ?[2]Array = null;
    errdefer if (stock) |o| for (o) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (lazy) {
        // KDA_LAZY_SEQ: the verify's output and final state from the launch without the per-row stores
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 4, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &st_shape, 4, st_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, dv, batch * hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
        try templateArgs(cfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
        const kern = try kda_prompt.kernelOf(&bsig_kernel, "turboexl3_glm53f_kda_bsig", &.{ "q", "k", "v", "g", "beta", "state_in", "T", "sigtab" }, &.{ "y", "state_out" }, bsig_source, "");
        var outs: [2]Array = undefined;
        try kda_prompt.applyKernel(s, kern, cfg, &.{ q, k, v, g, b, state, t_arr, tab }, &outs);
        stock = outs;
    }
    // the capture launch: y, the per-row states and the final state (beside KDA_LAZY_SEQ the per-row states alone kept,
    // so it runs only when a rollback reads them)
    const seq_stride = mlx.mlx_array_new_int(batch * hv * dv * dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const scfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(scfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(scfg, &y_shape, 4, out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(scfg, &[_]c_int{ rows, batch, hv, dv, dk }, 5, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(scfg, &st_shape, 4, st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(scfg, 32, dv, batch * hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(scfg, 32, 4, 1));
    try templateArgs(scfg, in_dt, st_dt, out_dt, dk, dv, hk, hv);
    const seq_kern = try kda_prompt.kernelOf(&bsig_seq_kernel, "turboexl3_glm53f_kda_bsig_seq", &.{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride", "sigtab" }, &.{ "y", "state_seq", "state_out" }, bsig_seq_source, "");
    var seq_outs: [3]Array = undefined;
    try kda_prompt.applyKernel(s, seq_kern, scfg, &.{ q, k, v, g, b, state, t_arr, seq_stride, tab }, &seq_outs);
    if (stock) |o| {
        _ = mlx.mlx_array_free(seq_outs[0]);
        _ = mlx.mlx_array_free(seq_outs[2]);
        r = .{ .y = o[0], .state = o[1], .state_seq = seq_outs[1] };
    } else {
        r = .{ .y = seq_outs[0], .state = seq_outs[2], .state_seq = seq_outs[1] };
    }
    bsig_runs += 1;
    if (!bsig_engaged) {
        bsig_engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_EXPSIG engaged: a {d}-row verify's capture recurrence made beta's sigmoid from the table{s}\n", .{ rows, if (lazy) " (KDA_LAZY_SEQ's launch and its lazy per-row states)" else "" });
    }
    return r;
}

// ── tests (MLX-free) ──

const testing = std.testing;

test "turboexl3_glm53f stream kda_trim: KDA_CONVIN's text is the prologue text as it is with the conv input's copy after it, from qkv and prev only" {
    try testing.expect(std.mem.startsWith(u8, convin_source, prologue_source));
    try testing.expectEqualStrings(convin_copy, convin_source[prologue_source.len..]);
    // the copy writes the conv input alone and reads the prologue text's own inputs
    try testing.expectEqual(@as(usize, 0), std.mem.count(u8, prologue_source, "conv_in["));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, convin_copy, "conv_in["));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, convin_copy, "= qkv["));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, convin_copy, "= prev["));
    for ([_][]const u8{ "q_out", "k_out", "v_out", "acc", "inv" }) |name| try testing.expect(std.mem.indexOf(u8, convin_copy, name) == null);
}

test "turboexl3_glm53f stream kda_trim: KDA_EXPSIG's capture texts are the copy's stock texts with their one beta read edited, and undoing the edit gives them back" {
    inline for (.{ .{ stock_source, bsig_source }, .{ stock_seq_source, bsig_seq_source } }) |pair| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, pair[0], beta_read));
        try testing.expectEqual(@as(usize, 0), std.mem.count(u8, pair[1], beta_read));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, pair[1], beta_read_tab));
        try testing.expectEqualStrings(pair[0], comptime replaceOnce(pair[1], beta_read_tab, beta_read));
        // beta is read once a row, through the table
        try testing.expectEqual(std.mem.count(u8, pair[0], "beta_["), std.mem.count(u8, pair[1], "beta_["));
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, pair[1], "sigtab["));
    }
    // the per-row stores only in the capture text
    try testing.expect(std.mem.indexOf(u8, bsig_source, "state_seq") == null);
    try testing.expect(std.mem.indexOf(u8, bsig_seq_source, "state_seq") != null);
}

test "turboexl3_glm53f stream kda_trim: refused without KDA_VERIFY (KDA_CONVIN); the request's lines count each switch's work" {
    try testing.expectEqual(@as(?[]const u8, null), refusal(false, false));
    try testing.expectEqual(@as(?[]const u8, null), refusal(true, true));
    try testing.expectEqualStrings("Glm53fKdaConvinNeedsVerify", refusal(true, false).?);
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("", summary(&buf));
    convin_on = true;
    expsig_on = true;
    convin_runs = 68;
    ea_made = 34;
    ea_uses = 68;
    bsig_runs = 68;
    defer {
        convin_on = false;
        expsig_on = false;
        convin_runs = 0;
        ea_made = 0;
        ea_uses = 0;
        bsig_runs = 0;
    }
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CONVIN prologues=68\n[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_EXPSIG exp_made=34 exp_uses=68 recurrences=68\n", summary(&buf));
}
