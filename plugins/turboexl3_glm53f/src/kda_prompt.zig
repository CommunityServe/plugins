//! `TURBOEXL3_GLM53F_KDA_PROLOGUE=1` and `TURBOEXL3_GLM53F_KDA_NORMGATE=1` (opt-in): two pieces of a KDA layer's prompt
//! pass, each one MLX custom kernel over this plugin's own op chain (src/mlx_serve/plugins/glm5_next/kda.zig, through
//! patches/turboexl3-model.patch), so the served words are the chain's. `TURBOEXL3_GLM53F_KDA_PERCORE=1` (opt-in,
//! rounding-class) runs the recurrence and the gate on the copy's per-core kernel instead, and
//! `TURBOEXL3_GLM53F_KDA_CHUNK=1` (opt-in, rounding-class) as the chunkwise delta rule, before the per-core route, and
//! `TURBOEXL3_GLM53F_KDA_FOLD=1` (opt-in, with KDA_CHUNK) folds the prologue into that launch, exact over the two.
//!
//! They stand in for the 1086 tok/s cell's KDA prompt route (mlx-stream cf30601: `kda=percore kda_prologue=fused`),
//! which this plugin cannot take exactly. The copy's per-core recurrence (sdk/blocks/kda/recurrence.zig `planFor`)
//! plans one threadgroup a GPU core at 128 value rows at most, and GLM-5.3-Flash's 64 x 128 value rows need 64 cores
//! where an M5 Max has 40, so on ai-server the copy runs the materialized f32 gate, the op-chain prologue and the stock
//! per-token recurrence; the per-core kernel rounds the delta rule otherwise. The cell's prologue text reproduced
//! mlx-stream's chain (its conv in f32), not this one.
//!
//! - the gate text (`gateTextWith`, which TURBOEXL3_GLM53F_KDA_VERIFY runs in a verify, src/kda_verify.zig):
//!   `kdaGateChain`'s add, cast, multiply, sigmoid, multiply and exp (the cast to f32 of an f32 sum is MLX's no-op) in
//!   one pass that reads the gate input and writes the f32 gate.
//! - `KDA_PROLOGUE`: the chain's concat, depthwise conv, SiLU, q / k / v slices and copies, two rms_norms and two scale
//!   multiplies (about 11 GB a KDA layer at 16K rows) in one pass that reads qkv and writes q, k and v (about 1.6 GB);
//!   the conv window as the chain keeps it (its last rows, the same owned copy).
//! - `KDA_NORMGATE`: the chain's gated norm after the recurrence (rms_norm over each 128-wide value head, sigmoid of
//!   the output gate, their product: about 1.9 GB of memory traffic and three new arrays a KDA layer at 16K rows) in
//!   one pass that reads y and the gate input and writes the out projection's input (about 0.8 GB). The text is the
//!   copy's own (kernels/gdn/gdn_normgate.metal); the copy's KDA layer runs it only beside the copy's per-core route
//!   (which declines a 40-core M5 Max) and at hc_prefill's rows at most, and runs the chain otherwise. Its sigmoid is a
//!   table of MLX's own op over every 16-bit value (the copy's, sdk/blocks/mlp/fused_activations.zig).
//! - `KDA_PERCORE` (rounding-class, David 2026-11-07: prefill kernels judged on prefill): the copy's per-core kernel
//!   (sdk/blocks/kda/recurrence.zig over kernels/kda/kda_percore.metal, oMLX's) for the prompt recurrence and the gate
//!   chain together, the gate built inside the kernel, at the 1086 tok/s cell's plan: two threadgroups a core where one
//!   a core would pass 128 rows (a 40-core M5 Max: 80 threadgroups x 104 rows). Not the chain's bits: its delta-rule
//!   folds are another summation order. src/kda_gpu_test.zig reports its largest differences from the chain.
//! - `KDA_CHUNK` (rounding-class, judged on prefill as KDA_PERCORE): the prompt recurrence and the gate as the chunkwise
//!   delta rule (fla's chunk_kda form, src/kernels/kda_chunk.metal): 16-row chunks, each through its triangular solve
//!   in f32 simdgroup matrices, the state carried once a chunk, not once a row (the per-core kernel's 16,356 dependent
//!   steps a prompt become 1,023 of matrix work). It takes kda.zig's per-core hooks (`percoreServes`, `percoreRecur`), so
//!   kda.zig is unchanged, and runs before the per-core route where both are on. Not the chain's bits (other sums);
//!   src/kda_gpu_test.zig reports its largest differences from the chain and from the per-core route.
//! - `KDA_FOLD` (exact over KDA_PROLOGUE and KDA_CHUNK, with KDA_CHUNK on): the prologue text inside the chunkwise
//!   launch (src/kernels/kda_fold.metal, through patches/turboexl3-model.patch): each chunk's q and k rows
//!   made from qkv and the conv window before them, one simdgroup a row as the text's, and v at each lane's own
//!   elements, each value the text's statements, rounded where the text rounds; q, k and v never reach memory (about
//!   1.6 GB of a KDA layer's memory traffic at 16K rows, written and read again) and the prologue's launch goes. The
//!   outputs are KDA_CHUNK's over KDA_PROLOGUE's bit for bit; src/kda_gpu_test.zig holds them to those bits.
//!
//! Each step of each exact text is the kernel the chain runs for it (the server's MLX 64ea011c: conv.metal
//! `depthwise_conv_1d`, unary_ops.h `Sigmoid` and `Exp`, binary_ops.h `Add` and `Multiply`, rms_norm.metal
//! `rms_single_row`; the copy's own `gated_delta_step_vec`), or a table of that kernel's outputs, in the same operand
//! order and types; MLX compiles a custom kernel in its safe math mode, as its own library is built (-fno-fast-math),
//! so the outputs are the chain's bits; src/kda_gpu_test.zig holds each to the chain on the device, bit for bit
//! (build.zig `kda-gpu-test`). Each runs only where it is written for the shapes and dtypes (else the chain runs) and
//! only in a prompt pass's forward (kda.zig: a prefill forward without a verify's capture).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const depthwise = @import("mlx_serve/sdk/blocks/conv/depthwise.zig");
const array = @import("mlx_serve/mlx/array.zig");
const run_base = @import("mlx_serve/sdk/bases/run.zig");
const kernel_text = @import("mlx_serve/mlx/kernels.zig");
const fused_activations = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const Array = mlx.mlx_array;
const Dtype = mlx.mlx_dtype;

/// The texts' header: unary_ops.h `Sigmoid` as a function (src/kernels/kda_prompt_header.metal).
const header_source = @embedFile("kernels/kda_prompt_header.metal");

/// MLX's promotion of two float dtypes (dtype.cpp `promote_types` over f16, bf16 and f32): the dtype when both are it,
/// else f32 (f32 over either, and bf16 with f16); null for a dtype the texts do not take.
pub fn promote(a: Dtype, b: Dtype) ?Dtype {
    if (!isFloat(a) or !isFloat(b)) return null;
    return if (a == b) a else .float32;
}
fn isFloat(d: Dtype) bool {
    return d == .float16 or d == .bfloat16 or d == .float32;
}

// ── the gate ──

const gate_source = @embedFile("kernels/kda_gate.metal");

/// The gate text's threadgroup: a row's elements are a whole number of them.
pub const gate_tg: c_int = 256;
pub const GateShape = struct { rows: c_int, n: c_int };

/// Whether the gate text takes a gate input of `shape` beside dt_bias and the expanded A_log of `n_dt` and `n_alog`
/// elements: one slot's rows [1, rows, n], n a whole number of threadgroups, each table n long.
pub fn gateServes(shape: []const c_int, n_dt: usize, n_alog: usize) ?GateShape {
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] < gate_tg) return null;
    if (@rem(shape[2], gate_tg) != 0) return null;
    const n: usize = @intCast(shape[2]);
    if (n_dt != n or n_alog != n) return null;
    return .{ .rows = shape[1], .n = shape[2] };
}

var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The gate text's launch, the switches aside: `g` f32 [1, rows, n] (a new array), with the shape and the add's dtype
/// it ran at.
pub const GateOut = struct { g: Array, rows: c_int, n: c_int, add_t: Dtype };

/// `kdaGateChain(a_log, a, dt_bias, bound)` as one text (src/kda_verify.zig in a verify), with `ea_kept`
/// (TURBOEXL3_GLM53F_KDA_EXPSIG's exp(A_log), made once a layer and kept by src/kda_trim.zig: not freed here) in place of
/// the exp(astype(A_log, f32)) it makes (null makes it); null on a shape or dtype the text does not take (the caller
/// runs the chain).
pub fn gateTextWith(s: mlx.mlx_stream, a: Array, dt_bias: Array, a_log: Array, ea_kept: ?Array, bound: f32) !?GateOut {
    const gs = gateServes(mlx.getShape(a), mlx.mlx_array_size(dt_bias), mlx.mlx_array_size(a_log)) orelse return null;
    const add_t = promote(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(dt_bias)) orelse return null;
    if (!isFloat(mlx.mlx_array_dtype(a_log))) return null;
    var made = Array{ .ctx = null };
    defer if (made.ctx != null) {
        _ = mlx.mlx_array_free(made);
    };
    const ea = ea_kept orelse blk: {
        // the chain's exp(astype(A_log, f32)), made as the chain makes it
        var alf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(alf);
        try mlx.check(mlx.mlx_astype(&alf, a_log, .float32, s));
        made = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_exp(&made, alf, s));
        break :blk made;
    };
    // the chain's bound: an f32 scalar
    const bound_arr = mlx.mlx_array_new_float(bound);
    defer _ = mlx.mlx_array_free(bound_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, gs.rows, gs.n }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, gs.n, gs.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, gate_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "AddT", add_t));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", gs.n));
    const ins = [_]Array{ a, dt_bias, ea, bound_arr };
    var outs: [1]Array = undefined;
    try applyKernel(s, try kernelOf(&gate_kernel, "turboexl3_glm53f_kda_gate", &.{ "a", "dt", "ea", "bound" }, &.{"g"}, gate_source, header_source), cfg, &ins, &outs);
    return .{ .g = outs[0], .rows = gs.rows, .n = gs.n, .add_t = add_t };
}

// ── the recurrence ──

pub const Recur = struct { y: Array, state: Array };

// ── the prologue ──

/// `TURBOEXL3_GLM53F_KDA_PROLOGUE=1`, set by the plugin at load, before any forward.
pub var prologue_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const prologue_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PROLOGUE=1: a KDA layer's prompt prologue as one text (the chain's depthwise conv, SiLU, q and k rms_norms and scales, and v, each the MLX kernel the chain runs, in its order and types), the conv window as the chain keeps it\n";
const prologue_source = @embedFile("kernels/kda_prologue.metal");

/// The prologue text's simdgroups a threadgroup.
pub const prologue_groups: c_int = 8;
/// The key head width the text is written for (rms_single_row's 32 lanes x 4 reads).
pub const head_dim: c_int = 128;
pub const ProShape = struct { rows: c_int, cdim: c_int };

/// Whether the prologue text takes qkv rows of `shape` at `hk` x `dk` key and `hv` x `dv` value heads and `taps` conv
/// taps: one slot's rows [1, rows, 3 x hk x dk], heads alike and 128 wide, whole threadgroups of (section, head)
/// groups.
pub fn prologueServes(shape: []const c_int, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) ?ProShape {
    if (hk != hv or dk != dv or dk != head_dim or hk < 1 or taps < 2) return null;
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 3 * hk * dk) return null;
    if (@rem(3 * hk, prologue_groups) != 0) return null;
    return .{ .rows = shape[1], .cdim = shape[2] };
}

/// What the chain leaves for the recurrence: q and k normed and scaled and v, each [1, rows, heads, 128] contiguous,
/// and the conv window the layer's state keeps; each a new array.
pub const Prologue = struct { q: Array, k: Array, v: Array, window: Array };

var prologue_kernel: ?mlx.mlx_fast_metal_kernel = null;
var prologue_engaged = false;
/// Launches since the load (src/kda_gpu_test.zig reads that the text ran).
pub var prologue_runs: u64 = 0;

/// The chain's prologue as one text over `qkv` after `prev` (the layer's conv window; null: a cold layer, zeros as the
/// chain takes them), `conv_w` [cdim, taps, 1], the chain's own scalar scales and its rms epsilon. Null with the switch
/// off or on a shape or dtype the text does not take (the caller runs the chain).
pub fn prologue(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    if (!prologue_on) return null;
    const p = (try prologueText(s, qkv, prev, conv_w, q_scale, k_scale, eps, hk, dk, hv, dv, taps)) orelse return null;
    prologue_runs += 1;
    if (!prologue_engaged) {
        prologue_engaged = true;
        const qs = mlx.getShape(qkv);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PROLOGUE engaged: {d} rows x {d} channels ({d} heads of {d}, {d} taps) in one text\n", .{ qs[1], qs[2], hk, dk, taps });
    }
    return p;
}

/// `prologue`'s text, whichever pass asks (`prologue` in a prompt pass, src/kda_verify.zig in a verify); null on a
/// shape or dtype the text does not take (the caller runs the chain).
pub fn prologueText(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    const ps = prologueServes(mlx.getShape(qkv), hk, dk, hv, dv, taps) orelse return null;
    const dt = mlx.mlx_array_dtype(qkv);
    if (!isFloat(dt)) return null;
    // The chain's conv weights, window and scales in qkv's dtype (else MLX promotes and the text is not the chain), the
    // scales scalars (the text reads them as values).
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
    const window = try windowOf(s, before, qkv, ps.rows, ps.cdim, taps);
    errdefer _ = mlx.mlx_array_free(window);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{ 1, ps.rows, hk, dk };
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 3 * hk, ps.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, prologue_groups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "DK", dk }, .{ "H", hk }, .{ "DIM", hk * dk }, .{ "CDIM", ps.cdim }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const ins = [_]Array{ qkv, before, conv_w, q_scale, k_scale, eps_arr };
    var outs: [3]Array = undefined;
    try applyKernel(s, try kernelOf(&prologue_kernel, "turboexl3_glm53f_kda_prologue", &.{ "qkv", "prev", "w", "q_scale", "k_scale", "eps" }, &.{ "q_out", "k_out", "v_out" }, prologue_source, header_source), cfg, &ins, &outs);
    return .{ .q = outs[0], .k = outs[1], .v = outs[2], .window = window };
}

/// conv1d.run's window over [before | qkv]: its rows [rows, rows + taps - 1), as an owned copy past the chain's
/// PREFILL_EVAL_MIN_SEQ rows (`materializedOwnedCopy`, as the chain makes it), else a slice.
fn windowOf(s: mlx.mlx_stream, before: Array, qkv: Array, rows: c_int, cdim: c_int, taps: c_int) !Array {
    const keep = taps - 1;
    var tail = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tail);
    if (rows >= keep) {
        try mlx.check(mlx.mlx_slice(&tail, qkv, &[_]c_int{ 0, rows - keep, 0 }, 3, &[_]c_int{ 1, rows, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    } else {
        const parts = [_]Array{ before, qkv };
        const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        try mlx.check(mlx.mlx_slice(&tail, cat, &[_]c_int{ 0, rows, 0 }, 3, &[_]c_int{ 1, rows + keep, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    }
    if (depthwise.compactConvStateEnabled() and rows >= run_base.PREFILL_EVAL_MIN_SEQ) {
        const owned = try array.materializedOwnedCopy(s, tail);
        _ = mlx.mlx_array_free(tail);
        depthwise.compact_conv_state_copy_count += 1;
        return owned;
    }
    return tail;
}

// ── the per-core recurrence (rounding-class) ──

/// `TURBOEXL3_GLM53F_KDA_PERCORE=1`, set by the plugin at load; `percoreInstall` then plans and probes the route.
pub var percore_on: bool = false;
/// The switch's load line, after the install: threadgroups, value rows a threadgroup, threadgroups a core, the GPU's
/// cores.
pub const percore_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PERCORE=1: a KDA layer's prompt recurrence and its bounded gate on the per-core kernel (the copy's kda_percore.metal, oMLX's; the gate built in the kernel, never materialized), {d} threadgroups x {d} value rows, {d} a core on {d} GPU cores (the 1086 tok/s cell's plan); rounding-class against the per-token route\n";

/// The copy's per-core kernel text (sdk/blocks/kda/recurrence.zig builds the same from it, with no header).
const percore_source = kernel_text.source("kernels/kda/kda_percore.metal");
/// Value rows a threadgroup carries at most (8 lanes a row: 1,024 threads), and threadgroups a core at most.
pub const percore_max_rows: c_int = 128;
pub const percore_max_per_core: c_int = 2;
pub const PercorePlan = struct { ntg: c_int, rows: c_int, per_core: c_int };

/// mlx-stream cf30601's `planFor` (the 1086 tok/s cell's): the fewest threadgroups a core, at most two, whose even
/// split of `h * dv` value rows keeps every threadgroup at `percore_max_rows` rows or fewer. The copy's own planFor
/// stops at one a core, so it declines GLM-5.3-Flash's 8,192 value rows on 40 cores. A row's arithmetic does not depend
/// on the plan (its eight lanes, its staged q / k / gate / v / beta and its folds are its own).
pub fn percorePlan(h: c_int, dv: c_int, cores: c_int) ?PercorePlan {
    const total = h * dv;
    if (cores < 1 or total < 2) return null;
    const per_core = @divTrunc(total + cores * percore_max_rows - 1, cores * percore_max_rows);
    if (per_core > percore_max_per_core) return null;
    const ntg = @min(cores * per_core, @divTrunc(total, 2));
    const rows = 2 * @divTrunc(@divTrunc(total, 2) + ntg - 1, ntg);
    return if (rows <= percore_max_rows) .{ .ntg = ntg, .rows = rows, .per_core = per_core } else null;
}

var percore_plan: ?PercorePlan = null;
var percore_heads: c_int = 0;
var percore_kernel: ?mlx.mlx_fast_metal_kernel = null;
var percore_engaged = false;
/// Launches since the load (src/kda_gpu_test.zig reads that the route ran).
pub var percore_runs: u64 = 0;

/// The route for `hv` value heads (key heads alike, 128 wide) on `cores` GPU cores: the plan, then one launch over
/// zeros at it, evaluated (Metal caps a kernel's threadgroup by its register use, and MLX reports that only at
/// evaluation). Refuses by name: a stream that runs no Metal kernels, another head layout, no plan for the cores, a
/// threadgroup past this GPU's cap.
pub fn percoreInstall(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, cores: u32) !PercorePlan {
    if (!mlx.streamIsGpu(s)) return error.Glm53fKdaPercoreStream;
    if (hk != hv or dk != head_dim or dv != head_dim) return error.Glm53fKdaPercoreLayout;
    const plan = percorePlan(hv, dv, @intCast(cores)) orelse return error.Glm53fKdaPercoreCores;
    if (!try percoreProbe(s, hv, plan)) return error.Glm53fKdaPercoreThreads;
    percore_plan = plan;
    percore_heads = hv;
    return plan;
}

/// One launch over zeros at `plan` for `h` heads (bf16 activations, as the prompt pass's), evaluated: false where this
/// GPU refuses the threadgroup.
fn percoreProbe(s: mlx.mlx_stream, h: c_int, plan: PercorePlan) !bool {
    var arrs: [6]Array = @splat(.{ .ctx = null });
    defer for (arrs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    const shapes = [_][]const c_int{ &.{ 1, 1, h, head_dim }, &.{ 1, 1, h * head_dim }, &.{ 1, 1, h }, &.{h}, &.{h * head_dim}, &.{ 1, h, head_dim, head_dim } };
    const dts = [_]Dtype{ .bfloat16, .bfloat16, .bfloat16, .float32, .float32, .float32 };
    for (&arrs, shapes, dts) |*x, sh, d| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(x, sh.ptr, sh.len, d, s));
    }
    const r = try percoreLaunch(s, plan, arrs[0], arrs[0], arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], 0.0, arrs[5]);
    defer {
        _ = mlx.mlx_array_free(r.y);
        _ = mlx.mlx_array_free(r.state);
    }
    mlx.check(mlx.mlx_array_eval(r.state)) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

/// Whether the per-core route takes a prompt span: the switch on and installed for these heads (key heads alike, 128
/// wide), 16-bit activations, an f32 state, and the layer's per-head A_log and f32 dt_bias (trunk.zig loads both for
/// GLM-5.3-Flash, `a_log_h` and `dt_f32`). kda.zig's hook for the chunkwise route too (TURBOEXL3_GLM53F_KDA_CHUNK
/// installed for these heads), which takes the same inputs.
pub fn percoreServes(hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: Dtype, state: Dtype, a_log_h: Array, dt_f32: Array) bool {
    const percore = percore_on and percore_plan != null and hv == percore_heads;
    if (!percore and !chunkReady(hv)) return false;
    if (hk != hv or dk != head_dim or dv != head_dim) return false;
    if ((act != .bfloat16 and act != .float16) or state != .float32) return false;
    if (a_log_h.ctx == null or dt_f32.ctx == null) return false;
    return mlx.mlx_array_size(a_log_h) == @as(usize, @intCast(hv)) and mlx.mlx_array_size(dt_f32) == @as(usize, @intCast(hv * dk));
}

/// The per-core recurrence over a prompt span (where `percoreServes`): q, k [B, T, H, 128] normed and scaled, v [B, T,
/// H, 128], the gate's pre-activation `a` [B, T, H * 128], beta [B, T, H], the per-head A_log, the f32 dt_bias, the
/// bound and the f32 state [B, H, 128, 128]: y [B, T, H, 128] in q's dtype and the new state, each a new array. The
/// chunkwise route's where it is installed for these heads (TURBOEXL3_GLM53F_KDA_CHUNK, before the per-core route).
pub fn percoreRecur(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log_h: Array, dt_f32: Array, lower_bound: f32, state: Array) !Recur {
    if (chunkReady(mlx.getShape(q)[2])) return chunkRecur(s, q, k, v, a, beta, a_log_h, dt_f32, lower_bound, state);
    const plan = percore_plan orelse return error.Glm53fKdaPercoreNotInstalled;
    const r = try percoreLaunch(s, plan, q, k, v, a, beta, a_log_h, dt_f32, lower_bound, state);
    percore_runs += 1;
    if (!percore_engaged) {
        percore_engaged = true;
        const qs = mlx.getShape(q);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_PERCORE engaged: {d} rows x {d} value heads on {d} threadgroups x {d} rows, the gate built in the kernel\n", .{ qs[1], qs[2], plan.ntg, plan.rows });
    }
    return r;
}

/// One launch at `plan`, as the copy's `launch` makes it: `a` and beta in q's dtype, A_log and dt_bias f32, the bound
/// an f32 [1], the copy's kernel text with no header.
fn percoreLaunch(s: mlx.mlx_stream, plan: PercorePlan, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log: Array, dt_bias: Array, lower_bound: f32, state: Array) !Recur {
    const qs = mlx.getShape(q);
    const b = qs[0];
    const t = qs[1];
    const h = qs[2];
    const dt = mlx.mlx_array_dtype(q);
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_astype(&a_c, a, dt, s));
    var beta_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_c);
    try mlx.check(mlx.mlx_astype(&beta_c, beta, dt, s));
    var alog = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alog);
    try mlx.check(mlx.mlx_astype(&alog, a_log, .float32, s));
    var dtb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dtb);
    try mlx.check(mlx.mlx_astype(&dtb, dt_bias, .float32, s));
    const lbv = [1]f32{lower_bound};
    const lb = mlx.mlx_array_new_data(&lbv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lb);
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, h, head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, h, head_dim, head_dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, plan.rows * 8 * plan.ntg, 1, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, plan.rows * 8, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", head_dim }, .{ "Dv", head_dim }, .{ "H", h }, .{ "NTG", plan.ntg }, .{ "MR", plan.rows } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ q, k, v, a_c, beta_c, alog, dtb, lb, state, t_arr };
    var outs: [2]Array = undefined;
    try applyKernel(s, try kernelOf(&percore_kernel, "turboexl3_glm53f_kda_percore", &.{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" }, &.{ "y", "state_out" }, percore_source, ""), cfg, &ins, &outs);
    return .{ .y = outs[0], .state = outs[1] };
}

// ── the chunkwise recurrence (rounding-class) ──

/// `TURBOEXL3_GLM53F_KDA_CHUNK=1`, set by the plugin at load; `chunkInstall` then checks the layer and probes the kernel.
pub var chunk_on: bool = false;
/// The switch's load line, after the install: rows a chunk, threadgroups, value rows a threadgroup.
pub const chunk_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CHUNK=1: a KDA layer's prompt recurrence and its bounded gate as the chunkwise delta rule (fla's chunk_kda form, src/kernels/kda_chunk.metal: {d}-row chunks, each through its triangular solve in f32 simdgroup matrices, the state carried once a chunk, the gate built in the kernel), {d} threadgroups x {d} value rows; rounding-class against the per-token route; before the per-core route where both are on\n";

const chunk_source = @embedFile("kernels/kda_chunk.metal");
/// The kernel's header: Metal's simdgroup matrices and the texts' `kda_sigmoid` (unary_ops.h `Sigmoid`, the chain's).
const chunk_header = "#include <metal_simdgroup_matrix>\n" ++ @embedFile("kernels/kda_prompt_header.metal");
/// The kernel's rows a chunk, value rows a threadgroup (8 simdgroups of 8 value rows) and threads a threadgroup.
pub const chunk_rows: c_int = 16;
pub const chunk_value_rows: c_int = 64;
pub const chunk_threads: c_int = 256;
/// The most negative gate bound the kernel takes: a chunk's factors, against its middle row, stay within e^(8 x 5) and
/// e^-(8 x 5), where f32 neither overflows nor flushes a term that counts.
pub const chunk_min_bound: f32 = -5.0;

/// Whether the chunkwise kernel takes a layer of `hk` x `dk` key and `hv` x `dv` value heads: heads alike and 128 wide
/// (its 16 key-channel blocks of 8, two threadgroups of 64 value rows a head).
pub fn chunkLayoutServes(hk: c_int, hv: c_int, dk: c_int, dv: c_int) bool {
    return hk == hv and hk >= 1 and dk == head_dim and dv == head_dim;
}

/// Whether the kernel takes the gate's bound: in [chunk_min_bound, 0) (a bounded gate; GLM-5.3-Flash's is -5).
pub fn chunkBoundServes(lower_bound: f32) bool {
    return lower_bound < 0 and lower_bound >= chunk_min_bound;
}

/// The kernel's threadgroups for `h` value heads of 128.
pub fn chunkGroups(h: c_int) c_int {
    return h * @divExact(head_dim, chunk_value_rows);
}

var chunk_heads: c_int = 0;
var chunk_kernel: ?mlx.mlx_fast_metal_kernel = null;
var chunk_engaged = false;
/// Launches since the load (src/kda_gpu_test.zig reads that the route ran).
pub var chunk_runs: u64 = 0;

fn chunkReady(hv: c_int) bool {
    return chunk_on and chunk_heads != 0 and hv == chunk_heads;
}

/// The route for `hv` value heads (key heads alike, 128 wide) at the gate bound `lower_bound`: one launch over zeros,
/// evaluated (its library built here; Metal caps a kernel's threadgroup by its register use, and MLX reports that only
/// at evaluation). Returns the threadgroups a launch. Refuses by name: a stream that runs no Metal kernels, another
/// head layout, a bound past the kernel's range, a threadgroup past this GPU's cap.
pub fn chunkInstall(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, lower_bound: f32) !c_int {
    if (!mlx.streamIsGpu(s)) return error.Glm53fKdaChunkStream;
    if (!chunkLayoutServes(hk, hv, dk, dv)) return error.Glm53fKdaChunkLayout;
    if (!chunkBoundServes(lower_bound)) return error.Glm53fKdaChunkBound;
    if (!try chunkProbe(s, hv, lower_bound)) return error.Glm53fKdaChunkThreads;
    chunk_heads = hv;
    return chunkGroups(hv);
}

/// One launch over zeros for `h` heads (bf16 activations, as the prompt pass's), evaluated: false where this GPU refuses
/// the threadgroup.
fn chunkProbe(s: mlx.mlx_stream, h: c_int, lower_bound: f32) !bool {
    var arrs: [6]Array = @splat(.{ .ctx = null });
    defer for (arrs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    const shapes = [_][]const c_int{ &.{ 1, 1, h, head_dim }, &.{ 1, 1, h * head_dim }, &.{ 1, 1, h }, &.{h}, &.{h * head_dim}, &.{ 1, h, head_dim, head_dim } };
    const dts = [_]Dtype{ .bfloat16, .bfloat16, .bfloat16, .float32, .float32, .float32 };
    for (&arrs, shapes, dts) |*x, sh, d| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(x, sh.ptr, sh.len, d, s));
    }
    const r = try chunkLaunch(s, arrs[0], arrs[0], arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], lower_bound, arrs[5]);
    defer {
        _ = mlx.mlx_array_free(r.y);
        _ = mlx.mlx_array_free(r.state);
    }
    mlx.check(mlx.mlx_array_eval(r.state)) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

/// The chunkwise recurrence over a prompt span, `percoreRecur`'s inputs and outputs: y [B, T, H, 128] in q's dtype and
/// the new f32 state, each a new array.
fn chunkRecur(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log_h: Array, dt_f32: Array, lower_bound: f32, state: Array) !Recur {
    const r = try chunkLaunch(s, q, k, v, a, beta, a_log_h, dt_f32, lower_bound, state);
    chunk_runs += 1;
    if (!chunk_engaged) {
        chunk_engaged = true;
        const qs = mlx.getShape(q);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_CHUNK engaged: {d} rows x {d} value heads in {d}-row chunks on {d} threadgroups, the gate built in the kernel\n", .{ qs[1], qs[2], chunk_rows, chunkGroups(qs[2]) });
    }
    return r;
}

/// One launch, with the per-core launch's inputs (`a` and beta in q's dtype, A_log and dt_bias f32, the bound an f32
/// [1]): grid (256 x the threadgroups, 1, B).
fn chunkLaunch(s: mlx.mlx_stream, q: Array, k: Array, v: Array, a: Array, beta: Array, a_log: Array, dt_bias: Array, lower_bound: f32, state: Array) !Recur {
    const qs = mlx.getShape(q);
    const b = qs[0];
    const t = qs[1];
    const h = qs[2];
    const dt = mlx.mlx_array_dtype(q);
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_astype(&a_c, a, dt, s));
    var beta_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_c);
    try mlx.check(mlx.mlx_astype(&beta_c, beta, dt, s));
    var alog = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alog);
    try mlx.check(mlx.mlx_astype(&alog, a_log, .float32, s));
    var dtb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dtb);
    try mlx.check(mlx.mlx_astype(&dtb, dt_bias, .float32, s));
    const lbv = [1]f32{lower_bound};
    const lb = mlx.mlx_array_new_data(&lbv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lb);
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, t, h, head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, h, head_dim, head_dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, chunk_threads * chunkGroups(h), 1, b));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, chunk_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", head_dim }, .{ "Dv", head_dim }, .{ "H", h } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ q, k, v, a_c, beta_c, alog, dtb, lb, state, t_arr };
    var outs: [2]Array = undefined;
    try applyKernel(s, try kernelOf(&chunk_kernel, "turboexl3_glm53f_kda_chunk", &.{ "q", "k", "v", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" }, &.{ "y", "state_out" }, chunk_source, chunk_header), cfg, &ins, &outs);
    return .{ .y = outs[0], .state = outs[1] };
}

// ── the prologue folded into the chunkwise recurrence (exact over KDA_PROLOGUE and KDA_CHUNK) ──

/// `TURBOEXL3_GLM53F_KDA_FOLD=1`, set by the plugin at load (with KDA_CHUNK on); `foldInstall` then probes the kernel.
pub var fold_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const fold_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_FOLD=1: a KDA layer's prompt prologue folded into the chunkwise recurrence's launch (src/kernels/kda_fold.metal: the prologue text's statements for q, k and v, read from the layer's qkv rows and the conv window before them), the conv window kept as the prologue keeps it; KDA_CHUNK's outputs over KDA_PROLOGUE's q, k and v, bit for bit\n";

const fold_source = @embedFile("kernels/kda_fold.metal");
var fold_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fold_ready = false;
var fold_taps: c_int = 0;
var fold_engaged = false;
/// Launches since the load (src/kda_gpu_test.zig reads that the route ran).
pub var fold_runs: u64 = 0;

/// The fold for `hv` value heads (key heads alike, 128 wide, the chunkwise route installed for them: KDA_CHUNK) and
/// the layer's `taps` conv taps: one launch over zeros, evaluated (Metal caps a kernel's threadgroup by its register
/// use). Refuses by name: the chunkwise route not installed for these heads, fewer than 2 taps (the prologue text's
/// least), a threadgroup past this GPU's cap.
pub fn foldInstall(s: mlx.mlx_stream, hv: c_int, taps: c_int, lower_bound: f32) !void {
    if (chunk_heads == 0 or hv != chunk_heads) return error.Glm53fKdaFoldNeedsChunk;
    if (taps < 2) return error.Glm53fKdaFoldTaps;
    if (!try foldProbe(s, hv, taps, lower_bound)) return error.Glm53fKdaFoldThreads;
    fold_taps = taps;
    fold_ready = true;
}

fn foldProbe(s: mlx.mlx_stream, h: c_int, taps: c_int, lower_bound: f32) !bool {
    const cdim = 3 * h * head_dim;
    var arrs: [10]Array = @splat(.{ .ctx = null });
    defer for (arrs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    const shapes = [_][]const c_int{ &.{ 1, 1, cdim }, &.{ 1, taps - 1, cdim }, &.{ cdim, taps, 1 }, &.{}, &.{}, &.{ 1, 1, h * head_dim }, &.{ 1, 1, h }, &.{h}, &.{h * head_dim}, &.{ 1, h, head_dim, head_dim } };
    const dts = [_]Dtype{ .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32, .float32, .float32 };
    for (&arrs, shapes, dts) |*x, sh, d| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(x, sh.ptr, sh.len, d, s));
    }
    const r = try foldLaunch(s, arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], 1e-6, arrs[5], arrs[6], arrs[7], arrs[8], lower_bound, arrs[9], h);
    defer {
        _ = mlx.mlx_array_free(r.y);
        _ = mlx.mlx_array_free(r.state);
    }
    mlx.check(mlx.mlx_array_eval(r.state)) catch |e| {
        if (e == error.MlxError and mlx.takeErrorIf("Thread group size")) return false;
        return e;
    };
    return true;
}

/// What the folded launch reads beside qkv, the conv window before the forward (zeros for a cold layer, as the prologue
/// text takes it), and the window the layer keeps, as the prologue text keeps it; each a new array.
pub const FoldWindow = struct { before: Array, window: Array };

/// Whether the fold takes a KDA layer's prompt forward, with its windows: the switch on and installed for these taps, a
/// GPU stream, kda.zig's per-core hooks taking the recurrence (`percoreServes`: here the chunkwise route, installed for
/// these heads), and the prologue text's layout and dtypes (`prologueServes`, the conv weights and scales in qkv's
/// dtype). Null where it does not (the caller runs the prologue and the recurrence as before).
pub fn foldWindow(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int, state: Dtype, a_log_h: Array, dt_f32: Array) !?FoldWindow {
    if (!fold_on or !fold_ready or taps != fold_taps or !chunkReady(hv) or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(qkv);
    if (!percoreServes(hk, hv, dk, dv, dt, state, a_log_h, dt_f32)) return null;
    const ps = prologueServes(mlx.getShape(qkv), hk, dk, hv, dv, taps) orelse return null;
    const wsh = mlx.getShape(conv_w);
    if (mlx.mlx_array_dtype(conv_w) != dt or wsh.len != 3 or wsh[0] != ps.cdim or wsh[1] != taps or wsh[2] != 1) return null;
    for ([_]Array{ q_scale, k_scale }) |x| if (mlx.mlx_array_dtype(x) != dt or mlx.mlx_array_ndim(x) != 0) return null;
    if (prev) |p| {
        const psh = mlx.getShape(p);
        if (psh.len != 3 or psh[0] != 1 or psh[1] != taps - 1 or psh[2] != ps.cdim or mlx.mlx_array_dtype(p) != dt) return null;
    }
    var before = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(before);
    if (prev) |p| {
        try mlx.check(mlx.mlx_array_set(&before, p));
    } else {
        try mlx.check(mlx.mlx_zeros(&before, &[_]c_int{ 1, taps - 1, ps.cdim }, 3, dt, s));
    }
    const window = try windowOf(s, before, qkv, ps.rows, ps.cdim, taps);
    return .{ .before = before, .window = window };
}

/// The folded launch over a prompt span (where `foldWindow` took it): qkv [1, T, 3 x H x 128], the window before it
/// (`FoldWindow.before`), the conv weights, the chain's q and k scales and its rms epsilon, then the chunkwise route's
/// inputs (`percoreRecur`'s): y [1, T, H, 128] in qkv's dtype and the new f32 state, each a new array.
pub fn foldRecur(s: mlx.mlx_stream, qkv: Array, before: Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, a: Array, beta: Array, a_log_h: Array, dt_f32: Array, lower_bound: f32, state: Array) !Recur {
    const r = try foldLaunch(s, qkv, before, conv_w, q_scale, k_scale, eps, a, beta, a_log_h, dt_f32, lower_bound, state, chunk_heads);
    fold_runs += 1;
    if (!fold_engaged) {
        fold_engaged = true;
        const qs = mlx.getShape(qkv);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_FOLD engaged: {d} rows x {d} value heads from qkv in {d}-row chunks on {d} threadgroups, the prologue in the launch\n", .{ qs[1], chunk_heads, chunk_rows, chunkGroups(chunk_heads) });
    }
    return r;
}

/// One launch for `h` heads, as the chunkwise launch (`a` and beta in qkv's dtype, A_log and dt_bias f32, the bound an
/// f32 [1]) with the prologue text's inputs (qkv, the window before it, the conv weights, the scales, the epsilon an
/// f32).
fn foldLaunch(s: mlx.mlx_stream, qkv: Array, before: Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, a: Array, beta: Array, a_log: Array, dt_bias: Array, lower_bound: f32, state: Array, h: c_int) !Recur {
    const qs = mlx.getShape(qkv);
    const t = qs[1];
    const taps = mlx.getShape(conv_w)[1];
    const dt = mlx.mlx_array_dtype(qkv);
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_astype(&a_c, a, dt, s));
    var beta_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_c);
    try mlx.check(mlx.mlx_astype(&beta_c, beta, dt, s));
    var alog = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(alog);
    try mlx.check(mlx.mlx_astype(&alog, a_log, .float32, s));
    var dtb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dtb);
    try mlx.check(mlx.mlx_astype(&dtb, dt_bias, .float32, s));
    const lbv = [1]f32{lower_bound};
    const lb = mlx.mlx_array_new_data(&lbv, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(lb);
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const t_arr = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t, h, head_dim }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, h, head_dim, head_dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, chunk_threads * chunkGroups(h), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, chunk_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dt));
    inline for (.{ .{ "Dk", head_dim }, .{ "Dv", head_dim }, .{ "H", h }, .{ "TAPS", taps } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ qkv, before, conv_w, q_scale, k_scale, eps_arr, a_c, beta_c, alog, dtb, lb, state, t_arr };
    var outs: [2]Array = undefined;
    try applyKernel(s, try kernelOf(&fold_kernel, "turboexl3_glm53f_kda_fold", &.{ "qkv", "prev", "w", "q_scale", "k_scale", "eps", "a", "beta", "A_log", "dt_bias", "lower_bound", "state_in", "T" }, &.{ "y", "state_out" }, fold_source, chunk_header), cfg, &ins, &outs);
    return .{ .y = outs[0], .state = outs[1] };
}

// ── the gated norm ──

/// `TURBOEXL3_GLM53F_KDA_NORMGATE=1`, set by the plugin at load, before any forward.
pub var normgate_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const normgate_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_NORMGATE=1: a KDA layer's prompt gated norm as one text (the copy's norm-gate text: rms_norm.metal `rms_single_row` over each 128-wide value head, MLX's own sigmoid tabulated over every 16-bit value, the multiply in the activation dtype), straight to the out projection's input\n";

/// The copy's norm-gate text (sdk/blocks/gdn/decode_fused.zig builds the same from it, with no header).
const normgate_source = kernel_text.source("kernels/gdn/gdn_normgate.metal");

/// Whether the norm-gate text takes y of `ysh` and the gate input of `zsh` at `hv` value heads of `dv`: y [B, T, hv,
/// 128] (rms_single_row's 32 lanes x 4 reads a head) and the gate input [B, T, hv x 128] (the heads' columns in order).
pub fn normGateServes(ysh: []const c_int, zsh: []const c_int, hv: c_int, dv: c_int) bool {
    if (dv != head_dim or hv < 1 or ysh.len != 4 or zsh.len != 3) return false;
    if (ysh[0] < 1 or ysh[1] < 1 or ysh[2] != hv or ysh[3] != dv) return false;
    return zsh[0] == ysh[0] and zsh[1] == ysh[1] and zsh[2] == hv * dv;
}

var normgate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var normgate_engaged = false;
/// Launches since the load (src/kda_gpu_test.zig reads that the text ran).
pub var normgate_runs: u64 = 0;

/// kda.zig's gated norm (`rms_norm(y, norm_w)` at the chain's epsilon, `sigmoid(z)`, their product) as one text: [B, T,
/// hv x dv] in y's dtype, the out projection's input, a new array. Null with the switch off, on a shape the text does
/// not take, or unless y, z and the norm weight share one 16-bit dtype (else MLX promotes the chain to f32): the caller
/// runs the chain.
pub fn normGate(s: mlx.mlx_stream, y: Array, z: Array, norm_w: Array, eps: f32, hv: c_int, dv: c_int) !?Array {
    if (!normgate_on) return null;
    const out = (try normGateText(s, y, z, norm_w, eps, hv, dv)) orelse return null;
    normgate_runs += 1;
    if (!normgate_engaged) {
        normgate_engaged = true;
        const ysh = mlx.getShape(y);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_NORMGATE engaged: {d} rows x {d} value heads of {d} in one text, the sigmoid tabulated\n", .{ ysh[0] * ysh[1], hv, dv });
    }
    return out;
}

/// `normGate`'s text, whichever pass asks (`normGate` in a prompt pass, src/kda_verify.zig in a verify); null on a shape
/// or dtype the text does not take (the caller runs the chain).
pub fn normGateText(s: mlx.mlx_stream, y: Array, z: Array, norm_w: Array, eps: f32, hv: c_int, dv: c_int) !?Array {
    const ysh = mlx.getShape(y);
    if (!normGateServes(ysh, mlx.getShape(z), hv, dv)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(z) != dt or mlx.mlx_array_dtype(norm_w) != dt) return null;
    if (mlx.mlx_array_size(norm_w) != @as(usize, @intCast(dv))) return null;
    const rows = ysh[0] * ysh[1];
    // MLX's sigmoid of every 16-bit value of the dtype, made by the op the chain runs, evaluated once
    const table = try fused_activations.sigmoidTableFor(s, dt);
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ysh[0], ysh[1], hv * dv }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, rows, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    // the gate input's rows are its own (no offset), the table's sigmoid, KDA's `y * sigmoid(z)` (not the swish)
    inline for (.{ .{ "HV", hv }, .{ "DV", dv }, .{ "ZSTRIDE", hv * dv }, .{ "ZOFF", 0 }, .{ "TAB", 1 }, .{ "SWISH", 0 } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    const ins = [_]Array{ y, z, norm_w, eps_arr, table };
    var outs: [1]Array = undefined;
    try applyKernel(s, try kernelOf(&normgate_kernel, "turboexl3_glm53f_kda_normgate", &.{ "y", "z", "norm_w", "eps", "sigtab" }, &.{"out"}, normgate_source, ""), cfg, &ins, &outs);
    return outs[0];
}

// ── the kernels ──

/// The kernel object in `slot`, built once (its library compiles at the first launch); src/kda_verify.zig's folded
/// launch builds on it too.
pub fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// One launch: `outs` (as many as the kernel's outputs) new arrays.
pub fn applyKernel(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array, outs: []Array) !void {
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

// ── tests (MLX-free) ──

const testing = std.testing;

test "turboexl3_glm53f stream kda prompt: MLX's float promotion, which the gate text's add takes: the dtype when both are it, else f32; no other dtype" {
    try testing.expectEqual(@as(?Dtype, .bfloat16), promote(.bfloat16, .bfloat16));
    try testing.expectEqual(@as(?Dtype, .float16), promote(.float16, .float16));
    // GLM-5.3-Flash's gate input (bf16, the low-rank second stage's) and dt_bias (f32 in every SC pack)
    try testing.expectEqual(@as(?Dtype, .float32), promote(.bfloat16, .float32));
    try testing.expectEqual(@as(?Dtype, .float32), promote(.float32, .bfloat16));
    try testing.expectEqual(@as(?Dtype, .float32), promote(.bfloat16, .float16));
    try testing.expectEqual(@as(?Dtype, .float32), promote(.float16, .float32));
    try testing.expectEqual(@as(?Dtype, null), promote(.int32, .float32));
    try testing.expectEqual(@as(?Dtype, null), promote(.bfloat16, .uint32));
}

test "turboexl3_glm53f stream kda prompt: the gate text takes GLM-5.3-Flash's prompt rows of 64 x 128 gate channels and refuses any other layout" {
    // a 16K prompt's main call and its 31-row tail, and the warm-up's 8 and 32 rows
    for ([_]c_int{ 16356, 31, 8, 32 }) |rows| {
        try testing.expectEqual(@as(?GateShape, .{ .rows = rows, .n = 8192 }), gateServes(&.{ 1, rows, 8192 }, 8192, 8192));
    }
    // two slots, a 2-D input, a table of another length, a row of part of a threadgroup
    try testing.expectEqual(@as(?GateShape, null), gateServes(&.{ 2, 16, 8192 }, 8192, 8192));
    try testing.expectEqual(@as(?GateShape, null), gateServes(&.{ 16, 8192 }, 8192, 8192));
    try testing.expectEqual(@as(?GateShape, null), gateServes(&.{ 1, 16, 8192 }, 64, 8192));
    try testing.expectEqual(@as(?GateShape, null), gateServes(&.{ 1, 16, 8192 }, 8192, 64));
    try testing.expectEqual(@as(?GateShape, null), gateServes(&.{ 1, 16, 8192 + 128 }, 8192 + 128, 8192 + 128));
}

test "turboexl3_glm53f stream kda prompt: the prologue text takes GLM-5.3-Flash's qkv rows (64 heads of 128, 4 taps) and refuses any other layout" {
    try testing.expectEqual(@as(?ProShape, .{ .rows = 16356, .cdim = 24576 }), prologueServes(&.{ 1, 16356, 24576 }, 64, 128, 64, 128, 4));
    // a cold 1-row prompt still reads 3 window rows: the text takes any row count
    try testing.expectEqual(@as(?ProShape, .{ .rows = 1, .cdim = 24576 }), prologueServes(&.{ 1, 1, 24576 }, 64, 128, 64, 128, 4));
    // rms_single_row's 32 x 4 lanes are a 128-wide head; heads alike; [q | k | v] wide; one slot
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 1, 16, 3 * 64 * 64 }, 64, 64, 64, 64, 4));
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 1, 16, 24576 }, 64, 128, 32, 128, 4));
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 1, 16, 24575 }, 64, 128, 64, 128, 4));
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 2, 16, 24576 }, 64, 128, 64, 128, 4));
    // whole threadgroups of 8 (section, head) groups: 3 x 2 heads is not
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 1, 16, 3 * 2 * 128 }, 2, 128, 2, 128, 4));
    try testing.expectEqual(@as(?ProShape, null), prologueServes(&.{ 1, 16, 24576 }, 64, 128, 64, 128, 1));
}

test "turboexl3_glm53f stream kda prompt: the per-core plan carries GLM-5.3-Flash's 64 x 128 value rows on 40 cores as 80 threadgroups x 104 rows, two a core, where the copy's plan declines" {
    // the 1086 tok/s cell's plan (mlx-stream cf30601 kda_percore.zig's own cases)
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 80, .rows = 104, .per_core = 2 }), percorePlan(64, 128, 40));
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 80, .rows = 104, .per_core = 1 }), percorePlan(64, 128, 80));
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 64, .rows = 128, .per_core = 1 }), percorePlan(64, 128, 64));
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 76, .rows = 108, .per_core = 1 }), percorePlan(64, 128, 76));
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 64, .rows = 128, .per_core = 2 }), percorePlan(64, 128, 32));
    try testing.expectEqual(@as(?PercorePlan, .{ .ntg = 76, .rows = 108, .per_core = 2 }), percorePlan(64, 128, 38));
    // three a core would be needed (past the cap), and no cores
    try testing.expectEqual(@as(?PercorePlan, null), percorePlan(64, 128, 30));
    try testing.expectEqual(@as(?PercorePlan, null), percorePlan(64, 128, 0));
    // the copy's own plan (one a core) declines the 40-core M5 Max, which is why the copy runs the per-token route
    try testing.expect(@import("mlx_serve/sdk/blocks/kda/recurrence.zig").planFor(64, 128, 40) == null);
    // every plan covers the rows in even ranges of at most percore_max_rows, two a core only past one a core's rows
    for ([_]c_int{ 32, 34, 36, 38, 40, 48, 56, 64, 72, 80, 96, 128 }) |cores| {
        const p = percorePlan(64, 128, cores).?;
        try testing.expect(p.ntg * p.rows >= 64 * 128 and @rem(p.rows, 2) == 0 and p.rows <= percore_max_rows);
        try testing.expectEqual(@as(c_int, if (cores * percore_max_rows >= 64 * 128) 1 else 2), p.per_core);
    }
}

test "turboexl3_glm53f stream kda prompt: the norm-gate text takes GLM-5.3-Flash's y of 64 value heads of 128 beside its gate input, and refuses any other layout" {
    // a 16K prompt's main call and its 31-row tail, and the warm-up's 8 rows
    for ([_]c_int{ 16356, 31, 8 }) |rows| try testing.expect(normGateServes(&.{ 1, rows, 64, 128 }, &.{ 1, rows, 8192 }, 64, 128));
    try testing.expect(normGateServes(&.{ 2, 5, 64, 128 }, &.{ 2, 5, 8192 }, 64, 128));
    // heads of 64 (not 32 lanes x 4), another head count, a gate input of other rows or width, a flat y, no rows
    try testing.expect(!normGateServes(&.{ 1, 16, 128, 64 }, &.{ 1, 16, 8192 }, 128, 64));
    try testing.expect(!normGateServes(&.{ 1, 16, 64, 128 }, &.{ 1, 16, 8192 }, 32, 128));
    try testing.expect(!normGateServes(&.{ 1, 16, 64, 128 }, &.{ 1, 15, 8192 }, 64, 128));
    try testing.expect(!normGateServes(&.{ 1, 16, 64, 128 }, &.{ 1, 16, 8192 + 128 }, 64, 128));
    try testing.expect(!normGateServes(&.{ 1, 16, 8192 }, &.{ 1, 16, 8192 }, 64, 128));
    try testing.expect(!normGateServes(&.{ 1, 0, 64, 128 }, &.{ 1, 0, 8192 }, 64, 128));
}

test "turboexl3_glm53f stream kda prompt: the chunkwise kernel takes GLM-5.3-Flash's 64 heads of 128 at its gate bound of -5 as 128 threadgroups, and refuses another layout or a bound past -5" {
    try testing.expect(chunkLayoutServes(64, 64, 128, 128));
    try testing.expectEqual(@as(c_int, 128), chunkGroups(64));
    // its threadgroups cover the value rows: 64 value rows each, 8 simdgroups of 8
    try testing.expectEqual(@as(c_int, 64 * 128), chunkGroups(64) * chunk_value_rows);
    try testing.expectEqual(chunk_threads, @divExact(chunk_value_rows, 8) * 32);
    // grouped heads, heads of 64, value heads of 64, no heads
    try testing.expect(!chunkLayoutServes(32, 64, 128, 128));
    try testing.expect(!chunkLayoutServes(64, 64, 64, 128));
    try testing.expect(!chunkLayoutServes(64, 64, 128, 64));
    try testing.expect(!chunkLayoutServes(0, 0, 128, 128));
    // GLM-5.3-Flash's bound (config.zig's default and the SC packs'), and gentler ones
    for ([_]f32{ -5.0, -4.0, -1.0, -0.25 }) |lb| try testing.expect(chunkBoundServes(lb));
    // past the kernel's range (a chunk's factors past e^40), no bound (the softplus gate), a positive one, NaN
    for ([_]f32{ -5.001, -6.0, -10.0, 0.0, 1.0, std.math.nan(f32) }) |lb| try testing.expect(!chunkBoundServes(lb));
}

test "turboexl3_glm53f stream kda prompt: every switch is off until the plugin sets it" {
    try testing.expect(!prologue_on and !percore_on and !normgate_on and !chunk_on and !fold_on);
}
