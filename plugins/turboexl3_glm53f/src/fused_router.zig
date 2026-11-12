//! `TURBOEXL3_GLM53F_ROUTER=1` (opt-in): a routed layer's router in a decode call (2 to 4 rows: an MTP or prompt-lookup
//! verify) in two launches where the stock route spends four, its routes and weights the stock route's bit for bit.
//!
//! The stock route (`moe.mlp`: `routerLogits`, then `route`) at those rows:
//! - x (bf16) cast to f32, then `mlx_matmul(x, W)` over the router's f32 copy (a transposed view of the stored bf16
//!   `[288, 4096]`, widened). With TF32 on (MLX's default) MLX takes its split-K NAX GEMM,
//!   `steel_gemm_splitk_nax_nt_float32_float32_bm64_bn64_bk256_wm2_wn2`: K split at 2048, one chain a 32-expert block and
//!   a partition, each 64 dependent load-then-MMA steps; then MLX's split-K accumulate.
//! - `router_kernel.moeRouterTopK` (`.sigmoid_bias`): sigmoid, + the correction bias, the top 8 by the lowest-index
//!   tie-break, the weights normalized and scaled, stored bf16.
//!
//! Here:
//! - phase 1 (kernels/router_p1.metal, over MLX's own NAX tile matmul: the kit's steel headers as lane_qmm's
//!   `prefillHeader` composes them): the same 18 chains, each the same `matmul2d` op on the same operands in the same k
//!   order, its weight reads prefetched by the whole threadgroup, the partials P[2][rows][288] f32 out;
//! - phase 2: the stock router kernel's own text (`router_kernel.ROUTER_SOURCES`, sigmoid_bias) with its logits read as
//!   (0 + P0) + P1, MLX's split-K accumulate (`out = 0; out += C0; out += C1`).
//! The operands are the stock route's numbers: x is bf16 and the weight f16 (the pack's) or bf16, and their f32 copies
//! are their exact widenings, so the weight is read as stored (the raw router the bank's lookahead also reads: half the
//! f32 copy's bytes) and widened in the fragment.
//!
//! Any call outside that plan routes as before (null): fewer than 2 or more than 4 rows (the MTP head's 1 row, a prompt
//! pass), a layer whose raw router is not registered (`register`, at the bank's open), another hidden, expert count,
//! router layout, routing mode or dtype, or MLX's f32 matmul off its TF32 NAX route.
//!
//! `TURBOEXL3_GLM53F_PREDICT=1` (opt-in, its own switch): the decode lookahead's next-layer scores (`predict`, from the
//! bank's predictor function, glm5n/experts.zig `scoresBf16`, which every caller of the bank's `scoresOf` reaches) in
//! two launches where the stock predictor spends six. The stock predictor multiplies bf16 x by the stored router's
//! transpose: MLX promotes bf16 and f16 to f32, so it casts x, copies the whole router to f32 (2.4 MB read, 4.7 MB
//! written a call), runs the same split-K NAX GEMM (the copy keeps the transposed layout: transpose_b) and its
//! accumulate, then MLX's Sigmoid and the correction add, f32 [rows, 288]. Here: phase 1 as the router's over the next
//! layer's router as stored, then the stock router text's sigmoid (MLX's Sigmoid, `metal::precise::exp`) over
//! (0 + P0) + P1 and the correction added, in one launch. The scores pick the lookahead's reads only; they are held to
//! the stock predictor's bit for bit, so the same records are read ahead. f16 routers only (the pack's): with a bf16
//! router the stock matmul stays bf16 (MLX's bf16 GEMM, its logits rounded to bf16), which phase 1 is not, so such a
//! call scores as before (`not_f16`).
//!
//! `TURBOEXL3_GLM53F_ROUTER_ONE=1` (opt-in, with ROUTER or PREDICT): each of their calls in one launch where it spends
//! two, at GLM-5.3-Flash's dims (K 4096, N 288: the plan's 18 chains; compile-time constants, the calls the plan takes).
//! Phase 1's text runs as it is but for its partial's store, an atomic store of the same f32 bits (device-coherent).
//! Every threadgroup then fences its stores at device scope and counts its arrival on the call's own counter, a fresh
//! zeroed array a call (kernels/router_one_arrive.metal); the one that counts last runs phase 2 and the other 17 return.
//! Phase 2 reads the partials device-coherently and runs the stock texts' statements: the router's score loop over every
//! row's (row, expert) pairs, then each row's selection on its own simdgroup (kernels/router_one_route.fmt.metal: the
//! stock text with its read and its indexing of rk, rw and sel by a row's slice edited, each edit found once); the
//! predictor's sigmoid and correction a pair (kernels/router_one_predict.fmt.metal). The same chains, sums, sigmoids
//! and selections in the same order: the same routes, weights and scores (src/router_gpu_test.zig holds them to the stock
//! route's and predictor's bit for bit). A call the plan does not take routes and scores as before.
//!
//! `TURBOEXL3_GLM53F_ROUTER_ONE1=1` (opt-in, with ROUTER): a registered layer's one-row call (the MTP head's one-row
//! draft steps with MTP_ROUTER; a trunk call of one row) in one launch. The stock route at one row casts x to f32, runs
//! MLX's gemv over the router's f32 copy (`gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0`: 18 threadgroups of four
//! simdgroups, four outputs a simdgroup, each a lane's chain of multiply-adds over k and a shuffle-down tree), then the
//! stock router kernel. Here each of the 18 threadgroups runs MLX's own `GEMVKernel::run` at those template arguments
//! (MLX's gemv.h, vendored unchanged in src/vendor/mlx/, four edits made at compile time, each found once: the two loaders
//! and `run` take the operands' stored types, x bf16 and the raw router f16 or bf16, widened to f32 by the loaders' own
//! casts, the exact values of the stock cast and copy; each logit stored device-coherently), then ROUTER_ONE's arrival,
//! and the threadgroup that counts last runs the stock router text with its read device-coherent. The same loads,
//! multiply-adds, lane reductions, sigmoids and selections in the same order: the same logits, routes and weights
//! (src/router_gpu_test.zig holds all three to the stock route's bit for bit). Any other call routes as before.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const moe = @import("mlx_serve/sdk/bases/blocks/moe.zig");
const router_kernel = @import("mlx_serve/sdk/blocks/moe/router_kernel.zig");
const routing = @import("mlx_serve/sdk/blocks/moe/routing.zig");
const device = @import("mlx_serve/mlx/device.zig");
const lane_qmm = @import("mlx_serve/sdk/blocks/qmm/lane_qmm.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_ROUTER=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER=1: a routed layer's router in a 2- to 4-row decode call (an MTP or prompt-lookup verify) in two launches: MLX's split-K NAX chains (the same matmul2d op on the same operands in the same k order, the weight read as stored) and the stock router text over their partials summed as MLX's split-K accumulate sums them, so the same routes and weights; any other call routes as before\n";

/// `TURBOEXL3_GLM53F_PREDICT=1`, set by the plugin at load, before any forward.
pub var predict_on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
/// `TURBOEXL3_GLM53F_ROUTER_ONE=1`: ROUTER's and PREDICT's calls each in one launch (the module doc).
pub var one_on: bool = false;
/// Its load line.
pub const one_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE=1: ROUTER's and PREDICT's two launches as one at GLM-5.3-Flash's dims (K 4096, N 288): phase 1's chains as they are, the partials stored device-coherently, and the threadgroup that arrives last (a counter a call) runs phase 2 over them, a row a simdgroup for the router: the same chains, sums, sigmoids and selections in the same order, so the same routes, weights and scores\n";
/// `TURBOEXL3_GLM53F_ROUTER_ONE1=1` (opt-in, with ROUTER): a registered layer's one-row call in one launch (the MTP
/// head's one-row draft step with MTP_ROUTER), MLX's own gemv kernel text (src/vendor/mlx/gemv.h) as its first phase.
pub var one1_on: bool = false;
/// Its load line.
pub const one1_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE1=1: a registered layer's one-row router call in one launch at GLM-5.3-Flash's dims (K 4096, N 288): MLX's own gemv kernel (its vendored text: x and the raw router read as stored and widened as the stock cast and f32 copy widen them, each logit stored device-coherently), and the stock router text in the threadgroup that arrives last: the same loads, multiply-adds, lane reductions, sigmoids and selections, so the same routes and weights\n";
/// `TURBOEXL3_GLM53F_MTP_ROUTER=1` (opt-in, with ROUTER), set by the plugin at load, before the model loads: the MTP
/// head's raw router is loaded with the head (patches/turboexl3-model.patch, only with this
/// switch) and registered (`registerHead`), so a draft step of 2 to 4 rows (step 0) routes through the plan as a verify's
/// layer does (ROUTER_ONE's one launch when on), the same routes and weights; its routes are counted apart
/// (`head_routes`), so the trunk's per-layer counts stay one a verify.
pub var mtp_on: bool = false;
/// Its load line.
pub const mtp_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_MTP_ROUTER=1: the MTP head's router registered with ROUTER's plan (its raw router loaded with the head): a 2- to 4-row draft step routes as a verify's layer does, the same routes and weights; its routes counted apart (head_routes)\n";
pub const predict_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_PREDICT=1: the decode lookahead's next-layer scores in a 2- to 4-row call over an f16 router in two launches: MLX's split-K NAX chains over the next layer's router as stored (no f32 copy of it) and MLX's sigmoid and the correction add over their partials summed as MLX's split-K accumulate sums them, so the same scores and the same reads ahead; any other call scores as before\n";

/// GLM-5.3-Flash's router, the geometry the stock route's MLX plan is read for: the hidden (K 4096: MLX splits it into
/// two partitions of 2048 at K <= 4096) and the routed experts (whole 32-expert blocks).
pub const hidden: c_int = 4096;
pub const experts: c_int = 288;
pub const partition: c_int = 2048;
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 4;
/// Phase 1's threadgroup (kernels/router_p1.metal: 32 experts by 64 k a stage, 16 B a thread).
pub const p1_threads: c_int = 256;
pub const p1_groups: c_int = @divExact(experts, 32) * @divExact(hidden, partition);

// ── the texts ──

/// Phase 1's text; its header is MLX's NAX tile matmul as the kit ships it (`mlx_steel_sources`: steel's defines,
/// type traits, integral constants and gemm/nax.h of the server's MLX), the one lane_qmm's prefill kernel composes.
pub const p1_source: [:0]const u8 = @embedFile("kernels/router_p1.metal");

/// The stock router text's logits read and phase 2's.
pub const stock_read = "float l = float(logits[rbase + e]);";
pub const sum_read = "float l = (0.0f + float(logits[rbase + e])) + float(logits[PSTRIDE + rbase + e]);";
/// Phase 2: the stock router text (sigmoid_bias) with its one logits read summing the partials.
pub const p2_source: [:0]const u8 = replaceOnce(router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias)], stock_read, sum_read);

/// The predictor's phase 2: the stock router text's sigmoid (router_sigmoid_score.metal, MLX's `Sigmoid` with
/// `metal::precise::exp`) over the summed partials, then the correction added as MLX's add adds it.
pub const predict_score_text = @import("mlx_serve/mlx/kernels.zig").source("kernels/moe/router_sigmoid_score.metal");
pub const predict_p2_source: [:0]const u8 = std.fmt.comptimePrint("{s}{s}{s}", .{
    "uint e = thread_position_in_grid.x;\nuint row = thread_position_in_grid.y;\nsize_t rbase = (size_t)row * (size_t)NE;\n",
    replaceOnce(predict_score_text, stock_read, sum_read),
    "\n  scores[rbase + e] = sg + correction[e];\n",
});

fn replaceOnce(comptime s: []const u8, comptime old: []const u8, comptime new: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, s, old) orelse @compileError("fused_router: the stock text has no `" ++ old ++ "`");
    if (std.mem.indexOfPos(u8, s, i + old.len, old) != null) @compileError("fused_router: the stock text reads its logits twice");
    return std.fmt.comptimePrint("{s}{s}{s}", .{ s[0..i], new, s[i + old.len ..] });
}

/// Each (old, new) of `edits` in turn, each `old` found exactly once (`replaceOnce`).
fn editsOnce(comptime s: []const u8, comptime edits: []const [2][]const u8) [:0]const u8 {
    @setEvalBranchQuota(1_000_000);
    comptime var out: [:0]const u8 = std.fmt.comptimePrint("{s}", .{s});
    inline for (edits) |e| out = replaceOnce(out, e[0], e[1]);
    return out;
}

/// `s` between `open` and `close` (each found exactly once, `close` after `open`).
fn between(comptime s: []const u8, comptime open: []const u8, comptime close: []const u8) []const u8 {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, s, open) orelse @compileError("fused_router: the stock text has no `" ++ open ++ "`");
    if (std.mem.indexOfPos(u8, s, i + open.len, open) != null) @compileError("fused_router: the stock text holds `" ++ open ++ "` twice");
    const j = std.mem.indexOfPos(u8, s, i + open.len, close) orelse @compileError("fused_router: the stock text has no `" ++ close ++ "` after `" ++ open ++ "`");
    if (std.mem.indexOfPos(u8, s, j + close.len, close) != null) @compileError("fused_router: the stock text holds `" ++ close ++ "` twice");
    return s[i + open.len .. j];
}

// ── TURBOEXL3_GLM53F_ROUTER_ONE's texts: a call in one launch ──

/// Phase 1's partial store, and the one launch's: an atomic store of the same f32 bits (device-coherent).
pub const p1_store = "out[row * N + nn * 16 + sc.x + j] = Dtile.frag_at(0, nn)[ii * 4 + j];";
pub const p1_store_one = "atomic_store_explicit(reinterpret_cast<device atomic_uint*>(out + row * N + nn * 16 + sc.x + j), as_type<uint>(Dtile.frag_at(0, nn)[ii * 4 + j]), memory_order_relaxed);";
/// The arrival: the stores fenced, the group counted, the last group on (kernels/router_one_arrive.metal).
pub const one_arrive: []const u8 = @embedFile("kernels/router_one_arrive.metal");
/// Phase 1 with its partial stored device-coherently, then the arrival: both one-launch texts open with it.
pub const p1_one: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk std.fmt.comptimePrint("{s}\n{s}", .{ replaceOnce(p1_source, p1_store, p1_store_one), one_arrive });
};
/// Phase 2's read in the one launch: the same (0 + P0) + P1, its partials staged in registers (`lp0`, `lp1`).
pub const one_read = "float l = (0.0f + lp0[k]) + lp1[k];";
/// The stock router text's (phase 2's) score loop and its selection, the pieces the one launch's phase 2 runs.
pub const route_loop_open = "for (uint e = tid; e < NE; e += TG) {\n";
pub const route_loop_close = "\n}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n";
pub const route_select_open = "if (tid < 32) {";
pub const route_body: []const u8 = between(p2_source, route_loop_open, route_loop_close);
pub const route_select: []const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    const i = std.mem.indexOf(u8, p2_source, route_select_open) orelse @compileError("fused_router: the stock router text has no selection");
    if (std.mem.indexOfPos(u8, p2_source, i + 1, route_select_open) != null) @compileError("fused_router: the stock router text selects twice");
    break :blk p2_source[i..];
};
/// The one launch's edits to the stock loop body: its read, and rk and rw at the pair's index (a row's slice).
pub const route_body_edits = [_][2][]const u8{
    .{ sum_read, one_read },
    .{ "rk[e] = ", "rk[i] = " },
    .{ "rw[e] = ", "rw[i] = " },
};
/// The one launch's edits to the stock selection: a row a simdgroup, rk, rw and sel at the row's slice.
pub const route_select_edits = [_][2][]const u8{
    .{ route_select_open, "if (row < uint(M)) {" },
    .{ "? rk[e] :", "? rk[rbase + e] :" },
    .{ "sel[r] = gidx;", "sel[row * NK + r] = gidx;" },
    .{ "myidx = sel[lane];", "myidx = sel[row * NK + lane];" },
    .{ "w = rw[myidx];", "w = rw[rbase + myidx];" },
};
pub const route_body_one: [:0]const u8 = editsOnce(route_body, &route_body_edits);
pub const route_select_one: [:0]const u8 = editsOnce(route_select, &route_select_edits);
/// The router's one launch: phase 1, the arrival, phase 2 (kernels/router_one_route.fmt.metal over the edited pieces).
pub const one_route_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk std.fmt.comptimePrint("{s}\n" ++ @embedFile("kernels/router_one_route.fmt.metal"), .{ p1_one, route_body_one, route_select_one });
};
/// The predictor's one launch: phase 1, the arrival, the stock sigmoid over the staged sum and the correction added
/// (kernels/router_one_predict.fmt.metal).
pub const predict_score_one: [:0]const u8 = replaceOnce(predict_score_text, stock_read, one_read);
pub const one_predict_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(1_000_000);
    break :blk std.fmt.comptimePrint("{s}\n" ++ @embedFile("kernels/router_one_predict.fmt.metal"), .{ p1_one, predict_score_one });
};
/// The one launch's header: phase 1's (the kit's NAX tile matmul) and the coherent partial read
/// (kernels/router_one_header.metal).
pub const one_header_text: []const u8 = @embedFile("kernels/router_one_header.metal");

comptime {
    // phase 2 runs on phase 1's threadgroup: the stock router's own launch at 288 experts is 256 threads too
    if (router_kernel.moeRouterThreadGroup(experts) != p1_threads) @compileError("fused_router: the stock router's threadgroup at 288 experts is not phase 1's 256 threads");
}

// ── TURBOEXL3_GLM53F_ROUTER_ONE1's texts: a one-row call in one launch ──

/// MLX's gemv.h: ml-explore/mlx c69303b00f5f's mlx/backend/metal/kernels/gemv.h, copied unchanged by
/// scripts/sync-mlx-gemv.sh (NOTICE: MLX). The stock route's one-row logits are its `gemv` kernel at
/// bm4_bn1_sm1_sn32_tm4_tn4, no axpby, over the router's f32 copy.
pub const gemv_h: []const u8 = @embedFile("vendor/mlx/gemv.h");
/// The piece the one launch compiles: from MLX's constant macro to the vector-matrix banner (the default accumulator
/// and GEMVKernel). Its includes are MLX's utils.h, which MLX puts before every custom kernel's source, and steel's
/// broadcast helpers, which GEMVKernel does not call.
pub const gemv_open = "#define MLX_MTL_CONST static constant constexpr const\n";
pub const gemv_close = "///////////////////////////////////////////////////////////////////////////////\n/// Vector matrix multiplication\n";
/// The edits to GEMVKernel, each found once: the two loaders take their source's type as a template argument and
/// `run` takes the matrix's and the vector's (x read as stored, bf16, and the raw router as stored, f16 or bf16, each
/// widened to f32 by the loaders' own static_cast: the exact widening the stock cast of x and the router's f32 copy
/// make), and each output is stored device-coherently (an atomic store of the same f32 bits). Every other statement is
/// MLX's.
pub const gemv_edits = [_][2][]const u8{
    .{ "template <typename U = T>\n  static METAL_FUNC void\n  load_unsafe(const device T* src,", "template <typename U = T, typename S = T>\n  static METAL_FUNC void\n  load_unsafe(const device S* src," },
    .{ "template <typename U = T>\n  static METAL_FUNC void load_safe(\n      const device T* src,", "template <typename U = T, typename S = T>\n  static METAL_FUNC void load_safe(\n      const device S* src," },
    .{ "  static METAL_FUNC void run(\n      const device T* mat [[buffer(0)]],\n      const device T* in_vec [[buffer(1)]],", "  template <typename MT, typename VT>\n  static METAL_FUNC void run(\n      const device MT* mat [[buffer(0)]],\n      const device VT* in_vec [[buffer(1)]]," },
    .{ "out_vec[out_row + tm] = static_cast<T>(result[tm]);", "atomic_store_explicit(reinterpret_cast<device atomic_uint*>(out_vec + out_row + tm), as_type<uint>(static_cast<T>(result[tm])), memory_order_relaxed);" },
};
pub const gemv_kernel: [:0]const u8 = blk: {
    @setEvalBranchQuota(200_000_000);
    break :blk editsOnce(gemv_open ++ between(gemv_h, gemv_open, gemv_close), &gemv_edits);
};
/// The one-row launch's header: the edited GEMVKernel, the arguments MLX's gemv kernel binds as constant buffers
/// (kernels/router_one1_header.metal) and the coherent read (kernels/router_one_header.metal's `pload`).
pub const one1_header: [:0]const u8 = blk: {
    @setEvalBranchQuota(200_000_000);
    break :blk std.fmt.comptimePrint("using namespace metal;\n{s}\n{s}\n{s}", .{ gemv_kernel, @embedFile("kernels/router_one1_header.metal"), one_header_text });
};
/// Phase 2 at one row: the stock router text (sigmoid_bias) with its one logits read device-coherent.
pub const one1_read = "float l = pload(logits, rbase + e);";
pub const one1_route_text: [:0]const u8 = replaceOnce(router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias)], stock_read, one1_read);
/// The one-row launch (kernels/router_one1.fmt.metal over phase 2's text).
pub const one1_source: [:0]const u8 = blk: {
    @setEvalBranchQuota(200_000_000);
    break :blk std.fmt.comptimePrint(@embedFile("kernels/router_one1.fmt.metal"), .{one1_route_text});
};
/// MLX's gemv geometry at GLM-5.3-Flash's router: four simdgroups a threadgroup, four outputs a simdgroup, 16 a
/// threadgroup, so 18 threadgroups of 128 threads; phase 2's score loop runs over those 128.
pub const one1_threads: c_int = 128;
pub const one1_groups: c_int = @divExact(experts, 16);

// ── the registry: each routed layer's raw router ──

pub const max_layers = 64;
var keys: [max_layers]usize = @splat(0);
var raws: [max_layers]Array = undefined;
var n_layers: usize = 0;
/// The MTP head's index among the registered routers (`registerHead`; null: not registered).
var head_index: ?usize = null;

/// Routes taken since the load, by registered layer, and the calls that routed as before: a registered layer's 2- to
/// 4-row call this plan does not take (`declined`, logged once a reason), an unregistered one's (the MTP head's, at
/// those rows), and every call of other rows.
pub var routes_by_layer: [max_layers]u64 = @splat(0);
pub var declined: u64 = 0;
pub var unregistered: u64 = 0;
pub var other_rows: u64 = 0;

/// The predictor's calls since the load: scored by the plan, declined (2 to 4 rows, logged once a reason), other rows.
pub var predictions: u64 = 0;
/// ROUTER_ONE's calls: the routes and the predictions that ran in one launch.
pub var one_routes: u64 = 0;
/// The MTP head's routes that ran in one launch (ROUTER_ONE with MTP_ROUTER; `one_routes` holds the trunk's).
pub var one_head_routes: u64 = 0;
pub var one_predictions: u64 = 0;
/// ROUTER_ONE1's one-row calls: the trunk's and the MTP head's (`one1_head_routes`).
pub var one1_routes: u64 = 0;
pub var one1_head_routes: u64 = 0;
pub var predict_declined: u64 = 0;
pub var predict_other: u64 = 0;

pub const Reason = enum { route, top_k, x, table, bias, tf32, stream };
var reasons_logged: [@typeInfo(Reason).@"enum".field_names.len]bool = @splat(false);
var engaged = false;
pub const PredictReason = enum { x, weight, not_f16, aligned, correction, tf32, stream };
var predict_reasons_logged: [@typeInfo(PredictReason).@"enum".field_names.len]bool = @splat(false);
var predict_engaged = false;

/// The weight types phase 1 reads (2 bytes: its words are 8 values): the pack's f16 and bf16.
pub fn weightServes(dt: mlx.mlx_dtype) bool {
    return dt == .float16 or dt == .bfloat16;
}

/// Layer `w`'s raw router (f16 or bf16 `[288, 4096]` as stored, row-contiguous, 16-byte aligned: phase 1 reads it 16 B a
/// thread), borrowed for the model's life. A router not in that form is not registered (its layer routes as before) and
/// says so.
pub fn register(w: *const anyopaque, raw: Array) !void {
    if (n_layers == max_layers) return error.Glm53fRouterLayers;
    if (raw.ctx == null or !weightServes(mlx.mlx_array_dtype(raw)) or !shapeIs(mlx.getShape(raw), &.{ experts, hidden }) or !rowContiguous(raw)) {
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER: a routed layer's raw router is not f16 or bf16 [{d}, {d}] row-contiguous: that layer routes as before\n", .{ experts, hidden });
        return;
    }
    try mlx.check(mlx.mlx_array_eval(raw));
    const addr: usize = switch (mlx.mlx_array_dtype(raw)) {
        .float16 => @intFromPtr(mlx.mlx_array_data_float16(raw) orelse return error.Glm53fRouterData),
        else => @intFromPtr(mlx.mlx_array_data_bfloat16(raw) orelse return error.Glm53fRouterData),
    };
    if (addr % 16 != 0) {
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER: a routed layer's raw router is not 16-byte aligned: that layer routes as before\n", .{});
        return;
    }
    keys[n_layers] = @intFromPtr(w);
    raws[n_layers] = raw;
    n_layers += 1;
}

/// The MTP head's raw router (`TURBOEXL3_GLM53F_MTP_ROUTER`), registered as a layer's is (`register`: one not in that
/// form is not registered and says so) and marked the head's, so its routes are counted apart. After the trunk's layers.
pub fn registerHead(w: *const anyopaque, raw: Array) !void {
    if (head_index != null) return error.Glm53fRouterHeadTwice;
    const before = n_layers;
    try register(w, raw);
    if (n_layers == before + 1) head_index = before;
}

fn isHead(li: usize) bool {
    return if (head_index) |h| h == li else false;
}

/// Whether the MTP head's router is registered.
pub fn headRegistered() bool {
    return head_index != null;
}

/// The registry and the counts emptied (a new model's load).
pub fn reset() void {
    n_layers = 0;
    head_index = null;
    one_head_routes = 0;
    keys = @splat(0);
    routes_by_layer = @splat(0);
    declined = 0;
    unregistered = 0;
    other_rows = 0;
    predictions = 0;
    predict_declined = 0;
    predict_other = 0;
    one_routes = 0;
    one_predictions = 0;
    one1_routes = 0;
    one1_head_routes = 0;
}

fn find(w: *const anyopaque) ?usize {
    const k = @intFromPtr(w);
    for (keys[0..n_layers], 0..) |x, i| if (x == k) return i;
    return null;
}

pub fn registered() usize {
    return n_layers;
}

/// The counts' line at a request's end: every registered layer's routes (a verify routes each layer once, so equal
/// counts mean every layer of every verify took the plan) and the calls that routed as before.
pub fn summary(buf: []u8) []const u8 {
    var lo: u64 = std.math.maxInt(u64);
    var hi: u64 = 0;
    var total: u64 = 0;
    var trunk: usize = 0;
    for (routes_by_layer[0..n_layers], 0..) |c, i| {
        if (head_index) |h| if (h == i) continue;
        lo = @min(lo, c);
        hi = @max(hi, c);
        total += c;
        trunk += 1;
    }
    if (trunk == 0) lo = 0;
    const line_ = std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER routes={d} layers={d} per_layer_min={d} per_layer_max={d} declined={d} unregistered={d} other={d}", .{ total, trunk, lo, hi, declined, unregistered, other_rows }) catch return buf[0..0];
    // TURBOEXL3_GLM53F_MTP_ROUTER: the head's routes apart
    var tail: []const u8 = undefined;
    if (head_index) |h| {
        tail = std.fmt.bufPrint(buf[line_.len..], " head_routes={d}\n", .{routes_by_layer[h]}) catch return buf[0..0];
    } else {
        tail = std.fmt.bufPrint(buf[line_.len..], "\n", .{}) catch return buf[0..0];
    }
    return buf[0 .. line_.len + tail.len];
}

/// The predictor's counts' line at a request's end.
pub fn predictSummary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_PREDICT predictions={d} declined={d} other={d}\n", .{ predictions, predict_declined, predict_other }) catch buf[0..0];
}

/// ROUTER_ONE's counts' line at a request's end: the routes and predictions that ran in one launch (equal to ROUTER's
/// routes and PREDICT's predictions when every call the plan took ran so).
pub fn oneSummary(buf: []u8) []const u8 {
    if (head_index != null) return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE routes={d} predictions={d} head_routes={d}\n", .{ one_routes, one_predictions, one_head_routes }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE routes={d} predictions={d}\n", .{ one_routes, one_predictions }) catch buf[0..0];
}

/// ROUTER_ONE1's counts' line at a request's end: the one-row calls that ran in one launch, the head's apart.
pub fn one1Summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE1 routes={d} head_routes={d}\n", .{ one1_routes, one1_head_routes }) catch buf[0..0];
}

// ── the plan's checks (MLX-free over shapes, dtypes and strides) ──

fn shapeIs(sh: []const c_int, want: []const c_int) bool {
    return std.mem.eql(c_int, sh, want);
}

/// Row-contiguous: every dimension of more than one element at its row-major stride.
fn rowContiguous(a: Array) bool {
    const sh = mlx.getShape(a);
    return stridesRowMajor(sh, mlx.mlx_array_strides(a)[0..sh.len]);
}

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

/// The router's f32 copy as the stock matmul reads it: [4096, 288] with strides [1, 4096] (the transposed view of a
/// row-contiguous [288, 4096]), which MLX's matmul takes as transpose_b (the `nt` kernel phase 1 reproduces).
pub fn tableLayout(sh: []const c_int, st: []const usize) bool {
    return sh.len == 2 and sh[0] == hidden and sh[1] == experts and st[0] == 1 and st[1] == @as(usize, @intCast(hidden));
}

/// The rows of a call over `x` of shape `sh` (every leading dimension), or null when its last is not the hidden.
pub fn rowsOf(sh: []const c_int) ?c_int {
    if (sh.len < 2 or sh.len > 4 or sh[sh.len - 1] != hidden) return null;
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| rows *= d;
    return rows;
}

fn check(w: *const moe.Moe.Weights, s: mlx.mlx_stream, x: Array) ?Reason {
    const sb = switch (w.route) {
        .sigmoid_bias => |r| r,
        else => return .route,
    };
    if (sb.n_group > 1) return .route;
    if (w.top_k < 1 or w.top_k > 32) return .top_k;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or !rowContiguous(x)) return .x;
    const t = w.router.w;
    if (t.ctx == null or w.router.s.ctx != null or w.router.x3 != null or w.router.rht != null or w.router.ternary or w.router.bias.ctx != null) return .table;
    if (mlx.mlx_array_dtype(t) != .float32) return .table;
    const tsh = mlx.getShape(t);
    if (tsh.len != 2 or !tableLayout(tsh, mlx.mlx_array_strides(t)[0..2])) return .table;
    if (sb.expert_bias.ctx == null or mlx.mlx_array_dtype(sb.expert_bias) != .float32 or mlx.mlx_array_size(sb.expert_bias) != @as(usize, @intCast(experts))) return .bias;
    if (!device.tf32Matmul()) return .tf32;
    if (!mlx.streamIsGpu(s)) return .stream;
    return null;
}

// ── the launches ──

var p1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var p2_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Phase 1's config a weight type (0 f16, 1 bf16) and row count (its output shape and template are the rows').
var p1_cfgs: [2][@as(usize, @intCast(max_rows)) + 1]?mlx.mlx_fast_metal_kernel_config = @splat(@splat(null));

const P2Key = struct { dims: [4]c_int = @splat(0), len: u8 = 0, k: c_int = 0, norm: bool = false, scale: u32 = 0 };
const P2Slot = struct { key: P2Key = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null, scale_arr: Array = .{ .ctx = null } };
var p2_slots: [4]P2Slot = @splat(.{});
var p2_next: usize = 0;

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

/// Phase 1 alone: MLX's split-K partials of `x` (bf16 [.., rows, 4096], 2 to 4 rows) against `raw` (f16 or bf16
/// [288, 4096]), f32 [2, rows, 288] (src/router_gpu_test.zig holds their sum to MLX's matmul bit for bit).
pub fn partials(s: mlx.mlx_stream, x: Array, raw: Array, rows: c_int) !Array {
    if (rows < min_rows or rows > max_rows) return error.Glm53fRouterRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.Glm53fRouterWeightType;
    const ri: usize = @intCast(rows);
    const wi: usize = if (wt == .float16) 0 else 1;
    const slot = &p1_cfgs[wi][ri];
    if (slot.* == null) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @divExact(hidden, partition), rows, experts }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, p1_threads * p1_groups, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, p1_threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "WT", wt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PS", partition));
        slot.* = cfg;
    }
    const kern = try kernelOf(&p1_kernel, "turboexl3_glm53f_router_p1", &.{ "x", "w" }, &.{"p"}, p1_source, try lane_qmm.prefillHeader(), false);
    var outs: [1]Array = undefined;
    try applyKernel(s, kern, slot.*.?, &.{ x, raw }, &outs);
    return outs[0];
}

/// Phase 2 alone: the stock router text over `p` (phase 1's partials of a call over x of shape `xsh`), inds u32 and
/// scores bf16 of shape [.., rows, k] as the stock route's.
pub fn topK(s: mlx.mlx_stream, p: Array, xsh: []const c_int, bias: Array, k: c_int, norm: bool, scale: f32) !routing.MoeRouting {
    const rows = rowsOf(xsh) orelse return error.Glm53fRouterRows;
    var key: P2Key = .{ .len = @intCast(xsh.len), .k = k, .norm = norm, .scale = @bitCast(scale) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    const slot = for (&p2_slots) |*sl| {
        if (sl.cfg != null and std.meta.eql(sl.key, key)) break sl;
    } else blk: {
        const sl = &p2_slots[p2_next];
        p2_next = (p2_next + 1) % p2_slots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        if (sl.scale_arr.ctx != null) _ = mlx.mlx_array_free(sl.scale_arr);
        sl.* = .{};
        const tg = router_kernel.moeRouterThreadGroup(experts);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var out_shape: [4]c_int = undefined;
        for (xsh[0 .. xsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[xsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TOUT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NE", experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NORM", @intFromBool(norm)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PSTRIDE", rows * experts));
        sl.* = .{ .key = key, .cfg = cfg, .scale_arr = mlx.mlx_array_new_float(scale) };
        break :blk sl;
    };
    const kern = try kernelOf(&p2_kernel, "turboexl3_glm53f_router_p2", &.{ "logits", "bias", "scale" }, &.{ "inds", "scores" }, p2_source, "", true);
    var outs: [2]Array = undefined;
    try applyKernel(s, kern, slot.cfg.?, &.{ p, bias, slot.scale_arr }, &outs);
    return .{ .inds = outs[0], .norm_scores = outs[1] };
}

/// Layer `w`'s routes for the normed MoE input `x` by the plan, or null (off, or a call it does not take: the caller
/// routes as before). The arrays are new (the caller adopts them as the stock route's).
pub fn route(s: mlx.mlx_stream, w: *const moe.Moe.Weights, x: Array) !?routing.MoeRouting {
    if (!on) return null;
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse {
        other_rows += 1;
        return null;
    };
    // TURBOEXL3_GLM53F_ROUTER_ONE1: a registered layer's one-row call in one launch (the stock gemv takes no TF32
    // route, so only the plan's other checks apply)
    if (rows == 1 and one1_on) one1: {
        const li = find(w) orelse break :one1;
        if (check(w, s, x)) |why| if (why != .tf32 or !mlx.streamIsGpu(s)) break :one1;
        const sb = w.route.sigmoid_bias;
        const r = try routeOne1(s, x, raws[li], sb.expert_bias, w.top_k, sb.norm, sb.scale);
        // the logits are the launch's scratch: the graph keeps them until the launch has run
        _ = mlx.mlx_array_free(r.logits);
        if (isHead(li)) one1_head_routes += 1 else one1_routes += 1;
        if (!one1_engaged) {
            one1_engaged = true;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE1 engaged: a one-row router in one launch ({d} threadgroups, MLX's gemv kernel, the stock router text in the last)\n", .{one1_groups});
        }
        return r.routing;
    }
    if (rows < min_rows or rows > max_rows) {
        other_rows += 1;
        return null;
    }
    const li = find(w) orelse {
        unregistered += 1;
        return null;
    };
    if (check(w, s, x)) |why| {
        declined += 1;
        if (!reasons_logged[@backingInt(why)]) {
            reasons_logged[@backingInt(why)] = true;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER: a {d}-row call routes as before: {s}\n", .{ rows, @tagName(why) });
        }
        return null;
    }
    const sb = w.route.sigmoid_bias;
    const r = if (one_on) try routeOne(s, x, raws[li], sb.expert_bias, w.top_k, sb.norm, sb.scale) else two: {
        const p = try partials(s, x, raws[li], rows);
        defer _ = mlx.mlx_array_free(p);
        break :two try topK(s, p, xsh, sb.expert_bias, w.top_k, sb.norm, sb.scale);
    };
    routes_by_layer[li] += 1;
    if (one_on) {
        if (isHead(li)) one_head_routes += 1 else one_routes += 1;
        if (!one_route_engaged) {
            one_route_engaged = true;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE engaged: a {d}-row router in one launch ({d} threadgroups, phase 2 in the last)\n", .{ rows, p1_groups });
        }
    }
    if (!engaged) {
        engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER engaged: a {d}-row router over {d} experts of {d} (top {d}), {d} layers registered\n", .{ rows, experts, hidden, w.top_k, n_layers });
    }
    return r;
}

var pp2_kernel: ?mlx.mlx_fast_metal_kernel = null;
const PKey = struct { dims: [4]c_int = @splat(0), len: u8 = 0 };
const PSlot = struct { key: PKey = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null };
var pp2_slots: [4]PSlot = @splat(.{});
var pp2_next: usize = 0;

/// The predictor's phase 2 alone: the stock sigmoid and the correction over `p` (phase 1's partials of a call over x of
/// shape `xsh`), f32 of x's leading dimensions by 288, as the stock predictor's.
pub fn predictScores(s: mlx.mlx_stream, p: Array, xsh: []const c_int, correction: Array) !Array {
    const rows = rowsOf(xsh) orelse return error.Glm53fRouterRows;
    var key: PKey = .{ .len = @intCast(xsh.len) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    const slot = for (&pp2_slots) |*sl| {
        if (sl.cfg != null and std.meta.eql(sl.key, key)) break sl;
    } else blk: {
        const sl = &pp2_slots[pp2_next];
        pp2_next = (pp2_next + 1) % pp2_slots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        sl.* = .{};
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var out_shape: [4]c_int = undefined;
        for (xsh[0 .. xsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[xsh.len - 1] = experts;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, experts, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NE", experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PSTRIDE", rows * experts));
        sl.* = .{ .key = key, .cfg = cfg };
        break :blk sl;
    };
    const kern = try kernelOf(&pp2_kernel, "turboexl3_glm53f_predict_p2", &.{ "logits", "correction" }, &.{"scores"}, predict_p2_source, "", true);
    var outs: [1]Array = undefined;
    try applyKernel(s, kern, slot.cfg.?, &.{ p, correction }, &outs);
    return outs[0];
}

fn predictCheck(s: mlx.mlx_stream, x: Array, raw: Array, correction: Array) ?PredictReason {
    if (mlx.mlx_array_dtype(x) != .bfloat16 or !rowContiguous(x)) return .x;
    if (raw.ctx == null or !weightServes(mlx.mlx_array_dtype(raw)) or !shapeIs(mlx.getShape(raw), &.{ experts, hidden }) or !rowContiguous(raw)) return .weight;
    // only an f16 router makes the stock matmul (bf16 x by it) promote to f32, the TF32 GEMM phase 1 reproduces
    if (!predictPromotes(mlx.mlx_array_dtype(raw))) return .not_f16;
    const addr = @intFromPtr(mlx.mlx_array_data_float16(raw) orelse return .aligned);
    if (addr % 16 != 0) return .aligned;
    if (correction.ctx == null or mlx.mlx_array_dtype(correction) != .float32 or mlx.mlx_array_size(correction) != @as(usize, @intCast(experts)) or !rowContiguous(correction)) return .correction;
    if (!device.tf32Matmul()) return .tf32;
    if (!mlx.streamIsGpu(s)) return .stream;
    return null;
}

/// Whether the stock predictor's matmul of bf16 x by a router of type `dt` runs in f32 (MLX's promotion of bf16 and f16):
/// the TF32 split-K GEMM phase 1 reproduces. A bf16 router keeps it bf16.
pub fn predictPromotes(dt: mlx.mlx_dtype) bool {
    return dt == .float16;
}

/// The stock predictor's scores (`add(sigmoid(astype(x @ router^T, f32)), correction)`, f32 of x's leading dimensions
/// by 288) for the next layer's router as stored (`raw`), by the plan, or null (off, or a call it does not take: the
/// caller scores as before). The array is new (the caller adopts it).
pub fn predict(s: mlx.mlx_stream, x: Array, raw: Array, correction: Array) !?Array {
    if (!predict_on) return null;
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse {
        predict_other += 1;
        return null;
    };
    if (rows < min_rows or rows > max_rows) {
        predict_other += 1;
        return null;
    }
    if (predictCheck(s, x, raw, correction)) |why| {
        predict_declined += 1;
        if (!predict_reasons_logged[@backingInt(why)]) {
            predict_reasons_logged[@backingInt(why)] = true;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PREDICT: a {d}-row call scores as before: {s}\n", .{ rows, @tagName(why) });
        }
        return null;
    }
    const out = if (one_on) try predictOne(s, x, raw, correction) else two: {
        const p = try partials(s, x, raw, rows);
        defer _ = mlx.mlx_array_free(p);
        break :two try predictScores(s, p, xsh, correction);
    };
    predictions += 1;
    if (one_on) {
        one_predictions += 1;
        if (!one_predict_engaged) {
            one_predict_engaged = true;
            log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE engaged: a {d}-row lookahead in one launch ({d} threadgroups, phase 2 in the last)\n", .{ rows, p1_groups });
        }
    }
    if (!predict_engaged) {
        predict_engaged = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PREDICT engaged: a {d}-row lookahead over {d} experts of {d} ({s} router)\n", .{ rows, experts, hidden, @tagName(mlx.mlx_array_dtype(raw)) });
    }
    return out;
}

// ── TURBOEXL3_GLM53F_ROUTER_ONE's launches ──

var one_route_engaged = false;
var one_predict_engaged = false;
var one_route_kernel: ?mlx.mlx_fast_metal_kernel = null;
var one_predict_kernel: ?mlx.mlx_fast_metal_kernel = null;
var one_header: ?[:0]const u8 = null;

/// The one launch's header: phase 1's (lane_qmm's `prefillHeader`: the kit's NAX tile matmul) and the coherent read.
fn oneHeader() ![:0]const u8 {
    if (one_header) |h| return h;
    const a = std.heap.c_allocator;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, try lane_qmm.prefillHeader());
    try out.appendSlice(a, one_header_text);
    const h = try out.toOwnedSliceSentinel(a, 0);
    one_header = h;
    return h;
}

/// A call's arrival counter: a fresh zeroed array, the call's own (no two calls share one, so no reset and no order
/// between calls). 16 words: MLX binds an input of 8 or more elements in device memory, fewer in constant memory.
pub const counter_words = 16;
fn arrivals() !Array {
    const zeros: [counter_words]u32 = @splat(0);
    const a = mlx.mlx_array_new_data(&zeros, &[_]c_int{counter_words}, 1, .uint32);
    if (a.ctx == null) return error.Glm53fRouterOneCounter;
    return a;
}

/// Phase 1's template arguments and the one launch's shared ones: the weight's type, the rows, GLM-5.3-Flash's dims,
/// phase 2's expert count and threadgroup, and the partials' part stride.
fn oneTemplates(cfg: mlx.mlx_fast_metal_kernel_config, wt: mlx.mlx_dtype, rows: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "WT", wt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", experts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PS", partition));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NE", experts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", p1_threads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PSTRIDE", rows * experts));
}

const OneKey = struct { wt: u8 = 0, dims: [4]c_int = @splat(0), len: u8 = 0, k: c_int = 0, norm: bool = false, scale: u32 = 0 };
const OneSlot = struct { key: OneKey = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null, scale_arr: Array = .{ .ctx = null } };
var one_slots: [4]OneSlot = @splat(.{});
var one_next: usize = 0;

/// A router call in one launch: the routes of `x` (bf16 [.., rows, 4096], 2 to 4 rows) by `raw` (f16 or bf16 [288,
/// 4096] as stored) and `bias`, inds u32 and scores bf16 of shape [.., rows, k] as the stock route's
/// (src/router_gpu_test.zig holds them to it bit for bit).
pub fn routeOne(s: mlx.mlx_stream, x: Array, raw: Array, bias: Array, k: c_int, norm: bool, scale: f32) !routing.MoeRouting {
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse return error.Glm53fRouterRows;
    if (rows < min_rows or rows > max_rows) return error.Glm53fRouterRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.Glm53fRouterWeightType;
    var key: OneKey = .{ .wt = @intFromBool(wt == .bfloat16), .len = @intCast(xsh.len), .k = k, .norm = norm, .scale = @bitCast(scale) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    const slot = for (&one_slots) |*sl| {
        if (sl.cfg != null and std.meta.eql(sl.key, key)) break sl;
    } else blk: {
        const sl = &one_slots[one_next];
        one_next = (one_next + 1) % one_slots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        if (sl.scale_arr.ctx != null) _ = mlx.mlx_array_free(sl.scale_arr);
        sl.* = .{};
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var out_shape: [4]c_int = undefined;
        for (xsh[0 .. xsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[xsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @divExact(hidden, partition), rows, experts }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, p1_threads * p1_groups, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, p1_threads, 1, 1));
        try oneTemplates(cfg, wt, rows);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TOUT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NORM", @intFromBool(norm)));
        sl.* = .{ .key = key, .cfg = cfg, .scale_arr = mlx.mlx_array_new_float(scale) };
        break :blk sl;
    };
    const kern = try kernelOf(&one_route_kernel, "turboexl3_glm53f_router_one", &.{ "x", "w", "bias", "scale", "counter" }, &.{ "p", "inds", "scores" }, one_route_source, try oneHeader(), false);
    const cnt = try arrivals();
    defer _ = mlx.mlx_array_free(cnt);
    var outs: [3]Array = undefined;
    try applyKernel(s, kern, slot.cfg.?, &.{ x, raw, bias, slot.scale_arr, cnt }, &outs);
    // the partials are the launch's scratch: the graph keeps them until the launch has run
    _ = mlx.mlx_array_free(outs[0]);
    return .{ .inds = outs[1], .norm_scores = outs[2] };
}

// ── TURBOEXL3_GLM53F_ROUTER_ONE1's launch ──

var one1_engaged = false;
var one1_kernel: ?mlx.mlx_fast_metal_kernel = null;
const One1Key = struct { wt: u8 = 0, dims: [4]c_int = @splat(0), len: u8 = 0, k: c_int = 0, norm: bool = false, scale: u32 = 0 };
const One1Slot = struct { key: One1Key = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null, scale_arr: Array = .{ .ctx = null } };
var one1_slots: [4]One1Slot = @splat(.{});
var one1_next: usize = 0;

/// A one-row call's launch outputs: the logits f32 [.., 1, 288] (MLX's gemv's, the stock matmul's) and the routes.
pub const One1 = struct { logits: Array, routing: routing.MoeRouting };

/// A one-row router call in one launch: the routes of `x` (bf16 [.., 1, 4096]) by `raw` (f16 or bf16 [288, 4096] as
/// stored) and `bias`, inds u32 and scores bf16 of shape [.., 1, k] as the stock route's, and its logits
/// (src/router_gpu_test.zig holds all three to it bit for bit).
pub fn routeOne1(s: mlx.mlx_stream, x: Array, raw: Array, bias: Array, k: c_int, norm: bool, scale: f32) !One1 {
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse return error.Glm53fRouterRows;
    if (rows != 1) return error.Glm53fRouterRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.Glm53fRouterWeightType;
    var key: One1Key = .{ .wt = @intFromBool(wt == .bfloat16), .len = @intCast(xsh.len), .k = k, .norm = norm, .scale = @bitCast(scale) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    const slot = for (&one1_slots) |*sl| {
        if (sl.cfg != null and std.meta.eql(sl.key, key)) break sl;
    } else blk: {
        const sl = &one1_slots[one1_next];
        one1_next = (one1_next + 1) % one1_slots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        if (sl.scale_arr.ctx != null) _ = mlx.mlx_array_free(sl.scale_arr);
        sl.* = .{};
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var logits_shape: [4]c_int = undefined;
        var out_shape: [4]c_int = undefined;
        for (xsh[0 .. xsh.len - 1], 0..) |d, i| {
            logits_shape[i] = d;
            out_shape[i] = d;
        }
        logits_shape[xsh.len - 1] = experts;
        out_shape[xsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &logits_shape, xsh.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, one1_threads * one1_groups, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, one1_threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TOUT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NE", experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", one1_threads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NORM", @intFromBool(norm)));
        sl.* = .{ .key = key, .cfg = cfg, .scale_arr = mlx.mlx_array_new_float(scale) };
        break :blk sl;
    };
    const kern = try kernelOf(&one1_kernel, "turboexl3_glm53f_router_one1", &.{ "x", "w", "bias", "scale", "counter" }, &.{ "logits", "inds", "scores" }, one1_source, one1_header, false);
    const cnt = try arrivals();
    defer _ = mlx.mlx_array_free(cnt);
    var outs: [3]Array = undefined;
    try applyKernel(s, kern, slot.cfg.?, &.{ x, raw, bias, slot.scale_arr, cnt }, &outs);
    return .{ .logits = outs[0], .routing = .{ .inds = outs[1], .norm_scores = outs[2] } };
}

const OnePKey = struct { wt: u8 = 0, dims: [4]c_int = @splat(0), len: u8 = 0 };
const OnePSlot = struct { key: OnePKey = .{}, cfg: ?mlx.mlx_fast_metal_kernel_config = null };
var one_pslots: [4]OnePSlot = @splat(.{});
var one_pnext: usize = 0;

/// A predictor call in one launch: the stock predictor's scores (f32 of x's leading dimensions by 288) of `x` by `raw`
/// (the next layer's router as stored) and `correction` (src/router_gpu_test.zig holds them to it bit for bit).
pub fn predictOne(s: mlx.mlx_stream, x: Array, raw: Array, correction: Array) !Array {
    const xsh = mlx.getShape(x);
    const rows = rowsOf(xsh) orelse return error.Glm53fRouterRows;
    if (rows < min_rows or rows > max_rows) return error.Glm53fRouterRows;
    const wt = mlx.mlx_array_dtype(raw);
    if (!weightServes(wt)) return error.Glm53fRouterWeightType;
    var key: OnePKey = .{ .wt = @intFromBool(wt == .bfloat16), .len = @intCast(xsh.len) };
    for (xsh, 0..) |d, i| key.dims[i] = d;
    const slot = for (&one_pslots) |*sl| {
        if (sl.cfg != null and std.meta.eql(sl.key, key)) break sl;
    } else blk: {
        const sl = &one_pslots[one_pnext];
        one_pnext = (one_pnext + 1) % one_pslots.len;
        if (sl.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        sl.* = .{};
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var out_shape: [4]c_int = undefined;
        for (xsh[0 .. xsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[xsh.len - 1] = experts;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @divExact(hidden, partition), rows, experts }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, xsh.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, p1_threads * p1_groups, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, p1_threads, 1, 1));
        try oneTemplates(cfg, wt, rows);
        sl.* = .{ .key = key, .cfg = cfg };
        break :blk sl;
    };
    const kern = try kernelOf(&one_predict_kernel, "turboexl3_glm53f_predict_one", &.{ "x", "w", "correction", "counter" }, &.{ "p", "scores" }, one_predict_source, try oneHeader(), false);
    const cnt = try arrivals();
    defer _ = mlx.mlx_array_free(cnt);
    var outs: [2]Array = undefined;
    try applyKernel(s, kern, slot.cfg.?, &.{ x, raw, correction, cnt }, &outs);
    _ = mlx.mlx_array_free(outs[0]);
    return outs[1];
}

// ── tests (MLX-free) ──

/// The edits undone, each new back to its old (the composition test's: the stock pieces again).
fn undone(comptime edits: []const [2][]const u8) [edits.len][2][]const u8 {
    var u: [edits.len][2][]const u8 = undefined;
    for (edits, 0..) |e, i| u[i] = .{ e[1], e[0] };
    return u;
}
const route_body_undo = undone(&route_body_edits);
const route_select_undo = undone(&route_select_edits);

const testing = std.testing;

test "turboexl3_glm53f stream router: phase 2 is the stock router text with its one logits read summing the two partials as MLX's split-K accumulate does" {
    const stock = router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias)];
    const i = std.mem.indexOf(u8, stock, stock_read).?;
    try testing.expect(std.mem.indexOf(u8, p2_source, stock_read) == null);
    try testing.expectEqualStrings(stock[0..i], p2_source[0..i]);
    try testing.expectEqualStrings(sum_read, p2_source[i .. i + sum_read.len]);
    try testing.expectEqualStrings(stock[i + stock_read.len ..], p2_source[i + sum_read.len ..]);
}

test "turboexl3_glm53f stream router: the predictor's phase 2 is the stock router text's sigmoid over the summed partials, then the correction added" {
    try testing.expect(std.mem.indexOf(u8, predict_p2_source, stock_read) == null);
    const i = std.mem.indexOf(u8, predict_p2_source, sum_read).?;
    const j = std.mem.indexOf(u8, predict_score_text, stock_read).?;
    // everything after the read is the stock text's (its sigmoid), then the add
    const after = predict_score_text[j + stock_read.len ..];
    try testing.expectEqualStrings(after, predict_p2_source[i + sum_read.len ..][0..after.len]);
    try testing.expectEqualStrings("\n  scores[rbase + e] = sg + correction[e];\n", predict_p2_source[i + sum_read.len + after.len ..]);
    try testing.expect(std.mem.indexOf(u8, after, "metal::precise::exp(metal::abs(l))") != null);
    var buf: [160]u8 = undefined;
    predictions = 41;
    predict_declined = 0;
    predict_other = 3;
    defer reset();
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_PREDICT predictions=41 declined=0 other=3\n", predictSummary(&buf));
}

test "turboexl3_glm53f stream router: a call's rows are its leading dimensions over the hidden; the f32 table is taken only as the transposed view MLX's matmul reads as transpose_b" {
    try testing.expectEqual(@as(?c_int, 3), rowsOf(&.{ 1, 3, 4096 }));
    try testing.expectEqual(@as(?c_int, 4), rowsOf(&.{ 4, 4096 }));
    try testing.expectEqual(@as(?c_int, null), rowsOf(&.{ 1, 3, 2048 }));
    try testing.expectEqual(@as(?c_int, null), rowsOf(&.{4096}));
    try testing.expect(tableLayout(&.{ 4096, 288 }, &.{ 1, 4096 }));
    try testing.expect(!tableLayout(&.{ 4096, 288 }, &.{ 288, 1 }));
    try testing.expect(!tableLayout(&.{ 4096, 256 }, &.{ 1, 4096 }));
    try testing.expect(weightServes(.float16) and weightServes(.bfloat16) and !weightServes(.float32));
    try testing.expect(predictPromotes(.float16) and !predictPromotes(.bfloat16));
    try testing.expect(stridesRowMajor(&.{ 1, 3, 4096 }, &.{ 99, 4096, 1 }));
    try testing.expect(!stridesRowMajor(&.{ 1, 3, 4096 }, &.{ 12288, 1, 3 }));
}

test "turboexl3_glm53f stream router: ROUTER_ONE's launch is phase 1 with its partial stored coherently, the arrival, and the stock texts' statements with only their read and their row indexing edited" {
    // phase 1: the two-launch text but for its store
    try testing.expectEqualStrings(p1_source, comptime editsOnce(p1_one[0 .. p1_one.len - one_arrive.len - 1], &.{.{ p1_store_one, p1_store }}));
    try testing.expectEqualStrings(one_arrive, p1_one[p1_one.len - one_arrive.len ..]);
    // the router's phase 2: undoing the edits gives the stock pieces back, and the pieces are the stock text's own
    try testing.expectEqualStrings(route_body, comptime editsOnce(route_body_one, &route_body_undo));
    try testing.expectEqualStrings(route_select, comptime editsOnce(route_select_one, &route_select_undo));
    const lo = std.mem.indexOf(u8, p2_source, route_loop_open).?;
    try testing.expectEqualStrings(p2_source[lo + route_loop_open.len ..][0..route_body.len], route_body);
    try testing.expect(std.mem.endsWith(u8, p2_source, route_select));
    // the selection reads only its row's slice, and the one texts read no partial but through the staged sum
    try testing.expect(std.mem.indexOf(u8, route_select_one, "tid") == null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, route_select_one, "rk["));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, route_select_one, "rw["));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, route_select_one, "sel["));
    for ([_][]const u8{ one_route_source, one_predict_source }) |t| {
        try testing.expect(std.mem.indexOf(u8, t, "logits[") == null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, t, one_read));
        try testing.expect(std.mem.startsWith(u8, t, p1_one));
    }
    // the predictor's phase 2: the stock sigmoid with its read edited, then the stock add
    try testing.expectEqualStrings(predict_score_text, comptime editsOnce(predict_score_one, &.{.{ one_read, stock_read }}));
    try testing.expect(std.mem.indexOf(u8, one_predict_source, "scores[rbase + e] = sg + correction[e];") != null);
    var buf: [160]u8 = undefined;
    one_routes = 82;
    one_predictions = 80;
    defer reset();
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE routes=82 predictions=80\n", oneSummary(&buf));
}

test "turboexl3_glm53f stream router: the summary line holds each registered layer's routes, so equal counts read as every layer of every verify" {
    reset();
    defer reset();
    const dummy: [3]u8 = .{ 0, 0, 0 };
    for (0..3) |i| {
        keys[i] = @intFromPtr(&dummy[i]);
        n_layers += 1;
    }
    routes_by_layer[0] = 5;
    routes_by_layer[1] = 5;
    routes_by_layer[2] = 4;
    declined = 1;
    other_rows = 7;
    try testing.expectEqual(@as(?usize, 1), find(&dummy[1]));
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER routes=14 layers=3 per_layer_min=4 per_layer_max=5 declined=1 unregistered=0 other=7\n", summary(&buf));
}

test "turboexl3_glm53f stream router: the MTP head's router (MTP_ROUTER) is counted apart, so the trunk's layers and per-layer counts stay one a verify and the head's routes read as head_routes" {
    reset();
    defer reset();
    const dummy: [4]u8 = .{ 0, 0, 0, 0 };
    // three trunk layers, then the head (registerHead's marking, MLX-free)
    for (0..4) |i| {
        keys[i] = @intFromPtr(&dummy[i]);
        n_layers += 1;
    }
    head_index = 3;
    try testing.expect(headRegistered() and isHead(3) and !isHead(2));
    routes_by_layer[0] = 5;
    routes_by_layer[1] = 5;
    routes_by_layer[2] = 5;
    routes_by_layer[3] = 2;
    one_routes = 15;
    one_head_routes = 2;
    one_predictions = 10;
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER routes=15 layers=3 per_layer_min=5 per_layer_max=5 declined=0 unregistered=0 other=0 head_routes=2\n", summary(&buf));
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE routes=15 predictions=10 head_routes=2\n", oneSummary(&buf));
    // a second head refuses by name
    try testing.expectError(error.Glm53fRouterHeadTwice, registerHead(&dummy[0], .{ .ctx = null }));
    // without the head: the lines as before
    head_index = null;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER routes=17 layers=4 per_layer_min=2 per_layer_max=5 declined=0 unregistered=0 other=0\n", summary(&buf));
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE routes=15 predictions=10\n", oneSummary(&buf));
}

test "turboexl3_glm53f stream router: ROUTER_ONE1's launch is MLX's gemv geometry at 288 outputs (18 threadgroups of 16 outputs, no tail), its arrival counts those 18, and its phase 2 is the stock router text with only its read made coherent; its line counts the head apart and reset empties it" {
    // MLX's gemv at bm4 bn1 sm1 sn32 tm4 tn4: 16 outputs a threadgroup, so 288 experts take 18 with no tail simdgroup,
    // and the last of those 18 to arrive runs phase 2
    try testing.expectEqual(@as(c_int, 18), one1_groups);
    try testing.expectEqual(@as(c_int, 0), @rem(experts, 16));
    try testing.expect(std.mem.indexOf(u8, one1_source, std.fmt.comptimePrint("constexpr uint RO1_NTG = {d}u;", .{one1_groups})) != null);
    // phase 2: the stock text's read replaced, every other byte the stock text's
    const stock = router_kernel.ROUTER_SOURCES[@backingInt(router_kernel.RouterMode.sigmoid_bias)];
    const i = std.mem.indexOf(u8, stock, stock_read).?;
    try testing.expectEqualStrings(stock[0..i], one1_route_text[0..i]);
    try testing.expectEqualStrings(one1_read, one1_route_text[i .. i + one1_read.len]);
    try testing.expectEqualStrings(stock[i + stock_read.len ..], one1_route_text[i + one1_read.len ..]);
    one1_routes = 7;
    one1_head_routes = 400;
    var buf: [160]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE1 routes=7 head_routes=400\n", one1Summary(&buf));
    reset();
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_ROUTER_ONE1 routes=0 head_routes=0\n", one1Summary(&buf));
}
