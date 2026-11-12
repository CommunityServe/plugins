//! turboexl3_ling's MoE tail at prompt widths, PF_MOE_TAIL (L3, kept as S48; exact): a layer-major MoE layer ends each
//! prompt span in one turboexl3 launch (`ling.moeTailWide`) in place
//! of the served chain after its wide routed call: the combine over the call's JOINLESS parts (`ling.combineParts`, f32
//! out), the streamed source's astype to bf16, the span's slice, the MoE block's add of the shared expert, MLX's add of
//! the residual and MLX's fast rms_norm at the next layer's start (or the final norm). The launch gives that chain's
//! words at every row count (turboexl3's `gpu ling tail wide` byte test at the standard request's spans).
//!
//! The pass (patches/ling-trunk.patch, table_forward.zig `hiddenSpans`) arms a MoE layer whose sum the
//! tail makes (`moe_block.tailServes`: a streamed source's routed sum plus an ungated shared expert) with its spans'
//! count (`arm`). `moeSpans` brackets each routed call with the spans it takes (`beginCall`, `endCall`); the source keeps
//! that call's JOINLESS parts (`parts`: the outputs, each pair's (source, row), the routes' weights in f32) and the MoE
//! block each span's shared expert output (`atSpan`, then `shared`). The residual loop takes a span's launch (`launch`):
//! its call's parts, its rows of loc and the weights, its shared output and the next norm's weight with eps after it
//! (`normWeights`). The chain's own arrays (the combine, its astype, the adds) are still built and freed unevaluated; a
//! span the launch refuses (more than 24 sources, a mix of f16 and f32 sources, a shape) keeps them. The layer's end
//! (`end`) drops every reference this module holds. One pass at a time, on one thread. A span's launch marks PF_MOE_TAIL
//! on the run-path line (src/ling_ran.zig).
const std = @import("std");
const ran = @import("ling_ran.zig");
const plan_mod = @import("ling_pf_tail_plan.zig");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");

const Array = mlx.mlx_array;

/// The sources a launch binds (turboexl3's `ling.tail.wide_max_sources`, JOINLESS's 24)
const max_sources: usize = t3.ling.tail.wide_max_sources;

const max_spans = plan_mod.max_spans;

var declined_said = false;

/// The first span the launch did not take (its shape or dtypes; the chain kept), logged once.
fn declined(e: anyerror) void {
    if (declined_said) return;
    declined_said = true;
    std.log.info("[turboexl3_ling] PF_MOE_TAIL: a prompt span kept the served chain ({s})", .{@errorName(e)});
}

// ── The armed layer (one at a time) ──

/// A routed call's parts: its outputs (references of this module's own), each pair's (source, row), the weights in f32;
/// `refused` when it had no output or more than a launch binds (its spans keep the chain)
const Call = struct {
    srcs: [max_sources]Array = @splat(.{ .ctx = null }),
    n_srcs: usize = 0,
    loc: Array = .{ .ctx = null },
    weights: Array = .{ .ctx = null },
    refused: bool = false,
};

var armed: ?u32 = null;
var plan: plan_mod.Plan = .{};
var calls: [max_spans]Call = @splat(.{});
var cur_call: ?usize = null;
var cur_span: ?usize = null;
var shared_out: [max_spans]Array = @splat(.{ .ctx = null });

fn put(slot: *Array, x: Array) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, x);
    slot.* = r;
}

fn clear(slot: *Array) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = .{ .ctx = null };
}

/// The pass arms model layer `layer`'s MoE block over `spans` spans; whatever an earlier layer left is dropped.
pub fn arm(layer: u32, spans: usize) void {
    end();
    if (!plan.arm(spans)) return;
    armed = layer;
}

/// `moeSpans`, before a routed call over spans `j0`..`j1` (`spans` the pass's rows a span): the call the source's
/// parts belong to. Nothing while no layer is armed.
pub fn beginCall(j0: usize, j1: usize, spans: []const u32) void {
    if (armed == null) return;
    cur_call = plan.beginCall(j0, j1, spans);
}

/// `moeSpans`, after the call: the source's parts are kept, or not.
pub fn endCall() void {
    cur_call = null;
}

/// The source's part at a wide call (its JOINLESS route): the outputs `outs`, each pair's (source, row) `loc` int32
/// [n, K, 2] and the routes' weights `w` f32 [n, K], each a reference of this module's own. Nothing while no call is
/// current; a call of no output, or past `max_sources`, is marked refused (its spans keep the chain).
pub fn parts(outs: []const Array, loc: Array, w: Array) void {
    const c = cur_call orelse return;
    const call = &calls[c];
    if (outs.len == 0 or outs.len > max_sources) {
        call.refused = true;
        return;
    }
    for (outs, call.srcs[0..outs.len]) |x, *slot| put(slot, x);
    call.n_srcs = outs.len;
    put(&call.loc, loc);
    put(&call.weights, w);
}

/// `moeSpans`, before span `j`'s finish: the MoE block's shared expert output that follows is span `j`'s.
pub fn atSpan(j: usize) void {
    if (armed == null or j >= plan.n_spans) return;
    cur_span = j;
}

/// The MoE block's part (`mlpFinish`, an ungated shared expert): the current span's shared expert output, a reference
/// of this module's own. Nothing while no span is current.
pub fn shared(x: Array) void {
    const j = cur_span orelse return;
    put(&shared_out[j], x);
    cur_span = null;
}

/// The next norm's operand: its weight `w` [D] widened to f32 (exact), then `eps` (0-dim f32), as one f32 [D + 1] (the
/// caller frees it). A norm without a weight is refused, as `narrowAddNorm` refuses it (the layer keeps the chain).
pub fn normWeights(s: mlx.mlx_stream, w: Array, eps: Array) !Array {
    if (w.ctx == null or eps.ctx == null) return error.LingPfTailNoNormWeight;
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_astype(&wf, w, .float32, s));
    var e1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e1);
    try mlx.check(mlx.mlx_reshape(&e1, eps, &[_]c_int{1}, 1, s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, wf);
    _ = mlx.mlx_vector_array_append_value(vec, e1);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

/// What the residual loop takes: the span's residual and its next norm, each in h's shape (the caller owns both)
pub const Out = struct { h: Array, normed: Array };

/// Span `j`'s tail in one launch over its call's parts: `h` [1, S, D] (the residual before the MoE's add), `we` the
/// next norm's operand (`normWeights`). Null where the span has no call's parts or no shared output kept, or the launch
/// refuses it (the caller keeps the chain).
pub fn launch(s: mlx.mlx_stream, j: usize, h: Array, we: Array) ?Out {
    if (armed == null) return null;
    const sp = plan.spanOf(j) orelse return null;
    const call = &calls[sp.call];
    if (call.refused or call.n_srcs == 0 or call.loc.ctx == null or shared_out[j].ctx == null) return null;
    const lsh = mlx.getShape(call.loc);
    if (lsh.len != 3) return null;
    const k = lsh[1];
    var loc_j = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(loc_j);
    var w_j = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_j);
    const end_row = sp.at + sp.rows;
    mlx.check(mlx.mlx_slice(&loc_j, call.loc, &[_]c_int{ sp.at, 0, 0 }, 3, &[_]c_int{ end_row, k, 2 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s)) catch return null;
    mlx.check(mlx.mlx_slice(&w_j, call.weights, &[_]c_int{ sp.at, 0 }, 2, &[_]c_int{ end_row, k }, 2, &[_]c_int{ 1, 1 }, 2, s)) catch return null;
    const out = t3.ling.moeTailWide(s, h, call.srcs[0..call.n_srcs], loc_j, w_j, shared_out[j], we) catch |e| {
        declined(e);
        return null;
    };
    ran.mark(.PF_MOE_TAIL);
    return .{ .h = out.h, .normed = out.normed };
}

/// The layer's end: every reference this module holds dropped, nothing armed.
pub fn end() void {
    for (&calls) |*c| {
        for (&c.srcs) |*x| clear(x);
        clear(&c.loc);
        clear(&c.weights);
        c.* = .{};
    }
    for (&shared_out) |*x| clear(x);
    armed = null;
    cur_call = null;
    cur_span = null;
    plan = .{};
}
