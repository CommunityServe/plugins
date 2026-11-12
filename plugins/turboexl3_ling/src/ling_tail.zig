//! turboexl3_ling's MoE tail (MOE_TAIL, exact): a MoE layer's end after its routed call's join runs in one launch
//! (turboexl3's `ling.moeTail`) at a call's 1 to 8 rows on the decode lane (Ling's top-k 8 puts 1 to 6 rows there; a
//! wider call is a prompt call and keeps its path). The stock end is six launches: the source's routed sum
//! (`sum(multiply(astype(routed, f32), expand_dims(astype(weights, f32), -1)), -2)`: the weights' astype, the multiply,
//! the sum), the MoE block's astype of the sum to bf16 and its add of the shared expert, then the forward's tail
//! (mlx-serve's add_norm: the residual add and the next layer's input norm, or the final norm). The launch is that
//! add_norm's own kernel text with its delta made in place: the routed sum in MLX's own statements and order
//! (turboexl3's combine), rounded where the chain rounds, so the residual and the normed rows are the chain's words
//! (turboexl3's `gpu ling tail` byte test, rows 1 to 8).
//!
//! The forward arms a MoE layer before its block (`arm`). The source keeps the call's joined rows and the routes'
//! weights (`rows`), the MoE block the shared expert's output (`shared`), and the layer's tail (table_forward.zig
//! `moeTailOf`) takes the three (`take`); the forward ends the layer (`end`), freeing them. The chain's own arrays (the
//! routed sum, its astype, the add) are still built and freed unevaluated. One forward at a time, on one thread.
//! `moeTailOf` marks the run-path line at each launch.
//!
//! ROUTED_JOIN (exact; on the tail): the source also keeps the call's outputs unjoined, the device wave's rows and each
//! miss part's, with each pair's (source, row) (`parts`), and the tail reads them where they are, so the streamer's join
//! (MLX's concatenate, one copy per output, and its take) is freed unevaluated. A call of one output in the routed order
//! (DEVROUTE's device wave, no miss) is its join's rows already: `ling.moeTail` reads that output as the join
//! (`direct`). A call of 2 to 8 outputs runs turboexl3's `ling.moeTailParts`, which reads each pair's row at its (source,
//! row). A call with more outputs than that launch binds takes the join. An armed call past the decode lane (7 or 8
//! rows: more than 48 routed ids) takes COMBINE's route in the source (ling_source.zig `rows`), which keeps no rows for
//! the tail, so that layer keeps the stock chain. `launch` marks the run-path line at each launch of either form.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ran = @import("ling_ran.zig");

const Array = mlx.mlx_array;

/// The widest call the launch takes (turboexl3's `ling.tail.max_rows`)
pub const max_rows: c_int = t3.ling.tail.max_rows;

/// The outputs the parts launch binds (turboexl3's `ling.tail.max_sources`)
const max_sources: usize = t3.ling.tail.max_sources;

var declined_said = false;
var join_declined_said = false;

/// The first call the launch did not take (its shape or dtypes; the chain kept), logged once.
pub fn declined(e: anyerror) void {
    if (declined_said) return;
    declined_said = true;
    std.log.info("[turboexl3_ling] MOE_TAIL: a call kept the stock chain ({s})", .{@errorName(e)});
}

// ── The armed layer's parts (one layer at a time) ──

var armed: ?u32 = null;
var routed: Array = .{ .ctx = null };
var weights: Array = .{ .ctx = null };
var shared_out: Array = .{ .ctx = null };
/// ROUTED_JOIN: the call's outputs (references of this module's own), their count (more than `max_sources`: none kept)
/// and each pair's (source, row); `direct`: the one output kept is the join's rows in order (no `loc`)
var srcs: [max_sources]Array = @splat(.{ .ctx = null });
var n_outs: usize = 0;
var loc: Array = .{ .ctx = null };
var direct = false;

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

/// The forward arms model layer `layer`'s MoE block; whatever an earlier layer left is freed.
pub fn arm(layer: u32) void {
    end();
    armed = layer;
}

/// The source's test at a routed call of model layer `layer`.
pub fn armedFor(layer: u32) bool {
    const a = armed orelse return false;
    return a == layer;
}

/// The source's part: the call's joined rows `ro` (f32 [n, k, hidden]) and the routes' weights `w` ([n, k]), each a
/// reference of this module's own. Nothing while no layer is armed.
pub fn rows(ro: Array, w: Array) void {
    if (armed == null) return;
    put(&routed, ro);
    put(&weights, w);
    clearOuts();
}

/// ROUTED_JOIN's part, after `rows`: the call's outputs unjoined, in the join's order, and each pair's (source, row)
/// (int32 [n, k, 2]), references of this module's own. `identity` (the streamer's): the call has one output and the
/// join is its rows in order, so only that output is kept (`direct`). With no output, or past `max_sources`, only their
/// count is kept (the tail takes the join). Nothing while no layer is armed or the source kept no rows.
pub fn parts(outs: []const Array, at: Array, identity: bool) void {
    if (armed == null or routed.ctx == null) return;
    clearOuts();
    n_outs = outs.len;
    if (outs.len == 0 or outs.len > max_sources) return;
    if (identity and outs.len == 1) {
        put(&srcs[0], outs[0]);
        direct = true;
        return;
    }
    for (outs, srcs[0..outs.len]) |x, *slot| put(slot, x);
    put(&loc, at);
}

fn clearOuts() void {
    for (&srcs) |*x| clear(x);
    clear(&loc);
    n_outs = 0;
    direct = false;
}

/// The MoE block's part: the shared expert's output, where its add follows the source's routed sum (a reference of this
/// module's own). Nothing while no layer is armed, or after the source kept no rows (a call off the decode lane).
pub fn shared(x: Array) void {
    if (armed == null or routed.ctx == null) return;
    put(&shared_out, x);
}

/// What the tail reads, borrowed until `end`: with ROUTED_JOIN the call's outputs (`loc` set, or `direct` for the one
/// output that is the join's rows in order) or, past `max_sources`, only their count (`n_outs`).
pub const Parts = struct { routed: Array, weights: Array, shared: Array, outs: []const Array = &.{}, n_outs: usize = 0, loc: ?Array = null, direct: bool = false };

/// The armed layer's three parts, or null where one is missing (the tail keeps the chain).
pub fn take() ?Parts {
    if (armed == null or routed.ctx == null or weights.ctx == null or shared_out.ctx == null) return null;
    return .{
        .routed = routed,
        .weights = weights,
        .shared = shared_out,
        .outs = if (loc.ctx != null or direct) srcs[0..n_outs] else &.{},
        .n_outs = n_outs,
        .loc = if (loc.ctx != null) loc else null,
        .direct = direct,
    };
}

/// The tail's launch over the armed layer's parts, the same words in each form: where the source kept the call's
/// outputs (ROUTED_JOIN), `ling.moeTail` over the one output that is the join's rows in order (`direct`) or
/// `ling.moeTailParts` over 2 to 8 outputs; else `ling.moeTail` over the join. The caller frees both results.
pub fn launch(s: mlx.mlx_stream, h: Array, p: Parts, w: Array, eps: Array) !t3.ling.tail.Out {
    if (p.direct) {
        const out = try t3.ling.moeTail(s, h, p.outs[0], p.weights, p.shared, w, eps);
        ran.mark(.ROUTED_JOIN);
        return out;
    }
    if (p.loc) |at| {
        const out = try t3.ling.moeTailParts(s, h, p.outs, at, p.weights, p.shared, w, eps);
        ran.mark(.ROUTED_JOIN);
        return out;
    }
    if (p.n_outs > max_sources) joinDeclined(p.n_outs);
    return t3.ling.moeTail(s, h, p.routed, p.weights, p.shared, w, eps);
}

/// The first call whose outputs the launch does not bind (it took the join), logged once.
fn joinDeclined(n: usize) void {
    if (join_declined_said) return;
    join_declined_said = true;
    std.log.info("[turboexl3_ling] ROUTED_JOIN: a call of {d} outputs took the join (the launch binds {d})", .{ n, max_sources });
}

/// The armed layer's tail is final (or the forward failed on the way): its parts freed, no layer armed.
pub fn end() void {
    armed = null;
    clear(&routed);
    clear(&weights);
    clear(&shared_out);
    clearOuts();
}
