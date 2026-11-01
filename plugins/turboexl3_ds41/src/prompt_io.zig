//! `MLX_SERVE_DSV41_PROMPT_IO` (read at load; unset or `0`: off): a prompt pass's routed calls timed and counted, and
//! the CED decoder's experts predicted at two encoder layers (the coverage probe), logged when the prompt ends (the
//! source's phase change to decode). Off, every hook returns at its first branch: nothing is evaluated or read back.
//! On, a routed call's routes are evaluated before the streamer's barrier reads them back (the same wait, taken first)
//! and a replaying chunk's probe runs twenty routers over its tail rows at encoder layers 10 and 19 (one wait each).
//!
//! - `[prompt-io]`, one per routed call of the prompt:
//!   - `ra_ms`: the model's read-ahead of the layer: its predicted routes read back (the host waits for the previous
//!     layer's tail and the predictor) and the streamer's read-ahead posted;
//!   - `attn_ms`: from the call to its routes evaluated (the GPU's attention and router);
//!   - `moe_ms`: the routed call itself (its groups' reads and waves, each drained);
//!   - `layer_ms`: since the previous call's end, and `other_ms` what the three leave of it;
//!   - over the layer (since the previous call's end): the stream's bytes read, its reads in flight (`rwall_ms`) and
//!     the host's read waits (`rwait_ms`); over the call: its routes' unique experts (misses and hits) and the
//!     read-ahead's hits among them;
//!   - `staged` / `adopted` (MLX_SERVE_DSV41_STAGE): the records staged in the layer's windows and those its routes took
//!     without a read (the rest were wasted).
//! - `[prompt-io] side=encoder` / `decoder` / `total`: the sums on each side of the CED split.
//! - `[stream-stats prompt]`: the stream's counters since the prompt's first call.
//! - `[ced-probe]`, one per decoder layer of a replaying chunk: its tail rows' routed experts (`actual`) against the
//!   layer's router run on the tail rows' stream at encoder layers 10 and 19 (each row's top-k, the experts ranked by
//!   votes, ties by id): `p10` / `p19` the share of `actual` among the top `cut` (the layer's prompt slot rows), `u10` /
//!   `u19` among every voted expert (and how many were voted); `res` the share resident when the call routes; `prev` the
//!   share among the top `cut` of the earlier prompts' routes of the layer (a static prior; `-` for a server's first).

const std = @import("std");
/// `MLX_SERVE_DSV41_PROMPT_TL` (profile builds): the prompt's GPU timeline marks its stages at these hooks.
const ptl = @import("prompt_tl.zig");

/// The switch (`configure`, at load). `2` also splits each wide layer's read-ahead wait (`split_ra`).
pub var on: bool = false;
/// `MLX_SERVE_DSV41_PROMPT_IO=2`: the model's read-ahead evaluates the stream's rows, then their collapse, before its
/// routes (`raStep`), so the `[prompt-io] wide` line splits its wait; two more syncs a wide layer, nothing else changes.
pub var split_ra: bool = false;

pub fn configure(value: ?[]const u8) void {
    on = if (value) |v| v.len > 0 and v[0] != '0' else false;
    split_ra = if (value) |v| std.mem.eql(u8, v, "2") else false;
    reset();
    prev = @splat(@splat(0));
    prev_n = @splat(0);
}

/// The CED split of the forward running now (the decoder's first layer) and its replay tail's rows; 0 when the
/// forward does not replay. Set by the model around a replaying chunk's layers.
pub var split: usize = 0;
pub var tail: usize = 0;

pub const max_layers = 64;
pub const max_experts = 512;
const max_calls = 512;
/// The encoder layers the probe runs the decoder's routers at.
pub const probe_layers = [2]usize{ 10, 19 };

/// The stream's counters a record keeps (the source's `Stats`).
pub const Counters = struct {
    routes: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,
    bytes: u64 = 0,
    wait_ns: u64 = 0,
    wall_ns: u64 = 0,
    ahead_posted: u64 = 0,
    ahead_hits: u64 = 0,
    ahead_bytes: u64 = 0,
    spec_issued: u64 = 0,
    spec_landed: u64 = 0,
    claimed: u64 = 0,
    /// the CED decoder prefetch's (`MLX_SERVE_DSV41_DEC_PREFETCH`): records posted, bytes landed
    fill_posted: u64 = 0,
    fill_bytes: u64 = 0,
    /// MLX_SERVE_DSV41_STAGE: records staged into the transient windows, those a route took without a read, their bytes.
    stage_posted: u64 = 0,
    stage_adopted: u64 = 0,
    stage_bytes: u64 = 0,
    /// MLX_SERVE_DSV41_STAGE_NEXT: of the staged records, those posted during the call before theirs.
    stage_early: u64 = 0,

    fn since(a: Counters, b: Counters) Counters {
        var d: Counters = .{};
        inline for (@typeInfo(Counters).@"struct".field_names) |f| @field(d, f) = @field(a, f) -| @field(b, f);
        return d;
    }
};

const Call = struct {
    layer: u16 = 0,
    rows: u32 = 0,
    /// a decoder layer's call of a replaying chunk (its tail rows)
    dec: bool = false,
    /// the layer's read-ahead (0 for none), the call's start, its routes evaluated, its end (ns)
    ra0: u64 = 0,
    ra1: u64 = 0,
    t0: u64 = 0,
    t_idx: u64 = 0,
    t1: u64 = 0,
    /// the layer's start (the previous call's end, or the prompt's first hook)
    t_layer: u64 = 0,
    /// a wide layer's read-ahead routes read back (`raEval`, 0: none), and its graph's steps built (`wide`, 0: none)
    ra_eval: u64 = 0,
    /// with `split_ra`: the read-ahead's stream rows and their collapse evaluated (`raStep`, 0: none)
    ra_cat: u64 = 0,
    ra_col: u64 = 0,
    g: [n_wide]u64 = @splat(0),
    c_layer: Counters = .{},
    c0: Counters = .{},
    c1: Counters = .{},
};

const Probe = struct {
    layer: u16 = 0,
    actual: u32 = 0,
    cut: u32 = 0,
    top: [2]u32 = .{ 0, 0 },
    voted_hit: [2]u32 = .{ 0, 0 },
    voted: [2]u32 = .{ 0, 0 },
    have: [2]bool = .{ false, false },
    res: u32 = 0,
    prev: ?u32 = null,
};

var active = false;
var t_prompt: u64 = 0;
var c_prompt: Counters = .{};
var calls: [max_calls]Call = undefined;
var n_calls: usize = 0;
var dropped: usize = 0;
/// The model's read-ahead of the layer about to call (its start and end), consumed by the call.
var ra_layer: ?u16 = null;
var ra_t: [2]u64 = .{ 0, 0 };
/// The wide layer about to call: its read-ahead's routes read back, and its graph's steps built (`wide`).
var ra_eval_layer: ?u16 = null;
var ra_step_layer: ?u16 = null;
var ra_step_t: [2]u64 = .{ 0, 0 };
var ra_eval_t: u64 = 0;
var wide_layer: ?u16 = null;
var wide_t: [n_wide]u64 = @splat(0);
/// The probe's votes per decoder layer, at each probe layer; which were taken this prompt.
var votes: [2][max_layers][max_experts]u16 = undefined;
var voted_in: [2][max_layers]bool = @splat(@splat(false));
var probes: [max_layers]Probe = undefined;
var n_probes: usize = 0;
/// The earlier prompts' tail routes per decoder layer (pair counts) and how many prompts gave them.
var prev: [max_layers][max_experts]u32 = @splat(@splat(0));
var prev_n: [max_layers]u32 = @splat(0);

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A new prompt (the source's phase change to prompt): nothing recorded yet.
pub fn reset() void {
    active = false;
    n_calls = 0;
    dropped = 0;
    ra_layer = null;
    ra_eval_layer = null;
    ra_step_layer = null;
    wide_layer = null;
    voted_in = @splat(@splat(false));
    n_probes = 0;
}

fn begin(c: Counters, t: u64) void {
    if (active) return;
    active = true;
    t_prompt = t;
    c_prompt = c;
}

/// The model's read-ahead of `layer` ran from `t0` to `t1` (its routes read back, the streamer's read-ahead posted).
pub fn readAhead(layer: usize, t0: u64, t1: u64) void {
    if (!on) return;
    ra_layer = @intCast(layer);
    ra_t = .{ t0, t1 };
}

/// A wide layer's graph, step by step as the model builds it before its routed call (host clock; nothing evaluates
/// between the steps): the attention-side mix and norm, the projections, every span's attention and output groups, the
/// output projection, the FFN-side mix and norm. The `[prompt-io] wide` line gives each step's build time and the read-
/// ahead's split into its routes' wait and its host work.
pub const Wide = enum { hc, qkv, attn, wob, ffn };
const n_wide = std.enums.values(Wide).len;

pub fn wide(layer: usize, w: Wide) void {
    if (!on) return;
    const l: u16 = @intCast(layer);
    if (wide_layer == null or wide_layer.? != l) {
        wide_layer = l;
        wide_t = @splat(0);
    }
    wide_t[@intFromEnum(w)] = now();
}

/// `split_ra`: the read-ahead of `layer` has its stream's rows (`cat`), then their collapse (`col`), evaluated.
pub const RaStep = enum { cat, col };

pub fn raStep(layer: usize, st: RaStep) void {
    if (!on) return;
    const l: u16 = @intCast(layer);
    if (ra_step_layer == null or ra_step_layer.? != l) {
        ra_step_layer = l;
        ra_step_t = .{ 0, 0 };
    }
    ra_step_t[@intFromEnum(st)] = now();
}

/// The model's read-ahead of `layer` has its routes back (the host's own work follows).
pub fn raEval(layer: usize) void {
    if (!on) return;
    ra_eval_layer = @intCast(layer);
    ra_eval_t = now();
}

/// A wide layer of the prompt begins (its input, before its read-ahead): the prompt timeline's `pre` stage.
pub fn layerBegin(layer: usize) void {
    if (!on) return;
    ptl.stage(layer, .pre);
}

/// The layer of the last routed call begun (the prompt timeline's stages after it).
var tl_layer: usize = 0;

/// A routed call of the prompt begins: `c` the stream's counters now.
pub fn callStart(layer: usize, rows: usize, c: Counters) void {
    if (!on) return;
    tl_layer = layer;
    ptl.stage(layer, .attn);
    const t = now();
    const t_first = if (ra_layer != null and ra_layer.? == layer) ra_t[0] else t;
    begin(c, t_first);
    if (n_calls == max_calls) {
        dropped += 1;
        return;
    }
    const prev_end = if (n_calls > 0) calls[n_calls - 1].t1 else t_prompt;
    const prev_c = if (n_calls > 0) calls[n_calls - 1].c1 else c_prompt;
    var r: Call = .{ .layer = @intCast(layer), .rows = @intCast(rows), .dec = wantsRoutes(layer), .t0 = t, .t_layer = prev_end, .c_layer = prev_c, .c0 = c };
    if (ra_layer) |l| if (l == layer) {
        r.ra0 = ra_t[0];
        r.ra1 = ra_t[1];
    };
    ra_layer = null;
    if (ra_eval_layer) |l| if (l == layer) {
        r.ra_eval = ra_eval_t;
    };
    ra_eval_layer = null;
    if (ra_step_layer) |l| if (l == layer) {
        r.ra_cat = ra_step_t[0];
        r.ra_col = ra_step_t[1];
    };
    ra_step_layer = null;
    if (wide_layer) |l| if (l == layer) {
        r.g = wide_t;
    };
    wide_layer = null;
    calls[n_calls] = r;
    n_calls += 1;
}

/// The call's routes are evaluated (the GPU's attention and router done).
pub fn callRouted() void {
    if (!on) return;
    ptl.stage(tl_layer, .moe);
    if (n_calls == 0) return;
    calls[n_calls - 1].t_idx = now();
}

/// The routed call returned (its groups drained): `c` the stream's counters now.
pub fn callEnd(c: Counters) void {
    if (!on) return;
    ptl.stage(tl_layer, .rest);
    if (n_calls == 0) return;
    const r = &calls[n_calls - 1];
    if (r.t1 != 0) return;
    r.t1 = now();
    r.c1 = c;
}

/// Whether the call of `layer` should read its routes back for the probe: a decoder layer of a replaying chunk.
pub fn wantsRoutes(layer: usize) bool {
    return on and tail > 0 and split > 0 and layer >= split and layer < max_layers;
}

/// The probe's prediction for decoder layer `layer` at probe point `slot` (0: layer 10, 1: layer 19): every tail
/// row's top-k experts.
pub fn predict(slot: usize, layer: usize, ids: []const u32) void {
    if (!on or layer >= max_layers) return;
    const v = &votes[slot][layer];
    @memset(v, 0);
    for (ids) |e| if (e < max_experts) {
        v[e] +|= 1;
    };
    voted_in[slot][layer] = true;
}

/// Decoder layer `layer`'s tail routes (`ids`, its rows' top-k), `cut` its prompt slot rows, `resident(e)` whether
/// expert `e` is resident before the call routes: the probe's coverage, then the routes kept as the static prior.
pub fn routed(layer: usize, ids: []const u16, cut: u32, ctx: anytype, comptime resident: fn (@TypeOf(ctx), u16) bool) void {
    if (!on or layer >= max_layers) return;
    var used: [max_experts]u32 = @splat(0);
    for (ids) |e| if (e < max_experts) {
        used[e] += 1;
    };
    var p: Probe = .{ .layer = @intCast(layer), .cut = cut };
    for (used, 0..) |u, e| if (u > 0) {
        p.actual += 1;
        if (resident(ctx, @intCast(e))) p.res += 1;
    };
    for (0..2) |s| if (voted_in[s][layer]) {
        p.have[s] = true;
        const v = &votes[s][layer];
        var top: [max_experts]u16 = undefined;
        const ranked = rank(u16, v, &top);
        for (ranked[0..@min(ranked.len, cut)]) |e| if (used[e] > 0) {
            p.top[s] += 1;
        };
        p.voted[s] = @intCast(ranked.len);
        for (ranked) |e| if (used[e] > 0) {
            p.voted_hit[s] += 1;
        };
    };
    if (prev_n[layer] > 0) {
        var top: [max_experts]u16 = undefined;
        const ranked = rank(u32, &prev[layer], &top);
        var hit: u32 = 0;
        for (ranked[0..@min(ranked.len, cut)]) |e| hit += @intFromBool(used[e] > 0);
        p.prev = hit;
    }
    for (used, 0..) |u, e| prev[layer][e] +|= u;
    prev_n[layer] += 1;
    if (n_probes < probes.len) {
        probes[n_probes] = p;
        n_probes += 1;
    }
}

/// The experts with a positive count, the larger count first, ties by the lower id.
fn rank(comptime C: type, counts: *const [max_experts]C, out: *[max_experts]u16) []u16 {
    var n: usize = 0;
    for (counts, 0..) |c, e| if (c > 0) {
        out[n] = @intCast(e);
        n += 1;
    };
    std.sort.pdq(u16, out[0..n], counts, struct {
        fn less(cs: *const [max_experts]C, x: u16, y: u16) bool {
            return if (cs[x] != cs[y]) cs[x] > cs[y] else x < y;
        }
    }.less);
    return out[0..n];
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

fn gb(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

fn share(n: u32, d: u32) f64 {
    return if (d == 0) 0 else @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(d));
}

const Side = struct {
    calls: u32 = 0,
    layer_ns: u64 = 0,
    ra_ns: u64 = 0,
    attn_ns: u64 = 0,
    moe_ns: u64 = 0,
    d: Counters = .{},
    uniq: u64 = 0,

    fn add(s: *Side, r: *const Call) void {
        s.calls += 1;
        s.layer_ns += r.t1 -| r.t_layer;
        s.ra_ns += r.ra1 -| r.ra0;
        s.attn_ns += r.t_idx -| r.t0;
        s.moe_ns += r.t1 -| r.t_idx;
        const dl = r.c1.since(r.c_layer);
        inline for (@typeInfo(Counters).@"struct".field_names) |f| @field(s.d, f) += @field(dl, f);
        const dc = r.c1.since(r.c0);
        s.uniq += dc.misses + dc.hits;
    }

    fn print(s: *const Side, name: []const u8) void {
        std.debug.print("[prompt-io] side={s} calls={d} layer_ms={d:.1} ra_ms={d:.1} attn_ms={d:.1} moe_ms={d:.1} uniq={d} read_gb={d:.2} rwall_ms={d:.1} rwait_ms={d:.1} misses={d} hits={d} ahead_hits={d} ahead_gb={d:.2} fill_posted={d} fill_gb={d:.2}\n", .{ name, s.calls, ms(s.layer_ns), ms(s.ra_ns), ms(s.attn_ns), ms(s.moe_ns), s.uniq, gb(s.d.bytes), ms(s.d.wall_ns), ms(s.d.wait_ns), s.d.misses, s.d.hits, s.d.ahead_hits, gb(s.d.ahead_bytes), s.d.fill_posted, gb(s.d.fill_bytes) });
    }
};

/// The prompt ended (the source's phase change to decode): its lines, `c` the stream's counters now.
/// A wide call's split (ns): its read-ahead's wait for its routes and its host work after, then each graph step's build
/// (from the read-ahead's end: the mix, projections, spans, output projection, FFN-side mix) and the router's to the call.
fn wideSteps(r: *const Call) [8]u64 {
    var out: [8]u64 = @splat(0);
    if (r.ra_eval != 0 and r.ra0 != 0) {
        out[0] = r.ra_eval -| r.ra0;
        out[1] = r.ra1 -| r.ra_eval;
    }
    var last = if (r.ra1 != 0) r.ra1 else r.g[0];
    for (r.g, 0..) |t, k| {
        if (t == 0) continue;
        out[2 + k] = t -| last;
        last = t;
    }
    out[7] = r.t0 -| last;
    return out;
}

pub fn report(c: Counters) void {
    if (on) ptl.stage(tl_layer, .end);
    if (!on or !active) {
        reset();
        return;
    }
    const t_end = now();
    var enc: Side = .{};
    var dec: Side = .{};
    var all: Side = .{};
    for (calls[0..n_calls], 0..) |*r, i| {
        if (r.t1 == 0) continue;
        const dl = r.c1.since(r.c_layer);
        const dc = r.c1.since(r.c0);
        const layer_ns = r.t1 -| r.t_layer;
        const ra_ns = r.ra1 -| r.ra0;
        const attn_ns = r.t_idx -| r.t0;
        const moe_ns = r.t1 -| r.t_idx;
        std.debug.print("[prompt-io] call={d} layer={d} rows={d} dec={d} uniq={d} ra_ms={d:.1} attn_ms={d:.1} moe_ms={d:.1} other_ms={d:.1} layer_ms={d:.1} read_gb={d:.3} rwall_ms={d:.1} rwait_ms={d:.1} misses={d} hits={d} ahead_hits={d} ahead_posted={d} staged={d} adopted={d} early={d}\n", .{ i, r.layer, r.rows, @intFromBool(r.dec), dc.misses + dc.hits, ms(ra_ns), ms(attn_ns), ms(moe_ns), ms(layer_ns -| (ra_ns + attn_ns + moe_ns)), ms(layer_ns), gb(dl.bytes), ms(dl.wall_ns), ms(dl.wait_ns), dc.misses, dc.hits, dc.ahead_hits, dl.ahead_posted, dl.stage_posted, dl.stage_adopted, dl.stage_early });
        if (r.g[0] != 0) {
            const steps = wideSteps(r);
            std.debug.print("[prompt-io] wide call={d} layer={d} ra_wait_ms={d:.1} ra_host_ms={d:.1} hc_ms={d:.1} qkv_ms={d:.1} attn_ms={d:.1} wob_ms={d:.1} ffn_ms={d:.1} route_ms={d:.1}", .{ i, r.layer, ms(steps[0]), ms(steps[1]), ms(steps[2]), ms(steps[3]), ms(steps[4]), ms(steps[5]), ms(steps[6]), ms(steps[7]) });
            if (r.ra_cat != 0 and r.ra_col != 0 and r.ra_eval != 0) std.debug.print(" ra_cat_ms={d:.1} ra_col_ms={d:.1} ra_route_ms={d:.1}", .{ ms(r.ra_cat -| r.ra0), ms(r.ra_col -| r.ra_cat), ms(r.ra_eval -| r.ra_col) });
            std.debug.print("\n", .{});
        }
        all.add(r);
        if (r.dec) dec.add(r) else enc.add(r);
    }
    enc.print("encoder");
    dec.print("decoder");
    all.print("total");
    const d = c.since(c_prompt);
    std.debug.print("[stream-stats prompt] wall_ms={d:.1} routes={d} hits={d} misses={d} evictions={d} read_gb={d:.2} read_wait_ms={d:.1} read_wall_ms={d:.1} ahead_posted={d} ahead_hits={d} ahead_gb={d:.2} spec_issued={d} spec_landed={d} claimed={d} fill_posted={d} fill_gb={d:.2} stage_posted={d} stage_adopted={d} stage_gb={d:.2} stage_early={d} calls={d} dropped={d}\n", .{ ms(t_end -| t_prompt), d.routes, d.hits, d.misses, d.evictions, gb(d.bytes), ms(d.wait_ns), ms(d.wall_ns), d.ahead_posted, d.ahead_hits, gb(d.ahead_bytes), d.spec_issued, d.spec_landed, d.claimed, d.fill_posted, gb(d.fill_bytes), d.stage_posted, d.stage_adopted, gb(d.stage_bytes), d.stage_early, n_calls, dropped });
    for (probes[0..n_probes]) |p| {
        var buf: [32]u8 = undefined;
        const prev_s = if (p.prev) |h| std.fmt.bufPrint(&buf, "{d:.2}", .{share(h, p.actual)}) catch "?" else "-";
        std.debug.print("[ced-probe] layer={d} actual={d} cut={d} p10={d:.2} p19={d:.2} u10={d:.2}/{d} u19={d:.2}/{d} res={d:.2} prev={s} have={d}{d}\n", .{ p.layer, p.actual, p.cut, share(p.top[0], p.actual), share(p.top[1], p.actual), share(p.voted_hit[0], p.actual), p.voted[0], share(p.voted_hit[1], p.actual), p.voted[1], share(p.res, p.actual), prev_s, @intFromBool(p.have[0]), @intFromBool(p.have[1]) });
    }
    reset();
}

test "prompt_io wide marks: a layer's call takes its own marks, another layer's are dropped, the split adds up" {
    const was = on;
    defer on = was;
    on = true;
    reset();
    defer reset();
    const r: Call = .{ .ra0 = 100, .ra1 = 160, .ra_eval = 140, .t0 = 400, .g = .{ 170, 200, 300, 330, 360 } };
    const steps = wideSteps(&r);
    try std.testing.expectEqualSlices(u64, &.{ 40, 20, 10, 30, 100, 30, 30, 40 }, &steps);
    var sum: u64 = 0;
    for (steps[2..]) |x| sum += x;
    try std.testing.expectEqual(r.t0 - r.ra1, sum);
    wide(3, .hc);
    wide(4, .hc);
    raEval(4);
    callStart(4, 16384, .{});
    try std.testing.expect(calls[0].g[0] != 0 and calls[0].ra_eval != 0);
    wide(5, .hc);
    callStart(6, 128, .{});
    try std.testing.expectEqual(@as(u64, 0), calls[1].g[0]);
    raStep(7, .cat);
    raStep(7, .col);
    callStart(7, 16384, .{});
    try std.testing.expect(calls[2].ra_cat != 0 and calls[2].ra_col >= calls[2].ra_cat);
    raStep(8, .cat);
    callStart(9, 16384, .{});
    try std.testing.expectEqual(@as(u64, 0), calls[3].ra_cat);
}

test "prompt_io rank: larger count first, ties by the lower id, zeros dropped" {
    var counts: [max_experts]u32 = @splat(0);
    counts[7] = 3;
    counts[2] = 3;
    counts[9] = 5;
    counts[1] = 1;
    var out: [max_experts]u16 = undefined;
    const r = rank(u32, &counts, &out);
    try std.testing.expectEqualSlices(u16, &.{ 9, 2, 7, 1 }, r);
}

test "prompt_io probe coverage: top cut, every voted, resident and the earlier prompts' prior" {
    const was = on;
    defer on = was;
    on = true;
    reset();
    prev = @splat(@splat(0));
    prev_n = @splat(0);
    // layer 20: rows voted 1,1,1,2,2,3 at layer 10; 5 only at layer 19
    predict(0, 20, &.{ 1, 1, 1, 2, 2, 3 });
    predict(1, 20, &.{5});
    const R = struct {
        fn f(_: void, e: u16) bool {
            return e == 2;
        }
    };
    routed(20, &.{ 1, 2, 4, 4 }, 2, {}, R.f);
    try std.testing.expectEqual(@as(usize, 1), n_probes);
    const p = probes[0];
    try std.testing.expectEqual(@as(u32, 3), p.actual); // 1, 2, 4
    try std.testing.expectEqual(@as(u32, 2), p.top[0]); // top 2 by votes: 1, 2
    try std.testing.expectEqual(@as(u32, 2), p.voted_hit[0]); // 1, 2 of 1, 2, 3
    try std.testing.expectEqual(@as(u32, 3), p.voted[0]);
    try std.testing.expectEqual(@as(u32, 0), p.top[1]);
    try std.testing.expectEqual(@as(u32, 1), p.res);
    try std.testing.expect(p.prev == null);
    // a second prompt: the first one's routes are the prior (4 twice, then 1, 2)
    reset();
    routed(20, &.{ 4, 7 }, 1, {}, R.f);
    try std.testing.expectEqual(@as(?u32, 1), probes[0].prev);
}
