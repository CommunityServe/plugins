//! `MLX_SERVE_DSV41_ROUTED_ONE`: a wide prompt call's resident routed experts as one turboexl3 `moeClamped` call over the
//! layer's base slot bank, in the router's [rows, top-k] layout, for the copied expert streamer
//! (sdk/weights/streamed/exl3_source.zig `Source.rows`, patches/turboexl3-wide-one.patch). The plan is MLX-free and
//! tested (src/one_plan.zig).
//!
//! The streamer's wide lane runs a call's pairs as DIG-X waves of at most 8 experts (five launches and one async eval
//! each, two waves in flight): at 16K rows about 48 waves a layer. It joins the waves' rows per pair ([rows, top-k,
//! hidden]: a concatenate and a take in the call, another pair in the lane's join) and the source weighs and sums them.
//! The one call is turboexl3's prefill chain over every pair (eight launches), with the weighted sum in its finish.
//!
//! - A call whose experts are all resident runs as the one call alone: the wide lane's barrier, read-ahead landing and
//!   seed, then the residents' routes (all hits, their rows held as the lane's deferred base call holds them), the call.
//! - A call with streamed experts (mode `1` and `check`, its resident pairs at least half the call's: `one_plan.takes`)
//!   runs its resident pairs as the one call, every streamed pair in it at weight 0, committed before the streamed
//!   experts' routes, so it runs while their reads land. The streamed experts then run on the lane's own schedule:
//!   routes `depth` groups ahead, every part's reads waited, the transient bank's waves drained before the group goes
//!   back, a persistent load's rows in a last call over the base bank. A combine adds their weighted rows to the one
//!   call's sum, in each row's top-k order.
//! - Any other call (`one_plan.takes` false, a decode-width call, the decode phase) keeps the wide lane.
//!
//! Exactness: turboexl3's GEMMs sum in another order than DIG-X's and round their inner products to f16, and the lane
//! rounds each pair's row to bf16 before the sum, so the one call's sum agrees with the lane's within rounding, not bit
//! for bit. `check` runs both on every call the one path takes, serves the lane's, and logs the comparison against the
//! bar turboexl3 holds two summation orders of one EXL3 GEMM to (rms of the difference within 2^-8 of the reference's
//! rms, the largest difference within 8e-2 of it).

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const log = @import("mlx_serve/sdk/util/log.zig");
const plan = @import("one_plan.zig");
const exact = @import("one_exact.zig");

/// Set at load from `MLX_SERVE_DSV41_ROUTED_ONE` (`one_plan.modeOf`), before the bank opens.
pub var mode: plan.Mode = .off;
pub const modeOf = plan.modeOf;

pub fn on() bool {
    return mode != .off;
}

/// The bar of `check`: turboexl3's for one EXL3 GEMM in two summation orders (its gpu_test.zig DENSE_PF_PARITY).
pub const bar_rms_diff: f64 = 3.9e-3;
pub const bar_ratio: f64 = 8e-2;

/// What the path did since the last request's end (`meters`).
pub const Stats = struct {
    wide: u64 = 0,
    /// Calls the one call served whole (every expert resident), and with streamed experts.
    one: u64 = 0,
    mixed: u64 = 0,
    /// Calls the wide lane kept.
    lane: u64 = 0,
    pairs: u64 = 0,
    /// Streamed pairs the one calls computed at weight 0.
    masked: u64 = 0,
    checks: u64 = 0,
    check_fail: u64 = 0,
    /// Checks over the bar beyond the lane's bf16 stores (`Gpu.beyond`).
    rounded_fail: u64 = 0,
    worst_ratio: f64 = 0,
    worst_rms_diff: f64 = 0,
    worst_rel: f64 = 0,
    worst_beyond: f64 = 0,
    /// Calls compared to the exact sum, and each path's worst error there (relative to the element, to its budget), and
    /// the lane's pair rows' worst.
    exact_calls: u64 = 0,
    exact_one_rel: f64 = 0,
    exact_lane_rel: f64 = 0,
    exact_one_mag: f64 = 0,
    exact_lane_mag: f64 = 0,
    exact_lane_pairs: f64 = 0,
};

/// The source's state for the path: the pack's SwiGLU clamp, the combine kernel, the host scratch, the counters.
pub const State = struct {
    limit: c_int = 0,
    combine: ?mlx.mlx_fast_metal_kernel = null,
    ids: std.ArrayList(u16) = .empty,
    resident: std.ArrayList(bool) = .empty,
    count: std.ArrayList(u32) = .empty,
    distinct: std.ArrayList(u16) = .empty,
    slot_of: std.ArrayList(u32) = .empty,
    slots: std.ArrayList(u32) = .empty,
    keep: std.ArrayList(f32) = .empty,
    start: std.ArrayList(u32) = .empty,
    by_expert: std.ArrayList(u32) = .empty,
    loc: std.ArrayList(i32) = .empty,
    seen: std.ArrayList(bool) = .empty,
    /// One group's transient rows (slot, act row, routed position), and the persistent loads' for the last call.
    w_slot: std.ArrayList(u32) = .empty,
    w_act: std.ArrayList(u32) = .empty,
    w_pos: std.ArrayList(u32) = .empty,
    d_slot: std.ArrayList(u32) = .empty,
    d_act: std.ArrayList(u32) = .empty,
    d_pos: std.ArrayList(u32) = .empty,
    /// The streamed experts' wave calls' outputs (kept), the combine's source in call order.
    kept: std.ArrayList(mlx.mlx_array) = .empty,
    stats: Stats = .{},

    const lists = .{ "ids", "resident", "count", "distinct", "slot_of", "slots", "keep", "start", "by_expert", "loc", "seen", "w_slot", "w_act", "w_pos", "d_slot", "d_act", "d_pos" };

    /// `limit`: the pack's `swiglu_limit`, a whole number turboexl3's clamped chain takes (1..128).
    pub fn init(limit: f64) !State {
        if (limit < 1 or limit > 128 or limit != @floor(limit)) return error.RoutedOneClamp;
        var s: State = .{ .limit = @intFromFloat(limit) };
        if (mode == .on or mode == .check) s.combine = try buildCombine();
        return s;
    }

    pub fn deinit(s: *State, a: std.mem.Allocator) void {
        if (s.combine) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
        inline for (lists) |name| @field(s, name).deinit(a);
        for (s.kept.items) |x| _ = mlx.mlx_array_free(x);
        s.kept.deinit(a);
        s.* = .{};
    }
};

/// Layer `layer`'s routed sum, f32 [n, hidden], of a prompt call (`ids` int32 [n, k], `wts` its weights, `x` the MoE
/// input [n, hidden]) through the one call, or null when the call keeps the wide lane (the caller runs it). `g`: the
/// source's backend; `h`: its hook (the lane's executor over the stream); `route_ids`: a route's width, past which a
/// call is wide. The result is kept (the caller's).
pub fn rows(st: *State, a: std.mem.Allocator, g: anytype, h: anytype, layer: u32, x: mlx.mlx_array, ids: mlx.mlx_array, wts: mlx.mlx_array, route_ids: usize) !?mlx.mlx_array {
    const shp = g.shapeOf(ids);
    const n: usize = @intCast(shp.dim(0));
    const k: usize = @intCast(shp.dim(1));
    if (n * k <= route_ids) return null;
    st.stats.wide += 1;
    // turboexl3's chain takes a bank whose gate, up and down share a rate: a layer that does not keeps the lane.
    const base = (try h.source.bankArrays(g, layer, .base)) orelse return error.RoutedOneNoBaseBank;
    if (!sameRate(base)) {
        st.stats.lane += 1;
        return null;
    }
    const ne: usize = h.n_experts;
    // The routing barrier, as the lane's (its own read of the ids is then a copy of what this eval produced).
    try st.ids.resize(a, n * k);
    const host = try g.hostIds(ids, st.ids.items);
    // The layer's read-ahead lands before its residency is read: an admitted record holds its row before it is there.
    if (h.wide_route.read_ahead) try h.source.awaitReadAhead(layer);
    try st.resident.resize(a, ne);
    for (st.resident.items, 0..) |*r, e| r.* = h.source.isResident(layer, @intCast(e));
    try st.count.resize(a, ne);
    try st.distinct.resize(a, ne);
    const sp = plan.split(host, st.resident.items, st.count.items, st.distinct.items);
    if (!plan.takes(mode, sp)) {
        st.stats.lane += 1;
        return null;
    }
    const y = try one(st, a, g, h, layer, x, wts, n, k, sp, route_ids, base);
    if (mode != .check) return g.keep(y);
    // The one call's rows are no longer held: the lane's loads may take them, so its sum is evaluated first.
    try g.evalAll(&.{y});
    // The lane's sum as `Source.rows` makes it: its rows [n, k, hidden], weighted, summed over k in f32.
    const lane32 = try g.astype(try h.at(layer).routed(g, x, ids), .float32);
    const w32 = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
    const ref = try g.sum(try g.mul(lane32, try g.expandDims(w32, -1)), -2, false);
    try check(st, a, g, h, layer, x, w32, y, ref, lane32, n, k, sp, base);
    return g.keep(ref);
}

/// The one call over the resident pairs (and, with streamed experts, `streamed`'s rows added): f32 [n, hidden].
fn one(st: *State, a: std.mem.Allocator, g: anytype, h: anytype, layer: u32, x: mlx.mlx_array, wts: mlx.mlx_array, n: usize, k: usize, sp: plan.Split, route_ids: usize, base: anytype) !mlx.mlx_array {
    const src = h.source;
    // The lane's seed: the call's hottest residents protected, its hottest misses admitted first.
    try src.seedPrefill(layer, st.ids.items);
    const res = sp.resident();
    try st.slot_of.resize(a, h.n_experts);
    @memset(st.slot_of.items, plan.no_slot);
    // The residents' rows pinned and held past their routes, as the lane's deferred base call holds its rows: no route
    // of the call takes them, and no read lands in them before the one call has run.
    var held = false;
    errdefer if (held) src.releaseHeld();
    var i: usize = 0;
    while (i < res.len) : (i += route_ids) {
        const grp = res[i..@min(i + route_ids, res.len)];
        const call = try src.route(layer, grp, &.{});
        {
            errdefer src.release(call);
            const sv = src.served(call);
            if (sv.n_parts != 0) return error.RoutedOneNotResident;
            for (grp, sv.refs, sv.waves) |e, ref, wave| {
                if (ref.bank != .base or wave != 0) return error.RoutedOneNotResident;
                st.slot_of.items[e] = ref.row;
            }
            held = true;
            try src.holdBase(call);
        }
        src.release(call);
    }
    // The call's tables over [rows, top-k]: each pair its expert's base row; a streamed pair a resident's at weight 0.
    try st.slots.resize(a, n * k);
    try st.keep.resize(a, n * k);
    const masked = plan.table(st.ids.items, st.slot_of.items, res, st.slots.items, st.keep.items);
    var bank: t3.Bank = undefined;
    inline for (.{ "gate", "up", "down" }) |f| {
        const p = @field(base, f);
        @field(bank, f) = .{ .trellis = try viewU16(g, p.code), .suh = p.rin, .svh = p.rout };
    }
    const hidden: c_int = h.hidden;
    const rn: c_int = @intCast(n);
    const rk: c_int = @intCast(k);
    // The lane's activations are the bf16 rows; the call reads them in f32, so its sum comes back in f32 as the lane's.
    const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
    const x3 = try g.reshape(try g.astype(xb, .float32), &.{ 1, rn, hidden });
    const inds = try g.hostArray(std.mem.sliceAsBytes(st.slots.items), &.{ 1, rn, rk }, .uint32);
    const w32 = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
    var sc = try g.reshape(w32, &.{ 1, rn, rk });
    if (masked > 0) sc = try g.mul(sc, try g.hostArray(std.mem.sliceAsBytes(st.keep.items), &.{ 1, rn, rk }, .float32));
    const y3 = try g.adopt(try t3.moeClamped(g.s, x3, bank, inds, sc, .mul1, st.limit));
    var y = try g.reshape(y3, &.{ rn, hidden });
    // Committed now: the GPU runs it while the host routes the streamed experts and their reads land.
    try g.asyncEval(&.{y});
    st.stats.pairs += n * k;
    st.stats.masked += masked;
    if (sp.allResident()) {
        st.stats.one += 1;
    } else {
        st.stats.mixed += 1;
        y = try streamed(st, a, g, h, layer, xb, w32, y, n, k, sp, route_ids);
    }
    // Every later write into these rows (a route or a read-ahead of this layer) follows a host wait on later work.
    held = false;
    src.releaseHeld();
    return y;
}

/// The lane's routes in flight at most (`expert_stream.max_wide_depth` is 5).
const max_depth = 8;

/// The streamed experts of a call whose resident pairs ran as the one call `y_res` (f32 [n, hidden]): their pairs on the
/// lane's schedule (`act` the bf16 rows), their weighted rows added to `y_res` (`w32` f32 [n, k]).
fn streamed(st: *State, a: std.mem.Allocator, g: anytype, h: anytype, layer: u32, act: mlx.mlx_array, w32: mlx.mlx_array, y_res: mlx.mlx_array, n: usize, k: usize, sp: plan.Split, route_ids: usize) !mlx.mlx_array {
    const src = h.source;
    const strm = sp.streamed();
    const n_groups = plan.groups(strm.len, route_ids);
    const depth: usize = @max(1, @min(@as(usize, h.wide_route.depth), max_depth));
    try st.start.resize(a, h.n_experts);
    try st.by_expert.resize(a, n * k);
    plan.bucket(st.ids.items, st.count.items, st.start.items, st.by_expert.items);
    try st.loc.resize(a, n * k);
    @memset(st.loc.items, -1);
    st.d_slot.clearRetainingCapacity();
    st.d_act.clearRetainingCapacity();
    st.d_pos.clearRetainingCapacity();
    defer {
        for (st.kept.items) |o| g.release(o);
        st.kept.clearRetainingCapacity();
    }
    const Call = @TypeOf(src.*).Call;
    var calls: [max_depth]?*Call = @splat(null);
    errdefer for (&calls) |*c| if (c.*) |cl| {
        src.release(cl);
        c.* = null;
    };
    for (0..@min(depth, n_groups)) |gi| calls[gi % depth] = try src.route(layer, plan.group(strm, route_ids, gi), &.{});
    var at: u32 = 0;
    for (0..n_groups) |gi| {
        const grp = plan.group(strm, route_ids, gi);
        const call = calls[gi % depth].?;
        const sv = src.served(call);
        for (0..sv.n_parts) |p| {
            try src.waitGu(call, @intCast(p));
            try src.waitDown(call, @intCast(p));
        }
        st.w_slot.clearRetainingCapacity();
        st.w_act.clearRetainingCapacity();
        st.w_pos.clearRetainingCapacity();
        const Kind = @TypeOf(sv.refs[0].bank);
        for (grp, sv.refs) |e, ref| {
            const prs = st.by_expert.items[st.start.items[e]..][0..st.count.items[e]];
            const into: [3]*std.ArrayList(u32) = switch (ref.bank) {
                .transient => .{ &st.w_slot, &st.w_act, &st.w_pos },
                // A seed miss loaded into a persistent row: the last call reads it, as the lane's deferred base call.
                .base => .{ &st.d_slot, &st.d_act, &st.d_pos },
                .ext => return error.RoutedOneExtBank,
            };
            for (prs) |p| {
                try into[0].append(a, ref.row);
                try into[1].append(a, @intCast(@as(usize, p) / k));
                try into[2].append(a, p);
            }
        }
        const k0 = st.kept.items.len;
        if (st.w_slot.items.len > 0) {
            const tb = h.banks[layer][@intFromEnum(@as(Kind, .transient))] orelse return error.RoutedOneNoTransientBank;
            try st.kept.ensureUnusedCapacity(a, 1);
            st.kept.appendAssumeCapacity(try h.math.prefill(g, layer, act, .{ .slot = st.w_slot.items, .act_row = st.w_act.items }, tb));
            at = plan.place(st.loc.items, st.w_pos.items, at);
        }
        // The group's waves drained before its transient rows go back (the next route may read into them).
        try h.math.finishPrefill(g);
        try g.evalAll(st.kept.items[k0..]);
        try src.holdBase(call);
        src.release(call);
        calls[gi % depth] = null;
        if (gi + depth < n_groups) calls[gi % depth] = try src.route(layer, plan.group(strm, route_ids, gi + depth), &.{});
    }
    if (st.d_slot.items.len > 0) {
        const bb = (try src.bankArrays(g, layer, .base)) orelse return error.RoutedOneNoBaseBank;
        try st.kept.ensureUnusedCapacity(a, 1);
        st.kept.appendAssumeCapacity(try h.math.prefill(g, layer, act, .{ .slot = st.d_slot.items, .act_row = st.d_act.items }, bb));
        at = plan.place(st.loc.items, st.d_pos.items, at);
        try h.math.finishPrefill(g);
        try g.evalAll(st.kept.items[st.kept.items.len - 1 ..]);
    }
    try st.seen.resize(a, at);
    if (!plan.placedOnce(st.ids.items, st.slot_of.items, st.loc.items, at, st.seen.items)) return error.RoutedOnePlacement;
    if (at == 0) return y_res;
    const rows_src = if (st.kept.items.len == 1) st.kept.items[0] else try g.concat(st.kept.items, 0);
    const nk: c_int = @intCast(n * k);
    const loc = try g.hostArray(std.mem.sliceAsBytes(st.loc.items), &.{nk}, .int32);
    return combineRows(st, g, y_res, rows_src, loc, try g.reshape(w32, &.{nk}), n, @intCast(h.hidden), k);
}

/// The combine: one output element a thread, the one call's sum plus each streamed pair's row times its weight, in
/// the row's top-k order, in f32 (`loc[r * K + j]` the pair's row in `src`, -1 for a resident pair).
const combine_source =
    \\const uint i = thread_position_in_grid.x;
    \\const uint r = i / D, c = i % D;
    \\if (r >= uint(y_shape[0])) return;
    \\float acc = y[i];
    \\for (int j = 0; j < K; ++j) {
    \\    const int l = loc[r * K + j];
    \\    if (l >= 0) acc += float(src[size_t(l) * D + c]) * w[r * K + j];
    \\}
    \\out[i] = acc;
;

fn buildCombine() !mlx.mlx_fast_metal_kernel {
    const ins: []const [*:0]const u8 = &.{ "y", "src", "loc", "w" };
    const outs: []const [*:0]const u8 = &.{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kern = mlx.mlx_fast_metal_kernel_new("turboexl3_ds41_routed_one_combine", iv, ov, combine_source, "", true, false);
    if (kern.ctx == null) return error.RoutedOneCombineKernel;
    return kern;
}

fn combineRows(st: *State, g: anytype, y: mlx.mlx_array, src_rows: mlx.mlx_array, loc: mlx.mlx_array, w: mlx.mlx_array, n: usize, hidden: usize, k: usize) !mlx.mlx_array {
    const kern = st.combine orelse return error.RoutedOneCombineKernel;
    const total = n * hidden;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ @intCast(n), @intCast(hidden) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, shape.len, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast((total + 255) / 256 * 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", @intCast(hidden)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    const ins = [_]mlx.mlx_array{ y, src_rows, loc, w };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, g.s));
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&o, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return g.adopt(o);
}

/// Whether a bank's gate, up and down code share a rate (their packed tiles' last dimension).
fn sameRate(b: anytype) bool {
    const r = lastDim(b.gate.code);
    return lastDim(b.up.code) == r and lastDim(b.down.code) == r;
}

fn lastDim(x: mlx.mlx_array) c_int {
    const sh = mlx.getShape(x);
    return if (sh.len == 0) 0 else sh[sh.len - 1];
}

/// A slot bank's trellis code (int16, as the streamer reads it into the rows) as turboexl3's u16, in place.
fn viewU16(g: anytype, code: mlx.mlx_array) !mlx.mlx_array {
    var v = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&v, code, .uint16, g.s)) catch |e| {
        _ = mlx.mlx_array_free(v);
        return e;
    };
    return g.adopt(v);
}

/// The lane's per-pair rows are stored in bf16 (`routeExpertBf16`: the DIG-X waves write bf16): each moves by at most half
/// a unit in its last place, 2^-8 of its value.
const bf16_half_ulp: f64 = 1.0 / 256.0;
/// `check` also compares a call of this many rows or more to the exact sum, on at most `exact_rows` sampled rows.
const exact_min_rows = 1024;
const exact_rows = 8;
const Exact = exact.Exact(t3.format);

/// `check`'s comparison on the GPU, f32, over the call's n x hidden elements, d = one - lane.
const Gpu = struct {
    max_abs: f64,
    rms_ref: f64,
    /// rms(d) / rms(lane)
    rms_diff: f64,
    max_ref: f64,
    /// max |d| / max(|lane|, rms_ref): the element-wise relative error, floored at the reference's rms
    max_rel: f64,
    /// max(|d| - 2^-8 sum_j |w_j| |lane row_j|): the difference beyond the lane's bf16 stores
    beyond: f64,
    /// at the largest |d|: the lane's value, the one path's, and the lane's budget sum_j |w_j| |row_j|
    ref_at: f64,
    one_at: f64,
    mag_at: f64,
    /// flat indices of the largest |d|, |lane|, relative error and excess, and the elements past 1e-2 relative
    at_abs: usize,
    at_ref: usize,
    at_rel: usize,
    at_beyond: usize,
    over: u32,
};

fn gpuCompare(g: anytype, y: mlx.mlx_array, ref: mlx.mlx_array, lane32: mlx.mlx_array, w32: mlx.mlx_array) !Gpu {
    const total: c_int = @intCast(mlx.mlx_array_size(ref));
    const yf = try g.reshape(y, &.{total});
    const rf = try g.reshape(ref, &.{total});
    const d = try g.sub(yf, rf);
    const ad = try g.abs(d);
    const ar = try g.abs(rf);
    const ms_d = try g.mean(try g.square(d), 0, false);
    const ms_r = try g.mean(try g.square(rf), 0, false);
    const rel = try g.div(ad, try g.maximum(ar, try g.sqrt(ms_r)));
    const mag = try g.reshape(try g.sum(try g.mul(try g.abs(lane32), try g.abs(try g.expandDims(w32, -1))), -2, false), &.{total});
    const beyond = try g.sub(ad, try g.mul(mag, try g.scalar(bf16_half_ulp, .float32)));
    const i_abs = try g.argmax(ad, 0);
    const i_ref = try g.argmax(ar, 0);
    const i_rel = try g.argmax(rel, 0);
    const i_beyond = try g.argmax(beyond, 0);
    const over = try g.astype(try g.sum(try g.astype(try g.greater(rel, try g.scalar(1e-2, .float32)), .uint32), 0, false), .uint32);
    var v: [9]f32 = undefined;
    _ = try g.hostF32(try g.stack(&.{ try g.max(ad, 0, false), ms_d, ms_r, try g.max(ar, 0, false), try g.max(rel, 0, false), try g.max(beyond, 0, false), try g.take(rf, i_abs, 0), try g.take(yf, i_abs, 0), try g.take(mag, i_abs, 0) }, 0), &v);
    var ix: [5]u32 = undefined;
    _ = try g.hostU32(try g.stack(&.{ i_abs, i_ref, i_rel, i_beyond, over }, 0), &ix);
    const rms_ref = @sqrt(@as(f64, v[2]));
    return .{
        .max_abs = v[0],
        .rms_ref = rms_ref,
        .rms_diff = @sqrt(@as(f64, v[1])) / floorOf(rms_ref),
        .max_ref = v[3],
        .max_rel = v[4],
        .beyond = v[5],
        .ref_at = v[6],
        .one_at = v[7],
        .mag_at = v[8],
        .at_abs = ix[0],
        .at_ref = ix[1],
        .at_rel = ix[2],
        .at_beyond = ix[3],
        .over = ix[4],
    };
}

fn floorOf(rms: f64) f64 {
    return if (rms > 0) rms else 1;
}

/// `check`: the one path's sum `y` against the lane's `ref` (from its rows `lane32` [n, k, hidden] f32 and the weights
/// `w32` [n, k] f32), logged with where the largest difference is and what the lane's bf16 stores allow there; a call of
/// `exact_min_rows` rows or more is also compared to the exact sum on sampled rows (`exactSample`).
fn check(st: *State, a: std.mem.Allocator, g: anytype, h: anytype, layer: u32, x: mlx.mlx_array, w32: mlx.mlx_array, y: mlx.mlx_array, ref: mlx.mlx_array, lane32: mlx.mlx_array, n: usize, k: usize, sp: plan.Split, base: anytype) !void {
    const hidden: usize = @intCast(h.hidden);
    // The row's weights and the exact sample take a top-k of at most 16 (DeepSeek-V4.1's is 6).
    if (k > 16) return error.RoutedOneCheckTopK;
    const c = try gpuCompare(g, y, ref, lane32, w32);
    const fl = floorOf(c.rms_ref);
    const ratio = c.max_abs / fl;
    const beyond = c.beyond / fl;
    const ok = c.rms_diff <= bar_rms_diff and ratio <= bar_ratio;
    const ok_rounded = c.rms_diff <= bar_rms_diff and beyond <= bar_ratio;
    st.stats.checks += 1;
    if (!ok) st.stats.check_fail += 1;
    if (!ok_rounded) st.stats.rounded_fail += 1;
    // NaN compares false: a NaN check is a failure, and each worst stays the last finite one.
    if (ratio > st.stats.worst_ratio) st.stats.worst_ratio = ratio;
    if (c.rms_diff > st.stats.worst_rms_diff) st.stats.worst_rms_diff = c.rms_diff;
    if (c.max_rel > st.stats.worst_rel) st.stats.worst_rel = c.max_rel;
    if (beyond > st.stats.worst_beyond) st.stats.worst_beyond = beyond;
    // The largest difference's row (its position in the call) and column, and the row's experts and weights.
    const row = c.at_abs / hidden;
    var wv: [16]f32 = undefined;
    const sel = try g.hostArray(std.mem.sliceAsBytes(&[_]u32{@intCast(row)}), &.{1}, .uint32);
    const wrow = try g.hostF32(try g.take(w32, sel, 0), wv[0..k]);
    var eb: [256]u8 = undefined;
    var ew = std.Io.Writer.fixed(&eb);
    for (st.ids.items[row * k ..][0..k], wrow) |e, w| ew.print("{d}:{d:.4},", .{ e, w }) catch {};
    log.info("[routed-one] check layer={d} rows={d} pairs={d} resident_pairs={d} experts={d}/{d} max_abs={e:.3} at={d}:{d} ref={e:.4} one={e:.4} mag={e:.4} row_experts={s} max_ref={e:.3} max_ref_row={d} rms_ref={e:.3} ratio={e:.3} rms_diff={e:.3} max_rel={e:.3} max_rel_at={d}:{d} over_1e-2={d} beyond_bf16={e:.3} beyond_at={d}:{d} bar={e:.1}/{e:.1} {s} rounded={s}\n", .{
        layer,                       n,                           n * k,          sp.resident_pairs, sp.n_resident,  sp.distinct.len, c.max_abs, row, c.at_abs % hidden, c.ref_at, c.one_at, c.mag_at, ew.buffered(),
        c.max_ref,                   c.at_ref / hidden,           c.rms_ref,      ratio,             c.rms_diff,     c.max_rel,       c.at_rel / hidden, c.at_rel % hidden, c.over, beyond, c.at_beyond / hidden, c.at_beyond % hidden,
        bar_rms_diff,                bar_ratio,                   if (ok) "PASS" else "FAIL", if (ok_rounded) "PASS" else "FAIL",
    });
    if (n >= exact_min_rows) try exactSample(st, a, g, h, layer, x, w32, y, ref, lane32, n, k, c, base);
}

/// One projection of a slot bank on the host: its code (u16 halfwords [rows, in/16, out/16, n]), rin (suh) and rout (svh)
/// f16 bits, read in place from the stream's unified-memory arrays.
const HostProj = struct {
    code: []const u16,
    rin: []const u16,
    rout: []const u16,
    in: usize,
    out: usize,
    rate: t3.format.Rate,
    stride: usize,

    fn of(p: anytype) !HostProj {
        const sh = mlx.getShape(p.code);
        if (sh.len != 4) return error.RoutedOneExactShape;
        const in: usize = @intCast(sh[1] * 16);
        const out: usize = @intCast(sh[2] * 16);
        const rate = t3.format.Rate.fromN(@intCast(sh[3])) orelse return error.RoutedOneExactRate;
        return .{ .code = try hostU16(p.code), .rin = try hostU16(p.rin), .rout = try hostU16(p.rout), .in = in, .out = out, .rate = rate, .stride = (in / 16) * (out / 16) * rate.halfwords() };
    }

    fn slot(p: HostProj, s: usize) struct { code: []const u16, suh: []const u16, svh: []const u16 } {
        return .{ .code = p.code[s * p.stride ..][0..p.stride], .suh = p.rin[s * p.in ..][0..p.in], .svh = p.rout[s * p.out ..][0..p.out] };
    }
};

fn hostU16(arr: mlx.mlx_array) ![]const u16 {
    const p = mlx.mlx_array_data_uint8(arr) orelse return error.RoutedOneExactNoData;
    const words: [*]const u16 = @ptrCast(@alignCast(p));
    return words[0..mlx.mlx_array_size(arr)];
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `check`'s exact comparison: up to `exact_rows` rows of the call (the largest difference's, the largest relative
/// error's, the largest excess's and the largest reference element's rows, the first two, the middle and the last), each
/// row's routed experts through `one_exact` in f64 from the base bank's own bytes (a row whose experts are not all at
/// the slots the one call read is skipped), against the one path's row, the lane's, and each of the lane's pair rows.
fn exactSample(st: *State, a: std.mem.Allocator, g: anytype, h: anytype, layer: u32, x: mlx.mlx_array, w32: mlx.mlx_array, y: mlx.mlx_array, ref: mlx.mlx_array, lane32: mlx.mlx_array, n: usize, k: usize, c: Gpu, base: anytype) !void {
    const t0 = nowNs();
    const hidden: usize = @intCast(h.hidden);
    const policy = &h.source.stream.layers[layer].policy;
    const cand = [_]usize{ c.at_abs / hidden, c.at_rel / hidden, c.at_beyond / hidden, c.at_ref / hidden, 0, 1, n / 2, n - 1 };
    var sample: [exact_rows]u32 = undefined;
    var nr: usize = 0;
    var skipped: usize = 0;
    next: for (cand) |r| {
        if (nr == exact_rows) break;
        for (sample[0..nr]) |q| if (q == r) continue :next;
        for (st.ids.items[r * k ..][0..k]) |e| {
            const s = st.slot_of.items[e];
            const cur = policy.slotOf(e) orelse plan.no_slot;
            if (s == plan.no_slot or cur != s) {
                skipped += 1;
                continue :next;
            }
        }
        sample[nr] = @intCast(r);
        nr += 1;
    }
    if (nr == 0) {
        log.info("[routed-one] exact layer={d} no sampled row has all its experts at the one call's slots (skipped={d})\n", .{ layer, skipped });
        return;
    }
    const gate = try HostProj.of(base.gate);
    const up = try HostProj.of(base.up);
    const down = try HostProj.of(base.down);
    const inter = gate.out;
    if (gate.in != hidden or up.out != inter or down.in != inter or down.out != hidden) return error.RoutedOneExactShape;
    // The sampled rows as the GPU paths hold them: the bf16 input, the weights, the one path's and the lane's sums, the
    // lane's pair rows.
    const sel = try g.hostArray(std.mem.sliceAsBytes(sample[0..nr]), &.{@intCast(nr)}, .uint32);
    const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
    const buf = try a.alloc(f32, nr * (3 * hidden + k + k * hidden));
    defer a.free(buf);
    const hx = buf[0 .. nr * hidden];
    const hy = buf[nr * hidden ..][0 .. nr * hidden];
    const hr = buf[2 * nr * hidden ..][0 .. nr * hidden];
    const hw = buf[3 * nr * hidden ..][0 .. nr * k];
    const hl = buf[3 * nr * hidden + nr * k ..][0 .. nr * k * hidden];
    _ = try g.hostF32(try g.astype(try g.take(xb, sel, 0), .float32), hx);
    _ = try g.hostF32(try g.take(y, sel, 0), hy);
    _ = try g.hostF32(try g.take(ref, sel, 0), hr);
    _ = try g.hostF32(try g.take(w32, sel, 0), hw);
    _ = try g.hostF32(try g.take(lane32, sel, 0), hl);
    // Each pair's exact row, the pairs of one expert through its decode together.
    const np = nr * k;
    const wide = @max(hidden, inter);
    const f = try a.alloc(f64, nr * hidden + np * hidden + np * (wide + 3 * inter));
    defer a.free(f);
    const x64 = f[0 .. nr * hidden];
    const d64 = f[nr * hidden ..][0 .. np * hidden];
    const scratch = f[nr * hidden + np * hidden ..];
    for (hx, x64) |v, *o| o.* = v;
    const ex = Exact.init();
    const limit: f64 = @floatFromInt(st.limit);
    var done: [exact_rows * 16]bool = @splat(false);
    var experts: usize = 0;
    for (0..np) |p0| if (!done[p0]) {
        const e = st.ids.items[@as(usize, sample[p0 / k]) * k + p0 % k];
        var users: [exact_rows * 16]usize = undefined;
        var nu: usize = 0;
        for (p0..np) |p| if (!done[p] and st.ids.items[@as(usize, sample[p / k]) * k + p % k] == e) {
            users[nu] = p;
            nu += 1;
            done[p] = true;
        };
        experts += 1;
        var xs: [exact_rows * 16][]const f64 = undefined;
        var ts: [exact_rows * 16][]f64 = undefined;
        var gs: [exact_rows * 16][]f64 = undefined;
        var us: [exact_rows * 16][]f64 = undefined;
        var td: [exact_rows * 16][]f64 = undefined;
        var ds: [exact_rows * 16][]f64 = undefined;
        for (users[0..nu], 0..) |p, u| {
            const sc = scratch[u * (wide + 3 * inter) ..][0 .. wide + 3 * inter];
            xs[u] = x64[(p / k) * hidden ..][0..hidden];
            ts[u] = sc[0..wide];
            gs[u] = sc[wide..][0..inter];
            us[u] = sc[wide + inter ..][0..inter];
            td[u] = sc[wide + 2 * inter ..][0..inter];
            ds[u] = d64[p * hidden ..][0..hidden];
        }
        const s = st.slot_of.items[e];
        const sg = gate.slot(s);
        const su = up.slot(s);
        const sd = down.slot(s);
        ex.project(sg.code, sg.suh, sg.svh, hidden, inter, gate.rate, xs[0..nu], ts[0..nu], gs[0..nu]);
        ex.project(su.code, su.suh, su.svh, hidden, inter, up.rate, xs[0..nu], ts[0..nu], us[0..nu]);
        var hs: [exact_rows * 16][]const f64 = undefined;
        for (gs[0..nu], us[0..nu], 0..) |gv, uv, u| {
            for (gv, uv) |*o, uu| o.* = exact.swigluClamped(o.*, uu, limit);
            hs[u] = gv;
        }
        ex.project(sd.code, sd.suh, sd.svh, inter, hidden, down.rate, hs[0..nu], td[0..nu], ds[0..nu]);
    };
    // Per sampled row: the exact sum and its budget sum_j |w_j| |pair_j|, each path's error against it.
    const fl = floorOf(c.rms_ref);
    var worst: Worst = .{};
    for (0..nr) |ri| {
        var e1: Err = .{};
        var e2: Err = .{};
        var pair_rel: f64 = 0;
        var max_y: f64 = 0;
        var ss_y: f64 = 0;
        for (0..k) |j| {
            const dj = d64[(ri * k + j) * hidden ..][0..hidden];
            var ssj: f64 = 0;
            for (dj) |v| ssj += v * v;
            const rms_j = floorOf(@sqrt(ssj / @as(f64, @floatFromInt(hidden))));
            for (dj, hl[(ri * k + j) * hidden ..][0..hidden]) |v, l| pair_rel = @max(pair_rel, @abs(@as(f64, l) - v) / @max(@abs(v), rms_j));
        }
        for (0..hidden) |col| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..k) |j| {
                const w: f64 = hw[ri * k + j];
                const v = d64[(ri * k + j) * hidden + col];
                sum += w * v;
                mag += @abs(w) * @abs(v);
            }
            max_y = @max(max_y, @abs(sum));
            ss_y += sum * sum;
            e1.add(hy[ri * hidden + col], sum, mag, fl);
            e2.add(hr[ri * hidden + col], sum, mag, fl);
            if (sample[ri] == c.at_abs / hidden and col == c.at_abs % hidden) worst = .{ .ri = ri, .col = col, .exact = sum, .mag = mag };
        }
        const hf: f64 = @floatFromInt(hidden);
        log.info("[routed-one] exact layer={d} row={d} max_exact={e:.4} rms_row={e:.3} one: max_abs={e:.3} rel={e:.3} vs_mag={e:.3} rms={e:.3} lane: max_abs={e:.3} rel={e:.3} vs_mag={e:.3} rms={e:.3} lane_pairs_rel={e:.3}\n", .{
            layer, sample[ri], max_y, @sqrt(ss_y / hf), e1.max_abs, e1.rel, e1.vs_mag, @sqrt(e1.ss / hf) / fl, e2.max_abs, e2.rel, e2.vs_mag, @sqrt(e2.ss / hf) / fl, pair_rel,
        });
        st.stats.exact_one_rel = @max(st.stats.exact_one_rel, e1.rel);
        st.stats.exact_lane_rel = @max(st.stats.exact_lane_rel, e2.rel);
        st.stats.exact_one_mag = @max(st.stats.exact_one_mag, e1.vs_mag);
        st.stats.exact_lane_mag = @max(st.stats.exact_lane_mag, e2.vs_mag);
        st.stats.exact_lane_pairs = @max(st.stats.exact_lane_pairs, pair_rel);
    }
    st.stats.exact_calls += 1;
    if (worst.ri) |ri| {
        const row = sample[ri];
        var pb: [1024]u8 = undefined;
        var pw = std.Io.Writer.fixed(&pb);
        for (0..k) |j| {
            const e = st.ids.items[@as(usize, row) * k + j];
            pw.print(" e={d}/slot={d}/w={d:.4}/exact={e:.5}/lane={e:.5}", .{ e, st.slot_of.items[e], hw[ri * k + j], d64[(ri * k + j) * hidden + worst.col], hl[(ri * k + j) * hidden + worst.col] }) catch {};
        }
        const one_v: f64 = hy[ri * hidden + worst.col];
        const lane_v: f64 = hr[ri * hidden + worst.col];
        log.info("[routed-one] exact worst layer={d} row={d} col={d} exact={e:.5} one={e:.5} lane={e:.5} one-exact={e:.3} lane-exact={e:.3} mag={e:.4} pairs:{s}\n", .{ layer, row, worst.col, worst.exact, one_v, lane_v, one_v - worst.exact, lane_v - worst.exact, worst.mag, pw.buffered() });
    }
    log.info("[routed-one] exact layer={d} rows={d} skipped={d} experts={d} ms={d}\n", .{ layer, nr, skipped, experts, (nowNs() - t0) / std.time.ns_per_ms });
}

/// A path's error against the exact sum over a row: the largest, relative to the element (floored at the call's rms) and
/// to the element's budget sum_j |w_j| |pair_j| (floored likewise), and the sum of squares.
const Err = struct {
    max_abs: f64 = 0,
    rel: f64 = 0,
    vs_mag: f64 = 0,
    ss: f64 = 0,

    fn add(e: *Err, got: f32, want: f64, mag: f64, fl: f64) void {
        const d = @abs(@as(f64, got) - want);
        e.max_abs = @max(e.max_abs, d);
        e.rel = @max(e.rel, d / @max(@abs(want), fl));
        e.vs_mag = @max(e.vs_mag, d / @max(mag, fl));
        e.ss += d * d;
    }
};

/// The largest GPU difference's element, when its row is sampled.
const Worst = struct { ri: ?usize = null, col: usize = 0, exact: f64 = 0, mag: f64 = 0 };

/// `[routed-one]`: the path's calls since the last request's end (nothing when it is off).
pub fn meters(st: *State, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const s = st.stats;
    st.stats = .{};
    var w = std.Io.Writer.fixed(buf);
    w.print("[routed-one] mode={s} wide_calls={d} one={d} mixed={d} lane={d} pairs={d} masked={d} checks={d} check_fail={d} rounded_fail={d} worst_ratio={e:.3} worst_rms_diff={e:.3} worst_rel={e:.3} worst_beyond_bf16={e:.3} exact_calls={d} exact_one_rel={e:.3} exact_lane_rel={e:.3} exact_one_vs_mag={e:.3} exact_lane_vs_mag={e:.3} exact_lane_pairs_rel={e:.3}\n", .{
        @tagName(mode), s.wide,           s.one,              s.mixed,         s.lane,         s.pairs,          s.masked,          s.checks,         s.check_fail,     s.rounded_fail,
        s.worst_ratio,  s.worst_rms_diff, s.worst_rel,        s.worst_beyond,  s.exact_calls,  s.exact_one_rel,  s.exact_lane_rel,  s.exact_one_mag,  s.exact_lane_mag, s.exact_lane_pairs,
    }) catch {};
    return w.buffered();
}
