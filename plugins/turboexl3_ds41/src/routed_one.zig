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
    worst_ratio: f64 = 0,
    worst_rms_diff: f64 = 0,
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
    const ref = try laneSum(g, h, layer, x, ids, wts);
    try check(st, g, layer, n, k, sp, y, ref);
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

/// The wide lane's routed sum of the call, as `Source.rows` makes it: the lane's rows [n, k, hidden], weighted, summed
/// over k in f32.
fn laneSum(g: anytype, h: anytype, layer: u32, x: mlx.mlx_array, ids: mlx.mlx_array, wts: mlx.mlx_array) !mlx.mlx_array {
    const ro = try h.at(layer).routed(g, x, ids);
    const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
    return g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false);
}

/// `check`: the one path's sum `y` against the lane's `ref`, logged.
fn check(st: *State, g: anytype, layer: u32, n: usize, k: usize, sp: plan.Split, y: mlx.mlx_array, ref: mlx.mlx_array) !void {
    const total: c_int = @intCast(mlx.mlx_array_size(ref));
    const d = try g.reshape(try g.sub(y, ref), &.{total});
    const r = try g.reshape(ref, &.{total});
    const v = try g.stack(&.{ try g.max(try g.abs(d), 0, false), try g.mean(try g.square(d), 0, false), try g.mean(try g.square(r), 0, false) }, 0);
    var out: [3]f32 = undefined;
    _ = try g.hostF32(v, &out);
    const rms_ref = @sqrt(@as(f64, out[2]));
    const ratio = if (rms_ref > 0) @as(f64, out[0]) / rms_ref else @as(f64, out[0]);
    const rms_diff = if (rms_ref > 0) @sqrt(@as(f64, out[1])) / rms_ref else @sqrt(@as(f64, out[1]));
    const ok = rms_diff <= bar_rms_diff and ratio <= bar_ratio;
    st.stats.checks += 1;
    if (!ok) st.stats.check_fail += 1;
    // NaN compares false: a NaN check is a failure, and worst stays the last finite one.
    if (ratio > st.stats.worst_ratio) st.stats.worst_ratio = ratio;
    if (rms_diff > st.stats.worst_rms_diff) st.stats.worst_rms_diff = rms_diff;
    log.info("[routed-one] check layer={d} rows={d} pairs={d} resident_pairs={d} experts={d}/{d} max_abs={e:.3} rms_ref={e:.3} ratio={e:.3} rms_diff={e:.3} bar={e:.1}/{e:.1} {s}\n", .{ layer, n, n * k, sp.resident_pairs, sp.n_resident, sp.distinct.len, out[0], rms_ref, ratio, rms_diff, bar_rms_diff, bar_ratio, if (ok) "PASS" else "FAIL" });
}

/// `[routed-one]`: the path's calls since the last request's end (nothing when it is off).
pub fn meters(st: *State, buf: []u8) []const u8 {
    if (mode == .off) return buf[0..0];
    const s = st.stats;
    st.stats = .{};
    var w = std.Io.Writer.fixed(buf);
    w.print("[routed-one] mode={s} wide_calls={d} one={d} mixed={d} lane={d} pairs={d} masked={d} checks={d} check_fail={d} worst_ratio={e:.3} worst_rms_diff={e:.3}\n", .{
        @tagName(mode), s.wide, s.one, s.mixed, s.lane, s.pairs, s.masked, s.checks, s.check_fail, s.worst_ratio, s.worst_rms_diff,
    }) catch {};
    return w.buffered();
}
