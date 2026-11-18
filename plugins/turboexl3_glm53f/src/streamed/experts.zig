//! The routed experts of the native DeepSeek-V4.1 model over the expert
//! streamer (track M, M3). Oracle: the EXL3 decode lane of the Python tier,
//! `plane_lane.PackedDecode.run` driving `exl3_lane.Exl3PackedOps`.
//!
//! Per routed-layer call the trunk hands the hook `xf [n, hidden]` and the
//! router's `indices [n, k]`. The hook reads the ids on the host (the routing
//! barrier), asks an expert source for their slots (`route`), runs the
//! residents' wave at once, then each miss part's gate/up once its gate/up
//! bytes landed (`waitGu`) and its down once the rest landed (`waitDown`),
//! and hands the call back (`release`: the rows stay pinned until the next
//! route's flush, after the eval that consumed them). The outputs come back
//! in the router's order as `[n, k, hidden]` f32; the trunk combines them.
//! `grow` is the one phase change; `flush` follows the forward's last eval.
//! A call of more rows than a route takes (a prefill chunk) is the wide lane:
//! the math's `prefill` over the slots its experts are served in.
//!
//! The source is the streamer's `Stream` bridge (`assertSource` checks its
//! contract). The math of a group of rows sharing a bank is the C2 `quant`
//! seam (quant.zig).

const std = @import("std");
const mlx = @import("sdk").mlx;
const ops = @import("ops.zig");
const expert_io = sdk_ext.expert.io;
const expert_policy = @import("sdk_ext.zig").expert.policy;
const expert_stream = @import("expert/stream.zig");
const expert_lookahead = sdk_ext.expert.lookahead;
const expert_event = sdk_ext.expert.event;
const sdk_ext = @import("sdk_ext.zig");
const quant = @import("sdk_ext.zig").quant;
const xq = @import("../exl3/quant.zig");

pub const max_route_ids = expert_policy.max_route_ids;
/// The widest decode / verify forward (the RC routes' and the verify block's rows): at most
/// `decode_forward_rows * top_k` routed ids, which construction proves fit one route (`max_route_ids`),
/// so a forward of <= 8 rows never reaches the wide lane.
pub const decode_forward_rows: u32 = 8;
/// The fewest routed ids a wide-lane call takes (one more than a route): refused below, by name.
pub const wide_min_ids: u32 = max_route_ids + 1;
pub const BankKind = expert_stream.BankKind;
pub const SlotRef = expert_stream.SlotRef;
pub const Stats = expert_stream.Stats;
pub const Error = expert_stream.Error;
const Load = expert_policy.Load;
const n_banks = std.meta.fieldNames(BankKind).len;

/// What one routed-layer call serves (`sdk_ext.expert.Served`).
pub const Served = sdk_ext.expert.Served;

/// One projection's slot arrays in a bank (the streamer's `ProjArrays` over
/// any backend): code int16 [rows, in/16, out/16, 16K], rout f16 [rows, out],
/// rin f16 [rows, in].
pub const ProjOf = xq.ProjArrays;

/// A bank's nine arrays by projection (the streamer's `BankArrays`; the quant seam's).
pub fn BankArraysOf(comptime T: type) type {
    return quant.BankArrays(ProjOf(T));
}

// ── The source contract (`sdk_ext.expert.assertSource`: the required set and each declared capability's) ──

pub const assertSource = sdk_ext.expert.assertSource;

// ── The executor: the model's routed hook over a source ──

/// The most sources JOINLESS's combine reads (`dsv41_kernel_routes.JoinlessCombine.sources`).
pub const joinless_sources = 24;

/// `planJoinless` with the combine's own source bound: up to `bound` outputs each read in place (`bound` >= the
/// outputs: no merge); above it the stock rule at `joinless_sources`. A combine that reads every source in place (MiMo
/// P7 `prefill_combine: "inplace"`) passes its bound; every other caller keeps `planJoinless`.
pub fn planJoinlessBound(rows: []const u32, order: []u32, src: []u8, off: []u32, bound: usize) usize {
    const n = rows.len;
    @memset(off, 0);
    if (n <= @max(bound, joinless_sources)) {
        for (src, 0..) |*s, i| s.* = @intCast(i);
        return n;
    }
    for (order, 0..) |*x, i| x.* = @intCast(i);
    std.sort.pdq(u32, order, rows, struct {
        fn lt(r: []const u32, x: u32, y: u32) bool {
            return if (r[x] != r[y]) r[x] < r[y] else x < y;
        }
    }.lt);
    const merged: u8 = joinless_sources - 1;
    @memset(src, 0);
    for (order[0 .. n - merged]) |i| src[i] = merged;
    var in_place: u8 = 0;
    var at: u32 = 0;
    for (rows, src, off) |r, *s, *o| {
        if (s.* == merged) {
            o.* = at;
            at += r;
        } else {
            s.* = in_place;
            in_place += 1;
        }
    }
    return joinless_sources;
}

/// Each assignment's (source, row) under a `planJoinless` plan, as int32 pairs: output i's join-ordered
/// rows (`pos`, the outputs' rows back to back) are rows `off[i]..` of source `src[i]`.
pub fn fillJoinlessLoc(rows: []const u32, src: []const u8, off: []const u32, pos: []const u32, loc: []i32) void {
    var j: usize = 0;
    for (rows, src, off) |r, s, o| {
        for (pos[j..][0..r], o..) |p, row| {
            loc[2 * p] = s;
            loc[2 * p + 1] = @intCast(row);
        }
        j += r;
    }
}

/// The merge's host scratch, reused across calls.
pub fn JoinlessScratch(comptime T: type) type {
    return struct {
        rows: std.ArrayList(u32) = .empty,
        order: std.ArrayList(u32) = .empty,
        src: std.ArrayList(u8) = .empty,
        off: std.ArrayList(u32) = .empty,
        merge_in: std.ArrayList(T) = .empty,

        pub fn deinit(s: *@This(), a: std.mem.Allocator) void {
            inline for (.{ &s.rows, &s.order, &s.src, &s.off, &s.merge_in }) |l| l.deinit(a);
        }
    };
}

/// (profile builds) A wide call's JOINLESS merge for the prefill profile's group line: its outputs, the combine's sources and
/// the rows the one concatenate copied (0: no merge, every output read in place).
pub const MergeStats = struct { outputs: u32 = 0, sources: u32 = 0, copied_rows: u64 = 0 };

/// `mergeJoinless` under the combine's source bound (`planJoinlessBound`).
pub fn mergeJoinlessBound(comptime G: type, g: *G, a: std.mem.Allocator, outs: *std.ArrayList(G.T), pos: []const u32, loc: []i32, sc: *JoinlessScratch(G.T), bound: usize) !u64 {
    const n_out = outs.items.len;
    try sc.rows.resize(a, n_out);
    try sc.order.resize(a, n_out);
    try sc.src.resize(a, n_out);
    try sc.off.resize(a, n_out);
    for (outs.items, sc.rows.items) |x, *r| r.* = @intCast(g.shapeOf(x).dim(0));
    const n_src = planJoinlessBound(sc.rows.items, sc.order.items, sc.src.items, sc.off.items, bound);
    fillJoinlessLoc(sc.rows.items, sc.src.items, sc.off.items, pos, loc);
    if (n_src == n_out) return 0;
    const last: u8 = @intCast(n_src - 1);
    sc.merge_in.clearRetainingCapacity();
    var copied: u64 = 0;
    for (outs.items, sc.src.items, sc.rows.items) |x, s, r| if (s == last) {
        try sc.merge_in.append(a, x);
        copied += r;
    };
    const cat = g.keep(try g.concat(sc.merge_in.items, 0));
    for (sc.merge_in.items) |x| g.release(x);
    var d: usize = 0;
    for (outs.items, sc.src.items) |x, s| if (s != last) {
        outs.items[d] = x;
        d += 1;
    };
    outs.shrinkRetainingCapacity(d);
    outs.appendAssumeCapacity(cat);
    return copied;
}

/// Construction-time routes of the executor.
pub const Routes = struct {
    /// Pass `route` the next routed layer's gate scores (the streamer's
    /// lookahead predictor, evaluated with the routing barrier).
    lookahead: bool = false,
    /// Event gates instead of host waits: every wave is built at once over
    /// event-wait aliases of the bank arrays (the typical tier's gate).
    gated: bool = false,
    /// The wide lane (calls of more than `max_route_ids` routed rows) through
    /// the math's `prefill(g, layer, x, rows, bank)` (a KEPT result) and
    /// `finishPrefill(g)`; off refuses a wide call (`PrefillLaneNotPorted`).
    prefill: bool = false,
};

/// The wide lane's read schedule, chosen at construction (`Options.wide`; each
/// exact: the same rows, slots and kernels, only the order and timing of reads
/// and evals change). Every call seeds its layer's residency with its own ids (the most routed experts protected in
/// the persistent rows), feeds its groups hottest first (routed rows descending, ties by id) and drains each group once
/// (its banks' waves queued). Its rows are bucketed by group once (a stable counting sort into the join's scratch,
/// `bucketRowsByGroup`), each group's bank passes over its own rows only, in the same order.
pub const Wide = struct {
    /// Groups in flight: 2 routes (reads) the next group before this
    /// group's waves (the source's `wideDepth` must allow it).
    depth: u8 = 1,
    /// P1: each layer's predicted seed read into its persistent rows (the empty ones, then those of the
    /// residents the prediction leaves out) while its attention runs (the model's predictor pass hands it
    /// over, `Hook.readAheadSeed`, in every prompt pass), landed at the routing barrier before the seed.
    /// Exact: it chooses reads and rows only.
    read_ahead: bool = false,
    /// turboexl3_glm53f (TURBOEXL3_GLM53F_PF_STAGE): a prompt call of at least 8 rows per expert (nearly every expert
    /// routed) routes its layer's misses before its routing barrier: as many as `depth` groups hold, in expert order,
    /// into the transient windows with every resident held (no staged miss evicts one: `holdResidents`), so their reads
    /// run while the GPU runs the layer's attention, which the barrier waits for. The call's groups are those staged
    /// groups first (their experts the call does not route stay in them, with no rows), then its other experts hottest
    /// first, every group on the stock route (no P7 plan: a staged route is live before the call's order is known). Exact:
    /// the same kernels over the same rows; only which rows hold a record, and when it is read, change.
    stage: bool = false,
    /// turboexl3_glm53f (TURBOEXL3_GLM53F_PF_DRAIN_LAG): a group's one drain, its slots' release and the
    /// route `depth` groups on wait for the next group's first committed chunk (the math's `LagPending`, whose hook the
    /// math's prompt call runs at that commit), or for the next group's banks when it commits none, in place of running
    /// before the next group is encoded: the GPU runs the group's last chunks while the host builds the next group's
    /// first. Exact: the same kernels over the same rows in the same order on the stream; a group's slots go back only
    /// after its own drain, and in group order. Needs `depth` 2 or more (the next group's route was made before) and a
    /// math with `LagPending`.
    drain_lag: bool = false,
};

/// A wide call's rows bucketed by group, each group's rows in ascending row order (a stable counting sort): `ids` the
/// call's routed ids (rows x top-k), `distinct` its experts, `grp_of[e]` expert e's group, `count[e]` (optional) its rows;
/// `off` [n_groups + 1] gets the groups' offsets, `rows` [ids.len] the rows. A pass over `rows[off[g]..off[g + 1]]` meets
/// exactly the rows of group g a pass over the whole call meets, in the same order (exact).
pub fn bucketRowsByGroup(ids: []const u16, distinct: []const u16, grp_of: []const u16, count: ?[]const u32, off: []u32, rows: []u32) void {
    @memset(off, 0);
    if (count) |c| {
        for (distinct) |e| off[grp_of[e] + 1] += c[e];
    } else for (ids) |e| {
        off[grp_of[e] + 1] += 1;
    }
    for (1..off.len) |gi| off[gi] += off[gi - 1];
    // Each row at its group's cursor (`off[gi]` advances to the group's end), then the starts restored.
    for (ids, 0..) |e, row| {
        const gi = grp_of[e];
        rows[off[gi]] = @intCast(row);
        off[gi] += 1;
    }
    var gi = off.len - 1;
    while (gi > 0) : (gi -= 1) off[gi] = off[gi - 1];
    off[0] = 0;
}

/// The routed-expert hook of `Model(G)` over source `S` with math `M`
/// (`gateUp(g, x, ids, gate, up)`, `down(g, h, ids, d)`). Bank arrays are
/// bound once at `init` (base, transient) and at `grow` (the grown rows).
/// `at(layer)` is the per-layer hook the trunk calls.
/// The MLX event wait (the gated arm's): aliases of `xs` into `outs`, which may hold anything (a caller's
/// `undefined`): the shim assigns each alias into its handle, so every handle is made fresh first.
fn mlxEventWait(g: *ops.MlxOps, event: expert_event.Event, xs: []const mlx.mlx_array, value: u64, deps: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    for (outs) |*o| o.* = mlx.mlx_array_new();
    try expert_event.wait(xs, event, value, deps, false, g.s, outs);
    for (outs) |*o| o.* = try g.adopt(o.*);
}

/// A math with optional projection arrays supplies its proven event projection at construction.
/// DeepSeek's default projection always aliases its three EXL3 arrays in their existing order.
fn Exl3EventProjection(comptime T: type) type {
    return struct {
        pub const count: usize = 3;
        pub fn inputs(p: ProjOf(T)) [count]T {
            return .{ p.code, p.rout, p.rin };
        }
        pub fn fromWaited(out: [count]T) ProjOf(T) {
            return .{ .code = out[0], .rout = out[1], .rin = out[2] };
        }
    };
}

pub fn ExpertsWith(comptime G: type, comptime S: type, comptime M: type, comptime routes: Routes) type {
    const Projection = if (@hasDecl(M, "Projection")) M.Projection else ProjOf;
    const EventProjection = if (@hasDecl(M, "EventProjection")) M.EventProjection(G.T) else Exl3EventProjection(G.T);
    const wide_event_gates = routes.gated and (if (@hasDecl(M, "wide_event_gates")) M.wide_event_gates else false);
    const predictor_decode_only = if (@hasDecl(M, "predictor_decode_only")) M.predictor_decode_only else false;
    const wide_profile = M.Profile;
    const bound_decode_rows = if (@hasDecl(M, "decode_forward_rows")) M.decode_forward_rows else decode_forward_rows;
    comptime {
        assertSource(S);
        if (routes.gated and !sdk_ext.expert.capsOf(S).event_gates) @compileError(@typeName(S) ++ ": a gated route needs a source with event gates");
        // A source's slot arrays are its quant's: a source and a quant that disagree on the format do not compile.
        if (@hasDecl(S, "Arrays") and S.Arrays(G.T) != Projection(G.T)) @compileError(@typeName(S) ++ ": its slot arrays are not the quant's");
    }
    return struct {
        const Self = @This();
        const T = G.T;
        pub const Arrays = quant.BankArrays(Projection(T));
        /// The math this hook runs (its construction-time capabilities: `has_parts`).
        pub const Math = M;

        /// A routed layer's gate (the lookahead predictor reads the next layer's).
        pub const Gate = struct { w: T, bias: T };

        a: std.mem.Allocator,
        source: *S,
        math: M,
        hidden: c_int,
        n_experts: u32,
        /// Per layer, per bank kind: what the math binds (null: no rows).
        banks: [][n_banks]?Arrays,
        /// Lookahead: every routed layer's gate, by layer.
        gates: []const Gate = &.{},
        /// Gated: the event the stream signals (an MLX backend's MTLSharedEvent).
        event: expert_event.Event = .{ .id = 0, .object = 0 },
        /// The backend the banks were bound on.
        g: *G,
        wide: WideScratch = .{},
        /// The wide lane's read schedule (`Options.wide`).
        wide_route: Wide = .{},
        /// The parts' combine's source bound (`runWideParts`: outputs above it merge into one source); `joinless_sources`
        /// unless the arch installs a combine that reads every source in place (`setJoinlessBound`).
        joinless_bound: usize = joinless_sources,
        /// `releaseTransient` nulled the transient bindings; `grow` binds decode's window 0.
        transient_released: bool = false,
        /// (profile builds: `wide_profile.enabled`) The last wide call's JOINLESS merge (`MergeStats`); compiled out elsewhere.
        last_merge: if (wide_profile.enabled) MergeStats else void = if (wide_profile.enabled) .{} else {},
        /// DEVROUTE (`Options.devroute`): per layer two resident LUTs (u32 [n_experts], host arrays written in place, kept),
        /// the one each layer's next decode call reads (`dev_parity`): a call reads one, its route's result is written
        /// into the other, so the host never writes a LUT a committed wave may still read. Empty without the route.
        devroute: bool = false,
        dev_lut: [][2]T = &.{},
        dev_parity: []u1 = &.{},
        /// Every wave runs as one group on the math's banked route (its `has_banked`; the quant installed it).
        /// `devroute`: DEVROUTE (exact): in decode, each call's hit wave is a device graph (the routed ids through the
        /// layer's resident LUT into the banked texts, every routed pair) committed behind the barrier's arrays, before the
        /// host's wait and route; the host then routes and builds only the miss parts; the join takes every hit pair from
        /// the device wave and every miss pair from its part.
        pub const Options = struct { gates: []const Gate = &.{}, event: ?expert_event.Event = null, wide: Wide = .{}, devroute: bool = false };

        /// The wide lane's host scratch, reused across calls.
        const WideScratch = struct {
            ids: std.ArrayList(u16) = .empty,
            /// Per expert: its index in `distinct`, or -1.
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            /// Per expert: its rows in the call (the feed's order).
            count: std.ArrayList(u32) = .empty,
            /// The unjoined route: a call's rows' routed positions, and its wave-ordered rows' call rows.
            call_pos: std.ArrayList(u32) = .empty,
            wave_pos: std.ArrayList(u32) = .empty,
            slot: std.ArrayList(u32) = .empty,
            act_row: std.ArrayList(u32) = .empty,
            pos: std.ArrayList(u32) = .empty,
            inv: std.ArrayList(u32) = .empty,
            kept: std.ArrayList(T) = .empty,
            /// JOINLESS: each assignment's (source, row), int32 pairs.
            loc: std.ArrayList(i32) = .empty,
            /// JOINLESS: the merge's plan (`mergeJoinless`).
            jl: JoinlessScratch(T) = .{},
            /// `Wide.route_cut`: the call's rows by group (`groupRows`; the rows themselves in `inv`, free until the join):
            /// per expert its group, the groups' row offsets.
            grp_of: std.ArrayList(u16) = .empty,
            grp_off: std.ArrayList(u32) = .empty,
            /// `Wide.stage`: the misses staged before the barrier, in their groups' order.
            staged: std.ArrayList(u16) = .empty,

            fn deinit(w: *WideScratch, a: std.mem.Allocator) void {
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.call_pos, &w.wave_pos, &w.slot, &w.act_row, &w.pos, &w.inv, &w.kept, &w.loc, &w.grp_of, &w.grp_off, &w.staged }) |l| l.deinit(a);
                w.jl.deinit(a);
            }
        };

        pub fn init(a: std.mem.Allocator, g: *G, source: *S, math: M, c: anytype) !Self {
            return initWith(a, g, source, math, c, .{});
        }

        pub fn initWith(a: std.mem.Allocator, g: *G, source: *S, math: M, c: anytype, opt: Options) !Self {
            // A decode-width forward must fit one route: its calls are the decode lane's by arithmetic.
            if (@as(u64, bound_decode_rows) * c.n_experts_per_tok > max_route_ids) return error.DecodeRowsWiderThanRoute;
            if (routes.lookahead and opt.gates.len != c.n_layers) return error.LookaheadNeedsGates;
            if (routes.gated and G == ops.MlxOps and opt.event == null) return error.GatedNeedsEvent;
            const wr = opt.wide;
            if (wr.depth < 1 or wr.depth > expert_stream.max_wide_depth) return error.InvalidWideRoute;
            if (comptime routes.prefill and !@hasDecl(S, "seedPrefill")) @compileError(@typeName(S) ++ ": the wide lane seeds its residency");
            if (wr.depth > 1 and !routes.prefill) return error.InvalidWideRoute;
            if (wr.read_ahead and comptime !@hasDecl(S, "readAheadSeed")) return error.InvalidWideRoute;
            if (wr.stage and comptime !(@hasDecl(S, "stageMisses") and @hasDecl(S, "holdResidents"))) return error.InvalidWideRoute;
            if (wr.drain_lag and (wr.depth < 2 or comptime !@hasDecl(M, "LagPending"))) return error.InvalidWideRoute;
            if (comptime !hasBanked()) @compileError(@typeName(M) ++ ": the decode lane runs the math's banked route");
            if (wr.depth > 1) {
                if (comptime @hasDecl(S, "wideDepth")) {
                    if (source.wideDepth() < wr.depth) return error.WideDepthExceedsSource;
                } else return error.WideDepthExceedsSource;
            }
            const banks = try a.alloc([n_banks]?Arrays, c.n_layers);
            errdefer a.free(banks);
            for (banks, 0..) |*b, l| {
                b.* = @splat(null);
                for ([_]BankKind{ .base, .transient }) |kind| b[@backingInt(kind)] = try bind(g, source, @intCast(l), kind);
            }
            var self: Self = .{ .a = a, .source = source, .math = math, .hidden = @intCast(c.hidden_size), .n_experts = c.n_routed_experts, .banks = banks, .gates = opt.gates, .g = g, .wide_route = wr };
            if (opt.event) |e| self.event = e;
            if (opt.devroute) {
                if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) return error.DevrouteNotInSource;
                self.devroute = true;
            }
            return self;
        }

        /// DEVROUTE: both LUTs of every layer written from the grown policies (the decode's first calls read them).
        fn devLutsAtGrow(self: *Self, g: *G) !void {
            if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) {
                return error.DevrouteNotInSource;
            } else return self.devLutsAtGrowIn(g);
        }

        fn devLutsAtGrowIn(self: *Self, g: *G) !void {
            if (self.dev_lut.len == 0) {
                const n_l = self.banks.len;
                self.dev_lut = try self.a.alloc([2]T, n_l);
                self.dev_parity = try self.a.alloc(u1, n_l);
                const zeros = try self.a.alloc(u32, self.n_experts);
                defer self.a.free(zeros);
                @memset(zeros, 0);
                for (self.dev_lut) |*pair| for (pair) |*x| {
                    x.* = g.keep(try g.hostArray(std.mem.sliceAsBytes(zeros), &.{@intCast(self.n_experts)}, .uint32));
                };
            }
            for (self.dev_lut, self.dev_parity, 0..) |pair, *par, l| {
                for (pair) |x| self.source.residentLut(@intCast(l), std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(x)))));
                par.* = 0;
            }
        }

        /// DEVROUTE: layer `layer`'s LUT for its next call, from the policy after this call's route; that call reads it.
        fn devLutAfterRoute(self: *Self, g: *G, layer: u32) !void {
            if (comptime !(@hasDecl(S, "residentLut") and @hasDecl(S, "decoding"))) {
                return error.DevrouteNotInSource;
            } else return self.devLutAfterRouteIn(g, layer);
        }

        fn devLutAfterRouteIn(self: *Self, g: *G, layer: u32) !void {
            const next = self.dev_parity[layer] +% 1;
            self.source.residentLut(layer, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(self.dev_lut[layer][next])))));
            self.dev_parity[layer] = next;
        }

        /// DEVROUTE: the call's every routed pair through the banked texts, the slots from the layer's LUT on the device
        /// (hits: their resident (bank, row), the host's plan's same; misses: base row 0, never joined); `[n * k, hidden]`
        /// in routed order.
        fn devWave(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            const nk: c_int = @intCast(n * k);
            const lut = self.dev_lut[layer][self.dev_parity[layer]];
            const ids = try g.take(lut, try g.reshape(indices, &.{nk}), 0);
            var tok: [max_route_ids]i32 = undefined;
            for (tok[0 .. n * k], 0..) |*t, p| t.* = @intCast(p / k);
            const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0 .. n * k]), &.{nk}, .int32), 0);
            const banks = try self.bankedArrays(layer, null);
            return self.math.downBanked(g, try self.math.gateUpBanked(g, x, ids, &banks), ids, &banks);
        }

        fn hasBanked() bool {
            return @hasDecl(M, "has_banked") and M.has_banked;
        }

        /// The router that scores layer `layer`'s read-ahead: the NEXT layer's
        /// (none after the last layer).
        pub fn predictorGate(self: *const Self, layer: u32) ?Gate {
            return if (layer + 1 < self.gates.len) self.gates[layer + 1] else null;
        }

        /// X19: a math that binds its predictor at construction (MiMo's `predict_router`) scores through its own
        /// instance; every other math keeps `nextScores`.
        fn scoresOf(self: *const Self, g: *G, xf: T, gate: Gate) !T {
            if (comptime @hasDecl(M, "predictorScoresBound")) return self.math.predictorScoresBound(g, xf, gate.w, gate.bias);
            return nextScores(g, xf, gate);
        }

        /// The architecture's predictor, or DeepSeek's sqrt(softplus(f32(x @ w^T))) + bias.
        fn nextScores(g: *G, xf: T, gate: Gate) !T {
            if (comptime @hasDecl(M, "predictorScores")) return M.predictorScores(g, xf, gate.w, gate.bias);
            const z = try g.astype(try g.matmul(xf, try g.transpose(gate.w)), .float32);
            return g.add(try g.sqrt(try g.logaddexp(z, try g.scalar(0, .float32))), gate.bias);
        }

        /// An event wait's aliases of `xs` (the GPU reads them after `value`), into `outs` (any contents). One wait (one
        /// pass-through primitive, one encoder split) whatever `xs.len`; the trace backend counts the waits (`wait_calls`).
        fn eventWait(self: *Self, g: *G, xs: []const T, value: u64, deps: []const T, outs: []T) !void {
            if (G == ops.MlxOps) {
                try mlxEventWait(g, self.event, xs, value, deps, outs);
            } else {
                if (comptime @hasField(G, "wait_calls")) g.wait_calls += 1;
                for (xs, outs) |x, *o| o.* = try g.eventAlias(x, value, deps.len);
            }
        }

        fn waitProj(self: *Self, g: *G, p: Projection(T), value: u64, deps: []const T) !Projection(T) {
            const xs = EventProjection.inputs(p);
            var outs: [EventProjection.count]T = undefined;
            try self.eventWait(g, &xs, value, deps, &outs);
            return EventProjection.fromWaited(outs);
        }

        pub fn deinit(self: *Self) void {
            for (self.dev_lut) |pair| for (pair) |x| self.g.release(x);
            self.a.free(self.dev_lut);
            self.a.free(self.dev_parity);
            self.wide.deinit(self.a);
            self.a.free(self.banks);
            self.* = undefined;
        }

        /// A bank with rows must have arrays the math can bind (MLX slot memory).
        fn bind(g: *G, source: *S, layer: u32, kind: BankKind) !?Arrays {
            const arrays = try source.bankArrays(g, layer, kind);
            if (arrays == null and source.bankRows(layer, kind) > 0) return error.SlotArraysUnbound;
            return arrays;
        }

        /// The phase change's first free: every layer's transient binding nulled (from here to `grow` a stray use
        /// fails on null, never on freed arrays), then the source frees the scratch. Returns the bytes freed. A
        /// refusal leaves the bindings null (fails closed; the Module refuses every later request).
        pub fn releaseTransient(self: *Self) !u64 {
            for (self.banks) |*b| b[@backingInt(BankKind.transient)] = null;
            self.transient_released = true;
            return self.source.releaseTransient();
        }

        /// The one phase change: the source grows, the grown rows (and after a release, decode's window 0) are bound.
        pub fn grow(self: *Self, g: *G, decode_rows: []const u32) !void {
            try self.source.grow(decode_rows);
            for (self.banks, 0..) |*b, l| {
                b[@backingInt(BankKind.ext)] = try bind(g, self.source, @intCast(l), .ext);
                if (self.transient_released) b[@backingInt(BankKind.transient)] = try bind(g, self.source, @intCast(l), .transient);
            }
            self.transient_released = false;
            if (self.devroute) try self.devLutsAtGrow(g);
        }

        /// DEVROUTE as bound at construction.
        /// Construction: the parts' combine reads up to `bound` sources in place (`runWideParts` merges only above it).
        pub fn setJoinlessBound(self: *Self, bound: usize) void {
            self.joinless_bound = @max(bound, joinless_sources);
        }

        /// The reverse phase change's free: the grown rows' and window 0's bindings nulled (a stray use fails on null),
        /// then the source frees them, the prompt rows' residents forgotten or kept (`residents`; the base bindings stay).
        /// Returns the bytes freed.
        pub fn shrink(self: *Self, prompt_rows: []const u32, residents: expert_stream.Residents) !u64 {
            for (self.banks) |*b| {
                b[@backingInt(BankKind.ext)] = null;
                b[@backingInt(BankKind.transient)] = null;
            }
            self.transient_released = true;
            return self.source.shrink(prompt_rows, residents);
        }

        /// The reverse phase change's allocation: the prompt's scratch re-created and bound. Returns its bytes (0 when
        /// the scratch stayed through decode: the release route off).
        pub fn regrowTransient(self: *Self, g: *G, released: bool) !u64 {
            const bytes: u64 = if (released) try self.source.regrowTransient() else 0;
            for (self.banks, 0..) |*b, l| b[@backingInt(BankKind.transient)] = try bind(g, self.source, @intCast(l), .transient);
            self.transient_released = false;
            return bytes;
        }

        /// After the forward's last eval: settles and unpins released calls.
        pub fn flush(self: *Self) !void {
            try self.source.flush();
        }

        pub const Hook = struct {
            ex: *Self,
            layer: u32,

            /// `routed`, with the caller's arrays that do not wait on this call (`hoist`: the layer's
            /// shared expert and HC tail) committed once its reads are issued and its hit wave
            /// started: they run during the read wait, not in the next routing barrier.
            pub fn routedHoist(h: Hook, g: *G, xf: T, indices: T, hoist: []const T) !T {
                return h.ex.run(g, h.layer, xf, indices, hoist);
            }

            /// turboexl3_glm53f: `routedHoist` ending before the join's take (JOIN_INDEXED, a decode-lane call): the
            /// concatenated outputs and each routed position's row in them (`Indexed`).
            pub fn routedHoistIndexed(h: Hook, g: *G, xf: T, indices: T, hoist: []const T) !Indexed {
                return h.ex.runEnd(g, h.layer, xf, indices, hoist, .indexed);
            }

            /// JOINLESS (a wide call only: more than max_route_ids ids): the unjoined outputs as the
            /// combine's sources and each assignment's (source, row); `releaseParts` after the combines are evaluated.
            pub fn routedParts(h: Hook, g: *G, xf: T, indices: T) !Parts {
                if (comptime !routes.prefill) return error.PrefillLaneNotPorted;
                const n: u32 = @intCast(g.shapeOf(xf).dim(0));
                const k: u32 = @intCast(g.shapeOf(indices).dim(1));
                if (comptime !@hasDecl(M, "prefill_min_tokens")) {
                    if (n * k <= max_route_ids) return error.WideLaneUnderMinIds;
                }
                return h.ex.runWideParts(g, h.layer, xf, indices, n, k);
            }

            pub fn releaseParts(h: Hook, g: *G) void {
                h.ex.releaseParts(g);
            }

            /// P1: the layer's predicted seed, experts hottest first, read ahead of its routed call.
            pub fn readAheadSeed(h: Hook, experts: []const u16) !void {
                if (comptime !@hasDecl(S, "readAheadSeed")) return error.ReadAheadNotStreamed;
                return h.ex.source.readAheadSeed(h.layer, experts);
            }
        };

        pub fn at(self: *Self, layer: u32) Hook {
            return .{ .ex = self, .layer = layer };
        }

        const Group = struct { bank: BankKind, n: u32 = 0, pos: [max_route_ids]u8 = undefined };

        /// Outputs in wave order and the routed position of each output row.
        /// `pos` holds each concatenated output row's routed position; DEVROUTE's device wave adds a row for every pair
        /// before the miss parts' rows (2 x the route's ids).
        const Acc = struct {
            outs: [max_route_ids]T = undefined,
            n_outs: usize = 0,
            pos: [2 * max_route_ids]u32 = undefined,
            n_pos: usize = 0,
        };

        const Wave = struct {
            groups: [n_banks]Group = undefined,
            n: usize = 0,
            ids: [n_banks]T = undefined,
            h: [n_banks]T = undefined,
        };

        /// A wave's bank arrays for the banked texts: each bank's; a bank without arrays (no rows bound, or not waited
        /// in a gated part) is given another bank's (the packed ids never name it).
        fn bankedArrays(self: *Self, layer: u32, over: ?*const [n_banks]?Arrays) ![n_banks]Arrays {
            const src = if (over) |o| o else &self.banks[layer];
            const fill = for (src) |b| {
                if (b) |arr| break arr;
            } else return error.BankedWaveWithoutArrays;
            var out: [n_banks]Arrays = undefined;
            for (src, &out) |b, *o| o.* = b orelse fill;
            return out;
        }

        /// The banked route's gate/up: wave `w`'s positions as one group in routed order, ids packed (bank << 24 | slot row).
        fn gateUpWaveBanked(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave {
            var wave: Wave = .{};
            const gr = &wave.groups[0];
            gr.* = .{ .bank = .base };
            var tok: [max_route_ids]i32 = undefined;
            var rows: [max_route_ids]u32 = undefined;
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                gr.pos[gr.n] = @intCast(pos);
                tok[gr.n] = @intCast(pos / k);
                rows[gr.n] = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
                gr.n += 1;
            }
            if (gr.n == 0) return wave;
            wave.n = 1;
            const banks = try self.bankedArrays(layer, over);
            const n: c_int = @intCast(gr.n);
            const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{n}, .int32), 0);
            wave.ids[0] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{n}, .uint32);
            wave.h[0] = try self.math.gateUpBanked(g, x, wave.ids[0], &banks);
            return wave;
        }

        /// The banked route's down: the wave's one group over every bank.
        fn downWaveBanked(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T {
            const first = acc.n_outs;
            if (wave.n == 0) return acc.outs[first..first];
            const banks = try self.bankedArrays(layer, over);
            acc.outs[acc.n_outs] = try self.math.downBanked(g, wave.h[0], wave.ids[0], &banks);
            acc.n_outs += 1;
            for (wave.groups[0].pos[0..wave.groups[0].n]) |pos| {
                acc.pos[acc.n_pos] = pos;
                acc.n_pos += 1;
            }
            return acc.outs[first..acc.n_outs];
        }

        /// The gated waves (the lookahead4 lane's event gate): every part's gate/up over bank arrays waited at `gu`, then
        /// part p's down over arrays waited at `down_first + p`, ordered after every gate/up and the previous part's down.
        /// Nothing waits on the host. One event wait per gate value: the gate and up arrays of every bank the call's misses
        /// read wait at `gu` together, then part p's down arrays of every such bank at `down_first + p` together (behind
        /// the same deps).
        fn gatedParts(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void {
            const pc = EventProjection.count;
            var gu: [n_banks]?Arrays = @splat(null);
            // The banks the misses read, in first appearance, each one's gate then up arrays.
            var order: [n_banks]usize = undefined;
            var nb: usize = 0;
            var seen: [n_banks]bool = @splat(false);
            var xs: [2 * n_banks * pc]T = undefined;
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or seen[b]) continue;
                seen[b] = true;
                const arrays = self.banks[layer][b].?;
                xs[2 * pc * nb ..][0..pc].* = EventProjection.inputs(arrays.gate);
                xs[2 * pc * nb + pc ..][0..pc].* = EventProjection.inputs(arrays.up);
                order[nb] = b;
                nb += 1;
            }
            if (nb > 0) {
                var outs: [2 * n_banks * pc]T = undefined;
                try self.eventWait(g, xs[0 .. 2 * pc * nb], gates.gu, &.{}, outs[0 .. 2 * pc * nb]);
                for (order[0..nb], 0..) |b, i| {
                    const arrays = self.banks[layer][b].?;
                    gu[b] = .{ .gate = EventProjection.fromWaited(outs[2 * pc * i ..][0..pc].*), .up = EventProjection.fromWaited(outs[2 * pc * i + pc ..][0..pc].*), .down = arrays.down };
                }
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gateUpWaveBanked(g, layer, xf, k, sv, @intCast(p + 1), &gu);
                for (waves[p].h[0..waves[p].n]) |h| {
                    hs[n_hs] = h;
                    n_hs += 1;
                }
            }
            var prev: []const T = &.{};
            for (0..sv.n_parts) |p| {
                var deps: [2 * max_route_ids]T = undefined;
                @memcpy(deps[0..n_hs], hs[0..n_hs]);
                @memcpy(deps[n_hs..][0..prev.len], prev);
                var dn = gu;
                var dx: [n_banks * pc]T = undefined;
                var nd: usize = 0;
                for (dn) |d| if (d) |arr| {
                    dx[pc * nd ..][0..pc].* = EventProjection.inputs(arr.down);
                    nd += 1;
                };
                if (nd > 0) {
                    var dout: [n_banks * pc]T = undefined;
                    try self.eventWait(g, dx[0 .. pc * nd], gates.down_first + p, deps[0 .. n_hs + prev.len], dout[0 .. pc * nd]);
                    var i: usize = 0;
                    for (&dn) |*d| if (d.*) |*arr| {
                        arr.down = EventProjection.fromWaited(dout[pc * i ..][0..pc].*);
                        i += 1;
                    };
                }
                prev = try self.downWaveBanked(g, layer, &waves[p], acc, &dn);
            }
        }

        /// `PackedDecode.run` (gate/up publish before down): the routing
        /// barrier, the residents' wave, then per miss part gate/up after
        /// `waitGu` and down after `waitDown`, each wave started on the GPU;
        /// release; the outputs in the router's order, `[n, k, hidden]` f32.
        pub fn run(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T) !T {
            return self.runEnd(g, layer, xf, indices, hoist, .take);
        }

        /// turboexl3_glm53f: a decode-lane call's outputs before the join's take (`Hook.routedHoistIndexed`,
        /// JOIN_INDEXED, TURBOEXL3_GLM53F_MOE_COMBINE1: the caller's combine reads the rows in place): the concatenated
        /// outputs [rows, hidden] and each routed position's row in them, u32 [n * k] (`joinOrder`; the stock join takes
        /// exactly these rows).
        pub const Indexed = struct { rows: T, order: T };
        /// How a call ends: the join's take (`run`, every caller's route) or the indexed rows (JOIN_INDEXED).
        const Ending = enum { take, indexed };

        fn runEnd(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T, comptime ending: Ending) !(if (ending == .take) T else Indexed) {
            const n: u32 = @intCast(g.shapeOf(xf).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            const n_ids = n * k;
            // A math with its own prefill token threshold routes by logical M.
            // DeepSeek keeps its decode / verify route through 48 top-6 ids.
            // A wide call is the wide lane's (`Hook.routedParts`).
            if (if (comptime @hasDecl(M, "prefill_min_tokens")) n >= M.prefill_min_tokens else n_ids > max_route_ids) return error.WideCallOnDecodeLane;
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [expert_lookahead.max_rows * 512]f32 = undefined;
            var scores: []const f32 = &.{};
            // The installed decode-only predictor follows the source's phase; the last layer predicts nothing.
            const predict = if (comptime routes.lookahead and predictor_decode_only) self.source.decoding() else routes.lookahead;
            const sc: ?T = if (if (predict) self.predictorGate(layer) else null) |gate| try self.scoresOf(g, xf, gate) else null;
            // The barrier's arrays: the ids, then the scores when there are any.
            var bar: [2]T = undefined;
            var n_bar: usize = 0;
            for ([_]?T{ indices, sc }) |x| if (x) |v| {
                bar[n_bar] = v;
                n_bar += 1;
            };
            // DEVROUTE (decode): the barrier's arrays committed, then the device hit wave behind them (before the host waits).
            const dev: ?T = if (comptime @hasDecl(S, "decoding")) (if (self.devroute and self.source.decoding()) try self.devWave(g, layer, xf, indices, n, k) else null) else null;
            if (dev) |d| {
                try g.asyncEval(bar[0..n_bar]);
                try g.asyncEval(&.{d});
            }
            if (n_bar > 1) try g.evalAll(bar[0..n_bar]);
            if (sc) |s| scores = try g.hostF32(s, score_buf[0 .. n * self.n_experts]);
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            const call = try self.source.route(layer, ids, scores);
            var released = false;
            errdefer if (!released) self.source.release(call);
            const sv = self.source.served(call);
            var acc: Acc = .{};
            if (dev) |d| {
                // DEVROUTE: every pair's row from the device wave; the miss parts' rows follow and replace theirs in the join.
                acc.outs[0] = d;
                acc.n_outs = 1;
                for (0..n_ids) |p| acc.pos[p] = @intCast(p);
                acc.n_pos = n_ids;
            } else {
                // Residents: gate/up and down at once.
                const hits = try self.gateUpWaveBanked(g, layer, xf, k, sv, 0, null);
                if (hits.n > 0) try g.asyncEval(try self.downWaveBanked(g, layer, &hits, &acc, null));
            }
            // VERIFY_ENCODE hoist: behind the hit wave, ahead of every miss wave.
            // turboexl3_glm53f: SHARED_HOIST: committed only when the call misses (a miss wave waits for its bytes); a call
            // with no miss part leaves it to the next barrier's evaluation, where the trunk's add reads it.
            if (sv.n_parts > 0 and hoist.len > 0) try g.asyncEval(hoist);
            if (routes.gated) {
                if (try self.source.gate(call)) |gates| {
                    try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
                }
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.source.waitGu(call, part);
                const wave = try self.gateUpWaveBanked(g, layer, xf, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                try self.source.waitDown(call, part);
                try g.asyncEval(try self.downWaveBanked(g, layer, &wave, &acc, null));
            }
            self.source.release(call);
            released = true;
            // DEVROUTE: the layer's next call reads the policy after this route (the other LUT of the pair).
            if (dev != null) try self.devLutAfterRoute(g, layer);
            return if (comptime ending == .take) self.join(g, &acc, n, k) else joinIndexed(g, &acc, n, k);
        }

        /// Each routed position's row in the concatenated outputs: the LAST row carrying it (DEVROUTE's miss parts follow
        /// the device wave and replace its rows for their pairs; the stock route carries each position once).
        fn joinOrder(pos: []const u32, n_ids: u32, out: *[max_route_ids]u32) u32 {
            for (pos, 0..) |p, j| out[p] = @intCast(j);
            return n_ids;
        }

        /// `take(concatenate(outputs), argsort(positions))`: the inverse permutation of the (unique)
        /// positions, made on the host; `[n, k, hidden]`.
        fn join(self: *const Self, g: *G, acc: *const Acc, n: u32, k: u32) !T {
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            const n_ids = joinOrder(acc.pos[0..acc.n_pos], n * k, &order);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// turboexl3_glm53f: `join` without its take: the concatenated outputs and the order the take would read them in
        /// (JOIN_INDEXED).
        fn joinIndexed(g: *G, acc: *const Acc, n: u32, k: u32) !Indexed {
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            const n_ids = joinOrder(acc.pos[0..acc.n_pos], n * k, &order);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return .{ .rows = joined, .order = ord };
        }

        /// P1's construction self-check (`Wide.read_ahead`), once on the real stream before any request:
        /// layer 0's records read ahead equal their demand reads, bit for bit, over two pool jobs.
        /// Returns how many records were checked.
        pub fn checkReadAhead(self: *Self) !u32 {
            if (comptime !@hasDecl(S, "checkReadAhead")) return error.ReadAheadNotStreamed;
            return self.source.checkReadAhead(0, expert_io.max_items + 1);
        }

        /// JOINLESS: the wide call's outputs unjoined (above `max_parts`, the smallest merged into one source:
        /// `mergeJoinless`) and each assignment's (source, row) as int32 [n, k, 2]; the combine reads the
        /// rows in place. The sources stay kept until `releaseParts`.
        pub const Parts = struct { outs: []const T, loc: T };

        fn runWideParts(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !Parts {
            try self.runWideCore(g, layer, xf, indices, n, k);
            const w = &self.wide;
            try w.loc.resize(self.a, 2 * n * k);
            if (comptime wide_profile.enabled) {
                const n_out = w.kept.items.len;
                const copied = try mergeJoinlessBound(G, g, self.a, &w.kept, w.pos.items, w.loc.items, &w.jl, self.joinless_bound);
                self.last_merge = .{ .outputs = @intCast(n_out), .sources = @intCast(w.kept.items.len), .copied_rows = copied };
            } else _ = try mergeJoinlessBound(G, g, self.a, &w.kept, w.pos.items, w.loc.items, &w.jl, self.joinless_bound);
            const loc = try g.hostArray(std.mem.sliceAsBytes(w.loc.items), &.{ @intCast(n), @intCast(k), 2 }, .int32);
            return .{ .outs = w.kept.items, .loc = loc };
        }

        /// Releases the outputs `runWideParts` kept (after the combines that read them are evaluated).
        pub fn releaseParts(self: *Self, g: *G) void {
            for (self.wide.kept.items) |x| g.release(x);
            self.wide.kept.clearRetainingCapacity();
        }

        /// A wide call's groups over its experts hottest first: the staged misses' ranks [0, sa) chunked apart from the
        /// call's others (`Wide.stage`; sa 0: one run of `max_route_ids` chunks).
        const Groups = struct {
            d: []const u16,
            sa: usize,
            seed_groups: usize,

            fn start(gs: Groups, gi: usize) usize {
                return if (gi < gs.seed_groups) gi * max_route_ids else gs.sa + (gi - gs.seed_groups) * max_route_ids;
            }

            fn of(gs: Groups, gi: usize) []const u16 {
                const s0 = gs.start(gi);
                return gs.d[s0..@min(s0 + max_route_ids, if (gi < gs.seed_groups) gs.sa else gs.d.len)];
            }

            /// The group of the expert at `fi` in the call's order (`start`'s inverse).
            fn index(gs: Groups, fi: usize) usize {
                return if (fi < gs.sa) fi / max_route_ids else gs.seed_groups + (fi - gs.sa) / max_route_ids;
            }
        };

        /// The call's rows bucketed by group (`bucketRowsByGroup`) into `inv` [n_ids] (the join's scratch, unused until the
        /// call's groups are done: no new per-row memory), group `gi`'s at `grp_off[gi]..grp_off[gi + 1]`. A group's bank
        /// passes then read its own rows only.
        fn groupRows(a: std.mem.Allocator, w: *WideScratch, groups: Groups, n_groups: usize) !void {
            try w.grp_of.resize(a, w.first.items.len);
            for (groups.d, 0..) |e, i| w.grp_of.items[e] = @intCast(groups.index(i));
            try w.grp_off.resize(a, n_groups + 1);
            try w.inv.resize(a, w.ids.items.len);
            bucketRowsByGroup(w.ids.items, groups.d, w.grp_of.items, w.count.items, w.grp_off.items, w.inv.items);
        }

        /// The wide lane: the routing barrier; the call's residency seeded from its ids (the most routed experts protected
        /// in the persistent rows); its distinct experts hottest first, routed in groups of at most `max_route_ids`, group
        /// g + 1 (its reads) routed before group g's waves at `Options.wide.depth` 2; per group, every part waited, then per
        /// bank the group's rows through the math's prompt route (slot = the row the source serves the expert in, act row
        /// = its token), the group's banks queued back to back and drained once before the group is released (the next
        /// route may refill those slots). The kernels read bf16 activations: another dtype is rounded to bf16 once, here.
        /// Each wave's output stays kept, each wave-ordered row's routed position in `pos`.
        fn runWideCore(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !void {
            if (comptime !(@hasDecl(M, "has_parts") and M.has_parts)) @compileError(@typeName(M) ++ ": the wide lane runs the math's prefillParts");
            const a = self.a;
            const w = &self.wide;
            wide_profile.countWideCall();
            const n_ids = n * k;
            // The wide lane takes prefill-width calls only (the prefill texts bind small inputs as
            // `constant`; a decode-width call is the decode lane's).
            if (comptime !@hasDecl(M, "prefill_min_tokens")) std.debug.assert(n_ids >= wide_min_ids);
            const depth: usize = self.wide_route.depth;
            // turboexl3_glm53f: `Wide.stage`: the layer's misses routed before the barrier into the transient windows, the
            // residents held meanwhile (the routes' misses take transient rows, or an empty row); their reads run while
            // the barrier's evaluation runs the layer's attention. The calls are handed to `calls` below.
            var staged_calls: [expert_stream.max_wide_depth]?*S.Call = @splat(null);
            errdefer for (&staged_calls) |*c| if (c.*) |cl| {
                self.source.release(cl);
                c.* = null;
            };
            var staged_n: usize = 0;
            if (comptime @hasDecl(S, "stageMisses") and @hasDecl(S, "holdResidents")) if (self.wide_route.stage and n_ids >= 8 * self.n_experts) {
                try w.staged.resize(a, depth * max_route_ids);
                staged_n = self.source.stageMisses(layer, w.staged.items);
                if (staged_n > 0) {
                    try self.source.holdResidents(layer);
                    defer self.source.releaseHeld();
                    var si: usize = 0;
                    while (si * max_route_ids < staged_n) : (si += 1) {
                        staged_calls[si] = try self.source.route(layer, w.staged.items[si * max_route_ids .. @min((si + 1) * max_route_ids, staged_n)], &.{});
                    }
                }
            };
            try w.ids.resize(a, n_ids);
            var tp = wide_profile.now();
            _ = try g.hostIds(indices, w.ids.items);
            wide_profile.charge(.barrier, tp);
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            // The call's experts in first appearance and (the feed's order) their rows: one pass.
            try w.distinct.ensureTotalCapacity(a, self.n_experts);
            try w.count.resize(a, self.n_experts);
            @memset(w.count.items, 0);
            for (w.ids.items) |e| {
                if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    w.distinct.appendAssumeCapacity(e);
                }
                w.count.items[e] += 1;
            }
            // The call's residency seed, then its experts hottest first.
            // P1: the layer's read-ahead lands before its seed (which protects the ones it chooses).
            if (comptime @hasDecl(S, "awaitReadAhead")) if (self.wide_route.read_ahead) try self.source.awaitReadAhead(layer);
            try self.source.seedPrefill(layer, w.ids.items);
            if (staged_n > 0) {
                // turboexl3_glm53f: `Wide.stage`: the staged experts first, in their routes' order (any the call does not
                // route among them, with no rows), then the call's other experts hottest first (the stock count order).
                // Not the source's own order: its P7 plans take a call's groups at its first route, and the staged
                // routes are live before this call's order is known.
                const staged = w.staged.items[0..staged_n];
                for (staged) |e| w.first.items[e] = -2;
                var r: usize = 0;
                for (w.distinct.items) |e| if (w.first.items[e] != -2) {
                    w.distinct.items[r] = e;
                    r += 1;
                };
                std.sort.pdq(u16, w.distinct.items[0..r], @as([]const u32, w.count.items), struct {
                    fn lt(c: []const u32, x: u16, y: u16) bool {
                        return if (c[x] != c[y]) c[x] > c[y] else x < y;
                    }
                }.lt);
                try w.distinct.resize(a, staged_n + r);
                std.mem.copyBackwards(u16, w.distinct.items[staged_n..][0..r], w.distinct.items[0..r]);
                @memcpy(w.distinct.items[0..staged_n], staged);
            } else self.source.hotOrder(layer, w.distinct.items, w.count.items);
            for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            const act = if (g.dtypeOf(xf) == .bfloat16) xf else try g.astype(xf, .bfloat16);
            w.pos.clearRetainingCapacity();
            w.kept.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            // `Wide.stage`: the staged ranks [0, sa) in groups of their own, the call's others after them (sa 0: one run).
            const sa: usize = staged_n;
            const groups: Groups = .{ .d = w.distinct.items, .sa = sa, .seed_groups = (sa + max_route_ids - 1) / max_route_ids };
            const n_groups = groups.seed_groups + (n_distinct - sa + max_route_ids - 1) / max_route_ids;
            try groupRows(a, w, groups, n_groups);
            // The live groups' calls, by group index mod `depth` (read ahead up to `depth` groups).
            var calls: [expert_stream.max_wide_depth]?*S.Call = @splat(null);
            errdefer {
                for (&calls) |*c| if (c.*) |cl| {
                    self.source.release(cl);
                    c.* = null;
                };
                for (w.kept.items) |x| g.release(x);
                w.kept.clearRetainingCapacity();
            }
            // `Wide.stage`: the staged routes are groups 0 .. (their errdefer's calls handed over)
            for (&staged_calls, 0..) |*c, i| if (c.*) |cl| {
                calls[i] = cl;
                c.* = null;
            };
            tp = wide_profile.now();
            for (0..@min(depth, n_groups)) |gi| {
                if (calls[gi % depth] == null) calls[gi % depth] = try self.source.route(layer, groups.of(gi), &.{});
            }
            wide_profile.charge(.route, tp);
            // turboexl3_glm53f: `Wide.drain_lag` (TURBOEXL3_GLM53F_PF_DRAIN_LAG): the pending group's drain, its call's
            // release and the route `depth` groups on, run at the next group's first commit (the math's `LagPending`)
            const lag = self.wide_route.drain_lag;
            const LagCtx = struct {
                ex: *Self,
                g: *G,
                layer: u32,
                calls: *[expert_stream.max_wide_depth]?*S.Call,
                groups: Groups,
                n_groups: usize,
                depth: usize,
                pub fn drain(c: @This(), k0: usize, k1: usize) !void {
                    const t = wide_profile.now();
                    try c.g.evalAll(c.ex.wide.kept.items[k0..k1]);
                    wide_profile.charge(.drain, t);
                }
                pub fn release(c: @This(), gi: usize) !void {
                    const t = wide_profile.now();
                    c.ex.source.release(c.calls[gi % c.depth].?);
                    c.calls[gi % c.depth] = null;
                    if (gi + c.depth < c.n_groups) c.calls[gi % c.depth] = try c.ex.source.route(c.layer, c.groups.of(gi + c.depth), &.{});
                    wide_profile.charge(.route, t);
                }
            };
            // (a math without `LagPending` never lags: `initWith` refuses the route)
            const NoLag = struct {
                fn begin(_: *@This()) void {}
                fn end(_: *@This(), _: usize, _: usize, _: usize) !void {}
                fn flush(_: *@This()) !void {}
                fn disarm(_: *@This()) void {}
            };
            const has_lag = comptime @hasDecl(M, "LagPending");
            var pending: if (has_lag) M.LagPending(LagCtx) else NoLag = if (has_lag) .{ .ctx = .{ .ex = self, .g = g, .layer = layer, .calls = &calls, .groups = groups, .n_groups = n_groups, .depth = depth } } else .{};
            // the hook disarmed on every way out (an error included; the calls' errdefer releases the pending group's)
            defer if (lag) pending.disarm();
            for (0..n_groups) |gi| {
                const start = groups.start(gi);
                const call = calls[gi % depth].?;
                const sv = self.source.served(call);
                tp = wide_profile.now();
                const ready: ?u64 = if (comptime wide_event_gates) blk: {
                    // Register only the group whose consumer is encoded next. Its final down value
                    // proves every gate/up and down read in the group complete.
                    const gates = (try self.source.gate(call)) orelse break :blk null;
                    break :blk gates.down_first + gates.n_parts - 1;
                } else blk: {
                    for (0..sv.n_parts) |p| {
                        try self.source.waitGu(call, @intCast(p));
                        try self.source.waitDown(call, @intCast(p));
                    }
                    break :blk null;
                };
                wide_profile.charge(.read_wait, tp);
                const k0 = w.kept.items.len;
                // turboexl3_glm53f: `Wide.drain_lag`: the pending group's drain armed for this group's first commit
                if (lag) pending.begin();
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    w.slot.clearRetainingCapacity();
                    w.act_row.clearRetainingCapacity();
                    // The group's own rows, in row order (`groupRows`), each bank's rows as a pass over the whole call
                    // meets them.
                    for (w.inv.items[w.grp_off.items[gi]..w.grp_off.items[gi + 1]]) |row| {
                        const e = w.ids.items[row];
                        const fi: usize = @intCast(w.first.items[e]);
                        const ref = sv.refs[fi - start];
                        if (ref.bank != kind) continue;
                        const act_row: u32 = @intCast(row / @as(usize, k));
                        try w.slot.append(a, ref.row);
                        try w.act_row.append(a, act_row);
                        try w.call_pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0) continue;
                    var b = self.banks[layer][@backingInt(kind)].?;
                    if (comptime wide_event_gates) if (ready) |value| {
                        b = .{
                            .gate = try self.waitProj(g, b.gate, value, &.{}),
                            .up = try self.waitProj(g, b.up, value, &.{}),
                            .down = try self.waitProj(g, b.down, value, &.{}),
                        };
                    };
                    // The waves' outputs as they are; each wave-ordered row's routed position.
                    w.wave_pos.clearRetainingCapacity();
                    try self.math.prefillParts(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b, a, &w.kept, &w.wave_pos);
                    for (w.wave_pos.items) |p| try w.pos.append(a, w.call_pos.items[p]);
                    w.call_pos.clearRetainingCapacity();
                }
                // The group's banks queued back to back, one drain before its slots go back.
                try self.math.finishPrefill(g);
                if (lag) {
                    // turboexl3_glm53f: `Wide.drain_lag`: the pending group's drain and release if this group
                    // committed nothing, then this group pending (its drain at the next group's first commit)
                    try pending.end(gi, k0, w.kept.items.len);
                } else {
                    tp = wide_profile.now();
                    try g.evalAll(w.kept.items[k0..]);
                    wide_profile.charge(.drain, tp);
                }
                if (!lag) {
                    tp = wide_profile.now();
                    self.source.release(call);
                    calls[gi % depth] = null;
                    if (gi + depth < n_groups) calls[gi % depth] = try self.source.route(layer, groups.of(gi + depth), &.{});
                    wide_profile.charge(.route, tp);
                }
            }
            // turboexl3_glm53f: `Wide.drain_lag`: the call's last group drained and released
            if (lag) try pending.flush();
        }
    };
}
