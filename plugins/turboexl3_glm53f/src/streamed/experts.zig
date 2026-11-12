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
//! Sources: `StreamSource` (the streamer's `Stream`) and `FakeSource` (the
//! streamer's residency policy with no reads, for host tests). The math of a
//! group of rows sharing a bank is the C2 `quant` seam (quant.zig): `QuantMath`
//! over an accepted quant (the EXL3 quant: PREP=rin + the decode GEMV, DIG-X
//! prefill waves) is the served math; `EagerChain` is the stock tier's op chain
//! around the quant's decode GEMV (the parity harnesses); `TraceGemv` /
//! `TraceMath` stand in on the trace backend.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const ops = @import("ops.zig");
const expert_bank = @import("expert/bank.zig");
const expert_io = sdk_ext.expert.io;
const expert_policy = @import("sdk_ext.zig").expert.policy;
const expert_stream = @import("expert/stream.zig");
const expert_lookahead = @import("expert/lookahead.zig");
const expert_event = sdk_ext.expert.event;
const xk = @import("../exl3/kernels.zig");
const sdk_ext = @import("sdk_ext.zig");
const quant = @import("sdk_ext.zig").quant;
const dt = @import("profile/decode_timers.zig");
const timeline = @import("profile/verify_timeline.zig");
const first_cycle = @import("profile/decode_first.zig");
const prof = @import("profile/prefill_timers.zig");
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

/// The stream's wave numbering of a call's ids (`expert_stream.wavesOf`).
const wavesOf = expert_stream.wavesOf;

/// Trace arrays in one record's geometry (`expert_stream.traceBank`).
const traceBank = expert_stream.traceBank;

/// DEVROUTE's resident map of one layer from its policy (`expert_stream.lutOf`).
const lutOf = expert_stream.lutOf;

// ── StreamSource: the streamer's Stream, the EXL3 source's side of the contract (`expert_stream.StreamSource`) ──

pub const StreamSource = expert_stream.StreamSource;

// ── FakeSource: the streamer's residency with no reads (host tests) ──

/// The streamer's residency policy (one `LayerPolicy` per layer, the Stream's
/// part cutting and slot arithmetic) with every load landed at once and no
/// memory: the fixture-backed stand-in for host tests. It keeps the Stream's
/// call rules (a route flushes released calls first; four calls at most;
/// `grow` once, with every call released) and logs every call it gets.
pub const FakeSource = struct {
    a: std.mem.Allocator,
    geom: expert_bank.Layer,
    n_experts: u32,
    policies: []expert_policy.LayerPolicy,
    base_rows: []u32,
    ext_rows: []u32,
    transient_rows: u32,
    records_per_part: u32,
    phase: expert_policy.Phase = .prefill,
    calls: [4]Call = @splat(.{}),
    log: std.ArrayList(Event) = .empty,
    /// Slots a deferred call still reads (`holdBase`): held in every later plan of `held_layer`.
    held: std.ArrayList(u32) = .empty,
    held_layer: u32 = 0,
    counters: Stats = .{},
    /// When set, events are stamped with its node count (op order checks).
    trace: ?*const ops.TraceOps = null,
    /// The lookahead's selection on the next layer's gate scores (the Stream's
    /// `speculate`), logged per call that passes scores.
    selector: ?expert_lookahead.Selector = null,
    picks: std.ArrayList(Pick) = .empty,
    gate_value: u64 = 0,
    /// P1: the layer whose read-ahead is live (the Stream's rule: its first route or the barrier lands it),
    /// and every admission in order (the tests' view of the seed the model handed over).
    ahead_layer: ?u32 = null,
    ahead_log: std.ArrayList(u16) = .empty,

    pub const caps: sdk_ext.expert.Caps = .{ .two_phase = true, .prompt_seed = true, .read_ahead = true, .wide = true, .event_gates = true };

    pub const Pick = struct { layer: u32, n: u8 = 0, experts: [expert_lookahead.max_budget]u16 = undefined };

    pub const Call = struct {
        state: enum { free, live, released } = .free,
        layer: u32 = 0,
        plan: expert_policy.Plan = .{},
        n_parts: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub const Event = struct {
        pub const Kind = enum { route, wait_gu, wait_down, release, flush, grow, gate, read_ahead, await_read_ahead };
        kind: Kind,
        layer: u32 = 0,
        part: u32 = 0,
        /// Trace nodes recorded before the event (0 without `trace`).
        at: usize = 0,
    };

    pub const Options = struct {
        hidden: u64,
        inter: u64,
        n_experts: u32,
        /// Persistent rows per layer before `grow`.
        rows: []const u32,
        transient_rows: u32 = max_route_ids,
        records_per_part: u32 = 3,
    };

    pub fn init(a: std.mem.Allocator, opt: Options) !FakeSource {
        const geom = expert_bank.layerSegments(3, opt.hidden, opt.inter) orelse return error.InvalidGeometry;
        const n = opt.rows.len;
        const policies = try a.alloc(expert_policy.LayerPolicy, n);
        errdefer a.free(policies);
        var n_init: usize = 0;
        errdefer for (policies[0..n_init]) |*p| p.deinit(a);
        for (policies, opt.rows) |*p, rows| {
            p.* = try expert_policy.LayerPolicy.init(a, opt.n_experts, rows);
            n_init += 1;
        }
        const base = try a.dupe(u32, opt.rows);
        errdefer a.free(base);
        const ext = try a.alloc(u32, n);
        @memset(ext, 0);
        return .{
            .a = a,
            .geom = geom,
            .n_experts = opt.n_experts,
            .policies = policies,
            .base_rows = base,
            .ext_rows = ext,
            .transient_rows = opt.transient_rows,
            .records_per_part = opt.records_per_part,
        };
    }

    pub fn deinit(self: *FakeSource) void {
        if (self.selector) |*sel| sel.deinit(self.a);
        self.picks.deinit(self.a);
        for (self.policies) |*p| p.deinit(self.a);
        self.a.free(self.policies);
        self.a.free(self.base_rows);
        self.a.free(self.ext_rows);
        self.log.deinit(self.a);
        self.held.deinit(self.a);
        self.ahead_log.deinit(self.a);
        self.* = undefined;
    }

    fn note(self: *FakeSource, ev: Event) void {
        var e = ev;
        if (self.trace) |t| e.at = t.nodes.items.len;
        self.log.append(self.a, e) catch @panic("fake source log: out of memory");
    }

    pub fn holdBase(self: *FakeSource, call: *Call) !void {
        if (self.held.items.len > 0 and self.held_layer != call.layer) return error.HeldOtherLayer;
        self.held_layer = call.layer;
        const pol = &self.policies[call.layer];
        for (call.plan.hitsOf()) |e| {
            const s = pol.slotOf(e).?;
            if (s < pol.capacity) try self.held.append(self.a, s);
        }
        for (call.plan.loadsOf()) |l| if (l.persistent) try self.held.append(self.a, l.slot);
    }

    pub fn releaseHeld(self: *FakeSource) void {
        self.held.clearRetainingCapacity();
    }

    pub fn seedPrefill(self: *FakeSource, layer: u32, ids: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        self.policies[layer].prepareSeed(ids);
    }

    pub fn seedRanks(self: *const FakeSource, layer: u32) u32 {
        return self.policies[layer].seed_ranks;
    }

    /// P1: the Stream's admission (`LayerPolicy.admitReadAhead`), no reads; `part` = experts admitted.
    pub fn readAheadSeed(self: *FakeSource, layer: u32, experts: []const u16) !void {
        if (self.phase != .prefill) return error.NotPrefill;
        if (self.ahead_layer) |l| try self.awaitReadAhead(l);
        var out: [max_ahead]expert_policy.LayerPolicy.ReadAhead = undefined;
        const admitted = self.policies[layer].admitReadAhead(experts, &out);
        const n = admitted.len;
        for (admitted) |r| self.ahead_log.append(self.a, r.expert) catch @panic("fake source read-ahead log: out of memory");
        if (n > 0) self.ahead_layer = layer;
        self.note(.{ .kind = .read_ahead, .layer = layer, .part = @intCast(n) });
    }

    pub fn awaitReadAhead(self: *FakeSource, layer: u32) Error!void {
        if (self.ahead_layer != layer) return;
        self.ahead_layer = null;
        self.note(.{ .kind = .await_read_ahead, .layer = layer });
    }

    const max_ahead = 1024;

    pub fn slotRef(self: *const FakeSource, layer: u32, slot: u32) SlotRef {
        const cap = self.policies[layer].capacity;
        if (slot < self.base_rows[layer]) return .{ .bank = .base, .row = slot };
        if (slot < cap) return .{ .bank = .ext, .row = slot - self.base_rows[layer] };
        return .{ .bank = .transient, .row = slot - cap };
    }

    pub fn route(self: *FakeSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        std.debug.assert(ids.len > 0 and ids.len <= max_route_ids);
        std.debug.assert(scores.len == 0 or self.selector != null);
        if (self.ahead_layer == layer) try self.awaitReadAhead(layer);
        try self.flush();
        const call = for (&self.calls) |*c| {
            if (c.state == .free) break c;
        } else return error.RoutesExhausted;
        call.* = .{ .layer = layer };
        const plan = &call.plan;
        const held: []const u32 = if (self.held.items.len > 0 and self.held_layer == layer) self.held.items else &.{};
        self.policies[layer].planWith(ids, self.phase, plan, .{ .held = held });
        // A row a live call still serves from, or a deferred call still reads, is never refilled.
        for (plan.loadsOf()) |l| {
            if (std.mem.indexOfScalar(u32, held, l.slot) != null) return error.SlotStillPinned;
            for (&self.calls) |*o| {
                if (o == call or o.state != .live or o.layer != layer) continue;
                if (std.mem.indexOfScalar(u32, o.plan.slotsOf(), l.slot) != null) return error.SlotStillPinned;
            }
        }
        // The Stream's parts: loads in file (expert) order; decode = bounded
        // parts of `records_per_part` (no two records touch), prefill = pool-job chunks.
        var order: [max_route_ids]Load = undefined;
        const n = plan.n_loads;
        @memcpy(order[0..n], plan.loadsOf());
        std.sort.insertion(Load, order[0..n], {}, struct {
            fn less(_: void, x: Load, y: Load) bool {
                return x.expert < y.expert;
            }
        }.less);
        var offsets: [max_route_ids]u64 = undefined;
        var lengths: [max_route_ids]u64 = undefined;
        for (order[0..n], 0..) |l, i| {
            offsets[i] = @as(u64, l.expert) * self.geom.record_bytes;
            lengths[i] = self.geom.logical_bytes;
        }
        var ends_buf: [max_route_ids]u32 = undefined;
        const ends = if (self.phase == .decode)
            expert_policy.boundedParts(offsets[0..n], lengths[0..n], self.records_per_part, &ends_buf)
        else blk: {
            var k: u32 = 0;
            var e: u32 = 0;
            while (e < n) : (k += 1) {
                e = @min(e + expert_io.max_items, n);
                ends_buf[k] = e;
            }
            break :blk ends_buf[0..k];
        };
        var parts: [max_route_ids][]const Load = undefined;
        var start: u32 = 0;
        for (ends, 0..) |end, p| {
            parts[p] = order[start..end];
            start = end;
        }
        call.n_parts = @intCast(ends.len);
        var hit_slots: [max_route_ids]u32 = undefined;
        for (plan.hitsOf(), hit_slots[0..plan.n_hits]) |e, *s| s.* = self.policies[layer].slotOf(e).?;
        wavesOf(plan, hit_slots[0..plan.n_hits], parts[0..ends.len], call.waves[0..ids.len]);
        for (plan.slotsOf(), call.refs[0..ids.len]) |s, *r| r.* = self.slotRef(layer, s);
        const c = &self.counters;
        c.route_calls += 1;
        c.expert_cache_hits += plan.n_hits;
        c.expert_cache_misses += plan.n_misses;
        c.expert_cache_evictions += plan.n_evictions;
        for (plan.loadsOf()) |l| {
            if (l.persistent) c.persistent_loads += 1 else c.transient_loads += 1;
        }
        if (scores.len > 0 and self.phase == .decode and layer + 1 < self.policies.len) {
            const sel = &self.selector.?;
            var pick: Pick = .{ .layer = layer + 1 };
            pick.n = @intCast(sel.select(scores, &self.policies[layer + 1], pick.experts[0..sel.budget]).len);
            self.picks.append(self.a, pick) catch @panic("fake source picks: out of memory");
        }
        call.state = .live;
        self.note(.{ .kind = .route, .layer = layer });
        return call;
    }

    /// The Stream's gate values: the call's gate/up wave at `gu`, part p's down at `down_first + p`.
    pub fn gate(self: *FakeSource, call: *Call) Error!?expert_stream.Gates {
        std.debug.assert(call.state == .live);
        if (call.n_parts == 0) return null;
        const lo = self.gate_value;
        self.gate_value = lo + 1 + call.n_parts;
        self.note(.{ .kind = .gate, .layer = call.layer, .part = call.n_parts });
        return .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = call.n_parts };
    }

    /// DEVROUTE: as `StreamSource.residentLut`.
    pub fn residentLut(self: *const FakeSource, layer: u32, out: []u32) void {
        lutOf(&self.policies[layer], self, layer, out);
    }

    pub fn decoding(self: *const FakeSource) bool {
        return self.phase == .decode;
    }

    pub fn served(_: *FakeSource, call: *const Call) Served {
        const n = call.plan.n_ids;
        return .{ .refs = call.refs[0..n], .waves = call.waves[0..n], .n_parts = call.n_parts };
    }

    pub fn waitGu(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_gu, .layer = call.layer, .part = part });
    }

    pub fn waitDown(self: *FakeSource, call: *Call, part: u32) Error!void {
        std.debug.assert(call.state == .live and part < call.n_parts);
        self.note(.{ .kind = .wait_down, .layer = call.layer, .part = part });
    }

    pub fn release(self: *FakeSource, call: *Call) void {
        std.debug.assert(call.state == .live);
        call.state = .released;
        self.note(.{ .kind = .release, .layer = call.layer });
    }

    pub fn flush(self: *FakeSource) Error!void {
        var any = false;
        for (&self.calls) |*c| if (c.state == .released) {
            c.state = .free;
            any = true;
        };
        if (any) self.note(.{ .kind = .flush });
    }

    pub fn grow(self: *FakeSource, decode_rows: []const u32) !void {
        if (self.phase != .prefill) return error.AlreadyGrown;
        if (decode_rows.len != self.policies.len) return error.InvalidRows;
        try self.flush();
        for (&self.calls) |*c| if (c.state != .free) return error.RoutesLive;
        for (self.policies, decode_rows) |*p, rows| {
            if (rows < p.capacity or rows > self.n_experts) return error.InvalidRows;
        }
        for (self.policies, decode_rows, self.ext_rows) |*p, rows, *ext| {
            ext.* = rows - p.capacity;
            p.grow(rows) catch unreachable;
        }
        self.phase = .decode;
        self.note(.{ .kind = .grow });
    }

    pub fn stats(self: *FakeSource) Stats {
        return self.counters;
    }

    pub fn isResident(self: *const FakeSource, layer: u32, expert: u16) bool {
        return self.policies[layer].slotOf(expert) != null;
    }

    pub fn bankRows(self: *FakeSource, layer: u32, kind: BankKind) u32 {
        return switch (kind) {
            .base => self.base_rows[layer],
            .ext => self.ext_rows[layer],
            .transient => self.transient_rows,
        };
    }

    pub fn bankArrays(self: *FakeSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        if (@TypeOf(g.*) != ops.TraceOps) @compileError("FakeSource binds trace arrays only");
        const rows = self.bankRows(layer, kind);
        if (rows == 0) return null;
        return try traceBank(g, &self.geom, rows);
    }

    /// Calls that are neither free nor released (tests).
    pub fn liveCalls(self: *const FakeSource) usize {
        var n: usize = 0;
        for (self.calls) |c| n += @intFromBool(c.state == .live);
        return n;
    }
};

// ── The math seam ──

/// `128 ** -0.5` as the Python float rounds to the f32 MLX takes.
const t128_scale: f32 = @bitCast(@as(u32, 0x3db504f3));

/// The stock tier's routed-expert math, `exl3_lane.Exl3PackedOps` op for op:
/// per projection `t128(gemv(t128(x * rin[slot])) ) * rout[slot]` (f32), the
/// clamped SwiGLU between gate/up and down. `Gemv.project(g, proj, xh, ids,
/// code)` is the EXL3 decode GEMV (`z = xh @ W_hat[slot]` in the trellis domain,
/// f32 [rows, out]): the accepted EXL3 quant's (`*const exl3_quant.Gemv(G)`).
/// The parity harnesses' math (the stock path reads f32 routed rows).
pub fn EagerChain(comptime G: type, comptime Gemv: type) type {
    return struct {
        const Self = @This();
        const T = G.T;

        gemv: Gemv,
        swiglu_limit: f64,

        pub fn init(gemv: Gemv, c: anytype) Self {
            return .{ .gemv = gemv, .swiglu_limit = c.swiglu_limit };
        }

        /// `tcq_runtime._t128`: the normalised 128-block Walsh-Hadamard over the last axis.
        fn t128(g: *G, v: T) !T {
            const sh = g.shapeOf(v);
            var blocks = sh;
            blocks.d[sh.n - 1] = @divExact(sh.d[sh.n - 1], 128);
            blocks.d[sh.n] = 128;
            blocks.n += 1;
            const h = try g.hadamard(try g.reshape(try g.astype(v, .float32), blocks.slice()), t128_scale);
            return g.reshape(h, sh.slice());
        }

        /// `Exl3PackedOps._project`.
        fn project(self: *const Self, g: *G, x: T, ids: T, p: ProjOf(T), proj: xq.Proj) !T {
            const rin = try g.astype(try g.take(p.rin, ids, 0), .float32);
            const xh = try t128(g, try g.mul(try g.astype(x, .float32), rin));
            const z = try self.gemv.project(g, proj, xh, ids, p.code);
            const rout = try g.astype(try g.take(p.rout, ids, 0), .float32);
            return g.mul(try t128(g, z), rout);
        }

        /// Gate and up of rows `x` [rows, hidden] at slot rows `ids` [rows]
        /// (uint32), then `_clamped_swiglu(g, u, limit)`: f32 [rows, inter].
        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const gg = try self.project(g, x, ids, gate, .gate);
            const uu = try self.project(g, x, ids, up, .up);
            const lim = self.swiglu_limit;
            const uc = try g.clip(uu, try g.scalar(-lim, .float32), try g.scalar(lim, .float32));
            const gc = try g.minimum(gg, try g.scalar(lim, .float32));
            return g.mul(try g.silu(gc), uc);
        }

        /// Down of the SwiGLU rows `h` [rows, inter]: f32 [rows, hidden].
        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.project(g, h, ids, d, .down);
        }
    };
}

/// The served routed-expert math: an accepted C2 quant (`quant.checkAccepted`),
/// its decode `gateUp` / `down` (the EXL3 quant: PREP=rin around the decode
/// GEMV) and its prefill waves. The quant reads bf16 routed rows: a call whose
/// rows arrive in another dtype is rounded to bf16 once, here (the RC tier's
/// MoE input is bf16 already).
pub fn QuantMath(comptime G: type, comptime Q: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        /// The widest decode-lane call (`gateUp` / `down`).
        pub const max_decode_rows: u32 = Q.max_decode_rows;
        q: *Q,

        pub fn init(q: *Q, _: anytype) Self {
            return .{ .q = q };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            return self.q.gateUp(g, xb, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.q.down(g, h, ids, d);
        }

        /// The quant's banked route, when it has one (a construction-time type choice; `Options.banked` binds it).
        pub const has_banked = @hasDecl(Q, "gateUpBanked");

        /// ids packed (bank << 24 | slot row); `banks` base, ext, transient.
        pub fn gateUpBanked(self: *const Self, g: *G, x: T, ids: T, banks: *const [n_banks]BankArraysOf(T)) !T {
            const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
            return self.q.gateUpBanked(g, xb, ids, banks);
        }

        pub fn downBanked(self: *const Self, g: *G, h: T, ids: T, banks: *const [n_banks]BankArraysOf(T)) !T {
            return self.q.downBanked(g, h, ids, banks);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.q.prefill(g, layer, x, rows, bank);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            return self.q.finishPrefill(g);
        }

        /// The quant's unjoined prefill, when it has one (a construction-time type choice).
        pub const has_parts = @hasDecl(Q, "prefillParts");

        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T), alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            return self.q.prefillParts(g, layer, x, rows, bank, alloc, outs, pos);
        }
    };
}

/// A decode math `D` with one prefill route `P` per layer (`call(g, x, rows,
/// bank)` / `finish(g)`: the quant's prefill shape), for host tests.
pub fn WithPrefillRoutes(comptime G: type, comptime D: type, comptime P: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        pub const max_decode_rows: u32 = if (@hasDecl(D, "max_decode_rows")) D.max_decode_rows else max_route_ids;
        d: D,
        routes: []P,

        pub fn gateUp(self: *const Self, g: *G, x: T, ids: T, gate: ProjOf(T), up: ProjOf(T)) !T {
            return self.d.gateUp(g, x, ids, gate, up);
        }

        pub fn down(self: *const Self, g: *G, h: T, ids: T, d: ProjOf(T)) !T {
            return self.d.down(g, h, ids, d);
        }

        pub fn prefill(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T)) !T {
            return self.routes[layer].call(g, x, rows, bank);
        }

        pub const has_parts = @hasDecl(P, "callParts");

        pub fn prefillParts(self: *const Self, g: *G, layer: u32, x: T, rows: quant.PrefillRows, bank: BankArraysOf(T), alloc: std.mem.Allocator, outs: *std.ArrayList(T), pos: *std.ArrayList(u32)) !void {
            return self.routes[layer].callParts(g, x, rows, bank, alloc, outs, pos);
        }

        pub fn finishPrefill(self: *const Self, g: *G) !void {
            for (self.routes) |*r| try r.finish(g);
        }
    };
}

/// The GEMV's launch contract on the trace backend (the kernel manifest's
/// inputs: xh f32 [rows, in], ids uint32 [rows], code int16 [cap, in/16,
/// out/16, 48] at K 3); the output is a kernel node f32 [rows, out].
pub const TraceGemv = struct {
    pub fn project(_: TraceGemv, g: *ops.TraceOps, _: xq.Proj, xh: u32, ids: u32, code: u32) !u32 {
        const sx = g.shapeOf(xh);
        const si = g.shapeOf(ids);
        const sc = g.shapeOf(code);
        if (g.dtypeOf(xh) != .float32 or g.dtypeOf(ids) != .uint32 or g.dtypeOf(code) != .int16) return error.GemvDtype;
        if (sx.n != 2 or si.n != 1 or sc.n != 4 or si.d[0] != sx.d[0]) return error.GemvShape;
        if (sc.d[1] * 16 != sx.d[1] or sc.d[3] != 48) return error.GemvShape;
        return g.kernel(&.{ sx.d[0], sc.d[2] * 16 }, .float32);
    }
};

/// Shape-level math for trace models whose widths no EXL3 bank has (the mini
/// config's hidden 64): one kernel node per projection pair.
pub const TraceMath = struct {
    hidden: c_int,
    inter: c_int,

    pub fn gateUp(self: *const TraceMath, g: *ops.TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        _ = gate;
        _ = up;
        if (g.shapeOf(x).d[1] != self.hidden or g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .float32);
    }

    pub fn down(self: *const TraceMath, g: *ops.TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        _ = d;
        if (g.shapeOf(h).d[1] != self.inter or g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .float32);
    }
};

// ── The executor: the model's routed hook over a source ──

/// `argsort(positions)` of unique positions: `inv[pos[j]] = j` (the row of
/// the joined outputs that holds routed position `pos[j]`).
fn invertPositions(pos: []const u32, inv: []u32) void {
    for (pos, 0..) |p, j| inv[p] = @intCast(j);
}

/// The most sources JOINLESS's combine reads (`dsv41_kernel_routes.JoinlessCombine.sources`).
pub const joinless_sources = 24;

/// JOINLESS's sources for outputs of `rows` rows each (join order), the minimal copy: up to
/// `joinless_sources` outputs are each read in place; above, the smallest `n - 23` (earlier first among
/// equals) are concatenated in join order into the last source and the rest are sources 0.. in join order.
/// Per output: `src[i]` its source, `off[i]` its first row there; returns the sources. `order`: scratch.
/// Exact: a concatenate copies words, and the combine folds each token's k products in k order.
pub fn planJoinless(rows: []const u32, order: []u32, src: []u8, off: []u32) usize {
    return planJoinlessBound(rows, order, src, off, joinless_sources);
}

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

/// JOINLESS's merge: `outs` (the KEPT outputs, join order; `pos`: each join-ordered row's assignment) becomes
/// the combine's sources under `planJoinless` (the in-place outputs, then the merged one, KEPT; its inputs
/// released) and `loc` each assignment's (source, row). Returns the rows copied.
pub fn mergeJoinless(comptime G: type, g: *G, a: std.mem.Allocator, outs: *std.ArrayList(G.T), pos: []const u32, loc: []i32, sc: *JoinlessScratch(G.T)) !u64 {
    return mergeJoinlessBound(G, g, a, outs, pos, loc, sc, joinless_sources);
}

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

/// The wide lane's read schedule, chosen at construction (`Options.wide`; both
/// exact: the same rows, slots and kernels, only the order and timing of reads
/// and evals change).
pub const Wide = struct {
    /// Each call seeds its layer's residency with its own ids (the most routed
    /// experts protected in the persistent rows) and drains each group once
    /// (its banks' waves queued).
    seed: bool = false,
    /// Each call feeds its groups hottest first (routed rows descending, ties by
    /// id) instead of in first appearance. `seed` + `hot_first` = the wide feed.
    hot_first: bool = false,
    /// Groups in flight: 2 routes (reads) the next group before this
    /// group's waves (the source's `wideDepth` must allow it).
    depth: u8 = 1,
    /// Experts with at most this many rows in the call run the decode lane's
    /// math (the GEMV, in chunks of the math's `max_decode_rows`) after their
    /// bank's wide waves, not the wide route (0: none). ROUNDING-CLASS against
    /// the wide route: its own reference family.
    cold_rows: u8 = 0,
    /// Every group's base-bank rows of a call built as ONE call's waves after the groups' transient
    /// waves (their slots held until then): fewer, fuller waves. Exact: a wave's rows are
    /// independent of its composition and the combine folds by routed position.
    defer_base: bool = false,
    /// P1: each layer's predicted seed read into its persistent rows (the empty ones, then those of the
    /// residents the prediction leaves out) while its attention runs (the model's predictor pass hands it
    /// over, `Hook.readAheadSeed`, in every prompt pass), landed at the routing barrier before the seed.
    /// Exact: it chooses reads and rows only. Needs `seed`.
    read_ahead: bool = false,
    /// P1b: with `defer_base` and the seed, the deferred base call runs once the seed's groups are read (after the first
    /// group with a transient row), so the later groups' reads overlap its drain; later base rows run in a last call.
    /// Exact: the same rows, slots and kernels; only the call's place in the schedule changes.
    base_at_seed: bool = false,
    /// With `base_at_seed`: the seed's ranks grouped apart from the stream's (the last seed group holds no stream
    /// record) and the base call issued after the last seed group, so only the seed's own reads precede it. Exact:
    /// each expert's rows and kernel are its own whatever its group; the combine folds by routed position.
    seed_aligned: bool = false,
    /// The route-plan cut (glm5next/route-plan, cd6f1ec): the call's rows bucketed by group once (a stable counting sort
    /// into the join's scratch, `bucketRowsByGroup`), each group's bank passes over its own rows only, in the same order
    /// (byte-identical plans), and the distinct experts with their counts in one pass. Off (the control): every bank
    /// pass of every group walks the call's every row. Exact: host planning only.
    route_cut: bool = false,
    /// turboexl3_glm53f (TURBOEXL3_GLM53F_PF_STAGE): a prompt call of at least 8 rows per expert (nearly every expert
    /// routed) routes its layer's misses before its routing barrier: as many as `depth` groups hold, in expert order,
    /// into the transient windows with every resident held (no staged miss evicts one: `holdResidents`), so their reads
    /// run while the GPU runs the layer's attention, which the barrier waits for. The call's groups are those staged
    /// groups first (their experts the call does not route stay in them, with no rows), then its other experts hottest
    /// first, every group on the stock route (no P7 plan: a staged route is live before the call's order is known). Exact:
    /// the same kernels over the same rows; only which rows hold a record, and when it is read, change. Needs `seed` and
    /// `hot_first`.
    stage: bool = false,
    /// turboexl3_glm53f (TURBOEXL3_GLM53F_PF_DRAIN_LAG): with `seed`, a group's one drain, its slots' release and the
    /// route `depth` groups on wait for the next group's first committed chunk (the math's `LagPending`, whose hook the
    /// math's prompt call runs at that commit), or for the next group's banks when it commits none, in place of running
    /// before the next group is encoded: the GPU runs the group's last chunks while the host builds the next group's
    /// first. Exact: the same kernels over the same rows in the same order on the stream; a group's slots go back only
    /// after its own drain, and in group order. Needs `seed`, `depth` 2 or more (the next group's route was made before),
    /// no `defer_base` or cold rows, and a math with `LagPending`.
    drain_lag: bool = false,
    pub const max_cold_rows = 8;
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

pub fn Experts(comptime G: type, comptime S: type, comptime M: type) type {
    return ExpertsWith(G, S, M, .{});
}

pub fn ExpertsWith(comptime G: type, comptime S: type, comptime M: type, comptime routes: Routes) type {
    const Projection = if (@hasDecl(M, "Projection")) M.Projection else ProjOf;
    const EventProjection = if (@hasDecl(M, "EventProjection")) M.EventProjection(G.T) else Exl3EventProjection(G.T);
    const wide_event_gates = routes.gated and (if (@hasDecl(M, "wide_event_gates")) M.wide_event_gates else false);
    const predictor_decode_only = if (@hasDecl(M, "predictor_decode_only")) M.predictor_decode_only else false;
    const wide_profile = if (@hasDecl(M, "Profile")) M.Profile else prof;
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
        /// The decode lane's wave stages, bound at construction (`Options.banked`): one group per bank (the stock route)
        /// or one banked group over every bank.
        gate_up_wave: *const GateUpWaveFn = gateUpWaveGrouped,
        down_wave: *const DownWaveFn = downWaveGrouped,
        /// Where a decode call commits the caller's hoist, bound at construction (`Options.hoist_first`): behind the
        /// hit wave (the stock route: `pre_barrier` commits nothing, `post_hit` commits the hoist), or right behind
        /// the routing barrier's arrays, before the host waits on them (HOIST_FIRST: `pre_barrier` commits both,
        /// `post_hit` nothing), so the GPU runs the hoist while the host wakes, routes and encodes the hit wave.
        pre_barrier: *const PreBarrierFn = preBarrierNone,
        post_hit: *const PostHitFn = postHitHoist,
        /// DEVROUTE (`Options.devroute`): per layer two resident LUTs (u32 [n_experts], host arrays written in place, kept),
        /// the one each layer's next decode call reads (`dev_parity`): a call reads one, its route's result is written
        /// into the other, so the host never writes a LUT a committed wave may still read. Empty without the route.
        devroute: bool = false,
        dev_lut: [][2]T = &.{},
        dev_parity: []u1 = &.{},
        /// A gated scheduler's decode call's miss parts, bound at construction (`bindMergeGateWaits`): each projection of
        /// each bank waited on its own (the stock route), or every array of one gate value in one wait (MERGE_GATE_WAITS).
        gated_parts: if (routes.gated) *const GatedPartsFn else void = if (routes.gated) gatedPartsStock else {},

        const GateUpWaveFn = fn (self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave;
        const GatedPartsFn = fn (self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void;
        const DownWaveFn = fn (self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T;
        const PreBarrierFn = fn (g: *G, barrier: []const T, hoist: []const T) anyerror!void;
        const PostHitFn = fn (g: *G, hoist: []const T) anyerror!void;

        /// `banked`: the math's banked route (its `has_banked`; the quant installed it) runs every wave as one group.
        /// `hoist_first`: HOIST_FIRST (exact: the same evals, only their commit order moves) commits a decode call's
        /// hoist right behind its routing barrier's arrays, before the barrier's wait, instead of behind the hit wave.
        /// `devroute`: DEVROUTE (exact): in decode, each call's hit wave is a device graph (the routed ids through the
        /// layer's resident LUT into the banked texts, every routed pair) committed behind the barrier's arrays, before the
        /// host's wait and route; the host then routes and builds only the miss parts; the join takes every hit pair from
        /// the device wave and every miss pair from its part. Needs the banked route.
        pub const Options = struct { gates: []const Gate = &.{}, event: ?expert_event.Event = null, wide: Wide = .{}, banked: bool = false, hoist_first: bool = false, devroute: bool = false };

        /// The wide lane's host scratch, reused across calls.
        const WideScratch = struct {
            ids: std.ArrayList(u16) = .empty,
            /// Per expert: its index in `distinct`, or -1.
            first: std.ArrayList(i32) = .empty,
            distinct: std.ArrayList(u16) = .empty,
            /// Per expert: its rows in the call (the feed's order).
            count: std.ArrayList(u32) = .empty,
            cold_slot: std.ArrayList(u32) = .empty,
            cold_act: std.ArrayList(u32) = .empty,
            cold_pos: std.ArrayList(u32) = .empty,
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
            /// The deferred base-bank rows (`Wide.defer_base`): slots, act rows, routed positions.
            def_slot: std.ArrayList(u32) = .empty,
            def_act: std.ArrayList(u32) = .empty,
            def_pos: std.ArrayList(u32) = .empty,
            /// `Wide.route_cut`: the call's rows by group (`groupRows`; the rows themselves in `inv`, free until the join):
            /// per expert its group, the groups' row offsets.
            grp_of: std.ArrayList(u16) = .empty,
            grp_off: std.ArrayList(u32) = .empty,
            /// `Wide.stage`: the misses staged before the barrier, in their groups' order.
            staged: std.ArrayList(u16) = .empty,

            fn deinit(w: *WideScratch, a: std.mem.Allocator) void {
                inline for (.{ &w.ids, &w.first, &w.distinct, &w.count, &w.cold_slot, &w.cold_act, &w.cold_pos, &w.call_pos, &w.wave_pos, &w.slot, &w.act_row, &w.pos, &w.inv, &w.kept, &w.loc, &w.def_slot, &w.def_act, &w.def_pos, &w.grp_of, &w.grp_off, &w.staged }) |l| l.deinit(a);
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
            if ((wr.seed or wr.hot_first or wr.depth > 1 or wr.cold_rows > 0) and !routes.prefill) return error.InvalidWideRoute;
            if (wr.cold_rows > Wide.max_cold_rows) return error.InvalidWideRoute;
            if (wr.seed and comptime !@hasDecl(S, "seedPrefill")) return error.InvalidWideRoute;
            if (wr.defer_base and (wr.cold_rows > 0 or !routes.prefill or comptime !@hasDecl(S, "holdBase"))) return error.InvalidWideRoute;
            if (wr.read_ahead and (!wr.seed or comptime !@hasDecl(S, "readAheadSeed"))) return error.InvalidWideRoute;
            if (wr.base_at_seed and (!wr.seed or !wr.defer_base)) return error.InvalidWideRoute;
            if (wr.seed_aligned and (!wr.base_at_seed or !wr.hot_first or comptime !@hasDecl(S, "seedRanks"))) return error.InvalidWideRoute;
            if (wr.stage and (!wr.seed or !wr.hot_first or wr.defer_base or wr.seed_aligned or comptime !(@hasDecl(S, "stageMisses") and @hasDecl(S, "holdResidents")))) return error.InvalidWideRoute;
            if (wr.drain_lag and (!wr.seed or wr.depth < 2 or wr.defer_base or wr.cold_rows > 0 or comptime !@hasDecl(M, "LagPending"))) return error.InvalidWideRoute;
            if (opt.banked and comptime !hasBanked()) return error.BankedNotInMath;
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
            if (comptime hasBanked()) {
                if (opt.banked) {
                    self.gate_up_wave = gateUpWaveBanked;
                    self.down_wave = downWaveBanked;
                }
            }
            if (opt.hoist_first) {
                self.pre_barrier = preBarrierHoist;
                self.post_hit = postHitNone;
            }
            if (opt.devroute) {
                if (!opt.banked) return error.DevrouteNeedsBanked;
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
            if (comptime !hasBanked()) {
                return error.DevrouteNeedsBanked;
            } else return self.devWaveIn(g, layer, xf, indices, n, k);
        }

        fn devWaveIn(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            const nk: c_int = @intCast(n * k);
            const lut = self.dev_lut[layer][self.dev_parity[layer]];
            const ids = try g.take(lut, try g.reshape(indices, &.{nk}), 0);
            var tok: [max_route_ids]i32 = undefined;
            for (tok[0 .. n * k], 0..) |*t, p| t.* = @intCast(p / k);
            const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0 .. n * k]), &.{nk}, .int32), 0);
            const banks = try self.bankedArrays(layer, null);
            return self.math.downBanked(g, try self.math.gateUpBanked(g, x, ids, &banks), ids, &banks);
        }

        /// The stock route: the barrier's wait commits its own arrays; nothing is committed ahead of it.
        fn preBarrierNone(_: *G, _: []const T, _: []const T) !void {}

        /// HOIST_FIRST: the barrier's arrays in one commit (the event the wait reads signals at its end), then the hoist
        /// in its own commit behind it, so the wait returns at the router's end while the GPU runs the hoist. A call
        /// with no hoist commits the barrier's arrays alone (the wait that follows is then only a wait).
        fn preBarrierHoist(g: *G, barrier: []const T, hoist: []const T) !void {
            try g.asyncEval(barrier);
            if (hoist.len > 0) try g.asyncEval(hoist);
        }

        /// The stock route: the hoist is committed behind the hit wave, ahead of every miss wave.
        fn postHitHoist(g: *G, hoist: []const T) !void {
            if (hoist.len > 0) try g.asyncEval(hoist);
        }

        /// HOIST_FIRST: the hoist was committed before the barrier's wait.
        fn postHitNone(_: *G, _: []const T) !void {}

        /// HOIST_FIRST as bound at construction.
        pub fn hoistFirst(self: *const Self) bool {
            return self.pre_barrier == &preBarrierHoist;
        }

        fn hasBanked() bool {
            return @hasDecl(M, "has_banked") and M.has_banked;
        }

        /// The router that scores layer `layer`'s read-ahead: the NEXT layer's
        /// (none after the last layer).
        pub fn predictorGate(self: *const Self, layer: u32) ?Gate {
            return if (layer + 1 < self.gates.len) self.gates[layer + 1] else null;
        }
        /// turboexl3_glm53f: READ_AHEAD3: model layer `layer`'s own predictor scores on `xf` (`scoresOf` with its router,
        /// the one the lookahead scores for it on the call before); null without the lookahead's gates.
        pub fn firstScores(self: *const Self, g: *G, xf: T, layer: u32) !?T {
            if (layer >= self.gates.len) return null;
            return try self.scoresOf(g, xf, self.gates[layer]);
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

        pub fn devRoute(self: *const Self) bool {
            return self.devroute;
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

            pub fn routed(h: Hook, g: *G, xf: T, indices: T) !T {
                return h.ex.run(g, h.layer, xf, indices, &.{});
            }

            /// `routed`, with the caller's arrays that do not wait on this call (`hoist`: the layer's
            /// shared expert and HC tail) committed once its reads are issued and its hit wave
            /// started: they run during the read wait, not in the next routing barrier.
            pub fn routedHoist(h: Hook, g: *G, xf: T, indices: T, hoist: []const T) !T {
                return h.ex.run(g, h.layer, xf, indices, hoist);
            }

            /// `routedHoist` ending before the join's take (JOIN_INDEXED, a decode-lane call): the concatenated outputs and
            /// each routed position's row in them, for a combine that reads the rows in place.
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

            /// (profile builds) The last wide call's JOINLESS merge; zeros elsewhere.
            pub fn lastMerge(h: Hook) MergeStats {
                return if (comptime wide_profile.enabled) h.ex.last_merge else .{};
            }

            /// P1 (`Wide.read_ahead`, fixed at construction): the model runs its predictor pass for this layer.
            pub fn readAhead(h: Hook) bool {
                return h.ex.wide_route.read_ahead;
            }

            /// The source is in its prompt phase (before its one phase change): the forward is a prompt's, so its predictor
            /// pass has rows to read ahead into. A source without a phase API is always prompting.
            pub fn prompting(h: Hook) bool {
                return if (comptime @hasDecl(S, "decoding")) !h.ex.source.decoding() else true;
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

        /// The positions of wave `w`, grouped by bank in first-appearance order
        /// (`Exl3PackedOps.gate_up`'s groups), each group's gate/up + SwiGLU.
        fn gateUpWaveGrouped(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, w: u8, over: ?*const [n_banks]?Arrays) anyerror!Wave {
            var wave: Wave = .{};
            for (sv.waves, sv.refs, 0..) |wv, ref, pos| {
                if (wv != w) continue;
                const gi = for (wave.groups[0..wave.n], 0..) |gr, i| {
                    if (gr.bank == ref.bank) break i;
                } else blk: {
                    wave.groups[wave.n] = .{ .bank = ref.bank };
                    wave.n += 1;
                    break :blk wave.n - 1;
                };
                const gr = &wave.groups[gi];
                gr.pos[gr.n] = @intCast(pos);
                gr.n += 1;
            }
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
                var tok: [max_route_ids]i32 = undefined;
                var rows: [max_route_ids]u32 = undefined;
                for (gr.pos[0..gr.n], tok[0..gr.n], rows[0..gr.n]) |pos, *t, *r| {
                    t.* = @intCast(pos / k);
                    r.* = sv.refs[pos].row;
                }
                const n: c_int = @intCast(gr.n);
                const x = try g.take(xf, try g.hostArray(std.mem.sliceAsBytes(tok[0..gr.n]), &.{n}, .int32), 0);
                wave.ids[i] = try g.hostArray(std.mem.sliceAsBytes(rows[0..gr.n]), &.{n}, .uint32);
                wave.h[i] = try self.math.gateUp(g, x, wave.ids[i], arrays.gate, arrays.up);
            }
            return wave;
        }

        /// Each group's down; the outputs join the call's accumulator.
        fn downWaveGrouped(self: *Self, g: *G, layer: u32, wave: *const Wave, acc: *Acc, over: ?*const [n_banks]?Arrays) anyerror![]const T {
            const first = acc.n_outs;
            for (wave.groups[0..wave.n], 0..) |*gr, i| {
                const arrays = (if (over) |o| o[@backingInt(gr.bank)] else self.banks[layer][@backingInt(gr.bank)]).?;
                acc.outs[acc.n_outs] = try self.math.down(g, wave.h[i], wave.ids[i], arrays.down);
                acc.n_outs += 1;
                for (gr.pos[0..gr.n]) |pos| {
                    acc.pos[acc.n_pos] = pos;
                    acc.n_pos += 1;
                }
            }
            return acc.outs[first..acc.n_outs];
        }

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

        /// The gated waves on the route bound at construction (`gated_parts`).
        fn gatedParts(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) !void {
            return self.gated_parts(self, g, layer, xf, k, sv, gates, acc);
        }

        /// MERGE_GATE_WAITS bound (a decode call's gated miss parts wait once per gate value), once at construction,
        /// before any call. Needs the gated route.
        pub fn bindMergeGateWaits(self: *Self) !void {
            if (comptime routes.gated) {
                self.gated_parts = gatedPartsMerged;
            } else return error.MergeGateWaitsNeedsGated;
        }

        /// MERGE_GATE_WAITS as bound at construction.
        pub fn mergeGateWaits(self: *const Self) bool {
            return if (comptime routes.gated) self.gated_parts == &gatedPartsMerged else false;
        }

        /// The gated waves (the lookahead4 lane's event gate): every part's
        /// gate/up over bank arrays waited at `gu`, then part p's down over
        /// arrays waited at `down_first + p`, ordered after every gate/up and
        /// the previous part's down. Nothing waits on the host. Each projection
        /// of each bank waits on its own (the stock route).
        fn gatedPartsStock(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void {
            var gu: [n_banks]?Arrays = @splat(null);
            for (sv.waves, sv.refs) |w, ref| {
                const b = @backingInt(ref.bank);
                if (w == 0 or gu[b] != null) continue;
                const arrays = self.banks[layer][b].?;
                gu[b] = .{ .gate = try self.waitProj(g, arrays.gate, gates.gu, &.{}), .up = try self.waitProj(g, arrays.up, gates.gu, &.{}), .down = arrays.down };
            }
            var waves: [max_route_ids]Wave = undefined;
            var hs: [max_route_ids]T = undefined;
            var n_hs: usize = 0;
            for (0..sv.n_parts) |p| {
                waves[p] = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), &gu);
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
                for (&dn) |*d| if (d.*) |*arr| {
                    arr.down = try self.waitProj(g, arr.down, gates.down_first + p, deps[0 .. n_hs + prev.len]);
                };
                prev = try self.down_wave(self, g, layer, &waves[p], acc, &dn);
            }
        }

        /// MERGE_GATE_WAITS: `gatedPartsStock`'s waves over the same arrays, gates and deps, with one event wait per gate
        /// value: the gate and up arrays of every bank the call's misses read wait at `gu` together, then part p's down
        /// arrays of every such bank at `down_first + p` together (behind the same deps). The stock route waits each
        /// projection of each bank on its own at those same values. Exact: every array aliases the same buffer behind the
        /// same value; only the waits (each one pass-through primitive and one encoder split) are fewer.
        fn gatedPartsMerged(self: *Self, g: *G, layer: u32, xf: T, k: u32, sv: Served, gates: expert_stream.Gates, acc: *Acc) anyerror!void {
            const pc = EventProjection.count;
            var gu: [n_banks]?Arrays = @splat(null);
            // The banks the misses read, in the stock route's order (first appearance), each one's gate then up arrays.
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
                waves[p] = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), &gu);
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
                prev = try self.down_wave(self, g, layer, &waves[p], acc, &dn);
            }
        }

        /// `PackedDecode.run` (gate/up publish before down): the routing
        /// barrier, the residents' wave, then per miss part gate/up after
        /// `waitGu` and down after `waitDown`, each wave started on the GPU;
        /// release; the outputs in the router's order, `[n, k, hidden]` f32.
        /// A residency query over the source (before the call's route admits and evicts; profile builds).
        const ResidentAt = struct {
            src: *S,
            layer: u32,
            fn of(r: ResidentAt, e: u16) bool {
                return r.src.isResident(r.layer, e);
            }
        };

        /// The stream's read-busy gauge (ns), for the profile builds' timeline.
        fn readGauge(self: *const Self) u64 {
            return if (comptime @hasDecl(S, "readWallNs")) self.source.readWallNs() else 0;
        }

        pub fn run(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T) !T {
            return self.runEnd(g, layer, xf, indices, hoist, .take);
        }

        /// A decode-lane call's outputs before the join's take (`Hook.routedHoistIndexed`, JOIN_INDEXED: the caller's
        /// combine reads the rows in place): the concatenated outputs [rows, hidden] and each routed position's row in
        /// them, u32 [n * k] (`joinOrder`; the stock join takes exactly these rows).
        pub const Indexed = struct { rows: T, order: T };

        /// How a call ends: the join's take (`run`, every caller's route) or the indexed rows (JOIN_INDEXED).
        const Ending = enum { take, indexed };

        fn runEnd(self: *Self, g: *G, layer: u32, xf: T, indices: T, hoist: []const T, comptime ending: Ending) !(if (ending == .take) T else Indexed) {
            const n: u32 = @intCast(g.shapeOf(xf).dim(0));
            const k: u32 = @intCast(g.shapeOf(indices).dim(1));
            const n_ids = n * k;
            // A math with its own prefill token threshold routes by logical M.
            // DeepSeek keeps its decode / verify route through 48 top-6 ids.
            if (if (comptime @hasDecl(M, "prefill_min_tokens")) n >= M.prefill_min_tokens else n_ids > max_route_ids) {
                // Only a decode-width layer hoists (the trunk's small-rows segments).
                std.debug.assert(hoist.len == 0);
                if (comptime ending == .indexed) {
                    // The indexed join serves the decode lane only (a wide call joins in its combine already).
                    return error.IndexedJoinWideCall;
                } else if (comptime !routes.prefill) {
                    return error.PrefillLaneNotPorted;
                } else {
                    return self.runWide(g, layer, xf, indices, n, k);
                }
            }
            var id_buf: [max_route_ids]u16 = undefined;
            var score_buf: [expert_lookahead.max_rows * 512]f32 = undefined;
            var scores: []const f32 = &.{};
            var tt = dt.now();
            // A0 (profile builds): the call's barrier, route and MoE build (`first_cycle.call`).
            const t_call = tt;
            if (comptime timeline.enabled) timeline.point(layer, .call, self.readGauge(), false);
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
            // HOIST_FIRST commits them and the hoist here (the evals below then only wait); the stock route nothing.
            try self.pre_barrier(g, bar[0..n_bar], hoist);
            if (n_bar > 1) try g.evalAll(bar[0..n_bar]);
            if (sc) |s| scores = try g.hostF32(s, score_buf[0 .. n * self.n_experts]);
            const ids = try g.hostIds(indices, id_buf[0..n_ids]);
            tt = dt.charge(.barrier, tt);
            const t_barrier = tt;
            if (comptime timeline.enabled) timeline.point(layer, .barrier, 0, false);
            // A0 (profile builds): the read gauge at the barrier's end; in the first cycle the tail set's reads (before
            // the route plans); in the prompt the tail set itself (a decode-width prompt call).
            const wall_b: u64 = if (comptime first_cycle.enabled and @hasDecl(S, "readWallNs")) self.source.readWallNs() else 0;
            const tail_reads: u64 = if (comptime first_cycle.enabled and @hasDecl(S, "isResident")) first_cycle.tailReads(layer, ResidentAt{ .src = self.source, .layer = layer }, ResidentAt.of) else 0;
            if (comptime first_cycle.enabled) {
                first_cycle.tailRecord(layer, ids, n, k);
                tt = dt.now();
            }
            const t_route0 = tt;
            const call = try self.source.route(layer, ids, scores);
            const t_route = dt.charge(.route, tt);
            dt.countCall();
            if (comptime timeline.enabled) timeline.point(layer, .route, 0, false);
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
                const hits = try self.gate_up_wave(self, g, layer, xf, k, sv, 0, null);
                if (hits.n > 0) try g.asyncEval(try self.down_wave(self, g, layer, &hits, &acc, null));
            }
            // VERIFY_ENCODE hoist: behind the hit wave, ahead of every miss wave (HOIST_FIRST: committed before the barrier's wait).
            // turboexl3_glm53f: SHARED_HOIST: committed only when the call misses (a miss wave waits for its bytes); a call
            // with no miss part leaves it to the next barrier's evaluation, where the trunk's add reads it.
            try self.post_hit(g, if (sv.n_parts > 0) hoist else &.{});
            if (comptime timeline.enabled) timeline.point(layer, .hit, 0, false);
            if (routes.gated) {
                if (try self.source.gate(call)) |gates| {
                    if (comptime timeline.enabled) timeline.gate(layer, gates.gu, gates.down_first, gates.n_parts);
                    try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
                }
            } else for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                tt = dt.now();
                try self.source.waitGu(call, part);
                _ = dt.charge(.read_wait, tt);
                const wave = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), null);
                try g.asyncEval(wave.h[0..wave.n]);
                tt = dt.now();
                try self.source.waitDown(call, part);
                _ = dt.charge(.read_wait, tt);
                try g.asyncEval(try self.down_wave(self, g, layer, &wave, &acc, null));
            }
            self.source.release(call);
            released = true;
            // DEVROUTE: the layer's next call reads the policy after this route (the other LUT of the pair).
            if (dev != null) try self.devLutAfterRoute(g, layer);
            if (comptime timeline.enabled) timeline.point(layer, .end, self.readGauge(), sv.n_parts > 0);
            if (comptime first_cycle.enabled) first_cycle.call(layer, t_barrier - t_call, t_route - t_route0, dt.now() - t_route, ids, sv.waves, wall_b, tail_reads);
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

        /// `join` without its take: the concatenated outputs and the order the take would read them in (JOIN_INDEXED).
        fn joinIndexed(g: *G, acc: *const Acc, n: u32, k: u32) !Indexed {
            const joined = try g.concat(acc.outs[0..acc.n_outs], 0);
            var order: [max_route_ids]u32 = undefined;
            const n_ids = joinOrder(acc.pos[0..acc.n_pos], n * k, &order);
            const ord = try g.hostArray(std.mem.sliceAsBytes(order[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
            return .{ .rows = joined, .order = ord };
        }

        /// The self-check's joined row as the host reads it: a math with its own projection (bf16 joins) casts to
        /// f32; DeepSeek's EXL3 joins are f32 already and are read as they are.
        fn checkRow(g: *G, x: T) !T {
            return if (comptime @hasDecl(M, "Projection")) g.astype(x, .float32) else x;
        }

        /// `checkGated` on layer 0's first `top_k` experts and one fixed bf16 row.
        pub fn checkGates(self: *Self, g: *G, a: std.mem.Allocator, top_k: u32) !void {
            return self.checkGatesAt(g, a, top_k, 0);
        }

        /// `checkGated` on the first routed layer's first `top_k` experts (cold at construction: every one a miss)
        /// and one fixed bf16 row.
        pub fn checkGatesAt(self: *Self, g: *G, a: std.mem.Allocator, top_k: u32, first_layer: u32) !void {
            const hidden: usize = @intCast(self.hidden);
            var ids: [max_route_ids]u16 = undefined;
            for (ids[0..top_k], 0..) |*d, i| d.* = @intCast(i);
            const row = try a.alloc(u16, hidden);
            defer a.free(row);
            for (row, 0..) |*v, i| v.* = ops.bf16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 97)) - 48)) / 64.0);
            const kh = @as(usize, top_k) * hidden;
            const outs = try a.alloc(f32, 2 * kh);
            defer a.free(outs);
            const m = g.mark();
            defer g.resetTo(m);
            const xf = try g.hostArray(std.mem.sliceAsBytes(row), &.{ 1, self.hidden }, .bfloat16);
            try self.checkGated(g, first_layer, xf, ids[0..top_k], outs[0..kh], outs[kh..]);
        }

        /// P1's construction self-check (`Wide.read_ahead`), once on the real stream before any request:
        /// layer 0's records read ahead equal their demand reads, bit for bit, over two pool jobs.
        /// Returns how many records were checked.
        pub fn checkReadAhead(self: *Self) !u32 {
            if (comptime !@hasDecl(S, "checkReadAhead")) return error.ReadAheadNotStreamed;
            return self.source.checkReadAhead(0, expert_io.max_items + 1);
        }

        /// LOOKAHEAD4's construction check, once on the real stream before any request: one call of
        /// `ids` (one row `xf`) over `layer`, whose misses' waves wait on event gates and are
        /// evaluated (the GPU held until the pool publishes their bytes), then the same slots' waves
        /// with no gate after the bytes landed (the host-waits arm's). The two outputs must match bit
        /// for bit (`gated` / `waited`: `k * hidden` f32 each) and the pool must have forced no gate.
        pub fn checkGated(self: *Self, g: *G, layer: u32, xf: T, ids: []const u16, gated: []f32, waited: []f32) !void {
            if (comptime !routes.gated) @compileError("checkGated needs the gated route");
            const k: u32 = @intCast(ids.len);
            const m = g.mark();
            defer g.resetTo(m);
            const call = try self.source.route(layer, ids, &.{});
            var released = false;
            errdefer if (!released) self.source.release(call);
            const sv = self.source.served(call);
            const gates = (try self.source.gate(call)) orelse return error.GateCheckNoMiss;
            var acc: Acc = .{};
            const hits = try self.gate_up_wave(self, g, layer, xf, k, sv, 0, null);
            if (hits.n > 0) _ = try self.down_wave(self, g, layer, &hits, &acc, null);
            try self.gatedParts(g, layer, xf, k, sv, gates, &acc);
            _ = try g.hostF32(try checkRow(g, try self.join(g, &acc, 1, k)), gated);
            var ref: Acc = .{};
            if (hits.n > 0) _ = try self.down_wave(self, g, layer, &hits, &ref, null);
            for (0..sv.n_parts) |p| {
                const part: u32 = @intCast(p);
                try self.source.waitGu(call, part);
                const wave = try self.gate_up_wave(self, g, layer, xf, k, sv, @intCast(p + 1), null);
                try self.source.waitDown(call, part);
                _ = try self.down_wave(self, g, layer, &wave, &ref, null);
            }
            _ = try g.hostF32(try checkRow(g, try self.join(g, &ref, 1, k)), waited);
            self.source.release(call);
            released = true;
            try self.source.flush();
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(gated), std.mem.sliceAsBytes(waited))) return error.GateCheckMismatch;
        }

        /// The wide lane: the routing barrier; the call's distinct experts (first
        /// appearance) routed in groups of at most `max_route_ids`; per group,
        /// every part waited, then per bank the group's rows through the layer's
        /// DIG-X route (slot = the row the source serves the expert in, act row =
        /// its token), drained and evaluated before the group is released (the
        /// next route may refill those slots); the outputs joined in the
        /// router's order, `[n, k, hidden]` f32. The DIG kernels read bf16
        /// activations (the lane of record's MoE input): another dtype is
        /// rounded to bf16 once, here. `Options.wide.seed` seeds the layer from
        /// the call and drains each group once; `.hot_first` orders the groups hottest first;
        /// `Options.wide.depth` 2 routes group g + 1 (its reads) before group g's waves.
        fn runWide(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !T {
            try self.runWideCore(g, layer, xf, indices, n, k, false);
            const a = self.a;
            const w = &self.wide;
            const n_ids = n * k;
            // `take(concatenate(outputs), argsort(positions))`, the permutation made on the host.
            const joined = try g.concat(w.kept.items, 0);
            for (w.kept.items) |x| g.release(x);
            w.kept.clearRetainingCapacity();
            try w.inv.resize(a, n_ids);
            invertPositions(w.pos.items, w.inv.items);
            const ord = try g.hostArray(std.mem.sliceAsBytes(w.inv.items), &.{@intCast(n_ids)}, .uint32);
            return g.reshape(try g.take(joined, ord, 0), &.{ @intCast(n), @intCast(k), self.hidden });
        }

        /// JOINLESS: the wide call's outputs unjoined (above `max_parts`, the smallest merged into one source:
        /// `mergeJoinless`) and each assignment's (source, row) as int32 [n, k, 2]; the combine reads the
        /// rows in place. The sources stay kept until `releaseParts`.
        pub const max_parts = joinless_sources;
        pub const Parts = struct { outs: []const T, loc: T };

        fn runWideParts(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32) !Parts {
            // The math's unjoined prefill when it has one: the combine reads the waves' own outputs.
            try self.runWideCore(g, layer, xf, indices, n, k, comptime @hasDecl(M, "has_parts") and M.has_parts);
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

        /// A wide call's groups over its experts hottest first: the seed's ranks [0, sa) chunked apart from the stream's
        /// (`Wide.seed_aligned`; sa 0: one run of `max_route_ids` chunks).
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

        /// The deferred base-bank rows gathered so far as one call's waves, drained; their slots stay held. `at_seed`:
        /// P1b's call at the seed (the profile's split charges it apart from the end call's).
        fn deferredBase(self: *Self, g: *G, layer: u32, act: T, comptime parts: bool, at_seed: bool) !void {
            const a = self.a;
            const w = &self.wide;
            if (w.def_slot.items.len == 0) return;
            wide_profile.setMode(if (at_seed) .base_seed else .base_end);
            defer wide_profile.setMode(.stream);
            const kb = w.kept.items.len;
            const b = self.banks[layer][@backingInt(BankKind.base)].?;
            const rows_: quant.PrefillRows = .{ .slot = w.def_slot.items, .act_row = w.def_act.items };
            if (parts) {
                w.wave_pos.clearRetainingCapacity();
                try self.math.prefillParts(g, layer, act, rows_, b, a, &w.kept, &w.wave_pos);
                for (w.wave_pos.items) |p| try w.pos.append(a, w.def_pos.items[p]);
            } else {
                try w.kept.ensureUnusedCapacity(a, 1);
                w.kept.appendAssumeCapacity(try self.math.prefill(g, layer, act, rows_, b));
                try w.pos.appendSlice(a, w.def_pos.items);
            }
            try self.math.finishPrefill(g);
            const td = wide_profile.now();
            try g.evalAll(w.kept.items[kb..]);
            wide_profile.charge(.drain, td);
            w.def_slot.clearRetainingCapacity();
            w.def_act.clearRetainingCapacity();
            w.def_pos.clearRetainingCapacity();
        }

        /// `Wide.route_cut`: the call's rows bucketed by group (`bucketRowsByGroup`) into `inv` [n_ids] (the join's scratch,
        /// unused until the call's groups are done: no new per-row memory), group `gi`'s at `grp_off[gi]..grp_off[gi + 1]`.
        /// A group's bank passes then read its own rows only. `counted`: `count` holds each expert's rows.
        fn groupRows(a: std.mem.Allocator, w: *WideScratch, groups: Groups, n_groups: usize, counted: bool) !void {
            try w.grp_of.resize(a, w.first.items.len);
            for (groups.d, 0..) |e, i| w.grp_of.items[e] = @intCast(groups.index(i));
            try w.grp_off.resize(a, n_groups + 1);
            try w.inv.resize(a, w.ids.items.len);
            bucketRowsByGroup(w.ids.items, groups.d, w.grp_of.items, if (counted) w.count.items else null, w.grp_off.items, w.inv.items);
        }

        /// The construction's route (`Wide.route_cut`), once per call: the core at its comptime instance.
        fn runWideCore(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32, comptime parts: bool) !void {
            return switch (self.wide_route.route_cut) {
                inline else => |cut| self.runWideCoreAt(g, layer, xf, indices, n, k, parts, cut),
            };
        }

        fn runWideCoreAt(self: *Self, g: *G, layer: u32, xf: T, indices: T, n: u32, k: u32, comptime parts: bool, comptime cut: bool) !void {
            const a = self.a;
            const w = &self.wide;
            wide_profile.countWideCall();
            const n_ids = n * k;
            // The wide lane takes prefill-width calls only (the prefill texts bind small inputs as
            // `constant`; a decode-width call is the decode lane's).
            if (comptime !@hasDecl(M, "prefill_min_tokens")) std.debug.assert(n_ids >= wide_min_ids);
            const feed = self.wide_route.seed;
            const hot_first = self.wide_route.hot_first;
            const depth: usize = self.wide_route.depth;
            const cold: u32 = self.wide_route.cold_rows;
            const cold_chunk: usize = if (@hasDecl(M, "max_decode_rows")) M.max_decode_rows else max_route_ids;
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
            // A0 (profile builds): the prompt's last rows' experts.
            if (comptime first_cycle.enabled) first_cycle.tailRecord(layer, w.ids.items, n, k);
            try w.first.resize(a, self.n_experts);
            @memset(w.first.items, -1);
            w.distinct.clearRetainingCapacity();
            const counted = feed or hot_first or cold > 0;
            if (comptime cut) {
                // The call's experts in first appearance and (the feed's order, the cold split) their rows: one pass.
                try w.distinct.ensureTotalCapacity(a, self.n_experts);
                if (counted) {
                    try w.count.resize(a, self.n_experts);
                    @memset(w.count.items, 0);
                    for (w.ids.items) |e| {
                        if (w.first.items[e] < 0) {
                            w.first.items[e] = @intCast(w.distinct.items.len);
                            w.distinct.appendAssumeCapacity(e);
                        }
                        w.count.items[e] += 1;
                    }
                } else for (w.ids.items) |e| if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    w.distinct.appendAssumeCapacity(e);
                };
            } else {
                for (w.ids.items) |e| if (w.first.items[e] < 0) {
                    w.first.items[e] = @intCast(w.distinct.items.len);
                    try w.distinct.append(a, e);
                };
                if (counted) {
                    try w.count.resize(a, self.n_experts);
                    @memset(w.count.items, 0);
                    for (w.ids.items) |e| w.count.items[e] += 1;
                }
            }
            // The call's residency seed (the seed route), then its experts hottest first (the order route).
            if (feed) {
                // P1: the layer's read-ahead lands before its seed (which protects the ones it chooses).
                if (comptime @hasDecl(S, "awaitReadAhead")) if (self.wide_route.read_ahead) try self.source.awaitReadAhead(layer);
                if (comptime @hasDecl(S, "seedPrefill")) try self.source.seedPrefill(layer, w.ids.items) else unreachable;
            }
            if (hot_first) {
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
                } else if (comptime @hasDecl(S, "hotOrder")) self.source.hotOrder(layer, w.distinct.items, w.count.items) else std.sort.pdq(u16, w.distinct.items, @as([]const u32, w.count.items), struct {
                    fn lt(c: []const u32, x: u16, y: u16) bool {
                        return if (c[x] != c[y]) c[x] > c[y] else x < y;
                    }
                }.lt);
                for (w.distinct.items, 0..) |e, i| w.first.items[e] = @intCast(i);
            }
            const act = if (g.dtypeOf(xf) == .bfloat16) xf else try g.astype(xf, .bfloat16);
            w.pos.clearRetainingCapacity();
            w.kept.clearRetainingCapacity();
            const defer_base = self.wide_route.defer_base;
            w.def_slot.clearRetainingCapacity();
            w.def_act.clearRetainingCapacity();
            w.def_pos.clearRetainingCapacity();
            const n_distinct = w.distinct.items.len;
            // Seed-aligned: the seed's ranks [0, sa) in groups of their own, the stream's after them (sa 0: one run).
            const aligned = self.wide_route.seed_aligned;
            const sa: usize = if (staged_n > 0) staged_n else if (comptime @hasDecl(S, "seedRanks")) (if (aligned) @min(self.source.seedRanks(layer), n_distinct) else 0) else 0;
            const groups: Groups = .{ .d = w.distinct.items, .sa = sa, .seed_groups = (sa + max_route_ids - 1) / max_route_ids };
            const n_groups = groups.seed_groups + (n_distinct - sa + max_route_ids - 1) / max_route_ids;
            if (comptime cut) try groupRows(a, w, groups, n_groups, counted);
            // P1b: the seed's base call is due after its last group (aligned: the last seed group; else the first group
            // with a transient row).
            var base_due = self.wide_route.base_at_seed;
            // The live groups' calls, by group index mod `depth` (read ahead up to `depth` groups).
            var calls: [expert_stream.max_wide_depth]?*S.Call = @splat(null);
            errdefer {
                for (&calls) |*c| if (c.*) |cl| {
                    self.source.release(cl);
                    c.* = null;
                };
                for (w.kept.items) |x| g.release(x);
                w.kept.clearRetainingCapacity();
                if (comptime @hasDecl(S, "holdBase")) if (defer_base) self.source.releaseHeld();
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
            const lag = feed and self.wide_route.drain_lag;
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
                const group = groups.of(gi);
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
                var streamed = false;
                for ([_]BankKind{ .base, .ext, .transient }) |kind| {
                    const kb = w.kept.items.len;
                    w.slot.clearRetainingCapacity();
                    w.act_row.clearRetainingCapacity();
                    w.cold_slot.clearRetainingCapacity();
                    w.cold_act.clearRetainingCapacity();
                    w.cold_pos.clearRetainingCapacity();
                    // The cut: the group's own rows, in row order (`groupRows`), each bank's rows as a pass over the whole
                    // call meets them; the control: every row of the call, the group's kept.
                    const passed: []const u32 = if (comptime cut) w.inv.items[w.grp_off.items[gi]..w.grp_off.items[gi + 1]] else &.{};
                    const n_pass = if (comptime cut) passed.len else w.ids.items.len;
                    for (0..n_pass) |pi| {
                        const row: usize = if (comptime cut) passed[pi] else pi;
                        const e = w.ids.items[row];
                        const fi: usize = @intCast(w.first.items[e]);
                        if (comptime !cut) if (fi < start or fi >= start + group.len) continue;
                        const ref = sv.refs[fi - start];
                        if (ref.bank != kind) continue;
                        const act_row: u32 = @intCast(row / @as(usize, k));
                        if (defer_base and kind == .base) {
                            try w.def_slot.append(a, ref.row);
                            try w.def_act.append(a, act_row);
                            try w.def_pos.append(a, @intCast(row));
                            continue;
                        }
                        if (cold > 0 and w.count.items[e] <= cold) {
                            try w.cold_slot.append(a, ref.row);
                            try w.cold_act.append(a, act_row);
                            try w.cold_pos.append(a, @intCast(row));
                            continue;
                        }
                        try w.slot.append(a, ref.row);
                        try w.act_row.append(a, act_row);
                        if (parts) try w.call_pos.append(a, @intCast(row)) else try w.pos.append(a, @intCast(row));
                    }
                    if (w.slot.items.len == 0 and w.cold_slot.items.len == 0) continue;
                    streamed = streamed or kind == .transient;
                    var b = self.banks[layer][@backingInt(kind)].?;
                    if (comptime wide_event_gates) if (ready) |value| {
                        b = .{
                            .gate = try self.waitProj(g, b.gate, value, &.{}),
                            .up = try self.waitProj(g, b.up, value, &.{}),
                            .down = try self.waitProj(g, b.down, value, &.{}),
                        };
                    };
                    const hot = w.slot.items.len > 0;
                    if (hot) {
                        if (parts) {
                            // The waves' outputs as they are; each wave-ordered row's routed position.
                            w.wave_pos.clearRetainingCapacity();
                            try self.math.prefillParts(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b, a, &w.kept, &w.wave_pos);
                            for (w.wave_pos.items) |p| try w.pos.append(a, w.call_pos.items[p]);
                            w.call_pos.clearRetainingCapacity();
                        } else {
                            try w.kept.ensureUnusedCapacity(a, 1);
                            const y = try self.math.prefill(g, layer, act, .{ .slot = w.slot.items, .act_row = w.act_row.items }, b);
                            w.kept.appendAssumeCapacity(y);
                        }
                    }
                    // Cold rows: the decode GEMV over their slots, encoded after the bank's wide waves
                    // (the act-row index is built right before its take).
                    var c0: usize = 0;
                    while (c0 < w.cold_slot.items.len) : (c0 += cold_chunk) {
                        const c1 = @min(c0 + cold_chunk, w.cold_slot.items.len);
                        const m: c_int = @intCast(c1 - c0);
                        const ar = try g.hostArray(std.mem.sliceAsBytes(w.cold_act.items[c0..c1]), &.{m}, .uint32);
                        const xs = try g.take(act, ar, 0);
                        const sid = try g.hostArray(std.mem.sliceAsBytes(w.cold_slot.items[c0..c1]), &.{m}, .uint32);
                        const h = try self.math.gateUp(g, xs, sid, b.gate, b.up);
                        try w.kept.ensureUnusedCapacity(a, 1);
                        w.kept.appendAssumeCapacity(g.keep(try self.math.down(g, h, sid, b.down)));
                        try w.pos.appendSlice(a, w.cold_pos.items[c0..c1]);
                    }
                    if (!feed) {
                        if (hot) try self.math.finishPrefill(g);
                        tp = wide_profile.now();
                        try g.evalAll(w.kept.items[kb..]);
                        wide_profile.charge(.drain, tp);
                    }
                }
                // Feed: the group's banks queued back to back, one drain before its slots go back.
                if (feed) {
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
                }
                if (!lag) {
                    tp = wide_profile.now();
                    // The group's base-bank slots stay pinned and held for the deferred call past its release.
                    if (comptime @hasDecl(S, "holdBase")) if (defer_base) try self.source.holdBase(call);
                    self.source.release(call);
                    calls[gi % depth] = null;
                    if (gi + depth < n_groups) calls[gi % depth] = try self.source.route(layer, groups.of(gi + depth), &.{});
                    wide_profile.charge(.route, tp);
                }
                // P1b: the seed has landed (its groups read, the next routes' reads posted): its base call drains now.
                if (base_due and (if (aligned) gi + 1 == groups.seed_groups else streamed)) {
                    base_due = false;
                    try self.deferredBase(g, layer, act, parts, true);
                }
            }
            // turboexl3_glm53f: `Wide.drain_lag`: the call's last group drained and released
            if (lag) try pending.flush();
            // The deferred base-bank rows: one call's waves over every group's (with P1b, the later ones'), then the
            // slots let go.
            if (comptime @hasDecl(S, "holdBase")) if (defer_base) {
                try self.deferredBase(g, layer, act, parts, false);
                self.source.releaseHeld();
            };
        }
    };
}

// ── Tests ──

const testing = std.testing;
const TraceOps = ops.TraceOps;

comptime {
    assertSource(StreamSource);
    assertSource(FakeSource);
}

/// Scripted host reads for the trace backend: the routed ids of each call.
const Script = struct {
    calls: []const []const u16,
    next: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }

    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *Script = @ptrCast(@alignCast(ctx));
        if (self.next >= self.calls.len) return error.ScriptExhausted;
        const c = self.calls[self.next];
        self.next += 1;
        if (c.len != out.len) return error.ScriptShape;
        @memcpy(out, c);
    }

    fn argmax(_: *anyopaque) anyerror!u32 {
        return error.NoPicks;
    }
};

fn kindsOf(log: []const FakeSource.Event, buf: []u8) []const u8 {
    for (log, buf[0..log.len]) |e, *c| c.* = switch (e.kind) {
        .route => 'R',
        .wait_gu => 'g',
        .wait_down => 'd',
        .release => 'r',
        .flush => 'f',
        .grow => 'G',
        .gate => 'E',
        .read_ahead => 'A',
        .await_read_ahead => 'W',
    };
    return buf[0..log.len];
}

/// Distinct banks among the positions of wave `w` (the groups the wave runs).
fn groupsOf(sv: Served, w: u8) usize {
    var seen: [n_banks]bool = @splat(false);
    var n: usize = 0;
    for (sv.waves, sv.refs) |wv, ref| {
        if (wv != w or seen[@backingInt(ref.bank)]) continue;
        seen[@backingInt(ref.bank)] = true;
        n += 1;
    }
    return n;
}

/// Trace nodes of kind `op` between consecutive log events: [0] before the
/// first event, [j] between events j - 1 and j, [len] after the last.
fn opsBetween(g: *const TraceOps, log: []const FakeSource.Event, first: usize, op: ops.Op, out: []usize) void {
    @memset(out, 0);
    for (g.nodes.items[first..], first..) |nd, i| if (nd.op == op) {
        const j = for (log, 0..) |e, jj| {
            if (e.at > i) break jj;
        } else log.len;
        out[j] += 1;
    };
}

const TestConfig = struct { hidden_size: u32, moe_intermediate_size: u32, n_layers: u32, n_experts_per_tok: u32 = 6, swiglu_limit: f64 = 10.0, n_routed_experts: u32 = undefined };

fn testConfig(hidden: u32, inter: u32, n_layers: u8) TestConfig {
    return .{ .hidden_size = hidden, .moe_intermediate_size = inter, .n_layers = n_layers };
}

/// The wide lane's host tests: a registry from the embedded manifest (no MLX).
fn hostRegistry() !xk.Registry {
    var diag: xk.Diag = .{};
    return xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("dsv41 experts: {s}\n", .{diag.message()});
        return e;
    };
}

/// A wide-lane route that records every call: its rows, the bank it was
/// handed and, from the fake source's live route, the served slots of the
/// group (so the test can derive what each call should have been).
const RecRoute = struct {
    a: std.mem.Allocator,
    calls: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    /// The source whose live route each call snapshots.
    var source: ?*FakeSource = null;

    const Rec = struct { slot: []u32, act_row: []u32, bank_code: u32, route: u64, refs: []SlotRef, at: usize, node: usize, act_dtype: ops.Dtype };

    pub fn init(a: std.mem.Allocator) RecRoute {
        return .{ .a = a };
    }

    pub fn deinit(self: *RecRoute, _: *TraceOps) void {
        for (self.calls.items) |r| {
            self.a.free(r.slot);
            self.a.free(r.act_row);
            self.a.free(r.refs);
        }
        self.calls.deinit(self.a);
    }

    pub fn call(self: *RecRoute, g: *TraceOps, act: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const src = source.?;
        // A deferred base-bank call runs after its groups' routes are released: no refs to snapshot.
        const live_refs: []const SlotRef = for (&src.calls) |*c| {
            if (c.state == .live) break c.refs[0..c.plan.n_ids];
        } else &.{};
        try self.calls.append(self.a, .{
            .slot = try self.a.dupe(u32, rows.slot),
            .act_row = try self.a.dupe(u32, rows.act_row.?),
            .bank_code = bank.gate.code,
            .route = src.counters.route_calls,
            .refs = try self.a.dupe(SlotRef, live_refs),
            .at = src.log.items.len,
            .node = g.nodes.items.len,
            .act_dtype = g.dtypeOf(act),
        });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *RecRoute, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

/// A test-only profile hook proves the optional math hook receives the shared
/// schedule's charges without reading clocks or initializing MLX.
const SmallPrefillProfile = struct {
    pub const enabled = true;
    pub const Mode = prof.Mode;
    pub var wide_calls: u32 = 0;
    pub var charges: u32 = 0;
    pub fn now() u64 {
        return 0;
    }
    pub fn charge(_: sdk_ext.profile.PrefillBucket, _: u64) void {
        charges += 1;
    }
    pub fn setMode(_: Mode) void {}
    pub fn countWideCall() void {
        wide_calls += 1;
    }
};

const SmallPrefillMath = struct {
    const Chain = EagerChain(TraceOps, TraceGemv);
    pub const decode_forward_rows: u32 = 1;
    pub const prefill_min_tokens: u32 = 2;
    pub const wide_event_gates = true;
    pub const Profile = SmallPrefillProfile;
    d: Chain,
    route: *RecRoute,
    pub fn gateUp(self: *const SmallPrefillMath, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.d.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const SmallPrefillMath, g: *TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.d.down(g, h, ids, d);
    }
    pub fn prefill(self: *const SmallPrefillMath, g: *TraceOps, _: u32, x: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const mark = g.mark();
        const y = g.keep(try self.route.call(g, x, rows, bank));
        g.resetTo(mark);
        return y;
    }
    pub fn finishPrefill(self: *const SmallPrefillMath, g: *TraceOps) !void {
        return self.route.finish(g);
    }
};

/// dump_prefill_waves.route_rows: slot j repeated counts[j] times, then
/// Fisher-Yates from the end with j = splitmix64(seed) output % (i + 1).
fn sampleRows(a: std.mem.Allocator, seed: u64, slots: []const u32, counts: []const u32) ![]u16 {
    var n: usize = 0;
    for (counts) |c| n += c;
    const rows = try a.alloc(u16, n);
    var i: usize = 0;
    for (slots, counts) |s, c| for (0..c) |_| {
        rows[i] = @intCast(s);
        i += 1;
    };
    var st = seed;
    var j = n;
    while (j > 1) {
        j -= 1;
        const r: usize = @intCast(xk.splitmix64(&st) % (j + 1));
        std.mem.swap(u16, &rows[j], &rows[r]);
    }
    return rows;
}

const SynthBank = expert_bank.SynthBank;

/// Every routed id's slot row holds its record's bytes (the rows the hook bound).
fn expectRowsHold(s: *expert_stream.Stream, sb: *const SynthBank, layer: u32, ids: []const u16, sv: Served) !void {
    const geom = &sb.bank.layers[layer];
    const r = for (&s.routes) |*rr| {
        if (rr.state != .free and rr.layer == layer and rr.plan.n_ids == ids.len) break rr;
    } else return error.NoRoute;
    for (ids, r.plan.slotsOf(), sv.refs) |e, slot, ref| {
        try testing.expectEqual(s.slotRef(layer, slot), ref);
        const off = sb.bank.recordOffset(layer, e);
        for (geom.segments, 0..) |seg, comp| {
            try testing.expectEqualSlices(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(layer, slot, @fromBackingInt(@intCast(comp)))[0..seg.length]);
        }
    }
}

/// One route's two decode calls (as the VERIFY_ENCODE hoist test's) through a hook built with or without HOIST_FIRST:
/// the trace's nodes other than commits, the source's log, and per call the commits before its first host read and
/// in all.
const HoistRun = struct {
    ops: std.ArrayList(ops.Op) = .empty,
    shapes: std.ArrayList(ops.Shape) = .empty,
    kinds: [32]u8 = undefined,
    n_kinds: usize = 0,
    before_read: [2]usize = .{ 0, 0 },
    commits: [2]usize = .{ 0, 0 },
    bound: bool = false,

    fn deinit(r: *HoistRun, a: std.mem.Allocator) void {
        r.ops.deinit(a);
        r.shapes.deinit(a);
    }

    const Values = struct {
        calls: []const []const u16,
        next: usize = 0,
        fn values(self: *Values) TraceOps.HostValues {
            return .{ .ctx = self, .ids = ids, .argmax = argmax, .f32s = f32s };
        }
        fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *Values = @ptrCast(@alignCast(ctx));
            if (self.next >= self.calls.len) return error.ScriptExhausted;
            const c = self.calls[self.next];
            self.next += 1;
            if (c.len != out.len) return error.ScriptShape;
            @memcpy(out, c);
        }
        fn f32s(_: *anyopaque, out: []f32) anyerror!void {
            @memset(out, 0);
        }
        fn argmax(_: *anyopaque) anyerror!u32 {
            return error.NoPicks;
        }
    };

    fn run(comptime R: Routes, a: std.mem.Allocator, hoist_first: bool, out: *HoistRun) !void {
        var c = testConfig(256, 128, 2);
        c.n_routed_experts = 16;
        const Chain = EagerChain(TraceOps, TraceGemv);
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        if (R.lookahead) src.selector = try expert_lookahead.Selector.init(a, 16, 6, std.math.inf(f32), 2);
        var g = TraceOps.init(a);
        defer g.deinit();
        const Ex = ExpertsWith(TraceOps, FakeSource, Chain, R);
        var gates: [2]Ex.Gate = undefined;
        for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
        var ex = try Ex.initWith(a, &g, &src, Chain.init(.{}, &c), &c, .{ .gates = if (R.lookahead) &gates else &.{}, .hoist_first = hoist_first });
        defer ex.deinit();
        out.bound = ex.hoistFirst();
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        // Call 1: six misses (two parts); call 2: three hits, one part.
        var vals: Values = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
        g.host_values = vals.values();
        const xf = try g.input(&.{ 2, 256 }, .bfloat16);
        const idx = try g.input(&.{ 2, 3 }, .int32);
        const tail = [_]u32{ try g.input(&.{ 2, 256 }, .float32), try g.input(&.{ 2, 3 }, .float32) };
        const first = g.nodes.items.len;
        var starts: [3]usize = undefined;
        for (0..2) |call| {
            starts[call] = g.nodes.items.len;
            _ = try ex.at(0).routedHoist(&g, xf, idx, &tail);
        }
        starts[2] = g.nodes.items.len;
        for (0..2) |call| {
            var read_seen = false;
            for (g.nodes.items[starts[call]..starts[call + 1]]) |nd| switch (nd.op) {
                .async_eval => {
                    out.commits[call] += 1;
                    if (!read_seen) out.before_read[call] += 1;
                },
                .host_read => read_seen = true,
                else => {},
            };
        }
        for (g.nodes.items[first..]) |nd| if (nd.op != .async_eval) {
            try out.ops.append(a, nd.op);
            try out.shapes.append(a, nd.shape);
        };
        out.n_kinds = kindsOf(src.log.items, &out.kinds).len;
        try ex.flush();
    }
};

/// One routed call of the gated test scheduler (FakeSource, the eager chain), stock or MERGE_GATE_WAITS: its nodes, its
/// waits' aliases, the waits it built, its parts and the banks its misses read.
const MergeWaitsRun = struct {
    nodes: []TraceOps.Node = &.{},
    waits: []TraceOps.Wait = &.{},
    calls: usize = 0,
    parts: u32 = 0,
    banks: usize = 0,

    fn deinit(r: *MergeWaitsRun, a: std.mem.Allocator) void {
        a.free(r.nodes);
        a.free(r.waits);
    }

    fn run(a: std.mem.Allocator, merge: bool, ids: []const u16) !MergeWaitsRun {
        const c = testConfig(256, 128, 2);
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        var g = TraceOps.init(a);
        defer g.deinit();
        const Chain = EagerChain(TraceOps, TraceGemv);
        var ex = try ExpertsWith(TraceOps, FakeSource, Chain, .{ .gated = true }).init(a, &g, &src, Chain.init(.{}, &c), &c);
        defer ex.deinit();
        // Off unless bound; bound once, at construction.
        try testing.expect(!ex.mergeGateWaits());
        if (merge) try ex.bindMergeGateWaits();
        try testing.expectEqual(merge, ex.mergeGateWaits());
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        var script: Script = .{ .calls = &.{ids} };
        g.host_values = script.values();
        const first = g.nodes.items.len;
        const w0 = g.waits.items.len;
        const c0 = g.wait_calls;
        _ = try ex.at(0).routed(&g, try g.input(&.{ 2, 256 }, .bfloat16), try g.input(&.{ 2, 3 }, .int32));
        const sv = src.served(&src.calls[0]);
        var kinds: [n_banks]bool = @splat(false);
        for (sv.waves, sv.refs) |w, r| {
            if (w > 0) kinds[@backingInt(r.bank)] = true;
        }
        var out: MergeWaitsRun = .{ .calls = g.wait_calls - c0, .parts = sv.n_parts };
        for (kinds) |k| out.banks += @intFromBool(k);
        out.nodes = try a.dupe(TraceOps.Node, g.nodes.items[first..]);
        errdefer a.free(out.nodes);
        out.waits = try a.dupe(TraceOps.Wait, g.waits.items[w0..]);
        return out;
    }
};

/// A host-only architecture fixture whose router is already transposed and F32.
const StageBPredictorMath = struct {
    pub const predictor_decode_only = true;
    inner: TraceMath,
    pub fn predictorScores(g: *TraceOps, xf: u32, router: u32, correction: u32) !u32 {
        return g.add(try g.sigmoid(try g.matmul(try g.astype(xf, .float32), router)), correction);
    }
    pub fn gateUp(self: *const StageBPredictorMath, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.inner.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const StageBPredictorMath, g: *TraceOps, h: u32, ids: u32, bank: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, bank);
    }
};

/// A wide-lane route over a real Stream (trace backend) that names, per call row, the expert
/// whose record its slot holds (verified byte for byte) and its act row; and records the live
/// routes and the evals so far at the call.
const StreamRec = struct {
    a: std.mem.Allocator,
    recs: std.ArrayList(Rec) = .empty,
    finishes: u32 = 0,

    var stream: ?*expert_stream.Stream = null;
    var bank: ?*const SynthBank = null;
    /// The hook's bound banks of the layer (to name a call's bank kind).
    var base_code: u32 = 0;
    var transient_code: u32 = 0;

    const Rec = struct { experts: []u16, act_row: []u32, base: bool, live: u32, evals: usize, slot: []u32 = &.{} };

    fn init(a: std.mem.Allocator) StreamRec {
        return .{ .a = a };
    }

    fn deinit(self: *StreamRec, _: *TraceOps) void {
        for (self.recs.items) |r| {
            self.a.free(r.experts);
            self.a.free(r.act_row);
            self.a.free(r.slot);
        }
        self.recs.deinit(self.a);
    }

    fn expertIn(s: *expert_stream.Stream, sb: *const SynthBank, slot: u32) !u16 {
        const geom = &sb.bank.layers[0];
        for (0..sb.bank.n_experts) |e| {
            const off = sb.bank.recordOffset(0, @intCast(e));
            const held = for (geom.segments, 0..) |seg, c| {
                if (!std.mem.eql(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(0, slot, @fromBackingInt(@intCast(c)))[0..seg.length])) break false;
            } else true;
            if (held) return @intCast(e);
        }
        return error.SlotHoldsNoRecord;
    }

    pub fn call(self: *StreamRec, g: *TraceOps, act: u32, rows: quant.PrefillRows, b: BankArraysOf(u32)) !u32 {
        const s = stream.?;
        const base = b.gate.code == base_code;
        if (!base and b.gate.code != transient_code) return error.UnknownBank;
        const cap = s.layers[0].policy.capacity;
        const experts = try self.a.alloc(u16, rows.slot.len);
        errdefer self.a.free(experts);
        for (rows.slot, experts) |r, *e| e.* = try expertIn(s, bank.?, if (base) r else cap + r);
        var live: u32 = 0;
        for (&s.routes) |*r| live += @intFromBool(r.state == .live);
        const act_rows = try self.a.dupe(u32, rows.act_row.?);
        errdefer self.a.free(act_rows);
        const slots = try self.a.dupe(u32, rows.slot);
        errdefer self.a.free(slots);
        try self.recs.append(self.a, .{ .experts = experts, .act_row = act_rows, .base = base, .live = live, .evals = g.evals.items.len, .slot = slots });
        return g.input(&.{ @intCast(rows.slot.len), g.shapeOf(act).d[1] }, .float32);
    }

    pub fn finish(self: *StreamRec, _: *TraceOps) !void {
        self.finishes += 1;
    }
};

/// Decode-lane math that records each gateUp call's bank, the expert whose record each slot
/// holds at the call (StreamRec's stream and bank) and the act rows (the act-row index is the
/// host array made right before the call's `take`), in calls of <= 8 rows.
const ColdRec = struct {
    inner: TraceMath,
    pub const max_decode_rows: u32 = 8;
    var log: std.ArrayList(Call) = .empty;
    const Call = struct { code: u32, experts: []u16, act_row: []u32, evals: usize };

    fn reset(a: std.mem.Allocator) void {
        for (log.items) |c| {
            a.free(c.experts);
            a.free(c.act_row);
        }
        log.clearAndFree(a);
    }

    pub fn gateUp(self: *const ColdRec, g: *TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        const slots = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(g.hostBytesOf(ids) orelse return error.NoHostBytes)));
        const acts = g.hostBytesOf(x - 1) orelse return error.NoHostBytes;
        const s = StreamRec.stream.?;
        const base = gate.code == StreamRec.base_code;
        const experts = try g.gpa.alloc(u16, slots.len);
        errdefer g.gpa.free(experts);
        for (slots, experts) |slot, *e| e.* = try StreamRec.expertIn(s, StreamRec.bank.?, if (base) slot else s.layers[0].policy.capacity + slot);
        try log.append(g.gpa, .{
            .code = gate.code,
            .experts = experts,
            .act_row = try g.gpa.dupe(u32, std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(acts)))),
            .evals = g.evals.items.len,
        });
        return self.inner.gateUp(g, x, ids, gate, up);
    }

    pub fn down(self: *const ColdRec, g: *TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, d);
    }
};

/// A wide route whose outputs carry, per row, its act row, slot and bank (host arrays the trace
/// records): `callParts` hands them out in waves of 3 rows, last rows first.
const EncRoute = struct {
    const hidden = 64;

    fn rowsOut(g: *TraceOps, a: std.mem.Allocator, rows: quant.PrefillRows, bank: BankArraysOf(u32), which: []const u32) !u32 {
        const v = try a.alloc(f32, which.len * hidden);
        defer a.free(v);
        @memset(v, 0);
        for (which, 0..) |i, r| {
            v[r * hidden] = @floatFromInt(rows.act_row.?[i]);
            v[r * hidden + 1] = @floatFromInt(rows.slot[i]);
            v[r * hidden + 2] = @floatFromInt(bank.gate.code);
        }
        return g.hostArray(std.mem.sliceAsBytes(v), &.{ @intCast(which.len), hidden }, .float32);
    }

    pub fn call(_: *EncRoute, g: *TraceOps, _: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32)) !u32 {
        const idx = try g.gpa.alloc(u32, rows.slot.len);
        defer g.gpa.free(idx);
        for (idx, 0..) |*x, i| x.* = @intCast(i);
        return rowsOut(g, g.gpa, rows, bank, idx);
    }

    pub fn callParts(_: *EncRoute, g: *TraceOps, _: u32, rows: quant.PrefillRows, bank: BankArraysOf(u32), alloc: std.mem.Allocator, outs: *std.ArrayList(u32), pos: *std.ArrayList(u32)) !void {
        var end: usize = rows.slot.len;
        while (end > 0) {
            const start = end -| 3;
            var which: [3]u32 = undefined;
            for (start..end, 0..) |i, j| which[j] = @intCast(i);
            try outs.append(alloc, try rowsOut(g, alloc, rows, bank, which[0 .. end - start]));
            try pos.appendSlice(alloc, which[0 .. end - start]);
            end = start;
        }
    }

    pub fn finish(_: *EncRoute, _: *TraceOps) !void {}
};

/// The adjacent runs JOINLESS merged in before the minimal copy (`per = ceil(n / 24)` adjacent outputs per
/// source), as a plan: the reference the merge's tests resolve against.
fn planAdjacentRuns(rows: []const u32, src: []u8, off: []u32) usize {
    const per = (rows.len + joinless_sources - 1) / joinless_sources;
    var at: u32 = 0;
    for (rows, src, off, 0..) |r, *s, *o, i| {
        if (i % per == 0) at = 0;
        s.* = @intCast(i / per);
        o.* = at;
        at += r;
    }
    return (rows.len + per - 1) / per;
}

/// The (output, row) a plan's (source, row) names.
fn resolvePlan(rows: []const u32, src: []const u8, off: []const u32, s: i32, row: i32) ?[2]u32 {
    if (s < 0 or row < 0) return null;
    const r_: u32 = @intCast(row);
    for (rows, src, off, 0..) |r, si, o, i| {
        if (si == s and r_ >= o and r_ < o + r) return .{ @intCast(i), r_ - o };
    }
    return null;
}

/// A decode math whose outputs carry, per row, its slot row and the arrays it read (host arrays the trace records):
/// the per-bank calls and the banked calls (packed ids) must hand every routed position the same words.
const BankEnc = struct {
    pub const has_banked = true;
    const w = 4;

    fn rowsOf(g: *TraceOps, x: u32) ![]const u32 {
        const b = g.hostBytesOf(x) orelse return error.NoHostBytes;
        return std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(b)));
    }

    fn floatsOf(g: *TraceOps, x: u32) ![]const f32 {
        const b = g.hostBytesOf(x) orelse return error.NoHostBytes;
        return std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(b)));
    }

    fn emit(g: *TraceOps, v: []const f32) !u32 {
        return g.hostArray(std.mem.sliceAsBytes(v), &.{ @intCast(v.len / w), w }, .float32);
    }

    pub fn gateUp(_: *const BankEnc, g: *TraceOps, _: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const r = try rowsOf(g, ids);
        for (r, 0..) |row, i| v[i * w ..][0..w].* = .{ @floatFromInt(row), @floatFromInt(gate.code), @floatFromInt(up.rin), 0 };
        return emit(g, v[0 .. r.len * w]);
    }

    pub fn down(_: *const BankEnc, g: *TraceOps, h: u32, _: u32, d: ProjOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const hv = try floatsOf(g, h);
        @memcpy(v[0..hv.len], hv);
        for (0..hv.len / w) |i| v[i * w + 3] = @floatFromInt(d.rout);
        return emit(g, v[0..hv.len]);
    }

    pub fn gateUpBanked(_: *const BankEnc, g: *TraceOps, _: u32, ids: u32, banks: *const [n_banks]BankArraysOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const r = try rowsOf(g, ids);
        for (r, 0..) |p, i| {
            const b = &banks[p >> 24];
            v[i * w ..][0..w].* = .{ @floatFromInt(p & 0xFFFFFF), @floatFromInt(b.gate.code), @floatFromInt(b.up.rin), 0 };
        }
        return emit(g, v[0 .. r.len * w]);
    }

    pub fn downBanked(_: *const BankEnc, g: *TraceOps, h: u32, ids: u32, banks: *const [n_banks]BankArraysOf(u32)) !u32 {
        var v: [max_route_ids * w]f32 = undefined;
        const hv = try floatsOf(g, h);
        const r = try rowsOf(g, ids);
        @memcpy(v[0..hv.len], hv);
        for (r, 0..) |p, i| v[i * w + 3] = @floatFromInt(banks[p >> 24].down.rout);
        return emit(g, v[0..hv.len]);
    }
};

/// TraceMath with the banked texts (DEVROUTE's structural test): shapes only.
const TraceBankedMath = struct {
    pub const has_banked = true;
    inner: TraceMath,
    pub fn gateUp(self: *const TraceBankedMath, g: *ops.TraceOps, x: u32, ids: u32, gate: ProjOf(u32), up: ProjOf(u32)) !u32 {
        return self.inner.gateUp(g, x, ids, gate, up);
    }
    pub fn down(self: *const TraceBankedMath, g: *ops.TraceOps, h: u32, ids: u32, d: ProjOf(u32)) !u32 {
        return self.inner.down(g, h, ids, d);
    }
    pub fn gateUpBanked(self: *const TraceBankedMath, g: *ops.TraceOps, x: u32, ids: u32, _: *const [n_banks]BankArraysOf(u32)) !u32 {
        if (g.shapeOf(x).d[1] != self.inner.hidden or g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inner.inter }, .float32);
    }
    pub fn downBanked(self: *const TraceBankedMath, g: *ops.TraceOps, h: u32, ids: u32, _: *const [n_banks]BankArraysOf(u32)) !u32 {
        if (g.shapeOf(h).d[1] != self.inner.inter or g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.inner.hidden }, .float32);
    }
};

/// One run of two decode calls under DEVROUTE and / or HOIST_FIRST (stack4's routed-path pair), recorded for comparison:
/// the non-commit nodes, and per call the classes of the commits before the routing barrier's read (B the barrier's
/// arrays, D the device hit wave, H the hoist) and the commits after the route.
const DevHoistRun = struct {
    ops: std.ArrayList(ops.Op) = .empty,
    shapes: std.ArrayList(ops.Shape) = .empty,
    pre: [2][8]u8 = undefined,
    n_pre: [2]usize = .{ 0, 0 },
    post_route: [2]usize = .{ 0, 0 },
    hoist_after_read: [2]usize = .{ 0, 0 },
    lut_ok: [2]bool = .{ false, false },
    flipped: [2]bool = .{ false, false },
    bound_dev: bool = false,
    bound_hoist: bool = false,

    fn deinit(r: *DevHoistRun, a: std.mem.Allocator) void {
        r.ops.deinit(a);
        r.shapes.deinit(a);
    }

    fn preOf(r: *const DevHoistRun, call: usize) []const u8 {
        return r.pre[call][0..r.n_pre[call]];
    }

    fn run(comptime R: Routes, a: std.mem.Allocator, devroute: bool, hoist_first: bool, out: *DevHoistRun) !void {
        var c = testConfig(256, 128, 2);
        c.n_routed_experts = 16;
        var src = try FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 16, .rows = &.{ 4, 4 } });
        defer src.deinit();
        if (R.lookahead) src.selector = try expert_lookahead.Selector.init(a, 16, 6, std.math.inf(f32), 2);
        var g = TraceOps.init(a);
        defer g.deinit();
        g.record_host = true;
        const Ex = ExpertsWith(TraceOps, FakeSource, TraceBankedMath, R);
        const math: TraceBankedMath = .{ .inner = .{ .hidden = 256, .inter = 128 } };
        var gates: [2]Ex.Gate = undefined;
        for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 16, 256 }, .bfloat16), .bias = try g.input(&.{16}, .float32) };
        var ex = try Ex.initWith(a, &g, &src, math, &c, .{ .gates = if (R.lookahead) &gates else &.{}, .banked = true, .devroute = devroute, .hoist_first = hoist_first });
        defer ex.deinit();
        out.bound_dev = ex.devRoute();
        out.bound_hoist = ex.hoistFirst();
        try ex.grow(&g, &.{ 8, 8 });
        src.log.clearRetainingCapacity();
        src.trace = &g;
        // Call 1: six misses (two parts); call 2: three hits, one part.
        var vals: HoistRun.Values = .{ .calls = &.{ &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 7, 1, 7, 9, 3 } } };
        g.host_values = vals.values();
        const xf = try g.input(&.{ 2, 256 }, .bfloat16);
        const idx = try g.input(&.{ 2, 3 }, .int32);
        const tail = [_]u32{ try g.input(&.{ 2, 256 }, .float32), try g.input(&.{ 2, 3 }, .float32) };
        const first = g.nodes.items.len;
        for (0..2) |call| {
            const parity0 = if (devroute) ex.dev_parity[0] else 0;
            const n0 = g.nodes.items.len;
            const c0 = g.commit_spans.items.len;
            const out_arr = try ex.at(0).routedHoist(&g, xf, idx, &tail);
            if (!g.shapeOf(out_arr).eql(ops.Shape.of(&.{ 2, 3, 256 }))) return error.JoinShape;
            // The call's route event (the last in the source's log) and its first host read.
            var r_at: usize = 0;
            for (src.log.items) |ev| if (ev.kind == .route) {
                r_at = ev.at;
            };
            var read_at: usize = g.nodes.items.len;
            for (g.nodes.items[n0..], n0..) |nd, i| if (nd.op == .host_read) {
                read_at = i;
                break;
            };
            var k = c0;
            for (g.nodes.items[n0..], n0..) |nd, i| if (nd.op == .async_eval) {
                const span = g.commit_spans.items[k];
                k += 1;
                const xs = g.committed.items[span[0]..span[1]];
                const cls: u8 = if (std.mem.indexOfScalar(u32, xs, idx) != null) 'B' else if (std.mem.indexOfScalar(u32, xs, tail[0]) != null) 'H' else 'D';
                if (i < read_at) {
                    out.pre[call][out.n_pre[call]] = cls;
                    out.n_pre[call] += 1;
                } else if (cls == 'H') out.hoist_after_read[call] += 1;
                if (i >= r_at and cls != 'H') out.post_route[call] += 1;
            };
            if (devroute) {
                out.flipped[call] = ex.dev_parity[0] != parity0;
                var want: [16]u32 = undefined;
                src.residentLut(0, &want);
                const got = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[0][ex.dev_parity[0]]))));
                out.lut_ok[call] = std.mem.eql(u32, &want, got);
            }
        }
        for (g.nodes.items[first..]) |nd| if (nd.op != .async_eval) {
            try out.ops.append(a, nd.op);
            try out.shapes.append(a, nd.shape);
        };
        try ex.flush();
    }
};
