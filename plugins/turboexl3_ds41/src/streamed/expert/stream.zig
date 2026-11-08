//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
const host_env = @import("sdk").env;
const sdk = @import("sdk");
const sdk_ext = @import("../sdk_ext.zig");
/// PROFILE builds only (`-Ddsv41-prefill-timers=true`): P1's read-ahead record; every call compiles to nothing otherwise.
const bo = @import("../build_flags.zig");
const mlx = @import("sdk").mlx;
const expert_bank = @import("bank.zig");
const expert_io = sdk_ext.expert.io;
const expert_policy = @import("../sdk_ext.zig").expert.policy;
const expert_lookahead = @import("lookahead.zig");
const exl3_quant = @import("../../exl3/quant.zig");

const S = sdk_ext.expert.stream.StreamOf(expert_bank, if (@hasDecl(bo, "dsv41_prefill_timers")) bo.dsv41_prefill_timers else false);
const n_components = S.n_components;
const gu_components = S.gu_components;
const Component = S.Component;
const Layer = S.Layer;
const LayerPolicy = S.LayerPolicy;
const Phase = S.Phase;
const Plan = S.Plan;
const max_route_ids = S.max_route_ids;
pub const HostSlotRows = S.HostSlotRows;
pub const LayerSlotBank = S.LayerSlotBank;
const mlxActive = S.mlxActive;
const evalArrays = S.evalArrays;
pub const SlotMemory = S.SlotMemory;
const evalRows = S.evalRows;
const Rows = S.Rows;
pub const source_caps = S.source_caps;
pub const uses_reader = S.uses_reader;
pub const BankKind = S.BankKind;
pub const SlotRef = S.SlotRef;
pub const Options = S.Options;
pub const read_ahead_probed = S.read_ahead_probed;
pub const ReadAheadProbe = S.ReadAheadProbe;
pub const GrowFill = S.GrowFill;
/// the pool's options and an extra prompt array (`S.BankPool`).
pub const PoolOptions = S.PoolOptions;
pub const SpareArray = S.SpareArray;
pub const Residents = S.Residents;
const ProbeSlot = S.ProbeSlot;
const no_probe = S.no_probe;
pub const Lookahead = S.Lookahead;
pub const Tally = S.Tally;
pub const Event = S.Event;
pub const Gates = S.Gates;
pub const Stats = S.Stats;
pub const Error = S.Error;
const SlotState = S.SlotState;
const SlotMeta = S.SlotMeta;
pub const Part = S.Part;
pub const Route = S.Route;
const route_capacity = S.route_capacity;
pub const max_wide_depth = S.max_wide_depth;
pub const phase_change_releases_wide_windows = S.phase_change_releases_wide_windows;
pub const transient_release_default = S.transient_release_default;
pub const decode_staging_rows = S.decode_staging_rows;
const wait_timeout_ns = S.wait_timeout_ns;
pub const Stream = S.Stream;
/// The EXL3 bank's slot arrays by projection.
pub const ProjArrays = expert_bank.ProjArrays;
pub const BankArrays = expert_bank.BankArrays;

/// A bank's arrays by projection, as the quant binds them (`sdk_ext.quant.BankArrays` of the EXL3 quant's `Arrays`).
pub fn BankArraysOf(comptime T: type) type {
    return sdk_ext.quant.BankArrays(exl3_quant.Arrays(T));
}

/// Per routed id of `r`, its wave: 0 for a slot resident at the call (a hit),
/// p + 1 for a slot part p loads.
pub fn wavesOf(plan: *const Plan, hit_slots: []const u32, part_loads: []const []const expert_policy.Load, out: []u8) void {
    var keys: [max_route_ids]u32 = undefined;
    var vals: [max_route_ids]u8 = undefined;
    var n: usize = 0;
    for (hit_slots) |s| {
        keys[n] = s;
        vals[n] = 0;
        n += 1;
    }
    for (part_loads, 1..) |loads, w| for (loads) |l| {
        keys[n] = l.slot;
        vals[n] = @intCast(w);
        n += 1;
    };
    for (plan.slotsOf(), out) |s, *w| w.* = vals[std.mem.indexOfScalar(u32, keys[0..n], s).?];
}

/// Trace arrays in one record's geometry, `rows` rows (a trace backend's `input`).
pub fn traceBank(g: anytype, geom: *const Layer, rows: u32) !BankArraysOf(@TypeOf(g.*).T) {
    var a: [n_components]@TypeOf(g.*).T = undefined;
    for (&a, geom.segments) |*x, seg| {
        var shape: [4]c_int = undefined;
        shape[0] = @intCast(rows);
        for (seg.shape[0..seg.rank], 1..) |d, i| shape[i] = @intCast(d);
        x.* = try g.input(shape[0 .. seg.rank + 1], switch (seg.dtype) {
            .I16 => .int16,
            .F16 => .float16,
        });
    }
    return .{
        .gate = .{ .code = a[0], .rout = a[1], .rin = a[2] },
        .up = .{ .code = a[3], .rout = a[4], .rin = a[5] },
        .down = .{ .code = a[6], .rout = a[7], .rin = a[8] },
    };
}

/// DEVROUTE's resident map of one layer from its policy: each resident expert's persistent slot as its packed
/// (bank << 24 | row) through `src.slotRef`; non-resident experts `missing` (the math's: 0, base row 0, or an id its
/// banked texts skip).
pub fn lutOf(pol: *const expert_policy.LayerPolicy, src: anytype, layer: u32, out: []u32, missing: u32) void {
    for (out, 0..) |*o, e| {
        o.* = missing;
        if (e >= pol.n_experts) continue;
        const s = pol.slotOf(@intCast(e)) orelse continue;
        if (s >= pol.capacity) continue;
        const ref = src.slotRef(layer, s);
        o.* = (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
    }
}

// ── StreamSource: the stream as an expert source (the EXL3 source's side of the contract) ──

/// The streamer as an expert source. Routes, waits, release, flush and growth
/// are the Stream's own; the call view comes from `refsOf` and the parts'
/// loads (`Route.partLoads`); the MLX arrays from `bankArrays` (slot memory
/// `.mlx`: with host rows the MLX executor refuses at construction).
pub const StreamSource = struct {
    stream: *Stream,
    calls: [n_routes]Call = @splat(.{}),

    /// the next routed layer's predicted routes, an unevaluated uint32 or int32
    /// [rows, k] array held from `setNext` until read: started on the GPU behind the call before that layer's barrier
    /// (`nextStart`), read once that call's routes are all made (`postNext`). Kept until read, replaced by the next
    /// `setNext`, or let go at a phase change.
    next_ids: mlx.mlx_array = .{ .ctx = null },
    next_layer: u32 = 0,
    /// What the stream supports; an arm installs a subset at construction.
    pub const caps = source_caps;
    /// The slot arrays it fills are the EXL3 quant's (`sdk_ext.expert`: a source's arrays are its quant's).
    pub const Arrays = exl3_quant.Arrays;

    /// One call per live route of the Stream's ring.
    pub const n_routes = @typeInfo(@FieldType(Stream, "routes")).array.len;

    pub const Call = struct {
        route: ?*Route = null,
        n_ids: u32 = 0,
        refs: [max_route_ids]SlotRef = undefined,
        waves: [max_route_ids]u8 = undefined,
    };

    pub fn init(stream: *Stream) StreamSource {
        return .{ .stream = stream };
    }

    /// Calls mirror the Stream's route ring one to one.
    fn callOf(self: *StreamSource, r: *Route) *Call {
        const i = (@intFromPtr(r) - @intFromPtr(&self.stream.routes[0])) / @sizeOf(Route);
        return &self.calls[i];
    }

    pub fn route(self: *StreamSource, layer: u32, ids: []const u16, scores: []const f32) Error!*Call {
        const r = try self.stream.route(layer, ids, scores);
        const call = self.callOf(r);
        call.* = .{ .route = r, .n_ids = @intCast(ids.len) };
        _ = self.stream.refsOf(r, &call.refs);
        var bufs: [max_route_ids][max_route_ids]expert_policy.Load = undefined;
        var parts: [max_route_ids][]const expert_policy.Load = undefined;
        for (0..r.n_parts) |p| parts[p] = r.partLoads(@intCast(p), &bufs[p]);
        wavesOf(&r.plan, r.hit_slots[0..r.plan.n_hits], parts[0..r.n_parts], call.waves[0..ids.len]);
        return call;
    }

    pub fn served(_: *StreamSource, call: *const Call) sdk_ext.expert.Served {
        return .{ .refs = call.refs[0..call.n_ids], .waves = call.waves[0..call.n_ids], .n_parts = call.route.?.n_parts };
    }

    /// DEVROUTE: layer `layer`'s resident map as the device reads it: `out[e]` = the packed (bank << 24 | row) of
    /// expert e's persistent slot, else `missing` (a miss: the device row is computed from base row 0 at 0, or skipped
    /// at the math's skip id, and never joined).
    pub fn residentLut(self: *const StreamSource, layer: u32, out: []u32, missing: u32) void {
        lutOf(&self.stream.layers[layer].policy, self.stream, layer, out, missing);
    }

    /// The decode phase (the grown banks; DEVROUTE's LUTs exist).
    pub fn decoding(self: *const StreamSource) bool {
        return self.stream.phase == .decode;
    }

    pub fn waitGu(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitGu(call.route.?, part);
    }

    pub fn waitDown(self: *StreamSource, call: *Call, part: u32) Error!void {
        return self.stream.waitDown(call.route.?, part);
    }

    /// The call's read-ahead step after its waves (the next layer's scores read off the barrier).
    pub fn speculateLate(self: *StreamSource, call: *Call, scores: []const f32) Error!void {
        return self.stream.speculateLate(call.route.?, scores);
    }
    pub fn release(self: *StreamSource, call: *Call) void {
        self.stream.release(call.route.?);
    }

    pub fn flush(self: *StreamSource) Error!void {
        return self.stream.flush();
    }

    pub fn grow(self: *StreamSource, decode_rows: []const u32) !void {
        return self.stream.grow(decode_rows);
    }

    /// The phase change's first free (`Stream.releaseTransient`): the bytes freed.
    pub fn releaseTransient(self: *StreamSource) !u64 {
        return self.stream.releaseTransient();
    }

    /// The reverse phase change's free (`Stream.shrink`): the bytes freed.
    pub fn shrink(self: *StreamSource, prompt_rows: []const u32, residents: Residents) !u64 {
        return self.stream.shrink(prompt_rows, residents);
    }

    /// The reverse phase change's allocation (`Stream.regrowTransient`): the bytes allocated.
    pub fn regrowTransient(self: *StreamSource) !u64 {
        return self.stream.regrowTransient();
    }

    pub fn seedPrefill(self: *StreamSource, layer: u32, ids: []const u16) !void {
        return self.stream.seedPrefill(layer, ids);
    }

    /// The last seedPrefill's seed ranks (`LayerPolicy.seed_ranks`).
    pub fn seedRanks(self: *const StreamSource, layer: u32) u32 {
        return self.stream.seedRanks(layer);
    }

    /// P1: the layer's predicted seed read ahead of its routes (`Stream.readAheadSeed`), then (staging) the rest of the
    /// prediction its call's first stream groups will load staged in their windows (`Stream.stageGroups`).
    pub fn readAheadSeed(self: *StreamSource, layer: u32, experts: []const u16) !void {
        return self.readAheadSeedVotes(layer, experts, &.{});
    }
    /// `readAheadSeed` with each expert's votes in the prediction (`votes`, by expert), for the staging's vote floor and
    /// its records' votes (the staging floor).
    pub fn readAheadSeedVotes(self: *StreamSource, layer: u32, experts: []const u16, votes: []const u32) !void {
        try self.stream.readAheadSeed(layer, experts);
        try self.stream.stageGroups(layer, experts, votes);
    }
    /// Next-layer staging: `layer`'s predicted routes `ids` (an unevaluated uint32 or int32 [rows, k] array: the
    /// routing kernel's ids, row-contiguous), held for the call before it; a held one not read yet is dropped.
    pub fn setNext(self: *StreamSource, layer: u32, ids: mlx.mlx_array) !void {
        self.dropNext();
        var held = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(held);
        try mlx.check(mlx.mlx_array_set(&held, ids));
        self.next_ids = held;
        self.next_layer = layer;
    }
    /// Next-layer staging: the held prediction let go.
    pub fn dropNext(self: *StreamSource) void {
        if (self.next_ids.ctx != null) _ = mlx.mlx_array_free(self.next_ids);
        self.next_ids = .{ .ctx = null };
    }
    /// Next-layer staging: `layer`'s call has its routes' ids on the host: the next layer's held prediction starts on
    /// the GPU behind it (nothing waits).
    pub fn nextStart(self: *StreamSource, layer: u32) !void {
        // A staging planned for this layer that no read-ahead of its own took becomes its own now.
        try self.stream.promoteFor(layer);
        if (self.next_ids.ctx == null or self.next_layer != layer + 1) return;
        const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{self.next_ids}, 1);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }
    /// Next-layer staging: `layer`'s call has made its last route: the next layer's held prediction read back once
    /// (each expert's votes, hottest first: the read-ahead's ranking) and read ahead (`Stream.readAheadNext`, the whole
    /// ranking so the row choice sees every named expert), with `windows` its stream misses planned into the windows
    /// the call lets go (`Stream.stageNext`, which posts the ones no route holds now; with each expert's votes, for the
    /// staging floor).
    pub fn postNext(self: *StreamSource, layer: u32) !void {
        if (self.next_ids.ctx == null or self.next_layer != layer + 1) return;
        defer self.dropNext();
        try mlx.check(mlx.mlx_array_eval(self.next_ids));
        const a = self.stream.allocator;
        const n_experts: usize = self.stream.bank.n_experts;
        const counts = try a.alloc(u32, n_experts);
        defer a.free(counts);
        @memset(counts, 0);
        const n = mlx.mlx_array_size(self.next_ids);
        switch (mlx.mlx_array_dtype(self.next_ids)) {
            // The routing kernel's ids (uint32), or a cast of them.
            .uint32 => for ((mlx.mlx_array_data_uint32(self.next_ids) orelse return error.MlxNoData)[0..n]) |e| {
                if (e >= n_experts) return error.StageNextIds;
                counts[e] += 1;
            },
            .int32 => for ((mlx.mlx_array_data_int32(self.next_ids) orelse return error.MlxNoData)[0..n]) |e| {
                if (e < 0 or e >= n_experts) return error.StageNextIds;
                counts[@intCast(e)] += 1;
            },
            else => return error.StageNextIds,
        }
        const ranked_buf = try a.alloc(u16, n_experts);
        defer a.free(ranked_buf);
        const ranked = expert_policy.rankHottest(counts, ranked_buf);
        try self.stream.readAheadNext(layer + 1, ranked);
        try self.stream.stageNext(layer + 1, ranked, counts);
    }
    /// Next-layer staging: a release after the call's last route (`block` false: a window the copier still reads waits
    /// for a later one), or the call's end (`block`: every window is let go): the planned windows no route holds posted
    /// (`Stream.postNextWindows`).
    pub fn postNextWindows(self: *StreamSource, block: bool) !void {
        try self.stream.postNextWindows(block);
    }
    /// the layer's staged parts its call's routes (`used`) do not take, withdrawn
    /// where no read has started (`Stream.trimStaged`).
    pub fn trimStaged(self: *StreamSource, layer: u32, used: []const u16) !void {
        return self.stream.trimStaged(layer, used);
    }
    /// Staging: the layer's staged groups, once, per window (`Stream.stagedWindows`).
    pub fn stagedWindows(self: *StreamSource, layer: u32, depth: usize, out: *[max_wide_depth][]const u16) bool {
        return self.stream.stagedWindows(layer, depth, out);
    }

    /// P1: the layer's read-ahead landed (`Stream.awaitReadAhead`).
    pub fn awaitReadAhead(self: *StreamSource, layer: u32) Error!void {
        return self.stream.awaitReadAhead(layer);
    }

    /// P1's construction self-check over `n` experts of `layer` not resident there (the highest ids):
    /// read ahead == demand read, bit for bit (`Stream.checkReadAhead`). Returns how many were checked.
    pub fn checkReadAhead(self: *StreamSource, layer: u32, n: u32) !u32 {
        const policy = &self.stream.layers[layer].policy;
        var experts: [max_route_ids]u16 = undefined;
        var k: u32 = 0;
        var e: u32 = policy.n_experts;
        while (e > 0 and k < @min(n, max_route_ids)) {
            e -= 1;
            if (policy.slotOf(@intCast(e)) != null) continue;
            experts[k] = @intCast(e);
            k += 1;
        }
        try self.stream.checkReadAhead(layer, experts[0..k]);
        return k;
    }

    /// Whether `expert` holds one of `layer`'s persistent slots now (the profile builds' residency query, before a route).
    pub fn isResident(self: *const StreamSource, layer: u32, expert: u16) bool {
        return self.stream.layers[layer].policy.slotOf(expert) != null;
    }

    /// The pool's demand read gauge now: wall ns with a demand (or pre-) read in flight (A0's per-layer split).
    pub fn readWallNs(self: *StreamSource) u64 {
        return @intCast(@max(self.stream.pool.readGauge()[4], 0));
    }

    /// A live prefill call's persistent slots kept pinned and held past its release (`Wide.defer_base`).
    pub fn holdBase(self: *StreamSource, call: *Call) !void {
        return self.stream.holdBase(call.route.?);
    }

    pub fn releaseHeld(self: *StreamSource) void {
        self.stream.releaseHeld();
    }

    /// Prefill routes one layer may hold live at once (`Stream.Options.wide_depth`).
    pub fn wideDepth(self: *const StreamSource) u8 {
        return self.stream.wide_depth;
    }

    pub fn stats(self: *StreamSource) Stats {
        return self.stream.stats();
    }

    /// Event gates of the call's reads (a stream built with `event`).
    pub fn gate(self: *StreamSource, call: *Call) Error!?Gates {
        return self.stream.gate(call.route.?);
    }

    pub fn bankRows(self: *StreamSource, layer: u32, kind: BankKind) u32 {
        const ls = &self.stream.layers[layer];
        return switch (kind) {
            .base => ls.base.rows,
            .ext => if (ls.ext) |e| e.rows else 0,
            .transient => self.stream.transient.rows,
        };
    }

    /// MLX backends: the stream's own arrays (never freed here); null for host rows or a bank without rows.
    /// Trace backends: inputs in the bank's geometry.
    pub fn bankArrays(self: *StreamSource, g: anytype, layer: u32, kind: BankKind) !?BankArraysOf(@TypeOf(g.*).T) {
        const G = @TypeOf(g.*);
        if (G.T == mlx.mlx_array) {
            const b = self.stream.bankArrays(layer, kind) orelse return null;
            return .{
                .gate = .{ .code = b.gate.code, .rout = b.gate.rout, .rin = b.gate.rin },
                .up = .{ .code = b.up.code, .rout = b.up.rout, .rin = b.up.rin },
                .down = .{ .code = b.down.code, .rout = b.down.rout, .rin = b.down.rin },
            };
        } else {
            if (comptime !@hasDecl(G, "input")) @compileError("StreamSource binds MLX arrays or a trace backend's inputs; " ++ @typeName(G) ++ " has neither");
            const rows = self.bankRows(layer, kind);
            if (rows == 0) return null;
            return try traceBank(g, &self.stream.bank.layers[layer], rows);
        }
    }
};
