//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
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
/// (bank << 24 | row) through `src.slotRef`; non-resident experts 0.
pub fn lutOf(pol: *const expert_policy.LayerPolicy, src: anytype, layer: u32, out: []u32) void {
    for (out, 0..) |*o, e| {
        o.* = 0;
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
    /// expert e's persistent slot, else 0 (a miss: the device row is computed from base row 0 and never joined).
    pub fn residentLut(self: *const StreamSource, layer: u32, out: []u32) void {
        lutOf(&self.stream.layers[layer].policy, self.stream, layer, out);
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

    pub fn seedPrefill(self: *StreamSource, layer: u32, ids: []const u16) !void {
        return self.stream.seedPrefill(layer, ids);
    }

    /// The last seedPrefill's seed ranks (`LayerPolicy.seed_ranks`).
    pub fn seedRanks(self: *const StreamSource, layer: u32) u32 {
        return self.stream.seedRanks(layer);
    }

    /// P1: the layer's predicted seed read ahead of its routes (`Stream.readAheadSeed`).
    pub fn readAheadSeed(self: *StreamSource, layer: u32, experts: []const u16) !void {
        return self.stream.readAheadSeed(layer, experts);
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

// ── Tests ──

const testing = std.testing;

const FixSeg = struct { component: []const u8, offset: u64, length: u64, sha256: []const u8, head16: []const u8 };
const FixRec = struct {
    layer: u32,
    expert: u32,
    sidecar_offset: u64,
    record_bytes: u64,
    logical_bytes: u64,
    v2_sha256: []const u8,
    v1_sha256: []const u8,
    segments: []const FixSeg,
};
const Fixture = struct { layer_set: []const FixRec, pick_set: []const FixRec };

const Env = struct {
    bank: expert_bank.Bank,
    text: []u8,
    parsed: std.json.Parsed(Fixture),

    fn close(self: *Env) void {
        self.parsed.deinit();
        testing.allocator.free(self.text);
        self.bank.deinit();
    }
};

/// A 2-layer synthetic bank on disk (hidden 64, inter 32: 2,880-byte records
/// in 4 KiB slots), opened, with its experts.bin image.
const SynthBank = expert_bank.SynthBank;

/// Routes `ids` and waits for every part (the stream's `serve`).
const serve = Stream.serve;

// ── Lookahead, pre-read and event gates ──
