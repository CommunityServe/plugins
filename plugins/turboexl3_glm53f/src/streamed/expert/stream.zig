//! Expert residency of the streamer. Slot rows: one bank per record
//! component, `rows` rows of that component's segment length, handed to the
//! read pool as nine destination addresses per row (`LayerSlotBank` is the
//! MLX-owned form the kernels bind, `HostSlotRows` the same layout in host
//! pages; `Options.slot_memory` picks one). `Stream`: per-layer slot pools,
//! routes, deferred release, growth, lookahead and event gates.

const std = @import("std");
const host_env = @import("mlx_host").env;
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

    /// DSV41_BANK + DSV41_PHASE0_FIXTURE, else the test is skipped.
    fn open() !Env {
        const dir = std.mem.span(host_env.get(.DSV41_BANK) orelse return error.SkipZigTest);
        const fixture = std.mem.span(host_env.get(.DSV41_PHASE0_FIXTURE) orelse return error.SkipZigTest);
        var diag: expert_bank.Diag = .{};
        var bank = expert_bank.Bank.open(testing.allocator, std.testing.io, dir, expert_bank.dsv41, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        errdefer bank.deinit();
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, fixture, testing.allocator, .limited(4 << 20));
        errdefer testing.allocator.free(text);
        const parsed = try std.json.parseFromSlice(Fixture, testing.allocator, text, .{ .ignore_unknown_fields = true });
        return .{ .bank = bank, .text = text, .parsed = parsed };
    }

    fn close(self: *Env) void {
        self.parsed.deinit();
        testing.allocator.free(self.text);
        self.bank.deinit();
    }
};

fn hexEq(hex: []const u8, bytes: []const u8) bool {
    var buf: [64]u8 = undefined;
    if (hex.len != 2 * bytes.len or hex.len > 2 * buf.len) return false;
    const got = std.fmt.hexToBytes(&buf, hex) catch return false;
    return std.mem.eql(u8, got, bytes);
}

/// Reads `set` (one job, <= 8 records of one geometry) through the pool into
/// `rows`, then checks every record against a direct pread of the full record,
/// both manifest digests and the Python reader's per-component sha256.
fn checkSet(name: []const u8, bank: *const expert_bank.Bank, set: []const FixRec, rows: anytype) !u64 {
    const n = set.len;
    var refs: [expert_io.max_items]expert_bank.RecordRef = undefined;
    var dests: [expert_io.max_items][n_components]u64 = undefined;
    for (set, 0..) |r, i| {
        try testing.expectEqual(bank.recordOffset(r.layer, r.expert), r.sidecar_offset);
        refs[i] = .{ .layer = r.layer, .expert = r.expert };
        dests[i] = rows.rowDest(@intCast(i));
    }
    // Stops (drains + joins) before the caller frees `rows`.
    var pool = try expert_io.Pool.start(testing.allocator, .{});
    defer pool.stop();
    const first = try expert_bank.submitRecords(pool, bank, refs[0..n], dests[0..n]);
    try pool.wait(first, @intCast(2 * n), 60 * std.time.ns_per_s);
    var calls: i64 = 0;
    var payload: i64 = 0;
    var returned: i64 = 0;
    var t0: i64 = std.math.maxInt(i64);
    var t1: i64 = 0;
    for (0..2 * n) |t| {
        const res = pool.result(first + @as(u32, @intCast(t)));
        try testing.expectEqual(expert_io.Status.ok, res.status);
        calls += res.preadv_calls;
        payload += res.payload;
        returned += res.bytes_returned;
        t0 = @min(t0, res.t_start_ns);
        t1 = @max(t1, res.t_end_ns);
    }
    std.debug.print("{s}: {d} records, pool payload {d} B in {d} preadv ({d} B returned) over {d} us; verify pread {d} B\n", .{ name, n, payload, calls, returned, @divTrunc(t1 - t0, 1000), n * set[0].record_bytes });

    const Sha256 = std.crypto.hash.sha2.Sha256;
    const whole = try testing.allocator.alloc(u8, @intCast(set[0].record_bytes));
    defer testing.allocator.free(whole);
    for (set, 0..) |r, i| {
        const layer = &bank.layers[r.layer];
        try testing.expectEqual(layer.record_bytes, r.record_bytes);
        var got: usize = 0;
        while (got < whole.len) {
            const k = std.c.pread(bank.sidecar.fd, whole[got..].ptr, whole.len - got, @intCast(r.sidecar_offset + got));
            if (k <= 0) return error.ShortRead;
            got += @intCast(k);
        }
        var d: [32]u8 = undefined;
        Sha256.hash(whole, &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).padded, &d);
        try testing.expect(hexEq(r.v2_sha256, &d));
        Sha256.hash(whole[0..@intCast(layer.logical_bytes)], &d, .{});
        try testing.expectEqualSlices(u8, &bank.digest(r.layer, r.expert).logical, &d);
        try testing.expect(hexEq(r.v1_sha256, &d));
        try testing.expectEqual(@as(usize, n_components), r.segments.len);
        for (layer.segments, r.segments, 0..) |seg, fs, c| {
            const comp: Component = @enumFromInt(c);
            try testing.expectEqualStrings(comp.name(), fs.component);
            try testing.expectEqual(r.sidecar_offset + seg.offset, fs.offset);
            try testing.expectEqual(seg.length, fs.length);
            const slot = rows.row(comp, @intCast(i))[0..@intCast(seg.length)];
            try testing.expectEqualSlices(u8, whole[@intCast(seg.offset)..][0..@intCast(seg.length)], slot);
            Sha256.hash(slot, &d, .{});
            try testing.expect(hexEq(fs.sha256, &d));
            try testing.expect(hexEq(fs.head16, slot[0..16]));
        }
    }
    return @intCast(calls);
}

/// A 2-layer synthetic bank on disk (hidden 64, inter 32: 2,880-byte records
/// in 4 KiB slots), opened, with its experts.bin image.
const SynthBank = expert_bank.SynthBank;

const test_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 };

/// Routes `ids` and waits for every part (the stream's `serve`).
const serve = Stream.serve;

/// Every routed id's slot holds its record's bytes (the stream's `expectServed` over the bank's image).
fn expectServed(s: *Stream, sb: *const SynthBank, r: *const Route, ids: []const u16) !void {
    return s.expectServed(r, ids, sb.image);
}

/// Expert `e`'s resident row of `layer` holds its record's bytes, component by component.
fn expectSlotHolds(s: *Stream, sb: *const SynthBank, layer: u32, e: u16) !void {
    const slot = s.layers[layer].policy.slotOf(e) orelse return error.NotResident;
    const off = sb.bank.recordOffset(layer, e);
    for (sb.bank.layers[layer].segments, 0..) |seg, comp| {
        try testing.expectEqualSlices(u8, sb.image[off + seg.offset ..][0..seg.length], s.slotRow(layer, slot, @fromBackingInt(@intCast(comp)))[0..seg.length]);
    }
}

fn readsOf(r: *const Route) u64 {
    var n: u64 = 0;
    for (r.partsOf()) |p| n += p.n_reads;
    return n;
}

/// One layer's prompt call on a fresh stream (the pool is the process's: one stream at a time), its predicted
/// seed read ahead first when given: every routed id served from its record; the call's hits and stats.
fn p1Call(sb: *const SynthBank, predicted: ?[]const u16) !struct { hits: u64, st: Stats } {
    const s = try Stream.init(testing.allocator, &sb.bank, .{ .rows = &.{ 6, 6 }, .max_route_ids = 12, .transient_rows = 12, .pool = test_pool });
    defer s.deinit();
    if (predicted) |p| {
        try s.readAheadSeed(0, p);
        try testing.expect(s.ahead.live and s.ahead.n == p.len);
        try s.awaitReadAhead(0);
        for (p) |e| {
            const slot = s.layers[0].policy.slotOf(e).?;
            try testing.expectEqual(SlotState.ready, s.locate(0, slot).meta.state);
            try testing.expectEqual(@as(u16, 0), s.pinsOf(0, slot));
        }
    }
    // The layer's routed ids at its barrier, then its distinct experts in two groups.
    try s.seedPrefill(0, &.{ 1, 2, 3, 1, 2, 1, 4, 5, 6, 9, 10, 4, 1, 2 });
    var hits: u64 = 0;
    for ([_][]const u16{ &.{ 1, 2, 4, 3, 5 }, &.{ 6, 9, 10 } }) |grp| {
        const r = try serve(s, 0, grp);
        try expectServed(s, sb, r, grp);
        hits += r.plan.n_hits;
        s.release(r);
    }
    try s.flush();
    return .{ .hits = hits, .st = s.stats() };
}

/// sha256 of a served slot's logical record (its nine component rows).
fn slotDigest(s: *Stream, layer: u32, slot: u32, geom: *const Layer) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (geom.segments, 0..) |seg, c| h.update(s.slotRow(layer, slot, @enumFromInt(c))[0..seg.length]);
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

/// The phase-1 one-layer recorded trace on the real bank, the slot rows in `memory`.
fn realBankTrace(memory: SlotMemory) !void {
    const dir = std.mem.span(host_env.get(.DSV41_BANK) orelse return error.SkipZigTest);
    const fixture = std.mem.span(host_env.get(.DSV41_PHASE1_ROUTE_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.testing.io;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var diag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, dir, expert_bank.dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(64 << 20));
    defer a.free(text);
    const policy_fixtures = @import("policy_test.zig");
    const parsed = try std.json.parseFromSlice(struct { bank_trace: policy_fixtures.BankTrace }, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const bt = parsed.value.bank_trace;
    const L = bt.layer;
    var rows: [40]u32 = @splat(0);
    rows[L] = bt.prefill_rows;
    const s = try Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = bt.transient, .transient_rows = bt.transient, .slot_memory = memory });
    defer s.deinit();
    const geom = &bank.layers[L];
    var served: u64 = 0;
    try s.seedPrefill(L, bt.seed);
    for ([_][]const policy_fixtures.FixPlan{ bt.prefill, bt.routes }, 0..) |plans, phase| {
        if (phase == 1) {
            rows[L] = bt.decode_rows;
            try s.grow(&rows);
        }
        for (plans) |want| {
            const r = try serve(s, L, want.ids);
            try policy_fixtures.expectPlan(&r.plan, want);
            // Every served expert's slot holds its record: sha256 == the runtime manifest's.
            for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |e, slot| {
                const d = slotDigest(s, L, slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, e).logical, &d);
                served += 1;
            }
            for (r.plan.loadsOf()) |l| {
                const d = slotDigest(s, L, l.slot, geom);
                try testing.expectEqualSlices(u8, &bank.digest(L, l.expert).logical, &d);
                served += 1;
            }
            s.release(r);
        }
    }
    try s.flush();
    const st = s.stats();
    const ru = std.posix.getrusage(std.c.rusage.SELF);
    std.debug.print(
        "real bank layer {d} ({s} rows): {d} routes, {d} served slots sha256-checked; hits {d} misses {d} evictions {d} persistent {d} transient {d} skipped {d}; {d} B read in {d} preadv ({d:.3} s read, {d} ms wall); {d} ms total; peak RSS {d} B\n",
        .{ L, @tagName(memory), st.route_calls, served, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions, st.persistent_loads, st.transient_loads, st.loads_skipped, st.expert_bytes_read, st.preadv_calls, st.expert_read_seconds, @divTrunc(st.read_wall_ns, std.time.ns_per_ms), @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms), ru.maxrss },
    );
    try testing.expectEqual(st.expert_cache_misses, st.persistent_loads + st.transient_loads);
    try testing.expectEqual((st.expert_cache_misses - st.loads_skipped) * geom.logical_bytes, st.expert_bytes_read);
}

fn expectShape(arr: mlx.mlx_array, dtype: mlx.mlx_dtype, want: []const c_int) !void {
    try testing.expectEqual(dtype, mlx.mlx_array_dtype(arr));
    try testing.expectEqual(want.len, mlx.mlx_array_ndim(arr));
    try testing.expectEqualSlices(c_int, want, mlx.mlx_array_shape(arr)[0..want.len]);
}

// ── Lookahead, pre-read and event gates ──

const la_pool: expert_io.Options = .{ .workers = 2, .staging_bytes = 16384, .tickets = 512 };

/// One score row over 32 experts: `top` in descending order, the rest 0.
fn scoresFor(top: []const u16) [32]f32 {
    var s: [32]f32 = @splat(0);
    for (top, 0..) |e, i| s[e] = @floatFromInt(top.len - i);
    return s;
}

fn waitCounter(s: *Stream, which: expert_io.Counter, at_least: i64) !void {
    var t: u32 = 0;
    while (s.pool.counter(which) < at_least) : (t += 1) {
        if (t > 10_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn waitWord(s: *const Stream, value: u64) !void {
    const w = s.eventWord().?;
    var t: u32 = 0;
    while (@as(u64, @intCast(@atomicLoad(i64, w, .acquire))) < value) : (t += 1) {
        if (t > 20_000) return error.Timeout;
        std.Io.sleep(std.testing.io, .fromMicroseconds(500), .awake) catch {};
    }
}

/// Routes a call and, as the GPU would, waits on its gates instead of the pool. A gate also opens when its read fails
/// or the watchdog forces it, so before anyone reads the slots: a forced gate is refused by name (GateForced), then
/// every part is settled (no wait once its gate opened; a failed read raises ReadFailed by name).
fn serveGated(s: *Stream, layer: u32, ids: []const u16, scores: []const f32) !*Route {
    const r = try s.route(layer, ids, scores);
    if (try s.gate(r)) |g| {
        try waitWord(s, g.down_first + g.n_parts - 1);
        if (s.pool.counter(.ev_wd_forced) != 0) return error.GateForced;
        for (0..r.n_parts) |i| try s.waitDown(r, @intCast(i));
    }
    return r;
}

/// The bytes of `rows` rows of the widest layer's record (the transient scratch's row).
fn transientBytes(s: *const Stream, rows: u64) u64 {
    var n: u64 = 0;
    for (s.bank.layers[s.transient_layer].segments) |seg| n += seg.length;
    return rows * n;
}

/// One request up to its reverse change, on a stream of 4 prompt rows per layer: each layer's prompt route over
/// `prompt[l]` (the 4 persistent rows, then the scratch; its plan into `plans[l]`), the phase change to 6 rows per layer,
/// then decode's routes over `decode` on both layers. Every served slot holds its record.
fn promptAndDecode(s: *Stream, sb: *const SynthBank, prompt: [2][]const u16, decode: []const []const u16, plans: *[2]Plan) !void {
    for (prompt, 0..) |ids, l| {
        const r = try serve(s, @intCast(l), ids);
        try expectServed(s, sb, r, ids);
        plans[l] = r.plan;
        s.release(r);
    }
    try s.flush();
    _ = try s.releaseTransient();
    try s.grow(&.{ 6, 6 });
    for (decode) |ids| for (0..2) |l| {
        const r = try serve(s, @intCast(l), ids);
        try expectServed(s, sb, r, ids);
        s.release(r);
    };
    try s.flush();
}

const turn_opt: Options = .{ .rows = &.{ 4, 4 }, .transient_rows = 5 * max_route_ids, .wide_depth = 5, .pool = test_pool, .transient_release = true };
const turn_prompt = [2][]const u16{ &.{ 1, 2, 3, 4, 9, 10 }, &.{ 5, 6, 7, 8, 11 } };
const turn_decode = [_][]const u16{ &.{ 1, 12 }, &.{ 2, 13, 3 }, &.{ 14, 1, 15, 2 }, &.{ 16, 4, 1 } };

/// One route of the reverse change on the shared history (`promptAndDecode` on a fresh stream: the read pool is one per
/// process), then a continuation's prompt on each layer over two of the residents decode left in its prompt rows and two
/// experts never routed (20, 21). Checks the rows stay the bill's prompt rows throughout and every served slot's bytes.
const TurnOutcome = struct { left: [2][4]u16, freed: u64, regrown: u64, kept: u64, hits: [2]u32, misses: [2]u32, reads: [2]u64, bytes: [2]u64 };

fn continuationTurn(sb: *const SynthBank, residents: Residents) !TurnOutcome {
    const s = try Stream.init(testing.allocator, &sb.bank, turn_opt);
    defer s.deinit();
    var plans: [2]Plan = undefined;
    try promptAndDecode(s, sb, turn_prompt, &turn_decode, &plans);
    var out: TurnOutcome = undefined;
    // What decode left in each layer's prompt rows (slots 0..3, full).
    for (&out.left, s.layers) |*row, *ls| for (row, ls.policy.slot_to_expert[0..4]) |*e, x| {
        try testing.expect(x != expert_policy.no_expert);
        e.* = x;
    };
    out.freed = try s.shrink(&.{ 4, 4 }, residents);
    for (s.layers, 0..) |*ls, l| {
        // The bill's prompt rows only, whatever the route.
        try testing.expectEqual(@as(u32, 4), ls.policy.capacity);
        try testing.expectEqual(@as(u32, 4), ls.base.rows);
        try testing.expect(ls.ext == null);
        for (ls.meta[4..]) |m| try testing.expect(std.meta.eql(m, SlotMeta{}));
        try testing.expectEqual(@as(usize, 0), ls.policy.protected.count());
        switch (residents) {
            .forget => try testing.expectEqual(@as(u32, 0), ls.policy.occupancy),
            // Every resident decode left there, in its slot; the slot's meta its landed record's, unpinned.
            .keep => for (out.left[l], 0..) |e, slot| {
                try testing.expectEqual(@as(?u32, @intCast(slot)), ls.policy.slotOf(e));
                try testing.expect(std.meta.eql(ls.meta[slot], SlotMeta{ .state = .ready, .layer = @intCast(l), .expert = e }));
            },
        }
    }
    out.kept = s.residentCount();
    out.regrown = try s.regrowTransient();
    for (0..2) |l| {
        const ids = [_]u16{ out.left[l][1], 20, out.left[l][3], 21 };
        const b0 = s.stats().expert_bytes_read;
        const r = try serve(s, @intCast(l), &ids);
        try expectServed(s, sb, r, &ids);
        if (residents == .keep) {
            try testing.expectEqualSlices(u16, &.{ out.left[l][1], out.left[l][3] }, r.plan.hitsOf());
            try testing.expectEqualSlices(u16, &.{ 20, 21 }, r.plan.missesOf());
        }
        // A persistent load takes a prompt row (keep: a resident decode used least recently), the rest the scratch.
        for (r.plan.loadsOf()) |ld| try testing.expect(if (ld.persistent) ld.slot < 4 else ld.slot >= 4 and ld.slot < 4 + 5 * max_route_ids);
        out.hits[l] = r.plan.n_hits;
        out.misses[l] = r.plan.n_misses;
        out.reads[l] = readsOf(r);
        s.release(r);
        try s.flush();
        out.bytes[l] = s.stats().expert_bytes_read - b0;
    }
    for (s.layers) |*ls| try testing.expect(ls.base.rows == 4 and ls.ext == null and ls.policy.capacity == 4);
    return out;
}

/// One replay of the phase-2 fixture's layers 13 and 14 (served lookahead 8:inf:2, 4 chunks, pre-read, gates) at the
/// served depth 5, with the transient release on its route or not: each decode route's signature (layer, window, hits,
/// loads with their slots and reads) into `sigs`, every served slot sha256-checked; returns the stream's Stats.
fn replayLookahead(a: std.mem.Allocator, bank: *const expert_bank.Bank, f: anytype, sfd: std.c.fd_t, release: bool, sigs: *std.ArrayList(u64)) !Stats {
    const L: u32 = 13;
    var rows: [40]u32 = @splat(0);
    rows[L] = 3;
    rows[L + 1] = 3;
    const s = try Stream.init(a, bank, .{ .rows = &rows, .max_route_ids = 6, .transient_rows = 5 * 6, .wide_depth = 5, .lookahead = .{}, .event = .{ .watchdog_ms = 10_000 }, .transient_release = release });
    defer s.deinit();
    const geom = &bank.layers[L];
    for ([_]u32{ L, L + 1 }) |l| {
        var seed: [3]u16 = undefined;
        const sorted = try a.dupe(u16, f.resident0[l]);
        defer a.free(sorted);
        std.sort.pdq(u16, sorted, {}, std.sort.asc(u16));
        @memcpy(&seed, sorted[0..3]);
        try s.seedPrefill(l, &seed);
        s.release(try serve(s, l, &seed));
    }
    rows[L] = 4;
    rows[L + 1] = 4;
    if (release) _ = try s.releaseTransient();
    try s.grow(&rows);
    var routes: u64 = 0;
    var score_row: [384]f32 = undefined;
    var raw: [384 * 4]u8 = undefined;
    var at: u64 = 0;
    var cycle: usize = 0;
    outer: while (cycle < f.rows.len) : (cycle += 1) {
        const m = f.rows[cycle];
        const c13 = f.calls[cycle * f.layers + L];
        const c14 = f.calls[cycle * f.layers + L + 1];
        for (0..m) |r| {
            if (routes >= 120) break :outer;
            const off = (at + L * m + r) * 384 * 4;
            if (std.c.pread(sfd, &raw, raw.len, @intCast(off)) != raw.len) return error.ShortRead;
            for (&score_row, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
            for ([_]struct { l: u32, ids: []const u16, scores: []const f32 }{
                .{ .l = L, .ids = c13.ids[6 * r ..][0..6], .scores = &score_row },
                .{ .l = L + 1, .ids = c14.ids[6 * r ..][0..6], .scores = &.{} },
            }) |call| {
                const rt = try serveGated(s, call.l, call.ids, call.scores);
                var h = std.hash.Wyhash.init(call.l);
                h.update(std.mem.asBytes(&rt.window));
                h.update(std.mem.asBytes(&rt.plan.n_hits));
                for (rt.plan.loadsOf(), rt.reads[0..rt.plan.n_loads]) |ld, rd| {
                    h.update(std.mem.asBytes(&ld.expert));
                    h.update(std.mem.asBytes(&ld.slot));
                    h.update(std.mem.asBytes(&ld.persistent));
                    h.update(std.mem.asBytes(&rd));
                }
                try sigs.append(a, h.final());
                for (rt.plan.slotsOf(), call.ids) |slot, e| {
                    const d = slotDigest(s, call.l, slot, geom);
                    try testing.expectEqualSlices(u8, &bank.digest(call.l, e).logical, &d);
                }
                routes += 1;
                s.release(rt);
            }
        }
        at += @as(u64, m) * (f.layers - 1);
    }
    try s.flush();
    return s.stats();
}

/// The phase-2 fixture's calls for one layer as M = 1 routes, and the P1 scores of each row.
const TraceCall = struct { ids: []const u16, pre: []const u16, sel: []const []const u16, cand: []const []const u16 };

fn streamInitDeinit(a: std.mem.Allocator, sb: *const SynthBank, opt: Options) !void {
    const s = try Stream.init(a, &sb.bank, opt);
    s.deinit();
}
