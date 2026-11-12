//! MiMo's six-component MXFP4 bank on the shared SDK stream. The SDK owns
//! residency, asynchronous reads, windows, pinning and release; this bridge
//! maps absolute model layers and exposes GatherQmm's {w, scales} bindings.
const std = @import("std");
const env = @import("mlx_host").env;
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const bank_mod = @import("bank.zig");
const expert_policy = sdk_ext.expert.policy;
const profile = @import("profile.zig");
const route_trace = @import("route_trace.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const gpu_warm = @import("gpu_warm.zig");
const diag = @import("diag.zig");
/// MiMo's bridge over the shared SDK stream (`stream_source.For`): its bank module, GatherQmm's {w, scales} slot bindings
/// and its read pool's staging.
const Mimo = @import("stream_source.zig").For(bank_mod, .{ .Arrays = sdk_ext.quant.GatherQmm.Arrays, .staging_bytes = io_staging_bytes });
const S = Mimo.S;
pub const Stream = Mimo.Stream;
pub const Options = Mimo.Options;
pub const SlotMemory = Mimo.SlotMemory;
pub const LayerSlotBank = Mimo.LayerSlotBank;
pub const Route = Mimo.Route;
pub const Part = Mimo.Part;
pub const BankKind = Mimo.BankKind;
pub const SlotRef = Mimo.SlotRef;
pub const Error = Mimo.Error;
pub const Stats = Mimo.Stats;
pub const Gates = Mimo.Gates;
pub const max_route_ids = Mimo.max_route_ids;
pub const uses_reader = Mimo.uses_reader;
pub const BankArrays = bank_mod.BankArrays;
pub const BankArraysOf = bank_mod.BankArraysOf;

pub const io_workers: u32 = 4;
pub const io_staging_bytes: u64 = 9 << 20;
/// P1's read-ahead admits min(rows, (tickets - 2 * max_items) / 2) records (`Stream.readAheadSeed`): 512 covers
/// every MiMo row a layer can hold (248 >= 124 at the fill; 256 tickets admitted only 120).
pub const io_tickets: u32 = 512;
pub const records_per_part: u32 = 2;
pub const wide_depth: u8 = 2;
pub const transient_rows: u32 = wide_depth * max_route_ids;
pub const spec_threads: u32 = 2;
pub const spec_slots: u32 = 4;
/// Decode's transient rows under the phase-rows route (`Options.transient_release`): the boundary frees the prompt's
/// `transient_rows` and the grow allocates window 0 (`decodeRouteIds`) plus the stream's decode staging rows
/// (`Stream.grow`).
pub fn decodeTransientRows(window: DecodeWindow) u32 {
    return decodeRouteIds(window) + S.decode_staging_rows;
}
/// R3 (mimo/decode-rows-audit): decode's window 0, chosen once at construction (settings `"decode_window"`). `verify`
/// (the default) sizes it to the decode lane's widest route, `decode_lane_route_ids` = 4 rows x top-8 = 32 (a decode
/// plan's transient loads are its distinct non-admitted misses, at most its ids: `LayerPolicy.planWith`); `route` is
/// the control, the SDK's `max_route_ids` (48). Every decode-phase routed call is a decode-lane call (Module routes any
/// wider forward through the prompt phase first; `Source.forward` refuses a decode-phase wide call under `verify`).
pub const DecodeWindow = enum { verify, route, invalid };
/// The decode lane's widest route: `source.decode_lane_rows` (4: the MTP verify) x `source.top_k` (8), pinned there.
pub const decode_lane_route_ids: u32 = 4 * 8;
comptime {
    if (decode_lane_route_ids > max_route_ids) @compileError("mimo_v2: the decode lane's route would not fit one SDK route");
}
pub fn decodeRouteIds(window: DecodeWindow) u32 {
    return switch (window) {
        .verify => decode_lane_route_ids,
        .route, .invalid => max_route_ids,
    };
}
pub const lookahead = Mimo.lookahead;

pub const options = Mimo.options;

/// P7 (`settings.prefill_read_schedule`): the prompt pass's read schedule, fixed at construction. `current` (the control):
/// the stream's stock schedule. `early`: the stream's `overlap` schedule (`S.PromptReads`): every group of a prompt call
/// planned at its first route exactly as the stock schedule plans it, each read posted as soon as its row is free, the
/// read-ahead posted in rank order (head now, tail resolved by the plans: read once, the double reads dropped) and landed
/// only where a group needs it. Same records in the same rows before any kernel reads them: exact by construction.
/// `early_whole`: `early` with the WHOLE read-ahead posted with the attention, in rank order (`S.AheadPost.whole`):
/// p7-20261009 try 2 measured `early` moving ~0.84 s/pass of GPU idle into the router stage, the tail (ranks past
/// depth x width) being read only after the attention, at the call's plans, while 4 pool workers share the drive with
/// the call's misses; `whole` reads it during the attention instead, at the cost of the stock schedule's double reads
/// (records the plans evict before any hit). The same admissions, plans, slots and kernels: exact by construction.
/// `early_whole_trim` (X21, ahead-trim-20261010): `early_whole`, then at the plans the read-ahead jobs the reader has not
/// started yet are withdrawn from its queue (they wrote no row) and their records are resolved by the plans as `early`'s
/// tail is (`S.AheadPost.whole_trim`): read once if a group hits them first, never read if a plan evicts them first. At
/// the next9 clock (~76 ms to the plans on a sliding layer) the reader is still on the head: the tail (ranks 96+) is
/// queued and its evicted-first records, ~254-263 per 16K pass = ~3.5 GB, are not read; at the clamped clock (~120 ms)
/// every job has started and the schedule is `early_whole`'s. The same admissions, plans, slots and kernels: exact by
/// construction.
/// X20 `auto` (mimo/sched-auto): `early` below a prompt-length threshold, `early_whole` at or above it, chosen once per
/// request at its start (`AutoSchedule`, `StreamSource.chooseReadSchedule`); the stream is built once for both.
pub const ReadSchedule = enum { current, early, early_whole, early_whole_trim, auto, invalid };

/// X20: `auto`'s default threshold (settings `prefill_read_schedule_threshold`): shortprompt1-20261010 measured `early`
/// ahead of `early_whole` on every short prompt (codectx-4k 341.2/341.4 vs 316.3/318.0, codectx-1k 90.6/90.5 vs
/// 85.6/86.8, std-1k 89.1/89.4 vs 85.6/85.8 prefill tok/s) and P7 try 3 measured `early_whole` +7 % over `early` at 16K.
pub const default_read_schedule_threshold: u32 = 8192;
/// X20: the largest threshold a setting may name (the checkpoint's `max_position_embeddings`, 1,048,576).
pub const max_read_schedule_threshold: u32 = 1 << 20;

/// X20 `auto`: the request's schedule by its prompt length, a construction-time table of two exact routes that share
/// everything the stream is built with (the overlap schedule, its ticket ring, the stock admissions, the route triple)
/// and differ only in the read-ahead posted with the attention (`S.AheadPost`): `below` (`early`, the head) for a
/// prompt of fewer than `threshold` tokens, `at_or_above` (`early_whole`, the whole) otherwise. Consulted once per
/// request, at its start, never per layer or call.
pub const AutoSchedule = struct {
    threshold: u32 = default_read_schedule_threshold,

    pub const below: ReadSchedule = .early;
    pub const at_or_above: ReadSchedule = .early_whole;

    pub fn of(t: AutoSchedule, prompt_tokens: u64) ReadSchedule {
        return if (prompt_tokens < t.threshold) below else at_or_above;
    }
};

/// The `early` schedule's ticket ring: its posts run up to the read-ahead (<= 248 records) plus a whole call's loads
/// (<= 256) ahead of their consumers, 2 tickets each; the stock ring (`io_tickets`) is sized for one group's.
pub const io_tickets_early: u32 = 2048;

pub const applyReadSchedule = Mimo.applyReadSchedule;

const aheadPost = Mimo.aheadPost;

pub const eventOptions = Mimo.eventOptions;

const wavesOf = Mimo.wavesOf;

pub const traceBank = Mimo.traceBank;

pub const StreamSource = Mimo.StreamSource;

fn fixtureBindings(source: *StreamSource, g: *@import("../ops.zig").MlxOps) ![2][3]?BankArrays {
    var banks: [2][3]?BankArrays = undefined;
    for (&banks, 1..) |*layer, absolute| for ([_]BankKind{ .base, .ext, .transient }, 0..) |kind, i| {
        layer[i] = try source.bankArrays(g, @intCast(absolute), kind);
    };
    return banks;
}

fn checkFixtureCall(source: *StreamSource, g: *@import("../ops.zig").MlxOps, call: *StreamSource.Call, ids: []const u16, banks: *const [3]?BankArrays, bytes: []const u8) !void {
    const testing = std.testing;
    const r = call.route.?;
    const layer = &source.stream.bank.layers[r.layer];
    const sv = source.served(call);
    try testing.expectEqual(ids.len, sv.refs.len);
    for (0..sv.n_parts) |part| {
        try source.waitGu(call, @intCast(part));
        try source.waitDown(call, @intCast(part));
    }
    for (ids, r.plan.slotsOf()) |id, slot| {
        const offset = source.stream.bank.recordOffset(r.layer, id);
        for (layer.segments, 0..) |seg, component| {
            try testing.expectEqualSlices(u8, bytes[offset + seg.offset ..][0..seg.length], source.stream.slotRow(r.layer, slot, @fromBackingInt(@intCast(component))));
        }
    }
    for ([_]usize{ 0, ids.len - 1 }) |i| {
        const mark = g.mark();
        defer g.resetTo(mark);
        const ref = sv.refs[i];
        const b = banks[@backingInt(ref.bank)].?;
        const leaf = [_]mlx.mlx_array{ b.gate.w, b.gate.scales.?, b.up.w, b.up.scales.?, b.down.w, b.down.scales.? };
        const row = [_]u32{ref.row};
        const index = try g.hostArray(std.mem.sliceAsBytes(&row), &.{1}, .uint32);
        var gathered: [bank_mod.n_components]mlx.mlx_array = undefined;
        for (&gathered, leaf) |*out, array| out.* = try g.take(array, index, 0);
        try g.evalAll(&gathered);
        try mlx.check(mlx.mlx_synchronize(g.s));
        const offset = source.stream.bank.recordOffset(r.layer, ids[i]);
        for (gathered, layer.segments) |array, seg| {
            const data = mlx.mlx_array_data_uint8(array) orelse return error.MlxNoData;
            try testing.expectEqualSlices(u8, bytes[offset + seg.offset ..][0..seg.length], data[0..seg.length]);
        }
    }
}

fn phaseRowsBoundary(prompt_rows: [2]u32, decode_rows: [2]u32) !void {
    const testing = std.testing;
    const a = testing.allocator;
    const geom = try bank_mod.Geometry.init(32, 32, 64, 1, 2);
    var bytes: [2 * 64 * 1632]u8 = @splat(7);
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, &bytes);
    defer side.deinit();
    var opts = options(&prompt_rows, .host);
    opts.max_route_ids = 8;
    opts.transient_rows = 16;
    opts.transient_release = true; // the route `Source.initBank` installs with decode rows
    const stream = try Stream.init(a, &side.bank, opts);
    defer stream.deinit();
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.releaseTransient());
    try stream.grow(&decode_rows);
    for (stream.layers, decode_rows, prompt_rows) |*ls, d, p| {
        // The growth request: decode - prompt rows per layer, the residents' base rows unmoved.
        try testing.expectEqual(d - p, ls.ext.?.rows);
        try testing.expectEqual(p, ls.base.rows);
        try testing.expectEqual(d, ls.policy.capacity);
    }
    try testing.expectEqual(opts.max_route_ids + S.decode_staging_rows, stream.transient.rows);
    try testing.expectEqual(@as(u32, max_route_ids), decodeTransientRows(.route));
    try testing.expectEqual(@as(u32, 32), decodeTransientRows(.verify));
    // The reverse change frees the grown rows and window 0, then the prompt's scratch comes back.
    var grown: u64 = 0;
    for (decode_rows, prompt_rows) |d, p| grown += d - p;
    try testing.expectEqual(@as(u64, (grown + 8) * geom.record_bytes), try stream.shrink(&prompt_rows, .forget));
    try testing.expectEqual(@as(u64, 16 * geom.record_bytes), try stream.regrowTransient());
    for (stream.layers, prompt_rows) |*ls, p| {
        try testing.expect(ls.ext == null);
        try testing.expectEqual(p, ls.policy.capacity);
    }
    try testing.expectEqual(@as(u32, 16), stream.transient.rows);
}

// ── HITPASS CPU tests: a host-memory MiMo stream (layer 0 dense, two routed layers, 64 experts, 4 rows each) over a
// tmp sidecar; pool pread workers and host slot rows only. No MLX array, device or lock. ──

const HitFixture = struct {
    const geom_args = .{ 32, 32, 64, 1, 2 };
    bytes: []u8,
    side: bank_mod.TestSidecar,
    stream: *Stream,

    /// In place: the stream retains `&f.side.bank`, the SDK bank `&f.side.owner`.
    fn init(f: *HitFixture, a: std.mem.Allocator, rows: []const u32) !void {
        return f.initOpts(a, options(rows, .host));
    }

    fn initOpts(f: *HitFixture, a: std.mem.Allocator, opts: Options) !void {
        const geom = try @call(.auto, bank_mod.Geometry.init, geom_args);
        f.bytes = try a.alloc(u8, geom.total_bytes);
        errdefer a.free(f.bytes);
        for (0..2) |layer| for (0..64) |expert| for (geom.segments, 0..) |seg, component| {
            const offset = geom.recordOffset(@intCast(layer + 1), @intCast(expert)) + seg.offset;
            for (f.bytes[offset..][0..seg.length], 0..) |*b, i| b.* = @truncate(layer * 131 + expert * 7 + component * 3 + i);
        };
        try f.side.init(a, geom, f.bytes);
        errdefer f.side.deinit();
        f.stream = try Stream.init(a, &f.side.bank, opts);
    }

    fn deinit(f: *HitFixture, a: std.mem.Allocator) void {
        f.stream.deinit();
        f.side.deinit();
        a.free(f.bytes);
    }
};

// ── P7 read schedule CPU proofs (host stream over the HitFixture's sidecar: two routed layers, 64 experts; pool pread
// workers, host rows; no MLX array, device or lock). The executor's depth-2 wide sequence is emulated: read-ahead, the
// executor's await, seed, hot order, groups 0-1 routed, then per group its reads landed, every routed expert's row
// checked against the sidecar, released, group + 2 routed. ──

/// Groups of 11 over the fixture's 64 experts: at most 6, the stream's route ring (`installReadSchedule`).
const P7Width: u32 = 11;

/// One emulated prompt call's record: per group its plan, window and hit slots (as routed), and (early) how many groups
/// past the first two had every read posted before the executor routed them (the depth schedule posts them only then).
const P7Call = struct {
    plans: [S.route_capacity]expert_policy.Plan = undefined,
    windows: [S.route_capacity]u8 = undefined,
    hit_slots: [S.route_capacity][max_route_ids]u32 = undefined,
    n_groups: u32 = 0,
    posted_ahead: u32 = 0,
};

/// Every component row of `slot` (relative layer `rel`) holds expert `e`'s record bytes from the sidecar.
fn p7ExpectRecord(f: *HitFixture, rel: u32, slot: u32, e: u16) !void {
    const geom = f.side.owner.geometry;
    for (geom.segments, 0..) |seg, c| {
        const want = f.bytes[geom.recordOffset(rel + 1, e) + seg.offset ..][0..seg.length];
        const got = f.stream.slotRow(rel, slot, @fromBackingInt(@intCast(c)))[0..seg.length];
        try std.testing.expectEqualSlices(u8, want, got);
    }
}

/// The emulated executor over one prompt call of absolute layer `layer`: `ids` the call's routed ids (rows x top-k),
/// `predicted` the read-ahead's ranking. Checks every routed expert's bytes when its group is consumed.
fn p7RunCall(f: *HitFixture, ss: *StreamSource, layer: u32, ids: []const u16, predicted: []const u16, out: *P7Call) !void {
    return p7RunCallProbed(f, ss, layer, ids, predicted, out, null);
}

/// `p7RunCall` with `probe` run mid-call: with the read-ahead in flight, and with the call planned (its first route).
fn p7RunCallProbed(f: *HitFixture, ss: *StreamSource, layer: u32, ids: []const u16, predicted: []const u16, out: *P7Call, probe: ?*const fn (*HitFixture, *StreamSource) anyerror!void) !void {
    const rel = layer - 1;
    try ss.readAheadSeed(layer, predicted);
    if (probe) |pr| try pr(f, ss);
    try ss.awaitReadAhead(layer);
    try ss.seedPrefill(layer, ids);
    var count: [64]u32 = @splat(0);
    var distinct_buf: [64]u16 = undefined;
    var nd: usize = 0;
    for (ids) |e| {
        if (count[e] == 0) {
            distinct_buf[nd] = e;
            nd += 1;
        }
        count[e] += 1;
    }
    const distinct = distinct_buf[0..nd];
    ss.hotOrder(layer, distinct, &count);
    const n_groups: u32 = @intCast((nd + P7Width - 1) / P7Width);
    out.n_groups = n_groups;
    const group = struct {
        fn of(d: []const u16, g: u32) []const u16 {
            return d[g * P7Width .. @min((g + 1) * P7Width, d.len)];
        }
    }.of;
    var calls: [2]?*StreamSource.Call = .{ null, null };
    for (0..@min(2, n_groups)) |g| calls[g] = try ss.route(layer, group(distinct, @intCast(g)), &.{});
    if (probe) |pr| try pr(f, ss);
    for (0..n_groups) |gi| {
        const g: u32 = @intCast(gi);
        const call = calls[g % 2].?;
        const r = call.route.?;
        out.plans[g] = r.plan;
        out.windows[g] = r.window;
        @memcpy(out.hit_slots[g][0..r.plan.n_hits], r.hit_slots[0..r.plan.n_hits]);
        for (0..r.n_parts) |pi| {
            try ss.waitGu(call, @intCast(pi));
            try ss.waitDown(call, @intCast(pi));
        }
        // The group's consumers: every routed expert's row holds its record.
        for (group(distinct, g)) |e| {
            const slot = blk: {
                for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |h, hs| if (h == e) break :blk hs;
                for (r.plan.loadsOf()) |l| if (l.expert == e) break :blk l.slot;
                return error.P7TestExpertNotServed;
            };
            try p7ExpectRecord(f, rel, slot, e);
        }
        ss.release(call);
        calls[g % 2] = null;
        if (g + 2 < n_groups) {
            // Issue order: group g + 2's reads already posted (all its parts) before the executor routes it.
            if (ss.early) out.posted_ahead += @intFromBool(ss.stream.p7.post_g > g + 2);
            calls[g % 2] = try ss.route(layer, group(distinct, g + 2), &.{});
        }
    }
    try ss.flush();
}

/// A call's routed ids: `rows` tokens x 8 distinct experts from a skewed draw over `span` experts (seeded).
fn p7Ids(rng: std.Random, rows: usize, span: u16, out: []u16) []u16 {
    var n: usize = 0;
    for (0..rows) |_| {
        var picked: [8]u16 = undefined;
        var k: usize = 0;
        while (k < 8) {
            // Skew: a squared uniform favours low ids (hot experts), so groups differ in reads and hits.
            const u = rng.float(f32);
            const e: u16 = @intFromFloat(@min(@as(f32, @floatFromInt(span - 1)), u * u * @as(f32, @floatFromInt(span))));
            if (std.mem.indexOfScalar(u16, picked[0..k], e) != null) continue;
            picked[k] = e;
            k += 1;
        }
        @memcpy(out[n..][0..8], &picked);
        n += 8;
    }
    return out[0..n];
}

/// The reader a P7 script runs on: its workers, and (X21) what keeps it behind the call's plans: a sleep injected on each
/// call's first read-ahead read (`hold_ns`) or a random delay before every range read (`delay_ns`, at most); `relocate`:
/// the stream built with the overlap schedule's relocation (`Options.prompt_relocate`, glm5_next's route; MiMo's off).
const P7Pool = struct { workers: u32 = io_workers, hold_ns: i64 = 0, delay_ns: i64 = 0, relocate: bool = false };

fn p7Fixture(f: *HitFixture, a: std.mem.Allocator, rows: []const u32, schedule: ReadSchedule) !StreamSource {
    return p7FixtureOn(f, a, rows, schedule, .{});
}

fn p7FixtureOn(f: *HitFixture, a: std.mem.Allocator, rows: []const u32, schedule: ReadSchedule, pool: P7Pool) !StreamSource {
    var opts = options(rows, .host);
    opts.max_route_ids = P7Width;
    opts.transient_rows = wide_depth * P7Width;
    opts.pool.workers = pool.workers;
    try applyReadSchedule(&opts, schedule);
    opts.prompt_relocate = pool.relocate;
    try f.initOpts(a, opts);
    errdefer f.deinit(a);
    var ss = StreamSource.init(f.stream);
    try ss.installReadSchedule(schedule, default_read_schedule_threshold);
    return ss;
}

fn p7ExpectSamePlans(want: *const P7Call, got: *const P7Call) !void {
    const testing = std.testing;
    try testing.expectEqual(want.n_groups, got.n_groups);
    for (0..want.n_groups) |g| {
        const x = &want.plans[g];
        const y = &got.plans[g];
        try testing.expectEqualSlices(u32, x.slotsOf(), y.slotsOf());
        try testing.expectEqualSlices(u16, x.hitsOf(), y.hitsOf());
        try testing.expectEqualSlices(u16, x.missesOf(), y.missesOf());
        try testing.expectEqual(x.n_loads, y.n_loads);
        for (x.loadsOf(), y.loadsOf()) |lx, ly| try testing.expect(lx.expert == ly.expert and lx.slot == ly.slot and lx.persistent == ly.persistent);
        try testing.expectEqual(x.n_evictions, y.n_evictions);
        for (x.evictionsOf(), y.evictionsOf()) |ex, ey| try testing.expect(ex.slot == ey.slot and ex.previous == ey.previous and ex.next == ey.next);
        try testing.expectEqual(want.windows[g], got.windows[g]);
        try testing.expectEqualSlices(u32, want.hit_slots[g][0..x.n_hits], got.hit_slots[g][0..y.n_hits]);
    }
}

/// One schedule's run over a fixed list of prompt calls (the pool is one per process: the two schedules run in turn).
const P7Run = struct {
    calls: []P7Call,
    stats: Stats,
    residents: [2][64]u16,
    /// X20: per request, the schedule its start chose (`chooseReadSchedule`) and its own bytes read and records dropped.
    chosen: [p7_max_requests]ReadSchedule = undefined,
    req_bytes: [p7_max_requests]u64 = undefined,
    req_dropped: [p7_max_requests]u64 = undefined,
    n_requests: usize = 0,
};
const p7_max_requests = 16;

/// The call list: per prompt, per routed layer, its routed ids and the read-ahead's ranking.
const P7Script = struct {
    ids: [][]const u16,
    pred: [][]const u16,
    /// A prompt boundary (residents forgotten, prompt counts zeroed) after every `per_prompt` calls.
    per_prompt: usize,
    /// X20: request r's prompt length (`prompt_tokens[r % len]`), handed to `chooseReadSchedule` at its start: `auto`
    /// chooses by it, a named schedule ignores it.
    prompt_tokens: []const u64 = &.{16_384},
};

fn p7RunScript(a: std.mem.Allocator, schedule: ReadSchedule, rows: []const u32, script: P7Script, out: *P7Run) !void {
    return p7RunScriptOn(a, schedule, rows, script, out, .{});
}

/// X21: an injected sleep on the call's first read-ahead read (the first predicted record not resident: the read-ahead's
/// first admission, its gate/up range's aligned offset), so a one-worker reader is still on the read-ahead's first job
/// when the call's plans exist and every later job is queued.
fn p7HoldFirstRead(f: *HitFixture, rel: u32, pred: []const u16, ns: i64) void {
    const pol = &f.stream.layers[rel].policy;
    for (pred) |e| if (pol.slotOf(e) == null) {
        const page: u64 = std.heap.pageSize();
        sdk_ext.expert.io.injectFault(f.stream.bank.spans(rel, e).gu_offset / page * page, 5, ns);
        return;
    };
}

fn p7RunScriptOn(a: std.mem.Allocator, schedule: ReadSchedule, rows: []const u32, script: P7Script, out: *P7Run, pool: P7Pool) !void {
    var f: HitFixture = undefined;
    var ss = try p7FixtureOn(&f, a, rows, schedule, pool);
    defer f.deinit(a);
    const inject = sdk_ext.expert.io.test_abi.abi;
    if (pool.delay_ns > 0) inject.q3ld_test_delay(0x7e7a11, pool.delay_ns);
    defer if (pool.delay_ns > 0) inject.q3ld_test_delay(0, 0);
    defer if (pool.hold_ns > 0) sdk_ext.expert.io.clearFaults();
    out.n_requests = 0;
    var at_start: Stats = .{};
    for (script.ids, script.pred, 0..) |ids, pred, i| {
        if (i % script.per_prompt == 0) {
            // X20: a request's start (nothing live: construction, or the boundary below): its schedule, once.
            const r = out.n_requests;
            if (r == p7_max_requests) return error.P7TestTooManyRequests;
            out.chosen[r] = try ss.chooseReadSchedule(script.prompt_tokens[r % script.prompt_tokens.len]);
            out.n_requests += 1;
            at_start = ss.stats();
        }
        const layer: u32 = @intCast(1 + i % 2);
        if (pool.hold_ns > 0) p7HoldFirstRead(&f, layer - 1, pred, pool.hold_ns);
        out.calls[i] = .{};
        try p7RunCall(&f, &ss, layer, ids, pred, &out.calls[i]);
        // X20: every call of a request runs the post its start chose (an overlap stream; a stock stream has no post).
        if (f.stream.prompt_reads == .overlap) try std.testing.expectEqual(aheadPost(out.chosen[out.n_requests - 1]), f.stream.prompt_ahead);
        if ((i + 1) % script.per_prompt == 0) {
            // A request's start as `Source.beginPrefill` makes it: routes and the read-ahead settled, then forgotten.
            try ss.stream.settleRoutes();
            _ = try ss.forgetResidents();
            ss.resetPromptCounts();
            const now = ss.stats();
            out.req_bytes[out.n_requests - 1] = now.expert_bytes_read - at_start.expert_bytes_read;
            out.req_dropped[out.n_requests - 1] = now.ahead_dropped - at_start.ahead_dropped;
        }
    }
    out.stats = ss.stats();
    for (f.stream.layers, &out.residents) |*ls, *res| @memcpy(res, ls.policy.slot_to_expert[0..64]);
}

/// The first P7 proof's 40 calls (seeded 0x7e7a11, rows 28): calls alternate layers 1 and 2, expert spans cycle over
/// 24..44 (3-4 groups of 11, under the fixture's 64); the read-ahead's ranking is the call's true hot order with
/// neighbours swapped, then (odd calls) a few cold experts pulled into the admitted ranks, head or tail (mispredictions:
/// drops, read-ahead rows evicted, seed misses).
const P7Calls40 = struct {
    const n_calls = 40;
    ids_store: [n_calls][64 * 8]u16 = undefined,
    pred_store: [n_calls][64]u16 = undefined,
    ids: [n_calls][]const u16 = undefined,
    pred: [n_calls][]const u16 = undefined,

    /// In place (the slices point into the stores).
    fn fill(c: *P7Calls40, rows: u32) void {
        var prng = std.Random.DefaultPrng.init(0x7e7a11);
        const rng = prng.random();
        for (0..n_calls) |i| {
            const span: u16 = @intCast(24 + 4 * ((i / 2) % 6));
            const x = p7Ids(rng, 40, span, &c.ids_store[i]);
            c.ids[i] = x;
            var count: [64]u32 = @splat(0);
            for (x) |e| count[e] += 1;
            var ranked: [64]u16 = undefined;
            const hot = expert_policy.rankHottest(&count, &ranked);
            @memcpy(c.pred_store[i][0..hot.len], hot);
            var k: usize = 1;
            while (k + 1 < hot.len) : (k += 3) std.mem.swap(u16, &c.pred_store[i][k], &c.pred_store[i][k + 1]);
            if (i % 2 == 1 and hot.len > rows) for (0..4) |_| {
                const a_pos = rng.uintLessThan(usize, rows);
                const b_pos = rows + rng.uintLessThan(usize, hot.len - rows);
                std.mem.swap(u16, &c.pred_store[i][a_pos], &c.pred_store[i][b_pos]);
            };
            c.pred[i] = c.pred_store[i][0..hot.len];
        }
    }
};

/// The crafted call of the "dropped" test: 40 experts on 40 rows (equal counts: the hot order is the id order), and a
/// read-ahead ranking whose ranks 22-27 are six cold experts (true ranks 34-39) that group 2's misses evict before group 3
/// routes them.
fn p7CraftedCall(ids_buf: *[64 * 8]u16, pred_buf: *[64]u16) struct { ids: []const u16, pred: []const u16 } {
    var n: usize = 0;
    for (0..40) |row| {
        for (0..8) |k| {
            ids_buf[n] = @intCast((row + k * 5) % 40);
            n += 1;
        }
    }
    var count: [64]u32 = @splat(0);
    for (ids_buf[0..n]) |e| count[e] += 1;
    var ranked: [64]u16 = undefined;
    const hot = expert_policy.rankHottest(&count, &ranked);
    @memcpy(pred_buf[0..22], hot[0..22]);
    @memcpy(pred_buf[22..28], hot[34..40]);
    @memcpy(pred_buf[28..40], hot[22..34]);
    return .{ .ids = ids_buf[0..n], .pred = pred_buf[0..40] };
}

/// One record's bytes in the P7 fixture's bank.
fn p7RecordBytes() !u64 {
    const geom = try @call(.auto, bank_mod.Geometry.init, HitFixture.geom_args);
    return geom.record_bytes;
}

/// The relocation proofs' second crafted call: 50 experts on 50 rows (equal counts: the hot order is the id order; five
/// groups of 11), and a read-ahead ranking whose ranks 22-27 are group 4's six experts (true ranks 44-49). Group 2's seed
/// misses evict them; group 4 (window 0) loads them into transient rows 0-5, of which group 2 (window 0) loaded rows 0-4
/// first: only true rank 49's row (5) has no earlier reader.
fn p7CraftedCall50(ids_buf: *[64 * 8]u16, pred_buf: *[64]u16) struct { ids: []const u16, pred: []const u16 } {
    var n: usize = 0;
    for (0..50) |row| {
        for (0..8) |k| {
            ids_buf[n] = @intCast((row + k * 5) % 50);
            n += 1;
        }
    }
    var count: [64]u32 = @splat(0);
    for (ids_buf[0..n]) |e| count[e] += 1;
    var ranked: [64]u16 = undefined;
    const hot = expert_policy.rankHottest(&count, &ranked);
    @memcpy(pred_buf[0..22], hot[0..22]);
    @memcpy(pred_buf[22..28], hot[44..50]);
    @memcpy(pred_buf[28..50], hot[22..44]);
    return .{ .ids = ids_buf[0..n], .pred = pred_buf[0..50] };
}

// ── X20 (mimo/sched-auto) CPU proofs: `prefill_read_schedule: auto` on the same host stream fixture. ──

/// X20: a choice past a request's start refuses (`Stream.setPromptAhead`) and leaves the request's post, either length.
fn p7NoChoice(f: *HitFixture, ss: *StreamSource) anyerror!void {
    const post = f.stream.prompt_ahead;
    for ([_]u64{ 1023, 16_384 }) |n| {
        if (ss.chooseReadSchedule(n)) |_| return error.P7TestChoiceMidRequest else |e| if (e != error.RoutesLive and e != error.PromptStarted) return e;
    }
    try std.testing.expectEqual(post, f.stream.prompt_ahead);
}

fn packedRef(ref: SlotRef) u32 {
    return (@as(u32, @backingInt(ref.bank)) << 24) | ref.row;
}

/// One call's route, its parts landed, released and flushed (the scheduler's sequence, without math).
fn hitRoute(ss: *StreamSource, layer: u32, ids: []const u16) !void {
    const call = try ss.route(layer, ids, &.{});
    for (0..call.route.?.n_parts) |p| {
        try ss.waitGu(call, @intCast(p));
        try ss.waitDown(call, @intCast(p));
    }
    ss.release(call);
    try ss.flush();
}

/// The LUT as the policy defines it: each resident expert's persistent slot (packed), else 0.
fn expectLutIsPolicy(ss: *const StreamSource, layer: u32, lut: []const u32) !void {
    const rel = ss.relativeLayer(layer);
    const pol = &ss.stream.layers[rel].policy;
    for (lut, 0..) |v, e| {
        const want: u32 = if (pol.slotOf(@intCast(e))) |s| (if (s < pol.capacity) packedRef(ss.stream.slotRef(rel, s)) else 0) else 0;
        try std.testing.expectEqual(want, v);
    }
}

/// The scheduler's math on the trace backend: MiMo's projection type, the banked waves' shapes only.
const HitTraceMath = struct {
    pub const has_banked = true;
    pub const decode_forward_rows: u32 = 1;
    pub fn Projection(comptime A: type) type {
        return sdk_ext.quant.GatherQmm.Arrays(A);
    }
    hidden: c_int,
    inter: c_int,
    const Tr = @import("../ops.zig").TraceOps;
    pub fn gateUp(self: *const HitTraceMath, g: *Tr, x: u32, _: u32, _: Projection(u32), _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn down(self: *const HitTraceMath, g: *Tr, h: u32, _: u32, _: Projection(u32)) !u32 {
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
    pub fn gateUpBanked(self: *const HitTraceMath, g: *Tr, x: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(x).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(x).d[0], self.inter }, .bfloat16);
    }
    pub fn downBanked(self: *const HitTraceMath, g: *Tr, h: u32, ids: u32, _: anytype) !u32 {
        if (g.shapeOf(ids).d[0] != g.shapeOf(h).d[0]) return error.MathShape;
        return g.kernel(&.{ g.shapeOf(h).d[0], self.hidden }, .bfloat16);
    }
};

const HitScript = struct {
    calls: []const []const u16,
    next: usize = 0,
    fn values(self: *HitScript) HitTraceMath.Tr.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const self: *HitScript = @ptrCast(@alignCast(ctx));
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


// ── end HITPASS CPU tests ──

/// Lever 2's synthetic prompt (8 tokens x top-8 over 64 experts): six tokens route {0..7}, the last two {8..11, 4, 5,
/// 12, 13}. Whole counts: 4, 5 -> 8; 0-3, 6, 7 -> 6; 8-13 -> 2. Last two tokens: 4, 5, 8-13 -> 2.
const seed_prompt = blk: {
    var ids: [8 * 8]u16 = undefined;
    for (0..6) |t| for (0..8) |j| {
        ids[t * 8 + j] = j;
    };
    for (6..8) |t| for ([_]u16{ 8, 9, 10, 11, 4, 5, 12, 13 }, 0..) |e, j| {
        ids[t * 8 + j] = e;
    };
    break :blk ids;
};

// Host-memory stream (two routed layers, 64 experts, 4 rows): the prompt pass routes each layer's experts in the
// installed hot order, so decode starts with exactly the installed seed. CPU only: host slot rows and pread workers.
