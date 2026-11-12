//! turboexl3_ling's own (not a copy): Ling-3.1-flash's routed experts from its EXL3 bank (`experts.bin` +
//! `expert-manifest-v2.json`), mlx-serve's `exl3_source.zig` with the EXL3 lane's math replaced by turboexl3's (the
//! plugin's src/ling_math.zig, through `mlx_host`): the bank at Ling's geometry (hidden 4096, expert width 1280, 512
//! experts, K 2 or 2.5 per layer), the same stream and schedule (event-gated waves, the next layer's reads predicted from
//! its router, the wide prompt calls' feed, banked decode waves), each layer's SwiGLU its own. Slot rows per phase: the
//! prompt rows during a prompt pass, and after `phase(.decode)` the decode rows each request's plan gives (RQ, the
//! plugin's src/ling_rq.zig), changed in place by a trim (`trimDecode`).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const xp = @import("experts.zig");
const expert_bank = @import("expert/bank.zig");
const expert_stream = @import("expert/stream.zig");
const expert_io = @import("sdk_ext/expert/io.zig");
const sdk_ext = @import("sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const expert_admission = @import("expert/admission.zig");
const ling_math = @import("mlx_host").ling_math;
/// A prompt call's routed combine, COMBINE (the plugin's src/ling_combine.zig).
const ling_combine = @import("mlx_host").ling_combine;
/// The MoE tail in one launch: a decode call's joined rows and weights for the layer's tail (the plugin's
/// src/ling_tail.zig).
const ling_tail = @import("mlx_host").ling_tail;
/// The MoE tail at prompt widths: a layer-major call's unjoined outputs, each pair's (source, row) and the weights for
/// its spans' tails (the plugin's src/ling_pf_tail.zig).
const ling_pf_tail = @import("mlx_host").ling_pf_tail;
/// The run-path line's marks (the plugin's src/ling_ran.zig): each served path below marks itself where it runs.
const ran = @import("mlx_host").ling_ran;
/// RQ (the plugin's src/ling_rq.zig): the bank's rows by phase, and per request the decode rows its plans give.
const rq = @import("mlx_host").ling_rq;
const G = ops.MlxOps;
const T = mlx.mlx_array;

const Math = ling_math.Math(G, xp.ProjOf(T), xp.BankArraysOf(T), sdk_ext.quant.PrefillRows);
const Hook = xp.ExpertsWith(G, xp.StreamSource, Math, .{ .prefill = true, .lookahead = true, .gated = true });
pub const Gate = Hook.Gate;
pub const Phase = enum { prompt, decode };

/// The trunk's MoE geometry the bank must match. `limits`: each routed layer's SwiGLU clamp (0 none), bank order.
pub const Geometry = struct { hidden_size: u32, moe_intermediate_size: u32, n_experts_per_tok: u32, n_routed_experts: u32, n_layers: u32, swiglu_limit: f64 = 0, limits: []const c_int, n_group: u32 = 0, topk_group: u32 = 0 };

pub const Budget = rq.Budget;

/// What the source holds, by term, at its rows.
pub const Bill = struct {
    row_bytes: u64,
    prompt_windows: u64,
    decode_windows: u64,
    staging: u64,
    rows: rq.Rows,

    pub fn prompt(b: Bill) u64 {
        return b.row_bytes * b.rows.prompt + b.prompt_windows + b.staging;
    }

    pub fn decode(b: Bill) u64 {
        return b.row_bytes * b.rows.decode + b.decode_windows + b.staging;
    }

    /// The terms RQ's plans and the bank's rows read (src/ling_rq.zig `Bank`), over `n_experts` experts a layer.
    pub fn terms(b: Bill, n_experts: u32) rq.Bank {
        return .{ .row_bytes = b.row_bytes, .prompt_windows = b.prompt_windows, .decode_windows = b.decode_windows, .staging = b.staging, .n_experts = n_experts };
    }
};

const wide_depth: u8 = 5;
/// The decode read-ahead (LOOKAHEAD_ROWS): a call reads ahead 2 records a row, up to 8 (a verify's rows each predict
/// their own next experts); 16 staging records. Residency only: the same bytes.
const lookahead: expert_stream.Lookahead = .{ .k = 8, .tau = std.math.inf(f32), .budget = 8, .budget_per_row = 2 };
/// The read-ahead picks by Ling's router rule (PREDICTOR): its group limit over the predictor's sigmoid + bias scores
/// (`Math.predictorScores`). Residency only: the same bytes.
fn lookaheadFor(geo: Geometry) expert_stream.Lookahead {
    var la = lookahead;
    if (geo.n_group > 1) {
        la.n_group = geo.n_group;
        la.topk_group = geo.topk_group;
    }
    return la;
}
/// The expert read pool's workers (READS_WIDE): 8, the C pool's most (`expert_io.max_workers`), not its default 4
/// (`expert_io.Options`), so more of a prompt chunk's expert reads are in flight at once. Residency only: every
/// record's bytes are the same.
const read_workers: u32 = 8;
const event_watchdog_ms = 2000;
const wide: xp.Wide = .{ .seed = true, .hot_first = true, .depth = wide_depth, .defer_base = true, .read_ahead = true, .base_at_seed = true, .seed_aligned = true };

/// The bank geometry turboexl3 decodes here: mul1, K 2 or K 2.5 per layer (32 or 40 halfwords a tile).
fn implementedOf(geo: Geometry) expert_bank.Implemented {
    return .{ .codebooks = &.{"mul1"}, .k = &.{2}, .frac_n = &.{40}, .hidden = geo.hidden_size, .inter = geo.moe_intermediate_size, .n_experts = geo.n_routed_experts, .n_layers = geo.n_layers };
}

/// The bank in `dir` read from its manifests (no slot memory allocated): its record bytes, the largest a layer.
pub fn recordBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, geo: Geometry) !u64 {
    var bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
    defer bank.deinit();
    var row: u64 = 0;
    for (bank.layers) |l| row += l.logical_bytes;
    return row;
}

/// The rows `budget` holds over the bank in `dir`, by term (no slot memory allocated).
pub fn billOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, geo: Geometry, budget: Budget) !Bill {
    var bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
    defer bank.deinit();
    return billBank(&bank, budget);
}

fn billBank(bank: *const expert_bank.Bank, budget: Budget) !Bill {
    var row: u64 = 0;
    var record: u64 = 0;
    var alt: u64 = 0;
    for (bank.layers) |l| {
        row += l.logical_bytes;
        record = @max(record, l.logical_bytes);
    }
    for (bank.layers) |l| if (l.logical_bytes != record) {
        alt = @max(alt, l.logical_bytes);
    };
    const window = sdk_ext.expert.policy.max_route_ids * (record + alt);
    const release = expert_stream.phase_change_releases_wide_windows and expert_stream.transient_release_default;
    var b: Bill = .{
        .row_bytes = row,
        .prompt_windows = wide_depth * window,
        .decode_windows = if (release) window else wide_depth * window,
        // turboexl3_ling: and the read pool's staging buffers past its default workers (the stock bill counts those), one
        // a worker.
        .staging = expert_admission.lookaheadCharge(record, 2 * lookahead.budget, std.heap.pageSize()) +
            @as(u64, read_workers - (expert_io.Options{}).workers) * (expert_io.Options{}).staging_bytes,
        .rows = undefined,
    };
    // Each phase's whole rows past its windows and the staging (src/ling_rq.zig `rowsOf`, whose arithmetic RQ's plans
    // take per request).
    b.rows = try rq.rowsOf(b.terms(bank.n_experts), budget);
    return b;
}

/// Routed ids a decode-phase call takes (the stream's route width); a wider call is a prompt pass.
pub const decode_route_ids: usize = sdk_ext.expert.policy.max_route_ids;

/// RQ (the plugin's src/ling_rq.zig): a phase change's or a trim's account, for its RQ line: the bank's rows before and
/// after it (decode's, or the prompt's base), its frees' time with MLX's synchronize and cache clear, its allocation's
/// time, and whether RQ's planner gave the rows.
pub const Change = struct { rows_before: u32 = 0, rows_after: u32 = 0, free_ns: u64 = 0, alloc_ns: u64 = 0, planned: bool = false };

/// RQ's plan at a phase change or a trim (plugin.zig `rqPlan`), run where the bank holds its prompt rows alone: its
/// grown rows and decode's window, or the prompt's windows, freed, MLX synchronized and its cache cleared, so MLX's
/// memory read there holds nothing of the bank's past its base. It gives the decode rows to grow to, or null for the
/// rows the bank opened with (no request has started: the load's warm-up). At a prompt start its rows are not used:
/// stage 1 keeps the prompt rows at the base.
pub const Planner = struct {
    ctx: *anyopaque,
    plan: *const fn (ctx: *anyopaque) ?u32,
};

/// A monotonic clock reading (ns), for RQ's free and allocation times.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Source = struct {
    a: std.mem.Allocator,
    g: G,
    bank: expert_bank.Bank,
    gates: []Gate,
    limits: []c_int,
    state: Math.State,
    first_layer: u32,
    prompt_rows: []u32,
    decode_rows: []u32,
    stream: *expert_stream.Stream,
    source: xp.StreamSource,
    hook: Hook,
    bill: Bill,
    phase: Phase,
    released: bool,
    /// The stream's counters at the current phase's start, and its decode steps so far (`[stream-stats]` lines).
    mark: expert_stream.Stats,
    steps: u32,
    /// D2a (PROMPT_AHEAD_FILL): this prompt pass started with `promptStart` (the span pass's forget), so its fill
    /// read-aheads are PROMPT_AHEAD_FILL's; a phase change ends it.
    fill_pass: bool,

    /// The bank in `dir` at the rows `budget` holds, streamed on `s` (the trunk's stream). `gates`: every routed layer's
    /// router (borrowed for the source's life); `first_layer`: the model layer of bank layer 0.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, geo: Geometry, gates: []const Gate, first_layer: u32, budget: Budget) !*Source {
        if (geo.limits.len != geo.n_layers) return error.LingLimitsPerLayer;
        if (comptime expert_stream.uses_reader) try sdk_ext.expert.takeReader();
        errdefer if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
        const self = try a.create(Source);
        errdefer a.destroy(self);
        // Every field in one literal (none has a declared default), so a field left out does not compile. The ones set
        // below, in the order their values depend on each other, start undefined here.
        self.* = .{
            .a = a,
            .g = undefined,
            .bank = undefined,
            .gates = undefined,
            .limits = undefined,
            .state = undefined,
            .first_layer = first_layer,
            .prompt_rows = undefined,
            .decode_rows = undefined,
            .stream = undefined,
            .source = undefined,
            .hook = undefined,
            .bill = undefined,
            .phase = .prompt,
            .released = false,
            .mark = .{},
            .steps = 0,
            .fill_pass = false,
        };
        self.bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
        errdefer self.bank.deinit();
        self.bill = try billBank(&self.bank, budget);
        self.gates = try a.dupe(Gate, gates);
        errdefer a.free(self.gates);
        self.limits = try a.dupe(c_int, geo.limits);
        errdefer a.free(self.limits);
        self.state = .{ .limits = self.limits };
        self.prompt_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.prompt_rows);
        @memset(self.prompt_rows, self.bill.rows.prompt);
        self.decode_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.decode_rows);
        @memset(self.decode_rows, self.bill.rows.decode);
        self.g = try G.init(a, s);
        errdefer self.g.deinit();
        try ling_math.configure();
        const event = try expert_event.createMetal();
        self.stream = try expert_stream.Stream.init(a, &self.bank, .{
            .rows = self.prompt_rows,
            .slot_memory = .{ .mlx = s },
            .lookahead = lookaheadFor(geo),
            // turboexl3_ling: the stream's own pool default (1024 tickets) at `read_workers` workers.
            .pool = .{ .tickets = 1024, .workers = read_workers },
            .event = .{ .backend = .{ .metal = event.object }, .watchdog_ms = event_watchdog_ms },
            .wide_depth = wide_depth,
            .transient_rows = @as(u32, wide_depth) * sdk_ext.expert.policy.max_route_ids,
            .transient_release = expert_stream.phase_change_releases_wide_windows and expert_stream.transient_release_default,
        });
        errdefer self.stream.deinit();
        // D2a (PROMPT_AHEAD_FILL): the read-ahead admits the prediction's top as many as the layer's rows.
        self.stream.ahead_fill = true;
        self.source = xp.StreamSource.init(self.stream);
        // D2d (PROMPT_KEEP): no prefill miss takes the row of a resident its call routes.
        for (self.stream.layers) |*ls| ls.policy.keep_routed = true;
        // D2c (PROMPT_PARK): D2's read-ahead parks the next layer's records past its rows.
        try self.stream.enablePark();
        // DEVROUTE (exact): each decode call's hit wave a device graph behind the routing barrier.
        self.hook = try Hook.initWith(a, &self.g, &self.source, .{ .state = &self.state }, geo, .{ .gates = self.gates, .event = event, .wide = wide, .banked = true, .hoist_first = true, .devroute = true });
        errdefer self.hook.deinit();
        const w = try self.hook.math.warm(&self.g, self.hook.banks, @intCast(geo.hidden_size));
        std.log.info("[turboexl3_ling] turboexl3's routed kernels compiled at load: {d} calls in {d} ms", .{ w.calls, w.ms });
        return self;
    }

    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.hook.deinit();
        self.stream.deinit();
        self.g.deinit();
        self.bank.deinit();
        self.a.free(self.gates);
        self.a.free(self.limits);
        self.a.free(self.prompt_rows);
        self.a.free(self.decode_rows);
        self.a.destroy(self);
        if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
    }

    fn bankLayer(self: *const Source, layer: u32) !u32 {
        if (layer < self.first_layer or layer - self.first_layer >= self.limits.len) return error.LingLayerNotRouted;
        return layer - self.first_layer;
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` f32 for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`.
    pub fn rows(self: *Source, layer: u32, x: T, idx: T, wts: T) !T {
        const bl = try self.bankLayer(layer);
        const g = &self.g;
        defer g.reset();
        self.state.layer = bl;
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        // COMBINE (src/ling_combine.zig): a prompt call's combine in one launch over its unjoined outputs.
        if (wideCall(g, x, ids)) return self.partsSum(g, bl, x, ids, wts);
        // DEVROUTE's run-path mark, at a decode call, which takes its route.
        if (self.phase == .decode and self.hook.devRoute()) ran.mark(.DEVROUTE);
        // ROUTED_JOIN (on the tail): a decode-lane call's outputs unjoined beside the join, which the layer's tail reads
        // in place (the join is then freed unevaluated with the sum below). A wider call (7 or 8 rows: more than the
        // decode lane's 48 ids), which `routedHoistParts` does not serve, took COMBINE's route above.
        const jp: ?Hook.JoinParts = if (ling_tail.armedFor(layer)) try self.hook.at(bl).routedHoistParts(g, x, ids, &.{}) else null;
        const ro = if (jp) |j| j.rows else try self.hook.at(bl).routed(g, x, ids);
        // MOE_TAIL: the joined rows and the weights for the layer's tail, which makes the sum below itself (this one is
        // freed unevaluated)
        if (ling_tail.armedFor(layer)) ling_tail.rows(ro, wts);
        if (jp) |j| ling_tail.parts(j.outs[0..j.n_outs], j.loc, j.identity);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        const out = try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false);
        return g.keep(out);
    }

    /// A call past one route's ids (`decode_route_ids`): the streamer's wide lane (`runEnd`'s test for a math with no
    /// prefill threshold).
    fn wideCall(g: *G, x: T, ids: T) bool {
        const n: usize = @intCast(g.shapeOf(x).dim(0));
        const k: usize = @intCast(g.shapeOf(ids).dim(1));
        return n * k > decode_route_ids;
    }

    /// COMBINE: a wide call through the streamer's JOINLESS route (`Hook.routedParts`: the waves' outputs unjoined, above
    /// 24 the smallest merged into one source, each pair's (source, row)), then turboexl3's Ling combine over them in one
    /// launch, the words of the join's take and `rows`' three ops. The sources' handles go once the combine is built (its
    /// graph holds them, as the join's concatenate holds the waves' outputs).
    fn partsSum(self: *Source, g: *G, bl: u32, x: T, ids: T, wts: T) !T {
        const h = self.hook.at(bl);
        const parts = try h.routedParts(g, x, ids);
        defer h.releaseParts(g);
        ran.mark(.COMBINE);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        // PF_MOE_TAIL: an armed layer's call keeps its outputs, loc and weights for its spans' tails (references of
        // src/ling_pf_tail.zig's own; nothing while no call is current)
        ling_pf_tail.parts(parts.outs, parts.loc, w);
        return g.keep(try g.adopt(try ling_combine.combineParts(g.s, parts.outs, parts.loc, w)));
    }

    /// A prompt pass's predicted routes of model layer `layer`: its experts, hottest first, read into its rows while its
    /// attention runs.
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.phase != .prompt) return;
        const bl = try self.bankLayer(layer);
        const n = self.bank.n_experts;
        const counts = try self.a.alloc(u32, n);
        defer self.a.free(counts);
        @memset(counts, 0);
        for (ids) |e| counts[e] += 1;
        const ranked = try self.a.alloc(u16, n);
        defer self.a.free(ranked);
        self.state.layer = bl;
        const before = self.source.stats().ahead_posted;
        try self.hook.at(bl).readAheadSeed(sdk_ext.expert.policy.rankHottest(counts, ranked));
        // D2's run-path marks, where a read-ahead posts records (D2, PROMPT_AHEAD, is this entry's one caller). D2a's
        // PROMPT_AHEAD_FILL marks only a fill read-ahead (`ahead_fill`) in a pass that `promptStart` began.
        const issued = self.source.stats().ahead_posted -| before;
        if (issued > 0) {
            ran.mark(.PROMPT_AHEAD);
            if (self.stream.ahead_fill and self.fill_pass) ran.mark(.PROMPT_AHEAD_FILL);
        }
    }

    /// D2a (PROMPT_AHEAD_FILL): a layer-major prompt pass starts with no protection left by earlier calls of the prompt
    /// phase (the engine's warm-up prompt runs in the same phase as the first request's, so its seed's protection would
    /// otherwise cut that request's seeds and the read-ahead's rows), as a later request's start after decode leaves it
    /// (`Stream.forgetPrompt`, `shrink(.keep)`'s): the residents stay. Residency only.
    pub fn promptStart(self: *Source) !void {
        try self.stream.forgetPrompt();
        self.fill_pass = true;
    }

    /// The rows of `p`: decode grows the slot banks once the prompt's windows are freed, to the rows the request's plan
    /// gives (RQ's `planner`, plugin.zig), read once the frees have landed; a prompt shrinks them back, keeping the
    /// residents decode left.
    pub fn setPhase(self: *Source, p: Phase, planner: Planner) !Change {
        if (p == self.phase) return .{};
        try self.hook.flush();
        // D2: the last read-ahead's records no call claimed are counted before the phase's line.
        self.stream.settleAheadClaims();
        self.logStats();
        self.mark = self.source.stats();
        self.steps = 0;
        self.fill_pass = false;
        var ch: Change = .{ .rows_before = if (self.phase == .decode) self.decode_rows[0] else self.prompt_rows[0] };
        const t0 = nowNs();
        switch (p) {
            .decode => {
                self.released = (try self.hook.releaseTransient()) > 0;
                _ = mlx.mlx_synchronize(self.g.s);
                self.g.clearCache();
                ch.free_ns = nowNs() - t0;
                const planned = planner.plan(planner.ctx);
                ch.planned = planned != null;
                @memset(self.decode_rows, self.decodeRows(planned));
                const t1 = nowNs();
                try self.hook.grow(&self.g, self.decode_rows);
                ch.alloc_ns = nowNs() - t1;
                ch.rows_after = self.decode_rows[0];
            },
            .prompt => {
                _ = try self.hook.shrink(self.prompt_rows, .keep);
                _ = mlx.mlx_synchronize(self.g.s);
                self.g.clearCache();
                ch.free_ns = nowNs() - t0;
                ch.planned = planner.plan(planner.ctx) != null;
                const t1 = nowNs();
                _ = try self.hook.regrowTransient(&self.g, self.released);
                ch.alloc_ns = nowNs() - t1;
                ch.rows_after = self.prompt_rows[0];
            },
        }
        self.phase = p;
        return ch;
    }

    /// RQ (plugin.zig): decode's rows changed in place, before an append past the request's plan (src/ling_rq.zig
    /// `crosses`), or for a request whose prompt runs in the decode phase: the grown rows and decode's window freed with
    /// the residents in the prompt rows kept (the stream's `shrink(.keep)`, as a prompt start frees them), MLX
    /// synchronized and its cache cleared, then the rows `planner` gives grown, with decode's window (the stream's
    /// `grow`). Only which experts are resident changes: the grown rows' residents are read again on demand.
    pub fn trimDecode(self: *Source, planner: Planner) !Change {
        if (self.phase != .decode) return error.LingTrimOutsideDecode;
        try self.hook.flush();
        var ch: Change = .{ .rows_before = self.decode_rows[0] };
        const t0 = nowNs();
        _ = try self.hook.shrink(self.prompt_rows, .keep);
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        ch.free_ns = nowNs() - t0;
        const planned = planner.plan(planner.ctx);
        ch.planned = planned != null;
        @memset(self.decode_rows, self.decodeRows(planned));
        const t1 = nowNs();
        try self.hook.grow(&self.g, self.decode_rows);
        ch.alloc_ns = nowNs() - t1;
        ch.rows_after = self.decode_rows[0];
        return ch;
    }

    /// The decode rows RQ's plan gives (null: the rows the bank opened with), between the base and a layer's experts.
    fn decodeRows(self: *const Source, planned: ?u32) u32 {
        const n = planned orelse self.bill.rows.decode;
        return @max(self.bill.rows.prompt, @min(n, self.bank.n_experts));
    }

    /// After the eval that consumed the forward's calls.
    pub fn settle(self: *Source) !void {
        try self.hook.flush();
        if (self.phase == .decode) {
            self.steps += 1;
            if (self.steps % 64 == 0) self.logStats();
        }
    }

    /// `[stream-stats]`: the current phase's expert reads so far (since its start): the routed experts' cache hits and
    /// misses, the bytes read from the bank and the read rate over the wall time a read was in flight.
    fn logStats(self: *Source) void {
        const s = self.source.stats();
        const m = self.mark;
        const hits = s.expert_cache_hits -| m.expert_cache_hits;
        const misses = s.expert_cache_misses -| m.expert_cache_misses;
        const bytes = s.expert_bytes_read -| m.expert_bytes_read;
        const wall_ns = s.read_wall_ns -| m.read_wall_ns;
        const total = hits + misses;
        const hit_pct: f64 = if (total > 0) 100.0 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total)) else 0;
        const gbs: f64 = if (wall_ns > 0) @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(wall_ns)) else 0;
        // D2d (PROMPT_KEEP): the prefill misses sent to a transient row to keep a resident their call routes; its
        // run-path mark at a prompt phase that kept one.
        const kept = s.kept_routed -| m.kept_routed;
        if (kept > 0 and self.phase == .prompt) ran.mark(.PROMPT_KEEP);
        // D2c (PROMPT_PARK): the records the read-ahead parked past the layers' rows that their call's plans claimed; its
        // run-path mark at a prompt phase that claimed one.
        const claimed = s.park_claimed -| m.park_claimed;
        if (claimed > 0 and self.phase == .prompt) ran.mark(.PROMPT_PARK);
        // The D2 family's counters, after the stock fields (zero where no read-ahead ran: decode, and a prompt outside the
        // layer-major pass). D2 (PROMPT_AHEAD): the prompt read-ahead's records read, claimed by their layer's call and
        // wasted. Then the D2 counters: the host's read waits (every one, and the read-ahead's landing apart); at each
        // read-ahead layer's barrier, the records read ahead that its call routes and the call's seed records not read
        // ahead; the prefill evictions of a resident its call routes (a later group reads it again); and the read-ahead
        // layers' misses by predicted rank (inside the rows, in the transient rows' worth after them, past those). D2d's
        // `kept` above. D2c: the records parked past the layers' rows, claimed by their call's plans (`claimed` above),
        // given up for a plan's own rows, and never claimed.
        var d2_buf: [720]u8 = undefined;
        const d2: []const u8 = std.fmt.bufPrint(&d2_buf, " ahead_issued={d} ahead_claimed={d} ahead_wasted={d} read_wait_s={d:.3} ahead_wait_s={d:.3} ahead_routed={d} seed_misses={d} routed_evicted={d} miss_rows={d} miss_next={d} miss_rest={d} kept_routed={d} park_issued={d} park_claimed={d} park_abandoned={d} park_wasted={d}", .{
            s.ahead_posted -| m.ahead_posted,
            s.ahead_claimed -| m.ahead_claimed,
            s.ahead_wasted -| m.ahead_wasted,
            @as(f64, @floatFromInt(s.read_wait_ns -| m.read_wait_ns)) / 1e9,
            @as(f64, @floatFromInt(s.ahead_wait_ns -| m.ahead_wait_ns)) / 1e9,
            s.ahead_hits -| m.ahead_hits,
            s.ahead_demand -| m.ahead_demand,
            s.routed_evicted -| m.routed_evicted,
            s.ahead_miss_rows -| m.ahead_miss_rows,
            s.ahead_miss_next -| m.ahead_miss_next,
            s.ahead_miss_rest -| m.ahead_miss_rest,
            kept,
            s.park_issued -| m.park_issued,
            claimed,
            s.park_abandoned -| m.park_abandoned,
            s.park_wasted -| m.park_wasted,
        }) catch "";
        // The read-ahead's records over the phase (every horizon): issued, claimed by a demand read, of those still in
        // flight at the claim (the prediction right, the read late), never claimed (abandoned), and refused a slot. With
        // the misses above: recall claimed / misses, precision claimed / issued, late claimed_inflight / claimed, waste
        // abandoned / issued. Plain counter reads at the line's own cadence. `spec_bytes`: what the speculative reads
        // read; `adopt_bytes`: the demand ranges copied out of them (in `read_bytes`, never read on demand), so the drive
        // read read_bytes - adopt_bytes + spec_bytes.
        const sp_issued = s.spec_issued -| m.spec_issued;
        const sp_claimed = s.claimed -| m.claimed;
        const sp_inflight = s.spec_claimed_inflight -| m.spec_claimed_inflight;
        const sp_abandoned = s.spec_abandoned -| m.spec_abandoned;
        const sp_noslot = s.spec_noslot -| m.spec_noslot;
        const sp_bytes = s.spec_bytes -| m.spec_bytes;
        const ad_bytes = s.adopt_bytes -| m.adopt_bytes;
        std.log.info("[stream-stats] {t} steps={d} hits={d} misses={d} hit_rate={d:.2}% read_bytes={d} read_wall_s={d:.3} read_GBps={d:.2} spec_issued={d} spec_claimed={d} spec_claimed_inflight={d} spec_abandoned={d} spec_noslot={d} spec_bytes={d} adopt_bytes={d}{s}", .{
            self.phase,                             self.steps, hits,      misses,     hit_pct,     bytes,
            @as(f64, @floatFromInt(wall_ns)) / 1e9, gbs,        sp_issued, sp_claimed, sp_inflight, sp_abandoned,
            sp_noslot,                              sp_bytes,   ad_bytes,  d2,
        });
    }
};
