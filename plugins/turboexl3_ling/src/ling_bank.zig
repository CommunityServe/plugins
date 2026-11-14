//! Ling-3.1-flash's routed experts streamed from the pack's `experts.bin` over the kit's expert stream
//! (`kit.streamed.expert.stream.StreamOf` over this bank module: slot rows, the residency policy, the read pool and the
//! prompt read-ahead are the kit's), with the routed math on turboexl3:
//! - a decode-width call (at most `decode_route_ids` routed pairs): one `moeLingBanked` launch over the layer's three
//!   slot banks (each pair at weight 1, `bank << 24 | row`), then turboexl3's Ling combine over the routes' weights;
//! - a wider (prompt) call: its distinct experts in groups of at most `S.max_route_ids`, each group fetched, then per
//!   slot bank one `moeLingRows` call over its pairs (the call's x read through the row map in turboexl3's prepare; the
//!   pairs stored as f16 where exact, PF_F16_PAIRS; a call of 16 pairs or fewer padded to 17 with zero-weight copies, so
//!   every prompt pair takes the prefill GEMM), each group's outputs evaluated before the next group reuses its slot
//!   rows; then turboexl3's Ling combine over the unjoined outputs (COMBINE: no join of the pairs' rows).
//! Each layer's SwiGLU is its own: Ling-3.1 clamps after the SiLU on its `expert_swiglu_limit_list` layers. Slot rows by
//! phase: the prompt rows through a prompt pass, the decode rows after `setPhase(.decode)` (the kit's transient release
//! and grow), the prompt rows again at the next prompt (its shrink, the residents kept).

const std = @import("std");
const stream_stats = @import("stream_stats.zig");
const prompt_plan = @import("prompt_plan.zig");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const expert = kit.streamed.expert;
const t3 = @import("turboexl3");
const manifest = @import("ling_manifest.zig");
const bank_rows = @import("bank_rows.zig");
const ling_math = @import("ling_math.zig");
const f16_pairs = @import("ling_f16_pairs.zig");
const rows_path = @import("ling_rows_path.zig");
const combine = @import("ling_combine.zig");
const log = @import("host/log.zig");

const Module = @This();
const A = mlx.mlx_array;

// ── the bank module the kit stream reads (`expert.assertBank`) ──

pub const n_components = manifest.n_components;
pub const gu_components = manifest.gu_components;
pub const Records = expert.io.Records(n_components, gu_components);
/// Ling-3.1-flash routes 8 experts a token (the lookahead selector's rank; this bank installs no lookahead).
pub const routed_top_k = 8;
pub const Component = enum(u8) { gate_code, gate_rout, gate_rin, up_code, up_rout, up_rin, down_code, down_rout, down_rin };
pub const Dtype = manifest.Dtype;
pub const Layer = manifest.Layer;

/// A slot array's dtype: the trellis is read as u16 (turboexl3's, the same bits as the pack's I16), the scales f16.
pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .I16 => .uint16,
        .F16 => .float16,
    };
}

pub const BankArrays = t3.Bank;

/// The slot arrays in record order (code, rout = svh, rin = suh per projection) as turboexl3's bank.
pub fn bankArraysOf(a: [n_components]A) BankArrays {
    return .{
        .gate = .{ .trellis = a[0], .svh = a[1], .suh = a[2] },
        .up = .{ .trellis = a[3], .svh = a[4], .suh = a[5] },
        .down = .{ .trellis = a[6], .svh = a[7], .suh = a[8] },
    };
}

pub const S = expert.stream.StreamOf(Module, false);

/// The pack's routed layers (bank layer b = model layer b + the first routed layer), `experts.bin` opened past the page
/// cache.
pub const Bank = struct {
    arena: std.heap.ArenaAllocator,
    layers: []Layer,
    n_experts: u32,
    sidecar: expert.io.UncachedFd,
    hidden: u32,
    inter: u32,

    pub fn open(a: std.mem.Allocator, dir: []const u8) !Bank {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const ar = arena.allocator();
        const sio = std.Io.Threaded.global_single_threaded.io();
        var d = std.Io.Dir.openDirAbsolute(sio, dir, .{}) catch return error.LingBankDir;
        defer d.close(sio);
        const raw = d.readFileAlloc(sio, "expert-manifest-v2.json", ar, .limited(64 << 20)) catch return error.LingBankManifest;
        const m = try manifest.parse(ar, raw);
        const path = try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, m.sidecar }, 0);
        const fd = expert.io.openUncachedFollowing(path, null) catch return error.LingBankSidecar;
        errdefer fd.close();
        if (fd.size < m.sidecar_size) return error.LingBankSidecarShort;
        return .{ .arena = arena, .layers = m.layers, .n_experts = m.n_experts, .sidecar = fd, .hidden = m.hidden, .inter = m.inter };
    }

    /// After the stream over the bank is gone.
    pub fn deinit(b: *Bank) void {
        b.sidecar.close();
        b.arena.deinit();
    }

    pub fn recordOffset(b: *const Bank, layer: u32, e: u32) u64 {
        const l = &b.layers[layer];
        return l.base_offset + @as(u64, e) * l.record_bytes;
    }

    pub fn spans(b: *const Bank, layer: u32, e: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, e);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }

    fn rowBytes(b: *const Bank) u64 {
        return manifest.rowBytes(.{ .layers = b.layers, .n_experts = b.n_experts, .hidden = b.hidden, .inter = b.inter, .sidecar = "", .sidecar_size = 0 });
    }

    fn widest(b: *const Bank) u64 {
        var n: u64 = 0;
        for (b.layers) |l| n = @max(n, l.logical_bytes);
        return n;
    }
};

// ── the source plugin.zig opens ──

/// The trunk's MoE geometry the bank must match; `limits`: each routed layer's SwiGLU clamp after the SiLU (0 none), in
/// bank order.
pub const Geometry = struct { hidden_size: u32, moe_intermediate_size: u32, n_experts_per_tok: u32, n_routed_experts: u32, n_layers: u32, limits: []const c_int };
pub const Budget = bank_rows.Budget;
pub const Phase = enum { prompt, decode };

/// Routed pairs a decode-phase call takes (`abi.Bank.decode_ids`): one route of the kit stream.
pub const decode_route_ids: usize = S.max_route_ids;
/// The read pool's workers (the served bank's).
const read_workers: u32 = 8;

/// What the source holds, by term, at its rows: a row of every routed layer, each phase's windows (the prompt's
/// transient scratch; decode's window 0 and the scratch, billed as the served bank billed them) and the pool's staging.
pub const Bill = struct {
    row_bytes: u64,
    prompt_windows: u64,
    decode_windows: u64,
    staging: u64,
    rows: bank_rows.Rows,

    fn of(b: *const Bank, budget: Budget) !Bill {
        const rec = b.widest();
        const terms: bank_rows.Bank = .{
            .row_bytes = b.rowBytes(),
            .prompt_windows = @as(u64, S.max_route_ids) * rec,
            .decode_windows = 2 * @as(u64, S.max_route_ids) * rec,
            .staging = read_workers * S.stagingBytes(b),
            .n_experts = b.n_experts,
        };
        return .{ .row_bytes = terms.row_bytes, .prompt_windows = terms.prompt_windows, .decode_windows = terms.decode_windows, .staging = terms.staging, .rows = try bank_rows.rowsOf(terms, budget) };
    }
};

fn checkGeometry(b: *const Bank, geo: Geometry) !void {
    if (geo.n_layers != b.layers.len or geo.limits.len != b.layers.len) return error.LingBankLayers;
    if (geo.n_routed_experts != b.n_experts or geo.hidden_size != b.hidden or geo.moe_intermediate_size != b.inter) return error.LingBankGeometry;
    if (geo.n_experts_per_tok == 0 or geo.n_experts_per_tok > combine_max_topk) return error.LingBankGeometry;
}

/// turboexl3's combine folds at most 8 routes a row.
const combine_max_topk = 8;

/// The bank's rows at `budget` (the host's `bank_rows`), from the pack in `dir`.
pub fn billOf(a: std.mem.Allocator, dir: []const u8, geo: Geometry, budget: Budget) !Bill {
    var b = try Bank.open(a, dir);
    defer b.deinit();
    try checkGeometry(&b, geo);
    return Bill.of(&b, budget);
}

pub const Source = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    bank: Bank,
    stream: *S.Stream,
    limits: []c_int,
    top_k: u32,
    first_layer: u32,
    prompt_rows: []u32,
    decode_rows: []u32,
    bill: Bill,
    phase: Phase = .prompt,
    /// The decode rows grown this request (C5's plan: at most the load's).
    grown: u32 = 0,
    /// This request's under-the-base warning was logged.
    warned: bool = false,
    /// The stream's counters where this request's prompt and decode phases began (`meters`), and its lines.
    at_prompt: expert.Stats = .{},
    at_decode: expert.Stats = .{},
    meter_buf: [1024]u8 = undefined,

    /// The bank in `dir` at the rows `budget` holds, slot arrays on `s` (the trunk's stream). `first_layer`: the model
    /// layer of bank layer 0. Takes the process's one expert reader.
    pub fn open(a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream, geo: Geometry, first_layer: u32, budget: Budget) !*Source {
        try expert.takeReader();
        errdefer expert.giveReader();
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .s = s, .bank = undefined, .stream = undefined, .limits = undefined, .top_k = geo.n_experts_per_tok, .first_layer = first_layer, .prompt_rows = undefined, .decode_rows = undefined, .bill = undefined };
        self.bank = try Bank.open(a, dir);
        errdefer self.bank.deinit();
        try checkGeometry(&self.bank, geo);
        self.bill = try Bill.of(&self.bank, budget);
        self.limits = try a.dupe(c_int, geo.limits);
        errdefer a.free(self.limits);
        self.prompt_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.prompt_rows);
        @memset(self.prompt_rows, self.bill.rows.prompt);
        self.decode_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.decode_rows);
        @memset(self.decode_rows, self.bill.rows.decode);
        try ling_math.configure();
        self.stream = try S.Stream.init(a, &self.bank, .{
            .rows = self.prompt_rows,
            .transient_rows = S.max_route_ids,
            .pool = .{ .tickets = 1024, .workers = read_workers },
            .staging_from_bank = true,
            .slot_memory = .{ .mlx = s },
            .transient_release = true,
        });
        return self;
    }

    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.s);
        self.stream.deinit();
        self.bank.deinit();
        self.a.free(self.limits);
        self.a.free(self.prompt_rows);
        self.a.free(self.decode_rows);
        self.a.destroy(self);
        expert.giveReader();
    }

    fn bankLayer(self: *const Source, layer: u32) !u32 {
        if (layer < self.first_layer or layer - self.first_layer >= self.limits.len) return error.LingLayerNotRouted;
        return layer - self.first_layer;
    }

    /// Model layer `layer`'s routed sum f32 `[n, hidden]` for the MoE input `x` `[n, hidden]`, the routes `idx` `[n, k]`
    /// and their weights `wts` `[n, k]`.
    pub fn rows(self: *Source, layer: u32, x: A, idx: A, wts: A) !A {
        const bl = try self.bankLayer(layer);
        const sh = mlx.getShape(idx);
        if (sh.len != 2 or sh[1] != @as(c_int, @intCast(self.top_k)) or mlx.getShape(x)[0] != sh[0]) return error.LingRoutesShape;
        const n: usize = @intCast(sh[0]);
        const k: usize = @intCast(sh[1]);
        const ids = try self.a.alloc(u16, n * k);
        defer self.a.free(ids);
        try hostIds(self.s, idx, ids, self.bank.n_experts);
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_astype(&w, wts, .float32, self.s));
        if (n * k <= decode_route_ids) return self.decodeCall(bl, x, ids, n, k, w);
        return self.promptCall(bl, x, ids, n, k, w);
    }

    /// One route over the call's experts, its reads waited on the host, then one banked launch over its pairs.
    fn decodeCall(self: *Source, bl: u32, x: A, ids: []const u16, n: usize, k: usize, w: A) !A {
        const s = self.s;
        try self.stream.flush();
        var distinct: [S.max_route_ids]u16 = undefined;
        const nd = distinctOf(ids, &distinct);
        const r = try self.stream.route(bl, distinct[0..nd], &.{});
        var refs: [S.max_route_ids]expert.SlotRef = undefined;
        const rf = self.stream.refsOf(r, &refs);
        for (0..r.n_parts) |p| {
            try self.stream.waitGu(r, @intCast(p));
            try self.stream.waitDown(r, @intCast(p));
        }
        var enc: [S.max_route_ids]u32 = undefined;
        var rowmap: [S.max_route_ids]u32 = undefined;
        for (ids, 0..) |e, i| {
            const at = std.mem.indexOfScalar(u16, distinct[0..nd], e).?;
            enc[i] = @as(u32, @intFromEnum(rf[at].bank)) << 24 | rf[at].row;
            rowmap[i] = @intCast(i / k);
        }
        const banks = self.banksOf(bl);
        const ids_a = mlx.mlx_array_new_data(&enc, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ids_a);
        const map_a = mlx.mlx_array_new_data(&rowmap, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(map_a);
        var xp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xp);
        try mlx.check(mlx.mlx_take_axis(&xp, x, map_a, 0, s));
        const y = try t3.ling.moeLingBanked(s, xp, &banks, ids_a, .mul1, self.limits[bl]);
        defer _ = mlx.mlx_array_free(y);
        self.stream.release(r);
        var y3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y3);
        try mlx.check(mlx.mlx_reshape(&y3, y, &[_]c_int{ @intCast(n), @intCast(k), mlx.getShape(y)[1] }, 3, s));
        return t3.ling.combineRouted(s, y3, w);
    }

    /// The layer's three slot banks as turboexl3 binds them; a bank the stream has no rows in yet takes the base's
    /// arrays (no pair names it).
    fn banksOf(self: *const Source, bl: u32) [3]t3.Bank {
        const base = self.stream.bankArrays(bl, .base).?;
        return .{ base, self.stream.bankArrays(bl, .ext) orelse base, self.stream.bankArrays(bl, .transient) orelse base };
    }

    /// A prompt call: groups of at most `S.max_route_ids` experts, each fetched and run per slot bank on the rows path,
    /// its outputs evaluated before the next group's route; then the combine over the unjoined outputs.
    fn promptCall(self: *Source, bl: u32, x: A, ids: []const u16, n: usize, k: usize, w: A) !A {
        try self.checkPrompt();
        const a = self.a;
        const s = self.s;
        var xb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb);
        try mlx.check(mlx.mlx_astype(&xb, x, .bfloat16, s));
        // PF_DFIN's whole-call rule, from the call's rows.
        const dfin = ling_math.dfinTakes(n);
        // Each pair's (source, row) in the call's outputs, the combine's `loc`.
        const loc = try a.alloc(i32, 2 * ids.len);
        defer a.free(loc);
        var sources: std.ArrayList(A) = .empty;
        defer {
            for (sources.items) |o| _ = mlx.mlx_array_free(o);
            sources.deinit(a);
        }
        // The call's pairs grouped by expert (src/prompt_plan.zig): expert e's pairs are `order[first[e]..first[e + 1]]`.
        const ne = self.bank.n_experts;
        const first = try a.alloc(u32, ne + 1);
        defer a.free(first);
        const order = try a.alloc(u32, ids.len);
        defer a.free(order);
        const fill = try a.alloc(u32, ne);
        defer a.free(fill);
        var experts: std.ArrayList(u16) = .empty;
        defer experts.deinit(a);
        try prompt_plan.byExpert(ids, ne, first, order, fill, &experts, a);
        // Each expert's slot row in the group's route (`slot_row[e]`) and bank (`bank_of[j]`, the group's j-th expert).
        const slot_row = try a.alloc(u32, ne);
        defer a.free(slot_row);
        var bank_of: [S.max_route_ids]u8 = undefined;
        const pair_rows = try a.alloc(u32, ids.len + ling_math.gemm_min_rows);
        defer a.free(pair_rows);
        const pair_inds = try a.alloc(u32, ids.len + ling_math.gemm_min_rows);
        defer a.free(pair_inds);
        const pair_at = try a.alloc(u32, ids.len);
        defer a.free(pair_at);
        const weights = try a.alloc(f32, ids.len + ling_math.gemm_min_rows);
        defer a.free(weights);
        var at: usize = 0;
        while (at < experts.items.len) {
            const group = experts.items[at..@min(at + S.max_route_ids, experts.items.len)];
            at += group.len;
            try self.stream.flush();
            const r = try self.stream.route(bl, group, &.{});
            var refs: [S.max_route_ids]expert.SlotRef = undefined;
            const rf = self.stream.refsOf(r, &refs);
            for (0..r.n_parts) |p| {
                try self.stream.waitGu(r, @intCast(p));
                try self.stream.waitDown(r, @intCast(p));
            }
            const banks = self.banksOf(bl);
            for (group, rf[0..group.len], bank_of[0..group.len]) |e, ref, *b| {
                slot_row[e] = ref.row;
                b.* = @intFromEnum(ref.bank);
            }
            // The banks in `banks`' order: base, ext, transient (`expert.BankKind`'s values).
            for (0..3) |bk| {
                // Every pair's call is its expert's bank's, the expert's pairs together in route order: a pair's math
                // does not depend on which bank its expert sits in (src/prompt_plan.zig's residency test).
                const m = prompt_plan.bankPairs(group, bank_of[0..group.len], @intCast(bk), first, order, pair_at);
                for (pair_at[0..m], 0..) |i, j| {
                    pair_rows[j] = @intCast(i / k);
                    pair_inds[j] = slot_row[ids[i]];
                }
                if (m == 0) continue;
                const src = sources.items.len;
                const y = try runRows(s, xb, banks[bk], pair_rows, pair_inds, weights, m, self.limits[bl], dfin);
                try mlx.check(mlx.mlx_array_eval(y));
                try sources.append(a, y);
                for (pair_at[0..m], 0..) |p, row| {
                    loc[2 * p] = @intCast(src);
                    loc[2 * p + 1] = @intCast(row);
                }
            }
            self.stream.release(r);
        }
        if (sources.items.len > t3.ling.combine_max_sources) try joinSources(s, a, &sources, loc);
        const loc_a = mlx.mlx_array_new_data(loc.ptr, &[_]c_int{ @intCast(n), @intCast(k), 2 }, 3, .int32);
        defer _ = mlx.mlx_array_free(loc_a);
        return combine.combineParts(s, sources.items, loc_a, w);
    }

    /// A prompt pass's predicted routes of model layer `layer` (every row's ids): its experts, hottest first, read into
    /// the layer's prompt rows while its attention runs (the kit's read-ahead seed; the layer's first route lands it).
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.phase != .prompt) return;
        const bl = try self.bankLayer(layer);
        const n = self.bank.n_experts;
        const counts = try self.a.alloc(u32, n);
        defer self.a.free(counts);
        @memset(counts, 0);
        for (ids) |e| if (e < n) {
            counts[e] += 1;
        };
        const ranked = try self.a.alloc(u16, n);
        defer self.a.free(ranked);
        try self.stream.readAheadSeed(bl, expert.policy.rankHottest(counts, ranked));
    }

    /// The rows of `p`: decode grows the slot banks once the prompt's scratch is released, to the rows C5's plan gives
    /// at the host's headroom (src/bank_rows.zig `decodePlan`); a prompt shrinks them back, keeping the residents decode
    /// left, then makes its scratch again. A decode plan past free RAM's cover still grows (at the base, so the next
    /// prompt's shrink holds) and is then refused by name.
    pub fn setPhase(self: *Source, p: Phase) !void {
        if (p == self.phase) return;
        try self.stream.flush();
        if (p == .decode) self.at_decode = self.stream.stats() else self.at_prompt = self.stream.stats();
        switch (p) {
            .decode => {
                _ = try self.stream.releaseTransient();
                _ = mlx.mlx_synchronize(self.s);
                const plan = self.planDecode(self.bill.rows.prompt, .decode);
                self.grown = plan.rows;
                @memset(self.decode_rows, plan.rows);
                try self.stream.grow(self.decode_rows);
                self.phase = .decode;
                if (plan.verdict == .refuse) return error.LingDecodeUnderBase;
                return;
            },
            .prompt => {
                _ = try self.stream.shrink(self.prompt_rows, .keep);
                _ = mlx.mlx_synchronize(self.s);
                _ = try self.stream.regrowTransient();
                self.warned = false;
            },
        }
        self.phase = p;
    }

    /// The request's `[stream-stats]` lines by phase (the host's `Bank.meters`): the stream's counters since the prompt
    /// began, and since decode began.
    pub fn meters(self: *Source) []const u8 {
        const now = self.stream.stats();
        const prompt = (if (self.phase == .decode) self.at_decode else now).since(self.at_prompt);
        const decode: expert.Stats = if (self.phase == .decode) now.since(self.at_decode) else .{};
        const a = stream_stats.line(&self.meter_buf, "prompt", prompt) catch return "";
        const b = stream_stats.line(self.meter_buf[a.len + 1 ..], "decode", decode) catch return a;
        self.meter_buf[a.len] = '\n';
        return self.meter_buf[0 .. a.len + 1 + b.len];
    }

    const PlanKind = enum { decode, trim };

    /// C5's decode or trim plan at the host's headroom with `now` rows held, its warning or refusal logged by name;
    /// without a host reading (a test), the load's decode rows.
    fn planDecode(self: *Source, now: u32, kind: PlanKind) bank_rows.Plan {
        const r = self.bill.rows;
        const h = log.headroom() orelse return .{ .rows = r.decode, .held = r.decode, .verdict = .fits };
        const plan = bank_rows.decodePlan(r, self.bill.row_bytes, now, h);
        const under = @as(i64, r.prompt) - plan.held;
        switch (plan.verdict) {
            .fits => if (kind == .decode and plan.rows < r.decode)
                log.info("[turboexl3_ling] C5 decode plan: {d} MiB of headroom holds {d} rows, under the load's {d}: the bank grows to {d}\n", .{ h >> 20, plan.held, r.decode, plan.rows }),
            .warn => if (!self.warned) {
                self.warned = true;
                log.warn("[turboexl3_ling] LingDecodeUnderBase: the {t} plan holds {d} rows at {d} MiB of headroom, {d} under the base {d}, within the {d} rows free RAM covers: the bank stays at its base\n", .{ kind, plan.held, h >> 20, under, r.prompt, bank_rows.coverRows(self.bill.row_bytes) });
            },
            .refuse => log.err("[turboexl3_ling] LingDecodeUnderBase: the {t} plan holds {d} rows at {d} MiB of headroom, {d} under the base {d}, past the {d} rows free RAM covers: refused\n", .{ kind, plan.held, h >> 20, under, r.prompt, bank_rows.coverRows(self.bill.row_bytes) }),
        }
        return plan;
    }

    /// C5's prompt check at a prompt forward's call: under the base (less than one slot row of the host's headroom, its
    /// KV and the pass as allocated so far) refused by name.
    fn checkPrompt(self: *Source) !void {
        const h = log.headroom() orelse return;
        const plan = bank_rows.promptPlan(self.bill.rows, self.bill.row_bytes, h);
        if (plan.verdict != .refuse) return;
        log.err("[turboexl3_ling] LingPromptUnderBase: the prompt pass has {d} MiB of headroom, under one slot row ({d} MiB) past the base {d}: refused\n", .{ h >> 20, self.bill.row_bytes >> 20, self.bill.rows.prompt });
        return error.LingPromptUnderBase;
    }

    /// After the eval that consumed a forward's calls: their slot rows unpinned; in decode, C5's trim check at the
    /// headroom the forward left.
    pub fn settle(self: *Source) !void {
        try self.stream.flush();
        if (self.phase == .decode and self.planDecode(self.grown, .trim).verdict == .refuse) return error.LingDecodeUnderBase;
    }
};

/// `idx` (the routes, any integer dtype) on the host as u16 expert ids, each below `n_experts`.
fn hostIds(s: mlx.mlx_stream, idx: A, out: []u16, n_experts: u32) !void {
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_astype(&u, idx, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(u));
    const d = mlx.mlx_array_data_uint32(u) orelse return error.LingRoutesRead;
    for (out, d[0..out.len]) |*o, v| {
        if (v >= n_experts) return error.LingExpertId;
        o.* = @intCast(v);
    }
}

/// `ids`' distinct values into `out`, first seen first; returns their count.
fn distinctOf(ids: []const u16, out: *[S.max_route_ids]u16) usize {
    var n: usize = 0;
    for (ids) |e| {
        if (std.mem.indexOfScalar(u16, out[0..n], e) != null) continue;
        out[n] = e;
        n += 1;
    }
    return n;
}

/// One slot bank's `m` pairs (`rows` the call's x rows, `inds` the bank's slot rows) on turboexl3's rows path, each at
/// weight 1: f16 rows where exact (PF_F16_PAIRS), else f32. A call of 16 pairs or fewer is padded to 17 with
/// zero-weight copies of its first pair, whose rows are dropped (`[0, m)` is the call's).
fn runRows(s: mlx.mlx_stream, xb: A, bank: t3.Bank, rows: []u32, inds: []u32, weights: []f32, m: usize, limit: c_int, dfin: bool) !A {
    const small = m < ling_math.gemm_min_rows;
    const padded = if (small) ling_math.gemm_min_rows else m;
    for (m..padded) |i| {
        rows[i] = rows[0];
        inds[i] = inds[0];
    }
    const wv = f16_pairs.weightsInto(weights, m, small, padded);
    const rows_a = mlx.mlx_array_new_data(rows.ptr, &[_]c_int{@intCast(padded)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(rows_a);
    const inds_a = mlx.mlx_array_new_data(inds.ptr, &[_]c_int{@intCast(padded)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds_a);
    const y = (try f16_pairs.call(s, xb, rows_a, bank, inds_a, wv, limit, dfin)) orelse blk: {
        const sc = mlx.mlx_array_new_data(wv.ptr, &[_]c_int{@intCast(padded)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        break :blk try rows_path.call(s, xb, rows_a, bank, inds_a, sc, limit, dfin);
    };
    if (!small) return y;
    defer _ = mlx.mlx_array_free(y);
    var cut = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cut);
    const hid = mlx.getShape(y)[1];
    try mlx.check(mlx.mlx_slice(&cut, y, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(m), hid }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return cut;
}

/// More sources than one combine launch binds: every source joined into one (in f32), each pair's row moved to its place
/// in the join.
fn joinSources(s: mlx.mlx_stream, a: std.mem.Allocator, sources: *std.ArrayList(A), loc: []i32) !void {
    const n = sources.items.len;
    const base = try a.alloc(i32, n);
    defer a.free(base);
    var total: i32 = 0;
    var wide: std.ArrayList(A) = .empty;
    defer {
        for (wide.items) |o| _ = mlx.mlx_array_free(o);
        wide.deinit(a);
    }
    for (sources.items, base) |src, *b| {
        b.* = total;
        total += mlx.getShape(src)[0];
        var f = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f, src, .float32, s));
        try wide.append(a, f);
    }
    const vec = mlx.mlx_vector_array_new_data(wide.items.ptr, wide.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, s));
    try mlx.check(mlx.mlx_array_eval(joined));
    var i: usize = 0;
    while (i < loc.len) : (i += 2) {
        loc[i + 1] += base[@intCast(loc[i])];
        loc[i] = 0;
    }
    for (sources.items) |o| _ = mlx.mlx_array_free(o);
    sources.clearRetainingCapacity();
    try sources.append(a, joined);
}

