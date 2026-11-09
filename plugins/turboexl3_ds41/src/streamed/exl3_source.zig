//! A resident trunk's routed experts from an EXL3 expert bank (`experts.bin` + `expert-manifest-v2.json`): the EXL3
//! lane (`sdk/quant/exl3/stream/`) over the expert stream, the source returning each call's routed sum. The served
//! schedule: event-gated waves, the next layer's reads predicted from its router, the wide prompt calls' feed (seed,
//! hottest first, 5 groups ahead, the base call at the seed), and the decode forms, banked waves, hoist and device
//! route. Slot rows per phase: the prompt rows during a prompt pass, the decode rows after `phase(.decode)`.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const xp = @import("experts.zig");
const expert_bank = @import("expert/bank.zig");
const expert_stream = @import("expert/stream.zig");
const sdk_ext = @import("sdk_ext.zig");
/// mlx-serve's expert usage profile (`expert-usage-v1.json`): the resident seed at the load
/// (expert_usage_cfg.zig: `TURBOEXL3_DS41_RESIDENT_SEED`, `TURBOEXL3_DS41_EXPERT_USAGE_DIR`).
const expert_usage = sdk_ext.expert.usage;
const usage_mod = @import("../bank/expert_usage_cfg.zig");
/// The plugin's scratch_waves.zig: the prefill waves' per-prompt regions.
const scratch_waves = @import("../bank/scratch_waves.zig");
const usage_arch = "deepseek_v41";
const expert_event = sdk_ext.expert.event;
const expert_admission = @import("expert/admission.zig");
const xk = @import("../exl3/kernels.zig");
const xq = @import("../exl3/quant.zig");
const kernel_set = sdk_ext.kernels.KernelSet(xk);
const G = ops.MlxOps;
const T = mlx.mlx_array;

const Math = xp.QuantMath(G, xq.Accepted(G));
const Hook = xp.ExpertsWith(G, xp.StreamSource, Math, .{ .prefill = true, .lookahead = true, .gated = true });
/// The plugin's routed.zig: the same schedule, the decode waves banked and DEVROUTE's device wave on turboexl3's
/// banked decode, as the streamer's own kernels run them; the prompt calls on the
/// streamer's own math (`Math`, the bounded DIG-X lane the prompt bill covers).
const routed_t3 = @import("../bank/routed.zig");
/// `TURBOEXL3_DS41_FRAC` (frac.zig): the bank's half-integer-rate layers, on the t3 math's turboexl3 calls.
const frac = @import("../bank/frac.zig");
const MathT3 = routed_t3.Math(G, xp.ProjOf(T), xp.BankArraysOf(T), sdk_ext.quant.PrefillRows, Math);
const HookT3 = xp.ExpertsWith(G, xp.StreamSource, MathT3, .{ .prefill = true, .lookahead = true, .gated = true });
/// The plugin's tail_fuse.zig: the combine and the model's hc expand in one kernel.
const tail_fuse = @import("../bank/tail_fuse.zig");
/// MLX's cache capped at decode, the prompt's cap restored for the
/// next prompt.
const decode_stop = @import("../bank/decode_stop.zig");
/// MLX's command-buffer split raised for a prompt, put back at decode.
const buffer_mb = @import("../bank/buffer_mb.zig");
/// decode's window 0 rows, the bill's decode term and the stream's
/// grow.
const dwin = @import("../bank/decode_window.zig");
const dsv41_routes = @import("../exl3/kernel_routes.zig");
const Combine = dsv41_routes.JoinlessCombine(G);
pub const Gate = Hook.Gate;
pub const Phase = enum { prompt, decode };

/// The trunk's MoE geometry the bank must match.
pub const Geometry = struct { hidden_size: u32, moe_intermediate_size: u32, n_experts_per_tok: u32, n_routed_experts: u32, n_layers: u32, swiglu_limit: f64 };

/// Bytes the slot rows and the source's own buffers may hold in each phase.
pub const Budget = struct { prompt: u64, decode: u64 };

/// What the source holds, by term, at its rows.
pub const Bill = struct {
    /// One slot row in every layer (each layer's record).
    row_bytes: u64,
    /// The prompt's transient windows (`wide_depth` routes of `max_route_ids` rows, per record geometry).
    prompt_windows: u64,
    /// Decode's: window 0 (the phase change releases the rest).
    decode_windows: u64,
    /// The lookahead's staging rows.
    staging: u64,
    rows: struct { prompt: u32, decode: u32 },

    pub fn prompt(b: Bill) u64 {
        return b.row_bytes * b.rows.prompt + b.prompt_windows + b.staging;
    }

    pub fn decode(b: Bill) u64 {
        return b.row_bytes * b.rows.decode + b.decode_windows + b.staging;
    }
};

const wide_depth: u8 = 5;
/// The decode's lookahead (k 8, every row's plain top-K, idle rule 0): the stream sized for the selector's widest
/// budget, each layer call reading `la_wide.base_budget` records ahead and a credit of `la_wide.per_round` records a round
/// past it (la_wide.zig).
const la_wide = @import("mlx_serve").streamed.expert.la_wide;
fn lookahead() expert_stream.Lookahead {
    return .{ .k = 8, .tau = std.math.inf(f32), .budget = sdk_ext.expert.lookahead.max_budget, .base = la_wide.base_budget, .wide = la_wide.per_round, .idle_busy = 0 };
}
const event_watchdog_ms = 2000;
const wide: xp.Wide = .{ .seed = true, .hot_first = true, .depth = wide_depth, .defer_base = true, .read_ahead = true, .base_at_seed = true, .seed_aligned = true };
const forms: xq.Forms = .{ .down_pair = true, .gu_one = true };

/// The bank geometry the EXL3 lane decodes for `geo`: mul1, K = 3 or 2 per layer, and with TURBOEXL3_DS41_FRAC on
///  the half-integer rates its t3 math runs on turboexl3 (K 1.5 and 2.5: 24 and 40
/// halfwords a tile).
fn implementedOf(geo: Geometry) expert_bank.Implemented {
    return .{ .codebooks = &.{"mul1"}, .k = &.{ 2, 3 }, .frac_n = frac.admitted(), .hidden = geo.hidden_size, .inter = geo.moe_intermediate_size, .n_experts = geo.n_routed_experts, .n_layers = geo.n_layers };
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
        .staging = expert_admission.lookaheadCharge(record, 2 * lookahead().budget, std.heap.pageSize()),
        .rows = undefined,
    };
    const n: u64 = bank.n_experts;
    const p = (budget.prompt -| (b.prompt_windows + b.staging)) / row;
    var d = (budget.decode -| (b.decode_windows + b.staging)) / row;
    if (p == 0) return error.StreamedRowsDoNotFit;
    // decode's window 0 at its rows, where the decode rows it leaves are
    // 2 * max_route_ids or more (the stream's `decodeWindow` takes it on the same rule; below it, the widest route's).
    const w: u64 = dwin.window_rows;
    if (release and w > 0 and w < sdk_ext.expert.policy.max_route_ids) {
        const dw = w * (record + alt);
        const dn = (budget.decode -| (dw + b.staging)) / row;
        if (@min(@max(dn, p), n) >= 2 * sdk_ext.expert.policy.max_route_ids) {
            b.decode_windows = dw;
            d = dn;
        }
    }
    b.rows = .{ .prompt = @intCast(@min(p, n)), .decode = @intCast(@min(@max(d, p), n)) };
    return b;
}

/// Routed ids a decode-phase call takes (the stream's route width); a wider call is a prompt pass.
pub const decode_route_ids: usize = sdk_ext.expert.policy.max_route_ids;

pub const Source = struct {
    a: std.mem.Allocator,
    g: G,
    set: *kernel_set.Set,
    q: *xq.Accepted(G),
    bank: expert_bank.Bank,
    gates: []Gate,
    prompt_rows: []u32,
    decode_rows: []u32,
    stream: *expert_stream.Stream,
    source: xp.StreamSource,
    hook: HookT3,
    gates_t3: []HookT3.Gate = &.{},
    t3_state: MathT3.State = .{ .limit = 0 },
    bill: Bill,
    phase: Phase = .prompt,
    released: bool = false,

    /// JOINLESS's combine, built and compiled at the load.
    jl: ?Combine = null,
    /// The bank in `dir` at the rows `budget` holds, streamed on `s` (the trunk's stream). `gates`: every routed layer's
    /// router (its arrays borrowed for the source's life).
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, geo: Geometry, gates: []const Gate, budget: Budget) !*Source {
        var diag: xk.Diag = .{};
        errdefer if (diag.len > 0) std.log.err("streamed exl3: {s}", .{diag.message()});
        // One expert reader per process: a second streamed load is refused by name.
        if (comptime expert_stream.uses_reader) try sdk_ext.expert.takeReader();
        errdefer if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.a = a;
        self.phase = .prompt;
        self.released = false;
        self.jl = null;
        self.bank = try expert_bank.Bank.open(a, io, dir, implementedOf(geo), null);
        errdefer self.bank.deinit();
        self.bill = try billBank(&self.bank, budget);
        self.gates = try a.dupe(Gate, gates);
        errdefer a.free(self.gates);
        self.prompt_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.prompt_rows);
        @memset(self.prompt_rows, self.bill.rows.prompt);
        self.decode_rows = try a.alloc(u32, geo.n_layers);
        errdefer a.free(self.decode_rows);
        @memset(self.decode_rows, self.bill.rows.decode);
        self.splitDecodeRows();
        self.g = try G.init(a, s);
        errdefer self.g.deinit();
        self.set = try kernel_set.Set.init(a, .{ .device = .{ .stream = s } }, &diag);
        errdefer self.set.deinit();
        self.set.install(G, &self.g);
        errdefer kernel_set.Set.uninstall(G, &self.g);
        self.q = try xq.accept(G, a, &self.g, .{ .kernels = self.set.ref() }, .{
            .hidden = geo.hidden_size,
            .inter = geo.moe_intermediate_size,
            .top_k = geo.n_experts_per_tok,
            .n_layers = geo.n_layers,
            .act = .{ .swiglu_clamped = geo.swiglu_limit },
            .input = .bfloat16,
        }, &diag);
        errdefer self.q.deinit(&self.g);
        try self.q.routeExpertBf16();
        try self.q.compileTexts(self.set, &xq.form_texts, &diag);
        try self.q.routeForms(&self.g, forms);
        try self.q.compileTexts(self.set, &xq.banked_texts, &diag);
        try self.q.routeBanked(&self.g);
        // The scratch waves: the waves' write-into twins derived from the pinned stock texts and compiled now,
        // every layer's uncarried waves on them; a twin that does not derive or compile refuses the load by name.
        // M1: the waves' 128-row GEMMs double-buffered, self-checked now (a failed check refuses the load by name);
        // before the scratch twins, which derive from the GEMM texts it selects.
        try self.q.routeM1(self.set, &diag);
        try self.q.routeScratchWaves(&self.g, &diag);
        // JOINLESS: the combine on the set the quant accepted, its bf16 text compiled and launched once now.
        {
            var jgeo = dsv41_routes.PrefillGeometry.derived;
            jgeo.hidden = geo.hidden_size;
            jgeo.n_experts_per_tok = geo.n_experts_per_tok;
            // A geometry the combine was not derived for joins as before.
            self.jl = Combine.init(&self.set.reg, &jgeo, &diag) catch |e| if (e == error.RouteInput) null else return e;
            if (self.jl != null) try self.set.selfCheck(a, &.{.dsv41_jl_combine_bf16}, .compile, &self.q.report, &diag);
        }
        const event = try expert_event.createMetal();
        // The bank pool: the wave sets' regions in the pool's prompt layout.
        var wave_spares: [scratch_waves.pool_shapes.len]expert_stream.SpareArray = undefined;
        for (&wave_spares, scratch_waves.pool_shapes) |*o, sh| o.* = .{ .shape = sh.shape, .ndim = sh.ndim, .dtype = sh.dtype };
        self.stream = try expert_stream.Stream.init(a, &self.bank, .{
            .rows = self.prompt_rows,
            .slot_memory = .{ .mlx = s },
            .lookahead = lookahead(),
            .event = .{ .backend = .{ .metal = event.object }, .watchdog_ms = event_watchdog_ms },
            .wide_depth = wide_depth,
            .transient_rows = @as(u32, wide_depth) * sdk_ext.expert.policy.max_route_ids,
            .transient_release = expert_stream.phase_change_releases_wide_windows and expert_stream.transient_release_default,
            // Decode's window 0 rows.
            .decode_window_rows = dwin.window_rows,
            // Staging: the staged records' tickets stay unclaimed beside every call's own (`Stream.stageGroups`).
            .pool = .{ .tickets = 4096 },
            .direct_parts = true,
            .bank_pool = .{ .decode_rows = self.decode_rows, .spare = &wave_spares },
        });
        errdefer self.stream.deinit();
        // The bank pool: the wave sets' regions take the pool's arrays (a reference each).
        {
            var spares: [scratch_waves.pool_shapes.len]mlx.mlx_array = undefined;
            for (&spares, 0..) |*sp, i| sp.* = self.stream.poolSpare(i) orelse return error.BankPoolSpare;
            try scratch_waves.adoptPool(&spares);
        }
        errdefer scratch_waves.dropPool();
        self.stream.prompt_stage = true;
        // Next-layer staging: the next layer's read-ahead and staged reads, the current call's last and its own stay
        // unclaimed together (the ring doubles again above).
        self.stream.prompt_stage_next = true;
        self.source = xp.StreamSource.init(self.stream);
        errdefer self.source.dropNext();
        self.gates_t3 = &.{};
        self.t3_state = .{ .limit = 0 };
        {
            if (geo.swiglu_limit < 1 or geo.swiglu_limit > 128 or geo.swiglu_limit != @floor(geo.swiglu_limit)) return error.RoutedT3Clamp;
            self.t3_state.limit = @intFromFloat(geo.swiglu_limit);
            self.gates_t3 = try a.alloc(HookT3.Gate, gates.len);
            errdefer a.free(self.gates_t3);
            for (self.gates_t3, gates) |*o, gt| o.* = .{ .w = gt.w, .bias = gt.bias };
            self.hook = try HookT3.initWith(a, &self.g, &self.source, .{ .state = &self.t3_state, .stock = Math.init(self.q, geo) }, geo, .{ .gates = self.gates_t3, .event = event, .wide = wide, .banked = true, .hoist_first = true, .devroute = true });
            errdefer self.hook.deinit();
            // The banked decode's kernels compiled now, not at a request's first launch of each pair count (routed.zig
            // `warm`).
            const w = try self.hook.math.warm(&self.g, self.hook.banks, @intCast(geo.hidden_size), .bfloat16);
            if (w.calls > 0) std.log.info("streamed exl3: the banked decode's kernels compiled at load: {d} calls in {d} ms", .{ w.calls, w.ms });
        }
        errdefer a.free(self.gates_t3);
        errdefer self.hook.deinit();
        // The resident seed (the prompt rows still empty: nothing has routed).
        try self.seedUsage(a, io, dir, @intCast(self.stream.layers.len), self.stream.bank.n_experts);
        return self;
    }

    /// the flat decode rows split across the layers at the flat bill's bytes or
    /// fewer, each layer between its prompt rows and 256, where the split's curves fit the pack (one info line and flat
    /// rows where they do not).
    fn splitDecodeRows(self: *Source) void {
        const dr = @import("../bank/decode_rows.zig");
        const s = self.splitWith(dr) catch |e| {
            std.log.info("streamed exl3: decode rows flat, {d} a layer: the split's curves do not fit this pack ({s})", .{ self.bill.rows.decode, @errorName(e) });
            return;
        };
        std.log.info("streamed exl3: decode rows split by layer: {any}; {d} B against the flat {d} rows' {d} B", .{ self.decode_rows, s.bytes, self.bill.rows.decode, s.flat_bytes });
    }
    fn splitWith(self: *Source, comptime dr: type) dr.Error!dr.Split {
        if (self.bank.layers.len != dr.n_layers) return error.DecodeRowsLayers;
        var bytes: [dr.n_layers]u64 = undefined;
        for (self.bank.layers, &bytes) |l, *b| b.* = l.logical_bytes;
        return dr.split(self.decode_rows, self.prompt_rows, &bytes, self.bank.n_experts);
    }
    /// The resident seed (`resident_seed`, as glm5_next: unset, `usage` when the profile's directory has
    /// expert-usage-v1.json): each layer's empty prompt rows filled with its most used experts, read now; the first
    /// request finds them resident. The seed changes no policy: a seeded expert is a resident as a kept one is
    /// (free seed rows: its row free for the prompt's read-ahead until a route uses it).
    fn seedUsage(self: *Source, a: std.mem.Allocator, io: std.Io, dir: []const u8, layers: u32, experts: u32) !void {
        var st: expert_usage.Settings = .{};
        st.seed = switch (usage_mod.ps().setting.seed) {
            .pack => null,
            .usage => .usage,
            .none => .none,
        };
        var d = try expert_usage.decide(a, io, &st, usage_mod.ps().setting.dirOr(dir), usage_arch, layers, experts);
        defer if (d.profile) |*p| p.deinit();
        const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
        var records: u64 = 0;
        if (d.profile) |*p| records = try self.stream.seedResidents(expert_usage.Order{ .p = p, .rows = self.prompt_rows }, expert_usage.Order.of);
        const ms: u64 = @intCast(@max(@divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds() - t0, std.time.ns_per_ms), 0));
        var buf: [expert_usage.max_label + 400]u8 = undefined;
        const seeded = expert_usage.Seeded.of(self.stream.layers, self.prompt_rows, records);
        std.log.info("streamed exl3: {s}", .{expert_usage.seedLine(&buf, usage_arch, &d, seeded, records * (self.bill.row_bytes / layers), ms)});
        std.log.info("streamed exl3: {d} seeded rows free until a route uses them: the prompt's read-ahead may take them", .{self.stream.freeSeed()});
    }
    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.g.s);
        scratch_waves.releaseAll();
        // The bank pool: the regions' references to the pool's arrays, before the stream frees the pool.
        scratch_waves.dropPool();
        self.source.dropNext();
        self.hook.deinit();
        self.a.free(self.gates_t3);
        self.stream.deinit();
        self.q.deinit(&self.g);
        kernel_set.Set.uninstall(G, &self.g);
        self.set.deinit();
        self.g.deinit();
        self.bank.deinit();
        self.a.free(self.gates);
        self.a.free(self.prompt_rows);
        self.a.free(self.decode_rows);
        self.a.destroy(self);
        if (comptime expert_stream.uses_reader) sdk_ext.expert.giveReader();
        // MLX's settings the phases changed (decode's cache cap, the prompt's buffer split) back for the next model.
        decode_stop.enterPrompt();
        buffer_mb.enterDecode();
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` f32 for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`.
    pub fn rows(self: *Source, layer: u32, x: T, idx: T, wts: T) !T {
        const g = &self.g;
        defer g.reset();
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        self.t3_state.n = 0;
        // The scratch waves: a wide prompt call's waves may take the regions.
        const arena = self.scratchBegin(ids);
        defer if (arena) scratch_waves.callEnd();
        const ro = try self.hook.at(layer).routed(g, x, ids);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        return g.keep(try g.sum(try g.mul(try g.astype(ro, .float32), try g.expandDims(w, -1)), -2, false));
    }

    /// a decode-lane call (at most `decode_route_ids` routed ids) ending in
    /// TAIL_FUSE's kernel: the call's joined rows read in place through their (source 0, row) table, each weighted, the
    /// six folded in the column reduce's order, the shared rows (`shared`, as the model made them, committed behind the
    /// call's waves as SHARED_HOIST commits them) added, rounded to the streams' dtype, then the hc expand of `res` by
    /// `pc` (f32 [n, hc + hc * hc], each row's post then comb): the layer's new streams, the words of `rows`'s take,
    /// weight and sum and the model's `ffn_post_shared`.
    pub fn rowsDecodeExpand(self: *Source, layer: u32, x: T, idx: T, wts: T, shared: T, res: T, pc: T) !T {
        const g = &self.g;
        defer g.reset();
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        self.t3_state.n = 0;
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        {
            const h = &self.hook;
            const lo = try h.at(layer).routedHoistLocated(g, x, ids, &.{});
            // The shared expert committed behind the call's waves, where SHARED_HOIST commits it.
            try g.asyncEval(&.{shared});
            return tail_fuse.expandDecode(g.s, &.{lo.rows}, lo.loc, w, shared, res, pc);
        }
    }
    /// The scratch waves: a wide prompt call (more routed rows than `decode_route_ids`) begins: the regions
    /// made or kept (`scratch_waves.callBegin`); false (the fresh path) in decode, for a narrow call, or where the
    /// regions do not fit.
    fn scratchBegin(self: *Source, ids: T) bool {
        if (self.phase != .prompt) return false;
        const sh = self.g.shapeOf(ids);
        const routed: usize = @intCast(sh.dim(0) * sh.dim(1));
        if (routed <= decode_route_ids) return false;
        return scratch_waves.callBegin() catch |e| blk: {
            std.log.warn("streamed exl3: the prefill waves' regions were refused ({s}); the call's waves take fresh buffers", .{@errorName(e)});
            break :blk false;
        };
    }
    /// JOINLESS's combine is built: `rowsShared` serves the wide calls.
    pub fn joinlessOn(self: *const Source) bool {
        return self.jl != null;
    }
    /// JOINLESS, a wide prompt call (more than `decode_route_ids` ids): `rows` plus `shared` (the layer's shared expert
    /// rows, f32 `[n, hidden]`) as one combine over the call's unjoined outputs, f32 `[n, hidden]`: the words of `rows`'s
    /// sum + `shared`, without the joins.
    pub fn rowsShared(self: *Source, layer: u32, x: T, idx: T, wts: T, shared: T) !T {
        return self.rowsSharedWith(layer, x, idx, wts, shared, null);
    }
    /// TAIL_FUSE: `rowsShared` with the model's hc expand of the streams `res` by `pc` (each row's post, then comb) in the
    /// combine's kernel (tail_fuse.zig): the layer's new streams, a new array the caller owns. `shared`
    /// as the model made it (the kernel widens it as the combine's f32 input was).
    pub fn rowsSharedExpand(self: *Source, layer: u32, x: T, idx: T, wts: T, shared: T, res: T, pc: T) !T {
        return self.rowsSharedWith(layer, x, idx, wts, shared, .{ .res = res, .pc = pc });
    }
    /// `rowsSharedExpand`'s streams written over `res` in place (the model's x1 region) by the kernel's
    /// write-into twin; returns its one-word carrier.
    pub fn rowsSharedExpandInto(self: *Source, layer: u32, x: T, idx: T, wts: T, shared: T, res: T, pc: T) !T {
        return self.rowsSharedWith(layer, x, idx, wts, shared, .{ .res = res, .pc = pc, .into = true });
    }
    const Expand = struct { res: T, pc: T, into: bool = false };
    fn rowsSharedWith(self: *Source, layer: u32, x: T, idx: T, wts: T, shared: T, ex: ?Expand) !T {
        const comb = self.jl orelse return error.JoinlessOff;
        const g = &self.g;
        defer g.reset();
        const ids = if (g.dtypeOf(idx) == .int32) idx else try g.astype(idx, .int32);
        self.t3_state.n = 0;
        const arena = self.scratchBegin(ids);
        defer if (arena) scratch_waves.callEnd();
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        {
            const h = &self.hook;
            const hk = h.at(layer);
            const parts = try hk.routedParts(g, x, ids);
            // The combine holds its sources until it is evaluated.
            defer hk.releaseParts(g);
            for (parts.outs) |o| if (g.dtypeOf(o) != g.dtypeOf(parts.outs[0])) return error.JoinlessMixedSources;
            if (ex) |e| {
                // The new streams over the model's region, the carrier returned.
                if (e.into) return tail_fuse.expandInto(g.s, parts.outs, parts.loc, w, shared, e.res, e.pc);
                return tail_fuse.expand(g.s, parts.outs, parts.loc, w, shared, e.res, e.pc);
            }
            return g.keep(try comb.call(g, parts.outs, parts.loc, w, shared));
        }
    }
    /// A prompt pass's predicted routes of `layer` (every row's ids): its experts, hottest first, read into its rows
    /// while its attention runs.
    /// Next-layer staging: routed layer `layer`'s predicted routes `ids` (an unevaluated int32 [rows, k] array), held
    /// for the call before it: started behind that call's barrier, read once its routes are all made
    /// (`StreamSource.postNext`).
    pub fn nextAhead(self: *Source, layer: u32, ids: mlx.mlx_array) !void {
        if (self.phase != .prompt or !self.stream.prompt_stage_next) return;
        try self.source.setNext(layer, ids);
    }
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.phase != .prompt) return;
        const n = self.bank.n_experts;
        const counts = try self.a.alloc(u32, n);
        defer self.a.free(counts);
        @memset(counts, 0);
        for (ids) |e| counts[e] += 1;
        const ranked = try self.a.alloc(u16, n);
        defer self.a.free(ranked);
        // The staging floor: the staging after the read-ahead sees each expert's votes (the hooks' readAheadSeed is
        // this source's).
        try self.source.readAheadSeedVotes(layer, sdk_ext.expert.policy.rankHottest(counts, ranked), counts);
    }

    /// The rows of `p`: decode grows the slot banks once the prompt's windows are freed; a prompt shrinks them back,
    /// keeping the residents decode left.
    pub fn setPhase(self: *Source, p: Phase) !void {
        if (p == self.phase) return;
        // Next-layer staging: a prediction no call read is let go.
        self.source.dropNext();
        {
            const h = &self.hook;
            try h.flush();
            switch (p) {
                .decode => {
                    // The PARTS route is the prompt's: decode's reads (the grow's fills, its misses) stage and copy.
                    self.stream.directPartsLive(false);
                    // The gate/up split is decode's: its reads (the grow's fills, its misses) split from here.
                    self.stream.guSplit(1);
                    // Decode's lookahead behind its demand from here, a claimed record's untaken pieces read by the demand.
                    self.stream.demandFirst(2);
                    // The scratch waves: the prompt's regions let go (MLX keeps a buffer while a command uses it).
                    scratch_waves.releaseAll();
                    self.released = (try h.releaseTransient()) > 0;
                    _ = mlx.mlx_synchronize(self.g.s);
                    self.g.clearCache();
                    try h.grow(&self.g, self.decode_rows);
                    // The decode stop: decode's cache capped where the bill took the stop's bytes back.
                    decode_stop.enterDecode();
                    // The prompt command-buffer split: MLX's own split back for decode (the prompt's work done above).
                    buffer_mb.enterDecode();
                },
                .prompt => {
                    // The decode stop: the prompt's cache cap back before decode's rows go.
                    decode_stop.enterPrompt();
                    // The prompt command-buffer split: the prompt's split, before its first op.
                    buffer_mb.enterPrompt();
                    self.stream.directPartsLive(true);
                    // The prompt's reads whole.
                    self.stream.guSplit(0);
                    // Demand first: off for the prompt's pool.
                    self.stream.demandFirst(0);
                    // `shrink` frees the grown rows and checks MLX's active bytes dropped by them, so no command may
                    // still hold them (the grow's own fills, a decode's last eval): a phase change right after the
                    // load's grow otherwise fails at random with GrownRowsStillReferenced and leaves the stream failed.
                    _ = mlx.mlx_synchronize(self.g.s);
                    _ = try h.shrink(self.prompt_rows, .keep);
                    _ = mlx.mlx_synchronize(self.g.s);
                    self.g.clearCache();
                    _ = try h.regrowTransient(&self.g, self.released);
                },
            }
        }
        self.phase = p;
    }

    /// After the eval that consumed the forward's calls.
    pub fn settle(self: *Source) !void {
        try self.hook.flush();
    }

    pub fn stats(self: *Source) expert_stream.Stats {
        return self.source.stats();
    }
};

/// `x`'s words as `dt` (MLX's view: the same buffer, bitcast), in `g`'s scope.
fn viewAs(g: *G, x: T, dt: mlx.mlx_dtype) !T {
    var r = mlx.mlx_array_new();
    mlx.check(mlx.mlx_view(&r, x, dt, g.s)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return g.adopt(r);
}
