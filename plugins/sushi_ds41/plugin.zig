//! DeepSeek-V4.1-Flash on sushi's own model code (`src/sushi/`, vendored): its STQ trunk, its rate-grouped experts
//! streamed from the pack's shards, and its layer pipeline. The plugin loads its own tensors (the host keeps none), so
//! the speed is sushi's. One request's state at a time; the host's table is the only seam.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");

// The vendored files reach these through `mlx_host` (the exl3 module) or by relative import.
pub const mlx = @import("src/sushi/mlx.zig");
pub const log = @import("src/sushi/log.zig");
pub const io_util = @import("src/sushi/io_util.zig");
const dsv = @import("src/sushi/dsv41/dsv41_model.zig");
const dstream = @import("src/sushi/dsv41/dsv41_stream.zig");
const source = @import("src/sushi/dsv41/dsv41_source.zig");
const engram = @import("src/sushi/dsv41/dsv41_engram.zig");
const stream = @import("src/sushi/expert_stream.zig");
const smodel = @import("src/sushi/model.zig");
const sc = @import("src/sushi/dsv41/dsv41_config.zig");
const vis = @import("src/sushi/dsv41/dsv41_vision.zig");
const rows_policy = @import("src/sushi/dsv41/dspark_rows.zig");
const borrow = @import("src/sushi/dsv41/dsv41_borrow.zig");
const depth_bounds = @import("src/sushi/mtp_depth_bounds.zig");
const vision_common = @import("src/sushi/vision_common.zig");
const attn_nax = @import("src/attn_nax.zig");
const index_nax = @import("src/index_nax.zig");
const vision_key = @import("src/vision_key.zig");
const copy_lookup = @import("src/copy_lookup.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "sushi_ds41",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout, .state_bytes = &stateBytes, .aux = &aux_hooks },
    .spec = &.{ .native_drafter = &native_drafter },
    .media = &.{ .tower = &mediaTower, .prepare = &mediaPrepare, .placeholder_rows = &placeholderRows },
    .weights = &.{ .keep_key = &keepNone, .resident_bytes = &residentBytes },
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Model = struct {
    cfg: sc.Config,
    weights: smodel.Weights,
    mdl: *dsv.Model,
    engine: *stream.Engine,
    experts: *dstream.StqStream,
    tower: ?vis.Dsv41Vision = null,
    s: mlx.mlx_stream,
    /// The raw token ids the state holds (the state's own history is compressed ids).
    ids: std.ArrayList(u32) = .empty,
    /// `ids` by n-gram, for the round's copy arm; follows `ids` lazily and is cleared with it (`clearIds`).
    lookup: copy_lookup.Index = .{},
    /// The prompt being prefilled: its restore point is taken where the state stands at its last token.
    arm: ?Arm = null,
    /// A completed capture waiting for the engine to take it (`auxSnapshot`).
    snap: ?*Snap = null,
    /// How many of a drafted block's rows a round verifies, and when drafting is not worth it (per request).
    policy: rows_policy.Policy = .{},
    /// A round whose verify the engine is accepting (`NativeDrafter.verify` until `finish`).
    open: ?Open = null,
    /// This request's copy rounds, the drafts they verified and kept; logged when the next request resets the state.
    copy: CopyStats = .{},
    prev_cache_limit: ?usize = null,
    /// The MLX buffer-pool cap in force (`borrow.poolCap` of the last pass's rows): `pool_bytes` at decode widths.
    pool_now: u64 = pool_bytes,
    /// Positions the expert cache was sized to leave room for (a request's state and one prefill chunk's working set).
    billed_context: u64 = default_context,
};

// ── bill ──

fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    const c = config.of(p);
    out.* = .{ .kv_per_token = c.stateBytesPerToken(), .fixed_slot_bytes = c.stateFixedBytes(false), .prefill_attn_keys = &attnKeys, .prefill_bytes = &prefillBytes };
}

fn attnKeys(p: *const abi.Parsed, seq: u64) callconv(.c) u64 {
    const c = config.of(p);
    return @min(seq, @as(u64, c.window) + c.index_topk + 1);
}

fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    _ = kv_bits;
    _ = warm;
    return config.of(p).chunkCap(chunk, seq);
}

/// Window layers keep a ring; a compressed layer reads the last KV source at or above it.
fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.layers, cap);
    var src: i16 = -1;
    for (0..first) |li| if (c.isKvSource(li)) {
        src = @intCast(li);
    };
    for (out[0..n], first..) |*l, li| {
        const ratio = c.ratios[li];
        const own = c.isKvSource(li);
        if (own) src = @intCast(li);
        l.* = if (ratio == 0)
            .{ .kind = .sliding, .window = c.window, .kv_heads = 1, .k_dim = c.head_dim }
        else
            .{ .kind = .full, .kv_source = if (own) -1 else src, .kv_heads = 1, .k_dim = c.head_dim / ratio };
    }
    return n;
}

// ── weights ──

fn keepNone(p: *const abi.Parsed, key: abi.Str) callconv(.c) bool {
    _ = p;
    _ = key;
    return false;
}

fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, settings_json: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    _ = settings_json;
    _ = shards;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = model_dir.slice();
    const trunk = source.residentBytes(io, gpa, dir) catch |e| return Verdict.named(.failed, @errorName(e));
    const tower = if (config.of(p).vision != null) source.visionBytes(io, gpa, dir, vision_common.vision_bf16) catch |e| return Verdict.named(.failed, @errorName(e)) else 0;
    const stages = if (config.private(p).dspark) source.dsparkBytes(io, gpa, dir) catch |e| return Verdict.named(.failed, @errorName(e)) else 0;
    out.* = trunk + tower + stages;
    return .ok;
}

// ── load ──

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        log.err("[sushi_ds41] load failed: {s}\n", .{@errorName(e)});
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "sushi_ds41: out of memory"),
            error.SsdBudgetBelowResident => Verdict.refuse(.too_large, "sushi_ds41: the streamed experts do not fit the free memory"),
            error.MetalOnly => Verdict.refuse(.unsupported, "sushi_ds41: its kernels run on Metal only"),
            else => Verdict.refuse(.failed, "sushi_ds41: the pack does not load"),
        };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Model {
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    if (!mlx.streamIsGpu(s)) return error.MetalOnly;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = ctx.model_dir.slice();
    const priv = config.private(ctx.parsed);
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .cfg = priv.cfg, .weights = undefined, .mdl = undefined, .engine = undefined, .experts = undefined, .s = s };
    const c = &m.cfg;

    var geo: stream.Geometry = .{
        .layers = @intCast(c.layers),
        .experts = @intCast(c.experts),
        .hidden = c.dim,
        .intermediate = c.moe_inter,
        .exl3_n = priv.exl3_n,
        .exl3_rates = priv.exl3_rates,
    };
    const per = try stream.expertBillFor(gpa, dir, geo, .exl3_k4);
    geo.shape = per.shape;
    geo.shape.pool_bytes = pool_bytes;
    m.weights = try source.loadResident(io, gpa, dir, .{ .dspark = priv.dspark, .vision = c.vision != null });
    errdefer m.weights.deinit();
    // The stages' storage is built after the room is read (the model builds it from tensors already loaded).
    const stages = if (priv.dspark) try source.dsparkBytes(io, gpa, dir) else 0;
    m.billed_context = ctxSetting(ctx.settings_json.slice()) orelse default_context;
    const room = try expertRoom(ctx, c, m.billed_context) -| stages;
    const ledger = try stream.budgetLedger(room, engram.cacheBytes(c.engram), 0, 0, geo.layers, geo.experts, @intCast(c.top_k), per.expert_bytes, stream.BOUNCE_BYTES, geo.shape);
    const plan = try stream.cachePlanBytesForGeometry(ledger.cache_bytes, geo, per.expert_bytes);
    log.info("[sushi_ds41] expert cache {d:.2} GB ({d} slots a layer) in {d:.2} GB of room\n", .{ gb(plan.cache_bytes), ledger.slots_per_layer, gb(room) });

    m.engine = try gpa.create(stream.Engine);
    errdefer gpa.destroy(m.engine);
    m.engine.* = try stream.Engine.initWithOptions(gpa, dir, geo, plan.cache_bytes, s, .{ .layout = .exl3_k4, .tile_major = true });
    errdefer m.engine.deinit();
    // Every pool preloaded with the pack's most used experts, as sushi's server does: the first requests find them.
    m.engine.warmCache() catch |e| log.warn("[sushi_ds41] expert cache warm failed: {s}\n", .{@errorName(e)});
    m.mdl = try dsv.init(gpa, m.cfg, &m.weights, dir, s, .{ .experts = false, .dspark = priv.dspark });
    errdefer m.mdl.deinit();
    const nax = attn_nax.fits(c.heads, c.head_dim, .bfloat16);
    log.info("[sushi_ds41] wide-span attention: {s}\n", .{if (nax) "matmul2d (the M5's matrix units)" else "attnFused (simdgroup matrices)"});
    const idx_nax = index_nax.fits(c.index_heads, c.index_head_dim, .bfloat16);
    log.info("[sushi_ds41] prefill index scores: {s}\n", .{if (idx_nax) "matmul2d (the M5's matrix units)" else "indexScoresTile (simdgroup matrices)"});
    // As sushi's server does: a synced per-stage `[dsv41-prof]` line per wide chunk, and with the verify profile per
    // decode-width chunk too.
    const verify_profile = std.c.getenv("SUSHI_DSV41_VERIFY_PROFILE") != null;
    if (verify_profile or std.c.getenv("SUSHI_DSV41_PREFILL_PROFILE") != null) {
        const pf = try gpa.create(dsv.Profile);
        pf.* = dsv.Profile.init(verify_profile);
        m.mdl.profile = pf;
    }
    // The tower borrows its tensors from `m.weights`, which the model left them in.
    if (c.vision != null) m.tower = try vis.Dsv41Vision.init(gpa, .{ .dsv41 = c.* }, &m.weights);
    errdefer if (m.tower) |*t| t.deinit();
    dsv.logMemory(m.mdl);
    m.experts = try gpa.create(dstream.StqStream);
    errdefer gpa.destroy(m.experts);
    m.experts.* = .{ .engine = m.engine, .cfg = &m.mdl.cfg };
    m.mdl.experts = .{ .hook = m.experts.hook() };
    if (m.engine.startPrefetch()) {
        m.mdl.prefetch_lead_rows = 3;
    } else |e| log.warn("[sushi_ds41] expert prefetch off: {s}\n", .{@errorName(e)});

    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(pool_bytes));
    m.prev_cache_limit = prev;
    // A failed warm costs the first requests their compiles, not the load.
    const had_error = mlx.errorPending();
    dsv.warmKernels(m.mdl) catch |e| {
        mlx.dropLatchedErrorUnless(had_error);
        log.warn("[sushi_ds41] kernel warm failed ({s}); requests compile their own\n", .{@errorName(e)});
    };
    _ = mlx.mlx_clear_cache();
    return m;
}

/// MLX's freed-buffer pool during decode, which the ledger bills.
const pool_bytes: u64 = 1 << 30;
/// The pool a prefill chunk of `sc.prefill_width` rows parks. At `pool_bytes` a 2048-row chunk frees and remakes its
/// buffers op after op, which halves prefill. sushi bounds a wide forward's pool with its MLX transient cap, which the
/// host's MLX does not have, so here the pool keeps a cap of its own and the expert room bills it.
const prefill_pool_bytes: u64 = 8 << 30;
/// The prompt positions the expert cache leaves room for when the model sets no `ctx_size`.
const default_context: u64 = 131072;
/// The host's default prefill chunk, which its admission bills.
const host_chunk: u64 = 8192;
/// What the box keeps unplanned beside the wired margin: the stop under the ceiling, the process's heap and the stream's
/// read buffers.
const host_bytes: u64 = 2_000_000_000 + (1 << 30);
/// Positions a request generates past its prompt.
const generation_rows: u64 = 8192;

fn gb(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// The GPU bytes left for the experts once the residents are loaded: the host's ceiling less its wired margin, the
/// process overhead and one request of `context` positions (its state and one prefill chunk's working set).
fn expertRoom(ctx: *const abi.LoadCtx, c: *const sc.Config, context: u64) !u64 {
    const ceiling = if (kit.host.Host.of(ctx.host)) |h| h.memoryCeiling() else |_| mlx.maxRecommendedWorkingSet();
    const request = c.chunkCap(host_chunk, context) +| (context +| generation_rows) *| c.stateBytesPerToken() +| c.stateFixedBytes(false) +|
        (prefill_pool_bytes - pool_bytes);
    return ceiling -| ctx.wired_margin_bytes -| host_bytes -| request;
}

/// The per-model `ctx_size` setting.
fn ctxSetting(settings_json: []const u8) ?u64 {
    if (settings_json.len == 0) return null;
    const S = struct { ctx_size: ?u64 = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.ctx_size;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (m.prev_cache_limit) |prev| {
        var unused: usize = 0;
        _ = mlx.mlx_set_cache_limit(&unused, prev);
    }
    if (m.snap) |sn| freeSnap(sn);
    if (m.tower) |*t| t.deinit();
    m.mdl.deinit();
    m.engine.deinit();
    gpa.destroy(m.engine);
    gpa.destroy(m.experts);
    m.weights.deinit();
    m.ids.deinit(gpa);
    m.lookup.deinit(gpa);
    gpa.destroy(m);
}

// ── forward ──

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (f.mode == .verify or f.mode == .batched_rows) return Verdict.refuse(.unsupported, "sushi_ds41: no speculative verify or batched rows");
    if (f.capture_layers.len > 0 or f.want_hidden) return Verdict.refuse(.unsupported, "sushi_ds41: no layer capture or hidden rows (not an embedding model)");
    if (f.slots.len > 1 or f.rows == 0) return Verdict.refuse(.unsupported, "sushi_ds41: one slot's rows at a time");
    const host = hostIds(m, f) catch return Verdict.refuse(.failed, "sushi_ds41: the token ids do not read back");
    defer host.deinit();
    const pos: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    if (pos == 0) reset(m);
    const st = dsv.ensureState(m.mdl) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    if (st.n != pos) return Verdict.refuse(.failed, "sushi_ds41: the position does not follow the model's state");
    if (pos + host.ids.len > m.billed_context + generation_rows) return Verdict.refuse(.failed, "sushi_ds41: the request is longer than the context the expert cache was sized for (raise ctx_size)");
    for (host.ids) |t| if (t >= m.cfg.vocab) return Verdict.refuse(.failed, "sushi_ds41: token id past the vocabulary");
    capture(m) catch return Verdict.refuse(.failed, "sushi_ds41: the restore point could not be taken");
    const want: dsv.Want = if (f.logits == .all_rows) .all else .last;
    const rows: ?dsv.VisionRows = if (f.media.ctx != null) .{ .rows = .{ .ctx = f.media.ctx }, .offset = f.media_offset } else null;
    // A host older than `Forward.progress` passes a shorter struct.
    const progress: ?*u64 = if (f.size >= @offsetOf(abi.Forward, "progress") + @sizeOf(?*u64)) f.progress else null;
    const logits = run(m, host.ids, want, rows, progress) catch |e| {
        log.err("[sushi_ds41] forward failed: {s}\n", .{@errorName(e)});
        return Verdict.refuse(.failed, "sushi_ds41: the forward failed");
    };
    if (f.logits == .none) _ = mlx.mlx_array_free(logits) else out.logits = .{ .ctx = logits.ctx };
    return .ok;
}

const HostIds = struct {
    ids: []const u32,
    flat: mlx.mlx_array,

    fn deinit(self: HostIds) void {
        if (self.flat.ctx != null) _ = mlx.mlx_array_free(self.flat);
    }
};

/// The forward's token ids on the host: the span, or a still-lazy device array read back.
fn hostIds(m: *Model, f: *const abi.Forward) !HostIds {
    if (f.ids.ctx == null) return .{ .ids = f.tokens.slice(), .flat = .{ .ctx = null } };
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, .{ .ctx = f.ids.ctx }, .uint32, m.s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const data = mlx.mlx_array_data_uint32(flat) orelse return error.InvalidInput;
    return .{ .ids = data[0..mlx.mlx_array_size(flat)], .flat = flat };
}

/// Empties the token history and its n-gram index together: a history that is replaced, not extended, must not keep
/// the old one's n-grams.
fn clearIds(m: *Model) void {
    m.ids.clearRetainingCapacity();
    m.lookup.clear();
}

const CopyStats = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0 };

fn reset(m: *Model) void {
    if (m.copy.rounds > 0) log.info("[sushi_ds41] copy rounds={d} drafted={d} accepted={d}\n", .{ m.copy.rounds, m.copy.drafted, m.copy.accepted });
    m.copy = .{};
    dsv.resetState(m.mdl);
    m.open = null;
    clearIds(m);
    m.policy = rows_policy.Policy.init(depth_bounds.active, m.cfg.dspark.block);
}

/// A prompt as sushi runs it: all but its last token in chunks (`prefill_width`, a short tail merged into the last), each
/// told what follows it so the next chunk's Engram rows read while it runs, then the last token alone for the logits.
/// Logits `[rows, vocab]`. `progress` gets the rows forwarded after each chunk (`Forward.progress`).
fn run(m: *Model, ids: []const u32, want: dsv.Want, media: ?dsv.VisionRows, progress: ?*u64) !mlx.mlx_array {
    var rows = media;
    const prefix = if (want == .last and ids.len > 1) ids.len - 1 else 0;
    var at: usize = 0;
    while (at < prefix) {
        var end = @min(at + sc.prefill_width, prefix);
        if (end < prefix and prefix - end < sc.tail_merge_rows) end = prefix;
        const next = if (end < prefix) @min(end + sc.prefill_width, prefix) else ids.len;
        // Another chunk of the prefix follows and is wider than the window (the merge keeps tails of `tail_merge_rows`
        // rows), so its CED tail overwrites this chunk's decoder rows before anything reads them: run this one without
        // its decoder (`extendState`).
        if (end < prefix and prefix - end > m.mdl.cfg.window and m.mdl.ced_split != null) try stateRows(m, ids[at..end], ids[end..next], rows) else _ = mlx.mlx_array_free(try forwardRows(m, ids[at..end], .last, ids[end..next], rows));
        if (rows) |*r| r.offset += std.mem.count(u32, ids[at..end], &.{m.cfg.image_token_id});
        at = end;
        if (progress) |p| @atomicStore(u64, p, at, .monotonic);
    }
    try capture(m);
    const logits = try forwardRows(m, ids[prefix..], want, &.{}, rows);
    defer _ = mlx.mlx_array_free(logits);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, logits, &[_]c_int{ -1, @intCast(m.cfg.vocab) }, 2, m.s));
    return o;
}

/// One pass over `rows`; the logits are the caller's to free (an intermediate chunk's are dropped).
fn forwardRows(m: *Model, rows: []const u32, want: dsv.Want, ahead: []const u32, media: ?dsv.VisionRows) !mlx.mlx_array {
    poolForRows(m, rows.len);
    m.engine.beginForward();
    const logits = try dsv.extendAhead(m.mdl, rows, want, dsv.SPAN, media, ahead);
    m.engine.finishForward(rows.len);
    try m.ids.appendSlice(gpa, rows);
    return logits;
}

/// `forwardRows` for a chunk that leaves only state behind (`dsv.extendState`).
fn stateRows(m: *Model, rows: []const u32, ahead: []const u32, media: ?dsv.VisionRows) !void {
    poolForRows(m, rows.len);
    m.engine.beginForward();
    try dsv.extendState(m.mdl, rows, dsv.SPAN, media, ahead);
    m.engine.finishForward(rows.len);
    try m.ids.appendSlice(gpa, rows);
}

/// Moves MLX's buffer-pool cap to the one a pass of `rows` parks, as sushi's server does per forward
/// (`dsv41_borrow.enterForward`): the decode cap up to a verify block, the prefill cap above it. A lower cap trims
/// lazily, so the clear makes the parked pool what the expert room billed.
fn poolForRows(m: *Model, rows: usize) void {
    const want = borrow.poolCap(rows, pool_bytes, prefill_pool_bytes);
    if (want == m.pool_now) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(want));
    if (want < m.pool_now) _ = mlx.mlx_clear_cache();
    m.pool_now = want;
}

fn stateBytes(mp: *anyopaque) callconv(.c) u64 {
    const m: *Model = @ptrCast(@alignCast(mp));
    return if (m.mdl.dec_state) |*st| dsv.stateBytes(st) else 0;
}

pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return error.NoFixture;
}

test {
    _ = config;
    _ = attn_nax;
}

// ── prefix cache ──

/// `vision_key` is the key of the target's ids under the prompt's image rows (`vision_key.prefixKey`), taken when the
/// prompt began: the host lends the rows to that call only.
const Arm = struct { target: usize, capture: bool, vision_key: u64 = 0 };

/// A restore point: the model's whole state at `n` (`dsv.captureState`), the tokens it was built from, and the key of
/// the image rows those tokens splice in (0 for none), which the host matches beside the ids.
const Snap = struct { slots: []dsv.Slot, n: usize, ids: []u32, vision_key: u64 = 0 };

fn freeSnap(sn: *Snap) void {
    for (sn.slots) |*sl| dsv.freeSlot(sl);
    gpa.free(sn.slots);
    gpa.free(sn.ids);
    gpa.destroy(sn);
}

/// Takes the armed restore point when the state stands at the prompt's last token.
fn capture(m: *Model) !void {
    const a = m.arm orelse return;
    if (!a.capture or m.snap != null) return;
    const st = try dsv.ensureState(m.mdl);
    if (st.n != a.target or m.ids.items.len != a.target) return;
    const sn = try gpa.create(Snap);
    errdefer gpa.destroy(sn);
    sn.n = a.target;
    sn.vision_key = a.vision_key;
    sn.ids = try gpa.dupe(u32, m.ids.items);
    errdefer gpa.free(sn.ids);
    sn.slots = try gpa.alloc(dsv.Slot, dsv.slotCount(m.mdl));
    errdefer gpa.free(sn.slots);
    @memset(sn.slots, dsv.nil_slot);
    try dsv.captureState(m.mdl, sn.slots);
    m.snap = sn;
    m.arm.?.capture = false;
}

fn snapOf(h: abi.Handle) ?*Snap {
    return @ptrCast(@alignCast(h));
}

const aux_hooks = abi.AuxHooks{
    .enabled = &auxEnabled,
    .describe = &auxDescribe,
    .prefix_key = &auxPrefixKey,
    .begin_prompt = &auxBeginPrompt,
    .reserve = &auxOk,
    .commit = &auxCommit,
    .snapshot = &auxSnapshot,
    .restore_at = &auxRestoreAt,
    .slice = &auxSlice,
    .handoff = &auxHandoff,
    .free = &auxFree,
    .bytes = &auxBytes,
    .import_plan = &auxImportPlan,
    .export_arrays = &auxExportArrays,
    .import_arrays = &auxImportArrays,
};

fn auxEnabled(_: *anyopaque) callconv(.c) bool {
    return true;
}

fn auxOk(_: *anyopaque, _: u32, _: u64) callconv(.c) Verdict {
    return .ok;
}

fn auxCommit(mp: *anyopaque, _: u32, len: u64) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const st = m.mdl.dec_state orelse return Verdict.refuse(.failed, "sushi_ds41: no live state to commit");
    if (len != st.n and (m.snap == null or m.snap.?.n != len)) return Verdict.refuse(.unsupported, "sushi_ds41: commit needs a completed boundary");
    return .ok;
}

/// The key of a prefix beside its ids: a hash of the image rows it covers, so two images (the same placeholder ids)
/// never share a restore point.
fn auxPrefixKey(mp: *anyopaque, ids: abi.Span(u32), rows: abi.Array, out: *u64) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = 0; // a refused key leaves 0
    out.* = keyOf(m, ids.slice(), rows) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn keyOf(m: *const Model, ids: []const u32, rows: abi.Array) !u64 {
    const r: ?mlx.mlx_array = if (rows.ctx != null) .{ .ctx = rows.ctx } else null;
    return vision_key.prefixKey(ids, m.cfg.image_token_id, r, m.s);
}

fn auxBeginPrompt(mp: *anyopaque, _: u32, prompt: abi.Span(u32), media: abi.Array, restored_at: u64, capture_on: bool) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (prompt.len == 0) return Verdict.refuse(.failed, "sushi_ds41: an empty prompt");
    // The restore point covers all but the last token; its rows are lent to this call only, so the key is taken now.
    const ids = prompt.slice()[0 .. prompt.len - 1];
    const key = if (capture_on and ids.len > 0) keyOf(m, ids, media) catch |e| return Verdict.named(.failed, @errorName(e)) else 0;
    if (restored_at == 0) {
        reset(m);
    } else {
        const st = dsv.ensureState(m.mdl) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
        if (st.n != restored_at or restored_at >= prompt.len or m.ids.items.len != restored_at)
            return Verdict.refuse(.failed, "sushi_ds41: the restored position differs from the prompt boundary");
    }
    if (m.snap) |sn| {
        freeSnap(sn);
        m.snap = null;
    }
    m.arm = .{ .target = prompt.len - 1, .capture = capture_on, .vision_key = key };
    return .ok;
}

fn auxSnapshot(mp: *anyopaque, _: u32, out: *abi.Handle) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    out.* = m.snap;
    m.snap = null;
    return .ok;
}

fn auxRestoreAt(mp: *anyopaque, _: u32, h: abi.Handle, pos: u64) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const sn = snapOf(h) orelse return Verdict.refuse(.unsupported, "sushi_ds41: no snapshot to restore");
    if (pos != sn.n) return Verdict.refuse(.unsupported, "sushi_ds41: restore needs the snapshot's exact position");
    clearIds(m);
    m.ids.appendSlice(gpa, sn.ids) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    dsv.restoreState(m.mdl, sn.slots, sn.n, sn.ids) catch |e| {
        clearIds(m);
        return Verdict.named(.failed, @errorName(e));
    };
    return .ok;
}

fn auxSlice(_: *anyopaque, h: abi.Handle, len: u64, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    const sn = snapOf(h) orelse return Verdict.refuse(.unsupported, "sushi_ds41: no snapshot to slice");
    if (len != sn.n) return Verdict.refuse(.unsupported, "sushi_ds41: a snapshot cannot be shortened through its rings");
    const copy = gpa.create(Snap) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    copy.n = sn.n;
    copy.vision_key = sn.vision_key;
    copy.ids = gpa.dupe(u32, sn.ids) catch {
        gpa.destroy(copy);
        return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    };
    copy.slots = gpa.alloc(dsv.Slot, sn.slots.len) catch {
        gpa.free(copy.ids);
        gpa.destroy(copy);
        return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    };
    @memset(copy.slots, dsv.nil_slot);
    for (sn.slots, copy.slots) |from, *to| for (from, 0..) |x, i| if (x.ctx != null) {
        to[i] = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&to[i], x);
    };
    out.* = copy;
    return .ok;
}

/// One state and no per-slot identity: a slot's restore point serves whichever slot continues it.
fn auxHandoff(_: *anyopaque, _: u32, _: u32) callconv(.c) Verdict {
    return .ok;
}

fn auxFree(_: *anyopaque, h: abi.Handle) callconv(.c) void {
    if (snapOf(h)) |sn| freeSnap(sn);
}

fn auxBytes(_: *anyopaque, h: abi.Handle) callconv(.c) u64 {
    const sn = snapOf(h) orelse return 0;
    var total: u64 = sn.ids.len * 4;
    for (sn.slots) |sl| {
        for (sl) |x| total += dsv.arrayBytes(x);
    }
    return total;
}

fn auxDescribe(_: *anyopaque, h: abi.Handle, out: *abi.AuxSnapshotInfo) callconv(.c) Verdict {
    const sn = snapOf(h) orelse return Verdict.refuse(.unsupported, "sushi_ds41: no snapshot to describe");
    out.* = .{ .n = sn.n, .ids = .of(sn.ids), .prefix_key = sn.vision_key };
    return .ok;
}

fn auxImportPlan(_: *anyopaque, _: abi.Str, peak: *u64) callconv(.c) Verdict {
    peak.* = 0;
    return Verdict.refuse(.unsupported, "sushi_ds41: snapshot import is unavailable");
}

fn auxExportArrays(_: *anyopaque, _: abi.Handle, out: *abi.AuxExport) callconv(.c) Verdict {
    out.tensor_len = 0;
    out.names_len = 0;
    out.header_len = 0;
    return Verdict.refuse(.unsupported, "sushi_ds41: snapshot export is unavailable");
}

fn auxImportArrays(_: *anyopaque, _: abi.Span(u32), _: abi.Str, _: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    return Verdict.refuse(.unsupported, "sushi_ds41: snapshot import is unavailable");
}

test "sushi_ds41 aux: a restore point's image key survives describe and slice; a prefix with no image keys 0" {
    const testing = std.testing;
    const sn = try gpa.create(Snap);
    sn.* = .{ .slots = try gpa.alloc(dsv.Slot, 0), .n = 3, .ids = try gpa.dupe(u32, &.{ 3, 17, 4 }), .vision_key = 0xabc };
    defer freeSnap(sn);
    var m: Model = undefined;
    m.cfg.image_token_id = 17;
    var info: abi.AuxSnapshotInfo = .{};
    try testing.expect(auxDescribe(&m, sn, &info).isOk());
    try testing.expectEqual(@as(u64, 0xabc), info.prefix_key);
    var copy: abi.Handle = null;
    try testing.expect(auxSlice(&m, sn, 3, &copy).isOk());
    defer auxFree(&m, copy);
    try testing.expectEqual(@as(u64, 0xabc), snapOf(copy).?.vision_key);
    try testing.expect(auxDescribe(&m, copy, &info).isOk());
    try testing.expectEqual(@as(u64, 0xabc), info.prefix_key);
    // Neither case reaches MLX: no rows, or rows (a pointer never read) but no image token among the ids.
    var key: u64 = 1;
    try testing.expect(auxPrefixKey(&m, .of(&.{ 3, 17, 4 }), .{}, &key).isOk());
    try testing.expectEqual(@as(u64, 0), key);
    key = 1;
    try testing.expect(auxPrefixKey(&m, .of(&.{ 3, 4, 5 }), .{ .ctx = @ptrFromInt(0x10) }, &key).isOk());
    try testing.expectEqual(@as(u64, 0), key);
}

// ── images ──

/// The host decodes the image to packed RGB8 (one frame); the release's processor bounds it by its own token budget.
fn mediaPrepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    _ = budget;
    if (in.kind != .image or in.frames > 1) return Verdict.refuse(.unsupported, "sushi_ds41: images only");
    const v = config.of(p).vision orelse return Verdict.refuse(.unsupported, "sushi_ds41: this pack has no vision tower");
    const rgb = in.rgb.slice();
    if (in.width == 0 or in.height == 0 or rgb.len < @as(usize, in.width) * in.height * 3) return Verdict.refuse(.failed, "sushi_ds41: the host's image has no pixels");
    const px = vis.preprocess(gpa, rgb, in.height, in.width, vis.Processor.of(v)) catch return Verdict.refuse(.failed, "sushi_ds41: the image's dimensions have no grid");
    // The host copies these float32 bytes on a connection thread (no MLX there) and calls `releasePixels`.
    out.* = .{
        .kind = .image,
        .rows = @intCast(vis.spanRows(px.grid_h, px.grid_w, v.downsample)),
        .grid = .{ 1, px.grid_h, px.grid_w },
        .bytes = .of(px.bytes),
        .free = &releasePixels,
    };
    return .ok;
}

fn releasePixels(item: *abi.MediaItem) callconv(.c) void {
    gpa.free(item.bytes.slice());
    item.bytes = .{};
}

/// The tower reads the host's pixels as a float32 `[grid_h * grid_w, feat]` array it makes on the inference thread.
fn mediaTower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "sushi_ds41: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "sushi_ds41: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

/// Every position of an image's span carries the image token; the span's delimiters are the tower's learned rows.
fn placeholderRows(p: *const abi.Parsed, item: *const abi.MediaItem, _: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
    const n = @min(item.rows, cap);
    @memset(out[0..n], config.of(p).image_token_id);
    return n;
}

// ── DSpark ──

const native_drafter = abi.NativeDrafter{ .step = &nativeStep, .verify = &nativeVerify, .finish = &nativeFinish };

/// Drafts a round can verify, and the widest verify the model's rings allow.
const max_drafts = 16;
const max_verify_rows: usize = sc.ring_slack;

fn nanosSince(start: std.Io.Timestamp) f32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @floatCast(@as(f64, @floatFromInt(start.untilNow(io, .awake).nanoseconds)) / 1e6);
}

fn now() std.Io.Timestamp {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .awake);
}

/// Checks shared by the round's entry points; the state the round starts from.
fn roundEntry(m: *Model, f: *const abi.Forward, out: *const abi.ForwardOut) ?Verdict {
    if (out.size < @sizeOf(abi.ForwardOut)) return Verdict.refuse(.unsupported, "sushi_ds41: the host's ForwardOut has no round");
    if (m.mdl.dspark == null) return Verdict.refuse(.unsupported, "sushi_ds41: the DSpark stages are not loaded");
    if (f.tokens.len != 1) return Verdict.refuse(.unsupported, "sushi_ds41: a round starts from one committed token");
    // A verify the engine never finished (its accept failed) is discarded; the position check below then refuses a
    // stale request.
    if (m.open != null) {
        m.open = null;
        dsv.finishVerify(m.mdl, 0) catch {};
    }
    const st = dsv.ensureState(m.mdl) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    if (f.positions.len > 0 and st.n != f.positions.slice()[0]) return Verdict.refuse(.failed, "sushi_ds41: the position does not follow the model's state");
    if (st.n + max_verify_rows > m.billed_context + generation_rows) return Verdict.refuse(.failed, "sushi_ds41: the request is longer than the context the expert cache was sized for (raise ctx_size)");
    return null;
}

fn roundFailed(e: anyerror) Verdict {
    log.err("[sushi_ds41] draft round failed: {s}\n", .{@errorName(e)});
    return Verdict.refuse(.failed, "sushi_ds41: the draft round failed");
}

/// One greedy round: the stages draft a block from `t1`, a verify chunk `[t1, drafts]` reads it through the trunk, the
/// longest prefix of drafts the trunk's own argmax agrees with stays and the trunk's token after it is pending.
fn nativeStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (roundEntry(m, f, out)) |refused| return refused;
    const suppress: ?mlx.mlx_array = if (f.suppress_mask.ctx != null) .{ .ctx = f.suppress_mask.ctx } else null;
    var open = openRound(m, f.tokens.slice()[0], f.accept_cap) catch |e| return roundFailed(e);
    defer _ = mlx.mlx_array_free(open.logits);
    const best = argmaxRows(m, open.logits, suppress, open.rows) catch |e| return roundFailed(e);
    defer gpa.free(best);
    var accepted: usize = 0;
    while (accepted + 1 < open.rows and best[accepted] == open.ids[1 + accepted]) accepted += 1;
    closeRound(m, &open, accepted, &out.round) catch |e| return roundFailed(e);
    out.round.next_token = best[accepted];
    return .ok;
}

/// The first half of a round the engine accepts under its own sampling: the verified rows and their logits.
fn nativeVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (roundEntry(m, f, out)) |refused| return refused;
    const open = openRound(m, f.tokens.slice()[0], f.accept_cap) catch |e| return roundFailed(e);
    out.logits = .{ .ctx = open.logits.ctx };
    out.round.n = @intCast(open.rows);
    out.round.rows = @intCast(open.rows);
    @memcpy(out.round.tokens[0..open.rows], open.ids[0..open.rows]);
    m.open = open;
    return .ok;
}

fn nativeFinish(mp: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    var open = m.open orelse return Verdict.refuse(.failed, "sushi_ds41: no verify is open");
    m.open = null;
    if (accepted >= open.rows) return Verdict.refuse(.failed, "sushi_ds41: more drafts accepted than verified");
    // The logits were handed to the engine in `nativeVerify`.
    closeRound(m, &open, accepted, &out.round) catch |e| return roundFailed(e);
    return .ok;
}

/// A drafted and verified block whose verify is open.
const Open = struct {
    ids: [max_drafts + 1]u32,
    rows: usize,
    conf: [max_drafts]f32,
    block_len: usize,
    drafted: bool,
    /// The drafts are the context's own continuation (the copy arm), not a stage block: no confidences to learn from.
    copied: bool,
    draft_ms: f32,
    verify_start: std.Io.Timestamp,
    logits: mlx.mlx_array,
};

fn openRound(m: *Model, t1: u32, cap: u32) !Open {
    if (t1 >= m.cfg.vocab) return error.TokenPastVocabulary;
    const draft_start = now();
    var o: Open = .{ .ids = undefined, .rows = 1, .conf = @splat(0), .block_len = 1, .drafted = false, .copied = false, .draft_ms = 0, .verify_start = undefined, .logits = .{ .ctx = null } };
    o.ids[0] = t1;
    if (try copyDrafts(m, t1, cap, o.ids[1..])) |n| {
        o.rows = 1 + n;
        o.block_len = 1 + n;
        o.drafted = true;
        o.copied = true;
        o.draft_ms = nanosSince(draft_start);
        return verifyOpen(m, o);
    }
    const plain = m.policy.next() == .plain;
    o.drafted = !plain;
    if (!plain) {
        var block = try dsv.draftHeads(m.mdl, t1, .conf);
        defer block.deinit();
        o.block_len = @min(block.ids.len, max_drafts + 1);
        @memcpy(o.ids[0..o.block_len], block.ids[0..o.block_len]);
        @memcpy(o.conf[0 .. o.block_len - 1], block.conf[0 .. o.block_len - 1]);
    }
    o.draft_ms = nanosSince(draft_start);
    const wanted: usize = if (plain) 1 else m.policy.rows(o.conf[0 .. o.block_len - 1]);
    var rows = std.math.clamp(wanted, 1, @min(@as(usize, cap) + 1, max_verify_rows));
    o.rows = @min(rows, o.block_len);
    return verifyOpen(m, o);
}

/// The copy arm: writes the context's continuation after `t1` to `drafts` and returns its length, or null for a stage
/// round.
fn copyDrafts(m: *Model, t1: u32, cap: u32, drafts: []u32) !?usize {
    try m.lookup.sync(gpa, m.ids.items);
    const room = @min(@as(usize, cap), max_verify_rows - 1, drafts.len);
    const found = m.lookup.matchBest(t1, @intCast(room));
    const n = copy_lookup.copyLen(found, room, m.cfg.dspark.block);
    if (n == 0) return null;
    @memcpy(drafts[0..n], found.?.draft[0..n]);
    return n;
}

/// Verifies `o.rows` of the round's ids as one rewindable chunk.
fn verifyOpen(m: *Model, open: Open) !Open {
    var o = open;
    // The model refuses an image id without rows, and the trunk never picks one: a draft holding it ends the chunk.
    if (m.cfg.image_token_id != 0) if (std.mem.indexOfScalar(u32, o.ids[1..o.rows], m.cfg.image_token_id)) |at| {
        o.rows = 1 + at;
    };
    o.verify_start = now();
    m.engine.beginForward();
    o.logits = try dsv.verify(m.mdl, o.ids[0..o.rows]);
    m.engine.finishForward(o.rows);
    return o;
}

/// Keeps `t1` and `accepted` drafts, teaches the policy what the round bought, and fills the round's tokens and position.
fn closeRound(m: *Model, o: *const Open, accepted: usize, r: *abi.Round) !void {
    try dsv.finishVerify(m.mdl, accepted);
    const verify_ms = nanosSince(o.verify_start);
    try m.ids.appendSlice(gpa, o.ids[0 .. 1 + accepted]);
    if (o.copied) {
        // The stage policy's calibration and backoff are about stage blocks; a copy round only teaches it the price of
        // a wide verify.
        m.policy.cost.observeRound(@intCast(o.rows), verify_ms, 0);
        m.copy.rounds += 1;
        m.copy.drafted += o.rows - 1;
        m.copy.accepted += accepted;
    } else if (!o.drafted) m.policy.observePlain(verify_ms) else m.policy.observeRound(o.conf[0 .. o.block_len - 1], @intCast(o.rows), @intCast(accepted), o.draft_ms, verify_ms);
    r.n = @intCast(1 + accepted);
    r.accepted = @intCast(accepted);
    r.rows = @intCast(o.rows);
    r.position = (try dsv.ensureState(m.mdl)).n;
    @memcpy(r.tokens[0 .. 1 + accepted], o.ids[0 .. 1 + accepted]);
}

/// The trunk's token at each verify row `[rows, vocab]`, a suppressed id never chosen (the sampler cannot draw it
/// either); host memory, the caller frees.
fn argmaxRows(m: *Model, logits: mlx.mlx_array, suppress: ?mlx.mlx_array, rows: usize) ![]u32 {
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    var src = logits;
    if (suppress) |mask| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        try mlx.check(mlx.mlx_where(&masked, mask, neg_inf, logits, m.s));
        src = masked;
    }
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_argmax_axis(&idx, src, -1, false, m.s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, idx, .uint32, m.s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const data = mlx.mlx_array_data_uint32(flat) orelse return error.InvalidInput;
    if (mlx.mlx_array_size(flat) != rows) return error.InvalidInput;
    return gpa.dupe(u32, data[0..rows]);
}
