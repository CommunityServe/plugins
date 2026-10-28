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
const dsv = @import("src/sushi/dsv41_model.zig");
const dstream = @import("src/sushi/dsv41_stream.zig");
const source = @import("src/sushi/dsv41_source.zig");
const engram = @import("src/sushi/dsv41_engram.zig");
const stream = @import("src/sushi/expert_stream.zig");
const smodel = @import("src/sushi/model.zig");
const sc = @import("src/sushi/dsv41_config.zig");
const vis = @import("src/sushi/dsv41_vision.zig");
const rows_policy = @import("src/sushi/dspark_rows.zig");
const depth_bounds = @import("src/sushi/mtp_depth_bounds.zig");
const stb = @import("stb");

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
    /// The prompt being prefilled: its restore point is taken where the state stands at its last token.
    arm: ?Arm = null,
    /// A completed capture waiting for the engine to take it (`auxSnapshot`).
    snap: ?*Snap = null,
    /// How many of a drafted block's rows a round verifies, and when drafting is not worth it (per request).
    policy: rows_policy.Policy = .{},
    /// A round whose verify the engine is accepting (`NativeDrafter.verify` until `finish`).
    open: ?Open = null,
    prev_cache_limit: ?usize = null,
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
    return config.of(p).transientBytes(chunk, seq);
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
    const tower = if (config.of(p).vision != null) source.visionBytes(io, gpa, dir) catch |e| return Verdict.named(.failed, @errorName(e)) else 0;
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
    const request = c.transientBytes(host_chunk, context) +| (context +| generation_rows) *| c.stateBytesPerToken() +| c.stateFixedBytes(false);
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
    const logits = run(m, host.ids, want, rows) catch |e| {
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

fn reset(m: *Model) void {
    dsv.resetState(m.mdl);
    m.open = null;
    m.ids.clearRetainingCapacity();
    m.policy = rows_policy.Policy.init(depth_bounds.active, m.cfg.dspark.block);
}

/// A prompt as sushi runs it: all but its last token in chunks (`prefill_width`, a short tail merged into the last), each
/// told what follows it so the next chunk's Engram rows read while it runs, then the last token alone for the logits.
/// Logits `[rows, vocab]`.
fn run(m: *Model, ids: []const u32, want: dsv.Want, media: ?dsv.VisionRows) !mlx.mlx_array {
    var rows = media;
    const prefix = if (want == .last and ids.len > 1) ids.len - 1 else 0;
    var at: usize = 0;
    while (at < prefix) {
        var end = @min(at + sc.prefill_width, prefix);
        if (end < prefix and prefix - end < sc.tail_merge_rows) end = prefix;
        const next = if (end < prefix) @min(end + sc.prefill_width, prefix) else ids.len;
        _ = mlx.mlx_array_free(try forwardRows(m, ids[at..end], .last, ids[end..next], rows));
        if (rows) |*r| r.offset += std.mem.count(u32, ids[at..end], &.{m.cfg.image_token_id});
        at = end;
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
    m.engine.beginForward();
    const logits = try dsv.extendAhead(m.mdl, rows, want, dsv.SPAN, media, ahead);
    m.engine.finishForward(rows.len);
    try m.ids.appendSlice(gpa, rows);
    return logits;
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
}

// ── prefix cache ──

const Arm = struct { target: usize, capture: bool };

/// A restore point: the model's whole state at `n` (`dsv.captureState`) and the tokens it was built from.
const Snap = struct { slots: []dsv.Slot, n: usize, ids: []u32 };

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
    .serialize = &auxSerialize,
    .deserialize = &auxDeserialize,
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

/// Text only: the key of a prefix is its ids.
fn auxPrefixKey(_: *anyopaque, _: abi.Span(u32), _: abi.Array, out: *u64) callconv(.c) Verdict {
    out.* = 0;
    return .ok;
}

fn auxBeginPrompt(mp: *anyopaque, _: u32, prompt: abi.Span(u32), media: abi.Array, restored_at: u64, capture_on: bool) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (prompt.len == 0) return Verdict.refuse(.failed, "sushi_ds41: an empty prompt");
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
    // A restore point is keyed by token ids alone, which two different images share.
    m.arm = .{ .target = prompt.len - 1, .capture = capture_on and media.ctx == null };
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
    m.ids.clearRetainingCapacity();
    m.ids.appendSlice(gpa, sn.ids) catch return Verdict.refuse(.failed, "sushi_ds41: out of memory");
    dsv.restoreState(m.mdl, sn.slots, sn.n, sn.ids) catch |e| {
        m.ids.clearRetainingCapacity();
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
    out.* = .{ .n = sn.n, .ids = .of(sn.ids), .prefix_key = 0 };
    return .ok;
}

fn auxSerialize(_: *anyopaque, _: abi.Handle, _: [*]u8, _: usize, len: *usize) callconv(.c) Verdict {
    len.* = 0;
    return Verdict.refuse(.unsupported, "sushi_ds41: snapshot serialization is unavailable");
}

fn auxDeserialize(_: *anyopaque, _: abi.Str, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    return Verdict.refuse(.unsupported, "sushi_ds41: snapshot deserialization is unavailable");
}

// ── images ──

/// Packed RGB8 of a JPEG, PNG, GIF or BMP, transparency composited onto white.
fn decodeRgb(encoded: []const u8) ?struct { rgb: []u8, w: u32, h: u32 } {
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    const rgba = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &channels, 4) orelse return null;
    defer stb.stbi_image_free(rgba);
    const px: usize = @intCast(w * h);
    const rgb = gpa.alloc(u8, px * 3) catch return null;
    for (0..px) |i| {
        const al: u16 = rgba[i * 4 + 3];
        for (0..3) |c| rgb[i * 3 + c] = @intCast((al * @as(u16, rgba[i * 4 + c]) + (255 - al) * 255) / 255);
    }
    return .{ .rgb = rgb, .w = @intCast(w), .h = @intCast(h) };
}

fn mediaPrepare(p: *const abi.Parsed, kind: abi.MediaKind, bytes: abi.Str, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    _ = budget; // the release's processor bounds the image by its own token budget
    if (kind != .image) return Verdict.refuse(.unsupported, "sushi_ds41: images only");
    const v = config.of(p).vision orelse return Verdict.refuse(.unsupported, "sushi_ds41: this pack has no vision tower");
    const src = decodeRgb(bytes.slice()) orelse return Verdict.refuse(.failed, "sushi_ds41: the image does not decode (JPEG, PNG, GIF or BMP)");
    defer gpa.free(src.rgb);
    const px = vis.preprocess(gpa, src.rgb, src.h, src.w, vis.Processor.of(v)) catch return Verdict.refuse(.failed, "sushi_ds41: the image's dimensions have no grid");
    defer gpa.free(px.bytes);
    const feat: c_int = @intCast(3 * v.patch * v.patch);
    const n: c_int = @intCast(px.grid_h * px.grid_w);
    out.* = .{
        .kind = .image,
        .rows = @intCast(vis.spanRows(px.grid_h, px.grid_w, v.downsample)),
        .grid = .{ 1, px.grid_h, px.grid_w },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(px.bytes.ptr, &[_]c_int{ n, feat }, 2, .float32).ctx },
    };
    return .ok;
}

fn mediaTower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "sushi_ds41: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "sushi_ds41: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

/// Every position of an image's span carries the image token; the span's delimiters are the tower's learned rows.
fn placeholderRows(p: *const abi.Parsed, item: *const abi.MediaItem, out: [*]u32, cap: u32) callconv(.c) u32 {
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
    if (m.open != null) return Verdict.refuse(.failed, "sushi_ds41: a verify is still open");
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
    draft_ms: f32,
    verify_start: std.Io.Timestamp,
    logits: mlx.mlx_array,
};

fn openRound(m: *Model, t1: u32, cap: u32) !Open {
    if (t1 >= m.cfg.vocab) return error.TokenPastVocabulary;
    const draft_start = now();
    const plain = m.policy.next() == .plain;
    var o: Open = .{ .ids = undefined, .rows = 1, .conf = @splat(0), .block_len = 1, .drafted = !plain, .draft_ms = 0, .verify_start = undefined, .logits = .{ .ctx = null } };
    o.ids[0] = t1;
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
    rows = @min(rows, o.block_len);
    // The model refuses an image id without rows, and the trunk never picks one: a draft holding it ends the chunk.
    if (m.cfg.image_token_id != 0) if (std.mem.indexOfScalar(u32, o.ids[1..rows], m.cfg.image_token_id)) |at| {
        rows = 1 + at;
    };
    o.rows = rows;
    o.verify_start = now();
    m.engine.beginForward();
    o.logits = try dsv.verify(m.mdl, o.ids[0..rows]);
    m.engine.finishForward(rows);
    return o;
}

/// Keeps `t1` and `accepted` drafts, teaches the policy what the round bought, and fills the round's tokens and position.
fn closeRound(m: *Model, o: *const Open, accepted: usize, r: *abi.Round) !void {
    try dsv.finishVerify(m.mdl, accepted);
    const verify_ms = nanosSince(o.verify_start);
    try m.ids.appendSlice(gpa, o.ids[0 .. 1 + accepted]);
    if (!o.drafted) m.policy.observePlain(verify_ms) else m.policy.observeRound(o.conf[0 .. o.block_len - 1], @intCast(o.rows), @intCast(accepted), o.draft_ms, verify_ms);
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
