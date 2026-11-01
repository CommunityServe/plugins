//! turboexl3_ds41: DeepSeek-V4.1-Flash's EXL3 packs in the streaming repack, on M5-class GPUs. The model is mlx-serve's
//! own deepseek_v41 (a build-time copy, src/mlx_serve), its routed experts stream from the pack's bank through
//! mlx-serve's expert streamer (module `mlx_stream`), turboexl3 runs the EXL3 trunk, and the DSpark stages draft as the
//! model's native drafter, with the engine's copy rounds and verify policy (src/rounds.zig).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");
/// What the streamer's `sdk` and turboexl3's read through this root (`mlx_host`): the kit's mlx-c layer, the copies'
/// log, env, io_util and globals, the MTP acceptance modes and the quant ABI.
pub const mlx = kit.mlx;
pub const log = @import("src/mlx_serve/sdk/util/log.zig");
pub const env = @import("src/mlx_serve/env.zig");
pub const io_util = @import("src/mlx_serve/sdk/util/io_util.zig");
pub const shared = @import("src/mlx_serve/sdk/util/shared.zig");
pub const mtp_acceptance = @import("src/mlx_serve/sdk/spec/mtp_acceptance.zig");
pub const quant_abi = kit.abi.quant;
/// The routed experts' turboexl3 math the copied streamer runs when the model setting asks (src/routed_t3.zig).
pub const routed_t3 = @import("src/routed_t3.zig");
/// A wide prompt call's resident routed experts as one turboexl3 call, when `MLX_SERVE_DSV41_ROUTED_ONE` asks
/// (src/routed_one.zig, read by the copied streamer's source).
pub const routed_one = @import("src/routed_one.zig");
/// The DSpark draft's Markov chain in B + 1 launches (src/markov.zig, called by the copied model's draft;
/// `MLX_SERVE_DSV41_MARKOV`).
const markov = @import("src/markov.zig");
const ds = @import("src/mlx_serve/plugins/deepseek_v41/plugin.zig");
const dsv41 = @import("src/mlx_serve/plugins/deepseek_v41/deepseek_v41.zig");
const v4plugin = @import("src/mlx_serve/plugins/deepseek_v4/plugin.zig");
const rounds = @import("src/rounds.zig");
const stage_prof = @import("src/stage_prof.zig");
const bill = @import("src/mlx_serve/plugins/deepseek_v41/bill.zig");
const shims = @import("src/host_shims.zig");
const t3kit = @import("turboexl3_kit");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "turboexl3_ds41",
    .kind = .model,
    .claims = &config.model_types,
};

/// The built-in's table but for its name, claim and parse, its load (`load`), the drafter (the engine reaches no
/// outside plugin's module: the stages draft through `NativeDrafter`) and its weights (the plugin reads the shards).
/// Its forward and resume also keep the drafter's token history (`forward`, `resumePrompt`), and its meters report the
/// drafter's rounds.
pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = ds.table.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .release_slot = ds.table.release_slot,
    .activate_slot = ds.table.activate_slot,
    .cache = &cache_hooks,
    .media = ds.table.media,
    .spec = &.{ .native_drafter = &native_drafter },
    .weights = &.{ .keep_key = &keepNone, .resident_bytes = ds.table.weights.resident_bytes },
    .meters = &meters,
};

const cache_hooks: abi.CacheHooks = blk: {
    var c = ds.table.cache.*;
    c.resume_prompt = &resumePrompt;
    break :blk c;
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// The plugin reads the pack's shards itself (`LoadCtx.weights` null): the engine loads none of them.
fn keepNone(p: *const abi.Parsed, key: abi.Str) callconv(.c) bool {
    _ = p;
    _ = key;
    return false;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    _ = kit.host.Host.of(ctx.host) catch |e| {
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // The streamer's events, slot buffers and launches, and turboexl3's kernels, all through the server's table.
    if (!shims.install(ctx.host)) {
        out.* = Verdict.refuse(.unsupported, "turboexl3_ds41: the server's host table is older than this plugin's kit");
        return null;
    }
    t3kit.bind(ctx.host.?);
    // The streamed bank's slot rows fit within the host's ceiling, not only the device's working set.
    bill.host_ceiling = &shims.memoryCeiling;
    routed_t3.enabled = config.routedT3(env.span(.MLX_SERVE_DSV41_ROUTED));
    routed_one.mode = routed_one.modeOf(env.span(.MLX_SERVE_DSV41_ROUTED_ONE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ROUTED_ONE is none of 0, 1, resident, check: the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    markov.mode = markov.modeOf(env.span(.MLX_SERVE_DSV41_MARKOV)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_MARKOV is none of 1, 0, flat, check: the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    markov.tally = .{};
    // Before the model's load, which compiles its decode stretches only when the stage profiler is off.
    stage_prof.configure(env.span(.MLX_SERVE_DSV41_STAGE_PROF));
    var own = ctx.*;
    own.weights = null;
    const mp = ds.table.load(&own, out) orelse return null;
    open = null;
    const on: rounds.Switches = .{
        .copy = config.switchOn(env.span(.MLX_SERVE_DSV41_DSPARK_COPY)),
        .policy = config.switchOn(env.span(.MLX_SERVE_DSV41_DSPARK_POLICY)),
    };
    // The policy reads each draft's confidence, as the engine has its built-in model's stages do.
    model(mp).mdl.ds_want_conf = on.policy;
    // The fused Markov chain's kernels built now, not in the first draft.
    if (model(mp).mdl.dspark) |*w| {
        const mm = model(mp).mdl;
        markov.warm(mm.s, w.markov_embed, w.markov_head, w.conf, mm.act, mm.ds_block, mm.cfg.hidden_size, mm.ds_want_conf or std.math.isFinite(mm.ds_conf_thr)) catch |e| {
            log.err("[turboexl3_ds41] the fused Markov chain's kernels did not build ({s}): the load is refused (MLX_SERVE_DSV41_MARKOV=0 runs the op chain)\n", .{@errorName(e)});
            ds.table.unload(mp);
            out.* = Verdict.refuse(.failed, "turboexl3_ds41: the fused Markov chain's kernels did not build");
            return null;
        };
    }
    dropPending();
    if (drafter) |*d| d.deinit();
    drafter = rounds.Drafter.init(gpa, on);
    log.info("[turboexl3_ds41] DSpark copy rounds {s}, verify policy {s}\n", .{ if (on.copy) "on" else "off (MLX_SERVE_DSV41_DSPARK_COPY=0)", if (on.policy) "on" else "off (MLX_SERVE_DSV41_DSPARK_POLICY=0)" });
    if (routed_one.on()) log.info("[turboexl3_ds41] wide prompt calls' resident routed experts as one turboexl3 call (MLX_SERVE_DSV41_ROUTED_ONE: {s}); [routed-one] lines at each request's end\n", .{@tagName(routed_one.mode)});
    if (stage_prof.on) log.info("[turboexl3_ds41] stage profiler on (MLX_SERVE_DSV41_STAGE_PROF): every stage of a forward synced and timed, the decode stretches eager; [stage-prof] lines at each request's end\n", .{});
    log.info("[turboexl3_ds41] DSpark Markov chain: {s}\n", .{switch (markov.mode) {
        .on => "fused, B + 1 launches a draft (MLX_SERVE_DSV41_MARKOV=0: the op chain)",
        .flat => "fused, no zig-zag (MLX_SERVE_DSV41_MARKOV=flat)",
        .check => "fused and the op chain, compared every draft; the op chain's served (MLX_SERVE_DSV41_MARKOV=check); [markov] lines at each request's end",
        .off => "the op chain (MLX_SERVE_DSV41_MARKOV=0)",
    }});
    return mp;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    dropOpen(model(mp));
    dropPending();
    if (drafter) |*d| d.deinit();
    drafter = null;
    ds.table.unload(mp);
}

fn model(mp: *anyopaque) *ds.Model {
    return @ptrCast(@alignCast(mp));
}

// ── DSpark: the stages' rounds as the model's native drafter (the in-tree engine's `moduleRound`) ──

const native_drafter: abi.NativeDrafter = .{ .step = &draftStep, .verify = &draftVerify, .finish = &draftFinish };

/// The drafter's token history, verify ladder and request policy (src/rounds.zig): one per process (one streamed bank
/// serves one model, one request at a time).
var drafter: ?rounds.Drafter = null;

/// What a round verifies, for what it teaches once its tokens are kept.
const Round = struct {
    kind: rounds.Kind,
    drafts: usize,
    /// A stage round's confidence logits of its verified drafts.
    conf: [16]f32 = undefined,
    watch: io_util.Stopwatch,
};

/// A verify `draftVerify` left open for `draftFinish`: one per process.
const Open = struct { pending: dsv41.DsparkPending, round: Round };
var open: ?Open = null;

fn dropOpen(m: *ds.Model) void {
    var o = open orelse return;
    open = null;
    defer o.pending.deinit();
    if (m.mdl.dec_state) |*st| closeVerify(m, st, &o.pending);
}

/// Closes a verify keeping `t1` alone (the verify's anchors freed, its other rows rolled back), on the paths that
/// abandon a round: the engine refuses the request after them.
fn closeVerify(m: *ds.Model, st: *dsv41.State, pending: *dsv41.DsparkPending) void {
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, pending, 0, 0) catch {
        if (drafter) |*d| d.history.lose();
        return;
    };
    defer r.deinit(gpa);
    if (drafter) |*d| d.history.write(st.n - r.tokens.len, r.tokens);
}

/// The round's checks and its state: one committed token, the stages loaded, the state at the engine's position.
fn roundState(m: *ds.Model, f: *const abi.Forward, out: *const abi.ForwardOut) union(enum) { st: *dsv41.State, refused: Verdict } {
    if (out.size < @sizeOf(abi.ForwardOut)) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: the host's ForwardOut has no round") };
    if (m.mdl.dspark == null or m.mdl.n_mtp == 0) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: the DSpark stages are not loaded") };
    if (drafter == null) return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: the drafter is not loaded") };
    if (f.tokens.len != 1) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: a round starts from one committed token") };
    if (f.tokens.slice()[0] >= m.cfg.vocab_size) return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: token id past the vocabulary") };
    // A verify the engine never finished is discarded.
    dropOpen(m);
    const st = if (m.mdl.dec_state) |*s| s else return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: a round needs the prompt's state") };
    if (f.positions.len > 0 and st.n != f.positions.slice()[0]) return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: the position does not follow the model's state") };
    settlePending(&drafter.?);
    return .{ .st = st };
}

const Drafted = struct { d: dsv41.DsparkDraft, round: Round };

/// The round's drafts from `t1` (`rounds.Drafter.plan`), at most what a verify carries and the round may keep: the
/// context's continuation, none, or the stages' draft cut by the policy.
fn draft(m: *ds.Model, dr: *rounds.Drafter, st: *dsv41.State, t1: u32, accept_cap: u32) !Drafted {
    const watch = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const max = @min(dsv41.maxVerifyDrafts(m.mdl), @as(usize, accept_cap), abi.max_round_tokens - 1);
    const plan = dr.plan(st.n, t1, max, accept_cap, m.mdl.ds_block);
    if (plan == .stage) {
        var d = try dsv41.dsparkDraft(m.mdl, gpa, st, t1);
        errdefer d.deinit(gpa);
        d.len = dr.cut(d.confidence[0..@min(d.len, max)]);
        var round: Round = .{ .kind = .stage, .drafts = d.len, .watch = watch };
        @memcpy(round.conf[0..d.len], d.confidence[0..d.len]);
        return .{ .d = d, .round = round };
    }
    const copied: []const u32 = if (plan == .copy) plan.copy else &.{};
    const ids = try gpa.alloc(u32, copied.len + 1);
    ids[0] = t1;
    @memcpy(ids[1..], copied);
    return .{ .d = .{ .ids = ids, .len = copied.len, .logits = &.{}, .confidence = &.{} }, .round = .{ .kind = plan, .drafts = copied.len, .watch = watch } };
}

/// A round's kept tokens: its phases to `[dspark-prof]`, what it teaches to the drafter, its tokens to the history.
fn kept(m: *ds.Model, round: *const Round, r: *dsv41.DsparkRound, st: *const dsv41.State) void {
    r.phases.draft_ns = if (round.kind == .stage) m.mdl.ds_draft_ns else 0;
    dsv41.dsparkObserve(m.mdl, r.phases);
    const dr = if (drafter) |*d| d else return;
    const conf: []const f32 = if (round.kind == .stage) round.conf[0..round.drafts] else &.{};
    dr.learn(round.kind, round.drafts, conf, r.accepted, msOf(r.phases.verify_ns), msOf(round.watch.read()));
    dr.history.write(st.n - r.tokens.len, r.tokens);
}

fn msOf(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}

fn roundFailed(e: anyerror) Verdict {
    log.err("[turboexl3_ds41] draft round failed: {s}\n", .{@errorName(e)});
    if (drafter) |*d| d.history.lose();
    return Verdict.named(.failed, @errorName(e));
}

/// One greedy round: `t1` and its drafts (`draft`) through the trunk as one verify chunk, the longest prefix of drafts
/// the trunk's own argmax (under the suppress mask) agrees with stays, and the trunk's token after it is the next
/// round's `t1`.
fn draftStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    const st = switch (roundState(m, f, out)) {
        .st => |s| s,
        .refused => |v| return v,
    };
    const t1 = f.tokens.slice()[0];
    var p = draft(m, &drafter.?, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer p.d.deinit(gpa);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &p.d) catch |e| return roundFailed(e);
    defer pending.deinit();
    const suppress: ?mlx.mlx_array = if (f.suppress_mask.ctx != null) .{ .ctx = f.suppress_mask.ctx } else null;
    var best: [abi.max_round_tokens]u32 = undefined;
    argmaxRows(m, pending.vl_g, suppress, best[0 .. p.d.len + 1]) catch |e| {
        closeVerify(m, st, &pending);
        return roundFailed(e);
    };
    var accepted: usize = 0;
    while (accepted < p.d.len and best[accepted] == p.d.ids[accepted + 1]) accepted += 1;
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, &pending, accepted, best[accepted]) catch |e| return roundFailed(e);
    defer r.deinit(gpa);
    kept(m, &p.round, &r, st);
    fillRound(&out.round, r.tokens, @intCast(p.d.len + 1), st.n);
    out.round.accepted = r.accepted;
    out.round.next_token = r.next_token;
    return .ok;
}

/// The first half of a round the engine accepts under its own sampling: the verified rows' ids and logits.
fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    const st = switch (roundState(m, f, out)) {
        .st => |s| s,
        .refused => |v| return v,
    };
    const t1 = f.tokens.slice()[0];
    var p = draft(m, &drafter.?, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer p.d.deinit(gpa);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &p.d) catch |e| return roundFailed(e);
    const rows = p.d.len + 1;
    const logits = rowsF32(m, pending.vl_g, rows) catch |e| {
        closeVerify(m, st, &pending);
        pending.deinit();
        return roundFailed(e);
    };
    out.logits = .{ .ctx = logits.ctx };
    fillRound(&out.round, p.d.ids[0..rows], @intCast(rows), st.n);
    out.round.accepted = 0;
    open = .{ .pending = pending, .round = p.round };
    return .ok;
}

/// The second half: keeps `t1` and the first `accepted` drafts of the open verify.
fn draftFinish(mp: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    var o = open orelse return Verdict.refuse(.failed, "turboexl3_ds41: no verify is open");
    open = null;
    defer o.pending.deinit();
    const st = if (m.mdl.dec_state) |*s| s else return Verdict.refuse(.failed, "turboexl3_ds41: the open verify lost its state");
    if (accepted > o.pending.b) {
        closeVerify(m, st, &o.pending);
        return Verdict.refuse(.failed, "turboexl3_ds41: more drafts accepted than verified");
    }
    // The engine picked the token after the accepted drafts itself: the round's own next token is not read.
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, &o.pending, accepted, 0) catch |e| return roundFailed(e);
    defer r.deinit(gpa);
    kept(m, &o.round, &r, st);
    fillRound(&out.round, r.tokens, @intCast(o.pending.b + 1), st.n);
    out.round.accepted = r.accepted;
    return .ok;
}

// ── The drafter's token history: what the state holds, by position ──

/// A decode token the forward took on the GPU, read into the history once its forward has run.
const PendingToken = struct { at: usize, id: mlx.mlx_array };
var pending_tok: ?PendingToken = null;

fn settlePending(dr: *rounds.Drafter) void {
    const p = pending_tok orelse return;
    pending_tok = null;
    defer _ = mlx.mlx_array_free(p.id);
    const tok = v4plugin.hostToken(p.id) catch return dr.history.lose();
    dr.history.write(p.at, &.{tok});
}

fn dropPending() void {
    const p = pending_tok orelse return;
    pending_tok = null;
    _ = mlx.mlx_array_free(p.id);
}

/// The built-in's forward, the tokens it took written into the drafter's history at their positions: host ids as
/// given, a prompt's device ids read back as the forward reads them (before it runs, so the read waits on none of it),
/// a decode token the forward keeps on the GPU read once the forward has read it (at the next round or forward).
fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const dr = if (drafter) |*d| d else return ds.table.forward(mp, f, out);
    settlePending(dr);
    const at: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    const on_gpu = f.ids.ctx != null and f.rows == 1 and f.logits != .all_rows;
    const in: ?v4plugin.Ids = if (on_gpu) null else v4plugin.Ids.of(f) catch null;
    defer if (in) |x| x.deinit();
    const v = ds.table.forward(mp, f, out);
    if (!v.isOk() or f.slots.len > 1 or (!on_gpu and in == null)) {
        dr.history.lose();
        return v;
    }
    if (in) |x| {
        dr.history.write(at, x.host);
    } else {
        var id = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&id, .{ .ctx = f.ids.ctx });
        pending_tok = .{ .at = at, .id = id };
    }
    return v;
}

/// The built-in's resume: the drafter's request begins on the prompt prefix the state kept.
fn resumePrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    const n = ds.table.cache.resume_prompt.?(mp, slot, prompt, media_rows);
    dropPending();
    if (drafter) |*d| d.begin(prompt.slice()[0..@min(@as(usize, @intCast(n)), prompt.len)]);
    return n;
}

var meter_buf: [2048]u8 = undefined;

/// The request's DSpark rounds beside the engine's `[spec-stats]`, which counts every native round a stage round; the
/// stage profiler's lines (src/stage_prof.zig) when it is on; the expert stream's counters since the last request, and
/// the one-call path's (src/routed_one.zig) when it is on; the Markov chain's drafts (src/markov.zig) when it is on.
fn meters(mp: *anyopaque, slot: u32, out: *abi.Str) callconv(.c) void {
    _ = slot;
    stage_prof.requestEnd();
    const head = if (drafter) |*d| d.meters(&meter_buf) else meter_buf[0..0];
    const mid = streamMeters(model(mp), meter_buf[head.len..]);
    const tail = markov.meters(meter_buf[head.len + mid.len ..]);
    out.* = abi.Str.of(meter_buf[0 .. head.len + mid.len + tail.len]);
}

const Exl3Source = @import("src/mlx_serve/sdk/weights/streamed.zig").Exl3.Source;
const StreamStats = @TypeOf(@as(*Exl3Source, undefined).stats());
/// The stream's counters at the last request's end (they count for the process).
var stream_prev: StreamStats = .{};

/// `[stream-stats]`: the streamed bank's routes, resident hits and misses (unique experts a route), evictions, bytes
/// read and the host's read waits since the last request (the first request's include the load's); then `[routed-one]`
/// (`routed_one.meters`).
fn streamMeters(m: *ds.Model, buf: []u8) []const u8 {
    const r = m.mdl.streamed orelse return buf[0..0];
    if (r.math != .source) return buf[0..0];
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    const s = src.stats();
    const p = stream_prev;
    stream_prev = s;
    var w = std.Io.Writer.fixed(buf);
    w.print("[stream-stats] routes={d} hits={d} misses={d} evictions={d} read_gb={d:.2} read_wait_ms={d:.1} read_wall_ms={d:.1} spec_issued={d} spec_landed={d} claimed={d}\n", .{
        s.route_calls -| p.route_calls,
        s.expert_cache_hits -| p.expert_cache_hits,
        s.expert_cache_misses -| p.expert_cache_misses,
        s.expert_cache_evictions -| p.expert_cache_evictions,
        @as(f64, @floatFromInt(s.expert_bytes_read -| p.expert_bytes_read)) / 1e9,
        @as(f64, @floatFromInt(s.read_wait_ns -| p.read_wait_ns)) / 1e6,
        @as(f64, @floatFromInt(s.read_wall_ns -| p.read_wall_ns)) / 1e6,
        s.spec_issued -| p.spec_issued,
        s.spec_landed -| p.spec_landed,
        s.claimed -| p.claimed,
    }) catch {};
    const n = w.buffered().len;
    return buf[0 .. n + routed_one.meters(&src.one, buf[n..]).len];
}

fn fillRound(round: *abi.Round, tokens: []const u32, rows: u32, position: usize) void {
    round.n = @intCast(tokens.len);
    round.rows = rows;
    round.position = position;
    @memcpy(round.tokens[0..tokens.len], tokens);
}

/// The verify's logits `[rows, vocab]` in f32, a new array.
fn rowsF32(m: *ds.Model, vl: mlx.mlx_array, rows: usize) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, vl, &[_]c_int{ @intCast(rows), @intCast(m.mdl.vocab) }, 2, m.s));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, r, .float32, m.s));
    return o;
}

/// Each verify row's argmax, the suppressed ids never picked.
fn argmaxRows(m: *ds.Model, vl: mlx.mlx_array, suppress: ?mlx.mlx_array, out: []u32) !void {
    const s = m.s;
    const logits = try rowsF32(m, vl, out.len);
    defer _ = mlx.mlx_array_free(logits);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (suppress) |mask| {
        const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(ninf);
        try mlx.check(mlx.mlx_where(&masked, mask, ninf, logits, s));
    } else try mlx.check(mlx.mlx_array_set(&masked, logits));
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, masked, -1, false, s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const data = mlx.mlx_array_data_int32(am32) orelse return error.NoData;
    for (out, data[0..out.len]) |*o, v| o.* = @intCast(v);
}

/// A streamed EXL3 pack has no small form: `plugin.json` says `"fixture": "none"`.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return error.NoFixture;
}

test {
    _ = config;
}
