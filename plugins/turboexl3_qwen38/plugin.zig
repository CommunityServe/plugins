//! turboexl3_qwen38: the Qwen3.8-Flash-Next EXL3 packs K2 to K5 on M5-class GPUs. The model is the direct one in
//! src/direct, turboexl3 runs its EXL3 math, and the pack's MTP head is the native drafter (src/spec/drafter.zig).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");
/// The plugin's mlx-c layer, log, env and io_util, the quant ABI the copies and turboexl3's table share, and its
/// globals; turboexl3's `sdk` reads the layer, the ABI and the globals through this root (`mlx_host`).
pub const mlx = @import("src/mlx_serve/mlx/ffi.zig");
pub const log = @import("src/mlx_serve/sdk/util/log.zig");
pub const env = @import("src/mlx_serve/env.zig");
pub const io_util = @import("src/mlx_serve/sdk/util/io_util.zig");
pub const quant_abi = @import("src/mlx_serve/abi/quant.zig");
pub const shared = @import("src/mlx_serve/sdk/util/shared.zig");
const device = @import("src/mlx_serve/mlx/device.zig");
const q4 = @import("src/direct/serve.zig");
const mtp_head = @import("src/direct/mtp_head.zig");
const warmup = @import("src/direct/warmup.zig");
const ple = @import("src/direct/ple.zig");
const cost_profile = @import("src/mlx_serve/plugins/qwen4_exp/cost_profile.zig");
const ev_plan = @import("src/mlx_serve/engines/mlx/spec/mtp/ev_plan.zig");
const depth_bounds = @import("src/mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const KVQuantConfig = @import("src/mlx_serve/engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const Drafter = @import("src/spec/drafter.zig").Drafter;
const aux = @import("src/cache/aux.zig");
const host_flags = @import("src/host_flags.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "turboexl3_qwen38",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &q4.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .forward_settle = &forwardSettle,
    .release_slot = &releaseSlot,
    .activate_slot = &activateSlot,
    .cache = &.{ .layout = &q4.layout, .aux = &aux_hooks, .resume_prompt = &resumePrompt },
    .spec = &.{ .native_drafter = &native_drafter },
    .weights = &.{ .keep_key = &keepNone, .resident_bytes = &q4.residentBill },
};

const native_drafter: abi.NativeDrafter = .{ .step = &draftStep, .verify = &draftVerify, .finish = &draftFinish };

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// The loaded model: the built-in's, and its MTP head's rounds.
const Served = struct {
    m: *q4.Model,
    drafter: Drafter,
    /// The head drafts for this server (its `mtp` load setting; `--no-mtp` states it false): the forwards follow it.
    mtp: bool,
    /// The prefill whose PLE rows are read ahead of its gathers: its slot, and the prompt's length, where it ends.
    ple: ?struct { slot: u32, end: u64 } = null,
    /// Each slot's prompt as `begin_prompt` armed it, and the snapshot its prefill left for the prefix cache.
    prompts: std.AutoHashMapUnmanaged(u32, Prompt) = .empty,
};

/// A slot's prompt for the prefix cache: its ids, whether the server wants its snapshot, and the snapshot once the
/// prefill reached its boundary (`aux.capturesAt`), until `snapshot` takes it.
const Prompt = struct {
    ids: []u32,
    capture: bool,
    snap: ?*aux.Snap = null,

    fn deinit(p: *Prompt) void {
        if (p.snap) |sn| {
            sn.deinit();
            gpa.destroy(sn);
        }
        gpa.free(p.ids);
    }
};

fn dropPrompt(s: *Served, slot: u32) void {
    if (s.prompts.fetchRemove(slot)) |kv| {
        var p = kv.value;
        p.deinit();
    }
}

fn served(mp: *anyopaque) *Served {
    return @ptrCast(@alignCast(mp));
}

/// The plugin reads the pack's shards itself (`LoadCtx.weights` null): the engine loads none of them.
fn keepNone(p: *const abi.Parsed, key: abi.Str) callconv(.c) bool {
    _ = p;
    _ = key;
    return false;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const h = kit.host.Host.of(ctx.host) catch |e| {
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_qwen38: " ++ @errorName(x)),
        };
        return null;
    };
    mlx.bind(h);
    if (!device.naxPresent()) {
        log.err("[turboexl3_qwen38] turboexl3_qwen38 needs an M5-class GPU (NAX)\n", .{});
        out.* = Verdict.refuse(.unsupported, "turboexl3_qwen38 needs an M5-class GPU (NAX)");
        return null;
    }
    // The server's own flags this plugin follows (src/host_flags.zig), set before anything reads them.
    host_flags.apply() catch |e| {
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_qwen38: " ++ @errorName(x)),
        };
        return null;
    };
    const s = gpa.create(Served) catch {
        out.* = Verdict.refuse(.failed, "turboexl3_qwen38: out of memory");
        return null;
    };
    var own = ctx.*;
    own.weights = null;
    const mp = q4.load(&own, out) orelse {
        gpa.destroy(s);
        return null;
    };
    const m: *q4.Model = @ptrCast(@alignCast(mp));
    s.* = .{ .m = m, .drafter = Drafter.init(m), .mtp = false };
    // A pack without a head, or a server whose `mtp` setting is off: no rounds and no head history.
    if (s.drafter.has() and (q4.mtpSetting(ctx.settings_json.slice()) orelse true)) {
        s.mtp = true;
        armHead(m);
        s.drafter.warmTwoChunk() catch |e| log.warn("[turboexl3_qwen38] the two-chunk warm-up failed ({s}): the first boundary is timed cold\n", .{@errorName(e)});
    }
    return s;
}

/// The head's load-time work, as #1005's load does it when the head serves (`qwen4_build_draft_rerank`, then its eager
/// `qwen4_warmup_spec_verify`): the rerank head built here rather than inside the first round, and each verify width's
/// kernels, to the depth cap `--mtp-max-depth` resolves to, compiled before any round is timed. The slots' KV is dense
/// (`KVCache.init`).
fn armHead(m: *q4.Model) void {
    _ = mtp_head.qwen4BuildDraftRerank(&m.trunk);
    const kv = KVQuantConfig.dense;
    const cap = ev_plan.resolveMtpDepthCapForProfile(depth_bounds.active.max, cost_profile.qwen4G17CostProfileForKv(&m.trunk, kv));
    warmup.warmupSpecVerify(&m.trunk, cap, kv) catch |e| log.warn("[turboexl3_qwen38] the verify warm-up failed ({s}): each width compiles in its first round\n", .{@errorName(e)});
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const s = served(mp);
    pleEnd(s);
    var it = s.prompts.valueIterator();
    while (it.next()) |p| p.deinit();
    s.prompts.deinit(gpa);
    s.drafter.deinit();
    q4.unload(s.m);
    gpa.destroy(s);
}

/// The host's slot caches are its engine's types, which this dylib's copies must never read as theirs: each slot runs
/// on the state the model keeps for its id (`Caps.owns_decode_state`, `slot_state`). A forward's streams also extend
/// the slot's MTP history.
fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const s = served(mp);
    var own = f.*;
    own.cache_ops = null;
    own.caches = .{};
    const follow = s.mtp and f.mode != .verify and f.logits != .none and f.embeds.ctx == null;
    if (follow) own.want_hidden = true;
    captureAt(s, f);
    const v = q4.forward(s.m, &own, out);
    // The prefill's last forward, or a failed one, ends its read-ahead.
    if (s.ple) |p| if (!v.isOk() or (f.positions.len == 1 and f.slots.len == 1 and f.slots.slice()[0] == p.slot and @as(u64, f.positions.slice()[0]) + f.rows >= p.end)) pleEnd(s);
    if (!v.isOk() or !follow) return v;
    const streams: mlx.mlx_array = .{ .ctx = out.hidden.ctx };
    followForward(s, f, streams) catch |e| log.warn("[turboexl3_qwen38] MTP history not extended ({s}): the slot's rounds draft nothing\n", .{@errorName(e)});
    if (!f.want_hidden) {
        if (streams.ctx != null) _ = mlx.mlx_array_free(streams);
        out.hidden = .{};
    }
    return v;
}

/// The prefix cache's snapshot of a slot whose prefill reaches its final span with this forward: the state before it.
fn captureAt(s: *Served, f: *const abi.Forward) void {
    if (f.positions.len != 1 or f.slots.len != 1) return;
    const slot = f.slots.slice()[0];
    const p = s.prompts.getPtr(slot) orelse return;
    const pos: u64 = f.positions.slice()[0];
    if (!p.capture or p.snap != null or !aux.capturesAt(pos, f.rows, p.ids.len)) return;
    p.snap = aux.capture(s.m, &s.drafter, slot, pos, p.ids[0..@intCast(pos)]) catch |e| blk: {
        log.warn("[turboexl3_qwen38] slot {d}: no prefix-cache snapshot at {d} ({s})\n", .{ slot, pos, @errorName(e) });
        break :blk null;
    };
}

/// Each slot's rows of a forward into its MTP history: one slot's `rows`, or one row per slot of a batched forward.
fn followForward(s: *Served, f: *const abi.Forward, streams: mlx.mlx_array) !void {
    if (streams.ctx == null) return error.NoStreams;
    const slots = @max(f.positions.len, 1);
    const ids_dev: mlx.mlx_array = .{ .ctx = f.ids.ctx };
    if (slots == 1) {
        const id: u32 = if (f.slots.len > 0) f.slots.slice()[0] else 0;
        const pos: u64 = if (f.positions.len > 0) f.positions.slice()[0] else 0;
        return s.drafter.follow(id, pos, f.rows, f.tokens.slice(), ids_dev, streams, true);
    }
    const st = s.m.trunk.s;
    for (0..slots) |i| {
        const id: u32 = if (f.slots.len > i) f.slots.slice()[i] else @intCast(i);
        const pos: u64 = f.positions.slice()[i];
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try axis0(&row, streams, i, st);
        var dev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dev);
        if (ids_dev.ctx != null) try axis0(&dev, ids_dev, i, st);
        const toks: []const u32 = if (f.tokens.len == slots) f.tokens.slice()[i .. i + 1] else &.{};
        s.drafter.follow(id, pos, 1, toks, dev, row, true) catch |e| log.warn("[turboexl3_qwen38] slot {d}: MTP history not extended ({s})\n", .{ id, @errorName(e) });
    }
}

/// Index `i` of `x`'s first axis, kept.
fn axis0(out: *mlx.mlx_array, x: mlx.mlx_array, i: usize, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    @memcpy(stop[0..sh.len], sh);
    start[0] = @intCast(i);
    stop[0] = @intCast(i + 1);
    try mlx.check(mlx.mlx_slice(out, x, &start, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
}

fn forwardSettle(mp: *anyopaque, slots: abi.Span(u32), how: abi.Settle, tokens: *const abi.Span(u32)) callconv(.c) Verdict {
    return q4.forwardSettle(served(mp).m, slots, how, tokens);
}

fn releaseSlot(mp: *anyopaque, slot: u32) callconv(.c) void {
    const s = served(mp);
    if (s.ple) |p| if (p.slot == slot) pleEnd(s);
    dropPrompt(s, slot);
    s.drafter.release(slot);
    q4.releaseSlot(s.m, slot);
}

/// A round's own `step` activates the slot's head: nothing to do ahead of it.
fn activateSlot(mp: *anyopaque, slot: u32) callconv(.c) Verdict {
    _ = mp;
    _ = slot;
    return .ok;
}

/// The prefix cache's module path (`CacheHooks.aux`) hands each prompt in through `begin_prompt`; this one is the
/// server's other path, which it takes for a model with no aux binding: every prompt prefills cold.
fn resumePrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    _ = media_rows;
    const s = served(mp);
    if (auxEnabled(mp)) return 0;
    handPrompt(s, slot, prompt.slice(), 0);
    return 0;
}

/// A prompt's prefill from `at`: its ids pair the MTP history's chunk-final rows with the token after them, and the
/// prefill reads its PLE rows ahead.
fn handPrompt(s: *Served, slot: u32, prompt: []const u32, at: u64) void {
    if (s.mtp) s.drafter.prompt(slot, prompt, at) catch |e| log.warn("[turboexl3_qwen38] slot {d}: prompt not kept ({s})\n", .{ slot, @errorName(e) });
    plePrefetch(s, slot, prompt, at);
}

/// A prefill of `prompt` from `at` on `slot` reads its PLE rows ahead of each chunk's gather, the session #1005 opens
/// around its prefill (`qwen4_ple_prefetch_begin`, an EXL3 trunk pack's, on the host gather): the history before the
/// rows is the hash's padding from position 0, else the slot's own PLE history where a restore left it. A staged row is
/// taken only by the gather that planned it, so the bytes are the gathers' own.
fn plePrefetch(s: *Served, slot: u32, prompt: []const u32, at: u64) void {
    pleEnd(s);
    const t = &s.m.trunk;
    if (!t.formats.profile or prompt.len < at + 2) return;
    const st = t.qwen4 orelse return;
    if (st.gpu != null) return;
    for (t.moe_layers, 0..) |*lw, l| if (lw.ple != null) {
        var prev: [8]u32 = @splat(st.hash.eos);
        if (at > 0) {
            const own = ownOf(s.m, slot) orelse return;
            if (l >= own.ssm.len) return;
            if (own.ssm[l].ple_prev_valid) prev = own.ssm[l].ple_prev;
        }
        ple.prefetchBegin(&st.table, &st.hash, prev[0 .. st.hash.ngram_size - 1], prompt[@intCast(at)..], at);
        s.ple = .{ .slot = slot, .end = prompt.len };
        return;
    };
}

fn ownOf(m: *q4.Model, id: u32) ?*@typeInfo(@FieldType(@FieldType(q4.Model, "own"), "items")).pointer.child {
    for (m.own.items) |*o| if (o.id == id) return o;
    return null;
}

// ── The prefix cache's module snapshots (`CacheHooks.aux`, src/cache/aux.zig) ──

const aux_hooks: abi.AuxHooks = .{
    .reserve = &auxReserve,
    .commit = &auxCommit,
    .snapshot = &auxSnapshot,
    .restore_at = &auxRestoreAt,
    .slice = &auxSlice,
    .handoff = &auxHandoff,
    .free = &auxFree,
    .bytes = &auxBytes,
    .enabled = &auxEnabled,
    .describe = &auxDescribe,
    .prefix_key = &auxPrefixKey,
    .begin_prompt = &auxBeginPrompt,
    .import_plan = &auxImportPlan,
    .export_arrays = &auxExportArrays,
    .import_arrays = &auxImportArrays,
};

fn snapOf(handle: abi.Handle) ?*aux.Snap {
    return @ptrCast(@alignCast(handle orelse return null));
}

/// Every loaded instance keeps its slots' state resident and unpaired.
fn auxEnabled(_: *anyopaque) callconv(.c) bool {
    return true;
}

fn auxReserve(_: *anyopaque, _: u32, _: u64) callconv(.c) Verdict {
    return .ok;
}

/// A boundary the slot reached: its live position, or its snapshot's.
fn auxCommit(mp: *anyopaque, slot: u32, len: u64) callconv(.c) Verdict {
    const s = served(mp);
    if (s.prompts.get(slot)) |p| if (p.snap) |sn| if (sn.n == len) return .ok;
    if (ownOf(s.m, slot)) |o| if (o.pos == len) return .ok;
    return Verdict.refuse(.unsupported, "turboexl3_qwen38: commit requires a boundary the slot reached");
}

/// Hands the slot's snapshot over; null when its prefill never reached a boundary.
fn auxSnapshot(mp: *anyopaque, slot: u32, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    const p = served(mp).prompts.getPtr(slot) orelse return .ok;
    out.* = p.snap;
    p.snap = null;
    return .ok;
}

fn auxRestoreAt(mp: *anyopaque, slot: u32, handle: abi.Handle, pos: u64) callconv(.c) Verdict {
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "turboexl3_qwen38: no snapshot to restore");
    if (pos != sn.n) return Verdict.refuse(.unsupported, "turboexl3_qwen38: restore requires the snapshot's exact position");
    const s = served(mp);
    pleEnd(s);
    aux.restore(s.m, &s.drafter, slot, sn) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn auxSlice(_: *anyopaque, handle: abi.Handle, len: u64, out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "turboexl3_qwen38: no snapshot to slice");
    if (len != sn.n) return Verdict.refuse(.unsupported, "turboexl3_qwen38: a snapshot is not shortened (its recurrent state holds one position)");
    out.* = aux.clone(sn) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

/// Not reached by the server's module path; refused rather than half done.
fn auxHandoff(_: *anyopaque, _: u32, _: u32) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3_qwen38: slot handoff is not supported");
}

fn auxFree(_: *anyopaque, handle: abi.Handle) callconv(.c) void {
    const sn = snapOf(handle) orelse return;
    sn.deinit();
    gpa.destroy(sn);
}

fn auxBytes(_: *anyopaque, handle: abi.Handle) callconv(.c) u64 {
    const sn = snapOf(handle) orelse return 0;
    return sn.bytes();
}

fn auxDescribe(_: *anyopaque, handle: abi.Handle, out: *abi.AuxSnapshotInfo) callconv(.c) Verdict {
    const sn = snapOf(handle) orelse return Verdict.refuse(.unsupported, "turboexl3_qwen38: no snapshot to describe");
    if (out.size < @sizeOf(abi.AuxSnapshotInfo)) return Verdict.refuse(.unsupported, "turboexl3_qwen38: snapshot metadata is too short");
    out.* = .{ .n = sn.n, .ids = .of(sn.ids), .prefix_key = 0 };
    return .ok;
}

/// Text only (no images yet): every prefix has the one identity its ids give it.
fn auxPrefixKey(_: *anyopaque, _: abi.Span(u32), rows: abi.Array, out: *u64) callconv(.c) Verdict {
    out.* = 0;
    if (rows.ctx != null) return Verdict.refuse(.unsupported, "turboexl3_qwen38: no image rows");
    return .ok;
}

/// A prompt's prefill on `slot` from `restored_at` (0: cold, the slot starts over at its first forward; past 0: the slot
/// stands where `restore_at` left it), with its snapshot armed when the server keeps one.
fn auxBeginPrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), rows: abi.Array, restored_at: u64, capture: bool) callconv(.c) Verdict {
    if (rows.ctx != null) return Verdict.refuse(.unsupported, "turboexl3_qwen38: no image rows");
    const s = served(mp);
    if (restored_at != 0) {
        const own = ownOf(s.m, slot) orelse return Verdict.refuse(.failed, "turboexl3_qwen38: the restored slot has no state");
        if (own.pos != restored_at or restored_at >= prompt.len) return Verdict.refuse(.failed, "turboexl3_qwen38: the restored position differs from the prompt's boundary");
    }
    dropPrompt(s, slot);
    const ids = gpa.dupe(u32, prompt.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    s.prompts.put(gpa, slot, .{ .ids = ids, .capture = capture }) catch |e| {
        gpa.free(ids);
        return Verdict.named(.failed, @errorName(e));
    };
    handPrompt(s, slot, prompt.slice(), restored_at);
    return .ok;
}

fn auxImportPlan(_: *anyopaque, _: abi.Str, out: *u64) callconv(.c) Verdict {
    out.* = 0;
    return Verdict.refuse(.unsupported, "turboexl3_qwen38: the disk tier holds no module snapshots yet");
}

fn auxExportArrays(_: *anyopaque, _: abi.Handle, out: *abi.AuxExport) callconv(.c) Verdict {
    out.tensor_len = 0;
    out.names_len = 0;
    out.header_len = 0;
    return Verdict.refuse(.unsupported, "turboexl3_qwen38: the disk tier holds no module snapshots yet");
}

fn auxImportArrays(_: *anyopaque, _: abi.Span(u32), _: abi.Str, _: abi.Span(abi.AuxTensor), out: *abi.Handle) callconv(.c) Verdict {
    out.* = null;
    return Verdict.refuse(.unsupported, "turboexl3_qwen38: the disk tier holds no module snapshots yet");
}

/// Ends the open read-ahead, if any (#1005 ends it on every exit of the prefill).
fn pleEnd(s: *Served) void {
    if (s.ple == null) return;
    s.ple = null;
    ple.plePrefetchEnd(&s.m.trunk);
}

fn draftStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    served(mp).drafter.step(f, out) catch |e| return named(e);
    return .ok;
}

fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    served(mp).drafter.verify(f, out) catch |e| return named(e);
    return .ok;
}

fn draftFinish(mp: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) Verdict {
    served(mp).drafter.finish(accepted, out) catch |e| return named(e);
    return .ok;
}

fn named(e: anyerror) Verdict {
    log.err("[turboexl3_qwen38] draft round failed: {s}\n", .{@errorName(e)});
    return Verdict.named(.failed, @errorName(e));
}

/// A fully EXL3 pack has no tiny form: `plugin.json` says `"fixture": "none"`.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return error.NoFixture;
}

test {
    _ = config;
    // The MTP round's pure parts (`zig build mlx-free-test`): the verify's PLE lend and a request's [spec-stats] line.
    _ = @import("src/spec/drafter.zig");
    // The server's flags as the plugin reads them (`zig build mlx-free-test`).
    _ = @import("src/host_flags.zig");
    // #1005's trunk_quant block lookup, MLX-free (`zig build mlx-free-test`).
    _ = @import("src/mlx_serve/sdk/quant/block_test.zig");
    // #1005's EXL3 tests (`zig build gpu-test`, the server's MLX linked); mlx-free-test's filters leave them out.
    _ = @import("src/gpu_tests.zig");
    // The direct model's own: the trellis n-gram table, its warm and the resident bill (mlx-free-test), the table's
    // head bias (gpu-test).
    _ = @import("src/direct/ple.zig");
    _ = @import("src/direct/load.zig");
    _ = @import("src/direct/serve.zig");
}
