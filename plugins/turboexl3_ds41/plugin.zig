//! turboexl3_ds41: DeepSeek-V4.1-Flash's EXL3 packs in the streaming repack, on M5-class GPUs. The model is mlx-serve's
//! own deepseek_v41 (a build-time copy, src/mlx_serve), its routed experts stream from the pack's bank through
//! mlx-serve's expert streamer (module `mlx_stream`), turboexl3 runs the EXL3 trunk, and the DSpark stages draft as the
//! model's native drafter.

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
const ds = @import("src/mlx_serve/plugins/deepseek_v41/plugin.zig");
const dsv41 = @import("src/mlx_serve/plugins/deepseek_v41/deepseek_v41.zig");
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
pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = ds.table.bill,
    .load = &load,
    .unload = &unload,
    .forward = ds.table.forward,
    .release_slot = ds.table.release_slot,
    .activate_slot = ds.table.activate_slot,
    .cache = ds.table.cache,
    .media = ds.table.media,
    .spec = &.{ .native_drafter = &native_drafter },
    .weights = &.{ .keep_key = &keepNone, .resident_bytes = ds.table.weights.resident_bytes },
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
    var own = ctx.*;
    own.weights = null;
    const mp = ds.table.load(&own, out) orelse return null;
    open = null;
    return mp;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    dropOpen(model(mp));
    ds.table.unload(mp);
}

fn model(mp: *anyopaque) *ds.Model {
    return @ptrCast(@alignCast(mp));
}

// ── DSpark: the stages' rounds as the model's native drafter (the in-tree engine's `dsparkRoundWith`) ──

const native_drafter: abi.NativeDrafter = .{ .step = &draftStep, .verify = &draftVerify, .finish = &draftFinish };

/// A verify `draftVerify` left open for `draftFinish`: one per process (one streamed bank serves one model).
const Open = struct { pending: dsv41.DsparkPending };
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
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, pending, 0, 0) catch return;
    r.deinit(gpa);
}

/// The round's checks and its state: one committed token, the stages loaded, the state at the engine's position.
fn roundState(m: *ds.Model, f: *const abi.Forward, out: *const abi.ForwardOut) union(enum) { st: *dsv41.State, refused: Verdict } {
    if (out.size < @sizeOf(abi.ForwardOut)) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: the host's ForwardOut has no round") };
    if (m.mdl.dspark == null or m.mdl.n_mtp == 0) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: the DSpark stages are not loaded") };
    if (f.tokens.len != 1) return .{ .refused = Verdict.refuse(.unsupported, "turboexl3_ds41: a round starts from one committed token") };
    if (f.tokens.slice()[0] >= m.cfg.vocab_size) return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: token id past the vocabulary") };
    // A verify the engine never finished is discarded.
    dropOpen(m);
    const st = if (m.mdl.dec_state) |*s| s else return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: a round needs the prompt's state") };
    if (f.positions.len > 0 and st.n != f.positions.slice()[0]) return .{ .refused = Verdict.refuse(.failed, "turboexl3_ds41: the position does not follow the model's state") };
    return .{ .st = st };
}

/// The stages' draft from `t1`, cut to what a verify carries and the round may keep.
fn draft(m: *ds.Model, st: *dsv41.State, t1: u32, accept_cap: u32) !dsv41.DsparkDraft {
    var d = try dsv41.dsparkDraft(m.mdl, gpa, st, t1);
    d.len = @min(d.len, dsv41.maxVerifyDrafts(m.mdl), @as(usize, accept_cap), abi.max_round_tokens - 1);
    return d;
}

fn roundFailed(e: anyerror) Verdict {
    log.err("[turboexl3_ds41] draft round failed: {s}\n", .{@errorName(e)});
    return Verdict.named(.failed, @errorName(e));
}

/// One greedy round: the stages draft from `t1`, a verify chunk `[t1, drafts]` reads them through the trunk, the
/// longest prefix of drafts the trunk's own argmax (under the suppress mask) agrees with stays, and the trunk's token
/// after it is the next round's `t1`.
fn draftStep(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    const st = switch (roundState(m, f, out)) {
        .st => |s| s,
        .refused => |v| return v,
    };
    const t1 = f.tokens.slice()[0];
    var d = draft(m, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer d.deinit(gpa);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &d) catch |e| return roundFailed(e);
    defer pending.deinit();
    const suppress: ?mlx.mlx_array = if (f.suppress_mask.ctx != null) .{ .ctx = f.suppress_mask.ctx } else null;
    var best: [abi.max_round_tokens]u32 = undefined;
    argmaxRows(m, pending.vl_g, suppress, best[0 .. d.len + 1]) catch |e| {
        closeVerify(m, st, &pending);
        return roundFailed(e);
    };
    var accepted: usize = 0;
    while (accepted < d.len and best[accepted] == d.ids[accepted + 1]) accepted += 1;
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, &pending, accepted, best[accepted]) catch |e| return roundFailed(e);
    defer r.deinit(gpa);
    r.phases.draft_ns = m.mdl.ds_draft_ns;
    dsv41.dsparkObserve(m.mdl, r.phases);
    fillRound(&out.round, r.tokens, @intCast(d.len + 1), st.n);
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
    var d = draft(m, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer d.deinit(gpa);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &d) catch |e| return roundFailed(e);
    const rows = d.len + 1;
    const logits = rowsF32(m, pending.vl_g, rows) catch |e| {
        closeVerify(m, st, &pending);
        pending.deinit();
        return roundFailed(e);
    };
    out.logits = .{ .ctx = logits.ctx };
    fillRound(&out.round, d.ids[0..rows], @intCast(rows), st.n);
    out.round.accepted = 0;
    open = .{ .pending = pending };
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
    fillRound(&out.round, r.tokens, @intCast(o.pending.b + 1), st.n);
    out.round.accepted = r.accepted;
    return .ok;
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
