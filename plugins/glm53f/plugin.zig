//! glm53f: GLM-5.3-Flash's runner for the turboexl3 EXL3 streaming repacks, with the speed changes of mlx-serve #1196.
//! The KDA-hybrid trunk (loader, KDA layer, slots, table glue) is the kit's (`kit.bases.kda_hybrid`); this plugin holds
//! GLM's own forward (the hyper-connections, the sparse attention, the MTP head) and fills the trunk's hooks with its
//! speed changes: the NAX router (`Hooks.router`) and the KDA stages (`KdaConsts.stages`). The pack's EXL3 linears and
//! its routed bank are the turboexl3_glm53f quant plugin's, reached through the host (`Host.quant_claimant`).
//! copied_from: mlx-serve src/plugins/glm5_next/plugin.zig@385c3957 (the table, layout, bill and forward; its Sushi
//! bill and affine fused decode left out), the load's hooks this plugin's.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const sdk = kit.sdk;
const mlx = kit.mlx;
const kh = kit.bases.kda_hybrid;
const host = kh.host;
const trunk = kh.trunk;
const fixture = kh.fixture;
const fwd = @import("src/table_forward.zig");
const head = @import("src/head.zig");
const stages = @import("src/stages.zig");
const router_nax = @import("src/router_nax.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;
pub const version = "0.1.0";

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{ .resident_bytes = &residentBytes },
    .spec = &.{ .trunk_weight = &host.trunkWeight, .head = &head.hooks },
    .meters = &host.meters,
};

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;

/// KDA layers keep a recurrent state; every fourth layer caches one latent row `k` and the indexer's key and pool
/// gate `v` per token, unrotated.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isLinearLayer(@intCast(li)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .nope = true, .kv_heads = 1, .k_dim = c.dsa_kv_lora_rank, .v_dim = 2 * c.dsa_index_head_dim };
    return n;
}

/// glm5_next's bill: dense bf16 bytes per token (one latent row plus the indexer's key and gate per sparse layer) and
/// the checkpoint. The prefill bill stays the engine's `glm5PrefillMemoryNeeded`.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    out.* = .{
        .kv_per_token = @as(u64, c.attnLayerCount()) * (c.dsa_kv_lora_rank + 2 * c.dsa_index_head_dim) * 2,
        .ssm_checkpoint = c.ssmCheckpointBytes(4),
    };
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const mp = host.load(plugin.name, ctx, .{ .router = .{ .fill = &router_nax.fill } }, out) orelse {
        router_nax.release();
        return null;
    };
    stages.install(host.of(mp)) catch |e| {
        unload(mp);
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    return mp;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    host.unload(mp);
    router_nax.release();
}

/// Every pack as stored (a streaming pack's bank is its quant plugin's bill), plus the drafts' readout
/// (`draftReadoutBytes`), as glm5_next bills a streaming pack.
fn residentBytes(p: *const abi.Parsed, _: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    out.* = shards + draftReadoutBytes(config.of(p));
    return .ok;
}

/// The MTP drafts' readout a GLM streaming pack lends (`trunk.draftReadout`, at `trunk.readoutBits`). An upper bound:
/// the load builds it only when the quant plugin runs the head.
pub fn draftReadoutBytes(c: *const config.Config) u64 {
    if (!c.streamed_bank or !c.isGlm5()) return 0;
    return trunk.draftReadoutBytes(c.vocab_size, c.hidden_size, trunk.readoutBits(c));
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "glm53f: forward mode not declared");
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "glm53f: a forward without cache_ops");
    trunk.bankPhase(m, if (f.mode == .batched_rows) @min(f.caches.len, fwd.MOE_DECODE_BATCH_MAX) else f.rows) catch |e| return Verdict.named(.failed, @errorName(e));
    if (f.mode == .batched_rows) {
        if (f.rows != 1 or f.tokens.len != f.caches.len or f.positions.len != f.caches.len) return Verdict.refuse(.unsupported, "glm53f: batched rows take one host token per slot");
        if (f.capture_layers.len > 0 or f.media.ctx != null) return Verdict.refuse(.unsupported, "glm53f: no captures or media on batched rows");
        batched(m, ops, f, out) catch |e| {
            host.clear(out, 0);
            return Verdict.named(.failed, @errorName(e));
        };
        return host.settleBank(m, f, out);
    }
    const v = host.forwardOne(plugin.name, fwd, m, ops, f, out);
    return if (v.isOk()) host.settleBank(m, f, out) else v;
}

/// N slots' next tokens in as few trunk forwards as the MoE row limit allows, each slot's token a row of one window:
/// logits `[N, V]` and, when asked, the final-normed hidden `[N, H]`.
fn batched(m: *host.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.caches.len;
    const groups = (n + fwd.MOE_DECODE_BATCH_MAX - 1) / fwd.MOE_DECODE_BATCH_MAX;
    const parts = try std.heap.c_allocator.alloc([2]Array, groups);
    defer std.heap.c_allocator.free(parts);
    var made: usize = 0;
    defer for (parts[0..made]) |p| for (p) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var start: usize = 0;
    while (start < n) : (start += fwd.MOE_DECODE_BATCH_MAX) {
        const end = @min(n, start + fwd.MOE_DECODE_BATCH_MAX);
        const ids = try host.hostIds(f.tokens.slice()[start..end]);
        defer _ = mlx.mlx_array_free(ids);
        const st: fwd.Step = .{
            .slots = .{ .ops = ops, .caches = f.caches.slice()[start..end], .positions = f.positions.slice()[start..end] },
            .batched = true,
        };
        const hidden = try fwd.hidden(m, &st, ids);
        defer _ = mlx.mlx_array_free(hidden);
        const logits = try m.lmHead(hidden);
        defer _ = mlx.mlx_array_free(logits);
        parts[made] = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        made += 1;
        try rows2d(m, logits, &parts[made - 1][0]);
        try rows2d(m, hidden, &parts[made - 1][1]);
        // Streamed experts: the next group's fetches rewrite the slot rows this group's last MoE layer reads.
        if ((m.experts != null or m.bank != null) and end < n) try evalParts(&parts[made - 1]);
    }
    out.logits = .{ .ctx = (try joinRows(m, parts[0..made], 0)).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = (try joinRows(m, parts[0..made], 1)).ctx };
}

fn evalParts(p: *const [2]Array) !void {
    const vec = mlx.mlx_vector_array_new_data(p, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// `[1, R, W]` as `[R, W]`.
fn rows2d(m: *host.Model, x: Array, dst: *Array) !void {
    const sh = mlx.getShape(x);
    try mlx.check(mlx.mlx_reshape(dst, x, &[_]c_int{ sh[1], sh[2] }, 2, m.s));
}

/// The groups' rows `which` (0 logits, 1 hidden) stacked in slot order, a new reference.
fn joinRows(m: *host.Model, parts: []const [2]Array, which: usize) !Array {
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    if (parts.len == 1) {
        _ = mlx.mlx_array_set(&all, parts[0][which]);
        return all;
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p[which]);
    try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, m.s));
    return all;
}

/// The tiny pack the conformance suite runs: every arm of the trunk (dense and MoE MLPs, KDA, sparse attention).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.glmFiles(a, 7) };
}

// ── tests ──

test "glm53f bill: glm5_next's terms, and a GLM streaming pack bills its 8-bit draft readout, 643 MiB at its head" {
    var c: config.Config = .{ .model_type = "glm5_next", .vocab_size = 154880, .hidden_size = 4096, .streamed_bank = true };
    try std.testing.expectEqual(@as(u64, 674_037_760), draftReadoutBytes(&c));
    c.streamed_bank = false;
    try std.testing.expectEqual(@as(u64, 0), draftReadoutBytes(&c));
}

test {
    _ = config;
    _ = router_nax;
    _ = @import("src/options.zig");
    _ = @import("src/gpu_tests.zig");
}
