//! turboexl3_ling: Ling-3.1-flash's EXL3 packs (turboderp's, in the streaming repack) on M5-class GPUs. The model is
//! mlx-serve's own bailing_hybrid (a build-time copy, src/mlx_serve: its parse, forward and MLA over glm5_next's trunk
//! and KDA), every EXL3 projection of the trunk on turboexl3, the routed experts streamed from the pack's bank through
//! mlx-serve's expert streamer (module `mlx_stream`) with turboexl3's routed math (src/ling_math.zig).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");
const switches = config.switches;
/// What the streamer's `sdk` and turboexl3 read through this root (`mlx_host`): the kit's mlx-c layer, the copies' log,
/// env, io_util and globals, the MTP acceptance modes and the quant ABI.
pub const mlx = kit.mlx;
pub const log = @import("src/mlx_serve/sdk/util/log.zig");
pub const env = @import("src/mlx_serve/env.zig");
pub const io_util = @import("src/mlx_serve/sdk/util/io_util.zig");
pub const shared = @import("src/mlx_serve/sdk/util/shared.zig");
pub const mtp_acceptance = @import("src/mlx_serve/sdk/spec/mtp_acceptance.zig");
pub const quant_abi = kit.abi.quant;
/// Ling's routed math on turboexl3, run by the copied streamer (src/ling_math.zig).
pub const ling_math = @import("src/ling_math.zig");
/// The run-path line's marks (src/ling_ran.zig): the served paths mark here as they run, the copied streamer's files
/// too, and `forward` reports them.
pub const ling_ran = @import("src/ling_ran.zig");
/// RQ (src/ling_rq.zig): the bank's decode rows planned per request; the copied streamer's source takes the bank's row
/// arithmetic from it.
pub const ling_rq = @import("src/ling_rq.zig");
const ling_exl3 = @import("src/ling_exl3.zig");
const clamps = @import("src/clamps.zig");
const ling_bill = @import("src/bill.zig");
const ling_weights = @import("src/ling_weights.zig");
const bh = @import("src/mlx_serve/plugins/bailing_hybrid/plugin.zig");
const fwd = @import("src/mlx_serve/plugins/bailing_hybrid/table_forward.zig");
/// Ling's NextN (MTP) head (src/ling_mtp.zig).
const ling_mtp = @import("src/ling_mtp.zig");
/// C4: the layer-major pass in segments and a forward's hidden rows written once (src/ling_lm_pass.zig), at the plan's
/// sizes (src/ling_segments.zig).
const lm_pass = @import("src/ling_lm_pass.zig");
const ling_segments = @import("src/ling_segments.zig");
/// C6: a call that would rebuild the rope-once key past the bill's point, refused by name (src/ling_rope1_spike.zig).
const ling_spike = @import("src/ling_rope1_spike.zig");
/// EMBED_ROWS: the embedding table never resident, its rows read from the pack per forward (src/ling_embed.zig).
const ling_embed = @import("src/ling_embed.zig");
/// KV_SCHEME_DENSE_COPIES: a warning, once a load, when an MLA append's view shows a quantized KV cache
/// (src/ling_kv_scheme.zig; the copied mla.zig's `rope1Attn` makes the call).
const kv_scheme = @import("src/ling_kv_scheme.zig");
/// A prompt call's routed combine in one launch (src/ling_combine.zig), read by the copied streamer's source.
pub const ling_combine = @import("src/ling_combine.zig");
/// A MoE layer's router and read-ahead scores in two launches (src/ling_router.zig), whose scores the copied streamer
/// takes too.
pub const ling_router = @import("src/ling_router.zig");
/// A MoE layer's routed sum, shared expert add and tail in one launch (src/ling_tail.zig), read by the copied source.
pub const ling_tail = @import("src/ling_tail.zig");
/// A layer-major MoE layer's tail at each prompt span in one launch over its routed call's unjoined outputs
/// (PF_MOE_TAIL, src/ling_pf_tail.zig), read by the copied source too.
pub const ling_pf_tail = @import("src/ling_pf_tail.zig");
const host = @import("src/mlx_serve/plugins/glm5_next/host.zig");
const trunk = @import("src/mlx_serve/plugins/glm5_next/trunk.zig");
const routed = @import("src/mlx_serve/sdk/weights/routed.zig");
const system = @import("src/mlx_serve/sdk/util/system.zig");
const LingSource = @import("mlx_stream").ling_source;
const shims = @import("src/host_shims.zig");
const t3kit = @import("turboexl3_kit");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "turboexl3_ling",
    .kind = .model,
    .claims = &config.model_types,
    // LAYER_MAJOR's off switch, for the parity instrument only (src/ling_switches.zig `layer_major_option`).
    .options = &.{switches.layer_major_option},
};

/// The built-in's table but for its name, claim and parse, its bill (a prompt's working set beside the bank), its load
/// (`load`: the trunk from the plugin's own read of the shards, then the bank), its forward (the bank's phase and settle
/// around the built-in's) and its weights (the engine loads none).
pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &cache_hooks,
    .weights = &.{ .keep_key = &keepKey },
    .spec = &spec_hooks,
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// The plugin reads the pack's shards itself (src/ling_weights.zig): the engine loads none of them.
fn keepKey(p: *const abi.Parsed, key: abi.Str) callconv(.c) bool {
    _ = p;
    _ = key;
    return false;
}

/// The built-in's bill, and what a prompt pass keeps free of the bank (`prefill_bytes`: the engine leaves the bank the
/// GPU ceiling less it, `LoadCtx.stream_room`; src/bill.zig).
fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    bh.table.bill(p, shape, out);
    out.kv_per_token = kvPerToken(p);
    out.prefill_bytes = &prefillBytes;
}

fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    // `chunk` is the width the host prices the pass at (mlx-serve's stream room: its WIDE_PREFILL_CHUNK, 16,384 rows);
    // no term of the bill depends on it. The host prices the room before `load`, at a fixed shape (16,384 + 1,024), when
    // the plugin has no served context: this answers at the bill's default context, and `openBank` moves the room to the
    // served context's reserve (C1, src/bill.zig `roomAt`).
    _ = .{ seq, chunk, kv_bits, warm };
    return prefillAt(p, ling_bill.default_context);
}

/// C1: a prompt pass's reserve beside the bank at a served context of `ctx` tokens: the KV of its longest whole prompt
/// (src/ling_rq.zig `promptCapacityMax`), prompt_work and free RAM (src/bill.zig `prompt`), the layer-major pass's
/// working set past prompt_work (C2, `layerMajorBytes`) and the prompt-only term (`promptOnlyBytes`).
fn prefillAt(p: *const abi.Parsed, ctx: u64) u64 {
    return prefillWith(p, ctx, layerMajorBytes(p, ctx));
}

/// `prefillAt` with the layer-major term `lm` already made (`layerMajorBytes` at `ctx`): C2's term reads every whole
/// prompt up to the context, so `openBank` makes it once.
fn prefillWith(p: *const abi.Parsed, ctx: u64, lm: u64) u64 {
    return ling_bill.prompt(kvPerToken(p), ling_rq.promptCapacityMax(ctx)) + lm + promptOnlyBytes(p);
}

/// The prompt pass's own term past the layer-major pass's, which `openBank` gives back to the decode rows: with the
/// layer-major pass, the server's own memory past MLX's counters (src/bill.zig `mem_refit_server`; C2 folds MEM_REFIT's
/// pass part into the pass's fitted working set, `layerMajorBytes`); without it, none. No part of it is the host's
/// pricing `chunk`, so both sides of the bill take one value. The rope-once cache's fallback (a
/// prompt call of 16 rows or fewer rebuilds the expanded key) adds no term: its spike sits inside the pass's idle working
/// set (src/bill.zig `ropeOnceSpike`'s test). The plugin declares the last-row logits, so the engine asks every prompt
/// row's logits only for prompt log-probabilities. The bill holds no term for those: `forward` refuses a prompt whose
/// engine step over them would not fit the plan's room (src/bill.zig `echoRefusalAt`).
fn promptOnlyBytes(p: *const abi.Parsed) u64 {
    _ = p;
    return if (switches.layerMajor()) ling_bill.mem_refit_server else 0;
}

/// C2: the layer-major pass's working set past `prompt_work` at the served context, `ctx` tokens: the fitted W at its
/// most over every whole prompt up to the context (src/bill.zig `passWorkBill`: the widest routed call, the checkpoints
/// its longest prompt names, the KV growth's copy, D2's group and the fitted part); 0 without the layer-major pass.
fn layerMajorBytes(p: *const abi.Parsed, ctx: u64) u64 {
    if (!switches.layerMajor()) return 0;
    return ling_bill.passWorkBill(passOf(p), ctx) -| ling_bill.prompt_work;
}

/// C2: the pass model's geometry (src/bill.zig `Pass`) for this pack: the routed pairs a row and the hidden width, one
/// checkpoint's bytes, one rope-once MLA layer's KV a token, D2's group a token (kv_b's rows and their joined copy) and
/// the routed calls' rows. With the layer-major pass, C4's segments as `spanMajorOr` runs them (src/ling_segments.zig:
/// whole routed calls, at most `segment_rows` rows a segment unless fewer than `segment_tail_rows` rows would remain
/// after it); without it, one pass (no term reads the model then).
fn passOf(p: *const abi.Parsed) ling_bill.Pass {
    const c = config.of(p);
    const lm = switches.layerMajor();
    return .{
        .top_k = c.num_experts_per_tok,
        .hidden = c.hidden_size,
        .checkpoint = c.ssmCheckpointBytes(2),
        .layer_kv = ling_bill.ropeOnceLayer(c.num_attention_heads, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim, c.mla_v_head_dim),
        .group = @as(u64, c.num_attention_heads) * (c.mla_qk_nope_head_dim + c.mla_v_head_dim) * 2 * 2,
        .call_rows = layer_major_rows,
        .tail_rows = layer_major_tail_rows,
        .segment_rows = if (lm) ling_segments.segment_rows else 0,
        .segment_tail_rows = if (lm) ling_segments.segment_tail_rows else 0,
    };
}

/// `[mem]`: MLX's active and cached bytes and the box's available memory at a load step (the bank's in-use reading).
fn memLine(comptime at: []const u8) void {
    var act: usize = 0;
    var cch: usize = 0;
    _ = mlx.mlx_get_active_memory(&act);
    _ = mlx.mlx_get_cache_memory(&cch);
    log.info("[turboexl3_ling] [mem] " ++ at ++ ": MLX active {d} MiB, cached {d} MiB; the box's in use {d} MiB\n", .{ act >> 20, cch >> 20, (system.getTotalMemBytes() -| system.getAvailableMemBytes()) >> 20 });
}

/// `[mem]` as a prompt pass ends (its phase turns to decode, before the turn clears MLX's cache): MLX's high-water mark
/// since the process began (active bytes, the cache apart: at the first turn, the prompt pass's peak), its active and
/// cached bytes now, and the bank's prompt rows. The bill's terms are fitted to it (src/bill.zig). The mark is never
/// reset here: the host's /props reads it.
fn promptEndMemLine(b: *const LingSource.Source) void {
    var pk: usize = 0;
    var act: usize = 0;
    var cch: usize = 0;
    _ = mlx.mlx_get_peak_memory(&pk);
    _ = mlx.mlx_get_active_memory(&act);
    _ = mlx.mlx_get_cache_memory(&cch);
    const rows: u64 = @as(u64, b.bill.rows.prompt) * b.bill.row_bytes;
    log.info("[turboexl3_ling] [mem] a prompt pass ends: MLX peak {d} MiB since the start (active bytes), now active {d} MiB and cached {d} MiB; the bank's prompt rows {d} MiB; the box's in use {d} MiB\n", .{ pk >> 20, act >> 20, cch >> 20, rows >> 20, (system.getTotalMemBytes() -| system.getAvailableMemBytes()) >> 20 });
}

/// The NextN head (src/ling_mtp.zig) and the trunk weights its drafts read (the embedding table, and the EXL3 head as
/// MLX affine 4-bit, the draft readout). The table lent is the shards' unevaluated tensor (EMBED_ROWS,
/// src/ling_embed.zig).
const spec_hooks: abi.SpecHooks = .{ .head = &ling_mtp.hooks, .trunk_weight = &ling_mtp.trunkWeight };

/// The built-in's layout, but each MLA layer on the rope-once cache (src/ling_mla_rope1.zig) holds one head: the roped
/// key (K, rope a token) and kv_b's rows (V, each head's nope keys and values). Every other hook is the built-in's.
const cache_hooks: abi.CacheHooks = blk: {
    var h = bh.table.cache.*;
    h.layout = &layout;
    break :blk h;
};
fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const n = bh.table.cache.layout.?(p, span, out, cap);
    const c = config.of(p);
    for (out[0..n]) |*l| if (l.kind == .full) {
        l.kv_heads = 1;
        l.k_dim = c.mla_qk_rope_head_dim;
        l.v_dim = c.num_attention_heads * (c.mla_qk_nope_head_dim + c.mla_v_head_dim);
    };
    return n;
}

/// KV bytes a token: each trunk MLA layer on the rope-once cache, kv_b's rows and the roped key once (src/bill.zig
/// `ropeOnceLayer`), and the NextN head's one MLA layer on its latent cache, its normed latent and roped shared key
/// (src/ling_mtp_latent.zig; src/bill.zig `headLatentKv`).
fn kvPerToken(p: *const abi.Parsed) u64 {
    const c = config.of(p);
    const per_layer = ling_bill.ropeOnceLayer(c.num_attention_heads, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim, c.mla_v_head_dim);
    return ling_bill.headLatentKv(c.attnLayerCount(), per_layer, c.mla_kv_lora_rank, c.mla_qk_rope_head_dim);
}

/// `kvPerToken`'s two caches apart, for RQ's plans (src/ling_rq.zig `Kv`): a trunk MLA layer's KV a token, the trunk's
/// MLA layers, and the NextN head's latent KV a token.
fn rqKv(p: *const abi.Parsed) ling_rq.Kv {
    const c = config.of(p);
    return .{
        .layer = ling_bill.ropeOnceLayer(c.num_attention_heads, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim, c.mla_v_head_dim),
        .layers = c.attnLayerCount(),
        .head = ling_bill.latentKv(1, c.mla_kv_lora_rank, c.mla_qk_rope_head_dim),
    };
}

/// KV_SCHEME_DENSE_COPIES' price (src/ling_kv_scheme.zig): one rope-once MLA layer's K and V as dense bf16 rows at the
/// served context, `ctx` tokens (C1, src/bill.zig `contextOf`; `ropeOnceLayer`), the copies a quantized cache's view
/// hands each call.
fn kvSchemePlan(p: *const abi.Parsed, ctx: u64) kv_scheme.Plan {
    const c = config.of(p);
    const per_layer = ling_bill.ropeOnceLayer(c.num_attention_heads, c.mla_qk_nope_head_dim, c.mla_qk_rope_head_dim, c.mla_v_head_dim);
    return .{ .tokens = ctx, .layer_bytes = ctx * per_layer };
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    _ = kit.host.Host.of(ctx.host) catch |e| {
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ling: " ++ @errorName(x)),
        };
        return null;
    };
    // The streamer's events, slot buffers and launches, and turboexl3's kernels, all through the server's table.
    if (!shims.install(ctx.host)) {
        out.* = Verdict.refuse(.unsupported, "turboexl3_ling: the server's host table is older than this plugin's kit");
        return null;
    }
    t3kit.bind(ctx.host.?);
    no_spans_said = false;
    // A load starts with no request and no plan (a reload in one process forgets the last model's).
    request_boundary = .{};
    echo_plan = null;
    rq = null;
    ling_mtp.rq_append = null;
    // C1: the served context (`--ctx-size`, which the load's settings name; src/bill.zig `contextOf`), which the bill holds.
    const served = ling_bill.contextOf(ctx.settings_json.slice()) catch {
        log.err("[turboexl3_ling] the load's settings name no context the bill can hold: {s}\n", .{ctx.settings_json.slice()});
        out.* = Verdict.refuse(.unsupported, "turboexl3_ling: LingCtxSizeUnreadable");
        return null;
    };
    // KV_SCHEME_DENSE_COPIES (src/ling_kv_scheme.zig): each load warns afresh, at its own model's price.
    kv_scheme.warned = false;
    kv_scheme.load_plan = kvSchemePlan(ctx.parsed, served);
    // turboexl3's configuration before any of its calls (the NextN head's dense readout runs before the bank opens).
    ling_math.configure() catch |e| {
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (!t3kit.nax.available()) {
        out.* = Verdict.refuse(.unsupported, "turboexl3_ling: an M5-class GPU (NAX) is needed (turboexl3's prefill kernels run on its matrix units)");
        return null;
    }
    // The trunk from the shards as stored, the NextN layer's with it (its head always loads), the engine's map unused.
    const io = std.Io.Threaded.global_single_threaded.io();
    const s: mlx.mlx_stream = if (ctx.stream.ctx) |p| .{ .ctx = p } else mlx.gpuStream();
    // No tensor's name starts with a NUL, so every tensor is read.
    const w = ling_weights.Weights.open(gpa, io, ctx.model_dir.slice(), "\x00", s) catch |e| {
        log.err("[turboexl3_ling] the pack's shards do not open: {s}\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    var w_open = true;
    defer if (w_open) w.deinit();
    log.info("[turboexl3_ling] {d} trunk tensors read as stored\n", .{w.count()});
    var own = ctx.*;
    own.weights = &w.src;
    // EMBED_ROWS (src/ling_embed.zig): the table's shard opens before the trunk loads, so the trunk keeps the table
    // unevaluated (ling-trunk.patch `evalTrunk`) and its forwards read the rows from the pack. A load that fails past
    // here closes it again.
    ling_embed.close();
    openEmbedRows(ctx, &w.src) catch |e| {
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    var loaded = false;
    defer if (!loaded) ling_embed.close();
    const mp = host.load(plugin.name, &own, out) orelse return null;
    const m = host.of(mp);
    // Only the pack's own EXL3 head (the head this plugin serves) is taken: a head made from the table would read the
    // table the trunk keeps unevaluated.
    if (!m.head.exl3) {
        log.err("[turboexl3_ling] EMBED_ROWS needs the pack's own EXL3 head (lm_head.trellis), and this pack's head is not one\n", .{});
        host.unload(mp);
        out.* = Verdict.refuse(.unsupported, "turboexl3_ling: EMBED_ROWS needs the pack's own EXL3 head");
        return null;
    }
    memLine("before the MTP head");
    ling_mtp.load(m, &w.src) catch |e| {
        log.err("[turboexl3_ling] the MTP head does not load: {s}\n", .{@errorName(e)});
        host.unload(mp);
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    // The shards' arrays the model did not keep (the NextN experts' per-expert copies, the codebook words) and MLX's
    // cached buffers go before the bank reads what is in use.
    memLine("after the MTP head");
    w.deinit();
    w_open = false;
    _ = mlx.mlx_clear_cache();
    memLine("after the shards' arrays went");
    openBank(m, ctx.parsed, ctx.model_dir.slice(), ctx.stream_room, served) catch |e| {
        log.err("[turboexl3_ling] the streamed bank does not open: {s}\n", .{@errorName(e)});
        host.unload(mp);
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    // EMBED_ROWS' load line, once the bank has taken the table's bytes (its `[bank-bill]` lines above).
    if (ling_embed.servedTable()) |t| log.info("[turboexl3_ling] EMBED_ROWS: the embedding table ({d} x {d} bf16, {d} MiB) is not resident, so the bank's slot rows took its bytes; each forward reads its rows from {s} at byte {d}, each distinct row once, and a decode-phase forward first waits for its token ids on the GPU\n", .{ t.rows, t.cols, ling_bill.embedRowsFreed(t.rows, t.cols) >> 20, ling_embed.servedShard(), t.offset });
    loaded = true;
    return mp;
}

/// The model's unload, then EMBED_ROWS' shard closed.
fn unload(mp: *anyopaque) callconv(.c) void {
    host.unload(mp);
    ling_embed.close();
}

/// EMBED_ROWS: the trunk's embedding table (`<prefix>.word_embeddings.weight`, the prefix the copied trunk.zig
/// `probePrefix` finds in `src`) opened in its shard for row reads, at the config's vocabulary and width. Refused by
/// name, with a log line: a pack without the table, a tied head, or a table that is not BF16 [vocab, hidden] in its shard.
fn openEmbedRows(ctx: *const abi.LoadCtx, src: *const quant_abi.Source) !void {
    const c = config.of(ctx.parsed);
    if (c.tie_word_embeddings) {
        log.err("[turboexl3_ling] EMBED_ROWS needs an untied head: this pack's head is its embedding table\n", .{});
        return error.LingEmbedRowsTiedHead;
    }
    var kb: [160]u8 = undefined;
    const key = for ([_][]const u8{ c.weight_prefix, "language_model.model", "model.language_model", "model" }) |p| {
        const name = std.fmt.bufPrint(&kb, "{s}.word_embeddings.weight", .{p}) catch continue;
        if (src.tensor(name).ctx != null) break name;
    } else {
        log.err("[turboexl3_ling] EMBED_ROWS: the pack has no word_embeddings table\n", .{});
        return error.LingEmbedRowsMissing;
    };
    ling_embed.open(ctx.model_dir.slice(), key, c.vocab_size, c.hidden_size) catch |e| {
        log.err("[turboexl3_ling] EMBED_ROWS: the pack's {s} does not open for row reads: {s}\n", .{ key, @errorName(e) });
        return e;
    };
}

/// The pack's bank bound to every routed layer, at the slot rows the box holds beside the loaded trunk and one prompt at
/// the served context, `ctx` tokens (C1: the host's room, `host_room`, is priced at the bill's default context).
fn openBank(m: *trunk.Model, p: *const abi.Parsed, dir: []const u8, host_room: u64, ctx: u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const c = m.c;
    const first: u32 = c.first_k_dense_replace;
    if (first >= m.layers.len) return error.LingNoRoutedLayers;
    const n_moe: u32 = @intCast(m.layers.len - first);
    const limits = try gpa.alloc(c_int, n_moe);
    defer gpa.free(limits);
    for (limits, first..) |*o, li| o.* = @intCast(clamps.routed(@intCast(li)));
    const gates = try gpa.alloc(LingSource.Gate, n_moe);
    defer gpa.free(gates);
    for (gates, m.layers[first..]) |*g, *l| {
        if (l.mlp != .moe) return error.LingDenseLayerPastFirst;
        g.* = .{ .w = l.router_raw, .bias = l.mlp.moe.route.sigmoid_bias.expert_bias };
    }
    const geo: LingSource.Geometry = .{ .hidden_size = c.hidden_size, .moe_intermediate_size = c.moe_intermediate_size, .n_experts_per_tok = c.num_experts_per_tok, .n_routed_experts = c.num_experts, .n_layers = n_moe, .limits = limits, .n_group = c.moe_n_group, .topk_group = c.moe_topk_group };
    // PF_DFIN's whole-call rule reads the model's routing (src/ling_math.zig `routing`), set before the bank's first
    // routed call (its warm-up)
    ling_math.routing = .{ .topk = geo.n_experts_per_tok, .experts = geo.n_routed_experts };
    _ = mlx.mlx_synchronize(m.s);
    // What the plugin holds: MLX's live and cached buffers (the trunk, the NextN head, the transients not yet freed;
    // not the embedding table, which stays unevaluated (EMBED_ROWS), so the slot rows take its bytes). The box's own
    // reading (total less available) moved with other processes and the file cache (the same load read 10.4 and
    // 24.3 GiB), and the bank with it; it is logged beside, not used. The host's wired margin keeps the rest of
    // the box.
    var act: usize = 0;
    var cch: usize = 0;
    _ = mlx.mlx_get_active_memory(&act);
    _ = mlx.mlx_get_cache_memory(&cch);
    const in_use: u64 = @as(u64, act) + cch;
    log.info("[turboexl3_ling] [bank-bill] the plugin holds {d} MiB (MLX live {d}, cached {d}); the box reads {d} MiB in use\n", .{ in_use >> 20, act >> 20, cch >> 20, (system.getTotalMemBytes() -| system.getAvailableMemBytes()) >> 20 });
    // C1: the served context's reserve in place of the default context's, which the host priced its room at
    // (src/bill.zig `roomAt`): the KV of its longest whole prompt (src/ling_rq.zig `promptCapacityMax`) and the
    // layer-major pass's working set at the context (C2, `layerMajorBytes`).
    const kv_tokens = ling_rq.promptCapacityMax(ctx);
    const priced = prefillAt(p, ling_bill.default_context);
    // A context whose KV alone takes the room is refused before C2's term, which reads every whole prompt up to it.
    const kv_room = ling_bill.roomAt(host_room, priced, ling_bill.prompt(kvPerToken(p), kv_tokens));
    const lm_back: u64 = if (kv_room != null) layerMajorBytes(p, ctx) else 0;
    const room = (if (kv_room != null) ling_bill.roomAt(host_room, priced, prefillWith(p, ctx, lm_back)) else null) orelse {
        log.err("[turboexl3_ling] a {d}-token context's reserve (KV for {d} tokens) leaves the bank no room: the host's room is {d} MiB at the default {d}-token reserve\n", .{ ctx, kv_tokens, host_room >> 20, ling_bill.default_context });
        return error.LingContextLeavesNoRoom;
    };
    log.info("[turboexl3_ling] C1: the bill holds a {d}-token context: KV for {d} tokens (the most mlx-serve allocates for a whole prompt up to it) and {d} checkpoints; room {d} MiB at the default {d}-token reserve, {d} MiB at this one\n", .{ ctx, kv_tokens, ling_rq.checkpointsOf(ctx), host_room >> 20, ling_bill.default_context, room >> 20 });
    // `room` leaves a prompt pass's reserve out (`prefillAt`); decode takes what its own leaves (src/bill.zig).
    // LM_DECODE_ROWS: the layer-major pass's prompt-only working set (`lm_back`, 0 without the pass) goes back to
    // the decode rows (the pass's arrays are gone once the prompt is in; the phase change clears MLX's cache).
    if (lm_back > 0) log.info("[turboexl3_ling] LM_DECODE_ROWS: the decode rows take back the layer-major pass's {d} MiB\n", .{lm_back >> 20});
    if (lm_back > 0) log.info("[turboexl3_ling] C2: the layer-major pass's working set the bill holds at the {d}-token context: {d} MiB (the fitted W at its most over every whole prompt up to it, prompt_work's {d} included)\n", .{ ctx, (lm_back + ling_bill.prompt_work) >> 20, ling_bill.prompt_work >> 20 });
    // The prompt-only term `prefillBytes` holds, given back to the decode rows: with the layer-major pass, MEM_REFIT's
    // server part (C2 folds its pass part into the pass's working set); none without it.
    const prompt_back = promptOnlyBytes(p);
    const fit = ling_bill.budget(kvPerToken(p), lm_back + prompt_back, room, in_use);
    const budget: LingSource.Budget = .{ .prompt = fit.prompt, .decode = fit.decode };
    log.info("[turboexl3_ling] [bank-bill] room {d} MiB, in use {d} MiB: prompt {d} MiB, decode {d} MiB for the slot rows\n", .{ room >> 20, in_use >> 20, budget.prompt >> 20, budget.decode >> 20 });
    const b = try LingSource.Source.open(gpa, io, dir, m.s, geo, gates, first, budget);
    log.info("[turboexl3_ling] streamed experts: {d} prompt / {d} decode slot rows per layer of {d} ({d} B a row of every layer; room {d} MiB, in use {d} MiB)\n", .{ b.bill.rows.prompt, b.bill.rows.decode, c.num_experts, b.bill.row_bytes, room >> 20, in_use >> 20 });
    // RQ (src/ling_rq.zig): the plans' terms from this open, and O's zero, MLX's active bytes with the bank at its
    // prompt rows and no request (Stream.init's prompt windows left out).
    _ = mlx.mlx_synchronize(m.s);
    var open_act: usize = 0;
    _ = mlx.mlx_get_active_memory(&open_act);
    rq = .{
        .terms = .{ .bank = b.bill.terms(c.num_experts), .budget = budget, .context = ctx, .kv_tokens = kv_tokens, .kv = rqKv(p), .rows = b.bill.rows, .pass_bill = ling_bill.prompt_work + lm_back, .resident = ling_bill.rerankHead(c.vocab_size, c.hidden_size), .free_ram = ling_bill.free_ram },
        .zero = @as(u64, open_act) -| b.stream.promptTransientBytes(),
        .pass = passOf(p),
    };
    ling_mtp.rq_append = &rqHeadAppend;
    // RQ's limit (src/ling_rq.zig `promptRefusal`): every whole prompt up to the served context keeps mlx-serve's KV
    // within the KV the bill holds (C1, `promptCapacityMax`).
    log.info("[turboexl3_ling] RQ: whole prompts up to the {d}-token context, mlx-serve's KV for each at most {d} tokens (D2: an MLA layer's prompt spans from position {d} on append in one call); a prompt forward whose KV would pass {d} tokens is refused by name\n", .{ ctx, kv_tokens, ling_rq.d2_from, kv_tokens });
    // MEM_REFIT's run-path mark: with the layer-major pass the bill's prompt-only term is the re-fit's server part.
    if (switches.layerMajor()) ling_ran.mark(.MEM_REFIT);
    // ROUTER_FUSED (src/ling_router.zig): the router's launch scores each call's read-ahead with the next routed layer's
    // gate, the streamer's.
    ling_router.setGates(first, gates);
    m.bank = b;
    const r = lingRouted(b);
    for (m.layers, 0..) |*l, li| if (l.mlp == .moe) {
        l.mlp.moe.streamed = .{ .routed = r, .layer = @intCast(li) };
    };
    // Rule 7 (src/bill.zig `echoRefusalAt`): the room the engine's prompt log-probability step has after a prompt
    // forward. It is the KV the plan holds and the terms idle from the end of a prompt forward until decode's first
    // forward (src/bill.zig `echoIdle`): prompt_work, and with the layer-major pass its working set past prompt_work
    // (R7e; C2: `layerMajorBytes`, MEM_REFIT's pass part folded in), less the host's resident rerank head (R7e-c,
    // `rerankHead`). At that forward the decode rows take that working set and MEM_REFIT's server part back, and
    // decode's own working set replaces prompt_work. The
    // KV at a prompt forward's end is mlx-serve's capacity for it (RQ; src/bill.zig `echoRowsMax`, `echoRefusalAt`).
    echo_plan = .{ .kv_tokens = kv_tokens, .kv_per_token = kvPerToken(p), .vocab = c.vocab_size, .idle = ling_bill.echoIdle(switches.layerMajor(), lm_back, ling_bill.rerankHead(c.vocab_size, c.hidden_size)) };
    log.info("[turboexl3_ling] prompt log-probabilities: whole prompts up to {d} tokens (the engine's step {d} B for each of a row's {d} vocabulary entries, KV {d} B a token, in the plan's room); a longer one is refused by name\n", .{ ling_bill.echoRowsMax(echo_plan.?), ling_bill.echo_entry_bytes, c.vocab_size, echo_plan.?.kv_per_token });
}

/// Rule 7 (src/bill.zig `echoRefusalAt`): the plan's room for the engine's prompt log-probability step, set when the
/// bank opens; null before a load.
var echo_plan: ?ling_bill.EchoPlan = null;
/// The last refusal's reason, which the engine logs as the forward returns.
var echo_why: [320]u8 = undefined;

/// The source as the MoE block's `Routed` hook: its routed sums, phases, read-ahead and settle.
fn lingRouted(src: *LingSource.Source) routed.Routed {
    const S = LingSource.Source;
    const C = struct {
        fn rows(ctx: *anyopaque, layer: u32, x: mlx.mlx_array, idx: mlx.mlx_array, wts: mlx.mlx_array) routed.Ret(mlx.mlx_array) {
            const s: *S = @ptrCast(@alignCast(ctx));
            return .of(s.rows(layer, x, idx, wts));
        }
        fn settle(ctx: *anyopaque) routed.Ret(void) {
            const s: *S = @ptrCast(@alignCast(ctx));
            return .of(s.settle());
        }
        fn phase(ctx: *anyopaque, prompt: bool) routed.Ret(void) {
            const s: *S = @ptrCast(@alignCast(ctx));
            _ = phaseTo(s, if (prompt) .prompt else .decode) catch |e| return .of(e);
            return .of({});
        }
        fn readAhead(ctx: *anyopaque, layer: u32, experts: []const u16) routed.Ret(void) {
            const s: *S = @ptrCast(@alignCast(ctx));
            return .of(s.readAhead(layer, experts));
        }
    };
    return .{ .ctx = src, .math = .{ .source = C.rows }, .settle_fn = C.settle, .phase_fn = C.phase, .read_ahead_fn = C.readAhead, .decode_ids = LingSource.decode_route_ids };
}

/// A forward: a call past the bank's decode lane in its prompt phase, every other in decode; the result evaluated
/// before the call's routes settle (their slot rows are free for the next call's reads).
pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "turboexl3_ling: forward mode not declared");
    const first_pos: u64 = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    // The run-path line per request (src/ling_ran.zig `Boundary`, `requestStart`, `reportRequest`): the boundary says
    // which forward starts a request and which is its first decode-phase forward, whose request prompt line prints first
    // and its decode line after it. It moves on only once no refusal below has stopped the forward.
    var boundary = request_boundary;
    const at_boundary = boundary.next(.{ .prefill = f.mode == .prefill, .pos0 = first_pos, .rows = f.rows });
    // A decode lane's rows: a forward of these or fewer runs in the decode phase, and the layer-major pass forwards a
    // span of these or fewer on its own (`spanMajorOr`).
    const lane_rows = LingSource.decode_route_ids / @max(m.c.num_experts_per_tok, 1);
    // RQ (src/ling_rq.zig): a prompt forward's KV as mlx-serve's capacity schedule grows it over the forward's appends,
    // each span's or D2's group's (none for a decode or verify forward, and none before the bank opens).
    const slot_id: u32 = if (f.slots.len > 0) f.slots.slice()[0] else 0;
    const rq_grow: ?ling_rq.Growth = if (rq) |*x| (if (f.mode == .prefill) rqPromptGrowth(x, f, slot_id, at_boundary.start, first_pos, lane_rows) else null) else null;
    // Rule 7 (src/bill.zig `echoRefusalAt`): a prompt forward that asks every row's logits (prompt log-probabilities)
    // is refused by name, before it changes anything, when the engine's step over them would not fit the plan's room
    // with the forward's KV at the capacity mlx-serve allocates for it. The engine answers any refused forward with its
    // HTTP 500 "generation failed"; the reason reaches the server log only.
    if (echo_plan) |plan| if (rq_grow) |g| {
        if (ling_bill.echoRefusalAt(&echo_why, plan, f.mode == .prefill, f.logits == .all_rows, first_pos, f.rows, g.cap)) |why| {
            log.err("[turboexl3_ling] {s}\n", .{why});
            return Verdict.named(.too_large, why);
        }
    };
    // RQ's refusal (option c, src/ling_rq.zig `promptRefusal`): a prompt forward whose KV mlx-serve would grow past the
    // KV the bill holds (C1: the served context's longest whole prompt's), refused by name before any append, as rule
    // 7's. No whole prompt from position 0 up to the served context does.
    if (rq) |*x| if (rq_grow) |g| if (ling_rq.promptRefusal(&rq_why, x.terms, first_pos, f.rows, g.cap)) |why| {
        log.err("[turboexl3_ling] {s}\n", .{why});
        return Verdict.named(.too_large, why);
    };
    // C6 (src/ling_rope1_spike.zig): a forward with a call of 2 to 16 rows (not an exact verify's) that ends past the
    // bill's point, where the rope-once fallback's rebuilt key would pass the prompt pass's idle working set, is refused
    // by name before any append, as RQ's. No whole prompt from position 0 is.
    {
        const kind: ling_spike.Kind = switch (f.mode) {
            .prefill => .prefill,
            .verify => if (m.caps.exact_drafted) .exact_verify else .verify,
            .decode, .batched_rows => .decode,
        };
        const spike = ling_bill.ropeOnceSpike(m.c.num_attention_heads, m.c.mla_qk_nope_head_dim, m.c.mla_qk_rope_head_dim);
        const pt = ling_spike.point(m.c.num_attention_heads, m.c.mla_qk_nope_head_dim, m.c.mla_qk_rope_head_dim);
        const ends: []const u64 = if (f.mode == .prefill) (if (hostSpans(f)) |hs| hs.ends else &.{}) else &.{};
        if (ling_spike.forwardRefusal(&spike_why, kind, first_pos, f.rows, ends, pt, spike)) |why| {
            log.err("[turboexl3_ling] {s}\n", .{why});
            return Verdict.named(.too_large, why);
        }
    }
    request_boundary = boundary;
    if (at_boundary.start) ling_ran.requestStart();
    if (at_boundary.first_decode) ling_ran.reportRequest(.prompt);
    defer if (at_boundary.first_decode) ling_ran.reportRequest(.decode);
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "turboexl3_ling: a forward without cache_ops");
    const b = m.bank orelse return Verdict.refuse(.failed, "turboexl3_ling: no bank");
    // RQ, whose state opens with the bank (`openBank`): the request's slot (a fresh entry at its start), and a prompt
    // forward's end and growth.
    const q = if (rq) |*x| x else return Verdict.refuse(.failed, "turboexl3_ling: the bank has no RQ state");
    rqAtForward(q, slot_id, at_boundary.start, first_pos, f.rows, rq_grow);
    const want: LingSource.Phase = if (f.rows > lane_rows) .prompt else .decode;
    if (want != b.phase) {
        if (want == .decode) {
            promptEndMemLine(b);
            // EMBED_ROWS: the table the trunk lends must still be unevaluated (logged by name once if not)
            ling_embed.checkLent(m.embed.w);
        }
        // Decode's end (the next prompt begins): PF_DFIN's counts at its last step, beside `[stream-stats]`'s own.
        if (want == .prompt) ling_math.logDfinCalls(.{ .decode = b.steps });
        log.info("[turboexl3_ling] phase {t} -> {t}: a {t} forward of {d} rows\n", .{ b.phase, want, f.mode, f.rows });
    }
    // RQ's re-plans at a phase change (`phaseTo`): a request's prompt start frees its decode rows, its decode start grows
    // the rows its plan gives.
    const replanned = phaseTo(b, want) catch |e| return rqVerdict(e);
    // RQ (`rqForward`): a request's start that changed no phase, a decode-phase forward that would append past its plan
    // (the trim), then the slot's capacity.
    rqForward(q, b, f, at_boundary.start, first_pos, replanned) catch |e| return rqVerdict(e);
    // After each prompt forward, whatever path it takes: PF_DFIN's counts (src/ling_math.zig `logDfinCalls`), then the
    // run-path line when a path ran that no earlier line named (src/ling_ran.zig `report`).
    defer if (f.mode == .prefill) {
        ling_math.logDfinCalls(.{ .prompt = f.rows });
        ling_ran.report(.prompt);
    };
    // MTP_HEAD_Q4: a prompt pass restarts the request's draft-step count; a verify after a draft step on the 4-bit head
    // marks its run path (src/ling_mtp.zig `q4AtVerify`).
    if (f.mode == .prefill) ling_mtp.q4PromptStart() else if (f.mode == .verify) ling_mtp.q4AtVerify();
    if (f.mode == .prefill and m.caps.whole_prompt_prefill) {
        if (hostSpans(f) != null) return spanMajor(m, ops, f, out, b);
        // A host without the span ABI (older than this build's kit): its calls would not be the chunk loop's bytes, so
        // each whole prompt is refused by name.
        if (!hostNamesSpans(f)) return refuseNoSpans();
        // A host with the span ABI that names no spans for this forward (the ABI's "the model chunks as it likes"): the
        // prompt in layer-major calls.
        return promptCalls(m, ops, f, out, b);
    }
    // The exact verify: a verify's rows take serial decode's one-row paths where several rows would not.
    ling_exl3.exact_rows = f.mode == .verify and m.caps.exact_drafted;
    defer ling_exl3.exact_rows = false;
    return forwardSettled(m, ops, f, out, b);
}

/// The request boundary over the forwards the host hands `forward` (src/ling_ran.zig `Boundary`).
var request_boundary: ling_ran.Boundary = .{};

/// RQ (src/ling_rq.zig): the bank's decode rows planned per request for the KV mlx-serve allocates. Its state is set
/// when the bank opens (`openBank`); null before a load's bank opens, as rule 7's plan.
var rq: ?Rq = null;
/// RQ's refusal (src/ling_rq.zig `promptRefusal`), which the engine logs as the forward returns.
var rq_why: [320]u8 = undefined;
/// C6's refusal (src/ling_rope1_spike.zig `forwardRefusal`), which the engine logs as the forward returns.
var spike_why: [384]u8 = undefined;
/// C5's over-plan refusal (src/ling_rq.zig `overVerdict`, `overRefusal`): its text, the forward's verdict's reason
/// (`rqVerdict`).
var rq_over_why: [320]u8 = undefined;
var rq_over: []const u8 = ling_rq.over_refusal_name;

/// A failed RQ step's verdict: C5's over-plan refusal by name, with the reason its plan logged; any other error by its
/// name.
fn rqVerdict(e: anyerror) Verdict {
    if (e == error.LingRqOverPlan) return Verdict.named(.too_large, rq_over);
    return Verdict.named(.failed, @errorName(e));
}

const Rq = struct {
    /// The plans' terms, from the bank's open (`openBank`).
    terms: ling_rq.Terms,
    /// O's zero: MLX's active bytes once the bank opened, less its prompt windows: the plugin's own memory with the bank
    /// at its prompt rows and no request.
    zero: u64,
    /// The bank's prompt windows when a reading holds them (`promptTransientBytes`), which O leaves out as its zero does:
    /// a request's start that changed no phase reads in the prompt phase, beside them; the phase changes and the trim
    /// read once they are freed (0).
    windows: u64 = 0,
    slots: ling_rq.SlotTable = .{},
    /// The current request's slot: the last trunk forward's (a head forward names none).
    cur: u32 = 0,
    /// The re-plan the planner makes next, and where: the position and rows its plan's end counts from.
    at: ling_rq.Kind = .prompt,
    pos: u64 = 0,
    n_rows: u64 = 1,
    /// The planner's last plan, and O as read (signed, so a reading under the zero shows in the line).
    plan: ling_rq.Plan = .{},
    o: i64 = 0,
    /// C5: the pass model's geometry (src/bill.zig `Pass`), for a prompt plan's own working set (`passWork`).
    pass: ling_bill.Pass,
};

/// The bank's phase `p` (ling_source.zig `setPhase`), and RQ's re-plan of the current request where the phase changes,
/// its line printed: true when RQ re-planned (a phase change with a request in the slot; the load's warm-up has none).
fn phaseTo(b: *LingSource.Source, p: LingSource.Phase) !bool {
    const q = if (rq) |*x| x else return error.LingBankWithoutRq;
    const pos: u64 = if (q.slots.get(q.cur)) |s| s.p else 0;
    const ch = try b.setPhase(p, rqPlanner(q, if (p == .prompt) .prompt else .decode, pos, 1));
    if (ch.planned) rqLine(q, ch);
    if (ch.planned and ling_rq.overVerdict(q.terms, q.plan) == .refuse) return error.LingRqOverPlan;
    return ch.planned;
}

/// RQ's planner for the current request's re-plan of `kind`, its plan's end counted from `pos` and a forward of
/// `n_rows` there (src/ling_rq.zig `decodeEnd`).
fn rqPlanner(q: *Rq, kind: ling_rq.Kind, pos: u64, n_rows: u64) LingSource.Planner {
    q.at = kind;
    q.pos = pos;
    q.n_rows = n_rows;
    return .{ .ctx = q, .plan = &rqPlan };
}

/// RQ's planner (`LingSource.Planner`), where the bank holds its prompt rows alone, MLX synchronized and its cache
/// cleared. O is MLX's active bytes now, less O's zero, the request's KV as allocated (both caches at their capacities
/// so far) and any prompt windows the reading holds (`Rq.windows`). The plan then gives the decode rows; with no request
/// in the slot (the engine's warm-up, a decode-shape forward with no prompt), null: the bank's own.
fn rqPlan(ctx: *anyopaque) ?u32 {
    const q: *Rq = @ptrCast(@alignCast(ctx));
    const s = q.slots.get(q.cur) orelse return null;
    var act: usize = 0;
    _ = mlx.mlx_get_active_memory(&act);
    const own = s.trunk_cap * q.terms.kv.trunk() + s.head_cap * q.terms.kv.head + q.windows;
    q.o = @as(i64, @intCast(act)) - @as(i64, @intCast(q.zero)) - @as(i64, @intCast(own));
    const other: u64 = if (q.o > 0) @intCast(q.o) else 0;
    q.plan = switch (q.at) {
        .prompt => ling_rq.planPrompt(q.terms, s, other, if (switches.layerMajor()) ling_bill.passWork(q.pass, s.p) else ling_bill.prompt_work),
        .decode, .trim => ling_rq.planDecode(q.terms, s, q.at, q.pos, q.n_rows, other),
    };
    if (q.at != .prompt) {
        s.plan_cap = q.plan.cap;
        s.plan_head_cap = q.plan.head_cap;
        s.rows = q.plan.rows;
    }
    return q.plan.rows;
}

/// RQ's line at each re-plan and trim, from which main measures a re-plan's cost: the slot, P, the restored prefix r,
/// the capacity the plan holds (the trunk's, the head's beside it), its KV, O as read, the rows before and after, and
/// the times of the frees and of the allocation. A request's first prints the engaged line before it; a plan under the
/// base prints the over-plan line after it. RQ's run-path mark (src/ling_ran.zig).
fn rqLine(q: *Rq, ch: LingSource.Change) void {
    const s = q.slots.get(q.cur) orelse return;
    const pl = q.plan;
    ling_ran.mark(.RQ);
    if (!s.said) {
        s.said = true;
        log.info("[turboexl3_ling] RQ engaged: slot={d} P={d} capacity={d} rows={d}\n", .{ s.id, s.p, pl.cap, ch.rows_after });
    }
    log.info("[turboexl3_ling] RQ {t}: slot={d} P={d} r={d} capacity={d} head={d} KV={d} MiB O={d} MiB rows {d} -> {d} free={d:.1} ms alloc={d:.1} ms\n", .{ pl.kind, s.id, s.p, s.r, pl.cap, pl.head_cap, pl.kv >> 20, @divTrunc(q.o, 1 << 20), ch.rows_before, ch.rows_after, msOf(ch.free_ns), msOf(ch.alloc_ns) });
    if (pl.kind == .prompt) log.info("[turboexl3_ling] RQ prompt plan (C5): slot={d} P={d}: KV {d} MiB at capacity {d}, the pass {d} MiB, O past the rerank head {d} MiB, against the bill's KV for {d} tokens and pass {d} MiB: room for {d} rows, base {d}\n", .{ s.id, s.p, pl.kv >> 20, pl.cap, pl.pass >> 20, (pl.other -| q.terms.resident) >> 20, q.terms.kv_tokens, q.terms.pass_bill >> 20, pl.held, q.terms.rows.prompt });
    // C5 (src/ling_rq.zig `overVerdict`): a plan under the base. A prompt plan, or a decode or trim plan under it by more
    // than the rows free RAM covers, is refused by name: the step that made it returns error.LingRqOverPlan, and the
    // forward's verdict carries this reason (`rqVerdict`). A decode or trim plan within the cover keeps the bank at its
    // base, with this warning.
    switch (ling_rq.overVerdict(q.terms, pl)) {
        .fits => {},
        .warn => log.warn("[turboexl3_ling] RQ over plan: slot={d} P={d}: with O at {d} MiB the {t} plan holds {d} rows, under the base {d} by {d}, within the {d} rows free RAM covers, so the bank stays at its base\n", .{ s.id, s.p, pl.other >> 20, pl.kind, pl.held, q.terms.rows.prompt, q.terms.rows.prompt -| pl.held, ling_rq.coverRows(q.terms) }),
        .refuse => {
            rq_over = ling_rq.overRefusal(&rq_over_why, q.terms, s, pl);
            log.err("[turboexl3_ling] {s}\n", .{rq_over});
        },
    }
}

fn msOf(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

/// RQ at a trunk forward, before its phase change: the request's slot (a fresh entry at a request's start; a forward
/// after another slot's is logged as interleaved, once a request), and a prompt forward's end and growth. A forward of
/// a slot with no request (the engine's warm-up: a decode-shape forward with no prompt and no KV) gets no plan.
fn rqAtForward(q: *Rq, slot_id: u32, start: bool, pos0: u64, n_rows: u64, pg: ?ling_rq.Growth) void {
    q.cur = slot_id;
    const prev = q.slots.last;
    const s = if (start) q.slots.begin(slot_id, pos0) else blk: {
        const interleaved = q.slots.touch(slot_id);
        const e = q.slots.get(slot_id) orelse return;
        if (interleaved and !e.said_interleave) {
            e.said_interleave = true;
            log.warn("[turboexl3_ling] RQ interleaved: slot {d} resumes after slot {d}'s forwards; O holds that request's KV as it is now, not its growth (said once a request)\n", .{ slot_id, prev orelse slot_id });
        }
        break :blk e;
    };
    if (pg) |g| {
        s.p = pos0 + n_rows;
        s.prompt = g;
    }
}

/// RQ: a prompt forward's KV growth, from its restored prefix at a request's start, else from its slot's capacity so far
/// (a later chunk of the chunk loop): over the spans the host names for it as the layer-major pass appends them
/// (src/ling_rq.zig `replayAppends`: each span alone, D2's groups in one append; a span of `lane` rows or fewer is
/// forwarded on its own), else at its own end (`replay`).
fn rqPromptGrowth(q: *Rq, f: *const abi.Forward, slot_id: u32, start: bool, pos0: u64, lane: u64) ling_rq.Growth {
    const from: u64 = if (start) pos0 else if (q.slots.get(slot_id)) |s| s.trunk_cap else pos0;
    if (hostSpans(f)) |hs| return ling_rq.replayAppends(from, pos0, hs.ends, lane);
    return ling_rq.replay(from, &[_]u64{pos0 + f.rows});
}

/// RQ at a trunk forward, after its phase change (src/ling_rq.zig):
/// - a request's start that changed no phase re-plans here: in the prompt phase its prompt's plan, read at once (MLX
///   synchronized and its cache cleared first; the bank holds its prompt rows and its prompt windows, which O leaves
///   out), and in decode (a prompt of a decode lane's rows or fewer) its decode plan, through a trim;
/// - a decode-phase forward whose append would pass its plan's capacity (`crosses`) trims the bank first;
/// - then the slot's trunk capacity takes the forward's append.
fn rqForward(q: *Rq, b: *LingSource.Source, f: *const abi.Forward, start: bool, pos0: u64, replanned: bool) !void {
    const s = q.slots.get(q.cur) orelse return;
    if (start and !replanned) {
        if (b.phase == .prompt) {
            const t0 = nowNs();
            _ = mlx.mlx_synchronize(b.g.s);
            b.g.clearCache();
            q.at = .prompt;
            q.windows = b.stream.promptTransientBytes();
            _ = rqPlan(q);
            q.windows = 0;
            rqLine(q, .{ .rows_before = b.prompt_rows[0], .rows_after = b.prompt_rows[0], .free_ns = nowNs() - t0, .planned = true });
            if (ling_rq.overVerdict(q.terms, q.plan) == .refuse) return error.LingRqOverPlan;
        } else try rqTrim(q, b, .decode, s.p, 1);
    } else if (b.phase == .decode and s.plan_cap > 0 and ling_rq.crosses(pos0, f.rows, s.plan_cap)) {
        try rqTrim(q, b, .trim, pos0, f.rows);
    }
    s.trunk_cap = if (f.mode == .prefill) @max(s.trunk_cap, s.prompt.cap) else ling_rq.nextCapacity(s.trunk_cap, pos0 + f.rows);
}

/// RQ's trim (ling_source.zig `trimDecode`): the decode rows shrunk to the bank's prompt rows, then grown at the rows the
/// re-plan of `kind` gives, its line printed.
fn rqTrim(q: *Rq, b: *LingSource.Source, kind: ling_rq.Kind, pos: u64, n_rows: u64) !void {
    const ch = try b.trimDecode(rqPlanner(q, kind, pos, n_rows));
    if (ch.planned) rqLine(q, ch);
    if (ch.planned and ling_rq.overVerdict(q.terms, q.plan) == .refuse) return error.LingRqOverPlan;
}

/// RQ at an MTP head forward (src/ling_mtp.zig `rq_append`): its append of `n_rows` rows at the head cache's length
/// `len`, the current request's. One that would pass the head's planned capacity trims the bank first, as a trunk
/// forward's crossing does (the plan holds `ling_rq.head_margin` rows past the trunk's capacity, so the trunk's own
/// trim comes first); then the slot's head capacity takes the append.
fn rqHeadAppend(mp: *anyopaque, len: u64, n_rows: u64) anyerror!void {
    const q = if (rq) |*x| x else return error.LingBankWithoutRq;
    const s = q.slots.get(q.cur) orelse return;
    const b = host.of(mp).bank orelse return;
    if (b.phase == .decode and s.plan_head_cap > 0 and ling_rq.crosses(len, n_rows, s.plan_head_cap)) try rqTrim(q, b, .trim, len, n_rows);
    s.head_cap = ling_rq.nextCapacity(s.head_cap, len + n_rows);
}

/// A monotonic clock reading (ns), for RQ's prompt-start reading.
fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// One forward through the copied model, its outputs evaluated, then the bank's routes settled.
fn forwardSettled(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, b: *LingSource.Source) Verdict {
    const v = host.forwardOne(plugin.name, fwd, m, ops, f, out);
    if (!v.isOk()) return v;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ out.logits, out.hidden }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, .{ .ctx = a.ctx });
    };
    mlx.check(mlx.mlx_eval(vec)) catch |e| return Verdict.named(.failed, @errorName(e));
    b.settle() catch |e| return Verdict.named(.failed, @errorName(e));
    if (b.phase == .decode) {
        // PF_DFIN's counts at the decode `[stream-stats]` cadence: every 64 decode steps, the source's own count.
        if (b.steps % 64 == 0) ling_math.logDfinCalls(.{ .decode = b.steps });
        // The run-path line after a decode round, when a path ran that no earlier line named. A prompt's short span runs
        // in the decode phase too, as a forward of its own inside the prompt forward (`spanMajorOr`): that forward's own
        // line reports it.
        if (f.mode != .prefill) ling_ran.report(.decode);
    }
    return .ok;
}

/// The layer-major prompt pass, where this build's kit names the chunk loop's spans (`switches.layerMajor`): the
/// engine hands the plugin the whole prompt (`Caps.whole_prompt_prefill`, config.zig). Where the engine names the chunk
/// loop's spans for the forward (`Forward.prompt_spans`) the prompt runs layer by layer with each span's bytes the chunk
/// loop's (`spanMajor`); where it names none, in calls of `layer_major_rows` rows, each call through every layer at once
/// (`promptCalls`: other bytes than the chunk loop's). Either way a routed layer reads each of its experts once a call
/// rather than once an engine chunk (4,096 rows).
pub const layer_major_rows: u32 = ling_segments.routed_rows;
/// A last call shorter than this joins the call before it (it would re-read the experts the call before it held).
pub const layer_major_tail_rows: u32 = ling_segments.tail_rows;
/// `promptCalls`' line, printed once a process.
var layer_major_calls_said = false;

/// The rows of the call at row `at` of a `total`-row prompt.
pub fn callRows(total: u32, at: u32) u32 {
    const left = total - at;
    return if (left < layer_major_rows + layer_major_tail_rows) left else layer_major_rows;
}

/// A whole prompt in layer-major calls: each call a forward of its rows at its position, evaluated and settled before
/// the next; the last call's logits (every call's, for all rows), every call's hidden rows when wanted.
fn promptCalls(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, b: *LingSource.Source) Verdict {
    if (f.positions.len != 1 or f.caches.len != 1) return Verdict.refuse(.unsupported, "turboexl3_ling: one slot's prompt at a time");
    if (f.media.ctx != null or f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "turboexl3_ling: a layer-major prompt carries no media or captures");
    const pos0 = f.positions.slice()[0];
    if (!layer_major_calls_said) {
        layer_major_calls_said = true;
        log.info("[turboexl3_ling] LAYER_MAJOR in calls of {d} rows: the engine names no spans, so not the chunk loop's bytes\n", .{layer_major_rows});
    }
    var logit_parts: std.ArrayList(mlx.mlx_array) = .empty;
    var hidden_parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (logit_parts.items) |a| _ = mlx.mlx_array_free(a);
        for (hidden_parts.items) |a| _ = mlx.mlx_array_free(a);
        logit_parts.deinit(gpa);
        hidden_parts.deinit(gpa);
    }
    var at: u32 = 0;
    while (at < f.rows) {
        const n = callRows(f.rows, at);
        const last = at + n == f.rows;
        var sub = f.*;
        sub.rows = n;
        const pos = [_]u32{pos0 + at};
        sub.positions = .of(&pos);
        var ids_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_view);
        if (f.ids.ctx != null) {
            mlx.check(mlx.mlx_slice(&ids_view, .{ .ctx = f.ids.ctx }, &[_]c_int{ 0, @intCast(at) }, 2, &[_]c_int{ 1, @intCast(at + n) }, 2, &[_]c_int{ 1, 1 }, 2, m.s)) catch |e| return Verdict.named(.failed, @errorName(e));
            sub.ids = .{ .ctx = ids_view.ctx };
        } else sub.tokens = .of(f.tokens.slice()[at .. at + n]);
        sub.logits = if (f.logits == .all_rows) .all_rows else if (last) f.logits else .none;
        var so: abi.ForwardOut = .{};
        const v = forwardSettled(m, ops, &sub, &so, b);
        if (!v.isOk()) return v;
        if (so.logits.ctx != null) logit_parts.append(gpa, .{ .ctx = so.logits.ctx }) catch {
            _ = mlx.mlx_array_free(.{ .ctx = so.logits.ctx });
            return Verdict.named(.failed, "OutOfMemory");
        };
        if (so.hidden.ctx != null) hidden_parts.append(gpa, .{ .ctx = so.hidden.ctx }) catch {
            _ = mlx.mlx_array_free(.{ .ctx = so.hidden.ctx });
            return Verdict.named(.failed, "OutOfMemory");
        };
        at += n;
        if (f.progress) |p| @atomicStore(u64, p, at, .monotonic);
    }
    inline for (.{ .{ &logit_parts, &out.logits, 0 }, .{ &hidden_parts, &out.hidden, 1 } }) |t| {
        const parts = t[0];
        if (parts.items.len == 1) {
            t[1].* = .{ .ctx = parts.items[0].ctx };
            parts.items.len = 0;
        } else if (parts.items.len > 1) {
            const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var joined = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&joined, vec, t[2], m.s)) catch |e| {
                _ = mlx.mlx_array_free(joined);
                return Verdict.named(.failed, @errorName(e));
            };
            t[1].* = .{ .ctx = joined.ctx };
        }
    }
    return .ok;
}

/// The engine's chunk-loop spans and checkpoint positions for a whole-prompt forward (`Forward.prompt_spans`,
/// `Forward.ssm_checkpoints`); null on a host whose `Forward` ends before them (`hostNamesSpans`), or an empty list.
const HostSpans = struct { ends: []const u64, checkpoints: []const u64 };
fn hostSpans(f: *const abi.Forward) ?HostSpans {
    if (!hostNamesSpans(f)) return null;
    const ends = f.prompt_spans.slice();
    if (ends.len == 0) return null;
    return .{ .ends = ends, .checkpoints = f.ssm_checkpoints.slice() };
}

/// The host's `Forward` carries the span ABI (`prompt_spans`, `ssm_checkpoints`), whether or not this forward names spans:
/// a host older than this build's kit sends a shorter `Forward`.
fn hostNamesSpans(f: *const abi.Forward) bool {
    return f.size >= @offsetOf(abi.Forward, "ssm_checkpoints") + @sizeOf(@TypeOf(f.ssm_checkpoints));
}

/// The layer-major prompt pass (this build's kit names spans) on a host that does not: each whole prompt is refused by
/// name, never run in calls of `layer_major_rows` rows; the load's first refusal logs it once.
const no_spans_fix = "build the plugin with the serving mlx-serve's own kit, or update mlx-serve to a release with whole-prompt spans (ddalcu/mlx-serve#1126)";
var no_spans_said = false;
fn refuseNoSpans() Verdict {
    if (!no_spans_said) {
        no_spans_said = true;
        log.err("[turboexl3_ling] this host names no chunk-loop spans (Forward.prompt_spans), so the layer-major prompt pass this build declares refuses each whole prompt: " ++ no_spans_fix ++ "\n", .{});
    }
    return Verdict.refuse(.unsupported, "turboexl3_ling: this host names no chunk-loop spans: " ++ no_spans_fix);
}

/// The layer-major prompt pass on an engine that names its chunk loop's spans: the whole prompt layer by layer, each
/// span's bytes the chunk loop's (table_forward `hiddenSpans`: every row-sensitive op on one span at a time, the routed
/// experts in calls over whole spans, `layer_major_rows` rows a call). A span of a decode lane's rows or fewer (the
/// engine forwards it as a decode forward) runs as a forward of its own between runs of wider spans. C4: each run goes
/// through every layer one segment at a time (src/ling_lm_pass.zig `runWide`, at most 32,768 rows a segment, cut at the
/// run's routed-call ends, a last group under 8,192 rows joined to the segment before it, so a prompt of 32,768 tokens
/// or fewer runs as one and mlx-serve's KV grows as the single pass's), and each KDA layer's state after each
/// span ending at a checkpoint position goes to the engine (`CacheOps.checkpoint`) once its segment has passed every
/// layer, in order, as the chunk loop files its checkpoints. Logits: the last row, or every span's rows (`all_rows`).
/// The hidden rows (`want_hidden`, the MTP head's history) are written once into one array as each segment ends
/// (`HiddenRows`). D2: at an MLA layer a segment's spans from 24,576 on (never the forward's first) append their keys
/// and values in one call (src/ling_rq.zig `d2Start`, `groupStart`), so mlx-serve grows the KV once for them, each
/// span's words as before.
fn spanMajor(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, b: *LingSource.Source) Verdict {
    return spanMajorOr(m, ops, f, out, b) catch |e| rqVerdict(e);
}

fn spanMajorOr(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, b: *LingSource.Source) !Verdict {
    const hs = hostSpans(f).?;
    if (f.positions.len != 1 or f.caches.len != 1) return Verdict.refuse(.unsupported, "turboexl3_ling: one slot's prompt at a time");
    if (f.media.ctx != null or f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "turboexl3_ling: a layer-major prompt carries no media or captures");
    const pos0: u64 = f.positions.slice()[0];
    const n = hs.ends.len;
    const rows = try gpa.alloc(u32, n);
    defer gpa.free(rows);
    var prev = pos0;
    for (hs.ends, rows) |e, *r| {
        if (e <= prev) return error.LayerMajorSpansNotAscending;
        r.* = @intCast(e - prev);
        prev = e;
    }
    if (prev != pos0 + f.rows) return error.LayerMajorSpansNotThePrompt;
    for (hs.checkpoints) |cp| if (std.mem.indexOfScalar(u64, hs.ends, cp) == null) return error.LayerMajorCheckpointOffSpan;
    // C4: the checkpoints are filed as the pass goes, so a host that names them must take them; it is asked first
    if (hs.checkpoints.len > 0) {
        if (ops.size < @offsetOf(abi.CacheOps, "checkpoint") + @sizeOf(@TypeOf(ops.checkpoint))) return Verdict.refuse(.unsupported, "turboexl3_ling: the engine names checkpoints but its CacheOps takes none");
        if (ops.checkpoint == null) return Verdict.refuse(.unsupported, "turboexl3_ling: the engine names checkpoints but takes none");
    }
    const lane_rows: u32 = @intCast(LingSource.decode_route_ids / @max(m.c.num_experts_per_tok, 1));
    // D2 (src/ling_rq.zig `d2Start`): from here an MLA layer's spans in a `hiddenSpans` call, two or more of them, append
    // their keys and values in one call, never the forward's first span; RQ's replay (`rqPromptGrowth`) counts the same.
    const d2_from = ling_rq.d2Start(pos0);
    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    const caches = f.caches.slice();
    var logit_parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (logit_parts.items) |x| _ = mlx.mlx_array_free(x);
        logit_parts.deinit(gpa);
    }
    // `want_hidden` (the MTP head's history): every span's final-normed hidden rows, in the prompt's order, as the
    // chunk loop's forwards return them (host.finish: the rows `hidden` returns), written once into one array (C4).
    var hidden: lm_pass.HiddenRows = .{ .s = m.s, .rows = f.rows };
    defer hidden.deinit();
    // The next checkpoint to file: each is filed once the forward has passed it in every layer, in order.
    var ci: usize = 0;
    var j: usize = 0;
    var at: u32 = 0;
    while (j < n) {
        const small = rows[j] <= lane_rows;
        var k = j + 1;
        if (!small) while (k < n and rows[k] > lane_rows) : (k += 1) {};
        var seg_rows: u32 = 0;
        for (rows[j..k]) |r| seg_rows += r;
        const last = k == n;
        const seg_pos: u32 = @intCast(pos0 + at);
        const run_end: u64 = pos0 + at + seg_rows;
        if (small) {
            // A span the engine forwards as a decode forward: one of its own (phase, routes and all).
            var sub = f.*;
            sub.rows = seg_rows;
            const p = [_]u32{seg_pos};
            sub.positions = .of(&p);
            var idv = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idv);
            try mlx.check(mlx.mlx_slice(&idv, ids, &[_]c_int{ 0, @intCast(at) }, 2, &[_]c_int{ 1, @intCast(at + seg_rows) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
            sub.ids = .{ .ctx = idv.ctx };
            sub.tokens = .{};
            sub.logits = if (f.logits == .all_rows) .all_rows else if (last) f.logits else .none;
            sub.want_hidden = f.want_hidden;
            _ = try phaseTo(b, .decode);
            var so: abi.ForwardOut = .{};
            const v = forwardSettled(m, ops, &sub, &so, b);
            if (!v.isOk()) return v;
            if (so.logits.ctx != null) try logit_parts.append(gpa, .{ .ctx = so.logits.ctx });
            if (so.hidden.ctx != null) {
                defer _ = mlx.mlx_array_free(.{ .ctx = so.hidden.ctx });
                if (f.want_hidden) try hidden.put(at, .{ .ctx = so.hidden.ctx });
            }
            // A checkpoint at this span's end: every KDA layer's state as the forward left it, filed now (the host
            // copies it).
            while (ci < hs.checkpoints.len and hs.checkpoints[ci] == run_end) : (ci += 1) {
                var st: [256]abi.LayerState = undefined;
                var ns: usize = 0;
                for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
                    if (ns == st.len) return error.LayerMajorTooManyLayers;
                    st[ns] = .{ .layer = @intCast(li), .conv = ops.state(caches[0], @intCast(li), .conv), .ssm = ops.state(caches[0], @intCast(li), .ssm) };
                    ns += 1;
                };
                const cv = ops.checkpoint.?(caches[0], run_end, .of(st[0..ns]));
                if (!cv.isOk()) return cv;
            }
        } else {
            // C4: the run through every layer a segment at a time (src/ling_lm_pass.zig `runWide`), its checkpoints filed
            // as each segment ends.
            var cend = ci;
            while (cend < hs.checkpoints.len and hs.checkpoints[cend] <= run_end) : (cend += 1) {}
            var idv = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idv);
            try mlx.check(mlx.mlx_slice(&idv, ids, &[_]c_int{ 0, @intCast(at) }, 2, &[_]c_int{ 1, @intCast(at + seg_rows) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
            var w: WideRun = .{ .m = m, .f = f, .b = b, .ops = ops, .cache = caches[0], .logit_parts = &logit_parts, .hidden = if (f.want_hidden) &hidden else null, .row0 = at, .n = k - j, .last = last };
            _ = lm_pass.runWide(m, ops, caches, idv, rows[j..k], seg_pos, hs.checkpoints[ci..cend], d2_from, ling_segments.segment_rows, w.hooks()) catch |e| {
                if (w.refused) |rv| return rv;
                return e;
            };
            ci = cend;
        }
        at += seg_rows;
        j = k;
        if (f.progress) |pr| @atomicStore(u64, pr, at, .monotonic);
    }
    if (ci != hs.checkpoints.len) return error.LayerMajorCheckpointOffSpan;
    // LAYER_MAJOR's run-path mark: the prompt ran layer by layer over the engine's spans, each span's bytes the chunk
    // loop's, its checkpoints filed through CacheOps.checkpoint.
    ling_ran.mark(.LAYER_MAJOR);
    // LAST_ROW_LOGITS's run-path mark: the engine asked this prompt's last row (`Caps.prompt_last_row`), so the head ran
    // over the last span only.
    if (f.logits == .last_row) ling_ran.mark(.LAST_ROW_LOGITS);
    // The logits: one part, or every span's rows joined.
    if (logit_parts.items.len == 1) {
        out.logits = .{ .ctx = logit_parts.items[0].ctx };
        logit_parts.items.len = 0;
    } else if (logit_parts.items.len > 1) {
        const vec = mlx.mlx_vector_array_new_data(logit_parts.items.ptr, logit_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, m.s));
        out.logits = .{ .ctx = joined.ctx };
    }
    // The hidden rows ([1, rows, H]), written once (C4), evaluated.
    if (try hidden.take()) |hd| {
        errdefer _ = mlx.mlx_array_free(hd);
        try mlx.check(mlx.mlx_array_eval(hd));
        out.hidden = .{ .ctx = hd.ctx };
    }
    return .ok;
}

/// C4's hooks for a run of wide spans (src/ling_lm_pass.zig `runWide`): before each segment the bank's prompt phase
/// (and at the run's first, the pass's start: no protection from earlier prompt-phase calls); after each the head's
/// logits (every span's rows for `all_rows`, else the forward's last row) and the hidden rows, evaluated, then the
/// bank's routes settled; each checkpoint through the host's op.
const WideRun = struct {
    m: *trunk.Model,
    f: *const abi.Forward,
    b: *LingSource.Source,
    ops: *const abi.CacheOps,
    cache: abi.Handle,
    logit_parts: *std.ArrayList(mlx.mlx_array),
    hidden: ?*lm_pass.HiddenRows,
    /// The run's first row in the forward, its spans, and whether it ends the forward.
    row0: u64,
    n: usize,
    last: bool,
    /// The host's refusal of a checkpoint, which the forward returns.
    refused: ?Verdict = null,

    fn hooks(w: *WideRun) lm_pass.Hooks {
        return .{ .ctx = w, .begin = &begin, .after = &after, .file = &file };
    }

    fn of(ctx: *anyopaque) *WideRun {
        return @ptrCast(@alignCast(ctx));
    }

    fn begin(ctx: *anyopaque, first: bool) anyerror!void {
        const w = of(ctx);
        _ = try phaseTo(w.b, .prompt);
        // PROMPT_AHEAD_FILL (D2a): no protection from earlier calls of the prompt phase (the engine's warm-up
        // prompt's) into this pass's seeds and read-aheads.
        if (first) try w.b.promptStart();
    }

    fn after(ctx: *anyopaque, j: usize, row: u64, outs: []const mlx.mlx_array) anyerror!void {
        const w = of(ctx);
        // The head: every span's rows (all_rows), else the last span's last row when this run ends the prompt.
        for (outs, j..) |hd, sj| {
            const want: abi.Logits = if (w.f.logits == .all_rows) .all_rows else if (w.last and sj + 1 == w.n) w.f.logits else .none;
            if (want == .none) continue;
            var sub = w.f.*;
            sub.logits = want;
            sub.want_hidden = false;
            var so: abi.ForwardOut = .{};
            var hr = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&hr, hd);
            try host.finish(w.m, &sub, hr, &so);
            if (so.logits.ctx != null) try w.logit_parts.append(gpa, .{ .ctx = so.logits.ctx });
        }
        if (w.hidden) |h| {
            var r = w.row0 + row;
            for (outs) |hd| {
                try h.put(r, hd);
                r += @intCast(mlx.getShape(hd)[1]);
            }
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (w.logit_parts.items) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        if (w.hidden) |h| if (h.dest.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(vec, h.dest);
        };
        try mlx.check(mlx.mlx_eval(vec));
        try w.b.settle();
    }

    fn file(ctx: *anyopaque, pos: u64, states: []const abi.LayerState) anyerror!void {
        const w = of(ctx);
        const v = w.ops.checkpoint.?(w.cache, pos, .of(states));
        if (!v.isOk()) {
            w.refused = v;
            return error.LingCheckpointRefused;
        }
    }
};

fn takeRef(a: mlx.mlx_array) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    if (a.ctx != null) _ = mlx.mlx_array_set(&r, a);
    return r;
}

/// A streamed EXL3 pack has no small form (plugin.json's fixture_reason).
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return error.NoFixture;
}
