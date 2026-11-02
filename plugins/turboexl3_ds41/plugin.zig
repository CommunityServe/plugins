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
/// A prompt pass's routed calls timed and counted, and the CED decoder's experts probed, when
/// `MLX_SERVE_DSV41_PROMPT_IO` asks (src/prompt_io.zig, hooked by the copied model and source).
pub const prompt_io = @import("src/prompt_io.zig");
/// The CED decoder layers' experts read into their rows while the encoder runs, when `MLX_SERVE_DSV41_DEC_PREFETCH`
/// asks (src/dec_prefetch.zig, posted by the copied source, planned by the copied model).
pub const dec_prefetch = @import("src/dec_prefetch.zig");
/// The read pool's demand ranges part by part, the parts on the file's block read straight into their rows
/// (`MLX_SERVE_DSV41_DIRECT_PARTS`, src/direct_parts.zig, read by the copied source at the stream's construction).
pub const direct_parts = @import("src/direct_parts.zig");
/// The DSpark draft's Markov chain in B + 1 launches (src/markov.zig, called by the copied model's draft;
/// `MLX_SERVE_DSV41_MARKOV`).
const markov = @import("src/markov.zig");
/// Each DSpark round's expert stream traffic and the process's disk reads beside its verify time (src/round_io.zig), on
/// with the DSpark profile (`MLX_SERVE_DIAG=dspark-profile`).
const round_io = @import("src/round_io.zig");
/// The decode's expert lookahead (`MLX_SERVE_DSV41_LOOKAHEAD`, src/stream_lookahead.zig), read by the copied stream
/// (exl3_source.zig, experts.zig) through this root.
pub const stream_lookahead = @import("src/stream_lookahead.zig");
/// The decode lookahead widened by a credit of records a round past each layer call's budget, and its `[lookahead]`
/// line (`MLX_SERVE_DSV41_LOOKAHEAD_WIDE`, src/la_wide.zig), read by the copied source and stream.
pub const la_wide = @import("src/la_wide.zig");
/// mlx-serve's expert usage profile on the streamed bank (src/expert_usage_cfg.zig: `MLX_SERVE_DSV41_RESIDENT_SEED`,
/// `MLX_SERVE_DSV41_EXPERT_USAGE_DIR`, `_RECORD`, `_LABEL`), read by the copied EXL3 source through this root: the
/// prompt rows seeded at the load with each layer's most used experts, and each request's routed experts recorded.
pub const expert_usage_cfg = @import("src/expert_usage_cfg.zig");
/// The decode route recorder (`MLX_SERVE_DSV41_ROUTES`, src/route_rec.zig: mlx-serve's `expert.routes` in the copied
/// stream, a file a request and its steps' tokens), compiled into route-recording builds only (`zig build
/// -Droute-rec=true`); the copied stream reads `route_recording` through this root.
pub const route_rec = @import("src/route_rec.zig");
/// The decode's slot rows split across the layers (`MLX_SERVE_DSV41_DECODE_ROWS=split`, src/decode_rows.zig: the flat
/// bill's decode bytes by a development corpus's misses-a-row curves), read by the copied EXL3 source through this root.
pub const decode_rows = @import("src/decode_rows.zig");
pub const route_recording: bool = @import("build_options").route_rec;
/// The usage seed's rows free until a route uses them (`MLX_SERVE_DSV41_SEED_FREE`, src/seed_free.zig), read by the
/// copied EXL3 source through this root: the prompt's read-ahead and the CED decoder prefetch may take them.
pub const seed_free = @import("src/seed_free.zig");
/// The read-ahead's prediction on the fused hc kernels (`MLX_SERVE_DSV41_RA_FUSED`, src/ra_fused.zig), read by the
/// copied model.
pub const ra_fused = @import("src/ra_fused.zig");
/// The decode lookahead's next-layer scores as MLX's wide GEMV (`MLX_SERVE_DSV41_LATE_GEMV`, src/late_gemv.zig), read by
/// the copied stream.
pub const late_gemv = @import("src/late_gemv.zig");
/// A decode layer's shared expert committed as soon as its routed call returns (`MLX_SERVE_DSV41_SHARED_HOIST`,
/// src/shared_hoist.zig), read by the copied model.
pub const shared_hoist = @import("src/shared_hoist.zig");
/// The t3 route's hit wave on turboexl3's grouped banked decode (`MLX_SERVE_DSV41_T3_GROUP`, src/t3_group.zig), read by
/// src/routed_t3.zig: turboexl3's banked group and DEVROUTE's id for a miss pair.
pub const t3_group = @import("src/t3_group.zig");
/// The embedding table read from the pack per lookup (`MLX_SERVE_DSV41_EMBED_PREAD`, src/embed_pread.zig), read by the
/// copied model.
pub const embed_pread = @import("src/embed_pread.zig");
/// The vision tower read from the pack for each image (`MLX_SERVE_DSV41_VISION_LAZY`, src/vision_lazy.zig), read by the
/// copied model's load and media hook.
pub const vision_lazy = @import("src/vision_lazy.zig");
/// A wide prompt layer's EXL3 `wo_a` over runs of its attention spans' rows (`MLX_SERVE_DSV41_WOA_ROWS`, src/woa_rows.zig),
/// read by the copied model.
pub const woa_rows = @import("src/woa_rows.zig");
/// A wide prompt call's routed sum and the layer's shared expert in one combine over the call's unjoined outputs
/// (`MLX_SERVE_DSV41_JOINLESS`, src/joinless.zig), read by the copied model and the copied stream's source.
pub const joinless = @import("src/joinless.zig");
/// A wide prompt layer's joinless combine and hc expand in one kernel (`MLX_SERVE_DSV41_TAIL_FUSE`, src/tail_fuse.zig),
/// read by the copied model and launched by the copied stream's source.
pub const tail_fuse = @import("src/tail_fuse.zig");
/// What a wide prompt call's MoE does not read, committed behind its routes before the routing barrier waits on them
/// (`MLX_SERVE_DSV41_WIDE_EARLY`, src/wide_early.zig), read by the copied model.
pub const wide_early = @import("src/wide_early.zig");
/// A wide prompt layer's new streams committed at its call's end, before the layer's teardown
/// (`MLX_SERVE_DSV41_REST_EARLY`, src/rest_early.zig), read by the copied model.
pub const rest_early = @import("src/rest_early.zig");
/// A wide prompt layer's query heads kept unroped, their roped last channels apart (`MLX_SERVE_DSV41_Q_TAIL`,
/// src/q_tail.zig), read by the copied model.
pub const q_tail = @import("src/q_tail.zig");
/// A wide prompt layer's EXL3 `wo_a` run on one prepare and one finish launch for its groups, each group's W_hat decoded
/// once a layer (`MLX_SERVE_DSV41_WOA_JOIN`, src/woa_join.zig), read by the copied model.
pub const woa_join = @import("src/woa_join.zig");
/// A wide prompt layer's dense EXL3 linears' matmul and finish as one launch at this machine's tiles
/// (`MLX_SERVE_DSV41_PG2`, src/pg2.zig; turboexl3's `pgemm` switch, its warm and `check`'s compare in src/pg2_t3.zig).
pub const pg2 = @import("src/pg2.zig");
const pg2_t3 = @import("src/pg2_t3.zig");
/// A wide prompt layer's input as the last wide layer's whole stream (`MLX_SERVE_DSV41_WHOLE_STREAM`, src/whole_stream.zig),
/// read by the copied model.
pub const whole_stream = @import("src/whole_stream.zig");
/// A wide prompt call's predicted transient loads read while its attention runs, into the windows its routes take
/// (`MLX_SERVE_DSV41_STAGE`, src/stage.zig), read by the copied stream, its source and the streamer's wide call.
pub const stage = @import("src/stage.zig");
/// A wide prompt call's staged records its routes do not take let go before the drive reaches them
/// (`MLX_SERVE_DSV41_STAGE_TRIM`, src/stage_trim.zig), read by the copied stream.
pub const stage_trim = @import("src/stage_trim.zig");
/// dq1b: decode's demand reads with each missed record's gate/up range as two range jobs read side by side
/// (`MLX_SERVE_DSV41_GU_SPLIT`, src/gu_split.zig), set on the read pool by the copied source at each phase change.
pub const gu_split = @import("src/gu_split.zig");
/// The decode's demand reads by record size, part and how they were served (src/demand_split.zig, dq1a's, with
/// [round-io]); read by the copied stream at each route's settle.
pub const demand_split = @import("src/demand_split.zig");
/// dq1b's `[cold-start]` (src/cold_start.zig, with [round-io]): the decode's first calls of each layer against the
/// layer's last prompt call; read by the copied routed call.
pub const cold_start = @import("src/cold_start.zig");
/// A wide prompt layer's predicted transient loads read during the call before it, into the windows that call's groups
/// let go (`MLX_SERVE_DSV41_STAGE_NEXT`, src/stage_next.zig), read by the copied model, stream and source.
pub const stage_next = @import("src/stage_next.zig");
/// Staging's vote floor and every staged record's votes (`MLX_SERVE_DSV41_STAGE_VOTES`, src/stage_votes.zig), read by the
/// copied stream and source.
pub const stage_votes = @import("src/stage_votes.zig");
/// A wide prompt layer's streams after its attention written into one per-prompt region (`MLX_SERVE_DSV41_SCRATCH`,
/// src/scratch_x1.zig), read by the copied model.
pub const scratch_x1 = @import("src/scratch_x1.zig");
/// The Engram tables read past the page cache, through their blocks (`MLX_SERVE_DSV41_ENGRAM_DIRECT`,
/// src/engram_direct.zig), read by the copied Engram tables.
pub const engram_direct = @import("src/engram_direct.zig");
/// Engram rows kept across a conversation's turns (`MLX_SERVE_DSV41_ENGRAM_DECODE`, `MLX_SERVE_DSV41_ENGRAM_LRU_MB`,
/// src/engram_decode.zig): an LRU a prompt grid's rows, the decode reads' and the rounds' bonus rows go into, a decode
/// read's misses read on a pool; read by the copied Engram tables and model.
pub const engram_decode = @import("src/engram_decode.zig");
/// A greedy round's next draft launched at the end of the round before it (`MLX_SERVE_DSV41_DRAFT_AHEAD`,
/// src/draft_ahead.zig).
pub const draft_ahead = @import("src/draft_ahead.zig");
/// Decode's window 0 at a few rows, its memory the bank's decode slot rows (`MLX_SERVE_DSV41_DECODE_WINDOW`,
/// src/decode_window.zig), read by the copied streamer's bill and grow.
pub const decode_window = @import("src/decode_window.zig");
/// The decode phase's budget takes part of the stop under the ceiling back, MLX's cache capped at decode instead
/// (`MLX_SERVE_DSV41_DECODE_STOP`, src/decode_stop.zig), read by the copied bill and source.
pub const decode_stop = @import("src/decode_stop.zig");
/// The wide prompt layers' per-prompt regions (src/scratch.zig), read by the copied model and stream.
pub const scratch = @import("src/scratch.zig");
/// A wide prompt call's DIG-X waves write into per-prompt regions (`MLX_SERVE_DSV41_SCRATCH_WAVES`,
/// src/scratch_waves.zig on src/scratch.zig's `Region`, its twins' texts by src/wave_twins.zig), read by the copied
/// quant and source.
pub const scratch_waves = @import("src/scratch_waves.zig");
/// Both phases' slot rows over one buffer (`MLX_SERVE_DSV41_BANK_POOL`, src/bank_pool.zig: the switch and the plan),
/// read by the copied stream and source.
pub const bank_pool = @import("src/bank_pool.zig");
/// A wide prompt layer's attention spans' scores and probabilities in a ring of two per-prompt slots
/// (`MLX_SERVE_DSV41_SCRATCH_ATTN`, src/scratch_attn.zig on src/scratch.zig's `Region`, its twins' texts by
/// src/attn_twins.zig), read by the copied model.
pub const scratch_attn = @import("src/scratch_attn.zig");
/// The wide gather's pv at 64 output columns a threadgroup and its tensor-op launches three-dimensional, the same words
/// (`MLX_SERVE_DSV41_ATTN_CORE`, src/attn_core.zig), read by the copied model.
pub const attn_core = @import("src/attn_core.zig");
/// A profile build (`zig build -Dplugin-profile=true`, never served): the copied stream's profile probes compiled in (its
/// `sdk` reads this through this root) and the decode verify's GPU timeline (`MLX_SERVE_DSV41_TIMELINE`,
/// src/verify_tl.zig).
pub const plugin_profile: bool = @import("build_options").plugin_profile;
const verify_tl = @import("src/verify_tl.zig");
/// The prompt pass's GPU timeline (`MLX_SERVE_DSV41_PROMPT_TL`, src/prompt_tl.zig): the verify timeline's store over a
/// prompt, its buffers tagged by layer and host stage at `[prompt-io]`'s hooks.
const prompt_tl = @import("src/prompt_tl.zig");
const timeline = @import("mlx_stream").verify_timeline;
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
    routed_t3.enabled = config.routedT3(env.span(.MLX_SERVE_DSV41_ROUTED)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ROUTED is neither turboexl3 nor streamer (unset: turboexl3): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: the t3 math compiles the banked kernels at turboexl3's banked group when the bank opens.
    t3_group.configure(env.span(.MLX_SERVE_DSV41_T3_GROUP), routed_t3.enabled) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_T3_GROUP is set to something other than 0 (unset: on with the t3 route, the default; 0: the banked texts): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: it leaves the embedding table and the vision tower on disk unless these say 0 (the bank,
    // sized from the memory in use when it opens, then takes their bytes as slot rows).
    embed_pread.configure(env.span(.MLX_SERVE_DSV41_EMBED_PREAD)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_EMBED_PREAD is set to something other than 0 (unset: read from the pack, 0: resident): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    vision_lazy.configure(env.span(.MLX_SERVE_DSV41_VISION_LAZY)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_VISION_LAZY is set to something other than 0 (unset: read for each image, 0: resident): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    routed_t3.configureGroup() catch |e| {
        log.err("[turboexl3_ds41] turboexl3 refused its banked group {d} ({s}): the load is refused\n", .{ t3_group.bankedGroup(), @errorName(e) });
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
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
    // Before the model's load: the stream sizes its lookahead when it opens.
    expert_usage_cfg.setting = expert_usage_cfg.parse(env.span(.MLX_SERVE_DSV41_RESIDENT_SEED), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_DIR), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_RECORD), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_LABEL)) catch |e| {
        log.err("[turboexl3_ds41] the expert usage settings are refused ({s}): MLX_SERVE_DSV41_RESIDENT_SEED is usage or none, the usage dir and record absolute paths, a label only with a record\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    ra_fused.configure(env.span(.MLX_SERVE_DSV41_RA_FUSED)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_RA_FUSED is set to something other than 0 (unset: on, 0: the op chain): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    late_gemv.configure(env.span(.MLX_SERVE_DSV41_LATE_GEMV)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_LATE_GEMV is set to something other than 0 (unset: on, 0: the f32 product): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    shared_hoist.configure(env.span(.MLX_SERVE_DSV41_SHARED_HOIST)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SHARED_HOIST is set to something other than 0 (unset: on, 0: in the layer's last stretch): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    woa_rows.configure(env.span(.MLX_SERVE_DSV41_WOA_ROWS)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_WOA_ROWS is neither 0 nor check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    whole_stream.configure(env.span(.MLX_SERVE_DSV41_WHOLE_STREAM)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_WHOLE_STREAM is not 0 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    bank_pool.configure(env.span(.MLX_SERVE_DSV41_BANK_POOL)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_BANK_POOL is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // The pool's arrays are the host's aliases (HostGpu.alias, mlx-serve's additive ABI entry): a host without it serves
    // with the pool off, as before the pool.
    if (bank_pool.on() and !shims.hasAlias()) {
        log.info("[turboexl3_ds41] [bank-pool] off: the server's host table has no HostGpu.alias entry (its HostGpu {d} bytes, the entry at bytes 96-104: mlx-serve without host-alias.patch); the streamed bank's rows freed and freshly allocated at each phase change, as before the pool\n", .{shims.gpuBytes()});
        bank_pool.mode = .off;
    }
    scratch_waves.configure(env.span(.MLX_SERVE_DSV41_SCRATCH_WAVES)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SCRATCH_WAVES is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    scratch_attn.configure(env.span(.MLX_SERVE_DSV41_SCRATCH_ATTN)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SCRATCH_ATTN is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    scratch_attn.configureHeads(env.span(.MLX_SERVE_DSV41_SCRATCH_HEADS)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SCRATCH_HEADS is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    attn_core.configure(env.span(.MLX_SERVE_DSV41_ATTN_CORE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ATTN_CORE is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    stage.configure(env.span(.MLX_SERVE_DSV41_STAGE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_STAGE is not 0 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    gu_split.configure(env.span(.MLX_SERVE_DSV41_GU_SPLIT)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_GU_SPLIT is none of 0, 1, check (unset: 1): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    stage_trim.configure(env.span(.MLX_SERVE_DSV41_STAGE_TRIM)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_STAGE_TRIM is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    stage_next.configure(env.span(.MLX_SERVE_DSV41_STAGE_NEXT)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_STAGE_NEXT is none of 0, 1, only, own (unset: only): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    stage_votes.configure(env.span(.MLX_SERVE_DSV41_STAGE_VOTES)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_STAGE_VOTES is not a number from 0 to 65535 (unset: 8): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    engram_direct.configure(env.span(.MLX_SERVE_DSV41_ENGRAM_DIRECT)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ENGRAM_DIRECT is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    engram_decode.mode = engram_decode.parse(env.span(.MLX_SERVE_DSV41_ENGRAM_DECODE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ENGRAM_DECODE is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    engram_decode.lru_mb = engram_decode.parseMb(env.span(.MLX_SERVE_DSV41_ENGRAM_LRU_MB)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ENGRAM_LRU_MB is not a whole number of MiB from 1 to 16384 (unset: 256): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: the bank's bill takes the LRU's bytes beside the host's (the load line says the rows).
    bill.engram_store_bytes = engram_decode.billBytes();
    // Before the model's load: the decode budget's part of the stop (MLX_SERVE_DSV41_DECODE_STOP).
    decode_stop.configure(env.span(.MLX_SERVE_DSV41_DECODE_STOP)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DECODE_STOP is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    bill.decode_stop_cut = decode_stop.cutBytes();
    draft_ahead.configure(env.span(.MLX_SERVE_DSV41_DRAFT_AHEAD)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DRAFT_AHEAD is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: the bank's bill and the stream's grow read the window's rows.
    decode_window.configure(env.span(.MLX_SERVE_DSV41_DECODE_WINDOW)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DECODE_WINDOW is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    scratch_x1.configure(env.span(.MLX_SERVE_DSV41_SCRATCH)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SCRATCH is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    joinless.configure(env.span(.MLX_SERVE_DSV41_JOINLESS)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_JOINLESS is neither 0 nor check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    tail_fuse.configure(env.span(.MLX_SERVE_DSV41_TAIL_FUSE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_TAIL_FUSE is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // The tail kernel is the joinless combine's: unset, the setting follows JOINLESS; set without it, the load is refused.
    if (tail_fuse.on() and joinless.mode != .on) {
        if (tail_fuse.explicit) {
            log.err("[turboexl3_ds41] MLX_SERVE_DSV41_TAIL_FUSE needs MLX_SERVE_DSV41_JOINLESS on (its kernel is the joinless combine's): the load is refused\n", .{});
            out.* = Verdict.refuse(.unsupported, "turboexl3_ds41: TailFuseJoinless");
            return null;
        }
        tail_fuse.mode = .off;
    }
    wide_early.configure(env.span(.MLX_SERVE_DSV41_WIDE_EARLY)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_WIDE_EARLY is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    rest_early.configure(env.span(.MLX_SERVE_DSV41_REST_EARLY)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_REST_EARLY is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    q_tail.configure(env.span(.MLX_SERVE_DSV41_Q_TAIL)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_Q_TAIL is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    woa_join.configure(env.span(.MLX_SERVE_DSV41_WOA_JOIN)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_WOA_JOIN is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's first prompt: turboexl3's fused matmul + finish at the PG2 setting.
    pg2.configure(env.span(.MLX_SERVE_DSV41_PG2)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_PG2 is none of 0, 1, check (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    pg2_t3.configure();
    seed_free.configure(env.span(.MLX_SERVE_DSV41_SEED_FREE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_SEED_FREE is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    decode_rows.mode = decode_rows.parse(env.span(.MLX_SERVE_DSV41_DECODE_ROWS)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DECODE_ROWS is refused ({s}): flat or split, or unset (the split where it fits the pack)\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    route_rec.setting = route_rec.parse(env.span(.MLX_SERVE_DSV41_ROUTES), route_recording) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ROUTES is refused ({s}): an absolute directory, in a route-recording build (zig build -Droute-rec=true)\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    verify_tl.setting = verify_tl.parse(env.span(.MLX_SERVE_DSV41_TIMELINE), plugin_profile) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_TIMELINE is refused ({s}): an absolute directory, in a profile build (zig build -Dplugin-profile=true)\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    stream_lookahead.setting = stream_lookahead.parse(env.span(.MLX_SERVE_DSV41_LOOKAHEAD)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_LOOKAHEAD is none of 0 to 4, with an optional :0 or :1: the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    la_wide.configure(env.span(.MLX_SERVE_DSV41_LOOKAHEAD_WIDE)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_LOOKAHEAD_WIDE is not a number from 0 to 64 (unset: 10): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load, which compiles its decode stretches only when the stage profiler is off.
    stage_prof.configure(env.span(.MLX_SERVE_DSV41_STAGE_PROF));
    prompt_io.configure(env.span(.MLX_SERVE_DSV41_PROMPT_IO));
    prompt_tl.setting = prompt_tl.parse(env.span(.MLX_SERVE_DSV41_PROMPT_TL), plugin_profile) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_PROMPT_TL is refused ({s}): an absolute directory, in a profile build (zig build -Dplugin-profile=true)\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    if (prompt_tl.setting.dirPath() != null and (!prompt_io.on or verify_tl.setting.dirPath() != null)) {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_PROMPT_TL needs MLX_SERVE_DSV41_PROMPT_IO on (its stages are its hooks) and MLX_SERVE_DSV41_TIMELINE unset (the two share one store): the load is refused\n", .{});
        out.* = Verdict.refuse(.unsupported, "turboexl3_ds41: PromptTimelineSettings");
        return null;
    }
    // Before the model's load: the stream's read pool reserves the fills' tickets at its construction.
    direct_parts.configure(env.span(.MLX_SERVE_DSV41_DIRECT_PARTS)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DIRECT_PARTS is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    dec_prefetch.configure(env.span(.MLX_SERVE_DSV41_DEC_PREFETCH)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DEC_PREFETCH is neither S nor S:K (S an encoder layer from 1, K 1..32) nor 0: the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // MLX_SERVE_DSV41_DSPARK_POLICY: unset or `fixed` the fixed policy on sampled requests (src/rounds.zig `Fixed`), on
    // MLX_SERVE_DSV41_DSPARK_COSTS's tables or else the built-in ones (`rounds.DEFAULT_COSTS`, fitted on this box); `1` the
    // live policy; `0` off. A table beside a policy that reads none is refused by name, as a malformed one is.
    const policy_raw = env.span(.MLX_SERVE_DSV41_DSPARK_POLICY);
    const fixed_on = if (policy_raw) |v| std.mem.eql(u8, v, "fixed") else true;
    const costs_raw = env.span(.MLX_SERVE_DSV41_DSPARK_COSTS);
    if (!fixed_on and costs_raw != null) {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DSPARK_COSTS is the fixed policy's table, and MLX_SERVE_DSV41_DSPARK_POLICY names another (unset or =fixed reads it): the load is refused\n", .{});
        out.* = Verdict.refuse(.unsupported, "turboexl3_ds41: DsparkCostsSetting");
        return null;
    }
    const fixed: ?rounds.Fixed = if (fixed_on) rounds.Fixed.parse(costs_raw orelse rounds.DEFAULT_COSTS) catch {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_DSPARK_COSTS is not v1,v2,...,vN/d (the verify ms by rows from one to N, 2 <= N <= 16, then the draft's ms) nor such tables by context U1:t1+...+tK (the bounds ascending, the last table unbounded, at most 8): the load is refused\n", .{});
        out.* = Verdict.refuse(.unsupported, "turboexl3_ds41: DsparkCostsSetting");
        return null;
    } else null;
    var own = ctx.*;
    own.weights = null;
    const mp = ds.table.load(&own, out) orelse return null;
    open = null;
    ptl_model = if (prompt_tl.setting.dirPath() != null) model(mp) else null;
    prompt_tl.hook = if (ptl_model != null) &ptlStage else null;
    const on: rounds.Switches = .{
        .copy = config.switchOn(env.span(.MLX_SERVE_DSV41_DSPARK_COPY)),
        .policy = fixed_on or config.switchOff(policy_raw),
        .fixed = fixed,
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
    // MLX_SERVE_DSV41_PG2: the tabled tiles' kernels compiled now, not in the first prompt.
    pg2_t3.warm(model(mp).mdl.s, model(mp).mdl.act) catch |e| {
        log.err("[turboexl3_ds41] pg2's kernels did not build ({s}): the load is refused (MLX_SERVE_DSV41_PG2=0 runs the chain)\n", .{@errorName(e)});
        ds.table.unload(mp);
        out.* = Verdict.refuse(.failed, "turboexl3_ds41: pg2's kernels did not build");
        return null;
    };
    dropPending();
    if (drafter) |*d| d.deinit();
    drafter = rounds.Drafter.init(gpa, on);
    log.info("[turboexl3_ds41] DSpark copy rounds {s}, verify policy {s}\n", .{ if (on.copy) "on" else "off (MLX_SERVE_DSV41_DSPARK_COPY=0)", if (on.fixed != null) (if (costs_raw != null) "fixed on sampled requests, on MLX_SERVE_DSV41_DSPARK_COSTS's tables (the cut and the backoff by context, never the clock; a greedy request's rounds verify the whole block; MLX_SERVE_DSV41_DSPARK_POLICY=1 the live policy, =0 off)" else "fixed on sampled requests, on the built-in tables fitted on an M5 Max (the cut and the backoff by context, never the clock; a greedy request's rounds verify the whole block; MLX_SERVE_DSV41_DSPARK_COSTS other tables, MLX_SERVE_DSV41_DSPARK_POLICY=1 the live policy, =0 off)") else if (on.policy) "the live policy (MLX_SERVE_DSV41_DSPARK_POLICY=1: the measured ladder and the round's wall time)" else "off: every round verifies the whole block (MLX_SERVE_DSV41_DSPARK_POLICY=0)" });
    log.info("[turboexl3_ds41] read pool parts: {s}\n", .{if (direct_parts.on) "each demand read's parts on the file's block read straight into their rows, the rest staged and copied (MLX_SERVE_DSV41_DIRECT_PARTS=0: every part staged); [direct-parts] lines at each prompt's and request's end" else "every part staged and copied (MLX_SERVE_DSV41_DIRECT_PARTS=0)"});
    if (routed_one.on()) log.info("[turboexl3_ds41] wide prompt calls' resident routed experts as one turboexl3 call (MLX_SERVE_DSV41_ROUTED_ONE: {s}); [routed-one] lines at each request's end\n", .{@tagName(routed_one.mode)});
    if (dec_prefetch.on) log.info("[turboexl3_ds41] CED decoder prefetch on (MLX_SERVE_DSV41_DEC_PREFETCH): the decoder layers' experts predicted at encoder layer {d}, {d} layers posted per prompt call, into their free rows only\n", .{ dec_prefetch.start, dec_prefetch.per_call });
    if (seed_free.on) log.info("[turboexl3_ds41] usage seed's rows free until routed (MLX_SERVE_DSV41_SEED_FREE=0: a seeded row is a resident from the load): the prompt's read-ahead and the CED decoder prefetch may take them\n", .{}) else log.info("[turboexl3_ds41] usage seed's rows residents from the load (MLX_SERVE_DSV41_SEED_FREE=0)\n", .{});
    if (prompt_io.on) log.info("[turboexl3_ds41] prompt timeline on (MLX_SERVE_DSV41_PROMPT_IO): [prompt-io], [stream-stats prompt] and [ced-probe] lines at each prompt's end{s}\n", .{if (prompt_io.split_ra) "; the read-ahead's wait split (=2: a sync at each step)" else ""});
    log.info("[turboexl3_ds41] read-ahead prediction: {s}\n", .{if (ra_fused.on) "on the fused hc kernels (MLX_SERVE_DSV41_RA_FUSED=0: the op chain)" else "the op chain (MLX_SERVE_DSV41_RA_FUSED=0)"});
    log.info("[turboexl3_ds41] decode lookahead's next-layer scores: {s}\n", .{if (late_gemv.on) "MLX's wide GEMV in the weights' half type (MLX_SERVE_DSV41_LATE_GEMV=0: the f32 product)" else "the f32 product (MLX_SERVE_DSV41_LATE_GEMV=0)"});
    log.info("[turboexl3_ds41] decode layers' shared expert: {s}\n", .{if (shared_hoist.on) "committed behind the hit wave (MLX_SERVE_DSV41_SHARED_HOIST=0: in the layer's last stretch)" else "in the layer's last stretch (MLX_SERVE_DSV41_SHARED_HOIST=0)"});
    log.info("[turboexl3_ds41] embedding table: {s}\n", .{if (embed_pread.on) "read from the pack at each lookup, not resident (MLX_SERVE_DSV41_EMBED_PREAD=0: resident)" else "resident (MLX_SERVE_DSV41_EMBED_PREAD=0)"});
    log.info("[turboexl3_ds41] vision tower: {s}\n", .{if (vision_lazy.on) "read from the pack for each image, not resident (MLX_SERVE_DSV41_VISION_LAZY=0: resident)" else "resident (MLX_SERVE_DSV41_VISION_LAZY=0)"});
    if (!routed_t3.enabled) log.info("[turboexl3_ds41] routed experts' decode on the streamer's own EXL3 kernels (MLX_SERVE_DSV41_ROUTED=streamer; unset: turboexl3)\n", .{});
    if (routed_t3.enabled) log.info("[turboexl3_ds41] t3 hit wave: {s}\n", .{if (t3_group.on) "turboexl3's grouped banked decode, up to four row-slots of one expert a weight decode, the device route's miss pairs skipped (MLX_SERVE_DSV41_T3_GROUP=0: the banked texts)" else "the banked texts (MLX_SERVE_DSV41_T3_GROUP=0)"});
    switch (woa_rows.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' wo_a span by span (MLX_SERVE_DSV41_WOA_ROWS=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' wo_a over runs of {d} rows, one grouped linear a run (MLX_SERVE_DSV41_WOA_ROWS=0: span by span)\n", .{woa_rows.rows}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' wo_a over runs of {d} rows and span by span, compared word for word, the span-by-span result served (MLX_SERVE_DSV41_WOA_ROWS=check): a [woa-rows] check line a layer\n", .{woa_rows.rows}),
    }
    if (whole_stream.on)
        log.info("[turboexl3_ds41] wide prompt layers take the last wide layer's whole stream and mix, not their spans' concatenation (MLX_SERVE_DSV41_WHOLE_STREAM=0: the concatenation)\n", .{})
    else
        log.info("[turboexl3_ds41] wide prompt layers concatenate their spans' streams (MLX_SERVE_DSV41_WHOLE_STREAM=0)\n", .{});
    if (stage.on)
        log.info("[turboexl3_ds41] wide prompt calls' predicted transient loads staged in their windows while the attention runs (MLX_SERVE_DSV41_STAGE=0: each stream group read by its own route); [prompt-io] staged= adopted= with PROMPT_IO\n", .{})
    else
        log.info("[turboexl3_ds41] wide prompt calls' stream groups read by their own routes (MLX_SERVE_DSV41_STAGE=0)\n", .{});
    switch (gu_split.mode) {
        .off => {},
        .on => log.info("[turboexl3_ds41] decode's demand reads: each missed record's gate/up range as two range jobs read side by side, split at the part boundary nearest its middle (MLX_SERVE_DSV41_GU_SPLIT, on by default; =0: whole); [gu-split] at each request's end\n", .{}),
        .check => log.info("[turboexl3_ds41] decode's demand reads: each missed record's gate/up range as two range jobs, checked (MLX_SERVE_DSV41_GU_SPLIT=check): each split range read again whole and compared with its rows before its publish; [gu-split] checked= mismatch= at each request's end\n", .{}),
    }
    switch (stage_trim.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt calls' staged jobs posted in file order, none withdrawn (MLX_SERVE_DSV41_STAGE_TRIM=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt calls' staged jobs posted hottest first, each one its routes take none of let go unread where no read has started (MLX_SERVE_DSV41_STAGE_TRIM=0: file order, none withdrawn){s}; [stage-trim] at each request's end\n", .{if (stage.on) "" else ": no effect without MLX_SERVE_DSV41_STAGE"}),
        .check => log.info("[turboexl3_ds41] wide prompt calls' staging trimmed at their routes, checked (MLX_SERVE_DSV41_STAGE_TRIM=check): as =1, each withdrawn ticket's status read back; [stage-trim] verified= bad= at each request's end\n", .{}),
    }
    if (stage_next.on)
        log.info("[turboexl3_ds41] wide prompt layers' read-ahead and predicted transient loads posted during the call before them, into its persistent rows and the windows its groups let go ({s}); [prompt-io] early= with PROMPT_IO\n", .{if (stage.on) "MLX_SERVE_DSV41_STAGE_NEXT=0: at the layer's start" else "no effect: MLX_SERVE_DSV41_STAGE=0"})
    else
        log.info("[turboexl3_ds41] wide prompt layers' read-ahead and staging posted at the layer's start (MLX_SERVE_DSV41_STAGE_NEXT=0)\n", .{});
    if (stage_next.on) log.info("[turboexl3_ds41] {s}\n", .{if (stage_next.skip_own) "a wide prompt layer the call before it read ahead for skips its own read-ahead (MLX_SERVE_DSV41_STAGE_NEXT=own: its own too)" else "every wide prompt layer also runs its own read-ahead (MLX_SERVE_DSV41_STAGE_NEXT=own)"});
    if (stage_votes.floor > 0)
        log.info("[turboexl3_ds41] wide prompt calls stage only the predicted stream misses at least {d} rows of the prediction name, the read-ahead keeping the whole ranking ({s}MLX_SERVE_DSV41_STAGE_VOTES=0: every one); [stage-votes] lines with PROMPT_IO\n", .{ stage_votes.floor, if (stage.on) "" else "no effect: MLX_SERVE_DSV41_STAGE=0; " })
    else
        log.info("[turboexl3_ds41] wide prompt calls stage every predicted stream miss their windows hold (MLX_SERVE_DSV41_STAGE_VOTES=0); [stage-votes] lines with PROMPT_IO\n", .{});
    if (engram_direct.on) log.info("[turboexl3_ds41] a prompt's Engram rows read once a table, past the page cache through their file system blocks, on up to 64 threads (MLX_SERVE_DSV41_ENGRAM_DIRECT=0: span by span through the page cache)\n", .{}) else log.info("[turboexl3_ds41] Engram rows read span by span through the page cache (MLX_SERVE_DSV41_ENGRAM_DIRECT=0)\n", .{});
    engram_decode.start();
    switch (draft_ahead.mode) {
        .off => log.info("[turboexl3_ds41] a greedy round's draft launched at its own start (MLX_SERVE_DSV41_DRAFT_AHEAD=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] a greedy round's next draft launched at the end of the round before it (MLX_SERVE_DSV41_DRAFT_AHEAD; =0 off): from its next token once its state is committed, when the next plan is the stages' block; the next round takes it when it fits; [draft-ahead] at each request's end\n", .{}),
        .check => log.info("[turboexl3_ds41] a greedy round's next draft launched ahead, checked (MLX_SERVE_DSV41_DRAFT_AHEAD=check): the next round makes its own too, compares ids and confidences word for word and serves its own; [draft-ahead] checks= words= differ= at each request's end\n", .{}),
    }
    switch (decode_stop.mode) {
        .off => log.info("[turboexl3_ds41] decode phase's stop under the ceiling at 2000 MB, the prompt's (MLX_SERVE_DSV41_DECODE_STOP=0): MLX's cache cap unchanged at decode; [decode-stop] off\n", .{}),
        .on, .check => log.info("[turboexl3_ds41] decode phase's stop under the ceiling at {d} MB, not 2000 (MLX_SERVE_DSV41_DECODE_STOP{s}): its budget takes the {d} MB back for slot rows, MLX's cache capped at {d} MiB from the phase change to decode, the prompt's cap restored at the change back; [decode-stop] at each request's end\n", .{ (2_000_000_000 - decode_stop.cut_bytes) / 1_000_000, if (decode_stop.mode == .check) "=check, as =1" else "; =0 off", decode_stop.cut_bytes / 1_000_000, decode_stop.cache_bytes >> 20 }),
    }
    switch (decode_window.mode) {
        .off => log.info("[turboexl3_ds41] decode's window 0 at the widest decode route's rows (MLX_SERVE_DSV41_DECODE_WINDOW=0); [mem-decode] off\n", .{}),
        .on => log.info("[turboexl3_ds41] decode's window 0 at {d} rows (MLX_SERVE_DSV41_DECODE_WINDOW; =0 off): a decode plan admits its declined misses past them, the window's memory the bank's decode slot rows; the [bank-bill] load line, [mem-decode] at each request's end\n", .{decode_window.window_rows}),
        .check => log.info("[turboexl3_ds41] decode's window 0 at {d} rows, checked (MLX_SERVE_DSV41_DECODE_WINDOW=check): as =1, the plans left past the window counted ([mem-decode] overflow=, 0 by construction)\n", .{decode_window.window_rows}),
    }
    switch (engram_decode.mode) {
        .off => log.info("[turboexl3_ds41] a decode read's Engram rows read from the tables every time (MLX_SERVE_DSV41_ENGRAM_DECODE=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] Engram rows kept across a conversation's turns (MLX_SERVE_DSV41_ENGRAM_DECODE; =0 off): an LRU of {d} MiB (MLX_SERVE_DSV41_ENGRAM_LRU_MB) billed at load, {s}every row a decode read reads put in; a decode read takes the rows it holds and reads the others on a {d}-thread pool; a round's bonus token's rows read while its draft runs; [engram-decode] lines at each request's end\n", .{ engram_decode.lru_mb, if (engram_direct.on) "a prompt grid's distinct rows and " else "(no prompt rows: MLX_SERVE_DSV41_ENGRAM_DIRECT=0) ", engram_decode.pool_threads }),
        .check => log.info("[turboexl3_ds41] Engram rows kept across a conversation's turns, checked (MLX_SERVE_DSV41_ENGRAM_DECODE=check): the LRU of {d} MiB as with =1, every row of a decode read read on the pool, the held ones compared, the read ones served; [engram-decode] checked= differ= at each request's end\n", .{engram_decode.lru_mb}),
    }
    switch (scratch_x1.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' streams after the attention in a fresh array each layer (MLX_SERVE_DSV41_SCRATCH=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' streams after the attention written into one per-prompt buffer (MLX_SERVE_DSV41_SCRATCH; =0: a fresh array each layer)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' streams after the attention served fresh, and the per-prompt buffer's compared with them word for word (MLX_SERVE_DSV41_SCRATCH=check): a [scratch] check line a wide layer\n", .{}),
    }
    switch (joinless.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt calls' routed outputs joined, the routed sum and shared expert by the op chain (MLX_SERVE_DSV41_JOINLESS=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt calls' routed sum and shared expert in one combine over the waves' outputs, no joins (MLX_SERVE_DSV41_JOINLESS=0: the joins and the op chain)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt calls' routed sum and shared expert in one combine and by the op chain, compared word for word, the op chain's served (MLX_SERVE_DSV41_JOINLESS=check): a [joinless] check line a wide call\n", .{}),
    }
    switch (tail_fuse.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' tail as the chain: the combine's f32 sum, its rounding and the hc expand ({s})\n", .{if (tail_fuse.explicit) "MLX_SERVE_DSV41_TAIL_FUSE=0" else "MLX_SERVE_DSV41_JOINLESS off: the tail kernel is the combine's"}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' joinless combine and hc expand in one kernel, the new streams written once (MLX_SERVE_DSV41_TAIL_FUSE=0: the chain)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' joinless combine and hc expand as the chain, served, and in one kernel over each call's first and last rows, compared word for word (MLX_SERVE_DSV41_TAIL_FUSE=check): a [tail-fuse] check line a wide call\n", .{}),
    }
    switch (wide_early.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt calls' shared expert and attention state evaluated by the layer's last eval, after the call (MLX_SERVE_DSV41_WIDE_EARLY=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt calls commit what their MoE does not read behind their routes, before the routing barrier waits: the routes, the next layer's predicted routes, then the shared expert and the attention state (MLX_SERVE_DSV41_WIDE_EARLY=0: after the call)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt calls commit what their MoE does not read behind their routes, and the shared expert is also built the old way and served, the two compared word for word (MLX_SERVE_DSV41_WIDE_EARLY=check): a [wide-early] check line a wide call\n", .{}),
    }
    if (rest_early.on)
        log.info("[turboexl3_ds41] wide prompt layers commit their new streams at the routed call's end, before the layer's teardown: the GPU runs the tail kernel while the host frees the layer (MLX_SERVE_DSV41_REST_EARLY=0: after the teardown)\n", .{})
    else
        log.info("[turboexl3_ds41] wide prompt layers' new streams evaluated by the layer's last eval, after its teardown (MLX_SERVE_DSV41_REST_EARLY=0)\n", .{});
    switch (q_tail.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers rope the whole query heads into a copy (MLX_SERVE_DSV41_Q_TAIL=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers keep the query heads unroped and rope their last channels apart, read together by the gather's scores (MLX_SERVE_DSV41_Q_TAIL=0: the whole roped copy)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' whole roped heads served, and the tail roped apart and the two-part scores compared with them word for word (MLX_SERVE_DSV41_Q_TAIL=check): a [q-tail] check line a wide layer\n", .{}),
    }
    switch (woa_join.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' EXL3 wo_a runs as turboexl3's grouped linear (MLX_SERVE_DSV41_WOA_JOIN=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' EXL3 wo_a runs on one prepare and one finish launch for their groups, the rows read in place, each group's W_hat decoded once a layer and kept across its runs (MLX_SERVE_DSV41_WOA_JOIN=0: turboexl3's grouped linear)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' EXL3 wo_a runs as turboexl3's grouped linear, served, and joined, compared word for word (MLX_SERVE_DSV41_WOA_JOIN=check): a [woa-join] check line a wide layer\n", .{}),
    }
    switch (pg2.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' dense EXL3 linears as MLX's matmul then the finish (MLX_SERVE_DSV41_PG2=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' dense EXL3 linears' matmul and finish as one launch at this machine's tiles ({d} shapes, MLX_SERVE_DSV41_PG2; =0 off): MLX's fused NAX kernel body and loop with the finish as its epilogue where MLX runs that kernel unsplit, the chain's words; [pg2] at each request's end\n", .{pg2_t3.shapes()}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' dense EXL3 linears checked (MLX_SERVE_DSV41_PG2=check): the chain served, pg2's launch run beside every call it takes ({d} shapes) and compared word for word; [pg2] checks= words= differ= at each request's end\n", .{pg2_t3.shapes()}),
    }
    if (stage_prof.on) log.info("[turboexl3_ds41] stage profiler on (MLX_SERVE_DSV41_STAGE_PROF): every stage of a forward synced and timed, the decode stretches eager; [stage-prof] lines at each request's end\n", .{});
    log.info("[turboexl3_ds41] DSpark Markov chain: {s}\n", .{switch (markov.mode) {
        .on => "fused, B + 1 launches a draft (MLX_SERVE_DSV41_MARKOV=0: the op chain)",
        .flat => "fused, no zig-zag (MLX_SERVE_DSV41_MARKOV=flat)",
        .check => "fused and the op chain, compared every draft; the op chain's served (MLX_SERVE_DSV41_MARKOV=check); [markov] lines at each request's end",
        .off => "the op chain (MLX_SERVE_DSV41_MARKOV=0)",
    }});
    if (plugin_profile) log.info("[turboexl3_ds41] a profile build (zig build -Dplugin-profile=true, never served): the stream's profile probes compiled in, the decode verify timeline {s}\n", .{if (verify_tl.setting.dirPath() != null) "on (MLX_SERVE_DSV41_TIMELINE): VERIFY_GPU_TIMELINE and ds41-timeline-<pid>-<seq>.json at each request's end" else "off (MLX_SERVE_DSV41_TIMELINE unset)"});
    if (prompt_tl.setting.dirPath()) |d| log.info("[turboexl3_ds41] prompt GPU timeline on (MLX_SERVE_DSV41_PROMPT_TL): each prompt's command buffers tagged by layer and stage (pre, attn, moe, rest), ds41-prompt-tl-<pid>-<seq>.json in {s} at each request's end\n", .{d});
    const la = stream_lookahead.setting;
    if (!la.isDefault()) log.info("[turboexl3_ds41] decode lookahead (MLX_SERVE_DSV41_LOOKAHEAD): {d} records read ahead a layer call{s}, idle rule {d}, the stream sized for {d}\n", .{ la.budget, if (la.predicts()) "" else " (the predictor off)", la.idle_busy, la.initBudget() });
    if (la_wide.on()) log.info("[turboexl3_ds41] decode lookahead widened (MLX_SERVE_DSV41_LOOKAHEAD_WIDE): about {d} records a round read ahead past each layer call's {d}, at most {d} more a call, the stream sized for the lookahead's widest budget{s}; a [lookahead] line at each request's end with the DSpark profile\n", .{ la_wide.per_round, la.budget, la_wide.max_extra, if (la.predicts()) "" else " (no effect: the predictor off)" });
    round_io.on = model(mp).mdl.ds_prof != null;
    demand_split.on = round_io.on;
    cold_start.on = round_io.on;
    if (demand_split.on) log.info("[turboexl3_ds41] [demand-split] on (with [round-io]): the decode's demand ranges by record size, part and how they were served (clean, beside the lookahead, adopted), at each request's end\n", .{});
    if (cold_start.on) log.info("[turboexl3_ds41] [cold-start] on (with [round-io]): each layer's first three decode calls' misses against its last prompt call's routes (all, the last 8 rows, the top 6 / 12 / 24 / 48 by votes), at each request's end\n", .{});
    if (round_io.on) {
        round_io.start(ioSnap(model(mp)));
        log.info("[turboexl3_ds41] [round-io] on (the DSpark profile): each round's expert stream traffic and the process's disk reads beside its verify time, the decode's and the rest's at each request's end\n", .{});
    }
    if (direct_parts.on) if (partsSnap(model(mp))) |ps| direct_parts.start(ps);
    if (gu_split.mode != .off) if (splitSnap(model(mp))) |ss| gu_split.start(ss);
    switch (bank_pool.mode) {
        .off => log.info("[turboexl3_ds41] the streamed bank's rows freed and freshly allocated at each phase change (MLX_SERVE_DSV41_BANK_POOL=0, or a host without HostGpu.alias)\n", .{}),
        .on => log.info("[turboexl3_ds41] the streamed bank's rows over one pool (MLX_SERVE_DSV41_BANK_POOL; =0: off): the prompt's windows and the wave sets' regions, then decode's window 0 and whole ext arrays over the same bytes, each pooled array its own buffer (HostGpu.alias); a [bank-pool] plan line at load\n", .{}),
        .check => log.info("[turboexl3_ds41] the streamed bank's rows over one pool, checked (MLX_SERVE_DSV41_BANK_POOL=check): every range that changes hands filled with 0xFF at its install, a [bank-pool] check line each\n", .{}),
    }
    switch (scratch_attn.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' attention spans' scores and probabilities in fresh buffers (MLX_SERVE_DSV41_SCRATCH_ATTN=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' attention spans on a ring of two per-prompt slots (MLX_SERVE_DSV41_SCRATCH_ATTN; =0: fresh buffers): each span's scores and probabilities in its slot, the gather's launches as write-into twins of their stock texts (compiled at load)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' attention spans on the ring, checked (MLX_SERVE_DSV41_SCRATCH_ATTN=check): the fresh launches served, each span's ring scores, probabilities and heads compared word for word; a [scratch-attn] check line a wide layer\n", .{}),
    }
    if (scratch_attn.heads_mode != .off) {
        if (scratch_attn.mode != .on) log.info("[turboexl3_ds41] MLX_SERVE_DSV41_SCRATCH_HEADS needs the ring (MLX_SERVE_DSV41_SCRATCH_ATTN on, the default): every run joined\n", .{}) else log.info("[turboexl3_ds41] wide prompt layers' full WOA_ROWS runs' grouped heads written straight into run regions as [rows, og, in], each run's wo_a reading its region (MLX_SERVE_DSV41_SCRATCH_HEADS{s}){s}\n", .{ if (scratch_attn.heads_mode == .on) "; =0: the join" else "=check", if (scratch_attn.heads_mode == .check) ": the join served, each full run's region compared word for word (an h: field on the [scratch-attn] check line)" else "" });
    } else log.info("[turboexl3_ds41] wide prompt layers' WOA_ROWS runs' grouped heads joined (MLX_SERVE_DSV41_SCRATCH_HEADS=0)\n", .{});
    switch (attn_core.mode) {
        .off => log.info("[turboexl3_ds41] wide prompt layers' attention gather on the stock pv text and launches (MLX_SERVE_DSV41_ATTN_CORE=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] wide prompt layers' attention gather with pv at 64 output columns a threadgroup and its tensor-op launches three-dimensional, (32, 1, 4) threadgroups, the same words (MLX_SERVE_DSV41_ATTN_CORE; =0: the stock text and launches)\n", .{}),
        .check => log.info("[turboexl3_ds41] wide prompt layers' attention gather checked (MLX_SERVE_DSV41_ATTN_CORE=check): the stock pv text and launches served, every call's scores and pv also run at 64 columns and three-dimensional, compared word for word; [attn-core] check lines a kind every 128 calls, at its first differing word and at the model's end\n", .{}),
    }
    switch (scratch_waves.mode) {
        .off => log.info("[turboexl3_ds41] prefill waves' intermediates in fresh buffers (MLX_SERVE_DSV41_SCRATCH_WAVES=0)\n", .{}),
        .on => log.info("[turboexl3_ds41] prefill waves into per-prompt regions (MLX_SERVE_DSV41_SCRATCH_WAVES; =0: fresh buffers): their launches as write-into twins of the stock texts (compiled at load) on two sets of a wave's intermediates at {d} rows, an uncarried call's waves on them; [scratch-waves] at each request's end\n", .{scratch_waves.set_rows}),
        .check => log.info("[turboexl3_ds41] prefill waves into per-prompt regions, checked (MLX_SERVE_DSV41_SCRATCH_WAVES=check): the fresh waves served, every launch's region output compared with the fresh one; [scratch-waves] per launch (words, differ, max_abs, max_ulps, over1) at each request's end\n", .{}),
    }
    return mp;
}

/// MLX_SERVE_DSV41_LOOKAHEAD_WIDE: the lookahead's records by base / extra and top-k position since load; null without the
/// EXL3 source.
fn laHist(m: *ds.Model) ?la_wide.Hist {
    const r = m.mdl.streamed orelse return null;
    if (r.math != .source) return null;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    return src.lookaheadHist();
}

/// The stream's PARTS counters now (`direct_parts`); null without the EXL3 source.
fn partsSnap(m: *ds.Model) ?direct_parts.Snap {
    const r = m.mdl.streamed orelse return null;
    if (r.math != .source) return null;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    return direct_parts.Snap.of(src.stats());
}

/// The stream's gate/up split counters now (`gu_split`); null without the EXL3 source.
fn splitSnap(m: *ds.Model) ?gu_split.Snap {
    const r = m.mdl.streamed orelse return null;
    if (r.math != .source) return null;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    return gu_split.Snap.of(src.stats());
}

/// The stream's counters and the process's disk reads now (`round_io`).
fn ioSnap(m: *ds.Model) round_io.Snap {
    const pr = round_io.proc();
    // The Engram reads' (both modes): the LRU's hits and evictions, the tables' read calls, the first Engram layer's waits.
    var o: round_io.Snap = .{
        .phys = pr.phys,
        .pageins = pr.pageins,
        .engram_hits = engram_decode.hits.load(.monotonic),
        .engram_ios = engram_direct.calls.load(.monotonic),
        .lru_evictions = engram_decode.evictions.load(.monotonic),
        .engram_wait_ns = engram_direct.wait_ns[0].load(.monotonic),
    };
    const r = m.mdl.streamed orelse return o;
    if (r.math != .source) return o;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    const s = src.stats();
    o.misses = s.expert_cache_misses;
    o.hits = s.expert_cache_hits;
    o.demand = s.expert_bytes_read;
    o.spec = s.spec_bytes;
    o.abandoned = s.spec_abandoned_bytes;
    o.claimed = s.claimed;
    o.issued = s.spec_issued;
    o.wall_ns = s.read_wall_ns;
    o.wait_ns = s.read_wait_ns;
    o.adopt = s.adopt_bytes;
    return o;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    dropOpen(model(mp));
    dropPending();
    dropAhead();
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
    /// Recorded on the verify timeline (profile builds, `tlBegin`).
    tl: bool = false,
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
    // MLX_SERVE_DSV41_DECODE_WINDOW: MLX's peak over the request's decode, from its first round (`[mem-decode]`).
    if (decode_window.on() and !mem_decode_armed) {
        _ = mlx.mlx_reset_peak_memory();
        mem_decode_armed = true;
    }
    return .{ .st = st };
}

/// MLX_SERVE_DSV41_DECODE_WINDOW: MLX's peak reset for this request's decode (at its first round; `resumePrompt` clears it).
var mem_decode_armed = false;

/// MLX_SERVE_DSV41_DECODE_WINDOW's `[mem-decode]` snapshot: MLX's bytes now and since the decode's first round, the
/// footprint, and the stream's decode admissions.
fn memDecodeSnap(m: *ds.Model) decode_window.Snap {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    var s: decode_window.Snap = .{ .active = active, .cache = cache, .peak = peak, .footprint = decode_window.footprint(), .declined = 0, .forced = 0, .overflow = 0, .window = 0 };
    const r = m.mdl.streamed orelse return s;
    if (r.math != .source) return s;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    const d = src.stream.decodeAdmissions();
    s.declined = d.declined;
    s.forced = d.forced;
    s.overflow = d.overflow;
    s.window = d.window;
    return s;
}

const Drafted = struct { d: dsv41.DsparkDraft, round: Round };

/// The round's drafts from `t1` (`rounds.Drafter.plan`), at most what a verify carries and the round may keep: the
/// context's continuation, none, or the stages' draft cut by the policy.
fn draft(m: *ds.Model, dr: *rounds.Drafter, st: *dsv41.State, t1: u32, accept_cap: u32) !Drafted {
    if (round_io.on) round_io.begin(ioSnap(m));
    const watch = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const max = @min(dsv41.maxVerifyDrafts(m.mdl), @as(usize, accept_cap), abi.max_round_tokens - 1);
    const plan = dr.plan(st.n, t1, max, accept_cap, m.mdl.ds_block);
    if (plan == .stage) {
        // MLX_SERVE_DSV41_DRAFT_AHEAD: the draft the round before launched, when it fits.
        var d = if (try aheadTake(m, st, t1)) |a| a else blk: {
            // MLX_SERVE_DSV41_ENGRAM_DECODE: the bonus token's Engram rows read while the draft runs.
            dsv41.engramEarly(m.mdl, st, t1);
            break :blk try dsv41.dsparkDraft(m.mdl, gpa, st, t1);
        };
        errdefer d.deinit(gpa);
        d.len = dr.cut(d.confidence[0..@min(d.len, max)]);
        routesRound(m, "stage", st.n, d.ids[0 .. d.len + 1]);
        var round: Round = .{ .kind = .stage, .drafts = d.len, .watch = watch };
        @memcpy(round.conf[0..d.len], d.confidence[0..d.len]);
        return .{ .d = d, .round = round };
    }
    // MLX_SERVE_DSV41_DRAFT_AHEAD: a launched draft no stage round takes.
    dropAhead();
    const copied: []const u32 = if (plan == .copy) plan.copy else &.{};
    const ids = try gpa.alloc(u32, copied.len + 1);
    ids[0] = t1;
    @memcpy(ids[1..], copied);
    routesRound(m, @tagName(plan), st.n, ids);
    return .{ .d = .{ .ids = ids, .len = copied.len, .logits = &.{}, .confidence = &.{} }, .round = .{ .kind = plan, .drafts = copied.len, .watch = watch } };
}

/// A round's kept tokens: its phases to `[dspark-prof]`, what it teaches to the drafter, its tokens to the history.
fn kept(m: *ds.Model, round: *const Round, r: *dsv41.DsparkRound, st: *const dsv41.State) void {
    if (round.tl) tlEnd(m, round.drafts + 1, r.accepted);
    r.phases.draft_ns = if (round.kind == .stage) m.mdl.ds_draft_ns else 0;
    dsv41.dsparkObserve(m.mdl, r.phases);
    if (round_io.on) round_io.end(ioSnap(m), round.drafts + 1, r.phases.verify_ns, round.watch.read());
    routesStep(m, round.drafts + 1, round.drafts, r.accepted, r.tokens, round.watch.read(), false);
    const dr = if (drafter) |*d| d else return;
    const conf: []const f32 = if (round.kind == .stage) round.conf[0..round.drafts] else &.{};
    dr.learn(round.kind, round.drafts, conf, r.accepted, msOf(r.phases.verify_ns), msOf(round.watch.read()));
    dr.history.write(st.n - r.tokens.len, r.tokens);
}

fn msOf(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}

fn roundFailed(e: anyerror) Verdict {
    round_io.abandon();
    dropAhead();
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
    const rec = tlBegin(m);
    drafter.?.greedy = true;
    var p = draft(m, &drafter.?, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer p.d.deinit(gpa);
    p.round.tl = rec;
    if (rec) tlVerify(.begin);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &p.d) catch |e| return roundFailed(e);
    defer pending.deinit();
    const suppress: ?mlx.mlx_array = if (f.suppress_mask.ctx != null) .{ .ctx = f.suppress_mask.ctx } else null;
    var best: [abi.max_round_tokens]u32 = undefined;
    argmaxRows(m, pending.vl_g, suppress, best[0 .. p.d.len + 1]) catch |e| {
        closeVerify(m, st, &pending);
        return roundFailed(e);
    };
    if (rec) tlVerify(.end);
    var accepted: usize = 0;
    while (accepted < p.d.len and best[accepted] == p.d.ids[accepted + 1]) accepted += 1;
    var r = dsv41.dsparkFinish(m.mdl, gpa, st, &pending, accepted, best[accepted]) catch |e| return roundFailed(e);
    defer r.deinit(gpa);
    kept(m, &p.round, &r, st);
    fillRound(&out.round, r.tokens, @intCast(p.d.len + 1), st.n);
    out.round.accepted = r.accepted;
    out.round.next_token = r.next_token;
    // MLX_SERVE_DSV41_DRAFT_AHEAD: the next round's draft from this round's next token, launched before the engine's
    // turn (the cap the next round would get: this one's less the tokens it kept).
    if (draft_ahead.on()) aheadLaunch(m, st, r.next_token, f.accept_cap -| @as(u32, @intCast(r.tokens.len)));
    return .ok;
}

/// MLX_SERVE_DSV41_DRAFT_AHEAD: the launched draft the next greedy round may take (one at most).
var ahead: ?dsv41.DraftAhead = null;

/// MLX_SERVE_DSV41_DRAFT_AHEAD: the launched draft let go (its GPU work runs out), counted dropped.
fn dropAhead() void {
    if (ahead) |*a| {
        a.deinit();
        ahead = null;
        draft_ahead.dropped += 1;
    }
}

/// MLX_SERVE_DSV41_DRAFT_AHEAD: the next greedy round's draft from `t1` launched now, when that round's plan would be the
/// stages' block (`rounds.Drafter.wouldStage` at the cap `cap`); its bonus token's Engram rows read early with it.
fn aheadLaunch(m: *ds.Model, st: *dsv41.State, t1: u32, cap: u32) void {
    dropAhead();
    const dr = if (drafter) |*d| d else return;
    if (!dsv41.draftAheadOk(m.mdl)) return;
    const max = @min(dsv41.maxVerifyDrafts(m.mdl), @as(usize, cap), abi.max_round_tokens - 1);
    if (!dr.wouldStage(st.n, t1, max, cap, m.mdl.ds_block)) return;
    // MLX_SERVE_DSV41_ENGRAM_DECODE: the bonus token's Engram rows read while the draft runs.
    dsv41.engramEarly(m.mdl, st, t1);
    ahead = dsv41.dsparkDraftLaunch(m.mdl, gpa, st, t1) catch {
        draft_ahead.failed += 1;
        return;
    };
    draft_ahead.launched += 1;
}

/// MLX_SERVE_DSV41_DRAFT_AHEAD: the launched draft read, when it is the one `dsparkDraft` would make now (a greedy round,
/// the same token, position and state), else dropped; with `check` the round's own draft is made too, compared with it
/// and served. Null: no draft fits.
fn aheadTake(m: *ds.Model, st: *dsv41.State, t1: u32) !?dsv41.DsparkDraft {
    if (ahead == null) return null;
    if (!drafter.?.greedy or !dsv41.draftAheadFits(m.mdl, st, &ahead.?, t1)) {
        dropAhead();
        return null;
    }
    var a = ahead.?;
    ahead = null;
    defer a.deinit();
    var d = try dsv41.dsparkDraftCollect(m.mdl, gpa, &a);
    draft_ahead.taken += 1;
    if (draft_ahead.mode != .check) return d;
    defer d.deinit(gpa);
    const own = try dsv41.dsparkDraft(m.mdl, gpa, st, t1);
    draft_ahead.compare(d.ids, own.ids, d.confidence, own.confidence, d.len, own.len);
    return own;
}

/// The first half of a round the engine accepts under its own sampling: the verified rows' ids and logits.
fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    const st = switch (roundState(m, f, out)) {
        .st => |s| s,
        .refused => |v| return v,
    };
    const t1 = f.tokens.slice()[0];
    const rec = tlBegin(m);
    drafter.?.greedy = false;
    // MLX_SERVE_DSV41_DRAFT_AHEAD: a sampled round takes no launched draft.
    dropAhead();
    var p = draft(m, &drafter.?, st, t1, f.accept_cap) catch |e| return roundFailed(e);
    defer p.d.deinit(gpa);
    p.round.tl = rec;
    if (rec) tlVerify(.begin);
    var pending = dsv41.dsparkBeginWith(m.mdl, gpa, st, t1, &p.d) catch |e| return roundFailed(e);
    const rows = p.d.len + 1;
    const logits = rowsF32(m, pending.vl_g, rows) catch |e| {
        closeVerify(m, st, &pending);
        pending.deinit();
        return roundFailed(e);
    };
    // A recorded verify's window closes at its last eval, which the engine's sampling makes after this returns.
    if (rec) {
        _ = mlx.mlx_array_eval(logits);
        tlVerify(.end);
    }
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
    round_io.abandon();
    // MLX_SERVE_DSV41_DRAFT_AHEAD: a forward moves the state past any launched draft.
    dropAhead();
    const dr = if (drafter) |*d| d else return ds.table.forward(mp, f, out);
    settlePending(dr);
    const at: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    const on_gpu = f.ids.ctx != null and f.rows == 1 and f.logits != .all_rows;
    const in: ?v4plugin.Ids = if (on_gpu) null else v4plugin.Ids.of(f) catch null;
    defer if (in) |x| x.deinit();
    const v = ds.table.forward(mp, f, out);
    if (v.isOk()) routesStep(model(mp), f.rows, 0, 0, &.{}, 0, true);
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

/// The request's prompt and the prefix its state kept (`resumePrompt`): the usage record's token counts.
var req_prompt: usize = 0;
var req_kept: usize = 0;

/// The built-in's resume: the drafter's request begins on the prompt prefix the state kept.
fn resumePrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    // MLX_SERVE_DSV41_DECODE_WINDOW: the next request's decode peak taken from its own first round.
    mem_decode_armed = false;
    // MLX_SERVE_DSV41_DRAFT_AHEAD: a new prompt's state takes no launched draft.
    dropAhead();
    const n = ds.table.cache.resume_prompt.?(mp, slot, prompt, media_rows);
    req_prompt = prompt.len;
    req_kept = @min(@as(usize, @intCast(n)), prompt.len);
    routesBegin(model(mp));
    dropPending();
    if (drafter) |*d| d.begin(prompt.slice()[0..@min(@as(usize, @intCast(n)), prompt.len)]);
    return n;
}

var meter_buf: [262144]u8 = undefined;

/// The request's DSpark rounds beside the engine's `[spec-stats]`, which counts every native round a stage round; the
/// stage profiler's lines (src/stage_prof.zig) when it is on; the expert stream's counters since the last request, and
/// the one-call path's (src/routed_one.zig) when it is on; the Markov chain's drafts (src/markov.zig) when it is on; the
/// Engram LRU's (src/engram_decode.zig) when it is on; each round's stream traffic (src/round_io.zig) with the DSpark
/// profile.
fn meters(mp: *anyopaque, slot: u32, out: *abi.Str) callconv(.c) void {
    _ = slot;
    ptlWrite(model(mp));
    tlWrite(model(mp));
    stage_prof.requestEnd();
    usageEnd(model(mp));
    routesWrite(model(mp));
    const head = if (drafter) |*d| d.meters(&meter_buf) else meter_buf[0..0];
    const mid = streamMeters(model(mp), meter_buf[head.len..]);
    const tail = markov.meters(meter_buf[head.len + mid.len ..]);
    const n0 = head.len + mid.len + tail.len;
    const dp = if (partsSnap(model(mp))) |ps| direct_parts.meters(ps, meter_buf[n0..]) else meter_buf[n0..n0];
    const swl = scratch_waves.meters(meter_buf[n0 + dp.len ..]);
    const edl = engram_decode.meters(meter_buf[n0 + dp.len + swl.len ..]);
    // MLX_SERVE_DSV41_DRAFT_AHEAD: the request's last round's launched draft has no round after it.
    dropAhead();
    const dal = draft_ahead.meters(meter_buf[n0 + dp.len + swl.len + edl.len ..]);
    const mdl_line = if (decode_window.on()) decode_window.meters(memDecodeSnap(model(mp)), meter_buf[n0 + dp.len + swl.len + edl.len + dal.len ..]) else meter_buf[0..0];
    const dsl = decode_stop.meters(meter_buf[n0 + dp.len + swl.len + edl.len + dal.len + mdl_line.len ..]);
    const stl = stage_trim.meters(meter_buf[n0 + dp.len + swl.len + edl.len + dal.len + mdl_line.len + dsl.len ..]);
    // MLX_SERVE_DSV41_LOOKAHEAD_WIDE: the request's records read ahead by base / extra and top-k position (the DSpark profile's).
    const lal = if (round_io.on) (if (laHist(model(mp))) |h| la_wide.meters(h, meter_buf[n0 + dp.len + swl.len + edl.len + dal.len + mdl_line.len + dsl.len + stl.len ..]) else meter_buf[0..0]) else meter_buf[0..0];
    // MLX_SERVE_DSV41_PG2: the [pg2] line (the counts since load).
    const pgc = pg2_t3.counts();
    const pgl = pg2.meters(pgc[0], pgc[1], meter_buf[n0 + dp.len + swl.len + edl.len + dal.len + mdl_line.len + dsl.len + stl.len + lal.len ..]);
    const n = n0 + dp.len + swl.len + edl.len + dal.len + mdl_line.len + dsl.len + stl.len + lal.len + pgl.len;
    const io = if (round_io.on) round_io.meters(ioSnap(model(mp)), meter_buf[n..]) else meter_buf[n..n];
    // dq1b: the decode's demand ranges by class, the gate/up split's counts, the cold start's first calls.
    const dmd = demand_split.meters(meter_buf[n + io.len ..]);
    const gsl = if (gu_split.mode != .off) (if (splitSnap(model(mp))) |ss| gu_split.meters(ss, meter_buf[n + io.len + dmd.len ..]) else meter_buf[0..0]) else meter_buf[0..0];
    const csl = cold_start.meters(meter_buf[n + io.len + dmd.len + gsl.len ..]);
    out.* = abi.Str.of(meter_buf[0 .. n + io.len + dmd.len + gsl.len + csl.len]);
}

/// `MLX_SERVE_DSV41_EXPERT_USAGE_RECORD`: the request's routed experts into the record, its file written (a failed write is
/// logged; the request already ended).
fn usageEnd(m: *ds.Model) void {
    if (expert_usage_cfg.setting.recordPath() == null) return;
    const r = m.mdl.streamed orelse return;
    if (r.math != .source) return;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    const at: usize = if (m.mdl.dec_state) |*st| st.n else 0;
    src.usageEnd(std.Io.Threaded.global_single_threaded.io(), req_prompt -| req_kept, at -| req_prompt) catch |e| log.err("[turboexl3_ds41] the expert usage record was not written: {s}\n", .{@errorName(e)});
}

// ── The decode verify's GPU timeline (`MLX_SERVE_DSV41_TIMELINE`, profile builds only) ──

var tl_rounds: verify_tl.Rounds = .{};
var tl_seq: u32 = 0;
var tl_failed = false;

comptime {
    if (verify_tl.max_rounds != timeline.max_cycles) @compileError("verify_tl.max_rounds is not the timeline's max_cycles");
}

/// A round's start: the timeline armed at the request's first round, the round recorded while the request has a cycle
/// left. False (every served build, the setting unset): not recorded.
fn tlBegin(m: *ds.Model) bool {
    if (comptime !plugin_profile) return false;
    if (verify_tl.setting.dirPath() == null or tl_failed) return false;
    if (!timeline.active) {
        timeline.install(m.mdl.s, timeline.max_cycles + 1, @intCast(m.mdl.n_layers)) catch |e| {
            log.err("[turboexl3_ds41] the verify timeline is off: {s}\n", .{@errorName(e)});
            tl_failed = true;
            return false;
        };
        tl_rounds.reset();
    }
    if (timeline.cycles >= timeline.max_cycles) return false;
    timeline.cycleBegin();
    tl_rounds.begin(timeline.cycles - 1, timeline.now(), tlCounters(m));
    return true;
}

/// A recorded round's verify: its forward's start, and the read of its last eval.
fn tlVerify(comptime edge: enum { begin, end }) void {
    if (comptime !plugin_profile) return;
    switch (edge) {
        .begin => {
            timeline.verifyBegin();
            tl_rounds.verifyBegin(timeline.now());
        },
        .end => {
            timeline.verifyEnd();
            tl_rounds.verifyEnd(timeline.now());
        },
    }
}

/// A recorded round's commit: its rows, the drafts kept and the stream's counters.
fn tlEnd(m: *ds.Model, rows: usize, accepted: usize) void {
    if (comptime !plugin_profile) return;
    tl_rounds.end(timeline.now(), tlCounters(m), @intCast(rows), @intCast(accepted));
}

fn tlCounters(m: *ds.Model) verify_tl.Counters {
    const r = m.mdl.streamed orelse return .{};
    if (r.math != .source) return .{};
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    const s = src.stats();
    return .{ .misses = s.expert_cache_misses, .hits = s.expert_cache_hits, .demand = s.expert_bytes_read };
}

/// The request's end: the recorded buffers settled, `VERIFY_GPU_TIMELINE` logged and the request's file written, the
/// timeline disarmed until the next request's first round.
fn tlWrite(m: *ds.Model) void {
    if (comptime !plugin_profile) return;
    // The store is the prompt timeline's when it is on (`ptlWrite`).
    if (verify_tl.setting.dirPath() == null or !timeline.active) return;
    const pending = timeline.settle(m.mdl.s, 2000);
    defer timeline.uninstall();
    var lb: [16384]u8 = undefined;
    log.info("[turboexl3_ds41] {s}\n", .{timeline.line(&lb, pending)});
    const dir = verify_tl.setting.dirPath() orelse return;
    var pb: [verify_tl.max_path + 64]u8 = undefined;
    const path = verify_tl.filePath(&pb, dir, @intCast(std.c.getpid()), tl_seq) catch return;
    tl_seq += 1;
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    tlJson(&aw.writer) catch |e| return log.err("[turboexl3_ds41] the verify timeline's file was not made: {s}\n", .{@errorName(e)});
    std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{ .sub_path = path, .data = aw.written() }) catch |e|
        return log.err("[turboexl3_ds41] the verify timeline was not written to {s}: {s}\n", .{ path, @errorName(e) });
    log.info("[turboexl3_ds41] verify timeline written: {s} ({d} rounds)\n", .{ path, tl_rounds.slice().len });
}

fn tlJson(w: *std.Io.Writer) !void {
    try w.writeAll("{\"timeline\": ");
    try timeline.writeJson(w);
    try w.writeAll(", ");
    try verify_tl.writeRounds(w, tl_rounds.slice());
    try w.writeAll("}\n");
}

// ── The prompt pass's GPU timeline (`MLX_SERVE_DSV41_PROMPT_TL`, profile builds only) ──

/// The loaded model while the prompt timeline is on (its stream and layers), else null.
var ptl_model: ?*ds.Model = null;
var ptl_marks: prompt_tl.Marks = .{};
var ptl_seq: u32 = 0;
var ptl_failed = false;

/// `prompt_tl.hook`: layer `layer`'s stage `st` starts now. A prompt's first stage installs the timeline (a failed install
/// turns it off for the process); `end` stops recording until the request's end writes the file.
fn ptlStage(layer: u32, st: prompt_tl.Stage) void {
    if (comptime !plugin_profile) return;
    const m = ptl_model orelse return;
    if (ptl_failed) return;
    if (!timeline.active) {
        if (st == .end) return;
        timeline.install(m.mdl.s, 2, @intCast(m.mdl.n_layers)) catch |e| {
            log.err("[turboexl3_ds41] the prompt timeline is off: {s}\n", .{@errorName(e)});
            ptl_failed = true;
            return;
        };
        ptl_marks.reset();
    }
    ptl_marks.add(timeline.now(), layer, st);
    timeline.promptStage(layer, @intFromEnum(st));
}

/// The request's end: the prompt's buffers settled, its file written, the timeline uninstalled until the next prompt.
fn ptlWrite(m: *ds.Model) void {
    if (comptime !plugin_profile) return;
    const dir = prompt_tl.setting.dirPath() orelse return;
    if (!timeline.active) return;
    const pending = timeline.settle(m.mdl.s, 2000);
    defer timeline.uninstall();
    var pb: [prompt_tl.max_path + 64]u8 = undefined;
    const path = prompt_tl.filePath(&pb, dir, @intCast(std.c.getpid()), ptl_seq) catch return;
    ptl_seq += 1;
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    ptlJson(&aw.writer) catch |e| return log.err("[turboexl3_ds41] the prompt timeline's file was not made: {s}\n", .{@errorName(e)});
    std.Io.Dir.cwd().writeFile(std.Io.Threaded.global_single_threaded.io(), .{ .sub_path = path, .data = aw.written() }) catch |e|
        return log.err("[turboexl3_ds41] the prompt timeline was not written to {s}: {s}\n", .{ path, @errorName(e) });
    const k = timeline.counts();
    log.info("[turboexl3_ds41] prompt timeline written: {s} ({d} buffers, {d} pending, {d} dropped, {d} marks, {d} marks dropped)\n", .{ path, @min(k[0], timeline.max_buffers), pending, k[2], ptl_marks.n, ptl_marks.dropped });
}

fn ptlJson(w: *std.Io.Writer) !void {
    try w.writeAll("{\"timeline\": ");
    try timeline.writeJson(w);
    try w.writeAll(", ");
    try prompt_tl.writeMarks(w, ptl_marks.slice(), ptl_marks.dropped, timeline.origin());
    try w.writeAll("}\n");
}

// ── The decode route recorder (`MLX_SERVE_DSV41_ROUTES`, route-recording builds only) ──

var route_log: route_rec.Log = .{};
var route_seq: u32 = 0;

/// The armed recorder's source; null in every other build or setting.
fn routesSource(m: *ds.Model) ?*Exl3Source {
    if (comptime !route_recording) return null;
    if (route_rec.setting.dirPath() == null) return null;
    const r = m.mdl.streamed orelse return null;
    if (r.math != .source) return null;
    const src: *Exl3Source = @ptrCast(@alignCast(r.ctx));
    return if (src.routesOn()) src else null;
}

fn routesBegin(m: *ds.Model) void {
    const src = routesSource(m) orelse return;
    src.routesBegin(@intCast(@min(req_prompt -| req_kept, std.math.maxInt(u32))));
    route_log.begin(gpa, req_prompt, req_kept) catch |e| log.err("[turboexl3_ds41] the route log lost a line: {s}\n", .{@errorName(e)});
}

fn routesRound(m: *ds.Model, kind: []const u8, pos: usize, rows: []const u32) void {
    if (routesSource(m) == null) return;
    route_log.round(kind, pos, rows);
}

/// A step's end: a round's (`only_routed` false) or a plain forward's that routed in the decode.
fn routesStep(m: *ds.Model, rows: usize, drafted: usize, accepted: usize, emitted: []const u32, wall_ns: u64, only_routed: bool) void {
    const src = routesSource(m) orelse return;
    if (!src.routesStep(@intCast(rows), @intCast(drafted), @intCast(accepted), @intCast(emitted.len), wall_ns, only_routed)) return;
    const pos: usize = if (m.mdl.dec_state) |*st| st.n else 0;
    route_log.step(gpa, pos, accepted, emitted) catch |e| log.err("[turboexl3_ds41] the route log lost a line: {s}\n", .{@errorName(e)});
}

/// The request's end: the recorder's file and the steps' log, `ds41-routes-<pid>-<seq>` in the setting's directory.
fn routesWrite(m: *ds.Model) void {
    const src = routesSource(m) orelse return;
    const dir = route_rec.setting.dirPath().?;
    const io = std.Io.Threaded.global_single_threaded.io();
    const pid: i64 = @intCast(std.c.getpid());
    var pb: [route_rec.max_path + 64]u8 = undefined;
    const bin = route_rec.filePath(&pb, dir, pid, route_seq, ".bin") catch return;
    const wrote = src.routesWrite(io, bin) catch |e| {
        log.err("[turboexl3_ds41] the decode routes were not written to {s}: {s}\n", .{ bin, @errorName(e) });
        return;
    };
    if (!wrote) return;
    log.info("[turboexl3_ds41] decode routes written: {s} ({d} steps)\n", .{ bin, route_log.steps });
    var tb: [route_rec.max_path + 64]u8 = undefined;
    const txt = route_rec.filePath(&tb, dir, pid, route_seq, ".steps.txt") catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = txt, .data = route_log.buf.items }) catch |e|
        log.err("[turboexl3_ds41] the route log was not written to {s}: {s}\n", .{ txt, @errorName(e) });
    route_seq += 1;
}

const Exl3Source = @import("src/mlx_serve/sdk/weights/streamed.zig").Exl3.Source;
const StreamStats = @TypeOf(@as(*Exl3Source, undefined).stats());
/// The stream's counters at the last request's end (they count for the process).
var stream_prev: StreamStats = .{};

/// `[stream-stats]`: the streamed bank's routes, resident hits and misses (unique experts a route), evictions, bytes
/// read and the host's read waits since the last request (the first request's include the load's); then `[routed-one]`
/// (`routed_one.meters`) and `[routed-t3]` (the turboexl3 route's banked calls and miss parts, `routed_t3`).
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
    const n1 = n + routed_one.meters(&src.one, buf[n..]).len;
    return buf[0 .. n1 + src.t3_state.meters(buf[n1..]).len];
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
