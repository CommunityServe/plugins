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
/// The routed experts' decode on turboexl3, run by the copied streamer (src/routed_t3.zig).
pub const routed_t3 = @import("src/routed_t3.zig");
/// A replaying prompt chunk's last decoder layer routes its last row only (src/l39_last.zig:
/// armed and served by the copied model, its one-row call forced through the copied experts' wide lane).
pub const l39_last = @import("src/l39_last.zig");
/// The DSpark draft's Markov chain in B + 1 launches (src/markov.zig, called by the copied model's draft).
const markov = @import("src/markov.zig");
/// The decode lookahead widened by a credit of records a round past each layer call's budget (src/la_wide.zig), read
/// by the copied source and stream.
pub const la_wide = @import("src/la_wide.zig");
/// mlx-serve's expert usage profile on the streamed bank (src/expert_usage_cfg.zig: `MLX_SERVE_DSV41_RESIDENT_SEED`,
/// `MLX_SERVE_DSV41_EXPERT_USAGE_DIR`, `_RECORD`, `_LABEL`), read by the copied EXL3 source through this root: the
/// prompt rows seeded at the load with each layer's most used experts, and each request's routed experts recorded.
pub const expert_usage_cfg = @import("src/expert_usage_cfg.zig");
/// The decode's slot rows split across the layers (src/decode_rows.zig: the flat bill's decode bytes by a development
/// corpus's misses-a-row curves), read by the copied EXL3 source through this root.
pub const decode_rows = @import("src/decode_rows.zig");
/// A wide prompt layer's EXL3 `wo_a` over runs of its attention spans' rows (src/woa_rows.zig), read by the copied
/// model.
pub const woa_rows = @import("src/woa_rows.zig");
/// A wide prompt layer's joinless combine and hc expand in one kernel, and its write-into twin over the x1 region
/// (src/tail_fuse.zig), read by the copied model and launched by the copied stream's source.
pub const tail_fuse = @import("src/tail_fuse.zig");
/// What a wide prompt call's MoE does not read, committed behind its routes before the routing barrier waits on them
/// (src/wide_early.zig), read by the copied model.
pub const wide_early = @import("src/wide_early.zig");
/// A wide prompt layer's attention projections' large outputs kept past its teardown, their buffers to MLX's cache at the
/// next wide layer's start (src/transient_keep.zig), read by the copied model.
pub const transient_keep = @import("src/transient_keep.zig");
/// MLX's command-buffer split raised for prompts through the host's `HostGpu.command_buffer_limits`
/// (src/buffer_mb.zig), read by the copied streamed source at its phase changes.
pub const buffer_mb = @import("src/buffer_mb.zig");
/// The streamed bank's decode bill with a decode phase's runtime floor (src/decode_floor.zig): the generated bill's
/// `decode_floor` hook.
pub const decode_floor = @import("src/decode_floor.zig");
/// The streamed bank's bill calibrated against the box's measured peaks (`MLX_SERVE_DSV41_BILL_CAL`, src/bill_cal.zig):
/// the generated bill's `cal` hook.
pub const bill_cal = @import("src/bill_cal.zig");
/// A wide prompt layer's EXL3 `wo_a` run on one prepare and one finish launch for its groups, each group's W_hat decoded
/// once a layer (src/woa_join.zig), read by the copied model.
pub const woa_join = @import("src/woa_join.zig");
/// A wide prompt layer's dense EXL3 linears' matmul and finish as one launch at this machine's tiles (pg2: turboexl3's
/// `pgemm` switch set on and its warm, src/pg2_t3.zig).
const pg2_t3 = @import("src/pg2_t3.zig");
/// A wide prompt call's predicted transient loads read while its attention runs, into the windows its routes take
/// (src/stage.zig), read by the copied stream, its source and the streamer's wide call.
pub const stage = @import("src/stage.zig");
/// A wide prompt call's staged records its routes do not take let go before the drive reaches them
/// (src/stage_trim.zig), read by the copied stream.
pub const stage_trim = @import("src/stage_trim.zig");
/// Staging's vote floor and every staged record's votes (src/stage_votes.zig), read by the copied stream and source.
pub const stage_votes = @import("src/stage_votes.zig");
/// A wide prompt layer's streams after its attention written into one per-prompt region (src/scratch_x1.zig), read by
/// the copied model.
pub const scratch_x1 = @import("src/scratch_x1.zig");
/// The Engram tables read past the page cache, through their blocks (src/engram_direct.zig), read by the copied Engram
/// tables.
pub const engram_direct = @import("src/engram_direct.zig");
/// Engram rows kept across a conversation's turns (`MLX_SERVE_DSV41_ENGRAM_LRU_MB`, src/engram_decode.zig): an LRU a
/// prompt grid's rows, the decode reads' and the rounds' bonus rows go into, a decode read's misses read on a pool; read
/// by the copied Engram tables and model.
pub const engram_decode = @import("src/engram_decode.zig");
/// Decode's window 0 at a few rows, its memory the bank's decode slot rows (src/decode_window.zig), read by the copied
/// streamer's bill and grow.
pub const decode_window = @import("src/decode_window.zig");
/// The decode phase's budget takes part of the stop under the ceiling back, MLX's cache capped at decode instead
/// (src/decode_stop.zig), read by the copied bill and source.
pub const decode_stop = @import("src/decode_stop.zig");
/// The wide prompt layers' per-prompt regions (src/scratch.zig), read by the copied model and stream.
pub const scratch = @import("src/scratch.zig");
/// A wide prompt call's DIG-X waves write into per-prompt regions (src/scratch_waves.zig on src/scratch.zig's `Region`,
/// its twins' texts by src/wave_twins.zig), read by the copied
/// quant and source.
pub const scratch_waves = @import("src/scratch_waves.zig");
/// A wide prompt layer's hyper-connection stages with fewer passes over the stream (src/hc_fuse.zig,
/// kernels/hc_mix_norm.metal and kernels/hc_expand_norm.metal), read by the copied model.
pub const hc_fuse = @import("src/hc_fuse.zig");
/// The wide router's logits as one launch of MLX's fused NAX GEMM body at this machine's tile
/// (src/gate_gemm.zig, read by the copied model's `gateLogits`).
pub const gate_gemm = @import("src/gate_gemm.zig");
/// Routed layers at a half-integer EXL3 rate (exllamav3 1.6's K 1.5 and 2.5) on turboexl3, the rest on their lanes as
/// before (`MLX_SERVE_DSV41_FRAC`, src/frac.zig, read by the copied bank, quant and the t3 math).
pub const frac = @import("src/frac.zig");
/// Both phases' slot rows over one buffer (src/bank_pool.zig: the plan, and the pool off on a host without
/// `HostGpu.alias`), read by the copied stream and source.
pub const bank_pool = @import("src/bank_pool.zig");
/// A wide prompt layer's attention spans' scores and probabilities in a ring of two per-prompt slots and a full run's
/// heads in a run region (src/scratch_attn.zig on src/scratch.zig's `Region`, its twins' texts by src/attn_twins.zig),
/// read by the copied model.
pub const scratch_attn = @import("src/scratch_attn.zig");
/// The wide gather's pv at 64 output columns a threadgroup and its tensor-op launches three-dimensional, the same words
/// (src/attn_core.zig), read by the copied model.
pub const attn_core = @import("src/attn_core.zig");
/// A wide prompt layer's attention spans as one flash kernel, not exact (`MLX_SERVE_DSV41_FLASH_ATTN`, src/flash_attn.zig,
/// kernels/attn_flash.metal), read by the copied model.
pub const flash_attn = @import("src/flash_attn.zig");
const ds = @import("src/mlx_serve/plugins/deepseek_v41/plugin.zig");
const dsv41 = @import("src/mlx_serve/plugins/deepseek_v41/deepseek_v41.zig");
const v4plugin = @import("src/mlx_serve/plugins/deepseek_v4/plugin.zig");
const rounds = @import("src/rounds.zig");
const bill = @import("src/mlx_serve/plugins/deepseek_v41/bill.zig");
/// MLX_SERVE_DSV41_BILL_CAL: src/bill_cal.zig's terms as the generated bill's `CalTerms`.
fn calTerms(context: u64) bill.CalTerms {
    const t = bill_cal.terms(context);
    return .{ .margin = t.margin, .wp = t.wp, .wd = t.wd };
}
extern "c" fn _NSGetArgc() *c_int;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*anyopaque, newlen: usize) c_int;
extern "c" fn _NSGetArgv() *[*][*:0]u8;
var cal_chip_buf: [128]u8 = undefined;
/// MLX_SERVE_DSV41_BILL_CAL's conditions as this server has them: the server's `--prefill-chunk` (its command line; null
/// when not given), MLX_SERVE_CACHE_LIMIT in bytes (null when unset or not a whole number), hw.memsize, the chip
/// (machdep.cpu.brand_string) and iogpu.wired_limit_mb (null where a reading fails).
fn calConditions() bill_cal.Conditions {
    var c: bill_cal.Conditions = .{ .prefill_chunk = null, .cache_limit = null, .memsize = null, .chip = null, .wired_limit_mb = null };
    const argc: usize = @intCast(_NSGetArgc().*);
    const argv = _NSGetArgv().*;
    var i: usize = 1;
    while (i + 1 < argc) : (i += 1) {
        if (std.mem.eql(u8, std.mem.span(argv[i]), "--prefill-chunk")) c.prefill_chunk = std.fmt.parseInt(u64, std.mem.span(argv[i + 1]), 10) catch null;
    }
    if (env.span(.MLX_SERVE_CACHE_LIMIT)) |v| c.cache_limit = std.fmt.parseInt(u64, std.mem.trim(u8, v, " "), 10) catch null;
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", &mem, &len, null, 0) == 0) c.memsize = mem;
    len = cal_chip_buf.len;
    if (sysctlbyname("machdep.cpu.brand_string", &cal_chip_buf, &len, null, 0) == 0 and len > 0) c.chip = std.mem.sliceTo(cal_chip_buf[0..len], 0);
    var wl: u64 = 0;
    len = @sizeOf(u64);
    if (sysctlbyname("iogpu.wired_limit_mb", &wl, &len, null, 0) == 0) c.wired_limit_mb = wl;
    return c;
}
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
    bill.decode_floor = decode_floor.floor_bytes;
    // MLX_SERVE_DSV41_BILL_CAL: the calibrated budget the bill reads at the bank's open (=0: the bill as before).
    bill_cal.configure(env.span(.MLX_SERVE_DSV41_BILL_CAL), calConditions()) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_BILL_CAL is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    bill.cal = if (bill_cal.on()) &calTerms else null;
    // MLX's command-buffer split raised for prompts through the host's HostGpu.command_buffer_limits (none on a host
    // without the entry); a user's MLX_MAX_MB_PER_BUFFER wins.
    buffer_mb.configure(env.span(.MLX_MAX_MB_PER_BUFFER), kit.host.Host.of(ctx.host) catch null);
    // Before the model's load: the bank's warm-up builds the banked kernels at this configuration.
    routed_t3.configure() catch |e| {
        log.err("[turboexl3_ds41] turboexl3 refused the banked decode's configuration ({s}): the load is refused\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load (the bank's claim and open read it): the half-integer-rate layers.
    frac.configure(env.span(.MLX_SERVE_DSV41_FRAC)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_FRAC is set to something other than 0 or 1 (unset: on; {s}): the load is refused\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: the stream sizes its lookahead when it opens.
    expert_usage_cfg.setting = expert_usage_cfg.parse(env.span(.MLX_SERVE_DSV41_RESIDENT_SEED), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_DIR), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_RECORD), env.span(.MLX_SERVE_DSV41_EXPERT_USAGE_LABEL)) catch |e| {
        log.err("[turboexl3_ds41] the expert usage settings are refused ({s}): MLX_SERVE_DSV41_RESIDENT_SEED is usage or none, the usage dir and record absolute paths, a label only with a record\n", .{@errorName(e)});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // The pool's arrays are the host's aliases (HostGpu.alias, mlx-serve's additive ABI entry): a host without it serves
    // with the pool off.
    if (!shims.hasAlias()) {
        log.info("[turboexl3_ds41] [bank-pool] off: the server's host table has no HostGpu.alias entry (its HostGpu {d} bytes, the entry at bytes 96-104: mlx-serve without host-alias.patch); the streamed bank's rows freed and freshly allocated at each phase change\n", .{shims.gpuBytes()});
        bank_pool.on = false;
    }
    engram_decode.lru_mb = engram_decode.parseMb(env.span(.MLX_SERVE_DSV41_ENGRAM_LRU_MB)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_ENGRAM_LRU_MB is not a whole number of MiB from 1 to 16384 (unset: 256): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    // Before the model's load: the bank's bill takes the LRU's bytes beside the host's (the load line says the rows).
    bill.engram_store_bytes = engram_decode.billBytes();
    bill.decode_stop_cut = decode_stop.cut_bytes;
    flash_attn.configure(env.span(.MLX_SERVE_DSV41_FLASH_ATTN)) catch |e| {
        log.err("[turboexl3_ds41] MLX_SERVE_DSV41_FLASH_ATTN is neither 0 nor 1 (unset: on): the load is refused\n", .{});
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
        return null;
    };
    scratch_attn.ring_regions = flash_attn.mode != .on;
    // Before the model's first prompt: turboexl3's fused matmul + finish (pg2) on.
    pg2_t3.configure();
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
    const on: rounds.Switches = .{
        .copy = config.switchOn(env.span(.MLX_SERVE_DSV41_DSPARK_COPY)),
        .policy = fixed_on or config.switchOff(policy_raw),
        .fixed = fixed,
        // aw1 (MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY): the fixed policy on greedy requests too, when set (=1; aw1s: off by default).
        .fixed_greedy = config.switchOff(env.span(.MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY)),
    };
    // The policy reads each draft's confidence, as the engine has its built-in model's stages do.
    model(mp).mdl.ds_want_conf = on.policy;
    // The fused Markov chain's kernels built now, not in the first draft.
    if (model(mp).mdl.dspark) |*w| {
        const mm = model(mp).mdl;
        markov.warm(mm.s, w.markov_embed, w.markov_head, w.conf, mm.act, mm.ds_block, mm.cfg.hidden_size, mm.ds_want_conf or std.math.isFinite(mm.ds_conf_thr)) catch |e| {
            log.err("[turboexl3_ds41] the fused Markov chain's kernels did not build ({s}): the load is refused\n", .{@errorName(e)});
            ds.table.unload(mp);
            out.* = Verdict.refuse(.failed, "turboexl3_ds41: the fused Markov chain's kernels did not build");
            return null;
        };
    }
    // pg2: the tabled tiles' kernels compiled now, not in the first prompt.
    pg2_t3.warm(model(mp).mdl.s, model(mp).mdl.act) catch |e| {
        log.err("[turboexl3_ds41] pg2's kernels did not build ({s}): the load is refused\n", .{@errorName(e)});
        ds.table.unload(mp);
        out.* = Verdict.refuse(.failed, "turboexl3_ds41: pg2's kernels did not build");
        return null;
    };
    dropPending();
    if (drafter) |*d| d.deinit();
    drafter = rounds.Drafter.init(gpa, on);
    if (on.fixed != null) log.info("[turboexl3_ds41] DSpark fixed policy on greedy requests: {s}\n", .{if (on.fixed_greedy) "on (MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY=1: aw1's cut and backoff on greedy rounds too, on lad2's tables)" else "off, the default: a greedy request's rounds verify the whole block (MLX_SERVE_DSV41_DSPARK_FIXED_GREEDY=1 cuts them)"});
    log.info("[turboexl3_ds41] DSpark copy rounds {s}, verify policy {s}\n", .{ if (on.copy) "on" else "off (MLX_SERVE_DSV41_DSPARK_COPY=0)", if (on.fixed != null) (if (costs_raw != null) "fixed on sampled requests, on MLX_SERVE_DSV41_DSPARK_COSTS's tables (the cut and the backoff by context, never the clock; greedy requests: the next line; MLX_SERVE_DSV41_DSPARK_POLICY=1 the live policy, =0 off)" else "fixed on sampled requests, on the built-in tables fitted on an M5 Max (the cut and the backoff by context, never the clock; greedy requests: the next line; MLX_SERVE_DSV41_DSPARK_COSTS other tables, MLX_SERVE_DSV41_DSPARK_POLICY=1 the live policy, =0 off)") else if (on.policy) "the live policy (MLX_SERVE_DSV41_DSPARK_POLICY=1: the measured ladder and the round's wall time)" else "off: every round verifies the whole block (MLX_SERVE_DSV41_DSPARK_POLICY=0)" });
    {
        var bc_buf: [512]u8 = undefined;
        log.info("{s}", .{bill_cal.loadLine(&bc_buf)});
    }
    return mp;
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
    // MLX's peak counted from the request's first round (the server's `/props` `peak_bytes` then reads the decode's).
    if (!decode_peak_armed) {
        _ = mlx.mlx_reset_peak_memory();
        decode_peak_armed = true;
    }
    return .{ .st = st };
}

/// MLX's peak reset for this request's decode (at its first round; `resumePrompt` clears it).
var decode_peak_armed = false;

const Drafted = struct { d: dsv41.DsparkDraft, round: Round };

/// The round's drafts from `t1` (`rounds.Drafter.plan`), at most what a verify carries and the round may keep: the
/// context's continuation, none, or the stages' draft cut by the policy.
fn draft(m: *ds.Model, dr: *rounds.Drafter, st: *dsv41.State, t1: u32, accept_cap: u32) !Drafted {
    const watch = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const max = @min(dsv41.maxVerifyDrafts(m.mdl), @as(usize, accept_cap), abi.max_round_tokens - 1);
    const plan = dr.plan(st.n, t1, max, accept_cap, m.mdl.ds_block);
    if (plan == .stage) {
        // the draft the round before launched, when it fits.
        var d = if (try aheadTake(m, st, t1)) |a| a else blk: {
            // turboexl3_ds41 (src/engram_decode.zig): the bonus token's Engram rows read while the draft runs.
            dsv41.engramEarly(m.mdl, st, t1);
            break :blk try dsv41.dsparkDraft(m.mdl, gpa, st, t1);
        };
        errdefer d.deinit(gpa);
        d.len = dr.cut(d.confidence[0..@min(d.len, max)]);
        var round: Round = .{ .kind = .stage, .drafts = d.len, .watch = watch };
        const nc = @min(d.len, d.confidence.len);
        @memcpy(round.conf[0..nc], d.confidence[0..nc]);
        return .{ .d = d, .round = round };
    }
    // a launched draft no stage round takes.
    dropAhead();
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
    drafter.?.greedy = true;
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
    // The next round's draft from this round's next token, launched before the engine's turn (the cap the next round
    // would get: this one's less the tokens it kept).
    aheadLaunch(m, st, r.next_token, f.accept_cap -| @as(u32, @intCast(r.tokens.len)));
    return .ok;
}

/// The launched draft the next greedy round may take (one at most; src/mlx_serve's `DraftAhead`).
var ahead: ?dsv41.DraftAhead = null;

/// The launched draft let go (its GPU work runs out).
fn dropAhead() void {
    if (ahead) |*a| {
        a.deinit();
        ahead = null;
    }
}

/// The next greedy round's draft from `t1` launched now, when that round's plan would be the
/// stages' block (`rounds.Drafter.wouldStage` at the cap `cap`); its bonus token's Engram rows read early with it.
fn aheadLaunch(m: *ds.Model, st: *dsv41.State, t1: u32, cap: u32) void {
    dropAhead();
    const dr = if (drafter) |*d| d else return;
    if (!dsv41.draftAheadOk(m.mdl)) return;
    const max = @min(dsv41.maxVerifyDrafts(m.mdl), @as(usize, cap), abi.max_round_tokens - 1);
    if (!dr.wouldStage(st.n, t1, max, cap, m.mdl.ds_block)) return;
    // turboexl3_ds41 (src/engram_decode.zig): the bonus token's Engram rows read while the draft runs.
    dsv41.engramEarly(m.mdl, st, t1);
    ahead = dsv41.dsparkDraftLaunch(m.mdl, gpa, st, t1) catch return;
}

/// The launched draft read, when it is the one `dsparkDraft` would make now (a greedy round, the same token, position and
/// state), else dropped. Null: no draft fits.
fn aheadTake(m: *ds.Model, st: *dsv41.State, t1: u32) !?dsv41.DsparkDraft {
    if (ahead == null) return null;
    if (!drafter.?.greedy or !dsv41.draftAheadFits(m.mdl, st, &ahead.?, t1)) {
        dropAhead();
        return null;
    }
    var a = ahead.?;
    ahead = null;
    defer a.deinit();
    return try dsv41.dsparkDraftCollect(m.mdl, gpa, &a);
}

/// The first half of a round the engine accepts under its own sampling: the verified rows' ids and logits.
fn draftVerify(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = model(mp);
    const st = switch (roundState(m, f, out)) {
        .st => |s| s,
        .refused => |v| return v,
    };
    const t1 = f.tokens.slice()[0];
    drafter.?.greedy = false;
    // a sampled round takes no launched draft.
    dropAhead();
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
    // a forward moves the state past any launched draft.
    dropAhead();
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

/// The request's prompt and the prefix its state kept (`resumePrompt`): the usage record's token counts.
var req_prompt: usize = 0;
var req_kept: usize = 0;

/// The built-in's resume: the drafter's request begins on the prompt prefix the state kept.
fn resumePrompt(mp: *anyopaque, slot: u32, prompt: abi.Span(u32), media_rows: abi.Array) callconv(.c) u64 {
    // The next request's decode peak taken from its own first round.
    decode_peak_armed = false;
    // a new prompt's state takes no launched draft.
    dropAhead();
    const n = ds.table.cache.resume_prompt.?(mp, slot, prompt, media_rows);
    req_prompt = prompt.len;
    req_kept = @min(@as(usize, @intCast(n)), prompt.len);
    dropPending();
    if (drafter) |*d| d.begin(prompt.slice()[0..@min(@as(usize, @intCast(n)), prompt.len)]);
    return n;
}

var meter_buf: [262144]u8 = undefined;

/// The request's DSpark rounds beside the engine's `[spec-stats]`, which counts every native round a stage round, and the
/// expert stream's counters since the last request; the request's routed experts into the usage record when it is on.
fn meters(mp: *anyopaque, slot: u32, out: *abi.Str) callconv(.c) void {
    _ = slot;
    usageEnd(model(mp));
    const head = if (drafter) |*d| d.meters(&meter_buf) else meter_buf[0..0];
    const mid = streamMeters(model(mp), meter_buf[head.len..]);
    // The request's last round's launched draft has no round after it.
    dropAhead();
    out.* = abi.Str.of(meter_buf[0 .. head.len + mid.len]);
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

const Exl3Source = @import("src/mlx_serve/sdk/weights/streamed.zig").Exl3.Source;
const StreamStats = @TypeOf(@as(*Exl3Source, undefined).stats());
/// The stream's counters at the last request's end (they count for the process).
var stream_prev: StreamStats = .{};

/// `[stream-stats]`: the streamed bank's routes, resident hits and misses (unique experts a route), evictions, bytes
/// read and the host's read waits since the last request (the first request's include the load's).
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
    return w.buffered();
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
