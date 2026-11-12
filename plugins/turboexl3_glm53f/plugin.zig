//! turboexl3_glm53f: turboderp's self-calibrated GLM-5.3-Flash EXL3 packs in the streaming repack, on M5-class GPUs. The
//! model is mlx-serve's own glm5_next (a build-time copy, src/mlx_serve), its routed experts stream from the pack's bank
//! through mlx-serve's expert streamer (module `mlx_stream`) and run on turboexl3 at each layer's rate
//! (src/routed_t3.zig), turboexl3 runs every EXL3 linear of the trunk, the head and the MTP layer, and the MTP layer's
//! experts are one resident turboexl3 bank (src/mtp_bank.zig).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");
/// What the streamer's `sdk` and turboexl3's read through this root (`mlx_host`): the kit's mlx-c layer, the copies'
/// log, env and globals, the MTP acceptance modes, the quant ABI and the routed experts' math.
pub const mlx = kit.mlx;
pub const log = @import("src/mlx_serve/sdk/util/log.zig");
pub const env = @import("src/mlx_serve/env.zig");
pub const shared = @import("src/mlx_serve/sdk/util/shared.zig");
pub const mtp_acceptance = @import("src/mlx_serve/sdk/spec/mtp_acceptance.zig");
pub const quant_abi = kit.abi.quant;
/// The routed experts' math on turboexl3, run by the copied streamer (src/routed_t3.zig).
pub const routed_t3 = @import("src/routed_t3.zig");
/// The switches as the manifest's options (src/options.zig), on a kit that takes them.
const options = @import("src/options.zig");

const glm = @import("src/mlx_serve/plugins/glm5_next/plugin.zig");
const host = @import("src/mlx_serve/plugins/glm5_next/host.zig");
const glm_fwd = @import("src/mlx_serve/plugins/glm5_next/table_forward.zig");
const trunk = @import("src/mlx_serve/plugins/glm5_next/trunk.zig");
const head = @import("src/mlx_serve/plugins/glm5_next/head.zig");
const streamed = @import("src/mlx_serve/sdk/weights/streamed.zig");
const resident = @import("src/mlx_serve/sdk/weights/resident.zig");
const system = @import("src/mlx_serve/sdk/util/system.zig");
const device = @import("src/mlx_serve/mlx/device.zig");
const Layout = @import("src/mlx_serve/sdk/quant/native.zig").Layout;
const pack_weights = @import("src/pack_weights.zig");
const mtp_bank = @import("src/mtp_bank.zig");
const stream_room = @import("src/stream_room.zig");
const bank_layers = @import("src/bank_layers.zig");
const stream_stats = @import("src/stream_stats.zig");
const box_memory = @import("src/box_memory.zig");
const kda_prompt = @import("src/kda_prompt.zig");
const kda_verify = @import("src/kda_verify.zig");
/// TURBOEXL3_GLM53F_KDA_LAZY_SEQ: a verify's per-row states as a launch of their own, run only when a rollback reads them
/// (patches/turboexl3-model.patch).
const kda_lazy = @import("src/kda_lazy.zig");
const hc_fuse = @import("src/hc_fuse.zig");
const async_eval = @import("src/async_eval.zig");
/// TURBOEXL3_GLM53F_PF_TAIL_EVAL: a prompt layer's tail handed to the GPU at the layer's end where the cadence's eval is
/// not due (patches/turboexl3-model.patch).
const tail_eval = @import("src/tail_eval.zig");
/// TURBOEXL3_GLM53F_PF_REUSE: a prompt pass's buffers reused (src/pf_reuse.zig): a KDA run's held in MLX's cache, read by
/// the copied forward, and the routed prefill's at turboexl3's size classes.
const pf_reuse = @import("src/pf_reuse.zig");
const kda_pipe = @import("src/kda_pipe.zig");
/// TURBOEXL3_GLM53F_ROUTER and TURBOEXL3_GLM53F_PREDICT (public: the streamer's predictor, glm5n/experts.zig
/// `scoresBf16`, reaches `predict` through `mlx_host`).
pub const fused_router = @import("src/fused_router.zig");
/// TURBOEXL3_GLM53F_MOE_COMBINE1 and TURBOEXL3_GLM53F_ROUTE_VIEW (public: the streamer's decode lane, glm5n/experts.zig
/// `forward`, reaches them through `mlx_host`).
pub const moe_combine1 = @import("src/moe_combine1.zig");
pub const route_view = @import("src/route_view.zig");
/// TURBOEXL3_GLM53F_DSA_VATTN: a DSA layer's verify-window sparse latent attention without the gather
/// (patches/turboexl3-model.patch).
const dsa_vattn = @import("src/dsa_vattn.zig");
const dsa_prompt = @import("src/dsa_prompt.zig");
/// TURBOEXL3_GLM53F_DSA_VSEL: a DSA layer's verify-window selection in two launches (patches/turboexl3-model.patch).
const dsa_verify = @import("src/dsa_verify.zig");
/// TURBOEXL3_GLM53F_DSA_GLM: a DSA layer's prompt indexer as one GLM-5.3-Flash launch and its output in one copy
/// (patches/turboexl3-model.patch).
const dsa_glm = @import("src/dsa_glm.zig");
/// TURBOEXL3_GLM53F_DSA_LFUSE: a DSA layer's verify-window sparse latent attention in two launches at GLM-5.3-Flash's
/// dims (patches/turboexl3-model.patch).
const dsa_lfuse = @import("src/dsa_lfuse.zig");
/// TURBOEXL3_GLM53F_DSA_HOLD: a prompt span's DSA layer keeps its q_abs and o_lat for its next DSA layer
/// (patches/turboexl3-model.patch).
const dsa_hold = @import("src/dsa_hold.zig");
/// TURBOEXL3_GLM53F_DSA_TAILS, _DSA_ROWS and _DSA_WFOLD: a decode or verify window's DSA indexer tail rows read before the
/// forward, its rows assembled by the pooled-keys launch, its selection from the head-weight projection as it is
/// (patches/turboexl3-model.patch).
const dsa_tails = @import("src/dsa_tails.zig");
const dsa_rows = @import("src/dsa_rows.zig");
const dsa_wfold = @import("src/dsa_wfold.zig");
/// TURBOEXL3_GLM53F_SMALLOP_FUSE: the decode and verify rows' small ops merged where a kernel gives the same bits
/// (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch).
const smallop = @import("src/smallop.zig");
/// TURBOEXL3_GLM53F_DRAFT_S0: the MTP draft's step 0 on SMALLOP's texts (patches/turboexl3-model.patch
/// reaches it from head.zig by path).
const draft_s0 = @import("src/draft_s0.zig");
/// TURBOEXL3_GLM53F_KDA_OPREP and TURBOEXL3_GLM53F_DSA_OPREP: a verify linear's prepared input written by its producer,
/// DEC-ONE's GEMV alone after it (patches/turboexl3-model.patch
/// reaches it from kda.zig and table_forward.zig by path).
const dec_prep = @import("src/dec_prep.zig");
/// TURBOEXL3_GLM53F_HEAD_LAST_ROW: the MTP head's merged first step's routed experts at its last row alone
/// (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch).
const draft_step = @import("src/draft_step.zig");
/// TURBOEXL3_GLM53F_TRUNK_F16_FREE: the trunk's plain f16 matrices let go of their f16 sources after their bf16 casts,
/// the freed bytes given to the bank's rows (patches/turboexl3-model.patch).
const f16_free = @import("src/f16_free.zig");
/// TURBOEXL3_GLM53F_KDA_CONVIN and TURBOEXL3_GLM53F_KDA_EXPSIG: a KDA verify's conv input written by its prologue text,
/// exp(A_log) kept a layer and beta's sigmoid in its capture launch (src/kda_trim.zig,
/// patches/turboexl3-model.patch); TURBOEXL3_GLM53F_CACHED_SCALARS: the clamped SwiGLU's
/// limit made once (src/cached_scalars.zig, patches/turboexl3-sdk.patch);
/// TURBOEXL3_GLM53F_HOST_ZERO: DEC-ONE's arrival counters made on the host (src/host_zero.zig, turboexl3's dense.zig).
const kda_trim = @import("src/kda_trim.zig");
const cached_scalars = @import("src/cached_scalars.zig");
const host_zero = @import("src/host_zero.zig");
/// TURBOEXL3_GLM53F_LOGITS_DUMP: each main-model forward's rows' top-64 log-probabilities written to a file (a check,
/// never served; `dumpForward`).
const logits_dump = @import("src/logits_dump.zig");

comptime {
    if (streamed.Glm5n.transient_rows != stream_room.streamer_transient_rows) @compileError("turboexl3_glm53f: src/stream_room.zig's streamer_transient_rows is not the streamer's transient_rows");
    // decode's window 0: the decode lane's rows at top-8 (`mimo/stream.zig` `decode_lane_route_ids`)
    if (streamed.Glm5n.decode_lane_rows * 8 != stream_room.streamer_window_rows) @compileError("turboexl3_glm53f: src/stream_room.zig's streamer_window_rows is not the streamer's decode window");
    // the lookahead's records read ahead a layer call
    if (streamed.Glm5n.spec_budget != stream_room.streamer_spec_budget) @compileError("turboexl3_glm53f: src/stream_room.zig's streamer_spec_budget is not the streamer's lookahead budget");
    // the read pool's workers and each one's staging
    if (streamed.Glm5n.io_workers_default != stream_room.streamer_io_workers or streamed.Glm5n.reader_staging_bytes != stream_room.streamer_reader_staging_bytes) @compileError("turboexl3_glm53f: src/stream_room.zig's streamer_io_workers and streamer_reader_staging_bytes are not the streamer's read pool's");
}
const shims = @import("src/host_shims.zig");
const t3 = @import("turboexl3");
const t3kit = @import("turboexl3_kit");

/// Shown at install and in Settings.
pub const version = "0.3.0";

/// Its switches are declared as the manifest's options where the kit's `sdk.Plugin` has `options` (mlx-serve #1131's), so
/// the app shows them on the plugin's page; an older kit (G_KITTI's 26.11.2) builds the declaration without them.
const kit_options = if (@hasField(kit.sdk.Plugin, "options")) options.forKit(@typeInfo(@FieldType(kit.sdk.Plugin, "options")).pointer.child) else {};
pub const plugin: kit.sdk.Plugin = if (@hasField(kit.sdk.Plugin, "options")) .{
    .name = "turboexl3_glm53f",
    .kind = .model,
    .claims = &config.model_types,
    .options = &kit_options,
} else .{
    .name = "turboexl3_glm53f",
    .kind = .model,
    .claims = &config.model_types,
};

/// The built-in's table but for its name, claim and parse, its bill (`bill`: the built-in's and one prompt's), its load
/// (`load`: the plugin reads the shards and both banks), its weights' bill and the trunk head the engine's MTP readout
/// borrows (`trunkWeight`).
pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = glm.table.cache,
    .weights = &.{ .keep_key = &keepNone, .resident_bytes = &residentBytes },
    .spec = &.{ .trunk_weight = &trunkWeight, .head = &head.hooks },
    .meters = &meters,
};

/// The streamed bank's counters (`sdk_ext.expert.Stats`, what `Glm5n.Source.inner.stats()` returns), named through the
/// streamer's module: naming the source's own type pulls its MLX kernels into the MLX-free test binary.
const StreamStats = @import("mlx_stream").expert_stats;
/// `[stream-stats]` (src/stream_stats.zig): one streamed bank a process.
var meter: stream_stats.Meter(StreamStats) = .{};

/// The built-in's forward and the stream's counters marked at each phase's first forward. A split prompt forward the copy
/// does not serve (patches/turboexl3-model.patch, `host.splitRefusal`) is refused by name first, so the stream's phase
/// does not see it.
fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    if (host.splitRefusal(plugin.name, glm_fwd, host.of(mp), f)) |v| return v;
    if (host.of(mp).bank) |b| meter.mark(f.mode == .prefill, f.mode == .verify, f.rows, b.inner);
    if (logits_dump.on) return dumpForward(mp, f, out);
    return glm.table.forward(mp, f, out);
}

/// `TURBOEXL3_GLM53F_LOGITS_DUMP` (src/logits_dump.zig): the built-in's forward with its final-normed hidden asked for
/// (one more reference: no op changes), then each row's top-k log-probabilities written from that hidden; the logits
/// and hidden the engine asked for are the forward's own (a hidden it did not ask for is freed here).
fn dumpForward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) Verdict {
    var fh = f.*;
    fh.want_hidden = true;
    const v = glm.table.forward(mp, &fh, out);
    if (!v.isOk()) return v;
    const hidden: Array = .{ .ctx = out.hidden.ctx };
    if (hidden.ctx == null) return v;
    dumpRows(host.of(mp), f, hidden) catch |e| {
        logits_dump.skipped += 1;
        if (logits_dump.skipped == 1) log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_LOGITS_DUMP: a forward's rows were not written ({s}); the forward is served as it is\n", .{@errorName(e)});
    };
    if (!f.want_hidden) {
        _ = mlx.mlx_array_free(hidden);
        out.hidden = .{};
    }
    return v;
}

/// One forward's record: the final-normed `hidden` [1, rows, hidden] of one slot through the LM head
/// `logits_dump.block_rows` rows at a time, each row's log-softmax in f32 and its top k (best first), with the rows'
/// input ids (the forward's host tokens, else its device ids). A forward of more than one slot is counted, not written.
fn dumpRows(m: *trunk.Model, f: *const abi.Forward, hidden: Array) !void {
    const s = m.s;
    const sh = mlx.getShape(hidden);
    if (f.caches.len != 1 or sh.len != 3 or sh[0] != 1) {
        logits_dump.skipped += 1;
        return;
    }
    const rows: u32 = @intCast(sh[1]);
    const k = logits_dump.k;
    const ids = try gpa.alloc(u32, rows);
    defer gpa.free(ids);
    @memset(ids, logits_dump.no_id);
    if (f.tokens.len == rows) @memcpy(ids, f.tokens.slice()) else if (f.ids.ctx != null and mlx.mlx_array_size(.{ .ctx = f.ids.ctx }) == rows) {
        var u = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(u);
        try mlx.check(mlx.mlx_astype(&u, .{ .ctx = f.ids.ctx }, .uint32, s));
        try mlx.check(mlx.mlx_array_eval(u));
        const p = mlx.mlx_array_data_uint32(u) orelse return error.Glm53fLogitsDumpIds;
        @memcpy(ids, p[0..rows]);
    }
    const top_ids = try gpa.alloc(u32, @as(usize, rows) * k);
    defer gpa.free(top_ids);
    const top_lp = try gpa.alloc(f32, @as(usize, rows) * k);
    defer gpa.free(top_lp);
    var b0: u32 = 0;
    while (b0 < rows) : (b0 += logits_dump.block_rows) {
        const b1 = @min(rows, b0 + logits_dump.block_rows);
        const n = b1 - b0;
        // a block's rows, its logits through the served head, their log-softmax in f32, its top k sorted best first
        var arrs: [11]Array = @splat(.{ .ctx = null });
        defer for (arrs) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        try mlx.check(mlx.mlx_slice(&arrs[0], hidden, &[_]c_int{ 0, @intCast(b0), 0 }, 3, &[_]c_int{ 1, @intCast(b1), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        arrs[1] = try m.lmHead(arrs[0]);
        try mlx.check(mlx.mlx_astype(&arrs[2], arrs[1], .float32, s));
        try mlx.check(mlx.mlx_logsumexp_axis(&arrs[3], arrs[2], -1, true, s));
        try mlx.check(mlx.mlx_subtract(&arrs[4], arrs[2], arrs[3], s));
        try mlx.check(mlx.mlx_negative(&arrs[5], arrs[4], s));
        try mlx.check(mlx.mlx_argpartition_axis(&arrs[6], arrs[5], @intCast(k - 1), -1, s));
        try mlx.check(mlx.mlx_slice(&arrs[7], arrs[6], &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(n), @intCast(k) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_take_along_axis(&arrs[8], arrs[5], arrs[7], -1, s));
        try mlx.check(mlx.mlx_argsort_axis(&arrs[9], arrs[8], -1, s));
        // the k ids and log-probabilities in the sorted order
        var sid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sid);
        var slp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slp);
        var uid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(uid);
        try mlx.check(mlx.mlx_take_along_axis(&sid, arrs[7], arrs[9], -1, s));
        try mlx.check(mlx.mlx_take_along_axis(&arrs[10], arrs[4], sid, -1, s));
        try mlx.check(mlx.mlx_astype(&uid, sid, .uint32, s));
        try mlx.check(mlx.mlx_contiguous(&slp, arrs[10], false, s));
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{ uid, slp }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        const pi = mlx.mlx_array_data_uint32(uid) orelse return error.Glm53fLogitsDumpData;
        const pl = mlx.mlx_array_data_float32(slp) orelse return error.Glm53fLogitsDumpData;
        @memcpy(top_ids[@as(usize, b0) * k ..][0 .. @as(usize, n) * k], pi[0 .. @as(usize, n) * k]);
        @memcpy(top_lp[@as(usize, b0) * k ..][0 .. @as(usize, n) * k], pl[0 .. @as(usize, n) * k]);
    }
    const slot: u32 = if (f.slots.len > 0) f.slots.slice()[0] else 0;
    const pos0: u32 = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    try logits_dump.write(@backingInt(f.mode), slot, pos0, ids, top_ids, top_lp);
}

/// The request's `[stream-stats]` lines: its prompt phase's and its decode phase's counters.
fn meters(mp: *anyopaque, _: u32, out: *abi.Str) callconv(.c) void {
    const b = host.of(mp).bank orelse {
        out.* = .{};
        return;
    };
    out.* = abi.Str.of(meter.lines(b.inner.stats(), stream_stats.Extra.of(b.inner)));
    // TURBOEXL3_GLM53F_ROUTER: its routes by layer since the load (equal counts: every layer of every verify)
    if (fused_router.on) {
        var buf: [320]u8 = undefined;
        log.info("{s}", .{fused_router.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_PREDICT: the lookahead's scores by the plan since the load
    if (fused_router.predict_on) {
        var buf: [200]u8 = undefined;
        log.info("{s}", .{fused_router.predictSummary(&buf)});
    }
    // TURBOEXL3_GLM53F_ROUTER_ONE: the routes and scores that ran in one launch since the load
    if (fused_router.one_on) {
        var buf: [160]u8 = undefined;
        log.info("{s}", .{fused_router.oneSummary(&buf)});
    }
    // TURBOEXL3_GLM53F_ROUTER_ONE1: the one-row calls that ran in one launch, the head's apart
    if (fused_router.one1_on) {
        var buf: [160]u8 = undefined;
        log.info("{s}", .{fused_router.one1Summary(&buf)});
    }
    // TURBOEXL3_GLM53F_SMALLOP_FUSE: each piece's launches and the calls that ran the stock ops since the load
    if (smallop.on) {
        var buf: [320]u8 = undefined;
        log.info("{s}", .{smallop.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_DRAFT_S0: step 0's pieces taken and declined since the load
    if (draft_s0.on) {
        var buf: [160]u8 = undefined;
        log.info("{s}", .{draft_s0.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_KDA_OPREP and _DSA_OPREP: the producers' calls and declines since the load
    if (dec_prep.anyOn()) {
        var buf: [200]u8 = undefined;
        log.info("{s}", .{dec_prep.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_KDA_LAZY_SEQ: the verifies that took the lazy branch since the load
    if (kda_lazy.on) {
        var buf: [96]u8 = undefined;
        log.info("{s}", .{kda_lazy.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_MOE_COMBINE1 and TURBOEXL3_GLM53F_ROUTE_VIEW: the calls combined in one launch and the calls whose
    // ids took the view since the load
    if (moe_combine1.on) {
        var buf: [96]u8 = undefined;
        log.info("{s}", .{moe_combine1.summary(&buf)});
    }
    if (route_view.on) {
        var buf: [96]u8 = undefined;
        log.info("{s}", .{route_view.summary(&buf)});
    }
    // TURBOEXL3_GLM53F_KDA_CONVIN, _KDA_EXPSIG, _CACHED_SCALARS and _HOST_ZERO: their work since the load
    if (kda_trim.convin_on or kda_trim.expsig_on) {
        var buf: [256]u8 = undefined;
        log.info("{s}", .{kda_trim.summary(&buf)});
    }
    if (cached_scalars.on) {
        var buf: [128]u8 = undefined;
        log.info("{s}", .{cached_scalars.summary(&buf)});
    }
    if (host_zero.on) {
        var buf: [96]u8 = undefined;
        log.info("{s}", .{host_zero.summary(&buf, host_zero.calls())});
    }
    // TURBOEXL3_GLM53F_DSA_TAILS: the served forwards and each DSA cache's appends found donated and copied since the
    // load; TURBOEXL3_GLM53F_DSA_ROWS and _DSA_WFOLD: the windows each served
    if (dsa_tails.on) {
        var buf: [640]u8 = undefined;
        log.info("{s}", .{dsa_tails.summary(&buf)});
    }
    if (dsa_rows.on or dsa_wfold.on) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_ROWS windows={d} TURBOEXL3_GLM53F_DSA_WFOLD windows={d}\n", .{ dsa_rows.runs, dsa_wfold.runs });
    // TURBOEXL3_GLM53F_PF_DRAIN_LAG: the prompt groups drained at the next group's first commit, and after the next
    // group's banks (it committed nothing), since the load
    if (routed_t3.drain_lag_on) log.info(routed_t3.lag_summary, .{ routed_t3.lag_lagged, routed_t3.lag_flushed });
    // TURBOEXL3_GLM53F_HEAD_LAST_ROW: the request's draft steps run at the last row alone
    if (draft_step.last_row) {
        var buf: [512]u8 = undefined;
        const l = draft_step.line(&buf);
        if (l.len != 0) log.info("{s}", .{l});
    }
}

/// mlx-serve's expert usage file (the decode seed's form), named through the streamer's module.
const expert_usage = @import("mlx_stream").expert_usage;

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const gpa = std.heap.c_allocator;

/// The built-in's bill (its KV row and checkpoint) and one prompt's (`Bill.prefill_bytes`): the host's stream room
/// (`LoadCtx.stream_room`) for an outside plugin is the GPU ceiling less this prompt bill and the wired margin, 0 without
/// one (src/stream_room.zig).
pub fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    glm.table.bill(p, shape, out);
    out.prefill_bytes = &prefillBytes;
}

/// The engine's `glm5PrefillMemoryNeeded` over this pack (src/stream_room.zig `prefillNeed`), on MLX's NAX gate.
fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    const need = stream_room.prefillNeed(config.of(p), seq, chunk, kv_bits, warm, device.naxAvailable());
    // TURBOEXL3_GLM53F_LOGITS_DUMP: the dump's head blocks beside a forward
    const dump: u64 = if (logits_dump.requested(&routed_t3.processEnv) != null) logits_dump.transientBytes(config.of(p).vocab_size) else 0;
    return need +| dump;
}

/// The plugin reads the pack's shards itself (`LoadCtx.weights` replaced): the engine loads none of them.
fn keepNone(p: *const abi.Parsed, key: abi.Str) callconv(.c) bool {
    _ = .{ p, key };
    return false;
}

/// The head the engine's MTP readout reads (`trunkWeight`): the EXL3 head's weights at 8-bit affine, group 64.
pub const head_bits: u32 = 8;
pub const head_group: u32 = 64;

/// What the load made beside the model (one model at a time: one streamed bank serves one Mac): the MTP bank's arrays
/// and the readout head.
const Extra = struct {
    mp: *anyopaque,
    mtp: ?mtp_bank.Arrays = null,
    head_q: ?[3]Array = null,

    fn deinit(e: *Extra) void {
        if (e.mtp) |arrs| for (arrs) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        if (e.head_q) |arrs| for (arrs) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        e.* = undefined;
    }
};
var extra: ?Extra = null;

fn refused(comptime what: []const u8, e: anyerror) Verdict {
    log.err("[turboexl3_glm53f] " ++ what ++ ": {s}\n", .{@errorName(e)});
    return Verdict.named(.failed, @errorName(e));
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    _ = kit.host.Host.of(ctx.host) catch |e| {
        out.* = switch (e) {
            inline else => |x| Verdict.refuse(.unsupported, "turboexl3_glm53f: " ++ @errorName(x)),
        };
        return null;
    };
    // The streamer's events, slot buffers and launches, and turboexl3's kernels, all through the server's table.
    if (!shims.install(ctx.host)) {
        out.* = Verdict.refuse(.unsupported, "turboexl3_glm53f: the server's host table is older than this plugin's kit");
        return null;
    }
    t3kit.bind(ctx.host.?);
    // turboexl3's prompt GEMMs and the streamer's texts run on MLX's NAX tiles: below the M5 they compute wrong.
    if (!device.naxAvailable()) {
        out.* = Verdict.refuse(.unsupported, "turboexl3_glm53f: an M5 or newer GPU (NAX) is required");
        return null;
    }
    const sw = readSwitches() catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_LAYER_BILL, TURBOEXL3_GLM53F_DECODE_ROOM, TURBOEXL3_GLM53F_GROW_UNFILLED, TURBOEXL3_GLM53F_PROMPT_KEEP, TURBOEXL3_GLM53F_SPEC_PICK1, TURBOEXL3_GLM53F_DEMAND_SPLIT, TURBOEXL3_GLM53F_SHARED_HOIST, TURBOEXL3_GLM53F_READ_AHEAD3, TURBOEXL3_GLM53F_ADOPT_PIECES, TURBOEXL3_GLM53F_BILL_FIT, TURBOEXL3_GLM53F_FIT_INUSE, TURBOEXL3_GLM53F_SEED_DECODE and TURBOEXL3_GLM53F_KEEP_WARM take 0 or 1 (unset: 1), TURBOEXL3_GLM53F_DEMAND_HELP 0 or 2 (unset: 2): the load is refused\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    // TURBOEXL3_GLM53F_BANKED: turboexl3's banked decode schedule, configured before any of its kernels is built (a
    // geometry's banked state reads it when made)
    const banked = routed_t3.BankedSchedule.read(&routed_t3.processEnv) catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_BANKED takes g4m3.t4.t4 (unset: g4m3.t4.t4): the load is refused\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (banked) |bs| {
        bs.apply() catch |e| {
            log.err("[turboexl3_glm53f] turboexl3 refused the banked schedule ({s}): the load is refused\n", .{@errorName(e)});
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        log.info(routed_t3.banked_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_BANKED").?, bs.textSet(), bs.gu, bs.dn });
    }
    // TURBOEXL3_GLM53F_DEC: turboexl3's dense decode GEMV schedule, configured before any dense engine is made (a dense
    // shape's engine reads it when made)
    const dec = routed_t3.DecSchedule.read(&routed_t3.processEnv) catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_DEC takes g1 (unset: g1; 0: today's): the load is refused\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (dec) |d| {
        d.apply() catch |e| {
            log.err("[turboexl3_glm53f] turboexl3 refused the dense decode schedule ({s}): the load is refused\n", .{@errorName(e)});
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        log.info(routed_t3.dec_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_DEC").?, t3.dense.dec_variants[d.variant].name });
    }
    // TURBOEXL3_GLM53F_PF_OPTS: the routed prefill's options, configured before any prefill engine is made (a geometry's
    // engine reads them when made)
    const pf_opts = routed_t3.PfOpts.read(&routed_t3.processEnv) catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_PF_OPTS takes bk128 (unset: bk128; 0: the served options): the load is refused\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (pf_opts) |o| {
        o.apply() catch |e| {
            log.err("[turboexl3_glm53f] turboexl3 refused the routed prefill options ({s}): the load is refused\n", .{@errorName(e)});
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        log.info(routed_t3.pf_opts_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PF_OPTS").?, o.bk128 });
    }
    // TURBOEXL3_GLM53F_PF_REUSE: a prompt pass's buffers reused (src/pf_reuse.zig): the routed prefill's row-slot arrays
    // at turboexl3's size classes, configured before any prefill engine is made, and a KDA run's buffers held in MLX's
    // cache from its first layer to its last (patches/turboexl3-model.patch), the raise the bill's
    // layer envelope a forward row
    pf_reuse.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_REUSE")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfReuseSwitch: TURBOEXL3_GLM53F_PF_REUSE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfReuseSwitch");
        return null;
    };
    if (pf_reuse.on) {
        pf_reuse.applySizeClasses() catch |e| {
            log.err("[turboexl3_glm53f] turboexl3 refused the routed prefill's size classes ({s}): the load is refused\n", .{@errorName(e)});
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        pf_reuse.cap_row_bytes = stream_room.envelopeRowBytes(config.of(ctx.parsed));
    }
    // TURBOEXL3_GLM53F_PF_MOE_HOLD: with PF_REUSE (its hooks and the routed prefill's size classes), a DSA layer's
    // routed prefill finds its first two chunks' arrays in MLX's cache (src/pf_reuse.zig); the hold installed once the
    // model's stream exists (below)
    pf_reuse.hold_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_MOE_HOLD")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfMoeHoldSwitch: TURBOEXL3_GLM53F_PF_MOE_HOLD takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfMoeHoldSwitch");
        return null;
    };
    if (pf_reuse.hold_on and !pf_reuse.on) {
        log.err("[turboexl3_glm53f] Glm53fPfMoeHoldWithoutReuse: TURBOEXL3_GLM53F_PF_MOE_HOLD needs TURBOEXL3_GLM53F_PF_REUSE=1 (its layer hooks, and the size classes the hold's arrays match): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfMoeHoldWithoutReuse");
        return null;
    }
    // TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP: with PF_MOE_HOLD, the hand-back's limit from the cache's bytes when they are
    // over it, and no hold before a forward's first routed call (src/pf_reuse.zig)
    pf_reuse.hold_top = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfMoeHoldTopSwitch: TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfMoeHoldTopSwitch");
        return null;
    };
    if (pf_reuse.hold_top and !pf_reuse.hold_on) {
        log.err("[turboexl3_glm53f] Glm53fPfMoeHoldTopWithoutHold: TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP needs TURBOEXL3_GLM53F_PF_MOE_HOLD=1 (the hold whose hand-back it raises): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfMoeHoldTopWithoutHold");
        return null;
    }
    // TURBOEXL3_GLM53F_PF_RUN_TRIM: with PF_REUSE, a KDA run's cache trimmed during its last layer's routed call, not at
    // the next DSA layer's first allocation (src/pf_reuse.zig)
    pf_reuse.run_trim = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_RUN_TRIM")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfRunTrimSwitch: TURBOEXL3_GLM53F_PF_RUN_TRIM takes 0 or 1 (unset: 0): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfRunTrimSwitch");
        return null;
    };
    if (pf_reuse.run_trim and !pf_reuse.on) {
        log.err("[turboexl3_glm53f] Glm53fPfRunTrimWithoutReuse: TURBOEXL3_GLM53F_PF_RUN_TRIM needs TURBOEXL3_GLM53F_PF_REUSE=1 (the runs whose cache it trims): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfRunTrimWithoutReuse");
        return null;
    }
    // TURBOEXL3_GLM53F_PF_INT8: the routed prefill's gate|up and down GEMMs on int8 (not exact), configured before any
    // prefill engine is made (a geometry's engine reads it when made); 3 the trunk's KDA qkv and o_proj on int8 too, 4
    // every trunk linear
    const pf_int8 = routed_t3.PfInt8.read(&routed_t3.processEnv) catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_PF_INT8 takes 1 (a scale each activation 128-block), 2 (one scale a row), 3 (1 and the trunk's KDA qkv and o_proj on int8) or 4 (1 and every trunk linear on int8) (unset or 0: the exact chain): the load is refused\n", .{@errorName(e)});
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (pf_int8) |p| {
        p.apply() catch |e| {
            log.err("[turboexl3_glm53f] turboexl3 refused the routed prefill's int8 scheme ({s}): the load is refused\n", .{@errorName(e)});
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        log.info(routed_t3.pf_int8_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PF_INT8").?, p.words() });
        if (p.scheme == 3) {
            const kda_q8 = routed_t3.trunk_int8_shapes;
            log.info(routed_t3.pf_trunk_int8_line, .{ kda_q8[0].k, kda_q8[0].n, kda_q8[1].k, kda_q8[1].n });
        }
        if (p.scheme == 4) log.info(routed_t3.pf_trunk_int8_all_line, .{routed_t3.trunk_int8_all.len});
    }
    // TURBOEXL3_GLM53F_LOGITS_DUMP: a check, never served: each main-model forward's rows' top-64 log-probabilities written
    // to <dir>/logits-<pid>.bin (src/logits_dump.zig, `dumpForward`); the file opened here, refused by name when the
    // directory is not absolute or the file does not open
    if (logits_dump.requested(&routed_t3.processEnv)) |dump_dir| {
        var dump_pb: [1024]u8 = undefined;
        const dump_path = logits_dump.open(dump_dir, &dump_pb) catch |e| {
            log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_LOGITS_DUMP takes an absolute directory the server can write in ('{s}'): the load is refused\n", .{ @errorName(e), dump_dir });
            out.* = Verdict.named(.failed, @errorName(e));
            return null;
        };
        log.info(logits_dump.line, .{ dump_dir, dump_path });
    }
    // TURBOEXL3_GLM53F_PG2: the trunk's linears at prefill rows on pg2 at the coded tiles, the table installed before the
    // first prompt (its kernels compiled once the model's stream exists, below)
    const pg2 = routed_t3.Pg2.read(&routed_t3.processEnv) catch |e| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_PG2 takes one tile code a trunk linear ({d} of 0-9 and a-z, turboexl3's pgemm.candidates; unset: 50115151111; 0: off): the load is refused\n", .{ @errorName(e), t3.glm53f.trunk.len });
        out.* = Verdict.named(.failed, @errorName(e));
        return null;
    };
    if (pg2) |p| {
        p.apply();
        log.info(routed_t3.pg2_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PG2").?, p.n, t3.glm53f.trunk.len });
    }
    // TURBOEXL3_GLM53F_PF_STAGE: a prompt call's misses routed before its routing barrier (the copied scheduler's
    // `Wide.stage`, src/streamed/experts.zig), set before the bank opens
    const stage = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_STAGE")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfStageSwitch: TURBOEXL3_GLM53F_PF_STAGE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfStageSwitch");
        return null;
    };
    streamed.Glm5n.setStageMisses(stage);
    if (stage) log.info(routed_t3.pf_stage_line, .{});
    // TURBOEXL3_GLM53F_PF_DRAIN_LAG: a prompt MoE group's drain, release and next route after the next group's first
    // committed chunk (the scheduler's `Wide.drain_lag`, src/streamed/experts.zig), set before the bank
    // opens. With PF_FEED only: without it a prompt call waits for its chunks at its end, so no group is left to lag.
    const drain_lag = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_DRAIN_LAG")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfDrainLagSwitch: TURBOEXL3_GLM53F_PF_DRAIN_LAG takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfDrainLagSwitch");
        return null;
    };
    if (drain_lag and !(routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_FEED")) catch false)) {
        log.err("[turboexl3_glm53f] Glm53fPfDrainLagWithoutFeed: TURBOEXL3_GLM53F_PF_DRAIN_LAG needs TURBOEXL3_GLM53F_PF_FEED=1 (a prompt call that waits for its chunks at its end leaves no group to lag): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfDrainLagWithoutFeed");
        return null;
    }
    streamed.Glm5n.setDrainLag(drain_lag);
    routed_t3.drain_lag_on = drain_lag;
    if (drain_lag) log.info(routed_t3.pf_drain_lag_line, .{});
    // TURBOEXL3_GLM53F_KDA_PROLOGUE and _KDA_NORMGATE: a KDA layer's prompt pieces as texts (src/kda_prompt.zig,
    // patches/turboexl3-model.patch), read at each prompt forward
    kda_prompt.prologue_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_PROLOGUE")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaPrologueSwitch: TURBOEXL3_GLM53F_KDA_PROLOGUE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaPrologueSwitch");
        return null;
    };
    kda_prompt.normgate_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_NORMGATE")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaNormgateSwitch: TURBOEXL3_GLM53F_KDA_NORMGATE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaNormgateSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_KDA_PERCORE: the prompt recurrence and gate on the per-core kernel, planned and probed once the
    // model's stream exists (below)
    kda_prompt.percore_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_PERCORE")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaPercoreSwitch: TURBOEXL3_GLM53F_KDA_PERCORE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaPercoreSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_KDA_CHUNK: the prompt recurrence and gate as the chunkwise delta rule, checked and probed once the
    // model's stream exists (below); before the per-core route where both are on
    kda_prompt.chunk_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_CHUNK")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaChunkSwitch: TURBOEXL3_GLM53F_KDA_CHUNK takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaChunkSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_KDA_FOLD: the prompt prologue folded into the chunkwise route's launch
    // (src/kernels/kda_fold.metal, patches/turboexl3-model.patch), probed after that route's install
    // (below); it needs KDA_CHUNK
    kda_prompt.fold_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_FOLD")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaFoldSwitch: TURBOEXL3_GLM53F_KDA_FOLD takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaFoldSwitch");
        return null;
    };
    if (kda_prompt.fold_on and !kda_prompt.chunk_on) {
        log.err("[turboexl3_glm53f] Glm53fKdaFoldNeedsChunk: TURBOEXL3_GLM53F_KDA_FOLD=1 folds the prologue into the chunkwise route's launch and needs TURBOEXL3_GLM53F_KDA_CHUNK=1: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaFoldNeedsChunk");
        return null;
    }
    // TURBOEXL3_GLM53F_KDA_PIPE: a KDA layer's prompt forward as row spans, each span's qkv linear on a side stream
    // (src/kda_pipe.zig, patches/turboexl3-model.patch), the stream made after the chunkwise route's
    // install (below); the spans run the chunkwise route after the prologue text or the fold
    kda_pipe.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_PIPE")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaPipeSwitch: TURBOEXL3_GLM53F_KDA_PIPE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaPipeSwitch");
        return null;
    };
    if (kda_pipe.on and !kda_prompt.chunk_on) {
        log.err("[turboexl3_glm53f] Glm53fKdaPipeNeedsChunk: TURBOEXL3_GLM53F_KDA_PIPE=1 runs the spans on the chunkwise route and needs TURBOEXL3_GLM53F_KDA_CHUNK=1: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaPipeNeedsChunk");
        return null;
    }
    if (kda_pipe.on and !kda_prompt.prologue_on and !kda_prompt.fold_on) {
        log.err("[turboexl3_glm53f] Glm53fKdaPipeNeedsPrologue: TURBOEXL3_GLM53F_KDA_PIPE=1 makes each span's q, k and v with the prologue text or the fold and needs TURBOEXL3_GLM53F_KDA_PROLOGUE=1 or TURBOEXL3_GLM53F_KDA_FOLD=1: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaPipeNeedsPrologue");
        return null;
    }
    // TURBOEXL3_GLM53F_ASYNC_EVAL: a prompt forward's cadence eval handed to the GPU without the host's wait
    // (src/async_eval.zig, patches/turboexl3-model.patch), read at each prompt forward's evals
    async_eval.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_ASYNC_EVAL")) catch {
        log.err("[turboexl3_glm53f] Glm53fAsyncEvalSwitch: TURBOEXL3_GLM53F_ASYNC_EVAL takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fAsyncEvalSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_PF_TAIL_EVAL: a prompt layer's tail handed to the GPU at the layer's end where the cadence's eval is
    // not due (src/tail_eval.zig, patches/turboexl3-model.patch), read at each prompt layer
    tail_eval.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PF_TAIL_EVAL")) catch {
        log.err("[turboexl3_glm53f] Glm53fPfTailEvalSwitch: TURBOEXL3_GLM53F_PF_TAIL_EVAL takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPfTailEvalSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_HC_FUSE: a prompt sublayer's hyper-connection expand and the next sublayer's collapse as one text
    // (src/hc_fuse.zig, patches/turboexl3-model.patch), read at each prompt sublayer
    hc_fuse.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_HC_FUSE")) catch {
        log.err("[turboexl3_glm53f] Glm53fHcFuseSwitch: TURBOEXL3_GLM53F_HC_FUSE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fHcFuseSwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_KDA_VERIFY: the prologue, gate and gated-norm texts in a KDA layer's verify forward too
    // (src/kda_verify.zig, patches/turboexl3-model.patch), read at each verify
    kda_verify.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_VERIFY")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaVerifySwitch: TURBOEXL3_GLM53F_KDA_VERIFY takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaVerifySwitch");
        return null;
    };
    // TURBOEXL3_GLM53F_KDA_LAZY_SEQ: a verify's per-row states only when a partial accept's rollback reads them
    // (src/kda_lazy.zig)
    kda_lazy.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_KDA_LAZY_SEQ")) catch {
        log.err("[turboexl3_glm53f] Glm53fKdaLazySeqSwitch: TURBOEXL3_GLM53F_KDA_LAZY_SEQ takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fKdaLazySeqSwitch");
        return null;
    };
    if (kda_lazy.on) log.info(kda_lazy.line, .{});
    // TURBOEXL3_GLM53F_KDA_CONVIN and _KDA_EXPSIG: a KDA verify's conv input written by its prologue text, exp(A_log) kept
    // a layer and beta's sigmoid in its capture launch (src/kda_trim.zig); CONVIN edits KDA_VERIFY's prologue text.
    // TURBOEXL3_GLM53F_CACHED_SCALARS: the clamped SwiGLU's limit made once (src/cached_scalars.zig). TURBOEXL3_GLM53F_HOST_ZERO: DEC-ONE's arrival counters
    // made on the host (src/host_zero.zig), set in turboexl3 before any dense call
    inline for (.{
        .{ &kda_trim.convin_on, "TURBOEXL3_GLM53F_KDA_CONVIN", "Glm53fKdaConvinSwitch" },
        .{ &kda_trim.expsig_on, "TURBOEXL3_GLM53F_KDA_EXPSIG", "Glm53fKdaExpsigSwitch" },
        .{ &cached_scalars.on, "TURBOEXL3_GLM53F_CACHED_SCALARS", "Glm53fCachedScalarsSwitch" },
        .{ &host_zero.on, "TURBOEXL3_GLM53F_HOST_ZERO", "Glm53fHostZeroSwitch" },
    }) |sw_| {
        sw_[0].* = routed_t3.parseSwitch(routed_t3.processEnv(sw_[1])) catch {
            log.err("[turboexl3_glm53f] " ++ sw_[2] ++ ": " ++ sw_[1] ++ " takes 0 or 1 (unset: 1): the load is refused\n", .{});
            out.* = Verdict.named(.failed, sw_[2]);
            return null;
        };
    }
    if (kda_trim.refusal(kda_trim.convin_on, kda_verify.on)) |name| {
        log.err("[turboexl3_glm53f] {s}: TURBOEXL3_GLM53F_KDA_CONVIN=1 edits TURBOEXL3_GLM53F_KDA_VERIFY's prologue text and needs it: the load is refused\n", .{name});
        out.* = Verdict.named(.failed, name);
        return null;
    }
    if (kda_trim.convin_on) log.info(kda_trim.convin_line, .{});
    if (kda_trim.expsig_on) log.info(kda_trim.expsig_line, .{});
    if (cached_scalars.on) log.info(cached_scalars.line, .{});
    if (host_zero.on) {
        host_zero.apply();
        log.info(host_zero.line, .{});
    }
    if (kda_prompt.prologue_on) log.info(kda_prompt.prologue_line, .{});
    if (kda_prompt.normgate_on) log.info(kda_prompt.normgate_line, .{});
    if (kda_verify.on) log.info(kda_verify.line, .{});
    if (hc_fuse.on) log.info(hc_fuse.line, .{});
    if (async_eval.on) log.info(async_eval.line, .{});
    if (tail_eval.on) log.info(tail_eval.line, .{});
    if (pf_reuse.on) log.info(pf_reuse.line, .{ pf_reuse.cap_row_bytes >> 10, t3.prefill.size_class_rows });
    if (pf_reuse.run_trim) log.info(pf_reuse.run_trim_line, .{});
    // TURBOEXL3_GLM53F_ROUTER: a routed layer's router in a 2- to 4-row call in two launches (src/fused_router.zig,
    // patches/turboexl3-sdk.patch), each layer's raw router registered when the bank opens
    fused_router.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_ROUTER")) catch {
        log.err("[turboexl3_glm53f] Glm53fRouterSwitch: TURBOEXL3_GLM53F_ROUTER takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouterSwitch");
        return null;
    };
    if (fused_router.on) log.info(fused_router.line, .{});
    // TURBOEXL3_GLM53F_PREDICT: the decode lookahead's next-layer scores in two launches (src/fused_router.zig,
    // src/streamed/glm5n/experts.zig)
    fused_router.predict_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_PREDICT")) catch {
        log.err("[turboexl3_glm53f] Glm53fPredictSwitch: TURBOEXL3_GLM53F_PREDICT takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fPredictSwitch");
        return null;
    };
    if (fused_router.predict_on) log.info(fused_router.predict_line, .{});
    // TURBOEXL3_GLM53F_ROUTER_ONE: ROUTER's and PREDICT's calls each in one launch (src/fused_router.zig); with
    // neither on it would never run, so it refuses the load by name
    fused_router.one_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_ROUTER_ONE")) catch {
        log.err("[turboexl3_glm53f] Glm53fRouterOneSwitch: TURBOEXL3_GLM53F_ROUTER_ONE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouterOneSwitch");
        return null;
    };
    if (fused_router.one_on and !fused_router.on and !fused_router.predict_on) {
        log.err("[turboexl3_glm53f] Glm53fRouterOneAlone: TURBOEXL3_GLM53F_ROUTER_ONE=1 makes ROUTER's and PREDICT's calls one launch each, and neither TURBOEXL3_GLM53F_ROUTER nor TURBOEXL3_GLM53F_PREDICT is on: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouterOneAlone");
        return null;
    }
    if (fused_router.one_on) log.info(fused_router.one_line, .{});
    // TURBOEXL3_GLM53F_MTP_ROUTER: the MTP head's router registered with ROUTER's plan too (src/fused_router.zig,
    // patches/turboexl3-model.patch: its raw router loaded with the head only with this
    // switch); without ROUTER it would never route, so it refuses the load by name
    fused_router.mtp_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_MTP_ROUTER")) catch {
        log.err("[turboexl3_glm53f] Glm53fMtpRouterSwitch: TURBOEXL3_GLM53F_MTP_ROUTER takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fMtpRouterSwitch");
        return null;
    };
    if (fused_router.mtp_on and !fused_router.on) {
        log.err("[turboexl3_glm53f] Glm53fMtpRouterAlone: TURBOEXL3_GLM53F_MTP_ROUTER=1 registers the MTP head's router with ROUTER's plan, and TURBOEXL3_GLM53F_ROUTER is not on: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fMtpRouterAlone");
        return null;
    }
    if (fused_router.mtp_on) log.info(fused_router.mtp_line, .{});
    // TURBOEXL3_GLM53F_ROUTER_ONE1: a registered layer's one-row router call in one launch (src/fused_router.zig, MLX's
    // gemv text vendored in src/vendor/mlx/gemv.h); only ROUTER registers the layers, so without it the load is refused
    fused_router.one1_on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_ROUTER_ONE1")) catch {
        log.err("[turboexl3_glm53f] Glm53fRouterOne1Switch: TURBOEXL3_GLM53F_ROUTER_ONE1 takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouterOne1Switch");
        return null;
    };
    if (fused_router.one1_on and !fused_router.on) {
        log.err("[turboexl3_glm53f] Glm53fRouterOne1Alone: TURBOEXL3_GLM53F_ROUTER_ONE1=1 runs a registered layer's one-row router call in one launch, and TURBOEXL3_GLM53F_ROUTER (which registers the layers) is not on: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouterOne1Alone");
        return null;
    }
    if (fused_router.one1_on) log.info(fused_router.one1_line, .{});
    // TURBOEXL3_GLM53F_MOE_COMBINE1: a decode call's join and combine in one launch (src/moe_combine1.zig)
    moe_combine1.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_MOE_COMBINE1")) catch {
        log.err("[turboexl3_glm53f] Glm53fMoeCombine1Switch: TURBOEXL3_GLM53F_MOE_COMBINE1 takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fMoeCombine1Switch");
        return null;
    };
    if (moe_combine1.on) log.info(moe_combine1.line, .{});
    // TURBOEXL3_GLM53F_ROUTE_VIEW: a decode call's route ids as an int32 view of the router's uint32 ids
    // (src/route_view.zig)
    route_view.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_ROUTE_VIEW")) catch {
        log.err("[turboexl3_glm53f] Glm53fRouteViewSwitch: TURBOEXL3_GLM53F_ROUTE_VIEW takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fRouteViewSwitch");
        return null;
    };
    if (route_view.on) log.info(route_view.line, .{});
    // TURBOEXL3_GLM53F_SMALLOP_FUSE: the decode and verify rows' small ops merged where a kernel gives the same bits
    // (src/smallop.zig, patches/turboexl3-model.patch and patches/turboexl3-sdk.patch), the dense layers' pairs
    // registered when the bank
    // opens
    smallop.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_SMALLOP_FUSE")) catch {
        log.err("[turboexl3_glm53f] Glm53fSmallopSwitch: TURBOEXL3_GLM53F_SMALLOP_FUSE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fSmallopSwitch");
        return null;
    };
    if (smallop.on) log.info(smallop.line, .{});
    // TURBOEXL3_GLM53F_DRAFT_S0: the MTP draft's step 0 (2 to 4 rows) on SMALLOP's texts (src/draft_s0.zig,
    // patches/turboexl3-model.patch); without SMALLOP_FUSE it would never run, so it refuses
    // the load by name
    draft_s0.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_DRAFT_S0")) catch {
        log.err("[turboexl3_glm53f] Glm53fDraftS0Switch: TURBOEXL3_GLM53F_DRAFT_S0 takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fDraftS0Switch");
        return null;
    };
    if (draft_s0.on and !smallop.on) {
        log.err("[turboexl3_glm53f] Glm53fDraftS0Alone: TURBOEXL3_GLM53F_DRAFT_S0=1 runs the draft's step 0 on SMALLOP's texts, and TURBOEXL3_GLM53F_SMALLOP_FUSE is not on: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fDraftS0Alone");
        return null;
    }
    if (draft_s0.on) log.info(draft_s0.line, .{});
    // TURBOEXL3_GLM53F_KDA_OPREP and _DSA_OPREP: a verify linear's producer writing its prepared input (src/dec_prep.zig,
    // patches/turboexl3-model.patch), each refused by name on a value other than 0 or 1
    inline for (.{
        .{ "TURBOEXL3_GLM53F_KDA_OPREP", "Glm53fKdaOprepSwitch", &dec_prep.kda_on, dec_prep.kda_line },
        .{ "TURBOEXL3_GLM53F_DSA_OPREP", "Glm53fDsaOprepSwitch", &dec_prep.dsa_o_on, dec_prep.dsa_o_line },
    }) |dp| {
        dp[2].* = routed_t3.parseSwitch(routed_t3.processEnv(dp[0])) catch {
            log.err("[turboexl3_glm53f] " ++ dp[1] ++ ": " ++ dp[0] ++ " takes 0 or 1 (unset: 1): the load is refused\n", .{});
            out.* = Verdict.named(.failed, dp[1]);
            return null;
        };
        if (dp[2].*) log.info(dp[3], .{});
    }
    // TURBOEXL3_GLM53F_HEAD_LAST_ROW: the merged first draft step's routed experts for its last row alone
    // (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch)
    draft_step.last_row = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_HEAD_LAST_ROW")) catch {
        log.err("[turboexl3_glm53f] Glm53fHeadLastRowSwitch: TURBOEXL3_GLM53F_HEAD_LAST_ROW takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fHeadLastRowSwitch");
        return null;
    };
    if (draft_step.last_row) log.info(draft_step.last_row_line, .{});
    draft_step.reset();
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the trunk's f16 sources let go by its loader (src/f16_free.zig,
    // patches/turboexl3-model.patch), so set before it; released after it, before the bank opens
    f16_free.on = routed_t3.parseSwitch(routed_t3.processEnv("TURBOEXL3_GLM53F_TRUNK_F16_FREE")) catch {
        log.err("[turboexl3_glm53f] Glm53fTrunkF16FreeSwitch: TURBOEXL3_GLM53F_TRUNK_F16_FREE takes 0 or 1 (unset: 1): the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fTrunkF16FreeSwitch");
        return null;
    };
    if (f16_free.on) log.info(f16_free.line, .{});
    f16_free.reset();
    // TURBOEXL3_GLM53F_DSA_ONEP and _DSA_VSEL: a DSA layer's prompt sparse attention (src/dsa_prompt.zig,
    // patches/turboexl3-model.patch) and its verify window's selection (src/dsa_verify.zig), read at each forward
    inline for (.{
        // TURBOEXL3_GLM53F_DSA_VATTN: a DSA layer's verify-window sparse latent attention without the gather
        // (src/dsa_vattn.zig, patches/turboexl3-model.patch), read at each verify
        .{ "TURBOEXL3_GLM53F_DSA_VATTN", "Glm53fDsaVattnSwitch", &dsa_vattn.on, dsa_vattn.line },
        .{ "TURBOEXL3_GLM53F_DSA_ONEP", "Glm53fDsaOnePSwitch", &dsa_prompt.onep_on, dsa_prompt.onep_line },
        .{ "TURBOEXL3_GLM53F_DSA_VSEL", "Glm53fDsaVselSwitch", &dsa_verify.vsel_on, dsa_verify.vsel_line },
        // TURBOEXL3_GLM53F_DSA_GLM: a DSA layer's prompt indexer and output on GLM-5.3-Flash's kernels (src/dsa_glm.zig,
        // patches/turboexl3-model.patch), read at each prompt forward
        .{ "TURBOEXL3_GLM53F_DSA_GLM", "Glm53fDsaGlmSwitch", &dsa_glm.on, dsa_glm.line },
        // TURBOEXL3_GLM53F_DSA_LFUSE: a DSA layer's verify-window sparse latent attention in two launches (src/dsa_lfuse.zig,
        // patches/turboexl3-model.patch), read at each verify; DSA_VATTN where it declines
        .{ "TURBOEXL3_GLM53F_DSA_LFUSE", "Glm53fDsaLfuseSwitch", &dsa_lfuse.on, dsa_lfuse.line },
        // TURBOEXL3_GLM53F_DSA_TAILS, _DSA_ROWS and _DSA_WFOLD: a decode or verify window's DSA indexer tail rows read
        // first, its rows built by the pooled-keys launch, its selection from the head-weight projection (src/dsa_tails.zig,
        // src/dsa_rows.zig, src/dsa_wfold.zig, patches/turboexl3-model.patch), read at each decode, verify and draft step
        .{ "TURBOEXL3_GLM53F_DSA_TAILS", "Glm53fDsaTailsSwitch", &dsa_tails.on, dsa_tails.line },
        .{ "TURBOEXL3_GLM53F_DSA_ROWS", "Glm53fDsaRowsSwitch", &dsa_rows.on, dsa_rows.line },
        .{ "TURBOEXL3_GLM53F_DSA_WFOLD", "Glm53fDsaWfoldSwitch", &dsa_wfold.on, dsa_wfold.line },
        // TURBOEXL3_GLM53F_DSA_HOLD: a prompt span's DSA layer's q_abs and o_lat held for its next DSA layer and released
        // into MLX's cache there (src/dsa_hold.zig, patches/turboexl3-model.patch), read at each prompt forward; a memory
        // trade
        .{ "TURBOEXL3_GLM53F_DSA_HOLD", "Glm53fDsaHoldSwitch", &dsa_hold.on, dsa_hold.line },
    }) |lever| {
        lever[2].* = routed_t3.parseSwitch(routed_t3.processEnv(lever[0])) catch {
            log.err("[turboexl3_glm53f] " ++ lever[1] ++ ": " ++ lever[0] ++ " takes 0 or 1 (unset: 1): the load is refused\n", .{});
            out.* = Verdict.named(.failed, lever[1]);
            return null;
        };
        if (lever[2].*) log.info(lever[3], .{});
    }
    const c = config.of(ctx.parsed);
    const dir = ctx.model_dir.slice();
    const io = std.Io.Threaded.global_single_threaded.io();
    // TURBOEXL3_GLM53F_BILL_FIT: the server's context limit (the load settings' ctx_size) and the fit's pack (sc_3.10),
    // each refused by name before the pack is read.
    var context: u64 = 0;
    // TURBOEXL3_GLM53F_FIT_INUSE changes BILL_FIT's allowance for the memory outside the server: alone it has no plan
    if (sw.fit_inuse and !sw.bill_fit) {
        log.err("[turboexl3_glm53f] Glm53fFitInUseAlone: TURBOEXL3_GLM53F_FIT_INUSE=1 sets TURBOEXL3_GLM53F_BILL_FIT's allowance for the memory outside the server, and BILL_FIT is off: the load is refused\n", .{});
        out.* = Verdict.named(.failed, "Glm53fFitInUseAlone");
        return null;
    }
    if (sw.bill_fit) {
        context = stream_room.contextLimit(gpa, ctx.settings_json.slice()) orelse {
            log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_BILL_FIT=1 sizes the rows for any request up to the server's context limit, and the load settings name none (no ctx_size: start the server with --ctx-size): the load is refused\n", .{});
            out.* = Verdict.named(.failed, @errorName(error.Glm53fBillFitContext));
            return null;
        };
        const fit_geo = streamed.Glm5n.geometryOf(gpa, io, dir) catch |e| {
            out.* = refused("the streamed bank's manifests do not read", e);
            return null;
        };
        if (fit_geo.record_bytes != stream_room.fit_record) {
            log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_BILL_FIT=1 is fitted to sc_3.10 (its widest record {d} B), and this pack's widest record is {d} B: the load is refused\n", .{ stream_room.fit_record, fit_geo.record_bytes });
            out.* = Verdict.named(.failed, @errorName(error.Glm53fBillFitPack));
            return null;
        }
    }
    // TURBOEXL3_GLM53F_SEED_DECODE: the pack's decode-weighted profile, refused by name when absent, before the pack is
    // read; its sha256 for the load line.
    var seed: ?SeedFile = null;
    if (sw.seed_decode) {
        seed = seedDecodeFile(io, dir) catch |e| switch (e) {
            error.FileNotFound => {
                log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_SEED_DECODE=1 seeds the prompt rows from the pack's {s}, and {s} has none: the load is refused\n", .{ streamed.Glm5n.seed_decode_file, dir });
                out.* = Verdict.named(.failed, @errorName(error.Glm53fSeedDecodeMissing));
                return null;
            },
            else => {
                out.* = refused("the pack's decode-weighted usage profile does not read", e);
                return null;
            },
        };
    }
    var res = resident.Resident.adopt(pack_weights.read(io, gpa, dir, c.num_hidden_layers, .{ .heads = c.num_attention_heads, .head_dim = c.dsa_head_dim, .latent = c.dsa_kv_lora_rank }) catch |e| {
        out.* = refused("the pack's shards do not load", e);
        return null;
    });
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE releases the source map right after the trunk's load (nothing after it reads it)
    var res_live = true;
    defer if (res_live) res.deinit();
    const src = res.source();
    var own = ctx.*;
    own.weights = &src;
    const mp = host.load(plugin.name, &own, out) orelse {
        f16_free.reset();
        return null;
    };
    const m = host.of(mp);
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the source map released, then the f16 sources the loader let go; what came back
    // (measured) goes to the bank's rows (`openBank`: BILL_FIT's bases less it)
    if (f16_free.on) {
        res.deinit();
        res_live = false;
        const n = f16_free.tensors;
        const stored = f16_free.nominal;
        const got = f16_free.release();
        var buf: [256]u8 = undefined;
        log.info("{s}", .{f16_free.released(&buf, n, stored, got)});
    }
    if (pg2 != null) routed_t3.Pg2.warm(m.s) catch |e| {
        out.* = refused("turboexl3's pg2 kernels do not compile", e);
        unload(mp);
        return null;
    };
    if (extra) |*e| e.deinit();
    extra = .{ .mp = mp };
    // What stays resident beside the bank, before its slot rows are sized from the memory left: the MTP layer's
    // experts, then the readout head.
    bindMtp(m, dir, &extra.?) catch |e| {
        out.* = refused("the MTP layer's bank does not load", e);
        unload(mp);
        return null;
    };
    extra.?.head_q = readoutHead(m) catch |e| blk: {
        log.warn("[turboexl3_glm53f] the MTP readout head did not build ({s}): MTP off\n", .{@errorName(e)});
        break :blk null;
    };
    _ = mlx.mlx_clear_cache();
    // The bank's room: the host's when it gives one, else the plugin's own (mlx-serve 26.11.1's driver gives an outside
    // plugin 0): the host's GPU ceiling less one prompt's bill and the host's wired margin (src/stream_room.zig).
    const env_mb: ?[]const u8 = if (std.c.getenv("MLX_SERVE_GPU_CEILING_MB")) |v| std.mem.span(v) else null;
    const ceiling = stream_room.gpuCeiling(env_mb, mlx.maxRecommendedWorkingSet(), system.getTotalMemBytes());
    const prompt = stream_room.prefillNeed(c, stream_room.room_seq, stream_room.room_chunk, 16, 0, device.naxAvailable());
    const room = stream_room.roomFor(ctx.stream_room, ceiling, prompt, ctx.wired_margin_bytes);
    if (room.source == .own) log.info("[turboexl3_glm53f] the host gives no stream room: the plugin's own, the ceiling {d} MiB less one 16K + 1K prompt's {d} MiB and the wired margin {d} MiB\n", .{ ceiling >> 20, prompt >> 20, ctx.wired_margin_bytes >> 20 });
    // TURBOEXL3_GLM53F_DECODE_ROOM: what the phase change frees of the prompt's bill, past decode's own, for decode's rows.
    const decode_extra: u64 = if (sw.decode_room) prompt -| stream_room.decodeNeed(c, stream_room.room_seq, stream_room.room_chunk, 16, 0, device.naxAvailable()) else 0;
    // TURBOEXL3_GLM53F_BILL_FIT: the box's limit (the server's GPU ceiling, within the RAM) and the context limit, for
    // the fixed plan at the bank's open.
    const fit: ?Fit = if (sw.bill_fit) .{ .limit = @min(ceiling, system.getTotalMemBytes()), .context = context } else null;
    openBank(m, dir, room.bytes, room.source, sw, decode_extra, fit) catch |e| {
        out.* = refused("the streamed bank does not open", e);
        unload(mp);
        return null;
    };
    if (seed) |sf| log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SEED_DECODE=1: the resident seed read {s} (sha256 {s}, {d} B) in place of expert-usage-v1.json\n", .{ streamed.Glm5n.seed_decode_file, sf.sha[0..12], sf.bytes });
    // TURBOEXL3_GLM53F_KDA_PERCORE: the plan for this GPU's cores and one launch over zeros at it (its library built
    // here), before the warm and any request; no plan or launch refuses the load by name
    if (kda_prompt.percore_on) {
        const cores = system.gpuCoreCount();
        const pp = kda_prompt.percoreInstall(m.s, @intCast(c.linear_num_key_heads), @intCast(c.linear_num_value_heads), @intCast(c.linear_key_head_dim), @intCast(c.linear_value_head_dim), cores) catch |e| {
            out.* = refused("TURBOEXL3_GLM53F_KDA_PERCORE has no per-core route on this GPU", e);
            unload(mp);
            return null;
        };
        log.info(kda_prompt.percore_line, .{ pp.ntg, pp.rows, pp.per_core, cores });
    }
    // TURBOEXL3_GLM53F_KDA_CHUNK: the layer's heads and gate bound checked and one launch over zeros (its library built
    // here), before the warm and any request; another layout, bound or a refused launch refuses the load by name
    if (kda_prompt.chunk_on) {
        const groups = kda_prompt.chunkInstall(m.s, @intCast(c.linear_num_key_heads), @intCast(c.linear_num_value_heads), @intCast(c.linear_key_head_dim), @intCast(c.linear_value_head_dim), c.kda_gate_lower_bound) catch |e| {
            out.* = refused("TURBOEXL3_GLM53F_KDA_CHUNK has no chunkwise route for this layer", e);
            unload(mp);
            return null;
        };
        log.info(kda_prompt.chunk_line, .{ kda_prompt.chunk_rows, groups, kda_prompt.chunk_value_rows });
    }
    // TURBOEXL3_GLM53F_KDA_FOLD: one launch over zeros (its library built here), after the chunkwise route's install; a
    // refused launch refuses the load by name
    if (kda_prompt.fold_on) {
        kda_prompt.foldInstall(m.s, @intCast(c.linear_num_value_heads), @intCast(c.linear_conv_kernel_dim), c.kda_gate_lower_bound) catch |e| {
            out.* = refused("TURBOEXL3_GLM53F_KDA_FOLD has no folded launch for this layer", e);
            unload(mp);
            return null;
        };
        log.info(kda_prompt.fold_line, .{});
    }
    // TURBOEXL3_GLM53F_KDA_PIPE: the side stream on the model stream's GPU, after the chunkwise route's install; a
    // stream that is not a GPU's refuses the load by name
    if (kda_pipe.on) {
        kda_pipe.install(m.s) catch |e| {
            out.* = refused("TURBOEXL3_GLM53F_KDA_PIPE has no side stream beside this model's stream", e);
            unload(mp);
            return null;
        };
        log.info(kda_pipe.line, .{ kda_pipe.span_rows, kda_pipe.min_last_rows });
    }
    // TURBOEXL3_GLM53F_PF_MOE_HOLD: the hold's sizes from the routed geometry on the chain turboexl3 runs (the int8
    // chain's with PF_INT8, applied above) and its launch run once on the model's stream (its library built here), before
    // the warm and any request; a refused launch refuses the load by name
    if (pf_reuse.hold_on) {
        const held = pf_reuse.holdInstall(m.s, c.hidden_size, c.moe_intermediate_size, routed_t3.prompt_rows) catch |e| {
            out.* = refused("TURBOEXL3_GLM53F_PF_MOE_HOLD has no hold launch on this model's stream", e);
            unload(mp);
            return null;
        };
        log.info(pf_reuse.hold_line, .{ pf_reuse.heldArrays(), held >> 20, routed_t3.prompt_rows, @tagName(pf_reuse.heldChain()) });
        if (pf_reuse.hold_top) log.info(pf_reuse.hold_top_line, .{});
    }
    return mp;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    host.unload(mp);
    // TURBOEXL3_GLM53F_KDA_PIPE: its side stream goes with the model
    kda_pipe.deinit();
    // TURBOEXL3_GLM53F_PF_MOE_HOLD: its arrays, stream and launch too
    pf_reuse.holdReset();
    // TURBOEXL3_GLM53F_SMALLOP_FUSE: the dense layers' joined svh and the counts too
    smallop.reset();
    // TURBOEXL3_GLM53F_HEAD_LAST_ROW: the counts
    draft_step.reset();
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the counts
    f16_free.reset();
    // TURBOEXL3_GLM53F_LOGITS_DUMP: the file goes with the model
    logits_dump.close();
    // TURBOEXL3_GLM53F_KDA_EXPSIG's kept exp(A_log) and TURBOEXL3_GLM53F_CACHED_SCALARS' kept scalars go with the model
    kda_trim.reset();
    cached_scalars.reset();
    if (extra) |*e| if (e.mp == mp) {
        e.deinit();
        extra = null;
    };
}

/// The MTP layer's 288 experts (`mtp-experts.bin`) as the head's resident turboexl3 bank; a pack without a head keeps
/// none.
fn bindMtp(m: *trunk.Model, dir: []const u8, e: *Extra) !void {
    const h = if (m.mtp) |*x| x else return;
    const c = m.c;
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, mtp_bank.manifest_file, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const man = try mtp_bank.Manifest.parse(gpa, text, c.hidden_size, c.moe_intermediate_size, c.num_experts);
    const arrs = try mtp_bank.read(gpa, dir, &man);
    e.mtp = arrs;
    // components: code, rout (svh), rin (suh) per projection
    h.moe.x3 = .{
        .gate = .{ .trellis = arrs[0], .suh = arrs[2], .svh = arrs[1] },
        .up = .{ .trellis = arrs[3], .suh = arrs[5], .svh = arrs[4] },
        .down = .{ .trellis = arrs[6], .suh = arrs[8], .svh = arrs[7] },
    };
    log.info("[turboexl3_glm53f] MTP layer: {d} experts resident at {d} halfwords a tile ({d} MiB)\n", .{ man.n_experts, man.geometry.n, man.residentBytes() >> 20 });
}

/// The head as the engine's MTP readout reads it (`SpecHooks.trunk_weight`, a dense or affine `[vocab, hidden]`): the
/// identity's rows through turboexl3's EXL3 head give its weights (in bf16), quantized to 8-bit affine; drafts only, the
/// served logits stay turboexl3's (`Model.lmHead`).
fn readoutHead(m: *trunk.Model) ![3]Array {
    const p = m.head.x3 orelse return error.Glm53fHeadNotExl3;
    const s = m.s;
    const hidden: usize = m.c.hidden_size;
    const eye = try gpa.alloc(u16, hidden * hidden);
    defer gpa.free(eye);
    @memset(eye, 0);
    for (0..hidden) |i| eye[i * hidden + i] = 0x3F80; // bf16 1.0
    const x = mlx.mlx_array_new_data(eye.ptr, &[_]c_int{ @intCast(hidden), @intCast(hidden) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const w = try t3.linear(s, x, p.trellis, p.suh, p.svh, .mul1);
    defer _ = mlx.mlx_array_free(w);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, w, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wt, mlx.mlx_optional_int.some(@intCast(head_group)), mlx.mlx_optional_int.some(@intCast(head_bits)), "affine", .{}, s));
    var q: [3]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (q) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&q, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
    const vec = mlx.mlx_vector_array_new_data(&q, q.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    return q;
}

/// `SpecHooks.trunk_weight`: the embedding as the trunk holds it, and the readout head (`readoutHead`) for the head.
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    if (which != .head) return host.trunkWeight(mp, which, out);
    const e = extra orelse return Verdict.refuse(.unsupported, "turboexl3_glm53f: no readout head");
    const q = e.head_q orelse return Verdict.refuse(.unsupported, "turboexl3_glm53f: no readout head");
    out.* = .{ .w = .{ .ctx = q[0].ctx }, .scales = .{ .ctx = q[1].ctx }, .biases = .{ .ctx = q[2].ctx }, .bits = head_bits, .group_size = head_group, .layout = @backingInt(Layout.affine) };
    return .ok;
}

/// `TURBOEXL3_GLM53F_SEED_DECODE`'s profile in the pack (`Glm5n.seed_decode_file`): its bytes' sha256 in hex and its
/// size, for the load line.
const SeedFile = struct { sha: [64]u8, bytes: u64 };
fn seedDecodeFile(io: std.Io, dir: []const u8) !SeedFile {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, streamed.Glm5n.seed_decode_file, gpa, .limited(expert_usage.max_file_bytes));
    defer gpa.free(text);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(text);
    const digest = h.finalResult();
    var out: SeedFile = .{ .sha = undefined, .bytes = text.len };
    _ = std.fmt.bufPrint(&out.sha, "{x}", .{&digest}) catch unreachable;
    return out;
}

/// The bank switches (src/stream_room.zig `Switches`), from the environment at load.
fn readSwitches() !stream_room.Switches {
    const help = try stream_room.parseDemandHelp(envSpan("TURBOEXL3_GLM53F_DEMAND_HELP"));
    return .{
        .layer_bill = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_LAYER_BILL")),
        .decode_room = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_DECODE_ROOM")),
        .grow_unfilled = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_GROW_UNFILLED")),
        .prompt_keep = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_PROMPT_KEEP")),
        .spec_pick1 = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_SPEC_PICK1")),
        .demand_split = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_DEMAND_SPLIT")),
        .shared_hoist = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_SHARED_HOIST")),
        .read_ahead3 = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_READ_AHEAD3")),
        .demand_first = help,
        .adopt_pieces = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_ADOPT_PIECES")),
        .bill_fit = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_BILL_FIT")),
        .fit_inuse = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_FIT_INUSE")),
        .seed_decode = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_SEED_DECODE")),
        .keep_warm = try stream_room.parseSwitch(envSpan("TURBOEXL3_GLM53F_KEEP_WARM")),
    };
}

/// A switch's value: the environment's, or when unset the plugin's own (src/options.zig `defaultOf`, the control's).
fn envSpan(comptime name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else options.defaultOf(name);
}

/// `TURBOEXL3_GLM53F_LAYER_BILL`: the plan from each routed layer's own slot-row bytes (the pack's v2 manifest,
/// src/bank_layers.zig) over the bank's own per-layer split (`splitRows`).
fn layerPlan(dir: []const u8, geo: streamed.Glm5n.Geometry, free: u64, decode_free: u64, top_k: u32, n_experts: u32) !stream_room.Plan {
    const io = std.Io.Threaded.global_single_threaded.io();
    const n: usize = geo.num_moe_layers;
    const bytes = try gpa.alloc(u64, n);
    defer gpa.free(bytes);
    try bank_layers.read(gpa, io, dir, bytes);
    // no row past the bank's widest record (the geometry the stream allocates on)
    for (bytes) |b| if (b > geo.record_bytes) return error.Glm53fBankLayers;
    const pr = try gpa.alloc(u32, n);
    defer gpa.free(pr);
    const dr = try gpa.alloc(u32, n);
    defer gpa.free(dr);
    return stream_room.rowsByLayer(free, decode_free, bytes, streamed.Glm5n.transient_rows, stream_room.streamer_window_rows, top_k, n_experts, &splitRows, pr, dr);
}

/// The bank's per-layer split of a plan, as `Source.open` makes it.
fn splitRows(plan: stream_room.Plan, prompt: []u32, decode: []u32) anyerror!void {
    return streamed.Glm5n.planRows(.{ .prompt = plan.prompt, .decode = plan.decode }, prompt, decode);
}

/// `TURBOEXL3_GLM53F_BILL_FIT`'s inputs to the bank's open: the box's limit (the server's GPU ceiling) and the server's
/// context limit.
const Fit = struct { limit: u64, context: u64 };

/// The pack's bank bound to every routed layer, at the slot rows the box holds beside the loaded residents and one
/// prompt (`room`: the host's `LoadCtx.stream_room` or the plugin's own, `source`), the decode phase also in
/// `decode_extra` (`TURBOEXL3_GLM53F_DECODE_ROOM`), billed at the widest row or each layer's own (`sw`); with `fit`
/// (`TURBOEXL3_GLM53F_BILL_FIT`) both phases' bytes are the fixed plan's instead (src/stream_room.zig `fitPlan`, the
/// whole read pool billed in it): the built-in's `openBank` over this plugin's copy of the GLM-5.3-Flash source.
fn openBank(m: *host.Model, dir: []const u8, room: u64, source: stream_room.RoomSource, sw: stream_room.Switches, decode_extra: u64, fit: ?Fit) !void {
    // TURBOEXL3_GLM53F_SMALLOP_FUSE: the dense layers' and the DSA layers' joined svh made before the memory available is
    // read, so the bank's rows are planned beside them
    try registerSmallop(m);
    const io = std.Io.Threaded.global_single_threaded.io();
    const geo = try streamed.Glm5n.geometryOf(gpa, io, dir);
    const total = system.getTotalMemBytes();
    const available = system.getAvailableMemBytes();
    // TURBOEXL3_GLM53F_KEEP_WARM: the keep-warm engine's queue, kernel, buffer and parked thread (its bill term).
    const warm_bytes: u64 = if (sw.keep_warm) streamed.Glm5n.keep_warm_bill_bytes else 0;
    var free = room -| (total -| available) -| warm_bytes;
    var decode_free = free +| decode_extra;
    // TURBOEXL3_GLM53F_BILL_FIT: the fixed plan, the whole read pool at the served lookahead budget and the served
    // workers billed (the 4 workers' staging in `readPoolBytes`), and the box's non-file memory outside the server read
    // now (past the fit's allowance for it, it comes off both phases).
    var plan: ?stream_room.FitPlan = null;
    var pool: u64 = 0;
    var rest: u64 = 0;
    var allow = stream_room.fit_rest;
    // TURBOEXL3_GLM53F_FIT_INUSE: the non-file memory outside the server at load (`rest`, as the fit reads it) and the
    // part of it in use (`in_use`: its purgeable and reusable pages left out), for the load line
    var outside_raw: u64 = 0;
    if (fit) |f| {
        pool = stream_room.readPoolBytes(geo.record_bytes, std.heap.pageSize(), stream_room.streamer_spec_budget) +| warm_bytes;
        const box = box_memory.nonFileBytes();
        if (box == 0) {
            log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_BILL_FIT=1 reads the box's non-file memory when the bank opens, and the kernel gave none: the load is refused\n", .{});
            return error.Glm53fBillFitMemory;
        }
        rest = box -| @as(u64, system.getAppMemFootprintMb()) * (1 << 20);
        // TURBOEXL3_GLM53F_FIT_INUSE: the allowance is the memory outside the server in use now and the margin
        // (`stream_room.fitInUseAllowance`), and that memory is the rest, so nothing is over it
        if (sw.fit_inuse) {
            const in_use_box = box_memory.inUseNonFileBytes();
            if (in_use_box == 0) {
                log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_FIT_INUSE=1 reads the box's non-file memory in use when the bank opens, and the kernel gave none: the load is refused\n", .{});
                return error.Glm53fFitInUseMemory;
            }
            outside_raw = rest;
            rest = in_use_box -| @as(u64, system.getAppMemFootprintMb()) * (1 << 20);
            allow = stream_room.fitInUseAllowance(rest);
        }
        // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the trunk's freed f16 sources (0 without the switch) in both phases, the
        // allowance for the memory outside the server the fit's own
        const p = stream_room.fitPlanFreed(m.c, f.limit, f.context, device.naxAvailable(), pool, rest, sw.decode_room, allow, f16_free.freed);
        free = p.prompt;
        decode_free = p.decode;
        plan = p;
    }
    const rows = if (sw.layer_bill)
        try layerPlan(dir, geo, free, decode_free, m.c.num_experts_per_tok, m.c.num_experts)
    else
        stream_room.rowsForPhases(free, decode_free, geo.record_bytes, geo.num_moe_layers, streamed.Glm5n.transient_rows, m.c.num_experts_per_tok, m.c.num_experts);
    if (sw.layer_bill) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_LAYER_BILL=1: the slot rows billed at each routed layer's own row (expert-manifest-v2.json), the widest {d} B, with both geometries' transient and window rows\n", .{geo.record_bytes});
    if (sw.decode_room) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DECODE_ROOM=1: the decode phase's rows also in the prompt bill less the decode bill ({d} MiB): {d} decode rows a layer beside {d} prompt rows\n", .{ (decode_free -| free) >> 20, rows.decode, rows.prompt });
    // with TURBOEXL3_GLM53F_BILL_FIT the lines name the plan's limit (the ceiling) as the room
    const shown_room = if (fit) |f| f.limit else room;
    const shown_source: stream_room.RoomSource = if (fit != null) .fit else source;
    const gates = try gpa.alloc(streamed.Glm5n.Gate, m.layers.len);
    defer gpa.free(gates);
    for (gates, m.layers) |*g, *l| g.* = if (l.mlp == .moe) .{ .w = l.router_raw, .bias = l.mlp.moe.route.sigmoid_bias.expert_bias } else .{ .w = .{ .ctx = null }, .bias = .{ .ctx = null } };
    _ = mlx.mlx_synchronize(m.s);
    // TURBOEXL3_GLM53F_GROW_UNFILLED: the decode grow's new rows without the zero fill (the copy's GrowFill.unfilled).
    streamed.Glm5n.setGrowUnfilled(sw.grow_unfilled);
    // TURBOEXL3_GLM53F_PROMPT_KEEP: no prompt-phase admission evicts a resident (the copy's LayerPolicy.prompt_keep).
    streamed.Glm5n.setPromptKeep(sw.prompt_keep);
    // TURBOEXL3_GLM53F_SPEC_PICK1: one record read ahead a layer call, the read pool unchanged.
    streamed.Glm5n.setSpecPick1(sw.spec_pick1);
    // TURBOEXL3_GLM53F_DEMAND_SPLIT: a decode call's lone record's gate/up span read in pieces by the idle workers.
    streamed.Glm5n.setDemandSplit(sw.demand_split);
    // TURBOEXL3_GLM53F_SHARED_HOIST: the bank's Routed (below) takes a decode call's shared expert as its hoist.
    streamed.Glm5n.setSharedHoist(sw.shared_hoist);
    // TURBOEXL3_GLM53F_READ_AHEAD3: the bank's Routed (below) gives the trunk's dense layers the first routed layer's reads.
    streamed.Glm5n.setReadAhead3(sw.read_ahead3);
    // TURBOEXL3_GLM53F_DEMAND_HELP: the read pool's demand-first mode in decode (the kit stream's demandFirst).
    streamed.Glm5n.setDemandFirst(sw.demand_first);
    // TURBOEXL3_GLM53F_ADOPT_PIECES: an adopted range's copy into its rows in pieces, the idle workers taking some.
    streamed.Glm5n.setAdoptPieces(sw.adopt_pieces);
    // TURBOEXL3_GLM53F_SEED_DECODE: the resident seed reads the pack's decode-weighted profile (the copy's seedDecision).
    streamed.Glm5n.setSeedDecode(sw.seed_decode);
    // TURBOEXL3_GLM53F_KEEP_WARM: the keep-warm engine on the stream's waits (this plugin's Metal shim, billed above).
    streamed.Glm5n.setKeepWarm(sw.keep_warm);
    // The switches whose mechanism is a seam of the kit's stream: refused by name where this kit's stream lacks it
    // (src/streamed/glm5n/experts.zig `seams`; off by default there, src/options.zig)
    if (streamed.Glm5n.missingSeam()) |name| {
        log.err("[turboexl3_glm53f] {s} is set, but this kit's expert stream (kit.streamed) has no seam for it: the load is refused (set it to 0, or build against a kit whose stream has it)\n", .{name});
        return error.Glm5NextStreamSeam;
    }
    const b = streamed.Glm5n.Source.open(gpa, io, dir, m.s, gates, geo.num_moe_layers, geo.num_experts, .{ .prompt = rows.prompt, .decode = rows.decode }) catch |e| {
        log.err("[turboexl3_glm53f] the streamed bank did not open at {d} prompt / {d} decode slot rows per layer: room {d} MiB ({s}), total {d} MiB, available {d} MiB, free {d} MiB, device pool {} ({s})\n", .{ rows.prompt, rows.decode, shown_room >> 20, @tagName(shown_source), total >> 20, available >> 20, free >> 20, mlx.weightsInDeviceMemory(), @errorName(e) });
        return e;
    };
    // the memory the rows were sized from (`LoadCtx.stream_room` less the box's memory in use; with
    // TURBOEXL3_GLM53F_BILL_FIT, the fixed plan's: src/stream_room.zig)
    log.info("[turboexl3_glm53f] streamed experts, {d} prompt / {d} decode slot rows per layer ({d} B the widest row): room {d} MiB ({s}), total {d} MiB, available {d} MiB, free {d} MiB, device pool {}\n", .{ rows.prompt, rows.decode, geo.record_bytes, shown_room >> 20, @tagName(shown_source), total >> 20, available >> 20, free >> 20, mlx.weightsInDeviceMemory() });
    if (sw.grow_unfilled) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_GROW_UNFILLED=1: the decode grow's new slot rows take their buffers without the zero fill (GrowFill.unfilled: a row is read only after its record landed)\n", .{});
    if (sw.prompt_keep) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PROMPT_KEEP=1: no prompt-phase admission evicts a resident: a prompt miss takes an empty prompt row, else a transient row, and the read-ahead takes empty rows only\n", .{});
    if (sw.spec_pick1) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SPEC_PICK1=1: the decode lookahead reads ahead one record a layer call, the most likely of the next layer's non-resident predictions (default {d}), so a call's second guess no longer shares the drive with its first and with the demand reads; the read pool keeps its {d} speculative slots and threads and its bytes; reads only, no slot or row count changes\n", .{ stream_room.streamer_spec_budget, 2 * stream_room.streamer_spec_budget });
    if (sw.demand_first != 0) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DEMAND_HELP={d}: the read pool puts decode's demand first (the kit stream's demandFirst({d}), set at each decode phase and back to 0 at each prompt phase): an unclaimed lookahead record reads only while no demand job is queued or running, in pieces, and a demand range that finds its record claimed and still in flight reads the record's untaken pieces itself; the same bytes into the same rows; no slot or row count changes\n", .{ sw.demand_first, sw.demand_first });
    if (sw.adopt_pieces) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ADOPT_PIECES=1: a demand read served from a lookahead record copies its bytes into its rows in up to {d} pieces (one a read pool worker), the workers with no read queued taking some, where the copy is one memcpy by the worker that adopts it; the same bytes into the same rows; no slot or row count changes\n", .{stream_room.streamer_io_workers});
    if (sw.read_ahead3) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_READ_AHEAD3=1: a decode window's first routed layer (model layer {d}) is scored on the last dense layer's MLP input and its likely records are read ahead before its route, as the lookahead reads every later layer's; speculative reads only, the routes and the outputs unchanged; no slot or row count changes\n", .{streamed.Glm5n.first_moe_layer});
    if (sw.shared_hoist) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SHARED_HOIST=1: a decode call's shared expert is built before its routed call and committed behind its hit wave, ahead of its miss waves, when the call misses, so the GPU runs it while the misses' bytes land; the same ops and the same add; no slot or row count changes\n", .{});
    if (sw.demand_split) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DEMAND_SPLIT=1: a decode call that reads one record reads its gate/up span as up to {d} pieces, a read pool job each, beside its down span ({d} workers), so the idle workers read it side by side; the same bytes into the same rows, its gate released after its last piece; no slot or row count changes\n", .{ streamed.Glm5n.demandSplitPieces(stream_room.streamer_io_workers), stream_room.streamer_io_workers });
    if (sw.keep_warm) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KEEP_WARM=1: the keep-warm engine on the expert stream's waits (the copy's gpu_warm at waits, over this plugin's own Metal shim, src/csrc/glm53f_gpu_warm.m): one simdgroup's FMA chain on its own command queue in ~0.5 ms chunks while a gated wait (a decode miss part, a prompt group's reads) is below the stream's event or the host waits on a read-ahead; no model array, graph or MLX queue touched; {d} B billed: the slot rows were sized from that much less\n", .{streamed.Glm5n.keep_warm_bill_bytes});
    if (fit) |f| {
        const p = plan.?;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_BILL_FIT=1: rows fixed by the fit (sc_3.10's measured peaks), not by the memory available: the box's non-file memory planned to peak at the {d} MiB ceiling for any request up to {d} tokens: the prompt phase's base {d} MiB + {d} MiB to the limit, the decode phase's {d} MiB + {d} MiB, the hot prefix cache {d} MiB, the read pool {d} B; outside the server at load {d} MiB of non-file memory against the fit's {d} MiB: {d} MiB over, taken off both phases\n", .{ f.limit >> 20, f.context, p.base_prompt >> 20, p.grow_prompt >> 20, p.base_decode >> 20, p.grow_decode >> 20, stream_room.hot_cache_bytes >> 20, pool, rest >> 20, allow >> 20, p.excess >> 20 });
        if (sw.fit_inuse) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_FIT_INUSE=1: BILL_FIT's allowance for the memory outside the server is what is in use there at load, {d} MiB (the non-file {d} MiB less its purgeable and reusable pages), and the {d} MiB margin: {d} MiB in place of the fit's fixed {d} MiB, so each phase's rows take {d} MiB more\n", .{ rest >> 20, outside_raw >> 20, stream_room.fit_inuse_margin >> 20, allow >> 20, stream_room.fit_rest >> 20, @as(i64, @intCast(@max(outside_raw, stream_room.fit_rest) >> 20)) - @as(i64, @intCast(allow >> 20)) });
        // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the bases the plan took (the fit's, with its allowance, less the trunk's freed f16
        // sources)
        if (f16_free.on) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_TRUNK_F16_FREE: the fit's bases less the {d} MiB the trunk's f16 sources freed: the prompt phase's {d} MiB, the decode phase's {d} MiB, so {d} prompt / {d} decode slot rows per layer under the same {d} MiB ceiling\n", .{ p.freed >> 20, (p.base_prompt -| p.freed) >> 20, (p.base_decode -| p.freed) >> 20, rows.prompt, rows.decode, f.limit >> 20 });
    }
    m.bank = b;
    const r = streamed.glm5nRouted(b);
    for (m.layers, 0..) |*l, li| if (l.mlp == .moe) {
        l.mlp.moe.streamed = .{ .routed = r, .layer = @intCast(li) };
    };
    // TURBOEXL3_GLM53F_ROUTER: each routed layer's raw router (the gates' bf16 [288, 4096]) for its decode calls
    fused_router.reset();
    if (fused_router.on) for (m.layers) |*l| if (l.mlp == .moe) try fused_router.register(&l.mlp.moe, l.router_raw);
    // TURBOEXL3_GLM53F_MTP_ROUTER: the MTP head's raw router (loaded with the head with the switch), after the trunk's
    if (fused_router.on and fused_router.mtp_on) if (m.mtp) |*h| {
        if (h.router_raw.ctx == null) {
            log.err("[turboexl3_glm53f] Glm53fMtpRouterRaw: TURBOEXL3_GLM53F_MTP_ROUTER=1 and the MTP head loaded no raw router (a streamed pack's head names mlp.gate.weight): the load is refused\n", .{});
            return error.Glm53fMtpRouterRaw;
        }
        try fused_router.registerHead(&h.moe, h.router_raw);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_MTP_ROUTER: the MTP head's router {s}\n", .{if (fused_router.headRegistered()) "registered after the trunk's" else "not in ROUTER's form: the head routes as before"});
    };
}

/// TURBOEXL3_GLM53F_SMALLOP_FUSE: every dense layer's gate and up (the trunk's first layers) registered for `denseMlp`'s
/// one launch (their svh joined: 48 KB a layer), and every DSA layer's q_a and kv_a (the trunk's and the MTP layer's) for
/// `dsaAttn`'s (4 KB a layer). A projection that is not EXL3 is not registered (its two linears run as before); two that do
/// not pair refuse the load by name.
fn registerSmallop(m: *host.Model) !void {
    smallop.reset();
    if (!smallop.on) return;
    for (m.layers, 0..) |*l, li| {
        if (l.mlp == .dense) {
            const dw = &l.mlp.dense;
            if (dw.gate.x3) |gate| if (dw.up.x3) |up| smallop.registerDense(m.s, gate, up) catch |e| {
                log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE=1: layer {d}'s dense gate and up do not pair ({s}): the load is refused\n", .{ li, @errorName(e) });
                return e;
            };
        }
        if (l.mixer == .dsa) try registerQkv(m.s, &l.mixer.dsa, li);
    }
    if (m.mtp) |*h| try registerQkv(m.s, &h.dsa, m.layers.len);
    log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE: {d} dense layers and {d} DSA layers registered, their svh joined: {d} + {d} B\n", .{ smallop.denseLayers(), smallop.qkvLayers(), smallop.denseBytes(), smallop.qkvBytes() });
}

fn registerQkv(s: mlx.mlx_stream, d: anytype, layer: usize) !void {
    const q_a = d.q_a.x3 orelse return;
    const kv_a = d.kv_a.x3 orelse return;
    smallop.registerQkv(s, q_a, kv_a) catch |e| {
        log.err("[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE=1: layer {d}'s DSA q_a and kv_a do not pair ({s}): the load is refused\n", .{ layer, @errorName(e) });
        return e;
    };
}

/// The pack held as stored beside the bank's fewest slot rows and transient rows (at the widest record), the MTP
/// layer's resident bank and the readout head.
fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    const c = config.of(p);
    const io = std.Io.Threaded.global_single_threaded.io();
    const geo = streamed.Glm5n.geometryOf(std.heap.page_allocator, io, model_dir.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    const mtp = mtpBytes(io, model_dir.slice(), c) catch |e| return Verdict.named(.failed, @errorName(e));
    // 8-bit codes, and a bf16 scale and bias a group of 64
    const head_q = @as(u64, c.vocab_size) * c.hidden_size * (head_bits * head_group + 2 * 16) / (8 * head_group);
    out.* = shards +| (@as(u64, c.num_experts_per_tok) * geo.num_moe_layers + streamed.Glm5n.transient_rows) * geo.record_bytes +| mtp +| head_q;
    return .ok;
}

fn mtpBytes(io: std.Io, dir: []const u8, c: *const config.Config) !u64 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, mtp_bank.manifest_file, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const man = try mtp_bank.Manifest.parse(gpa, text, c.hidden_size, c.moe_intermediate_size, c.num_experts);
    return man.residentBytes();
}

/// The conformance suite's tiny pack: none (a streamed EXL3 pack has no small form; plugin.json's `fixture_reason`).
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return error.NoFixture;
}

test {
    _ = config;
    // the switches' options and the kit's rules over them (by the stream test's filter)
    _ = options;
    // build.zig's stream_options test (rooted here, by name)
    _ = @import("src/stream_options_test.zig");
    // TURBOEXL3_GLM53F_KEEP_WARM's engine on its trace backend (by the stream test's filter)
    _ = @import("src/keep_warm_test.zig");
    _ = @import("src/banked_rows_test.zig");
    _ = @import("src/row_plan_test.zig");
    // the split prompt forward's refusals and spans (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch)
    _ = @import("src/split_test.zig");
    // the stream math's switch test (by the filter's name)
    _ = routed_t3;
    _ = stream_room;
    _ = bank_layers;
    // TURBOEXL3_GLM53F_FIT_INUSE's in-use pages (by the filter's name)
    _ = box_memory;
    // the KDA prompt texts' switches and layouts (by the filter's name)
    _ = kda_prompt;
    // the KDA verify texts' switch and verify widths (by the filter's name)
    _ = kda_verify;
    // TURBOEXL3_GLM53F_KDA_LAZY_SEQ's summary (by the filter's name)
    _ = kda_lazy;
    // the hyper-connection text's switch and layouts (by the filter's name)
    _ = hc_fuse;
    // the async cadence eval's switch (by the stream test's filter)
    _ = async_eval;
    // the per-layer tail eval's cadence (by the stream test's filter)
    _ = tail_eval;
    // the prompt buffer reuse's schedule (by the stream test's filter)
    _ = pf_reuse;
    // TURBOEXL3_GLM53F_LOGITS_DUMP's record read-back (by the stream test's filter)
    _ = logits_dump;
    // the KDA prompt spans' plan (by the stream test's filter)
    _ = kda_pipe;
    // the KDA prompt texts' and the hyper-connection text's device parity (build.zig `kda-gpu-test`, by its own filter's
    // names)
    _ = @import("src/kda_gpu_test.zig");
    _ = @import("src/hc_gpu_test.zig");
    // TURBOEXL3_GLM53F_ROUTER's texts and checks (by the filter's name) and its device parity (build.zig
    // `router-gpu-test`, by its own filter's name)
    _ = fused_router;
    _ = @import("src/router_gpu_test.zig");
    // TURBOEXL3_GLM53F_MOE_COMBINE1's and TURBOEXL3_GLM53F_ROUTE_VIEW's admissions (by the filter's name) and their device
    // parity (build.zig `moe-combine-gpu-test`, by its own filter's name)
    _ = moe_combine1;
    _ = route_view;
    _ = @import("src/moe_combine_gpu_test.zig");
    // TURBOEXL3_GLM53F_PF_MOE_HOLD's fit to the routed chain's own arrays (the moe-combine device test's filter)
    _ = @import("src/pf_hold_gpu_test.zig");
    // TURBOEXL3_GLM53F_SMALLOP_FUSE's admission and registry (by the filter's name) and its device parity (build.zig
    // `smallop-gpu-test`, by its own filter's name)
    _ = smallop;
    _ = @import("src/smallop_gpu_test.zig");
    // TURBOEXL3_GLM53F_DRAFT_S0's admission and request line (by the filter's name); its device parity is
    // smallop-gpu-test's
    _ = draft_s0;
    // the prepared-input producers' admission and request line (by the filter's name) and their device parity (build.zig
    // `dec-prep-gpu-test`, by its own filter's name)
    _ = dec_prep;
    _ = @import("src/dec_prep_gpu_test.zig");
    // the head's draft-step lines (by the filter's name) and TURBOEXL3_GLM53F_HEAD_LAST_ROW's device check (build.zig
    // `draft-gpu-test`, by its own filter's name)
    _ = draft_step;
    _ = @import("src/draft_gpu_test.zig");
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE's load line (by the filter's name); its plan is stream_room's test
    _ = f16_free;
    // the DSA prompt texts' switches and DSA_LFUSE's window (by the filter's name), and the DSA texts' device checks
    // (build.zig `dsa-gpu-test`, by its own filter's name)
    _ = dsa_vattn;
    _ = @import("src/dsa_vattn_gpu_test.zig");
    _ = dsa_lfuse;
    _ = @import("src/dsa_lfuse_gpu_test.zig");
    _ = dsa_prompt;
    _ = dsa_verify;
    _ = @import("src/dsa_gpu_test.zig");
    // TURBOEXL3_GLM53F_DSA_GLM's plan (by the filter's name) and its device checks (build.zig `dsa-gpu-test`, by its own
    // filter's name)
    _ = dsa_glm;
    _ = @import("src/dsa_glm_gpu_test.zig");
    // TURBOEXL3_GLM53F_DSA_HOLD's holder (by the filter's name)
    _ = dsa_hold;
    // the launch trims' texts, refusals and lines (by the filter's name); their device checks are kda-gpu-test's and
    // smallop-gpu-test's
    _ = kda_trim;
    _ = cached_scalars;
    _ = host_zero;
    // TURBOEXL3_GLM53F_DSA_TAILS', _DSA_ROWS' and _DSA_WFOLD's admissions (by the filter's name) and their device checks
    // (build.zig `dsa-gpu-test`, by its own filter's name)
    _ = dsa_tails;
    _ = dsa_rows;
    _ = dsa_wfold;
    _ = @import("src/dsa_vpath_gpu_test.zig");
}

test "turboexl3_glm53f stream stats: each request's prompt and decode lines are the stream's counters since each phase's first forward" {
    try stream_stats.expectPhases(StreamStats);
}

