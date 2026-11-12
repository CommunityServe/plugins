//! turboexl3_glm53f: GLM-5.3-Flash's exllamav3 (EXL3) streaming packs, as a quant plugin. mlx-serve's own glm5_next
//! serves the pack; this plugin loads the trellis linears the pack stores and the MTP layer's experts
//! (`mtp-experts.bin`), runs them on turboexl3's Metal kernels, and streams the routed experts from the pack's
//! `experts.bin` (`open_bank`: the GLM bank in src/streamed/ and src/exl3/ over the kit's expert stream, module
//! `mlx_stream`). The bank's modules reach this file as `mlx_host` for the MLX layer, the log and the switches.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.quant;
const t3 = @import("turboexl3");
const tkit = @import("turboexl3_kit");
const stream = @import("mlx_stream");
const glm5n = stream.glm5n_source;
const shims = @import("src/host_shims.zig");
const mtp_bank = @import("src/mtp_bank.zig");
const bank_rows = @import("src/bank_rows.zig");
const bank_layers = @import("src/bank_layers.zig");
const options = @import("src/options.zig");
const stream_stats = @import("src/stream_stats.zig");
pub const config = @import("config.zig");

// ── what the bank's modules read from `mlx_host` ──

pub const mlx = kit.mlx;
pub const log = @import("src/host/log.zig");
pub const env = @import("src/host/env.zig");
pub const shared = @import("src/host/shared.zig");
pub const quant_abi = abi;
pub const routed_t3 = @import("src/routed_t3.zig");
/// The model's levers on the bank's calls (the fused router, the one-launch combine, the route view): mlx-serve's
/// glm5_next carries none of them, so the bank takes its stock paths.
pub const fused_router = struct {
    pub fn predict(_: mlx.mlx_stream, _: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_array) !?mlx.mlx_array {
        return null;
    }
};
pub const moe_combine1 = struct {
    pub const on = false;
    pub fn combine(_: mlx.mlx_stream, _: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_array) !?mlx.mlx_array {
        return null;
    }
};
pub const route_view = struct {
    pub fn view(_: mlx.mlx_stream, _: mlx.mlx_array) !?mlx.mlx_array {
        return null;
    }
};
pub const mtp_acceptance = struct {};

/// Shown at install and in Settings.
pub const version = "0.4.0";

pub const plugin = kit.sdk.Plugin{ .name = "turboexl3_glm53f", .kind = .quant, .options = &options.forKit(@typeInfo(@FieldType(kit.sdk.Plugin, "options")).pointer.child) };

pub const table = abi.QuantTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .validate = &config.validate,
    .bill = &config.bill,
    .open = &open,
    .close = &close,
    .load_tensor = &loadTensor,
    .load_bank = &loadBank,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
    .swiglu = &swiglu,
    .bank_rows = &bankRows,
    .open_bank = &openBank,
};

comptime {
    // The row bill's constants are the streamer's.
    if (glm5n.transient_rows != bank_rows.streamer_transient_rows) @compileError("src/bank_rows.zig's streamer_transient_rows is not the streamer's transient_rows");
    if (glm5n.decode_lane_rows * 8 != bank_rows.streamer_window_rows) @compileError("src/bank_rows.zig's streamer_window_rows is not the streamer's decode window");
}

const Verdict = abi.Verdict;
const A = mlx.mlx_array;
const gpa = std.heap.c_allocator;
const io = std.Io.Threaded.global_single_threaded.io();
/// The one codebook `validate` admits.
const dec: t3.format.Decode = .mul1;
const missing = Verdict.refuse(.failed, "MissingWeight");
const oom = Verdict.refuse(.failed, "turboexl3_glm53f: out of memory");
/// The bank's prompt GEMMs and turboexl3's prefill run on MLX's NAX tiles: below the M5 they compute wrong.
const no_nax = Verdict.refuse(.unsupported, "turboexl3_glm53f: an M5 or newer GPU (NAX) is required");

/// A switch's value: the environment's, or when unset the plugin's own (src/options.zig `defaultOf`).
fn envSpan(comptime name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else options.defaultOf(name);
}

fn refused(comptime what: []const u8, e: anyerror) Verdict {
    log.err("[turboexl3_glm53f] " ++ what ++ ": {s}\n", .{@errorName(e)});
    return Verdict.named(.failed, @errorName(e));
}

/// Kernels, logs and the stream's MLX shims through the server's table; turboexl3's schedules set before any of its
/// kernels is built, as the switches name them.
fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    const h = kit.host.Host.of(ctx.host) catch |e| return Verdict.named(.unsupported, @errorName(e));
    _ = config.codebookOf(ctx.block_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_glm53f: not an exllamav3 mul1 block");
    if (!shims.install(ctx.host)) return Verdict.refuse(.unsupported, "turboexl3_glm53f: the server's host table is older than this plugin's kit");
    tkit.bind(h.t);
    log.use(h);
    if (!tkit.nax.available()) return no_nax;
    configure(.{ .ctx = ctx.stream.ctx }) catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

/// turboexl3's banked and dense decode schedules, the routed prefill's options and int8 scheme, the trunk's prefill tile
/// codes (their kernels compiled on `s`), and the bank's prompt staging; each refused by name at a bad value.
fn configure(s: mlx.mlx_stream) !void {
    if (try routed_t3.BankedSchedule.read(&routed_t3.processEnv)) |bs| {
        try bs.apply();
        log.info(routed_t3.banked_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_BANKED").?, bs.textSet(), bs.gu, bs.dn });
    }
    if (try routed_t3.DecSchedule.read(&routed_t3.processEnv)) |d| {
        try d.apply();
        log.info(routed_t3.dec_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_DEC").?, t3.dense.dec_variants[d.variant].name });
    }
    if (try routed_t3.PfOpts.read(&routed_t3.processEnv)) |o| {
        try o.apply();
        log.info(routed_t3.pf_opts_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PF_OPTS").?, o.bk128 });
    }
    if (try routed_t3.PfInt8.read(&routed_t3.processEnv)) |p| {
        try p.apply();
        log.info(routed_t3.pf_int8_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PF_INT8").?, p.words() });
    }
    if (try routed_t3.Pg2.read(&routed_t3.processEnv)) |p| {
        p.apply();
        try routed_t3.Pg2.warm(s);
        log.info(routed_t3.pg2_line, .{ routed_t3.processEnv("TURBOEXL3_GLM53F_PG2").?, p.n, t3.glm53f.trunk.len });
    }
    const stage = try routed_t3.parseSwitch(envSpan("TURBOEXL3_GLM53F_PF_STAGE"));
    glm5n.setStageMisses(stage);
    if (stage) log.info(routed_t3.pf_stage_line, .{});
    const drain_lag = try routed_t3.parseSwitch(envSpan("TURBOEXL3_GLM53F_PF_DRAIN_LAG"));
    if (drain_lag and !try routed_t3.parseSwitch(envSpan("TURBOEXL3_GLM53F_PF_FEED"))) {
        log.err("[turboexl3_glm53f] Glm53fPfDrainLagWithoutFeed: TURBOEXL3_GLM53F_PF_DRAIN_LAG needs TURBOEXL3_GLM53F_PF_FEED=1\n", .{});
        return error.Glm53fPfDrainLagWithoutFeed;
    }
    glm5n.setDrainLag(drain_lag);
    routed_t3.drain_lag_on = drain_lag;
    if (drain_lag) log.info(routed_t3.pf_drain_lag_line, .{});
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

/// `<base>.trellis` (U16 [in/16, out/16, n]) beside `<base>.suh` (f16 [in]) and `<base>.svh` (f16 [out]), as exllamav3
/// stores a linear.
fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const p = (partsOf(ctx, name.slice()) catch return Verdict.refuse(.failed, "turboexl3_glm53f: a name too long")) orelse return missing;
    if (!shaped(p, in_dim)) return not_shaped;
    out.* = .{ .w = share(p[0]), .scales = share(p[1]), .biases = share(p[2]) };
    return .ok;
}

const not_shaped = Verdict.refuse(.unsupported, "turboexl3_glm53f: not an EXL3 linear of the expected input width (trellis [in/16, out/16, n], suh f16 [in], svh f16 [out])");

/// `p` (trellis, suh, svh) as exllamav3 stores a linear of `in_dim` inputs (0: any).
fn shaped(p: [3]A, in_dim: u32) bool {
    const t = mlx.getShape(p[0]);
    const u = mlx.getShape(p[1]);
    const v = mlx.getShape(p[2]);
    if (t.len != 3 or u.len != 1 or v.len != 1 or u[0] != t[0] * 16 or v[0] != t[1] * 16) return false;
    if (in_dim != 0 and u[0] != in_dim) return false;
    return mlx.mlx_array_dtype(p[1]) == .float16 and mlx.mlx_array_dtype(p[2]) == .float16;
}

/// `base`'s trellis, suh and svh as the source holds them (borrowed); null when the pack has no such linear.
fn partsOf(ctx: *const abi.LoadCtx, base: []const u8) !?[3]A {
    var out: [3]A = undefined;
    for (&out, [_][]const u8{ "trellis", "suh", "svh" }) |*a, part| {
        var buf: [192]u8 = undefined;
        a.* = .{ .ctx = ctx.source.tensor(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, part })).ctx };
        if (a.ctx == null) return null;
    }
    return out;
}

/// A new handle on `a`.
fn share(a: A) abi.Array {
    var h = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&h, a);
    return .{ .ctx = h.ctx };
}

/// The MTP layer's routed experts `<head>.mlp.experts.<proj>` (the head's names in glm5_next, `mtp.0.block`) as one
/// resident bank read from the pack's `mtp-experts.bin`: code U16 [experts, in/16, out/16, n], suh and svh f16.
fn loadBank(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const full = name.slice();
    const proj: usize = for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, i| {
        if (std.mem.endsWith(u8, full, p)) break i;
    } else return missing;
    if (std.mem.indexOf(u8, full, ".mtp.0.block.mlp.experts.") == null) return missing;
    const geo = headGeometry(ctx.config_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_glm53f: the config names no hidden_size, moe_intermediate_size or n_routed_experts");
    const want_in: u32 = if (proj == 2) geo.inter else geo.hidden;
    if (in_dim != 0 and in_dim != want_in) return not_shaped;
    const model_dir = ctx.model_dir.slice();
    var d = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch |e| return refused("the pack's folder does not open", e);
    defer d.close(io);
    const text = d.readFileAlloc(io, mtp_bank.manifest_file, gpa, .limited(64 << 20)) catch |e| return refused("the MTP bank's manifest does not read", e);
    defer gpa.free(text);
    const man = mtp_bank.Manifest.parse(gpa, text, geo.hidden, geo.inter, geo.experts) catch |e| return refused("the MTP bank's manifest", e);
    const arrs = mtp_bank.readProj(gpa, model_dir, &man, proj) catch |e| return refused("the MTP bank does not read", e);
    // A record's segments are code, rout (exllamav3 svh), rin (suh).
    out.* = .{ .w = .{ .ctx = arrs[0].ctx }, .scales = .{ .ctx = arrs[2].ctx }, .biases = .{ .ctx = arrs[1].ctx } };
    return .ok;
}

const HeadGeometry = struct { hidden: u32, inter: u32, experts: u32 };

fn headGeometry(config_json: []const u8) ?HeadGeometry {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const T = struct { hidden_size: u32 = 0, moe_intermediate_size: u32 = 0, n_routed_experts: u32 = 0 };
    const S = struct { text_config: T = .{}, hidden_size: u32 = 0, moe_intermediate_size: u32 = 0, n_routed_experts: u32 = 0 };
    const c = std.json.parseFromSliceLeaky(S, arena.allocator(), config_json, .{ .ignore_unknown_fields = true }) catch return null;
    const t = c.text_config;
    const g: HeadGeometry = .{
        .hidden = if (t.hidden_size > 0) t.hidden_size else c.hidden_size,
        .inter = if (t.moe_intermediate_size > 0) t.moe_intermediate_size else c.moe_intermediate_size,
        .experts = if (t.n_routed_experts > 0) t.n_routed_experts else c.n_routed_experts,
    };
    return if (g.hidden == 0 or g.inter == 0 or g.experts == 0 or g.experts > 4096) null else g;
}

fn release(w: *abi.Weight) callconv(.c) void {
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
    };
    w.* = .{};
}

fn arr(a: abi.Array) A {
    return .{ .ctx = a.ctx };
}

fn projOf(w: *const abi.Weight) t3.Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const p = projOf(w);
    const y = t3.linear(.{ .ctx = s.ctx }, arr(x), p.trellis, p.suh, p.svh, dec) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3_glm53f: routed experts run through swiglu");
}

/// The MTP head's routed experts (`inds` set) on turboexl3's routed MoE, clamped at `limit`; a dense SwiGLU runs the
/// caller's chain.
fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (inds.ctx == null) return Verdict.refuse(.unsupported, "turboexl3_glm53f: a dense SwiGLU runs the caller's chain");
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (opts.limit > 0)
        t3.moeClamped(st, arr(x), bank, arr(inds), arr(scores), dec, opts.limit)
    else
        t3.moe(st, arr(x), bank, arr(inds), arr(scores), dec, opts.verify_rows)) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

// ── the streamed bank (`experts.bin`) ──

/// The bank switches, from the environment (or the plugin's defaults) at the bank's open.
fn readSwitches() !bank_rows.Switches {
    return .{
        .layer_bill = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_LAYER_BILL")),
        .grow_unfilled = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_GROW_UNFILLED")),
        .prompt_keep = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_PROMPT_KEEP")),
        .spec_pick1 = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_SPEC_PICK1")),
        .demand_first = try bank_rows.parseDemandHelp(envSpan("TURBOEXL3_GLM53F_DEMAND_HELP")),
        .adopt_pieces = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_ADOPT_PIECES")),
        .seed_decode = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_SEED_DECODE")),
        .keep_warm = try bank_rows.parseSwitch(envSpan("TURBOEXL3_GLM53F_KEEP_WARM")),
    };
}

const Planned = struct { rows: bank_rows.Plan, billed: bank_rows.Billed };

/// The rows `budget` buys over the bank in `dir`: each phase's bytes less the keep-warm engine's, billed at each routed
/// layer's own row (`LAYER_BILL`) or the widest; and the bytes each phase bills (the keep-warm engine's in both).
fn plan(dir: []const u8, sw: bank_rows.Switches, geo: glm5n.Geometry, budget: abi.BankBudget, top_k: u32) !Planned {
    const warm: u64 = if (sw.keep_warm) glm5n.keep_warm_bill_bytes else 0;
    const free = budget.prompt -| warm;
    const decode_free = budget.decode -| warm;
    if (!sw.layer_bill) {
        const p = bank_rows.rowsForPhases(free, decode_free, geo.record_bytes, geo.num_moe_layers, glm5n.transient_rows, top_k, geo.num_experts);
        const b = bank_rows.billedUniform(p, geo.record_bytes, geo.num_moe_layers, glm5n.transient_rows);
        return .{ .rows = p, .billed = .{ .prompt = b.prompt + warm, .decode = b.decode + warm } };
    }
    const n: usize = geo.num_moe_layers;
    const bytes = try gpa.alloc(u64, n);
    defer gpa.free(bytes);
    try bank_layers.read(gpa, io, dir, bytes);
    for (bytes) |b| if (b > geo.record_bytes) return error.Glm53fBankLayers;
    const pr = try gpa.alloc(u32, n);
    defer gpa.free(pr);
    const dr = try gpa.alloc(u32, n);
    defer gpa.free(dr);
    const p = try bank_rows.rowsByLayer(free, decode_free, bytes, glm5n.transient_rows, bank_rows.streamer_window_rows, top_k, geo.num_experts, &splitRows, pr, dr);
    const b = bank_rows.billedByLayer(p, dr, bytes, glm5n.transient_rows, bank_rows.streamer_window_rows);
    return .{ .rows = p, .billed = .{ .prompt = b.prompt + warm, .decode = b.decode + warm } };
}

fn splitRows(p: bank_rows.Plan, prompt: []u32, decode: []u32) anyerror!void {
    return glm5n.planRows(.{ .prompt = p.prompt, .decode = p.decode }, prompt, decode);
}

fn bankRefusal(e: anyerror) Verdict {
    return if (e == error.StreamedRowsDoNotFit) Verdict.refuse(.too_large, "StreamedRowsDoNotFit") else Verdict.named(.failed, @errorName(e));
}

fn bankRows(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.BankRows) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const sw = readSwitches() catch |e| return Verdict.named(.unsupported, @errorName(e));
    const g = glm5n.geometryOf(gpa, io, ctx.model_dir.slice()) catch |e| return bankRefusal(e);
    const p = (plan(ctx.model_dir.slice(), sw, g, budget, geo.top_k) catch |e| return bankRefusal(e)).rows;
    out.* = .{ .prompt = p.prompt, .decode = p.decode, .row_bytes = g.record_bytes * g.num_moe_layers };
    return .ok;
}

/// The load line's account of the one bank a process streams.
var about: [160]u8 = undefined;

fn openBank(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const sw = readSwitches() catch |e| return Verdict.named(.unsupported, @errorName(e));
    glm5n.setGrowUnfilled(sw.grow_unfilled);
    glm5n.setPromptKeep(sw.prompt_keep);
    glm5n.setSpecPick1(sw.spec_pick1);
    glm5n.setDemandFirst(sw.demand_first);
    glm5n.setAdoptPieces(sw.adopt_pieces);
    glm5n.setSeedDecode(sw.seed_decode);
    glm5n.setKeepWarm(sw.keep_warm);
    if (glm5n.missingSeam()) |name| {
        log.err("[turboexl3_glm53f] {s} is set, but this kit's expert stream (kit.streamed) has no seam for it: the bank is refused (set it to 0)\n", .{name});
        return Verdict.named(.unsupported, "Glm5NextStreamSeam");
    }
    const dir = ctx.model_dir.slice();
    // TURBOEXL3_GLM53F_SEED_DECODE: the profile's sha256 for its line (the open below refuses a pack without it by name)
    const seed: ?SeedFile = if (sw.seed_decode) seedDecodeFile(dir) catch null else null;
    const g = glm5n.geometryOf(gpa, io, dir) catch |e| return bankRefusal(e);
    const planned = plan(dir, sw, g, budget, geo.top_k) catch |e| return bankRefusal(e);
    const p = planned.rows;
    const b = Bank.open(ctx, geo, g, .{ .prompt = p.prompt, .decode = p.decode }) catch |e| return bankRefusal(e);
    const line = std.fmt.bufPrint(&about, "{d} prompt / {d} decode slot rows per layer ({d} B the widest row)", .{ p.prompt, p.decode, g.record_bytes }) catch "";
    engaged(sw, g, seed);
    // a bank-fit.json base is a phase's measured peak less these
    log.info("[turboexl3_glm53f] bank bytes: prompt {d} B, decode {d} B (budget {d} / {d} B)\n", .{ planned.billed.prompt, planned.billed.decode, budget.prompt, budget.decode });
    out.* = .{
        .ctx = b,
        .close = &Bank.close,
        .about = .of(line),
        .decode_ids = glm5n.decode_lane_rows * geo.top_k,
        .rows = &Bank.rows,
        .settle = &Bank.settle,
        .phase = &Bank.phase,
        .read_ahead = &Bank.readAhead,
    };
    // `[stream-stats]` where the kit's Bank carries the request's summary lines
    if (comptime @hasField(abi.Bank, "meters")) out.meters = &Bank.meters;
    // A prompt call's rows each run by themselves (turboexl3's prompt GEMMs reduce a row over its own K only, in an order
    // the call's other rows do not move): a split prompt forward may run two prompt forwards' rows in one call.
    if (comptime @hasField(abi.Bank, "rows_independent")) out.rows_independent = true;
    return .ok;
}

/// The bank switches' lines at the bank's open, as G2 prints them, so a served run's log shows which engaged.
fn engaged(sw: bank_rows.Switches, g: glm5n.Geometry, seed: ?SeedFile) void {
    if (seed) |sf| log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SEED_DECODE=1: the resident seed read {s} (sha256 {s}, {d} B) in place of expert-usage-v1.json\n", .{ glm5n.seed_decode_file, sf.sha[0..12], sf.bytes });
    if (sw.layer_bill) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_LAYER_BILL=1: the slot rows billed at each routed layer's own row (expert-manifest-v2.json), the widest {d} B, with both geometries' transient and window rows\n", .{g.record_bytes});
    if (sw.grow_unfilled) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_GROW_UNFILLED=1: the decode grow's new slot rows take their buffers without the zero fill (GrowFill.unfilled: a row is read only after its record landed)\n", .{});
    if (sw.prompt_keep) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_PROMPT_KEEP=1: no prompt-phase admission evicts a resident: a prompt miss takes an empty prompt row, else a transient row, and the read-ahead takes empty rows only\n", .{});
    if (sw.spec_pick1) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SPEC_PICK1=1: the decode lookahead reads ahead one record a layer call, the most likely of the next layer's non-resident predictions (default {d}); the read pool keeps its {d} speculative slots and threads and its bytes; reads only, no slot or row count changes\n", .{ bank_rows.streamer_spec_budget, 2 * bank_rows.streamer_spec_budget });
    if (sw.demand_first != 0) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DEMAND_HELP={d}: the read pool puts decode's demand first (the kit stream's demandFirst({d}), set at each decode phase and back to 0 at each prompt phase); the same bytes into the same rows; no slot or row count changes\n", .{ sw.demand_first, sw.demand_first });
    if (sw.adopt_pieces) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_ADOPT_PIECES=1: a demand read served from a lookahead record copies its bytes into its rows in up to {d} pieces (one a read pool worker); the same bytes into the same rows; no slot or row count changes\n", .{bank_rows.streamer_io_workers});
    if (sw.keep_warm) log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KEEP_WARM=1: the keep-warm engine on the expert stream's waits (gpu_warm at waits, over this plugin's own Metal shim, src/csrc/glm53f_gpu_warm.m); no model array, graph or MLX queue touched; {d} B billed: the slot rows were sized from that much less\n", .{glm5n.keep_warm_bill_bytes});
}

/// `TURBOEXL3_GLM53F_SEED_DECODE`'s profile in the pack (`glm5n.seed_decode_file`): its bytes' sha256 in hex and its
/// size, for its line.
const SeedFile = struct { sha: [64]u8, bytes: u64 };
fn seedDecodeFile(dir: []const u8) !SeedFile {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, glm5n.seed_decode_file, gpa, .limited(stream.expert_usage.max_file_bytes));
    defer gpa.free(text);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(text);
    const digest = h.finalResult();
    var out: SeedFile = .{ .sha = undefined, .bytes = text.len };
    _ = std.fmt.bufPrint(&out.sha, "{x}", .{&digest}) catch unreachable;
    return out;
}

/// The kit's stream counters (`kit.streamed.expert.Stats`), named through the streamer's module: naming the source's own
/// type pulls its MLX kernels into the MLX-free test binary.
const StreamStats = stream.expert_stats;
/// `[stream-stats]` (src/stream_stats.zig): one streamed bank a process.
var meter: stream_stats.Meter(StreamStats) = .{};

/// The GLM bank and the routers it predicts with: each routed layer's router as the bank reads it (bf16
/// `[experts, hidden]`, from the trunk's f32 `[hidden, experts]`: the same values), held for the bank's life.
const Bank = struct {
    src: *glm5n.Source,
    gates: []glm5n.Gate,

    fn open(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, g: glm5n.Geometry, slot_rows: glm5n.Rows) !*Bank {
        const given = geo.gates orelse return error.Glm53fBankGates;
        const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
        const gates = try gpa.alloc(glm5n.Gate, geo.n_layers);
        @memset(gates, .{ .w = .{ .ctx = null }, .bias = .{ .ctx = null } });
        errdefer freeGates(gates);
        for (gates, given[0..geo.n_layers]) |*o, x| {
            if (x.w.ctx == null) continue;
            o.* = .{ .w = try routerAsStored(arr(x.w), s), .bias = share2(arr(x.bias)) };
        }
        const src = try glm5n.Source.open(gpa, io, ctx.model_dir.slice(), s, gates, g.num_moe_layers, g.num_experts, slot_rows);
        errdefer src.deinit();
        const b = try gpa.create(Bank);
        b.* = .{ .src = src, .gates = gates };
        return b;
    }

    fn freeGates(gates: []glm5n.Gate) void {
        for (gates) |x| for ([_]A{ x.w, x.bias }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        gpa.free(gates);
    }

    fn of(ctx: ?*anyopaque) *Bank {
        return @ptrCast(@alignCast(ctx.?));
    }
    fn done(r: anyerror!void) Verdict {
        r catch |e| return Verdict.named(.failed, @errorName(e));
        return .ok;
    }
    fn close(ctx: ?*anyopaque) callconv(.c) void {
        const b = of(ctx);
        b.src.deinit();
        freeGates(b.gates);
        gpa.destroy(b);
    }
    fn rows(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        const y = of(ctx).src.rows(layer, arr(x), arr(idx), arr(wts)) catch |e| return Verdict.named(.failed, @errorName(e));
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    fn settle(ctx: ?*anyopaque) callconv(.c) Verdict {
        return done(of(ctx).src.settle());
    }
    fn phase(ctx: ?*anyopaque, prompt: bool) callconv(.c) Verdict {
        const b = of(ctx);
        meter.mark(prompt, b.src.inner);
        return done(b.src.setPhase(prompt));
    }
    /// The request's `[stream-stats]` lines: its prompt phase's and its decode phase's counters.
    fn meters(ctx: ?*anyopaque, out: *abi.Str) callconv(.c) void {
        const b = of(ctx);
        out.* = abi.Str.of(meter.lines(b.src.inner.stats(), stream_stats.Extra.of(b.src.inner)));
    }
    fn readAhead(ctx: ?*anyopaque, layer: u32, ids: [*]const u16, n: usize) callconv(.c) Verdict {
        return done(of(ctx).src.readAhead(layer, ids[0..n]));
    }
};

/// The trunk's router `w` (f32 `[hidden, experts]`) as the pack stores it: bf16 `[experts, hidden]`, evaluated.
fn routerAsStored(w: A, s: mlx.mlx_stream) !A {
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose(&t, w, s));
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&b, t, .bfloat16, s));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_contiguous(&o, b, false, s));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// A new handle on `a`, or none for none.
fn share2(a: A) A {
    if (a.ctx == null) return a;
    var h = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&h, a);
    return h;
}

pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"glm5_next\"}" }}, .block = "{\"quant_method\":\"exl3\",\"codebook\":\"mul1\"}" };
}

test "turboexl3_glm53f: [stream-stats] splits the stream's own counters at the phase change" {
    try stream_stats.expectPhases(StreamStats);
}

test {
    _ = config;
    _ = bank_rows;
    _ = options;
    _ = mtp_bank;
    _ = bank_layers;
    _ = routed_t3;
}
