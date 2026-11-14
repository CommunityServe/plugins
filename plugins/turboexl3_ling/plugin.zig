//! turboexl3_ling: Ling-3.1-flash's exllamav3 (EXL3) streaming packs, as a quant plugin. mlx-serve's own bailing_hybrid
//! serves the pack (its forward: KDA, MLA, the grouped MoE, the head); this plugin loads the trunk's trellis linears the
//! pack stores and runs them on turboexl3's Metal kernels, and streams the routed experts of layers 4..71 from the pack's
//! `experts.bin` over the kit's expert stream (`open_bank`, src/ling_bank.zig), their math on turboexl3. The template is
//! turboexl3_ds41.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.quant;
const t3 = @import("turboexl3");
const tkit = @import("turboexl3_kit");
const shims = @import("src/host_shims.zig");
const clamps = @import("src/clamps.zig");
const ling_math = @import("src/ling_math.zig");
const log = @import("src/host/log.zig");
const LingSource = @import("src/ling_bank.zig");
const bank_rows = @import("src/bank_rows.zig");
pub const config = @import("config.zig");

const mlx = kit.mlx;

/// Shown at install and in Settings.
pub const version = "0.2.0";

pub const plugin = kit.sdk.Plugin{ .name = "turboexl3_ling", .kind = .quant, .options = &.{} };

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
    .swiglu_limit_out = true,
    // `matmul` at up to DECODE_ROWS_MAX rows runs turboexl3's decode GEMV (DEC-ONE to 4 rows, the chain past it, the
    // one-row table at one row), whose every row has its one-row call's bytes at each of Ling's dense shapes
    // (turboexl3's DEC-ONE byte test with those shapes and the served one-row launch). Past it the prompt GEMM.
    .exact_rows = t3.kernels.DECODE_ROWS_MAX,
};

const Verdict = abi.Verdict;
const A = mlx.mlx_array;
const gpa = std.heap.c_allocator;
/// The one codebook `validate` admits.
const dec: t3.format.Decode = .mul1;
const missing = Verdict.refuse(.failed, "MissingWeight");
const oom = Verdict.refuse(.failed, "turboexl3_ling: out of memory");
/// turboexl3's prompt kernels and the bank's run on MLX's matrix units (NAX): an M5-class GPU.
const no_nax = Verdict.refuse(.unsupported, "turboexl3_ling: an M5-class GPU (NAX) is needed (turboexl3's prompt kernels run on its matrix units)");

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    const h = kit.host.Host.of(ctx.host) catch |e| return Verdict.named(.unsupported, @errorName(e));
    // The pack again (validate's process may not be this one): the codebook, the bank's files, the clamps.
    config.checkPack(ctx.block_json.slice(), ctx.config_json.slice(), ctx.dir) catch |e| return Verdict.named(.unsupported, @errorName(e));
    // Kernels from the server's cache, launched through its launcher; MLX's matrix units only where its NAX gate holds.
    tkit.bind(h.t);
    // The kit stream's events, slot buffers and launches, and the plugin's log lines, through the server's table.
    if (!shims.install(ctx.host)) return Verdict.refuse(.unsupported, "turboexl3_ling: the server's host table is older than this plugin's kit");
    log.use(h);
    if (!tkit.nax.available()) return no_nax;
    // turboexl3's configuration as the served stack sets it (src/ling_math.zig `configure`): the banked decode's grouped
    // texts on t4, the one-row table, the fused routed finish (PF_DFIN).
    ling_math.configure() catch |e| return Verdict.named(.failed, @errorName(e));
    return .ok;
}

fn close(_: *abi.LoadCtx) callconv(.c) void {}

/// `<base>.trellis` (I16 [in/16, out/16, n]) beside `<base>.suh` (f16 [in]) and `<base>.svh` (f16 [out]), as exllamav3
/// stores a linear (KDA's joined qkv_proj and the head included).
fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const p = (partsOf(ctx, name.slice()) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long")) orelse return missing;
    if (!shaped(p, in_dim)) return not_shaped;
    out.* = .{ .w = share(p[0]), .scales = share(p[1]), .biases = share(p[2]) };
    return .ok;
}

const not_shaped = Verdict.refuse(.unsupported, "turboexl3_ling: not an EXL3 linear of the expected input width (trellis [in/16, out/16, n], suh f16 [in], svh f16 [out])");

/// `p` (trellis, suh, svh) as exllamav3 stores a linear of `in_dim` inputs (0: any).
fn shaped(p: [3]A, in_dim: u32) bool {
    const t = mlx.getShape(p[0]);
    const u = mlx.getShape(p[1]);
    const v = mlx.getShape(p[2]);
    if (t.len != 3 or u.len != 1 or v.len != 1 or u[0] != t[0] * 16 or v[0] != t[1] * 16) return false;
    if (in_dim != 0 and u[0] != in_dim) return false;
    return mlx.mlx_array_dtype(p[1]) == .float16 and mlx.mlx_array_dtype(p[2]) == .float16;
}

/// The routed experts `<prefix>.<proj>` as one resident bank (the MTP head's, when mlx-serve loads Ling's head): the
/// pack's per-expert linears `<prefix>.<e>.<proj>` stacked on a new first axis, the trellis read as u16 (the same bits).
fn loadBank(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const full = name.slice();
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return missing;
    const v = stackParts(ctx, full[0..dot], full[dot..], in_dim, out);
    if (!v.isOk()) return v;
    // Every expert the layer's router scores (`<layer>.mlp.gate.weight` [experts, hidden]): an expert missing from the
    // pack would end the stack early, and a route past its end reads another tensor's bytes.
    const routed = routerRows(ctx, full[0..dot]);
    if (routed == null or routed.? != mlx.getShape(arr(out.w))[0]) {
        release(out);
        return Verdict.refuse(.unsupported, "turboexl3_ling: the bank holds another count of experts than its router scores");
    }
    return .ok;
}

/// The experts the router beside `<layer>.mlp.experts` scores; null without one.
fn routerRows(ctx: *const abi.LoadCtx, experts: []const u8) ?c_int {
    if (!std.mem.endsWith(u8, experts, ".experts")) return null;
    var buf: [192]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}.gate.weight", .{experts[0 .. experts.len - ".experts".len]}) catch return null;
    const g: A = .{ .ctx = ctx.source.tensor(name).ctx };
    if (g.ctx == null) return null;
    const sh = mlx.getShape(g);
    return if (sh.len == 2) sh[0] else null;
}

/// The linears `<prefix>.<i><suffix>` for i = 0.. stacked on a new first axis, the trellis viewed as u16.
fn stackParts(ctx: *const abi.LoadCtx, prefix: []const u8, suffix: []const u8, in_dim: u32, out: *abi.Weight) Verdict {
    var parts: [3]std.ArrayList(A) = .{ .empty, .empty, .empty };
    defer for (&parts) |*l| l.deinit(gpa);
    var e: usize = 0;
    while (true) : (e += 1) {
        var buf: [192]u8 = undefined;
        const each = std.fmt.bufPrint(&buf, "{s}.{d}{s}", .{ prefix, e, suffix }) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long");
        const p = (partsOf(ctx, each) catch return Verdict.refuse(.failed, "turboexl3_ling: a name too long")) orelse break;
        if (!shaped(p, in_dim)) return not_shaped;
        for (&parts, p) |*l, a| l.append(gpa, a) catch return oom;
    }
    if (e == 0) return missing;
    var stacked: [3]A = @splat(.{ .ctx = null });
    for (&stacked, &parts) |*s, *l| s.* = stack(ctx, l.items) catch |err| {
        for (stacked) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        }
        return Verdict.named(.failed, @errorName(err));
    };
    var trellis = mlx.mlx_array_new();
    const viewed = mlx.mlx_view(&trellis, stacked[0], .uint16, .{ .ctx = ctx.stream.ctx }) == 0 and mlx.mlx_array_eval(trellis) == 0;
    _ = mlx.mlx_array_free(stacked[0]);
    stacked[0] = trellis;
    if (!viewed) {
        for (stacked) |x| _ = mlx.mlx_array_free(x);
        return Verdict.refuse(.failed, "turboexl3_ling: the bank's trellis does not view as u16");
    }
    out.* = .{ .w = .{ .ctx = stacked[0].ctx }, .scales = .{ .ctx = stacked[1].ctx }, .biases = .{ .ctx = stacked[2].ctx } };
    return .ok;
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

/// `xs` stacked on a new first axis and evaluated (each evaluated first: a stack waiting on hundreds of queued disk
/// reads times out).
fn stack(ctx: *const abi.LoadCtx, xs: []const A) !A {
    for (xs) |x| try mlx.check(mlx.mlx_array_eval(x));
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_stack_axis(&o, vec, 0, .{ .ctx = ctx.stream.ctx }));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// A new handle on `a`.
fn share(a: A) abi.Array {
    var h = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&h, a);
    return .{ .ctx = h.ctx };
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

/// Every trunk linear the pack stores in EXL3 (KDA's qkv / f / g / o projections, MLA's, the dense layers', the shared
/// experts' and the head): turboexl3's `linear` (DEC-ONE at decode rows on the one-row table, the fused GEMM chain at
/// prompt rows).
fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = t3.linear(st, arr(x), arr(w.w), arr(w.scales), arr(w.biases), dec) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3_ling: routed experts run through swiglu");
}

/// Resident routed experts (a format bank: the MTP head's) on turboexl3's Ling SwiGLU. `Swiglu.limit_out` (> 0) is
/// Ling-3.1's clamp after the SiLU, which the table declares (`swiglu_limit_out`); `Swiglu.limit` is a clamp on the gate's
/// input (GLM's), which Ling never asks, so a call with it is refused.
fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    if (inds.ctx == null) return Verdict.refuse(.unsupported, "turboexl3_ling: a dense SwiGLU runs the caller's chain");
    if (opts.limit != 0) return Verdict.refuse(.unsupported, "turboexl3_ling: a gate-input clamp (Ling-3.1 clamps after the SiLU)");
    const limit = clamps.ofHost(limitOut(opts)) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = t3.moeLing(st, arr(x), bank, arr(inds), arr(scores), dec, @intCast(limit)) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

/// `opts.limit_out` where the caller's struct reaches it (a host built before the field passes a shorter one: none).
fn limitOut(opts: *const abi.Swiglu) f32 {
    return if (opts.size >= @offsetOf(abi.Swiglu, "limit_out") + @sizeOf(f32)) opts.limit_out else 0;
}

// ── the streamed bank (`experts.bin`) ──

/// The bank's geometry: the host's (`BankGeometry`, every trunk layer) past the pack's dense layers, with each routed
/// layer's clamp after the SiLU: the host's (`limitOut`) where its geometry reaches `limits_out`, checked against the
/// pack's config.json list (src/clamps.zig `routedFrom`); the config's list alone on a host built before the field.
/// `limits` holds `n_layers - first` entries.
fn geometry(geo: *const abi.BankGeometry, l: config.Ling, limits: []c_int) !LingSource.Geometry {
    if (geo.n_layers != l.layers or l.first >= geo.n_layers) return error.LingBankGeometry;
    const n_moe = geo.n_layers - l.first;
    if (limits.len != n_moe) return error.LingBankGeometry;
    const host = geo.size >= @offsetOf(abi.BankGeometry, "limits_out") + @sizeOf(?[*]const f32);
    for (limits, l.first..) |*o, li| {
        const layer: u32 = @intCast(li);
        o.* = @intCast(if (host) try clamps.routedFrom(layer, geo.limitOut(layer)) else clamps.routed(layer));
    }
    return .{ .hidden_size = geo.hidden, .moe_intermediate_size = geo.inter, .n_experts_per_tok = geo.top_k, .n_routed_experts = geo.n_experts, .n_layers = n_moe, .limits = limits };
}

fn budgetOf(b: abi.BankBudget) LingSource.Budget {
    // TODO(box): the host's room is bailing_hybrid's prefill_bytes (the engine's arithmetic, not the model plugin's
    // memfit); the served bank had 214 prompt / 236 decode rows at 18K. Re-fit from the new shape's [mem] peaks.
    return .{ .prompt = b.prompt, .decode = b.decode };
}

fn bankRefusal(e: anyerror) Verdict {
    return if (e == error.StreamedRowsDoNotFit) Verdict.refuse(.too_large, "StreamedRowsDoNotFit") else Verdict.named(.failed, @errorName(e));
}

fn bankRows(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.BankRows) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const l = config.lingOf(ctx.config_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_ling: not a Ling config");
    const limits = gpa.alloc(c_int, geo.n_layers -| l.first) catch return oom;
    defer gpa.free(limits);
    const g = geometry(geo, l, limits) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const b = LingSource.billOf(gpa, ctx.model_dir.slice(), g, budgetOf(budget)) catch |e| return bankRefusal(e);
    out.* = .{ .prompt = b.rows.prompt, .decode = b.rows.decode, .row_bytes = b.row_bytes };
    return .ok;
}

/// The load line's account of the one bank a process streams.
var about: [160]u8 = undefined;

fn openBank(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const src = openSource(ctx, geo, budget) catch |e| return bankRefusal(e);
    const line = std.fmt.bufPrint(&about, "{d} prompt / {d} decode slot rows per layer ({d} B a row)", .{ src.bill.rows.prompt, src.bill.rows.decode, src.bill.row_bytes }) catch "";
    log.info("[turboexl3_ling] streamed experts: {s}\n", .{line});
    // The bank's bytes in each phase (its rows, windows and staging) beside the budget it planned from: a bank-fit
    // calibration's input (`bank-fit.json`'s base = the phase's peak less these).
    const b = src.bill;
    log.info("[turboexl3_ling] bank bytes: prompt {d} B, decode {d} B (budget {d} / {d} B)\n", .{ bank_rows.phaseBytes(b, .prompt), bank_rows.phaseBytes(b, .decode), budget.prompt, budget.decode });
    out.* = .{
        .ctx = src,
        .close = &Bank.close,
        .about = .of(line),
        .decode_ids = @intCast(LingSource.decode_route_ids),
        .rows = &Bank.rows,
        .settle = &Bank.settle,
        .phase = &Bank.phase,
        .read_ahead = &Bank.readAhead,
        .meters = &Bank.meters,
        // A decode call's pairs each take moeLing's top-1 bytes at weight 1, combined row by row: a row's bytes are its
        // own call's (the host's exact verify calls one decode lane at a time).
        .exact_rows = true,
    };
    return .ok;
}

fn openSource(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget) !*LingSource.Source {
    const l = config.lingOf(ctx.config_json.slice()) orelse return error.LingBankGeometry;
    const limits = try gpa.alloc(c_int, geo.n_layers -| l.first);
    defer gpa.free(limits);
    const g = try geometry(geo, l, limits);
    // PF_DFIN's whole-call rule reads the model's routing.
    ling_math.routing = .{ .topk = g.n_experts_per_tok, .experts = g.n_routed_experts };
    return LingSource.Source.open(gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }, g, l.first, budgetOf(budget));
}

/// `abi.Bank`'s entries over the source.
const Bank = struct {
    fn of(ctx: ?*anyopaque) *LingSource.Source {
        return @ptrCast(@alignCast(ctx.?));
    }
    fn done(r: anyerror!void) Verdict {
        r catch |e| return Verdict.named(.failed, @errorName(e));
        return .ok;
    }
    fn close(ctx: ?*anyopaque) callconv(.c) void {
        of(ctx).deinit();
    }
    fn rows(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        const y = of(ctx).rows(layer, arr(x), arr(idx), arr(wts)) catch |e| return Verdict.named(.failed, @errorName(e));
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    fn settle(ctx: ?*anyopaque) callconv(.c) Verdict {
        return done(of(ctx).settle());
    }
    fn phase(ctx: ?*anyopaque, prompt: bool) callconv(.c) Verdict {
        return done(of(ctx).setPhase(if (prompt) .prompt else .decode));
    }
    fn readAhead(ctx: ?*anyopaque, layer: u32, ids: [*]const u16, n: usize) callconv(.c) Verdict {
        return done(of(ctx).readAhead(layer, ids[0..n]));
    }
    fn meters(ctx: ?*anyopaque, out: *abi.Str) callconv(.c) void {
        out.* = .of(of(ctx).meters());
    }
};

pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"bailing_hybrid\",\"num_hidden_layers\":2,\"first_k_dense_replace\":1}" }}, .block = "{\"quant_method\":\"exl3\",\"codebook\":\"mul1\"}" };
}
