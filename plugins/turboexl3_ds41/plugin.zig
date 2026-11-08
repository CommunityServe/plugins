//! turboexl3_ds41: DeepSeek-V4.1-Flash's exllamav3 (EXL3) packs, as a quant plugin. mlx-serve's own deepseek_v41
//! serves the pack; this plugin loads the trellis linears and the DSpark stages' expert banks the pack stores and runs
//! them on turboexl3's Metal kernels, and streams the routed experts from the pack's `experts.bin` (`open_bank`: the
//! expert streamer and its EXL3 lane in src/, module `mlx_stream`).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.quant;
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const tkit = @import("turboexl3_kit");
const woaj = @import("src/woa_join.zig");
const shared_swiglu = @import("src/shared_swiglu.zig");
const stream = @import("mlx_stream");
const exl3 = stream.exl3_source;
const shims = @import("src/host_shims.zig");
pub const config = @import("config.zig");

/// Shown at install and in Settings.
pub const version = "0.2.0";

pub const plugin = kit.sdk.Plugin{ .name = "turboexl3_ds41", .kind = .quant, .options = &.{
    .{ .name = "TURBOEXL3_DS41_FRAC", .kind = .bool, .default = "1", .help = "Serve layers at a half-integer rate (off: a pack with them is refused)" },
    .{ .name = "TURBOEXL3_DS41_RESIDENT_SEED", .kind = .text, .help = "usage: fill the prompt rows with each layer's most used experts at load; none: no seed. Empty: by the pack" },
    .{ .name = "TURBOEXL3_DS41_EXPERT_USAGE_DIR", .kind = .text, .help = "The folder the seed reads expert-usage-v1.json from (absolute). Empty: the pack's" },
    .{ .name = "TURBOEXL3_DS41_EXPERT_USAGE_RECORD", .kind = .text, .help = "Count each request's routed experts into this file (absolute path)" },
    .{ .name = "TURBOEXL3_DS41_EXPERT_USAGE_LABEL", .kind = .text, .help = "The recorded prompt set's name (needs the record)" },
} };

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

const Verdict = abi.Verdict;
const A = mlx.mlx_array;
const gpa = std.heap.c_allocator;
const decodes = [_]t3.format.Decode{ .mul1, .mcg };
const missing = Verdict.refuse(.failed, "MissingWeight");

fn open(ctx: *abi.LoadCtx) callconv(.c) Verdict {
    const h = kit.host.Host.of(ctx.host) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const cb = config.codebookOf(ctx.block_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_ds41: not an exllamav3 block");
    // Kernels from the server's cache, launched through its launcher; MLX's matrix units only where its NAX gate holds.
    tkit.bind(h.t);
    // The streamer's events, slot buffers and launches, and its log lines, through the server's table.
    if (!shims.install(ctx.host)) return Verdict.refuse(.unsupported, "turboexl3_ds41: the server's host table is older than this plugin's kit");
    stream.sdk.log.use(h);
    const nax = tkit.nax.available();
    // A streaming pack's bank needs NAX: refused here, before its trunk loads.
    if (!nax) if (ctx.dir) |d| if (d.fileBytes("experts.bin") > 0) return no_nax;
    // The banked decode: up to four row-slots of one expert a weight decode, the M = 3 member path, the t4 GEMVs.
    var cf = t3.config.get().*;
    cf.nax = nax;
    cf.banked_group = 4;
    cf.banked_m3 = true;
    cf.banked_variant_gu = t3.groupVariantOf("t4") orelse return Verdict.refuse(.failed, "turboexl3_ds41: turboexl3 has no t4 GEMV variant");
    cf.banked_variant_dn = cf.banked_variant_gu;
    t3.configure(cf) catch |e| return Verdict.named(.failed, @errorName(e));
    woaj.build(gpa) catch |e| return Verdict.named(.failed, @errorName(e));
    // Prompt rows' matmul and finish as one launch at the tiles turboexl3 tables (pg2: the same words), compiled now.
    if (nax) {
        t3.pgemm.mode.get().* = .on;
        const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
        t3.pgemm.warm(s, .bfloat16, null) catch |e| return Verdict.named(.failed, @errorName(e));
        t3.pgemm.warm(s, .float32, .{ 5120, 512 }) catch |e| return Verdict.named(.failed, @errorName(e));
    }
    ctx.state = @ptrCast(@constCast(&decodes[@intFromEnum(cb)]));
    return .ok;
}

fn close(ctx: *abi.LoadCtx) callconv(.c) void {
    ctx.state = null;
}

fn layoutOf(ctx: *const abi.LoadCtx) u8 {
    const d: *const t3.format.Decode = @ptrCast(@alignCast(ctx.state.?));
    return @intCast(d - &decodes[0]);
}

/// `<base>.trellis` (I16 [in/16, out/16, n]) beside `<base>.suh` (f16 [in]) and `<base>.svh` (f16 [out]), as exllamav3
/// stores a linear; a grouped one (`wo_a`) as its slices `<base>.slice.<g>`, stacked (`grouped` set).
fn loadTensor(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const p = (partsOf(ctx, name.slice()) catch return Verdict.refuse(.failed, "turboexl3_ds41: a name too long")) orelse return loadGroups(ctx, name.slice(), in_dim, out);
    const sh = mlx.getShape(p[0]);
    if (sh.len != 3 or (in_dim != 0 and sh[0] * 16 != in_dim)) return Verdict.refuse(.unsupported, "turboexl3_ds41: an EXL3 linear of another input width");
    out.* = .{ .w = share(p[0]), .scales = share(p[1]), .biases = share(p[2]), .layout = layoutOf(ctx) };
    return .ok;
}

/// `Weight.flags`: a grouped linear (its x is [rows, groups, in]).
const grouped: u32 = 1;

fn loadGroups(ctx: *const abi.LoadCtx, base: []const u8, in_dim: u32, out: *abi.Weight) Verdict {
    var buf: [192]u8 = undefined;
    const prefix = std.fmt.bufPrint(&buf, "{s}.slice", .{base}) catch return Verdict.refuse(.failed, "turboexl3_ds41: a name too long");
    const v = stackParts(ctx, prefix, "", in_dim, false, out);
    if (!v.isOk()) return v;
    out.flags = grouped;
    // The joined run's kernels compiled at the first grouped linear's load, for the trunk's bf16 rows.
    if (!woa_warmed) woaj.warm(.{ .ctx = ctx.stream.ctx }, projOf(out), .bfloat16) catch |e| return Verdict.named(.failed, @errorName(e));
    woa_warmed = true;
    return .ok;
}

var woa_warmed = false;

/// The routed experts `<prefix>.<proj>` as one bank: the pack's per-expert linears `<prefix>.<e>.<proj>` stacked on a
/// new first axis, the trellis read as u16 (the same bits).
fn loadBank(ctx: *const abi.LoadCtx, name: abi.Str, in_dim: u32, out: *abi.Weight) callconv(.c) Verdict {
    const full = name.slice();
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return missing;
    return stackParts(ctx, full[0..dot], full[dot..], in_dim, true, out);
}

/// The linears `<prefix>.<i><suffix>` for i = 0.. stacked on a new first axis, the trellis viewed as u16 on `u16`.
fn stackParts(ctx: *const abi.LoadCtx, prefix: []const u8, suffix: []const u8, in_dim: u32, u16_view: bool, out: *abi.Weight) Verdict {
    var parts: [3]std.ArrayList(A) = .{ .empty, .empty, .empty };
    defer for (&parts) |*l| l.deinit(gpa);
    var e: usize = 0;
    while (true) : (e += 1) {
        var buf: [192]u8 = undefined;
        const each = std.fmt.bufPrint(&buf, "{s}.{d}{s}", .{ prefix, e, suffix }) catch return Verdict.refuse(.failed, "turboexl3_ds41: a name too long");
        const p = (partsOf(ctx, each) catch return Verdict.refuse(.failed, "turboexl3_ds41: a name too long")) orelse break;
        if (in_dim != 0 and mlx.getShape(p[0])[0] * 16 != in_dim) return Verdict.refuse(.unsupported, "turboexl3_ds41: an EXL3 linear of another input width");
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
    if (u16_view) {
        var trellis = mlx.mlx_array_new();
        const viewed = mlx.mlx_view(&trellis, stacked[0], .uint16, .{ .ctx = ctx.stream.ctx }) == 0 and mlx.mlx_array_eval(trellis) == 0;
        _ = mlx.mlx_array_free(stacked[0]);
        stacked[0] = trellis;
        if (!viewed) {
            for (stacked) |x| _ = mlx.mlx_array_free(x);
            return Verdict.refuse(.failed, "turboexl3_ds41: the bank's trellis does not view as u16");
        }
    }
    out.* = .{ .w = .{ .ctx = stacked[0].ctx }, .scales = .{ .ctx = stacked[1].ctx }, .biases = .{ .ctx = stacked[2].ctx }, .layout = layoutOf(ctx) };
    return .ok;
}

const oom = Verdict.refuse(.failed, "turboexl3_ds41: out of memory");

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

/// A grouped linear's decoded weights, kept across one layer's runs (`woaj.run`); dropped for another weight's, at a
/// decode call, or with the weight.
var hats: woaj.Hats = .{};
var hats_of: ?*anyopaque = null;

fn dropHats() void {
    hats.deinit();
    hats = .{};
    hats_of = null;
}

fn release(w: *abi.Weight) callconv(.c) void {
    if (hats_of != null and hats_of == w.w.ctx) dropHats();
    for ([_]abi.Array{ w.w, w.scales, w.biases, w.side }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
    };
    w.* = .{};
}

fn arr(a: abi.Array) A {
    return .{ .ctx = a.ctx };
}

fn matmul(w: *const abi.Weight, x: abi.Array, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    const y = (if (w.flags & grouped != 0) groupedMatmul(w, arr(x), st) else t3.linear(st, arr(x), arr(w.w), arr(w.scales), arr(w.biases), decodes[w.layout])) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

/// x [rows, groups, in] through each group's linear: a prompt run on one prepare and one finish launch with the
/// layer's decoded weights kept (`woaj.run`), else turboexl3's grouped linear.
fn groupedMatmul(w: *const abi.Weight, x: A, s: mlx.mlx_stream) !A {
    if (mlx.getShape(x)[0] > t3.kernels.DECODE_ROWS_MAX) {
        if (hats_of != w.w.ctx) {
            dropHats();
            hats_of = w.w.ctx;
        }
        if (try woaj.run(s, x, projOf(w), &hats)) |y| return y;
    } else if (hats_of != null) dropHats();
    return t3.linearGrouped(s, x, projOf(w), decodes[w.layout]);
}

fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "turboexl3_ds41: routed experts run through swiglu");
}

fn swiglu(gate: *const abi.Weight, up: *const abi.Weight, down: *const abi.Weight, x: abi.Array, inds: abi.Array, scores: abi.Array, opts: *const abi.Swiglu, s: abi.Stream, out: *abi.Array) callconv(.c) Verdict {
    const bank: t3.Bank = .{ .gate = projOf(gate), .up = projOf(up), .down = projOf(down) };
    const st: mlx.mlx_stream = .{ .ctx = s.ctx };
    // No routes: a dense MLP (the shared expert), in one launch where turboexl3 takes the rows.
    if (inds.ctx == null) {
        const y = (shared_swiglu.fused(st, arr(x), bank.gate, bank.up, bank.down, @floatFromInt(opts.limit), mlx.mlx_array_dtype(arr(x))) catch |e| return Verdict.named(.failed, @errorName(e))) orelse
            return Verdict.refuse(.unsupported, "turboexl3_ds41: the dense SwiGLU at these rows runs the caller's chain");
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    const dec = decodes[gate.layout];
    const y = (if (opts.limit > 0)
        t3.moeClamped(st, arr(x), bank, arr(inds), arr(scores), dec, opts.limit)
    else
        t3.moe(st, arr(x), bank, arr(inds), arr(scores), dec, opts.verify_rows)) catch |e| return Verdict.named(.failed, @errorName(e));
    out.* = .{ .ctx = y.ctx };
    return .ok;
}

fn projOf(w: *const abi.Weight) t3.Proj {
    return .{ .trellis = arr(w.w), .suh = arr(w.scales), .svh = arr(w.biases) };
}

// ── the streamed bank (`experts.bin`) ──

const io = std.Io.Threaded.global_single_threaded.io();

fn geometry(geo: *const abi.BankGeometry) exl3.Geometry {
    return .{ .hidden_size = geo.hidden, .moe_intermediate_size = geo.inter, .n_experts_per_tok = geo.top_k, .n_routed_experts = geo.n_experts, .n_layers = geo.n_layers, .swiglu_limit = geo.swiglu_limit };
}

/// The host's budget with what this bank's decode gives back: the cache stop MLX holds less of at decode
/// (`decode_stop.cut_bytes`), and the runtime floor the host bills decode at (512 MiB, its bill's x 5/4) over the 128
/// MiB this bank's decode measured.
fn budgetOf(b: abi.BankBudget) exl3.Budget {
    const back: u64 = stream.decode_stop.cut_bytes + (512 - 128) * (1 << 20) * 5 / 4;
    return .{ .prompt = b.prompt, .decode = if (b.decode > 0) b.decode + back else 0 };
}

fn bankRefusal(e: anyerror) Verdict {
    return if (e == error.StreamedRowsDoNotFit) Verdict.refuse(.too_large, "StreamedRowsDoNotFit") else Verdict.named(.failed, @errorName(e));
}

/// The prompt lane's GEMMs are built on MLX's NAX tiles: below the M5 they compute wrong (f64 self-check, M4).
const no_nax = Verdict.refuse(.unsupported, "turboexl3_ds41: the streamed bank needs an M5 or newer GPU (its prompt kernels run on the matrix units)");

fn bankRows(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.BankRows) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    const b = exl3.billOf(gpa, io, ctx.model_dir.slice(), geometry(geo), budgetOf(budget)) catch |e| return bankRefusal(e);
    out.* = .{ .prompt = b.rows.prompt, .decode = b.rows.decode, .row_bytes = b.row_bytes };
    return .ok;
}

/// The settings the bank reads at its open, each refused by name at a bad value.
fn configureBank() !void {
    stream.buffer_mb.configure(env("MLX_MAX_MB_PER_BUFFER"));
    try stream.frac.configure(env("TURBOEXL3_DS41_FRAC"));
    stream.expert_usage_cfg.ps().setting = try stream.expert_usage_cfg.parse(env("TURBOEXL3_DS41_RESIDENT_SEED"), env("TURBOEXL3_DS41_EXPERT_USAGE_DIR"), env("TURBOEXL3_DS41_EXPERT_USAGE_RECORD"), env("TURBOEXL3_DS41_EXPERT_USAGE_LABEL"));
}

fn env(comptime name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// The load line's account of the one bank a process streams.
var about: [160]u8 = undefined;

fn openBank(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) Verdict {
    if (!tkit.nax.available()) return no_nax;
    configureBank() catch |e| return Verdict.named(.unsupported, @errorName(e));
    const g = geo.gates orelse return Verdict.refuse(.failed, "turboexl3_ds41: the bank's geometry has no gates");
    const gates = gpa.alloc(exl3.Gate, geo.n_layers) catch return oom;
    defer gpa.free(gates);
    for (gates, g[0..geo.n_layers]) |*o, x| o.* = .{ .w = arr(x.w), .bias = arr(x.bias) };
    const hc = config.hcOf(ctx.config_json.slice()) orelse return Verdict.refuse(.unsupported, "turboexl3_ds41: the config names no text_config.hc_mult");
    const src = exl3.Source.open(gpa, io, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }, geometry(geo), gates, budgetOf(budget)) catch |e| return bankRefusal(e);
    warmDecodeTail(.{ .ctx = ctx.stream.ctx }, geo.top_k, geo.hidden, hc) catch |e| {
        src.deinit();
        return Verdict.named(.failed, @errorName(e));
    };
    const line = std.fmt.bufPrint(&about, "{d} prompt / {d} decode slot rows per layer ({d} B a row)", .{ src.bill.rows.prompt, src.bill.rows.decode, src.bill.row_bytes }) catch "";
    const jl = src.joinlessOn();
    out.* = .{
        .ctx = src,
        .close = &Bank.close,
        .about = .of(line),
        .decode_ids = @intCast(exl3.decode_route_ids),
        .rows = &Bank.rows,
        .settle = &Bank.settle,
        .phase = &Bank.phase,
        .read_ahead = &Bank.readAhead,
        .rows_shared = if (jl) &Bank.rowsShared else null,
        .rows_expand = if (jl) &Bank.rowsExpand else null,
        .rows_expand_into = if (jl) &Bank.rowsExpandInto else null,
        .rows_decode_expand = &Bank.rowsDecodeExpand,
        .next_ahead = &Bank.nextAhead,
    };
    return .ok;
}

/// The bank's fused decode tail (`tail_fuse.expandDecode`) run once on zeros of a decode call's shapes, the shared
/// rows in the activation dtype and in f32, so its pipelines compile at the load, not at a request's first decode call.
fn warmDecodeTail(s: mlx.mlx_stream, k: u32, d: u32, hc: u32) !void {
    var held: [8]A = undefined;
    var n: usize = 0;
    defer for (held[0..n]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const Z = struct {
        fn of(st: mlx.mlx_stream, h: *[8]A, i: *usize, shape: []const c_int, dt: mlx.mlx_dtype) !A {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, st));
            h[i.*] = o;
            i.* += 1;
            return o;
        }
    };
    const ki: c_int = @intCast(k);
    const di: c_int = @intCast(d);
    const hi: c_int = @intCast(hc);
    const rows = try Z.of(s, &held, &n, &.{ ki, di }, .float32);
    const loc = try Z.of(s, &held, &n, &.{ 1, ki, 2 }, .int32);
    const w = try Z.of(s, &held, &n, &.{ 1, ki }, .float32);
    const res = try Z.of(s, &held, &n, &.{ 1, hi, di }, .bfloat16);
    const pc = try Z.of(s, &held, &n, &.{ 1, hi + hi * hi }, .float32);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |sdt| {
        const shared = try Z.of(s, &held, &n, &.{ 1, di }, sdt);
        const y = try stream.tail_fuse.expandDecode(s, &.{rows}, loc, w, shared, res, pc);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
}

/// `abi.Bank`'s entries over the source.
const Bank = struct {
    fn of(ctx: ?*anyopaque) *exl3.Source {
        return @ptrCast(@alignCast(ctx.?));
    }
    fn done(r: anyerror!void) Verdict {
        r catch |e| return Verdict.named(.failed, @errorName(e));
        return .ok;
    }
    fn put(r: anyerror!A, out: *abi.Array) Verdict {
        const y = r catch |e| return Verdict.named(.failed, @errorName(e));
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    fn close(ctx: ?*anyopaque) callconv(.c) void {
        of(ctx).deinit();
    }
    fn rows(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        return put(of(ctx).rows(layer, arr(x), arr(idx), arr(wts)), out);
    }
    /// The joinless combine runs on the wide lane's waves: a narrower call (the trunk's last-row replay) is forced there.
    fn rowsShared(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, shared: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        stream.l39_last.ps().force = true;
        defer stream.l39_last.ps().force = false;
        return put(of(ctx).rowsShared(layer, arr(x), arr(idx), arr(wts), arr(shared)), out);
    }
    fn rowsExpand(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, shared: abi.Array, res: abi.Array, pc: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        return put(of(ctx).rowsSharedExpand(layer, arr(x), arr(idx), arr(wts), arr(shared), arr(res), arr(pc)), out);
    }
    fn rowsExpandInto(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, shared: abi.Array, res: abi.Array, pc: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        return put(of(ctx).rowsSharedExpandInto(layer, arr(x), arr(idx), arr(wts), arr(shared), arr(res), arr(pc)), out);
    }
    fn rowsDecodeExpand(ctx: ?*anyopaque, layer: u32, x: abi.Array, idx: abi.Array, wts: abi.Array, shared: abi.Array, res: abi.Array, pc: abi.Array, out: *abi.Array) callconv(.c) Verdict {
        return put(of(ctx).rowsDecodeExpand(layer, arr(x), arr(idx), arr(wts), arr(shared), arr(res), arr(pc)), out);
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
    fn nextAhead(ctx: ?*anyopaque, layer: u32, ids: abi.Array) callconv(.c) Verdict {
        return done(of(ctx).nextAhead(layer, arr(ids)));
    }
};

pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    _ = a;
    return .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"deepseek_v41\"}" }}, .block = "{\"quant_method\":\"exl3\",\"codebook\":\"mul1\"}" };
}
