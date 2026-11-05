//! The shared expert of wide prefill chunks on the Neural Engine, beside the GPU's routed experts
//! (`SUSHI_DS41_ANE_SHARED=1|2`, the number of ANE units; lossy, off by default).

const std = @import("std");
const kit = @import("mlx_serve");
const log = @import("sushi/log.zig");
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const cfg_mod = @import("sushi/dsv41/dsv41_config.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;

pub const supported = @hasDecl(kit.abi.host, "HostAne");
const HostAne = if (supported) kit.abi.host.HostAne else opaque {};
const Bank = if (supported) kit.abi.host.AneBank else opaque {};
const Program = if (supported) kit.abi.host.AneBankProgram else opaque {};

const max_units = 2;

const Job = struct { st: *const AneState, unit: usize, layer: u32 = 0, ok: bool = false };

pub const AneState = struct {
    ane: *const HostAne,
    n_units: usize,
    unit_rows: u32,
    chunk_rows: usize,
    layers: usize,
    programs: [max_units]?*Program = @splat(null),
    planes: [max_units]A = @splat(sc_mod.none),
    jobs: [max_units]Job = undefined,
    threads: [max_units]?std.Thread = @splat(null),
    off: bool = false,
    chunks: u64 = 0,

    fn start(st: *AneState, layer: usize) void {
        for (0..st.n_units) |u| {
            st.jobs[u] = .{ .st = st, .unit = u, .layer = @intCast(layer) };
            st.threads[u] = std.Thread.spawn(.{}, runUnit, .{&st.jobs[u]}) catch null;
        }
    }

    fn join(st: *AneState) bool {
        var ok = true;
        for (0..st.n_units) |u| {
            if (st.threads[u]) |t| t.join() else runUnit(&st.jobs[u]);
            ok = ok and st.jobs[u].ok;
        }
        return ok;
    }
};

fn runUnit(j: *Job) void {
    var err: [256]u8 = @splat(0);
    j.ok = j.st.ane.bank_program_eval(j.st.ane.ctx, j.st.programs[j.unit].?, j.layer, &err, err.len);
    if (!j.ok) log.warn("[sushi_ds41] ane shared: unit {d} layer {d}: {s}\n", .{ j.unit, j.layer, std.mem.sliceTo(&err, 0) });
}

pub fn setAneShared(m: *dsv.Model, host: ?*const anyopaque) void {
    const raw = std.c.getenv("SUSHI_DS41_ANE_SHARED") orelse return;
    const units = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch 1;
    if (comptime supported) setOnHost(m, host, units) else log.warn("[sushi_ds41] ane shared: off (built against a kit without Host.ane)\n", .{});
}

pub fn deinit(m: *dsv.Model, st: *AneState) void {
    if (comptime supported) free(m, st) else unreachable;
}

pub fn moe(sc: *Scope, m: *const dsv.Model, st: *AneState, li: usize, ly: *const dsv.Layer, h: A, hk: dsv.Experts.Hook, r: kernels.Routes, ride: []const A) !?A {
    if (comptime supported) return moeOnHost(sc, m, st, li, ly, h, hk, r, ride) else unreachable;
}

fn setOnHost(m: *dsv.Model, host: ?*const anyopaque, units: usize) void {
    const h = kit.host.Host.of(host) catch null;
    const ane = (if (h) |x| x.ane() else null) orelse
        return log.warn("[sushi_ds41] ane shared: off (this host has no Neural Engine to offer)\n", .{});
    m.ane = build(m, ane, units) catch |e| return log.warn("[sushi_ds41] ane shared: off ({s})\n", .{@errorName(e)});
}

fn free(m: *dsv.Model, st: *AneState) void {
    for (st.planes) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (st.programs) |p| if (p) |x| st.ane.bank_program_free(st.ane.ctx, x);
    m.gpa.destroy(st);
}

fn aneError(what: []const u8, err: []const u8) error{AneShared} {
    log.warn("[sushi_ds41] ane shared: {s}: {s}\n", .{ what, std.mem.sliceTo(err, 0) });
    return error.AneShared;
}

fn build(m: *dsv.Model, ane: *const HostAne, n_units: usize) !*AneState {
    const d: u32 = @intCast(m.cfg.dim);
    const ffn: u32 = @intCast(m.layers[0].shared[0].outDim());
    const chunk = cfg_mod.prefill_width;
    if (n_units == 0 or n_units > max_units or chunk % (32 * n_units) != 0) return error.AneSharedShape;
    const unit_rows: u32 = @intCast(chunk / n_units);
    const layers = @min(m.ced_split orelse m.cfg.layers, m.cfg.layers);

    var err: [512]u8 = @splat(0);
    var banks: [max_units]?*Bank = @splat(null);
    defer for (banks) |b| if (b) |x| ane.bank_free(ane.ctx, x);
    for (banks[0..n_units]) |*b| b.* = ane.bank_create(ane.ctx) orelse return error.AneSharedBank;
    for (0..layers) |li| {
        var sc = Scope.init(m.gpa, m.s);
        defer sc.deinit();
        const sh = &m.layers[li].shared;
        const g = try int8Rows(&sc, m, &sh[0], d);
        const u = try int8Rows(&sc, m, &sh[1], d);
        const dn = try int8Rows(&sc, m, &sh[2], ffn);
        for (banks[0..n_units]) |b| {
            const proc = ane.bank_add_swiglu(ane.ctx, b.?, d, ffn, unit_rows, g.q, g.s, u.q, u.s, dn.q, dn.s, m.cfg.swiglu_limit, &err, err.len);
            if (proc != @as(i32, @intCast(li))) return aneError("program", &err);
        }
    }

    const st = try m.gpa.create(AneState);
    st.* = .{ .ane = ane, .n_units = n_units, .unit_rows = unit_rows, .chunk_rows = chunk, .layers = layers };
    errdefer free(m, st);
    for (0..n_units) |u| {
        var name: [64:0]u8 = @splat(0);
        _ = try std.fmt.bufPrint(&name, "ds41_shared_r{d}_u{d}of{d}", .{ unit_rows, u, n_units });
        const die: i32 = if (n_units == 1) 0 else @intCast(u + 1);
        const p = ane.bank_finish(ane.ctx, banks[u].?, &name, die, null, null, &err, err.len) orelse {
            banks[u] = null;
            return aneError("compile", &err);
        };
        banks[u] = null;
        st.programs[u] = p;
        const plane = [_]c_int{ci(@as(usize, d) * unit_rows)};
        st.planes[u] = mlx.mlx_array_new_data_managed(@ptrCast(ane.bank_program_input(ane.ctx, p).?), &plane, 1, .float16, noFree);
    }
    log.info("[sushi_ds41] ane shared: {d} layers x {d}x{d} SwiGLU, {d} unit(s) of {d} rows\n", .{ layers, d, ffn, n_units, unit_rows });
    try selfCheck(m, st);
    return st;
}

const Int8 = struct { q: [*]const i8, s: [*]const f32 };

extern "c" fn mlx_eye(res: *A, n: c_int, m: c_int, k: c_int, dtype: mlx.mlx_dtype, s: sc_mod.S) c_int;

fn int8Rows(sc: *Scope, m: *const dsv.Model, lin: *const dsv.Lin, in: u32) !Int8 {
    var eye = mlx.mlx_array_new();
    const identity = try sc.res(mlx_eye(&eye, ci(in), ci(in), 0, m.act, m.s), &eye);
    const w = try sc.astype(try sc.t2(try lin.apply(sc, identity)), .float32);
    const amax = try sc.maximum(try sc.max(try sc.op1(mlx.mlx_abs, w), 1, true), try sc.f(1e-30));
    const scale = try sc.div(amax, try sc.f(127));
    var r = mlx.mlx_array_new();
    const rounded = try sc.res(mlx.mlx_round(&r, try sc.div(w, scale), 0, sc.s), &r);
    const q = try sc.contiguous(try sc.astype(rounded, .int8));
    try mlx.check(mlx.mlx_array_eval(q));
    const s = try sc_mod.hostF32(try sc.contiguous(try sc.reshape(scale, &.{-1})));
    return .{ .q = @ptrCast(mlx.mlx_array_data_uint8(q) orelse return error.HostRead), .s = s.ptr };
}

fn noFree(_: ?*anyopaque) callconv(.c) void {}

const PACK_SOURCE =
    \\uint r = thread_position_in_grid.x; uint c = thread_position_in_grid.y;
    \\if (r >= uint(CH) || c >= uint(D)) return;
    \\uint u = r / uint(R);
    \\device half *p = (device half *)(u == 0 ? p0 : p1);
    \\float v = r < uint(ROWS) ? static_cast<float>(x[r * uint(D) + c]) : 0.0f;
    \\p[c * uint(R) + (r - u * uint(R))] = static_cast<half>(v);
    \\if (r == 0 && c == 0) done[0] = 0;
;
var pack_def = kernels.Def{ .name = "dsv41_ane_pack", .ins = &.{ "x", "p0", "p1" }, .outs = &.{"done"}, .source = PACK_SOURCE };

fn pack(sc: *Scope, st: *const AneState, h: A) !A {
    const d = dim(h, 1);
    var out: [1]A = undefined;
    try kernels.launch(sc, &pack_def, &.{ h, st.planes[0], st.planes[st.n_units - 1] }, &.{.{ .shape = &.{1}, .dt = .float16 }}, .{ st.chunk_rows, d, 1 }, .{ 32, 8, 1 }, &.{
        .{ .name = "ROWS", .v = ci(dim(h, 0)) }, .{ .name = "D", .v = ci(d) }, .{ .name = "R", .v = ci(st.unit_rows) }, .{ .name = "CH", .v = ci(st.chunk_rows) },
    }, &.{}, &out);
    return out[0];
}

fn output(sc: *Scope, st: *const AneState, rows: usize, d: usize) !A {
    var parts: [max_units]A = undefined;
    const shape = [_]c_int{ ci(d), ci(st.unit_rows) };
    for (0..st.n_units) |u| {
        const ptr: *anyopaque = @ptrCast(st.ane.bank_program_output(st.ane.ctx, st.programs[u].?).?);
        parts[u] = try sc.keep(mlx.mlx_array_new_data_managed(ptr, &shape, 2, .float16, noFree));
    }
    const y = try sc.t2(try sc.concat(parts[0..st.n_units], 1));
    return sc.astype(try sc.slice(y, 0, 0, rows), .float32);
}

fn dot(sc: *Scope, a: A, b: A) !f32 {
    return (try sc_mod.hostF32(try sc.sum(try sc.reshape(try sc.mul(a, b), &.{-1}), 0, false)))[0];
}

fn selfCheck(m: *dsv.Model, st: *AneState) !void {
    var sc = Scope.init(m.gpa, m.s);
    defer sc.deinit();
    var hn = mlx.mlx_array_new();
    const shape = [_]c_int{ ci(st.chunk_rows), ci(m.cfg.dim) };
    const h = try sc.res(mlx.mlx_random_normal(&hn, &shape, 2, m.act, 0, 1, sc_mod.none, m.s), &hn);
    try mlx.check(mlx.mlx_array_eval(try pack(&sc, st, h)));
    var worst: struct { layer: usize = 0, cos: f32 = 2 } = .{};
    for (0..st.layers) |li| {
        var lsc = Scope.init(m.gpa, m.s);
        defer lsc.deinit();
        const gpu = try lsc.astype(try dsv.sharedExpert(&lsc, m, li, &m.layers[li], h), .float32);
        st.start(li);
        if (!st.join()) return error.AneSharedEval;
        const ane = try output(&lsc, st, st.chunk_rows, m.cfg.dim);
        const cos = try dot(&lsc, ane, gpu) / @sqrt(try dot(&lsc, ane, ane) * try dot(&lsc, gpu, gpu));
        if (!(cos >= worst.cos)) worst = .{ .layer = li, .cos = cos };
    }
    log.info("[sushi_ds41] ane shared: self-check {d} layers: worst cos {d:.6} (layer {d})\n", .{ st.layers, worst.cos, worst.layer });
    if (!(worst.cos > 0.99)) return error.AneSharedSelfCheck;
}

fn waitForPreviousChunk(m: *const dsv.Model) !void {
    try mlx.check(mlx.mlx_synchronize(m.s));
}

fn moeOnHost(sc: *Scope, m: *const dsv.Model, st: *AneState, li: usize, ly: *const dsv.Layer, h: A, hk: dsv.Experts.Hook, r: kernels.Routes, ride: []const A) !?A {
    const rows = dim(h, 0);
    if (st.off or m.tap != null or li >= st.layers or rows != st.chunk_rows) return null;
    if (li == 0) {
        try waitForPreviousChunk(m);
        st.chunks += 1;
    }
    const x = try pack(sc, st, h);
    try dsv.submit(x);
    try dsv.submitWith(r.idx, ride);
    try mlx.check(mlx.mlx_array_eval(x));
    st.start(li);
    const routed = routedSum(sc, hk, li, h, r) catch |e| {
        _ = st.join();
        return e;
    };
    const shared = if (st.join()) try output(sc, st, rows, dim(h, 1)) else try gpuFallback(sc, m, st, li, ly, h);
    const out = try sc.astype(try sc.add(routed, shared), m.act);
    try if (dsv.pipeOn(m, rows)) dsv.submit(out) else mlx.check(mlx.mlx_array_eval(out));
    if (li + 1 == st.layers) log.info("[sushi_ds41] ane shared: chunk {d} rows={d} layers={d}\n", .{ st.chunks, rows, st.layers });
    return out;
}

fn routedSum(sc: *Scope, hk: dsv.Experts.Hook, li: usize, h: A, r: kernels.Routes) !A {
    const y = try hk.apply(hk.ctx, sc, li, h, r.idx, r.wts, r.logits, .none);
    try dsv.submit(y);
    return y;
}

fn gpuFallback(sc: *Scope, m: *const dsv.Model, st: *AneState, li: usize, ly: *const dsv.Layer, h: A) !A {
    st.off = true;
    log.warn("[sushi_ds41] ane shared: eval failed at layer {d}, off for the rest of this run\n", .{li});
    return sc.astype(try dsv.sharedExpert(sc, m, li, ly, h), .float32);
}
