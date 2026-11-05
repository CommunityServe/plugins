//! The shared expert of a wide prefill chunk on the Neural Engine (`SUSHI_DS41_ANE_SHARED`, a prototype, off by
//! default): one int8 SwiGLU procedure per covered layer in one bank per unit, built and run through the host's Neural
//! Engine (`Host.ane`). With two units each die takes half the chunk's rows, so a layer's program takes half the time and both
//! dies run at once. The GPU runs the routed experts meanwhile; the model adds this output before the next layer.
//!
//! I/O is fp16 in channel-major planes: unit u reads `[hidden][rows]` of rows `u * rows ..`, and writes `[hidden][rows]`.

const std = @import("std");
const log = @import("sushi/log.zig");
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const cfg_mod = @import("sushi/dsv41/dsv41_config.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const none = sc_mod.none;
const dim = sc_mod.dim;
const ci = sc_mod.ci;

const kit = @import("mlx_serve");
/// Whether the kit carries the host's Neural Engine table (mlx-serve after ddalcu/mlx-serve#1083; the 26.11.1 release
/// kit has none). Without it the entry points below compile to "off" and nothing that names the table is analyzed.
pub const supported = @hasDecl(kit.abi.host, "HostAne");
const HostAne = if (supported) kit.abi.host.HostAne else opaque {};
const Bank = if (supported) kit.abi.host.AneBank else opaque {};
const Mlp = if (supported) kit.abi.host.AneBankProgram else opaque {};

pub const max_units = 2;

/// Monotonic nanoseconds.
pub fn nowNs() u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(std.Io.Timestamp.now(io, .boot).nanoseconds);
}
pub const max_layers = 64;

/// Per-output-row int8 of a row-major `[rows, cols]` f32 weight: `q` `[rows, cols]`, `s` `[rows]` (w ~ q * s).
pub fn quantizeRows(w: []const f32, rows: usize, cols: usize, q: []i8, s: []f32) void {
    for (0..rows) |r| {
        const row = w[r * cols ..][0..cols];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        s[r] = scale;
        const inv = 1.0 / scale;
        for (row, q[r * cols ..][0..cols]) |v, *o| o.* = @intFromFloat(std.math.clamp(@round(v * inv), -127, 127));
    }
}

pub const Programs = struct {
    ane: *const HostAne,
    units: [max_units]?*Mlp = @splat(null),
    banks: [max_units]?*Bank = @splat(null),
    n_units: usize,
    /// Rows a unit's program takes; a chunk is `n_units * rows` rows (zero-padded).
    rows: u32,
    hidden: u32,
    ffn: u32,
    /// The bank procedure of each layer, -1 where the layer has none.
    proc: [max_layers]i32 = @splat(-1),

    pub fn init(ane: *const HostAne, n_units: usize, chunk_rows: u32, hidden: u32, ffn: u32) !Programs {
        if (n_units == 0 or n_units > max_units or chunk_rows % (32 * @as(u32, @intCast(n_units))) != 0) return error.AneSharedShape;
        var p: Programs = .{ .ane = ane, .n_units = n_units, .rows = chunk_rows / @as(u32, @intCast(n_units)), .hidden = hidden, .ffn = ffn };
        for (0..n_units) |u| p.banks[u] = ane.bank_create(ane.ctx) orelse return error.AneSharedBank;
        return p;
    }

    pub fn deinit(self: *Programs) void {
        for (&self.banks) |*b| if (b.*) |x| {
            self.ane.bank_free(self.ane.ctx, x);
            b.* = null;
        };
        for (&self.units) |*u| if (u.*) |x| {
            self.ane.bank_program_free(self.ane.ctx, x);
            u.* = null;
        };
    }

    /// Adds layer `li`'s program to every unit's bank: `gate`/`up` `[ffn, hidden]`, `down` `[hidden, ffn]`, int8 rows.
    pub fn addLayer(self: *Programs, li: usize, gq: []const i8, gs: []const f32, uq: []const i8, us: []const f32, dq: []const i8, ds: []const f32, limit: f32) !void {
        var err: [256]u8 = @splat(0);
        var idx: c_int = -1;
        for (0..self.n_units) |u| {
            const got = self.ane.bank_add_swiglu(self.ane.ctx, self.banks[u].?, self.hidden, self.ffn, self.rows, gq.ptr, gs.ptr, uq.ptr, us.ptr, dq.ptr, ds.ptr, limit, &err, err.len);
            if (got < 0) {
                log.err("[sushi_ds41] ane shared: layer {d} program: {s}\n", .{ li, std.mem.sliceTo(&err, 0) });
                return error.AneSharedBank;
            }
            if (idx >= 0 and got != idx) return error.AneSharedBank;
            idx = got;
        }
        self.proc[li] = idx;
    }

    /// Compiles (or restores from the compile cache) and loads each unit's bank, pinned to die u + 1.
    pub fn finish(self: *Programs) !void {
        var err: [512]u8 = @splat(0);
        for (0..self.n_units) |u| {
            var name: [64]u8 = undefined;
            const txt = try std.fmt.bufPrint(name[0 .. name.len - 1], "ds41_shared_r{d}_u{d}of{d}", .{ self.rows, u, self.n_units });
            name[txt.len] = 0;
            const nm: [*:0]const u8 = @ptrCast(&name);
            // A single unit gets no instance hint (the runtime picks a die); two pin one die each.
            const inst: i32 = if (self.n_units == 1) 0 else @intCast(u + 1);
            const b = self.banks[u].?;
            self.banks[u] = null;
            self.units[u] = self.ane.bank_finish(self.ane.ctx, b, nm, inst, null, null, &err, err.len) orelse {
                log.err("[sushi_ds41] ane shared: compile of unit {d}: {s}\n", .{ u, std.mem.sliceTo(&err, 0) });
                return error.AneSharedCompile;
            };
            log.info("[sushi_ds41] ane shared: unit {d} (die hint {d}) {d} rows, compile {d:.2} s, cache {s}\n", .{ u, inst, self.rows, self.ane.bank_program_compile_seconds(self.ane.ctx, self.units[u].?), if (self.ane.bank_program_cache_hit(self.ane.ctx, self.units[u].?)) "hit" else "miss" });
        }
    }

    pub fn slabLen(self: *const Programs) usize {
        return @as(usize, self.hidden) * self.rows;
    }

    pub fn input(self: *const Programs, u: usize) [*]f16 {
        return self.ane.bank_program_input(self.ane.ctx, self.units[u].?).?;
    }

    pub fn output(self: *const Programs, u: usize) [*]const f16 {
        return self.ane.bank_program_output(self.ane.ctx, self.units[u].?).?;
    }

    const Job = struct { p: *const Programs, u: usize, proc: u32, x: ?[*]const f16, ok: bool = false, ns: u64 = 0, copy_ns: u64 = 0 };

    fn runUnit(j: *Job) void {
        const t0 = nowNs();
        const n = j.p.slabLen();
        const dst = j.p.input(j.u);
        // A null input means the GPU already wrote the plane (`input`).
        if (j.x) |x| @memcpy(dst[0..n], x[j.u * n ..][0..n]);
        j.copy_ns = nowNs() - t0;
        var err: [256]u8 = @splat(0);
        j.ok = j.p.ane.bank_program_eval(j.p.ane.ctx, j.p.units[j.u].?, j.proc, &err, err.len);
        if (!j.ok) log.warn("[sushi_ds41] ane shared: eval unit {d}: {s}\n", .{ j.u, std.mem.sliceTo(&err, 0) });
        j.ns = nowNs() - t0;
    }

    pub const Ran = struct { ok: bool, unit_ns: [max_units]u64, copy_ns: u64 = 0 };

    /// Layer `li` over `x`: `n_units` channel-major slabs of `slabLen()` fp16 each, back to back. Unit 0 runs on the
    /// calling thread, unit 1 on its own; both copy their slab in and evaluate. The outputs are in `output(u)`.
    pub fn run(self: *const Programs, li: usize, x: [*]const f16) Ran {
        var jobs: [max_units]Job = undefined;
        const proc: u32 = @intCast(self.proc[li]);
        for (0..self.n_units) |u| jobs[u] = .{ .p = self, .u = u, .proc = proc, .x = x };
        var th: ?std.Thread = null;
        if (self.n_units > 1) th = std.Thread.spawn(.{}, runUnit, .{&jobs[1]}) catch null;
        runUnit(&jobs[0]);
        if (self.n_units > 1) {
            if (th) |t| t.join() else runUnit(&jobs[1]);
        }
        var r: Ran = .{ .ok = true, .unit_ns = @splat(0) };
        for (0..self.n_units) |u| {
            r.ok = r.ok and jobs[u].ok;
            r.unit_ns[u] = jobs[u].ns;
            r.copy_ns = @max(r.copy_ns, jobs[u].copy_ns);
        }
        return r;
    }

    /// `run` with every unit on its own thread, returned at once: the caller queues GPU work, then `join`s.
    pub const Pending = struct {
        jobs: [max_units]Job = undefined,
        th: [max_units]?std.Thread = @splat(null),
        n: usize = 0,

        pub fn join(self: *Pending) Ran {
            for (0..self.n) |u| {
                if (self.th[u]) |t| t.join() else runUnit(&self.jobs[u]);
            }
            var r: Ran = .{ .ok = true, .unit_ns = @splat(0) };
            for (0..self.n) |u| {
                r.ok = r.ok and self.jobs[u].ok;
                r.unit_ns[u] = self.jobs[u].ns;
                r.copy_ns = @max(r.copy_ns, self.jobs[u].copy_ns);
            }
            return r;
        }
    };

    /// Starts layer `li` over `x` on every unit; `p` must stay put until `p.join()`.
    pub fn start(self: *const Programs, p: *Pending, li: usize, x: ?[*]const f16) void {
        const proc: u32 = @intCast(self.proc[li]);
        p.n = self.n_units;
        for (0..self.n_units) |u| {
            p.jobs[u] = .{ .p = self, .u = u, .proc = proc, .x = x };
            // A unit whose thread did not start runs in `join`.
            p.th[u] = std.Thread.spawn(.{}, runUnit, .{&p.jobs[u]}) catch null;
        }
    }
};

// ── The model's side: one SwiGLU program per covered layer, run beside the GPU's routed experts ──

/// The shared expert of wide prefill chunks (`SUSHI_DS41_ANE_SHARED=1|2`, the unit count; lossy: int8 weights, fp16
/// activations). Lives on `Model.ane`; the slice calls `moe` from its MoE layer and `deinit` from its own.
pub const AneState = struct {
    progs: Programs,
    /// Rows of a chunk the programs take (`n_units * rows`): the prefill width. Other widths run on the GPU.
    chunk_rows: usize,
    /// Layers `0..layers` have programs (the CED encoder's, which run every row of a chunk).
    layers: usize,
    /// The input planes as MLX arrays over the surfaces (no copy), written by `ane_pack_def` on the GPU. Off when the
    /// self-check finds them differ from the host copy: a transposed copy the ANE threads memcpy in.
    pack: bool = true,
    planes: [max_units]A = @splat(none),
    evals: u64 = 0,
    fails: u64 = 0,
    disabled: bool = false,
    chunks: u64 = 0,
    chunk: struct { layers: u32 = 0, ane: u64 = 0, ane_max: u64 = 0, join: u64 = 0 } = .{},
};

/// Reads `SUSHI_DS41_ANE_SHARED` (1 or 2 units) and builds the programs on the host's Neural Engine (`host`, the
/// load's `LoadCtx.host`). No ANE on this host, or a failure, leaves the GPU path serving.
pub fn setAneShared(m: *dsv.Model, host: ?*const anyopaque) void {
    if (std.c.getenv("SUSHI_DS41_ANE_SHARED") == null) return;
    if (comptime supported) setOnHost(m, host) else log.warn("[sushi_ds41] ane shared: off (built against a kit without Host.ane)\n", .{});
}

fn setOnHost(m: *dsv.Model, host: ?*const anyopaque) void {
    const raw = std.c.getenv("SUSHI_DS41_ANE_SHARED") orelse return;
    const units = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch 1;
    const ane = blk: {
        const h = kit.host.Host.of(host) catch break :blk null;
        break :blk h.ane();
    } orelse {
        log.warn("[sushi_ds41] ane shared: off (this host has no Neural Engine to offer)\n", .{});
        return;
    };
    m.ane = init(m, ane, units) catch |e| {
        log.warn("[sushi_ds41] ane shared: off ({s})\n", .{@errorName(e)});
        return;
    };
}

pub fn deinit(m: *dsv.Model, st: *AneState) void {
    if (comptime supported) deinitOnHost(m, st) else unreachable; // `setAneShared` never sets `Model.ane` without the table.
}

fn deinitOnHost(m: *dsv.Model, st: *AneState) void {
    for (st.planes) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    st.progs.deinit();
    m.gpa.destroy(st);
}

extern "c" fn mlx_eye(res: *A, n: c_int, m: c_int, k: c_int, dtype: mlx.mlx_dtype, s: sc_mod.S) c_int;

/// `lin`'s effective `[out, in]` weight in f32 on the host (its kernel run on the identity, so the codec's transforms
/// come out as they apply), into `dst`.
fn linF32(m: *const dsv.Model, lin: *const dsv.Lin, in: usize, dst: []f32) !void {
    var sc = Scope.init(m.gpa, m.s);
    defer sc.deinit();
    var eye = mlx.mlx_array_new();
    const I = try sc.res(mlx_eye(&eye, ci(in), ci(in), 0, m.act, m.s), &eye);
    const wt = try lin.apply(&sc, I); // [in, out]
    const w = try sc.contiguous(try sc.astype(try sc.t2(wt), .float32));
    const host = try sc_mod.hostF32(w);
    if (host.len != dst.len) return error.Dsv41WeightLayout;
    @memcpy(dst, host);
}

fn init(m: *dsv.Model, ane: *const HostAne, units: usize) !*AneState {
    const c = &m.cfg;
    const d: usize = c.dim;
    const f: usize = m.layers[0].shared[0].outDim();
    const chunk: usize = cfg_mod.prefill_width;
    const layers = @min(m.ced_split orelse c.layers, @min(c.layers, max_layers));
    var progs = try Programs.init(ane, units, @intCast(chunk), @intCast(d), @intCast(f));
    errdefer progs.deinit();
    const gpa = m.gpa;
    const wf = try gpa.alloc(f32, d * f);
    defer gpa.free(wf);
    const gq = try gpa.alloc(i8, d * f);
    defer gpa.free(gq);
    const uq = try gpa.alloc(i8, d * f);
    defer gpa.free(uq);
    const dq = try gpa.alloc(i8, d * f);
    defer gpa.free(dq);
    const gs = try gpa.alloc(f32, f);
    defer gpa.free(gs);
    const us = try gpa.alloc(f32, f);
    defer gpa.free(us);
    const ds = try gpa.alloc(f32, d);
    defer gpa.free(ds);
    const t0 = nowNs();
    for (0..layers) |li| {
        const ly = &m.layers[li];
        if (ly.shared[0].outDim() != f or ly.shared[2].outDim() != d) return error.Dsv41WeightLayout;
        try linF32(m, &ly.shared[0], d, wf);
        quantizeRows(wf, f, d, gq, gs);
        try linF32(m, &ly.shared[1], d, wf);
        quantizeRows(wf, f, d, uq, us);
        try linF32(m, &ly.shared[2], f, wf);
        quantizeRows(wf, d, f, dq, ds);
        try progs.addLayer(li, gq, gs, uq, us, dq, ds, c.swiglu_limit);
    }
    const t1 = nowNs();
    try progs.finish();
    const st = try gpa.create(AneState);
    errdefer gpa.destroy(st);
    st.* = .{ .progs = progs, .chunk_rows = chunk, .layers = layers };
    for (0..progs.n_units) |u| {
        const shape = [_]c_int{ci(progs.slabLen())};
        st.planes[u] = mlx.mlx_array_new_data_managed(@ptrCast(st.progs.input(u)), &shape, 1, .float16, noFree);
    }
    log.info("[sushi_ds41] ane shared: {d} layers x {d}x{d} SwiGLU (limit {d}), {d} unit(s) of {d} rows, chunks of {d} rows; weights {d:.1} s, programs {d:.1} s\n", .{
        layers, d, f, c.swiglu_limit, units, progs.rows, chunk, @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(nowNs() - t1)) / 1e9,
    });
    try selfCheck(m, st);
    return st;
}

/// `h` (`[chunk_rows, dim]`) as the units' channel-major fp16 slabs.
fn input(sc: *Scope, st: *const AneState, h: A) !A {
    const d = dim(h, 1);
    const x = try sc.astype(h, .float16);
    const nu = st.progs.n_units;
    return sc.contiguous(try sc.transpose(try sc.reshape(x, &.{ ci(nu), ci(st.progs.rows), ci(d) }), &.{ 0, 2, 1 }));
}

fn noFree(_: ?*anyopaque) callconv(.c) void {}

/// `x` `[ROWS, D]` into the units' input planes (fp16, channel-major `[D][R]` each, unit `r / R` taking row `r`), zero
/// past `ROWS`. Writes through the planes' const pointers: they are the ANE's IOSurfaces, wrapped without a copy.
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
    const p1 = if (st.progs.n_units > 1) st.planes[1] else st.planes[0];
    try kernels.launch(sc, &pack_def, &.{ h, st.planes[0], p1 }, &.{.{ .shape = &.{1}, .dt = .float16 }}, .{ st.chunk_rows, d, 1 }, .{ 32, 8, 1 }, &.{
        .{ .name = "ROWS", .v = ci(dim(h, 0)) }, .{ .name = "D", .v = ci(d) }, .{ .name = "R", .v = ci(st.progs.rows) }, .{ .name = "CH", .v = ci(st.chunk_rows) },
    }, &.{}, &out);
    return out[0];
}

/// The units' outputs as f32 `[rows, dim]`.
fn output(sc: *Scope, st: *const AneState, rows: usize, d: usize) !A {
    var parts: [max_units]A = undefined;
    const shape = [_]c_int{ ci(d), ci(st.progs.rows) };
    for (0..st.progs.n_units) |u| {
        const ptr: *anyopaque = @ptrCast(@constCast(st.progs.output(u)));
        // Zero-copy: MLX wraps a page-aligned, page-multiple buffer as a Metal buffer without a copy. The output surface
        // outlives the array (the programs own it) and the next eval writes it only after this layer is evaluated.
        parts[u] = try sc.keep(mlx.mlx_array_new_data_managed(ptr, &shape, 2, .float16, noFree));
    }
    const y = try sc.t2(try sc.concat(parts[0..st.progs.n_units], 1)); // [chunk, d]
    return sc.astype(try sc.slice(y, 0, 0, rows), .float32);
}

/// Every covered layer against its GPU chain on random rows, then the GPU-packed planes against the host copy.
fn selfCheck(m: *dsv.Model, st: *AneState) !void {
    var sc = Scope.init(m.gpa, m.s);
    defer sc.deinit();
    const d = m.cfg.dim;
    var hn = mlx.mlx_array_new();
    const shape = [_]c_int{ ci(st.chunk_rows), ci(d) };
    const h = try sc.res(mlx.mlx_random_normal(&hn, &shape, 2, m.act, 0, 1, none, m.s), &hn);
    const x = try input(&sc, st, h);
    try mlx.check(mlx.mlx_array_eval(x));
    const ptr: [*]const f16 = @ptrCast(@alignCast(mlx.mlx_array_data_uint8(x) orelse return error.HostRead));
    // Layer 0 last, so its output stays in the planes for the pack check.
    var worst: struct { li: usize = 0, cos: f32 = 2, ratio: f32 = 1, ran: Programs.Ran = .{ .ok = true, .unit_ns = @splat(0) } } = .{};
    var li = st.layers;
    while (li > 0) {
        li -= 1;
        var lsc = Scope.init(m.gpa, m.s);
        defer lsc.deinit();
        const ref = try lsc.astype(try dsv.sharedExpert(&lsc, m, li, &m.layers[li], h), .float32);
        try mlx.check(mlx.mlx_array_eval(ref));
        const ran = st.progs.run(li, ptr);
        if (!ran.ok) return error.AneSharedEval;
        const yl = try output(&lsc, st, st.chunk_rows, d);
        const dot = try lsc.sum(try lsc.reshape(try lsc.mul(yl, ref), &.{-1}), 0, false);
        const ny = try lsc.sum(try lsc.reshape(try lsc.mul(yl, yl), &.{-1}), 0, false);
        const nr = try lsc.sum(try lsc.reshape(try lsc.mul(ref, ref), &.{-1}), 0, false);
        const v = [3]f32{ (try sc_mod.hostF32(dot))[0], (try sc_mod.hostF32(ny))[0], (try sc_mod.hostF32(nr))[0] };
        const cos = v[0] / @sqrt(v[1] * v[2]);
        // `!(cos >= worst.cos)` also takes a NaN, which then fails the check below.
        if (!(cos >= worst.cos)) worst = .{ .li = li, .cos = cos, .ratio = @sqrt(v[1] / v[2]), .ran = ran };
    }
    log.info("[sushi_ds41] ane shared: self-check {d} layers, {d} random rows: worst cos {d:.6} (layer {d}) vs the GPU chain, |ane|/|gpu| {d:.4}, unit ms {d:.2}/{d:.2}\n", .{ st.layers, st.chunk_rows, worst.cos, worst.li, worst.ratio, @as(f64, @floatFromInt(worst.ran.unit_ns[0])) / 1e6, @as(f64, @floatFromInt(worst.ran.unit_ns[1])) / 1e6 });
    if (!(worst.cos > 0.99)) return error.AneSharedSelfCheck;
    // The GPU-packed planes: the same rows must give the same output, bit for bit.
    const y0 = try sc.contiguous(try output(&sc, st, st.chunk_rows, d));
    try mlx.check(mlx.mlx_array_eval(y0));
    const done = try pack(&sc, st, h);
    try mlx.check(mlx.mlx_array_eval(done));
    var pend: Programs.Pending = .{};
    st.progs.start(&pend, 0, null);
    if (!pend.join().ok) return error.AneSharedEval;
    const y1 = try output(&sc, st, st.chunk_rows, d);
    const diff = try sc.max(try sc.reshape(try sc.op1(mlx.mlx_abs, try sc.sub(y1, y0)), &.{-1}), 0, false);
    const md = (try sc_mod.hostF32(diff))[0];
    log.info("[sushi_ds41] ane shared: GPU-packed input planes vs host copy: max |diff| {d}\n", .{md});
    if (md != 0) st.pack = false;
}

/// Ends a layer: evaluated, or under the prefill pipeline (`pipeOn`) only submitted, so the caller's `pipeSubmit` owns
/// the wait as on the GPU path.
///
/// The ANE planes are reused by every covered layer, and the pipeline lets the host run ahead of the GPU. Still safe:
/// - Input planes: layer L's ANE run is joined before `moe` returns, so layer L+1's pack cannot overwrite them under it.
/// - Output planes: layer L's add reads them on the GPU, possibly after `moe` returned. Layer L+1 waits for its own
///   input `x` (built from layer L's output, so after that add) before its ANE run writes them.
/// - The next chunk's layer 0 does not depend on this chunk's last add, and nothing between chunks waits for it: a chunk
///   another follows ends in `extendState`, which drains only when the experts stream. So `moe` synchronizes the stream
///   at layer 0, before the chunk's first plane write: one wait per chunk.
/// Dropping the wait for `x` breaks the second point, the layer-0 sync the third.
fn layerOut(m: *const dsv.Model, out: A) !void {
    try if (dsv.pipeOn(m, dim(out, 0))) dsv.submit(out) else mlx.check(mlx.mlx_array_eval(out));
}

/// Layer `li`'s routed experts and shared expert of a wide chunk, the shared expert on the Neural Engine while the GPU
/// runs the routed ones. Evaluated, or submitted under the prefill pipeline (`layerOut`). Null where the ANE does not
/// take this layer (the GPU path serves).
pub fn moe(sc: *Scope, m: *const dsv.Model, st: *AneState, li: usize, ly: *const dsv.Layer, h: A, hk: dsv.Experts.Hook, r: kernels.Routes, ride: []const A) !?A {
    if (comptime supported) return moeOnHost(sc, m, st, li, ly, h, hk, r, ride) else unreachable; // as `deinit`
}

fn moeOnHost(sc: *Scope, m: *const dsv.Model, st: *AneState, li: usize, ly: *const dsv.Layer, h: A, hk: dsv.Experts.Hook, r: kernels.Routes, ride: []const A) !?A {
    const rows = dim(h, 0);
    if (st.disabled or m.tap != null or li >= st.layers or rows != st.chunk_rows) return null;
    if (li == 0) {
        // The previous chunk's last covered layer may still be reading the output planes (`layerOut`).
        try mlx.check(mlx.mlx_synchronize(m.s));
        st.chunk = .{};
        st.chunks += 1;
    }
    const x = try if (st.pack) pack(sc, st, h) else input(sc, st, h);
    try dsv.submit(x);
    // The router's picks queued right behind the input (with the FFN's `ride`, as the decode path sends them): the
    // host's read of them then finds them done, so the wait for `x` adds no GPU bubble before the routed launches.
    try dsv.submitWith(r.idx, ride);
    // The ANE starts as soon as the layer's input is there, so it runs beside the host's read of the picks and the
    // routed launches alike.
    try mlx.check(mlx.mlx_array_eval(x));
    const ptr: ?[*]const f16 = if (st.pack) null else @ptrCast(@alignCast(mlx.mlx_array_data_uint8(x) orelse return error.HostRead));
    var pend: Programs.Pending = .{};
    st.progs.start(&pend, li, ptr);
    const routed = hk.apply(hk.ctx, sc, li, h, r.idx, r.wts, r.logits, .none) catch |e| {
        _ = pend.join();
        return e;
    };
    dsv.submit(routed) catch |e| {
        _ = pend.join();
        return e;
    };
    const t0 = nowNs();
    const ran = pend.join();
    const join_ns = nowNs() - t0;
    if (!ran.ok) {
        st.fails += 1;
        if (st.fails == 1) log.warn("[sushi_ds41] ane shared: eval failed at layer {d}; this layer falls back to the GPU\n", .{li});
        if (st.fails >= 8) {
            st.disabled = true;
            log.warn("[sushi_ds41] ane shared: {d} failures, off for the rest of this run\n", .{st.fails});
        }
        const out = try sc.astype(try sc.add(routed, try sc.astype(try dsv.sharedExpert(sc, m, li, ly, h), .float32)), m.act);
        try layerOut(m, out);
        return out;
    }
    st.evals += st.progs.n_units;
    const y = try output(sc, st, rows, dim(h, 1));
    const out = try sc.astype(try sc.add(routed, y), m.act);
    try layerOut(m, out);
    note(st, li, rows, @max(ran.unit_ns[0], ran.unit_ns[1]), join_ns);
    return out;
}

/// One line per chunk once its last covered layer ran: the units' time and the host's wait for them, per layer.
fn note(st: *AneState, li: usize, rows: usize, ane_ns: u64, join_ns: u64) void {
    const a = &st.chunk;
    a.layers += 1;
    a.ane += ane_ns;
    a.ane_max = @max(a.ane_max, ane_ns);
    a.join += join_ns;
    if (li + 1 != st.layers) return;
    const n: f64 = @floatFromInt(a.layers);
    log.info("[sushi_ds41] ane shared: chunk {d} rows={d} layers={d} per-layer ms: ane {d:.2} (max {d:.2}) join {d:.2}; evals {d} fails {d}\n", .{
        st.chunks, rows, a.layers, @as(f64, @floatFromInt(a.ane)) / 1e6 / n, @as(f64, @floatFromInt(a.ane_max)) / 1e6, @as(f64, @floatFromInt(a.join)) / 1e6 / n, st.evals, st.fails,
    });
}
