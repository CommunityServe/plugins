//! copied_from: src/sdk/blocks/hc/contract_prefill.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The hc prefill read's NAX launches (`hc_prefill_nax.zig`) in the kernel contract: each bit-equal to the MLX chain it
//! replaces at its served shapes, and held to float64 at gates 1 and 4. Shapes: rows = tokens, n = output columns, k =
//! the reduction (the down launches: n = R, k = HC H; the up launches: n = H, k = R); normInv's as hc_prefill_norm's.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const device = @import("../../../mlx/device.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const hpn = @import("hc_prefill_nax.zig");
const hp = @import("hc_prefill.zig");
const area = @import("contract.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const HC = area.HC;
const HID = area.HID;
const K = area.K;
const R = area.R;

fn cInt(v: usize) c_int {
    return @intCast(v);
}

const kernels = std.enums.values(hpn.Kernel);
var launches: [kernels.len]metal.Launch = undefined;

/// Threadgroup bytes at the largest instantiation: the split partials' 28 KB budget, the staged rows, the up stage.
fn tgBytes(k: hpn.Kernel) u32 {
    return switch (k) {
        .nax_down, .fuse_down => 28 * 1024,
        .fuse_down_norm => 2 * 64 * 40 * 2,
        .rw_down => @intCast(hpn.rwStageBytes(2, 2, 2)),
        .nax_upmix, .fuse_upmix => 2 * 4 * 32 * 40 * 2,
        .norm_inv => 128,
        .read_down, .read_upmix, .rw_upmix => 0,
    };
}

/// The launch the wrapper builds; the NAX kernels sit on MLX's steel headers (composed at runtime) and tensor ops.
fn declare(k: hpn.Kernel) !*const metal.Launch {
    const t = hpn.text(k);
    const plain = k == .norm_inv;
    launches[@intFromEnum(k)] = .{
        .name = t.name,
        .inputs = t.inputs,
        .outputs = t.outputs,
        .source = t.source,
        .header = try hpn.header(t.header),
        .tg_bytes = tgBytes(k),
        .m1 = if (plain) 1024 else 0,
        .m2 = if (plain) 1024 else 0,
        .offline = plain,
        .min_gen = if (plain) 13 else 17,
    };
    return &launches[@intFromEnum(k)];
}

/// A dispatch as the contract's geometry: grid, threadgroup and template args in the wrapper's order.
fn geometryOf(d: hpn.Dispatch) metal.Geometry {
    var g = metal.Geometry{ .grid = d.grid, .tg = .{ d.threads, 1, 1 } };
    if (d.t) g.dtype("T", .bfloat16);
    for (d.ints[0..d.n]) |a| g.int(a.name, a.v);
    return g;
}

/// hc_prefill_norm.metal's normalized value and its two roundings: bf16(f32(bf16(f32(x) inv)) w).
fn normValue(x: f64, inv: f64, w: f64) f64 {
    return host.rt(.bfloat16, host.rt(.bfloat16, host.rt(.float32, x * inv)) * w);
}

/// Gate 4's integer rows of `n` values, each summing to exactly 0, so fp32 sums are exact in any order: row 0 is `big`,
/// ones, then minus their total in powers of two, which no fp16 or bf16 order (pairwise included) sums back to 0; the
/// others hold random integers (the first `wide` multiples of 16 past 2048) and their negations shuffled.
fn zeroSumRows(rng: std.Random, v: []f64, n: usize, big: f64, wide: usize) void {
    const h = n / 2;
    for (0..v.len / n) |r| {
        const row = v[r * n ..][0..n];
        if (r == 0) {
            row[0] = big;
            @memset(row[1..h], 1);
            var rem: u64 = @intFromFloat(big + @as(f64, @floatFromInt(h - 1)));
            var i = h;
            while (rem > 0) : (i += 1) {
                const p = @as(u64, 1) << @intCast(63 - @clz(rem));
                row[i] = -@as(f64, @floatFromInt(p));
                rem -= p;
            }
            @memset(row[i..], 0);
            continue;
        }
        for (row[0..h], 0..) |*x, j| x.* = @floatFromInt(if (j < wide) 16 * (128 + rng.uintLessThan(u32, 128)) else 1 + rng.uintLessThan(u32, 64));
        for (row[h..], row[0..h]) |*y, x| y.* = -x;
        rng.shuffle(f64, row[h..]);
    }
}

/// `table` at the bits of `v` rounded to bf16 (the launches' lookup after MLX's store rounding).
fn lookup(table: []const f64, v: f64) f64 {
    return table[spec.toBf16(@floatCast(v))];
}

/// x . w in float64 (with its terms' magnitude), or with `acc` its sum in that accumulator and order.
fn dotIn(x: []const f64, w: []const f64, terms: []f64, acc: ?spec.Accum, order: spec.Order) [2]f64 {
    const d = host.dot(x, w);
    const ac = acc orelse return d;
    for (terms, x, w) |*t, a, b| t.* = a * b;
    return .{ spec.sumIn(ac, order, terms), d[1] };
}

/// `table` (a [65536] bf16 device array) bound as an input byte for byte, and its values on the host.
fn bindTable(rig: *metal.Rig, a: std.mem.Allocator, table: mlx.mlx_array) !struct { mlx.mlx_array, []f64 } {
    const raw = try a.dupe(u8, try rig.bytesOf(&.{table}));
    const vals = try a.alloc(f64, 65536);
    for (vals, 0..) |*v, i| v.* = spec.read(.bf16, raw, i);
    return .{ try rig.inputRaw(.bfloat16, &.{65536}, raw), vals };
}

/// The chain's silu (x sigmoid(x)) and inject gate (2 sigmoid(x)) of every bf16 pattern, bound with their values.
fn bindFuseTables(rig: *metal.Rig, a: std.mem.Allocator, out: *[2]mlx.mlx_array, vals: *[2][]f64) !void {
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    var g = ops.Graph.init(rig.a, rig.stream());
    defer g.deinit();
    const x = try g.adopt(mlx.mlx_array_new_data(&bits, &[_]c_int{65536}, 1, .bfloat16));
    const sig = try g.sigmoid(x);
    const tabs = [2]mlx.mlx_array{ try g.mul(x, sig), try g.mul(sig, try g.scalar(2, .bfloat16)) };
    for (tabs, out, vals) |t, *o, *v| {
        const b = try bindTable(rig, a, t);
        o.* = b[0];
        v.* = b[1];
    }
}

/// `values` as u32 words of bf16 rows (the wrappers' `down_rows` / `up_rows`).
fn bindRows(rig: *metal.Rig, a: std.mem.Allocator, values: []const f64, rows: usize, cols: usize) !mlx.mlx_array {
    const raw = try a.alloc(u8, values.len * 2);
    for (values, 0..) |v, i| std.mem.writeInt(u16, raw[2 * i ..][0..2], spec.toBf16(@floatCast(v)), .little);
    return rig.inputRaw(.uint32, &.{ cInt(rows), cInt(cols / 2) }, raw);
}

// ── the down launches: naxDown, fuseDown, readDown, fuseDownNorm, rwDown ──

const DownKind = enum { nax, fuse, read, fuse_norm, rw };

const Down = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: DownKind,
    inj: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    /// The A rows [M, K]: the normalized rows (norm kinds: hcpf_norm of the stream `xs`, its row scales and weight).
    rows: []f64 = &.{},
    w: []f64 = &.{},
    iw: []f64 = &.{},
    tabs: [2][]f64 = .{ &.{}, &.{} },
    args: [8]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: DownKind, inj: bool) Down {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .inj = inj, .arena = .init(a) };
    }

    pub fn deinit(self: *Down) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn norm(self: *const Down) bool {
        return self.kind == .fuse_norm or self.kind == .rw;
    }

    /// The one-chain launches take any row count here: gates 1, 2 and 4 run a few rows (the wrappers serve them only
    /// where MLX runs one chain over K).
    fn dispatch(self: *const Down, shape: Shape) ?hpn.Dispatch {
        const m = cInt(shape.rows);
        const n = cInt(shape.n);
        const k = cInt(shape.k);
        if (shape.k % HC != 0) return null;
        const h = cInt(shape.k / HC);
        return switch (self.kind) {
            .nax => hpn.naxDownDispatch(m, n, k),
            .fuse => hpn.fuseDownDispatch(m, n, k, HC, self.inj),
            .read => hpn.readDownDispatch(m, n, k, HC),
            .fuse_norm => hpn.fuseDownNormDispatch(m, n, HC, h, self.inj),
            .rw => hpn.rwDownDispatch(m, n, HC, h),
        };
    }

    pub fn geometry(self: *const Down, shape: Shape) ?metal.Geometry {
        var g = geometryOf(self.dispatch(shape) orelse return null);
        const m = cInt(shape.rows);
        g.out(&.{ 1, m, cInt(shape.n) }, .bfloat16);
        if (self.kind == .nax) return g;
        if (self.inj) {
            g.out(&.{ 1, m, HC, 1 }, .bfloat16);
        } else {
            g.out(&.{1}, .bfloat16);
            g.unread = 0b10;
        }
        return g;
    }

    pub fn prepare(self: *Down, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        const h = k / HC;
        const x = try a.alloc(f64, m * k);
        self.w = try a.alloc(f64, n * k);
        self.iw = try a.alloc(f64, k * HC);
        if (fill == .stress) {
            // Zero-sum rows and weights of alternating sign by column: every product is 0 exactly in fp32 (silu 0, an
            // allowance at the output's floor), and any fp16 or bf16 sum off 0 leaves one column of a pair positive.
            zeroSumRows(rng, x, k, 16384, 0);
            for (0..n) |j| @memset(self.w[j * k ..][0..k], if (j % 2 == 0) 1 else -1);
            @memset(self.iw, 1);
        } else {
            host.uniformInto(rng, x, -3, 3, .bfloat16);
            host.uniformInto(rng, self.w, -0.04, 0.04, .bfloat16);
            host.uniformInto(rng, self.iw, -0.05, 0.05, .bfloat16);
            if (fill == .nonfinite) host.plant(x, m, k);
        }
        var invs: []f64 = &.{};
        var nw: []f64 = &.{};
        self.rows = x;
        if (self.norm()) {
            invs = try a.alloc(f64, m * HC);
            nw = try a.alloc(f64, HC * h);
            if (fill == .stress) {
                @memset(invs, 1);
                @memset(nw, 1);
            } else {
                host.uniformInto(rng, invs, 0.25, 1, .float32);
                host.uniformInto(rng, nw, 0.5, 1.5, .bfloat16);
            }
            self.rows = try a.alloc(f64, m * k);
            for (0..m) |r| for (0..k) |c| {
                self.rows[r * k + c] = normValue(x[r * k + c], invs[r * HC + c / h], nw[c]);
            };
        }
        const rig = &self.rig;
        rig.reset();
        const S = cInt(m);
        const xin = try rig.input(.bfloat16, &.{ 1, S, cInt(k) }, x);
        const w = try bindRows(rig, a, self.w, n, k);
        const rows = try host.intScalar(rig, S);
        if (self.kind == .nax) {
            self.args[0..3].* = .{ xin, w, rows };
            self.n_args = 3;
            return true;
        }
        var t: [2]mlx.mlx_array = undefined;
        try bindFuseTables(rig, a, &t, &self.tabs);
        // Without an inject the launch never reads `iw`; the silu table stands in, as the wrapper binds it.
        const iw = if (self.inj) try rig.input(.bfloat16, &.{ cInt(k), HC }, self.iw) else t[0];
        if (self.norm()) {
            const inv_in = try rig.input(.float32, &.{cInt(m * HC)}, invs);
            const nw_in = try rig.input(.bfloat16, &.{ HC, cInt(h) }, nw);
            self.args = .{ xin, inv_in, nw_in, w, iw, t[0], t[1], rows };
            self.n_args = 8;
        } else {
            self.args[0..6].* = .{ xin, w, iw, t[0], t[1], rows };
            self.n_args = 6;
        }
        return true;
    }

    pub fn run(self: *Down) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Down) usize {
        const m = self.shape.rows;
        return m * self.shape.n + if (self.kind != .nax and self.inj) m * HC else 0;
    }

    /// act (the product, or the silu table at its bf16 rounding) then inj (the gate table at the inject product's).
    fn outputsIn(self: *const Down, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const sh = self.shape;
        const a = self.rig.a;
        const terms = try a.alloc(f64, sh.k);
        defer a.free(terms);
        const col = try a.alloc(f64, sh.k);
        defer a.free(col);
        for (0..sh.rows) |r| {
            const x = self.rows[r * sh.k ..][0..sh.k];
            for (0..sh.n) |j| {
                const y = dotIn(x, self.w[j * sh.k ..][0..sh.k], terms, acc, order);
                const at = r * sh.n + j;
                want[at] = if (self.kind == .nax) y[0] else lookup(self.tabs[0], y[0]);
                if (mag) |mm| mm[at] = y[1];
            }
            if (self.kind == .nax or !self.inj) continue;
            for (0..HC) |c| {
                for (col, 0..) |*v, i| v.* = self.iw[i * HC + c];
                const y = dotIn(x, col, terms, acc, order);
                const at = sh.rows * sh.n + r * HC + c;
                want[at] = lookup(self.tabs[1], y[0]);
                if (mag) |mm| mm[at] = y[1];
            }
        }
    }

    pub fn reference(self: *const Down, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const Down, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The norm's two roundings as MLX ops: bf16(f32(bf16(f32(xs) invs)) f32(nw)), as [1, M, K].
    fn normRows(self: *Down, g: *ops.Graph) !mlx.mlx_array {
        const m = cInt(self.shape.rows);
        const h = cInt(self.shape.k / HC);
        const x4 = try g.astype(try g.reshape(self.args[0], &.{ m, HC, h }), .float32);
        const t1 = try g.astype(try g.mul(x4, try g.reshape(self.args[1], &.{ m, HC, 1 })), .bfloat16);
        const t2 = try g.mul(try g.astype(t1, .float32), try g.astype(self.args[2], .float32));
        return g.reshape(try g.astype(t2, .bfloat16), &.{ 1, m, cInt(self.shape.k) });
    }

    /// The chain: MLX's matmul of the rows and the down rows' transposed view, then x sigmoid(x); MLX's matmul of the
    /// rows and the inject weight, then 2 sigmoid.
    pub fn replaced(self: *Down) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const off: usize = if (self.norm()) 3 else 1;
        const rows = if (self.norm()) try self.normRows(&g) else self.args[0];
        const wt = try g.call(mlx.mlx_transpose, .{try g.call(mlx.mlx_view, .{ self.args[off], mlx.mlx_dtype.bfloat16 })});
        const y = try g.matmul(rows, wt);
        if (self.kind == .nax) return self.rig.bytesOf(&.{y});
        const act = try g.mul(y, try g.sigmoid(y));
        if (!self.inj) return self.rig.bytesOf(&.{act});
        const raw = try g.matmul(rows, self.args[off + 1]);
        const gate = try g.mul(try g.sigmoid(raw), try g.scalar(2, .bfloat16));
        return self.rig.bytesOf(&.{ act, gate });
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Down) ![]const u8 {
        const s = self.rig.stream();
        const S = cInt(self.shape.rows);
        const x = self.args;
        if (self.kind == .nax) {
            const y = (try hpn.naxDown(s, x[0], x[1], 1, S)) orelse return error.Declined;
            defer _ = mlx.mlx_array_free(y);
            return self.rig.bytesOf(&.{y});
        }
        const none: mlx.mlx_array = .{ .ctx = null };
        const got = switch (self.kind) {
            .fuse => try hpn.fuseDown(s, x[0], x[1], if (self.inj) x[2] else none, .{ .silu = x[3], .gate = x[4] }, 1, S, HC),
            .read => try hpn.readDown(s, x[0], x[1], x[2], .{ .silu = x[3], .gate = x[4] }, 1, S, HC),
            .fuse_norm => try hpn.fuseDownNorm(s, x[0], x[1], x[2], x[3], if (self.inj) x[4] else none, .{ .silu = x[5], .gate = x[6] }, 1, S),
            .rw => try hpn.rwDown(s, x[0], x[1], x[2], x[3], x[4], .{ .silu = x[5], .gate = x[6] }, 1, S),
            .nax => unreachable,
        } orelse return error.Declined;
        defer got.deinit();
        return if (self.inj) self.rig.bytesOf(&.{ got.act, got.inj }) else self.rig.bytesOf(&.{got.act});
    }
};

// ── the up + mix launches: naxUpMix, readUpMix, upMixNorm, rwUpMix ──

const UpKind = enum { nax, read, norm, rw };

const UpMix = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: UpKind,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    act: []f64 = &.{},
    w: []f64 = &.{},
    /// The normalized streams [M, HC, H] (norm kinds: hcpf_norm of the stream, its row scales and weight).
    normed: []f64 = &.{},
    tab: []f64 = &.{},
    args: [7]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, kind: UpKind) UpMix {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .kind = kind, .arena = .init(a) };
    }

    pub fn deinit(self: *UpMix) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn norm(self: *const UpMix) bool {
        return self.kind == .norm or self.kind == .rw;
    }

    pub fn geometry(self: *const UpMix, shape: Shape) ?metal.Geometry {
        const m = cInt(shape.rows);
        const h = cInt(shape.n);
        const r = cInt(shape.k);
        const d = switch (self.kind) {
            .nax, .norm => hpn.upDispatch(m, r, HC, h),
            .read => hpn.regDispatch(m, r, HC, h, false),
            .rw => hpn.regDispatch(m, r, HC, h, true),
        } orelse return null;
        var g = geometryOf(d);
        g.out(&.{ 1, m, h }, .bfloat16);
        return g;
    }

    pub fn prepare(self: *UpMix, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const m = shape.rows;
        const h = shape.n;
        const r = shape.k;
        self.act = try a.alloc(f64, m * r);
        self.w = try a.alloc(f64, HC * h * r);
        const x = try a.alloc(f64, m * HC * h);
        if (fill == .stress) {
            // Zero-sum rows, stream weights of alternating sign and normalized values paired by sign: the mix is 0 exactly
            // in fp32 (every sigmoid 1/2), while an fp16 or bf16 up off 0 pulls each pair's sigmoids apart.
            zeroSumRows(rng, self.act, r, 4096, 32);
            for (0..HC * h) |row| @memset(self.w[row * r ..][0..r], if ((row / h) % 2 == 0) 1 else -1);
            for (0..m) |t| for (0..HC / 2) |pair| for (0..h) |c| {
                const mag: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
                x[(t * HC + 2 * pair) * h + c] = mag;
                x[(t * HC + 2 * pair + 1) * h + c] = -mag;
            };
        } else {
            host.uniformInto(rng, self.act, -3, 3, .bfloat16);
            host.uniformInto(rng, self.w, -0.1, 0.1, .bfloat16);
            host.uniformInto(rng, x, -3, 3, .bfloat16);
            if (fill == .nonfinite) {
                host.plant(self.act, m, r);
                host.plant(x, m * HC, h);
            }
        }
        var invs: []f64 = &.{};
        var nw: []f64 = &.{};
        self.normed = x;
        if (self.norm()) {
            invs = try a.alloc(f64, m * HC);
            nw = try a.alloc(f64, HC * h);
            if (fill == .stress) {
                @memset(invs, 1);
                @memset(nw, 1);
            } else {
                host.uniformInto(rng, invs, 0.25, 1, .float32);
                host.uniformInto(rng, nw, 0.5, 1.5, .bfloat16);
            }
            self.normed = try a.alloc(f64, m * HC * h);
            for (0..m * HC) |row| for (0..h) |c| {
                self.normed[row * h + c] = normValue(x[row * h + c], invs[row], nw[(row % HC) * h + c]);
            };
        }
        const rig = &self.rig;
        rig.reset();
        const S = cInt(m);
        const act = try rig.input(.bfloat16, &.{ 1, S, cInt(r) }, self.act);
        const w = try bindRows(rig, a, self.w, HC * h, r);
        const tab = try bindTable(rig, a, try hp.sigmoidTable(rig.stream()));
        self.tab = tab[1];
        const rows = try host.intScalar(rig, S);
        if (self.norm()) {
            const xs = try rig.input(.bfloat16, &.{ 1, S, cInt(HC * h) }, x);
            const inv_in = try rig.input(.float32, &.{cInt(m * HC)}, invs);
            const nw_in = try rig.input(.bfloat16, &.{ HC, cInt(h) }, nw);
            self.args = .{ act, w, xs, inv_in, nw_in, tab[0], rows };
            self.n_args = 7;
        } else {
            const normed = try rig.input(.bfloat16, &.{ 1, S, HC, cInt(h) }, x);
            self.args[0..5].* = .{ act, w, normed, tab[0], rows };
            self.n_args = 5;
        }
        return true;
    }

    pub fn run(self: *UpMix) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const UpMix) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Per (row, column): each stream's up rounded to bf16, its sigmoid from the table, bf16 after every product and
    /// every sum in stream order, times bf16(1 / HC) (the last rounding left to the bound).
    fn outputsIn(self: *const UpMix, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const sh = self.shape;
        const h = sh.n;
        const r = sh.k;
        const terms = try self.rig.a.alloc(f64, r);
        defer self.rig.a.free(terms);
        const inv = host.rt(.bfloat16, 1.0 / @as(f64, HC));
        for (0..sh.rows) |t| for (0..h) |c| {
            var value: f64 = 0;
            var m: f64 = 0;
            for (0..HC) |s| {
                const up = dotIn(self.act[t * r ..][0..r], self.w[(s * h + c) * r ..][0..r], terms, acc, order);
                const sig = lookup(self.tab, host.rt(.bfloat16, up[0]));
                const product = host.rt(.bfloat16, sig * self.normed[(t * HC + s) * h + c]);
                value = if (s == 0) product else host.rt(.bfloat16, product + value);
                m += @abs(product);
            }
            want[t * h + c] = value * inv;
            if (mag) |mm| mm[t * h + c] = m;
        };
    }

    pub fn reference(self: *const UpMix, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const UpMix, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    /// The chain: MLX's matmul of act and the up rows' transposed view, then sigmoid -> multiply -> mean over the
    /// streams (the mix kernel's bits, `hc_prefill_mix bf16`), the normalized streams as MLX ops on the norm kinds.
    pub fn replaced(self: *UpMix) ![]const u8 {
        var g = ops.Graph.init(self.rig.a, self.rig.stream());
        defer g.deinit();
        const m = cInt(self.shape.rows);
        const h = cInt(self.shape.n);
        const wt = try g.call(mlx.mlx_transpose, .{try g.call(mlx.mlx_view, .{ self.args[1], mlx.mlx_dtype.bfloat16 })});
        const up4 = try g.reshape(try g.matmul(self.args[0], wt), &.{ 1, m, HC, h });
        const normed = if (self.norm()) blk: {
            const x4 = try g.astype(try g.reshape(self.args[2], &.{ 1, m, HC, h }), .float32);
            const t1 = try g.astype(try g.mul(x4, try g.reshape(self.args[3], &.{ 1, m, HC, 1 })), .bfloat16);
            const t2 = try g.mul(try g.astype(t1, .float32), try g.astype(self.args[4], .float32));
            break :blk try g.astype(t2, .bfloat16);
        } else self.args[2];
        return self.rig.bytesOf(&.{try g.mean(try g.mul(try g.sigmoid(up4), normed), 2, false)});
    }

    fn wrapper(self: *UpMix) ![]const u8 {
        const s = self.rig.stream();
        const S = cInt(self.shape.rows);
        const x = self.args;
        const got = switch (self.kind) {
            .nax => try hpn.naxUpMix(s, x[0], x[1], x[2], 1, S),
            .read => try hpn.readUpMix(s, x[0], x[1], x[2], 1, S),
            .norm => try hpn.upMixNorm(s, x[0], x[1], x[2], x[3], x[4], 1, S),
            .rw => try hpn.rwUpMix(s, x[0], x[1], x[2], x[3], x[4], 1, S),
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        return self.rig.bytesOf(&.{got});
    }
};

// ── normInv ──

/// hc_prefill_norm's shapes: rows = the streams (1..8), n = tokens, k = the hidden width.
const NormInv = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    wr: bool,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    x: []f64 = &.{},
    wo: []f64 = &.{},
    wi: []f64 = &.{},
    args: [4]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, wr: bool) NormInv {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .wr = wr, .arena = .init(a) };
    }

    pub fn deinit(self: *NormInv) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const NormInv, shape: Shape) ?metal.Geometry {
        if (shape.n <= 16) return null;
        const d = hpn.normInvDispatch(cInt(shape.n), cInt(shape.rows), cInt(shape.k), self.wr) orelse return null;
        var g = geometryOf(d);
        const s = cInt(shape.n);
        if (self.wr) g.out(&.{ 1, s, cInt(shape.rows * shape.k) }, .bfloat16) else g.out(&.{ 1, 1, 1 }, .bfloat16);
        g.out(&.{cInt(shape.n * shape.rows)}, .float32);
        if (!self.wr) g.unread = 0b01;
        return g;
    }

    pub fn prepare(self: *NormInv, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const s = shape.n;
        const width = shape.rows * shape.k;
        self.x = try host.uniform(a, rng, s * width, -3, 3, .bfloat16);
        self.wo = try host.uniform(a, rng, s * shape.k, -2, 2, .bfloat16);
        self.wi = try host.uniform(a, rng, s * shape.rows, 0, 2, .bfloat16);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.x, s * shape.rows, shape.k),
            .stress => {
                for (self.x) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -64, 64));
                @memset(self.wo, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const S = cInt(s);
        const x = try rig.input(.bfloat16, &.{ 1, S, cInt(width) }, self.x);
        self.args = .{
            x,
            try rig.scalar(area.EPS),
            if (self.wr) try rig.input(.bfloat16, &.{ 1, S, 1, cInt(shape.k) }, self.wo) else x,
            if (self.wr) try rig.input(.bfloat16, &.{ 1, S, cInt(shape.rows), 1 }, self.wi) else x,
        };
        return true;
    }

    pub fn run(self: *NormInv) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const NormInv) usize {
        const n = self.shape.n * self.shape.rows;
        return if (self.wr) n * self.shape.k + n else n;
    }

    /// The stream after the pending write (the chain's exact values), row `row` (token * streams + stream).
    fn streamAt(self: *const NormInv, row: usize, i: usize) f64 {
        const h = self.shape.k;
        const x = self.x[row * h + i];
        if (!self.wr) return x;
        const token = row / self.shape.rows;
        return host.rt(.bfloat16, x + host.rt(.bfloat16, self.wo[token * h + i] * self.wi[row]));
    }

    fn outputsIn(self: *const NormInv, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) !void {
        const h = self.shape.k;
        const rows = self.shape.n * self.shape.rows;
        const sq = try self.rig.a.alloc(f64, h);
        defer self.rig.a.free(sq);
        const base: usize = if (self.wr) rows * h else 0;
        for (0..rows) |row| {
            for (sq, 0..) |*q, i| {
                const v = self.streamAt(row, i);
                q.* = v * v;
                if (self.wr) {
                    want[row * h + i] = v;
                    if (mag) |m| m[row * h + i] = 0;
                }
            }
            const ss = if (acc) |ac| spec.sumIn(ac, order, sq) else host.sum(sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(h)) + area.EPS);
            want[base + row] = inv;
            // The squares are positive: the sum's relative error reaches the scale halved.
            if (mag) |m| m[base + row] = @abs(inv) / 2;
        }
    }

    pub fn reference(self: *const NormInv, want: []f64, mag: []f64) !void {
        try self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const NormInv, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.outputsIn(out, null, acc, order);
    }

    fn wrapper(self: *NormInv) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        // The norm weight gives the wrapper its shape only.
        const w = try g.zeros(&.{ cInt(self.shape.rows), cInt(self.shape.k) }, .bfloat16);
        const x = self.args;
        const got = (try hpn.normInv(s, x[0], w, x[1], 1, cInt(self.shape.n), if (self.wr) .{ .out = x[2], .inj = x[3] } else null)) orelse return error.Declined;
        defer got.deinit();
        return if (self.wr) self.rig.bytesOf(&.{ got.stream, got.invs }) else self.rig.bytesOf(&.{got.invs});
    }
};

// ── specs ──

fn depthK(shape: Shape) usize {
    return shape.k;
}

fn depthUp(shape: Shape) usize {
    return shape.k + HC;
}

/// Flash Next's down product at a split-K tail (17 rows) and one chain over K (3414: MLX's edge).
const down_split: []const Shape = &.{ .{ .rows = 17, .n = R, .k = K }, .{ .rows = 3414, .n = R, .k = K } };
/// The one-chain launches: MLX's edge and the 8192-row chunk.
const down_chain: []const Shape = &.{ .{ .rows = 3414, .n = R, .k = K }, .{ .rows = 8192, .n = R, .k = K } };
const up_rows: []const Shape = &.{ .{ .rows = 17, .n = HID, .k = R }, .{ .rows = 1359, .n = HID, .k = R } };
const up_fold: []const Shape = &.{ .{ .rows = 3414, .n = HID, .k = R }, .{ .rows = 8192, .n = HID, .k = R } };

fn naxSpec(comptime name: []const u8, comptime replaces: []const u8, depth: *const fn (Shape) usize, mids: f64, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .mids = mids,
        .out = .bf16,
        .replaces = replaces,
        .served = served,
        .nax = true,
    };
}

const down_chain_text = "matmul (MLX's NAX gemm, split-K where MLX splits)";
const fuse_text = down_chain_text ++ " -> silu; matmul -> 2 sigmoid";
const fold_text = "astype -> multiply -> astype -> multiply -> astype -> " ++ fuse_text;
const up_text = "matmul -> sigmoid -> multiply -> mean";

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(Down, .init(a, try declare(.nax_down), .nax, false), naxSpec("hc_prefill_nax_down bf16", down_chain_text, depthK, 0, down_split), null);
    try set.add(Down, .init(a, try declare(.fuse_down), .fuse, true), naxSpec("hc_prefill_fuse_down bf16", fuse_text, depthK, 1, down_split), null);
    try set.add(Down, .init(a, try declare(.fuse_down), .fuse, false), naxSpec("hc_prefill_fuse_down bf16 (no inject)", down_chain_text ++ " -> silu", depthK, 1, down_split), null);
    try set.add(Down, .init(a, try declare(.read_down), .read, true), naxSpec("hc_prefill_read_down bf16", fuse_text, depthK, 1, down_chain), null);
    try set.add(Down, .init(a, try declare(.fuse_down_norm), .fuse_norm, true), naxSpec("hc_prefill_fuse_down_norm bf16", fold_text, depthK, 3, down_chain), null);
    try set.add(Down, .init(a, try declare(.rw_down), .rw, true), naxSpec("hc_prefill_rw_down bf16", fold_text, depthK, 3, down_chain), null);
    try set.add(UpMix, .init(a, try declare(.nax_upmix), .nax), naxSpec("hc_prefill_nax_upmix bf16", up_text, depthUp, 1 + HC, up_rows), null);
    try set.add(UpMix, .init(a, try declare(.read_upmix), .read), naxSpec("hc_prefill_read_upmix bf16", up_text, depthUp, 1 + HC, up_rows), null);
    try set.add(UpMix, .init(a, try declare(.fuse_upmix), .norm), naxSpec("hc_prefill_fuse_upmix bf16", "astype -> multiply -> astype -> multiply -> astype; " ++ up_text, depthUp, 3 + HC, up_fold), null);
    try set.add(UpMix, .init(a, try declare(.rw_upmix), .rw), naxSpec("hc_prefill_rw_upmix bf16", "astype -> multiply -> astype -> multiply -> astype; " ++ up_text, depthUp, 3 + HC, up_fold), null);
    inline for (.{ false, true }) |wr| try set.add(NormInv, .init(a, try declare(.norm_inv), wr), .{
        .name = if (wr) "hc_prefill_norm_inv (pending write)" else "hc_prefill_norm_inv",
        .accum = .f32,
        .depth = area.prefillDepth,
        // The mean, + eps, precise rsqrt, and its ulp.
        .ops = 4,
        .out = .f32,
        .outs = if (wr) &.{ .{ .elem = .bf16, .count = streamCount }, .{ .elem = .f32, .count = invCount } } else &.{},
        .served = area.prefill_shapes,
    }, null);
}

fn streamCount(s: Shape) usize {
    return s.rows * s.n * s.k;
}

fn invCount(s: Shape) usize {
    return s.rows * s.n;
}

// ── tests ──

/// The declared launch at a served shape, against the shipped wrapper's bytes.
fn drift(t: anytype, shape: Shape) !void {
    const a = std.testing.allocator;
    try std.testing.expect(try t.prepare(shape, .random, 7));
    try t.run();
    const ours = try a.dupe(u8, t.rig.outputBytes());
    defer a.free(ours);
    try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
}

test "kernel contract hc prefill: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    inline for (.{ false, true }) |wr| {
        var n = NormInv.init(a, try declare(.norm_inv), wr);
        defer n.deinit();
        try drift(&n, area.prefill_shapes[0]);
    }
    if (!device.naxAvailable()) return error.SkipZigTest;
    const downs = [_]struct { hpn.Kernel, DownKind, bool, Shape }{
        .{ .nax_down, .nax, false, down_split[0] },
        .{ .nax_down, .nax, false, down_split[1] },
        .{ .fuse_down, .fuse, true, down_split[0] },
        .{ .fuse_down, .fuse, false, down_split[1] },
        .{ .read_down, .read, true, down_chain[0] },
        .{ .fuse_down_norm, .fuse_norm, true, down_chain[0] },
        .{ .rw_down, .rw, true, down_chain[0] },
    };
    for (downs) |c| {
        var d = Down.init(a, try declare(c[0]), c[1], c[2]);
        defer d.deinit();
        try drift(&d, c[3]);
    }
    const ups = [_]struct { hpn.Kernel, UpKind, Shape }{
        .{ .nax_upmix, .nax, up_rows[0] },
        .{ .read_upmix, .read, up_rows[0] },
        .{ .fuse_upmix, .norm, up_fold[0] },
        .{ .rw_upmix, .rw, up_fold[0] },
    };
    for (ups) |c| {
        var u = UpMix.init(a, try declare(c[0]), c[1]);
        defer u.deinit();
        try drift(&u, c[2]);
    }
}
