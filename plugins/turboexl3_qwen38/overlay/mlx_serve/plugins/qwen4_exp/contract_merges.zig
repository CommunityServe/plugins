//! copied_from: src/plugins/qwen4_exp/contract_merges.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's decode-merge kernels in the kernel contract, each bit-equal to the op chain it replaces: a placed MTP
//! draft row's key visibility (`mlxserve_qsa_place_vis`: rows = the pick list's length, n = kv), the QSA key ring's
//! rebuild (`mlxserve_qsa_ring_cat`: rows = the keys a forward adds, n = the 128 key dims, k = the ring's rows held),
//! and the PLE layer's fused tail (`mlxserve_ple_dec_*`: rows = tokens, n = the hc * hidden stream width, k = the conv
//! state's rows for the concat).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const merges = @import("qsa/merges.zig");
const ple = @import("ple.zig");
const head_place = @import("qsa/head_place.zig");
const Trunk = @import("trunk.zig").Trunk;
const qsaMaskFromBlocks = @import("../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const sigmoidTableFor = @import("../../sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const QSA_RING_ROWS = @import("qsa/history.zig").QSA_RING_ROWS;

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;

/// The block size the contract runs at (Qwen3.8-Flash-Next's).
const RATIO = 4;
const SENTINEL: f64 = std.math.maxInt(i32);
const D = 128;

fn none(_: Shape) usize {
    return 0;
}

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

// ── a placed draft row's key visibility ──

const place_vis_launch: metal.Launch = .{
    .name = "mlxserve_qsa_place_vis",
    .inputs = &.{ "blocks", "live_end", "dead_end" },
    .outputs = &.{"mask"},
    .source = merges.PLACE_VIS_SOURCE,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

const PlaceVis = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &place_vis_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    picks: []f64 = &.{},
    live_end: i32 = 0,
    dead_end: i32 = 0,
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) PlaceVis {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *PlaceVis) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    /// The wrapper's launch: one thread a key, the step's kv wide.
    pub fn geometry(_: *const PlaceVis, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n < 2 or shape.k != 1) return null;
        const kv: c_int = @intCast(shape.n);
        var g = metal.Geometry{ .grid = .{ kv, 1, 1 }, .tg = .{ @min(kv, 256), 1, 1 } };
        g.out(&.{ 1, 1, 1, kv }, .bool_);
        g.int("KB", shape.rows);
        g.int("RATIO", RATIO);
        return g;
    }

    /// Sorted picks with INT_MAX sentinels after them, a live end and a dead end inside the history (the placed row
    /// past them). `.nonfinite` (integer ids have no NaN) plants the edges instead: block 0, the last whole block, the
    /// tail's own block, an id past every key, a live end on a block boundary and no rows past the dead end.
    pub fn prepare(self: *PlaceVis, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const kv = shape.n;
        const nb = kv / RATIO;
        self.picks = try a.alloc(f64, shape.rows);
        @memset(self.picks, SENTINEL);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (fill) {
            .random, .stress => {
                if (nb > 0) {
                    const pool = try a.alloc(usize, nb);
                    for (pool, 0..) |*p, i| p.* = i;
                    rng.shuffle(usize, pool);
                    const n = @min(shape.rows, 1 + rng.uintLessThan(usize, nb));
                    for (self.picks[0..n], pool[0..n]) |*p, b| p.* = @floatFromInt(b);
                }
                const le = rng.uintLessThan(usize, kv);
                self.live_end = @intCast(le);
                self.dead_end = @intCast(le + rng.uintLessThan(usize, kv - le));
            },
            .nonfinite => {
                const edges = [_]f64{ 0, @floatFromInt(nb -| 1), @floatFromInt(nb), @floatFromInt(nb + 7) };
                const n = @min(shape.rows, edges.len);
                for (self.picks[0..n], edges[0..n]) |*p, e| p.* = e;
                self.live_end = @intCast(kv / 2 / RATIO * RATIO);
                self.dead_end = @intCast(kv);
            },
        }
        std.mem.sort(f64, self.picks, {}, std.sort.asc(f64));
        const rig = &self.rig;
        rig.reset();
        self.args[0] = try rig.input(.int32, &.{ 1, 1, @intCast(shape.rows) }, self.picks);
        self.args[1] = try rig.input(.int32, &.{}, &.{@floatFromInt(self.live_end)});
        self.args[2] = try rig.input(.int32, &.{}, &.{@floatFromInt(self.dead_end)});
        return true;
    }

    pub fn run(self: *PlaceVis) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const PlaceVis) usize {
        return self.shape.n;
    }

    /// The row mask over the picks, then `headPlaceVisibility`'s ops: visible or recent, and not dead.
    pub fn reference(self: *const PlaceVis, want: []f64, mag: []f64) !void {
        const tail = self.shape.n / RATIO * RATIO;
        const le: i64 = self.live_end;
        const recent_from = @divFloor(le, RATIO) * RATIO;
        for (want, mag, 0..) |*w, *m, ti| {
            const t: i64 = @intCast(ti);
            const blk: f64 = @floatFromInt(ti / RATIO);
            const picked = for (self.picks) |p| (if (p == blk) break true) else false;
            const visible = ti >= tail or picked;
            const not_dead = t < le or t >= self.dead_end;
            w.* = if ((visible or t >= recent_from) and not_dead) 1 else 0;
            m.* = 0;
        }
    }

    pub fn check(self: *PlaceVis, _: spec.Gate) !?[]const u8 {
        const want = try self.arena.allocator().alloc(f64, 2 * self.count());
        try self.reference(want[0..self.count()], want[self.count()..]);
        const out = self.rig.outputBytes();
        for (want[0..self.count()], 0..) |w, i| if (spec.read(.bool, out, i) != w) return "a key's visibility differs from the placed row's";
        return null;
    }

    /// The chain on the same picks: the row mask's op chain (`qsaMaskFromBlocks`), then `headPlaceVisibility`.
    pub fn replaced(self: *PlaceVis) ![]const u8 {
        const s = self.rig.stream();
        const kv: c_int = @intCast(self.shape.n);
        const mask = try qsaMaskFromBlocks(s, self.args[0], kv, RATIO);
        defer free(mask);
        const p: Trunk.HeadPlace = .{ .live_end = self.args[1], .dead_end = self.dead_end };
        const out = try head_place.headPlaceVisibility(s, mask, &p, RATIO, kv);
        defer free(out);
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same picks.
    fn wrapper(self: *PlaceVis) ![]const u8 {
        const out = try merges.placeVis(self.rig.stream(), self.args[0], self.args[1], self.dead_end, @intCast(self.shape.n), RATIO);
        defer free(out);
        return self.rig.bytesOf(&.{out});
    }
};

// ── the key ring's rebuild ──

const ring_launch: metal.Launch = .{
    .name = "mlxserve_qsa_ring_cat",
    .inputs = &.{ "old", "kr", "zero" },
    .outputs = &.{"out"},
    .source = merges.RING_CAT_SOURCE,
    .contiguous = false,
    .tg_bytes = 0,
    .m1 = 1024,
    .m2 = 1024,
};

/// The indexer projection's width the new keys are a strided slice of: four query heads, then the key.
const PROJ_HEADS = 5;

const RingCat = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &ring_launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    old: []f64 = &.{},
    proj: []f64 = &.{},
    /// The keys' strided view of the projection input (owned).
    kr: mlx.mlx_array = .{ .ctx = null },
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) RingCat {
        return .{ .rig = metal.Rig.init(a), .arena = .init(a) };
    }

    pub fn deinit(self: *RingCat) void {
        free(self.kr);
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(_: *const RingCat, shape: Shape) ?metal.Geometry {
        const ring: usize = @intCast(QSA_RING_ROWS);
        if (shape.n != D or shape.rows < 1 or shape.rows >= ring or shape.k > ring or shape.k + shape.rows <= ring) return null;
        const n: c_int = QSA_RING_ROWS * D;
        var g = metal.Geometry{ .grid = .{ n, 1, 1 }, .tg = .{ @min(n, 256), 1, 1 } };
        g.out(&.{ 1, QSA_RING_ROWS, D }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("RING", QSA_RING_ROWS);
        return g;
    }

    /// bf16 keys with -0 among them (the owned copy's add makes them +0); `.nonfinite` plants NaN and both infinities
    /// in the kept rows and the new keys.
    pub fn prepare(self: *RingCat, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const held = shape.k;
        const add = shape.rows;
        const w = PROJ_HEADS * D;
        self.old = try a.alloc(f64, held * D);
        self.proj = try a.alloc(f64, add * w);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for ([_][]f64{ self.old, self.proj }) |xs| for (xs) |*v| {
            v.* = if (rng.uintLessThan(u32, 16) == 0) -0.0 else spec.roundTo(.bf16, rng.float(f64) * 4 - 2);
        };
        if (fill == .nonfinite) {
            const keep = @as(usize, @intCast(QSA_RING_ROWS)) - add;
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (put, 0..) |v, i| {
                self.old[(held - keep + i % keep) * D + 3 * i] = v;
                self.proj[(i % add) * w + (PROJ_HEADS - 1) * D + 5 * i] = v;
            }
        }
        const rig = &self.rig;
        rig.reset();
        free(self.kr);
        self.kr = .{ .ctx = null };
        self.args[0] = try rig.input(.bfloat16, &.{ 1, @intCast(held), D }, self.old);
        const proj = try rig.input(.bfloat16, &.{ 1, @intCast(add), @intCast(w) }, self.proj);
        self.kr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&self.kr, proj, &[_]c_int{ 0, 0, (PROJ_HEADS - 1) * D }, 3, &[_]c_int{ 1, @intCast(add), @intCast(w) }, 3, &[_]c_int{ 1, 1, 1 }, 3, rig.stream()));
        self.args[1] = self.kr;
        self.args[2] = try rig.input(.bfloat16, &.{}, &.{0});
        return true;
    }

    pub fn run(self: *RingCat) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(_: *const RingCat) usize {
        return @as(usize, @intCast(QSA_RING_ROWS)) * D;
    }

    /// The ring's last rows then the new keys, each + 0 (a -0 key reads +0, NaN stays NaN).
    pub fn reference(self: *const RingCat, want: []f64, mag: []f64) !void {
        const ring: usize = @intCast(QSA_RING_ROWS);
        const held = self.shape.k;
        const keep = ring - self.shape.rows;
        for (0..ring) |r| for (0..D) |d| {
            const v = if (r < keep) self.old[(held - keep + r) * D + d] else self.proj[(r - keep) * PROJ_HEADS * D + (PROJ_HEADS - 1) * D + d];
            want[r * D + d] = v + 0.0;
            mag[r * D + d] = @abs(v);
        };
    }

    /// The rebuilt ring's bytes are the chain's at every gate (NaN payloads included).
    pub fn check(self: *RingCat, _: spec.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const ours = try a.dupe(u8, self.rig.outputBytes());
        if (!std.mem.eql(u8, ours, try self.replaced())) return "the rebuilt ring differs from the concat and owned copy's bytes";
        return null;
    }

    /// `qsaAppendKeys`' rebuild on the same keys: the ring's last rows, concatenated with the new keys, then
    /// `qsaInstallRing`'s owned copy (`materializedOwnedCopy`).
    pub fn replaced(self: *RingCat) ![]const u8 {
        const s = self.rig.stream();
        const held: c_int = @intCast(self.shape.k);
        const keep: c_int = QSA_RING_ROWS - @as(c_int, @intCast(self.shape.rows));
        var tail = mlx.mlx_array_new();
        defer free(tail);
        try mlx.check(mlx.mlx_slice(&tail, self.args[0], &[_]c_int{ 0, held - keep, 0 }, 3, &[_]c_int{ 1, held, D }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const parts = [_]mlx.mlx_array{ tail, self.kr };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, s));
        const out = try materializedOwnedCopy(s, cat);
        defer free(out);
        return self.rig.bytesOf(&.{out});
    }

    fn wrapper(self: *RingCat) ![]const u8 {
        const out = try merges.ringCat(self.rig.stream(), self.args[0], self.kr, 1, D, .bfloat16);
        defer free(out);
        return self.rig.bytesOf(&.{out});
    }
};

// ── the PLE layer's fused tail ──

/// Hyper-connection streams the contract's PLE runs at (Flash Next's).
const HC = 4;

fn pleLaunches() [5]metal.Launch {
    var out: [5]metal.Launch = undefined;
    for (&out, 0..) |*l, i| {
        const kind: ple.PleDecKind = @enumFromInt(@as(u8, @intCast(i)));
        l.* = .{
            .name = std.mem.span(ple.ple_dec_names[i]),
            .inputs = ple.pleDecInputs(kind),
            .outputs = &.{ple.pleDecOutput(kind)},
            .source = ple.ple_dec_sources[i],
            .tg_bytes = 0,
            .m1 = 1024,
            .m2 = 1024,
        };
    }
    return out;
}
const ple_launches = pleLaunches();

/// One fused-tail launch at rows tokens of a stream n = HC * hidden wide (`k`: the conv state's rows, the concat's).
const PleDec = struct {
    rig: metal.Rig,
    kind: ple.PleDecKind,
    launch: *const metal.Launch,
    arena: std.heap.ArenaAllocator,
    shape: Shape = undefined,
    args: [5]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, kind: ple.PleDecKind) PleDec {
        return .{ .rig = metal.Rig.init(a), .kind = kind, .launch = &ple_launches[@intFromEnum(kind)], .arena = .init(a) };
    }

    pub fn deinit(self: *PleDec) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    fn hidden(shape: Shape) usize {
        return shape.n / HC;
    }

    fn outShape(self: *const PleDec, shape: Shape, buf: *[4]c_int) []const c_int {
        const s: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(hidden(shape));
        const c: c_int = @intCast(shape.n);
        return switch (self.kind) {
            .kq => blk: {
                buf.* = .{ 1, s, HC, h };
                break :blk buf[0..4];
            },
            .gv, .tail, .tail_res => blk: {
                buf.* = .{ 1, s, c, 0 };
                break :blk buf[0..3];
            },
            .cat => blk: {
                buf.* = .{ 1, @as(c_int, @intCast(shape.k)) + s, c, 0 };
                break :blk buf[0..3];
            },
        };
    }

    /// The wrapper's launch: one thread an output element, 256 a group.
    pub fn geometry(self: *const PleDec, shape: Shape) ?metal.Geometry {
        if (shape.rows < 1 or shape.n < HC or shape.n % HC != 0) return null;
        if (self.kind == .cat and shape.k < 1) return null;
        var buf: [4]c_int = undefined;
        const sh = self.outShape(shape, &buf);
        var n: c_int = 1;
        for (sh) |d| n *= d;
        var g = metal.Geometry{ .grid = .{ @divTrunc(n + 255, 256) * 256, 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(sh, .bfloat16);
        g.dtype("T", .bfloat16);
        const h: c_int = @intCast(hidden(shape));
        switch (self.kind) {
            .kq => g.int("HW", HC * h),
            .gv => {
                g.int("H", h);
                g.int("HC", HC);
            },
            .cat => {
                g.int("C", @as(c_int, @intCast(shape.n)));
                g.int("SL", @as(c_int, @intCast(shape.k)));
            },
            .tail, .tail_res => {},
        }
        return g;
    }

    fn fillBf16(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f64, hi: f64) ![]f64 {
        const xs = try a.alloc(f64, n);
        for (xs) |*v| v.* = spec.roundTo(.bf16, lo + rng.float(f64) * (hi - lo));
        return xs;
    }

    /// The gate or conv input holding NaN, both infinities and both zeros at known elements (`.nonfinite`).
    fn plant(xs: []f64) void {
        const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), -0.0, 0.0 };
        for (put, 0..) |v, i| if (i * 7 < xs.len) {
            xs[i * 7] = v;
        };
    }

    pub fn prepare(self: *PleDec, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const c = shape.n;
        const h = hidden(shape);
        const rig = &self.rig;
        rig.reset();
        const s: c_int = @intCast(rows);
        const hi: c_int = @intCast(h);
        const ci: c_int = @intCast(c);
        var buf: [4]c_int = undefined;
        const osh = self.outShape(shape, &buf);
        var total: c_int = 1;
        for (osh) |d| total *= d;
        const n_size = mlx.mlx_array_new_int(total);
        switch (self.kind) {
            .kq => {
                const nk = try fillBf16(a, rng, rows * c, -3, 3);
                const nq = try fillBf16(a, rng, rows * c, -3, 3);
                if (fill == .nonfinite) plant(nk);
                self.args[0] = try rig.input(.bfloat16, &.{ 1, s, HC, hi }, nk);
                self.args[1] = try rig.input(.bfloat16, &.{ HC, hi }, try fillBf16(a, rng, c, 0.25, 1.75));
                self.args[2] = try rig.input(.bfloat16, &.{ 1, s, HC, hi }, nq);
                self.args[3] = try rig.input(.bfloat16, &.{ HC, hi }, try fillBf16(a, rng, c, 0.25, 1.75));
                self.args[4] = n_size;
                self.n_args = 5;
            },
            .gv => {
                const gate = try fillBf16(a, rng, rows * HC, -40, 40);
                if (fill == .nonfinite) plant(gate);
                self.args[0] = try rig.input(.bfloat16, &.{ 1, s, HC, 1 }, gate);
                self.args[1] = try rig.input(.bfloat16, &.{ 1, s, hi }, try fillBf16(a, rng, rows * h, -2, 2));
                self.args[2] = try ple.pleGateTable(rig.stream(), hi);
                self.args[3] = n_size;
                self.n_args = 4;
            },
            .cat => {
                const sl = shape.k;
                self.args[0] = try rig.input(.bfloat16, &.{ 1, @intCast(sl), ci }, try fillBf16(a, rng, sl * c, -2, 2));
                const nc = try fillBf16(a, rng, rows * c, -3, 3);
                if (fill == .nonfinite) plant(nc);
                self.args[1] = try rig.input(.bfloat16, &.{ 1, s, ci }, nc);
                self.args[2] = try rig.input(.bfloat16, &.{ci}, try fillBf16(a, rng, c, 0.25, 1.75));
                self.args[3] = n_size;
                self.args[4] = mlx.mlx_array_new_int(s);
                self.n_args = 5;
            },
            .tail, .tail_res => {
                const conv = try fillBf16(a, rng, rows * c, -12, 12);
                if (fill == .nonfinite) plant(conv);
                self.args[0] = try rig.input(.bfloat16, &.{ 1, s, ci }, conv);
                self.args[1] = try rig.input(.bfloat16, &.{ 1, s, ci }, try fillBf16(a, rng, rows * c, -2, 2));
                self.args[2] = try sigmoidTableFor(rig.stream(), .bfloat16);
                self.args[3] = n_size;
                self.n_args = 4;
                if (self.kind == .tail_res) {
                    self.args[4] = try rig.input(.bfloat16, &.{ 1, s, ci }, try fillBf16(a, rng, rows * c, -8, 8));
                    self.n_args = 5;
                }
            },
        }
        // the 0-dim sizes ride the rig's inputs (freed with them); the tables are the wrappers' process-wide ones
        try rig.ins.append(rig.a, .{ .parent = n_size, .view = mlx.mlx_array_new(), .len = 1 });
        if (self.kind == .cat) try rig.ins.append(rig.a, .{ .parent = self.args[4], .view = mlx.mlx_array_new(), .len = 1 });
        return true;
    }

    pub fn run(self: *PleDec) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const PleDec) usize {
        var buf: [4]c_int = undefined;
        var n: usize = 1;
        for (self.outShape(self.shape, &buf)) |d| n *= @intCast(d);
        return n;
    }

    /// The chain's own bits stand as the reference (`check`, `replaced`).
    pub fn reference(_: *const PleDec, _: []f64, _: []f64) !void {
        return error.ChainIsTheReference;
    }

    pub fn check(self: *PleDec, _: spec.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const ours = try a.dupe(u8, self.rig.outputBytes());
        if (!std.mem.eql(u8, ours, try self.replaced())) return "an element differs from the op chain's bits";
        return null;
    }

    /// The chain's ops on the same inputs: the group norms' weight products and the keys x queries product (kq); the
    /// gate's seven ops and its value product (gv); the conv input's weight product and concat (cat); silu, the conv
    /// branch's add and the call site's residual add (tail).
    pub fn replaced(self: *PleDec) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        var t = [4]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (t) |a| free(a);
        const out: mlx.mlx_array = switch (self.kind) {
            .kq => blk: {
                try mlx.check(mlx.mlx_multiply(&t[0], x[0], x[1], s));
                try mlx.check(mlx.mlx_multiply(&t[1], x[2], x[3], s));
                try mlx.check(mlx.mlx_multiply(&t[2], t[0], t[1], s));
                break :blk t[2];
            },
            .gv => blk: {
                const h: c_int = @intCast(hidden(self.shape));
                const s_rows: c_int = @intCast(self.shape.rows);
                const g = try ple.pleGateChain(s, x[0], h);
                defer free(g);
                try mlx.check(mlx.mlx_reshape(&t[0], x[1], &[_]c_int{ 1, s_rows, 1, h }, 4, s));
                try mlx.check(mlx.mlx_multiply(&t[1], g, t[0], s));
                try mlx.check(mlx.mlx_reshape(&t[2], t[1], &[_]c_int{ 1, s_rows, HC * h }, 3, s));
                break :blk t[2];
            },
            .cat => blk: {
                try mlx.check(mlx.mlx_multiply(&t[0], x[1], x[2], s));
                const parts = [_]mlx.mlx_array{ x[0], t[0] };
                const vec = mlx.mlx_vector_array_new_data(&parts, 2);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_concatenate_axis(&t[1], vec, 1, s));
                break :blk t[1];
            },
            .tail, .tail_res => blk: {
                try mlx.check(mlx.mlx_sigmoid(&t[0], x[0], s));
                try mlx.check(mlx.mlx_multiply(&t[1], x[0], t[0], s));
                try mlx.check(mlx.mlx_add(&t[2], x[1], t[1], s));
                if (self.kind == .tail) break :blk t[2];
                try mlx.check(mlx.mlx_add(&t[3], x[4], t[2], s));
                break :blk t[3];
            },
        };
        return self.rig.bytesOf(&.{out});
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *PleDec) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const rows: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(hidden(self.shape));
        const out = switch (self.kind) {
            .kq => try ple.pleDecKq(s, x[0], x[1], x[2], x[3]),
            .gv => try ple.pleDecGv(s, x[0], x[1], x[2], 1, rows, HC, h),
            .cat => try ple.pleDecCat(s, x[0], x[1], x[2], 1, @intCast(self.shape.k), rows, @intCast(self.shape.n)),
            .tail => try ple.pleDecTail(s, x[0], x[1], x[2], null),
            .tail_res => try ple.pleDecTail(s, x[0], x[1], x[2], x[4]),
        };
        defer free(out);
        return self.rig.bytesOf(&.{out});
    }
};

fn pleSpec(comptime name: []const u8, replaces: []const u8, served: []const Shape) spec.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = .bf16,
        .replaces = replaces,
        .nan = .{ .stated = "elementwise: NaN, both infinities and both zeros at known elements give the op chain's bits" },
        .served = served,
    };
}

pub fn register(set: *kc.Set) !void {
    try set.add(PlaceVis, .init(set.a), .{
        .name = "qsa_place_vis (a placed draft row's key visibility)",
        .accum = .none,
        .depth = none,
        .out = .bool,
        .replaces = "the row mask's op chain (qsaMaskFromBlocks) -> headPlaceVisibility (arange, compares, or, and)",
        .nan = .{ .stated = "integer block ids and ends: the edge ids, a block-aligned live end and no rows past the dead end give the reference mask" },
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        // just past the 2048-key budget, a longer history, a tiny cache
        .served = &.{ .{ .rows = 512, .n = 2053, .k = 1 }, .{ .rows = 512, .n = 8079, .k = 1 }, .{ .rows = 16, .n = 65, .k = 1 } },
    }, null);
    try set.add(RingCat, .init(set.a), .{
        .name = "qsa_ring_cat bf16 (the key ring's rebuild)",
        .accum = .none,
        .depth = none,
        .out = .bf16,
        .replaces = "slice -> concatenate -> zeros + add (qsaAppendKeys' rebuild, qsaInstallRing's owned copy)",
        .nan = .propagate,
        // a decode row on a full ring, a verify step, the widest add a ring rebuilds
        .served = &.{ .{ .rows = 1, .n = D, .k = 32 }, .{ .rows = 4, .n = D, .k = 30 }, .{ .rows = 31, .n = D, .k = 2 } },
    }, null);
    // a decode row, a verify step, a short odd width (the grid's tail) over Flash Next's 4 x 2560 stream
    const served = &[_]Shape{ .{ .rows = 1, .n = HC * 2560, .k = 9 }, .{ .rows = 4, .n = HC * 2560, .k = 9 }, .{ .rows = 3, .n = HC * 37, .k = 9 } };
    try set.add(PleDec, .init(set.a, .kq), pleSpec("ple_dec_kq bf16 (the key and query norms' weight products, their product)", "multiply x3", served), null);
    try set.add(PleDec, .init(set.a, .gv), pleSpec("ple_dec_gv bf16 (the gate's seven ops by table, the value product)", "the gate's seven ops -> multiply -> reshape", served), null);
    try set.add(PleDec, .init(set.a, .cat), pleSpec("ple_dec_cat bf16 (the conv input's weight product and concat)", "multiply -> concatenate", served), null);
    try set.add(PleDec, .init(set.a, .tail), pleSpec("ple_dec_tail bf16 (silu by table, the conv branch's add)", "sigmoid -> multiply -> add", served), null);
    try set.add(PleDec, .init(set.a, .tail_res), pleSpec("ple_dec_tail_res bf16 (the tail and the residual add)", "sigmoid -> multiply -> add -> add", served), null);
}

// ── tests ──

test "kernel contract qwen4_exp merges: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    inline for (.{ PlaceVis, RingCat, PleDec, PleDec, PleDec, PleDec, PleDec }, 0..) |T, i| {
        const k = set.kernels.items[i];
        const p: *T = @ptrCast(@alignCast(k.ctx));
        for (k.spec.served) |shape| {
            try std.testing.expect(try p.prepare(shape, .random, 7));
            try p.run();
            const ours = try a.dupe(u8, p.rig.outputBytes());
            defer a.free(ours);
            try std.testing.expectEqualSlices(u8, try p.wrapper(), ours);
        }
    }
}
