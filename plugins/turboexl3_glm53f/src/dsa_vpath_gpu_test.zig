//! The device checks of TURBOEXL3_GLM53F_DSA_TAILS, _DSA_ROWS and _DSA_WFOLD (src/dsa_tails.zig, src/dsa_rows.zig,
//! src/dsa_wfold.zig), run by the build step `dsa-gpu-test` (linked to the server's MLX) under
//! TURBOEXL3_GLM53F_DSA_GPU=1; `zig build test` compiles them and its filter leaves them out. Every check is bit for bit
//! against the stock path at GLM-5.3-Flash's indexer dims (rows 256 wide: a 128-wide key and gate; pools of 4; 32
//! index heads; 2048 selected tokens).
//!
//! - `DSA_TAILS`: the tail rows `begin` reads (the trunk's 11 layers in one launch, the head's one in another) against
//!   the stock slices of the same cache views, deep in a 16K cache (3 rows) and at its first rows (1 and 2):
//!       TURBOEXL3_GLM53F_DSA_BYTES tails case=<name> layers=<n> rows=<r> differing=<n>/<total> served=<n> PASS|FAIL
//!   the pooled-keys kernel's rows from them against its rows from the stock slices, at 1 to 4 rows with the window's
//!   first pool begun 1, 2 or 3 rows before it (so a pool it completes reads them):
//!       TURBOEXL3_GLM53F_DSA_BYTES tails_pooled case=<name> rows=<S> differing=<n>/<total> PASS|FAIL
//!   and the appends themselves, the trunk's 11 and the head's 1, each as the KV cache makes it
//!   (`KVCache.updateDense`: the view let go, a slice_update into the buffer, the old handle let go) on a cache like the
//!   KV cache's (a buffer of its own, its rows a multiple of the cache's 256-row step), a forward's appends evaluated
//!   together: with no read before them (bare: every buffer reused, the check that the caches can be), after the stock
//!   reads (every one copied) and after one `begin` (every one reused, its copy outside every cache's buffer), the stock
//!   and `begin` appends leaving the same bytes; each layer's result listed:
//!       TURBOEXL3_GLM53F_DSA_TAILS_DONATE case=<trunk|head> layers=<n> bare_donated=<n>/<n> stock_copied=<n>/<n>
//!       tails_donated=<n>/<n> tails_apart=<n>/<n> same_bytes=<n>/<n> attempts=<n> by_layer=L3:donated,... PASS|FAIL
//! - `DSA_ROWS`: the rows the pooled-keys launch builds from the key and the f32 gate against glm5_next
//!   `withPooledKeys` over the stock rows (the gate cast, the two concatenated), at 1, 2, 3, 4 and 16 rows starting at
//!   each place in a pool deep in the cache, at the cache's first rows (0 to 3 cached), and with gates on and next to
//!   bf16 rounding ties:
//!       TURBOEXL3_GLM53F_DSA_BYTES rows case=<name> rows=<S> pos0=<p> prev=<n> differing=<n>/<total> ran=<n> PASS|FAIL
//! - `DSA_WFOLD`: the selection from the head-weight projection against glm5_next `selectTokens` over the stock chain's
//!   weights (the cast and both scale multiplies), with DSA_VSEL on (vs=vsel) and off (vs=perrow), at 1 to 4 and 16
//!   rows in a 16K context, its end, 2K and 32K contexts, zero weights and keys repeating every 7 pools:
//!       TURBOEXL3_GLM53F_DSA_BYTES wfold case=<name> rows=<S> pools=<P> vs=<vsel|perrow> differing=<n>/<total> ran=<n>
//!       PASS|FAIL
//! The run's last line: `TURBOEXL3_GLM53F_DSA_VPATH_PARITY lines=<n> fails=<n> PASS|FAIL`.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const dsa_tails = @import("dsa_tails.zig");
const dsa_rows = @import("dsa_rows.zig");
const dsa_wfold = @import("dsa_wfold.zig");
const dsa_verify = @import("dsa_verify.zig");
const dsa_gpu = @import("dsa_gpu_test.zig");
const Array = mlx.mlx_array;

const ix = dsa_gpu.ix;
const row_w: c_int = 2 * ix.head_dim;

fn free(a: Array) void {
    _ = mlx.mlx_array_free(a);
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Element-wise comparison of two same-shape arrays of 16- or 32-bit elements, as bits.
const Diff = struct { differing: usize, total: usize };
fn diffBits(s: mlx.mlx_stream, a: Array, b: Array) !Diff {
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return error.VpathShape;
    const dt = mlx.mlx_array_dtype(a);
    if (dt != mlx.mlx_array_dtype(b)) return error.VpathDtype;
    var ac = mlx.mlx_array_new();
    defer free(ac);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    var bc = mlx.mlx_array_new();
    defer free(bc);
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    try evalAll(&.{ ac, bc });
    const n = mlx.mlx_array_size(ac);
    var d: usize = 0;
    switch (dt) {
        .bfloat16 => {
            const pa = (mlx.mlx_array_data_bfloat16(ac) orelse return error.VpathData)[0..n];
            const pb = (mlx.mlx_array_data_bfloat16(bc) orelse return error.VpathData)[0..n];
            for (pa, pb) |x, y| d += @intFromBool(x != y);
        },
        .int32 => {
            const pa = (mlx.mlx_array_data_int32(ac) orelse return error.VpathData)[0..n];
            const pb = (mlx.mlx_array_data_int32(bc) orelse return error.VpathData)[0..n];
            for (pa, pb) |x, y| d += @intFromBool(x != y);
        },
        else => return error.VpathDtype,
    }
    return .{ .differing = d, .total = n };
}

fn sliceRows(s: mlx.mlx_stream, a: Array, axis: usize, lo: c_int, hi: c_int) !Array {
    const sh = mlx.getShape(a);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    for (sh, 0..) |v, i| stop[i] = v;
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &start, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
    return out;
}

fn reshape(s: mlx.mlx_stream, a: Array, shape: []const c_int) !Array {
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

/// The stock tail `dsaAttn` slices from a cache view [1, 1, T, 256]: its last n rows as [1, n, 256].
fn stockTail(s: mlx.mlx_stream, view: Array, n: c_int) !Array {
    const t = mlx.getShape(view)[2];
    const tail = try sliceRows(s, view, 2, t - n, t);
    defer free(tail);
    return reshape(s, tail, &.{ 1, n, row_w });
}

/// The mock slot's cache handle (`dsa_tails` compares it, never reads through it).
var mock_cache: u8 = 0;
/// glm5_next `Slots` as `dsa_tails` reads it: one slot's cache handle and position, a view per layer.
const MockView = struct {
    v: Array,
    pub fn deinit(self: *MockView) void {
        self.* = .{ .v = .{ .ctx = null } };
    }
};
const MockSlots = struct {
    caches: []const ?*anyopaque,
    positions: []const u32,
    layers: []const u32,
    views: []const Array,
    pub fn view(self: MockSlots, i: usize, layer: u32, s: mlx.mlx_stream) !MockView {
        _ = s;
        if (i != 0) return error.VpathSlot;
        for (self.layers, self.views) |l, v| if (l == layer) return .{ .v = v };
        return error.VpathLayer;
    }
};

/// A cache buffer [1, 1, cap, 256] and its view of the first `t` rows, both evaluated (the cache after a forward).
const Cache = struct {
    buf: Array,
    view: Array,
    fn init(s: mlx.mlx_stream, seed: *u64, t: c_int, cap: c_int) !Cache {
        const buf = try dsa_gpu.normal(s, seed, &.{ 1, 1, cap, row_w }, .bfloat16, 1.0);
        errdefer free(buf);
        const view = try sliceRows(s, buf, 2, 0, t);
        errdefer free(view);
        try mlx.check(mlx.mlx_array_eval(view));
        return .{ .buf = buf, .view = view };
    }
    fn deinit(c: *Cache) void {
        free(c.view);
        free(c.buf);
    }
};

const trunk_layer_ids = [_]u32{ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 };
const TailCase = struct { name: []const u8, layers: []const u32, t: c_int, rows: c_int };
const tail_cases = [_]TailCase{
    .{ .name = "trunk16k", .layers = &trunk_layer_ids, .t = 16390, .rows = 3 },
    .{ .name = "trunk_at1", .layers = &trunk_layer_ids, .t = 1, .rows = 3 },
    .{ .name = "trunk_at2", .layers = &trunk_layer_ids, .t = 2, .rows = 1 },
    .{ .name = "head16k", .layers = &.{0}, .t = 16390, .rows = 1 },
    .{ .name = "head_at1", .layers = &.{0}, .t = 1, .rows = 2 },
};

/// One `begin` over fresh caches for `c`: each layer's rows against the stock tail.
fn tailBytes(s: mlx.mlx_stream, seed: *u64, c: TailCase, fails: *usize, lines: *usize) !void {
    var caches: [11]Cache = undefined;
    var made: usize = 0;
    defer for (caches[0..made]) |*x| x.deinit();
    var views: [11]Array = undefined;
    for (c.layers, 0..) |_, i| {
        caches[i] = try Cache.init(s, seed, c.t, c.t + 61);
        views[i] = caches[i].view;
        made += 1;
    }
    const handles = [_]?*anyopaque{@ptrCast(&mock_cache)};
    const pos = [_]u32{@intCast(c.t)};
    const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = c.layers, .views = views[0..made] };
    const n_prev: c_int = @min(c.t, 3);
    const before = dsa_tails.served;
    try dsa_tails.begin(s, sl, false, &.{ 1, c.rows }, c.layers);
    defer dsa_tails.end();
    var diff: usize = 0;
    var total: usize = 0;
    var missing: usize = 0;
    for (c.layers, 0..) |layer, i| {
        const want = try stockTail(s, views[i], n_prev);
        defer free(want);
        const got = dsa_tails.take(sl, 0, layer, n_prev) orelse {
            missing += 1;
            continue;
        };
        const d = try diffBits(s, want, got);
        diff += d.differing;
        total += d.total;
    }
    const served = dsa_tails.served - before;
    const ok = missing == 0 and diff == 0 and total == c.layers.len * @as(usize, @intCast(n_prev * row_w)) and served == c.layers.len;
    if (!ok) fails.* += 1;
    lines.* += 1;
    std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES tails case={s} layers={d} rows={d} differing={d}/{d} served={d} {s}\n", .{ c.name, c.layers.len, n_prev, diff, total, served, if (ok) "PASS" else "FAIL" });
}

/// The pooled-keys kernel's rows from `begin`'s tail against its rows from the stock tail: a window of `rows` rows at
/// `t` over one head cache.
fn tailPooled(s: mlx.mlx_stream, seed: *u64, t: c_int, rows: c_int, fails: *usize, lines: *usize) !void {
    var c = try Cache.init(s, seed, t, t + 61);
    defer c.deinit();
    const chunk = try dsa_gpu.normal(s, seed, &.{ 1, rows, row_w }, .bfloat16, 1.0);
    defer free(chunk);
    const ape = try dsa_gpu.normal(s, seed, &.{ ix.kpool, ix.head_dim }, .float32, 1.0);
    defer free(ape);
    const handles = [_]?*anyopaque{@ptrCast(&mock_cache)};
    const pos = [_]u32{@intCast(t)};
    const layers = [_]u32{0};
    const views = [_]Array{c.view};
    const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = &layers, .views = &views };
    const want_prev = try stockTail(s, c.view, 3);
    defer free(want_prev);
    const want = try glm5.withPooledKeys(s, want_prev, chunk, ape, t, ix);
    defer free(want);
    try dsa_tails.begin(s, sl, false, &.{ 1, rows }, &layers);
    defer dsa_tails.end();
    const got_prev = dsa_tails.take(sl, 0, 0, 3) orelse return error.VpathTailsDeclined;
    const got = try glm5.withPooledKeys(s, got_prev, chunk, ape, t, ix);
    defer free(got);
    const d = try diffBits(s, want, got);
    const ok = d.differing == 0 and d.total > 0;
    if (!ok) fails.* += 1;
    lines.* += 1;
    std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES tails_pooled case=at{d}_mod{d} rows={d} differing={d}/{d} {s}\n", .{ t, @rem(t, ix.kpool), rows, d.differing, d.total, if (ok) "PASS" else "FAIL" });
}

/// A forward's DSA appends as the KV cache makes them (`KVCache.updateDense`: the view let go, the rows written into the
/// buffer, the old handle let go), one fresh cache a layer (a buffer of its own, `seed + i`'s rows, `cap` rows: a
/// multiple of the cache's 256-row step), the forward's appends evaluated together. `mode`: no read before them (bare,
/// its rows from `other`), the stock read of each view (stock), or one `begin` (tails). Each buffer reused (donated) or
/// not, each copy outside its cache's buffer (tails), and each cache after it.
const max_layers = trunk_layer_ids.len;
const Mode = enum { bare, stock, tails };
const Appends = struct {
    n: usize = 0,
    donated: [max_layers]bool = @splat(false),
    apart: [max_layers]bool = @splat(false),
    after: [max_layers]Array = undefined,
    fn deinit(x: *Appends) void {
        for (x.after[0..x.n]) |a| free(a);
    }
};
fn appendAll(s: mlx.mlx_stream, layers: []const u32, seed: u64, t: c_int, cap: c_int, rows: c_int, chunk: Array, ape: Array, other: Array, mode: Mode, settle_ms: u64) !Appends {
    const nl = layers.len;
    var bufs: [max_layers]Array = undefined;
    var views: [max_layers]Array = undefined;
    var old: [max_layers]usize = undefined;
    var live: [max_layers]u2 = @splat(0); // 2: buffer and view held, 1: the buffer, 0: neither
    var made: usize = 0;
    defer for (bufs[0..made], views[0..made], live[0..made]) |b, v, l| {
        if (l >= 2) free(v);
        if (l >= 1) free(b);
    };
    const nbytes: usize = @as(usize, @intCast(cap)) * @as(usize, @intCast(row_w)) * 2;
    for (0..nl) |i| {
        var sd = seed + @as(u64, @intCast(i));
        bufs[i] = try dsa_gpu.normal(s, &sd, &.{ 1, 1, cap, row_w }, .bfloat16, 1.0);
        live[i] = 1;
        made += 1;
        views[i] = try sliceRows(s, bufs[i], 2, 0, t);
        live[i] = 2;
        try mlx.check(mlx.mlx_array_eval(views[i]));
        old[i] = @intFromPtr(mlx.mlx_array_data_bfloat16(bufs[i]) orelse return error.VpathData);
    }
    // the caches as a forward finds them: their last command buffers done
    const ts = std.c.timespec{ .sec = @intCast(settle_ms / std.time.ms_per_s), .nsec = @intCast((settle_ms % std.time.ms_per_s) * std.time.ns_per_ms) };
    try mlx.check(mlx.mlx_synchronize(s));
    _ = std.c.nanosleep(&ts, null);
    const handles = [_]?*anyopaque{@ptrCast(&mock_cache)};
    const pos = [_]u32{@intCast(t)};
    const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = layers, .views = views[0..made] };
    var out: Appends = .{};
    if (mode == .tails) {
        try dsa_tails.begin(s, sl, false, &.{ 1, rows }, layers);
        // the forward's graph is built while the rows' command buffer runs: here, its completion awaited
        try mlx.check(mlx.mlx_synchronize(s));
        _ = std.c.nanosleep(&ts, null);
        const ga = dsa_tails.gatheredAddr() orelse return error.VpathTailsDeclined;
        for (old[0..nl], 0..) |o, i| out.apart[i] = ga < o or ga >= o + nbytes;
    }
    defer dsa_tails.end();
    var upds: [max_layers]Array = undefined;
    var nu: usize = 0;
    errdefer for (upds[0..nu]) |u| free(u);
    for (layers, 0..) |layer, i| {
        var prev = mlx.mlx_array_new();
        defer free(prev);
        switch (mode) {
            .bare => _ = mlx.mlx_array_set(&prev, other),
            .stock => {
                const p = try stockTail(s, views[i], 3);
                defer free(p);
                _ = mlx.mlx_array_set(&prev, p);
            },
            .tails => _ = mlx.mlx_array_set(&prev, dsa_tails.take(sl, 0, layer, 3) orelse return error.VpathTailsDeclined),
        }
        const pooled = try glm5.withPooledKeys(s, prev, chunk, ape, t, ix);
        defer free(pooled);
        const r4 = try reshape(s, pooled, &.{ 1, 1, rows, row_w });
        defer free(r4);
        free(views[i]);
        live[i] = 1;
        var upd = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice_update(&upd, bufs[i], r4, &[_]c_int{ 0, 0, t, 0 }, 4, &[_]c_int{ 1, 1, t + rows, row_w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s)) catch |e| {
            free(upd);
            return e;
        };
        upds[i] = upd;
        nu += 1;
        free(bufs[i]);
        live[i] = 0;
    }
    try evalAll(upds[0..nu]);
    for (upds[0..nu], old[0..nu], 0..) |u, o, i| {
        out.donated[i] = @intFromPtr(mlx.mlx_array_data_bfloat16(u) orelse return error.VpathData) == o;
        out.after[i] = u;
        out.n += 1;
    }
    return out;
}

const DonateCase = struct { name: []const u8, layers: []const u32 };
const donate_cases = [_]DonateCase{
    .{ .name = "trunk", .layers = &trunk_layer_ids },
    .{ .name = "head", .layers = &.{0} },
};

fn donateCheck(s: mlx.mlx_stream, seed: *u64, c: DonateCase, fails: *usize, lines: *usize) !void {
    // a cache 16,390 rows in, its capacity 65 steps of 256 rows: a page-aligned buffer, as the KV cache's are
    const t: c_int = 16390;
    const cap: c_int = 16640;
    const rows: c_int = 3;
    const nl = c.layers.len;
    const base_seed = seed.*;
    seed.* += 64;
    const chunk = try dsa_gpu.normal(s, seed, &.{ 1, rows, row_w }, .bfloat16, 1.0);
    defer free(chunk);
    const ape = try dsa_gpu.normal(s, seed, &.{ ix.kpool, ix.head_dim }, .float32, 1.0);
    defer free(ape);
    const other = try dsa_gpu.normal(s, seed, &.{ 1, 3, row_w }, .bfloat16, 1.0);
    defer free(other);
    // every append reused, every one copied, every one reused: up to three tries each, each waiting longer for the
    // command buffers' completion handlers to let go of what they read
    const settles = [_]u64{ 1, 20, 200 };
    var attempts: usize = 0;
    var bare_n: usize = 0;
    for (settles) |ms| {
        attempts += 1;
        var b = try appendAll(s, c.layers, base_seed, t, cap, rows, chunk, ape, other, .bare, ms);
        defer b.deinit();
        bare_n = 0;
        for (b.donated[0..nl]) |dn| bare_n += @intFromBool(dn);
        if (bare_n == nl) break;
    }
    var stock = try appendAll(s, c.layers, base_seed, t, cap, rows, chunk, ape, other, .stock, settles[0]);
    defer stock.deinit();
    var stock_copied: usize = 0;
    for (stock.donated[0..nl]) |dn| stock_copied += @intFromBool(!dn);
    var kept: [max_layers]bool = @splat(false);
    var tails_n: usize = 0;
    var apart_n: usize = 0;
    var same: usize = 0;
    for (settles) |ms| {
        attempts += 1;
        var got = try appendAll(s, c.layers, base_seed, t, cap, rows, chunk, ape, other, .tails, ms);
        defer got.deinit();
        tails_n = 0;
        apart_n = 0;
        same = 0;
        for (0..nl) |i| {
            const d = try diffBits(s, stock.after[i], got.after[i]);
            same += @intFromBool(d.differing == 0 and d.total > 0);
            tails_n += @intFromBool(got.donated[i]);
            apart_n += @intFromBool(got.apart[i]);
            kept[i] = got.donated[i];
        }
        if (tails_n == nl) break;
    }
    var by: [max_layers * 16]u8 = undefined;
    var n: usize = 0;
    for (c.layers, kept[0..nl]) |layer, k| n += (std.fmt.bufPrint(by[n..], "{s}L{d}:{s}", .{ if (n == 0) "" else ",", layer, if (k) "donated" else "copied" }) catch break).len;
    const ok = bare_n == nl and stock_copied == nl and tails_n == nl and apart_n == nl and same == nl;
    if (!ok) fails.* += 1;
    lines.* += 1;
    std.debug.print("TURBOEXL3_GLM53F_DSA_TAILS_DONATE case={s} layers={d} bare_donated={d}/{d} stock_copied={d}/{d} tails_donated={d}/{d} tails_apart={d}/{d} same_bytes={d}/{d} attempts={d} by_layer={s} {s}\n", .{ c.name, nl, bare_n, nl, stock_copied, nl, tails_n, nl, apart_n, nl, same, nl, attempts, by[0..n], if (ok) "PASS" else "FAIL" });
}

/// f32 gates [1, S, 128] on (`tie` 0x8000) or next to (0x7fff, 0x8001) a bf16 rounding tie, every value's upper half
/// random.
fn tieGates(a: std.mem.Allocator, rand: std.Random, rows: c_int) !Array {
    const n: usize = @intCast(rows * ix.head_dim);
    const v = try a.alloc(u32, n);
    defer a.free(v);
    const lows = [_]u32{ 0x8000, 0x7fff, 0x8001, 0x0000 };
    for (v, 0..) |*x, i| {
        // a finite upper half: sign, exponent 0x70..0x8f, a random mantissa
        const hi: u32 = (@as(u32, rand.int(u1)) << 15) | (@as(u32, 0x70 + rand.uintLessThan(u32, 0x20)) << 7) | rand.uintLessThan(u32, 0x80);
        x.* = (hi << 16) | lows[i % lows.len];
    }
    const out = mlx.mlx_array_new_data(v.ptr, &[_]c_int{ 1, rows, ix.head_dim }, 3, .float32);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The pooled-keys rows built from the key and gate against `withPooledKeys` over the stock rows.
fn rowsCase(a: std.mem.Allocator, rand: std.Random, s: mlx.mlx_stream, seed: *u64, name: []const u8, rows: c_int, pos0: c_int, ties: bool, fails: *usize, lines: *usize) !void {
    const ikn = try dsa_gpu.normal(s, seed, &.{ 1, rows, ix.head_dim }, .bfloat16, 1.0);
    defer free(ikn);
    const gate = if (ties) try tieGates(a, rand, rows) else try dsa_gpu.normal(s, seed, &.{ 1, rows, ix.head_dim }, .float32, 3.0);
    defer free(gate);
    const ape = try dsa_gpu.normal(s, seed, &.{ ix.kpool, ix.head_dim }, .float32, 1.0);
    defer free(ape);
    const n_prev: c_int = @min(pos0, ix.kpool - 1);
    const prev = try dsa_gpu.normal(s, seed, &.{ 1, @max(n_prev, 1), row_w }, .bfloat16, 1.0);
    defer free(prev);
    // the stock rows: the gate cast to the key's dtype, the two concatenated
    var gate_c = mlx.mlx_array_new();
    defer free(gate_c);
    try mlx.check(mlx.mlx_astype(&gate_c, gate, .bfloat16, s));
    var rows3 = mlx.mlx_array_new();
    defer free(rows3);
    {
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{ ikn, gate_c }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&rows3, vec, -1, s));
    }
    const want = try glm5.withPooledKeys(s, if (n_prev > 0) prev else rows3, rows3, ape, pos0, ix);
    defer free(want);
    dsa_rows.on = true;
    defer dsa_rows.on = false;
    const before = dsa_rows.runs;
    const got = (try dsa_rows.pooled(s, if (n_prev > 0) prev else ikn, n_prev, ikn, gate, ape, pos0)) orelse return error.VpathRowsDeclined;
    defer free(got);
    const d = try diffBits(s, want, got);
    const ran = dsa_rows.runs - before;
    const ok = d.differing == 0 and d.total > 0 and ran == 1;
    if (!ok) fails.* += 1;
    lines.* += 1;
    std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES rows case={s} rows={d} pos0={d} prev={d} differing={d}/{d} ran={d} {s}\n", .{ name, rows, pos0, n_prev, d.differing, d.total, ran, if (ok) "PASS" else "FAIL" });
}

const wfold_cases = [_]dsa_gpu.SelCase{
    .{ .name = "verify16k", .t = 16390, .rows = 3, .fill = .random },
    .{ .name = "verify16k_rows2", .t = 16389, .rows = 2, .fill = .random },
    .{ .name = "verify16k_rows4", .t = 16391, .rows = 4, .fill = .random },
    .{ .name = "verify16k_rows1", .t = 16388, .rows = 1, .fill = .random },
    .{ .name = "verify16k_rows16", .t = 16403, .rows = 16, .fill = .random },
    .{ .name = "verify16k_end", .t = 17411, .rows = 3, .fill = .random },
    .{ .name = "verify2k", .t = 2103, .rows = 3, .fill = .random },
    .{ .name = "verify32k", .t = 32771, .rows = 4, .fill = .random },
    .{ .name = "verify16k_zero_weights", .t = 16390, .rows = 3, .fill = .zero_weights },
    .{ .name = "verify16k_repeating_keys", .t = 16390, .rows = 3, .fill = .repeating_keys },
};

/// The fold's selection from the projection `wp` against the stock chain's (the cast, the heads' scale, then
/// `selectTokens`, which multiplies by the head dim's scale) with DSA_VSEL as `vsel`.
fn wfoldCase(s: mlx.mlx_stream, seed: *u64, c: dsa_gpu.SelCase, fails: *usize, lines: *usize) !void {
    var x = try dsa_gpu.selInputs(s, seed, c);
    defer x.deinit();
    // the projection: bf16 [1, S, 32] (zeros for the zero-weight case)
    const wp = if (c.fill == .zero_weights) blk: {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ 1, c.rows, ix.heads }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(z));
        break :blk z;
    } else try dsa_gpu.normal(s, seed, &.{ 1, c.rows, ix.heads }, .bfloat16, 1.0);
    defer free(wp);
    // the stock chain's weights (table_forward `dsaAttn`): f32, times 1/sqrt(index heads)
    var wpf = mlx.mlx_array_new();
    defer free(wpf);
    try mlx.check(mlx.mlx_astype(&wpf, wp, .float32, s));
    const hscale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.heads))));
    defer free(hscale);
    var wts = mlx.mlx_array_new();
    defer free(wts);
    try mlx.check(mlx.mlx_multiply(&wts, wpf, hscale, s));
    dsa_wfold.on = true;
    const before = dsa_wfold.runs;
    const got = (try dsa_wfold.select(s, x.q, wp, x.cache)) orelse return error.VpathWfoldDeclined;
    dsa_wfold.on = false;
    defer free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const ran = dsa_wfold.runs - before;
    for ([_]bool{ true, false }) |vsel| {
        dsa_verify.vsel_on = vsel;
        const want = try glm5.selectTokens(s, x.q, wts, x.cache, ix);
        dsa_verify.vsel_on = false;
        defer free(want);
        const d = try diffBits(s, want, got);
        const ok = d.differing == 0 and d.total > 0 and ran == 1;
        if (!ok) fails.* += 1;
        lines.* += 1;
        std.debug.print("TURBOEXL3_GLM53F_DSA_BYTES wfold case={s} rows={d} pools={d} vs={s} differing={d}/{d} ran={d} {s}\n", .{ c.name, c.rows, @divTrunc(c.t, ix.kpool), if (vsel) "vsel" else "perrow", d.differing, d.total, ran, if (ok) "PASS" else "FAIL" });
    }
}

test "turboexl3_glm53f dsa gpu: the verify path's DSA_TAILS rows and donated append, DSA_ROWS' pooled rows and DSA_WFOLD's selection are the stock path's bits" {
    if (std.c.getenv("TURBOEXL3_GLM53F_DSA_GPU") == null) return error.SkipZigTest;
    defer {
        dsa_tails.on = false;
        dsa_tails.end();
        dsa_rows.on = false;
        dsa_wfold.on = false;
        dsa_verify.vsel_on = false;
    }
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var seed: u64 = 20261109;
    var fails: usize = 0;
    var lines: usize = 0;
    // DSA_TAILS
    dsa_tails.on = true;
    for (tail_cases) |c| try tailBytes(s, &seed, c, &fails, &lines);
    for ([_]c_int{ 16389, 16390, 16391 }) |t| {
        for ([_]c_int{ 1, 2, 3, 4 }) |rows| try tailPooled(s, &seed, t, rows, &fails, &lines);
    }
    for (donate_cases) |c| try donateCheck(s, &seed, c, &fails, &lines);
    dsa_tails.on = false;
    // DSA_ROWS
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for ([_]c_int{ 1, 2, 3, 4, 16 }) |rows| {
        for ([_]c_int{ 16384, 16385, 16386, 16387 }) |pos0| try rowsCase(a, rand, s, &seed, "deep", rows, pos0, false, &fails, &lines);
    }
    for ([_]c_int{ 0, 1, 2, 3 }) |pos0| try rowsCase(a, rand, s, &seed, "first_rows", 4, pos0, false, &fails, &lines);
    for ([_]c_int{ 3, 16 }) |rows| try rowsCase(a, rand, s, &seed, "gate_ties", rows, 16385, true, &fails, &lines);
    // DSA_WFOLD
    for (wfold_cases) |c| try wfoldCase(s, &seed, c, &fails, &lines);
    std.debug.print("TURBOEXL3_GLM53F_DSA_VPATH_PARITY lines={d} fails={d} {s}\n", .{ lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
