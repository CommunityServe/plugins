//! The direct model's hyper-connections on the K packs: the read before each sublayer and the final mixer, its write
//! (deferred into the next read's first launch) and the flush. Prefill rows take normInv, rw_down and rw_upmix where MLX
//! runs one chain over K, else the norm kernel (a mixer: MLX's), fuse_down and read_upmix; one row takes the two-launch
//! read, 2..16 rows decode3, wider rows MLX's chain. Every K2–K5 hc block is dense bf16 at hc 4 x 2560, rank 320.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const lane_qmm = @import("../mlx_serve/sdk/blocks/qmm/lane_qmm.zig");
const sigmoidTableFor = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const applyClosure = @import("../mlx_serve/engines/mlx/compiled.zig").applyClosure;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const linear = @import("linear.zig");
const HcWeights = @import("../mlx_serve/plugins/qwen4_exp/weights.zig").HcWeights;
const hc_fused = @import("../mlx_serve/plugins/qwen4_exp/hc_fused.zig");
const hpn = @import("../mlx_serve/sdk/blocks/hc/hc_prefill_nax.zig");
const hc_decode2 = @import("../mlx_serve/sdk/blocks/hc/hc_decode2.zig");
const max_seq = @import("../mlx_serve/sdk/blocks/hc/hc_prefill.zig").max_seq;
pub const HcRead = @import("../mlx_serve/plugins/qwen4_exp/hc.zig").HcRead;
pub const HcPending = hc_fused.HcPending;
const HcFusedOut = hc_fused.HcFusedOut;
const HC_FUSED_MAX_ROWS = hc_fused.HC_FUSED_MAX_ROWS;

/// The sublayer's input read off the streams `h`, a pending write applied first: by the read's first launch, which
/// hands back the written stream as `h`, else by MLX's write.
pub fn read(self: *Trunk, h: *mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pending: *?HcPending) !HcRead {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        if (try fusedFor(self, h.*, w, batch, seq_len, pd.*)) |o| {
            _ = mlx.mlx_array_free(h.*);
            h.* = o.stream;
            return .{ .mixed = o.mixed, .inj = o.inj };
        }
        h.* = try write(self, h.*, pd.out, pd.inj, batch, seq_len);
    }
    return readPlain(self, h.*, w, batch, seq_len);
}

/// The sublayer's input read off `stream`, no write pending: a sigmoid-mixed mean of the normalized streams, and the
/// write gates `2 sigma(inject)` (Qwen4ExpTextGatedResidual.forward).
pub fn readPlain(self: *Trunk, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int) !HcRead {
    if (try fusedFor(self, stream, w, batch, seq_len, null)) |o| return .{ .mixed = o.mixed, .inj = o.inj };
    const n4 = try groupNorm(self, stream, w.norm_w, batch, seq_len);
    defer _ = mlx.mlx_array_free(n4);
    return readNormed(self, n4, w, batch, seq_len);
}

/// Defer `stream += out * inj` to the next read where its first launch takes it (prefill rows when that read's
/// `inject_flat` is set, and up to 16 rows); otherwise write now.
pub fn writeOrDefer(self: *Trunk, h: *mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int, inject_flat: mlx.mlx_array, pending: *?HcPending) !void {
    std.debug.assert(pending.* == null);
    const prefill = inject_flat.ctx != null and geometry(batch, seq_len) and mlx.mlx_array_dtype(inj) == .bfloat16;
    if ((prefill or batch * seq_len <= HC_FUSED_MAX_ROWS) and mlx.mlx_array_dtype(h.*) == mlx.mlx_array_dtype(out)) {
        var pd: HcPending = .{ .out = mlx.mlx_array_new(), .inj = mlx.mlx_array_new() };
        errdefer pd.deinit();
        try mlx.check(mlx.mlx_array_set(&pd.out, out));
        try mlx.check(mlx.mlx_array_set(&pd.inj, inj));
        pending.* = pd;
        return;
    }
    h.* = try write(self, h.*, out, inj, batch, seq_len);
}

/// A deferred write applied now.
pub fn flush(self: *Trunk, h: *mlx.mlx_array, batch: c_int, seq_len: c_int, pending: *?HcPending) !void {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        h.* = try write(self, h.*, pd.out, pd.inj, batch, seq_len);
    }
}

/// Each stream of `x` `[B, S, hc*hidden]` RMS-normed and scaled by `w` `[hc, hidden]` (carrying the reference's +1):
/// `[B, S, hc, hidden]`. Caller frees.
pub fn groupNorm(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const shape4 = [_]c_int{ batch, seq_len, @intCast(self.config.hc_count), @intCast(self.config.hidden_size) };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, self.s));
    var n4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4, x4, self.ones_hidden, self.config.rms_norm_eps, self.s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, n4, w, self.s));
    return out;
}

/// The K packs' hc blocks, whose geometry fixes the launches' plans below; anything else is a named error.
fn kPack(self: *const Trunk, stream: mlx.mlx_array, w: *const HcWeights) !void {
    if (!self.config.trunk_format) return error.DirectModelHcNotExl3Trunk;
    if (mlx.mlx_array_dtype(stream) != .bfloat16 or mlx.mlx_array_dtype(w.norm_w) != .bfloat16 or w.down_rows.ctx == null or w.up_rows.ctx == null or (w.inject_w.ctx != null and w.inject_flat.ctx == null)) return error.DirectModelHcNotDenseBf16;
    if (self.config.hc_count != 4 or self.config.hidden_size != 2560 or mlx.getShape(w.down_rows)[0] != 320) return error.DirectModelHcGeometry;
}

/// The read on the fused launches, `pend` applied first (the written stream is `.stream`, else null-ctx): prefill rows
/// with an inject, or up to 16 rows; null for MLX's chain.
fn fusedFor(self: *Trunk, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pend: ?HcPending) !?HcFusedOut {
    try kPack(self, stream, w);
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    if (w.inject_flat.ctx != null and geometry(batch, seq_len)) {
        // Where MLX runs one chain over K for the down and inject products, the normalized rows are never written.
        const m = batch * seq_len;
        if (mlxSplitK(m, mlx.getShape(w.down_rows)[0], hc * hidden) == null and mlxSplitK(m, hc, hc * hidden) == null) return try foldNorm(self, stream, w, batch, seq_len, pend);
        const n = try norm(self.s, stream, w.norm_w, self.rms_eps_arr, batch, seq_len, hc, hidden, pend);
        defer _ = mlx.mlx_array_free(n.normalized);
        errdefer if (n.stream.ctx != null) {
            _ = mlx.mlx_array_free(n.stream);
        };
        const r = try readNormed(self, n.normalized, w, batch, seq_len);
        return .{ .stream = n.stream, .mixed = r.mixed, .inj = r.inj };
    }
    if (batch * seq_len > HC_FUSED_MAX_ROWS) return null;
    return try decodeRead(self, stream, w, batch, seq_len, pend);
}

/// The read with the normalized rows never written: normInv's row scales, the down and the up + mix normalizing.
fn foldNorm(self: *Trunk, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pend: ?HcPending) !HcFusedOut {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const tabs = try fuseTables(self);
    const pre = try normInv(self.s, stream, self.rms_eps_arr, batch, seq_len, hc, hidden, pend);
    defer _ = mlx.mlx_array_free(pre.invs);
    errdefer if (pre.stream.ctx != null) {
        _ = mlx.mlx_array_free(pre.stream);
    };
    const xs = if (pend != null) pre.stream else stream;
    const f = try rwDown(self.s, xs, pre.invs, w, tabs, batch, seq_len, hc, hidden);
    defer _ = mlx.mlx_array_free(f.act);
    errdefer _ = mlx.mlx_array_free(f.inj);
    const mixed = try upMix(self.s, f.act, w.up_rows, .{ .ctx = null }, .{ xs, pre.invs, w.norm_w }, batch, seq_len, hc, hidden);
    return .{ .stream = pre.stream, .mixed = mixed, .inj = f.inj };
}

/// The read from the normalized streams `n4` [B, S, hc, hidden]: fuse_down and read_upmix on prefill rows, else MLX's.
fn readNormed(self: *Trunk, n4: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int) !HcRead {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    var n_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n_flat);
    try mlx.check(mlx.mlx_reshape(&n_flat, n4, &[_]c_int{ batch, seq_len, hc * hidden }, 3, self.s));
    // down/inject carry the reference's `/hc` in their weights (load-time fold).
    if (geometry(batch, seq_len)) {
        const f = try fuseDown(self.s, n_flat, w.down_rows, w.inject_flat, try fuseTables(self), batch, seq_len, hc);
        defer _ = mlx.mlx_array_free(f.act);
        errdefer if (f.inj.ctx != null) {
            _ = mlx.mlx_array_free(f.inj);
        };
        return .{ .mixed = try upMix(self.s, f.act, w.up_rows, n4, null, batch, seq_len, hc, hidden), .inj = f.inj };
    }
    const act = blk: {
        const down = try linear.qmatmul(self, n_flat, w.down_w, w.down_s, w.down_b);
        defer _ = mlx.mlx_array_free(down);
        break :blk try tail(self.compiled_hc_silu, &.{down});
    };
    defer _ = mlx.mlx_array_free(act);
    const up = try linear.qmatmul(self, act, w.up_w, w.up_s, w.up_b);
    defer _ = mlx.mlx_array_free(up);
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_reshape(&up4, up, &[_]c_int{ batch, seq_len, hc, hidden }, 4, self.s));
    const mixed = try tail(self.compiled_hc_mix, &.{ up4, n4 });
    errdefer _ = mlx.mlx_array_free(mixed);
    var inj = mlx.mlx_array_new();
    if (w.inject_w.ctx != null) {
        const raw = try linear.qmatmul(self, n_flat, w.inject_w, w.inject_s, w.inject_b);
        defer _ = mlx.mlx_array_free(raw);
        const g = try tail(self.compiled_hc_inj, &.{raw});
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_reshape(&inj, g, &[_]c_int{ batch, seq_len, hc, 1 }, 4, self.s));
    }
    return .{ .mixed = mixed, .inj = inj };
}

/// `stream + flatten(out[..., None, :] * inj)`. Consumes `stream`.
fn write(self: *Trunk, stream: mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    var out4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out4);
    try mlx.check(mlx.mlx_reshape(&out4, out, &[_]c_int{ batch, seq_len, 1, hidden }, 4, self.s));
    var stream4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream4);
    try mlx.check(mlx.mlx_reshape(&stream4, stream, &[_]c_int{ batch, seq_len, hc, hidden }, 4, self.s));
    const next4 = try tail(self.compiled_hc_write, &.{ stream4, out4, inj });
    defer _ = mlx.mlx_array_free(next4);
    var next = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(next);
    try mlx.check(mlx.mlx_reshape(&next, next4, &[_]c_int{ batch, seq_len, hc * hidden }, 3, self.s));
    _ = mlx.mlx_array_free(stream);
    return next;
}

/// The trunk's compiled hc tail `c` on `ins` (silu, mix, inject gate, write): every K pack's load compiles all four.
fn tail(c: ?mlx.mlx_closure, ins: []const mlx.mlx_array) !mlx.mlx_array {
    return (try applyClosure(c, ins)) orelse error.DirectModelHcTailNotCompiled;
}

var fuse_tables: ?hpn.FuseTables = null;

/// The fused downs' silu and gate tables: the trunk's compiled silu and inject gate of every bf16 pattern, at the rank
/// the chain calls them (3), so a lookup holds the chain's bits.
fn fuseTables(self: *Trunk) !hpn.FuseTables {
    if (fuse_tables) |t| return t;
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(&bits, &[_]c_int{ 1, 1, 65536 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const silu3 = try tail(self.compiled_hc_silu, &.{x});
    defer _ = mlx.mlx_array_free(silu3);
    const gate3 = try tail(self.compiled_hc_inj, &.{x});
    defer _ = mlx.mlx_array_free(gate3);
    var t: hpn.FuseTables = .{ .silu = mlx.mlx_array_new(), .gate = mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(t.silu);
        _ = mlx.mlx_array_free(t.gate);
    }
    try mlx.check(mlx.mlx_reshape(&t.silu, silu3, &[_]c_int{65536}, 1, self.s));
    try mlx.check(mlx.mlx_reshape(&t.gate, gate3, &[_]c_int{65536}, 1, self.s));
    fuse_tables = t;
    return t;
}

/// The prefill launches' rows: one or two rows of 17 up to `max_seq` tokens (an 8192-token chunk with its merged tail).
fn geometry(batch: c_int, seq: c_int) bool {
    return batch >= 1 and batch <= 2 and seq > 16 and seq <= @divTrunc(max_seq, batch);
}

// ── the prefill launches ──

const Norm = struct { normalized: mlx.mlx_array, stream: mlx.mlx_array };

/// The norm kernel (a threadgroup a stream row, four values a thread): the normalized streams and, with `pend`, the
/// written stream (else null-ctx).
fn norm(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, hc: c_int, hidden: c_int, pend: ?HcPending) !Norm {
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, .norm, &.{ .{ .shape = &.{ batch, seq, hc, hidden } }, .{ .shape = if (pend != null) &.{ batch, seq, hc * hidden } else &.{ 1, 1, 1 } } }, .{ batch * seq * hc * @divTrunc(hidden, 4), 1, 1 }, @divTrunc(hidden, 4), true, &.{ .{ "H", hidden }, .{ "HC", hc }, .{ "WR", @intFromBool(pend != null) } }, &.{ x, w, eps, if (pend) |p| p.out else x, if (pend) |p| p.inj else x }, &res);
    if (pend == null) {
        _ = mlx.mlx_array_free(res[1]);
        res[1] = .{ .ctx = null };
    }
    return .{ .normalized = res[0], .stream = res[1] };
}

/// The norm kernel's f32 row scales [B S hc] alone, and its written stream (else null-ctx).
fn normInv(s: mlx.mlx_stream, x: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, hc: c_int, hidden: c_int, pend: ?HcPending) !hpn.NormInv {
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, .norm_inv, &.{ .{ .shape = if (pend != null) &.{ batch, seq, hc * hidden } else &.{ 1, 1, 1 } }, .{ .shape = &.{batch * seq * hc}, .dt = .float32 } }, .{ batch * seq * hc * @divTrunc(hidden, 4), 1, 1 }, @divTrunc(hidden, 4), true, &.{ .{ "H", hidden }, .{ "HC", hc }, .{ "WR", @intFromBool(pend != null) } }, &.{ x, eps, if (pend) |p| p.out else x, if (pend) |p| p.inj else x }, &res);
    if (pend == null) {
        _ = mlx.mlx_array_free(res[0]);
        res[0] = .{ .ctx = null };
    }
    return .{ .stream = res[0], .invs = res[1] };
}

/// The down product, silu, inject product and gate in one launch: act [B, S, R] from x [B, S, K] and `down_rows` (u32
/// words of the [R, K] bf16 rows), inj [B, S, hc, 1] from x . `iw` ([K, hc]; null-ctx for a mixer, and `inj` too).
fn fuseDown(s: mlx.mlx_stream, x: mlx.mlx_array, down_rows: mlx.mlx_array, iw: mlx.mlx_array, tabs: hpn.FuseTables, batch: c_int, seq: c_int, hc: c_int) !hpn.FuseOut {
    const ws = mlx.getShape(down_rows);
    const n = ws[0];
    const k = ws[1] * 2;
    const m = batch * seq;
    const inj = iw.ctx != null;
    const p = downPlan(m, n, k);
    const threads = 32 * p.np * p.wm * p.wn;
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    // Without an inject the kernel never reads `iw`; the silu table stands in.
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, .fuse_down, &.{ .{ .shape = &.{ batch, seq, n } }, .{ .shape = if (inj) &.{ batch, seq, hc, 1 } else &.{1} } }, tiled(m, 16 * p.tm * p.wm, @divExact(n, 32 * p.wn), p.swz, threads), threads, false, &.{ .{ "K", k }, .{ "N", n }, .{ "HC", hc }, .{ "INJ", @intFromBool(inj) }, .{ "TM", p.tm }, .{ "WM", p.wm }, .{ "WN", p.wn }, .{ "NP", p.np }, .{ "PART", p.part }, .{ "SWZ", p.swz }, .{ "BAR", p.bar } }, &.{ x, down_rows, if (inj) iw else tabs.silu, tabs.silu, tabs.gate, rows }, &res);
    if (!inj) {
        _ = mlx.mlx_array_free(res[1]);
        res[1] = .{ .ctx = null };
    }
    return .{ .act = res[0], .inj = res[1] };
}

/// fuseDown normalizing in its loader from the stream `xs`, the row scales `invs` and the norm weight, its inject
/// chain in simdgroups of its own.
fn rwDown(s: mlx.mlx_stream, xs: mlx.mlx_array, invs: mlx.mlx_array, w: *const HcWeights, tabs: hpn.FuseTables, batch: c_int, seq: c_int, hc: c_int, hidden: c_int) !hpn.FuseOut {
    const n = mlx.getShape(w.down_rows)[0];
    const m = batch * seq;
    // rwPlan's first tiles, which fit rank 320 at hidden 2560: 64 x 160 with one inject simdgroup, 64-deep stages.
    const p: hpn.RwPlan = .{ .tm = 2, .wm = 2, .wn = 5, .swz = 2, .sd = 2 };
    const tiles_n = @divExact(n, 32 * p.wn);
    const threads = 32 * (p.wm * p.wn + @divTrunc(p.wm + tiles_n - 1, tiles_n));
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    var res: [2]mlx.mlx_array = undefined;
    try launch(s, .rw_down, &.{ .{ .shape = &.{ batch, seq, n } }, .{ .shape = &.{ batch, seq, hc, 1 } } }, tiled(m, 16 * p.tm * p.wm, tiles_n, p.swz, threads), threads, false, &.{ .{ "K", hc * hidden }, .{ "N", n }, .{ "H", hidden }, .{ "HC", hc }, .{ "TM", p.tm }, .{ "WM", p.wm }, .{ "WN", p.wn }, .{ "SWZ", p.swz }, .{ "SD", p.sd } }, &.{ xs, invs, w.norm_w, w.down_rows, w.inject_flat, tabs.silu, tabs.gate, rows }, &res);
    return .{ .act = res[0], .inj = res[1] };
}

/// The up product with the mix as its epilogue, in registers: mixed [B, S, H] from act, reading `normalized` or
/// (rw_upmix) normalizing `from` = { the stream, its row scales, the norm weight }.
fn upMix(s: mlx.mlx_stream, act: mlx.mlx_array, up_rows: mlx.mlx_array, normalized: mlx.mlx_array, from: ?[3]mlx.mlx_array, batch: c_int, seq: c_int, hc: c_int, hidden: c_int) !mlx.mlx_array {
    const m = batch * seq;
    // regPlan's first tiles, which fit hc 4 x 2560: 4 x 2 simdgroups of 16 rows x 32 columns.
    const p: hpn.RegPlan = .{ .tm = 1, .wm = 4, .wn = 2, .swz = 2, .bar = 0 };
    const threads = 32 * p.wm * p.wn;
    const tmpl = [_]Tmpl{ .{ "K", mlx.getShape(up_rows)[1] * 2 }, .{ "H", hidden }, .{ "HC", hc }, .{ "TM", p.tm }, .{ "WM", p.wm }, .{ "WN", p.wn }, .{ "SWZ", p.swz }, .{ "BAR", p.bar }, .{ "PF", 0 } };
    const rows = mlx.mlx_array_new_int(m);
    defer _ = mlx.mlx_array_free(rows);
    const tab = try sigmoidTableFor(s, .bfloat16);
    const grid = tiled(m, 16 * p.tm * p.wm, @divExact(hidden, 32 * p.wn), p.swz, threads);
    var out: [1]mlx.mlx_array = undefined;
    if (from) |f| {
        try launch(s, .rw_upmix, &.{.{ .shape = &.{ batch, seq, hidden } }}, grid, threads, true, &tmpl, &.{ act, up_rows, f[0], f[1], f[2], tab, rows }, &out);
    } else try launch(s, .read_upmix, &.{.{ .shape = &.{ batch, seq, hidden } }}, grid, threads, true, tmpl[0..8], &.{ act, up_rows, normalized, tab, rows }, &out);
    return out[0];
}

/// The grid of a launch over BM-row x (32 TILES_N)-column blocks with MLX's threadblock swizzle (2^swz row tiles a column).
fn tiled(m: c_int, bm: c_int, tiles_n: c_int, swz: c_int, threads: c_int) [3]c_int {
    const tile = @as(c_int, 1) << @intCast(swz);
    const tiles_m = @divTrunc(m + bm - 1, bm);
    return .{ tiles_n * tile * threads, @divTrunc(tiles_m + tile - 1, tile), 1 };
}

/// Where MLX's bf16 NAX matmul [m, k] x [k, n] splits K (matmul.cpp's steel_matmul_axpby), its partition width: at
/// k 10240, 2048 on the 'd' device class for m, n <= 2048, else 4096. Null where it runs one chain over K.
fn mlxSplitK(m: c_int, n: c_int, k: c_int) ?c_int {
    const mx = @max(m, n);
    if (!(k >= 3 * mx or (mx <= 1024 and k > 2 * mx))) return null;
    const arch = mlx.gpuArchitectureName();
    return if (arch.len > 0 and arch[arch.len - 1] == 'd' and m <= 2048 and n <= 2048) 2048 else 4096;
}

/// fuseDown's tiles, splitting K where MLX does: one chain in 64 x 160 tiles, a barrier every 2 steps; else 32 x 64
/// tiles a partition (16 x 32 where 5 partitions' parked partials pass 28 KB).
fn downPlan(m: c_int, n: c_int, k: c_int) hpn.DownPlan {
    const part = mlxSplitK(m, n, k) orelse return .{ .tm = 2, .wm = 2, .wn = 5, .np = 1, .part = k, .swz = 0, .bar = 2 };
    const np = @divTrunc(k + part - 1, part);
    const fits = (np - 1) * 2 * 16 * 2 * 36 * 4 <= 28 * 1024;
    return .{ .tm = if (fits) 2 else 1, .wm = 1, .wn = if (fits) 2 else 1, .np = np, .part = part, .swz = 0, .bar = 0 };
}

// ── the decode launches ──

var eps_arr: ?mlx.mlx_array = null;
var eps_val: f32 = 0;

/// Their epsilon operand, a [1] array (they read `eps[0]`), cached per value.
fn epsArray(eps: f32) mlx.mlx_array {
    if (eps_arr == null or eps_val != eps) {
        if (eps_arr) |e| _ = mlx.mlx_array_free(e);
        var ev = eps;
        eps_arr = mlx.mlx_array_new_data(&ev, &[_]c_int{1}, 1, .float32);
        eps_val = eps;
    }
    return eps_arr.?;
}

/// A read's geometry, which fixes its configs; the last 16 stay armed (round-robin), so no layer re-arms.
const Key = struct { hc: c_int, h: c_int, r: c_int, inj: c_int, wr: c_int, rows: c_int };
const Armed = struct { key: Key, cfgs: [2]mlx.mlx_fast_metal_kernel_config };
var armed: [16]?Armed = @splat(null);
var armed_next: usize = 0;

/// The read of 1..16 rows: one in two launches (the norm folded into the down, then the up with the mix), more in
/// decode3's pair. The first launch applies `pend` (the written stream is `.stream`, else null-ctx).
fn decodeRead(self: *Trunk, x: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq: c_int, pend: ?HcPending) !HcFusedOut {
    const s = self.s;
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const inj: c_int = @intFromBool(w.inject_flat.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    const d3 = batch * seq > 1;
    // decode3 reads its down rows with 16-byte loads; the loader's rows are evaluated and aligned.
    if (d3 and !aligned16(w.down_rows)) return error.DirectModelHcRowsUnaligned;
    const key: Key = .{ .hc = hc, .h = hidden, .r = mlx.getShape(w.down_rows)[0], .inj = inj, .wr = wr, .rows = batch * seq };
    const cfgs = for (armed) |a| {
        if (a != null and std.meta.eql(a.?.key, key)) break a.?.cfgs;
    } else blk: {
        const fresh = if (d3) try d3Configs(key) else try oneRowConfigs(key);
        if (armed[armed_next]) |old| for (old.cfgs) |c| {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        armed[armed_next] = .{ .key = key, .cfgs = fresh };
        armed_next = (armed_next + 1) % armed.len;
        break :blk fresh;
    };
    const nw = w.norm_w;
    const iw = if (inj == 1) w.inject_flat else nw;
    const wo = if (pend) |p| p.out else nw;
    const wi = if (pend) |p| p.inj else nw;
    const eps = epsArray(self.config.rms_norm_eps);
    var nd: [3]mlx.mlx_array = undefined;
    // Dense rows carry no scales: the one-row launches never read the operands bound in their place.
    if (d3) try run(.d3n, cfgs[0], &.{ x, nw, iw, eps, wo, wi, w.down_rows }, s, &nd) else try run(.nd, cfgs[0], &.{ x, nw, iw, eps, wo, wi, w.down_rows, nw, nw }, s, &nd);
    defer for (nd) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var up: [2]mlx.mlx_array = undefined;
    if (d3) try run(.u3, cfgs[1], &.{ nd[0], nd[1], w.up_rows }, s, &up) else try run(.u2, cfgs[1], &.{ nd[0], nd[1], w.up_rows, nw, nw }, s, &up);
    defer for (up) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var out: HcFusedOut = .{ .mixed = mlx.mlx_array_new(), .inj = .{ .ctx = null }, .stream = .{ .ctx = null } };
    errdefer inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(out, name).ctx != null) _ = mlx.mlx_array_free(@field(out, name));
    };
    try mlx.check(mlx.mlx_reshape(&out.mixed, up[0], &[_]c_int{ batch, seq, hidden }, 3, s));
    if (inj == 1) {
        out.inj = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out.inj, up[1], &[_]c_int{ batch, seq, hc, 1 }, 4, s));
    }
    if (wr == 1) {
        const xsh = mlx.getShape(x);
        out.stream = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out.stream, nd[2], xsh.ptr, @intCast(xsh.len), s));
    }
    return out;
}

/// The two-launch read's configs: the norm + down launch over the down rows in groups of 8 (a simdgroup a row, the
/// inject's partials a group more), then the up + mix over 16-column groups.
fn oneRowConfigs(key: Key) ![2]mlx.mlx_fast_metal_kernel_config {
    const K = key.hc * key.h;
    const nd = try config(&.{ .{ .shape = &.{key.rows * K} }, .{ .shape = &.{key.rows * key.hc * (key.r + key.hc)}, .dt = .float32 }, .{ .shape = &.{if (key.wr == 1) key.rows * K else 1} } }, .{ 256, (@divExact(key.r, 8) + key.inj) * key.hc, key.rows }, 256, true, &.{ .{ "HC", key.hc }, .{ "H", key.h }, .{ "R", key.r }, .{ "GS", 16 }, .{ "BITS", 16 }, .{ "INJ", key.inj }, .{ "WR", key.wr }, .{ "RPS", 1 }, .{ "ROWS_TG", 8 } });
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(nd);
    return .{ nd, try config(&.{ .{ .shape = &.{key.rows * key.h} }, .{ .shape = &.{key.rows * key.hc} } }, .{ 512, @divExact(key.h, 16), key.rows }, 512, true, &.{ .{ "HC", key.hc }, .{ "H", key.h }, .{ "R", key.r }, .{ "GS", 16 }, .{ "BITS", 16 }, .{ "INJ", key.inj }, .{ "COLS", 16 } }) };
}

/// decode3's configs: a down threadgroup sums each slot once, a simdgroup computing D3_DRP down rows in turn; one up lane
/// per (row, stream).
fn d3Configs(key: Key) ![2]mlx.mlx_fast_metal_kernel_config {
    const K = key.hc * key.h;
    const dblk = d3Blocks(key.rows, hc_decode2.D3_RBD);
    const ublk = d3Blocks(key.rows, @divExact(32, key.hc));
    const down = try config(&.{ .{ .shape = &.{key.rows * K} }, .{ .shape = &.{key.rows * key.hc * (key.r + key.hc)}, .dt = .float32 }, .{ .shape = &.{if (key.wr == 1) key.rows * K else 1} } }, .{ 256, (@divExact(key.r, hc_decode2.D3_SGD * hc_decode2.D3_DRP) + key.inj) * key.hc, dblk[1] }, 256, true, &.{ .{ "HC", key.hc }, .{ "H", key.h }, .{ "R", key.r }, .{ "INJ", key.inj }, .{ "WR", key.wr }, .{ "DRP", hc_decode2.D3_DRP }, .{ "RBD", dblk[0] }, .{ "ROWS", key.rows }, .{ "LF", hc_decode2.D3_LF } });
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(down);
    return .{ down, try config(&.{ .{ .shape = &.{key.rows * key.h} }, .{ .shape = &.{key.rows * key.hc} } }, .{ 32 * hc_decode2.D3_SGU, @divExact(key.h, hc_decode2.D3_SGU), ublk[1] }, 32 * hc_decode2.D3_SGU, true, &.{ .{ "HC", key.hc }, .{ "H", key.h }, .{ "R", key.r }, .{ "INJ", key.inj }, .{ "SGU", hc_decode2.D3_SGU }, .{ "RBU", ublk[0] }, .{ "ROWS", key.rows }, .{ "WH", hc_decode2.D3_WH }, .{ "LF", hc_decode2.D3_LF } }) };
}

/// decode3's row blocks of at most `most` rows, balanced: rows per block and blocks.
fn d3Blocks(rows: c_int, most: c_int) [2]c_int {
    const n = @divFloor(rows + most - 1, most);
    return .{ @divFloor(rows + n - 1, n), n };
}

/// A lazy weight has no data pointer yet, so it is evaluated here: the launch must not depend on whether an earlier read
/// already did.
fn aligned16(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0) return false;
    if (!avail) mlx.check(mlx.mlx_array_eval(a)) catch return false;
    const p = mlx.mlx_array_data_uint32(a) orelse return false;
    return @intFromPtr(p) % 16 == 0;
}

// ── kernels and configs ──

const Kernel = enum { norm, norm_inv, fuse_down, read_upmix, rw_down, rw_upmix, nd, u2, d3n, u3 };
const Header = enum { none, nax, fuse, read, pack8 };
var built: [@typeInfo(Kernel).@"enum".field_names.len]?mlx.mlx_fast_metal_kernel = @splat(null);
var headers: [3]?[:0]const u8 = @splat(null);

/// `k`'s kernel, built at its first launch: its name, inputs, outputs, source and header.
fn kernelFor(k: Kernel) !mlx.mlx_fast_metal_kernel {
    if (built[@intFromEnum(k)]) |x| return x;
    const nd_outs: []const [*:0]const u8 = &.{ "xn_out", "parts_out", "xs_out" };
    const up_outs: []const [*:0]const u8 = &.{ "mixed_out", "inj_out" };
    const t: struct { [*:0]const u8, []const [*:0]const u8, []const [*:0]const u8, [*:0]const u8, Header } = switch (k) {
        .norm => .{ "msv_hc_prefill_norm", &.{ "x", "w", "eps", "wo", "wi" }, &.{ "normed", "stream" }, kernelSource("kernels/hc/hc_prefill_norm.metal"), .none },
        .norm_inv => .{ "msv_hc_prefill_norm_inv", &.{ "x", "eps", "wo", "wi" }, &.{ "stream", "invs" }, kernelSource("kernels/hc/hc_prefill_norm_inv.metal"), .none },
        .fuse_down => .{ "msv_hc_prefill_fuse_down", &.{ "x", "w", "iw", "stab", "gtab", "rows" }, &.{ "act", "inj" }, kernelSource("kernels/hc/hc_prefill_fuse_down.metal"), .fuse },
        .read_upmix => .{ "msv_hc_prefill_read_upmix", &.{ "act", "w", "normed", "sigtab", "rows" }, &.{"out"}, kernelSource("kernels/hc/hc_prefill_read_upmix.metal"), .read },
        .rw_down => .{ "msv_hc_prefill_rw_down", &.{ "xs", "invs", "nw", "w", "iw", "stab", "gtab", "rows" }, &.{ "act", "inj" }, kernelSource("kernels/hc/hc_prefill_rw_down.metal"), .read },
        .rw_upmix => .{ "msv_hc_prefill_rw_upmix", &.{ "act", "w", "xs", "invs", "nw", "sigtab", "rows" }, &.{"out"}, kernelSource("kernels/hc/hc_prefill_rw_upmix.metal"), .read },
        .nd => .{ "mlxserve_hc_read_nd", &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" }, nd_outs, kernelSource("kernels/hc/hc_decode2_nd.metal"), .pack8 },
        .u2 => .{ "mlxserve_hc_read_u2", &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" }, up_outs, kernelSource("kernels/hc/hc_decode2_u2.metal"), .pack8 },
        .d3n => .{ "mlxserve_hc_read_d3n", &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q" }, nd_outs, kernelSource("kernels/hc/hc_decode3_nd.metal"), .pack8 },
        .u3 => .{ "mlxserve_hc_read_u3", &.{ "xn_in", "parts_in", "uw_q" }, up_outs, kernelSource("kernels/hc/hc_decode3_up.metal"), .pack8 },
    };
    const ins = mlx.mlx_vector_string_new_data(t[1].ptr, t[1].len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(t[2].ptr, t[2].len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const x = mlx.mlx_fast_metal_kernel_new(t[0], ins, outs, t[3], try header(t[4]), true, false);
    if (x.ctx == null) return error.MetalKernelCompileFailed;
    built[@intFromEnum(k)] = x;
    return x;
}

/// The decode pack header, or MLX's steel NAX headers with each layer of the prefill read's helpers one file more.
fn header(h: Header) ![*:0]const u8 {
    switch (h) {
        .none => return "",
        .pack8 => return kernelSource("kernels/hc/hc_decode2_pack8_header.metal"),
        else => {},
    }
    var base: [:0]const u8 = try lane_qmm.prefillHeader();
    inline for (.{ Header.nax, Header.fuse, Header.read }, .{ "nax", "fuse", "read" }, 0..) |layer, file, i| {
        if (headers[i] == null) headers[i] = try std.mem.concatWithSentinel(std.heap.c_allocator, u8, &.{ base, kernelSource("kernels/hc/hc_prefill_" ++ file ++ ".metal") }, 0);
        base = headers[i].?;
        if (h == layer) return base.ptr;
    }
    unreachable;
}

const Out = struct { shape: []const c_int, dt: mlx.mlx_dtype = .bfloat16 };
const Tmpl = struct { [*:0]const u8, c_int };

/// A launch's config: outputs, grid (threads), threadgroup width, bf16 `T` if taken, template ints in kernel order.
fn config(outs: []const Out, grid: [3]c_int, tg: c_int, t: bool, tmpl: []const Tmpl) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    if (t) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    for (tmpl) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, a[0], a[1]));
    return c;
}

/// A prefill launch of `k`, its config built for the call.
fn launch(s: mlx.mlx_stream, k: Kernel, outs: []const Out, grid: [3]c_int, tg: c_int, t: bool, tmpl: []const Tmpl, ins: []const mlx.mlx_array, res: []mlx.mlx_array) !void {
    const cfg = try config(outs, grid, tg, t, tmpl);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try run(k, cfg, ins, s, res);
}

/// `k` applied to `inputs`; its outputs land in `res` (the caller's).
fn run(k: Kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, s: mlx.mlx_stream, res: []mlx.mlx_array) !void {
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, try kernelFor(k), v, cfg, s));
    var got: usize = 0;
    errdefer for (res[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(r, o, i));
    }
}
