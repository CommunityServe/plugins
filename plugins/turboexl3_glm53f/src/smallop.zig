//! `TURBOEXL3_GLM53F_SMALLOP_FUSE=1` (opt-in, exact): the small ops around the decode and verify forwards' big launches,
//! merged where a kernel gives the same bits (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch):
//! - a MoE layer's routed sum and shared expert, handed unadded to the next sublayer's hcPre (glm5_next.zig's
//!   `hcPre`'s defer2 text adds them in T as `mlx_add` rounds them, then expands): one dispatch less a MoE layer;
//! - a DSA layer's q_a and kv_a RMSNorms in one launch (`normPair`, smallop_norm_pair.metal);
//! - a dense layer's gate and up in one DEC-ONE launch (`denseGateUp`: turboexl3's glm53f.shared_gu, each member at its
//!   own K split, suh and svh);
//! - a DSA layer's q_a and kv_a in one DEC-ONE launch (`dsaQkv`: turboexl3's glm53f.pair_ks, each member at its own K
//!   split, 8 and 32, suh and svh);
//! - the MTP head's draft step at one row: its embedding row, both norms and their concatenation in one launch
//!   (`front`, smallop_front.metal), its first residual and post-norm through the kit's `addNorm`, and its last
//!   residual (the MoE's two terms and the residual) and final norm in one launch (`add2Norm`, smallop_add2_norm.metal).
//! The norms are the kit's add_norm.fmt.metal mirror of MLX's rms_norm single-row kernel. Every piece declines (null:
//! the caller runs the stock ops) off its rows, shapes or dtypes; the decode and verify rows are 1 to `max_rows`.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const log = @import("mlx_serve/sdk/util/log.zig");
/// glm5_next.zig's `hc_pre2_ok`: whether the defer2 text runs (the summary's `defer2`)
const glm5 = @import("mlx_serve/plugins/glm5_next/glm5_next.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_SMALLOP_FUSE`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (an A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE=1: decode and verify rows' small ops merged where a kernel gives the same bits: a MoE layer's shared-expert add inside the next hcPre, a DSA layer's q_a and kv_a in one launch and its two norms in one launch, a dense layer's gate and up in one launch, and the MTP draft step's embedding, norms, concatenation and residual adds (the kit's mirror of MLX's rms_norm)\n";

/// The rows the pieces take: decode and verify rows (glm5_next.zig's `HC_PRE_MAX_ROWS`)
pub const max_rows: c_int = 16;
/// The widest row the norm mirror takes in one pass (MLX's single-row kernel: ceil(D / 4) threads, at most 1024)
pub const max_norm_width: c_int = 4096;

/// Launches since the load, by piece; `declined`: calls a piece did not take (the stock ops ran)
pub var hc_pairs: u64 = 0;
pub var pairs: u64 = 0;
pub var qkv: u64 = 0;
pub var dense: u64 = 0;
pub var fronts: u64 = 0;
pub var tails: u64 = 0;
pub var declined: u64 = 0;
var engaged_said = false;

fn engaged(what: []const u8) void {
    if (engaged_said) return;
    engaged_said = true;
    log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE engaged: first piece {s}\n", .{what});
}

// ── admission (MLX-free) ──

/// Whether the norm mirror takes a row of `d` (one pass of ceil(d / 4) threads, whole simdgroups)
pub fn normWidthServes(d: c_int) bool {
    return d >= 128 and d <= max_norm_width and @rem(d, 128) == 0;
}

/// Whether `front` takes a draft step: one row, a bf16 embedding table of `width` unquantized, a bf16 hidden of `width`,
/// an int32 or uint32 token id
pub fn frontServes(rows: c_int, width: c_int, table_dt: mlx.mlx_dtype, quantized: bool, hidden_dt: mlx.mlx_dtype, ids_dt: mlx.mlx_dtype) bool {
    return rows == 1 and normWidthServes(width) and table_dt == .bfloat16 and !quantized and hidden_dt == .bfloat16 and (ids_dt == .int32 or ids_dt == .uint32);
}

/// Whether `normPair` takes two norms of `d0` and `d1` over `rows` rows of one dtype
pub fn pairServes(rows: c_int, d0: c_int, d1: c_int, dt: mlx.mlx_dtype) bool {
    return rows >= 1 and rows <= max_rows and normWidthServes(d0) and normWidthServes(d1) and d0 >= d1 and (dt == .bfloat16 or dt == .float16);
}

/// Whether `add2Norm` takes `rows` rows of `d` in one dtype
pub fn tailServes(rows: c_int, d: c_int, dt: mlx.mlx_dtype) bool {
    return rows >= 1 and rows <= max_rows and normWidthServes(d) and (dt == .bfloat16 or dt == .float16);
}

// ── the kernels ──

const front_source = @embedFile("kernels/smallop_front.metal");
const pair_source = @embedFile("kernels/smallop_norm_pair.metal");
const tail_source = @embedFile("kernels/smallop_add2_norm.metal");
var front_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pair_kernel: ?mlx.mlx_fast_metal_kernel = null;
var tail_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn rowsOf(sh: []const c_int) c_int {
    var r: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| r *= d;
    return r;
}

fn run(kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array, out: []Array, s: mlx.mlx_stream) !void {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != out.len) return error.MetalKernelBadOutputCount;
    for (out) |*a| a.* = .{ .ctx = null };
    errdefer for (out) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (out, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    }
}

/// The draft step's front: `cat[enorm(embed[ids]), hnorm(hidden)]` [1, 1, 2 width] as the stock step makes it (the
/// table's row taken, both norms, the concatenation along the last axis), one launch. `table` is the trunk's embedding
/// as loaded (bf16 [vocab, width], unquantized: `quantized` false), `ids` int32 or uint32 [1, 1], `hidden` bf16 [1, 1, width],
/// `eps` the trunk's f32 scalar (`Model.rms_eps`). Null where it does not serve.
pub fn front(s: mlx.mlx_stream, table: Array, quantized: bool, ids: Array, hidden: Array, we: Array, wh: Array, eps: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const tsh = mlx.getShape(table);
    const hsh = mlx.getShape(hidden);
    if (tsh.len != 2 or hsh.len != 3 or hsh[0] != 1 or mlx.mlx_array_size(ids) != 1) return null;
    const width = tsh[1];
    if (hsh[2] != width or mlx.getShape(we)[0] != width or mlx.getShape(wh)[0] != width or mlx.mlx_array_dtype(we) != .bfloat16 or mlx.mlx_array_dtype(wh) != .bfloat16) return null;
    if (!frontServes(rowsOf(hsh), width, mlx.mlx_array_dtype(table), quantized, mlx.mlx_array_dtype(hidden), mlx.mlx_array_dtype(ids))) {
        declined += 1;
        return null;
    }
    const tn = @divTrunc(width, 4);
    const k = try kernelOf(&front_kernel, "turboexl3_glm53f_smallop_front", &.{ "ids", "E", "H", "WE", "WH", "eps" }, &.{"CAT"}, front_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 2 * width }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 2 * tn, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tn, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", width));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TN", tn));
    var o: [1]Array = undefined;
    try run(k, cfg, &.{ ids, table, hidden, we, wh, eps }, &o, s);
    fronts += 1;
    engaged("the draft step's front");
    return o[0];
}

/// A DSA layer's `rms_norm(q_a, w0)` and `rms_norm(kv_a, w1)` at decode and verify rows, as the two `fast::rms_norm`
/// calls make them, one launch: [q_resid, lat], each in its input's shape. Null where it does not serve.
pub fn normPair(s: mlx.mlx_stream, x0: Array, w0: Array, x1: Array, w1: Array, eps: Array) !?[2]Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const s0 = mlx.getShape(x0);
    const s1 = mlx.getShape(x1);
    if (s0.len < 2 or s0.len != s1.len) return null;
    for (s0[0 .. s0.len - 1], s1[0 .. s1.len - 1]) |a, b| if (a != b) return null;
    const dt = mlx.mlx_array_dtype(x0);
    const d0 = s0[s0.len - 1];
    const d1 = s1[s1.len - 1];
    if (mlx.mlx_array_dtype(x1) != dt or mlx.mlx_array_dtype(w0) != dt or mlx.mlx_array_dtype(w1) != dt or mlx.getShape(w0)[0] != d0 or mlx.getShape(w1)[0] != d1) return null;
    const rows = rowsOf(s0);
    if (!pairServes(rows, d0, d1, dt)) {
        if (rows <= max_rows) declined += 1;
        return null;
    }
    const tn0 = @divTrunc(d0, 4);
    const k = try kernelOf(&pair_kernel, "turboexl3_glm53f_smallop_norm_pair", &.{ "X0", "W0", "X1", "W1", "eps" }, &.{ "Y0", "Y1" }, pair_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, s0.ptr, s0.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, s1.ptr, s1.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 2 * tn0, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tn0, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "D0", d0 }, .{ "TN0", tn0 }, .{ "D1", d1 }, .{ "TN1", @divTrunc(d1, 4) } }) |ta| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, ta[0], ta[1]));
    }
    var o: [2]Array = undefined;
    try run(k, cfg, &.{ x0, w0, x1, w1, eps }, &o, s);
    pairs += 1;
    engaged("a DSA layer's two norms");
    return o;
}

/// The draft step's last residual and norm: `rms_norm(h + (a + b), w)`, each add in T as `mlx_add` rounds it (the MoE's
/// routed sum `a` and shared expert `b`, then the residual `h`), one launch: the normed rows in `h`'s shape. Null where
/// it does not serve.
pub fn add2Norm(s: mlx.mlx_stream, h: Array, a: Array, b: Array, w: Array, eps: Array) !?Array {
    if (!on or !mlx.streamIsGpu(s)) return null;
    const sh = mlx.getShape(h);
    if (sh.len < 2) return null;
    const dt = mlx.mlx_array_dtype(h);
    const d = sh[sh.len - 1];
    for ([_]Array{ a, b }) |x| {
        if (mlx.mlx_array_dtype(x) != dt or !std.mem.eql(c_int, mlx.getShape(x), sh)) return null;
    }
    if (mlx.mlx_array_dtype(w) != dt or mlx.getShape(w)[0] != d) return null;
    const rows = rowsOf(sh);
    if (!tailServes(rows, d, dt)) {
        declined += 1;
        return null;
    }
    const tn = @divTrunc(d, 4);
    const k = try kernelOf(&tail_kernel, "turboexl3_glm53f_smallop_add2_norm", &.{ "H", "A", "B", "W", "eps" }, &.{"HN"}, tail_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tn * rows, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tn, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TN", tn));
    var o: [1]Array = undefined;
    try run(k, cfg, &.{ h, a, b, w, eps }, &o, s);
    tails += 1;
    engaged("the draft step's last residual");
    return o[0];
}

// ── the joined pairs: a dense layer's gate | up, a DSA layer's q_a | kv_a ──

/// A registry of pairs of EXL3 projections of one input: member 0's trellis (the key) and the two svh joined along OUT
const Entry = struct { key: ?*anyopaque, svh: Array };
fn Registry(comptime max: usize) type {
    return struct {
        entries: [max]Entry = undefined,
        n: usize = 0,
        bytes: u64 = 0,
        fn clear(r: *@This()) void {
            for (r.entries[0..r.n]) |e| _ = mlx.mlx_array_free(e.svh);
            r.n = 0;
            r.bytes = 0;
        }
        /// m0 and m1 (as stored): their svh joined. Refused, by the error turboexl3 names, where the two do not pair.
        fn register(r: *@This(), s: mlx.mlx_stream, m0: t3.Proj, m1: t3.Proj) !void {
            if (r.n == max) return error.Glm53fSmallopLayers;
            const p = try t3.glm53f.shared_gu.pairOf(m0, m1);
            const svh = try t3.glm53f.shared_gu.joinSvh(s, m0.svh, m1.svh);
            r.entries[r.n] = .{ .key = m0.trellis.ctx, .svh = svh };
            r.n += 1;
            r.bytes += 2 * @as(u64, p.out0 + p.out1);
        }
        /// The registered pair's joined svh, MLX-free, or null (the switch off, a linear not EXL3, a pair not registered)
        fn admit(r: *const @This(), m0: ?t3.Proj, m1: ?t3.Proj) ?Array {
            if (!on) return null;
            const a = m0 orelse return null;
            _ = m1 orelse return null;
            for (r.entries[0..r.n]) |e| if (e.key == a.trellis.ctx) return e.svh;
            return null;
        }
    };
}
/// The dense layers' gate | up (3 in GLM-5.3-Flash) and the DSA layers' q_a | kv_a (the trunk's 11 and the MTP layer's)
var dense_reg: Registry(8) = .{};
var qkv_reg: Registry(16) = .{};

/// Drops the registries and the counts (the bank's opening, before it registers again; the model's unload).
pub fn reset() void {
    dense_reg.clear();
    qkv_reg.clear();
    hc_pairs = 0;
    pairs = 0;
    qkv = 0;
    dense = 0;
    fronts = 0;
    tails = 0;
    declined = 0;
    engaged_said = false;
}

/// One dense layer's gate and up (EXL3 projections as stored)
pub fn registerDense(s: mlx.mlx_stream, gate: t3.Proj, up: t3.Proj) !void {
    return dense_reg.register(s, gate, up);
}

/// One DSA layer's q_a and kv_a (EXL3 projections as stored)
pub fn registerQkv(s: mlx.mlx_stream, q_a: t3.Proj, kv_a: t3.Proj) !void {
    return qkv_reg.register(s, q_a, kv_a);
}

pub fn denseLayers() usize {
    return dense_reg.n;
}

pub fn denseBytes() u64 {
    return dense_reg.bytes;
}

pub fn qkvLayers() usize {
    return qkv_reg.n;
}

pub fn qkvBytes() u64 {
    return qkv_reg.bytes;
}

fn admitDense(gate: ?t3.Proj, up: ?t3.Proj) ?Array {
    return dense_reg.admit(gate, up);
}

fn admitQkv(q_a: ?t3.Proj, kv_a: ?t3.Proj) ?Array {
    return qkv_reg.admit(q_a, kv_a);
}

/// A DSA layer's q_a and kv_a of x at decode and verify rows in one launch (turboexl3's glm53f.pair_ks: each member its
/// own linear's bytes at its own K split), new references the caller owns; null (the caller runs the two linears) where
/// it does not serve.
pub fn dsaQkv(s: mlx.mlx_stream, q_a: ?t3.Proj, kv_a: ?t3.Proj, x: Array) !?[2]Array {
    const svh = admitQkv(q_a, kv_a) orelse return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2 or rowsOf(xsh) > max_rows) return null;
    const r = t3.glm53f.pair_ks.linear2(s, x, q_a.?, kv_a.?, svh, .mul1) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => {
            declined += 1;
            return null;
        },
        else => return e,
    };
    qkv += 1;
    engaged("a DSA layer's q_a and kv_a");
    return r;
}

/// A dense layer's gate and up of x at decode and verify rows in one launch (turboexl3's glm53f.shared_gu: each member
/// its own linear's bytes), new references the caller owns; null (the caller runs the two linears) where it does not
/// serve.
pub fn denseGateUp(s: mlx.mlx_stream, gate: ?t3.Proj, up: ?t3.Proj, x: Array) !?[2]Array {
    const svh = admitDense(gate, up) orelse return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2 or rowsOf(xsh) > max_rows) return null;
    const r = t3.glm53f.shared_gu.gateUp(s, x, gate.?, up.?, svh, .mul1) catch |e| switch (e) {
        error.Exl3EngineNotImplemented => {
            declined += 1;
            return null;
        },
        else => return e,
    };
    dense += 1;
    engaged("a dense layer's gate and up");
    return r;
}

/// The counts' line for the request summary (`ModelTable.meters`). `defer2`: whether the next hcPre takes a MoE layer's
/// two terms in its defer2 text (on), in the stock add and defer text (declined: Metal caps the text's threadgroup), or
/// has not met a pair yet (unlaunched).
pub fn summary(buf: []u8) []const u8 {
    const d2 = if (glm5.hc_pre2_ok) |ok| (if (ok) "on" else "declined") else "unlaunched";
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE hc_pairs={d} defer2={s} norm_pairs={d} dsa_qkv={d} dense={d} fronts={d} tails={d} declined={d} dense_layers={d} dsa_layers={d}\n", .{ hc_pairs, d2, pairs, qkv, dense, fronts, tails, declined, dense_reg.n, qkv_reg.n }) catch "[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE (summary too long)\n";
}

const testing = std.testing;

test "turboexl3_glm53f stream smallop: each piece takes decode and verify rows of the widths it mirrors and declines prompt rows, other widths and dtypes; a dense or DSA pair is taken only when registered, each in its own registry" {
    // the draft front: one row of 4096, a bf16 table unquantized, a bf16 hidden, int32 or uint32 ids
    try testing.expect(frontServes(1, 4096, .bfloat16, false, .bfloat16, .int32));
    try testing.expect(!frontServes(2, 4096, .bfloat16, false, .bfloat16, .int32));
    try testing.expect(!frontServes(1, 4096, .bfloat16, true, .bfloat16, .int32));
    try testing.expect(!frontServes(1, 4096, .bfloat16, false, .float32, .int32));
    try testing.expect(frontServes(1, 4096, .bfloat16, false, .bfloat16, .uint32));
    try testing.expect(!frontServes(1, 4096, .bfloat16, false, .bfloat16, .float32));
    try testing.expect(!frontServes(1, 8192, .bfloat16, false, .bfloat16, .int32));
    // the DSA pair: q_a 1536 and kv_a 512 at 1 to 16 rows; a prompt's rows, a pair wider second, f32 decline
    for (1..17) |r| try testing.expect(pairServes(@intCast(r), 1536, 512, .bfloat16));
    try testing.expect(!pairServes(17, 1536, 512, .bfloat16));
    try testing.expect(!pairServes(16356, 1536, 512, .bfloat16));
    try testing.expect(!pairServes(3, 512, 1536, .bfloat16));
    try testing.expect(!pairServes(3, 1536, 512, .float32));
    try testing.expect(!pairServes(3, 1536, 520, .bfloat16));
    // the draft tail: rows of 4096
    try testing.expect(tailServes(1, 4096, .bfloat16));
    try testing.expect(!tailServes(1, 4100, .bfloat16));
    // the registries: off, a missing projection, an unregistered and a registered pair (no array made or freed)
    defer {
        dense_reg.n = 0;
        qkv_reg.n = 0;
        on = false;
    }
    var keys: [2]u8 = undefined;
    const proj = struct {
        fn of(key: ?*anyopaque) t3.Proj {
            return .{ .trellis = .{ .ctx = key }, .suh = .{ .ctx = null }, .svh = .{ .ctx = null } };
        }
    }.of;
    const gate = proj(&keys[0]);
    const up = proj(&keys[1]);
    const joined: Array = .{ .ctx = &keys[1] };
    dense_reg.n = 0;
    qkv_reg.n = 0;
    on = false;
    try testing.expect(admitDense(gate, up) == null);
    on = true;
    try testing.expect(admitDense(null, up) == null);
    try testing.expect(admitDense(gate, null) == null);
    try testing.expect(admitDense(gate, up) == null);
    dense_reg.entries[0] = .{ .key = gate.trellis.ctx, .svh = joined };
    dense_reg.n = 1;
    try testing.expectEqual(joined.ctx, (admitDense(gate, up) orelse return error.TestUnexpectedResult).ctx);
    try testing.expect(admitDense(up, gate) == null);
    // a DSA pair: its own registry (the dense layer's key is not a DSA layer's)
    try testing.expect(admitQkv(gate, up) == null);
    qkv_reg.entries[0] = .{ .key = up.trellis.ctx, .svh = joined };
    qkv_reg.n = 1;
    try testing.expectEqual(joined.ctx, (admitQkv(up, gate) orelse return error.TestUnexpectedResult).ctx);
    try testing.expect(admitQkv(gate, up) == null);
    try testing.expect(admitQkv(up, null) == null);
    on = false;
    try testing.expect(admitQkv(up, gate) == null);
    on = true;
    var buf: [320]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE hc_pairs=0 defer2=unlaunched norm_pairs=0 dsa_qkv=0 dense=0 fronts=0 tails=0 declined=0 dense_layers=1 dsa_layers=1\n", summary(&buf));
    try testing.expect(std.mem.startsWith(u8, line, "[turboexl3_glm53f] TURBOEXL3_GLM53F_SMALLOP_FUSE=1: "));
}
