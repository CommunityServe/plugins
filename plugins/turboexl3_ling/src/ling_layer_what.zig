//! turboexl3_ling's layer-major layer's dense prefill linears over W_hats decoded once for its spans (LAYER_WHAT).
//! Lossless: it removes launches and keeps every word. P3: SHARED_LAYER_WHAT's mechanism (src/ling_shared_layer.zig) on
//! the trunk's other dense linears that decode a W_hat per span.
//! Under LAYER_MAJOR the prompt pass runs each layer over every span before the next layer (the copied table_forward.zig
//! `hiddenSpans`). A per-call EXL3 linear at a wide span (more than 32 rows: the matmul path) decodes its trellis to an
//! f16 W_hat on every call: 5 times a layer at the standard request's spans (4 of 4,096 rows and 1,965; the 31-row span
//! runs the register path, which decodes into registers and makes no W_hat). `hiddenSpans` opens a scope at the layer's
//! start (`begin`) that holds the layer's EXL3 linears of every family (`Family`):
//! - kda: a KDA layer's q|k|v, f and g, on KDA_PROJ_PROMPT's grouped launch (src/ling_kdaproj.zig `projPrompt`, over
//!   turboexl3's `ling.kdaProjPromptW`; KDA_GATE_EPI's `projPromptGate` over `ling.kdaProjPromptWGate` first), else
//!   each member's `linear`;
//! - o: o_proj, a KDA layer's and an MLA layer's (`dense`); a KDA layer's from the norm-gate launch's prepared rows
//!   first (KDA_NORM_PREP, src/ling_kda_norm_prep.zig `oProj`: `preparedDense`, then `linearPrepared`), else `linear`;
//! - mla: an MLA layer's q_a, q_b and kv_a (or a direct q_proj);
//! - dense: a dense layer's MLP gate, up and down, up on SWIGLU_EPI's epilogue (src/ling_swiglu_epi.zig `up`, over
//!   turboexl3's `lingUpSwigluW`), else its `linear`.
//! The layer's first wide span decodes each held linear's W_hat by the per-call launch (`Dense.dequantize`, lazy) and the
//! scope keeps it; each wide span runs the per-call launches but the decode over the kept array: `linear`'s prepare,
//! MLX's matmul and finish (`Dense.pieceInner`, `Dense.finishOf`: src/ling_shared_layer.zig `linearOver`), the grouped
//! launch or the epilogue, so every word is the per-call path's (the same launches on the same operands; the decode
//! writes each weight from its own trellis tile). The scope keys each array on the trellis handle the model's `Linear`
//! holds. `end` drops the scope's arrays: before a MoE layer's routed calls (their route read evaluates the mixer, so
//! the W_hats are gone before the routed working set exists) and after a dense layer's MLP; the layer's graph holds what
//! it reads until it is evaluated. Not here: b_proj, kv_b and the MLA gate are plain f16 (no trellis); the routed experts
//! are the streamer's and the shared expert is SHARED_LAYER_WHAT's; the head and the NextN layer run outside the layers.
//! Memory: at most one layer's held W_hats at once, f16 in x out x 2: a KDA layer's q|k|v (192 MiB), f, g and o_proj
//! (64 MiB each), 384 MiB; an MLA layer's q_a (12), q_b (36), kv_a (5, 640 stored outputs) and o_proj (64), 117 MiB; a
//! dense layer adds its MLP's gate, up and down (80 MiB each), 624 MiB. Alive only while the layer's mixer (and a dense
//! layer's MLP) is evaluated, when no routed call holds the bill's layer-major routed rows (3,584 MiB at the 18,432-row
//! call), and inside MEM_REFIT's headroom (src/bill.zig `mem_refit_extra`): no bill term.
//! Null keeps the per-call path: outside a scope (the chunk loop, decode and verify rows, the head, the NextN layer), a
//! linear the scope does not hold, rows the matmul path does not take, pg2 on, an un-rotated resident trellis.
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ran = @import("ling_ran.zig");
/// `linearOver`: `linear`'s launches over a given W_hat
const sl = @import("ling_shared_layer.zig");

pub const Proj = t3.Proj;
const Dense = t3.dense.Dense;

/// The linears a scope holds, by family (the scope's and the pass's counts)
pub const Family = enum(u2) {
    /// a KDA layer's q|k|v, f and g
    kda,
    /// o_proj, a KDA layer's and an MLA layer's
    o,
    /// an MLA layer's q_a, q_b and kv_a (or its direct q_proj)
    mla,
    /// a dense layer's MLP: gate, up and down
    dense,
};

// ── the scope: one layer's spans (`hiddenSpans`) ──

/// A held linear: its trellis handle's key, its projection, its family, the kept W_hat (null until its first wide span)
/// and the scope's calls that read it
const Held = struct { key: usize, p: Proj, family: Family, w: mlx.mlx_array = .{ .ctx = null }, reads: u32 = 0 };
/// The most linears a scope holds: an MLA layer's five (q or q_a, q_b, kv_a, o_proj) or a dense KDA layer's seven, with
/// room to spare
const max_held = 12;
var held: [max_held]Held = undefined;
var n_held: usize = 0;
var scope_open: bool = false;

/// By family: the W_hats read at least once (each decoded once, by the call that first read it) and the calls that read
/// a kept W_hat (every wide span's call of a held linear)
pub const Counts = struct {
    decoded: [4]u32 = @splat(0),
    read: [4]u32 = @splat(0),

    pub fn decodedAll(c: Counts) u32 {
        var n: u32 = 0;
        for (c.decoded) |v| n += v;
        return n;
    }
    pub fn readAll(c: Counts) u32 {
        var n: u32 = 0;
        for (c.read) |v| n += v;
        return n;
    }
    /// The decodes the scope or pass saved: the reads past each W_hat's first
    pub fn saved(c: Counts) u32 {
        return c.readAll() - c.decodedAll();
    }
};

/// A layer-major pass's counts so far (`passBegin` resets them): every closed scope's and the open one's reads, the
/// layers that read a kept W_hat, and their wide spans past the first (each layer's most-read W_hat's reads less one)
var pass_counts: Counts = .{};
var pass_layers: u32 = 0;
var pass_reused: u32 = 0;

/// The scope's and the pass's state and counts (the byte test)
pub const Stats = struct {
    open: bool,
    /// the linears the scope holds, and the W_hats it keeps
    held: usize,
    kept: usize,
    /// the open scope's counts
    scope: Counts,
    pass: Counts,
    layers: u32,
    spans_reused: u32,
};

pub fn stats() Stats {
    var c: Counts = .{};
    var k: usize = 0;
    for (held[0..n_held]) |h| {
        const f = @intFromEnum(h.family);
        c.read[f] += h.reads;
        c.decoded[f] += @intFromBool(h.reads > 0);
        k += @intFromBool(h.w.ctx != null);
    }
    return .{ .open = scope_open, .held = n_held, .kept = k, .scope = c, .pass = pass_counts, .layers = pass_layers, .spans_reused = pass_reused };
}

/// A layer-major pass begins (`hiddenSpans`, before its first layer): its counts from zero.
pub fn passBegin() void {
    pass_counts = .{};
    pass_layers = 0;
    pass_reused = 0;
}

/// A layer-major pass's layers are done: a pass whose layers read a kept W_hat past its first read marks the run-path
/// line.
pub fn passEnd() void {
    if (pass_counts.saved() > 0) ran.mark(.LAYER_WHAT);
}

/// A layer's spans begin (`hiddenSpans`, before its mixer): a scope holding the layer's EXL3 linears, from the trunk's
/// `Layer` (`lw`: its `mixer`, a KDA layer's `qkv`, `qkv_parts`, `a` (f), `z` (g) and `out`, an MLA layer's `q`, `q_a`,
/// `q_b`, `kv_a` and `o`; its `mlp`, a dense layer's `gate`, `up` and `down`; each a `Linear`'s `w`, `s`, `b` and
/// `exl3`), nothing kept yet.
pub fn begin(lw: anytype) void {
    drop();
    scope_open = true;
    switch (lw.mixer) {
        .kda => |la| {
            hold(la.qkv, .kda);
            for (la.qkv_parts) |q| hold(q, .kda);
            hold(la.a, .kda);
            hold(la.z, .kda);
            hold(la.out, .o);
        },
        .mla => |at| {
            hold(at.q, .mla);
            hold(at.q_a, .mla);
            hold(at.q_b, .mla);
            hold(at.kv_a, .mla);
            hold(at.o, .o);
        },
        else => {},
    }
    switch (lw.mlp) {
        .dense => |dw| {
            hold(dw.gate, .dense);
            hold(dw.up, .dense);
            hold(dw.down, .dense);
        },
        else => {},
    }
}

/// `l` (a `Linear`) held in the scope when it is an EXL3 projection
fn hold(l: anytype, f: Family) void {
    if (!l.exl3) return;
    const key = @intFromPtr(l.w.ctx orelse return);
    if (n_held == max_held) return;
    held[n_held] = .{ .key = key, .p = .{ .trellis = l.w, .suh = l.s, .svh = l.b }, .family = f };
    n_held += 1;
}

/// A layer's spans are done (before a MoE layer's routed calls, after a dense layer's MLP; again at the layer's end, a
/// no-op then): the scope's W_hats dropped (the layer's graph holds what it reads until it is evaluated), its counts
/// into the pass's.
pub fn end() void {
    if (scope_open) {
        var most: u32 = 0;
        for (held[0..n_held]) |h| most = @max(most, h.reads);
        if (most > 0) pass_layers += 1;
        if (most > 1) pass_reused += most - 1;
    }
    drop();
    scope_open = false;
}

fn drop() void {
    for (held[0..n_held]) |h| if (h.w.ctx != null) {
        _ = mlx.mlx_array_free(h.w);
    };
    n_held = 0;
}

/// The held linear of `trellis` in the open scope
fn find(trellis: mlx.mlx_array) ?*Held {
    const key = @intFromPtr(trellis.ctx orelse return null);
    for (held[0..n_held]) |*h| if (h.key == key) return h;
    return null;
}

/// `h`'s W_hat in the scope: the kept array (borrowed), else decoded now by `d`'s per-call launch (lazy) and kept
fn what(s: mlx.mlx_stream, h: *Held, d: *Dense) !mlx.mlx_array {
    if (h.w.ctx == null) h.w = try d.dequantize(s, h.p.trellis);
    return h.w;
}

/// A served call read `h`'s kept W_hat: the first read is the decode's, every later one a decode saved
fn record(h: *Held) void {
    h.reads += 1;
    const f = @intFromEnum(h.family);
    pass_counts.read[f] += 1;
    if (h.reads == 1) pass_counts.decoded[f] += 1;
}

/// x [..., in] as rows: their count, their width and dtype
const Rows = struct { n: usize, in: u32, dtype: mlx.mlx_dtype };

fn rowsOf(x: mlx.mlx_array) ?Rows {
    const xsh = mlx.getShape(x);
    if (xsh.len < 1) return null;
    var n: usize = 1;
    for (xsh[0 .. xsh.len - 1]) |r| n *= @intCast(r);
    return .{ .n = n, .in = @intCast(xsh[xsh.len - 1]), .dtype = mlx.mlx_array_dtype(x) };
}

/// `linear`'s matmul path at `n` rows for `p` on `d`, the arm that decodes a W_hat per call: not the GEMV's or the
/// register path's rows, not the fused-GEMM option, pg2 off, no un-rotated resident W_std (`linearOn`'s order)
fn matmulPath(d: *const Dense, p: Proj, n: usize) bool {
    if (d.pathFor(n) != .matmul) return false;
    if (t3.pgemm.mode.get().* != .off) return false;
    return !(n > t3.dense.unrot_min_rows and t3.dense.residentUnrot(p.trellis));
}

/// Where KDA_PROJ_PROMPT's launch takes a member's operands at `n` rows (`Dense.prefillOperands`): the matmul path's
/// rows, no fused-GEMM option, no un-rotated resident W_std (pg2 plays no part there: the grouped launch is its own)
fn operandsPath(d: *const Dense, p: Proj, n: usize) bool {
    if (d.pathFor(n) != .matmul or d.fused_prefill) return false;
    return !(n > t3.dense.unrot_min_rows and t3.dense.residentUnrot(p.trellis));
}

// ── the calls ──

/// `linear(x, p)` (src/ling_exl3.zig `linear`: every EXL3 `Linear` of the copied model) over the scope's kept W_hat of
/// `p` at a wide span: `linear`'s prepare, MLX's matmul against the W_hat and its finish (src/ling_shared_layer.zig
/// `linearOver`), y in x's dtype with x's shape, its last axis `p`'s outputs; the caller owns y. Null (no launch) where
/// the scope does not serve the call: no scope open, `p` not held, rows `linear` serves another way, pg2 on, an
/// un-rotated resident trellis, a shape or dtype `linear` refuses (its own error then).
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, p: Proj) !?mlx.mlx_array {
    if (!scope_open) return null;
    const h = find(p.trellis) orelse return null;
    const v = rowsOf(x) orelse return null;
    if (v.dtype != .bfloat16 and v.dtype != .float16 and v.dtype != .float32) return null;
    const d = t3.denseOf(p, v.in, v.dtype, .mul1) catch return null;
    if (!matmulPath(d, p, v.n)) return null;
    const w = try what(s, h, d);
    const y = try sl.linearOver(s, d, x, p, w, v.n);
    record(h);
    return y;
}

/// KDA_NORM_PREP's check before its launch (L5, kept: src/ling_kda_norm_prep.zig `oProj`):
/// whether the scope serves `linear` of `p` at `rows` rows of an input `in` wide in `dtype`, by `linear`'s own checks
/// (the scope open, `p` held, bf16, f16 or f32, the matmul path, pg2 off, no un-rotated resident trellis). The `Dense`
/// `linear` takes there (`t3.denseOf` at that width and dtype), which the caller's launch writes the prepared rows for;
/// null otherwise. Nothing is decoded or counted.
pub fn preparedDense(p: Proj, rows: usize, in: u32, dtype: mlx.mlx_dtype) ?*Dense {
    if (!scope_open) return null;
    _ = find(p.trellis) orelse return null;
    if (dtype != .bfloat16 and dtype != .float16 and dtype != .float32) return null;
    const d = t3.denseOf(p, in, dtype, .mul1) catch return null;
    if (!matmulPath(d, p, rows)) return null;
    return d;
}

/// `linear(x, p)` over the scope's kept W_hat of `p` from x's prepared rows xp (f16 [rows, in], natural order: what
/// `linear`'s prepare makes of x) on `d` (`preparedDense`'s, which the caller checked before its launch): MLX's matmul
/// against the W_hat and `linear`'s finish (`Dense.piecePrepared`, `Dense.finishOf`), y [rows, out] in d's out dtype,
/// reshaped to `shape` (x's leading axes, then `p`'s outputs). These are `linearOver`'s launches less its prepare, so
/// with xp that prepare's words y is `linear`'s words. The call reads the kept W_hat (the scope's first call decodes it)
/// and counts as `linear`'s does. The caller owns y. error.Exl3EngineNotImplemented where `p` is not held.
pub fn linearPrepared(s: mlx.mlx_stream, d: *Dense, xp: mlx.mlx_array, p: Proj, rows: usize, shape: []const c_int) !mlx.mlx_array {
    const h = find(p.trellis) orelse return error.Exl3EngineNotImplemented;
    const w = try what(s, h, d);
    const z = try d.piecePrepared(s, xp, w, rows);
    defer _ = mlx.mlx_array_free(z);
    const y2 = try d.finishOf(s, z, p.svh, rows);
    defer _ = mlx.mlx_array_free(y2);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, shape.ptr, shape.len, s));
    record(h);
    return y;
}

/// KDA_PROJ_PROMPT's launch (src/ling_kdaproj.zig `projPrompt`: a KDA layer's q|k|v, f and g of x) over the scope's
/// kept W_hats of the three at a wide span: turboexl3's `ling.kdaProjPromptW`, each member's prepare and the grouped
/// launch, `ling.kdaProjPrompt`'s words; owned results, each [..., out_m]. Null (no launch) where the scope does not
/// serve the call: no scope open, a member not held, a call `kdaProjPrompt` does not take (its tile and `takes`, a
/// member's operands).
pub fn kdaProjPrompt(s: mlx.mlx_stream, x: mlx.mlx_array, p: [3]Proj) !?[3]mlx.mlx_array {
    if (!scope_open) return null;
    var hs: [3]*Held = undefined;
    for (p, &hs) |m, *h| h.* = find(m.trellis) orelse return null;
    const v = rowsOf(x) orelse return null;
    const pf = t3.ling.kdaproj_pf;
    const tile = pf.tileFor(v.in) orelse return null;
    if (!pf.takes(p, v.n, v.dtype, tile)) return null;
    var ds: [3]*Dense = undefined;
    for (p, &ds) |m, *d| {
        d.* = t3.denseOf(m, v.in, v.dtype, .mul1) catch return null;
        if (!operandsPath(d.*, m, v.n)) return null;
    }
    var w: [3]mlx.mlx_array = undefined;
    for (hs, ds, &w) |h, d, *wm| wm.* = try what(s, h, d);
    const ys = t3.ling.kdaProjPromptW(s, x, p, w, .mul1) catch |e| switch (e) {
        // its declines are the checks above: one here keeps the per-call launch, and nothing is counted
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    for (hs) |h| record(h);
    return ys;
}

/// KDA_GATE_EPI's launch (S52: src/ling_kdaproj.zig `projPromptGate`, a KDA layer's q|k|v, f and g of x with f's output
/// its bounded gate) over the scope's kept W_hats of the three at a wide span: turboexl3's `ling.kdaProjPromptWGate`,
/// each member's prepare and the grouped launch, `ling.kdaProjPromptGate`'s words; owned results (q|k|v, g f32 and z).
/// Null (no launch, nothing decoded or counted) where the scope does not serve the call (`kdaProjPrompt`'s checks) or
/// turboexl3 refuses the gate's operands (`gateTakes`).
pub fn kdaProjPromptGate(s: mlx.mlx_stream, x: mlx.mlx_array, p: [3]Proj, gate: t3.ling.kdaproj_pf.Gate) !?[3]mlx.mlx_array {
    if (!scope_open) return null;
    const pf = t3.ling.kdaproj_pf;
    if (!pf.gateTakes(p, gate)) return null;
    var hs: [3]*Held = undefined;
    for (p, &hs) |m, *h| h.* = find(m.trellis) orelse return null;
    const v = rowsOf(x) orelse return null;
    const tile = pf.tileFor(v.in) orelse return null;
    if (!pf.takes(p, v.n, v.dtype, tile)) return null;
    var ds: [3]*Dense = undefined;
    for (p, &ds) |m, *d| {
        d.* = t3.denseOf(m, v.in, v.dtype, .mul1) catch return null;
        if (!operandsPath(d.*, m, v.n)) return null;
    }
    var w: [3]mlx.mlx_array = undefined;
    for (hs, ds, &w) |h, d, *wm| wm.* = try what(s, h, d);
    const ys = t3.ling.kdaProjPromptWGate(s, x, p, w, gate, .mul1) catch |e| switch (e) {
        // its declines are the checks above: one here keeps the per-call launch, and nothing is counted
        error.Exl3EngineNotImplemented => return null,
        else => return e,
    };
    for (hs) |h| record(h);
    return ys;
}

/// SWIGLU_EPI's epilogue for a dense layer's up (src/ling_swiglu_epi.zig `up`, at its `.dense` site) over the scope's
/// kept W_hat of up at a wide span: turboexl3's `lingUpSwigluW`, the epilogue's words (h, x's dtype and shape with its
/// last axis up's outputs); the caller owns h. `gate` the caller's `linear` of the gate projection, `limit` 0 or the
/// clamp, `tab` the table at it. Null (no launch) where the scope does not serve the call: no scope open, up not held,
/// rows the matmul path does not take, pg2 on, an un-rotated resident trellis, a dtype the epilogue does not take.
pub fn upSwiglu(s: mlx.mlx_stream, x: mlx.mlx_array, up: Proj, gate: mlx.mlx_array, limit: u32, tab: mlx.mlx_array) !?mlx.mlx_array {
    if (!scope_open) return null;
    const h = find(up.trellis) orelse return null;
    const v = rowsOf(x) orelse return null;
    if (v.dtype != .bfloat16 and v.dtype != .float16) return null;
    const d = t3.denseOf(up, v.in, v.dtype, .mul1) catch return null;
    if (!matmulPath(d, up, v.n)) return null;
    const w = try what(s, h, d);
    const y = (try t3.lingUpSwigluW(s, x, up, w, gate, limit, tab, .mul1)) orelse return null;
    record(h);
    return y;
}
