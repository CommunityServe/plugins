//! The device parity of the KDA prompt texts (src/kda_prompt.zig), run by the build step `kda-gpu-test` (linked to the
//! server's MLX) under TURBOEXL3_GLM53F_KDA_GPU=1; `zig build test` compiles it and its filter leaves it out.
//!
//! kda.zig's own forward (patches/turboexl3-model.patch) over GLM-5.3-Flash's KDA geometry (64 heads of 128, 4
//! conv taps, the low-rank gates, a hidden of 4096, A_log per head) with plain bf16 projections of seeded random
//! weights: a cold chunk of T rows, then a warm chunk of 31 rows on the state it leaves (the conv window and the f32
//! state carried), once with every switch off (the op chain, kdaGateChain and the stock recurrence), once with the
//! per-core route as served (`percore_base`: with the prologue text), once with the chunkwise route with the prologue
//! text (`chunk_base`) and once for each switch set. Each chunk's output, conv window and state is compared with its
//! reference run's (`vs`): the exact texts bit for bit (`KDA_BYTES`), each alone and all together against the off run,
//! the norm-gate text beside the per-core route against `percore_base`, the norm-gate text and the per-core switch
//! beside the chunkwise route (which runs before it) against `chunk_base`, and the prologue folded into the chunkwise
//! route (TURBOEXL3_GLM53F_KDA_FOLD: one launch for `chunk_base`'s two) alone, as served and before the norm-gate text
//! against `chunk_base`; the per-core route by its largest differences from the off run, and the chunkwise route by its
//! largest differences from the off run and from `percore_base` (`KDA_DIFF`: rounding-class, so no bitwise gate; FAIL
//! on a nonfinite value or a largest difference past `percore_rel_bound` of the reference's largest magnitude). One
//! line each, and one a set for the routes that ran (PASS when each one the set turns on ran in both chunks and no
//! other did; else the run compared a route with itself):
//!
//!     TURBOEXL3_GLM53F_KDA_BYTES T=<rows> set=<set> vs=<reference> out=<output> dtype=<dtype>
//!         differing=<n>/<total> nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_KDA_DIFF T=<rows> set=<set> vs=<reference> out=<output> dtype=<dtype> differing=<n>/<total>
//!         max_abs=<x> ref_max=<x> rel=<max_abs / ref_max> nonfinite=<n> PASS|FAIL
//!     TURBOEXL3_GLM53F_KDA_ENGAGED T=<rows> set=<set> prologue=<n> normgate=<n> percore=<n> chunk=<n> fold=<n>
//!         PASS|FAIL
//!
//! the per-core plan's line, `TURBOEXL3_GLM53F_KDA_PERCORE_PLAN ...`, the chunkwise route's,
//! `TURBOEXL3_GLM53F_KDA_CHUNK_PLAN ...`, the fold's, `TURBOEXL3_GLM53F_KDA_FOLD_PLAN ...`, and the run's last line,
//! `TURBOEXL3_GLM53F_KDA_PARITY lines=<n> diff_lines=<n> engaged=<n> fails=<n> PASS|FAIL`. The verify section (below
//! `verify_outputs`) prints its own byte lines (`set=verify`) and `TURBOEXL3_GLM53F_KDA_VERIFY_ENGAGED ...` lines.
//! TURBOEXL3_GLM53F_KDA_LAZY_SEQ beside KDA_VERIFY prints the same verifies' byte lines against KDA_VERIFY's
//! (`set=verify_lazy vs=verify`),
//! `TURBOEXL3_GLM53F_KDA_LAZY_ENGAGED T=<rows> lazy=<n> seq_computed_unread=<n> PASS|FAIL` (the lazy branch in both
//! verifies, the per-row states not computed while only the rest was read), and for each partial accept `a` of a
//! verify (its per-row state `a`, sliced as the host's rollback slices it, then read by the second verify) the rolled
//! state and the second verify's output, state and per-row states against KDA_VERIFY's same run
//! (`set=verify_lazy_rollback<a> vs=verify_rollback<a>`). The launch trims
//! (src/kda_trim.zig: TURBOEXL3_GLM53F_KDA_CONVIN, TURBOEXL3_GLM53F_KDA_EXPSIG, and both beside KDA_LAZY_SEQ) print the
//! same verifies' byte lines against KDA_VERIFY's (`set=verify_convin vs=verify`, `set=verify_expsig vs=verify`,
//! `set=verify_trim_lazy vs=verify`), one `TURBOEXL3_GLM53F_KDA_TRIM_ENGAGED T=<rows> set=<set> convin=<n>
//! exp_uses=<n> bsig=<n> lazy=<n> seq_computed_unread=<n> prologue=<n> gate=<n> PASS|FAIL` line a width and set, and both
//! beside KDA_LAZY_SEQ from each partial accept's rolled-back state (`set=verify_trim_lazy_rollback<a>
//! vs=verify_rollback<a>`).
//! TURBOEXL3_GLM53F_KDA_PIPE's section (below `pipe_sets`) prints the side stream's line,
//! `TURBOEXL3_GLM53F_KDA_PIPE_PLAN ...`, its byte lines (`set=pipe_<set> vs=<set>`: the spans against the forward
//! unspanned) and one `TURBOEXL3_GLM53F_KDA_PIPE_ENGAGED T=<rows> set=<set> pipe=<n> spans=<n> prologue=<n> chunk=<n>
//! fold=<n> normgate=<n> PASS|FAIL` line a row count and set. Under turboexl3's trunk int8 at the q|k|v linear's shape
//! (TURBOEXL3_GLM53F_PF_INT8=3's `Config.pf_trunk_int8`) the same spans over the served set (`pipe_sets`' last) against
//! the forward unspanned on int8 (`set=q8pipe_<set> vs=q8_<set>`), and one
//! `TURBOEXL3_GLM53F_KDA_PIPE_INT8_ENGAGED T=<rows> set=<set> pipe=<n> spans=<n> int8_unspanned=<n> int8_spanned=<n>
//! PASS|FAIL` line a row count (the unspanned forward's one int8 call, one a span).
const std = @import("std");
const t3 = @import("turboexl3");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const trunk = @import("mlx_serve/plugins/glm5_next/trunk.zig");
const config = @import("mlx_serve/plugins/glm5_next/config.zig");
const kda = @import("mlx_serve/plugins/glm5_next/kda.zig");
const kda_prompt = @import("kda_prompt.zig");
const kda_verify = @import("kda_verify.zig");
const kda_pipe = @import("kda_pipe.zig");
const kda_lazy = @import("kda_lazy.zig");
const kda_trim = @import("kda_trim.zig");
const system = @import("mlx_serve/sdk/util/system.zig");
const Array = mlx.mlx_array;

/// GLM-5.3-Flash's KDA layer.
const hidden: c_int = 4096;
const heads: c_int = 64;
const head_dim: c_int = 128;
const gate_rank: c_int = 128;
const conv_taps: c_int = 4;
/// The rows of the cold chunk (a short prompt's, the warm-up's class, a 16K prompt's main call) and of the warm one.
pub const rows_tested = [_]c_int{ 17, 995, 16356 };
pub const tail_rows: c_int = 31;

const Set = struct { name: []const u8, prologue: bool = false, normgate: bool = false, percore: bool = false, chunk: bool = false, fold: bool = false };
/// Each exact text alone and both together: bit for bit against the off run.
pub const sets = [_]Set{
    .{ .name = "prologue", .prologue = true },
    .{ .name = "normgate", .normgate = true },
    .{ .name = "all", .prologue = true, .normgate = true },
};
/// The per-core route alone and with the prologue text (which feeds it the chain's q, k and v): its largest
/// differences from the off run.
pub const percore_sets = [_]Set{
    .{ .name = "percore", .percore = true },
    .{ .name = "percore_prologue", .percore = true, .prologue = true },
};
/// The per-core route as served (with the prologue text), and the exact texts after its recurrence on top of it: bit
/// for bit against it.
pub const percore_base = percore_sets[1];
pub const percore_exact_sets = [_]Set{
    .{ .name = "percore_prologue_normgate", .percore = true, .prologue = true, .normgate = true },
};
/// The chunkwise route (TURBOEXL3_GLM53F_KDA_CHUNK) alone and with the prologue text: its largest differences from the
/// off run and from the per-core route as served.
pub const chunk_sets = [_]Set{
    .{ .name = "chunk", .chunk = true },
    .{ .name = "chunk_prologue", .chunk = true, .prologue = true },
};
/// The chunkwise route with the prologue text, and the norm-gate text after its recurrence and the per-core switch
/// beside it (the chunkwise route runs before the per-core one): bit for bit against it.
pub const chunk_base = chunk_sets[1];
pub const chunk_exact_sets = [_]Set{
    .{ .name = "chunk_prologue_normgate", .chunk = true, .prologue = true, .normgate = true },
    .{ .name = "chunk_prologue_percore", .chunk = true, .prologue = true, .percore = true },
};
/// The prologue folded into the chunkwise route (TURBOEXL3_GLM53F_KDA_FOLD, which needs it) alone, as served beside
/// control step 14's prologue and per-core switches (the fold takes the prologue text's place), and with the norm-gate
/// text after it: bit for bit against the chunkwise route with the prologue text.
pub const fold_sets = [_]Set{
    .{ .name = "chunk_fold", .chunk = true, .fold = true },
    .{ .name = "chunk_prologue_percore_fold", .chunk = true, .prologue = true, .percore = true, .fold = true },
    .{ .name = "chunk_fold_normgate", .chunk = true, .fold = true, .normgate = true },
};
/// The reference runs a set's outputs are compared with.
const Ref = enum { off, percore, chunk };
/// TURBOEXL3_GLM53F_KDA_PIPE: kda.zig's prompt forward as 4,096-row spans, bit for bit against the same forward
/// unspanned, over an exllamav3 q|k|v linear (the spans take turboexl3's matmul path; a trellis of random codes at
/// `pipe_k` bits, the sign vectors +-1 and +-1/64): a cold chunk of a 16K prompt's main call (four spans, the last
/// 4,068 rows) and of 8,197 rows (two spans, the 5 rows past 8,192 in the last), then the warm 31-row chunk (unspanned)
/// on the window and state the spans leave; for each route the spans take beside control step 14's prologue and
/// per-core switches: the chunkwise route and the fold, each with and without the norm-gate text after it.
pub const pipe_rows = [_]c_int{ 8197, 16356 };
pub const pipe_sets = [_]Set{
    .{ .name = "chunk_prologue_percore", .chunk = true, .prologue = true, .percore = true },
    .{ .name = "chunk_prologue_percore_normgate", .chunk = true, .prologue = true, .percore = true, .normgate = true },
    .{ .name = "chunk_prologue_percore_fold", .chunk = true, .prologue = true, .percore = true, .fold = true },
    .{ .name = "chunk_prologue_percore_fold_normgate", .chunk = true, .prologue = true, .percore = true, .fold = true, .normgate = true },
};
/// The trellis rate of the spans' q|k|v linear, in bits a weight (the matmul path decodes every rate to f16 first).
const pipe_k: c_int = 4;
pub const outputs = [_][]const u8{ "y1", "conv1", "ssm1", "y2", "conv2", "ssm2" };
/// The verify texts (src/kda_verify.zig, TURBOEXL3_GLM53F_KDA_VERIFY): a cold prompt chunk of `verify_prompt_rows` on
/// the chain, then two verifies of each width in `kda_verify.verify_rows` in a row (the second from the first's window
/// and state), each with capture: its output, conv window, state, the capture's conv input and per-row states. Of the
/// per-row states (`seq`, [T, B, Hv, Dv, Dk]) the rows the recurrence writes are compared, the first T - 1: its last row
/// is the capture kernel's unwritten row (the final state is its own output), which no rollback reads (a partial accept
/// of `a` drafts reads row a < T - 1; a full accept keeps the final state), so it holds whatever the allocator's buffer
/// held. A 1-row verify writes none: its `seq` lines compare 0 elements.
pub const verify_prompt_rows: c_int = 17;
pub const verify_outputs = [_][]const u8{ "y1", "conv1", "ssm1", "input1", "seq1", "y2", "conv2", "ssm2", "input2", "seq2" };
/// The launch trims' verify sets (src/kda_trim.zig), each beside KDA_VERIFY and held to its verifies bit for bit:
/// TURBOEXL3_GLM53F_KDA_CONVIN, TURBOEXL3_GLM53F_KDA_EXPSIG, and both beside TURBOEXL3_GLM53F_KDA_LAZY_SEQ (with its
/// rollbacks, `set=verify_trim_lazy_rollback<a> vs=verify_rollback<a>`).
pub const Trim = struct { convin: bool = false, expsig: bool = false };
pub const TrimSet = struct { name: []const u8, trim: Trim, lazy: bool };
pub const trim_sets = [_]TrimSet{
    .{ .name = "verify_convin", .trim = .{ .convin = true }, .lazy = false },
    .{ .name = "verify_expsig", .trim = .{ .expsig = true }, .lazy = false },
    .{ .name = "verify_trim_lazy", .trim = .{ .convin = true, .expsig = true }, .lazy = true },
};

fn setTrim(t: Trim) void {
    kda_trim.convin_on = t.convin;
    kda_trim.expsig_on = t.expsig;
}

/// The byte lines a run prints, its difference lines and its engagement lines.
pub const byte_lines = rows_tested.len * (sets.len + percore_exact_sets.len + chunk_exact_sets.len + fold_sets.len) * outputs.len + (2 + trim_sets.len) * kda_verify.verify_rows.len * verify_outputs.len + 2 * rollback_lines + pipe_rows.len * pipe_sets.len * outputs.len + pipe_rows.len * outputs.len;
/// TURBOEXL3_GLM53F_KDA_LAZY_SEQ's rollback lines: each partial accept of each verify width (a < rows - 1), four outputs
/// each (the rolled state, the second verify's output, state and per-row states).
pub const rollback_outputs = [_][]const u8{ "rolled", "y2", "ssm2", "seq2" };
pub const rollback_lines = blk: {
    var n: usize = 0;
    for (kda_verify.verify_rows) |r| n += @as(usize, @intCast(@max(r - 1, 0))) * rollback_outputs.len;
    break :blk n;
};
pub const diff_lines = rows_tested.len * (percore_sets.len + 2 * chunk_sets.len) * outputs.len;
pub const engaged_lines = rows_tested.len * (sets.len + percore_sets.len + percore_exact_sets.len + chunk_sets.len + chunk_exact_sets.len + fold_sets.len) + (2 + trim_sets.len) * kda_verify.verify_rows.len + pipe_rows.len * pipe_sets.len + pipe_rows.len;
/// A rounding-class route's largest difference from its reference, over the reference's largest magnitude, past which
/// the line fails: 1e-2, far above a summation order's rounding (bf16 outputs round at 2^-8 of their own magnitude) and
/// far below a wrong gate, head or row (an error of the outputs' own size).
pub const percore_rel_bound: f64 = 1e-2;

fn setSwitches(set: Set) void {
    kda_prompt.prologue_on = set.prologue;
    kda_prompt.normgate_on = set.normgate;
    kda_prompt.percore_on = set.percore;
    kda_prompt.chunk_on = set.chunk;
    kda_prompt.fold_on = set.fold;
}

/// Seeded normal values of `dtype` at `scale`, evaluated.
fn normal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Seeded uniform f32 values in [lo, hi), cast to `dtype`, evaluated.
fn uniform(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, lo: f32, hi: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, u, dtype, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Seeded random trellis codes for an exllamav3 linear of `in_dim` x `out_dim` at `k` bits: u16 [in/16, out/16, 16 k].
fn trellisOf(s: mlx.mlx_stream, seed: *u64, in_dim: c_int, out_dim: c_int, k: c_int) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape = [_]c_int{ @divExact(in_dim, 16), @divExact(out_dim, 16), 16 * k };
    try mlx.check(mlx.mlx_random_bits(&out, &shape, shape.len, 2, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Seeded signs +-`scale` in f16 [n] (an exllamav3 linear's suh or svh), evaluated.
fn signsF16(s: mlx.mlx_stream, seed: *u64, n: c_int, scale: f32) !Array {
    const u = try uniform(s, seed, &.{n}, .float32, 0.0, 2.0);
    defer _ = mlx.mlx_array_free(u);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_floor(&b, u, s));
    const two = mlx.mlx_array_new_float(2.0 * scale);
    defer _ = mlx.mlx_array_free(two);
    const one_s = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(one_s);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_multiply(&b2, b, two, s));
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_subtract(&v, b2, one_s, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, v, .float16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn newRef(a: Array) Array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, a);
    return r;
}

const Outs = [outputs.len]Array;

fn freeOuts(o: *Outs) void {
    for (o) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
}

/// The two chunks through kda.forward under `set`: their outputs, conv windows and states, evaluated.
fn runSet(m: *trunk.Model, la: *const trunk.Kda, x1: Array, x2: Array, rows: c_int, set: Set) !Outs {
    setSwitches(set);
    var e: kda.Entry = .{};
    defer e.deinit();
    var out: Outs = @splat(.{ .ctx = null });
    errdefer freeOuts(&out);
    out[0] = try kda.forward(m, x1, la, &e, 1, rows, true, false);
    out[1] = newRef(e.conv_state);
    out[2] = newRef(e.ssm_state);
    try evalAll(out[0..3]);
    out[3] = try kda.forward(m, x2, la, &e, 1, tail_rows, true, false);
    out[4] = newRef(e.conv_state);
    out[5] = newRef(e.ssm_state);
    try evalAll(out[3..6]);
    return out;
}

const VOuts = [verify_outputs.len]Array;

/// The cold prompt chunk `x0` on the chain (every switch off), then two verifies (`xa`, `xb`, `rows` rows each, with
/// capture) with TURBOEXL3_GLM53F_KDA_VERIFY at `verify`: each verify's output, conv window, state, conv input and
/// per-row states, evaluated.
fn runVerify(m: *trunk.Model, la: *const trunk.Kda, x0: Array, xa: Array, xb: Array, rows: c_int, verify: bool, lazy: bool, trim: Trim) !VOuts {
    setSwitches(.{ .name = "off" });
    kda_verify.on = false;
    kda_lazy.on = false;
    setTrim(.{});
    defer {
        kda_verify.on = false;
        kda_lazy.on = false;
        setTrim(.{});
    }
    var e: kda.Entry = .{};
    defer e.deinit();
    var out: VOuts = @splat(.{ .ctx = null });
    errdefer for (&out) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    };
    const y0 = try kda.forward(m, x0, la, &e, 1, verify_prompt_rows, true, false);
    defer _ = mlx.mlx_array_free(y0);
    try evalAll(&.{ y0, e.conv_state, e.ssm_state });
    kda_verify.on = verify;
    kda_lazy.on = lazy;
    setTrim(trim);
    // a verify is a prefill-mode forward past one row (the server's is_prefill), a 1-row verify a decode one
    for ([_]Array{ xa, xb }, 0..) |x, i| {
        const o = out[5 * i ..][0..5];
        o[0] = try kda.forward(m, x, la, &e, 1, rows, rows > 1, true);
        o[1] = newRef(e.conv_state);
        o[2] = newRef(e.ssm_state);
        o[3] = newRef(e.spec_conv_input);
        o[4] = newRef(e.spec_state_seq);
        if (lazy) {
            // everything but the per-row states first, as a full accept reads it: the lazy branch's states unrun
            try evalAll(o[0..4]);
            var avail = false;
            try mlx.check(mlx._mlx_array_is_available(&avail, o[4]));
            if (avail) lazy_seq_early += 1;
        }
        try evalAll(o);
    }
    return out;
}

/// The lazy verifies whose per-row states were computed while only the rest was read (`runVerify`; 0 when lazy).
var lazy_seq_early: u64 = 0;

/// The cold prompt chunk `x0` on the chain, a verify of `xa` (`rows` rows, with capture, KDA_VERIFY on,
/// TURBOEXL3_GLM53F_KDA_LAZY_SEQ at `lazy`) with its output, window and state read (not its per-row states), the
/// rollback of a partial accept of `a` drafts (the per-row state `a` sliced [a, a + 1) and reshaped as the host's
/// rollbackSsmFromCapture takes it, the layer's state; the window kept), then the second verify `xb` from it: the rolled
/// state and the second verify's output, state and per-row states, evaluated.
fn runRollback(m: *trunk.Model, la: *const trunk.Kda, x0: Array, xa: Array, xb: Array, rows: c_int, a: c_int, lazy: bool, trim: Trim) ![rollback_outputs.len]Array {
    const s = m.s;
    setSwitches(.{ .name = "off" });
    kda_verify.on = false;
    kda_lazy.on = false;
    setTrim(.{});
    defer {
        kda_verify.on = false;
        kda_lazy.on = false;
        setTrim(.{});
    }
    var e: kda.Entry = .{};
    defer e.deinit();
    var out: [rollback_outputs.len]Array = @splat(.{ .ctx = null });
    errdefer for (&out) |*o| if (o.ctx != null) {
        _ = mlx.mlx_array_free(o.*);
        o.* = .{ .ctx = null };
    };
    const y0 = try kda.forward(m, x0, la, &e, 1, verify_prompt_rows, true, false);
    defer _ = mlx.mlx_array_free(y0);
    try evalAll(&.{ y0, e.conv_state, e.ssm_state });
    kda_verify.on = true;
    kda_lazy.on = lazy;
    setTrim(trim);
    const y1 = try kda.forward(m, xa, la, &e, 1, rows, rows > 1, true);
    defer _ = mlx.mlx_array_free(y1);
    try evalAll(&.{ y1, e.conv_state, e.ssm_state, e.spec_conv_input });
    const sh = mlx.getShape(e.spec_state_seq);
    if (sh.len != 5 or a < 0 or a + 1 >= sh[0]) return error.KdaGpuShape;
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, e.spec_state_seq, &[_]c_int{ a, 0, 0, 0, 0 }, 5, &[_]c_int{ a + 1, sh[1], sh[2], sh[3], sh[4] }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s));
    out[0] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out[0], row, &[_]c_int{ sh[1], sh[2], sh[3], sh[4] }, 4, s));
    _ = mlx.mlx_array_free(e.ssm_state);
    e.ssm_state = newRef(out[0]);
    out[1] = try kda.forward(m, xb, la, &e, 1, rows, rows > 1, true);
    out[2] = newRef(e.ssm_state);
    out[3] = newRef(e.spec_state_seq);
    try evalAll(&out);
    return out;
}

/// `diff` over the first `n` rows (axis 0) of each.
fn diffRows(s: mlx.mlx_stream, want: Array, got: Array, n: c_int) !Diff {
    const sh = mlx.getShape(want);
    if (sh.len < 1 or sh.len > 8 or n < 1 or n > sh[0]) return error.KdaGpuShape;
    const start: [8]c_int = @splat(0);
    var stop: [8]c_int = undefined;
    const stride: [8]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    stop[0] = n;
    const nd: usize = sh.len;
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_slice(&w, want, &start, nd, &stop, nd, &stride, nd, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_slice(&g, got, &start, nd, &stop, nd, &stride, nd, s));
    return diff(s, w, g);
}

fn evalAll(arrs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

const Diff = struct { differing: usize, total: usize, nonfinite: usize, max_abs: f64 = 0, ref_max: f64 = 0 };

fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The largest |got - want| and |want| over the elements, as f64.
fn track(d: *Diff, want: f32, got: f32) void {
    const w: f64 = want;
    const g: f64 = got;
    d.max_abs = @max(d.max_abs, @abs(g - w));
    d.ref_max = @max(d.ref_max, @abs(w));
}

/// `got` against `want` element by element, as bits (both made contiguous: copies of the same bits).
fn diff(s: mlx.mlx_stream, want: Array, got: Array) !Diff {
    const dt = mlx.mlx_array_dtype(want);
    if (dt != mlx.mlx_array_dtype(got)) return error.KdaGpuDtype;
    if (!std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.KdaGpuShape;
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got, false, s));
    try evalAll(&.{ wc, gc });
    const n = mlx.mlx_array_size(wc);
    var d: Diff = .{ .differing = 0, .total = n, .nonfinite = 0 };
    switch (dt) {
        .bfloat16 => {
            const pw = (mlx.mlx_array_data_bfloat16(wc) orelse return error.KdaGpuData)[0..n];
            const pg = (mlx.mlx_array_data_bfloat16(gc) orelse return error.KdaGpuData)[0..n];
            for (pw, pg) |a, b| {
                if (a != b) d.differing += 1;
                if ((b & 0x7F80) == 0x7F80) d.nonfinite += 1;
                track(&d, bf16Value(a), bf16Value(b));
            }
        },
        .float16 => {
            const pw = (mlx.mlx_array_data_float16(wc) orelse return error.KdaGpuData)[0..n];
            const pg = (mlx.mlx_array_data_float16(gc) orelse return error.KdaGpuData)[0..n];
            for (pw, pg) |a, b| {
                const ab: u16 = @bitCast(a);
                const bb: u16 = @bitCast(b);
                if (ab != bb) d.differing += 1;
                if ((bb & 0x7C00) == 0x7C00) d.nonfinite += 1;
                track(&d, a, b);
            }
        },
        .float32 => {
            const pw = (mlx.mlx_array_data_float32(wc) orelse return error.KdaGpuData)[0..n];
            const pg = (mlx.mlx_array_data_float32(gc) orelse return error.KdaGpuData)[0..n];
            for (pw, pg) |a, b| {
                const ab: u32 = @bitCast(a);
                const bb: u32 = @bitCast(b);
                if (ab != bb) d.differing += 1;
                if ((bb & 0x7F800000) == 0x7F800000) d.nonfinite += 1;
                track(&d, a, b);
            }
        },
        else => return error.KdaGpuDtype,
    }
    return d;
}

/// A_log per value head [h], expanded per key channel [h * 128] as trunk.zig loads it (head-major), evaluated.
fn expandHeads(s: mlx.mlx_stream, per_head: Array, h: c_int) !Array {
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, per_head, &[_]c_int{ h, 1 }, 2, s));
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ h, head_dim }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, bc, &[_]c_int{h * head_dim}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, flat, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

test "turboexl3_glm53f kda gpu: kda.zig's forward with each exact KDA prompt text on is the op chain's bits (beside the per-core or chunkwise route, that route's bits), and the per-core and chunkwise routes within their bound of them, a cold chunk of 17, 995 and 16,356 rows and then a warm 31-row chunk" {
    if (std.c.getenv("TURBOEXL3_GLM53F_KDA_GPU") == null) return error.SkipZigTest;
    defer setSwitches(.{ .name = "off" });
    // KDA_EXPSIG's kept exp(A_log) let go with the weights
    defer kda_trim.reset();
    const s = mlx.gpuStream();
    const cfg: config.Config = .{ .hidden_size = @intCast(hidden), .linear_num_key_heads = @intCast(heads), .linear_num_value_heads = @intCast(heads), .linear_key_head_dim = @intCast(head_dim), .linear_value_head_dim = @intCast(head_dim), .linear_conv_kernel_dim = @intCast(conv_taps), .kda_gate_lower_bound = -5.0 };
    var m: trunk.Model = .{ .c = &cfg, .s = s, .caps = .{}, .embed = .{}, .head = .{}, .final_norm = .{} };
    // the KDA constants kda.forward makes at its first call (Model.deinit would reach the streamed bank's code too)
    defer for ([_]Array{ m.kda.q_scale, m.kda.k_scale, m.kda.ones, m.kda.eps }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    var seed: u64 = 20261107;
    const dim = heads * head_dim;
    // The weights, plain bf16 [in, out] (pre-transposed, as the loader leaves a plain projection), at about unit gain.
    var owned: [12]Array = @splat(.{ .ctx = null });
    defer for (owned) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    owned[0] = try normal(s, &seed, &.{ hidden, 3 * dim }, .bfloat16, 1.0 / 64.0);
    owned[1] = try normal(s, &seed, &.{ hidden, gate_rank }, .bfloat16, 1.0 / 64.0);
    owned[2] = try normal(s, &seed, &.{ gate_rank, dim }, .bfloat16, 1.0 / 11.0);
    owned[3] = try normal(s, &seed, &.{ hidden, gate_rank }, .bfloat16, 1.0 / 64.0);
    owned[4] = try normal(s, &seed, &.{ gate_rank, dim }, .bfloat16, 1.0 / 11.0);
    owned[5] = try normal(s, &seed, &.{ hidden, heads }, .bfloat16, 1.0 / 64.0);
    owned[6] = try normal(s, &seed, &.{ dim, hidden }, .bfloat16, 1.0 / 90.0);
    owned[7] = try normal(s, &seed, &.{ 3 * dim, conv_taps, 1 }, .bfloat16, 0.5);
    // A_log per value head and expanded per key channel, and dt_bias, as the SC packs store and trunk.zig loads them
    // (f32), the gated norm's weight bf16
    owned[8] = try uniform(s, &seed, &.{heads}, .float32, 0.0, 2.5);
    owned[11] = try expandHeads(s, owned[8], heads);
    owned[9] = try normal(s, &seed, &.{dim}, .float32, 1.0);
    owned[10] = try uniform(s, &seed, &.{head_dim}, .bfloat16, 0.5, 1.5);
    const la: trunk.Kda = .{
        .qkv = .{ .w = owned[0] },
        .z = .{ .w = owned[1] },
        .z2 = .{ .w = owned[2] },
        .a = .{ .w = owned[3] },
        .a2 = .{ .w = owned[4] },
        .b = .{ .w = owned[5] },
        .out = .{ .w = owned[6] },
        .conv_w = owned[7],
        .A_log = owned[11],
        .dt_bias = owned[9],
        .norm_w = owned[10],
        .a_log_h = owned[8],
        .dt_f32 = owned[9],
    };
    var fails: usize = 0;
    // the per-core route's plan for this GPU's cores, and its probe launch, as the plugin's load makes them
    const cores = system.gpuCoreCount();
    if (kda_prompt.percoreInstall(s, heads, heads, head_dim, head_dim, cores)) |p| {
        std.debug.print("TURBOEXL3_GLM53F_KDA_PERCORE_PLAN cores={d} ntg={d} rows={d} per_core={d} PASS\n", .{ cores, p.ntg, p.rows, p.per_core });
    } else |e| {
        fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_KDA_PERCORE_PLAN cores={d} refused={s} FAIL\n", .{ cores, @errorName(e) });
    }
    // the chunkwise route's layout and bound checks and its probe launch, as the plugin's load makes them
    if (kda_prompt.chunkInstall(s, heads, heads, head_dim, head_dim, cfg.kda_gate_lower_bound)) |groups| {
        std.debug.print("TURBOEXL3_GLM53F_KDA_CHUNK_PLAN groups={d} value_rows={d} chunk_rows={d} PASS\n", .{ groups, kda_prompt.chunk_value_rows, kda_prompt.chunk_rows });
    } else |e| {
        fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_KDA_CHUNK_PLAN refused={s} FAIL\n", .{@errorName(e)});
    }
    // the fold's probe launch, as the plugin's load makes it (after the chunkwise route's install)
    if (kda_prompt.foldInstall(s, heads, conv_taps, cfg.kda_gate_lower_bound)) |_| {
        std.debug.print("TURBOEXL3_GLM53F_KDA_FOLD_PLAN groups={d} taps={d} PASS\n", .{ kda_prompt.chunkGroups(heads), conv_taps });
    } else |e| {
        fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_KDA_FOLD_PLAN refused={s} FAIL\n", .{@errorName(e)});
    }
    // the spans' side stream, as the plugin's load makes it
    if (kda_pipe.install(s)) |_| {
        std.debug.print("TURBOEXL3_GLM53F_KDA_PIPE_PLAN span_rows={d} min_last_rows={d} PASS\n", .{ kda_pipe.span_rows, kda_pipe.min_last_rows });
    } else |e| {
        fails += 1;
        std.debug.print("TURBOEXL3_GLM53F_KDA_PIPE_PLAN refused={s} FAIL\n", .{@errorName(e)});
    }
    defer kda_pipe.deinit();
    for (rows_tested) |rows| {
        const x1 = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x1);
        const x2 = try normal(s, &seed, &.{ 1, tail_rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x2);
        var want = try runSet(&m, &la, x1, x2, rows, .{ .name = "off" });
        defer freeOuts(&want);
        var want_percore = try runSet(&m, &la, x1, x2, rows, percore_base);
        defer freeOuts(&want_percore);
        var want_chunk = try runSet(&m, &la, x1, x2, rows, chunk_base);
        defer freeOuts(&want_chunk);
        // the exact sets bit for bit against the off run, the per-core sets by their largest differences from it, the
        // exact texts beside the per-core route bit for bit against that route, the chunkwise sets by their largest
        // differences from the off run and from the per-core route, and the texts beside the chunkwise route bit for
        // bit against that route
        inline for (.{
            .{ &sets, false, .{Ref.off} },
            .{ &percore_sets, true, .{Ref.off} },
            .{ &percore_exact_sets, false, .{Ref.percore} },
            .{ &chunk_sets, true, .{ Ref.off, Ref.percore } },
            .{ &chunk_exact_sets, false, .{Ref.chunk} },
            .{ &fold_sets, false, .{Ref.chunk} },
        }) |kind| {
            const rounding = kind[1];
            for (kind[0]) |set| {
                const before = [5]u64{ kda_prompt.prologue_runs, kda_prompt.normgate_runs, kda_prompt.percore_runs, kda_prompt.chunk_runs, kda_prompt.fold_runs };
                var got = try runSet(&m, &la, x1, x2, rows, set);
                defer freeOuts(&got);
                const ran = [5]u64{ kda_prompt.prologue_runs - before[0], kda_prompt.normgate_runs - before[1], kda_prompt.percore_runs - before[2], kda_prompt.chunk_runs - before[3], kda_prompt.fold_runs - before[4] };
                // each route the set turns on ran in both chunks, and no other ran (the chunkwise route runs before
                // the per-core one; the fold, on the chunkwise route, takes the prologue text's place and the chunkwise
                // launch's)
                const engaged = ran[0] == @as(u64, if (set.prologue and !set.fold) 2 else 0) and ran[1] == @as(u64, if (set.normgate) 2 else 0) and ran[2] == @as(u64, if (set.percore and !set.chunk) 2 else 0) and ran[3] == @as(u64, if (set.chunk and !set.fold) 2 else 0) and ran[4] == @as(u64, if (set.fold) 2 else 0);
                if (!engaged) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_ENGAGED T={d} set={s} prologue={d} normgate={d} percore={d} chunk={d} fold={d} {s}\n", .{ rows, set.name, ran[0], ran[1], ran[2], ran[3], ran[4], if (engaged) "PASS" else "FAIL" });
                inline for (kind[2]) |ref_kind| {
                    const ref: *const Outs = switch (ref_kind) {
                        .off => &want,
                        .percore => &want_percore,
                        .chunk => &want_chunk,
                    };
                    const ref_name = switch (ref_kind) {
                        .off => "off",
                        .percore => percore_base.name,
                        .chunk => chunk_base.name,
                    };
                    for (outputs, ref.*, got) |name, w, g| {
                        const d = try diff(s, w, g);
                        const dtype_name = @tagName(mlx.mlx_array_dtype(w));
                        if (rounding) {
                            const rel: f64 = if (d.ref_max > 0) d.max_abs / d.ref_max else d.max_abs;
                            const ok = d.total > 0 and d.nonfinite == 0 and std.math.isFinite(d.max_abs) and std.math.isFinite(d.ref_max) and rel <= percore_rel_bound;
                            if (!ok) fails += 1;
                            std.debug.print("TURBOEXL3_GLM53F_KDA_DIFF T={d} set={s} vs={s} out={s} dtype={s} differing={d}/{d} max_abs={e:.3} ref_max={e:.3} rel={e:.3} nonfinite={d} {s}\n", .{ rows, set.name, ref_name, name, dtype_name, d.differing, d.total, d.max_abs, d.ref_max, rel, d.nonfinite, if (ok) "PASS" else "FAIL" });
                        } else {
                            const ok = d.differing == 0 and d.total > 0;
                            if (!ok) fails += 1;
                            std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set={s} vs={s} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, set.name, ref_name, name, dtype_name, d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
                        }
                    }
                }
            }
        }
    }
    // the verify texts: the chain's verifies against TURBOEXL3_GLM53F_KDA_VERIFY's, bit for bit, after the same prompt
    {
        const x0 = try normal(s, &seed, &.{ 1, verify_prompt_rows, hidden }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(x0);
        for (kda_verify.verify_rows) |rows| {
            const xa = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(xa);
            const xb = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(xb);
            const before_off = [3]u64{ kda_verify.prologue_runs, kda_verify.gate_runs, kda_verify.normgate_runs };
            const want = try runVerify(&m, &la, x0, xa, xb, rows, false, false, .{});
            defer for (want) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
            const before = [3]u64{ kda_verify.prologue_runs, kda_verify.gate_runs, kda_verify.normgate_runs };
            const got = try runVerify(&m, &la, x0, xa, xb, rows, true, false, .{});
            defer for (got) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
            const ran = [3]u64{ kda_verify.prologue_runs - before[0], kda_verify.gate_runs - before[1], kda_verify.normgate_runs - before[2] };
            // the three texts ran in both verifies of the switched run, and in none of the chain's
            const chain_runs = (before[0] - before_off[0]) + (before[1] - before_off[1]) + (before[2] - before_off[2]);
            const engaged = ran[0] == 2 and ran[1] == 2 and ran[2] == 2 and chain_runs == 0;
            if (!engaged) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_KDA_VERIFY_ENGAGED T={d} prologue={d} gate={d} normgate={d} chain_runs={d} {s}\n", .{ rows, ran[0], ran[1], ran[2], chain_runs, if (engaged) "PASS" else "FAIL" });
            for (verify_outputs, want, got) |name, w, g| {
                const seq = std.mem.startsWith(u8, name, "seq");
                const d: Diff = if (!seq) try diff(s, w, g) else if (rows > 1) try diffRows(s, w, g, rows - 1) else .{ .differing = 0, .total = 0, .nonfinite = 0 };
                const ok = d.differing == 0 and (d.total > 0 or (seq and rows == 1));
                if (!ok) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=verify out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
            }
            // TURBOEXL3_GLM53F_KDA_LAZY_SEQ beside KDA_VERIFY: the lazy branch in both verifies, the per-row states not
            // computed while the rest was read, and every output KDA_VERIFY's bit for bit
            const before_lz = [2]u64{ kda_lazy.verifies, lazy_seq_early };
            const lazied = try runVerify(&m, &la, x0, xa, xb, rows, true, true, .{});
            defer for (lazied) |a| if (a.ctx != null) {
                _ = mlx.mlx_array_free(a);
            };
            const lran = [2]u64{ kda_lazy.verifies - before_lz[0], lazy_seq_early - before_lz[1] };
            const lengaged = lran[0] == 2 and lran[1] == 0;
            if (!lengaged) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_KDA_LAZY_ENGAGED T={d} lazy={d} seq_computed_unread={d} {s}\n", .{ rows, lran[0], lran[1], if (lengaged) "PASS" else "FAIL" });
            for (verify_outputs, got, lazied) |name, w, g| {
                const seq = std.mem.startsWith(u8, name, "seq");
                const d: Diff = if (!seq) try diff(s, w, g) else if (rows > 1) try diffRows(s, w, g, rows - 1) else .{ .differing = 0, .total = 0, .nonfinite = 0 };
                const ok = d.differing == 0 and (d.total > 0 or (seq and rows == 1));
                if (!ok) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=verify_lazy vs=verify out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
            }
            // the launch trims beside KDA_VERIFY (src/kda_trim.zig): KDA_CONVIN's prologue in both verifies (and
            // KDA_VERIFY's prologue count with it), KDA_EXPSIG's kept exp(A_log) read by both gates and its capture launch
            // in both, both beside KDA_LAZY_SEQ with its lazy states uncomputed while the rest was read; every output
            // KDA_VERIFY's bit for bit
            for (trim_sets) |ts| {
                const before_t = [7]u64{ kda_trim.convin_runs, kda_trim.ea_uses, kda_trim.bsig_runs, kda_lazy.verifies, lazy_seq_early, kda_verify.prologue_runs, kda_verify.gate_runs };
                const trimmed = try runVerify(&m, &la, x0, xa, xb, rows, true, ts.lazy, ts.trim);
                defer for (trimmed) |a| if (a.ctx != null) {
                    _ = mlx.mlx_array_free(a);
                };
                const tran = [7]u64{ kda_trim.convin_runs - before_t[0], kda_trim.ea_uses - before_t[1], kda_trim.bsig_runs - before_t[2], kda_lazy.verifies - before_t[3], lazy_seq_early - before_t[4], kda_verify.prologue_runs - before_t[5], kda_verify.gate_runs - before_t[6] };
                const want_n = [2]u64{ 0, 2 };
                const tengaged = tran[0] == want_n[@intFromBool(ts.trim.convin)] and tran[1] == want_n[@intFromBool(ts.trim.expsig)] and tran[2] == want_n[@intFromBool(ts.trim.expsig)] and tran[3] == want_n[@intFromBool(ts.lazy)] and tran[4] == 0 and tran[5] == 2 and tran[6] == 2;
                if (!tengaged) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_TRIM_ENGAGED T={d} set={s} convin={d} exp_uses={d} bsig={d} lazy={d} seq_computed_unread={d} prologue={d} gate={d} {s}\n", .{ rows, ts.name, tran[0], tran[1], tran[2], tran[3], tran[4], tran[5], tran[6], if (tengaged) "PASS" else "FAIL" });
                for (verify_outputs, got, trimmed) |name, w, g| {
                    const seq = std.mem.startsWith(u8, name, "seq");
                    const d: Diff = if (!seq) try diff(s, w, g) else if (rows > 1) try diffRows(s, w, g, rows - 1) else .{ .differing = 0, .total = 0, .nonfinite = 0 };
                    const ok = d.differing == 0 and (d.total > 0 or (seq and rows == 1));
                    if (!ok) fails += 1;
                    std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set={s} vs=verify out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, ts.name, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
                }
            }
            // each partial accept's rollback: the rolled state and the verify after it, the lazy run against KDA_VERIFY's
            var acc: c_int = 0;
            while (acc + 1 < rows) : (acc += 1) {
                const rw = try runRollback(&m, &la, x0, xa, xb, rows, acc, false, .{});
                defer for (rw) |a| if (a.ctx != null) {
                    _ = mlx.mlx_array_free(a);
                };
                const rl = try runRollback(&m, &la, x0, xa, xb, rows, acc, true, .{});
                defer for (rl) |a| if (a.ctx != null) {
                    _ = mlx.mlx_array_free(a);
                };
                // the launch trims beside KDA_LAZY_SEQ: the lazy capture launch with beta's sigmoid made in it, run by
                // the rollback's read
                const rt = try runRollback(&m, &la, x0, xa, xb, rows, acc, true, trim_sets[2].trim);
                defer for (rt) |a| if (a.ctx != null) {
                    _ = mlx.mlx_array_free(a);
                };
                for (rollback_outputs, rw, rt) |name, w, g| {
                    const seq = std.mem.startsWith(u8, name, "seq");
                    const d: Diff = if (!seq) try diff(s, w, g) else try diffRows(s, w, g, rows - 1);
                    const ok = d.differing == 0 and d.total > 0;
                    if (!ok) fails += 1;
                    std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=verify_trim_lazy_rollback{d} vs=verify_rollback{d} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, acc, acc, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
                }
                for (rollback_outputs, rw, rl) |name, w, g| {
                    const seq = std.mem.startsWith(u8, name, "seq");
                    const d: Diff = if (!seq) try diff(s, w, g) else try diffRows(s, w, g, rows - 1);
                    const ok = d.differing == 0 and d.total > 0;
                    if (!ok) fails += 1;
                    std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=verify_lazy_rollback{d} vs=verify_rollback{d} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, acc, acc, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
                }
            }
        }
    }
    // TURBOEXL3_GLM53F_KDA_PIPE: the spans against the forward unspanned, over an exllamav3 q|k|v linear
    {
        const qkv_out = 3 * dim;
        const trellis = try trellisOf(s, &seed, hidden, qkv_out, pipe_k);
        defer _ = mlx.mlx_array_free(trellis);
        const suh = try signsF16(s, &seed, hidden, 1.0);
        defer _ = mlx.mlx_array_free(suh);
        const svh = try signsF16(s, &seed, qkv_out, 1.0 / 64.0);
        defer _ = mlx.mlx_array_free(svh);
        var la_x3 = la;
        la_x3.qkv = .{ .w = trellis, .x3 = .{ .trellis = trellis, .suh = suh, .svh = svh } };
        defer kda_pipe.on = false;
        for (pipe_rows) |rows| {
            const x1 = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x1);
            const x2 = try normal(s, &seed, &.{ 1, tail_rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x2);
            const spans: u64 = kda_pipe.planFor(rows).?.n;
            for (pipe_sets) |set| {
                kda_pipe.on = false;
                var want = try runSet(&m, &la_x3, x1, x2, rows, set);
                defer freeOuts(&want);
                kda_pipe.on = true;
                const before = [5]u64{ kda_pipe.runs, kda_prompt.prologue_runs, kda_prompt.chunk_runs, kda_prompt.fold_runs, kda_prompt.normgate_runs };
                var got = try runSet(&m, &la_x3, x1, x2, rows, set);
                defer freeOuts(&got);
                kda_pipe.on = false;
                const ran = [5]u64{ kda_pipe.runs - before[0], kda_prompt.prologue_runs - before[1], kda_prompt.chunk_runs - before[2], kda_prompt.fold_runs - before[3], kda_prompt.normgate_runs - before[4] };
                // the cold chunk spanned once; each span and the warm chunk on the set's route (the fold in the
                // prologue text's place and the chunkwise launch's), the norm-gate text once a chunk
                const route: u64 = spans + 1;
                const engaged = ran[0] == 1 and ran[1] == @as(u64, if (set.fold) 0 else route) and ran[2] == @as(u64, if (set.fold) 0 else route) and ran[3] == @as(u64, if (set.fold) route else 0) and ran[4] == @as(u64, if (set.normgate) 2 else 0);
                if (!engaged) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_PIPE_ENGAGED T={d} set={s} pipe={d} spans={d} prologue={d} chunk={d} fold={d} normgate={d} {s}\n", .{ rows, set.name, ran[0], spans, ran[1], ran[2], ran[3], ran[4], if (engaged) "PASS" else "FAIL" });
                for (outputs, want, got) |name, w, g| {
                    const d = try diff(s, w, g);
                    const ok = d.differing == 0 and d.total > 0;
                    if (!ok) fails += 1;
                    std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=pipe_{s} vs={s} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, set.name, set.name, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
                }
            }
        }
    }
    // TURBOEXL3_GLM53F_PF_INT8=3: the spans on turboexl3's trunk int8 (the q|k|v linear's shape named) against the
    // forward unspanned on it, bit for bit, over the served set
    {
        const qkv_out = 3 * dim;
        const trellis = try trellisOf(s, &seed, hidden, qkv_out, pipe_k);
        defer _ = mlx.mlx_array_free(trellis);
        const suh = try signsF16(s, &seed, hidden, 1.0);
        defer _ = mlx.mlx_array_free(suh);
        const svh = try signsF16(s, &seed, qkv_out, 1.0 / 64.0);
        defer _ = mlx.mlx_array_free(svh);
        var la_x3 = la;
        la_x3.qkv = .{ .w = trellis, .x3 = .{ .trellis = trellis, .suh = suh, .svh = svh } };
        const saved_cfg = t3.config.get().*;
        const saved_shapes = t3.trunk_q8.shapes.get().*;
        const q8_shapes = [1]t3.trunk_q8.Shape{.{ .k = @intCast(hidden), .n = @intCast(qkv_out) }};
        var q8_cfg = saved_cfg;
        q8_cfg.pf_trunk_int8 = true;
        try t3.configure(q8_cfg);
        t3.trunk_q8.shapes.get().* = @as([]const t3.trunk_q8.Shape, &q8_shapes);
        defer {
            t3.configure(saved_cfg) catch {};
            t3.trunk_q8.shapes.get().* = saved_shapes;
            kda_pipe.on = false;
        }
        const set = pipe_sets[pipe_sets.len - 1];
        for (pipe_rows) |rows| {
            const x1 = try normal(s, &seed, &.{ 1, rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x1);
            const x2 = try normal(s, &seed, &.{ 1, tail_rows, hidden }, .bfloat16, 1.0);
            defer _ = mlx.mlx_array_free(x2);
            const spans: u64 = kda_pipe.planFor(rows).?.n;
            kda_pipe.on = false;
            const c0 = t3.trunk_q8.calls.get().*.load(.monotonic);
            var want = try runSet(&m, &la_x3, x1, x2, rows, set);
            defer freeOuts(&want);
            const c1 = t3.trunk_q8.calls.get().*.load(.monotonic);
            kda_pipe.on = true;
            const r0 = kda_pipe.runs;
            var got = try runSet(&m, &la_x3, x1, x2, rows, set);
            defer freeOuts(&got);
            kda_pipe.on = false;
            const c2 = t3.trunk_q8.calls.get().*.load(.monotonic);
            // the unspanned forward's cold chunk one int8 call (the warm chunk's 31 rows take the register path), the
            // spanned one a span
            const engaged = kda_pipe.runs - r0 == 1 and c1 - c0 == 1 and c2 - c1 == spans;
            if (!engaged) fails += 1;
            std.debug.print("TURBOEXL3_GLM53F_KDA_PIPE_INT8_ENGAGED T={d} set={s} pipe={d} spans={d} int8_unspanned={d} int8_spanned={d} {s}\n", .{ rows, set.name, kda_pipe.runs - r0, spans, c1 - c0, c2 - c1, if (engaged) "PASS" else "FAIL" });
            for (outputs, want, got) |name, w, g| {
                const d = try diff(s, w, g);
                const ok = d.differing == 0 and d.total > 0;
                if (!ok) fails += 1;
                std.debug.print("TURBOEXL3_GLM53F_KDA_BYTES T={d} set=q8pipe_{s} vs=q8_{s} out={s} dtype={s} differing={d}/{d} nonfinite={d} {s}\n", .{ rows, set.name, set.name, name, @tagName(mlx.mlx_array_dtype(w)), d.differing, d.total, d.nonfinite, if (ok) "PASS" else "FAIL" });
            }
        }
    }
    std.debug.print("TURBOEXL3_GLM53F_KDA_PARITY lines={d} diff_lines={d} engaged={d} fails={d} {s}\n", .{ byte_lines, diff_lines, engaged_lines, fails, if (fails == 0) "PASS" else "FAIL" });
    try std.testing.expectEqual(@as(usize, 0), fails);
}
