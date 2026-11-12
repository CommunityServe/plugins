//! Producers that write a verify linear's prepared input themselves, so turboexl3's DEC-ONE runs its GEMV alone
//! (`t3.decPrepared`) where it runs prep_sc's launch, then its GEMV: one dependent launch fewer a call, the same bytes.
//! At 2 to 4 rows (a verify's) DEC-ONE prepares an input of a projection wider than 1024 outputs in prep_sc's launch:
//! xp, the input rounded to f16, times suh, through the 128-point Hadamard and its scale, f16 in lane order; xs, each
//! 128-block's sum of its stored halves; the GEMV's arrival counters zeroed. `t3.decPrepSpec` says what that call
//! takes (null at one row, where the prepare runs inside the GEMV launch, or where a variant reads f32 prepared rows:
//! these producers write f16). Each producer is a GLM-5.3-Flash text, the dims its constants (src/kernels/):
//!
//! - `TURBOEXL3_GLM53F_KDA_OPREP=1` (`kdaOut`, kernels/kda_oprep.metal): a KDA layer's gated norm in a verify (64 value
//!   heads of 128, the norm-gate text's statements) writes the out projection's (8192 -> 4096) prepared input: the
//!   gated norm's launch and prep_sc's become one; 34 a verify;
//! - `TURBOEXL3_GLM53F_DSA_OPREP=1` (`dsaOut`, kernels/dsa_oprep.metal): a DSA layer's out projection (16384 -> 4096)
//!   prepares its input from the attention's per-head values [1, 64, rows, 256] where they lie: the copy into row order
//!   and prep_sc's launch become one; 11 a verify.
//!
//! Each value is the stock op's: the producer's own statements, then prep_s's on the same values in the same lanes, so
//! xp and xs are prep_sc's bytes and the GEMV's output is the stock projection's (src/dec_prep_gpu_test.zig holds each
//! to the stock chain bit for bit at 2, 3 and 4 rows under the default and g1 GEMVs). Any other row count, shape or dtype
//! keeps the stock ops (null).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const t3 = @import("turboexl3");
const log = @import("mlx_serve/sdk/util/log.zig");
const fused_activations = @import("mlx_serve/sdk/blocks/mlp/fused_activations.zig");
const Array = mlx.mlx_array;

/// The switches, set by the plugin at load, before any forward.
pub var kda_on: bool = false;
pub var dsa_o_on: bool = false;
/// The switches' load lines (the A/B step reads "<switch>=1: ").
pub const kda_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_OPREP=1: a KDA layer's gated norm in a verify writes the out projection's prepared input (turboexl3's prep_sc statements after the norm-gate text's), and DEC-ONE runs its GEMV alone: one launch fewer a layer, the same bytes\n";
pub const dsa_o_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_OPREP=1: a DSA layer's out projection in a verify prepares its input from the per-head values where they lie (prep_sc's statements), and DEC-ONE runs its GEMV alone: the copy into row order and prep_sc's launch become one, the same bytes\n";

/// GLM-5.3-Flash's widths the texts are written for.
pub const kda_heads: c_int = 64;
pub const kda_dv: c_int = 128;
pub const kda_in: u32 = 8192;
pub const dsa_heads: c_int = 64;
pub const dsa_hd: c_int = 256;
pub const dsa_in: u32 = 16384;
/// The rows a verify takes (DEC-ONE's prepare runs inside its launch at one row; past 4 rows DEC-ONE does not take it).
pub const min_rows: c_int = 2;
pub const max_rows: c_int = 4;

/// Calls since the load, by producer; `declined`: calls a producer did not take at a verify's rows.
pub var kda_calls: u64 = 0;
pub var dsa_o_calls: u64 = 0;
pub var declined: u64 = 0;
var engaged_said: [2]bool = .{ false, false };

fn engaged(which: usize, what: []const u8, rows: c_int) void {
    if (engaged_said[which]) return;
    engaged_said[which] = true;
    log.info("[turboexl3_glm53f] {s} engaged: {d} rows\n", .{ what, rows });
}

// ── admission (MLX-free) ──

/// Whether a verify of `rows` rows is a producer's: 2 to 4.
pub fn rowsServe(rows: c_int) bool {
    return rows >= min_rows and rows <= max_rows;
}

/// Whether DEC-ONE's contract for a call is one the producers write: an f16 xp (a variant reading f32 declines), the
/// blocks the projection's input has.
pub fn specServes(sp: t3.dense.Dense.DecPrep, in: u32) bool {
    return !sp.xp_f32 and sp.nb == in / 128 and sp.ncnt >= 8;
}

// ── the kernels ──

const kda_source = @embedFile("kernels/kda_oprep.metal");
const dsa_o_source = @embedFile("kernels/dsa_oprep.metal");
var kda_kernel: ?mlx.mlx_fast_metal_kernel = null;
var dsa_o_kernel: ?mlx.mlx_fast_metal_kernel = null;

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

fn run(kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const Array, outs: []Array, s: mlx.mlx_stream) !void {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = x;
        got = i + 1;
    }
}

fn templ(cfg: mlx.mlx_fast_metal_kernel_config, names: []const [*:0]const u8, vals: []const c_int) !void {
    for (names, vals) |n, v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
}

/// A projection's prepared input as DEC-ONE's GEMV reads it: xp f16 [rows, in], xs f32 [rows, in / 128], cnt u32 zeroed.
pub const Prepared = struct {
    xp: Array,
    xs: Array,
    cnt: Array,
    pub fn deinit(p: Prepared) void {
        for ([_]Array{ p.xp, p.xs, p.cnt }) |a| _ = mlx.mlx_array_free(a);
    }
};

/// DEC-ONE's GEMV over `pr` for projection `p`: [lead..., out] for the stock call's leading dims `lead` (its rows).
pub fn project(s: mlx.mlx_stream, pr: Prepared, p: t3.Proj, lead: []const c_int) !Array {
    const y = try t3.decPrepared(s, pr.xp, pr.xs, pr.cnt, p, .mul1, .bfloat16);
    defer _ = mlx.mlx_array_free(y);
    var sh: [4]c_int = undefined;
    if (lead.len + 1 > sh.len) return error.BadExl3Shape;
    @memcpy(sh[0..lead.len], lead);
    sh[lead.len] = mlx.getShape(y)[1];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &sh, lead.len + 1, s));
    return out;
}

/// DEC-ONE's contract for `p` (in -> out) at `rows`, null where the producers do not serve it.
fn specOf(p: t3.Proj, in: u32, rows: c_int) !?t3.dense.Dense.DecPrep {
    const sp = (try t3.decPrepSpec(p, in, .bfloat16, .mul1, @intCast(rows))) orelse return null;
    return if (specServes(sp, in)) sp else null;
}

/// TURBOEXL3_GLM53F_KDA_OPREP: kda.zig's verify tail, `out(normGate(y, z))`, with the gated norm writing the out
/// projection's prepared input: y [B, T, 64, 128] (the recurrence's output), z [B, T, 8192] (the output gate's
/// projection), norm_w [128], all bf16. [B, T, 4096] as the stock tail gives it; null where it does not serve.
pub fn kdaOut(s: mlx.mlx_stream, y: Array, z: Array, norm_w: Array, eps: f32, hv: c_int, dv: c_int, out_proj: ?t3.Proj) !?Array {
    if (!kda_on or !mlx.streamIsGpu(s)) return null;
    const p = out_proj orelse return null;
    const ysh = mlx.getShape(y);
    const zsh = mlx.getShape(z);
    if (ysh.len != 4 or zsh.len != 3 or zsh[0] != ysh[0] or zsh[1] != ysh[1]) return null;
    const rows = ysh[0] * ysh[1];
    if (!rowsServe(rows)) return null;
    if (hv != kda_heads or dv != kda_dv or ysh[2] != hv or ysh[3] != dv or zsh[2] != hv * dv or mlx.mlx_array_size(norm_w) != @as(usize, @intCast(dv)) or
        mlx.mlx_array_dtype(y) != .bfloat16 or mlx.mlx_array_dtype(z) != .bfloat16 or mlx.mlx_array_dtype(norm_w) != .bfloat16)
    {
        declined += 1;
        return null;
    }
    const sp = (try specOf(p, kda_in, rows)) orelse {
        declined += 1;
        return null;
    };
    const table = try fused_activations.sigmoidTableFor(s, .bfloat16);
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const k = try kernelOf(&kda_kernel, "turboexl3_glm53f_kda_oprep", &.{ "y", "z", "norm_w", "eps", "sigtab", "suh" }, &.{ "xp", "xs", "cnt" }, kda_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, @intCast(kda_in) }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, @intCast(sp.nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(sp.ncnt)}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, rows, kda_heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try templ(cfg, &.{ "NCNT", "NTH" }, &.{ @intCast(sp.ncnt), 32 * rows * kda_heads });
    var o: [3]Array = undefined;
    try run(k, cfg, &.{ y, z, norm_w, eps_arr, table, p.suh }, &o, s);
    const pr: Prepared = .{ .xp = o[0], .xs = o[1], .cnt = o[2] };
    defer pr.deinit();
    const out = try project(s, pr, p, ysh[0..2]);
    kda_calls += 1;
    engaged(0, "TURBOEXL3_GLM53F_KDA_OPREP", rows);
    return out;
}

/// TURBOEXL3_GLM53F_DSA_OPREP: table_forward.zig's DSA tail at a verify's rows, `o_proj(reshape(transpose(o)))`, with
/// the out projection's input prepared from o [1, 64, rows, 256] (bf16, the attention's per-head values) where it lies:
/// [1, rows, 4096] as the stock tail gives it (the caller keeps its cache dependency on it); null where it does not
/// serve.
pub fn dsaOut(s: mlx.mlx_stream, o: Array, o_proj: ?t3.Proj) !?Array {
    if (!dsa_o_on or !mlx.streamIsGpu(s)) return null;
    const p = o_proj orelse return null;
    const sh = mlx.getShape(o);
    if (sh.len != 4 or sh[0] != 1) return null;
    const rows = sh[2];
    if (!rowsServe(rows)) return null;
    if (sh[1] != dsa_heads or sh[3] != dsa_hd or mlx.mlx_array_dtype(o) != .bfloat16) {
        declined += 1;
        return null;
    }
    const sp = (try specOf(p, dsa_in, rows)) orelse {
        declined += 1;
        return null;
    };
    const k = try kernelOf(&dsa_o_kernel, "turboexl3_glm53f_dsa_oprep", &.{ "o", "suh" }, &.{ "xp", "xs", "cnt" }, dsa_o_source);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, @intCast(dsa_in) }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, @intCast(sp.nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(sp.ncnt)}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(sp.nb * 32), rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try templ(cfg, &.{ "ROWS", "NCNT", "NTH" }, &.{ rows, @intCast(sp.ncnt), @intCast(sp.nth) });
    var outs: [3]Array = undefined;
    try run(k, cfg, &.{ o, p.suh }, &outs, s);
    const pr: Prepared = .{ .xp = outs[0], .xs = outs[1], .cnt = outs[2] };
    defer pr.deinit();
    const out = try project(s, pr, p, &.{ 1, rows });
    dsa_o_calls += 1;
    engaged(1, "TURBOEXL3_GLM53F_DSA_OPREP", rows);
    return out;
}

/// The counts' line at a request's end.
pub fn summary(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_DEC_PREP kda_oprep={d} dsa_oprep={d} declined={d}\n", .{ kda_calls, dsa_o_calls, declined }) catch buf[0..0];
}

/// Whether any producer is on (the plugin's request line).
pub fn anyOn() bool {
    return kda_on or dsa_o_on;
}

const testing = std.testing;

test "turboexl3_glm53f stream dec_prep: the producers take a verify's 2 to 4 rows and DEC-ONE's f16 contract at their projections' blocks, decline one row, five rows, an f32 contract or other blocks; the request line counts each producer" {
    for ([_]c_int{ 2, 3, 4 }) |r| try testing.expect(rowsServe(r));
    for ([_]c_int{ 0, 1, 5, 16 }) |r| try testing.expect(!rowsServe(r));
    try testing.expect(specServes(.{ .xp_f32 = false, .nb = 64, .ncnt = 32, .nth = 64 * 32 * 3 }, kda_in));
    try testing.expect(!specServes(.{ .xp_f32 = true, .nb = 64, .ncnt = 32, .nth = 64 * 32 * 3 }, kda_in));
    try testing.expect(!specServes(.{ .xp_f32 = false, .nb = 128, .ncnt = 32, .nth = 128 * 32 * 3 }, kda_in));
    try testing.expect(specServes(.{ .xp_f32 = false, .nb = 128, .ncnt = 32, .nth = 128 * 32 * 2 }, dsa_in));
    const saved = .{ kda_calls, dsa_o_calls, declined };
    defer {
        kda_calls = saved[0];
        dsa_o_calls = saved[1];
        declined = saved[2];
    }
    kda_calls = 340;
    dsa_o_calls = 110;
    declined = 1;
    var buf: [200]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_DEC_PREP kda_oprep=340 dsa_oprep=110 declined=1\n", summary(&buf));
}
