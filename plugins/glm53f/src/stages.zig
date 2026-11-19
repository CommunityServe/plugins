//! #1196's KDA speed changes as the kit trunk's KDA stages (`trunk.KdaConsts.stages`, `kda_hybrid.kda.Stages`): each
//! stage serves where #1196's arm ran and its switch is on, and gives null (or `serves` false) everywhere else, so the
//! trunk's stock stage runs. The trunk asks a stage only on GLM's prompt rows (one slot, 17 rows or more, off the
//! per-core arm) or verify rows (one slot, captured), as #1196's `kda.zig` did.
//! - prologue (GLM53F_KDA_PROMPT_PROLOGUE): `kda_prologue.prologue`.
//! - recurrence (GLM53F_KDA_PROMPT_CHUNK, then GLM53F_KDA_PROMPT_PERCORE; both on by default, rounding-class).
//! - verify_prework, verify_gate (GLM53F_KDA_VERIFY; the gate's exp(A_log) in its launch with GLM53F_KDA_EXPSIG): the
//!   packed prework at MLX's sigmoid op (`decode_fused.gdnPreworkFused`, `.table`) with the capture's conv input, and
//!   `kda_verify.gate`.
//! - verify_capture (GLM53F_KDA_LAZY_SEQ): `kda_verify.captureLazy`.
//! - epilogue: the prompt rows' norm-gate (GLM53F_KDA_PROMPT_NORMGATE, `gdnNormGatePrompt`) and the verify rows'
//!   (GLM53F_KDA_VERIFY, `gdnNormGateChain`).

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = kit.log;
const kda = kit.bases.kda_hybrid.kda;
const trunk = kit.bases.kda_hybrid.trunk;
const decode_fused = kit.blocks.gdn.decode_fused;
const gdn_decode = kit.blocks.gdn.decode;
const options = @import("options.zig");
const kda_prologue = @import("kda_prologue.zig");
const kda_verify = @import("kda_verify.zig");
const kda_chunk = @import("kda_chunk.zig");
const kda_percore = @import("kda_percore.zig");

const Array = mlx.mlx_array;

/// The stages, every one set: each checks its own switch.
pub const all: kda.Stages = .{
    .prologue = &prologue,
    .recurrence = .{ .serves = &recurServes, .run = &recurRun },
    .verify_gate = &verifyGate,
    .verify_capture = &verifyCapture,
    .verify_prework = &verifyPrework,
    .epilogue = &epilogue,
};

/// The model's stages after its load, and the load's checks of the chunkwise route (`kda_chunk.loadCheck`) and the
/// per-core route (no plan or launch on this GPU at the pack's heads). Both routes are on by default: where one cannot
/// serve this GPU and the user did not set its switch, it is dropped with an info line and the KDA prompt takes the
/// next route; a switch the user set refuses the load by name.
pub fn install(m: *trunk.Model) !void {
    m.kda.stages = all;
    kda_chunk.loadCheck(m.s, m.c) catch |e| {
        if (options.explicit(.kda_prompt_chunk)) return e;
        log.info("glm53f: GLM53F_KDA_PROMPT_CHUNK (on by default) cannot serve this GPU ({s}); the KDA prompt takes the next route\n", .{@errorName(e)});
        options.drop(.kda_prompt_chunk);
    };
    if (options.on(.kda_prompt_percore)) {
        const c = m.c;
        if (!kda_percore.serves(m.s, @intCast(c.linear_num_key_heads), @intCast(c.linear_num_value_heads), @intCast(c.linear_key_head_dim), @intCast(c.linear_value_head_dim), .bfloat16, m.ssmStateDtype())) {
            if (options.explicit(.kda_prompt_percore)) {
                log.err("glm53f: GLM53F_KDA_PROMPT_PERCORE is set, and the per-core route has no plan or launch on this GPU at the pack's KDA heads\n", .{});
                return error.KdaPercoreNoPlan;
            }
            log.info("glm53f: GLM53F_KDA_PROMPT_PERCORE (on by default) has no plan or launch on this GPU at the pack's KDA heads; the KDA prompt takes the next route\n", .{});
            options.drop(.kda_prompt_percore);
        }
    }
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn prologue(s: mlx.mlx_stream, in: kda.PrologueIn) anyerror!?kda.PrologueOut {
    if (!options.on(.kda_prompt_prologue)) return null;
    const p = (try kda_prologue.prologue(s, in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.qk_eps, in.hk, in.dk, in.hv, in.dv, in.taps)) orelse return null;
    return .{ .q = p.q, .k = p.k, .v = p.v, .window = p.window };
}

/// The chunkwise route where its switch is on and it serves, else the per-core route where its switch is on and it
/// serves (#1196's order).
const Route = enum { chunk, percore };

fn routeOf(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype, a_log_h: Array, dt_f32: Array, lower_bound: f32) ?Route {
    if (options.on(.kda_prompt_chunk) and kda_chunk.serves(s, hk, hv, dk, dv, act, state, a_log_h, dt_f32, lower_bound)) return .chunk;
    if (options.on(.kda_prompt_percore) and a_log_h.ctx != null and dt_f32.ctx != null and kda_percore.serves(s, hk, hv, dk, dv, act, state)) return .percore;
    return null;
}

fn recurServes(s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype, a_log_h: Array, dt_f32: Array, lower_bound: f32) bool {
    return routeOf(s, hk, hv, dk, dv, act, state, a_log_h, dt_f32, lower_bound) != null;
}

fn recurRun(s: mlx.mlx_stream, in: kda.RecurIn) anyerror!kda.RecurOut {
    const qs = mlx.getShape(in.q);
    const vs = mlx.getShape(in.v);
    const route = routeOf(s, qs[2], vs[2], qs[3], vs[3], mlx.mlx_array_dtype(in.q), mlx.mlx_array_dtype(in.state), in.a_log_h, in.dt_f32, in.lower_bound) orelse return error.KdaStageRecurrenceNotServed;
    switch (route) {
        .chunk => {
            const r = try kda_chunk.recur(s, in.q, in.k, in.v, in.a, in.beta, in.a_log_h, in.dt_f32, in.lower_bound, in.state);
            return .{ .y = r.y, .state = r.state };
        },
        .percore => {
            const r = try kda_percore.recur(s, in.q, in.k, in.v, in.a, in.beta, in.a_log_h, in.dt_f32, in.lower_bound, in.state);
            return .{ .y = r.y, .state = r.state };
        },
    }
}

fn verifyGate(s: mlx.mlx_stream, in: kda.GateIn) anyerror!?Array {
    if (!options.on(.kda_verify)) return null;
    return kda_verify.gate(s, in.a, in.dt_bias, in.A_log, in.lower_bound, options.on(.kda_expsig));
}

fn verifyCapture(s: mlx.mlx_stream, in: kda.CaptureIn) anyerror!?kda.CaptureOut {
    if (!options.on(.kda_lazy_seq) or !mlx.streamIsGpu(s)) return null;
    const r = try kda_verify.captureLazy(s, .{ in.q, in.k, in.v, in.g, in.beta, in.state, in.rows_scalar }, in.in_dtype, in.out_dtype, in.hk, in.hv, in.dk, in.dv, in.batch, in.rows);
    return .{ .y = r.y, .state = r.state, .state_seq = r.state_seq };
}

/// The packed prework at MLX's sigmoid op (a verify is held to the composed chain's bytes), its gate not made (KDA's
/// is per channel), and the capture's conv input as the chain makes it: #1196's verify prework.
fn verifyPrework(s: mlx.mlx_stream, in: kda.PreworkIn) anyerror!?kda.PreworkOut {
    if (!options.on(.kda_verify)) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    const conv_dim = 2 * in.hk * in.dk + in.hv * in.dv;
    var zero_conv: Array = .{ .ctx = null };
    defer free(zero_conv);
    const incoming = in.conv_state orelse blk: {
        zero_conv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&zero_conv, &[_]c_int{ 1, in.taps - 1, conv_dim }, 3, dt, s));
        break :blk zero_conv;
    };
    const pre = (decode_fused.gdnPreworkFused(s, .{
        .qkv = in.qkv,
        .qkv_off = 0,
        .qkv_stride = conv_dim,
        .b = in.b,
        .b_off = 0,
        .b_stride = in.hv,
        // the per-head gate is not made (`gate` false): its operands are bound, never read
        .a = in.b,
        .a_off = 0,
        .a_stride = in.hv,
        .A_log = in.b,
        .dt_bias = in.b,
        .conv_state = incoming,
        .conv_w = in.conv_w,
        .q_scale = in.q_scale,
        .k_scale = in.k_scale,
        .qk_eps = in.qk_eps,
        .hk = in.hk,
        .hv = in.hv,
        .dk = in.dk,
        .dv = in.dv,
        .seq = in.rows,
        .batch = 1,
        .gate = false,
        .g_dtype = gdn_decode.gateDtype(dt, .float32),
        .table = true,
    }) catch null) orelse return null;
    free(pre.g);
    errdefer inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.beta }) |a| free(a);
    const conv_input = try kda_verify.convInput(s, incoming, in.qkv);
    kda_verify.runs.prework += 1;
    return .{ .q = pre.q, .k = pre.k, .v = pre.v, .beta = pre.beta, .window = pre.conv_state, .conv_input = conv_input };
}

var verify_said = false;
var prompt_normgate_said = false;

fn epilogue(s: mlx.mlx_stream, in: kda.EpilogueIn) anyerror!?Array {
    const value_dim = in.hv * in.dv;
    if (in.verify) {
        if (!options.on(.kda_verify)) return null;
        const flat = try decode_fused.gdnNormGateChain(s, in.y, in.z, 0, value_dim, in.norm_w, in.eps, false, in.hv, in.dv, in.batch, in.rows);
        if (flat != null) kda_verify.runs.normgate += 1;
        if (!verify_said) {
            verify_said = true;
            log.info("[kda] verify engaged: S={d} Hv={d} prework={d} gate={d} norm-gate={}\n", .{ in.rows, in.hv, kda_verify.runs.prework, kda_verify.runs.gate, flat != null });
        }
        return flat;
    }
    if (!options.on(.kda_prompt_normgate)) return null;
    const flat = (try decode_fused.gdnNormGatePrompt(s, in.y, in.z, 0, value_dim, in.norm_w, in.eps, false, in.hv, in.dv, in.rows)) orelse return null;
    if (!prompt_normgate_said) {
        prompt_normgate_said = true;
        log.info("[kda] prompt norm-gate engaged: S={d} Hv={d}\n", .{ in.rows, in.hv });
    }
    return flat;
}
