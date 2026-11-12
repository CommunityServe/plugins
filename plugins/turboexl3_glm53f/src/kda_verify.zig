//! `TURBOEXL3_GLM53F_KDA_VERIFY=1` (opt-in): a KDA layer's verify forward runs the prompt pass's three exact texts
//! (src/kda_prompt.zig) where the op chain runs its ops a launch each. A verify is kda.zig's `capture` forward: the
//! trunk's forward of every decode round (an MTP or prompt-lookup verify, 1 + the drafted rows), which keeps every row's
//! state for a partial accept's rollback, so the fused decode route (`kda_fused`) never takes it and, on an exllamav3
//! pack (no row-joined projection), each KDA layer runs the chain:
//!
//! - the prologue (`kda_prompt.prologueText`): conv1d.run's concatenate, depthwise conv and SiLU, the q and k slices'
//!   rms_norms and scale multiplies, and v, about ten launches, as one; the conv window as the chain keeps it;
//! - the bounded gate (`kda_prompt.gateTextWith`): kdaGateChain's add, cast, multiply, sigmoid, multiply and exp as one;
//! - the gated norm (`kda_prompt.normGateText`): the rms_norm over each value head, the output gate's sigmoid and their
//!   product as one, straight to the out projection's input.
//!
//! The capture's conv input, which the rollback reads, is the chain's own MLX concatenate of [window | qkv]. The capture
//! recurrence (every row's state) stays the chain's. Each text runs the MLX kernels the chain runs for
//! its steps, in their order and types, so the served words are the chain's; src/kda_gpu_test.zig holds the layer's
//! outputs, window, state, conv input and per-row states to the chain bit for bit at verify widths.
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kda_prompt = @import("kda_prompt.zig");
/// TURBOEXL3_GLM53F_KDA_CONVIN and TURBOEXL3_GLM53F_KDA_EXPSIG: the prologue text writing the conv input, exp(A_log)
/// kept a layer.
const kda_trim = @import("kda_trim.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_KDA_VERIFY=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_VERIFY=1: a KDA layer's verify forward (every decode round's trunk forward, its capture kept) runs the prompt pass's exact prologue, gate and gated-norm texts (the chain's depthwise conv, SiLU, q and k rms_norms and scales, v, its bounded gate, and its rms_norm, sigmoid and multiply, each the MLX kernel the chain runs, in its order and types); the capture's conv input is the chain's concatenate\n";

/// Launches since the load (src/kda_gpu_test.zig reads that the texts ran in a verify).
pub var prologue_runs: u64 = 0;
pub var gate_runs: u64 = 0;
pub var normgate_runs: u64 = 0;
var engaged = false;

/// What the chain leaves a verify's recurrence and rollback: q and k normed and scaled and v (each [1, rows, heads, 128]
/// contiguous), the conv window the layer's state keeps, and the conv input [window | qkv] the capture keeps; each a
/// new array.
pub const Prologue = struct { q: Array, k: Array, v: Array, window: Array, input: Array };

/// The chain's prologue for a verify over `qkv` after `prev` (the layer's conv window; null: a cold layer, zeros as the
/// chain takes them): `kda_prompt.prologueText` and the capture's conv input. Null with the switch off or on a shape or
/// dtype the text does not take (the caller runs the chain).
pub fn prologue(s: mlx.mlx_stream, qkv: Array, prev: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int) !?Prologue {
    if (!on) return null;
    // TURBOEXL3_GLM53F_KDA_CONVIN: the prologue text writes the conv input too, the window its last rows
    // (src/kda_trim.zig); null (off, or a shape the text does not take) leaves the text and the concatenate below
    if (try kda_trim.prologueConvIn(s, qkv, prev, conv_w, q_scale, k_scale, eps, hk, dk, hv, dv, taps)) |t| {
        ran(qkv, hk, dk, taps);
        return .{ .q = t.q, .k = t.k, .v = t.v, .window = t.window, .input = t.input };
    }
    const p = (try kda_prompt.prologueText(s, qkv, prev, conv_w, q_scale, k_scale, eps, hk, dk, hv, dv, taps)) orelse return null;
    errdefer inline for (.{ p.q, p.k, p.v, p.window }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const input = try convInput(s, qkv, prev, taps);
    ran(qkv, hk, dk, taps);
    return .{ .q = p.q, .k = p.k, .v = p.v, .window = p.window, .input = input };
}

/// A verify's prologue ran: counted, and the switch's engage line once.
fn ran(qkv: Array, hk: c_int, dk: c_int, taps: c_int) void {
    prologue_runs += 1;
    if (!engaged) {
        engaged = true;
        const qs = mlx.getShape(qkv);
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_KDA_VERIFY engaged: a {d}-row verify over {d} channels ({d} heads of {d}, {d} taps), the prologue, gate and gated-norm texts\n", .{ qs[1], qs[2], hk, dk, taps });
    }
}

/// `kdaGateChain(a_log, a, dt_bias, bound)` in a verify: `kda_prompt.gateTextWith`'s f32 [1, rows, n]. Null with the switch
/// off or on a shape or dtype the text does not take (the caller runs the chain).
pub fn gate(s: mlx.mlx_stream, a: Array, dt_bias: Array, a_log: Array, bound: f32) !?Array {
    if (!on) return null;
    // TURBOEXL3_GLM53F_KDA_EXPSIG: exp(A_log) made once a layer and kept (src/kda_trim.zig); null makes it in the text's
    // call, as before
    const ea = try kda_trim.expALog(s, a_log);
    const o = (try kda_prompt.gateTextWith(s, a, dt_bias, a_log, ea, bound)) orelse return null;
    gate_runs += 1;
    return o.g;
}

/// kda.zig's gated norm in a verify: `kda_prompt.normGateText`'s out projection input. Null with the switch off or on a
/// shape or dtype the text does not take (the caller runs the chain).
pub fn normGate(s: mlx.mlx_stream, y: Array, z: Array, norm_w: Array, eps: f32, hv: c_int, dv: c_int) !?Array {
    if (!on) return null;
    const out = (try kda_prompt.normGateText(s, y, z, norm_w, eps, hv, dv)) orelse return null;
    normgate_runs += 1;
    return out;
}

/// conv1d.run's `input`: [prev | qkv] along the rows (MLX's concatenate, as the chain makes it), prev the window or, on
/// a cold layer, zeros [batch, taps - 1, channels] of qkv's dtype.
fn convInput(s: mlx.mlx_stream, qkv: Array, prev: ?Array, taps: c_int) !Array {
    var cold = Array{ .ctx = null };
    defer if (cold.ctx != null) {
        _ = mlx.mlx_array_free(cold);
    };
    const qs = mlx.getShape(qkv);
    const before = prev orelse blk: {
        cold = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&cold, &[_]c_int{ qs[0], taps - 1, qs[2] }, 3, mlx.mlx_array_dtype(qkv), s));
        break :blk cold;
    };
    const parts = [_]Array{ before, qkv };
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

// ── tests (MLX-free) ──

const testing = std.testing;

/// The verify widths: 1 + the drafted rows (mlx-serve's depth pin is 2, a prompt-lookup round drafts the pinned depth
/// too; up to 4 rows stay on the decode lane).
pub const verify_rows = [_]c_int{ 1, 2, 3, 4 };

test "turboexl3_glm53f stream kda verify: the three texts take GLM-5.3-Flash's verify rows (64 heads of 128, 4 taps, 64 x 128 gate channels, y of 64 value heads beside its gate input)" {
    for (verify_rows) |rows| {
        try testing.expectEqual(@as(?kda_prompt.ProShape, .{ .rows = rows, .cdim = 24576 }), kda_prompt.prologueServes(&.{ 1, rows, 24576 }, 64, 128, 64, 128, 4));
        try testing.expectEqual(@as(?kda_prompt.GateShape, .{ .rows = rows, .n = 8192 }), kda_prompt.gateServes(&.{ 1, rows, 8192 }, 8192, 8192));
        try testing.expect(kda_prompt.normGateServes(&.{ 1, rows, 64, 128 }, &.{ 1, rows, 8192 }, 64, 128));
    }
}

test "turboexl3_glm53f stream kda verify: the switch is off until the plugin sets it" {
    try testing.expect(!on);
}
