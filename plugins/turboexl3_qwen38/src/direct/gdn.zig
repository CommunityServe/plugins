//! The direct model's GDN layer (36 of the 48) on the K packs: the turboexl3 projections (GDN-OUT's z inner product on
//! the side stream), the joined a | b GEMM, the prework, the gated-delta recurrence (a prefill chunk's whole or in row
//! pieces beside the side stream's tails, decode, verify with its replay, a batched tick's), the sigmoid norm-gate and
//! out_proj. Today's `trunk/gdn{,_tail,_ab,_core}.zig` and `sdk/blocks/gdn` launchers, cut to what the K path runs.
//! Every stream here is a Metal GPU stream (the trunk's or the side stream) and NAX is pinned on M5, so the copies'
//! stream and NAX checks are gone.

const std = @import("std");
const turboexl3 = @import("turboexl3");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const abi = @import("../mlx_serve/abi/quant.zig");
const kernelSource = @import("../mlx_serve/mlx/kernels.zig").source;
const gpuArchitecture = @import("../mlx_serve/mlx/device.zig").gpuArchitecture;
const scalarOf = @import("../mlx_serve/mlx/array.zig").scalarOf;
const materializedOwnedCopy = @import("../mlx_serve/mlx/array.zig").materializedOwnedCopy;
const sigmoidTableFor = @import("../mlx_serve/sdk/blocks/mlp/fused_activations.zig").sigmoidTableFor;
const sliceQkvPart = @import("../mlx_serve/sdk/blocks/linear/fused_rows.zig").sliceQkvPart;
const hc_max_seq = @import("../mlx_serve/sdk/blocks/hc/hc_prefill.zig").max_seq;
const PREFILL_EVAL_MIN_SEQ = @import("../mlx_serve/engines/mlx/eval_cadence.zig").PREFILL_EVAL_MIN_SEQ;
const SideStream = @import("../mlx_serve/plugins/qwen4_exp/side_stream.zig").SideStream;
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const linear = @import("linear.zig");
const SSMCacheEntry = @import("../mlx_serve/engines/mlx/ssm/entry.zig").SSMCacheEntry;
const LinearAttnWeights = @import("../mlx_serve/sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const GdnAb = @import("../mlx_serve/sdk/blocks/gdn/gated_delta_net.zig").GdnAb;
// The copies' types (A moves them at the cut-over).
const decode_types = @import("../mlx_serve/sdk/blocks/gdn/decode.zig");
const Geometry = decode_types.Geometry;
const Inputs = decode_types.Inputs;
const Recur = decode_types.Recur;
const RecurSeqFoldReplay = decode_types.RecurSeqFoldReplay;
const GdnPrework = @import("../mlx_serve/sdk/blocks/gdn/decode_fused.zig").GdnPrework;
const GdnPreworkArgs = @import("../mlx_serve/sdk/blocks/gdn/decode_fused.zig").GdnPreworkArgs;
const RecurOut = @import("../mlx_serve/plugins/qwen4_exp/trunk/gdn_core.zig").RecurOut;
const tail_types = @import("../mlx_serve/plugins/qwen4_exp/trunk/gdn_tail.zig");
const OutProj = tail_types.OutProj;
const MlxGemm = tail_types.MlxGemm;
const Plan = tail_types.Plan;

/// The layer's output for `x` (the hc read's mixed input), its recurrent and conv state advanced in `ssm`.
pub fn layer(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, ssm: *SSMCacheEntry, layer_idx: usize, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    _ = layer_idx;
    // The K packs project q|k|v, z, a and b apart (a and b joined at load): a qkvz / ba pair or a row-joined group is
    // another pack's layout.
    if (la.combined_proj) return error.DirectModelGdnCombinedProjection;
    if (la.in.w.ctx != null) return error.DirectModelGdnRowJoinedProjection;
    const cfg = &self.config;
    const num_k_heads: c_int = @intCast(cfg.linear_num_key_heads);
    const num_v_heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dk: c_int = @intCast(cfg.linear_key_head_dim);
    const dv: c_int = @intCast(cfg.linear_value_head_dim);
    // Every fused arm takes conv kernel 4 and 128-wide heads.
    if (cfg.linear_conv_kernel_dim != 4 or dk != 128 or dv != 128) return error.DirectModelGdnGeometry;
    const key_dim: c_int = dk * num_k_heads;
    const value_dim: c_int = dv * num_v_heads;
    const conv_dim: c_int = key_dim * 2 + value_dim;

    var qkv: mlx.mlx_array = .{ .ctx = null };
    var z_proj: mlx.mlx_array = .{ .ctx = null };
    var a_proj: mlx.mlx_array = .{ .ctx = null };
    var b_proj: mlx.mlx_array = .{ .ctx = null };
    // [a | 0 | b] from one matmul: a_proj / b_proj are its views, the prework reads it in place.
    var ab_all: mlx.mlx_array = .{ .ctx = null };
    // GDN-OUT: in_proj_z stopped at its inner product, which the fused tail finishes; `z_proj` is the format's finish of
    // it, built lazily, so MLX runs it only if the chain's norm-gate reads it.
    var z_inner: mlx.mlx_array = .{ .ctx = null };
    defer inline for (.{ qkv, z_proj, a_proj, b_proj, ab_all, z_inner }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    // GDN-OUT on a prefill chunk (the packed prework's prefill rows) without a spec capture.
    const gdn_out = is_prefill and gdnPrefillFusedFor(seq_len, batch) and !self.spec_capture_ssm;
    if (if (gdn_out) try outProj(self, x, la, batch, seq_len) else null) |proj| {
        qkv = proj.qkv;
        z_proj = proj.z;
        z_inner = proj.z_inner;
    } else {
        qkv = try linear.qmatmul(self, x, la.qkv_w, la.qkv_s, la.qkv_b);
        z_proj = try linear.qmatmul(self, x, la.z_w, la.z_s, la.z_b);
    }
    try abParts(self, x, la, &ab_all, &a_proj, &b_proj);
    if (self.gdn_q_scale == null) {
        const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
        self.gdn_q_scale = try scalarOf(inv_scale, mlx.mlx_array_dtype(qkv), self.s);
        self.gdn_k_scale = try scalarOf(@sqrt(inv_scale), mlx.mlx_array_dtype(qkv), self.s);
    }
    const eps = epsOf(self);
    const geo: Geometry = .{ .hk = num_k_heads, .hv = num_v_heads, .dk = dk, .dv = dv };

    // Decode (one row, no capture): the prework and the recurrence in one launch (K1), then the fused norm-gate.
    if (batch == 1 and seq_len == 1 and !self.spec_capture_ssm and ssm.initialized and ssm.ssm_state.ctx != null) fast: {
        const r = (try decodeRecur(geo, rowInputs(self, la, ssm, qkv, z_proj, a_proj, b_proj, eps), self.s)) orelse break :fast;
        defer _ = mlx.mlx_array_free(r.y);
        const flat = (try normGateFused(self.s, r.y, z_proj, 0, value_dim, la.norm_w, eps, num_v_heads, dv, 1, 1)) orelse {
            _ = mlx.mlx_array_free(r.conv_state);
            _ = mlx.mlx_array_free(r.ssm_state);
            break :fast;
        };
        defer _ = mlx.mlx_array_free(flat);
        install(&ssm.conv_state, r.conv_state);
        install(&ssm.ssm_state, r.ssm_state);
        return linear.qmatmul(self, flat, la.out_w, la.out_s, la.out_b);
    }

    // Verify rows (spec capture, up to 16). Up to 8, the replay fold: one launch for the recurrence, the norm-gate and
    // the rollback's conv input. It keeps the round's input state and writes the rows' recurrence inputs, from which a
    // partial accept replays the kept rows (`ssmRollbackFromCapture`). Past 8, K1S: the recurrence with each row's state
    // captured, then the norm-gate and the rollback's conv input.
    if (batch == 1 and seq_len <= MAX_SEQ and self.spec_capture_ssm and ssm.initialized and ssm.ssm_state.ctx != null) fast: {
        if (try recurSeqFoldReplay(geo, seq_len, rowInputs(self, la, ssm, qkv, z_proj, a_proj, b_proj, eps), self.s)) |f| {
            defer _ = mlx.mlx_array_free(f.gated);
            // a second handle to the input state the fold reads: the replay's starting point
            var state_in = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&state_in, ssm.ssm_state));
            inline for (.{ &ssm.spec_state_seq, &ssm.spec_conv_input, &ssm.spec_state_in, &ssm.spec_prework.k, &ssm.spec_prework.v, &ssm.spec_prework.gb }) |h| if (h.ctx != null) {
                _ = mlx.mlx_array_free(h.*);
                h.* = .{ .ctx = null };
            };
            ssm.spec_conv_input = f.conv_input;
            ssm.spec_state_in = state_in;
            ssm.spec_prework = f.prework;
            ssm.spec_replay = true;
            install(&ssm.conv_state, f.conv_state);
            install(&ssm.ssm_state, f.ssm_state);
            return linear.qmatmul(self, f.gated, la.out_w, la.out_s, la.out_b);
        }
        const r = (try recurSeq(geo, seq_len, rowInputs(self, la, ssm, qkv, z_proj, a_proj, b_proj, eps), self.s)) orelse break :fast;
        defer _ = mlx.mlx_array_free(r.y);
        const flat = (try normGateFused(self.s, r.y, z_proj, 0, value_dim, la.norm_w, eps, num_v_heads, dv, 1, seq_len)) orelse {
            inline for (.{ r.conv_state, r.ssm_state, r.state_seq }) |a| _ = mlx.mlx_array_free(a);
            break :fast;
        };
        defer _ = mlx.mlx_array_free(flat);
        // the rollback slices the conv INPUT, as the composed path stashes it
        var conv_input = mlx.mlx_array_new();
        {
            const arr = [_]mlx.mlx_array{ ssm.conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s)) catch |e| {
                inline for (.{ conv_input, r.conv_state, r.ssm_state, r.state_seq }) |a| _ = mlx.mlx_array_free(a);
                return e;
            };
        }
        install(&ssm.spec_conv_input, conv_input);
        install(&ssm.spec_state_seq, r.state_seq);
        install(&ssm.conv_state, r.conv_state);
        install(&ssm.ssm_state, r.ssm_state);
        return linear.qmatmul(self, flat, la.out_w, la.out_s, la.out_b);
    }

    // The rest: prefill chunks, batched ticks (and their verifies), cold or short chunks.
    var q_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled);
    var k_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_scaled);
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    var fused_prework = false;
    var g_fused: ?mlx.mlx_array = null;
    var beta_fused: ?mlx.mlx_array = null;
    // Wide prework is token-local; a cold chunk supplies the same zero history as conv1dWithCache.
    const prefill_fused = is_prefill and gdnPrefillFusedFor(seq_len, batch);
    if ((batch * seq_len <= GDN_FUSED_MAX_ROWS or prefill_fused) and (ssm.initialized or prefill_fused)) blk: {
        var zero_conv = mlx.mlx_array{ .ctx = null };
        defer if (zero_conv.ctx != null) {
            _ = mlx.mlx_array_free(zero_conv);
        };
        const cold = !ssm.initialized;
        if (cold) {
            zero_conv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero_conv, &[_]c_int{ batch, 3, conv_dim }, 3, mlx.mlx_array_dtype(qkv), self.s));
        }
        const incoming_conv = if (cold) zero_conv else ssm.conv_state;
        // a and b read in place from the joined product (columns 0 and b_off of rows n wide), not its views
        const ab_in = ab_all.ctx != null;
        const pw_args: GdnPreworkArgs = .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = if (ab_in) ab_all else b_proj,
            .b_off = if (ab_in) la.ab.b_off else 0,
            .b_stride = if (ab_in) la.ab.n else num_v_heads,
            .a = if (ab_in) ab_all else a_proj,
            .a_off = 0,
            .a_stride = if (ab_in) la.ab.n else num_v_heads,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .conv_state = incoming_conv,
            .conv_w = la.conv1d_w,
            .q_scale = self.gdn_q_scale.?,
            .k_scale = self.gdn_k_scale.?,
            .qk_eps = cfg.gdnQkNormEps(),
            .hk = num_k_heads,
            .hv = num_v_heads,
            .dk = dk,
            .dv = dv,
            .seq = seq_len,
            .batch = batch,
            .gate = true,
            .g_dtype = gateDtype(mlx.mlx_array_dtype(qkv), self.ssmStateDtype()),
        };
        // a prefill chunk's from the row stream (the GDN core's), a decode width's from the packed kernel
        const pre = (corePrework(self.s, pw_args) catch null) orelse ((preworkFused(self.s, pw_args) catch null) orelse break :blk);
        if (self.spec_capture_ssm) {
            // the rollback slices the conv INPUT: the concat the composed path stashes (lazy, one launch)
            const arr = [_]mlx.mlx_array{ incoming_conv, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&ssm.spec_conv_input, vec, 1, self.s)) catch {
                // the composed chain rather than state with no rollback capture
                inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta }) |a| _ = mlx.mlx_array_free(a);
                break :blk;
            };
        }
        install(&ssm.conv_state, pre.conv_state);
        ssm.initialized = true;
        install(&q_scaled, pre.q);
        install(&k_scaled, pre.k);
        install(&v_heads, pre.v);
        fused_prework = true;
        g_fused = pre.g;
        beta_fused = pre.beta;
    }

    if (!fused_prework) {
        // conv1d over the cached conv state, then q, k [B, S, key_dim] and v [B, S, value_dim] in head dims
        const conv_out = try conv1dWithCache(self, qkv, la.conv1d_w, ssm, batch, conv_dim);
        defer _ = mlx.mlx_array_free(conv_out);
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        try mlx.check(mlx.mlx_slice(&q_flat, conv_out, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, key_dim }, 3, &strides3, 3, self.s));
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        try mlx.check(mlx.mlx_slice(&k_flat, conv_out, &[_]c_int{ 0, 0, key_dim }, 3, &[_]c_int{ batch, seq_len, key_dim * 2 }, 3, &strides3, 3, self.s));
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &[_]c_int{ 0, 0, key_dim * 2 }, 3, &[_]c_int{ batch, seq_len, conv_dim }, 3, &strides3, 3, self.s));
        var q_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_heads);
        var k_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_heads);
        try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &[_]c_int{ batch, seq_len, num_k_heads, dk }, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k_heads, k_flat, &[_]c_int{ batch, seq_len, num_k_heads, dk }, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &[_]c_int{ batch, seq_len, num_v_heads, dv }, 4, self.s));
        // q = (1/dk) * rms_norm(q), k = (1/sqrt(dk)) * rms_norm(k); the weight-less norm's ones are cached on the Trunk
        if (self.gdn_ones_w == null) {
            var w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&w, &[_]c_int{dk}, 1, mlx.mlx_array_dtype(qkv), self.s));
            self.gdn_ones_w = w;
        }
        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, self.gdn_ones_w.?, cfg.gdnQkNormEps(), self.s));
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, self.gdn_q_scale.?, self.s));
        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, self.gdn_ones_w.?, cfg.gdnQkNormEps(), self.s));
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, self.gdn_k_scale.?, self.s));
    }

    // g = exp(-exp(A_log) * softplus(a + dt_bias)) [B, S, Hv] (the compiled closure), beta = sigmoid(b)
    const g = g_fused orelse try computeGdnGate(self, la.A_log, a_proj, la.dt_bias);
    defer if (g.ctx != null) {
        _ = mlx.mlx_array_free(g);
    };
    const beta = beta_fused orelse blk: {
        var bt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&bt, b_proj, self.s));
        break :blk bt;
    };
    defer _ = mlx.mlx_array_free(beta);
    // a slot whose first chunk the composed chain took
    if (ssm.ssm_state.ctx == null) {
        ssm.ssm_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &[_]c_int{ batch, num_v_heads, dv, dk }, 4, self.ssmStateDtype(), self.s));
    }

    // The recurrence: y in the activations' dtype, the state in its own.
    const gdn_out_dtype: mlx.mlx_dtype = if (mlx.mlx_array_dtype(q_scaled) == .float16) .float16 else .bfloat16;
    var y_bthd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_bthd);
    if (self.spec_capture_ssm) {
        // a batched tick's verify: the stock kernel, writing each row's state for the rollback
        try stockRecur(true, self.s, q_scaled, k_scaled, v_heads, g, beta, ssm, &y_bthd, geo, batch, seq_len, gdn_out_dtype);
    } else if (gdnBlockedEligible(seq_len, dk, dv, num_k_heads, num_v_heads)) {
        // Prefill widths: the GDN core recurrence, the pipelined route's bytes (qwen4_exp's route on M5). Over row
        // pieces where the overlap serves, each piece's GDN-OUT tail on the side stream beside the next piece.
        if (is_prefill and batch == 1 and seq_len >= 2 * MIN_PIECE and fused_prework and z_inner.ctx != null) {
            if (try overlap(self, la, ssm, z_inner, mlx.mlx_array_dtype(x), q_scaled, k_scaled, v_heads, g, beta, gdn_out_dtype, num_k_heads, num_v_heads, dk, dv, seq_len, eps)) |out| return out;
        }
        const r = try coreRecur(self.s, q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, num_k_heads, num_v_heads, dk, dv, gdn_out_dtype);
        install(&y_bthd, r.y);
        install(&ssm.ssm_state, r.state);
    } else try stockRecur(false, self.s, q_scaled, k_scaled, v_heads, g, beta, ssm, &y_bthd, geo, batch, seq_len, gdn_out_dtype);

    // GDN-OUT: z's finish, the norm-gate and out_proj's prepare in one launch, then out_proj from the prepared rows.
    if (z_inner.ctx != null) if (try outFused(self, y_bthd, z_inner, la, mlx.mlx_array_dtype(x), num_v_heads, dv, batch, seq_len, eps)) |out| return out;
    // the fused norm-gate at decode widths, else the chain's: rms_norm(y) * sigmoid(z)
    if (try normGateFused(self.s, y_bthd, z_proj, 0, value_dim, la.norm_w, eps, num_v_heads, dv, batch, seq_len)) |flat| {
        defer _ = mlx.mlx_array_free(flat);
        return linear.qmatmul(self, flat, la.out_w, la.out_s, la.out_b);
    }
    var z_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &[_]c_int{ batch, seq_len, num_v_heads, dv }, 4, self.s));
    const y_normed = try linear.rmsNorm(self, y_bthd, la.norm_w);
    defer _ = mlx.mlx_array_free(y_normed);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, self.s));
    var out_gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_gated);
    try mlx.check(mlx.mlx_multiply(&out_gated, y_normed, gate, self.s));
    var out_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_flat);
    try mlx.check(mlx.mlx_reshape(&out_flat, out_gated, &[_]c_int{ batch, seq_len, value_dim }, 3, self.s));
    return linear.qmatmul(self, out_flat, la.out_w, la.out_s, la.out_b);
}

/// The rms-norm eps (0-dim f32), cached on the Trunk.
fn epsOf(self: *Trunk) mlx.mlx_array {
    if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(self.config.rms_norm_eps);
    return self.gdn_eps.?;
}

/// `v` in place of what `dst` held.
fn install(dst: *mlx.mlx_array, v: mlx.mlx_array) void {
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = v;
}

/// The decode and verify kernels' inputs: this call's projections, the slot's states, the layer's weights.
fn rowInputs(self: *const Trunk, la: *const LinearAttnWeights, ssm: *const SSMCacheEntry, qkv: mlx.mlx_array, z: mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, eps: mlx.mlx_array) Inputs {
    return .{
        .qkv = qkv,
        .z = z,
        .a = a,
        .b = b,
        .conv_state = ssm.conv_state,
        .ssm_state = ssm.ssm_state,
        .conv_w = la.conv1d_w,
        .A_log = la.A_log,
        .dt_bias = la.dt_bias,
        .q_scale = self.gdn_q_scale.?,
        .k_scale = self.gdn_k_scale.?,
        .norm_w = la.norm_w,
        .eps = eps,
        .signs = .{ .ctx = null },
    };
}

// ── projections ──

/// in_proj_a and in_proj_b into `a_proj` / `b_proj`: column views of the one joined product where `abProject` serves
/// (`ab_all` takes it), else the two GEMMs.
fn abParts(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, ab_all: *mlx.mlx_array, a_proj: *mlx.mlx_array, b_proj: *mlx.mlx_array) !void {
    if (try abProject(x, la.ab, self.s)) |ab| {
        ab_all.* = ab;
        a_proj.* = mlx.mlx_array_new();
        b_proj.* = mlx.mlx_array_new();
        try sliceQkvPart(a_proj, ab, 0, la.ab.widths[0], self.s);
        try sliceQkvPart(b_proj, ab, la.ab.b_off, la.ab.widths[1], self.s);
    } else {
        a_proj.* = try linear.qmatmul(self, x, la.a_w, la.a_s, la.a_b);
        b_proj.* = try linear.qmatmul(self, x, la.b_w, la.b_s, la.b_b);
    }
}

/// x [B, S, hidden] times the joined [a | 0 | b] view in one MLX matmul, [B, S, n] (a at column 0, b at `ab.b_off`);
/// null where the pair keeps its two GEMMs: not joined at load, another dtype, or rows where MLX would not walk K alike
/// for both widths.
fn abProject(x: mlx.mlx_array, ab: GdnAb, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (ab.w.ctx == null) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3 or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(ab.w)) return null;
    if (!sameChains(xsh[0] * xsh[1], xsh[2], &ab.widths, ab.n)) return null;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, ab.w, s));
    return y;
}

/// MLX's NAX split-K pick for [m, k] x [k, n] (steel_matmul_axpby's Case 2), batch collapsed into m.
fn splitK(m: c_int, n: c_int, k: c_int) bool {
    const mx = @max(m, n);
    return k >= 3 * mx or (mx <= 1024 and k > 2 * mx);
}

/// Whether MLX walks K alike for the joined product [m, k] x [k, n] and each part's [m, k] x [k, w]: past the gemv
/// routes (m >= 16), the same split-K decision, and a partition K alone fixes when it splits.
fn sameChains(m: c_int, k: c_int, widths: []const c_int, n: c_int) bool {
    if (m < 16 or k < 1) return false;
    const split = splitK(m, n, k);
    if (split and k > 4096) return false;
    for (widths) |w| if (splitK(m, w, k) != split) return false;
    return true;
}

/// The cached conv state (zeros on a cold slot) before `x`, the depthwise conv1d and SiLU; the state advanced to the last
/// 3 rows.
fn conv1dWithCache(self: *Trunk, x: mlx.mlx_array, conv_w: mlx.mlx_array, ssm: *SSMCacheEntry, batch: c_int, cdim: c_int) !mlx.mlx_array {
    const kernel: c_int = 4;
    var conv_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_input);
    var zero_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero_state);
    if (!ssm.initialized) try mlx.check(mlx.mlx_zeros(&zero_state, &[_]c_int{ batch, kernel - 1, cdim }, 3, mlx.mlx_array_dtype(x), self.s));
    {
        const arr = [_]mlx.mlx_array{ if (ssm.initialized) ssm.conv_state else zero_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    }
    const ci_len = mlx.getShape(conv_input)[1];
    var new_conv_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &[_]c_int{ 0, ci_len - (kernel - 1), 0 }, 3, &[_]c_int{ batch, ci_len, cdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
    // An owned copy of a chunk-wide parent's tail, so the slice does not keep the whole chunk alive until the next forward.
    if (ci_len - (kernel - 1) >= PREFILL_EVAL_MIN_SEQ) {
        errdefer _ = mlx.mlx_array_free(new_conv_state);
        const owned = try materializedOwnedCopy(self.s, new_conv_state);
        _ = mlx.mlx_array_free(new_conv_state);
        new_conv_state = owned;
    }
    install(&ssm.conv_state, new_conv_state);
    ssm.initialized = true;
    // a spec capture keeps the conv input ([B, 3+T, conv_dim]): a partial accept slices its conv state from it
    if (self.spec_capture_ssm) {
        if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
        ssm.spec_conv_input = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&ssm.spec_conv_input, conv_input);
    }
    var conv_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_out);
    try mlx.check(mlx.mlx_conv1d(&conv_out, conv_input, conv_w, 1, 0, 1, cdim, self.s));
    return linear.silu(self, conv_out);
}

/// g in the state's dtype, shape of `a`, from the compiled gate closure (`Trunk.compileGdnGate`, built at load on Metal).
fn computeGdnGate(self: *const Trunk, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
    const compiled = self.compiled_gdn_gate orelse return error.DirectModelGdnGateNotCompiled;
    const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
    const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
    if (mlx.mlx_vector_array_size(out_vec) != 1) return error.DirectModelGdnGateOutputs;
    var g = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
    return g;
}

// ── GDN-OUT: the trunk format's parts (`abi.QuantTable`'s `linear_*` entries) ──

/// The trunk format and its weight of the map's `.weight` array `w`; null for a linear the trunk runs itself.
const Part = struct { t: *const abi.QuantTable, w: *const abi.Weight };

fn partOf(self: *const Trunk, w: mlx.mlx_array) ?Part {
    const t = self.formats.trunk orelse return null;
    return .{ .t = t, .w = self.formats.linearOf(w) orelse return null };
}

fn arrayOf(a: abi.Array) mlx.mlx_array {
    return .{ .ctx = a.ctx };
}

/// A part's result, or null where the format refuses the call (the chain then runs).
fn took(v: abi.Verdict, out: abi.Array) ?mlx.mlx_array {
    return if (v.isOk()) arrayOf(out) else null;
}

fn linearSplit(p: Part, x: mlx.mlx_array, side: ?mlx.mlx_stream, inner: bool, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_split orelse return null;
    const opts: abi.LinearParts = .{ .side = .{ .ctx = if (side) |r| r.ctx else null }, .inner = inner };
    var out: abi.Array = .{};
    return took(f(p.w, .{ .ctx = x.ctx }, &opts, .{ .ctx = s.ctx }, &out), out);
}

fn linearFinish(p: Part, parts: []const mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_finish orelse return null;
    var buf: [MAX_PIECES]abi.Array = undefined;
    for (parts, buf[0..parts.len]) |a, *o| o.* = .{ .ctx = a.ctx };
    const opts: abi.LinearParts = .{ .dtype = @intFromEnum(dt), .parts = .of(buf[0..parts.len]) };
    var out: abi.Array = .{};
    return took(f(p.w, &opts, .{ .ctx = s.ctx }, &out), out);
}

fn linearPrepared(p: Part, xp: mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_prepared orelse return null;
    const opts: abi.LinearParts = .{ .dtype = @intFromEnum(dt) };
    var out: abi.Array = .{};
    return took(f(p.w, .{ .ctx = xp.ctx }, &opts, .{ .ctx = s.ctx }, &out), out);
}

fn linearPieces(p: Part, rows: []const u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_pieces orelse return null;
    const opts: abi.LinearParts = .{ .dtype = @intFromEnum(dt), .rows = .of(rows) };
    var out: abi.Array = .{};
    return took(f(p.w, &opts, .{ .ctx = s.ctx }, &out), out);
}

fn linearPiece(p: Part, xp: mlx.mlx_array, held: mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?mlx.mlx_array {
    const f = p.t.linear_piece orelse return null;
    const opts: abi.LinearParts = .{ .dtype = @intFromEnum(dt), .prepared = true, .held = .{ .ctx = held.ctx } };
    var out: abi.Array = .{};
    return took(f(p.w, .{ .ctx = xp.ctx }, &opts, .{ .ctx = s.ctx }, &out), out);
}

/// GDN-OUT's projections: qkv, in_proj_z stopped at its inner product (the work after the prepare on the side stream's
/// `gdnz`), and the chain's z: the format's finish of that inner product, lazily. Null where the format refuses z's rows
/// (nothing launched).
fn outProj(self: *Trunk, x: mlx.mlx_array, la: *const LinearAttnWeights, batch: c_int, seq_len: c_int) !?OutProj {
    const zp = partOf(self, la.z_w) orelse return null;
    const side = SideStream.forPart(.gdnz, self, SideStream.rowsOf(x));
    const qkv = try linear.qmatmul(self, x, la.qkv_w, la.qkv_s, la.qkv_b);
    errdefer _ = mlx.mlx_array_free(qkv);
    const zi = linearSplit(zp, x, side, true, self.s) orelse {
        _ = mlx.mlx_array_free(qkv);
        return null;
    };
    errdefer _ = mlx.mlx_array_free(zi);
    const z2 = linearFinish(zp, &.{zi}, mlx.mlx_array_dtype(x), side orelse self.s) orelse {
        _ = mlx.mlx_array_free(zi);
        _ = mlx.mlx_array_free(qkv);
        return null;
    };
    defer _ = mlx.mlx_array_free(z2);
    var z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_reshape(&z, z2, &[_]c_int{ batch, seq_len, mlx.getShape(z2)[1] }, 3, self.s));
    return .{ .qkv = qkv, .z = z, .z_inner = zi };
}

/// GDN-OUT's tail: out_proj(norm-gate(y, z)) from z's inner product through the fused prepare (`outPrep`) and the
/// format's prepared-rows linear, [B, S, hidden] in the activations' dtype `act`. Null where either declines; the
/// prepare's output is then dropped unevaluated and the chain's norm-gate reads the lazy z.
fn outFused(self: *Trunk, y: mlx.mlx_array, zi: mlx.mlx_array, la: *const LinearAttnWeights, act: mlx.mlx_dtype, hv: c_int, dv: c_int, batch: c_int, seq_len: c_int, eps: mlx.mlx_array) !?mlx.mlx_array {
    const zp = partOf(self, la.z_w) orelse return null;
    const op = partOf(self, la.out_w) orelse return null;
    // the norm-gate reads y and z in one dtype, and z's finish rounds to the activations'
    if (mlx.mlx_array_dtype(y) != act) return null;
    const xp = (try outPrep(self.s, y, zi, arrayOf(zp.w.biases), la.norm_w, eps, arrayOf(op.w.scales), hv, dv, batch, seq_len)) orelse return null;
    defer _ = mlx.mlx_array_free(xp);
    const y2 = linearPrepared(op, xp, act, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(y2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y2, &[_]c_int{ batch, seq_len, mlx.getShape(y2)[1] }, 3, self.s));
    return out;
}

/// GDN-OUT's launch, turboexl3's `gdnOutPrep`: out_proj's prepared rows xp f16 [B*S, Hv*Dv] from the recurrence's y
/// [B, S, Hv, Dv], in_proj_z's inner product zq f16 [B*S, Hv*Dv] with its finish vector svh, the norm weight, eps and
/// out_proj's prepare vector suh, with the sigmoid gate. Prefill widths only; null where it declines.
fn outPrep(s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, svh: mlx.mlx_array, norm_w: mlx.mlx_array, eps: mlx.mlx_array, suh: mlx.mlx_array, hv: c_int, dv: c_int, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (dv != 128 or !gdnPrefillFusedFor(seq, batch)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(norm_w) != dt) return null;
    const ysh = mlx.getShape(y);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv) return null;
    const none: mlx.mlx_array = .{ .ctx = null };
    const z: turboexl3.Proj = .{ .trellis = none, .suh = none, .svh = svh };
    const o: turboexl3.Proj = .{ .trellis = none, .suh = suh, .svh = none };
    const norm: turboexl3.gdn_out.Norm = .{ .w = norm_w, .eps = eps, .sigtab = try sigmoidTableFor(s, dt) };
    return turboexl3.gdnOutPrep(s, y, zq, z, norm, o) catch |e| switch (e) {
        error.Exl3EngineNotImplemented, error.BadExl3Shape, error.BadExl3Dtype => return null,
        else => return e,
    };
}

// ── the overlap: a chunk's recurrence in row pieces, each piece's tail on the side stream ──

/// The smallest piece: out_proj's GEMM at 1024 rows still launches 16 M tiles x out/128 N tiles.
const MIN_PIECE: c_int = 1024;
/// The most pieces: the format's finish over pieces reads up to four.
const MAX_PIECES: usize = 4;
/// Piece boundaries fall on the NAX GEMM's M tile, so every row sits where it sits in the whole call's tiles.
const ALIGN: c_int = 128;

/// The kernel MLX's matmul runs for a row-contiguous f16 / bf16 [m, k] x [k, n] on a device of class `devc` (the last
/// letter of its architecture), with or without NAX: split K, else the NAX or steel kernel and its tile. Two row counts
/// with one key run one kernel, whose K loop gives every row the same summation order.
fn mlxGemm(m: c_int, n: c_int, k: c_int, nax: bool, devc: u8) MlxGemm {
    const big = devc == 's' or devc == 'c' or devc == 'd';
    if (nax) {
        if (splitK(m, n, k)) return .{ .split = true, .nax = true };
        if (!big) return .{ .nax = true, .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
        return .{ .nax = true, .bm = 64, .bn = 128, .bk = if (k >= 8192 and k > m + n) 64 else 256, .wm = 2, .wn = 4 };
    }
    // a small output over a long K splits K (Max and Ultra up to a larger output)
    const tm: i64 = @divTrunc(m + 15, 16);
    const tn: i64 = @divTrunc(n + 15, 16);
    const thr: i64 = if (devc == 's' or devc == 'd') 2048 else 1024;
    if (tm * tn <= thr and @divTrunc(k, 16) >= 8 and k >= @max(m, n)) return .{ .split = true };
    if (devc == 'g' or devc == 'p') return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    if (devc == 'd') {
        if (@as(i64, m) * n >= 1 << 20 and 2 * @max(m, n) <= k) return .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
        return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    }
    return .{ .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 };
}

/// Whether out_proj's GEMM on a piece of `m` rows runs the whole call's kernel (of `full` rows) with NAX and without, on
/// this device's class or, unknown (0), on every class; and neither splits K.
fn gemmSame(m: c_int, full: c_int, n: c_int, k: c_int, devc: u8) bool {
    const this_class = [_]u8{devc};
    const classes: []const u8 = if (devc != 0) &this_class else "gpsdm";
    for (classes) |c| for ([_]bool{ true, false }) |nax| {
        const a = mlxGemm(m, n, k, nax, c);
        const b = mlxGemm(full, n, k, nax, c);
        if (a.split or b.split or !std.meta.eql(a, b)) return false;
    };
    return true;
}

var devc_cached: ?u8 = null;

/// The device class letter MLX's GEMM routing reads: the architecture's last letter, 0 when unknown.
fn deviceClass() u8 {
    if (devc_cached) |c| return c;
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse "";
    const c: u8 = if (arch.len > 0) arch[arch.len - 1] else 0;
    devc_cached = c;
    return c;
}

/// The pieces of a `rows`-row chunk whose out_proj is [rows, k] x [k, n]: rows / MIN_PIECE of them, at most MAX_PIECES,
/// equal pieces on ALIGN-row boundaries with the remainder in the last; the most such pieces for which every piece's
/// GEMM runs the whole call's kernel. Null below two.
fn plan(rows: c_int, n: c_int, k: c_int, dc: u8) ?Plan {
    if (rows < 2 * MIN_PIECE) return null;
    var np: usize = @min(MAX_PIECES, @as(usize, @intCast(@divTrunc(rows, MIN_PIECE))));
    while (np >= 2) : (np -= 1) {
        const base = @divTrunc(@divTrunc(rows, @as(c_int, @intCast(np))), ALIGN) * ALIGN;
        var p = Plan{ .n = np, .start = @splat(rows) };
        var same = true;
        for (0..np) |i| {
            p.start[i] = @as(c_int, @intCast(i)) * base;
            const len = if (i + 1 < np) base else rows - p.start[i];
            same = same and gemmSame(len, rows, n, k, dc);
        }
        if (same) return p;
    }
    return null;
}

/// Rows [r0, r1) of a [1, S, ...] array along S (a view), recorded on `s`.
fn rowsView(dst: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(src);
    if (sh.len < 2 or sh.len > 4) return error.BadGdnOverlapShape;
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(0);
    const strides: [4]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[1] = r0;
    stop[1] = r1;
    try mlx.check(mlx.mlx_slice(dst, src, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
}

/// `arrs` submitted (no wait): each stream starts them once its earlier evals are done, and a later eval that reads one
/// on another stream waits for this eval's event there alone.
fn submit(arrs: []const mlx.mlx_array) !void {
    const ev = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_async_eval(ev));
}

/// A prefill chunk's recurrence and its GDN-OUT tail over row pieces (`plan`). Piece i's recurrence runs on this stream
/// from the state piece i - 1 left (the f32 state crosses a boundary unrounded, as at a chunk boundary); its fused
/// prepare and out_proj's inner product on the side stream (the last piece's here); one finish reads every piece's.
/// Submission: W alone as soon as it is built; each piece but the last as soon as it is built (one eval would encode the
/// whole recurrence chain before any tail); the last piece, its tail and the finish together.
/// The layer's output [1, S, hidden] with the final state installed in `ssm`, or null with `ssm` untouched: the caller
/// runs the whole chunk.
fn overlap(self: *Trunk, la: *const LinearAttnWeights, ssm: *SSMCacheEntry, z_inner: mlx.mlx_array, act: mlx.mlx_dtype, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, out_dtype: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, seq_len: c_int, eps: mlx.mlx_array) !?mlx.mlx_array {
    const op = partOf(self, la.out_w) orelse return null;
    const zp = partOf(self, la.z_w) orelse return null;
    const value_dim = hv * dv;
    // the pieces carry the state exactly only in f32, and each piece's tail is the one GDN-OUT gives the whole chunk
    const zsh = mlx.getShape(z_inner);
    if (mlx.mlx_array_dtype(ssm.ssm_state) != .float32 or out_dtype != act) return null;
    if (mlx.mlx_array_dtype(z_inner) != .float16 or zsh.len != 2 or zsh[0] != seq_len or zsh[1] != value_dim) return null;
    const osh = mlx.getShape(la.out_w);
    if (osh.len != 3 or osh[0] * 16 != value_dim) return null;
    const p = plan(seq_len, osh[1] * 16, value_dim, deviceClass()) orelse return null;
    // the tails' stream: the side stream (this half's, under the prefill wavefront)
    const side = SideStream.get(self.s) orelse return null;
    const n = p.n;
    // out_proj's W, made once on the side stream, ahead of in_proj_z's GEMM and the first tail
    var piece_rows: [MAX_PIECES]u32 = undefined;
    for (0..n) |i| piece_rows[i] = @intCast(p.start[i + 1] - p.start[i]);
    const W = linearPieces(op, piece_rows[0..n], out_dtype, side) orelse return null;
    defer _ = mlx.mlx_array_free(W);
    try submit(&[_]mlx.mlx_array{W});
    // per piece: the recurrence's y and state, the prepared rows, out_proj's inner product
    var ys: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var states: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var flats: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    var inners: [MAX_PIECES]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for ([_][]mlx.mlx_array{ &ys, &states, &flats, &inners }) |set| for (set) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (0..n) |i| {
        const r0 = p.start[i];
        const r1 = p.start[i + 1];
        // the piece's rows of the prework's outputs: views (one batch row, so each is row-contiguous)
        var rows: [5]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (rows) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_]mlx.mlx_array{ q, k, v, g, beta }, &rows) |src, *dst| try rowsView(dst, src, r0, r1, self.s);
        const r = try coreRecur(self.s, rows[0], rows[1], rows[2], rows[3], rows[4], if (i == 0) ssm.ssm_state else states[i - 1], hk, hv, dk, dv, out_dtype);
        ys[i] = r.y;
        states[i] = r.state;
        const ts = if (i + 1 < n) side else self.s;
        var zi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zi);
        try mlx.check(mlx.mlx_slice(&zi, z_inner, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r1, value_dim }, 2, &[_]c_int{ 1, 1 }, 2, ts));
        flats[i] = (try outPrep(ts, ys[i], zi, arrayOf(zp.w.biases), la.norm_w, eps, arrayOf(op.w.scales), hv, dv, 1, r1 - r0)) orelse return null;
        inners[i] = linearPiece(op, flats[i], W, out_dtype, ts) orelse return null;
        if (i + 1 < n) try submit(&[_]mlx.mlx_array{ inners[i], states[i] });
    }
    const y2 = linearFinish(op, inners[0..n], out_dtype, self.s) orelse return null;
    defer _ = mlx.mlx_array_free(y2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const y2sh = mlx.getShape(y2);
    try mlx.check(mlx.mlx_reshape(&out, y2, &[_]c_int{ 1, seq_len, y2sh[y2sh.len - 1] }, 3, self.s));
    // the last piece's state is the chunk's
    install(&ssm.ssm_state, states[n - 1]);
    states[n - 1] = .{ .ctx = null };
    try submit(&[_]mlx.mlx_array{ out, ssm.ssm_state });
    return out;
}

// ── kernels ──

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// One launch of `kernel` on `s` and its `n` outputs, which the caller owns.
fn launch(comptime n: usize, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, config: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) ![n]mlx.mlx_array {
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != n) return error.MetalKernelBadOutputCount;
    var out: [n]mlx.mlx_array = @splat(.{ .ctx = null });
    errdefer for (out) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, outputs_vec, i));
    }
    return out;
}

// ── the prework ──

/// Rows (batch*seq) the decode-width fusions serve.
const GDN_FUSED_MAX_ROWS: c_int = 16;

/// The prefill arm's rows: 17 <= S, one or two batch rows, within the hc prefill kernels' rows.
fn gdnPrefillFusedFor(seq: c_int, batch: c_int) bool {
    return !(seq < 17 or batch < 1 or batch > 2 or seq > @divTrunc(hc_max_seq, batch));
}

/// The dtype the kernels (GT) round the decay gate to: under bf16 activations the state's.
fn gateDtype(act: mlx.mlx_dtype, state: mlx.mlx_dtype) mlx.mlx_dtype {
    return if (act == .bfloat16) state else act;
}

const GDN_KERNEL_HEADER = kernelSource("kernels/gdn/log1p.metal");
const PREWORK_INPUTS = [_][*:0]const u8{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq", "qk_eps" };
const PREWORK_OUTPUTS = [_][*:0]const u8{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" };
const PreworkCfgKey = struct { hk: c_int, hv: c_int, seq: c_int, batch: c_int, c: c_int, qs: c_int, qo: c_int, bs: c_int, bo: c_int, as: c_int, ao: c_int, dt: mlx.mlx_dtype, gt: mlx.mlx_dtype };

/// What both prework kernels read in place: 128-wide heads, every input in the activations' bf16 / f16, conv_w [C, 4,
/// ..], qkv [B, S, stride] with its C channels at `qkv_off`, b and a with their Hv columns in rows `b_stride` /
/// `a_stride` wide, the conv state [B, 3, C]. The activations' dtype, or null.
fn preworkFits(in: GdnPreworkArgs, c_dim: c_int) ?mlx.mlx_dtype {
    if (in.dk != 128 or in.dv != 128 or !in.gate) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    inline for (.{ in.b, in.a, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.dt_bias }) |arr| {
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    }
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[1] != 4) return null;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != in.batch or qsh[1] != in.seq or qsh[2] != in.qkv_stride or in.qkv_off + c_dim > in.qkv_stride) return null;
    const bsh = mlx.getShape(in.b);
    if (bsh.len != 3 or bsh[0] != in.batch or bsh[1] != in.seq or bsh[2] != in.b_stride or in.b_off + in.hv > in.b_stride) return null;
    const ash = mlx.getShape(in.a);
    if (ash.len != 3 or ash[0] != in.batch or ash[1] != in.seq or ash[2] != in.a_stride or in.a_off + in.hv > in.a_stride) return null;
    const csh = mlx.getShape(in.conv_state);
    if (csh.len != 3 or csh[0] != in.batch or csh[1] != 3 or csh[2] != c_dim) return null;
    return dt;
}

/// The prework's six outputs: q, k [B, S, Hk, Dk], v [B, S, Hv, Dv], the next conv state [B, 3, C], g in `gt`, beta.
fn preworkOutputs(config: mlx.mlx_fast_metal_kernel_config, in: GdnPreworkArgs, c_dim: c_int, dt: mlx.mlx_dtype, gt: mlx.mlx_dtype) !void {
    const q_shape = [_]c_int{ in.batch, in.seq, in.hk, in.dk };
    const g_shape = [_]c_int{ in.batch, in.seq, in.hv };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ in.batch, in.seq, in.hv, in.dv }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ in.batch, 3, c_dim }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, gt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, dt));
}

fn preworkOf(o: [6]mlx.mlx_array) GdnPrework {
    return .{ .q = o[0], .k = o[1], .v = o[2], .conv_state = o[3], .g = o[4], .beta = o[5] };
}

// The packed GDN prework at decode and verify widths (S 1..9, B*S <= 16; batched ticks): one launch for the conv-state
// concat, the depthwise conv1d, SiLU, the q/k/v split, the q/k rms-norm-and-scale, the next conv state, the forget gate
// and beta, bit-identical to the composed chain.
const GDN_PREWORK_SOURCE = kernelSource("kernels/gdn/gdn_prework.metal");
var prework_kernel: ?mlx.mlx_fast_metal_kernel = null;
var prework_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var prework_cfg_key: PreworkCfgKey = std.mem.zeroes(PreworkCfgKey);

/// Null: the composed chain. The caller owns the six outputs (and installs `conv_state` into the slot's entry).
fn preworkFused(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (in.seq < 1 or in.seq > 9 or in.batch < 1 or in.batch * in.seq > GDN_FUSED_MAX_ROWS) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const dt = preworkFits(in, c_dim) orelse return null;
    const gt = in.g_dtype orelse dt;
    const key = PreworkCfgKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .dt = dt, .gt = gt };
    if (prework_cfg == null or !std.meta.eql(prework_cfg_key, key)) {
        if (prework_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        prework_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try preworkOutputs(config, in, c_dim, dt, gt);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, in.batch * in.seq, 2 * in.hk + in.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", gt));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", @as(c_int, 3) }, .{ "C", c_dim }, .{ "TAB", @as(c_int, 0) }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off }, .{ "GATE", @as(c_int, 1) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        prework_cfg = config;
        prework_cfg_key = key;
    }
    if (prework_kernel == null) prework_kernel = try makeKernel("mlxserve_gdn_prework2", &PREWORK_INPUTS, &PREWORK_OUTPUTS, GDN_PREWORK_SOURCE, GDN_KERNEL_HEADER);
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const eps_arr = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    // the sigmoid table's slot holds a placeholder at these widths (TAB 0)
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, in.qkv, seq_arr, eps_arr };
    return preworkOf(try launch(6, prework_kernel.?, &inputs_arr, prework_cfg.?, s));
}

// The GDN core prework (kernels/gdn_core_prework.metal) for prefill chunks: the packed prework's outputs and bytes, a
// run of rows per simdgroup where the packed prework runs a threadgroup per (row, head slot).
const CORE_PREWORK_SOURCE = kernelSource("plugins/qwen4_exp/kernels/gdn_core_prework.metal");
/// Rows a run (whole 4-row groups) and head slots a threadgroup.
const PW_ROWS: c_int = 4;
const PW_NSG: c_int = 8;
var core_pw_kernel: ?mlx.mlx_fast_metal_kernel = null;
var core_pw_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var core_pw_cfg_key: PreworkCfgKey = std.mem.zeroes(PreworkCfgKey);

/// The prefill arm's widths (17 <= S, B <= 2); null where it declines (nothing launched). The caller owns the outputs.
fn corePrework(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (!gdnPrefillFusedFor(in.seq, in.batch)) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const dt = preworkFits(in, c_dim) orelse return null;
    // 8-byte row reads and stores: qkv's row stride and column offset, and the channel count, on 4-element boundaries
    if (@rem(in.qkv_stride, 4) != 0 or @rem(in.qkv_off, 4) != 0 or @rem(c_dim, 4) != 0) return null;
    const gt = in.g_dtype orelse dt;
    const nslot: c_int = 2 * in.hk + in.hv;
    const key = PreworkCfgKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .dt = dt, .gt = gt };
    if (core_pw_cfg == null or !std.meta.eql(core_pw_cfg_key, key)) {
        if (core_pw_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        core_pw_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try preworkOutputs(config, in, c_dim, dt, gt);
        // NSG head slots (simdgroups) a threadgroup, one threadgroup row per run of R rows of a batch row
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * PW_NSG, in.batch * @divTrunc(in.seq + PW_ROWS - 1, PW_ROWS), @divTrunc(nslot + PW_NSG - 1, PW_NSG)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * PW_NSG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", gt));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", @as(c_int, 3) }, .{ "C", c_dim }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off }, .{ "GATE", @as(c_int, 1) }, .{ "R", PW_ROWS }, .{ "NSG", PW_NSG } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        core_pw_cfg = config;
        core_pw_cfg_key = key;
    }
    if (core_pw_kernel == null) core_pw_kernel = try makeKernel("msv_gdn_core_prework", &PREWORK_INPUTS, &PREWORK_OUTPUTS, CORE_PREWORK_SOURCE, GDN_KERNEL_HEADER);
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const eps_arr = mlx.mlx_array_new_float(in.qk_eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, try sigmoidTableFor(s, dt), seq_arr, eps_arr };
    return preworkOf(try launch(6, core_pw_kernel.?, &inputs_arr, core_pw_cfg.?, s));
}

// ── the recurrence ──

/// The pipelined route's rows and geometry: a prefill width (64 rows on), Dk 128, Dv a multiple of 32, whole GQA groups.
fn gdnBlockedEligible(seq_len: c_int, dk: c_int, dv: c_int, num_k_heads: c_int, num_v_heads: c_int) bool {
    if (seq_len < 64 or dk != 128) return false;
    if (dv < 32 or @rem(dv, 32) != 0) return false;
    return num_k_heads > 0 and @rem(num_v_heads, num_k_heads) == 0;
}

/// mlx-lm's `gated_delta_step` with the per-head gate (`g` [B, T, Hv], read through its raw buffer with a float cast);
/// `capture_seq` also writes the state after every step but the last into `state_seq` [T, B, Hv, Dv, Dk] (a partial
/// accept reads at most row T-2; a full accept keeps `state_out`).
fn stockSource(comptime capture_seq: bool) [:0]const u8 {
    const seq_base = if (capture_seq) "auto seq_base = (n * Dv + dv_idx) * Dk;\n" else "";
    const seq_write = if (capture_seq) kernelSource("kernels/gdn/gated_delta_seq_write.metal") else "";
    return std.fmt.comptimePrint(kernelSource("kernels/gdn/gated_delta.fmt.metal"), .{ seq_base, "auto g_ = g + b_idx * T * Hv;", "g_[hv_idx]", seq_write, "  g_ += Hv;" });
}

var stock_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

/// The stock recurrence ((32, Dv, B*Hv) threads in (32, 4, 1) groups, as mlx-lm) over q, k [B, T, Hk, Dk], v [B, T,
/// Hv, Dv], g, beta [B, T, Hv] from the slot's state: y in `out_dtype` into `y`, the next state into `ssm`, with
/// `capture_seq` each row's into `ssm.spec_state_seq` (the final state its own output, so no view pins the capture).
fn stockRecur(comptime capture_seq: bool, s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, ssm: *SSMCacheEntry, y: *mlx.mlx_array, geo: Geometry, batch: c_int, seq_len: c_int, out_dtype: mlx.mlx_dtype) !void {
    const st_dtype = mlx.mlx_array_dtype(ssm.ssm_state);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, seq_len, geo.hv, geo.dv }, 4, out_dtype));
    if (capture_seq) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ seq_len, batch, geo.hv, geo.dv, geo.dk }, 5, st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, geo.hv, geo.dv, geo.dk }, 4, st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, geo.dv, batch * geo.hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    inline for (.{ .{ "Dk", geo.dk }, .{ "Dv", geo.dv }, .{ "Hk", geo.hk }, .{ "Hv", geo.hv } }) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
    const t_scalar = mlx.mlx_array_new_int(seq_len);
    defer _ = mlx.mlx_array_free(t_scalar);
    const slot = &stock_kernels[@intFromBool(capture_seq)];
    if (capture_seq) {
        const seq_stride = mlx.mlx_array_new_int(batch * geo.hv * geo.dv * geo.dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        if (slot.* == null) slot.* = try makeKernel("gated_delta_step_seq", &.{ "q", "k", "v", "g", "beta", "state_in", "T", "seq_stride" }, &.{ "y", "state_seq", "state_out" }, stockSource(true), "");
        const out = try launch(3, slot.*.?, &.{ q, k, v, g, beta, ssm.ssm_state, t_scalar, seq_stride }, config, s);
        install(y, out[0]);
        install(&ssm.spec_state_seq, out[1]);
        install(&ssm.ssm_state, out[2]);
    } else {
        if (slot.* == null) slot.* = try makeKernel("gated_delta_step", &.{ "q", "k", "v", "g", "beta", "state_in", "T" }, &.{ "y", "state_out" }, stockSource(false), "");
        const out = try launch(2, slot.*.?, &.{ q, k, v, g, beta, ssm.ssm_state, t_scalar }, config, s);
        install(y, out[0]);
        install(&ssm.ssm_state, out[1]);
    }
}

// The GDN core recurrence (kernels/gdn_core_recur.metal): the pipelined kernel with its block and threadgroup sizes as
// template ints, and its bytes.
const CORE_RECUR_SOURCE = kernelSource("plugins/qwen4_exp/kernels/gdn_core_recur.metal");
const CORE_RECUR_HEADER = kernelSource("plugins/qwen4_exp/kernels/gdn_core_recur_header.metal");
/// Tokens a staged block, value rows a threadgroup, k and q staged in the input dtype (KQF 0), one value row a lane.
const TB: c_int = 16;
const DB: c_int = 32;
const KQF: c_int = 0;
const RPT: c_int = 1;
/// Threads a threadgroup: 8 lanes a row group.
const CORE_NT: c_int = @divExact(8 * DB, RPT);
var core_recur_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// One launch over q, k [B, T, Hk, Dk], v [B, T, Hv, Dv], g, beta [B, T, Hv] from `state` [B, Hv, Dv, Dk]: y [B, T, Hv,
/// Dv] in `out_dtype` and the state after the last row in the state's dtype.
fn coreRecur(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state: mlx.mlx_array, hk: c_int, hv: c_int, dk: c_int, dv: c_int, out_dtype: mlx.mlx_dtype) !RecurOut {
    if (dk != 128 or dv <= 0 or @rem(dv, DB) != 0) return error.GdnCoreRecurGeometry;
    const qsh = mlx.getShape(q);
    if (qsh.len != 4) return error.GdnCoreRecurGeometry;
    const batch = qsh[0];
    const t = qsh[1];
    const st_dtype = mlx.mlx_array_dtype(state);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, t, hv, dv }, 4, out_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, hv, dv, dk }, 4, st_dtype));
    // NT threads own DB value rows; Dv / DB threadgroups per value head
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, CORE_NT * @divExact(dv, DB), hv, batch));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, CORE_NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", out_dtype));
    const ints = .{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv }, .{ "TB", TB }, .{ "DB", DB }, .{ "KQF", KQF }, .{ "RPT", RPT } };
    inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
    const t_scalar = mlx.mlx_array_new_int(t);
    defer _ = mlx.mlx_array_free(t_scalar);
    if (core_recur_kernel == null) core_recur_kernel = try makeKernel("msv_gdn_core_recur", &.{ "q", "k", "v", "g", "beta", "state_in", "T" }, &.{ "y", "state_out" }, CORE_RECUR_SOURCE, CORE_RECUR_HEADER);
    const out = try launch(2, core_recur_kernel.?, &.{ q, k, v, g, beta, state, t_scalar }, config, s);
    return .{ .y = out[0], .state = out[1] };
}

// ── decode and verify: the prework and the recurrence in one launch ──
// K1 runs one head over SPLIT threadgroups, each recomputing its head's conv / SiLU / q-k norm prework before its slice
// of the recurrence rows. K1S runs it over TL tokens and writes each row's state but the last (`state_seq`), which a
// partial accept reads. The replay fold runs K1 over TL tokens at one threadgroup per head with the norm-gate and the
// rollback's conv input folded in, and writes the rows' recurrence inputs (what `replay` reads) in place of per-step
// states. M5 runs its 1024-thread pipeline (the copy probes each width for GPUs that cap it lower).

const K1_SOURCE = kernelSource("kernels/gdn/decode_k1.metal");
const K1S_SOURCE = kernelSource("kernels/gdn/decode_k1s_head.metal") ++ "\n    if (lane == 0) y[(t * HV + hv) * DV + dv] = static_cast<T>(out);\n" ++ kernelSource("kernels/gdn/decode_k1s_tail.metal");
const K1S_FOLD_REPLAY_SOURCE = "threadgroup float ys[TL][DV];\n" ++ kernelSource("kernels/gdn/decode_k1s_head.metal") ++ "\n    if (lane == 0) ys[t][dv] = float(static_cast<T>(out));\n" ++ kernelSource("kernels/gdn/decode_k1s_tail_noseq.metal") ++ "\n" ++ kernelSource("kernels/gdn/decode_k1s_fold_epilogue.metal") ++ "\n" ++ kernelSource("kernels/gdn/decode_k1s_prework_out.metal");
const K1_INPUTS = [_][*:0]const u8{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" };
const SPLIT: c_int = 4;
const NT: c_int = 256; // 4 dv rows per simdgroup
/// K1S takes verifies up to 16 rows, one dv row per simdgroup on NAX: a token's rows take one reduction pair in
/// sequence, and each row's arithmetic is K1's.
const MAX_SEQ: c_int = 16;
const SEQ_NT: c_int = 1024;
/// The fold keeps a head's y rows in threadgroup memory beside the q/k/v rows; past 8 they pass 32 KB.
const FOLD_MAX_SEQ: c_int = 8;
const FOLD_NT: c_int = 1024;
const CfgKey = struct { g: Geometry, dt: mlx.mlx_dtype, st: mlx.mlx_dtype };

fn rowsAre(a: mlx.mlx_array, t_len: c_int, width: c_int) bool {
    const sh = mlx.getShape(a);
    return sh.len == 3 and sh[0] == 1 and sh[1] == t_len and sh[2] == width;
}

fn sizeIs(a: mlx.mlx_array, n: c_int) bool {
    return mlx.mlx_array_size(a) == @as(usize, @intCast(n));
}

/// The kernels index every input as a row-major block of a fixed size: per-token rows exactly [1, T, width], weights and
/// states exactly the elements indexed. `gate`: z, norm_w and eps are read too.
fn inputsFit(g: Geometry, t_len: c_int, in: Inputs, gate: bool) bool {
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const base = rowsAre(in.qkv, t_len, c) and rowsAre(in.a, t_len, g.hv) and rowsAre(in.b, t_len, g.hv) and
        sizeIs(in.conv_state, 3 * c) and sizeIs(in.ssm_state, g.hv * g.dv * g.dk) and sizeIs(in.conv_w, 4 * c) and
        sizeIs(in.A_log, g.hv) and sizeIs(in.dt_bias, g.hv) and sizeIs(in.q_scale, 1) and sizeIs(in.k_scale, 1);
    if (!gate) return base;
    return base and rowsAre(in.z, t_len, g.hv * g.dv) and sizeIs(in.norm_w, g.dv) and sizeIs(in.eps, 1);
}

/// The geometry and dtypes K1, K1S and the fold take (every input the activations' bf16 / f16, the state that or f32), with
/// `gated` the fold's z, norm_w and f32 eps: the config key, or null.
fn kernelKey(g: Geometry, t_len: c_int, in: Inputs, gated: bool) ?CfgKey {
    if (g.dk != 128 or g.dv != 128 or @rem(g.hv, g.hk) != 0) return null;
    const dt = mlx.mlx_array_dtype(in.qkv);
    if (dt != .bfloat16 and dt != .float16) return null;
    for ([_]mlx.mlx_array{ in.a, in.b, in.conv_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale }) |arr|
        if (mlx.mlx_array_dtype(arr) != dt) return null;
    if (gated and (mlx.mlx_array_dtype(in.z) != dt or mlx.mlx_array_dtype(in.norm_w) != dt or mlx.mlx_array_dtype(in.eps) != .float32)) return null;
    const st = mlx.mlx_array_dtype(in.ssm_state);
    if (st != dt and st != .float32) return null;
    if (!inputsFit(g, t_len, in, gated)) return null;
    return .{ .g = g, .dt = dt, .st = st };
}

/// T, StT, GT and the geometry's template ints, with NT and SPLIT.
fn decodeTemplate(cfg: mlx.mlx_fast_metal_kernel_config, key: CfgKey, nt: c_int, split: c_int) !void {
    const g = key.g;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", key.st));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "GT", gateDtype(key.dt, key.st)));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", 2 * g.hk * g.dk + g.hv * g.dv }, .{ "NT", nt }, .{ "SPLIT", split } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
}

var k1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var k1_key: ?CfgKey = null;
var k1_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

/// K1: y [1, 1, Hv, Dv], the next conv state and the final state; z, norm_w and eps are not read. Null outside the
/// kernel's geometry or dtypes.
fn decodeRecur(g: Geometry, in: Inputs, s: mlx.mlx_stream) !?Recur {
    if (@rem(g.hv * g.dv, 1024) != 0) return null;
    const key = kernelKey(g, 1, in, false) orelse return null;
    if (k1_kernel == null) k1_kernel = try makeKernel("msv_gdn_decode_recur", &K1_INPUTS, &.{ "y", "conv_out", "state_out" }, K1_SOURCE, GDN_KERNEL_HEADER);
    if (k1_key == null or !std.meta.eql(k1_key.?, key)) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, g.hv, g.dv }, 4, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, 2 * g.hk * g.dk + g.hv * g.dv }, 3, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, key.st));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * SPLIT * NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, NT, 1, 1));
        try decodeTemplate(cfg, key, NT, SPLIT);
        if (k1_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(k1_cfg);
        k1_cfg = cfg;
        k1_key = key;
    }
    const out = try launch(3, k1_kernel.?, &.{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale }, k1_cfg, s);
    return .{ .y = out[0], .conv_state = out[1], .ssm_state = out[2] };
}

var k1s_kernel: ?mlx.mlx_fast_metal_kernel = null;
var seq_cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var seq_cfg_key: ?CfgKey = null;

/// K1S, K1 over t_len tokens (up to MAX_SEQ) with each row's state captured: y [1, T, Hv, Dv], the next conv state, the
/// final state and state_seq [T, 1, Hv, Dv, Dk] (row T-1 unwritten). Null outside the kernel's geometry or dtypes.
fn recurSeq(g: Geometry, t_len: c_int, in: Inputs, s: mlx.mlx_stream) !?decode_types.RecurSeq {
    if (t_len < 1 or t_len > MAX_SEQ) return null;
    const key = kernelKey(g, t_len, in, false) orelse return null;
    if (k1s_kernel == null) k1s_kernel = try makeKernel("msv_gdn_decode_recur_seq", &K1_INPUTS, &.{ "y", "conv_out", "state_out", "state_seq" }, K1S_SOURCE, GDN_KERNEL_HEADER);
    if (seq_cfg_key == null or !std.meta.eql(seq_cfg_key.?, key)) {
        for (&seq_cfgs) |*slot| if (slot.*) |c| {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
            slot.* = null;
        };
        seq_cfg_key = key;
    }
    const idx: usize = @intCast(t_len);
    if (seq_cfgs[idx] == null) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t_len, g.hv, g.dv }, 4, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, 2 * g.hk * g.dk + g.hv * g.dv }, 3, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, key.st));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, 5, key.st));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * SPLIT * SEQ_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, SEQ_NT, 1, 1));
        try decodeTemplate(cfg, key, SEQ_NT, SPLIT);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TL", t_len));
        seq_cfgs[idx] = cfg;
    }
    const out = try launch(4, k1s_kernel.?, &.{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale }, seq_cfgs[idx].?, s);
    return .{ .y = out[0], .conv_state = out[1], .ssm_state = out[2], .state_seq = out[3] };
}

var fold_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fold_cfgs: [FOLD_MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var fold_cfg_key: ?CfgKey = null;

/// The replay fold over a chain's T verify rows (up to 8): gated [1, T, Hv*Dv] (rms_norm(y) * sigmoid(z)), the next conv
/// state, the final state, conv_input [1, 3+T, C] and `prework` (pk [T, Hk, Dk], pv [T, Hv, Dv], pg [T, Hv, 2], f32).
/// Null outside the kernel's geometry or dtypes.
fn recurSeqFoldReplay(g: Geometry, t_len: c_int, in: Inputs, s: mlx.mlx_stream) !?RecurSeqFoldReplay {
    if (t_len < 1 or t_len > FOLD_MAX_SEQ) return null;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    // the replay's grid copies the 3 x C conv window, one thread an element
    if (3 * c > 32 * g.dv * g.hv) return null;
    const key = kernelKey(g, t_len, in, true) orelse return null;
    if (fold_kernel == null) fold_kernel = try makeKernel("msv_gdn_decode_recur_seq_fold_replay", &(K1_INPUTS ++ [_][*:0]const u8{ "z", "norm_w", "eps" }), &.{ "gated", "conv_out", "state_out", "conv_in", "pk", "pv", "pg" }, K1S_FOLD_REPLAY_SOURCE, GDN_KERNEL_HEADER);
    if (fold_cfg_key == null or !std.meta.eql(fold_cfg_key.?, key)) {
        for (&fold_cfgs) |*slot| if (slot.*) |cf| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cf);
            slot.* = null;
        };
        fold_cfg_key = key;
    }
    const idx: usize = @intCast(t_len);
    if (fold_cfgs[idx] == null) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t_len, g.hv * g.dv }, 3, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c }, 3, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, key.st));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3 + t_len, c }, 3, key.dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, g.hk, g.dk }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, g.hv, g.dv }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, g.hv, 2 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * FOLD_NT, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, FOLD_NT, 1, 1));
        try decodeTemplate(cfg, key, FOLD_NT, 1);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TL", t_len));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SWISH", 0));
        fold_cfgs[idx] = cfg;
    }
    const out = try launch(7, fold_kernel.?, &.{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale, in.z, in.norm_w, in.eps }, fold_cfgs[idx].?, s);
    return .{ .gated = out[0], .conv_state = out[1], .ssm_state = out[2], .conv_input = out[3], .prework = .{ .k = out[4], .v = out[5], .gb = out[6] } };
}

// ── the fused norm-gate ──
// rms_norm(y, norm_w) * sigmoid(z) -> flat [B, S, Hv*Dv] in one launch at decode and verify widths: `rms_single_row`'s
// reduction and MLX's sigmoid, bit-identical to the composed chain. z is read at (ZOFF, ZSTRIDE).

const GDN_NORMGATE_SOURCE = kernelSource("kernels/gdn/gdn_normgate.metal");
var normgate_kernel: ?mlx.mlx_fast_metal_kernel = null;
const NormGateCfgKey = struct { hv: c_int, seq: c_int, batch: c_int, zs: c_int, zo: c_int, dt: mlx.mlx_dtype };
var normgate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var normgate_cfg_key: NormGateCfgKey = std.mem.zeroes(NormGateCfgKey);

/// y [B, S, Hv, Dv] and z's rows of [.. ZOFF + Hv*Dv ..] (bf16 / f16 alike), norm_w [Dv], eps (0-dim f32). Null: the
/// composed chain.
fn normGateFused(s: mlx.mlx_stream, y: mlx.mlx_array, z: mlx.mlx_array, z_off: c_int, z_stride: c_int, norm_w: mlx.mlx_array, eps: mlx.mlx_array, hv: c_int, dv: c_int, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (dv != 128 or seq < 1 or seq > GDN_FUSED_MAX_ROWS or batch < 1 or batch * seq > GDN_FUSED_MAX_ROWS) return null;
    const dt = mlx.mlx_array_dtype(y);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(z) != dt or mlx.mlx_array_dtype(norm_w) != dt) return null;
    const ysh = mlx.getShape(y);
    const zsh = mlx.getShape(z);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv) return null;
    if (zsh.len != 3 or zsh[0] != batch or zsh[1] != seq or zsh[2] != z_stride or z_off + hv * dv > z_stride) return null;
    const key = NormGateCfgKey{ .hv = hv, .seq = seq, .batch = batch, .zs = z_stride, .zo = z_off, .dt = dt };
    if (normgate_cfg == null or !std.meta.eql(normgate_cfg_key, key)) {
        if (normgate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        normgate_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ batch, seq, hv * dv }, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, batch * seq, hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        const ints = .{ .{ "HV", hv }, .{ "DV", dv }, .{ "ZSTRIDE", z_stride }, .{ "ZOFF", z_off }, .{ "TAB", @as(c_int, 0) }, .{ "SWISH", @as(c_int, 0) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        normgate_cfg = config;
        normgate_cfg_key = key;
    }
    if (normgate_kernel == null) normgate_kernel = try makeKernel("mlxserve_gdn_normgate", &.{ "y", "z", "norm_w", "eps", "sigtab" }, &.{"out"}, GDN_NORMGATE_SOURCE, "");
    // the sigmoid table's slot holds a placeholder at these widths (TAB 0)
    const out = try launch(1, normgate_kernel.?, &.{ y, z, norm_w, eps, y }, normgate_cfg.?, s);
    return out[0];
}
