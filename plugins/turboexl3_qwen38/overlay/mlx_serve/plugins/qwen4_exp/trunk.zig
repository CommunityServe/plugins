//! copied_from: src/plugins/qwen4_exp/trunk.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The qwen4_exp trunk module: the loaded weights (layers with their hyper-connection and PLE blocks, the final
//! mixer, the MTP head), the n-gram state, and the per-model caches its forward keeps (QSA block constants and pooled
//! rope, the compiled closures, quant params, decode requant copies, angle rows, the persistent batched GDN group).
//! Its methods are the family's forward (`forward.zig`, `mtp.zig`, `ple.zig`, `hc.zig`, `qsa/`) and the trunk ops
//! (`trunk/`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../sdk/config/model.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const mtp = @import("mtp.zig");
const distributed_mod = @import("../../engines/mlx/cluster/distributed.zig");
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const PersistentSsmGroup = @import("../../engines/mlx/ssm/group.zig").PersistentSsmGroup;
const VerifyLaps = @import("../../engines/mlx/batched/row_axis_verify.zig").VerifyLaps;
const HcWeights = @import("weights.zig").HcWeights;
const QsaPooledRope = @import("qsa/pooled.zig").QsaPooledRope;
const QsaBlockConsts = @import("qsa/pooled.zig").QsaBlockConsts;
const BitsCache = @import("../../sdk/quant/params.zig").BitsCache;
const AttnDqCopy = @import("../../sdk/blocks/attention/decode_attn_quant.zig").AttnDqCopy;
const QkAngleSlot = @import("../../sdk/blocks/rope/qk_norm_rope.zig").QkAngleSlot;
const DtypeCastCache = @import("../../mlx/dtype.zig").DtypeCastCache;
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");

pub const Trunk = struct {
    /// Borrowed: its owner (the plugin's model, or the engine's legacy shell) frees it after the trunk.
    config: model.ModelConfig,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,

    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    /// A ternary 2-bit pack: its projections read the 2-bit kernels (`trunk/linear.zig` `qmatmul`).
    ternary_2bit: bool = false,
    /// The pack's quant formats bound at load (`trunk/formats.zig`): the trunk format's linears, the expert format's
    /// banks, the numerics.
    formats: @import("trunk/formats.zig").Formats = .{},
    moe_layers: []MoeLayerWeights,
    /// Arrays the loader built (transposed bf16, folded hyper-connection weights): freed with the trunk.
    moe_owned_bf16: []mlx.mlx_array = &.{},
    /// The final hyper-connection read, in place of `model.norm`.
    qwen4_mixer: HcWeights,
    /// The n-gram hash and the mmapped table (plus its GPU arm).
    qwen4: ?*qwen4_mod.Qwen4State = null,
    /// The in-checkpoint MTP layer; null without one (or on a pipeline rank).
    qwen4_mtp: ?mtp.Qwen4Mtp = null,
    /// Whose per-request state the head holds (`qwen4MtpActivate`).
    qwen4_mtp_owner: ?*mtp.Qwen4MtpState = null,

    /// `[hidden]` ones for the weight-less RMS norms; 1.0 (bf16) for the dispatch probe; `rms_norm_eps` as f32.
    ones_hidden: mlx.mlx_array,
    one: mlx.mlx_array,
    rms_eps_arr: mlx.mlx_array,
    /// YaRN: the rope denominators, the mscale vector, the host inverse frequencies; null on an unscaled rope.
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,
    yarn_inv_freq: ?[]const f64 = null,
    yarn_mscale_cast: DtypeCastCache = .{},

    /// Run the residual streams in f32 (the fixture oracle's tighter bar).
    qwen4_stream_f32: bool = false,
    qsa_consts: QsaBlockConsts = .{},
    qsa_pooled_rope: QsaPooledRope = .{},
    /// Bumped by every forward entry that can reach QSA; keys the per-forward QSA scratch.
    fwd_gen: u64 = 0,
    /// Set while the wavefront builds a chunk: the stream the PLE layer's id read runs on (the CPU's), so it waits behind
    /// neither half's GPU queue (`plePrepareIds`).
    ple_ids_stream: mlx.mlx_stream = .{ .ctx = null },
    /// A prefill chunk's stream as built up to its PLE layer, borrowed while that layer runs: `pleKickOff` starts it on the
    /// GPU once the ids are read, so those layers run under the host gather (the PLE prefetch).
    ple_kick: mlx.mlx_array = .{ .ctx = null },
    /// Test meter: how many times `qsaScoreBank` built or extended its transposed operand.
    qsa_score_bank_builds: usize = 0,
    /// Pipeline rank 0: the worker holding the later layers; tensor rank 0: the worker holding the other expert slices.
    dist_peer: ?*distributed_mod.Peer = null,
    dist_tp: ?*distributed_mod.Peer = null,

    compiled_moe_routing: ?mlx.mlx_closure = null,
    compiled_gdn_gate: ?mlx.mlx_closure = null,
    /// The hyper-connection elementwise tails, one kernel each: silu(down), mean(sigma(up)*normed), 2sigma(inject),
    /// stream + out*inj.
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,
    gdn_ones_w: ?mlx.mlx_array = null,
    gdn_q_scale: ?mlx.mlx_array = null,
    gdn_k_scale: ?mlx.mlx_array = null,
    gdn_eps: ?mlx.mlx_array = null,
    /// A verify forward in flight: the recurrent layers keep each row's state for the partial-accept rollback.
    spec_capture_ssm: bool = false,
    bits_cache: BitsCache = .{},
    /// Decode-only int8/nvfp4 copies of dense attention projections, by the weight's pointer (null = ineligible).
    attn_dq: std.AutoHashMapUnmanaged(usize, ?AttnDqCopy) = .empty,
    qk_angle_cache: [2]QkAngleSlot = .{ .{}, .{} },
    /// The round-cost tracer's per-forward arm bits.
    cost_trace_active: bool = false,
    cost_attention: u8 = 0,
    cost_moe: u8 = 0,
    verify_laps: VerifyLaps = .{},
    /// The joined `[1, sum(rows), V]` logits of the last grouped verify whose head ran once (released by the next).
    verify_joined_logits: mlx.mlx_array = .{ .ctx = null },
    /// Layer stride for the decode async-eval ladder (0 = off) and the batched tick's.
    decode_async_ladder: u32 = 0,
    decode_async_ladder_qwen4: u32 = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT,
    ssm_group: PersistentSsmGroup = .{},
    /// Test seams (no env): `false` runs the per-tick merge/split and the per-slot KV append.
    persistent_group_state_override: ?bool = null,
    batched_kv_append_override: ?bool = null,
    /// The MTP cost profile's trunk fingerprints (homogeneous affine-4/gs-64; the oQ4e mixed layout).
    mtp_uniform_affine_trunk: bool = false,
    mtp_oqe_affine_trunk: bool = false,
    /// The fused paths' per-layer state (`trunk/fusion.zig`).
    fused: @import("trunk/fusion.zig").Scratch = .{},


    // attention.zig
    // forward.zig
    pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    // hc.zig
    pub const compileQwen4Hc = @import("hc.zig").compileQwen4Hc;
    pub const HcRead = @import("hc.zig").HcRead;
    // mtp.zig
    pub const HeadPlace = @import("mtp.zig").HeadPlace;
    pub const loadQwen4Mtp = @import("mtp.zig").loadQwen4Mtp;
    pub const qwen4BuildDraftRerank = @import("mtp.zig").qwen4BuildDraftRerank;
    pub const qwen4DraftRerankReady = @import("mtp.zig").qwen4DraftRerankReady;
    pub const qwen4DraftSelect = @import("mtp.zig").qwen4DraftSelect;
    pub const qwen4MtpActivate = @import("mtp.zig").qwen4MtpActivate;
    pub const qwen4MtpForward = @import("mtp.zig").qwen4MtpForward;
    pub const qwen4MtpForwardOn = @import("mtp.zig").qwen4MtpForwardOn;
    pub const Qwen4MtpLive = @import("mtp.zig").Qwen4MtpLive;
    pub const Qwen4MtpOut = @import("mtp.zig").Qwen4MtpOut;
    pub const Qwen4MtpProject = @import("mtp.zig").Qwen4MtpProject;
    pub const qwen4MtpRelease = @import("mtp.zig").qwen4MtpRelease;
    pub const Qwen4MtpState = @import("mtp.zig").Qwen4MtpState;
    pub const qwen4MtpStateNew = @import("mtp.zig").qwen4MtpStateNew;
    pub const qwen4MtpTruncate = @import("mtp.zig").qwen4MtpTruncate;
    // ple.zig
    pub const plePrefetchEnd = @import("ple.zig").plePrefetchEnd;
    pub const releasePlePending = @import("ple.zig").releasePlePending;
    // qsa/decode_gather.zig
    // qsa/indexer.zig
    // qsa/pooled.zig
    pub const markQsaPooledRopeStale = @import("qsa/pooled.zig").markQsaPooledRopeStale;
    // trunk/formats.zig
    pub const reordered = @import("trunk/formats.zig").reordered;
    // trunk/linear.zig
    pub const qmatmul = @import("trunk/linear.zig").qmatmul;
    pub const lmHeadProject = @import("trunk/linear.zig").lmHeadProject;
    pub const quantParamsHinted = @import("trunk/linear.zig").quantParamsHinted;
    pub const attnProj = @import("trunk/linear.zig").attnProj;
    pub const attnDqFor = @import("trunk/linear.zig").attnDqFor;
    pub const rmsNorm = @import("trunk/linear.zig").rmsNorm;
    pub const silu = @import("trunk/linear.zig").silu;
    pub const computeGeglu = @import("trunk/linear.zig").computeGeglu;
    pub const embedding = @import("trunk/linear.zig").embedding;
    pub const lmHeadForeign = @import("trunk/linear.zig").lmHeadForeign;
    // trunk/attn.zig
    // trunk/gdn.zig
    pub const compileGdnGate = @import("trunk/gdn.zig").compileGdnGate;
    // trunk/moe.zig
    pub const moeRoutingUncompiled = @import("trunk/moe.zig").moeRoutingUncompiled;
    pub const compileMoeRouting = @import("trunk/moe.zig").compileMoeRouting;
    // trunk/verify.zig
    // trunk/batched.zig
    pub const flushDeferredPleGroup = @import("ple.zig").flushDeferredPleGroup;
    pub const settleDeferredPle = @import("ple.zig").settleDeferredPle;
    pub const discardDeferredPle = @import("ple.zig").discardDeferredPle;
    // trunk/warmup.zig
    pub const warmupSpecVerify = @import("trunk/warmup.zig").warmupSpecVerify;

    pub fn actDtype(self: *const Trunk) mlx.mlx_dtype {
        return self.config.actDtype();
    }

    pub fn ssmStateDtype(self: *const Trunk) mlx.mlx_dtype {
        return self.config.ssmStateDtype();
    }

    /// The compiled routing, GDN gate and hyper-connection closures the forward reads.
    pub fn prepare(self: *Trunk) void {
        if (mlx.noGpuBackend()) return;
        self.compileMoeRouting();
        if (self.config.linear_num_key_heads > 0) self.compileGdnGate();
        self.compileQwen4Hc();
    }

    pub fn deinit(self: *Trunk) void {
        if (self.verify_joined_logits.ctx != null) _ = mlx.mlx_array_free(self.verify_joined_logits);
        lane_qmm.release(@intFromPtr(self));
        inline for (.{ self.compiled_moe_routing, self.compiled_gdn_gate, self.compiled_hc_silu, self.compiled_hc_mix, self.compiled_hc_inj, self.compiled_hc_write }) |c| {
            if (c) |cl| _ = mlx.mlx_closure_free(cl);
        }
        inline for (.{ self.gdn_ones_w, self.gdn_q_scale, self.gdn_k_scale, self.gdn_eps, self.rope_freqs_yarn, self.yarn_mscale }) |a| {
            if (a) |arr| _ = mlx.mlx_array_free(arr);
        }
        if (self.yarn_inv_freq) |f| self.allocator.free(f);
        _ = mlx.mlx_array_free(self.ones_hidden);
        _ = mlx.mlx_array_free(self.one);
        _ = mlx.mlx_array_free(self.rms_eps_arr);
        self.qsa_consts.deinit();
        self.qsa_pooled_rope.deinit();
        self.yarn_mscale_cast.deinit();
        {
            var it = self.attn_dq.valueIterator();
            while (it.next()) |maybe| if (maybe.*) |copy| {
                _ = mlx.mlx_array_free(copy.w);
                _ = mlx.mlx_array_free(copy.s);
                if (copy.b.ctx != null) _ = mlx.mlx_array_free(copy.b);
            };
            self.attn_dq.deinit(self.allocator);
        }
        for (&self.qk_angle_cache) |*slot| {
            if (slot.arr.ctx != null) _ = mlx.mlx_array_free(slot.arr);
        }
        self.ssm_group.deinit();
        self.allocator.free(self.moe_layers);
        if (self.qwen4_mtp) |*m| {
            m.deinit(self.allocator);
            self.qwen4_mtp = null;
        }
        if (self.qwen4) |st| {
            st.deinit();
            self.allocator.destroy(st);
            self.qwen4 = null;
        }
        for (self.moe_owned_bf16) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(self.moe_owned_bf16);
        self.formats.deinit(self.allocator);
    }
};
