//! This plugin's own, in place of mlx-serve's `src/plugins/qwen4_exp/trunk.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the trunk with the fields the K packs' forward reads; its methods are mlx-serve's at
//! 1b3e624b.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const model = @import("../../sdk/config/model.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const mtp = @import("mtp.zig");
const distributed_mod = @import("../../engines/mlx/cluster/distributed.zig");
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const PersistentSsmGroup = @import("../../engines/mlx/ssm/group.zig").PersistentSsmGroup;
const HcWeights = @import("weights.zig").HcWeights;
const QsaPooledRope = @import("qsa/pooled.zig").QsaPooledRope;
const QsaBlockConsts = @import("qsa/pooled.zig").QsaBlockConsts;
const BitsCache = @import("../../sdk/quant/params.zig").BitsCache;
const AttnDqCopy = @import("../../sdk/blocks/attention/decode_attn_quant.zig").AttnDqCopy;
const QkAngleSlot = @import("../../sdk/blocks/rope/qk_norm_rope.zig").QkAngleSlot;
const DtypeCastCache = @import("../../mlx/dtype.zig").DtypeCastCache;
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");
pub const Trunk = struct {
    config: model.ModelConfig,
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    emb_w: mlx.mlx_array,
    emb_s: mlx.mlx_array,
    emb_b: mlx.mlx_array,
    lm_head_w: mlx.mlx_array,
    lm_head_s: mlx.mlx_array,
    lm_head_b: mlx.mlx_array,
    ternary_2bit: bool = false,
    formats: @import("trunk/formats.zig").Formats = .{},
    moe_layers: []MoeLayerWeights,
    moe_owned_bf16: []mlx.mlx_array = &.{},
    qwen4_mixer: HcWeights,
    qwen4: ?*qwen4_mod.Qwen4State = null,
    qwen4_mtp: ?mtp.Qwen4Mtp = null,
    qwen4_mtp_owner: ?*mtp.Qwen4MtpState = null,
    ones_hidden: mlx.mlx_array,
    one: mlx.mlx_array,
    rms_eps_arr: mlx.mlx_array,
    rope_freqs_yarn: ?mlx.mlx_array = null,
    yarn_mscale: ?mlx.mlx_array = null,
    yarn_inv_freq: ?[]const f64 = null,
    yarn_mscale_cast: DtypeCastCache = .{},
    qwen4_stream_f32: bool = false,
    qsa_consts: QsaBlockConsts = .{},
    qsa_pooled_rope: QsaPooledRope = .{},
    fwd_gen: u64 = 0,
    ple_ids_stream: mlx.mlx_stream = .{ .ctx = null },
    ple_kick: mlx.mlx_array = .{ .ctx = null },
    dist_peer: ?*distributed_mod.Peer = null,
    dist_tp: ?*distributed_mod.Peer = null,
    compiled_moe_routing: ?mlx.mlx_closure = null,
    compiled_gdn_gate: ?mlx.mlx_closure = null,
    compiled_hc_silu: ?mlx.mlx_closure = null,
    compiled_hc_mix: ?mlx.mlx_closure = null,
    compiled_hc_inj: ?mlx.mlx_closure = null,
    compiled_hc_write: ?mlx.mlx_closure = null,
    gdn_ones_w: ?mlx.mlx_array = null,
    gdn_q_scale: ?mlx.mlx_array = null,
    gdn_k_scale: ?mlx.mlx_array = null,
    gdn_eps: ?mlx.mlx_array = null,
    spec_capture_ssm: bool = false,
    bits_cache: BitsCache = .{},
    attn_dq: std.AutoHashMapUnmanaged(usize, ?AttnDqCopy) = .empty,
    qk_angle_cache: [2]QkAngleSlot = .{ .{}, .{} },
    verify_joined_logits: mlx.mlx_array = .{ .ctx = null },
    decode_async_ladder: u32 = 0,
    decode_async_ladder_qwen4: u32 = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT,
    ssm_group: PersistentSsmGroup = .{},
    mtp_uniform_affine_trunk: bool = false,
    mtp_oqe_affine_trunk: bool = false,
    fused: @import("trunk/fusion.zig").Scratch = .{},
    pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT = @import("forward.zig").QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    pub const compileQwen4Hc = @import("hc.zig").compileQwen4Hc;
    pub const HcRead = @import("hc.zig").HcRead;
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
    pub const plePrefetchEnd = @import("ple.zig").plePrefetchEnd;
    pub const releasePlePending = @import("ple.zig").releasePlePending;
    pub const markQsaPooledRopeStale = @import("qsa/pooled.zig").markQsaPooledRopeStale;
    pub const reordered = @import("trunk/formats.zig").reordered;
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
    pub const compileGdnGate = @import("trunk/gdn.zig").compileGdnGate;
    pub const moeRoutingUncompiled = @import("trunk/moe.zig").moeRoutingUncompiled;
    pub const compileMoeRouting = @import("trunk/moe.zig").compileMoeRouting;
    pub const flushDeferredPleGroup = @import("ple.zig").flushDeferredPleGroup;
    pub const settleDeferredPle = @import("ple.zig").settleDeferredPle;
    pub const discardDeferredPle = @import("ple.zig").discardDeferredPle;
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
