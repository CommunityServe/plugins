//! The DeepSeek-V4.1 trunk's kernel routes: the kernel side of the arch module's lever routes
//! (RCTAIL, RCPROJ, HCTAPE, K3 / K36, DRAFTRC, decode batch 2 and prefill batch 2). This is the
//! kernel set's second consumer beside the EXL3 routed-expert quant (`exl3_quant`). Its subset
//! (`kernels`, 55 of the set) is self-checked by `accept` on the set the load context owns; the
//! arch builds its routes after that. The routes below are the lanes' own launches, moved here
//! verbatim from exl3_kernel_ops.zig.

const std = @import("std");
const mlx = @import("sdk").mlx;
const xk = @import("kernels.zig");
const sdk_ext = @import("../streamed/sdk_ext.zig");
const selfcheck = @import("selfcheck.zig");
const kr = sdk_ext.kernels.Routes(xk);
const ks = sdk_ext.kernels.KernelSet(xk);

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Entry = xk.Entry;
const Vars = xk.Vars;
const LaunchConfig = xk.LaunchConfig;
const Dtype = mlx.mlx_dtype;
const Refusal = kr.Refusal;
const refuse = kr.refuse;
const argOf = kr.argOf;
const dims = kr.dims;
const Shape = kr.Shape;
const expectInput = kr.expectInput;
const launchRule = kr.launchRule;
const rowsOf = kr.rowsOf;
const Statics = kr.Statics;
const RowPlans = kr.RowPlans;
const no_vars = kr.no_vars;
const rowsVars = kr.rowsVars;
const declaresFn = kr.declaresFn;

/// This consumer's subset of the kernel set: every kernel of record but the EXL3 quant's.
pub const kernels = [_]Kernel{
    .q3rc_gate_part,
    .q3rc_router_tail,
    .q3rc_premix_part,
    .q3rc_premix_fin,
    .q3dk_sinkhorn16_hc4_it20,
    .q3rc_mxfp8_fma,
    .q3ht_combine,
    .q3ht_collapse_norm,
    .q3ht_combine_collapse_norm,
    .q3ht_mixfin,
    .mtplx_dsv4_sinkhorn_hc4_it20,
    .mtplx_dsv41_fp_rmsnorm_tg128_d1280,
    .mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64,
    .mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd,
    .mtplx_dsv41_fp_rope_h64_hd512_rd64_inv,
    .q3rc_gate_part__n128,
    .q3rc_router_tail__n128_top3,
    .q3ht_combine__f32,
    .q3ht_collapse_norm__f32,
    .q3ht_combine_collapse_norm__f32,
    .q3ht_combine_collapse_norm__f32_rbf16,
    .q3drc_mxfp8_fma_f32x,
    .q3rc_mxfp8_fma__draft,
    .dsv41_woa_decode_transpose_32,
    .mtplx_dsv41_index_topk_select,
    .q3_attnfuse_softmax,
    .q3_attnfuse_softmax__ls128,
    .dsv41_mxfp8_m1rows,
    .dsv41_head_m1rows,
    .dsv41_smallm_all,
    .dsv41_smallm_all__bf16,
    .q3_ph_index_score,
    .q3_ph_qkvec_win,
    .q3_ph_qkvec_win__kvf32,
    .q3_ph_qkvec_win__qf32,
    .q3_ph_qkvec_win__qf32_kvf32,
    .q3_ph_qkvec_cmp,
    .q3_ph_qkvec_cmp__kvbf16,
    .q3_ph_qkvec_cmp__qf32,
    .q3_ph_pvvec_win,
    .q3_ph_pvvec_win__kvf32,
    .q3_ph_pvvec_cmp,
    .q3_ph_pvvec_cmp__kvbf16,
    .q3_ph_qkrope_win,
    .q3_ph_qkrope_win__kvf32,
    .q3_ph_qkrope_win__qf32,
    .q3_ph_qkrope_win__qf32_kvf32,
    .q3_ph_qkrope_cmp,
    .q3_ph_qkrope_cmp__kvbf16,
    .q3_ph_qkrope_cmp__qf32,
    .q3_ph_pvrope_win,
    .q3_ph_pvrope_win__kvf32,
    .q3_ph_pvrope_cmp,
    .q3_ph_pvrope_cmp__kvbf16,
    .q3pf_hc_mix_rsqrt,
    .q3pf_hc_pre_norm,
    .q3pf_hc_mix_rsqrt__f32,
    .q3pf_hc_pre_norm__f32,
    .q3sk_combine,
    .q3sk_combine__rbf16,
    .q3jl_combine,
    .dsv41_jl_combine_bf16,
    .dsv41_hcpost_tf32,
    .dsv41_hcpost_tf32__rbf16,
    .dsv41_hcpost_tf32_bf16,
};

/// The arch's kernel acceptance, once per backend before its routes are built: this subset's
/// self-check plan on the load context's kernel set, recorded in `report` and judged (refused:
/// SelfCheckFailed, `diag` naming the first failing kernel / check / site).
pub fn accept(a: Allocator, set: *const ks.Set, report: *selfcheck.Report, diag: *xk.Diag) !void {
    try set.selfCheck(a, &kernels, .compile, report, diag);
}

// ── RCTAIL (DSV41_DECODE_RCTAIL = router, hcpremix, sinkhorn) ──

/// router: the MoE gate on q3rc_gate_part (split-K logits) + q3rc_router_tail (sqrt(softplus),
/// + bias, top-6, weights / sum x 1.5): `RouterKernels.run`, decode / verify rows (M <= 8).
pub fn Router(comptime G: type) type {
    return struct {
        const Self = @This();
        part: *const Entry,
        tail: *const Entry,
        part_p: RowPlans(G, 8),
        tail_p: RowPlans(G, 8),
        w: G.T,
        bias: G.T,

        /// `w` the gate weight (bf16 [N, 5120]), `bias` the selection bias (f32 [N]): N 384 = the
        /// verify router (top-6), N 128 = the DSpark draft router (DRAFTRC member router: the
        /// __n128 / __n128_top3 variants, top-3). Another N is refused (RouteInput).
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, bias: G.T, diag: ?*xk.Diag) !Self {
            const n = rowsOf(G, g, w, 0);
            const part, const tail = switch (n) {
                384 => .{ reg.get(.q3rc_gate_part), reg.get(.q3rc_router_tail) },
                128 => .{ reg.get(.q3rc_gate_part__n128), reg.get(.q3rc_router_tail__n128_top3) },
                else => return refuse(diag, error.RouteInput, "exl3 kernel ops: a router of {d} experts (the registry carries 384 and the draft's 128)", .{n}),
            };
            try expectInput(G, g, part, "w", w, &no_vars, diag);
            try expectInput(G, g, tail, "bias", bias, &no_vars, diag);
            var part_p: RowPlans(G, 8) = try .init(g, part, null, diag);
            errdefer part_p.deinit(g);
            const tail_p: RowPlans(G, 8) = try .init(g, tail, null, diag);
            return .{ .part = part, .tail = tail, .part_p = part_p, .tail_p = tail_p, .w = g.keep(w), .bias = g.keep(bias) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.part_p.deinit(g);
            self.tail_p.deinit(g);
            g.release(self.w);
            g.release(self.bias);
        }

        /// The `_gate_topk_impl` seam at rows <= 8: `run(xf.astype(f32), gweight, gbias)`.
        pub fn gateTopk(self: *const Self, g: *G, xf: G.T) ![2]G.T {
            return self.call(g, try g.astype(xf, .float32));
        }

        /// x [M, 5120] f32 -> (weights [M, top-k] f32, indices [M, top-k] i32): top-6 / top-3.
        pub fn call(self: *const Self, g: *G, x: G.T) ![2]G.T {
            const m = rowsOf(G, g, x, 0);
            var part: [1]G.T = undefined;
            try self.part_p.launch(g, m, &.{ x, self.w }, &part);
            var out: [2]G.T = undefined;
            try self.tail_p.launch(g, m, &.{ part[0], self.bias }, &out);
            return out;
        }
    };
}

/// hcpremix: the HC premix GEMV on q3rc_premix_part + q3rc_premix_fin: `PremixKernels.run`.
pub fn Premix(comptime G: type) type {
    return struct {
        const Self = @This();
        part: *const Entry,
        fin: *const Entry,
        part_p: RowPlans(G, 8),
        fin_p: RowPlans(G, 8),
        w: G.T,

        /// `w` the premix weight (f32 [24, 20480]).
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, diag: ?*xk.Diag) !Self {
            const part = reg.get(.q3rc_premix_part);
            const fin = reg.get(.q3rc_premix_fin);
            try expectInput(G, g, part, "w", w, &no_vars, diag);
            var part_p: RowPlans(G, 8) = try .init(g, part, null, diag);
            errdefer part_p.deinit(g);
            const fin_p: RowPlans(G, 8) = try .init(g, fin, null, diag);
            return .{ .part = part, .fin = fin, .part_p = part_p, .fin_p = fin_p, .w = g.keep(w) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.part_p.deinit(g);
            self.fin_p.deinit(g);
            g.release(self.w);
        }

        /// The `_q3_attnkernel_mm(a, w)` seam at <= 8 rows: a [..., 20480] f32 -> [..., 24] f32
        /// (a 1-d lead stays the kernel's [M, 24]).
        pub fn mm(self: *const Self, g: *G, a: G.T) !G.T {
            const sh = dims(G, g, a);
            const lead = sh.slice()[0 .. sh.n - 1];
            var m: c_int = 1;
            for (lead) |d| m *= d;
            const out = try self.call(g, try g.reshape(a, &.{ m, sh.slice()[sh.n - 1] }));
            if (lead.len == 1) return out;
            var shape: Shape = .of(lead);
            shape.d[shape.n] = @intCast(argOf(self.fin, "part").shape[2].m);
            shape.n += 1;
            return g.reshape(out, shape.slice());
        }

        /// x [M, 20480] f32 -> out [M, 24] f32.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            const m = rowsOf(G, g, x, 0);
            var part: [1]G.T = undefined;
            try self.part_p.launch(g, m, &.{ x, self.w }, &part);
            var out: [1]G.T = undefined;
            try self.fin_p.launch(g, m, &.{part[0]}, &out);
            return out[0];
        }
    };
}

/// SINKHORN_METAL: `_sinkhorn_kernel_apply` as the RCTAIL sinkhorn member rebinds it: up to 32
/// matrices on the 16-lane kernel (`Sinkhorn16.run`, bitwise the stock result), more (prefill
/// chunks) on the stock K3 text (`deepseek_v4._sinkhorn_kernel_apply`).
pub fn Sinkhorn(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_mats = 32;
        e: *const Entry,
        k3: *const Entry,
        plans: RowPlans(G, max_mats),
        nmat: [max_mats]G.T,

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const e = reg.get(.q3dk_sinkhorn16_hc4_it20);
            var s: Self = .{ .e = e, .k3 = reg.get(.mtplx_dsv4_sinkhorn_hc4_it20), .plans = try .init(g, e, "", null), .nmat = undefined };
            errdefer s.plans.deinit(g);
            var built: usize = 0;
            errdefer for (s.nmat[0..built]) |a| g.release(a);
            for (0..max_mats) |i| {
                const n: i32 = @intCast(i + 1);
                s.nmat[i] = g.keep(try g.hostArray(std.mem.asBytes(&n), &.{}, .int32));
                built += 1;
            }
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            for (self.nmat) |a| g.release(a);
        }

        /// comb [..., 4, 4] f32 (n = numel / 16 matrices) -> the Sinkhorn projection, comb's shape.
        pub fn call(self: *const Self, g: *G, comb: G.T) !G.T {
            const sh = dims(G, g, comb);
            var numel: usize = 1;
            for (sh.slice()) |v| numel *= @intCast(v);
            const n = numel / 16;
            const c3 = try g.reshape(comb, &.{ @intCast(n), 4, 4 });
            var out: [1]G.T = undefined;
            if (n <= max_mats) {
                try self.plans.launch(g, n, &.{ c3, self.nmat[n - 1] }, &out);
            } else {
                const count: i32 = @intCast(n);
                const nm = try g.hostArray(std.mem.asBytes(&count), &.{}, .int32);
                try launchRule(G, g, self.k3, &rowsVars(n), &.{ c3, nm }, &out);
            }
            return g.reshape(out[0], sh.slice());
        }
    };
}

// ── RCPROJ (DSV41_DECODE_RCPROJ = mxfp8, woarc) and DRAFT_HEAD = both (the head site) ──

/// The decode-once mxfp8 FMA sites: the four verify projections, the o-LoRA wo_a (8 groups) and
/// the head (DRAFT_HEAD = both: the bf16 head quantized once to mxfp8 gs 32).
pub const RcSite = enum { wq_a, wkv, wq_b, wo_b, woa, head };

/// One site on q3rc_mxfp8_fma at its pinned geometry, a launch plan per M = 1..8: `Kernels.run`.
pub fn RcProj(comptime G: type) type {
    return FmaSiteProj(G, RcSite);
}

// ── DRAFTRC proj (DSV41_DECODE_DRAFTRC member proj): the DSpark draft block's rcproj sites ──

/// The draft's rcproj sites (`q3_decode_draftrc_candidate.DRAFT_SHAPES`): the verify sites the
/// draft stages share, and the draft-only main_proj / shared expert.
pub const DraftSite = enum { wq_a, wkv, wq_b, wo_b, woa, main_proj, shared_w13, shared_w2 };

/// One draft site at one x dtype (`DraftProjKernels.run`): bf16 x -> the registered FMA text
/// (q3rc_mxfp8_fma at the verify sites, its plan variant q3rc_mxfp8_fma__draft at the
/// draft-only sites), f32 x -> q3drc_mxfp8_fma_f32x; the output has x's dtype; a plan per M =
/// 1..8 at the site's pinned geometry. The dtype is the call site's (fixed at construction).
pub fn DraftProj(comptime G: type) type {
    return FmaSiteProj(G, DraftSite);
}

/// `RcProj` and `DraftProj`: one mxfp8 FMA site (`Site`: `RcSite` or `DraftSite`) bound to its text at construction
/// (`init`: the verify sites' text, or the draft's by site and x dtype); the same plans, weights and calls.
fn FmaSiteProj(comptime G: type, comptime Site: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        site: Site,
        plans: RowPlans(G, max_rows),
        w: G.T,
        scales: G.T,

        pub const init = if (Site == RcSite) initRc else initDraft;

        /// `w` the packed mxfp8 weight (u32 [G N, K / 4]), `scales` its e8m0 scales (u8 [G N, K / 32]).
        fn initRc(g: *G, reg: *const xk.Registry, site: RcSite, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            return bind(g, reg.get(.q3rc_mxfp8_fma), site, w, scales, diag);
        }

        /// `x_dtype` bf16 or f32 (the call site's activations), `w` the packed mxfp8 weight (u32
        /// [G N, K / 4]), `scales` its e8m0 scales (u8 [G N, K / 32]).
        fn initDraft(g: *G, reg: *const xk.Registry, site: DraftSite, x_dtype: Dtype, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const draft_only = switch (site) {
                .main_proj, .shared_w13, .shared_w2 => true,
                else => false,
            };
            const e = switch (x_dtype) {
                .bfloat16 => reg.get(if (draft_only) .q3rc_mxfp8_fma__draft else .q3rc_mxfp8_fma),
                .float32 => reg.get(.q3drc_mxfp8_fma_f32x),
                else => return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: draft {t} at x {t}: the registry carries bf16 and f32 x", .{ site, x_dtype }),
            };
            return bind(g, e, site, w, scales, diag);
        }

        fn bind(g: *G, e: *const Entry, site: Site, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const s = e.site(@tagName(site)) orelse unreachable;
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            try expectInput(G, g, e, "scales", scales, &vars, diag);
            return .{ .e = e, .site = site, .plans = try .init(g, e, @tagName(site), diag), .w = g.keep(w), .scales = g.keep(scales) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
            g.release(self.scales);
        }

        /// The `QuantizedLinear.__call__` seam (these linears carry no bias; the draft's `Tr.qlinear`): x [..., G K]
        /// with 1..8 rows -> y [..., G N] at x's dtype.
        pub fn linear(self: *const Self, g: *G, x: G.T) !G.T {
            const sh = dims(G, g, x);
            const lead = sh.slice()[0 .. sh.n - 1];
            var m: c_int = 1;
            for (lead) |d| m *= d;
            const y = try self.call(g, try g.reshape(x, &.{ m, sh.slice()[sh.n - 1] }));
            var shape: Shape = .of(lead);
            shape.d[shape.n] = @intCast(self.plans.cfg[0].out_shapes[0][1]);
            shape.n += 1;
            return g.reshape(y, shape.slice());
        }

        /// x [M, G K] (row-contiguous: bf16 at the verify sites, the construction dtype at the draft's), M = 1..8 ->
        /// y [M, G N].
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, self.scales, x }, &out);
            return out[0];
        }
    };
}

// ── HCTAPE (DSV41_DECODE_HCTAPE = all) ──

/// The HC tail kernels: `HcTapeKernels` (combine, collapse_norm, combine_collapse_norm,
/// mixfin). The stream dtype is a template (OT): bf16 = the verify barrier's texts, f32 = the
/// DSpark draft stages' variants (DRAFTRC member tape; the norm weight stays bf16). The f32
/// stream's fused call on a bf16 residual is its own route (`HcTapeMixed`).
pub fn HcTape(comptime G: type) type {
    return struct {
        const Self = @This();
        combine_e: *const Entry,
        collapse_e: *const Entry,
        fused_e: *const Entry,
        mixfin_e: *const Entry,
        combine_p: RowPlans(G, 8),
        collapse_p: RowPlans(G, 8),
        fused_p: RowPlans(G, 8),
        mixfin_p: RowPlans(G, 8),

        pub fn init(g: *G, reg: *const xk.Registry, stream: Dtype, diag: ?*xk.Diag) !Self {
            const ce, const le, const fe = switch (stream) {
                .bfloat16 => .{ reg.get(.q3ht_combine), reg.get(.q3ht_collapse_norm), reg.get(.q3ht_combine_collapse_norm) },
                .float32 => .{ reg.get(.q3ht_combine__f32), reg.get(.q3ht_collapse_norm__f32), reg.get(.q3ht_combine_collapse_norm__f32) },
                else => return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: HCTAPE stream {t} (OT): the registry carries bf16 and f32", .{stream}),
            };
            const me = reg.get(.q3ht_mixfin);
            var combine_p: RowPlans(G, 8) = try .init(g, ce, null, diag);
            errdefer combine_p.deinit(g);
            var collapse_p: RowPlans(G, 8) = try .init(g, le, null, diag);
            errdefer collapse_p.deinit(g);
            var fused_p: RowPlans(G, 8) = try .init(g, fe, null, diag);
            errdefer fused_p.deinit(g);
            return .{
                .combine_e = ce,
                .collapse_e = le,
                .fused_e = fe,
                .mixfin_e = me,
                .combine_p = combine_p,
                .collapse_p = collapse_p,
                .fused_p = fused_p,
                .mixfin_p = try .init(g, me, null, diag),
            };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.combine_p.deinit(g);
            self.collapse_p.deinit(g);
            self.fused_p.deinit(g);
            self.mixfin_p.deinit(g);
        }

        /// x [M, D], r [M, 4, D], post [M, 4] f32, comb [M, 16] f32 -> h [M, 4, D].
        pub fn combine(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.combine_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb }, &out);
            return out[0];
        }

        /// s [M, 4, D], pre [M, 4] f32, w [D] -> (sf [M, 4 D] f32, ssq [M] f32, y [M, D]).
        pub fn collapseNorm(self: *const Self, g: *G, s: G.T, pre: G.T, w: G.T) ![3]G.T {
            var out: [3]G.T = undefined;
            try self.collapse_p.launch(g, rowsOf(G, g, s, 0), &.{ s, pre, w }, &out);
            return out;
        }

        /// -> (h [M, 4, D], hf [M, 4 D] f32, ssq [M] f32, y [M, D]).
        pub fn combineCollapseNorm(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T, pre: G.T, w: G.T) ![4]G.T {
            var out: [4]G.T = undefined;
            try self.fused_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb, pre, w }, &out);
            return out;
        }

        /// mm [M, 24] f32, ssq [M] f32, scale [3] f32, base [24] f32 -> (pre [M, 4], post [M, 4], comb [M, 16]) f32.
        pub fn mixfin(self: *const Self, g: *G, mm: G.T, ssq: G.T, scale: G.T, base: G.T) ![3]G.T {
            var out: [3]G.T = undefined;
            try self.mixfin_p.launch(g, rowsOf(G, g, mm, 0), &.{ mm, ssq, scale, base }, &out);
            return out;
        }
    };
}

/// The f32 stream's fused call on a bf16 residual (x f32, r [M, 4, D] bf16: the DSpark draft's
/// stage-0 first ffn prep): `q3ht_combine_collapse_norm__f32_rbf16`, its own route so that no
/// call checks which stream it serves.
pub fn HcTapeMixed(comptime G: type) type {
    return struct {
        const Self = @This();
        fused_p: RowPlans(G, 8),

        pub fn init(g: *G, reg: *const xk.Registry, diag: ?*xk.Diag) !Self {
            return .{ .fused_p = try .init(g, reg.get(.q3ht_combine_collapse_norm__f32_rbf16), null, diag) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.fused_p.deinit(g);
        }

        /// -> (h [M, 4, D] f32, hf [M, 4 D] f32, ssq [M] f32, y [M, D] f32).
        pub fn call(self: *const Self, g: *G, x: G.T, r: G.T, post: G.T, comb: G.T, pre: G.T, w: G.T) ![4]G.T {
            var out: [4]G.T = undefined;
            try self.fused_p.launch(g, rowsOf(G, g, x, 0), &.{ x, r, post, comb, pre, w }, &out);
            return out;
        }
    };
}

// ── ATTN_FUSED_PROJ (K36) ──

pub const RopeDir = enum { fwd, inv };

/// The decode / verify projection-chain glue (`deepseek_v41_fused_proj_kernels`, rows <= 8 by
/// `_fused_proj_use`): the q-latent RMSNorm, the KV RMSNorm + k_pe RoPE, the query RoPE (bf16
/// in) and the attention output's inverse RoPE (f32 in); all store bf16.
pub fn FusedProj(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        rms: *const Entry,
        rms_rope: *const Entry,
        fwd: *const Entry,
        inv: *const Entry,
        q_norm: G.T,
        kv_norm: G.T,
        rms_statics: Statics(G),
        rope_statics: Statics(G),
        ints: [max_rows]G.T,
        rms_p: RowPlans(G, max_rows),
        rms_rope_p: RowPlans(G, max_rows),
        fwd_p: RowPlans(G, max_rows),
        inv_p: RowPlans(G, max_rows),

        /// `q_norm` / `kv_norm`: the layer's q_norm (bf16 [1280]) and kv_norm (bf16 [512]) weights.
        /// `eps`: the model's rms_norm_eps. Both RMSNorm texts bind their registered eps static
        /// (the lane's, 1e-20); another value is refused here, never run.
        pub fn init(g: *G, reg: *const xk.Registry, q_norm: G.T, kv_norm: G.T, eps: f32, diag: ?*xk.Diag) !Self {
            const rms = reg.get(.mtplx_dsv41_fp_rmsnorm_tg128_d1280);
            const rms_rope = reg.get(.mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64);
            inline for (.{ rms, rms_rope }) |e| {
                const want: f32 = @floatCast(argOf(e, "eps").domain.floats[0]);
                if (eps != want) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} is registered at eps {e}, the model's is {e}", .{ e.kernel, want, eps });
            }
            try expectInput(G, g, rms, "weight", q_norm, &no_vars, diag);
            try expectInput(G, g, rms_rope, "weight", kv_norm, &no_vars, diag);
            const fwd = reg.get(.mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd);
            const inv = reg.get(.mtplx_dsv41_fp_rope_h64_hd512_rd64_inv);
            var s: Self = .{ .rms = rms, .rms_rope = rms_rope, .fwd = fwd, .inv = inv, .q_norm = undefined, .kv_norm = undefined, .rms_statics = undefined, .rope_statics = undefined, .ints = undefined, .rms_p = undefined, .rms_rope_p = undefined, .fwd_p = undefined, .inv_p = undefined };
            s.rms_p = try .init(g, rms, null, diag);
            errdefer s.rms_p.deinit(g);
            s.rms_rope_p = try .init(g, rms_rope, null, diag);
            errdefer s.rms_rope_p.deinit(g);
            s.fwd_p = try .init(g, fwd, null, diag);
            errdefer s.fwd_p.deinit(g);
            s.inv_p = try .init(g, inv, null, diag);
            errdefer s.inv_p.deinit(g);
            s.rms_statics = try Statics(G).init(g, rms);
            errdefer s.rms_statics.deinit(g);
            s.rope_statics = try Statics(G).init(g, rms_rope);
            errdefer s.rope_statics.deinit(g);
            var built: usize = 0;
            errdefer for (s.ints[0..built]) |a| g.release(a);
            for (0..max_rows) |i| {
                const v: i32 = @intCast(i + 1);
                s.ints[i] = g.keep(try g.hostArray(std.mem.asBytes(&v), &.{}, .int32));
                built += 1;
            }
            s.q_norm = g.keep(q_norm);
            s.kv_norm = g.keep(kv_norm);
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.rms_p.deinit(g);
            self.rms_rope_p.deinit(g);
            self.fwd_p.deinit(g);
            self.inv_p.deinit(g);
            self.rms_statics.deinit(g);
            self.rope_statics.deinit(g);
            for (self.ints) |a| g.release(a);
            g.release(self.q_norm);
            g.release(self.kv_norm);
        }

        fn rowsIn(g: *G, x: G.T, width: usize) !struct { Shape, u64 } {
            const sh = dims(G, g, x);
            var numel: usize = 1;
            for (sh.slice()) |v| numel *= @intCast(v);
            const rows = numel / width;
            std.debug.assert(!(rows < 1 or rows > max_rows));
            return .{ sh, rows };
        }

        /// seq rows of cos / sin (`S`; 0 rows = the call's rows).
        fn seqOf(g: *G, cos: G.T, rows: u64) !u64 {
            const s: u64 = rowsOf(G, g, cos, 0);
            const seq = if (s == 0) rows else s;
            std.debug.assert(!(seq > max_rows));
            return seq;
        }

        /// `rmsnorm(wq_a(x), q_norm_weight, eps)`: x [..., 1280] bf16 -> x's shape, bf16.
        pub fn qNorm(self: *const Self, g: *G, x: G.T) !G.T {
            const sh, const rows = try rowsIn(g, x, 1280);
            var out: [1]G.T = undefined;
            try self.rms_p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 1280 }), self.q_norm, self.rms_statics.arrays[2] }, &out);
            return g.reshape(out[0], sh.slice());
        }

        /// `rmsnorm_rope(wkv(x), kv_norm_weight, eps, cos, sin)`: x [..., 512] bf16, cos / sin f32 [S, 32].
        pub fn kvNormRope(self: *const Self, g: *G, x: G.T, cos: G.T, sin: G.T) !G.T {
            const sh, const rows = try rowsIn(g, x, 512);
            const seq = try seqOf(g, cos, rows);
            var out: [1]G.T = undefined;
            try self.rms_rope_p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 512 }), self.kv_norm, self.rope_statics.arrays[2], cos, sin, self.ints[seq - 1] }, &out);
            return g.reshape(out[0], sh.slice());
        }

        /// `rope_heads(x [..., 64, 512], cos, sin, inverse)` -> x's shape, bf16.
        pub fn ropeHeads(self: *const Self, g: *G, x: G.T, cos: G.T, sin: G.T, dir: RopeDir) !G.T {
            const sh, const rows = try rowsIn(g, x, 64 * 512);
            const seq = try seqOf(g, cos, rows);
            var out: [1]G.T = undefined;
            const p = if (dir == .fwd) &self.fwd_p else &self.inv_p;
            try p.launch(g, rows, &.{ try g.reshape(x, &.{ @intCast(rows), 64, 512 }), cos, sin, self.ints[rows - 1], self.ints[seq - 1] }, &out);
            return g.reshape(out[0], sh.slice());
        }
    };
}

// ── Decode batch 2: the texts the RC tiers of record still run beside the RC routes ──

/// The native pipeline's wo_a expansion (`fused_transpose.make_transpose`): one layer's packed
/// mxfp8 wo_a -> bf16 [8, 4096, 1024], the layout the o-LoRA matmul reads. On the RC tiers
/// (RCPROJ woarc) the decode ring is never filled: a prefill-sized M expands the layer's wo_a on
/// demand and matmuls (`make_woarc_call`). One fixed launch, built once.
pub fn WoaRingTranspose(comptime G: type) type {
    const prepared = declaresFn(G, "prepareLaunch");
    return struct {
        const Self = @This();
        e: *const Entry,
        cfg: LaunchConfig,
        prep: if (prepared) G.Prepared else void,

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const e = reg.get(.dsv41_woa_decode_transpose_32);
            var s: Self = .{ .e = e, .cfg = xk.launchFor(e, &no_vars, null) catch unreachable, .prep = undefined };
            if (prepared) s.prep = try g.prepareLaunch(e.kernel, &s.cfg);
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            if (prepared) g.releasePrepared(&self.prep);
        }

        /// A layer's pair, checked once when the layer binds it (not per call).
        pub fn checkLayer(self: *const Self, g: *G, packed_w: G.T, scales: G.T, diag: ?*xk.Diag) Refusal!void {
            try expectInput(G, g, self.e, "packed", packed_w, &no_vars, diag);
            try expectInput(G, g, self.e, "scales", scales, &no_vars, diag);
        }

        /// packed u32 [8192, 1024] (e4m3 codes, 4 per word), scales u8 [8192, 128] (e8m0) ->
        /// bf16 [8, 4096, 1024] = T(scale * e4m3(code)), transposed per o-LoRA group.
        pub fn call(self: *const Self, g: *G, packed_w: G.T, scales: G.T) !G.T {
            var out: [1]G.T = undefined;
            if (prepared) {
                try g.launchPrepared(&self.prep, &.{ packed_w, scales }, &out);
            } else {
                try g.launch(self.e.kernel, &.{ packed_w, scales }, &self.cfg, &out);
            }
            return out[0];
        }
    };
}

/// DSV41_INDEX_TOPK=metal (`q3_indextopk_candidate.metal_select`): the DSA indexer's row
/// selection in one dispatch (radix select + one ascending emit pass), decode / verify rows and
/// the prefill rows of the rewritten `_q3_ph_select` (ATTNHALF idxscore: topkgeom is not on the
/// tier). k = min(512, N) and the output width = k on the tier (ATTN_CORE_COMPILE / K29 off); the
/// flag is k >= N. N (the compressed count) grows during decode, so each call builds its launch
/// config and its N / k scalars (the lane's own per-call ints); the 0 / 1 flags are built once.
/// The model geometry the prefill-attention, indexer and combine texts were derived for (the lane's
/// DeepSeek-V4.1 shapes, baked into their texts and launch rules). A route is built only for a model
/// whose config matches, field by field; any other is refused at construction, by name, never run.
pub const PrefillGeometry = struct {
    n_heads: u32,
    head_dim: u32,
    rope_head_dim: u32,
    window: u32,
    index_topk: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    n_experts_per_tok: u32,
    hidden: u32,
    hc_mult: u32,

    pub const derived: PrefillGeometry = .{ .n_heads = 64, .head_dim = 512, .rope_head_dim = 64, .window = 128, .index_topk = 512, .index_n_heads = 32, .index_head_dim = 128, .n_experts_per_tok = 6, .hidden = 5120, .hc_mult = 4 };

    /// `what` names the route; every field is compared (a subset would let a shape through).
    pub fn admit(got: *const PrefillGeometry, what: []const u8, diag: ?*xk.Diag) Refusal!void {
        inline for (comptime std.meta.fieldNames(PrefillGeometry)) |name| {
            const want = @field(derived, name);
            const have = @field(got.*, name);
            if (have != want) return refuse(diag, error.RouteInput, "exl3 kernel ops: {s} is derived for {s} {d}, the model's is {d}", .{ what, name, want, have });
        }
    }
};

pub fn IndexTopk(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const index_topk = 512;
        e: *const Entry,
        flag: [2]G.T,

        pub fn init(g: *G, reg: *const xk.Registry, geo: *const PrefillGeometry, diag: ?*xk.Diag) !Self {
            try geo.admit("mtplx_dsv41_index_topk_select", diag);
            var s: Self = .{ .e = reg.get(.mtplx_dsv41_index_topk_select), .flag = undefined };
            var built: usize = 0;
            errdefer for (s.flag[0..built]) |x| g.release(x);
            for (&s.flag, 0..) |*f, i| {
                const v: i32 = @intCast(i);
                f.* = g.keep(try g.hostArray(std.mem.asBytes(&v), &.{}, .int32));
                built += 1;
            }
            return s;
        }

        pub fn deinit(self: *Self, g: *G) void {
            for (self.flag) |x| g.release(x);
        }

        /// score f32 [M, N] (the indexer's final scores, -inf outside reach / candidates), clen
        /// int32 [M] -> .{ sel int32 [M, k] (ascending selected indices, -1 padded), mask bool
        /// [M, N] }, M >= 1 (decode / verify or a prefill chunk), N >= 1.
        pub fn select(self: *const Self, g: *G, score: G.T, clen: G.T) ![2]G.T {
            const sh = dims(G, g, score);
            std.debug.assert(!(sh.n != 2));
            const rows: u64 = @intCast(sh.d[0]);
            const n: u64 = @intCast(sh.d[1]);
            std.debug.assert(!(rows < 1));
            std.debug.assert(!(n < 1));
            const k = @min(index_topk, n);
            var vars: Vars = .initFill(0);
            vars.set(.rows, rows);
            vars.set(.ncomp, n);
            vars.set(.topk, k);
            vars.set(.width, k);
            const cfg = xk.launchFor(self.e, &vars, null) catch unreachable;
            const n32: i32 = @intCast(n);
            const k32: i32 = @intCast(k);
            const n_arr = try g.hostArray(std.mem.asBytes(&n32), &.{}, .int32);
            const k_arr = try g.hostArray(std.mem.asBytes(&k32), &.{}, .int32);
            var out: [2]G.T = undefined;
            try g.launch(self.e.kernel, &.{ score, clen, n_arr, k_arr, k_arr, self.flag[@intFromBool(k >= n)] }, &cfg, &out);
            return out;
        }
    };
}

/// DSV41_ATTN_FUSE=softmax (`q3_attnfuse_candidate.make_fused`): the verify attention's scale /
/// mask / sink-softmax chain in one kernel, one threadgroup per (row, head); ls = MLX's
/// row_reduce_simple threadgroup (32 for k <= 512, 128 for 512 < k <= 1024). The tier's two key
/// counts (128 on the window-only layers, 640 elsewhere) launch from per-M tables built here;
/// another k in (64, 1024] (warm-up, fewer compressed rows than index_topk) builds its launch per
/// call. Another k is the stock chain's (the lane's route); rows > 8 are the prefill cores'
/// (`PrefillAttn`, the same entries per call).
pub fn AttnSoftmax(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        pub const heads = 64;
        ls32: *const Entry,
        ls128: *const Entry,
        statics: Statics(G),
        k128: RowPlans(G, max_rows),
        k640: RowPlans(G, max_rows),

        pub fn init(g: *G, reg: *const xk.Registry, diag: ?*xk.Diag) !Self {
            const ls32 = reg.get(.q3_attnfuse_softmax);
            const ls128 = reg.get(.q3_attnfuse_softmax__ls128);
            var v128: Vars = .initFill(0);
            v128.set(.keys, 128);
            var v640: Vars = .initFill(0);
            v640.set(.keys, 640);
            var k128: RowPlans(G, max_rows) = try .initAt(g, ls32, null, &v128, diag);
            errdefer k128.deinit(g);
            var k640: RowPlans(G, max_rows) = try .initAt(g, ls128, null, &v640, diag);
            errdefer k640.deinit(g);
            return .{ .ls32 = ls32, .ls128 = ls128, .statics = try Statics(G).init(g, ls32), .k128 = k128, .k640 = k640 };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.k128.deinit(g);
            self.k640.deinit(g);
            self.statics.deinit(g);
        }

        /// qk f32 [1, M, 64, k] (the UNSCALED QK), valid bool [1, M, k] (a strided view is
        /// fine), sink f32 [1, 1, 64, 1] -> .{ ex f32 [1, M, 64, k], denom f32 [1, M, 64, 1] }.
        pub fn call(self: *const Self, g: *G, qk: G.T, valid: G.T, sink: G.T) ![2]G.T {
            const sh = dims(G, g, qk);
            std.debug.assert(!(sh.n != 4));
            const rows: u64 = @intCast(sh.d[1]);
            const k: u64 = @intCast(sh.d[3]);
            std.debug.assert(!(rows < 1 or rows > max_rows));
            const ins = [_]G.T{ qk, valid, sink, self.statics.arrays[3] };
            var out: [2]G.T = undefined;
            if (k == 128) {
                try self.k128.launch(g, rows, &ins, &out);
            } else if (k == 640) {
                try self.k640.launch(g, rows, &ins, &out);
            } else if (k > 64 and k <= 1024) {
                const e = if (k <= 512) self.ls32 else self.ls128;
                var vars: Vars = .initFill(0);
                vars.set(.rows, rows);
                vars.set(.keys, k);
                const cfg = xk.launchFor(e, &vars, null) catch unreachable;
                try g.launch(e.kernel, &ins, &cfg, &out);
            } else return error.KeysOutOfPlan;
            return out;
        }
    };
}

/// The minvariant mxfp8 member's sites the RC tiers still run (RCPROJ mxfp8 owns wq_a / wkv /
/// wq_b / wo_b there): the indexer wq_b, the shared expert, the Engram wkv.
pub const M1Site = enum { indexer_wq_b, shared_w1_w3, shared_w2, engram_wkv };

/// DSV41_MXFP8_ROWS=m1order at one site (`M1Rows.run`): x [M, K] bf16 -> y [M, N] bf16, each
/// row in MLX's M = 1 mxfp8_qmv_fast order, at the lane's pinned (R, V) per M (M = 1: R 4, V 1).
pub fn Mxfp8Rows(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        site: M1Site,
        plans: RowPlans(G, max_rows),
        w: G.T,
        scales: G.T,

        /// `w` the packed mxfp8 weight (u32 [N, K / 4]), `scales` its e8m0 scales (u8 [N, K / 32]).
        pub fn init(g: *G, reg: *const xk.Registry, site: M1Site, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const e = reg.get(.dsv41_mxfp8_m1rows);
            const s = e.site(@tagName(site)) orelse unreachable;
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            try expectInput(G, g, e, "scales", scales, &vars, diag);
            return .{ .e = e, .site = site, .plans = try .init(g, e, @tagName(site), diag), .w = g.keep(w), .scales = g.keep(scales) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
            g.release(self.scales);
        }

        /// x [M, K] bf16 (row-contiguous), M = 1..8 -> y [M, N] bf16.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, self.scales, x }, &out);
            return out[0];
        }
    };
}

/// The minvariant woa_head member's head route (`Kernels.run_head`): the bf16 output head at
/// M = 1..8 in MLX's M = 1 gemv order (the verify's M 2..8; the draft head's W103 call). RCTAIL
/// headpad (M 5 / 7 -> the same call on M + 1 rows, a zero row appended, the first M kept) is
/// the caller's.
pub fn HeadRows(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        plans: RowPlans(G, max_rows),
        w: G.T,

        /// `w` the bf16 head weight [129280, 5120].
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, diag: ?*xk.Diag) !Self {
            const e = reg.get(.dsv41_head_m1rows);
            try expectInput(G, g, e, "w", w, &no_vars, diag);
            return .{ .e = e, .plans = try .init(g, e, "", diag), .w = g.keep(w) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
        }

        /// x [M, 5120] bf16, M = 1..8 -> y [M, 129280] bf16.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, x }, &out);
            return out[0];
        }
    };
}

/// HEAD_MODE mxfp8 on RCPROJ's `head` site: the quantized output head at M = 1..8 (the verify rows and
/// the draft block), the head's packed codes and e8m0 scales as MLX's mxfp8 quantize lays them out.
pub fn HeadMx(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        plans: RowPlans(G, max_rows),
        w: G.T,
        scales: G.T,

        /// `w` u32 [129280, 1280], `scales` u8 [129280, 160] (checked against the site here, once).
        pub fn init(g: *G, reg: *const xk.Registry, w: G.T, scales: G.T, diag: ?*xk.Diag) !Self {
            const e = reg.get(.q3rc_mxfp8_fma);
            const s = e.site("head") orelse return refuse(diag, error.RouteInput, "exl3 kernel ops: q3rc_mxfp8_fma has no head site", .{});
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            try expectInput(G, g, e, "scales", scales, &vars, diag);
            return .{ .e = e, .plans = try .init(g, e, "head", diag), .w = g.keep(w), .scales = g.keep(scales) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
            g.release(self.scales);
        }

        /// x [M, 5120] bf16, M = 1..8 -> y [M, 129280] bf16.
        pub fn call(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, x, 0), &.{ self.w, self.scales, x }, &out);
            return out[0];
        }
    };
}

/// The minvariant attention member's sites the RC tiers still run (RCTAIL owns the HC premix and
/// the MoE gate there): the compressor wkv / wgate (f32 x on the ratio-2 layers, bf16 on ratio 1),
/// the indexer wk (f32 / bf16 latent) and the indexer weights_proj (bf16).
pub const SmallMSite = enum { cmp_f32, wk_f32, cmp_bf16, wk_bf16, wproj };

/// DSV41_ATTN_KERNEL=smallm_all at one site (`SmallMAll.run`): a [M, K] -> out [M, N], each row
/// in MLX's M = 1 gemv order; f32 x sites on dsv41_smallm_all (f32 out), bf16 x sites on its
/// __bf16 variant (bf16 out); w [N, K] bf16.
pub fn SmallM(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const max_rows = 8;
        e: *const Entry,
        site: SmallMSite,
        plans: RowPlans(G, max_rows),
        w: G.T,

        pub fn init(g: *G, reg: *const xk.Registry, site: SmallMSite, w: G.T, diag: ?*xk.Diag) !Self {
            const e = switch (site) {
                .cmp_f32, .wk_f32 => reg.get(.dsv41_smallm_all),
                .cmp_bf16, .wk_bf16, .wproj => reg.get(.dsv41_smallm_all__bf16),
            };
            const s = e.site(@tagName(site)) orelse unreachable;
            var vars: Vars = .initFill(0);
            xk.siteVars(s, &vars);
            try expectInput(G, g, e, "w", w, &vars, diag);
            return .{ .e = e, .site = site, .plans = try .init(g, e, @tagName(site), diag), .w = g.keep(w) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.plans.deinit(g);
            g.release(self.w);
        }

        /// a [M, K] (the site's x dtype), M = 1..8 -> out [M, N].
        pub fn call(self: *const Self, g: *G, a: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.plans.launch(g, rowsOf(G, g, a, 0), &.{ a, self.w }, &out);
            return out[0];
        }
    };
}

// ── Prefill batch 2: the P line's prefill-rows texts (ATTNHALF, ATTN hcnorm, SMALLK) ──
//
// Each call launches at its own sizes (the chunk's rows, the context's store rows and key
// count): a per-call launch config, as the other prefill routes. The instantiation (dtype set)
// and the model constants are chosen and checked at construction.

/// ATTNHALF idxscore (`q3_prefill_attnhalf_candidate.make_ops(..)['idxscore']`, the rewritten
/// `_q3_ph_select` at rows > 8): out[s, n] = sum_h relu(q_h . k_n) w_h where n < clen[s], else
/// -inf, in one kernel (MLX's gemm_loop + the topksum epilogue). The lane casts q and index_k to
/// f32 before the launch (exact widenings); INDEX_TOPK's select then takes out.
pub fn IdxScore(comptime G: type) type {
    return struct {
        const Self = @This();
        e: *const Entry,

        pub fn init(reg: *const xk.Registry, geo: *const PrefillGeometry, diag: ?*xk.Diag) Refusal!Self {
            try geo.admit("q3_ph_index_score", diag);
            return .{ .e = reg.get(.q3_ph_index_score) };
        }

        /// q [1, S, 32, 128] (the roped indexer q, any float dtype), index_k [1, N, 128] (any float
        /// dtype), w f32 [1, S, 32] (the head weights), clen int32 [S] (reach) -> f32 [1, S, N].
        pub fn call(self: *const Self, g: *G, q: G.T, index_k: G.T, w: G.T, clen: G.T) !G.T {
            var vars: Vars = .initFill(0);
            vars.set(.rows, rowsOf(G, g, q, 1));
            vars.set(.ncomp, rowsOf(G, g, index_k, 1));
            const q32 = try g.astype(q, .float32);
            const k32 = try g.astype(index_k, .float32);
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.e, &vars, &.{ q32, k32, w, clen }, &out);
            return out[0];
        }
    };
}

/// The two prefill attention cores: corevec (the attncore core at rows 9..32) and ropefuse
/// (`_attend` at rows > 32: q roped in the QK load, o inverse-roped into the o-LoRA group layout).
pub const CoreKind = enum { vec, rope };

/// A text's instantiation at template dtypes (TQ when the text has one, TKV): the base or one of
/// its variants; null when the registry carries none (the lane never warmed it).
fn instantiation(reg: *const xk.Registry, base: Kernel, tq: Dtype, tkv: Dtype) ?*const Entry {
    for (&reg.entries) |*e| {
        if ((e.variant_of orelse e.kernel) != base) continue;
        var ok = true;
        for (e.template) |x| {
            if (std.mem.eql(u8, x.name, "TQ")) ok = ok and x.value.dtype == tq;
            if (std.mem.eql(u8, x.name, "TKV")) ok = ok and x.value.dtype == tkv;
        }
        if (ok) return e;
    }
    return null;
}

/// ATTNHALF corevec / ropefuse (`make_vec_core` / `make_rope_core` over the lane's kernels): the
/// selected-keys attention of one layer at prefill rows as three launches, QK (the unscaled
/// scores + the valid mask over the 128 window keys and, on a compressed layer, the kc selected
/// compressed keys), ATTN_FUSE's softmax (scale, mask, sink; ls 32 up to 512 keys, 128 above) and
/// PV. One route per layer kind, built at construction: the query dtype, the window store's dtype
/// (bf16 on layer 0, f32 on the others) and compressed or window-only.
pub fn PrefillAttn(comptime G: type) type {
    return struct {
        const Self = @This();
        pub const heads = 64;
        pub const head_dim = 512;
        pub const window = 128;
        pub const max_kc = 512;
        kind: CoreKind,
        cmp: bool,
        qk: *const Entry,
        pv: *const Entry,
        s32: *const Entry,
        s128: *const Entry,
        statics: Statics(G),

        /// `q` the query dtype (bf16 on the tier), `ring` the window store's dtype, `cmp` a
        /// compressed layer (its store f32). A dtype set the lane does not warm is refused
        /// (TemplateNotRegistered).
        pub fn init(g: *G, reg: *const xk.Registry, geo: *const PrefillGeometry, kind: CoreKind, q: Dtype, ring: Dtype, cmp: bool, diag: ?*xk.Diag) !Self {
            try geo.admit(if (kind == .rope) "the ropefuse prefill core" else "the corevec prefill core", diag);
            const Stage = struct { vec: Kernel, rope: Kernel };
            const qk_base: Stage = if (cmp) .{ .vec = .q3_ph_qkvec_cmp, .rope = .q3_ph_qkrope_cmp } else .{ .vec = .q3_ph_qkvec_win, .rope = .q3_ph_qkrope_win };
            const pv_base: Stage = if (cmp) .{ .vec = .q3_ph_pvvec_cmp, .rope = .q3_ph_pvrope_cmp } else .{ .vec = .q3_ph_pvvec_win, .rope = .q3_ph_pvrope_win };
            const qb, const pb = switch (kind) {
                .vec => .{ qk_base.vec, pv_base.vec },
                .rope => .{ qk_base.rope, pv_base.rope },
            };
            const qk = instantiation(reg, qb, q, ring) orelse return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: {t} has no instantiation at q {t}, window store {t}", .{ qb, q, ring });
            const pv = instantiation(reg, pb, q, ring) orelse return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: {t} has no instantiation at window store {t}", .{ pb, ring });
            const s32 = reg.get(.q3_attnfuse_softmax);
            return .{ .kind = kind, .cmp = cmp, .qk = qk, .pv = pv, .s32 = s32, .s128 = reg.get(.q3_attnfuse_softmax__ls128), .statics = try Statics(G).init(g, s32) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.statics.deinit(g);
        }

        /// q [1, S, 64, 512] (UN-roped for rope), win [1, T, 512] (the window store), widx int32 /
        /// wval bool [S, 128] (Attention._window_selected_idx), on a compressed layer
        /// `cmp_kv` = .{ ckv f32 [1, Nc, 512], cidx int32 [1, S, kc] (-1 = no key) }, sink f32
        /// [1, 1, 64, 1], for rope `rope` = .{ qcos, qsin } f32 [S, 32] -> vec: o f32 [1, S, 64,
        /// 512]; rope: o f32 [8, S, 4096] (the lane hands on its [1, S, 8, 4096] transposed view).
        pub fn attend(self: *const Self, g: *G, q: G.T, win: G.T, widx: G.T, wval: G.T, cmp_kv: ?[2]G.T, sink: G.T, rope: ?[2]G.T) !G.T {
            std.debug.assert(!((cmp_kv != null) != self.cmp or (rope != null) != (self.kind == .rope)));
            const s = rowsOf(G, g, q, 1);
            var vars: Vars = .initFill(0);
            vars.set(.rows, s);
            vars.set(.ring, rowsOf(G, g, win, 1));
            var k: u64 = window;
            if (cmp_kv) |c| {
                const kc = rowsOf(G, g, c[1], 2);
                if (kc < 1 or kc > max_kc) return error.KeysOutOfPlan;
                vars.set(.store, rowsOf(G, g, c[0], 1));
                vars.set(.kc, kc);
                k += kc;
            }
            // the lane: contiguous(broadcast_to(idx[None], (1, s, W))), a reshape at b = 1
            const sw = [_]c_int{ 1, @intCast(s), window };
            const widx3 = try g.reshape(widx, &sw);
            const wval3 = try g.reshape(wval, &sw);
            var kv: [7]G.T = undefined; // win, widx, wval (+ ckv, cidx) (+ qcos, qsin)
            var n_kv: usize = 0;
            for ([_]G.T{ win, widx3, wval3 }) |x| {
                kv[n_kv] = x;
                n_kv += 1;
            }
            if (cmp_kv) |c| for (c) |x| {
                kv[n_kv] = x;
                n_kv += 1;
            };
            if (rope) |r| for (r) |x| {
                kv[n_kv] = x;
                n_kv += 1;
            };
            var ins: [9]G.T = undefined; // q | ex, denom; then the kv operands
            ins[0] = q;
            @memcpy(ins[1 .. 1 + n_kv], kv[0..n_kv]);
            var sv: [2]G.T = undefined;
            try launchRule(G, g, self.qk, &vars, ins[0 .. 1 + n_kv], &sv);
            var soft_vars: Vars = .initFill(0);
            soft_vars.set(.rows, s);
            soft_vars.set(.keys, k);
            var ed: [2]G.T = undefined;
            try launchRule(G, g, if (k <= 512) self.s32 else self.s128, &soft_vars, &.{ sv[0], sv[1], sink, self.statics.arrays[3] }, &ed);
            ins[0] = ed[0];
            ins[1] = ed[1];
            @memcpy(ins[2 .. 2 + n_kv], kv[0..n_kv]);
            var o: [1]G.T = undefined;
            try launchRule(G, g, self.pv, &vars, ins[0 .. 2 + n_kv], &o);
            return o[0];
        }
    };
}

/// ATTN hcnorm (`q3_prefill_attn_candidate.make_ops(..)['hc_rsqrt' / 'hc_pre_norm']`, rows >= 32):
/// DecoderLayer._mixes' rsqrt(mean(square(flat)) + eps) and _rmsnorm(_hc_pre(h, pre), w, eps), one
/// kernel each in the stock reduction order. The HC stream's dtype and the model's eps are fixed at
/// construction.
pub fn HcNorm(comptime G: type) type {
    return struct {
        const Self = @This();
        rsq: *const Entry,
        pre: *const Entry,
        rsq_st: Statics(G),
        pre_st: Statics(G),

        /// `stream` the HC state's dtype (bf16 on the tier, f32 for an f32 stream); `eps` the
        /// model's rms_norm_eps: another value than the registered constant is refused.
        pub fn init(g: *G, reg: *const xk.Registry, geo: *const PrefillGeometry, stream: Dtype, eps: f32, diag: ?*xk.Diag) !Self {
            try geo.admit("the pf_hc norms", diag);
            const rsq, const pre = switch (stream) {
                .bfloat16 => .{ reg.get(.q3pf_hc_mix_rsqrt), reg.get(.q3pf_hc_pre_norm) },
                .float32 => .{ reg.get(.q3pf_hc_mix_rsqrt__f32), reg.get(.q3pf_hc_pre_norm__f32) },
                else => return refuse(diag, error.TemplateNotRegistered, "exl3 kernel ops: the HC norms carry a bf16 or an f32 stream, not {t}", .{stream}),
            };
            const want: f32 = @floatCast(argOf(rsq, "eps").domain.floats[0]);
            if (eps != want) return refuse(diag, error.RouteInput, "exl3 kernel ops: {t} is registered at eps {e}, the model's is {e}", .{ rsq.kernel, want, eps });
            var rsq_st = try Statics(G).init(g, rsq);
            errdefer rsq_st.deinit(g);
            return .{ .rsq = rsq, .pre = pre, .rsq_st = rsq_st, .pre_st = try Statics(G).init(g, pre) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.rsq_st.deinit(g);
            self.pre_st.deinit(g);
        }

        /// x [1, S, 4, 5120] (the stream dtype) -> r f32 [1, S, 1] = rsqrt(mean(x^2) + eps).
        pub fn rsqrt(self: *const Self, g: *G, x: G.T) !G.T {
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.rsq, &rowsVars(rowsOf(G, g, x, 1)), &.{ x, self.rsq_st.arrays[1], self.rsq_st.arrays[2] }, &out);
            return out[0];
        }

        /// h [1, S, 4, 5120] (the stream dtype), pre f32 [1, S, 4], w bf16 [5120] (the layer's
        /// attention or MoE norm weight) -> y [1, S, 5120] (the stream dtype).
        pub fn preNorm(self: *const Self, g: *G, h: G.T, pre: G.T, w: G.T) !G.T {
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.pre, &rowsVars(rowsOf(G, g, h, 1)), &.{ h, pre, w, self.pre_st.arrays[3], self.pre_st.arrays[4] }, &out);
            return out[0];
        }
    };
}

/// SMALLK combine (`q3_prefill_smallk_candidate.CombineKernel`, `_moe_combine_dispatch` at rows >
/// 32): (routed * w) summed over the 6 routed experts in col_reduce_small order, + shared, one f32
/// kernel.
pub fn SmallKCombine(comptime G: type) type {
    return struct {
        const Self = @This();
        e: *const Entry,

        /// kv16-opt: the same text over bf16 routed rows (the DIG-X waves' bf16 expert outputs)
        bf16: *const Entry,

        pub fn init(reg: *const xk.Registry, geo: *const PrefillGeometry, diag: ?*xk.Diag) Refusal!Self {
            try geo.admit("q3sk_combine", diag);
            return .{ .e = reg.get(.q3sk_combine), .bf16 = reg.get(.q3sk_combine__rbf16) };
        }

        /// routed f32 or bf16 [n, 6, 5120] (the unweighted expert outputs, read as f32), weights f32 [n, 6], shared
        /// f32 [n, 5120] -> f32 [n, 5120].
        pub fn call(self: *const Self, g: *G, routed: G.T, weights: G.T, shared: G.T) !G.T {
            var out: [1]G.T = undefined;
            try launchRule(G, g, if (g.dtypeOf(routed) == .bfloat16) self.bf16 else self.e, &rowsVars(rowsOf(G, g, routed, 0)), &.{ routed, weights, shared }, &out);
            return out[0];
        }
    };
}

/// PREFILL_HCPOST: the Hyper-Connection combine (`_hc_post_impl`) at prompt widths in one pass, on the compiled HcPost
/// region's words: the einsum's NAX f32 GEMM numerics (TF32-truncated operands, each product flushed to a signed zero
/// below 2^-126, the K = 4 terms summed in pairs) and the region's tail (post x, then + mixed). Exactness domain
/// (kbench hcpostx2553b18r2): word for word on every normal-range and mixed-edge input tested; an input whose products
/// all fall below 2^-126 is not covered (not seen: comb >= ~1e-7). The model checks it against the region at
/// construction.
pub fn HcPostTf32(comptime G: type) type {
    return struct {
        const Self = @This();
        f32_res: *const Entry,
        bf16_res: *const Entry,
        /// kv16-opt: x, res and h in the stream's bf16 (the same text: x read through static_cast<float>, h stored as
        /// the f32 word rounded to bfloat, the region's `astype(.., x dtype)`).
        bf16: *const Entry,

        pub fn init(reg: *const xk.Registry, geo: *const PrefillGeometry, diag: ?*xk.Diag) Refusal!Self {
            try geo.admit("dsv41_hcpost_tf32", diag);
            return .{ .f32_res = reg.get(.dsv41_hcpost_tf32), .bf16_res = reg.get(.dsv41_hcpost_tf32__rbf16), .bf16 = reg.get(.dsv41_hcpost_tf32_bf16) };
        }

        /// x [n, 5120], res [n, 4, 5120], post f32 [n, 4], comb f32 [n, 16] (`...jk` flattened) -> [n, 4, 5120]: x f32
        /// with res f32 or bf16 -> f32 h; x and res bf16 (kv16's stream) -> bf16 h.
        pub fn call(self: *const Self, g: *G, x: G.T, res: G.T, post: G.T, comb: G.T) !G.T {
            const e = if (g.dtypeOf(x) == .bfloat16) self.bf16 else if (g.dtypeOf(res) == .bfloat16) self.bf16_res else self.f32_res;
            var out: [1]G.T = undefined;
            try launchRule(G, g, e, &rowsVars(rowsOf(G, g, x, 0)), &.{ x, res, post, comb }, &out);
            return out[0];
        }
    };
}

/// JOINLESS (`q3_prefill_joinless_candidate.JoinlessKernel`): SMALLK's combine reading each routed row
/// from the fused call output that computed it, through a per-assignment (source, row) table, so the
/// joined [n, 6, 5120] routed array is never built. Exact (the same f32 words, SMALLK's fold order).
pub fn JoinlessCombine(comptime G: type) type {
    return struct {
        const Self = @This();
        /// the text's source slots (the lane's NSRC); unused slots alias source 0
        pub const sources = 24;
        e: *const Entry,
        /// kv16-opt: the text over bf16 sources (the DIG-X waves' bf16 expert outputs; the same f32 arithmetic)
        bf16: *const Entry,

        pub fn init(reg: *const xk.Registry, geo: *const PrefillGeometry, diag: ?*xk.Diag) Refusal!Self {
            try geo.admit("q3jl_combine", diag);
            return .{ .e = reg.get(.q3jl_combine), .bf16 = reg.get(.dsv41_jl_combine_bf16) };
        }

        /// outs: the layer's routed outputs as sources (1..24, each f32 [r_i, 5120]; a layer with more
        /// merges its smallest first, `deepseek_v41_experts.mergeJoinless`), loc int32 [n, 6, 2] (the
        /// (source, row) of each assignment), weights f32 [n, 6], shared f32 [n, 5120] -> f32 [n, 5120].
        pub fn call(self: *const Self, g: *G, outs: []const G.T, loc: G.T, weights: G.T, shared: G.T) !G.T {
            std.debug.assert(!(outs.len == 0 or outs.len > sources));
            var ins: [sources + 3]G.T = undefined;
            for (ins[0..sources], 0..) |*x, i| x.* = outs[if (i < outs.len) i else 0];
            ins[sources..].* = .{ loc, weights, shared };
            var vars = rowsVars(rowsOf(G, g, weights, 0));
            vars.set(.src, rowsOf(G, g, outs[0], 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, if (g.dtypeOf(outs[0]) == .bfloat16) self.bf16 else self.e, &vars, &ins, &out);
            return out[0];
        }
    };
}

// ── Tests ──

const testing = std.testing;
const kt = sdk_ext.kernels.Trace(xk);
const Trace = kt.Trace;
const testRegistry = kt.testRegistry;
const expectLaunch = kt.expectLaunch;
const expectPreparedDecode = kt.expectPreparedDecode;
const sampleAt = kt.sampleAt;
const sampleWith = kt.sampleWith;
const isDecode2 = kt.isDecode2;
const isPrefill2 = kt.isPrefill2;
const templateInt = kt.templateInt;
