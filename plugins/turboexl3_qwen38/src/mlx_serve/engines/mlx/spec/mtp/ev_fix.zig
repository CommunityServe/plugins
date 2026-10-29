//! copied_from: src/engines/mlx/spec/mtp/ev_fix.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Two-chunk MTP plans priced at their chunk-A boundary: a plan keeps its second chunk only when it beats every single
//! chunk it spans, with the boundary at this model's measured price. Only the qwen4_exp head over an EXL3 pack
//! (`MtpHeadRef.exl3Profile`) uses it. It plans only: the verify still commits the target's own tokens.
const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");
const round_cost = @import("../round_cost.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const MtpCostSource = @import("ev_plan.zig").MtpCostSource;
const MtpEvCosts = @import("ev_costs.zig").MtpEvCosts;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;
const mtpEvExpectedTokens = @import("ev_plan.zig").mtpEvExpectedTokens;
const MTP_EV_COST_BETA = @import("regime.zig").MTP_EV_COST_BETA;

/// What two-chunk rounds cost and returned on this model (`Transformer.mtp_ev_fix`), fed by solo rounds of kept plans.
/// It lives on the model like `round_cost`, so the warm-up's measurements price the next request from its first round.
pub const Model = struct {
    /// EMA (ms) of one boundary as the round sees it: the confidence read plus, when the gate fires, chunk B's build
    /// and dispatch. 0 = never measured.
    read_ms: f32 = 0,
    /// A two-chunk round's inter-round wall over the table's single chunk at the width it drafted, on steady-shape
    /// rounds: a running mean, then an EMA. It can sit below the read: the GPU runs chunk A during the read either way.
    excess_ms: f32 = 0,
    excess_n: u32 = 0,
    /// Decayed sums: extension tokens realized vs promised by the ideal gate (extend exactly when chunk A lands), and
    /// extensions fired vs that gate's firing probability.
    gain_real: f32 = 0,
    gain_ideal: f32 = 0,
    fire_real: f32 = 0,
    fire_ideal: f32 = 0,
    /// The first collapse to one chunk is logged once per model.
    logged: bool = false,
};

/// This request's record (`Generator.mtp_ev_fix`).
pub const Request = struct {
    /// Two-chunk plans priced, and how many of them became one chunk (`[spec-stats] ev_fix`).
    priced: u32 = 0,
    collapsed: u32 = 0,
    /// The last round's inter-round wall when it is a clean boundary sample (a running clock, the same shape as the
    /// two rounds before it), else 0. Written by `mtpRoundEndObserve`, read by `mtpEvFixObserve`.
    wall_ms: f32 = 0,
};

/// What a priced plan reads besides the acceptance EMAs and the cost source.
pub const Inputs = struct {
    /// One boundary in the cost source's floor units (`boundaryCost`).
    boundary: f32,
    /// The gate's realized share of the ideal gate's extension tokens and firings (`ratios`); 1 = ideal.
    gain_ratio: f32 = 1.0,
    fire_ratio: f32 = 1.0,
};

/// Steady two-chunk rounds the realized excess needs before it replaces the read as the boundary's price.
pub const EXCESS_MIN: u32 = 8;

/// Ideal-gate evidence the realized ratios start from: a gate that extends on the wrong rounds prices itself out over
/// a few rounds, not on one unlucky one.
pub const PRIOR: f32 = 2.0;

pub const Ratios = struct { gain: f32, fire: f32 };

/// The boundary's price in ms: the realized excess once steady rounds measured it (floored at 0), else the read as the
/// round sees it, else null (never measured).
pub fn boundaryMs(st: Model) ?f32 {
    if (st.excess_n >= EXCESS_MIN) return @max(st.excess_ms, 0.0);
    if (st.read_ms > 0) return st.read_ms;
    return null;
}

/// The boundary in floor units: ms through the table's own scale, else through this request's round (an m_lo round
/// costs T(m_lo)); the cost profile's `sync` until a considered round has measured one.
pub fn boundaryCost(costs: MtpEvCosts, src: MtpCostSource, boundary_ms: ?f32, round_ms: f32, m_lo: u32) f32 {
    const ms = boundary_ms orelse return costs.sync;
    if (src.fromTable()) return ms * src.scale;
    if (round_ms > 0) return ms / round_ms * src.roundCost(m_lo, false);
    return costs.sync;
}

/// Fold one steady two-chunk round's excess: a running mean over the first rounds (one noisy round must not seed an
/// EMA), MTP_EV_COST_BETA after.
pub fn excessFold(st: *Model, excess_ms: f32) void {
    const n: f32 = @floatFromInt(st.excess_n);
    const beta = @max(1.0 / (n + 1.0), MTP_EV_COST_BETA);
    st.excess_ms += beta * (excess_ms - st.excess_ms);
    st.excess_n +|= 1;
}

/// The gate's realized yield against the ideal gate's promise. The yield is capped at 1; the firing rate is not (a
/// gate that fires on failed chunks pays the extension more often).
pub fn ratios(st: Model) Ratios {
    const k = PRIOR;
    return .{
        .gain = std.math.clamp((st.gain_real + k) / (st.gain_ideal + k), 0.0, 1.0),
        .fire = @max((st.fire_real + k) / (st.fire_ideal + k), 0.0),
    };
}

/// Fold one round of a kept two-chunk plan (decay MTP_EV_COST_BETA). `drafted > m_lo` means the gate fired; the
/// extension's tokens are the accepted drafts past m_lo.
pub fn fold(st: *Model, plan: MtpRoundPlan, drafted: u32, accepted: u32) void {
    // A plan narrowed to one chunk after it was kept (a group cap, a lookup chain) never reached the boundary.
    if (!(plan.fire_ideal > 0) or plan.m_hi <= plan.m_lo) return;
    const keep = 1.0 - MTP_EV_COST_BETA;
    const fired = drafted > plan.m_lo;
    const gain: f32 = if (fired and accepted > plan.m_lo) @floatFromInt(accepted - plan.m_lo) else 0.0;
    st.gain_real = st.gain_real * keep + gain;
    st.gain_ideal = st.gain_ideal * keep + plan.gain_ideal;
    st.fire_real = st.fire_real * keep + @as(f32, if (fired) 1.0 else 0.0);
    st.fire_ideal = st.fire_ideal * keep + plan.fire_ideal;
}

/// Keep `plan`'s two chunks only when they beat every single chunk from m_lo to min(m_hi, m_lo_max), valued at the
/// ideal gate scaled by the realized ratios; else the best of those single chunks, the standing base kept unless a
/// challenger beats it by the switch margin. A kept plan carries the ideal gate's promise to the round end.
pub fn gate(plan: MtpRoundPlan, a: []const f32, src: MtpCostSource, m_lo_max: u32, in: Inputs) MtpRoundPlan {
    const n: u32 = @intCast(a.len);
    const m_lo = plan.m_lo;
    const m_hi = @min(plan.m_hi, n);
    if (plan.width_trial or m_lo == 0 or m_hi <= m_lo) return plan;
    // P_A: chunk A lands whole, the only rounds the ideal gate extends.
    var p_a: f32 = 1.0;
    for (a[0..m_lo]) |x| p_a *= x;
    const e_lo = mtpEvExpectedTokens(a, m_lo);
    const gain_ideal = mtpEvExpectedTokens(a, m_hi) - e_lo;
    var ext_cost: f32 = 0.0;
    var k: u32 = m_lo + 1;
    while (k <= m_hi) : (k += 1) ext_cost += src.marginal(k);
    const t_lo = src.roundCost(m_lo, false);
    const fire = std.math.clamp(p_a * in.fire_ratio, 0.0, 1.0);
    const gain = gain_ideal * std.math.clamp(in.gain_ratio, 0.0, 1.0);
    const two_r = (e_lo + gain) / (t_lo + @max(in.boundary, 0.0) + fire * ext_cost);
    const top = @min(m_hi, @max(m_lo, m_lo_max));
    var best_w: u32 = m_lo;
    var best_r: f32 = e_lo / t_lo;
    var w: u32 = m_lo + 1;
    while (w <= top) : (w += 1) {
        const r = mtpEvExpectedTokens(a, w) / src.roundCost(w, false);
        if (r > best_r) {
            best_r = r;
            best_w = w;
        }
    }
    if (two_r > best_r) {
        var kept = plan;
        kept.m_hi = m_hi;
        kept.fire_ideal = p_a;
        kept.gain_ideal = gain_ideal;
        return kept;
    }
    const standing = m_lo_max -| 1;
    if (standing != best_w and standing >= m_lo and standing <= top) {
        const standing_r = mtpEvExpectedTokens(a, standing) / src.roundCost(standing, false);
        if (best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) best_w = standing;
    }
    return .{ .m_lo = best_w, .m_hi = best_w, .tau_ln = 0.0 };
}

/// This request's head prices its two-chunk plans (`MtpHeadRef.exl3Profile`).
pub fn mtpEvFixOn(self: *const Generator) bool {
    return if (self.mtp) |h| h.exl3Profile() else false;
}

/// `gate` on this round's base plan with the model's measurements: the boundary in floor units through the plan's own
/// cost source, and the gate's realized yield.
pub fn mtpEvFixPlan(self: *Generator, plan: MtpRoundPlan, cap: u32, src: MtpCostSource) MtpRoundPlan {
    const st = &self.xfm.mtp_ev_fix;
    const r = ratios(st.*);
    const ms = boundaryMs(st.*);
    const gated = gate(plan, self.mtp_ev_accept[0..cap], src, self.mtp_ev_m_lo_prev + 1, .{
        .boundary = boundaryCost(self.mtp_ev_costs, src, ms, self.mtp_ev_round_ms, plan.m_lo),
        .gain_ratio = r.gain,
        .fire_ratio = r.fire,
    });
    self.mtp_ev_fix.priced +|= 1;
    if (!ev_fix_said) {
        ev_fix_said = true;
        log.info("[mtp] ev-fix engaged: two-chunk plans priced at this model's chunk-A boundary (first {d}->{d}, {d:.2} ms measured)\n", .{ plan.m_lo, plan.m_hi, ms orelse 0 });
    }
    if (gated.m_hi == gated.m_lo) {
        self.mtp_ev_fix.collapsed +|= 1;
        if (!st.logged) {
            st.logged = true;
            log.info("[mtp] ev-fix: two-chunk {d}->{d} does not repay its {d:.2} ms chunk-A boundary -> single chunk {d}\n", .{ plan.m_lo, plan.m_hi, ms orelse 0, gated.m_lo });
        }
    }
    return gated;
}

var ev_fix_said = false;

/// Round end of a kept two-chunk plan, solo rounds only. Runs after `mtpRoundEndObserve`: two-chunk rounds never feed
/// the table, so its single chunk at the drafted width is this round's counterfactual.
pub fn mtpEvFixObserve(self: *Generator, plan: MtpRoundPlan, drafted: u32, accepted: u32) void {
    if (!(plan.fire_ideal > 0) or plan.m_hi <= plan.m_lo or !self.spec_cost_solo) return;
    const st = &self.xfm.mtp_ev_fix;
    fold(st, plan, drafted, accepted);
    if (self.mtp_ev_fix.wall_ms > 0) {
        const t = &self.xfm.round_cost;
        if (t.measuredMs(drafted, t.bucketOf(self.mtpKvLen()))) |single_ms| excessFold(st, self.mtp_ev_fix.wall_ms - single_ms);
    }
}

// ── tests ──

const testing = std.testing;
const depth_bounds = @import("../mtp_depth_bounds.zig");
const ev_plan = @import("ev_plan.zig");
const Transformer = @import("../../transformer.zig").Transformer;
const Qwen4Trunk = @import("../../../../plugins/qwen4_exp/trunk.zig").Trunk;

/// Per-index conditional acceptance at cap 3.
const fix_accept = [_]f32{ 0.88, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85 };
