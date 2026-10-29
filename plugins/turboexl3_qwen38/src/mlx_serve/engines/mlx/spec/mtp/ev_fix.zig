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

/// A planning state after the warm-up at cap 3: single-chunk cells at w2 and w3 (two-chunk rounds never feed the
/// table), w2 the cheaper per token.
fn measuredTable(t: *round_cost.Table, kv: u32) void {
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(2, kv, 26.9, 2.40, true, false);
        _ = t.observe(3, kv, 30.4, 2.55, true, false);
    }
}

/// Per-index conditional acceptance at cap 3.
const fix_accept = [_]f32{ 0.88, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85 };

test "mtp ev fix: a boundary no gate repays plans the single chunk the extension reaches" {
    const kv: u32 = 18000;
    var t = round_cost.Table{};
    measuredTable(&t, kv);
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const src = Generator.MtpCostSource.init(costs, kv, &t);
    try testing.expect(src.fromTable());
    const a = fix_accept[0..3];
    // The legacy base reads w2's token cell over w3's, drops to m_lo 2 and opens the horizon to 3.
    const base = Generator.mtpBasePlan(.legacy, a, 3, src, 4);
    try testing.expectEqual(@as(u32, 2), base.m_lo);
    try testing.expectEqual(@as(u32, 3), base.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -1.0014), base.tau_ln, 1e-3);
    // A 2.9 ms boundary through the table's own scale.
    const boundary = boundaryCost(costs, src, 2.9, 0, base.m_lo);
    try testing.expectApproxEqAbs(2.9 * src.scale, boundary, 1e-6);
    // Not even an ideal gate repays it: the round is the single chunk at 3.
    const fixed = gate(base, a, src, 4, .{ .boundary = boundary });
    try testing.expectEqual(@as(u32, 3), fixed.m_lo);
    try testing.expectEqual(@as(u32, 3), fixed.m_hi);
    try testing.expectEqual(@as(f32, 0.0), fixed.tau_ln);
    try testing.expectEqual(@as(f32, 0.0), fixed.fire_ideal);
    // Break-even against single 3: (1 - P_A) * (T(3) - T(2)).
    const break_even_ms = (1.0 - a[0] * a[1]) * (src.roundCost(3, false) - src.roundCost(2, false)) / src.scale;
    try testing.expectApproxEqAbs(@as(f32, 0.882), break_even_ms, 2e-3);
    const over = gate(base, a, src, 4, .{ .boundary = (break_even_ms + 0.05) * src.scale });
    try testing.expectEqual(@as(u32, 3), over.m_lo);
    try testing.expectEqual(@as(u32, 3), over.m_hi);
    const under = gate(base, a, src, 4, .{ .boundary = (break_even_ms - 0.05) * src.scale });
    try testing.expectEqual(@as(u32, 2), under.m_lo);
    try testing.expectEqual(@as(u32, 3), under.m_hi);
}

test "mtp ev fix: a cheap boundary keeps a two-chunk plan that pays, and a confident chunk A still extends" {
    const kv: u32 = 18000;
    var t = round_cost.Table{};
    measuredTable(&t, kv);
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const src = Generator.MtpCostSource.init(costs, kv, &t);
    const a = fix_accept[0..3];
    const base = Generator.mtpBasePlan(.legacy, a, 3, src, 4);
    // 0.5 ms, under the break-even, with no realized history yet (the ideal gate): kept as planned.
    const cheap = gate(base, a, src, 4, .{ .boundary = boundaryCost(costs, src, 0.5, 0, base.m_lo) });
    try testing.expectEqual(@as(u32, 2), cheap.m_lo);
    try testing.expectEqual(@as(u32, 3), cheap.m_hi);
    try testing.expectEqual(base.tau_ln, cheap.tau_ln);
    try testing.expectApproxEqAbs(a[0] * a[1], cheap.fire_ideal, 1e-6);
    try testing.expectApproxEqAbs(Generator.mtpEvExpectedTokens(a, 3) - Generator.mtpEvExpectedTokens(a, 2), cheap.gain_ideal, 1e-6);
    // The boundary's own decision is untouched: a confident chunk A clears tau, a doubtful one does not.
    const confident = [_]f32{ @log(@as(f32, 0.99)), @log(@as(f32, 0.98)) };
    const doubtful = [_]f32{ @log(@as(f32, 0.5)), @log(@as(f32, 0.6)) };
    try testing.expect(Generator.mtpChainLogConf(&confident) >= cheap.tau_ln);
    try testing.expect(Generator.mtpChainLogConf(&doubtful) < cheap.tau_ln);

    // A surface whose third verify row is a cliff (prior only, no table): the extension pays at the profile's own
    // sync, and a dear boundary leaves the single chunk at m_lo, the cliff making 3 the worst single chunk.
    const cliff = Generator.MtpEvCosts{ .draft = 0.02, .per_pos_lo = 0.08, .per_pos_hi = 0.38, .flat_max = 2, .sync = 0.01 };
    const easy = [_]f32{ 0.85, 0.82, 0.97 };
    const prior = Generator.MtpCostSource.init(cliff, 0, null);
    const plan = Generator.mtpEvPlanSrc(&easy, 3, prior, 3);
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    const kept = gate(plan, &easy, prior, 3, .{ .boundary = boundaryCost(cliff, prior, null, 0, plan.m_lo) });
    try testing.expectEqual(@as(u32, 2), kept.m_lo);
    try testing.expectEqual(@as(u32, 3), kept.m_hi);
    try testing.expect(kept.fire_ideal > 0);
    const confident_easy = [_]f32{ @log(@as(f32, 0.99)), @log(@as(f32, 0.97)) };
    try testing.expect(Generator.mtpChainLogConf(&confident_easy) >= kept.tau_ln);
    const dear = gate(plan, &easy, prior, 3, .{ .boundary = 0.15 });
    try testing.expectEqual(@as(u32, 2), dear.m_lo);
    try testing.expectEqual(@as(u32, 2), dear.m_hi);
}

test "mtp ev fix: the gate's realized yield prices a gate that extends on the wrong rounds" {
    var st: Model = .{};
    const ideal = ratios(st);
    try testing.expectEqual(@as(f32, 1.0), ideal.gain);
    try testing.expectEqual(@as(f32, 1.0), ideal.fire);
    // A plan the fix did not keep folds nothing.
    fold(&st, .{ .m_lo = 2, .m_hi = 3, .tau_ln = -1.0 }, 3, 3);
    try testing.expectEqual(Model{}, st);
    // One fold, exactly: a fired round whose extension draft landed.
    const plan = MtpRoundPlan{ .m_lo = 2, .m_hi = 3, .tau_ln = -1.0, .fire_ideal = 0.748, .gain_ideal = 0.6358 };
    fold(&st, plan, 3, 3);
    try testing.expectEqual(@as(f32, 1.0), st.gain_real);
    try testing.expectEqual(@as(f32, 1.0), st.fire_real);
    try testing.expectApproxEqAbs(@as(f32, 0.6358), st.gain_ideal, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.748), st.fire_ideal, 1e-6);
    // A gate that fires on 4 rounds in 6 (the ideal gate: 3 in 4) and lands the extension draft on 3 of those 4.
    // (drafted, accepted) per round, m_lo 2.
    st = .{};
    const rounds = [_][2]u32{ .{ 3, 3 }, .{ 3, 3 }, .{ 2, 2 }, .{ 3, 2 }, .{ 3, 3 }, .{ 2, 1 } };
    for (0..8) |_| {
        for (rounds) |r| fold(&st, plan, r[0], r[1]);
    }
    const real = ratios(st);
    try testing.expect(real.gain > 0.70 and real.gain < 0.85);
    try testing.expect(real.fire > 0.80 and real.fire < 0.95);
    // That gate cannot buy back even the 0.5 ms boundary the ideal gate keeps.
    const kv: u32 = 18000;
    var t = round_cost.Table{};
    measuredTable(&t, kv);
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const src = Generator.MtpCostSource.init(costs, kv, &t);
    const a = fix_accept[0..3];
    const base = Generator.mtpBasePlan(.legacy, a, 3, src, 4);
    const boundary = boundaryCost(costs, src, 0.5, 0, base.m_lo);
    try testing.expectEqual(@as(u32, 2), gate(base, a, src, 4, .{ .boundary = boundary }).m_lo);
    const priced = gate(base, a, src, 4, .{ .boundary = boundary, .gain_ratio = real.gain, .fire_ratio = real.fire });
    try testing.expectEqual(@as(u32, 3), priced.m_lo);
    try testing.expectEqual(@as(u32, 3), priced.m_hi);
}

test "mtp ev fix: the boundary is priced from the first considered round on" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const prior = Generator.MtpCostSource.init(costs, 0, null);
    // Never measured: the cost profile's `sync`, so the first considered round still runs and measures it.
    try testing.expect(boundaryMs(.{}) == null);
    try testing.expectEqual(costs.sync, boundaryCost(costs, prior, null, 0, 2));
    // Measured before any table: through this request's round, an m_lo round being T(m_lo) floor units.
    try testing.expectApproxEqAbs(2.9 / 29.0 * Generator.mtpEvRoundCost(costs, 2, false), boundaryCost(costs, prior, 2.9, 29.0, 2), 1e-6);
    // With the table: its own ms -> floor-unit scale, whatever the round EMA says.
    var t = round_cost.Table{};
    measuredTable(&t, 18000);
    const src = Generator.MtpCostSource.init(costs, 18000, &t);
    try testing.expectApproxEqAbs(2.9 * src.scale, boundaryCost(costs, src, 2.9, 29.0, 2), 1e-6);
    // The read EMA seeds on its first sample, so the warm-up's first considered round prices the next request.
    try testing.expectApproxEqAbs(@as(f32, 2.9), Generator.mtpEmaMs(0, 2_900_000), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.9 + 0.1 * (3.4 - 2.9)), Generator.mtpEmaMs(2.9, 3_400_000), 1e-5);
    try testing.expectEqual(@as(?f32, 2.9), boundaryMs(.{ .read_ms = 2.9 }));
}

test "mtp ev fix: steady two-chunk rounds replace the read with what the shape really adds" {
    // A read + chunk B of 3.25 ms a round, while steady two-chunk rounds ran under 2 ms over the single chunk.
    var st: Model = .{ .read_ms = 3.25 };
    const samples = [_]f32{ 2.9, 0.4, 1.8, 1.2, 2.3, 0.9, 1.6, 1.7 };
    for (samples[0 .. samples.len - 1]) |x| {
        excessFold(&st, x);
        // Fewer than EXCESS_MIN rounds: the read still prices the boundary.
        try testing.expectEqual(@as(?f32, 3.25), boundaryMs(st));
    }
    excessFold(&st, samples[samples.len - 1]);
    try testing.expectEqual(EXCESS_MIN, st.excess_n);
    // A running mean, not an EMA seeded by the first (noisy) round.
    var sum: f32 = 0;
    for (samples) |x| sum += x;
    try testing.expectApproxEqAbs(sum / @as(f32, @floatFromInt(samples.len)), boundaryMs(st).?, 1e-5);
    // Past the running-mean window it is the cost EMA.
    var ema: Model = .{ .excess_ms = 1.6, .excess_n = 20 };
    excessFold(&ema, 2.6);
    try testing.expectApproxEqAbs(@as(f32, 1.6 + MTP_EV_COST_BETA * 1.0), ema.excess_ms, 1e-5);
    // A shape that saves time (negative excess) prices a free boundary, never a negative one.
    const cheaper: Model = .{ .read_ms = 3.0, .excess_ms = -0.4, .excess_n = 9 };
    try testing.expectEqual(@as(?f32, 0.0), boundaryMs(cheaper));
}

test "mtp ev fix: a seeded request on the warm-up's table and boundary plans the single chunk, on an EXL3 pack only" {
    const before = depth_bounds.active;
    const policy_before = ev_plan.mtp_depth_policy_override;
    defer {
        depth_bounds.active = before;
        ev_plan.mtp_depth_policy_override = policy_before;
    }
    depth_bounds.active = .{};
    // The fixture's 1000-token prompt plans under `.legacy` only when forced, as a prompt past MTP_ACCEPT_MAX_KV does.
    ev_plan.mtp_depth_policy_override = .legacy;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    var exl3: Qwen4Trunk = undefined;
    exl3.formats = .{ .profile = true };
    var plain: Qwen4Trunk = undefined;
    plain.formats = .{};
    // m_lo_prev 3: seeded from a single-chunk warm-up; 2: from a request that ended in two-chunk mode.
    for ([_]u32{ 3, 2 }) |seeded_m_lo| {
        for ([_]bool{ false, true }) |on| {
            var gen = ev_plan.mtpBoundsPlanFixture(&xfm, &prompt, 3);
            gen.mtp = .{ .qwen4 = if (on) &exl3 else &plain };
            xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
            measuredTable(&xfm.round_cost, 1000);
            // What the warm-up left on the model: one measured boundary, no realized gate history.
            xfm.mtp_ev_fix = .{ .read_ms = 2.9 };
            gen.mtp_ev_accept = fix_accept;
            gen.mtp_ev_m_lo_prev = seeded_m_lo;
            const plan = gen.mtpRoundPlan();
            if (on) {
                try testing.expectEqual(@as(u32, 3), plan.m_lo);
                try testing.expectEqual(@as(u32, 3), plan.m_hi);
                try testing.expectEqual(@as(u32, 1), gen.mtp_ev_fix.priced);
                try testing.expectEqual(@as(u32, 1), gen.mtp_ev_fix.collapsed);
            } else {
                // Off: neither the dry-spell gate (fresh streak) nor the regime gate (two-chunk unmeasured in this
                // request) stops the flip.
                try testing.expectEqual(@as(u32, 2), plan.m_lo);
                try testing.expectEqual(@as(u32, 3), plan.m_hi);
                try testing.expectEqual(@as(u32, 0), gen.mtp_ev_fix.priced);
            }
            // The base the next round, and the next request's EV seed, start from is the one that runs.
            try testing.expectEqual(plan.m_lo, gen.mtp_ev_m_lo_prev);
        }
    }
    // A boundary under the break-even keeps the plan two-chunk, carrying the ideal gate's promise.
    var cheap_gen = ev_plan.mtpBoundsPlanFixture(&xfm, &prompt, 3);
    cheap_gen.mtp = .{ .qwen4 = &exl3 };
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    measuredTable(&xfm.round_cost, 1000);
    xfm.mtp_ev_fix = .{ .read_ms = 0.5 };
    cheap_gen.mtp_ev_accept = fix_accept;
    cheap_gen.mtp_ev_m_lo_prev = 3;
    const cheap = cheap_gen.mtpRoundPlan();
    try testing.expectEqual(@as(u32, 2), cheap.m_lo);
    try testing.expectEqual(@as(u32, 3), cheap.m_hi);
    try testing.expect(cheap.fire_ideal > 0);
    try testing.expectEqual(@as(u32, 0), cheap_gen.mtp_ev_fix.collapsed);
}

test "mtp ev fix: a kept plan's round folds its outcome and a clean wall's excess over the single chunk" {
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    var gen = ev_plan.mtpBoundsPlanFixture(&xfm, &prompt, 3);
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    measuredTable(&xfm.round_cost, 1000);
    xfm.mtp_ev_fix = .{};
    gen.spec_cost_solo = true;
    const plan = MtpRoundPlan{ .m_lo = 2, .m_hi = 3, .tau_ln = -1.0, .fire_ideal = 0.748, .gain_ideal = 0.6358 };
    // A fired round with a clean wall: the extension landed, and the wall runs over the table's single chunk at 3.
    gen.mtp_ev_fix.wall_ms = 32.0;
    gen.mtpEvFixObserve(plan, 3, 3);
    const single = xfm.round_cost.measuredMs(3, xfm.round_cost.bucketOf(gen.mtpKvLen())).?;
    try testing.expectEqual(@as(u32, 1), xfm.mtp_ev_fix.excess_n);
    try testing.expectApproxEqAbs(32.0 - single, xfm.mtp_ev_fix.excess_ms, 1e-4);
    try testing.expectEqual(@as(f32, 1.0), xfm.mtp_ev_fix.gain_real);
    // No clean wall: the outcome folds, the excess does not.
    gen.mtp_ev_fix.wall_ms = 0;
    gen.mtpEvFixObserve(plan, 2, 2);
    try testing.expectEqual(@as(u32, 1), xfm.mtp_ev_fix.excess_n);
    // A grouped round (not solo) and an unpriced plan fold nothing.
    const after = xfm.mtp_ev_fix;
    gen.spec_cost_solo = false;
    gen.mtpEvFixObserve(plan, 3, 3);
    gen.spec_cost_solo = true;
    gen.mtpEvFixObserve(.{ .m_lo = 2, .m_hi = 3, .tau_ln = -1.0 }, 3, 3);
    try testing.expectEqual(after, xfm.mtp_ev_fix);
}
