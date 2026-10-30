//! copied_from: src/engines/mlx/spec/mtp/planner.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MTP round planner entry points, width trials and the EV round update.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const round_cost = @import("../round_cost.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_DEPTH_WINDOW = @import("depth.zig").MTP_DEPTH_WINDOW;
const MTP_DISABLE_BELOW = @import("depth.zig").MTP_DISABLE_BELOW;
const MTP_EV_EMA_BETA = @import("depth.zig").MTP_EV_EMA_BETA;
const MTP_EV_WARMUP_ROUNDS = @import("depth.zig").MTP_EV_WARMUP_ROUNDS;
const MtpCostSource = @import("ev_plan.zig").MtpCostSource;
const MtpRoundPlan = @import("ev_plan.zig").MtpRoundPlan;
const mtpBasePlan = @import("ev_plan.zig").mtpBasePlan;
const mtpDepthPolicy = @import("ev_plan.zig").mtpDepthPolicy;
const mtpDepthPolicyFor = @import("ev_plan.zig").mtpDepthPolicyFor;
const mtpEvExpectedTokens = @import("ev_plan.zig").mtpEvExpectedTokens;
const mtpEvObserve = @import("ev_plan.zig").mtpEvObserve;
const mtpExtDryAllows = @import("regime.zig").mtpExtDryAllows;
const mtpExtDryThresholdFor = @import("regime.zig").mtpExtDryThresholdFor;
const mtpPinnedDepth = @import("regime.zig").mtpPinnedDepth;
const mtpPlanInBounds = @import("regime.zig").mtpPlanInBounds;
const mtpRegimeExplorePeriod = @import("regime.zig").mtpRegimeExplorePeriod;
const mtpRegimeForce = @import("regime.zig").mtpRegimeForce;
pub fn mtpRoundPlanInner(self: *Generator) MtpRoundPlan {
    if (mtpPinnedDepth()) |pinned| {
        const d = @min(pinned, self.mtp_depth);
        self.mtp_ev_m_lo_prev = d;
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    if (self.mtp_serial_accept) {
        const d = depth_bounds.clampWidth(self.mtpExactDepth(@min(@max(@as(u32, 1), self.mtp_depth), 2)), depth_bounds.floorFor(depth_bounds.active, self.mtp_depth), self.mtp_depth);
        self.mtp_ev_m_lo_prev = d;
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
    const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), if (self.mtp) |h| h.maxDepth() else mtp_mod.MAX_DEPTH);
    var cap: u32 = cap_row;
    const kv_len = self.mtpKvLen();
    if (self.mtp_ev_rounds < MTP_EV_WARMUP_ROUNDS) {
        const d = depth_bounds.clampWidth(self.mtp_depth_current, depth_bounds.floorFor(depth_bounds.active, cap), cap);
        self.mtp_ev_m_lo_prev = d;
        // Independent of the depth controller's kill switch and warmup.
        _ = self.mtpAdaptiveSerialStep(d, kv_len);
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, &self.xfm.round_cost);
    if (src.fromTable() and !self.xfm.round_cost.first_use_logged) {
        self.xfm.round_cost.first_use_logged = true;
        var buf: [256]u8 = undefined;
        log.info("[mtp] cost table: bucket {s} measured {s} (ms/tok) replaces the fitted surface (scale {d:.4} at w{d})\n", .{
            round_cost.bucketName(self.xfm.round_cost.layout, src.bucket),
            self.xfm.round_cost.formatBucket(src.bucket, &buf),
            src.scale,
            self.xfm.round_cost.narrowestMeasured(src.bucket) orelse 0,
        });
    }
    // The per-silicon row is the COLD-START cap: the plan may exceed it
    // only up to the widest TRUSTED width (never onto the prior's guess
    // above the row — that reopens the regime gate on ties the row had
    // closed), and the width trial may reach one past that to measure.
    if (src.fromTable()) cap = @min(cap_free, @max(cap_row, self.xfm.round_cost.widestMeasured(src.bucket) orelse cap_row));
    const policy = mtpDepthPolicyFor(mtpDepthPolicy(), kv_len);
    var plan = mtpPlanInBounds(mtpBasePlan(policy, self.mtp_ev_accept[0..cap], cap, src, self.mtp_ev_m_lo_prev + 1), depth_bounds.floorFor(depth_bounds.active, cap), cap);
    // A two-chunk plan pays its chunk-A boundary or becomes the best single chunk (`ev_fix.zig`), before the base-depth
    // bookkeeping: the streak, the climb damping and the next request's EV seed read the plan that runs.
    if (plan.m_hi > plan.m_lo and self.mtpEvFixOn()) plan = mtpPlanInBounds(self.mtpEvFixPlan(plan, cap, src), depth_bounds.floorFor(depth_bounds.active, cap), cap);
    if (plan.m_lo == self.mtp_ev_m_lo_prev) self.mtp_m_lo_streak +|= 1 else self.mtp_m_lo_streak = 0;
    self.mtp_ev_m_lo_prev = plan.m_lo;
    // Live-cost lever: shorten dry exploration bursts when the MEASURED
    // chunk-A sync is an expensive fraction of the round (mtpExtDryThresholdFor).
    const dry_threshold = mtpExtDryThresholdFor(self.mtp_ev_sync_ms, self.mtp_ev_round_ms);
    if (plan.m_hi > plan.m_lo and
        !mtpExtDryAllows(&self.mtp_ext_dry_streak, &self.mtp_ext_cooldown, dry_threshold))
    {
        plan.m_hi = plan.m_lo;
        plan.tau_ln = 0.0;
    }
    if (plan.m_hi > plan.m_lo) {
        const force = mtpRegimeForce(&self.mtp_regime, self.mtp_ev_rounds);
        const worse = self.mtp_regime.sched_verdict;
        if (worse != null and worse != self.mtp_regime_verdict) {
            self.mtp_regime_verdict = worse;
            const r = self.mtp_regime;
            log.info("[mtp] regime gate: two-chunk {d:.2} ms/tok vs single {d:.2} ms/tok -> two-chunk {s} (period {d})\n", .{ r.two_ms / r.two_tok, r.one_ms / r.one_tok, if (worse.?) "throttled" else "every round", mtpRegimeExplorePeriod(r) });
        }
        if (force == false) {
            plan.m_hi = plan.m_lo;
            plan.tau_ln = 0.0;
        }
    }
    if (self.mtpAdaptiveSerialStep(plan.m_lo, kv_len)) return plan;
    // Width trial: a single-chunk round at the width the table needs
    // next (`mtpWidthTrialTarget`), one 2-round block per period. Never
    // inside a regime trial block (that block is the regime's own
    // measurement), and only while solo.
    if (self.spec_cost_solo and self.mtp_ev_rounds >= self.mtp_regime.trial_end) {
        const base_settled = self.mtp_m_lo_streak >= 2;
        if (mtpWidthTrialTarget(&self.xfm.round_cost, kv_len, plan, cap_free, base_settled)) |target| {
            // A next width nobody has timed is priced from the prior: learn it now, not a period from now.
            if (policy == .accept and self.xfm.round_cost.rawMs(target, src.bucket) == null) self.mtp_width_trial.startAt(self.mtp_ev_rounds);
            const period = if (policy == .accept)
                mtpProbePeriod(&self.mtp_ev_accept, src, plan.m_lo)
            else
                mtpWidthTrialPeriod(&self.xfm.round_cost, kv_len, plan.m_lo);
            const reread = round_cost.schedulePeriodReread(self.xfm.round_cost.layout);
            if (mtpWidthTrialForce(&self.mtp_width_trial, self.mtp_ev_rounds, period, reread)) {
                plan = mtpPlanInBounds(mtpWidthTrialPlan(target), depth_bounds.floorFor(depth_bounds.active, cap_free), cap_free);
            }
        }
    }
    return plan;
}
/// The single-chunk plan a width trial runs. Built from a scalar on
/// purpose: `plan = .{ .m_lo = plan.m_lo + 1, .m_hi = plan.m_lo + 1 }`
/// reads the already-written m_lo for m_hi (result-location aliasing)
/// and planned a two-chunk round — the simulated loop caught it.
pub fn mtpWidthTrialPlan(width: u32) MtpRoundPlan {
    return .{ .m_lo = width, .m_hi = width, .tau_ln = 0.0, .width_trial = true };
}
/// Which width a trial measures: the plan's own base when the bucket
/// the plan reads has not measured it, else an unmeasured m_lo+1 under
/// any shape, else a periodic m_lo+1 on single-chunk plans. Null =
/// nothing to try.
pub fn mtpWidthTrialTarget(t: *const round_cost.Table, kv_len: u32, plan: MtpRoundPlan, cap: u32, base_settled: bool) ?u32 {
    const b = t.bucketToRead(kv_len) orelse t.bucketOf(kv_len);
    // A single-chunk plan feeds its own base every round; only a
    // two-chunk plan (extensions and syncs never feed) owes a trial of
    // it — and only once the base has stopped climbing, or the first EV
    // rounds trial widths the plan is merely passing through (M1 Pro
    // 27B warm boot: one block at w3 in rep 1).
    if (t.measuredMs(plan.m_lo, b) == null) return if (plan.m_hi > plan.m_lo and base_settled) plan.m_lo else null;
    // m_lo-1 is deliberately NOT trialled: every trial block is a 3-4%
    // hit on the request that carries it (peer cells), the
    // prior below the anchor already prices narrower widths, and the
    // one cliff this was for (M1 Pro 27B, 4 -> 5) is found from 4 by
    // the m_lo+1 trial below.
    if (plan.m_lo >= cap) return null;
    // An unmeasured m_lo+1 is measured under ANY shape: a two-chunk
    // plan extends there every round on echo, blind to its cost (the
    // M1 Pro 27B's 4 -> 5 at +150 ms); the regime gate would catch it,
    // the table should know it — unless its first sample already
    // settled it as clearly worse. A settled one is re-tried only where
    // a single-chunk plan cannot otherwise reach it, at the long period.
    const up = plan.m_lo + 1;
    if (t.measuredMs(up, b) == null and !t.clearlyWorse(up, plan.m_lo, b)) return up;
    if (plan.m_hi == plan.m_lo) return up;
    return null;
}
/// Width-trial schedule: same shape as the regime's (explicit
/// trial_end/next_trial, a 2-round block — transition then measurement),
/// idempotent per round because `mtpRoundPlan` has two call sites.
pub const MtpWidthTrial = round_cost.TrialSchedule;
/// `reread` is the layout's answer (`round_cost.schedulePeriodReread`): every layout re-reads
/// the period so a cold bucket cannot inherit a neighbour's date.
pub fn mtpWidthTrialForce(t: *MtpWidthTrial, round_idx: u32, period: u32, reread: bool) bool {
    return t.force(round_idx, period, reread);
}
/// Rounds between width trials: the regime's drag rule on the measured
/// ms/tok gap between m_lo+1 and m_lo (a width G worse, tried once in
/// G/DRAG rounds, costs ~DRAG of throughput); the default period while
/// either is unmeasured, so an unknown width is learned soon.
pub fn mtpWidthTrialPeriod(t: *const round_cost.Table, kv_len: u32, m_lo: u32) u32 {
    const b = t.bucketToRead(kv_len) orelse return round_cost.EXPLORE_PERIOD_COLD;
    // An untrusted target keeps the cold period until it is either
    // trusted or settled as clearly worse (then its raw sample sizes
    // the gap): a raw 10%-better w5 read as a 60-round period and no
    // 22-round request ever trialled it again (M4 Max 27B, w5 stuck
    // at one sample).
    if (t.msPerTok(m_lo + 1, b) == null and !t.clearlyWorse(m_lo + 1, m_lo, b)) return round_cost.EXPLORE_PERIOD_COLD;
    return round_cost.trialPeriod(t.msPerTok(m_lo, b), t.rawMsPerTok(m_lo + 1, b));
}
/// Probe period when the plan's tokens come from the acceptance EMAs: the same drag rule,
/// over the EV rates of m_lo and m_lo+1. The probe is what refreshes `a[m_lo]`, so a stale
/// estimate shortens its own period as soon as the shallower indices move.
pub fn mtpProbePeriod(a: []const f32, src: MtpCostSource, m_lo: u32) u32 {
    const t = src.table orelse return round_cost.EXPLORE_PERIOD_COLD;
    if (!src.fromTable()) return round_cost.EXPLORE_PERIOD_COLD;
    if (t.measuredMs(m_lo + 1, src.bucket) == null) {
        // One sample at a verify-width cliff is enough to stop asking at the cold period.
        if (!t.clearlyWorse(m_lo + 1, m_lo, src.bucket)) return round_cost.EXPLORE_PERIOD_COLD;
        return round_cost.trialPeriod(t.msPerTok(m_lo, src.bucket), t.rawMsPerTok(m_lo + 1, src.bucket));
    }
    const here = src.roundCost(m_lo, false) / mtpEvExpectedTokens(a, m_lo);
    const up = src.roundCost(m_lo + 1, false) / mtpEvExpectedTokens(a, m_lo + 1);
    return round_cost.trialPeriod(here, up);
}
/// Track the only evidence that can justify sticky-disable: whether the
/// first draft landed while the EV base depth was exactly one. Wider base
/// rounds reset the probation window, and later extension misses do not
/// count against the depth-one floor. Returns a rate only after a full
/// fresh window has been observed.
pub fn mtpFloorDisableObserve(
    drafted_window: *[MTP_DEPTH_WINDOW]u8,
    accepted_window: *[MTP_DEPTH_WINDOW]u8,
    window_idx: *u32,
    m_lo: u32,
    drafted: u32,
    accepted: u32,
) ?f32 {
    std.debug.assert(drafted >= 1);
    if (m_lo != 1) {
        window_idx.* = 0;
        return null;
    }
    const idx = window_idx.* % MTP_DEPTH_WINDOW;
    drafted_window[idx] = 1;
    accepted_window[idx] = @intFromBool(accepted > 0);
    window_idx.* += 1;
    const n = @min(window_idx.*, MTP_DEPTH_WINDOW);
    if (n < MTP_DEPTH_WINDOW) return null;
    var accepted_sum: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) accepted_sum += accepted_window[i];
    return @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(n));
}
/// EV-mode per-round update: EMAs always; during warmup the legacy
/// windowed controller keeps running (today's behavior while EMAs fill);
/// post-warmup only the sticky disable floor is checked — EV owns depth.
pub fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
    mtpEvObserve(&self.mtp_ev_accept, drafted, accepted, MTP_EV_EMA_BETA);
    self.mtp_ev_rounds += 1;
    if (mtpPinnedDepth() != null) return;
    if (self.mtp_ev_rounds <= MTP_EV_WARMUP_ROUNDS) {
        self.updateMtpDepth(drafted, accepted);
        // Warmup may evaluate several depths. None of that mixed evidence
        // belongs in the post-warmup depth-one sticky-disable window.
        if (self.mtp_ev_rounds == MTP_EV_WARMUP_ROUNDS) self.mtp_window_idx = 0;
        return;
    }
    // EV owns promotion/demotion. Sticky-disable is judged only from a
    // full, homogeneous window of first-draft outcomes at base depth one.
    const rate = mtpFloorDisableObserve(
        &self.mtp_window_drafted,
        &self.mtp_window_accepted,
        &self.mtp_window_idx,
        self.mtp_ev_m_lo_prev,
        drafted,
        accepted,
    ) orelse return;
    if (rate < MTP_DISABLE_BELOW) {
        log.info(
            "  mtp=disabled (EV: depth-1 first-draft rate {d:.2} < {d:.2})\n",
            .{ rate, MTP_DISABLE_BELOW },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .acceptance;
    }
}
// ── tests ──
