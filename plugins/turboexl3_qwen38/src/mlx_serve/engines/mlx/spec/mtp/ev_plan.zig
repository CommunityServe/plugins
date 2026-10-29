//! copied_from: src/engines/mlx/spec/mtp/ev_plan.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MTP EV round planning: depth caps, cost sources, depth policy.
const depth_bounds = @import("../mtp_depth_bounds.zig");
const std = @import("std");
const env = @import("../../../../env.zig");
const transformer_mod = @import("../../transformer.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");
const round_cost = @import("../round_cost.zig");
const ane_mod = @import("../../../../sdk/ane/offload.zig");
const Transformer = transformer_mod.Transformer;
const testing = std.testing;
const depth_mod = @import("depth.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_DEFAULT_CAP = @import("depth.zig").MTP_ADAPTIVE_DEFAULT_CAP;
const MTP_ADAPTIVE_NAX_CAP = @import("depth.zig").MTP_ADAPTIVE_NAX_CAP;
const MTP_EV_EXPLORE_MIN_R = @import("depth.zig").MTP_EV_EXPLORE_MIN_R;
const MTP_EV_TAU_MAX = @import("depth.zig").MTP_EV_TAU_MAX;
const MTP_EV_TAU_MIN = @import("depth.zig").MTP_EV_TAU_MIN;
const MtpEvCosts = @import("ev_costs.zig").MtpEvCosts;
const init = @import("../../generate/generator.zig").Generator.init;


/// Write the model's round-cost table to disk when this request folded
/// new samples (request end; inference thread, so no lock).
pub fn persistRoundCost(self: *Generator) void {
    const t = &self.xfm.round_cost;
    const folded = round_cost.totalFolded(t);
    if (folded == t.stored_at) return;
    round_cost.storeCached(self.timer.io, self.xfm.round_cost_key_buf[0..self.xfm.round_cost_key_len], t);
    t.stored_at = folded;
}

/// KV length the next verify forward reads.
pub fn mtpKvLen(self: *const Generator) u32 {
    return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
}

/// Marginal round cost of draft position k (1-based), kv-blind. This is
/// the fitted SHAPE; `mtpEvMarginalCostAt` applies the kv term.
pub fn mtpEvMarginalCost(costs: MtpEvCosts, k: u32) f32 {
    const verify_cost = if (costs.nax_from != 0 and k >= costs.nax_from)
        costs.per_pos_nax
    else if (k <= costs.flat_max)
        costs.per_pos_lo
    else
        costs.per_pos_hi;
    return costs.draft + verify_cost;
}

/// One round's draft plan. `m_hi > m_lo` means "pay the chunk-A sync and
/// extend to m_hi when the chain log-confidence clears tau_ln".
pub const MtpRoundPlan = struct {
    m_lo: u32,
    m_hi: u32,
    tau_ln: f32,
    /// A width-trial round (single-chunk at m_lo+1, measuring the round
    /// cost table): skipped by the regime gate, which compares shapes
    /// at ONE base depth.
    width_trial: bool = false,
    /// Set only on a two-chunk plan `ev_fix.gate` kept: the ideal gate's firing probability and the extension tokens it
    /// promises, which the round's outcome folds against. 0 = not priced.
    fire_ideal: f32 = 0,
    gain_ideal: f32 = 0,
};

/// Resolve the configured depth cap. 0 = auto (`--mtp-max-depth` not passed):
/// MTP_ADAPTIVE_NAX_CAP only when the EV controller and a calibrated G17
/// cost profile are both active, MTP_ADAPTIVE_DEFAULT_CAP otherwise, and
/// DEFAULT_DEPTH in fixed mode. Explicit values always win.
pub fn mtpDepthCapForProfile(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile) u32 {
    const chip = ane_mod.chipBrand();
    const cap = mtpDepthCapResolved(configured, adaptive, profile, chip);
    // Name the row ONCE when a per-silicon measurement is what fenced the
    // depth. Without it `[spec-stats] depth=4` on an M1 Pro reads the same
    // as the EV controller having chosen 4, or as `--mtp-max-depth 4` — and
    // the fence is exactly what someone debugging MTP there needs to see.
    if (configured == 0 and adaptive and !depth_mod.mtp_depth_cap_logged) {
        const row = mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP);
        if (cap == row.cap and row.cap < MTP_ADAPTIVE_DEFAULT_CAP) {
            depth_mod.mtp_depth_cap_logged = true;
            log.info("[mtp] adaptive depth cap {d} ({s} row, default {d})\n", .{
                row.cap,
                row.label,
                MTP_ADAPTIVE_DEFAULT_CAP,
            });
        }
    }
    return depth_bounds.liftCap(cap, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// The cap with no per-silicon row applied: explicit wins, else the
/// adaptive default (fixed mode keeps DEFAULT_DEPTH).
pub fn mtpDepthCapFree(configured: u32) u32 {
    const free = if (configured != 0) @min(mtp_mod.MAX_DEPTH, @max(1, configured)) else MTP_ADAPTIVE_DEFAULT_CAP;
    return depth_bounds.liftCap(free, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// Same, with the chip string injected (tests, and the one live caller).
pub fn mtpDepthCapForProfileChip(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile, chip: []const u8) u32 {
    return mtpDepthCapResolved(configured, adaptive, profile, chip);
}

/// Same, with the machine's MEASURED verify ladder when one exists.
///
/// The per-silicon rows exist because the EV controller cannot see time:
/// it scores accepted-tokens-per-round, which on an M1 Pro IMPROVED
/// (3.65 -> 4.00) while realized tok/s fell 21%. A measured curve sees
/// exactly the cliff those rows encode, on whatever chip and whatever
/// quant geometry is actually resident — which the chip key cannot
/// express (the split-K lane is 4-bit/g64 only, so a 6-bit pack on the
/// same box is a different answer). The calibrated NAX profiles keep
/// their own cap until a probe on that silicon is validated against it.
pub fn mtpDepthCapResolved(
    configured: u32,
    adaptive: bool,
    profile: mtp_mod.MtpCostProfile,
    chip: []const u8,
) u32 {
    if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
    if (!adaptive) return mtp_mod.DEFAULT_DEPTH;
    return switch (profile) {
        // The per-silicon row is the COLD-START cap: the measured
        // round-cost table may plan above it on trusted widths.
        .generic => mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP).cap,
        .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 => MTP_ADAPTIVE_NAX_CAP,
        // The dense 27B q4/gs64 surface stops paying past depth 6 on M5.
        .g17_nax_q4_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
        // qwen4's measured surface has no NAX region to reach: depths 7-8
        // price at .345/pos against sub-60% tail acceptance even on
        // saturated echo, so the calibrated cap keeps the default — the
        // NAX cap exists for surfaces that flatten past position 6.
        .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
    };
}

pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
    return mtpDepthCapForProfile(configured, true, profile);
}

/// Legacy q8 boolean selector retained for source compatibility.
pub fn mtpDepthCapFor(configured: u32, adaptive: bool, nax_profile: bool) u32 {
    const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
    return mtpDepthCapForProfile(configured, adaptive, profile);
}

/// Legacy q8 boolean selector retained for source compatibility.
pub fn resolveMtpDepthCap(configured: u32, nax_profile: bool) u32 {
    const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
    return resolveMtpDepthCapForProfile(configured, profile);
}

/// Expected committed tokens for an m-deep round: the always-committed t1
/// plus the acceptance chain sum (draft k lands iff drafts 0..k all land).
pub fn mtpEvExpectedTokens(a: []const f32, m: u32) f32 {
    var chain: f32 = 1.0;
    var tok: f32 = 1.0;
    var k: u32 = 0;
    while (k < m and k < a.len) : (k += 1) {
        chain *= a[k];
        tok += chain;
    }
    return tok;
}

pub fn mtpEvMarginalCostAt(costs: MtpEvCosts, k: u32, kv_len: u32) f32 {
    _ = kv_len;
    return mtpEvMarginalCost(costs, k);
}

/// Round cost in verify-base units (piecewise per-position marginals).
pub fn mtpEvRoundCost(costs: MtpEvCosts, m: u32, with_sync: bool) f32 {
    return mtpEvRoundCostAt(costs, m, with_sync, 0);
}

pub fn mtpEvRoundCostAt(costs: MtpEvCosts, m: u32, with_sync: bool, kv_len: u32) f32 {
    var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
    var k: u32 = 1;
    while (k <= m) : (k += 1) c += mtpEvMarginalCostAt(costs, k, kv_len);
    return c;
}

/// Pure EV plan: pick (m_lo, m_hi, tau) maximizing expected tok/round-cost.
/// `m_lo_max` damps the base-depth climb (hysteresis — the caller passes
/// last round's m_lo + 1); demotions are never damped.
///  1. m_lo = argmax over single-chunk depths of E(m)/T(m).
///  2. m_hi = deepest position whose marginal chain still pays under FULL
///     confidence in chunk A (the best case the gate can certify).
///  3. tau: extend when the confidence-implied chain beats the stop rate
///     on the margin — c*S/dt > r  =>  tau = r*dt/S.
/// There is deliberately NO separate "is the sync worth it" gate: tau
/// already keeps low-confidence rounds single-chunk, the horizon check
/// collapses m_hi on cold EMAs (killing the sync entirely), and a
/// prior-weighted expected-gain gate measurably starves exploration —
/// deep indices are only observed when extension fires, so a gate fed by
/// their priors blocks the first trial forever (live: ext_rounds=0 on
/// pure echo).
pub fn mtpEvPlanFor(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32) MtpRoundPlan {
    return mtpEvPlanForAt(a, cap_in, costs, m_lo_max, 0);
}

pub fn mtpEvPlanForAt(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32, kv_len: u32) MtpRoundPlan {
    return mtpEvPlanSrc(a, cap_in, MtpCostSource.init(costs, kv_len, null), m_lo_max);
}

/// Smallest marginal the table may report, in floor units: a measured
/// pair that reads a wider round CHEAPER is noise, and a non-positive
/// marginal would make every extension free.
pub const MTP_EV_TABLE_MIN_MARGINAL: f32 = 0.02;

/// Where the EV plan's round costs come from: the measured round-cost
/// table when the kv bucket (or its nearest active neighbour) has
/// MIN_WIDTHS measured widths, else the fitted surface as the cold-start
/// prior. The table is in ms; the plan compares ratios but has one
/// absolute threshold (`MTP_EV_EXPLORE_MIN_R`), so the table is scaled
/// into FLOOR UNITS at its narrowest measured width — the two sources
/// agree there by construction and the table's slopes take over.
pub const MtpCostSource = struct {
    costs: MtpEvCosts,
    kv_len: u32 = 0,
    table: ?*const round_cost.Table = null,
    bucket: usize = 0,
    scale: f32 = 0,
    /// False = the table prices round TIME only and tokens come from the acceptance EMAs.
    table_tokens: bool = true,

    pub fn init(costs: MtpEvCosts, kv_len: u32, table: ?*const round_cost.Table) MtpCostSource {
        var src = MtpCostSource{ .costs = costs, .kv_len = kv_len, .table = table };
        const t = table orelse return src;
        const b = t.bucketToRead(kv_len) orelse return src;
        const ref = t.narrowestMeasured(b) orelse return src;
        const ref_ms = t.measuredMs(ref, b) orelse return src;
        if (!(ref_ms > 0) or ref > mtp_mod.MAX_DEPTH) return src;
        src.bucket = b;
        src.scale = mtpEvRoundCostAt(costs, ref, false, kv_len) / ref_ms;
        return src;
    }

    pub fn fromTable(self: MtpCostSource) bool {
        return self.scale > 0;
    }

    /// Realized tokens per single-chunk round at a MEASURED width, else
    /// null (the acceptance-EMA model fills in). The model's E(6) on the
    /// M4 Max @16k said 6.85 while the measured single-6 round emitted
    /// 6.0 — the 6th draft's rejections cost a rollback the model cannot
    /// see, and the table can.
    pub fn measuredTokens(self: MtpCostSource, m: u32) ?f32 {
        if (self.scale <= 0 or !self.table_tokens) return null;
        return self.table.?.measuredTok(m, self.bucket);
    }

    pub fn roundCost(self: MtpCostSource, m: u32, with_sync: bool) f32 {
        if (self.scale > 0) {
            const sync: f32 = if (with_sync) self.costs.sync else 0.0;
            const t = self.table.?;
            if (t.roundMs(m, self.bucket)) |ms| return ms * self.scale + sync;
            // Past the widest measured width every extra position costs
            // max(last measured slope, prior marginal): continuous with
            // the table, never more optimistic than the prior (a shallow
            // measured slope run upward let the plan race to 8 unmeasured).
            if (t.widestMeasured(self.bucket)) |w| {
                if (m > w) {
                    const slope = (t.lastSlope(self.bucket) orelse 0.0) * self.scale;
                    var c = t.measuredMs(w, self.bucket).? * self.scale;
                    var k: u32 = w + 1;
                    while (k <= m) : (k += 1) {
                        c += @max(slope, mtpEvMarginalCostAt(self.costs, k, self.kv_len));
                        // One sample is evidence for WORSE, never for
                        // cheaper: an untrusted cell floors the cost. Round TIME
                        // alone is stable to a few percent, so where no token
                        // column is read the sample prices its width.
                        if (t.rawMs(k, self.bucket)) |raw| c = if (self.table_tokens) @max(c, raw * self.scale) else raw * self.scale;
                    }
                    return c + sync;
                }
            }
        }
        return mtpEvRoundCostAt(self.costs, m, with_sync, self.kv_len);
    }

    pub fn marginal(self: MtpCostSource, k: u32) f32 {
        if (self.scale > 0 and k >= 1) {
            return @max(self.roundCost(k, false) - self.roundCost(k - 1, false), MTP_EV_TABLE_MIN_MARGINAL);
        }
        return mtpEvMarginalCostAt(self.costs, k, self.kv_len);
    }
};

/// How the base depth is chosen. `.accept`: expected tokens come from the acceptance EMAs
/// (every round refreshes every index it drafted) and the table prices round TIME only, in
/// ONE chunk, so no round reads confidences mid-round. `.legacy`: the table's realized
/// tokens where measured, plus the confidence-gated chunk B.
pub const MtpDepthPolicy = enum { accept, legacy };
/// `.accept` serves KV lengths below this. It was fitted where a round is CPU-build bound and
/// the chunk-A confidence read costs its whole wait; from here up the verify forward dominates,
/// the read overlaps GPU work, and a shallow base + chunk B undercuts every single-chunk
/// depth, so the legacy planner stays.
pub const MTP_ACCEPT_MAX_KV: u32 = 8192;

pub fn mtpDepthPolicyFor(configured: MtpDepthPolicy, kv_len: u32) MtpDepthPolicy {
    return if (kv_len >= MTP_ACCEPT_MAX_KV) .legacy else configured;
}

/// Test seam: forces the configured policy (past `MTP_ACCEPT_MAX_KV` it is legacy either way).
pub var mtp_depth_policy_override: ?MtpDepthPolicy = null;
pub fn mtpDepthPolicy() MtpDepthPolicy {
    return mtp_depth_policy_override orelse .accept;
}

pub fn mtpBasePlan(policy: MtpDepthPolicy, a: []const f32, cap: u32, src_in: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
    var src = src_in;
    src.table_tokens = policy == .legacy;
    var plan = mtpEvPlanSrc(a, cap, src, m_lo_max);
    if (policy == .accept) {
        plan.m_hi = plan.m_lo;
        plan.tau_ln = 0.0;
    }
    return plan;
}

pub fn mtpEvPlanSrc(a: []const f32, cap_in: u32, src: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
    const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
    const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
    var m_lo: u32 = 1;
    var best_r: f32 = 0.0;
    // With measured costs the standing base (m_lo_max - 1, what the
    // caller planned last round) keeps its place unless a challenger
    // beats it by the switch margin: cells are EMAs of a few samples
    // and a 5% reversal between neighbours is noise, while every
    // change of base is a transition round the table will not count
    // and the model pays for.
    const standing: u32 = if (src.fromTable()) @min(lo_cap, m_lo_max -| 1) else 0;
    var standing_r: f32 = 0.0;
    var m: u32 = 1;
    while (m <= lo_cap) : (m += 1) {
        const tok = src.measuredTokens(m) orelse mtpEvExpectedTokens(a, m);
        const r = tok / src.roundCost(m, false);
        if (m == standing) standing_r = r;
        if (r > best_r) {
            best_r = r;
            m_lo = m;
        }
    }
    if (standing >= 1 and m_lo != standing and best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) {
        m_lo = standing;
        best_r = standing_r;
    }
    if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
    var m_hi: u32 = m_lo;
    var cond: f32 = 1.0;
    var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
    var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
    while (m_hi < cap) {
        cond *= a[m_hi];
        const mc = src.marginal(m_hi + 1);
        if (cond <= best_r * mc) break;
        s_sum += cond;
        t_sum += mc;
        m_hi += 1;
    }
    if (m_hi == m_lo) {
        // Exploration valve for the horizon itself: the
        // index one past m_lo is observable ONLY when extension fires —
        // at m_lo-deep rounds a[m_lo] never updates — so an EMA dragged
        // cold under an earlier workload (or an easier tau regime)
        // would otherwise close the horizon FOREVER: m stays at m_lo,
        // a[m_lo] never re-proves itself, and pure echo runs at
        // ext_rounds=0 (the exact class the TAU_MAX valve exists for,
        // resurfaced by refit #3's honest marginals). Whenever the base
        // itself pays, keep ONE extension position reachable at the
        // clamped tau; the dry-spell gate bounds the consideration tax
        // on workloads where confidence never clears it.
        if (best_r <= MTP_EV_EXPLORE_MIN_R) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const mc = src.marginal(m_lo + 1);
        // A MEASURED marginal the position cannot repay even at full
        // confidence (1 <= best_r * mc) closes the valve: the valve
        // exists to observe a[m_lo], and with the table that is the
        // width trial's job. The fitted prior keeps the valve exactly
        // as Phase 1 measured it.
        if (src.fromTable() and best_r * mc >= 1.0) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const s = @max(a[m_lo], 1e-6);
        const tau_x = std.math.clamp(best_r * mc / s, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_lo + 1, .tau_ln = @log(tau_x) };
    }
    // The TAU_MAX clamp doubles as the exploration valve: on razor-thin
    // horizons the honest tau approaches 1 ("never extend"), and 0.95
    // lets near-perfect-confidence rounds through so the deep EMAs can
    // observe reality at all.
    const tau = std.math.clamp(best_r * t_sum / s_sum, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
    return .{ .m_lo = m_lo, .m_hi = m_hi, .tau_ln = @log(tau) };
}

/// Update the conditional acceptance EMAs from one realized round.
/// Acceptance is prefix-structured: indices < accepted saw a success, the
/// index AT `accepted` saw the reject (when one happened), and deeper
/// indices were never conditionally reached — no observation.
pub fn mtpEvObserve(a: []f32, drafted: u32, accepted: u32, beta: f32) void {
    var i: usize = 0;
    while (i < accepted and i < a.len) : (i += 1) a[i] += beta * (1.0 - a[i]);
    if (accepted < drafted and accepted < a.len) a[accepted] += beta * (0.0 - a[accepted]);
}

/// Chain log-confidence of a drafted chunk: sum of per-draft log p_head,
/// clamped to <= 0 per term (a log-prob is never positive; bf16 noise
/// can be). NaN poisons to -inf so a broken confidence can never extend.
pub fn mtpChainLogConf(confs: []const f32) f32 {
    var sum: f32 = 0.0;
    for (confs) |c| {
        if (std.math.isNan(c)) return -std.math.inf(f32);
        sum += @min(0.0, c);
    }
    return sum;
}

// ── tests ──


pub fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = @splat(0);
    g.mtp_window_accepted = @splat(0);
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = @splat(Generator.MTP_EV_PRIOR);
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

pub fn mtpBoundsPlanFixture(xfm: *Transformer, prompt: []u32, depth: u32) Generator {
    xfm.config.model_type = "qwen4_exp";
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = xfm.round_cost.observe(2, 1000, 29.0, 2.6, true, false);
        _ = xfm.round_cost.observe(3, 1000, 31.0, 2.9, true, false);
        _ = xfm.round_cost.observe(4, 1000, 36.0, 3.1, true, false);
        _ = xfm.round_cost.observe(5, 1000, 42.0, 3.5, true, false);
    }
    return Generator{
        .xfm = xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .prompt_ids_owned = prompt,
        .mtp_depth = depth,
        .mtp_depth_free = depth,
        .mtp_ev_rounds = 64,
        .mtp_ev_accept = .{ 0.92, 0.9, 0.88, 0.86, 0.84, 0.8, 0.8, 0.8 },
        .mtp_ev_m_lo_prev = 4,
        .spec_cost_solo = false,
    };
}
