//! copied_from: src/engines/mlx/spec/mtp/depth.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MTP draft depth and proposal policy (greedy tail, draft temperature, depth decisions).
const depth_bounds = @import("../mtp_depth_bounds.zig");
const log = @import("../../../../sdk/util/log.zig");
const Generator = @import("../../generate/generator.zig").Generator;


// ── MTP adaptive depth ──
// Unlike the drafter's binary gate, the MTP head has a useful fallback
// BETWEEN "full depth" and "off": depth 1. Measured on Qwen3.6-27B
// (M4 Max, greedy): creative content runs 48% per-draft at depth 2
// (a regression vs AR) but 73% at depth 1 (1.11× AR); code runs 89% at
// depth 2 (1.45× AR). A windowed controller demotes/promotes between
// depths and only disables outright when even depth 1 can't pay for its
// verify overhead.
pub const MTP_DEPTH_WINDOW: u32 = 16; // rounds in the moving window
pub const MTP_DEPTH_SWITCH_WARMUP: u32 = 5; // rounds before re-evaluating after a switch
// Thresholds assume the capture-based rollback (no re-forward on partial
// accept): a rejected draft costs ONLY its own MTP-layer + draft-head
// pass (~2 ms), not a second trunk forward (~30-50 ms). Extra depth pays
// whenever the marginal accept probability clears draft-cost/trunk-cost
// ≈ 0.05-0.10, so the demote floor sits far lower than the old
// re-forward-era 0.60 — hysteresis band keeps switch churn down.
pub const MTP_DEMOTE_BELOW: f32 = 0.40; // per-draft rate at depth > 1 → step down
pub const MTP_PROMOTE_ABOVE: f32 = 0.60; // per-draft rate below configured depth → step up
// Disable floor = the MEASURED depth-1 breakeven plus margin, not a
// quality judgment: a d1 round costs ~AR+6 ms (44 vs 38.4 ms at 8K on
// the 27B, mtp-trace) and yields (1+p) tokens, so speculation
// pays down to p ≈ 0.15 — at p=0.45 it is +27% over AR. The old 0.50
// floor sticky-disabled mid-request on the oQ4e head at long context
// (window rate dips 0.45-0.55) and cratered 16K/32K ladder decode to
// bare AR (24-26 tok/s vs oMLX's 41-47, which never fully disables).
pub const MTP_DISABLE_BELOW: f32 = 0.20; // per-draft rate at depth 1 → disable (sticky)
pub const MTP_PROMOTE_COOLDOWN: u32 = 32; // rounds promotion stays blocked after a demotion

// ── Sharpened stochastic draft proposals (Lightning-class acceptance) ──
// Drafts for a stochastic target are SAMPLED from a fixed sharper
// distribution (constants mirror oMLX's _DRAFT_SAMPLER_*: their comment —
// matched-temp drafting "collapses to ~10-20% on high-entropy content"),
// acceptance is the full Leviathan/Chen ratio min(1, p/q) with q = the
// draft sampler's own filtered distribution, and rejection re-samples
// from normalize(max(p-q, 0)). Output distribution provably equals the
// target's filtered p for ANY proposal q (pinned by the toy-vocab
// exactness test); q only moves the ACCEPTANCE RATE, which is why the
// draft head's quantization never affects correctness.

/// Pure depth-policy step. `rate` is the windowed per-draft acceptance
/// probability. Returns the new depth; 0 means "disable speculation".
pub fn mtpNextDepth(current: u32, configured: u32, rate: f32) u32 {
    if (current > 1 and rate < MTP_DEMOTE_BELOW) return current - 1;
    if (current <= 1 and rate < MTP_DISABLE_BELOW) return 0;
    if (current < configured and rate > MTP_PROMOTE_ABOVE) return current + 1;
    return current;
}

/// Confidence-gated depth decision. Demoting is cheap (still
/// speculating) so it reacts on a small sample; DISABLING is sticky and
/// PROMOTING raises verify cost, so both require more evidence. A
/// 16-round window at a true 73% per-draft rate essentially never dips
/// below the 0.50 disable floor; a 5-round window does (observed live:
/// an early-story cold streak disabled a request that would have run
/// 1.11x at depth 1).
pub fn mtpDepthDecision(current: u32, configured: u32, rate: f32, window_rounds: u32, promote_blocked: bool) u32 {
    const next_depth = mtpNextDepth(current, configured, rate);
    if (next_depth == 0 and window_rounds < MTP_DEPTH_WINDOW) return current;
    if (next_depth > current and (window_rounds < 8 or promote_blocked)) return current;
    return next_depth;
}

/// Windowed adaptive-depth update, called once per nextMtp round with
/// that round's (drafted, accepted) counts.
pub fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
    const idx = self.mtp_window_idx % MTP_DEPTH_WINDOW;
    self.mtp_window_drafted[idx] = @intCast(drafted);
    self.mtp_window_accepted[idx] = @intCast(accepted);
    self.mtp_window_idx += 1;
    if (self.mtp_promote_cooldown > 0) self.mtp_promote_cooldown -= 1;
    if (self.mtp_rounds_since_switch < MTP_DEPTH_SWITCH_WARMUP) {
        self.mtp_rounds_since_switch += 1;
        return;
    }
    const n = @min(self.mtp_window_idx, MTP_DEPTH_WINDOW);
    var drafted_sum: u32 = 0;
    var accepted_sum: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        drafted_sum += self.mtp_window_drafted[i];
        accepted_sum += self.mtp_window_accepted[i];
    }
    if (drafted_sum == 0) return;
    const rate = @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(drafted_sum));
    const decided = mtpDepthDecision(self.mtp_depth_current, self.mtp_depth, rate, n, self.mtp_promote_cooldown > 0);
    const next_depth = if (decided == 0) 0 else depth_bounds.clampWidth(decided, depth_bounds.floorFor(depth_bounds.active, self.mtp_depth), self.mtp_depth);
    if (next_depth == self.mtp_depth_current) return;
    if (next_depth == 0) {
        log.info(
            "  mtp=disabled (windowed per-draft rate {d:.2} < {d:.2} at depth 1)\n",
            .{ rate, MTP_DISABLE_BELOW },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .acceptance;
        return;
    }
    log.debug("  [mtp-depth] {d} -> {d} (windowed per-draft rate {d:.2})\n", .{ self.mtp_depth_current, next_depth, rate });
    if (next_depth < self.mtp_depth_current) self.mtp_promote_cooldown = MTP_PROMOTE_COOLDOWN;
    self.mtp_depth_current = next_depth;
    self.mtp_rounds_since_switch = 0;
    // Reset both windows so the new depth is judged on its own rounds.
    self.mtp_window_idx = 0;
    self.mtp_price.reset();
}

// ── MTP EV (expected-value) adaptive controller ──
// Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
// marginal chain decays with index, so one global depth wastes verify
// width on hard stretches and leaves easy stretches (code boilerplate
// where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
// per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
// plans each round as two chunks: a base chunk `m_lo` (the static EV
// optimum), then — when the head's own confidence on chunk A clears a
// cost-derived threshold tau — an extension to `m_hi`. Only rounds that
// CONSIDER extension pay the one bounded chunk-A sync; when the plan
// collapses to m_lo == m_hi the round is byte-identical in shape to the
// fixed-depth path (no confidence graph, no sync).

/// Default depth cap when `--mtp-max-depth` is not passed (0 = auto) and the
/// EV controller is active. 6 keeps the verify forward at seq 1+6 = 7,
/// the split-K verify-qmm kernel's ceiling on M1-M4. Eligible M5/G17
/// targets use MTP_ADAPTIVE_NAX_CAP instead: their measured NAX round-cost
/// surface makes depths 7/8 profitable. Explicit depths always win.
/// This is the DEFAULT ROW only: `mtp.adaptiveDepthCapForMachine` lowers
/// it on chips whose verify-width cliff was measured (M1 Pro: 4).
pub const MTP_ADAPTIVE_DEFAULT_CAP: u32 = 6;
pub const MTP_ADAPTIVE_NAX_CAP: u32 = 8;
/// One-shot guard for the per-silicon cap log (the row is a property of
/// the machine, so it is worth saying once per process, not per request).
pub var mtp_depth_cap_logged: bool = false;
/// Rounds of legacy (fixed-depth windowed) behavior while the EMAs fill.
/// Warmup, but converges in ROUNDS, not 43 s of offline calibration.
pub const MTP_EV_WARMUP_ROUNDS: u32 = 10;
/// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
/// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
/// single unlucky round move the plan.
pub const MTP_EV_EMA_BETA: f32 = 0.15;
/// Optimistic prior for unobserved indices. Deliberately ABOVE the
/// measured average per-draft rate (~77%): a deep index is only ever
/// observed when extension fires, and on this cost surface the
/// break-even conditional acceptance for a ramp position is ~0.78 — a
/// realistic prior would sit razor-under it and extension would never
/// get its first trial (measured live: ext_rounds=0 on a pure-echo
/// workload). The tau gate (only near-perfect-confidence rounds extend)
/// plus demote-fast EMAs bound the cost of an optimistic trial to a few
/// rounds.
pub const MTP_EV_PRIOR: f32 = 0.85;
/// Clamp band for the extension confidence threshold.
pub const MTP_EV_TAU_MIN: f32 = 0.05;
pub const MTP_EV_TAU_MAX: f32 = 0.95;
/// Minimum base-round EV (tokens per round-cost) for the exploration
/// horizon: below this the base itself barely beats AR, so keeping an
/// extension position reachable would only add sync tax to a dying
/// speculation (the sticky-disable floor is likely next anyway). 1.10
/// keeps fully-cold EMAs collapsed on every cost profile (cold best_r:
/// generic 1.00, G17-NAX 1.07/1.08).
pub const MTP_EV_EXPLORE_MIN_R: f32 = 1.10;

// ── tests ──

