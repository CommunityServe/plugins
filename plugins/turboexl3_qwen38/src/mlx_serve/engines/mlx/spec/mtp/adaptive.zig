//! copied_from: src/engines/mlx/spec/mtp/adaptive.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Adaptive MTP serial switching: price window, votes, probes, head release.
const std = @import("std");
const env = @import("../../../../env.zig");
const log = @import("../../../../sdk/util/log.zig");
const round_cost = @import("../round_cost.zig");
const testing = std.testing;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_CONFIRM = @import("regime.zig").MTP_ADAPTIVE_CONFIRM;
const MTP_ADAPTIVE_MARGIN = @import("regime.zig").MTP_ADAPTIVE_MARGIN;
const MTP_ADAPTIVE_PROBE_TOKENS = @import("regime.zig").MTP_ADAPTIVE_PROBE_TOKENS;
const MtpHeadRef = @import("../mtp_head_ref.zig").MtpHeadRef;
const mtpRoundOff0 = @import("chain.zig").mtpRoundOff0;

pub const MTP_ADAPTIVE_PROBE_WARM: u32 = 2;
/// KV below which the switch does not exist: no vote, no probe. Rounds lose to serial from
/// the 32-64k bucket up; below that the probes cost more than a switch could buy.
pub const MTP_ADAPTIVE_MIN_KV: u32 = 32768;

/// May the adaptive switch run at this context at all? Read before the vote and the probe.
pub fn mtpAdaptiveKvEligible(kv_len: u32, min_kv: u32) bool {
    return kv_len >= min_kv;
}

/// The bucket the adaptive switch speaks in. Every site resolves it the same way (the
/// decision and the re-entry once disagreed and oscillated); the fallback is the table's
/// own grid, never the free `bucketFor`.
pub fn mtpAdaptiveBucketOf(read: ?usize, kv_len: u32, layout: round_cost.Layout) usize {
    return read orelse round_cost.bucketForLayout(kv_len, layout);
}

pub fn mtpAdaptiveBucket(self: *const Generator, kv_len: u32) usize {
    const t = &self.xfm.round_cost;
    return mtpAdaptiveBucketOf(t.bucketToRead(kv_len), kv_len, t.layout);
}

/// Must the realized price window be dropped? It prices one arm in one bucket.
pub fn mtpAdaptiveRegimeMoved(
    prev_bucket: ?usize,
    prev_arm: MtpAdaptiveArm,
    bucket: ?usize,
    arm: MtpAdaptiveArm,
) bool {
    if (prev_arm != arm) return true;
    if (prev_bucket == null and bucket == null) return false;
    if (prev_bucket == null or bucket == null) return true;
    return prev_bucket.? != bucket.?;
}

/// Apply the rule above around a `MtpAdaptive` transition (both call sites).
pub fn mtpAdaptiveSyncWindow(self: *Generator, prev_bucket: ?usize, prev_arm: MtpAdaptiveArm) void {
    if (mtpAdaptiveRegimeMoved(prev_bucket, prev_arm, self.mtp_adaptive.bucket, self.mtp_adaptive.arm)) {
        self.mtp_price.reset();
    }
}

/// Rounds in the realized price window. Full-window-only: a restore's warmup rounds must not vote.
pub const MTP_PRICE_WINDOW: u32 = 16;

/// This request's realized ms per emitted token over a trailing window of speculative
/// rounds. Width trials are skipped, extension rounds kept. A new request starts empty.
pub const MtpPriceWindow = struct {
    ms: [MTP_PRICE_WINDOW]f32 = @splat(0),
    tok: [MTP_PRICE_WINDOW]f32 = @splat(0),
    idx: u32 = 0,
    filled: u32 = 0,

    pub fn reset(self: *MtpPriceWindow) void {
        self.idx = 0;
        self.filled = 0;
    }

    pub fn observe(self: *MtpPriceWindow, round_ms: f32, emitted: f32, width_trial: bool) void {
        if (width_trial) return;
        if (!std.math.isFinite(round_ms) or !(round_ms > 0)) return;
        if (!std.math.isFinite(emitted) or !(emitted > 0)) return;
        const i = self.idx % MTP_PRICE_WINDOW;
        self.ms[i] = round_ms;
        self.tok[i] = emitted;
        self.idx +%= 1;
        if (self.filled < MTP_PRICE_WINDOW) self.filled += 1;
    }

    /// Realized ms per emitted token, or null until the window is full. Summed, not averaged per round.
    pub fn msPerTok(self: *const MtpPriceWindow) ?f32 {
        if (self.filled < MTP_PRICE_WINDOW) return null;
        var ms: f32 = 0;
        var tok: f32 = 0;
        for (self.ms, self.tok) |m, t| {
            ms += m;
            tok += t;
        }
        if (!(ms > 0) or !(tok > 0)) return null;
        const v = ms / tok;
        return if (std.math.isFinite(v)) v else null;
    }
};

pub const MtpAdaptiveVote = enum { undecided, mtp, serial };
pub const MtpAdaptiveArm = enum { undecided, mtp, serial };
pub const MtpAdaptiveAction = enum { none, to_serial, to_mtp };
/// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
pub const MtpSerialExit = enum { none, drain, capture };

/// One round's vote. Both prices must be measured: the table's cell (cross-request, but
/// its `tok` column is a workload mixture) and this request's window (per-request, empty
/// until full). A modeled denominator under-predicted committed tokens by 12-31% and
/// switched where MTP was ~10% faster. `serial` needs BOTH; anything missing = `.undecided`.
pub fn mtpAdaptiveVoteFor(
    table_ms_tok: ?f32,
    window_ms_tok: ?f32,
    serial_ms_tok: ?f32,
    margin: f32,
) MtpAdaptiveVote {
    const tbl = table_ms_tok orelse return .undecided;
    const win = window_ms_tok orelse return .undecided;
    const ser = serial_ms_tok orelse return .undecided;
    if (!std.math.isFinite(tbl) or !(tbl > 0)) return .undecided;
    if (!std.math.isFinite(win) or !(win > 0)) return .undecided;
    if (!std.math.isFinite(ser) or !(ser > 0)) return .undecided;
    if (!std.math.isFinite(margin) or margin < 0) return .undecided;
    const bar = ser * (1.0 + margin);
    return if (tbl > bar and win > bar) .serial else .mtp;
}

/// Arm a bounded serial probe, returning the bucket it will measure. One bucket parameter
/// (the flag and the "already measured?" check once keyed on different buckets). The
/// sample lands in the bucket the request is IN.
/// A probe buys the LAST missing input: a request that cannot fill its price window can
/// never use what the probe teaches it and just pays 8 serial tokens.
pub fn mtpSerialProbeUseful(window_ms_tok: ?f32) bool {
    return window_ms_tok != null;
}

pub fn mtpSerialProbeArm(
    t: *round_cost.Table,
    bucket: usize,
    solo: bool,
    idle: bool,
    may_resume: bool,
    useful: bool,
) ?usize {
    if (!solo or !idle or !may_resume or !useful) return null;
    if (bucket >= round_cost.N_BUCKETS) return null;
    if (t.serialMsPerTok(bucket) != null) return null;
    // Bounded retries, not a one-shot flag: an interrupted probe must not burn the bucket's only chance.
    if (t.serial_probes[bucket] >= round_cost.MAX_SERIAL_PROBES) return null;
    t.serial_probes[bucket] += 1;
    return bucket;
}

/// The standing choice and the streak behind it. Pure.
pub const MtpAdaptive = struct {
    arm: MtpAdaptiveArm = .undecided,
    confirm: u32 = 0,
    bucket: ?usize = null,
    /// A re-entry was refused because the head's position bookkeeping is out of sync; latched.
    reentry_declined: bool = false,
    /// The switch left MTP on a module-owned head, so the arm is sticky: no re-entry, probe
    /// or round for the rest of the request. That is what buys the head's release.
    sticky_serial: bool = false,
    /// Round index of the last `round()`: `mtpRoundPlan` has two call sites per round.
    last_round: ?u32 = null,
    switches: u32 = 0,

    pub fn round(self: *MtpAdaptive, round_idx: u32, bucket: usize, vote: MtpAdaptiveVote, need: u32) MtpAdaptiveAction {
        if (self.sticky_serial) return .none;
        if (self.last_round) |r| {
            if (r == round_idx) return .none;
        }
        self.last_round = round_idx;
        // A crossing invalidates the streak.
        if (self.bucket == null or self.bucket.? != bucket) {
            self.bucket = bucket;
            self.confirm = 0;
        }
        switch (vote) {
            .undecided => {
                self.confirm = 0;
                return .none;
            },
            .mtp => {
                self.confirm = 0;
                self.arm = .mtp;
                return .none;
            },
            .serial => {
                self.confirm += 1;
                if (self.confirm < @max(@as(u32, 1), need)) return .none;
                self.confirm = 0;
                self.arm = .serial;
                self.switches += 1;
                return .to_serial;
            },
        }
    }

    /// Latch a refused re-entry: back to the serial arm, never ask again for this request.
    pub fn declineReentry(self: *MtpAdaptive) void {
        self.arm = .serial;
        self.reentry_declined = true;
    }

    /// Make the serial arm permanent for this request (a `to_serial` on a module-owned head).
    pub fn stickSerial(self: *MtpAdaptive) void {
        self.arm = .serial;
        self.sticky_serial = true;
    }

    /// Re-open the decision once the context crosses into another bucket; never inside one: a
    /// re-entry there resumes a head whose history did not grow across the serial block.
    pub fn serialTick(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
        // Sticky outranks the bucket crossing.
        if (self.sticky_serial) return .none;
        if (self.reentry_declined) return .none;
        if (self.arm != .serial) return .none;
        if (self.bucket) |b| if (b == bucket) return .none;
        return self.reopen(bucket);
    }

    fn reopen(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
        self.bucket = bucket;
        self.confirm = 0;
        self.arm = .undecided;
        self.switches += 1;
        return .to_mtp;
    }
};

/// Drift between the head's position bookkeeping and what its next forward demands
/// (`pos_offset == pos_base + seq_offset`; `seq_offset == 0` re-seeds). 0 = in sync.
pub fn mtpHeadPositionDrift(pos_base: i64, seq_offset: usize, next_pos_offset: i64) i64 {
    if (seq_offset == 0) return 0;
    return next_pos_offset - (pos_base + @as(i64, @intCast(seq_offset)));
}

/// The drift the next round would hit. A serial block advances the trunk while the head's
/// `seq_offset` stays put. Null = no head state.
pub fn mtpHeadDriftNow(self: *const Generator) ?i64 {
    if (self.mtp_cache == null) return null;
    const mc = &self.mtp_cache.?;
    return switch (mc.*) {
        .qwen, .plugin => 0,
        .qwen4 => |r| blk: {
            r.t.qwen4MtpActivate(r.st);
            const m = &(r.t.qwen4_mtp orelse break :blk 0);
            const off0 = mtpRoundOff0(self.mtp_hist_stash, mc.step());
            break :blk mtpHeadPositionDrift(@intCast(m.pos_base), m.seq_offset, @intCast(off0 + 1));
        },
    };
}

/// May this request resume MTP after a serial block? An M-RoPE turn ropes the head at an
/// absolute position where the content gap is a wrong answer, so vision turns never come back.
pub fn mtpAdaptiveHeadMayResume(self: *const Generator) bool {
    // The one predicate all three round-start doors ask (re-entry, `apply_stash`, the probe).
    if (self.mtp_adaptive.sticky_serial) return false;
    return self.ctx.mrope_pos == null;
}

/// Does a `to_serial` on this head make the arm sticky? Only a module-owned head.
pub fn stickyOnSerialSwitch(head: ?MtpHeadRef) bool {
    const h = head orelse return false;
    return h.moduleOwned();
}

/// Does this tick newly release the module head? True exactly once per request.
pub fn stickyReleaseNow(sticky: bool, already_released: bool) bool {
    return sticky and !already_released;
}

/// Has this request released the module-owned MTP head?
pub fn mtpModuleHeadReleased(self: *const Generator) bool {
    return self.mtp_head_released;
}

/// Armed but not landed. The release lives in `nextMtp`'s serial branch and the batched
/// tick never calls `nextMtp`, so `scheduler.batchable` keeps the slot out of a group for the
/// one tick the release needs.
pub fn mtpReleasePending(self: *const Generator) bool {
    return self.mtp_adaptive.sticky_serial and !self.mtp_head_released;
}

/// Release the module-owned head at a serial block boundary, after `mtpDetachHead`; never
/// where the switch is decided (that round still runs).
pub fn mtpMaybeReleaseModuleHead(self: *Generator) void {
    if (!stickyReleaseNow(self.mtp_adaptive.sticky_serial, self.mtp_head_released)) return;
    self.mtp_head_released = true;
    log.info(
        "  [mtp] sticky-serial: module head released for the rest of the request (kv {d}) — other slots may claim it\n",
        .{self.mtpKvLen()},
    );
}

pub fn mtpAdaptiveArchEligible(self: *const Generator) bool {
    return self.mtp != null;
}

/// Rounds in a row with the tokens-per-round EMA (bonus included) under this before a
/// sidecar head may pay a serial probe.
pub const MTP_PROBE_LOW_TOK: f32 = 2.0;
pub const MTP_PROBE_LOW_ROUNDS: u32 = 16;

pub fn mtpLowAcceptStreak(streak: u32, round_tokens: f32) u32 {
    return if (round_tokens < MTP_PROBE_LOW_TOK) streak +| 1 else 0;
}

/// The module head's switch was calibrated with an immediate probe; a sidecar head wins
/// wherever it accepts, so only a run of near-serial rounds buys its probe.
pub fn mtpSerialProbeEarned(module_head: bool, low_streak: u32) bool {
    return module_head or low_streak >= MTP_PROBE_LOW_ROUNDS;
}

pub fn mtpAdaptiveSerialStep(self: *Generator, m_lo: u32, kv_len: u32) bool {
    // Read after the plan (m_lo is the width it prices) and before the width trial.
    if (        self.mtpAdaptiveArchEligible() and
        mtpAdaptiveKvEligible(kv_len, MTP_ADAPTIVE_MIN_KV))
    {
        const t = &self.xfm.round_cost;
        const b = self.mtpAdaptiveBucket(kv_len);
        const table_ms_tok = t.msPerTok(m_lo, b);
        const window_ms_tok = self.mtp_price.msPerTok();
        const serial_ms = t.serialMsPerTok(b);
        const vote = mtpAdaptiveVoteFor(table_ms_tok, window_ms_tok, serial_ms, MTP_ADAPTIVE_MARGIN);
        const prev_bucket = self.mtp_adaptive.bucket;
        const prev_arm = self.mtp_adaptive.arm;
        const action = self.mtp_adaptive.round(self.mtp_ev_rounds, b, vote, MTP_ADAPTIVE_CONFIRM);
        self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
        if (action == .to_serial) {
            // The round under construction still runs; the switch takes effect next tick.
            log.info(
                "  [mtp] adaptive: bucket {s} mtp table {d:.2} / window {d:.2} ms/tok (w{d}) vs serial {d:.2} ms/tok -> serial (from the next round)\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, b), table_ms_tok.?, window_ms_tok.?, m_lo, serial_ms.? },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .adaptive;
            // On a module-owned head the switch is one-way; the latch is consumed at the block boundary.
            if (stickyOnSerialSwitch(self.mtp)) self.mtp_adaptive.stickSerial();
            return true;
        }
        // Nothing to decide with: teach the bucket a serial token, once.
        const idle = self.mtp_serial_left == 0 and self.mtp_serial_exit == .none;
        const useful = mtpSerialProbeUseful(window_ms_tok) and mtpSerialProbeEarned(if (self.xfm.qwen4Trunk()) |trunk| trunk.qwen4_mtp != null else false, self.mtp_low_accept);
        if (mtpSerialProbeArm(t, b, self.spec_cost_solo, idle, self.mtpAdaptiveHeadMayResume(), useful)) |own| {
            self.mtp_serial_left = MTP_ADAPTIVE_PROBE_TOKENS;
            log.info(
                "  [mtp] adaptive: bucket {s} has no serial cell -> probing {d} serial tokens\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, own), MTP_ADAPTIVE_PROBE_TOKENS },
            );
            return true;
        }
    }
    return false;
}

// ── tests ──

