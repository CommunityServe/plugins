//! This plugin's own, in place of mlx-serve's `src/engines/mlx/generate/generator.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the part of a request's `Generator` that #1005's MTP round policy reads and writes,
//! with that policy's methods, which are the copies' own (`engines/mlx/spec/mtp/`: planner, depth, ev_plan, ev_fix,
//! regime, adaptive, observe, and chain's prompt lookup). Each field keeps the type and default #1005's Generator gives
//! it. The plugin's drafter (src/spec/drafter.zig) keeps one per slot and drives the round around it; `xfm` is the
//! model's share (`Model`).

const std = @import("std");
const io_util = @import("../../../sdk/util/io_util.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const mtp_lookup = @import("../spec/mtp_lookup.zig");
const round_cost = @import("../spec/round_cost.zig");
const ev_fix = @import("../spec/mtp/ev_fix.zig");
const ev_costs = @import("../spec/mtp/ev_costs.zig");
const ev_plan = @import("../spec/mtp/ev_plan.zig");
const adaptive = @import("../spec/mtp/adaptive.zig");
const regime = @import("../spec/mtp/regime.zig");
const depth = @import("../spec/mtp/depth.zig");
const planner = @import("../spec/mtp/planner.zig");
const observe = @import("../spec/mtp/observe.zig");
const chain = @import("../spec/mtp/chain.zig");
const serial = @import("../spec/mtp/serial.zig");
pub const MtpHeadRef = @import("../spec/mtp_head_ref.zig").MtpHeadRef;
const Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;

/// Why a request left speculative decoding for the rest of its generation (#1005's `prefill_chunk.SpecDisableReason`).
pub const SpecDisableReason = enum { none, acceptance, max_ctx, think_bound, tool_choice, adaptive, company };

/// What the policy keeps on the model (the engine's `Transformer`): the measured round-cost table, the chunk-A
/// boundary's price, the exact-depth costs, and the trunk whose head drafts.
pub const Model = struct {
    trunk: *Trunk,
    round_cost: round_cost.Table = .{},
    mtp_depth_free: u32 = 0,
    mtp_depth_ms: [8]f32 = @splat(0),
    mtp_depth_acc: f32 = 0.8,
    mtp_depth_rounds: u32 = 0,
    mtp_ev_fix: ev_fix.Model = .{},

    pub fn qwen4Trunk(self: *const Model) ?*Trunk {
        return self.trunk;
    }
};

/// The engine's forward context, as far as the policy reads it: a turn on an M-RoPE table never resumes MTP.
pub const Ctx = struct {
    mrope_pos: ?*const anyopaque = null,
};

pub const Generator = struct {
    xfm: *Model,
    mtp: ?MtpHeadRef = null,
    ctx: Ctx = .{},
    timer: io_util.Stopwatch,
    prompt_ids_owned: []u32 = &.{},
    generated_ids: std.ArrayList(u32) = .empty,
    completion_tokens: u32 = 0,
    max_tokens: u32 = 0,
    model_has_mtp: bool = false,

    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    mtp_depth_free: u32 = mtp_mod.DEFAULT_DEPTH,
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    mtp_attempted: u64 = 0,
    mtp_accepted_tokens: u64 = 0,
    mtp_window_drafted: [depth.MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_accepted: [depth.MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    mtp_promote_cooldown: u32 = 0,
    mtp_lookup_rounds: u64 = 0,
    mtp_lookup_accepted: u64 = 0,
    mtp_lookup_idx: ?mtp_lookup.Index = null,
    mtp_lookup_gen: usize = 0,
    mtp_lookup_ema: f32 = @floatFromInt(mtp_lookup.MAX_DRAFT),
    mtp_round_ema: f32 = 0,
    mtp_after_lookup: bool = false,
    mtp_lookup_streak: bool = false,
    mtp_ext_dry_streak: u32 = 0,
    mtp_ext_cooldown: u32 = 0,
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = @splat(depth.MTP_EV_PRIOR),
    mtp_ev_rounds: u32 = 0,
    mtp_ev_m_lo_prev: u32 = 1,
    mtp_ev_costs: ev_costs.MtpEvCosts = ev_costs.MTP_EV_DEFAULT_COSTS,
    mtp_ev_sync_ms: f32 = 0,
    mtp_ev_round_ms: f32 = 0,
    mtp_ev_fix: ev_fix.Request = .{},
    mtp_regime: regime.MtpRegime = .{},
    mtp_regime_verdict: ?bool = null,
    mtp_width_trial: planner.MtpWidthTrial = .{},
    mtp_m_lo_streak: u32 = 0,
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    spec_round_prev_two_chunk: bool = false,
    spec_round_prev_two_chunk2: bool = false,
    mtp_regime_clock: ?io_util.Stopwatch = null,
    spec_cost_solo: bool = true,
    mtp_planner_owned: bool = false,
    mtp_planner_width: ?u8 = null,
    mtp_batch_head: bool = false,
    spec_disabled_runtime: bool = false,
    spec_disable_reason: SpecDisableReason = .none,
    mtp_adaptive: adaptive.MtpAdaptive = .{},
    mtp_price: adaptive.MtpPriceWindow = .{},
    mtp_serial_left: u32 = 0,
    mtp_low_accept: u32 = 0,
    mtp_tok_ema: f32 = 0,
    mtp_serial_exit: adaptive.MtpSerialExit = .none,
    mtp_serial_clock: ?io_util.Stopwatch = null,
    mtp_serial_warm: u32 = 0,
    mtp_serial_accept: bool = false,
    mtp_cycle_watch: ?io_util.Stopwatch = null,
    mtp_kept: u32 = 0,
    mtp_tried: u32 = 0,

    pub const mtpChainLogConf = ev_plan.mtpChainLogConf;

    // The policy's methods, the copies' own.
    pub const mtpRoundPlanInner = planner.mtpRoundPlanInner;
    pub const updateMtpEvRound = planner.updateMtpEvRound;
    pub const updateMtpDepth = depth.updateMtpDepth;
    pub const mtpKvLen = ev_plan.mtpKvLen;
    pub const mtpEvFixOn = ev_fix.mtpEvFixOn;
    pub const mtpEvFixPlan = ev_fix.mtpEvFixPlan;
    pub const mtpEvFixObserve = ev_fix.mtpEvFixObserve;
    pub const mtpRegimeWallMs = regime.mtpRegimeWallMs;
    pub const mtpAdaptiveArchEligible = adaptive.mtpAdaptiveArchEligible;
    pub const mtpAdaptiveBucket = adaptive.mtpAdaptiveBucket;
    pub const mtpAdaptiveHeadMayResume = adaptive.mtpAdaptiveHeadMayResume;
    pub const mtpAdaptiveSerialStep = adaptive.mtpAdaptiveSerialStep;
    pub const mtpAdaptiveSyncWindow = adaptive.mtpAdaptiveSyncWindow;
    pub const mtpAdaptiveModelOk = serial.mtpAdaptiveModelOk;
    pub const observeSerialTick = serial.observeSerialTick;
    pub const serialCellWanted = serial.serialCellWanted;
    pub const mtpSerialWallMs = serial.mtpSerialWallMs;
    pub const mtpExactAcceptance = observe.mtpExactAcceptance;
    pub const mtpExactDepth = observe.mtpExactDepth;
    pub const mtpExactObserve = observe.mtpExactObserve;
    pub const mtpRoundEndObserve = observe.mtpRoundEndObserve;
    pub const specObserveRound = observe.specObserveRound;
    pub const mtpLookupIndex = chain.mtpLookupIndex;
    pub const mtpLookupDrafts = chain.mtpLookupDrafts;
    pub const mtpLookupCosts = chain.mtpLookupCosts;
    pub const mtpLookupObserve = chain.mtpLookupObserve;
    pub const mtpRoundAcceptObserve = chain.mtpRoundAcceptObserve;
    pub const mtpRoundUntimed = chain.mtpRoundUntimed;

    pub fn deinit(self: *Generator, a: std.mem.Allocator) void {
        if (self.mtp_lookup_idx) |*ix| ix.deinit();
        self.mtp_lookup_idx = null;
        self.generated_ids.deinit(a);
    }
};
