//! copied_from: src/engines/mlx/spec/mtp/serial.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! MTP serial-mode switching: pre-draft open, head detach, serial ticks and clocks.
const std = @import("std");
const env = @import("../../../../env.zig");
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const DrafterStepResult = @import("../drafter.zig").DrafterStepResult;
const Generator = @import("../../generate/generator.zig").Generator;
const MTP_ADAPTIVE_PROBE_WARM = @import("adaptive.zig").MTP_ADAPTIVE_PROBE_WARM;
const MTP_ADAPTIVE_MIN_KV = @import("adaptive.zig").MTP_ADAPTIVE_MIN_KV;
const MTP_GROUP_ROWS_MAX = @import("round.zig").MTP_GROUP_ROWS_MAX;
const MtpPreDraft = @import("chain.zig").MtpPreDraft;
const SubLap = @import("trace.zig").SubLap;
const isEosId = @import("../../../../sdk/request/stops.zig").isEosId;
const mtpAdaptiveKvEligible = @import("adaptive.zig").mtpAdaptiveKvEligible;
const mtpChainBuildBatched = @import("chain_build.zig").mtpChainBuildBatched;
const mtpChainDispatch = @import("chain_build.zig").mtpChainDispatch;
const mtpChainDispatchBatched = @import("chain_build.zig").mtpChainDispatchBatched;
const mtpPredraftEnabled = @import("trace.zig").mtpPredraftEnabled;
const mtpTraceEnabled = @import("trace.zig").mtpTraceEnabled;


// Serial blocks inside nextMtp. Leaving MTP is cheap; coming back needs `t1 NOT in cache`
// and an `h_prev` for that position: `drainPipelineForSpec` lands the first, the capture
// tick the second. The head's committed history does not grow across a serial block
// (a content gap that costs acceptance for a while); M-RoPE turns never come back.

/// Inter-tick wall clock for the table's serial cell (twin of `mtpRegimeWallMs`). Null on the first read of a block.
pub fn mtpSerialWallMs(self: *Generator) ?f32 {
    if (self.mtp_serial_clock) |*c| {
        const ns = c.read();
        c.reset();
        return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
    }
    self.mtp_serial_clock = io_util.Stopwatch.init(self.timer.io);
    return null;
}

pub fn mtpAdaptiveModelOk(self: *const Generator) bool {
    return self.model_has_mtp;
}

/// Will anyone read a serial cell for this model? Gated on the model, not the request: a
/// per-request `enable_mtp:false` on a calibrated checkpoint is the cleanest serial sample.
pub fn serialCellWanted(self: *const Generator) bool {
    if (!self.mtpAdaptiveModelOk()) return false;
    // Not `xfm.qwen4_mtp != null`: the weights load with the trunk even under `--no-mtp`.
    if (!self.model_has_mtp) return false;
    return mtpAdaptiveKvEligible(self.mtpKvLen(), MTP_ADAPTIVE_MIN_KV);
}

/// Fold one plain serial decode token into the model's serial cell. The first
/// `MTP_ADAPTIVE_PROBE_WARM` ticks of a block are transitions; contention restarts the clock.
pub fn observeSerialTick(self: *Generator) void {
    if (!self.serialCellWanted()) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    if (!self.spec_cost_solo) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    const ms = self.mtpSerialWallMs() orelse return;
    const warm = self.mtp_serial_warm < MTP_ADAPTIVE_PROBE_WARM;
    if (warm) self.mtp_serial_warm += 1;
    _ = self.xfm.round_cost.observeSerial(self.mtpKvLen(), ms, true, warm);
}
