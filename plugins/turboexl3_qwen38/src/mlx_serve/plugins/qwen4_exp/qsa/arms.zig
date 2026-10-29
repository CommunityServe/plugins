//! copied_from: src/plugins/qwen4_exp/qsa/arms.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Which QSA attention arm served each full-attention call (`QsaArm`, `QsaArmCounts`, decline reasons): the
//! per-request `[qsa-arms]` meter.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const Trunk = @import("../trunk.zig").Trunk;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const OneShotBits = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").widthBucket;

// ── QSA arm meters ──
//
// A single process-global `engaged` bool is not a meter: it fires on the
// first request of a session and then says nothing about the width or the
// context length that is actually slow, and a DECLINE says nothing at all —
// so a gather that silently hands every 62k decode step back to the dense
// mask (a full kv8 dequant per layer per step) reads exactly like a healthy
// run. These three pieces make the arm that ran observable: one-shot bits
// per (arm, width bucket) for engagement, one-shot bits per reason for
// declines, and a per-request tally.

/// Which arm served one qwen4_exp full-attention call.
pub const QsaArm = enum(u2) { mask = 0, decode_gather = 1, verify_gather = 2, prefill_gather = 3 };

/// Why a gather arm handed the call back to the dense mask.
pub const QsaDeclineReason = enum(u5) {
    disabled,
    width,
    geometry,
    dtype,
    kv_floor,
    no_win,
    quant_layout,
    gather_shape,

    pub fn text(self: QsaDeclineReason) []const u8 {
        return switch (self) {
            .disabled => "kill switch",
            .width => "query width not this arm's",
            .geometry => "geometry",
            .dtype => "dtype",
            .kv_floor => "kv below the arm's floor",
            .no_win => "no sparsity win (gathered rows >= kv)",
            .quant_layout => "quant triple layout",
            .gather_shape => "gathered slab shape",
        };
    }
};

/// Decline reporter threaded through one gather call: `no(reason)` logs at
/// most once per (arm, reason) for the process and hands the caller the
/// decline. `s`/`kv` are filled in as soon as the guards learn them (-1 =
/// not yet known at that guard).
pub const QsaDecline = struct {
    bits: *OneShotBits,
    arm: []const u8,
    s: c_int = -1,
    kv: c_int = -1,

    pub fn no(self: *QsaDecline, r: QsaDeclineReason) ?mlx.mlx_array {
        if (self.bits.take(@backingInt(r))) {
            log.info(
                "[{s}] declined: {s} (S={d} kv={d}) — the dense [S, kv] mask arm serves this call\n",
                .{ self.arm, r.text(), self.s, self.kv },
            );
        }
        return null;
    }
};

/// Per-request tally of which arm served each full-attention call. One
/// increment per attention layer; read and reset where the request's
/// `[qsa-arms]` line is emitted. It lives on the REQUEST's `ForwardCtx`
/// (the Generator's own), never on the shared Trunk: qwen4 text slots
/// batch-decode on one Trunk and a cold prefill interleaves with other
/// slots' decode ticks, so a Trunk-level tally prints the union of
/// every concurrent slot and leaves the next finisher an empty one.
pub const QsaArmCounts = struct {
    mask: u32 = 0,
    decode: u32 = 0,
    verify: u32 = 0,
    prefill: u32 = 0,

    pub fn note(self: *QsaArmCounts, arm: QsaArm) void {
        switch (arm) {
            .mask => self.mask +|= 1,
            .decode_gather => self.decode +|= 1,
            .verify_gather => self.verify +|= 1,
            .prefill_gather => self.prefill +|= 1,
        }
    }
    /// A gather served the call: which gather is the query WIDTH's business.
    pub fn noteGather(self: *QsaArmCounts, seq_len: c_int) void {
        self.note(switch (qsaWidthBucket(seq_len)) {
            0 => .decode_gather,
            1 => .verify_gather,
            else => .prefill_gather,
        });
    }
    pub fn add(self: *QsaArmCounts, o: QsaArmCounts) void {
        self.mask +|= o.mask;
        self.decode +|= o.decode;
        self.verify +|= o.verify;
        self.prefill +|= o.prefill;
    }
    pub fn total(self: QsaArmCounts) u32 {
        return self.mask +| self.decode +| self.verify +| self.prefill;
    }
    /// Read and clear this request's tally. Null when nothing was counted
    /// (not a QSA arch, or below the sparse budget) so a caller never prints
    /// a meaningless `qsa=mask 0/0`.
    pub fn take(self: *QsaArmCounts) ?QsaArmCounts {
        const c = self.*;
        self.reset();
        if (c.total() == 0) return null;
        return c;
    }
    /// The arm that served the most calls; null when nothing was counted.
    /// Ties break toward the CHEAPEST arm that could have run so a tie never
    /// reads as "the mask arm is fine".
    pub fn majority(self: QsaArmCounts) ?QsaArm {
        if (self.total() == 0) return null;
        var best: QsaArm = .mask;
        var best_n: u32 = self.mask;
        if (self.decode >= best_n) {
            best = .decode_gather;
            best_n = self.decode;
        }
        if (self.verify >= best_n) {
            best = .verify_gather;
            best_n = self.verify;
        }
        if (self.prefill >= best_n) {
            best = .prefill_gather;
            best_n = self.prefill;
        }
        return best;
    }
    pub fn reset(self: *QsaArmCounts) void {
        self.* = .{};
    }
};

/// One batched call serves every slot in the group with the SAME arm (the
/// stacked-mask SDPA), so it counts once on each participating slot's own
/// tally — a single shared counter would report the group as one request.
pub fn noteQsaArmForSlots(slots: []const *ForwardCtx, arm: QsaArm) void {
    for (slots) |slot_ctx| slot_ctx.qsa_arms.note(arm);
}

/// Engagement meters, one-shot per (arm, width bucket): bit = arm*4 + bucket.
pub var qsa_engaged_bits: OneShotBits = .{};
pub var qsa_decode_decline_bits: OneShotBits = .{};
pub var qsa_dec_engaged_bits: OneShotBits = .{};
pub var qsa_verify_decline_bits: OneShotBits = .{};

pub fn qsaEngagedBit(arm: QsaArm, seq_len: c_int) u5 {
    return @as(u5, @backingInt(arm)) * 4 + @as(u5, qsaWidthBucket(seq_len));
}

// ── tests ──
