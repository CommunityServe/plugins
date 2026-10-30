//! This plugin's own, in place of mlx-serve's `src/engines/mlx/spec/mtp_head_ref.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the one head this plugin drafts with, qwen4_exp's (`.qwen4`), and the facts #1005's round
//! policy asks of it, answered as #1005's `MtpHeadRef` answers them for that arm.

const std = @import("std");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;

/// Most drafts one MTP round verifies: the head chain or a prompt-lookup run.
pub const MAX_ROUND_DRAFTS: usize = @max(mtp_mod.MAX_DEPTH, mtp_lookup.MAX_DRAFT_STRONG);

pub const MtpHeadRef = union(enum) {
    qwen4: *Trunk,

    /// Per-request state swaps onto the module (`Qwen4MtpState`): nothing is shared.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        _ = self;
        return false;
    }

    pub fn maxDepth(self: MtpHeadRef) u32 {
        _ = self;
        return std.math.maxInt(u32);
    }

    /// The qwen4_exp head over an EXL3 pack (`Formats.profile`): it drafts its argmax for every request and its
    /// two-chunk plans pay their chunk-A boundary (`ev_fix.zig`).
    pub fn exl3Profile(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen4 => |t| t.formats.profile,
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never published one.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .qwen4 => |t| if (t.qwen4_mtp) |*m| {
                m.ev_seed_accept = accept;
                m.ev_seed_m_lo = m_lo;
            },
        }
    }
};
