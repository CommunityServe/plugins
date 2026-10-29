//! copied_from: src/engines/mlx/diag/profile.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/diag/profile.
//! Owns debugTraceHead, DecodeProf, layer_cap_env, layerCap, decode_prof, decode_prof_enabled, and 5 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");

// ── Decode sub-block profiler (diag decode-profile) ──
// Serializes the S=1 decode forward with per-sub-block evals to attribute the
// per-token GPU cost across embed / attn / mlp / lm_head. DIAGNOSTIC ONLY: the
// evals defeat the async pipeline (absolute tok/s drops while active), so the
// signal is the RELATIVE split plus the serialized total vs the pipelined
// per-token time — a serialized total near the pipelined per-token means we are
// compute-bound; far below it means the gap is dispatch/pipeline overhead.
const DecodeProf = struct {
    calls: u64 = 0,
    embed_ns: u64 = 0,
    attn_ns: u64 = 0,
    gdn_ns: u64 = 0,
    mlp_ns: u64 = 0,
    lmhead_ns: u64 = 0,
    // MoE internals (summed across all MoE layers per token)
    moe_router_ns: u64 = 0,
    moe_experts_ns: u64 = 0,
    moe_shared_ns: u64 = 0,
};

/// DIAGNOSTIC (diag layer-cap): clamp the layer loop so a ms-vs-layers
/// sweep separates per-layer slope from fixed cost. Unset — the only shipped
/// state — returns the model's real depth.
var layer_cap_env: ?usize = null;

pub fn layerCap(n: usize) usize {
    if (layer_cap_env == null) {
        layer_cap_env = blk: {
            const raw = env.get(.LAYER_CAP) orelse break :blk 0;
            break :blk std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        };
    }
    const cap = layer_cap_env.?;
    return if (cap == 0) n else @min(cap, n);
}

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ProfileState = struct {
    decode_prof: DecodeProf = .{},
    decode_prof_enabled: ?bool = null,
};
pub const profileState_shared = @import("../../../sdk/util/shared.zig").Shared(ProfileState, "decode_profile", .{});
pub inline fn profileState() *ProfileState {
    return profileState_shared.get();
}

pub const ProfClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    pub fn init() ProfClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    pub fn lap(self: *ProfClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};

pub fn decodeProfileEnabled() bool {
    if (profileState().decode_prof_enabled) |v| return v;
    const v = env.get(.DECODE_PROFILE) != null;
    profileState().decode_prof_enabled = v;
    return v;
}

/// Extra per-MoE-layer elementwise dispatches injected by the sizing probe
/// (diag `dispatch-probe`). 0 = off, which is every non-diagnostic run.
var dispatch_probe_cached: ?usize = null;
pub fn dispatchProbeCount() usize {
    if (dispatch_probe_cached) |v| return v;
    const raw = env.get(.DISPATCH_PROBE);
    const n: usize = if (raw) |r| std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0 else 0;
    dispatch_probe_cached = n;
    return n;
}
