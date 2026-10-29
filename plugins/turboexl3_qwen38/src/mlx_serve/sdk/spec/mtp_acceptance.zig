//! copied_from: src/sdk/spec/mtp_acceptance.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const testing = std.testing;

/// Installed once before generation. Exact remains the default route.
pub const Mode = union(enum) {
    exact,
    typical: struct { delta: f32, eps: f32 = 1.0 },
    tokenv3: f32,
};

pub fn parse(typical_raw: ?[]const u8, tokenv3_raw: ?[]const u8) !Mode {
    if (typical_raw != null and tokenv3_raw != null) return error.ConflictingAcceptanceModes;
    if (typical_raw) |raw| {
        const delta = std.fmt.parseFloat(f32, raw) catch return error.InvalidAcceptanceValue;
        if (!std.math.isFinite(delta) or delta <= 0) return error.InvalidAcceptanceValue;
        return .{ .typical = .{ .delta = delta } };
    }
    if (tokenv3_raw) |raw| {
        const alpha = std.fmt.parseFloat(f32, raw) catch return error.InvalidAcceptanceValue;
        if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return error.InvalidAcceptanceValue;
        return .{ .tokenv3 = alpha };
    }
    return .exact;
}

pub const DEFAULT_TYPICAL_DELTA: f32 = 0.2;
pub const DEFAULT_TOKENV3_ALPHA: f32 = 0.95;

/// The per-model settings vocabulary: a mode by name at its default threshold.
pub fn fromName(s: []const u8) ?Mode {
    if (std.mem.eql(u8, s, "exact")) return .exact;
    if (std.mem.eql(u8, s, "typical")) return .{ .typical = .{ .delta = DEFAULT_TYPICAL_DELTA } };
    if (std.mem.eql(u8, s, "tokenv3")) return .{ .tokenv3 = DEFAULT_TOKENV3_ALPHA };
    return null;
}

pub fn name(mode: Mode) []const u8 {
    return switch (mode) {
        .exact => "exact",
        .typical => "typical",
        .tokenv3 => "tokenv3",
    };
}

/// CPU oracle for the batched GPU entropy graph. `p` is sampler-filtered and
/// normalized; zero-mass entries make no contribution to Shannon entropy.
pub fn typicalThreshold(p: []const f64, delta: f64, eps: f64) f64 {
    var entropy: f64 = 0;
    for (p) |mass| if (mass > 0) {
        entropy -= mass * @log(mass);
    };
    return @min(eps, delta * @exp(-entropy));
}

pub fn typicalAccept(p_draft: f32, threshold: f32) bool {
    return p_draft > threshold;
}

pub fn tokenV3Deferred(p_draft: f64, peak_p: f64, alpha: f64) bool {
    return p_draft < peak_p * (1.0 - alpha);
}

/// Equation 11 target for the TokenV3 verifier. Used as a CPU oracle for the
/// GPU graph, which computes the same array without a host synchronization.
pub fn tokenV3Target(p: []const f64, q: []const f64, alpha: f64, pi: []f64) !void {
    if (p.len == 0 or q.len != p.len or pi.len != p.len) return error.InvalidDistributionShape;
    if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return error.InvalidAcceptanceValue;
    var peak: f64 = 0;
    for (p) |mass| peak = @max(peak, mass);
    var eta: f64 = 0;
    for (p, q) |p_mass, q_mass| {
        if (tokenV3Deferred(p_mass, peak, alpha)) eta += q_mass;
    }
    for (p, q, pi) |p_mass, q_mass, *target_mass| {
        target_mass.* = (if (tokenV3Deferred(p_mass, peak, alpha)) @as(f64, 0) else q_mass) + p_mass * eta;
    }
}

pub fn specAcceptProb(p_draft: f32, q_draft: f32) f32 {
    return @min(1.0, p_draft / @max(q_draft, 1e-12));
}

/// These three entrypoints are bound on the Generator at construction, so
/// the measured per-token loop has no mode switch or mode validation.
pub const PrefixFn = *const fn ([]const f32, ?[]const f32, ?[]const bool, *std.Random.DefaultPrng) u32;

pub fn exactPrefix(p: []const f32, q: ?[]const f32, _: ?[]const bool, prng: *std.Random.DefaultPrng) u32 {
    var accepted: u32 = 0;
    for (p, 0..) |mass, k| {
        const accept_prob = if (q) |proposal| specAcceptProb(mass, proposal[k]) else @min(1.0, mass);
        if (prng.random().float(f32) >= accept_prob) break;
        accepted += 1;
    }
    return accepted;
}

pub fn typicalPrefix(p: []const f32, threshold_opt: ?[]const f32, _: ?[]const bool, _: *std.Random.DefaultPrng) u32 {
    const threshold = threshold_opt.?; // installed typical graph always supplies it
    var accepted: u32 = 0;
    for (p, threshold) |mass, floor| {
        if (!typicalAccept(mass, floor)) break;
        accepted += 1;
    }
    return accepted;
}

pub fn tokenV3Prefix(pi_draft: []const f32, q: ?[]const f32, deferred_opt: ?[]const bool, prng: *std.Random.DefaultPrng) u32 {
    const deferred = deferred_opt.?; // installed TokenV3 graph always supplies it
    var accepted: u32 = 0;
    for (pi_draft, deferred, 0..) |mass, is_deferred, k| {
        if (is_deferred) {
            const density = if (q) |proposal| proposal[k] else 1.0;
            if (prng.random().float(f32) >= specAcceptProb(mass, density)) break;
        }
        accepted += 1;
    }
    return accepted;
}
