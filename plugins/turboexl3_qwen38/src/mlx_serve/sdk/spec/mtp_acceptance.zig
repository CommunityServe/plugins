//! copied_from: src/sdk/spec/mtp_acceptance.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const testing = std.testing;

/// Installed once before generation. Exact remains the default route.
pub const Mode = union(enum) {
    exact,
    typical: struct { delta: f32, eps: f32 = 1.0 },
    tokenv3: f32,
};

pub const DEFAULT_TYPICAL_DELTA: f32 = 0.2;
pub const DEFAULT_TOKENV3_ALPHA: f32 = 0.95;

/// These three entrypoints are bound on the Generator at construction, so
/// the measured per-token loop has no mode switch or mode validation.
pub const PrefixFn = *const fn ([]const f32, ?[]const f32, ?[]const bool, *std.Random.DefaultPrng) u32;
