//! copied_from: src/engines/mlx/spec/mtp_depth_bounds.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! `--mtp-min-depth` / `--mtp-max-depth`: the range every MTP draft-depth decision stays inside.

const std = @import("std");

pub const Bounds = struct {
    min: u32 = 1,
    /// 0 = no flag: the planner's own cap (cost profile, per-silicon row) applies.
    max: u32 = 0,

    /// The one depth every round drafts when both flags name it.
    pub fn pinned(self: Bounds) ?u32 {
        return if (self.max != 0 and self.min == self.max) self.min else null;
    }
};

/// Process-wide, set once from the launch flags before a model loads.
pub var active: Bounds = .{};

pub const deprecated_flag_message = "--mtp-depth is now --mtp-max-depth; the old spelling is still accepted with the same meaning";

pub fn parseDepth(text: []const u8, max_depth: u32) error{NotADepth}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.NotADepth;
    if (n < 1 or n > max_depth) return error.NotADepth;
    return n;
}

/// The deprecated `--mtp-depth`: any integer, clamped into 1..max_depth as it always was.
pub fn parseClamped(text: []const u8, max_depth: u32) error{NotADepth}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.NotADepth;
    return std.math.clamp(n, 1, max_depth);
}

pub fn validate(b: Bounds) error{MinAboveMax}!void {
    if (b.max != 0 and b.min > b.max) return error.MinAboveMax;
}

/// A cap the planner chose on its own, lifted to the explicit floor.
pub fn liftCap(cap: u32, b: Bounds, max_depth: u32) u32 {
    return @min(if (b.max != 0) @min(b.max, max_depth) else max_depth, @max(cap, b.min));
}

/// The floor a round can honour: a head count or verify budget below `--mtp-min-depth` wins.
pub fn floorFor(b: Bounds, cap: u32) u32 {
    return @min(b.min, @max(cap, 1));
}

pub fn clampWidth(width: u32, lo: u32, hi: u32) u32 {
    return @min(@max(width, @min(lo, hi)), hi);
}

/// The width the grouped planner gives a row of cap `cap`: a pinned depth for every row, else the
/// planner's own choice held to the range. Width 0 is a plain tick, not a depth, and stays.
pub fn plannerWidth(width: u32, cap: u32, b: Bounds) u32 {
    if (b.pinned()) |d| return @min(d, cap);
    if (width == 0) return 0;
    const hi = lookupCap(cap, b);
    return clampWidth(width, floorFor(b, hi), hi);
}

/// The widest prompt-lookup draft: its own cap, held to an explicit `--mtp-max-depth`.
pub fn lookupCap(base: u32, b: Bounds) u32 {
    return if (b.max != 0) @min(base, b.max) else base;
}
