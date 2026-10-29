//! copied_from: src/engines/mlx/spec/mtp_group_planner.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
const std = @import("std");
const env = @import("../../../env.zig");
const testing = std.testing;

pub const MAX_ROWS = 8;
pub const MAX_DEPTH = 16;
pub const MIN_SAMPLES = 3;

pub const History = struct {
    trials: [MAX_DEPTH]u16 = @splat(0),
    accepted: [MAX_DEPTH]u16 = @splat(0),
    rounds: u32 = 0,

};

pub const Row = struct {
    history: *const History,
    cap: u8 = 4,
    latency_ms: f32 = 100,
};

pub const Price = struct {
    ms: f32,
    variance: f32 = 0,
    samples: u32 = MIN_SAMPLES,
    max_gap_ms: f32 = 0,
    /// Throughput estimates the mean; variance still bounds an individual round.
    mean_upper_ms: ?f32 = null,
    mean_lower_ms: ?f32 = null,
};

pub const Decision = struct {
    widths: [MAX_ROWS]u8 = @splat(0),
    n: usize = 0,
    rate: f32 = 0,
    plain_rate: f32 = 0,
    upper_ms: f32 = 0,
};

fn validPrice(price: Price) bool {
    if (price.mean_upper_ms) |bound| if (!std.math.isFinite(bound) or bound < price.ms) return false;
    if (price.mean_lower_ms) |bound| if (!std.math.isFinite(bound) or bound > price.ms) return false;
    return price.samples >= MIN_SAMPLES and std.math.isFinite(price.ms) and price.ms > 0 and
        std.math.isFinite(price.variance) and price.variance >= 0 and
        std.math.isFinite(price.max_gap_ms) and price.max_gap_ms >= 0;
}

fn consider(rows: []const Row, provider: anytype, widths: []const u8, baseline: f32, decision: *Decision) void {
    var tokens: f32 = 0;
    var speculative = false;
    for (rows, widths) |row, width| {
        if (width > row.cap or width > MAX_DEPTH) return;
        if (width > 0) {
            speculative = true;
            if (row.history.trials[width - 1] < MIN_SAMPLES) return;
        }
        tokens += row.history.expected(width);
    }
    if (!speculative) return;
    const price = provider.price(widths) orelse return;
    if (!validPrice(price)) return;
    const upper = price.ms + 2 * @sqrt(price.variance);
    const delay = @max(upper, price.max_gap_ms);
    for (rows) |row| if (delay > row.latency_ms) return;
    const rate = 1000 * tokens / (price.mean_upper_ms orelse upper);
    if (rate <= baseline * 1.05 or rate <= decision.rate) return;
    @memcpy(decision.widths[0..rows.len], widths);
    decision.rate = rate;
    decision.upper_ms = upper;
}

pub fn choose(rows: []const Row, provider: anytype) Decision {
    if (rows.len == 0 or rows.len > MAX_ROWS) return .{};
    var decision = Decision{ .n = rows.len };
    for (rows) |row| if (!std.math.isFinite(row.latency_ms) or row.latency_ms <= 0) return decision;
    const plain = provider.price(decision.widths[0..rows.len]) orelse return decision;
    if (!validPrice(plain)) return decision;
    const lower = plain.mean_lower_ms orelse (plain.ms - 2 * @sqrt(plain.variance));
    if (lower <= 0) return decision;
    decision.plain_rate = 1000 * @as(f32, @floatFromInt(rows.len)) / plain.ms;
    decision.rate = decision.plain_rate;
    decision.upper_ms = plain.ms + 2 * @sqrt(plain.variance);
    const baseline = 1000 * @as(f32, @floatFromInt(rows.len)) / lower;
    var max_cap: u8 = 0;
    for (rows) |row| max_cap = @max(max_cap, row.cap);
    for ([_]u8{ 1, 2, 4, 8, 16 }) |width| {
        var base: [MAX_ROWS]u8 = @splat(0);
        for (rows, 0..) |row, i| base[i] = @min(width, row.cap);
        consider(rows, provider, base[0..rows.len], baseline, &decision);
        for (rows, 0..) |row, i| {
            for ([_]u8{ base[i] -| 1, @min(base[i] + 1, row.cap), 0 }) |near| {
                var candidate = base;
                candidate[i] = near;
                consider(rows, provider, candidate[0..rows.len], baseline, &decision);
            }
        }
        for ([_]bool{ false, true }) |drop| {
            var scores: [MAX_ROWS]f32 = @splat(0);
            var order: [MAX_ROWS]usize = undefined;
            for (rows, 0..) |row, i| {
                order[i] = i;
                scores[i] = if (drop) 1 - row.history.expected(base[i]) else row.history.expected(@min(width * 2, @min(row.cap, MAX_DEPTH))) - row.history.expected(base[i]);
            }
            std.sort.insertion(usize, order[0..rows.len], scores, struct {
                fn before(values: [MAX_ROWS]f32, a: usize, b: usize) bool {
                    return values[a] > values[b];
                }
            }.before);
            var candidate = base;
            for (order[0 .. rows.len - 1]) |i| {
                candidate[i] = if (drop) 0 else @min(width * 2, @min(rows[i].cap, MAX_DEPTH));
                consider(rows, provider, candidate[0..rows.len], baseline, &decision);
            }
        }
        if (width >= max_cap) break;
    }
    return decision;
}

const Fixture = struct {
    mode: enum { ordinary, noisy, marginal, ragged, zero, unknown } = .ordinary,
    calls: usize = 0,

    fn price(self: *Fixture, widths: []const u8) ?Price {
        self.calls += 1;
        var plain = true;
        for (widths) |width| plain = plain and width == 0;
        if (plain) return .{ .ms = 20, .samples = 100 };
        if (self.mode == .unknown) return .{ .ms = 10, .samples = 2 };
        if (widths.len == 4) {
            if (std.mem.eql(u8, widths, &.{ 0, 0, 2, 2 }) and self.mode == .zero) return .{ .ms = 31 };
            if (std.mem.eql(u8, widths, &.{ 1, 1, 2, 2 })) return .{ .ms = 40 };
            if (std.mem.eql(u8, widths, &.{ 1, 1, 1, 1 })) return .{ .ms = 45 };
            if (std.mem.eql(u8, widths, &.{ 2, 2, 2, 2 })) return .{ .ms = 60 };
            return null;
        }
        if (std.mem.eql(u8, widths, &.{ 1, 1 })) return switch (self.mode) {
            .noisy => .{ .ms = 35, .variance = 25 },
            .marginal => .{ .ms = 38 },
            else => .{ .ms = 21 },
        };
        if (std.mem.eql(u8, widths, &.{ 2, 2 })) return .{ .ms = 70 };
        if (std.mem.eql(u8, widths, &.{ 4, 4 })) return .{ .ms = 100 };
        return null;
    }
};

pub const Action = enum { plain, prime, speculate };

pub const Recovery = struct {
    last_spec_tokens: u32 = 0,
    rounds: u8 = 0,
    burst: u8 = 0,

    pub fn width(self: Recovery, completed: u32, remaining: u32, cap: u8, probes: u8) ?u8 {
        if (probes < 12 or remaining < 96 or cap == 0 or self.rounds >= 8) return null;
        if (self.burst == 0 and completed -| self.last_spec_tokens < 48) return null;
        return @min(2, cap);
    }

};

pub const DEFAULT_ENABLED = true;
pub var enabled_override: ?bool = null;

pub fn enabled() bool {
    return enabled_override orelse DEFAULT_ENABLED;
}

pub const LATENCY_MS: f32 = 100;

pub const Execution = struct {
    n: usize = 0,
    prime: bool = false,
    plain: [MAX_ROWS]usize = undefined,
    speculative: [MAX_ROWS]usize = undefined,
    plain_n: usize = 0,
    speculative_n: usize = 0,
    capture: [MAX_ROWS]bool = @splat(false),
};

pub const Entry = enum { wait_predraft, drain, choose };
