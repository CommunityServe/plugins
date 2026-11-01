//! `MLX_SERVE_DSV41_LOOKAHEAD` (read at load, before the stream opens), MLX-free: the decode's expert lookahead, which
//! reads the next layer's predicted experts ahead of its route (the copied exl3_source.zig and experts.zig read it, the
//! patch patches/turboexl3-zz3-lookahead.patch). Unset: the streamer's own constants, records read ahead a layer call 2
//! and the idle rule 0 (a speculative thread past the first starts a chunk only while no demand read runs).
//! `<budget>[:<idle>]`: the records read ahead a layer call, 0 to 4 (0: the predictor off, nothing read ahead), and the
//! idle rule, 0 or 1 (1: also while one demand read runs). The lookahead only reads ahead: the served bytes are the
//! same at every setting.
//!
//! The stream sizes its lookahead (staging slots, speculative threads) for `initBudget`: today's 2 below it, so budgets
//! 0 to 2 keep today's memory bill and slot rows and change only how many records a call reads ahead.

const std = @import("std");

/// The streamer's own constants (`exl3_source.zig`).
pub const default_budget: u32 = 2;
pub const max_budget: u32 = 4;

pub const Setting = struct {
    budget: u32 = default_budget,
    idle_busy: u32 = 0,

    /// The budget the stream is built for: the staging rows and speculative threads.
    pub fn initBudget(s: Setting) u32 {
        return @max(s.budget, default_budget);
    }

    /// Whether the decode's calls read the next layer's router for the lookahead at all.
    pub fn predicts(s: Setting) bool {
        return s.budget > 0;
    }

    pub fn isDefault(s: Setting) bool {
        return s.budget == default_budget and s.idle_busy == 0;
    }
};

/// Set at load.
pub var setting: Setting = .{};

/// The setting's value, or `error.LookaheadSetting` for one it does not name (the load refuses it by name).
pub fn parse(value: ?[]const u8) error{LookaheadSetting}!Setting {
    const v = value orelse return .{};
    if (v.len == 0) return .{};
    var s: Setting = .{};
    const colon = std.mem.indexOfScalar(u8, v, ':');
    const b = v[0 .. colon orelse v.len];
    if (b.len != 1 or b[0] < '0' or b[0] > '0' + max_budget) return error.LookaheadSetting;
    s.budget = b[0] - '0';
    if (colon) |c| {
        const i = v[c + 1 ..];
        if (i.len != 1 or (i[0] != '0' and i[0] != '1')) return error.LookaheadSetting;
        s.idle_busy = i[0] - '0';
    }
    return s;
}

// ── tests ──

const testing = std.testing;

test "lookahead: unset or empty is the streamer's own constants" {
    for ([_]?[]const u8{ null, "" }) |v| {
        const s = try parse(v);
        try testing.expect(s.isDefault() and s.predicts());
        try testing.expectEqual(@as(u32, 2), s.initBudget());
    }
}

test "lookahead: a budget 0 to 4 and an idle rule 0 or 1; anything else is refused by name" {
    const off = try parse("0");
    try testing.expect(!off.predicts());
    try testing.expectEqual(@as(u32, 2), off.initBudget());
    try testing.expectEqual(@as(u32, 1), (try parse("1")).budget);
    try testing.expectEqual(@as(u32, 2), (try parse("1")).initBudget());
    try testing.expect((try parse("2")).isDefault());
    const four = try parse("4");
    try testing.expectEqual(@as(u32, 4), four.budget);
    try testing.expectEqual(@as(u32, 4), four.initBudget());
    const idle = try parse("2:1");
    try testing.expect(idle.budget == 2 and idle.idle_busy == 1 and !idle.isDefault());
    try testing.expect((try parse("2:0")).isDefault());
    for ([_][]const u8{ "5", "off", "-1", "22", "2:2", "2:", ":1", " 2", "2 " }) |v| try testing.expectError(error.LookaheadSetting, parse(v));
}
