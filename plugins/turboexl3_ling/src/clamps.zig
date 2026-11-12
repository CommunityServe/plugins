//! Ling-3.1's per-layer SwiGLU clamps, MLX-free: `expert_swiglu_limit_list` (the routed experts) and
//! `share_expert_swiglu_limit_list` (the shared expert), one entry a trunk layer, 0 off, else the limit L of
//! min(silu(g), L) * clamp(u, -L, L). Read from the pack's config at parse (config.zig), held for the process's one
//! loaded pack, and read by the forward and the bank. An absent list is all 0 (Ling 3.0); a list of the wrong length,
//! a non-integer, a negative or a value above 128 (turboexl3's clamp bound) is refused by name.

const std = @import("std");

pub const max_layers = 512;

pub const Error = error{ LingClampListLength, LingClampValue };

var routed_limits: [max_layers]u8 = @splat(0);
var shared_limits: [max_layers]u8 = @splat(0);

pub fn routed(layer: u32) u32 {
    return if (layer < max_layers) routed_limits[layer] else 0;
}

pub fn shared(layer: u32) u32 {
    return if (layer < max_layers) shared_limits[layer] else 0;
}

/// Both lists from the config `doc`, for `n_layers` trunk layers.
pub fn read(doc: std.json.ObjectMap, n_layers: usize) Error!void {
    var r: [max_layers]u8 = @splat(0);
    var s: [max_layers]u8 = @splat(0);
    try list(doc.get("expert_swiglu_limit_list"), n_layers, &r);
    try list(doc.get("share_expert_swiglu_limit_list"), n_layers, &s);
    routed_limits = r;
    shared_limits = s;
}

fn list(v: ?std.json.Value, n_layers: usize, out: *[max_layers]u8) Error!void {
    const val = v orelse return;
    if (val == .null) return;
    if (val != .array or n_layers > max_layers or val.array.items.len != n_layers) return error.LingClampListLength;
    for (val.array.items, out[0..n_layers]) |e, *o| {
        const f: f64 = switch (e) {
            .integer => |i| @floatFromInt(i),
            .float => |x| x,
            else => return error.LingClampValue,
        };
        if (!std.math.isFinite(f) or f < 0 or f > 128 or f != @floor(f)) return error.LingClampValue;
        o.* = @intFromFloat(f);
    }
}

const testing = std.testing;

fn readJson(json: []const u8, n: usize) Error!void {
    const p = std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{}) catch unreachable;
    defer p.deinit();
    return read(p.value.object, n);
}

test "ling clamps: the two lists by layer, absent lists all 0, and malformed lists refused by name" {
    try readJson("{\"expert_swiglu_limit_list\":[0,0,10],\"share_expert_swiglu_limit_list\":[0,10,10]}", 3);
    try testing.expectEqual(@as(u32, 0), routed(1));
    try testing.expectEqual(@as(u32, 10), routed(2));
    try testing.expectEqual(@as(u32, 10), shared(1));
    try testing.expectEqual(@as(u32, 0), routed(600));
    try readJson("{}", 3);
    try testing.expectEqual(@as(u32, 0), routed(2));
    try testing.expectError(error.LingClampListLength, readJson("{\"expert_swiglu_limit_list\":[0,10]}", 3));
    try testing.expectError(error.LingClampValue, readJson("{\"expert_swiglu_limit_list\":[0,10.5,0]}", 3));
    try testing.expectError(error.LingClampValue, readJson("{\"expert_swiglu_limit_list\":[0,-1,0]}", 3));
    try testing.expectError(error.LingClampValue, readJson("{\"expert_swiglu_limit_list\":[0,129,0]}", 3));
    try testing.expectError(error.LingClampValue, readJson("{\"share_expert_swiglu_limit_list\":[0,\"10\",0]}", 3));
}
