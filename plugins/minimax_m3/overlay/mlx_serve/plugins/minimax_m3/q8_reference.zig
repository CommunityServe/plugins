//! copied_from: src/plugins/minimax_m3/q8_reference.zig@9df2c52b504cce5bbf22f074dab168fd8584f961
const std = @import("std");
pub const group_size = 32;
pub fn vectorBytes(dim: u32) !u64 {
    if (dim == 0 or dim % group_size != 0) return error.MiniMaxQ8Shape;
    return @as(u64, dim) + @as(u64, dim / group_size) * 4;
}
pub fn cacheBytes(layers: u32, kv_heads: u32, dim: u32, index_dim: u32, tokens: u64) !u64 {
    return @as(u64, layers) * tokens * (2 * @as(u64, kv_heads) * try vectorBytes(dim) + try vectorBytes(index_dim));
}
pub fn quantize(values: []const f32, codes: []i8, scales: []f32) !void {
    if (values.len == 0 or values.len % group_size != 0 or codes.len != values.len or scales.len != values.len / group_size) return error.MiniMaxQ8Shape;
    for (scales, 0..) |*scale, group| {
        var max: f32 = 0;
        for (values[group * group_size ..][0..group_size]) |value| {
            if (!std.math.isFinite(value)) return error.MiniMaxQ8NonFinite;
            max = @max(max, @abs(value));
        }
        scale.* = if (max > 0) @max(max / 127, 1e-30) else 1;
        for (values[group * group_size ..][0..group_size], codes[group * group_size ..][0..group_size]) |value, *code| {
            const scaled = std.math.clamp(value / scale.*, -127, 127);
            const floor = @floor(scaled);
            const low: i32 = @intFromFloat(floor);
            const fraction = scaled - floor;
            const rounded = if (fraction < 0.5 or (fraction == 0.5 and @mod(low, 2) == 0)) low else low + 1;
            code.* = @intCast(rounded);
        }
    }
}
pub fn dequantize(codes: []const i8, scales: []const f32, out: []f32) !void {
    if (codes.len == 0 or codes.len % group_size != 0 or out.len != codes.len or scales.len != codes.len / group_size) return error.MiniMaxQ8Shape;
    for (codes, out, 0..) |code, *value, i| value.* = @as(f32, @floatFromInt(code)) * scales[i / group_size];
}
