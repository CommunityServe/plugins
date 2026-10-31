//! Directional ablation on the plugin side: the request's direction from `abi.Forward.ablation` armed on the model for
//! one call. The calibration tap's files are `tap.zig`.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");

pub const Refusal = error{ AblationStrength, AblationSlots };

/// Whether the kit this plugin is built against has `Forward.ablation`; without it the plugin serves none.
pub const supported = @hasField(abi.Forward, "ablation");

/// One slot's ablation as the forward hands it.
pub const Spec = struct { directions: abi.Array, strength: f32 };

/// The forward's ablation, or null: a host older than `Forward.ablation` passes a shorter struct, and a strength of 0
/// is no ablation.
pub fn of(f: *const abi.Forward) Refusal!?Spec {
    if (comptime !supported) return null else {
        if (f.size < @offsetOf(abi.Forward, "ablation") + @sizeOf(?*const abi.Ablation)) return null;
        const a = f.ablation orelse return null;
        if (a.strengths.len != 1 or a.strengths.len != f.slots.len) return error.AblationSlots;
        const s = a.strengths.slice()[0];
        if (!(s >= 0 and s <= 1)) return error.AblationStrength;
        return if (s == 0) null else .{ .directions = a.directions, .strength = s };
    }
}

/// Arms `a` on the model for one call: `r` `[dim]`, `col` `[dim, 1]` and `s`, owned here until `disarm`.
pub fn arm(mdl: *dsv.Model, a: Spec, dim: usize, s: mlx.mlx_stream) (error{AblationShape} || anyerror)!void {
    const d: mlx.mlx_array = .{ .ctx = a.directions.ctx };
    if (d.ctx == null) return error.AblationShape;
    const sh = mlx.getShape(d);
    if (sh.len != 2 or sh[0] != 1 or sh[1] != dim or mlx.mlx_array_dtype(d) != .float32) return error.AblationShape;
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, d, &[_]c_int{@intCast(dim)}, 1, s));
    var col = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, d, &[_]c_int{ @intCast(dim), 1 }, 2, s));
    mdl.ablation = .{ .r = r, .col = col, .s = mlx.mlx_array_new_float(a.strength) };
}

/// Whether a forward of `mode` feeds the calibration tap: prompt passes only, never a one-token decode step.
pub fn tapsMode(mode: abi.Mode) bool {
    return mode == .prefill;
}

/// Drops the call's ablation; the graphs built under it keep their own references.
pub fn disarm(mdl: *dsv.Model) void {
    const ab = mdl.ablation orelse return;
    for ([_]mlx.mlx_array{ ab.r, ab.col, ab.s }) |x| _ = mlx.mlx_array_free(x);
    mdl.ablation = null;
}

test "sushi_ds41 ablation: the tap reads prompt passes only" {
    try std.testing.expect(tapsMode(.prefill));
    for ([_]abi.Mode{ .decode, .verify, .batched_rows }) |mode| try std.testing.expect(!tapsMode(mode));
}

test "sushi_ds41 ablation: a host's forward without the field, a null one, or strength 0 is no ablation; bad tables refuse" {
    if (comptime supported) try ofCases() else return error.SkipZigTest;
}

fn ofCases() !void {
    const dir: abi.Array = .{};
    const one = [_]u32{0};
    var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .slots = .of(&one) };
    try std.testing.expect((try of(&f)) == null);
    var zero = [_]f32{0};
    var ab: abi.Ablation = .{ .directions = dir, .strengths = .of(&zero) };
    f.ablation = &ab;
    try std.testing.expect((try of(&f)) == null);
    var half = [_]f32{0.5};
    ab.strengths = .of(&half);
    try std.testing.expectEqual(@as(f32, 0.5), (try of(&f)).?.strength);
    var big = [_]f32{1.5};
    ab.strengths = .of(&big);
    try std.testing.expectError(error.AblationStrength, of(&f));
    var two = [_]f32{ 1, 1 };
    ab.strengths = .of(&two);
    try std.testing.expectError(error.AblationSlots, of(&f));
    f.size = @offsetOf(abi.Forward, "ablation");
    try std.testing.expect((try of(&f)) == null);
}
