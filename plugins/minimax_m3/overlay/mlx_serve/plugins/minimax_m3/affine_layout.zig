//! copied_from: src/plugins/minimax_m3/affine_layout.zig@dc508a5da248397f6c952e75839f1de58aab5be2
const std = @import("std");
const sdk_ext = @import("mlx_stream").sdk_ext;
const mlx = @import("mlx_stream").sdk.mlx;
pub const n_components = 9;
pub const gu_components = 6;
pub const Records = sdk_ext.expert.Records(n_components, gu_components);
pub const Dtype = enum { U32, BF16 };
pub const Component = enum(u8) {
    gate_weight, gate_scales, gate_biases,
    up_weight, up_scales, up_biases,
    down_weight, down_scales, down_biases,
    pub fn name(c: Component) []const u8 {
        return names[@intFromEnum(c)];
    }
    const names = [_][]const u8{
        "gate.weight", "gate.scales", "gate.biases",
        "up.weight", "up.scales", "up.biases",
        "down.weight", "down.scales", "down.biases",
    };
};
pub const Segment = struct {
    offset: u64,
    length: u64,
    dtype: Dtype,
    shape: [3]u64 = @splat(0),
    rank: u8,
};
pub const Layer = struct {
    model_layer: u32,
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};
pub const Arrays = sdk_ext.quant.GatherQmm.Arrays(mlx.mlx_array);
pub const BankArrays = sdk_ext.quant.BankArrays(Arrays);
pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .w = x[0], .scales = x[1], .biases = x[2] },
        .up = .{ .w = x[3], .scales = x[4], .biases = x[5] },
        .down = .{ .w = x[6], .scales = x[7], .biases = x[8] },
    };
}
pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) { .U32 => .uint32, .BF16 => .bfloat16 };
}
pub fn layerSegments(bits: u32, hidden: u64, inter: u64) ?[n_components]Segment {
    if (bits * hidden % 32 != 0 or bits * inter % 32 != 0 or hidden % 64 != 0 or inter % 64 != 0) return null;
    var segs: [n_components]Segment = undefined;
    var off: u64 = 0;
    for (0..3) |p| {
        const out: u64 = if (p == 2) hidden else inter;
        const in: u64 = if (p == 2) inter else hidden;
        const parts = [3]Segment{
            .{ .offset = 0, .length = out * (in * bits / 32) * 4, .dtype = .U32, .shape = .{ out, in * bits / 32, 0 }, .rank = 2 },
            .{ .offset = 0, .length = out * (in / 64) * 2, .dtype = .BF16, .shape = .{ out, in / 64, 0 }, .rank = 2 },
            .{ .offset = 0, .length = out * (in / 64) * 2, .dtype = .BF16, .shape = .{ out, in / 64, 0 }, .rank = 2 },
        };
        for (parts, 0..) |sg, j| {
            segs[p * 3 + j] = sg;
            segs[p * 3 + j].offset = off;
            off += sg.length;
        }
    }
    return segs;
}
pub fn logicalBytes(segs: *const [n_components]Segment) u64 {
    return segs[n_components - 1].offset + segs[n_components - 1].length;
}
pub fn recordBytes(logical: u64) u64 {
    return std.mem.alignForward(u64, logical, 4096);
}
