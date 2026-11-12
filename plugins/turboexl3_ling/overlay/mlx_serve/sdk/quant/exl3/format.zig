//! turboexl3_ling's own, in place of mlx-serve's `sdk/quant/exl3/format.zig` (Sushi's `expert_quant` expert banks, which
//! the copied trunk reads for a Sushi GLM pack): a Ling pack has no `expert_quant` block (its routed experts are the
//! streamed bank's), so the trunk never takes this arm; a pack that names one is refused by name.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;

pub const Spec = struct {};
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const Decode = struct { codebook: enum(u8) { mul1 } = .mul1, window: Window = .{}, table: ?u32 = null };
pub const Window = struct {
    pub fn bits(_: Window) u8 {
        return 16;
    }
};

pub fn parse(root: std.json.ObjectMap, topk: u32) !Spec {
    _ = root;
    _ = topk;
    return error.Exl3SushiExpertQuantUnsupported;
}

pub fn decode(spec: Spec) Decode {
    _ = spec;
    return .{};
}

pub fn admit(p: Proj, experts: u32, in_dim: u32, out_dim: u32, spec: Spec) !void {
    _ = .{ p, experts, in_dim, out_dim, spec };
    return error.Exl3SushiExpertQuantUnsupported;
}
