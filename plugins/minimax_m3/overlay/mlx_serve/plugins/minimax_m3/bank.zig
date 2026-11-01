//! copied_from: src/plugins/minimax_m3/bank.zig@dc508a5da248397f6c952e75839f1de58aab5be2
//! MiniMax uses the shared nine-part NVFP4 wire format. The reader and dequantization
//! are deliberately shared with GLM: only the routed topology differs.
const std = @import("std");
const shared = @import("nvfp4_bank.zig");
pub const Kind = enum { nvfp4, affine, bf16 };
pub fn parseFormat(name: []const u8) !Kind {
    if (std.mem.eql(u8, name, shared.format)) return .nvfp4;
    if (std.mem.eql(u8, name, @import("affine_bank.zig").format)) return .affine;
    if (std.mem.eql(u8, name, @import("bf16_bank.zig").format)) return .bf16;
    return error.MiniMaxExpertFormatUnsupported;
}
pub const format = shared.format;
pub const layout = shared.layout;
pub const n_components = shared.n_components;
pub const gu_components = shared.gu_components;
pub const routed_top_k = 4;
pub const Records = shared.Records;
pub const Component = shared.Component;
pub const Layer = shared.Layer;
pub const SDKBank = shared.SDKBank;
pub const Bank = shared.Bank;
pub const Geometry = shared.Geometry;
pub const Arrays = shared.Arrays;
pub const BankArrays = shared.BankArrays;
pub const BankArraysOf = shared.BankArraysOf;
pub const bankArraysOf = shared.bankArraysOf;
pub const mlxDtype = shared.mlxDtype;
pub const Inspection = struct {
    kind: Kind,
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    logical_bytes: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    row_bytes: u64,
    fn of(kind: Kind, geometry: anytype) Inspection {
        return .{ .kind = kind, .hidden_size = geometry.hidden_size, .intermediate_size = geometry.intermediate_size, .num_experts = geometry.num_experts, .first_moe_layer = geometry.first_moe_layer, .num_moe_layers = geometry.num_moe_layers, .logical_bytes = geometry.logical_bytes, .record_bytes = geometry.record_bytes, .gate_up_bytes = geometry.gate_up_bytes, .total_bytes = geometry.total_bytes, .row_bytes = geometry.slotRowBytes() };
    }
    pub fn slotRowBytes(self: *const Inspection) u64 {
        return self.row_bytes;
    }
};
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Inspection {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const marker = try @import("mlx_stream").sdk_ext.expert.bank_file.readManifest(struct { format: []const u8 }, arena.allocator(), io, dir, "expert-manifest.json", null);
    const kind = try parseFormat(marker.format);
    return switch (kind) {
        .nvfp4 => Inspection.of(kind, try shared.inspect(a, io, dir)),
        .affine => Inspection.of(kind, try @import("affine_bank.zig").inspect(a, io, dir)),
        .bf16 => Inspection.of(kind, try @import("bf16_bank.zig").inspect(a, io, dir)),
    };
}
pub const SDK = struct {
    pub const n_components = shared.n_components;
    pub const gu_components = shared.gu_components;
    pub const Records = shared.Records;
    pub const Component = shared.Component;
    pub const Layer = shared.Layer;
    pub const Bank = shared.SDKBank;
    pub const mlxDtype = shared.mlxDtype;
    pub const BankArrays = shared.BankArrays;
    pub const bankArraysOf = shared.bankArraysOf;
    pub const routed_top_k = 4;
    pub const slot_line_bytes = shared.scale_line_bytes;
    pub const slot_min_bytes = shared.scale_array_min_bytes;
    pub const slot_array_min_bytes: u64 = std.heap.page_size_max;
};
/// The bank holds the routed experts of the model `m` describes: its widths, expert count and MoE layers.
pub fn requireModel(g: anytype, m: *const @import("geometry.zig").Config) !void {
    if (g.hidden_size != m.hidden_size or g.intermediate_size != m.intermediate_size or g.num_experts != m.n_routed_experts or
        g.first_moe_layer != @import("geometry.zig").first_moe_layer or g.num_moe_layers != m.moeLayers())
        return error.MiniMaxUnsupportedBankTopology;
}
