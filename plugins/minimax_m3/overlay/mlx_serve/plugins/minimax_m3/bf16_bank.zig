//! copied_from: src/plugins/minimax_m3/bf16_bank.zig@9df2c52b504cce5bbf22f074dab168fd8584f961
//! MiniMax routed experts as a BF16 expert bank: `experts.bin` beside `expert-manifest.json`, one record per
//! (MoE layer, expert) in the `bf16-gud-3part` layout: gate_proj.weight BF16 [inter, hidden], up_proj.weight BF16
//! [inter, hidden], down_proj.weight BF16 [hidden, inter], back to back, records page-aligned. The bf16 pack the plugin
//! rules ask for: the release ships NVFP4 experts, so a converted bank holds their values rounded to BF16.
const std = @import("std");
const sdk_ext = @import("mlx_stream").sdk_ext;
const mlx = @import("mlx_stream").sdk.mlx;
const bank_file = sdk_ext.expert.bank_file;
const Module = @This();
pub const format = "mlx-stream-bf16-expert-bank-v1";
pub const layout = "bf16-gud-3part";
pub const n_components = 3;
pub const gu_components = 2;
pub const routed_top_k = 4;
pub const page = 4096;
pub const Records = sdk_ext.expert.Records(n_components, gu_components);
pub const Dtype = enum { BF16 };
pub const Component = enum(u8) {
    gate_weight,
    up_weight,
    down_weight,
    pub fn name(c: Component) []const u8 {
        return part_names[@intFromEnum(c)];
    }
};
const part_names = [_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" };
pub const Segment = struct { offset: u64, length: u64, dtype: Dtype = .BF16, shape: [3]u64 = @splat(0), rank: u8 = 2 };
pub const Layer = struct {
    model_layer: u32,
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};
/// One projection's slot array: its BF16 weights [slots, out, in].
pub fn Arrays(comptime T: type) type {
    return struct { w: T };
}
pub const BankArrays = sdk_ext.quant.BankArrays(Arrays(mlx.mlx_array));
pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{ .gate = .{ .w = x[0] }, .up = .{ .w = x[1] }, .down = .{ .w = x[2] } };
}
pub fn mlxDtype(_: Dtype) mlx.mlx_dtype {
    return .bfloat16;
}
pub const requireModel = @import("bank.zig").requireModel;
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    logical_bytes: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]Segment,
    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, count: u64) !Geometry {
        if (hidden == 0 or inter == 0 or hidden > 65536 or inter > 65536 or experts == 0 or experts > 65535 or count == 0 or count > 4096 or first > 4096 or first + count > 4096) return error.MiniMaxBf16Geometry;
        const gu = inter * hidden * 2;
        const segments = [n_components]Segment{
            .{ .offset = 0, .length = gu, .shape = .{ inter, hidden, 0 } },
            .{ .offset = gu, .length = gu, .shape = .{ inter, hidden, 0 } },
            .{ .offset = 2 * gu, .length = gu, .shape = .{ hidden, inter, 0 } },
        };
        const logical = 3 * gu;
        const stride = std.mem.alignForward(u64, logical, page);
        return .{ .hidden_size = @intCast(hidden), .intermediate_size = @intCast(inter), .num_experts = @intCast(experts), .first_moe_layer = @intCast(first), .num_moe_layers = @intCast(count), .logical_bytes = logical, .record_bytes = stride, .gate_up_bytes = 2 * gu, .total_bytes = try std.math.mul(u64, try std.math.mul(u64, experts, count), stride), .segments = segments };
    }
    pub fn slotRowBytes(self: *const Geometry) u64 {
        return self.logical_bytes;
    }
};
const Manifest = struct {
    format: []const u8,
    data_file: []const u8,
    layout: []const u8,
    alignment: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_experts: u64,
    first_moe_layer: u64,
    num_moe_layers: u64,
    record_bytes: u64,
    record_stride: u64,
    gate_up_bytes: u64,
    down_bytes: u64,
    source: struct { repository: []const u8, revision: []const u8 },
    parts: []const struct { name: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 },
    records: []const struct { layer: u64, expert: u64, offset: u64, length: u64, sha256: []const u8 },
    output_files: std.json.Value,
};
pub const Bank = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    digests: [][32]u8 = &.{},
    sidecar_fd: std.c.fd_t = -1,
    sidecar_file_size: u64 = 0,
    pub fn fromText(a: std.mem.Allocator, text: []const u8) !Bank {
        const parsed = try std.json.parseFromSlice(Manifest, a, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return adopt(a, parsed.value);
    }
    fn adopt(a: std.mem.Allocator, m: Manifest) !Bank {
        if (!std.mem.eql(u8, m.format, format) or !std.mem.eql(u8, m.layout, layout) or !std.mem.eql(u8, m.data_file, "experts.bin") or m.alignment != page) return error.MiniMaxBf16Format;
        const g = try Geometry.init(m.hidden_size, m.intermediate_size, m.num_experts, m.first_moe_layer, m.num_moe_layers);
        if (m.record_bytes != g.logical_bytes or m.record_stride != g.record_bytes or m.gate_up_bytes != g.gate_up_bytes or m.down_bytes != g.logical_bytes - g.gate_up_bytes) return error.MiniMaxBf16Geometry;
        if (m.parts.len != n_components) return error.MiniMaxBf16Parts;
        for (m.parts, g.segments, part_names) |part, want, name| {
            if (!std.mem.eql(u8, part.name, name) or !std.mem.eql(u8, part.dtype, "BF16") or !std.mem.eql(u64, part.shape, want.shape[0..want.rank]) or part.offset != want.offset or part.length != want.length) return error.MiniMaxBf16Parts;
        }
        if (m.source.repository.len == 0 or m.source.revision.len != 40) return error.MiniMaxBf16Source;
        var revision: [20]u8 = undefined;
        _ = std.fmt.hexToBytes(&revision, m.source.revision) catch return error.MiniMaxBf16Source;
        if (m.records.len != @as(u64, g.num_moe_layers) * g.num_experts) return error.MiniMaxBf16Records;
        var out: Bank = .{ .allocator = a, .geometry = g };
        errdefer out.deinit();
        out.digests = try a.alloc([32]u8, m.records.len);
        for (m.records, out.digests, 0..) |r, *digest, i| {
            if (r.layer != g.first_moe_layer + i / g.num_experts or r.expert != i % g.num_experts or r.offset != @as(u64, i) * g.record_bytes or r.length != g.logical_bytes or r.sha256.len != 64) return error.MiniMaxBf16Records;
            _ = std.fmt.hexToBytes(digest, r.sha256) catch return error.MiniMaxBf16Records;
        }
        if (m.output_files != .object) return error.MiniMaxBf16Output;
        const file = m.output_files.object.get("experts.bin") orelse return error.MiniMaxBf16Output;
        if (file != .object) return error.MiniMaxBf16Output;
        const size = file.object.get("size") orelse return error.MiniMaxBf16Output;
        const sha = file.object.get("sha256") orelse return error.MiniMaxBf16Output;
        if (size != .integer or size.integer != g.total_bytes or sha != .string or sha.string.len != 64) return error.MiniMaxBf16Output;
        return out;
    }
    fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        if (!std.fs.path.isAbsolute(dir)) return error.BankDirNotAbsolute;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        return adopt(a, try bank_file.readManifest(Manifest, arena.allocator(), io, dir, "expert-manifest.json", null));
    }
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        var out = try read(a, io, dir);
        errdefer out.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
        defer a.free(path);
        const sidecar = try bank_file.openSidecar(path, out.geometry.total_bytes, .exact, null);
        out.sidecar_fd = sidecar.fd;
        out.sidecar_file_size = sidecar.size;
        return out;
    }
    pub fn deinit(self: *Bank) void {
        if (self.sidecar_fd >= 0) _ = std.c.close(self.sidecar_fd);
        self.allocator.free(self.digests);
        self.* = undefined;
    }
};
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var bank = try Bank.read(a, io, dir);
    defer bank.deinit();
    return bank.geometry;
}
pub const SDKBank = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    record_bytes: u64,
    sidecar: sdk_ext.expert.UncachedFd,
    pub fn init(a: std.mem.Allocator, owner: *const Bank) !SDKBank {
        const g = &owner.geometry;
        const entries = try a.alloc(Layer, g.num_moe_layers);
        for (entries, 0..) |*entry, i| entry.* = .{ .model_layer = @intCast(g.first_moe_layer + i), .record_bytes = g.record_bytes, .logical_bytes = g.logical_bytes, .base_offset = i * @as(u64, g.num_experts) * g.record_bytes, .segments = g.segments };
        return .{ .allocator = a, .layers = entries, .n_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .record_bytes = g.record_bytes, .sidecar = .{ .fd = owner.sidecar_fd, .size = owner.sidecar_file_size } };
    }
    pub fn deinit(self: *SDKBank) void {
        self.allocator.free(self.layers);
        self.* = undefined;
    }
    pub fn spans(self: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const offset = self.recordOffset(layer, expert);
        return .{ .gu_offset = offset, .down_offset = offset + self.layers[layer].segments[gu_components].offset };
    }
    pub fn recordOffset(self: *const SDKBank, layer: u32, expert: u32) u64 {
        return self.layers[layer].base_offset + @as(u64, expert) * self.record_bytes;
    }
};
pub const SDK = struct {
    pub const n_components = Module.n_components;
    pub const gu_components = Module.gu_components;
    pub const Component = Module.Component;
    pub const Layer = Module.Layer;
    pub const Records = Module.Records;
    pub const Bank = SDKBank;
    pub const BankArrays = Module.BankArrays;
    pub const bankArraysOf = Module.bankArraysOf;
    pub const mlxDtype = Module.mlxDtype;
    pub const routed_top_k = Module.routed_top_k;
    pub const slot_array_min_bytes = std.heap.page_size_max;
};
