//! copied_from: src/plugins/minimax_m3/legacy.zig@90fb57282d374c0989a46503b12f4fd9be300963
//! MiniMax M3.1 Flash load config: settings, load facts, the memory bill and the reader staging.
const std = @import("std");
const sdk = @import("mlx_stream").sdk;
const model_config = @import("geometry.zig");
const bank = @import("bank.zig");
const prepared = @import("resident_header.zig");
const usage = @import("mlx_stream").sdk_ext.expert.usage;
const resident_layout = @import("resident_layout.zig");
const cache_layout = @import("cache.zig");
pub const name = "minimax_m3";
pub const default_context = model_config.default_context;
pub const Module = @import("module.zig").Module;
pub const Config = struct {
    model: model_config.Config = .{},
    model_dir: ?[]const u8 = null,
    context: u32 = default_context,
    kv_quant_bits: u8 = 0,
    expert_rows: u32 = 4,
    expert_rows_forced: bool = false,
    settings_error: bool = false,
    residents_format: resident_layout.Format = .bf16,
    expert_format: bank.Kind = .nvfp4,
    residency: usage.Settings = .{},
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw == .null) return;
        if (raw != .object) {
            c.settings_error = true;
            return;
        }
        _ = c.residency.apply(raw.object);
        // Speculation and multimodal execution are deliberately refused.
        for (raw.object.keys(), raw.object.values()) |key, val| {
            if (std.mem.eql(u8, key, "expert_rows")) {
                if (val == .integer and val.integer >= 4 and val.integer <= 128) {
                    c.expert_rows = @intCast(val.integer);
                    c.expert_rows_forced = true;
                } else c.settings_error = true;
            } else if (std.mem.eql(u8, key, "mtp_lane") or std.mem.eql(u8, key, "mtp") or std.mem.eql(u8, key, "vision")) {
                if (val != .bool or val.bool) c.settings_error = true;
            }
        }
    }
};
pub fn applyLoadFacts(c: *Config, facts: *const sdk.LoadFacts) !void {
    if (c.settings_error) return error.MiniMaxUnsupportedSettings;
    try c.residency.check(name);
    if (facts.kv_quant_bits) |bits| {
        if (bits != 0 and bits != 8) return error.MiniMaxKvPrecisionUnsupported;
        c.kv_quant_bits = @intCast(bits);
    }
    if (facts.effective_context) |ctx| {
        if (ctx == 0 or ctx > c.model.max_position_embeddings) return error.MiniMaxContextUnsupported;
        c.context = @intCast(ctx);
    }
    if (facts.expert_rows) |rows| {
        c.expert_rows = rows;
        c.expert_rows_forced = true;
    }
    if (facts.expert_prefill_rows) |rows| if (rows != c.expert_rows) return error.MiniMaxPhaseRowsUnsupported;
    if (c.expert_rows < 4 or c.expert_rows > 128) return error.MiniMaxExpertRows;
}
pub fn memoryBill(a: std.mem.Allocator, io: std.Io, c: *const Config) !sdk.MemoryBill {
    const dir = c.model_dir orelse return error.MiniMaxModelDirectoryRequired;
    const geometry = try bank.inspect(a, io, dir);
    if (geometry.kind != c.expert_format) return error.MiniMaxExpertFormatMismatch;
    try bank.requireModel(&geometry, &c.model);
    const residents = try prepared.residentBytesBound(a, io, dir, 64 << 30);
    const slots = geometry.slotRowBytes();
    // Capacity is rounded, and a growing layer can briefly retain its old
    // buffers beside the replacement. Other layers retain one capacity each.
    const kv = if (c.kv_quant_bits == 8)
        try @import("q8_reference.zig").cacheBytes(c.model.n_layers, c.model.n_kv_heads, c.model.head_dim, c.model.index_head_dim, cache_layout.paddedCapacity(c.context))
    else
        c.model.kvBytes(cache_layout.paddedCapacity(c.context));
    const kv_copy = kv / c.model.n_layers;
    const reader = readerStagingBytes(&geometry);
    const scratch = @import("experts.zig").chunk_routes * slots + 256 * @as(u64, c.model.hidden_size) + 8 * @as(u64, c.model.n_heads) * c.context;
    const span = @import("prefill_step.zig").spanBytes(c.model.hidden_size, c.context) +
        @import("prefill_step.zig").sparseTileBytes(c.model.index_n_heads, c.model.index_block_size, c.context);
    const terms = try a.dupe(sdk.MemoryBill.Term, &.{
        .{ .name = "residents", .bytes = .{ residents, residents }, .at_construction = true },
        .{ .name = "kv_and_index", .bytes = .{ kv, kv }, .at_construction = false },
        .{ .name = "kv_append", .bytes = .{ kv_copy, kv_copy }, .at_construction = false },
        .{ .name = "reader_staging", .bytes = .{ reader, reader }, .at_construction = true },
        .{ .name = "serial_scratch", .bytes = .{ scratch, scratch }, .at_construction = false },
        .{ .name = "prefill_span", .bytes = .{ span, 0 }, .at_construction = false },
        .{ .name = "mlx_runtime_allowance", .bytes = .{ 1 << 30, 1 << 30 }, .at_construction = false },
    });
    return .{ .terms = terms, .per_row = slots * geometry.num_moe_layers };
}
pub fn readerStagingBytes(geometry: anytype) u64 {
    const page: u64 = std.heap.pageSize();
    const widest = @max(geometry.gate_up_bytes, geometry.logical_bytes - geometry.gate_up_bytes);
    return 2 * (std.mem.alignForward(u64, widest, page) + page);
}
test "MiniMax load facts refuse unsupported KV, context and phase rows" {
    var c: Config = .{};
    try applyLoadFacts(&c, &.{ .effective_context = 8192, .expert_rows = 8, .wired_margin_bytes = 0 });
    try std.testing.expectEqual(@as(u32, 8192), c.context);
    try std.testing.expectEqual(@as(u32, 8), c.expert_rows);
    try applyLoadFacts(&c, &.{ .kv_quant_bits = 8, .wired_margin_bytes = 0 });
    try std.testing.expectEqual(@as(u8, 8), c.kv_quant_bits);
    try std.testing.expectError(error.MiniMaxKvPrecisionUnsupported, applyLoadFacts(&c, &.{ .kv_quant_bits = 4, .wired_margin_bytes = 0 }));
    try std.testing.expectError(error.MiniMaxContextUnsupported, applyLoadFacts(&c, &.{ .effective_context = 1048577, .wired_margin_bytes = 0 }));
    try std.testing.expectError(error.MiniMaxPhaseRowsUnsupported, applyLoadFacts(&c, &.{ .expert_prefill_rows = 4, .wired_margin_bytes = 0 }));
}
test "MiniMax settings refuse speculative and vision lanes" {
    for ([_][]const u8{ "{\"mtp_lane\":true}", "{\"vision\":true}" }) |text| {
        const raw = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer raw.deinit();
        var c: Config = .{};
        c.applySettings(raw.value);
        try std.testing.expectError(error.MiniMaxUnsupportedSettings, applyLoadFacts(&c, &.{ .wired_margin_bytes = 0 }));
    }
}
test "MiniMax forwards shared usage settings and rejects malformed usage labels" {
    const a = std.testing.allocator;
    const raw = try std.json.parseFromSlice(std.json.Value, a, "{\"resident_seed\":\"none\",\"expert_usage_record\":\"/tmp/minimax-usage-fixture.json\",\"expert_usage_label\":\"fixture\"}", .{});
    defer raw.deinit();
    var c: Config = .{};
    c.applySettings(raw.value);
    try applyLoadFacts(&c, &.{ .wired_margin_bytes = 0 });
    try std.testing.expectEqual(usage.Seed.none, c.residency.seed.?);
    try std.testing.expectEqualStrings("fixture", c.residency.label());
    try std.testing.expectEqualStrings("/tmp/minimax-usage-fixture.json", c.residency.recordPath().?);
    const bad = try std.json.parseFromSlice(std.json.Value, a, "{\"expert_usage_label\":\"fixture\"}", .{});
    defer bad.deinit();
    var b: Config = .{};
    b.applySettings(bad.value);
    try std.testing.expectError(error.ExpertUsageLabelWithoutRecord, applyLoadFacts(&b, &.{ .wired_margin_bytes = 0 }));
}
