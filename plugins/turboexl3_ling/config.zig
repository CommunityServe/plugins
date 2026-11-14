//! The plugin's claim, validation and bill, MLX-free. It claims exllamav3's `quantization_config` (`quant_method`
//! exl3) on a Ling pack (`bailing_hybrid`); mlx-serve's own bailing_hybrid serves the pack (it reads that block and the
//! `experts.bin` beside it as a streaming pack) and runs the trunk's EXL3 linears and the routed experts through this
//! plugin. The pack is checked against what the plugin serves: the mul1 codebook, the streaming repack's bank files,
//! and Ling-3.1's per-layer SwiGLU clamps (src/clamps.zig), which the bank applies to the routed experts itself.

const std = @import("std");
const abi = @import("mlx_serve").abi.quant;
const clamps = @import("src/clamps.zig");

/// The exllamav3 codebook this plugin serves: turboexl3's Ling kernels and the bank decode mul1 only.
pub const Codebook = enum(u8) { mul1 };

const Block = struct { quant_method: []const u8 = "", codebook: []const u8 = "" };

fn parseBlock(json: []const u8) ?Block {
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    return std.json.parseFromSliceLeaky(Block, fba.allocator(), json, .{ .ignore_unknown_fields = true }) catch null;
}

/// The block's codebook; null when it is no exllamav3 block or names another codebook.
pub fn codebookOf(block_json: []const u8) ?Codebook {
    const b = parseBlock(block_json) orelse return null;
    if (!std.mem.eql(u8, b.quant_method, "exl3")) return null;
    return std.meta.stringToEnum(Codebook, b.codebook);
}

/// What the bank reads from the model's config beside the host's `BankGeometry`: the dense layers before the first
/// routed one (bank layer = model layer - `first`) and the trunk's layer count the clamp lists are read against.
pub const Ling = struct { first: u32, layers: u32 };

/// `config_json` as a Ling pack this plugin reads (null: another model, or a field out of range).
pub fn lingOf(config_json: []const u8) ?Ling {
    var buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const S = struct { model_type: []const u8 = "", first_k_dense_replace: u32 = 0, num_hidden_layers: u32 = 0 };
    const c = std.json.parseFromSliceLeaky(S, fba.allocator(), config_json, .{ .ignore_unknown_fields = true }) catch return null;
    if (!std.mem.eql(u8, c.model_type, "bailing_hybrid")) return null;
    if (c.num_hidden_layers == 0 or c.num_hidden_layers > clamps.max_layers or c.first_k_dense_replace >= c.num_hidden_layers) return null;
    return .{ .first = c.first_k_dense_replace, .layers = c.num_hidden_layers };
}

pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    const b = parseBlock(block_json.slice()) orelse return .no;
    if (!std.mem.eql(u8, b.quant_method, "exl3") or lingOf(config_json.slice()) == null) return .no;
    return .by_marker;
}

pub const PackError = error{ Exl3CodebookUnsupported, Exl3PackNotRepacked, LingConfigUnreadable } || clamps.Error;

/// The bank files the streamer opens (the v2 manifest for the records, the v1 one for the runtime).
const bank_files = [_][]const u8{ "experts.bin", "expert-manifest-v2.json", "expert-manifest.json" };

/// The pack against what the plugin serves, and its clamps read (the process's one loaded pack).
pub fn checkPack(block_json: []const u8, config_json: []const u8, dir: ?*const abi.PackDir) PackError!void {
    if (codebookOf(block_json) == null) return error.Exl3CodebookUnsupported;
    if (dir) |d| for (bank_files) |f| if (!d.has(f)) return error.Exl3PackNotRepacked;
    const l = lingOf(config_json) orelse return error.LingConfigUnreadable;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return error.LingConfigUnreadable;
    if (doc != .object) return error.LingConfigUnreadable;
    try clamps.read(doc.object, l.layers);
}

/// The codebook and the clamps; the bank's files are `open`'s to check (the load's pack dir), so a pack without them is
/// refused by name there, before the trunk loads.
pub fn validate(block_json: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Verdict {
    _ = dir;
    checkPack(block_json.slice(), config_json.slice(), null) catch |e| return switch (e) {
        inline else => |x| abi.Verdict.refuse(.unsupported, "turboexl3_ling: " ++ @errorName(x)),
    };
    return .ok;
}

/// Resident as the pack stores it (the bank's slot rows are `bank_rows`' bill, apart).
pub fn bill(block_json: abi.Str, model_dir: abi.Str, stored: u64) callconv(.c) u64 {
    _ = block_json;
    _ = model_dir;
    return stored;
}

// ── tests ──

const testing = std.testing;

const ling31 = @embedFile("tests/fixtures/config-2.60bpw.json");
const block = "{\"quant_method\":\"exl3\",\"bits\":2.59,\"codebook\":\"mul1\"}";

test "turboexl3_ling config: claims exllamav3's block on a Ling pack only, by marker" {
    try testing.expectEqual(abi.Claim.by_marker, claims(.of(block), .of(ling31)));
    try testing.expectEqual(abi.Claim.no, claims(.of(block), .of("{\"model_type\":\"deepseek_v41\"}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"quant_method\":\"fp8\"}"), .of(ling31)));
    try testing.expectEqual(abi.Claim.no, claims(.of("{"), .of(ling31)));
}

test "turboexl3_ling config: Ling-3.1-flash's bank geometry beside the host's: 4 dense layers of 72" {
    const l = lingOf(ling31).?;
    try testing.expectEqual(Ling{ .first = 4, .layers = 72 }, l);
    try testing.expect(lingOf("{\"model_type\":\"bailing_hybrid\",\"num_hidden_layers\":4,\"first_k_dense_replace\":4}") == null);
}

test "turboexl3_ling config: the pack's clamps are read (routed 65-71, shared 64-71 at 10); another codebook is refused by name" {
    try checkPack(block, ling31, null);
    try testing.expectEqual(@as(u32, 0), clamps.routed(64));
    try testing.expectEqual(@as(u32, 10), clamps.routed(65));
    try testing.expectEqual(@as(u32, 10), clamps.shared(64));
    try testing.expectError(error.Exl3CodebookUnsupported, checkPack("{\"quant_method\":\"exl3\",\"codebook\":\"mcg\"}", ling31, null));
    try testing.expectError(error.LingConfigUnreadable, checkPack(block, "{\"model_type\":\"qwen4_exp\"}", null));
}
