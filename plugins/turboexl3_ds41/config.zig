//! The plugin's claim, validation and bill, MLX-free. It claims exllamav3's `quantization_config` (`quant_method`
//! exl3) on a DeepSeek-V4.1 pack; mlx-serve's own deepseek_v41 serves the pack and reads those weights through it.

const std = @import("std");
const abi = @import("mlx_serve").abi.quant;

/// exllamav3's codebooks, in `Weight.layout` order: turboexl3 decodes both.
pub const Codebook = enum(u8) { mul1, mcg };

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

fn isDsv41(config_json: []const u8) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const S = struct { model_type: []const u8 = "" };
    const c = std.json.parseFromSliceLeaky(S, arena.allocator(), config_json, .{ .ignore_unknown_fields = true }) catch return false;
    return std.mem.eql(u8, c.model_type, "deepseek_v41");
}

/// The trunk's hyper-connection streams (`text_config.hc_mult`): the fused tail's `res` is [rows, hc, hidden].
pub fn hcOf(config_json: []const u8) ?u32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const S = struct { text_config: struct { hc_mult: u32 = 0 } = .{} };
    const c = std.json.parseFromSliceLeaky(S, arena.allocator(), config_json, .{ .ignore_unknown_fields = true }) catch return null;
    return if (c.text_config.hc_mult > 0) c.text_config.hc_mult else null;
}

pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    const b = parseBlock(block_json.slice()) orelse return .no;
    if (!std.mem.eql(u8, b.quant_method, "exl3") or !isDsv41(config_json.slice())) return .no;
    return .by_marker;
}

pub fn validate(block_json: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Verdict {
    _ = config_json;
    _ = dir;
    if (codebookOf(block_json.slice()) == null) return abi.Verdict.refuse(.unsupported, "turboexl3_ds41: the codebook is neither mul1 nor mcg");
    return .ok;
}

/// Resident as the pack stores it.
pub fn bill(block_json: abi.Str, model_dir: abi.Str, stored: u64) callconv(.c) u64 {
    _ = block_json;
    _ = model_dir;
    return stored;
}

// ── tests ──

const testing = std.testing;

const sc281 = @embedFile("tests/fixtures/config.json");

test "turboexl3_ds41 config: claims exllamav3's block on a DeepSeek-V4.1 pack only; validates its codebook by name" {
    const block = "{\"quant_method\":\"exl3\",\"bits\":2.81,\"codebook\":\"mul1\"}";
    try testing.expectEqual(abi.Claim.by_marker, claims(.of(block), .of(sc281)));
    try testing.expectEqual(abi.Claim.no, claims(.of(block), .of("{\"model_type\":\"qwen4_exp\"}")));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"quant_method\":\"fp8\"}"), .of(sc281)));
    try testing.expectEqual(abi.Claim.no, claims(.of("{"), .of(sc281)));
    const dir: abi.PackDir = undefined;
    try testing.expect(validate(.of(block), .of(sc281), &dir).isOk());
    try testing.expectEqual(Codebook.mcg, codebookOf("{\"quant_method\":\"exl3\",\"codebook\":\"mcg\"}").?);
    try testing.expectEqual(abi.Verdict.Code.unsupported, validate(.of("{\"quant_method\":\"exl3\",\"codebook\":\"3inst\"}"), .of(sc281), &dir).code);
}

test "turboexl3_ds41 config: the trunk's hyper-connection streams from text_config" {
    try testing.expectEqual(@as(?u32, 4), hcOf(sc281));
    try testing.expectEqual(@as(?u32, null), hcOf("{\"model_type\":\"deepseek_v41\"}"));
}
