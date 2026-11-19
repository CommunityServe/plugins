//! The plugin's claim, validation and bill, MLX-free. It claims exllamav3's `quantization_config` (`quant_method` exl3)
//! on a GLM-5.3-Flash pack in the streaming repack (`expert_streaming` with `family` turboexl3, the converter's
//! LAYOUT.md); mlx-serve's own glm5_next serves the pack and reads its EXL3 linears, the MTP layer's experts and the
//! routed bank through this plugin.

const std = @import("std");
const abi = @import("mlx_serve").abi.quant;

/// The exllamav3 codebook this plugin serves: the bank's kernels decode mul1 only.
pub const Codebook = enum(u8) { mul1 };

/// The bank block's family this plugin claims (LAYOUT.md `expert_streaming.family`).
pub const family = "turboexl3";

/// The files the two banks open, at the pack's root: the routed layers' bank and the MTP layer's (src/mtp_bank.zig).
pub const bank_files = [_][]const u8{ "experts.bin", "expert-manifest-v2.json", "expert-manifest.json", "mtp-experts.bin", "mtp-expert-manifest-v2.json", "mtp-expert-manifest.json" };

/// The bank block's fields as the converter writes them (LAYOUT.md); `source` is provenance, not read.
const bank_block = [_][2][]const u8{
    .{ "family", family },
    .{ "format", "mtplx-expert-manifest-v2" },
    .{ "manifest", "expert-manifest-v2.json" },
    .{ "runtime_manifest", "expert-manifest.json" },
    .{ "mtp_manifest", "mtp-expert-manifest-v2.json" },
    .{ "mtp_runtime_manifest", "mtp-expert-manifest.json" },
    .{ "quant", "exl3-mul1-k-by-layer" },
};

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

fn str(v: std.json.Value, key: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(key) orelse return null;
    return if (f == .string) f.string else null;
}

fn isEq(a: ?[]const u8, b: []const u8) bool {
    return if (a) |x| std.mem.eql(u8, x, b) else false;
}

/// A glm5_next pack whose bank block is this plugin's family, with every field `bank_block` names (`all`) or the
/// family alone.
fn isRepack(config_json: []const u8, all: bool) bool {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return false;
    if (doc != .object or !isEq(str(doc, "model_type"), "glm5_next")) return false;
    const es = doc.object.get("expert_streaming") orelse return false;
    if (!all) return isEq(str(es, "family"), family);
    for (bank_block) |kv| if (!isEq(str(es, kv[0]), kv[1])) return false;
    return true;
}

pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    const b = parseBlock(block_json.slice()) orelse return .no;
    if (!std.mem.eql(u8, b.quant_method, "exl3") or !isRepack(config_json.slice(), false)) return .no;
    return .by_marker;
}

pub fn validate(block_json: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Verdict {
    if (codebookOf(block_json.slice()) == null) return abi.Verdict.refuse(.unsupported, "turboexl3_glm53f: the codebook is not mul1");
    if (!isRepack(config_json.slice(), true)) return abi.Verdict.refuse(.unsupported, "turboexl3_glm53f: the expert_streaming block is not the streaming repack's (LAYOUT.md)");
    for (bank_files) |f| if (!dir.has(f)) return abi.Verdict.refuse(.unsupported, "turboexl3_glm53f: the pack lacks its banks (experts.bin, mtp-experts.bin and their manifests)");
    return .ok;
}

/// Resident as the pack stores it; the bank's slot rows are the host's stream room.
pub fn bill(block_json: abi.Str, model_dir: abi.Str, stored: u64) callconv(.c) u64 {
    _ = block_json;
    _ = model_dir;
    return stored;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("mlx_serve").sdk.fixture;

const sc310 = @embedFile("tests/fixtures/config.json");
const block = "{\"quant_method\":\"exl3\",\"version\":\"1.6.1\",\"bits\":3.09,\"codebook\":\"mul1\"}";

/// `json` with the first `old` replaced by `new`.
fn edited(a: std.mem.Allocator, json: []const u8, old: []const u8, new: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, json, old) orelse return error.TestUnexpectedResult;
    return std.mem.concat(a, u8, &.{ json[0..at], new, json[at + old.len ..] });
}

test "turboexl3_glm53f config: claims exllamav3's block on the streaming repack only" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(abi.Claim.by_marker, claims(.of(block), .of(sc310)));
    try testing.expectEqual(abi.Claim.no, claims(.of("{\"quant_method\":\"fp8\"}"), .of(sc310)));
    try testing.expectEqual(abi.Claim.no, claims(.of("{"), .of(sc310)));
    try testing.expectEqual(abi.Claim.no, claims(.of(block), .of(try edited(a, sc310, "\"family\": \"turboexl3\"", "\"family\": \"mlx-stream\""))));
    try testing.expectEqual(abi.Claim.no, claims(.of(block), .of(try edited(a, sc310, "\"model_type\": \"glm5_next\",\n  \"transformers", "\"model_type\": \"deepseek_v41\",\n  \"transformers"))));
    try testing.expectEqual(abi.Claim.no, claims(.of(block), .of("{\"model_type\":\"glm5_next\"}")));
}

test "turboexl3_glm53f config: validates mul1 with both banks; refuses another codebook, bank block or a missing bank by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files: [bank_files.len + 1]fixture.File = undefined;
    files[0] = .{ .name = "config.json", .bytes = sc310 };
    for (bank_files, files[1..]) |n, *f| f.* = .{ .name = n, .bytes = "" };
    const full: fixture.MemDir = .{ .files = &files };
    const dir = full.packDir();
    try testing.expect(validate(.of(block), .of(sc310), &dir).isOk());
    try testing.expectEqualStrings("turboexl3_glm53f: the codebook is not mul1", validate(.of("{\"quant_method\":\"exl3\",\"codebook\":\"mcg\"}"), .of(sc310), &dir).why.slice());
    const k3 = try edited(a, sc310, "\"quant\": \"exl3-mul1-k-by-layer\"", "\"quant\": \"exl3-k3-mul1\"");
    try testing.expectEqual(abi.Verdict.Code.unsupported, validate(.of(block), .of(k3), &dir).code);
    const no_mtp: fixture.MemDir = .{ .files = files[0..4] };
    const nd = no_mtp.packDir();
    try testing.expectEqualStrings("turboexl3_glm53f: the pack lacks its banks (experts.bin, mtp-experts.bin and their manifests)", validate(.of(block), .of(sc310), &nd).why.slice());
}
