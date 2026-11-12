//! The plugin's claims and parse, MLX-free. A narrow claim: a GLM-5.3-Flash pack (`glm5_next`) whose config carries
//! exllamav3's `quantization_config` (`quant_method` exl3) and the streaming repack's bank block with `family`
//! `turboexl3` (the converter's LAYOUT.md); the built-in glm5_next refuses that block by name and keeps every other pack.
//! The parse is the built-in's (the copy's `plugins/glm5_next/config.zig`) over the config without the bank block, after
//! the pack is checked against what this plugin serves: the mul1 codebook, the bank block's fields, and the two banks
//! beside the trunk (`experts.bin`, `mtp-experts.bin` and their manifests). The parsed facts are then the built-in's for a
//! streamed pack.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const glm = @import("src/mlx_serve/plugins/glm5_next/config.zig");
const log = @import("src/mlx_serve/sdk/util/log.zig");
const stream_room = @import("src/stream_room.zig");

pub const Config = glm.Config;
pub const of = glm.of;
pub const freeParsed = glm.freeParsed;
pub const model_types = [_][]const u8{"glm5_next"};

const gpa = std.heap.c_allocator;

/// The bank block's family this plugin claims (LAYOUT.md `expert_streaming.family`).
pub const family = "turboexl3";

/// `v.object[key]` as a string, or null.
fn str(v: std.json.Value, key: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(key) orelse return null;
    return if (f == .string) f.string else null;
}

fn isEq(a: ?[]const u8, b: []const u8) bool {
    return if (a) |x| std.mem.eql(u8, x, b) else false;
}

pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) kit.abi.common.Claim {
    _ = dir;
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch return .no;
    if (doc != .object) return .no;
    const qc = doc.object.get("quantization_config") orelse return .no;
    if (!isEq(str(qc, "quant_method"), "exl3")) return .no;
    const es = doc.object.get("expert_streaming") orelse return .no;
    return if (isEq(str(es, "family"), family)) .by_marker else .no;
}

pub const PackError = error{ Exl3CodebookUnsupported, Exl3QuantConfigNotAnObject, Glm53fBankBlock, Glm53fPackNotRepacked };

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

/// The files the two banks open, at the pack's root: the routed layers' bank (the copied streamer) and the MTP layer's
/// (src/mtp_bank.zig).
pub const bank_files = [_][]const u8{ "experts.bin", "expert-manifest-v2.json", "expert-manifest.json", "mtp-experts.bin", "mtp-expert-manifest-v2.json", "mtp-expert-manifest.json" };

/// The pack against what the plugin serves. JSON that does not parse is the built-in parse's to refuse.
pub fn checkPack(dir: *const abi.PackDir, json: []const u8) PackError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}) catch return;
    if (doc != .object) return;
    const qc = doc.object.get("quantization_config") orelse return error.Exl3QuantConfigNotAnObject;
    if (qc != .object) return error.Exl3QuantConfigNotAnObject;
    // exllamav3 names its codebook; turboexl3 decodes mul1.
    if (!isEq(str(qc, "codebook"), "mul1")) return error.Exl3CodebookUnsupported;
    const es = doc.object.get("expert_streaming") orelse return error.Glm53fBankBlock;
    for (bank_block) |kv| if (!isEq(str(es, kv[0]), kv[1])) return error.Glm53fBankBlock;
    for (bank_files) |f| if (!dir.has(f)) return error.Glm53fPackNotRepacked;
}

/// `json` without its root's `expert_streaming` block, as the built-in parse reads a resident pack; caller-owned.
pub fn withoutBankBlock(a: std.mem.Allocator, json: []const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{});
    if (doc != .object) return error.InvalidConfigField;
    _ = doc.object.orderedRemove("expert_streaming");
    var w: std.Io.Writer.Allocating = .init(a);
    errdefer w.deinit();
    try std.json.Stringify.value(doc, .{}, &w.writer);
    return w.toOwnedSlice();
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    checkPack(dir, config_json.slice()) catch |e| {
        log.err("[turboexl3_glm53f] the pack is refused: {s}\n", .{@errorName(e)});
        return switch (e) {
            inline else => |x| abi.Verdict.refuse(.unsupported, "turboexl3_glm53f: " ++ @errorName(x)),
        };
    };
    if (span != null) return abi.Verdict.refuse(.unsupported, "turboexl3_glm53f: no Thunderbolt pair (a streamed bank serves one Mac)");
    const text = withoutBankBlock(gpa, config_json.slice()) catch return abi.Verdict.refuse(.failed, "turboexl3_glm53f: config.json does not parse");
    defer gpa.free(text);
    const v = glm.parse(dir, abi.Str.of(text), settings_json, span, stub, out);
    if (v.isOk()) served(out);
    return v;
}

/// The built-in's facts for a streamed pack (`glmFacts` on `expert_streaming`): the routed experts stream from the
/// bank and one request decodes at a time (the bank's decode lane takes a few rows a call). The vision tower is not
/// served yet: no images are advertised. With a kit that has it, the split prompt forward (`Caps.split_prefill`): the
/// copied forward runs a prompt's two spans layer by layer with one streamed routed call a layer
/// (patches/turboexl3-model.patch and patches/turboexl3-sdk.patch).
fn served(p: *abi.Parsed) void {
    const c: *Config = @ptrCast(@alignCast(p.private.?));
    c.streamed = true;
    c.advertises_images = false;
    p.caps.streamed_weights = true;
    p.caps.batches_decode = false;
    p.caps.images = false;
    if (@hasField(abi.Caps, "split_prefill")) p.caps.split_prefill = true;
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

const sc310 = @embedFile("tests/fixtures/config.json");
const no_bank = [_]fixture.File{.{ .name = "config.json", .bytes = sc310 }};
const repacked = [_]fixture.File{
    .{ .name = "config.json", .bytes = sc310 },
    .{ .name = "experts.bin", .bytes = "" },
    .{ .name = "expert-manifest-v2.json", .bytes = "{}" },
    .{ .name = "expert-manifest.json", .bytes = "{}" },
    .{ .name = "mtp-experts.bin", .bytes = "" },
    .{ .name = "mtp-expert-manifest-v2.json", .bytes = "{}" },
    .{ .name = "mtp-expert-manifest.json", .bytes = "{}" },
};
const no_mtp_bank = repacked[0..4];

fn claimOf(model_type: []const u8, json: []const u8) kit.abi.common.Claim {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return claims(abi.Str.of(model_type), abi.Str.of(json), &dir);
}

fn parseOf(files: []const fixture.File, json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

/// `json` with the first `old` replaced by `new`.
fn edited(a: std.mem.Allocator, json: []const u8, old: []const u8, new: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, json, old) orelse return error.TestUnexpectedResult;
    return std.mem.concat(a, u8, &.{ json[0..at], new, json[at + old.len ..] });
}

test "turboexl3_glm53f claims the SC repack by marker, and neither the built-in's bank nor a pack without exllamav3's block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(kit.abi.common.Claim.by_marker, claimOf("glm5_next", sc310));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("glm5_next_text", sc310));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("deepseek_v41", sc310));
    for ([_][]const u8{ "{", "[]", "7", "" }) |json| try testing.expectEqual(kit.abi.common.Claim.no, claimOf("glm5_next", json));
    // the built-in's 3.05 repack (mlx-stream's K3 bank and fp8 residents) and an MLX pack: the built-in keeps them
    const k3 = try edited(a, sc310, "\"family\": \"turboexl3\"", "\"family\": \"mlx-stream\"");
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("glm5_next", k3));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("glm5_next", "{\"model_type\":\"glm5_next\",\"quantization\":{\"group_size\":64,\"bits\":4}}"));
    const not_exl3 = try edited(a, sc310, "\"quant_method\": \"exl3\"", "\"quant_method\": \"fp8\"");
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("glm5_next", not_exl3));
}

test "turboexl3_glm53f parses the SC 3.10 repack into the built-in's streamed shell, serially decoded" {
    var p: abi.Parsed = .{};
    const v = parseOf(&repacked, sc310, &p);
    try testing.expect(v.isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 154880), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 4096), p.shell.hidden_size);
    try testing.expectEqual(@as(u32, 45), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 8), p.shell.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 2048), p.shell.moe_intermediate_size);
    try testing.expect(p.caps.streamed_weights and !p.caps.batches_decode and !p.caps.images);
    if (@hasField(abi.Caps, "split_prefill")) try testing.expect(p.caps.split_prefill);
    const c = of(&p);
    try testing.expect(c.streamed);
    try testing.expectEqual(@as(u32, 288), c.num_experts);
    try testing.expectEqual(@as(u32, 3), c.first_k_dense_replace);
    try testing.expectEqual(@as(f32, 10.0), c.swiglu_limit);
    try testing.expect(c.swiglu_clamp);
}

test "turboexl3_glm53f refuses by name a pack before its repack, without its MTP bank, with another codebook or bank block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mcg = try edited(a, sc310, "\"codebook\": \"mul1\"", "\"codebook\": \"mcg\"");
    const k3 = try edited(a, sc310, "\"quant\": \"exl3-mul1-k-by-layer\"", "\"quant\": \"exl3-k3-mul1\"");
    const cases = [_]struct { files: []const fixture.File, json: []const u8, why: []const u8 }{
        .{ .files = &no_bank, .json = sc310, .why = "turboexl3_glm53f: Glm53fPackNotRepacked" },
        .{ .files = no_mtp_bank, .json = sc310, .why = "turboexl3_glm53f: Glm53fPackNotRepacked" },
        .{ .files = &repacked, .json = mcg, .why = "turboexl3_glm53f: Exl3CodebookUnsupported" },
        .{ .files = &repacked, .json = k3, .why = "turboexl3_glm53f: Glm53fBankBlock" },
    };
    for (cases) |c| {
        var p: abi.Parsed = .{};
        const v = parseOf(c.files, c.json, &p);
        try testing.expect(!v.isOk() and p.private == null);
        try testing.expectEqualStrings(c.why, v.why.slice());
    }
}

test "turboexl3_glm53f hands the built-in parse the config without its bank block, every other key kept" {
    const text = try withoutBankBlock(testing.allocator, sc310);
    defer testing.allocator.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "expert_streaming") == null);
    try testing.expect(std.mem.indexOf(u8, text, "\"quant_method\":\"exl3\"") != null);
    try testing.expect(std.mem.indexOf(u8, text, "\"n_routed_experts\":288") != null);
}

test "turboexl3_glm53f stream room: the SC 3.10 config's prompt bill is the engine's on its own GLM fixture" {
    var p: abi.Parsed = .{};
    try testing.expect(parseOf(&repacked, sc310, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    // the parse reads the release's layer types and linear_attn_config: 11 DSA layers, 34 KDA of 64 x 128 at conv 4
    try testing.expectEqual(@as(u32, 11), c.attnLayerCount());
    try testing.expectEqual(@as(u64, 147_619_840), c.ssmCheckpointBytes(4));
    try testing.expectEqual(@as(u64, 21_368_258_560), stream_room.prefillNeed(c, stream_room.room_seq, stream_room.room_chunk, 16, 0, true));
}

test {
    _ = @import("src/mtp_bank.zig");
    _ = @import("src/pack_weights.zig");
    _ = @import("src/bank_layers.zig");
    _ = stream_room;
}
