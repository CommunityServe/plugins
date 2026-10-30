//! The plugin's claims and parse, MLX-free. A narrow claim: a Qwen3.8 pack (`qwen4_exp`) whose config names a
//! `trunk_quant` block, which is a fully EXL3 pack. The built-in `qwen4_exp` keeps every other pack. The parse is the
//! direct model's (`src/direct/config.zig`, the K packs' keys), after the pack's EXL3 blocks are checked against what
//! turboexl3 decodes.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const q4 = @import("src/direct/config.zig");

pub const model_types = q4.model_types;

const gpa = std.heap.c_allocator;

/// The widest top-k turboexl3's routed-bank reduction holds.
const max_topk = 32;

pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) kit.abi.common.Claim {
    _ = dir;
    if (!isQwen38(model_type.slice())) return .no;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch return .no;
    if (doc != .object) return .no;
    // Any trunk block is this plugin's: `parse` refuses by name the ones turboexl3 does not decode.
    const t = doc.object.get("trunk_quant") orelse return .no;
    return if (t == .null) .no else .by_marker;
}

fn isQwen38(mt: []const u8) bool {
    for (model_types) |m| if (std.mem.eql(u8, mt, m)) return true;
    return false;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    checkPack(config_json.slice()) catch |e| return switch (e) {
        inline else => |x| abi.Verdict.refuse(.unsupported, "turboexl3_qwen38: " ++ @errorName(x)),
    };
    const v = q4.parse(dir, config_json, settings_json, span, stub, out);
    if (v.isOk()) served(out);
    return v;
}

/// Drafts per MTP round (src/spec/drafter.zig). The host hands a native drafter no MTP depth (`--mtp-depth` stays the
/// engine's): 3, the server's own `--mtp-max-depth` default and the depth of #1005's K2..K4 rows.
pub const mtp_depth: u32 = 3;

/// The built-in's facts as this plugin serves them: each slot's state is the model's own (the host's caches are its
/// engine's types), which the prefix cache keeps through the plugin's module snapshots (`CacheHooks.aux`), the engine's
/// MTP rounds do not reach it (the head drafts as the model's native drafter instead), and no images (no media hooks).
fn served(p: *abi.Parsed) void {
    const head = p.spec.mtp_layout != .none;
    p.caps.owns_decode_state = true;
    p.caps.slot_state = true;
    p.caps.prefix_cache = true;
    p.caps.spec_rollback = false;
    p.caps.images = false;
    p.caps.video = false;
    p.spec = .{ .native_stages = if (head) mtp_depth else 0, .native_block = if (head) mtp_depth + 1 else 0 };
}

pub const freeParsed = q4.freeParsed;
pub const of = q4.of;

const BlockError = error{ NotExl3, Exl3CodebookUnsupported, Exl3WindowUnsupported, Exl3RateUnsupported, Exl3TopKExceedsReduceBank };

/// The pack's EXL3 blocks against what turboexl3 decodes, named as turboexl3's own validation names them. JSON that
/// does not parse is the built-in parse's to refuse.
fn checkPack(json: []const u8) BlockError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}) catch return;
    if (doc == .object) try checkBlocks(doc.object);
}

fn checkBlocks(root: std.json.ObjectMap) BlockError!void {
    if (root.get("trunk_quant")) |t| if (t == .object) try checkBlock(t.object, false);
    if (root.get("expert_quant")) |e| if (e != .null) {
        if (e != .object) return error.NotExl3;
        try checkBlock(e.object, true);
        const text = if (root.get("text_config")) |tc| (if (tc == .object) tc.object else root) else root;
        if (text.get("num_experts_per_tok")) |k| if (k == .integer and k.integer > max_topk) return error.Exl3TopKExceedsReduceBank;
    };
}

/// turboexl3's rules (its `plugin/config.zig` `parseBlock`): format `exl3`; codebook `mul1` or `mcg`; window 8 to 16;
/// a routed bank's rate `k` an even number of halfwords per tile in 16..128, a trunk block's rate per linear.
fn checkBlock(o: std.json.ObjectMap, bank: bool) BlockError!void {
    const f = o.get("format") orelse return error.NotExl3;
    if (f != .string or !std.mem.eql(u8, f.string, "exl3")) return error.NotExl3;
    if (o.get("codebook")) |c| {
        if (c != .string or (!std.mem.eql(u8, c.string, "mul1") and !std.mem.eql(u8, c.string, "mcg"))) return error.Exl3CodebookUnsupported;
    } else if (bank) return error.Exl3CodebookUnsupported;
    if (o.get("window")) |w| if (w != .integer or w.integer < 8 or w.integer > 16) return error.Exl3WindowUnsupported;
    if (bank) {
        const k = o.get("k") orelse return error.Exl3RateUnsupported;
        const scaled: f64 = switch (k) {
            .integer => |i| @as(f64, @floatFromInt(i)) * 16.0,
            .float => |x| x * 16.0,
            else => return error.Exl3RateUnsupported,
        };
        const n = @round(scaled);
        if (@abs(scaled - n) > 1e-6 or n < 16 or n > 128 or @mod(n, 2) != 0) return error.Exl3RateUnsupported;
    }
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

const k_packs = [_][]const u8{
    @embedFile("tests/fixtures/k2.json"),
    @embedFile("tests/fixtures/k3.json"),
    @embedFile("tests/fixtures/k4.json"),
    @embedFile("tests/fixtures/k5.json"),
};

fn claimOf(model_type: []const u8, json: []const u8) kit.abi.common.Claim {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return claims(abi.Str.of(model_type), abi.Str.of(json), &dir);
}

fn parseOf(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

/// `json` with the first `old` replaced by `new`.
fn edited(a: std.mem.Allocator, json: []const u8, old: []const u8, new: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, json, old) orelse return error.TestUnexpectedResult;
    return std.mem.concat(a, u8, &.{ json[0..at], new, json[at + old.len ..] });
}

test "turboexl3_qwen38 claims the K2 to K5 packs by marker" {
    for (k_packs) |json| try testing.expectEqual(kit.abi.common.Claim.by_marker, claimOf("qwen4_exp", json));
}

test "turboexl3_qwen38 claims nothing from broken JSON, another model type, or a pack with no trunk block or a null one" {
    for ([_][]const u8{ "{", "[]", "7", "" }) |json| try testing.expectEqual(kit.abi.common.Claim.no, claimOf("qwen4_exp", json));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("llama", k_packs[2]));
    // an MLX-quantized pack: no `trunk_quant` block, so the built-in keeps it
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("qwen4_exp", "{\"model_type\":\"qwen4_exp\",\"quantization\":{\"group_size\":64,\"bits\":4}}"));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("qwen4_exp", "{\"model_type\":\"qwen4_exp\",\"trunk_quant\":null}"));
}

test "turboexl3_qwen38 parses the K4 pack into the built-in's shell, its own slot state, its head as native drafter" {
    var p: abi.Parsed = .{};
    const v = parseOf(k_packs[2], &p);
    try testing.expect(v.isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 248320), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 2560), p.shell.hidden_size);
    try testing.expectEqual(@as(u32, 48), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 24), p.shell.num_attention_heads);
    try testing.expectEqual(@as(u32, 2), p.shell.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), p.shell.head_dim);
    try testing.expectEqual(@as(u32, 10), p.shell.num_experts_per_tok);
    try testing.expect(p.caps.moe and p.caps.batches_decode and p.caps.owns_decode_state and p.caps.slot_state);
    try testing.expect(p.caps.prefix_cache and !p.caps.spec_rollback and !p.caps.images);
    try testing.expectEqual(abi.MtpLayout.none, p.spec.mtp_layout);
    try testing.expectEqual(mtp_depth, p.spec.native_stages);
    try testing.expectEqual(abi.DType.f32, p.shell.state_dtype);
    try testing.expect(of(&p).trunk_format and of(&p).expert_quant);
    try testing.expectEqualStrings("ngram_embedding.safetensors", of(&p).ngramTableFile());
}

test "turboexl3_qwen38 refuses by name a trunk block that is no object and EXL3 blocks turboexl3 does not decode" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const k4 = k_packs[2];
    const trunk = "\"trunk_quant\": {\n    \"format\": \"exl3\",\n    \"codebook\": \"mul1\",\n    \"window\": 16,";
    const cases = [_]struct { json: []const u8, why: []const u8 }{
        .{ .json = try edited(a, k4, trunk, "\"trunk_quant\": \"exl3\", \"x\": {"), .why = "turboexl3_qwen38: TrunkQuantNotAnObject" },
        .{ .json = try edited(a, k4, "\"codebook\": \"mul1\",\n    \"window\": 16,", "\"codebook\": \"3inst\",\n    \"window\": 16,"), .why = "turboexl3_qwen38: Exl3CodebookUnsupported" },
        .{ .json = try edited(a, k4, "\"window\": 16,", "\"window\": 20,"), .why = "turboexl3_qwen38: Exl3WindowUnsupported" },
        .{ .json = try edited(a, k4, "\"format\": \"exl3\",\n    \"codebook\": \"mul1\",\n    \"window\"", "\"format\": \"gptq\",\n    \"codebook\": \"mul1\",\n    \"window\""), .why = "turboexl3_qwen38: NotExl3" },
        .{ .json = try edited(a, k4, "\"k\": 4,", "\"k\": 9.25,"), .why = "turboexl3_qwen38: Exl3RateUnsupported" },
    };
    for (cases) |c| {
        try testing.expectEqual(kit.abi.common.Claim.by_marker, claimOf("qwen4_exp", c.json));
        var p: abi.Parsed = .{};
        const v = parseOf(c.json, &p);
        try testing.expect(!v.isOk() and p.private == null);
        try testing.expectEqualStrings(c.why, v.why.slice());
    }
}

test {
    // The NAX gate's pure parts (the plugin's own mlx-c layer): MLX-free, no GPU.
    _ = @import("src/mlx_serve/mlx/device.zig");
    // The plugin's globals, one per name (turboexl3's `Global` takes them too).
    _ = @import("src/mlx_serve/sdk/util/shared.zig");
    // The prefix cache's snapshot boundary.
    _ = @import("src/cache/aux.zig");
}
