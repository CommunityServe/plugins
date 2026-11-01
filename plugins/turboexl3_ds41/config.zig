//! The plugin's claims and parse, MLX-free. A narrow claim: a DeepSeek-V4.1 pack (`deepseek_v41`) whose config carries
//! exllamav3's `quantization_config` (`quant_method` exl3), which mlx-serve's own deepseek_v41 does not read; the built-in
//! keeps every other V4.1 pack. The parse is the built-in's (the copy's `plugins/deepseek_v41/config.zig`), after the
//! pack is checked against what this plugin serves: the mul1 codebook, and the streaming repack's expert bank beside the
//! trunk (`experts.bin` and its two manifests).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const ds = @import("src/mlx_serve/plugins/deepseek_v41/config.zig");
const log = @import("src/mlx_serve/sdk/util/log.zig");

pub const model_types = ds.model_types;
pub const Config = ds.Config;
pub const of = ds.of;
pub const freeParsed = ds.freeParsed;

const gpa = std.heap.c_allocator;

pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) kit.abi.common.Claim {
    _ = dir;
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch return .no;
    if (doc != .object) return .no;
    const qc = doc.object.get("quantization_config") orelse return .no;
    if (qc != .object) return .no;
    const m = qc.object.get("quant_method") orelse return .no;
    return if (m == .string and std.mem.eql(u8, m.string, "exl3")) .by_marker else .no;
}

const PackError = error{ Exl3CodebookUnsupported, Exl3PackNotRepacked, Exl3QuantConfigNotAnObject };

/// The bank files the copied streamer opens (`expert/bank.zig`: the v2 manifest for the records, the v1 one for the
/// runtime), all three at the pack's root.
const bank_files = [_][]const u8{ "experts.bin", "expert-manifest-v2.json", "expert-manifest.json" };

/// The pack against what the plugin serves. JSON that does not parse is the built-in parse's to refuse.
fn checkPack(dir: *const abi.PackDir, json: []const u8) PackError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}) catch return;
    if (doc != .object) return;
    const qc = doc.object.get("quantization_config") orelse return error.Exl3QuantConfigNotAnObject;
    if (qc != .object) return error.Exl3QuantConfigNotAnObject;
    // exllamav3 names its codebook; turboexl3 and the streamer's bank kernels both decode mul1.
    const cb = qc.object.get("codebook") orelse return error.Exl3CodebookUnsupported;
    if (cb != .string or !std.mem.eql(u8, cb.string, "mul1")) return error.Exl3CodebookUnsupported;
    for (bank_files) |f| if (!dir.has(f)) return error.Exl3PackNotRepacked;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    checkPack(dir, config_json.slice()) catch |e| {
        log.err("[turboexl3_ds41] the pack is refused: {s}\n", .{@errorName(e)});
        return switch (e) {
            inline else => |x| abi.Verdict.refuse(.unsupported, "turboexl3_ds41: " ++ @errorName(x)),
        };
    };
    if (span != null) return abi.Verdict.refuse(.unsupported, "turboexl3_ds41: no Thunderbolt pair (a streamed bank serves one Mac)");
    const v = ds.parse(dir, config_json, settings_json, span, stub, out);
    if (v.isOk()) served(out);
    return v;
}

/// `MLX_SERVE_DSV41_ROUTED=turboexl3` (the server's environment: mlx-serve hands a plugin's load only the settings it
/// knows): the streamed bank's experts run on turboexl3 (src/routed_t3.zig) instead of the streamer's own EXL3 kernels.
pub fn routedT3(value: ?[]const u8) bool {
    const v = value orelse return false;
    return std.mem.eql(u8, v, "turboexl3");
}

/// `MLX_SERVE_DSV41_DSPARK_COPY` and `MLX_SERVE_DSV41_DSPARK_POLICY` (the server's environment, as `routedT3`): the
/// drafter's copy rounds and verify policy (src/rounds.zig), on unless the value starts with `0`.
pub fn switchOn(value: ?[]const u8) bool {
    const v = value orelse return true;
    return v.len == 0 or v[0] != '0';
}

/// The built-in's facts as this plugin serves them: the bank streams (the copy's parse saw `experts.bin`), the state is
/// the module's own, the host loads none of the shards (`weights.keep_key` refuses every key: the plugin reads them
/// itself), and the DSpark stages draft as the model's native drafter.
fn served(p: *abi.Parsed) void {
    p.caps.owns_decode_state = true;
    p.caps.cluster_plan = false;
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

const sc281 = @embedFile("tests/fixtures/config.json");
const no_bank = [_]fixture.File{.{ .name = "config.json", .bytes = sc281 }};
const repacked = [_]fixture.File{
    .{ .name = "config.json", .bytes = sc281 },
    .{ .name = "experts.bin", .bytes = "" },
    .{ .name = "expert-manifest-v2.json", .bytes = "{}" },
    .{ .name = "expert-manifest.json", .bytes = "{}" },
};

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

test "turboexl3_ds41 claims the SC 2.81 pack by marker, and nothing without an exl3 quantization_config" {
    try testing.expectEqual(kit.abi.common.Claim.by_marker, claimOf("deepseek_v41", sc281));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("deepseek_v4", sc281));
    for ([_][]const u8{ "{", "[]", "7", "" }) |json| try testing.expectEqual(kit.abi.common.Claim.no, claimOf("deepseek_v41", json));
    // an MLX pack and the in-tree repack of one: no exllamav3 block, so the built-in keeps them
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("deepseek_v41", "{\"model_type\":\"deepseek_v41\",\"quantization\":{\"group_size\":32,\"bits\":4}}"));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("deepseek_v41", "{\"model_type\":\"deepseek_v41\",\"quantization_config\":{\"quant_method\":\"fp8\"}}"));
}

test "turboexl3_ds41 parses the repacked pack into the built-in's streamed shell with its DSpark stages" {
    var p: abi.Parsed = .{};
    const v = parseOf(&repacked, sc281, &p);
    try testing.expect(v.isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 129280), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 5120), p.shell.hidden_size);
    try testing.expectEqual(@as(u32, 40), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 6), p.shell.num_experts_per_tok);
    try testing.expect(p.caps.owns_decode_state and p.caps.streamed_weights and p.caps.whole_prompt_prefill);
    try testing.expect(!p.caps.prefix_cache and !p.caps.slot_state and !p.caps.cluster_plan);
    try testing.expectEqual(@as(u32, 3), p.spec.native_stages);
    try testing.expectEqual(@as(u32, 5), p.spec.native_block);
    const c = of(&p);
    try testing.expect(c.dsv41_streamed and c.dsv41_exl3_bank and c.dsv41_vision);
    try testing.expectEqual(@as(u32, 2), c.dsv41_n_engram_layers);
    try testing.expectEqual(@as(u32, 128), c.dsv41_dspark_experts);
}

test "turboexl3_ds41 refuses by name the pack before its repack and a codebook it does not decode" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const mcg = try edited(a, sc281, "\"codebook\": \"mul1\"", "\"codebook\": \"mcg\"");
    const cases = [_]struct { files: []const fixture.File, json: []const u8, why: []const u8 }{
        .{ .files = &no_bank, .json = sc281, .why = "turboexl3_ds41: Exl3PackNotRepacked" },
        .{ .files = &repacked, .json = mcg, .why = "turboexl3_ds41: Exl3CodebookUnsupported" },
    };
    for (cases) |c| {
        var p: abi.Parsed = .{};
        const v = parseOf(c.files, c.json, &p);
        try testing.expect(!v.isOk() and p.private == null);
        try testing.expectEqualStrings(c.why, v.why.slice());
    }
}

test "turboexl3_ds41 runs the routed experts on turboexl3 only when MLX_SERVE_DSV41_ROUTED names it" {
    try testing.expect(routedT3("turboexl3"));
    for ([_]?[]const u8{ null, "", "0", "streamer", "turboexl3 " }) |v| try testing.expect(!routedT3(v));
}

test "turboexl3_ds41 keeps its DSpark copy rounds and verify policy on unless the setting starts with 0" {
    for ([_]?[]const u8{ null, "", "1", "on" }) |v| try testing.expect(switchOn(v));
    for ([_][]const u8{ "0", "00", "0n" }) |v| try testing.expect(!switchOn(v));
}

test {
    _ = @import("src/rounds.zig");
    _ = @import("src/stage_prof.zig");
    _ = @import("src/one_plan.zig");
    _ = @import("src/markov_plan.zig");
    _ = @import("src/prompt_io.zig");
    _ = @import("src/dec_prefetch.zig");
    _ = @import("src/round_io.zig");
    _ = @import("src/stream_lookahead.zig");
    _ = @import("src/expert_usage_cfg.zig");
    _ = @import("src/route_rec.zig");
    _ = @import("src/decode_rows.zig");
    _ = @import("src/seed_free.zig");
    _ = @import("src/ra_fused.zig");
    _ = @import("src/woa_rows.zig");
    _ = @import("src/verify_tl.zig");
    _ = @import("src/late_gemv.zig");
    _ = @import("src/shared_hoist.zig");
}
