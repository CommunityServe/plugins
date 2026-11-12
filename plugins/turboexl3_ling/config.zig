//! The plugin's claims and parse, MLX-free. A narrow claim: a Ling pack (`bailing_hybrid`) whose config carries
//! exllamav3's `quantization_config` (`quant_method` exl3), which mlx-serve's own bailing_hybrid does not read; the
//! built-in keeps every other Ling pack. The parse is the built-in's (the copy's `plugins/bailing_hybrid/config.zig`),
//! after the pack is checked against what this plugin serves (the mul1 codebook, the streaming repack's expert bank
//! beside the trunk) and Ling-3.1's per-layer clamps are read (src/clamps.zig); the parsed config then streams.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const ling = @import("src/mlx_serve/plugins/bailing_hybrid/config.zig");
const family = @import("src/mlx_serve/plugins/glm5_next/config.zig");
const log = @import("src/mlx_serve/sdk/util/log.zig");
const clamps = @import("src/clamps.zig");
const bill = @import("src/bill.zig");
pub const switches = @import("src/ling_switches.zig");

/// A kit field the measured defaults need: its name, the mlx-serve pull request that added it (both are in mlx-serve
/// 26.11.3) and what a build without it would lose.
pub const KitNeed = struct { field: []const u8, name: []const u8, pr: []const u8, without: []const u8 };

const without_spans = "the layer-major prompt pass (LAYER_MAJOR) and the paths it carries would not run";
const without_last_row = "LAST_ROW_LOGITS would act on nothing";

/// The first field the measured defaults need that a kit's `Forward`, `CacheOps` or `Caps` lacks; null when it has them
/// all.
pub fn kitGap(comptime Forward: type, comptime CacheOps: type, comptime Caps: type) ?KitNeed {
    const needs = .{
        .{ Forward, KitNeed{ .field = "prompt_spans", .name = "Forward.prompt_spans", .pr = "1126", .without = without_spans } },
        .{ Forward, KitNeed{ .field = "ssm_checkpoints", .name = "Forward.ssm_checkpoints", .pr = "1126", .without = without_spans } },
        .{ CacheOps, KitNeed{ .field = "checkpoint", .name = "CacheOps.checkpoint", .pr = "1126", .without = without_spans } },
        .{ Caps, KitNeed{ .field = "prompt_last_row", .name = "Caps.prompt_last_row", .pr = "1133", .without = without_last_row } },
    };
    inline for (needs) |n| if (!@hasField(n[0], n[1].field)) return n[1];
    return null;
}

// A build against a kit without them stops here and names the field, instead of serving without the paths they
// decide. Every build then has both: the plugin declares the whole-prompt prefill and the last-row logits (`served`),
// and plugin.zig still refuses by name a serving mlx-serve older than its kit (`refuseNoSpans`).
comptime {
    // comptimePrint's formatting runs at comptime: room for it, so the error is the one below and not the quota's.
    @setEvalBranchQuota(10_000);
    if (kitGap(abi.Forward, abi.CacheOps, abi.Caps)) |g| @compileError(std.fmt.comptimePrint(
        "turboexl3_ling: this plugin kit has no {s} (ddalcu/mlx-serve#{s}), so {s}. Build against the mlx-serve 26.11.3 " ++
            "kit that KIT_VERSION pins, or point MLX_SERVE_KIT at a kit from mlx-serve 26.11.3 or later.",
        .{ g.name, g.pr, g.without },
    ));
}

pub const model_types = ling.model_types;
pub const Config = family.Config;
pub const of = ling.of;
pub const freeParsed = ling.freeParsed;

const gpa = std.heap.c_allocator;

pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) kit.abi.common.Claim {
    _ = dir;
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch return .no;
    return if (isExl3(doc)) .by_marker else .no;
}

fn isExl3(doc: std.json.Value) bool {
    if (doc != .object) return false;
    const qc = doc.object.get("quantization_config") orelse return false;
    if (qc != .object) return false;
    const m = qc.object.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "exl3");
}

pub const PackError = error{ Exl3CodebookUnsupported, Exl3PackNotRepacked, Exl3QuantConfigNotAnObject, LingClampListLength, LingClampValue, LingLayerCount };

/// The bank files the copied streamer opens (the v2 manifest for the records, the v1 one for the runtime).
const bank_files = [_][]const u8{ "experts.bin", "expert-manifest-v2.json", "expert-manifest.json" };

/// The pack against what the plugin serves, and its clamps read. JSON that does not parse is the built-in parse's to
/// refuse.
pub fn checkPack(dir: *const abi.PackDir, json: []const u8) PackError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}) catch return;
    if (doc != .object) return;
    const qc = doc.object.get("quantization_config") orelse return error.Exl3QuantConfigNotAnObject;
    if (qc != .object) return error.Exl3QuantConfigNotAnObject;
    // exllamav3 names its codebook; turboexl3 and the bank read mul1 only.
    const cb = qc.object.get("codebook") orelse return error.Exl3CodebookUnsupported;
    if (cb != .string or !std.mem.eql(u8, cb.string, "mul1")) return error.Exl3CodebookUnsupported;
    for (bank_files) |f| if (!dir.has(f)) return error.Exl3PackNotRepacked;
    const nl = doc.object.get("num_hidden_layers") orelse return error.LingLayerCount;
    if (nl != .integer or nl.integer < 1 or nl.integer > clamps.max_layers) return error.LingLayerCount;
    try clamps.read(doc.object, @intCast(nl.integer));
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    checkPack(dir, config_json.slice()) catch |e| {
        log.err("[turboexl3_ling] the pack is refused: {s}\n", .{@errorName(e)});
        return switch (e) {
            inline else => |x| abi.Verdict.refuse(.unsupported, "turboexl3_ling: " ++ @errorName(x)),
        };
    };
    if (span != null) return abi.Verdict.refuse(.unsupported, "turboexl3_ling: no Thunderbolt pair (a streamed bank serves one Mac)");
    const v = ling.parse(dir, config_json, settings_json, span, stub, out);
    if (v.isOk()) served(out);
    return v;
}

/// The built-in's facts as this plugin serves them: the routed experts stream from the bank (one request decodes at a
/// time), the routed experts' weights are not the loader's, and no drafter but the MTP head reads the trunk's table.
fn served(p: *abi.Parsed) void {
    const c: *Config = @constCast(of(p));
    c.streamed = true;
    p.caps.streamed_weights = true;
    p.caps.batches_decode = false;
    // No DFlash layer capture, so mlx-serve declines any drafter at load (a `--drafter`, or one in the pack's folder):
    // 66aad949's `Bound.drafters` is the cap and the trunk's weights (src/engines/mlx/plugin/driver.zig:336), and
    // without it the load names the drafter in the log and runs without it (src/engines/mlx/load.zig:321-323). A DFlash drafter takes its rows
    // from the embedding table the trunk lends (src/engines/mlx/spec/dflash.zig:44-47), which EMBED_ROWS leaves unread
    // (src/ling_embed.zig): 1,228 MiB that the bill does not hold. The forward fills no capture either: the layer-major
    // pass refuses one (plugin.zig `spanMajorOr`, `promptCalls`). The MTP head still borrows the table and head.
    p.caps.dflash_capture = false;
    // C1: an auto-sized context never passes the context the bill holds when the load names none (src/bill.zig
    // `default_context`). `--ctx-size` sets any (mlx-serve's manual context wins over this cap), and the load's
    // settings name it, so the bill holds it (src/bill.zig `contextOf`).
    p.shell.context_cap = @min(p.shell.context_cap, bill.default_context);
    // The exact verify (plugin.zig `forward`): a verify row's bytes are serial decode's.
    p.caps.exact_drafted = true;
    // The layer-major prompt pass (plugin.zig `spanMajor`): the whole prompt a forward, unless the parity instrument
    // turns it off (src/ling_switches.zig `layerMajor`).
    if (switches.layerMajor()) p.caps.whole_prompt_prefill = true;
    // The last-row prompt logits (plugin.zig `spanMajorOr`): a prompt the engine reads only at its last row is asked that
    // row alone (`Caps.prompt_last_row`), so the head runs over the last span only.
    p.caps.prompt_last_row = true;
    p.cache.layout_namespace = abi.Str.of(layoutNamespace());
    // The NextN (MTP) head (src/ling_mtp.zig): in the checkpoint, reading the trunk's final-normed hidden, one draft a
    // round by default (exllamav3's draft size; --mtp-depth sets another).
    p.spec.mtp_layout = .in_checkpoint;
    p.spec.mtp_default_depth = 1;
    p.spec.head_reads_final_norm = true;
}

/// The SSD root's layout namespace of the served MLA cache, the rope-once layout with the MTP head's latent cache
/// (src/ling_switches.zig `layoutNamespace`, where the restore check's part A reads it too): an entry persisted under
/// another layout never restores into it.
pub const layoutNamespace = switches.layoutNamespace;

// ── tests ──

const testing = std.testing;

const MemDir = kit.sdk.fixture.MemDir;

test "turboexl3_ling claims: a bailing_hybrid pack with an exl3 quantization_config by marker, nothing else" {
    const mem: MemDir = .{ .files = &.{} };
    const empty = mem.packDir();
    try testing.expectEqual(kit.abi.common.Claim.by_marker, claims(.of("bailing_hybrid"), .of("{\"quantization_config\":{\"quant_method\":\"exl3\"}}"), &empty));
    try testing.expectEqual(kit.abi.common.Claim.no, claims(.of("bailing_hybrid"), .of("{\"quantization\":{\"bits\":4}}"), &empty));
    try testing.expectEqual(kit.abi.common.Claim.no, claims(.of("bailing_hybrid"), .of("{\"quantization_config\":{\"quant_method\":\"fp8\"}}"), &empty));
    try testing.expectEqual(kit.abi.common.Claim.no, claims(.of("deepseek_v41"), .of("{\"quantization_config\":{\"quant_method\":\"exl3\"}}"), &empty));
    try testing.expectEqual(kit.abi.common.Claim.no, claims(.of("bailing_hybrid"), .of("not json"), &empty));
}

test "turboexl3_ling pack: the 2.60 config parses with its clamps; a pack without the bank or with another codebook is refused by name" {
    const json = @embedFile("tests/fixtures/config-2.60bpw.json");
    const with_bank: MemDir = .{ .files = &.{ .{ .name = "experts.bin", .bytes = "" }, .{ .name = "expert-manifest-v2.json", .bytes = "" }, .{ .name = "expert-manifest.json", .bytes = "" } } };
    const dir = with_bank.packDir();
    try checkPack(&dir, json);
    try testing.expectEqual(@as(u32, 0), clamps.routed(64));
    try testing.expectEqual(@as(u32, 10), clamps.routed(65));
    try testing.expectEqual(@as(u32, 10), clamps.shared(64));
    try testing.expectEqual(@as(u32, 0), clamps.shared(63));
    const without: MemDir = .{ .files = &.{.{ .name = "experts.bin", .bytes = "" }} };
    const bare = without.packDir();
    try testing.expectError(error.Exl3PackNotRepacked, checkPack(&bare, json));
    try testing.expectError(error.Exl3CodebookUnsupported, checkPack(&dir, "{\"quantization_config\":{\"quant_method\":\"exl3\",\"codebook\":\"mcg\"}}"));
}

test "turboexl3_ling kit: a kit without #1126's span fields or #1133's last-row flag is named by the first field it lacks; this kit has them all" {
    const Old = struct { rows: u32 = 0 };
    const OldCaps = struct { exact_drafted: bool = false };
    const HalfSpans = struct { prompt_spans: u8 = 0 };
    const Spans = struct { prompt_spans: u8 = 0, ssm_checkpoints: u8 = 0 };
    const Ops = struct { checkpoint: u8 = 0 };
    try testing.expectEqualStrings("Forward.prompt_spans", kitGap(Old, Ops, OldCaps).?.name);
    try testing.expectEqualStrings("Forward.ssm_checkpoints", kitGap(HalfSpans, Ops, OldCaps).?.name);
    try testing.expectEqualStrings("CacheOps.checkpoint", kitGap(Spans, Old, OldCaps).?.name);
    const last = kitGap(Spans, Ops, OldCaps).?;
    try testing.expectEqualStrings("Caps.prompt_last_row", last.name);
    try testing.expectEqualStrings("1133", last.pr);
    try testing.expect(kitGap(abi.Forward, abi.CacheOps, abi.Caps) == null);
}

test "turboexl3_ling caps: the built-in parse asks for DFlash's layer capture, and the plugin's parse clears it with the layer-major pass on and at LAYER_MAJOR=0, so mlx-serve declines any drafter at load, names it in the log and runs without it, and none reads the embedding table EMBED_ROWS leaves unread" {
    const json = @embedFile("tests/fixtures/config-2.60bpw.json");
    const with_bank: MemDir = .{ .files = &.{ .{ .name = "experts.bin", .bytes = "" }, .{ .name = "expert-manifest-v2.json", .bytes = "" }, .{ .name = "expert-manifest.json", .bytes = "" } } };
    const dir = with_bank.packDir();
    // the built-in's facts (the copied bailing_hybrid config.zig `lingFacts`)
    var built_in: abi.Parsed = .{};
    try testing.expect(ling.parse(&dir, .of(json), .{}, null, true, &built_in).isOk());
    defer freeParsed(&built_in);
    try testing.expect(built_in.caps.dflash_capture);
    defer switches.setLayerMajorOffForTests(null);
    for ([_]bool{ false, true }) |off| {
        switches.setLayerMajorOffForTests(off);
        var p: abi.Parsed = .{};
        try testing.expect(parse(&dir, .of(json), .{}, null, true, &p).isOk());
        defer freeParsed(&p);
        // the layer-major pass follows the switch, and the capture stays off at either value
        try testing.expectEqual(!off, p.caps.whole_prompt_prefill);
        try testing.expect(!p.caps.dflash_capture);
    }
}

test "turboexl3_ling context: an auto-sized context is capped at the bill's default context, 32,768 tokens of the 2.60 pack's 262,144 (C1: --ctx-size sets any other, which the load's settings name and the bill holds)" {
    const json = @embedFile("tests/fixtures/config-2.60bpw.json");
    const with_bank: MemDir = .{ .files = &.{ .{ .name = "experts.bin", .bytes = "" }, .{ .name = "expert-manifest-v2.json", .bytes = "" }, .{ .name = "expert-manifest.json", .bytes = "" } } };
    const dir = with_bank.packDir();
    var built_in: abi.Parsed = .{};
    try testing.expect(ling.parse(&dir, .of(json), .{}, null, true, &built_in).isOk());
    defer freeParsed(&built_in);
    try testing.expectEqual(@as(u64, 262144), built_in.shell.context_cap);
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(bill.default_context, p.shell.context_cap);
}

test {
    _ = clamps;
    // KV_SCHEME_DENSE_COPIES' decision, MLX-free (src/ling_kv_scheme.zig).
    _ = @import("src/ling_kv_scheme.zig");
    // C6's refusal, MLX-free (src/ling_rope1_spike.zig): plugin.zig, which serves it, is the dylib's root, not a test
    // root, so its tests run here.
    _ = @import("src/ling_rope1_spike.zig");
}
