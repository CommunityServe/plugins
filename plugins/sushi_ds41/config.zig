//! The plugin's claims and parse. A narrow claim: `model_type` deepseek_v41 AND an `expert_quant` format this plugin
//! loads, so the built-in plugin keeps every other V4.1 pack. Parsing is sushi's own (`dsv41_config.zig`).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const sc = @import("src/sushi/dsv41_config.zig");
const exl3 = @import("sushi_exl3");

pub const model_types = [_][]const u8{"deepseek_v41"};

/// `expert_quant.format` values this plugin loads; a new quant arm is a loader plus an entry here.
const formats = [_][]const u8{"stq"};

const gpa = std.heap.c_allocator;

/// What `bill` and `load` read.
pub const Private = struct { cfg: sc.Config, exl3_n: u32, exl3_rates: ?u128, dspark: bool };

pub fn private(p: *const abi.Parsed) *const Private {
    return @ptrCast(@alignCast(p.private.?));
}

pub fn of(p: *const abi.Parsed) *const sc.Config {
    return &private(p).cfg;
}

fn root(tree: *const std.json.Parsed(std.json.Value)) ?std.json.ObjectMap {
    return if (tree.value == .object) tree.value.object else null;
}

fn supportedFormat(o: std.json.ObjectMap) bool {
    const block = o.get("expert_quant") orelse return false;
    if (block != .object) return false;
    const format = block.object.get("format") orelse return false;
    if (format != .string) return false;
    for (formats) |f| if (std.mem.eql(u8, f, format.string)) return true;
    return false;
}

pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) kit.abi.common.Claim {
    _ = dir;
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    const tree = std.json.parseFromSlice(std.json.Value, gpa, config_json.slice(), .{}) catch return .no;
    defer tree.deinit();
    return if (root(&tree)) |o| (if (supportedFormat(o)) .by_marker else .no) else .no;
}

fn refuse(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "sushi_ds41: " ++ why);
}

fn eosIds(o: std.json.ObjectMap, shell: *abi.Shell) void {
    const v = o.get("eos_token_id") orelse return;
    switch (v) {
        .integer => |i| if (i >= 0) {
            shell.eos[0] = @intCast(i);
            shell.eos_count = 1;
        },
        .array => |a| for (a.items) |item| if (item == .integer and item.integer >= 0 and shell.eos_count < abi.max_eos) {
            shell.eos[shell.eos_count] = @intCast(item.integer);
            shell.eos_count += 1;
        },
        else => {},
    }
}

/// The context the rope serves: the YaRN-extended original window, capped by the declared one.
fn contextCap(c: *const sc.Config) u64 {
    const scaled: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(c.yarn_orig)) * c.yarn_factor));
    return if (c.max_pos == 0) scaled else @min(scaled, c.max_pos);
}

/// The per-model `mtp` setting: `false` keeps the DSpark stages off; they load by default when the pack ships them.
fn mtpAllowed(settings_json: []const u8) bool {
    if (settings_json.len == 0) return true;
    const S = struct { mtp: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return true;
    defer parsed.deinit();
    return parsed.value.mtp orelse true;
}

/// Whether the pack's weight index names the DSpark stages' tensors.
fn shipsStages(dir: *const abi.PackDir) bool {
    const buf = gpa.alloc(u8, 8 << 20) catch return false;
    defer gpa.free(buf);
    var why: abi.Verdict = .{};
    const raw = dir.readFile("model.safetensors.index.json", buf, &why) orelse return false;
    return std.mem.indexOf(u8, raw, "\"mtp.0.main_proj.weight\"") != null or std.mem.indexOf(u8, raw, "\"mtp.0.main_proj.trellis\"") != null;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    if (span != null) return abi.Verdict.refuse(.unsupported, "sushi_ds41: no layer split (a Thunderbolt pair is not supported)");
    const tree = std.json.parseFromSlice(std.json.Value, gpa, config_json.slice(), .{}) catch return refuse("config.json does not parse");
    defer tree.deinit();
    const o = root(&tree) orelse return refuse("config.json is not an object");
    if (!supportedFormat(o)) return abi.Verdict.refuse(.unsupported, "sushi_ds41: expert_quant.format is not one this plugin loads (stq)");
    const text = if (o.get("text_config")) |t| (if (t == .object) t.object else return refuse("text_config is not an object")) else o;
    const priv = gpa.create(Private) catch return refuse("out of memory");
    errdefer gpa.destroy(priv);
    priv.cfg = sc.parse(text, o) catch return refuse("config.json is not a DeepSeek-V4.1 geometry sushi serves");
    if (priv.cfg.expert_dec == null or priv.cfg.trunk_dec == null) return refuse("an STQ pack names both expert_quant and trunk_quant");
    const experts = exl3.parseQuantBlock(o.get("expert_quant").?) catch return refuse("expert_quant does not parse");
    priv.exl3_n = experts.rate.n;
    priv.exl3_rates = experts.rates;
    priv.dspark = priv.cfg.dspark.block > 0 and priv.cfg.n_mtp > 0 and mtpAllowed(settings_json.slice()) and shipsStages(dir);
    const c = &priv.cfg;
    var p: abi.Parsed = .{
        .caps = .{ .owns_decode_state = true, .prefix_cache = true, .moe = true, .long_ctx = true, .images = c.vision != null, .streamed_weights = true },
        .chat = .{ .thinking_default = .on, .effort_vocab = .dsv41, .template_family = abi.Str.of(model_types[0]) },
        .spec = .{ .native_stages = if (priv.dspark) c.n_mtp else 0, .native_block = if (priv.dspark) c.dspark.block else 0, .policy = .{ .drafter_ok = false } },
        .placeholder_ids = .{ .image = if (c.vision != null) c.image_token_id else 0 },
        .private = priv,
    };
    p.shell = .{
        .vocab_size = c.vocab,
        .hidden_size = c.dim,
        .intermediate_size = c.moe_inter,
        .num_hidden_layers = c.layers,
        .num_attention_heads = c.heads,
        .num_key_value_heads = 1,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.top_k,
        .moe_intermediate_size = c.moe_inter,
        .max_position_embeddings = c.max_pos,
        .context_cap = contextCap(c),
        .bos = if (o.get("bos_token_id")) |b| (if (b == .integer) b.integer else -1) else -1,
        .act_dtype = .bf16,
        // Not the built-in's name: the host sizes `deepseek_v41` from its own config fields, which an outside plugin cannot fill.
        .model_type = abi.Str.of("sushi_deepseek_v41"),
    };
    eosIds(o, &p.shell);
    if (!stub) kit.generation.apply(dir, &p.shell);
    out.* = p;
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |c| gpa.destroy(@as(*Private, @ptrCast(@alignCast(c))));
    p.private = null;
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

const stq = @embedFile("tests/stq.json");
const iq = @embedFile("tests/iq-mlx.json");

fn claimOf(json: []const u8) kit.abi.common.Claim {
    const mem: fixture.MemDir = .{ .files = &.{} };
    const dir = mem.packDir();
    return claims(abi.Str.of(model_types[0]), abi.Str.of(json), &dir);
}

test "sushi_ds41 claims an STQ pack by marker and leaves the release and iQ-MLX packs to the built-in" {
    try testing.expectEqual(kit.abi.common.Claim.by_marker, claimOf(stq));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf(iq));
    try testing.expectEqual(kit.abi.common.Claim.no, claimOf("{"));
}

test "sushi_ds41 parses the STQ pack's geometry" {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = stq }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(stq), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 40), p.shell.num_hidden_layers);
    try testing.expect(p.caps.owns_decode_state and p.caps.streamed_weights);
    // The bill's arithmetic runs unchecked in a release build: Debug traps an overflow.
    const c = of(&p);
    for ([_]u64{ 1, 256, 2048, 8192 }) |chunk| {
        for ([_]u64{ 1, 4096, 131072, 1 << 20 }) |seq| try testing.expect(c.transientBytes(chunk, seq) > 0 and c.decodeTransientBytes(chunk, seq) > 0);
    }
    // The mini pack names no stage tensors in its (absent) index: no native drafter.
    try testing.expectEqual(@as(u32, 0), p.spec.native_stages);
}

test "sushi_ds41 arms its native drafter only for a pack that ships the stages and a model that allows them" {
    const index = "{\"weight_map\":{\"mtp.0.main_proj.trellis\":\"model-mtp-0.safetensors\"}}";
    const mem: fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = stq }, .{ .name = "model.safetensors.index.json", .bytes = index } } };
    const dir = mem.packDir();
    var on: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(stq), .{}, null, true, &on).isOk());
    defer freeParsed(&on);
    try testing.expectEqual(@as(u32, 3), on.spec.native_stages);
    try testing.expectEqual(@as(u32, 5), on.spec.native_block);
    var off: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(stq), abi.Str.of("{\"mtp\":false}"), null, true, &off).isOk());
    defer freeParsed(&off);
    try testing.expectEqual(@as(u32, 0), off.spec.native_stages);
}
