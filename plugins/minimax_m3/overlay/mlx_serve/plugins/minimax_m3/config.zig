//! copied_from: src/plugins/minimax_m3/config.zig@2d42b6c7e8b4b0b0b1372789c425b966d73a9dbd
//! MiniMax M3.1 Flash's prepared text pack claim and host facts.
const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const geometry = @import("geometry.zig");
const residents = @import("resident_layout.zig");
pub const plugin = @import("mlx_serve").sdk.plugin.Plugin{
    .name = "minimax_m3",
    .kind = .model,
    .claims = &model_types,
};
pub const model_types = [_][]const u8{ "minimax_m3_vl", "minimax_m3", "minimax_m3_text" };
pub const BankFormat = enum { nvfp4, affine, bf16 };
pub const Config = struct {
    model: geometry.Config,
    bank_format: BankFormat,
    residents_format: residents.Format,
    settings_json: []u8,
};
const gpa = std.heap.c_allocator;
fn typeMatches(model_type: []const u8) bool {
    for (model_types) |t| if (std.mem.eql(u8, model_type, t)) return true;
    return false;
}
fn bankFormat(root: std.json.ObjectMap) ?BankFormat {
    const es = root.get("expert_streaming") orelse return null;
    if (es != .object) return null;
    const family = es.object.get("family") orelse return null;
    const manifest = es.object.get("manifest") orelse return null;
    const format = es.object.get("format") orelse return null;
    if (family != .string or !std.mem.eql(u8, family.string, "mlx-stream") or
        manifest != .string or !std.mem.eql(u8, manifest.string, "expert-manifest.json") or format != .string) return null;
    if (std.mem.eql(u8, format.string, "mlx-stream-nvfp4-expert-bank-v1")) return .nvfp4;
    if (std.mem.eql(u8, format.string, "mlx-stream-affine-expert-bank-v1")) return .affine;
    if (std.mem.eql(u8, format.string, "mlx-stream-bf16-expert-bank-v1")) return .bf16;
    return null;
}
fn settingsJson(root: std.json.ObjectMap, overrides: []const u8) ![]u8 {
    const supplied = try std.json.parseFromSlice(std.json.Value, gpa, if (overrides.len == 0) "null" else overrides, .{});
    defer supplied.deinit();
    const prepared = root.get("expert_streaming").?.object.get("settings") orelse .null;
    var merged: std.json.ObjectMap = .empty;
    defer merged.deinit(gpa);
    for ([_]std.json.Value{ prepared, supplied.value }) |settings| {
        if (settings == .null) continue;
        if (settings != .object) return error.MiniMaxSettingsMustBeObject;
        for (settings.object.keys(), settings.object.values()) |key, value| {
            if (value != .null) try merged.put(gpa, key, value);
        }
    }
    return std.json.Stringify.valueAlloc(gpa, std.json.Value{ .object = merged }, .{});
}
pub fn claims(model_type: abi.Str, config_json: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!typeMatches(model_type.slice())) return .no;
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json.slice(), .{}) catch return .no;
    defer parsed.deinit();
    return if (parsed.value == .object and bankFormat(parsed.value.object) != null) .by_marker else .no;
}
pub fn parse(_: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, _: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    if (span) |s| if (s.first != 0 or (s.count != 0 and s.count != geometry.max_layers))
        return abi.Verdict.refuse(.unsupported, "minimax_m3: distributed layer spans are unsupported");
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "minimax_m3: invalid config.json");
    defer parsed.deinit();
    if (parsed.value != .object) return abi.Verdict.refuse(.failed, "minimax_m3: config.json must be an object");
    const format = bankFormat(parsed.value.object) orelse
        return abi.Verdict.refuse(.unsupported, "minimax_m3: prepared NVFP4 or affine expert_streaming bank required");
    var diag: geometry.Diag = .{};
    const model = geometry.Config.parse(gpa, config_json.slice(), &diag) catch
        return if (diag.field.len > 0) abi.Verdict.named(.unsupported, diag.field) else abi.Verdict.refuse(.unsupported, "minimax_m3: unsupported M3.1 Flash text geometry");
    const resident_format = residents.Format.parse(parsed.value.object.get("residents_format") orelse .{ .string = "bf16" }) catch
        return abi.Verdict.refuse(.unsupported, "minimax_m3: residents_format must be bf16, mxfp8, or mixed");
    const settings = settingsJson(parsed.value.object, settings_json.slice()) catch |e|
        return abi.Verdict.named(.failed, @errorName(e));
    const cfg = gpa.create(Config) catch {
        gpa.free(settings);
        return abi.Verdict.refuse(.failed, "minimax_m3: out of memory");
    };
    cfg.* = .{ .model = model, .bank_format = format, .residents_format = resident_format, .settings_json = settings };
    out.* = .{
        .caps = .{ .owns_decode_state = true, .prefix_cache = false, .moe = true, .whole_prompt_prefill = true, .streamed_weights = true },
        .chat = .{ .template_family = abi.Str.of("minimax_m3") },
        .shell = .{
            .vocab_size = model.vocab_size,
            .hidden_size = model.hidden_size,
            .intermediate_size = model.intermediate_size,
            .num_hidden_layers = model.n_layers,
            .num_attention_heads = model.n_heads,
            .num_key_value_heads = model.n_kv_heads,
            .head_dim = model.head_dim,
            .num_experts_per_tok = model.n_experts_per_tok,
            .moe_intermediate_size = model.intermediate_size,
            .max_position_embeddings = model.max_position_embeddings,
            .context_cap = @min(model.max_position_embeddings, geometry.default_context),
            .act_dtype = .bf16,
            .model_type = abi.Str.of("minimax_m3_vl"),
        },
        .private = cfg,
    };
    return .ok;
}
pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |raw| {
        const cfg: *Config = @ptrCast(@alignCast(raw));
        gpa.free(cfg.settings_json);
        gpa.destroy(cfg);
    }
    p.private = null;
}
pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}
test "MiniMax claims only prepared expert banks and refuses unsupported formats" {
    const t = std.testing;
    const src = @embedFile("config_fixture.json");
    const marked = try std.mem.replaceOwned(u8, t.allocator, src, "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},");
    defer t.allocator.free(marked);
    const dir: abi.PackDir = undefined;
    try t.expectEqual(abi.Claim.no, claims(abi.Str.of("minimax_m3_vl"), abi.Str.of(src), &dir));
    try t.expectEqual(abi.Claim.by_marker, claims(abi.Str.of("minimax_m3_vl"), abi.Str.of(marked), &dir));
}
test "MiniMax parse names the geometry field it refuses" {
    const a = std.testing.allocator;
    const marked = try std.mem.replaceOwned(u8, a, @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},");
    defer a.free(marked);
    const bad = try std.mem.replaceOwned(u8, a, marked, "\"head_dim\":128", "\"head_dim\":64");
    defer a.free(bad);
    try std.testing.expect(!std.mem.eql(u8, bad, marked));
    const dir: abi.PackDir = undefined;
    var p: abi.Parsed = .{};
    const v = parse(&dir, .of(bad), .{}, null, false, &p);
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
    try std.testing.expectEqualStrings("head_dim", v.why.slice());
}
test "MiniMax parse admits MXFP8 residents and refuses an unknown resident format" {
    const a = std.testing.allocator;
    const marked = try std.mem.replaceOwned(u8, a, @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},\"residents_format\":\"mxfp8\",");
    defer a.free(marked);
    const dir: abi.PackDir = undefined;
    var p: abi.Parsed = .{};
    try std.testing.expect(parse(&dir, .of(marked), .{}, null, false, &p).isOk());
    try std.testing.expectEqual(residents.Format.mxfp8, of(&p).residents_format);
    freeParsed(&p);
    const bad = try std.mem.replaceOwned(u8, a, marked, "\"residents_format\":\"mxfp8\"", "\"residents_format\":\"fp8\"");
    defer a.free(bad);
    try std.testing.expectEqual(abi.Verdict.Code.unsupported, parse(&dir, .of(bad), .{}, null, false, &p).code);
}
test "MiniMax default context advertises the window its module can serve" {
    const a = std.testing.allocator;
    const marked = try std.mem.replaceOwned(u8, a, @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"},");
    defer a.free(marked);
    const dir: abi.PackDir = undefined;
    var p: abi.Parsed = .{};
    try std.testing.expect(parse(&dir, .of(marked), .{}, null, false, &p).isOk());
    defer freeParsed(&p);
    try std.testing.expectEqual(@as(u64, 4096), p.shell.context_cap);
    try std.testing.expectEqual(@as(u64, 1048576), p.shell.max_position_embeddings);
}
test "MiniMax prepared settings survive native parse and model settings override them" {
    const a = std.testing.allocator;
    const marked = try std.mem.replaceOwned(u8, a, @embedFile("config_fixture.json"), "\"model_type\":\"minimax_m3_vl\",", "\"model_type\":\"minimax_m3_vl\",\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-nvfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\",\"settings\":{\"expert_rows\":16,\"resident_seed\":\"none\"}},");
    defer a.free(marked);
    const dir: abi.PackDir = undefined;
    var p: abi.Parsed = .{};
    try std.testing.expect(parse(&dir, .of(marked), .of("{\"expert_rows\":32,\"resident_seed\":null}"), null, false, &p).isOk());
    defer freeParsed(&p);
    const settings = try std.json.parseFromSlice(std.json.Value, a, of(&p).settings_json, .{});
    defer settings.deinit();
    try std.testing.expectEqual(@as(i64, 32), settings.value.object.get("expert_rows").?.integer);
    const seed = settings.value.object.get("resident_seed") orelse return error.MissingPreparedSetting;
    try std.testing.expect(seed == .string);
    try std.testing.expectEqualStrings("none", seed.string);
}
