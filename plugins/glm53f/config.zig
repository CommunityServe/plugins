//! glm53f's claim and parse, MLX-free. It claims a GLM-5.3-Flash streaming repack by its markers: exllamav3's
//! `quantization_config` (`quant_method` exl3), the converter's `expert_streaming` block of family turboexl3 and the
//! pack's experts.bin. mlx-serve's own glm5_next keeps every other pack (it claims by `model_type`). The parse is
//! mlx-serve's glm5_next parse over the KDA-hybrid family config (`kit.bases.kda_hybrid.config`), its facts the same, so
//! the host bills and schedules the pack as it does for glm5_next. Text only: the vision tower is not read.
//! copied_from: mlx-serve src/plugins/glm5_next/config.zig@385c3957 (`readGlm` without its vision keys, `glmFacts`).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const sdk = kit.sdk;
const family = kit.bases.kda_hybrid.config;
const options = @import("src/options.zig");

pub const Config = family.Config;
pub const of = family.of;
pub const freeParsed = family.freeParsed;
const Objects = family.Objects;
const ParseError = family.ParseError;
const field = family.field;
const value = family.value;
const u32v = family.u32v;
const f32v = family.f32v;

pub const model_types = [_][]const u8{ "glm5_next", "glm5_next_text" };

const plugin_options = options.forKit(sdk.plugin.Option);

pub const plugin = sdk.Plugin{
    .name = "glm53f",
    .kind = .model,
    .options = &plugin_options,
};

/// The bank block's family this plugin claims (the converter's LAYOUT.md `expert_streaming.family`).
pub const bank_family = "turboexl3";

/// Whether `config_json` is a GLM-5.3-Flash streaming repack this plugin serves: a glm5_next `model_type`, exllamav3's
/// `quantization_config`, an `expert_streaming` block of `bank_family`, and no Sushi `expert_quant`.
pub fn isRepack(model_type: []const u8, config_json: []const u8) bool {
    for (model_types) |t| {
        if (std.mem.eql(u8, t, model_type)) break;
    } else return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return false;
    if (doc != .object) return false;
    const qc = doc.object.get("quantization_config") orelse return false;
    if (qc != .object or !strIs(qc.object.get("quant_method"), "exl3")) return false;
    const es = doc.object.get("expert_streaming") orelse return false;
    if (es != .object or !strIs(es.object.get("family"), bank_family)) return false;
    return doc.object.get("expert_quant") == null;
}

fn strIs(v: ?std.json.Value, want: []const u8) bool {
    const x = v orelse return false;
    return x == .string and std.mem.eql(u8, x.string, want);
}

/// A marker claim, which beats the built-in glm5_next's claim by `model_type`; any other pack is not this plugin's.
pub fn claims(model_type: abi.Str, config_json: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!isRepack(model_type.slice(), config_json.slice())) return .no;
    return if (dir.has("experts.bin")) .by_marker else .no;
}

/// glm5_next's family arm (mlx-serve src/plugins/glm5_next/config.zig `readGlm`), without the vision tower's keys.
fn readGlm(c: *Config, o: Objects) ParseError!void {
    const root = o.root;
    const cfg = o.cfg;
    c.model_type = "glm5_next";
    c.weight_prefix = if (field(root, "text_config") != null) "language_model.model" else "model";
    c.has_sliding_window = false;
    c.rope_scaling_factor = 1.0;
    c.rope_local_base_freq = c.rope_theta;
    c.moe_sigmoid_router = true;
    if (field(cfg, "n_routed_experts")) |v| c.num_experts = try u32v(v);
    if (field(cfg, "norm_topk_prob")) |v| if (v == .bool) {
        c.moe_route_norm = v.bool;
    };
    if (field(cfg, "routed_scaling_factor")) |v| c.router_scaling_factor = try f32v(v);
    if (field(cfg, "n_group")) |v| c.moe_n_group = try u32v(v);
    if (field(cfg, "topk_group")) |v| c.moe_topk_group = try u32v(v);
    if (field(cfg, "n_shared_experts")) |v| if (v == .integer) {
        c.shared_expert_intermediate_size = std.math.mul(u32, try u32v(v), c.moe_intermediate_size) catch return error.InvalidConfigField;
    };
    if (field(cfg, "swiglu_limit")) |v| c.swiglu_limit = try f32v(v);
    c.swiglu_clamp = true;

    if (field(cfg, "qk_rope_head_dim")) |v| if (v == .integer and v.integer != 0) return error.Unsupported;
    if (field(cfg, "q_lora_rank")) |v| c.dsa_q_lora_rank = try u32v(v);
    if (field(cfg, "kv_lora_rank")) |v| c.dsa_kv_lora_rank = try u32v(v);
    if (field(cfg, "qk_nope_head_dim")) |v| c.dsa_head_dim = try u32v(v);
    if (field(cfg, "v_head_dim")) |v| if (try u32v(v) != c.dsa_head_dim) return error.Unsupported;
    c.head_dim = c.dsa_head_dim;
    c.num_key_value_heads = c.num_attention_heads;
    if (field(cfg, "index_topk")) |v| c.dsa_index_topk = try u32v(v);
    if (field(cfg, "index_kpool")) |v| c.dsa_index_kpool = try u32v(v);
    if (field(cfg, "index_n_heads")) |v| c.dsa_index_heads = try u32v(v);
    if (field(cfg, "index_head_dim")) |v| c.dsa_index_head_dim = try u32v(v);
    inline for (.{ "index_kpool_compress", "index_kpool_always_select_tail", "mla_use_nope" }) |key|
        if (field(cfg, key)) |v| if (v == .bool and !v.bool) return error.Unsupported;
    if (c.dsa_index_kpool == 0 or c.dsa_index_topk % c.dsa_index_kpool != 0) return error.Unsupported;
    if (field(cfg, "indexer_types")) |v| for ((try value(.array, v)).items) |t| {
        if (!std.mem.eql(u8, try value(.string, t), "full")) return error.Unsupported;
    };

    c.full_attention_interval = 4;
    if (field(cfg, "layer_types")) |v| for ((try value(.array, v)).items, 0..) |t, i| {
        const name = try value(.string, t);
        const sparse = std.mem.eql(u8, name, "deepseek_sparse_attention") or std.mem.eql(u8, name, "full_attention");
        if (sparse == c.isLinearLayer(@intCast(i))) return error.Unsupported;
    };
    if (field(cfg, "mlp_layer_types")) |v| {
        c.first_k_dense_replace = 0;
        for ((try value(.array, v)).items, 0..) |t, i| if (std.mem.eql(u8, try value(.string, t), "dense")) {
            if (c.first_k_dense_replace != i) return error.Unsupported;
            c.first_k_dense_replace += 1;
        };
    } else if (field(cfg, "first_k_dense_replace")) |v| c.first_k_dense_replace = try u32v(v);

    c.linear_num_key_heads = c.num_attention_heads;
    c.linear_key_head_dim = 128;
    if (field(cfg, "linear_attn_config")) |lac_v| {
        const lac = try value(.object, lac_v);
        if (field(lac, "num_heads")) |v| c.linear_num_key_heads = try u32v(v);
        if (field(lac, "head_dim")) |v| c.linear_key_head_dim = try u32v(v);
        if (field(lac, "short_conv_kernel_size")) |v| c.linear_conv_kernel_dim = try u32v(v);
        if (field(lac, "gate_lower_bound")) |v| c.kda_gate_lower_bound = try f32v(v);
    }
    if (c.kda_gate_lower_bound == 0.0) c.kda_gate_lower_bound = -5.0;
    c.linear_num_value_heads = c.linear_num_key_heads;
    c.linear_value_head_dim = c.linear_key_head_dim;
    c.kda_vector_gate = true;
    c.kda_sigmoid_out_gate = true;

    c.dsv4_hc_mult = 4;
    if (field(cfg, "hc_mult")) |v| c.dsv4_hc_mult = try u32v(v);
    if (c.dsv4_hc_mult != 4) return error.Unsupported;
    c.dsv4_hc_sinkhorn_iters = 20;
    if (field(cfg, "hc_sinkhorn_iters")) |v| c.dsv4_hc_sinkhorn_iters = try u32v(v);
    if (field(cfg, "hc_eps")) |v| c.dsv4_hc_eps = try f32v(v);
}

/// glm5_next's `Parsed` facts (mlx-serve src/plugins/glm5_next/config.zig `glmFacts`), with no images.
fn glmFacts(c: *const Config, p: *abi.Parsed) void {
    p.caps = .{
        .batches_decode = true,
        // Each slot's DSA cache is advanced by its own call: a batched tick never pads or stacks KV.
        .batch_pads_kv = false,
        .moe = c.num_experts > 0,
        .experts_stream = c.num_experts > 0,
        .dflash_capture = true,
        .dflash_policy = true,
        .images = false,
    };
    if (c.streamed_bank) family.bankFacts(p);
    // J5 (src/prompt.zig): the engine hands a streaming pack's whole prompt to one forward, by the chunk loop's spans.
    if (c.streamed_bank and options.on(.whole_prompt)) p.caps.whole_prompt_prefill = true;
    p.shell.state_dtype = .f32;
    p.chat.thinking_default = .on;
    p.chat.effort_vocab = .glm5;
    // A pool's completing indexer row stores the pool's key in its gate half; v2: the KDA q/k norm epsilon.
    p.cache.layout_namespace = abi.Str.of("glm5-pooled-keys-v2");
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .f32 };
    // Its head drafts one slot at a time: MTP rounds never group.
    p.spec = .{ .mtp_default_depth = 3, .mtp_layout = .in_checkpoint, .head_reads_final_norm = true, .policy = .{ .mtp_group_solo = true } };
    p.placeholder_ids.image = c.image_token_id;
    p.weights.prefix = abi.Str.of(c.weight_prefix);
    if (c.modelopt) p.weights.quant_layout = abi.Str.of("modelopt");
    // The engine's own glm5_next config with this binding: the pack's weight renames (the hyper-connections, kv_b's
    // halves, the MTP layer), the family fields its GLM bills read, and the streaming bank's facts. Without it an
    // outside plugin's config is these facts alone.
    p.engine_config = true;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (refusal(config_json.slice())) |v| return v;
    return family.parseWith("glm53f", &model_types, readGlm, glmFacts, dir, config_json, span, stub, out);
}

/// What the family parse would take but this plugin does not serve, refused by name: a Sushi `expert_quant` pack (its
/// experts run on mlx-serve's own EXL3 format, which a plugin cannot reach) and fp8_block128 residents (the kit has no
/// fp8 reader).
fn refusal(config_json: []const u8) ?abi.Verdict {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return null;
    if (doc != .object) return null;
    if (doc.object.get("expert_quant") != null) return refused("glm53f: a Sushi expert_quant pack (mlx-serve's glm5_next serves it)");
    if (doc.object.get("residents_format") != null) return refused("glm53f: a residents_format pack (fp8_block128 residents need mlx-serve's own reader)");
    return null;
}

fn refused(comptime why: []const u8) abi.Verdict {
    kit.log.err(why ++ "\n", .{});
    return abi.Verdict.refuse(.unsupported, why);
}

// ── tests ──

const testing = std.testing;

/// The fixture's config.json (a GLM-5.3-Flash sc_3.10bpw streaming repack, unchanged).
const repack_json = @embedFile("tests/fixtures/config.json");

fn packDir(json: []const u8, bank: bool) sdk.fixture.MemDir {
    const S = struct {
        var files: [2]sdk.fixture.File = undefined;
    };
    S.files[0] = .{ .name = "config.json", .bytes = json };
    S.files[1] = .{ .name = "experts.bin", .bytes = "" };
    return .{ .files = S.files[0..if (bank) 2 else 1] };
}

test "glm53f claim: a turboexl3 streaming repack beside its experts.bin by marker; anything else not" {
    const mem = packDir(repack_json, true);
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(abi.Str.of("glm5_next"), abi.Str.of(repack_json), &dir));
    // no bank file, another model type
    const bare = packDir(repack_json, false);
    const bare_dir = bare.packDir();
    try testing.expectEqual(abi.Claim.no, claims(abi.Str.of("glm5_next"), abi.Str.of(repack_json), &bare_dir));
    try testing.expectEqual(abi.Claim.no, claims(abi.Str.of("bailing_hybrid"), abi.Str.of(repack_json), &dir));
    // a pack without exllamav3's block, of another bank family, or a Sushi expert_quant pack: the built-in's
    inline for (.{
        "{\"model_type\":\"glm5_next\",\"expert_streaming\":{\"family\":\"turboexl3\"}}",
        "{\"model_type\":\"glm5_next\",\"quantization_config\":{\"quant_method\":\"exl3\"},\"expert_streaming\":{\"family\":\"mlx-stream\"}}",
        "{\"model_type\":\"glm5_next\",\"quantization_config\":{\"quant_method\":\"exl3\"},\"expert_streaming\":{\"family\":\"turboexl3\"},\"expert_quant\":{}}",
        "{\"model_type\":\"glm5_next\",\"quantization_config\":{\"quant_method\":\"affine\"},\"expert_streaming\":{\"family\":\"turboexl3\"}}",
        "{",
    }) |json| try testing.expectEqual(abi.Claim.no, claims(abi.Str.of("glm5_next"), abi.Str.of(json), &dir));
}

test "glm53f parse: the repack parses as glm5_next does, streamed, decoded one request at a time, without images" {
    const mem = packDir(repack_json, true);
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(repack_json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expect(c.isGlm5() and c.streamed_bank);
    try testing.expectEqualStrings("glm5_next", p.shell.model_type.slice());
    try testing.expect(p.caps.streamed_weights and !p.caps.batches_decode and !p.caps.images and p.caps.moe);
    try testing.expectEqualStrings("glm5-pooled-keys-v2", p.cache.layout_namespace.slice());
    try testing.expectEqual(@as(f32, -5.0), c.kda_gate_lower_bound);
    try testing.expectEqual(@as(u32, 4), c.dsv4_hc_mult);
}

test "glm53f parse: the repack asks for the engine's own glm5_next config (its renames, family fields and bills)" {
    const mem = packDir(repack_json, true);
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(repack_json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.wantsEngineConfig());
}

test "glm53f parse: the repack with a Sushi expert_quant block or fp8 residents, a RoPE'd sparse attention and broken JSON are refused by name" {
    // the repack as the family parse takes it, with a block this plugin does not serve
    inline for (.{ "{\"expert_quant\":{\"format\":\"exl3\"},", "{\"residents_format\":\"fp8_block128\"," }) |extra| {
        const json = try std.mem.concat(testing.allocator, u8, &.{ extra, std.mem.trimStart(u8, repack_json, " \n{") });
        defer testing.allocator.free(json);
        const mem = packDir(json, true);
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        const v = parse(&dir, abi.Str.of(json), .{}, null, true, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
    inline for (.{
        "{\"model_type\":\"glm5_next_text\",\"num_hidden_layers\":2,\"kv_lora_rank\":64,\"index_kpool\":4,\"index_topk\":8,\"qk_rope_head_dim\":64}",
        "{\"model_type\":\"glm_moe_dsa\",\"num_hidden_layers\":2}",
    }) |json| {
        const mem = packDir(json, false);
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        const v = parse(&dir, abi.Str.of(json), .{}, null, true, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
    const mem = packDir("{", false);
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parse(&dir, abi.Str.of("{"), .{}, null, true, &p).code);
}

test "glm53f parse: a streaming pack takes the whole prompt only with GLM53F_WHOLE_PROMPT on" {
    const mem = packDir(repack_json, true);
    const dir = mem.packDir();
    defer options.forced[@backingInt(options.Switch.whole_prompt)] = null;
    for ([_]bool{ false, true }) |on| {
        options.forced[@backingInt(options.Switch.whole_prompt)] = on;
        var p: abi.Parsed = .{};
        try testing.expect(parse(&dir, abi.Str.of(repack_json), .{}, null, true, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(on, p.caps.whole_prompt_prefill);
        // a whole prompt read at its last row asks for that row alone (`Caps.prompt_last_row`), so the pass's head
        // runs over its tail; without it the engine asks every prompt row's logits
        try testing.expectEqual(on, p.caps.prompt_last_row);
        // the engine keeps its GLM bills: the plugin prices no prompt of its own
        try testing.expect(p.caps.streamed_weights);
    }
}

test {
    _ = options;
}
