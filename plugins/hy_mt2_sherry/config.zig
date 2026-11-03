//! Hy-MT2 1.8B, Sherry/STQ1_0 1.25-bit MLX packs (Sube126/Hy-MT2-1.8B-1.25Bit-MLX and the kuotient parent).
//!
//! Claims and parse are MLX-free (discovery runs them with no device). The pack is mlx-lm's `hunyuan_v1_dense`
//! with `config.json.model_file = sherry_model.py`: `sherry_embed_bits` is the marker that tells a Sherry pack
//! from a plain Hunyuan dense one (the built-in does not serve either, but a future one might).
//!
//! Geometry the forward's Metal kernels require (checked here, refused by name):
//!   * every STQ linear's input width is a multiple of 2048 (the GEMV walks K in 2048-wide chunks),
//!   * every STQ output width is a multiple of 8 (`R_ROWS`),
//!   * the embedding is 6-bit affine, group 64.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;

pub const model_types = [_][]const u8{"hunyuan_v1_dense"};

/// A plain `hunyuan_v1_dense` pack (if one ever appears) stays with the engine: only a Sherry pack has the
/// `sherry_embed_bits` marker or points `model_file` at `sherry_model.py`.
pub fn claims(model_type: abi.Str, config_json: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    const json = config_json.slice();
    if (std.mem.indexOf(u8, json, "\"sherry_embed_bits\"") != null or
        std.mem.indexOf(u8, json, "sherry_model.py") != null) return .by_marker;
    return .no;
}

/// What `bill`, `load` and the forward read.
pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    /// `rope_scaling.alpha` of the pack's DynamicNTKAlphaRoPE (1000 on Hy-MT2).
    rope_alpha: f32,
    max_position_embeddings: u64,
    embed_bits: u8,
    embed_group: u32,
};

/// The served context cap. The pack declares 262144, but its KV is 64 KiB/token (2 x 4 kv-heads x 128 x 32
/// layers x 2 B): a full 256K context would bill 16 GiB. The host can raise it with `--ctx-size` only up to this,
/// so a translation model gets a sane default instead of an admission refusal at load.
const served_context: u64 = 32768;

/// The default context this pack serves: well under its 262144-token window, because its KV is 64 KiB/token and a
/// full window would bill 16 GiB. `context_cap` is only the default — mlx-serve's `--ctx-size` raises it (it serves
/// 65536 when asked), and `max_position_embeddings` stays the pack's own window so `/v1/models` reports the ceiling.
/// mlx-serve's per-model `ctx_size` setting does not reach a plugin (`LoadCtx.settings_json` arrives empty), so the
/// server-wide flag is the lever.
fn contextCap(max_position_embeddings: u64) u64 {
    return @min(max_position_embeddings, served_context);
}

const Raw = struct {
    model_type: []const u8 = "",
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    intermediate_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: ?u32 = null,
    head_dim: ?u32 = null,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000,
    rope_scaling: ?struct {
        alpha: ?f64 = null,
        type: ?[]const u8 = null,
    } = null,
    max_position_embeddings: u64 = 0,
    tie_word_embeddings: bool = false,
    use_qk_norm: bool = true,
    sherry_embed_bits: ?u8 = null,
    eos_token_id: ?u32 = null,
    bos_token_id: ?u32 = null,
};

const gpa = std.heap.c_allocator;

fn refuse(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    // This plugin has no per-model setting of its own; mlx-serve hands an outside plugin an empty table.
    _ = settings_json;
    if (span != null) return abi.Verdict.refuse(.unsupported, "hy_mt2_sherry: no layer split");
    const parsed = std.json.parseFromSlice(Raw, gpa, config_json.slice(), .{ .ignore_unknown_fields = true }) catch
        return refuse("hy_mt2_sherry: config.json does not parse");
    defer parsed.deinit();
    const r = parsed.value;

    if (!std.mem.eql(u8, r.model_type, model_types[0]))
        return refuse("hy_mt2_sherry: model_type is not hunyuan_v1_dense");
    if (!r.tie_word_embeddings)
        return refuse("hy_mt2_sherry: the pack's embeddings must be tied");
    if (!r.use_qk_norm)
        return refuse("hy_mt2_sherry: the forward implements use_qk_norm only");
    if (r.sherry_embed_bits != 6)
        return refuse("hy_mt2_sherry: the embedding is not the 6-bit affine layout");
    const scaling = r.rope_scaling orelse
        return refuse("hy_mt2_sherry: no rope_scaling (DynamicNTKAlphaRoPE is required)");
    if (scaling.type) |t| if (!std.mem.eql(u8, t, "dynamic"))
        return refuse("hy_mt2_sherry: rope_scaling is not dynamic");
    const alpha64 = scaling.alpha orelse
        return refuse("hy_mt2_sherry: rope_scaling has no alpha");

    const kv = r.num_key_value_heads orelse r.num_attention_heads;
    const hd = r.head_dim orelse if (r.num_attention_heads > 0) r.hidden_size / r.num_attention_heads else 0;
    if (r.vocab_size == 0 or r.vocab_size > 1 << 20 or
        r.hidden_size < 2048 or r.hidden_size > 1 << 14 or r.hidden_size % 2048 != 0 or
        r.intermediate_size < 2048 or r.intermediate_size > 1 << 16 or r.intermediate_size % 2048 != 0 or
        r.num_hidden_layers == 0 or r.num_hidden_layers > 256 or
        r.num_attention_heads == 0 or r.num_attention_heads > 256 or
        kv == 0 or r.num_attention_heads % kv != 0 or
        hd == 0 or hd > 512 or hd % 2 != 0 or
        !(r.rms_norm_eps > 0) or !(r.rope_theta > 0) or
        !(alpha64 > 0) or !(alpha64 < 1e9) or
        r.max_position_embeddings == 0)
        return refuse("hy_mt2_sherry: config.json dimensions out of range");
    // Both STQ kernels assume the output rows divide by R_ROWS = 8, and the GEMV walks `x` in 2048-wide chunks
    // (`NCH = K / 2048`), so every STQ input width must be a multiple of 2048 too: hidden for q/k/v/gate/up, the
    // query width for o_proj, the ffn width for down_proj.
    const q_width = r.num_attention_heads * hd;
    if (q_width % 8 != 0 or (kv * hd) % 8 != 0 or r.intermediate_size % 8 != 0 or r.hidden_size % 8 != 0)
        return refuse("hy_mt2_sherry: a projection width is not a multiple of 8");
    if (r.hidden_size % 2048 != 0 or q_width % 2048 != 0 or r.intermediate_size % 2048 != 0)
        return refuse("hy_mt2_sherry: a projection input width is not a multiple of 2048");

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = true } };
    p.shell = .{
        .vocab_size = r.vocab_size,
        .hidden_size = r.hidden_size,
        .intermediate_size = r.intermediate_size,
        .num_hidden_layers = r.num_hidden_layers,
        .num_attention_heads = r.num_attention_heads,
        .num_key_value_heads = kv,
        .head_dim = hd,
        .max_position_embeddings = r.max_position_embeddings,
        .context_cap = contextCap(r.max_position_embeddings),
        // The GEMV kernel loads `x` through `device half4*`: activations stay fp16, as mlx-lm runs this pack.
        .act_dtype = .f16,
        .model_type = abi.Str.of(model_types[0]),
    };
    if (r.bos_token_id) |b| p.shell.bos = b;
    if (r.eos_token_id) |e| {
        p.shell.eos[0] = e;
        p.shell.eos_count = 1;
    }
    if (!stub) kit.generation.apply(dir, &p.shell);

    const cfg = gpa.create(Config) catch return refuse("hy_mt2_sherry: out of memory");
    cfg.* = .{
        .vocab_size = r.vocab_size,
        .hidden_size = r.hidden_size,
        .intermediate_size = r.intermediate_size,
        .num_hidden_layers = r.num_hidden_layers,
        .num_attention_heads = r.num_attention_heads,
        .num_key_value_heads = kv,
        .head_dim = hd,
        .rms_norm_eps = r.rms_norm_eps,
        .rope_theta = r.rope_theta,
        .rope_alpha = @floatCast(alpha64),
        .max_position_embeddings = r.max_position_embeddings,
        .embed_bits = 6,
        .embed_group = 64,
    };
    p.private = cfg;
    out.* = p;
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |c| gpa.destroy(@as(*Config, @ptrCast(@alignCast(c))));
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

// ── sizes the bill and the load share ──

/// Bytes of one STQ1_0 projection [out, in]: qs [out, in/8] u8 + sign [out, in/32] u8 + scales [out, in/256] f16.
pub fn stqBytes(out_dim: u64, in_dim: u64) u64 {
    return out_dim * in_dim / 8 + out_dim * in_dim / 32 + out_dim * (in_dim / 256) * 2;
}

/// Bytes of the 6-bit affine embedding [vocab, hidden]: packed u32 + scales/biases f16 per group of 64.
pub fn embedBytes(c: *const Config) u64 {
    const v: u64 = c.vocab_size;
    const h: u64 = c.hidden_size;
    return v * (h * 6 / 32) * 4 + 2 * v * (h / c.embed_group) * 2;
}

/// The dense f16 tiles one layer's seven projections materialize while a wide prefill builds it: every `stqMatmul`
/// hands its tile to the GEMM and only the layer's `evalKeep` frees them, so all seven are live at once. Upper bound
/// per tile: [N, K] f16. This is the prefill transient `bill` reports.
pub fn perLayerPrefillBytes(c: *const Config) u64 {
    const h: u64 = c.hidden_size;
    const ff: u64 = c.intermediate_size;
    const q: u64 = @as(u64, c.num_attention_heads) * c.head_dim;
    const kv: u64 = @as(u64, c.num_key_value_heads) * c.head_dim;
    const elems = q * h + 2 * kv * h + h * q + ff * h * 2 + h * ff;
    return 2 * elems;
}

/// Resident bytes of the whole pack: the STQ linears, the embedding, every RMSNorm weight.
pub fn residentBytes(c: *const Config) u64 {
    const h: u64 = c.hidden_size;
    const ff: u64 = c.intermediate_size;
    const q: u64 = @as(u64, c.num_attention_heads) * c.head_dim;
    const kv: u64 = @as(u64, c.num_key_value_heads) * c.head_dim;
    const layer = stqBytes(q, h) + stqBytes(kv, h) * 2 + stqBytes(h, q) +
        stqBytes(ff, h) * 2 + stqBytes(h, ff) +
        2 * h * 2 + 2 * c.head_dim * 2;
    return embedBytes(c) + h * 2 + @as(u64, c.num_hidden_layers) * layer;
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

const good =
    \\{"model_type":"hunyuan_v1_dense","architectures":["HunYuanDenseV1ForCausalLM"],"vocab_size":120818,
    \\"hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":32,"num_attention_heads":16,
    \\"num_key_value_heads":4,"head_dim":128,"rms_norm_eps":1e-5,"rope_theta":10000,
    \\"rope_scaling":{"type":"dynamic","alpha":1000.0},"max_position_embeddings":262144,
    \\"tie_word_embeddings":true,"use_qk_norm":true,"sherry_embed_bits":6,
    \\"bos_token_id":120000,"eos_token_id":120020,"model_file":"sherry_model.py"}
;
const good_short =
    \\{"model_type":"hunyuan_v1_dense","vocab_size":120818,"hidden_size":2048,"intermediate_size":6144,
    \\"num_hidden_layers":32,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":128,
    \\"rope_scaling":{"type":"dynamic","alpha":1000.0},"max_position_embeddings":262144,
    \\"tie_word_embeddings":true,"use_qk_norm":true,"sherry_embed_bits":6,"eos_token_id":120020}
;

fn parseFiles(files: []const fixture.File, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(files[0].bytes), .{}, null, stub, out);
}

test "hy_mt2_sherry claims only a Sherry pack" {
    const dir: abi.PackDir = undefined;
    try testing.expectEqual(abi.Claim.by_marker, claims(abi.Str.of("hunyuan_v1_dense"), abi.Str.of(good), &dir));
    try testing.expectEqual(abi.Claim.no, claims(abi.Str.of("hunyuan_v1_dense"), abi.Str.of("{\"model_type\":\"hunyuan_v1_dense\"}"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(abi.Str.of("llama"), abi.Str.of(good), &dir));
}

test "hy_mt2_sherry parse: the pack's geometry lands in the shell and the private config" {
    const files = [_]fixture.File{
        .{ .name = "config.json", .bytes = good },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7,\"top_k\":20,\"top_p\":0.6}" },
    };
    var p: abi.Parsed = .{};
    try testing.expect(parseFiles(&files, false, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 120818), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 32), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 16), p.shell.num_attention_heads);
    try testing.expectEqual(@as(u32, 4), p.shell.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), p.shell.head_dim);
    try testing.expectEqual(@as(u64, 262144), p.shell.max_position_embeddings);
    try testing.expectEqual(served_context, p.shell.context_cap);
    try testing.expectEqual(abi.DType.f16, p.shell.act_dtype);
    try testing.expectEqual(@as(i64, 120000), p.shell.bos);
    try testing.expectEqual(@as(u32, 120020), p.shell.eos[0]);
    try testing.expectEqual(@as(u32, 1), p.shell.eos_count);
    try testing.expectEqual(@as(f32, 1000), c.rope_alpha);
    try testing.expectEqual(@as(u8, 6), c.embed_bits);
    try testing.expectEqual(@as(u32, 64), c.embed_group);
    try testing.expectEqual(@as(f32, 0.7), p.shell.gen.temperature);
}

test "hy_mt2_sherry parse: a non-Sherry or non-STQ-shaped pack is refused by name, nothing to free" {
    inline for (.{
        // no sherry_embed_bits
        "{\"model_type\":\"hunyuan_v1_dense\",\"vocab_size\":120818,\"hidden_size\":2048,\"intermediate_size\":6144,\"num_hidden_layers\":32,\"num_attention_heads\":16,\"num_key_value_heads\":4,\"head_dim\":128,\"rope_scaling\":{\"type\":\"dynamic\",\"alpha\":1000},\"max_position_embeddings\":262144,\"tie_word_embeddings\":true,\"use_qk_norm\":true}",
        // 4-bit embedding
        "{\"model_type\":\"hunyuan_v1_dense\",\"vocab_size\":120818,\"hidden_size\":2048,\"intermediate_size\":6144,\"num_hidden_layers\":32,\"num_attention_heads\":16,\"num_key_value_heads\":4,\"head_dim\":128,\"rope_scaling\":{\"type\":\"dynamic\",\"alpha\":1000},\"max_position_embeddings\":262144,\"tie_word_embeddings\":true,\"use_qk_norm\":true,\"sherry_embed_bits\":4}",
        // hidden width outside the GEMV's 2048 stride
        "{\"model_type\":\"hunyuan_v1_dense\",\"vocab_size\":120818,\"hidden_size\":3072,\"intermediate_size\":6144,\"num_hidden_layers\":32,\"num_attention_heads\":16,\"num_key_value_heads\":4,\"head_dim\":128,\"rope_scaling\":{\"type\":\"dynamic\",\"alpha\":1000},\"max_position_embeddings\":262144,\"tie_word_embeddings\":true,\"use_qk_norm\":true,\"sherry_embed_bits\":6}",
        // untied head: the pack has no lm_head
        "{\"model_type\":\"hunyuan_v1_dense\",\"vocab_size\":120818,\"hidden_size\":2048,\"intermediate_size\":6144,\"num_hidden_layers\":32,\"num_attention_heads\":16,\"num_key_value_heads\":4,\"head_dim\":128,\"rope_scaling\":{\"type\":\"dynamic\",\"alpha\":1000},\"max_position_embeddings\":262144,\"tie_word_embeddings\":false,\"use_qk_norm\":true,\"sherry_embed_bits\":6}",
        "{",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseFiles(&.{.{ .name = "config.json", .bytes = json }}, true, &p);
        try testing.expect(!v.isOk() and v.why.len > 0 and p.private == null);
    }
}

test "hy_mt2_sherry parse: a query width the GEMV cannot walk is refused, not served with zeros" {
    // 16 heads x head_dim 64 = 1024-wide o_proj input: NCH would be 0 and decode would project o_proj to zeros.
    const narrow =
        \\{"model_type":"hunyuan_v1_dense","vocab_size":120818,"hidden_size":2048,"intermediate_size":6144,
        \\"num_hidden_layers":32,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":64,
        \\"rope_scaling":{"type":"dynamic","alpha":1000.0},"max_position_embeddings":262144,
        \\"tie_word_embeddings":true,"use_qk_norm":true,"sherry_embed_bits":6}
    ;
    var p: abi.Parsed = .{};
    const v = parseFiles(&.{.{ .name = "config.json", .bytes = narrow }}, true, &p);
    try testing.expect(!v.isOk() and p.private == null);
}

test "hy_mt2_sherry context: the default cap is 32768 and never past the pack's own window" {
    try testing.expectEqual(@as(u64, 32768), contextCap(262144));
    // A pack trained shorter keeps its own window.
    try testing.expectEqual(@as(u64, 8192), contextCap(8192));
}

test "hy_mt2_sherry sizes: the prefill transient is every projection tile of a layer, not three" {
    var p: abi.Parsed = .{};
    const files = [_]fixture.File{.{ .name = "config.json", .bytes = good_short }};
    try testing.expect(parseFiles(&files, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    // q/o 2048x2048, k/v 512x2048, gate/up 6144x2048, down 2048x6144, all f16.
    const elems = 2048 * 2048 * 2 + 512 * 2048 * 2 + 6144 * 2048 * 2 + 2048 * 6144;
    try testing.expectEqual(2 * elems, perLayerPrefillBytes(c));
    const mib = perLayerPrefillBytes(c) >> 20;
    try testing.expect(mib >= 92 and mib <= 93);
}

test "hy_mt2_sherry sizes: one layer and the whole pack match the published 454 MB" {
    var p: abi.Parsed = .{};
    const files = [_]fixture.File{.{ .name = "config.json", .bytes = good_short }};
    try testing.expect(parseFiles(&files, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    // 2048x2048 q/o, 512x2048 k/v, 6144x2048 gate/up, 2048x6144 down + norms
    const layer = stqBytes(2048, 2048) * 2 + stqBytes(512, 2048) * 2 +
        stqBytes(6144, 2048) * 2 + stqBytes(2048, 6144) + 2 * 2048 * 2 + 2 * 128 * 2;
    try testing.expectEqual(layer * 32 + embedBytes(c) + 2048 * 2, residentBytes(c));
    try testing.expect(residentBytes(c) > 440_000_000 and residentBytes(c) < 470_000_000);
}
