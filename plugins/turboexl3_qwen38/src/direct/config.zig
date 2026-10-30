//! The direct model's config: a Qwen3.8-Flash-Next EXL3 pack (`qwen4_exp`, the K packs), read from its config.json
//! with the engine's semantics, MLX-free. One struct the parse fills; its fields carry the engine's `ModelConfig`
//! names and types, and the load makes the copied trunk's `ModelConfig` from it field by field. Text only: the vision
//! tower's keys are read for the placeholder ids alone, and a shape the K packs never have (YaRN, another quant mode,
//! a layer span) is refused by name.

const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const generation = @import("mlx_serve").generation;
const pack_keys = @import("../mlx_serve/sdk/quant/exl3/pack_keys.zig");

pub const model_types = [_][]const u8{ "qwen4_exp", "qwen4_exp_text" };

pub const QuantMode = @import("../mlx_serve/sdk/quant/params.zig").QuantMode;
pub const HiddenAct = @import("../mlx_serve/sdk/blocks/mlp/activations.zig").HiddenAct;
pub const Qwen4NormConvention = enum { delta, folded };

/// The n-gram hash's fixed arrays (`NgramHash`): the widest n-gram and the most heads it holds.
pub const max_ngram_size = 8;
pub const max_heads = 32;

/// The parsed pack. Defaults are the engine's, so a key a pack omits reads as it always has; the constants are the
/// arch's (`toModelConfig`'s).
pub const Config = struct {
    model_type: []const u8 = "qwen4_exp",
    weight_prefix: []const u8 = "language_model.model",

    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    max_position_embeddings: u32 = 0,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 1000000.0,
    rope_scaling_factor: f32 = 1.0,
    query_pre_attn_scalar: u32 = 256,
    partial_rotary_factor: f32 = 1.0,
    tie_word_embeddings: bool = false,
    bos_token_id: ?u32 = null,
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    // The arch's constants.
    hidden_act: HiddenAct = .silu,
    norm_has_offset: bool = false,
    has_final_norm: bool = false, // the hyper-connection mixer replaces model.norm
    scale_embeddings: bool = false,
    has_pre_ff_norm: bool = false,
    has_qk_norm: bool = true,
    has_sliding_window: bool = false,
    attn_output_gate: bool = true,
    kda_sigmoid_out_gate: bool = true, // output_gate_type "sigmoid"
    hadamard_block: u32 = 0,

    /// The pack's nominal `quantization` block (no trunk tensor is affine: `trunk_quant` names the format).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,

    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    /// The routed banks are EXL3 (`expert_quant`): the load parses the block's codebook, window and rate.
    expert_quant: bool = false,
    /// The trunk's linears are a quant format's (`trunk_quant`): always, on a K pack.
    trunk_format: bool = false,

    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    full_attention_interval: u32 = 0,
    /// The GDN state and decay gate in f32, as the reference keeps them whatever `mamba_ssm_dtype` says.
    ssm_state_f32: bool = true,

    hc_count: u32 = 4,
    hc_lowrank: u32 = 320,
    ple_layer_idx: i32 = -1, // 0-based; the config lists 1-based ids
    ple_embed_dim: u32 = 0,
    ple_conv_kernel: u32 = 4,
    ngram_size: u32 = 3,
    heads_per_ngram: u32 = 8,
    ngram_vocab_base: u64 = 20_000_000,
    ngram_vocab_divisor: u32 = 128,
    ngram_seed: u64 = 1234,
    split_ngram_parts: u32 = 128,
    /// The n-gram table's file in the pack (`ngram_table.file`); empty = `ngram_table.bin`.
    ngram_table_file_buf: [96]u8 = @splat(0),
    ngram_table_file_len: u8 = 0,
    /// The table's path (the pack's folder and `ngramTableFile`), set by the load; `deinit` frees it.
    ngram_table_path: ?[]const u8 = null,
    /// The text config's first eos: the n-gram hash's segment reset token.
    ngram_eos: u32 = 0,
    qwen4_norm_convention: ?Qwen4NormConvention = null,
    indexer_n_heads: u32 = 0,
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,

    // The vision tower's keys the host's facts read (the server never sends this plugin an image).
    qwen_vision: bool = false,
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    image_token_id: u32 = 0,
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,
    gen_min_p: ?f32 = null,
    gen_enable_thinking: ?bool = null,

    fn addEos(self: *Config, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    fn mergeEos(self: *Config, id: u32) void {
        if (std.mem.indexOfScalar(u32, self.eos_token_ids[0..self.num_eos_tokens], id) == null) self.addEos(id);
    }

    /// A GatedDeltaNet layer; every `full_attention_interval`-th layer is full attention.
    pub fn isLinearLayer(self: Config, layer_idx: u32) bool {
        if (self.full_attention_interval == 0) return false;
        return ((layer_idx + 1) % self.full_attention_interval) != 0;
    }

    pub fn isFullLayer(self: *const Config, li: u32) bool {
        return !self.isLinearLayer(li);
    }

    /// The layers that hold an attention KV cache.
    pub fn attnCacheLayerCount(self: *const Config) u32 {
        if (self.full_attention_interval == 0) return self.num_hidden_layers;
        var n: u32 = 0;
        var i: u32 = 0;
        while (i < self.num_hidden_layers) : (i += 1) {
            if (!self.isLinearLayer(i)) n += 1;
        }
        return n;
    }

    /// The PLE layer (a single rank holds every layer).
    pub fn pleLocalLayer(self: *const Config) ?u32 {
        if (self.ple_layer_idx < 0 or self.ple_layer_idx >= self.num_hidden_layers) return null;
        return @intCast(self.ple_layer_idx);
    }

    /// The longest sequence the rope resolves (no YaRN on a K pack): `max_position_embeddings`.
    pub fn contextCap(self: *const Config) u32 {
        return self.max_position_embeddings;
    }

    /// QSA's pooled key history: bytes per token across the attention layers.
    pub fn qsaHistoryBytesPerToken(self: *const Config) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return @as(u64, self.attnCacheLayerCount()) * self.indexer_head_dim * 2 / ratio;
    }

    /// Bytes one SSM checkpoint holds: the recurrent state and conv window of every linear layer.
    pub fn ssmCheckpointBytes(self: *const Config) u64 {
        if (self.linear_num_value_heads == 0) return 0;
        const linear: u64 = @as(u64, self.num_hidden_layers) -| self.attnCacheLayerCount();
        if (linear == 0) return 0;
        const elem: u64 = if (self.ssm_state_f32) 4 else 2;
        const state: u64 = @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim * self.linear_key_head_dim * elem;
        const conv_dim: u64 = 2 * @as(u64, self.linear_num_key_heads) * self.linear_key_head_dim + @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim;
        return linear * (state + (@as(u64, self.linear_conv_kernel_dim) -| 1) * conv_dim * 2);
    }

    /// Activations are bf16.
    pub fn actDtype(self: *const Config) @import("../mlx_serve/mlx/ffi.zig").mlx_dtype {
        _ = self;
        return .bfloat16;
    }

    /// The GDN recurrent state and its decay gate: f32 (the reference's), unless the lever turned it off.
    pub fn ssmStateDtype(self: *const Config) @import("../mlx_serve/mlx/ffi.zig").mlx_dtype {
        return if (self.ssm_state_f32) .float32 else .bfloat16;
    }

    /// The per-head q/k RMS norm epsilon of the GDN prework.
    pub fn gdnQkNormEps(self: *const Config) f32 {
        _ = self;
        return 1e-6;
    }

    pub fn isQwen4(self: *const Config) bool {
        _ = self;
        return true;
    }

    pub fn isMoe(self: *const Config) bool {
        return self.num_experts > 0;
    }

    /// The n-gram table's file name in the pack: `ngram_table.file`, else `ngram_table.bin`.
    pub fn ngramTableFile(self: *const Config) []const u8 {
        return if (self.ngram_table_file_len > 0) self.ngram_table_file_buf[0..self.ngram_table_file_len] else "ngram_table.bin";
    }

    /// Frees the load's allocations (`ngram_table_path`); idempotent.
    pub fn deinit(self: *Config, a: std.mem.Allocator) void {
        if (self.ngram_table_path) |p| a.free(p);
        self.ngram_table_path = null;
    }
};

pub const ParseError = pack_keys.Error || error{
    InvalidConfigField,
    UnsupportedQuantMode,
    UnsupportedQuantBits,
    InvalidQwen4NormConvention,
    InvalidQwen4ConfigField,
    InvalidQwen4PleLayer,
    InvalidQwen4NgramSize,
    InvalidQwen4NgramHeads,
    InvalidQwen4NgramVocab,
    InvalidQwen4PleGeometry,
    InvalidQwen4Indexer,
    /// A shape no K pack has: another family's keys, a rope that scales, a pack without its trunk format.
    Unsupported,
    OutOfMemory,
};

// Readers with the engine parser's semantics: a JSON null reads as an omitted key.

fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn u32Of(v: std.json.Value) ParseError!u32 {
    if (v != .integer) return error.InvalidConfigField;
    return std.math.cast(u32, v.integer) orelse error.InvalidConfigField;
}

fn f32Of(v: std.json.Value) ParseError!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| @floatCast(x),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.InvalidConfigField;
    return f;
}

fn readU32(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| out.* = try u32Of(v);
}

/// `key` into `out` when it is an integer; another type leaves the field as it was.
fn readU32If(obj: std.json.ObjectMap, key: []const u8, out: *u32) ParseError!void {
    if (field(obj, key)) |v| if (v == .integer) {
        out.* = try u32Of(v);
    };
}

fn readF32(obj: std.json.ObjectMap, key: []const u8, out: *f32) ParseError!void {
    if (field(obj, key)) |v| out.* = try f32Of(v);
}

fn readBool(obj: std.json.ObjectMap, key: []const u8, out: *bool) void {
    if (field(obj, key)) |v| if (v == .bool) {
        out.* = v.bool;
    };
}

/// A qwen4 key read strictly: a non-negative integer or a named refusal.
fn q4U64(obj: std.json.ObjectMap, key: []const u8) ParseError!?u64 {
    const v = field(obj, key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

fn q4U32(obj: std.json.ObjectMap, key: []const u8) ParseError!?u32 {
    const v = try q4U64(obj, key) orelse return null;
    return std.math.cast(u32, v) orelse error.InvalidQwen4ConfigField;
}

/// Keys of another family's shapes.
const foreign_keys = [_][]const u8{
    "use_bidirectional_attention", "global_head_dim", "num_global_key_value_heads", "num_kv_shared_layers",
    "attention_k_eq_v",            "final_logit_softcapping", "hidden_size_per_layer_input", "top_k_experts",
};
const foreign_root_keys = [_][]const u8{ "pooling_mode", "audio_config", "audio_token_id", "boa_token_id", "eoa_token_index", "eoa_token_id" };

/// config.json into `c`.
pub fn parseJson(c: *Config, root: std.json.ObjectMap) ParseError!void {
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return error.InvalidConfigField) else root;
    for (foreign_keys) |k| if (field(cfg, k) != null) return error.Unsupported;
    for (foreign_root_keys) |k| if (field(root, k) != null) return error.Unsupported;

    try readU32(cfg, "vocab_size", &c.vocab_size);
    try readU32(cfg, "hidden_size", &c.hidden_size);
    if (field(cfg, "intermediate_size")) |v| {
        c.intermediate_size = try u32Of(v);
        c.intermediate_size_declared = true;
    }
    try readU32(cfg, "num_hidden_layers", &c.num_hidden_layers);
    try readU32(cfg, "num_attention_heads", &c.num_attention_heads);
    try readU32(cfg, "num_key_value_heads", &c.num_key_value_heads);
    try readU32(cfg, "head_dim", &c.head_dim);
    try readU32(cfg, "max_position_embeddings", &c.max_position_embeddings);
    try readF32(cfg, "rms_norm_eps", &c.rms_norm_eps);
    try readF32(cfg, "rope_theta", &c.rope_theta);
    try readU32(cfg, "query_pre_attn_scalar", &c.query_pre_attn_scalar);
    try readU32If(cfg, "num_experts", &c.num_experts);
    try readU32If(cfg, "num_experts_per_tok", &c.num_experts_per_tok);
    try readU32If(cfg, "moe_intermediate_size", &c.moe_intermediate_size);
    try readU32If(cfg, "shared_expert_intermediate_size", &c.shared_expert_intermediate_size);
    try readU32(cfg, "linear_num_key_heads", &c.linear_num_key_heads);
    try readU32(cfg, "linear_num_value_heads", &c.linear_num_value_heads);
    try readU32(cfg, "linear_key_head_dim", &c.linear_key_head_dim);
    try readU32(cfg, "linear_value_head_dim", &c.linear_value_head_dim);
    try readU32(cfg, "linear_conv_kernel_dim", &c.linear_conv_kernel_dim);
    try readU32(cfg, "full_attention_interval", &c.full_attention_interval);
    if (field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        c.bos_token_id = try u32Of(v);
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "full_attention") != null or field(rp.object, "sliding_attention") != null) return error.Unsupported;
        // A rope that scales (YaRN and the like) is no K pack's.
        if (field(rp.object, "rope_type")) |rt| if (rt != .string or !std.mem.eql(u8, rt.string, "default")) return error.Unsupported;
        try readF32(rp.object, "rope_theta", &c.rope_theta);
        try readF32(rp.object, "partial_rotary_factor", &c.partial_rotary_factor);
        readBool(rp.object, "mrope_interleaved", &c.mrope_interleaved);
        if (field(rp.object, "mrope_section")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
            if (i >= 3) break;
            if (item == .integer) c.mrope_section[i] = try u32Of(item);
        };
    };
    if (field(cfg, "rope_scaling") != null) return error.Unsupported;
    c.rope_local_base_freq = c.rope_theta;
    readBool(root, "tie_word_embeddings", &c.tie_word_embeddings);
    readBool(cfg, "tie_word_embeddings", &c.tie_word_embeddings);
    if (c.max_position_embeddings == 0) try readU32If(root, "max_position_embeddings", &c.max_position_embeddings);

    try parseQuantization(c, root);

    if (field(root, "eos_token_id")) |v| switch (v) {
        .integer => |i| c.addEos(std.math.cast(u32, i) orelse return error.InvalidConfigField),
        .array => |arr| for (arr.items) |item| if (item == .integer) c.addEos(try u32Of(item)),
        else => {},
    };
    // The vision tower's ids, for the host's placeholders (no image reaches this plugin).
    c.qwen_vision = if (field(root, "vision_config")) |vv| vv == .object else false;
    try readU32If(root, "image_token_id", &c.image_token_id);
    if (field(root, "image_token_index")) |v| if (v == .integer and c.image_token_id == 0) {
        c.image_token_id = try u32Of(v);
    };
    try readU32If(root, "video_token_id", &c.video_token_id);
    try readU32If(root, "vision_start_token_id", &c.vision_start_token_id);
    try readU32If(root, "vision_end_token_id", &c.vision_end_token_id);

    if (root.get("qwen4_norm_convention")) |v| {
        if (v != .string) return error.InvalidQwen4NormConvention;
        c.qwen4_norm_convention = std.meta.stringToEnum(Qwen4NormConvention, v.string) orelse return error.InvalidQwen4NormConvention;
    }
    // A K pack is fully EXL3: its trunk block names the format, and its n-gram table file is read.
    c.trunk_format = try pack_keys.hasTrunkQuant(root);
    if (!c.trunk_format) return error.Unsupported;
    if (try pack_keys.ngramTableFile(root)) |f| {
        if (f.len > c.ngram_table_file_buf.len) return error.InvalidConfigField;
        @memcpy(c.ngram_table_file_buf[0..f.len], f);
        c.ngram_table_file_len = @intCast(f.len);
    }
    if (field(root, "expert_quant")) |eq| {
        if (eq != .object) return error.InvalidConfigField;
        c.expert_quant = true;
    }
    if (field(cfg, "query_pre_attn_scalar") == null) c.query_pre_attn_scalar = c.head_dim;
    c.ple_embed_dim = c.hidden_size;
    if (try q4U32(cfg, "hc_count")) |v| c.hc_count = v;
    if (try q4U32(cfg, "hc_lowrank")) |v| c.hc_lowrank = v;
    {
        const v = field(cfg, "ple_layer_ids") orelse return error.InvalidQwen4PleLayer;
        if (v != .array or v.array.items.len != 1 or v.array.items[0] != .integer) return error.InvalidQwen4PleLayer;
        const id = v.array.items[0].integer;
        if (id < 1 or id > @as(i64, c.num_hidden_layers)) return error.InvalidQwen4PleLayer;
        c.ple_layer_idx = std.math.cast(i32, id - 1) orelse return error.InvalidQwen4PleLayer;
    }
    if (try q4U32(cfg, "ple_embed_dim")) |v| c.ple_embed_dim = v;
    if (try q4U32(cfg, "ple_conv_kernel_size")) |v| c.ple_conv_kernel = v;
    if (try q4U32(cfg, "ngram_size")) |v| c.ngram_size = v;
    if (try q4U32(cfg, "heads_per_ngram")) |v| c.heads_per_ngram = v;
    if (try q4U64(cfg, "ngram_vocab_size_base")) |v| c.ngram_vocab_base = v;
    if (try q4U32(cfg, "make_ngram_vocab_size_divisible_by")) |v| c.ngram_vocab_divisor = v;
    if (try q4U64(cfg, "seed")) |v| c.ngram_seed = v;
    if (try q4U32(cfg, "split_ngram_parts")) |v| c.split_ngram_parts = v;
    if (try q4U32(cfg, "indexer_n_heads")) |v| c.indexer_n_heads = v;
    if (try q4U32(cfg, "indexer_head_dim")) |v| c.indexer_head_dim = v;
    if (try q4U32(cfg, "indexer_budget")) |v| c.indexer_budget = v;
    if (try q4U32(cfg, "indexer_compress_ratio")) |v| c.indexer_compress_ratio = v;
    if (field(cfg, "eos_token_id")) |v| {
        switch (v) {
            .integer => |i| c.ngram_eos = std.math.cast(u32, i) orelse return error.InvalidConfigField,
            .array => |arr| if (arr.items.len > 0 and arr.items[0] == .integer) {
                c.ngram_eos = try u32Of(arr.items[0]);
            },
            else => {},
        }
        if (c.num_eos_tokens == 0) c.addEos(c.ngram_eos);
    }
    try validate(c);
    if (c.linear_num_key_heads == 0 or c.linear_num_value_heads == 0 or c.linear_num_value_heads % c.linear_num_key_heads != 0)
        return error.InvalidConfigField;
}

/// The nominal block: affine at a width MLX quantizes (no trunk tensor reads it).
fn parseQuantization(c: *Config, root: std.json.ObjectMap) ParseError!void {
    const qv = field(root, "quantization") orelse return;
    if (qv != .object) return error.InvalidConfigField;
    const q = qv.object;
    try readU32(q, "bits", &c.quant_bits);
    try readU32(q, "group_size", &c.quant_group_size);
    if (field(q, "mode")) |m| if (m == .string) {
        c.quant_mode = std.meta.stringToEnum(QuantMode, m.string) orelse return error.UnsupportedQuantMode;
    };
    if (c.quant_mode != .affine) return error.UnsupportedQuantMode;
    if (c.quant_bits != 0) switch (c.quant_bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.UnsupportedQuantBits,
    };
}

/// Every bound the forward indexes a fixed array with or divides by, refused by name.
fn validate(c: *const Config) ParseError!void {
    if (c.ngram_size < 2 or c.ngram_size > max_ngram_size) return error.InvalidQwen4NgramSize;
    if (c.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (@as(u64, c.ngram_size - 1) * c.heads_per_ngram > max_heads) return error.InvalidQwen4NgramHeads;
    if (c.ngram_vocab_divisor == 0 or c.ngram_vocab_base < 2) return error.InvalidQwen4NgramVocab;
    if (c.ple_embed_dim == 0 or c.ple_embed_dim % ((c.ngram_size - 1) * c.heads_per_ngram) != 0 or c.split_ngram_parts == 0)
        return error.InvalidQwen4PleGeometry;
    if (c.indexer_n_heads > 0) {
        if (c.indexer_head_dim == 0 or c.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (c.indexer_budget < c.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    if (c.ple_layer_idx < 0 or c.ple_layer_idx >= c.num_hidden_layers) return error.InvalidQwen4PleLayer;
}

/// `generation_config.json`: the sampling recommendations and its eos ids (merged after config.json's).
fn applyGeneration(c: *Config, dir: *const abi.PackDir, buf: []u8) void {
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("generation_config.json", buf, &why) orelse return;
    var shell: abi.Shell = .{};
    generation.applyJson(bytes, &shell);
    const g = shell.gen;
    c.gen_temperature = if (std.math.isNan(g.temperature)) null else g.temperature;
    c.gen_top_p = if (std.math.isNan(g.top_p)) null else g.top_p;
    c.gen_top_k = if (g.top_k < 0) null else @intCast(g.top_k);
    c.gen_min_p = if (std.math.isNan(g.min_p)) null else g.min_p;
    c.gen_enable_thinking = switch (g.enable_thinking) {
        .unset => null,
        .on => true,
        .off => false,
    };
    for (shell.eos[0..shell.eos_count]) |id| c.mergeEos(id);
}

const gpa = std.heap.c_allocator;

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

/// config.json (and, unless `stub`, generation_config.json) into `c`.
pub fn parseInto(c: *Config, dir: *const abi.PackDir, config_json: []const u8, stub: bool) ParseError!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return error.InvalidConfigField;
    if (doc != .object) return error.InvalidConfigField;
    const mt = field(doc.object, "model_type") orelse return error.Unsupported;
    if (mt != .string or (!std.mem.eql(u8, mt.string, model_types[0]) and !std.mem.eql(u8, mt.string, model_types[1]))) return error.Unsupported;
    try parseJson(c, doc.object);
    // A sentence-transformers pooling sidecar makes the pack an embedding model.
    if (dir.has("1_Pooling/config.json")) return error.Unsupported;
    if (stub) return;
    const buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buf);
    applyGeneration(c, dir, buf);
    if (c.gen_top_k == null) c.gen_top_k = 20;
    if (c.gen_top_p == null) c.gen_top_p = 0.95;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return abi.Verdict.refuse(.unsupported, "turboexl3_qwen38: a layer span (the cluster split) is not served");
    const c = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "turboexl3_qwen38: out of memory");
    c.* = .{};
    parseInto(c, dir, config_json.slice(), stub) catch |e| {
        gpa.destroy(c);
        return switch (e) {
            error.Unsupported => abi.Verdict.refuse(.unsupported, "turboexl3_qwen38: a shape the K packs do not have"),
            error.OutOfMemory => abi.Verdict.refuse(.failed, "turboexl3_qwen38: out of memory"),
            inline else => |x| abi.Verdict.refuse(.failed, "turboexl3_qwen38: " ++ @errorName(x)),
        };
    };
    out.* = facts(c);
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| gpa.destroy(@as(*Config, @ptrCast(@alignCast(ptr))));
    p.private = null;
}

/// What the host reads off a parsed pack (the copy's facts for a K pack, field for field).
pub fn facts(c: *Config) abi.Parsed {
    var p: abi.Parsed = .{ .private = c };
    p.caps = .{
        .batches_decode = true,
        // Each slot's QSA keys are read per slot inside one stacked forward; the per-slot attention arm declines.
        .batch_per_slot_attn = false,
        .shared_readonly_state = true,
        .long_ctx = true,
        .moe = c.num_experts > 0,
        .spec_rollback = true,
        .images = c.qwen_vision,
        .video = c.qwen_vision,
    };
    p.shell = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.num_experts_per_tok,
        .moe_intermediate_size = c.moe_intermediate_size,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.contextCap(),
        .quant = .{ .bits = @intCast(@min(c.quant_bits, 255)), .group_size = c.quant_group_size, .mode = if (c.quant_bits == 0) .none else .affine },
        .gen = .{
            .temperature = c.gen_temperature orelse std.math.nan(f32),
            .top_p = c.gen_top_p orelse std.math.nan(f32),
            .min_p = c.gen_min_p orelse std.math.nan(f32),
            .top_k = if (c.gen_top_k) |k| @intCast(k) else -1,
            .enable_thinking = if (c.gen_enable_thinking) |t| (if (t) .on else .off) else .unset,
        },
        .bos = if (c.bos_token_id) |b| b else -1,
        .act_dtype = .bf16,
        .state_dtype = if (c.ssm_state_f32) .f32 else .bf16,
        .model_type = abi.Str.of(model_types[0]),
    };
    @memcpy(p.shell.eos[0..c.num_eos_tokens], c.eos_token_ids[0..c.num_eos_tokens]);
    p.shell.eos_count = c.num_eos_tokens;
    p.chat = .{
        .tool_dialect = .qwen_xml,
        .sampling_defaults = .{ .top_k = 20, .top_p = 0.95 },
    };
    p.cache = .{
        .layout_namespace = if (c.ssm_state_f32) abi.Str.of("gdn-f32-state-v1") else .{},
        .ssm_state = .{
            .entries = c.num_hidden_layers,
            .dtype = if (c.ssm_state_f32) .f32 else .bf16,
            // The aux buffers (PLE window, QSA keys) are written in place.
            .snapshot_policy = .in_place,
            .checkpoint_bytes = c.ssmCheckpointBytes(),
        },
    };
    p.spec = .{
        .mtp_layout = .in_checkpoint,
        .head_module_owned = true,
        .policy = .{ .drafter_ok = false, .mtp_group_solo = true, .draft_temperature = 1.0 },
    };
    p.placeholder_ids = .{
        .image = c.image_token_id,
        .video = c.video_token_id,
        .vision_start = c.vision_start_token_id,
        .vision_end = c.vision_end_token_id,
    };
    p.weights = .{ .prefix = abi.Str.of("language_model.model") };
    return p;
}
