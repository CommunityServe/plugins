//! copied_from: src/plugins/qwen4_exp/model_config.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The engine's `ModelConfig` from the family's `Config`, for the trunk module and the weight loader that still read
//! it: the same names one to one plus the arch's constants. Pinned field for field against the engine's own parse.

const std = @import("std");
const model = @import("../../sdk/config/model.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const Config = @import("config.zig").Config;

fn convert(comptime Dst: type, src: anytype) Dst {
    const Src = @TypeOf(src);
    if (Src == Dst) return src;
    return switch (@typeInfo(Src)) {
        .optional => if (src) |v| convert(@typeInfo(Dst).optional.child, v) else null,
        else => std.meta.stringToEnum(Dst, @tagName(src)).?,
    };
}

/// `dir` (the pack's folder, absent in a stub) places the n-gram table the way the engine's `parseConfig` does:
/// `ngram_table.bin` (or the file the config names) beside the weights, or the shards the index names, measured.
/// `ModelConfig.deinit` frees it.
pub fn toModelConfig(a: std.mem.Allocator, c: *const Config, dir: ?[]const u8) !model.ModelConfig {
    var m: model.ModelConfig = .{};
    inline for (@typeInfo(Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "embedded_ngram") or std.mem.eql(u8, name, "mrope_inv_freq") or std.mem.eql(u8, name, "expert_quant")) continue;
        @field(m, name) = convert(@TypeOf(@field(m, name)), @field(c, name));
    }
    m.model_type = "qwen4_exp";
    m.legacy.caps.long_ctx = true;
    m.weight_prefix = "language_model.model";
    m.norm_has_offset = false;
    m.has_final_norm = false; // the hyper-connection mixer replaces model.norm
    m.scale_embeddings = false;
    m.has_pre_ff_norm = false;
    m.has_qk_norm = true;
    m.hidden_act = .silu;
    m.has_sliding_window = false;
    m.attn_output_gate = true;
    m.kda_sigmoid_out_gate = true; // output_gate_type "sigmoid"
    m.rope_scaling_factor = 1.0;
    m.rope_local_base_freq = m.rope_theta;
    if (dir) |d| {
        m.ngram_table_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ d, m.ngramTableFile() });
        errdefer m.deinit(a);
        if (try qwen4_exp.inspectEmbedded(d, try model.qwen4EmbeddedSpec(&m))) |info| m.embedded_ple_payload_bytes = info.payload_bytes;
        if (c.expert_quant) m.exl3 = try sushiSpec(a, d, m.num_experts_per_tok);
    }
    return m;
}


/// A Sushi pack's `expert_quant` (config.json in `dir`): its EXL3 banks' codebook, window and rate.
fn sushiSpec(a: std.mem.Allocator, dir: []const u8, top_k: u32) !exl3.Spec {
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    const bytes = try d.readFileAlloc(io, "config.json", a, .limited(16 * 1024 * 1024));
    defer a.free(bytes);
    const doc = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.InvalidConfigJson;
    return exl3.parse(doc.value.object, top_k);
}

// ── tests ──

const testing = std.testing;
const abi = @import("mlx_serve").abi.plugin;
const fixture = @import("mlx_serve").sdk.fixture;
const config = @import("config.zig");

/// The plugin's parse of `files` (staged on disk for the engine), through `toModelConfig`, equals the engine's
/// `parseConfig` of the same folder in every field.
fn expectEngineParse(files: []const fixture.File) !void {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    var want = try model.parseConfigLegacy(io, testing.allocator, path);
    defer want.deinit(testing.allocator);
    const mem: fixture.MemDir = .{ .files = files };
    const pd = mem.packDir();
    var p: abi.Parsed = .{};
    const v = config.parse(&pd, .of(files[0].bytes), .{}, null, false, &p);
    if (!v.isOk()) {
        std.debug.print("parse: {s}\n", .{v.why.slice()});
        return error.Refused;
    }
    defer config.freeParsed(&p);
    var got = try toModelConfig(testing.allocator, config.of(&p), path);
    defer got.deinit(testing.allocator);
    try testing.expectEqualDeep(want, got);
    try testing.expectEqual(want.contextCap(), @as(u32, @intCast(p.shell.context_cap)));
    try testing.expectEqualSlices(u32, want.eosTokenSlice(), p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(want.ssmCheckpointBytes(), p.cache.ssm_state.checkpoint_bytes);
    try testing.expectEqual(want.attnCacheLayerCount(), config.of(&p).attnLayers());
    try testing.expectEqualStrings(want.cacheLayoutNamespace() orelse "", p.cache.layout_namespace.slice());
}

const iq_pack = @embedFile("../../fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json");
const iq_generation =
    \\{"bos_token_id": 248044, "do_sample": true, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
    \\ "temperature": 1.0, "top_k": 20, "top_p": 0.95}
;
const nvidia_pack = @embedFile("../../fixtures/model-configs/qwen4_exp-nvidia-fp4.json");
const nvidia_processor =
    \\{"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
    \\ "merge_size": 2, "processor_class": "Qwen3VLProcessor"}
;
