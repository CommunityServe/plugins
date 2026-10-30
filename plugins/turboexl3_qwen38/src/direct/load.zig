//! The direct model's load of a K pack: the weights (the pack's safetensors, read past the page cache when they
//! outweigh half the RAM), the trunk's layers bound by name, the trunk and expert formats' linears and banks through
//! turboexl3, the hyper-connection mixer, the n-gram hash and trellis table, the MTP head, and the trunk they make. The
//! copies' load chain for the K packs alone (`qwen4_exp/plugin.zig`'s `loadModel`, `trunk_load.zig`,
//! `engines/mlx/load/`'s layer binder and weight helpers, `qwen4_exp/load.zig`, `sdk/config/model.zig`'s load half and
//! `sdk/weights/`), into the copy's types (`Model`, `Trunk`, `Weights`): every array the forward reads comes out as the
//! copy's load makes it, in the same order. A shape no K pack has is refused by name.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("mlx_serve").abi.plugin;
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const log = @import("../mlx_serve/sdk/util/log.zig");
const crt = @import("../mlx_serve/sdk/util/crt.zig");
const system = @import("../mlx_serve/sdk/util/system.zig");
const model = @import("../mlx_serve/sdk/config/model.zig");
const exl3 = @import("../mlx_serve/sdk/quant/exl3/format.zig");
const transposeBf16Weight = @import("../mlx_serve/mlx/array.zig").transposeBf16Weight;
const bf16Scalar = @import("../mlx_serve/mlx/array.zig").bf16Scalar;
const KVCache = @import("../mlx_serve/engines/mlx/kv/cache.zig").KVCache;
const MoeLayerWeights = @import("../mlx_serve/engines/mlx/layer_weights.zig").MoeLayerWeights;
const LinearAttnWeights = @import("../mlx_serve/sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const MoeMlpWeights = @import("../mlx_serve/sdk/blocks/moe/weights.zig").MoeMlpWeights;
const HcWeights = @import("../mlx_serve/plugins/qwen4_exp/weights.zig").HcWeights;
const PleWeights = @import("../mlx_serve/plugins/qwen4_exp/weights.zig").PleWeights;
const qwen4_mod = @import("../mlx_serve/plugins/qwen4_exp/qwen4_exp.zig");
const NgramTable = qwen4_mod.NgramTable;
const mtp = @import("../mlx_serve/plugins/qwen4_exp/mtp.zig");
const trunk_formats = @import("../mlx_serve/plugins/qwen4_exp/trunk/formats.zig");
const checkQsaCompressRatio = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").checkQsaCompressRatio;
const mtp_nax_profile = @import("../mlx_serve/engines/mlx/spec/mtp_nax_profile.zig");
const Trunk = @import("../mlx_serve/plugins/qwen4_exp/trunk.zig").Trunk;
const fused_rows = @import("../mlx_serve/sdk/blocks/linear/fused_rows.zig");
const checkExl3Bank = @import("../mlx_serve/engines/mlx/load/exl3_bank.zig").checkExl3Bank;
const config = @import("config.zig");
const ple = @import("ple.zig");

/// The loaded model (the glue's) and the pack's weights.
const Model = @import("serve.zig").Model;
pub const Weights = model.Weights;
/// The trunk's load options (`trunk_load.Options`): the lossy MXFP8 head setting, and the pack's format blocks and host
/// table for the quant formats that serve its EXL3 weights.
const Options = struct { head_mxfp8: bool = false, pack: trunk_formats.Pack = .{} };

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

// ── the model ──

/// The model a K pack loads into: its config (the parse's, with the table's path this load places), the weights read
/// from the pack's shards, and the trunk they make, prepared. The plugin reads the shards itself: its `keep_key` keeps
/// none and its load hands no map on, so the engine's map (`Resident.take`) is no K path.
pub fn loadModel(ctx: *const abi.LoadCtx) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    if (ctx.weights != null) {
        log.err("[qwen4] the direct load reads the pack's shards itself; the host handed it the engine's map\n", .{});
        return error.DirectModelHostWeights;
    }
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .cfg = undefined, .weights = undefined, .trunk = undefined, .scratch = try KVCache.init(gpa, 0), .caps = ctx.parsed.caps };
    errdefer m.scratch.deinit();
    const dir = ctx.model_dir.slice();
    // As `toModelConfig` places it: the table's file beside the weights. `cfg` owns the path until `m.cfg` takes it.
    var cfg = config.of(ctx.parsed).*;
    cfg.ngram_table_path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, cfg.ngramTableFile() });
    var cfg_owns_path = true;
    errdefer if (cfg_owns_path) cfg.deinit(gpa);
    // The weights load resolves the prefix the checkpoint uses (the copy resolves it again for the engine's map).
    m.weights = try loadModelWeights(io, gpa, dir, &cfg);
    errdefer m.weights.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const spec: ?exl3.Spec = if (cfg.expert_quant) try expertSpec(arena.allocator(), ctx, cfg.num_experts_per_tok) else null;
    m.cfg = modelConfigOf(&cfg, spec);
    cfg_owns_path = false;
    errdefer m.cfg.deinit(gpa);
    m.trunk = try loadTrunk(gpa, &cfg, spec, &m.weights, .{
        // `lmHeadMxfp8On`: the setting, else on only for a quantized pack on an integrated CUDA GPU (never on Metal).
        .head_mxfp8 = headMxfp8Setting(ctx.settings_json.slice()) orelse false,
        .pack = try packOf(arena.allocator(), ctx),
    });
    m.trunk.prepare();
    return m;
}

/// The routed banks' EXL3 block of the pack's config.json, as the copies' `toModelConfig` reads it: refused by name
/// when this build has no kernels for its codebook, window or `topk` routed experts.
fn expertSpec(a: std.mem.Allocator, ctx: *const abi.LoadCtx, topk: u32) !exl3.Spec {
    const n = ctx.dir.fileBytes("config.json");
    if (n == 0 or n > 16 << 20) return error.ConfigUnreadable;
    var why: Verdict = .{};
    const text = ctx.dir.readFile("config.json", try a.alloc(u8, @intCast(n)), &why) orelse return error.ConfigUnreadable;
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .object) return error.InvalidConfigField;
    return exl3.parse(doc.object, topk);
}

/// `lm_head_mxfp8` as the load settings state it; null = the default.
fn headMxfp8Setting(settings_json: []const u8) ?bool {
    if (settings_json.len == 0) return null;
    const S = struct { lm_head_mxfp8: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    return parsed.value.lm_head_mxfp8;
}

/// The pack's config text and the format blocks its quant formats claim (`trunk_quant`, `expert_quant`), read through
/// the pack's folder into `a` and re-serialized as `quant_block.blocksOf` writes them, with the host table they load
/// with.
fn packOf(a: std.mem.Allocator, ctx: *const abi.LoadCtx) !trunk_formats.Pack {
    var p: trunk_formats.Pack = .{ .host = ctx.host, .model_dir = ctx.model_dir.slice() };
    const n = ctx.dir.fileBytes("config.json");
    if (n == 0) return p;
    if (n > 16 << 20) return error.ConfigTooLarge;
    var why: Verdict = .{};
    p.config_json = ctx.dir.readFile("config.json", try a.alloc(u8, @intCast(n)), &why) orelse return error.ConfigUnreadable;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, p.config_json, .{}) catch return p;
    if (doc != .object) return p;
    if (doc.object.get("expert_quant")) |v| if (v == .object) {
        p.expert_quant = try std.json.Stringify.valueAlloc(a, v, .{});
    };
    if (doc.object.get("trunk_quant")) |v| if (v == .object) {
        p.trunk_quant = try std.json.Stringify.valueAlloc(a, v, .{});
    };
    return p;
}

/// The copied trunk's `ModelConfig` from the direct config, as `toModelConfig` builds it: the same names one to one,
/// the routed banks' EXL3 block (`spec`, in place of the config's flag) and the legacy parse's long-context caps.
fn modelConfigOf(c: *const config.Config, spec: ?exl3.Spec) model.ModelConfig {
    var m: model.ModelConfig = .{ .exl3 = spec };
    inline for (@typeInfo(config.Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "expert_quant")) continue;
        @field(m, name) = convert(@TypeOf(@field(m, name)), @field(c, name));
    }
    m.legacy.caps.long_ctx = true;
    return m;
}

fn convert(comptime Dst: type, src: anytype) Dst {
    const Src = @TypeOf(src);
    if (Src == Dst) return src;
    return switch (@typeInfo(Src)) {
        .optional => if (src) |v| convert(@typeInfo(Dst).optional.child, v) else null,
        else => std.meta.stringToEnum(Dst, @tagName(src)).?,
    };
}

// ── the weights ──

/// The text model's weights (`loadModelWeights`, a K pack's arms): every shard the index names, past the page cache
/// when the pack outweighs half the RAM, under the prefix the checkpoint uses, its stream norms folded from their
/// stored deltas. `admitExl3Backend` passes: the plugin loads on an M5-class GPU only, whose Metal kernels serve both
/// formats.
fn loadModelWeights(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, cfg: *config.Config) !Weights {
    var weights = try loadWeightsOpt(io, a, model_dir, .{ .nocache = nocacheLoad(io, model_dir) });
    errdefer weights.deinit();
    resolveWeightPrefix(cfg, &weights);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try resolveAndFoldQwen4Norms(cfg, &weights, s, model_dir);
    return weights;
}

/// Whether the weights load reads the shards past the page cache (`nocache`): when what it loads is more than half the
/// machine's RAM (`nocacheFor`), since MLX's own reader leaves each shard's pages cached beside the arrays and such a
/// pack peaks near twice its weights. A K pack's n-gram table is its own file, outside the shards.
fn nocacheLoad(io: std.Io, model_dir: []const u8) bool {
    return nocacheFor(modelDiskBytes(io, model_dir), system.getTotalMemBytes());
}

/// `weights` bytes loaded on a machine of `ram` bytes read past the page cache when they are more than half of it (an
/// unknown RAM size, 0, never does).
fn nocacheFor(weights: u64, ram: u64) bool {
    return ram > 0 and weights > ram / 2;
}

/// The bytes of the shards the index names, or of every `*.safetensors` in the folder without one (preflight's
/// `modelDiskBytes`).
fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (indexShards(io, dir, a)) |names| {
        var total: u64 = 0;
        for (names) |n| total += fileBytes(io, dir, n);
        return total;
    }
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if ((e.kind == .file or e.kind == .sym_link) and std.mem.endsWith(u8, e.name, ".safetensors")) total += fileBytes(io, dir, e.name);
    }
    return total;
}

fn fileBytes(io: std.Io, dir: std.Io.Dir, name: []const u8) u64 {
    const st = dir.statFile(io, name, .{}) catch return 0;
    return if (st.kind == .file) @intCast(st.size) else 0;
}

/// The distinct shard names of the index's `weight_map`, each inside the folder; null without one.
fn indexShards(io: std.Io, dir: std.Io.Dir, a: std.mem.Allocator) ?[]const []const u8 {
    const text = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(256 << 20)) catch return null;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return null;
    if (doc != .object) return null;
    const map = doc.object.get("weight_map") orelse return null;
    if (map != .object) return null;
    var seen: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (map.object.values()) |v| {
        if (v != .string or std.mem.indexOf(u8, v.string, "..") != null or std.fs.path.isAbsolute(v.string)) continue;
        seen.put(a, v.string, {}) catch return null;
    }
    return seen.keys();
}

/// tensor -> shard from the index's `weight_map`, strings owned by `arena`.
const Owners = std.StringHashMapUnmanaged([]const u8);

/// How a shard loads: past the page cache or not, and which of its tensors another shard owns.
const LoadOpts = struct { nocache: bool = false, shard: []const u8 = "", owners: ?*const Owners = null };

/// Every `*.safetensors` the index names (every one, without an index) into one map, lazily on the CPU stream
/// (`loadWeightsOpt` / `loadWeightsFromOpenDir`): a tensor two shards carry loads from the shard the index names. A K
/// pack holds no byte-wise affine word (`viewByteAffineWeights`' Flint export): the trunk format refuses one by name.
fn loadWeightsOpt(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, opts: LoadOpts) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var weights = Weights.init(a);
    errdefer weights.deinit();

    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);

    // The index names the shards: anything else (a K pack's n-gram table) is not the weights'.
    var referenced = indexShardSet(io, dir);
    defer if (referenced) |*r| freeShardSet(r);

    var owners_arena = std.heap.ArenaAllocator.init(a);
    defer owners_arena.deinit();
    var owners: ?Owners = indexOwners(io, owners_arena.allocator(), dir);

    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Regular files and symlinks: a Hugging Face snapshot links each shard into its blobs.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) {
            log.warn("skipping {s}: not named by model.safetensors.index.json\n", .{entry.name});
            continue;
        };

        const path_slice = try std.fmt.allocPrint(a, "{s}/{s}", .{ model_dir, entry.name });
        defer a.free(path_slice);
        const path = try a.dupeSentinel(u8, path_slice, 0);
        defer a.free(path);

        var shard_opts = opts;
        shard_opts.shard = entry.name;
        if (owners) |*o| shard_opts.owners = o;
        try loadSafetensorsFile(a, &weights, path, s, shard_opts);
        file_count += 1;
    }

    // A folder whose shards never landed (an interrupted download) holds no weights.
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found) — the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model (e.g. `mlx-serve pull <model>`) or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }
    return weights;
}

/// One shard's tensors into `weights` (`loadSafetensorsFile`, a K pack's options): past the page cache with
/// `opts.nocache`, without the tensors another shard owns and the vision tower's. The copy narrows an f16 side tensor
/// or 1-D table to bf16; a K pack's f16 tensors are the formats' operands and the norms the fold casts, so another is
/// refused by name.
fn loadSafetensorsFile(a: std.mem.Allocator, weights: *Weights, path: [*:0]const u8, s: mlx.mlx_stream, opts: LoadOpts) !void {
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);

    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);

    // The Darwin test is the condition's last, comptime operand: off Darwin the reader is never compiled.
    if (opts.nocache and comptime builtin.os.tag.isDarwin())
        try nocache.loadSafetensors(&tensor_map, &meta_map, path, s)
    else
        try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);

    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();

        const ret = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (ret != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }

        const key_str = std.mem.span(key.?);

        const foreign = if (opts.owners) |o| (if (o.get(key_str)) |owner| !std.mem.eql(u8, owner, opts.shard) else false) else false;
        if (foreign or !keepsTensor(key_str)) {
            _ = mlx.mlx_array_free(value);
            continue;
        }
        if (narrowsF16(key_str, mlx.mlx_array_ndim(value), mlx.mlx_array_dtype(value))) {
            log.err("[qwen4] {s} is an f16 table the K packs do not have (the copied load narrows it to bf16)\n", .{key_str});
            _ = mlx.mlx_array_free(value);
            return error.DirectModelF16Table;
        }

        const owned_key = try a.dupe(u8, key_str);
        try weights.map.put(owned_key, value);
    }
}

/// A tensor the text forward reads: the Qwen3-VL tower's (`model.visual.`, as the K packs spell it) is left out, as
/// `shouldKeepWeightKey` leaves every tower out with vision off; a K pack holds no other.
fn keepsTensor(key: []const u8) bool {
    return !std.mem.startsWith(u8, key, "model.visual.");
}

/// Whether the copied load narrows this f16 tensor to bf16 (`narrowsLoadedF16` under a K pack's options): an f16 side
/// tensor or 1-D table that is neither a trunk format's operand (`keepsF16Operand`) nor a stream norm the fold casts.
fn narrowsF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16 or keepsF16Operand(key, ndim) or qwen4NormNeedsFold(key)) return false;
    return std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases") or ndim == 1;
}

/// An EXL3 trunk pack's f16 tensor a kernel reads in f16: a trunk format's 1-D `.scales` / `.biases` (suh / svh), and
/// the trellis n-gram table's per-head bias, which its rows add in f16.
fn keepsF16Operand(key: []const u8, ndim: usize) bool {
    if (std.mem.endsWith(u8, key, ".ple.ple_embedding.head_bias")) return true;
    return ndim == 1 and (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases"));
}

/// The set of shard basenames in the index's `weight_map`, or null when there is no usable index.
fn indexShardSet(io: std.Io, dir: std.Io.Dir) ?std.StringHashMapUnmanaged(void) {
    const a = std.heap.page_allocator;
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024)) catch return null;
    defer a.free(raw);
    var parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const wm = parsed.value.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var set: std.StringHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string) continue;
        if (set.contains(v.string)) continue;
        const key = a.dupe(u8, v.string) catch continue;
        set.put(a, key, {}) catch {
            a.free(key);
            continue;
        };
    }
    // An index none of whose shards exist is stale (the repo was re-sharded after it was written): the folder is the
    // set.
    var any_present = false;
    var keys = set.keyIterator();
    while (keys.next()) |k| {
        _ = dir.statFile(io, k.*, .{}) catch continue;
        any_present = true;
        break;
    }
    if (!any_present) {
        log.warn("model.safetensors.index.json names no shard in this directory; loading every *.safetensors instead\n", .{});
        freeShardSet(&set);
        return null;
    }
    return set;
}

/// Frees a set from `indexShardSet`.
fn freeShardSet(set: *std.StringHashMapUnmanaged(void)) void {
    var keys = set.keyIterator();
    while (keys.next()) |k| std.heap.page_allocator.free(k.*);
    set.deinit(std.heap.page_allocator);
}

/// tensor -> shard from the index's `weight_map`, strings owned by `arena`; an owner shard not on disk claims nothing.
fn indexOwners(io: std.Io, arena: std.mem.Allocator, dir: std.Io.Dir) ?Owners {
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", arena, .limited(64 * 1024 * 1024)) catch return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{}) catch return null;
    if (parsed != .object) return null;
    const wm = parsed.object.get("weight_map") orelse return null;
    if (wm != .object) return null;
    var out: Owners = .empty;
    var it = wm.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) continue;
        _ = dir.statFile(io, e.value_ptr.string, .{}) catch continue;
        out.put(arena, e.key_ptr.*, e.value_ptr.string) catch return null;
    }
    return out;
}

/// An MLX IO reader whose reads bypass the page cache (`sdk/weights/nocache_reader.zig`, mlx-stream's, MIT). MLX's own
/// safetensors reader is a plain open + pread, so a load leaves the file's pages cached beside the array buffers: a
/// pack near the RAM size then has its weights compressed under the cache. This one opens the file with F_NOCACHE and
/// read-ahead off; `mlx_load_safetensors_reader` reads the header through `read` and each tensor through
/// `read_at_offset`. Darwin only. A failed read panics naming the file, as MLX's own reader throws.
const nocache = struct {
    /// One read's staging buffer: page-aligned (the page allocator), a multiple of every page size.
    const stage_bytes: usize = 8 << 20;

    /// The reader's state, freed by MLX (`free`) once no array it loaded needs it. C allocator: MLX may free it from an
    /// IO thread.
    const Desc = struct {
        fd: std.c.fd_t,
        size: u64,
        /// The sequential position (`read`, `seek`, `tell`: the header parse).
        pos: u64 = 0,
        label: [:0]u8,

        fn open(path: [:0]const u8) !*Desc {
            if (comptime !builtin.os.tag.isDarwin()) return error.NoCacheUnsupported;
            const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.FileNotFound;
            errdefer _ = std.c.close(fd);
            if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0 or std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
            const size = std.c.lseek(fd, 0, std.c.SEEK.END);
            if (size < 0) return error.NoCacheStat;
            const a = std.heap.c_allocator;
            const d = try a.create(Desc);
            errdefer a.destroy(d);
            d.* = .{ .fd = fd, .size = @intCast(size), .label = try a.dupeSentinel(u8, path, 0) };
            return d;
        }

        /// `buf.len` bytes at `off` through a page-aligned stage: macOS honours F_NOCACHE only for page-aligned reads
        /// (offset, length and destination), and safetensors offsets and MLX buffers are not aligned.
        fn readAt(d: *const Desc, buf: []u8, off: u64) void {
            if (buf.len == 0) return;
            const page: u64 = std.heap.pageSize();
            const stage = std.heap.page_allocator.alloc(u8, stage_bytes) catch
                std.debug.panic("nocache reader: {s}: no {d} B staging buffer", .{ d.label, stage_bytes });
            defer std.heap.page_allocator.free(stage);
            var pos = off;
            const end = off + buf.len;
            var out: usize = 0;
            while (pos < end) {
                const a0 = pos - pos % page;
                const want_end = @min(end, a0 + stage_bytes);
                const need: usize = @intCast(want_end - a0);
                const len = std.mem.alignForward(usize, need, @intCast(page));
                var got: usize = 0;
                while (got < need) {
                    const n = std.c.pread(d.fd, stage[got..].ptr, len - got, @intCast(a0 + got));
                    if (n < 0) {
                        if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
                        std.debug.panic("nocache reader: {s}: pread of {d} B at {d} failed, errno {d}", .{ d.label, len - got, a0 + got, std.c._errno().* });
                    }
                    if (n == 0) std.debug.panic("nocache reader: {s}: short read at {d} ({d} B file)", .{ d.label, a0 + got, d.size });
                    got += @intCast(n);
                }
                const s0: usize = @intCast(pos - a0);
                @memcpy(buf[out..][0 .. need - s0], stage[s0..need]);
                out += need - s0;
                pos = want_end;
            }
        }
    };

    fn of(ctx: ?*anyopaque) *Desc {
        return @ptrCast(@alignCast(ctx.?));
    }

    fn isOpen(ctx: ?*anyopaque) callconv(.c) bool {
        return of(ctx).fd >= 0;
    }

    fn tell(ctx: ?*anyopaque) callconv(.c) usize {
        return @intCast(of(ctx).pos);
    }

    fn seek(ctx: ?*anyopaque, off: i64, whence: c_int) callconv(.c) void {
        const d = of(ctx);
        const base: i64 = switch (whence) {
            0 => 0,
            1 => @intCast(d.pos),
            2 => @intCast(d.size),
            else => std.debug.panic("nocache reader: {s}: seek whence {d}", .{ d.label, whence }),
        };
        d.pos = @intCast(base + off);
    }

    fn read(ctx: ?*anyopaque, data: [*]u8, n: usize) callconv(.c) void {
        const d = of(ctx);
        d.readAt(data[0..n], d.pos);
        d.pos += n;
    }

    fn readAtOffset(ctx: ?*anyopaque, data: [*]u8, n: usize, off: usize) callconv(.c) void {
        of(ctx).readAt(data[0..n], off);
    }

    fn write(ctx: ?*anyopaque, _: [*]const u8, _: usize) callconv(.c) void {
        std.debug.panic("nocache reader: {s}: write on a reader", .{of(ctx).label});
    }

    fn label(ctx: ?*anyopaque) callconv(.c) [*:0]const u8 {
        return of(ctx).label.ptr;
    }

    fn free(ctx: ?*anyopaque) callconv(.c) void {
        const d = of(ctx);
        _ = std.c.close(d.fd);
        std.heap.c_allocator.free(d.label);
        std.heap.c_allocator.destroy(d);
    }

    const vtable: mlx.mlx_io_vtable = .{ .is_open = isOpen, .good = isOpen, .tell = tell, .seek = seek, .read = read, .read_at_offset = readAtOffset, .write = write, .label = label, .free = free };

    /// `path`'s tensors past the page cache, lazily as `mlx_load_safetensors` loads them.
    fn loadSafetensors(tensors: *mlx.mlx_map_string_to_array, meta: *mlx.mlx_map_string_to_string, path: [*:0]const u8, s: mlx.mlx_stream) !void {
        const d = try Desc.open(std.mem.span(path));
        const r = mlx.mlx_io_reader_new(d, vtable);
        defer _ = mlx.mlx_io_reader_free(r);
        try mlx.check(mlx.mlx_load_safetensors_reader(tensors, meta, r, s));
    }
};

/// The nesting the checkpoint's trunk ships under, when it holds none of its weights under the config's: flat,
/// mlx-community's re-nest, or meta's VL original (`resolveWeightPrefix`; the direct config's is the re-nest).
fn resolveWeightPrefix(cfg: *config.Config, weights: *const Weights) void {
    const candidates = [_][]const u8{ "language_model.model", "model.language_model", "model" };
    if (hasWeightsUnder(weights, cfg.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, cfg.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        cfg.weight_prefix = p;
        return;
    }
}

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// The stream norms a K pack stores as deltas from one (`qwen4_norm_convention: delta`): each norm weight
/// (`qwen4NormNeedsFold`) becomes 1 + w in bf16, lazily on the CPU stream `s` (`resolveAndFoldQwen4Norms`'s delta arm).
/// A pack that stores them folded, or says neither, is another pack's (the copy infers or assumes its convention).
fn resolveAndFoldQwen4Norms(cfg: *config.Config, weights: *Weights, s: mlx.mlx_stream, model_dir: []const u8) !void {
    if (cfg.qwen4_norm_convention == null or cfg.qwen4_norm_convention.? != .delta) {
        log.err("[qwen4] {s} does not mark its stream norms as deltas (qwen4_norm_convention: delta), as the K packs do\n", .{model_dir});
        return error.DirectModelNormConvention;
    }
    var it = weights.map.iterator();
    while (it.next()) |entry| {
        if (!qwen4NormNeedsFold(entry.key_ptr.*)) continue;
        const value = entry.value_ptr.*;
        entry.value_ptr.* = try foldQwen4Norm(value, s);
        _ = mlx.mlx_array_free(value);
    }
}

/// `1 + w` in bf16, lazily on `s`.
fn foldQwen4Norm(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try mlx.check(mlx.mlx_add(&shifted, value, one, s));
    var folded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(folded);
    try mlx.check(mlx.mlx_astype(&folded, shifted, .bfloat16, s));
    return folded;
}

fn qwen4NormNeedsFold(key: []const u8) bool {
    const suffixes = [_][]const u8{
        "hc_norm.weight",      "q_norm.weight",         "k_norm.weight",        "q_layernorm.weight",           "k_layernorm.weight",
        "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

// ── the trunk ──

/// The trunk of a K pack's loaded map (`trunk_load.load`, its EXL3 arms): the trunk format's linears, the embedding and
/// head, the layers bound by name (each GDN layer's a | b joined) and their expert banks, one batched eval of every
/// weight, the final mixer, the n-gram hash and trellis table, and the MTP head. The trunk keeps its own copy of `cfg`
/// (the copied trunk's `ModelConfig`); `weights` must outlive it (the trunk borrows its arrays).
fn loadTrunk(a: std.mem.Allocator, cfg: *const config.Config, spec: ?exl3.Spec, weights: *Weights, opts: Options) !Trunk {
    const s = mlx.gpuStream();
    // The formats, the MTP head and the copied trunk read the engine's config.
    const mc = modelConfigOf(cfg, spec);
    // The pack's quant formats: the trunk format's linears now, the expert format's banks once the layers are bound.
    var fmts: trunk_formats.Loading = undefined;
    try fmts.begin(a, &mc, weights, opts.pack, s);
    // On an error the formats close before their weights go (`Loading.abort`'s order).
    errdefer fmts.formats.deinit(a);
    defer fmts.end();
    const prefix = cfg.weight_prefix;
    var name_buf: [256]u8 = undefined;

    // Dense tables (float dtype) carry no scales; the nominal affine block's quantized linears carry biases.
    const bias_mandatory = cfg.quant_bits > 0 and cfg.quant_mode.hasBiases();
    const emb_w = getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "weight") orelse {
        log.err("MISSING WEIGHT: {s}.embed_tokens.weight\n", .{prefix});
        return error.MissingWeight;
    };
    const emb_dense = floatDtypeTable(mlx.mlx_array_dtype(emb_w));
    const emb_s = if (emb_dense) mlx.mlx_array_new() else getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "scales") orelse {
        log.err("MISSING WEIGHT: {s}.embed_tokens.scales\n", .{prefix});
        return error.MissingWeight;
    };
    const emb_b = if (emb_dense) mlx.mlx_array_new() else getNamedWeight(weights, &name_buf, prefix, "embed_tokens", "biases") orelse blk: {
        if (bias_mandatory) {
            log.err("MISSING WEIGHT: {s}.embed_tokens.biases\n", .{prefix});
            return error.MissingWeight;
        }
        break :blk mlx.mlx_array_new();
    };

    // A nested decoder's head is a sibling of `model`: a K pack's is a trunk-format linear, untied.
    const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
    const lm_head_w = getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.weight", lm_prefix) orelse {
        log.err("MISSING WEIGHT: lm_head.weight\n", .{});
        return error.MissingWeight;
    };
    const head_dense = floatDtypeTable(mlx.mlx_array_dtype(lm_head_w));
    const lm_head_s = if (head_dense) mlx.mlx_array_new() else try getWeightFmt(weights, &name_buf, "{s}.lm_head.scales", lm_prefix);
    const lm_head_b = if (head_dense)
        mlx.mlx_array_new()
    else if (bias_mandatory)
        try getWeightFmt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix)
    else
        getWeightFmtOpt(weights, &name_buf, "{s}.lm_head.biases", lm_prefix) orelse mlx.mlx_array_new();
    if (opts.head_mxfp8) {
        // The lossy MXFP8 head quantizes a dense bf16 head at load, which no K pack has.
        if (lm_head_s.ctx == null) {
            log.err("[lm-head] mxfp8 (lossy): the direct model quantizes no head at load\n", .{});
            return error.DirectModelHeadMxfp8;
        }
    }

    // `unservedLayout` names nothing here: the config fields it reads are the direct config's arch constants, and the
    // K binder binds none of the layer fields it checks.
    const ml = try initMoeLayers(a, cfg, spec, weights, &name_buf, s);
    var owned: std.ArrayList(mlx.mlx_array) = .fromOwnedSlice(ml.owned_bf16);
    errdefer {
        for (owned.items) |arr| _ = mlx.mlx_array_free(arr);
        owned.deinit(a);
    }
    errdefer a.free(ml.moe_layers);
    // The joined a | b GEMM: an EXL3 trunk pack's.
    for (ml.moe_layers, 0..) |*lw, i| if (lw.attn == .linear) try joinInPlace(&lw.attn.linear, weights, &name_buf, prefix, @intCast(i), &owned, a, s);
    for (ml.moe_layers, 0..) |*lw, i| try fmts.banks(prefix, @intCast(i), &lw.mlp.moe, cfg.hidden_size, cfg.moe_intermediate_size);

    const ones_hidden = blk: {
        const one_val = bf16Scalar(1.0, s);
        defer _ = mlx.mlx_array_free(one_val);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&o, &[_]c_int{@intCast(cfg.hidden_size)}, 1, one_val, .bfloat16, s));
        break :blk o;
    };
    errdefer _ = mlx.mlx_array_free(ones_hidden);

    // One batched eval of every weight the trunk reads.
    {
        const all_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(all_vec);
        inline for (.{ emb_w, emb_s, emb_b, lm_head_w, lm_head_s, lm_head_b }) |arr| {
            if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(all_vec, arr);
        }
        for (ml.moe_layers) |lw| {
            if (lw.hc_attn) |*hw| appendStructArrays(all_vec, hw);
            if (lw.hc_mlp) |*hw| appendStructArrays(all_vec, hw);
            if (lw.ple) |*pw| appendStructArrays(all_vec, pw);
            appendStructArrays(all_vec, &lw.mlp.moe);
            switch (lw.attn) {
                .full => |fa| appendStructArrays(all_vec, &fa),
                .linear => |la| appendStructArrays(all_vec, &la),
            }
        }
        try mlx.check(mlx.mlx_eval(all_vec));
    }

    // The final hyper-connection read and the module-owned n-gram hash + mmapped table.
    var mb: [160]u8 = undefined;
    const mixer = try loadHcWeights(weights, std.fmt.bufPrint(&mb, "{s}.hyper_connection_mixer", .{prefix}) catch unreachable, false, cfg.hc_count, cfg.hidden_size, &owned, a, s);
    try checkQsaCompressRatio(cfg.indexer_compress_ratio);
    const st = try a.create(qwen4_mod.Qwen4State);
    errdefer a.destroy(st);
    st.* = .{
        // The `0` is the PLE's ordinal among the config's injection points, not `ple_layer_idx`.
        .hash = try qwen4_mod.NgramHash.init(cfg.vocab_size, cfg.ngram_size, cfg.heads_per_ngram, cfg.ngram_vocab_base, cfg.ngram_vocab_divisor, cfg.ngram_seed, 0, cfg.ngram_eos),
        .table = try openTable(cfg.ngram_table_path orelse return error.MissingNgramTable),
    };
    errdefer st.deinit();
    if (st.table.rows != st.hash.total_rows or st.table.dim * st.hash.n_heads != cfg.ple_embed_dim) {
        log.err("[qwen4] ngram_table.bin geometry {d}x{d} does not match the config ({d} rows, {d} heads x dim)\n", .{ st.table.rows, st.table.dim, st.hash.total_rows, st.hash.n_heads });
        return error.NgramTableMismatch;
    }
    // A K pack's table is a trellis (`exl3_ngram_trellis`): the affine and bf16 tables are other packs'.
    if (st.table.trellis_k == 0) {
        log.err("[qwen4] {s} is not a trellis n-gram table (exl3_ngram_trellis), as a K pack's is\n", .{cfg.ngram_table_path.?});
        return error.DirectModelNgramTableNotTrellis;
    }
    try trellisHeadBias(&st.table, weights, st.hash.n_heads);
    var weights_bytes: usize = 0;
    _ = mlx.mlx_get_active_memory(&weights_bytes);
    // `ple_gpu.load`'s answer for a trellis table on a Metal GPU: the host gather (its kernel reads affine and bf16
    // rows), `st.gpu` null. `calibrateArm` measures nothing on a trellis table (its pooled arm decodes on the workers).
    // The weights load just evicted the table from the page cache: the warm reads it back when both fit.
    if (warmFits(weights_bytes, &st.table)) startWarm(&st.table);
    var head = try mtp.loadQwen4Mtp(a, mc, weights, &name_buf, s);
    errdefer if (head) |*h| h.deinit(a);
    if (head) |*h| {
        switch (h.layer.mlp) {
            .moe => |*mw| try fmts.banks(mtp.headPrefix(weights) orelse "", 0, mw, 0, 0),
            .dense => {},
        }
    }

    // The MTP profile's inputs, which the policy reads (its calibrated model is another family's). No ternary kernels:
    // a K pack's nominal block is 4-bit at group 64, never the ternary 2-bit at 128.
    const head_n: c_int = blk: {
        const sh = mlx.getShape(lm_head_w);
        break :blk if (sh.len == 2) sh[0] else 0;
    };
    const calibrated = mtp_nax_profile.mtpNaxCalibratedModelFrom(&mc, head_n) and cfg.quant_group_size == 64 and cfg.quant_mode == .affine;
    const uniform_bits = cfg.quant_bits == 4 or cfg.quant_bits == 6 or cfg.quant_bits == 8;
    return .{
        .config = mc,
        .s = s,
        .allocator = a,
        .emb_w = emb_w,
        .emb_s = emb_s,
        .emb_b = emb_b,
        .lm_head_w = lm_head_w,
        .lm_head_s = lm_head_s,
        .lm_head_b = lm_head_b,
        .formats = fmts.formats,
        .moe_layers = ml.moe_layers,
        .moe_owned_bf16 = try owned.toOwnedSlice(a),
        .qwen4_mixer = mixer,
        .qwen4 = st,
        .qwen4_mtp = head,
        .ones_hidden = ones_hidden,
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(cfg.rms_norm_eps),
        .mtp_uniform_affine_trunk = calibrated and uniform_bits and mtp_nax_profile.mtpNaxUniformAffineTrunkFrom(&mc, ml.moe_layers),
        .mtp_oqe_affine_trunk = calibrated and cfg.quant_bits == 4 and mtp_nax_profile.mtpNaxOqeAffineTrunkFrom(&mc, ml.moe_layers),
    };
}

// ── the layers ──

/// The layers a K pack binds (`initMoeLayers`, its qwen4_exp arms): each layer's two hyper-connection blocks and, on
/// its one layer, the PLE; a GatedDeltaNet layer's separate projections or a full-attention layer's q/k/v/o with the
/// QSA indexer; the router, the routed EXL3 banks and the gated shared expert. Dense bf16 projections are
/// pre-transposed into `owned_bf16`; every other array stays the map's. The recurrent state lives on each slot: no
/// template entries.
fn initMoeLayers(a: std.mem.Allocator, cfg: *const config.Config, exl3_spec: ?exl3.Spec, weights: *Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !struct { moe_layers: []MoeLayerWeights, owned_bf16: []mlx.mlx_array } {
    // Every layer of a K pack routes EXL3 banks: a dense MLP and the affine banks are other packs'.
    if (!cfg.isMoe()) {
        log.err("[qwen4] the pack has no routed experts; a K pack's every layer does\n", .{});
        return error.DirectModelHasNoDenseMlp;
    }
    const spec = exl3_spec orelse {
        log.err("[qwen4] the pack's routed experts are not EXL3 (no expert_quant block), as a K pack's are\n", .{});
        return error.DirectModelNeedsExl3Experts;
    };
    const prefix = cfg.weight_prefix;
    const moe_layers = try a.alloc(MoeLayerWeights, cfg.num_hidden_layers);
    errdefer a.free(moe_layers);
    var owned_bf16: std.ArrayList(mlx.mlx_array) = .empty;
    errdefer {
        for (owned_bf16.items) |arr| _ = mlx.mlx_array_free(arr);
        owned_bf16.deinit(a);
    }
    const ple_layer = cfg.pleLocalLayer();

    for (0..cfg.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const lw = &moe_layers[i];
        // No layer norms: the hyper-connection read normalizes. `alloc` skips the defaults, so the layer is set whole.
        lw.* = .{ .input_norm = mlx.mlx_array_new(), .post_attn_norm = mlx.mlx_array_new(), .is_linear = cfg.isLinearLayer(li), .attn = undefined, .mlp = undefined };
        var hb: [96]u8 = undefined;
        lw.hc_attn = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.attn_hyper_connection", .{ prefix, li }) catch unreachable, true, cfg.hc_count, cfg.hidden_size, &owned_bf16, a, s);
        lw.hc_mlp = try loadHcWeights(weights, std.fmt.bufPrint(&hb, "{s}.layers.{d}.mlp_hyper_connection", .{ prefix, li }) catch unreachable, true, cfg.hc_count, cfg.hidden_size, &owned_bf16, a, s);
        if (ple_layer == li) lw.ple = try loadPleWeights(weights, name_buf, prefix, li, cfg.hc_count, cfg.hidden_size, &owned_bf16, a, s);

        if (lw.is_linear) {
            // Separate qkv, z, a and b (a pack of the combined qkvz | ba form misses these names): qkv, z and out_proj
            // through the trunk format, a and b dense bf16, transposed here for a plain matmul.
            lw.attn = .{ .linear = .{
                .qkv_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.weight"),
                .qkv_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.scales") orelse mlx.mlx_array_new(),
                .qkv_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_qkv.biases") orelse mlx.mlx_array_new(),
                .z_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_z.weight"),
                .z_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.scales") orelse mlx.mlx_array_new(),
                .z_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_z.biases") orelse mlx.mlx_array_new(),
                .a_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_a.weight"),
                .a_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.scales") orelse mlx.mlx_array_new(),
                .a_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_a.biases") orelse mlx.mlx_array_new(),
                .b_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.in_proj_b.weight"),
                .b_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.scales") orelse mlx.mlx_array_new(),
                .b_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.in_proj_b.biases") orelse mlx.mlx_array_new(),
                .conv1d_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.conv1d.weight"),
                .A_log = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.A_log"),
                .dt_bias = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.dt_bias"),
                .norm_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.norm.weight"),
                .out_w = try getLayerWeight(weights, name_buf, prefix, li, "linear_attn.out_proj.weight"),
                .out_s = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.scales") orelse mlx.mlx_array_new(),
                .out_b = getLayerWeightOpt(weights, name_buf, prefix, li, "linear_attn.out_proj.biases") orelse mlx.mlx_array_new(),
            } };
            const la = &lw.attn.linear;
            try maybeTransposeForBf16(&la.qkv_w, la.qkv_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&la.z_w, la.z_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&la.a_w, la.a_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&la.b_w, la.b_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&la.out_w, la.out_s, &owned_bf16, a, s);
            if (la.qkv_s.ctx != null) {
                var parts = [_][3]*mlx.mlx_array{ .{ &la.qkv_w, &la.qkv_s, &la.qkv_b }, .{ &la.z_w, &la.z_s, &la.z_b }, .{ &la.a_w, &la.a_s, &la.a_b }, .{ &la.b_w, &la.b_s, &la.b_b } };
                const names = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b" };
                la.in = try fused_rows.fuseRowsInPlace(&parts, 0, &names, weights, name_buf, prefix, li, &owned_bf16, a, s);
            }
        } else {
            // q, k, v and o through the trunk format (scales mandatory under the nominal block), the arch's QK norms.
            const k_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.k_proj.scales", cfg.quant_bits);
            const k_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_proj.weight");
            const k_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.k_proj.biases", cfg);
            const v_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.v_proj.weight");
            const v_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.v_proj.scales", cfg.quant_bits);
            const v_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.v_proj.biases", cfg);
            lw.attn = .{ .full = .{
                .q_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_proj.weight"),
                .q_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.q_proj.scales", cfg.quant_bits),
                .q_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.q_proj.biases", cfg),
                .k_w = k_w,
                .k_s = k_s,
                .k_b = k_b,
                .v_w = v_w,
                .v_s = v_s,
                .v_b = v_b,
                .o_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.o_proj.weight"),
                .o_s = try getLayerScaleOrEmpty(weights, name_buf, prefix, li, "self_attn.o_proj.scales", cfg.quant_bits),
                .o_b = try getLayerBias(weights, name_buf, prefix, li, "self_attn.o_proj.biases", cfg),
                .q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.q_norm.weight"),
                .k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.k_norm.weight"),
            } };
            const fa = &lw.attn.full;
            try maybeTransposeForBf16(&fa.q_w, fa.q_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&fa.k_w, fa.k_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&fa.o_w, fa.o_s, &owned_bf16, a, s);
            try maybeTransposeForBf16(&fa.v_w, fa.v_s, &owned_bf16, a, s);
            if (cfg.indexer_n_heads > 0) {
                fa.idx_qk_w = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.weight");
                fa.idx_qk_s = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.scales") orelse mlx.mlx_array_new();
                fa.idx_qk_b = getLayerWeightOpt(weights, name_buf, prefix, li, "self_attn.indexer.index_qk_proj.biases") orelse mlx.mlx_array_new();
                fa.idx_q_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.q_layernorm.weight");
                fa.idx_k_norm = try getLayerWeight(weights, name_buf, prefix, li, "self_attn.indexer.k_layernorm.weight");
                try maybeTransposeForBf16(&fa.idx_qk_w, fa.idx_qk_s, &owned_bf16, a, s);
            }
            // No q | k | v join: the binder joins them for a row-exact arch on NAX (`rowExactArch`), which qwen4_exp is
            // not (no `exact_drafted` cap, not a qwen3_5 model).
        }

        // The routed banks bind their trellis, suh and svh in the affine slots; the router, the shared expert's gate
        // and the shared expert's three projections (through the trunk format) beside them.
        var exbuf: [64]u8 = undefined;
        lw.mlp = .{ .moe = .{
            .router_w = try getLayerWeight(weights, name_buf, prefix, li, "mlp.gate.weight"),
            .router_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.scales") orelse mlx.mlx_array_new(),
            .router_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.gate.biases") orelse mlx.mlx_array_new(),
            .switch_gate_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", "trellis")),
            .switch_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", "suh")) orelse mlx.mlx_array_new(),
            .switch_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.gate_proj", "svh")) orelse mlx.mlx_array_new(),
            .switch_up_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", "trellis")),
            .switch_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", "suh")) orelse mlx.mlx_array_new(),
            .switch_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.up_proj", "svh")) orelse mlx.mlx_array_new(),
            .switch_down_w = try getLayerWeight(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", "trellis")),
            .switch_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", "suh")) orelse mlx.mlx_array_new(),
            .switch_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, moeExpertSuffix(&exbuf, "mlp.switch_mlp.down_proj", "svh")) orelse mlx.mlx_array_new(),
            .shared_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.weight") orelse mlx.mlx_array_new(),
            .shared_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.scales") orelse mlx.mlx_array_new(),
            .shared_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.gate_proj.biases") orelse mlx.mlx_array_new(),
            .shared_up_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.weight") orelse mlx.mlx_array_new(),
            .shared_up_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.scales") orelse mlx.mlx_array_new(),
            .shared_up_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.up_proj.biases") orelse mlx.mlx_array_new(),
            .shared_down_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.weight") orelse mlx.mlx_array_new(),
            .shared_down_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.scales") orelse mlx.mlx_array_new(),
            .shared_down_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert.down_proj.biases") orelse mlx.mlx_array_new(),
            .shared_expert_gate_w = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.weight"),
            .shared_expert_gate_s = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.scales") orelse mlx.mlx_array_new(),
            .shared_expert_gate_b = getLayerWeightOpt(weights, name_buf, prefix, li, "mlp.shared_expert_gate.biases") orelse mlx.mlx_array_new(),
        } };
        const mw = &lw.mlp.moe;
        try checkExl3Bank(mw, cfg, spec);
        try maybeTransposeForBf16(&mw.router_w, mw.router_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.switch_gate_w, mw.switch_gate_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.switch_up_w, mw.switch_up_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.switch_down_w, mw.switch_down_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.shared_gate_w, mw.shared_gate_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.shared_up_w, mw.shared_up_s, &owned_bf16, a, s);
        try maybeTransposeForBf16(&mw.shared_down_w, mw.shared_down_s, &owned_bf16, a, s);
        if (mw.shared_expert_gate_w) |*seg| {
            // A K pack's gate is a dense bf16 [1, hidden] row (`dequantizeSmall` has nothing to dequantize): the router
            // kernel dots its contiguous [hidden] copy with x.
            if (mw.shared_expert_gate_s.?.ctx != null) {
                log.err("[qwen4] layer {d}'s shared expert gate is quantized; a K pack's is dense bf16\n", .{li});
                return error.DirectModelQuantizedSharedGate;
            }
            try maybeTransposeForBf16(seg, mw.shared_expert_gate_s.?, &owned_bf16, a, s);
            if (mlx.mlx_array_size(seg.*) == cfg.hidden_size) {
                var flat = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(flat);
                try mlx.check(mlx.mlx_reshape(&flat, seg.*, &[_]c_int{@intCast(cfg.hidden_size)}, 1, s));
                var vec = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_contiguous(&vec, flat, false, s));
                try owned_bf16.append(a, vec);
                mw.shared_gate_vec = vec;
            }
        }
        // `moe_affine4.bindViews`' answer for EXL3 banks: it slices one expert out of 3-D banks only, and a K pack's
        // trellis is 4-D and its suh and svh 2-D, so the views are the banks themselves (nothing is made or owned).
        mw.bank_views = .{
            .gate = .{ .w = mw.switch_gate_w, .s = mw.switch_gate_s, .b = mw.switch_gate_b },
            .up = .{ .w = mw.switch_up_w, .s = mw.switch_up_s, .b = mw.switch_up_b },
            .down = .{ .w = mw.switch_down_w, .s = mw.switch_down_s, .b = mw.switch_down_b },
        };
    }

    // The PLE is placed by exact equality above and nothing downstream notices its absence; a silently PLE-less model
    // emits plausible text.
    const has_ple = try a.alloc(bool, moe_layers.len);
    defer a.free(has_ple);
    for (moe_layers, 0..) |*lw, i| has_ple[i] = lw.ple != null;
    if (!qwen4PleInstalledAt(has_ple, if (ple_layer) |l| @intCast(l) else -1)) {
        log.err("[qwen4] the PLE was not installed at layer {d} ({d} layers carry PLE weights)\n", .{ cfg.ple_layer_idx, std.mem.count(bool, has_ple, &.{true}) });
        return error.Qwen4PleNotInstalled;
    }

    return .{
        .moe_layers = moe_layers,
        .owned_bf16 = try owned_bf16.toOwnedSlice(a),
    };
}

/// True when the layer loop installed the PLE on exactly the layer the config names (a negative index asks for none).
fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}

/// One hyper-connection block (`loadHcWeights`, a K pack's dense arm): the grouped norm, the down and up mix
/// projections, the reference's 1/hc folded into the down projection, both as the bf16 rows the fused read binds, and
/// (`with_inject`) the inject projection with its 1/hc and its row-major copy for the decode kernel.
fn loadHcWeights(weights: *const Weights, base: []const u8, with_inject: bool, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !HcWeights {
    var nb: [160]u8 = undefined;
    const down = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down", .{base}) catch unreachable, owned_bf16, a, s);
    const up = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up", .{base}) catch unreachable, owned_bf16, a, s);
    var out: HcWeights = .{
        .norm_w = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.hc_norm.weight", .{base}) catch unreachable, hc, hidden, owned_bf16, a, s),
        .down_w = down.w,
        .down_s = down.sc,
        .down_b = down.bi,
        .up_w = up.w,
        .up_s = up.sc,
        .up_b = up.bi,
    };
    // A K pack's mix projections are dense bf16; quantized ones are another pack's.
    if (out.down_s.ctx != null or out.up_s.ctx != null) {
        log.err("[qwen4] {s}'s mix projections are quantized; a K pack's are dense bf16\n", .{base});
        return error.DirectModelQuantizedHyperConnection;
    }
    // The reference divides the down and inject projections by hc_count: folded into the dense weights.
    const inv_hc = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(hc)));
    defer _ = mlx.mlx_array_free(inv_hc);
    const down_src = weights.get(std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down.weight", .{base}) catch unreachable).?;
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_astype(&k, inv_hc, mlx.mlx_array_dtype(down_src), s));
    var scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&scaled, down_src, k, s));
    try mlx.check(mlx.mlx_array_eval(scaled));
    try owned_bf16.append(a, scaled);
    out.down_w = try transposeBf16Weight(scaled, s);
    try owned_bf16.append(a, out.down_w);
    const up_src = weights.get(std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up.weight", .{base}) catch unreachable).?;
    if (mlx.mlx_array_dtype(down_src) == .bfloat16 and mlx.mlx_array_dtype(up_src) == .bfloat16) {
        out.down_rows = try rowsView(scaled, owned_bf16, a, s);
        out.up_rows = try rowsView(up_src, owned_bf16, a, s);
    }
    if (with_inject) {
        const inj = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.block_inject_weight", .{base}) catch unreachable, owned_bf16, a, s);
        if (inj.sc.ctx != null) {
            log.err("[qwen4] {s}'s inject projection is quantized; a K pack's is dense bf16\n", .{base});
            return error.DirectModelQuantizedHyperConnection;
        }
        out.inject_w = inj.w;
        out.inject_s = inj.sc;
        out.inject_b = inj.bi;
        try scaleDense(&out.inject_w, inv_hc, owned_bf16, a, s);
        var flat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&flat, out.inject_w, false, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        try owned_bf16.append(a, flat);
        out.inject_flat = flat;
    }
    return out;
}

/// `x`'s bytes as u32 words (evaluated, owned): the dense rows the fused hyper-connection read binds.
fn rowsView(x: mlx.mlx_array, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    var v = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&v, x, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(v));
    try owned_bf16.append(a, v);
    return v;
}

/// The PLE layer's weights (`loadPleWeights`): the key and value projections, the grouped norms and the short conv.
fn loadPleWeights(weights: *const Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !PleWeights {
    var nb: [160]u8 = undefined;
    const key = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.key_proj", .{ prefix, li }) catch unreachable, owned_bf16, a, s);
    const value = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.value_proj", .{ prefix, li }) catch unreachable, owned_bf16, a, s);
    return .{
        .key_w = key.w,
        .key_s = key.sc,
        .key_b = key.bi,
        .value_w = value.w,
        .value_s = value.sc,
        .value_b = value.bi,
        .norm_key = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_key.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, a, s),
        .norm_query = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_query.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, a, s),
        .norm_conv = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_conv.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, a, s),
        .conv_w = try getLayerWeight(weights, name_buf, prefix, li, "ple.conv1d.weight"),
    };
}

/// At load, after the binder transposed the bf16 parts: a GDN layer's in_proj_a / in_proj_b become views of one
/// `joinRows` buffer (`la.a_w` / `la.b_w` the parts' transposed views, `la.ab.w` the whole buffer's), the map's entries
/// the row views, and the binder's transposed views leave `owned`, so the originals are released (no weight resident
/// twice). Nothing changes for a quantized part or parts the map does not hold under their names
/// (`gdn_ab.joinInPlace`).
fn joinInPlace(la: *LinearAttnWeights, weights: *Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, owned: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !void {
    if (la.combined_proj or la.a_w.ctx == null or la.b_w.ctx == null) return;
    if (la.a_s.ctx != null or la.b_s.ctx != null) {
        log.err("[gdn-ab] in_proj_a and in_proj_b are quantized; a K pack's are bf16, which the joined GEMM reads\n", .{});
        return error.DirectModelGdnAbIsBf16;
    }
    const names = [_][]const u8{ "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight" };
    const old = [_]mlx.mlx_array{ la.a_w, la.b_w };
    var rows: [2]mlx.mlx_array = undefined;
    for (names, old, 0..) |nm, tv, i| {
        const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}", .{ prefix, li, nm }) catch return;
        rows[i] = weights.get(name) orelse return;
        // the map's [w, k] rows under the binder's transposed [k, w] view
        const rs = mlx.getShape(rows[i]);
        const ts = mlx.getShape(tv);
        if (rs.len != 2 or ts.len != 2 or rs[0] != ts[1] or rs[1] != ts[0]) return;
    }
    const j = joinRows(rows[0], rows[1], s) catch |e| switch (e) {
        error.GdnAbShape => return,
        else => return e,
    };
    try owned.append(a, j.joined);
    const a_t = try transposeBf16Weight(j.a, s);
    try owned.append(a, a_t);
    const b_t = try transposeBf16Weight(j.b, s);
    try owned.append(a, b_t);
    const w_t = try transposeBf16Weight(j.joined, s);
    try owned.append(a, w_t);
    // the binder's transposed views go; the map takes the row views (freeing the originals' handles)
    for (old) |o| {
        for (owned.items, 0..) |arr, idx| {
            if (arr.ctx == o.ctx) {
                _ = mlx.mlx_array_free(arr);
                _ = owned.swapRemove(idx);
                break;
            }
        }
    }
    const views = [_]mlx.mlx_array{ j.a, j.b };
    for (names, views) |nm, view| {
        const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}", .{ prefix, li, nm }) catch unreachable;
        weights.replace(name, view);
    }
    la.a_w = a_t;
    la.b_w = b_t;
    la.ab = .{ .w = w_t, .n = j.n, .b_off = j.b_off, .widths = .{ mlx.getShape(j.a)[0], mlx.getShape(j.b)[0] } };
}


const Joined = struct { joined: mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, b_off: c_int, n: c_int };

/// The joined buffer of row-major parts a [wa, k] and b [wb, k] (one 16-bit float dtype): [a | 0 | b], b at the first
/// multiple of 32 at or past a's end (MLX's NAX simdgroup tile is 32 columns wide), contiguous and evaluated, and each
/// part as a row view of it (the caller owns the three arrays).
fn joinRows(a_rows: mlx.mlx_array, b_rows: mlx.mlx_array, s: mlx.mlx_stream) !Joined {
    const ash = mlx.getShape(a_rows);
    const bsh = mlx.getShape(b_rows);
    const dt = mlx.mlx_array_dtype(a_rows);
    if (ash.len != 2 or bsh.len != 2 or ash[1] != bsh[1] or ash[0] < 1 or bsh[0] < 1) return error.GdnAbShape;
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(b_rows) != dt) return error.GdnAbShape;
    const wa = ash[0];
    const k = ash[1];
    const b_off = @divTrunc(wa + 31, 32) * 32;
    const n = b_off + bsh[0];
    var pad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pad);
    var pieces: [3]mlx.mlx_array = .{ a_rows, b_rows, .{ .ctx = null } };
    var count: usize = 2;
    if (b_off > wa) {
        try mlx.check(mlx.mlx_zeros(&pad, &[_]c_int{ b_off - wa, k }, 2, dt, s));
        pieces = .{ a_rows, pad, b_rows };
        count = 3;
    }
    const vec = mlx.mlx_vector_array_new_data(&pieces, count);
    defer _ = mlx.mlx_vector_array_free(vec);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
    var joined = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_contiguous(&joined, cat, false, s));
    try mlx.check(mlx.mlx_array_eval(joined));
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_slice(&a, joined, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ wa, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_slice(&b, joined, &[_]c_int{ b_off, 0 }, 2, &[_]c_int{ n, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return .{ .joined = joined, .a = a, .b = b, .b_off = b_off, .n = n };
}

// ── the n-gram table ──

/// A trellis table's per-head bias (`*.ple.ple_embedding.head_bias`, [heads, dim], read once) handed to the table,
/// which adds it to each row in f16 (`trellisHeadBias`). The load keeps it f16 (`keepsF16Operand`): another dtype, or a
/// pack without one, is refused by name.
fn trellisHeadBias(table: *NgramTable, weights: *Weights, n_heads: u32) !void {
    var it = weights.map.iterator();
    while (it.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, ".ple.ple_embedding.head_bias")) {
        const hb = e.value_ptr.*;
        try mlx.check(mlx.mlx_array_eval(hb));
        const n = mlx.mlx_array_size(hb);
        if (mlx.mlx_array_dtype(hb) != .float16) {
            log.err("[qwen4] {s} is {s}; the trellis n-gram table adds an f16 bias\n", .{ e.key_ptr.*, @tagName(mlx.mlx_array_dtype(hb)) });
            return error.NgramTableHeader;
        }
        const p = mlx.mlx_array_data_float16(hb) orelse return error.MissingWeight;
        return setHeadBias(table, @as([*]const u16, @ptrCast(p))[0..n], n_heads);
    };
    log.err("[qwen4] the trellis n-gram table needs its per-head bias (ple.ple_embedding.head_bias), which the pack does not hold\n", .{});
    return error.MissingWeight;
}

/// The trellis rows' per-hash-head bias (`n_heads` x `dim` f16 bits), copied widened to f32 into the table, which owns
/// it (`NgramTable.setHeadBias`; the table's close frees it).
pub fn setHeadBias(table: *NgramTable, bias_bits: []const u16, n_heads: u32) !void {
    if (n_heads == 0 or bias_bits.len != @as(usize, n_heads) * table.dim) return error.NgramTableHeader;
    const wide = try std.heap.page_allocator.alloc(f32, bias_bits.len);
    for (wide, bias_bits) |*w, b| w.* = @floatCast(@as(f16, @bitCast(b)));
    if (table.head_bias) |old| std.heap.page_allocator.free(old);
    table.head_bias = wide;
    table.n_heads = n_heads;
}

/// The trellis n-gram table mmapped through its kept fd (the copy's `NgramTable.open`), with the gather's workers
/// (`ple.poolCreate`) and the prefetch's readers (`ple.aheadCreate`); the table's close frees both.
pub fn openTable(path: []const u8) !NgramTable {
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= pbuf.len) return error.NameTooLong;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    const fd = crt.openRead(pbuf[0..path.len :0]);
    if (fd < 0) return error.FileNotFound;
    errdefer crt.close(fd);
    const size: usize = @intCast(@max(crt.seekEnd(fd), 0));
    if (size == 0) return error.StatFailed;
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(map);
    if (size < 8) return error.NgramTableTruncated;
    const hlen: usize = @intCast(std.mem.readInt(u64, map[0..8], .little));
    if (hlen > size - 8) return error.NgramTableTruncated;
    var t = try NgramTable.parse(map, map[8 .. 8 + hlen], 8 + hlen);
    t.fd = fd;
    t.pool = ple.poolCreate() catch null;
    if (t.trellis_k != 0 and t.pool != null) t.ahead = ple.aheadCreate() catch null;
    return t;
}

/// The n-gram warm runs when the loaded weights and the table both fit in RAM (`exl3WarmFits`); past that the table's
/// pages only evict the weights', which the warm's own off line says (`trunk_load.warmFits`, an EXL3 trunk pack's arm).
fn warmFits(weights_bytes: usize, table: *const NgramTable) bool {
    const table_bytes = warmTotal(table);
    const ram = system.getTotalMemBytes();
    if (exl3WarmFits(weights_bytes, table_bytes, ram)) return true;
    const gib = 1024.0 * 1024.0 * 1024.0;
    log.info("[qwen4] ngram table warm: declined - {d:.1} GB of weights and a {d:.1} GB table pass {d:.1} GB of RAM; the first long prompt faults the table in from SSD\n", .{
        @as(f64, @floatFromInt(weights_bytes)) / gib,
        @as(f64, @floatFromInt(table_bytes)) / gib,
        @as(f64, @floatFromInt(ram)) / gib,
    });
    return false;
}

/// On when the loaded weights and the table both fit in the machine's RAM (an unknown RAM size, 0, keeps it on).
fn exl3WarmFits(weights: u64, table: u64, ram: u64) bool {
    return ram == 0 or weights +| table <= ram;
}

/// One 8 MB pread of the warm.
const warm_chunk: usize = 8 << 20;

const WarmRegion = struct { fd: c_int, off: u64, len: u64 };

/// The `i`-th byte range the warm reads: a K pack's table is its own file, read whole through its kept fd.
fn warmRegion(table: *const NgramTable, i: usize) ?WarmRegion {
    return if (i == 0 and table.fd >= 0) .{ .fd = table.fd, .off = 0, .len = table.map.len } else null;
}

/// The bytes the warm reads (`warmRegion`'s ranges).
fn warmTotal(table: *const NgramTable) u64 {
    var total: u64 = 0;
    var i: usize = 0;
    while (warmRegion(table, i)) |r| : (i += 1) total += r.len;
    return total;
}

/// Reads the whole table through its fd once, in the background, so the first prompt's PLE gathers hit a warm page
/// cache (`NgramTable.startWarm`; its off arm is a test seam's). The table sits at its final address (the thread holds
/// it); its close stops the thread and joins it.
fn startWarm(table: *NgramTable) void {
    if (warmRegion(table, 0) == null or table.warm_thread != null) return;
    const total = warmTotal(table);
    table.warm_stop.store(false, .release);
    log.info("[qwen4] ngram table warm: started, {d:.1} GB in the background (page cache)\n", .{asGb(total)});
    table.warm_thread = std.Thread.spawn(.{}, warmMain, .{ table, total }) catch null;
}

fn warmMain(table: *NgramTable, total: u64) void {
    var scratch: [warm_chunk]u8 align(16) = undefined;
    const wio = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(wio, .boot);
    var i: usize = 0;
    while (warmRegion(table, i)) |r| : (i += 1) {
        var off: u64 = 0;
        while (off < r.len) {
            if (table.warm_stop.load(.acquire)) return;
            const want: usize = @intCast(@min(r.len - off, warm_chunk));
            const got = crt.pread(r.fd, &scratch, want, @intCast(r.off + off));
            if (got <= 0) return;
            off += @intCast(got);
        }
    }
    const secs: f64 = @as(f64, @floatFromInt(t0.untilNow(wio, .boot).nanoseconds)) / 1e9;
    log.info("[qwen4] ngram table warm: done, {d:.1} GB in {d:.1} s (page cache)\n", .{ asGb(total), secs });
}

/// GiB, as the copy's warm lines print them.
fn asGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

// ── the weight helpers (`engines/mlx/load/weights.zig`) ──

const Triple = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array };

/// One quantized-or-dense projection triple by full tensor base name, a dense weight pre-transposed (`weightTriple`; a
/// K pack's dense ones are bf16, so nothing is dequantized first).
fn weightTriple(weights: *const Weights, base: []const u8, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !Triple {
    var nb: [160]u8 = undefined;
    const w = weights.get(std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch unreachable) orelse {
        log.err("MISSING WEIGHT: {s}.weight\n", .{base});
        return error.MissingWeight;
    };
    const sc = weights.get(std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    const bi = weights.get(std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    var wm = w;
    try maybeTransposeForBf16(&wm, sc, owned_bf16, a, s);
    return .{ .w = wm, .sc = sc, .bi = bi };
}

/// In place: a weight with no scales is plain bf16, replaced by its pre-transposed `[in, out]` form (owned) so the
/// projection dispatches to a plain matmul. A quantized or absent weight is left as it is.
fn maybeTransposeForBf16(w: *mlx.mlx_array, sc: mlx.mlx_array, owned: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !void {
    if (sc.ctx != null) return;
    if (w.ctx == null) return;
    const transposed = try transposeBf16Weight(w.*, s);
    w.* = transposed;
    try owned.append(a, transposed);
}

/// A grouped-norm weight `[hc*hidden]` viewed as `[hc, hidden]` (owned).
fn groupedNormView(weights: *const Weights, name: []const u8, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
    const shape = [_]c_int{ @intCast(hc), @intCast(hidden) };
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, raw, &shape, 2, s));
    try owned_bf16.append(a, view);
    return view;
}

/// A dense weight multiplied by the scalar `k` in the weight's own dtype (an f32 scalar would widen the residual
/// stream), evaluated and owned (`scaleTriple`'s dense arm).
fn scaleDense(w: *mlx.mlx_array, k: mlx.mlx_array, owned_bf16: *std.ArrayList(mlx.mlx_array), a: std.mem.Allocator, s: mlx.mlx_stream) !void {
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_astype(&kt, k, mlx.mlx_array_dtype(w.*), s));
    var w2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&w2, w.*, kt, s));
    try mlx.check(mlx.mlx_array_eval(w2));
    try owned_bf16.append(a, w2);
    w.* = w2;
}

const Field = struct { name: [:0]const u8, type: type };

/// `T`'s fields in declaration order.
fn structFields(comptime T: type) []const Field {
    return comptime blk: {
        const names = std.meta.fieldNames(T);
        const types = std.meta.fieldTypes(T);
        var result: [names.len]Field = undefined;
        for (names, types, 0..) |name, ty, i| result[i] = .{ .name = name, .type = ty };
        const final = result;
        break :blk &final;
    };
}

/// Appends every set array of a weight struct to `vec`, in declaration order, an optional one when it is there
/// (`appendStructArrays` and the per-kind appenders: a K pack's GDN projections are separate, so none of the combined
/// form's z / a fields is skipped).
fn appendStructArrays(vec: mlx.mlx_vector_array, w: anytype) void {
    inline for (comptime structFields(@TypeOf(w.*))) |field| {
        if (comptime field.type == mlx.mlx_array) {
            const arr = @field(w, field.name);
            if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
        } else if (comptime field.type == ?mlx.mlx_array) {
            if (@field(w, field.name)) |arr| {
                if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        }
    }
}

/// A table stored dense (a float dtype) carries no scales, whatever the config's nominal bits.
fn floatDtypeTable(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float16 or dtype == .float32;
}

fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

/// `{prefix}.{base}.{suffix}`.
fn getNamedWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, base: []const u8, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}.{s}", .{ prefix, base, suffix }) catch unreachable;
    return weights.get(name);
}

fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// `<container>.<leaf>` into `buf`, beside the layer name's own buffer.
fn moeExpertSuffix(buf: []u8, container: []const u8, leaf: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ container, leaf }) catch unreachable;
}

/// A quantization scale: none in a dense pack (`quant_bits` 0), mandatory under a quantization block.
fn getLayerScaleOrEmpty(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) error{MissingWeight}!mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return try getLayerWeight(weights, buf, prefix, layer, suffix);
}

/// A `.biases` tensor: optional without a quantization block, mandatory under an affine one (the direct parse admits no
/// other mode).
fn getLayerBias(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, cfg: *const config.Config) error{MissingWeight}!mlx.mlx_array {
    if (cfg.quant_bits == 0) return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
    if (cfg.quant_mode.hasBiases()) return try getLayerWeight(weights, buf, prefix, layer, suffix);
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
}

// ── tests ──

test "direct load: an EXL3 trunk pack warms the n-gram table only when its weights and the table both fit in RAM" {
    const gib: u64 = 1 << 30;
    try std.testing.expect(exl3WarmFits(50 * gib, 51 * gib, 128 * gib));
    try std.testing.expect(!exl3WarmFits(80 * gib, 51 * gib, 128 * gib));
    try std.testing.expect(exl3WarmFits(80 * gib, 51 * gib, 0));
}

test "direct load: a trellis n-gram table takes its per-head bias from the map, and a pack without one is refused" {
    const testing = std.testing;
    const empty: [0]u8 align(std.heap.page_size_min) = .{};
    var table: NgramTable = .{ .map = &empty, .rows = 1, .dim = 4, .bits = 6, .group_size = 0, .w_off = 0, .s_off = 0, .b_off = 0, .wcols = 3, .scols = 0, .trellis_k = 6, .gpu_owns_map = true };
    defer table.close();
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    try testing.expectError(error.MissingWeight, trellisHeadBias(&table, &w, 2));
    // f16 1, -1, 0, 0.5, 2, 1, 0.25, -2
    const bits = [_]u16{ 0x3C00, 0xBC00, 0x0000, 0x3800, 0x4000, 0x3C00, 0x3400, 0xC000 };
    try w.put("m.layers.0.ple.ple_embedding.head_bias", mlx.mlx_array_new_data(&bits, &[_]c_int{ 2, 4 }, 2, .float16));
    try trellisHeadBias(&table, &w, 2);
    try testing.expectEqualSlices(f32, &.{ 1, -1, 0, 0.5, 2, 1, 0.25, -2 }, table.head_bias.?);
    try testing.expectError(error.NgramTableHeader, trellisHeadBias(&table, &w, 3));
}

