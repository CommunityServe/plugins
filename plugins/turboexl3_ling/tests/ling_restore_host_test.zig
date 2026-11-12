//! turboexl3_ling's restore byte check, part B: the served mlx-serve's own cache code (GPU; it runs with
//! MLX_SERVE_LING_RESTORE_IN set to part A's file, else it skips). The plugin never builds this file. The check's
//! runner copies it into a copy of the served tree's src/ and makes it the only test of that tree's unit-test root
//! (src/tests.zig), beside that test block's two llama engine imports (the linked llama shim calls `mlx_llama_log`,
//! which engines/llama/bridge.zig exports). It is then built by that tree's own build (`zig build test-build
//! -Dtest-filter='turboexl3_ling restore bytes'`: the filter keeps out the tests of the files it imports), so the cache
//! code it drives is the served tree's, compiled as the host compiles its tests.
//! - Part A wrote the served cache: the trunk's MLA layers on the rope-once cache (the roped key once a token as K
//!   [1, 1, T, 64], kv_b's rows as V [1, 1, T, 16384]) and the MTP head's layer on its latent cache (K [1, 1, T, 512],
//!   V [1, 1, T, 64]).
//! - The SSD roots. The served engine roots a model's SSD entries at the fingerprint of its folder and its layout
//!   namespace (engines/mlx/load.zig; disk_tier.zig `modelFingerprintWithLayout`), and a tier reads its own root only.
//!   Part A's metadata names the served cache's namespace as config.zig serves it (`ns`). The tiers sit at its root for
//!   one model folder (`model_dir`, which only seeds the hash), so only the namespace tells it from the roots earlier
//!   builds of this plugin served for other layouts (`earlier_namespaces`: the expanded cache with the MTP head's
//!   latent, the expanded cache alone with no namespace, and the rope-once cache without the head's latent). ROOTS: the
//!   served namespace is named, and its root differs from each earlier root.
//! - Part A wrote two prompts, one for each commit path, and each goes into a slot of its own as the served prefill
//!   keeps it: a KVCache of the model's layers with the prompt's rows reserved (the prompt and one chunk, as
//!   generate/init.zig reserves them), one SSMCacheEntry a layer, and the MTP head's one-layer cache. Part A's writes
//!   go through the host's CacheOps (kv/cache_ops.zig, the table that the plugin calls): each span's appends in call
//!   order, then at each checkpoint every KDA layer's states through CacheOps.checkpoint, which files them on the
//!   prefill's list (ssm/checkpoint.zig `PromptCheckpoints`, at the positions that `wholePromptPlan` names).
//! - The RAM tier and the SSD tier as engines/mlx/load.zig sets them up for this pack: 32 entries, hybrid, the thinning
//!   of an arch that is not long-context gated, 16 checkpoints an entry and the legacy 8 a disk entry, 1024-token
//!   chunks, 24 GB, and the background writer armed as `armDiskTier` arms it (not SSD-first).
//! - SPILL, the served config: the RAM tier's budget is at its clamp's floor (1 byte, admission/hot_cache_plan.zig
//!   `clampedPrefixCacheMem`), so the commit declines and the decline spills the candidate to the SSD tier
//!   (prefix_cache.zig `spillDeclinedToDisk`). The spill writes at most 4 GiB a commit (disk_tier.zig
//!   `DECLINE_SPILL_FLUSH_FLOOR`): the checkpoints first, at most half of it, then 1024-token chunks while the bytes
//!   written are under it. The 16,700-token prompt would stop at 13,312 tokens (296,064 bytes a token), short of its
//!   last checkpoints. So SPILL commits the 9,000-token prompt, which the spill holds whole, and its LIMIT line states
//!   the limit with the entry's own sizes: SPILL passes only on a whole entry, never on a shorter restore.
//! - FLUSH: a RAM tier with room keeps the 16,700-token prompt's entry, and its flush writes the whole entry with the
//!   MTP head's spec sidecar (`flushPendingDisk`, then `flushOwedSlice`). A fresh tier over the same folder restores
//!   it.
//! - The MTP head's rows: FLUSH persists them in the spec sidecar, and on mlx-serve 66aad949 (#1150) the decline spill
//!   writes the sidecar too, so a SPILL restore also holds head rows (restore B's DONE line: head words = K words).
//! - Restores: one per checkpoint, each into a fresh KVCache, fresh SSM entries and a fresh head cache
//!   (`lookupAndRestore`). The deepest is asked as the next turn asks it (the prompt and 3 more tokens), the others at
//!   100 tokens past their checkpoint.
//! - Checks, word for word against what part A wrote: every restored row of every MLA layer (K and V) and of the head,
//!   and every KDA layer's restored conv and ssm state against its checkpoint's. Also the plans, the rows the slot
//!   holds after part A's writes, each SSD entry's length and checkpoints, each restore point and the rows restored.
//! - Plants on this test's own copies (never a host setting), each of which must make its comparison differ: on the
//!   deepest SPILL restore, layer 39's restored keys zeroed (P8a's form), the rope part of its last 256 restored key
//!   rows zeroed (P8b's form; the whole key in the rope-once layout), one bit of its last restored row's roped key
//!   flipped, and one flipped bit in one restored ssm state (a red check); on the deepest FLUSH restore, the head's
//!   last 256 restored rope-key rows zeroed. Four plants.
//! - CROSS, after SPILL's restores: a restore across roots refused by the roots' names. The SPILL folder holds the
//!   served entry alone. A tier opened over it at each earlier build's root must hold no entry and restore nothing.
//!   The same lookup at the served root (the control) must restore the entry with its keys in the served layout. So
//!   the same tokens would restore, and only the root's name keeps them out: an SSD tier an earlier build wrote never
//!   restores into the served cache. Each run's folder goes when its checks are done, so the disk holds one entry at a
//!   time.
//! Each line starts with `TURBOEXL3_LING_RESTORE_BYTES B` and ends in PASS or FAIL. The DONE line totals them.
const std = @import("std");
const mlx = @import("mlx/ffi.zig");
const abi = @import("abi/plugin.zig");
const KVCache = @import("engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("engines/mlx/ssm/entry.zig").SSMCacheEntry;
const cache_ops = @import("engines/mlx/kv/cache_ops.zig");
const checkpoint = @import("engines/mlx/ssm/checkpoint.zig");
const prefill_chunk = @import("engines/mlx/generate/prefill_chunk.zig");
const prefix_cache = @import("engines/mlx/cache/prefix_cache.zig");
const disk_tier = @import("engines/mlx/cache/ssd/disk_tier.zig");

const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
const gpa = std.heap.c_allocator;
const tag = "TURBOEXL3_LING_RESTORE_BYTES B";

/// The served config: the prefill width for this MoE arch (4096, `--prefill-chunk` unset), `--ssm-checkpoint-stride`
/// 256, 16 checkpoints an entry, `--prefix-cache-disk 24GB`, 32 RAM entries
const width: usize = 4096;
const base_stride: usize = 256;
const cp_max: u32 = 16;
const ram_entries: u32 = 32;
const disk_bytes: u64 = 24 << 30;
/// The layer that the plants change (P8a and P8b changed layer 39)
const plant_layer: u32 = 39;
/// The rows that the P8b-form plants zero
const plant_rows: c_int = 256;

/// The two commit paths, each with part A's prompt of the same name
const Path = enum { spill, flush };

/// The model folder whose fingerprint roots every tier here (disk_tier.zig `modelFingerprintWithLayout`). It is
/// absolute and need not exist (a config.json is hashed only where there is one), so only the layout namespace tells
/// the roots apart.
const model_dir = "/turboexl3_ling-restore-bytes/pack";

/// The layout namespaces earlier builds of this plugin served, each a layout this build never reads: the expanded MLA
/// cache with the MTP head's latent, the expanded cache alone ("": no namespace), and the rope-once cache without the
/// head's latent. An SSD tier those builds wrote must never restore into the served cache (CROSS, ROOTS).
const earlier_namespaces = [_][]const u8{ "ling-mtp-head-latent-v1", "", "ling-mla-rope1-v1" };

/// A namespace for a line ("" is none)
fn nsText(ns: []const u8) []const u8 {
    return if (ns.len == 0) "none" else ns;
}

/// The SSD roots: the served cache's namespace as part A recorded it, and the fingerprints (the folder name a tier takes
/// under its base) of its root and of each earlier build's root (`earlier_namespaces`, in that order)
const Roots = struct {
    ns: []u8,
    served: []u8,
    earlier: [earlier_namespaces.len][]u8,

    /// The served namespace from part A's metadata, and every root's fingerprint
    fn load(in_path: [*:0]const u8) !Roots {
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var arrays = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(arrays);
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_load_safetensors(&arrays, &meta, in_path, cpu));
        var v: [*:0]const u8 = undefined;
        if (mlx.mlx_map_string_to_string_get(&v, meta, "ns") != 0) {
            std.debug.print("{s} FAIL: part A's file names no layout namespace (ns): it is from before the one-layout check, so run this part A again\n", .{tag});
            return error.NoLayoutNamespace;
        }
        const ns = try gpa.dupe(u8, std.mem.span(v));
        errdefer gpa.free(ns);
        const layout: ?[]const u8 = if (ns.len == 0) null else ns;
        const served = try disk_tier.modelFingerprintWithLayout(gpa, std.testing.io, model_dir, layout);
        errdefer gpa.free(served);
        var earlier: [earlier_namespaces.len][]u8 = undefined;
        var made: usize = 0;
        errdefer for (earlier[0..made]) |fp| gpa.free(fp);
        for (earlier_namespaces, &earlier) |e, *fp| {
            const e_layout: ?[]const u8 = if (e.len == 0) null else e;
            fp.* = try disk_tier.modelFingerprintWithLayout(gpa, std.testing.io, model_dir, e_layout);
            made += 1;
        }
        return .{ .ns = ns, .served = served, .earlier = earlier };
    }

    fn deinit(r: *Roots) void {
        gpa.free(r.ns);
        gpa.free(r.served);
        for (r.earlier) |fp| gpa.free(fp);
    }
};

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn ab(a: Array) abi.Array {
    return .{ .ctx = a.ctx };
}

fn mx(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// Part A's file
const Input = struct {
    arrays: mlx.mlx_map_string_to_array,

    fn get(self: Input, comptime fmt: []const u8, args: anytype) !Array {
        var buf: [64]u8 = undefined;
        const key = try std.fmt.bufPrintSentinel(&buf, fmt, args, 0);
        var a = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_get(&a, self.arrays, key) != 0 or a.ctx == null) {
            free(a);
            std.debug.print("{s} FAIL: part A's file has no {s}\n", .{ tag, key });
            return error.MissingArray;
        }
        return a;
    }

    fn u32s(self: Input, comptime name: []const u8) ![]u32 {
        const a = try self.get(name, .{});
        defer free(a);
        try mlx.check(mlx.mlx_array_eval(a));
        const p = mlx.mlx_array_data_uint32(a) orelse return error.NoData;
        return gpa.dupe(u32, p[0..mlx.mlx_array_size(a)]);
    }
};

/// `a` made contiguous and evaluated, for reading its words on the host
fn evaluated(a: Array, s: Stream) !Array {
    var c = mlx.mlx_array_new();
    errdefer free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

/// The words of `got` that differ from `want` (no array, or a shape or dtype that differs, counts every word)
fn differing(got: Array, want: Array, s: Stream) !usize {
    const n = mlx.mlx_array_size(want);
    if (got.ctx == null) return n;
    if (!std.mem.eql(c_int, mlx.getShape(got), mlx.getShape(want)) or mlx.mlx_array_dtype(got) != mlx.mlx_array_dtype(want)) return n;
    const cg = try evaluated(got, s);
    defer free(cg);
    const cw = try evaluated(want, s);
    defer free(cw);
    const isz = mlx.mlx_array_itemsize(cw);
    const pg = mlx.mlx_array_data_uint8(cg) orelse return error.NoData;
    const pw = mlx.mlx_array_data_uint8(cw) orelse return error.NoData;
    if (std.mem.eql(u8, pg[0 .. n * isz], pw[0 .. n * isz])) return 0;
    var d: usize = 0;
    for (0..n) |i| d += @intFromBool(!std.mem.eql(u8, pg[i * isz ..][0..isz], pw[i * isz ..][0..isz]));
    return d;
}

/// The rows a `[B, H, T, D]` array holds (T)
fn rowCount(a: Array) usize {
    return @intCast(mlx.getShape(a)[2]);
}

/// `a[:, :, 0:n, :]` (a view)
fn rows(a: Array, n: c_int, s: Stream) !Array {
    const sh = mlx.getShape(a);
    var r = mlx.mlx_array_new();
    errdefer free(r);
    try mlx.check(mlx.mlx_slice(&r, a, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], n, sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return r;
}

/// A copy of `a` (`[B, H, T, D]`) with columns `c0`..`D` of its last `nz` rows zeroed, every head
fn zeroedTail(a: Array, nz: c_int, c0: c_int, s: Stream) !Array {
    const sh = mlx.getShape(a);
    var z = mlx.mlx_array_new();
    defer free(z);
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ sh[0], sh[1], nz, sh[3] - c0 }, 4, mlx.mlx_array_dtype(a), s));
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_slice_update(&out, a, z, &[_]c_int{ 0, 0, sh[2] - nz, c0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// The first bf16 word of `got` that differs from `want` (the same shape): its index and both words' bits
const FirstDiff = struct { index: usize = 0, got: u16 = 0, want: u16 = 0, found: bool = false };

fn firstDiff(got: Array, want: Array, s: Stream) !FirstDiff {
    if (mlx.mlx_array_itemsize(want) != 2 or mlx.mlx_array_size(got) != mlx.mlx_array_size(want)) return .{};
    const cg = try evaluated(got, s);
    defer free(cg);
    const cw = try evaluated(want, s);
    defer free(cw);
    const pg = mlx.mlx_array_data_uint8(cg) orelse return error.NoData;
    const pw = mlx.mlx_array_data_uint8(cw) orelse return error.NoData;
    for (0..mlx.mlx_array_size(cw)) |i| {
        const g = std.mem.readInt(u16, pg[2 * i ..][0..2], .little);
        const w = std.mem.readInt(u16, pw[2 * i ..][0..2], .little);
        if (g != w) return .{ .index = i, .got = g, .want = w, .found = true };
    }
    return .{};
}

/// A bf16 word's value
fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Where a plant's first differing word sits in `[B, H, T, D]` (head, row, column) and both values, for its line
fn whereText(buf: []u8, d: FirstDiff, shape: []const c_int) ![]const u8 {
    if (!d.found) return "none";
    const cols: usize = @intCast(shape[3]);
    const rows_n: usize = @intCast(shape[2]);
    const i = d.index;
    return std.fmt.bufPrint(buf, "head {d}, row {d}, column {d}: expected {d} (0x{x:0>4}), got {d} (0x{x:0>4})", .{ i / (cols * rows_n), (i / cols) % rows_n, i % cols, bf16(d.want), d.want, bf16(d.got), d.got });
}

/// A copy of `a` with the first bit of its word `word` flipped (`a` made contiguous: row-major over its shape)
fn flippedAt(a: Array, word: usize, s: Stream) !Array {
    const c = try evaluated(a, s);
    defer free(c);
    const isz = mlx.mlx_array_itemsize(c);
    const n = mlx.mlx_array_size(c) * isz;
    if (word * isz >= n) return error.WordOutOfRange;
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    const buf = try gpa.alloc(u8, n);
    defer gpa.free(buf);
    @memcpy(buf, p[0..n]);
    buf[word * isz] ^= 1;
    const sh = mlx.getShape(c);
    const out = mlx.mlx_array_new_data(buf.ptr, sh.ptr, @intCast(sh.len), mlx.mlx_array_dtype(c));
    if (out.ctx == null) return error.MlxError;
    return out;
}

/// A copy of `a` with the first bit of its first word flipped (the SSM comparison's red check)
fn flipped(a: Array, s: Stream) !Array {
    return flippedAt(a, 0, s);
}

/// What part A wrote at one layer: its appends joined in call order. `fmt` names a span's array: a trunk layer's
/// (`<p>.k.l{d}.s{d}`, `<p>.v.l{d}.s{d}`, with `layer`) or the head's (`<p>.hk.s{d}`, `<p>.hv.s{d}`, `head`).
fn written(in: Input, comptime fmt: []const u8, comptime head: bool, layer: u32, spans: usize, s: Stream) !Array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (0..spans) |j| {
        const a = if (head) try in.get(fmt, .{j}) else try in.get(fmt, .{ layer, j });
        defer free(a);
        _ = mlx.mlx_vector_array_append_value(vec, a);
    }
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try gpa.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| inline for (.{ e.conv_state, e.ssm_state, e.spec_state_seq, e.spec_conv_input }) |a| free(a);
    gpa.free(es);
}

/// The per-model cache as engines/mlx/load.zig sets it up for this pack (not long-context gated, the RAM tier on,
/// hybrid, no module binding), with its SSD tier armed as `armDiskTier` arms it on an arch that is not SSD-first, at
/// `root` under `base` (load.zig's root: the fingerprint of the model folder and the layout namespace, `Roots`)
fn tiers(hc: *prefix_cache.HotPrefixCache, base: []const u8, root: []const u8) !void {
    hc.cp_thin = .min_span;
    hc.ssm_low_anchors = 0;
    hc.hybrid = true;
    hc.ssm_checkpoint_max = cp_max;
    hc.disk = try disk_tier.DiskTier.init(gpa, std.testing.io, base, root, disk_bytes, disk_tier.DEFAULT_CHUNK_TOKENS);
    const d = &hc.disk.?;
    d.cp_thin = .oldest;
    d.ssm_max_per_entry = disk_tier.SSM_DISK_MAX_PER_ENTRY_LEGACY;
    d.ssm_low_anchors = 0;
    d.enableBackgroundWriter();
}

/// The SSD tier's folder for one path: `$MLX_SERVE_LING_RESTORE_DISK/<name>` when that is set (an existing folder the
/// runner keeps, to read the files the tier wrote), else the test's temporary folder
fn tierRoot(tmp: *std.testing.TmpDir, buf: *[512]u8, comptime name: []const u8) ![]const u8 {
    if (std.c.getenv("MLX_SERVE_LING_RESTORE_DISK")) |d| return std.fmt.bufPrint(buf, "{s}/" ++ name, .{std.mem.span(d)});
    return buf[0..try tmp.dir.realPath(std.testing.io, buf)];
}

/// The SSD tier's one entry, if it holds exactly one
fn onlyEntry(d: *const disk_tier.DiskTier) ?*const disk_tier.IndexEntry {
    return if (d.entryCount() == 1) &d.entries.items[0] else null;
}

const Totals = struct {
    checks: usize = 0,
    failed: usize = 0,
    restores: usize = 0,
    k: usize = 0,
    v: usize = 0,
    conv: usize = 0,
    ssm: usize = 0,
    head: usize = 0,
    rows: usize = 0,
    plants: usize = 0,
    /// The cross-root lookups (CROSS)
    cross: usize = 0,

    fn line(t: *Totals, ok: bool, comptime fmt: []const u8, args: anytype) void {
        t.checks += 1;
        if (!ok) t.failed += 1;
        std.debug.print(tag ++ " " ++ fmt, args);
        std.debug.print(" {s}\n", .{if (ok) "PASS" else "FAIL"});
    }
};

/// What one path's restores compare against, and the run's totals
const Ctx = struct {
    in: Input,
    s: Stream,
    t: *Totals,
    /// The run's name on its lines (`SPILL`, `FLUSH`)
    name: []const u8,
    /// Part A's name for this path's prompt (`spill`, `flush`)
    prompt: []const u8,
    n_layers: usize,
    mla: []const u32,
    kda: []const u32,
    tokens: []const u32,
    cps: []const u32,
    /// The rope key's width: the last `rope` columns of each head's key
    rope: c_int,
    /// Part A's rows, joined, at each MLA layer (null elsewhere), and the head's
    k: []Array,
    v: []Array,
    hk: Array = .{ .ctx = null },
    hv: Array = .{ .ctx = null },

    fn line(cx: *Ctx, ok: bool, comptime fmt: []const u8, args: anytype) void {
        cx.t.line(ok, fmt, args);
    }

    /// A plant's line (`what`, with `args`): its copy must differ from what part A wrote
    fn plant(cx: *Ctx, comptime what: []const u8, args: anytype, got: Array, want: Array) !void {
        const d = try differing(got, want, cx.s);
        var wb: [192]u8 = undefined;
        const at = try whereText(&wb, try firstDiff(got, want, cx.s), mlx.getShape(want));
        cx.t.plants += @intFromBool(d > 0);
        cx.t.checks += 1;
        if (d == 0) cx.t.failed += 1;
        std.debug.print(tag ++ " PLANT {s}: ", .{cx.name});
        std.debug.print(what, args);
        std.debug.print(" in this test's copy: differing {d}/{d}; the first differing word: {s}; {s}\n", .{ d, mlx.mlx_array_size(want), at, if (d > 0) "caught PASS" else "missed FAIL" });
    }
};

/// Layer `plant_layer`'s restored keys with each P8 plant in this test's own copy
fn plants(cx: *Ctx, got_k: Array, want_k: Array, at: usize) !void {
    var zk = mlx.mlx_array_new();
    defer free(zk);
    const ks = mlx.getShape(got_k);
    try mlx.check(mlx.mlx_zeros(&zk, ks.ptr, ks.len, mlx.mlx_array_dtype(got_k), cx.s));
    try cx.plant("P8a's form: layer {d}'s {d} restored rows of keys (K, every head and column) zeroed", .{ plant_layer, at }, zk, want_k);
    const nz: c_int = @min(plant_rows, ks[2]);
    const pk = try zeroedTail(got_k, nz, ks[3] - cx.rope, cx.s);
    defer free(pk);
    try cx.plant("P8b's form: layer {d}'s last {d} restored rope-key rows (the rope part of K: rows {d} to {d}, columns {d} to {d} of every head) zeroed", .{ plant_layer, nz, ks[2] - nz, ks[2] - 1, ks[3] - cx.rope, ks[3] - 1 }, pk, want_k);
    // One restored roped-key row with one word changed: head 0's last row, the first word of its rope part (the whole
    // key's first word in the rope-once layout).
    const word: usize = @intCast((ks[2] - 1) * ks[3] + ks[3] - cx.rope);
    const fk = try flippedAt(got_k, word, cx.s);
    defer free(fk);
    try cx.plant("one bit of layer {d}'s last restored roped-key row flipped (head 0, row {d}, column {d})", .{ plant_layer, ks[2] - 1, ks[3] - cx.rope }, fk, want_k);
}

/// One restore into a fresh slot as the next request's lookup makes it, then every restored word against what part A
/// wrote. `head_expected`: the path persists the MTP head's spec sidecar. `plant`: run the P8 plants and the red check
/// on this restore. `plant_head`: run the head's plant on it.
fn restore(cx: *Ctx, hc: *prefix_cache.HotPrefixCache, comptime path: []const u8, ci: usize, head_expected: bool, plant: bool, plant_head: bool) !void {
    const at = cx.cps[ci];
    const deepest = ci + 1 == cx.cps.len;
    var q: std.ArrayList(u32) = .empty;
    defer q.deinit(gpa);
    try q.appendSlice(gpa, if (deepest) cx.tokens else cx.tokens[0 .. at + 100]);
    try q.appendSlice(gpa, if (deepest) &[_]u32{ 156_901, 156_902, 156_903 } else &[_]u32{156_900});
    var kv = try KVCache.init(gpa, @intCast(cx.n_layers));
    defer kv.deinit();
    const ssm = try newEntries(cx.n_layers);
    defer freeEntries(ssm);
    var head = try KVCache.init(gpa, 1);
    defer head.deinit();
    var head_base: usize = 0;
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&kv, &moe_off, ssm, cx.s, q.items, false, &.{}, null, .{ .cache = &head, .base_pos = &head_base });
    cx.t.restores += 1;
    const ops = &cache_ops.ops;
    const stream: abi.Stream = .{ .ctx = cx.s.ctx };
    var slot: cache_ops.Slot = .{ .kv = &kv, .ssm = ssm };
    var kd: usize = 0;
    var kn: usize = 0;
    var vd: usize = 0;
    var vn: usize = 0;
    var rows_ok = true;
    for (cx.mla) |li| {
        var view: abi.KvView = .{};
        if (!ops.view(slot.handle(), li, stream, &view).isOk() or view.k.ctx == null) {
            rows_ok = false;
            continue;
        }
        defer ops.release(&view);
        const want_k = try rows(cx.k[li], @intCast(at), cx.s);
        defer free(want_k);
        const want_v = try rows(cx.v[li], @intCast(at), cx.s);
        defer free(want_v);
        rows_ok = rows_ok and rowCount(mx(view.k)) == at and rowCount(mx(view.v)) == at;
        kd += try differing(mx(view.k), want_k, cx.s);
        kn += mlx.mlx_array_size(want_k);
        vd += try differing(mx(view.v), want_v, cx.s);
        vn += mlx.mlx_array_size(want_v);
        if (plant and li == plant_layer) try plants(cx, mx(view.k), want_k, at);
    }
    var cd: usize = 0;
    var cn: usize = 0;
    var sd: usize = 0;
    var sn: usize = 0;
    for (cx.kda) |li| {
        const conv = try cx.in.get("{s}.conv.c{d}.l{d}", .{ cx.prompt, ci, li });
        defer free(conv);
        const st = try cx.in.get("{s}.ssm.c{d}.l{d}", .{ cx.prompt, ci, li });
        defer free(st);
        cd += try differing(ssm[li].conv_state, conv, cx.s);
        cn += mlx.mlx_array_size(conv);
        sd += try differing(ssm[li].ssm_state, st, cx.s);
        sn += mlx.mlx_array_size(st);
        if (plant and li == cx.kda[0]) {
            const f = try flipped(ssm[li].ssm_state, cx.s);
            defer free(f);
            const d = try differing(f, st, cx.s);
            var wr: [192]u8 = undefined;
            const fr = try whereText(&wr, try firstDiff(f, st, cx.s), mlx.getShape(st));
            cx.line(d == 1, "RED {s}: one flipped bit in layer {d}'s restored ssm state (this test's copy): differing {d} (want 1); the differing word: layer {d}, {s}", .{ cx.name, li, d, li, fr });
        }
    }
    // The MTP head's cache: compared where the path restored it.
    var hd: usize = 0;
    var hn: usize = 0;
    var head_ok = !head_expected;
    var head_buf: [128]u8 = undefined;
    var head_note: []const u8 = "none restored (mtp_base none, 0 rows): the decline spill writes no spec sidecar until fork H's fix";
    if (res.mtp_base) |hb| {
        var hslot: cache_ops.Slot = .{ .kv = &head };
        var view: abi.KvView = .{};
        if (head.step > 0 and ops.view(hslot.handle(), 0, stream, &view).isOk() and view.k.ctx != null) {
            defer ops.release(&view);
            const want_k = try rows(cx.hk, @intCast(hb + head.step), cx.s);
            defer free(want_k);
            const want_v = try rows(cx.hv, @intCast(hb + head.step), cx.s);
            defer free(want_v);
            hd += try differing(mx(view.k), want_k, cx.s);
            hd += try differing(mx(view.v), want_v, cx.s);
            hn += mlx.mlx_array_size(want_k) + mlx.mlx_array_size(want_v);
            // Rows the path restored must be part A's, on either path (a spill that writes the sidecar, after fork H's
            // fix, is compared as a flush is).
            head_ok = hb == 0 and hb + head.step == at and hd == 0;
            if (plant_head) {
                const vs = mlx.getShape(want_v);
                const nz: c_int = @min(plant_rows, vs[2]);
                const pv = try zeroedTail(mx(view.v), nz, 0, cx.s);
                defer free(pv);
                try cx.plant("the head's form: the MTP head's last {d} restored rope-key rows (its V: rows {d} to {d}) zeroed", .{ nz, vs[2] - nz, vs[2] - 1 }, pv, want_v);
            }
        } else head_ok = false;
        head_note = try std.fmt.bufPrint(&head_buf, "base {d}, {d} rows, K and V words {d} differing {d}", .{ hb, head.step, hn, hd });
    } else if (head.step != 0) head_ok = false;
    cx.t.k += kn;
    cx.t.v += vn;
    cx.t.conv += cn;
    cx.t.ssm += sn;
    cx.t.head += hn;
    const ok = res.matched == at and rows_ok and kd == 0 and vd == 0 and cd == 0 and sd == 0 and head_ok;
    cx.t.rows += at * cx.mla.len;
    cx.line(ok, "{s} restore at checkpoint {d} ({s}): matched {d} (want {d}), the KV {d} rows; MLA: {d} layers x {d} rows = {d} rows, K words {d} differing {d}, V words {d} differing {d}; KDA: {d} layers, conv words {d} differing {d}, ssm words {d} differing {d}; MTP head: {s}", .{
        path, at, if (deepest) "the next turn's prompt" else "100 tokens past it", res.matched, at, kv.step, cx.mla.len, at, at * cx.mla.len, kn, kd, vn, vd, cx.kda.len, cn, cd, sn, sd, head_note,
    });
}

/// The decline spill's byte cap, stated with SPILL's own entry: it must hold its whole prompt, and the flush prompt
/// (`n_flush` tokens, `cps_flush`) through the same spill must stop short of it, which is why SPILL takes the shorter
/// one.
fn limit(cx: *Ctx, e: *const disk_tier.IndexEntry, n_flush: usize, cps_flush: []const u32) void {
    const cap: u64 = disk_tier.DECLINE_SPILL_FLUSH_FLOOR;
    const chunk_tokens: u64 = disk_tier.DEFAULT_CHUNK_TOKENS;
    const whole = e.kv_len == cx.tokens.len and std.mem.eql(u32, e.ssm_positions, cx.cps);
    var total: u64 = 0;
    for (e.chunk_bytes) |b| total += b;
    for (e.ssm_bytes) |b| total += b;
    const chunk: u64 = if (e.chunk_bytes.len > 0) e.chunk_bytes[0] else 0;
    const ckpt: u64 = if (e.ssm_bytes.len > 0) e.ssm_bytes[0] else 0;
    // The flush prompt through the same spill: every checkpoint first (inside half the cap), then chunk k while the
    // bytes written before it are under the cap (the first one always), as disk_tier.zig `appendCommitWithSpecBounded`.
    const nf: u64 = n_flush;
    const ncf: u64 = cps_flush.len;
    const ssm_f: u64 = ckpt * ncf;
    const chunks_f: u64 = (nf + chunk_tokens - 1) / chunk_tokens;
    var k: u64 = 0;
    while (chunk > 0 and k < chunks_f and (k == 0 or ssm_f + k * chunk < cap)) : (k += 1) {}
    const kv_f: u64 = @min(nf, k * chunk_tokens);
    var top: u32 = 0;
    for (cps_flush) |c| {
        if (c <= kv_f) top = c;
    }
    const ok = whole and kv_f < nf and ssm_f <= cap / 2;
    cx.line(ok, "LIMIT {s}: the decline spill writes at most {d} B a commit (disk_tier.zig DECLINE_SPILL_FLUSH_FLOOR): the checkpoints first, at most half of it, then {d}-token chunks while the bytes written are under it. SPILL's entry holds its whole prompt: kv_len {d} of {d}, checkpoints {any} (the plan's {any}), {d} chunks (a whole one {d} B) and {d} checkpoints ({d} B each), {d} B in all. At those sizes the flush prompt's {d} tokens and {d} checkpoints would stop after {d} chunks: kv_len {d}, restorable to its checkpoint {d} only. So SPILL takes the {d}-token prompt", .{
        cx.name, cap, chunk_tokens, e.kv_len, cx.tokens.len, e.ssm_positions, cx.cps, e.chunk_bytes.len, chunk, e.ssm_bytes.len, ckpt, total, nf, ncf, k, kv_f, top, cx.tokens.len,
    });
}

/// One commit path, end to end: part A's prompt of its name into a slot of its own, the commit to an SSD tier at the
/// served root under `tmp`'s folder (`tierRoot`), and a restore at each of its checkpoints. Part A's file is opened for
/// this path alone, so its arrays go when the path ends.
fn run(t: *Totals, in_path: [*:0]const u8, s: Stream, comptime path: Path, tmp: *std.testing.TmpDir, roots: *const Roots) !void {
    // part A's name for this path's prompt, and the lines' name for the run
    const p = comptime @tagName(path);
    const name = comptime if (path == .spill) "SPILL" else "FLUSH";
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &meta, in_path, cpu));
    const in: Input = .{ .arrays = arrays };
    const tokens = try in.u32s(p ++ ".tokens");
    defer gpa.free(tokens);
    const spans = try in.u32s(p ++ ".span_ends");
    defer gpa.free(spans);
    const cps = try in.u32s(p ++ ".checkpoints");
    defer gpa.free(cps);
    const mla = try in.u32s("mla_layers");
    defer gpa.free(mla);
    const kda = try in.u32s("kda_layers");
    defer gpa.free(kda);
    const dims = try in.u32s("mla_dims");
    defer gpa.free(dims);
    if (dims.len != 5) return error.BadMlaDims;
    const n_layers = mla.len + kda.len;
    const n = tokens.len;

    var cx: Ctx = .{ .in = in, .s = s, .t = t, .name = name, .prompt = p, .n_layers = n_layers, .mla = mla, .kda = kda, .tokens = tokens, .cps = cps, .rope = @intCast(dims[2]), .k = try gpa.alloc(Array, n_layers), .v = try gpa.alloc(Array, n_layers) };
    @memset(cx.k, .{ .ctx = null });
    @memset(cx.v, .{ .ctx = null });
    defer {
        for (cx.k) |a| free(a);
        for (cx.v) |a| free(a);
        gpa.free(cx.k);
        gpa.free(cx.v);
        free(cx.hk);
        free(cx.hv);
    }

    // The plan: the chunk loop's spans and checkpoints for this prompt, as generate/init.zig asks for them.
    {
        const stride = prefill_chunk.effectiveSsmCheckpointStride(base_stride, width);
        var plan = try prefill_chunk.wholePromptPlan(gpa, n, width, true, stride, 0, false, false);
        defer plan.deinit(gpa);
        var same = plan.spans.len == spans.len and plan.checkpoints.len == cps.len;
        if (same) for (plan.spans, spans) |a, b| {
            same = same and a == b;
        };
        if (same) for (plan.checkpoints, cps) |a, b| {
            same = same and a == b;
        };
        cx.line(same, "PLAN {s}: the host's wholePromptPlan({d}, {d}, stride {d}): spans {any}, checkpoints {any}; part A's spans {any}, checkpoints {any}: the same", .{ name, n, width, stride, plan.spans, plan.checkpoints, spans, cps });
    }

    // The slot, and the prefill list that CacheOps.checkpoint files on.
    var kv = try KVCache.init(gpa, @intCast(n_layers));
    defer kv.deinit();
    kv.reserve(n + width);
    const ssm = try newEntries(n_layers);
    defer freeEntries(ssm);
    var head = try KVCache.init(gpa, 1);
    defer head.deinit();
    const positions = try gpa.alloc(u64, cps.len);
    defer gpa.free(positions);
    for (positions, cps) |*pp, c| pp.* = c;
    var list: std.ArrayList(checkpoint.SSMCheckpoint) = .empty;
    defer {
        for (list.items) |*cp| cp.deinit(gpa);
        list.deinit(gpa);
    }
    var pc: checkpoint.PromptCheckpoints = .{ .allocator = gpa, .list = &list, .positions = positions, .layers = n_layers, .s = s, .max = cp_max, .thin = .oldest, .low_anchors = 0 };
    const sink = pc.sink();
    var slot: cache_ops.Slot = .{ .kv = &kv, .ssm = ssm, .checkpoints = &sink };
    var head_slot: cache_ops.Slot = .{ .kv = &head };
    const ops = &cache_ops.ops;
    const stream: abi.Stream = .{ .ctx = s.ctx };

    // Part A's writes, in its call order: each span's appends (the trunk's MLA layers, then the head), then at each
    // checkpoint every KDA layer's states, filed as plugin.zig `spanMajor` files them.
    var ci: usize = 0;
    for (spans, 0..) |end, j| {
        for (mla) |li| {
            const k = try in.get(p ++ ".k.l{d}.s{d}", .{ li, j });
            defer free(k);
            const v = try in.get(p ++ ".v.l{d}.s{d}", .{ li, j });
            defer free(v);
            var view: abi.KvView = .{};
            const verdict = ops.append(slot.handle(), li, ab(k), ab(v), 0, stream, &view);
            if (!verdict.isOk()) {
                std.debug.print("{s} FAIL: CacheOps.append refused layer {d} span {d}: {s}\n", .{ tag, li, j, verdict.why.slice() });
                return error.AppendRefused;
            }
            ops.release(&view);
        }
        {
            const k = try in.get(p ++ ".hk.s{d}", .{j});
            defer free(k);
            const v = try in.get(p ++ ".hv.s{d}", .{j});
            defer free(v);
            var view: abi.KvView = .{};
            if (!ops.append(head_slot.handle(), 0, ab(k), ab(v), 0, stream, &view).isOk()) return error.AppendRefused;
            ops.release(&view);
        }
        if (ci < cps.len and cps[ci] == end) {
            var states: [256]abi.LayerState = undefined;
            var held: [256][2]Array = undefined;
            if (kda.len > states.len) return error.TooManyLayers;
            var got: usize = 0;
            defer for (held[0..got]) |h| {
                free(h[0]);
                free(h[1]);
            };
            for (kda, 0..) |li, i| {
                const conv = try in.get(p ++ ".conv.c{d}.l{d}", .{ ci, li });
                const st = in.get(p ++ ".ssm.c{d}.l{d}", .{ ci, li }) catch |e| {
                    free(conv);
                    return e;
                };
                held[i] = .{ conv, st };
                got = i + 1;
                states[i] = .{ .layer = li, .conv = ab(conv), .ssm = ab(st) };
            }
            const op = cache_ops.ops.checkpointOp() orelse return error.NoCheckpointOp;
            const verdict = op(slot.handle(), end, abi.Span(abi.LayerState).of(states[0..kda.len]));
            if (!verdict.isOk()) {
                std.debug.print("{s} FAIL: CacheOps.checkpoint refused {d}: {s}\n", .{ tag, end, verdict.why.slice() });
                return error.CheckpointRefused;
            }
            ci += 1;
        }
    }

    // What the prefill wrote, joined at each layer, and what the slot holds after the appends.
    var hold_d: usize = 0;
    var hold_n: usize = 0;
    var hold_rows = true;
    for (mla) |li| {
        cx.k[li] = try written(in, p ++ ".k.l{d}.s{d}", false, li, spans.len, s);
        cx.v[li] = try written(in, p ++ ".v.l{d}.s{d}", false, li, spans.len, s);
        var view: abi.KvView = .{};
        if (!ops.view(slot.handle(), li, stream, &view).isOk()) {
            hold_rows = false;
            continue;
        }
        defer ops.release(&view);
        hold_rows = hold_rows and rowCount(mx(view.k)) == n;
        hold_d += try differing(mx(view.k), cx.k[li], s);
        hold_d += try differing(mx(view.v), cx.v[li], s);
        hold_n += mlx.mlx_array_size(cx.k[li]) + mlx.mlx_array_size(cx.v[li]);
    }
    cx.hk = try written(in, p ++ ".hk.s{d}", true, 0, spans.len, s);
    cx.hv = try written(in, p ++ ".hv.s{d}", true, 0, spans.len, s);
    {
        var view: abi.KvView = .{};
        if (ops.view(head_slot.handle(), 0, stream, &view).isOk()) {
            defer ops.release(&view);
            hold_rows = hold_rows and rowCount(mx(view.k)) == n;
            hold_d += try differing(mx(view.k), cx.hk, s);
            hold_d += try differing(mx(view.v), cx.hv, s);
            hold_n += mlx.mlx_array_size(cx.hk) + mlx.mlx_array_size(cx.hv);
        } else hold_rows = false;
    }
    // kv.step moves only when global layer 0 is written (kv/cache.zig); layer 0 is a KDA layer here, so it stays 0, as
    // it does in the served engine.
    cx.line(hold_rows and hold_d == 0 and list.items.len == cps.len and ci == cps.len, "HOLDS {s}: the slot after part A's writes: {d} MLA layers and the head at {d} rows each (kv.step {d}: it moves with layer 0 only, a KDA layer), K and V differing {d}/{d}; checkpoints filed {d} (want {d})", .{ name, mla.len, n, kv.step, hold_d, hold_n, list.items.len, cps.len });

    var buf: [512]u8 = undefined;
    const base = try tierRoot(tmp, &buf, p);
    const root = roots.served;
    switch (path) {
        // The served config: the RAM tier declines the commit and spills the candidate to the SSD tier; the same
        // process restores from it, as the next turn does.
        .spill => {
            var hc = prefix_cache.HotPrefixCache.initWithMem(gpa, ram_entries, 1);
            defer hc.deinit();
            try tiers(&hc, base, root);
            const owned = try list.toOwnedSlice(gpa);
            const st = try hc.commitWithState(&kv, tokens, false, &.{}, owned, null, .{ .cache = &head, .base_pos = 0 });
            hc.disk.?.drainWriter();
            const d = &hc.disk.?;
            cx.line(st == .declined and hc.entryCount() == 0 and d.entryCount() == 1, "{s} commit, the RAM tier at 1 byte (the clamp's floor): {s}; RAM entries {d}, SSD entries {d}", .{ name, @tagName(st), hc.entryCount(), d.entryCount() });
            const flush_tokens = try in.u32s("flush.tokens");
            defer gpa.free(flush_tokens);
            const flush_cps = try in.u32s("flush.checkpoints");
            defer gpa.free(flush_cps);
            if (onlyEntry(d)) |e| {
                limit(&cx, e, flush_tokens.len, flush_cps);
            } else cx.line(false, "LIMIT {s}: the SSD tier holds {d} entries, not one", .{ name, d.entryCount() });
            for (0..cps.len) |i| try restore(&cx, &hc, name, i, false, i + 1 == cps.len, false);
        },
        // A RAM tier with room keeps the entry; its flush writes it with the head's spec sidecar. A fresh tier over the
        // same folder (a restart) restores from the SSD tier alone.
        .flush => {
            {
                var hc = prefix_cache.HotPrefixCache.initWithMem(gpa, ram_entries, 0);
                defer hc.deinit();
                try tiers(&hc, base, root);
                const owned = try list.toOwnedSlice(gpa);
                const st = try hc.commitWithState(&kv, tokens, false, &.{}, owned, null, .{ .cache = &head, .base_pos = 0 });
                hc.flushPendingDisk(s);
                var slices: usize = 1;
                while (slices < 256) : (slices += 1) {
                    hc.disk.?.drainWriter();
                    if (hc.flushOwedSlice(s) == .done) break;
                }
                hc.disk.?.drainWriter();
                const d = &hc.disk.?;
                const e = onlyEntry(d);
                const kv_len: u32 = if (e) |x| x.kv_len else 0;
                const kept: []const u32 = if (e) |x| x.ssm_positions else &.{};
                const sidecar = if (e) |x| x.spec_mtp != null else false;
                const whole = kv_len == n and std.mem.eql(u32, kept, cps);
                cx.line(st == .ok and whole and sidecar, "{s} commit, a RAM tier with room: {s}; flushed in {d} slice(s): SSD entries {d}, kv_len {d} of {d}, checkpoints {any} (the plan's {any}), the MTP head's spec sidecar: {s}", .{ name, @tagName(st), slices, d.entryCount(), kv_len, n, kept, cps, if (sidecar) "yes" else "no" });
            }
            var hc = prefix_cache.HotPrefixCache.initWithMem(gpa, ram_entries, 0);
            defer hc.deinit();
            try tiers(&hc, base, root);
            for (0..cps.len) |i| try restore(&cx, &hc, name, i, true, false, i + 1 == cps.len);
        },
    }
}

/// ROOTS: the served namespace is named, and its root for `model_dir` differs from each earlier build's root
fn rootsLine(t: *Totals, roots: *const Roots) void {
    var apart = true;
    for (roots.earlier) |fp| apart = apart and !std.mem.eql(u8, roots.served, fp);
    const named = roots.ns.len > 0;
    t.line(apart and named, "ROOTS the SSD roots of {s} by layout namespace: the served cache's ({s}, part A's, as config.zig serves it) {s}; the earlier builds' roots {s} ({s}), {s} ({s}) and {s} ({s}): the served root apart from each {s}", .{
        model_dir,                          nsText(roots.ns),  roots.served,                       nsText(earlier_namespaces[0]), roots.earlier[0],
        nsText(earlier_namespaces[1]),      roots.earlier[1],  nsText(earlier_namespaces[2]),      roots.earlier[2],              if (apart) "yes" else "NO",
    });
}

/// CROSS: a restore across roots is refused by the roots' names. The SPILL folder (`tmp`, its `run`'s, which the caller
/// keeps until this check is done) holds the served entry alone, at the served root. Over that folder, with the deepest
/// SPILL restore's lookup (the next turn's prompt): a tier at each earlier build's root (`earlier_namespaces`) must hold
/// no entry and restore nothing; a tier at the served root (the control) must restore the entry, layer `plant_layer`'s
/// keys in the served layout.
fn crossRoot(t: *Totals, in_path: [*:0]const u8, s: Stream, tmp: *std.testing.TmpDir, roots: *const Roots) !void {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var arrays = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(arrays);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&arrays, &meta, in_path, cpu));
    const in: Input = .{ .arrays = arrays };
    const mla = try in.u32s("mla_layers");
    defer gpa.free(mla);
    const kda = try in.u32s("kda_layers");
    defer gpa.free(kda);
    const n_layers = mla.len + kda.len;
    const stream: abi.Stream = .{ .ctx = s.ctx };
    const tokens = try in.u32s("spill.tokens");
    defer gpa.free(tokens);
    const cps = try in.u32s("spill.checkpoints");
    defer gpa.free(cps);
    const deepest: c_int = @intCast(cps[cps.len - 1]);
    // the served keys at `plant_layer` as part A wrote its first span: [1, 1, rows, rope]
    const own_k = try in.get("spill.k.l{d}.s{d}", .{ plant_layer, 0 });
    defer free(own_k);
    const osh = mlx.getShape(own_k);
    const want = [4]c_int{ 1, osh[1], deepest, osh[3] };
    var q: std.ArrayList(u32) = .empty;
    defer q.deinit(gpa);
    try q.appendSlice(gpa, tokens);
    try q.appendSlice(gpa, &[_]u32{ 156_901, 156_902, 156_903 });
    var buf: [512]u8 = undefined;
    const base = try tierRoot(tmp, &buf, "spill");
    // each earlier build's root first, then the served root (the control)
    for (0..earlier_namespaces.len + 1) |i| {
        const own = i == earlier_namespaces.len;
        const root = if (own) roots.served else roots.earlier[i];
        var hc = prefix_cache.HotPrefixCache.initWithMem(gpa, ram_entries, 1);
        defer hc.deinit();
        try tiers(&hc, base, root);
        const entries = hc.disk.?.entryCount();
        var kv = try KVCache.init(gpa, @intCast(n_layers));
        defer kv.deinit();
        const ssm = try newEntries(n_layers);
        defer freeEntries(ssm);
        var head = try KVCache.init(gpa, 1);
        defer head.deinit();
        var head_base: usize = 0;
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&kv, &moe_off, ssm, s, q.items, false, &.{}, null, .{ .cache = &head, .base_pos = &head_base });
        t.cross += 1;
        var slot: cache_ops.Slot = .{ .kv = &kv, .ssm = ssm };
        var view: abi.KvView = .{};
        const viewed = cache_ops.ops.view(slot.handle(), plant_layer, stream, &view).isOk() and view.k.ctx != null;
        defer if (viewed) cache_ops.ops.release(&view);
        const got: []const c_int = if (viewed) mlx.getShape(mx(view.k)) else &.{};
        const restored: usize = if (viewed) rowCount(mx(view.k)) else 0;
        if (own) {
            const ok = entries == 1 and res.matched == cps[cps.len - 1] and std.mem.eql(c_int, got, &want);
            t.line(ok, "CROSS the served entry at its own root ({s}, {s}), the control: SSD entries {d} (want 1), matched {d} (want {d}), layer {d}'s restored K {any} (want {any}, the rope-once layout)", .{
                nsText(roots.ns), root, entries, res.matched, cps[cps.len - 1], plant_layer, got, want,
            });
        } else {
            const ok = entries == 0 and res.matched == 0 and restored == 0;
            t.line(ok, "CROSS the served entry under an earlier build's root ({s}, {s}): SSD entries {d} (want 0), matched {d} (want 0), layer {d}'s rows restored {d}: refused by the root's name", .{
                nsText(earlier_namespaces[i]), root, entries, res.matched, plant_layer, restored,
            });
        }
    }
}

test "turboexl3_ling restore bytes (B): the served host's decline spill and flush restore the served cache (the trunk's rope-once MLA cache and the MTP head's latent) and every checkpoint's states, byte for byte, under its own SSD root, and a tier at an earlier build's root finds none of it (MLX_SERVE_LING_RESTORE_IN)" {
    const in_path = std.c.getenv("MLX_SERVE_LING_RESTORE_IN") orelse return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    var roots = try Roots.load(in_path);
    defer roots.deinit();
    var t: Totals = .{};
    rootsLine(&t, &roots);
    // Each run's SSD folder goes when its checks are done, so the disk holds one entry's files at a time: SPILL's
    // until the cross-root lookups over it, FLUSH's until its restores.
    {
        var spill_dir = std.testing.tmpDir(.{ .iterate = true });
        defer spill_dir.cleanup();
        try run(&t, in_path, s, .spill, &spill_dir, &roots);
        try crossRoot(&t, in_path, s, &spill_dir, &roots);
    }
    var flush_dir = std.testing.tmpDir(.{ .iterate = true });
    defer flush_dir.cleanup();
    try run(&t, in_path, s, .flush, &flush_dir, &roots);
    const ok = t.failed == 0 and t.plants == 4 and t.cross == earlier_namespaces.len + 1;
    std.debug.print("{s} DONE checks={d} failed={d} restores={d} MLA rows compared={d} words compared K={d} V={d} conv={d} ssm={d} head={d} plants caught={d}/4 cross-root lookups={d}/4 {s}\n", .{
        tag, t.checks, t.failed, t.restores, t.rows, t.k, t.v, t.conv, t.ssm, t.head, t.plants, t.cross, if (ok) "BYTES_OK" else "BYTES_DIFFER",
    });
    try std.testing.expect(ok);
}
