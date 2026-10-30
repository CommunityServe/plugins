//! copied_from: src/plugins/qwen4_exp/qwen4_exp.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Qwen3.8-Flash-Next (`model_type` qwen4_exp) host-side pieces: the hashed
//! n-gram embedding (PLE) id math and the mmapped quantized n-gram table.
//!
//! The 51B-parameter table (320M rows x 160) is never copied to the GPU:
//! a token touches 16 rows, so the rows are dequantized from the mmap on the
//! host and only the [T, 2560] result is sent, or (serial forwards, when the
//! working set fits it) a kernel reads the mapping in place (`ple_gpu.zig`).
//! Format: `ngram_table.bin` is a safetensors-format file holding one merged
//! affine table (`weight` U32 [R, dim*bits/32], `scales`/`biases` BF16
//! [R, dim/gs]) written by `tests/convert_qwen38_flash_next.py`, or one
//! merged RAW BF16 table (`weight` BF16 [R, dim], no scales/biases,
//! `"bits":"16"`) for bit-exact PLE lookups. A fully EXL3 pack names its own
//! file (`ngram_table.file`): `exl3_ngram_trellis` rows, K-bit MUL1 trellis
//! rings decoded on the host, plus a per-head bias the pack's shards hold.

const std = @import("std");
const crt = @import("../../sdk/util/crt.zig");
const env = @import("../../env.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const log = @import("../../sdk/util/log.zig");
const ple_gpu = @import("ple_gpu.zig");
const embedded_ple = @import("ple_table.zig");
pub const EmbeddedSpec = embedded_ple.EmbeddedSpec;
pub const EmbeddedInfo = embedded_ple.EmbeddedInfo;
pub const inspectEmbedded = embedded_ple.inspectEmbedded;
pub const embeddedTensorName = embedded_ple.embeddedTensorName;

const SPLITMIX_GAMMA: u64 = 0x9E3779B97F4A7C15;
const SPLITMIX_M1: u64 = 0xBF58476D1CE4E5B9;
const SPLITMIX_M2: u64 = 0x94D049BB133111EB;
const PRIME_1: u64 = 10007;

fn splitmix64(v0: u64) u64 {
    var v = v0 +% SPLITMIX_GAMMA;
    v = (v ^ (v >> 30)) *% SPLITMIX_M1;
    v = (v ^ (v >> 27)) *% SPLITMIX_M2;
    return v ^ (v >> 31);
}

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

fn nthPrimeAfter(start: u64, count: u32) u64 {
    var p = start;
    for (0..count) |_| {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

/// Config-driven bounds; `model.validateQwen4Config` refuses a checkpoint past them at load.
pub const MAX_HEADS = 32;
pub const MAX_NGRAM_SIZE = 8;

/// Everything `Qwen4ExpTextNGramEmbedding.__init__` derives from the config.
pub const NgramHash = struct {
    ngram_size: u32,
    heads_per_ngram: u32,
    n_heads: u32,
    eos: u32,
    multipliers: [8]i64,
    vocab: [MAX_HEADS]i64,
    offsets: [MAX_HEADS]i64,
    total_rows: u64,

    /// `ple_layer_index` is the PLE's ordinal among the config's injection points (0 for the
    /// one we support). Fallible: every bound writes a fixed array.
    pub fn init(unigram_vocab: u32, ngram_size: u32, heads_per_ngram: u32, vocab_base: u64, divisor: u64, seed: u64, ple_layer_index: u32, eos: u32) !NgramHash {
        if (ngram_size < 2 or ngram_size > MAX_NGRAM_SIZE) return error.InvalidQwen4NgramSize;
        if (heads_per_ngram == 0 or (ngram_size - 1) * heads_per_ngram > MAX_HEADS) {
            return error.InvalidQwen4NgramHeads;
        }
        if (divisor == 0 or vocab_base < 2) return error.InvalidQwen4NgramVocab;
        var h: NgramHash = .{
            .ngram_size = ngram_size,
            .heads_per_ngram = heads_per_ngram,
            .n_heads = (ngram_size - 1) * heads_per_ngram,
            .eos = eos,
            .multipliers = @splat(0),
            .vocab = @splat(0),
            .offsets = @splat(0),
            .total_rows = 0,
        };
        const max_long: u64 = (1 << 63) - 1;
        const half_bound: u64 = @max(1, (max_long / @max(unigram_vocab, 1)) / 2);
        const base_seed: u64 = seed +% PRIME_1 *% ple_layer_index;
        for (0..ngram_size) |i| {
            const v = base_seed +% SPLITMIX_GAMMA *% (@as(u64, i) + 1);
            h.multipliers[i] = @intCast(2 * (splitmix64(v) % half_bound) + 1);
        }
        var total: u64 = 0;
        for (0..h.n_heads) |i| {
            const global = ple_layer_index * h.n_heads + @as(u32, @intCast(i));
            const size = nthPrimeAfter(vocab_base - 1, global + 1);
            h.vocab[i] = @intCast(size);
            h.offsets[i] = @intCast(total);
            total += size;
        }
        h.total_rows = (total + divisor - 1) / divisor * divisor;
        return h;
    }

    /// Row ids for `ids`, given the (ngram_size-1) tokens that precede them
    /// (`eos` for a fresh sequence). `out` is `[ids.len][n_heads]` row-major.
    /// Mirrors `_shift_right_ignore_eos` + the mixed-id hash: a shifted token
    /// is `eos` when the shift crosses the most recent eos before it.
    pub fn rowIds(self: *const NgramHash, prev: []const u32, ids: []const u32, out: []i64) void {
        const ctx: usize = self.ngram_size - 1;
        std.debug.assert(prev.len == ctx and out.len == ids.len * self.n_heads);
        var last_eos: i64 = -1;
        var t: usize = 0;
        while (t < ctx + ids.len) : (t += 1) {
            const tok = tokAt(prev, ids, t);
            if (t >= ctx) {
                const seg_pos: i64 = @as(i64, @intCast(t)) - (last_eos + 1);
                var mixed: i64 = @as(i64, tok) *% self.multipliers[0];
                const row = out[(t - ctx) * self.n_heads ..][0..self.n_heads];
                var n: usize = 2;
                var pos: usize = 1;
                while (n <= self.ngram_size) : (n += 1) {
                    while (pos < n) : (pos += 1) {
                        const shifted: u32 = if (seg_pos >= @as(i64, @intCast(pos)) and t >= pos) tokAt(prev, ids, t - pos) else self.eos;
                        mixed ^= @as(i64, shifted) *% self.multipliers[pos];
                    }
                    const h0 = (n - 2) * self.heads_per_ngram;
                    for (h0..h0 + self.heads_per_ngram) |h| {
                        row[h] = @mod(mixed, self.vocab[h]) + self.offsets[h];
                    }
                }
            }
            if (tok == self.eos) last_eos = @intCast(t);
        }
    }

    fn tokAt(prev: []const u32, ids: []const u32, i: usize) u32 {
        return if (i < prev.len) prev[i] else ids[i - prev.len];
    }
};

/// The merged quantized table, mmapped read-only.
pub var warm_override: ?bool = null;

fn warmEnabled() bool {
    return warm_override orelse true;
}

/// An EXL3 trunk pack's warm, decided at load: on when its loaded weights and the table both fit in the machine's RAM
/// (an unknown RAM size, 0, keeps it on); past that the table's pages only evict the weights'.
pub fn exl3WarmFits(weights: u64, table: u64, ram: u64) bool {
    return ram == 0 or weights +| table <= ram;
}

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const WarmState = struct {
    /// What the background page-cache warm has read so far, and the table's total size. Published
    /// by the warm thread, read lock-free by metrics and `/props`; zero when nothing is warming.
    live_warm_bytes: std.atomic.Value(u64) = .init(0),
    live_warm_total: std.atomic.Value(u64) = .init(0),
};
pub const warm_shared = @import("../../sdk/util/shared.zig").Shared(WarmState, "qwen4_warm", .{});
pub inline fn warmBytes() *WarmState {
    return warm_shared.get();
}
const empty_embedded_map: [0]u8 align(std.heap.page_size_min) = .{};

/// A progress line at each 8 GB step or after 10 s of silence, never twice per step. Pure.
pub const WARM_LOG_BYTES: u64 = 8 << 30;
pub const WARM_LOG_NS: u64 = 10_000_000_000;

pub const WarmProgress = struct {
    next_bytes: u64 = WARM_LOG_BYTES,
    last_ns: u64 = 0,

    pub fn should(self: *WarmProgress, bytes: u64, elapsed_ns: u64) bool {
        const by_bytes = bytes >= self.next_bytes;
        const by_time = elapsed_ns -| self.last_ns >= WARM_LOG_NS;
        if (!by_bytes and !by_time) return false;
        while (self.next_bytes <= bytes) self.next_bytes += WARM_LOG_BYTES;
        self.last_ns = elapsed_ns;
        return true;
    }
};

fn asGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

pub const NgramTable = struct {
    map: []align(std.heap.page_size_min) const u8,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    w_off: usize,
    s_off: usize,
    b_off: usize,
    wcols: u32,
    scols: u32,
    /// Kept open for the pool's `pread` gather (page faults on one mapping
    /// serialize on the VM map lock; preads run in parallel).
    fd: c_int = -1,
    pool: ?*PrefetchPool = null,
    /// Boot-time page-cache warm: the weights load evicts this file, and the
    /// first long prompt then faults 48 rows/token from SSD (38k: 174 s vs
    /// 55 s warm). preads through the kept fd, never the mapping.
    warm_thread: ?std.Thread = null,
    warm_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    warm_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Set once `ple_gpu.wrap` hands the mapping to a no-copy Metal buffer: MLX unmaps it
    /// when its last reference drops, so a kernel still in flight never reads freed pages.
    gpu_owns_map: bool = false,
    embedded: ?*embedded_ple.EmbeddedTable = null,
    fmt: embedded_ple.Format = .affine,
    /// An `exl3_ngram_trellis` table: each row a ring of `dim` K-bit symbols decoded through MUL1 (`wcols` halfwords:
    /// an f16 scale, then the symbol stream); 0 for the affine and BF16 formats.
    trellis_k: u32 = 0,
    /// The trellis rows' per-hash-head bias, [n_heads][dim] f16 values widened to f32, owned (`setHeadBias`).
    n_heads: u32 = 0,
    head_bias: ?[]f32 = null,
    /// The PLE prefetch's readers (`prefetchBegin`): a trellis table's prefill rows, read ahead of their gathers.
    ahead: ?*Ahead = null,

    pub fn openEmbedded(model_dir: []const u8, expected: EmbeddedSpec) !NgramTable {
        var source = (try embedded_ple.openEmbedded(model_dir, expected)) orelse return error.EmbeddedPleMissing;
        errdefer source.close();
        const ptr = try std.heap.page_allocator.create(embedded_ple.EmbeddedTable);
        ptr.* = source;
        var table: NgramTable = .{
            .map = empty_embedded_map[0..],
            .rows = ptr.rows,
            .dim = ptr.dim,
            .bits = ptr.bits,
            .group_size = ptr.group_size,
            .w_off = 0,
            .s_off = 0,
            .b_off = 0,
            .wcols = ptr.wcols,
            .scols = ptr.scols,
            .embedded = ptr,
            .fmt = ptr.fmt,
        };
        table.pool = PrefetchPool.create() catch null;
        return table;
    }

    pub fn embeddedPayloadBytes(self: *const NgramTable) u64 {
        return if (self.embedded) |source| source.payload_bytes else 0;
    }

    pub fn open(path: []const u8) !NgramTable {
        if (comptime @import("builtin").os.tag == .windows) return error.NgramTableUnsupportedOnWindows;
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..path.len], path);
        pbuf[path.len] = 0;
        const fd = crt.openRead(pbuf[0..path.len :0]);
        if (fd < 0) return error.FileNotFound;
        errdefer crt.close(fd);
        // File size via lseek-to-end: 0.17 has no portable fstat wrapper on
        // Linux (std.c.Stat is void there) and this loader has no `std.Io`.
        const size: usize = @intCast(@max(crt.seekEnd(fd), 0));
        if (size == 0) return error.StatFailed;
        const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        errdefer std.posix.munmap(map);
        if (size < 8) return error.NgramTableTruncated;
        const hlen: usize = @intCast(std.mem.readInt(u64, map[0..8], .little));
        if (hlen > size - 8) return error.NgramTableTruncated;
        var t = try parse(map, map[8 .. 8 + hlen], 8 + hlen);
        t.fd = fd;
        t.pool = PrefetchPool.create() catch null;
        if (t.trellis_k != 0 and t.pool != null) t.ahead = Ahead.create() catch null;
        return t;
    }

    /// Widths `mx.quantize` packs and `dequantRow` unpacks, plus 16 = raw
    /// BF16 rows (no scales/biases; `row` copies them out converted to f32).
    fn bitsSupported(bits: u32) bool {
        return switch (bits) {
            2, 3, 4, 5, 6, 8, 16 => true,
            else => false,
        };
    }

    const HeaderRegion = embedded_ple.HeaderRegion;

    /// One header entry, every access checked and the region proven to hold exactly
    /// `rows x cols x elem` bytes inside the mapping.
    fn headerRegion(
        obj: std.json.ObjectMap,
        key: []const u8,
        dtype: []const u8,
        elem: u64,
        map_len: usize,
        data_off: usize,
    ) !HeaderRegion {
        const r = embedded_ple.headerRegion(obj, key, dtype, elem, map_len, data_off, .matrix) catch |err| return switch (err) {
            error.TensorHeader, error.TensorDtype => error.NgramTableHeader,
            error.TensorRegion => error.NgramTableRegion,
            error.TensorTruncated => error.NgramTableTruncated,
        };
        if (r.cols > std.math.maxInt(u32) or r.rows > std.math.maxInt(u32)) return error.NgramTableRegion;
        return r;
    }

    /// Absent `format` stamp is accepted once, loudly; a different format is a refusal.
    var stamp_warned: bool = false;

    fn parse(map: []align(std.heap.page_size_min) const u8, header: []const u8, data_off: usize) !NgramTable {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.NgramTableHeader;
        if (parsed != .object) return error.NgramTableHeader;
        const obj = parsed.object;
        const meta_v = obj.get("__metadata__") orelse return error.NgramTableHeader;
        if (meta_v != .object) return error.NgramTableHeader;
        const meta = meta_v.object;
        if (meta.get("format")) |f| if (f == .string and std.mem.eql(u8, f.string, "exl3_ngram_trellis")) return parseTrellis(map, obj, meta, data_off);
        if (meta.get("format")) |f| {
            if (f != .string or !std.mem.eql(u8, f.string, "mlx-serve-ngram")) return error.NgramTableHeader;
        } else if (!stamp_warned) {
            stamp_warned = true;
            log.info("[qwen4] ngram table has no \"format\" stamp (written before the converter added it); accepting\n", .{});
        }
        const bits_v = meta.get("bits") orelse return error.NgramTableHeader;
        const gs_v = meta.get("group_size") orelse return error.NgramTableHeader;
        if (bits_v != .string or gs_v != .string) return error.NgramTableHeader;
        const bits: u32 = std.fmt.parseInt(u32, bits_v.string, 10) catch return error.NgramTableHeader;
        const gs: u32 = std.fmt.parseInt(u32, gs_v.string, 10) catch return error.NgramTableHeader;
        if (!bitsSupported(bits)) return error.NgramTableBits;

        // Raw BF16 mode: one `weight` BF16 [rows, dim] region, no
        // scales/biases. `wcols`/`scols` stay 0; only `row`'s raw arm reads it.
        if (bits == 16) {
            const w = try headerRegion(obj, "weight", "BF16", 2, map.len, data_off);
            if (w.cols > std.math.maxInt(u32)) return error.NgramTableRegion;
            return .{
                .map = map,
                .rows = w.rows,
                .dim = @intCast(w.cols),
                .bits = bits,
                .group_size = gs,
                .w_off = data_off + @as(usize, @intCast(w.start)),
                .s_off = 0,
                .b_off = 0,
                .wcols = 0,
                .scols = 0,
            };
        }

        if (gs == 0 or gs > 1024) return error.NgramTableBits;

        const w = try headerRegion(obj, "weight", "U32", 4, map.len, data_off);
        const sc = try headerRegion(obj, "scales", "BF16", 2, map.len, data_off);
        const bi = try headerRegion(obj, "biases", "BF16", 2, map.len, data_off);
        // The three regions describe the same rows and may not overlap.
        if (sc.rows != w.rows or bi.rows != w.rows or sc.cols != bi.cols) return error.NgramTableRegion;
        if (w.overlaps(sc) or w.overlaps(bi) or sc.overlaps(bi)) return error.NgramTableRegion;

        const dim: u64 = sc.cols * gs;
        if (dim > std.math.maxInt(u32)) return error.NgramTableRegion;
        if (dim * bits != w.cols * 32) return error.NgramTableHeader;

        return .{
            .map = map,
            .rows = w.rows,
            .dim = @intCast(dim),
            .bits = bits,
            .group_size = gs,
            .w_off = data_off + @as(usize, @intCast(w.start)),
            .s_off = data_off + @as(usize, @intCast(sc.start)),
            .b_off = data_off + @as(usize, @intCast(bi.start)),
            .wcols = @intCast(w.cols),
            .scols = @intCast(sc.cols),
        };
    }

    /// The `exl3_ngram_trellis` format: one I16 tensor [rows, 1 + dim*K/16] whose row is the ring's f16 scale and then
    /// its dim*K-bit stream (stream bits [i*K, (i+1)*K) are symbol i). Metadata: `K` 2..8, `row_dim`, codebook MUL1.
    fn parseTrellis(map: []align(std.heap.page_size_min) const u8, obj: std.json.ObjectMap, meta: std.json.ObjectMap, data_off: usize) !NgramTable {
        const k_v = meta.get("K") orelse return error.NgramTableHeader;
        if (k_v != .string) return error.NgramTableHeader;
        const K = std.fmt.parseInt(u32, k_v.string, 10) catch return error.NgramTableHeader;
        if (K < 2 or K > 8) return error.NgramTableBits;
        const dim_v = meta.get("row_dim") orelse return error.NgramTableHeader;
        if (dim_v != .string) return error.NgramTableHeader;
        const dim = std.fmt.parseInt(u32, dim_v.string, 10) catch return error.NgramTableHeader;
        if (dim == 0 or dim > max_trellis_dim or (dim * K) % 16 != 0) return error.NgramTableHeader;
        if (meta.get("codebook")) |c| if (c != .string or !std.mem.eql(u8, c.string, "mul1")) return error.NgramTableHeader;
        var key: ?[]const u8 = null;
        var it = obj.iterator();
        while (it.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, ".trellis")) {
            if (key != null) return error.NgramTableHeader;
            key = e.key_ptr.*;
        };
        const r = try headerRegion(obj, key orelse return error.NgramTableHeader, "I16", 2, map.len, data_off);
        if (r.cols != 1 + (dim * K) / 16) return error.NgramTableHeader;
        return .{
            .map = map,
            .rows = r.rows,
            .dim = dim,
            .bits = K,
            .group_size = 0,
            .w_off = data_off + @as(usize, @intCast(r.start)),
            .s_off = 0,
            .b_off = 0,
            .wcols = @intCast(r.cols),
            .scols = 0,
            .trellis_k = K,
        };
    }

    /// The trellis rows' per-hash-head bias (`n_heads` x `dim` f16 bits), copied widened to f32.
    pub fn setHeadBias(self: *NgramTable, bias_bits: []const u16, n_heads: u32) !void {
        if (n_heads == 0 or bias_bits.len != @as(usize, n_heads) * self.dim) return error.NgramTableHeader;
        const wide = try std.heap.page_allocator.alloc(f32, bias_bits.len);
        for (wide, bias_bits) |*w, b| w.* = @floatCast(@as(f16, @bitCast(b)));
        if (self.head_bias) |old| std.heap.page_allocator.free(old);
        self.head_bias = wide;
        self.n_heads = n_heads;
    }

    /// Row `r` of a trellis table decoded from the mapping, plus `head`'s bias (null: the stored ring alone).
    fn trellisRowMapped(self: *const NgramTable, r: u64, head: ?u32, out: []f32) void {
        const nb: usize = @as(usize, self.wcols) * 2;
        self.trellisDecode(self.map[self.w_off + r * nb ..][0..nb], head, out);
    }

    /// The ring on a row's bytes (`raw`: the f16 scale word, then the dim*K-bit stream) at the table's K.
    fn trellisDecode(self: *const NgramTable, raw: []const u8, head: ?u32, out: []f32) void {
        switch (self.trellis_k) {
            inline 2...8 => |k| self.trellisDecodeK(k, raw, head, out),
            else => unreachable, // `parseTrellis` admits K 2..8
        }
    }

    /// value_i = mul1(state_i) * scale (+ the head's bias_i): state_i stacks symbols i, i-1, ... (K bits each, circular
    /// over dim) from bit 0 into a 16-bit codeword. The symbols are cut from a copy of the stream padded with its first
    /// two bytes (a K <= 8 symbol spans two bytes, the ring wraps there), the states built 16 lanes at a time, MUL1
    /// through `mul1_by_sum`.
    fn trellisDecodeK(self: *const NgramTable, comptime K: u32, raw: []const u8, head: ?u32, out: []f32) void {
        const L = 16;
        const V = @Vector(L, u32);
        const stack: u32 = (15 + K) / K;
        const lead: usize = stack - 1;
        const kmask: u32 = (1 << K) - 1;
        const dim: usize = self.dim;
        const nbytes: usize = dim * K / 8;
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[0..2], .little))));
        var sb: [max_trellis_dim + 2]u8 = undefined;
        @memcpy(sb[0..nbytes], raw[2..][0..nbytes]);
        sb[nbytes] = sb[0];
        sb[nbytes + 1] = sb[1];
        // syms[lead + i] = symbol i; syms[0 .. lead] = the ring's last symbols, so state i reads syms[lead + i - j]
        var syms: [8 + max_trellis_dim]u32 = undefined;
        for (0..dim) |i| {
            const bit = i * K;
            const w: u32 = std.mem.readInt(u16, sb[bit >> 3 ..][0..2], .little);
            syms[lead + i] = (w >> @intCast(bit & 7)) & kmask;
        }
        for (0..lead) |j| syms[j] = syms[dim + j];
        const bias: ?[]const f32 = if (head) |h| (if (self.head_bias) |hb| hb[@as(usize, h) * dim ..][0..dim] else null) else null;
        const vscale: @Vector(L, f32) = @splat(scale);
        var i: usize = 0;
        while (i + L <= dim) : (i += L) {
            var st: V = syms[lead + i ..][0..L].*;
            inline for (1..stack) |j| st |= @as(V, syms[lead + i - j ..][0..L].*) << @as(@Vector(L, u5), @splat(@intCast(j * K)));
            const code = st & @as(V, @splat(0xFFFF));
            const x = code *% @as(V, @splat(mul1_mult));
            const y = (x & @as(V, @splat(0x00FF00FF))) + ((x >> @as(@Vector(L, u5), @splat(8))) & @as(V, @splat(0x00FF00FF)));
            const t = (y & @as(V, @splat(0xFFFF))) + (y >> @as(@Vector(L, u5), @splat(16)));
            var m: @Vector(L, f32) = undefined;
            inline for (0..L) |k| m[k] = mul1_by_sum[t[k]];
            var v = m * vscale;
            if (bias) |b| v += @as(@Vector(L, f32), b[i..][0..L].*);
            out[i..][0..L].* = v;
        }
        while (i < dim) : (i += 1) {
            var state: u32 = syms[lead + i];
            inline for (1..stack) |j| state |= syms[lead + i - j] << @intCast(j * K);
            var v: f32 = mul1_by_sum[mul1ByteSum(@truncate(state))] * scale;
            if (bias) |b| v += b[i];
            out[i] = v;
        }
    }

    /// A gather's trellis rows, row i (of the gather) at head `(first + i) % n_heads`.
    fn gatherTrellis(self: *const NgramTable, row_ids: []const i64, first: usize, out: []f32) void {
        const heads: usize = @max(self.n_heads, 1);
        for (row_ids, first..) |r, i| self.trellisRowMapped(@intCast(r), @as(u32, @intCast(i % heads)), out[(i - first) * self.dim ..][0..self.dim]);
    }

    /// Whether row `r`'s trellis bytes are in the page cache (`mincore` over the pages they span).
    fn rowResident(self: *const NgramTable, r: u64) bool {
        if (comptime @import("builtin").os.tag == .windows) return true; // `open` refuses there
        if (cold_test_force) return false;
        const nb: usize = @as(usize, self.wcols) * 2;
        const page = std.heap.pageSize();
        const off = self.w_off + @as(usize, @intCast(r)) * nb;
        const a0 = off & ~(page - 1);
        const a1 = (off + nb - 1) & ~(page - 1);
        var vec: [2]u8 = .{ 0, 0 };
        if (mincore(self.map.ptr + a0, a1 - a0 + page, &vec) != 0) return false;
        return (vec[0] & 1) != 0 and (a1 == a0 or (vec[1] & 1) != 0);
    }

    pub fn close(self: *NgramTable) void {
        // The readers pread through `fd`: joined before it closes.
        if (self.ahead) |a| a.destroy();
        self.ahead = null;
        if (self.warm_thread) |th| {
            self.warm_stop.store(true, .release);
            th.join();
            self.warm_thread = null;
        }
        warmBytes().live_warm_bytes.store(0, .release);
        warmBytes().live_warm_total.store(0, .release);
        if (self.pool) |p| p.destroy();
        self.pool = null;
        if (self.head_bias) |hb| std.heap.page_allocator.free(hb);
        self.head_bias = null;
        if (self.fd >= 0) crt.close(self.fd);
        self.fd = -1;
        if (self.embedded) |source| {
            source.close();
            std.heap.page_allocator.destroy(source);
            self.embedded = null;
        } else if (!self.gpu_owns_map and !(comptime @import("builtin").os.tag == .windows)) std.posix.munmap(self.map); // never mapped on Windows
    }

    /// Sample before background warming; explicit read-path overrides skip calibration.
    pub fn calibrateArm(self: *const NgramTable) void {
        self.calibrate(.load);
    }

    /// Only the inference thread may use the reader pool or update its measured policy.
    fn refreshAfterWarm(self: *const NgramTable) void {
        const p = self.pool orelse return;
        if (plePrefillPrefetchMode() == .kv_gated and p.warm_finished.swap(false, .acq_rel))
            self.calibrate(.warmed);
    }

    fn calibrate(self: *const NgramTable, phase: CalibrationPhase) void {
        // A trellis table's pooled arm decodes on the workers (`gather`): no read path to measure.
        if (self.trellis_k != 0 or self.bits == 16 or plePrefillPrefetchMode() != .kv_gated) return;
        const p = self.pool orelse return;
        if ((self.fd < 0 and self.embedded == null) or self.rows < 2 * PLE_CALIBRATION_ROWS) return;
        if (self.wcols * 4 + self.scols * 4 > PrefetchPool.ROW_BUF) return;

        // Each arm times the gather's own read + dequant on fresh rows, and the verdict is
        // the median trial, so one stalled sample cannot pick the arm.
        const out = std.heap.page_allocator.alloc(f32, PLE_CALIBRATION_ROWS * @as(usize, self.dim)) catch return;
        defer std.heap.page_allocator.free(out);
        const io = std.Io.Threaded.global_single_threaded.io();
        var serial_ns: [PLE_CALIBRATION_TRIALS]u64 = undefined;
        var pool_ns: [PLE_CALIBRATION_TRIALS]u64 = undefined;
        for (0..PLE_CALIBRATION_TRIALS) |trial| {
            var serial_rows: [PLE_CALIBRATION_ROWS]i64 = undefined;
            var pool_rows: [PLE_CALIBRATION_ROWS]i64 = undefined;
            calibrationRows(self.rows, &serial_rows, &pool_rows, phase, trial);
            var sw = io_util.Stopwatch.init(io);
            self.gatherSerial(&serial_rows, out);
            serial_ns[trial] = sw.read();
            sw.reset();
            if (!self.gatherPooled(p, &pool_rows, out)) return;
            pool_ns[trial] = sw.read();
        }
        const serial = medianNs(&serial_ns);
        const pool = medianNs(&pool_ns);
        if (serial == 0 or pool == 0) return;
        p.recordCalibration(.{ .serial_ns = serial, .pool_ns = pool, .phase = phase });
        log.info("[qwen4] ngram gather calibration ({s}): {s} (median of {d} trials, {d} rows per arm: serial {d:.2} ms, pool {d:.2} ms)\n", .{
            @tagName(phase),
            if (p.prefer_pool) "POOLED" else "SERIAL",
            PLE_CALIBRATION_TRIALS,
            PLE_CALIBRATION_ROWS,
            @as(f64, @floatFromInt(serial)) / 1e6,
            @as(f64, @floatFromInt(pool)) / 1e6,
        });
    }

    fn gatherSerial(self: *const NgramTable, row_ids: []const i64, out: []f32) void {
        for (row_ids, 0..) |r, i| self.row(@intCast(r), out[i * self.dim ..][0..self.dim]);
    }

    /// Workers pread each round's rows, the caller dequantizes them. False when a round
    /// fails; `out` is then partial and the caller re-reads it serially.
    fn gatherPooled(self: *const NgramTable, p: *PrefetchPool, row_ids: []const i64, out: []f32) bool {
        const wl: usize = self.wcols * 4;
        const sl: usize = self.scols * embedded_ple.scaleBytes(self.fmt);
        var start: usize = 0;
        while (start < row_ids.len) : (start += PrefetchPool.MAX_ROWS) {
            const end = @min(start + PrefetchPool.MAX_ROWS, row_ids.len);
            if (!p.run(self, row_ids[start..end])) return false;
            for (start..end) |i| {
                const b = &p.bufs[i - start];
                self.dequantRow(b[0..wl], b[wl .. wl + sl], b[wl + sl .. wl + 2 * sl], out[i * self.dim ..][0..self.dim]);
            }
        }
        return true;
    }

    const WARM_CHUNK: usize = 8 << 20;

    /// The `i`-th byte range the warm reads: the whole file, or an embedded pack's shard parts.
    fn warmRegion(self: *const NgramTable, i: usize) ?embedded_ple.Region {
        if (self.embedded) |source| return source.region(i);
        return if (i == 0 and self.fd >= 0) .{ .fd = self.fd, .off = 0, .len = self.map.len } else null;
    }

    /// The bytes the warm reads (`warmRegion`'s ranges).
    pub fn warmTotal(self: *const NgramTable) u64 {
        var total: u64 = 0;
        var i: usize = 0;
        while (self.warmRegion(i)) |r| : (i += 1) total += r.len;
        return total;
    }

    /// Read the whole table through its fds once, in the background, so the
    /// first prompt's PLE gathers hit a warm page cache. Call only once the
    /// table sits at its final address (the thread holds `self`).
    pub fn startWarm(self: *NgramTable) void {
        if (self.warmRegion(0) == null or self.warm_thread != null) return;
        // The off arm says so: a cold first request faults rows off the SSD (38k prompt: 174 s vs 55 s).
        if (!warmEnabled()) {
            log.info("[qwen4] ngram table warm: disabled - the first long prompt faults the table in from SSD\n", .{});
            return;
        }
        var total: u64 = 0;
        var i: usize = 0;
        while (self.warmRegion(i)) |r| : (i += 1) total += r.len;
        self.warm_stop.store(false, .release);
        self.warm_bytes.store(0, .release);
        warmBytes().live_warm_bytes.store(0, .release);
        warmBytes().live_warm_total.store(total, .release);
        log.info("[qwen4] ngram table warm: started, {d:.1} GB in the background (page cache)\n", .{asGb(total)});
        self.warm_thread = std.Thread.spawn(.{}, warmMain, .{ self, total }) catch null;
    }

    fn warmMain(self: *NgramTable, total: u64) void {
        var scratch: [WARM_CHUNK]u8 align(16) = undefined;
        const wio = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(wio, .boot);
        var done: u64 = 0;
        var prog: WarmProgress = .{};
        var i: usize = 0;
        while (self.warmRegion(i)) |r| : (i += 1) {
            var off: u64 = 0;
            while (off < r.len) {
                if (self.warm_stop.load(.acquire)) return;
                const want: usize = @intCast(@min(r.len - off, WARM_CHUNK));
                const got = crt.pread(r.fd, &scratch, want, @intCast(r.off + off));
                if (got <= 0) return;
                off += @intCast(got);
                done += @intCast(got);
                self.warm_bytes.store(done, .release);
                warmBytes().live_warm_bytes.store(done, .release);
                // One clock read per 8 MB pread is free next to the read itself.
                const el: u64 = @intCast(t0.untilNow(wio, .boot).nanoseconds);
                if (prog.should(done, el)) log.info("[qwen4] ngram table warm: {d:.1}/{d:.1} GB after {d:.0} s\n", .{ asGb(done), asGb(total), @as(f64, @floatFromInt(el)) / 1e9 });
            }
        }
        // Completion does not imply residency when the table exceeds available RAM.
        // Publish a request; recalibration must not race the inference thread's pool use.
        if (self.pool) |p| p.warm_finished.store(true, .release);
        const secs: f64 = @as(f64, @floatFromInt(t0.untilNow(wio, .boot).nanoseconds)) / 1e9;
        log.info("[qwen4] ngram table warm: done, {d:.1} GB in {d:.1} s (page cache)\n", .{ asGb(total), secs });
    }

    /// Dequantize one row into `out[0..dim]` (mx.quantize packing: element i
    /// sits at bit offset i * bits of the little-endian u32 stream and may
    /// straddle a word boundary at 3/5/6 bits).
    /// A trellis table's row is its stored ring, no head's bias (`gather` adds each row's).
    pub fn row(self: *const NgramTable, r: u64, out: []f32) void {
        std.debug.assert(r < self.rows and out.len >= self.dim);
        if (self.trellis_k != 0) return self.trellisRowMapped(r, null, out);
        if (self.embedded) |source| {
            const parts = source.rowParts(r);
            self.dequantRow(parts.weight, parts.scales, parts.biases, out);
            return;
        }
        // Raw BF16 arm: straight convert, no scales/biases.
        if (self.bits == 16) {
            const raw = self.map[self.w_off + r * self.dim * 2 ..][0 .. self.dim * 2];
            var i: u32 = 0;
            while (i < self.dim) : (i += 1) {
                out[i] = bf16ToF32(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            }
            return;
        }
        const words = self.map[self.w_off + r * self.wcols * 4 ..][0 .. self.wcols * 4];
        const scales = self.map[self.s_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        const biases = self.map[self.b_off + r * self.scols * 2 ..][0 .. self.scols * 2];
        self.dequantRow(words, scales, biases, out);
    }

    fn dequantRow(self: *const NgramTable, words: []const u8, scales: []const u8, biases: []const u8, out: []f32) void {
        if (self.fmt == .nvfp4) return dequantNvfp4Row(words, scales, self.embedded.?.scale, out[0..self.dim]);
        const mask: u32 = (@as(u32, 1) << @intCast(self.bits)) - 1;
        // An embedded pack's global weight scale; 1.0 leaves every other table exact.
        const ws: f32 = if (self.embedded) |source| source.scale else 1.0;
        var i: u32 = 0;
        while (i < self.dim) : (i += 1) {
            const off = i * self.bits;
            const w = off / 32;
            const shift = off % 32;
            var v: u64 = std.mem.readInt(u32, words[w * 4 ..][0..4], .little);
            if (shift + self.bits > 32) v |= @as(u64, std.mem.readInt(u32, words[w * 4 + 4 ..][0..4], .little)) << 32;
            const q: u32 = @truncate((v >> @intCast(shift)) & mask);
            const g = i / self.group_size;
            const sc = bf16ToF32(std.mem.readInt(u16, scales[g * 2 ..][0..2], .little));
            const bi = bf16ToF32(std.mem.readInt(u16, biases[g * 2 ..][0..2], .little));
            out[i] = (@as(f32, @floatFromInt(q)) * sc + bi) * ws;
        }
    }

    /// Gather + concatenate the `n_heads` rows of each token: `out` is
    /// `[ids.len / n_heads][n_heads * dim]` row-major. `kv_len` is the context position this
    /// gather runs at and picks the wide arm; the output is byte-identical either way.
    pub fn gather(self: *const NgramTable, row_ids: []const i64, out: []f32, kv_len: u64) void {
        // Trellis rows arrive token-major, head-minor (`NgramHash.rowIds`): row i takes head i % n_heads's bias. The ring
        // decode, not the read, is the cost, so a prefill-width gather decodes on the pool's workers.
        if (self.trellis_k != 0) {
            const wide = row_ids.len > PrefetchPool.MAX_ROWS;
            if (wide and plePrefillPrefetchMode() != .off) if (self.pool) |p| {
                // the rows the prefetch staged for this gather decode from their staged bytes (`AheadView.take`)
                var view: ?AheadView = if (self.ahead) |a| a.beginGather(kv_len, row_ids) else null;
                p.ahead_view = if (view) |*v| v else null;
                p.runTrellis(self, row_ids, out);
                p.ahead_view = null;
                if (view) |*v| self.ahead.?.endGather(v);
                return;
            };
            if (!wide and self.fd >= 0 and !cold_pread_disabled_for_test) if (self.pool) |p| if (p.coldTrellisRows(self, row_ids, out)) return;
            return self.gatherTrellis(row_ids, 0, out);
        }
        // Raw BF16 arm rides no pool (single region, no dequant): serial walk.
        if (self.bits == 16) return self.gatherSerial(row_ids, out);
        const need: usize = self.wcols * 4 + self.scols * embedded_ple.scaleBytes(self.fmt) * 2;
        // Avoid pool wakeups on resident tables unless calibration or long KV warrants them.
        const wide = row_ids.len > PrefetchPool.MAX_ROWS;
        if (wide) self.refreshAfterWarm();
        const prefer_pool = if (self.pool) |p| p.prefer_pool else false;
        const wide_ok = !wide or plePrefillPrefetchEnabled(kv_len, prefer_pool);
        // Announce the arm that actually runs, not the lever that permits it.
        const readable = self.fd >= 0 or self.embedded != null;
        const pooled = wide_ok and self.pool != null and readable and need <= PrefetchPool.ROW_BUF;
        if (wide) notePrefillGatherArm(pooled, row_ids.len, prefer_pool);
        if (self.pool) |p| if (readable and need <= PrefetchPool.ROW_BUF and wide_ok) {
            if (self.gatherPooled(p, row_ids, out)) return;
        };
        self.gatherSerial(row_ids, out);
    }

    /// One (row, region) pread into the pool's row buffer. False on a short read.
    fn preadSite(self: *const NgramTable, r: u64, region: usize, buf: []u8) bool {
        if (comptime @import("builtin").os.tag == .windows) unreachable; // `open` refuses on Windows
        const wl: usize = self.wcols * 4;
        const sl: usize = self.scols * embedded_ple.scaleBytes(self.fmt);
        const off: usize, const dst: []u8 = switch (region) {
            0 => .{ self.w_off + r * wl, buf[0..wl] },
            1 => .{ self.s_off + r * sl, buf[wl .. wl + sl] },
            else => .{ self.b_off + r * sl, buf[wl + sl .. wl + 2 * sl] },
        };
        if (self.embedded) |source| return source.preadPart(r, region, dst);
        return crt.pread(self.fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }

    /// A prefill of `ids`, its first token at kv position `base` after the n-gram history `prev`, is about to run: every
    /// chunk's rows are planned and read in the background, in position order, so each chunk's gather finds them staged.
    /// Ends any open session first. A trellis table on the pooled gather only, and a prompt long enough to hide behind.
    pub fn prefetchBegin(self: *NgramTable, hash: *const NgramHash, prev: []const u32, ids: []const u32, base: u64) void {
        self.prefetchEnd();
        if (plePrefillPrefetchMode() == .off) return;
        if (self.trellis_k == 0) return aheadDeclined("not a trellis table");
        if (self.fd < 0) return aheadDeclined("an embedded table (its rows sit in the model's shards)");
        if (self.pool == null) return aheadDeclined("no reader pool");
        if (ids.len < AHEAD_MIN_TOKENS or hash.n_heads == 0 or prev.len != hash.ngram_size - 1) return;
        const nb: usize = @as(usize, self.wcols) * 2;
        if (nb == 0 or nb > PrefetchPool.ROW_BUF) return aheadDeclined("a trellis row wider than a reader's buffer");
        const a = self.ahead orelse blk: {
            const made = Ahead.create() catch |err| {
                if (!ahead_failed_said.swap(true, .monotonic)) log.warn("[qwen4] PLE prefetch: no reader threads ({s}); the gathers read their own rows\n", .{@errorName(err)});
                return;
            };
            self.ahead = made;
            break :blk made;
        };
        a.begin(.{ .hash = hash, .prev = prev, .ids = ids, .base = base, .fd = self.fd, .w_off = self.w_off, .nb = nb, .budget = AHEAD_BUDGET }) catch |err| {
            if (!ahead_failed_said.swap(true, .monotonic)) log.warn("[qwen4] PLE prefetch: no session ({s}); the gathers read their own rows\n", .{@errorName(err)});
            return;
        };
        if (!ahead_said.swap(true, .monotonic)) log.info("[qwen4] PLE prefetch engaged: {d} readers stage each prefill's rows ahead of its gathers\n", .{a.n_threads});
    }

    /// The prefetch's first decline for a reason the table fixes, said once: each gather reads its own rows.
    fn aheadDeclined(why: []const u8) void {
        if (!ahead_declined_said.swap(true, .monotonic)) log.info("[qwen4] PLE prefetch declined: {s}; each gather reads its own rows\n", .{why});
    }

    /// Ends the open prefetch session, if any; each reader finishes its row in flight. The prefill calls it on every exit.
    pub fn prefetchEnd(self: *NgramTable) void {
        if (self.ahead) |a| a.end();
    }

    /// Whether a prefill chunk starts its layers before the PLE layer ahead of the host gather (`Trunk.ple_kick`): while
    /// a prefetch session is open.
    pub fn prefetchKick(self: *const NgramTable) bool {
        const a = self.ahead orelse return false;
        return a.open.load(.acquire);
    }
};

/// The widest trellis ring a table may have: the decode's stack buffers.
const max_trellis_dim = 1024;

/// exllamav3's MUL1 multiplier: a codeword's weight depends on the byte sum of `code * mul1_mult` alone.
const mul1_mult: u32 = @import("../../sdk/quant/exl3/format.zig").sushi.format.MUL1_MULT;

/// MUL1's codeword hash reduced to its byte sum t in [0, 1020].
fn mul1ByteSum(code: u16) u32 {
    const x = @as(u32, code) *% mul1_mult;
    const y = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
    return (y & 0xFFFF) + (y >> 16);
}

/// MUL1's weight per byte sum t, widened to f32: f16(fma(1024 + t, half 0x1EEE, half 0xC931)).
const mul1_by_sum: [1021]f32 = blk: {
    @setEvalBranchQuota(100_000);
    var lut: [1021]f32 = undefined;
    const inverse: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0x1EEE))));
    const bias: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0xC931))));
    for (&lut, 0..) |*v, t| {
        const h: f32 = @floatCast(@as(f16, @bitCast(@as(u16, 0x6400 + t))));
        const q: f16 = @floatCast(@mulAdd(f32, h, inverse, bias));
        v.* = @floatCast(q);
    }
    break :blk lut;
};

const E2M1 = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

const E4M3: [256]f32 = blk: {
    @setEvalBranchQuota(100_000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, b| {
        const e: i32 = @intCast((b >> 3) & 15);
        const m: f32 = @floatFromInt(b & 7);
        const mag = if (e == 0) m / 8.0 * std.math.pow(f32, 2, -6) else (1 + m / 8.0) * std.math.pow(f32, 2, @as(f32, @floatFromInt(e - 7)));
        v.* = if (b & 0x80 != 0) -mag else mag;
    }
    break :blk t;
};

/// One NVFP4 row: e2m1 codes low nibble first, an e4m3 scale per 16, times the table's global scale.
fn dequantNvfp4Row(codes: []const u8, scales: []const u8, global: f32, out: []f32) void {
    for (0..out.len / 16) |g| {
        const sc = E4M3[scales[g]] * global;
        for (codes[g * 8 ..][0..8], 0..) |b, j| {
            out[g * 16 + 2 * j] = E2M1[b & 15] * sc;
            out[g * 16 + 2 * j + 1] = E2M1[b >> 4] * sc;
        }
    }
}

/// Persistent gather workers. Every row's three regions are one SSD read on
/// the cold 32 GB table (~100 us), 48 per token: serial mmap faults were ~5 ms
/// of every decode step, 16 fault threads ~0.7 ms, and more threads got SLOWER
/// (faults on one mapping serialize on the VM map lock), so workers `pread`
/// instead and dequantize their rows in place. Workers wake on a generation
/// bump and count themselves down; the caller spins (the job is ~100 us).
const PrefetchPool = struct {
    const N = 48;
    const MAX_ROWS = 64;
    const ROW_BUF = 512;
    /// The warmer only publishes this flag; measured state belongs to the inference thread.
    warm_finished: std.atomic.Value(bool) = .init(false),
    prefer_pool: bool = false,
    calib: Calibration = .{},
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    table: ?*const NgramTable = null,
    rows: []const i64 = &.{},
    /// A round's job: `pread` fans the rows' part preads into `bufs` (the caller dequantizes); `trellis` hands each
    /// worker a contiguous share of the rows to decode into `out` (any count of rows); `trellis_cold` preads the rows
    /// `cold` marks into `bufs`.
    job: Job = .pread,
    out: []f32 = &.{},
    bufs: [MAX_ROWS][ROW_BUF]u8 = undefined,
    /// `trellis_cold`: the rows (of `rows`, by index) the workers read into `bufs`.
    cold: [MAX_ROWS]bool = undefined,
    /// `trellis`: rows not resident are read with a pread each (`cold_rows` counts them); the prefetch's view of the
    /// gather's rows, set around the round by `gather`.
    cold_rows: std.atomic.Value(u64) = .init(0),
    ahead_view: ?*AheadView = null,
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    /// Fan-out rounds issued; the engagement counter the prefill test reads.
    runs: std.atomic.Value(u64) = .init(0),
    /// The trellis decode's engagement line has been said (inference thread only).
    trellis_said: bool = false,
    threads: [N]std.Thread = undefined,

    fn recordCalibration(self: *PrefetchPool, result: Calibration) void {
        self.calib = result;
        self.prefer_pool = plePrefillPrefetchArm(result.serial_ns, result.pool_ns);
    }

    fn create() !*PrefetchPool {
        const a = std.heap.page_allocator;
        const p = try a.create(PrefetchPool);
        p.* = .{};
        var started: usize = 0;
        errdefer {
            p.shutdown(started);
            a.destroy(p);
        }
        for (0..N) |i| {
            // Past std's 256 KiB thread-local signal stack: glibc carves static TLS out of the thread's stack.
            p.threads[i] = try std.Thread.spawn(.{ .stack_size = 1024 * 1024 }, worker, .{ p, i });
            started += 1;
        }
        return p;
    }

    fn destroy(self: *PrefetchPool) void {
        self.shutdown(N);
        std.heap.page_allocator.destroy(self);
    }

    fn shutdown(self: *PrefetchPool, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    const Job = enum { pread, trellis, trellis_cold };

    /// Fan the per-row part preads over the workers; rows land in `bufs`.
    fn run(self: *PrefetchPool, table: *const NgramTable, rows: []const i64) bool {
        return self.dispatch(table, rows, .pread, &.{});
    }

    /// Decode a trellis table's rows on the workers, row i into `out[i * dim ..]` at head i % n_heads.
    fn runTrellis(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, out: []f32) void {
        if (rows.len >= PREFILL_SAY_MIN_ROWS and !self.trellis_said) {
            self.trellis_said = true;
            log.info("[qwen4] PLE trellis gather engaged: decoded on {d} workers ({d} rows)\n", .{ N, rows.len });
        }
        self.cold_rows.store(0, .monotonic);
        _ = self.dispatch(table, rows, .trellis, out);
        noteColdPread(self.cold_rows.load(.monotonic), rows.len, true);
    }

    /// A decode step's trellis rows (at most MAX_ROWS): those the page cache lacks are read by the workers in parallel
    /// and decoded from `bufs`, the rest from the mapping. False when none is cold or a read came up short.
    fn coldTrellisRows(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, out: []f32) bool {
        const nb: usize = @as(usize, table.wcols) * 2;
        if (rows.len > MAX_ROWS or nb > ROW_BUF or nb == 0) return false;
        var n_cold: usize = 0;
        for (rows, 0..) |r, i| {
            self.cold[i] = !table.rowResident(@intCast(r));
            if (self.cold[i]) n_cold += 1;
        }
        if (n_cold == 0) return false;
        if (!self.dispatch(table, rows, .trellis_cold, &.{})) return false;
        noteColdPread(n_cold, rows.len, false);
        const dim: usize = table.dim;
        const heads: usize = @max(table.n_heads, 1);
        for (rows, 0..) |r, i| {
            const head: u32 = @intCast(i % heads);
            const o = out[i * dim ..][0..dim];
            if (self.cold[i]) table.trellisDecode(self.bufs[i][0..nb], head, o) else table.trellisRowMapped(@intCast(r), head, o);
        }
        return true;
    }

    fn dispatch(self: *PrefetchPool, table: *const NgramTable, rows: []const i64, job: Job, out: []f32) bool {
        _ = self.runs.fetchAdd(1, .monotonic);
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.table = table;
        self.rows = rows;
        self.job = job;
        self.out = out;
        self.failed.store(0, .release);
        self.pending.store(N, .release);
        self.gen += 1;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        while (self.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
        return self.failed.load(.acquire) == 0;
    }

    fn worker(self: *PrefetchPool, idx: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        var seen: u64 = 0;
        while (true) {
            self.mu.lockUncancelable(io);
            while (self.gen == seen and !self.quit) self.cv.wait(io, &self.mu) catch {};
            if (self.quit) {
                self.mu.unlock(io);
                return;
            }
            seen = self.gen;
            const table = self.table.?;
            const rows = self.rows;
            const job = self.job;
            const out = self.out;
            const view = self.ahead_view;
            self.mu.unlock(io);
            switch (job) {
                .pread => {
                    const parts = embedded_ple.partCount(table.fmt);
                    var i = idx;
                    while (i < rows.len * parts) : (i += N) {
                        if (!table.preadSite(@intCast(rows[i / parts]), i % parts, &self.bufs[i / parts])) _ = self.failed.fetchAdd(1, .acq_rel);
                    }
                },
                .trellis_cold => {
                    const nb: usize = @as(usize, table.wcols) * 2;
                    var i = idx;
                    while (i < rows.len) : (i += N) {
                        if (!self.cold[i]) continue;
                        const off = table.w_off + @as(usize, @intCast(rows[i])) * nb;
                        if (crt.pread(table.fd, &self.bufs[i], nb, @intCast(off)) != @as(isize, @intCast(nb))) _ = self.failed.fetchAdd(1, .acq_rel);
                    }
                },
                .trellis => {
                    const dim: usize = table.dim;
                    const heads: usize = @max(table.n_heads, 1);
                    const nb: usize = @as(usize, table.wcols) * 2;
                    const preads = table.fd >= 0 and nb <= ROW_BUF;
                    const share = (rows.len + N - 1) / N;
                    const end = @min((idx + 1) * share, rows.len);
                    var buf: [ROW_BUF]u8 = undefined;
                    var n_cold: u64 = 0;
                    var n_ahead: u64 = 0;
                    var i = @min(idx * share, rows.len);
                    while (i < end) : (i += 1) {
                        const r: u64 = @intCast(rows[i]);
                        const head: u32 = @intCast(i % heads);
                        const o = out[i * dim ..][0..dim];
                        // a row the prefetch staged, or one the page cache lacks read with a pread: the file's bytes
                        if (view) |v| if (v.take(i, rows[i])) |raw| {
                            table.trellisDecode(raw, head, o);
                            n_ahead += 1;
                            continue;
                        };
                        if (preads and !table.rowResident(r) and crt.pread(table.fd, &buf, nb, @intCast(table.w_off + r * nb)) == @as(isize, @intCast(nb))) {
                            table.trellisDecode(buf[0..nb], head, o);
                            n_cold += 1;
                            continue;
                        }
                        table.trellisRowMapped(r, head, o);
                    }
                    if (n_cold != 0) _ = self.cold_rows.fetchAdd(n_cold, .monotonic);
                    if (n_ahead != 0) _ = view.?.hits.fetchAdd(n_ahead, .monotonic);
                },
            }
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }
};

/// A decode step's trellis rows the page cache lacks are read with parallel preads (`PrefetchPool.coldTrellisRows`), and
/// a prefill gather's with one pread each on its worker, not faulted in through the mapping: faults on one mapping
/// serialize on the VM map lock. The bytes, so the output, are the mapping's.
extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;

/// Test seams: a decode step's cold-row preads off (`cold_pread_disabled_for_test`); every row counted as not resident
/// (`cold_test_force`).
pub var cold_pread_disabled_for_test = false;
pub var cold_test_force = false;

/// [decode, prefill]: the cold-row preads' one-shot engagement lines.
var cold_pread_said: [2]std.atomic.Value(bool) = .{ .init(false), .init(false) };
fn noteColdPread(cold: u64, rows: usize, wide: bool) void {
    if (cold == 0 or cold_pread_said[@intFromBool(wide)].swap(true, .monotonic)) return;
    log.info("[qwen4] PLE {s} gather cold preads engaged: {d} of {d} rows not resident, read in parallel\n", .{ if (wide) "prefill" else "decode", cold, rows });
}

// ── The PLE prefetch: a prefill's trellis rows, read ahead of its gathers ──

/// Positions per plan segment, the unit a ring slot holds.
const AHEAD_SEG_TOKENS: usize = 4096;
/// Rows per reader task; every segment is a whole number of them (`aheadGeometry` sizes segments in 64-token steps).
const AHEAD_BATCH: usize = 64;
/// Segments one gather can see; its rows past them are read the plain way.
const AHEAD_VIEW_SEGS: usize = 8;
/// A shorter prompt is one small gather with nothing to hide behind: no session.
const AHEAD_MIN_TOKENS: usize = 1024;
const AHEAD_READERS: usize = 32;
/// The staging ring's byte budget; at least two segments are kept whatever it holds.
const AHEAD_BUDGET: usize = 64 << 20;
const AH_NO_SEG: usize = std.math.maxInt(usize);

/// A planned row's state: FREE -> READING (a reader claimed it) -> STAGED or FAILED (a short read), or FREE -> TAKEN (the
/// gather got there first and reads it itself).
const AH_FREE: u8 = 0;
const AH_READING: u8 = 1;
const AH_STAGED: u8 = 2;
const AH_TAKEN: u8 = 3;
const AH_FAILED: u8 = 4;

/// How long a gather waits for a row a reader is reading before it reads the row itself.
const AHEAD_WAIT_NS: u64 = 2_000_000;
/// Test seam: the wait's bound in place of `AHEAD_WAIT_NS`.
pub var ahead_wait_override: ?u64 = null;
var ahead_said = std.atomic.Value(bool).init(false);
var ahead_failed_said = std.atomic.Value(bool).init(false);
var ahead_declined_said = std.atomic.Value(bool).init(false);

/// Row ids of positions [a, b) of `ids`, which follow the (ngram_size-1)-token history `prev0`: `NgramHash.rowIds` over
/// `ids[a..b]` given the history a chunked gather holds at `a` (`advancePlePrev`). A row depends only on its token and
/// the ngram_size-1 before it, so every chunking hashes the same rows. `out` is `[(b - a) * n_heads]`.
pub fn planRows(hash: *const NgramHash, prev0: []const u32, ids: []const u32, a: usize, b: usize, out: []i64) void {
    const ctx: usize = hash.ngram_size - 1;
    std.debug.assert(prev0.len == ctx and a <= b and b <= ids.len);
    var prev: [MAX_NGRAM_SIZE]u32 = undefined;
    for (0..ctx) |j| {
        const g = a + j; // an index into prev0 ++ ids
        prev[j] = if (g < ctx) prev0[g] else ids[g - ctx];
    }
    hash.rowIds(prev[0..ctx], ids[a..b], out);
}

pub const AheadGeometry = struct { seg_tokens: usize, seg_rows: usize, n_segs: usize, slots: usize, bytes: usize };

/// Segments and ring slots for an `n_tokens` prompt of `nh` rows a token, `nb` bytes a row: `AHEAD_SEG_TOKENS`-token
/// segments (one, rounded up to 64 tokens, for a shorter prompt), as many slots as `budget` holds, at least two and at
/// most one per segment. `bytes` is what the session allocates for them.
pub fn aheadGeometry(n_tokens: usize, nh: usize, nb: usize, budget: usize) AheadGeometry {
    const seg_tokens = @min(AHEAD_SEG_TOKENS, std.mem.alignForward(usize, @max(n_tokens, 1), 64));
    const seg_rows = seg_tokens * nh;
    const n_segs = (n_tokens + seg_tokens - 1) / seg_tokens;
    const slot_bytes = seg_rows * (nb + @sizeOf(i64) + 1);
    const slots = @min(n_segs, @max(2, budget / @max(slot_bytes, 1)));
    return .{ .seg_tokens = seg_tokens, .seg_rows = seg_rows, .n_segs = n_segs, .slots = slots, .bytes = slots * slot_bytes };
}

/// A stride coprime with `n`: the help sweep visits batch (h * stride) % n for h in [0, n), every batch once, spread over
/// the gather's range.
pub fn aheadStride(n: usize) usize {
    if (n <= 2) return 1;
    var s: usize = @max(1, n * 618 / 1000);
    while (std.math.gcd(s, n) != 1) s += 1;
    return s;
}

const AheadSlot = struct {
    /// The segment published here, or AH_NO_SEG (empty, or being planned). Under `Ahead.mu`.
    seg: usize = AH_NO_SEG,
    /// Reader tasks inside this slot. Under `Ahead.mu`: the slot is replanned only at zero.
    inflight: u32 = 0,
    /// `[seg_rows]` planned row ids (token-major, head-minor), written before the slot is published, never after.
    rows: []i64,
    state: []std.atomic.Value(u8),
    /// `[seg_rows][nb]` staged bytes; a row's are written by the reader that claimed it, before it stores AH_STAGED.
    bytes: []u8,
};

const AheadPlan = struct { hash: *const NgramHash, prev: []const u32, ids: []const u32, base: u64, fd: c_int, w_off: usize, nb: usize, budget: usize };

const AheadTask = union(enum) {
    plan: struct { seg: usize, slot: *AheadSlot },
    read: struct { slot: *AheadSlot, lo: usize, hi: usize },
};

/// One prefill's plan and staging ring, from `prefetchBegin` to `prefetchEnd`: page-allocator memory bounded by the
/// budget, plus the prompt's ids.
const AheadSession = struct {
    hash: NgramHash,
    /// The n-gram history before `tokens[0]` (its first ngram_size-1 entries).
    prev0: [MAX_NGRAM_SIZE]u32 = @splat(0),
    tokens: []u32,
    /// The kv position of `tokens[0]`: a gather at `kv_len` starts at plan position `kv_len - base`.
    base: u64,
    nh: usize,
    nb: usize,
    fd: c_int,
    w_off: usize,
    seg_tokens: usize,
    seg_rows: usize,
    n_segs: usize,
    slots: []AheadSlot,
    stop: std.atomic.Value(bool) = .init(false),
    // under Ahead.mu:
    /// The next segment to plan; those below it are published, or were skipped (the gathers were past them).
    planned: usize = 0,
    planning: bool = false,
    /// Positions [0, consumed) were gathered: nothing below is read, kept or looked up again.
    consumed: usize = 0,
    /// Plan rows below this were offered to the background sweep (or are below `consumed`).
    bg_next: usize = 0,
    gathering: bool = false,
    gathers: u32 = 0,
    busy: u32 = 0,
    /// The running gather's help sweep: batches coop_first + (h * coop_stride) % coop_n for h in [coop_next, coop_n),
    /// clipped to plan rows [coop_lo, coop_hi).
    coop_lo: usize = 0,
    coop_hi: usize = 0,
    coop_first: usize = 0,
    coop_n: usize = 0,
    coop_stride: usize = 1,
    coop_next: usize = 0,
    // the readers add the first two; the rest are under Ahead.mu (`noteGather` as each gather begins, `endGather` an
    // engaged one's staging)
    read_ok: std.atomic.Value(u64) = .init(0),
    read_bad: std.atomic.Value(u64) = .init(0),
    n_gathers: u32 = 0,
    /// Of `n_gathers`, those that ran without the ring: their first segment was not in it yet.
    unengaged: u32 = 0,
    gather_rows: u64 = 0,
    hits: u64 = 0,
    waits: u64 = 0,
    gave_up: u64 = 0,
    rows_mem: []i64 = &.{},
    state_mem: []std.atomic.Value(u8) = &.{},
    bytes_mem: []u8 = &.{},

    fn create(plan: AheadPlan) !*AheadSession {
        const pa = std.heap.page_allocator;
        const nh: usize = plan.hash.n_heads;
        const geo = aheadGeometry(plan.ids.len, nh, plan.nb, plan.budget);
        if (geo.slots == 0 or plan.prev.len > MAX_NGRAM_SIZE) return error.EmptyPrefetchPlan;
        const s = try pa.create(AheadSession);
        errdefer pa.destroy(s);
        const tokens = try pa.dupe(u32, plan.ids);
        errdefer pa.free(tokens);
        const cap = geo.slots * geo.seg_rows;
        const rows_mem = try pa.alloc(i64, cap);
        errdefer pa.free(rows_mem);
        const state_mem = try pa.alloc(std.atomic.Value(u8), cap);
        errdefer pa.free(state_mem);
        const bytes_mem = try pa.alloc(u8, cap * plan.nb);
        errdefer pa.free(bytes_mem);
        const slots = try pa.alloc(AheadSlot, geo.slots);
        for (slots, 0..) |*sl, i| sl.* = .{
            .rows = rows_mem[i * geo.seg_rows ..][0..geo.seg_rows],
            .state = state_mem[i * geo.seg_rows ..][0..geo.seg_rows],
            .bytes = bytes_mem[i * geo.seg_rows * plan.nb ..][0 .. geo.seg_rows * plan.nb],
        };
        s.* = .{
            .hash = plan.hash.*,
            .tokens = tokens,
            .base = plan.base,
            .nh = nh,
            .nb = plan.nb,
            .fd = plan.fd,
            .w_off = plan.w_off,
            .seg_tokens = geo.seg_tokens,
            .seg_rows = geo.seg_rows,
            .n_segs = geo.n_segs,
            .slots = slots,
            .rows_mem = rows_mem,
            .state_mem = state_mem,
            .bytes_mem = bytes_mem,
        };
        @memcpy(s.prev0[0..plan.prev.len], plan.prev);
        return s;
    }

    fn deinit(s: *AheadSession) void {
        const pa = std.heap.page_allocator;
        pa.free(s.slots);
        pa.free(s.bytes_mem);
        pa.free(s.state_mem);
        pa.free(s.rows_mem);
        pa.free(s.tokens);
        pa.destroy(s);
    }

    fn totalRows(s: *const AheadSession) usize {
        return s.tokens.len * s.nh;
    }

    /// One past segment `seg`'s last position.
    fn segEnd(s: *const AheadSession, seg: usize) usize {
        return @min((seg + 1) * s.seg_tokens, s.tokens.len);
    }

    fn segRowCount(s: *const AheadSession, seg: usize) usize {
        return (s.segEnd(seg) - seg * s.seg_tokens) * s.nh;
    }

    fn slotOf(s: *AheadSession, seg: usize) *AheadSlot {
        return &s.slots[seg % s.slots.len];
    }

    /// One of this plan's gathers, of `n` plan rows, is beginning (under `Ahead.mu`).
    fn noteGather(s: *AheadSession, n: usize, engaged: bool) void {
        s.n_gathers += 1;
        if (!engaged) s.unengaged += 1;
        s.gather_rows += n;
    }

    /// The next task (under `Ahead.mu`), or null: nothing to do now.
    fn pick(s: *AheadSession) ?AheadTask {
        if (s.stop.load(.acquire)) return null;
        // while a gather runs, its own rows only: no IO or CPU spent past it while it is the critical path
        if (s.gathering) return s.pickHelp();
        if (s.pickPlan()) |t| return t;
        return s.pickBackground();
    }

    fn pickPlan(s: *AheadSession) ?AheadTask {
        if (s.planning) return null;
        while (s.planned < s.n_segs and s.segEnd(s.planned) <= s.consumed) s.planned += 1;
        if (s.planned >= s.n_segs) return null;
        const seg = s.planned;
        const slot = s.slotOf(seg);
        if (slot.seg != AH_NO_SEG) {
            // a full ring: its oldest slot is reused once every gather is past that segment and no reader is in it
            if (slot.inflight != 0 or s.segEnd(slot.seg) > s.consumed) return null;
            slot.seg = AH_NO_SEG;
        }
        s.planning = true;
        return .{ .plan = .{ .seg = seg, .slot = slot } };
    }

    fn pickBackground(s: *AheadSession) ?AheadTask {
        s.bg_next = @max(s.bg_next, s.consumed * s.nh);
        if (s.bg_next >= s.totalRows()) return null;
        const seg = s.bg_next / s.seg_rows;
        const slot = s.slotOf(seg);
        if (slot.seg != seg) return null; // not planned yet: the planner's publish wakes the readers
        const lo = s.bg_next - seg * s.seg_rows;
        const hi = @min(lo + AHEAD_BATCH, s.segRowCount(seg));
        s.bg_next = seg * s.seg_rows + hi;
        slot.inflight += 1;
        return .{ .read = .{ .slot = slot, .lo = lo, .hi = hi } };
    }

    fn pickHelp(s: *AheadSession) ?AheadTask {
        while (s.coop_next < s.coop_n) {
            const h = s.coop_next;
            s.coop_next += 1;
            const b = s.coop_first + (h * s.coop_stride) % s.coop_n;
            const lo_q = @max(b * AHEAD_BATCH, s.coop_lo);
            const hi_q = @min((b + 1) * AHEAD_BATCH, s.coop_hi);
            if (lo_q >= hi_q) continue;
            const seg = lo_q / s.seg_rows; // a batch never straddles two segments (seg_rows is a multiple of it)
            const slot = s.slotOf(seg);
            if (slot.seg != seg) continue; // not planned: the gather reads those rows itself
            slot.inflight += 1;
            return .{ .read = .{ .slot = slot, .lo = lo_q - seg * s.seg_rows, .hi = hi_q - seg * s.seg_rows } };
        }
        return null;
    }

    /// Outside the lock: a plan task owns its unpublished slot, a read task claims each row before it touches it.
    fn run(s: *AheadSession, t: AheadTask) void {
        switch (t) {
            .plan => |p| s.runPlan(p.seg, p.slot),
            .read => |r| s.runRead(r.slot, r.lo, r.hi),
        }
    }

    fn runPlan(s: *AheadSession, seg: usize, slot: *AheadSlot) void {
        const a = seg * s.seg_tokens;
        const b = s.segEnd(seg);
        @memset(slot.state[0 .. (b - a) * s.nh], std.atomic.Value(u8).init(AH_FREE));
        const ctx: usize = s.hash.ngram_size - 1;
        var p = a;
        while (p < b) {
            if (s.stop.load(.monotonic)) return;
            const q = @min(p + 512, b);
            planRows(&s.hash, s.prev0[0..ctx], s.tokens, p, q, slot.rows[(p - a) * s.nh .. (q - a) * s.nh]);
            p = q;
        }
    }

    fn runRead(s: *AheadSession, slot: *AheadSlot, lo: usize, hi: usize) void {
        var ok: u64 = 0;
        var bad: u64 = 0;
        var k = lo;
        while (k < hi) : (k += 1) {
            if (s.stop.load(.monotonic)) break;
            const st = &slot.state[k];
            if (st.cmpxchgStrong(AH_FREE, AH_READING, .monotonic, .monotonic) != null) continue;
            const r: usize = @intCast(slot.rows[k]);
            const dst = slot.bytes[k * s.nb ..][0..s.nb];
            if (crt.pread(s.fd, dst.ptr, s.nb, @intCast(s.w_off + r * s.nb)) == @as(isize, @intCast(s.nb))) {
                st.store(AH_STAGED, .release);
                ok += 1;
            } else {
                st.store(AH_FAILED, .release);
                bad += 1;
            }
        }
        if (ok != 0) _ = s.read_ok.fetchAdd(ok, .monotonic);
        if (bad != 0) _ = s.read_bad.fetchAdd(bad, .monotonic);
    }

    /// A task is done (under `Ahead.mu`). True when a waiter may proceed: a new segment, a slot the planner may reuse, or
    /// no reader left for `end`.
    fn finish(s: *AheadSession, t: AheadTask) bool {
        s.busy -= 1;
        switch (t) {
            .plan => |p| {
                s.planning = false;
                if (!s.stop.load(.acquire)) {
                    p.slot.seg = p.seg;
                    s.planned = @max(s.planned, p.seg + 1);
                }
                return true;
            },
            .read => |r| {
                r.slot.inflight -= 1;
                return s.busy == 0 or (r.slot.inflight == 0 and s.planned < s.n_segs);
            },
        }
    }

    /// The request's summary (`Ahead.end`), at debug level.
    fn report(s: *const AheadSession) void {
        const mb = @as(f64, @floatFromInt(s.bytes_mem.len + s.rows_mem.len * @sizeOf(i64) + s.state_mem.len)) / 1048576.0;
        log.debug("[qwen4] PLE prefetch: {d} tokens over {d} slots ({d:.0} MB): read {d} rows ({d} short); {d} gathers ({d} not engaged) took {d} of their {d} rows staged, waited on {d}, gave up on {d}\n", .{
            s.tokens.len, s.slots.len, mb, s.read_ok.load(.monotonic), s.read_bad.load(.monotonic), s.n_gathers, s.unengaged, s.hits, s.gather_rows, s.waits, s.gave_up,
        });
    }
};

/// One gather's window on the open session, resolved under `Ahead.mu` by `beginGather`: the segments it sees cannot be
/// replanned before `endGather`. The pool's workers call `take` concurrently.
const AheadView = struct {
    sess: *AheadSession,
    /// Plan rows [q0, q_end) are the gather's rows [0, q_end - q0); positions up to `end_pos`.
    q0: usize,
    q_end: usize,
    end_pos: usize,
    first_seg: usize,
    n_slots: usize = 0,
    /// Segments first_seg.. as published when the gather began (null: not planned yet; those rows are read the plain way).
    slots: [AHEAD_VIEW_SEGS]?*AheadSlot = @splat(null),
    hits: std.atomic.Value(u64) = .init(0),
    waits: std.atomic.Value(u64) = .init(0),
    gave_up: std.atomic.Value(u64) = .init(0),

    /// The staged bytes of the gather's row `i` (row id `r`), or null: the gather reads that row the plain way. A row no
    /// reader has claimed is claimed here, so no reader reads it after; one a reader is reading is waited for, at most
    /// `AHEAD_WAIT_NS`. A row whose planned id is not `r` is left alone.
    fn take(v: *AheadView, i: usize, r: i64) ?[]const u8 {
        const q = v.q0 + i;
        if (q >= v.q_end) return null;
        const s = v.sess;
        const seg = q / s.seg_rows;
        const si = seg - v.first_seg;
        if (si >= v.n_slots) return null;
        const slot = v.slots[si] orelse return null;
        const k = q - seg * s.seg_rows;
        if (slot.rows[k] != r) return null;
        const st = &slot.state[k];
        var cur = st.load(.acquire);
        if (cur == AH_FREE) cur = st.cmpxchgStrong(AH_FREE, AH_TAKEN, .acquire, .acquire) orelse return null;
        if (cur == AH_READING) cur = v.waitRead(st);
        if (cur != AH_STAGED) return null;
        return slot.bytes[k * s.nb ..][0..s.nb];
    }

    fn waitRead(v: *AheadView, st: *std.atomic.Value(u8)) u8 {
        _ = v.waits.fetchAdd(1, .monotonic);
        const sw = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        var spins: u32 = 0;
        while (true) {
            const cur = st.load(.acquire);
            if (cur != AH_READING) return cur;
            spins +%= 1;
            if (spins % 64 == 0 and sw.read() >= (ahead_wait_override orelse AHEAD_WAIT_NS)) {
                _ = v.gave_up.fetchAdd(1, .monotonic);
                return AH_READING;
            }
            std.atomic.spinLoopHint();
        }
    }
};

/// The prefetch's readers and the open session (at most one: the running prefill's), for a trellis table: the prompt's
/// rows are planned (`planRows`) and `pread` in position order into a bounded ring of segments while earlier chunks run;
/// a gather takes a staged row only where its planned id is the one it asks for (`AheadView.take`), so the output is the
/// same bytes. The inference thread opens and ends sessions and runs the gathers; the readers plan and read, and while a
/// gather runs they only help it. All but a row's state and bytes is under `mu`.
const Ahead = struct {
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    quit: bool = false,
    session: ?*AheadSession = null,
    /// A session is open: the inference thread reads it without the lock (`NgramTable.prefetchKick`).
    open: std.atomic.Value(bool) = .init(false),
    n_threads: usize = 0,
    threads: [AHEAD_READERS]std.Thread = undefined,

    fn create() !*Ahead {
        const pa = std.heap.page_allocator;
        const self = try pa.create(Ahead);
        self.* = .{};
        var started: usize = 0;
        errdefer {
            self.shutdown(started);
            pa.destroy(self);
        }
        for (0..AHEAD_READERS) |i| {
            // the pool's stack size: glibc carves static TLS out of a thread's stack
            self.threads[i] = try std.Thread.spawn(.{ .stack_size = 1024 * 1024 }, workerMain, .{self});
            started += 1;
        }
        self.n_threads = started;
        return self;
    }

    fn shutdown(self: *Ahead, started: usize) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        self.quit = true;
        self.cv.broadcast(io);
        self.mu.unlock(io);
        for (self.threads[0..started]) |t| t.join();
    }

    fn destroy(self: *Ahead) void {
        self.end();
        self.shutdown(self.n_threads);
        std.heap.page_allocator.destroy(self);
    }

    fn workerMain(self: *Ahead) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        while (!self.quit) {
            const s = self.session orelse {
                self.cv.waitUncancelable(io, &self.mu);
                continue;
            };
            const task = s.pick() orelse {
                self.cv.waitUncancelable(io, &self.mu);
                continue;
            };
            s.busy += 1;
            self.mu.unlock(io);
            s.run(task);
            self.mu.lockUncancelable(io);
            if (s.finish(task)) self.cv.broadcast(io);
        }
        self.mu.unlock(io);
    }

    /// Opens a session for one prefill and wakes the readers. Fails (nothing changes) without memory or with one open.
    fn begin(self: *Ahead, plan: AheadPlan) !void {
        const s = try AheadSession.create(plan);
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        const busy = self.session != null;
        if (!busy) {
            self.session = s;
            self.open.store(true, .release);
            self.cv.broadcast(io);
        }
        self.mu.unlock(io);
        if (busy) {
            s.deinit();
            return error.PrefetchSessionOpen;
        }
    }

    /// Stops and frees the open session, if any: each reader finishes the row it is reading, a running gather completes.
    fn end(self: *Ahead) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        const s = self.session orelse {
            self.mu.unlock(io);
            return;
        };
        s.stop.store(true, .release);
        self.cv.broadcast(io);
        while (s.busy != 0 or s.gathers != 0) self.cv.waitUncancelable(io, &self.mu);
        self.session = null;
        self.open.store(false, .release);
        self.mu.unlock(io);
        s.report();
        s.deinit();
    }

    /// A gather of `rows` (token-major, `nh` a token) at `kv_len` is starting: its view of the session, or null when it does
    /// not engage. One that is not this plan's (another position, a co-tenant's rows, a position already gathered) changes
    /// nothing; one that is but finds its first segment not in the ring yet still moves the cursors past its positions.
    fn beginGather(self: *Ahead, kv_len: u64, rows: []const i64) ?AheadView {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        const s = self.session orelse return null;
        if (s.stop.load(.acquire) or s.gathering) return null;
        if (rows.len < s.nh or rows.len % s.nh != 0 or kv_len < s.base) return null;
        const pos: usize = @intCast(kv_len - s.base);
        // forward only: the ring may already hold later segments where earlier positions were
        if (pos < s.consumed or pos >= s.tokens.len) return null;
        const end_pos = @min(pos + rows.len / s.nh, s.tokens.len);
        var v: AheadView = .{ .sess = s, .q0 = pos * s.nh, .q_end = end_pos * s.nh, .end_pos = end_pos, .first_seg = pos / s.seg_tokens };
        var seg = v.first_seg;
        while (seg * s.seg_tokens < end_pos and v.n_slots < AHEAD_VIEW_SEGS) : (seg += 1) {
            const slot = s.slotOf(seg);
            v.slots[v.n_slots] = if (slot.seg == seg) slot else null;
            v.n_slots += 1;
        }
        // this plan's own gather: its first token's planned rows are its rows (a co-tenant's at this position are not)
        const slot0 = v.slots[0] orelse {
            // Not in the ring yet: the gather reads its rows itself. When its first token's planned rows are its rows,
            // the cursors still move past it, or a chunk as wide as the ring would keep the planner blocked.
            const hist: usize = s.hash.ngram_size - 1;
            var want: [MAX_HEADS]i64 = undefined;
            planRows(&s.hash, s.prev0[0..hist], s.tokens, pos, pos + 1, want[0..s.nh]);
            if (!std.mem.eql(i64, want[0..s.nh], rows[0..s.nh])) return null;
            s.noteGather(v.q_end - v.q0, false);
            s.consumed = @max(s.consumed, end_pos);
            s.bg_next = @max(s.bg_next, v.q_end);
            self.cv.broadcast(io);
            return null;
        };
        const k0 = v.q0 - v.first_seg * s.seg_rows;
        if (!std.mem.eql(i64, slot0.rows[k0..][0..s.nh], rows[0..s.nh])) return null;
        s.noteGather(v.q_end - v.q0, true);
        s.gathering = true;
        s.gathers += 1;
        s.coop_lo = @max(v.q0, s.bg_next);
        s.coop_hi = v.q_end;
        s.coop_next = 0;
        s.coop_n = 0;
        if (s.coop_lo < s.coop_hi) {
            s.coop_first = s.coop_lo / AHEAD_BATCH;
            s.coop_n = (s.coop_hi - 1) / AHEAD_BATCH - s.coop_first + 1;
            s.coop_stride = aheadStride(s.coop_n);
        }
        self.cv.broadcast(io);
        return v;
    }

    /// The gather `v` is done (its pool round returned): the gathers are past its positions, so the readers move on.
    fn endGather(self: *Ahead, v: *AheadView) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        const s = v.sess;
        s.gathering = false;
        s.gathers -= 1;
        s.coop_n = 0;
        s.consumed = @max(s.consumed, v.end_pos);
        s.bg_next = @max(s.bg_next, v.q_end);
        s.hits += v.hits.load(.monotonic);
        s.waits += v.waits.load(.monotonic);
        s.gave_up += v.gave_up.load(.monotonic);
        self.cv.broadcast(io);
        self.mu.unlock(io);
    }
};

/// One-shot engagement lines per arm; both arms say something.
/// Narrowest gather that is a genuine prefill chunk rather than a warmup forward.
pub const PREFILL_SAY_MIN_ROWS: usize = 1024;

/// [arm][bucket]: arm 0/1 = serial/pooled, bucket 0/1 = warmup/prefill width.
pub var ple_prefill_arm_said: [2][2]std.atomic.Value(bool) =
    .{ .{ .init(false), .init(false) }, .{ .init(false), .init(false) } };

fn notePrefillGatherArm(pooled: bool, rows: usize, prefer_pool: bool) void {
    const arm: usize = if (pooled) 1 else 0;
    const bucket: usize = if (rows >= PREFILL_SAY_MIN_ROWS) 1 else 0;
    if (ple_prefill_arm_said[arm][bucket].swap(true, .monotonic)) return;
    const width: []const u8 = if (bucket == 1) "prefill width" else "warmup width";
    const why: []const u8 = if (plePrefillPrefetchMode() != .kv_gated) "test override" else if (prefer_pool) "pool measured faster" else "KV gate";
    if (pooled) {
        const batches = (rows + PrefetchPool.MAX_ROWS - 1) / PrefetchPool.MAX_ROWS;
        log.info("[qwen4] PLE prefill gather: POOLED ({s}: {d} rows, {d} batches of {d}; {s})\n", .{ width, rows, batches, PrefetchPool.MAX_ROWS, why });
    } else {
        log.info("[qwen4] PLE prefill gather: SERIAL mmap walk ({s}: {d} rows; {s}, threshold {d})\n", .{ width, rows, why, plePrefillPrefetchMinKv() });
    }
}

/// Test seams for the prefill gate below.
pub var ple_prefill_prefetch_override: ?bool = null;
pub var ple_prefill_min_kv_override: ?u64 = null;

/// Long KV can evict a table that was resident at load. This gate supplements calibration.
pub const PREFILL_PREFETCH_MIN_KV: u64 = 262144;

pub const PrefillPrefetchMode = enum { off, kv_gated, on };

const PLE_CALIBRATION_ROWS = 128;
const PLE_CALIBRATION_TRIALS = 5;
const CalibrationPhase = enum { load, warmed };
const Calibration = struct { serial_ns: u64 = 0, pool_ns: u64 = 0, phase: CalibrationPhase = .load };

fn medianNs(samples: []u64) u64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return samples[samples.len / 2];
}

/// Fresh rows per phase and trial, so no trial reads pages an earlier one faulted in.
fn calibrationRows(rows: u64, serial_rows: []i64, pool_rows: []i64, phase: CalibrationPhase, trial: usize) void {
    const half = rows / 2;
    var seed: u64 = (if (phase == .load) SPLITMIX_GAMMA else SPLITMIX_M1) +% @as(u64, trial) *% SPLITMIX_M2;
    for (serial_rows) |*r| {
        r.* = @intCast(splitmix64(seed) % half);
        seed +%= SPLITMIX_GAMMA;
    }
    for (pool_rows) |*r| {
        r.* = @intCast(half + splitmix64(seed) % (rows - half));
        seed +%= SPLITMIX_GAMMA;
    }
}

/// Require a win greater than 20%; ties, noise and unreadable timers retain serial.
pub fn plePrefillPrefetchArm(serial_ns: u64, pool_ns: u64) bool {
    if (serial_ns == 0 or pool_ns == 0) return false;
    return @as(u128, pool_ns) * 100 < @as(u128, serial_ns) * 80;
}

pub fn plePrefillPrefetchWanted(mode: PrefillPrefetchMode, kv_len: u64, min_kv: u64, prefer_pool: bool) bool {
    return switch (mode) {
        .off => false,
        .on => true,
        .kv_gated => prefer_pool or kv_len >= min_kv,
    };
}

fn plePrefillPrefetchMinKv() u64 {
    return ple_prefill_min_kv_override orelse PREFILL_PREFETCH_MIN_KV;
}

fn plePrefillPrefetchMode() PrefillPrefetchMode {
    const v = ple_prefill_prefetch_override orelse return .kv_gated;
    return if (v) .on else .off;
}

fn plePrefillPrefetchEnabled(kv_len: u64, prefer_pool: bool) bool {
    return plePrefillPrefetchWanted(plePrefillPrefetchMode(), kv_len, plePrefillPrefetchMinKv(), prefer_pool);
}

pub fn bf16ToF32(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// Round-to-nearest-even f32 -> bf16 bits, the PLE rows' upload format.
pub fn bf16Rne(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

// ── tests ──

const testing = std.testing;

/// Module-owned state for one loaded qwen4_exp model: the n-gram hash and
/// the mmapped table. Non-null on `Trunk.qwen4` ⇒ the arch is served
/// serially with speculation off (`ownsModuleDecodeState`), which is what
/// the per-request PLE/indexer state in `SSMCacheEntry.aux_state` needs
/// until the snapshot machinery carries it.
pub const Qwen4State = struct {
    hash: NgramHash,
    table: NgramTable,
    /// The table as one no-copy GPU buffer; null = the host gather serves every forward.
    gpu: ?ple_gpu.Table = null,

    pub fn deinit(self: *Qwen4State) void {
        self.table.close();
        if (self.gpu) |g| g.release();
        self.gpu = null;
    }
};

/// rows 4, dim 64, 4-bit, group 32 => wcols 8, scols 2; w 128 B, s/b 16 B each.
const NGRAM_GOOD_HEADER =
    "{\"__metadata__\":{\"format\":\"mlx-serve-ngram\",\"bits\":\"4\",\"group_size\":\"32\"}," ++
    "\"weight\":{\"dtype\":\"U32\",\"shape\":[4,8],\"data_offsets\":[0,128]}," ++
    "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[128,144]}," ++
    "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[144,160]}}";
const NGRAM_GOOD_BYTES = 160;

const AheadTestStats = struct { open: bool = false, hits: u64 = 0, gathers: u32 = 0, unengaged: u32 = 0, rows: u64 = 0, read_ok: u64 = 0, read_bad: u64 = 0, consumed: usize = 0 };

/// Plays the readers on a session without threads: a read task stages its rows without IO.
const AheadTestPlay = struct {
    fn done(sess: *AheadSession, t: AheadTask) void {
        switch (t) {
            .plan => |p| sess.runPlan(p.seg, p.slot),
            .read => |r| for (r.slot.state[r.lo..r.hi]) |*st| {
                _ = st.cmpxchgStrong(AH_FREE, AH_STAGED, .monotonic, .monotonic);
            },
        }
        _ = sess.finish(t);
    }
};
