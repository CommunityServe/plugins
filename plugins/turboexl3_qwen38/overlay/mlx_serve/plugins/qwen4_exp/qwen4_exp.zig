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
const log = @import("../../sdk/util/log.zig");
const ple_gpu = @import("ple_gpu.zig");
const embedded_ple = @import("ple_table.zig");

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

};

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

    pub fn parse(map: []align(std.heap.page_size_min) const u8, header: []const u8, data_off: usize) !NgramTable {
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

};

/// The widest trellis ring a table may have: the decode's stack buffers.
const max_trellis_dim = 1024;

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

};



// ── The PLE prefetch: a prefill's trellis rows, read ahead of its gathers ──

/// Segments one gather can see; its rows past them are read the plain way.
const AHEAD_VIEW_SEGS: usize = 8;
const AHEAD_READERS: usize = 32;
const AH_NO_SEG: usize = std.math.maxInt(usize);



pub const AheadGeometry = struct { seg_tokens: usize, seg_rows: usize, n_segs: usize, slots: usize, bytes: usize };

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

    fn deinit(s: *AheadSession) void {
        const pa = std.heap.page_allocator;
        pa.free(s.slots);
        pa.free(s.bytes_mem);
        pa.free(s.state_mem);
        pa.free(s.rows_mem);
        pa.free(s.tokens);
        pa.destroy(s);
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

};


const CalibrationPhase = enum { load, warmed };
const Calibration = struct { serial_ns: u64 = 0, pool_ns: u64 = 0, phase: CalibrationPhase = .load };

// ── tests ──

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

