//! Reads past the page cache (`openNoCache`, `noCache`, `readAligned`), the on-disk row gather and its
//! prefetch pool, file residency, and bf16 rounding: the plugin's own I/O helpers (from #749's host io_util).

const std = @import("std");
const builtin = @import("builtin");

pub const NoCacheOptions = struct {
    /// Keep the kernel's read-ahead (a file streamed front to back).
    read_ahead: bool = false,
    /// Open a symlink's target; false refuses a symlink (ELOOP).
    follow_symlinks: bool = true,
};

/// Reads of `fd` bypass the page cache (F_NOCACHE); read-ahead off unless asked.
/// Darwin only: elsewhere it refuses, so no caller silently reads through the cache.
pub fn noCache(fd: std.c.fd_t, opts: NoCacheOptions) error{ NoCacheFcntl, NoCacheUnsupported }!void {
    if (comptime !builtin.os.tag.isDarwin()) return error.NoCacheUnsupported;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.NoCacheFcntl;
    if (!opts.read_ahead and std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) return error.NoCacheFcntl;
}

// ── Reads past the page cache: aligned reads, a table's row gather, residency ──

/// readAligned's staging buffer: page-aligned (the page allocator), a multiple of every page size.
const stage_bytes: usize = 8 << 20;

// ── Row gather: a table's rows past the page cache ──

/// A table's rows past the page cache (the input embedding's host rows): `row_bytes` at `base + id * row_bytes` of an
/// F_NOCACHE descriptor, gathered into the caller's id order. Every read is page-aligned in offset, length and
/// destination, and checked, since macOS keeps only aligned reads out of the page cache (nocache_reader's `Desc.readAt`). Each distinct
/// row is read once; rows whose aligned pages touch are read as one run; `helpers` threads and the caller take the runs
/// in parallel. `init` allocates everything once (the page-aligned stages, the sort and run scratch, the threads): a
/// gather allocates nothing. One gather at a time (the model's thread); the descriptor stays its owner's.
pub const RowGather = struct {
    /// One reader's stage: the widest run it reads (whole pages).
    pub const stage_len: usize = 128 << 10;
    pub const Item = struct { id: u32, at: u32 };
    /// `len` aligned bytes at `off` (aligned) hold `items[first..end]`'s rows; the last of them ends `need` bytes in.
    pub const Run = struct { off: u64, len: u64, need: u64, first: u32, end: u32 };

    fd: std.c.fd_t,
    base: u64,
    row_bytes: usize,
    rows: u64,
    page: usize,
    /// Ids per piece (a longer call runs in pieces of this many).
    max_ids: usize,
    /// Below this many ids the caller reads alone (no helper woken).
    parallel_min: usize,
    /// `helpers + 1` stages; `[0]` is the caller's.
    stages: [][]u8,
    items: []Item,
    runs: []Run,
    threads: []std.Thread,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    gen: u64 = 0,
    quit: bool = false,
    /// The piece in flight (set before `gen` moves, read after it).
    out: []u8 = &.{},
    n_runs: usize = 0,
    next: std.atomic.Value(usize) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(u32) = .init(0),
    unaligned: std.atomic.Value(u32) = .init(0),

    pub fn init(fd: std.c.fd_t, base: u64, row_bytes: usize, rows: u64, helpers: usize, max_ids: usize, parallel_min: usize) !*RowGather {
        const page = std.heap.pageSize();
        if (row_bytes == 0 or max_ids == 0 or row_bytes + 2 * page > stage_len or stage_len % page != 0) return error.GatherGeometry;
        const a = std.heap.c_allocator;
        const self = try a.create(RowGather);
        errdefer a.destroy(self);
        self.* = .{ .fd = fd, .base = base, .row_bytes = row_bytes, .rows = rows, .page = page, .max_ids = max_ids, .parallel_min = parallel_min, .stages = &.{}, .items = &.{}, .runs = &.{}, .threads = &.{} };
        self.stages = try a.alloc([]u8, helpers + 1);
        errdefer a.free(self.stages);
        var made: usize = 0;
        errdefer for (self.stages[0..made]) |st| std.heap.page_allocator.free(st);
        for (self.stages) |*st| {
            st.* = try std.heap.page_allocator.alloc(u8, stage_len);
            made += 1;
            if (@intFromPtr(st.*.ptr) % page != 0) return error.GatherGeometry;
        }
        self.items = try a.alloc(Item, max_ids);
        errdefer a.free(self.items);
        self.runs = try a.alloc(Run, max_ids);
        errdefer a.free(self.runs);
        self.threads = try a.alloc(std.Thread, helpers);
        errdefer a.free(self.threads);
        var started: usize = 0;
        errdefer self.stop(started);
        for (self.threads, 0..) |*t, i| {
            t.* = try std.Thread.spawn(.{ .stack_size = 64 * 1024 }, helper, .{ self, i });
            started += 1;
        }
        return self;
    }

    pub fn deinit(self: *RowGather) void {
        self.stop(self.threads.len);
        const a = std.heap.c_allocator;
        for (self.stages) |st| std.heap.page_allocator.free(st);
        a.free(self.stages);
        a.free(self.items);
        a.free(self.runs);
        a.free(self.threads);
        a.destroy(self);
    }

    /// The sorted items' runs: each distinct row once, rows whose aligned pages touch in one run up to a stage.
    fn plan(self: *RowGather, n: usize) usize {
        const p: u64 = self.page;
        const rb: u64 = self.row_bytes;
        const items = self.items[0..n];
        var nr: usize = 0;
        var i: usize = 0;
        while (i < n) {
            const o0 = self.base + @as(u64, items[i].id) * rb;
            const start = o0 - o0 % p;
            var end_row = o0 + rb;
            var j = i + 1;
            while (j < n) : (j += 1) {
                if (items[j].id == items[j - 1].id) continue;
                const o = self.base + @as(u64, items[j].id) * rb;
                if (o - o % p > self.alignUp(end_row)) break;
                if (self.alignUp(o + rb) - start > stage_len) break;
                end_row = o + rb;
            }
            self.runs[nr] = .{ .off = start, .len = self.alignUp(end_row) - start, .need = end_row - start, .first = @intCast(i), .end = @intCast(j) };
            nr += 1;
            i = j;
        }
        return nr;
    }

    fn helper(self: *RowGather, idx: usize) void {
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
            self.mu.unlock(io);
            self.work(self.stages[idx + 1]);
            _ = self.pending.fetchSub(1, .acq_rel);
        }
    }

    /// A run's read is page-aligned in offset, length and destination, and fits its stage.
    pub fn aligned(self: *const RowGather, run: Run, stage: []const u8) bool {
        return run.off % self.page == 0 and run.len % self.page == 0 and @intFromPtr(stage.ptr) % self.page == 0 and run.len <= stage.len and run.need <= run.len;
    }

};

// ── Residency probes (proof tests) ──

// ── Row prefetch, safetensors header entries, bf16 rounding (shared with plugins through sdk.io_util) ──

// ── Tests (temp files; the reads checked against a plain pread of the same bytes) ──

const testing = std.testing;

/// A temp file of `len` pattern bytes and its absolute path.
const TmpFile = struct {
    dir: std.testing.TmpDir,
    path: [:0]u8,
    image: []u8,

    fn init(len: usize) !TmpFile {
        var dir = std.testing.tmpDir(.{});
        errdefer dir.cleanup();
        const image = try testing.allocator.alloc(u8, len);
        errdefer testing.allocator.free(image);
        for (image, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 11);
        try dir.dir.writeFile(testing.io, .{ .sub_path = "f.bin", .data = image });
        var root: [512]u8 = undefined;
        const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/f.bin", .{root[0..try dir.dir.realPath(testing.io, &root)]}, 0);
        return .{ .dir = dir, .path = path, .image = image };
    }

    fn deinit(t: *TmpFile) void {
        testing.allocator.free(t.path);
        testing.allocator.free(t.image);
        t.dir.cleanup();
    }
};
