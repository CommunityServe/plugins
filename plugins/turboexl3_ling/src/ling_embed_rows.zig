//! EMBED_ROWS' table and its row reads, MLX-free (src/ling_embed.zig serves them). A pack keeps the embedding as one
//! tensor in one shard (Ling-3.1-flash: `model.word_embeddings.weight`, BF16 [157184, 4096] in
//! model-00001-of-00024-nonexpert.safetensors). The pack's index (model.safetensors.index.json) names the shard, and
//! the shard's safetensors header gives the tensor's dtype, shape and bytes. A forward's rows are read from there as
//! stored, so they are the words the resident table's `take` gives. After mlx-serve 66aad949's
//! src/plugins/deepseek_v41/prompt/embed_pread.zig (DeepSeek V4.1's embedding read by pread), over the trunk's own
//! table name: each distinct id read once, the reads in parallel on GCD's pool, the repeats copied; the OS page cache
//! keeps the rows a decode reads again.
const std = @import("std");
const builtin = @import("builtin");

/// The most bytes the index and a shard's header may take (the copied streamer's `scanShard` bound).
pub const json_max: u64 = 256 << 20;

/// The table in its shard: row 0's first byte, its rows, its columns (bf16 words) and the bytes a row.
pub const Table = struct {
    offset: u64,
    rows: u32,
    cols: u32,
    row_bytes: usize,

    /// The table's bytes: what a resident copy of it holds.
    pub fn bytes(t: Table) u64 {
        return @as(u64, t.rows) * t.row_bytes;
    }

    /// Row `id`'s first byte in the shard (`id` below `rows`).
    pub fn rowOffset(t: Table, id: u32) u64 {
        return t.offset + @as(u64, id) * t.row_bytes;
    }
};

/// The shard the index text `index` names for `key`, parsed in `a` (an arena: the name may point into it). Refused by
/// name: an index that does not parse, has no `weight_map` entry for `key`, or names a file outside the pack's folder.
pub fn shardOf(a: std.mem.Allocator, index: []const u8, key: []const u8) ![]const u8 {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, index, .{}) catch return error.LingEmbedRowsIndex;
    if (doc != .object) return error.LingEmbedRowsIndex;
    const wm = doc.object.get("weight_map") orelse return error.LingEmbedRowsIndex;
    if (wm != .object) return error.LingEmbedRowsIndex;
    const v = wm.object.get(key) orelse return error.LingEmbedRowsIndex;
    if (v != .string or v.string.len == 0) return error.LingEmbedRowsIndex;
    if (std.mem.indexOfScalar(u8, v.string, '/') != null or std.mem.indexOf(u8, v.string, "..") != null) return error.LingEmbedRowsIndex;
    return v.string;
}

/// `key` in a shard of `file_size` bytes whose header (the JSON after the 8-byte length) is `header`, parsed in `a` (an
/// arena): BF16 [rows, cols], exactly rows * cols * 2 bytes, all inside the file. Refused by name otherwise:
/// LingEmbedRowsMissing (no such tensor), LingEmbedRowsNotBf16, LingEmbedRowsShape (another shape or byte count),
/// LingEmbedRowsTruncated (bytes past the end of the file), LingEmbedRowsHeader (a header that does not parse).
pub fn tableOf(a: std.mem.Allocator, header: []const u8, file_size: u64, key: []const u8, rows: u32, cols: u32) !Table {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return error.LingEmbedRowsHeader;
    if (doc != .object) return error.LingEmbedRowsHeader;
    const t = doc.object.get(key) orelse return error.LingEmbedRowsMissing;
    if (t != .object) return error.LingEmbedRowsHeader;
    const dt = t.object.get("dtype") orelse return error.LingEmbedRowsHeader;
    if (dt != .string) return error.LingEmbedRowsHeader;
    if (!std.mem.eql(u8, dt.string, "BF16")) return error.LingEmbedRowsNotBf16;
    const shape = t.object.get("shape") orelse return error.LingEmbedRowsHeader;
    if (shape != .array) return error.LingEmbedRowsHeader;
    if (shape.array.items.len != 2) return error.LingEmbedRowsShape;
    const n_rows = intOf(shape.array.items[0]) orelse return error.LingEmbedRowsHeader;
    const n_cols = intOf(shape.array.items[1]) orelse return error.LingEmbedRowsHeader;
    if (n_rows != rows or n_cols != cols) return error.LingEmbedRowsShape;
    const offs = t.object.get("data_offsets") orelse return error.LingEmbedRowsHeader;
    if (offs != .array or offs.array.items.len != 2) return error.LingEmbedRowsHeader;
    const lo = intOf(offs.array.items[0]) orelse return error.LingEmbedRowsHeader;
    const hi = intOf(offs.array.items[1]) orelse return error.LingEmbedRowsHeader;
    const row_bytes: usize = @as(usize, cols) * 2;
    if (hi < lo or hi - lo != @as(u64, rows) * row_bytes) return error.LingEmbedRowsShape;
    const start = std.math.add(u64, 8 + @as(u64, header.len), lo) catch return error.LingEmbedRowsTruncated;
    const end = std.math.add(u64, start, hi - lo) catch return error.LingEmbedRowsTruncated;
    if (end > file_size) return error.LingEmbedRowsTruncated;
    return .{ .offset = start, .rows = rows, .cols = cols, .row_bytes = row_bytes };
}

fn intOf(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |x| if (x >= 0) @intCast(x) else null,
        else => null,
    };
}

/// The table's shard, open for row reads.
pub const Source = struct {
    fd: c_int,
    table: Table,
    /// The shard's file name in the pack (the load's line names it), owned.
    shard: []u8,

    pub const Stats = struct {
        /// The rows read from the shard: the call's distinct ids.
        reads: usize = 0,
    };

    /// The table `key` in the pack in `dir`: the shard its index names, opened read-only (no read-ahead: rows are read
    /// at random), its header read and checked (`tableOf`). Every refusal is named (error.LingEmbedRows*).
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, key: []const u8, rows: u32, cols: u32) !Source {
        if (comptime !builtin.os.tag.isDarwin()) return error.LingEmbedRowsOpen;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const aa = arena.allocator();
        const index_path = try std.fmt.allocPrintSentinel(aa, "{s}/model.safetensors.index.json", .{dir}, 0);
        const index = readFile(aa, index_path, json_max) orelse return error.LingEmbedRowsIndex;
        const name = try shardOf(aa, index, key);
        const path = try std.fmt.allocPrintSentinel(aa, "{s}/{s}", .{ dir, name }, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.LingEmbedRowsOpen;
        errdefer _ = std.c.close(fd);
        _ = std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0));
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 8) return error.LingEmbedRowsHeader;
        const file_size: u64 = @intCast(size);
        var hl: [8]u8 = undefined;
        if (!preadAll(fd, &hl, 0)) return error.LingEmbedRowsHeader;
        const hlen = std.mem.readInt(u64, &hl, .little);
        if (hlen == 0 or hlen > json_max or hlen > file_size - 8) return error.LingEmbedRowsHeader;
        const hbuf = try aa.alloc(u8, @intCast(hlen));
        if (!preadAll(fd, hbuf, 8)) return error.LingEmbedRowsHeader;
        const t = try tableOf(aa, hbuf, file_size, key, rows, cols);
        const owned = try gpa.dupe(u8, name);
        return .{ .fd = fd, .table = t, .shard = owned };
    }

    pub fn close(self: *Source, gpa: std.mem.Allocator) void {
        if (self.fd >= 0) _ = std.c.close(self.fd);
        self.fd = -1;
        gpa.free(self.shard);
    }

    /// The rows of `ids` into `out` (ids.len rows of `table.row_bytes`, in `ids`' order): each distinct id read once,
    /// in parallel, then copied to its repeats. Refused by name: an id past the table (LingEmbedRowsTokenPastTable), an
    /// `out` of another size (LingEmbedRowsShape), a read that fails or ends early (LingEmbedRowsRead).
    pub fn read(self: *const Source, gpa: std.mem.Allocator, ids: []const u32, out: []u8) !Stats {
        const t = self.table;
        if (ids.len > std.math.maxInt(u32) or out.len != ids.len * t.row_bytes) return error.LingEmbedRowsShape;
        for (ids) |v| if (v >= t.rows) return error.LingEmbedRowsTokenPastTable;
        if (ids.len == 0) return .{};
        // (id, position) pairs in id order: each distinct id's first position reads, the rest copy it
        const order = try gpa.alloc(u64, ids.len);
        defer gpa.free(order);
        for (order, ids, 0..) |*o, v, i| o.* = (@as(u64, v) << 32) | @as(u64, i);
        std.mem.sort(u64, order, {}, std.sort.asc(u64));
        const firsts = try gpa.alloc(u32, ids.len);
        defer gpa.free(firsts);
        var n_first: usize = 0;
        for (order, 0..) |o, k| {
            if (k > 0 and (order[k - 1] >> 32) == (o >> 32)) continue;
            firsts[n_first] = @truncate(o);
            n_first += 1;
        }
        var job: Job = .{ .src = self, .ids = ids, .pos = firsts[0..n_first], .out = out };
        job.runAll();
        if (job.bad.load(.acquire)) return error.LingEmbedRowsRead;
        var src_pos: usize = 0;
        for (order, 0..) |o, k| {
            const p: usize = @as(u32, @truncate(o));
            if (k == 0 or (order[k - 1] >> 32) != (o >> 32)) {
                src_pos = p;
                continue;
            }
            @memcpy(out[p * t.row_bytes ..][0..t.row_bytes], out[src_pos * t.row_bytes ..][0..t.row_bytes]);
        }
        return .{ .reads = n_first };
    }
};

/// One call's reads: row `ids[pos[i]]` into its place in `out`, for each `i`, on GCD's pooled workers (a decode's few
/// rows and a prompt's thousands alike: no thread to spawn).
const Job = struct {
    src: *const Source,
    ids: []const u32,
    pos: []const u32,
    out: []u8,
    bad: std.atomic.Value(bool) = .init(false),

    fn runAll(self: *Job) void {
        if (comptime builtin.os.tag.isDarwin()) {
            dispatch_apply_f(self.pos.len, dispatch_get_global_queue(qos_user_initiated, 0), self, applyOne);
        } else for (0..self.pos.len) |i| self.one(i);
    }

    fn applyOne(ctx: ?*anyopaque, i: usize) callconv(.c) void {
        const self: *Job = @ptrCast(@alignCast(ctx.?));
        self.one(i);
    }

    fn one(self: *Job, i: usize) void {
        const t = self.src.table;
        const p: usize = self.pos[i];
        if (!preadAll(self.src.fd, self.out[p * t.row_bytes ..][0..t.row_bytes], t.rowOffset(self.ids[p]))) self.bad.store(true, .release);
    }
};

const qos_user_initiated: isize = 0x19;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;
extern "c" fn dispatch_apply_f(iterations: usize, queue: ?*anyopaque, context: ?*anyopaque, work: *const fn (?*anyopaque, usize) callconv(.c) void) void;

/// `dst` filled from byte `off` of `fd` (pread until done, an interrupted read retried; false on an error or the end of
/// the file).
fn preadAll(fd: c_int, dst: []u8, off: u64) bool {
    var got: usize = 0;
    while (got < dst.len) {
        const r = std.c.pread(fd, dst[got..].ptr, dst.len - got, @intCast(off + got));
        if (r < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (r <= 0) return false;
        got += @intCast(r);
    }
    return true;
}

/// The file at `path`, at most `max` bytes, in `a`; null when it does not open, is empty or larger, or does not read.
fn readFile(a: std.mem.Allocator, path: [:0]const u8, max: u64) ?[]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const size = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size <= 0 or @as(u64, @intCast(size)) > max) return null;
    const raw = a.alloc(u8, @intCast(size)) catch return null;
    if (!preadAll(fd, raw, 0)) return null;
    return raw;
}

// ── tests ──

/// A test pack in a fresh folder under /tmp, as a pack keeps its embedding: an index naming the shard for `key`, and the
/// shard: its header, a leading tensor (so the table's data does not start at 0), then `key`'s words, row r word j the
/// bits `wordAt(r, j)`. `Spec`'s other fields make the faulty packs the refusals are tested on. Tests only (the GPU byte
/// test, src/ling_embed_test.zig, writes its pack with it too).
pub const TestPack = struct {
    dir: [:0]u8,
    shard_path: [:0]u8,
    index_path: [:0]u8,

    pub const Spec = struct {
        key: []const u8 = test_key,
        rows: u32,
        cols: u32,
        dtype: []const u8 = "BF16",
        /// The shape the header declares, when it is not [rows, cols] (a 1-D shape when `shape_cols` is 0).
        shape_rows: ?u32 = null,
        shape_cols: ?u32 = null,
        /// The table's data cut short by this many bytes at the end of the file (the header still declares all of it).
        short_by: usize = 0,
        /// The name the index gives the shard (the shard is written as `shard-1.safetensors` whatever it says).
        index_shard: []const u8 = "shard-1.safetensors",
        /// The header's key for the table, when it is not `key` (the index still names `key`).
        header_key: ?[]const u8 = null,
        /// The table's words, row-major (rows * cols of them), in place of `wordAt`'s.
        words: ?[]const u16 = null,
        /// A distinct folder for packs made in one test.
        tag: u32 = 0,
    };

    pub const test_key = "model.word_embeddings.weight";

    pub fn make(gpa: std.mem.Allocator, spec: Spec) !TestPack {
        const dir = try std.fmt.allocPrintSentinel(gpa, "/tmp/turboexl3-ling-embed-rows-{d}-{d}-{d}-{d}", .{ std.c.getpid(), spec.rows, spec.cols, spec.tag }, 0);
        errdefer gpa.free(dir);
        _ = std.c.mkdir(dir.ptr, @as(std.c.mode_t, 0o755));
        const shard_path = try std.fmt.allocPrintSentinel(gpa, "{s}/shard-1.safetensors", .{dir}, 0);
        errdefer gpa.free(shard_path);
        const index_path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
        errdefer gpa.free(index_path);
        const lead: usize = 24;
        const tbytes: usize = @as(usize, spec.rows) * spec.cols * 2;
        const sr = spec.shape_rows orelse spec.rows;
        const sc = spec.shape_cols orelse spec.cols;
        var shape_buf: [64]u8 = undefined;
        const shape = if (sc == 0) try std.fmt.bufPrint(&shape_buf, "[{d}]", .{sr}) else try std.fmt.bufPrint(&shape_buf, "[{d},{d}]", .{ sr, sc });
        const header = try std.fmt.allocPrint(gpa, "{{\"__metadata__\":{{\"format\":\"pt\"}},\"lead\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}}}", .{ lead, lead, spec.header_key orelse spec.key, spec.dtype, shape, lead, lead + tbytes });
        defer gpa.free(header);
        const file = try gpa.alloc(u8, 8 + header.len + lead + tbytes);
        defer gpa.free(file);
        std.mem.writeInt(u64, file[0..8], header.len, .little);
        @memcpy(file[8..][0..header.len], header);
        @memset(file[8 + header.len ..][0..lead], 0xEE);
        const body = file[8 + header.len + lead ..];
        if (spec.words) |ws| {
            if (ws.len != @as(usize, spec.rows) * spec.cols) return error.TestPackWords;
            for (ws, 0..) |w, i| std.mem.writeInt(u16, body[i * 2 ..][0..2], w, .little);
        } else for (0..spec.rows) |r| for (0..spec.cols) |j| std.mem.writeInt(u16, body[(r * spec.cols + j) * 2 ..][0..2], wordAt(r, j), .little);
        try writeFile(shard_path, file[0 .. file.len - spec.short_by]);
        const index = try std.fmt.allocPrint(gpa, "{{\"metadata\":{{}},\"weight_map\":{{\"lead\":\"shard-1.safetensors\",\"{s}\":\"{s}\"}}}}", .{ spec.key, spec.index_shard });
        defer gpa.free(index);
        try writeFile(index_path, index);
        return .{ .dir = dir, .shard_path = shard_path, .index_path = index_path };
    }

    pub fn deinit(self: *TestPack, gpa: std.mem.Allocator) void {
        _ = std.c.unlink(self.shard_path.ptr);
        _ = std.c.unlink(self.index_path.ptr);
        _ = std.c.rmdir(self.dir.ptr);
        gpa.free(self.shard_path);
        gpa.free(self.index_path);
        gpa.free(self.dir);
    }

    /// Row `r` word `j`'s bits (r * 4,099 + j * 7, xor-shifted, modulo 2^16), so another row or word reads other bits
    /// at almost every word.
    pub fn wordAt(r: usize, j: usize) u16 {
        const x: u32 = @truncate(r *% 4099 +% j *% 7);
        return @truncate(x ^ (x >> 7) ^ 0x5A5A);
    }

    fn writeFile(path: [:0]const u8, bytes: []const u8) !void {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.TestPackWrite;
        defer _ = std.c.close(fd);
        var done: usize = 0;
        while (done < bytes.len) {
            const w = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
            if (w <= 0) return error.TestPackWrite;
            done += @intCast(w);
        }
    }
};

const testing = std.testing;

fn distinctOf(gpa: std.mem.Allocator, ids: []const u32) !usize {
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(gpa);
    for (ids) |v| try seen.put(gpa, v, {});
    return seen.count();
}

test "EMBED_ROWS' table: a forward's rows by id, in the ids' order, each distinct id read once and its repeats copied, at 1, 7 and 3,000 ids and at Ling's 4,096-word rows, the table's first and last rows among them" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE3BED);
    const rnd = prng.random();
    for ([_][2]u32{ .{ 600, 48 }, .{ 16, 4096 } }) |geo| {
        const rows = geo[0];
        const cols = geo[1];
        var pack = try TestPack.make(gpa, .{ .rows = rows, .cols = cols });
        defer pack.deinit(gpa);
        var src = try Source.open(gpa, pack.dir, TestPack.test_key, rows, cols);
        defer src.close(gpa);
        try testing.expectEqual(@as(usize, cols) * 2, src.table.row_bytes);
        try testing.expectEqualStrings("shard-1.safetensors", src.shard);
        for ([_]usize{ 1, 7, 3000 }) |n| {
            const ids = try gpa.alloc(u32, n);
            defer gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = if (n > 1 and i % 5 == 4) ids[i - 1] else rnd.uintLessThan(u32, rows);
            if (n > 2) {
                ids[0] = 0;
                ids[n - 1] = rows - 1;
            }
            const out = try gpa.alloc(u8, n * src.table.row_bytes);
            defer gpa.free(out);
            @memset(out, 0);
            const st = try src.read(gpa, ids, out);
            try testing.expectEqual(try distinctOf(gpa, ids), st.reads);
            for (ids, 0..) |v, i| for (0..cols) |j| {
                try testing.expectEqual(TestPack.wordAt(v, j), std.mem.readInt(u16, out[(i * cols + j) * 2 ..][0..2], .little));
            };
        }
    }
}

test "EMBED_ROWS' table: an id past the table and an output of another size are refused by name" {
    const gpa = testing.allocator;
    var pack = try TestPack.make(gpa, .{ .rows = 17, .cols = 8, .tag = 1 });
    defer pack.deinit(gpa);
    var src = try Source.open(gpa, pack.dir, TestPack.test_key, 17, 8);
    defer src.close(gpa);
    var one: [16]u8 = undefined;
    try testing.expectError(error.LingEmbedRowsTokenPastTable, src.read(gpa, &.{17}, &one));
    try testing.expectError(error.LingEmbedRowsShape, src.read(gpa, &.{ 1, 2 }, &one));
    // the last row reads
    const st = try src.read(gpa, &.{16}, &one);
    try testing.expectEqual(@as(usize, 1), st.reads);
    try testing.expectEqual(TestPack.wordAt(16, 7), std.mem.readInt(u16, one[14..16], .little));
}

test "EMBED_ROWS' table: a pack whose table is not BF16 [rows, cols] inside its shard, or whose index does not name it in the pack's folder, is refused by name" {
    const gpa = testing.allocator;
    const Case = struct { spec: TestPack.Spec, want: anyerror };
    const cases = [_]Case{
        .{ .spec = .{ .rows = 16, .cols = 8, .dtype = "F16", .tag = 10 }, .want = error.LingEmbedRowsNotBf16 },
        .{ .spec = .{ .rows = 16, .cols = 8, .shape_rows = 15, .tag = 11 }, .want = error.LingEmbedRowsShape },
        .{ .spec = .{ .rows = 16, .cols = 8, .shape_cols = 4, .tag = 12 }, .want = error.LingEmbedRowsShape },
        .{ .spec = .{ .rows = 16, .cols = 8, .shape_cols = 0, .tag = 13 }, .want = error.LingEmbedRowsShape },
        .{ .spec = .{ .rows = 16, .cols = 8, .short_by = 1, .tag = 14 }, .want = error.LingEmbedRowsTruncated },
        .{ .spec = .{ .rows = 16, .cols = 8, .header_key = "model.embed_tokens.weight", .tag = 15 }, .want = error.LingEmbedRowsMissing },
        .{ .spec = .{ .rows = 16, .cols = 8, .index_shard = "../shard-1.safetensors", .tag = 16 }, .want = error.LingEmbedRowsIndex },
        .{ .spec = .{ .rows = 16, .cols = 8, .index_shard = "sub/shard-1.safetensors", .tag = 17 }, .want = error.LingEmbedRowsIndex },
        .{ .spec = .{ .rows = 16, .cols = 8, .index_shard = "shard-2.safetensors", .tag = 18 }, .want = error.LingEmbedRowsOpen },
    };
    for (cases) |c| {
        var pack = try TestPack.make(gpa, c.spec);
        defer pack.deinit(gpa);
        try testing.expectError(c.want, Source.open(gpa, pack.dir, TestPack.test_key, 16, 8));
    }
    // the right pack asked for another geometry, or for a key its index lacks; and no pack at all
    var pack = try TestPack.make(gpa, .{ .rows = 16, .cols = 8, .tag = 19 });
    defer pack.deinit(gpa);
    try testing.expectError(error.LingEmbedRowsShape, Source.open(gpa, pack.dir, TestPack.test_key, 17, 8));
    try testing.expectError(error.LingEmbedRowsShape, Source.open(gpa, pack.dir, TestPack.test_key, 16, 16));
    try testing.expectError(error.LingEmbedRowsIndex, Source.open(gpa, pack.dir, "model.embed_tokens.weight", 16, 8));
    try testing.expectError(error.LingEmbedRowsIndex, Source.open(gpa, "/tmp/turboexl3-ling-embed-rows-no-such-pack", TestPack.test_key, 16, 8));
}

test "EMBED_ROWS' table at Ling-3.1-flash's geometry: 157,184 x 4,096 bf16 is 1,287,651,328 bytes (1,228 MiB), row 0 at the header's end plus its data_offsets, row r r x 8,192 bytes on, and a data range past the file refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const rows: u32 = 157184;
    const cols: u32 = 4096;
    const tbytes: u64 = 1_287_651_328;
    // a header as the repack writes one: the table after another tensor's 5,120 bytes
    const header = try std.fmt.allocPrint(aa, "{{\"model.norm.weight\":{{\"dtype\":\"BF16\",\"shape\":[2560],\"data_offsets\":[0,5120]}},\"model.word_embeddings.weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[5120,{d}]}}}}", .{ rows, cols, 5120 + tbytes });
    const data_end = 8 + header.len + 5120 + tbytes;
    const t = try tableOf(aa, header, data_end, TestPack.test_key, rows, cols);
    try testing.expectEqual(tbytes, t.bytes());
    try testing.expectEqual(@as(u64, 1228), t.bytes() >> 20);
    try testing.expectEqual(@as(usize, 8192), t.row_bytes);
    try testing.expectEqual(@as(u64, 8 + header.len + 5120), t.offset);
    try testing.expectEqual(t.offset + 157183 * 8192, t.rowOffset(157183));
    try testing.expectEqual(t.offset + tbytes, t.rowOffset(157183) + t.row_bytes);
    // one byte short of the table's end: refused
    try testing.expectError(error.LingEmbedRowsTruncated, tableOf(aa, header, data_end - 1, TestPack.test_key, rows, cols));
    // offsets past a signed 64-bit integer do not parse as one (refused), and a range near its end lies past the file
    const past = "{\"model.word_embeddings.weight\":{\"dtype\":\"BF16\",\"shape\":[1,4],\"data_offsets\":[9223372036854775800,9223372036854775808]}}";
    try testing.expectError(error.LingEmbedRowsHeader, tableOf(aa, past, std.math.maxInt(u64), TestPack.test_key, 1, 4));
    const far = "{\"model.word_embeddings.weight\":{\"dtype\":\"BF16\",\"shape\":[1,4],\"data_offsets\":[9223372036854775799,9223372036854775807]}}";
    try testing.expectError(error.LingEmbedRowsTruncated, tableOf(aa, far, 1 << 40, TestPack.test_key, 1, 4));
    try testing.expectError(error.LingEmbedRowsHeader, tableOf(aa, "not json", 1 << 20, TestPack.test_key, rows, cols));
}
