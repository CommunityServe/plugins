//! The a4k record layout's byte gate (zzzzt), a test of its own in `zig build test` (build.zig), MLX-free: the copied read
//! pool (its C file alone, no MLX, no stream) reads the same records' parts from a classic file and from an a4k one
//! (expert/a4k.zig's geometry), each record's rows and lengths handed to it in the file's order on disk as the patched
//! `expert.io.Records` hands them (`a4k.toDisk`), DeepSeek-V4.1's K = 3 and K = 2 records (two layers of two experts):
//! through the staging route, through the PARTS route (every a4k code read straight into its row: three parts a record
//! against the classic layout's gate code alone), from a speculative record read whole (the record's extent) and
//! adopted, and from a pre-read range bound to the demand that needs it (the gate/up range at the record's lead). Every
//! slot row holds the record's bytes either way.
//!
//! The real-file gate: with A4K_GATE_PACKS=<classic pack dir>:<a4k pack dir> set, the records of the two packs'
//! experts.bin are read through the PARTS route, each pack's layer table from its expert-manifest-v2.json (the a4k one
//! checked against `a4k.geometry`), and every slot row compared: every expert of the first K = 3 and the first K = 2
//! layer, and one expert of every other layer. Unset, that test only checks the setting's parse.

const std = @import("std");
const testing = std.testing;
const a4k = @import("mlx_serve/sdk/weights/streamed/expert/a4k.zig");

// a4k.zig's own tests (its geometry at DeepSeek-V4.1's records, the order's permutation) run with these.
test {
    _ = a4k;
}

extern fn q3ld_spec_config(nthreads: i32, bufs: ?[*]const u64, nslots: i32, slot_bytes: i64, rec_len: i64, chunk: i64, counters: ?[*]i64) c_int;
extern fn q3ld_spec_streams(idle_busy: i32) c_int;
extern fn q3ld_direct_config(on: i32) c_int;
extern fn q3ld_parts_config(block: i64) c_int;
extern fn q3ld_start(nw: i32, staging_ptrs: [*]const u64, sbytes: i64, psize: i64, res: [*]i64, n_tickets: i64, log: [*]i64, n_log: i64, gauge: *[6]i64) c_int;
extern fn q3ld_submit(fd: i32, file_size: i64, deadline: i64, n: i32, ngu: i32, ndown: i32, offsets: [*]const i64, rows: [*]const [*]const u64, lens: [*]const i64, first: i64) c_int;
extern fn q3ld_spec_step_len(fd: i32, file_size: i64, cur: i64, n: i32, bases: ?[*]const i64, len: i64) i32;
extern fn q3ld_pre_config(ngu: i32, ndown: i32, lens: ?[*]const i64) c_int;
extern fn q3ld_pre_read_lens(fd: i32, file_size: i64, tag: i64, n: i32, bases: [*]const i64, lens: [*]const i64) i32;
extern fn q3ld_seq() i64;
extern fn q3ld_wait(seen: i64, timeout_ns: i64) i64;
extern fn q3ld_quiesce(timeout_ns: i64) c_int;
extern fn q3ld_stop() c_int;

/// The pool's Metal event backend (src/host_shims.zig in the plugin): never armed here, every gate off.
export fn q3ld_mtl_signal(event: ?*anyopaque, value: u64) void {
    _ = event;
    _ = value;
}

const counters_n = 75;
/// The pool's counters (q3_lookahead4_exl3.c's SC_ enum).
const sc_landed = 4;
const sc_claimed = 8;
const sc_adopt_ranges = 13;
const sc_pre_started = 50;
const sc_pre_bound = 51;
const sc_pre_cancelled = 52;
const sc_pre_served = 54;
const sc_parts = 72;
const res_w = 8;
const st_ok = 0;
const st_pending = -1;

const hidden: u64 = 5120;
const inter: u64 = 2304;
const block: u64 = 4096;
const n_comp = a4k.n_components;
const ngu = a4k.gu_components;
const experts = 2;
const workers = 2;
const staging_bytes: u64 = 9 << 20;
const tickets = 16;
const n_spec = 4;

const Layout = enum { classic, a4k };

/// One layer's record geometry in a layout: each component's length and offset (component order), its order on disk, the
/// record's extent and its bytes in the file.
const Geom = struct {
    len: [n_comp]u64,
    off: [n_comp]u64,
    disk: [n_comp]u8,
    extent: u64,
    record: u64,
    fn of(k: u64, layout: Layout) Geom {
        var len: [n_comp]u64 = undefined;
        for (0..3) |p| {
            const in: u64 = if (p == 2) inter else hidden;
            const out: u64 = if (p == 2) hidden else inter;
            len[p * 3] = (in / 16) * (out / 16) * 16 * k * 2;
            len[p * 3 + 1] = out * 2;
            len[p * 3 + 2] = in * 2;
        }
        if (layout == .a4k) {
            const g = a4k.geometry(len).?;
            return .{ .len = len, .off = g.offsets, .disk = a4k.disk_order, .extent = g.extent, .record = g.record_bytes };
        }
        var off: [n_comp]u64 = undefined;
        var at: u64 = 0;
        for (&off, len) |*o, n| {
            o.* = at;
            at += n;
        }
        return .{ .len = len, .off = off, .disk = a4k.identity, .extent = at, .record = std.mem.alignForward(u64, at, block) };
    }
    fn guStart(g: Geom) u64 {
        return g.off[g.disk[0]];
    }
    fn downStart(g: Geom) u64 {
        return g.off[g.disk[ngu]];
    }
    fn guLen(g: Geom) u64 {
        var n: u64 = 0;
        for (g.len[0..ngu]) |x| n += x;
        return n;
    }
    /// One record's lengths in the order on disk, as the pool takes them.
    fn diskLens(g: Geom) [n_comp]i64 {
        var out: [n_comp]i64 = undefined;
        for (&out, a4k.toDisk(u64, n_comp, &g.len, &g.disk)) |*o, x| o.* = @intCast(x);
        return out;
    }
};

const ks = [2]u64{ 3, 2 };

fn layerBase(layout: Layout, layer: usize) u64 {
    return if (layer == 0) 0 else experts * Geom.of(ks[0], layout).record;
}

fn recordOffset(layout: Layout, layer: usize, e: usize) u64 {
    return layerBase(layout, layer) + e * Geom.of(ks[layer], layout).record;
}

const page = std.heap.page_size_min;
const Map = []align(page) u8;

fn mapBytes(n: usize) !Map {
    return std.posix.mmap(null, std.mem.alignForward(usize, n + page, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
}

/// The records' parts: [layer][expert][component] bytes, the same in both layouts.
const Parts = struct {
    bytes: [2][experts][n_comp][]u8 = undefined,
    fn init(a: std.mem.Allocator, seed: u64) !Parts {
        var p: Parts = .{};
        var prng = std.Random.DefaultPrng.init(seed);
        const g3 = Geom.of(3, .classic);
        const g2 = Geom.of(2, .classic);
        var made: usize = 0;
        errdefer for (0..made) |i| a.free(p.bytes[i / (experts * n_comp)][(i / n_comp) % experts][i % n_comp]);
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| {
            const n = (if (l == 0) g3 else g2).len[c];
            p.bytes[l][e][c] = try a.alloc(u8, n);
            made += 1;
            prng.random().bytes(p.bytes[l][e][c]);
        };
        return p;
    }
    fn deinit(p: *Parts, a: std.mem.Allocator) void {
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| a.free(p.bytes[l][e][c]);
    }
    /// The file image of both layers' records in `layout` (zero pads).
    fn image(p: *const Parts, a: std.mem.Allocator, layout: Layout) ![]u8 {
        const n = layerBase(layout, 1) + experts * Geom.of(ks[1], layout).record;
        const img = try a.alloc(u8, n);
        @memset(img, 0);
        for (0..2) |l| {
            const g = Geom.of(ks[l], layout);
            for (0..experts) |e| for (0..n_comp) |c| {
                const at = recordOffset(layout, l, e) + g.off[c];
                @memcpy(img[at..][0..g.len[c]], p.bytes[l][e][c]);
            };
        }
        return img;
    }
};

const TempFile = struct {
    path: [:0]u8,
    fd: c_int,
    size: u64,
    fn create(a: std.mem.Allocator, tag: []const u8, bytes: []const u8) !TempFile {
        const path = try std.fmt.allocPrintSentinel(a, "/tmp/turboexl3_ds41-a4k-{s}-{d}.bin", .{ tag, std.c.getpid() }, 0);
        errdefer a.free(path);
        const w = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (w < 0) return error.TempFile;
        var done: usize = 0;
        while (done < bytes.len) {
            const r = std.c.write(w, bytes[done..].ptr, bytes.len - done);
            if (r <= 0) {
                _ = std.c.close(w);
                return error.TempFile;
            }
            done += @intCast(r);
        }
        _ = std.c.close(w);
        const fd = try openUncached(path);
        return .{ .path = path, .fd = fd, .size = bytes.len };
    }
    fn destroy(f: TempFile, a: std.mem.Allocator) void {
        _ = std.c.close(f.fd);
        _ = std.c.unlink(f.path);
        a.free(f.path);
    }
};

/// The reader's descriptor: read-only, the page cache bypassed, no read-ahead (expert/io.zig openUncached).
fn openUncached(path: [*:0]const u8) !c_int {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.Open;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0 or std.c.fcntl(fd, std.c.F.RDAHEAD, @as(c_int, 0)) != 0) {
        _ = std.c.close(fd);
        return error.Open;
    }
    return fd;
}

/// Each record's rows, a map a component (on the block), in component order.
const Rows = struct {
    maps: [2][experts][n_comp]Map = undefined,
    ptr: [2][experts][n_comp]u64 = undefined,
    fn init() !Rows {
        var r: Rows = .{};
        var made: usize = 0;
        errdefer for (0..made) |i| std.posix.munmap(r.maps[i / (experts * n_comp)][(i / n_comp) % experts][i % n_comp]);
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| {
            const m = try mapBytes(Geom.of(ks[l], .classic).len[c]);
            r.maps[l][e][c] = m;
            made += 1;
            r.ptr[l][e][c] = @intFromPtr(m.ptr);
        };
        return r;
    }
    fn deinit(r: *Rows) void {
        for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| std.posix.munmap(r.maps[l][e][c]);
    }
    fn row(r: *const Rows, l: usize, e: usize, c: usize) []const u8 {
        const p: [*]const u8 = @ptrFromInt(r.ptr[l][e][c]);
        return p[0..Geom.of(ks[l], .classic).len[c]];
    }
};

fn waitAll(res: []const i64, first: usize, count: usize) !void {
    var seen = q3ld_seq();
    var rounds: u32 = 0;
    while (true) {
        var done = true;
        for (first..first + count) |t| {
            if (@atomicLoad(i64, &res[t * res_w], .acquire) == st_pending) done = false;
        }
        if (done) return;
        rounds += 1;
        if (rounds > 300) return error.PoolTimeout;
        seen = q3ld_wait(seen, 100 * std.time.ns_per_ms);
    }
}

fn waitCounter(counters: *const [counters_n]i64, k: usize, want: i64) !void {
    var i: u32 = 0;
    const ts = std.c.timespec{ .sec = 0, .nsec = 10_000_000 };
    while (@atomicLoad(i64, &counters[k], .acquire) < want) : (i += 1) {
        if (i > 3000) return error.PoolTimeout;
        _ = std.c.nanosleep(&ts, null);
    }
}

const Route = enum { staged, parts, spec, pre };

/// A pool over `file` (laid out `layout`) through `route`, both layers (one submit a layer, its two experts, each landed
/// before the next), into `rows`; each record's rows and lengths handed in the layout's order on disk. `.spec`: each
/// layer's records first read whole by the speculative class (the record's extent) and the demand adopting them;
/// `.pre`: each layer's gate/up and down ranges pre-read first (at the record's gate/up start), the demand binding the
/// ones in flight. A pre-range's worker holds at its bind point once read, so with two workers the two gate/up ranges
/// (queued first) are in flight at the submit and bound, the two down ranges still queued and cancelled (the demand
/// reads them). Returns the counters.
fn run(file: TempFile, layout: Layout, route: Route, rows: *const Rows) ![counters_n]i64 {
    var counters: [counters_n]i64 = @splat(0);
    var res: [tickets * res_w]i64 = @splat(0);
    var log: [tickets]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const stage = try std.posix.mmap(null, workers * staging_bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(stage);
    var ptrs: [workers]u64 = undefined;
    for (&ptrs, 0..) |*p, w| p.* = @intFromPtr(stage.ptr) + w * staging_bytes;
    const psize: u64 = std.heap.pageSize();
    const rec_len = Geom.of(ks[0], layout).extent;
    const slot = std.mem.alignForward(u64, rec_len, psize) + 2 * psize;
    const spec_on = route == .spec or route == .pre;
    const spec_mem = try std.posix.mmap(null, n_spec * slot, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(spec_mem);
    var bufs: [n_spec]u64 = undefined;
    for (&bufs, 0..) |*b, i| b.* = @intFromPtr(spec_mem.ptr) + i * slot;
    if (spec_on) {
        const chunk = std.mem.alignForward(u64, slot / 4, psize);
        try testing.expectEqual(@as(c_int, 0), q3ld_spec_config(1, &bufs, n_spec, @intCast(slot), @intCast(rec_len), @intCast(chunk), &counters));
        try testing.expectEqual(@as(c_int, 0), q3ld_spec_streams(0));
    } else try testing.expectEqual(@as(c_int, 0), q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, 0), q3ld_direct_config(0));
    try testing.expectEqual(@as(c_int, 0), q3ld_parts_config(if (route == .parts) @intCast(block) else 0));
    try testing.expectEqual(@as(c_int, 0), q3ld_start(workers, &ptrs, staging_bytes, @intCast(psize), &res, tickets, &log, tickets, &gauge));
    var stopped = false;
    defer if (!stopped) {
        _ = q3ld_quiesce(10 * std.time.ns_per_s);
        _ = q3ld_stop();
    };
    if (route == .pre) {
        const lens = Geom.of(ks[0], layout).diskLens();
        try testing.expectEqual(@as(c_int, 0), q3ld_pre_config(ngu, n_comp - ngu, &lens));
    }
    for (0..2) |l| {
        const g = Geom.of(ks[l], layout);
        const lens = g.diskLens();
        if (route == .spec) {
            var bases: [experts]i64 = undefined;
            for (&bases, 0..) |*b, e| b.* = @intCast(recordOffset(layout, l, e));
            try testing.expectEqual(@as(i32, experts), q3ld_spec_step_len(file.fd, @intCast(file.size), @intCast(2 * l), experts, &bases, @intCast(g.extent)));
            try waitCounter(&counters, sc_landed, @intCast(experts * (l + 1)));
        }
        if (route == .pre) {
            var bases: [experts]i64 = undefined;
            for (&bases, 0..) |*b, e| b.* = @intCast(recordOffset(layout, l, e) + g.guStart());
            try testing.expectEqual(@as(i32, 2 * experts), q3ld_pre_read_lens(file.fd, @intCast(file.size), @intCast(100 + l), experts, &bases, &lens));
            try waitCounter(&counters, sc_pre_started, @intCast(workers * (l + 1)));
        }
        var offsets: [2 * experts]i64 = undefined;
        var drows: [experts][n_comp]u64 = undefined;
        var item: [experts][*]const u64 = undefined;
        for (0..experts) |e| {
            offsets[e] = @intCast(recordOffset(layout, l, e) + g.guStart());
            offsets[experts + e] = @intCast(recordOffset(layout, l, e) + g.downStart());
            drows[e] = a4k.toDisk(u64, n_comp, &rows.ptr[l][e], &g.disk);
            item[e] = &drows[e];
        }
        try testing.expectEqual(@as(c_int, 0), q3ld_submit(file.fd, @intCast(file.size), -1, experts, ngu, n_comp - ngu, &offsets, &item, &lens, @intCast(4 * l)));
        try waitAll(&res, 4 * l, 2 * experts);
        for (0..2 * experts) |i| try testing.expectEqual(@as(i64, st_ok), res[(4 * l + i) * res_w]);
    }
    try testing.expectEqual(@as(c_int, 0), q3ld_quiesce(10 * std.time.ns_per_s));
    // the pre-read class disarmed, so the next pool (and test) starts without it
    if (route == .pre) try testing.expectEqual(@as(c_int, 0), q3ld_pre_config(0, 0, null));
    const out = counters;
    try testing.expectEqual(@as(c_int, 0), q3ld_stop());
    stopped = true;
    return out;
}

fn expectRowsHoldParts(rows: *const Rows, parts: *const Parts) !void {
    for (0..2) |l| for (0..experts) |e| for (0..n_comp) |c| {
        try testing.expectEqualSlices(u8, parts.bytes[l][e][c], rows.row(l, e, c));
    };
}

test "a4k byte gate: the pool's slot rows hold every part from the classic and the a4k file, staged, by PARTS, adopted from a speculative record and bound to a pre-read" {
    const a = testing.allocator;
    var parts = try Parts.init(a, 0xa4_4096);
    defer parts.deinit(a);
    const img_c = try parts.image(a, .classic);
    defer a.free(img_c);
    const img_a = try parts.image(a, .a4k);
    defer a.free(img_a);
    const fc = try TempFile.create(a, "classic", img_c);
    defer fc.destroy(a);
    const fa = try TempFile.create(a, "a4k", img_a);
    defer fa.destroy(a);
    var code_bytes: i64 = 0;
    var gate_bytes: i64 = 0;
    var total: i64 = 0;
    for (0..2) |l| {
        const g = Geom.of(ks[l], .classic);
        code_bytes += @intCast(experts * (g.len[0] + g.len[3] + g.len[6]));
        gate_bytes += @intCast(experts * g.len[0]);
        total += @intCast(experts * g.extent);
    }
    for ([_]Route{ .staged, .parts, .spec, .pre }) |route| {
        for ([_]Layout{ .classic, .a4k }) |layout| {
            var rows = try Rows.init();
            defer rows.deinit();
            const cnt = try run(if (layout == .a4k) fa else fc, layout, route, &rows);
            try expectRowsHoldParts(&rows, &parts);
            switch (route) {
                .staged => try testing.expectEqual([3]i64{ 0, 0, 0 }, cnt[sc_parts..][0..3].*),
                // PARTS: the a4k file's three codes a record read straight into their rows, the classic one's gate code alone
                .parts => if (layout == .a4k)
                    try testing.expectEqual([3]i64{ 3 * 2 * experts, code_bytes, total - code_bytes }, cnt[sc_parts..][0..3].*)
                else
                    try testing.expectEqual([3]i64{ 2 * experts, gate_bytes, total - gate_bytes }, cnt[sc_parts..][0..3].*),
                // every record read whole by the speculative class and both its ranges adopted
                .spec => {
                    try testing.expectEqual(@as(i64, 2 * experts), cnt[sc_claimed]);
                    try testing.expectEqual(@as(i64, 2 * 2 * experts), cnt[sc_adopt_ranges]);
                },
                // each layer's two gate/up ranges pre-read (at the record's gate/up start, the layout's lengths) and bound to
                // the demand, their results served from the pre-ranges; its two down ranges cancelled
                .pre => {
                    try testing.expectEqual(@as(i64, 2 * workers), cnt[sc_pre_bound]);
                    try testing.expectEqual(@as(i64, 2 * workers), cnt[sc_pre_served]);
                    try testing.expectEqual(@as(i64, 2 * (2 * experts - workers)), cnt[sc_pre_cancelled]);
                },
            }
        }
    }
}

test "a4k byte gate: the geometry the test reads by is the bank's (a4k.zig), the spans contiguous and the down range after the gate/up one" {
    for (ks) |k| {
        const g = Geom.of(k, .a4k);
        try testing.expectEqual(g.guStart() + g.guLen(), g.downStart());
        for ([_]usize{ 0, 3, 6 }) |c| try testing.expectEqual(@as(u64, 0), g.off[c] % block);
        try testing.expectEqual(Geom.of(k, .classic).record + block, g.record);
        try testing.expect(g.guLen() <= staging_bytes);
    }
}

// ── The real-file gate (A4K_GATE_PACKS) ──

const SegJ = struct { component: []const u8, offset: u64, length: u64 };
const LayerJ = struct { layer: u64, K: u64, record_bytes: u64, base_offset: u64, segments: []const SegJ };
const SideJ = struct { file: []const u8, size: u64, layout: ?[]const u8 = null };
const DimsJ = struct { n_experts: u64, n_layers: u64 };
const ManJ = struct { sidecar: SideJ, layers: []const LayerJ, dims: DimsJ };

fn readWhole(a: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.Open;
    defer _ = std.c.close(fd);
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(a);
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const r = std.c.read(fd, &buf, buf.len);
        if (r < 0) return error.Read;
        if (r == 0) break;
        try list.appendSlice(a, buf[0..@intCast(r)]);
    }
    return list.toOwnedSlice(a);
}

const Pack = struct {
    parsed: std.json.Parsed(ManJ),
    text: []u8,
    fd: c_int,
    size: u64,
    layout: Layout,
    fn open(a: std.mem.Allocator, dir: []const u8) !Pack {
        const mp = try std.fmt.allocPrintSentinel(a, "{s}/expert-manifest-v2.json", .{dir}, 0);
        defer a.free(mp);
        const text = try readWhole(a, mp);
        errdefer a.free(text);
        const parsed = try std.json.parseFromSlice(ManJ, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        const m = parsed.value;
        var layout: Layout = .classic;
        if (m.sidecar.layout) |name| {
            if (std.mem.eql(u8, name, "a4k")) {
                layout = .a4k;
            } else if (!std.mem.eql(u8, name, "classic")) return error.Layout;
        }
        const bp = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, m.sidecar.file }, 0);
        defer a.free(bp);
        const fd = try openUncached(bp);
        return .{ .parsed = parsed, .text = text, .fd = fd, .size = m.sidecar.size, .layout = layout };
    }
    fn close(p: *Pack, a: std.mem.Allocator) void {
        _ = std.c.close(p.fd);
        p.parsed.deinit();
        a.free(p.text);
    }
    /// Layer `l`'s geometry from the manifest; an a4k layer's offsets must be `a4k.geometry`'s.
    fn geom(p: *const Pack, l: usize) !Geom {
        const t = p.parsed.value.layers[l];
        if (t.segments.len != n_comp) return error.Segments;
        var g: Geom = .{ .len = undefined, .off = undefined, .disk = if (p.layout == .a4k) a4k.disk_order else a4k.identity, .extent = 0, .record = t.record_bytes };
        for (t.segments, 0..) |s, c| {
            g.len[c] = s.length;
            g.off[c] = s.offset;
            g.extent = @max(g.extent, s.offset + s.length);
        }
        const want = Geom.of(t.K, p.layout);
        if (!std.mem.eql(u64, &want.len, &g.len) or !std.mem.eql(u64, &want.off, &g.off) or want.record != g.record) return error.GeometryDiffers;
        return g;
    }
};

test "a4k real-file gate (A4K_GATE_PACKS=<classic pack>:<a4k pack>): the two packs' records give the same slot rows" {
    const env = std.c.getenv("A4K_GATE_PACKS") orelse return;
    const a = testing.allocator;
    const spec = std.mem.span(env);
    const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.A4kGatePacks;
    var pc = try Pack.open(a, spec[0..colon]);
    defer pc.close(a);
    var pa = try Pack.open(a, spec[colon + 1 ..]);
    defer pa.close(a);
    try testing.expectEqual(Layout.classic, pc.layout);
    try testing.expectEqual(Layout.a4k, pa.layout);
    const nl: usize = @intCast(pc.parsed.value.dims.n_layers);
    const ne: u64 = pc.parsed.value.dims.n_experts;
    try testing.expectEqual(pc.parsed.value.dims.n_layers, pa.parsed.value.dims.n_layers);
    // the first K = 3 and the first K = 2 layer whole, one expert of every other layer
    var whole: [2]?usize = .{ null, null };
    for (pc.parsed.value.layers, 0..) |t, l| {
        if (t.K == 3 and whole[0] == null) whole[0] = l;
        if (t.K == 2 and whole[1] == null) whole[1] = l;
    }
    var counters: [counters_n]i64 = @splat(0);
    var res: [tickets * res_w]i64 = @splat(0);
    var log: [tickets]i64 = @splat(0);
    var gauge: [6]i64 = @splat(0);
    const stage = try std.posix.mmap(null, workers * staging_bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(stage);
    var ptrs: [workers]u64 = undefined;
    for (&ptrs, 0..) |*p, w| p.* = @intFromPtr(stage.ptr) + w * staging_bytes;
    try testing.expectEqual(@as(c_int, 0), q3ld_spec_config(0, null, 0, 0, 0, 0, &counters));
    try testing.expectEqual(@as(c_int, 0), q3ld_direct_config(0));
    try testing.expectEqual(@as(c_int, 0), q3ld_parts_config(@intCast(block)));
    try testing.expectEqual(@as(c_int, 0), q3ld_start(workers, &ptrs, staging_bytes, @intCast(std.heap.pageSize()), &res, tickets, &log, tickets, &gauge));
    defer {
        _ = q3ld_quiesce(10 * std.time.ns_per_s);
        _ = q3ld_stop();
    }
    // one row set a pack, each component's row as long as its longest segment (K = 3)
    var maps: [2][n_comp]Map = undefined;
    const g3 = Geom.of(3, .classic);
    for (0..2) |s| for (0..n_comp) |c| {
        maps[s][c] = try mapBytes(g3.len[c]);
    };
    defer for (0..2) |s| for (0..n_comp) |c| std.posix.munmap(maps[s][c]);
    var n_records: u64 = 0;
    var direct_parts: i64 = 0;
    for (0..nl) |l| {
        const gc = try pc.geom(l);
        const ga = try pa.geom(l);
        const tc = pc.parsed.value.layers[l];
        const ta = pa.parsed.value.layers[l];
        const all = (whole[0] != null and whole[0].? == l) or (whole[1] != null and whole[1].? == l);
        const lu: u64 = @intCast(l);
        var e: u64 = if (all) 0 else (7 * lu) % ne;
        const end: u64 = if (all) ne else e + 1;
        while (e < end) : (e += 1) {
            for ([_]usize{ 0, 1 }) |s| {
                const p = if (s == 0) &pc else &pa;
                const g = if (s == 0) gc else ga;
                const t = if (s == 0) tc else ta;
                const rec = t.base_offset + e * t.record_bytes;
                const offsets = [2]i64{ @intCast(rec + g.guStart()), @intCast(rec + g.downStart()) };
                var rows: [n_comp]u64 = undefined;
                for (&rows, 0..) |*r, c| r.* = @intFromPtr(maps[s][c].ptr);
                const drows = a4k.toDisk(u64, n_comp, &rows, &g.disk);
                const item = [1][*]const u64{&drows};
                const lens = g.diskLens();
                try testing.expectEqual(@as(c_int, 0), q3ld_submit(p.fd, @intCast(p.size), -1, 1, ngu, n_comp - ngu, &offsets, &item, &lens, 0));
                try waitAll(&res, 0, 2);
                try testing.expectEqual(@as(i64, st_ok), res[0]);
                try testing.expectEqual(@as(i64, st_ok), res[res_w]);
            }
            for (0..n_comp) |c| {
                const n = gc.len[c];
                try testing.expectEqualSlices(u8, maps[0][c][0..n], maps[1][c][0..n]);
            }
            n_records += 1;
        }
    }
    try testing.expectEqual(@as(c_int, 0), q3ld_quiesce(10 * std.time.ns_per_s));
    direct_parts = counters[sc_parts];
    std.debug.print("a4k real-file gate: {d} records of {d} layers read from both packs, every slot row equal; {d} parts read direct\n", .{ n_records, nl, direct_parts });
    // every record's gate code direct from the classic pack, all three codes from the a4k one
    try testing.expectEqual(@as(i64, @intCast(4 * n_records)), direct_parts);
    try testing.expect(n_records >= 2 * ne);
}
