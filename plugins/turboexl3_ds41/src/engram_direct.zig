//! `MLX_SERVE_DSV41_ENGRAM_DIRECT` (read at load; unset, empty or `1`: on; `0`: off; anything else is refused by name):
//! a prompt's Engram rows read once per table and past the page cache; a `[engram]` line for each prompt grid either
//! way.
//!
//! The pack's tables (oMLX's lossless layout, ~100 GB a layer) keep a row's e4m3 codes (256 B) and its e8m0 scales
//! (8 B) in two regions of one file. Off, a prompt grid's rows are read span by span (256 tokens: 6,144 ids a table),
//! each span's distinct rows by two preads a row through the page cache: a 16K-token prompt's ~587K row reads cost ~522K
//! 16 KiB page-ins (~8 GiB of file cache, ~1.8 s for the first table, layer 1 waiting ~1.1 s with the pack's files
//! evicted: en1, w-pio-20261101-1419). The page cache absorbs the rows that repeat across spans (~47%) and the scale
//! pages shared by neighbouring rows; read past it one pread at a time, the same reads cost a 4 KiB device IO each
//! (en1's direct arm: 1.17M IOs, 8% slower).
//! On, a prompt grid's table is read whole: every span's ids sorted and made distinct, the distinct rows read past the
//! cache (F_NOCACHE: the file system blocks that hold them, fstatfs `f_bsize`, APFS 4,096 B, straight into an aligned
//! buffer) on up to 64 threads, each a contiguous run of the sorted rows that keeps its last scale block for the rows
//! after it, then each span's rows copied out of them; a decode token's rows read past the cache as before, a pread a
//! half-row. Exact: the same bytes land in the same rows; only how they are read changes.

const std = @import("std");
const builtin = @import("builtin");

/// The switch (`configure`, at load, before the tables open: on unless `=0`).
pub var on: bool = false;

pub const Error = error{EngramDirectSetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.EngramDirectSetting;
}

/// The most bytes one read through the blocks may span: a longer row reads with one pread as before.
pub const max_span: usize = 16384;

/// The tables' files read through their blocks (`register`): fd and block.
const max_files = 8;
var files: [max_files]struct { fd: c_int, block: u64 } = @splat(.{ .fd = -1, .block = 0 });
var n_files: usize = 0;

/// At a table's open, with the switch on: the file set F_NOCACHE and its block kept, when the block is a power of two
/// in [512, 4096] (else the file stays on the page cache).
pub fn register(fd: c_int) void {
    if (!on or comptime !builtin.os.tag.isDarwin()) return;
    if (n_files == max_files) return;
    var st: Statfs = undefined;
    if (fstatfs(fd, &st) != 0) return;
    const b: u64 = st.f_bsize;
    if (b < 512 or b > 4096 or b & (b - 1) != 0) return;
    if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) == -1) return;
    files[n_files] = .{ .fd = fd, .block = b };
    n_files += 1;
}
/// Darwin's `struct statfs` (64-bit inodes, arm64's only layout): only `f_bsize` is read.
const Statfs = extern struct { f_bsize: u32, f_iosize: i32, rest: [2160]u8 };
extern "c" fn fstatfs(fd: c_int, buf: *Statfs) c_int;
comptime {
    if (@sizeOf(Statfs) != 2168) @compileError("engram_direct: struct statfs is 2,168 bytes on Darwin");
}

/// At a table's close: its file let go of (a later file may take the same descriptor).
pub fn unregister(fd: c_int) void {
    var i: usize = 0;
    while (i < n_files) {
        if (files[i].fd == fd) {
            files[i] = files[n_files - 1];
            n_files -= 1;
        } else i += 1;
    }
}

fn blockOf(fd: c_int) u64 {
    for (files[0..n_files]) |f| if (f.fd == fd) return f.block;
    return 0;
}

/// Counters since the load (both modes): rows read, read calls (the IOs), the bytes they asked of the files, the
/// model's waits on the prefetch by table (0: the first Engram layer's, 1: the rest); `distinct`: a whole-table read's
/// distinct rows.
pub var rows: std.atomic.Value(u64) = .init(0);
pub var distinct: std.atomic.Value(u64) = .init(0);
pub var calls: std.atomic.Value(u64) = .init(0);
pub var bytes: std.atomic.Value(u64) = .init(0);
pub var wait_ns: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) };

pub const Snapshot = struct { rows: u64, distinct: u64, calls: u64, bytes: u64, wait_ns: [2]u64 };
pub fn snapshot() Snapshot {
    return .{ .rows = rows.load(.monotonic), .distinct = distinct.load(.monotonic), .calls = calls.load(.monotonic), .bytes = bytes.load(.monotonic), .wait_ns = .{ wait_ns[0].load(.monotonic), wait_ns[1].load(.monotonic) } };
}
/// The model's wait on table `table`'s span (the prefetch's `take`).
pub fn waited(table: usize, ns: u64) void {
    _ = wait_ns[@min(table, 1)].fetchAdd(ns, .monotonic);
}

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `dst.len` bytes at `off` of `fd`: through the blocks that cover them when `fd` is registered (one F_NOCACHE pread of
/// whole blocks into an aligned buffer, the bytes copied out), else one pread. False on a short read.
pub fn readRow(fd: c_int, dst: []u8, off: u64) bool {
    const block = blockOf(fd);
    const lo = if (block > 0) off / block * block else off;
    const hi = if (block > 0) std.mem.alignForward(u64, off + dst.len, block) else off + dst.len;
    if (block == 0 or hi - lo > max_span) {
        _ = calls.fetchAdd(1, .monotonic);
        _ = bytes.fetchAdd(dst.len, .monotonic);
        return std.c.pread(fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }
    var buf: [max_span]u8 align(16384) = undefined;
    const n: usize = @intCast(hi - lo);
    const got = std.c.pread(fd, &buf, n, @intCast(lo));
    _ = calls.fetchAdd(1, .monotonic);
    _ = bytes.fetchAdd(n, .monotonic);
    const skip: usize = @intCast(off - lo);
    if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
    @memcpy(dst, buf[skip..][0..dst.len]);
    return true;
}

/// One thread's last read through a file's blocks: the next row whose bytes it covers is copied from it, no read.
pub const Block = struct {
    lo: u64 = std.math.maxInt(u64),
    len: usize = 0,
    buf: [max_span]u8 align(16384) = undefined,
    /// `dst.len` bytes at `off` of `fd`: from the kept block when it covers them, else read through the blocks that
    /// cover them (kept for the next row); one pread when `fd` is not registered or the span is past `max_span`.
    pub fn read(b: *Block, fd: c_int, dst: []u8, off: u64) bool {
        const block = blockOf(fd);
        if (block == 0) return readRow(fd, dst, off);
        if (b.lo != std.math.maxInt(u64) and off >= b.lo and off + dst.len <= b.lo + b.len) {
            const at: usize = @intCast(off - b.lo);
            @memcpy(dst, b.buf[at..][0..dst.len]);
            return true;
        }
        const lo = off / block * block;
        const hi = std.mem.alignForward(u64, off + dst.len, block);
        if (hi - lo > max_span) return readRow(fd, dst, off);
        const n: usize = @intCast(hi - lo);
        b.lo = std.math.maxInt(u64);
        const got = std.c.pread(fd, &b.buf, n, @intCast(lo));
        _ = calls.fetchAdd(1, .monotonic);
        _ = bytes.fetchAdd(n, .monotonic);
        const skip: usize = @intCast(off - lo);
        if (got < 0 or @as(usize, @intCast(got)) < skip + dst.len) return false;
        b.lo = lo;
        b.len = @intCast(got);
        @memcpy(dst, b.buf[skip..][0..dst.len]);
        return true;
    }
};

test "engram_direct setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(try parse("1"));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "on", "2", "01", "true", " 1" }) |v| try std.testing.expectError(error.EngramDirectSetting, parse(v));
}
