//! copied_from: src/sdk/weights/nocache_reader.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! An MLX IO reader whose reads bypass the page cache (mlx-stream's, MIT). MLX's own safetensors reader is a plain
//! open + pread, so a load leaves the file's pages cached beside the array buffers: a pack near the RAM size then
//! has its weights compressed under the cache until the kernel kills the process. This one opens the file with
//! F_NOCACHE and read-ahead off; `mlx_load_safetensors_reader` reads the header through `read` and each tensor
//! through `read_at_offset`. Darwin only. A failed read panics naming the file, as MLX's own reader throws.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");

/// One read's staging buffer: page-aligned (the page allocator), a multiple of every page size.
const stage_bytes: usize = 8 << 20;

/// The reader's state, freed by MLX (`free`) once no array it loaded needs it. C allocator: MLX may free it
/// from an IO thread.
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

    /// `buf.len` bytes at `off` through a page-aligned stage: macOS honours F_NOCACHE only for page-aligned
    /// reads (offset, length and destination), and safetensors offsets and MLX buffers are not aligned.
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
pub fn loadSafetensors(tensors: *mlx.mlx_map_string_to_array, meta: *mlx.mlx_map_string_to_string, path: [*:0]const u8, s: mlx.mlx_stream) !void {
    const d = try Desc.open(std.mem.span(path));
    const r = mlx.mlx_io_reader_new(d, vtable);
    defer _ = mlx.mlx_io_reader_free(r);
    try mlx.check(mlx.mlx_load_safetensors_reader(tensors, meta, r, s));
}
