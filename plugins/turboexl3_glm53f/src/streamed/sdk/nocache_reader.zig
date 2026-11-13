//! An MLX IO reader whose reads bypass the page cache. MLX's own safetensors
//! reader is a plain open + pread, so a load leaves the file's pages cached
//! next to the array buffers. This one opens the file with F_NOCACHE and
//! read-ahead off; `mlx_load_safetensors_reader` reads the header through
//! `read` and each tensor through `read_at_offset` into the array's buffer.
//! A failed read panics naming the file, as MLX's own reader throws.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("mlx_host").mlx;
const io_util = @import("io_util.zig");

/// One read's staging buffer: page-aligned (the page allocator), a multiple of every page size.
const stage_bytes: usize = 8 << 20;

/// The reader's state (MLX owns it once handed over; `free` releases it).
/// Allocated with the C allocator: MLX may free it from an IO thread.
pub const Desc = struct {
    fd: std.c.fd_t,
    size: u64,
    /// The sequential position (`read`, `seek`, `tell`: the header parse).
    pos: u64 = 0,
    label: [:0]u8,

    pub fn close(d: *Desc) void {
        _ = std.c.close(d.fd);
        std.heap.c_allocator.free(d.label);
        std.heap.c_allocator.destroy(d);
    }

};

fn of(ctx: ?*anyopaque) *Desc {
    return @ptrCast(@alignCast(ctx.?));
}

fn free(ctx: ?*anyopaque) callconv(.c) void {
    of(ctx).close();
}

// ── Tests (host: no MLX array; the reader's callbacks driven as MLX drives them) ──

const testing = std.testing;

/// A safetensors file's tensor spans, parsed through the reader's `read` / `seek` / `tell` as MLX does.
const Span = struct { off: u64, len: u64 };
