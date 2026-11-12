//! Conformance helpers for `kv`'s lanes (the proposal's KV truncate / rewind equivalence): a row-id backend and the
//! lane-equivalence check every lane of a module-owned decode state must pass. CPU only (no device arrays).

const std = @import("std");
const mlx = @import("sdk").mlx;
const kv = @import("kv.zig");
const testing = std.testing;

/// A lane array's shape on `RowOps`: `[1, rows, 3]` (the feature axis elided), the layout the lanes slice on axis 1.
pub const RowShape = struct {
    n: u8 = 0,
    d: [kv.max_dims]c_int = @splat(0),

    pub fn slice(self: *const RowShape) []const c_int {
        return self.d[0..self.n];
    }

};

/// Arrays of row ids (`[1, n]` rows, the feature axis elided): enough of the
/// op surface for the lanes, so every append / compaction / view / trim is
/// checked against the absolute positions it must hold.
pub const RowOps = struct {
    pub const T = u32;
    gpa: std.mem.Allocator,
    arrays: std.ArrayList(std.ArrayList(i64)) = .empty,
    kept: std.ArrayList(u32) = .empty,
    allocs: u32 = 0,

    pub fn deinit(g: *RowOps) void {
        for (g.arrays.items) |*a| a.deinit(g.gpa);
        g.arrays.deinit(g.gpa);
        g.kept.deinit(g.gpa);
    }

    pub fn rows(g: *RowOps, x: u32) []const i64 {
        return g.arrays.items[x].items;
    }

    pub fn shapeOf(g: *RowOps, x: u32) RowShape {
        return RowShape.of(&.{ 1, @intCast(g.arrays.items[x].items.len), 3 });
    }
    pub fn dtypeOf(_: *RowOps, _: u32) mlx.mlx_dtype {
        return .float32;
    }
    pub fn keep(g: *RowOps, x: u32) u32 {
        g.kept.append(g.gpa, x) catch unreachable;
        return x;
    }
    pub fn release(g: *RowOps, x: u32) void {
        const i = std.mem.indexOfScalar(u32, g.kept.items, x) orelse unreachable;
        _ = g.kept.swapRemove(i);
    }
    pub fn hostArray(g: *RowOps, bytes: []const u8, _: []const c_int, _: mlx.mlx_dtype) !u32 {
        const s = std.mem.bytesAsSlice(i32, bytes);
        return g.new(&.{s[1]});
    }
    pub fn slice(g: *RowOps, x: u32, start: []const c_int, stop: []const c_int, _: []const c_int) !u32 {
        const r = g.rows(x);
        if (stop[1] > r.len) return error.SliceBounds;
        var out: [4096]i64 = undefined;
        const lo: usize = @intCast(start[1]);
        const hi: usize = @intCast(stop[1]);
        @memcpy(out[0 .. hi - lo], r[lo..hi]);
        return g.new(out[0 .. hi - lo]);
    }
    pub fn concat(g: *RowOps, xs: []const u32, _: c_int) !u32 {
        var out: [4096]i64 = undefined;
        var n: usize = 0;
        for (xs) |x| {
            const r = g.rows(x);
            @memcpy(out[n..][0..r.len], r);
            n += r.len;
        }
        return g.new(out[0..n]);
    }
};
