//! The calibration tap's files (`SUSHI_DS41_TAP_DIR`): each prompt's per-layer stream as `tap-<seq>.npy` and a line in
//! `index.jsonl`. Needs MLX but not the plugin kit, so the Metal tests drive it.

const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");

/// The tap's rows for one prompt as `<dir>/tap-<seq>.npy` (f32 `[layers + 1, rows, dim]`, the rows in prompt order,
/// last one the prompt's last token) and a line in `<dir>/index.jsonl`. Clears the tap.
pub fn dumpTap(gpa: std.mem.Allocator, tap: *dsv.StreamTap, dir: []const u8, seq: u64, ids: []const u32, ablated: bool, s: mlx.mlx_stream) !void {
    defer tap.clear();
    const sites = tap.sites.len;
    if (sites == 0 or tap.sites[0].items.len == 0) return;
    var per_site = try gpa.alloc(mlx.mlx_array, sites);
    defer gpa.free(per_site);
    var made: usize = 0;
    defer for (per_site[0..made]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for (tap.sites, 0..) |list, i| {
        const v = mlx.mlx_vector_array_new_data(list.items.ptr, list.items.len);
        defer _ = mlx.mlx_vector_array_free(v);
        per_site[i] = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_concatenate_axis(&per_site[i], v, 0, s)); // [rows, dim]
    }
    const v = mlx.mlx_vector_array_new_data(per_site.ptr, sites);
    defer _ = mlx.mlx_vector_array_free(v);
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_stack_axis(&all, v, 0, s)); // [sites, rows, dim]
    try mlx.check(mlx.mlx_array_eval(all));
    const shape = mlx.getShape(all);
    const data = (mlx.mlx_array_data_float32(all) orelse return error.TapNotF32)[0..mlx.mlx_array_size(all)];

    const io = std.Io.Threaded.global_single_threaded.io();
    const existed = if (std.Io.Dir.cwd().access(io, dir, .{})) true else |_| false;
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
    defer d.close(io);
    // Prompt activations and ids: a directory the tap makes is the owner's only.
    if (!existed) {
        const dir_z = try gpa.dupeSentinel(u8, dir, 0);
        defer gpa.free(dir_z);
        _ = std.c.chmod(dir_z, 0o700);
    }
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "tap-{d:0>6}.npy", .{seq});
    var header_buf: [256]u8 = undefined;
    const npy = try npyHeader(&header_buf, @intCast(shape[0]), @intCast(shape[1]), @intCast(shape[2]));
    const body = std.mem.sliceAsBytes(data);
    const file_bytes = try gpa.alloc(u8, npy.len + body.len);
    defer gpa.free(file_bytes);
    @memcpy(file_bytes[0..npy.len], npy);
    @memcpy(file_bytes[npy.len..], body);
    try d.writeFile(io, .{ .sub_path = name, .data = file_bytes, .flags = .{ .permissions = .fromMode(0o600) } });

    var line: std.Io.Writer.Allocating = .init(gpa);
    defer line.deinit();
    try line.writer.print("{{\"seq\":{d},\"file\":\"{s}\",\"ablated\":{},\"prompt_tokens\":{d},\"rows\":{d},\"tail_ids\":[", .{ seq, name, ablated, ids.len, shape[1] });
    const tail = ids[ids.len - @min(ids.len, 8) ..];
    for (tail, 0..) |t, i| try line.writer.print("{s}{d}", .{ if (i == 0) "" else ",", t });
    try line.writer.writeAll("]}\n");
    var idx = d.openFile(io, "index.jsonl", .{ .mode = .write_only }) catch |e| switch (e) {
        error.FileNotFound => try d.createFile(io, "index.jsonl", .{ .permissions = .fromMode(0o600) }),
        else => return e,
    };
    defer idx.close(io);
    const end = (try idx.stat(io)).size;
    try idx.writePositionalAll(io, line.written(), end);
}

/// The next tap file number in `dir`: one past the highest number any `tap-<seq>.npy` file or `index.jsonl` line holds,
/// so a restarted server never reuses a number, even after a dump that wrote its file but not its index line.
pub fn nextTapSeq(gpa: std.mem.Allocator, dir_path: []const u8) u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var next: u64 = 0;
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.startsWith(u8, e.name, "tap-") or !std.mem.endsWith(u8, e.name, ".npy")) continue;
        const n = std.fmt.parseInt(u64, e.name[4 .. e.name.len - 4], 10) catch continue;
        next = @max(next, n + 1);
    }
    const text = dir.readFileAlloc(io, "index.jsonl", gpa, .limited(256 << 20)) catch return next;
    defer gpa.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const at = std.mem.indexOf(u8, line, "\"seq\":") orelse continue;
        var end = at + 6;
        while (end < line.len and std.ascii.isDigit(line[end])) end += 1;
        const n = std.fmt.parseInt(u64, line[at + 6 .. end], 10) catch continue;
        next = @max(next, n + 1);
    }
    return next;
}

/// A version 1.0 `.npy` header for little-endian f32 `[a, b, c]`, padded to 64 bytes.
fn npyHeader(buf: []u8, a: usize, b: usize, c: usize) ![]const u8 {
    const dict = try std.fmt.bufPrint(buf[10..], "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}, {d}), }}", .{ a, b, c });
    var total = 10 + dict.len + 1;
    const pad = (64 - total % 64) % 64;
    if (total + pad > buf.len) return error.NoSpaceLeft;
    @memset(buf[10 + dict.len ..][0..pad], ' ');
    total += pad;
    buf[total - 1] = '\n';
    @memcpy(buf[0..6], "\x93NUMPY");
    buf[6] = 1;
    buf[7] = 0;
    std.mem.writeInt(u16, buf[8..10], @intCast(total - 10), .little);
    return buf[0..total];
}

test "sushi_ds41 tap: the npy header is a 64-byte multiple that names the shape" {
    var buf: [256]u8 = undefined;
    const h = try npyHeader(&buf, 44, 5, 5120);
    try std.testing.expectEqual(@as(usize, 0), h.len % 64);
    try std.testing.expect(std.mem.startsWith(u8, h, "\x93NUMPY\x01\x00"));
    try std.testing.expect(std.mem.indexOf(u8, h, "'shape': (44, 5, 5120)") != null);
    try std.testing.expectEqual(@as(u8, '\n'), h[h.len - 1]);
    try std.testing.expectEqual(@as(u16, @intCast(h.len - 10)), std.mem.readInt(u16, h[8..10], .little));
}


test "sushi_ds41 tap: the next number is past every file and index line, whichever is higher" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    try std.testing.expectEqual(@as(u64, 0), nextTapSeq(std.testing.allocator, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "index.jsonl", .data = "{\"seq\":0,\"ablated\":false}\n{\"seq\":7,\"ablated\":true}\n" });
    try std.testing.expectEqual(@as(u64, 8), nextTapSeq(std.testing.allocator, root));
    // A dump that wrote its file but not its line.
    try tmp.dir.writeFile(io, .{ .sub_path = "tap-000011.npy", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "" });
    try std.testing.expectEqual(@as(u64, 12), nextTapSeq(std.testing.allocator, root));
    try std.testing.expectEqual(@as(u64, 0), nextTapSeq(std.testing.allocator, "/nonexistent/tap"));
}
