//! Directional ablation on the plugin side: the request's direction from `abi.Forward.ablation` armed on the model for
//! one call, and the calibration tap's dump (`SUSHI_DS41_TAP_DIR`).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;
const mlx = @import("sushi/mlx.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");

pub const Refusal = error{ AblationStrength, AblationSlots };

/// Whether the kit this plugin is built against has `Forward.ablation`; without it the plugin serves none.
pub const supported = @hasField(abi.Forward, "ablation");

/// One slot's ablation as the forward hands it.
pub const Spec = struct { directions: abi.Array, strength: f32 };

/// The forward's ablation, or null: a host older than `Forward.ablation` passes a shorter struct, and a strength of 0
/// is no ablation.
pub fn of(f: *const abi.Forward) Refusal!?Spec {
    if (comptime !supported) return null else {
        if (f.size < @offsetOf(abi.Forward, "ablation") + @sizeOf(?*const abi.Ablation)) return null;
        const a = f.ablation orelse return null;
        if (a.strengths.len != 1 or a.strengths.len != f.slots.len) return error.AblationSlots;
        const s = a.strengths.slice()[0];
        if (!(s >= 0 and s <= 1)) return error.AblationStrength;
        return if (s == 0) null else .{ .directions = a.directions, .strength = s };
    }
}

/// Arms `a` on the model for one call: `r` `[dim]`, `col` `[dim, 1]` and `s`, owned here until `disarm`.
pub fn arm(mdl: *dsv.Model, a: Spec, dim: usize, s: mlx.mlx_stream) (error{AblationShape} || anyerror)!void {
    const d: mlx.mlx_array = .{ .ctx = a.directions.ctx };
    if (d.ctx == null) return error.AblationShape;
    const sh = mlx.getShape(d);
    if (sh.len != 2 or sh[0] != 1 or sh[1] != dim or mlx.mlx_array_dtype(d) != .float32) return error.AblationShape;
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, d, &[_]c_int{@intCast(dim)}, 1, s));
    var col = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, d, &[_]c_int{ @intCast(dim), 1 }, 2, s));
    mdl.ablation = .{ .r = r, .col = col, .s = mlx.mlx_array_new_float(a.strength) };
}

/// Drops the call's ablation; the graphs built under it keep their own references.
pub fn disarm(mdl: *dsv.Model) void {
    const ab = mdl.ablation orelse return;
    for ([_]mlx.mlx_array{ ab.r, ab.col, ab.s }) |x| _ = mlx.mlx_array_free(x);
    mdl.ablation = null;
}

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

/// The next tap file number in `dir`: the lines `index.jsonl` already holds, so a restarted server neither overwrites
/// a file nor repeats a number.
pub fn nextTapSeq(gpa: std.mem.Allocator, dir: []const u8) u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = std.fs.path.join(gpa, &.{ dir, "index.jsonl" }) catch return 0;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20)) catch return 0;
    defer gpa.free(text);
    return std.mem.count(u8, text, "\n");
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

test "sushi_ds41 ablation: the npy header is a 64-byte multiple that names the shape" {
    var buf: [256]u8 = undefined;
    const h = try npyHeader(&buf, 44, 5, 5120);
    try std.testing.expectEqual(@as(usize, 0), h.len % 64);
    try std.testing.expect(std.mem.startsWith(u8, h, "\x93NUMPY\x01\x00"));
    try std.testing.expect(std.mem.indexOf(u8, h, "'shape': (44, 5, 5120)") != null);
    try std.testing.expectEqual(@as(u8, '\n'), h[h.len - 1]);
    try std.testing.expectEqual(@as(u16, @intCast(h.len - 10)), std.mem.readInt(u16, h[8..10], .little));
}

test "sushi_ds41 ablation: a host's forward without the field, a null one, or strength 0 is no ablation; bad tables refuse" {
    if (comptime supported) try ofCases() else return error.SkipZigTest;
}

fn ofCases() !void {
    const dir: abi.Array = .{};
    const one = [_]u32{0};
    var f: abi.Forward = .{ .mode = .decode, .rows = 1, .tokens = .{}, .slots = .of(&one) };
    try std.testing.expect((try of(&f)) == null);
    var zero = [_]f32{0};
    var ab: abi.Ablation = .{ .directions = dir, .strengths = .of(&zero) };
    f.ablation = &ab;
    try std.testing.expect((try of(&f)) == null);
    var half = [_]f32{0.5};
    ab.strengths = .of(&half);
    try std.testing.expectEqual(@as(f32, 0.5), (try of(&f)).?.strength);
    var big = [_]f32{1.5};
    ab.strengths = .of(&big);
    try std.testing.expectError(error.AblationStrength, of(&f));
    var two = [_]f32{ 1, 1 };
    ab.strengths = .of(&two);
    try std.testing.expectError(error.AblationSlots, of(&f));
    f.size = @offsetOf(abi.Forward, "ablation");
    try std.testing.expect((try of(&f)) == null);
}
