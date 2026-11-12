//! `TURBOEXL3_GLM53F_LOGITS_DUMP=<absolute directory>` (a check, never served; off by default): each forward of the main
//! model writes its rows' top `k` log-probabilities to `<dir>/logits-<pid>.bin`, for the int8 prefill's KLD run (two
//! servers' files, read offline). plugin.zig's `forward` asks the copy for the final-normed hidden (`want_hidden`: one
//! more reference, no op changed), and once the forward is done (it has evaluated that hidden, every slot row it read
//! behind it) projects the hidden through the LM head itself, `block_rows` rows at a time, takes each row's log-softmax in
//! f32 and its top `k`, and hands `write` one record. The logits and hidden the engine asked for are the forward's own,
//! so the served bytes are unchanged; the cost is the LM head again over every row and a wait a forward.
//!
//! A record, little-endian: u32 `magic`, u32 mode (`abi.Mode`), u32 slot, u32 first position, u32 rows, u32 k, then the
//! rows' input ids (u32; `no_id` where the forward named none), their top-k token ids (u32, each row's best first) and
//! their top-k log-probabilities (f32). MLX-free: the switch, the file and the record.

const std = @import("std");

/// "LGD1"
pub const magic: u32 = 0x3144474C;
/// Log-probabilities kept a row.
pub const k: u32 = 64;
/// Rows the LM head runs at a time (its f32 logits: `block_rows` x vocab x 4 bytes, a few times over in flight).
pub const block_rows: u32 = 256;
/// A row whose input id the forward did not name.
pub const no_id: u32 = 0xFFFF_FFFF;

/// Set by the plugin at load when the switch names a directory.
pub var on: bool = false;
var fd: c_int = -1;
/// Records and rows written since the load; forwards not recorded (more than one slot).
pub var records: u64 = 0;
pub var rows: u64 = 0;
pub var skipped: u64 = 0;

/// The switch's value, or null when unset or empty.
pub fn requested(env: *const fn ([:0]const u8) ?[]const u8) ?[]const u8 {
    const v = env("TURBOEXL3_GLM53F_LOGITS_DUMP") orelse return null;
    return if (v.len == 0) null else v;
}

/// What a forward's dump holds in MLX at once, bytes: a block's logits in f32, its log-softmax, its negation, the
/// partition's indices and the head's own output, for a vocabulary of `vocab` (the prefill bill adds it while on).
pub fn transientBytes(vocab: u64) u64 {
    return @as(u64, block_rows) * vocab * 4 * 5;
}

/// Opens `<dir>/logits-<pid>.bin` (created, or emptied): `dir` absolute, at most 900 bytes, an existing directory.
pub fn open(dir: []const u8, path_buf: *[1024]u8) ![:0]const u8 {
    if (!std.fs.path.isAbsolute(dir) or dir.len > 900) return error.Glm53fLogitsDumpPath;
    const path = std.fmt.bufPrintSentinel(path_buf, "{s}/logits-{d}.bin", .{ dir, std.c.getpid() }, 0) catch return error.Glm53fLogitsDumpPath;
    const f = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (f < 0) return error.Glm53fLogitsDumpOpen;
    if (fd >= 0) _ = std.c.close(fd);
    fd = f;
    on = true;
    records = 0;
    rows = 0;
    skipped = 0;
    return path;
}

/// Closes the file (the switch off).
pub fn close() void {
    if (fd >= 0) _ = std.c.close(fd);
    fd = -1;
    on = false;
}

fn writeAll(bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const r = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (r <= 0) return error.Glm53fLogitsDumpWrite;
        done += @intCast(r);
    }
}

/// One forward's record: `ids` its rows' input ids, `top_ids` and `top_lp` their top-k (rows x k each, row by row).
pub fn write(mode: u32, slot: u32, pos0: u32, ids: []const u32, top_ids: []const u32, top_lp: []const f32) !void {
    if (fd < 0) return error.Glm53fLogitsDumpClosed;
    const n: u32 = @intCast(ids.len);
    if (top_ids.len != @as(usize, n) * k or top_lp.len != @as(usize, n) * k) return error.Glm53fLogitsDumpShape;
    const head = [6]u32{ magic, mode, slot, pos0, n, k };
    try writeAll(std.mem.sliceAsBytes(&head));
    try writeAll(std.mem.sliceAsBytes(ids));
    try writeAll(std.mem.sliceAsBytes(top_ids));
    try writeAll(std.mem.sliceAsBytes(top_lp));
    records += 1;
    rows += n;
}

/// The load line: `{s}` the directory, then the file.
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_LOGITS_DUMP={s}: each main-model forward's rows' top-64 log-probabilities written to {s} (a check, never served: the LM head again over every row, a wait a forward)\n";

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_LOGITS_DUMP's records read back as written (header, input ids, top-k ids and log-probabilities, row by row); a relative or missing directory is refused" {
    const t = std.testing;
    var pb: [1024]u8 = undefined;
    try t.expectError(error.Glm53fLogitsDumpPath, open("relative/dir", &pb));
    try t.expectError(error.Glm53fLogitsDumpOpen, open("/nonexistent-turboexl3-glm53f-dir", &pb));
    try t.expect(!on);
    const path = try open("/tmp", &pb);
    defer {
        close();
        _ = std.c.unlink(path);
    }
    try t.expect(on);
    var ids = [_]u32{ 7, no_id };
    var top_ids: [2 * k]u32 = undefined;
    var top_lp: [2 * k]f32 = undefined;
    for (&top_ids, &top_lp, 0..) |*i, *l, j| {
        i.* = @intCast(1000 + j);
        l.* = -@as(f32, @floatFromInt(j)) / 8.0;
    }
    try write(0, 3, 40, &ids, &top_ids, &top_lp);
    try write(1, 3, 42, ids[0..1], top_ids[0..k], top_lp[0..k]);
    try t.expectError(error.Glm53fLogitsDumpShape, write(1, 3, 43, ids[0..1], top_ids[0..1], top_lp[0..k]));
    try t.expect(records == 2 and rows == 3);
    // read back
    const r = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try t.expect(r >= 0);
    defer _ = std.c.close(r);
    var buf: [2 * (6 + 2 + 2 * 2 * k) * 4]u8 align(4) = undefined;
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.read(r, buf[got..].ptr, buf.len - got);
        if (n <= 0) break;
        got += @intCast(n);
    }
    const want_len = (6 + 2 + 2 * 2 * k) * 4 + (6 + 1 + 2 * k) * 4;
    try t.expectEqual(@as(usize, want_len), got);
    const w = std.mem.bytesAsSlice(u32, buf[0..got]);
    try t.expectEqualSlices(u32, &.{ magic, 0, 3, 40, 2, k, 7, no_id }, w[0..8]);
    try t.expectEqualSlices(u32, &top_ids, w[8 .. 8 + 2 * k]);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&top_lp), std.mem.sliceAsBytes(w[8 + 2 * k .. 8 + 4 * k]));
    const second = 8 + 4 * k;
    try t.expectEqualSlices(u32, &.{ magic, 1, 3, 42, 1, k, 7 }, w[second .. second + 7]);
    try t.expectEqualSlices(u32, top_ids[0..k], w[second + 7 .. second + 7 + k]);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(top_lp[0..k]), std.mem.sliceAsBytes(w[second + 7 + k .. second + 7 + 2 * k]));
    try t.expect(transientBytes(154880) == 256 * 154880 * 20);
}
