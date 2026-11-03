//! `MLX_SERVE_DSV41_BILL_PIN` (read at load; unset or empty: off; a file path: pinned): the streamed bank's slot rows
//! come from the bill's budget (`streamBudget`: MLX's recommended working set less the wired margin, the host's bytes,
//! the Engram LRU's and the memory in use), and the memory in use is every process's, read live at the bank's open
//! (hw.memsize less the available pages), so what the other processes hold that minute moves the rows: two servers of
//! one window can get rows apart. Pinned, the file holds the memory in use, in bytes, on one decimal line: a valid one is
//! used in place of the live reading at every one of the bill's reads; with no file, the bank's open takes the live
//! reading, writes it (a temporary beside the file, then renamed over it) and uses it, so the servers after it read the
//! same value. A file whose content is no such value refuses the load by name. Exact: only the input the row count comes
//! from is fixed; the rows, the plans and the words follow from it as from any reading.
//!
//! The load line names the mode and the value (`[turboexl3_ds41] the streamed bank's bill: pinned= in_use_mb= file=`);
//! the `[bank-bill]` line's `pinned=` says what the bank's open did (`off`, `read`, `wrote`, `unwritten`: the write
//! failed and the live reading was used) beside its `in_use_mb=`, which is the pinned value when pinned (off: the live
//! reading at that line, after the bank's rows were allocated, as before).

const std = @import("std");

pub const Error = error{ BillPinSetting, BillPinFile };

pub const Mode = enum { off, pending, read, wrote, unwritten };

/// The pin's state: `off` (unset), `pending` (set, no file yet: the bank's open writes it), `read` (the file's value),
/// `wrote` (the bank's open wrote its live reading), `unwritten` (that write failed: the live reading used).
pub var mode: Mode = .off;
var file_buf: [1024:0]u8 = undefined;
var file: ?[:0]const u8 = null;
var value: ?u64 = null;

/// The largest value a pin file may hold (1 TiB): a reading past it is no memory reading.
const max_bytes: u64 = 1 << 40;

pub fn configure(raw: ?[]const u8) Error!void {
    mode = .off;
    file = null;
    value = null;
    const v = raw orelse return;
    if (v.len == 0) return;
    if (v.len >= file_buf.len) return error.BillPinSetting;
    @memcpy(file_buf[0..v.len], v);
    file_buf[v.len] = 0;
    file = file_buf[0..v.len :0];
    value = try readPin(file.?);
    mode = if (value != null) .read else .pending;
}

pub fn on() bool {
    return mode != .off;
}

/// `bill.in_use_pin`: the value the bill reads in place of `live`; the first call with no file writes `live` and pins it.
pub fn pin(live: u64) u64 {
    if (value) |v| return v;
    value = live;
    mode = if (writePin(file.?, live)) .wrote else .unwritten;
    return live;
}

/// `bill.in_use_pinned`: the `[bank-bill]` line's `pinned=`.
pub fn modeName() []const u8 {
    return @tagName(mode);
}

/// The pin file's value: null when there is no file; a file holding no value (one decimal line, 0 < bytes < 1 TiB)
/// is an error.
fn readPin(p: [:0]const u8) Error!?u64 {
    if (std.c.access(p.ptr, std.c.F_OK) != 0) return null;
    const fd = std.c.open(p.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.BillPinFile;
    defer _ = std.c.close(fd);
    var buf: [64]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return error.BillPinFile;
    const text = std.mem.trim(u8, buf[0..@intCast(n)], " \t\r\n");
    const bytes = std.fmt.parseInt(u64, text, 10) catch return error.BillPinFile;
    if (bytes == 0 or bytes >= max_bytes) return error.BillPinFile;
    return bytes;
}

/// `bytes` as one decimal line into `p`: a temporary beside it (`<p>.tmp-<pid>`), then renamed over it. False when any
/// step fails (the temporary removed).
fn writePin(p: [:0]const u8, bytes: u64) bool {
    var tmp_buf: [1100]u8 = undefined;
    const tmp = zPrint(&tmp_buf, "{s}.tmp-{d}", .{ p, std.c.getpid() }) catch return false;
    var line_buf: [32]u8 = undefined;
    const line = std.fmt.bufPrint(&line_buf, "{d}\n", .{bytes}) catch return false;
    const fd = std.c.open(tmp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return false;
    var done: usize = 0;
    while (done < line.len) {
        const r = std.c.write(fd, line[done..].ptr, line.len - done);
        if (r <= 0) {
            _ = std.c.close(fd);
            _ = std.c.unlink(tmp.ptr);
            return false;
        }
        done += @intCast(r);
    }
    if (std.c.close(fd) != 0 or std.c.rename(tmp.ptr, p.ptr) != 0) {
        _ = std.c.unlink(tmp.ptr);
        return false;
    }
    return true;
}

/// `fmt` into `buf` with a 0 after it.
fn zPrint(buf: []u8, comptime fmt: []const u8, args: anytype) error{NoSpaceLeft}![:0]const u8 {
    const s = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

/// The load line (after the bank's open: `read`, `wrote` or `unwritten` when pinned; `pending` when no bank opened).
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    const head = "[turboexl3_ds41] the streamed bank's bill: pinned=";
    (switch (mode) {
        .off => w.print(head ++ "off (MLX_SERVE_DSV41_BILL_PIN unset: the memory in use read live at the bank's open)\n", .{}),
        .pending => w.print(head ++ "pending file={s} (MLX_SERVE_DSV41_BILL_PIN: no value yet; the bank's open reads the memory in use live, writes it there and uses it)\n", .{file.?}),
        .read => w.print(head ++ "read in_use_mb={d} file={s} (MLX_SERVE_DSV41_BILL_PIN: the memory in use the file's, not this minute's)\n", .{ value.? >> 20, file.? }),
        .wrote => w.print(head ++ "wrote in_use_mb={d} file={s} (MLX_SERVE_DSV41_BILL_PIN: read live at the bank's open and written there, for the servers after it)\n", .{ value.? >> 20, file.? }),
        .unwritten => w.print(head ++ "unwritten in_use_mb={d} file={s} (MLX_SERVE_DSV41_BILL_PIN: read live at the bank's open and used; the write failed, so the next server reads live too)\n", .{ value.? >> 20, file.? }),
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

fn testPath(buf: []u8, tag: []const u8) [:0]const u8 {
    return zPrint(buf, "/tmp/turboexl3_ds41-billpin-{s}-{d}.txt", .{ tag, std.c.getpid() }) catch unreachable;
}

test "bill pin: unset or empty off; a missing file pending, the bank's first read writes it and every later one reads it" {
    defer {
        mode = .off;
        file = null;
        value = null;
    }
    try configure(null);
    try testing.expectEqual(Mode.off, mode);
    try configure("");
    try testing.expectEqual(Mode.off, mode);
    try testing.expect(!on());
    var lb: [512]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, loadLine(&lb), "pinned=off") != null);

    var pb: [128]u8 = undefined;
    const p = testPath(&pb, "w");
    _ = std.c.unlink(p.ptr);
    defer _ = std.c.unlink(p.ptr);
    try configure(p);
    try testing.expectEqual(Mode.pending, mode);
    try testing.expect(on());
    try testing.expectEqual(@as(u64, 101_930_000_000), pin(101_930_000_000));
    try testing.expectEqual(Mode.wrote, mode);
    try testing.expectEqualStrings("wrote", modeName());
    // a later reading of the same load keeps the first
    try testing.expectEqual(@as(u64, 101_930_000_000), pin(95_000_000_000));

    // the next server reads it whatever its live reading
    try configure(p);
    try testing.expectEqual(Mode.read, mode);
    try testing.expectEqual(@as(u64, 101_930_000_000), pin(97_000_000_000));
    try testing.expectEqualStrings("read", modeName());
    var eb: [64]u8 = undefined;
    const want = try std.fmt.bufPrint(&eb, "pinned=read in_use_mb={d} ", .{@as(u64, 101_930_000_000) >> 20});
    try testing.expect(std.mem.indexOf(u8, loadLine(&lb), want) != null);
}

test "bill pin: a file holding no value refuses the load; a name too long refuses it" {
    defer {
        mode = .off;
        file = null;
        value = null;
    }
    var pb: [128]u8 = undefined;
    const p = testPath(&pb, "bad");
    defer _ = std.c.unlink(p.ptr);
    for ([_][]const u8{ "", "abc\n", "0\n", "-5\n", "2000000000000\n", "12 34\n" }) |content| {
        const fd = std.c.open(p.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
        try testing.expect(fd >= 0);
        if (content.len > 0) try testing.expect(std.c.write(fd, content.ptr, content.len) == @as(isize, @intCast(content.len)));
        _ = std.c.close(fd);
        try testing.expectError(error.BillPinFile, configure(p));
    }
    const long: [1100]u8 = @splat('a');
    try testing.expectError(error.BillPinSetting, configure(&long));
}

test "bill pin: a write that cannot land leaves the live reading used, unwritten" {
    defer {
        mode = .off;
        file = null;
        value = null;
    }
    try configure("/tmp/turboexl3_ds41-billpin-no-such-dir/pin.txt");
    try testing.expectEqual(Mode.pending, mode);
    try testing.expectEqual(@as(u64, 33_000_000_000), pin(33_000_000_000));
    try testing.expectEqual(Mode.unwritten, mode);
    try testing.expectEqual(@as(u64, 33_000_000_000), pin(34_000_000_000));
}
