//! `MLX_SERVE_DSV41_FLASH_ATTN`'s MLX-free parts (src/flash_attn.zig is the switch's MLX side): the settings, the
//! kernel's text and its heads-region twin, and `check`'s stats over the picks and the probabilities. Std only, tested by
//! `zig build test`.

const std = @import("std");
const twins = @import("attn_twins.zig");

pub const Mode = enum { off, on, check };

pub const Error = error{
    /// MLX_SERVE_DSV41_FLASH_ATTN is none of 0, 1, check (unset or empty: on).
    FlashAttnSetting,
    /// MLX_SERVE_DSV41_FLASH_DUMP is not an absolute directory path under 900 bytes.
    FlashDumpSetting,
};

/// On unless the setting says otherwise (David 2026-11-03: flash attention judged on speed and quality, not words).
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.FlashAttnSetting;
}

/// Whether the setting names on itself (`1`), not by default: a check that runs the gather's ring beside it refuses an
/// explicit 1 and turns the default off.
pub fn explicitOn(value: ?[]const u8) bool {
    const v = value orelse return false;
    return std.mem.eql(u8, v, "1");
}

pub fn parseDump(value: ?[]const u8) Error!?[]const u8 {
    const v = value orelse return null;
    if (v.len == 0) return null;
    if (v.len >= 900 or !std.fs.path.isAbsolute(v)) return error.FlashDumpSetting;
    return v;
}

/// Groups of 16 heads a threadgroup (the fa1u microbench's choice: fa4).
pub const hgn: c_int = 4;

/// The kernel's text; its heads-region twin replaces the grouped store's address with the run region's row layout
/// (`[rows, og, in]`, the span's first row `soff[1]`), as SCRATCH_HEADS' pv twin does.
pub const source = @embedFile("kernels/attn_flash.metal");
pub const region_store = twins.Sub{
    .old = "  return o + (size_t(h / hgr) * L + l) * size_t(hgr * D) + size_t(h % hgr) * D + d;\n",
    .new = "  return o + (size_t(soff[1] + l) * G + size_t(h / hgr)) * size_t(hgr * D) + size_t(h % hgr) * D + d;\n",
};

/// The heads-region twin's text (the caller frees it): the region bound whole as `o_dst`, a carrier word, `dep`, `soff`
/// and `dep2` added (src/attn_twins.zig).
pub fn regionTwin(a: std.mem.Allocator, header: []const u8) ![:0]u8 {
    return twins.deriveWith(a, source, header, &.{.{ .name = "o", .mtype = "OT", .off = null }}, &.{}, &.{region_store}, &.{"dep2"});
}

/// Neighbouring rows' picks: the mean share of a row's live picks its next row also picks, and the mean union of
/// `g`-row groups' live picks in units of `k` (1.0: the group shares one set).
pub const PickStats = struct { rows: usize = 0, neigh: f64 = 0, union4: f64 = 0, union8: f64 = 0, union16: f64 = 0 };

/// `picks` [rows, k] ascending ids, -1 for none (the top-k's order).
pub fn pickStats(gpa: std.mem.Allocator, picks: []const i32, rows: usize, k: usize) !PickStats {
    var st: PickStats = .{ .rows = rows };
    if (rows < 2 or k == 0) return st;
    var shared: f64 = 0;
    var pairs: usize = 0;
    for (0..rows - 1) |l| {
        const a = picks[l * k ..][0..k];
        const b = picks[(l + 1) * k ..][0..k];
        const na = liveLen(a);
        if (na == 0) continue;
        shared += @as(f64, @floatFromInt(common(a[0..na], b[0..liveLen(b)]))) / @as(f64, @floatFromInt(na));
        pairs += 1;
    }
    st.neigh = if (pairs > 0) shared / @as(f64, @floatFromInt(pairs)) else 0;
    const buf = try gpa.alloc(i32, 16 * k);
    defer gpa.free(buf);
    inline for (.{ 4, 8, 16 }, 0..) |g, gi| {
        var sum: f64 = 0;
        var groups: usize = 0;
        var l0: usize = 0;
        while (l0 + g <= rows) : (l0 += g) {
            var n: usize = 0;
            for (0..g) |i| {
                const row = picks[(l0 + i) * k ..][0..k];
                const nl = liveLen(row);
                @memcpy(buf[n..][0..nl], row[0..nl]);
                n += nl;
            }
            if (n == 0) continue;
            std.mem.sort(i32, buf[0..n], {}, std.sort.asc(i32));
            var u: usize = 1;
            for (1..n) |i| u += @intFromBool(buf[i] != buf[i - 1]);
            sum += @as(f64, @floatFromInt(u)) / @as(f64, @floatFromInt(k));
            groups += 1;
        }
        const v = if (groups > 0) sum / @as(f64, @floatFromInt(groups)) else 0;
        switch (gi) {
            0 => st.union4 = v,
            1 => st.union8 = v,
            else => st.union16 = v,
        }
    }
    return st;
}

/// The live prefix of an ascending pick row (the top-k pads with -1 past a row's reach).
fn liveLen(row: []const i32) usize {
    var n: usize = 0;
    while (n < row.len and row[n] >= 0) n += 1;
    return n;
}

/// How many ids two ascending lists share.
fn common(a: []const i32, b: []const i32) usize {
    var i: usize = 0;
    var j: usize = 0;
    var c: usize = 0;
    while (i < a.len and j < b.len) {
        if (a[i] == b[j]) {
            c += 1;
            i += 1;
            j += 1;
        } else if (a[i] < b[j]) i += 1 else j += 1;
    }
    return c;
}

/// The probabilities' blocks no head needs, for the descending-order skip question: of each (row, 128-key block), the
/// share whose largest probability over every head is below 2^-11 (`all11`) and below 2^-8 (`all8`), and of each
/// (row, head, 128-key block) the share below 2^-11 (`head11`); blocks with no live key are not counted.
pub const MassStats = struct { blocks: usize = 0, all11: f64 = 0, all8: f64 = 0, head11: f64 = 0 };

/// `p` [rows, heads, nk] (the gather's normalised probabilities, f32 here).
pub fn massStats(p: []const f32, rows: usize, heads: usize, nk: usize) MassStats {
    var st: MassStats = .{};
    const t11: f32 = 1.0 / 2048.0;
    const t8: f32 = 1.0 / 256.0;
    var n_all11: usize = 0;
    var n_all8: usize = 0;
    var n_head11: usize = 0;
    var n_head: usize = 0;
    for (0..rows) |l| {
        var c0: usize = 0;
        while (c0 < nk) : (c0 += 128) {
            const c1 = @min(nk, c0 + 128);
            var mx_all: f32 = 0;
            var any = false;
            for (0..heads) |h| {
                const row = p[(l * heads + h) * nk ..][c0..c1];
                var mx: f32 = 0;
                for (row) |v| mx = @max(mx, v);
                if (mx > 0) any = true;
                mx_all = @max(mx_all, mx);
                n_head += 1;
                n_head11 += @intFromBool(mx < t11);
            }
            if (!any) continue;
            st.blocks += 1;
            n_all11 += @intFromBool(mx_all < t11);
            n_all8 += @intFromBool(mx_all < t8);
        }
    }
    const nb: f64 = @floatFromInt(@max(st.blocks, 1));
    st.all11 = @as(f64, @floatFromInt(n_all11)) / nb;
    st.all8 = @as(f64, @floatFromInt(n_all8)) / nb;
    st.head11 = @as(f64, @floatFromInt(n_head11)) / @as(f64, @floatFromInt(@max(n_head, 1)));
    return st;
}

// ── tests ──

const testing = std.testing;

test "flash_attn setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try testing.expectEqual(Mode.on, try parse(null));
    try testing.expectEqual(Mode.on, try parse(""));
    try testing.expectEqual(Mode.on, try parse("1"));
    try testing.expectEqual(Mode.off, try parse("0"));
    try testing.expectEqual(Mode.check, try parse("check"));
    try testing.expect(explicitOn("1"));
    for ([_]?[]const u8{ null, "", "0", "check" }) |v| try testing.expect(!explicitOn(v));
    for ([_][]const u8{ "on", "2", "true", "Check", " 1" }) |v| try testing.expectError(error.FlashAttnSetting, parse(v));
    try testing.expectEqual(@as(?[]const u8, null), try parseDump(null));
    try testing.expectError(error.FlashDumpSetting, parseDump("relative/dir"));
    try testing.expectEqualStrings("/tmp/x", (try parseDump("/tmp/x")).?);
}

test "flash_attn region twin: the grouped store's address moved to the run region's rows, the rest of the text kept" {
    const t = try regionTwin(testing.allocator, "");
    defer testing.allocator.free(t);
    try testing.expect(std.mem.indexOf(u8, t, region_store.new) != null);
    try testing.expect(std.mem.indexOf(u8, t, region_store.old) == null);
    try testing.expect(std.mem.startsWith(u8, t, "device OT* o = (device OT*) o_dst;\n"));
}

test "flash_attn pick stats: neighbour overlap and group unions over ascending, -1-padded rows" {
    // rows: {0,1,2,3}, {1,2,3,4}, {2,3,4,-1}, {2,3,4,5}
    const picks = [_]i32{ 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, -1, 2, 3, 4, 5 };
    const st = try pickStats(testing.allocator, &picks, 4, 4);
    // 3/4, 3/4, 3/3 -> mean 0.8333
    try testing.expectApproxEqAbs(@as(f64, (0.75 + 0.75 + 1.0) / 3.0), st.neigh, 1e-9);
    // one 4-row group: union {0..5} = 6 ids = 1.5 k
    try testing.expectApproxEqAbs(@as(f64, 1.5), st.union4, 1e-9);
}

test "flash_attn mass stats: blocks no head needs" {
    // 1 row, 2 heads, 256 keys: block 0 has 0.5 on head 0; block 1 at most 2^-12 on both heads
    var p: [2 * 256]f32 = @splat(0);
    p[0] = 0.5;
    p[200] = 1.0 / 4096.0;
    p[256 + 130] = 1.0 / 4096.0;
    const st = massStats(&p, 1, 2, 256);
    try testing.expectEqual(@as(usize, 2), st.blocks);
    try testing.expectApproxEqAbs(@as(f64, 0.5), st.all11, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.5), st.all8, 1e-9);
    // (row, head, block): head 0 block 0 = 0.5 (no), head 1 block 0 = 0 (yes), head 0 block 1 (yes), head 1 block 1 (yes)
    try testing.expectApproxEqAbs(@as(f64, 0.75), st.head11, 1e-9);
}
