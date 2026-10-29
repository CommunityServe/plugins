//! This plugin's own, in place of mlx-serve's `src/mlx/device.zig` (a cut edge of scripts/sync-mlx-serve.sh): the GPU's
//! generation and MLX's NAX gate (an M5-class GPU, G17 or later, G18 for a phone part, on macOS 26.2 or later), which
//! the copies' kernel choices read. No fallback lever: the gate is the machine's.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("ffi.zig");

pub const DeviceGen = struct { gen: u32, phone: bool };

/// A test seam of the copies; never set here.
pub var device_gen_override: ?DeviceGen = null;

/// `applegpu_g17s` / `g18p` into its generation and whether it is a phone part; 0 for anything else.
pub fn naxArchGeneration(arch: []const u8) DeviceGen {
    const none: DeviceGen = .{ .gen = 0, .phone = false };
    var buf: [128]u8 = undefined;
    if (arch.len < 3) return none;
    const a = std.ascii.lowerString(buf[0..@min(arch.len, buf.len)], arch[0..@min(arch.len, buf.len)]);
    const tail = if (std.mem.startsWith(u8, a, "applegpu_g")) a[10..] else if (a[0] == 'g') a[1..] else return none;
    const digits = std.mem.indexOfNone(u8, tail, "0123456789") orelse tail.len;
    if (digits == 0 or tail.len - digits > 1) return none;
    if (digits < tail.len and !std.ascii.isAlphabetic(tail[digits])) return none;
    return .{ .gen = std.fmt.parseInt(u32, tail[0..digits], 10) catch 0, .phone = digits < tail.len and tail[digits] == 'p' };
}

var gen_cache: ?DeviceGen = null;

pub fn deviceGeneration() DeviceGen {
    if (device_gen_override) |g| return g;
    if (gen_cache == null) {
        var buf: [128]u8 = undefined;
        gen_cache = naxArchGeneration(gpuArchitecture(&buf) orelse "");
    }
    return gen_cache.?;
}

/// `ver` (`26.4.1`) at least `major.minor`.
pub fn macosVersionAtLeast(ver: []const u8, major: u32, minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const ma = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const mi = if (it.next()) |p| std.fmt.parseInt(u32, p, 10) catch 0 else 0;
    return ma > major or (ma == major and mi >= minor);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

pub fn macosProductVersion(buf: []u8) ?[]const u8 {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    var len = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return null;
    const v = std.mem.trimEnd(u8, buf[0..@min(len, buf.len)], "\x00");
    return if (v.len == 0) null else v;
}

/// The default device's architecture name (MLX's `architecture` device info).
pub fn gpuArchitecture(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device_new();
    defer _ = mlx.mlx_device_free(dev);
    if (mlx.c.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var name: [*c]const u8 = null;
    if (mlx.c.mlx_device_info_get_string(&name, info, "architecture") != 0 or name == null) return null;
    const arch = std.mem.span(name);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

pub fn computeNaxAvailable() bool {
    var arch_buf: [128]u8 = undefined;
    const g = naxArchGeneration(gpuArchitecture(&arch_buf) orelse "");
    if (g.gen < @as(u32, if (g.phone) 18 else 17)) return false;
    var ver_buf: [64]u8 = undefined;
    return macosVersionAtLeast(macosProductVersion(&ver_buf) orelse "", 26, 2);
}

var nax_cache: ?bool = null;

pub fn naxAvailable() bool {
    if (nax_cache == null) nax_cache = computeNaxAvailable();
    return nax_cache.?;
}

test "the NAX gate's pure parts: G17 (G18 for a phone part), macOS 26.2" {
    const t = std.testing;
    try t.expectEqual(DeviceGen{ .gen = 17, .phone = false }, naxArchGeneration("AppleGPU_G17S"));
    try t.expectEqual(DeviceGen{ .gen = 18, .phone = true }, naxArchGeneration("g18p"));
    for ([_][]const u8{ "air64_v27", "metal_27", "foo17", "gfoobar27", "g17dd", "applegpu_g", "applegpu_g17_extra", "" }) |a|
        try t.expectEqual(@as(u32, 0), naxArchGeneration(a).gen);
    try t.expect(macosVersionAtLeast("26.2", 26, 2) and macosVersionAtLeast("26.4.1", 26, 2) and macosVersionAtLeast("27", 26, 2));
    try t.expect(!macosVersionAtLeast("26.1", 26, 2) and !macosVersionAtLeast("26", 26, 2) and !macosVersionAtLeast("26.x", 26, 2));
}
