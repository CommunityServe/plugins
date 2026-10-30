//! The server's own launch flags this plugin follows, read off the process's arguments at load (`_NSGetArgv`): no load
//! setting carries them. `known` has one line per flag, and each value parses by the server's own rules, so the plugin
//! takes exactly what the server takes. The server exits on a value it refuses; one that still reaches the plugin
//! refuses the load by name.

const std = @import("std");
const depth_bounds = @import("mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const mtp_mod = @import("mlx_serve/sdk/spec/mtp/mtp.zig");
const log = @import("mlx_serve/sdk/util/log.zig");

/// What the flags say: null where the process names none.
pub const Flags = struct {
    mtp_min_depth: ?u32 = null,
    mtp_max_depth: ?u32 = null,

    /// The draft-depth range as the server's configure sets it from `--mtp-min-depth` and `--mtp-max-depth`, checked
    /// as its main checks it; null when neither is named.
    pub fn depthBounds(f: Flags) error{MinAboveMax}!?depth_bounds.Bounds {
        if (f.mtp_min_depth == null and f.mtp_max_depth == null) return null;
        const b: depth_bounds.Bounds = .{ .min = f.mtp_min_depth orelse 1, .max = f.mtp_max_depth orelse 0 };
        try depth_bounds.validate(b);
        return b;
    }
};

/// The flags this plugin reads, one line each: the server's spelling, the `Flags` field it sets, and how its value
/// parses. A flag named twice keeps its last value, as the server's parse does.
const known = .{
    .{ .name = "--mtp-min-depth", .field = "mtp_min_depth", .value = depth },
    .{ .name = "--mtp-max-depth", .field = "mtp_max_depth", .value = depth },
    .{ .name = "--mtp-depth", .field = "mtp_max_depth", .value = oldDepth },
};

/// A draft depth: an integer in 1..MAX_DEPTH.
fn depth(text: []const u8) error{NotADepth}!u32 {
    return depth_bounds.parseDepth(text, mtp_mod.MAX_DEPTH);
}

/// `--mtp-max-depth`'s deprecated spelling `--mtp-depth`: any integer, clamped into 1..MAX_DEPTH.
fn oldDepth(text: []const u8) error{NotADepth}!u32 {
    return depth_bounds.parseClamped(text, mtp_mod.MAX_DEPTH);
}

/// A flag whose value the server refuses, and that value.
pub const Bad = struct { flag: []const u8 = "", value: []const u8 = "" };

/// The flags in `args` (`argv[0]` first). A flag's value is the argument after it, as the server reads it.
pub fn parse(args: []const [*:0]const u8, bad: *Bad) !Flags {
    var out: Flags = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.span(args[i]);
        inline for (known) |k| if (std.mem.eql(u8, arg, k.name)) {
            i += 1;
            const text: []const u8 = if (i < args.len) std.mem.span(args[i]) else "";
            @field(out, k.field) = k.value(text) catch |e| {
                bad.* = .{ .flag = k.name, .value = text };
                return e;
            };
            break;
        };
    }
    return out;
}

extern "c" fn _NSGetArgc() *const c_int;
extern "c" fn _NSGetArgv() *const [*]const [*:0]const u8;

/// What the server's flags set in this plugin: the MTP round policy's depth range (`depth_bounds.active`), with one line
/// when a depth flag is named. A value the server refuses fails here by name, with a line naming the flag.
pub fn apply() !void {
    var bad: Bad = .{};
    const f = parse(_NSGetArgv().*[0..@intCast(_NSGetArgc().*)], &bad) catch |e| {
        log.err("[turboexl3_qwen38] {s} '{s}' is not a value the server takes ({s})\n", .{ bad.flag, bad.value, @errorName(e) });
        return e;
    };
    const bounds = f.depthBounds() catch |e| {
        log.err("[turboexl3_qwen38] --mtp-min-depth {d} is above --mtp-max-depth {d}\n", .{ f.mtp_min_depth.?, f.mtp_max_depth.? });
        return e;
    };
    const b = bounds orelse return;
    depth_bounds.active = b;
    if (b.pinned()) |n| {
        log.info("[turboexl3_qwen38] MTP depth pinned to {d}\n", .{n});
    } else if (b.max != 0) {
        log.info("[turboexl3_qwen38] MTP depth bounded {d}..{d}\n", .{ b.min, b.max });
    } else {
        log.info("[turboexl3_qwen38] MTP depth bounded {d}..its adaptive cap\n", .{b.min});
    }
}

test "host flags: the depth flags take what the server takes" {
    const testing = std.testing;
    var bad: Bad = .{};
    // None named: no bounds. `serve` comes first, as the server's subcommand does.
    const none = try (try parse(&.{ "mlx-serve", "serve", "--model", "/m", "--no-mtp" }, &bad)).depthBounds();
    try testing.expect(none == null);
    // Equal flags pin a depth.
    const pin = try (try parse(&.{ "mlx-serve", "--mtp-min-depth", "4", "--mtp-max-depth", "4" }, &bad)).depthBounds();
    try testing.expectEqual(@as(?u32, 4), pin.?.pinned());
    // `--mtp-depth` is the max, any integer clamped into 1..MAX_DEPTH, and the flag named last wins.
    const old = try (try parse(&.{ "mlx-serve", "--mtp-max-depth", "3", "--mtp-depth", "12" }, &bad)).depthBounds();
    try testing.expectEqual(depth_bounds.Bounds{ .max = mtp_mod.MAX_DEPTH }, old.?);
    // What the server refuses: a depth outside 1..MAX_DEPTH, a flag without a value, a min above the max.
    try testing.expectError(error.NotADepth, parse(&.{ "mlx-serve", "--mtp-min-depth", "9" }, &bad));
    try testing.expectEqualStrings("--mtp-min-depth", bad.flag);
    try testing.expectError(error.NotADepth, parse(&.{ "mlx-serve", "--mtp-max-depth" }, &bad));
    try testing.expectError(error.MinAboveMax, (try parse(&.{ "mlx-serve", "--mtp-min-depth", "5", "--mtp-max-depth", "4" }, &bad)).depthBounds());
}
