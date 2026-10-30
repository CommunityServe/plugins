//! The server's own launch flags this plugin follows, read off the process's arguments at load (`_NSGetArgv`): no load
//! setting carries them. `known` has one line per flag, and each value parses by the server's own rules, so the plugin
//! takes exactly what the server takes. The server exits on a value it refuses; one that still reaches the plugin
//! refuses the load by name.

const std = @import("std");
const depth_bounds = @import("mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const mtp_mod = @import("mlx_serve/sdk/spec/mtp/mtp.zig");
const log = @import("mlx_serve/sdk/util/log.zig");
const attn_core = @import("direct/attn_core.zig");

/// What the flags say: null (a switch false) where the process names none.
pub const Flags = struct {
    mtp_min_depth: ?u32 = null,
    mtp_max_depth: ?u32 = null,
    /// `--int8-qk`: the dense prefill attention on int8 Q and K.
    int8_qk: bool = false,

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
/// parses (`value`), or, for a switch, which takes no value, what naming it sets (`set`). A flag named twice keeps its
/// last value, as the server's parse does.
const known = .{
    .{ .name = "--mtp-min-depth", .field = "mtp_min_depth", .value = depth },
    .{ .name = "--mtp-max-depth", .field = "mtp_max_depth", .value = depth },
    .{ .name = "--mtp-depth", .field = "mtp_max_depth", .value = oldDepth },
    .{ .name = "--int8-qk", .field = "int8_qk", .set = true },
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

/// The flags in `args` (`argv[0]` first). A flag's value is the argument after it, as the server reads it; a switch
/// takes none.
pub fn parse(args: []const [*:0]const u8, bad: *Bad) !Flags {
    var out: Flags = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.span(args[i]);
        inline for (known) |k| if (std.mem.eql(u8, arg, k.name)) {
            if (@hasField(@TypeOf(k), "set")) {
                @field(out, k.field) = k.set;
            } else {
                i += 1;
                const text: []const u8 = if (i < args.len) std.mem.span(args[i]) else "";
                @field(out, k.field) = k.value(text) catch |e| {
                    bad.* = .{ .flag = k.name, .value = text };
                    return e;
                };
            }
            break;
        };
    }
    return out;
}

extern "c" fn _NSGetArgc() *const c_int;
extern "c" fn _NSGetArgv() *const [*]const [*:0]const u8;

/// What the server's flags set in this plugin: the MTP round policy's depth range (`depth_bounds.active`), with one line
/// when a depth flag is named, and the dense prefill attention's int8 Q/K (`attn_core.int8_qk`), with one line when
/// `--int8-qk` is. A value the server refuses fails here by name, with a line naming the flag.
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
    attn_core.int8_qk = f.int8_qk;
    if (f.int8_qk) log.info("[turboexl3_qwen38] --int8-qk: a dense prefill block of 64 rows or more attends on int8 Q and K (lossy)\n", .{});
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

test "host flags: the depth flags and --int8-qk take what the server takes" {
    const testing = std.testing;
    var bad: Bad = .{};
    // None named: no bounds, no int8 Q/K. `serve` comes first, as the server's subcommand does.
    const plain = try parse(&.{ "mlx-serve", "serve", "--model", "/m", "--no-mtp" }, &bad);
    try testing.expect(try plain.depthBounds() == null);
    try testing.expect(!plain.int8_qk);
    // `--int8-qk` is a switch: it takes no value, so the flag after it is still read.
    const q8 = try parse(&.{ "mlx-serve", "serve", "--int8-qk", "--mtp-max-depth", "3" }, &bad);
    try testing.expect(q8.int8_qk);
    try testing.expectEqual(@as(?u32, 3), q8.mtp_max_depth);
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
