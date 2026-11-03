//! `MLX_SERVE_DSV41_BUFFER_MB` (read at load; unset, empty or `prompt`: 500 MB during prompts; a whole number N > 0: N
//! MB during prompts; `0`: off; anything else refused by name): MLX's command-buffer split (the MB of bound arrays a
//! command buffer takes before MLX commits it: MLX_MAX_MB_PER_BUFFER at the Metal device's start, 50 MB on the M5 Max)
//! raised for the prompt phase through the host's `HostGpu.command_buffer_limits`, and put back at the change to decode.
//!
//! MLX reads MLX_MAX_MB_PER_BUFFER once, when mlx-serve's `main()` builds its Metal device (before any plugin loads), and
//! the setting is process-wide; measured process-wide at 500 on ctl46 it lifted prefill (mb3p: cold +3.05%, new text
//! +3.93%) and moved decode both ways (mb3d: card -0.21%, greedy +0.21%). So the plugin raises it for prompts only: the
//! copied streamed source's phase change to a prompt (`enterPrompt`, beside DECODE_STOP's) sets `prompt_mb` and keeps
//! the split before it; the change to decode (`enterDecode`, after the prompt's last eval and the grow) puts it back.
//! Every stream's next commit check reads the new split: only where buffers split moves, so the same words (a text
//! gate). The ops limit is left as it is.
//!
//! A user's MLX_MAX_MB_PER_BUFFER wins: set, the plugin leaves MLX's split alone at every phase. A host whose `HostGpu`
//! stops before `command_buffer_limits` (older than the member): MLX's split throughout; the load line says which.
//! The member is read at its offset (`limits_at`) when the host's table holds it, as src/host_shims.zig reads `alias`,
//! so the plugin builds against a kit that predates it (MLX_SERVE_REF's: its HostGpu 96 bytes).
//! `[buffer-mb] mode= prompt_mb= decode_mb= enters= exits=` at each request's end.

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.host;

pub const Mode = enum { off, prompt };

/// The setting (`configure`, at load).
pub var mode: Mode = .prompt;
/// The prompt phase's split, MB.
pub var prompt_mb: i32 = 500;
/// The default prompt split, MB.
pub const default_mb: i32 = 500;

pub const Error = error{BufferMbSetting};

/// Why the split is not raised though the setting is on (the load line names it).
pub const Held = enum { none, user_set, no_host_member };
pub var held: Held = .none;

/// `HostGpu.command_buffer_limits` in the host's table (mlx-serve's additive ABI entry, ddalcu/mlx-serve#1065, after
/// `alias` at byte 96): bytes 104 to 112.
pub const limits_at: usize = 104;
const Limits = *const fn (ctx: ?*anyopaque, ops: i32, mb: i32, prev_ops: *i32, prev_mb: *i32) callconv(.c) abi.Verdict;
comptime {
    if (@hasField(abi.HostGpu, "command_buffer_limits")) {
        if (@offsetOf(abi.HostGpu, "command_buffer_limits") != limits_at) @compileError("this kit's HostGpu.command_buffer_limits is not at buffer_mb.limits_at");
        if (@FieldType(abi.HostGpu, "command_buffer_limits") != Limits) @compileError("this kit's HostGpu.command_buffer_limits is not buffer_mb.Limits");
    }
}
/// The host's setter, from a table whose `size` reaches it (112 bytes or more); null on an older host's (96 or 104).
fn hostLimits(g: *const abi.HostGpu) ?Limits {
    if (g.size < limits_at + @sizeOf(Limits)) return null;
    const at: *const Limits = @ptrFromInt(@intFromPtr(g) + limits_at);
    return at.*;
}
var limits: ?Limits = null;
var limits_ctx: ?*anyopaque = null;
/// The split in effect before the prompt's raise (null: not raised).
var before: ?i32 = null;
// The meters since load.
var last_decode_mb: i32 = 0;
var enters: u64 = 0;
var exits: u64 = 0;

/// `value`: null, empty or `prompt` the default 500; a whole number N > 0 N; `0` off; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!struct { mode: Mode, mb: i32 } {
    const v = value orelse return .{ .mode = .prompt, .mb = default_mb };
    if (v.len == 0 or std.mem.eql(u8, v, "prompt")) return .{ .mode = .prompt, .mb = default_mb };
    if (std.mem.eql(u8, v, "0")) return .{ .mode = .off, .mb = 0 };
    const n = std.fmt.parseInt(i32, v, 10) catch return error.BufferMbSetting;
    if (n <= 0) return error.BufferMbSetting;
    return .{ .mode = .prompt, .mb = n };
}

/// At load: the setting, the user's own MLX_MAX_MB_PER_BUFFER (`user`, null unset) and the host (`host`: its
/// `command_buffer_limits`, null on a host without the member).
pub fn configure(value: ?[]const u8, user: ?[]const u8, host: ?kit.host.Host) Error!void {
    const p = try parse(value);
    mode = p.mode;
    prompt_mb = p.mb;
    held = .none;
    limits = null;
    if (mode == .off) return;
    if (user != null) {
        held = .user_set;
        return;
    }
    const h = host orelse {
        held = .no_host_member;
        return;
    };
    limits = hostLimits(h.gpu()) orelse {
        held = .no_host_member;
        return;
    };
    limits_ctx = h.gpu().ctx;
}

/// The phase change to a prompt (before its first op): the split raised to `prompt_mb`, the one before it kept.
pub fn enterPrompt() void {
    const f = limits orelse return;
    if (before != null) return;
    var po: i32 = 0;
    var pm: i32 = 0;
    if (!f(limits_ctx, -1, prompt_mb, &po, &pm).isOk()) return;
    before = pm;
    last_decode_mb = pm;
    enters += 1;
}

/// The phase change to decode (the prompt's work done): the split before the prompt's raise put back.
pub fn enterDecode() void {
    const f = limits orelse return;
    const b = before orelse return;
    var po: i32 = 0;
    var pm: i32 = 0;
    _ = f(limits_ctx, -1, b, &po, &pm);
    before = null;
    exits += 1;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] MLX's command-buffer split left as it is in every phase (MLX_SERVE_DSV41_BUFFER_MB=0)\n", .{}),
        .prompt => switch (held) {
            .none => w.print("[turboexl3_ds41] MLX's command buffers split at {d} MB during prompts, MLX's own split at decode (MLX_SERVE_DSV41_BUFFER_MB; =0 off): [buffer-mb] at each request's end\n", .{prompt_mb}),
            .user_set => w.print("[turboexl3_ds41] MLX_MAX_MB_PER_BUFFER is set: MLX's split as the user set it in every phase (MLX_SERVE_DSV41_BUFFER_MB yields to it)\n", .{}),
            .no_host_member => w.print("[turboexl3_ds41] the host has no HostGpu.command_buffer_limits: MLX's split in every phase (MLX_MAX_MB_PER_BUFFER at launch sets it for the whole process)\n", .{}),
        },
    }) catch return buf[0..0];
    return w.buffered();
}

/// The meters' line (`[buffer-mb]`), "" when the split is never raised.
pub fn meters(buf: []u8) []const u8 {
    if (limits == null) return buf[0..0];
    var w: std.Io.Writer = .fixed(buf);
    w.print("[buffer-mb] mode={t} prompt_mb={d} decode_mb={d} enters={d} exits={d}\n", .{ mode, prompt_mb, last_decode_mb, enters, exits }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "buffer_mb setting: unset, empty or prompt 500; N; 0 off; anything else refused by name" {
    for ([_]?[]const u8{ null, "", "prompt" }) |v| {
        const p = try parse(v);
        try testing.expectEqual(Mode.prompt, p.mode);
        try testing.expectEqual(@as(i32, 500), p.mb);
    }
    try testing.expectEqual(@as(i32, 256), (try parse("256")).mb);
    try testing.expectEqual(Mode.off, (try parse("0")).mode);
    for ([_][]const u8{ "-1", "on", "500mb", " 500", "1.5" }) |v| try testing.expectError(error.BufferMbSetting, parse(v));
}

test "buffer_mb: the user's MLX_MAX_MB_PER_BUFFER wins, a host without the member holds it, the phase hooks no-op" {
    try configure(null, "500", null);
    try testing.expectEqual(Held.user_set, held);
    try configure(null, null, null);
    try testing.expectEqual(Held.no_host_member, held);
    enterPrompt();
    enterDecode();
    try testing.expectEqual(@as(u64, 0), enters);
    var buf: [512]u8 = undefined;
    try testing.expectEqual(@as(usize, 0), meters(&buf).len);
}

test "buffer_mb: the host's setter read at its offset, only from a table whose size reaches it" {
    const T = struct {
        var calls: u32 = 0;
        fn set(_: ?*anyopaque, _: i32, _: i32, prev_ops: *i32, prev_mb: *i32) callconv(.c) abi.Verdict {
            calls += 1;
            prev_ops.* = 0;
            prev_mb.* = 50;
            return .ok;
        }
    };
    // A table with room past the kit's HostGpu (96 bytes in a kit before the member, 112 with it).
    var table: extern struct { g: abi.HostGpu, tail: [2]usize } = undefined;
    const slot: *Limits = @ptrFromInt(@intFromPtr(&table.g) + limits_at);
    slot.* = &T.set;
    for ([_]u32{ 96, 104, 111 }) |size| {
        table.g.size = size;
        try testing.expect(hostLimits(&table.g) == null);
    }
    table.g.size = 112;
    const got = hostLimits(&table.g) orelse return error.TestUnexpectedResult;
    var po: i32 = -1;
    var pm: i32 = -1;
    try testing.expect(got(null, -1, 500, &po, &pm).isOk());
    try testing.expectEqual(@as(i32, 50), pm);
    try testing.expectEqual(@as(u32, 1), T.calls);
}
