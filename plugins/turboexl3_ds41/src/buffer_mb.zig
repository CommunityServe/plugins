//! MLX's command-buffer split (the MB of bound arrays a
//! command buffer takes before MLX commits it: MLX_MAX_MB_PER_BUFFER at the Metal device's start, 50 MB on the M5 Max)
//! raised for the prompt phase through the host's `HostGpu.command_buffer_limits`, and put back at the change to decode.
//!
//! MLX reads MLX_MAX_MB_PER_BUFFER once, when mlx-serve's `main()` builds its Metal device (before any plugin loads), and
//! the setting is process-wide; measured process-wide at 500 on ctl46 it lifted prefill (mb3p: cold +3.05%, new text
//! +3.93%) and moved decode both ways (mb3d: card -0.21%, greedy +0.21%). So the plugin raises it for prompts only: the
//! copied streamed source's phase change to a prompt (`enterPrompt`, beside src/decode_stop.zig's) sets `prompt_mb` and keeps
//! the split before it; the change to decode (`enterDecode`, after the prompt's last eval and the grow) puts it back.
//! Every stream's next commit check reads the new split: only where buffers split moves, so the same words (a text
//! gate). The ops limit is left as it is.
//!
//! A user's MLX_MAX_MB_PER_BUFFER wins: set, the plugin leaves MLX's split alone at every phase. A host whose `HostGpu`
//! stops before `command_buffer_limits` (older than the member): MLX's split throughout.
//! The member is read at its offset (`limits_at`) when the host's table holds it, as src/host_shims.zig reads `alias`,
//! so the plugin builds against a kit that predates it (MLX_SERVE_REF's: its HostGpu 96 bytes).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.host;

/// The prompt phase's split, MB.
pub const prompt_mb: i32 = 500;

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

/// At load: the user's own MLX_MAX_MB_PER_BUFFER (`user`, null unset) and the host (`host`: its
/// `command_buffer_limits`, null on a host without the member). The split is raised only with neither in the way.
pub fn configure(user: ?[]const u8, host: ?kit.host.Host) void {
    limits = null;
    if (user != null) return;
    const h = host orelse return;
    limits = hostLimits(h.gpu()) orelse return;
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
}

/// The phase change to decode (the prompt's work done): the split before the prompt's raise put back.
pub fn enterDecode() void {
    const f = limits orelse return;
    const b = before orelse return;
    var po: i32 = 0;
    var pm: i32 = 0;
    _ = f(limits_ctx, -1, b, &po, &pm);
    before = null;
}

const testing = std.testing;

test "buffer_mb: the user's MLX_MAX_MB_PER_BUFFER wins, a host without the member holds it, the phase hooks no-op" {
    configure("500", null);
    try testing.expect(limits == null);
    configure(null, null);
    try testing.expect(limits == null);
    enterPrompt();
    try testing.expect(before == null);
    enterDecode();
    try testing.expect(before == null);
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
