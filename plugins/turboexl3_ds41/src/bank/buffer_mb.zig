//! MLX's command-buffer split (the MB of bound arrays a command buffer takes before MLX commits it, 50 on the M5
//! Max) raised to `prompt_mb` for a streamed bank's prompt phase and put back at the change to decode: a prompt's
//! buffers split less (prefill +3%), the same words. A user's MLX_MAX_MB_PER_BUFFER wins at every phase.

const std = @import("std");

/// The prompt phase's split, MB.
pub const prompt_mb: i32 = 500;

extern "c" fn dsv41_command_buffer_limits(ops: c_int, mb: c_int, prev_ops: *c_int, prev_mb: *c_int) c_int;
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    on: bool = false,
    before: ?i32 = null,
};
pub const shared_state = @import("sdk").shared.Shared(State, "dsv41_buffer_mb", .{});
pub inline fn ps() *State {
    return shared_state.get();
}
/// The split in effect before the prompt's raise (null: not raised).

/// At load: the split is raised only where the user sets no MLX_MAX_MB_PER_BUFFER of their own (`user`).
pub fn configure(user: ?[]const u8) void {
    ps().on = user == null;
}

fn set(mb: i32) ?i32 {
    var po: c_int = 0;
    var pm: c_int = 0;
    return if (dsv41_command_buffer_limits(-1, mb, &po, &pm) == 0) pm else null;
}

/// The phase change to a prompt (before its first op): the split raised to `prompt_mb`, the one before it kept.
pub fn enterPrompt() void {
    if (!ps().on or ps().before != null) return;
    ps().before = set(prompt_mb);
}

/// The phase change to decode (the prompt's work done): the split before the prompt's raise put back.
pub fn enterDecode() void {
    const b = ps().before orelse return;
    _ = set(b);
    ps().before = null;
}
