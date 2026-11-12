//! turboexl3_glm53f: the MTP head's draft-step switch, read at load (plugin.zig) and taken by the copied head
//! (src/mlx_serve/plugins/glm5_next/head.zig, patches/turboexl3-model.patch).
//! - `TURBOEXL3_GLM53F_HEAD_LAST_ROW` (`last_row`): the merged first draft step (the committed rows' history and the draft
//!   row in one forward) runs its routed experts for the last row alone (sdk/bases/blocks/moe.zig `mlpLastRow`): the
//!   other rows' MoE outputs are never read, and the last row's bytes are the whole call's for it. A history append runs
//!   as before; a step past turboexl3's 16 decode rows runs whole (declined).
const std = @import("std");

pub var last_row: bool = false;
/// Since the last `line` (one request's): HEAD_LAST_ROW's merged steps (the last row alone, and those the block
/// declined, run whole).
pub var last_rows: u64 = 0;
pub var declined: u64 = 0;

pub const last_row_line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_HEAD_LAST_ROW=1: the MTP head's merged first draft step (the committed rows' history and the draft row in one forward) runs its routed experts for the last row alone, the only row whose output is read; the routing, the shared expert, the attention and the cache appends over every row as before; the same bits for that row; no slot or row count changes\n";

/// The request's line for `meters` (empty with the switch off): HEAD_LAST_ROW's merged steps; then its counts start
/// over for the next request.
pub fn line(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    if (last_row) w.print("[turboexl3_glm53f] TURBOEXL3_GLM53F_HEAD_LAST_ROW: {d} merged draft steps ran their routed experts for the last row alone, {d} ran whole (declined)\n", .{ last_rows, declined }) catch {};
    reset();
    return w.buffered();
}

pub fn reset() void {
    last_rows = 0;
    declined = 0;
}

test "turboexl3_glm53f stream draft step: HEAD_LAST_ROW's line is its merged steps and those declined, and each request's starts over" {
    const t = std.testing;
    reset();
    defer {
        last_row = false;
        reset();
    }
    var buf: [512]u8 = undefined;
    // off: no line
    try t.expectEqualStrings("", line(&buf));
    last_row = true;
    last_rows = 2;
    declined = 1;
    try t.expectEqualStrings("[turboexl3_glm53f] TURBOEXL3_GLM53F_HEAD_LAST_ROW: 2 merged draft steps ran their routed experts for the last row alone, 1 ran whole (declined)\n", line(&buf));
    try t.expect(last_rows == 0 and declined == 0);
}
