//! `MLX_SERVE_DSV41_SCRATCH_TAIL` (read at load; unset, empty or `1`: on; `0`: off; `check`; any other value is refused
//! by name): a wide prompt layer's new streams written in place over its x1 region, not into a fresh buffer.
//!
//! Off, TAIL_FUSE's kernel (src/tail_fuse.zig: the joinless combine and the hc expand in one kernel) writes the layer's
//! new streams `[L, hc, D]` into a fresh MLX buffer (671 MB of bf16 at 16K rows), and the next layer's hc expand writes its x1
//! into the x1 region (`MLX_SERVE_DSV41_SCRATCH`, src/scratch_x1.zig) from them. MLX holds a buffer until its command
//! buffer completes and reuses a cached one only within two pages of the size asked, under the kit's 8 GB pool: a
//! layer's transients pass through that pool between one stream buffer's release and the next one's allocation, so each
//! layer's streams take a fresh buffer (pt2 counted 2.34 GB of fresh buffers a layer in the rest stage; pt4 read the
//! tail's output allocation as ~11.5 ms a wide layer). On (`1`), with the x1 region serving the layer (SCRATCH on and the
//! region within its cap), one MoE group over every span, and the next layer a wide one of the same call whose launches
//! before its hc expand are the streams' only readers (no read-ahead of its own: STAGE_NEXT's `only`; no DSpark row),
//! the kernel's write-into twin writes the new streams over the region itself: each thread reads its row's HC copies of
//! x1 (and every other input) before it stores the row's HC outputs at those words, and no other launch reads the
//! region's x1 after the tail, so the words are the stock kernel's; the region is then published as the layer's
//! streams, and the next layer's hc expand reads and writes the region in place the same way (each of its threads reads
//! its row's copies first). The twin binds the stock kernel's inputs and one output (30 buffers, as the stock's; Metal's
//! argument table holds 31). The call's last wide layer keeps the fresh buffer: its streams are read after the call's
//! last eval (the CED replay, the head, the next chunk). `check`, within the one routed call: the stock kernel's fresh
//! streams first, evaluated before the twin is encoded, then the twin over the region, its words compared with the
//! stock's (a `[scratch-tail] check` line a layer), the stock's served. src/gpu_check.zig checks both in-place kernels on
//! the GPU at the served template arguments (`zig build gpu-check`, run in a GPU window).

const std = @import("std");

pub const Mode = enum { off, on, check };

/// The switch (`configure`, at load).
pub var mode: Mode = .on;

pub const Error = error{ScratchTailSetting};

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.ScratchTailSetting;
}

/// One layer's `check`: the words compared and those that differ.
pub fn report(layer: usize, words: u64, differ: u64) void {
    std.debug.print("[scratch-tail] check layer={d} words={d} differ={d}\n", .{ layer, words, differ });
}

test "scratch_tail setting: on unset, empty or 1; 0 off; check; anything else refused by name" {
    try std.testing.expectEqual(Mode.on, try parse(null));
    try std.testing.expectEqual(Mode.on, try parse(""));
    try std.testing.expectEqual(Mode.off, try parse("0"));
    try std.testing.expectEqual(Mode.on, try parse("1"));
    try std.testing.expectEqual(Mode.check, try parse("check"));
    for ([_][]const u8{ "on", "2", "01", "true", "tail", "CHECK" }) |v| try std.testing.expectError(error.ScratchTailSetting, parse(v));
}
