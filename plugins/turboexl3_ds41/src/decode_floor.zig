//! `MLX_SERVE_DSV41_DECODE_FLOOR` (read at load; unset, empty or `1`: on; `0`: off; anything else is refused by name): the
//! streamed bank's decode bill with a decode phase's runtime floor in place of a prompt pass's. The bill's decode budget
//! (`streamBudget`) takes the decode phase's working set out: the state at `context + stream_generation` positions, one
//! row's arrays and `PREFILL_RUNTIME_FLOOR_BYTES` (512 MiB, a prompt pass's runtime floor), the sum times 5/4: 804 MiB at
//! the bank's 32768-token rung, 640 of it the floor. Decode's runtime need beside its state is MLX's peak over its active
//! bytes: 51 MiB on the 16K card and 29 on greedy (l39d's `[mem-decode]`, w-20261103-0515; MLX's cache is the stop's,
//! DECODE_STOP's 1 GiB cap). On, the decode phase's working set is billed with `floor_bytes` (128 MiB) instead: 325 MiB,
//! 480 MiB back to the decode slot rows (one more row at 470 MiB, two where the remainder under a row is 460 MiB or more);
//! the prompt's bill and its rows are unchanged. Exact: only the decode rows' count moves (a route's bytes are the
//! records' in any row). The decode phase then holds that much more; David's ruling (2026-11-03, "server's peak"): a
//! request's peak is its prompt's, which decode stays under (wf1d: about 3.4 GiB below).

const std = @import("std");

pub const Mode = enum { off, on };
/// Set at load (`configure`), before the bank opens.
pub var mode: Mode = .off;

pub const Error = error{DecodeFloorSetting};

/// The decode phase's runtime floor when on: 2.5 times MLX's largest measured peak over active at decode.
pub const floor_bytes: u64 = 128 << 20;

/// `value`: null, empty or `1` on; `0` off; anything else refused by name.
pub fn parse(value: ?[]const u8) Error!Mode {
    const v = value orelse return .on;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.DecodeFloorSetting;
}

pub fn configure(value: ?[]const u8) Error!void {
    mode = try parse(value);
}

/// The generated bill's `decode_floor`: `floor_bytes` on, null off (the prompt pass's floor, as before).
pub fn floor() ?u64 {
    return if (mode == .on) floor_bytes else null;
}

/// The load line.
pub fn loadLine(buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    (switch (mode) {
        .off => w.print("[turboexl3_ds41] the streamed bank's decode bill with a prompt pass's runtime floor, 512 MiB (MLX_SERVE_DSV41_DECODE_FLOOR=0)\n", .{}),
        .on => w.print("[turboexl3_ds41] the streamed bank's decode bill with a decode floor of {d} MiB, not a prompt pass's 512 (MLX_SERVE_DSV41_DECODE_FLOOR; =0: a prompt pass's): its working set 480 MiB less, those bytes the decode slot rows'; [bank-bill] ws_decode_mb=\n", .{floor_bytes >> 20}),
    }) catch return buf[0..0];
    return w.buffered();
}

const testing = std.testing;

test "decode floor setting: unset, empty or 1 on; 0 off; anything else refused by name" {
    const was = mode;
    defer mode = was;
    for ([_]?[]const u8{ null, "", "1" }) |v| try testing.expectEqual(Mode.on, try parse(v));
    try testing.expectEqual(Mode.off, try parse("0"));
    for ([_][]const u8{ "on", "128", "check", " 1" }) |v| try testing.expectError(error.DecodeFloorSetting, parse(v));
    try configure(null);
    try testing.expectEqual(@as(?u64, 128 << 20), floor());
    try configure("0");
    try testing.expectEqual(@as(?u64, null), floor());
}

test "decode floor: the 32K rung's decode working set 480 MiB less, one decode row at the pack's 470 MiB" {
    // prefillBytes at 40960 positions, one row (the pack's config: 3200 B a token, 40 layers' 144-row window rings of
    // 512 halves, a row's held, attention, index and MoE bytes), times 5/4
    const state: u64 = 40960 * 3200 + 40 * 144 * 512 * 2;
    const row_bytes: u64 = 119_808 + 376_832 + 491_520 + 144_384;
    const prompt_floor: u64 = 512 << 20;
    const stock = (state + row_bytes + prompt_floor) * 5 / 4;
    const on_ = (state + row_bytes + floor_bytes) * 5 / 4;
    try testing.expectEqual(@as(u64, 843_717_120), stock);
    try testing.expectEqual(@as(u64, 503_316_480), stock - on_);
    try testing.expectEqual(@as(u64, 1), (stock - on_) / 492_810_240);
}
