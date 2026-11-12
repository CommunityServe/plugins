//! The slot-row plans on the bank's own per-layer split (`Glm5n.planRows`), MLX-free, at g53smd's bank open (the sc_3.10
//! pack, the plugin's own room at the 102,400 MiB ceiling and the 2 GiB margin, 28,123 MiB in use): the widest bill's
//! rows (the load line's 100 / 102), the per-layer bill's (TURBOEXL3_GLM53F_LAYER_BILL), the decode room's
//! (TURBOEXL3_GLM53F_DECODE_ROOM) and both's, each phase's slot arrays as the stream allocates them within its bytes.

const std = @import("std");
const Glm5n = @import("mlx_serve/sdk/weights/streamed.zig").Glm5n;
const stream_room = @import("stream_room.zig");
const Plan = stream_room.Plan;

const testing = std.testing;

fn split(plan: Plan, prompt: []u32, decode: []u32) anyerror!void {
    return Glm5n.planRows(.{ .prompt = plan.prompt, .decode = plan.decode }, prompt, decode);
}

/// One slot bank's bytes at `rows` rows of a layer at `n` halfwords a tile, as the stream allocates it: each of the nine
/// arrays at `Glm5n.slotArrayRows` rows (the page floor included) of its segment (code i16 [in/16, out/16, n] a
/// projection, rout f16 [out], rin f16 [in]).
fn bankBytes(n: u32, rows: u32) !u64 {
    if (rows == 0) return 0;
    const code: u64 = 65_536 * @as(u64, n);
    const seg = [9]u64{ code, 4096, 8192, code, 4096, 8192, code, 8192, 4096 };
    var t: u64 = 0;
    for (seg, 0..) |s, c| t += @as(u64, try Glm5n.slotArrayRows(n, c, rows)) * s;
    return t;
}

/// The sc_3.10 pack: bank layer 0 (model layer 3) at K4 (64 halfwords), layers 4-44 at K3 (48).
fn widths() [42]u32 {
    var ns: [42]u32 = @splat(48);
    ns[0] = 64;
    return ns;
}

const free: u64 = (102_400 << 20) - 21_368_258_560 - (2 << 30) - (28_123 << 20);
/// The decode room's bytes at the room's cell (src/stream_room.zig's decode bill test).
const decode_extra: u64 = 16_707_181_312;

/// The prompt phase's slot arrays (every layer's base rows, the two geometries' 96 transient rows) and the decode
/// phase's (the base rows, each grown layer's ext rows, the two geometries' 32-row window 0) at `plan`.
fn phaseBytes(plan: Plan) !struct { prompt: u64, decode: u64 } {
    const ns = widths();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    try split(plan, &pr, &dr);
    var p: u64 = try bankBytes(64, 96) + try bankBytes(48, 96);
    var d: u64 = try bankBytes(64, 32) + try bankBytes(48, 32);
    for (ns, pr, dr) |n, a, b| {
        const base = try bankBytes(n, a);
        p += base;
        d += base + try bankBytes(n, b - a);
    }
    return .{ .prompt = p, .decode = d };
}

test "turboexl3_glm53f stream rows: g53smd's bank open on the bank's own split, each bill's rows and each phase's slot arrays within its bytes" {
    var layer_bytes: [42]u64 = @splat(9_474_048);
    layer_bytes[0] = 12_619_776;
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    const t = stream_room.streamer_transient_rows;
    const w = stream_room.streamer_window_rows;
    // the widest bill (the control): its rows, and its arrays far within its bytes (41 layers billed a third over)
    const widest = stream_room.rowsFor(free, 12_619_776, 42, t, 8, 288);
    try testing.expectEqual(Plan{ .prompt = 100, .decode = 102 }, widest);
    const wb = try phaseBytes(widest);
    try testing.expect(wb.prompt + 12_000_000_000 < free and wb.decode + 12_000_000_000 < free);
    // the per-layer bill
    const layer = try stream_room.rowsByLayer(free, free, &layer_bytes, t, w, 8, 288, &split, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 130, .decode = 133 }, layer);
    const lb = try phaseBytes(layer);
    try testing.expect(lb.prompt <= free and lb.decode <= free);
    try testing.expect((try phaseBytes(.{ .prompt = layer.prompt, .decode = layer.decode + 1 })).decode > free);
    // the decode room on the widest bill: decode's rows only
    const room = stream_room.rowsForPhases(free, free + decode_extra, 12_619_776, 42, t, 8, 288);
    try testing.expectEqual(Plan{ .prompt = 100, .decode = 134 }, room);
    try testing.expect((try phaseBytes(room)).decode <= free + decode_extra);
    // both: the per-layer bill with the decode room
    const both = try stream_room.rowsByLayer(free, free + decode_extra, &layer_bytes, t, w, 8, 288, &split, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 130, .decode = 174 }, both);
    const bb = try phaseBytes(both);
    try testing.expect(bb.prompt <= free and bb.decode <= free + decode_extra);
    try testing.expect((try phaseBytes(.{ .prompt = both.prompt, .decode = both.decode + 1 })).decode > free + decode_extra);
}

test "turboexl3_glm53f stream rows: TURBOEXL3_GLM53F_BILL_FIT's fixed plan at ai-server's ceiling and --ctx-size 32768 on the bank's own split, each phase's slot arrays within its bytes" {
    // the plan's bytes (src/stream_room.zig's fit test): the prompt rows with both geometries' transient rows, and the
    // decode rows with window 0, at the 112,640 MiB ceiling and a 32,768-token context limit
    const c = stream_room.glmFixture();
    const plan = stream_room.fitPlan(&c, 112_640 << 20, 32_768, true, stream_room.readPoolBytes(stream_room.fit_record, 16 << 10, stream_room.streamer_spec_budget), 0, true, stream_room.fit_rest);
    // (the kit's read pool, its tickets 32 KiB over the evidence's: src/stream_room.zig `readPoolBytes`)
    try testing.expectEqual(@as(u64, 69_143_726_950 - 32_768), plan.prompt);
    try testing.expectEqual(@as(u64, 82_948_186_112 - 32_768), plan.decode);
    var layer_bytes: [42]u64 = @splat(9_474_048);
    layer_bytes[0] = 12_619_776;
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    const rows = try stream_room.rowsByLayer(plan.prompt, plan.decode, &layer_bytes, stream_room.streamer_transient_rows, stream_room.streamer_window_rows, 8, 288, &split, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 167, .decode = 204 }, rows);
    const b = try phaseBytes(rows);
    try testing.expect(b.prompt <= plan.prompt and b.decode <= plan.decode);
    try testing.expect((try phaseBytes(.{ .prompt = rows.prompt, .decode = rows.decode + 1 })).decode > plan.decode);
    try testing.expect((try phaseBytes(.{ .prompt = rows.prompt + 1, .decode = rows.decode })).prompt > plan.prompt);
}

test "turboexl3_glm53f stream rows: TURBOEXL3_GLM53F_BILL_FIT's plan on the bank's own split at the A/B's ceiling and the probes', each phase's slot arrays within its bytes" {
    const c = stream_room.glmFixture();
    const pool = stream_room.readPoolBytes(stream_room.fit_record, 16 << 10, stream_room.streamer_spec_budget);
    var layer_bytes: [42]u64 = @splat(9_474_048);
    layer_bytes[0] = 12_619_776;
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    // the ceiling (MiB), the allowance, the rows: 112,640 the A/B's, 110,592 the build windows' probes' (whose load
    // lines read 161 / 199)
    const cases = [_]struct { mib: u64, allow: u64, rows: Plan }{
        .{ .mib = 112_640, .allow = stream_room.fit_rest, .rows = .{ .prompt = 167, .decode = 204 } },
        .{ .mib = 110_592, .allow = stream_room.fit_rest, .rows = .{ .prompt = 161, .decode = 199 } },
    };
    for (cases) |k| {
        const plan = stream_room.fitPlan(&c, k.mib << 20, 32_768, true, pool, 0, true, k.allow);
        const rows = try stream_room.rowsByLayer(plan.prompt, plan.decode, &layer_bytes, stream_room.streamer_transient_rows, stream_room.streamer_window_rows, 8, 288, &split, &pr, &dr);
        try testing.expectEqual(k.rows, rows);
        const b = try phaseBytes(rows);
        try testing.expect(b.prompt <= plan.prompt and b.decode <= plan.decode);
        try testing.expect((try phaseBytes(.{ .prompt = rows.prompt, .decode = rows.decode + 1 })).decode > plan.decode);
        try testing.expect((try phaseBytes(.{ .prompt = rows.prompt + 1, .decode = rows.decode })).prompt > plan.prompt);
    }
}
