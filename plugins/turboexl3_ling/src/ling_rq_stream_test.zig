//! turboexl3_ling's MLX-free test of RQ (src/ling_rq.zig) on the copied streamer, in src/ling_streamer_test.zig's
//! binary: the bank's decode rows grown and trimmed in the order the overlay's ling_source.zig `setPhase` and
//! `trimDecode` change them, through the copied hook (`ExpertsWith`) on the trace backend over `FakeSource` (the
//! streamer's residency with no reads or memory), at the rows RQ plans for three requests and a trim.
const std = @import("std");
const testing = std.testing;
const xp = @import("mlx_serve/sdk/weights/streamed/experts.zig");
const ops = @import("mlx_serve/sdk/weights/streamed/ops.zig");
const ling_rq = @import("ling_rq.zig");

const mib: u64 = 1 << 20;

/// The served bank as RQ plans it (src/ling_rq.zig's tests: dylib 01ca9769's load lines, its logged budgets at the low
/// end of their MiB).
fn served() ling_rq.Terms {
    const row: u64 = 336_427_008;
    const window: u64 = 48 * (row / 68);
    const bank: ling_rq.Bank = .{ .row_bytes = row, .prompt_windows = 5 * window, .decode_windows = window, .staging = 118_489_088, .n_experts = 512 };
    const budget: ling_rq.Budget = .{ .prompt = 69998 * mib, .decode = 76329 * mib };
    return .{ .bank = bank, .budget = budget, .context = 32768, .kv_tokens = 32768, .kv = .{ .layer = 32_896, .layers = 9, .head = 1_152 }, .rows = ling_rq.rowsOf(bank, budget) catch unreachable };
}

/// A request at its decode start: its prompt's capacity, and the head's history (its rows less one) appended.
fn decodeStart(p: u64) ling_rq.Slot {
    return .{ .p = p, .trunk_cap = ling_rq.wholePromptCapacity(p), .head_cap = ling_rq.nextCapacity(0, p - 1) };
}

/// The hook's config (the copied experts.zig `TestConfig`'s fields): two routed layers of the trace geometry over 512
/// experts, six a token, so a decode-width forward fits one route.
const Cfg = struct { hidden_size: u32 = 256, moe_intermediate_size: u32 = 128, n_layers: u32 = 2, n_experts_per_tok: u32 = 6, swiglu_limit: f64 = 10.0, n_routed_experts: u32 = 512 };
const Chain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, Chain, .{});

/// Every layer's banks as the hook binds them at `rows` persistent rows: the base rows; the grown rows past them, every
/// projection's (none at the base); the transient rows, unless a phase change freed them and nothing has bound them
/// again (`transient` false). The source's capacity and grown rows are the same.
fn expectBanks(g: *ops.TraceOps, ex: *const Ex, src: *xp.FakeSource, base: u32, rows: u32, transient: bool) !void {
    for (ex.banks, 0..) |b, l| {
        const layer: u32 = @intCast(l);
        try testing.expectEqual(rows, src.policies[l].capacity);
        try testing.expectEqual(rows - base, src.bankRows(layer, .ext));
        const bb = b[@backingInt(xp.BankKind.base)] orelse return error.BaseRowsUnbound;
        try testing.expectEqual(@as(c_int, @intCast(base)), g.shapeOf(bb.gate.code).d[0]);
        if (rows > base) {
            const eb = b[@backingInt(xp.BankKind.ext)] orelse return error.GrownRowsUnbound;
            inline for (.{ "gate", "up", "down" }) |proj| {
                try testing.expectEqual(@as(c_int, @intCast(rows - base)), g.shapeOf(@field(eb, proj).code).d[0]);
            }
        } else try testing.expect(b[@backingInt(xp.BankKind.ext)] == null);
        if (transient) {
            const tb = b[@backingInt(xp.BankKind.transient)] orelse return error.TransientRowsUnbound;
            try testing.expectEqual(@as(c_int, @intCast(src.transient_rows)), g.shapeOf(tb.gate.code).d[0]);
        } else try testing.expect(b[@backingInt(xp.BankKind.transient)] == null);
    }
}

/// `ids` routed on every layer in the source's phase, a route's ids at a time, each call released and flushed.
fn routeAll(src: *xp.FakeSource, ids: []const u16) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, xp.max_route_ids);
        for (0..src.policies.len) |l| src.release(try src.route(@intCast(l), ids[at..][0..n], &.{}));
        try src.flush();
        at += n;
    }
}

test "RQ on the copied streamer: three requests grow the bank to the decode rows RQ plans for them (259, 241, 227), a trim shrinks it to its base and grows it at fewer rows (235), keeping the prompt rows' residents, and every bank is bound and sized to its rows" {
    const a = testing.allocator;
    const t = served();
    const base = t.rows.prompt;
    // The plans: each request's decode start, and the second's trim, a 2-row verify at 25,599 past the 25,600 tokens
    // its plan holds (the head's cache at the 28,928 the plan held for it).
    const s1 = decodeStart(1120);
    const r1 = ling_rq.planDecode(t, &s1, .decode, 1120, 1, 0).rows;
    const s2 = decodeStart(18380);
    const p2 = ling_rq.planDecode(t, &s2, .decode, 18380, 1, 0);
    var s2t = s2;
    s2t.trunk_cap = p2.cap;
    s2t.head_cap = p2.head_cap;
    try testing.expect(ling_rq.crosses(25599, 2, p2.cap));
    const r2t = ling_rq.planDecode(t, &s2t, .trim, 25599, 2, 0).rows;
    const s3 = decodeStart(31652);
    const r3 = ling_rq.planDecode(t, &s3, .decode, 31652, 1, 0).rows;
    const planned = [_]u32{ base, r1, p2.rows, r2t, r3 };
    try testing.expectEqualSlices(u32, &.{ 214, 259, 241, 235, 227 }, &planned);

    // The bank: two routed layers at the served base over 512 experts, in its prompt phase.
    const base_rows = [_]u32{ base, base };
    var src = try xp.FakeSource.init(a, .{ .hidden = 256, .inter = 128, .n_experts = 512, .rows = &base_rows });
    defer src.deinit();
    var g = ops.TraceOps.init(a);
    defer g.deinit();
    const c: Cfg = .{};
    var ex = try Ex.init(a, &g, &src, Chain.init(.{}, &c), &c);
    defer ex.deinit();
    try expectBanks(&g, &ex, &src, base, base, true);
    // Request 1's decode start (setPhase(.decode)): the prompt's windows freed, then the rows its plan gives grown.
    _ = try ex.releaseTransient();
    try expectBanks(&g, &ex, &src, base, base, false);
    try ex.grow(&g, &[_]u32{ r1, r1 });
    try expectBanks(&g, &ex, &src, base, r1, true);
    // Request 2's prompt start (setPhase(.prompt)): the grown rows and decode's window freed, the prompt rows'
    // residents kept, the windows regrown. Its prompt fills each layer's prompt rows with experts 0 to 213.
    _ = try ex.shrink(&base_rows, .keep);
    try expectBanks(&g, &ex, &src, base, base, false);
    _ = try ex.regrowTransient(&g, true);
    try expectBanks(&g, &ex, &src, base, base, true);
    var ids: [512]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    try routeAll(&src, ids[0..base]);
    // Its decode start, and 27 experts its decode reads into the grown rows (300 to 326).
    _ = try ex.releaseTransient();
    try ex.grow(&g, &[_]u32{ p2.rows, p2.rows });
    try expectBanks(&g, &ex, &src, base, p2.rows, true);
    try routeAll(&src, ids[300..327]);
    for (src.policies) |*pol| try testing.expectEqual(p2.rows, pol.occupancy);
    // The trim (trimDecode): the grown rows and decode's window freed, the prompt rows' residents kept, then the trim's
    // rows grown with decode's window. Experts 0 to 213 are still resident; 300 to 326 are gone with their rows.
    _ = try ex.shrink(&base_rows, .keep);
    try expectBanks(&g, &ex, &src, base, base, false);
    try ex.grow(&g, &[_]u32{ r2t, r2t });
    try expectBanks(&g, &ex, &src, base, r2t, true);
    for (0..src.policies.len) |l| {
        for (ids[0..base]) |e| try testing.expect(src.isResident(@intCast(l), e));
        for (ids[300..327]) |e| try testing.expect(!src.isResident(@intCast(l), e));
    }
    // Request 3: its prompt start, then its decode start at its rows.
    _ = try ex.shrink(&base_rows, .keep);
    _ = try ex.regrowTransient(&g, true);
    _ = try ex.releaseTransient();
    try ex.grow(&g, &[_]u32{ r3, r3 });
    try expectBanks(&g, &ex, &src, base, r3, true);
}
