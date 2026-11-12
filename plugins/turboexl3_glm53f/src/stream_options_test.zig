//! GLM-5.3-Flash's bank over the kit's expert stream (`kit.streamed`), MLX-free: the options `Exl3Source.init` hands the
//! kit's stream (`glm5n/experts.zig servedStreamOptions`) and the checks it runs on them before the stream opens, with no
//! slot memory, read pool, event or GPU. Every SC pack stores its layer 3 at another K than layers 4-44, so its bank has
//! two geometries, on which the kit's stream refuses prompt lever 1's relocation (InvalidOptions: the first smoke's
//! "the streamed bank does not open"); the served options turn it off there. `zig build test -Dglm53f-packs=<pack
//! dir>[,...]` (GLM53F_PACKS in the test's environment) runs the same checks over those packs' own manifests and
//! experts.bin (opened, no record read). build.zig roots this test at the plugin, MLX left undefined. The read pool's
//! bill is checked against the kit's own pool sizing (`expert.io.poolOptions`, `poolBytes`), and the switches whose
//! mechanism is a seam of the kit's stream against what this kit has.

const std = @import("std");
const streamed = @import("mlx_serve/sdk/weights/streamed.zig");
const Glm5n = streamed.Glm5n;
const stream_room = @import("stream_room.zig");

const testing = std.testing;

/// The 42 routed layers' tile widths (halfwords) of a pack: layer 3 (bank layer 0) at `l3`, layers 4-44 at `rest`.
fn layout(l3: u32, rest: u32) [42]u32 {
    var ns: [42]u32 = @splat(rest);
    ns[0] = l3;
    return ns;
}

/// The SC packs' routed layouts (LAYOUT.md "K by layer"): sc_2.10 K3 / K2, sc_2.60 K3.5 / K2.5, sc_3.10 K4 / K3,
/// sc_4.10 K5 / K4.
const sc_layouts = [_][2]u32{ .{ 48, 32 }, .{ 56, 40 }, .{ 64, 48 }, .{ 80, 64 } };

/// Slot rows a load may plan (`plugin.zig streamRows`): the fewest, a mid plan, every expert resident.
const plans = [_]struct { prompt: u32, decode: u32 }{ .{ .prompt = 8, .decode = 8 }, .{ .prompt = 48, .decode = 96 }, .{ .prompt = 288, .decode = 288 } };

test "turboexl3_glm53f stream: every SC pack's two code widths pass the stream's checks on the served options with prompt lever 1's relocation off; one code width keeps it" {
    for (sc_layouts) |sc| {
        const ns = layout(sc[0], sc[1]);
        for (plans) |p| try testing.expect(try Glm5n.checkStreamOptionsN(testing.allocator, &ns, .{ .prompt = p.prompt, .decode = p.decode }));
    }
    // one code width (the 3.05 pack's K3 bank): the relocation stays on
    const k3 = layout(48, 48);
    for (plans) |p| try testing.expect(!try Glm5n.checkStreamOptionsN(testing.allocator, &k3, .{ .prompt = p.prompt, .decode = p.decode }));
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_GROW_UNFILLED's served options grow unfilled on every SC layout, and only while it is on" {
    defer Glm5n.setGrowUnfilled(false);
    for (sc_layouts) |sc| {
        const ns = layout(sc[0], sc[1]);
        for (plans) |p| {
            Glm5n.setGrowUnfilled(false);
            const off = try Glm5n.servedFillN(testing.allocator, &ns, .{ .prompt = p.prompt, .decode = p.decode });
            try testing.expect(!off.unfilled and !off.relocate);
            Glm5n.setGrowUnfilled(true);
            const on = try Glm5n.servedFillN(testing.allocator, &ns, .{ .prompt = p.prompt, .decode = p.decode });
            try testing.expect(on.unfilled and !on.relocate);
        }
    }
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_BILL_FIT's read pool bill is the kit stream's own read pool on every SC layout, at the served budget and threads" {
    const page = std.heap.pageSize();
    for (sc_layouts) |sc| {
        const ns = layout(sc[0], sc[1]);
        for (plans) |pl| {
            const x = try Glm5n.servedLookaheadN(testing.allocator, &ns, .{ .prompt = pl.prompt, .decode = pl.decode });
            // the stream's budget (the one the plugin bills, TURBOEXL3_GLM53F_SPEC_PICK1's picks below it) and 2 threads;
            // the rule past the first thread the stream's 0
            try testing.expectEqual(stream_room.streamer_spec_budget, x.budget);
            try testing.expectEqual(@as(u32, 2), x.threads);
            try testing.expectEqual(@as(u32, 0), x.idle_busy);
            // the staging, the ticket ring and the speculative slots `Pool.start` maps (`expert.io.poolBytes`)
            try testing.expectEqual(x.pool_bytes, stream_room.readPoolBytes(x.record_bytes, page, stream_room.streamer_spec_budget));
        }
    }
}

test "turboexl3_glm53f stream: a switch whose seam the kit's stream lacks is refused by name, and none is while all are off" {
    defer {
        Glm5n.setPromptKeep(false);
        Glm5n.setDemandSplit(false);
        Glm5n.setDemandFirst(0);
        Glm5n.setAdoptPieces(false);
    }
    try testing.expect(Glm5n.missingSeam() == null);
    const S = Glm5n.seams;
    Glm5n.setPromptKeep(true);
    if (!S.prompt_keep) try testing.expectEqualStrings("TURBOEXL3_GLM53F_PROMPT_KEEP", Glm5n.missingSeam().?);
    Glm5n.setPromptKeep(false);
    Glm5n.setDemandSplit(true);
    if (!S.demand_split) try testing.expectEqualStrings("TURBOEXL3_GLM53F_DEMAND_SPLIT", Glm5n.missingSeam().?);
    Glm5n.setDemandSplit(false);
    // DEMAND_HELP is the kit stream's own demandFirst: no seam
    Glm5n.setDemandFirst(2);
    try testing.expect(Glm5n.missingSeam() == null);
    Glm5n.setDemandFirst(0);
    Glm5n.setAdoptPieces(true);
    if (!S.adopt_pieces) try testing.expectEqualStrings("TURBOEXL3_GLM53F_ADOPT_PIECES", Glm5n.missingSeam().?);
    Glm5n.setAdoptPieces(false);
    try testing.expect(Glm5n.missingSeam() == null);
}

/// A usage profile of one layer of two experts (the Recorder's first form), its counts `counts` and prompt set `set`.
fn profileText(comptime counts: []const u8, comptime set: []const u8) []const u8 {
    return "{\"model_type\":\"glm5_next\",\"layers\":1,\"experts\":2,\"counts\":[" ++ counts ++ "],\"source\":{\"requests\":1,\"prompt_tokens\":9,\"decode_tokens\":1,\"prompt_set\":\"" ++ set ++ "\",\"binary_sha256\":\"0\"}}";
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_SEED_DECODE's seed reads expert-usage-decode-v1.json only while it is on, and with it on a pack without the file is refused by name" {
    defer Glm5n.setSeedDecode(false);
    const a = testing.allocator;
    // a pack with both profiles: the full record's counts and the decode-only record's
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const full = try std.fs.path.join(a, &.{ dir, "expert-usage-v1.json" });
    defer a.free(full);
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = full, .data = profileText("[7,1]", "coding") });
    const dec = try std.fs.path.join(a, &.{ dir, "expert-usage-decode-v1.json" });
    defer a.free(dec);
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = dec, .data = profileText("[1,5]", "coding [decode-only]") });
    Glm5n.setSeedDecode(false);
    try testing.expectEqualStrings("expert-usage-v1.json", Glm5n.seedFile());
    {
        var d = try Glm5n.seedDecision(a, testing.io, dir, 1, 2);
        defer d.profile.?.deinit();
        try testing.expectEqualSlices(u64, &.{ 7, 1 }, d.profile.?.counts);
    }
    Glm5n.setSeedDecode(true);
    try testing.expectEqualStrings("expert-usage-decode-v1.json", Glm5n.seedFile());
    {
        var d = try Glm5n.seedDecision(a, testing.io, dir, 1, 2);
        defer d.profile.?.deinit();
        try testing.expectEqualSlices(u64, &.{ 1, 5 }, d.profile.?.counts);
    }
    // a pack with neither: refused by name with the switch on, no seed without it
    var empty = testing.tmpDir(.{});
    defer empty.cleanup();
    var empty_buf: [std.fs.max_path_bytes]u8 = undefined;
    const empty_dir = empty_buf[0..try empty.dir.realPath(testing.io, &empty_buf)];
    try testing.expectError(error.ExpertUsageMissing, Glm5n.seedDecision(a, testing.io, empty_dir, 1, 2));
    Glm5n.setSeedDecode(false);
    try testing.expect((try Glm5n.seedDecision(a, testing.io, empty_dir, 1, 2)).reason == .no_file);
}

test "turboexl3_glm53f stream: the packs in -Dglm53f-packs pass the stream's checks on the served options with prompt lever 1's relocation off (their own manifests and experts.bin)" {
    const v = std.c.getenv("GLM53F_PACKS") orelse return error.SkipZigTest;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
    var n: usize = 0;
    while (it.next()) |dir| {
        for (plans) |p| try testing.expect(try Glm5n.checkStreamOptions(testing.allocator, testing.io, dir, .{ .prompt = p.prompt, .decode = p.decode }));
        std.debug.print("GLM53F_STREAM_OPTIONS {s} served=ok relocate=off\n", .{dir});
        n += 1;
    }
    try testing.expect(n > 0);
}
