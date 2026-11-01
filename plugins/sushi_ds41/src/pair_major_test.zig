//! Pair-major packs (`expert_quant.tile_order: "pair_major"`, routed trellis stored as `trellis_pm`), without a GPU:
//! the tile permutation, the tensor and shard checks that run before a weight loads, the zero pad tensors, and the
//! fill rules that let a page-aligned expert read straight into its slot.

const std = @import("std");
const exl3 = @import("sushi_exl3");
const source = @import("sushi/dsv41/dsv41_source.zig");
const expert_io = @import("sushi/expert_io.zig");
const expert_quant = @import("sushi/expert_quant.zig");

const testing = std.testing;
const gl = exl3.group_layout;
const TileOrder = exl3.format.TileOrder;

test "sushi_ds41 pair-major: tile-major puts an output pair's k range in one run, and the inverse returns the pack order" {
    const it = 5;
    const ot = 4;
    const n = 3;
    var src: [it * ot * n]u16 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast(i);
    var dst: [src.len]u16 = undefined;
    gl.tileMajor(u16, &dst, &src, it, ot, n);
    for (0..ot / 2) |pair| for (0..it) |tk| for (0..2) |o| {
        const want = src[(tk * ot + pair * 2 + o) * n ..][0..n];
        try testing.expectEqualSlices(u16, want, dst[((pair * it + tk) * 2 + o) * n ..][0..n]);
    };

    for ([_][3]usize{ .{ 5, 4, 3 }, .{ 1, 2, 1 }, .{ 16, 8, 24 }, .{ 3, 10, 7 } }) |g| {
        const gi, const go, const gn = g;
        const pack = try testing.allocator.alloc(u16, gi * go * gn);
        defer testing.allocator.free(pack);
        for (pack, 0..) |*v, i| v.* = @truncate(i *% 40503 +% 17);
        const major = try testing.allocator.alloc(u16, pack.len);
        defer testing.allocator.free(major);
        const back = try testing.allocator.alloc(u16, pack.len);
        defer testing.allocator.free(back);
        gl.tileMajor(u16, major, pack, gi, go, gn);
        gl.tileMajorInverse(u16, back, major, gi, go, gn);
        try testing.expectEqualSlices(u16, pack, back);
    }
}

test "sushi_ds41 pair-major: a trellis_pm name and its 5-D shape read as the same tile grid, and a layer refuses a wrong or mixed one" {
    const name = try gl.Name.parse("gate_proj.g1.trellis_pm");
    try testing.expect(name.pair_major and name.part == 0 and name.group == 1);
    try testing.expectEqual(TileOrder.pair_major, name.tileOrder());
    try testing.expect(!(try gl.Name.parse("gate_proj.g1.trellis")).pair_major);
    try testing.expectError(error.Exl3GroupNameInvalid, gl.Name.parse("gate_proj.g1.suh_pm"));
    try testing.expectError(error.Exl3GroupNameInvalid, gl.Name.parse("gate_proj.trellis_pm.x"));

    const d = gl.tileDims(.pair_major, &.{ 3, 4, 16, 2, 24 }).?;
    try testing.expectEqual(gl.TileDims{ .it = 16, .ot = 8, .n = 24 }, d);
    try testing.expectEqual(gl.TileDims{ .it = 16, .ot = 8, .n = 24 }, gl.tileDims(.pack, &.{ 3, 16, 8, 24 }).?);
    try testing.expectEqual(@as(?gl.TileDims, null), gl.tileDims(.pair_major, &.{ 3, 16, 8, 24 }));
    try testing.expectEqual(@as(?gl.TileDims, null), gl.tileDims(.pair_major, &.{ 3, 4, 16, 3, 24 }));
    try testing.expectEqual(@as(?gl.TileDims, null), gl.tileDims(.pack, &.{ 3, 4, 16, 2, 24 }));

    var ok = gl.Layout.init(256, 128, .{ .n = 64 });
    _ = try ok.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 32 }, .u16);
    _ = try ok.add("down_proj.g0.trellis_pm", &.{ 3, 8, 8, 2, 32 }, .u16);
    try testing.expectEqual(TileOrder.pair_major, ok.tile_order.?);
    try testing.expectEqual(@as(u32, 32), ok.groups[0].rates[0].n);
    try testing.expectEqual(@as(u32, 32), ok.groups[0].rates[2].n);

    var wrong = gl.Layout.init(256, 128, .{ .n = 64 });
    try testing.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 16, 8, 32 }, .u16));
    try testing.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 4, 32 }, .u16));
    try testing.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 8, 16, 2, 32 }, .u16));
    try testing.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 16, 4, 2, 32 }, .u16));
    try testing.expectError(error.Exl3TrellisGeometry, wrong.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 66 }, .u16));
    try testing.expectError(error.Exl3GroupGeometry, wrong.add("gate_proj.g0.trellis", &.{ 3, 4, 16, 2, 32 }, .u16));

    var mixed = gl.Layout.init(256, 128, .{ .n = 64 });
    _ = try mixed.add("gate_proj.g0.trellis_pm", &.{ 3, 4, 16, 2, 32 }, .u16);
    try testing.expectError(error.Exl3TileOrderMixed, mixed.add("up_proj.g0.trellis", &.{ 3, 16, 8, 32 }, .u16));
}

test "sushi_ds41 pair-major: a shard stamp must name the config's order, and no stamp is the pack order" {
    try exl3.shardTileOrderAgrees(.pack, null);
    try exl3.shardTileOrderAgrees(.pair_major, "pair_major");
    try testing.expectError(error.Exl3ShardStampMismatch, exl3.shardTileOrderAgrees(.pair_major, null));
    try testing.expectError(error.Exl3ShardStampMismatch, exl3.shardTileOrderAgrees(.pack, "pair_major"));
    try testing.expectError(error.Exl3ShardStampMismatch, exl3.shardTileOrderAgrees(.pair_major, "rowmajor"));
}

const Tensor = struct { name: []const u8, dtype: []const u8 = "U8", shape: []const u64 };

fn elementBytes(dtype: []const u8) u64 {
    return if (std.mem.eql(u8, dtype, "U16")) 2 else 1;
}

/// A safetensors shard of zero tensors; `stamp` is its `__metadata__` `tile_order`.
fn writeShard(dir: std.Io.Dir, file: []const u8, stamp: ?[]const u8, tensors: []const Tensor) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(testing.allocator);
    try header.append(testing.allocator, '{');
    if (stamp) |s| try header.print(testing.allocator, "\"__metadata__\":{{\"tile_order\":\"{s}\"}},", .{s});
    var total: u64 = 0;
    for (tensors, 0..) |t, i| {
        var bytes: u64 = elementBytes(t.dtype);
        for (t.shape) |d| bytes *= d;
        if (i > 0) try header.append(testing.allocator, ',');
        try header.print(testing.allocator, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, k| try header.print(testing.allocator, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try header.print(testing.allocator, "],\"data_offsets\":[{d},{d}]}}", .{ total, total + bytes });
        total += bytes;
    }
    try header.append(testing.allocator, '}');
    const out = try testing.allocator.alloc(u8, 8 + header.items.len + total);
    defer testing.allocator.free(out);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    @memset(out[8 + header.items.len ..], 0);
    try dir.writeFile(testing.io, .{ .sub_path = file, .data = out });
}

fn realPath(dir: std.Io.Dir, buf: []u8) ![]const u8 {
    return buf[0..try dir.realPath(testing.io, buf)];
}

// A 256 x 128 expert: gate and up are 16 x 8 tiles, down is 8 x 16; three experts of n = 32 halfwords.
const gate_pm: Tensor = .{ .name = "model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis_pm", .dtype = "U16", .shape = &.{ 3, 4, 16, 2, 32 } };
const down_pm: Tensor = .{ .name = "model.layers.3.mlp.switch_mlp.down_proj.g0.trellis_pm", .dtype = "U16", .shape = &.{ 3, 8, 8, 2, 32 } };
const gate_pack: Tensor = .{ .name = "model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis", .dtype = "U16", .shape = &.{ 3, 16, 8, 32 } };
const up_pack: Tensor = .{ .name = "model.layers.3.mlp.switch_mlp.up_proj.g0.trellis", .dtype = "U16", .shape = &.{ 3, 16, 8, 32 } };

fn validate(tensors: []const Tensor, stamp: ?[]const u8, order: TileOrder) !void {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-00001-of-00001.safetensors", stamp, tensors);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try source.validateTileOrder(testing.io, testing.allocator, try realPath(tmp.dir, &buf), order, 256, 128);
}

test "sushi_ds41 pair-major: a pack whose order the config, the tensor names, the shapes or the shard stamps disagree on is refused by name" {
    try validate(&.{ gate_pm, down_pm }, "pair_major", .pair_major);
    try validate(&.{ gate_pack, up_pack }, null, .pack);
    try testing.expectError(error.Exl3TileOrderMismatch, validate(&.{ gate_pm, down_pm }, "pair_major", .pack));
    try testing.expectError(error.Exl3TileOrderMismatch, validate(&.{ gate_pack, up_pack }, null, .pair_major));
    // Both names in one shard, or in two shards.
    try testing.expectError(error.Exl3TileOrderMixed, validate(&.{ gate_pm, up_pack }, "pair_major", .pair_major));
    {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeShard(tmp.dir, "model-00001-of-00002.safetensors", "pair_major", &.{gate_pm});
        try writeShard(tmp.dir, "model-00002-of-00002.safetensors", null, &.{up_pack});
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        try testing.expectError(error.Exl3TileOrderMixed, source.validateTileOrder(testing.io, testing.allocator, try realPath(tmp.dir, &buf), .pair_major, 256, 128));
    }
    // The input and output grids swapped.
    const swapped: Tensor = .{ .name = gate_pm.name, .dtype = "U16", .shape = &.{ 3, 16, 4, 2, 32 } };
    try testing.expectError(error.Exl3GroupGeometry, validate(&.{swapped}, "pair_major", .pair_major));
    // The shard that holds the routed tensors stamps the order they lie in, or nothing for the pack order.
    try testing.expectError(error.Exl3ShardStampMismatch, validate(&.{ gate_pm, down_pm }, null, .pair_major));
    try testing.expectError(error.Exl3ShardStampMismatch, validate(&.{ gate_pack, up_pack }, "pair_major", .pack));
}

test "sushi_ds41 pair-major: a weight map naming both orders is mixed, one naming neither is the pack order" {
    const cases = [_]struct { json: []const u8, want: anyerror!TileOrder }{
        .{ .json = "{\"model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis\":\"a\",\"model.layers.3.mlp.switch_mlp.gate_proj.g0.suh\":\"a\"}", .want = .pack },
        .{ .json = "{\"model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis_pm\":\"a\",\"mtp.0.mlp.switch_mlp.up_proj.g0.trellis_pm\":\"b\"}", .want = .pair_major },
        .{ .json = "{\"embed.weight\":\"a\"}", .want = .pack },
        .{ .json = "{\"model.layers.3.mlp.switch_mlp.gate_proj.g0.trellis_pm\":\"a\",\"mtp.0.mlp.switch_mlp.up_proj.g0.trellis\":\"b\"}", .want = error.Exl3TileOrderMixed },
    };
    for (cases) |c| {
        const tree = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.json, .{});
        defer tree.deinit();
        const got = expert_quant.tileOrderOfMap(tree.value.object);
        if (c.want) |order| try testing.expectEqual(order, try got) else |err| try testing.expectError(err, got);
    }
}

test "sushi_ds41 pair-major: a repack's zero pad tensors are never kept, billed or counted as weights" {
    const pad = source.pad_prefix ++ "model-00001-of-00002.0";
    for ([_]source.Parts{ .{}, .{ .dspark = true }, .{ .experts = true }, .{ .vision = true } }) |parts| {
        try testing.expect(!source.keeps(pad, parts));
        try testing.expect(!source.keeps(source.pad_prefix ++ "model-00001-of-00002.1", parts));
    }
    try testing.expect(source.keeps("embed.weight", .{}));

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-00001-of-00002.safetensors", "pair_major", &.{
        .{ .name = "embed.weight", .shape = &.{1000} },
        .{ .name = pad, .shape = &.{9000} },
        .{ .name = "mtp.0.attn.wq_a.weight", .shape = &.{7000} },
        .{ .name = "__pad__.model-00001-of-00002.1", .shape = &.{12000} },
    });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try realPath(tmp.dir, &buf);
    try testing.expectEqual(@as(u64, 1000), try source.residentBytes(testing.io, testing.allocator, root));
    try testing.expectEqual(@as(u64, 7000), try source.dsparkBytes(testing.io, testing.allocator, root));
}

test "sushi_ds41 pair-major: tensorRegion reads a 5-D trellis_pm tensor's shape" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(tmp.dir, "shard.safetensors", "pair_major", &.{ gate_pm, .{ .name = "x.scales", .shape = &.{ 4, 2 } } });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const full = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/shard.safetensors", .{try realPath(tmp.dir, &buf)}, 0);
    defer testing.allocator.free(full);
    const fd = try expert_io.openHinted(full, .{ .nocache = false, .readahead_off = false });
    defer _ = std.c.close(fd);
    const r = try expert_io.tensorRegion(testing.allocator, fd, gate_pm.name);
    try testing.expectEqual(@as(u8, 5), r.rank);
    try testing.expectEqualSlices(u64, &.{ 3, 4, 16, 2, 32 }, r.shape[0..5]);
}

fn pagedSpan(dst: []u8, page: u64, file_page: u64, dst_page: u64) expert_io.FillSpan {
    return .{ .file = 0, .offset = file_page * page, .len = page, .dst = dst[dst_page * page ..].ptr };
}

test "sushi_ds41 pair-major: a page-aligned span whose neighbour in the file lands elsewhere reads alone and direct; adjacent ones still coalesce" {
    const page: u64 = @intCast(expert_io.pageSize());
    const mem = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(expert_io.align_bytes), @intCast(page * 6));
    defer testing.allocator.free(mem);

    // Pages 0 and 1 of the file land in slots 0 and 3: two direct reads, not one bounced run.
    var apart = [_]expert_io.FillSpan{ pagedSpan(mem, page, 0, 0), pagedSpan(mem, page, 1, 3) };
    var plan = try expert_io.FillPlan.init(testing.allocator, &apart, 64 << 20, 1, false);
    defer plan.deinit();
    try testing.expectEqual(@as(usize, 2), plan.jobs.len);
    try testing.expect(plan.jobs[0].direct and plan.jobs[1].direct);

    // The same pages landing next to each other are one read.
    var together = [_]expert_io.FillSpan{ pagedSpan(mem, page, 0, 0), pagedSpan(mem, page, 1, 1) };
    var joined = try expert_io.FillPlan.init(testing.allocator, &together, 64 << 20, 1, false);
    defer joined.deinit();
    try testing.expectEqual(@as(usize, 1), joined.jobs.len);
    try testing.expect(joined.jobs[0].direct);

    // Unaligned neighbours keep the old rule: they are one bounced read whatever their destinations.
    var small = [_]expert_io.FillSpan{
        .{ .file = 0, .offset = 100, .len = 100, .dst = mem.ptr },
        .{ .file = 0, .offset = 200, .len = 100, .dst = mem[4096..].ptr },
    };
    var legacy = try expert_io.FillPlan.init(testing.allocator, &small, 64 << 20, 1, false);
    defer legacy.deinit();
    try testing.expectEqual(@as(usize, 1), legacy.jobs.len);
    try testing.expect(!legacy.jobs[0].direct);
}

test "sushi_ds41 pair-major: a fill partition charges each read, so a layer's small tail reads stay off the last big read's worker" {
    var dst: [9 * 4096]u8 align(expert_io.align_bytes) = undefined;
    var spans: [9]expert_io.FillSpan = undefined;
    for (&spans, 0..) |*s, i| s.* = .{ .file = 0, .offset = i * (4 << 20), .len = if (i < 3) 2211840 else 10240, .dst = dst[i * 4096 ..].ptr };
    var plan = try expert_io.FillPlan.init(testing.allocator, &spans, 64 << 20, 4, false);
    defer plan.deinit();
    var w: usize = 0;
    while (!(plan.bounds[w] <= 2 and 2 < plan.bounds[w + 1])) w += 1;
    try testing.expect(plan.bounds[w + 1] - 3 <= 2);
}

test "sushi_ds41 pair-major: page-aligned experts read straight into their slots land the file's bytes, counted as direct and never reshuffled" {
    const page = expert_io.pageSize();
    const bytes = try testing.allocator.alloc(u8, page * 8);
    defer testing.allocator.free(bytes);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    prng.random().bytes(bytes);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "experts.bin", .data = bytes });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/experts.bin", .{try realPath(tmp.dir, &buf)}, 0);
    defer testing.allocator.free(path);
    const fd = try expert_io.openHinted(path, .{});
    defer _ = std.c.close(fd);

    const slots = try testing.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(expert_io.align_bytes), page * 8);
    defer testing.allocator.free(slots);
    // File pages 1, 2 and 5 land in slots 4, 0 and 6.
    const spans = [_]expert_io.FillSpan{
        .{ .file = 0, .offset = page, .len = page, .dst = slots[4 * page ..].ptr },
        .{ .file = 0, .offset = 2 * page, .len = page, .dst = slots[0..].ptr },
        .{ .file = 0, .offset = 5 * page, .len = page, .dst = slots[6 * page ..].ptr },
    };
    for ([_]bool{ false, true }) |force_bounce| {
        @memset(slots, 0);
        var plan = try expert_io.FillPlan.init(testing.allocator, &spans, 64 << 20, 1, force_bounce);
        defer plan.deinit();
        const pool = try expert_io.FillPool.create(testing.allocator, .{ .workers = 1 });
        defer pool.destroy();
        try pool.run(&.{fd}, &plan);
        try testing.expectEqualSlices(u8, bytes[page .. 2 * page], slots[4 * page ..][0..page]);
        try testing.expectEqualSlices(u8, bytes[2 * page .. 3 * page], slots[0..page]);
        try testing.expectEqualSlices(u8, bytes[5 * page .. 6 * page], slots[6 * page ..][0..page]);
        const st = pool.stats();
        try testing.expectEqual(@as(u64, 0), st.bytes_reshuffled);
        try testing.expectEqual(@as(u64, if (force_bounce) 0 else 3), st.reads_direct);
        try testing.expectEqual(@as(u64, if (force_bounce) 3 * page else 0), st.bytes_bounced);
    }
}
