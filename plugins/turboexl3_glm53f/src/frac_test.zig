//! The routed experts' tile gate, MLX-free (build.zig adds it to `zig build test`): turboexl3's own reference decode (its
//! src/core/format.zig, `decodeTile`: the codeword windows its kernels are tested against) of real routed-expert tiles of
//! turboderp's GLM-5.3-Flash self-calibrated branches against exllamav3 1.6's rule for them (exllamav3_ext/quant/frac.cu:
//! weight i takes KA or KA + 1 fresh bits by mask 0xAAAA at a half-integer K, KA at an integer one; its state the 16 ring
//! bits ending at the running sum, MSB-first in 32-bit words; mul1 after): n 56 (sc_2.60 layer 3, K 3.5), n 40 (sc_2.60,
//! K 2.5) and every integer rate the packs route at, n 32, 48, 64 and 80 (their states also equal the converter
//! reference's, exl3_ref.dq_states). src/glm_tiles.json holds each tile's stored words and that rule's f16 bits
//! (row-major), written by scripts/gen_glm_tiles.py from HTTP Range reads of the branches' shards.

const std = @import("std");
const format = @import("turboexl3_format");
const testing = std.testing;

const fixture = @embedFile("glm_tiles.json");

const Tile = struct { src: []const u8, n: u32, words: []const u16, expect: []const u16 };

test "turboexl3_glm53f tiles: turboexl3's tile decode is exllamav3's rule on turboderp's GLM-5.3-Flash tiles at every routed rate (n 32, 40, 48, 56, 64, 80)" {
    const parsed = try std.json.parseFromSlice(struct { tiles: []const Tile }, testing.allocator, fixture, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var count: [129]u32 = @splat(0);
    for (parsed.value.tiles) |t| {
        const rate = format.Rate.fromN(t.n) orelse return error.FracFixtureRate;
        try testing.expectEqual(@as(usize, t.n), t.words.len);
        try testing.expectEqual(@as(usize, 256), t.expect.len);
        var out: [256]u16 = undefined;
        format.decodeTile(t.words, rate, format.Decode.mul1, &out);
        testing.expectEqualSlices(u16, t.expect, &out) catch |e| {
            std.debug.print("glm tile fixture: {s} differs\n", .{t.src});
            return e;
        };
        var cw: [256]u16 = undefined;
        format.unpackTile(t.words, rate, &cw);
        for ([_]usize{ 0, 1, 2, 3, 15, 16, 17, 127, 128, 254, 255 }) |i| try testing.expectEqual(format.codewordByDefinition(t.words, rate, i), cw[i]);
        count[t.n] += 1;
    }
    for ([_]usize{ 32, 40, 48, 56, 64, 80 }) |n| try testing.expect(count[n] >= 14);
}
