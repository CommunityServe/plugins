//! MLX_SERVE_DSV41_FRAC's tile gate, MLX-free (build.zig adds it to `zig build test`): turboexl3's own reference decode
//! (its src/format.zig, `decodeTile`: the codeword windows its kernels are tested against) of real tiles from
//! turboderp's DeepSeek-V4.1-Flash-1017 EXL3 branches, at the half-integer rates n 40 (K 2.5) and n 24 (K 1.5) and at
//! the integer n 32 / n 48 as controls, against exllamav3 1.6.4's rule for them (exllamav3_ext/quant/frac.cu: weight i
//! takes KA or KA + 1 fresh bits by mask 0xAAAA, its state the 16 ring bits ending at the running sum, MSB-first in
//! 32-bit words; mul1 after). src/frac_tiles.json holds each tile's stored words and that rule's f16 bits (row-major),
//! written by the box tool tools/gen_frac_fixture.py through tools/frac_ref.py from HTTP Range reads of the branches'
//! shards; its integer controls' states equal the 2.81 pack's converter reference (exl3_ref.dq_states) there.

const std = @import("std");
const format = @import("turboexl3_format");
const testing = std.testing;

const fixture = @embedFile("frac_tiles.json");

const Tile = struct { src: []const u8, n: u32, words: []const u16, expect: []const u16 };

test "frac: turboexl3's tile decode is exllamav3 frac.cu's rule on turboderp's tiles, n 40 and n 24 (n 32 / 48 too)" {
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
            std.debug.print("frac fixture: {s} differs\n", .{t.src});
            return e;
        };
        // the plain bit-by-bit definition agrees with the unpacked windows (turboexl3's own check, on these tiles)
        var cw: [256]u16 = undefined;
        format.unpackTile(t.words, rate, &cw);
        for ([_]usize{ 0, 1, 2, 3, 15, 16, 17, 127, 128, 254, 255 }) |i| try testing.expectEqual(format.codewordByDefinition(t.words, rate, i), cw[i]);
        count[t.n] += 1;
    }
    for ([_]usize{ 24, 32, 40, 48 }) |n| try testing.expect(count[n] >= 14);
}

test "frac: turboexl3's windows end where frac.cu's ring sums do, at n 24 and n 40" {
    for ([_]u32{ 24, 40 }) |n| {
        const rate = format.Rate.fromN(n).?;
        const ka = n / 16;
        var s: usize = 0;
        for (0..256) |i| {
            s += ka + ((@as(u32, 0xAAAA) >> @intCast(i % 16)) & 1);
            try testing.expectEqual(s, rate.bitEnd(i));
        }
        try testing.expectEqual(@as(usize, 16) * n, s);
    }
}
