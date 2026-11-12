//! What the routed math hands turboexl3 from the stream's slot banks, MLX-free. g53smc (2026-11-06) served the sc_3.10
//! pack at 101 prompt / 104 decode slot rows: the warmup's decode forward failed (EngineFailed: the routed seam's name for
//! any error the host does not know), and the first prompt then found the grown rows still referenced. The default
//! per-layer split gives two layers 3 ext rows at that plan; the stream floors every slot array at one VM page, so those
//! banks' 4 KiB scale rows hold 4 rows beside 3 code rows, and turboexl3 checks a bank's suh and svh at the code's rows
//! (BadExl3Shape). At 8 / 8 rows no layer grew (no ext bank), so g53smb never reached it. The math slices each scale array
//! to its code's rows (`routed_t3.fitRows`).

const std = @import("std");
const Glm5n = @import("mlx_serve/sdk/weights/streamed.zig").Glm5n;
const routed_t3 = @import("routed_t3.zig");

const testing = std.testing;

/// Bank components: code, rout (svh), rin (suh) per projection (gate, up, down).
const code = [_]usize{ 0, 3, 6 };
const scales = [_]usize{ 1, 2, 4, 5, 7, 8 };

test "turboexl3_glm53f stream banks: the default split grows two layers by 3 rows at 101 / 104, none at 8 / 8" {
    var prompt: [42]u32 = undefined;
    var decode: [42]u32 = undefined;
    try Glm5n.planRows(.{ .prompt = 101, .decode = 104 }, &prompt, &decode);
    var small: usize = 0;
    for (prompt, decode) |p, d| {
        try testing.expectEqual(@as(u32, 101), p);
        if (d - p >= 1 and d - p <= 3) small += 1;
    }
    try testing.expectEqual(@as(u32, 3), decode[12] - prompt[12]);
    try testing.expectEqual(@as(u32, 3), decode[13] - prompt[13]);
    try testing.expectEqual(@as(usize, 2), small);
    try Glm5n.planRows(.{ .prompt = 8, .decode = 8 }, &prompt, &decode);
    for (prompt, decode) |p, d| try testing.expectEqual(p, d);
}

test "turboexl3_glm53f stream banks: a bank of 1 to 3 rows has longer scale arrays than code arrays; the math's fit gives turboexl3 every scale at the code's rows" {
    // K3 (48 halfwords, layers 4-44 of sc_3.10) and K4 (64, its layer 3), and sc_2.60's 40 and 56
    for ([_]u32{ 40, 48, 56, 64 }) |n| {
        var r: u32 = 1;
        while (r <= 8) : (r += 1) {
            var longer = false;
            for (code) |c| try testing.expectEqual(r, try Glm5n.slotArrayRows(n, c, r));
            for (scales) |c| {
                const got = try Glm5n.slotArrayRows(n, c, r);
                try testing.expect(got >= r);
                if (got > r) longer = true;
                // the fit: sliced to the code's rows exactly when longer, so every scale reaches turboexl3 at r rows
                const fit = routed_t3.fitRows(@intCast(r), @intCast(got));
                if (got > r) {
                    try testing.expectEqual(@as(?c_int, @intCast(r)), fit);
                } else try testing.expectEqual(@as(?c_int, null), fit);
            }
            // the page floor (16 KiB) over 4 KiB rows (gate and up rout, down rin): 4 rows at least
            try testing.expectEqual(r < 4, longer);
        }
    }
}
