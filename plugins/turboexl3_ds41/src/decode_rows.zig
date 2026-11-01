//! The decode's slot rows split across the layers (`MLX_SERVE_DSV41_DECODE_ROWS`, read at load), MLX-free: unset or
//! empty, the split on a pack whose geometry the curves fit and flat rows (every layer the bill's decode rows, one info
//! line) on any other; `flat`, flat rows; `split`, the split required (a geometry it does not fit logged by name with
//! log.err, the rows flat); any other value refused by name. The copied EXL3 source
//! (patches/turboexl3-zzz1-decode-rows.patch) hands `split` the flat rows, each layer's prompt rows and record bytes
//! and the bank's expert count when it opens; the split spends the flat bill's decode bytes (or fewer), each layer between
//! its prompt rows (at least the curves' first count) and 256 rows, one row at a time where it saves the most misses a
//! byte on the layer's curve below. The stream grows each layer to its own rows at the phase change, as it does flat.
//!
//! The curves: each routed layer's decode misses (summed over the corpus's rounds) at 16 slot-row counts, 130 to 256,
//! the streamer's decode residency policy replayed offline (the route recorder, `zig build -Droute-rec=true`) over
//! decode routes recorded on a development corpus kept apart from the code-ctx test case and its corpus: corpus1 and
//! corpus2 (windows w-corpus-20261101-0428 and w-corpus-20261101-0448; Python's asyncio, difflib and heapq/bisect and
//! Zig's hash map sources with tasks of their own, three chat questions; card and greedy, 8 requests, 1948 decode
//! rounds, fixed full-block verifies). Never the code-ctx prompts: those validated the split, held out (rr1,
//! w-20261101-0359: 7.8% fewer decode misses at the flat 152 rows' bytes, 8.5% at 176's).
//!
//! The curves fit this pack's geometry only (DeepSeek-V4.1-Flash 1017's 2.81bpw EXL3 repack): 40 routed layers, 384
//! experts, each layer's record as the repack lays it out (31 K3 layers of 13315584 logical bytes, the K2 layers 23
//! and 32 to 39 of 8891904). Any other geometry is refused by name and the rows stay flat.

const std = @import("std");

/// `auto` (unset): the split where the curves fit the pack, else flat; `flat`; `split` (required).
pub const Mode = enum { auto, flat, split };

/// Set at load.
pub var mode: Mode = .auto;

/// `MLX_SERVE_DSV41_DECODE_ROWS` (null or empty: auto; `flat`; `split`; any other value refused).
pub fn parse(value: ?[]const u8) error{DecodeRowsSetting}!Mode {
    const v = value orelse return .auto;
    if (v.len == 0) return .auto;
    if (std.mem.eql(u8, v, "flat")) return .flat;
    if (std.mem.eql(u8, v, "split")) return .split;
    return error.DecodeRowsSetting;
}

pub const n_layers = 40;
pub const n_experts: u32 = 384;
/// The most rows a layer gets (the curves' last count).
pub const max_rows: u32 = 256;
/// Each layer's record (logical bytes, the bill's measure) in the pack the curves come from.
pub const layer_bytes: [n_layers]u64 = blk: {
    var b: [n_layers]u64 = @splat(13315584);
    for ([_]usize{ 23, 32, 33, 34, 35, 36, 37, 38, 39 }) |l| b[l] = 8891904;
    break :blk b;
};
/// The slot-row counts the curves are measured at.
pub const grid = [_]u32{ 130, 136, 142, 148, 152, 158, 164, 170, 176, 182, 190, 200, 212, 226, 240, 256 };
/// Per layer, its decode misses over the development corpus's 1948 rounds at each `grid` count.
pub const curves = [n_layers][grid.len]u32{
    .{ 11697, 10991, 10284, 9620, 9202, 8612, 8049, 7551, 7046, 6600, 6047, 5392, 4683, 3963, 3328, 2763 },
    .{ 10184, 9664, 9100, 8657, 8378, 7915, 7457, 7047, 6662, 6248, 5770, 5188, 4547, 3875, 3303, 2857 },
    .{ 6708, 6249, 5813, 5371, 5153, 4800, 4402, 4072, 3805, 3524, 3221, 2850, 2490, 2104, 1819, 1569 },
    .{ 4199, 3886, 3595, 3327, 3134, 2890, 2662, 2476, 2294, 2118, 1932, 1714, 1508, 1338, 1228, 1155 },
    .{ 6201, 5784, 5430, 5077, 4841, 4518, 4239, 3960, 3693, 3442, 3131, 2788, 2458, 2125, 1885, 1668 },
    .{ 4741, 4392, 4095, 3826, 3682, 3434, 3221, 2995, 2808, 2609, 2382, 2178, 1923, 1677, 1508, 1382 },
    .{ 3472, 3227, 3009, 2824, 2697, 2519, 2352, 2200, 2044, 1900, 1750, 1582, 1438, 1298, 1194, 1133 },
    .{ 3178, 2929, 2686, 2485, 2348, 2142, 1969, 1822, 1672, 1546, 1421, 1285, 1169, 1080, 1023, 994 },
    .{ 3387, 3104, 2848, 2577, 2402, 2205, 2012, 1816, 1680, 1551, 1410, 1286, 1156, 1047, 993, 965 },
    .{ 3288, 3032, 2773, 2538, 2380, 2195, 2026, 1873, 1759, 1619, 1477, 1312, 1174, 1069, 1013, 982 },
    .{ 3702, 3424, 3169, 2931, 2790, 2594, 2399, 2215, 2037, 1899, 1737, 1530, 1367, 1237, 1162, 1106 },
    .{ 3399, 3145, 2895, 2692, 2544, 2348, 2195, 2033, 1880, 1752, 1608, 1456, 1293, 1177, 1094, 1052 },
    .{ 4896, 4532, 4144, 3850, 3654, 3389, 3130, 2902, 2683, 2490, 2258, 2006, 1771, 1561, 1407, 1293 },
    .{ 6367, 5898, 5478, 5108, 4898, 4549, 4215, 3932, 3667, 3432, 3155, 2835, 2497, 2177, 1921, 1704 },
    .{ 3572, 3294, 3019, 2794, 2667, 2465, 2307, 2124, 1979, 1844, 1678, 1534, 1414, 1291, 1199, 1129 },
    .{ 6952, 6445, 5987, 5571, 5308, 4929, 4590, 4292, 4038, 3760, 3456, 3080, 2671, 2241, 1976, 1730 },
    .{ 2460, 2256, 2061, 1884, 1786, 1636, 1501, 1418, 1313, 1237, 1163, 1077, 1009, 949, 921, 906 },
    .{ 4297, 3998, 3734, 3472, 3333, 3107, 2903, 2728, 2541, 2401, 2217, 2012, 1803, 1601, 1453, 1340 },
    .{ 3411, 3191, 2976, 2763, 2610, 2459, 2291, 2147, 2001, 1885, 1738, 1589, 1434, 1328, 1241, 1172 },
    .{ 9127, 8461, 7842, 7316, 6975, 6516, 6087, 5655, 5298, 4919, 4475, 3990, 3477, 2913, 2468, 2054 },
    .{ 2714, 2514, 2321, 2110, 2013, 1867, 1731, 1612, 1510, 1412, 1314, 1219, 1131, 1061, 1005, 987 },
    .{ 2801, 2603, 2438, 2240, 2138, 1991, 1842, 1702, 1601, 1512, 1397, 1285, 1193, 1116, 1060, 1039 },
    .{ 3412, 3143, 2908, 2705, 2572, 2368, 2176, 2013, 1878, 1779, 1635, 1484, 1327, 1193, 1117, 1069 },
    .{ 5741, 5312, 4907, 4550, 4337, 4035, 3765, 3539, 3314, 3087, 2837, 2511, 2191, 1898, 1671, 1496 },
    .{ 2201, 2027, 1895, 1772, 1701, 1594, 1479, 1398, 1307, 1242, 1146, 1069, 1002, 947, 924, 917 },
    .{ 1507, 1391, 1284, 1187, 1138, 1064, 1012, 952, 907, 867, 819, 781, 762, 747, 746, 746 },
    .{ 2385, 2215, 2052, 1899, 1818, 1697, 1575, 1464, 1357, 1265, 1187, 1090, 1004, 952, 921, 911 },
    .{ 3749, 3475, 3235, 3017, 2893, 2703, 2510, 2322, 2163, 2019, 1890, 1698, 1525, 1362, 1249, 1176 },
    .{ 2052, 1899, 1770, 1638, 1572, 1484, 1383, 1294, 1220, 1168, 1090, 1011, 937, 898, 881, 875 },
    .{ 3244, 2995, 2787, 2566, 2450, 2275, 2125, 1996, 1851, 1748, 1609, 1457, 1322, 1201, 1122, 1079 },
    .{ 3283, 3044, 2823, 2614, 2511, 2339, 2174, 2038, 1908, 1792, 1637, 1505, 1350, 1231, 1159, 1100 },
    .{ 3263, 2989, 2765, 2556, 2437, 2276, 2119, 1953, 1825, 1697, 1571, 1433, 1300, 1193, 1118, 1083 },
    .{ 3387, 3137, 2885, 2678, 2536, 2363, 2175, 2014, 1882, 1757, 1632, 1487, 1362, 1237, 1160, 1109 },
    .{ 3738, 3441, 3167, 2932, 2783, 2576, 2394, 2224, 2084, 1961, 1784, 1615, 1454, 1340, 1238, 1177 },
    .{ 3965, 3683, 3399, 3155, 3013, 2799, 2609, 2416, 2251, 2107, 1909, 1710, 1545, 1405, 1274, 1199 },
    .{ 3982, 3674, 3410, 3173, 3033, 2851, 2658, 2468, 2317, 2167, 2009, 1807, 1644, 1468, 1352, 1260 },
    .{ 3830, 3495, 3252, 3031, 2888, 2679, 2481, 2326, 2177, 2043, 1866, 1660, 1512, 1366, 1255, 1191 },
    .{ 5390, 5014, 4680, 4375, 4191, 3905, 3645, 3402, 3177, 2983, 2749, 2435, 2159, 1893, 1662, 1496 },
    .{ 5874, 5442, 5027, 4697, 4456, 4140, 3851, 3576, 3320, 3052, 2786, 2454, 2129, 1862, 1670, 1460 },
    .{ 6452, 6010, 5609, 5244, 5006, 4674, 4364, 4075, 3802, 3553, 3235, 2870, 2517, 2193, 1905, 1676 },
};

pub const Error = error{ DecodeRowsLayers, DecodeRowsExperts, DecodeRowsRecordBytes, DecodeRowsRange, DecodeRowsBudget };

pub const Split = struct {
    /// The split's bytes and the flat rows' (the bill's measure: rows x each layer's record).
    bytes: u64,
    flat_bytes: u64,
};

/// The misses one more row saves at layer `l` holding `r` rows: its curve's slope on the segment `r` is in (0 at
/// the last count).
fn gain(l: usize, r: u32) f64 {
    var i: usize = 0;
    while (i + 1 < grid.len and grid[i + 1] <= r) i += 1;
    if (i + 1 >= grid.len) return 0;
    const dm = @as(f64, @floatFromInt(curves[l][i])) - @as(f64, @floatFromInt(curves[l][i + 1]));
    return dm / @as(f64, @floatFromInt(grid[i + 1] - grid[i]));
}

/// `rows` (each layer's flat decode rows) split across the layers: at most the flat rows' bytes, each layer from
/// `max(prompt rows, grid[0])` to `max_rows`, a row at a time to the layer that saves the most misses a byte (ties:
/// the lower layer). `bytes`: each layer's record; `experts`: the bank's expert count. On any error `rows` is
/// unchanged (a geometry the curves do not fit, flat rows outside the curves' counts, prompt rows the flat bytes do not
/// cover).
pub fn split(rows: []u32, prompt: []const u32, bytes: []const u64, experts: u32) Error!Split {
    if (rows.len != n_layers or prompt.len != n_layers or bytes.len != n_layers) return error.DecodeRowsLayers;
    if (experts != n_experts) return error.DecodeRowsExperts;
    for (bytes, layer_bytes) |b, want| if (b != want) return error.DecodeRowsRecordBytes;
    var flat: u64 = 0;
    for (rows, bytes) |r, b| {
        if (r < grid[0] or r > max_rows) return error.DecodeRowsRange;
        flat += @as(u64, r) * b;
    }
    var out: [n_layers]u32 = undefined;
    var used: u64 = 0;
    for (&out, prompt, bytes) |*o, p, b| {
        o.* = @max(p, grid[0]);
        if (o.* > max_rows) return error.DecodeRowsRange;
        used += @as(u64, o.*) * b;
    }
    if (used > flat) return error.DecodeRowsBudget;
    while (true) {
        var best: f64 = 0;
        var pick: ?usize = null;
        for (0..n_layers) |l| {
            if (out[l] >= max_rows or used + bytes[l] > flat) continue;
            const g = gain(l, out[l]) / @as(f64, @floatFromInt(bytes[l]));
            if (g > best) {
                best = g;
                pick = l;
            }
        }
        const l = pick orelse break;
        out[l] += 1;
        used += bytes[l];
    }
    @memcpy(rows, &out);
    return .{ .bytes = used, .flat_bytes = flat };
}

// ── tests ──

const testing = std.testing;

test "decode rows setting: unset is auto, flat and split by name, anything else refused" {
    try testing.expectEqual(Mode.auto, try parse(null));
    try testing.expectEqual(Mode.auto, try parse(""));
    try testing.expectEqual(Mode.flat, try parse("flat"));
    try testing.expectEqual(Mode.split, try parse("split"));
    for ([_][]const u8{ "auto", "0", "1", "Split", "on", "flat " }) |v| try testing.expectError(error.DecodeRowsSetting, parse(v));
}

fn flatRows(r: u32) [n_layers]u32 {
    return @splat(r);
}

test "decode rows split: the corpus's split at the flat 152 and 176 rows' bytes, each layer in range" {
    const prompt = flatRows(130);
    const want152 = [n_layers]u32{ 240, 226, 170, 142, 170, 143, 130, 130, 136, 130, 136, 130, 152, 170, 142, 170, 130, 136, 130, 200, 130, 130, 130, 200, 130, 130, 130, 136, 130, 130, 130, 136, 152, 164, 170, 170, 164, 182, 200, 201 };
    const want176 = [n_layers]u32{ 256, 240, 212, 176, 200, 182, 152, 158, 170, 158, 170, 158, 182, 200, 158, 226, 142, 164, 152, 240, 148, 136, 164, 226, 130, 130, 130, 170, 130, 148, 148, 152, 182, 190, 200, 182, 200, 212, 212, 240 };
    for ([_]u32{ 152, 176 }, [_]*const [n_layers]u32{ &want152, &want176 }) |f, want| {
        var rows = flatRows(f);
        const s = try split(&rows, &prompt, &layer_bytes, n_experts);
        try testing.expectEqualSlices(u32, want, &rows);
        try testing.expect(s.bytes <= s.flat_bytes);
        var flat: u64 = 0;
        for (layer_bytes) |b| flat += f * b;
        try testing.expectEqual(flat, s.flat_bytes);
        for (rows) |r| try testing.expect(r >= 130 and r <= max_rows);
    }
}

test "decode rows split: prompt rows above the curves' first count are each layer's floor" {
    const prompt = flatRows(150);
    var rows = flatRows(160);
    _ = try split(&rows, &prompt, &layer_bytes, n_experts);
    for (rows) |r| try testing.expect(r >= 150 and r <= max_rows);
}

test "decode rows split: a geometry the curves do not fit is refused by name and the rows stay flat" {
    const prompt = flatRows(130);
    var rows = flatRows(152);
    try testing.expectError(error.DecodeRowsExperts, split(&rows, &prompt, &layer_bytes, 256));
    var other = layer_bytes;
    other[23] = 13315584;
    try testing.expectError(error.DecodeRowsRecordBytes, split(&rows, &prompt, &other, n_experts));
    try testing.expectError(error.DecodeRowsLayers, split(rows[0..39], prompt[0..39], layer_bytes[0..39], n_experts));
    try testing.expectEqualSlices(u32, &flatRows(152), &rows);
}

test "decode rows split: flat rows outside the curves' counts, or prompt rows the flat bytes do not cover, are refused" {
    var low = flatRows(120);
    try testing.expectError(error.DecodeRowsRange, split(&low, &flatRows(120), &layer_bytes, n_experts));
    var high = flatRows(300);
    try testing.expectError(error.DecodeRowsRange, split(&high, &flatRows(130), &layer_bytes, n_experts));
    var rows = flatRows(140);
    try testing.expectError(error.DecodeRowsBudget, split(&rows, &flatRows(150), &layer_bytes, n_experts));
    try testing.expectEqualSlices(u32, &flatRows(140), &rows);
}
