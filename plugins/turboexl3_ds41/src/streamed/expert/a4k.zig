//! The a4k record layout of an EXL3 v2 expert bank (`sidecar.layout` "a4k" in expert-manifest-v2.json
//! ): each record's nine segments at offsets that put every code on a 4 KiB block, so the read pool's PARTS route
//! (direct parts) reads all three codes straight into their slot rows (the classic layout puts only the gate's code on
//! the block: the up and down codes sit 2,560 and 1,024 B off it). On disk: a zero lead of (-(the gate/up span's
//! scales)) mod 4096, the gate/up span [gate.rout gate.rin up.rout up.rin gate.code up.code], then the down span
//! [down.code down.rout down.rin], the record padded to 4096. Each span stays contiguous and the down span follows the
//! gate/up span, as the pool reads a span (its parts back to back) and pre-reads (the down range at the gate/up range's
//! end). The segments' lengths, dtypes and shapes are the classic layout's; only the offsets and the order on disk
//! differ. std only: the bank (expert/bank.zig), the read pool's records (the kit's `streamed.expert.io`) and the plugin's byte
//! gate (a4k_test.zig) share it.
const std = @import("std");

pub const n_components = 9;
pub const gu_components = 6;
pub const block: u64 = 4096;
/// The a4k order on disk of the components (bank.zig's `Component` order: gate code, rout, rin, up ..., down ...).
pub const disk_order = [n_components]u8{ 1, 2, 4, 5, 0, 3, 6, 7, 8 };
/// The components' own order: a classic bank's order on disk.
pub const identity = [n_components]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };

pub const Geometry = struct {
    /// Each component's offset in its record (component order).
    offsets: [n_components]u64,
    /// The zero bytes before the gate/up span.
    lead: u64,
    /// One past the record's last segment.
    extent: u64,
    /// The record in the file: `extent` padded to `block`.
    record_bytes: u64,
};

/// The a4k offsets of a record whose segments are `lengths` long (component order), or null where a code's length is not
/// a multiple of `block` (no code could then start on the block in every record).
pub fn geometry(lengths: [n_components]u64) ?Geometry {
    for ([_]usize{ 0, 3, 6 }) |c| if (lengths[c] == 0 or lengths[c] % block != 0) return null;
    var smalls: u64 = 0;
    for (disk_order[0..4]) |c| smalls += lengths[c];
    const lead = (block - smalls % block) % block;
    var g: Geometry = .{ .offsets = undefined, .lead = lead, .extent = 0, .record_bytes = 0 };
    var off = lead;
    for (disk_order) |c| {
        g.offsets[c] = off;
        off += lengths[c];
    }
    g.extent = off;
    g.record_bytes = std.mem.alignForward(u64, off, block);
    return g;
}

/// One record's values (one per component, component order) in the on-disk `order`.
pub fn toDisk(comptime T: type, comptime n: usize, x: *const [n]T, order: []const u8) [n]T {
    var out: [n]T = undefined;
    for (&out, order[0..n]) |*o, c| o.* = x[c];
    return out;
}

const testing = std.testing;

/// DeepSeek-V4.1's segment lengths at K (hidden 5,120, intermediate 2,304; bank.zig layerSegments).
fn dsv41Lengths(k: u64) [n_components]u64 {
    const code = (5120 / 16) * (2304 / 16) * 16 * k * 2;
    return .{ code, 2304 * 2, 5120 * 2, code, 2304 * 2, 5120 * 2, code, 5120 * 2, 2304 * 2 };
}

test "a4k puts every code of DeepSeek-V4.1's K = 3 and K = 2 records on the block, the spans contiguous, 4 KiB more a record" {
    for ([_]u64{ 3, 2 }) |k| {
        const lens = dsv41Lengths(k);
        const g = geometry(lens).?;
        try testing.expectEqual(@as(u64, 3072), g.lead);
        for ([_]usize{ 0, 3, 6 }) |c| try testing.expectEqual(@as(u64, 0), g.offsets[c] % block);
        // the gate/up span: its parts back to back from the lead, in disk order; the down span right after it
        var at = g.lead;
        for (disk_order) |c| {
            try testing.expectEqual(at, g.offsets[c]);
            at += lens[c];
        }
        try testing.expectEqual(at, g.extent);
        var logical: u64 = 0;
        for (lens) |n| logical += n;
        try testing.expectEqual(std.mem.alignForward(u64, logical, block) + block, g.record_bytes);
        // the six gate/up components first on disk, the three down ones after
        for (disk_order[0..gu_components]) |c| try testing.expect(c < gu_components);
        for (disk_order[gu_components..]) |c| try testing.expect(c >= gu_components);
    }
    try testing.expectEqual(@as(u64, 13_320_192), geometry(dsv41Lengths(3)).?.record_bytes);
    try testing.expectEqual(@as(u64, 8_896_512), geometry(dsv41Lengths(2)).?.record_bytes);
    // a code off the block has no a4k geometry
    var odd = dsv41Lengths(3);
    odd[3] += 2;
    try testing.expect(geometry(odd) == null);
}

test "a4k toDisk reorders one record's values and the identity order keeps them" {
    const x = [n_components]u64{ 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    try testing.expectEqualSlices(u64, &.{ 11, 12, 14, 15, 10, 13, 16, 17, 18 }, &toDisk(u64, n_components, &x, &disk_order));
    try testing.expectEqualSlices(u64, &x, &toDisk(u64, n_components, &x, &identity));
    // a permutation: every component once
    var seen: [n_components]bool = @splat(false);
    for (disk_order) |c| {
        try testing.expect(!seen[c]);
        seen[c] = true;
    }
}
