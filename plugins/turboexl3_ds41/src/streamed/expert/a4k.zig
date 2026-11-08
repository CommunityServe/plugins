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

/// The read pool's order on disk for `bank` (`expert_io.Options.disk_order`, `max` components, the rest 0): the bank's
/// own (`Bank.diskOrder`: an a4k bank's `disk_order`), or null (a classic bank, or a bank type without the hook) for the
/// components' order. The stream's `Stream.init` sets its pool options from it, so `Records.onDisk` lays each record's
/// rows and lengths out in the file's order.
pub fn poolDiskOrder(comptime Bank: type, bank: *const Bank, comptime max: usize) ?[max]u8 {
    if (comptime !@hasDecl(Bank, "diskOrder")) return null;
    const d = bank.diskOrder() orelse return null;
    var order: [max]u8 = @splat(0);
    @memcpy(order[0..d.len], &d);
    return order;
}

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
