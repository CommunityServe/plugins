//! This plugin's own: the prefix cache's module snapshots (`CacheHooks.aux`) over the state the model keeps for each
//! slot: the trunk's KV cache, its recurrent entries (GDN pair, PLE window and history, QSA leftover and pooled bank) and
//! the MTP head's history, each copied as #1005's prefix cache copies them (`KVCacheSnapshot.trimmedCopy`,
//! `captureSsmCheckpoint` with the QSA history attached), so a later decode never writes into a buffer a snapshot holds.
//!
//! A prompt's snapshot is taken at the start of its prefill's final span: the forward that ends at the prompt's last
//! token, never inside one. With the server's prefix cache on it holds back 30 tokens (`SSM_SNAPSHOT_BACKOFF`), so the
//! boundary is where #1005's own end-of-prompt checkpoint sits, and a restore's tail is the cold run's final span, row
//! for row; with a one-row final span it is the ABI's `prompt.len - 1`. `describe` names the boundary taken.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const q4 = @import("../mlx_serve/plugins/qwen4_exp/plugin.zig");
const kv_cache = @import("../mlx_serve/sdk/kv/cache.zig");
const checkpoint = @import("../mlx_serve/engines/mlx/ssm/checkpoint.zig");
const qsa_checkpoint = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig");
const Drafter = @import("../spec/drafter.zig").Drafter;

const KVCacheSnapshot = kv_cache.KVCacheSnapshot;
const SSMCheckpoint = checkpoint.SSMCheckpoint;
const OwnSlot = @typeInfo(@FieldType(@FieldType(q4.Model, "own"), "items")).pointer.child;
const gpa = std.heap.c_allocator;

/// One prompt's state at its boundary `n`: the prompt's first `n` ids, the trunk's KV and recurrent entries there, and
/// the head's history (null when the slot kept none: its restore drafts nothing until the next cold prompt).
pub const Snap = struct {
    n: u64,
    ids: []u32,
    kv: KVCacheSnapshot,
    ssm: SSMCheckpoint,
    head: ?Drafter.Head = null,

    pub fn deinit(sn: *Snap) void {
        sn.kv.deinit();
        sn.ssm.deinit(gpa);
        if (sn.head) |*h| h.deinit();
        gpa.free(sn.ids);
    }

    /// Resident bytes, every held array counted once.
    pub fn bytes(sn: *const Snap) u64 {
        var total = kvBytes(&sn.kv) + checkpoint.ssmCheckpointBytes(&sn.ssm);
        if (sn.head) |*h| total += h.bytes();
        return total;
    }
};

/// Whether a forward of `rows` at `pos` is the final span of a prompt that ends at `end`, past its first token: the
/// snapshot's boundary.
pub fn capturesAt(pos: u64, rows: u32, end: u64) bool {
    return pos > 0 and pos + rows == end;
}

/// Slot `id`'s state at `n` (its trunk standing there), with `ids` the prompt's first `n`, copied.
pub fn capture(m: *q4.Model, d: *Drafter, id: u32, n: u64, ids: []const u32) !*Snap {
    const own = ownOf(m, id) orelse return error.AuxSlotGone;
    if (own.pos != n or ids.len != n) return error.AuxBoundary;
    const s = m.trunk.s;
    var shared = try own.cache.snapshot();
    defer shared.deinit();
    var kv = try shared.trimmedCopy(@intCast(n), s);
    errdefer kv.deinit();
    var cps = [1]SSMCheckpoint{try checkpoint.captureSsmCheckpoint(gpa, own.ssm, @intCast(n), s)};
    errdefer cps[0].deinit(gpa);
    try qsa_checkpoint.attachQsaHistoryToLatest(&cps, own.ssm, s);
    var head = try d.headSnap(id, n);
    errdefer if (head) |*h| h.deinit();
    const owned = try gpa.dupe(u32, ids);
    errdefer gpa.free(owned);
    const sn = try gpa.create(Snap);
    sn.* = .{ .n = n, .ids = owned, .kv = kv, .ssm = cps[0], .head = head };
    return sn;
}

/// Slot `id` from `sn`: the trunk's slot started over and given the snapshot's state (its arrays shared: the server
/// frees the snapshot once restored), standing at `sn.n`, and the head's history.
pub fn restore(m: *q4.Model, d: *Drafter, id: u32, sn: *const Snap) !void {
    const own = try m.ownSlot(id, 0);
    errdefer own.pos = 0;
    try own.cache.restore(&sn.kv);
    try checkpoint.restoreSsmCheckpoint(own.ssm, &sn.ssm);
    own.pos = sn.n;
    try d.headRestore(id, sn.n, if (sn.head) |*h| h else null);
}

/// A second snapshot of the same boundary (`slice` at the snapshot's own length), its arrays copied.
pub fn clone(sn: *const Snap) !*Snap {
    const s = mlx.gpuStream();
    var kv = try sn.kv.trimmedCopy(@intCast(sn.n), s);
    errdefer kv.deinit();
    var ssm = try checkpoint.shareSsmCheckpoint(gpa, &sn.ssm);
    errdefer ssm.deinit(gpa);
    var head = if (sn.head) |*h| try h.clone() else null;
    errdefer if (head) |*h| h.deinit();
    const owned = try gpa.dupe(u32, sn.ids);
    errdefer gpa.free(owned);
    const out = try gpa.create(Snap);
    out.* = .{ .n = sn.n, .ids = owned, .kv = kv, .ssm = ssm, .head = head };
    return out;
}

/// The bytes of a KV snapshot's arrays.
pub fn kvBytes(snap: *const KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |a| {
            if (a.ctx != null) total += @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
        }
    }
    return total;
}

fn ownOf(m: *q4.Model, id: u32) ?*OwnSlot {
    for (m.own.items) |*o| if (o.id == id) return o;
    return null;
}

const testing = std.testing;

test "aux: a prompt's snapshot is taken at the start of its final span, never inside a forward or at position 0" {
    // #1005's split, 8192 + 8192 + 1328 then a 31-row final span: only that span's start.
    try testing.expect(!capturesAt(0, 8192, 17743));
    try testing.expect(!capturesAt(16384, 1328, 17743));
    try testing.expect(capturesAt(17712, 31, 17743));
    // No backoff: the final span is the last token alone.
    try testing.expect(capturesAt(17742, 1, 17743));
    // A prompt the first forward covers whole has no boundary before it.
    try testing.expect(!capturesAt(0, 12, 12));
    // A decode past the prompt is never one.
    try testing.expect(!capturesAt(17743, 1, 17743));
}
