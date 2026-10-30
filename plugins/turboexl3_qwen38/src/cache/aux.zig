//! This plugin's own: the prefix cache's module snapshots (`CacheHooks.aux`) over the state the model keeps for each
//! slot: the trunk's KV cache, its recurrent entries (GDN pair, PLE window and history, QSA leftover and pooled bank) and
//! the MTP head's history, each copied as #1005's prefix cache copies them (`KVCacheSnapshot.trimmedCopy`,
//! `captureSsmCheckpoint` with the QSA history attached), so a later decode never writes into a buffer a snapshot holds.
//!
//! A prompt's snapshot is taken at the start of its prefill's final span: the forward that ends at the prompt's last
//! token, never inside one. With the server's prefix cache on it holds back 30 tokens (`SSM_SNAPSHOT_BACKOFF`), so the
//! boundary is where #1005's own end-of-prompt checkpoint sits, and a restore's tail is the cold run's final span, row
//! for row; with a one-row final span it is the ABI's `prompt.len - 1`. `describe` names the boundary taken.
//!
//! The server's disk tier (`--prefix-cache-disk`, and `--shutdown-persist-secs` at exit) keeps a snapshot in
//! src/cache/codec.zig's form: `export_arrays` hands it the header and a reference to each array, and `import_arrays`
//! builds the snapshot back from them, field for field, so a prompt resumed from disk runs as one resumed from RAM.

const std = @import("std");
const abi = @import("mlx_serve").abi.plugin;
const mlx = @import("../mlx_serve/mlx/ffi.zig");
const q4 = @import("../direct/serve.zig");
const state = @import("../direct/cache.zig");
const qsa_checkpoint = @import("../mlx_serve/plugins/qwen4_exp/qsa/checkpoint.zig");
const QSA_RING_ROWS = @import("../mlx_serve/plugins/qwen4_exp/qsa/history.zig").QSA_RING_ROWS;
const Drafter = @import("../spec/drafter.zig").Drafter;
const codec = @import("codec.zig");

const KVCacheSnapshot = state.KVCacheSnapshot;
const KVCacheEntry = state.KVCacheEntry;
const SSMCheckpoint = state.SSMCheckpoint;
const SSMCacheEntrySnapshot = state.SSMCacheEntrySnapshot;
const OwnSlot = @typeInfo(@FieldType(@FieldType(q4.Model, "own"), "items")).pointer.child;
const gpa = std.heap.c_allocator;
const none: mlx.mlx_array = .{ .ctx = null };

comptime {
    if (@typeInfo(@FieldType(SSMCacheEntrySnapshot, "ple_prev")).array.len != codec.ngram) @compileError("codec.ngram differs from a snapshot's PLE history");
    if (@typeInfo(@FieldType(@FieldType(Drafter.Head, "marks"), "items")).array.len != codec.max_marks) @compileError("codec.max_marks differs from the head's marks");
}

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
        var total = kvBytes(&sn.kv) + state.ssmCheckpointBytes(&sn.ssm);
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
    var cps = [1]SSMCheckpoint{try state.captureSsmCheckpoint(gpa, own.ssm, @intCast(n), s)};
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
    try state.restoreSsmCheckpoint(own.ssm, &sn.ssm);
    own.pos = sn.n;
    try d.headRestore(id, sn.n, if (sn.head) |*h| h else null);
}

/// A second snapshot of the same boundary (`slice` at the snapshot's own length), its arrays copied.
pub fn clone(sn: *const Snap) !*Snap {
    const s = mlx.gpuStream();
    var kv = try sn.kv.trimmedCopy(@intCast(sn.n), s);
    errdefer kv.deinit();
    var ssm = try state.shareSsmCheckpoint(gpa, &sn.ssm);
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
        inline for (.{ e.keys, e.values }) |a| {
            if (a.ctx != null) total += @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
        }
    }
    return total;
}

fn ownOf(m: *q4.Model, id: u32) ?*OwnSlot {
    for (m.own.items) |*o| if (o.id == id) return o;
    return null;
}

// ── The disk tier: a snapshot in src/cache/codec.zig's form ──

/// The loaded instance a snapshot's header must name: what every array and field of a snapshot follows on it (the layer
/// kinds, the KV, GDN, PLE and QSA dims and dtypes, the copies' QSA ring, the head), hashed, its layers and its head.
pub fn geometry(m: *const q4.Model) codec.Geometry {
    const c = &m.cfg;
    const head = m.trunk.qwen4_mtp != null;
    var h = std.hash.Wyhash.init(0x7138_a001);
    std.hash.autoHash(&h, .{
        c.num_hidden_layers,     c.full_attention_interval, c.linear_attn_tail_from, c.num_key_value_heads,
        c.head_dim,              c.actDtype(),              c.linear_num_key_heads,  c.linear_num_value_heads,
        c.linear_key_head_dim,   c.linear_value_head_dim,   c.linear_conv_kernel_dim, c.ssmStateDtype(),
        c.hidden_size,           c.hc_count,                c.ple_layer_idx,         c.ple_embed_dim,
        c.ple_conv_kernel,       c.ngram_size,              c.indexer_n_heads,       c.indexer_head_dim,
        c.indexer_budget,        c.indexer_compress_ratio,  QSA_RING_ROWS,           head,
    });
    return .{ .key = h.final(), .layers = c.num_hidden_layers, .head = head };
}

/// `import_plan`: the peak an import of `header` holds, once the header is one this instance takes. No allocation.
pub fn importPlan(m: *const q4.Model, header: []const u8) !u64 {
    const p = try codec.decode(header);
    try codec.check(&p, geometry(m));
    return codec.peak(&p);
}

/// `export_arrays`: `sn`'s header, and a new reference to each of its arrays with an element, named by where it sits,
/// into the server's storage. A failure leaves no reference behind and the lengths zero.
pub fn exportArrays(m: *const q4.Model, sn: *Snap, out: *abi.AuxExport) !void {
    out.tensor_len = 0;
    out.names_len = 0;
    out.header_len = 0;
    const tensors = out.tensors orelse return error.AuxSnapshotStorage;
    const names = out.names orelse return error.AuxSnapshotStorage;
    const header_buf = out.header orelse return error.AuxSnapshotStorage;
    const p = try planOf(m, sn);
    const header = try codec.encode(&p, header_buf[0..out.header_cap]);
    var w: Export = .{ .sn = sn, .tensors = tensors[0..out.tensor_cap], .names = names[0..out.names_cap] };
    errdefer for (w.tensors[0..w.len]) |t| {
        _ = mlx.mlx_array_free(.{ .ctx = t.array.ctx });
    };
    try p.visit(&w);
    out.tensor_len = w.len;
    out.names_len = w.used;
    out.header_len = header.len;
}

/// `import_arrays`: the snapshot `header` states, its boundary `ids`. The header is checked against this instance and
/// each of `arrays` (borrowed) against the header before anything is built; then each is copied onto the model's stream
/// (an empty one made anew) and evaluated: the snapshot exported, field for field. The slot's KV cache grows past it at
/// its first write, as after a RAM restore.
pub fn importArrays(m: *const q4.Model, ids: []const u32, header: []const u8, arrays: []const abi.AuxTensor) !*Snap {
    const p = try codec.decode(header);
    try codec.check(&p, geometry(m));
    if (ids.len != p.n) return error.AuxSnapshotBoundary;
    try checkArrays(&p, arrays);
    const sn = try gpa.create(Snap);
    sn.* = .{ .n = p.n, .ids = &.{}, .kv = .{ .entries = &.{}, .step = p.kv.step, .allocator = gpa }, .ssm = .{ .pos = p.pos, .layers = &.{} } };
    errdefer {
        sn.deinit();
        gpa.destroy(sn);
    }
    sn.ids = try gpa.dupe(u32, ids);
    sn.kv.entries = try kvEntries(&p.kv);
    sn.ssm.layers = try ssmLayers(p.ssm[0..p.layers]);
    if (p.head) |*h| {
        sn.head = .{
            .kv = .{ .entries = &.{}, .step = h.kv.step, .allocator = gpa },
            .entry = .{ .pos = h.pos, .layers = &.{} },
            .meta = .{ .stash_len = h.stash_len, .origin = h.origin, .cache_step = h.cache_step },
            .marks = .{},
        };
        const hd = &sn.head.?;
        hd.kv.entries = try kvEntries(&h.kv);
        hd.entry.layers = try ssmLayers((&h.layer)[0..1]);
        for (hd.marks.items[0..h.marks], h.mark_pos[0..h.marks]) |*mk, pos| mk.pos = pos;
        hd.marks.len = h.marks;
    }
    const s = m.trunk.s;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrays) |t| {
        const dst = arrayOf(sn, codec.Slot.parse(t.name.slice()).?);
        try mlx.check(mlx.mlx_copy(dst, .{ .ctx = t.array.ctx }, s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, dst.*));
    }
    var empty: Empty = .{ .sn = sn, .s = s, .vec = vec };
    try p.visit(&empty);
    if (mlx.mlx_vector_array_size(vec) > 0) try mlx.check(mlx.mlx_eval(vec));
    return sn;
}

/// `sn` as its header states it: every field but the ids, each array's dtype and shape, checked as an import checks it.
fn planOf(m: *const q4.Model, sn: *const Snap) !codec.Plan {
    const g = geometry(m);
    var p: codec.Plan = .{ .key = g.key, .n = try toWord(sn.n), .pos = try toWord(sn.ssm.pos) };
    try kvPlan(&p.kv, &sn.kv);
    if (sn.ssm.layers.len > p.ssm.len) return error.AuxSnapshotArray;
    p.layers = @intCast(sn.ssm.layers.len);
    for (sn.ssm.layers, p.ssm[0..p.layers]) |*l, *out| out.* = try layerPlan(l);
    if (sn.head) |*h| {
        if (h.entry.layers.len != 1) return error.AuxSnapshotArray;
        p.head = .{
            .pos = try toWord(h.entry.pos),
            .layer = try layerPlan(&h.entry.layers[0]),
            .stash_len = try toWord(h.meta.stash_len),
            .origin = h.meta.origin,
            .cache_step = try toWord(h.meta.cache_step),
            .marks = @intCast(h.marks.len),
        };
        const hp = &p.head.?;
        try kvPlan(&hp.kv, &h.kv);
        for (h.marks.slice(), hp.mark_pos[0..hp.marks], hp.mark_rows[0..hp.marks]) |mk, *pos, *rows| {
            pos.* = mk.pos;
            rows.* = try descOf(mk.rows);
        }
    }
    try codec.check(&p, g);
    return p;
}

fn kvPlan(out: *codec.Kv, kv: *const KVCacheSnapshot) !void {
    if (kv.entries.len > out.entries.len) return error.AuxSnapshotArray;
    out.step = try toWord(kv.step);
    out.len = @intCast(kv.entries.len);
    for (kv.entries, out.entries[0..out.len]) |*e, *o| o.* = .{
        .offset = try toWord(e.offset),
        .initialized = e.initialized,
        .keys = try descOf(e.keys),
        .values = try descOf(e.values),
    };
}

fn layerPlan(l: *const SSMCacheEntrySnapshot) !codec.Layer {
    return .{
        .initialized = l.initialized,
        .ple_prev_valid = l.ple_prev_valid,
        .qsa_ratio = l.qsa_ratio,
        .qsa_rows = l.qsa_rows,
        .ple_prev = l.ple_prev,
        .conv = try descOf(l.conv_state),
        .state = try descOf(l.ssm_state),
        .aux = try descOf(l.aux_state),
        .pooled = try descOf(l.qsa_pooled),
    };
}

/// An array's dtype and shape; a null handle is none.
fn descOf(x: mlx.mlx_array) !codec.Desc {
    if (x.ctx == null) return .{};
    const sh = mlx.getShape(x);
    if (sh.len > codec.max_rank) return error.AuxSnapshotArray;
    var d: codec.Desc = .{ .present = true, .dtype = mlx.mlx_array_dtype(x), .rank = @intCast(sh.len) };
    for (sh, d.dims[0..sh.len]) |v, *o| o.* = std.math.cast(u32, v) orelse return error.AuxSnapshotArray;
    return d;
}

fn toWord(v: usize) !u32 {
    return std.math.cast(u32, v) orelse error.AuxSnapshotArray;
}

/// Each of `arrays` a tensor of `p` by its name, named once, with that tensor's dtype and shape, and none missing.
fn checkArrays(p: *const codec.Plan, arrays: []const abi.AuxTensor) !void {
    if (arrays.len != try codec.tensorCount(p)) return error.AuxSnapshotTensors;
    var seen: [codec.max_tensors]*const codec.Desc = undefined;
    for (arrays, 0..) |t, i| {
        if (t.size < @sizeOf(abi.AuxTensor) or t.array.ctx == null) return error.AuxSnapshotTensorShape;
        const slot = codec.Slot.parse(t.name.slice()) orelse return error.AuxSnapshotTensorName;
        const d = p.desc(slot) orelse return error.AuxSnapshotTensorName;
        if (!d.travels()) return error.AuxSnapshotTensorName;
        if (std.mem.indexOfScalar(*const codec.Desc, seen[0..i], d) != null) return error.AuxSnapshotTensorTwice;
        if (!std.meta.eql(try descOf(.{ .ctx = t.array.ctx }), d.*)) return error.AuxSnapshotTensorShape;
        seen[i] = d;
    }
}

/// A KV snapshot's entries as `kv` states them, their arrays none until the import sets them.
fn kvEntries(kv: *const codec.Kv) ![]KVCacheEntry {
    const out = try gpa.alloc(KVCacheEntry, kv.len);
    for (kv.entries[0..kv.len], out) |*e, *o| o.* = .{ .keys = none, .values = none, .key_view = none, .value_view = none, .offset = e.offset, .initialized = e.initialized };
    return out;
}

/// A checkpoint's layers as `layers` state them, their arrays none until the import sets them.
fn ssmLayers(layers: []const codec.Layer) ![]SSMCacheEntrySnapshot {
    const out = try gpa.alloc(SSMCacheEntrySnapshot, layers.len);
    for (layers, out) |*l, *o| o.* = .{
        .conv_state = none,
        .ssm_state = none,
        .initialized = l.initialized,
        .qsa_ratio = l.qsa_ratio,
        .qsa_rows = l.qsa_rows,
        .ple_prev = l.ple_prev,
        .ple_prev_valid = l.ple_prev_valid,
    };
    return out;
}

/// Where `slot`'s array sits in `sn` (a slot of `sn`'s own plan).
fn arrayOf(sn: *Snap, slot: codec.Slot) *mlx.mlx_array {
    const i = slot.index;
    switch (slot.owner) {
        .trunk => return switch (slot.field) {
            .keys => &sn.kv.entries[i].keys,
            .values => &sn.kv.entries[i].values,
            .mark => unreachable,
            else => layerArray(&sn.ssm.layers[i], slot.field),
        },
        .head => {
            const h = &sn.head.?;
            return switch (slot.field) {
                .keys => &h.kv.entries[i].keys,
                .values => &h.kv.entries[i].values,
                .mark => &h.marks.items[i].rows,
                else => layerArray(&h.entry.layers[0], slot.field),
            };
        },
    }
}

fn layerArray(l: *SSMCacheEntrySnapshot, f: codec.Field) *mlx.mlx_array {
    return switch (f) {
        .conv => &l.conv_state,
        .state => &l.ssm_state,
        .aux => &l.aux_state,
        .pooled => &l.qsa_pooled,
        .keys, .values, .mark => unreachable,
    };
}

/// `export_arrays`' walk: each array with an element named into `names` and referenced into `tensors`.
const Export = struct {
    sn: *Snap,
    tensors: []abi.AuxTensor,
    names: []u8,
    len: usize = 0,
    used: usize = 0,

    pub fn visit(w: *Export, slot: codec.Slot, d: *const codec.Desc) !void {
        if (!d.travels()) return;
        var buf: [codec.max_name]u8 = undefined;
        const name = slot.name(&buf);
        if (w.len == w.tensors.len or w.names.len - w.used < name.len) return error.AuxSnapshotStorage;
        const dst = w.names[w.used..][0..name.len];
        @memcpy(dst, name);
        var x = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_array_set(&x, arrayOf(w.sn, slot).*));
        w.tensors[w.len] = .{ .name = .of(dst), .array = .{ .ctx = x.ctx } };
        w.len += 1;
        w.used += name.len;
    }
};

/// `import_arrays`' walk over the arrays with no element, which travel in the header alone: each made anew.
const Empty = struct {
    sn: *Snap,
    s: mlx.mlx_stream,
    vec: mlx.mlx_vector_array,

    pub fn visit(e: *Empty, slot: codec.Slot, d: *const codec.Desc) !void {
        if (d.travels()) return;
        var dims: [codec.max_rank]c_int = undefined;
        for (d.shape(), dims[0..d.rank]) |v, *o| o.* = @intCast(v);
        const dst = arrayOf(e.sn, slot);
        try mlx.check(mlx.mlx_zeros(dst, &dims, d.rank, d.dtype, e.s));
        try mlx.check(mlx.mlx_vector_array_append_value(e.vec, dst.*));
    }
};

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
