//! This plugin's own: a prefix-cache snapshot (src/cache/aux.zig's `Snap`) as the server's disk tier keeps it, MLX-free.
//! The header `export_arrays` writes, and `import_plan` and `import_arrays` read, holds every field of the snapshot but
//! its ids (the tier keeps the tokens) and its arrays, each array's dtype and shape; the arrays travel as the export's
//! tensors, one per array with an element, named by where it sits (`Slot`). A header names the geometry it was written
//! for (`Geometry.key`): a loaded instance with another refuses it.
//!
//! The header is the magic, then little-endian u32 words: the key (low, high) and `n`; the trunk's KV cache (its step,
//! entry count, and each entry's offset, initialized and arrays); its checkpoint (position, layer count, each layer's
//! fields and arrays); whether a head follows, then the head's KV cache, its checkpoint's position and layer, its meta
//! (stash length, origin, cache step) and its marks (count, each one's position and rows). An array is a presence word,
//! then its dtype, rank and dims.

const std = @import("std");
const mlx = @import("../mlx_serve/mlx/ffi.zig");

/// The header's first bytes; a change to its form takes a new one.
pub const magic = "TQ38A001";
/// The layers a header holds at most (the K packs have 48); the head's KV cache has one entry more.
pub const max_layers = 64;
pub const max_rank = 4;
/// The head's QSA marks and a layer's PLE history (`QSA_HEAD_MARKS_MAX`, `MAX_NGRAM_SIZE`: aux.zig checks them).
pub const max_marks = 32;
pub const ngram = 8;
/// The disk tier's bound on a header (`module_record.MAX_HEADER_BYTES`), which the largest plan's fits.
pub const max_header = 64 * 1024;
pub const max_name = 32;
/// The arrays a plan holds at most: both KV caches' keys and values, the trunk's layers' four, the head's, its marks.
pub const max_tensors = 2 * max_layers + 4 * max_layers + 2 * (max_layers + 1) + 4 + max_marks;
const max_dim: u32 = std.math.maxInt(c_int);

comptime {
    const desc_words = 3 + max_rank;
    const entry_words = 2 + 2 * desc_words;
    const layer_words = 4 + ngram + 4 * desc_words;
    const words = 3 + (2 + max_layers * entry_words) + (2 + max_layers * layer_words) + 1 +
        (2 + (max_layers + 1) * entry_words) + 1 + layer_words + 4 + max_marks * (1 + desc_words);
    if (magic.len + 4 * words > max_header) @compileError("the largest header outgrows the disk tier's");
}

/// The dtypes an array may have (the disk tier's), by their header code.
const dtypes = [_]mlx.mlx_dtype{ .bool_, .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64, .float16, .float32, .bfloat16 };
const widths = [_]u64{ 1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 2 };

fn code(dt: mlx.mlx_dtype) ?u32 {
    for (dtypes, 0..) |d, i| if (d == dt) return @intCast(i);
    return null;
}

/// An array of the snapshot: none (a null handle), or its dtype and shape.
pub const Desc = struct {
    present: bool = false,
    dtype: mlx.mlx_dtype = .float32,
    rank: u8 = 0,
    dims: [max_rank]u32 = @splat(0),

    pub fn shape(d: *const Desc) []const u32 {
        return d.dims[0..d.rank];
    }

    /// Present and with an element: the export lists it as a tensor (an empty array lives in the header alone).
    pub fn travels(d: *const Desc) bool {
        return d.present and std.mem.indexOfScalar(u32, d.shape(), 0) == null;
    }

    pub fn bytes(d: *const Desc) !u64 {
        var n: u64 = widths[code(d.dtype) orelse return error.AuxSnapshotArray];
        for (d.shape()) |v| n = std.math.mul(u64, n, v) catch return error.AuxSnapshotArray;
        return n;
    }
};

/// A KV cache entry (`KVCacheEntry`): its offset, whether a write initialized it, its keys' and values' arrays.
pub const Entry = struct {
    offset: u32 = 0,
    initialized: bool = false,
    keys: Desc = .{},
    values: Desc = .{},
};

/// A KV cache snapshot (`KVCacheSnapshot`): its step and entries, the trunk's one per layer, the head's one more.
pub const Kv = struct {
    step: u32 = 0,
    len: u32 = 0,
    entries: [max_layers + 1]Entry = @splat(.{}),
};

/// A layer's recurrent snapshot (`SSMCacheEntrySnapshot`): its flags, QSA counts, PLE history and arrays.
pub const Layer = struct {
    initialized: bool = false,
    ple_prev_valid: bool = false,
    qsa_ratio: i32 = 4,
    qsa_rows: i32 = 0,
    ple_prev: [ngram]u32 = @splat(0),
    conv: Desc = .{},
    state: Desc = .{},
    aux: Desc = .{},
    pooled: Desc = .{},
};

/// The head's history (`Drafter.Head`): its KV cache, its QSA checkpoint (position, one layer), its meta and marks.
pub const Head = struct {
    kv: Kv = .{},
    pos: u32 = 0,
    layer: Layer = .{},
    stash_len: u32 = 0,
    origin: i32 = 0,
    cache_step: u32 = 0,
    marks: u32 = 0,
    mark_pos: [max_marks]i32 = @splat(0),
    mark_rows: [max_marks]Desc = @splat(.{}),
};

/// The loaded instance a header must have been written for: its key (aux.zig hashes what a snapshot's arrays and fields
/// follow on it), its layer count and whether its pack has a head.
pub const Geometry = struct { key: u64, layers: u32, head: bool };

pub const Owner = enum { trunk, head };
pub const Field = enum { keys, values, conv, state, aux, pooled, mark };

/// Where an array sits in a snapshot: the trunk's or the head's, which field, and its entry, layer or mark.
pub const Slot = struct {
    owner: Owner,
    field: Field,
    index: u32 = 0,

    /// Its tensor's name, `owner.field.index` (`trunk.keys.3`).
    pub fn name(s: Slot, buf: *[max_name]u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}.{s}.{d}", .{ @tagName(s.owner), @tagName(s.field), s.index }) catch unreachable;
    }

    /// The slot a tensor's name spells, in its one spelling.
    pub fn parse(text: []const u8) ?Slot {
        var it = std.mem.splitScalar(u8, text, '.');
        const owner = std.meta.stringToEnum(Owner, it.next() orelse return null) orelse return null;
        const field = std.meta.stringToEnum(Field, it.next() orelse return null) orelse return null;
        const index = std.fmt.parseInt(u32, it.next() orelse return null, 10) catch return null;
        if (it.next() != null) return null;
        const s: Slot = .{ .owner = owner, .field = field, .index = index };
        var buf: [max_name]u8 = undefined;
        return if (std.mem.eql(u8, s.name(&buf), text)) s else null;
    }
};

/// A snapshot but its ids and arrays: what its header holds.
pub const Plan = struct {
    key: u64 = 0,
    n: u32 = 0,
    kv: Kv = .{},
    /// The trunk's checkpoint: its position and layers.
    pos: u32 = 0,
    layers: u32 = 0,
    ssm: [max_layers]Layer = @splat(.{}),
    head: ?Head = null,

    /// The array at `s`; null where the plan has no such slot (an index past its count, a field its owner lacks).
    pub fn desc(p: *const Plan, s: Slot) ?*const Desc {
        switch (s.owner) {
            .trunk => return switch (s.field) {
                .keys, .values => entryDesc(&p.kv, s),
                .mark => null,
                else => if (s.index < p.layers) layerDesc(&p.ssm[s.index], s.field) else null,
            },
            .head => {
                const h = if (p.head) |*hd| hd else return null;
                return switch (s.field) {
                    .keys, .values => entryDesc(&h.kv, s),
                    .mark => if (s.index < h.marks) &h.mark_rows[s.index] else null,
                    else => if (s.index == 0) layerDesc(&h.layer, s.field) else null,
                };
            },
        }
    }

    /// `v.visit(slot, desc)` for each array the snapshot holds, in the export's order.
    pub fn visit(p: *const Plan, v: anytype) !void {
        try kvVisit(v, .trunk, &p.kv);
        for (p.ssm[0..p.layers], 0..) |*l, i| try layerVisit(v, .trunk, l, @intCast(i));
        const h = if (p.head) |*hd| hd else return;
        try kvVisit(v, .head, &h.kv);
        try layerVisit(v, .head, &h.layer, 0);
        for (h.mark_rows[0..h.marks], 0..) |*d, i| try one(v, .{ .owner = .head, .field = .mark, .index = @intCast(i) }, d);
    }
};

fn entryDesc(kv: *const Kv, s: Slot) ?*const Desc {
    if (s.index >= kv.len) return null;
    const e = &kv.entries[s.index];
    return if (s.field == .keys) &e.keys else &e.values;
}

fn layerDesc(l: *const Layer, f: Field) ?*const Desc {
    return switch (f) {
        .conv => &l.conv,
        .state => &l.state,
        .aux => &l.aux,
        .pooled => &l.pooled,
        .keys, .values, .mark => null,
    };
}

fn kvVisit(v: anytype, owner: Owner, kv: *const Kv) !void {
    for (kv.entries[0..kv.len], 0..) |*e, i| {
        try one(v, .{ .owner = owner, .field = .keys, .index = @intCast(i) }, &e.keys);
        try one(v, .{ .owner = owner, .field = .values, .index = @intCast(i) }, &e.values);
    }
}

fn layerVisit(v: anytype, owner: Owner, l: *const Layer, i: u32) !void {
    inline for (.{ Field.conv, Field.state, Field.aux, Field.pooled }) |f| try one(v, .{ .owner = owner, .field = f, .index = i }, layerDesc(l, f).?);
}

fn one(v: anytype, s: Slot, d: *const Desc) !void {
    if (d.present) try v.visit(s, d);
}

/// Whether the loaded instance `g` takes `p`: its key and counts, its boundary, and its KV entries as a capture leaves
/// them.
pub fn check(p: *const Plan, g: Geometry) !void {
    if (p.key != g.key or p.kv.len != g.layers or p.layers != g.layers) return error.AuxSnapshotLayout;
    if (p.n == 0 or p.pos != p.n) return error.AuxSnapshotBoundary;
    try checkKv(&p.kv, p.n);
    const h = if (p.head) |*hd| hd else return;
    if (!g.head or h.kv.len != g.layers + 1) return error.AuxSnapshotLayout;
    try checkKv(&h.kv, h.kv.step);
}

/// A KV snapshot's entries as `trimmedCopy(len)` leaves them: at most `len` rows each, an initialized one holding its
/// keys and values `[B, H, offset, D]`, any other none.
fn checkKv(kv: *const Kv, len: u32) !void {
    if (kv.step > len) return error.AuxSnapshotBoundary;
    for (kv.entries[0..kv.len]) |*e| {
        if (e.offset > len) return error.AuxSnapshotBoundary;
        for ([_]*const Desc{ &e.keys, &e.values }) |d| {
            if (d.present != e.initialized) return error.AuxSnapshotKv;
            if (d.present and (d.rank != 4 or d.dims[2] != e.offset)) return error.AuxSnapshotKv;
        }
    }
}

/// `p`'s header in `buf`.
pub fn encode(p: *const Plan, buf: []u8) ![]const u8 {
    if (buf.len < magic.len) return error.AuxSnapshotStorage;
    @memcpy(buf[0..magic.len], magic);
    var w: Writer = .{ .buf = buf, .len = magic.len };
    try fields(&w, p);
    return buf[0..w.len];
}

/// The plan `bytes` holds, all of it and nothing past it.
pub fn decode(bytes: []const u8) !Plan {
    if (bytes.len < magic.len or !std.mem.eql(u8, bytes[0..magic.len], magic)) return error.AuxSnapshotHeader;
    var p: Plan = .{};
    var r: Reader = .{ .bytes = bytes, .at = magic.len };
    try fields(&r, &p);
    if (r.at != bytes.len) return error.AuxSnapshotHeader;
    return p;
}

/// The header's words in their order, written (`Writer`, from a `*const Plan`) or read (`Reader`, into a `*Plan`) by
/// the one walk.
fn fields(io: anytype, p: anytype) !void {
    const reads = @TypeOf(io.*).reads;
    try io.key(&p.key);
    try io.word(&p.n);
    try kvFields(io, &p.kv, max_layers);
    try io.word(&p.pos);
    try io.count(&p.layers, max_layers);
    for (p.ssm[0..p.layers]) |*l| try layerFields(io, l);
    var head = p.head != null;
    try io.flag(&head);
    if (!head) return;
    if (reads) p.head = .{};
    const h = &p.head.?;
    try kvFields(io, &h.kv, max_layers + 1);
    try io.word(&h.pos);
    try layerFields(io, &h.layer);
    try io.word(&h.stash_len);
    try io.int(&h.origin);
    try io.word(&h.cache_step);
    try io.count(&h.marks, max_marks);
    for (h.mark_pos[0..h.marks], h.mark_rows[0..h.marks]) |*pos, *d| {
        try io.int(pos);
        try descFields(io, d);
    }
}

fn kvFields(io: anytype, kv: anytype, max: u32) !void {
    try io.word(&kv.step);
    try io.count(&kv.len, max);
    for (kv.entries[0..kv.len]) |*e| {
        try io.word(&e.offset);
        try io.flag(&e.initialized);
        try descFields(io, &e.keys);
        try descFields(io, &e.values);
    }
}

fn layerFields(io: anytype, l: anytype) !void {
    try io.flag(&l.initialized);
    try io.flag(&l.ple_prev_valid);
    try io.int(&l.qsa_ratio);
    try io.int(&l.qsa_rows);
    for (&l.ple_prev) |*v| try io.word(v);
    inline for (.{ "conv", "state", "aux", "pooled" }) |f| try descFields(io, &@field(l, f));
}

fn descFields(io: anytype, d: anytype) !void {
    try io.flag(&d.present);
    if (!d.present) return;
    try io.dtype(&d.dtype);
    try io.rank(&d.rank);
    for (d.dims[0..d.rank]) |*v| try io.dim(v);
}

/// A plan's words into a header: a value the form cannot hold is refused.
const Writer = struct {
    const reads = false;
    buf: []u8,
    len: usize,

    fn put(w: *Writer, v: u32) !void {
        if (w.buf.len - w.len < 4) return error.AuxSnapshotStorage;
        std.mem.writeInt(u32, w.buf[w.len..][0..4], v, .little);
        w.len += 4;
    }
    fn key(w: *Writer, v: *const u64) !void {
        try w.put(@truncate(v.*));
        try w.put(@truncate(v.* >> 32));
    }
    fn word(w: *Writer, v: *const u32) !void {
        try w.put(v.*);
    }
    fn int(w: *Writer, v: *const i32) !void {
        try w.put(@bitCast(v.*));
    }
    fn flag(w: *Writer, v: *const bool) !void {
        try w.put(@intFromBool(v.*));
    }
    fn count(w: *Writer, v: *const u32, max: u32) !void {
        if (v.* > max) return error.AuxSnapshotArray;
        try w.put(v.*);
    }
    fn rank(w: *Writer, v: *const u8) !void {
        if (v.* > max_rank) return error.AuxSnapshotArray;
        try w.put(v.*);
    }
    fn dim(w: *Writer, v: *const u32) !void {
        if (v.* > max_dim) return error.AuxSnapshotArray;
        try w.put(v.*);
    }
    fn dtype(w: *Writer, v: *const mlx.mlx_dtype) !void {
        try w.put(code(v.*) orelse return error.AuxSnapshotArray);
    }
};

/// A header's words into a plan: a word out of its range, or a header that ends early, is refused.
const Reader = struct {
    const reads = true;
    bytes: []const u8,
    at: usize,

    fn take(r: *Reader) !u32 {
        if (r.bytes.len - r.at < 4) return error.AuxSnapshotHeader;
        defer r.at += 4;
        return std.mem.readInt(u32, r.bytes[r.at..][0..4], .little);
    }
    fn key(r: *Reader, v: *u64) !void {
        const lo = try r.take();
        v.* = @as(u64, try r.take()) << 32 | lo;
    }
    fn word(r: *Reader, v: *u32) !void {
        v.* = try r.take();
    }
    fn int(r: *Reader, v: *i32) !void {
        v.* = @bitCast(try r.take());
    }
    fn flag(r: *Reader, v: *bool) !void {
        v.* = switch (try r.take()) {
            0 => false,
            1 => true,
            else => return error.AuxSnapshotHeader,
        };
    }
    fn count(r: *Reader, v: *u32, max: u32) !void {
        v.* = try r.take();
        if (v.* > max) return error.AuxSnapshotHeader;
    }
    fn rank(r: *Reader, v: *u8) !void {
        const w = try r.take();
        if (w > max_rank) return error.AuxSnapshotHeader;
        v.* = @intCast(w);
    }
    fn dim(r: *Reader, v: *u32) !void {
        v.* = try r.take();
        if (v.* > max_dim) return error.AuxSnapshotHeader;
    }
    fn dtype(r: *Reader, v: *mlx.mlx_dtype) !void {
        const w = try r.take();
        if (w >= dtypes.len) return error.AuxSnapshotHeader;
        v.* = dtypes[w];
    }
};

/// Every array a plan holds, and of them the tensors (those with an element) and their bytes.
const Count = struct {
    arrays: u64 = 0,
    tensors: usize = 0,
    bytes: u64 = 0,

    pub fn visit(c: *Count, _: Slot, d: *const Desc) !void {
        c.arrays += 1;
        if (!d.travels()) return;
        c.tensors += 1;
        c.bytes = std.math.add(u64, c.bytes, try d.bytes()) catch return error.AuxSnapshotArray;
    }
};

/// The tensors an export of `p` lists.
pub fn tensorCount(p: *const Plan) !usize {
    var c: Count = .{};
    try p.visit(&c);
    return c.tensors;
}

/// What an import of `p` holds at once beyond the server's own state: the arrays the server loaded and the snapshot's
/// copies of them, and the host side, the ids and 4 KiB for the wrappers of each array, entry and layer.
pub fn peak(p: *const Plan) !u64 {
    var c: Count = .{};
    try p.visit(&c);
    var wrappers: u64 = c.arrays + p.kv.len + p.layers;
    if (p.head) |*h| wrappers += h.kv.len + 1 + h.marks;
    const arrays = std.math.mul(u64, c.bytes, 2) catch return error.AuxSnapshotArray;
    return std.math.add(u64, arrays, @as(u64, p.n) * 4 + wrappers * 4096) catch return error.AuxSnapshotArray;
}

const testing = std.testing;

fn testDesc(dt: mlx.mlx_dtype, dims: []const u32) Desc {
    var d: Desc = .{ .present = true, .dtype = dt, .rank = @intCast(dims.len) };
    @memcpy(d.dims[0..dims.len], dims);
    return d;
}

/// Each array's name reads back to its slot, and its slot to the array.
const Names = struct {
    p: *const Plan,

    pub fn visit(n: *Names, s: Slot, d: *const Desc) !void {
        var buf: [max_name]u8 = undefined;
        const back = Slot.parse(s.name(&buf)) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(s, back);
        try testing.expectEqual(d, n.p.desc(back).?);
    }
};

test "aux codec: a snapshot's header reads back field for field, from its own layout only, and names each array once" {
    // Four layers, attention every second: the trunk's KV on 1 and 3, the GDN pair on 0 and 2 with a PLE window on 2,
    // QSA leftovers and banks on 1 and 3 (layer 3's leftover empty), and a head: its KV entry, QSA layer and one mark.
    var p: Plan = .{ .key = 0x0123_4567_89ab_cdef, .n = 9, .pos = 9, .layers = 4 };
    p.kv.len = 4;
    for ([_]usize{ 1, 3 }) |l| p.kv.entries[l] = .{ .offset = 9, .initialized = true, .keys = testDesc(.bfloat16, &.{ 1, 2, 9, 8 }), .values = testDesc(.bfloat16, &.{ 1, 2, 9, 8 }) };
    for ([_]usize{ 0, 2 }) |l| p.ssm[l] = .{ .initialized = true, .conv = testDesc(.bfloat16, &.{ 1, 3, 24 }), .state = testDesc(.float32, &.{ 1, 4, 8, 8 }) };
    p.ssm[2].aux = testDesc(.bfloat16, &.{ 1, 6, 32 });
    p.ssm[2].ple_prev = .{ 5, 6, 0, 0, 0, 0, 0, 0 };
    p.ssm[2].ple_prev_valid = true;
    p.ssm[1] = .{ .qsa_rows = 9, .aux = testDesc(.bfloat16, &.{ 1, 1, 16 }), .pooled = testDesc(.bfloat16, &.{ 1, 2, 16 }) };
    p.ssm[3] = .{ .qsa_rows = 9, .aux = testDesc(.bfloat16, &.{ 1, 0, 16 }), .pooled = testDesc(.bfloat16, &.{ 1, 2, 16 }) };
    var h: Head = .{ .pos = 8, .stash_len = 8, .origin = -1, .cache_step = 8, .marks = 1 };
    h.kv.len = 5;
    h.kv.step = 8;
    h.kv.entries[4] = .{ .offset = 8, .initialized = true, .keys = testDesc(.bfloat16, &.{ 1, 2, 8, 8 }), .values = testDesc(.bfloat16, &.{ 1, 2, 8, 8 }) };
    h.layer = .{ .qsa_rows = 8, .aux = testDesc(.bfloat16, &.{ 1, 8, 16 }), .pooled = testDesc(.bfloat16, &.{ 1, 2, 16 }) };
    h.mark_pos[0] = 4;
    h.mark_rows[0] = testDesc(.bfloat16, &.{ 1, 4, 16 });
    p.head = h;
    const g: Geometry = .{ .key = p.key, .layers = 4, .head = true };
    try check(&p, g);

    var buf: [max_header]u8 = undefined;
    const bytes = try encode(&p, &buf);
    const back = try decode(bytes);
    try testing.expect(std.meta.eql(p, back));

    // Only the whole header reads: every shorter one, one with a word past it, a flag past 1 (entry 0's initialized),
    // another magic.
    for (0..bytes.len) |cut| try testing.expectError(error.AuxSnapshotHeader, decode(bytes[0..cut]));
    const len = bytes.len;
    @memset(buf[len..][0..4], 0);
    try testing.expectError(error.AuxSnapshotHeader, decode(buf[0 .. len + 4]));
    std.mem.writeInt(u32, buf[32..36], 2, .little);
    try testing.expectError(error.AuxSnapshotHeader, decode(buf[0..len]));
    std.mem.writeInt(u32, buf[32..36], 0, .little);
    buf[0] ^= 1;
    try testing.expectError(error.AuxSnapshotHeader, decode(buf[0..len]));

    // Another layout, layer count or a pack without a head refuses it, and so does a KV entry no capture leaves.
    try testing.expectError(error.AuxSnapshotLayout, check(&back, .{ .key = g.key ^ 1, .layers = 4, .head = true }));
    try testing.expectError(error.AuxSnapshotLayout, check(&back, .{ .key = g.key, .layers = 5, .head = true }));
    try testing.expectError(error.AuxSnapshotLayout, check(&back, .{ .key = g.key, .layers = 4, .head = false }));
    var bad = back;
    bad.kv.entries[3].values.dims[2] = 8;
    try testing.expectError(error.AuxSnapshotKv, check(&bad, g));

    // The tensors: the trunk's 4 KV and 8 recurrent arrays (layer 3's empty leftover lives in the header alone), the
    // head's 2 KV, its leftover and bank, and its mark; each name in its one spelling.
    try testing.expectEqual(@as(usize, 17), try tensorCount(&back));
    var names: Names = .{ .p = &back };
    try back.visit(&names);
    try testing.expect(Slot.parse("trunk.keys.03") == null);
    try testing.expect(Slot.parse("trunk.keys.3.0") == null);
    try testing.expect(back.desc(Slot.parse("trunk.mark.0").?) == null);
    try testing.expect(back.desc(.{ .owner = .head, .field = .conv, .index = 1 }) == null);
    // The peak covers the arrays the server loads and the snapshot's copies of them.
    try testing.expect(try peak(&back) >= 2 * 4992);
}
