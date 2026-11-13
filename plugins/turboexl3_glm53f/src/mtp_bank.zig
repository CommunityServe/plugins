//! The MTP layer's routed experts (model layer 45) as one resident turboexl3 bank, read from the pack's own one-layer
//! bank (`mtp-experts.bin`, its `mtp-expert-manifest-v2.json`; the converter's LAYOUT.md): the record format of the routed
//! bank, nine segments a record (each projection's code, rout = exllamav3 svh, rin = suh), K by its layer entry. The head
//! runs it through turboexl3's `moeClamped` (`swiglu`, the block's format banks in mlx-serve's glm5_next).
//! - `Manifest.parse` (MLX-free): the manifest's dims, quantization and layer entry, every segment checked against the
//!   record geometry its K gives (`geometry`), the sidecar's size against 288 records.
//! - `readProj`: one projection's components of the 288 records, each read with pread into one host buffer, then one
//!   MLX array: code U16 [288, in/16, out/16, n] (the stored I16 words), rout and rin F16 [288, width].

const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const crt = @import("host/crt.zig");

pub const manifest_file = "mtp-expert-manifest-v2.json";
pub const sidecar_file = "mtp-experts.bin";
pub const record_alignment: u64 = 4096;
/// mul1's multiplier (exllamav3's codebook; the converter checks every `.mul1` against it and stores it here once).
pub const mul1_multiplier: u64 = 0x83DCD12D;

pub const Component = enum(u8) { gate_code, gate_rout, gate_rin, up_code, up_rout, up_rin, down_code, down_rout, down_rin };
pub const n_components = 9;

pub const Segment = struct { offset: u64, length: u64, shape: [3]u64, rank: u8 };

/// One record's nine segments at `n` halfwords a tile (16 K) for (hidden, inter): per projection its code I16
/// [in/16, out/16, n], rout F16 [out], rin F16 [in], back to back; the record padded to `record_alignment`.
pub const Geometry = struct {
    n: u32,
    hidden: u64,
    inter: u64,
    segments: [n_components]Segment,
    logical_bytes: u64,
    record_bytes: u64,
};

pub fn geometry(n: u32, hidden: u64, inter: u64) ?Geometry {
    if (n < 16 or n > 128 or n % 2 != 0 or hidden == 0 or inter == 0 or hidden % 16 != 0 or inter % 16 != 0) return null;
    var g: Geometry = .{ .n = n, .hidden = hidden, .inter = inter, .segments = undefined, .logical_bytes = 0, .record_bytes = 0 };
    var off: u64 = 0;
    for (0..3) |p| {
        const in: u64 = if (p == 2) inter else hidden;
        const out: u64 = if (p == 2) hidden else inter;
        const parts = [3]Segment{
            .{ .offset = 0, .length = (in / 16) * (out / 16) * n * 2, .shape = .{ in / 16, out / 16, n }, .rank = 3 },
            .{ .offset = 0, .length = out * 2, .shape = .{ out, 0, 0 }, .rank = 1 },
            .{ .offset = 0, .length = in * 2, .shape = .{ in, 0, 0 }, .rank = 1 },
        };
        for (parts, 0..) |seg, j| {
            g.segments[p * 3 + j] = seg;
            g.segments[p * 3 + j].offset = off;
            off += seg.length;
        }
    }
    g.logical_bytes = off;
    g.record_bytes = std.mem.alignForward(u64, off, record_alignment);
    return g;
}

/// A v2 layer's K (an integer or exllamav3's half-integer rate) as halfwords a tile, or null.
pub fn halfwordsOfK(k: f64) ?u32 {
    if (!(k >= 1.0 and k <= 8.0)) return null;
    const x = k * 16.0;
    const n: u32 = @intFromFloat(x);
    if (@as(f64, @floatFromInt(n)) != x or n % 2 != 0) return null;
    return n;
}

pub const Error = error{ Glm53fMtpManifest, Glm53fMtpGeometry, Glm53fMtpCodebook };

/// The manifest as the bank reads it: one layer of `n_experts` records at `geometry`, from `base_offset`.
pub const Manifest = struct {
    n_experts: u32,
    geometry: Geometry,
    base_offset: u64,
    sidecar_size: u64,

    const SegJson = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
    const Json = struct {
        format: []const u8,
        quantization: struct { mode: []const u8, codebook: []const u8, codebook_multiplier: u64 },
        dims: struct { hidden: u64, inter: u64, n_experts: u64, n_layers: u64 },
        sidecar: struct { file: []const u8, alignment: u64, size: u64 },
        layers: []const struct { layer: u64, K: f64, record_bytes: u64, logical_bytes: u64, base_offset: u64, segments: []const SegJson },
    };

    /// The manifest's text against the geometry it declares (`hidden` / `inter`: the config's); refused by name.
    pub fn parse(a: std.mem.Allocator, text: []const u8, hidden: u64, inter: u64, n_experts: u32) Error!Manifest {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const j = std.json.parseFromSliceLeaky(Json, arena.allocator(), text, .{ .ignore_unknown_fields = true }) catch return error.Glm53fMtpManifest;
        if (!std.mem.eql(u8, j.format, "mtplx-expert-manifest-v2")) return error.Glm53fMtpManifest;
        if (!std.mem.eql(u8, j.quantization.mode, "exl3") or !std.mem.eql(u8, j.quantization.codebook, "mul1") or j.quantization.codebook_multiplier != mul1_multiplier) return error.Glm53fMtpCodebook;
        if (j.dims.hidden != hidden or j.dims.inter != inter or j.dims.n_experts != n_experts or j.dims.n_layers != 1 or j.layers.len != 1) return error.Glm53fMtpGeometry;
        if (!std.mem.eql(u8, j.sidecar.file, sidecar_file) or j.sidecar.alignment != record_alignment) return error.Glm53fMtpManifest;
        const l = j.layers[0];
        const n = halfwordsOfK(l.K) orelse return error.Glm53fMtpGeometry;
        const g = geometry(n, hidden, inter) orelse return error.Glm53fMtpGeometry;
        if (l.layer != 0 or l.record_bytes != g.record_bytes or l.logical_bytes != g.logical_bytes or l.segments.len != n_components) return error.Glm53fMtpGeometry;
        for (l.segments, g.segments, 0..) |sj, sg, c| {
            const name = @tagName(@as(Component, @enumFromInt(c)));
            // the manifest names components `gate_proj.code`, ...: projection, then part
            var want: [32]u8 = undefined;
            const us = std.mem.indexOfScalar(u8, name, '_').?;
            const wname = std.fmt.bufPrint(&want, "{s}_proj.{s}", .{ name[0..us], name[us + 1 ..] }) catch unreachable;
            if (!std.mem.eql(u8, sj.component, wname)) return error.Glm53fMtpGeometry;
            if (!std.mem.eql(u8, sj.dtype, if (sg.rank == 3) "I16" else "F16")) return error.Glm53fMtpGeometry;
            if (sj.offset != sg.offset or sj.length != sg.length or sj.shape.len != sg.rank) return error.Glm53fMtpGeometry;
            for (sj.shape, sg.shape[0..sg.rank]) |x, y| if (x != y) return error.Glm53fMtpGeometry;
        }
        const total = std.math.mul(u64, g.record_bytes, n_experts) catch return error.Glm53fMtpGeometry;
        if (l.base_offset != 0 or j.sidecar.size != total) return error.Glm53fMtpGeometry;
        return .{ .n_experts = n_experts, .geometry = g, .base_offset = l.base_offset, .sidecar_size = j.sidecar.size };
    }

    /// Record `e`'s segment `c`: its file offset.
    pub fn offsetOf(m: *const Manifest, e: u32, c: usize) u64 {
        return m.base_offset + @as(u64, e) * m.geometry.record_bytes + m.geometry.segments[c].offset;
    }

    /// The bytes the resident bank holds.
    pub fn residentBytes(m: *const Manifest) u64 {
        var n: u64 = 0;
        for (m.geometry.segments) |s| n += s.length;
        return n * m.n_experts;
    }
};

/// One projection's three arrays (component order: code, rout = svh, rin = suh), each a new reference the caller owns.
pub const Proj = [3]mlx.mlx_array;

/// Projection `proj`'s (0 gate, 1 up, 2 down) segments of every record in `dir` (`m` its parsed manifest), one
/// component at a time, evaluated.
pub fn readProj(a: std.mem.Allocator, dir: []const u8, m: *const Manifest, proj: usize) !Proj {
    if (proj > 2) return error.Glm53fMtpGeometry;
    var pb: [std.fs.max_path_bytes:0]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ dir, sidecar_file }, 0);
    const fd = crt.openRead(path);
    if (fd < 0) return error.Glm53fMtpBankOpen;
    defer crt.close(fd);
    if (@as(u64, @intCast(@max(crt.seekEnd(fd), 0))) < m.sidecar_size) return error.Glm53fMtpBankShort;
    var out: Proj = @splat(.{ .ctx = null });
    errdefer for (out) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    for (&out, proj * 3..) |*o, c| {
        const seg = m.geometry.segments[c];
        const len: usize = @intCast(seg.length);
        const buf = try a.alloc(u8, len * m.n_experts);
        defer a.free(buf);
        for (0..m.n_experts) |e| {
            var got: usize = 0;
            const dst = buf[e * len ..][0..len];
            while (got < len) {
                const r = crt.pread(fd, dst[got..].ptr, len - got, @intCast(m.offsetOf(@intCast(e), c) + got));
                if (r <= 0) return error.Glm53fMtpBankRead;
                got += @intCast(r);
            }
        }
        var shape: [4]c_int = undefined;
        shape[0] = @intCast(m.n_experts);
        for (0..seg.rank) |d| shape[d + 1] = @intCast(seg.shape[d]);
        const dt: mlx.mlx_dtype = if (seg.rank == 3) .uint16 else .float16;
        o.* = mlx.mlx_array_new_data(buf.ptr, &shape, @intCast(seg.rank + 1), dt);
        try mlx.check(mlx.mlx_array_eval(o.*));
    }
    return out;
}

// ── tests ──

const testing = std.testing;

/// A one-layer manifest at (hidden, inter) and K `k`, as the converter writes it (LAYOUT.md).
fn manifestText(a: std.mem.Allocator, k: []const u8, n: u32, hidden: u64, inter: u64, experts: u32, size_delta: i64) ![]u8 {
    const g = geometry(n, hidden, inter).?;
    var segs: std.Io.Writer.Allocating = .init(a);
    defer segs.deinit();
    for (g.segments, 0..) |s, c| {
        const name = @tagName(@as(Component, @enumFromInt(c)));
        const us = std.mem.indexOfScalar(u8, name, '_').?;
        try segs.writer.print("{s}{{\"component\":\"{s}_proj.{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ if (c == 0) "" else ",", name[0..us], name[us + 1 ..], if (s.rank == 3) "I16" else "F16" });
        for (s.shape[0..s.rank], 0..) |d, i| try segs.writer.print("{s}{d}", .{ if (i == 0) "" else ",", d });
        try segs.writer.print("],\"offset\":{d},\"length\":{d}}}", .{ s.offset, s.length });
    }
    const size: i64 = @as(i64, @intCast(g.record_bytes * experts)) + size_delta;
    return std.fmt.allocPrint(a,
        \\{{"format":"mtplx-expert-manifest-v2","model_key":"glm5next-flash-mtp-expert-exl3","first_moe_layer":45,
        \\"quantization":{{"mode":"exl3","codebook":"mul1","codebook_multiplier":2212286765}},
        \\"dims":{{"hidden":{d},"inter":{d},"n_experts":{d},"n_layers":1}},
        \\"sidecar":{{"file":"mtp-experts.bin","alignment":4096,"size":{d}}},
        \\"layers":[{{"layer":0,"model_layer":45,"K":{s},"record_bytes":{d},"logical_bytes":{d},"base_offset":0,"segments":[{s}]}}],
        \\"records":[]}}
    , .{ hidden, inter, experts, size, k, g.record_bytes, g.logical_bytes, segs.written() });
}

test "turboexl3_glm53f mtp bank: GLM-5.3-Flash's records at K3 and K4 are LAYOUT.md's sizes, no padding" {
    try testing.expectEqual(@as(u64, 9_474_048), geometry(48, 4096, 2048).?.record_bytes);
    try testing.expectEqual(@as(u64, 12_619_776), geometry(64, 4096, 2048).?.record_bytes);
    try testing.expectEqual(@as(u64, 7_901_184), geometry(40, 4096, 2048).?.record_bytes);
    try testing.expectEqual(@as(u64, 11_046_912), geometry(56, 4096, 2048).?.record_bytes);
    const g = geometry(64, 4096, 2048).?;
    try testing.expectEqual(g.logical_bytes, g.record_bytes);
    // LAYOUT.md: mtp-experts.bin at K4 is 3,634,495,488 bytes
    try testing.expectEqual(@as(u64, 3_634_495_488), g.record_bytes * 288);
    try testing.expectEqual(@as(?u32, 40), halfwordsOfK(2.5));
    try testing.expectEqual(@as(?u32, null), halfwordsOfK(2.3));
}

test "turboexl3_glm53f mtp bank: the converter's manifest parses at K3 and K4; another codebook, K, size or dims is refused by name" {
    const a = testing.allocator;
    for ([_]struct { k: []const u8, n: u32 }{ .{ .k = "3", .n = 48 }, .{ .k = "4", .n = 64 } }) |c| {
        const t = try manifestText(a, c.k, c.n, 256, 128, 4, 0);
        defer a.free(t);
        const m = try Manifest.parse(a, t, 256, 128, 4);
        try testing.expectEqual(c.n, m.geometry.n);
        try testing.expectEqual(m.geometry.record_bytes + m.geometry.segments[3].offset, m.offsetOf(1, 3));
    }
    const short = try manifestText(a, "4", 64, 256, 128, 4, -4096);
    defer a.free(short);
    try testing.expectError(error.Glm53fMtpGeometry, Manifest.parse(a, short, 256, 128, 4));
    const wrong_k = try manifestText(a, "3", 64, 256, 128, 4, 0);
    defer a.free(wrong_k);
    try testing.expectError(error.Glm53fMtpGeometry, Manifest.parse(a, wrong_k, 256, 128, 4));
    const ok = try manifestText(a, "4", 64, 256, 128, 4, 0);
    defer a.free(ok);
    try testing.expectError(error.Glm53fMtpGeometry, Manifest.parse(a, ok, 512, 128, 4));
    const mcg = try std.mem.replaceOwned(u8, a, ok, "\"codebook\":\"mul1\"", "\"codebook\":\"mcg\"");
    defer a.free(mcg);
    try testing.expectError(error.Glm53fMtpCodebook, Manifest.parse(a, mcg, 256, 128, 4));
}
