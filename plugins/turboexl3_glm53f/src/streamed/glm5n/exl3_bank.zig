//! GLM-5.3-Flash's routed experts as an EXL3 K=3 expert bank (the `exl3` expert family): DeepSeek-V4.1's bank-v2
//! format at GLM's dims, read by DeepSeek's reader (`src/expert/bank.zig Bank.open`: `expert-manifest-v2.json`, the v1
//! runtime `expert-manifest.json`, `experts.bin`), the stream's bank module over it (`SDK`), and the checks the pack's
//! quantization must pass before anything is read (`checkPeek`). The pack contract is the converter's
//! (R/mimo/glm5next-exl3-pack-contract.md): bank layer b = model layer - first_moe_layer; per record nine segments,
//! per projection (gate, up, down) its trellis `code` I16 [in / 16, out / 16, 48], `rout` (exl3 svh) F16 [out] and `rin`
//! (exl3 suh) F16 [in], back to back; 9,474,048 B a record at GLM's dims (2,313 pages: no padding). The codebook
//! multiplier lives in the manifest (`quantization.codebook_multiplier`), not in the record: a record carries no
//! global-scale word, so the NVFP4 bank's scale-line isolation has nothing to isolate here (the slot arrays are the
//! segments' rows, each a whole number of 128-byte lines).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../sdk_ext.zig");
const quant = sdk_ext.quant;
const expert_bank = @import("../expert/bank.zig");
const xq = @import("../../exl3/quant.zig");
const kernels = @import("exl3_kernels.zig");
const Module = @This();

/// The pack's `expert_streaming` block (config.json): the family, the bank-v2 format, its two manifests, the quant.
pub const format = expert_bank.format_v2;
pub const manifest_file = "expert-manifest-v2.json";

pub const n_components = expert_bank.n_components;
pub const gu_components = expert_bank.gu_components;
pub const Records = expert_bank.Records;
pub const Component = expert_bank.Component;
pub const Layer = expert_bank.Layer;
pub const Segment = expert_bank.Segment;
pub const routed_top_k = 8;
pub const page = expert_bank.record_alignment;

/// GLM-5.3-Flash's bank the texts decode: mul1 at K = 3 on every layer, hidden 4096, inter 2048, 288 experts, 42 layers.
/// turboexl3_glm53f: turboderp's self-calibrated packs' rates, every routed layer on turboexl3: K 2, 3, 4, 5 and exllamav3
/// 1.6's K 2.5 and 3.5 (40 and 56 halfwords a tile), one K a layer.
pub const flash: expert_bank.Implemented = .{ .codebooks = &.{kernels.codebook}, .k = &.{ 2, 3, 4, 5 }, .frac_n = &.{ 40, 56 }, .hidden = kernels.hidden, .inter = kernels.inter, .n_experts = 288, .n_layers = 42 };
pub const flash_first_moe_layer: u32 = 3;

/// One projection's slot arrays (DeepSeek's `ProjArrays`: code, rout, rin).
pub fn Arrays(comptime T: type) type {
    return xq.ProjArrays(T);
}

pub fn BankArraysOf(comptime T: type) type {
    return quant.BankArrays(Arrays(T));
}

pub const BankArrays = BankArraysOf(mlx.mlx_array);

pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{ .gate = .{ .code = x[0], .rout = x[1], .rin = x[2] }, .up = .{ .code = x[3], .rout = x[4], .rin = x[5] }, .down = .{ .code = x[6], .rout = x[7], .rin = x[8] } };
}

pub fn mlxDtype(d: expert_bank.Dtype) mlx.mlx_dtype {
    return expert_bank.mlxDtype(d);
}

/// The record geometry, in the field names the source, the bill and the profile read (the NVFP4 bank's `Geometry`).
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    logical_bytes: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]Segment,

    /// turboexl3_glm53f: the geometry of `ns.len` layers of `experts` records at (hidden, inter), layer L at `ns[L]`
    /// halfwords a tile (DeepSeek's `layerSegmentsN`): the record bytes, segments and gate/up span the widest layer's
    /// (every bound and bill reads the widest record), the total the whole bank's.
    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, ns: []const u32) !Geometry {
        if (experts == 0 or experts >= std.math.maxInt(u16) or ns.len == 0 or ns.len > 4096 or first > 4096) return error.InvalidGeometry;
        var l = expert_bank.layerSegmentsN(ns[0], hidden, inter) orelse return error.InvalidGeometry;
        var total: u64 = 0;
        for (ns) |n| {
            const x = expert_bank.layerSegmentsN(n, hidden, inter) orelse return error.InvalidGeometry;
            if (x.record_bytes > l.record_bytes) l = x;
            total = std.math.add(u64, total, std.math.mul(u64, x.record_bytes, experts) catch return error.InvalidGeometry) catch return error.InvalidGeometry;
        }
        const layers: u64 = ns.len;
        return .{
            .hidden_size = @intCast(hidden),
            .intermediate_size = @intCast(inter),
            .num_experts = @intCast(experts),
            .first_moe_layer = @intCast(first),
            .num_moe_layers = @intCast(layers),
            .logical_bytes = l.logical_bytes,
            .record_bytes = l.record_bytes,
            .gate_up_bytes = l.segments[gu_components].offset,
            .total_bytes = total,
            .segments = l.segments,
        };
    }

    pub fn requireFlash(g: *const Geometry) !void {
        if (g.hidden_size != flash.hidden or g.intermediate_size != flash.inter or g.num_experts != flash.n_experts or g.first_moe_layer != flash_first_moe_layer or g.num_moe_layers != flash.n_layers)
            return error.UnsupportedTopology;
    }
};

/// The bank's quantization as the texts decode it, field by field (the v2 manifest's `quantization`, DeepSeek's quant
/// `claims` at GLM's dims): mode exl3, the mul1 codebook at its multiplier, the tile, the Hadamard, the scales, K = 3 on
/// every layer and the dims `want`. Refused once, by name, before any read.
pub fn checkPeek(v: *const quant.BankPeek, want: expert_bank.Implemented, diag: *sdk.Diag) !void {
    const q = v.quantization;
    const f = xq.format;
    if (!strIs(q, "mode", f.mode)) return refuse(diag, error.Glm5NextExl3Mode, "quantization.mode \"{s}\" (the texts decode {s})", .{ quant.str(q, "mode") orelse "", f.mode });
    if (!strIs(q, "codebook", f.codebook)) return refuse(diag, error.Glm5NextExl3Codebook, "quantization.codebook \"{s}\" (the texts decode {s} only)", .{ quant.str(q, "codebook") orelse "", f.codebook });
    if ((quant.int(q, "codebook_multiplier") orelse -1) != f.multiplier) return refuse(diag, error.Glm5NextExl3Codebook, "quantization.codebook_multiplier {?d} (mul1 is {d})", .{ quant.int(q, "codebook_multiplier"), f.multiplier });
    const tile = quant.obj(q, "tile");
    if ((quant.int(tile, "size") orelse -1) != f.tile_size or !strIs(tile, "layout", f.tile_layout) or !strIs(tile, "bitstream", f.tile_bitstream))
        return refuse(diag, error.Glm5NextExl3Tile, "quantization.tile {?d} / {s} / {s} (the texts read {d} / {s} / {s})", .{ quant.int(tile, "size"), quant.str(tile, "layout") orelse "", quant.str(tile, "bitstream") orelse "", f.tile_size, f.tile_layout, f.tile_bitstream });
    const had = quant.obj(q, "hadamard");
    if ((quant.int(had, "size") orelse -1) != f.hadamard_size or !strIs(had, "order", f.hadamard_order) or !strIs(had, "scale", f.hadamard_scale))
        return refuse(diag, error.Glm5NextExl3Hadamard, "quantization.hadamard {?d} / {s} / {s} (the texts rotate {d} / {s} / {s})", .{ quant.int(had, "size"), quant.str(had, "order") orelse "", quant.str(had, "scale") orelse "", f.hadamard_size, f.hadamard_order, f.hadamard_scale });
    const sc = quant.obj(q, "scales");
    if (!strIs(sc, "rin", f.scale_rin) or !strIs(sc, "rout", f.scale_rout))
        return refuse(diag, error.Glm5NextExl3Scales, "quantization.scales rin {s} / rout {s} (the texts read {s} / {s})", .{ quant.str(sc, "rin") orelse "", quant.str(sc, "rout") orelse "", f.scale_rin, f.scale_rout });
    if (v.hidden != want.hidden or v.inter != want.inter or v.n_experts != want.n_experts or v.n_layers != want.n_layers or v.layers.len != v.n_layers)
        return refuse(diag, error.Glm5NextExl3Dims, "dims hidden {d} / inter {d} / {d} experts / {d} layers ({d} entries) (the texts are instantiated at {d} / {d}, the bank is {d} x {d})", .{ v.hidden, v.inter, v.n_experts, v.n_layers, v.layers.len, want.hidden, want.inter, want.n_experts, want.n_layers });
    // turboexl3_glm53f: each layer at a K turboexl3 serves here (a half-integer one peeks as bits 0: its code tiles' width).
    for (v.layers, 0..) |l, li| {
        const n = codeWidth(l) orelse return refuse(diag, error.Glm5NextExl3K, "layer {d}: no gate code segment", .{li});
        const ok = if (l.bits != 0) std.mem.indexOfScalar(u32, want.k, l.bits) != null and n == 16 * l.bits else std.mem.indexOfScalar(u32, want.frac_n, n) != null;
        if (!ok) return refuse(diag, error.Glm5NextExl3K, "layer {d} K {d} ({d} halfwords a tile; served: K {any}, halfwords {any})", .{ li, l.bits, n, want.k, want.frac_n });
    }
}
/// turboexl3_glm53f: a peeked layer's halfwords a tile, its gate code segment's last dim.
fn codeWidth(l: quant.LayerPeek) ?u32 {
    if (l.segments.len == 0 or l.segments[0].shape.len != 3) return null;
    return std.math.cast(u32, l.segments[0].shape[2]);
}

fn strIs(v: std.json.Value, field: []const u8, want: []const u8) bool {
    return std.mem.eql(u8, quant.str(v, field) orelse return false, want);
}

fn refuse(diag: *sdk.Diag, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    diag.set("glm5_next exl3 bank: " ++ fmt, args);
    return err;
}

/// The bank's geometry for the bill and the admission (pure host): the v2 manifest peeked (records skipped, no sidecar),
/// its quantization checked (`checkPeek`), the geometry its dims give.
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8, first_moe_layer: u32, want: expert_bank.Implemented, diag: *sdk.Diag) !Geometry {
    var p = try expert_bank.peek(a, io, dir, diag);
    defer p.deinit();
    try checkPeek(&p.view, want, diag);
    const ns = try a.alloc(u32, p.view.layers.len);
    defer a.free(ns);
    for (ns, p.view.layers) |*n, l| n.* = codeWidth(l).?;
    return Geometry.init(p.view.hidden, p.view.inter, p.view.n_experts, first_moe_layer, ns);
}

/// The open bank: DeepSeek's reader over both manifests and `experts.bin` (every record's place and digest, the
/// sidecar opened past the page cache), after `checkPeek`.
pub const Bank = struct {
    owner: expert_bank.Bank,
    geometry: Geometry,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, first_moe_layer: u32, want: expert_bank.Implemented, diag: *sdk.Diag) !Bank {
        const geometry = try inspect(a, io, dir, first_moe_layer, want, diag);
        var owner = try expert_bank.Bank.open(a, io, dir, want, diag);
        errdefer owner.deinit();
        // turboexl3_glm53f: each layer's segments are its own K's (the reader checked them against `layerSegmentsN`).
        if (owner.sidecar_size != geometry.total_bytes) return refuse(diag, error.Glm5NextExl3Dims, "sidecar {d} B, the geometry's {d} B", .{ owner.sidecar_size, geometry.total_bytes });
        return .{ .owner = owner, .geometry = geometry };
    }

    pub fn deinit(b: *Bank) void {
        b.owner.deinit();
        b.* = undefined;
    }
};

/// The stream's view of an open Bank (the NVFP4 SDKBank's fields): borrowed layers and sidecar.
pub const SDKBank = struct {
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    record_bytes: u64,
    sidecar: sdk_ext.expert.UncachedFd,

    pub fn init(_: std.mem.Allocator, owner: *const Bank) !SDKBank {
        if (owner.owner.sidecar.fd < 0) return error.SidecarNotOpen;
        return .{ .layers = owner.owner.layers, .n_experts = owner.owner.n_experts, .first_moe_layer = owner.geometry.first_moe_layer, .record_bytes = owner.geometry.record_bytes, .sidecar = owner.owner.sidecar };
    }

    pub fn deinit(b: *SDKBank) void {
        b.* = undefined;
    }

    /// SDK layer 0 is the bank's layer 0 (model layer `first_moe_layer`).
    pub fn recordOffset(b: *const SDKBank, layer: u32, expert: u32) u64 {
        const l = &b.layers[layer];
        return l.base_offset + @as(u64, expert) * l.record_bytes;
    }

    pub fn spans(b: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }
};

/// StreamOf's bank module (no `slot_line_bytes`: nothing to isolate).
pub const SDK = struct {
    pub const n_components = Module.n_components;
    pub const gu_components = Module.gu_components;
    pub const Records = Module.Records;
    pub const Component = Module.Component;
    pub const Layer = Module.Layer;
    pub const Bank = SDKBank;
    pub const mlxDtype = Module.mlxDtype;
    pub const BankArrays = Module.BankArrays;
    pub const bankArraysOf = Module.bankArraysOf;
    pub const routed_top_k = Module.routed_top_k;
    /// Every slot array one VM page at least (`StreamOf.slot_array_min_bytes`): the direct reads' 4 KiB block alignment
    /// (a one-row ext of a 4 KiB rin / rout array was refused with DirectReadsRow).
    pub const slot_array_min_bytes: u64 = std.heap.page_size_max;
    /// turboexl3_glm53f: a pack's two code widths (its layer 3 and the rest) keep separate transient scratches.
    pub const per_geometry_transients = true;
};

/// The slot arrays the stream binds, checked once per bank at construction: the texts' shapes and dtypes.
pub fn checkArrays(g: anytype, which: []const u8, p: Arrays(@TypeOf(g.*).T), down: bool) !void {
    const d = kernels.projDims(if (down) .down else .gate_up);
    const sc = g.shapeOf(p.code);
    const so = g.shapeOf(p.rout);
    const si = g.shapeOf(p.rin);
    // turboexl3_glm53f: the code tiles at the layer's own width (even, 16..128 halfwords).
    const ok = sc.n == 4 and sc.d[1] == d.code[0] and sc.d[2] == d.code[1] and sc.d[3] >= 16 and sc.d[3] <= 128 and @mod(sc.d[3], 2) == 0 and g.dtypeOf(p.code) == .int16 and
        so.n == 2 and so.d[0] == sc.d[0] and so.d[1] == d.rout and g.dtypeOf(p.rout) == .float16 and
        si.n == 2 and si.d[0] == sc.d[0] and si.d[1] == d.rin and g.dtypeOf(p.rin) == .float16;
    if (!ok) {
        sdk.log.err("glm5_next: {s} slot arrays are {any} {any} {any}, not the exl3 bank's [slots, {d}, {d}, {d}] i16, [slots, {d}] f16, [slots, {d}] f16\n", .{ which, sc.slice(), so.slice(), si.slice(), d.code[0], d.code[1], d.code[2], d.rout, d.rin });
        return error.Glm5NextSlotArrays;
    }
}

// ── Tests ──

const testing = std.testing;

/// A synthetic two-layer, eight-expert bank at GLM's expert shape (DeepSeek's writer at hidden 4096 / inter 2048),
/// its v2 manifest completed (`fullQuantization`).
const Synth = struct {
    tmp: testing.TmpDir,
    root: []const u8,
    rbuf: [512]u8,
    /// The bank's experts.bin as written (`Stream.expectServed`'s image).
    image: []u8,

    fn close(self: *Synth) void {
        testing.allocator.free(self.image);
        self.tmp.cleanup();
        testing.allocator.destroy(self);
    }
};
