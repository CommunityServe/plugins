//! copied_from: src/plugins/minimax_m3/fixture.zig@37e90382b439f163474f99d3c78522ae88caaf12
//! The tiny pack the conformance suite runs (`plugin.zig` `conformanceFixture`): the released text architecture at
//! fixture size (`fixture_config.json`: hidden 512, five layers, three dense then two MoE, eight query heads over four
//! KV heads of 128, eight experts), BF16 residents and an NVFP4 expert bank in the prepared pack's layout
//! (`nvfp4_bank.zig`). Seeded random weights; MLX-free.
const std = @import("std");
const sdk = @import("mlx_serve").sdk;
const geometry = @import("geometry.zig");
const nvfp4 = @import("nvfp4_bank.zig");
const bf16_bank = @import("bf16_bank.zig");
pub const config_json = @embedFile("fixture_config.json");
/// The fixture's experts as an NVFP4 bank (the prepared pack's) or as BF16 (`bf16_bank.zig`) holding the same values:
/// every fixture weight is a 2-bit code times a power of two, exact in BF16.
pub const BankKind = enum { nvfp4, bf16 };
/// `decided`: the parity fixture. Every norm weight -0.5 (gain 0.5) and the routing bias 1000x the draws, so the top-4
/// experts are fixed and the exact math rounded to bf16 stays within 0.999 of float64 on every row: the parity bar then
/// measures the continuous math. The routing and block-selection rules are proven apart on identical scores.
pub const Conditioning = enum { conformance, decided };
/// The fixture pack's files: `config.json`, the residents (`model.safetensors` and its index) and the expert bank
/// (`experts.bin`, `expert-manifest.json`). `a` should be an arena.
pub fn files(a: std.mem.Allocator, seed: u64) ![]const sdk.fixture.File {
    return filesOf(a, seed, .nvfp4);
}
pub fn filesOf(a: std.mem.Allocator, seed: u64, kind: BankKind) ![]const sdk.fixture.File {
    return variantOf(a, seed, kind, .conformance);
}
pub fn variantOf(a: std.mem.Allocator, seed: u64, kind: BankKind, cond: Conditioning) ![]const sdk.fixture.File {
    const norm: f32 = if (cond == .decided) -0.5 else 1;
    const bias: f32 = if (cond == .decided) 100 else 0.1;
    const c = try geometry.Config.parse(a, config_json, null);
    var t = sdk.tensors.Tensors.init(a, seed);
    const h: u64 = c.hidden_size;
    const d: u64 = c.head_dim;
    for (0..c.n_layers) |i| {
        try t.fill("model.layers.{d}.input_layernorm.weight", .{i}, &.{h}, .BF16, norm);
        try t.fill("model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}, .BF16, norm);
        try t.rand("model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ c.n_heads * d, h }, .BF16, 0.05);
        try t.rand("model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ c.n_kv_heads * d, h }, .BF16, 0.05);
        try t.rand("model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ c.n_kv_heads * d, h }, .BF16, 0.05);
        try t.rand("model.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, c.n_heads * d }, .BF16, 0.05);
        try t.fill("model.layers.{d}.self_attn.q_norm.weight", .{i}, &.{d}, .BF16, norm);
        try t.fill("model.layers.{d}.self_attn.k_norm.weight", .{i}, &.{d}, .BF16, norm);
        try t.rand("model.layers.{d}.self_attn.index_q_proj.weight", .{i}, &.{ @as(u64, c.index_n_heads) * c.index_head_dim, h }, .BF16, 0.05);
        try t.rand("model.layers.{d}.self_attn.index_k_proj.weight", .{i}, &.{ c.index_head_dim, h }, .BF16, 0.05);
        try t.fill("model.layers.{d}.self_attn.index_q_norm.weight", .{i}, &.{c.index_head_dim}, .BF16, norm);
        try t.fill("model.layers.{d}.self_attn.index_k_norm.weight", .{i}, &.{c.index_head_dim}, .BF16, norm);
        if (c.isMoe(i)) {
            const si: u64 = c.shared_intermediate_size;
            try t.rand("model.layers.{d}.block_sparse_moe.shared_experts.gate_proj.weight", .{i}, &.{ si, h }, .BF16, 0.05);
            try t.rand("model.layers.{d}.block_sparse_moe.shared_experts.up_proj.weight", .{i}, &.{ si, h }, .BF16, 0.05);
            try t.rand("model.layers.{d}.block_sparse_moe.shared_experts.down_proj.weight", .{i}, &.{ h, si }, .BF16, 0.05);
            try t.rand("model.layers.{d}.block_sparse_moe.gate.weight", .{i}, &.{ c.n_routed_experts, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.block_sparse_moe.e_score_correction_bias", .{i}, &.{c.n_routed_experts}, .F32, bias);
        } else {
            const di: u64 = c.dense_intermediate_size;
            try t.rand("model.layers.{d}.mlp.gate_proj.weight", .{i}, &.{ di, h }, .BF16, 0.05);
            try t.rand("model.layers.{d}.mlp.up_proj.weight", .{i}, &.{ di, h }, .BF16, 0.05);
            try t.rand("model.layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, di }, .BF16, 0.05);
        }
    }
    try t.rand("model.embed_tokens.weight", .{}, &.{ c.vocab_size, h }, .BF16, 0.05);
    try t.rand("lm_head.weight", .{}, &.{ c.vocab_size, h }, .BF16, 0.05);
    try t.fill("model.norm.weight", .{}, &.{h}, .BF16, norm);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"metadata\":{},\"weight_map\":{");
    for (t.list.items, 0..) |e, i| try index.print(a, "{s}\"{s}\":\"model.safetensors\"", .{ if (i == 0) "" else ",", e.name });
    try index.appendSlice(a, "}}");
    const quantized = try expertBank(a, &c, seed +% 1);
    const bank = if (kind == .bf16) try bf16Bank(a, &c, quantized.data) else quantized;
    const cfg = if (kind == .bf16) try std.mem.replaceOwned(u8, a, config_json, nvfp4.format, bf16_bank.format) else config_json;
    const out = try a.alloc(sdk.fixture.File, 5);
    out[0] = .{ .name = "config.json", .bytes = cfg };
    out[1] = .{ .name = "model.safetensors", .bytes = try t.bytes() };
    out[2] = .{ .name = "model.safetensors.index.json", .bytes = index.items };
    out[3] = .{ .name = "experts.bin", .bytes = bank.data };
    out[4] = .{ .name = "expert-manifest.json", .bytes = bank.manifest };
    return out;
}
const Bank = struct { data: []const u8, manifest: []const u8 };
/// Every (MoE layer, expert) record: random E2M1 codes, E4M3 block scales of 0.25 to 1 and a global scale whose
/// g / 2688 is 1/64 (weights of at most 6/64), page-aligned as the prepared pack's.
fn expertBank(a: std.mem.Allocator, c: *const geometry.Config, seed: u64) !Bank {
    const g = try nvfp4.Geometry.init(c.hidden_size, c.intermediate_size, c.n_routed_experts, geometry.first_moe_layer, c.moeLayers());
    const data = try a.alloc(u8, g.total_bytes);
    @memset(data, 0);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const scale_codes = [_]u8{ 0x28, 0x30, 0x38 }; // E4M3 0.25, 0.5, 1.
    const global: f32 = 2688.0 / 64.0;
    var records: std.ArrayList(u8) = .empty;
    const count = @as(usize, g.num_experts) * g.num_moe_layers;
    for (0..count) |i| {
        const rec = data[i * g.record_bytes ..][0..g.logical_bytes];
        for (g.segments) |seg| {
            const part = rec[seg.offset..][0..seg.length];
            switch (seg.dtype) {
                .U32 => rnd.bytes(part),
                .U8 => for (part) |*b| {
                    b.* = scale_codes[rnd.uintLessThan(usize, scale_codes.len)];
                },
                .F32 => std.mem.writeInt(u32, part[0..4], @bitCast(global), .little),
            }
        }
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(rec, &digest, .{});
        try records.print(a, "{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{x}\"}}", .{
            if (i == 0) "" else ",", g.first_moe_layer + i / g.num_experts, i % g.num_experts, i * g.record_bytes, g.logical_bytes, &digest,
        });
    }
    var parts: std.ArrayList(u8) = .empty;
    for (g.segments, 0..) |seg, i| {
        const comp: nvfp4.Component = @fromBackingInt(@intCast(i));
        try parts.print(a, "{s}{{\"name\":\"{s}\",\"dtype\":\"{t}\",\"shape\":[", .{ if (i == 0) "" else ",", comp.name(), seg.dtype });
        for (seg.shape[0..seg.rank], 0..) |dim, k| try parts.print(a, "{s}{d}", .{ if (k == 0) "" else ",", dim });
        try parts.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ seg.offset, seg.length });
    }
    var file_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &file_digest, .{});
    const manifest = try std.fmt.allocPrint(a,
        \\{{"format":"{s}","data_file":"experts.bin","layout":"{s}","alignment":{d},"hidden_size":{d},"intermediate_size":{d},
        \\"num_experts":{d},"first_moe_layer":{d},"num_moe_layers":{d},"record_bytes":{d},"record_stride":{d},"gate_up_bytes":{d},
        \\"down_bytes":{d},"quantization":{{"mode":"nvfp4","bits":{d},"group_size":{d}}},
        \\"source":{{"repository":"fixture/minimax_m3","revision":"0000000000000000000000000000000000000000"}},
        \\"parts":[{s}],"records":[{s}],"output_files":{{"experts.bin":{{"size":{d},"sha256":"{x}"}}}}}}
    , .{
        nvfp4.format,        nvfp4.layout,        nvfp4.page,       g.hidden_size,      g.intermediate_size, g.num_experts,
        g.first_moe_layer,   g.num_moe_layers,    g.logical_bytes,  g.record_bytes,     g.gate_up_bytes,     g.logical_bytes - g.gate_up_bytes,
        nvfp4.bits,          nvfp4.group_size,    parts.items,      records.items,      g.total_bytes,       &file_digest,
    });
    return .{ .data = data, .manifest = manifest };
}
/// The NVFP4 bank `nv` (as `expertBank` writes it) dequantized into the BF16 layout; refuses a value BF16 would round.
fn bf16Bank(a: std.mem.Allocator, c: *const geometry.Config, nv: []const u8) !Bank {
    const q = try nvfp4.Geometry.init(c.hidden_size, c.intermediate_size, c.n_routed_experts, geometry.first_moe_layer, c.moeLayers());
    const g = try bf16_bank.Geometry.init(c.hidden_size, c.intermediate_size, c.n_routed_experts, geometry.first_moe_layer, c.moeLayers());
    const data = try a.alloc(u8, g.total_bytes);
    @memset(data, 0);
    var records: std.ArrayList(u8) = .empty;
    const count = @as(usize, g.num_experts) * g.num_moe_layers;
    for (0..count) |i| {
        const src = nv[i * q.record_bytes ..][0..q.logical_bytes];
        const dst = data[i * g.record_bytes ..][0..g.logical_bytes];
        for (0..3) |p| {
            const w = q.segments[p * 3];
            const sc = q.segments[p * 3 + 1];
            const gs: f32 = @bitCast(std.mem.readInt(u32, src[q.segments[p * 3 + 2].offset..][0..4], .little));
            const out_rows = w.shape[0];
            const words_row = w.shape[1];
            const words = try a.alloc(u32, words_row);
            const row = try a.alloc(f32, words_row * 8);
            for (0..out_rows) |o| {
                for (words, 0..) |*word, k| word.* = std.mem.readInt(u32, src[w.offset + (o * words_row + k) * 4 ..][0..4], .little);
                nvfp4.dequantRow(words, src[sc.offset + o * sc.shape[1] ..][0..sc.shape[1]], gs, row);
                const seg = g.segments[p];
                for (row, 0..) |v, j| {
                    const bits: u32 = @bitCast(v);
                    if ((bits & 0xffff) != 0) return error.FixtureNotExactInBf16;
                    std.mem.writeInt(u16, dst[seg.offset + (o * row.len + j) * 2 ..][0..2], @intCast(bits >> 16), .little);
                }
            }
        }
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(dst, &digest, .{});
        try records.print(a, "{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{x}\"}}", .{
            if (i == 0) "" else ",", g.first_moe_layer + i / g.num_experts, i % g.num_experts, i * g.record_bytes, g.logical_bytes, &digest,
        });
    }
    var parts: std.ArrayList(u8) = .empty;
    for (g.segments, 0..) |seg, i| {
        const comp: bf16_bank.Component = @enumFromInt(i);
        try parts.print(a, "{s}{{\"name\":\"{s}\",\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"offset\":{d},\"length\":{d}}}", .{ if (i == 0) "" else ",", comp.name(), seg.shape[0], seg.shape[1], seg.offset, seg.length });
    }
    var file_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &file_digest, .{});
    const manifest = try std.fmt.allocPrint(a,
        \\{{"format":"{s}","data_file":"experts.bin","layout":"{s}","alignment":{d},"hidden_size":{d},"intermediate_size":{d},
        \\"num_experts":{d},"first_moe_layer":{d},"num_moe_layers":{d},"record_bytes":{d},"record_stride":{d},"gate_up_bytes":{d},
        \\"down_bytes":{d},"source":{{"repository":"fixture/minimax_m3","revision":"0000000000000000000000000000000000000000"}},
        \\"parts":[{s}],"records":[{s}],"output_files":{{"experts.bin":{{"size":{d},"sha256":"{x}"}}}}}}
    , .{
        bf16_bank.format, bf16_bank.layout, bf16_bank.page, g.hidden_size, g.intermediate_size, g.num_experts, g.first_moe_layer, g.num_moe_layers,
        g.logical_bytes, g.record_bytes, g.gate_up_bytes, g.logical_bytes - g.gate_up_bytes, parts.items, records.items, g.total_bytes, &file_digest,
    });
    return .{ .data = data, .manifest = manifest };
}
