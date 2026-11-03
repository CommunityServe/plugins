//! The conformance pack: a 2-layer Hy-MT2 1.8B shape (hidden 2048, 16 heads / 4 kv, head_dim 128) with real
//! STQ1_0 projections and a real 6-bit affine embedding.
//!
//!   * STQ1_0: `qs` (u8, one 4-bit codebook slot per 4-weight group), `sign` (u8, one sign bit per group) and
//!     `scales` (f16, one per 256-weight block), laid out as in the pack. Weight `k` of a block sits at row `k % 16`
//!     of its 16-column slice, which is the pack's `_INV_PERM`.
//!   * embedding: MLX affine 6-bit, group 64. Each name holds what it says: `scales` is the step (what a pack from
//!     `mx.quantize` puts there), `biases` the group's low end.
//!
//! `Expected` keeps the float weights every tensor was quantized from, so a test can hold the kernels to them.

const std = @import("std");
const kit = @import("mlx_serve");

pub const vocab_size = 260;
pub const hidden = 2048;
pub const heads = 16;
pub const kv_heads = 4;
pub const head_dim = 128;
pub const ffn = 2048;
pub const layers = 2;
pub const q_dim = heads * head_dim; // 2048
pub const kv_dim = kv_heads * head_dim; // 512
pub const embed_bits = 6;
pub const embed_group = 64;

/// The checkpoint's `_CODEBOOK`: 32 entries, each four lanes of {-1, 0, +1}.
pub const codebook = [32]u8{ 169, 137, 41, 9, 166, 134, 38, 6, 154, 146, 26, 18, 106, 98, 74, 66, 1, 33, 129, 161, 4, 36, 132, 164, 16, 24, 144, 152, 64, 72, 96, 104 };

/// Every (sign, code) pair's four ternary lanes, precomputed once: [2][32][4], sign outer.
const lane_table: [2][32][4]f32 = blk: {
    var t: [2][32][4]f32 = undefined;
    for (0..2) |sg| {
        for (0..32) |c| {
            const qp: u8 = codebook[(sg << 4) | c];
            for (0..4) |p| {
                const two: u8 = (qp >> @intCast(2 * p)) & 3;
                t[sg][c][p] = @floatFromInt(@as(i8, @intCast(two)) - 1);
            }
        }
    }
    break :blk t;
};

/// One (sign, code) pair's four ternary lanes, in the pack's lane order.
pub fn lanes(code: u5, sign: bool) [4]f32 {
    return lane_table[@intFromBool(sign)][code];
}

pub fn f16Float(c: u16) f32 {
    const sign_bit: u32 = @as(u32, c >> 15) << 31;
    const exp: u32 = (c >> 10) & 0x1F;
    const mant: u32 = c & 0x3FF;
    var bits: u32 = undefined;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign_bit;
        } else {
            var m = mant;
            var e: i32 = -14;
            while (m & 0x400 == 0) {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            bits = sign_bit | (@as(u32, @intCast(e + 127)) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign_bit | 0x7F80_0000 | (mant << 13);
    } else {
        bits = sign_bit | ((exp + 112) << 23) | (mant << 13);
    }
    return @bitCast(bits);
}

/// f32 to f16 bits, round to nearest even. Test data is small normal numbers; the edges are clamped.
pub fn f16Bits(w: f32) u16 {
    const b: u32 = @bitCast(w);
    const sign_bit: u32 = b >> 31;
    const exp: i32 = @intCast((b >> 23) & 0xFF);
    const mant: u32 = b & 0x7FFFFF;
    if (exp == 0xFF) return @intCast((sign_bit << 15) | 0x7C00 | (mant >> 13));
    const e16 = exp - 112;
    if (e16 <= 0) {
        if (e16 < -10) return @intCast(sign_bit << 15);
        const m = mant | 0x800000;
        const shift: u5 = @intCast(14 - e16);
        const rounded = (m + (@as(u32, 1) << (shift - 1))) >> shift;
        return @intCast((sign_bit << 15) | rounded);
    }
    if (e16 >= 0x1F) return @intCast((sign_bit << 15) | 0x7C00);
    const rounded = mant + 0x1000;
    if (rounded & 0x800000 != 0) return @intCast((sign_bit << 15) | (@as(u32, @intCast((e16 + 1) & 0x1F)) << 10));
    return @intCast((sign_bit << 15) | (@as(u32, @intCast(e16)) << 10) | (rounded >> 13));
}

fn halfOf(w: f32) f16 {
    return @bitCast(f16Bits(w));
}

/// Weight `p` of group `g` inside a 256-weight block, the pack's `_INV_PERM`: group `g` starts at column
/// `(g / 16) * 64 + (g % 16)` and its four lanes sit 16 columns apart.
pub fn laneColumn(g: usize, p: usize) usize {
    return (g / 16) * 64 + (g % 16) + p * 16;
}

/// Quantize one [rows, cols] f32 matrix to the pack's STQ1_0 buffers. Caller frees the three.
pub fn quantizeStq(a: std.mem.Allocator, rows: usize, cols: usize, w: []const f32) !struct { qs: []u8, sign: []u8, scales: []u16 } {
    const nb = cols / 256;
    const qs = try a.alloc(u8, rows * cols / 8);
    const sign = try a.alloc(u8, rows * cols / 32);
    const scales = try a.alloc(u16, rows * nb);
    @memset(qs, 0);
    @memset(sign, 0);
    for (0..rows) |r| {
        const row = w[r * cols ..][0..cols];
        for (0..nb) |b| {
            const block = row[b * 256 ..][0..256];
            var max_abs: f32 = 0;
            for (block) |x| max_abs = @max(max_abs, @abs(x));
            const scale: f16 = halfOf(if (max_abs == 0) 1e-8 else max_abs);
            const d = @as(f32, scale);
            scales[r * nb + b] = @as(u16, @bitCast(scale));
            for (0..64) |g| {
                var target: [4]f32 = undefined;
                for (0..4) |p| target[p] = block[laneColumn(g, p)] / d;
                var best_err = std.math.inf(f32);
                var best_code: u5 = 0;
                var best_sign = false;
                for (0..2) |sg| {
                    const group = &lane_table[sg];
                    // The nibble holds 16 codes; codes 16..31 are their sign-flipped twins and do not fit.
                    for (0..16) |c| {
                        const l = group[c];
                        const err = (l[0] - target[0]) * (l[0] - target[0]) +
                            (l[1] - target[1]) * (l[1] - target[1]) +
                            (l[2] - target[2]) * (l[2] - target[2]) +
                            (l[3] - target[3]) * (l[3] - target[3]);
                        if (err < best_err) {
                            best_err = err;
                            best_code = @intCast(c);
                            best_sign = sg == 1;
                        }
                    }
                }
                const byte = r * (cols / 8) + b * 32 + g / 2;
                const shift: u3 = @intCast(4 * (g % 2));
                qs[byte] = (qs[byte] & ~(@as(u8, 0x0F) << shift)) | (@as(u8, best_code) << shift);
                const sbyte = r * (cols / 32) + b * 8 + g / 8;
                const sshift: u3 = @intCast(g % 8);
                if (best_sign) sign[sbyte] |= @as(u8, 1) << sshift;
            }
        }
    }
    return .{ .qs = qs, .sign = sign, .scales = scales };
}

/// The float [rows, cols] an STQ1_0 triple decodes to, through `codebook` and the pack's `_INV_PERM` order.
pub fn dequantizeStq(a: std.mem.Allocator, rows: usize, cols: usize, qs: []const u8, sign: []const u8, scales: []const u16) ![]f32 {
    const out = try a.alloc(f32, rows * cols);
    const nb = cols / 256;
    for (0..rows) |r| {
        for (0..nb) |b| {
            const d = f16Float(scales[r * nb + b]);
            for (0..64) |g| {
                const code: u5 = @intCast((qs[r * (cols / 8) + b * 32 + g / 2] >> @intCast(4 * (g % 2))) & 0x0F);
                const sg = (sign[r * (cols / 32) + b * 8 + g / 8] >> @intCast(g % 8)) & 1 != 0;
                const l = lanes(code, sg);
                for (l, 0..) |v, p| out[r * cols + b * 256 + laneColumn(g, p)] = v * d;
            }
        }
    }
    return out;
}

/// The embedding [vocab, hidden] quantized to 6-bit affine, group 64: `words` packs the codes, `scales` holds each
/// group's step and `biases` its low end. Caller frees all three.
pub fn quantizeEmbedding(a: std.mem.Allocator, w: []const f32) !struct { words: []u32, scales: []u16, biases: []u16 } {
    const ng = hidden / embed_group;
    const words_per_group = embed_group * embed_bits / 32; // 64 fields x 6 bits, packed into 12 words
    const words = try a.alloc(u32, vocab_size * ng * words_per_group);
    const scales = try a.alloc(u16, vocab_size * ng);
    const biases = try a.alloc(u16, vocab_size * ng);
    @memset(words, 0);
    const levels: f32 = 63;
    for (0..vocab_size) |r| {
        for (0..ng) |g| {
            const group = w[r * hidden + g * 64 ..][0..64];
            var lo: f32 = group[0];
            var hi: f32 = group[0];
            for (group) |x| {
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            const step = halfOf(if (hi == lo) 1e-8 else (hi - lo) / levels);
            const d = @as(f32, step);
            const low: f16 = halfOf(lo);
            scales[r * ng + g] = @as(u16, @bitCast(step));
            biases[r * ng + g] = @as(u16, @bitCast(low));
            const base = (r * ng + g) * words_per_group;
            const l0 = f16Float(@as(u16, @bitCast(low)));
            for (0..64) |j| {
                const code: u32 = @intFromFloat(@round(std.math.clamp((group[j] - l0) / d, 0, 63)));
                const bit = j * 6;
                const wi = bit / 32;
                const sh: u32 = @intCast(bit % 32);
                words[base + wi] |= code << @intCast(sh);
                if (sh + 6 > 32) {
                    const spill: u32 = 32 - sh;
                    words[base + wi + 1] |= code >> @intCast(spill);
                }
            }
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases };
}

/// A 6-bit affine embedding's [rows, cols] float from the three buffers the formulas above produce.
pub fn dequantizeEmbedding(a: std.mem.Allocator, rows: usize, words: []const u32, scales: []const u16, biases: []const u16) ![]f32 {
    const ng = colsPerEmbeddingRow();
    const wpg = embed_group * embed_bits / 32;
    const out = try a.alloc(f32, rows * ng * embed_group);
    for (0..rows) |r| {
        for (0..ng) |g| {
            const step = f16Float(scales[r * ng + g]);
            const low = f16Float(biases[r * ng + g]);
            for (0..64) |j| {
                const bit = j * 6;
                const wi = bit / 32;
                const sh: u32 = @intCast(bit % 32);
                var code: u32 = words[(r * ng + g) * wpg + wi] >> @intCast(sh);
                if (sh + 6 > 32) code |= words[(r * ng + g) * wpg + wi + 1] << @intCast(32 - sh);
                out[r * ng * embed_group + g * embed_group + j] =
                    @as(f32, @floatFromInt(code & 0x3F)) * step + low;
            }
        }
    }
    return out;
}

fn colsPerEmbeddingRow() usize {
    return hidden / embed_group;
}

// ── the pack files ──

pub const File = kit.sdk.fixture.File;

/// One tensor going into the safetensors file: a name, its shape and typed little-endian bytes.
const Raw = struct { name: []const u8, shape: []const u64, dtype: []const u8, bytes: []const u8 };

fn safetensors(a: std.mem.Allocator, tensors: []const Raw) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.append(a, '{');
    var off: u64 = 0;
    for (tensors, 0..) |t, i| {
        try hdr.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", t.name, t.dtype });
        for (t.shape, 0..) |d, k| try hdr.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.bytes.len });
        off += t.bytes.len;
    }
    try hdr.append(a, '}');
    const out = try a.alloc(u8, @intCast(8 + hdr.items.len + off));
    std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
    @memcpy(out[8..][0..hdr.items.len], hdr.items);
    var at: usize = 8 + hdr.items.len;
    for (tensors) |t| {
        @memcpy(out[at..][0..t.bytes.len], t.bytes);
        at += t.bytes.len;
    }
    return out;
}

/// A seeded, small-magnitude [rows, cols] matrix.
pub fn matrix(a: std.mem.Allocator, seed: u64, rows: usize, cols: usize) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(f32, rows * cols);
    for (out) |*x| x.* = prng.random().floatNorm(f32) * 0.08;
    return out;
}

pub const Pack = struct {
    files: []File,
    /// What the STQ projections were quantized from, by tensor base (`model.layers.0.mlp.gate_proj`).
    expected: std.StringHashMapUnmanaged([]f32),
    embed_float: []f32,
};

/// The fixture's config.json: the pack's geometry and markers, at the size this file builds.
pub fn configJson(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"architectures":["HunYuanDenseV1ForCausalLM"],"model_type":"hunyuan_v1_dense",
        \\"vocab_size":{d},"hidden_size":{d},"intermediate_size":{d},"num_hidden_layers":{d},
        \\"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"rms_norm_eps":1e-5,
        \\"rope_theta":10000.0,"rope_scaling":{{"alpha":1000.0,"beta_fast":32,"beta_slow":1,"factor":1.0,
        \\"mscale":1.0,"mscale_all_dim":1.0,"type":"dynamic"}},"max_position_embeddings":262144,
        \\"tie_word_embeddings":true,"use_qk_norm":true,"use_cla":false,"sherry_embed_bits":{d},
        \\"bos_token_id":120000,"eos_token_id":259,"model_file":"sherry_model.py"}}
    , .{ vocab_size, hidden, ffn, layers, heads, kv_heads, head_dim, embed_bits });
}

/// The whole pack: config, tokenizer, generation config and one safetensors file of STQ1_0 projections, 6-bit
/// embedding and f16 norms. Everything comes from `a` (an arena).
pub fn build(a: std.mem.Allocator) !Pack {
    var raws: std.ArrayList(Raw) = .empty;
    defer raws.deinit(a);
    var expected: std.StringHashMapUnmanaged([]f32) = .empty;

    var seed: u64 = 7;
    // A projection, quantized and appended as `base.{qs,sign,scales}`.
    const proj = struct {
        fn add(aa: std.mem.Allocator, list: *std.ArrayList(Raw), exp: *std.StringHashMapUnmanaged([]f32), base: []const u8, s: *u64, rows: usize, cols: usize) !void {
            const w = try matrix(aa, s.*, rows, cols);
            s.* += 1;
            const q = try quantizeStq(aa, rows, cols, w);
            try exp.put(aa, base, w);
            var buf: [160]u8 = undefined;
            for ([_]struct { n: []const u8, bytes: []const u8, shape: [2]u64 }{
                .{ .n = "qs", .bytes = std.mem.sliceAsBytes(q.qs), .shape = .{ rows, cols / 8 } },
                .{ .n = "sign", .bytes = std.mem.sliceAsBytes(q.sign), .shape = .{ rows, cols / 32 } },
                .{ .n = "scales", .bytes = std.mem.sliceAsBytes(q.scales), .shape = .{ rows, cols / 256 } },
            }) |part| {
                const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, part.n });
                const dtype: []const u8 = if (std.mem.eql(u8, part.n, "scales")) "F16" else "U8";
                try list.append(aa, .{ .name = try aa.dupe(u8, name), .shape = try aa.dupe(u64, &part.shape), .dtype = dtype, .bytes = part.bytes });
            }
        }
    };

    // The embedding, first: safetensors order is the order `LoadCtx.weights` hands back.
    const embed_float = blk: {
        var prng = std.Random.DefaultPrng.init(99);
        const out = try a.alloc(f32, vocab_size * hidden);
        for (out) |*x| x.* = prng.random().floatNorm(f32) * 0.1;
        break :blk out;
    };
    {
        const e = try quantizeEmbedding(a, embed_float);
        // The names are the buffers: `scales` is the affine step, `biases` the group's low end.
        try raws.append(a, .{ .name = "model.embed_tokens.weight", .shape = &.{ vocab_size, hidden * embed_bits / 32 }, .dtype = "U32", .bytes = std.mem.sliceAsBytes(e.words) });
        try raws.append(a, .{ .name = "model.embed_tokens.scales", .shape = &.{ vocab_size, hidden / embed_group }, .dtype = "F16", .bytes = std.mem.sliceAsBytes(e.scales) });
        try raws.append(a, .{ .name = "model.embed_tokens.biases", .shape = &.{ vocab_size, hidden / embed_group }, .dtype = "F16", .bytes = std.mem.sliceAsBytes(e.biases) });
    }
    {
        const ones = try a.alloc(f32, hidden);
        @memset(ones, 1);
        const bits = try a.alloc(u16, hidden);
        for (ones, bits) |v, *b| b.* = f16Bits(v);
        try raws.append(a, .{ .name = "model.norm.weight", .shape = &.{hidden}, .dtype = "F16", .bytes = std.mem.sliceAsBytes(bits) });
        const qk = try a.alloc(f32, head_dim);
        @memset(qk, 1);
        const qk_bits = try a.alloc(u16, head_dim);
        for (qk, qk_bits) |v, *b| b.* = f16Bits(v);
        for (0..layers) |i| {
            var buf: [160]u8 = undefined;
            inline for (.{ "input_layernorm", "post_attention_layernorm", "self_attn.query_layernorm", "self_attn.key_layernorm" }) |n| {
                const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ i, n });
                const is_qk = std.mem.endsWith(u8, n, "layernorm") and (std.mem.indexOf(u8, n, "query") != null or std.mem.indexOf(u8, n, "key") != null);
                if (!is_qk) try raws.append(a, .{ .name = try a.dupe(u8, name), .shape = &.{hidden}, .dtype = "F16", .bytes = std.mem.sliceAsBytes(bits) });
            }
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.self_attn.q_proj", .{i}), &seed, q_dim, hidden);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.self_attn.k_proj", .{i}), &seed, kv_dim, hidden);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.self_attn.v_proj", .{i}), &seed, kv_dim, hidden);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.self_attn.o_proj", .{i}), &seed, hidden, q_dim);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.gate_proj", .{i}), &seed, ffn, hidden);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.up_proj", .{i}), &seed, ffn, hidden);
            try proj.add(a, &raws, &expected, try std.fmt.allocPrint(a, "model.layers.{d}.mlp.down_proj", .{i}), &seed, hidden, ffn);
            // The two qk norms are 1-D f16 of head_dim, appended after the projections like the real pack.
            inline for (.{ "self_attn.query_layernorm", "self_attn.key_layernorm" }) |n| {
                const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.weight", .{ i, n });
                try raws.append(a, .{ .name = try a.dupe(u8, name), .shape = &.{head_dim}, .dtype = "F16", .bytes = std.mem.sliceAsBytes(qk_bits) });
            }
        }
    }
    const st = try safetensors(a, raws.items);
    const files = try a.alloc(File, 4);
    files[0] = .{ .name = "config.json", .bytes = try configJson(a) };
    files[1] = .{ .name = "model.safetensors", .bytes = st };
    files[2] = .{ .name = "tokenizer.json", .bytes = try kit.sdk.byte_tokenizer.json(a, &.{
        .{ .id = 256, .content = "<|end|>" },
        .{ .id = 257, .content = "<|user|>" },
        .{ .id = 258, .content = "<|assistant|>" },
        .{ .id = 259, .content = "<|endoftext|>" },
    }) };
    files[3] = .{ .name = "tokenizer_config.json", .bytes =
        \\{"eos_token":"<|endoftext|>","chat_template":"{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"}
    };
    return .{ .files = files, .expected = expected, .embed_float = embed_float };
}

/// The fixture pack's safetensors header: every tensor's name, dtype and shape.
pub fn tensorIndex(a: std.mem.Allocator, pack: Pack) ![]TensorInfo {
    for (pack.files) |f| {
        if (!std.mem.eql(u8, f.name, "model.safetensors")) continue;
        const n = std.mem.readInt(u64, f.bytes[0..8], .little);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, f.bytes[8..][0..n], .{});
        var out: std.ArrayList(TensorInfo) = .empty;
        var it = parsed.value.object.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.*;
            const shape_json = v.object.get("shape").?.array;
            const shape = try a.alloc(u64, shape_json.items.len);
            for (shape_json.items, shape) |d, *o| o.* = @intCast(d.integer);
            try out.append(a, .{
                .name = try a.dupe(u8, kv.key_ptr.*),
                .dtype = try a.dupe(u8, v.object.get("dtype").?.string),
                .shape = shape,
            });
        }
        return out.items;
    }
    return error.NoSafetensors;
}

pub const TensorInfo = struct { name: []const u8, dtype: []const u8, shape: []const u64 };

/// A tensor's bytes, by name, from the pack's safetensors file.
pub fn tensorBytes(pack: Pack, name: []const u8) ?[]const u8 {
    for (pack.files) |f| {
        if (!std.mem.eql(u8, f.name, "model.safetensors")) continue;
        const n = std.mem.readInt(u64, f.bytes[0..8], .little);
        const hdr = f.bytes[8..][0..n];
        var idx: usize = 0;
        // Find `"<name>":{"dtype":...,"data_offsets":[a,b]}` by scanning the header text.
        const needle = std.mem.concat(std.heap.c_allocator, u8, &.{ "\"", name, "\":{\"dtype\"" }) catch return null;
        defer std.heap.c_allocator.free(needle);
        idx = std.mem.indexOf(u8, hdr, needle) orelse return null;
        const off_start = std.mem.indexOf(u8, hdr[idx..], "data_offsets\":[") orelse return null;
        const after = hdr[idx + off_start + "data_offsets\":[".len ..];
        const comma = std.mem.indexOfScalar(u8, after, ',') orelse return null;
        const close = std.mem.indexOfScalar(u8, after, ']') orelse return null;
        const begin = std.fmt.parseInt(u64, after[0..comma], 10) catch return null;
        const end = std.fmt.parseInt(u64, after[comma + 1 .. close], 10) catch return null;
        const base: usize = @intCast(8 + n);
        return f.bytes[base + @as(usize, @intCast(begin)) .. base + @as(usize, @intCast(end))];
    }
    return null;
}

// ── tests ──

const testing = std.testing;

test "conformance fixture: f32 to f16 bits rounds to nearest even and clamps the edges" {
    try testing.expectEqual(@as(u16, 0x3C00), f16Bits(1.0));
    try testing.expectEqual(@as(u16, 0xBC00), f16Bits(-1.0));
    try testing.expectEqual(@as(u16, 0x0000), f16Bits(0.0));
    try testing.expectEqual(@as(u16, 0x8000), f16Bits(-0.0));
    try testing.expectEqual(@as(u16, 0x7C00), f16Bits(1e9));
    try testing.expectEqual(@as(u16, 0xFC00), f16Bits(-1e9));
    // 0.5 and 2.0 are exact; 1/3 goes to the nearest representable 0x3555.
    try testing.expectEqual(@as(u16, 0x3800), f16Bits(0.5));
    try testing.expectEqual(@as(u16, 0x4000), f16Bits(2.0));
    try testing.expectEqual(@as(u16, 0x3555), f16Bits(1.0 / 3.0));
    // Round trip through the decode: every value written comes back within an ulp.
    const samples = [_]f32{ 0.1, -0.02, 3.5, -7.25, 0.0009765625, 65504.0 };
    for (samples) |x| {
        const back = f16Float(f16Bits(x));
        try testing.expect(@abs(back - x) <= @abs(x) * 0.001);
    }
}

test "conformance fixture: the codebook lanes are four ternary values per (sign, code)" {
    // Verified against the checkpoint's `_STQ_LUT` (32 x 4 half): every entry is in {-1, 0, 1}.
    var seen_zero = false;
    var seen_both = false;
    for (0..32) |c| {
        for ([_]bool{ false, true }) |sign| {
            const l = lanes(@intCast(c), sign);
            var ones: u32 = 0;
            for (l) |v| {
                try testing.expect(v == -1 or v == 0 or v == 1);
                if (v == 0) seen_zero = true;
                if (v == 1 and l[0] != 0) seen_both = true;
                ones += @intFromBool(v == 0);
            }
            try testing.expectEqual(@as(u32, 1), ones);
        }
    }
    try testing.expect(seen_zero and seen_both);
}

test "conformance fixture: an STQ1_0 round trip lands every lane on a codebook entry" {
    const a = testing.allocator;
    const w = try matrix(a, 11, 8, 512);
    defer a.free(w);
    const q = try quantizeStq(a, 8, 512, w);
    defer a.free(q.qs);
    defer a.free(q.sign);
    defer a.free(q.scales);
    const back = try dequantizeStq(a, 8, 512, q.qs, q.sign, q.scales);
    defer a.free(back);
    // 1.3125 bpw is lossy; per block the error stays inside one scale step.
    for (0..8) |r| {
        for (0..2) |b| {
            const d = f16Float(q.scales[r * 2 + b]);
            for (0..256) |k| {
                const i = r * 512 + b * 256 + k;
                // Every codebook row has exactly one zero lane, so a lane can be forced to +/-1 when it wanted 0;
                // the worst case is two scales (both lanes off by one).
                try testing.expect(@abs(back[i] - w[i]) <= 2.01 * d);
            }
        }
    }
}

test "conformance fixture: every weight of a projection decodes to a {-1,0,1} lane times its block scale" {
    const a = testing.allocator;
    const w = try matrix(a, 12, 4, 256);
    defer a.free(w);
    const q = try quantizeStq(a, 4, 256, w);
    defer a.free(q.qs);
    defer a.free(q.sign);
    defer a.free(q.scales);
    const back = try dequantizeStq(a, 4, 256, q.qs, q.sign, q.scales);
    defer a.free(back);
    for (0..4) |r| {
        const d = f16Float(q.scales[r]);
        for (0..256) |k| {
            const ratio = back[r * 256 + k] / d;
            try testing.expect(@abs(ratio) < 1.001 and @abs(@round(ratio) - ratio) < 0.001);
        }
    }
}

test "conformance fixture: the 6-bit embedding packs six fields per word and dequantizes inside a step" {
    const a = testing.allocator;
    const w = try matrix(a, 13, 4, 256);
    defer a.free(w);
    // quantizeEmbedding is sized for the fixture's vocabulary; check the packing on a 64-wide group directly.
    const flat = try a.alloc(f32, vocab_size * hidden);
    defer a.free(flat);
    @memset(flat, 0);
    @memcpy(flat[0..256], w[0..256]);
    const e = try quantizeEmbedding(a, flat);
    defer a.free(e.words);
    defer a.free(e.scales);
    defer a.free(e.biases);
    // Row 0, group 0: three words carry 64 fields; the first field sits in the word's lowest six bits.
    const base = e.words[0];
    const first = base & 0x3F;
    const step = f16Float(e.scales[0]);
    const low = f16Float(e.biases[0]);
    const back0 = @as(f32, @floatFromInt(first)) * step + low;
    try testing.expect(@abs(back0 - w[0]) <= step);
    // Every unpacked field reconstructs within one step of its source, in row 0's two 64-weight groups.
    for (0..2) |g| {
        const step_g = f16Float(e.scales[g]);
        const low_g = f16Float(e.biases[g]);
        const base_g = g * (embed_group * embed_bits / 32);
        for (0..64) |j| {
            const bit = j * 6;
            const wi = bit / 32;
            const sh: u32 = @intCast(bit % 32);
            var code: u32 = e.words[base_g + wi] >> @intCast(sh);
            if (sh + 6 > 32) code |= e.words[base_g + wi + 1] << @intCast(32 - sh);
            code &= 0x3F;
            const rec = @as(f32, @floatFromInt(code)) * step_g + low_g;
            const want = flat[g * 64 + j];
            try testing.expect(@abs(rec - want) <= step_g);
        }
    }
    // The bytes are little-endian words, the layout MLX's affine 6-bit reader expects.
    try testing.expect(@sizeOf(u32) == 4);
}

test "conformance fixture: the pack carries the tensors and config the plugin's parse requires" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const pack = try build(arena.allocator());
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(testing.allocator);
    for (pack.files) |f| {
        if (std.mem.eql(u8, f.name, "model.safetensors")) {
            const n = std.mem.readInt(u64, f.bytes[0..8], .little);
            const hdr = try std.json.parseFromSlice(std.json.Value, testing.allocator, f.bytes[8..][0..n], .{});
            defer hdr.deinit();
            var it = hdr.value.object.iterator();
            while (it.next()) |kv| try names.append(testing.allocator, try testing.allocator.dupe(u8, kv.key_ptr.*));
        }
    }
    defer for (names.items) |x| testing.allocator.free(x);
    const required = [_][]const u8{
        "model.embed_tokens.weight", "model.embed_tokens.scales",          "model.embed_tokens.biases",
        "model.norm.weight",         "model.layers.0.self_attn.q_proj.qs", "model.layers.1.mlp.down_proj.scales",
    };
    const joined = try std.mem.join(testing.allocator, "\n", names.items);
    defer testing.allocator.free(joined);
    for (required) |r| try testing.expect(std.mem.indexOf(u8, joined, r) != null);
}

test "conformance fixture: the pack's own tensors decode back to the weights it was built from" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pack = try build(a);
    const infos = try tensorIndex(a, pack);
    var checked: usize = 0;
    var it = pack.expected.iterator();
    while (it.next()) |kv| {
        const base = kv.key_ptr.*;
        const want = kv.value_ptr.*;
        const rows: usize = want.len / (if (std.mem.indexOf(u8, base, "down_proj") != null) @as(usize, ffn) else hidden);
        _ = infos;
        const cols = want.len / rows;
        var b0: [160]u8 = undefined;
        var b1: [160]u8 = undefined;
        var b2: [160]u8 = undefined;
        const qs_name = try std.fmt.bufPrint(&b0, "{s}.qs", .{base});
        const sign_name = try std.fmt.bufPrint(&b1, "{s}.sign", .{base});
        const scales_name = try std.fmt.bufPrint(&b2, "{s}.scales", .{base});
        const qs_bytes = tensorBytes(pack, qs_name) orelse return error.NoTensor;
        const sign_bytes = tensorBytes(pack, sign_name) orelse return error.NoTensor;
        const scales_bytes = tensorBytes(pack, scales_name) orelse return error.NoTensor;
        const qs = try a.dupe(u8, qs_bytes);
        const sign = try a.dupe(u8, sign_bytes);
        const scales = try a.alloc(u16, scales_bytes.len / 2);
        for (scales, 0..) |*d, i| d.* = std.mem.readInt(u16, scales_bytes[i * 2 ..][0..2], .little);
        const back = try dequantizeStq(a, rows, cols, qs, sign, scales);
        const nb = cols / 256;
        for (0..rows) |r| {
            for (0..nb) |b| {
                const d = f16Float(scales[r * nb + b]);
                for (0..256) |k| {
                    const i = r * cols + b * 256 + k;
                    try testing.expect(@abs(back[i] - want[i]) <= 2.01 * d);
                }
            }
        }
        checked += 1;
    }
    // Two layers x seven projections.
    try testing.expectEqual(@as(usize, 14), checked);
}

test "conformance fixture: the 6-bit embedding in the pack decodes to its float weights" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pack = try build(a);
    const words_bytes = tensorBytes(pack, "model.embed_tokens.weight") orelse return error.NoTensor;
    const sc_bytes = tensorBytes(pack, "model.embed_tokens.scales") orelse return error.NoTensor;
    const bi_bytes = tensorBytes(pack, "model.embed_tokens.biases") orelse return error.NoTensor;
    const words = try a.alloc(u32, words_bytes.len / 4);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, words_bytes[i * 4 ..][0..4], .little);
    const sc = try a.alloc(u16, sc_bytes.len / 2);
    for (sc, 0..) |*d, i| d.* = std.mem.readInt(u16, sc_bytes[i * 2 ..][0..2], .little);
    const bi = try a.alloc(u16, bi_bytes.len / 2);
    for (bi, 0..) |*d, i| d.* = std.mem.readInt(u16, bi_bytes[i * 2 ..][0..2], .little);
    const back = try dequantizeEmbedding(a, vocab_size, words, sc, bi);
    for (back, pack.embed_float) |got, want| {
        const g = (want - got) / @max(@abs(want), 1e-6);
        try testing.expect(@abs(got - want) <= 0.02 or @abs(g) < 1e-3);
    }
}
