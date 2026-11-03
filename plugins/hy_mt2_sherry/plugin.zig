//! Hy-MT2-1.8B 1.25-bit MLX packs on mlx-serve: `hunyuan_v1_dense` with Sherry's STQ1_0 linears and the 6-bit
//! affine embedding, ported from the pack's `sherry_model.py` (kuotient/Hy-MT2-1.8B-1.25Bit-MLX).
//!
//! The layout is not MLX's own: every projection is three buffers (`qs` u8 [N, K/8], `sign` u8 [N, K/32],
//! `scales` f16 [N, K/256]) holding llama.cpp's STQ1_0 codebook, so the forward owns its matmul:
//!
//!   * up to 64 rows per slot (decode, speculative verify, short prompts): the pack's Metal GEMV. One row takes the
//!     per-token kernel; several rows and/or slots take the batch kernel, which loads each packed weight word once and
//!     dots it against every sequence. The 1.31-bpw buffers are read in place, no dense copy.
//!   * above 64 rows per slot (a long prompt): the pack's one-shot dequant kernel expands one projection to a dense
//!     f16 tile and the stock GEMM runs; the tile is transient and the 1.31-bpw buffers stay resident. The residual is
//!     evaluated per layer so the tiles of 32 layers never pile up in the lazy graph (the same reason sherry_model.py
//!     evals per layer).
//!
//! Everything else (tokenizer, chat template, sampling, KV cache, continuous batching, prefix cache) is the host's.

const std = @import("std");
const kit = @import("mlx_serve");
const c = @import("mlx_c");
const abi = kit.abi.plugin;
pub const config = @import("config.zig");

/// Shown at install and in Settings.
pub const version = "0.1.0";

pub const plugin = kit.sdk.Plugin{
    .name = "hy_mt2_sherry",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{},
    .weights = &.{},
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

/// Output rows per simdgroup in the GEMV (the pack's measured default on Apple silicon).
const r_rows: c_int = 8;
const qk_k: u32 = 256;

// ── the pack's Metal sources ──

const stq_codebook = [32]u8{ 169, 137, 41, 9, 166, 134, 38, 6, 154, 146, 26, 18, 106, 98, 74, 66, 1, 33, 129, 161, 4, 36, 132, 164, 16, 24, 144, 152, 64, 72, 96, 104 };

const codebook: []const u8 = "constant constexpr const uint8_t STQ_CODEBOOK[32] = {" ++
    "169,137,41,9,166,134,38,6,154,146,26,18,106,98,74,66," ++
    "1,33,129,161,4,36,132,164,16,24,144,152,64,72,96,104};\n" ++
    "constant half4 STQ_CB4[32] = {" ++
    "half4(0,1,1,1),half4(0,1,-1,1),half4(0,1,1,-1),half4(0,1,-1,-1)," ++
    "half4(1,0,1,1),half4(1,0,-1,1),half4(1,0,1,-1),half4(1,0,-1,-1)," ++
    "half4(1,1,0,1),half4(1,-1,0,1),half4(1,1,0,-1),half4(1,-1,0,-1)," ++
    "half4(1,1,1,0),half4(1,-1,1,0),half4(1,1,-1,0),half4(1,-1,-1,0)," ++
    "half4(0,-1,-1,-1),half4(0,-1,1,-1),half4(0,-1,-1,1),half4(0,-1,1,1)," ++
    "half4(-1,0,-1,-1),half4(-1,0,1,-1),half4(-1,0,-1,1),half4(-1,0,1,1)," ++
    "half4(-1,-1,0,-1),half4(-1,1,0,-1),half4(-1,-1,0,1),half4(-1,1,0,1)," ++
    "half4(-1,-1,-1,0),half4(-1,1,-1,0),half4(-1,-1,1,0),half4(-1,1,1,0)};\n";

const metal_header: [:0]const u8 = "#include <metal_stdlib>\nusing namespace metal;\n" ++ codebook;

/// sherry_model.py `_GEMV_SRC` (decode, one row per grid.z). `x` is [tokens, K] f16, `qs`/`sgn` u8, `scales` f16,
/// `out` [tokens, N] f16. Grid (32, N/R_ROWS, tokens), threadgroup (32, 1, 1).
const gemv_src: [:0]const u8 =
    \\    uint row0  = thread_position_in_grid.y * R_ROWS;
    \\    uint batch = thread_position_in_grid.z;
    \\    uint lane  = thread_position_in_grid.x;
    \\
    \\    const device T* x_b = x + (size_t)batch * K;
    \\
    \\    float acc[R_ROWS];
    \\    #pragma clang loop unroll(full)
    \\    for (uint r = 0; r < R_ROWS; ++r) acc[r] = 0.0f;
    \\
    \\    const uint m  = lane & 7u;
    \\    const uint h  = m >> 1u;
    \\    const uint q0 = 8u * (m & 1u);
    \\
    \\    for (uint ch = 0; ch < NCH; ++ch) {
    \\      for (uint it = 0; it < 2; ++it) {
    \\        const uint blk = it * 4u + ((8u * lane) >> 6);
    \\        const uint xoff = ch * 2048u + blk * 256u + h * 64u + q0;
    \\        const device T* xc = x_b + xoff;
    \\
    \\        half4 xr[8];
    \\        xr[0] = *((const device half4*)(xc + 0u));
    \\        xr[1] = *((const device half4*)(xc + 4u));
    \\        xr[2] = *((const device half4*)(xc + 16u));
    \\        xr[3] = *((const device half4*)(xc + 20u));
    \\        xr[4] = *((const device half4*)(xc + 32u));
    \\        xr[5] = *((const device half4*)(xc + 36u));
    \\        xr[6] = *((const device half4*)(xc + 48u));
    \\        xr[7] = *((const device half4*)(xc + 52u));
    \\
    \\        #pragma clang loop unroll(full)
    \\        for (uint r = 0; r < R_ROWS; ++r) {
    \\            const uint row = row0 + r;
    \\            uint qw = ((const device uint*)(qs + (size_t)row * (K / 8) + ch * 256u + it * 128u))[lane];
    \\            uchar sw = *((const device uchar*)(sgn + (size_t)row * (K / 32) + ch * 64u + it * 32u) + lane);
    \\            float inner = 0.0f;
    \\            #pragma clang loop unroll(full)
    \\            for (uint c = 0; c < 8; ++c) {
    \\                const uint pc = (qw >> (c * 4u)) & 0x0Fu;
    \\                const uint G  = (uint)((sw >> c) & 1u);
    \\                const uint i  = c & 3u;
    \\                const uint k  = c >> 2u;
    \\                half wa = (pc & 12u) ? xr[0+k][i] : xr[2+k][i];
    \\                half wb = (pc & 8u)  ? xr[2+k][i] : xr[4+k][i];
    \\                half wc = ((pc & 12u) == 12u) ? xr[4+k][i] : xr[6+k][i];
    \\                half u = wa + ((pc & 1u) ? -wb : wb) + ((pc & 2u) ? -wc : wc);
    \\                inner += (float)(G ? -u : u);
    \\            }
    \\            half scl = scales[(size_t)row * NB + ch * 8u + blk];
    \\            acc[r] += inner * (float)scl;
    \\        }
    \\      }
    \\    }
    \\
    \\    #pragma clang loop unroll(full)
    \\    for (uint r = 0; r < R_ROWS; ++r) {
    \\        float total = simd_sum(acc[r]);
    \\        if (lane == 0) {
    \\            out[(size_t)batch * N + row0 + r] = (T)total;
    \\        }
    \\    }
;

/// sherry_model.py `_DEQUANT_SRC` (prefill). Expands [N, K] packed STQ1_0 to a dense [N, K] f16 tile, scale folded.
/// Grid (8, N, N_blocks), threadgroup (8, 1, 1).
const dequant_src: [:0]const u8 =
    \\    const uint w = thread_position_in_grid.x;   // qs word 0..7
    \\    const uint row = thread_position_in_grid.y;
    \\    const uint b = thread_position_in_grid.z;
    \\    const device uint8_t* qs_r = qs  + (size_t)row * (K / 8)  + b * 32u;
    \\    const device uint8_t* sg_r = sgn + (size_t)row * (K / 32) + b * 8u;
    \\    const uint8_t sg8 = sg_r[w];
    \\    const float d = (float)scales[(size_t)row * NB + b];
    \\    const uint xbase = ((w >> 1) << 6) + (w & 1u) * 8u;
    \\    #pragma clang loop unroll(full)
    \\    for (uint j = 0; j < 8u; ++j) {
    \\        const uint code = (qs_r[4u * w + (j >> 1)] >> ((j & 1u) * 4u)) & 0xFu;
    \\        const uint sbit = (sg8 >> j) & 1u;
    \\        const half4 wq = STQ_CB4[(sbit << 4) | code];
    \\        const size_t base = (size_t)row * K + b * 256u + xbase + j;
    \\        dense[base +  0u] = (half)((float)wq.x * d);
    \\        dense[base + 16u] = (half)((float)wq.y * d);
    \\        dense[base + 32u] = (half)((float)wq.z * d);
    \\        dense[base + 48u] = (half)((float)wq.w * d);
    \\    }
;

/// sherry_model.py `_BATCH_GEMV_SRC` (multi-slot decode): x is [BAT, K], one token per sequence, and every packed
/// weight word is loaded once and dotted against all sequences. Grid (32, N/R_ROWS, 1), threadgroup (32, 1, 1).
const batch_gemv_src: [:0]const u8 =
    \\    uint row0 = thread_position_in_grid.y * R_ROWS;
    \\    uint lane = thread_position_in_grid.x;
    \\
    \\    const uint CPL = 4u;
    \\    const uint LPB = 64u / CPL;                 // lanes covering one block
    \\    const uint NIT = K / (128u * CPL);          // weight-load iterations
    \\    const uint l   = lane & (LPB - 1u);
    \\    const uint m   = (CPL * l) >> 3u;
    \\    const uint cb  = (CPL * l) & 7u;
    \\    const uint xloc = 64u * (m >> 1u) + 8u * (m & 1u) + cb;
    \\
    \\    float acc[BAT][R_ROWS];
    \\    #pragma clang loop unroll(full)
    \\    for (uint b = 0; b < BAT; ++b)
    \\        #pragma clang loop unroll(full)
    \\        for (uint r = 0; r < R_ROWS; ++r) acc[b][r] = 0.0f;
    \\
    \\    for (uint it = 0; it < NIT; ++it) {
    \\        const uint blk = ((CPL * lane) >> 6u) + it * (CPL / 2u);
    \\
    \\        half y[BAT][4u * CPL];
    \\        for (uint b = 0; b < BAT; ++b) {
    \\            const device T* xp = x + (size_t)b * K + (size_t)blk * 256u + xloc;
    \\            #pragma clang loop unroll(full)
    \\            for (uint s = 0; s < 4u; ++s)
    \\                #pragma clang loop unroll(full)
    \\                for (uint t = 0; t < CPL / 4u; ++t) {
    \\                    half4 v = *((const device half4*)(xp + 16u * s + 4u * t));
    \\                    y[b][s * CPL + 4u * t + 0u] = v.x;
    \\                    y[b][s * CPL + 4u * t + 1u] = v.y;
    \\                    y[b][s * CPL + 4u * t + 2u] = v.z;
    \\                    y[b][s * CPL + 4u * t + 3u] = v.w;
    \\                }
    \\        }
    \\
    \\        #pragma clang loop unroll(full)
    \\        for (uint r = 0; r < R_ROWS; ++r) {
    \\            const uint row = row0 + r;
    \\            ushort qw = ((const device ushort*)(qs + (size_t)row * (K / 8)
    \\                                            + it * (16u * CPL)))[lane];
    \\            uint sw = (uint)sgn[(size_t)row * (K / 32) + it * (4u * CPL)
    \\                                + ((CPL * lane) >> 3u)] >> ((CPL * lane) & 7u);
    \\
    \\            half WT[4u][CPL];
    \\            #pragma clang loop unroll(full)
    \\            for (uint j = 0; j < CPL; ++j) {
    \\                const uint pc = (uint)(qw >> (j * 4u)) & 0x0Fu;
    \\                const uint G  = (sw >> j) & 1u;
    \\                half4 w = ((const device half4*)lut)[(G << 4) | pc];
    \\                WT[0][j] = w.x;
    \\                WT[1][j] = w.y;
    \\                WT[2][j] = w.z;
    \\                WT[3][j] = w.w;
    \\            }
    \\            half scl = scales[(size_t)row * NB + blk];
    \\
    \\            #pragma clang loop unroll(full)
    \\            for (uint b = 0; b < BAT; ++b) {
    \\                half inner = 0.0h;
    \\                #pragma clang loop unroll(full)
    \\                for (uint s = 0; s < 4u; ++s)
    \\                    #pragma clang loop unroll(full)
    \\                    for (uint j = 0; j < CPL; ++j)
    \\                        inner += WT[s][j] * y[b][s * CPL + j];
    \\                acc[b][r] += (float)inner * (float)scl;
    \\            }
    \\        }
    \\    }
    \\
    \\    #pragma clang loop unroll(full)
    \\    for (uint b = 0; b < BAT; ++b)
    \\        #pragma clang loop unroll(full)
    \\        for (uint r = 0; r < R_ROWS; ++r) {
    \\            float total = simd_sum(acc[b][r]);
    \\            if (lane == 0) out[(size_t)b * N + row0 + r] = (T)total;
    \\        }
;

// ── weights ──

const Stq = struct {
    n: u32,
    k: u32,
    qs: c.mlx_array = .{},
    sign: c.mlx_array = .{},
    scales: c.mlx_array = .{},
};

const Layer = struct {
    ln1: c.mlx_array = .{},
    ln2: c.mlx_array = .{},
    qn: c.mlx_array = .{},
    kn: c.mlx_array = .{},
    q: Stq,
    k: Stq,
    v: Stq,
    o: Stq,
    gate: Stq,
    up: Stq,
    down: Stq,
};

const Model = struct {
    cfg: config.Config,
    caps: abi.Caps,
    stream: c.mlx_stream,
    host: kit.host.Host,
    embed_w: c.mlx_array = .{},
    embed_scales: c.mlx_array = .{},
    embed_biases: c.mlx_array = .{},
    norm: c.mlx_array = .{},
    freqs: c.mlx_array = .{},
    layers: []Layer,
    gemv: c.mlx_fast_metal_kernel = .{},
    batch_gemv: c.mlx_fast_metal_kernel = .{},
    dequant: c.mlx_fast_metal_kernel = .{},
    /// [32, 4] f16: each (sign << 4 | code) entry's four ternary lanes, the batch GEMV's table.
    stq_lut: c.mlx_array = .{},
};

fn stq(out_features: u32, in_features: u32) Stq {
    return .{ .n = out_features, .k = in_features };
}

fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    const k = config.of(p);
    out.* = .{
        .load_bytes = config.residentBytes(k) + (32 << 20), // + kernels, freqs, graph handles
        // 2 (K and V) x kv_heads x head_dim x layers x 2 B (f16).
        .kv_per_token = 2 * @as(u64, k.num_key_value_heads) * k.head_dim * k.num_hidden_layers * 2,
        // Every projection's dense tile is live until the layer's evalKeep: the pack's own geometry is 92 MiB.
        .prefill_streams = config.perLayerPrefillBytes(k),
    };
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) ?c.mlx_fast_metal_kernel {
    const in_vec = c.mlx_vector_string_new_data(@ptrCast(@constCast(ins.ptr)), ins.len);
    defer _ = c.mlx_vector_string_free(in_vec);
    const out_vec = c.mlx_vector_string_new_data(@ptrCast(@constCast(outs.ptr)), outs.len);
    defer _ = c.mlx_vector_string_free(out_vec);
    const k = c.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, metal_header.ptr, true, false);
    return if (k.ctx == null) null else k;
}

fn newRef(src: c.mlx_array) c.mlx_array {
    var r = c.mlx_array_new();
    _ = c.mlx_array_set(&r, src);
    return r;
}

/// Take tensor `name` into `dst`, refusing (by name) one whose dtype or shape is not what the kernels read: config
/// JSON is untrusted, and a short `qs` buffer would make the GEMV read past it.
fn takeShaped(src: *const kit.abi.quant.Source, dst: *c.mlx_array, name: []const u8, dtype: c.mlx_dtype, shape: []const c_int) bool {
    const t = src.tensor(name);
    if (t.ctx == null) {
        noteRefusal(name, .missing);
        return false;
    }
    const a: c.mlx_array = .{ .ctx = t.ctx };
    // The host narrows f16 tables to bf16 at load (`[dtype] narrowed ... 1-D f16 tables to bf16`); `narrowF16` casts
    // every table the kernels read back, so either float width is accepted here and only the shape must match.
    const float_ok = dtype == @as(c.mlx_dtype, @intCast(c.MLX_FLOAT16)) and
        (c.mlx_array_dtype(a) == @as(c.mlx_dtype, @intCast(c.MLX_FLOAT16)) or
            c.mlx_array_dtype(a) == @as(c.mlx_dtype, @intCast(c.MLX_BFLOAT16)));
    const ok = (float_ok or c.mlx_array_dtype(a) == dtype) and
        shapeMatches(c.mlx_array_dtype(a), @intCast(c.mlx_array_ndim(a)), c.mlx_array_shape(a), c.mlx_array_dtype(a), shape);
    if (!ok) {
        noteRefusal(name, .shape);
        return false;
    }
    return c.mlx_array_set(dst, a) == 0;
}

/// The first tensor `load` refused and why: static, so the refusal can name it (one load runs at a time).
var refused_name: [128]u8 = undefined;
var refused_len: usize = 0;
var refused_reason: enum { none, missing, shape } = .none;

fn noteRefusal(name: []const u8, reason: @TypeOf(refused_reason)) void {
    refused_len = @min(name.len, refused_name.len);
    @memcpy(refused_name[0..refused_len], name[0..refused_len]);
    refused_reason = reason;
}

/// Whether a tensor's dtype and shape are exactly what a kernel was told to read.
pub fn shapeMatches(got_dtype: c.mlx_dtype, got_dim: c_int, got: [*c]const c_int, want_dtype: c.mlx_dtype, want: []const c_int) bool {
    if (got_dtype != want_dtype) return false;
    if (got_dim != @as(c_int, @intCast(want.len))) return false;
    for (want, 0..) |w, i| if (got[i] != w) return false;
    return true;
}

/// `[rows, cols]` of `dtype`.
fn take2(src: *const kit.abi.quant.Source, dst: *c.mlx_array, name: []const u8, dtype: c.mlx_dtype, rows: u32, cols: u32) bool {
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    return takeShaped(src, dst, name, dtype, &shape);
}

/// `[n]` of `dtype`.
fn take1(src: *const kit.abi.quant.Source, dst: *c.mlx_array, name: []const u8, dtype: c.mlx_dtype, n: u32) bool {
    const shape = [_]c_int{@intCast(n)};
    return takeShaped(src, dst, name, dtype, &shape);
}

/// The host narrows every 1-D f16 table (the RMSNorm weights) to bf16 at load; mixing bf16 with f16 activations
/// promotes MLX's ops to f32, and the STQ GEMV reads `x` through `device half4*`. Keep every operand the kernels read
/// fp16, as mlx-lm runs the pack.
fn narrowF16(arr: *c.mlx_array, s: c.mlx_stream) !void {
    if (c.mlx_array_dtype(arr.*) == c.MLX_FLOAT16) return;
    var r = c.mlx_array_new();
    errdefer _ = c.mlx_array_free(r);
    if (c.mlx_astype(&r, arr.*, @intCast(c.MLX_FLOAT16), s) != 0) return opFailed();
    _ = c.mlx_array_free(arr.*);
    arr.* = r;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const why = "hy_mt2_sherry: ";
    const src = ctx.weights orelse {
        out.* = Verdict.refuse(.unsupported, why ++ "the engine did not load the pack's tensors");
        return null;
    };
    const cfg = config.of(ctx.parsed).*;
    const m = gpa.create(Model) catch {
        out.* = Verdict.refuse(.failed, why ++ "out of memory");
        return null;
    };
    const layers = gpa.alloc(Layer, cfg.num_hidden_layers) catch {
        gpa.destroy(m);
        out.* = Verdict.refuse(.failed, why ++ "out of memory");
        return null;
    };
    const q: u32 = cfg.num_attention_heads * cfg.head_dim;
    const kv: u32 = cfg.num_key_value_heads * cfg.head_dim;
    for (layers) |*l| l.* = .{
        .q = stq(q, cfg.hidden_size),
        .k = stq(kv, cfg.hidden_size),
        .v = stq(kv, cfg.hidden_size),
        .o = stq(cfg.hidden_size, q),
        .gate = stq(cfg.intermediate_size, cfg.hidden_size),
        .up = stq(cfg.intermediate_size, cfg.hidden_size),
        .down = stq(cfg.hidden_size, cfg.intermediate_size),
    };
    m.* = .{
        .cfg = cfg,
        .caps = ctx.parsed.caps,
        .stream = .{ .ctx = ctx.stream.ctx },
        .host = kit.host.Host.of(ctx.host) catch {
            gpa.free(layers);
            gpa.destroy(m);
            out.* = Verdict.refuse(.unsupported, why ++ "the host table is missing");
            return null;
        },
        .layers = layers,
    };

    // Kernels first so a shader that will not compile refuses before 454 MB of tensors are taken.
    m.gemv = makeKernel("stq1_gemv_v9c8", &.{ "x", "qs", "sgn", "scales" }, &.{"out"}, gemv_src.ptr) orelse {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the STQ GEMV kernel does not compile");
        return null;
    };
    m.dequant = makeKernel("stq1_dequant", &.{ "qs", "sgn", "scales" }, &.{"dense"}, dequant_src.ptr) orelse {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the STQ dequant kernel does not compile");
        return null;
    };
    m.batch_gemv = makeKernel("stq1_batch_gemv_bcl4", &.{ "x", "qs", "sgn", "scales", "lut" }, &.{"out"}, batch_gemv_src.ptr) orelse {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the STQ batch GEMV kernel does not compile");
        return null;
    };
    var lut: [32 * 4]f16 = undefined;
    for (stq_codebook, 0..) |qp, i| {
        for (0..4) |lane_i| {
            const code_i: i8 = @intCast((qp >> @intCast(2 * lane_i)) & 3);
            lut[i * 4 + lane_i] = @floatFromInt(code_i - 1);
        }
    }
    m.stq_lut = c.mlx_array_new_data(&lut, &[_]c_int{ 32, 4 }, 2, @intCast(c.MLX_FLOAT16));

    const u8_dt: c.mlx_dtype = @intCast(c.MLX_UINT8);
    const u32_dt: c.mlx_dtype = @intCast(c.MLX_UINT32);
    const f16_dt: c.mlx_dtype = @intCast(c.MLX_FLOAT16);
    const ng = cfg.hidden_size / cfg.embed_group;
    const mut = cfg.vocab_size;
    var all = take2(src, &m.embed_w, "model.embed_tokens.weight", u32_dt, mut, cfg.hidden_size * cfg.embed_bits / 32) and
        take2(src, &m.embed_scales, "model.embed_tokens.scales", f16_dt, mut, ng) and
        take2(src, &m.embed_biases, "model.embed_tokens.biases", f16_dt, mut, ng) and
        take1(src, &m.norm, "model.norm.weight", f16_dt, cfg.hidden_size);
    var name_buf: [128]u8 = undefined;
    for (layers, 0..) |*l, i| {
        inline for (.{
            .{ "input_layernorm", "ln1" },          .{ "post_attention_layernorm", "ln2" },
            .{ "self_attn.query_layernorm", "qn" }, .{ "self_attn.key_layernorm", "kn" },
        }) |w| {
            const name = std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}.weight", .{ i, w[0] }) catch unreachable;
            const width: u32 = if (std.mem.indexOf(u8, w[0], "query") != null or std.mem.indexOf(u8, w[0], "key") != null)
                cfg.head_dim
            else
                cfg.hidden_size;
            all = all and take1(src, &@field(l, w[1]), name, f16_dt, width);
        }
        inline for (.{
            .{ "self_attn.q_proj", "q" }, .{ "self_attn.k_proj", "k" }, .{ "self_attn.v_proj", "v" },
            .{ "self_attn.o_proj", "o" }, .{ "mlp.gate_proj", "gate" }, .{ "mlp.up_proj", "up" },
            .{ "mlp.down_proj", "down" },
        }) |w| {
            const base = std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}.", .{ i, w[0] }) catch unreachable;
            const st = &@field(l, w[1]);
            var buf2: [128]u8 = undefined;
            const qs_name = std.fmt.bufPrint(&buf2, "{s}qs", .{base}) catch unreachable;
            all = all and take2(src, &st.qs, qs_name, u8_dt, st.n, st.k / 8);
            const sign_name = std.fmt.bufPrint(&buf2, "{s}sign", .{base}) catch unreachable;
            all = all and take2(src, &st.sign, sign_name, u8_dt, st.n, st.k / 32);
            const scales_name = std.fmt.bufPrint(&buf2, "{s}scales", .{base}) catch unreachable;
            all = all and take2(src, &st.scales, scales_name, f16_dt, st.n, st.k / qk_k);
        }
    }
    if (!all) {
        const named = refused_name[0..refused_len];
        unload(m);
        out.* = switch (refused_reason) {
            .missing => Verdict.named(.failed, "hy_mt2_sherry: a weight is missing"),
            .shape => Verdict.named(.failed, "hy_mt2_sherry: a weight has the wrong shape or dtype"),
            .none => Verdict.named(.failed, "hy_mt2_sherry: a weight is missing"),
        };
        if (refused_len > 0) {
            std.debug.print("[hy_mt2_sherry] refused {s}: {s}\n", .{ @tagName(refused_reason), named });
        }
        return null;
    }

    narrowF16(&m.embed_scales, m.stream) catch {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the embedding scales do not cast to f16");
        return null;
    };
    narrowF16(&m.embed_biases, m.stream) catch {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the embedding biases do not cast to f16");
        return null;
    };
    narrowF16(&m.norm, m.stream) catch {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the final norm does not cast to f16");
        return null;
    };
    for (layers) |*l| {
        inline for (.{ "ln1", "ln2", "qn", "kn" }) |f| narrowF16(&@field(l, f), m.stream) catch {
            unload(m);
            out.* = Verdict.refuse(.failed, why ++ "a norm weight does not cast to f16");
            return null;
        };
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |f| narrowF16(&@field(l, f).scales, m.stream) catch {
            unload(m);
            out.* = Verdict.refuse(.failed, why ++ "a scale table does not cast to f16");
            return null;
        };
    }

    // DynamicNTKAlphaRoPE: base * alpha^(dim/(dim-2)), then base^(2i/dim) per pair.
    const hd: f64 = @floatFromInt(cfg.head_dim);
    const base: f64 = @as(f64, cfg.rope_theta) * std.math.pow(f64, cfg.rope_alpha, hd / (hd - 2));
    var freqs: [256]f32 = undefined;
    for (0..cfg.head_dim / 2) |i| {
        freqs[i] = @floatCast(std.math.pow(f64, base, 2.0 * @as(f64, @floatFromInt(i)) / hd));
    }
    const shape = [_]c_int{@intCast(cfg.head_dim / 2)};
    m.freqs = c.mlx_array_new_data(&freqs, &shape, 1, @intCast(c.MLX_FLOAT32));
    const vec = c.mlx_vector_array_new_data(&[_]c.mlx_array{m.freqs}, 1);
    defer _ = c.mlx_vector_array_free(vec);
    if (c.mlx_eval(vec) != 0) {
        unload(m);
        out.* = Verdict.refuse(.failed, why ++ "the RoPE frequencies do not evaluate");
        return null;
    }
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    _ = c.mlx_array_free(m.embed_w);
    _ = c.mlx_array_free(m.embed_scales);
    _ = c.mlx_array_free(m.embed_biases);
    _ = c.mlx_array_free(m.norm);
    _ = c.mlx_array_free(m.freqs);
    for (m.layers) |l| {
        inline for (.{ "ln1", "ln2", "qn", "kn" }) |f| _ = c.mlx_array_free(@field(l, f));
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |f| {
            _ = c.mlx_array_free(@field(l, f).qs);
            _ = c.mlx_array_free(@field(l, f).sign);
            _ = c.mlx_array_free(@field(l, f).scales);
        }
    }
    if (m.gemv.ctx != null) c.mlx_fast_metal_kernel_free(m.gemv);
    if (m.batch_gemv.ctx != null) c.mlx_fast_metal_kernel_free(m.batch_gemv);
    if (m.dequant.ctx != null) c.mlx_fast_metal_kernel_free(m.dequant);
    _ = c.mlx_array_free(m.stq_lut);
    gpa.free(m.layers);
    gpa.destroy(m);
}

// ── the forward ──

fn opFailed() error{MlxOpFailed} {
    return error.MlxOpFailed;
}

fn expectOk(rc: c_int) !void {
    if (rc != 0) return opFailed();
}

/// Arrays of one forward, freed together. `evalKeep` drops everything but the residual after an eval, so a
/// prompt's transient dequant tiles do not survive the layer that made them.
const Graph = struct {
    s: c.mlx_stream,
    live: std.ArrayList(c.mlx_array) = .empty,

    fn deinit(g: *Graph) void {
        for (g.live.items) |x| _ = c.mlx_array_free(x);
        g.live.deinit(gpa);
    }

    fn track(g: *Graph, x: c.mlx_array) !void {
        try g.live.append(gpa, x);
    }

    fn op(g: *Graph, comptime f: anytype, args: anytype) !c.mlx_array {
        var r = c.mlx_array_new();
        const outp: [*c]c.mlx_array = &r;
        if (@call(.auto, f, .{outp} ++ args ++ .{g.s}) != 0) {
            _ = c.mlx_array_free(r);
            return opFailed();
        }
        try g.live.append(gpa, r);
        return r;
    }

    fn reshape(g: *Graph, x: c.mlx_array, shape: []const c_int) !c.mlx_array {
        return g.op(c.mlx_reshape, .{ x, shape.ptr, shape.len });
    }

    fn linear(g: *Graph, x: c.mlx_array, w: c.mlx_array) !c.mlx_array {
        const axes = [_]c_int{ 1, 0 };
        return g.op(c.mlx_matmul, .{ x, try g.op(c.mlx_transpose_axes, .{ w, &axes, axes.len }) });
    }

    fn swap12(g: *Graph, x: c.mlx_array) !c.mlx_array {
        const axes = [_]c_int{ 0, 2, 1, 3 };
        return g.op(c.mlx_transpose_axes, .{ x, &axes, axes.len });
    }

    /// Evaluate `keep`, then free every other array this graph made and keep exactly one reference on `keep`.
    /// Owns a new reference on `keep` (the caller's handle is freed like the rest), so a caller whose `keep` is a
    /// plain local array must hand the result back to the caller's variable.
    fn evalKeep(g: *Graph, keep: c.mlx_array) !c.mlx_array {
        const vec = c.mlx_vector_array_new_data(&[_]c.mlx_array{keep}, 1);
        defer _ = c.mlx_vector_array_free(vec);
        try expectOk(c.mlx_eval(vec));
        var out = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(out);
        try expectOk(c.mlx_array_set(&out, keep));
        for (g.live.items) |a| _ = c.mlx_array_free(a);
        g.live.clearRetainingCapacity();
        try g.live.append(gpa, out);
        return out;
    }
};

fn someInt(v: u32) c.mlx_optional_int {
    return .{ .value = @intCast(v), .has_value = true };
}

fn emptyArray() c.mlx_array {
    return c.mlx_array_new();
}

fn exportArray(x: c.mlx_array) abi.Array {
    return .{ .ctx = newRef(x).ctx };
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "hy_mt2_sherry: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "hy_mt2_sherry: no layer capture");
    const ops = f.cache_ops orelse return Verdict.refuse(.failed, "hy_mt2_sherry: a decoder needs the engine's cache");
    const slots = f.caches.len;
    if (slots == 0 or f.rows == 0 or f.positions.len != slots)
        return Verdict.refuse(.failed, "hy_mt2_sherry: tokens are not rows x slots");
    if (f.ids.ctx == null) {
        const ids = f.tokens.slice();
        if (ids.len != f.rows * slots) return Verdict.refuse(.failed, "hy_mt2_sherry: tokens are not rows x slots");
        for (ids) |t| if (t >= m.cfg.vocab_size) return Verdict.refuse(.failed, "hy_mt2_sherry: token id past the vocabulary");
    }
    run(m, f, ops, out) catch return Verdict.refuse(.failed, "hy_mt2_sherry: an MLX op failed");
    return .ok;
}

/// Multi-slot decode: every slot is one row, and each projection reads its packed weights once for all of them
/// (the pack's `_BATCH_GEMV_SRC`). Attention stays per slot, because each slot owns its KV cache and position.
/// Up to 64 rows per slot (decode, speculative verify, short prompts): every projection reads its packed weights
/// once for all rows and all slots (the pack's batch GEMV); attention stays per slot because each slot owns its KV
/// cache and position. Rows longer than that take the per-slot dequant + GEMM path in `run`.
fn runTrunk(m: *Model, f: *const abi.Forward, ops: *const abi.CacheOps, out: *abi.ForwardOut) !void {
    const k = m.cfg;
    const slots_u32: u32 = @intCast(f.caches.len);
    const slots: c_int = @intCast(slots_u32);
    const rows_u32: u32 = f.rows;
    const rows: c_int = @intCast(rows_u32);
    const total_u32 = slots_u32 * rows_u32;
    const nh: c_int = @intCast(k.num_attention_heads);
    const nkv: c_int = @intCast(k.num_key_value_heads);
    const hd: c_int = @intCast(k.head_dim);
    const hidden: c_int = @intCast(k.hidden_size);
    const qdim = nh * hd;
    const kdim = nkv * hd;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(k.head_dim)));
    const no_base = c.mlx_optional_float{ .value = 0, .has_value = false };
    const mask_mode: [*:0]const u8 = if (rows > 1) "causal" else "";
    const nil = emptyArray();
    defer _ = c.mlx_array_free(nil);

    var g: Graph = .{ .s = m.stream };
    defer g.deinit();

    // Slot-major [slots * rows]: the engine hands tokens rows per slot, slot after slot.
    const ids = if (f.ids.ctx != null)
        try g.reshape(try g.op(c.mlx_slice, .{
            c.mlx_array{ .ctx = f.ids.ctx },
            &[_]c_int{ 0, 0 },
            2,
            &[_]c_int{ slots, rows },
            2,
            &[_]c_int{ 1, 1 },
            2,
        }), &.{@intCast(total_u32)})
    else blk: {
        const toks = f.tokens.slice();
        const a = c.mlx_array_new_data(toks.ptr, &[_]c_int{@intCast(total_u32)}, 1, @intCast(c.MLX_UINT32));
        try g.track(a);
        break :blk a;
    };

    var x = try embedRows(m, &g, ids, @intCast(total_u32));
    // A wide forward's dequant tiles (total rows > 64) are freed per layer; the batch GEMV path has none.
    const per_layer = total_u32 > 64;
    if (per_layer and rows > 1) x = try g.evalKeep(x);

    for (m.layers, 0..) |l, li| {
        const h = try g.op(c.mlx_fast_rms_norm, .{ x, l.ln1, k.rms_norm_eps });
        const q = try stqLinearRows(m, &g, &l.q, h, total_u32);
        const kk = try stqLinearRows(m, &g, &l.k, h, total_u32);
        const vv = try stqLinearRows(m, &g, &l.v, h, total_u32);

        var parts: std.ArrayList(c.mlx_array) = .empty;
        defer parts.deinit(gpa);
        for (0..slots_u32) |i| {
            const first: c_int = @intCast(i * rows_u32);
            const pos: c_int = @intCast(f.positions.slice()[i]);
            const qi = try g.reshape(try slotRows(&g, q, first, rows, qdim), &.{ 1, rows, nh, hd });
            const ki = try g.reshape(try slotRows(&g, kk, first, rows, kdim), &.{ 1, rows, nkv, hd });
            const vi = try g.reshape(try slotRows(&g, vv, first, rows, kdim), &.{ 1, rows, nkv, hd });
            var qh = try g.op(c.mlx_fast_rope, .{ try g.swap12(qi), hd, false, no_base, 1.0, pos, m.freqs });
            var kh = try g.op(c.mlx_fast_rope, .{ try g.swap12(ki), hd, false, no_base, 1.0, pos, m.freqs });
            const vh = try g.swap12(vi);
            qh = try g.op(c.mlx_fast_rms_norm, .{ qh, l.qn, k.rms_norm_eps });
            kh = try g.op(c.mlx_fast_rms_norm, .{ kh, l.kn, k.rms_norm_eps });

            var view: abi.KvView = .{};
            const app = ops.append(f.caches.slice()[i], @intCast(li), .{ .ctx = kh.ctx }, .{ .ctx = vh.ctx }, 0, .{ .ctx = g.s.ctx }, &view);
            if (!app.isOk()) return opFailed();
            defer ops.release(&view);
            const attn = try g.op(c.mlx_fast_scaled_dot_product_attention, .{
                qh,
                c.mlx_array{ .ctx = view.k.ctx },
                c.mlx_array{ .ctx = view.v.ctx },
                scale,
                mask_mode,
                nil,
                nil,
                false,
            });
            try parts.append(gpa, try g.reshape(try g.swap12(attn), &.{ rows, qdim }));
        }
        const vec = c.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
        defer _ = c.mlx_vector_array_free(vec);
        const o = try stqLinearRows(m, &g, &l.o, try g.op(c.mlx_concatenate_axis, .{ vec, 0 }), total_u32);

        const summed = try g.op(c.mlx_add, .{ x, o });
        const h2 = try g.op(c.mlx_fast_rms_norm, .{ summed, l.ln2, k.rms_norm_eps });
        const gate = try stqLinearRows(m, &g, &l.gate, h2, total_u32);
        const act = try g.op(c.mlx_multiply, .{ gate, try g.op(c.mlx_sigmoid, .{gate}) });
        const up = try stqLinearRows(m, &g, &l.up, h2, total_u32);
        const mlp = try stqLinearRows(m, &g, &l.down, try g.op(c.mlx_multiply, .{ act, up }), total_u32);
        x = try g.op(c.mlx_add, .{ summed, mlp });
        if (per_layer) x = try g.evalKeep(x);
    }

    const normed = try g.op(c.mlx_fast_rms_norm, .{ x, m.norm, k.rms_norm_eps });
    // `.last_row`: one hidden row per slot, in slot order. `.all_rows`: every row, slot after slot.
    var picked: c.mlx_array = normed;
    if (f.logits == .last_row) {
        var lasts: std.ArrayList(c.mlx_array) = .empty;
        defer lasts.deinit(gpa);
        for (0..slots_u32) |i| {
            const at: c_int = @intCast(i * rows_u32 + rows_u32 - 1);
            try lasts.append(gpa, try row(&g, normed, at, hidden));
        }
        const lvec = c.mlx_vector_array_new_data(lasts.items.ptr, lasts.items.len);
        defer _ = c.mlx_vector_array_free(lvec);
        picked = try g.op(c.mlx_concatenate_axis, .{ lvec, 0 });
    }
    if (f.logits != .none) out.logits = exportArray(try logits(m, &g, picked));
    if (f.want_hidden) out.hidden = exportArray(picked);
}

fn run(m: *Model, f: *const abi.Forward, ops: *const abi.CacheOps, out: *abi.ForwardOut) !void {
    // <= 64 rows per slot: one weight pass for every row and slot (decode, verify, short prompts).
    if (f.rows <= 64) return runTrunk(m, f, ops, out);
    var hiddens: std.ArrayList(c.mlx_array) = .empty;
    defer {
        for (hiddens.items) |x| _ = c.mlx_array_free(x);
        hiddens.deinit(gpa);
    }
    // One graph per slot: a multi-slot prefill evaluates layer by layer (to free its dequant tiles), and that must
    // never free a slot the previous slot already produced.
    for (f.caches.slice(), f.positions.slice(), 0..) |cache, pos, i| {
        var g: Graph = .{ .s = m.stream };
        defer g.deinit();
        const slot: c_int = @intCast(i);
        const rows: c_int = @intCast(f.rows);
        const ids = if (f.ids.ctx != null)
            try g.reshape(try g.op(c.mlx_slice, .{
                c.mlx_array{ .ctx = f.ids.ctx },
                &[_]c_int{ slot, 0 },
                2,
                &[_]c_int{ slot + 1, rows },
                2,
                &[_]c_int{ 1, 1 },
                2,
            }), &.{rows})
        else blk: {
            const toks = f.tokens.slice()[i * f.rows ..][0..f.rows];
            const a = c.mlx_array_new_data(toks.ptr, &[_]c_int{rows}, 1, @intCast(c.MLX_UINT32));
            try g.track(a);
            break :blk a;
        };
        const x = try decoder(m, &g, ops, cache, ids, @intCast(pos), f.rows);
        const h = if (f.logits == .last_row) try g.op(c.mlx_slice, .{
            x,
            &[_]c_int{ rows - 1, 0 },
            2,
            &[_]c_int{ rows, @intCast(m.cfg.hidden_size) },
            2,
            &[_]c_int{ 1, 1 },
            2,
        }) else x;
        try hiddens.append(gpa, newRef(h));
    }

    var g: Graph = .{ .s = m.stream };
    defer g.deinit();
    const vec = c.mlx_vector_array_new_data(hiddens.items.ptr, hiddens.items.len);
    defer _ = c.mlx_vector_array_free(vec);
    const h = try g.op(c.mlx_concatenate_axis, .{ vec, 0 });
    if (f.logits != .none) out.logits = exportArray(try logits(m, &g, h));
    if (f.want_hidden) out.hidden = exportArray(h);
}

/// The tied head, read through the 6-bit affine embedding (no dense 495 MB copy).
fn logits(m: *Model, g: *Graph, h: c.mlx_array) !c.mlx_array {
    return g.op(c.mlx_quantized_matmul, .{
        h,
        m.embed_w,
        m.embed_scales,
        m.embed_biases,
        true,
        someInt(m.cfg.embed_group),
        someInt(m.cfg.embed_bits),
        "affine",
    });
}

/// ids [rows] u32 → [1, rows, hidden] f16, dequantizing only the rows looked up.
fn embed(m: *Model, g: *Graph, ids: c.mlx_array, rows: c_int) !c.mlx_array {
    return g.reshape(try embedRows(m, g, ids, rows), &.{ 1, rows, @intCast(m.cfg.hidden_size) });
}

fn embedRows(m: *Model, g: *Graph, ids: c.mlx_array, rows: c_int) !c.mlx_array {
    _ = rows;
    const w = try g.op(c.mlx_take_axis, .{ m.embed_w, ids, 0 });
    const s = try g.op(c.mlx_take_axis, .{ m.embed_scales, ids, 0 });
    const b = try g.op(c.mlx_take_axis, .{ m.embed_biases, ids, 0 });
    const nil = emptyArray();
    defer _ = c.mlx_array_free(nil);
    return g.op(c.mlx_dequantize, .{
        w,
        s,
        b,
        someInt(m.cfg.embed_group),
        someInt(m.cfg.embed_bits),
        "affine",
        nil,
        c.mlx_optional_dtype{ .value = @intCast(c.MLX_FLOAT16), .has_value = true },
    });
}

fn decoder(m: *Model, g: *Graph, ops: *const abi.CacheOps, cache: abi.Handle, ids: c.mlx_array, pos: c_int, rows_u32: u32) !c.mlx_array {
    const k = m.cfg;
    const t: c_int = @intCast(rows_u32);
    const hidden: c_int = @intCast(k.hidden_size);
    var x = try embed(m, g, ids, t);
    // A prompt's dequant tiles are freed per layer; decode keeps the whole (tiny) graph to one eval at the end.
    if (t > 1) x = try g.evalKeep(x);

    const nh: c_int = @intCast(k.num_attention_heads);
    const nkv: c_int = @intCast(k.num_key_value_heads);
    const hd: c_int = @intCast(k.head_dim);
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(k.head_dim)));
    const no_base = c.mlx_optional_float{ .value = 0, .has_value = false };

    for (m.layers, 0..) |l, li| {
        const h = try g.op(c.mlx_fast_rms_norm, .{ x, l.ln1, k.rms_norm_eps });

        const q3 = try g.reshape(try stqLinear(m, g, &l.q, h, t), &.{ 1, t, nh, hd });
        const k3 = try g.reshape(try stqLinear(m, g, &l.k, h, t), &.{ 1, t, nkv, hd });
        const v3 = try g.reshape(try stqLinear(m, g, &l.v, h, t), &.{ 1, t, nkv, hd });

        var q = try g.op(c.mlx_fast_rope, .{ try g.swap12(q3), hd, false, no_base, 1.0, pos, m.freqs });
        var kk = try g.op(c.mlx_fast_rope, .{ try g.swap12(k3), hd, false, no_base, 1.0, pos, m.freqs });
        const v = try g.swap12(v3);

        q = try g.op(c.mlx_fast_rms_norm, .{ q, l.qn, k.rms_norm_eps });
        kk = try g.op(c.mlx_fast_rms_norm, .{ kk, l.kn, k.rms_norm_eps });

        var view: abi.KvView = .{};
        const app = ops.append(cache, @intCast(li), .{ .ctx = kk.ctx }, .{ .ctx = v.ctx }, 0, .{ .ctx = g.s.ctx }, &view);
        if (!app.isOk()) return opFailed();
        defer ops.release(&view);

        const nil = emptyArray();
        defer _ = c.mlx_array_free(nil);
        const mask_mode: [*:0]const u8 = if (t > 1) "causal" else "";
        const attn = try g.op(c.mlx_fast_scaled_dot_product_attention, .{
            q,
            c.mlx_array{ .ctx = view.k.ctx },
            c.mlx_array{ .ctx = view.v.ctx },
            scale,
            mask_mode,
            nil,
            nil,
            false,
        });
        const o = try stqLinear(m, g, &l.o, try g.reshape(try g.swap12(attn), &.{ 1, t, nh * hd }), t);

        const summed = try g.op(c.mlx_add, .{ x, o });
        const h2 = try g.op(c.mlx_fast_rms_norm, .{ summed, l.ln2, k.rms_norm_eps });
        const gate = try stqLinear(m, g, &l.gate, h2, t);
        const act = try g.op(c.mlx_multiply, .{ gate, try g.op(c.mlx_sigmoid, .{gate}) });
        const up = try stqLinear(m, g, &l.up, h2, t);
        const mlp = try stqLinear(m, g, &l.down, try g.op(c.mlx_multiply, .{ act, up }), t);
        x = try g.op(c.mlx_add, .{ summed, mlp });

        // Drop the layer's dequant tiles now; only the residual survives into the next layer.
        if (t > 1) x = try g.evalKeep(x);
    }

    const normed = try g.op(c.mlx_fast_rms_norm, .{ x, m.norm, k.rms_norm_eps });
    return g.reshape(normed, &.{ t, hidden });
}

/// One row `i` of a 2-D array, as [1, dim].
fn row(g: *Graph, x: c.mlx_array, i: c_int, dim: c_int) !c.mlx_array {
    return g.op(c.mlx_slice, .{
        x,
        &[_]c_int{ i, 0 },
        2,
        &[_]c_int{ i + 1, dim },
        2,
        &[_]c_int{ 1, 1 },
        2,
    });
}

/// Rows `[first, first + n)` of a 2-D array, as [n, dim].
fn slotRows(g: *Graph, x: c.mlx_array, first: c_int, n: c_int, dim: c_int) !c.mlx_array {
    return g.op(c.mlx_slice, .{
        x,
        &[_]c_int{ first, 0 },
        2,
        &[_]c_int{ first + n, dim },
        2,
        &[_]c_int{ 1, 1 },
        2,
    });
}

/// `x [rows, K]` through an STQ1_0 projection, rows = sequences decode in lockstep: the batch GEMV reads each
/// packed weight word once for all of them; above 8 rows it runs in 8-row chunks (a 1-row tail takes the GEMV).
fn stqLinearRows(m: *Model, g: *Graph, w: *const Stq, x: c.mlx_array, rows: u32) !c.mlx_array {
    const r: c_int = @intCast(rows);
    if (r == 1) return stqGemv(m, g, w, x);
    // Past 64 rows the dense tile + GEMM wins (the pack's measured crossover); the tile is transient.
    if (r > 64) return stqMatmul(m, g, w, x);
    if (r <= 8) return stqBatchOne(m, g, w, x, r);
    var parts: std.ArrayList(c.mlx_array) = .empty;
    defer parts.deinit(gpa);
    const k: c_int = @intCast(w.k);
    var i: c_int = 0;
    while (i < r) {
        const n = @min(8, r - i);
        const xi = try g.op(c.mlx_slice, .{
            x,
            &[_]c_int{ i, 0 },
            2,
            &[_]c_int{ i + n, k },
            2,
            &[_]c_int{ 1, 1 },
            2,
        });
        const yi = if (n == 1)
            try stqGemv(m, g, w, try g.reshape(xi, &.{ 1, k }))
        else
            try stqBatchOne(m, g, w, xi, n);
        try parts.append(gpa, yi);
        i += n;
    }
    const vec = c.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = c.mlx_vector_array_free(vec);
    return g.op(c.mlx_concatenate_axis, .{ vec, 0 });
}

fn stqBatchOne(m: *Model, g: *Graph, w: *const Stq, x: c.mlx_array, bat: c_int) !c.mlx_array {
    const n: c_int = @intCast(w.n);
    const kk: c_int = @intCast(w.k);
    const nb: c_int = @intCast(w.k / qk_k);
    // Accumulators are BAT * R_ROWS floats: fewer rows per simdgroup as the batch widens (the pack's own knob).
    const r: c_int = if (bat <= 4) @divTrunc(8, bat) else 2;
    const cfg = c.mlx_fast_metal_kernel_config_new();
    defer _ = c.mlx_fast_metal_kernel_config_free(cfg);
    try expectOk(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ bat, n }, 2, @intCast(c.MLX_FLOAT16)));
    try expectOk(c.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(n, r), 1));
    try expectOk(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", @intCast(c.MLX_FLOAT16)));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", kk));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NB", nb));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BAT", bat));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R_ROWS", r));

    var outs = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(outs);
    const ins = c.mlx_vector_array_new_data(&[_]c.mlx_array{ x, w.qs, w.sign, w.scales, m.stq_lut }, 5);
    defer _ = c.mlx_vector_array_free(ins);
    try expectOk(c.mlx_fast_metal_kernel_apply(&outs, m.batch_gemv, ins, cfg, g.s));
    var res = c.mlx_array_new();
    errdefer _ = c.mlx_array_free(res);
    try expectOk(c.mlx_vector_array_get(&res, outs, 0));
    try g.track(res);
    return res;
}

/// `x [1, t, k]` through an STQ1_0 projection: the GEMV for one row, a transient dense tile + GEMM above.
fn stqLinear(m: *Model, g: *Graph, w: *const Stq, x: c.mlx_array, t: c_int) !c.mlx_array {
    const f16_dtype: c.mlx_dtype = @intCast(c.MLX_FLOAT16);
    const x16 = if (c.mlx_array_dtype(x) == c.MLX_FLOAT16) x else try g.op(c.mlx_astype, .{ x, f16_dtype });
    const x2 = try g.reshape(x16, &.{ t, @intCast(w.k) });
    const y = if (t == 1) try stqGemv(m, g, w, x2) else try stqMatmul(m, g, w, x2);
    return g.reshape(y, &.{ 1, t, @intCast(w.n) });
}

fn stqGemv(m: *Model, g: *Graph, w: *const Stq, x: c.mlx_array) !c.mlx_array {
    const cfg = c.mlx_fast_metal_kernel_config_new();
    defer _ = c.mlx_fast_metal_kernel_config_free(cfg);
    const tokens: c_int = 1;
    const n: c_int = @intCast(w.n);
    const kk: c_int = @intCast(w.k);
    try expectOk(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ tokens, n }, 2, @intCast(c.MLX_FLOAT16)));
    try expectOk(c.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(n, r_rows), tokens));
    try expectOk(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", @intCast(c.MLX_FLOAT16)));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", kk));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", n));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NB", @intCast(w.k / qk_k)));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NCH", @intCast(w.k / 2048)));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R_ROWS", r_rows));

    var outs = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(outs);
    const ins = c.mlx_vector_array_new_data(&[_]c.mlx_array{ x, w.qs, w.sign, w.scales }, 4);
    defer _ = c.mlx_vector_array_free(ins);
    try expectOk(c.mlx_fast_metal_kernel_apply(&outs, m.gemv, ins, cfg, g.s));
    var r = c.mlx_array_new();
    errdefer _ = c.mlx_array_free(r);
    try expectOk(c.mlx_vector_array_get(&r, outs, 0));
    try g.track(r);
    return r;
}

fn stqMatmul(m: *Model, g: *Graph, w: *const Stq, x: c.mlx_array) !c.mlx_array {
    const n: c_int = @intCast(w.n);
    const kk: c_int = @intCast(w.k);
    const nb: c_int = @intCast(w.k / qk_k);
    const cfg = c.mlx_fast_metal_kernel_config_new();
    defer _ = c.mlx_fast_metal_kernel_config_free(cfg);
    try expectOk(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ n, kk }, 2, @intCast(c.MLX_FLOAT16)));
    try expectOk(c.mlx_fast_metal_kernel_config_set_grid(cfg, 8, n, nb));
    try expectOk(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, 8, 1, 1));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", kk));
    try expectOk(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NB", nb));

    var outs = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(outs);
    const ins = c.mlx_vector_array_new_data(&[_]c.mlx_array{ w.qs, w.sign, w.scales }, 3);
    defer _ = c.mlx_vector_array_free(ins);
    try expectOk(c.mlx_fast_metal_kernel_apply(&outs, m.dequant, ins, cfg, g.s));
    var dense = c.mlx_array_new();
    errdefer _ = c.mlx_array_free(dense);
    try expectOk(c.mlx_vector_array_get(&dense, outs, 0));
    try g.track(dense);
    return g.linear(x, dense);
}

/// The conformance pack (`src/conformance_fixture.zig`): a 2-layer Hy-MT2 shape with real STQ1_0 projections and a
/// real 6-bit embedding, so `mlx-serve plugin check` loads it and runs a forward through both Metal kernels.
pub fn conformanceFixture(a: std.mem.Allocator) !kit.sdk.fixture.Fixture {
    const f = try @import("src/conformance_fixture.zig").build(a);
    return .{ .files = f.files };
}

test {
    _ = config;
    _ = @import("src/conformance_fixture.zig");
}

test "load: a tensor whose shape or dtype is not what the kernel reads is refused" {
    const u8_dt: c.mlx_dtype = @intCast(c.MLX_UINT8);
    const f16_dt: c.mlx_dtype = @intCast(c.MLX_FLOAT16);
    const want = [_]c_int{ 512, 64 }; // k_proj.qs at hidden 2048
    const exact = [_]c_int{ 512, 64 };
    const short = [_]c_int{ 512, 32 };
    const wide = [_]c_int{ 512, 64 };
    try std.testing.expect(shapeMatches(u8_dt, 2, &exact, u8_dt, &want));
    // A qs buffer half the declared width: the GEMV would read past it.
    try std.testing.expect(!shapeMatches(u8_dt, 2, &short, u8_dt, &want));
    // f16 scales passed where u8 qs is read.
    try std.testing.expect(!shapeMatches(f16_dt, 2, &wide, u8_dt, &want));
    // A 3-D tensor where a 2-D one is read.
    try std.testing.expect(!shapeMatches(u8_dt, 3, &wide, u8_dt, &want));
}
