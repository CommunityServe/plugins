//! copied_from: src/sdk/quant/params.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! sdk/quant/params.
//! Owns BITS_CACHE_CAP, QuantParams, QuantParamsCache, BitsCache, quantParamsOrDense, affineParamsFromGeometry, and 4 more.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const testing = std.testing;

// ── Quantization params cache ──
// A tiny lock-free pointer → (bits, group_size) cache. Quantized weights are loaded
// once at init and reused for every forward pass; we detect (or pre-bind) params on
// first touch and serve hits thereafter. Uses open addressing with linear probing on
// a fixed-size array — this fits in L1 and keeps the cost of a lookup to ~5ns,
// matching the perf commit's intent of "eliminate per-call detect overhead" while
// supporting mixed-precision quantization (Gemma-4 MoE per-layer bits, etc.).
pub const BITS_CACHE_CAP: usize = 1024; // plenty for 60 layers × ~10 quant weights × factor
pub const QuantParams = struct {
    bits: u32,
    group_size: u32,
    mode: QuantMode = .affine,

    /// The model-wide scheme a config declares (`quant_bits`, `quant_group_size`, `quant_mode`).
    pub fn of(cfg: anytype) QuantParams {
        return .{ .bits = cfg.quant_bits, .group_size = cfg.quant_group_size, .mode = cfg.quant_mode };
    }
};

const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = @splat(null),
    vals_bits: [BITS_CACHE_CAP]u8 = @splat(0),
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = @splat(0),
    vals_mode: [BITS_CACHE_CAP]u8 = @splat(0),

    pub inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert params for `key`. Returns false if the probe window saturated
    /// (caller should treat as "fall through to detection" — but realistically
    /// never fires given BITS_CACHE_CAP is 1024 vs ~2k weights max).
    pub fn put(self: *QuantParamsCache, key: *anyopaque, qp: QuantParams) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(qp.bits);
                self.vals_gs_div8[idx] = @intCast(qp.group_size / 8);
                self.vals_mode[idx] = @backingInt(qp.mode);
                return true;
            }
        }
        return false;
    }
};

// Backwards-compatibility alias — keeps existing field names readable.
pub const BitsCache = QuantParamsCache;

pub fn affineParamsFromGeometry(w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: u32) ?QuantParams {
    if (sc.ctx == null or in_dim == 0) return null;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (w_cols == 0 or s_cols == 0) return null;
    if ((w_cols * 32) % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = (w_cols * 32) / in_dim;
    const gs = in_dim / s_cols;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    switch (gs) {
        32, 64, 128 => {},
        else => return null,
    }
    return .{ .bits = bits, .group_size = gs, .mode = .affine };
}

/// The fp mode a solved (bits, group_size) pair belongs to. Each format
/// DEFINES its block size — MXFP8/MXFP4 are 32, NVFP4 is 16 — so the pair
/// names exactly one of them. Anything else is a layout we don't serve;
/// returning null keeps a checkpoint's own declaration rather than inventing a
/// mode for it.
fn fpQuantModeFor(bits: u32, group_size: u32) ?QuantMode {
    if (bits == 8 and group_size == 32) return .mxfp8;
    if (bits == 4 and group_size == 32) return .mxfp4;
    if (bits == 4 and group_size == 16) return .nvfp4;
    return null;
}

/// Per-weight (bits, group_size, mode) for an fp-quantized tensor, solved from
/// the packed geometry against the activation's inner dim: a row holds
/// `w_cols * 32` packed bits over `in_dim` values, and `s_cols` scales cover
/// them in groups.
///
/// Returns null when the geometry AGREES with the model-wide declaration —
/// the caller then returns the config verbatim, so a consistent checkpoint
/// resolves exactly as it did before this existed.
fn fpParamsFromGeometry(model: QuantParams, w_cols: u32, s_cols: u32, in_dim: u32) ?QuantParams {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = w_cols * 32;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const group_size = in_dim / s_cols;
    if (bits == model.bits and group_size == model.group_size) return null;
    const mode = fpQuantModeFor(bits, group_size) orelse return null;
    return .{ .bits = bits, .group_size = group_size, .mode = mode };
}

pub fn computeQuantParams(model: QuantParams, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;

    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (model.mode != .affine) {
            // A per-tensor override can live INSIDE the fp family: LFM2.5-2.6B-MLX's
            // mxfp4 build stores `model.embed_tokens` as mxfp8. Every fp mode
            // carries uint8 (E8M0/E4M3) scales, so the scales-dtype rule above
            // cannot separate them and the embedding resolved at the model-wide
            // 4 bits — an `mlx_dequantize` shape rejection, i.e. an MLX error,
            // i.e. a dead server on the first request. The packed geometry says
            // it exactly; solve, exactly as the affine arm below does.
            if (in_dim) |in| {
                if (fpParamsFromGeometry(model, w_cols, s_cols, in)) |qp| return qp;
            } else if (s_cols > 0 and (w_cols * 32) % s_cols == 0) {
                const per_group = (w_cols * 32) / s_cols;
                const bits: u32 = if (per_group == 256) 8 else 4;
                const gs = per_group / bits;
                if (fpQuantModeFor(bits, gs)) |mode| return .{ .bits = bits, .group_size = gs, .mode = mode };
            }
            return model;
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }

    if (model.mode == .affine) {
        // Exact per-weight solve when the caller knows the input dim — a
        // hinted tensor whose geometry contradicts the config's group size
        // (MTP sidecars mix gs 128/64/32 over a gs-64 trunk) resolves to its
        // TRUE params. Consistent tensors solve to the config values anyway.
        if (in_dim) |in| {
            if (affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        }
        return .{
            .bits = detectQuantBits(w, sc, model.group_size),
            .group_size = model.group_size,
            .mode = .affine,
        };
    }

    if (in_dim) |in| {
        if (in > 0 and s_cols > 0 and w_cols > 0) {
            return .{ .bits = (w_cols * 32) / in, .group_size = in / s_cols, .mode = .affine };
        }
    }
    if (w_cols > 0 and s_cols > 0) {
        const ratio = (w_cols * 32) / s_cols; // = bits × group_size
        return .{ .bits = @max(ratio / 64, 1), .group_size = 64, .mode = .affine };
    }
    return .{ .bits = 8, .group_size = 64, .mode = .affine };
}

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,
    /// Raw ggml blocks (src/sdk/quant/gguf). Never reaches an MLX quantized op:
    /// the per-tensor type rides on the weight, see `mlx_gguf.kernels.Info`.
    gguf,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
            .gguf => "gguf",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }

    /// The e2m1 nibble modes (byte scales, no biases): MXFP4 g32, NVFP4 g16.
    pub fn isFp4(self: QuantMode) bool {
        return self == .mxfp4 or self == .nvfp4;
    }
};

/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
pub fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

// ── tests ──
