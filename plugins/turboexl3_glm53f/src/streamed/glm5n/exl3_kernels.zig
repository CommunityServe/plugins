//! GLM-5.3-Flash's EXL3 K=3 expert geometry, as the bank's records and the load lines name it: hidden 4096, inter 2048,
//! 16 x 16 tiles of K = 3 (48 halfwords a tile), the mul1 codebook and the clamped SwiGLU. turboexl3 runs the math
//! (src/routed_t3.zig).

const xq = @import("../../exl3/quant.zig");

pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
/// Bits per weight of the bank's trellis (every layer): 16 x K halfwords a 16 x 16 tile.
pub const k_bits: u32 = 3;
pub const tile_words: u32 = 16 * k_bits;
/// The codebook the texts decode (exl3 mul1, the multiplier in the texts).
pub const codebook = xq.format.codebook;
pub const codebook_multiplier: u64 = xq.format.multiplier;
/// The swiglu clamp the texts fuse (gu_epi, onepass): GLM-5.3-Flash's `swiglu_limit` is DeepSeek's.
pub const swiglu_limit: f64 = xq.fused_act.swiglu_clamped;

comptime {
    if (tile_words != 48) @compileError("glm5_next exl3: the bank's tiles are K = 3 (48 halfwords)");
}

/// The dims of a projection's slot arrays (without the leading rows): code, rout, rin.
pub const ProjDims = struct { code: [3]u32, rout: u32, rin: u32 };
pub const Which = enum { gate_up, down };
pub fn projDims(which: Which) ProjDims {
    const in: u32, const out: u32 = if (which == .down) .{ inter, hidden } else .{ hidden, inter };
    return .{ .code = .{ in / 16, out / 16, tile_words }, .rout = out, .rin = in };
}
