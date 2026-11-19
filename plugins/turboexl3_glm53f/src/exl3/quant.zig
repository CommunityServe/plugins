//! The streamed EXL3 bank's quant facts: the bank format the GLM bank reads (`format`, the bank manifest's
//! `quantization` object), the activation the routed SwiGLU clamps at, and a projection's per-slot arrays.

const quant = @import("../streamed/sdk_ext.zig").quant;

/// One projection's per-slot arrays: the streamer's `ProjArrays` {code, rout, rin}.
pub fn Arrays(comptime T: type) type {
    return ProjArrays(T);
}

/// The prompt pass's widest wave, in experts.
pub const wave_max = 16;

/// The clamped SwiGLU at 10.0.
pub const fused_act: quant.Activation = .{ .swiglu_clamped = 10.0 };

/// The bank format, field by field, as the bank manifest's `quantization` object writes it (exllamav3 v1.4.2:
/// codebook.cuh, exl3_dq.cuh, pack.cu, hadamard.cu).
pub const format = struct {
    pub const mode = "exl3";
    pub const codebook = "mul1";
    pub const multiplier: u64 = 0x83DCD12D;
    pub const tile_size = 16;
    pub const tile_layout = "exl3-tensor-core";
    pub const tile_bitstream = "msb-first-swap16";
    pub const hadamard_size = 128;
    pub const hadamard_order = "sylvester-natural";
    pub const hadamard_scale = "1/sqrt(128)";
    pub const scale_rin = "exl3.suh";
    pub const scale_rout = "exl3.svh";
};

pub fn ProjArrays(comptime T: type) type {
    return struct { code: T, rout: T, rin: T };
}
