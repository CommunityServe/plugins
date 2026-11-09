//! MLX's quantization modes, as checkpoints and quants name them.

const std = @import("std");

pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,
    /// Raw ggml blocks (lib/mlx-serve-gguf). Never reaches an MLX quantized op:
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
};
