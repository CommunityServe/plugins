//! copied_from: src/engines/mlx/methods/yarn.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The legacy `Transformer`'s methods over `sdk/blocks/rope/yarn.zig` (what reads the engine's model state).

const mlx = @import("../../../mlx/ffi.zig");
const scalarOf = @import("../../../mlx/array.zig").scalarOf;
const Transformer = @import("../transformer.zig").Transformer;
const constTableAs = @import("../../../mlx/dtype.zig").constTableAs;
const mrope = @import("../../../sdk/vision/mrope.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;

/// This checkpoint's YaRN spectrum: the config's rope geometry (theta, the
/// rotating slice, the pre-trained window) resolved into `mrope.Yarn`, the one
/// implementation of HF's `_compute_yarn_parameters` in the engine. Only
/// meaningful when `config.rope_yarn`.
pub fn yarnSpec(config: *const ModelConfig) mrope.Yarn {
    return .{
        .theta = config.rope_theta,
        .factor = config.yarn_factor,
        .orig_max = config.yarn_orig_max_pos,
        .rotary_dim = config.yarnRotaryDim(),
        .beta_fast = config.yarn_beta_fast,
        .beta_slow = config.yarn_beta_slow,
        .truncate = config.yarn_truncate,
    };
}

/// The mscale the host cos/sin tables are multiplied by: HF's `attention_factor`
/// when the checkpoint scales its rope, 1.0 (a no-op) when it does not.
pub fn yarnMscale(config: *const ModelConfig) f64 {
    return if (config.rope_yarn) config.yarn_attention_factor else 1.0;
}
