//! copied_from: src/sdk/weights/streamed.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The streamed weight source: routed experts read from SSD into slot rows, for a trunk whose MoE reads them through
//! `Routed`. The streamer is mlx-stream's (`streamed/expert/`, `streamed/sdk_ext/expert/`, the read pool in
//! `streamed/csrc/`; module `mlx_stream`). Sources: the MXFP4 record bank (`experts.bin` + `expert-manifest.json`,
//! `streamed/bank_source.zig`; MiMo's and DeepSeek-V4.1's experts) and a standard MLX pack's affine experts as
//! downloaded (`openPack`, `streamed/mlx_pack.zig`). Without mlx-stream, or without a bank, opening refuses
//! by name.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const plugin = if (build_options.mlx_stream) @import("mlx_stream") else struct {};

const routed = @import("routed.zig");
pub const Proj = routed.Proj;
pub const Banks = routed.Banks;
pub const Fetched = routed.Fetched;
pub const max_call_ids = routed.max_call_ids;
pub const Routed = routed.Routed;

/// The per-model `stream_experts` setting: on (a quarter of each routed layer's experts in slot rows), or the slot
/// rows per layer.
pub const Setting = union(enum) {
    on,
    rows: u32,

};

/// An EXL3 expert bank (`streamed/exl3_source.zig`) for a trunk with no EXL3 math: the source returns routed sums.
pub const Exl3 = if (build_options.mlx_stream) plugin.exl3_source else struct {};

/// GLM-5.3-Flash's EXL3 bank (`streamed/glm5n_source.zig`): the source returns routed sums.
pub const Glm5n = if (build_options.mlx_stream) plugin.glm5n_source else struct {};

pub const Streamed = struct {
    pub const refusal = abi.Verdict.refuse(.unsupported, "streamed weights: no expert bank, or a build without mlx-stream");

    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert).
        rows: ?[]const u32 = null,
    };

    const Mxfp4 = if (build_options.mlx_stream) plugin.bank_source.Mxfp4 else void;
    const Pack = if (build_options.mlx_stream) plugin.mlx_pack.Source else void;

    src: union(enum) { mxfp4: *Mxfp4, pack: *Pack },

};

// ── tests ──
