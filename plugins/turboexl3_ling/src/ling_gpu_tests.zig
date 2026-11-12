//! turboexl3_ling's GPU byte tests, one test binary (`zig build gpu-test -Dmlx-lib=<staged MLX>`, build.zig; each test
//! skips unless MLX_SERVE_LING_GPU_TESTS=1): the KDA paths (ling_kda_test.zig) and the exact verify's dense rows
//! (ling_rows_test.zig), the exact verify's MLA rows (ling_verify_mla_test.zig), the MLA prompt spans
//! (ling_mla_spans_test.zig), the latent MLA decode at the served state (ling_mla_latent_test.zig), the prefill SwiGLU epilogue against the host's own chain
//! (ling_swiglu_epi_test.zig), a layer-major layer's spans over its shared expert's W_hats decoded once against the
//! per-call path (ling_shared_layer_test.zig), and over its other dense prefill linears' W_hats decoded once
//! (ling_layer_what_test.zig), and the rope-once cache's views, decode and verify rows and prompt rows against the
//! expanded cache's calls (ling_mla_rope1_test.zig), the served turboexl3 Config that `configure` writes
//! (ling_configure_test.zig), the run-path marks of the paths whose launch is their only evidence
//! (ling_ran_marks_test.zig), EMBED_ROWS' embedding rows read from the pack against the resident table's take
//! (ling_embed_test.zig), KDA_GATE_EPI's gate in the prompt projection's launch against the served path
//! (ling_kda_gate_epi_test.zig), and KDA_NORM_PREP's gated norm written into o_proj's prepared rows against the served
//! path (ling_kda_norm_prep_test.zig).
test {
    _ = @import("ling_kda_test.zig");
    _ = @import("ling_rows_test.zig");
    _ = @import("ling_verify_mla_test.zig");
    _ = @import("ling_mla_spans_test.zig");
    _ = @import("ling_mla_latent_test.zig");
    _ = @import("ling_swiglu_epi_test.zig");
    _ = @import("ling_shared_layer_test.zig");
    _ = @import("ling_layer_what_test.zig");
    _ = @import("ling_mla_rope1_test.zig");
    _ = @import("ling_configure_test.zig");
    _ = @import("ling_ran_marks_test.zig");
    _ = @import("ling_embed_test.zig");
    _ = @import("ling_kda_gate_epi_test.zig");
    _ = @import("ling_kda_norm_prep_test.zig");
}

// What turboexl3's `sdk` (src/turboexl3_sdk.zig) reads through `mlx_host` in this binary (build.zig): the names
// plugin.zig gives it, from the same files, so no copied file is in two modules.
pub const mlx = @import("mlx_serve").mlx;
pub const shared = @import("mlx_serve/sdk/util/shared.zig");
pub const quant_abi = @import("mlx_serve").abi.quant;
