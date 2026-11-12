// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE1 (src/fused_router.zig): the one-row launch's arguments to MLX's
// matrix-vector kernel (GEMVKernel::run, src/vendor/mlx/gemv.h as the plugin edits it) that MLX's gemv kernel binds as
// constant buffers: GLM-5.3-Flash's router, K 4096 inputs (the row's length and the matrix's row stride) and N 288
// outputs; alpha, beta and the bias stride unread (no axpby).
constant int ro1_in = 4096;
constant int ro1_out = 288;
constant int ro1_ld = 4096;
constant float ro1_alpha = 1.0f;
constant float ro1_beta = 0.0f;
constant int ro1_bias_stride = 0;
