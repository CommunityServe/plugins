// turboexl3_glm53f TURBOEXL3_GLM53F_MOE_COMBINE1 (src/moe_combine1.zig): a decode or verify call's routed join and
// combine as one launch. The stock chain it replaces (glm5n/experts.zig `combineStock` after the join's take): the take
// of each routed position's row from the call's concatenated outputs, astype f32, the multiply by the f32 weight (MLX
// Multiply, its own kernel: one f32 rounding, never contracted), the axis-1 sum of [n, 8, H] = col_reduce_small with
// threadgroup y = 8 (reduce.cpp strided_reduce_general_dispatch: 8 rows < 32), so each product is first added to the 0
// init, then lid.y 0 folds k = 1..7 in order as op(product_k, total) (reduce_col.h), then astype bf16 (one rounding).
// The prompt pass's combine text (mimo/kernels/mimo_prefill_combine.metal) states the same arithmetic; this one reads
// one source through the join's order. No contraction into FMA.
#pragma METAL fp contract(off)
