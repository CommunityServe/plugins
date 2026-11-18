// turboexl3_glm53f TURBOEXL3_GLM53F_MOE_COMBINE1: per token t and 4 columns a thread (H / 4 threads a token), the 8
// routed rows read in place through the join's order (`order[t * TOPK + k]`: the row of the concatenated outputs the
// stock take reads for position (t, k)), each product f32(row_k) * f32(w_k) rounded to f32, added to the +0 init,
// folded k = 0..7 in router order as product_k + total, rounded to bf16 once: mimo_prefill_combine.metal's statements
// with its source pick replaced by the order's row. The weights in their stored type WT (bf16 or f32): their widening
// is exact, as the stock chain's astype is.
constexpr uint C = 4;
constexpr uint DC = uint(H) / C;
const uint gid = thread_position_in_grid.x;
const uint t = gid / DC;
const uint col = (gid - t * DC) * C;
const ulong lb = ulong(t) * ulong(TOPK);
float acc[C];
{
    const float wk = static_cast<float>(w[lb]);
    const device T* p = rows + ulong(order[lb]) * ulong(H) + col;
    for (uint i = 0; i < C; ++i) {
        acc[i] = static_cast<float>(p[i]) * wk + 0.0f;
    }
}
for (uint k = 1; k < uint(TOPK); ++k) {
    const float wk = static_cast<float>(w[lb + k]);
    const device T* p = rows + ulong(order[lb + k]) * ulong(H) + col;
    for (uint i = 0; i < C; ++i) {
        const float q = static_cast<float>(p[i]) * wk + 0.0f;
        acc[i] = q + acc[i];
    }
}
device bfloat16_t* o = y + ulong(t) * ulong(H) + col;
for (uint i = 0; i < C; ++i) {
    o[i] = static_cast<bfloat16_t>(acc[i]);
}
