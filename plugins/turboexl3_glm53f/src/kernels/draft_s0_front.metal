// TURBOEXL3_GLM53F_DRAFT_S0, the MTP draft's step 0 front (src/draft_s0.zig `front`): each of the step's rows' embedding
// E[ids[row]] under the enorm weight WE and the trunk's hidden H[row] under the hnorm weight WH, each through its own
// RMSNorm, written as their concatenation CAT [1, rows, 2D] (a row's embedding half first). Grid: 2 x rows threadgroups
// of TN threads (x: 0 the embedding row, 1 the hidden row; y: the row). Each threadgroup runs smallop_front.metal's
// statements at its row (the kit's add_norm.fmt.metal mirror of MLX's rms_norm single-row kernel: four consecutive
// elements a thread, the per-simdgroup sums folded through a 32-slot table, precise rsqrt, `w * T(x * inv)` in T), so a
// row's bytes are the one-row front's and the stock step's (take, both norms, the concatenation).
const uint t = thread_position_in_threadgroup.x;
const uint r = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
threadgroup float local_sums[32];
float hv[NC * 4];
float acc = 0.0f;
const size_t row_at = (r == 0u) ? size_t(ids[row]) * size_t(D) : size_t(row) * size_t(D);
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (c < D) {
        hv[p * 4 + i] = (r == 0u) ? float(E[row_at + c]) : float(H[row_at + c]);
    }
    acc += hv[p * 4 + i] * hv[p * 4 + i];
}
acc = simd_sum(acc);
if (sg == 0) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) local_sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = simd_sum(local_sums[lane]);
const float inv = metal::precise::rsqrt(total / float(D) + eps);
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    if (c < D) CAT[(size_t(row) * 2u + size_t(r)) * size_t(D) + c] = T(float((r == 0u) ? WE[c] : WH[c]) * float(T(hv[p * 4 + i] * inv)));
}
