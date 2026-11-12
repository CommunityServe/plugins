// TURBOEXL3_GLM53F_SMALLOP_FUSE, the draft step's front (src/smallop.zig `front`): one decode row's embedding E[ids[0]]
// under the enorm weight WE and the trunk's hidden H under the hnorm weight WH, each through its own RMSNorm, written
// as their concatenation CAT [1, 1, 2D] (the embedding's half first). Grid: 2 threadgroups of TN threads (0 the
// embedding row, 1 the hidden row). Each threadgroup is the kit's add_norm.fmt.metal mirror of MLX's rms_norm
// single-row kernel at its row: four consecutive elements a thread, the per-simdgroup sums folded through a 32-slot
// table, precise rsqrt, `w * T(x * inv)` in T.
const uint t = thread_position_in_threadgroup.x;
const uint r = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
threadgroup float local_sums[32];
float hv[NC * 4];
float acc = 0.0f;
const size_t row_at = (r == 0u) ? size_t(ids[0]) * size_t(D) : size_t(0);
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (c < D) {
        hv[p * 4 + i] = (r == 0u) ? float(E[row_at + c]) : float(H[c]);
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
    if (c < D) CAT[size_t(r) * D + c] = T(float((r == 0u) ? WE[c] : WH[c]) * float(T(hv[p * 4 + i] * inv)));
}
