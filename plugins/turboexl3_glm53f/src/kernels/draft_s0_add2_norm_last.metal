// TURBOEXL3_GLM53F_DRAFT_S0, the MTP draft's step 0 last residual at its last row (src/draft_s0.zig `add2NormLast`):
// smallop_add2_norm.metal's statements at row R0 of H, A and B ([1, R0 + 1, D]): the MoE's routed sum A and shared expert
// B added in T as `mlx_add` rounds them, then the residual H in T, then RMSNorm under W (the kit's mirror of MLX's
// rms_norm single-row kernel), written to HN [1, 1, D]; one threadgroup of TN threads. The stock step adds over every
// row, takes the last and norms it: that row's bytes.
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
threadgroup float local_sums[32];
float hv[NC * 4];
float acc = 0.0f;
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (c < D) {
        const size_t at = size_t(R0) * size_t(D) + c;
        const float delta = float(T(float(A[at]) + float(B[at])));
        const T hn = T(float(H[at]) + delta);
        hv[p * 4 + i] = float(hn);
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
    if (c < D) HN[c] = T(float(W[c]) * float(T(hv[p * 4 + i] * inv)));
}
