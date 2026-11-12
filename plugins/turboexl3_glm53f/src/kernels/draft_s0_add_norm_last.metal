// TURBOEXL3_GLM53F_DRAFT_S0 beside TURBOEXL3_GLM53F_HEAD_LAST_ROW, the MTP draft's step 0 last residual over the MoE's
// last row alone (src/draft_s0.zig `addNormLast`, head.zig's `tailLast`): row R0 of the residual H ([1, R0 + 1, D]) plus
// the MoE's last-row output D1 ([1, 1, D]) in T as `mlx_add` rounds it, then RMSNorm under W (the kit's mirror of MLX's
// rms_norm single-row kernel, as smallop_add2_norm.metal runs it), written to HN [1, 1, D]; one threadgroup of TN
// threads. The stock `tailLast` slices that row, adds and norms it: the same bytes.
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
        const T hn = T(float(H[size_t(R0) * size_t(D) + c]) + float(D1[c]));
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
