// TURBOEXL3_GLM53F_SMALLOP_FUSE, a DSA layer's two RMSNorms (src/smallop.zig `normPair`): q_a's rows X0 [R, D0] under W0
// into Y0 and kv_a's rows X1 [R, D1] under W1 into Y1, in one launch. Grid: 2 threadgroups of TN0 threads a row (x 0
// the q_a row, x 1 the kv_a row; y the row). Each threadgroup is the kit's add_norm.fmt.metal mirror of MLX's rms_norm
// single-row kernel at its own width and thread count (TNm = ceil(Dm / 4)): four consecutive elements a thread, the
// per-simdgroup sums folded through a 32-slot table, precise rsqrt, `w * T(x * inv)` in T. The narrower member's
// threads past its own TN1 hold nothing: their simdgroups' slots fold 0.0, the value MLX's zeroed slots hold, and their
// passes past its own NC add 0.0 to an accumulator that is never -0.0.
const uint t = thread_position_in_threadgroup.x;
const uint m = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC0 = (D0 + TN0 * 4 - 1) / (TN0 * 4);
constexpr int NC1 = (D1 + TN1 * 4 - 1) / (TN1 * 4);
constexpr int NCX = NC0 > NC1 ? NC0 : NC1;
const int dm = (m == 0u) ? D0 : D1;
const int tnm = (m == 0u) ? TN0 : TN1;
const int ncm = (m == 0u) ? NC0 : NC1;
threadgroup float local_sums[32];
float hv[NCX * 4];
float acc = 0.0f;
for (int p = 0; p < NCX; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * tnm * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (p < ncm && int(t) < tnm && c < dm) {
        hv[p * 4 + i] = (m == 0u) ? float(X0[size_t(row) * D0 + c]) : float(X1[size_t(row) * D1 + c]);
    }
    acc += hv[p * 4 + i] * hv[p * 4 + i];
}
acc = simd_sum(acc);
if (sg == 0) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) local_sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = simd_sum(local_sums[lane]);
const float inv = metal::precise::rsqrt(total / float(dm) + eps);
for (int p = 0; p < NCX; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * tnm * 4 + int(t) * 4 + i;
    if (p < ncm && int(t) < tnm && c < dm) {
        if (m == 0u) Y0[size_t(row) * D0 + c] = T(float(W0[c]) * float(T(hv[p * 4 + i] * inv)));
        else Y1[size_t(row) * D1 + c] = T(float(W1[c]) * float(T(hv[p * 4 + i] * inv)));
    }
}
