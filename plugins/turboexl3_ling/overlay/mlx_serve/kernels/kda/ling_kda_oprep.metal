// turboexl3_ling's own: the header of the KDA step's epilogue (kernels/kda/ling_kda_epi.metal, src/ling_kda_epi.zig):
// o_proj's DEC-ONE prepare of one 128-block of a row of y, turboexl3's dense
// text_prep_s statements at LANE_ORDER 1 (prep_sc's, the launch DEC-ONE puts before its GEMV): the f16 round, o_proj's
// suh, the 128-point Hadamard, the scale and the f16 round into lane order, the block's sum. Only the row's place is the
// caller's (`row`, `b`, `L` in place of prep_sc's grid), so the prepared rows and block sums are the words prep_sc writes.
template <typename TX, uint DIMC>
inline void ling_kda_oprep(const device TX* x, const device half* suh, device half* xp, device float* xs, const uint row, const uint b, const uint L) {
    constexpr uint NB = DIMC / 128u;
    constexpr uint LANE_ORDER = 1u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    float sum = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u;
        const uint q = (r & 7u) >> 1u;
        const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const half hv = half(v[j] * HAD_SCALE);
        xp[row * DIMC + ((LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i)] = hv;
        sum += float(hv);
    }
    sum = simd_sum(sum);
    if (L == 0u) xs[row * NB + b] = sum;
}
