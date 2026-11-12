// TURBOEXL3_GLM53F_KDA_OPREP (src/dec_prep.zig `kdaOut`): a KDA layer's gated norm in a verify, written as the out
// projection's prepared input, one launch where KDA_VERIFY runs its gated-norm text and turboexl3's DEC-ONE runs
// prep_sc's launch before its GEMV. GLM-5.3-Flash's KDA: 64 value heads of 128, the out projection 8192 -> 4096, so a
// head is one 128-block of the projection's input. Grid (32, rows, 64), one simdgroup a (row, head).
// The gated norm's statements (kda_prompt.normGateText's text at TAB 1, SWISH 0): lane L holds the head's values
// 4L .. 4L + 3, their squares summed in the simdgroup, MLX's precise rsqrt at the chain's epsilon, `norm_w * T(x * inv)`
// in T, the output gate's sigmoid read from MLX's table, `normed * sigmoid` in T. Then four shuffles a value put each
// lane on the prepare's values (lane L: L + 32 j, j 0 .. 3), and turboexl3's prep_s statements run on them: the f16
// round, x suh, the 128-point Hadamard in five lane butterflies and one in-lane stage, the scale, the f16 store in lane
// order, the block's sum of its stored halves; the GEMV's arrival counters zeroed (prep_sc's).
constexpr uint HV = 64u;
constexpr uint DV = 128u;
constexpr uint DIM = HV * DV;
constexpr uint NB = DIM / 128u;
constexpr float HAD_SCALE = 0.08838834764831845f;
const uint lane = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y;
const uint head = threadgroup_position_in_grid.z;
const uint base = (row * HV + head) * DV + lane * 4u;
float yv[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4u; ++i) {
    yv[i] = float(y[base + i]);
    sumsq += yv[i] * yv[i];
}
sumsq = simd_sum(sumsq);
const float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
const uint zbase = row * DIM + head * DV + lane * 4u;
float gv[4];
for (uint i = 0; i < 4u; ++i) {
    const T normed = norm_w[lane * 4u + i] * T(yv[i] * inv);
    const T zv = z[zbase + i];
    const T sig = sigtab[as_type<ushort>(zv)];
    gv[i] = float(normed * sig);
}
const uint k = lane & 3u;
float v[4];
for (uint j = 0; j < 4u; ++j) {
    const ushort src = ushort((lane >> 2u) + 8u * j);
    const float a0 = simd_shuffle(gv[0], src);
    const float a1 = simd_shuffle(gv[1], src);
    const float a2 = simd_shuffle(gv[2], src);
    const float a3 = simd_shuffle(gv[3], src);
    const float x = (k == 0u) ? a0 : ((k == 1u) ? a1 : ((k == 2u) ? a2 : a3));
    v[j] = float(half(x)) * float(suh[head * 128u + lane + 32u * j]);
}
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (lane & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
float sum = 0.0f;
for (uint j = 0; j < 4u; ++j) {
    const uint i = head * 128u + lane + 32u * j;
    const uint r = i & 15u;
    const uint q = (r & 7u) >> 1u;
    const uint m = (r & 1u) | ((r >> 3u) << 1u);
    const half hv = half(v[j] * HAD_SCALE);
    xp[row * DIM + ((i & ~15u) + 4u * q + m)] = hv;
    sum += float(hv);
}
sum = simd_sum(sum);
if (lane == 0u) xs[row * NB + head] = sum;
for (uint c = (row * NB + head) * 32u + lane; c < uint(NCNT); c += uint(NTH)) cnt[c] = 0u;
