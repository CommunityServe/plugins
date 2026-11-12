// TURBOEXL3_GLM53F_DSA_OPREP (src/dec_prep.zig `dsaOut`): a DSA layer's out projection input at verify rows, prepared
// in place from the attention's per-head values, one launch where the stock forward copies the values from head order
// into row order (its transpose and reshape) and turboexl3's DEC-ONE runs prep_sc's launch before its GEMV.
// GLM-5.3-Flash's DSA: 64 heads of 256 values, the out projection 16384 -> 4096, so each 128-block of a row's input
// lies inside one head (block b: head b / 2, values (b % 2) 128 ..). The values O [1, 64, ROWS, 256] in head order are
// read where they lie: row r's input value h 256 + d is O's (h ROWS + r) 256 + d. Grid (128 x 32, ROWS), one
// simdgroup a (block, row), lane L on the block's values L + 32 j (j 0 .. 3), turboexl3's prep_s statements: the f16
// round, x suh, the 128-point Hadamard in five lane butterflies and one in-lane stage, the scale, the f16 store in lane
// order, the block's sum of its stored halves; the GEMV's arrival counters zeroed (prep_sc's).
constexpr uint H = 64u;
constexpr uint HD = 256u;
constexpr uint DIM = H * HD;
constexpr uint NB = DIM / 128u;
constexpr float HAD_SCALE = 0.08838834764831845f;
const uint b = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint L = thread_index_in_simdgroup;
const uint head = b >> 1u;
const uint first = (head * uint(ROWS) + row) * HD + (b & 1u) * 128u;
float v[4];
for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(o[first + L + 32u * j]))) * float(suh[b * 128u + L + 32u * j]);
for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
{ float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
float sum = 0.0f;
for (uint j = 0; j < 4u; ++j) {
    const uint i = b * 128u + L + 32u * j;
    const uint r = i & 15u;
    const uint q = (r & 7u) >> 1u;
    const uint m = (r & 1u) | ((r >> 3u) << 1u);
    const half hv = half(v[j] * HAD_SCALE);
    xp[row * DIM + ((i & ~15u) + 4u * q + m)] = hv;
    sum += float(hv);
}
sum = simd_sum(sum);
if (L == 0u) xs[row * NB + b] = sum;
for (uint c = (row * NB + b) * 32u + L; c < uint(NCNT); c += uint(NTH)) cnt[c] = 0u;
