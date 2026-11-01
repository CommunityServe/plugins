// One thread a row's 4 columns: the joinless combine (dsv41_jl_combine_bf16's words: each routed row read where its wave
// wrote it, times its weight, folded in MLX's column reduce order, the shared row added, all f32, uncontracted), rounded
// once to T as the model's astype does, then glm5's hc expand of the row's HC streams (out[k] = post[k] * y + sum_j
// comb[j][k] * res[j], f32 fmas in its order, rounded once to T). Inputs: s0..s23 the routed outputs (S), loc int32
// [n, K, 2] (each assignment's source and row), w f32 [n, K], sh [n, D] (ST), res [n, HC, D] (T), pc f32 [n, HC + HC*HC]
// (each row's post, then comb [HC][HC]); out [n, HC, D] (T).
constexpr uint C = 4;
constexpr uint DC = uint(D) / C;
const uint g = thread_position_in_grid.x;
const uint row = g / DC;
if (row >= uint(NROWS)) return;
const uint col = (g - row * DC) * C;
const ulong lbase = ulong(row) * ulong(K);
float t[C];
{
    const float wv = static_cast<float>(w[lbase]);
    const device S* p0 = tf_pick<S>(loc[2 * lbase], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
    const ulong o0 = ulong(loc[2 * lbase + 1]) * ulong(D) + ulong(col);
    for (uint i = 0; i < C; ++i) {
        const float p = static_cast<float>(p0[o0 + i]) * wv;
        t[i] = p + 0.0f;
    }
}
for (uint k = 1; k < uint(K); ++k) {
    const float wv = static_cast<float>(w[lbase + k]);
    const device S* pk = tf_pick<S>(loc[2 * (lbase + k)], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
    const ulong ok = ulong(loc[2 * (lbase + k) + 1]) * ulong(D) + ulong(col);
    for (uint i = 0; i < C; ++i) {
        const float p = static_cast<float>(pk[ok + i]) * wv;
        t[i] = (p + 0.0f) + t[i];
    }
}
const ulong obase = ulong(row) * ulong(D) + ulong(col);
float y[C];
for (uint i = 0; i < C; ++i) {
    y[i] = static_cast<float>(static_cast<T>(t[i] + static_cast<float>(sh[obase + i])));
}
const device float* pr = pc + ulong(row) * ulong(HC + HC * HC);
const device float* cb = pr + HC;
float sv[HC][C];
for (int j = 0; j < HC; ++j) {
    for (uint i = 0; i < C; ++i) {
        sv[j][i] = static_cast<float>(res[(ulong(row) * ulong(HC) + ulong(j)) * ulong(D) + ulong(col) + i]);
    }
}
for (int k2 = 0; k2 < HC; ++k2) {
    for (uint i = 0; i < C; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < HC; ++j) acc = metal::fma(cb[j * HC + k2], sv[j][i], acc);
        out[(ulong(row) * ulong(HC) + ulong(k2)) * ulong(D) + ulong(col) + i] = static_cast<T>(metal::fma(pr[k2], y[i], acc));
    }
}
