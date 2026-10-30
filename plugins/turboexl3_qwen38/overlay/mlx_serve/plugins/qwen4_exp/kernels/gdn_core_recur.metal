// SPDX-License-Identifier: Apache-2.0
// The GDN core recurrence: kernels/gdn/gdn_pipelined.metal (the oMLX port, `_pipelined_source(12)`) with its geometry as
// template ints, TB, DB and KQF, plus RPT, the value rows a lane carries (served: TB 16, DB 32, KQF 0, RPT 1).
// Every floating-point operation of a value row is the original's, in the original's order:
//   - a value row is 8 lanes x 16 channels; lane `seg` owns channels 4*(seg + 8*i) .. +3, i = 0..3;
//   - per token: st *= g; a2 = the fma chain over (xy, zw) for i = 0..3; p = a2.x + a2.y; the xor 4/2/1 add tree;
//     delta = (v - p) * beta; st = fma(k, delta, st); o2 = the fma chain; y partial = o2.x + o2.y;
//   - y's 8 partials go through the same pairing tree ((p_s + p_s^4) + (p_s^2 + p_s^6), then the ^1 pair) in a full
//     block's 4-token transposed reduction and in a ragged block's shuffle_down reduction (fp add is commutative, so every
//     lane's view of that tree gives the same bits; the original relies on it the same way).
// So a row's y and state do not depend on where blocks, pieces or threadgroups fall; each template int moves operands or
// groups the work:
//   TB   tokens per staged block, a multiple of 4 (the original: 12)
//   DB   value rows per threadgroup (8, 16, 32 or 64; the original: 16)
//   KQF  0: k and q staged in threadgroup memory as InT (the original); 1: k as float; 2: k and q as float
//   RPT  value rows per lane (1: the original's; 2: rows rg and rg + DB/2 on one lane, sharing the token's k, q, g and
//        beta reads; each row keeps its own state, dots, trees and y reduction)
    constexpr int NT = 8 * DB / RPT;   // threads: 8 lanes per row group
    constexpr int DBR = DB / RPT;      // row groups; row group rg's lanes own rows rg + DBR * r, r < RPT
    const int tid = thread_position_in_threadgroup.x;
    const int hv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int dv0 = threadgroup_position_in_grid.x * DB;
    const int rg = tid / 8;   // row group within the threadgroup
    const int seg = tid % 8;  // lane within the row
    typedef typename gcore_sel<(KQF >= 1), float, InT>::type KT;
    typedef typename gcore_sel<(KQF >= 2), float, InT>::type QT;

    // k/q stages as vec4 arrays: a float stage needs 16-byte rows for its float4 loads.
    threadgroup vec<KT, 4> k_s[TB][Dk / 4];
    threadgroup vec<QT, 4> q_s[TB][Dk / 4];
    threadgroup float v_s[TB][DB];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];
    threadgroup OutT y_s[TB][DB];

    const size_t krow = (size_t)Hk * Dk;
    const size_t vrow = (size_t)Hv * Dv;
    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    device OutT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    // fp32 state row fragments: row rg + DBR * r, channels 4*(seg + 8*i) .. +3, i = 0..3
    float4 st[RPT][4];
    for (int r = 0; r < RPT; ++r) {
        const device vec<StT, 4>* S_in = (const device vec<StT, 4>*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + rg + DBR * r) * Dk);
        for (int i = 0; i < 4; ++i) st[r][i] = float4(S_in[seg + 8 * i]);
    }

    // register prefetch of one block (k/q rows, v slice, g, beta)
    constexpr int KQ4 = TB * Dk / 4;
    constexpr int NKQ = (KQ4 + NT - 1) / NT;
    constexpr int V4 = TB * DB / 4;
    constexpr int NV = (V4 + NT - 1) / NT;
    vec<InT, 4> pk[NKQ];
    vec<InT, 4> pq[NKQ];
    vec<InT, 4> pv[NV];
    float pg = 0.0f, pb = 0.0f;
#define GCORE_FETCH(T0N) { \
        const int ttn = min(TB, T - (T0N)); \
        for (int j = 0; j < NKQ; ++j) { \
            const int e = tid + j * NT; \
            if (e < ttn * (Dk / 4)) { \
                const int er = e / (Dk / 4), c4 = e % (Dk / 4); \
                const size_t off = (size_t)((T0N) + er) * krow + 4 * c4; \
                pk[j] = *(const device vec<InT, 4>*)(k_base + off); \
                pq[j] = *(const device vec<InT, 4>*)(q_base + off); \
            } \
        } \
        for (int j = 0; j < NV; ++j) { \
            const int e = tid + j * NT; \
            if (e < ttn * (DB / 4)) { \
                const int er = e / (DB / 4), c4 = e % (DB / 4); \
                pv[j] = *(const device vec<InT, 4>*)( \
                    v_base + (size_t)((T0N) + er) * vrow + 4 * c4); \
            } \
        } \
        if (tid < ttn) { \
            pg = (float)g[((size_t)b * T + (T0N) + tid) * Hv + hv]; \
            pb = (float)beta[((size_t)b * T + (T0N) + tid) * Hv + hv]; \
        } \
    }
    // The transposed y reduction of row r's 4 tokens R0 .. R0+3 (the original's, verbatim, on that row's partials).
#define GCORE_YRED(R0, r) { \
        const bool qb2 = (seg & 4) != 0, qb1 = (seg & 2) != 0; \
        float qa = qb2 ? pq4[r][2] : pq4[r][0]; \
        float qb = qb2 ? pq4[r][3] : pq4[r][1]; \
        qa += simd_shuffle_xor(qb2 ? pq4[r][0] : pq4[r][2], 4); \
        qb += simd_shuffle_xor(qb2 ? pq4[r][1] : pq4[r][3], 4); \
        float qx = qb1 ? qb : qa; \
        qx += simd_shuffle_xor(qb1 ? qa : qb, 2); \
        qx += simd_shuffle_xor(qx, 1); \
        if ((seg & 1) == 0) y_s[(R0) + (qb2 ? 2 : 0) + (qb1 ? 1 : 0)][rg + DBR * (r)] = static_cast<OutT>(qx); \
    }

    GCORE_FETCH(0)

    float4 k_next[4];
    float g_next = 0.0f, b_next = 0.0f;
    float v_next[RPT];
    float pq_prev[RPT];
    float pq4[RPT][4];
    for (int r = 0; r < RPT; ++r) {
        v_next[r] = 0.0f;
        pq_prev[r] = 0.0f;
        for (int c = 0; c < 4; ++c) pq4[r][c] = 0.0f;
    }
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        for (int j = 0; j < NKQ; ++j) {
            const int e = tid + j * NT;
            if (e < tt * (Dk / 4)) {
                const int er = e / (Dk / 4), c4 = e % (Dk / 4);
                k_s[er][c4] = vec<KT, 4>(pk[j]);
                q_s[er][c4] = vec<QT, 4>(pq[j]);
            }
        }
        for (int j = 0; j < NV; ++j) {
            const int e = tid + j * NT;
            if (e < tt * (DB / 4)) {
                const int er = e / (DB / 4), c4 = e % (DB / 4);
                *(threadgroup float4*)(&v_s[er][4 * c4]) = float4(pv[j]);
            }
        }
        if (tid < tt) {
            g_s[tid] = pg;
            b_s[tid] = pb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + TB < T) GCORE_FETCH(t0 + TB)
        for (int i = 0; i < 4; ++i) k_next[i] = float4(k_s[0][seg + 8 * i]);
        g_next = g_s[0];
        b_next = b_s[0];
        for (int r = 0; r < RPT; ++r) v_next[r] = v_s[0][rg + DBR * r];

        if (tt == TB) {
            // Full block: the original's unrolled steps (`_pipelined_source`), y reduced 4 tokens at a time.
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < TB; ++j) {
                float4 kc[4];
                for (int i = 0; i < 4; ++i) kc[i] = k_next[i];
                const float gc = g_next, bc = b_next;
                float vc[RPT];
                for (int r = 0; r < RPT; ++r) vc[r] = v_next[r];
                if (j + 1 < TB) {
                    for (int i = 0; i < 4; ++i) k_next[i] = float4(k_s[j + 1][seg + 8 * i]);
                    g_next = g_s[j + 1];
                    b_next = b_s[j + 1];
                    for (int r = 0; r < RPT; ++r) v_next[r] = v_s[j + 1][rg + DBR * r];
                }
                float2 a2[RPT];
                for (int r = 0; r < RPT; ++r) {
                    a2[r] = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        st[r][i] = st[r][i] * gc;
                        a2[r] = fma(st[r][i].xy, kc[i].xy, a2[r]);
                        a2[r] = fma(st[r][i].zw, kc[i].zw, a2[r]);
                    }
                }
                if (j >= 4 && (j & 3) == 0) for (int r = 0; r < RPT; ++r) GCORE_YRED(j - 4, r)
                float p[RPT];
                for (int r = 0; r < RPT; ++r) {
                    p[r] = a2[r].x + a2[r].y;
                    p[r] += simd_shuffle_xor(p[r], 4);
                    p[r] += simd_shuffle_xor(p[r], 2);
                    p[r] += simd_shuffle_xor(p[r], 1);
                }
                float4 qc[4];
                for (int i = 0; i < 4; ++i) qc[i] = float4(q_s[j][seg + 8 * i]);
                for (int r = 0; r < RPT; ++r) {
                    const float delta = (vc[r] - p[r]) * bc;
                    float2 o2 = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        st[r][i] = fma(kc[i], float4(delta), st[r][i]);
                        o2 = fma(st[r][i].xy, qc[i].xy, o2);
                        o2 = fma(st[r][i].zw, qc[i].zw, o2);
                    }
                    pq4[r][j & 3] = o2.x + o2.y;
                }
            }
            for (int r = 0; r < RPT; ++r) GCORE_YRED(TB - 4, r)
        } else {
            // Ragged block: the original's loop, y of token t-1 reduced during token t.
            for (int t = 0; t < tt; ++t) {
                float4 kc[4];
                for (int i = 0; i < 4; ++i) kc[i] = k_next[i];
                const float gc = g_next, bc = b_next;
                float vc[RPT];
                for (int r = 0; r < RPT; ++r) vc[r] = v_next[r];
                if (t + 1 < tt) {
                    for (int i = 0; i < 4; ++i) k_next[i] = float4(k_s[t + 1][seg + 8 * i]);
                    g_next = g_s[t + 1];
                    b_next = b_s[t + 1];
                    for (int r = 0; r < RPT; ++r) v_next[r] = v_s[t + 1][rg + DBR * r];
                }
                float2 a2[RPT];
                for (int r = 0; r < RPT; ++r) {
                    a2[r] = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        st[r][i] = st[r][i] * gc;
                        a2[r] = fma(st[r][i].xy, kc[i].xy, a2[r]);
                        a2[r] = fma(st[r][i].zw, kc[i].zw, a2[r]);
                    }
                }
                if (t > 0) {
                    for (int r = 0; r < RPT; ++r) {
                        float yq = pq_prev[r];
                        yq += simd_shuffle_down(yq, 4);
                        yq += simd_shuffle_down(yq, 2);
                        yq += simd_shuffle_down(yq, 1);
                        if (seg == 0) y_s[t - 1][rg + DBR * r] = static_cast<OutT>(yq);
                    }
                }
                float p[RPT];
                for (int r = 0; r < RPT; ++r) {
                    p[r] = a2[r].x + a2[r].y;
                    p[r] += simd_shuffle_xor(p[r], 4);
                    p[r] += simd_shuffle_xor(p[r], 2);
                    p[r] += simd_shuffle_xor(p[r], 1);
                }
                float4 qc[4];
                for (int i = 0; i < 4; ++i) qc[i] = float4(q_s[t][seg + 8 * i]);
                for (int r = 0; r < RPT; ++r) {
                    const float delta = (vc[r] - p[r]) * bc;
                    float2 o2 = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        st[r][i] = fma(kc[i], float4(delta), st[r][i]);
                        o2 = fma(st[r][i].xy, qc[i].xy, o2);
                        o2 = fma(st[r][i].zw, qc[i].zw, o2);
                    }
                    pq_prev[r] = o2.x + o2.y;
                }
            }
            for (int r = 0; r < RPT; ++r) {
                float yq = pq_prev[r];
                yq += simd_shuffle_down(yq, 4);
                yq += simd_shuffle_down(yq, 2);
                yq += simd_shuffle_down(yq, 1);
                if (seg == 0) y_s[tt - 1][rg + DBR * r] = static_cast<OutT>(yq);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < tt * (DB / 4); e += NT) {
            const int er = e / (DB / 4), c4 = e % (DB / 4);
            *(device vec<OutT, 4>*)(y_base + (size_t)(t0 + er) * vrow + 4 * c4) =
                *(threadgroup vec<OutT, 4>*)(&y_s[er][4 * c4]);
        }
    }
#undef GCORE_FETCH
#undef GCORE_YRED

    for (int r = 0; r < RPT; ++r) {
        device vec<StT, 4>* S_out = (device vec<StT, 4>*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + rg + DBR * r) * Dk);
        for (int i = 0; i < 4; ++i) S_out[seg + 8 * i] = vec<StT, 4>(st[r][i]);
    }
