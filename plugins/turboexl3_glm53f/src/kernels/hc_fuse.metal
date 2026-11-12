// TURBOEXL3_GLM53F_HC_FUSE (src/hc_fuse.zig): a prompt sublayer's hyper-connection expand and the next sublayer's
// collapse in one dispatch over each 8 rows, each phase the statements of the copy's own text in its order and types
// (src/mlx_serve/plugins/glm5_next/kernels):
//   1. expand.metal: the new stream, post * x + comb^T @ stream in f32 multiply-adds, rounded to T, written once;
//   2. mixes.metal: the next sublayer's mixes of the new stream's rows, rms(row) scaling the dot products with fn on
//      simdgroup matrices, SG simdgroups splitting K = HC * D, this threadgroup's 8 rows as its 8-row block, the
//      cross-simdgroup sums in its order (kept in threadgroup memory, not written out);
//   3. collapse.metal: gates_header.metal's msv_hc_gates of each row on its own simdgroup (simdgroup r: row r of the
//      block), then the pre-weighted sum of the row's 4 new streams, rounded to T.
// The mixes and the collapse read the new stream back from this threadgroup's own rows, behind a device barrier.
// Grid (ceil(rows / 8) * 32 * SG, 1, 1), threadgroup (32 * SG, 1, 1), SG = 8: the mixes text's launch.
const uint tg = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
constexpr int K = HC * D;
constexpr int MIX = (2 + HC) * HC;
constexpr int KS = K / SG;
constexpr uint D4 = (uint)D / 4;
constexpr float EPS = EPS_INT * 1e-9;
const int rows = params;

// 1. expand.metal, a thread a (row, 4 columns) as its grid point
for (uint w = tid; w < 8 * D4; w += 32 * SG) {
    const uint row = tg * 8 + w / D4;
    const uint d4 = w % D4;
    if (row >= uint(rows)) continue;
    const auto pr = post + row * HC;
    const auto cb = comb + row * HC * HC;
    const float4 xv = float4(*(const device vec<T, 4>*)(x + (size_t)row * D + 4 * d4));
    float4 sv[HC];
    for (int j = 0; j < HC; ++j)
        sv[j] = float4(*(const device vec<T, 4>*)(res + ((size_t)row * HC + j) * D + 4 * d4));
    for (int i = 0; i < HC; ++i) {
        float4 acc = 0.0f;
        for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
        *(device vec<T, 4>*)(out + ((size_t)row * HC + i) * D + 4 * d4) = vec<T, 4>(fma(float4(pr[i]), xv, acc));
    }
}
threadgroup_barrier(mem_flags::mem_device);

// 2. mixes.metal over the new stream
threadgroup float part[SG][8][MIX];
threadgroup float ssp[SG][32];
threadgroup float rowss[8];
threadgroup float mixt[8][MIX];
threadgroup float pre_shared[8][HC];
{
    const short qid = short(lane / 4);
    const short mr = (qid & 4) + short((lane / 2) % 4);
    const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
    const int r = int(tg) * 8 + mr;
    const bool rv = r < rows;
    const device T* xrow = (const device T*)out + (size_t)(rv ? r : 0) * K;
    simdgroup_matrix<float, 8, 8> a, b;
    simdgroup_matrix<float, 8, 8> c[MIX / 8];
    for (int t = 0; t < MIX / 8; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float ss = 0.0f;
    for (int k0 = int(sg) * KS; k0 < int(sg + 1) * KS; k0 += 8) {
        const float2 av = rv ? float2(float(xrow[k0 + mc]), float(xrow[k0 + mc + 1])) : float2(0.0f);
        ss = fma(av.x, av.x, fma(av.y, av.y, ss));
        reinterpret_cast<thread float2&>(a.thread_elements()) = av;
        const device float* wcol = fn + (size_t)mc * K + k0 + mr;
        for (int t = 0; t < MIX / 8; ++t) {
            reinterpret_cast<thread float2&>(b.thread_elements()) = float2(wcol[t * 8 * K], wcol[(t * 8 + 1) * K]);
            simdgroup_multiply_accumulate(c[t], a, b, c[t]);
        }
    }
    for (int t = 0; t < MIX / 8; ++t) {
        const float2 cv = reinterpret_cast<thread float2&>(c[t].thread_elements());
        part[sg][mr][t * 8 + mc] = cv.x;
        part[sg][mr][t * 8 + mc + 1] = cv.y;
    }
    ssp[sg][lane] = ss;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8) {
    float acc = 0.0f;
    for (uint g = 0; g < SG; ++g)
        for (uint l = 0; l < 32; ++l) {
            const short q = short(l / 4);
            if ((q & 4) + short((l / 2) % 4) == short(tid)) acc += ssp[g][l];
        }
    rowss[tid] = acc;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8 * MIX) {
    const int row = int(tid) / MIX;
    const int col = int(tid) % MIX;
    const int gr = int(tg) * 8 + row;
    if (gr < rows) {
        float acc = 0.0f;
        for (uint g = 0; g < SG; ++g) acc += part[g][row][col];
        mixt[row][col] = acc * metal::precise::rsqrt(rowss[row] / float(K) + eps[0]);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// 3. collapse.metal: the gates of row sg of the block on simdgroup sg, then the pre-weighted sum of the 4 streams
if (int(tg) * 8 + int(sg) < rows) {
    const uint row = tg * 8 + sg;
    msv_hc_gates<HC, ITERS>(&mixt[sg][0], base, scale[0], scale[1], scale[2], EPS, lane,
                           &pre_shared[sg][0], post_out + row * HC, comb_out + row * HC * HC);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint w = tid; w < 8 * D4; w += 32 * SG) {
    const uint rb = w / D4;
    const uint row = tg * 8 + rb;
    const uint d4 = w % D4;
    if (row >= uint(rows)) continue;
    const float p0 = pre_shared[rb][0];
    const float p1 = pre_shared[rb][1];
    const float p2 = pre_shared[rb][2];
    const float p3 = pre_shared[rb][3];
    const device T* x_row  = (const device T*)out + row * (HC * D);
    device T*       out_row = (device T*)collapsed + row * D;
    using T4 = vec<T, 4>;
    const device T4* x_row0 = (const device T4*)(x_row + 0*D);
    const device T4* x_row1 = (const device T4*)(x_row + 1*D);
    const device T4* x_row2 = (const device T4*)(x_row + 2*D);
    const device T4* x_row3 = (const device T4*)(x_row + 3*D);
    device T4*       out4   = (device T4*)out_row;
    float4 x0 = float4(x_row0[d4]);
    float4 x1 = float4(x_row1[d4]);
    float4 x2 = float4(x_row2[d4]);
    float4 x3 = float4(x_row3[d4]);
    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    out4[d4] = T4(result);
}
