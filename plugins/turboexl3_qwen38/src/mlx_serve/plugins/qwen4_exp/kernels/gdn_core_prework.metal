// SPDX-License-Identifier: Apache-2.0
// The GDN core prework: kernels/gdn/gdn_prework.metal (`mlxserve_gdn_prework2`) at the widths its prefill arm takes
// (17 <= S, B <= 2, a per-head gate) as a row stream. One simdgroup walks a run of R rows of one head slot: lane L keeps
// channels 4L .. 4L+3 (the packed kernel's lane map) and their 16 taps in registers, reads each qkv row once as one
// 8-byte load (four rows ahead of use), keeps the last three rows as the conv window (the run's first three come from the
// rows before it, or from the incoming conv state), and stores 8 bytes a row.
// Every output element is the packed kernel's statement on the same operands in the same order: `acc += float(x) *
// float(w)` over taps 0..3 (oldest row first) from 0.0f, T(acc), MLX's sigmoid from the same table (the packed kernel's
// TAB arm, which it takes from 17 rows: every width this kernel serves), the T product; q and k: the lane's sum of
// squares over its 4 channels in order, simd_sum over the same 32 lanes, the precise rsqrt, `scale * (T(1) * T(x * inv))`;
// beta and g: the packed kernel's lane-0 statements, one row per lane; the next conv state: the last three raw rows.
// Template ints: HK, HV, DK, DV, NKEEP (3), C, QSTRIDE, QOFF, BSTRIDE, BOFF, ASTRIDE, AOFF, GATE (the packed kernel's), R (rows
// per run, a multiple of 4), NSG (head slots per threadgroup). Grid: (32 NSG, B * ceil(S / R), ceil((2 HK + HV) / NSG)).
static_assert(NKEEP == 3, "GDN core prework: a 4-tap conv");
static_assert(R % 4 == 0, "GDN core prework: runs of whole 4-row groups");
const uint lane = thread_index_in_simdgroup;
constexpr uint NSLOT = 2 * uint(HK) + uint(HV);
const uint logical_head = threadgroup_position_in_grid.z * uint(NSG) + simdgroup_index_in_threadgroup;
if (logical_head >= NSLOT) return;
const uint S = uint(seq);
const uint nrun = (S + uint(R) - 1) / uint(R);
const uint b = threadgroup_position_in_grid.y / nrun;
const uint r0 = (threadgroup_position_in_grid.y - b * nrun) * uint(R);
const uint r1 = min(r0 + uint(R), S);
constexpr uint q_heads = uint(HK);
constexpr uint k_head_base = uint(HK);
constexpr uint v_head_base = 2 * uint(HK);
const bool is_q = logical_head < q_heads;
const bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
const bool qk = is_q || is_k;
const uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
const uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
const uint ch = channel_base + lane * 4;
// the lane's taps: wf[4 i + tap] = conv_w[(ch + i) * 4 + tap]
float wf[16];
for (uint e = 0; e < 16; ++e) wf[e] = float(conv_w[ch * 4 + e]);
// this batch row's channels ch .. ch + 3: qkv row ip at qx + ip * QSTRIDE, the incoming conv state's row j at cx + j * C
const device T* qx = qkv + (size_t)b * S * uint(QSTRIDE) + uint(QOFF) + ch;
const device T* cx = conv_state + (size_t)b * uint(NKEEP) * uint(C) + ch;
// the conv window: rows r0 - 3, r0 - 2, r0 - 1 (a row before 0 is the incoming conv state's row + 3, as the packed kernel's
// input_row < NKEEP reads)
vec<T, 4> x0, x1, x2;
{
    vec<T, 4> w3[3];
    for (int j = 0; j < 3; ++j) {
        const int ip = int(r0) + j - 3;
        w3[j] = ip < 0 ? *(const device vec<T, 4>*)(cx + (size_t)(ip + NKEEP) * uint(C))
                       : *(const device vec<T, 4>*)(qx + (size_t)ip * uint(QSTRIDE));
    }
    x0 = w3[0];
    x1 = w3[1];
    x2 = w3[2];
}
const T scale = is_q ? q_scale : k_scale;
device T* qk_out = is_q ? q_out : k_out;
for (uint rb = r0; rb < r1; rb += 4) {
    // four rows read ahead of their use
    vec<T, 4> xr[4];
    for (uint u = 0; u < 4; ++u) {
        if (rb + u < r1) xr[u] = *(const device vec<T, 4>*)(qx + (size_t)(rb + u) * uint(QSTRIDE));
    }
    for (uint u = 0; u < 4; ++u) {
        const uint r = rb + u;
        if (r >= r1) break;
        const vec<T, 4> x3 = xr[u];
        const uint row = b * S + r;
        T activated[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float acc = 0.0f;
            acc += float(x0[i]) * wf[4 * i + 0];
            acc += float(x1[i]) * wf[4 * i + 1];
            acc += float(x2[i]) * wf[4 * i + 2];
            acc += float(x3[i]) * wf[4 * i + 3];
            const T conv = T(acc);
            // MLX's unary Sigmoid, the packed kernel's table (its TAB arm from 17 rows)
            const T sig = sigtab[as_type<ushort>(conv)];
            const T act = conv * sig;
            activated[i] = act;
            float value = float(act);
            sumsq += value * value;
        }
        if (qk) {
            sumsq = simd_sum(sumsq);
            float inv = metal::precise::rsqrt(sumsq / float(DK) + qk_eps);
            vec<T, 4> o;
            for (uint i = 0; i < 4; ++i) {
                // ones-weight rms_norm rounding (T(x*inv)), then the separate scalar multiply's rounding
                const T rms = T(1) * T(float(activated[i]) * inv);
                const T value = scale * rms;
                o[i] = value;
            }
            *(device vec<T, 4>*)(qk_out + ((size_t)row * uint(HK) + head) * uint(DK) + lane * 4) = o;
        } else {
            *(device vec<T, 4>*)(v_out + ((size_t)row * uint(HV) + head) * uint(DV) + lane * 4) =
                vec<T, 4>(activated[0], activated[1], activated[2], activated[3]);
        }
        // next conv state = rows [S, S + NKEEP) of concat(conv_state, qkv): this row's raw channels
        if (r + uint(NKEEP) >= S) {
            *(device vec<T, 4>*)(conv_out + ((size_t)b * uint(NKEEP) + (r + uint(NKEEP) - S)) * uint(C) + ch) = x3;
        }
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
}
// beta = sigmoid(b) (MLX unary formula); g = exp(-exp(A_log) * softplus(a + dt_bias)) with the compiled chain's own casts:
// the packed kernel's lane-0 statements, lane L taking the run's rows r0 + L, r0 + L + 32, ...
if (!qk) {
    for (uint r = r0 + lane; r < r1; r += 32) {
        const uint row = b * S + r;
        const T bv = b_in[row * uint(BSTRIDE) + uint(BOFF) + head];
        const T bsig = sigtab[as_type<ushort>(bv)];
        beta_out[row * uint(HV) + head] = bsig;
        if constexpr (GATE) {
            const GT apd = GT(float(a_in[row * uint(ASTRIDE) + uint(AOFF) + head]) + float(dt_bias[head]));
            float sp = msv_log1p(metal::precise::exp(float(apd)));
            float ea = metal::precise::exp(float(A_log[head]));
            g_out[row * uint(HV) + head] = GT(metal::precise::exp(-(ea * sp)));
        }
    }
}
